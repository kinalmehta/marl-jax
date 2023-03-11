"""
This is a fork of
  https://github.com/deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py
with some modifications to work with MARL setup.
"""
"""Program definition for a distributed layout of MARL agent based on a builder."""

import itertools
from typing import Callable, Optional

import dm_env
import jax
import launchpad as lp
import reverb
from acme import core, environment_loop
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers, snapshotter, utils
from acme.jax.experiments import config
from acme.tf import savers as tf_savers
from acme.utils import counting, lp_utils

from marl import specs as ma_specs
from marl.experiments import config as ma_config

ActorId = int


def make_distributed_experiment(
    experiment: ma_config.MAExperimentConfig,
    num_actors: int,
    gpu_actors: list = [],
    *,
    num_learner_nodes: int = 1,
    num_actors_per_node: int = 1,
    multithreading_colocate_learner_and_reverb: bool = False,
    checkpointing_config: Optional[ma_config.CheckpointingConfig] = None,
    make_snapshot_models: Optional[config.SnapshotModelFactory[
        builders.Networks]] = None,
    name="agent",
    program: Optional[lp.Program] = None,
):
  """Builds distributed agent based on a builder."""

  if multithreading_colocate_learner_and_reverb and num_learner_nodes > 1:
    raise ValueError(
        "Replay and learner colocation is not yet supported when the learner is"
        " spread across multiple nodes (num_learner_nodes > 1). Please contact"
        " Acme devs if this is a feature you want. Got:"
        "\tmultithreading_colocate_learner_and_reverb="
        f"{multithreading_colocate_learner_and_reverb}"
        f"\tnum_learner_nodes={num_learner_nodes}.")

  if checkpointing_config is None:
    checkpointing_config = ma_config.CheckpointingConfig()

  def build_replay():
    """The replay storage."""
    dummy_seed = 1
    spec = experiment.environment_spec or ma_specs.MAEnvironmentSpec(
        experiment.environment_factory(dummy_seed))
    network = experiment.network_factory(
        spec.get_single_agent_environment_specs())
    policy = experiment.builder.make_policy(network, spec, evaluation=False)
    return experiment.builder.make_replay_tables(spec, policy)

  def build_model_saver(variable_source: core.VariableSource):
    environment = experiment.environment_factory(0)
    specs = ma_specs.MAEnvironmentSpec(environment)
    networks = experiment.network_factory(
        specs.get_single_agent_environment_specs())

    models = make_snapshot_models(networks, specs)
    # TODO(raveman): Decouple checkpointing and snapshotting configs.
    return snapshotter.JAXSnapshotter(
        variable_source=variable_source,
        models=models,
        path=checkpointing_config.directory,
        subdirectory="snapshots",
        add_uid=checkpointing_config.add_uid,
    )

  def build_counter():
    counter = counting.Counter()
    if experiment.resume_training:
      checkpointer = tf_savers.Checkpointer(
          objects_to_save={"counter": counter},
          directory=checkpointing_config.directory,
          subdirectory="counter",
          time_delta_minutes=checkpointing_config.model_time_delta_minutes,
          add_uid=checkpointing_config.add_uid,
          max_to_keep=checkpointing_config.max_to_keep,
      )
      checkpointer.restore()
    return savers.CheckpointingRunner(
        counter,
        key="counter",
        subdirectory="counter",
        time_delta_minutes=5,
        directory=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid,
        max_to_keep=checkpointing_config.max_to_keep,
    )

  def build_learner(
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      counter: Optional[counting.Counter] = None,
      primary_learner: Optional[core.Learner] = None,
  ):
    """The Learning part of the agent."""

    dummy_seed = 1
    spec = experiment.environment_spec or ma_specs.MAEnvironmentSpec(
        experiment.environment_factory(dummy_seed))

    # Creates the networks to optimize (online) and target networks.
    networks = experiment.network_factory(
        spec.get_single_agent_environment_specs())

    iterator = experiment.builder.make_dataset_iterator(replay)
    # make_dataset_iterator is responsible for putting data onto appropriate
    # training devices, so here we apply prefetch, so that data is copied over
    # in the background.
    iterator = utils.prefetch(iterable=iterator, buffer_size=1)
    counter = counting.Counter(counter, "learner")
    learner = experiment.builder.make_learner(
        random_key,
        networks,
        iterator,
        experiment.logger_factory,
        spec,
        replay,
        counter,
    )

    if primary_learner is None:
      if experiment.resume_training:
        checkpointer = tf_savers.Checkpointer(
            objects_to_save={"learner": learner},
            directory=checkpointing_config.directory,
            subdirectory="learner",
            time_delta_minutes=checkpointing_config.model_time_delta_minutes,
            add_uid=checkpointing_config.add_uid,
            max_to_keep=checkpointing_config.max_to_keep,
        )
        checkpointer.restore()
      learner = savers.CheckpointingRunner(
          learner,
          key="learner",
          subdirectory="learner",
          time_delta_minutes=checkpointing_config.model_time_delta_minutes,
          directory=checkpointing_config.directory,
          add_uid=checkpointing_config.add_uid,
          max_to_keep=checkpointing_config.max_to_keep,
      )
    else:
      learner.restore(primary_learner.save())
      # NOTE: This initially synchronizes secondary learner states with the
      # primary one. Further synchronization should be handled by the learner
      # properly doing a pmap/pmean on the loss/gradients, respectively.

    return learner

  def build_actor(
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      actor_id: ActorId,
  ) -> environment_loop.EnvironmentLoop:
    """The actor process."""
    environment_key, actor_key = jax.random.split(random_key)
    # Create environment and policy core.

    # Environments normally require uint32 as a seed.
    environment = experiment.environment_factory(
        utils.sample_uint32(environment_key))
    environment_spec = ma_specs.MAEnvironmentSpec(environment)

    networks = experiment.network_factory(
        environment_spec.get_single_agent_environment_specs())
    policy_network = experiment.builder.make_policy(
        networks, environment_spec, evaluation=False)
    adder = experiment.builder.make_adder(replay, environment_spec,
                                          policy_network)
    actor = experiment.builder.make_actor(actor_key, policy_network,
                                          environment_spec, variable_source,
                                          adder)

    # Create logger and counter.
    counter = counting.Counter(counter, "actor")
    logger = experiment.logger_factory("actor", counter.get_steps_key(),
                                       actor_id)
    # Create the loop to connect environment and agent.
    return environment_loop.EnvironmentLoop(
        environment, actor, counter, logger, observers=experiment.observers)

  def build_evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: core.VariableSource,
      eval_env_factory: Callable[[int], dm_env.Environment],
      evaluator_id: ActorId,
  ) -> environment_loop.EnvironmentLoop:
    """The actor process."""
    environment_key, actor_key = jax.random.split(random_key)
    # Create environment and policy core.

    # Environments normally require uint32 as a seed.
    environment = experiment.environment_factory(
        utils.sample_uint32(environment_key))
    environment_spec = ma_specs.MAEnvironmentSpec(environment)
    eval_env = eval_env_factory(utils.sample_uint32(environment_key))

    networks = experiment.network_factory(
        environment_spec.get_single_agent_environment_specs())
    policy_network = experiment.builder.make_policy(
        networks, environment_spec, evaluation=False)
    evaluator = experiment.builder.make_evaluator(actor_key, policy_network,
                                                  eval_env.num_agents,
                                                  variable_source)

    # Create logger and counter.
    # counter = counting.Counter(counter, f"evaluator_{evaluator_id}")
    logger = experiment.logger_factory(f"evaluator_{evaluator_id}")
    # Create the loop to connect environment and agent.
    return environment_loop.EnvironmentLoop(
        eval_env,
        evaluator,
        logger=logger,
        should_update=False,
        observers=experiment.observers)

  if not program:
    program = lp.Program(name=name)

  key = jax.random.PRNGKey(experiment.seed)

  replay_node = lp.ReverbNode(
      build_replay,
      checkpoint_time_delta_minutes=(
          checkpointing_config.replay_checkpointing_time_delta_minutes),
  )
  replay = replay_node.create_handle()

  counter = program.add_node(lp.CourierNode(build_counter), label="counter")

  if experiment.max_num_actor_steps is not None:
    program.add_node(
        lp.CourierNode(lp_utils.StepsLimiter, counter,
                       experiment.max_num_actor_steps),
        label="counter",
    )

  learner_key, key = jax.random.split(key)
  learner_node = lp.CourierNode(build_learner, learner_key, replay, counter)
  learner = learner_node.create_handle()
  variable_sources = [learner]

  if multithreading_colocate_learner_and_reverb:
    program.add_node(
        lp.MultiThreadingColocation([learner_node, replay_node]),
        label="learner",
    )
  else:
    program.add_node(replay_node, label="replay")

    with program.group("learner"):
      program.add_node(learner_node)

      # Maybe create secondary learners, necessary when using multi-host
      # accelerators.
      # Warning! If you set num_learner_nodes > 1, make sure the learner class
      # does the appropriate pmap/pmean operations on the loss/gradients,
      # respectively.
      for _ in range(1, num_learner_nodes):
        learner_key, key = jax.random.split(key)
        variable_sources.append(
            program.add_node(
                lp.CourierNode(
                    build_learner,
                    learner_key,
                    replay,
                    primary_learner=learner,
                )))
        # NOTE: Secondary learners are used to load-balance get_variables calls,
        # which is why they get added to the list of available variable sources.
        # NOTE: Only the primary learner checkpoints.
        # NOTE: Do not pass the counter to the secondary learners to avoid
        # double counting of learner steps.

  # Create all actor threads.
  offset = len(gpu_actors)
  *actor_keys, key = jax.random.split(key, num_actors + 1)
  variable_sources = itertools.cycle(variable_sources)
  if offset > 0:
    # with program.group("gpu_actor"):
    for aid, (akey,
              vsource) in enumerate(zip(actor_keys[:offset], variable_sources)):
      with program.group(gpu_actors[aid]):
        program.add_node(
            lp.CourierNode(build_actor, akey, replay, vsource, counter,
                           aid + 1))

  with program.group("actor"):
    actor_nodes = [
        lp.CourierNode(build_actor, akey, replay, vsource, counter, aid)
        for aid, (
            akey,
            vsource) in enumerate(zip(actor_keys[offset:], variable_sources))
    ]

    # Create (maybe colocated) actor nodes.
    if num_actors_per_node == 1:
      for actor_node in actor_nodes:
        program.add_node(actor_node)
    else:
      for i in range(0, num_actors, num_actors_per_node):
        program.add_node(
            lp.MultiThreadingColocation(actor_nodes[i:i + num_actors_per_node]))

  if experiment.evaluator_env_factories:
    with program.group("evaluator"):
      eval_nodes = []
      for idx, evaluator_env_factory in enumerate(
          experiment.evaluator_env_factories):
        evaluator_key, key = jax.random.split(key)
        eval_nodes.append(
            lp.CourierNode(
                build_evaluator,
                evaluator_key,
                learner,
                evaluator_env_factory,
                evaluator_id=idx,
            ),)
      program.add_node(
          lp.MultiThreadingColocation(eval_nodes), label="evaluator")

  if make_snapshot_models and checkpointing_config:
    program.add_node(
        lp.CourierNode(build_model_saver, learner), label="model_saver")

  return program
