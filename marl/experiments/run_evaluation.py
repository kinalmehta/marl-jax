"""
This is a fork of
  https://github.com/deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py
with some modifications to work with MARL setup.
"""
"""Runner used for executing local MARL agent."""

import acme
from acme import core
from acme.jax import variable_utils
from acme.tf import savers as tf_savers
from acme.utils import counting
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

from marl import specs as ma_specs
from marl.experiments import config as ma_config
from marl.utils import experiment_utils as ma_utils


def run_evaluation(
    experiment: ma_config.MAExperimentConfig,
    checkpointing_config: ma_config.CheckpointingConfig,
    environment_name: str,
    num_eval_episodes: int = 3,
):
  """Runs a simple, single-threaded evaluation loop using the default evaluators.

    Arguments:
      experiment: Definition and configuration of the agent to run.
      checkpointing_config: Configuration for checkpointing to load checkpoint from.
    """

  key = jax.random.PRNGKey(experiment.seed)

  # Create the environment and get its spec.
  environment = experiment.environment_factory(experiment.seed)
  environment_specs: ma_specs.MAEnvironmentSpec = experiment.environment_spec
  scenario_spec: ma_specs.MAEnvironmentSpec = ma_specs.MAEnvironmentSpec(
      environment)

  # Create the networks and policy.
  network = experiment.network_factory(
      environment_specs.get_single_agent_environment_specs())

  # Parent counter allows to share step counts between train and eval loops and
  # the learner, so that it is possible to plot for example evaluator's return
  # value as a function of the number of training episodes.
  parent_counter = counting.Counter(time_delta=0.0)

  # Create actor, and learner for generating, storing, and consuming
  # data respectively.
  dataset = None  # fakes.transition_dataset_from_spec(environment_specs.get_agent_environment_specs())

  learner_key, key = jax.random.split(key)
  learner = experiment.builder.make_learner(
      random_key=learner_key,
      networks=network,
      dataset=dataset,
      logger_fn=experiment.logger_factory,
      environment_spec=environment_specs,
  )

  s0 = learner._combined_states.params.copy()
  checkpointer = tf_savers.Checkpointer(
      objects_to_save={"learner": learner},
      directory=checkpointing_config.directory,
      subdirectory="learner",
      time_delta_minutes=checkpointing_config.model_time_delta_minutes,
      add_uid=checkpointing_config.add_uid,
      max_to_keep=checkpointing_config.max_to_keep,
  )
  checkpointer.restore()
  s1 = learner._combined_states.params.copy()

  # testing that the learner parameters are actually loaded
  for k, v in s0.items():
    for k_, v_ in v.items():
      assert (s0[k][k_] - s1[k][k_]
             ).sum() != 0, f'New parameters are the same as old {k}.{k_}'
  print(f'Learner parameters successfully updated!')

  variable_client = variable_utils.VariableClient(
      client=learner, key="network", update_period=int(1), device="cpu")

  # Create the evaluation actor and loop.
  eval_counter = counting.Counter(
      parent_counter, prefix=environment_name, time_delta=0.0)
  eval_logger = experiment.logger_factory(
      label=environment_name,
      steps_key=eval_counter.get_steps_key(),
      task_instance=0)

  eval_actor = Evaluate(
      network.forward_fn,
      network.initial_state_fn,
      n_agents=environment.num_agents,
      n_params=environment_specs.num_agents,
      variable_client=variable_client,
      rng=hk.PRNGSequence(experiment.seed),
  )
  eval_loop = acme.EnvironmentLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=eval_logger,
      should_update=False,
  )

  eval_loop.run(num_episodes=num_eval_episodes)


class Evaluate(core.Actor):

  def __init__(
      self,
      forward_fn,
      initial_state_fn,
      n_agents,
      n_params,
      variable_client,
      rng,
  ):
    self.forward_fn = forward_fn
    self.n_agents = n_agents
    self.n_params = n_params
    self._rng = rng
    self._variable_client = variable_client

    def initialize_states(rng_sequence: hk.PRNGSequence,) -> list[hk.LSTMState]:
      """Initialize the recurrent states of the actor."""
      states = list()
      for _ in range(self.n_agents):
        states.append(initial_state_fn(next(rng_sequence)))
      return states

    self._initial_states = ma_utils.merge_data(initialize_states(self._rng))
    self._p_forward = jax.vmap(self.forward_fn)

  def select_action(self, observations):
    if self._states is None:
      self._states = self._initial_states
      self.update(True)
      self.loaded_params = self._params
      self.selected_params = jax.random.choice(
          next(self._rng), self.n_params, (self.n_agents,), replace=True)
      self.episode_params = ma_utils.select_idx(self.loaded_params,
                                                self.selected_params)

    (logits, _), new_states = self._p_forward(self.episode_params, observations,
                                              self._states)
    # probability based action selection
    # actions = jax.random.categorical(next(self._rng), logits)
    # greedy action selection
    actions = jnp.argmax(logits, axis=-1)

    self._states = new_states
    return jax.tree_util.tree_map(lambda a: [*a], actions)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._states = None

  def observe(self, action, next_timestep: dm_env.TimeStep):
    pass

  def update(self, wait: bool = True):
    self._variable_client.update(wait)

  @property
  def _params(self) -> hk.Params:
    return self._variable_client.params
