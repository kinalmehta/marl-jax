import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import datetime
import functools

from absl import app
from absl import flags
import jax
import launchpad as lp

from marl import experiments
from marl import specs as ma_specs
from marl.agents import impala
from marl.agents import opre
from marl.agents.networks import ArrayFE
from marl.agents.networks import ImageFE
from marl.agents.networks import MeltingpotFE
from marl.experiments import config as ma_config
from marl.experiments import inference_server
from marl.utils import helpers
from marl.utils import lp_utils as ma_lp_utils
from marl.utils.experiment_utils import make_experiment_logger

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "async_distributed",
    False,
    "Should an agent be executed in an off-policy distributed way",
)
flags.DEFINE_bool("run_eval", False, "Whether to run evaluation.")
flags.DEFINE_bool(
    "all_parallel", False,
    "Flag to run all agents in parallel using vmap. Only use if GPU with large memory is available."
)
flags.DEFINE_enum(
    "env_name",
    "overcooked",
    ["meltingpot", "overcooked"],
    "Environment to train on",
)
flags.DEFINE_string(
    "map_name", "cramped_room",
    "Meltingpot/Overcooked Map to train on. Only used when 'env_name' is 'meltingpot' or 'overcooked'"
)
flags.DEFINE_enum("algo_name", "IMPALA",
                  ["IMPALA", "PopArtIMPALA", "OPRE", "PopArtOPRE"],
                  "Algorithm to train")
flags.DEFINE_bool("record_video", False,
                  "Whether to record videos. (Only use during evaluation)")
flags.DEFINE_integer("reward_scale", 1, "Reward scale factor.")
flags.DEFINE_bool("prosocial", False,
                  "Whether to use shared reward for prosocial training.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 200_000_000, "Number of env steps to run.")
flags.DEFINE_string("exp_log_dir", "./results/",
                    "Directory to store experiment logs in.")
flags.DEFINE_bool("use_tb", True, "Flag to enable tensorboard logging.")
flags.DEFINE_bool("use_wandb", False, "Flag to enable wandb.ai logging.")
flags.DEFINE_string("wandb_entity", "", "Entity name for wandb account.")
flags.DEFINE_string("wandb_project", "marl-jax",
                    "Project name for wandb logging.")
flags.DEFINE_string("wandb_tags", "", "Comma separated list of tags for wandb.")
flags.DEFINE_string("available_gpus", "0", "Comma separated list of GPU ids.")
flags.DEFINE_integer(
    "num_actors", 2,
    "Number of actors to use (should be less than total number of CPU cores).")
flags.DEFINE_integer("actors_per_node", 1, "Number of actors per thread.")
flags.DEFINE_bool("inference_server", False, "Whether to run inference server.")
flags.DEFINE_string("experiment_dir", None,
                    "Directory to resume experiment from.")


def build_experiment_config():
  """Builds experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.

  # creating the following values so that FLAGS doesn't need to be pickled
  map_name = FLAGS.map_name
  reward_scale = FLAGS.reward_scale
  autoreset = False
  prosocial = FLAGS.prosocial
  record = FLAGS.record_video
  memory_efficient = not FLAGS.all_parallel

  if FLAGS.experiment_dir:
    assert FLAGS.algo_name in FLAGS.experiment_dir, f"experiment_dir must be a {FLAGS.algo_name} experiment"
    assert FLAGS.env_name in FLAGS.experiment_dir, f"experiment_dir must be a {FLAGS.env_name} experiment"
    assert FLAGS.map_name in FLAGS.experiment_dir, f"experiment_dir must be a {FLAGS.env_name} experiment with map_name {FLAGS.map_name}"
    experiment_dir = FLAGS.experiment_dir
    experiment_name = experiment_dir.split("/")[-1]
  else:
    experiment_name = f"{FLAGS.algo_name}_{FLAGS.seed}_{FLAGS.env_name}"
    experiment_name += f"_{FLAGS.map_name}"
    experiment_name += f"_{datetime.datetime.now()}"
    experiment_name = experiment_name.replace(" ", "_")
    experiment_dir = os.path.join(FLAGS.exp_log_dir, experiment_name)

  wandb_config = {
      "project": FLAGS.wandb_project,
      "entity": FLAGS.wandb_entity,
      "name": experiment_name,
      "group": experiment_name,
      "resume": True if FLAGS.experiment_dir else False,
      "tags": [st for st in FLAGS.wandb_tags.split(",") if st],
  }

  feature_extractor = ArrayFE

  # Create environment factory
  if FLAGS.env_name == "overcooked":
    env_factory = lambda seed: helpers.make_overcooked_environment(
        seed,
        map_name,
        autoreset=autoreset,
        reward_scale=reward_scale,
        global_observation_sharing=True,
        record=record)
    num_options = 8
  elif FLAGS.env_name == "ssd":
    env_factory = lambda seed: helpers.make_ssd_environment(
        seed,
        map_name,
        autoreset=autoreset,
        reward_scale=reward_scale,
        team_reward=prosocial,
        global_observation_sharing=True,
        record=record)
    feature_extractor = ImageFE
    num_options = 8
  elif FLAGS.env_name == "meltingpot":
    env_factory = lambda seed: helpers.env_factory(
        seed,
        map_name,
        autoreset=autoreset,
        shared_reward=prosocial,
        reward_scale=reward_scale,
        shared_obs=True,
        record=record)
    feature_extractor = MeltingpotFE
    num_options = 16
  else:
    raise ValueError(f"Unknown env_name {FLAGS.env_name}")

  environment_specs = ma_specs.MAEnvironmentSpec(env_factory(0))

  if FLAGS.algo_name == "IMPALA":
    # Create network
    network_factory = functools.partial(
        impala.make_network, feature_extractor=feature_extractor)
    network = network_factory(
        environment_specs.get_single_agent_environment_specs())
    # Construct the agent.
    config = impala.IMPALAConfig(
        n_agents=environment_specs.num_agents,
        memory_efficient=memory_efficient)
    core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
    builder = impala.IMPALABuilder(config, core_state_spec=core_spec)
  elif FLAGS.algo_name == "PopArtIMPALA":
    # Create network
    network_factory = functools.partial(
        impala.make_network_2, feature_extractor=feature_extractor)
    network = network_factory(
        environment_specs.get_single_agent_environment_specs())
    # Construct the agent.
    config = impala.IMPALAConfig(
        n_agents=environment_specs.num_agents,
        memory_efficient=memory_efficient)
    core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
    builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

  elif FLAGS.algo_name == "OPRE":
    # Create network
    network_factory = functools.partial(
        opre.make_network,
        num_options=num_options,
        feature_extractor=feature_extractor)
    network = network_factory(
        environment_specs.get_single_agent_environment_specs())
    # Construct the agent.
    config = opre.OPREConfig(
        n_agents=environment_specs.num_agents,
        num_options=num_options,
        memory_efficient=memory_efficient)
    core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
    builder = opre.OPREBuilder(config, core_state_spec=core_spec)
  elif FLAGS.algo_name == "PopArtOPRE":
    # Create network
    network_factory = functools.partial(
        opre.make_network_2,
        num_options=num_options,
        feature_extractor=feature_extractor)
    network = network_factory(
        environment_specs.get_single_agent_environment_specs())
    # Construct the agent.
    config = opre.OPREConfig(
        n_agents=environment_specs.num_agents,
        num_options=num_options,
        memory_efficient=memory_efficient)
    core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
    builder = opre.PopArtOPREBuilder(config, core_state_spec=core_spec)
  else:
    raise ValueError(f"Unknown algo_name {FLAGS.algo_name}")

  return (
      experiments.MAExperimentConfig(
          builder=builder,
          environment_factory=env_factory,
          network_factory=network_factory,
          logger_factory=functools.partial(
              make_experiment_logger,
              log_dir=experiment_dir,
              use_tb=FLAGS.use_tb,
              use_wandb=FLAGS.use_wandb,
              wandb_config=wandb_config,
          ),
          environment_spec=environment_specs,
          evaluator_env_factories=None,
          seed=FLAGS.seed,
          max_num_actor_steps=FLAGS.num_steps,
          resume_training=True if FLAGS.experiment_dir else False,
      ),
      experiment_dir,
  )


def main(_):
  assert not FLAGS.record_video, "Video recording is not supported during training"
  config, experiment_dir = build_experiment_config()
  ckpt_config = ma_config.CheckpointingConfig(
      max_to_keep=3, directory=experiment_dir, add_uid=False)
  if FLAGS.async_distributed:

    nodes_on_gpu = helpers.node_allocation(
        FLAGS.available_gpus,
        FLAGS.inference_server)
    program = experiments.make_distributed_experiment(
          experiment=config,
          num_actors=FLAGS.num_actors * FLAGS.actors_per_node,
          inference_server_config=inference_server.InferenceServerConfig(
              batch_size=min(8, FLAGS.num_actors // 2),
              update_period=1,
              timeout=datetime.timedelta(
                  seconds=1, milliseconds=0, microseconds=0),
          ) if FLAGS.inference_server else None,
          num_actors_per_node=FLAGS.actors_per_node,
          checkpointing_config=ckpt_config,
      )
    local_resources = ma_lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=nodes_on_gpu)

    lp.launch(
        program,
        launch_type="local_mp",
        terminal="current_terminal",
        local_resources=local_resources,
    )
  else:
    experiments.run_experiment(
        experiment=config,
        checkpointing_config=ckpt_config,
        num_eval_episodes=0)


if __name__ == "__main__":
  app.run(main)
