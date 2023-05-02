import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
import sys

cwd = os.getcwd()
sys.path.append(cwd)
os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import functools

from absl import app
from absl import flags
from meltingpot.python import scenario

from marl import experiments
from marl.experiments import config as ma_config
from marl.utils import helpers
from marl.utils.experiment_utils import make_experiment_logger
import train

FLAGS = flags.FLAGS


def main(_):
  if FLAGS.experiment_dir is None:
    raise ValueError("experiment_dir must be specified")

  config, experiment_dir = train.build_experiment_config()

  ckpt_config = ma_config.CheckpointingConfig(
      max_to_keep=3, directory=experiment_dir, add_uid=False)

  config.logger_factory = functools.partial(
      make_experiment_logger, log_dir=experiment_dir, use_tb=False)

  if FLAGS.env_name == "meltingpot":

    # running evaluation on substrate
    experiments.run_evaluation(
        config, ckpt_config, environment_name=FLAGS.map_name)

    scenarios_for_substrate = sorted(
        list(scenario.SCENARIOS_BY_SUBSTRATE[FLAGS.map_name]))

    # running evaluations on scenarios
    for scenario_name in scenarios_for_substrate:
      env_factory = functools.partial(
          helpers.make_meltingpot_scenario, scenario_name=scenario_name)
      env_factory(0)
      config.environment_factory = env_factory
      experiments.run_evaluation(
          config, ckpt_config, environment_name=scenario_name)
  else:
    # running evaluation on substrate
    experiments.run_evaluation(
        config,
        ckpt_config,
        environment_name=f"{FLAGS.env_name}_{FLAGS.map_name}")


if __name__ == "__main__":
  app.run(main)
