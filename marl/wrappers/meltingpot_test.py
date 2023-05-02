"""Tests for MeltingPot Environment Wrapper"""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from acme import types
from acme import wrappers
from acme.jax import utils
import dm_env
import tree

SKIP_MELTINGPOT_TESTS = False
SKIP_MELTINGPOT_MESSAGE = "meltingpot not installed"

import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from meltingpot.python import scenario
from meltingpot.python import substrate
from ml_collections import config_dict

from marl.wrappers import meltingpot_wrapper


def _validate_spec(spec: types.NestedSpec, value: types.NestedArray):
  """Validate a value from a potentially nested spec."""
  tree.assert_same_structure(value, spec)
  tree.map_structure(lambda s, v: s.validate(v), spec, value)


def env_creator_substrate(env_config):
  env = substrate.build(config_dict.ConfigDict(env_config))
  return env


def env_creator_scenario(env_config):
  env = scenario.build(config_dict.ConfigDict(env_config))
  return env


from typing import Any, NamedTuple


class EnvironmentSpec(NamedTuple):
  """Full specification of the domains used by a given environment."""

  # TODO(b/144758674): Use NestedSpec type here.
  observations: Any
  actions: Any
  rewards: Any
  discounts: Any


def make_environment_spec(environment: dm_env.Environment) -> EnvironmentSpec:
  """Returns an `EnvironmentSpec` describing values used by an environment."""
  return EnvironmentSpec(
      observations=environment.observation_spec(),
      actions=environment.action_spec(),
      rewards=environment.reward_spec(),
      discounts=environment.discount_spec(),
  )


@unittest.skipIf(SKIP_MELTINGPOT_TESTS, SKIP_MELTINGPOT_MESSAGE)
class MeltingPotEnvironmentTest(parameterized.TestCase):

  def test_env_run(self):
    env_config = substrate.get_config("running_with_scissors_in_the_matrix"
                                     )  # running_with_scissors_in_the_matrix
    raw_env = env_creator_substrate(env_config)

    env = meltingpot_wrapper.MeltingPotWrapper(raw_env)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.ObservationActionRewardWrapper(env)
    environment_spec = make_environment_spec(env)

    num_agents = env.num_agents

    env.reset()

    print(num_agents)
    print(environment_spec.discounts)
    print(environment_spec.observations[0])
    print(environment_spec.actions)
    print(environment_spec.rewards)
    # print(timestep)
    dummy_obs = utils.zeros_like(environment_spec.observations)
    dummy_obs_batched = utils.add_batch_dim(dummy_obs)

    print(dummy_obs_batched)
    # self.assertIn('observation', timestep)
    # self.assertIn('reward', timestep)
    # self.assertIn('discount', timestep)
    # self.assertIn('done', timestep)


if __name__ == "__main__":
  absltest.main()
