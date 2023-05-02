"""A wrapper that merges the data of all agents in the leading dimension for a multi-agent environment."""

from acme import specs
from acme import types
from acme.wrappers import base
import dm_env
import numpy as np
import tree


class MergeWrapper(base.EnvironmentWrapper):
  """A wrapper that adds hierarchy vector for diversity into the observation."""

  def __init__(self, environment: dm_env.Environment):
    super().__init__(environment)
    self._num_players = len(self._environment.action_spec())

  def _stack_data(self, data: types.NestedArray) -> types.NestedArray:
    return tree.map_structure(lambda *x: np.stack(x), *data)

  def _update_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    new_timestep = dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=self._stack_data(timestep.reward),
        discount=self._stack_data(timestep.discount),
        observation=self._stack_data(timestep.observation))
    return new_timestep

  def reset(self) -> dm_env.TimeStep:
    return self._update_timestep(self._environment.reset())

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    return self._update_timestep(self._environment.step(action))

  def _update_spec(self, old_spec):
    new_spec = tree.map_structure(
        lambda s: specs.Array((self._num_players, *s.shape), s.dtype),
        old_spec[0])
    return new_spec

  def observation_spec(self) -> specs.Array:
    obs_spec = self._update_spec(self._environment.observation_spec())
    return obs_spec

  def action_spec(self) -> specs.Array:
    act_spec = self._update_spec(self._environment.action_spec())
    return act_spec

  def reward_spec(self):
    rew_spec = self._update_spec(self._environment.reward_spec())
    return rew_spec

  def discount_spec(self):
    disc_spec = self._update_spec(self._environment.discount_spec())
    return disc_spec

  def single_observation_spec(self) -> specs.Array:
    return self._environment.observation_spec()[0]

  def single_action_spec(self) -> specs.Array:
    return self._environment.action_spec()[0]

  def single_reward_spec(self) -> specs.Array:
    return self._environment.reward_spec()[0]

  def single_discount_spec(self) -> specs.Array:
    return self._environment.discount_spec()[0]
