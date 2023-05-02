"""Wraps an Melting Pot RL environment to be used as a dm_env environment."""

from typing import Union

from acme import specs
from acme import types
import dm_env
import dmlab2d
from meltingpot.python.utils.scenarios.scenario import Scenario
from meltingpot.python.utils.substrates.substrate import Substrate
import numpy as np

from marl import types as marl_types

USED_OBS_KEYS = {"global", "RGB", "INVENTORY", "READY_TO_SHOOT"}


class MeltingPotWrapper(dmlab2d.Environment):
  """Environment wrapper for MeltingPot RL environments."""

  # Note: we don't inherit from base.EnvironmentWrapper because that class
  # assumes that the wrapped environment is a dm_env.Environment.

  def __init__(self,
               environment: Union[Substrate, Scenario],
               shared_reward: bool = False,
               reward_scale: float = 1.0):
    self._environment = environment
    self.reward_scale = reward_scale
    self._reset_next_step = True
    self._shared_reward = shared_reward
    self.is_turn_based = False
    self.num_agents = len(self._environment.action_spec())
    self.num_actions = self._environment.action_spec()[0].num_values
    self.agents = list(range(self.num_agents))
    self.obs_spec = [
        self._remove_unwanted_observations(obs_spec)
        for obs_spec in self._environment.observation_spec()
    ]

  def _remove_unwanted_observations(self, observation: marl_types.Observation):
    """Removes unwanted observations from a marl observation."""
    return {
        key: value for key, value in observation.items() if key in USED_OBS_KEYS
    }

  def _refine_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Refines a dm_env.TimeStep to use dict instead of list for data of multiple-agents."""
    reward = [reward * self.reward_scale for reward in timestep.reward]
    if self._shared_reward:
      reward = [np.mean(reward)] * self.num_agents
    discount = [timestep.discount] * self.num_agents
    observation = [
        self._remove_unwanted_observations(agent_obs)
        for agent_obs in timestep.observation
    ]
    # observation = timestep.observation
    return dm_env.TimeStep(timestep.step_type, reward, discount, observation)

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    timestep = self._environment.reset()
    timestep = self._refine_timestep(timestep)
    return timestep

  def step(self, actions: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()
    # actions = [actions[f"agent_{i}"] for i in range(self.num_agents)]
    timestep = self._environment.step(actions)
    timestep = self._refine_timestep(timestep)
    if timestep.last():
      self._reset_next_step = True
      self._env_done = True
    return timestep

  def env_done(self) -> bool:
    """Check if env is done.
        Returns:
            bool: bool indicating if env is done.
        """
    done = not self.agents or self._env_done
    return done

  def observation_spec(self) -> list[marl_types.Observation]:
    return self.obs_spec

  def action_spec(self,) -> list[specs.DiscreteArray]:
    return self._environment.action_spec()

  def reward_spec(self) -> list[specs.Array]:
    return self._environment.reward_spec()

  def discount_spec(self) -> list[specs.BoundedArray]:
    return [self._environment.discount_spec()] * self.num_agents

  def extras_spec(self) -> list[specs.BoundedArray]:
    """Extra data spec.
        Returns:
            List[specs.BoundedArray]: spec for extra data.
        """
    return list()

  @property
  def environment(self) -> dmlab2d.Environment:
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    """Expose any other attributes of the underlying environment."""
    return getattr(self._environment, name)
