"""A wrapper that puts the previous action and reward into the observation for a multi-agent environment."""

from acme import types
from acme.wrappers import base
import dm_env
import tree


class ObservationActionRewardWrapper(base.EnvironmentWrapper):
  """A wrapper that puts the previous action and reward into the observation."""

  # def _fix_obs_types(self, obs: types.NestedArray) -> types.NestedArray:
  #   """Process global observation."""
  #   obs = dict(obs)
  #   if "global" in obs:
  #     obs["global"] = dict(obs["global"])
  #     obs["global"]["observation"] = dict(obs["global"]["observation"])
  #   return obs

  def reset(self) -> dm_env.TimeStep:
    # Initialize with zeros of the appropriate shape/dtype.
    action = tree.map_structure(lambda x: x.generate_value(),
                                self._environment.action_spec())
    reward = tree.map_structure(lambda x: x.generate_value(),
                                self._environment.reward_spec())
    timestep = self._environment.reset()
    new_timestep = self._augment_observation(action, reward, timestep)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    new_timestep = self._augment_observation(action, timestep.reward, timestep)
    return new_timestep

  def _augment_observation(
      self,
      action: types.NestedArray,
      reward: types.NestedArray,
      timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    assert (
        type(timestep.observation) in [list, tuple]
    ), f"Unsupported observation type {type(timestep.observation)}. Only supports `list` type observations for a multi-agent environment."

    new_obs = list()
    for agent_no, obs in enumerate(timestep.observation):
      # obs = self._fix_obs_types(obs)
      new_obs.append({
          "observation": obs,
          "action": action[agent_no],
          "reward": reward[agent_no],
      })
    return timestep._replace(observation=new_obs)

  def observation_spec(self):
    new_obs_spec = list()
    obs_spec = self._environment.observation_spec()
    action_spec = self._environment.action_spec()
    reward_spec = self._environment.reward_spec()
    for agent_no, obs_s in enumerate(obs_spec):
      # obs_s = self._fix_obs_types(obs_s)
      new_obs_spec.append({
          "observation": obs_s,
          "action": action_spec[agent_no],
          "reward": reward_spec[agent_no],
      })
    return new_obs_spec
