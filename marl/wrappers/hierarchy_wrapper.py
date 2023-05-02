"""A wrapper that adds diversity vector into the observation for a multi-agent environment."""

from acme import specs
from acme import types
from acme.wrappers import base
import dm_env
import numpy as np


class HierarchyVecWrapper(base.EnvironmentWrapper):
  """A wrapper that adds hierarchy vector for diversity into the observation."""

  def __init__(self,
               environment: dm_env.Environment,
               diversity_dim: int,
               seed: int = 0):
    super().__init__(environment)
    np.random.seed(seed)
    self.diversity_dim = diversity_dim
    self._num_players = len(self._environment.action_spec())
    self._step = 0
    self._reset_every = 500
    self.diversity_val = None

  def reset(self) -> dm_env.TimeStep:
    # Initialize with zeros of the appropriate shape/dtype.
    timestep = self._environment.reset()
    if self.diversity_val is not None:
      assert self.diversity_val < self.diversity_dim, "Diversity value out of bounds."
      self.cur_diversity_vec = np.ones(
          self._num_players, dtype=np.int32) * self.diversity_val
    self._step = 0
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def _augment_observation(
      self,
      timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    assert (
        type(timestep.observation) in [list, tuple]
    ), f"Unsupported observation type {type(timestep.observation)}. Only supports `list` type observations for a multi-agent environment."

    # initialize the diversity vector
    if (self.diversity_val is None) and (self._step % self._reset_every == 0):
      self.cur_diversity_vec = np.random.randint(
          0, self.diversity_dim, size=self._num_players, dtype=np.int32)

    new_obs = list()
    for agent_no, obs in enumerate(timestep.observation):
      if type(obs) is dict:
        obs["observation"]["agent_id"] = self.cur_diversity_vec[agent_no]
        if "global" in obs:
          obs["global"]["observation"]["agent_id"] = self.cur_diversity_vec
      else:
        raise Exception("unknown observation type")
      new_obs.append(obs)

    self._step += 1
    return timestep._replace(observation=new_obs)

  def observation_spec(self):
    new_obs_spec = list()
    obs_spec = self._environment.observation_spec()
    diversity_vec_spec = specs.DiscreteArray(
        self.diversity_dim, name="agent_id")
    diversity_global_spec = specs.BoundedArray(
        shape=(self._num_players,),
        dtype=diversity_vec_spec.dtype,
        minimum=diversity_vec_spec.minimum,
        maximum=diversity_vec_spec.maximum,
        name=diversity_vec_spec.name,
    )

    for obs_s in obs_spec:
      if type(obs_s) is dict:
        obs_s["observation"]["agent_id"] = diversity_vec_spec
        if "global" in obs_s:
          obs_s["global"]["observation"]["agent_id"] = diversity_global_spec
      else:
        raise Exception("unknown observation type")
      new_obs_spec.append(obs_s)
    return new_obs_spec
