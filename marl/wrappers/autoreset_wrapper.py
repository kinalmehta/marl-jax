"""A wrapper that adds diversity vector into the observation for a multi-agent environment."""

from acme import types
from acme.wrappers import base
import dm_env


class AutoResetWrapper(base.EnvironmentWrapper):
  """A wrapper that adds hierarchy vector for diversity into the observation."""

  def __init__(self, environment: dm_env.Environment):
    super().__init__(environment)

  def reset(self) -> dm_env.TimeStep:
    return self._environment.reset()

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    if self._reset_next_step:
      return self.reset()
    return self._environment.step(action)
