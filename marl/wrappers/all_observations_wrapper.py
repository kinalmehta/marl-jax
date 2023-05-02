# Modified from https://github.com/deepmind/meltingpot/blob/main/meltingpot/python/utils/scenarios/wrappers/all_observations_wrapper.py

from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any, Union

from acme.wrappers import base as acme_base
import dm_env
from meltingpot.python.utils.substrates import substrate
import numpy as np

GLOBAL_KEY = 'global'
OBSERVATIONS_KEY = 'observation'
REWARDS_KEY = 'reward'
ACTIONS_KEY = 'action'


class Wrapper(substrate.Substrate):
  """Exposes actions/observations/rewards from all players to all players."""

  def __init__(self,
               env: substrate.Substrate,
               observations_to_share: Collection[str] = (),
               share_actions: bool = False,
               share_rewards: bool = False) -> None:
    """Wraps an environment.

    Args:
      env: environment to wrap. When this wrapper closes env will also be
        closed.
      observations_to_share: observation keys to share with other players.
      share_actions: whether to show other players actions.
      share_rewards: whether to show other players rewards.
    """
    super().__init__(env)
    self._observations_to_share = observations_to_share
    self._share_actions = share_actions
    self._share_rewards = share_rewards

    action_spec = super().action_spec()
    self._num_players = len(action_spec)
    self._missing_actions = [spec.generate_value() for spec in action_spec]
    self._action_dtype = action_spec[0].dtype

  def _shared_observation(self, observations: Sequence[Mapping[str, Any]],
                          rewards: Sequence[Union[float, np.ndarray]],
                          actions: Sequence[int]):
    """Returns shared observations."""
    # We assume that this comes from this wrapper and so all shared observations
    # are the same for all players.
    shared_observation = dict(observations[0].get(GLOBAL_KEY, {}))

    additional_observations = dict({
        name: np.stack([obs[name] for obs in observations
                       ]) for name in self._observations_to_share
    })
    if additional_observations:
      shared_observation[OBSERVATIONS_KEY] = dict(
          shared_observation.get(OBSERVATIONS_KEY, {}),
          **additional_observations)

    if self._share_rewards:
      shared_observation[REWARDS_KEY] = np.stack(rewards)

    if self._share_actions:
      shared_observation[ACTIONS_KEY] = np.array(
          actions, dtype=self._action_dtype)

    return dict(shared_observation)

  def _adjusted_timestep(self, timestep: dm_env.TimeStep,
                         actions: Sequence[int]) -> dm_env.TimeStep:
    """Returns timestep with shared observations."""
    shared_observation = self._shared_observation(
        observations=timestep.observation,
        rewards=timestep.reward,
        actions=actions)
    if not shared_observation:
      return timestep
    observations = tuple(
        dict(obs, **{GLOBAL_KEY: shared_observation})
        for obs in timestep.observation)
    return timestep._replace(observation=observations)

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    return self._adjusted_timestep(timestep, self._missing_actions)

  def step(self, actions: Sequence[int]) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().step(actions)
    return self._adjusted_timestep(timestep, actions)

  def _shared_observation_spec(self, observation_spec: Mapping[str, Any],
                               reward_spec: dm_env.specs.Array,
                               action_spec: dm_env.specs.DiscreteArray):
    """Returns spec of shared observations."""
    shared_observation_spec = dict(observation_spec.get(GLOBAL_KEY, {}))

    additional_spec = {}
    for name in self._observations_to_share:
      spec = observation_spec[name]
      additional_spec[name] = spec.replace(
          shape=(self._num_players,) + spec.shape, name=name)
    if additional_spec:
      shared_observation_spec[OBSERVATIONS_KEY] = dict(
          shared_observation_spec.get(OBSERVATIONS_KEY, {}), **additional_spec)

    if self._share_rewards:
      shared_observation_spec[REWARDS_KEY] = reward_spec.replace(
          shape=(self._num_players,), name=REWARDS_KEY)

    if self._share_actions:
      shared_observation_spec[ACTIONS_KEY] = dm_env.specs.BoundedArray(
          shape=(self._num_players,),
          dtype=action_spec.dtype,
          minimum=action_spec.minimum,
          maximum=action_spec.maximum,
          name=ACTIONS_KEY)

    return dict(shared_observation_spec)

  def observation_spec(self):
    """See base class."""
    observation_spec = super().observation_spec()
    assert all(spec == observation_spec[0] for spec in observation_spec)
    observation_spec = observation_spec[0]

    action_spec = super().action_spec()
    assert all(spec == action_spec[0] for spec in action_spec)
    action_spec = action_spec[0]

    reward_spec = super().reward_spec()
    assert all(spec == reward_spec[0] for spec in reward_spec)
    reward_spec = reward_spec[0]

    shared_observation_spec = self._shared_observation_spec(
        observation_spec=observation_spec,
        reward_spec=reward_spec,
        action_spec=action_spec)
    observation_spec = dict(observation_spec,
                            **{GLOBAL_KEY: shared_observation_spec})
    return (observation_spec,) * self._num_players


class AllObservationWrapper(acme_base.EnvironmentWrapper):
  """Exposes actions/observations/rewards from all players to all players."""

  def __init__(self,
               env,
               observations_to_share: Sequence[str] = (),
               share_actions: bool = False,
               share_rewards: bool = False) -> None:
    """Wraps an environment.

    Args:
      env: environment to wrap. When this wrapper closes env will also be
        closed.
      observations_to_share: observation keys to share with other players.
      share_actions: whether to show other players actions.
      share_rewards: whether to show other players rewards.
    """
    super().__init__(env)
    self._observations_to_share = observations_to_share
    self._share_actions = share_actions
    self._share_rewards = share_rewards

    action_spec = super().action_spec()
    self._num_players = len(action_spec)
    self._missing_actions = [spec.generate_value() for spec in action_spec]
    self._action_dtype = action_spec[0].dtype

  def _shared_observation(self, observations: Sequence[Mapping[str, Any]],
                          rewards: Sequence[Union[float, np.ndarray]],
                          actions: Sequence[int]):
    """Returns shared observations."""
    # We assume that this comes from this wrapper and so all shared observations
    # are the same for all players.
    shared_observation = dict(observations[0].get(GLOBAL_KEY, {}))

    additional_observations = dict({
        name: np.stack([obs[name] for obs in observations
                       ]) for name in self._observations_to_share
    })
    if additional_observations:
      shared_observation[OBSERVATIONS_KEY] = dict(
          shared_observation.get(OBSERVATIONS_KEY, {}),
          **additional_observations)

    if self._share_rewards:
      shared_observation[REWARDS_KEY] = np.stack(rewards)

    if self._share_actions:
      shared_observation[ACTIONS_KEY] = np.array(
          actions, dtype=self._action_dtype)

    return dict(shared_observation)

  def _adjusted_timestep(self, timestep: dm_env.TimeStep,
                         actions: Sequence[int]) -> dm_env.TimeStep:
    """Returns timestep with shared observations."""
    shared_observation = self._shared_observation(
        observations=timestep.observation,
        rewards=timestep.reward,
        actions=actions)
    if not shared_observation:
      return timestep
    observations = tuple(
        dict(obs, **{GLOBAL_KEY: shared_observation})
        for obs in timestep.observation)
    return timestep._replace(observation=observations)

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    return self._adjusted_timestep(timestep, self._missing_actions)

  def step(self, actions: Sequence[int]) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().step(actions)
    return self._adjusted_timestep(timestep, actions)

  def _shared_observation_spec(self, observation_spec: Mapping[str, Any],
                               reward_spec: dm_env.specs.Array,
                               action_spec: dm_env.specs.DiscreteArray):
    """Returns spec of shared observations."""
    shared_observation_spec = dict(observation_spec.get(GLOBAL_KEY, {}))

    additional_spec = {}
    for name in self._observations_to_share:
      spec = observation_spec[name]
      additional_spec[name] = spec.replace(
          shape=(self._num_players,) + spec.shape, name=name)
    if additional_spec:
      shared_observation_spec[OBSERVATIONS_KEY] = dict(
          shared_observation_spec.get(OBSERVATIONS_KEY, {}), **additional_spec)

    if self._share_rewards:
      shared_observation_spec[REWARDS_KEY] = reward_spec.replace(
          shape=(self._num_players,), name=REWARDS_KEY)

    if self._share_actions:
      shared_observation_spec[ACTIONS_KEY] = dm_env.specs.BoundedArray(
          shape=(self._num_players,),
          dtype=action_spec.dtype,
          minimum=action_spec.minimum,
          maximum=action_spec.maximum,
          name=ACTIONS_KEY)

    return dict(shared_observation_spec)

  def observation_spec(self):
    """See base class."""
    observation_spec = super().observation_spec()
    assert all(spec == observation_spec[0] for spec in observation_spec)
    observation_spec = observation_spec[0]

    action_spec = super().action_spec()
    assert all(spec == action_spec[0] for spec in action_spec)
    action_spec = action_spec[0]

    reward_spec = super().reward_spec()
    assert all(spec == reward_spec[0] for spec in reward_spec)
    reward_spec = reward_spec[0]

    shared_observation_spec = self._shared_observation_spec(
        observation_spec=observation_spec,
        reward_spec=reward_spec,
        action_spec=action_spec)
    observation_spec = dict(observation_spec,
                            **{GLOBAL_KEY: shared_observation_spec})
    return (observation_spec,) * self._num_players