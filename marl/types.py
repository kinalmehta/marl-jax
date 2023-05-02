import dataclasses
from typing import Callable, Generic, NamedTuple, Union

from acme import types
from acme.agents.jax.actor_core import RecurrentState
from acme.jax import networks
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
import jax.numpy as jnp
import numpy as np
import optax
from rlax import PopArtState

SingleAgentAction = Union[int, float, np.ndarray]
Action = Union[SingleAgentAction, dict[str, SingleAgentAction]]
Actions = list[Action]

SingleAgentReward = Union[int, float]
Reward = Union[SingleAgentReward, dict[str, SingleAgentReward]]
Discount = Reward

# Only simple observations & discrete action spaces for now.
Observation = Union[jnp.ndarray, dict[str, np.ndarray]]
Observations = list[Observation]
Action = int
Actions = list[Action]
Outputs = tuple[tuple[networks.Logits, networks.Value], RecurrentState]
PolicyValueInitFn = Callable[[networks.PRNGKey, RecurrentState],
                             networks.Params]
PolicyValueFn = Callable[[networks.Params, Observation, RecurrentState],
                         Outputs]
CriticFn = Callable[[networks.Params, Observation], Outputs]
RecurrentStateFn = Callable[[jax_types.PRNGKey], RecurrentState]


@dataclasses.dataclass
class RecurrentNetworks(Generic[RecurrentState]):
  """Pure functions representing recurrent network components.

    Attributes:
      forward_fn: Selects next action using the network at the given recurrent
        state.

      unroll_init_fn: Initializes params for forward_fn and unroll_fn.

      unroll_fn: Applies the unrolled network to a sequence of observations, for
        learning.

      initial_state_fn: Recurrent state at the beginning of an episode.
    """
  forward_fn: PolicyValueFn  # Inference mode forward
  unroll_fn: PolicyValueFn  # Training mode forward
  unroll_init_fn: PolicyValueInitFn  # Initialize params for unroll_fn
  initial_state_fn: RecurrentStateFn  # Initial recurrent state


@dataclasses.dataclass
class ActorCriticRecurrentNetworks(Generic[RecurrentState]):
  """Pure functions representing recurrent network components.

    Attributes:
      forward_fn: Selects next action using the network at the given recurrent
        state.

      unroll_init_fn: Initializes params for forward_fn and unroll_fn.

      unroll_fn: Applies the unrolled network to a sequence of observations, for
        learning.

      initial_state_fn: Recurrent state at the beginning of an episode.

      critic_fn: Critic network.

      critic_init_fn: Initializes params for critic_fn.
    """
  forward_fn: PolicyValueFn  # Inference mode forward
  unroll_fn: PolicyValueFn  # Training mode forward
  unroll_init_fn: PolicyValueInitFn  # Initialize params for unroll_fn
  initial_state_fn: RecurrentStateFn  # Initial recurrent state
  critic_fn: CriticFn  # Critic forward


@dataclasses.dataclass
class PopArtLayer:
  """Container for PopArt state and update function."""
  init_fn: Callable[[], PopArtState]
  update_fn: Callable


class TrainingState(NamedTuple):
  """Training state consists of network parameters and optimiser state."""
  params: networks_lib.Params
  opt_state: optax.OptState


class PopArtTrainingState(NamedTuple):
  """Training state for PopArt normalized networks."""
  params: networks_lib.Params
  opt_state: optax.OptState
  popart_state: PopArtState


class TrainingData(NamedTuple):
  """Container for agent specific training data."""
  observation: types.NestedArray
  action: types.NestedArray
  reward: types.NestedArray
  discount: types.NestedArray
  extras: types.NestedArray


class Transition(NamedTuple):
  """Container for a transition."""
  observations: types.NestedArray
  actions: types.NestedArray
  rewards: types.NestedArray
  discounts: types.NestedArray
  next_observations: types.NestedArray
  extras: types.NestedArray = ()
  next_extras: types.NestedArray = ()
