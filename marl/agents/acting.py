"""Multi-agent actor implementation."""

from typing import Optional

from acme import adders
from acme import core
from acme.jax import variable_utils
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

from marl import types
from marl.utils import experiment_utils as ma_utils


class MAActor(core.Actor):
  """A recurrent multi-agent actor."""

  _states: list[hk.LSTMState]
  _prev_states: list[hk.LSTMState]
  _prev_logits: list[jnp.ndarray]

  def __init__(
      self,
      forward_fn: types.PolicyValueFn,
      initial_state_fn: types.RecurrentStateFn,
      n_agents: int,
      rng: hk.PRNGSequence,
      variable_client: Optional[variable_utils.VariableClient] = None,
      adder: Optional[adders.Adder] = None,
  ):
    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._forward = forward_fn

    self._rng = rng
    self.n_agents = n_agents

    def initialize_states(rng_sequence: hk.PRNGSequence,) -> list[hk.LSTMState]:
      """Initialize the recurrent states of the actor."""
      states = list()
      for _ in range(self.n_agents):
        states.append(initial_state_fn(next(rng_sequence)))
      return states

    self._initial_states = ma_utils.merge_data(initialize_states(self._rng))

  def select_action(self, observations: types.Observations) -> types.Actions:
    if self._states is None:
      self._states = self._initial_states

    observations = jax.tree_util.tree_map(lambda x: jnp.array(x), observations)
    (logits, _), new_states = self._forward(self._params, observations,
                                            self._states)

    actions = jax.random.categorical(next(self._rng), logits)

    self._prev_logits = logits
    self._prev_states = self._states
    self._states = new_states

    return jax.tree_util.tree_map(lambda a: [*a], actions)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._states = None

  def observe(
      self,
      action: types.Actions,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return
    extras = {
        "logits": self._prev_logits,
        "core_state": {
            "hidden": self._prev_states.hidden,
            "cell": self._prev_states.cell
        },
    }
    self._adder.add(ma_utils.merge_data(action), next_timestep, extras)

  def update(self, wait: bool = False):
    if self._variable_client is not None:
      self._variable_client.update(wait)

  @property
  def _params(self) -> Optional[hk.Params]:
    if self._variable_client is None:
      # If self._variable_client is None then we assume self._forward  does not
      # use the parameters it is passed and just return None.
      return None
    return self._variable_client.params
