"""Multi-agent evaluator implementation."""

from acme import core
from acme.jax import variable_utils
import dm_env
import haiku as hk
import jax

from marl.utils import experiment_utils as ma_utils


class MAEvaluator(core.Actor):
  """A recurrent multi-agent Evaluator."""

  _states: list[hk.LSTMState]

  def __init__(
      self,
      forward_fn,
      initial_state_fn,
      n_agents: int,
      n_params: int,
      variable_client: variable_utils.VariableClient,
      rng: hk.PRNGSequence,
  ):
    self.forward_fn = forward_fn
    self.initial_state_fn = initial_state_fn
    self.n_agents = n_agents
    self.n_params = n_params
    self._rng = rng
    self._variable_client = variable_client

    def initialize_states(rng_sequence: hk.PRNGSequence,) -> list[hk.LSTMState]:
      """Initialize the recurrent states of the actor."""
      states = list()
      for _ in range(self.n_agents):
        states.append(initial_state_fn(next(rng_sequence)))
      return states

    self._initial_states = ma_utils.merge_data(initialize_states(self._rng))
    self._p_forward = jax.vmap(self.forward_fn)

  def select_action(self, observations):
    if self._states is None:
      self._states = self._initial_states
      self.update(True)
      self.loaded_params = self._params
      self.selected_params = jax.random.choice(
          next(self._rng), self.n_params, (self.n_agents,), replace=True)
      self.episode_params = ma_utils.select_idx(self.loaded_params,
                                                self.selected_params)

    observations = ma_utils.merge_data(observations)
    (logits, _), new_states = self._p_forward(self.episode_params, observations,
                                              self._states)
    actions = jax.random.categorical(next(self._rng), logits)

    self._states = new_states
    return jax.tree_util.tree_map(lambda a: [*a], actions)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._states = None

  def observe(self, action, next_timestep: dm_env.TimeStep):
    pass

  def update(self, wait: bool = True):
    self._variable_client.update(wait)

  @property
  def _params(self) -> hk.Params:
    return self._variable_client.params
