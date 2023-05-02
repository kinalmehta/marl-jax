"""Multi agent learner implementation."""

from collections.abc import Iterator
from collections.abc import Sequence
import time
from typing import Callable, Optional

from absl import logging
import acme
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb

from marl import types
from marl.utils import experiment_utils as ma_utils

_PMAP_AXIS_NAME = "data"


class MALearner(acme.Learner):

  def __init__(
      self,
      network: types.RecurrentNetworks,
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      n_agents: int,
      random_key: networks_lib.PRNGKey,
      loss_fn: Callable,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    local_devices = jax.local_devices()
    process_id = jax.process_index()
    logging.info("Learner process id: %s. Devices passed: %s", process_id,
                 devices)
    logging.info(
        "Learner process id: %s. Local devices from JAX API: %s",
        process_id,
        local_devices,
    )
    self._devices = devices or local_devices
    self._local_devices = [d for d in self._devices if d in local_devices]

    self._iterator = iterator

    self.network = network
    self.optimizer = optimizer
    self.n_agents = n_agents
    self.n_devices = len(self._local_devices)
    self._rng = hk.PRNGSequence(random_key)

    def make_initial_state(key: jnp.ndarray) -> types.TrainingState:
      """Initialises the training state (parameters and optimiser state)."""
      key, key_initial_state = jax.random.split(key)
      # Note: parameters do not depend on the batch size, so initial_state below
      # does not need a batch dimension.
      initial_state = network.initial_state_fn(key_initial_state)

      # Initialise main model and auxiliary model parameters
      initial_params = network.unroll_init_fn(key, initial_state)

      initial_opt_state = optimizer.init(initial_params)
      return (
          types.TrainingState(
              params=initial_params,
              opt_state=initial_opt_state,
          ),
          key,
      )

    # Initialize Params for Each Network
    def make_initial_states(key: jnp.ndarray) -> list[types.TrainingState]:
      states = list()
      for _ in range(self.n_agents):
        agent_state, key = make_initial_state(key)
        states.append(agent_state)
      return states

    @jax.jit
    def sgd_step(
        state: types.TrainingState, sample: types.TrainingData
    ) -> tuple[types.TrainingState, dict[str, jnp.ndarray]]:
      """Computes an SGD step, returning new state and metrics for logging."""

      # Compute gradients.
      grad_fn = jax.grad(self._loss_fn, has_aux=True)
      gradients, metrics = grad_fn(state.params, sample)

      # Apply updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      metrics.update({
          "param_norm": optax.global_norm(new_params),
          "param_updates_norm": optax.global_norm(updates),
      })

      new_state = types.TrainingState(
          params=new_params,
          opt_state=new_opt_state,
      )

      return new_state, metrics

    # Initialise training state (parameters and optimiser state).
    self._states = make_initial_states(next(self._rng))
    self._combined_states = ma_utils.merge_data(self._states)

    self._loss_fn = loss_fn(network=network)

    self._sgd_step = jax.pmap(
        sgd_step, axis_name=_PMAP_AXIS_NAME, in_axes=(0, 2))

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        "learner", steps_key=self._counter.get_steps_key())

    # Initialize prediction function and initial LSTM states
    self._predict_fn = jax.pmap(
        jax.vmap(network.forward_fn, in_axes=(0, 1, 1), out_axes=(1, 1)),
        devices=self._local_devices)

  def _get_initial_lstm_states(self):

    def initialize_states(rng_sequence: hk.PRNGSequence,) -> list[hk.LSTMState]:
      """Initialize the recurrent states of the actor."""
      states = list()
      for _ in range(self.n_agents):
        states.append(self.network.initial_state_fn(next(rng_sequence)))
      return states

    _initial_lstm_states = ma_utils.merge_data(initialize_states(self._rng))
    return _initial_lstm_states

  def _get_actions(self, observations, lstm_states):
    """Returns actions for each agent."""
    (logits,
     _), updated_lstm_states = self._predict_fn(self._combined_states.params,
                                                observations, lstm_states)
    actions = jax.random.categorical(next(self._rng), logits)
    return actions, logits, updated_lstm_states

  def step(self):
    """Does a step of SGD and logs the results."""
    samples = next(self._iterator)

    samples = samples.data
    samples = types.TrainingData(
        observation=samples.observation,
        action=samples.action,
        reward=samples.reward,
        discount=samples.discount,
        extras=samples.extras,
    )

    self._step_on_data(samples)

  def _step_on_data(self, samples):
    # Do a batch of SGD.
    start = time.time()

    new_states = []
    results = []
    for i in range(0, self.n_agents, self.n_devices):
      cur_state = ma_utils.slice_data(self._combined_states, i, self.n_devices)
      cur_sample = types.TrainingData(
          observation=slice_data_2(samples.observation, i, self.n_devices),
          action=slice_data_2(samples.action, i, self.n_devices),
          reward=slice_data_2(samples.reward, i, self.n_devices),
          discount=slice_data_2(samples.discount, i, self.n_devices),
          extras=slice_data_2(samples.extras, i, self.n_devices),
      )

      new_state, result = self._sgd_step(cur_state, cur_sample)

      new_states.append(jax.device_get(new_state))
      results.append(jax.device_get(result))

    self._combined_states = ma_utils.concat_data(new_states)
    results = ma_utils.concat_data(results, axis=None)

    # Update our counts and record them.
    counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

    # Shuffle the parameters every 1 learner steps
    if counts["learner_steps"] % 1 == 0:
      selected_order = jax.random.choice(
          next(self._rng), self.n_agents, (self.n_agents,), replace=False)
      # converting the selected_order to numpy as the parameters are also in numpy
      selected_order = jax.device_get(selected_order)

      shuffle_state = self.save()
      shuffle_state = ma_utils.select_idx(shuffle_state, selected_order)
      self.restore(shuffle_state)

    # Maybe write logs.
    self._logger.write({**results, **counts})

  def get_variables(self, names: Sequence[str]) -> list[networks_lib.Params]:
    # Return first replica of parameters.
    return [self._combined_states.params]

  def save(self) -> types.TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return self._combined_states

  def restore(self, state: types.TrainingState):
    self._combined_states = state


class MALearnerPopArt(MALearner):

  def __init__(
      self,
      network: types.RecurrentNetworks,
      popart: types.PopArtLayer,
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      n_agents: int,
      random_key: networks_lib.PRNGKey,
      loss_fn: Callable,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    local_devices = jax.local_devices()
    process_id = jax.process_index()
    logging.info("Learner process id: %s. Devices passed: %s", process_id,
                 devices)
    logging.info(
        "Learner process id: %s. Local devices from JAX API: %s",
        process_id,
        local_devices,
    )
    self._devices = devices or local_devices
    self._local_devices = [d for d in self._devices if d in local_devices]

    self._iterator = iterator

    self.network = network
    popart = popart(_PMAP_AXIS_NAME)
    self.optimizer = optimizer
    self.n_agents = n_agents
    self.n_devices = len(self._local_devices)
    self._rng = hk.PRNGSequence(random_key)

    def make_initial_state(key: jnp.ndarray) -> types.PopArtTrainingState:
      """Initialises the training state (parameters and optimiser state)."""
      key, key_initial_state = jax.random.split(key)
      # Note: parameters do not depend on the batch size, so initial_state below
      # does not need a batch dimension.
      initial_state = network.initial_state_fn(key_initial_state)

      # Initialise main model and auxiliary model parameters
      initial_params = network.unroll_init_fn(key, initial_state)

      initial_opt_state = optimizer.init(initial_params)
      return (
          types.PopArtTrainingState(
              params=initial_params,
              opt_state=initial_opt_state,
              popart_state=popart.init_fn(),
          ),
          key,
      )

    # Initialize Params for Each Network
    def make_initial_states(
        key: jnp.ndarray) -> list[types.PopArtTrainingState]:
      states = list()
      for _ in range(self.n_agents):
        agent_state, key = make_initial_state(key)
        states.append(agent_state)
      return states

    @jax.jit
    def sgd_step(
        state: types.PopArtTrainingState, sample: types.TrainingData
    ) -> tuple[types.PopArtTrainingState, dict[str, jnp.ndarray]]:
      """Computes an SGD step, returning new state and metrics for logging."""

      # Compute gradients.
      grad_fn = jax.grad(self._loss_fn, has_aux=True)
      gradients, (new_popart_state, metrics) = grad_fn(state.params,
                                                       state.popart_state,
                                                       sample)

      # Apply updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      metrics.update({
          "param_norm": optax.global_norm(new_params),
          "param_updates_norm": optax.global_norm(updates),
      })

      new_state = types.PopArtTrainingState(
          params=new_params,
          opt_state=new_opt_state,
          popart_state=new_popart_state,
      )

      return new_state, metrics

    # Initialise training state (parameters and optimiser state).
    self._states = make_initial_states(next(self._rng))
    self._combined_states = ma_utils.merge_data(self._states)

    self._loss_fn = loss_fn(network=network, popart_update_fn=popart.update_fn)

    self._sgd_step = jax.pmap(
        sgd_step, axis_name=_PMAP_AXIS_NAME, in_axes=(0, 2))

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        "learner", steps_key=self._counter.get_steps_key())

    # Initialize prediction function and initial LSTM states
    self._predict_fn = jax.pmap(
        jax.vmap(network.forward_fn, in_axes=(0, 1, 1), out_axes=(1, 1)),
        devices=self._local_devices)


def slice_data_2(data, i: int, n_devices: int):
  """
    Slice the merged data on the (agent's index) index 2 based on the available devices.
    """
  return jax.tree_util.tree_map(lambda x: x[:, :, i:i + n_devices], data)
