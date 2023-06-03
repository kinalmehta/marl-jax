"""
Defines Inference Server class used for centralised inference.
Ref: https://github.com/deepmind/acme/blob/master/acme/jax/inference_server.py
"""

from collections.abc import Sequence
import dataclasses
import datetime
import threading
from typing import Any, Callable, Generic, Optional, TypeVar

import acme
from acme.jax import variable_utils
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import launchpad as lp

from marl.utils import experiment_utils


@dataclasses.dataclass
class InferenceServerConfig:
  """Configuration options for centralised inference.

    Attributes:
      batch_size: How many elements to batch together per single inference call.
          Auto-computed when not specified.
      update_period: Frequency of updating variables from the variable source.
          It is passed to VariableClient. Auto-computed when not specified.
      timeout: Time after which incomplete batch is executed (batch is padded,
          so there batch handler is always called with batch_size elements).
          By default timeout is effectively disabled (set to 30 days).
    """

  batch_size: Optional[int] = None
  update_period: Optional[int] = None
  timeout: datetime.timedelta = datetime.timedelta(days=30)


InferenceServerHandler = TypeVar("InferenceServerHandler")


class InferenceServer(Generic[InferenceServerHandler]):
  """Centralised, batched inference server."""

  def __init__(
      self,
      handler: InferenceServerHandler,
      variable_source: acme.VariableSource,
      devices: Sequence[jax.xla.Device],
      config: InferenceServerConfig,
  ):
    """Constructs an inference server object.

        Args:
          handler: A callable or a mapping of callables to be exposed
            through the inference server.
          variable_source: Source of variables
          devices: Devices used for executing handlers. All devices are used in
            parallel.
          config: Inference Server configuration.
        """
    self._variable_source = variable_source
    self._variable_client = None
    self._keys = []
    self._devices = devices
    self._config = config
    self._call_cnt = 0
    self._device_params = [None] * len(self._devices)
    self._device_params_ids = [None] * len(self._devices)
    self._mutex = threading.Lock()
    self._handler = jax.tree_map(self._build_handler, handler, is_leaf=callable)

  @property
  def handler(self) -> InferenceServerHandler:
    return self._handler

  def _dereference_params(self, arg):
    """Replaces VariableReferences with their corresponding param values."""

    if not isinstance(arg, variable_utils.VariableReference):
      # All arguments but VariableReference are returned without modifications.
      arg = tree_stack(arg)
      return arg

    # Due to batching dimension we take the first element.
    variable_name = arg.variable_name[0]

    if variable_name not in self._keys:
      # Create a new VariableClient which also serves new variables.
      self._keys.append(variable_name)
      self._variable_client = variable_utils.VariableClient(
          client=self._variable_source,
          key=self._keys,
          update_period=self._config.update_period)

    params = self._variable_client.params
    device_idx = self._call_cnt % len(self._devices)
    # Select device via round robin, and update its params if they changed.
    if self._device_params_ids[device_idx] != id(params):
      self._device_params_ids[device_idx] = id(params)
      self._device_params[device_idx] = jax.device_put(
          params, self._devices[device_idx])

    # Return the params that are located on the chosen device.
    device_params = self._device_params[device_idx]
    if len(self._keys) == 1:
      return device_params
    return device_params[self._keys.index(variable_name)]

  def _build_handler(self, handler: Callable[..., Any]) -> Callable[..., Any]:
    """Builds a batched handler for a given callable handler and its name."""

    def dereference_params_and_call_handler(*args, **kwargs):
      with self._mutex:
        # Dereference args corresponding to params, leaving others unchanged.
        args_with_dereferenced_params = [
            self._dereference_params(arg) for arg in args
        ]
        kwargs_with_dereferenced_params = {
            key: self._dereference_params(value)
            for key, value in kwargs.items()
        }
        self._call_cnt += 1

        # Maybe update params, depending on client configuration.
        if self._variable_client is not None:
          self._variable_client.update()

      op = handler(*args_with_dereferenced_params,
                   **kwargs_with_dereferenced_params)

      op = experiment_utils.split_data(op)
      return op

    return lp.batched_handler(
        batch_size=self._config.batch_size,
        timeout=self._config.timeout,
        pad_batch=True,
        max_parallelism=2 * len(self._devices),
    )(
        dereference_params_and_call_handler)


def _postprocess_data(data):
  if type(data) is dict:
    data = {k: _postprocess_data(v) for k, v in data.items()}
    return data
  elif type(data) is list:
    return jnp.array(data)
  elif type(data) is hk.LSTMState:
    return hk.LSTMState(
        hidden=jnp.array(data.hidden), cell=jnp.array(data.cell))
  else:
    raise ValueError(f"Unsupported data type: {type(data)}")


postprocess_data = jax.jit(_postprocess_data)


def _tree_stack(trees):
  """
  Stack the leaves of a tree of arrays. Python Lists are considered leaves.
  e.g. 
    a = {'a': [jnp.ndarray, jnp.ndarray], 'b': [jnp.ndarray, jnp.ndarray]}
    tree_stack(a) = {'a': jnp.ndarray, 'b': jnp.ndarray} the dim 0 of the new leaves is 2
  """
  leaves, treedef = tree_flatten(trees, is_leaf=lambda x: isinstance(x, list))
  result_leaves = [jnp.array(leaf) for leaf in leaves]
  return treedef.unflatten(result_leaves)


tree_stack = jax.jit(_tree_stack)
