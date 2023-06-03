"""Utility definitions for MARL experiments."""

from collections.abc import Mapping
from typing import Any, Optional, Union

from acme.utils import loggers
import jax
import jax.numpy as jnp

from marl.utils import loggers as marl_loggers


def make_experiment_logger(
    label: str,
    steps_key: Optional[str] = None,
    task_instance: int = 0,
    log_dir: Optional[str] = "~/marl-jax",
    use_tb: bool = True,
    use_wandb: bool = False,
    wandb_config: Mapping[str, Any] = {},
) -> loggers.Logger:
  # del task_instance
  if task_instance == 0 and (label in [
      "actor", "learner"
  ]) and use_wandb:  # or "evaluator" in label
    use_wandb = True
    print("Enable wandb for learner, first actor.")
  else:
    use_wandb = False
  if steps_key is None:
    steps_key = f"{label}_steps"
  return marl_loggers.make_default_logger(
      label=label,
      log_dir=log_dir,
      use_tb=use_tb,
      use_wandb=use_wandb,
      wandb_config=wandb_config,
      steps_key=steps_key,
  )


@jax.jit
def transpose(data, axis=(0, 1)):

  def _transpose(x):
    assert len(
        x.shape) >= len(axis), f"Invalid transpose axis {x.shape}, {axis}"
    return jnp.transpose(x, axis + tuple(range(len(axis), len(x.shape))))

  return jax.tree_util.tree_map(_transpose, data)


@jax.jit
def concat_data(data, axis=0):
  """
    Concatenate data from multiple pmap processes into a single array.
    """
  return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=axis), *data)


@jax.jit
def merge_data(data, axis=0):
  """
    Merge data from multiple agents into a single array.
    A new dimension to added to the original array at location 0 equal to the number of agents.
    """
  return jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=axis), *data)


@jax.jit
def split_data(data):
  return jax.tree_util.tree_map(lambda x: list(x), data)


@jax.jit
def merge_dimensions(data, start=0, end=1):

  def merge_dims(x):
    shp = x.shape
    return jnp.reshape(x, shp[0:start] + (-1,) + shp[end + 1:])

  return jax.tree_util.tree_map(merge_dims, data)


def slice_data(data, i: int, n_devices: int):
  """
    Slice the merged data based on the available devices.
    """
  return jax.tree_util.tree_map(lambda s: s[i:i + n_devices], data)


@jax.jit
def select_idx(data, i: Union[int, list]):
  """
    Select the i-th element of the array.
    """
  return jax.tree_util.tree_map(lambda s: s[i], data)
