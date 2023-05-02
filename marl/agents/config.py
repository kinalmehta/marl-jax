"""Multi-agent RL config."""

import dataclasses
from typing import Optional, Union

from acme import types
from acme.adders import reverb as adders_reverb
import numpy as np
import optax


@dataclasses.dataclass
class MAConfig:
  """Configuration options for multi-agent RL."""

  memory_efficient: bool = True

  seed: int = 0
  discount: float = 0.99
  sequence_length: int = 40
  sequence_period: Optional[int] = None
  variable_update_period: int = 300

  # Environment Details
  n_agents: int = None

  # Optimization configuration
  use_parameter_sampling: bool = False

  # Optimizer configuration
  batch_size: int = 32
  learning_rate: Union[float, optax.Schedule] = 4e-4

  rmsprop_decay: float = 0.99
  rmsprop_eps: float = 1e-5
  rmsprop_momentum: float = 0
  rmsprop_init: float = 0.0

  max_gradient_norm: float = 2.0

  # PopArt configuration
  only_art = True
  step_size = 1e-5
  scale_lb = 1e-2
  scale_ub = 1e6

  # # Loss configuration.
  baseline_cost: float = 1.0
  entropy_cost: float = 0.003
  max_abs_reward: float = np.inf

  # Replay options
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  num_prefetch_threads: Optional[int] = None
  samples_per_insert: Optional[float] = 1.0
  max_queue_size: Union[int, types.Batches] = types.Batches(10)

  def __post_init__(self):
    # if isinstance(self.n_agents, int):  # config for a 11GB GPU
    #   max_data = 2 * 128 * 60
    #   while self.batch_size * self.sequence_length * self.n_agents > max_data:
    #     if self.batch_size > 32:
    #       self.batch_size //= 2
    #     elif self.sequence_length > 20:
    #       self.sequence_length //= 2
    #     else:
    #       break
    if isinstance(self.max_queue_size, types.Batches):
      self.max_queue_size *= self.batch_size
    assert (self.max_queue_size > self.batch_size + 1), """
        max_queue_size must be strictly larger than the batch size:
        - during the last step in an episode we might write 2 sequences to
          Reverb at once (that's how SequenceAdder works)
        - Reverb does insertion/sampling in multiple threads, so data is
          added asynchronously at unpredictable times. Therefore we need
          additional buffer size in order to avoid deadlocks."""
