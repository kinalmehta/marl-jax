"""OPRE config."""
import dataclasses

from marl.agents.config import MAConfig

# # Meltingpot v2 config
# @dataclasses.dataclass
# class OPREConfig(MAConfig):
#   """Configuration options for OPRE."""

#   learning_rate = 8e-4

#   # Loss configuration.
#   pg_mix: bool = False
#   baseline_cost: float = 1.0
#   entropy_cost: float = 0.003
#   options_entropy_cost: float = 0.04
#   options_kl_cost: float = 0.01

#   num_options: int = 8


# Meltingpot v1 config
@dataclasses.dataclass
class OPREConfig(MAConfig):
  """Configuration options for OPRE."""

  max_gradient_norm: float = 40.0
  learning_rate = 4e-4

  # Loss configuration.
  pg_mix: bool = False
  baseline_cost: float = 0.5
  entropy_cost: float = 0.003
  options_entropy_cost: float = 0.01
  options_kl_cost: float = 0.01

  num_options: int = 8
