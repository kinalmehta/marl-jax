"""IMPALA config."""

import dataclasses

from marl.agents.config import MAConfig


@dataclasses.dataclass
class IMPALAConfig(MAConfig):
  """Configuration options for MAIMPALA."""

  # Loss configuration.
  baseline_cost: float = 0.5
  entropy_cost: float = 0.003
