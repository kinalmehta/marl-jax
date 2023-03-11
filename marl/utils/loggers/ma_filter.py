"""Wrapper for logger to handle multi-agent logging."""

from acme.utils.loggers import base


class MAFilter(base.Logger):
  """Logger which writes to another logger, filtering any list values to be mapped to agent_{idx}."""

  def __init__(self, to: base.Logger):
    """Initializes the logger.

        Args:
          to: A `Logger` object to which the current object will forward its results
            when `write` is called.
        """
    self._to = to

  def write(self, values: base.LoggingData):
    new_values = dict()
    for key, value in values.items():
      if type(value) in [list, tuple] or hasattr(value, "__len__"):
        for idx, val in enumerate(value):
          new_values[f"agent_{idx}/{key}"] = val
      else:
        new_values[key] = value
    self._to.write(new_values)

  def close(self):
    self._to.close()
