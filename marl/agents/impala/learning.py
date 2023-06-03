"""Learner for the MAIMPALA actor-critic agent."""

from collections.abc import Iterator
from collections.abc import Sequence
import functools
from typing import Optional

from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import jax
import numpy as np
import optax
import reverb

from marl import types
from marl.agents import learning
from marl.agents import learning_memory_efficient
from marl.agents.impala.loss import batched_art_impala_loss
from marl.agents.impala.loss import batched_popart_impala_loss
from marl.agents.impala.loss import impala_loss

_PMAP_AXIS_NAME = "data"


class IMPALALearner(learning.MALearner):

  def __init__(
      self,
      network: types.RecurrentNetworks,
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      n_agents: int,
      random_key: networks_lib.PRNGKey,
      discount: float = 0.99,
      baseline_cost: float = 1.0,
      entropy_cost: float = 0.0,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    loss_fn = functools.partial(
        impala_loss,
        discount=discount,
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost,
        max_abs_reward=max_abs_reward,
    )
    super().__init__(network, iterator, optimizer, n_agents, random_key,
                     loss_fn, counter, logger, devices)


class PopArtIMPALALearner(learning.MALearnerPopArt):

  def __init__(
      self,
      network: types.RecurrentNetworks,
      popart: types.PopArtLayer,
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      n_agents: int,
      random_key: networks_lib.PRNGKey,
      discount: float = 0.99,
      baseline_cost: float = 1.0,
      entropy_cost: float = 0.0,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    loss_fn = (
        functools.partial(
            batched_art_impala_loss,
            discount=discount,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            max_abs_reward=max_abs_reward,
        ) if popart[1] else functools.partial(
            batched_popart_impala_loss,
            discount=discount,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            max_abs_reward=max_abs_reward,
        ))
    super().__init__(network, popart[0], iterator, optimizer, n_agents,
                     random_key, loss_fn, counter, logger, devices)


class IMPALALearnerME(learning_memory_efficient.MALearner):

  def __init__(
      self,
      network: types.RecurrentNetworks,
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      n_agents: int,
      random_key: networks_lib.PRNGKey,
      discount: float = 0.99,
      baseline_cost: float = 1.0,
      entropy_cost: float = 0.0,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    loss_fn = functools.partial(
        impala_loss,
        discount=discount,
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost,
        max_abs_reward=max_abs_reward,
    )
    super().__init__(network, iterator, optimizer, n_agents, random_key,
                     loss_fn, counter, logger, devices)


class PopArtIMPALALearnerME(learning_memory_efficient.MALearnerPopArt):

  def __init__(
      self,
      network: types.RecurrentNetworks,
      popart: types.PopArtLayer,
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      n_agents: int,
      random_key: networks_lib.PRNGKey,
      discount: float = 0.99,
      baseline_cost: float = 1.0,
      entropy_cost: float = 0.0,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    loss_fn = (
        functools.partial(
            batched_art_impala_loss,
            discount=discount,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            max_abs_reward=max_abs_reward,
        ) if popart[1] else functools.partial(
            batched_popart_impala_loss,
            discount=discount,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            max_abs_reward=max_abs_reward,
        ))
    super().__init__(network, popart[0], iterator, optimizer, n_agents,
                     random_key, loss_fn, counter, logger, devices)
