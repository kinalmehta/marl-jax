"""Learner for the OPRE agent."""

import functools
from typing import Iterator, Optional, Sequence

import jax
import numpy as np
import optax
import reverb
from acme.jax import networks as networks_lib
from acme.utils import counting, loggers

from marl import types
from marl.agents import learning, learning_memory_efficient
from marl.agents.opre.loss import (
    batched_art_opre_loss,
    batched_opre_loss,
    batched_popart_opre_loss,
)

_PMAP_AXIS_NAME = "data"


class OPRELearner(learning.MALearner):

  def __init__(self,
               network: types.RecurrentNetworks,
               iterator: Iterator[reverb.ReplaySample],
               optimizer: optax.GradientTransformation,
               n_agents: int,
               random_key: networks_lib.PRNGKey,
               discount: float = 0.99,
               baseline_cost: float = 1.0,
               entropy_cost: float = 0.0,
               options_entropy_cost: float = 0.0,
               options_kl_cost: float = 0.0,
               pg_mix: bool = False,
               max_abs_reward: float = np.inf,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               devices: Optional[Sequence[jax.xla.Device]] = None):
    loss_fn = functools.partial(
        batched_opre_loss,
        discount=discount,
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost,
        options_entropy_cost=options_entropy_cost,
        options_kl_cost=options_kl_cost,
        pg_mix=pg_mix,
        max_abs_reward=max_abs_reward)
    super().__init__(network, iterator, optimizer, n_agents, random_key,
                     loss_fn, counter, logger, devices)


class PopArtOPRELearner(learning.MALearnerPopArt):

  def __init__(self,
               network: types.RecurrentNetworks,
               popart: types.PopArtLayer,
               iterator: Iterator[reverb.ReplaySample],
               optimizer: optax.GradientTransformation,
               n_agents: int,
               random_key: networks_lib.PRNGKey,
               discount: float = 0.99,
               baseline_cost: float = 1.0,
               entropy_cost: float = 0.0,
               options_entropy_cost: float = 0.0,
               options_kl_cost: float = 0.0,
               pg_mix: bool = False,
               max_abs_reward: float = np.inf,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               devices: Optional[Sequence[jax.xla.Device]] = None):
    loss_fn = functools.partial(
        batched_art_opre_loss,
        discount=discount,
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost,
        options_entropy_cost=options_entropy_cost,
        options_kl_cost=options_kl_cost,
        pg_mix=pg_mix,
        max_abs_reward=max_abs_reward) if popart[1] else functools.partial(
            batched_popart_opre_loss,
            discount=discount,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            options_entropy_cost=options_entropy_cost,
            options_kl_cost=options_kl_cost,
            pg_mix=pg_mix,
            max_abs_reward=max_abs_reward)
    super().__init__(network, popart[0], iterator, optimizer, n_agents,
                     random_key, loss_fn, counter, logger, devices)


class OPRELearnerME(learning_memory_efficient.MALearner):

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
      options_entropy_cost: float = 0.0,
      options_kl_cost: float = 0.0,
      pg_mix: bool = False,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    loss_fn = functools.partial(
        batched_opre_loss,
        discount=discount,
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost,
        options_entropy_cost=options_entropy_cost,
        options_kl_cost=options_kl_cost,
        pg_mix=pg_mix,
        max_abs_reward=max_abs_reward,
    )
    super().__init__(network, iterator, optimizer, n_agents, random_key,
                     loss_fn, counter, logger, devices)


class PopArtOPRELearnerME(learning_memory_efficient.MALearnerPopArt):

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
      options_entropy_cost: float = 0.0,
      options_kl_cost: float = 0.0,
      pg_mix: bool = False,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
  ):
    loss_fn = (
        functools.partial(
            batched_art_opre_loss,
            discount=discount,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            options_entropy_cost=options_entropy_cost,
            options_kl_cost=options_kl_cost,
            pg_mix=pg_mix,
            max_abs_reward=max_abs_reward,
        ) if popart[1] else functools.partial(
            batched_popart_opre_loss,
            discount=discount,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            options_entropy_cost=options_entropy_cost,
            options_kl_cost=options_kl_cost,
            pg_mix=pg_mix,
            max_abs_reward=max_abs_reward,
        ))
    super().__init__(network, popart[0], iterator, optimizer, n_agents,
                     random_key, loss_fn, counter, logger, devices)
