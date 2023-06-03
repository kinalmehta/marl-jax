"""OPRE Builder."""

from collections.abc import Iterator
from typing import Any, Callable, Optional

from acme import core
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import optax
import reverb
import rlax

from marl import specs as ma_specs
from marl import types
from marl.agents.builder import MABuilder
from marl.agents.opre.config import OPREConfig
from marl.agents.opre.learning import OPRELearner
from marl.agents.opre.learning import OPRELearnerME
from marl.agents.opre.learning import PopArtOPRELearner
from marl.agents.opre.learning import PopArtOPRELearnerME
from marl.modules import popart_simple


class OPREBuilder(MABuilder):
  """OPRE Builder."""

  def __init__(
      self,
      config: OPREConfig,
      core_state_spec: hk.LSTMState,
      table_extension: Optional[Callable[[], Any]] = None,
  ):
    super().__init__(config, core_state_spec, table_extension)

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: types.RecurrentNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: ma_specs.MAEnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client

    optimizer = optax.chain(
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.rmsprop(
            self._config.learning_rate,
            decay=self._config.rmsprop_decay,
            eps=self._config.rmsprop_eps,
            initial_scale=self._config.rmsprop_init,
            momentum=self._config.rmsprop_momentum
            if self._config.rmsprop_momentum != 0 else None,
        ),
    )

    learner = OPRELearnerME if self._config.memory_efficient else OPRELearner
    return learner(
        network=networks,
        iterator=dataset,
        optimizer=optimizer,
        n_agents=self._config.n_agents,
        random_key=random_key,
        discount=self._config.discount,
        baseline_cost=self._config.baseline_cost,
        entropy_cost=self._config.entropy_cost,
        options_entropy_cost=self._config.options_entropy_cost,
        options_kl_cost=self._config.options_kl_cost,
        pg_mix=self._config.pg_mix,
        max_abs_reward=self._config.max_abs_reward,
        counter=counter,
        logger=logger_fn(label="learner"),
    )


class PopArtOPREBuilder(MABuilder):
  """OPRE Builder."""

  def __init__(
      self,
      config: OPREConfig,
      core_state_spec: hk.LSTMState,
      table_extension: Optional[Callable[[], Any]] = None,
  ):
    super().__init__(config, core_state_spec, table_extension)

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: types.RecurrentNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: ma_specs.MAEnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client

    optimizer = optax.chain(
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.rmsprop(
            self._config.learning_rate,
            decay=self._config.rmsprop_decay,
            eps=self._config.rmsprop_eps * 1000,
            initial_scale=self._config.rmsprop_init,
            momentum=self._config.rmsprop_momentum
            if self._config.rmsprop_momentum != 0 else None,
        ),
    )

    def _popart(axis):
      init_fn, update_fn = rlax.popart(
          num_outputs=1,
          step_size=self._config.step_size,
          scale_lb=self._config.scale_lb,
          scale_ub=self._config.scale_ub,
          axis_name=axis,
      )
      return types.PopArtLayer(init_fn=init_fn, update_fn=update_fn)

    def _art(axis):
      init_fn, update_fn = popart_simple(
          num_outputs=1,
          step_size=self._config.step_size,
          scale_lb=self._config.scale_lb,
          scale_ub=self._config.scale_ub,
          axis_name=axis,
      )
      return types.PopArtLayer(init_fn=init_fn, update_fn=update_fn)

    learner = PopArtOPRELearnerME if self._config.memory_efficient else PopArtOPRELearner
    return learner(
        network=networks,
        popart=(_art if self._config.only_art else _popart,
                self._config.only_art),
        iterator=dataset,
        optimizer=optimizer,
        n_agents=self._config.n_agents,
        random_key=random_key,
        discount=self._config.discount,
        baseline_cost=self._config.baseline_cost,
        entropy_cost=self._config.entropy_cost,
        options_entropy_cost=self._config.options_entropy_cost,
        options_kl_cost=self._config.options_kl_cost,
        pg_mix=self._config.pg_mix,
        max_abs_reward=self._config.max_abs_reward,
        counter=counter,
        logger=logger_fn(label="learner"),
    )
