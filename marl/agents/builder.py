"""Multi-agent Builder."""

from collections.abc import Iterator
from typing import Any, Callable, Optional

import acme
from acme import adders
from acme import core
from acme.adders import reverb as reverb_adders
from acme.agents.jax import builders
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
import haiku as hk
import jax
import jax.numpy as jnp
import reverb

from marl import specs as ma_specs
from marl import types
from marl.agents.acting import MAActor
from marl.agents.config import MAConfig
from marl.agents.evaluating import MAEvaluator
from marl.utils import experiment_utils as ma_utils


class MABuilder(builders.ActorLearnerBuilder[types.RecurrentNetworks,
                                             types.RecurrentNetworks,
                                             reverb.ReplaySample,]):
  """Multi-agent Builder."""

  def __init__(
      self,
      config: MAConfig,
      core_state_spec: hk.LSTMState,
      table_extension: Optional[Callable[[], Any]] = None,
  ):
    """Creates an IMPALA learner."""
    self._config = config
    core_state_spec = {
        "hidden": core_state_spec.hidden,
        "cell": core_state_spec.cell
    }
    self._core_state_spec = core_state_spec
    self._sequence_length = self._config.sequence_length
    self._table_extension = table_extension

  def make_replay_tables(
      self,
      environment_spec: ma_specs.MAEnvironmentSpec,
      policy: types.RecurrentNetworks,
  ) -> list[reverb.Table]:
    """The queue; use XData or INFO log."""
    del policy
    env_specs = environment_spec.get_agent_environment_specs()
    num_actions = environment_spec.get_single_agent_environment_specs(
    ).actions.num_values
    num_agents = environment_spec.num_agents
    core_state = ma_utils.merge_data([self._core_state_spec] * num_agents)
    extra_specs = {
        "logits": jnp.ones(shape=(
            num_agents,
            num_actions,
        ), dtype=jnp.float32),
        "core_state": core_state,
    }

    signature = reverb_adders.SequenceAdder.signature(
        env_specs, extra_specs, sequence_length=self._config.sequence_length)

    # Maybe create rate limiter.
    # Setting the samples_per_insert ratio less than the default of 1.0, allows
    # the agent to drop data for the benefit of using data from most up-to-date
    # policies to compute its learner updates.
    samples_per_insert = self._config.samples_per_insert
    if samples_per_insert:
      if samples_per_insert > 1.0 or samples_per_insert <= 0.0:
        raise ValueError(
            "Impala requires a samples_per_insert ratio in the range (0, 1],"
            f" but received {samples_per_insert}.")
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          samples_per_insert=samples_per_insert,
          min_size_to_sample=1,
          error_buffer=self._config.batch_size,
      )
    else:
      limiter = reverb.rate_limiters.MinSize(1)

    table_extensions = []
    if self._table_extension is not None:
      table_extensions = [self._table_extension()]
    queue = reverb.Table(
        name=self._config.replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_queue_size,
        max_times_sampled=1,
        rate_limiter=limiter,
        extensions=table_extensions,
        signature=signature,
    )
    return [queue]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset."""
    batch_size_per_learner = self._config.batch_size // jax.process_count()
    batch_size_per_device, ragged = divmod(self._config.batch_size,
                                           jax.device_count())
    if ragged:
      raise ValueError(
          "Learner batch size must be divisible by total number of devices!")

    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=batch_size_per_device,
        num_parallel_calls=None,
        max_in_flight_samples_per_worker=2 * batch_size_per_learner,
    )
    # return utils.device_put(dataset.as_numpy_iterator(), jax.local_devices()[0])
    return dataset.as_numpy_iterator(
    ) if self._config.memory_efficient else utils.multi_device_put(
        dataset.as_numpy_iterator(), jax.local_devices())

  def make_adder(
      self,
      replay_client: reverb.Client,
      environment_spec: Optional[ma_specs.MAEnvironmentSpec],
      policy: Optional[types.RecurrentNetworks],
  ) -> Optional[adders.Adder]:
    """Creates an adder which handles observations."""
    del environment_spec, policy
    # Note that the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    return reverb_adders.SequenceAdder(
        client=replay_client,
        priority_fns={self._config.replay_table_name: None},
        period=self._config.sequence_period or (self._sequence_length - 1),
        sequence_length=self._sequence_length,
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: types.RecurrentNetworks,
      environment_spec: ma_specs.MAEnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec

    variable_client = variable_utils.VariableClient(
        client=variable_source,
        key="network",
        update_period=self._config.variable_update_period,
        device="cpu",
    )

    return MAActor(
        forward_fn=policy.forward_fn,
        initial_state_fn=policy.initial_state_fn,
        n_agents=self._config.n_agents,
        rng=hk.PRNGSequence(random_key),
        variable_client=variable_client,
        adder=adder,
    )

  def make_evaluator(
      self,
      random_key: networks_lib.PRNGKey,
      policy: types.RecurrentNetworks,
      n_agents: int,
      variable_source: Optional[core.VariableSource] = None,
  ):

    variable_client = variable_utils.VariableClient(
        client=variable_source,
        key="network",
        update_period=1,  # using 1 as update is called only at the start of episode
        device="cpu",
    )

    return MAEvaluator(
        forward_fn=policy.forward_fn,
        initial_state_fn=policy.initial_state_fn,
        n_agents=n_agents,
        n_params=self._config.n_agents,
        rng=hk.PRNGSequence(random_key),
        variable_client=variable_client,
    )

  def make_policy(
      self,
      networks: types.RecurrentNetworks,
      environment_spec: ma_specs.MAEnvironmentSpec,
      evaluation: bool = False,
  ) -> types.RecurrentNetworks:
    del environment_spec, evaluation
    return networks
