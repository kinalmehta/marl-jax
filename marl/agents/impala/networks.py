from acme.jax import utils
from acme.specs import EnvironmentSpec
import haiku as hk
import jax.numpy as jnp

from marl.agents.networks import make_haiku_networks
from marl.agents.networks import make_haiku_networks_2

# Useful type aliases
Images = jnp.ndarray

batch_concat = utils.batch_concat
add_batch_dim = utils.add_batch_dim


def make_network(environment_spec: EnvironmentSpec,
                 feature_extractor: hk.Module,
                 recurrent_dim: int = 128):

  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.unroll(inputs, state)

  return make_haiku_networks(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
  )


def make_network_2(environment_spec: EnvironmentSpec,
                   feature_extractor: hk.Module,
                   recurrent_dim: int = 128):

  def forward_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.unroll(inputs, state)

  def critic_fn(inputs):
    model = IMPALANetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        feature_extractor=feature_extractor)
    return model.critic(inputs)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )


class IMPALANetwork(hk.RNNCore):
  """Network architecture as described in MeltingPot paper"""

  def __init__(self, num_actions, recurrent_dim, feature_extractor):
    super().__init__(name="impala_network")
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._recurrent = hk.LSTM(recurrent_dim)
    self._policy_layer = hk.Linear(num_actions, name="policy")
    self._value_layer = hk.Linear(1, name="value_layer")

  def __call__(self, inputs, state: hk.LSTMState):
    op = self._embed(inputs)
    op, new_state = self._recurrent(op, state)
    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._recurrent.initial_state(batch_size)

  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    op = self._embed(inputs)

    # fix for dynamic_unroll with reverb sampled data
    # state = hk.LSTMState(state.hidden, state.cell)

    # unrolling the time dimension
    op, new_states = hk.static_unroll(self._recurrent, op,
                                      state)  # , return_all_states=True)

    logits = self._policy_layer(op)
    value = jnp.squeeze(self._value_layer(op), axis=-1)
    return (logits, value, op), new_states

  def critic(self, inputs):
    return jnp.squeeze(self._value_layer(inputs), axis=-1)
