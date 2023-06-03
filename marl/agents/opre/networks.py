from acme.jax import utils
from acme.specs import EnvironmentSpec
import haiku as hk
import jax
import jax.numpy as jnp

from marl.agents.networks import make_haiku_networks
from marl.agents.networks import make_haiku_networks_2

# Useful type aliases
Images = jnp.ndarray

batch_concat = utils.batch_concat
add_batch_dim = utils.add_batch_dim


def make_network(environment_spec: EnvironmentSpec,
                 feature_extractor: hk.Module,
                 recurrent_dim: int = 128,
                 num_options: int = 16):

  def forward_fn(inputs, state: hk.LSTMState):
    model = OPRENetwork(
        num_actions=environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        num_options=num_options,
        feature_extractor=feature_extractor,
    )
    return model(inputs, state)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = OPRENetwork(
        num_actions=environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        num_options=num_options,
        feature_extractor=feature_extractor,
    )
    return model.unroll(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = OPRENetwork(
        num_actions=environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        num_options=num_options,
        feature_extractor=feature_extractor,
    )
    return model.initial_state(batch_size)

  return make_haiku_networks(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
  )


def make_network_2(environment_spec: EnvironmentSpec,
                   feature_extractor: hk.Module,
                   recurrent_dim: int = 128,
                   num_options: int = 16):

  def forward_fn(inputs, state: hk.LSTMState):
    model = OPRENetwork(
        num_actions=environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        num_options=num_options,
        feature_extractor=feature_extractor,
    )
    return model(inputs, state)

  def unroll_fn(inputs, state: hk.LSTMState):
    model = OPRENetwork(
        num_actions=environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        num_options=num_options,
        feature_extractor=feature_extractor,
    )
    return model.unroll(inputs, state)

  def initial_state_fn(batch_size=None) -> hk.LSTMState:
    model = OPRENetwork(
        num_actions=environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        num_options=num_options,
        feature_extractor=feature_extractor,
    )
    return model.initial_state(batch_size)

  def critic_fn(feat, options):
    model = OPRENetwork(
        environment_spec.actions.num_values,
        recurrent_dim=recurrent_dim,
        num_options=num_options,
        feature_extractor=feature_extractor,
    )
    return model.critic(feat, options)

  return make_haiku_networks_2(
      env_spec=environment_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      critic_fn=critic_fn,
  )


class OPRENetwork(hk.RNNCore):
  """Combining Embed and option prediction as a single inference network."""

  def __init__(self, num_actions, recurrent_dim, num_options,
               feature_extractor):
    super().__init__(name="opre_network")
    self.num_options = num_options
    self.num_actions = num_actions
    self._embed = feature_extractor(num_actions)
    self._recurrent = hk.LSTM(recurrent_dim)
    self._p_options_head = hk.Linear(num_options)
    self._q_options_head = hk.Linear(num_options)

    self._policy_layer_options = hk.Linear(num_options * num_actions)
    self._value_layer_options = hk.Linear(num_options, name="value_layer")

  def __call__(self, inputs, state: hk.LSTMState):
    feat = self._embed(inputs)
    feat, new_state = self._recurrent(feat, state)

    p_options = self._p_options_head(feat)
    p_options = jax.nn.softmax(p_options, axis=-1)
    assert len(p_options.shape) == 1, "bad options shape"  # (num_options,)

    options_action_probs = self._policy_layer_options(feat)
    options_action_probs = options_action_probs.reshape(
        (-1, self.num_options, self.num_actions))
    options_action_probs = options_action_probs.squeeze()
    values = self._value_layer_options(feat)

    assert len(options_action_probs.shape
              ) == 2, "bad action probs shape"  # (num_options, num_actions)
    final_action_probs = (options_action_probs *
                          jnp.expand_dims(p_options, axis=-1)).sum(-2)

    assert len(values.shape) == 1, "bad value shape"  # (num_options,)
    final_values = values @ p_options

    return (final_action_probs, final_values), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._recurrent.initial_state(batch_size)

  def unroll(self, inputs, state: hk.LSTMState):
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    feat = self._embed(inputs)

    # uncomment below line to fix datatype for dynamic_unroll with reverb sampled data
    # state = hk.LSTMState(state.hidden, state.cell)
    feat, new_states = hk.static_unroll(self._recurrent, feat, state)
    p_options = self._p_options_head(feat)  # (B, num_options)
    p_options = jax.nn.softmax(p_options, axis=-1)  # (B, num_options)

    options_action_probs = self._policy_layer_options(feat)
    options_action_probs = options_action_probs.reshape(
        (-1, self.num_options, self.num_actions))
    options_action_probs = options_action_probs.squeeze()
    options_values = self._value_layer_options(feat)

    logits = (options_action_probs * jnp.expand_dims(p_options, axis=-1)).sum(
        -2)  # (B, num_actions)
    values = (options_values * p_options).sum(-1)  # B

    auxiliary_state = hk.BatchApply(self._embed)(
        inputs["observation"]["global"])  # B, num_agents, embedding_dim
    # auxiliary_state = jnp.reshape(auxiliary_state,
    #                               (auxiliary_state.shape[0], -1))
    auxiliary_state = jnp.sum(auxiliary_state, axis=1)  # B, embedding_dim

    q_options = self._q_options_head(auxiliary_state)  # B, num_options
    q_options = jax.nn.softmax(q_options, axis=-1)  # B, num_options
    q_logits = (
        options_action_probs *
        jnp.expand_dims(jax.lax.stop_gradient(q_options), axis=-1)).sum(-2)
    q_values = (options_values * q_options).sum(-1)

    return (
        logits,
        values,
        feat,
        p_options,
        q_logits,
        q_values,
        q_options,
    ), new_states

  def critic(self, feat, options):
    options_values = self._value_layer_options(feat)
    q_values = (options_values * options).sum(-1)
    return q_values


class EtaC(hk.Module):
  """Final mixture model which predicts action probabilities and values for z models"""

  def __init__(self, num_options, num_actions):
    super().__init__(name="eta_c")
    self.num_options = num_options
    self.num_actions = num_actions
    self._policy_layer = hk.Linear(num_options * num_actions)
    self._value_layer = hk.Linear(num_options, name="value_layer")

  def __call__(self, inputs: jnp.array):
    action_probs = self._policy_layer(inputs)
    action_probs = action_probs.reshape(
        (-1, self.num_options, self.num_actions))
    action_probs = action_probs.squeeze()
    value_preds = self._value_layer(inputs)
    return action_probs, value_preds
