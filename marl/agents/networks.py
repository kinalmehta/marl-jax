from typing import Any

from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.specs import EnvironmentSpec
import haiku as hk
import jax
import jax.numpy as jnp

from marl import types

# Useful type aliases
Images = jnp.ndarray


def make_haiku_networks(
    env_spec: EnvironmentSpec,
    forward_fn: Any,
    initial_state_fn: Any,
    unroll_fn: Any,
) -> types.RecurrentNetworks[types.RecurrentState]:
  """Builds functional network from recurrent model definitions."""
  # Make networks purely functional.
  forward_hk = hk.without_apply_rng(hk.transform(forward_fn))
  initial_state_hk = hk.without_apply_rng(hk.transform(initial_state_fn))
  unroll_hk = hk.without_apply_rng(hk.transform(unroll_fn))

  # Note: batch axis is not needed for the actors.
  dummy_obs = utils.zeros_like(env_spec.observations)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs)

  def unroll_init_fn(rng: networks_lib.PRNGKey,
                     initial_state: types.RecurrentState) -> hk.Params:
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  return types.RecurrentNetworks(
      forward_fn=forward_hk.apply,
      unroll_fn=unroll_hk.apply,
      unroll_init_fn=unroll_init_fn,
      initial_state_fn=(lambda rng, batch_size=None: initial_state_hk.apply(
          initial_state_hk.init(rng), batch_size)),
  )


def make_haiku_networks_2(
    env_spec: EnvironmentSpec,
    forward_fn: Any,
    initial_state_fn: Any,
    unroll_fn: Any,
    critic_fn: Any,
) -> types.ActorCriticRecurrentNetworks[types.RecurrentState]:
  """Builds functional network from recurrent model definitions."""
  # Make networks purely functional.
  forward_hk = hk.without_apply_rng(hk.transform(forward_fn))
  initial_state_hk = hk.without_apply_rng(hk.transform(initial_state_fn))
  unroll_hk = hk.without_apply_rng(hk.transform(unroll_fn))
  critic_hk = hk.without_apply_rng(hk.transform(critic_fn))

  # Note: batch axis is not needed for the actors.
  dummy_obs = utils.zeros_like(env_spec.observations)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs)

  def unroll_init_fn(rng: networks_lib.PRNGKey,
                     initial_state: types.RecurrentState) -> hk.Params:
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  return types.ActorCriticRecurrentNetworks(
      forward_fn=forward_hk.apply,
      critic_fn=critic_hk.apply,
      unroll_fn=unroll_hk.apply,
      unroll_init_fn=unroll_init_fn,
      initial_state_fn=(lambda rng, batch_size=None: initial_state_hk.apply(
          initial_state_hk.init(rng), batch_size)),
  )


class ArrayFE(hk.Module):

  def __init__(self, num_actions, hidden_dim=64):
    super().__init__("array_features")
    self.num_actions = num_actions
    self._layer = hk.Sequential([
        hk.Linear(hidden_dim),
        jax.nn.relu,
        hk.Linear(hidden_dim),
        jax.nn.relu,
    ])

  def __call__(self, inputs):
    op = self._layer(inputs["observation"]["agent_obs"])
    action = jax.nn.one_hot(inputs["action"], num_classes=self.num_actions)
    combined_op = jnp.concatenate([op, action], axis=-1)
    return combined_op


class ImageFE(hk.Module):

  def __init__(self, num_actions):
    super().__init__("image_features")
    self.num_actions = num_actions
    self._cnn = hk.Sequential([
        hk.Conv2D(16, [4, 4], 1, padding="VALID"),
        jax.nn.relu,
        hk.Conv2D(32, [4, 4], 1, padding="VALID"),
        jax.nn.relu,
    ])
    self._ff = hk.Sequential([
        hk.Linear(64),
        jax.nn.relu,
        hk.Linear(64),
        jax.nn.relu,
    ])

  def __call__(self, inputs) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs["observation"]["agent_obs"])
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

    outputs = self._cnn(inputs["observation"]["agent_obs"])

    if batched_inputs:
      outputs = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      outputs = jnp.reshape(outputs, [-1])  # [D]

    outputs = self._ff(outputs)
    action = jax.nn.one_hot(inputs["action"], num_classes=self.num_actions)
    combined_op = jnp.concatenate([outputs, action], axis=-1)
    return combined_op


class MeltingpotFE(hk.Module):

  def __init__(self, num_actions):
    super().__init__("meltingpot_features")
    self.num_actions = num_actions
    self._visual_torso = VisualFeatures()

  def __call__(self, inputs):
    # extract environment observation from the full observation object
    obs = inputs["observation"]

    # extract visual features form RGB observation
    ip_img = obs["RGB"].astype(jnp.float32) / 255
    vis_op = self._visual_torso(ip_img)

    # extract other observations
    inventory, ready_to_shoot = obs["INVENTORY"], obs["READY_TO_SHOOT"]

    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs["action"], num_classes=self.num_actions)  # [B, A]

    # Add dummy trailing dimensions to rewards if necessary.
    while ready_to_shoot.ndim < inventory.ndim:
      ready_to_shoot = jnp.expand_dims(ready_to_shoot, axis=-1)

    # check if the dimensions of all the tensors match
    # assert vis_op.ndim==inventory.ndim==ready_to_shoot.ndim==action.ndim

    # concatenate all the results
    combined_op = jnp.concatenate([vis_op, ready_to_shoot, inventory, action],
                                  axis=-1)

    return combined_op


class VisualFeatures(hk.Module):
  """Simple convolutional stack from MeltingPot paper."""

  def __init__(self):
    super().__init__(name="meltingpot_visual_features")
    self._cnn = hk.Sequential([
        hk.Conv2D(16, [8, 8], 8, padding="VALID"),
        jax.nn.relu,
        hk.Conv2D(32, [4, 4], 1, padding="VALID"),
        jax.nn.relu,
    ])
    self._ff = hk.Sequential([
        hk.Linear(64),
        jax.nn.relu,
        hk.Linear(64),
        jax.nn.relu,
    ])

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

    outputs = self._cnn(inputs)

    if batched_inputs:
      outputs = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      outputs = jnp.reshape(outputs, [-1])  # [D]

    outputs = self._ff(outputs)
    return outputs


class Discriminator(hk.Module):

  def __init__(self, diversity_dim, discriminator_ensembles):
    super().__init__(name="discriminator")
    self._diversity_dim = diversity_dim
    self._discriminator_ensembles = discriminator_ensembles
    self._layer = hk.Linear(diversity_dim * discriminator_ensembles)

  def __call__(self, inputs: jnp.array):
    op = self._layer(inputs)
    op = op.reshape((-1, self._discriminator_ensembles, self._diversity_dim))
    # op = jax.nn.softmax(op, axis=-1)
    return op
