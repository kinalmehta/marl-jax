"""Loss function for IMPALA (Espeholt et al., 2018) [1].
Adapted from https://github.com/deepmind/acme/blob/master/acme/jax/losses/impala.py

[1] https://arxiv.org/abs/1802.01561
"""

from typing import Callable

from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import rlax

from marl import types

# from marl.modules.debug import db


def batched_art_impala_loss(
    network: types.RecurrentNetworks,
    popart_update_fn,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
) -> Callable[[hk.Params, types.TrainingData], jnp.DeviceArray]:
  """Builds the standard entropy-regularised IMPALA loss function.

    Args:
      network: An IMPALANetworks object containing a callable which maps
        (params, observations_sequence, initial_state) -> ((logits, value), state)
      discount: The standard geometric discount rate to apply.
      max_abs_reward: Optional symmetric reward clipping to apply.
      baseline_cost: Weighting of the critic loss relative to the policy loss.
      entropy_cost: Weighting of the entropy regulariser relative to policy loss.

    Returns:
      A loss function with signature (params, data) -> loss_scalar.
    """
  unroll_fn = jax.vmap(network.unroll_fn, in_axes=(None, 0, 0))
  critic_fn = jax.vmap(network.critic_fn, in_axes=(None, 0))
  categorical_importance_sampling_ratios = jax.vmap(
      rlax.categorical_importance_sampling_ratios)
  vtrace = jax.vmap(rlax.vtrace)
  policy_gradient_loss = jax.vmap(rlax.policy_gradient_loss)
  entropy_loss = jax.vmap(rlax.entropy_loss)

  def loss_fn(params: hk.Params, popart_state: rlax.PopArtState,
              sample: types.TrainingData) -> jnp.DeviceArray:
    """Batched, entropy-regularised actor-critic loss with V-trace."""

    # Extract the data
    data = sample
    observations, actions, rewards, discounts, extra = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.extras,
    )

    initial_state = extra["core_state"]
    initial_state = hk.LSTMState(
        hidden=initial_state["hidden"][:, 0], cell=initial_state["cell"][:, 0])
    behaviour_logits = extra["logits"]

    # Apply reward clipping
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

    # Unroll current policy over observations
    (logits, norm_values,
     hidden_features), _ = unroll_fn(params, observations, initial_state)

    # Compute importance sampling weights: current policy / behavior policy
    rhos = categorical_importance_sampling_ratios(logits[:, :-1],
                                                  behaviour_logits[:, :-1],
                                                  actions[:, :-1])

    # V-trace Returns
    indices = jnp.zeros_like(
        norm_values, dtype=jnp.int32)  # Only one output for normalization

    values = rlax.unnormalize(popart_state, norm_values, indices)
    stop_gradients_bool = jnp.zeros(values.shape[:-1], dtype=bool)
    discount_t = discounts[:, :-1] * discount

    vtrace_errors = vtrace(
        v_tm1=values[:, :-1],
        v_t=values[:, 1:],
        r_t=rewards[:, :-1],
        discount_t=discount_t,
        rho_tm1=rhos,
        stop_target_gradients=stop_gradients_bool,
    )
    vtrace_targets = vtrace_errors + values[:, :-1]

    # popart_targets = rewards[:, :-1] + discount_t * values[:, 1:]

    # jax.debug.print("old norm values: {} {}", jnp.max(norm_values), jnp.min(norm_values))
    # jax.debug.print("old values: {} {}", jnp.max(values), jnp.min(values))
    # jax.debug.print("old errors: {} {}", jnp.max(vtrace_errors), jnp.min(vtrace_errors))

    # db(norm_values, "OLD norm_values")
    # db(values, "OLD values")
    # db(vtrace_errors, "OLD vtrace_errors")
    # db(vtrace_targets, "OLD vtrace_targets")
    # db(popart_state, "OLD popart_state")
    # jax.debug.breakpoint()

    # PopArt statistics update
    new_popart_state = popart_update_fn(popart_state, vtrace_targets,
                                        indices[:, :-1])

    # db(new_popart_state, "NEW popart_state")
    # jax.debug.breakpoint()

    # Critic loss using V-trace with updated PopArt parameters
    new_values = rlax.unnormalize(new_popart_state, norm_values, indices)
    new_vtrace_errors = vtrace(
        v_tm1=new_values[:, :-1],
        v_t=new_values[:, 1:],
        r_t=rewards[:, :-1],
        discount_t=discount_t,
        rho_tm1=rhos,
        stop_target_gradients=stop_gradients_bool,
    )
    new_vtrace_targets = new_vtrace_errors + new_values[:, :-1]
    new_vtrace_targets_norm = rlax.normalize(new_popart_state,
                                             new_vtrace_targets,
                                             indices[:, :-1])
    new_vtrace_targets_norm = jax.lax.stop_gradient(new_vtrace_targets_norm)
    critic_loss = new_vtrace_targets_norm - norm_values[:, :-1]
    critic_loss = jnp.mean(jnp.square(critic_loss))

    # db(norm_values, "norm_values")
    # db(new_values, "NEW_values")
    # db(new_vtrace_errors, "NEW_vtrace_errors")
    # db(new_vtrace_targets, "NEW_vtrace_targets")
    # jax.debug.breakpoint()

    # jax.debug.print("new norm values: {} {}", jnp.max(new_norm_values), jnp.min(new_norm_values))
    # jax.debug.print("new values: {} {}", jnp.max(new_values), jnp.min(new_values))
    # jax.debug.print("new errors: {} {}", jnp.max(new_vtrace_errors), jnp.min(new_vtrace_errors))

    # V-trace Advantage
    q_bootstrap = jnp.concatenate(
        [
            new_vtrace_targets[:, 1:],
            new_values[:, -1:],
        ],
        axis=1,
    )
    q_estimate = rewards[:, :-1] + discount_t * q_bootstrap
    q_estimate_norm = rlax.normalize(new_popart_state, q_estimate,
                                     indices[:, :-1])
    clipped_pg_rho_tm1 = jnp.minimum(1.0, rhos)
    pg_advantages = clipped_pg_rho_tm1 * (q_estimate_norm - norm_values[:, :-1])

    # Policy gradient loss
    w_t = jnp.ones_like(rewards[:, :-1])
    pg_loss = policy_gradient_loss(
        logits_t=logits[:, :-1],
        a_t=actions[:, :-1],
        adv_t=pg_advantages,
        w_t=w_t,
    )
    pg_loss = jnp.mean(pg_loss)

    # Entropy regulariser
    pi_entropy = entropy_loss(logits[:, :-1], w_t)
    pi_entropy = jnp.mean(pi_entropy)

    # Combine weighted sum of actor & critic losses, averaged over the sequence
    critic_loss *= baseline_cost
    pi_entropy *= entropy_cost
    mean_loss = pg_loss + critic_loss + pi_entropy  # []

    metrics = {
        "total_loss": mean_loss,
        "policy_loss": pg_loss,
        "critic_loss": critic_loss,
        "pi_entropy_loss": pi_entropy,
        "extrinsic_reward": jnp.mean(rewards),
    }

    return mean_loss, (new_popart_state, metrics)

  return loss_fn


def batched_popart_impala_loss(
    network: types.RecurrentNetworks,
    popart_update_fn,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
) -> Callable[[hk.Params, types.TrainingData], jnp.DeviceArray]:
  """Builds the standard entropy-regularised IMPALA loss function.

    Args:
      network: An IMPALANetworks object containing a callable which maps
        (params, observations_sequence, initial_state) -> ((logits, value), state)
      discount: The standard geometric discount rate to apply.
      max_abs_reward: Optional symmetric reward clipping to apply.
      baseline_cost: Weighting of the critic loss relative to the policy loss.
      entropy_cost: Weighting of the entropy regulariser relative to policy loss.

    Returns:
      A loss function with signature (params, data) -> loss_scalar.
    """
  unroll_fn = jax.vmap(network.unroll_fn, in_axes=(None, 0, 0))
  critic_fn = jax.vmap(network.critic_fn, in_axes=(None, 0))
  categorical_importance_sampling_ratios = jax.vmap(
      rlax.categorical_importance_sampling_ratios)
  vtrace = jax.vmap(rlax.vtrace)
  policy_gradient_loss = jax.vmap(rlax.policy_gradient_loss)
  entropy_loss = jax.vmap(rlax.entropy_loss)

  def loss_fn(params: hk.Params, popart_state: rlax.PopArtState,
              sample: types.TrainingData) -> jnp.DeviceArray:
    """Batched, entropy-regularised actor-critic loss with V-trace."""

    # Extract the data
    data = sample
    observations, actions, rewards, discounts, extra = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.extras,
    )

    initial_state = extra["core_state"]
    initial_state = hk.LSTMState(
        hidden=initial_state["hidden"][:, 0], cell=initial_state["cell"][:, 0])
    behaviour_logits = extra["logits"]

    # Apply reward clipping
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

    # Unroll current policy over observations
    (logits, norm_values,
     hidden_features), _ = unroll_fn(params, observations, initial_state)

    # Compute importance sampling weights: current policy / behavior policy
    rhos = categorical_importance_sampling_ratios(logits[:, :-1],
                                                  behaviour_logits[:, :-1],
                                                  actions[:, :-1])

    # V-trace Returns
    indices = jnp.zeros_like(
        norm_values, dtype=jnp.int32)  # Only one output for normalization

    values = rlax.unnormalize(popart_state, norm_values, indices)
    stop_gradients_bool = jnp.zeros(values.shape[:-1], dtype=bool)
    discount_t = discounts[:, :-1] * discount

    vtrace_errors = vtrace(
        v_tm1=values[:, :-1],
        v_t=values[:, 1:],
        r_t=rewards[:, :-1],
        discount_t=discount_t,
        rho_tm1=rhos,
        stop_target_gradients=stop_gradients_bool,
    )
    vtrace_targets = vtrace_errors + values[:, :-1]

    # jax.debug.print("old norm values: {} {}", jnp.max(norm_values), jnp.min(norm_values))
    # jax.debug.print("old values: {} {}", jnp.max(values), jnp.min(values))
    # jax.debug.print("old errors: {} {}", jnp.max(vtrace_errors), jnp.min(vtrace_errors))

    # PopArt statistics update
    final_value_layer = "impala_network/~/value_layer"
    mutable_params = hk.data_structures.to_mutable_dict(params)
    linear_params = mutable_params[final_value_layer]
    popped_linear_params, new_popart_state = popart_update_fn(
        linear_params, popart_state, vtrace_targets, indices[:, :-1])
    mutable_params[final_value_layer] = popped_linear_params
    popped_params = hk.data_structures.to_immutable_dict(mutable_params)

    # Critic loss using V-trace with updated PopArt parameters
    new_norm_values = critic_fn(popped_params, hidden_features)
    new_values = rlax.unnormalize(new_popart_state, new_norm_values, indices)
    new_vtrace_errors = vtrace(
        v_tm1=new_values[:, :-1],
        v_t=new_values[:, 1:],
        r_t=rewards[:, :-1],
        discount_t=discount_t,
        rho_tm1=rhos,
        stop_target_gradients=stop_gradients_bool,
    )
    new_vtrace_targets = new_vtrace_errors + new_values[:, :-1]
    new_vtrace_targets_norm = rlax.normalize(new_popart_state,
                                             new_vtrace_targets,
                                             indices[:, :-1])
    new_vtrace_targets_norm = jax.lax.stop_gradient(new_vtrace_targets_norm)
    critic_loss = new_vtrace_targets_norm - new_norm_values[:, :-1]
    critic_loss = jnp.mean(jnp.square(critic_loss))

    # jax.debug.print("new norm values: {} {}", jnp.max(new_norm_values), jnp.min(new_norm_values))
    # jax.debug.print("new values: {} {}", jnp.max(new_values), jnp.min(new_values))
    # jax.debug.print("new errors: {} {}", jnp.max(new_vtrace_errors), jnp.min(new_vtrace_errors))

    # V-trace Advantage
    q_bootstrap = jnp.concatenate(
        [
            new_vtrace_targets[:, 1:],
            new_values[:, -1:],
        ],
        axis=1,
    )
    q_estimate = rewards[:, :-1] + discount_t * q_bootstrap
    q_estimate_norm = rlax.normalize(new_popart_state, q_estimate,
                                     indices[:, :-1])
    clipped_pg_rho_tm1 = jnp.minimum(1.0, rhos)
    pg_advantages = clipped_pg_rho_tm1 * (
        q_estimate_norm - new_norm_values[:, :-1])

    # Policy gradient loss
    w_t = jnp.ones_like(rewards[:, :-1])
    pg_loss = policy_gradient_loss(
        logits_t=logits[:, :-1],
        a_t=actions[:, :-1],
        adv_t=pg_advantages,
        w_t=w_t,
    )
    pg_loss = jnp.mean(pg_loss)

    # Entropy regulariser
    pi_entropy = entropy_loss(logits[:, :-1], w_t)
    pi_entropy = jnp.mean(pi_entropy)

    # Combine weighted sum of actor & critic losses, averaged over the sequence
    critic_loss *= baseline_cost
    pi_entropy *= entropy_cost
    mean_loss = pg_loss + critic_loss + pi_entropy  # []

    metrics = {
        "total_loss": mean_loss,
        "policy_loss": pg_loss,
        "critic_loss": critic_loss,
        "pi_entropy_loss": pi_entropy,
        "extrinsic_reward": jnp.mean(rewards),
    }

    return mean_loss, (new_popart_state, metrics)

  return loss_fn


def popart_impala_loss(
    network: types.RecurrentNetworks,
    popart_update_fn,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
) -> Callable[[hk.Params, types.TrainingData], jnp.DeviceArray]:
  """Builds the standard entropy-regularised IMPALA loss function.

    Args:
      network: An IMPALANetworks object containing a callable which maps
        (params, observations_sequence, initial_state) -> ((logits, value), state)
      discount: The standard geometric discount rate to apply.
      max_abs_reward: Optional symmetric reward clipping to apply.
      baseline_cost: Weighting of the critic loss relative to the policy loss.
      entropy_cost: Weighting of the entropy regulariser relative to policy loss.

    Returns:
      A loss function with signature (params, data) -> loss_scalar.
    """

  # popart_update_fn = jax.vmap(popart_update_fn, in_axes=(None, None, 0, None))

  def loss_fn(params: hk.Params, popart_state: rlax.PopArtState,
              sample: types.TrainingData) -> jnp.DeviceArray:
    """Batched, entropy-regularised actor-critic loss with V-trace."""

    # Extract the data
    data = sample
    observations, actions, rewards, discounts, extra = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.extras,
    )

    initial_state = extra["core_state"]
    initial_state = hk.LSTMState(
        hidden=initial_state["hidden"][0], cell=initial_state["cell"][0])
    behaviour_logits = extra["logits"]

    # Apply reward clipping
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

    # Unroll current policy over observations
    (logits, norm_values,
     hidden_features), _ = network.unroll_fn(params, observations,
                                             initial_state)

    # Compute importance sampling weights: current policy / behavior policy
    rhos = rlax.categorical_importance_sampling_ratios(logits[:-1],
                                                       behaviour_logits[:-1],
                                                       actions[:-1])

    # V-trace Returns
    indices = jnp.zeros_like(
        norm_values, dtype=jnp.int32)  # Only one output for normalization
    values = rlax.unnormalize(popart_state, norm_values, indices)
    discount_t = discounts[:-1] * discount
    vtrace_errors = rlax.vtrace(
        v_tm1=values[:-1],
        v_t=values[1:],
        r_t=rewards[:-1],
        discount_t=discount_t,
        rho_tm1=rhos,
        stop_target_gradients=False,
    )
    vtrace_targets = vtrace_errors + values[:-1]

    # PopArt statistics update
    final_value_layer = "impala_network/~/value_layer"
    mutable_params = hk.data_structures.to_mutable_dict(params)
    linear_params = mutable_params[final_value_layer]
    popped_linear_params, new_popart_state = popart_update_fn(
        linear_params, popart_state, vtrace_targets, indices[:-1])
    mutable_params[final_value_layer] = popped_linear_params
    popped_params = hk.data_structures.to_immutable_dict(mutable_params)

    # Critic loss
    new_norm_values = network.critic_fn(popped_params, hidden_features)
    new_values = rlax.unnormalize(new_popart_state, new_norm_values, indices)

    vtrace_targets_norm = rlax.normalize(new_popart_state, vtrace_targets,
                                         indices[:-1])
    vtrace_targets_norm = jax.lax.stop_gradient(vtrace_targets_norm)
    critic_loss = vtrace_targets_norm - new_norm_values[:-1]
    critic_loss = jnp.mean(jnp.square(critic_loss))

    # V-trace Advantage
    q_bootstrap = jnp.concatenate(
        [
            vtrace_targets[1:],
            new_values[-1:],
        ],
        axis=0,
    )
    q_estimate = rewards[:-1] + discount_t * q_bootstrap
    q_estimate_norm = rlax.normalize(new_popart_state, q_estimate, indices[:-1])
    clipped_pg_rho_tm1 = jnp.minimum(1.0, rhos)
    pg_advantages = clipped_pg_rho_tm1 * (
        q_estimate_norm - new_norm_values[:-1])

    # Policy gradient loss
    w_t = jnp.ones_like(rewards[:-1])
    policy_gradient_loss = rlax.policy_gradient_loss(
        logits_t=logits[:-1],
        a_t=actions[:-1],
        adv_t=pg_advantages,
        w_t=w_t,
    )
    policy_gradient_loss = jnp.mean(policy_gradient_loss)

    # Entropy regulariser
    entropy_loss = rlax.entropy_loss(logits[:-1], w_t)
    entropy_loss = jnp.mean(entropy_loss)

    # Combine weighted sum of actor & critic losses, averaged over the sequence
    critic_loss *= baseline_cost
    entropy_loss *= entropy_cost
    mean_loss = policy_gradient_loss + critic_loss + entropy_loss  # []

    metrics = {
        "total_loss": mean_loss,
        "policy_loss": policy_gradient_loss,
        "critic_loss": critic_loss,
        "pi_entropy_loss": entropy_loss,
        "extrinsic_reward": jnp.mean(rewards),
    }

    return mean_loss, (new_popart_state, metrics)

  return utils.mapreduce(loss_fn, in_axes=(None, None, 0))


def impala_loss(
    network: types.RecurrentNetworks,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
) -> Callable[[hk.Params, types.TrainingData], jnp.DeviceArray]:
  """Builds the standard entropy-regularised IMPALA loss function.

    Args:
      network: An IMPALANetworks object containing a callable which maps
        (params, observations_sequence, initial_state) -> ((logits, value), state)
      discount: The standard geometric discount rate to apply.
      max_abs_reward: Optional symmetric reward clipping to apply.
      baseline_cost: Weighting of the critic loss relative to the policy loss.
      entropy_cost: Weighting of the entropy regulariser relative to policy loss.

    Returns:
      A loss function with signature (params, data) -> loss_scalar.
    """

  def loss_fn(params: hk.Params, sample: types.TrainingData) -> jnp.DeviceArray:
    """Batched, entropy-regularised actor-critic loss with V-trace."""

    # Extract the data.
    data = sample
    observations, actions, rewards, discounts, extra = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.extras,
    )

    initial_state = extra["core_state"]
    initial_state = hk.LSTMState(
        hidden=initial_state["hidden"][0], cell=initial_state["cell"][0])
    behaviour_logits = extra["logits"]

    # Apply reward clipping.
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

    # Unroll current policy over observations.
    (logits, values, _), _ = network.unroll_fn(params, observations,
                                               initial_state)

    # Compute importance sampling weights: current policy / behavior policy.
    rhos = rlax.categorical_importance_sampling_ratios(logits[:-1],
                                                       behaviour_logits[:-1],
                                                       actions[:-1])

    # Critic loss.
    vtrace_returns = rlax.vtrace_td_error_and_advantage(
        v_tm1=values[:-1],
        v_t=values[1:],
        r_t=rewards[:-1],
        discount_t=discounts[:-1] * discount,
        rho_tm1=rhos,
    )
    critic_loss = jnp.square(vtrace_returns.errors)
    critic_loss = jnp.mean(critic_loss)

    # Policy gradient loss.
    policy_gradient_loss = rlax.policy_gradient_loss(
        logits_t=logits[:-1],
        a_t=actions[:-1],
        adv_t=vtrace_returns.pg_advantage,
        w_t=jnp.ones_like(rewards[:-1]),
    )
    policy_gradient_loss = jnp.mean(policy_gradient_loss)

    # Entropy regulariser.
    entropy_loss = rlax.entropy_loss(logits[:-1], jnp.ones_like(rewards[:-1]))
    entropy_loss = jnp.mean(entropy_loss)

    # Combine weighted sum of actor & critic losses, averaged over the sequence.
    critic_loss *= baseline_cost
    entropy_loss *= entropy_cost
    mean_loss = policy_gradient_loss + critic_loss + entropy_loss  # []

    metrics = {
        "total_loss": mean_loss,
        "policy_loss": policy_gradient_loss,
        "critic_loss": critic_loss,
        "pi_entropy_loss": entropy_loss,
        "extrinsic_reward": jnp.mean(rewards),
    }

    return mean_loss, metrics

  return utils.mapreduce(loss_fn, in_axes=(None, 0))
