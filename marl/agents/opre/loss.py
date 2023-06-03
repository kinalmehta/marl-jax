"""Loss function for OPRE[1].

[1] https://arxiv.org/abs/1906.01470v2
"""

from typing import Callable

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import rlax

from marl import types


def batched_art_opre_loss(
    network: types.RecurrentNetworks,
    popart_update_fn,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
    options_entropy_cost: float = 0.0,
    options_kl_cost: float = 0.0,
    pg_mix: bool = False,
) -> Callable[[hk.Params, types.TrainingData], jnp.DeviceArray]:
  """Builds the Batched OPRE loss function.

    Args:
      network: An OPRENetworks object containing a callable which maps
        (params, observations_sequence, initial_state) -> ((logits, value), state)
      discount: The standard geometric discount rate to apply.
      max_abs_reward: Optional symmetric reward clipping to apply.
      baseline_cost: Weighting of the critic loss relative to the policy loss.
      entropy_cost: Weighting of the entropy regulariser relative to policy loss.
      options_entropy_cost: Weighting of the options entropy regulariser.
      options_kl_cost: Weighting of the options KL divergence regulariser.

    Returns:
      A loss function with signature (params, popart_params, data) -> loss_scalar.
    """
  unroll_fn = jax.vmap(network.unroll_fn, in_axes=(None, 0, 0))
  critic_fn = jax.vmap(network.critic_fn, in_axes=(None, 0, 0))
  categorical_importance_sampling_ratios = jax.vmap(
      rlax.categorical_importance_sampling_ratios)
  vtrace = jax.vmap(rlax.vtrace)
  policy_gradient_loss = jax.vmap(rlax.policy_gradient_loss)
  entropy_loss = jax.vmap(rlax.entropy_loss)

  def loss_fn(params: hk.Params, popart_state: rlax.PopArtState,
              sample: types.TrainingData) -> jnp.DeviceArray:
    """Batched, MAHRL loss with V-trace."""

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
        hidden=initial_state["hidden"][:, 0], cell=initial_state["cell"][:, 0])
    behaviour_logits = extra["logits"]  # B, T, action_dim

    # Apply reward clipping.
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)  # B, T

    # Unroll current policy over observations.
    (mu_logits, norm_mu_values, feat, p_options, pi_logits, norm_values,
     q_options), _ = unroll_fn(params, observations, initial_state)

    # Compute importance sampling weights: current auxiliary policy / behavior policy.
    rhos = categorical_importance_sampling_ratios(pi_logits[:, :-1],
                                                  behaviour_logits[:, :-1],
                                                  actions[:, :-1])

    # V-trace Returns for PopArt update
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

    # PopArt statistics update
    new_popart_state = popart_update_fn(popart_state, vtrace_targets,
                                        indices[:, :-1])

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

    # jax.debug.print("new norm values: {} {}", jnp.max(new_norm_values), jnp.min(new_norm_values))
    # jax.debug.print("new values: {} {}", jnp.max(new_values), jnp.min(new_values))
    # jax.debug.print("new errors: {} {}", jnp.max(new_vtrace_errors), jnp.min(new_vtrace_errors))

    # V-trace Advantage
    # TODO: verify how batching effects concatenation
    q_bootstrap = jnp.concatenate(
        [
            new_vtrace_targets[:, 1:],
            new_values[:, -1:],
        ],
        axis=1,
    )  # B, T, value
    q_estimate = rewards[:, :-1] + discount_t * q_bootstrap
    q_estimate_norm = rlax.normalize(new_popart_state, q_estimate,
                                     indices[:, :-1])
    clipped_pg_rho_tm1 = jnp.minimum(1.0, rhos)
    pg_advantages = clipped_pg_rho_tm1 * (q_estimate_norm - norm_values[:, :-1])

    # Policy gradient loss from pi.
    w_t = jnp.ones_like(rewards[:, :-1])
    pi_pg_loss = policy_gradient_loss(
        logits_t=pi_logits[:, :-1],
        a_t=actions[:, :-1],
        adv_t=pg_advantages,
        w_t=w_t,
    )
    pi_pg_loss = jnp.mean(pi_pg_loss)

    # (Optional) Policy gradient loss from mu.
    if pg_mix:
      # rhos_mu = categorical_importance_sampling_ratios(mu_logits[:, :-1],
      #                                                  behaviour_logits[:, :-1],
      #                                                  actions[:, :-1])
      # vtrace_mu = vtrace_td_error_and_advantage(
      #     v_tm1=mu_values[:, :-1],
      #     v_t=mu_values[:, 1:],
      #     r_t=rewards[:, :-1],
      #     discount_t=discounts[:, :-1] * discount,
      #     rho_tm1=rhos_mu,
      # )

      # mix_pg = policy_gradient_loss(
      #     logits_t=mu_logits[:, :-1],
      #     a_t=actions[:, :-1],
      #     adv_t=vtrace_mu.pg_advantage,
      #     w_t=w_t,
      # )
      # mix_pg = jnp.mean(mix_pg)
      # pi_pg_loss += jnp.mean(mix_pg)
      raise Exception("PGMix for popart not available")

    # Entropy regulariser.
    pi_entropy = entropy_loss(pi_logits[:, :-1], w_t)
    pi_entropy = jnp.mean(pi_entropy)

    # Entropy for options over time
    t_q_options = jnp.transpose(
        q_options[:, :-1], (0, 2, 1))  # B, T, option_dim -> B, option_dim, T
    t_entropy = entropy_loss(t_q_options, jnp.ones(t_q_options.shape[:2]))
    t_entropy = jnp.mean(t_entropy)

    # entropy for options over batches
    b_q_options = jnp.transpose(
        q_options[:, :-1], (1, 2, 0))  # B, T, option_dim -> T, option_dim, B
    b_entropy = entropy_loss(b_q_options, jnp.ones(b_q_options.shape[:2]))
    b_entropy = jnp.mean(b_entropy)

    options_entropy = b_entropy - t_entropy

    # KL loss
    kl_pq = distrax.Categorical(probs=q_options).kl_divergence(
        distrax.Categorical(probs=p_options))
    kl_pq = jnp.mean(kl_pq)

    # Combine weighted sum of actor & critic losses, averaged over the sequence.
    critic_loss *= baseline_cost
    pi_entropy *= entropy_cost
    options_entropy *= options_entropy_cost
    kl_pq *= options_kl_cost
    mean_loss = pi_pg_loss + critic_loss + pi_entropy + options_entropy + kl_pq  # []

    metrics = {
        "total_loss": mean_loss,
        "policy_loss": pi_pg_loss,
        "critic_loss": critic_loss,
        "pi_entropy_loss": pi_entropy,
        "options_entropy_loss": options_entropy,
        "options_time_entropy": t_entropy,
        "options_batch_entropy": b_entropy,
        "options_kl_loss": kl_pq,
        "extrinsic_reward": jnp.mean(rewards),
    }

    return mean_loss, (new_popart_state, metrics)

  return loss_fn


def batched_popart_opre_loss(
    network: types.RecurrentNetworks,
    popart_update_fn,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
    options_entropy_cost: float = 0.0,
    options_kl_cost: float = 0.0,
    pg_mix: bool = False,
) -> Callable[[hk.Params, types.TrainingData], jnp.DeviceArray]:
  """Builds the Batched OPRE loss function.

    Args:
      network: An OPRENetworks object containing a callable which maps
        (params, observations_sequence, initial_state) -> ((logits, value), state)
      discount: The standard geometric discount rate to apply.
      max_abs_reward: Optional symmetric reward clipping to apply.
      baseline_cost: Weighting of the critic loss relative to the policy loss.
      entropy_cost: Weighting of the entropy regulariser relative to policy loss.
      options_entropy_cost: Weighting of the options entropy regulariser.
      options_kl_cost: Weighting of the options KL divergence regulariser.

    Returns:
      A loss function with signature (params, popart_params, data) -> loss_scalar.
    """
  unroll_fn = jax.vmap(network.unroll_fn, in_axes=(None, 0, 0))
  critic_fn = jax.vmap(network.critic_fn, in_axes=(None, 0, 0))
  categorical_importance_sampling_ratios = jax.vmap(
      rlax.categorical_importance_sampling_ratios)
  vtrace = jax.vmap(rlax.vtrace)
  policy_gradient_loss = jax.vmap(rlax.policy_gradient_loss)
  entropy_loss = jax.vmap(rlax.entropy_loss)

  def loss_fn(params: hk.Params, popart_state: rlax.PopArtState,
              sample: types.TrainingData) -> jnp.DeviceArray:
    """Batched, OPRE loss with V-trace."""

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
        hidden=initial_state["hidden"][:, 0], cell=initial_state["cell"][:, 0])
    behaviour_logits = extra["logits"]  # B, T, action_dim

    # Apply reward clipping.
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)  # B, T

    # Unroll current policy over observations.
    (
        mu_logits,
        norm_mu_values,
        feat,
        p_options,
        pi_logits,
        norm_values,
        q_options,
    ), _ = unroll_fn(params, observations, initial_state)

    # Compute importance sampling weights: current auxiliary policy / behavior policy.
    rhos = categorical_importance_sampling_ratios(pi_logits[:, :-1],
                                                  behaviour_logits[:, :-1],
                                                  actions[:, :-1])

    # V-trace Returns for PopArt update
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

    # PopArt statistics update
    final_value_layer = "opre_network/~/value_layer"
    mutable_params = hk.data_structures.to_mutable_dict(params)
    linear_params = mutable_params[final_value_layer]
    popped_linear_params, new_popart_state = popart_update_fn(
        linear_params, popart_state, vtrace_targets, indices[:, :-1])
    mutable_params[final_value_layer] = popped_linear_params
    popped_params = hk.data_structures.to_immutable_dict(mutable_params)

    # Critic loss using V-trace with updated PopArt parameters
    new_norm_values = critic_fn(popped_params, feat, q_options)
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

    # V-trace Advantage
    # TODO: verify how batching effects concatenation
    q_bootstrap = jnp.concatenate(
        [
            new_vtrace_targets[:, 1:],
            new_values[:, -1:],
        ],
        axis=1,
    )  # B, T, value
    q_estimate = rewards[:, :-1] + discount_t * q_bootstrap
    q_estimate_norm = rlax.normalize(new_popart_state, q_estimate,
                                     indices[:, :-1])
    clipped_pg_rho_tm1 = jnp.minimum(1.0, rhos)
    pg_advantages = clipped_pg_rho_tm1 * (
        q_estimate_norm - new_norm_values[:, :-1])

    # Policy gradient loss from pi.
    w_t = jnp.ones_like(rewards[:, :-1])
    pi_pg_loss = policy_gradient_loss(
        logits_t=pi_logits[:, :-1],
        a_t=actions[:, :-1],
        adv_t=pg_advantages,
        w_t=w_t,
    )
    pi_pg_loss = jnp.mean(pi_pg_loss)

    # (Optional) Policy gradient loss from mu.
    if pg_mix:
      # rhos_mu = categorical_importance_sampling_ratios(mu_logits[:, :-1],
      #                                                  behaviour_logits[:, :-1],
      #                                                  actions[:, :-1])
      # vtrace_mu = vtrace_td_error_and_advantage(
      #     v_tm1=mu_values[:, :-1],
      #     v_t=mu_values[:, 1:],
      #     r_t=rewards[:, :-1],
      #     discount_t=discounts[:, :-1] * discount,
      #     rho_tm1=rhos_mu,
      # )

      # mix_pg = policy_gradient_loss(
      #     logits_t=mu_logits[:, :-1],
      #     a_t=actions[:, :-1],
      #     adv_t=vtrace_mu.pg_advantage,
      #     w_t=w_t,
      # )
      # mix_pg = jnp.mean(mix_pg)
      # pi_pg_loss += jnp.mean(mix_pg)
      raise Exception("PGMix for popart not available")

    # Entropy regulariser.
    pi_entropy = entropy_loss(pi_logits[:, :-1], w_t)
    pi_entropy = jnp.mean(pi_entropy)

    # Entropy for options over time
    t_q_options = jnp.transpose(
        q_options[:, :-1], (0, 2, 1))  # B, T, option_dim -> B, option_dim, T
    t_entropy = entropy_loss(t_q_options, jnp.ones(t_q_options.shape[:2]))
    t_entropy = jnp.mean(t_entropy)

    # entropy for options over batches
    b_q_options = jnp.transpose(
        q_options[:, :-1], (1, 2, 0))  # B, T, option_dim -> T, option_dim, B
    b_entropy = entropy_loss(b_q_options, jnp.ones(b_q_options.shape[:2]))
    b_entropy = jnp.mean(b_entropy)

    options_entropy = b_entropy - t_entropy

    # KL loss
    kl_pq = distrax.Categorical(probs=q_options).kl_divergence(
        distrax.Categorical(probs=p_options))
    kl_pq = jnp.mean(kl_pq)

    # Combine weighted sum of actor & critic losses, averaged over the sequence.
    critic_loss *= baseline_cost
    pi_entropy *= entropy_cost
    options_entropy *= options_entropy_cost
    kl_pq *= options_kl_cost
    mean_loss = pi_pg_loss + critic_loss + pi_entropy + options_entropy + kl_pq  # []

    metrics = {
        "total_loss": mean_loss,
        "policy_loss": pi_pg_loss,
        "critic_loss": critic_loss,
        "pi_entropy_loss": pi_entropy,
        "options_entropy_loss": options_entropy,
        "options_time_entropy": t_entropy,
        "options_batch_entropy": b_entropy,
        "options_kl_loss": kl_pq,
        "extrinsic_reward": jnp.mean(rewards),
    }

    return mean_loss, (new_popart_state, metrics)

  return loss_fn


def batched_opre_loss(
    network: types.RecurrentNetworks,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
    options_entropy_cost: float = 0.0,
    options_kl_cost: float = 0.0,
    pg_mix: bool = False,
) -> Callable[[hk.Params, types.TrainingData], jnp.DeviceArray]:
  """Builds the Batched OPRE loss function.

    Args:
      network: An OPRENetworks object containing a callable which maps
        (params, observations_sequence, initial_state) -> ((logits, value), state)
      discount: The standard geometric discount rate to apply.
      max_abs_reward: Optional symmetric reward clipping to apply.
      baseline_cost: Weighting of the critic loss relative to the policy loss.
      entropy_cost: Weighting of the entropy regulariser relative to policy loss.
      options_entropy_cost: Weighting of the options entropy regulariser.
      options_kl_cost: Weighting of the options KL divergence regulariser.

    Returns:
      A loss function with signature (params, data) -> loss_scalar.
    """
  unroll_fn = jax.vmap(network.unroll_fn, in_axes=(None, 0, 0))
  categorical_importance_sampling_ratios = jax.vmap(
      rlax.categorical_importance_sampling_ratios)
  vtrace_td_error_and_advantage = jax.vmap(rlax.vtrace_td_error_and_advantage)
  policy_gradient_loss = jax.vmap(rlax.policy_gradient_loss)
  entropy_loss = jax.vmap(rlax.entropy_loss)

  def loss_fn(params: hk.Params, sample: types.TrainingData) -> jnp.DeviceArray:
    """Batched, OPRE loss with V-trace."""

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
        hidden=initial_state["hidden"][:, 0], cell=initial_state["cell"][:, 0])
    behaviour_logits = extra["logits"]  # B, T, action_dim

    # Apply reward clipping.
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)  # B, T

    # Unroll current policy over observations.
    (
        mu_logits,
        mu_values,
        _,
        p_options,
        pi_logits,
        values,
        q_options,
    ), _ = unroll_fn(params, observations, initial_state)

    # Compute importance sampling weights: current auxiliary policy / behavior policy.
    rhos = categorical_importance_sampling_ratios(pi_logits[:, :-1],
                                                  behaviour_logits[:, :-1],
                                                  actions[:, :-1])

    # V-trace Returns
    vtrace_returns = vtrace_td_error_and_advantage(
        v_tm1=values[:, :-1],
        v_t=values[:, 1:],
        r_t=rewards[:, :-1],
        discount_t=discounts[:, :-1] * discount,
        rho_tm1=rhos,
    )
    critic_loss = jnp.square(vtrace_returns.errors)
    critic_loss = jnp.mean(critic_loss)

    # Policy gradient loss from pi.
    w_t = jnp.ones_like(rewards[:, :-1])
    pi_pg_loss = policy_gradient_loss(
        logits_t=pi_logits[:, :-1],
        a_t=actions[:, :-1],
        adv_t=vtrace_returns.pg_advantage,
        w_t=w_t,
    )
    pi_pg_loss = jnp.mean(pi_pg_loss)

    # (Optional) Policy gradient loss from mu.
    if pg_mix:
      rhos_mu = categorical_importance_sampling_ratios(mu_logits[:, :-1],
                                                       behaviour_logits[:, :-1],
                                                       actions[:, :-1])
      vtrace_mu = vtrace_td_error_and_advantage(
          v_tm1=mu_values[:, :-1],
          v_t=mu_values[:, 1:],
          r_t=rewards[:, :-1],
          discount_t=discounts[:, :-1] * discount,
          rho_tm1=rhos_mu,
      )

      mix_pg = policy_gradient_loss(
          logits_t=mu_logits[:, :-1],
          a_t=actions[:, :-1],
          adv_t=vtrace_mu.pg_advantage,
          w_t=w_t,
      )
      mix_pg = jnp.mean(mix_pg)
      pi_pg_loss += jnp.mean(mix_pg)

    # Entropy regulariser.
    pi_entropy = entropy_loss(pi_logits[:, :-1], w_t)
    pi_entropy = jnp.mean(pi_entropy)

    # Entropy for options over time
    t_q_options = jnp.transpose(
        q_options[:, :-1], (0, 2, 1))  # B, T, option_dim -> B, option_dim, T
    t_entropy = entropy_loss(t_q_options, jnp.ones(t_q_options.shape[:2]))
    t_entropy = jnp.mean(t_entropy)

    # entropy for options over batches
    b_q_options = jnp.transpose(
        q_options[:, :-1], (1, 2, 0))  # B, T, option_dim -> T, option_dim, B
    b_entropy = entropy_loss(b_q_options, jnp.ones(b_q_options.shape[:2]))
    b_entropy = jnp.mean(b_entropy)

    options_entropy = b_entropy - t_entropy

    # KL loss
    kl_pq = distrax.Categorical(probs=q_options).kl_divergence(
        distrax.Categorical(probs=p_options))
    kl_pq = jnp.mean(kl_pq)

    # Combine weighted sum of actor & critic losses, averaged over the sequence.
    critic_loss *= baseline_cost
    pi_entropy *= entropy_cost
    options_entropy *= options_entropy_cost
    kl_pq *= options_kl_cost
    mean_loss = pi_pg_loss + critic_loss + pi_entropy + options_entropy + kl_pq  # []

    metrics = {
        "total_loss": mean_loss,
        "policy_loss": pi_pg_loss,
        "critic_loss": critic_loss,
        "pi_entropy_loss": pi_entropy,
        "options_entropy_loss": options_entropy,
        "options_time_entropy": t_entropy,
        "options_batch_entropy": b_entropy,
        "options_kl_loss": kl_pq,
        "extrinsic_reward": jnp.mean(rewards),
    }

    return mean_loss, metrics

  return loss_fn
