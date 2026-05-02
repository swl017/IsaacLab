"""Bit-exact regression: with aux_loss_scale=0.0, gradients on shared
parameters (encoder MLP + GRU + policy_layer + value_layer) must be
bit-identical to a baseline run with tri_head_enabled=False.

This is the wired-but-off gate. If aux_loss leaks into the optimizer step
even when its scale is 0, gradients diverge from baseline silently.
"""
import traceback

import gymnasium as gym
import numpy as np
import torch

from mappo_with_aux import (
    MAPPOWithAuxPolicy,
    MAPPOWithAuxValue,
    compute_aux_nll_loss,
)


def _seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _build_policy(device, *, tri_head_enabled, seed=42):
    """Build a MAPPOWithAuxPolicy with a known-deterministic init.

    We construct in a controlled order so that — with the same seed — the
    encoder + policy_layer parameters are bit-identical regardless of
    tri_head_enabled. The tri_head consumes RNG state AFTER the policy_layer
    is initialized.
    """
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    _seed_all(seed)
    policy = MAPPOWithAuxPolicy(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        hidden_size=32, gru_num_layers=1, gru_hidden_size=32,
        num_envs=4, sequence_length=4,
        tri_head_enabled=tri_head_enabled,
        tri_head_hidden=(16,),
    ).to(device)
    return policy


def _shared_param_names(policy):
    """Names of params that exist in BOTH head-on and head-off models —
    encoder MLP + GRU + policy_layer + log_std_parameter."""
    names = []
    for name, _ in policy.named_parameters():
        if name.startswith("tri_head"):
            continue
        names.append(name)
    return names


def _build_synthetic_batch(device, B=4, S=4, obs_dim=47, act_dim=7):
    _seed_all(0)  # batch generation uses its own deterministic seed
    states = torch.randn(B, S, obs_dim, device=device)
    actions = torch.randn(B, S, act_dim, device=device)
    advantages = torch.randn(B * S, device=device)
    log_prob_old = torch.randn(B * S, device=device) * 0.5
    returns = torch.randn(B * S, device=device)
    values_old = torch.randn(B * S, device=device)
    terminated = torch.zeros(B, S, dtype=torch.bool, device=device)
    return states, actions, advantages, log_prob_old, returns, values_old, terminated


def _ppo_step_no_aux(policy, value_layer, batch, device, ratio_clip=0.2,
                     value_clip=0.2, ent_scale=0.01, val_scale=1.0):
    """Run one full PPO update step using only policy/value/entropy losses
    (no aux loss). Returns the gradient dict after backward (no optimizer step).
    """
    states, actions, advantages, log_prob_old, returns, values_old, terminated = batch
    B, S = states.shape[:2]

    # Forward pass through the policy
    flat_actions = actions.flatten(0, 1)
    policy_logits, log_std, policy_outputs = policy.compute(
        {"states": states, "rnn": [None], "terminated": terminated}, role="policy",
    )
    # Build a Gaussian distribution to compute log_prob over the taken actions.
    distribution = torch.distributions.Normal(policy_logits, log_std.exp())
    next_log_prob = distribution.log_prob(flat_actions).sum(dim=-1)

    # Toy "value layer" forward — use the same encoder but a separate head we own
    # so the value loss has a real gradient path through the encoder.
    output, _ = policy.compute_base(
        {"states": states, "rnn": [None], "terminated": terminated},
    )
    predicted_values = value_layer(output).squeeze(-1)

    # Standard PPO terms (matching MAPPO_RNN structure, no episode mask).
    ratio = torch.exp(next_log_prob - log_prob_old)
    surrogate = advantages * ratio
    surrogate_clipped = advantages * torch.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

    # Value clipping (matching MAPPO_RNN)
    predicted_values_clipped = values_old + torch.clip(
        predicted_values - values_old, min=-value_clip, max=value_clip,
    )
    value_loss = val_scale * (returns - predicted_values_clipped).pow(2).mean()

    # Entropy
    entropy = distribution.entropy()
    if entropy.dim() > 1:
        entropy = entropy.sum(dim=-1)
    entropy_loss = -ent_scale * entropy.mean()

    total = policy_loss + value_loss + entropy_loss
    total.backward()


def _ppo_step_with_aux(policy, value_layer, batch, device, *, aux_loss_scale,
                       ratio_clip=0.2, value_clip=0.2, ent_scale=0.01, val_scale=1.0):
    """Same as _ppo_step_no_aux but with the aux NLL term added.
    Used to verify that aux_loss_scale=0.0 produces bit-identical gradients.
    """
    states, actions, advantages, log_prob_old, returns, values_old, terminated = batch
    B, S = states.shape[:2]

    flat_actions = actions.flatten(0, 1)
    policy_logits, log_std, policy_outputs = policy.compute(
        {"states": states, "rnn": [None], "terminated": terminated}, role="policy",
    )
    distribution = torch.distributions.Normal(policy_logits, log_std.exp())
    next_log_prob = distribution.log_prob(flat_actions).sum(dim=-1)

    output, _ = policy.compute_base(
        {"states": states, "rnn": [None], "terminated": terminated},
    )
    predicted_values = value_layer(output).squeeze(-1)

    ratio = torch.exp(next_log_prob - log_prob_old)
    surrogate = advantages * ratio
    surrogate_clipped = advantages * torch.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

    predicted_values_clipped = values_old + torch.clip(
        predicted_values - values_old, min=-value_clip, max=value_clip,
    )
    value_loss = val_scale * (returns - predicted_values_clipped).pow(2).mean()

    entropy = distribution.entropy()
    if entropy.dim() > 1:
        entropy = entropy.sum(dim=-1)
    entropy_loss = -ent_scale * entropy.mean()

    # Aux loss path (mirrors _update logic) — uses synthetic supervision data.
    tri_mu = policy_outputs["tri_mu"]                      # (B*S, 3)
    tri_log_var = policy_outputs["tri_log_var"]            # (B*S, 3)
    tri_target = torch.zeros_like(tri_mu)                  # synthetic target
    tri_valid = torch.ones(B * S, dtype=torch.bool, device=device)
    ep_step = torch.full((B * S,), 100, dtype=torch.int32, device=device)

    aux_loss, _ = compute_aux_nll_loss(
        tri_mu=tri_mu, tri_log_var=tri_log_var,
        tri_target=tri_target, tri_valid=tri_valid,
        episode_step=ep_step, episode_start_mask_steps=0,
        nll_clamp=1000.0,
    )

    total = policy_loss + value_loss + entropy_loss + aux_loss_scale * aux_loss
    total.backward()


def _make_value_layer(device, gru_hidden_size=32, seed=42):
    """Construct a separate Linear value layer to use across both A and B."""
    _seed_all(seed + 1)  # different sub-seed; same in both runs
    return torch.nn.Linear(gru_hidden_size, 1).to(device)


def run_bit_exact_tests(results, device, verbose=False):
    """Two policies with the same seed:
       A: tri_head_enabled=False
       B: tri_head_enabled=True, aux_loss_scale=0.0
       After one PPO update step, gradients on shared params must match exactly.
    """
    try:
        # Build A (head removed)
        policy_a = _build_policy(device, tri_head_enabled=False, seed=42)
        value_a = _make_value_layer(device, gru_hidden_size=32, seed=42)

        # Build B (head present)
        policy_b = _build_policy(device, tri_head_enabled=True, seed=42)
        value_b = _make_value_layer(device, gru_hidden_size=32, seed=42)

        # Confirm encoder + policy_layer + log_std are bit-identical at init.
        for name in _shared_param_names(policy_a):
            pa = dict(policy_a.named_parameters())[name]
            pb = dict(policy_b.named_parameters())[name]
            if not torch.equal(pa, pb):
                raise AssertionError(
                    f"INIT divergence on shared param '{name}' before any update; "
                    f"max-abs-diff={(pa - pb).abs().max().item()}"
                )

        # Same value-layer init must also match (it's the same construction code).
        if not torch.equal(value_a.weight, value_b.weight):
            raise AssertionError("INIT divergence on value_layer.weight")
        if not torch.equal(value_a.bias, value_b.bias):
            raise AssertionError("INIT divergence on value_layer.bias")

        # Build identical synthetic batch.
        batch = _build_synthetic_batch(device)

        # Forward+backward A (no aux head)
        for p in policy_a.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in value_a.parameters():
            if p.grad is not None:
                p.grad.zero_()
        _ppo_step_no_aux(policy_a, value_a, batch, device)

        # Forward+backward B with aux_loss_scale=0
        for p in policy_b.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in value_b.parameters():
            if p.grad is not None:
                p.grad.zero_()
        _ppo_step_with_aux(policy_b, value_b, batch, device, aux_loss_scale=0.0)

        # Compare gradients on shared params.
        a_params = dict(policy_a.named_parameters())
        b_params = dict(policy_b.named_parameters())
        diffs = []
        for name in _shared_param_names(policy_a):
            ga = a_params[name].grad
            gb = b_params[name].grad
            if ga is None and gb is None:
                continue
            if ga is None or gb is None:
                diffs.append(f"{name}: one side has None grad")
                continue
            if not torch.equal(ga, gb):
                max_diff = (ga - gb).abs().max().item()
                diffs.append(f"{name}: max-abs-diff={max_diff}")

        # Value layer
        for name in ("weight", "bias"):
            ga = getattr(value_a, name).grad
            gb = getattr(value_b, name).grad
            if not torch.equal(ga, gb):
                diffs.append(f"value_layer.{name}: max-abs-diff={(ga - gb).abs().max().item()}")

        if diffs:
            raise AssertionError(
                "Bit-exact gradient regression failed on " + str(len(diffs)) + " param(s):\n  "
                + "\n  ".join(diffs)
            )
        results.add_pass("Shared-param gradients bit-exact at aux_loss_scale=0.0")
    except Exception:
        results.add_fail("Shared-param gradients bit-exact at aux_loss_scale=0.0",
                         traceback.format_exc())

    # ---- Sanity: with aux_loss_scale > 0, gradients DO differ ----
    try:
        policy_a = _build_policy(device, tri_head_enabled=False, seed=42)
        value_a = _make_value_layer(device, gru_hidden_size=32, seed=42)
        policy_b = _build_policy(device, tri_head_enabled=True, seed=42)
        value_b = _make_value_layer(device, gru_hidden_size=32, seed=42)
        batch = _build_synthetic_batch(device)

        for p in policy_a.parameters():
            if p.grad is not None: p.grad.zero_()
        for p in value_a.parameters():
            if p.grad is not None: p.grad.zero_()
        _ppo_step_no_aux(policy_a, value_a, batch, device)

        for p in policy_b.parameters():
            if p.grad is not None: p.grad.zero_()
        for p in value_b.parameters():
            if p.grad is not None: p.grad.zero_()
        _ppo_step_with_aux(policy_b, value_b, batch, device, aux_loss_scale=0.1)

        # At least one shared param's gradient should differ now.
        any_diff = False
        a_params = dict(policy_a.named_parameters())
        b_params = dict(policy_b.named_parameters())
        for name in _shared_param_names(policy_a):
            ga = a_params[name].grad
            gb = b_params[name].grad
            if ga is None and gb is None:
                continue
            if not torch.equal(ga, gb):
                any_diff = True
                break
        assert any_diff, (
            "Aux loss with scale=0.1 did NOT change shared-param gradients — the "
            "aux gradient is silently disabled even when scale > 0."
        )
        results.add_pass("Aux loss with scale>0 modifies shared-param gradients (positive control)")
    except Exception:
        results.add_fail(
            "Aux loss with scale>0 modifies shared-param gradients (positive control)",
            traceback.format_exc(),
        )
