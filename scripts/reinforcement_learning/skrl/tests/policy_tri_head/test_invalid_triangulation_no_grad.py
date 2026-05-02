"""When the validity mask is all-False, the auxiliary loss must produce zero
gradient on the tri_head's parameters. This guards against the failure mode
where the loss is computed but happens to be technically nonzero in some
degenerate way that leaks gradient.
"""
import traceback

import gymnasium as gym
import numpy as np
import torch

from mappo_with_aux import MAPPOWithAuxPolicy, compute_aux_nll_loss


def run_no_grad_tests(results, device, verbose=False):
    try:
        torch.manual_seed(0)
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        policy = MAPPOWithAuxPolicy(
            obs_space, act_space, device,
            hidden_size=64, gru_num_layers=1, gru_hidden_size=64,
            num_envs=4, sequence_length=8,
            tri_head_enabled=True, tri_head_hidden=(32,),
        ).to(device)

        # Forward to produce tri_mu, tri_log_var with grad-tracking.
        B, S = 4, 8
        states = torch.randn(B, S, obs_space.shape[0], device=device)
        terminated = torch.zeros(B, S, dtype=torch.bool, device=device)
        out, _, extra = policy.compute(
            {"states": states, "rnn": [None], "terminated": terminated}, role="policy"
        )
        tri_mu = extra["tri_mu"]
        tri_log_var = extra["tri_log_var"]

        # All-False mask -> aux_loss = 0 and gradient should be zero everywhere.
        N = B * S
        target = torch.randn(N, 3, device=device)
        valid = torch.zeros(N, dtype=torch.bool, device=device)
        episode_step = torch.full((N,), 100, dtype=torch.int64, device=device)

        loss, _ = compute_aux_nll_loss(
            tri_mu, tri_log_var, target, valid, episode_step,
            episode_start_mask_steps=0, nll_clamp=1e9,
        )
        assert loss.item() == 0.0, f"all-False mask should give zero loss, got {loss.item()}"

        # Zero existing grads, then backward through the (zero) aux loss.
        for p in policy.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()

        # Tri head param gradients should all be zero (or None — same effect).
        tri_grad_norm = 0.0
        for p in policy.tri_head.parameters():
            if p.grad is not None:
                tri_grad_norm += p.grad.detach().pow(2).sum().item()
        assert tri_grad_norm == 0.0, \
            f"tri_head grad leaked under all-False mask: ‖grad‖²={tri_grad_norm}"

        # Encoder/action params likewise should not receive gradient from a zero loss.
        encoder_grad_norm = 0.0
        for p in list(policy.net.parameters()) + list(policy.gru.parameters()):
            if p.grad is not None:
                encoder_grad_norm += p.grad.detach().pow(2).sum().item()
        assert encoder_grad_norm == 0.0, \
            f"encoder grad leaked under all-False aux mask: ‖grad‖²={encoder_grad_norm}"

        results.add_pass("All-False mask: zero gradient on tri_head AND encoder")
    except Exception:
        results.add_fail("All-False mask: zero gradient on tri_head AND encoder",
                         traceback.format_exc())

    # ---- Sanity: with a valid mask, gradient IS produced ----
    try:
        torch.manual_seed(1)
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        policy = MAPPOWithAuxPolicy(
            obs_space, act_space, device,
            hidden_size=64, gru_num_layers=1, gru_hidden_size=64,
            num_envs=4, sequence_length=8,
            tri_head_enabled=True, tri_head_hidden=(32,),
        ).to(device)

        B, S = 4, 8
        states = torch.randn(B, S, obs_space.shape[0], device=device)
        terminated = torch.zeros(B, S, dtype=torch.bool, device=device)
        _, _, extra = policy.compute(
            {"states": states, "rnn": [None], "terminated": terminated}, role="policy"
        )
        tri_mu = extra["tri_mu"]
        tri_log_var = extra["tri_log_var"]

        N = B * S
        target = tri_mu.detach() + 1.0 + torch.randn(N, 3, device=device)  # nonzero error
        valid = torch.ones(N, dtype=torch.bool, device=device)
        episode_step = torch.full((N,), 100, dtype=torch.int64, device=device)

        for p in policy.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss, _ = compute_aux_nll_loss(
            tri_mu, tri_log_var, target, valid, episode_step,
            episode_start_mask_steps=0, nll_clamp=1e9,
        )
        assert loss.item() > 0.0, "valid mask + nonzero error should give positive loss"
        loss.backward()

        tri_grad_norm = 0.0
        for p in policy.tri_head.parameters():
            if p.grad is not None:
                tri_grad_norm += p.grad.detach().pow(2).sum().item()
        assert tri_grad_norm > 0.0, \
            f"valid mask should produce nonzero tri_head grad, got ‖grad‖²={tri_grad_norm}"
        results.add_pass("Valid mask + nonzero error: gradient flows to tri_head")
    except Exception:
        results.add_fail("Valid mask + nonzero error: gradient flows to tri_head",
                         traceback.format_exc())
