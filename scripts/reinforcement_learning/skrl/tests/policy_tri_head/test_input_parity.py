"""Intent 2 — input parity: the head's forward must run on a single agent's
obs in isolation, with no leakage of multi-agent shape requirements.

The ticket's deployment story requires each drone to run the network on its
own observation. This test forwards a per-agent obs of shape (B, S, obs_dim)
or (B, obs_dim) and asserts the head produces the right output shape without
needing any cross-agent tensor.
"""
import traceback

import gymnasium as gym
import numpy as np
import torch

from mappo_with_aux import MAPPOWithAuxPolicy


def run_input_parity_tests(results, device, verbose=False):
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    # ---- Single-agent rollout step (B, obs_dim) ----
    try:
        policy = MAPPOWithAuxPolicy(
            obs_space, act_space, device,
            hidden_size=64, gru_num_layers=1, gru_hidden_size=64,
            num_envs=8, sequence_length=8,
            tri_head_enabled=True, tri_head_hidden=(32,),
        ).to(device)

        B = 8
        states = torch.randn(B, obs_space.shape[0], device=device)
        out, _, extra = policy.compute({"states": states, "rnn": [None]}, role="policy")
        assert out.shape == (B, act_space.shape[0])
        assert extra["tri_mu"].shape == (B, 3)
        assert extra["tri_log_var"].shape == (B, 3)
        results.add_pass("Single-agent rollout step (B, obs_dim)")
    except Exception:
        results.add_fail("Single-agent rollout step (B, obs_dim)", traceback.format_exc())

    # ---- Single-agent training sequence (B, S, obs_dim) ----
    try:
        policy = MAPPOWithAuxPolicy(
            obs_space, act_space, device,
            hidden_size=64, gru_num_layers=1, gru_hidden_size=64,
            num_envs=8, sequence_length=8,
            tri_head_enabled=True, tri_head_hidden=(32,),
        ).to(device)

        B, S = 8, 4
        states = torch.randn(B, S, obs_space.shape[0], device=device)
        terminated = torch.zeros(B, S, dtype=torch.bool, device=device)
        out, _, extra = policy.compute(
            {"states": states, "rnn": [None], "terminated": terminated}, role="policy"
        )
        # compute_base flattens (B, S, H) -> (B*S, H)
        assert out.shape == (B * S, act_space.shape[0])
        assert extra["tri_mu"].shape == (B * S, 3)
        assert extra["tri_log_var"].shape == (B * S, 3)
        results.add_pass("Single-agent training sequence (B, S, obs_dim)")
    except Exception:
        results.add_fail("Single-agent training sequence (B, S, obs_dim)",
                         traceback.format_exc())

    # ---- B=1 corner case (single drone, single env) ----
    try:
        policy = MAPPOWithAuxPolicy(
            obs_space, act_space, device,
            hidden_size=32, gru_num_layers=1, gru_hidden_size=32,
            num_envs=1, sequence_length=4,
            tri_head_enabled=True, tri_head_hidden=(16,),
        ).to(device)

        states = torch.randn(1, obs_space.shape[0], device=device)
        out, _, extra = policy.compute({"states": states, "rnn": [None]}, role="policy")
        assert out.shape == (1, act_space.shape[0])
        assert extra["tri_mu"].shape == (1, 3)
        assert extra["tri_log_var"].shape == (1, 3)
        results.add_pass("B=1 corner case (deployment-shaped input)")
    except Exception:
        results.add_fail("B=1 corner case (deployment-shaped input)",
                         traceback.format_exc())
