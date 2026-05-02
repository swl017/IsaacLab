"""Regression guard for Intent 1a: action head must NOT consume tri_head outputs.

Asserts policy_layer.in_features equals gru_hidden_size — not gru_hidden_size + 6.
If this test ever fails, someone wired the tri head's output into the action head
input (Intent 1b), which is intentionally out of scope for this ticket.
"""
import traceback

import gymnasium as gym
import numpy as np
import torch

from mappo_with_aux import MAPPOWithAuxPolicy


def run_no_skip_tests(results, device, verbose=False):
    try:
        gru_hidden_size = 64
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        policy = MAPPOWithAuxPolicy(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            hidden_size=64,
            gru_num_layers=1,
            gru_hidden_size=gru_hidden_size,
            num_envs=4,
            sequence_length=8,
            tri_head_enabled=True,
            tri_head_hidden=(32,),
        ).to(device)

        in_features = policy.policy_layer.in_features
        assert in_features == gru_hidden_size, (
            f"policy_layer.in_features={in_features}, expected {gru_hidden_size}. "
            f"If this is gru_hidden_size + 6 ({gru_hidden_size + 6}), the action head is "
            f"consuming tri_head outputs (Intent 1b) — out of scope for ticket 031."
        )
        assert in_features != gru_hidden_size + 6, (
            "policy_layer.in_features matches the skip-connection signature; "
            "Intent 1b is rejected for this ticket."
        )
        results.add_pass("policy_layer.in_features == gru_hidden_size")
    except Exception:
        results.add_fail("policy_layer.in_features == gru_hidden_size", traceback.format_exc())

    # The tri head reads from gru_hidden_size as well (same point as the action head).
    try:
        # tri_head is nn.Sequential(Linear, ELU, Linear); the first Linear's in_features
        # should be gru_hidden_size.
        first_linear = None
        for m in policy.tri_head.modules():
            if isinstance(m, torch.nn.Linear):
                first_linear = m
                break
        assert first_linear is not None, "tri_head has no Linear layers"
        assert first_linear.in_features == gru_hidden_size, (
            f"tri_head first-layer in_features={first_linear.in_features}, "
            f"expected {gru_hidden_size}"
        )
        results.add_pass("tri_head first-layer reads from gru hidden state")
    except Exception:
        results.add_fail("tri_head first-layer reads from gru hidden state",
                         traceback.format_exc())
