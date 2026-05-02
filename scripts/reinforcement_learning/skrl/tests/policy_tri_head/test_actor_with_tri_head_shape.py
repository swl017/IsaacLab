"""Shape + clamp tests for MAPPOWithAuxPolicy.compute().

Verifies:
- policy_logits has the expected (B*S, num_actions) shape after compute_base.
- tri_mu and tri_log_var have shape (B*S, 3).
- tri_log_var values lie inside the configured clamp range.
- With tri_head_enabled=False, no tri keys appear in extra_dict.
"""
import traceback

import gymnasium as gym
import numpy as np
import torch

from mappo_with_aux import MAPPOWithAuxPolicy


def _make_policy(device, *, tri_head_enabled=True, num_envs=4, hidden_size=64,
                 gru_hidden_size=64, gru_num_layers=1, log_var_clamp=(-10.0, 4.0)):
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    policy = MAPPOWithAuxPolicy(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        hidden_size=hidden_size,
        gru_num_layers=gru_num_layers,
        gru_hidden_size=gru_hidden_size,
        num_envs=num_envs,
        sequence_length=8,
        tri_head_enabled=tri_head_enabled,
        tri_head_hidden=(32,),
        tri_log_var_clamp=log_var_clamp,
    ).to(device)
    return policy, obs_space, act_space


def run_shape_tests(results, device, verbose=False):
    # ---- Sequence forward (B=4, S=8) ----
    try:
        policy, obs_space, act_space = _make_policy(device)
        B, S = 4, 8
        states = torch.randn(B, S, obs_space.shape[0], device=device)
        terminated = torch.zeros(B, S, dtype=torch.bool, device=device)
        # First-call hidden state is None; compute_base will allocate zeros.
        out, log_std, extra = policy.compute(
            {"states": states, "rnn": [None], "terminated": terminated}, role="policy"
        )
        # compute_base flattens (B, S, H) -> (B*S, H) for sequence inputs, so
        # policy_layer output is (B*S, num_actions).
        assert out.shape == (B * S, act_space.shape[0]), \
            f"policy_logits shape: expected ({B*S}, {act_space.shape[0]}), got {tuple(out.shape)}"
        assert "tri_mu" in extra and "tri_log_var" in extra, \
            f"extra missing tri keys: keys={list(extra.keys())}"
        assert extra["tri_mu"].shape == (B * S, 3), \
            f"tri_mu shape: expected ({B*S}, 3), got {tuple(extra['tri_mu'].shape)}"
        assert extra["tri_log_var"].shape == (B * S, 3), \
            f"tri_log_var shape: expected ({B*S}, 3), got {tuple(extra['tri_log_var'].shape)}"
        # rnn passthrough should still be present.
        assert "rnn" in extra and isinstance(extra["rnn"], list), \
            f"rnn passthrough missing or wrong type: {type(extra.get('rnn'))}"
        results.add_pass("Sequence forward shape (B, S)")
    except Exception:
        results.add_fail("Sequence forward shape (B, S)", traceback.format_exc())

    # ---- Single-step forward (B only) ----
    try:
        policy, obs_space, act_space = _make_policy(device)
        B = 4
        states = torch.randn(B, obs_space.shape[0], device=device)
        out, log_std, extra = policy.compute(
            {"states": states, "rnn": [None]}, role="policy"
        )
        assert out.shape == (B, act_space.shape[0]), \
            f"policy_logits shape (single-step): expected ({B}, {act_space.shape[0]}), got {tuple(out.shape)}"
        assert extra["tri_mu"].shape == (B, 3), \
            f"tri_mu shape (single-step): expected ({B}, 3), got {tuple(extra['tri_mu'].shape)}"
        assert extra["tri_log_var"].shape == (B, 3), \
            f"tri_log_var shape (single-step): expected ({B}, 3), got {tuple(extra['tri_log_var'].shape)}"
        results.add_pass("Single-step forward shape")
    except Exception:
        results.add_fail("Single-step forward shape", traceback.format_exc())

    # ---- log_var clamp range ----
    try:
        clamp = (-10.0, 4.0)
        policy, obs_space, _ = _make_policy(device, log_var_clamp=clamp)
        # Push the head's pre-clamp output to extreme values via parameter manipulation.
        # We don't know the head's internal weights' magnitudes a priori, so we just
        # forward a large input and verify the output is bounded.
        with torch.no_grad():
            # Inflate the final layer's bias on the log_var dims to push outputs above 4.
            for m in reversed(list(policy.tri_head.modules())):
                if isinstance(m, torch.nn.Linear) and m.out_features == 6:
                    m.bias[3:].fill_(1e3)
                    break
        states = torch.randn(2, 4, obs_space.shape[0], device=device)
        terminated = torch.zeros(2, 4, dtype=torch.bool, device=device)
        _, _, extra = policy.compute(
            {"states": states, "rnn": [None], "terminated": terminated}, role="policy"
        )
        lv = extra["tri_log_var"]
        assert lv.max().item() <= clamp[1] + 1e-6, \
            f"tri_log_var exceeds upper clamp: max={lv.max().item()} > {clamp[1]}"
        assert lv.min().item() >= clamp[0] - 1e-6, \
            f"tri_log_var below lower clamp: min={lv.min().item()} < {clamp[0]}"
        results.add_pass("log_var clamp respected")
    except Exception:
        results.add_fail("log_var clamp respected", traceback.format_exc())

    # ---- tri_head_enabled=False removes the head ----
    try:
        policy, obs_space, _ = _make_policy(device, tri_head_enabled=False)
        assert policy.tri_head is None, "tri_head should be None when disabled"
        states = torch.randn(3, 4, obs_space.shape[0], device=device)
        terminated = torch.zeros(3, 4, dtype=torch.bool, device=device)
        _, _, extra = policy.compute(
            {"states": states, "rnn": [None], "terminated": terminated}, role="policy"
        )
        assert "tri_mu" not in extra and "tri_log_var" not in extra, \
            f"tri keys should be absent when disabled; got {list(extra.keys())}"
        results.add_pass("tri_head_enabled=False removes head")
    except Exception:
        results.add_fail("tri_head_enabled=False removes head", traceback.format_exc())
