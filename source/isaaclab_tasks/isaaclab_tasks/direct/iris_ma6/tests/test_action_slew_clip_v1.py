#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ticket 044 — per-channel slew-rate clip on raw actions.

A single Isaac Sim ``SimulationApp`` is shared across all tests (multiple
``gym.make`` calls in one process are not supported). The env is built with
the slew clip enabled by default; tests toggle ``cfg.enable_action_slew_clip``
at runtime between ``_pre_physics_step`` calls. The flag-off branch must
produce bit-exact pre-patch ``_actions``.

Coverage:
  1. cfg defaults (B1 / PX4-strict).
  2. Bit-exact flag-off regression.
  3. Slew enforcement (‖Δa‖_∞ ≤ δ_max) under random sequences.
  4. Composition with the existing [-1, 1] clamp at the boundary.
  5. Post-reset first-action bounded (validates the _last_actions zero_() bug fix).
  6. cmd_vel_delta accumulator correctness.
  7. slew_saturation_acc per-channel correctness.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ticket 044 slew-clip tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test_cfg import IrisMA6TestEnvCfg

TASK = "Isaac-Iris-MA6-Direct-Test-v0"


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}", flush=True)

    def add_fail(self, name: str, msg: str):
        self.failed.append((name, msg))
        print(f"  ✗ {name}", flush=True)
        for line in msg.split("\n")[:25]:
            print(f"      {line}", flush=True)

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 80, flush=True)
        print("TEST SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}", flush=True)
        if self.failed:
            print("\nFAILED:", flush=True)
            for n, m in self.failed:
                print(f"  {n}: {m.splitlines()[0] if m else ''}", flush=True)
        print("=" * 80, flush=True)
        return not self.failed


def _make_actions(unwrapped, vals: list[float]) -> dict:
    """Build per-agent action dict with the same 7D vector across all envs."""
    actions = {}
    vec = torch.tensor(vals, dtype=torch.float32, device=unwrapped.device)
    for agent_id in unwrapped.cfg.possible_agents:
        a = vec.unsqueeze(0).expand(unwrapped.num_envs, -1).clone()
        actions[agent_id] = a
    return actions


def _slew_max_vec(cfg) -> torch.Tensor:
    """Per-channel δ_max as a (7,) tensor matching the env's _action_slew_max."""
    return torch.tensor([
        cfg.action_slew_vel_xy,
        cfg.action_slew_vel_xy,
        cfg.action_slew_vel_z,
        cfg.action_slew_yaw_rate,
        cfg.action_slew_gimbal_yaw_rate,
        cfg.action_slew_gimbal_pitch_rate,
        cfg.action_slew_zoom_rate,
    ], dtype=torch.float32)


# ===========================================================================
# Category 1 — cfg defaults (no env)
# ===========================================================================

def test_cfg_defaults_b1(r: TestResults):
    """Default cfg matches the PX4-strict (B1) values from the ticket."""
    try:
        cfg = IrisMA6TestEnvCfg()
        assert cfg.enable_action_slew_clip is True, \
            f"default enable_action_slew_clip should be True, got {cfg.enable_action_slew_clip}"
        # PX4-strict per ticket 044
        assert cfg.action_slew_vel_xy == 0.020, f"vel_xy={cfg.action_slew_vel_xy}"
        assert cfg.action_slew_vel_z == 0.053, f"vel_z={cfg.action_slew_vel_z}"
        assert cfg.action_slew_yaw_rate == 0.30, f"yaw_rate={cfg.action_slew_yaw_rate}"
        assert cfg.action_slew_gimbal_yaw_rate == 0.40
        assert cfg.action_slew_gimbal_pitch_rate == 0.40
        assert cfg.action_slew_zoom_rate == 0.20
        r.add_pass("cfg defaults: enable=True, PX4-strict δ values match B1")
    except Exception:
        r.add_fail("cfg_defaults_b1", traceback.format_exc())


# ===========================================================================
# Category 2 — env-side tests (single SimulationApp)
# ===========================================================================

def test_env_slew_max_buffer(r: TestResults, unwrapped):
    """_action_slew_max is shape (7,) and matches cfg."""
    try:
        expected = _slew_max_vec(unwrapped.cfg).to(unwrapped.device)
        assert unwrapped._action_slew_max.shape == (7,), \
            f"_action_slew_max shape {unwrapped._action_slew_max.shape}"
        assert torch.allclose(unwrapped._action_slew_max, expected, atol=1e-7), \
            f"_action_slew_max {unwrapped._action_slew_max} != expected {expected}"
        r.add_pass(f"_action_slew_max buffer: shape (7,), values {unwrapped._action_slew_max.tolist()}")
    except Exception:
        r.add_fail("env_slew_max_buffer", traceback.format_exc())


def test_flag_off_bit_exact_regression(r: TestResults, unwrapped):
    """With enable_action_slew_clip=False, _actions = clamp(action, -1, 1) exactly."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_action_slew_clip = False
        # _last_actions is from previous test — overwrite with a known value to
        # confirm the slew clip is NOT being applied. If it were, _actions would
        # be pulled toward _last_actions.
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].zero_()

        # Action that would otherwise be heavily clipped by slew:
        # action = [1.0]*7, _last_actions = 0 → slew clip would force action to
        # [+δ_max]. With flag off, action stays at [+1.0] (post-[-1,1] clamp).
        actions = _make_actions(unwrapped, [1.0] * 7)
        unwrapped._pre_physics_step(actions)
        for agent_id in cfg.possible_agents:
            got = unwrapped._actions[agent_id]
            expected = torch.ones_like(got)
            assert torch.equal(got, expected), \
                f"{agent_id}: flag-off path mutated _actions: got {got[0]} expected {expected[0]}"

        # Also test a negative-edge case (action = -1.0)
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].zero_()
        actions = _make_actions(unwrapped, [-1.0] * 7)
        unwrapped._pre_physics_step(actions)
        for agent_id in cfg.possible_agents:
            got = unwrapped._actions[agent_id]
            expected = -torch.ones_like(got)
            assert torch.equal(got, expected), \
                f"{agent_id}: flag-off path mutated _actions (neg edge)"

        r.add_pass("flag-off bit-exact: _actions = clamp(action, -1, 1) ± δ ignored")
    except Exception:
        r.add_fail("flag_off_bit_exact_regression", traceback.format_exc())


def test_slew_enforcement_random_sequence(r: TestResults, unwrapped):
    """With slew clip on, ‖_actions[t] - _last_actions[t-1]‖_∞ ≤ δ_max."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_action_slew_clip = True
        # Zero _last_actions to start clean
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].zero_()
        slew_max = _slew_max_vec(cfg).to(unwrapped.device)  # (7,)

        # 5 random action steps; verify Δa ≤ δ_max at every step
        torch.manual_seed(123)
        for step in range(5):
            actions = {}
            for agent_id in cfg.possible_agents:
                a = torch.empty(unwrapped.num_envs, 7, device=unwrapped.device).uniform_(-1.0, 1.0)
                actions[agent_id] = a
            unwrapped._pre_physics_step(actions)
            for agent_id in cfg.possible_agents:
                got = unwrapped._actions[agent_id]
                # Δa relative to the buffer that was current AT the time of clip.
                # After _pre_physics_step, _actions holds the post-clip action; we
                # validate by reconstructing the constraint: |got - last_before_step| ≤ δ.
                # _last_actions hasn't been updated yet in this flow (that happens
                # in _get_rewards), so it still holds the prev step's value.
                diff = (got - unwrapped._last_actions[agent_id]).abs()
                # Allow ULP-level slack for fp32 clamp
                violation = (diff - slew_max.unsqueeze(0)).max().item()
                assert violation <= 1e-6, \
                    f"step {step} {agent_id}: max Δa violation {violation:.6f} > δ_max"
            # Simulate the _get_rewards update of _last_actions
            for agent_id in cfg.possible_agents:
                unwrapped._last_actions[agent_id] = unwrapped._actions[agent_id].clone()
        r.add_pass("slew enforcement: ‖Δa‖_∞ ≤ δ_max over 5 random steps")
    except Exception:
        r.add_fail("slew_enforcement_random_sequence", traceback.format_exc())


def test_clamp_composition(r: TestResults, unwrapped):
    """When _last_actions=0.95 and δ=0.30, effective range is [0.65, 1.0].
    The slew clip's upper bound (1.25) is overridden by the [-1, 1] clamp."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_action_slew_clip = True
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].fill_(0.95)
        # vel_xy δ = 0.020; we'll test on yaw_rate (δ=0.30) since that's where
        # the boundary effect is most visible.
        # Action = 1.5 (out of [-1, 1]) for yaw_rate (ch 3); should clamp to 1.0
        # then slew-clip to min(1.0, 0.95+0.30) = 1.0.
        actions = _make_actions(unwrapped, [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])
        unwrapped._pre_physics_step(actions)
        for agent_id in cfg.possible_agents:
            got_yaw = unwrapped._actions[agent_id][:, 3]
            # Expected: clamp(1.5, -1, 1) = 1.0 → slew clamp(1.0, 0.95-0.30, 0.95+0.30) = 1.0
            assert torch.allclose(got_yaw, torch.full_like(got_yaw, 1.0), atol=1e-6), \
                f"{agent_id}: yaw_rate composition got {got_yaw[0]:.4f} expected 1.0"
        # Now the OPPOSITE edge: _last=-0.95, action=-1.5 → clamp(-1.5,-1,1) = -1.0
        # → slew clamp(-1.0, -0.95-0.30, -0.95+0.30) = -1.0
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].fill_(-0.95)
        actions = _make_actions(unwrapped, [0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0])
        unwrapped._pre_physics_step(actions)
        for agent_id in cfg.possible_agents:
            got_yaw = unwrapped._actions[agent_id][:, 3]
            assert torch.allclose(got_yaw, torch.full_like(got_yaw, -1.0), atol=1e-6), \
                f"{agent_id}: yaw_rate neg-edge composition got {got_yaw[0]:.4f} expected -1.0"
        r.add_pass("clamp composition: [-1, 1] dominates the slew bound at the boundary")
    except Exception:
        r.add_fail("clamp_composition", traceback.format_exc())


def test_post_reset_first_action_bounded(r: TestResults, unwrapped):
    """After reset, _last_actions[env_ids] == 0, so first action ≤ δ_max."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_action_slew_clip = True
        # Pollute _last_actions to a non-zero value so the bug fix has work to do.
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].fill_(0.7)

        # Reset all envs
        N = unwrapped.num_envs
        env_ids = torch.arange(N, device=unwrapped.device)
        unwrapped._reset_idx(env_ids)

        # After reset, _last_actions MUST be zero for the reset envs.
        # This is the load-bearing assertion for the zero_() → = 0.0 bug fix.
        for agent_id in cfg.possible_agents:
            assert torch.equal(
                unwrapped._last_actions[agent_id],
                torch.zeros_like(unwrapped._last_actions[agent_id]),
            ), f"{agent_id}: _last_actions not zeroed post-reset"

        # First action with magnitude 1.0 → should be clipped to ±δ_max.
        slew_max = _slew_max_vec(cfg).to(unwrapped.device)
        actions = _make_actions(unwrapped, [1.0] * 7)
        unwrapped._pre_physics_step(actions)
        for agent_id in cfg.possible_agents:
            got = unwrapped._actions[agent_id]
            expected = slew_max.unsqueeze(0).expand_as(got)
            assert torch.allclose(got, expected, atol=1e-6), \
                f"{agent_id}: post-reset action {got[0]} != δ_max {expected[0]}"
        r.add_pass("post-reset first action bounded to ±δ_max (validates _last_actions reset fix)")
    except Exception:
        r.add_fail("post_reset_first_action_bounded", traceback.format_exc())


def test_slew_saturation_metric(r: TestResults, unwrapped):
    """slew_saturation_acc counts per-channel steps where the clip moved the action."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_action_slew_clip = True
        # Reset to start clean
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].zero_()
            unwrapped._slew_saturation_acc[agent_id].zero_()

        # Action far in excess of δ_max on vel channels → 100% saturation.
        # All-zeros on others → 0% saturation.
        actions = _make_actions(unwrapped, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        unwrapped._pre_physics_step(actions)

        for agent_id in cfg.possible_agents:
            sat = unwrapped._slew_saturation_acc[agent_id]  # (N, 7)
            # ch 0,1,2 saturated (action 1.0, _last 0 → would need clip)
            for ch in [0, 1, 2]:
                assert torch.all(sat[:, ch] == 1.0), \
                    f"{agent_id}: ch {ch} saturation acc should be 1.0, got {sat[0, ch]}"
            # ch 3,4,5,6 not saturated (action 0.0, _last 0 → no clip needed)
            for ch in [3, 4, 5, 6]:
                assert torch.all(sat[:, ch] == 0.0), \
                    f"{agent_id}: ch {ch} saturation acc should be 0, got {sat[0, ch]}"
        r.add_pass("slew_saturation_acc: per-channel correctness on a known input")
    except Exception:
        r.add_fail("slew_saturation_metric", traceback.format_exc())


def test_cmd_vel_delta_metric(r: TestResults, unwrapped):
    """_cmd_vel_delta_acc accumulates |cmd_vel[t] - cmd_vel[t-1]| per step."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_action_slew_clip = True
        # Reset accumulators + buffers
        unwrapped._cmd_vel_delta_acc.zero_()
        unwrapped._cmd_vel_prev.zero_()
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id].zero_()

        # First step: cmd_vel = some non-zero value → delta = cmd_vel (since prev was zero)
        actions = _make_actions(unwrapped, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        unwrapped._pre_physics_step(actions)
        cmd_vel_after_first = unwrapped.cmd_vel.clone()
        # Accumulator should equal |cmd_vel_after_first - 0|
        expected = cmd_vel_after_first.abs()
        assert torch.allclose(unwrapped._cmd_vel_delta_acc, expected, atol=1e-6), \
            f"first-step accumulator mismatch: {unwrapped._cmd_vel_delta_acc[0,0]} vs {expected[0,0]}"

        # _cmd_vel_prev now holds cmd_vel_after_first.
        assert torch.allclose(unwrapped._cmd_vel_prev, cmd_vel_after_first, atol=1e-6), \
            "_cmd_vel_prev not updated to cmd_vel_after_first"

        # Second step: same action → cmd_vel diff should equal slew increment.
        # On first call _last=0, action=1 → clipped to δ_xy=0.02. _last_actions
        # hasn't been updated (no _get_rewards), so this also clips to 0.02 again.
        # In this artificial test (no reward update between steps), _actions stays
        # the same and cmd_vel stays the same → cmd_vel_delta accumulates 0 on the
        # second step.
        # To make this test meaningful, simulate the reward path's _last_actions update:
        for agent_id in cfg.possible_agents:
            unwrapped._last_actions[agent_id] = unwrapped._actions[agent_id].clone()
        # Now apply another action; new cmd_vel will be different.
        actions = _make_actions(unwrapped, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        unwrapped._pre_physics_step(actions)
        cmd_vel_after_second = unwrapped.cmd_vel.clone()
        # Increment accumulator by |cmd_vel_after_second - cmd_vel_after_first|.
        expected_inc = (cmd_vel_after_second - cmd_vel_after_first).abs()
        total_expected = cmd_vel_after_first.abs() + expected_inc
        assert torch.allclose(unwrapped._cmd_vel_delta_acc, total_expected, atol=1e-6), \
            f"second-step accumulator mismatch"
        r.add_pass("cmd_vel_delta_acc: |Δ cmd_vel| accumulation correct over 2 steps")
    except Exception:
        r.add_fail("cmd_vel_delta_metric", traceback.format_exc())


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 80, flush=True)
    print("TICKET 044 — PER-CHANNEL SLEW-RATE CLIP TEST SUITE", flush=True)
    print("=" * 80, flush=True)

    r = TestResults()

    print("\n--- Category 1 — cfg defaults (no env) ---", flush=True)
    test_cfg_defaults_b1(r)

    cfg = IrisMA6TestEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 42
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    unwrapped = env.unwrapped

    try:
        print("\n--- Category 2a — env buffer ---", flush=True)
        test_env_slew_max_buffer(r, unwrapped)

        print("\n--- Category 2b — flag-off bit-exact regression ---", flush=True)
        test_flag_off_bit_exact_regression(r, unwrapped)

        print("\n--- Category 2c — slew enforcement (random sequence) ---", flush=True)
        test_slew_enforcement_random_sequence(r, unwrapped)

        print("\n--- Category 2d — clamp composition at the boundary ---", flush=True)
        test_clamp_composition(r, unwrapped)

        print("\n--- Category 2e — post-reset first-action bounded (validates bug fix) ---", flush=True)
        test_post_reset_first_action_bounded(r, unwrapped)

        print("\n--- Category 2f — slew saturation metric ---", flush=True)
        test_slew_saturation_metric(r, unwrapped)

        print("\n--- Category 2g — cmd_vel_delta metric ---", flush=True)
        test_cmd_vel_delta_metric(r, unwrapped)
    finally:
        env.close()

    success = r.print_summary()
    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
