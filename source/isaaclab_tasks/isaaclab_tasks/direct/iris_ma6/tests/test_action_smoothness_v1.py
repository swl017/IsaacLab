#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ticket 043 — prev-action obs + first-order LP on cmd_vel.

A single Isaac Sim ``SimulationApp`` is shared across all tests (multiple
``gym.make`` calls in one process are not supported). The env is built with
both feature flags ON so the observation space includes the prev-action
tail; the LP / prev-action runtime branches are exercised by toggling
``cfg.enable_action_lowpass`` and ``cfg.enable_prev_action_obs`` at runtime
between ``_pre_physics_step`` calls. The flags-off branch produces
``cmd_vel = action * scale`` byte-for-byte — the bit-exact regression
acceptance criterion.

Pure-math tests (LP convergence, very-large-τ first-step attenuation,
sampling-rate invariance) run on the EMA formula directly and do not
require env construction.
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ticket 043 action smoothness tests")
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


# ===========================================================================
# Category 1 — cfg defaults (no env build needed)
# ===========================================================================

def test_cfg_defaults(r: TestResults):
    """Default cfg flips both flags on with the ticket-043 τ defaults."""
    try:
        cfg = IrisMA6TestEnvCfg()
        assert cfg.enable_prev_action_obs is True, \
            f"default enable_prev_action_obs should be True, got {cfg.enable_prev_action_obs}"
        assert cfg.enable_action_lowpass is True, \
            f"default enable_action_lowpass should be True, got {cfg.enable_action_lowpass}"
        assert cfg.action_lowpass_tau_vel_xy_s == 0.08, \
            f"τ_vel_xy={cfg.action_lowpass_tau_vel_xy_s}"
        assert cfg.action_lowpass_tau_vel_z_s == 0.08
        assert cfg.action_lowpass_tau_yaw_rate_s == 0.08
        assert cfg.action_lowpass_tau_gimbal_yaw_rate_s == 0.04
        assert cfg.action_lowpass_tau_gimbal_pitch_rate_s == 0.04
        assert cfg.action_lowpass_tau_zoom_rate_s == 0.10
        r.add_pass("cfg defaults: flags=True, τ values match ticket 043")
    except Exception:
        r.add_fail("cfg_defaults", traceback.format_exc())


def test_cfg_obs_dim_accounting(r: TestResults):
    """obs_dim follows: base 47 (A=2) + 7 (prev-action) [+6 if triangulation]."""
    try:
        # Both flags on, no triangulation
        cfg = IrisMA6TestEnvCfg()
        cfg.num_agents = 2
        cfg.enable_triangulation = False
        cfg.enable_prev_action_obs = True
        cfg.__post_init__()
        assert cfg.observation_spaces["drone_0"] == 31 + 16 + 7, \
            f"prev-action on, tri off, A=2: expected 54, got {cfg.observation_spaces['drone_0']}"

        # Bit-exact regression: both flags off, no triangulation → 47 (pre-patch)
        cfg = IrisMA6TestEnvCfg()
        cfg.num_agents = 2
        cfg.enable_triangulation = False
        cfg.enable_prev_action_obs = False
        cfg.enable_action_lowpass = False
        cfg.__post_init__()
        assert cfg.observation_spaces["drone_0"] == 31 + 16, \
            f"flags off, A=2: expected 47 (pre-patch), got {cfg.observation_spaces['drone_0']}"

        # Triangulation + prev-action: 47 + 6 + 7 = 60
        cfg = IrisMA6TestEnvCfg()
        cfg.num_agents = 2
        cfg.enable_triangulation = True
        cfg.enable_prev_action_obs = True
        cfg.__post_init__()
        assert cfg.observation_spaces["drone_0"] == 31 + 16 + 6 + 7, \
            f"prev-action + tri, A=2: expected 60, got {cfg.observation_spaces['drone_0']}"

        r.add_pass("cfg obs_dim: prev-action-off → 47 (regression); on → 54 (+ tri = 60)")
    except Exception:
        r.add_fail("cfg_obs_dim_accounting", traceback.format_exc())


# ===========================================================================
# Category 2 — pure-math tests on the EMA formula (no env)
# ===========================================================================

def test_lp_formula_convergence(r: TestResults):
    """EMA converges to the raw command after ~4τ/dt steps within 2%."""
    try:
        dt = 0.04
        tau = 0.08
        alpha = 1.0 - math.exp(-dt / tau)
        target = 1.0
        x = 0.0
        n_steps_to_4tau = int(round(4.0 * tau / dt))  # = 8
        for _ in range(n_steps_to_4tau):
            x = alpha * target + (1.0 - alpha) * x
        # After 4τ, an exact first-order filter reaches 1 − e^-4 ≈ 0.9817
        expected_4tau = 1.0 - math.exp(-4.0)
        assert abs(x - expected_4tau) < 1e-3, \
            f"after 4τ EMA at {x:.4f}, expected ≈ {expected_4tau:.4f}"
        r.add_pass(f"LP convergence: x→{x:.4f} after 4τ ({n_steps_to_4tau} steps) ≈ 1−e⁻⁴")
    except Exception:
        r.add_fail("lp_formula_convergence", traceback.format_exc())


def test_lp_large_tau_attenuates_first_step(r: TestResults):
    """τ=1.0 s, dt=0.04 → first-step output ≈ 0.04 × command (≪ 1)."""
    try:
        dt = 0.04
        tau = 1.0
        alpha = 1.0 - math.exp(-dt / tau)
        assert alpha < 0.05, f"α should be <0.05 for τ=1, dt=0.04; got {alpha}"
        out = alpha * 1.0 + (1.0 - alpha) * 0.0
        assert out < 0.05, f"first-step output should be ≪ command; got {out}"
        # Long-run: after ≈ 4τ = 100 steps, near command
        x = 0.0
        for _ in range(int(4.0 * tau / dt)):
            x = alpha * 1.0 + (1.0 - alpha) * x
        assert abs(x - (1.0 - math.exp(-4.0))) < 1e-3, \
            f"long-run mismatch: x={x:.4f}"
        r.add_pass(f"large-τ attenuation: α={alpha:.4f} → first step {out:.4f}, ~4τ converges")
    except Exception:
        r.add_fail("lp_large_tau_attenuates_first_step", traceback.format_exc())


def test_lp_sampling_rate_invariance(r: TestResults):
    """Step response at dt=0.04 (1× decimation) and dt=0.02 (2× decimation)
    match at the same physical time within 1e-3."""
    try:
        tau = 0.08
        physical_t_end = 0.4  # 5 τ

        # Coarse: dt = 0.04, 10 steps
        dt1 = 0.04
        a1 = 1.0 - math.exp(-dt1 / tau)
        x1 = 0.0
        n1 = int(round(physical_t_end / dt1))
        for _ in range(n1):
            x1 = a1 * 1.0 + (1.0 - a1) * x1

        # Fine: dt = 0.02, 20 steps
        dt2 = 0.02
        a2 = 1.0 - math.exp(-dt2 / tau)
        x2 = 0.0
        n2 = int(round(physical_t_end / dt2))
        for _ in range(n2):
            x2 = a2 * 1.0 + (1.0 - a2) * x2

        diff = abs(x1 - x2)
        assert diff < 1e-12, \
            f"sampling-rate invariance broken: x1={x1:.6f} x2={x2:.6f} diff={diff:.2e}"
        # Both equal the closed-form 1 − exp(−t/τ)
        closed = 1.0 - math.exp(-physical_t_end / tau)
        assert abs(x1 - closed) < 1e-12, f"x1={x1} vs closed-form {closed}"
        r.add_pass(
            f"sampling-rate invariance: dt={dt1} → {x1:.6f}, dt={dt2} → {x2:.6f}, "
            f"both = 1−exp(−t/τ)"
        )
    except Exception:
        r.add_fail("lp_sampling_rate_invariance", traceback.format_exc())


# ===========================================================================
# Category 3 — env-side tests (single SimulationApp)
# ===========================================================================

def test_env_buffers_and_alpha(r: TestResults, unwrapped):
    """_cmd_vel_filt has correct shape/dtype; _lp_alpha matches formula."""
    try:
        N = unwrapped.num_envs
        A = len(unwrapped.cfg.possible_agents)
        assert unwrapped._cmd_vel_filt.shape == (N, A, 7), \
            f"_cmd_vel_filt shape {unwrapped._cmd_vel_filt.shape} != ({N}, {A}, 7)"
        assert unwrapped._lp_alpha.shape == (7,), \
            f"_lp_alpha shape {unwrapped._lp_alpha.shape} != (7,)"
        dt = float(unwrapped.cfg.sim.dt) * int(unwrapped.cfg.decimation)
        expected = torch.tensor([
            1.0 - math.exp(-dt / unwrapped.cfg.action_lowpass_tau_vel_xy_s),
            1.0 - math.exp(-dt / unwrapped.cfg.action_lowpass_tau_vel_xy_s),
            1.0 - math.exp(-dt / unwrapped.cfg.action_lowpass_tau_vel_z_s),
            1.0 - math.exp(-dt / unwrapped.cfg.action_lowpass_tau_yaw_rate_s),
            1.0 - math.exp(-dt / unwrapped.cfg.action_lowpass_tau_gimbal_yaw_rate_s),
            1.0 - math.exp(-dt / unwrapped.cfg.action_lowpass_tau_gimbal_pitch_rate_s),
            1.0 - math.exp(-dt / unwrapped.cfg.action_lowpass_tau_zoom_rate_s),
        ], dtype=torch.float32, device=unwrapped.device)
        assert torch.allclose(unwrapped._lp_alpha, expected, atol=1e-6), \
            f"_lp_alpha {unwrapped._lp_alpha} != expected {expected}"
        r.add_pass(f"buffers + α: _cmd_vel_filt={tuple(unwrapped._cmd_vel_filt.shape)}, α={unwrapped._lp_alpha.tolist()}")
    except Exception:
        r.add_fail("env_buffers_and_alpha", traceback.format_exc())


def test_env_flags_off_regression(r: TestResults, unwrapped):
    """Both flags OFF: cmd_vel = action * scale exactly; _cmd_vel_filt untouched."""
    try:
        cfg = unwrapped.cfg
        # Snapshot prior _cmd_vel_filt (zeros at start; reset zeroed it).
        unwrapped._cmd_vel_filt.zero_()
        snap_filt = unwrapped._cmd_vel_filt.clone()

        cfg.enable_prev_action_obs = False
        cfg.enable_action_lowpass = False

        # Action: full-range across all 7 dims, with sign mix on vz.
        actions = _make_actions(unwrapped, [0.5, -0.3, 0.7, 0.4, 0.6, -0.2, 0.8])
        unwrapped._pre_physics_step(actions)

        mlv = unwrapped._max_lin_vel  # (N,)
        max_yaw = cfg.max_yaw_rate
        for idx, _ in enumerate(cfg.possible_agents):
            got = unwrapped.cmd_vel[:, idx, :]
            # vx / vy
            assert torch.allclose(got[:, 0], 0.5 * mlv, atol=1e-6), \
                f"vx mismatch agent {idx}"
            assert torch.allclose(got[:, 1], -0.3 * mlv, atol=1e-6), \
                f"vy mismatch agent {idx}"
            # vz — under asymmetric z envelope (default True), positive z is
            # scaled by max_vel_z_up, not _max_lin_vel.
            if cfg.enable_asymmetric_z_envelope:
                expected_vz = torch.full_like(got[:, 2], 0.7 * cfg.max_vel_z_up)
            else:
                expected_vz = 0.7 * mlv
            assert torch.allclose(got[:, 2], expected_vz, atol=1e-6), \
                f"vz mismatch agent {idx}: got {got[:, 2][0].item()} vs {expected_vz[0].item() if hasattr(expected_vz, 'shape') else expected_vz}"
            assert torch.allclose(got[:, 3], torch.full_like(got[:, 3], 0.4 * max_yaw), atol=1e-6), \
                f"yaw_rate mismatch"
            for ch, val in zip([4, 5, 6], [0.6, -0.2, 0.8]):
                assert torch.allclose(got[:, ch], torch.full_like(got[:, ch], val), atol=1e-6), \
                    f"normalized ch {ch} mismatch"

        # _cmd_vel_filt should be UNTOUCHED (still snapshot value)
        assert torch.equal(unwrapped._cmd_vel_filt, snap_filt), \
            "flags-off path mutated _cmd_vel_filt"

        r.add_pass("flags-off regression: cmd_vel matches action*scale; _cmd_vel_filt untouched")
    except Exception:
        r.add_fail("env_flags_off_regression", traceback.format_exc())


def test_env_lp_on_first_step(r: TestResults, unwrapped):
    """LP ON, prev-action ON: first step → cmd_vel = α × raw, _cmd_vel_filt matches."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_prev_action_obs = True
        cfg.enable_action_lowpass = True

        unwrapped._cmd_vel_filt.zero_()
        actions = _make_actions(unwrapped, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Capture raw cmd_vel that *would* be written before LP overwrites it.
        unwrapped._pre_physics_step(actions)

        mlv = unwrapped._max_lin_vel  # (N,)
        alpha = unwrapped._lp_alpha
        # Expected: filt_vx = α0 * (1.0 * mlv) + (1-α0) * 0 = α0 * mlv
        for idx, _ in enumerate(cfg.possible_agents):
            got = unwrapped.cmd_vel[:, idx, :]
            filt = unwrapped._cmd_vel_filt[:, idx, :]
            # vx
            expected_vx = alpha[0] * mlv
            assert torch.allclose(got[:, 0], expected_vx, atol=1e-5), \
                f"agent {idx}: first-step LP vx {got[0, 0]} != α*mlv {expected_vx[0]}"
            assert torch.allclose(filt[:, 0], expected_vx, atol=1e-5), \
                f"agent {idx}: _cmd_vel_filt vx mismatch"
            # cmd_vel must equal _cmd_vel_filt after LP
            assert torch.allclose(got, filt, atol=1e-6), \
                f"agent {idx}: cmd_vel != _cmd_vel_filt after LP"
            # Other channels with zero action stay zero
            for ch in range(1, 7):
                assert torch.allclose(got[:, ch], torch.zeros_like(got[:, ch]), atol=1e-6), \
                    f"agent {idx}: ch {ch} should be 0, got {got[0, ch]}"
        r.add_pass(f"LP first step: cmd_vel == α × cmd_vel_raw, _cmd_vel_filt mirrors")
    except Exception:
        r.add_fail("env_lp_on_first_step", traceback.format_exc())


def test_env_lp_convergence_constant_action(r: TestResults, unwrapped):
    """Repeated constant action: _cmd_vel_filt converges toward the raw command."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_prev_action_obs = True
        cfg.enable_action_lowpass = True
        unwrapped._cmd_vel_filt.zero_()
        actions = _make_actions(unwrapped, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Run 12 policy steps (~ 6τ at τ=0.08, dt=0.04)
        for _ in range(12):
            unwrapped._pre_physics_step(actions)

        mlv = unwrapped._max_lin_vel  # (N,)
        # cmd_vx target = 1.0 * mlv per env
        target_vx = mlv
        for idx, _ in enumerate(cfg.possible_agents):
            got_vx = unwrapped.cmd_vel[:, idx, 0]
            # Within ~3% (~5τ would be 1 − e⁻⁵ ≈ 0.9933; 12 × 0.04 / 0.08 = 6τ)
            ratio = got_vx / target_vx.clamp(min=1e-9)
            assert (ratio.min().item() > 0.95 and ratio.max().item() <= 1.0 + 1e-5), \
                f"agent {idx}: convergence ratio range [{ratio.min().item()}, {ratio.max().item()}]"
        r.add_pass("LP convergence (12 steps): cmd_vx ≥ 0.95 × target across envs/agents")
    except Exception:
        r.add_fail("env_lp_convergence_constant_action", traceback.format_exc())


def test_env_prev_action_in_obs(r: TestResults, unwrapped):
    """With prev-action-obs ON, last 7 dims of ego-obs = _cmd_vel_filt[:, idx, :]."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_prev_action_obs = True
        cfg.enable_action_lowpass = True
        # Drive a non-zero filtered command.
        unwrapped._cmd_vel_filt.zero_()
        actions = _make_actions(unwrapped, [0.5, -0.4, 0.3, 0.2, 0.6, -0.1, 0.7])
        unwrapped._pre_physics_step(actions)

        obs = unwrapped._get_observations()
        # Per-agent observation total dim = 47 (no triangulation) + 7 = 54.
        # Ego portion ends at dim 31 + 7 = 38; inter-agent fills 38..54.
        # Append happens at position 31..38 in ego_obs, BEFORE inter-agent.
        ego_end = 31 + 7  # index of inter-agent start
        for idx, agent_id in enumerate(cfg.possible_agents):
            o = obs[agent_id]
            assert o.shape[-1] == cfg.observation_spaces[agent_id], \
                f"agent {agent_id}: obs dim {o.shape[-1]} != cfg {cfg.observation_spaces[agent_id]}"
            tail = o[:, 31:38]
            expected = unwrapped._cmd_vel_filt[:, idx, :]
            assert torch.allclose(tail, expected, atol=1e-5), \
                f"agent {agent_id}: ego prev-action tail {tail[0]} != _cmd_vel_filt {expected[0]}"
        r.add_pass(f"prev-action in obs: ego[31:38] == _cmd_vel_filt for all agents")
    except Exception:
        r.add_fail("env_prev_action_in_obs", traceback.format_exc())


def test_env_lp_off_prev_on_passthrough(r: TestResults, unwrapped):
    """LP OFF, prev-action ON: _cmd_vel_filt is a passthrough of cmd_vel."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_prev_action_obs = True
        cfg.enable_action_lowpass = False
        unwrapped._cmd_vel_filt.zero_()

        actions = _make_actions(unwrapped, [0.3, 0.2, 0.1, 0.5, 0.4, -0.3, 0.6])
        unwrapped._pre_physics_step(actions)

        for idx, _ in enumerate(cfg.possible_agents):
            got = unwrapped.cmd_vel[:, idx, :]
            filt = unwrapped._cmd_vel_filt[:, idx, :]
            assert torch.allclose(got, filt, atol=1e-6), \
                f"agent {idx}: passthrough mismatch — cmd_vel != _cmd_vel_filt"
        r.add_pass("LP off + prev-action on: _cmd_vel_filt mirrors cmd_vel (passthrough)")
    except Exception:
        r.add_fail("env_lp_off_prev_on_passthrough", traceback.format_exc())


def test_env_reset_clears_filter(r: TestResults, unwrapped):
    """_reset_idx zeros _cmd_vel_filt for the reset envs."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_prev_action_obs = True
        cfg.enable_action_lowpass = True
        # Drive non-zero filter state.
        unwrapped._cmd_vel_filt.zero_()
        actions = _make_actions(unwrapped, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        for _ in range(5):
            unwrapped._pre_physics_step(actions)
        assert unwrapped._cmd_vel_filt.abs().max().item() > 0, "expected non-zero filter state"

        # Reset half the envs.
        N = unwrapped.num_envs
        reset_ids = torch.arange(0, N, 2, device=unwrapped.device)
        unwrapped._reset_idx(reset_ids)

        # Filter zeroed for reset envs, untouched for others (sign check).
        assert torch.equal(
            unwrapped._cmd_vel_filt[reset_ids],
            torch.zeros_like(unwrapped._cmd_vel_filt[reset_ids]),
        ), "reset envs should have zero _cmd_vel_filt"
        non_reset = torch.arange(1, N, 2, device=unwrapped.device)
        assert unwrapped._cmd_vel_filt[non_reset].abs().max().item() > 0, \
            "non-reset envs should retain filter state"
        r.add_pass("reset zeros _cmd_vel_filt for reset envs only")
    except Exception:
        r.add_fail("env_reset_clears_filter", traceback.format_exc())


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 80, flush=True)
    print("TICKET 043 — ACTION SMOOTHNESS (prev-action obs + cmd_vel LP) TEST SUITE", flush=True)
    print("=" * 80, flush=True)

    r = TestResults()

    print("\n--- Category 1 — cfg defaults / obs-dim accounting (pre-build) ---", flush=True)
    test_cfg_defaults(r)
    test_cfg_obs_dim_accounting(r)

    print("\n--- Category 2 — pure-math LP tests (no env) ---", flush=True)
    test_lp_formula_convergence(r)
    test_lp_large_tau_attenuates_first_step(r)
    test_lp_sampling_rate_invariance(r)

    # Single env build — Isaac Sim's SimulationApp is global per-process, so we
    # toggle the runtime flags between tests rather than rebuilding. The env
    # is built with the default cfg (both flags on) so obs_dim and buffers
    # are sized for prev-action-obs.
    cfg = IrisMA6TestEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 42  # ticket-034 strict-seed assertion
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    unwrapped = env.unwrapped

    try:
        print("\n--- Category 3a — env buffers / α derivation ---", flush=True)
        test_env_buffers_and_alpha(r, unwrapped)

        print("\n--- Category 3b — flags OFF regression (cmd_vel = action*scale) ---", flush=True)
        test_env_flags_off_regression(r, unwrapped)

        print("\n--- Category 3c — LP runtime semantics ---", flush=True)
        test_env_lp_on_first_step(r, unwrapped)
        test_env_lp_convergence_constant_action(r, unwrapped)

        print("\n--- Category 3d — prev-action concat into ego obs ---", flush=True)
        test_env_prev_action_in_obs(r, unwrapped)

        print("\n--- Category 3e — LP-off passthrough into _cmd_vel_filt ---", flush=True)
        test_env_lp_off_prev_on_passthrough(r, unwrapped)

        print("\n--- Category 3f — reset clears filter state ---", flush=True)
        test_env_reset_clears_filter(r, unwrapped)
    finally:
        env.close()

    success = r.print_summary()
    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
