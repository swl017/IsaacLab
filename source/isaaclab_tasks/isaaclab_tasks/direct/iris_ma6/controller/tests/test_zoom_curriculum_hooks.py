#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum-hook tests for the SIYI A8 zoom dead-time buffer (mas/037).

Mirrors the gimbal mas/036 dead-time-distribution test pattern. Validates:
- ``set_dead_time_curriculum_scale(0.0)`` then ``reset`` → all per-env τ_d = 0.
- ``set_dead_time_curriculum_scale(1.0)`` then ``reset`` (large num_envs) →
  empirical mean within ±2σ/√N of cfg.dead_time_mean_s.
- Per-env reset semantics: env_ids subset only resamples those envs.
- Bit-exact regression: scale=0 produces an identical trace to a no-buffer
  control run (early-out path active).
- ``model="first_order"``: ``set_dead_time_curriculum_scale`` is a no-op
  (no exception, no state mutation, scale stays 0).

Run:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/test_zoom_curriculum_hooks.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zoom curriculum-hook tests (mas/037)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback

import torch

from isaaclab_tasks.direct.iris_ma6.controller import ZoomController, ZoomControllerCfg


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  PASS  {name}")

    def add_fail(self, name: str, msg: str):
        self.failed.append((name, msg))
        first = msg.splitlines()[0] if msg else ""
        print(f"  FAIL  {name}: {first}")

    def summary(self) -> bool:
        print("-" * 70)
        print(f"Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        for n, m in self.failed:
            print(f"\n{n}:\n{m}")
        return not self.failed


def _siyi_cfg(**overrides) -> ZoomControllerCfg:
    base = dict(
        model="siyi_a8",
        tau_zoom=0.091,
        max_zoom_rate=2.0,
        v_max_levels_per_s=3.16,
        quantum_levels=0.1,
        dead_time_mean_s=0.100,
        dead_time_std_s=0.018,
        dead_time_max_s=0.150,
        dead_time_curriculum_scale=0.0,
    )
    base.update(overrides)
    return ZoomControllerCfg(**base)


def test_scale_zero_force_zero(results: TestResults, device):
    """scale=0 + reset → all τ_d are exactly 0 (force-zero path)."""
    try:
        cfg = _siyi_cfg()
        ctrl = ZoomController(cfg, num_envs=64, device=device)
        ctrl.set_dead_time_curriculum_scale(0.0)
        ctrl.reset()
        assert (ctrl.dead_time_seconds == 0.0).all().item(), (
            f"scale=0 should force τ_d=0; got max={ctrl.dead_time_seconds.max().item()}"
        )
        results.add_pass("scale=0 → all τ_d == 0")
    except Exception:
        results.add_fail("scale=0 force zero", traceback.format_exc())


def test_scale_one_distribution(results: TestResults, device):
    """scale=1 + reset on a large batch → mean within Gaussian SE of cfg mean."""
    try:
        torch.manual_seed(0xA8A8)
        N = 4096
        cfg = _siyi_cfg()
        ctrl = ZoomController(cfg, num_envs=N, device=device)
        ctrl.set_dead_time_curriculum_scale(1.0)
        ctrl.reset()

        td = ctrl.dead_time_seconds
        emp_mean = td.mean().item()
        # Standard error of the mean ~ std/√N. With N=4096 and std~0.018,
        # SE ≈ 2.8e-4. 4σ tolerance gives a comfortable bound that still
        # catches gross misconfiguration.
        se = cfg.dead_time_std_s / math.sqrt(N)
        tol = 4.0 * se
        assert abs(emp_mean - cfg.dead_time_mean_s) < tol, (
            f"empirical mean {emp_mean:.5f} differs from cfg mean "
            f"{cfg.dead_time_mean_s} by more than 4·SE ({tol:.5f})"
        )
        # Also verify the cap is respected.
        assert td.max().item() <= cfg.dead_time_max_s + 1e-6, (
            f"τ_d max {td.max().item()} exceeds cap {cfg.dead_time_max_s}"
        )
        # And τ_d >= 0.
        assert td.min().item() >= 0.0, f"τ_d min {td.min().item()} negative"
        results.add_pass(f"scale=1 distribution: mean={emp_mean:.4f} (cfg={cfg.dead_time_mean_s})")
    except Exception:
        results.add_fail("scale=1 distribution", traceback.format_exc())


def test_partial_reset(results: TestResults, device):
    """reset(env_ids=subset) only resamples those envs."""
    try:
        cfg = _siyi_cfg()
        ctrl = ZoomController(cfg, num_envs=8, device=device)
        # Initial: scale=0 reset → all τ_d = 0
        ctrl.set_dead_time_curriculum_scale(0.0)
        ctrl.reset()
        assert (ctrl.dead_time_seconds == 0.0).all().item()

        # Now ramp to scale=1 and reset only envs [0, 1, 2].
        ctrl.set_dead_time_curriculum_scale(1.0)
        ctrl.reset(env_ids=torch.tensor([0, 1, 2], device=device))

        td = ctrl.dead_time_seconds
        # Envs 0..2 should have non-zero τ_d (with very high probability).
        assert (td[:3] > 0.0).all().item(), f"reset envs should have τ_d > 0; got {td[:3]}"
        # Envs 3..7 should still be 0 (untouched).
        assert (td[3:] == 0.0).all().item(), f"unaffected envs should retain 0; got {td[3:]}"
        results.add_pass("Partial reset: only env_ids subset resampled")
    except Exception:
        results.add_fail("Partial reset", traceback.format_exc())


def test_scale_zero_bit_exact_regression(results: TestResults, device):
    """At scale=0, the dead-time stage is a no-op: trace identical (within
    FP) to a controller that bypasses the buffer entirely. Locks in the
    pre-mas/037 invariance gate for siyi_a8 mode at curriculum-bootstrap."""
    try:
        cfg_buf = _siyi_cfg(tau_zoom=0.05)
        ctrl_buf = ZoomController(cfg_buf, num_envs=4, device=device)
        ctrl_buf.set_dead_time_curriculum_scale(0.0)
        ctrl_buf.reset()

        # Reference controller: same cfg but with dead_time_max_s=0 → buffer
        # alloc skipped entirely (the early-out path in _apply_dead_time).
        cfg_ref = _siyi_cfg(tau_zoom=0.05, dead_time_max_s=0.0)
        ctrl_ref = ZoomController(cfg_ref, num_envs=4, device=device)
        ctrl_ref.reset()

        torch.manual_seed(0xC0FFEE)
        dt = 0.04
        max_diff = 0.0
        for _ in range(150):
            cmd = torch.empty(4, device=device).uniform_(-1.0, 1.0)
            out_buf = ctrl_buf.compute_control(cmd, dt)
            out_ref = ctrl_ref.compute_control(cmd, dt)
            diff = (out_buf - out_ref).abs().max().item()
            max_diff = max(max_diff, diff)
        assert max_diff < 1e-6, f"scale=0 path diverged from no-buffer path: max |Δ| = {max_diff}"
        results.add_pass(f"scale=0 bit-exact vs no-buffer (max |Δ|={max_diff:.2e})")
    except Exception:
        results.add_fail("scale=0 bit-exact regression", traceback.format_exc())


def test_first_order_mode_hook_noop(results: TestResults, device):
    """In first_order mode, set_dead_time_curriculum_scale must be a silent
    no-op (the curriculum loop calls it unconditionally on every step).
    """
    try:
        cfg = ZoomControllerCfg(model="first_order")
        ctrl = ZoomController(cfg, num_envs=4, device=device)
        # No exception
        ctrl.set_dead_time_curriculum_scale(0.5)
        ctrl.set_dead_time_curriculum_scale(1.0)
        # Internal scale should NOT change in first_order mode.
        assert ctrl.dead_time_curriculum_scale == 0.0, (
            f"first_order should ignore scale writes; got {ctrl.dead_time_curriculum_scale}"
        )
        # And reset should not allocate the buffer or sample τ_d.
        ctrl.reset()
        assert ctrl._dead_time_buffer is None
        assert (ctrl.dead_time_seconds == 0.0).all().item()
        results.add_pass("first_order: set_dead_time_curriculum_scale is no-op")
    except Exception:
        results.add_fail("first_order hook no-op", traceback.format_exc())


def main():
    print("=" * 70)
    print("Zoom curriculum-hook tests (mas/037 dead-time buffer)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = TestResults()
    test_scale_zero_force_zero(results, device)
    test_scale_one_distribution(results, device)
    test_partial_reset(results, device)
    test_scale_zero_bit_exact_regression(results, device)
    test_first_order_mode_hook_noop(results, device)

    ok = results.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
