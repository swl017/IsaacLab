#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""mas/036 step-input dead-time buffer test.

Acceptance: with ``dead_time_mean_s = 0.07, std = 0`` and
``dead_time_curriculum_scale = 1.0``, a step rate command applied at t=0
must produce zero output for at least ``round(0.07/dt) - 1`` consecutive
sim steps and start tracking by step ``round(0.07/dt) + 1``.

Also verifies bit-exact pre-mas/036 behavior at curriculum_scale = 0.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="mas/036 dead-time buffer tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import sys
import traceback
from datetime import datetime

import torch

from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop import GimbalRateLoop
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop_cfg import GimbalRateLoopCfg


VERBOSE = args_cli.test_verbose


# ---------------------------------------------------------------- TestResults


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")

    def add_fail(self, name: str, err: str):
        self.failed.append((name, err))
        print(f"  ✗ {name}")
        for line in err.split("\n")[:5]:
            print(f"    {line}")

    def add_error(self, name: str, err: str):
        self.errors.append((name, err))
        print(f"  ERROR {name}")
        print(f"    {err.splitlines()[-1] if err else ''}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY (mas/036 dead-time buffer)")
        print("=" * 80)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}, Errors: {len(self.errors)}")
        if self.failed:
            for n, e in self.failed:
                print(f"  {n}: {e.splitlines()[0] if e else ''}")
        return len(self.failed) == 0 and len(self.errors) == 0


# ---------------------------------------------------------------- helpers


def simulate_step_with_dead_time(
    mean_s: float,
    dt: float,
    duration_s: float,
    cmd_radps: float,
    device: torch.device,
    n_envs: int = 4,
    std_s: float = 0.0,
    max_s: float = 0.15,
) -> tuple[list[float], int]:
    """Run the rate loop with a constant yaw rate command and a fixed
    dead time. Returns (per-step yaw output [rad/s], expected step
    count = round(mean_s / dt))."""
    cfg = GimbalRateLoopCfg(
        dead_time_mean_s=mean_s,
        dead_time_std_s=std_s,
        dead_time_max_s=max_s,
    )
    rl = GimbalRateLoop(cfg, num_envs=n_envs, device=device)
    rl.set_progress(1.0)
    rl.set_dead_time_curriculum_scale(1.0)
    rl.reset()

    cmd = torch.zeros(n_envs, 2, device=device)
    cmd[:, 0] = cmd_radps  # yaw

    n_steps = int(round(duration_s / dt))
    out = []
    for _ in range(n_steps):
        omega = rl.step(cmd, dt)
        out.append(omega[0, 0].item())
    expected_d = int(round(mean_s / dt))
    return out, expected_d


# ---------------------------------------------------------------- tests


def run_step_input_dead_time(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Step input: zero output for round(0.07/dt) - 1 steps, then track")
    print("=" * 80)
    dt = 0.01
    mean_s = 0.07
    cmd = 0.5  # rad/s, well within saturation
    try:
        out, d = simulate_step_with_dead_time(
            mean_s=mean_s, dt=dt, duration_s=0.30,
            cmd_radps=cmd, device=device,
            n_envs=4, std_s=0.0,
        )
        if VERBOSE:
            print(f"    expected dead-time depth d = {d}")
            print(f"    first 12 outputs (rad/s): "
                  f"{[f'{x:.4f}' for x in out[:12]]}")

        # Lower bound: at least d-1 zero steps starting from index 0.
        # Index i is step i+1 (1-based). The buffer holds the cmd in
        # cold state for n_pushed <= d → indices 0..d-1 are zero.
        # That gives d zero steps, ≥ d-1.
        zero_count = 0
        for x in out:
            if abs(x) < 1e-12:
                zero_count += 1
            else:
                break
        assert zero_count >= d - 1, (
            f"expected ≥ {d - 1} zero steps, got {zero_count}. "
            f"first values: {out[:d + 2]}"
        )

        # Upper bound: motion has begun by step d+1 (index d).
        assert abs(out[d]) > 1e-9 or abs(out[d + 1]) > 1e-9, (
            f"expected motion by step {d + 1} (index {d}), "
            f"got out[{d}]={out[d]:.6f}, out[{d + 1}]={out[d + 1]:.6f}"
        )
        results.add_pass(
            f"step input @ d={d}: zero for {zero_count} steps, motion by step {d + 1}"
        )
    except Exception:
        results.add_fail("step input dead-time", traceback.format_exc())


def run_pre_mas036_regression(results: TestResults, device: torch.device):
    """At curriculum_scale = 0 the dead-time buffer must be a no-op:
    behavior identical to mas/035 (rate loop with no dead time)."""
    print("\n" + "=" * 80)
    print("Regression: scale=0 produces bit-exact pre-mas/036 output")
    print("=" * 80)
    dt = 0.01
    cmd_radps = 0.5

    cfg = GimbalRateLoopCfg(
        dead_time_mean_s=0.07,
        dead_time_std_s=0.02,
        dead_time_max_s=0.15,
        dead_time_curriculum_scale=0.0,
    )

    try:
        rl = GimbalRateLoop(cfg, num_envs=4, device=device)
        rl.set_progress(1.0)
        rl.reset()
        cmd = torch.zeros(4, 2, device=device)
        cmd[:, 0] = cmd_radps

        # Reference trace: a fresh rate loop with mas/035 behavior — i.e.
        # max_s = 0 to disable the buffer path entirely.
        cfg_ref = GimbalRateLoopCfg(dead_time_max_s=0.0)
        rl_ref = GimbalRateLoop(cfg_ref, num_envs=4, device=device)
        rl_ref.set_progress(1.0)
        rl_ref.reset()

        max_diff = 0.0
        for _ in range(50):
            o = rl.step(cmd, dt)
            o_ref = rl_ref.step(cmd, dt)
            max_diff = max(max_diff, (o - o_ref).abs().max().item())

        assert max_diff < 1e-7, f"scale=0 deviates by {max_diff:.2e} from pre-mas/036"
        results.add_pass(f"scale=0 bit-exact vs pre-mas/036 (max diff {max_diff:.2e})")
    except Exception:
        results.add_fail("scale=0 regression", traceback.format_exc())


def run_zero_command_zero_output(results: TestResults, device: torch.device):
    """Sanity: zero command → zero output regardless of dead-time depth."""
    print("\n" + "=" * 80)
    print("Sanity: zero command → zero output (dead-time depth irrelevant)")
    print("=" * 80)
    dt = 0.01
    try:
        cfg = GimbalRateLoopCfg(
            dead_time_mean_s=0.10, dead_time_std_s=0.0, dead_time_max_s=0.15
        )
        rl = GimbalRateLoop(cfg, num_envs=8, device=device)
        rl.set_progress(1.0)
        rl.set_dead_time_curriculum_scale(1.0)
        rl.reset()

        cmd = torch.zeros(8, 2, device=device)
        max_abs = 0.0
        for _ in range(40):
            o = rl.step(cmd, dt)
            max_abs = max(max_abs, o.abs().max().item())
        assert max_abs == 0.0, f"zero cmd produced |omega|_max = {max_abs:.6e}"
        results.add_pass("zero command → zero output across episode")
    except Exception:
        results.add_fail("zero command zero output", traceback.format_exc())


def run_per_episode_constant(results: TestResults, device: torch.device):
    """The dead time must be constant within an episode (sampled once at
    reset)."""
    print("\n" + "=" * 80)
    print("Per-episode constant: dead_time_seconds doesn't change between steps")
    print("=" * 80)
    dt = 0.01
    try:
        cfg = GimbalRateLoopCfg(
            dead_time_mean_s=0.07, dead_time_std_s=0.02, dead_time_max_s=0.15
        )
        rl = GimbalRateLoop(cfg, num_envs=32, device=device)
        rl.set_progress(1.0)
        rl.set_dead_time_curriculum_scale(1.0)
        rl.reset()

        snap_pre = rl.dead_time_seconds.clone()
        cmd = torch.zeros(32, 2, device=device)
        cmd[:, 0] = 0.3
        for _ in range(20):
            rl.step(cmd, dt)
        snap_post = rl.dead_time_seconds.clone()

        assert torch.equal(snap_pre, snap_post), (
            "dead_time_seconds changed within an episode "
            f"(max Δ = {(snap_post - snap_pre).abs().max().item():.6e})"
        )
        # Now reset and confirm a fresh sample is drawn.
        rl.reset()
        snap_after_reset = rl.dead_time_seconds.clone()
        # With std > 0 over 32 envs the chance of an exact-equal redraw
        # is astronomically small. We check at least one env differs.
        assert not torch.equal(snap_post, snap_after_reset), (
            "dead_time_seconds did not change after reset"
        )
        results.add_pass("dead-time constant within episode, resampled on reset")
    except Exception:
        results.add_fail("per-episode constant", traceback.format_exc())


def run_buffer_depth_cap(results: TestResults, device: torch.device):
    """Per-env dead times sampled with mean >> max_s must clip to max_s."""
    print("\n" + "=" * 80)
    print("Buffer cap: mean >> max_s clips to max_s (no buffer overflow)")
    print("=" * 80)
    dt = 0.01
    try:
        cfg = GimbalRateLoopCfg(
            dead_time_mean_s=2.0,    # absurdly long
            dead_time_std_s=0.0,
            dead_time_max_s=0.15,    # hard cap
        )
        rl = GimbalRateLoop(cfg, num_envs=8, device=device)
        rl.set_progress(1.0)
        rl.set_dead_time_curriculum_scale(1.0)
        rl.reset()

        # All sampled dead times must be ≤ max_s.
        assert (rl.dead_time_seconds <= cfg.dead_time_max_s + 1e-9).all().item(), (
            f"max sampled dead time {rl.dead_time_seconds.max().item():.4f} "
            f"exceeds cap {cfg.dead_time_max_s}"
        )

        # And the buffer ring depth (after first step) is bounded.
        cmd = torch.zeros(8, 2, device=device)
        cmd[:, 0] = 0.4
        rl.step(cmd, dt)
        max_buf_steps = math.ceil(cfg.dead_time_max_s / dt)
        steps = rl.dead_time_steps
        assert (steps <= max_buf_steps).all().item(), (
            f"dead_time_steps max {steps.max().item()} exceeds buffer depth {max_buf_steps}"
        )
        results.add_pass(f"sampled dead times clipped to {cfg.dead_time_max_s}s; "
                         f"step counts ≤ {max_buf_steps}")
    except Exception:
        results.add_fail("buffer depth cap", traceback.format_exc())


def run_curriculum_blends(results: TestResults, device: torch.device):
    """Half-scale curriculum should produce roughly half the dead time."""
    print("\n" + "=" * 80)
    print("Curriculum: scale=0.5 produces ~half the mean dead time vs scale=1")
    print("=" * 80)
    try:
        cfg = GimbalRateLoopCfg(
            dead_time_mean_s=0.10, dead_time_std_s=0.0, dead_time_max_s=0.20
        )
        rl = GimbalRateLoop(cfg, num_envs=16, device=device)
        rl.set_progress(1.0)

        rl.set_dead_time_curriculum_scale(1.0)
        rl.reset()
        full_mean = rl.dead_time_seconds.mean().item()

        rl.set_dead_time_curriculum_scale(0.5)
        rl.reset()
        half_mean = rl.dead_time_seconds.mean().item()

        rl.set_dead_time_curriculum_scale(0.0)
        rl.reset()
        zero_mean = rl.dead_time_seconds.mean().item()

        assert abs(full_mean - 0.10) < 1e-6, f"full scale: {full_mean}"
        assert abs(half_mean - 0.05) < 1e-6, f"half scale: {half_mean}"
        assert abs(zero_mean) < 1e-6, f"zero scale: {zero_mean}"
        results.add_pass(
            f"curriculum scaling: 0={zero_mean:.4f}, 0.5={half_mean:.4f}, 1.0={full_mean:.4f}"
        )
    except Exception:
        results.add_fail("curriculum blend", traceback.format_exc())


def run_partial_reset(results: TestResults, device: torch.device):
    """reset(env_ids) only resamples + clears buffer for the specified
    envs. Other envs keep their dead time and buffer state."""
    print("\n" + "=" * 80)
    print("Partial reset: only specified envs get fresh dead-time samples")
    print("=" * 80)
    try:
        cfg = GimbalRateLoopCfg(
            dead_time_mean_s=0.07, dead_time_std_s=0.02, dead_time_max_s=0.15
        )
        rl = GimbalRateLoop(cfg, num_envs=8, device=device)
        rl.set_progress(1.0)
        rl.set_dead_time_curriculum_scale(1.0)
        rl.reset()
        snap = rl.dead_time_seconds.clone()

        env_ids = torch.tensor([1, 3, 5], device=device)
        rl.reset(env_ids=env_ids)
        new = rl.dead_time_seconds.clone()

        # Untouched envs (0, 2, 4, 6, 7) must be exactly equal.
        untouched_mask = torch.tensor(
            [True, False, True, False, True, False, True, True], device=device
        )
        assert torch.equal(snap[untouched_mask], new[untouched_mask]), (
            "untouched envs' dead time changed"
        )
        # At least one of the reset envs differs.
        assert not torch.equal(snap[env_ids], new[env_ids]), (
            "no reset env's dead time changed (extremely improbable)"
        )
        results.add_pass("partial reset only resamples specified envs")
    except Exception:
        results.add_fail("partial reset", traceback.format_exc())


def main() -> int:
    print("=" * 80)
    print("GIMBAL DEAD-TIME BUFFER TEST SUITE (mas/036)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")

    torch.manual_seed(0)

    results = TestResults()
    for fn in (
        run_step_input_dead_time,
        run_pre_mas036_regression,
        run_zero_command_zero_output,
        run_per_episode_constant,
        run_buffer_depth_cap,
        run_curriculum_blends,
        run_partial_reset,
    ):
        try:
            fn(results, device)
        except Exception:
            results.add_error(fn.__name__, traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
