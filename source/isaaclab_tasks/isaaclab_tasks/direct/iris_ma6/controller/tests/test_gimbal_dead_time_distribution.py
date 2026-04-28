#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""mas/036 dead-time empirical distribution test.

Acceptance: 10k env-resets at curriculum_scale = 1.0 must produce a
distribution whose empirical mean matches the configured mean within
±5 ms and whose support covers the measured 40–100 ms range. The cfg
defaults are sourced from
``/home/usrg/mas/src/scripts/sim2real_model_fitting/output/gimbal_dead_time_fit.json``.

A second sub-test resamples at curriculum_scale = 0.5 and confirms the
mean shrinks by ~50 % (the linear scaling property mas/036 promises the
curriculum).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="mas/036 dead-time distribution tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
        print("TEST SUMMARY (mas/036 dead-time distribution)")
        print("=" * 80)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}, Errors: {len(self.errors)}")
        if self.failed:
            for n, e in self.failed:
                print(f"  {n}: {e.splitlines()[0] if e else ''}")
        return len(self.failed) == 0 and len(self.errors) == 0


# ---------------------------------------------------------------- helpers


def collect_samples(
    cfg: GimbalRateLoopCfg, n_samples: int, scale: float, device: torch.device
) -> torch.Tensor:
    """Collect ``n_samples`` per-env dead-time samples by allocating a
    rate loop with that many envs and calling reset()."""
    rl = GimbalRateLoop(cfg, num_envs=n_samples, device=device)
    rl.set_dead_time_curriculum_scale(scale)
    rl.reset()
    return rl.dead_time_seconds.clone()


# ---------------------------------------------------------------- tests


def run_full_scale_distribution(results: TestResults, device: torch.device):
    """Empirical distribution at full curriculum scale must match the
    configured Gaussian within tolerance, and cover the measured range."""
    print("\n" + "=" * 80)
    print("Full scale: 10k samples — mean within ±5 ms, range covers 40–100 ms")
    print("=" * 80)
    cfg = GimbalRateLoopCfg()  # uses fit defaults: mean=0.066, std=0.016, max=0.120
    n_samples = 10_000
    try:
        samples = collect_samples(cfg, n_samples, scale=1.0, device=device)
        mean = samples.mean().item()
        std = samples.std(unbiased=False).item()
        sample_min = samples.min().item()
        sample_max = samples.max().item()
        if VERBOSE:
            print(f"    n={n_samples}, mean={mean*1000:.2f} ms, std={std*1000:.2f} ms, "
                  f"range=[{sample_min*1000:.1f}, {sample_max*1000:.1f}] ms")

        # Mean within ±5 ms of the configured mean.
        expected_mean = cfg.dead_time_mean_s
        assert abs(mean - expected_mean) < 0.005, (
            f"empirical mean {mean*1000:.2f} ms deviates >5 ms from "
            f"configured {expected_mean*1000:.1f} ms"
        )

        # Std within 10 % of the configured std (clip artifacts can shave it).
        expected_std = cfg.dead_time_std_s
        assert abs(std - expected_std) / max(expected_std, 1e-9) < 0.10, (
            f"empirical std {std*1000:.2f} ms deviates >10 % from "
            f"configured {expected_std*1000:.1f} ms"
        )

        # Support covers the measured range. The bench saw 40–100 ms; we
        # demand the empirical sample stretches across that span.
        assert sample_min < 0.045, (
            f"empirical min {sample_min*1000:.1f} ms misses the 40 ms tail"
        )
        assert sample_max > 0.095, (
            f"empirical max {sample_max*1000:.1f} ms misses the 100 ms tail"
        )

        # Hard cap on samples (clipping at dead_time_max_s).
        assert sample_max <= cfg.dead_time_max_s + 1e-6, (
            f"empirical max {sample_max*1000:.1f} ms exceeds cap "
            f"{cfg.dead_time_max_s*1000:.1f} ms"
        )

        results.add_pass(
            f"10k samples: mean={mean*1000:.2f} ms (target {expected_mean*1000:.1f}), "
            f"std={std*1000:.2f} ms (target {expected_std*1000:.1f}), "
            f"range=[{sample_min*1000:.1f}, {sample_max*1000:.1f}] ms"
        )
    except Exception:
        results.add_fail("full-scale distribution", traceback.format_exc())


def run_half_scale_distribution(results: TestResults, device: torch.device):
    """Half curriculum scale halves both mean and std (linear scaling)."""
    print("\n" + "=" * 80)
    print("Half scale: 10k samples — mean ≈ 0.5 × configured mean")
    print("=" * 80)
    cfg = GimbalRateLoopCfg()
    n_samples = 10_000
    try:
        samples = collect_samples(cfg, n_samples, scale=0.5, device=device)
        mean = samples.mean().item()
        std = samples.std(unbiased=False).item()
        if VERBOSE:
            print(f"    n={n_samples}, mean={mean*1000:.2f} ms, std={std*1000:.2f} ms")

        expected_mean = 0.5 * cfg.dead_time_mean_s
        expected_std = 0.5 * cfg.dead_time_std_s
        # ±2.5 ms tolerance (half the full-scale tolerance).
        assert abs(mean - expected_mean) < 0.0025, (
            f"half-scale mean {mean*1000:.2f} ms deviates >2.5 ms from "
            f"target {expected_mean*1000:.1f} ms"
        )
        assert abs(std - expected_std) / max(expected_std, 1e-9) < 0.10, (
            f"half-scale std {std*1000:.2f} ms deviates >10 % from "
            f"target {expected_std*1000:.1f} ms"
        )
        results.add_pass(
            f"half-scale 10k: mean={mean*1000:.2f} ms (target {expected_mean*1000:.1f}), "
            f"std={std*1000:.2f} ms (target {expected_std*1000:.1f})"
        )
    except Exception:
        results.add_fail("half-scale distribution", traceback.format_exc())


def run_zero_scale_exact_zero(results: TestResults, device: torch.device):
    """At scale=0 every per-env dead time must be exactly zero — the
    bit-exact pre-mas/036 contract."""
    print("\n" + "=" * 80)
    print("Zero scale: every sample is exactly 0 (no Gaussian tails leak)")
    print("=" * 80)
    cfg = GimbalRateLoopCfg()
    n_samples = 10_000
    try:
        samples = collect_samples(cfg, n_samples, scale=0.0, device=device)
        max_abs = samples.abs().max().item()
        assert max_abs == 0.0, (
            f"scale=0 produced nonzero dead times (max |t| = {max_abs:.6e})"
        )
        results.add_pass("zero scale: all 10k samples exactly 0")
    except Exception:
        results.add_fail("zero scale", traceback.format_exc())


def main() -> int:
    print("=" * 80)
    print("GIMBAL DEAD-TIME DISTRIBUTION TEST SUITE (mas/036)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")

    torch.manual_seed(0)

    results = TestResults()
    for fn in (
        run_full_scale_distribution,
        run_half_scale_distribution,
        run_zero_scale_exact_zero,
    ):
        try:
            fn(results, device)
        except Exception:
            results.add_error(fn.__name__, traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
