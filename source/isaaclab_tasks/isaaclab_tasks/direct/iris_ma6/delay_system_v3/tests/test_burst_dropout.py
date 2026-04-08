#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BurstDropoutSampler (Gilbert-Elliott model).

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/test_burst_dropout.py
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/test_burst_dropout.py --test-verbose
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run BurstDropoutSampler test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- NOW import everything else ---
import sys
import torch
import traceback
from datetime import datetime

from isaaclab_tasks.direct.iris_ma6.delay_system_v3.burst_dropout import (
    BurstDropoutCfg,
    BurstDropoutSampler,
)
from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import DistributionCfg

VERBOSE = args_cli.test_verbose


class TestResults:
    def __init__(self):
        self.passed, self.failed, self.errors = [], [], []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")

    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
        for line in error.split("\n")[:5]:
            print(f"    {line}")

    def add_error(self, name: str, error: str):
        self.errors.append((name, error))
        print(f"  ERROR {name}\n    {error[:200]}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print(f"\n{'='*80}\nTEST SUMMARY\n{'='*80}")
        print(f"Total: {total}  Passed: {len(self.passed)}  Failed: {len(self.failed)}  Errors: {len(self.errors)}")
        if self.failed:
            print(f"\n{'-'*80}\nFAILED:")
            for name, err in self.failed:
                print(f"  {name}: {err[:200]}")
        if self.errors:
            print(f"\n{'-'*80}\nERRORS:")
            for name, err in self.errors:
                print(f"  {name}: {err[:200]}")
        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def run_tests(results: TestResults, device: torch.device):
    print(f"\n{'='*80}\nTesting BurstDropoutSampler\n{'='*80}")

    num_envs = 64
    num_agents = 3

    # ==================================================================
    # Test 1: Initialization
    # ==================================================================
    try:
        cfg = BurstDropoutCfg(
            enabled=True,
            p_onset=0.02,
            p_recovery=0.1,
            good_dropout_prob=0.01,
            bad_dropout_prob=0.9,
        )
        sampler = BurstDropoutSampler(cfg, num_envs=num_envs, num_agents=num_agents, device=device)

        assert sampler.num_envs == num_envs
        assert sampler.num_agents == num_agents
        assert sampler.state.shape == (num_envs, num_agents, num_agents)
        assert sampler.dropout_mask.shape == (num_envs, num_agents, num_agents)
        assert (sampler.state == 0).all(), "Initial state should be all Good (0)"
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # ==================================================================
    # Test 2: Advance produces valid masks
    # ==================================================================
    try:
        sampler.advance()
        mask = sampler.dropout_mask
        assert mask.dtype == torch.bool
        assert mask.shape == (num_envs, num_agents, num_agents)
        results.add_pass("Advance produces valid masks")
    except Exception as e:
        results.add_fail("Advance produces valid masks", traceback.format_exc())

    # ==================================================================
    # Test 3: sample_mask returns correct slice
    # ==================================================================
    try:
        sampler.advance()
        full_mask = sampler.dropout_mask.clone()
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                slice_mask = sampler.sample_mask(i, j)
                assert slice_mask.shape == (num_envs,), f"Expected ({num_envs},), got {slice_mask.shape}"
                assert torch.equal(slice_mask, full_mask[:, i, j])
        results.add_pass("sample_mask returns correct slice")
    except Exception as e:
        results.add_fail("sample_mask returns correct slice", traceback.format_exc())

    # ==================================================================
    # Test 4: sample_mask is READ (idempotent)
    # ==================================================================
    try:
        sampler.advance()
        m1 = sampler.sample_mask(0, 1).clone()
        m2 = sampler.sample_mask(0, 1).clone()
        assert torch.equal(m1, m2), "sample_mask should return same result on repeated calls"
        results.add_pass("sample_mask is idempotent (READ)")
    except Exception as e:
        results.add_fail("sample_mask is idempotent (READ)", traceback.format_exc())

    # ==================================================================
    # Test 5: Reset restores Good state
    # ==================================================================
    try:
        # Run many steps to get some Bad states
        for _ in range(100):
            sampler.advance()
        assert sampler.state.sum() > 0, "Should have some Bad states after 100 steps"

        # Full reset
        sampler.reset()
        assert (sampler.state == 0).all(), "All states should be Good after reset"
        assert (sampler.dropout_mask == False).all(), "Dropout mask should be False after reset"  # noqa: E712

        # Per-env reset
        for _ in range(100):
            sampler.advance()
        reset_ids = torch.tensor([0, 5, 10], device=device)
        sampler.reset(reset_ids)
        assert (sampler.state[reset_ids] == 0).all(), "Reset envs should be Good"
        results.add_pass("Reset restores Good state")
    except Exception as e:
        results.add_fail("Reset restores Good state", traceback.format_exc())

    # ==================================================================
    # Test 6: Burst statistics — mean burst length
    # ==================================================================
    try:
        cfg_stats = BurstDropoutCfg(
            enabled=True,
            p_onset=0.02,
            p_recovery=0.1,  # Expected mean burst length = 10
            good_dropout_prob=0.0,  # No dropout in Good state (cleaner measurement)
            bad_dropout_prob=1.0,   # Always dropout in Bad state
        )
        sampler_stats = BurstDropoutSampler(
            cfg_stats, num_envs=256, num_agents=2, device=device
        )

        # Track burst lengths on channel 0->1
        num_steps = 5000
        in_burst = torch.zeros(256, dtype=torch.bool, device=device)
        burst_length = torch.zeros(256, dtype=torch.long, device=device)
        burst_lengths = []

        print("  Running 5000-step burst statistics...", end="", flush=True)
        for step in range(num_steps):
            if step % 1000 == 0 and step > 0:
                print(f".{step}", end="", flush=True)
            sampler_stats.advance()
            is_bad = sampler_stats.state[:, 0, 1] == 1

            # Track burst starts
            new_burst = is_bad & ~in_burst
            # Track burst ends
            burst_ended = ~is_bad & in_burst

            # Record completed burst lengths
            if burst_ended.any():
                completed = burst_length[burst_ended]
                burst_lengths.extend(completed.tolist())

            # Update tracking
            burst_length = torch.where(is_bad, burst_length + 1, torch.zeros_like(burst_length))
            in_burst = is_bad

        print(" done")

        if len(burst_lengths) < 10:
            results.add_fail(
                "Burst statistics — mean burst length",
                f"Only {len(burst_lengths)} bursts observed (need >= 10)"
            )
        else:
            mean_burst = sum(burst_lengths) / len(burst_lengths)
            expected_mean = 1.0 / cfg_stats.p_recovery  # 10.0
            tolerance = 3.0  # Allow ±3 steps tolerance

            if VERBOSE:
                print(f"    Bursts observed: {len(burst_lengths)}")
                print(f"    Mean burst length: {mean_burst:.2f} (expected: {expected_mean:.1f})")
                print(f"    Min/Max burst: {min(burst_lengths)}/{max(burst_lengths)}")

            assert abs(mean_burst - expected_mean) < tolerance, (
                f"Mean burst length {mean_burst:.2f} deviates from expected "
                f"{expected_mean:.1f} by more than {tolerance}"
            )
            results.add_pass(
                f"Burst statistics — mean burst length ({mean_burst:.1f} ≈ {expected_mean:.0f})"
            )
    except Exception as e:
        results.add_fail("Burst statistics — mean burst length", traceback.format_exc())

    # ==================================================================
    # Test 7: Burst statistics — mean good-run length
    # ==================================================================
    try:
        # Reuse sampler_stats from above, reset and re-run
        sampler_stats.reset()

        num_steps = 5000
        in_good = torch.ones(256, dtype=torch.bool, device=device)
        good_length = torch.ones(256, dtype=torch.long, device=device)
        good_lengths = []

        print("  Running 5000-step good-run statistics...", end="", flush=True)
        for step in range(num_steps):
            if step % 1000 == 0 and step > 0:
                print(f".{step}", end="", flush=True)
            sampler_stats.advance()
            is_good = sampler_stats.state[:, 0, 1] == 0

            good_ended = ~is_good & in_good
            if good_ended.any():
                completed = good_length[good_ended]
                good_lengths.extend(completed.tolist())

            good_length = torch.where(is_good, good_length + 1, torch.zeros_like(good_length))
            in_good = is_good

        print(" done")

        if len(good_lengths) < 10:
            results.add_fail(
                "Burst statistics — mean good-run length",
                f"Only {len(good_lengths)} good runs observed (need >= 10)"
            )
        else:
            mean_good = sum(good_lengths) / len(good_lengths)
            expected_good = 1.0 / cfg_stats.p_onset  # 50.0
            tolerance = 15.0

            if VERBOSE:
                print(f"    Good runs observed: {len(good_lengths)}")
                print(f"    Mean good-run length: {mean_good:.2f} (expected: {expected_good:.1f})")

            assert abs(mean_good - expected_good) < tolerance, (
                f"Mean good-run length {mean_good:.2f} deviates from expected "
                f"{expected_good:.1f} by more than {tolerance}"
            )
            results.add_pass(
                f"Burst statistics — mean good-run length ({mean_good:.1f} ≈ {expected_good:.0f})"
            )
    except Exception as e:
        results.add_fail("Burst statistics — mean good-run length", traceback.format_exc())

    # ==================================================================
    # Test 8: Asymmetry — channels i->j and j->i are independent
    # ==================================================================
    try:
        cfg_asym = BurstDropoutCfg(
            enabled=True,
            p_onset=0.1,  # High onset to get many bursts quickly
            p_recovery=0.2,
            good_dropout_prob=0.0,
            bad_dropout_prob=1.0,
        )
        sampler_asym = BurstDropoutSampler(
            cfg_asym, num_envs=512, num_agents=3, device=device
        )

        # Run many steps and collect state correlation
        states_01 = []
        states_10 = []
        for _ in range(200):
            sampler_asym.advance()
            states_01.append(sampler_asym.state[:, 0, 1].float())
            states_10.append(sampler_asym.state[:, 1, 0].float())

        s01 = torch.stack(states_01)  # (200, 512)
        s10 = torch.stack(states_10)

        # Correlation should be low (channels are independent)
        # Stack and compute Pearson correlation across time for each env
        corr_per_env = []
        for e in range(min(50, 512)):  # Check first 50 envs
            c = torch.corrcoef(torch.stack([s01[:, e], s10[:, e]]))[0, 1]
            if not torch.isnan(c):
                corr_per_env.append(c.item())

        if len(corr_per_env) > 0:
            mean_corr = sum(corr_per_env) / len(corr_per_env)
            if VERBOSE:
                print(f"    Mean correlation between channels 0->1 and 1->0: {mean_corr:.3f}")
            assert abs(mean_corr) < 0.3, (
                f"Channel correlation {mean_corr:.3f} too high — channels should be independent"
            )
        results.add_pass("Asymmetry — channels are independent")
    except Exception as e:
        results.add_fail("Asymmetry — channels are independent", traceback.format_exc())

    # ==================================================================
    # Test 9: set_burst_params curriculum control
    # ==================================================================
    try:
        cfg_curr = BurstDropoutCfg(enabled=True, p_onset=0.05, p_recovery=0.1)
        sampler_curr = BurstDropoutSampler(
            cfg_curr, num_envs=16, num_agents=2, device=device
        )

        # Initially p_onset = 0.05
        sampler_curr.set_burst_params(p_onset=0.0, p_recovery=0.1)
        assert (sampler_curr._p_onset == 0.0).all(), "p_onset should be 0 after set"

        # Run many steps — should never enter Bad state if p_onset = 0
        for _ in range(100):
            sampler_curr.advance()
        assert (sampler_curr.state == 0).all(), "No bursts should occur with p_onset=0"

        # Set high onset
        sampler_curr.set_burst_params(p_onset=1.0, p_recovery=0.0)
        sampler_curr.advance()
        # After one step, all Good channels should transition to Bad
        assert (sampler_curr.state == 1).all(), "All should be Bad with p_onset=1, p_recovery=0"

        results.add_pass("set_burst_params curriculum control")
    except Exception as e:
        results.add_fail("set_burst_params curriculum control", traceback.format_exc())

    # ==================================================================
    # Test 10: Per-episode randomization
    # ==================================================================
    try:
        cfg_rand = BurstDropoutCfg(
            enabled=True,
            p_onset=0.02,
            p_recovery=0.1,
            p_onset_distribution=DistributionCfg(
                type="uniform", mean=0.02, half_range=0.015, min_value=0.001, max_value=0.1
            ),
            p_recovery_distribution=DistributionCfg(
                type="uniform", mean=0.1, half_range=0.05, min_value=0.01, max_value=0.5
            ),
        )
        sampler_rand = BurstDropoutSampler(
            cfg_rand, num_envs=64, num_agents=2, device=device
        )

        # Randomize all
        sampler_rand.randomize_params()
        p_onset_vals = sampler_rand._p_onset.clone()
        assert not (p_onset_vals == p_onset_vals[0]).all(), "p_onset should vary across envs"
        assert (p_onset_vals >= 0.001).all() and (p_onset_vals <= 0.1).all(), "p_onset out of range"

        # Randomize subset
        reset_ids = torch.tensor([0, 1, 2], device=device)
        old_vals = sampler_rand._p_onset.clone()
        sampler_rand.randomize_params(reset_ids)
        # Non-reset envs should be unchanged
        assert torch.equal(
            sampler_rand._p_onset[3:], old_vals[3:]
        ), "Non-reset envs should keep old p_onset"

        results.add_pass("Per-episode randomization")
    except Exception as e:
        results.add_fail("Per-episode randomization", traceback.format_exc())

    # ==================================================================
    # Test 11: Scales to >2 agents
    # ==================================================================
    try:
        cfg_5 = BurstDropoutCfg(enabled=True, p_onset=0.05, p_recovery=0.1)
        sampler_5 = BurstDropoutSampler(cfg_5, num_envs=16, num_agents=5, device=device)
        assert sampler_5.state.shape == (16, 5, 5)

        for _ in range(50):
            sampler_5.advance()

        # Check all off-diagonal channels work
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                m = sampler_5.sample_mask(i, j)
                assert m.shape == (16,)

        results.add_pass("Scales to 5 agents")
    except Exception as e:
        results.add_fail("Scales to 5 agents", traceback.format_exc())


def main():
    print(f"{'='*80}\nBURST DROPOUT TEST SUITE\n{'='*80}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Time: {datetime.now():%Y-%m-%d %H:%M:%S}")

    results = TestResults()
    try:
        run_tests(results, device)
    except Exception as e:
        results.add_error("Suite-level", traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
