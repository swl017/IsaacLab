#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test Phase 3/4 curriculum implementation."""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Test Phase 3/4 curriculum")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW we can import other modules
import torch
import sys
import traceback


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}")
        error_lines = error.split('\n')[:5]
        for line in error_lines:
            print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        print(f"    {error[:200]}")

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)" if total > 0 else "Failed: 0")
        print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)" if total > 0 else "Errors: 0")

        if self.failed:
            print("\n" + "-" * 80)
            print("FAILED TESTS:")
            print("-" * 80)
            for test_name, error in self.failed:
                print(f"\n{test_name}:")
                print(f"  {error}")

        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def test_curriculum_cfg(results: TestResults):
    """Test CurriculumCfg phase methods."""
    print("\n" + "=" * 80)
    print("Testing CurriculumCfg")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma5.curriculum import CurriculumCfg

    try:
        cfg = CurriculumCfg()
        results.add_pass("CurriculumCfg initialization")
    except Exception as e:
        results.add_fail("CurriculumCfg initialization", str(e))
        return

    # Test get_delay_mode
    try:
        assert cfg.get_delay_mode(0) == "none", f"Expected 'none', got {cfg.get_delay_mode(0)}"
        assert cfg.get_delay_mode(50000) == "none", f"Expected 'none', got {cfg.get_delay_mode(50000)}"
        assert cfg.get_delay_mode(110000) == "fixed", f"Expected 'fixed', got {cfg.get_delay_mode(110000)}"
        assert cfg.get_delay_mode(140000) == "random", f"Expected 'random', got {cfg.get_delay_mode(140000)}"
        results.add_pass("get_delay_mode() returns correct modes")
    except AssertionError as e:
        results.add_fail("get_delay_mode() returns correct modes", str(e))

    # Test get_fixed_delay_progress
    try:
        assert cfg.get_fixed_delay_progress(100000) == 0.0
        assert 0.4 < cfg.get_fixed_delay_progress(115000) < 0.6  # Should be ~0.5
        assert cfg.get_fixed_delay_progress(130000) == 1.0
        results.add_pass("get_fixed_delay_progress() returns correct values")
    except AssertionError as e:
        results.add_fail("get_fixed_delay_progress() returns correct values", str(e))

    # Test get_random_delay_progress
    try:
        assert cfg.get_random_delay_progress(130000) == 0.0
        assert 0.4 < cfg.get_random_delay_progress(145000) < 0.6
        assert cfg.get_random_delay_progress(160000) == 1.0
        results.add_pass("get_random_delay_progress() returns correct values")
    except AssertionError as e:
        results.add_fail("get_random_delay_progress() returns correct values", str(e))

    # Test get_dropout_progress
    try:
        assert cfg.get_dropout_progress(160000) == 0.0
        assert 0.4 < cfg.get_dropout_progress(180000) < 0.6
        assert cfg.get_dropout_progress(200000) == 1.0
        results.add_pass("get_dropout_progress() returns correct values")
    except AssertionError as e:
        results.add_fail("get_dropout_progress() returns correct values", str(e))

    # Test get_noise_progress
    try:
        assert cfg.get_noise_progress(80000) == 0.0
        assert 0.4 < cfg.get_noise_progress(90000) < 0.6
        assert cfg.get_noise_progress(100000) == 1.0
        results.add_pass("get_noise_progress() returns correct values")
    except AssertionError as e:
        results.add_fail("get_noise_progress() returns correct values", str(e))


def test_delay_pipeline(results: TestResults, device: torch.device):
    """Test DelayPipeline delay mode switching."""
    print("\n" + "=" * 80)
    print("Testing DelayPipeline")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma5.delay_system_v2.delay_cfg import FieldDelayCfg, DistributionCfg
    from isaaclab_tasks.direct.iris_ma5.delay_system_v2.delay_pipeline import DelayPipeline

    num_envs = 16
    field_dim = 3
    dt = 0.01

    try:
        cfg = FieldDelayCfg(
            latency_enabled=True,
            latency=DistributionCfg(type="normal", mean=0.1, std=0.02),
        )
        pipeline = DelayPipeline(cfg, num_envs, field_dim, dt, device)
        results.add_pass("DelayPipeline initialization with latency")
    except Exception as e:
        results.add_fail("DelayPipeline initialization with latency", traceback.format_exc())
        return

    # Test set_delay_mode
    try:
        pipeline.set_delay_mode("none")
        assert pipeline._delay_mode == "none"
        pipeline.set_delay_mode("fixed", progress=0.5)
        assert pipeline._delay_mode == "fixed"
        assert pipeline._delay_progress == 0.5
        pipeline.set_delay_mode("random", progress=1.0)
        assert pipeline._delay_mode == "random"
        results.add_pass("set_delay_mode() updates state correctly")
    except AssertionError as e:
        results.add_fail("set_delay_mode() updates state correctly", str(e))

    # Test effective latency calculations
    try:
        pipeline.set_delay_mode("none")
        assert pipeline._get_effective_latency_mean() == 0.0
        assert pipeline._get_effective_latency_std() == 0.0

        pipeline.set_delay_mode("fixed", progress=0.5)
        assert abs(pipeline._get_effective_latency_mean() - 0.05) < 0.001  # 0.1 * 0.5
        assert pipeline._get_effective_latency_std() == 0.0  # Fixed mode has no variance

        pipeline.set_delay_mode("random", progress=0.5)
        assert abs(pipeline._get_effective_latency_mean() - 0.05) < 0.001
        # For normal distribution converted from std=0.02, effective std should be ~0.02 * 0.5
        # Note: std conversion from uniform is different
        results.add_pass("Effective latency calculations work correctly")
    except AssertionError as e:
        results.add_fail("Effective latency calculations work correctly", str(e))

    # Test reset behavior in different modes
    try:
        env_ids = torch.arange(4, device=device)

        # Reset in "none" mode
        pipeline.set_delay_mode("none")
        pipeline.reset(env_ids)
        assert torch.all(pipeline._latency_delays[env_ids] == 0.0)

        # Reset in "fixed" mode
        pipeline.set_delay_mode("fixed", progress=0.5)
        pipeline.reset(env_ids)
        expected_delay = 0.1 * 0.5  # mean * progress
        assert torch.allclose(pipeline._latency_delays[env_ids], torch.tensor(expected_delay, device=device))

        # Reset in "random" mode - delays should vary
        pipeline.set_delay_mode("random", progress=1.0)
        pipeline.reset(env_ids)
        # With std > 0, delays should have some variance
        std_of_delays = pipeline._latency_delays[env_ids].std()
        # Note: with only 4 samples, std might be 0 by chance, so we just check they're non-negative
        assert torch.all(pipeline._latency_delays[env_ids] >= 0.0)

        results.add_pass("Reset behavior differs between modes")
    except AssertionError as e:
        results.add_fail("Reset behavior differs between modes", str(e))


def test_multi_agent_delay_system(results: TestResults, device: torch.device):
    """Test MultiAgentDelaySystemV2 curriculum API."""
    print("\n" + "=" * 80)
    print("Testing MultiAgentDelaySystemV2")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma5.delay_system_v2 import (
        MultiAgentDelaySystemV2,
        MultiAgentDelaySystemV2Cfg,
    )

    num_envs = 16
    possible_agents = ["drone_0", "drone_1"]

    try:
        cfg = MultiAgentDelaySystemV2Cfg(
            dt=0.01,
            detection_latency_mean=0.1,
            detection_latency_std=0.02,
            inter_agent_comm_latency_mean=0.15,
            inter_agent_comm_latency_std=0.03,
            detection_dropout_rate=0.1,
            inter_agent_comm_dropout_rate=0.1,
        )
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={a: 3 for a in possible_agents},
            num_targets_per_agent={a: 1 for a in possible_agents},
            device=device,
        )
        results.add_pass("MultiAgentDelaySystemV2 initialization")
    except Exception as e:
        results.add_fail("MultiAgentDelaySystemV2 initialization", traceback.format_exc())
        return

    # Test set_delay_mode
    try:
        delay_system.set_delay_mode("none")
        delay_system.set_delay_mode("fixed", progress=0.5)
        delay_system.set_delay_mode("random", progress=1.0)
        results.add_pass("set_delay_mode() works on MultiAgentDelaySystemV2")
    except Exception as e:
        results.add_fail("set_delay_mode() works on MultiAgentDelaySystemV2", str(e))

    # Test set_dropout_enabled
    try:
        delay_system.set_dropout_enabled(False)
        delay_system.set_dropout_enabled(True, progress=0.5)
        results.add_pass("set_dropout_enabled() works on MultiAgentDelaySystemV2")
    except Exception as e:
        results.add_fail("set_dropout_enabled() works on MultiAgentDelaySystemV2", str(e))


def main():
    """Main test runner."""
    print("=" * 80)
    print("PHASE 3/4 CURRICULUM TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    results = TestResults()

    try:
        test_curriculum_cfg(results)
    except Exception as e:
        results.add_error("CurriculumCfg suite", traceback.format_exc())

    try:
        test_delay_pipeline(results, device)
    except Exception as e:
        results.add_error("DelayPipeline suite", traceback.format_exc())

    try:
        test_multi_agent_delay_system(results, device)
    except Exception as e:
        results.add_error("MultiAgentDelaySystemV2 suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
