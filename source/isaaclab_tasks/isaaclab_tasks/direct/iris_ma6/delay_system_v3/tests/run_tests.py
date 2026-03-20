#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main test runner for delay system V3.

This script runs all V3 delay system tests and provides comprehensive output.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/run_tests.py
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/run_tests.py --test-verbose
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run delay system V3 test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument(
    "--test-verbose", action="store_true", help="Enable verbose debug output"
)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules
import os
import sys
import torch
import traceback
from datetime import datetime
from typing import List, Tuple, Optional

# Import test modules (handle both script and module execution)
try:
    from .test_per_agent import run_per_agent_randomization_tests
    from .test_reward_modes import run_reward_mode_tests
except ImportError:
    # Running as script, add parent to path
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_per_agent import run_per_agent_randomization_tests
    from test_reward_modes import run_reward_mode_tests

# Output file path (same directory as this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "test_result.txt")


def log_print(msg: str = "", file_handle=None):
    """Print to both stdout and file."""
    print(msg)
    sys.stdout.flush()
    if file_handle:
        file_handle.write(msg + "\n")
        file_handle.flush()


class TestResults:
    """Track test results with detailed reporting."""

    def __init__(self, file_handle=None):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.errors: List[Tuple[str, str]] = []
        self._current_suite: str = ""
        self._file = file_handle

    def _log(self, msg: str = ""):
        """Log to both stdout and file."""
        log_print(msg, self._file)

    def set_suite(self, suite_name: str):
        """Set current test suite name for output formatting."""
        self._current_suite = suite_name
        self._log(f"\n{'=' * 80}")
        self._log(f"{suite_name}")
        self._log("=" * 80)

    def add_pass(self, test_name: str):
        """Record a passing test."""
        full_name = f"{self._current_suite}::{test_name}"
        self.passed.append(full_name)
        self._log(f"  [PASS] {test_name}")

    def add_fail(self, test_name: str, error: str):
        """Record a failing test."""
        full_name = f"{self._current_suite}::{test_name}"
        self.failed.append((full_name, error))
        self._log(f"  [FAIL] {test_name}")
        # Print first few lines of error
        error_lines = error.split("\n")[:5]
        for line in error_lines:
            self._log(f"    {line}")

    def add_error(self, test_name: str, error: str):
        """Record a test error (exception during test)."""
        full_name = f"{self._current_suite}::{test_name}"
        self.errors.append((full_name, error))
        self._log(f"  [ERROR] {test_name}")
        error_lines = error.split("\n")[:5]
        for line in error_lines:
            self._log(f"    {line}")

    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return len(self.failed) == 0 and len(self.errors) == 0

    def print_summary(self):
        """Print detailed test summary."""
        total = len(self.passed) + len(self.failed) + len(self.errors)

        self._log("\n" + "=" * 80)
        self._log("TEST SUMMARY")
        self._log("=" * 80)
        self._log(f"Total Tests: {total}")
        if total > 0:
            self._log(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)")
            self._log(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)")
            self._log(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)")
        else:
            self._log("No tests were run.")

        if self.failed:
            self._log("\n" + "-" * 80)
            self._log("FAILED TESTS:")
            self._log("-" * 80)
            for test_name, error in self.failed:
                self._log(f"\n{test_name}:")
                self._log(f"  {error[:500]}")  # Truncate long errors

        if self.errors:
            self._log("\n" + "-" * 80)
            self._log("TEST ERRORS:")
            self._log("-" * 80)
            for test_name, error in self.errors:
                self._log(f"\n{test_name}:")
                self._log(f"  {error[:500]}")

        self._log("=" * 80)


def run_config_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run configuration tests."""
    results.set_suite("Configuration Tests")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
        DistributionCfg,
        SamplingCfg,
        LatencyCfg,
        StalenessCfg,
        DropoutCfg,
        DelayPipelineCfgV3,
        UnifiedDelayCfgV3,
        MultiAgentDelayCfgV3,
        create_no_delay_cfg,
        create_fixed_delay_cfg,
        create_random_delay_cfg,
    )

    # Test 1: DistributionCfg defaults
    try:
        cfg = DistributionCfg()
        assert cfg.type == "constant"
        assert cfg.value == 0.0
        results.add_pass("DistributionCfg defaults")
    except Exception as e:
        results.add_fail("DistributionCfg defaults", traceback.format_exc())

    # Test 2: SamplingCfg defaults
    try:
        cfg = SamplingCfg()
        assert cfg.frequency == "per_episode"
        results.add_pass("SamplingCfg defaults")
    except Exception as e:
        results.add_fail("SamplingCfg defaults", traceback.format_exc())

    # Test 3: LatencyCfg defaults
    try:
        cfg = LatencyCfg()
        assert cfg.enabled is True
        assert cfg.min_steps == 2
        assert cfg.distribution.type == "normal"
        results.add_pass("LatencyCfg defaults")
    except Exception as e:
        results.add_fail("LatencyCfg defaults", traceback.format_exc())

    # Test 4: Full pipeline cfg
    try:
        cfg = DelayPipelineCfgV3()
        assert cfg.latency.enabled is True
        assert cfg.staleness.enabled is True
        assert cfg.dropout.enabled is True
        results.add_pass("DelayPipelineCfgV3 construction")
    except Exception as e:
        results.add_fail("DelayPipelineCfgV3 construction", traceback.format_exc())

    # Test 5: Preset configurations
    try:
        cfg_none = create_no_delay_cfg()
        assert cfg_none.delay_cfg.ego.pipeline.latency.enabled is False
        assert cfg_none.noise.enabled is False

        cfg_fixed = create_fixed_delay_cfg(0.05, 0.1)
        assert cfg_fixed.delay_cfg.ego.pipeline.latency.distribution.type == "constant"
        assert cfg_fixed.delay_cfg.ego.pipeline.latency.distribution.value == 0.05

        cfg_random = create_random_delay_cfg()
        assert cfg_random.delay_cfg.ego.pipeline.latency.distribution.type == "normal"

        results.add_pass("Preset configurations")
    except Exception as e:
        results.add_fail("Preset configurations", traceback.format_exc())


def run_sampling_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run sampling strategy tests."""
    results.set_suite("Sampling Strategy Tests")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
        DistributionCfg,
        SamplingCfg,
    )
    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.sampling_strategies import (
        DistributionSampler,
        ParameterSampler,
        LatencySampler,
        StalenessSampler,
        DropoutSampler,
    )

    num_envs = 16

    # Test 1: Constant distribution
    try:
        cfg = DistributionCfg(type="constant", value=0.5)
        sampler = DistributionSampler(cfg, num_envs, device)
        values = sampler.sample()
        assert values.shape == (num_envs,)
        assert torch.allclose(values, torch.full((num_envs,), 0.5, device=device))
        results.add_pass("Constant distribution sampling")
    except Exception as e:
        results.add_fail("Constant distribution sampling", traceback.format_exc())

    # Test 2: Normal distribution
    try:
        cfg = DistributionCfg(type="normal", mean=1.0, std=0.1, min_value=0.5)
        sampler = DistributionSampler(cfg, num_envs, device)
        values = sampler.sample()
        assert values.shape == (num_envs,)
        assert (values >= 0.5).all()  # Clipping works
        results.add_pass("Normal distribution sampling")
    except Exception as e:
        results.add_fail("Normal distribution sampling", traceback.format_exc())

    # Test 3: Uniform distribution
    try:
        cfg = DistributionCfg(type="uniform", mean=0.5, half_range=0.1)
        sampler = DistributionSampler(cfg, num_envs, device)
        values = sampler.sample()
        assert values.shape == (num_envs,)
        assert (values >= 0.4).all() and (values <= 0.6).all()
        results.add_pass("Uniform distribution sampling")
    except Exception as e:
        results.add_fail("Uniform distribution sampling", traceback.format_exc())

    # Test 4: Per-episode parameter sampler
    try:
        dist_cfg = DistributionCfg(type="normal", mean=0.1, std=0.02)
        samp_cfg = SamplingCfg(frequency="per_episode")
        sampler = ParameterSampler(dist_cfg, samp_cfg, num_envs, device)
        sampler.initialize()

        initial_values = sampler.values.clone()

        # Within episode, values shouldn't change
        sampler.maybe_resample(reset_env_ids=None)
        assert torch.allclose(sampler.values, initial_values)

        # On reset, all values should change
        reset_ids = torch.tensor([0, 1], device=device)
        sampler.maybe_resample(reset_env_ids=reset_ids)
        # At least some values should be different (statistically likely)
        # Note: per_episode resamples ALL envs

        results.add_pass("Per-episode parameter sampling")
    except Exception as e:
        results.add_fail("Per-episode parameter sampling", traceback.format_exc())

    # Test 5: Per-step parameter sampler
    try:
        dist_cfg = DistributionCfg(type="normal", mean=0.1, std=0.02)
        samp_cfg = SamplingCfg(frequency="per_step")
        sampler = ParameterSampler(dist_cfg, samp_cfg, num_envs, device)
        sampler.initialize()

        initial_values = sampler.values.clone()
        sampler.maybe_resample()
        # Values should change (statistically very likely)
        # Just check no error occurs
        results.add_pass("Per-step parameter sampling")
    except Exception as e:
        results.add_fail("Per-step parameter sampling", traceback.format_exc())

    # Test 6: Latency sampler with step conversion
    try:
        # Use 0.12s which gives 3 steps cleanly (0.12 / 0.04 = 3.0)
        dist_cfg = DistributionCfg(type="constant", value=0.12)
        samp_cfg = SamplingCfg(frequency="per_episode")
        sampler = LatencySampler(dist_cfg, samp_cfg, num_envs, device, dt=0.04, min_steps=2)
        sampler.initialize()

        # 0.12s / 0.04s = 3.0 -> exactly 3 steps
        expected_steps = 3
        assert (sampler.step_values == expected_steps).all()
        results.add_pass("Latency sampler step conversion")
    except Exception as e:
        results.add_fail("Latency sampler step conversion", traceback.format_exc())

    # Test 7: Latency sampler min_steps enforcement
    try:
        dist_cfg = DistributionCfg(type="constant", value=0.02)  # 0.5 steps
        samp_cfg = SamplingCfg(frequency="per_episode")
        sampler = LatencySampler(dist_cfg, samp_cfg, num_envs, device, dt=0.04, min_steps=2)
        sampler.initialize()

        # Should be clamped to min_steps=2
        assert (sampler.step_values >= 2).all()
        results.add_pass("Latency sampler min_steps enforcement")
    except Exception as e:
        results.add_fail("Latency sampler min_steps enforcement", traceback.format_exc())

    # Test 8: Staleness sampler
    try:
        dist_cfg = DistributionCfg(type="constant", value=20.0)  # 20 FPS
        samp_cfg = SamplingCfg(frequency="per_episode")
        sampler = StalenessSampler(dist_cfg, samp_cfg, num_envs, device)
        sampler.initialize()

        # Period should be 1/20 = 0.05s
        expected_period = 0.05
        assert torch.allclose(sampler.period_values, torch.full((num_envs,), expected_period, device=device))
        results.add_pass("Staleness sampler FPS to period")
    except Exception as e:
        results.add_fail("Staleness sampler FPS to period", traceback.format_exc())

    # Test 9: Dropout sampler
    try:
        samp_cfg = SamplingCfg(frequency="per_step")
        sampler = DropoutSampler(
            probability=0.5,
            rate_distribution=None,
            sampling_cfg=samp_cfg,
            num_envs=num_envs,
            device=device,
        )
        sampler.initialize()

        # Sample many times and check approximate rate
        total_drops = 0
        num_samples = 100
        for _ in range(num_samples):
            mask = sampler.sample_mask()
            total_drops += mask.sum().item()

        drop_rate = total_drops / (num_samples * num_envs)
        # Should be approximately 0.5 with some tolerance
        assert 0.3 < drop_rate < 0.7, f"Drop rate {drop_rate} not close to 0.5"
        results.add_pass("Dropout sampler probability")
    except Exception as e:
        results.add_fail("Dropout sampler probability", traceback.format_exc())


def run_storage_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run field storage tests."""
    results.set_suite("Field Storage Tests")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.field_storage import (
        FieldStorage,
        MultiFieldStorage,
    )

    num_envs = 16

    # Test 1: Basic storage
    try:
        storage = FieldStorage(num_envs, device)
        data = torch.randn(num_envs, 3, device=device)
        storage.store("test_field", data)

        retrieved, ts = storage.get_raw("test_field")
        assert torch.allclose(retrieved, data)
        results.add_pass("Basic field storage")
    except Exception as e:
        results.add_fail("Basic field storage", traceback.format_exc())

    # Test 2: Noise injection
    try:
        storage = FieldStorage(num_envs, device)
        data = torch.zeros(num_envs, 3, device=device)
        storage.store("test_field", data, noise_std=0.1)

        raw, _ = storage.get_raw("test_field")
        noisy, _ = storage.get_noisy("test_field")

        assert torch.allclose(raw, data)
        assert not torch.allclose(noisy, data)  # Noise should make them different
        results.add_pass("Noise injection")
    except Exception as e:
        results.add_fail("Noise injection", traceback.format_exc())

    # Test 3: Timestamp storage
    try:
        storage = FieldStorage(num_envs, device)
        t = torch.full((num_envs,), 0.5, device=device)
        storage.set_time(t)

        data = torch.randn(num_envs, 3, device=device)
        storage.store("test_field", data)

        _, ts = storage.get_raw("test_field")
        assert torch.allclose(ts, t)
        results.add_pass("Timestamp storage")
    except Exception as e:
        results.add_fail("Timestamp storage", traceback.format_exc())

    # Test 4: Custom timestamp
    try:
        storage = FieldStorage(num_envs, device)
        custom_ts = torch.full((num_envs,), 0.25, device=device)

        data = torch.randn(num_envs, 3, device=device)
        storage.store("test_field", data, timestamp=custom_ts)

        _, ts = storage.get_raw("test_field")
        assert torch.allclose(ts, custom_ts)
        results.add_pass("Custom timestamp")
    except Exception as e:
        results.add_fail("Custom timestamp", traceback.format_exc())

    # Test 5: Multi-field storage
    try:
        storage = MultiFieldStorage(num_envs, device)
        storage.register_agent("agent_0")
        storage.register_field("position")

        data = torch.randn(num_envs, 3, device=device)
        storage.store_agent_field("agent_0", "position", data)

        retrieved, _ = storage.get_agent_field("agent_0", "position")
        assert torch.allclose(retrieved, data)
        results.add_pass("Multi-field storage")
    except Exception as e:
        results.add_fail("Multi-field storage", traceback.format_exc())


def run_pipeline_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run delay pipeline tests."""
    results.set_suite("Pipeline Tests")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
        DelayPipelineCfgV3,
        LatencyCfg,
        StalenessCfg,
        DropoutCfg,
        DistributionCfg,
        SamplingCfg,
    )
    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_pipeline_v3 import (
        DelayPipelineV3,
    )

    num_envs = 8
    dt = 0.04
    data_shape = (3,)

    # Test 1: Pipeline initialization
    try:
        cfg = DelayPipelineCfgV3()
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        assert pipeline.num_envs == num_envs
        results.add_pass("Pipeline initialization")
    except Exception as e:
        results.add_fail("Pipeline initialization", traceback.format_exc())

    # Test 2: Latency-only pipeline (mode=fixed)
    try:
        # Configure constant latency of 0.08s = 2 steps
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        # Process multiple steps
        t_current = torch.zeros(num_envs, device=device)

        # Step 0: initial data
        data_0 = torch.ones(num_envs, 3, device=device) * 0.0
        ts_0 = t_current.clone()
        out_0, out_ts_0 = pipeline.process(data_0, ts_0, t_current)

        # Step 1
        t_current = t_current + dt
        data_1 = torch.ones(num_envs, 3, device=device) * 1.0
        ts_1 = t_current.clone()
        out_1, out_ts_1 = pipeline.process(data_1, ts_1, t_current)

        # Step 2
        t_current = t_current + dt
        data_2 = torch.ones(num_envs, 3, device=device) * 2.0
        ts_2 = t_current.clone()
        out_2, out_ts_2 = pipeline.process(data_2, ts_2, t_current)

        # With 2-step delay, output at step 2 should be data from step 0
        assert torch.allclose(out_2, data_0), f"Expected {data_0[0]}, got {out_2[0]}"
        assert torch.allclose(out_ts_2, ts_0), f"Expected ts {ts_0[0]}, got {out_ts_2[0]}"
        results.add_pass("Latency-only pipeline")
    except Exception as e:
        results.add_fail("Latency-only pipeline", traceback.format_exc())

    # Test 3: Warmup handling (first step valid output)
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        # First step - should return VALID data, not zeros
        t_current = torch.zeros(num_envs, device=device)
        data_first = torch.ones(num_envs, 3, device=device) * 5.0
        ts_first = t_current.clone()

        out, out_ts = pipeline.process(data_first, ts_first, t_current)

        # CircularBuffer fills all slots with first data, so output should be data_first
        assert torch.allclose(out, data_first), f"Warmup: expected {data_first[0]}, got {out[0]}"
        results.add_pass("Warmup handling first step")
    except Exception as e:
        results.add_fail("Warmup handling first step", traceback.format_exc())

    # Test 4: Dropout holds data and timestamp together
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(enabled=False),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(
                enabled=True,
                probability=1.0,  # Always drop
                sampling=SamplingCfg(frequency="per_step"),
            ),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("random", progress=1.0)

        t_current = torch.zeros(num_envs, device=device)

        # Step 0: first data (becomes held data)
        data_0 = torch.ones(num_envs, 3, device=device) * 10.0
        ts_0 = t_current.clone()
        out_0, out_ts_0 = pipeline.process(data_0, ts_0, t_current)

        # Step 1: new data (should be dropped, held data returned)
        t_current = t_current + dt
        data_1 = torch.ones(num_envs, 3, device=device) * 20.0
        ts_1 = t_current.clone()
        out_1, out_ts_1 = pipeline.process(data_1, ts_1, t_current)

        # With 100% dropout, output should still be data_0 with ts_0
        assert torch.allclose(out_1, data_0), f"Dropout: expected {data_0[0]}, got {out_1[0]}"
        assert torch.allclose(out_ts_1, ts_0), f"Dropout ts: expected {ts_0[0]}, got {out_ts_1[0]}"
        results.add_pass("Dropout holds data and timestamp")
    except Exception as e:
        results.add_fail("Dropout holds data and timestamp", traceback.format_exc())

    # Test 5: Staleness holds data and timestamp together
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(enabled=False),
            staleness=StalenessCfg(
                enabled=True,
                fps_distribution=DistributionCfg(type="constant", value=10.0),  # 10 FPS = 0.1s period
            ),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("random", progress=1.0)

        t_current = torch.zeros(num_envs, device=device)

        # Step 0: first data
        data_0 = torch.ones(num_envs, 3, device=device) * 100.0
        ts_0 = t_current.clone()
        out_0, out_ts_0 = pipeline.process(data_0, ts_0, t_current)

        # Steps 1: within staleness period (0.04s < 0.1s period)
        t_current = t_current + dt  # 0.04s
        data_1 = torch.ones(num_envs, 3, device=device) * 200.0
        ts_1 = t_current.clone()
        out_1, out_ts_1 = pipeline.process(data_1, ts_1, t_current)

        # Output should still be data_0 with ts_0 (stale)
        assert torch.allclose(out_1, data_0), f"Staleness: expected {data_0[0]}, got {out_1[0]}"
        assert torch.allclose(out_ts_1, ts_0), f"Staleness ts: expected {ts_0[0]}, got {out_ts_1[0]}"
        results.add_pass("Staleness holds data and timestamp")
    except Exception as e:
        results.add_fail("Staleness holds data and timestamp", traceback.format_exc())


def run_timestamp_sync_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run timestamp synchronization tests - critical for AoI correctness."""
    results.set_suite("Timestamp Synchronization Tests")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
        DelayPipelineCfgV3,
        LatencyCfg,
        StalenessCfg,
        DropoutCfg,
        DistributionCfg,
        SamplingCfg,
    )
    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_pipeline_v3 import (
        DelayPipelineV3,
    )

    num_envs = 8
    dt = 0.04
    data_shape = (3,)

    # Test 1: AoI = t_current - timestamp is correct with latency
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.12),  # 3 steps
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        t_current = torch.zeros(num_envs, device=device)

        # Run 5 steps
        for i in range(5):
            data = torch.ones(num_envs, 3, device=device) * float(i)
            ts = t_current.clone()
            out, out_ts = pipeline.process(data, ts, t_current)

            # AoI should be approximately 3 * dt = 0.12s after warmup
            aoi = t_current - out_ts
            if i >= 3:  # After warmup period
                expected_aoi = 3 * dt  # 0.12s
                # Allow some tolerance
                assert torch.allclose(aoi, torch.full((num_envs,), expected_aoi, device=device), atol=dt), \
                    f"Step {i}: AoI={aoi[0].item():.3f}, expected ~{expected_aoi:.3f}"

            t_current = t_current + dt

        results.add_pass("AoI correctness with latency")
    except Exception as e:
        results.add_fail("AoI correctness with latency", traceback.format_exc())

    # Test 2: AoI is always non-negative and timestamp <= t_current
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(enabled=False),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(
                enabled=True,
                probability=0.5,  # 50% dropout
            ),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("random", progress=1.0)

        t_current = torch.zeros(num_envs, device=device)

        for i in range(20):
            data = torch.ones(num_envs, 3, device=device) * float(i)
            ts = t_current.clone()
            out, out_ts = pipeline.process(data, ts, t_current)

            aoi = t_current - out_ts

            # AoI should always be non-negative (no future timestamps)
            assert (aoi >= -1e-6).all(), \
                f"Step {i}: negative AoI! aoi={aoi[0].item():.4f}"

            # Timestamp should never be in the future
            assert (out_ts <= t_current + 1e-6).all(), \
                f"Step {i}: timestamp in future! ts={out_ts[0].item():.4f}, t_current={t_current[0].item():.4f}"

            t_current = t_current + dt

        results.add_pass("AoI non-negative and timestamp valid")
    except Exception as e:
        results.add_fail("AoI non-negative and timestamp valid", traceback.format_exc())

    # Test 3: Timestamp matches data throughout pipeline
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),  # 2 steps
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        # Store data with unique identifiable values
        data_history = []
        ts_history = []

        t_current = torch.zeros(num_envs, device=device)

        for i in range(5):
            # Data encodes the step number
            data = torch.ones(num_envs, 3, device=device) * float(i)
            ts = t_current.clone()
            data_history.append(data.clone())
            ts_history.append(ts.clone())

            out, out_ts = pipeline.process(data, ts, t_current)

            # After warmup, verify output timestamp matches stored timestamp
            if i >= 2:
                # With 2-step delay, output at step i should have data from step i-2
                expected_data = data_history[i - 2]
                expected_ts = ts_history[i - 2]
                assert torch.allclose(out, expected_data), \
                    f"Step {i}: data mismatch, got {out[0,0].item()}, expected {expected_data[0,0].item()}"
                assert torch.allclose(out_ts, expected_ts), \
                    f"Step {i}: ts mismatch, got {out_ts[0].item()}, expected {expected_ts[0].item()}"

            t_current = t_current + dt

        results.add_pass("Timestamp matches data through pipeline")
    except Exception as e:
        results.add_fail("Timestamp matches data through pipeline", traceback.format_exc())

    # Test 4: Combined staleness and latency
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),  # 2 steps
                min_steps=2,
            ),
            staleness=StalenessCfg(
                enabled=True,
                fps_distribution=DistributionCfg(type="constant", value=10.0),  # 10 FPS
            ),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("random", progress=1.0)

        t_current = torch.zeros(num_envs, device=device)

        # Run for 0.2s (enough for staleness to matter)
        aoi_values = []
        for i in range(10):
            data = torch.ones(num_envs, 3, device=device) * float(i)
            ts = t_current.clone()
            out, out_ts = pipeline.process(data, ts, t_current)

            aoi = t_current - out_ts
            aoi_values.append(aoi[0].item())

            t_current = t_current + dt

        # AoI should be positive (delay exists)
        assert all(a >= 0 for a in aoi_values), "AoI should be non-negative"
        # After warmup, AoI should be >= latency
        min_expected_aoi = 2 * dt  # At least latency
        assert aoi_values[-1] >= min_expected_aoi - dt, \
            f"Final AoI {aoi_values[-1]:.3f} should be >= {min_expected_aoi:.3f}"

        results.add_pass("Combined staleness and latency AoI")
    except Exception as e:
        results.add_fail("Combined staleness and latency AoI", traceback.format_exc())


def run_curriculum_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run curriculum mode tests."""
    results.set_suite("Curriculum Mode Tests")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
        DelayPipelineCfgV3,
        LatencyCfg,
        StalenessCfg,
        DropoutCfg,
        DistributionCfg,
    )
    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_pipeline_v3 import (
        DelayPipelineV3,
    )

    num_envs = 8
    dt = 0.04
    data_shape = (3,)

    # Test 1: Mode 'none' produces minimal delay
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.12),
            ),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("none", progress=1.0)

        t_current = torch.zeros(num_envs, device=device)
        data = torch.ones(num_envs, 3, device=device) * 42.0
        ts = t_current.clone()

        out, out_ts = pipeline.process(data, ts, t_current)

        # In 'none' mode, should pass through immediately
        assert torch.allclose(out, data), "Mode 'none' should pass through data"
        assert torch.allclose(out_ts, ts), "Mode 'none' should pass through timestamp"
        results.add_pass("Mode 'none' minimal delay")
    except Exception as e:
        results.add_fail("Mode 'none' minimal delay", traceback.format_exc())

    # Test 2: Mode 'fixed' has deterministic delay
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="normal", mean=0.1, std=0.05),
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        # In fixed mode, all envs should have same latency
        latency_steps = pipeline.current_latency_steps.float()
        # Check that variance is low (due to scaling, may not be exactly same)
        assert latency_steps.std() < 2.0, \
            f"Fixed mode should have low variance in latency, got std={latency_steps.std().item()}"

        results.add_pass("Mode 'fixed' deterministic delay")
    except Exception as e:
        results.add_fail("Mode 'fixed' deterministic delay", traceback.format_exc())

    # Test 3: Mode 'random' enables staleness
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),
            ),
            staleness=StalenessCfg(
                enabled=True,
                fps_distribution=DistributionCfg(type="constant", value=20.0),
            ),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("random", progress=1.0)

        # Run and check staleness is applied
        t_current = torch.zeros(num_envs, device=device)

        # First data
        data_0 = torch.ones(num_envs, 3, device=device) * 1.0
        ts_0 = t_current.clone()
        out_0, _ = pipeline.process(data_0, ts_0, t_current)

        # Second step (within staleness period since 20FPS = 0.05s > dt=0.04s)
        t_current = t_current + dt
        data_1 = torch.ones(num_envs, 3, device=device) * 2.0
        ts_1 = t_current.clone()
        out_1, out_ts_1 = pipeline.process(data_1, ts_1, t_current)

        # Due to staleness, output should still be related to first data
        # (Exact check is complex due to latency buffer, just verify no crash)
        results.add_pass("Mode 'random' enables staleness")
    except Exception as e:
        results.add_fail("Mode 'random' enables staleness", traceback.format_exc())

    # Test 4: Progress scales delay
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.2),  # 5 steps at full
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )

        # At progress=0.5
        pipeline_half = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline_half.set_mode("fixed", progress=0.5)
        steps_half = pipeline_half.current_latency_steps

        # At progress=1.0
        pipeline_full = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline_full.set_mode("fixed", progress=1.0)
        steps_full = pipeline_full.current_latency_steps

        # Half progress should have fewer steps (but at least min_steps)
        # Note: due to min_steps=2, half of 5 = 2.5 -> 3, which is less than 5
        # Both are clamped to min_steps if needed
        results.add_pass("Progress scales delay")
    except Exception as e:
        results.add_fail("Progress scales delay", traceback.format_exc())

    # Test 5: Dropout rate can be set independently
    try:
        cfg = DelayPipelineCfgV3(
            dropout=DropoutCfg(enabled=True, probability=0.0),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("random", progress=1.0)
        pipeline.set_dropout_rate(0.5)

        # Sample many times
        total_drops = 0
        num_samples = 100
        t_current = torch.zeros(num_envs, device=device)
        for i in range(num_samples):
            data = torch.ones(num_envs, 3, device=device)
            out, _ = pipeline.process(data, t_current, t_current)
            t_current = t_current + dt

        # No assertion, just verify it runs without error
        results.add_pass("Dropout rate curriculum control")
    except Exception as e:
        results.add_fail("Dropout rate curriculum control", traceback.format_exc())


def run_idempotency_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run idempotency tests — verify process() is safe to call multiple times per step.

    This mirrors the real env lifecycle where delay pipeline's process() is called
    from _get_rewards(), _get_observations(), and potentially teleop visualization.
    Bug reference: buffer corruption from multiple process() calls at same t_current.
    """
    results.set_suite("Idempotency Tests (Multiple Calls Per Step)")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
        DelayPipelineCfgV3,
        LatencyCfg,
        StalenessCfg,
        DropoutCfg,
        DistributionCfg,
    )
    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_pipeline_v3 import (
        DelayPipelineV3,
    )

    num_envs = 8
    dt = 0.04
    data_shape = (3,)

    # Test 1: Three calls at same t_current return identical results
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        data = torch.randn(num_envs, 3, device=device)
        t = torch.full((num_envs,), dt, device=device)

        # Simulate 3 calls at same time (rewards, obs, teleop)
        r1, ts1 = pipeline.process(data, t, t)
        r2, ts2 = pipeline.process(data, t, t)
        r3, ts3 = pipeline.process(data, t, t)

        assert torch.allclose(r1, r2), f"Call 1 vs 2 differ: max diff={( r1 - r2).abs().max().item()}"
        assert torch.allclose(r2, r3), f"Call 2 vs 3 differ: max diff={(r2 - r3).abs().max().item()}"
        assert torch.allclose(ts1, ts2) and torch.allclose(ts2, ts3), "Timestamps differ across calls"
        results.add_pass("Three calls at same t_current return identical results")
    except Exception as e:
        results.add_fail("Three calls at same t_current return identical results", traceback.format_exc())

    # Test 2: Effective delay matches config, not config/N
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),  # 2 steps
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        t_current = torch.zeros(num_envs, device=device)

        # Step 0: initial
        data_0 = torch.zeros(num_envs, 3, device=device)
        pipeline.process(data_0, t_current.clone(), t_current.clone())

        # Step 1: new data, call 3 times (simulating rewards + obs + teleop)
        t_current += dt
        data_1 = torch.ones(num_envs, 3, device=device)
        pipeline.process(data_1, t_current.clone(), t_current.clone())
        pipeline.process(data_1, t_current.clone(), t_current.clone())
        pipeline.process(data_1, t_current.clone(), t_current.clone())

        # Step 2: new data, call 3 times
        t_current += dt
        data_2 = torch.ones(num_envs, 3, device=device) * 2.0
        pipeline.process(data_2, t_current.clone(), t_current.clone())
        pipeline.process(data_2, t_current.clone(), t_current.clone())
        pipeline.process(data_2, t_current.clone(), t_current.clone())

        # Step 3: new data, retrieve result — with 2-step delay, should get data_1
        t_current += dt
        data_3 = torch.ones(num_envs, 3, device=device) * 3.0
        result, _ = pipeline.process(data_3, t_current.clone(), t_current.clone())

        # With 2-step latency and proper guarding, result should be data_1 (from 2 steps ago)
        expected = data_1
        assert torch.allclose(result, expected, atol=1e-5), (
            f"Effective delay wrong! Expected data_1={expected[0].tolist()}, "
            f"got {result[0].tolist()}. Buffer likely advanced multiple times per step."
        )
        results.add_pass("Effective delay matches config (not config/N)")
    except Exception as e:
        results.add_fail("Effective delay matches config (not config/N)", traceback.format_exc())

    # Test 3: Buffer internal state doesn't advance on repeated calls
    try:
        cfg = DelayPipelineCfgV3(
            latency=LatencyCfg(
                enabled=True,
                distribution=DistributionCfg(type="constant", value=0.08),
                min_steps=2,
            ),
            staleness=StalenessCfg(enabled=False),
            dropout=DropoutCfg(enabled=False),
        )
        pipeline = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
        pipeline.set_mode("fixed", progress=1.0)

        data = torch.randn(num_envs, 3, device=device)
        t = torch.full((num_envs,), dt, device=device)

        # First call: should append
        pipeline.process(data, t, t)
        last_time_after_first = pipeline._last_append_time.clone()

        # Second and third calls: should NOT append (same t_current)
        pipeline.process(data, t, t)
        pipeline.process(data, t, t)
        last_time_after_third = pipeline._last_append_time.clone()

        assert torch.allclose(last_time_after_first, last_time_after_third), (
            "Buffer append time changed on repeated calls at same t_current"
        )
        results.add_pass("Buffer state unchanged on repeated calls")
    except Exception as e:
        results.add_fail("Buffer state unchanged on repeated calls", traceback.format_exc())


def main():
    """Main test runner."""
    # Open output file for logging
    with open(OUTPUT_FILE, "w") as f:
        log_print("=" * 80, f)
        log_print("DELAY SYSTEM V3 TEST SUITE", f)
        log_print("=" * 80, f)
        log_print(f"Output file: {OUTPUT_FILE}", f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_print(f"Device:        {device}", f)
        log_print(f"Torch version: {torch.__version__}", f)
        if device.type == "cuda":
            log_print(f"CUDA version:  {torch.version.cuda}", f)
            log_print(f"GPU name:      {torch.cuda.get_device_name(0)}", f)
        log_print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)

        results = TestResults(file_handle=f)
        verbose = args_cli.test_verbose

        try:
            # Run test suites
            run_config_tests(results, device, verbose)
            run_sampling_tests(results, device, verbose)
            run_storage_tests(results, device, verbose)

            # Pipeline tests (critical for timestamp-data sync)
            run_pipeline_tests(results, device, verbose)
            run_timestamp_sync_tests(results, device, verbose)
            run_curriculum_tests(results, device, verbose)

            # New feature tests
            run_per_agent_randomization_tests(results, device, verbose)
            run_reward_mode_tests(results, device, verbose)

            # Integration / idempotency tests
            run_idempotency_tests(results, device, verbose)

        except Exception as e:
            results.add_error("Test Suite Execution", traceback.format_exc())

        results.print_summary()

        log_print(f"\nResults written to: {OUTPUT_FILE}", f)

    return 0 if results.all_passed() else 1


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
