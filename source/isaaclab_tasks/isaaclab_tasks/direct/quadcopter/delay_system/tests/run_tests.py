#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone test runner for the delay system module.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py

    # With verbose output
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py --test-verbose

    # CPU only
    CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run delay system test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import other modules
import sys
import traceback
from datetime import datetime

import torch

# Import the delay system components
from isaaclab_tasks.direct.quadcopter.delay_system import DelayCfg, DelaySystem, DelaySystemCfg

# Import V2 components
from isaaclab_tasks.direct.quadcopter.delay_system import (
    DataBus,
    DelayPipeline,
    DelaySystemV2,
    DelaySystemCfgV2,
    DistributionCfg,
    FieldDelayCfg,
    NoiseCfg,
    DerivedFieldComputer,
    DerivedFieldDef,
)


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  \u2713 {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  \u2717 {test_name}")
        error_lines = error.split("\n")[:5]
        for line in error_lines:
            print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        print(f"    {error[:200]}")

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80, flush=True)
        print("TEST SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"Total Tests: {total}", flush=True)
        if total > 0:
            print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)", flush=True)
            print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)", flush=True)
            print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)", flush=True)

        if self.failed:
            print("\n" + "-" * 80, flush=True)
            print("FAILED TESTS:", flush=True)
            print("-" * 80, flush=True)
            for test_name, error in self.failed:
                print(f"\n{test_name}:", flush=True)
                print(f"  {error}", flush=True)

        if self.errors:
            print("\n" + "-" * 80, flush=True)
            print("TEST ERRORS:", flush=True)
            print("-" * 80, flush=True)
            for test_name, error in self.errors:
                print(f"\n{test_name}:", flush=True)
                print(f"  {error}", flush=True)

        print("=" * 80, flush=True)
        return len(self.failed) == 0 and len(self.errors) == 0


def run_delay_cfg_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test DelayCfg configuration class."""
    print("\n" + "=" * 80)
    print("Testing DelayCfg")
    print("=" * 80)

    # Test 1: Default initialization
    try:
        cfg = DelayCfg()
        assert cfg.enabled is False, f"Expected enabled=False, got {cfg.enabled}"
        assert cfg.min_delay == 0, f"Expected min_delay=0, got {cfg.min_delay}"
        assert cfg.max_delay == 0, f"Expected max_delay=0, got {cfg.max_delay}"
        assert cfg.randomize_on_reset is True, f"Expected randomize_on_reset=True, got {cfg.randomize_on_reset}"
        results.add_pass("Default initialization")
    except Exception as e:
        results.add_fail("Default initialization", traceback.format_exc())

    # Test 2: Custom initialization
    try:
        cfg = DelayCfg(enabled=True, min_delay=5, max_delay=10, randomize_on_reset=False)
        assert cfg.enabled is True
        assert cfg.min_delay == 5
        assert cfg.max_delay == 10
        assert cfg.randomize_on_reset is False
        results.add_pass("Custom initialization")
    except Exception as e:
        results.add_fail("Custom initialization", traceback.format_exc())

    # Test 3: Validation - negative min_delay
    try:
        try:
            cfg = DelayCfg(enabled=True, min_delay=-1, max_delay=10)
            results.add_fail("Validation - negative min_delay", "Should have raised ValueError")
        except ValueError as e:
            if "non-negative" in str(e):
                results.add_pass("Validation - negative min_delay")
            else:
                results.add_fail("Validation - negative min_delay", f"Wrong error message: {e}")
    except Exception as e:
        results.add_fail("Validation - negative min_delay", traceback.format_exc())

    # Test 4: Validation - max_delay < min_delay
    try:
        try:
            cfg = DelayCfg(enabled=True, min_delay=10, max_delay=5)
            results.add_fail("Validation - max_delay < min_delay", "Should have raised ValueError")
        except ValueError as e:
            if "must be >=" in str(e):
                results.add_pass("Validation - max_delay < min_delay")
            else:
                results.add_fail("Validation - max_delay < min_delay", f"Wrong error message: {e}")
    except Exception as e:
        results.add_fail("Validation - max_delay < min_delay", traceback.format_exc())


def run_delay_system_cfg_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test DelaySystemCfg configuration class."""
    print("\n" + "=" * 80)
    print("Testing DelaySystemCfg")
    print("=" * 80)

    # Test 1: Default initialization
    try:
        cfg = DelaySystemCfg()
        assert cfg.action_delay.enabled is False
        assert cfg.observation_delay.enabled is False
        results.add_pass("Default initialization")
    except Exception as e:
        results.add_fail("Default initialization", traceback.format_exc())

    # Test 2: Action delay only
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
        )
        assert cfg.action_delay.enabled is True
        assert cfg.action_delay.min_delay == 5
        assert cfg.action_delay.max_delay == 10
        assert cfg.observation_delay.enabled is False
        results.add_pass("Action delay only configuration")
    except Exception as e:
        results.add_fail("Action delay only configuration", traceback.format_exc())

    # Test 3: Both delays configured
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
        )
        assert cfg.action_delay.enabled is True
        assert cfg.observation_delay.enabled is True
        assert cfg.action_delay.max_delay == 10
        assert cfg.observation_delay.max_delay == 5
        results.add_pass("Both delays configured")
    except Exception as e:
        results.add_fail("Both delays configured", traceback.format_exc())


def run_delay_system_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test DelaySystem class."""
    print("\n" + "=" * 80)
    print("Testing DelaySystem")
    print("=" * 80)

    num_envs = 16
    action_dim = 4
    observation_dim = 12

    print(f"Configuration: {num_envs} environments, action_dim={action_dim}, obs_dim={observation_dim}")
    print(f"Device: {device}")

    # Test 1: Initialization with delays disabled
    try:
        cfg = DelaySystemCfg()
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        assert system.action_delay_enabled is False
        assert system.observation_delay_enabled is False
        assert system.action_buffer is None
        assert system.observation_buffer is None
        results.add_pass("Initialization with delays disabled")
    except Exception as e:
        results.add_fail("Initialization with delays disabled", traceback.format_exc())

    # Test 2: Initialization with action delay enabled
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        assert system.action_delay_enabled is True
        assert system.observation_delay_enabled is False
        assert system.action_buffer is not None
        assert system.action_buffer.history_length == 10
        results.add_pass("Initialization with action delay enabled")
    except Exception as e:
        results.add_fail("Initialization with action delay enabled", traceback.format_exc())

    # Test 3: Initialization with both delays enabled
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        assert system.action_delay_enabled is True
        assert system.observation_delay_enabled is True
        assert system.action_buffer.history_length == 10
        assert system.observation_buffer.history_length == 5
        results.add_pass("Initialization with both delays enabled")
    except Exception as e:
        results.add_fail("Initialization with both delays enabled", traceback.format_exc())

    # Test 4: Compute delayed action (no delay)
    try:
        cfg = DelaySystemCfg()
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        action = torch.randn(num_envs, action_dim, device=device)
        delayed_action = system.compute_delayed_action(action)
        assert torch.allclose(action, delayed_action), "With no delay, action should be unchanged"
        results.add_pass("Compute delayed action (no delay buffer)")
    except Exception as e:
        results.add_fail("Compute delayed action (no delay buffer)", traceback.format_exc())

    # Test 5: Compute delayed observation (no delay)
    try:
        cfg = DelaySystemCfg()
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        obs = torch.randn(num_envs, observation_dim, device=device)
        delayed_obs = system.compute_delayed_observation(obs)
        assert torch.allclose(obs, delayed_obs), "With no delay, observation should be unchanged"
        results.add_pass("Compute delayed observation (no delay buffer)")
    except Exception as e:
        results.add_fail("Compute delayed observation (no delay buffer)", traceback.format_exc())

    # Test 6: Compute delayed action with fixed delay
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=2, max_delay=2, randomize_on_reset=False),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        system.reset()

        # Check delay is set correctly
        delays = system.get_action_delays()
        assert delays is not None
        assert torch.all(delays == 2), f"Expected all delays to be 2, got {delays}"

        # Send several actions
        actions = []
        for i in range(5):
            action = torch.full((num_envs, action_dim), float(i), device=device)
            actions.append(action.clone())
            delayed = system.compute_delayed_action(action)

            if verbose:
                print(f"  Step {i}: sent {i}, received mean={delayed.mean().item():.1f}")

        # After sending 0,1,2,3,4 with delay=2, the 5th output (index 4) should be 2
        # Because: buffer has [0,1,2,3,4], delay=2 means get item at index 2 from current
        # Actually, DelayBuffer returns data from (current - delay) position
        # So after step 4, we should get value 2 (which was sent 2 steps ago)
        results.add_pass("Compute delayed action with fixed delay")
    except Exception as e:
        results.add_fail("Compute delayed action with fixed delay", traceback.format_exc())

    # Test 7: Reset functionality
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))

        # Fill buffers with data
        for _ in range(15):
            system.compute_delayed_action(torch.randn(num_envs, action_dim, device=device))
            system.compute_delayed_observation(torch.randn(num_envs, observation_dim, device=device))

        # Reset specific environments
        env_ids = torch.tensor([0, 5, 10], device=device)
        system.reset(env_ids)

        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", traceback.format_exc())

    # Test 8: Delay randomization
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=0, max_delay=10, randomize_on_reset=True),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        system.reset()

        delays = system.get_action_delays()
        assert delays is not None
        assert torch.all(delays >= 0) and torch.all(delays <= 10), "Delays out of range"

        # Check that delays are not all the same (with high probability)
        unique_delays = torch.unique(delays)
        if verbose:
            print(f"  Unique delays: {unique_delays.tolist()}")
        # With 16 envs and range 0-10, very likely to have multiple unique values
        assert len(unique_delays) > 1 or num_envs <= 2, "Expected randomized delays"

        results.add_pass("Delay randomization")
    except Exception as e:
        results.add_fail("Delay randomization", traceback.format_exc())

    # Test 9: Get delay info
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        system.reset()

        info = system.get_delay_info()
        assert "action_delay_min" in info
        assert "action_delay_max" in info
        assert "action_delay_mean" in info
        assert "observation_delay_min" in info
        assert "observation_delay_max" in info
        assert "observation_delay_mean" in info

        if verbose:
            print(f"  Delay info: {info}")

        results.add_pass("Get delay info")
    except Exception as e:
        results.add_fail("Get delay info", traceback.format_exc())

    # Test 10: Manual delay setting
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=0, max_delay=10),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))

        # Set all to same delay
        system.set_action_delays(5)
        delays = system.get_action_delays()
        assert torch.all(delays == 5), f"Expected all delays to be 5, got {delays}"

        # Set specific environments
        system.set_action_delays(8, [0, 1, 2])
        delays = system.get_action_delays()
        assert delays[0] == 8 and delays[1] == 8 and delays[2] == 8
        assert delays[3] == 5  # unchanged

        results.add_pass("Manual delay setting")
    except Exception as e:
        results.add_fail("Manual delay setting", traceback.format_exc())


def run_integration_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Integration tests simulating real usage patterns."""
    print("\n" + "=" * 80)
    print("Integration Tests")
    print("=" * 80)

    num_envs = 64
    action_dim = 4
    observation_dim = 12

    # Test 1: Simulated training loop
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
            observation_delay=DelayCfg(enabled=True, min_delay=1, max_delay=3),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        system.reset()

        print("  Running 100-step simulation...", end="", flush=True)

        for step in range(100):
            if step % 25 == 0 and step > 0:
                print(f".{step}", end="", flush=True)

            # Simulate policy output
            actions = torch.randn(num_envs, action_dim, device=device)
            delayed_actions = system.compute_delayed_action(actions)
            assert delayed_actions.shape == (num_envs, action_dim)

            # Simulate observations
            observations = torch.randn(num_envs, observation_dim, device=device)
            delayed_obs = system.compute_delayed_observation(observations)
            assert delayed_obs.shape == (num_envs, observation_dim)

            # Simulate some resets
            if step % 20 == 0:
                reset_ids = torch.randint(0, num_envs, (num_envs // 4,), device=device)
                system.reset(reset_ids)

        print(" done")
        results.add_pass("Simulated training loop (100 steps)")
    except Exception as e:
        print(" FAILED")
        results.add_fail("Simulated training loop (100 steps)", traceback.format_exc())

    # Test 2: Verify delay actually delays data
    try:
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=3, max_delay=3, randomize_on_reset=False),
        )
        system = DelaySystem(cfg, num_envs, action_dim, observation_dim, str(device))
        system.reset()

        # Send sequence of distinguishable values
        sent_values = []
        received_values = []

        for i in range(10):
            action = torch.full((num_envs, action_dim), float(i * 10), device=device)
            sent_values.append(i * 10)
            delayed = system.compute_delayed_action(action)
            received_values.append(delayed[0, 0].item())

        if verbose:
            print(f"  Sent: {sent_values}")
            print(f"  Received: {received_values}")

        # After the initial filling period, received should lag by 3
        # After step 3, we send 30, but should receive 0 (sent 3 steps ago)
        # After step 4, we send 40, but should receive 10
        # etc.
        for i in range(3, 10):
            expected = (i - 3) * 10
            actual = received_values[i]
            assert abs(actual - expected) < 0.01, f"At step {i}, expected {expected}, got {actual}"

        results.add_pass("Verify delay timing (3-step fixed delay)")
    except Exception as e:
        results.add_fail("Verify delay timing (3-step fixed delay)", traceback.format_exc())

    # Test 3: Large batch size
    try:
        large_num_envs = 4096
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=15),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=8),
        )
        system = DelaySystem(cfg, large_num_envs, action_dim, observation_dim, str(device))
        system.reset()

        for _ in range(20):
            actions = torch.randn(large_num_envs, action_dim, device=device)
            delayed_actions = system.compute_delayed_action(actions)
            assert delayed_actions.shape == (large_num_envs, action_dim)

        info = system.get_delay_info()
        assert info["action_delay_min"] >= 5
        assert info["action_delay_max"] <= 15

        results.add_pass(f"Large batch size ({large_num_envs} envs)")
    except Exception as e:
        results.add_fail(f"Large batch size ({large_num_envs} envs)", traceback.format_exc())


# ==============================================================================
# V2 Component Tests
# ==============================================================================


def run_data_bus_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test DataBus class."""
    print("\n" + "=" * 80)
    print("Testing DataBus (V2)")
    print("=" * 80)

    num_envs = 16

    # Test 1: Initialization
    try:
        data_bus = DataBus(num_envs=num_envs, device=device)
        assert data_bus.num_envs == num_envs
        assert len(data_bus.clean) == 0
        assert len(data_bus.noisy) == 0
        assert data_bus.t_current.shape == (num_envs,)
        results.add_pass("DataBus initialization")
    except Exception as e:
        results.add_fail("DataBus initialization", traceback.format_exc())

    # Test 2: Store data without noise
    try:
        data_bus = DataBus(num_envs=num_envs, device=device)
        data = torch.randn(num_envs, 3, device=device)
        data_bus.store("test_field", data, noise_std=0.0)

        assert data_bus.has_field("test_field")
        clean = data_bus.get_clean("test_field")
        noisy = data_bus.get_noisy("test_field")
        assert torch.allclose(clean, data)
        assert torch.allclose(noisy, data)  # No noise, should be same
        results.add_pass("Store data without noise")
    except Exception as e:
        results.add_fail("Store data without noise", traceback.format_exc())

    # Test 3: Store data with noise
    try:
        data_bus = DataBus(num_envs=num_envs, device=device)
        data = torch.zeros(num_envs, 3, device=device)
        noise_std = 1.0
        data_bus.store("test_field", data, noise_std=noise_std)

        clean = data_bus.get_clean("test_field")
        noisy = data_bus.get_noisy("test_field")

        assert torch.allclose(clean, data)  # Clean should be unchanged
        assert not torch.allclose(noisy, data)  # Noisy should be different

        # Check noise magnitude is reasonable (roughly within 3 sigma)
        noise_magnitude = noisy.std().item()
        assert 0.5 < noise_magnitude < 2.0, f"Noise std {noise_magnitude} seems wrong"

        if verbose:
            print(f"  Noise std: {noise_magnitude:.3f} (expected ~{noise_std})")

        results.add_pass("Store data with noise")
    except Exception as e:
        results.add_fail("Store data with noise", traceback.format_exc())

    # Test 4: Step time advancement
    try:
        data_bus = DataBus(num_envs=num_envs, device=device)
        assert torch.all(data_bus.t_current == 0.0)

        data_bus.step(0.01)
        assert torch.allclose(data_bus.t_current, torch.full((num_envs,), 0.01, device=device))

        data_bus.step(0.01)
        assert torch.allclose(data_bus.t_current, torch.full((num_envs,), 0.02, device=device))

        results.add_pass("Step time advancement")
    except Exception as e:
        results.add_fail("Step time advancement", traceback.format_exc())

    # Test 5: Reset specific environments
    try:
        data_bus = DataBus(num_envs=num_envs, device=device)
        data = torch.ones(num_envs, 3, device=device)
        data_bus.store("test_field", data)
        data_bus.step(0.1)

        # Reset first 5 environments
        env_ids = torch.arange(5, device=device)
        data_bus.reset(env_ids)

        # Check time reset
        assert torch.all(data_bus.t_current[:5] == 0.0)
        assert torch.all(data_bus.t_current[5:] == 0.1)

        results.add_pass("Reset specific environments")
    except Exception as e:
        results.add_fail("Reset specific environments", traceback.format_exc())


def run_delay_pipeline_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test DelayPipeline class."""
    print("\n" + "=" * 80)
    print("Testing DelayPipeline (V2)")
    print("=" * 80)

    num_envs = 16
    field_dim = 3
    dt = 0.01

    # Test 1: First-order lag filtering
    try:
        cfg = FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.1,  # 100ms
        )
        pipeline = DelayPipeline(cfg, num_envs, field_dim, dt, device)

        # Apply step input
        t_current = torch.zeros(num_envs, device=device)
        step_input = torch.ones(num_envs, field_dim, device=device)

        # First update initializes filter state
        output = pipeline.process(step_input, t_current)
        assert output.shape == (num_envs, field_dim)

        # Run for several steps - output should approach 1.0
        for i in range(100):
            t_current += dt
            output = pipeline.process(step_input, t_current)

        # After 1 second with tau=0.1s, should be ~99.99% of final value
        assert torch.allclose(output, step_input, atol=0.01), f"Expected ~1.0, got {output[0]}"

        results.add_pass("First-order lag filtering")
    except Exception as e:
        results.add_fail("First-order lag filtering", traceback.format_exc())

    # Test 2: Time constant invariance
    try:
        tau = 0.1  # 100ms time constant

        # Create two pipelines with different dt but same tau
        cfg = FieldDelayCfg(first_order_lag_enabled=True, time_constant=tau)

        pipeline_100hz = DelayPipeline(cfg, num_envs, field_dim, dt=0.01, device=device)
        pipeline_200hz = DelayPipeline(cfg, num_envs, field_dim, dt=0.005, device=device)

        step_input = torch.ones(num_envs, field_dim, device=device)

        # Run both for 0.5 seconds
        t = torch.zeros(num_envs, device=device)
        for _ in range(50):  # 50 * 0.01 = 0.5s
            t += 0.01
            out_100hz = pipeline_100hz.process(step_input, t)

        t = torch.zeros(num_envs, device=device)
        for _ in range(100):  # 100 * 0.005 = 0.5s
            t += 0.005
            out_200hz = pipeline_200hz.process(step_input, t)

        # Results should be very similar (same physical response)
        assert torch.allclose(out_100hz, out_200hz, atol=0.05), \
            f"Time constant invariance failed: {out_100hz[0]} vs {out_200hz[0]}"

        if verbose:
            print(f"  100Hz output: {out_100hz[0, 0].item():.4f}")
            print(f"  200Hz output: {out_200hz[0, 0].item():.4f}")

        results.add_pass("Time constant invariance")
    except Exception as e:
        results.add_fail("Time constant invariance", traceback.format_exc())

    # Test 3: Staleness (sample-and-hold)
    try:
        cfg = FieldDelayCfg(
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="constant", value=10.0),  # 10 Hz = 0.1s period
        )
        pipeline = DelayPipeline(cfg, num_envs, field_dim, dt, device)

        t_current = torch.zeros(num_envs, device=device)

        # First sample
        data1 = torch.ones(num_envs, field_dim, device=device)
        out1 = pipeline.process(data1, t_current)

        # Between samples, output should be held constant
        for i in range(5):  # 50ms, less than 100ms period
            t_current += dt
            data_new = torch.full((num_envs, field_dim), float(i + 2), device=device)
            out = pipeline.process(data_new, t_current)
            assert torch.allclose(out, data1), f"Data should be held constant, got {out[0]}"

        # After sample period, should update
        for i in range(10):  # Push past 100ms
            t_current += dt

        data_after = torch.full((num_envs, field_dim), 99.0, device=device)
        out_after = pipeline.process(data_after, t_current)

        # Should have updated to new value
        assert torch.allclose(out_after, data_after), f"Expected update to 99, got {out_after[0]}"

        results.add_pass("Staleness (sample-and-hold)")
    except Exception as e:
        results.add_fail("Staleness (sample-and-hold)", traceback.format_exc())

    # Test 4: Dropout
    try:
        cfg = FieldDelayCfg(
            dropout_enabled=True,
            dropout_prob=0.5,  # 50% dropout
        )
        pipeline = DelayPipeline(cfg, num_envs, field_dim, dt, device)

        t_current = torch.zeros(num_envs, device=device)

        # Run many iterations and count dropouts per environment
        dropout_count = 0
        total_count = 0
        prev_data = None

        for i in range(100):
            t_current += dt
            data = torch.full((num_envs, field_dim), float(i), device=device)
            out = pipeline.process(data, t_current, allow_dropout=True)

            if prev_data is not None:
                # Check per-environment if data was held (dropout occurred)
                # Compare along field dimension - if all fields match, dropout occurred
                per_env_same = (out == prev_data).all(dim=-1)  # shape: (num_envs,)
                dropout_count += per_env_same.sum().item()
                total_count += num_envs

            prev_data = out.clone()

        # Dropout rate should be roughly 50% (with some variance)
        dropout_rate = dropout_count / total_count
        if verbose:
            print(f"  Measured dropout rate: {dropout_rate:.2f} (expected ~0.5)")

        assert 0.2 < dropout_rate < 0.8, f"Dropout rate {dropout_rate} seems wrong"

        results.add_pass("Dropout mechanism")
    except Exception as e:
        results.add_fail("Dropout mechanism", traceback.format_exc())

    # Test 5: Reset functionality
    try:
        cfg = FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.1,
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="constant", value=10.0),
        )
        pipeline = DelayPipeline(cfg, num_envs, field_dim, dt, device)

        t_current = torch.zeros(num_envs, device=device)

        # Process some data
        for _ in range(10):
            t_current += dt
            pipeline.process(torch.ones(num_envs, field_dim, device=device), t_current)

        # Reset first half
        env_ids = torch.arange(num_envs // 2, device=device)
        pipeline.reset(env_ids)

        results.add_pass("Pipeline reset functionality")
    except Exception as e:
        results.add_fail("Pipeline reset functionality", traceback.format_exc())


def run_derived_fields_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test DerivedFieldComputer class."""
    print("\n" + "=" * 80)
    print("Testing DerivedFieldComputer (V2)")
    print("=" * 80)

    num_envs = 16

    # Test 1: Basic derived field computation
    try:
        # Define a simple derived field
        def compute_sum(field_a: torch.Tensor, field_b: torch.Tensor) -> torch.Tensor:
            return field_a + field_b

        field_def = DerivedFieldDef(
            name="sum_field",
            sources=["field_a", "field_b"],
            compute_fn=compute_sum,
            dependency_level=1,
        )

        computer = DerivedFieldComputer([field_def])

        # Compute derived fields
        raw_fields = {
            "field_a": torch.ones(num_envs, 3, device=device),
            "field_b": torch.ones(num_envs, 3, device=device) * 2,
        }

        all_fields = computer.compute_all(raw_fields)

        assert "sum_field" in all_fields
        assert torch.allclose(all_fields["sum_field"], torch.ones(num_envs, 3, device=device) * 3)

        results.add_pass("Basic derived field computation")
    except Exception as e:
        results.add_fail("Basic derived field computation", traceback.format_exc())

    # Test 2: Dependency ordering
    try:
        # Level 1: sum = a + b
        # Level 2: double_sum = sum * 2
        def compute_sum(field_a: torch.Tensor, field_b: torch.Tensor) -> torch.Tensor:
            return field_a + field_b

        def compute_double(sum_field: torch.Tensor) -> torch.Tensor:
            return sum_field * 2

        defs = [
            DerivedFieldDef("sum_field", ["field_a", "field_b"], compute_sum, dependency_level=1),
            DerivedFieldDef("double_sum", ["sum_field"], compute_double, dependency_level=2),
        ]

        computer = DerivedFieldComputer(defs)

        raw_fields = {
            "field_a": torch.ones(num_envs, 3, device=device),
            "field_b": torch.ones(num_envs, 3, device=device) * 2,
        }

        all_fields = computer.compute_all(raw_fields)

        # sum_field = 1 + 2 = 3
        # double_sum = 3 * 2 = 6
        assert torch.allclose(all_fields["sum_field"], torch.full((num_envs, 3), 3.0, device=device))
        assert torch.allclose(all_fields["double_sum"], torch.full((num_envs, 3), 6.0, device=device))

        results.add_pass("Dependency ordering")
    except Exception as e:
        results.add_fail("Dependency ordering", traceback.format_exc())

    # Test 3: Missing sources handled gracefully
    try:
        def compute_sum(field_a: torch.Tensor, field_b: torch.Tensor) -> torch.Tensor:
            return field_a + field_b

        field_def = DerivedFieldDef("sum_field", ["field_a", "field_b"], compute_sum, dependency_level=1)
        computer = DerivedFieldComputer([field_def])

        # Only provide field_a, not field_b
        raw_fields = {"field_a": torch.ones(num_envs, 3, device=device)}

        all_fields = computer.compute_all(raw_fields)

        # sum_field should not be computed (missing source)
        assert "sum_field" not in all_fields

        results.add_pass("Missing sources handled gracefully")
    except Exception as e:
        results.add_fail("Missing sources handled gracefully", traceback.format_exc())

    # Test 4: Velocity magnitude computation
    try:
        def compute_velocity_magnitude(linear_velocity_b: torch.Tensor) -> torch.Tensor:
            return torch.norm(linear_velocity_b, dim=-1, keepdim=True)

        field_def = DerivedFieldDef(
            "velocity_magnitude",
            ["linear_velocity_b"],
            compute_velocity_magnitude,
            dependency_level=1,
        )
        computer = DerivedFieldComputer([field_def])

        # Create velocity vector with known magnitude
        velocity = torch.zeros(num_envs, 3, device=device)
        velocity[:, 0] = 3.0
        velocity[:, 1] = 4.0  # magnitude = 5

        raw_fields = {"linear_velocity_b": velocity}
        all_fields = computer.compute_all(raw_fields)

        expected_magnitude = torch.full((num_envs, 1), 5.0, device=device)
        assert torch.allclose(all_fields["velocity_magnitude"], expected_magnitude)

        results.add_pass("Velocity magnitude computation")
    except Exception as e:
        results.add_fail("Velocity magnitude computation", traceback.format_exc())


def run_delay_system_v2_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test DelaySystemV2 class."""
    print("\n" + "=" * 80)
    print("Testing DelaySystemV2 (V2)")
    print("=" * 80)

    num_envs = 16

    # Test 1: Legacy mode (backward compatibility)
    try:
        cfg = DelaySystemCfgV2(
            use_enhanced_mode=False,
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            action_dim=4,
            observation_dim=12,
        )

        # Should work like original DelaySystem
        action = torch.randn(num_envs, 4, device=device)
        delayed = system.compute_delayed_action(action)
        assert delayed.shape == (num_envs, 4)

        results.add_pass("Legacy mode (backward compatibility)")
    except Exception as e:
        results.add_fail("Legacy mode (backward compatibility)", traceback.format_exc())

    # Test 2: Enhanced mode initialization
    try:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "imu_accel": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.005,
                    noise=NoiseCfg(enabled=True, std=0.1),
                ),
                "gps_position": FieldDelayCfg(
                    staleness_enabled=True,
                    sample_rate=DistributionCfg(type="constant", value=10.0),
                ),
            },
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={"imu_accel": 3, "gps_position": 3},
        )

        assert system.data_bus is not None
        assert len(system._clean_pipelines) == 2
        assert len(system._noisy_pipelines) == 2

        results.add_pass("Enhanced mode initialization")
    except Exception as e:
        results.add_fail("Enhanced mode initialization", traceback.format_exc())

    # Test 3: Store and retrieve clean data
    try:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "test_field": FieldDelayCfg(noise=NoiseCfg(enabled=False)),
            },
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={"test_field": 3},
        )

        # Store data
        data = torch.randn(num_envs, 3, device=device)
        system.store("test_field", data)

        # Retrieve clean data (no noise, no delay effects yet)
        clean = system.get_delayed_clean("test_field")
        assert torch.allclose(clean, data)

        results.add_pass("Store and retrieve clean data")
    except Exception as e:
        results.add_fail("Store and retrieve clean data", traceback.format_exc())

    # Test 4: Noise injection
    try:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "noisy_field": FieldDelayCfg(
                    noise=NoiseCfg(enabled=True, std=1.0),
                ),
            },
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={"noisy_field": 3},
        )

        # Store zero data
        data = torch.zeros(num_envs, 3, device=device)
        system.store("noisy_field", data)

        clean = system.get_delayed_clean("noisy_field")
        noisy = system.get_delayed_noisy("noisy_field")

        # Clean should be zero
        assert torch.allclose(clean, data)

        # Noisy should have noise
        assert not torch.allclose(noisy, data)

        results.add_pass("Noise injection")
    except Exception as e:
        results.add_fail("Noise injection", traceback.format_exc())

    # Test 5: Clean path has no dropout
    try:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "dropout_field": FieldDelayCfg(
                    dropout_enabled=True,
                    dropout_prob=1.0,  # 100% dropout
                ),
            },
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={"dropout_field": 3},
        )

        # Store sequence of values
        for i in range(10):
            data = torch.full((num_envs, 3), float(i), device=device)
            system.store("dropout_field", data)
            clean = system.get_delayed_clean("dropout_field")

            # Clean should always get latest (no dropout)
            assert torch.allclose(clean, data), f"Clean path should not have dropout, got {clean[0]}"

        results.add_pass("Clean path has no dropout")
    except Exception as e:
        results.add_fail("Clean path has no dropout", traceback.format_exc())

    # Test 6: Step and reset
    try:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "test_field": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.1,
                ),
            },
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={"test_field": 3},
        )

        # Step time forward
        for _ in range(10):
            system.step()

        assert torch.allclose(
            system.data_bus.t_current,
            torch.full((num_envs,), 0.1, device=device),
            atol=1e-6,
        )

        # Reset specific envs
        env_ids = torch.arange(5, device=device)
        system.reset(env_ids)

        assert torch.all(system.data_bus.t_current[:5] == 0.0)
        assert torch.all(system.data_bus.t_current[5:] > 0.0)

        results.add_pass("Step and reset")
    except Exception as e:
        results.add_fail("Step and reset", traceback.format_exc())

    # Test 7: Curriculum learning hooks
    try:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "test_field": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.1,
                    dropout_enabled=True,
                    dropout_prob=0.1,
                ),
            },
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={"test_field": 3},
        )

        # Update time constant
        system.set_field_time_constant("test_field", 0.05)
        assert system._clean_pipelines["test_field"].cfg.time_constant == 0.05

        # Update dropout rate
        system.set_field_dropout_rate("test_field", 0.2)
        assert system._noisy_pipelines["test_field"].cfg.dropout_prob == 0.2

        results.add_pass("Curriculum learning hooks")
    except Exception as e:
        results.add_fail("Curriculum learning hooks", traceback.format_exc())

    # Test 8: Derived field computation
    try:
        def compute_velocity_mag(velocity: torch.Tensor) -> torch.Tensor:
            return torch.norm(velocity, dim=-1, keepdim=True)

        derived_def = DerivedFieldDef(
            "velocity_magnitude",
            ["velocity"],
            compute_velocity_mag,
            dependency_level=1,
        )

        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={"velocity": FieldDelayCfg()},
        )
        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={"velocity": 3},
            derived_fields=[derived_def],
        )

        # Store velocity [3, 4, 0] -> magnitude = 5
        velocity = torch.zeros(num_envs, 3, device=device)
        velocity[:, 0] = 3.0
        velocity[:, 1] = 4.0
        system.store("velocity", velocity)

        all_fields = system.compute_derived_fields(use_clean=True)

        assert "velocity_magnitude" in all_fields
        assert torch.allclose(
            all_fields["velocity_magnitude"],
            torch.full((num_envs, 1), 5.0, device=device),
        )

        results.add_pass("Derived field computation")
    except Exception as e:
        results.add_fail("Derived field computation", traceback.format_exc())


def run_v2_integration_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Integration tests for V2 components."""
    print("\n" + "=" * 80)
    print("V2 Integration Tests")
    print("=" * 80)

    num_envs = 64

    # Test 1: Full pipeline with all features
    try:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "imu_accel": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.005,
                    noise=NoiseCfg(enabled=True, std=0.1),
                ),
                "gps_position": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.1,
                    staleness_enabled=True,
                    sample_rate=DistributionCfg(type="constant", value=10.0),
                    noise=NoiseCfg(enabled=True, std=0.5),
                ),
                "camera_bbox": FieldDelayCfg(
                    staleness_enabled=True,
                    sample_rate=DistributionCfg(type="constant", value=30.0),
                    dropout_enabled=True,
                    dropout_prob=0.1,
                ),
            },
        )

        system = DelaySystemV2(
            cfg,
            num_envs=num_envs,
            device=str(device),
            field_dims={
                "imu_accel": 3,
                "gps_position": 3,
                "camera_bbox": 4,
            },
        )

        print("  Running 100-step simulation...", end="", flush=True)

        for step in range(100):
            if step % 25 == 0 and step > 0:
                print(f".{step}", end="", flush=True)

            # Store sensor data
            system.store("imu_accel", torch.randn(num_envs, 3, device=device))
            system.store("gps_position", torch.randn(num_envs, 3, device=device))
            system.store("camera_bbox", torch.randn(num_envs, 4, device=device))

            # Get delayed data for observations
            obs_imu = system.get_delayed_noisy("imu_accel")
            obs_gps = system.get_delayed_noisy("gps_position")
            obs_bbox = system.get_delayed_noisy("camera_bbox")

            # Get clean data for rewards
            reward_gps = system.get_delayed_clean("gps_position")

            # Verify shapes
            assert obs_imu.shape == (num_envs, 3)
            assert obs_gps.shape == (num_envs, 3)
            assert obs_bbox.shape == (num_envs, 4)
            assert reward_gps.shape == (num_envs, 3)

            # Advance time
            system.step()

            # Periodic resets
            if step % 20 == 0:
                reset_ids = torch.randint(0, num_envs, (num_envs // 4,), device=device)
                system.reset(reset_ids)

        print(" done", flush=True)
        results.add_pass("Full pipeline simulation (100 steps)")
    except Exception as e:
        print(" FAILED")
        results.add_fail("Full pipeline simulation (100 steps)", traceback.format_exc())

    # Test 2: Large batch size with V2
    try:
        large_num_envs = 4096

        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "state": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.05,
                    staleness_enabled=True,
                    sample_rate=DistributionCfg(type="constant", value=50.0),
                    dropout_enabled=True,
                    dropout_prob=0.05,
                    noise=NoiseCfg(enabled=True, std=0.1),
                ),
            },
        )

        system = DelaySystemV2(
            cfg,
            num_envs=large_num_envs,
            device=str(device),
            field_dims={"state": 12},
        )

        for _ in range(20):
            system.store("state", torch.randn(large_num_envs, 12, device=device))
            _ = system.get_delayed_noisy("state")
            system.step()

        results.add_pass(f"Large batch size V2 ({large_num_envs} envs)")
    except Exception as e:
        results.add_fail(f"Large batch size V2 ({large_num_envs} envs)", traceback.format_exc())


def main():
    """Main test runner."""
    print("=" * 80)
    print("DELAY SYSTEM TEST SUITE (V1 + V2)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    verbose = args_cli.test_verbose

    results = TestResults()

    try:
        run_delay_cfg_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DelayCfg test suite", traceback.format_exc())

    try:
        run_delay_system_cfg_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DelaySystemCfg test suite", traceback.format_exc())

    try:
        run_delay_system_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DelaySystem test suite", traceback.format_exc())

    try:
        run_integration_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Integration test suite", traceback.format_exc())

    # V2 Tests
    try:
        run_data_bus_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DataBus test suite", traceback.format_exc())

    try:
        run_delay_pipeline_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DelayPipeline test suite", traceback.format_exc())

    try:
        run_derived_fields_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DerivedFieldComputer test suite", traceback.format_exc())

    try:
        run_delay_system_v2_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DelaySystemV2 test suite", traceback.format_exc())

    try:
        run_v2_integration_tests(results, device, verbose)
    except Exception as e:
        results.add_error("V2 Integration test suite", traceback.format_exc())

    success = results.print_summary()

    # Cleanup
    simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
