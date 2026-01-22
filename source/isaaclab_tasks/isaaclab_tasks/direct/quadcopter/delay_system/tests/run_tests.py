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
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        if total > 0:
            print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)")
            print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)")
            print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)")

        if self.failed:
            print("\n" + "-" * 80)
            print("FAILED TESTS:")
            print("-" * 80)
            for test_name, error in self.failed:
                print(f"\n{test_name}:")
                print(f"  {error}")

        if self.errors:
            print("\n" + "-" * 80)
            print("TEST ERRORS:")
            print("-" * 80)
            for test_name, error in self.errors:
                print(f"\n{test_name}:")
                print(f"  {error}")

        print("=" * 80)
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


def main():
    """Main test runner."""
    print("=" * 80)
    print("DELAY SYSTEM TEST SUITE")
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

    success = results.print_summary()

    # Cleanup
    simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
