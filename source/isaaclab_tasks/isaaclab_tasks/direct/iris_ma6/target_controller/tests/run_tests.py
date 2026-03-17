#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone test runner for target controller module.

This script tests the target controller without requiring the full Isaac Sim
environment, focusing on the velocity generation and FSM logic.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/target_controller/tests/run_tests.py
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run target controller test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules
import sys
import traceback
from datetime import datetime

import torch


class TestResults:
    """Track test results with pass/fail status."""

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

        if self.errors:
            print("\n" + "-" * 80)
            print("TEST ERRORS:")
            print("-" * 80)
            for test_name, error in self.errors:
                print(f"\n{test_name}:")
                print(f"  {error}")

        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def run_config_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test configuration dataclass."""
    print("\n" + "=" * 80)
    print("Testing Configuration")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.target_controller import (
        TargetControllerCfg,
        BehaviorProfile,
        BEHAVIOR_PROFILES,
    )

    # Test 1: Configuration instantiation
    try:
        cfg = TargetControllerCfg()
        assert cfg.max_speed_start == 3.0
        assert cfg.max_speed_end == 12.0
        results.add_pass("Configuration instantiation")
    except Exception as e:
        results.add_fail("Configuration instantiation", str(e))

    # Test 2: Behavior profiles
    try:
        assert "kamikaze" in BEHAVIOR_PROFILES
        assert "standard" in BEHAVIOR_PROFILES
        assert "evasive" in BEHAVIOR_PROFILES
        assert "stealth" in BEHAVIOR_PROFILES

        kamikaze = BEHAVIOR_PROFILES["kamikaze"]
        assert kamikaze.speed_multiplier == 1.5
        assert kamikaze.evasion_agility == 0.0
        results.add_pass("Behavior profiles defined")
    except Exception as e:
        results.add_fail("Behavior profiles defined", str(e))


def run_fsm_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test behavior FSM."""
    print("\n" + "=" * 80)
    print("Testing Behavior FSM")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.target_controller import (
        TargetControllerCfg,
        BehaviorFSM,
        FSMState,
        VelocityMode,
    )

    cfg = TargetControllerCfg()
    num_envs = 16
    num_targets = 4

    # Test 1: FSM initialization
    try:
        fsm = BehaviorFSM(cfg, num_envs, num_targets, device)
        assert fsm.fsm_state.shape == (num_envs, num_targets)
        assert fsm.alive.shape == (num_envs, num_targets)
        results.add_pass("FSM initialization")
    except Exception as e:
        results.add_fail("FSM initialization", str(e))

    # Test 2: FSM reset
    try:
        fsm = BehaviorFSM(cfg, num_envs, num_targets, device)
        env_ids = torch.arange(num_envs, device=device)
        fsm.reset(env_ids)

        # After reset, all should be alive and in APPROACH state
        assert fsm.alive.all()
        assert (fsm.fsm_state == FSMState.APPROACH).all()
        results.add_pass("FSM reset")
    except Exception as e:
        results.add_fail("FSM reset", str(e))

    # Test 3: Get alive indices
    try:
        fsm = BehaviorFSM(cfg, num_envs, num_targets, device)
        fsm.reset(torch.arange(num_envs, device=device))

        alive_indices = fsm.get_alive_indices()
        assert len(alive_indices) == num_envs * num_targets
        results.add_pass("Get alive indices")
    except Exception as e:
        results.add_fail("Get alive indices", str(e))

    # Test 4: Get state indices
    try:
        fsm = BehaviorFSM(cfg, num_envs, num_targets, device)
        fsm.reset(torch.arange(num_envs, device=device))

        approach_indices = fsm.get_state_indices(FSMState.APPROACH)
        assert len(approach_indices) == num_envs * num_targets

        evade_indices = fsm.get_state_indices(FSMState.EVADE)
        assert len(evade_indices) == 0
        results.add_pass("Get state indices")
    except Exception as e:
        results.add_fail("Get state indices", str(e))


def run_velocity_generator_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test velocity generators."""
    print("\n" + "=" * 80)
    print("Testing Velocity Generators")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.target_controller import TargetControllerCfg
    from isaaclab_tasks.direct.iris_ma6.target_controller.velocity_generators import (
        LinearModeGenerator,
        CircularModeGenerator,
        ApproachModeGenerator,
        EvadeModeGenerator,
    )

    cfg = TargetControllerCfg()
    num_envs = 16
    num_targets = 4

    # Test 1: Linear mode generator
    try:
        generator = LinearModeGenerator(cfg, num_envs, num_targets, device)
        generator.reset(torch.arange(num_envs, device=device))

        indices = torch.arange(num_envs * num_targets, device=device)
        pos = torch.randn(num_envs * num_targets, 3, device=device) * 50
        v_cmd = generator.compute(indices, pos, curriculum_progress=0.5, dt=0.02)

        assert v_cmd.shape == (num_envs * num_targets, 3)
        assert not torch.isnan(v_cmd).any()
        results.add_pass("Linear mode generator")
    except Exception as e:
        results.add_fail("Linear mode generator", traceback.format_exc())

    # Test 2: Circular mode generator
    try:
        generator = CircularModeGenerator(cfg, num_envs, num_targets, device)
        generator.reset(torch.arange(num_envs, device=device))

        # Set centers
        centers = torch.zeros(num_envs * num_targets, 3, device=device)
        generator.set_center(torch.arange(num_envs * num_targets, device=device), centers)

        indices = torch.arange(num_envs * num_targets, device=device)
        pos = torch.randn(num_envs * num_targets, 3, device=device) * 50
        pos[:, 2] = torch.abs(pos[:, 2]) + 10  # Ensure positive altitude
        v_cmd = generator.compute(indices, pos, curriculum_progress=0.5, dt=0.02)

        assert v_cmd.shape == (num_envs * num_targets, 3)
        assert not torch.isnan(v_cmd).any()
        results.add_pass("Circular mode generator")
    except Exception as e:
        results.add_fail("Circular mode generator", traceback.format_exc())

    # Test 3: Approach mode generator
    try:
        generator = ApproachModeGenerator(cfg, num_envs, num_targets, device)
        generator.reset(torch.arange(num_envs, device=device))

        indices = torch.arange(num_envs * num_targets, device=device)
        pos = torch.randn(num_envs * num_targets, 3, device=device) * 100
        pos[:, 2] = torch.abs(pos[:, 2]) + 20
        facility = torch.zeros(num_envs * num_targets, 3, device=device)

        v_cmd = generator.compute(
            indices, pos, curriculum_progress=0.5, dt=0.02, facility_position=facility
        )

        assert v_cmd.shape == (num_envs * num_targets, 3)
        assert not torch.isnan(v_cmd).any()
        results.add_pass("Approach mode generator")
    except Exception as e:
        results.add_fail("Approach mode generator", traceback.format_exc())

    # Test 4: Evade mode generator
    try:
        generator = EvadeModeGenerator(cfg, num_envs, num_targets, device)
        generator.reset(torch.arange(num_envs, device=device))

        # Set evasion agility
        generator.set_evasion_agility(
            torch.arange(num_envs * num_targets, device=device),
            torch.ones(num_envs * num_targets, device=device) * 0.5,
        )

        indices = torch.arange(num_envs * num_targets, device=device)
        pos = torch.randn(num_envs * num_targets, 3, device=device) * 50
        pos[:, 2] = torch.abs(pos[:, 2]) + 20

        # Mock interceptor data
        num_defenders = 6
        interceptor_pos = torch.randn(num_envs, num_defenders, 3, device=device) * 100
        interceptor_roles = torch.ones(num_envs, num_defenders, dtype=torch.long, device=device)

        v_cmd = generator.compute(
            indices,
            pos,
            curriculum_progress=0.5,
            dt=0.02,
            interceptor_positions=interceptor_pos,
            interceptor_roles=interceptor_roles,
        )

        assert v_cmd.shape == (num_envs * num_targets, 3)
        assert not torch.isnan(v_cmd).any()
        results.add_pass("Evade mode generator")
    except Exception as e:
        results.add_fail("Evade mode generator", traceback.format_exc())

    # Test 5: Linear mode direction changes
    try:
        generator = LinearModeGenerator(cfg, num_envs, num_targets, device)
        generator.reset(torch.arange(num_envs, device=device))

        indices = torch.arange(num_envs * num_targets, device=device)
        pos = torch.randn(num_envs * num_targets, 3, device=device) * 50

        # Get initial velocity
        v1 = generator.compute(indices, pos, curriculum_progress=0.5, dt=0.02).clone()

        # Simulate time passing to trigger update
        generator.update_timer[:] = 100.0  # Force update

        # Get new velocity after update
        v2 = generator.compute(indices, pos, curriculum_progress=0.5, dt=0.02)

        # Direction should have changed
        # (Not all will change due to randomness, but timer should reset)
        assert generator.update_timer.max() < 1.0  # Timer was reset
        results.add_pass("Linear mode direction changes")
    except Exception as e:
        results.add_fail("Linear mode direction changes", traceback.format_exc())


def run_curriculum_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test curriculum scaling."""
    print("\n" + "=" * 80)
    print("Testing Curriculum Scaling")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.target_controller import TargetControllerCfg
    from isaaclab_tasks.direct.iris_ma6.target_controller.velocity_generators import (
        LinearModeGenerator,
    )

    cfg = TargetControllerCfg()
    num_envs = 16
    num_targets = 4

    # Test 1: Max speed scaling
    try:
        generator = LinearModeGenerator(cfg, num_envs, num_targets, device)

        speed_at_0 = generator._get_max_speed(0.0)
        speed_at_1 = generator._get_max_speed(1.0)
        speed_at_half = generator._get_max_speed(0.5)

        assert speed_at_0 == cfg.max_speed_start
        assert speed_at_1 == cfg.max_speed_end
        assert speed_at_0 < speed_at_half < speed_at_1
        results.add_pass("Max speed scaling")
    except Exception as e:
        results.add_fail("Max speed scaling", str(e))

    # Test 2: Update interval scaling
    try:
        generator = LinearModeGenerator(cfg, num_envs, num_targets, device)

        # Sample intervals at different progress levels
        intervals_0 = generator._sample_update_interval(100, 0.0)
        intervals_1 = generator._sample_update_interval(100, 1.0)

        # At progress 0, intervals should be longer (easier)
        # At progress 1, intervals should be shorter (harder)
        mean_0 = intervals_0.mean().item()
        mean_1 = intervals_1.mean().item()

        assert mean_0 > mean_1
        results.add_pass("Update interval scaling")
    except Exception as e:
        results.add_fail("Update interval scaling", str(e))

    # Test 3: Geofence scaling
    try:
        generator = LinearModeGenerator(cfg, num_envs, num_targets, device)

        geofence_0 = generator._get_geofence_size(0.0)
        geofence_1 = generator._get_geofence_size(1.0)

        assert geofence_0 == cfg.geofence_min_size
        assert geofence_1 == cfg.geofence_max_size
        assert geofence_0 < geofence_1
        results.add_pass("Geofence scaling")
    except Exception as e:
        results.add_fail("Geofence scaling", str(e))


def main():
    """Main test runner."""
    print("=" * 80)
    print("TARGET CONTROLLER TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()
    verbose = args_cli.test_verbose

    try:
        run_config_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Configuration test suite", traceback.format_exc())

    try:
        run_fsm_tests(results, device, verbose)
    except Exception as e:
        results.add_error("FSM test suite", traceback.format_exc())

    try:
        run_velocity_generator_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Velocity generator test suite", traceback.format_exc())

    try:
        run_curriculum_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Curriculum test suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
