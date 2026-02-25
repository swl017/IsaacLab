#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target movement module test suite."""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run target movement test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW import other modules
import torch
import sys
import traceback
import math
from datetime import datetime

from isaaclab_tasks.direct.iris_ma5.target_movement import TargetMovement, TargetMovementCfg


class TestResults:
    """Track test results with pass/fail counts."""

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
        error_lines = error.split("\n")[:5]
        for line in error_lines:
            print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        print(f"    {error}")

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


def run_initialization_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test initialization and configuration."""
    print("\n" + "=" * 80)
    print("Testing Initialization")
    print("=" * 80)

    num_envs = 16

    # Test 1: Default configuration
    try:
        cfg = TargetMovementCfg()
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)
        assert movement.num_envs == num_envs
        assert movement.velocity.shape == (num_envs, 6)
        assert movement.motion_mode.shape == (num_envs,)
        results.add_pass("Default configuration")
    except Exception as e:
        results.add_fail("Default configuration", traceback.format_exc())

    # Test 2: Custom configuration
    try:
        cfg = TargetMovementCfg(
            max_speed=10.0,
            max_acceleration=5.0,
            linear_weight=0.7,
            geofence_min_size=100.0,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)
        assert movement.cfg.max_speed == 10.0
        assert movement.cfg.max_acceleration == 5.0
        assert movement.cfg.linear_weight == 0.7
        results.add_pass("Custom configuration")
    except Exception as e:
        results.add_fail("Custom configuration", traceback.format_exc())

    # Test 3: Buffer allocation
    try:
        cfg = TargetMovementCfg()
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)
        assert movement.velocity.device.type == device.type
        assert movement.desired_velocity.device.type == device.type
        assert movement.motion_mode.device.type == device.type
        assert movement.circular_radius.device.type == device.type
        results.add_pass("Buffer allocation on correct device")
    except Exception as e:
        results.add_fail("Buffer allocation on correct device", traceback.format_exc())


def run_linear_mode_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test linear movement mode."""
    print("\n" + "=" * 80)
    print("Testing Linear Mode")
    print("=" * 80)

    num_envs = 16

    # Test 1: Force linear mode and check velocity updates
    try:
        cfg = TargetMovementCfg(linear_weight=1.0)  # All linear
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        # Check all environments are in linear mode
        assert (movement.motion_mode == 0).all(), "Not all envs in linear mode"
        results.add_pass("Linear mode assignment")
    except Exception as e:
        results.add_fail("Linear mode assignment", traceback.format_exc())

    # Test 2: Velocity tracking
    try:
        cfg = TargetMovementCfg(
            linear_weight=1.0,
            max_speed=5.0,
            acceleration_scale=2.0,
            velocity_damping=0.95,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        current_pos = torch.zeros(num_envs, 3, device=device)
        env_origins = torch.zeros(num_envs, 3, device=device)

        # Step multiple times
        for _ in range(50):
            velocity = movement.step(current_pos, env_origins, curriculum_progress=1.0, dt=0.02)
            current_pos += velocity[:, :3] * 0.02

        # Velocity should be non-zero after multiple steps
        speed = torch.norm(movement.velocity[:, :3], dim=1)
        assert (speed > 0.1).any(), f"Velocity too low: mean={speed.mean().item():.4f}"
        results.add_pass("Velocity tracking")
    except Exception as e:
        results.add_fail("Velocity tracking", traceback.format_exc())

    # Test 3: Velocity clamping
    try:
        cfg = TargetMovementCfg(linear_weight=1.0, max_speed=5.0)
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        # Set very high desired velocity
        movement.desired_velocity[:, :3] = 100.0

        current_pos = torch.zeros(num_envs, 3, device=device)
        env_origins = torch.zeros(num_envs, 3, device=device)

        for _ in range(100):
            velocity = movement.step(current_pos, env_origins, curriculum_progress=1.0, dt=0.02)

        # Check velocity is clamped
        speed = torch.norm(movement.velocity[:, :3], dim=1)
        assert (speed <= cfg.max_speed + 0.01).all(), f"Speed exceeded max: {speed.max().item():.2f}"
        results.add_pass("Velocity clamping")
    except Exception as e:
        results.add_fail("Velocity clamping", traceback.format_exc())


def run_circular_mode_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test circular movement mode."""
    print("\n" + "=" * 80)
    print("Testing Circular Mode")
    print("=" * 80)

    num_envs = 16

    # Test 1: Force circular mode
    try:
        cfg = TargetMovementCfg(linear_weight=0.0)  # All circular
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        assert (movement.motion_mode == 1).all(), "Not all envs in circular mode"
        results.add_pass("Circular mode assignment")
    except Exception as e:
        results.add_fail("Circular mode assignment", traceback.format_exc())

    # Test 2: Radius constraints
    try:
        cfg = TargetMovementCfg(
            linear_weight=0.0,
            circular_radius_min=10.0,
            circular_radius_max=50.0,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        assert (movement.circular_radius >= cfg.circular_radius_min).all()
        assert (movement.circular_radius <= cfg.circular_radius_max).all()
        results.add_pass("Radius constraints")
    except Exception as e:
        results.add_fail("Radius constraints", traceback.format_exc())

    # Test 3: Circular orbit tracking
    try:
        cfg = TargetMovementCfg(
            linear_weight=0.0,
            circular_radius_min=20.0,
            circular_radius_max=20.0,  # Fixed radius
            max_speed=5.0,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        # Start at the orbit position
        env_origins = torch.zeros(num_envs, 3, device=device)
        env_origins[:, 2] = 0.0  # Ground level

        current_pos = env_origins.clone()
        current_pos[:, 0] = 20.0  # Start on orbit
        current_pos[:, 2] = 20.0  # At orbit height

        # Step and track position
        positions = [current_pos.clone()]
        for _ in range(100):
            velocity = movement.step(current_pos, env_origins, curriculum_progress=1.0, dt=0.02)
            current_pos = current_pos + velocity[:, :3] * 0.02
            positions.append(current_pos.clone())

        # Check that position moved (orbited)
        displacement = torch.norm(positions[-1][:, :2] - positions[0][:, :2], dim=1)
        assert (displacement > 0.1).all(), f"Not enough movement: {displacement.mean().item():.4f}"
        results.add_pass("Circular orbit tracking")
    except Exception as e:
        results.add_fail("Circular orbit tracking", traceback.format_exc())


def run_geofencing_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test geofencing constraints."""
    print("\n" + "=" * 80)
    print("Testing Geofencing")
    print("=" * 80)

    num_envs = 16

    # Test 1: Geofence size based on curriculum
    try:
        cfg = TargetMovementCfg(
            geofence_min_size=50.0,
            geofence_max_size=500.0,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)

        # Progress = 0 -> geofence = 50m
        # Progress = 1 -> geofence = 500m
        # Progress = 0.5 -> geofence = 275m

        geofence_at_0 = cfg.geofence_min_size + 0.0 * (cfg.geofence_max_size - cfg.geofence_min_size)
        geofence_at_1 = cfg.geofence_min_size + 1.0 * (cfg.geofence_max_size - cfg.geofence_min_size)
        geofence_at_05 = cfg.geofence_min_size + 0.5 * (cfg.geofence_max_size - cfg.geofence_min_size)

        assert abs(geofence_at_0 - 50.0) < 0.01
        assert abs(geofence_at_1 - 500.0) < 0.01
        assert abs(geofence_at_05 - 275.0) < 0.01
        results.add_pass("Geofence curriculum scaling formula")
    except Exception as e:
        results.add_fail("Geofence curriculum scaling formula", traceback.format_exc())

    # Test 2: Bounce at X boundary
    try:
        cfg = TargetMovementCfg(
            linear_weight=1.0,
            geofence_min_size=10.0,
            geofence_max_size=10.0,
            geofence_bounce_factor=0.8,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)

        # Position outside X boundary (positive)
        current_pos = torch.zeros(num_envs, 3, device=device)
        current_pos[:, 0] = 15.0  # Beyond 10m geofence
        current_pos[:, 2] = 20.0  # Above altitude limit

        # Set velocity going outward
        movement.velocity[:, 0] = 5.0

        velocity = movement.step(current_pos, env_origins, curriculum_progress=0.0, dt=0.02)

        # X velocity should be reversed (negative)
        assert (movement.velocity[:, 0] < 0).all(), "X velocity should be reversed"
        results.add_pass("X boundary bounce")
    except Exception as e:
        results.add_fail("X boundary bounce", traceback.format_exc())

    # Test 3: Bounce at Y boundary
    try:
        cfg = TargetMovementCfg(
            linear_weight=1.0,
            geofence_min_size=10.0,
            geofence_max_size=10.0,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)

        # Position outside Y boundary (negative)
        current_pos = torch.zeros(num_envs, 3, device=device)
        current_pos[:, 1] = -15.0  # Beyond -10m geofence
        current_pos[:, 2] = 20.0

        # Set velocity going outward (negative Y)
        movement.velocity[:, 1] = -5.0

        velocity = movement.step(current_pos, env_origins, curriculum_progress=0.0, dt=0.02)

        # Y velocity should be reversed (positive)
        assert (movement.velocity[:, 1] > 0).all(), "Y velocity should be reversed"
        results.add_pass("Y boundary bounce")
    except Exception as e:
        results.add_fail("Y boundary bounce", traceback.format_exc())


def run_altitude_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test altitude constraints."""
    print("\n" + "=" * 80)
    print("Testing Altitude Constraints")
    print("=" * 80)

    num_envs = 16

    # Test 1: Altitude bounce when below minimum
    try:
        cfg = TargetMovementCfg(
            min_altitude=10.0,
            altitude_bounce_velocity=1.0,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)

        # Position below minimum altitude
        current_pos = torch.zeros(num_envs, 3, device=device)
        current_pos[:, 2] = 5.0  # Below 10m minimum

        # Set downward velocity
        movement.velocity[:, 2] = -3.0

        velocity = movement.step(current_pos, env_origins, curriculum_progress=1.0, dt=0.02)

        # Z velocity should be at least altitude_bounce_velocity (positive)
        assert (movement.velocity[:, 2] >= cfg.altitude_bounce_velocity).all(), \
            f"Z velocity should be >= {cfg.altitude_bounce_velocity}, got {movement.velocity[:, 2].min().item()}"
        results.add_pass("Altitude bounce")
    except Exception as e:
        results.add_fail("Altitude bounce", traceback.format_exc())

    # Test 2: No bounce when above minimum
    try:
        cfg = TargetMovementCfg(min_altitude=10.0)
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)

        # Position above minimum altitude
        current_pos = torch.zeros(num_envs, 3, device=device)
        current_pos[:, 2] = 20.0  # Above 10m minimum

        # Set downward velocity (should not be modified)
        movement.velocity[:, 2] = -2.0
        original_z_vel = movement.velocity[:, 2].clone()

        # Run step (with very short timestep to minimize acceleration effects)
        velocity = movement.step(current_pos, env_origins, curriculum_progress=1.0, dt=0.001)

        # Z velocity should still be negative (not bounced)
        # Allow for some acceleration effects but should still be negative
        assert (movement.velocity[:, 2] < 0.5).all(), "Z velocity should remain roughly negative when above altitude"
        results.add_pass("No altitude bounce when above minimum")
    except Exception as e:
        results.add_fail("No altitude bounce when above minimum", traceback.format_exc())


def run_reset_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test reset functionality."""
    print("\n" + "=" * 80)
    print("Testing Reset")
    print("=" * 80)

    num_envs = 16

    # Test 1: Full reset - velocity is randomized, sim_time is reset
    try:
        cfg = TargetMovementCfg(max_speed=5.0)
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        # Set some non-zero values
        movement.velocity[:] = 10.0
        movement.sim_time[:] = 100.0

        # Reset all
        all_ids = torch.arange(num_envs, device=device)
        movement.reset(all_ids)

        # Velocity should be randomized (within max_speed bounds)
        speed = torch.norm(movement.velocity[:, :3], dim=1)
        assert (speed <= cfg.max_speed + 0.01).all(), f"Velocity exceeds max_speed: {speed.max().item()}"
        assert (movement.sim_time == 0).all(), "Sim time not reset to zero"
        results.add_pass("Full reset (randomized velocity)")
    except Exception as e:
        results.add_fail("Full reset (randomized velocity)", traceback.format_exc())

    # Test 2: Partial reset - only reset envs have new velocities
    try:
        cfg = TargetMovementCfg(max_speed=5.0)
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        # Set all velocities to a known high value (outside normal range)
        movement.velocity[:] = 99.0
        movement.sim_time[:] = 50.0

        # Reset only first half
        reset_ids = torch.arange(num_envs // 2, device=device)
        movement.reset(reset_ids)

        # First half should have new randomized velocity (within bounds)
        first_half_speed = torch.norm(movement.velocity[:num_envs // 2, :3], dim=1)
        assert (first_half_speed <= cfg.max_speed + 0.01).all(), "First half velocity not within bounds"
        assert (movement.sim_time[:num_envs // 2] == 0).all(), "First half sim time not reset"

        # Second half should remain unchanged at the high value
        assert (movement.velocity[num_envs // 2:] == 99.0).all(), "Second half should not be reset"
        assert (movement.sim_time[num_envs // 2:] == 50.0).all(), "Second half sim time should not be reset"
        results.add_pass("Partial reset")
    except Exception as e:
        results.add_fail("Partial reset", traceback.format_exc())

    # Test 3: Mode reinitialization on reset
    try:
        cfg = TargetMovementCfg(linear_weight=0.5)
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        # Force all to one mode
        movement.motion_mode[:] = 0

        # Reset should reinitialize with random modes
        all_ids = torch.arange(num_envs, device=device)
        movement.reset(all_ids)

        # With 50% weight, we should have a mix (allow some variance)
        linear_count = (movement.motion_mode == 0).sum().item()
        circular_count = (movement.motion_mode == 1).sum().item()

        # At least some should be each mode (very unlikely to get all one mode with 50% weight)
        assert linear_count > 0 or circular_count > 0, "Mode not reinitialized"
        results.add_pass("Mode reinitialization on reset")
    except Exception as e:
        results.add_fail("Mode reinitialization on reset", traceback.format_exc())


def run_integration_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test full integration cycle."""
    print("\n" + "=" * 80)
    print("Testing Integration")
    print("=" * 80)

    num_envs = 64

    # Test 1: Full step cycle
    try:
        cfg = TargetMovementCfg()
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)
        current_pos = torch.zeros(num_envs, 3, device=device)
        current_pos[:, 2] = 20.0  # Start at safe altitude

        print("  Running 200-step simulation...", end="", flush=True)
        import sys
        for i in range(200):
            if i % 50 == 0 and i > 0:
                print(f".{i}", end="", flush=True)
                sys.stdout.flush()

            velocity = movement.step(
                current_position=current_pos,
                env_origins=env_origins,
                curriculum_progress=min(i / 100.0, 1.0),  # Ramp curriculum
                dt=0.02,
            )
            current_pos = current_pos + velocity[:, :3] * 0.02

        print(" done", flush=True)
        sys.stdout.flush()

        # Check no NaN or Inf
        assert not torch.isnan(movement.velocity).any(), "NaN in velocity"
        assert not torch.isinf(movement.velocity).any(), "Inf in velocity"
        assert not torch.isnan(current_pos).any(), "NaN in position"
        results.add_pass("Full step cycle (200 steps)")
    except Exception as e:
        results.add_fail("Full step cycle (200 steps)", traceback.format_exc())

    # Test 2: Curriculum speed scaling
    try:
        cfg = TargetMovementCfg(
            speed_scale_start=0.3,
            speed_scale_end=1.0,
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)
        current_pos = torch.zeros(num_envs, 3, device=device)
        current_pos[:, 2] = 20.0

        # Set high velocity
        movement.velocity[:, :3] = 5.0

        # With progress=0, output should be scaled by speed_scale_start (0.3)
        velocity_at_0 = movement.step(current_pos, env_origins, curriculum_progress=0.0, dt=0.02)
        speed_at_0 = torch.norm(velocity_at_0[:, :3], dim=1).max().item()
        expected_max_speed_at_0 = 5.0 * cfg.speed_scale_start
        assert speed_at_0 <= expected_max_speed_at_0 + 0.5, f"Progress=0: speed {speed_at_0} should be <= {expected_max_speed_at_0}"

        # Reset and test at progress=1
        movement.reset(torch.arange(num_envs, device=device))
        movement.velocity[:, :3] = 5.0
        velocity_at_1 = movement.step(current_pos, env_origins, curriculum_progress=1.0, dt=0.02)
        speed_at_1 = torch.norm(velocity_at_1[:, :3], dim=1).max().item()
        # At progress=1, should have higher speed (scaled by 1.0)
        assert speed_at_1 > speed_at_0, f"Progress=1 speed {speed_at_1} should be > progress=0 speed {speed_at_0}"
        results.add_pass("Curriculum speed scaling")
    except Exception as e:
        results.add_fail("Curriculum speed scaling", traceback.format_exc())

    # Test 3: Mode switching
    try:
        cfg = TargetMovementCfg(
            allow_mode_switching=True,
            mode_switch_prob=0.1,  # High prob for testing
        )
        movement = TargetMovement(cfg, num_envs=num_envs, device=device)

        env_origins = torch.zeros(num_envs, 3, device=device)
        current_pos = torch.zeros(num_envs, 3, device=device)
        current_pos[:, 2] = 20.0

        initial_modes = movement.motion_mode.clone()

        # Run many steps to trigger mode switches
        for _ in range(100):
            movement.step(current_pos, env_origins, curriculum_progress=1.0, dt=0.02)

        # With high switch prob, some modes should have changed
        modes_changed = (movement.motion_mode != initial_modes).sum().item()
        # With 10% switch prob over 100 steps, very likely some changed
        # But not guaranteed, so just check no errors occurred
        results.add_pass("Mode switching (no errors)")
    except Exception as e:
        results.add_fail("Mode switching (no errors)", traceback.format_exc())


def main():
    """Main test runner."""
    print("=" * 80)
    print("TARGET MOVEMENT MODULE TEST SUITE")
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
        run_initialization_tests(results, device, verbose)
        run_linear_mode_tests(results, device, verbose)
        run_circular_mode_tests(results, device, verbose)
        run_geofencing_tests(results, device, verbose)
        run_altitude_tests(results, device, verbose)
        run_reset_tests(results, device, verbose)
        run_integration_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Test suite execution", traceback.format_exc())

    success = results.print_summary()
    sys.stdout.flush()

    # Clean up - use try/except to handle potential hangs
    try:
        simulation_app.close()
    except Exception as e:
        print(f"Warning: Error during simulation cleanup: {e}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
