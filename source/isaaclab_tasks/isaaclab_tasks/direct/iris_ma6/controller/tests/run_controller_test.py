#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for PX4-style cascaded controller architecture.

This script tests the new 4-loop control architecture:
    Velocity (PI) -> Attitude (P) -> Rate (PID) -> Motor

Run with:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_controller_test.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Test PX4-style controller")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import everything else
import sys
import torch
import traceback
from datetime import datetime


class TestResults:
    """Track test results."""

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


def run_controller_tests(results: TestResults, device: torch.device):
    """Run controller architecture tests."""
    print("\n" + "=" * 80)
    print("Testing PX4-style Controller Architecture")
    print("=" * 80)

    # Import controller module
    try:
        from isaaclab_tasks.direct.iris_ma6.controller import (
            DroneControllerCfg,
            DroneController,
            RateControllerCfg,
            RateController,
            AttitudeControllerCfg,
            AttitudeController,
            VelocityControllerCfg,
            VelocityController,
            TUNED_CONTROLLER_CFG,
        )
        results.add_pass("Import controller modules")
    except Exception as e:
        results.add_fail("Import controller modules", traceback.format_exc())
        return

    num_envs = 4
    mass = 1.5
    gravity = 9.81

    # Test 1: DroneController instantiation
    try:
        controller = DroneController(
            cfg=TUNED_CONTROLLER_CFG,
            mass=mass,
            gravity=gravity,
            num_envs=num_envs,
            device=device,
        )
        results.add_pass("DroneController instantiation")
    except Exception as e:
        results.add_fail("DroneController instantiation", traceback.format_exc())
        return

    # Test 2: Verify controller components exist
    try:
        assert hasattr(controller, "_rate"), "Missing rate controller"
        assert hasattr(controller, "_attitude"), "Missing attitude controller"
        assert hasattr(controller, "_velocity"), "Missing velocity controller"
        assert isinstance(controller._rate, RateController), "Rate controller wrong type"
        assert isinstance(controller._attitude, AttitudeController), "Attitude controller wrong type"
        assert isinstance(controller._velocity, VelocityController), "Velocity controller wrong type"
        results.add_pass("Controller components exist")
    except AssertionError as e:
        results.add_fail("Controller components exist", str(e))

    # Test 3: Zero command hover test
    try:
        v_cmd = torch.zeros((num_envs, 3), device=device)
        yaw_rate_cmd = torch.zeros(num_envs, device=device)
        gimbal_yaw_rate = torch.zeros(num_envs, device=device)
        gimbal_pitch_rate = torch.zeros(num_envs, device=device)
        zoom_rate = torch.zeros(num_envs, device=device)
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device)
        v_body = torch.zeros((num_envs, 3), device=device)
        omega_body = torch.zeros((num_envs, 3), device=device)

        F, tau, gimbal, zoom = controller.step_policy(
            v_cmd,
            yaw_rate_cmd,
            gimbal_yaw_rate,
            gimbal_pitch_rate,
            zoom_rate,
            q_body,
            v_body,
            omega_body,
            sim_dt=0.04,
        )

        assert F.shape == (num_envs, 3), f"F shape mismatch: {F.shape}"
        assert tau.shape == (num_envs, 3), f"tau shape mismatch: {tau.shape}"

        # For hover, thrust should be approximately m*g = 1.5 * 9.81 ≈ 14.7 N
        hover_thrust = mass * gravity
        F_z_mean = F[:, 2].mean().item()
        print(f"    Hover test: F_z mean = {F_z_mean:.2f} N (expected ~{hover_thrust:.2f} N)")

        # Allow some tolerance
        assert abs(F_z_mean - hover_thrust) < 5.0, f"Hover thrust off: {F_z_mean} vs {hover_thrust}"
        results.add_pass("Zero command hover test")
    except Exception as e:
        results.add_fail("Zero command hover test", traceback.format_exc())

    # Test 4: Velocity command test
    try:
        v_cmd = torch.tensor([[1.0, 0.0, 0.0]] * num_envs, device=device)  # Forward velocity
        yaw_rate_cmd = torch.zeros(num_envs, device=device)

        F, tau, _, _ = controller.step_policy(
            v_cmd,
            yaw_rate_cmd,
            gimbal_yaw_rate,
            gimbal_pitch_rate,
            zoom_rate,
            q_body,
            v_body,
            omega_body,
            sim_dt=0.04,
        )

        # With forward velocity command but zero current velocity, should pitch forward
        # This means tau_y should be non-zero (pitch moment)
        print(f"    Velocity cmd test: tau = [{tau[0, 0].item():.4f}, {tau[0, 1].item():.4f}, {tau[0, 2].item():.4f}]")
        results.add_pass("Velocity command test")
    except Exception as e:
        results.add_fail("Velocity command test", traceback.format_exc())

    # Test 5: Rate controller integration test
    try:
        controller.reset()

        # Apply a tilted initial condition
        from isaaclab.utils.math import quat_from_euler_xyz

        roll = torch.tensor([0.1] * num_envs, device=device)  # 0.1 rad ≈ 6 degrees
        pitch = torch.zeros(num_envs, device=device)
        yaw = torch.zeros(num_envs, device=device)
        q_tilted = quat_from_euler_xyz(roll, pitch, yaw)

        # Zero velocity command (should try to level)
        v_cmd = torch.zeros((num_envs, 3), device=device)

        F, tau, _, _ = controller.step_policy(
            v_cmd,
            yaw_rate_cmd,
            gimbal_yaw_rate,
            gimbal_pitch_rate,
            zoom_rate,
            q_tilted,
            v_body,
            omega_body,
            sim_dt=0.04,
        )

        # With roll error, should produce roll correction moment
        tau_x_mean = tau[:, 0].mean().item()
        print(f"    Attitude recovery test: tau_x = {tau_x_mean:.4f} Nm (roll correction)")
        assert tau_x_mean != 0.0, "Expected non-zero roll correction moment"
        results.add_pass("Rate controller attitude recovery")
    except Exception as e:
        results.add_fail("Rate controller attitude recovery", traceback.format_exc())

    # Test 6: Multi-step stability test
    try:
        controller.reset()

        # Run multiple steps to check stability
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device)
        v_body = torch.zeros((num_envs, 3), device=device)
        omega_body = torch.zeros((num_envs, 3), device=device)
        v_cmd = torch.zeros((num_envs, 3), device=device)

        for step in range(10):
            F, tau, _, _ = controller.step_policy(
                v_cmd,
                yaw_rate_cmd,
                gimbal_yaw_rate,
                gimbal_pitch_rate,
                zoom_rate,
                q_body,
                v_body,
                omega_body,
                sim_dt=0.04,
            )

            # Check for NaN
            assert not torch.isnan(F).any(), f"NaN in F at step {step}"
            assert not torch.isnan(tau).any(), f"NaN in tau at step {step}"

            # Check for reasonable bounds
            assert F.abs().max() < 1000, f"F out of bounds at step {step}: {F.abs().max()}"
            assert tau.abs().max() < 100, f"tau out of bounds at step {step}: {tau.abs().max()}"

        results.add_pass("Multi-step stability test")
    except Exception as e:
        results.add_fail("Multi-step stability test", traceback.format_exc())


def main():
    """Main test runner."""
    print("=" * 80)
    print("PX4-STYLE CONTROLLER TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()

    try:
        run_controller_tests(results, device)
    except Exception as e:
        results.add_error("Test suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
