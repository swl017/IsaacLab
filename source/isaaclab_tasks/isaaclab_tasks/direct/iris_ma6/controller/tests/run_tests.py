#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone test runner for iris_ma6 controller module.

This script runs all unit tests for the controller components.
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run iris_ma6 controller test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules
import math
import sys
import traceback
from datetime import datetime

import torch
from isaaclab.utils.math import quat_rotate


class TestResults:
    """Track test results with pass/fail reporting."""

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


def run_mixer_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run mixer matrix tests."""
    print("\n" + "=" * 80)
    print("Testing MixerMatrix")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import MixerMatrix

    num_envs = 16
    arm_length = 0.22
    k_f = 8.54858e-06
    k_m = 1.0e-07

    # Test 1: Initialization
    try:
        mixer = MixerMatrix(arm_length=arm_length, k_f=k_f, k_m=k_m, device=device)
        assert mixer.mixer_matrix.shape == (4, 4)
        assert mixer.mixer_matrix_inv.shape == (4, 4)
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return  # Can't continue without mixer

    # Test 2: Aggregate (thrusts -> wrench)
    try:
        # Equal thrusts should give zero moments
        thrusts = torch.ones((num_envs, 4), device=device) * 2.0  # 2N per rotor
        total_thrust, moments = mixer.aggregate(thrusts)

        assert total_thrust.shape == (num_envs,)
        assert moments.shape == (num_envs, 3)
        assert torch.allclose(total_thrust, torch.tensor(8.0, device=device), atol=1e-5)
        assert torch.allclose(moments, torch.zeros_like(moments), atol=1e-5)

        if verbose:
            print(f"    Total thrust: {total_thrust[0].item():.3f} N")
            print(f"    Moments: {moments[0].tolist()}")

        results.add_pass("Aggregate - equal thrusts")
    except AssertionError as e:
        results.add_fail("Aggregate - equal thrusts", str(e))
    except Exception as e:
        results.add_fail("Aggregate - equal thrusts", traceback.format_exc())

    # Test 3: Roll moment generation
    try:
        # Increase left motors (M2, M3), decrease right motors (M1, M4) -> positive roll
        thrusts = torch.tensor([[1.0, 3.0, 3.0, 1.0]], device=device).expand(num_envs, 4)
        total_thrust, moments = mixer.aggregate(thrusts)

        # Roll moment (tau_x) should be positive
        assert moments[:, 0].mean() > 0, f"Roll moment should be positive, got {moments[0, 0].item()}"

        if verbose:
            print(f"    Roll moment (tau_x): {moments[0, 0].item():.4f} Nm")

        results.add_pass("Aggregate - roll moment generation")
    except AssertionError as e:
        results.add_fail("Aggregate - roll moment generation", str(e))
    except Exception as e:
        results.add_fail("Aggregate - roll moment generation", traceback.format_exc())

    # Test 4: Pitch moment generation
    try:
        # Increase rear motors (M2, M4), decrease front motors (M1, M3) -> positive pitch (nose down)
        thrusts = torch.tensor([[1.0, 3.0, 1.0, 3.0]], device=device).expand(num_envs, 4)
        total_thrust, moments = mixer.aggregate(thrusts)

        # Pitch moment (tau_y) should be positive
        assert moments[:, 1].mean() > 0, f"Pitch moment should be positive, got {moments[0, 1].item()}"

        if verbose:
            print(f"    Pitch moment (tau_y): {moments[0, 1].item():.4f} Nm")

        results.add_pass("Aggregate - pitch moment generation")
    except AssertionError as e:
        results.add_fail("Aggregate - pitch moment generation", str(e))
    except Exception as e:
        results.add_fail("Aggregate - pitch moment generation", traceback.format_exc())

    # Test 5: Yaw moment generation (ENU convention)
    try:
        # Increase CW motors (M3, M4), decrease CCW motors (M1, M2) -> positive yaw (CCW = nose left)
        thrusts = torch.tensor([[1.0, 1.0, 3.0, 3.0]], device=device).expand(num_envs, 4)
        total_thrust, moments = mixer.aggregate(thrusts)

        # Yaw moment (tau_z) should be positive in ENU
        assert moments[:, 2].mean() > 0, f"Yaw moment should be positive, got {moments[0, 2].item()}"

        if verbose:
            print(f"    Yaw moment (tau_z): {moments[0, 2].item():.4f} Nm")

        results.add_pass("Aggregate - yaw moment generation (ENU)")
    except AssertionError as e:
        results.add_fail("Aggregate - yaw moment generation (ENU)", str(e))
    except Exception as e:
        results.add_fail("Aggregate - yaw moment generation (ENU)", traceback.format_exc())

    # Test 6: Allocate (wrench -> thrusts)
    try:
        thrust_cmd = torch.full((num_envs,), 10.0, device=device)  # 10N total
        moment_cmd = torch.zeros((num_envs, 3), device=device)

        thrusts = mixer.allocate(thrust_cmd, moment_cmd)

        assert thrusts.shape == (num_envs, 4)
        # Each rotor should get ~2.5N
        assert torch.allclose(thrusts, torch.full_like(thrusts, 2.5), atol=0.1)

        results.add_pass("Allocate - thrust only")
    except AssertionError as e:
        results.add_fail("Allocate - thrust only", str(e))
    except Exception as e:
        results.add_fail("Allocate - thrust only", traceback.format_exc())

    # Test 7: Round-trip (allocate then aggregate)
    try:
        thrust_cmd = torch.full((num_envs,), 15.0, device=device)
        moment_cmd = torch.tensor([[0.5, -0.3, 0.1]], device=device).expand(num_envs, 3)

        thrusts = mixer.allocate(thrust_cmd, moment_cmd, thrust_min=0.0)
        total_thrust, moments = mixer.aggregate(thrusts)

        # Should recover original commands (within tolerance)
        assert torch.allclose(total_thrust, thrust_cmd, atol=0.5)
        assert torch.allclose(moments, moment_cmd, atol=0.1)

        results.add_pass("Round-trip - allocate then aggregate")
    except AssertionError as e:
        results.add_fail("Round-trip - allocate then aggregate", str(e))
    except Exception as e:
        results.add_fail("Round-trip - allocate then aggregate", traceback.format_exc())

    # Test 8: Thrust to omega conversion
    try:
        thrusts = torch.full((num_envs, 4), 1.0, device=device)  # 1N per rotor
        omega = mixer.thrust_to_omega(thrusts)

        expected_omega = math.sqrt(1.0 / k_f)
        assert torch.allclose(omega, torch.full_like(omega, expected_omega), atol=1.0)

        if verbose:
            print(f"    Omega for 1N thrust: {omega[0, 0].item():.1f} rad/s")

        results.add_pass("Thrust to omega conversion")
    except AssertionError as e:
        results.add_fail("Thrust to omega conversion", str(e))
    except Exception as e:
        results.add_fail("Thrust to omega conversion", traceback.format_exc())


def run_motor_dynamics_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run motor dynamics tests."""
    print("\n" + "=" * 80)
    print("Testing MotorDynamics")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import MotorDynamics, MotorDynamicsCfg

    num_envs = 16
    cfg = MotorDynamicsCfg()

    # Test 1: Initialization
    try:
        motor = MotorDynamics(cfg=cfg, num_envs=num_envs, device=device)
        assert motor.omega.shape == (num_envs, 4)
        assert torch.allclose(motor.omega, torch.full_like(motor.omega, cfg.omega_min))
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # Test 2: First-order step response
    try:
        motor.reset()
        omega_cmd = torch.full((num_envs, 4), cfg.omega_max, device=device)
        dt = 0.01

        # After one tau, should reach ~63% of final value
        steps_per_tau = int(cfg.tau_motor / dt)
        for _ in range(steps_per_tau):
            motor.step(omega_cmd, dt)

        # Should be approximately 63% of the way
        expected = cfg.omega_min + 0.632 * (cfg.omega_max - cfg.omega_min)
        actual = motor.omega[0, 0].item()

        assert abs(actual - expected) / expected < 0.1, f"Expected ~{expected:.1f}, got {actual:.1f}"

        if verbose:
            print(f"    After 1 tau: expected ~{expected:.1f}, got {actual:.1f} rad/s")

        results.add_pass("First-order step response (63% in tau)")
    except AssertionError as e:
        results.add_fail("First-order step response (63% in tau)", str(e))
    except Exception as e:
        results.add_fail("First-order step response (63% in tau)", traceback.format_exc())

    # Test 3: Saturation at omega_max
    try:
        motor.reset()
        omega_cmd = torch.full((num_envs, 4), cfg.omega_max * 2, device=device)  # Above max

        for _ in range(100):
            motor.step(omega_cmd, 0.01)

        assert torch.all(motor.omega <= cfg.omega_max + 1e-5)
        results.add_pass("Saturation at omega_max")
    except AssertionError as e:
        results.add_fail("Saturation at omega_max", str(e))
    except Exception as e:
        results.add_fail("Saturation at omega_max", traceback.format_exc())

    # Test 4: Saturation at omega_min
    try:
        motor.reset()
        omega_cmd = torch.full((num_envs, 4), 0.0, device=device)  # Below min

        for _ in range(100):
            motor.step(omega_cmd, 0.01)

        assert torch.all(motor.omega >= cfg.omega_min - 1e-5)
        results.add_pass("Saturation at omega_min")
    except AssertionError as e:
        results.add_fail("Saturation at omega_min", str(e))
    except Exception as e:
        results.add_fail("Saturation at omega_min", traceback.format_exc())

    # Test 5: Thrust/torque computation
    try:
        motor.reset()
        # Set to known omega
        test_omega = 500.0
        motor._omega = torch.full((num_envs, 4), test_omega, device=device)

        thrust, torque = motor.compute_thrust_torque()

        expected_thrust = cfg.k_f * test_omega**2
        expected_torque = cfg.k_m * test_omega**2

        assert torch.allclose(thrust, torch.full_like(thrust, expected_thrust), rtol=1e-4)
        assert torch.allclose(torque, torch.full_like(torque, expected_torque), rtol=1e-4)

        if verbose:
            print(f"    Thrust at {test_omega} rad/s: {thrust[0, 0].item():.4f} N")
            print(f"    Torque at {test_omega} rad/s: {torque[0, 0].item():.6f} Nm")

        results.add_pass("Thrust/torque computation")
    except AssertionError as e:
        results.add_fail("Thrust/torque computation", str(e))
    except Exception as e:
        results.add_fail("Thrust/torque computation", traceback.format_exc())

    # Test 6: Body wrench computation
    try:
        motor.reset()
        # Equal speeds should give pure thrust, no moments
        test_omega = 600.0
        motor._omega = torch.full((num_envs, 4), test_omega, device=device)

        force, moment = motor.compute_body_wrench()

        assert force.shape == (num_envs, 3)
        assert moment.shape == (num_envs, 3)
        # Force should be in +Z direction
        assert torch.all(force[:, :2].abs() < 1e-5)
        assert torch.all(force[:, 2] > 0)
        # Moments should be near zero
        assert torch.allclose(moment, torch.zeros_like(moment), atol=1e-4)

        if verbose:
            print(f"    Body force: {force[0].tolist()}")
            print(f"    Body moment: {moment[0].tolist()}")

        results.add_pass("Body wrench computation")
    except AssertionError as e:
        results.add_fail("Body wrench computation", str(e))
    except Exception as e:
        results.add_fail("Body wrench computation", traceback.format_exc())

    # Test 7: Reset functionality
    try:
        motor._omega = torch.full((num_envs, 4), 800.0, device=device)
        env_ids = torch.tensor([0, 5, 10], device=device)

        motor.reset(env_ids)

        assert torch.allclose(motor.omega[env_ids], torch.full((3, 4), cfg.omega_min, device=device))
        assert not torch.allclose(motor.omega[1], torch.full((4,), cfg.omega_min, device=device))

        results.add_pass("Reset functionality (partial)")
    except AssertionError as e:
        results.add_fail("Reset functionality (partial)", str(e))
    except Exception as e:
        results.add_fail("Reset functionality (partial)", traceback.format_exc())


def run_zoom_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run zoom controller tests."""
    print("\n" + "=" * 80)
    print("Testing ZoomController")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import ZoomController, ZoomControllerCfg

    num_envs = 16
    cfg = ZoomControllerCfg()

    # Test 1: Initialization
    try:
        zoom = ZoomController(cfg=cfg, num_envs=num_envs, device=device)
        assert zoom.zoom.shape == (num_envs,)
        assert torch.allclose(zoom.zoom, torch.ones(num_envs, device=device))
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # Test 2: First-order response
    try:
        zoom.reset()
        dt = 0.01

        # Command max zoom rate
        zoom_rate_cmd = torch.ones(num_envs, device=device)  # +1 = zoom in

        # After one tau, should reach ~63% of target
        steps_per_tau = int(cfg.tau_zoom / dt)
        for _ in range(steps_per_tau):
            zoom.compute_control(zoom_rate_cmd, dt)

        # Target after this time: 1.0 + max_rate * tau_zoom
        target = 1.0 + cfg.max_zoom_rate * cfg.tau_zoom
        expected = 1.0 + 0.632 * (target - 1.0)
        actual = zoom.zoom[0].item()

        assert abs(actual - expected) / expected < 0.2, f"Expected ~{expected:.2f}, got {actual:.2f}"

        if verbose:
            print(f"    After 1 tau: expected ~{expected:.2f}, got {actual:.2f}")

        results.add_pass("First-order response")
    except AssertionError as e:
        results.add_fail("First-order response", str(e))
    except Exception as e:
        results.add_fail("First-order response", traceback.format_exc())

    # Test 3: Range limits
    try:
        zoom.reset()
        dt = 0.01

        # Zoom way in
        for _ in range(500):
            zoom.compute_control(torch.ones(num_envs, device=device), dt)

        assert torch.all(zoom.zoom <= cfg.zoom_max + 1e-5)

        # Zoom way out
        for _ in range(1000):
            zoom.compute_control(-torch.ones(num_envs, device=device), dt)

        assert torch.all(zoom.zoom >= cfg.zoom_min - 1e-5)

        results.add_pass("Range limits")
    except AssertionError as e:
        results.add_fail("Range limits", str(e))
    except Exception as e:
        results.add_fail("Range limits", traceback.format_exc())

    # Test 4: FOV computation
    try:
        zoom.reset()
        base_fov = 90.0

        # At 1x zoom, FOV should equal base
        fov = zoom.get_fov(base_fov)
        assert torch.allclose(fov, torch.full((num_envs,), base_fov, device=device))

        # At 2x zoom, FOV should be halved
        zoom.set_zoom(torch.tensor(2.0, device=device))
        fov = zoom.get_fov(base_fov)
        assert torch.allclose(fov, torch.full((num_envs,), 45.0, device=device))

        results.add_pass("FOV computation")
    except AssertionError as e:
        results.add_fail("FOV computation", str(e))
    except Exception as e:
        results.add_fail("FOV computation", traceback.format_exc())


def run_gimbal_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run gimbal controller tests."""
    print("\n" + "=" * 80)
    print("Testing GimbalController")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import GimbalController, GimbalControllerCfg
    from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller import YAW_JOINT_OFFSET

    num_envs = 16
    cfg = GimbalControllerCfg()

    # Default args for compute_control
    omega_zero = torch.zeros((num_envs, 3), device=device)
    # joint_positions_actual [pitch, yaw, roll] — raw yaw includes YAW_JOINT_OFFSET
    jp_zero = torch.zeros((num_envs, 3), device=device)
    jp_zero[:, 1] = YAW_JOINT_OFFSET  # yaw slot has offset applied

    # Test 1: Initialization
    try:
        gimbal = GimbalController(cfg=cfg, num_envs=num_envs, device=device)
        assert gimbal.yaw.shape == (num_envs,)
        assert gimbal.roll.shape == (num_envs,)
        assert gimbal.pitch.shape == (num_envs,)
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # Test 2: Rate integration
    try:
        gimbal.reset()
        dt = 0.01

        # Command yaw rate
        yaw_rate = torch.ones(num_envs, device=device) * 0.5  # 50% of max
        pitch_rate = torch.zeros(num_envs, device=device)
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        # After 0.5s, should have rotated about 0.5 * 0.5 * max_rate * 0.5 (accounting for lag)
        for _ in range(50):
            gimbal.compute_control(yaw_rate, pitch_rate, q_body, dt, omega_zero, jp_zero)

        assert gimbal.yaw[0].abs() > 0.1, f"Yaw should have increased, got {gimbal.yaw[0].item()}"

        if verbose:
            print(f"    Yaw after 0.5s: {math.degrees(gimbal.yaw[0].item()):.1f} deg")

        results.add_pass("Rate integration")
    except AssertionError as e:
        results.add_fail("Rate integration", str(e))
    except Exception as e:
        results.add_fail("Rate integration", traceback.format_exc())

    # Test 3: Joint limits
    try:
        gimbal.reset()
        dt = 0.01
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        # Command max yaw rate for long time
        for _ in range(500):
            gimbal.compute_control(
                torch.ones(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                q_body,
                dt,
                omega_zero,
                jp_zero,
            )

        assert torch.all(gimbal.yaw <= cfg.yaw_limits[1] + 1e-5)
        assert torch.all(gimbal.yaw >= cfg.yaw_limits[0] - 1e-5)

        results.add_pass("Joint limits")
    except AssertionError as e:
        results.add_fail("Joint limits", str(e))
    except Exception as e:
        results.add_fail("Joint limits", traceback.format_exc())

    # Test 4: Instant analytical tracking (body-frame targets track in one step)
    try:
        cfg_inst = GimbalControllerCfg(feedback_blend=0.0)
        g_inst = GimbalController(cfg=cfg_inst, num_envs=num_envs, device=device)
        g_inst.reset()
        dt = 0.01
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        jp_inst = torch.zeros((num_envs, 3), device=device)
        jp_inst[:, 1] = YAW_JOINT_OFFSET

        # Set a world-frame target azimuth and step once
        g_inst._azimuth_world = torch.full((num_envs,), 0.5, device=device)

        g_inst.compute_control(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            q_body, dt, omega_zero, jp_inst,
        )

        # With analytical decomposition, yaw should match azimuth instantly
        actual = g_inst.yaw[0].item()
        expected = 0.5

        if verbose:
            print(f"    Azimuth=0.5, body yaw={actual:.4f} (expected {expected:.4f})")

        assert abs(actual - expected) < 0.001, (
            f"Analytical tracking should be instant: expected {expected:.4f}, got {actual:.4f}"
        )
        results.add_pass("Instant analytical tracking (1 step)")
    except AssertionError as e:
        results.add_fail("Instant analytical tracking", str(e))
    except Exception as e:
        results.add_fail("Instant analytical tracking", traceback.format_exc())

    # Test 5: World-frame stabilization (gimbal compensates for drone tilt)
    try:
        gimbal.reset()
        dt = 0.01

        # Start with drone level, gimbal pointing forward (+X world)
        q_level = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        # Track actual joint positions (simulating perfect actuator tracking)
        jp_track = jp_zero.clone()

        # Run a few steps to stabilize at forward direction
        for _ in range(50):
            gimbal.compute_control(
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                q_level,
                dt,
                omega_zero,
                jp_track,
            )
            # Simulate perfect actuator: actual = commanded + offset
            jp_track[:, 0] = gimbal.pitch
            jp_track[:, 1] = gimbal.yaw + YAW_JOINT_OFFSET
            jp_track[:, 2] = gimbal.roll

        # Record initial body-frame yaw
        yaw_level = gimbal.yaw[0].item()

        # Now tilt the drone 30 degrees in yaw (rotate around Z)
        angle = math.radians(30)
        q_tilted = torch.tensor(
            [[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]], device=device
        ).expand(num_envs, 4)

        # Run more steps - gimbal should compensate to maintain world-frame direction
        for _ in range(100):
            gimbal.compute_control(
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                q_tilted,
                dt,
                omega_zero,
                jp_track,
            )
            jp_track[:, 0] = gimbal.pitch
            jp_track[:, 1] = gimbal.yaw + YAW_JOINT_OFFSET
            jp_track[:, 2] = gimbal.roll

        # After compensation, body-frame yaw should have changed by ~30 deg
        yaw_tilted = gimbal.yaw[0].item()
        yaw_change = abs(yaw_tilted - yaw_level)

        # The gimbal should have counter-rotated to compensate
        assert yaw_change > math.radians(20), (
            f"Gimbal should compensate for drone rotation, "
            f"got only {math.degrees(yaw_change):.1f} deg change"
        )

        if verbose:
            print(f"    Yaw level: {math.degrees(yaw_level):.1f} deg")
            print(f"    Yaw tilted: {math.degrees(yaw_tilted):.1f} deg")
            print(f"    Compensation: {math.degrees(yaw_change):.1f} deg")

        results.add_pass("World-frame stabilization")
    except AssertionError as e:
        results.add_fail("World-frame stabilization", str(e))
    except Exception as e:
        results.add_fail("World-frame stabilization", traceback.format_exc())

    # Test 6: World-frame azimuth stable during continuous yaw rotation
    try:
        dt = 0.01
        zero_rate = torch.zeros(num_envs, device=device)
        cfg_rot = GimbalControllerCfg(feedback_blend=0.0)
        g_rot = GimbalController(cfg=cfg_rot, num_envs=num_envs, device=device)
        g_rot.reset()

        jp_rot = jp_zero.clone()
        errors = []
        angle = 0.0
        for step in range(200):
            angle += 1.0 * dt  # 1 rad/s yaw rotation
            cos_ha = math.cos(angle / 2)
            sin_ha = math.sin(angle / 2)
            q_rot = torch.tensor([[cos_ha, 0.0, 0.0, sin_ha]], device=device).expand(num_envs, 4)
            g_rot.compute_control(zero_rate, zero_rate, q_rot, dt, omega_zero, jp_rot)
            jp_rot[:, 0] = g_rot.pitch
            jp_rot[:, 1] = g_rot.yaw + YAW_JOINT_OFFSET
            jp_rot[:, 2] = g_rot.roll
            errors.append(g_rot.azimuth_world[0].abs().item())

        rms = (sum(e**2 for e in errors) / len(errors)) ** 0.5

        if verbose:
            print(f"    Azimuth RMS during 1 rad/s yaw: {math.degrees(rms):.4f} deg")

        assert rms < math.radians(0.1), f"Azimuth should stay at 0 during rotation, RMS={math.degrees(rms):.3f}°"
        results.add_pass(f"Azimuth stable during yaw rotation (RMS={math.degrees(rms):.4f}°)")
    except AssertionError as e:
        results.add_fail("Azimuth stable during yaw rotation", str(e))
    except Exception as e:
        results.add_fail("Azimuth stable during yaw rotation", traceback.format_exc())

    # Test 7: Feedback corrects internal state drift
    try:
        cfg_fb = GimbalControllerCfg(feedback_blend=0.1)
        g_fb = GimbalController(cfg=cfg_fb, num_envs=num_envs, device=device)
        g_fb.reset()
        dt = 0.01
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        # Artificially perturb internal yaw by 0.1 rad (simulating drift)
        g_fb._yaw += 0.1

        # Actual joint positions reflect truth (yaw=0 + offset)
        jp_truth = torch.zeros((num_envs, 3), device=device)
        jp_truth[:, 1] = YAW_JOINT_OFFSET  # yaw actual = 0 + offset

        initial_error = g_fb._yaw[0].item()

        # Run 50 steps with feedback
        for _ in range(50):
            g_fb.compute_control(
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                q_body,
                dt,
                omega_zero,
                jp_truth,
            )

        final_error = g_fb._yaw[0].item()

        if verbose:
            print(f"    Initial drift: {math.degrees(initial_error):.2f} deg")
            print(f"    After 50 steps: {math.degrees(final_error):.2f} deg")

        assert abs(final_error) < abs(initial_error) * 0.5, (
            f"Feedback should reduce drift: initial={math.degrees(initial_error):.2f}°, "
            f"final={math.degrees(final_error):.2f}°"
        )
        results.add_pass("Feedback corrects drift")
    except AssertionError as e:
        results.add_fail("Feedback corrects drift", str(e))
    except Exception as e:
        results.add_fail("Feedback corrects drift", traceback.format_exc())

    # Test 8: No swing under abrupt tilt change
    try:
        cfg_ns = GimbalControllerCfg(feedback_blend=0.0)
        g_ns = GimbalController(cfg=cfg_ns, num_envs=num_envs, device=device)
        g_ns.reset()
        dt = 0.01
        jp_ns = torch.zeros((num_envs, 3), device=device)
        jp_ns[:, 1] = YAW_JOINT_OFFSET

        # Start level, pointing forward (azimuth=0)
        q_level = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        g_ns.compute_control(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            q_level, dt, omega_zero, jp_ns,
        )
        az_before = g_ns.azimuth_world[0].item()

        # Abruptly tilt drone 30 degrees in yaw
        angle = math.radians(30)
        q_tilted = torch.tensor(
            [[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]], device=device
        ).expand(num_envs, 4)

        # Single step after tilt — body-frame yaw should adapt instantly
        jp_ns[:, 1] = g_ns.yaw + YAW_JOINT_OFFSET
        g_ns.compute_control(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            q_tilted, dt, omega_zero, jp_ns,
        )
        az_after = g_ns.azimuth_world[0].item()

        # World-frame azimuth should NOT change (no swing)
        az_drift = abs(az_after - az_before)

        if verbose:
            print(f"    Azimuth before tilt: {math.degrees(az_before):.2f} deg")
            print(f"    Azimuth after tilt:  {math.degrees(az_after):.2f} deg")
            print(f"    Drift: {math.degrees(az_drift):.4f} deg")

        assert az_drift < math.radians(0.1), (
            f"World-frame azimuth should not change on tilt: drifted {math.degrees(az_drift):.3f} deg"
        )
        results.add_pass("No swing under abrupt tilt")
    except AssertionError as e:
        results.add_fail("No swing under abrupt tilt", str(e))
    except Exception as e:
        results.add_fail("No swing under abrupt tilt", traceback.format_exc())

    # Test 9: Cross-axis coupling at +/-30 deg with first-order actuator model
    # Simulates the actuator as a first-order lag (tau = d/k = 50/1000 = 0.05s).
    # With decoupled roll, roll saturation should NOT affect LOS pointing accuracy.
    try:
        from isaaclab.utils.math import quat_from_euler_xyz

        cfg_cx = GimbalControllerCfg(feedback_blend=0.1)
        g_cx = GimbalController(cfg=cfg_cx, num_envs=num_envs, device=device)
        g_cx.reset()
        dt = 0.01
        tau_act = 0.05  # d/k = 50/1000
        alpha = 1.0 - math.exp(-dt / tau_act)

        # Simulated actuator positions [pitch, yaw, roll]
        act_pos = torch.zeros((num_envs, 3), device=device)
        act_pos[:, 1] = YAW_JOINT_OFFSET

        # Initialize
        q_level = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        for _ in range(20):
            pos_tgt, vel_tgt = g_cx.compute_control(
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                q_level, dt, omega_zero, act_pos,
            )
            pos_target = torch.stack([pos_tgt[2], pos_tgt[0] + YAW_JOINT_OFFSET, pos_tgt[1]], dim=-1)
            vel_target = torch.stack([vel_tgt[2], vel_tgt[0], vel_tgt[1]], dim=-1)
            act_pos = act_pos + alpha * (pos_target - act_pos) + vel_target * dt

        initial_dir = g_cx.get_camera_direction_world(q_level)

        max_error_deg = 0.0
        pitch_angle = 0.0
        roll_angle = 0.0
        rate_deg_per_step = 0.2

        maneuvers = [
            (150, "pitch", +1), (150, "roll", +1),
            (300, "pitch", -1), (300, "roll", -1),
            (300, "pitch", +1), (300, "roll", +1),
            (150, "pitch", -1), (150, "roll", -1),
        ]

        for steps, axis, sign in maneuvers:
            for _ in range(steps):
                delta = math.radians(rate_deg_per_step * sign)
                if axis == "pitch":
                    pitch_angle += delta
                else:
                    roll_angle += delta

                omega = torch.zeros((num_envs, 3), device=device)
                if axis == "pitch":
                    omega[:, 1] = delta / dt
                else:
                    omega[:, 0] = delta / dt

                q_man = quat_from_euler_xyz(
                    torch.full((num_envs,), roll_angle, device=device),
                    torch.full((num_envs,), pitch_angle, device=device),
                    torch.zeros(num_envs, device=device),
                )

                pos_tgt, vel_tgt = g_cx.compute_control(
                    torch.zeros(num_envs, device=device),
                    torch.zeros(num_envs, device=device),
                    q_man, dt, omega, act_pos,
                )

                pos_target = torch.stack([pos_tgt[2], pos_tgt[0] + YAW_JOINT_OFFSET, pos_tgt[1]], dim=-1)
                vel_target = torch.stack([vel_tgt[2], vel_tgt[0], vel_tgt[1]], dim=-1)
                # First-order actuator with velocity feedforward
                act_pos = act_pos + alpha * (pos_target - act_pos) + vel_target * dt

                # Camera forward from actual joint positions
                a_yaw = act_pos[:, 1] - YAW_JOINT_OFFSET
                a_roll = act_pos[:, 2]
                a_pitch = act_pos[:, 0]
                cy, sy = torch.cos(a_yaw), torch.sin(a_yaw)
                cr, sr = torch.cos(a_roll), torch.sin(a_roll)
                cp, sp = torch.cos(a_pitch), torch.sin(a_pitch)
                fwd_body = torch.stack([cy*cp + sy*sr*sp, sy*cp - cy*sr*sp, -cr*sp], dim=-1)
                fwd_world = quat_rotate(q_man, fwd_body)

                dot = (initial_dir * fwd_world).sum(dim=-1).clamp(-1.0, 1.0)
                error_deg = math.degrees(torch.acos(dot)[0].item())
                max_error_deg = max(max_error_deg, error_deg)

        if verbose:
            print(f"    Max direction error during +/-30 deg maneuver: {max_error_deg:.2f} deg")

        # With decoupled roll, roll saturation does not cascade into LOS error.
        # The analytical position loop gives exact yaw/pitch independent of roll.
        # Remaining error is only from actuator lag tracking the position targets.
        # Remaining error is from the simulated first-order actuator lag (tau=0.05s),
        # not from roll-yaw/pitch coupling (which is now zero by construction).
        assert max_error_deg < 20.0, (
            f"Cross-axis error too high: {max_error_deg:.2f} deg exceeds 20 deg limit"
        )
        results.add_pass(f"Cross-axis +/-30 deg decoupled (max err={max_error_deg:.2f} deg)")
    except AssertionError as e:
        results.add_fail("Cross-axis coupling", str(e))
    except Exception as e:
        results.add_fail("Cross-axis coupling", traceback.format_exc())

    # Test 10: Position/velocity coherence
    # Verify that (pos[t+1] - pos[t]) / dt ~ vel[t]
    try:
        cfg_coh = GimbalControllerCfg(feedback_blend=0.0)
        g_coh = GimbalController(cfg=cfg_coh, num_envs=num_envs, device=device)
        g_coh.reset()
        dt = 0.01
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        jp_coh = torch.zeros((num_envs, 3), device=device)
        jp_coh[:, 1] = YAW_JOINT_OFFSET

        # Set a target to create non-zero velocity
        g_coh._azimuth_world = torch.full((num_envs,), 0.3, device=device)
        g_coh._elevation_world = torch.full((num_envs,), 0.2, device=device)

        # Step and record
        prev_yaw = g_coh.yaw[0].item()
        prev_roll = g_coh.roll[0].item()
        prev_pitch = g_coh.pitch[0].item()

        pos_tgt, vel_tgt = g_coh.compute_control(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            q_body, dt, omega_zero, jp_coh,
        )

        # Finite difference: (pos_new - pos_old) / dt should match vel_tgt
        fd_yaw = (g_coh.yaw[0].item() - prev_yaw) / dt
        fd_roll = (g_coh.roll[0].item() - prev_roll) / dt
        fd_pitch = (g_coh.pitch[0].item() - prev_pitch) / dt

        vel_yaw = vel_tgt[0][0].item()
        vel_roll = vel_tgt[1][0].item()
        vel_pitch = vel_tgt[2][0].item()

        max_diff = max(abs(fd_yaw - vel_yaw), abs(fd_roll - vel_roll), abs(fd_pitch - vel_pitch))

        if verbose:
            print(f"    FD yaw={fd_yaw:.3f}, vel_yaw={vel_yaw:.3f}")
            print(f"    FD roll={fd_roll:.3f}, vel_roll={vel_roll:.3f}")
            print(f"    FD pitch={fd_pitch:.3f}, vel_pitch={vel_pitch:.3f}")
            print(f"    Max diff: {max_diff:.6f}")

        assert max_diff < 1e-4, (
            f"Position/velocity incoherence: max diff={max_diff:.6f}"
        )
        results.add_pass("Position/velocity coherence")
    except AssertionError as e:
        results.add_fail("Position/velocity coherence", str(e))
    except Exception as e:
        results.add_fail("Position/velocity coherence", traceback.format_exc())


def run_attitude_controller_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run attitude controller tests.

    The AttitudeController (4-loop architecture) is P-only and outputs rate setpoints,
    not torques. The rate setpoint is then fed to the RateController.
    """
    print("\n" + "=" * 80)
    print("Testing AttitudeController")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import (
        AttitudeController,
        AttitudeControllerCfg,
    )

    num_envs = 16
    cfg = AttitudeControllerCfg()

    # Test 1: Initialization (4-loop architecture: no mixer required)
    try:
        att = AttitudeController(cfg=cfg, num_envs=num_envs, device=device)
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # Test 2: Identity quaternion (no error) -> zero rate setpoint
    try:
        q_des = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        q_current = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        rate_setpoint = att.compute_control(
            q_des=q_des,
            yaw_rate_des=torch.zeros(num_envs, device=device),
            q_current=q_current,
        )

        assert rate_setpoint.shape == (num_envs, 3)
        # At identity, rate setpoint should be near zero
        assert torch.allclose(rate_setpoint, torch.zeros_like(rate_setpoint), atol=1e-5)

        if verbose:
            print(f"    Rate setpoint at identity: {rate_setpoint[0].tolist()}")

        results.add_pass("Identity quaternion (no error)")
    except AssertionError as e:
        results.add_fail("Identity quaternion (no error)", str(e))
    except Exception as e:
        results.add_fail("Identity quaternion (no error)", traceback.format_exc())

    # Test 3: Roll error generates corrective rate setpoint
    try:
        # 10 degree roll error
        angle = math.radians(10)
        q_des = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        q_current = torch.tensor(
            [[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]], device=device
        ).expand(num_envs, 4)

        rate_setpoint = att.compute_control(
            q_des=q_des,
            yaw_rate_des=torch.zeros(num_envs, device=device),
            q_current=q_current,
        )

        # Rate setpoint should have non-zero roll rate to correct the error
        roll_rate = rate_setpoint[:, 0].mean().item()
        assert abs(roll_rate) > 0.1, f"Should have roll rate correction, got {roll_rate}"

        if verbose:
            print(f"    Roll rate setpoint for 10° error: {math.degrees(roll_rate):.1f} deg/s")

        results.add_pass("Roll error generates corrective rate setpoint")
    except AssertionError as e:
        results.add_fail("Roll error generates corrective rate setpoint", str(e))
    except Exception as e:
        results.add_fail("Roll error generates corrective rate setpoint", traceback.format_exc())

    # Test 4: Attitude error computation
    try:
        # Test attitude error is correct
        angle = math.radians(20)
        q_des = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        q_curr = torch.tensor(
            [[math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]], device=device
        ).expand(num_envs, 4)  # Pitch rotation

        att_err = att.compute_attitude_error(q_des, q_curr)

        # Error should be primarily in pitch (Y axis)
        assert att_err.shape == (num_envs, 3)
        assert abs(att_err[0, 1].item()) > abs(att_err[0, 0].item())  # Pitch > Roll
        assert abs(att_err[0, 1].item()) > abs(att_err[0, 2].item())  # Pitch > Yaw

        if verbose:
            print(f"    Attitude error: {att_err[0].tolist()}")

        results.add_pass("Attitude error computation")
    except AssertionError as e:
        results.add_fail("Attitude error computation", str(e))
    except Exception as e:
        results.add_fail("Attitude error computation", traceback.format_exc())


def run_velocity_controller_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run velocity controller tests."""
    print("\n" + "=" * 80)
    print("Testing VelocityController")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import VelocityController, VelocityControllerCfg

    num_envs = 16
    cfg = VelocityControllerCfg()
    mass = 1.5  # kg
    gravity = 9.81  # m/s^2

    # Test 1: Initialization
    try:
        vel = VelocityController(cfg=cfg, mass=mass, gravity=gravity, num_envs=num_envs, device=device)
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # Test 2: Hover (zero velocity command)
    try:
        v_des = torch.zeros((num_envs, 3), device=device)
        yaw_rate_des = torch.zeros(num_envs, device=device)
        v_current = torch.zeros((num_envs, 3), device=device)
        q_current = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        q_des, thrust, yaw_rate_out = vel.compute_control(
            v_des=v_des, yaw_rate_des=yaw_rate_des, v_current=v_current, q_current=q_current, dt=0.01
        )

        assert q_des.shape == (num_envs, 4)
        assert thrust.shape == (num_envs,)

        # At hover, desired attitude should be near identity
        q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        assert torch.allclose(q_des, q_identity.expand(num_envs, 4), atol=0.1)

        # Thrust should be approximately hover thrust (m * g)
        hover_thrust = mass * gravity
        assert (thrust.mean().item() - hover_thrust) / hover_thrust < 0.5

        if verbose:
            print(f"    q_des: {q_des[0].tolist()}")
            print(f"    Thrust: {thrust.mean().item():.2f} N")

        results.add_pass("Hover (zero velocity)")
    except AssertionError as e:
        results.add_fail("Hover (zero velocity)", str(e))
    except Exception as e:
        results.add_fail("Hover (zero velocity)", traceback.format_exc())

    # Test 3: Forward velocity command tilts forward
    try:
        vel.reset()
        v_des = torch.zeros((num_envs, 3), device=device)
        v_des[:, 0] = 5.0  # Forward velocity command
        yaw_rate_des = torch.zeros(num_envs, device=device)
        v_current = torch.zeros((num_envs, 3), device=device)
        q_current = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        q_des, thrust, _ = vel.compute_control(
            v_des=v_des, yaw_rate_des=yaw_rate_des, v_current=v_current, q_current=q_current, dt=0.01
        )

        # Forward velocity should result in forward pitch (negative pitch in FLU)
        # Extract pitch from quaternion
        pitch = 2 * torch.atan2(q_des[:, 2], q_des[:, 0])  # Approximation for small angles

        # Should have some pitch
        assert pitch.abs().mean() > 0.01, f"Should have pitch, got {pitch.mean().item()}"

        if verbose:
            print(f"    Pitch for 5 m/s forward: {math.degrees(pitch[0].item()):.1f} deg")

        results.add_pass("Forward velocity tilts attitude")
    except AssertionError as e:
        results.add_fail("Forward velocity tilts attitude", str(e))
    except Exception as e:
        results.add_fail("Forward velocity tilts attitude", traceback.format_exc())

    # Test 4: Integral anti-windup
    try:
        vel.reset()
        # Create large persistent error
        v_des = torch.full((num_envs, 3), 10.0, device=device)
        yaw_rate_des = torch.zeros(num_envs, device=device)
        v_current = torch.zeros((num_envs, 3), device=device)
        q_current = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        # Run many iterations
        for _ in range(100):
            vel.compute_control(v_des=v_des, yaw_rate_des=yaw_rate_des, v_current=v_current, q_current=q_current, dt=0.1)

        # Integral should be bounded by anti-windup
        integral_norm = vel._vel_integral.norm(dim=-1).mean().item()
        assert integral_norm < 100, f"Integral should be bounded, got {integral_norm}"

        if verbose:
            print(f"    Integral norm after 100 steps: {integral_norm:.2f}")

        results.add_pass("Integral anti-windup")
    except AssertionError as e:
        results.add_fail("Integral anti-windup", str(e))
    except Exception as e:
        results.add_fail("Integral anti-windup", traceback.format_exc())


def run_aerodynamics_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run aerodynamics tests."""
    print("\n" + "=" * 80)
    print("Testing AerodynamicEffects")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import AerodynamicEffects, AerodynamicsCfg

    num_envs = 16

    # Test 1: Level 0 - no effects
    try:
        cfg = AerodynamicsCfg(fidelity_level=0)
        aero = AerodynamicEffects(cfg=cfg, num_envs=num_envs, device=device)

        v_body = torch.randn((num_envs, 3), device=device) * 5.0
        omega_rotors = torch.full((num_envs, 4), 500.0, device=device)
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        F, tau = aero.compute_forces(v_body, omega_rotors, q_body, dt=0.01)

        assert torch.allclose(F, torch.zeros_like(F))
        assert torch.allclose(tau, torch.zeros_like(tau))

        results.add_pass("Level 0 - disabled")
    except AssertionError as e:
        results.add_fail("Level 0 - disabled", str(e))
    except Exception as e:
        results.add_fail("Level 0 - disabled", traceback.format_exc())

    # Test 2: Level 1 - basic drag
    try:
        cfg = AerodynamicsCfg(fidelity_level=1)
        aero = AerodynamicEffects(cfg=cfg, num_envs=num_envs, device=device)

        # Moving forward at 10 m/s
        v_body = torch.zeros((num_envs, 3), device=device)
        v_body[:, 0] = 10.0
        omega_rotors = torch.full((num_envs, 4), 500.0, device=device)
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        F, tau = aero.compute_forces(v_body, omega_rotors, q_body, dt=0.01)

        # Drag should oppose motion (negative X)
        assert F[:, 0].mean() < 0, f"Drag should be negative, got {F[0, 0].item()}"
        # No wind, so tau should be near zero at this level
        assert torch.allclose(tau, torch.zeros_like(tau), atol=1e-5)

        if verbose:
            print(f"    Drag at 10 m/s: {F[0].tolist()}")

        results.add_pass("Level 1 - basic drag")
    except AssertionError as e:
        results.add_fail("Level 1 - basic drag", str(e))
    except Exception as e:
        results.add_fail("Level 1 - basic drag", traceback.format_exc())

    # Test 3: Set fidelity level
    try:
        cfg = AerodynamicsCfg(fidelity_level=1)
        aero = AerodynamicEffects(cfg=cfg, num_envs=num_envs, device=device)

        assert aero.fidelity_level == 1
        aero.set_fidelity_level(2)
        assert aero.fidelity_level == 2

        # Invalid level should raise
        try:
            aero.set_fidelity_level(5)
            results.add_fail("Set fidelity level", "Should raise for invalid level")
        except ValueError:
            pass

        results.add_pass("Set fidelity level")
    except Exception as e:
        results.add_fail("Set fidelity level", traceback.format_exc())

    # Test 4: Reset clears gust state
    try:
        cfg = AerodynamicsCfg(fidelity_level=2)
        aero = AerodynamicEffects(cfg=cfg, num_envs=num_envs, device=device)

        # Run a few steps to build up gust
        v_body = torch.zeros((num_envs, 3), device=device)
        omega_rotors = torch.full((num_envs, 4), 500.0, device=device)
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        for _ in range(10):
            aero.compute_forces(v_body, omega_rotors, q_body, dt=0.01)

        # Reset specific envs
        env_ids = torch.tensor([0, 5, 10], device=device)
        aero.reset(env_ids)

        assert torch.allclose(aero._gust[env_ids], torch.zeros((3, 3), device=device))

        results.add_pass("Reset clears gust state")
    except AssertionError as e:
        results.add_fail("Reset clears gust state", str(e))
    except Exception as e:
        results.add_fail("Reset clears gust state", traceback.format_exc())


def run_integration_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Run integration tests with DroneController."""
    print("\n" + "=" * 80)
    print("Testing DroneController (Integration)")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller import DroneController, DroneControllerCfg

    num_envs = 16
    cfg = DroneControllerCfg()
    mass = 1.5  # kg
    gravity = 9.81  # m/s^2

    # Test 1: Initialization
    try:
        drone = DroneController(cfg=cfg, mass=mass, gravity=gravity, num_envs=num_envs, device=device)
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # Test 2: Hover step
    try:
        # Zero velocity command
        v_cmd = torch.zeros((num_envs, 3), device=device)
        yaw_rate_cmd = torch.zeros(num_envs, device=device)
        gimbal_yaw_rate_cmd = torch.zeros(num_envs, device=device)
        gimbal_pitch_rate_cmd = torch.zeros(num_envs, device=device)
        zoom_rate_cmd = torch.zeros(num_envs, device=device)

        # Current state
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4).contiguous()
        v_body = torch.zeros((num_envs, 3), device=device)
        omega_body = torch.zeros((num_envs, 3), device=device)

        gimbal_jp = torch.zeros((num_envs, 3), device=device)
        gimbal_jp[:, 1] = -math.pi / 2  # yaw offset

        F_body, tau_body, gimbal_targets, gimbal_vel_targets, zoom_level = drone.step_policy(
            v_cmd=v_cmd,
            yaw_rate_cmd=yaw_rate_cmd,
            gimbal_yaw_rate_cmd=gimbal_yaw_rate_cmd,
            gimbal_pitch_rate_cmd=gimbal_pitch_rate_cmd,
            zoom_rate_cmd=zoom_rate_cmd,
            q_body=q_body,
            v_body=v_body,
            omega_body=omega_body,
            sim_dt=0.04,  # 25 Hz policy
            gimbal_joint_positions=gimbal_jp,
        )

        assert F_body.shape == (num_envs, 3)
        assert tau_body.shape == (num_envs, 3)
        # gimbal_targets is a tuple of 3 tensors (yaw, roll, pitch)
        assert len(gimbal_targets) == 3
        assert gimbal_targets[0].shape == (num_envs,)
        assert zoom_level.shape == (num_envs,)

        # At hover, thrust should be approximately m*g (in Z direction)
        hover_force = mass * gravity
        # Force is in body frame, so Z component should be positive (upward thrust)
        assert F_body[:, 2].mean().item() > 0, "Force should have positive Z for hover"

        if verbose:
            print(f"    F_body: {F_body[0].tolist()}")
            print(f"    tau_body: {tau_body[0].tolist()}")
            print(f"    Gimbal yaw: {gimbal_targets[0][0].item():.2f}")
            print(f"    Zoom level: {zoom_level[0].item():.2f}")

        results.add_pass("Hover step")
    except AssertionError as e:
        results.add_fail("Hover step", str(e))
    except Exception as e:
        results.add_fail("Hover step", traceback.format_exc())

    # Test 3: Multiple inner loop steps (based on control_dt)
    try:
        drone.reset()

        # Step policy at 25 Hz (sim_dt = 0.04s), inner loop at 100 Hz (control_dt = 0.01s)
        # Should run 4 inner loop iterations
        gimbal_jp2 = torch.zeros((num_envs, 3), device=device)
        gimbal_jp2[:, 1] = -math.pi / 2

        F_body, tau_body, _, _, _ = drone.step_policy(
            v_cmd=torch.zeros((num_envs, 3), device=device),
            yaw_rate_cmd=torch.zeros(num_envs, device=device),
            gimbal_yaw_rate_cmd=torch.zeros(num_envs, device=device),
            gimbal_pitch_rate_cmd=torch.zeros(num_envs, device=device),
            zoom_rate_cmd=torch.zeros(num_envs, device=device),
            q_body=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4).contiguous(),
            v_body=torch.zeros((num_envs, 3), device=device),
            omega_body=torch.zeros((num_envs, 3), device=device),
            sim_dt=0.04,
            gimbal_joint_positions=gimbal_jp2,
        )

        # Just verify it runs (inner loop is implicit)
        num_substeps = max(1, int(0.04 / cfg.control_dt))
        results.add_pass(f"Inner loop ({num_substeps} substeps)")
    except Exception as e:
        results.add_fail("Inner loop", traceback.format_exc())

    # Test 4: Reset functionality
    try:
        drone._motor._omega = torch.full((num_envs, 4), 800.0, device=device)
        drone._zoom._zoom = torch.full((num_envs,), 5.0, device=device)

        env_ids = torch.tensor([0, 5, 10], device=device)
        drone.reset(env_ids)

        # Motor omega should be reset for specified envs
        cfg_motor = cfg.motor
        assert torch.allclose(
            drone.motor_dynamics.omega[env_ids], torch.full((3, 4), cfg_motor.omega_min, device=device)
        )
        # Zoom should be reset
        assert torch.allclose(drone.zoom_controller.zoom[env_ids], torch.ones(3, device=device))

        results.add_pass("Reset functionality")
    except AssertionError as e:
        results.add_fail("Reset functionality", str(e))
    except Exception as e:
        results.add_fail("Reset functionality", traceback.format_exc())


def main():
    """Main test runner."""
    print("=" * 80)
    print("IRIS_MA6 CONTROLLER TEST SUITE")
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
        run_mixer_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Mixer suite", traceback.format_exc())

    try:
        run_motor_dynamics_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Motor dynamics suite", traceback.format_exc())

    try:
        run_zoom_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Zoom suite", traceback.format_exc())

    try:
        run_gimbal_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Gimbal suite", traceback.format_exc())

    try:
        run_attitude_controller_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Attitude controller suite", traceback.format_exc())

    try:
        run_velocity_controller_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Velocity controller suite", traceback.format_exc())

    try:
        run_aerodynamics_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Aerodynamics suite", traceback.format_exc())

    try:
        run_integration_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Integration suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
