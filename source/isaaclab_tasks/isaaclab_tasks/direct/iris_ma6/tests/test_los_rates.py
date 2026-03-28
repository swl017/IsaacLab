#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test LOS rate computation: spherical decomposition formulas and comparison
with true gimbal Jacobian kinematics (what body_ang_vel_w for pitch_link gives).

Tests verify that:
1. Spherical decomposition correctly converts omega_w -> (az_rate, el_rate)
2. compute_combined_angular_velocity matches true gimbal Jacobian result
3. End-to-end: body_rate + joint_rates -> LOS rates matches ground truth
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test LOS rate computation")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import sys
import traceback
from datetime import datetime

import torch

from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

try:
    from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller import YAW_JOINT_OFFSET
except ImportError:
    from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller_analytical import YAW_JOINT_OFFSET

from isaaclab_tasks.direct.iris_ma6.delay_system_v3.derived_field_computers import (
    compute_combined_angular_velocity,
)

VERBOSE = args_cli.test_verbose


class TestResults:
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
        for line in error.split("\n")[:8]:
            print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        for line in error.split("\n")[:5]:
            print(f"    {line}")

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


# ─── Helper functions ────────────────────────────────────────────────────────


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Create wxyz quaternion from Euler angles (intrinsic XYZ)."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def spherical_rates_from_omega(
    omega_w: torch.Tensor, az: torch.Tensor, el: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute azimuth/elevation rates from angular velocity (the formulas under test).

    For a ray direction d = [cos(el)*cos(az), cos(el)*sin(az), sin(el)]:
        d/dt(az) = omega_z - tan(el) * (omega_x * cos(az) + omega_y * sin(az))
        d/dt(el) = omega_x * sin(az) - omega_y * cos(az)
    """
    az_rate = omega_w[:, 2] - torch.tan(el) * (
        omega_w[:, 0] * torch.cos(az) + omega_w[:, 1] * torch.sin(az)
    )
    el_rate = omega_w[:, 0] * torch.sin(az) - omega_w[:, 1] * torch.cos(az)
    return az_rate, el_rate


def ray_direction_from_az_el(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """(N,) az, el -> (N, 3) unit ray direction."""
    cos_el = torch.cos(el)
    return torch.stack([cos_el * torch.cos(az), cos_el * torch.sin(az), torch.sin(el)], dim=-1)


def az_el_from_direction(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 3) direction -> (N,) az, (N,) el."""
    az = torch.atan2(d[:, 1], d[:, 0])
    el = torch.atan2(d[:, 2], torch.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2))
    return az, el


def gimbal_jacobian_body(yaw: float, roll: float) -> torch.Tensor:
    """True gimbal Jacobian for ZXY (yaw->roll->pitch) chain.

    Maps [pitch_rate, yaw_rate, roll_rate] -> omega_gimbal in body frame.

    The gimbal chain is:
      yaw around body Z -> roll around rotated X -> pitch around further-rotated Y

    omega_gimbal_body = yaw_rate * [0, 0, 1]
                      + roll_rate * R_z(yaw) @ [1, 0, 0]
                      + pitch_rate * R_z(yaw) @ R_x(roll) @ [0, 1, 0]
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    # Columns: [pitch_rate, yaw_rate, roll_rate]
    J = torch.tensor([
        [-sy * cr,  0.0,  cy],   # X
        [ cy * cr,  0.0,  sy],   # Y
        [ sr,       1.0,  0.0],  # Z
    ], dtype=torch.float64)
    return J  # (3, 3)


# ─── Test suites ─────��───────────────────────────────────────────────────────


def run_spherical_decomposition_tests(results: TestResults, device: torch.device):
    """Test that the spherical rate formulas correctly decompose omega_w into (az_rate, el_rate)."""
    print("\n" + "=" * 80)
    print("Test 1: Spherical Decomposition Formula Correctness")
    print("=" * 80)

    N = 64
    dt = 1e-6  # tiny dt for numerical differentiation

    # Test cases: (description, az_values, el_values, omega_values)
    test_cases = [
        ("Small angles, pure yaw rotation (omega_z only)",
         torch.zeros(N, device=device),
         torch.zeros(N, device=device),
         torch.stack([torch.zeros(N, device=device),
                      torch.zeros(N, device=device),
                      torch.ones(N, device=device) * 0.5], dim=-1)),

        ("Small angles, pure pitch rotation (omega_y only)",
         torch.zeros(N, device=device),
         torch.zeros(N, device=device),
         torch.stack([torch.zeros(N, device=device),
                      -torch.ones(N, device=device) * 0.3,
                      torch.zeros(N, device=device)], dim=-1)),

        ("Non-zero az/el, mixed omega",
         torch.linspace(-2.5, 2.5, N, device=device),
         torch.linspace(-1.0, 1.0, N, device=device),
         torch.randn(N, 3, device=device) * 0.5),

        ("Near-forward direction with random omega",
         torch.randn(N, device=device) * 0.3,
         torch.randn(N, device=device) * 0.2,
         torch.randn(N, 3, device=device)),

        ("Large azimuth range (near +-pi)",
         torch.linspace(-3.0, 3.0, N, device=device),
         torch.zeros(N, device=device),
         torch.randn(N, 3, device=device) * 0.3),
    ]

    for desc, az, el, omega_w in test_cases:
        try:
            # Compute rates using the formula under test
            az_rate, el_rate = spherical_rates_from_omega(omega_w, az, el)

            # Numerical verification: perturb ray by omega and check az/el change
            d = ray_direction_from_az_el(az, el)
            # d_dot = omega x d
            d_dot = torch.cross(omega_w, d, dim=-1)
            d_new = d + d_dot * dt
            d_new = d_new / d_new.norm(dim=-1, keepdim=True)

            az_new, el_new = az_el_from_direction(d_new)
            # Handle azimuth wrapping
            az_diff = az_new - az
            az_diff = torch.remainder(az_diff + torch.pi, 2 * torch.pi) - torch.pi
            numerical_az_rate = az_diff / dt
            numerical_el_rate = (el_new - el) / dt

            az_err = (az_rate - numerical_az_rate).abs().max().item()
            el_err = (el_rate - numerical_el_rate).abs().max().item()

            if VERBOSE:
                print(f"    az_rate err: {az_err:.2e}, el_rate err: {el_err:.2e}")

            # Tolerance accounts for numerical differentiation error (O(dt))
            assert az_err < 1.0, f"az_rate max error {az_err:.4e} exceeds tolerance"
            assert el_err < 1.0, f"el_rate max error {el_err:.4e} exceeds tolerance"
            results.add_pass(f"Spherical decomposition: {desc}")
        except Exception as e:
            results.add_fail(f"Spherical decomposition: {desc}", traceback.format_exc())


def run_jacobian_comparison_tests(results: TestResults, device: torch.device):
    """Compare compute_combined_angular_velocity (simplified) vs true gimbal Jacobian.

    This test exposes that the simplified model (pitch->body Y, yaw->body Z) diverges
    from the true gimbal kinematics, especially at the nominal YAW_JOINT_OFFSET = -pi/2.
    """
    print("\n" + "=" * 80)
    print("Test 2: Simplified Model vs True Gimbal Jacobian")
    print("=" * 80)
    print(f"  YAW_JOINT_OFFSET = {YAW_JOINT_OFFSET:.4f} rad ({math.degrees(YAW_JOINT_OFFSET):.1f} deg)")

    N = 1  # single env for clarity

    # Scenario: body at identity orientation, gimbal at various yaw angles
    q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # identity
    body_ang_vel_b = torch.zeros(N, 3, device=device)  # stationary body
    body_ang_vel_w = torch.zeros(N, 3, device=device)

    # Test with pure pitch rate at different gimbal yaw angles
    pitch_rate = 1.0  # rad/s

    test_angles = [
        ("Logical yaw=0 (physical yaw=-pi/2, nominal forward)",
         0.0, YAW_JOINT_OFFSET),
        ("Logical yaw=pi/4 (physical yaw=-pi/4)",
         math.pi / 4, math.pi / 4 + YAW_JOINT_OFFSET),
        ("Logical yaw=0, physical yaw=0 (if offset were 0)",
         None, 0.0),
        ("Physical yaw=-pi/3",
         None, -math.pi / 3),
    ]

    for desc, logical_yaw, physical_yaw in test_angles:
        try:
            # Joint positions: [pitch, yaw, roll] - physical joint angles
            joint_pos = torch.tensor([[0.0, physical_yaw, 0.0]], device=device)
            # Pure pitch rate: [pitch_rate, 0, 0]
            joint_vel = torch.tensor([[pitch_rate, 0.0, 0.0]], device=device)

            # --- Simplified model (from compute_combined_angular_velocity) ---
            combined_w_simplified, combined_b_simplified = compute_combined_angular_velocity(
                body_angular_velocity_w=body_ang_vel_w,
                body_angular_velocity_b=body_ang_vel_b,
                joint_velocities_b=joint_vel,
                body_orientation_w=q_body,
                joint_positions_b=joint_pos,
            )

            # --- True gimbal Jacobian ---
            J = gimbal_jacobian_body(physical_yaw, 0.0).to(device=device, dtype=torch.float32)
            # J maps [pitch_rate, yaw_rate, roll_rate] -> omega_gimbal_body
            joint_vel_vec = torch.tensor([pitch_rate, 0.0, 0.0], device=device)
            true_omega_gimbal_b = J @ joint_vel_vec
            true_combined_b = body_ang_vel_b[0] + true_omega_gimbal_b
            # With identity body orientation, world = body
            true_combined_w = true_combined_b.unsqueeze(0)

            error = (combined_w_simplified[0] - true_combined_w[0]).abs()
            max_err = error.max().item()

            print(f"\n  {desc}:")
            print(f"    Physical yaw: {physical_yaw:.4f} rad ({math.degrees(physical_yaw):.1f} deg)")
            print(f"    Simplified omega_w: [{combined_w_simplified[0, 0]:.4f}, {combined_w_simplified[0, 1]:.4f}, {combined_w_simplified[0, 2]:.4f}]")
            print(f"    True omega_w:       [{true_combined_w[0, 0]:.4f}, {true_combined_w[0, 1]:.4f}, {true_combined_w[0, 2]:.4f}]")
            print(f"    Error:              [{error[0]:.4f}, {error[1]:.4f}, {error[2]:.4f}]  max={max_err:.4f}")

            # This is a diagnostic test - we EXPECT errors at large yaw angles
            # Record whether the simplified model matches at this angle
            if max_err < 0.01:
                results.add_pass(f"Jacobian match: {desc} (error={max_err:.4e})")
            else:
                results.add_fail(
                    f"Jacobian match: {desc}",
                    f"Simplified model diverges from true Jacobian.\n"
                    f"Max error: {max_err:.4f} rad/s\n"
                    f"Simplified: {combined_w_simplified[0].tolist()}\n"
                    f"True:       {true_combined_w[0].tolist()}\n"
                    f"This shows compute_combined_angular_velocity needs the gimbal Jacobian."
                )
        except Exception as e:
            results.add_error(f"Jacobian comparison: {desc}", traceback.format_exc())


def run_end_to_end_los_rate_tests(results: TestResults, device: torch.device):
    """End-to-end test: body_rate + joint_rates -> combined_w -> LOS rates.

    Compares the full pipeline using the simplified model vs true Jacobian,
    showing the downstream effect on LOS rate accuracy.
    """
    print("\n" + "=" * 80)
    print("Test 3: End-to-End LOS Rate Pipeline")
    print("=" * 80)

    N = 16

    # Random body orientations (small tilts)
    roll = torch.randn(N, device=device) * 0.1
    pitch = torch.randn(N, device=device) * 0.1
    yaw = torch.randn(N, device=device) * 0.3
    q_body = quat_from_euler_xyz(roll, pitch, yaw).to(device)

    # Body angular velocity (small)
    body_ang_vel_b = torch.randn(N, 3, device=device) * 0.2
    body_ang_vel_w = quat_rotate(q_body, body_ang_vel_b)

    # Gimbal at nominal forward: physical yaw = YAW_JOINT_OFFSET, pitch varies
    gimbal_pitch = torch.randn(N, device=device) * 0.3
    gimbal_physical_yaw = torch.full((N,), YAW_JOINT_OFFSET, device=device) + torch.randn(N, device=device) * 0.2
    gimbal_roll = torch.zeros(N, device=device)
    joint_pos = torch.stack([gimbal_pitch, gimbal_physical_yaw, gimbal_roll], dim=-1)

    # Joint velocities
    joint_vel = torch.randn(N, 3, device=device) * 0.5

    # --- Simplified model pipeline ---
    combined_w_simp, _ = compute_combined_angular_velocity(
        body_angular_velocity_w=body_ang_vel_w,
        body_angular_velocity_b=body_ang_vel_b,
        joint_velocities_b=joint_vel,
        body_orientation_w=q_body,
        joint_positions_b=joint_pos,
    )

    # --- True Jacobian pipeline ---
    true_combined_w_list = []
    for i in range(N):
        J = gimbal_jacobian_body(
            gimbal_physical_yaw[i].item(), gimbal_roll[i].item()
        ).to(device=device, dtype=torch.float32)
        omega_gimbal_b = J @ joint_vel[i]
        true_combined_b = body_ang_vel_b[i] + omega_gimbal_b
        true_combined_w = quat_rotate(q_body[i:i+1], true_combined_b.unsqueeze(0))
        true_combined_w_list.append(true_combined_w[0])
    true_combined_w = torch.stack(true_combined_w_list)

    # Compute LOS angles (same for both since they use joint_pos, not velocities)
    logical_yaw = gimbal_physical_yaw - YAW_JOINT_OFFSET
    cos_p = torch.cos(gimbal_pitch)
    dir_body = torch.stack([
        cos_p * torch.cos(logical_yaw),
        cos_p * torch.sin(logical_yaw),
        -torch.sin(gimbal_pitch),
    ], dim=-1)
    dir_world = quat_rotate(q_body, dir_body)
    az = torch.atan2(dir_world[:, 1], dir_world[:, 0])
    el = torch.atan2(dir_world[:, 2], torch.sqrt(dir_world[:, 0]**2 + dir_world[:, 1]**2))

    # LOS rates from simplified
    az_rate_simp, el_rate_simp = spherical_rates_from_omega(combined_w_simp, az, el)
    # LOS rates from true Jacobian
    az_rate_true, el_rate_true = spherical_rates_from_omega(true_combined_w, az, el)

    # --- Numerical ground truth (finite difference with true Jacobian omega) ---
    dt = 1e-5
    d = ray_direction_from_az_el(az, el)
    d_dot = torch.cross(true_combined_w, d, dim=-1)
    d_new = d + d_dot * dt
    d_new = d_new / d_new.norm(dim=-1, keepdim=True)
    az_new, el_new = az_el_from_direction(d_new)
    az_diff = torch.remainder(az_new - az + torch.pi, 2 * torch.pi) - torch.pi
    az_rate_numerical = az_diff / dt
    el_rate_numerical = (el_new - el) / dt

    # Report errors
    simp_az_err = (az_rate_simp - az_rate_numerical).abs()
    simp_el_err = (el_rate_simp - el_rate_numerical).abs()
    true_az_err = (az_rate_true - az_rate_numerical).abs()
    true_el_err = (el_rate_true - el_rate_numerical).abs()

    print(f"\n  Simplified model LOS rate errors (vs numerical ground truth):")
    print(f"    az_rate: mean={simp_az_err.mean():.4f}, max={simp_az_err.max():.4f} rad/s")
    print(f"    el_rate: mean={simp_el_err.mean():.4f}, max={simp_el_err.max():.4f} rad/s")
    print(f"\n  True Jacobian LOS rate errors (vs numerical ground truth):")
    print(f"    az_rate: mean={true_az_err.mean():.6f}, max={true_az_err.max():.6f} rad/s")
    print(f"    el_rate: mean={true_el_err.mean():.6f}, max={true_el_err.max():.6f} rad/s")

    omega_err = (combined_w_simp - true_combined_w).abs()
    print(f"\n  Combined omega_w error (simplified vs true):")
    print(f"    mean={omega_err.mean():.4f}, max={omega_err.max():.4f} rad/s")

    # Test: true Jacobian pipeline should match numerical ground truth closely
    try:
        assert true_az_err.max().item() < 0.1, f"True Jacobian az_rate error too large: {true_az_err.max():.4e}"
        assert true_el_err.max().item() < 0.1, f"True Jacobian el_rate error too large: {true_el_err.max():.4e}"
        results.add_pass("True Jacobian LOS rates match numerical ground truth")
    except Exception as e:
        results.add_fail("True Jacobian LOS rates match numerical ground truth", str(e))

    # Test: quantify simplified model error
    try:
        simp_max = max(simp_az_err.max().item(), simp_el_err.max().item())
        true_max = max(true_az_err.max().item(), true_el_err.max().item())
        print(f"\n  Simplified max LOS rate error: {simp_max:.4f} rad/s")
        print(f"  True Jacobian max LOS rate error: {true_max:.6f} rad/s")
        print(f"  Ratio: {simp_max / max(true_max, 1e-10):.1f}x worse")
        if simp_max > 0.05:
            results.add_fail(
                "Simplified model LOS rate accuracy",
                f"Max error {simp_max:.4f} rad/s is significant.\n"
                f"compute_combined_angular_velocity needs the gimbal Jacobian\n"
                f"to be accurate at YAW_JOINT_OFFSET = {YAW_JOINT_OFFSET:.2f}"
            )
        else:
            results.add_pass(f"Simplified model LOS rate accuracy (max err={simp_max:.4f})")
    except Exception as e:
        results.add_error("Simplified model accuracy assessment", traceback.format_exc())


def run_body_ang_vel_reference_test(results: TestResults, device: torch.device):
    """Verify that the true Jacobian matches what body_ang_vel_w would give.

    Since we can't run physics here, we verify the Jacobian math is self-consistent:
    given omega_camera_body from the Jacobian, rotating it to world frame should be
    the correct pitch-link angular velocity that Isaac Sim's body_ang_vel_w would report.
    """
    print("\n" + "=" * 80)
    print("Test 4: Jacobian Self-Consistency (body_ang_vel_w Reference)")
    print("=" * 80)

    N = 1
    device_d = torch.device("cpu")

    # Known scenario: body rotating + gimbal moving
    q_body = quat_from_euler_xyz(
        torch.tensor([0.05]), torch.tensor([0.1]), torch.tensor([0.3])
    ).to(device_d)
    body_ang_vel_b = torch.tensor([[0.1, -0.05, 0.2]], device=device_d)

    # Gimbal at nominal forward with some pitch
    physical_yaw = YAW_JOINT_OFFSET + 0.1
    physical_roll = 0.02
    physical_pitch = -0.15
    joint_vel = torch.tensor([[0.3, -0.1, 0.05]], device=device_d)  # pitch_rate, yaw_rate, roll_rate

    J = gimbal_jacobian_body(physical_yaw, physical_roll).float()
    omega_gimbal_b = J @ joint_vel[0]
    expected_camera_omega_b = body_ang_vel_b[0] + omega_gimbal_b
    expected_camera_omega_w = quat_rotate(q_body, expected_camera_omega_b.unsqueeze(0))

    print(f"  Body ang vel (body):    {body_ang_vel_b[0].tolist()}")
    print(f"  Gimbal omega (body):    {omega_gimbal_b.tolist()}")
    print(f"  Camera omega (body):    {expected_camera_omega_b.tolist()}")
    print(f"  Camera omega (world):   {expected_camera_omega_w[0].tolist()}")
    print(f"  ^ This is what robot.data.body_ang_vel_w[:, pitch_link_idx] should give")

    # Compare with simplified model
    combined_w_simp, combined_b_simp = compute_combined_angular_velocity(
        body_angular_velocity_w=quat_rotate(q_body, body_ang_vel_b),
        body_angular_velocity_b=body_ang_vel_b,
        joint_velocities_b=joint_vel,
        body_orientation_w=q_body,
        joint_positions_b=torch.tensor([[physical_pitch, physical_yaw, physical_roll]], device=device_d),
    )

    body_err = (combined_b_simp[0] - expected_camera_omega_b).abs()
    world_err = (combined_w_simp[0] - expected_camera_omega_w[0]).abs()

    print(f"\n  Simplified camera omega (body):  {combined_b_simp[0].tolist()}")
    print(f"  Simplified camera omega (world): {combined_w_simp[0].tolist()}")
    print(f"  Body-frame error:  {body_err.tolist()}  max={body_err.max():.4f}")
    print(f"  World-frame error: {world_err.tolist()}  max={world_err.max():.4f}")

    try:
        assert world_err.max().item() < 0.01, (
            f"Simplified model diverges from true Jacobian by {world_err.max():.4f} rad/s"
        )
        results.add_pass("Simplified model matches body_ang_vel_w reference")
    except AssertionError:
        results.add_fail(
            "Simplified model matches body_ang_vel_w reference",
            f"World-frame error: {world_err.tolist()}, max={world_err.max():.4f}\n"
            f"The simplified axis assumption (pitch->Y, yaw->Z) is wrong at\n"
            f"physical_yaw={physical_yaw:.2f} ({math.degrees(physical_yaw):.0f} deg).\n"
            f"compute_combined_angular_velocity must use the gimbal Jacobian."
        )


def main():
    import io
    import os

    # Redirect stdout to a tee: both file and original stdout
    log_path = os.path.join(os.path.dirname(__file__), "test_result.txt")
    log_file = open(log_path, "w")
    original_stdout = sys.stdout

    class Tee:
        def __init__(self, *targets):
            self.targets = targets
        def write(self, data):
            for t in self.targets:
                t.write(data)
                t.flush()
        def flush(self):
            for t in self.targets:
                t.flush()

    sys.stdout = Tee(original_stdout, log_file)

    print("=" * 80)
    print("LOS RATE COMPUTATION TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()

    try:
        run_spherical_decomposition_tests(results, device)
    except Exception as e:
        results.add_error("Spherical decomposition suite", traceback.format_exc())

    try:
        run_jacobian_comparison_tests(results, device)
    except Exception as e:
        results.add_error("Jacobian comparison suite", traceback.format_exc())

    try:
        run_end_to_end_los_rate_tests(results, device)
    except Exception as e:
        results.add_error("End-to-end LOS rate suite", traceback.format_exc())

    try:
        run_body_ang_vel_reference_test(results, device)
    except Exception as e:
        results.add_error("body_ang_vel_w reference suite", traceback.format_exc())

    success = results.print_summary()
    log_file.close()
    sys.stdout = original_stdout
    print(f"\nResults written to: {log_path}")
    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
