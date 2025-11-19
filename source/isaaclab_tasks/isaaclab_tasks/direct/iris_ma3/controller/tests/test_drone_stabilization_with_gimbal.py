#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test drone stabilization under gimbal actuation.

This test verifies that the base body controller remains stable and effective
while gimbal joints are being actuated. It checks for:
1. Attitude stability (roll/pitch should remain near level)
2. Velocity tracking accuracy
3. No adverse coupling between gimbal motion and base body control

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_drone_stabilization_with_gimbal.py \
        --num_envs 1 \
        --num_steps 500 \
        --headless
"""

from __future__ import annotations

import argparse
import torch
import math

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Test drone stabilization under gimbal actuation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps to run.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after AppLauncher
from isaaclab_tasks.direct.quadcopter.quadcopter_env_v2 import QuadcopterEnv, QuadcopterEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat


def test_drone_stabilization_with_gimbal(num_envs=1, num_steps=500):
    """Test drone stabilization while gimbal is actuating.

    This test:
    1. Commands constant forward velocity to the drone
    2. Simultaneously commands aggressive gimbal motions
    3. Measures base body attitude stability and velocity tracking
    4. Verifies no adverse coupling between gimbal and base control

    Args:
        num_envs: Number of parallel environments
        num_steps: Number of simulation steps to run
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment configuration
    cfg = QuadcopterEnvCfg()
    cfg.scene.num_envs = num_envs

    # Create environment
    print(f"\n{'='*70}")
    print(f"Creating QuadcopterEnv with {num_envs} environments...")
    print(f"{'='*70}\n")
    env = QuadcopterEnv(cfg)

    # Check if gimbal exists
    if not env.has_gimbal:
        print(f"\n{'='*70}")
        print(f"WARNING: Robot does not have gimbal joints!")
        print(f"This test will only check base body stabilization without gimbal.")
        print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print(f"Starting stabilization test for {num_steps} steps...")
    print(f"Test scenario:")
    print(f"  Base command: 1 m/s forward velocity")
    if env.has_gimbal:
        print(f"  Gimbal motion: Aggressive sinusoidal yaw/pitch (±90°/±45°)")
    print(f"Expected result:")
    print(f"  - Roll/pitch should remain near 0° (level flight)")
    print(f"  - Forward velocity should track 1 m/s")
    print(f"  - Gimbal motion should NOT disturb base body control")
    print(f"{'='*70}\n")

    # Reset environment
    obs, _ = env.reset()

    # Tracking statistics
    roll_errors = []
    pitch_errors = []
    vel_errors = []
    ang_vel_magnitudes = []

    # Target forward velocity
    target_vel = 1.0  # m/s forward

    for step in range(num_steps):
        # Base body command: constant forward velocity
        cmd_vel_x = target_vel  # 1 m/s forward
        cmd_vel_y = 0.0
        cmd_vel_z = 0.0
        cmd_yaw_rate = 0.0

        # Gimbal commands: aggressive sinusoidal motion
        if env.has_gimbal:
            t = step / num_steps * 4 * math.pi  # 2 full cycles
            cmd_gimbal_yaw = math.sin(t) * 0.8  # ±80% of max range
            cmd_gimbal_pitch = math.sin(t * 1.5 + math.pi/3) * 0.6  # ±60% of max range
        else:
            cmd_gimbal_yaw = 0.0
            cmd_gimbal_pitch = 0.0

        # Actions: [vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch]
        actions = torch.tensor([[
            cmd_vel_x,
            cmd_vel_y,
            cmd_vel_z,
            cmd_yaw_rate,
            cmd_gimbal_yaw,
            cmd_gimbal_pitch
        ]], device=device).expand(num_envs, 6)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Get base body state
        curr_quat_w = env._robot.data.root_quat_w.clone()
        curr_lin_vel_w = env._robot.data.root_lin_vel_w.clone()
        curr_ang_vel_b = env._robot.data.root_ang_vel_b.clone()

        # Compute attitude (Euler angles)
        # euler_xyz_from_quat returns a tuple: (roll, pitch, yaw)
        roll_tensor, pitch_tensor, yaw_tensor = euler_xyz_from_quat(curr_quat_w)
        roll = roll_tensor[0].item()
        pitch = pitch_tensor[0].item()
        yaw = yaw_tensor[0].item()

        # Compute velocity error
        vel_error_x = abs(curr_lin_vel_w[0, 0].item() - target_vel)

        # Track statistics
        roll_errors.append(abs(roll))
        pitch_errors.append(abs(pitch))
        vel_errors.append(vel_error_x)
        ang_vel_magnitudes.append(torch.norm(curr_ang_vel_b[0]).item())

        # Log every 50 steps
        if step % 50 == 0:
            print(f"\n=== Step {step}/{num_steps} ===")
            print(f"Base Body State:")
            print(f"  Attitude (r,p,y): [{roll*57.3:.2f}, {pitch*57.3:.2f}, {yaw*57.3:.2f}] deg")
            print(f"  Velocity (world): [{curr_lin_vel_w[0, 0].item():.3f}, {curr_lin_vel_w[0, 1].item():.3f}, {curr_lin_vel_w[0, 2].item():.3f}] m/s")
            print(f"  Angular velocity: {torch.norm(curr_ang_vel_b[0]).item():.3f} rad/s")
            print(f"Tracking Errors:")
            print(f"  Roll error: {abs(roll)*57.3:.2f}°")
            print(f"  Pitch error: {abs(pitch)*57.3:.2f}°")
            print(f"  Velocity error: {vel_error_x:.3f} m/s")

            if env.has_gimbal:
                gimbal_yaw = env._robot.data.joint_pos[:, env.gimbal_joint_idx["yaw"]][0].item()
                gimbal_pitch = env._robot.data.joint_pos[:, env.gimbal_joint_idx["pitch"]][0].item()
                print(f"Gimbal State:")
                print(f"  Yaw: {gimbal_yaw*57.3:.1f}°, Pitch: {gimbal_pitch*57.3:.1f}°")

    # Compute statistics (ignore first 100 steps for settling)
    settling_steps = min(100, num_steps // 4)
    roll_errors_settled = roll_errors[settling_steps:]
    pitch_errors_settled = pitch_errors[settling_steps:]
    vel_errors_settled = vel_errors[settling_steps:]
    ang_vel_settled = ang_vel_magnitudes[settling_steps:]

    mean_roll_error = sum(roll_errors_settled) / len(roll_errors_settled)
    max_roll_error = max(roll_errors_settled)
    mean_pitch_error = sum(pitch_errors_settled) / len(pitch_errors_settled)
    max_pitch_error = max(pitch_errors_settled)
    mean_vel_error = sum(vel_errors_settled) / len(vel_errors_settled)
    max_vel_error = max(vel_errors_settled)
    mean_ang_vel = sum(ang_vel_settled) / len(ang_vel_settled)
    max_ang_vel = max(ang_vel_settled)

    # Print final results
    print(f"\n{'='*70}")
    print(f"DRONE STABILIZATION TEST RESULTS (after {settling_steps} settling steps)")
    print(f"{'='*70}")
    print(f"Attitude Stability (should be near 0°):")
    print(f"  Mean roll error:  {mean_roll_error*57.3:.2f}° (max: {max_roll_error*57.3:.2f}°)")
    print(f"  Mean pitch error: {mean_pitch_error*57.3:.2f}° (max: {max_pitch_error*57.3:.2f}°)")
    print(f"Velocity Tracking (target: {target_vel} m/s forward):")
    print(f"  Mean error: {mean_vel_error:.3f} m/s (max: {max_vel_error:.3f} m/s)")
    print(f"Angular Velocity (should be low for stable flight):")
    print(f"  Mean: {mean_ang_vel:.3f} rad/s (max: {max_ang_vel:.3f} rad/s)")
    print(f"{'='*70}")

    # Pass/Fail criteria
    attitude_pass = (mean_roll_error < 0.1) and (mean_pitch_error < 0.1)  # < 5.7° mean error
    velocity_pass = mean_vel_error < 0.3  # < 0.3 m/s mean error
    stability_pass = mean_ang_vel < 1.0  # < 1.0 rad/s mean angular velocity

    if attitude_pass and velocity_pass and stability_pass:
        print(f"\n✓ TEST PASSED: Drone remains stable under gimbal actuation")
        print(f"  Base body control is NOT adversely affected by gimbal motion")
    else:
        print(f"\n✗ TEST FAILED:")
        if not attitude_pass:
            print(f"  - Attitude instability detected:")
            print(f"    Mean roll: {mean_roll_error*57.3:.2f}°, Mean pitch: {mean_pitch_error*57.3:.2f}°")
        if not velocity_pass:
            print(f"  - Velocity tracking error too high: {mean_vel_error:.3f} m/s")
        if not stability_pass:
            print(f"  - Angular velocity too high: {mean_ang_vel:.3f} rad/s")
        print(f"  Gimbal motion may be coupling into base body control!")

    print(f"{'='*70}\n")

    # Close environment
    env.close()


if __name__ == "__main__":
    # Run test
    test_drone_stabilization_with_gimbal(
        num_envs=args_cli.num_envs,
        num_steps=args_cli.num_steps
    )

    # Close simulation
    simulation_app.close()
