#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test gimbal joint actuation.

This test verifies that gimbal joints (yaw, pitch, roll) respond correctly to position commands.
It applies sinusoidal position targets and measures tracking accuracy.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_gimbal_actuation.py \
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
parser = argparse.ArgumentParser(description="Test gimbal joint actuation")
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


def test_gimbal_actuation(num_envs=1, num_steps=500):
    """Test gimbal joint actuation with sinusoidal position commands.

    This test:
    1. Applies sinusoidal position commands to gimbal joints
    2. Measures tracking error between commanded and actual joint positions
    3. Verifies gimbal servos are functioning correctly

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
        print(f"ERROR: Robot does not have gimbal joints!")
        print(f"This test requires a robot with yaw_joint, pitch_joint, and roll_joint.")
        print(f"Current robot config: {cfg.robot}")
        print(f"{'='*70}\n")
        env.close()
        return

    print(f"\n{'='*70}")
    print(f"Starting gimbal actuation test for {num_steps} steps...")
    print(f"Test parameters:")
    print(f"  Yaw amplitude: ±90° (±{math.pi/2:.3f} rad)")
    print(f"  Pitch amplitude: ±45° (±{math.pi/4:.3f} rad)")
    print(f"  Frequency: {2*math.pi/num_steps:.4f} rad/step")
    print(f"{'='*70}\n")

    # Reset environment
    obs, _ = env.reset()

    # Tracking error statistics
    yaw_errors = []
    pitch_errors = []
    roll_errors = []

    for step in range(num_steps):
        # Generate sinusoidal gimbal commands
        # Yaw: ±90° (±π/2 rad), frequency: 1 cycle over num_steps
        # Pitch: ±45° (±π/4 rad), frequency: 1 cycle over num_steps
        t = step / num_steps * 2 * math.pi

        cmd_gimbal_yaw = math.sin(t) * 0.5  # Range: [-0.5, 0.5] -> [-90°, 90°] after scaling
        cmd_gimbal_pitch = math.sin(t + math.pi/4) * 0.25  # Range: [-0.25, 0.25] -> [-45°, 45°]

        # Actions: [vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch]
        # Keep base body commands at zero (hover)
        actions = torch.zeros(num_envs, 6, device=device)
        actions[:, 4] = cmd_gimbal_yaw  # Gimbal yaw command
        actions[:, 5] = cmd_gimbal_pitch  # Gimbal pitch command

        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Get current gimbal joint positions
        gimbal_yaw_actual = env._robot.data.joint_pos[:, env.gimbal_joint_idx["yaw"]].clone()
        gimbal_pitch_actual = env._robot.data.joint_pos[:, env.gimbal_joint_idx["pitch"]].clone()
        gimbal_roll_actual = env._robot.data.joint_pos[:, env.gimbal_joint_idx["roll"]].clone()

        # Compute expected positions (after action scaling)
        gimbal_yaw_target = cmd_gimbal_yaw * cfg.max_gimbal_yaw_angle
        gimbal_pitch_target = cmd_gimbal_pitch * cfg.max_gimbal_pitch_angle

        # Compute tracking errors
        yaw_error = (gimbal_yaw_actual[0] - gimbal_yaw_target).abs().item()
        pitch_error = (gimbal_pitch_actual[0] - gimbal_pitch_target).abs().item()

        yaw_errors.append(yaw_error)
        pitch_errors.append(pitch_error)
        roll_errors.append(gimbal_roll_actual[0].abs().item())  # Roll should stay near 0 (stabilized)

        # Log every 50 steps
        if step % 50 == 0:
            print(f"\n=== Step {step}/{num_steps} ===")
            print(f"Gimbal Commands (normalized):")
            print(f"  Yaw: {cmd_gimbal_yaw:.3f}, Pitch: {cmd_gimbal_pitch:.3f}")
            print(f"Gimbal Targets (rad):")
            print(f"  Yaw: {gimbal_yaw_target:.3f}, Pitch: {gimbal_pitch_target:.3f}")
            print(f"Gimbal Actual (rad):")
            print(f"  Yaw: {gimbal_yaw_actual[0].item():.3f}, Pitch: {gimbal_pitch_actual[0].item():.3f}, Roll: {gimbal_roll_actual[0].item():.3f}")
            print(f"Tracking Errors (rad):")
            print(f"  Yaw: {yaw_error:.4f} ({yaw_error*57.3:.2f}°)")
            print(f"  Pitch: {pitch_error:.4f} ({pitch_error*57.3:.2f}°)")
            print(f"  Roll deviation: {gimbal_roll_actual[0].abs().item():.4f} ({gimbal_roll_actual[0].abs().item()*57.3:.2f}°)")

    # Compute statistics (ignore first 100 steps for settling)
    settling_steps = min(100, num_steps // 4)
    yaw_errors_settled = yaw_errors[settling_steps:]
    pitch_errors_settled = pitch_errors[settling_steps:]
    roll_errors_settled = roll_errors[settling_steps:]

    mean_yaw_error = sum(yaw_errors_settled) / len(yaw_errors_settled)
    max_yaw_error = max(yaw_errors_settled)
    mean_pitch_error = sum(pitch_errors_settled) / len(pitch_errors_settled)
    max_pitch_error = max(pitch_errors_settled)
    mean_roll_dev = sum(roll_errors_settled) / len(roll_errors_settled)
    max_roll_dev = max(roll_errors_settled)

    # Print final results
    print(f"\n{'='*70}")
    print(f"GIMBAL ACTUATION TEST RESULTS (after {settling_steps} settling steps)")
    print(f"{'='*70}")
    print(f"Yaw Tracking:")
    print(f"  Mean error: {mean_yaw_error:.4f} rad ({mean_yaw_error*57.3:.2f}°)")
    print(f"  Max error:  {max_yaw_error:.4f} rad ({max_yaw_error*57.3:.2f}°)")
    print(f"Pitch Tracking:")
    print(f"  Mean error: {mean_pitch_error:.4f} rad ({mean_pitch_error*57.3:.2f}°)")
    print(f"  Max error:  {max_pitch_error:.4f} rad ({max_pitch_error*57.3:.2f}°)")
    print(f"Roll Stabilization (should be near 0):")
    print(f"  Mean deviation: {mean_roll_dev:.4f} rad ({mean_roll_dev*57.3:.2f}°)")
    print(f"  Max deviation:  {max_roll_dev:.4f} rad ({max_roll_dev*57.3:.2f}°)")
    print(f"{'='*70}")

    # Pass/Fail criteria
    yaw_pass = mean_yaw_error < 0.1  # < 5.7° mean error
    pitch_pass = mean_pitch_error < 0.1  # < 5.7° mean error
    roll_pass = mean_roll_dev < 0.2  # < 11.5° mean deviation

    if yaw_pass and pitch_pass and roll_pass:
        print(f"\n✓ TEST PASSED: Gimbal actuation is working correctly")
    else:
        print(f"\n✗ TEST FAILED:")
        if not yaw_pass:
            print(f"  - Yaw tracking error too high: {mean_yaw_error:.4f} rad")
        if not pitch_pass:
            print(f"  - Pitch tracking error too high: {mean_pitch_error:.4f} rad")
        if not roll_pass:
            print(f"  - Roll stabilization failed: {mean_roll_dev:.4f} rad deviation")

    print(f"{'='*70}\n")

    # Close environment
    env.close()


if __name__ == "__main__":
    # Run test
    test_gimbal_actuation(
        num_envs=args_cli.num_envs,
        num_steps=args_cli.num_steps
    )

    # Close simulation
    simulation_app.close()
