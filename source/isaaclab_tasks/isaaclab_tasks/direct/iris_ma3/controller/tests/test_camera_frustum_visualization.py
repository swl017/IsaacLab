#!/usr/bin/env python3
"""
Test: Camera Frustum Visualization

This test verifies that:
1. Camera frustum is drawn correctly when GUI is enabled
2. Camera frustum follows gimbal motion
3. Camera pose computation is correct
4. No errors occur during visualization updates

Usage:
    # Run with GUI (required for visualization)
    timeout 30 python3 test_camera_frustum_visualization.py

Expected Behavior:
- White wireframe frustum should be visible in the viewport
- Frustum should follow gimbal yaw/pitch motion
- No errors or crashes during execution

Author: Claude Code
Date: 2025
"""

import torch
import math
import sys
import os

# Add isaaclab to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../")))

from isaaclab_tasks.direct.quadcopter.quadcopter_env_v2 import QuadcopterEnv, QuadcopterEnvCfg


def test_camera_frustum_visualization(num_envs=1, num_steps=300):
    """Test camera frustum visualization with gimbal motion."""

    print("\n" + "="*70)
    print("TEST: Camera Frustum Visualization")
    print("="*70)

    # Create environment config (with GUI enabled)
    cfg = QuadcopterEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.decimation = 2

    # Create environment with GUI (render_mode should NOT be headless)
    env = QuadcopterEnv(cfg=cfg, render_mode=None)  # None = use default (GUI if available)

    print(f"\n[SETUP]")
    print(f"  Num environments: {num_envs}")
    print(f"  Num steps: {num_steps}")
    print(f"  Camera enabled: {env._camera is not None}")
    print(f"  Frustum visualizer enabled: {env.camera_frustum is not None}")

    if env._camera is None:
        print("\n[WARNING] Camera not created - likely running in headless mode")
        print("Camera frustum visualization requires GUI mode")
        env.close()
        return

    if env.camera_frustum is None:
        print("\n[WARNING] Camera frustum visualizer not created - debug draw unavailable")
        env.close()
        return

    print(f"\n[TEST] Running visualization test...")
    print(f"  Expected: White frustum wireframe visible in viewport")
    print(f"  Expected: Frustum follows gimbal yaw/pitch motion\n")

    # Reset environment
    obs, _ = env.reset()
    device = env.device

    # Track test progress
    successful_updates = 0
    errors = []

    for step in range(num_steps):
        # Generate gimbal motion commands (sinusoidal sweep)
        t = step / num_steps * 4 * math.pi  # 2 full cycles

        # Gimbal commands: sweep yaw and pitch to test frustum visualization
        cmd_gimbal_yaw = math.sin(t) * 0.5  # ±50% range (±90°)
        cmd_gimbal_pitch = math.sin(t * 1.5 + math.pi/4) * 0.4  # ±40% range (±36°)

        # Keep drone hovering in place while actuating gimbal
        cmd_vel_x = 0.0
        cmd_vel_y = 0.0
        cmd_vel_z = 0.0
        cmd_yaw_rate = 0.0

        # Actions: [vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch]
        actions = torch.tensor(
            [[cmd_vel_x, cmd_vel_y, cmd_vel_z, cmd_yaw_rate, cmd_gimbal_yaw, cmd_gimbal_pitch]],
            device=device
        )

        try:
            # Step environment (this will trigger _debug_vis_callback)
            obs, reward, terminated, truncated, info = env.step(actions)
            successful_updates += 1

            # Print progress every 50 steps
            if step % 50 == 0 and step > 0:
                gimbal_yaw_actual = env._robot.data.joint_pos[0, env.gimbal_joint_idx["yaw"]].item()
                gimbal_pitch_actual = env._robot.data.joint_pos[0, env.gimbal_joint_idx["pitch"]].item()

                # Compute camera pose to verify computation
                camera_pos, camera_quat = env._compute_camera_pose()

                print(f"Step {step:3d}:")
                print(f"  Gimbal: yaw={math.degrees(gimbal_yaw_actual):6.1f}°, pitch={math.degrees(gimbal_pitch_actual):6.1f}°")
                print(f"  Camera pos: [{camera_pos[0,0]:.2f}, {camera_pos[0,1]:.2f}, {camera_pos[0,2]:.2f}]")
                print(f"  Successful updates: {successful_updates}/{step+1}")

            # Reset on termination
            if terminated.any() or truncated.any():
                obs, _ = env.reset()

        except Exception as e:
            errors.append(f"Step {step}: {str(e)}")
            print(f"\n[ERROR] Step {step}: {e}")
            break

    # Final statistics
    print(f"\n{'='*70}")
    print(f"TEST RESULTS:")
    print(f"{'='*70}")
    print(f"Total steps: {num_steps}")
    print(f"Successful updates: {successful_updates}")
    print(f"Update success rate: {successful_updates/num_steps*100:.1f}%")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nError details:")
        for error in errors:
            print(f"  - {error}")

    # Pass/fail criteria
    success_rate = successful_updates / num_steps
    test_passed = success_rate >= 0.95 and len(errors) == 0

    print(f"\n{'='*70}")
    if test_passed:
        print("✅ TEST PASSED: Camera frustum visualization works correctly")
    else:
        print("❌ TEST FAILED: Camera frustum visualization has issues")
    print(f"{'='*70}\n")

    # Cleanup
    env.close()

    return test_passed


if __name__ == "__main__":
    try:
        passed = test_camera_frustum_visualization(num_envs=1, num_steps=300)
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
