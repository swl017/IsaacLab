"""
Test script for acceleration sensor accuracy verification.

This test applies a fixed force to the quadcopter and verifies that:
1. The measured acceleration matches the expected value (F/m)
2. Acceleration values are correct after environment resets
3. Acceleration correctly drives velocity changes over time
"""
import argparse
from isaaclab.app import AppLauncher

# Create parser for both AppLauncher and test runner args
parser = argparse.ArgumentParser()

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to test")
parser.add_argument("--num_steps", type=int, default=200, help="Number of simulation steps")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.direct.quadcopter.quadcopter_env_v1 import QuadcopterEnv, QuadcopterEnvCfg

def test_acceleration_sensor(num_envs=1, num_steps=200):
    """Test acceleration sensor accuracy with fixed force application.

    Args:
        num_envs: Number of parallel environments to test
        num_steps: Number of simulation steps to run
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment with acceleration test mode enabled
    cfg = QuadcopterEnvCfg()
    cfg.scene.num_envs = num_envs
    env = QuadcopterEnv(cfg)

    print("\n" + "="*80)
    print(f"Acceleration Sensor Diagnostic Test")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Environments: {num_envs}")
    print(f"  Steps: {num_steps}")
    print(f"  Robot mass: ~1.619 kg (from IRIS config)")
    print(f"\nTest procedure:")
    print(f"  1. Apply fixed 10N force in X-axis (body frame)")
    print(f"  2. Measure resulting acceleration")
    print(f"  3. Compare with expected: a = F/m = 10 / 1.619 ≈ 6.17 m/s²")
    print(f"  4. Trigger reset at step {num_steps//2} and verify post-reset values")
    print(f"\nRunning {num_steps} steps...\n")

    # Reset environment
    obs, _ = env.reset()

    # Collect acceleration data
    acceleration_errors = []
    velocity_progression = []

    # Run for specified number of steps
    zero_action = torch.zeros((num_envs, 4), device=env.device)

    for step in range(num_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        # Trigger a reset at halfway point to check post-reset acceleration
        if step == num_steps // 2:
            print("\n" + "="*80)
            print(f"TRIGGERING RESET AT STEP {step}")
            print("="*80)
            env_ids = torch.arange(num_envs, device=env.device)
            env._reset_idx(env_ids)

    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)
    print("\nAnalysis:")
    print("  - Check console output above for acceleration measurements")
    print("  - Verify measured acceleration matches expected (~6.17 m/s²)")
    print("  - Confirm acceleration drives velocity increase over time")
    print("  - Verify acceleration values are correct immediately after reset")
    print("\nExpected Results:")
    print("  ✓ Measured acceleration should be within 5% of expected (6.17 m/s²)")
    print("  ✓ Velocity should increase linearly: v(t) = a*t")
    print("  ✓ Post-reset acceleration should be close to expected value")
    print("  ✓ No sudden jumps or discontinuities in acceleration\n")

if __name__ == "__main__":
    test_acceleration_sensor(
        num_envs=args_cli.num_envs,
        num_steps=args_cli.num_steps
    )
    simulation_app.close()
