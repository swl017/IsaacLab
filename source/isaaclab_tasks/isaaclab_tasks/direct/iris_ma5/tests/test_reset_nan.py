#!/usr/bin/env python3
"""Test script to verify NaN fix after reset in IrisMA env."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test reset NaN fix")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

# Import to register the environment
import isaaclab_tasks.direct.iris_ma3  # noqa: F401


def test_reset_nan():
    """Test that reset doesn't cause NaN in observations."""
    print("=" * 80)
    print("Testing IrisMA Reset NaN Fix")
    print("=" * 80)

    # Create environment
    print("\n1. Creating environment...")
    from isaaclab_tasks.utils import load_cfg_from_registry
    env_cfg = load_cfg_from_registry("Isaac-Iris-MA3-Direct-v0", "env_cfg_entry_point")
    env_cfg.scene.num_envs = 2
    env = gym.make("Isaac-Iris-MA3-Direct-v0", cfg=env_cfg)

    print(f"  Environment created with {env_cfg.scene.num_envs} envs")

    # Initial reset
    print("\n2. Initial reset...")
    obs, info = env.reset()

    # Check initial observations for NaN
    print("\n3. Checking initial observations for NaN...")
    has_nan = False
    for agent_id, agent_obs in obs.items():
        if torch.isnan(agent_obs).any():
            nan_count = torch.isnan(agent_obs).sum().item()
            print(f"  ✗ {agent_id}: NaN detected ({nan_count} values)")
            has_nan = True
        else:
            print(f"  ✓ {agent_id}: No NaN detected")

    if has_nan:
        print("\n✗ FAILED: NaN detected in initial observations!")
        env.close()
        simulation_app.close()
        return False

    # Run a few steps
    print("\n4. Running 10 steps...")
    for step in range(10):
        # Create dummy actions (zero actions) - 7 dims: [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        actions = {}
        for agent_id in env.unwrapped.cfg.possible_agents:
            actions[agent_id] = torch.zeros(env_cfg.scene.num_envs, 7, device=env.unwrapped.device)

        obs, reward, terminated, truncated, info = env.step(actions)

        # Check for NaN in observations
        for agent_id, agent_obs in obs.items():
            if torch.isnan(agent_obs).any():
                nan_count = torch.isnan(agent_obs).sum().item()
                print(f"  Step {step}: ✗ {agent_id}: NaN detected ({nan_count} values)")
                has_nan = True
                break

        if has_nan:
            break

    if has_nan:
        print(f"\n✗ FAILED: NaN detected during simulation!")
        env.close()
        simulation_app.close()
        return False

    print("  All 10 steps completed without NaN")

    # Force a reset of specific environments
    print("\n5. Testing manual reset of env 0...")
    env.unwrapped._reset_idx(torch.tensor([0], device=env.unwrapped.device))

    # Get observations after manual reset
    obs = env.unwrapped._get_observations()

    print("\n6. Checking observations after manual reset...")
    for agent_id, agent_obs in obs.items():
        if torch.isnan(agent_obs).any():
            nan_count = torch.isnan(agent_obs).sum().item()
            print(f"  ✗ {agent_id}: NaN detected ({nan_count} values)")
            # Print which indices have NaN
            nan_indices = torch.nonzero(torch.isnan(agent_obs))
            print(f"    NaN indices (first 10): {nan_indices[:10].tolist()}")
            has_nan = True
        else:
            print(f"  ✓ {agent_id}: No NaN detected")

    if has_nan:
        print("\n✗ FAILED: NaN detected after manual reset!")
        env.close()
        simulation_app.close()
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - No NaN detected!")
    print("=" * 80)

    env.close()
    simulation_app.close()
    return True


if __name__ == "__main__":
    import sys
    success = test_reset_nan()
    sys.exit(0 if success else 1)
