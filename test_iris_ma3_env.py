#!/usr/bin/env python3
"""Simple test script for Isaac-Iris-MA3-Direct-v0 environment."""

import torch
import gymnasium as gym

# Import Isaac Lab
import isaaclab_tasks  # noqa: F401

def test_environment():
    """Test the Isaac-Iris-MA3-Direct-v0 environment."""
    print("=" * 80)
    print("Testing Isaac-Iris-MA3-Direct-v0 Environment")
    print("=" * 80)

    # Create the environment
    print("\n[1/4] Creating environment...")
    try:
        env = gym.make("Isaac-Iris-MA3-Direct-v0", num_envs=2, headless=True)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Reset the environment
    print("\n[2/4] Resetting environment...")
    try:
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Number of agents: {len(obs)}")
        for agent_id, agent_obs in obs.items():
            print(f"  Agent '{agent_id}' obs shape: {agent_obs.shape}")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False

    # Step through environment
    print("\n[3/4] Stepping through environment...")
    try:
        for step in range(5):
            # Sample random actions for all agents
            actions = {}
            for agent_id in env.unwrapped.cfg.possible_agents:
                action_space = env.action_space[agent_id]
                actions[agent_id] = torch.tensor(
                    action_space.sample(),
                    device=env.unwrapped.device
                ).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

            obs, rewards, terminated, truncated, info = env.step(actions)

            if step == 0:
                print(f"✓ Step {step + 1} completed")
                for agent_id, reward in rewards.items():
                    print(f"  Agent '{agent_id}' reward shape: {reward.shape}, mean: {reward.mean().item():.4f}")

        print(f"✓ Successfully completed 5 steps")
    except Exception as e:
        print(f"✗ Failed during stepping: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False

    # Close the environment
    print("\n[4/4] Closing environment...")
    try:
        env.close()
        print("✓ Environment closed successfully")
    except Exception as e:
        print(f"✗ Failed to close environment: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    return True

if __name__ == "__main__":
    import sys
    success = test_environment()
    sys.exit(0 if success else 1)
