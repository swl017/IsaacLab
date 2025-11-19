"""
Test script for random formation generator.
"""
import argparse
from isaaclab.app import AppLauncher

# Create parser for both AppLauncher and test runner args
parser = argparse.ArgumentParser()

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from initial_states import InitialStatesRandomizer, InitialStatesRandomizerCfg

def test_formation_generator():
    """Test the formation generator with various configurations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize randomizer
    num_envs = 4
    num_agents = 3
    randomizer = InitialStatesRandomizer(num_envs, device)

    print(f"\nTesting with {num_envs} environments and {num_agents} agents per environment")
    print("=" * 70)

    # Test 1: Random formation type selection
    print("\n[Test 1] Random formation type selection")
    root_states = randomizer.get_random_formation(num_agents)
    print(f"Output shape: {root_states.shape}")
    print(f"Expected: torch.Size([{num_envs}, {num_agents}, 13])")
    assert root_states.shape == (num_envs, num_agents, 13), "Shape mismatch!"
    print("✓ Shape correct")

    # Check positions
    positions = root_states[:, :, 0:3]
    print(f"\nPosition ranges:")
    print(f"  X: [{positions[:, :, 0].min():.2f}, {positions[:, :, 0].max():.2f}]")
    print(f"  Y: [{positions[:, :, 1].min():.2f}, {positions[:, :, 1].max():.2f}]")
    print(f"  Z: [{positions[:, :, 2].min():.2f}, {positions[:, :, 2].max():.2f}]")

    # Check orientations (quaternions)
    orientations = root_states[:, :, 3:7]
    quat_norms = orientations.norm(dim=-1)
    print(f"\nQuaternion norms (should be ~1.0):")
    print(f"  Min: {quat_norms.min():.6f}, Max: {quat_norms.max():.6f}")
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-4), "Invalid quaternions!"
    print("✓ Quaternions valid")

    # Check velocities (should be small, not zero anymore)
    velocities = root_states[:, :, 7:13]
    lin_vel = velocities[:, :, 0:3]
    ang_vel = velocities[:, :, 3:6]

    print(f"\nVelocities (should be small initial values):")
    print(f"  Linear velocity - Max abs: {lin_vel.abs().max():.4f}, Mean abs: {lin_vel.abs().mean():.4f}")
    print(f"  Angular velocity - Max abs: {ang_vel.abs().max():.4f}, Mean abs: {ang_vel.abs().mean():.4f}")

    # Check that velocities are non-zero (randomized)
    assert lin_vel.abs().max() > 0, "Linear velocities should be non-zero!"
    assert ang_vel.abs().max() > 0, "Angular velocities should be non-zero!"
    print("✓ Velocities are randomized (non-zero)")

    # Test 2: Line formation
    print("\n[Test 2] Line formation")
    root_states_line = randomizer.get_random_formation(num_agents, formation_type="line")
    print(f"Output shape: {root_states_line.shape}")
    positions_line = root_states_line[:, :, 0:3]

    # Check that agents are roughly collinear (for first env)
    env_0_positions = positions_line[0]
    print(f"\nEnv 0 positions:")
    for i, pos in enumerate(env_0_positions):
        print(f"  Agent {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # Calculate distances between adjacent agents
    distances = []
    for i in range(num_agents - 1):
        dist = (env_0_positions[i] - env_0_positions[i+1]).norm()
        distances.append(dist.item())
    print(f"\nDistances between adjacent agents: {[f'{d:.2f}' for d in distances]}")
    print(f"Min: {min(distances):.2f}m, Max: {max(distances):.2f}m")

    # Test 3: Grid formation
    print("\n[Test 3] Grid formation")
    num_agents_grid = 4  # 2x2 grid
    root_states_grid = randomizer.get_random_formation(num_agents_grid, formation_type="grid")
    print(f"Output shape: {root_states_grid.shape}")
    positions_grid = root_states_grid[:, :, 0:3]

    env_0_positions_grid = positions_grid[0]
    print(f"\nEnv 0 grid positions (should form 2x2):")
    for i, pos in enumerate(env_0_positions_grid):
        print(f"  Agent {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # Test 4: Separation constraints
    print("\n[Test 4] Checking minimum separation")
    min_sep = randomizer.cfg.min_agent_separation
    violations = 0

    for env_idx in range(num_envs):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = (positions[env_idx, i] - positions[env_idx, j]).norm()
                if dist < min_sep * 0.95:  # 5% tolerance
                    violations += 1
                    print(f"  Violation in env {env_idx}: agents {i}-{j} distance = {dist:.2f}m")

    if violations == 0:
        print(f"✓ All agent pairs maintain minimum separation ({min_sep}m)")
    else:
        print(f"✗ Found {violations} separation violations!")

    # Test 5: Ground clearance
    print("\n[Test 5] Checking ground clearance")
    min_height = randomizer.cfg.ground_clearance_min
    min_z = positions[:, :, 2].min()
    print(f"Minimum Z position: {min_z:.2f}m (threshold: {min_height}m)")
    if min_z >= min_height:
        print("✓ All agents above ground clearance")
    else:
        print("✗ Some agents below ground clearance!")

    # Test 6: Scale factor
    print("\n[Test 6] Testing scale factor")
    root_states_small = randomizer.get_random_formation(num_agents, formation_type="grid", scale_factor=0.5)
    root_states_large = randomizer.get_random_formation(num_agents, formation_type="grid", scale_factor=2.0)

    pos_small = root_states_small[0, :, 0:3]
    pos_large = root_states_large[0, :, 0:3]

    # Calculate spread (max pairwise distance)
    spread_small = max([(pos_small[i] - pos_small[j]).norm().item()
                        for i in range(num_agents) for j in range(i+1, num_agents)])
    spread_large = max([(pos_large[i] - pos_large[j]).norm().item()
                        for i in range(num_agents) for j in range(i+1, num_agents)])

    print(f"Formation spread with scale_factor=0.5: {spread_small:.2f}m")
    print(f"Formation spread with scale_factor=2.0: {spread_large:.2f}m")
    print(f"Ratio (should be ~4.0): {spread_large / (spread_small + 1e-6):.2f}")

    # Test 7: Velocity bounds and scale factor
    print("\n[Test 7] Testing velocity bounds with scale factor")

    # Get expected bounds from config
    max_lin_vel = randomizer.iris_cfg.max_lin_vel
    max_yaw_rate = randomizer.iris_cfg.max_yaw_rate

    # Test with scale_factor = 1.0
    root_states_vel = randomizer.get_random_formation(num_agents, scale_factor=1.0)
    lin_vel_1 = root_states_vel[:, :, 7:10]
    ang_vel_yaw_1 = root_states_vel[:, :, 12:13]

    print(f"\nWith scale_factor=1.0:")
    print(f"  Linear velocity range: [{lin_vel_1.min():.4f}, {lin_vel_1.max():.4f}]")
    print(f"  Expected bounds: [{-max_lin_vel/5:.4f}, {max_lin_vel/5:.4f}]")
    print(f"  Yaw rate range: [{ang_vel_yaw_1.min():.4f}, {ang_vel_yaw_1.max():.4f}]")
    print(f"  Expected bounds: [{-max_yaw_rate:.4f}, {max_yaw_rate:.4f}]")

    # Check bounds are respected
    assert lin_vel_1.abs().max() <= max_lin_vel / 5 * 1.01, "Linear velocity exceeds bounds!"
    assert ang_vel_yaw_1.abs().max() <= max_yaw_rate * 1.01, "Yaw rate exceeds bounds!"
    print("✓ Velocities within bounds for scale_factor=1.0")

    # Test with scale_factor = 0.5 (should have smaller velocities)
    root_states_vel_small = randomizer.get_random_formation(num_agents, scale_factor=0.5)
    lin_vel_05 = root_states_vel_small[:, :, 7:10]
    ang_vel_yaw_05 = root_states_vel_small[:, :, 12:13]

    print(f"\nWith scale_factor=0.5:")
    print(f"  Linear velocity range: [{lin_vel_05.min():.4f}, {lin_vel_05.max():.4f}]")
    print(f"  Expected bounds: [{-max_lin_vel/10:.4f}, {max_lin_vel/10:.4f}]")
    print(f"  Yaw rate range: [{ang_vel_yaw_05.min():.4f}, {ang_vel_yaw_05.max():.4f}]")
    print(f"  Expected bounds: [{-max_yaw_rate*0.5:.4f}, {max_yaw_rate*0.5:.4f}]")

    assert lin_vel_05.abs().max() <= max_lin_vel / 10 * 1.01, "Linear velocity exceeds bounds!"
    assert ang_vel_yaw_05.abs().max() <= max_yaw_rate * 0.5 * 1.01, "Yaw rate exceeds bounds!"
    print("✓ Velocities scaled correctly with scale_factor=0.5")

    # Test with scale_factor = 2.0 (should have larger velocities)
    root_states_vel_large = randomizer.get_random_formation(num_agents, scale_factor=2.0)
    lin_vel_20 = root_states_vel_large[:, :, 7:10]
    ang_vel_yaw_20 = root_states_vel_large[:, :, 12:13]

    print(f"\nWith scale_factor=2.0:")
    print(f"  Linear velocity range: [{lin_vel_20.min():.4f}, {lin_vel_20.max():.4f}]")
    print(f"  Expected bounds: [{-max_lin_vel*2/5:.4f}, {max_lin_vel*2/5:.4f}]")
    print(f"  Yaw rate range: [{ang_vel_yaw_20.min():.4f}, {ang_vel_yaw_20.max():.4f}]")
    print(f"  Expected bounds: [{-max_yaw_rate*2:.4f}, {max_yaw_rate*2:.4f}]")

    assert lin_vel_20.abs().max() <= max_lin_vel * 2 / 5 * 1.01, "Linear velocity exceeds bounds!"
    assert ang_vel_yaw_20.abs().max() <= max_yaw_rate * 2.0 * 1.01, "Yaw rate exceeds bounds!"
    print("✓ Velocities scaled correctly with scale_factor=2.0")

    print("\n" + "=" * 70)
    print("All tests completed!")

if __name__ == "__main__":
    test_formation_generator()
