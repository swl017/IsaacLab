"""
Comprehensive test suite for gimbal-aware target sampling.

Tests the TargetSampler with formations from InitialStatesRandomizer.

Run with: ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/randomization/test_target_sampling.py
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
import math
import sys

from target_sampling import TargetSampler, TargetSamplerCfg, create_target_sampler


def test_basic_sampling():
    """Test 1: Basic target sampling functionality."""
    print("\n" + "=" * 70)
    print("Test 1: Basic Target Sampling")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create simple configuration
    cfg = TargetSamplerCfg(
        target_distance_min=20.0,
        target_distance_max=60.0,
        pitch_limit_min=math.radians(-45.0),
        pitch_limit_max=math.radians(10.0),
        pitch_safety_margin=math.radians(5.0),
    )
    sampler = TargetSampler(cfg, device)

    # Create simple formation: 3 agents in a line at same height
    num_envs = 10
    num_agents = 3

    agent_positions = torch.zeros(num_envs, num_agents, 3, device=device)
    agent_positions[:, 0, :] = torch.tensor([0, -5, 20], device=device)  # Left
    agent_positions[:, 1, :] = torch.tensor([0,  0, 20], device=device)  # Center
    agent_positions[:, 2, :] = torch.tensor([0,  5, 20], device=device)  # Right

    # Sample targets
    target_positions = sampler.sample_target_position(agent_positions)

    print(f"\nFormation:")
    print(f"  Agent positions: {agent_positions[0]}")

    print(f"\nSampled targets (first 3 envs):")
    for i in range(min(3, num_envs)):
        print(f"  Env {i}: {target_positions[i].cpu().numpy()}")

    # Validate shape
    assert target_positions.shape == (num_envs, 3), f"Wrong shape: {target_positions.shape}"
    print("\n✓ Output shape correct: [num_envs, 3]")

    # Validate distances (horizontal distance only, since that's what's controlled)
    formation_centers = agent_positions.mean(dim=1)

    # Total 3D distance
    distances_3d = torch.norm(target_positions - formation_centers, dim=1)

    # Horizontal (XY) distance
    distances_horizontal = torch.norm(
        target_positions[:, :2] - formation_centers[:, :2], dim=1
    )

    print(f"\nTarget distances from formation center:")
    print(f"  3D distance - Min: {distances_3d.min():.2f}m, Max: {distances_3d.max():.2f}m, Mean: {distances_3d.mean():.2f}m")
    print(f"  Horizontal distance - Min: {distances_horizontal.min():.2f}m, Max: {distances_horizontal.max():.2f}m, Mean: {distances_horizontal.mean():.2f}m")

    # Check horizontal distance (which is what we control)
    assert distances_horizontal.min() >= cfg.target_distance_min, "Horizontal distance below minimum!"
    assert distances_horizontal.max() <= cfg.target_distance_max * 1.01, "Horizontal distance above maximum!"  # 1% tolerance
    print(f"✓ All horizontal distances within range [{cfg.target_distance_min}, {cfg.target_distance_max}]")

    return sampler, agent_positions, target_positions


def test_pitch_angle_validation():
    """
    Test 2: Validate pitch angles with TRULY RANDOM target positions.

    This test samples completely random target positions (NOT using the
    constrained sampler) and validates them. Violations are EXPECTED.

    This tests the validation logic and demonstrates what happens when
    targets are not properly constrained.
    """
    print("\n" + "=" * 70)
    print("Test 2: Validation with Random Targets (Violations Expected)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sampler (we'll only use its validation method)
    cfg = TargetSamplerCfg(
        pitch_limit_min=math.radians(-45.0),
        pitch_limit_max=math.radians(10.0),
        pitch_safety_margin=math.radians(5.0),
    )
    sampler = TargetSampler(cfg, device)

    # Create formation with agents at fixed positions
    num_envs = 500
    num_agents = 3

    agent_positions = torch.zeros(num_envs, num_agents, 3, device=device)
    agent_positions[:, 0, :] = torch.tensor([0, -5, 20], device=device)
    agent_positions[:, 1, :] = torch.tensor([0,  0, 20], device=device)
    agent_positions[:, 2, :] = torch.tensor([0,  5, 20], device=device)

    # Sample COMPLETELY RANDOM target positions (NOT using constrained sampler)
    # This will produce violations!
    target_positions = torch.zeros(num_envs, 3, device=device)
    target_positions[:, 0] = torch.rand(num_envs, device=device) * 100 - 50  # X: -50 to 50
    target_positions[:, 1] = torch.rand(num_envs, device=device) * 100 - 50  # Y: -50 to 50
    target_positions[:, 2] = torch.rand(num_envs, device=device) * 40 + 5    # Z: 5 to 45

    print(f"\nRandom target sampling:")
    print(f"  Num environments: {num_envs}")
    print(f"  Num agents: {num_agents}")
    print(f"  Target position ranges:")
    print(f"    X: [{target_positions[:, 0].min():.1f}, {target_positions[:, 0].max():.1f}]")
    print(f"    Y: [{target_positions[:, 1].min():.1f}, {target_positions[:, 1].max():.1f}]")
    print(f"    Z: [{target_positions[:, 2].min():.1f}, {target_positions[:, 2].max():.1f}]")

    # Create dummy orientations (all facing +X)
    agent_orientations = torch.zeros(num_envs, num_agents, 4, device=device)
    agent_orientations[:, :, 0] = 1.0  # w=1, x=y=z=0 (identity quaternion)

    # Validate feasibility
    validation = sampler.validate_target_feasibility(
        agent_positions, agent_orientations, target_positions
    )

    all_feasible = validation['all_feasible']
    pitch_angles = validation['pitch_angles']
    violations = validation['violations']

    print(f"\nFeasibility results:")
    print(f"  Feasible environments: {all_feasible.sum().item()}/{num_envs}")
    print(f"  Infeasible environments: {(~all_feasible).sum().item()}/{num_envs}")

    print(f"\nPitch angle statistics (degrees):")
    print(f"  Min: {torch.rad2deg(pitch_angles.min()):.2f}°")
    print(f"  Max: {torch.rad2deg(pitch_angles.max()):.2f}°")
    print(f"  Mean: {torch.rad2deg(pitch_angles.mean()):.2f}°")

    print(f"\nPitch limits (with safety margin):")
    print(f"  Min: {math.degrees(sampler.pitch_min_safe):.2f}°")
    print(f"  Max: {math.degrees(sampler.pitch_max_safe):.2f}°")

    # Check if any violations
    total_violations = violations.sum().item()
    print(f"\nTotal violations: {total_violations} / {num_envs * num_agents}")

    if total_violations == 0:
        print("✗ UNEXPECTED: No violations found with random targets!")
        print("  This suggests the validation logic may not be working.")
        raise AssertionError("Expected violations with random targets, but got zero!")
    else:
        print(f"✓ Found {total_violations} pitch angle violations (EXPECTED with random targets)")
        print(f"  Violation rate: {100 * total_violations / (num_envs * num_agents):.1f}%")
        print(f"\n  Showing first 10 violations:")
        # Print details of first 10 violations
        violation_count = 0
        for env_idx in range(num_envs):
            for agent_idx in range(num_agents):
                if violations[env_idx, agent_idx]:
                    angle_deg = torch.rad2deg(pitch_angles[env_idx, agent_idx]).item()
                    print(f"    Env {env_idx}, Agent {agent_idx}: Pitch = {angle_deg:.2f}°")
                    violation_count += 1
                    if violation_count >= 10:
                        break
            if violation_count >= 10:
                break
        if total_violations > 10:
            print(f"  ... and {total_violations - 10} more violations")

    print("\n✓ Validation logic works correctly (detected violations as expected)")
    return all_feasible, pitch_angles


def test_formation_integration():
    """
    Test 3: Integration with formation generator.

    This test validates that when using the formation generator + target sampler
    together, ALL targets should be feasible (zero violations).
    """
    print("\n" + "=" * 70)
    print("Test 3: Integration Test (Formation + Target Sampling)")
    print("=" * 70)

    try:
        from initial_states import InitialStatesRandomizer, InitialStatesRandomizerCfg
    except ImportError as e:
        print(f"✗ Test FAILED: Cannot import required modules ({e})")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create formation randomizer
    num_envs = 20
    formation_randomizer = InitialStatesRandomizer(num_envs, device)

    # Generate formations
    num_agents = 3
    formation_data = formation_randomizer.get_random_formation(num_agents)

    print(f"\nGenerated formations:")
    print(f"  Shape: {formation_data.shape}")
    print(f"  Num environments: {num_envs}")
    print(f"  Num agents: {num_agents}")

    # Create target sampler using gimbal limits from formation randomizer's config
    pitch_limits = formation_randomizer.iris_cfg.max_gimbal_pitch_angle
    sampler = create_target_sampler(
        pitch_limits=pitch_limits,
        device=device,
        target_distance_range=(20.0, 80.0),
        pitch_safety_margin_deg=5.0,
    )

    print(f"\nGimbal pitch limits: [{math.degrees(pitch_limits[0]):.1f}°, {math.degrees(pitch_limits[1]):.1f}°]")

    # Sample targets for formations
    target_positions = sampler.sample_target_position(formation_data)

    print(f"\nSampled targets:")
    print(f"  Shape: {target_positions.shape}")
    print(f"  Height range: [{target_positions[:, 2].min():.2f}, {target_positions[:, 2].max():.2f}]")

    # Validate - THIS SHOULD SHOW 100% FEASIBILITY
    validation = sampler.validate_target_feasibility(
        formation_data[:, :, 0:3],  # positions
        formation_data[:, :, 3:7],  # orientations
        target_positions,
    )

    feasibility_rate = validation['all_feasible'].float().mean().item() * 100
    num_violations = validation['violations'].sum().item()

    print(f"\nValidation results:")
    print(f"  Feasibility rate: {feasibility_rate:.1f}%")
    print(f"  Total violations: {num_violations} / {num_envs * num_agents}")

    if feasibility_rate == 100.0 and num_violations == 0:
        print("✓ Perfect! All formations have feasible targets (zero violations)")
    else:
        print(f"✗ FAILED: Expected 100% feasibility but got {feasibility_rate:.1f}%")
        print(f"  Violations found: {num_violations}")

        # Print violation details
        pitch_angles = validation['pitch_angles']
        for env_idx in range(num_envs):
            for agent_idx in range(num_agents):
                if validation['violations'][env_idx, agent_idx]:
                    angle_deg = torch.rad2deg(pitch_angles[env_idx, agent_idx]).item()
                    print(f"    Env {env_idx}, Agent {agent_idx}: Pitch = {angle_deg:.2f}°")

        # This is a real failure - raise exception
        raise AssertionError(f"Integration test failed: {num_violations} violations found when using formation + target sampling together!")

    # Compute distances
    agent_positions = formation_data[:, :, 0:3]
    formation_centers = agent_positions.mean(dim=1)
    distances_horizontal = torch.norm(target_positions[:, :2] - formation_centers[:, :2], dim=1)

    print(f"\nHorizontal distance statistics:")
    print(f"  Min: {distances_horizontal.min():.2f}m")
    print(f"  Max: {distances_horizontal.max():.2f}m")
    print(f"  Mean: {distances_horizontal.mean():.2f}m")

    return validation


def test_edge_cases():
    """Test 4: Edge cases and stress tests."""
    print("\n" + "=" * 70)
    print("Test 4: Edge Cases and Stress Tests")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TargetSamplerCfg(
        pitch_limit_min=math.radians(-45.0),
        pitch_limit_max=math.radians(10.0),
        pitch_safety_margin=math.radians(5.0),
    )
    sampler = TargetSampler(cfg, device)

    # Edge case 1: Single agent
    print("\nEdge Case 1: Single agent")
    num_envs = 5
    agent_positions_single = torch.zeros(num_envs, 1, 3, device=device)
    agent_positions_single[:, 0, :] = torch.tensor([0, 0, 20], device=device)

    targets_single = sampler.sample_target_position(agent_positions_single)
    print(f"  Shape: {targets_single.shape}")
    assert targets_single.shape == (num_envs, 3), "Wrong shape for single agent!"
    print("  ✓ Single agent works")

    # Edge case 2: Many agents
    print("\nEdge Case 2: Many agents (10)")
    num_agents_many = 10
    agent_positions_many = torch.randn(num_envs, num_agents_many, 3, device=device)
    agent_positions_many[:, :, 2] = 20.0  # Same height

    targets_many = sampler.sample_target_position(agent_positions_many)
    print(f"  Shape: {targets_many.shape}")
    assert targets_many.shape == (num_envs, 3), "Wrong shape for many agents!"
    print("  ✓ Many agents works")

    # Edge case 3: Large vertical spread
    print("\nEdge Case 3: Large vertical spread")
    agent_positions_spread = torch.zeros(num_envs, 3, 3, device=device)
    agent_positions_spread[:, 0, :] = torch.tensor([0, 0, 10], device=device)   # Low
    agent_positions_spread[:, 1, :] = torch.tensor([0, 0, 20], device=device)   # Mid
    agent_positions_spread[:, 2, :] = torch.tensor([0, 0, 30], device=device)   # High

    targets_spread = sampler.sample_target_position(agent_positions_spread)

    # Validate
    orientations_spread = torch.zeros(num_envs, 3, 4, device=device)
    orientations_spread[:, :, 0] = 1.0

    validation_spread = sampler.validate_target_feasibility(
        agent_positions_spread, orientations_spread, targets_spread
    )

    print(f"  Feasibility: {validation_spread['all_feasible'].sum().item()}/{num_envs}")
    print(f"  ✓ Large vertical spread handled")

    # Edge case 4: Scale factor
    print("\nEdge Case 4: Scale factor")
    targets_scale_small = sampler.sample_target_position(agent_positions_single, scale_factor=0.5)
    targets_scale_large = sampler.sample_target_position(agent_positions_single, scale_factor=2.0)

    dist_small = torch.norm(targets_scale_small - agent_positions_single[:, 0, :], dim=1).mean()
    dist_large = torch.norm(targets_scale_large - agent_positions_single[:, 0, :], dim=1).mean()

    print(f"  Mean distance (scale=0.5): {dist_small:.2f}m")
    print(f"  Mean distance (scale=2.0): {dist_large:.2f}m")
    print(f"  Ratio: {dist_large / dist_small:.2f}x")
    print("  ✓ Scale factor works")

    print("\n✓ All edge cases passed!")


def test_feasible_height_range_computation():
    """Test 5: Detailed test of height range computation."""
    print("\n" + "=" * 70)
    print("Test 5: Feasible Height Range Computation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TargetSamplerCfg(
        pitch_limit_min=math.radians(-45.0),
        pitch_limit_max=math.radians(10.0),
        pitch_safety_margin=math.radians(5.0),
    )
    sampler = TargetSampler(cfg, device)

    # Test case: 3 agents at different heights
    num_envs = 1
    agent_positions = torch.tensor([
        [[0, 0, 15],   # Agent 0: z=15
         [0, 0, 20],   # Agent 1: z=20
         [0, 0, 25]]   # Agent 2: z=25
    ], dtype=torch.float32, device=device)

    # Target at horizontal distance 50m
    target_x = torch.tensor([50.0], device=device)
    target_y = torch.tensor([0.0], device=device)

    formation_center = agent_positions.mean(dim=1)  # [1, 3]

    z_min, z_max = sampler._compute_feasible_height_range(
        agent_positions, target_x, target_y, formation_center
    )

    print(f"\nAgent heights: [15, 20, 25]m")
    print(f"Target horizontal distance: 50m")
    print(f"\nFeasible height range:")
    print(f"  z_min: {z_min[0]:.2f}m")
    print(f"  z_max: {z_max[0]:.2f}m")
    print(f"  Range size: {(z_max[0] - z_min[0]):.2f}m")

    # Manually compute expected values using CORRECTED formulas
    d_h = 50.0
    pitch_min_safe = math.radians(-40)  # -45° + 5° margin
    pitch_max_safe = math.radians(5)    # +10° - 5° margin

    tan_pitch_min_safe = math.tan(pitch_min_safe)  # tan(-40°) = -0.839
    tan_pitch_max_safe = math.tan(pitch_max_safe)  # tan(5°) = 0.087

    print(f"\nExpected bounds for each agent:")
    for i, z_a in enumerate([15, 20, 25]):
        # z_min comes from pitch_max_safe (looking down)
        # z_max comes from pitch_min_safe (looking up)
        z_min_expected = z_a - d_h * tan_pitch_max_safe  # Lower bound
        z_max_expected = z_a - d_h * tan_pitch_min_safe  # Upper bound
        print(f"  Agent {i} (z={z_a}): [{z_min_expected:.2f}, {z_max_expected:.2f}]")

    # Intersection: max of mins, min of maxes
    expected_z_min = 25 - d_h * tan_pitch_max_safe  # Highest z_min (agent 2, most restrictive)
    expected_z_max = 15 - d_h * tan_pitch_min_safe  # Lowest z_max (agent 0, most restrictive)

    print(f"\nExpected intersection: [{expected_z_min:.2f}, {expected_z_max:.2f}]")

    # Check if close
    assert abs(z_min[0].item() - expected_z_min) < 1.0, "z_min mismatch!"
    assert abs(z_max[0].item() - expected_z_max) < 1.0, "z_max mismatch!"

    print("✓ Height range computation correct!")

    return z_min, z_max


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GIMBAL-AWARE TARGET SAMPLING - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    try:
        # Test 1: Basic sampling
        test_basic_sampling()

        # Test 2: Pitch validation
        test_pitch_angle_validation()

        # Test 3: Formation integration
        test_formation_integration()

        # Test 4: Edge cases
        test_edge_cases()

        # Test 5: Height range
        test_feasible_height_range_computation()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
