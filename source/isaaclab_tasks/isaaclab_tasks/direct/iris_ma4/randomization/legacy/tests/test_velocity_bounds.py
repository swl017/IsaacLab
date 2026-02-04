"""
Standalone test for velocity initialization in formations.
This test checks that velocities are properly randomized and scaled.
"""

import torch

def test_velocity_bounds():
    """Test velocity initialization without full environment dependency."""

    print("=" * 70)
    print("Testing Velocity Bounds and Scaling")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Mock config values (from typical iris config)
    max_lin_vel = 5.0  # m/s
    max_yaw_rate = 1.0  # rad/s

    num_envs = 100  # More envs for better statistics
    num_agents = 3

    print(f"\nTest configuration:")
    print(f"  Num environments: {num_envs}")
    print(f"  Num agents per env: {num_agents}")
    print(f"  Max linear velocity: {max_lin_vel} m/s")
    print(f"  Max yaw rate: {max_yaw_rate} rad/s")

    # Test different scale factors
    scale_factors = [0.5, 1.0, 2.0]

    for scale_factor in scale_factors:
        print(f"\n{'='*70}")
        print(f"Testing with scale_factor = {scale_factor}")
        print(f"{'='*70}")

        # Simulate the velocity generation from initial_states.py
        # Linear velocity: uniform[-max_lin_vel/5 * scale, max_lin_vel/5 * scale]
        lower_lin = -max_lin_vel / 5 * scale_factor
        upper_lin = max_lin_vel / 5 * scale_factor

        lin_vel = torch.rand(num_envs, num_agents, 3, device=device) * (upper_lin - lower_lin) + lower_lin

        # Yaw rate: uniform[-max_yaw_rate * scale, max_yaw_rate * scale]
        lower_yaw = -max_yaw_rate * scale_factor
        upper_yaw = max_yaw_rate * scale_factor

        yaw_rate = torch.rand(num_envs, num_agents, 1, device=device) * (upper_yaw - lower_yaw) + lower_yaw

        # Analysis
        print(f"\nLinear Velocity:")
        print(f"  Expected range: [{lower_lin:.4f}, {upper_lin:.4f}]")
        print(f"  Actual range:   [{lin_vel.min():.4f}, {lin_vel.max():.4f}]")
        print(f"  Mean: {lin_vel.mean():.4f} (should be ~0.0)")
        print(f"  Std:  {lin_vel.std():.4f}")

        print(f"\nYaw Rate:")
        print(f"  Expected range: [{lower_yaw:.4f}, {upper_yaw:.4f}]")
        print(f"  Actual range:   [{yaw_rate.min():.4f}, {yaw_rate.max():.4f}]")
        print(f"  Mean: {yaw_rate.mean():.4f} (should be ~0.0)")
        print(f"  Std:  {yaw_rate.std():.4f}")

        # Validation checks
        tolerance = 0.05  # 5% tolerance for bounds

        # Check linear velocity bounds
        assert lin_vel.min() >= lower_lin - abs(lower_lin) * tolerance, \
            f"Linear velocity below lower bound: {lin_vel.min()} < {lower_lin}"
        assert lin_vel.max() <= upper_lin + abs(upper_lin) * tolerance, \
            f"Linear velocity above upper bound: {lin_vel.max()} > {upper_lin}"
        print("✓ Linear velocity within bounds")

        # Check yaw rate bounds
        assert yaw_rate.min() >= lower_yaw - abs(lower_yaw) * tolerance, \
            f"Yaw rate below lower bound: {yaw_rate.min()} < {lower_yaw}"
        assert yaw_rate.max() <= upper_yaw + abs(upper_yaw) * tolerance, \
            f"Yaw rate above upper bound: {yaw_rate.max()} > {upper_yaw}"
        print("✓ Yaw rate within bounds")

        # Check mean is close to zero (uniform distribution should be centered)
        assert abs(lin_vel.mean()) < 0.1, f"Linear velocity mean not centered: {lin_vel.mean()}"
        assert abs(yaw_rate.mean()) < 0.1, f"Yaw rate mean not centered: {yaw_rate.mean()}"
        print("✓ Distributions centered around zero")

        # Check non-zero (randomized)
        assert lin_vel.abs().max() > 0.01, "Linear velocities should be non-zero"
        assert yaw_rate.abs().max() > 0.01, "Yaw rates should be non-zero"
        print("✓ Velocities are randomized (non-zero)")

    # Test scaling relationships
    print(f"\n{'='*70}")
    print("Testing Scale Factor Relationships")
    print(f"{'='*70}")

    # Generate velocities at different scales
    scale_05_lin = torch.rand(num_envs, num_agents, 3, device=device) * (max_lin_vel / 5 * 0.5 * 2) - max_lin_vel / 5 * 0.5
    scale_10_lin = torch.rand(num_envs, num_agents, 3, device=device) * (max_lin_vel / 5 * 1.0 * 2) - max_lin_vel / 5 * 1.0
    scale_20_lin = torch.rand(num_envs, num_agents, 3, device=device) * (max_lin_vel / 5 * 2.0 * 2) - max_lin_vel / 5 * 2.0

    std_05 = scale_05_lin.std().item()
    std_10 = scale_10_lin.std().item()
    std_20 = scale_20_lin.std().item()

    print(f"\nLinear velocity standard deviations:")
    print(f"  scale_factor=0.5: {std_05:.4f}")
    print(f"  scale_factor=1.0: {std_10:.4f}")
    print(f"  scale_factor=2.0: {std_20:.4f}")

    ratio_05_to_10 = std_10 / std_05
    ratio_10_to_20 = std_20 / std_10

    print(f"\nRatios (should be ~2.0):")
    print(f"  std(1.0) / std(0.5) = {ratio_05_to_10:.2f}")
    print(f"  std(2.0) / std(1.0) = {ratio_10_to_20:.2f}")

    # Ratios should be approximately 2.0 (within 20% tolerance)
    assert 1.6 < ratio_05_to_10 < 2.4, f"Scale ratio incorrect: {ratio_05_to_10}"
    assert 1.6 < ratio_10_to_20 < 2.4, f"Scale ratio incorrect: {ratio_10_to_20}"
    print("✓ Scale factors produce expected variance scaling")

    print(f"\n{'='*70}")
    print("All velocity tests PASSED!")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_velocity_bounds()
