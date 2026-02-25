#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test gimbal pointing computation.

This test verifies that:
1. GimbalStabilizer correctly computes angles to point at a target
2. Computed angles are within gimbal limits
3. Camera pointing direction matches target direction after applying angles
4. Formation + target + gimbal integration produces feasible configurations

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/randomization/test_gimbal_pointing.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test gimbal pointing computation")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import sys
import traceback

from isaaclab_tasks.direct.iris_ma5.controller import GimbalStabilizer, GimbalStabilizerCfg
from isaaclab_tasks.direct.iris_ma5.randomization import (
    InitialStatesRandomizer,
    TargetSampler,
    TargetSamplerCfg
)


class TestResults:
    """Track test results with pass/fail counts."""

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
        for line in error.split('\n')[:5]:
            print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        print(f"    {error[:200]}")

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}, Errors: {len(self.errors)}")

        if self.failed:
            print("\nFailed tests:")
            for name, error in self.failed:
                print(f"  - {name}")

        return len(self.failed) == 0 and len(self.errors) == 0


def run_basic_gimbal_pointing_tests(results: TestResults, device: torch.device):
    """Test basic gimbal pointing computation."""
    print("\n" + "=" * 80)
    print("Testing Basic Gimbal Pointing")
    print("=" * 80)

    cfg = GimbalStabilizerCfg()
    stabilizer = GimbalStabilizer(cfg, device=str(device))

    # Test 1: Target directly in front (should give yaw=0, pitch=0)
    try:
        drone_pos = torch.tensor([[0.0, 0.0, 10.0]], device=device)
        drone_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # Identity
        target_pos = torch.tensor([[10.0, 0.0, 10.0]], device=device)  # 10m in front, same height

        yaw, roll, pitch = stabilizer.compute_stabilized_angles_from_target_point(
            target_pos, drone_pos, drone_quat
        )

        if abs(yaw[0].item()) < 0.1 and abs(pitch[0].item()) < 0.1:
            results.add_pass(f"Target directly in front (yaw={yaw[0].item():.3f}, pitch={pitch[0].item():.3f})")
        else:
            results.add_fail("Target directly in front", f"Expected yaw~0, pitch~0, got yaw={yaw[0].item():.3f}, pitch={pitch[0].item():.3f}")
    except Exception as e:
        results.add_fail("Target directly in front", traceback.format_exc())

    # Test 2: Target to the left (should give positive yaw)
    try:
        target_pos = torch.tensor([[0.0, 10.0, 10.0]], device=device)  # 10m to the left
        yaw, roll, pitch = stabilizer.compute_stabilized_angles_from_target_point(
            target_pos, drone_pos, drone_quat
        )
        if yaw[0].item() > 0.5:  # Should be around pi/2 ~ 1.57
            results.add_pass(f"Target to the left (yaw={math.degrees(yaw[0].item()):.1f} deg)")
        else:
            results.add_fail("Target to the left", f"Expected positive yaw, got {math.degrees(yaw[0].item()):.1f} deg")
    except Exception as e:
        results.add_fail("Target to the left", traceback.format_exc())

    # Test 3: Target to the right (should give negative yaw)
    try:
        target_pos = torch.tensor([[0.0, -10.0, 10.0]], device=device)  # 10m to the right
        yaw, roll, pitch = stabilizer.compute_stabilized_angles_from_target_point(
            target_pos, drone_pos, drone_quat
        )
        if yaw[0].item() < -0.5:  # Should be around -pi/2 ~ -1.57
            results.add_pass(f"Target to the right (yaw={math.degrees(yaw[0].item()):.1f} deg)")
        else:
            results.add_fail("Target to the right", f"Expected negative yaw, got {math.degrees(yaw[0].item()):.1f} deg")
    except Exception as e:
        results.add_fail("Target to the right", traceback.format_exc())

    # Test 4: Target below (should give positive pitch - looking down)
    try:
        target_pos = torch.tensor([[10.0, 0.0, 5.0]], device=device)  # 5m below drone
        yaw, roll, pitch = stabilizer.compute_stabilized_angles_from_target_point(
            target_pos, drone_pos, drone_quat
        )
        if pitch[0].item() > 0.1:  # Looking down is positive pitch
            results.add_pass(f"Target below (pitch={math.degrees(pitch[0].item()):.1f} deg)")
        else:
            results.add_fail("Target below", f"Expected positive pitch, got {math.degrees(pitch[0].item()):.1f} deg")
    except Exception as e:
        results.add_fail("Target below", traceback.format_exc())

    # Test 5: Target above (should give negative pitch - looking up)
    try:
        target_pos = torch.tensor([[10.0, 0.0, 15.0]], device=device)  # 5m above drone
        yaw, roll, pitch = stabilizer.compute_stabilized_angles_from_target_point(
            target_pos, drone_pos, drone_quat
        )
        if pitch[0].item() < -0.1:  # Looking up is negative pitch
            results.add_pass(f"Target above (pitch={math.degrees(pitch[0].item()):.1f} deg)")
        else:
            results.add_fail("Target above", f"Expected negative pitch, got {math.degrees(pitch[0].item()):.1f} deg")
    except Exception as e:
        results.add_fail("Target above", traceback.format_exc())


def run_pointing_direction_verification(results: TestResults, device: torch.device):
    """Verify that camera pointing direction matches target after applying angles."""
    print("\n" + "=" * 80)
    print("Testing Camera Pointing Direction Verification")
    print("=" * 80)

    cfg = GimbalStabilizerCfg()
    stabilizer = GimbalStabilizer(cfg, device=str(device))

    num_tests = 50

    try:
        # Random drone positions
        drone_pos = torch.randn(num_tests, 3, device=device) * 10
        drone_pos[:, 2] = torch.abs(drone_pos[:, 2]) + 5  # Ensure positive Z

        # Identity quaternion for simplicity
        drone_quat = torch.zeros(num_tests, 4, device=device)
        drone_quat[:, 0] = 1.0

        # Random targets (ensure reasonable distance)
        target_offset = torch.randn(num_tests, 3, device=device) * 20
        target_offset[:, :2] = target_offset[:, :2].clamp(-30, 30)  # XY in range
        target_offset[:, 2] = target_offset[:, 2].clamp(-5, 5)  # Z offset small
        target_pos = drone_pos + target_offset
        target_pos[:, 2] = torch.clamp(target_pos[:, 2], 1.0, 50.0)

        # Ensure targets are not too close
        dist_to_target = (target_pos - drone_pos).norm(dim=-1)
        far_enough_mask = dist_to_target > 3.0

        if far_enough_mask.sum() < 10:
            results.add_fail("Pointing direction verification", "Not enough valid test cases")
            return

        drone_pos = drone_pos[far_enough_mask]
        drone_quat = drone_quat[far_enough_mask]
        target_pos = target_pos[far_enough_mask]
        actual_num_tests = drone_pos.shape[0]

        # Compute gimbal angles
        yaw, roll, pitch = stabilizer.compute_stabilized_angles_from_target_point(
            target_pos, drone_pos, drone_quat
        )

        # Get camera pointing direction after applying angles
        pointing_dir = stabilizer.get_camera_pointing_direction(yaw, roll, pitch, drone_quat)

        # Compute expected direction (normalized vector from drone to target)
        expected_dir = target_pos - drone_pos
        expected_dir = expected_dir / (expected_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute angular error
        dot_product = (pointing_dir * expected_dir).sum(dim=-1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angular_error = torch.acos(dot_product)  # radians

        mean_error_deg = torch.rad2deg(angular_error.mean()).item()
        max_error_deg = torch.rad2deg(angular_error.max()).item()

        print(f"  Tested {actual_num_tests} random configurations")
        print(f"  Angular error - Mean: {mean_error_deg:.2f} deg, Max: {max_error_deg:.2f} deg")

        # Allow some error due to angle clamping at limits
        if mean_error_deg < 5.0:
            results.add_pass(f"Camera points at target (mean error: {mean_error_deg:.2f} deg)")
        else:
            results.add_fail("Camera pointing error too high", f"Mean error: {mean_error_deg:.2f} deg > 5 deg")

        # Also check that most individual errors are small
        small_error_count = (angular_error < math.radians(10.0)).sum().item()
        small_error_pct = 100.0 * small_error_count / actual_num_tests

        if small_error_pct > 80.0:
            results.add_pass(f"{small_error_pct:.1f}% of tests have < 10 deg error")
        else:
            results.add_fail("Too many large pointing errors", f"Only {small_error_pct:.1f}% have < 10 deg error")

    except Exception as e:
        results.add_fail("Pointing direction verification", traceback.format_exc())


def run_formation_target_integration_test(results: TestResults, device: torch.device):
    """Test that formations + target sampling produce feasible gimbal configurations."""
    print("\n" + "=" * 80)
    print("Testing Formation + Target + Gimbal Integration")
    print("=" * 80)

    num_envs = 100
    num_agents = 2

    try:
        # Create components
        formation_randomizer = InitialStatesRandomizer(num_envs, device)

        gimbal_cfg = GimbalStabilizerCfg()
        stabilizer = GimbalStabilizer(gimbal_cfg, device=str(device))

        target_cfg = TargetSamplerCfg(
            pitch_limit_min=gimbal_cfg.pitch_limits[0],
            pitch_limit_max=gimbal_cfg.pitch_limits[1],
            pitch_safety_margin=math.radians(25.0)
        )
        target_sampler = TargetSampler(target_cfg, device)

        # Test with each formation type
        for formation_type in ["planar", "grid", "line"]:
            print(f"\n  Testing formation type: {formation_type}")

            # Configure for curriculum-scaled formations
            formation_randomizer.cfg.z_variation_curriculum_enabled = True
            formation_randomizer.cfg.z_variation_min = (0.0, 0.0)
            formation_randomizer.cfg.z_variation_max = (0.0, 2.0)
            formation_randomizer.cfg.max_agent_separation = 5.0
            formation_randomizer.cfg.min_agent_separation = 2.0
            formation_randomizer.cfg.line_max_z_component = 0.3

            # Use scale_factor based on formation type
            if formation_type == "planar":
                scale_factor = 0.2
            elif formation_type == "grid":
                scale_factor = 0.5
            else:
                scale_factor = 0.8

            # Generate formation
            formation_data = formation_randomizer.get_random_formation(
                num_agents=num_agents,
                num_envs=num_envs,
                formation_type=formation_type,
                scale_factor=scale_factor
            )

            # Sample targets
            target_positions = target_sampler.sample_target_position(formation_data, scale_factor=scale_factor)

            # Compute gimbal angles for each agent and check feasibility
            all_pitches_within_limits = True
            pitch_violations = []

            for agent_idx in range(num_agents):
                agent_pos = formation_data[:, agent_idx, 0:3]
                agent_quat = formation_data[:, agent_idx, 3:7]

                yaw, roll, pitch = stabilizer.compute_stabilized_angles_from_target_point(
                    target_positions, agent_pos, agent_quat
                )

                # Check if pitch is within limits
                pitch_min, pitch_max = gimbal_cfg.pitch_limits
                below_min = pitch < pitch_min
                above_max = pitch > pitch_max

                if below_min.any() or above_max.any():
                    all_pitches_within_limits = False
                    if below_min.any():
                        violation = (pitch_min - pitch[below_min]).max().item()
                        pitch_violations.append(violation)
                    if above_max.any():
                        violation = (pitch[above_max] - pitch_max).max().item()
                        pitch_violations.append(violation)

            if all_pitches_within_limits:
                results.add_pass(f"{formation_type}: all pitches within limits")
            else:
                max_violation_deg = math.degrees(max(pitch_violations))
                results.add_fail(
                    f"{formation_type}: pitch limits violated",
                    f"Max violation: {max_violation_deg:.1f} deg"
                )

            # Also check height spread for formation
            positions = formation_data[:, :, 2]  # Z coordinates
            height_spreads = positions.max(dim=1)[0] - positions.min(dim=1)[0]
            mean_height_spread = height_spreads.mean().item()
            max_height_spread = height_spreads.max().item()

            print(f"    Height spread - Mean: {mean_height_spread:.2f}m, Max: {max_height_spread:.2f}m")

            if formation_type == "planar" and max_height_spread < 0.1:
                results.add_pass(f"{formation_type}: has no height variation")
            elif formation_type == "planar":
                results.add_fail(f"{formation_type}: unexpected height variation", f"Max spread: {max_height_spread:.2f}m")

    except Exception as e:
        results.add_fail("Formation + target integration", traceback.format_exc())


def run_curriculum_scaling_test(results: TestResults, device: torch.device):
    """Test that curriculum scaling affects formation geometry correctly."""
    print("\n" + "=" * 80)
    print("Testing Curriculum Scaling")
    print("=" * 80)

    num_envs = 50
    num_agents = 2

    try:
        formation_randomizer = InitialStatesRandomizer(num_envs, device)

        # Enable curriculum-controlled z variation
        formation_randomizer.cfg.z_variation_curriculum_enabled = True
        formation_randomizer.cfg.z_variation_min = (0.0, 0.0)
        formation_randomizer.cfg.z_variation_max = (0.0, 3.0)
        formation_randomizer.cfg.max_agent_separation = 5.0
        formation_randomizer.cfg.min_agent_separation = 2.0

        # Test at scale_factor = 0 (should have minimal height variation)
        formation_scale0 = formation_randomizer.get_random_formation(
            num_agents=num_agents,
            num_envs=num_envs,
            formation_type="grid",
            scale_factor=0.0
        )
        positions_scale0 = formation_scale0[:, :, 0:3]
        z_spread_0 = (positions_scale0[:, :, 2].max(dim=1)[0] - positions_scale0[:, :, 2].min(dim=1)[0]).mean().item()

        # Test at scale_factor = 1 (should have more height variation)
        formation_scale1 = formation_randomizer.get_random_formation(
            num_agents=num_agents,
            num_envs=num_envs,
            formation_type="grid",
            scale_factor=1.0
        )
        positions_scale1 = formation_scale1[:, :, 0:3]
        z_spread_1 = (positions_scale1[:, :, 2].max(dim=1)[0] - positions_scale1[:, :, 2].min(dim=1)[0]).mean().item()

        print(f"  Z spread at scale_factor=0: {z_spread_0:.3f}m")
        print(f"  Z spread at scale_factor=1: {z_spread_1:.3f}m")

        if z_spread_0 < 0.1:
            results.add_pass(f"Scale 0 has minimal height variation ({z_spread_0:.3f}m)")
        else:
            results.add_fail("Scale 0 height variation too large", f"{z_spread_0:.3f}m > 0.1m")

        if z_spread_1 > z_spread_0:
            results.add_pass(f"Scale 1 has more height variation than scale 0")
        else:
            results.add_fail("Scale 1 should have more variation", f"{z_spread_1:.3f}m <= {z_spread_0:.3f}m")

        # Test agent separation scaling
        agent_dist_0 = (positions_scale0[:, 0, :] - positions_scale0[:, 1, :]).norm(dim=-1).mean().item()
        agent_dist_1 = (positions_scale1[:, 0, :] - positions_scale1[:, 1, :]).norm(dim=-1).mean().item()

        print(f"  Agent distance at scale_factor=0: {agent_dist_0:.2f}m")
        print(f"  Agent distance at scale_factor=1: {agent_dist_1:.2f}m")

        if agent_dist_1 > agent_dist_0:
            results.add_pass(f"Scale 1 has larger agent separation")
        else:
            results.add_fail("Scale 1 should have larger separation", f"{agent_dist_1:.2f}m <= {agent_dist_0:.2f}m")

    except Exception as e:
        results.add_fail("Curriculum scaling test", traceback.format_exc())


def main():
    print("=" * 80)
    print("GIMBAL POINTING TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")

    results = TestResults()

    try:
        run_basic_gimbal_pointing_tests(results, device)
        run_pointing_direction_verification(results, device)
        run_formation_target_integration_test(results, device)
        run_curriculum_scaling_test(results, device)
    except Exception as e:
        results.add_error("Test suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
