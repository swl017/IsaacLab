#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Initial States Module Test Suite for iris_ma6.

This test suite validates the initial states generation functionality including:
- Configuration validation
- Agent placement within cylinder bounds
- Minimum clearance enforcement
- Target distance bounds
- Designated observer selection and gimbal pointing
- Curriculum-based parameter scaling
- Velocity bounds

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/initial_states/tests/run_tests.py

    # With verbose output
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/initial_states/tests/run_tests.py --test-verbose
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run initial states test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW we can import other modules
import math
import sys
import torch
import traceback
from datetime import datetime

# Import module components
from isaaclab_tasks.direct.iris_ma6.initial_states import (
    InitialStatesCfg,
    InitialStatesResult,
    InitialStatesGenerator,
    InitialStates,
)


# ==============================================================================
# Test Results Tracking
# ==============================================================================


class TestResults:
    """Track test results with pass/fail reporting."""

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
        error_lines = error.split("\n")[:5]
        for line in error_lines:
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
        print(f"Total Tests: {total}")
        if total > 0:
            print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)")
            print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)")
            print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)")
        else:
            print("No tests run")

        if self.failed:
            print("\n" + "-" * 80)
            print("FAILED TESTS:")
            print("-" * 80)
            for test_name, error in self.failed:
                print(f"\n{test_name}:")
                print(f"  {error}")

        if self.errors:
            print("\n" + "-" * 80)
            print("TEST ERRORS:")
            print("-" * 80)
            for test_name, error in self.errors:
                print(f"\n{test_name}:")
                print(f"  {error}")

        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


# ==============================================================================
# Configuration Tests
# ==============================================================================


def run_config_tests(results: TestResults, device: torch.device, verbose: bool):
    """Test InitialStatesCfg configuration."""
    print("\n" + "=" * 80)
    print("Testing InitialStatesCfg")
    print("=" * 80)

    # Test 1: Default configuration values
    try:
        cfg = InitialStatesCfg()
        assert cfg.cylinder_diameter_min == 30.0
        assert cfg.cylinder_diameter_max == 100.0
        assert cfg.target_distance_min == 30.0
        assert cfg.target_distance_max == 200.0
        assert cfg.agent_clearance == 10.0
        assert cfg.designated_observer_mode == "random"
        results.add_pass("Default configuration values")
    except AssertionError as e:
        results.add_fail("Default configuration values", str(e))
    except Exception as e:
        results.add_fail("Default configuration values", traceback.format_exc())

    # Test 2: Custom configuration values
    try:
        cfg = InitialStatesCfg(
            cylinder_diameter_min=50.0,
            cylinder_diameter_max=150.0,
            target_distance_min=50.0,
            target_distance_max=300.0,
            agent_clearance=15.0,
            designated_observer_mode="fixed",
        )
        assert cfg.cylinder_diameter_min == 50.0
        assert cfg.cylinder_diameter_max == 150.0
        assert cfg.target_distance_min == 50.0
        assert cfg.target_distance_max == 300.0
        assert cfg.agent_clearance == 15.0
        assert cfg.designated_observer_mode == "fixed"
        results.add_pass("Custom configuration values")
    except AssertionError as e:
        results.add_fail("Custom configuration values", str(e))
    except Exception as e:
        results.add_fail("Custom configuration values", traceback.format_exc())


# ==============================================================================
# Generator Tests
# ==============================================================================


def run_generator_tests(results: TestResults, device: torch.device, verbose: bool):
    """Test InitialStatesGenerator."""
    print("\n" + "=" * 80)
    print("Testing InitialStatesGenerator")
    print("=" * 80)

    num_envs = 32
    num_agents = 3
    cfg = InitialStatesCfg()

    # Test 1: Generator initialization
    try:
        generator = InitialStatesGenerator(
            cfg=cfg,
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
        )
        assert generator.num_envs == num_envs
        assert generator.num_agents == num_agents
        results.add_pass("Generator initialization")
    except Exception as e:
        results.add_fail("Generator initialization", traceback.format_exc())
        return  # Can't continue without generator

    # Test 2: Generate at progress=0.0 (easy)
    try:
        result = generator.generate(curriculum_progress=0.0)
        assert result.agent_positions.shape == (num_envs, num_agents, 3)
        assert result.agent_orientations.shape == (num_envs, num_agents, 4)
        assert result.target_positions.shape == (num_envs, 3)
        assert result.gimbal_joint_positions.shape == (num_envs, num_agents, 3)
        assert result.zoom_levels.shape == (num_envs, num_agents)
        assert result.designated_observer_idx.shape == (num_envs,)
        results.add_pass("Generate at progress=0.0")
    except AssertionError as e:
        results.add_fail("Generate at progress=0.0", str(e))
    except Exception as e:
        results.add_fail("Generate at progress=0.0", traceback.format_exc())

    # Test 3: Generate at progress=1.0 (hard)
    try:
        result = generator.generate(curriculum_progress=1.0)
        assert result.agent_positions.shape == (num_envs, num_agents, 3)
        # Check no NaN or inf
        assert not torch.isnan(result.agent_positions).any()
        assert not torch.isinf(result.agent_positions).any()
        assert not torch.isnan(result.target_positions).any()
        results.add_pass("Generate at progress=1.0 (no NaN/inf)")
    except AssertionError as e:
        results.add_fail("Generate at progress=1.0 (no NaN/inf)", str(e))
    except Exception as e:
        results.add_fail("Generate at progress=1.0 (no NaN/inf)", traceback.format_exc())

    # Test 4: Agent positions within cylinder
    try:
        result = generator.generate(curriculum_progress=0.5)
        centers = result.cylinder_centers
        positions = result.agent_positions

        # Check XY distance from center (should be <= diameter/2)
        max_expected_radius = cfg.cylinder_diameter_max / 2
        for env_idx in range(num_envs):
            center_xy = centers[env_idx, :2]
            for agent_idx in range(num_agents):
                pos_xy = positions[env_idx, agent_idx, :2]
                dist_xy = (pos_xy - center_xy).norm()
                assert dist_xy <= max_expected_radius + 1.0, \
                    f"Env {env_idx}, Agent {agent_idx}: XY dist {dist_xy:.2f} > max {max_expected_radius:.2f}"

        results.add_pass("Agent positions within cylinder bounds")
    except AssertionError as e:
        results.add_fail("Agent positions within cylinder bounds", str(e))
    except Exception as e:
        results.add_fail("Agent positions within cylinder bounds", traceback.format_exc())

    # Test 5: Agent clearance maintained
    try:
        result = generator.generate(curriculum_progress=0.5)
        positions = result.agent_positions
        clearance = cfg.agent_clearance

        violations = 0
        for env_idx in range(num_envs):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = (positions[env_idx, i] - positions[env_idx, j]).norm()
                    if dist < clearance - 0.5:  # Small tolerance
                        violations += 1
                        if verbose:
                            print(f"    Clearance violation: env {env_idx}, agents {i},{j}, dist={dist:.2f}")

        assert violations == 0, f"Found {violations} clearance violations"
        results.add_pass("Agent clearance maintained")
    except AssertionError as e:
        results.add_fail("Agent clearance maintained", str(e))
    except Exception as e:
        results.add_fail("Agent clearance maintained", traceback.format_exc())

    # Test 6: Target distance bounds at progress=0
    try:
        result = generator.generate(curriculum_progress=0.0)
        distances = result.distances_to_target

        # At progress=0, distance should be close to min
        mean_dist = distances.mean().item()
        assert cfg.target_distance_min - 5.0 <= mean_dist <= cfg.target_distance_min + 15.0, \
            f"Mean distance {mean_dist:.2f} not near min {cfg.target_distance_min}"

        results.add_pass("Target distance bounds at progress=0")
    except AssertionError as e:
        results.add_fail("Target distance bounds at progress=0", str(e))
    except Exception as e:
        results.add_fail("Target distance bounds at progress=0", traceback.format_exc())

    # Test 7: Target distance bounds at progress=1
    try:
        result = generator.generate(curriculum_progress=1.0)
        distances = result.distances_to_target

        # At progress=1, should have some larger distances
        max_dist = distances.max().item()
        assert max_dist > cfg.target_distance_min + 10.0, \
            f"Max distance {max_dist:.2f} not large enough for progress=1"

        results.add_pass("Target distance bounds at progress=1")
    except AssertionError as e:
        results.add_fail("Target distance bounds at progress=1", str(e))
    except Exception as e:
        results.add_fail("Target distance bounds at progress=1", traceback.format_exc())


# ==============================================================================
# Designated Observer Tests
# ==============================================================================


def run_observer_tests(results: TestResults, device: torch.device, verbose: bool):
    """Test designated observer functionality."""
    print("\n" + "=" * 80)
    print("Testing Designated Observer")
    print("=" * 80)

    num_envs = 32
    num_agents = 3

    # Test 1: Fixed mode always selects agent 0
    try:
        cfg = InitialStatesCfg(designated_observer_mode="fixed")
        generator = InitialStatesGenerator(cfg, num_envs, num_agents, device)
        result = generator.generate(curriculum_progress=0.5)

        assert (result.designated_observer_idx == 0).all(), \
            "Fixed mode should always select agent 0"
        results.add_pass("Fixed observer mode")
    except AssertionError as e:
        results.add_fail("Fixed observer mode", str(e))
    except Exception as e:
        results.add_fail("Fixed observer mode", traceback.format_exc())

    # Test 2: Random mode selects different agents
    try:
        cfg = InitialStatesCfg(designated_observer_mode="random")
        generator = InitialStatesGenerator(cfg, num_envs, num_agents, device)
        result = generator.generate(curriculum_progress=0.5)

        unique_observers = result.designated_observer_idx.unique()
        # With 32 envs and 3 agents, should have multiple unique observers
        assert len(unique_observers) > 1, \
            "Random mode should select different agents"
        results.add_pass("Random observer mode")
    except AssertionError as e:
        results.add_fail("Random observer mode", str(e))
    except Exception as e:
        results.add_fail("Random observer mode", traceback.format_exc())

    # Test 3: Observer gimbal points at target (within tolerance)
    try:
        cfg = InitialStatesCfg(designated_observer_mode="fixed")
        generator = InitialStatesGenerator(cfg, num_envs, num_agents, device)
        result = generator.generate(curriculum_progress=0.5)

        # For each env, verify observer's gimbal points toward target
        max_angle_error_deg = 0.0
        for env_idx in range(min(num_envs, 8)):  # Check first 8 envs
            observer_idx = result.designated_observer_idx[env_idx].item()
            agent_pos = result.agent_positions[env_idx, observer_idx]
            agent_quat = result.agent_orientations[env_idx, observer_idx]
            target_pos = result.target_positions[env_idx]
            gimbal_angles = result.gimbal_joint_positions[env_idx, observer_idx]

            # Compute expected angles
            expected_yaw, expected_pitch = generator._compute_gimbal_angles_to_target(
                agent_pos.unsqueeze(0),
                agent_quat.unsqueeze(0),
                target_pos.unsqueeze(0),
            )

            # Compare
            yaw_error = abs(gimbal_angles[0].item() - expected_yaw.item())
            pitch_error = abs(gimbal_angles[2].item() - expected_pitch.item())

            # Convert to degrees for easier interpretation
            yaw_error_deg = math.degrees(yaw_error)
            pitch_error_deg = math.degrees(pitch_error)

            max_angle_error_deg = max(max_angle_error_deg, yaw_error_deg, pitch_error_deg)

            if verbose:
                print(f"    Env {env_idx}: yaw error={yaw_error_deg:.2f}deg, pitch error={pitch_error_deg:.2f}deg")

        # Allow 1 degree tolerance
        assert max_angle_error_deg < 1.0, \
            f"Observer gimbal error too large: {max_angle_error_deg:.2f} deg"
        results.add_pass("Observer gimbal points at target")
    except AssertionError as e:
        results.add_fail("Observer gimbal points at target", str(e))
    except Exception as e:
        results.add_fail("Observer gimbal points at target", traceback.format_exc())


# ==============================================================================
# Gimbal Curriculum Tests
# ==============================================================================


def run_gimbal_curriculum_tests(results: TestResults, device: torch.device, verbose: bool):
    """Test gimbal curriculum-based randomization."""
    print("\n" + "=" * 80)
    print("Testing Gimbal Curriculum")
    print("=" * 80)

    num_envs = 64
    num_agents = 3

    # Test 1: At progress=0, all agents should point at target (gradual mode)
    try:
        cfg = InitialStatesCfg(gimbal_curriculum_mode="gradual")
        generator = InitialStatesGenerator(cfg, num_envs, num_agents, device)
        result = generator.generate(curriculum_progress=0.0)

        # Verify non-observer agents also point at target
        total_pointing = 0
        total_checked = 0

        for env_idx in range(num_envs):
            observer_idx = result.designated_observer_idx[env_idx].item()
            for agent_idx in range(num_agents):
                if agent_idx == observer_idx:
                    continue  # Skip observer

                agent_pos = result.agent_positions[env_idx, agent_idx]
                agent_quat = result.agent_orientations[env_idx, agent_idx]
                target_pos = result.target_positions[env_idx]
                gimbal_angles = result.gimbal_joint_positions[env_idx, agent_idx]

                expected_yaw, expected_pitch = generator._compute_gimbal_angles_to_target(
                    agent_pos.unsqueeze(0),
                    agent_quat.unsqueeze(0),
                    target_pos.unsqueeze(0),
                )

                yaw_error = abs(gimbal_angles[0].item() - expected_yaw.item())
                pitch_error = abs(gimbal_angles[2].item() - expected_pitch.item())

                # If error < 1 degree, consider it pointing
                if math.degrees(yaw_error) < 1.0 and math.degrees(pitch_error) < 1.0:
                    total_pointing += 1
                total_checked += 1

        # At progress=0, all should point (100%)
        pointing_ratio = total_pointing / total_checked if total_checked > 0 else 0
        assert pointing_ratio > 0.95, \
            f"At progress=0, expected >95% pointing, got {pointing_ratio*100:.1f}%"
        results.add_pass("Gimbal curriculum at progress=0 (all pointing)")
    except AssertionError as e:
        results.add_fail("Gimbal curriculum at progress=0 (all pointing)", str(e))
    except Exception as e:
        results.add_fail("Gimbal curriculum at progress=0 (all pointing)", traceback.format_exc())

    # Test 2: At progress=1, some agents should have random gimbal
    try:
        cfg = InitialStatesCfg(gimbal_curriculum_mode="gradual")
        generator = InitialStatesGenerator(cfg, num_envs, num_agents, device)
        result = generator.generate(curriculum_progress=1.0)

        total_pointing = 0
        total_checked = 0

        for env_idx in range(num_envs):
            observer_idx = result.designated_observer_idx[env_idx].item()
            for agent_idx in range(num_agents):
                if agent_idx == observer_idx:
                    continue

                agent_pos = result.agent_positions[env_idx, agent_idx]
                agent_quat = result.agent_orientations[env_idx, agent_idx]
                target_pos = result.target_positions[env_idx]
                gimbal_angles = result.gimbal_joint_positions[env_idx, agent_idx]

                expected_yaw, expected_pitch = generator._compute_gimbal_angles_to_target(
                    agent_pos.unsqueeze(0),
                    agent_quat.unsqueeze(0),
                    target_pos.unsqueeze(0),
                )

                yaw_error = abs(gimbal_angles[0].item() - expected_yaw.item())
                pitch_error = abs(gimbal_angles[2].item() - expected_pitch.item())

                if math.degrees(yaw_error) < 5.0 and math.degrees(pitch_error) < 5.0:
                    total_pointing += 1
                total_checked += 1

        # At progress=1, should have mix (not all pointing)
        pointing_ratio = total_pointing / total_checked if total_checked > 0 else 0
        assert pointing_ratio < 0.5, \
            f"At progress=1, expected <50% pointing, got {pointing_ratio*100:.1f}%"
        results.add_pass("Gimbal curriculum at progress=1 (randomized)")
    except AssertionError as e:
        results.add_fail("Gimbal curriculum at progress=1 (randomized)", str(e))
    except Exception as e:
        results.add_fail("Gimbal curriculum at progress=1 (randomized)", traceback.format_exc())


# ==============================================================================
# Velocity Tests
# ==============================================================================


def run_velocity_tests(results: TestResults, device: torch.device, verbose: bool):
    """Test velocity generation and bounds."""
    print("\n" + "=" * 80)
    print("Testing Velocity Generation")
    print("=" * 80)

    num_envs = 32
    num_agents = 3
    cfg = InitialStatesCfg()
    generator = InitialStatesGenerator(cfg, num_envs, num_agents, device)

    # Test 1: Agent velocities at progress=0 should be ~0
    try:
        result = generator.generate(curriculum_progress=0.0)
        agent_vel_mag = result.agent_linear_velocities.norm(dim=-1)
        max_vel = agent_vel_mag.max().item()

        assert max_vel < 0.5, f"At progress=0, expected low velocity, got max={max_vel:.2f}"
        results.add_pass("Agent velocity at progress=0 (near zero)")
    except AssertionError as e:
        results.add_fail("Agent velocity at progress=0 (near zero)", str(e))
    except Exception as e:
        results.add_fail("Agent velocity at progress=0 (near zero)", traceback.format_exc())

    # Test 2: Agent velocities at progress=1 within bounds
    try:
        result = generator.generate(curriculum_progress=1.0)
        agent_vel_mag = result.agent_linear_velocities.norm(dim=-1)
        max_vel = agent_vel_mag.max().item()

        assert max_vel <= cfg.agent_max_velocity + 0.1, \
            f"Agent velocity {max_vel:.2f} exceeds max {cfg.agent_max_velocity}"
        results.add_pass("Agent velocity at progress=1 (within bounds)")
    except AssertionError as e:
        results.add_fail("Agent velocity at progress=1 (within bounds)", str(e))
    except Exception as e:
        results.add_fail("Agent velocity at progress=1 (within bounds)", traceback.format_exc())

    # Test 3: Target velocities at progress=0 should be ~0
    try:
        result = generator.generate(curriculum_progress=0.0)
        target_vel_mag = result.target_velocities[:, :3].norm(dim=-1)
        max_vel = target_vel_mag.max().item()

        assert max_vel < 0.5, f"At progress=0, expected low target velocity, got max={max_vel:.2f}"
        results.add_pass("Target velocity at progress=0 (near zero)")
    except AssertionError as e:
        results.add_fail("Target velocity at progress=0 (near zero)", str(e))
    except Exception as e:
        results.add_fail("Target velocity at progress=0 (near zero)", traceback.format_exc())

    # Test 4: Target velocities at progress=1 within bounds
    try:
        result = generator.generate(curriculum_progress=1.0)
        target_vel_mag = result.target_velocities[:, :3].norm(dim=-1)
        max_vel = target_vel_mag.max().item()

        assert max_vel <= cfg.target_max_velocity + 0.1, \
            f"Target velocity {max_vel:.2f} exceeds max {cfg.target_max_velocity}"
        results.add_pass("Target velocity at progress=1 (within bounds)")
    except AssertionError as e:
        results.add_fail("Target velocity at progress=1 (within bounds)", str(e))
    except Exception as e:
        results.add_fail("Target velocity at progress=1 (within bounds)", traceback.format_exc())


# ==============================================================================
# Integration Tests
# ==============================================================================


def run_integration_tests(results: TestResults, device: torch.device, verbose: bool):
    """Test full integration via InitialStates wrapper."""
    print("\n" + "=" * 80)
    print("Testing InitialStates Wrapper (Integration)")
    print("=" * 80)

    num_envs = 32
    num_agents = 3
    cfg = InitialStatesCfg()

    # Test 1: Wrapper initialization
    try:
        initial_states = InitialStates(
            cfg=cfg,
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
        )
        assert initial_states.num_envs == num_envs
        assert initial_states.num_agents == num_agents
        results.add_pass("Wrapper initialization")
    except Exception as e:
        results.add_fail("Wrapper initialization", traceback.format_exc())
        return

    # Test 2: Generate via wrapper
    try:
        result = initial_states.generate(curriculum_progress=0.5)
        assert isinstance(result, InitialStatesResult)
        assert result.agent_positions.shape == (num_envs, num_agents, 3)
        results.add_pass("Generate via wrapper")
    except Exception as e:
        results.add_fail("Generate via wrapper", traceback.format_exc())

    # Test 3: Partial env_ids generation
    try:
        env_ids = torch.tensor([0, 5, 10, 15], device=device)
        result = initial_states.generate(
            env_ids=env_ids,
            curriculum_progress=0.5,
        )
        # Result should have shape for subset
        assert result.agent_positions.shape == (4, num_agents, 3)
        assert result.target_positions.shape == (4, 3)
        results.add_pass("Partial env_ids generation")
    except Exception as e:
        results.add_fail("Partial env_ids generation", traceback.format_exc())

    # Test 4: Config update
    try:
        initial_states.update_config(target_distance_max=150.0)
        assert initial_states.cfg.target_distance_max == 150.0

        # Verify the change affects generation - config should propagate
        result = initial_states.generate(curriculum_progress=1.0)
        max_dist = result.distances_to_target.max().item()

        # Agent-to-target distance can exceed target_distance_max by up to cylinder radius
        # Max possible: target_distance_max (150) + cylinder_radius_max (50) = 200
        # We verify it's bounded reasonably (within geometry limits)
        max_expected = 150.0 + initial_states.cfg.cylinder_diameter_max / 2 + 10.0
        assert max_dist <= max_expected, \
            f"Max distance {max_dist:.1f} exceeds geometric bound {max_expected:.1f}"

        # Also verify distances are non-zero and reasonable
        assert result.distances_to_target.min().item() > 0.0
        assert not torch.isnan(result.distances_to_target).any()

        results.add_pass("Config update")
    except Exception as e:
        results.add_fail("Config update", traceback.format_exc())


# ==============================================================================
# Zoom Tests
# ==============================================================================


def run_zoom_tests(results: TestResults, device: torch.device, verbose: bool):
    """Test zoom level generation."""
    print("\n" + "=" * 80)
    print("Testing Zoom Level Generation")
    print("=" * 80)

    num_envs = 32
    num_agents = 3
    cfg = InitialStatesCfg()
    generator = InitialStatesGenerator(cfg, num_envs, num_agents, device)

    # Test 1: Zoom levels within range
    try:
        result = generator.generate(curriculum_progress=0.5)
        zoom_min = result.zoom_levels.min().item()
        zoom_max = result.zoom_levels.max().item()

        assert zoom_min >= cfg.zoom_initial_min - 0.01, \
            f"Zoom min {zoom_min} below config min {cfg.zoom_initial_min}"
        expected_zoom_max = cfg.zoom_initial_max_start + 0.5 * (
            cfg.zoom_initial_max_end - cfg.zoom_initial_max_start
        )
        assert zoom_max <= expected_zoom_max + 0.01, \
            f"Zoom max {zoom_max} above expected max {expected_zoom_max} at progress=0.5"

        results.add_pass("Zoom levels within range")
    except AssertionError as e:
        results.add_fail("Zoom levels within range", str(e))
    except Exception as e:
        results.add_fail("Zoom levels within range", traceback.format_exc())

    # Test 2: Zoom values have variation
    try:
        result = generator.generate(curriculum_progress=0.5)
        zoom_std = result.zoom_levels.std().item()

        assert zoom_std > 0.5, f"Zoom levels lack variation: std={zoom_std:.3f}"
        results.add_pass("Zoom levels have variation")
    except AssertionError as e:
        results.add_fail("Zoom levels have variation", str(e))
    except Exception as e:
        results.add_fail("Zoom levels have variation", traceback.format_exc())


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Main test runner."""
    print("=" * 80)
    print("INITIAL STATES MODULE TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verbose = args_cli.test_verbose

    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Verbose:       {verbose}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()

    try:
        run_config_tests(results, device, verbose)
        run_generator_tests(results, device, verbose)
        run_observer_tests(results, device, verbose)
        run_gimbal_curriculum_tests(results, device, verbose)
        run_velocity_tests(results, device, verbose)
        run_zoom_tests(results, device, verbose)
        run_integration_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Test suite", traceback.format_exc())

    success = results.print_summary()
    print(f"\nFinished:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
