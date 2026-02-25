#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for distance-based formation generation.

Run with:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/randomization/tests/run_tests.py
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run distance-based formation test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import other modules
import sys
import math
import torch
import traceback
from datetime import datetime


class TestResults:
    """Track test results with pass/fail counts."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}")
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
        print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)" if total > 0 else "Failed: 0")
        print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)" if total > 0 else "Errors: 0")

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


def run_formation_cfg_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test formation configuration classes."""
    print("\n" + "=" * 80)
    print("Testing DistanceBasedFormationCfg")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma5.randomization import DistanceBasedFormationCfg

    # Test 1: Default configuration
    try:
        cfg = DistanceBasedFormationCfg()
        assert cfg.distance_min == 10.0, f"Expected distance_min=10.0, got {cfg.distance_min}"
        assert cfg.distance_max == 80.0, f"Expected distance_max=80.0, got {cfg.distance_max}"
        assert cfg.min_agent_separation == 5.0, f"Expected min_agent_separation=5.0, got {cfg.min_agent_separation}"
        results.add_pass("Default configuration values")
    except Exception as e:
        results.add_fail("Default configuration values", str(e))

    # Test 2: Custom configuration
    try:
        cfg = DistanceBasedFormationCfg(
            distance_min=15.0,
            distance_max=100.0,
            min_agent_separation=8.0,
        )
        assert cfg.distance_min == 15.0
        assert cfg.distance_max == 100.0
        assert cfg.min_agent_separation == 8.0
        results.add_pass("Custom configuration values")
    except Exception as e:
        results.add_fail("Custom configuration values", str(e))


def run_generator_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test distance-based formation generator."""
    print("\n" + "=" * 80)
    print("Testing DistanceBasedFormationGenerator")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma5.randomization import (
        DistanceBasedFormationCfg,
        DistanceBasedFormationGenerator,
    )

    num_envs = 16
    num_agents = 2
    print(f"Configuration: {num_envs} environments, {num_agents} agents")

    # Test 1: Generator initialization
    try:
        cfg = DistanceBasedFormationCfg()
        generator = DistanceBasedFormationGenerator(
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

    # Test 2: Basic generation
    try:
        result = generator.generate(scale_factor=0.5)
        assert result.agent_root_states.shape == (num_envs, num_agents, 13), \
            f"Expected shape ({num_envs}, {num_agents}, 13), got {result.agent_root_states.shape}"
        assert result.target_position.shape == (num_envs, 3), \
            f"Expected target shape ({num_envs}, 3), got {result.target_position.shape}"
        results.add_pass("Basic generation (scale_factor=0.5)")
    except Exception as e:
        results.add_fail("Basic generation", traceback.format_exc())

    # Test 3: Distance constraints at scale_factor=0 (close)
    try:
        cfg = DistanceBasedFormationCfg(distance_min=10.0, distance_max=80.0)
        generator = DistanceBasedFormationGenerator(cfg=cfg, num_envs=num_envs, num_agents=num_agents, device=device)

        result = generator.generate(scale_factor=0.0)
        distances = result.distances_to_target  # [num_envs, num_agents]

        # At scale=0, distances should be close to distance_min
        mean_dist = distances.mean().item()
        max_dist = distances.max().item()

        if verbose:
            print(f"    scale_factor=0: mean={mean_dist:.1f}m, max={max_dist:.1f}m")

        # Allow some tolerance for distance variation
        assert mean_dist < 20.0, f"Expected mean distance < 20m at scale=0, got {mean_dist:.1f}m"
        results.add_pass(f"Distance constraints at scale=0 (mean={mean_dist:.1f}m)")
    except Exception as e:
        results.add_fail("Distance constraints at scale=0", traceback.format_exc())

    # Test 4: Distance constraints at scale_factor=1 (far)
    try:
        result = generator.generate(scale_factor=1.0)
        distances = result.distances_to_target

        mean_dist = distances.mean().item()
        min_dist = distances.min().item()

        if verbose:
            print(f"    scale_factor=1: mean={mean_dist:.1f}m, min={min_dist:.1f}m")

        # At scale=1, distances should be closer to distance_max
        assert mean_dist > 30.0, f"Expected mean distance > 30m at scale=1, got {mean_dist:.1f}m"
        results.add_pass(f"Distance constraints at scale=1 (mean={mean_dist:.1f}m)")
    except Exception as e:
        results.add_fail("Distance constraints at scale=1", traceback.format_exc())

    # Test 5: All agents within distance bounds
    try:
        cfg = DistanceBasedFormationCfg(distance_min=10.0, distance_max=80.0, validate_distances=True)
        generator = DistanceBasedFormationGenerator(cfg=cfg, num_envs=100, num_agents=num_agents, device=device)

        # Test multiple scale factors
        for scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = generator.generate(scale_factor=scale)
            distances = result.distances_to_target

            min_dist = distances.min().item()
            max_dist = distances.max().item()

            # Distances should be within bounds (with small tolerance for numerical errors)
            assert min_dist >= cfg.distance_min - 1.0, \
                f"At scale={scale}: min distance {min_dist:.1f}m < {cfg.distance_min}m"
            assert max_dist <= cfg.distance_max + 1.0, \
                f"At scale={scale}: max distance {max_dist:.1f}m > {cfg.distance_max}m"

        results.add_pass("All agents within distance bounds (100 envs, 5 scales)")
    except Exception as e:
        results.add_fail("All agents within distance bounds", traceback.format_exc())

    # Test 6: Formation types
    try:
        for formation_type in ["planar", "grid", "line"]:
            result = generator.generate(scale_factor=0.5, formation_type=formation_type)
            assert result.agent_root_states.shape == (100, num_agents, 13)
            assert all(ft == formation_type for ft in result.formation_types)
        results.add_pass("Formation types (planar, grid, line)")
    except Exception as e:
        results.add_fail("Formation types", traceback.format_exc())

    # Test 7: Target height at mean agent height
    try:
        result = generator.generate(scale_factor=0.5)
        agent_heights = result.agent_root_states[:, :, 2]  # [num_envs, num_agents]
        mean_agent_height = agent_heights.mean(dim=1)  # [num_envs]
        target_height = result.target_position[:, 2]  # [num_envs]

        height_diff = (target_height - mean_agent_height).abs()
        max_diff = height_diff.max().item()

        # Target should be within target_height_offset_range of mean agent height
        assert max_diff < 5.0, f"Target height differs from mean agent height by {max_diff:.1f}m"
        results.add_pass(f"Target height near mean agent height (max diff={max_diff:.1f}m)")
    except Exception as e:
        results.add_fail("Target height near mean agent height", traceback.format_exc())

    # Test 8: Agent separation
    try:
        cfg = DistanceBasedFormationCfg(min_agent_separation=5.0)
        generator = DistanceBasedFormationGenerator(cfg=cfg, num_envs=50, num_agents=3, device=device)

        result = generator.generate(scale_factor=0.5)
        positions = result.agent_root_states[:, :, 0:3]  # [50, 3, 3]

        # Check pairwise distances
        min_sep = float("inf")
        for i in range(3):
            for j in range(i + 1, 3):
                dist = (positions[:, i] - positions[:, j]).norm(dim=-1).min().item()
                min_sep = min(min_sep, dist)

        assert min_sep >= cfg.min_agent_separation * 0.9, \
            f"Min agent separation {min_sep:.1f}m < {cfg.min_agent_separation}m"
        results.add_pass(f"Agent separation maintained (min={min_sep:.1f}m)")
    except Exception as e:
        results.add_fail("Agent separation maintained", traceback.format_exc())


def run_randomizer_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test Randomizer wrapper class."""
    print("\n" + "=" * 80)
    print("Testing Randomizer")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma5.randomization import Randomizer, DistanceBasedFormationCfg

    num_envs = 16
    num_agents = 2

    # Test 1: Randomizer initialization
    try:
        randomizer = Randomizer(
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
        )
        assert randomizer.num_envs == num_envs
        assert randomizer.num_agents == num_agents
        results.add_pass("Randomizer initialization (default config)")
    except Exception as e:
        results.add_fail("Randomizer initialization", traceback.format_exc())
        return

    # Test 2: Randomizer with custom config
    try:
        cfg = DistanceBasedFormationCfg(distance_min=15.0, distance_max=60.0)
        randomizer = Randomizer(
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
            cfg=cfg,
        )
        assert randomizer.cfg.distance_min == 15.0
        assert randomizer.cfg.distance_max == 60.0
        results.add_pass("Randomizer initialization (custom config)")
    except Exception as e:
        results.add_fail("Randomizer initialization (custom config)", traceback.format_exc())

    # Test 3: generate_formation_and_target method
    try:
        result = randomizer.generate_formation_and_target(scale_factor=0.5)
        assert result.agent_root_states.shape == (num_envs, num_agents, 13)
        assert result.target_position.shape == (num_envs, 3)
        assert result.formation_center.shape == (num_envs, 3)
        assert result.distances_to_target.shape == (num_envs, num_agents)
        results.add_pass("generate_formation_and_target returns correct shapes")
    except Exception as e:
        results.add_fail("generate_formation_and_target", traceback.format_exc())

    # Test 4: Partial env_ids
    try:
        env_ids = torch.tensor([0, 2, 5, 7], device=device)
        result = randomizer.generate_formation_and_target(
            env_ids=env_ids,
            scale_factor=0.5,
        )
        assert result.agent_root_states.shape == (len(env_ids), num_agents, 13)
        results.add_pass("Partial env_ids subset generation")
    except Exception as e:
        results.add_fail("Partial env_ids subset generation", traceback.format_exc())


def main():
    """Main test runner."""
    print("=" * 80)
    print("DISTANCE-BASED FORMATION TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()
    verbose = args_cli.test_verbose

    try:
        run_formation_cfg_tests(results, device, verbose)
        run_generator_tests(results, device, verbose)
        run_randomizer_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Test suite execution", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
