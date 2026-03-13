#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CBF Safety Filter Module Test Suite

Standalone test runner for the CBF safety components.
Uses AppLauncher pattern required by Isaac Lab.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/run_tests.py
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/run_tests.py --test-verbose
"""

import argparse

from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run CBF Safety Filter test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument(
    "--test-verbose",
    action="store_true",
    help="Enable verbose debug output for tests",
)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules
import sys
import traceback
from datetime import datetime

import torch


class TestResults:
    """Tracks test results with pass/fail/error counts."""

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

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        if total > 0:
            print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)")
            print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)")
            print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)")

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


def run_cpa_reward_shaper_tests(results: TestResults, device: torch.device, verbose: bool):
    """Run tests for CPARewardShaper."""
    print("\n" + "=" * 80)
    print("Testing CPARewardShaper")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.cbf_safety import CPARewardShaper, CPARewardShaperCfg

    num_envs = 16
    num_agents = 3
    dt = 0.04  # 25 Hz policy

    # Test 1: Initialization
    try:
        cfg = CPARewardShaperCfg(D_s=2.0, gamma=2.0, T=1.0, lambda_cbf=1.0)
        shaper = CPARewardShaper(cfg, num_envs, num_agents, device)
        assert shaper.cfg.D_s == 2.0
        assert shaper.num_agents == num_agents
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return  # Can't continue without shaper

    # Test 2: Stationary drones - no penalty
    try:
        # Place drones far apart and stationary
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0   # Agent 0 at origin
        positions[:, 1, 0] = 10.0  # Agent 1 at x=10
        positions[:, 2, 0] = 20.0  # Agent 2 at x=20
        positions[:, :, 2] = 3.0   # All at z=3

        velocities = torch.zeros(num_envs, num_agents, 3, device=device)

        penalty = shaper.compute_penalty(positions, velocities, dt)
        assert penalty.shape == (num_envs,)
        assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1e-5), \
            f"Expected zero penalty for stationary far drones, got {penalty.mean().item()}"
        results.add_pass("Stationary drones far apart - no penalty")
    except Exception as e:
        results.add_fail("Stationary drones far apart - no penalty", str(e))

    # Test 3: Drones approaching each other - should have penalty
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 4.0  # 4m apart (> D_s=2m)
        positions[:, 2, 0] = 8.0

        # Agent 0 and 1 approaching each other at 2 m/s
        velocities = torch.zeros(num_envs, num_agents, 3, device=device)
        velocities[:, 0, 0] = 2.0   # Agent 0 moving right
        velocities[:, 1, 0] = -2.0  # Agent 1 moving left

        penalty = shaper.compute_penalty(positions, velocities, dt)
        # They will collide, so penalty should be > 0
        assert penalty.shape == (num_envs,)
        # At 4m/s relative approach and 4m apart, CPA distance is 0 (collision)
        # With dt=0.04, after one step they are at 3.84m apart
        assert penalty.mean() > 0, f"Expected positive penalty, got {penalty.mean().item()}"

        if verbose:
            print(f"    Approaching drones penalty: {penalty.mean().item():.4f}")
        results.add_pass("Approaching drones - positive penalty")
    except Exception as e:
        results.add_fail("Approaching drones - positive penalty", str(e))

    # Test 4: Parallel flight - minimal penalty
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, :] = torch.tensor([0.0, 0.0, 3.0], device=device)
        positions[:, 1, :] = torch.tensor([0.0, 3.0, 3.0], device=device)  # 3m apart in y
        positions[:, 2, :] = torch.tensor([0.0, 6.0, 3.0], device=device)

        # All moving in same direction at same speed
        velocities = torch.zeros(num_envs, num_agents, 3, device=device)
        velocities[:, :, 0] = 5.0  # All moving +x at 5 m/s

        penalty = shaper.compute_penalty(positions, velocities, dt)
        # CPA is current distance since relative velocity is zero
        # Distance is 3m, D_s=2m, so h_CPA = 9 - 4 = 5 > 0, safe
        # Should have minimal/no penalty
        assert penalty.mean() < 0.1, f"Expected low penalty for parallel flight, got {penalty.mean().item()}"

        if verbose:
            print(f"    Parallel flight penalty: {penalty.mean().item():.4f}")
        results.add_pass("Parallel flight - minimal penalty")
    except Exception as e:
        results.add_fail("Parallel flight - minimal penalty", str(e))

    # Test 5: Get CPA info
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 5.0
        positions[:, 2, 0] = 10.0

        velocities = torch.zeros(num_envs, num_agents, 3, device=device)
        velocities[:, 0, 0] = 1.0

        info = shaper.get_cpa_info(positions, velocities)
        assert "h_cpa" in info
        assert "tau" in info
        assert "d_cpa" in info
        assert "min_h_cpa" in info
        assert info["h_cpa"].shape[0] == num_envs

        if verbose:
            print(f"    CPA info: h_cpa min={info['min_h_cpa'].mean().item():.2f}")
        results.add_pass("Get CPA info")
    except Exception as e:
        results.add_fail("Get CPA info", str(e))

    # Test 6: Reset (no-op but should not error)
    try:
        shaper.reset()
        shaper.reset(torch.tensor([0, 1, 2], device=device))
        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", str(e))


def run_deploy_filter_tests(results: TestResults, device: torch.device, verbose: bool):
    """Run tests for RobustDeploymentFilter."""
    print("\n" + "=" * 80)
    print("Testing RobustDeploymentFilter")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.cbf_safety import RobustDeploymentFilter, RobustDeploymentFilterCfg

    num_envs = 16
    num_agents = 3

    # Test 1: Initialization
    try:
        cfg = RobustDeploymentFilterCfg(
            D_s=2.0,
            v_max=15.0,
            tau_delay_max=0.2,
            tau_px4=0.3,
            gamma_deploy=1.0,
            num_iters=2,
        )
        filter = RobustDeploymentFilter(cfg, num_envs, num_agents, device)

        # Check D_deploy computation: D_s + v_max * (tau_delay + tau_px4)
        expected_D_deploy = 2.0 + 15.0 * (0.2 + 0.3)
        assert abs(filter.D_deploy - expected_D_deploy) < 1e-5, \
            f"Expected D_deploy={expected_D_deploy}, got {filter.D_deploy}"

        if verbose:
            print(f"    D_deploy = {filter.D_deploy:.2f}m")
        results.add_pass("Initialization with D_deploy computation")
    except Exception as e:
        results.add_fail("Initialization with D_deploy computation", traceback.format_exc())
        return

    # Test 2: No constraint active when drones far apart
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 20.0  # 20m apart (> D_deploy ~9.5m)
        positions[:, 2, 0] = 40.0

        v_nom = torch.ones(num_envs, num_agents, 3, device=device)  # All moving at (1,1,1)

        v_safe, info = filter.filter(v_nom, positions)

        # Should not modify velocities
        assert torch.allclose(v_safe, v_nom, atol=1e-5), \
            "Velocities should not be modified when far apart"
        assert info["deploy_cbf/filter_active_fraction"] == 0.0

        results.add_pass("No constraint when far apart")
    except Exception as e:
        results.add_fail("No constraint when far apart", str(e))

    # Test 3: Constraint active when close
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 5.0  # 5m apart (< D_deploy ~9.5m)
        positions[:, 2, 0] = 20.0

        # Agent 0 moving toward agent 1
        v_nom = torch.zeros(num_envs, num_agents, 3, device=device)
        v_nom[:, 0, 0] = 10.0  # Approaching at 10 m/s

        v_safe, info = filter.filter(v_nom, positions)

        # Constraint should be active for agent 0
        assert info["deploy_cbf/filter_active_fraction"] > 0
        # Velocity toward agent 1 should be reduced
        assert v_safe[:, 0, 0].mean() < v_nom[:, 0, 0].mean()

        if verbose:
            print(f"    Original v_nom[0,x]: {v_nom[0, 0, 0].item():.2f}")
            print(f"    Filtered v_safe[0,x]: {v_safe[0, 0, 0].item():.2f}")
        results.add_pass("Constraint active when close")
    except Exception as e:
        results.add_fail("Constraint active when close", str(e))

    # Test 4: Conservative with zero neighbor velocities
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 5.0
        positions[:, 2, 0] = 20.0

        v_nom = torch.zeros(num_envs, num_agents, 3, device=device)
        v_nom[:, 0, 0] = 10.0

        # With neighbor_velocities=None (conservative assumption)
        v_safe, _ = filter.filter(v_nom, positions, neighbor_velocities=None)

        # Should still filter appropriately
        assert v_safe[:, 0, 0].mean() < v_nom[:, 0, 0].mean()
        results.add_pass("Conservative with None neighbor velocities")
    except Exception as e:
        results.add_fail("Conservative with None neighbor velocities", str(e))

    # Test 5: Barrier values
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, :] = torch.tensor([0.0, 0.0, 3.0], device=device)
        positions[:, 1, :] = torch.tensor([5.0, 0.0, 3.0], device=device)
        positions[:, 2, :] = torch.tensor([10.0, 0.0, 3.0], device=device)

        h_values = filter.get_barrier_values(positions)
        assert h_values.shape == (num_envs, num_agents, num_agents)

        # Check diagonal is inf
        for i in range(num_agents):
            assert h_values[:, i, i].isinf().all()

        # Check off-diagonal values
        # Distance 0-1: 5m, h = 25 - D_deploy^2
        expected_h_01 = 25.0 - filter.D_deploy**2
        assert torch.allclose(h_values[:, 0, 1], torch.full((num_envs,), expected_h_01, device=device), atol=0.1)

        if verbose:
            print(f"    h[0,1] = {h_values[0, 0, 1].item():.2f}, expected = {expected_h_01:.2f}")
        results.add_pass("Barrier values computation")
    except Exception as e:
        results.add_fail("Barrier values computation", str(e))

    # Test 6: Reset
    try:
        filter.reset()
        assert filter.filter_active.sum() == 0
        filter.reset(torch.tensor([0, 5, 10], device=device))
        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", str(e))


def run_cbf_manager_tests(results: TestResults, device: torch.device, verbose: bool):
    """Run tests for CBFManager."""
    print("\n" + "=" * 80)
    print("Testing CBFManager")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.cbf_safety import CBFManager, CBFManagerCfg

    num_envs = 16
    num_agents = 3
    dt = 0.04

    # Test 1: Training mode initialization
    try:
        cfg = CBFManagerCfg(
            enable_training_penalty=True,
            enable_deployment_filter=False,
            enable_collision_termination=True,
        )
        manager = CBFManager(cfg, num_envs, num_agents, device)
        assert manager.cpa_shaper is not None
        assert manager.deploy_filter is None
        results.add_pass("Training mode initialization")
    except Exception as e:
        results.add_fail("Training mode initialization", traceback.format_exc())
        return

    # Test 2: Compute training penalty
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 4.0
        positions[:, 2, 0] = 8.0

        velocities = torch.zeros(num_envs, num_agents, 3, device=device)
        velocities[:, 0, 0] = 2.0
        velocities[:, 1, 0] = -2.0

        penalty = manager.compute_training_penalty(positions, velocities, dt)
        assert penalty.shape == (num_envs,)
        assert penalty.mean() > 0

        if verbose:
            print(f"    Training penalty: {penalty.mean().item():.4f}")
        results.add_pass("Compute training penalty")
    except Exception as e:
        results.add_fail("Compute training penalty", str(e))

    # Test 3: Collision detection
    try:
        # No collision
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 5.0
        positions[:, 2, 0] = 10.0

        collided = manager.check_collisions(positions)
        assert collided.shape == (num_envs,)
        assert not collided.any(), "Should have no collisions"

        # With collision
        positions[:, 1, 0] = 1.0  # 1m apart, collision_distance=2m
        collided = manager.check_collisions(positions)
        assert collided.all(), "All envs should have collision"

        results.add_pass("Collision detection")
    except Exception as e:
        results.add_fail("Collision detection", str(e))

    # Test 4: Diagnostics
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 5.0
        positions[:, 2, 0] = 10.0

        metrics = manager.get_diagnostics(positions)
        assert "safety/min_separation_mean" in metrics
        assert "safety/min_separation_min" in metrics
        assert "safety/collision_fraction" in metrics

        if verbose:
            print(f"    Min separation: {metrics['safety/min_separation_min']:.2f}m")
        results.add_pass("Diagnostics computation")
    except Exception as e:
        results.add_fail("Diagnostics computation", str(e))

    # Test 5: Deployment mode
    try:
        cfg = CBFManagerCfg(
            enable_training_penalty=False,
            enable_deployment_filter=True,
        )
        manager_deploy = CBFManager(cfg, num_envs, num_agents, device)
        assert manager_deploy.cpa_shaper is None
        assert manager_deploy.deploy_filter is not None

        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 5.0
        positions[:, 2, 0] = 20.0

        v_nom = torch.zeros(num_envs, num_agents, 3, device=device)
        v_nom[:, 0, 0] = 10.0

        v_safe, info = manager_deploy.filter_actions(v_nom, positions)
        assert v_safe.shape == v_nom.shape

        results.add_pass("Deployment mode filter_actions")
    except Exception as e:
        results.add_fail("Deployment mode filter_actions", str(e))

    # Test 6: Training penalty returns zeros when disabled
    try:
        cfg = CBFManagerCfg(enable_training_penalty=False, enable_deployment_filter=False)
        manager_disabled = CBFManager(cfg, num_envs, num_agents, device)

        positions = torch.randn(num_envs, num_agents, 3, device=device)
        velocities = torch.randn(num_envs, num_agents, 3, device=device)

        penalty = manager_disabled.compute_training_penalty(positions, velocities, dt)
        assert torch.allclose(penalty, torch.zeros_like(penalty))
        results.add_pass("Training penalty disabled returns zeros")
    except Exception as e:
        results.add_fail("Training penalty disabled returns zeros", str(e))

    # Test 7: Properties
    try:
        assert manager.D_s == 2.0
        assert manager.D_deploy > manager.D_s  # D_deploy should be inflated
        assert manager.lambda_cbf == 1.0
        results.add_pass("Property accessors")
    except Exception as e:
        results.add_fail("Property accessors", str(e))

    # Test 8: Reset
    try:
        manager.reset()
        manager.reset(torch.tensor([0, 5, 10], device=device))
        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", str(e))


def run_diagnostics_tests(results: TestResults, device: torch.device, verbose: bool):
    """Run tests for CBFDiagnostics."""
    print("\n" + "=" * 80)
    print("Testing CBFDiagnostics")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.cbf_safety import CBFDiagnostics, CBFDiagnosticsCfg

    num_envs = 16
    num_agents = 3

    # Test 1: Initialization
    try:
        cfg = CBFDiagnosticsCfg(D_s=2.0)
        diag = CBFDiagnostics(cfg, num_envs, num_agents, device)
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return

    # Test 2: Compute metrics - no collision
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 5.0
        positions[:, 2, 0] = 10.0

        metrics = diag.compute_metrics(positions)
        assert metrics["safety/collision_fraction"] == 0.0
        assert abs(metrics["safety/min_separation_min"] - 5.0) < 0.01

        if verbose:
            print(f"    Metrics (no collision): {metrics}")
        results.add_pass("Compute metrics - no collision")
    except Exception as e:
        results.add_fail("Compute metrics - no collision", str(e))

    # Test 3: Compute metrics - with collision
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, 0] = 0.0
        positions[:, 1, 0] = 1.0  # 1m apart < D_s=2m
        positions[:, 2, 0] = 10.0

        metrics = diag.compute_metrics(positions)
        assert metrics["safety/collision_fraction"] == 1.0
        assert abs(metrics["safety/min_separation_min"] - 1.0) < 0.01

        if verbose:
            print(f"    Metrics (collision): {metrics}")
        results.add_pass("Compute metrics - with collision")
    except Exception as e:
        results.add_fail("Compute metrics - with collision", str(e))

    # Test 4: Pairwise distances
    try:
        positions = torch.zeros(num_envs, num_agents, 3, device=device)
        positions[:, 0, :] = torch.tensor([0.0, 0.0, 0.0], device=device)
        positions[:, 1, :] = torch.tensor([3.0, 0.0, 0.0], device=device)
        positions[:, 2, :] = torch.tensor([0.0, 4.0, 0.0], device=device)

        distances = CBFDiagnostics.compute_pairwise_distances(positions, num_agents, device)
        assert distances.shape == (num_envs, num_agents, num_agents)

        # Check diagonal is inf
        for i in range(num_agents):
            assert distances[:, i, i].isinf().all()

        # Check specific distances
        assert torch.allclose(distances[:, 0, 1], torch.full((num_envs,), 3.0, device=device), atol=0.01)
        assert torch.allclose(distances[:, 0, 2], torch.full((num_envs,), 4.0, device=device), atol=0.01)
        assert torch.allclose(distances[:, 1, 2], torch.full((num_envs,), 5.0, device=device), atol=0.01)

        results.add_pass("Pairwise distances computation")
    except Exception as e:
        results.add_fail("Pairwise distances computation", str(e))


def main():
    """Main test runner."""
    print("=" * 80)
    print("CBF SAFETY FILTER MODULE TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    verbose = args_cli.test_verbose
    results = TestResults()

    try:
        run_diagnostics_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Diagnostics test suite", traceback.format_exc())

    try:
        run_cpa_reward_shaper_tests(results, device, verbose)
    except Exception as e:
        results.add_error("CPA Reward Shaper test suite", traceback.format_exc())

    try:
        run_deploy_filter_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Deploy Filter test suite", traceback.format_exc())

    try:
        run_cbf_manager_tests(results, device, verbose)
    except Exception as e:
        results.add_error("CBF Manager test suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
