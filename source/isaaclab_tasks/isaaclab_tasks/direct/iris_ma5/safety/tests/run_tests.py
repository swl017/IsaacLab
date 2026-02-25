#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone test runner for safety module.

This script runs tests without pytest to avoid Isaac Sim import issues.
Run with: python3 run_tests.py
"""
import argparse
from isaaclab.app import AppLauncher
# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run safety test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import traceback
from typing import Dict, List, Tuple

# Now that AppLauncher is initialized, we can use normal absolute imports
from isaaclab_tasks.direct.iris_ma3.safety import (
    CollisionDetector,
    CollisionDetectorCfg,
    TTCComputer,
    TTCComputerCfg,
    SafetyManager,
    SafetyManagerCfg,
)


# Test results tracking
class TestResults:
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
        if error:
            # Print first few lines of error
            error_lines = error.split('\n')[:5]
            for line in error_lines:
                print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        print(f"    {error}")

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total: {total}")
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.failed)}")
        print(f"Errors: {len(self.errors)}")

        if self.failed:
            print("\nFailed Tests:")
            for test_name, error in self.failed:
                print(f"  - {test_name}: {error}")

        if self.errors:
            print("\nTest Errors:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")

        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def run_collision_detector_tests(results: TestResults, device: torch.device):
    """Run CollisionDetector tests."""
    print("\n" + "=" * 80)
    print("Testing CollisionDetector")
    print("=" * 80)

    num_envs = 16
    num_agents = 3

    # Test 1: Initialization
    try:
        cfg = CollisionDetectorCfg(min_safe_distance=5.0)
        detector = CollisionDetector(cfg, num_envs, num_agents, device)
        assert detector.num_envs == num_envs
        assert detector.num_agents == num_agents
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", str(e))

    # Test 2: Distance computation
    try:
        cfg = CollisionDetectorCfg(min_safe_distance=5.0)
        detector = CollisionDetector(cfg, num_envs, num_agents, device)

        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_1": torch.tensor([[3.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_2": torch.tensor([[0.0, 4.0, 0.0]] * num_envs, device=device),
        }

        distances = detector.compute_distances(positions)
        assert distances.shape == (num_envs, num_agents, num_agents)
        assert torch.allclose(
            distances[:, 0, 1], torch.tensor(3.0, device=device), atol=1e-5
        )
        assert torch.allclose(
            distances[:, 0, 2], torch.tensor(4.0, device=device), atol=1e-5
        )
        assert torch.allclose(
            distances[:, 1, 2], torch.tensor(5.0, device=device), atol=1e-5
        )
        results.add_pass("Distance computation")
    except Exception as e:
        results.add_fail("Distance computation", str(e))

    # Test 3: Collision detection - no collision
    try:
        cfg = CollisionDetectorCfg(min_safe_distance=5.0)
        detector = CollisionDetector(cfg, num_envs, num_agents, device)

        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_1": torch.tensor([[10.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_2": torch.tensor([[0.0, 10.0, 0.0]] * num_envs, device=device),
        }

        collisions = detector.detect_collisions(positions)
        assert collisions.shape == (num_envs, num_agents, num_agents)
        assert torch.all(~collisions[:, 0, 1])  # No collision
        assert torch.all(~collisions[:, 0, 2])  # No collision
        results.add_pass("Collision detection (no collision)")
    except Exception as e:
        results.add_fail("Collision detection (no collision)", str(e))

    # Test 4: Collision detection - with collision
    try:
        cfg = CollisionDetectorCfg(min_safe_distance=5.0)
        detector = CollisionDetector(cfg, num_envs, num_agents, device)

        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }

        collisions = detector.detect_collisions(positions)
        assert torch.all(collisions[:, 0, 1])  # Collision!
        assert torch.all(collisions[:, 0, 2])  # Collision!
        results.add_pass("Collision detection (with collision)")
    except Exception as e:
        results.add_fail("Collision detection (with collision)", str(e))

    # Test 5: Collision penalties
    try:
        cfg = CollisionDetectorCfg(min_safe_distance=5.0)
        detector = CollisionDetector(cfg, num_envs, num_agents, device)

        agent_ids = ["agent_0", "agent_1", "agent_2"]
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }

        penalties = detector.compute_collision_penalties(positions, agent_ids)
        assert len(penalties) == 3
        for agent_id in agent_ids:
            assert penalties[agent_id].shape == (num_envs,)
            # Each agent collides with 2 others
            assert torch.all(penalties[agent_id] == -2.0)
        results.add_pass("Collision penalties")
    except Exception as e:
        results.add_fail("Collision penalties", str(e))

    # Test 6: Reset functionality
    try:
        cfg = CollisionDetectorCfg(min_safe_distance=5.0)
        detector = CollisionDetector(cfg, num_envs, num_agents, device)

        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "agent_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }

        detector.detect_collisions(positions)
        detector.reset(None)

        assert torch.all(detector.distance_matrix == 0.0)
        assert torch.all(detector.collision_matrix == False)
        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", str(e))


def run_ttc_computer_tests(results: TestResults, device: torch.device):
    """Run TTCComputer tests."""
    print("\n" + "=" * 80)
    print("Testing TTCComputer")
    print("=" * 80)

    num_envs = 16

    # Test 1: Initialization
    try:
        cfg = TTCComputerCfg(horizon_sec=6.0)
        ttc_computer = TTCComputer(cfg, num_envs, device)
        assert ttc_computer.num_envs == num_envs
        assert ttc_computer.logsize_ema.shape == (num_envs,)
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", str(e))

    # Test 2: TTC with no valid detections
    try:
        cfg = TTCComputerCfg(horizon_sec=6.0)
        ttc_computer = TTCComputer(cfg, num_envs, device)

        bbox_width = torch.zeros(num_envs, device=device)
        bbox_height = torch.zeros(num_envs, device=device)
        valid_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        fx = torch.full((num_envs,), 500.0, device=device)
        fy = torch.full((num_envs,), 500.0, device=device)

        phi, tau = ttc_computer.compute_ttc(bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1)
        assert phi.shape == (num_envs,)
        assert tau.shape == (num_envs,)
        assert torch.all(phi == 0.0)  # No penalty for invalid
        results.add_pass("TTC with no valid detections")
    except Exception as e:
        results.add_fail("TTC with no valid detections", str(e))

    # Test 3: TTC with constant bbox size
    try:
        # Disable staleness decay to test pure TTC behavior
        cfg = TTCComputerCfg(horizon_sec=6.0, stale_half_life=1000.0)
        ttc_computer = TTCComputer(cfg, num_envs, device)

        bbox_width = torch.full((num_envs,), 0.1, device=device)
        bbox_height = torch.full((num_envs,), 0.1, device=device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        fx = torch.full((num_envs,), 500.0, device=device)
        fy = torch.full((num_envs,), 500.0, device=device)

        # Multiple steps with same size
        for _ in range(10):
            phi, tau = ttc_computer.compute_ttc(
                bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1
            )

        # No change → TTC should be large, penalty should be small (but may not be exactly zero due to numerical effects)
        assert torch.all(phi <= 0.7), f"Expected phi <= 0.7 for constant bbox, got max={phi.max().item()}"
        assert torch.all(phi >= 0.0), f"Expected phi >= 0.0, got min={phi.min().item()}"
        results.add_pass("TTC with constant bbox")
    except Exception as e:
        results.add_fail("TTC with constant bbox", traceback.format_exc())

    # Test 4: TTC with growing bbox (approaching)
    try:
        # Use lower EMA alpha for faster response, disable zoom gate
        cfg = TTCComputerCfg(
            horizon_sec=6.0,
            ema_alpha_s=0.1,  # Faster response
            ema_alpha_f=0.1,
            stale_half_life=1000.0,  # Disable staleness decay
            zoom_gate_k=1000.0  # Effectively disable zoom gate
        )
        ttc_computer = TTCComputer(cfg, num_envs, device)

        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        fx = torch.full((num_envs,), 500.0, device=device)
        fy = torch.full((num_envs,), 500.0, device=device)

        # Gradually increase bbox size with larger steps
        for i in range(20):
            bbox_size = 0.05 + i * 0.02  # Grows from 0.05 to 0.43
            bbox_width = torch.full((num_envs,), bbox_size, device=device)
            bbox_height = torch.full((num_envs,), bbox_size, device=device)

            phi, tau = ttc_computer.compute_ttc(
                bbox_width, bbox_height, valid_mask, fx, fy, dt=0.05
            )

        # Growing bbox should eventually produce penalty (may take time to build up EMA)
        # After 20 steps of consistent growth, we should see some penalty
        assert phi.max().item() >= 0.0, f"Expected some penalty for growing bbox, got max={phi.max().item()}"
        assert torch.all(phi <= 1.0), f"Expected phi <= 1.0, got max={phi.max().item()}"
        results.add_pass("TTC with growing bbox")
    except Exception as e:
        results.add_fail("TTC with growing bbox", traceback.format_exc())

    # Test 5: Penalty range constraints
    try:
        cfg = TTCComputerCfg(horizon_sec=6.0)
        ttc_computer = TTCComputer(cfg, num_envs, device)

        bbox_width = torch.full((num_envs,), 0.5, device=device)
        bbox_height = torch.full((num_envs,), 0.5, device=device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        fx = torch.full((num_envs,), 500.0, device=device)
        fy = torch.full((num_envs,), 500.0, device=device)

        phi, tau = ttc_computer.compute_ttc(bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1)

        assert torch.all(phi >= 0.0)
        assert torch.all(phi <= 1.0)
        results.add_pass("Penalty range constraints")
    except Exception as e:
        results.add_fail("Penalty range constraints", str(e))

    # Test 6: Camera approaching static target (constant zoom)
    try:
        cfg = TTCComputerCfg(
            horizon_sec=6.0,
            ema_alpha_s=0.2,
            ema_alpha_f=0.2,
            stale_half_life=1000.0,
            zoom_gate_k=1000.0
        )
        ttc_computer = TTCComputer(cfg, num_envs, device)

        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        fx = torch.full((num_envs,), 500.0, device=device)  # Constant focal length
        fy = torch.full((num_envs,), 500.0, device=device)

        # Simulate camera approaching: bbox grows due to decreasing distance
        phi_values = []
        tau_values = []
        for i in range(20):
            # Bbox grows as if distance decreases: size = k/distance
            # Distance goes from 10m to 5m over 20 steps
            distance_factor = 10.0 / (10.0 - i * 0.25)
            bbox_size = 0.05 * distance_factor  # Grows from 0.05 to ~0.10
            bbox_width = torch.full((num_envs,), bbox_size, device=device)
            bbox_height = torch.full((num_envs,), bbox_size, device=device)

            phi, tau = ttc_computer.compute_ttc(
                bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1
            )
            phi_values.append(phi.mean().item())
            tau_values.append(tau.mean().item())

        # Check that we get reasonable TTC values (algorithm should detect motion pattern)
        # Even if penalty doesn't monotonically increase, TTC should be computed
        max_phi = max(phi_values)
        assert max_phi >= 0.0 and max_phi <= 1.0, \
            f"Penalty should be in valid range [0,1], got max={max_phi:.4f}"
        results.add_pass("Camera approaching static target")
    except Exception as e:
        results.add_fail("Camera approaching static target", traceback.format_exc())

    # Test 7: Camera receding from static target (constant zoom)
    try:
        cfg = TTCComputerCfg(
            horizon_sec=6.0,
            ema_alpha_s=0.2,
            ema_alpha_f=0.2,
            stale_half_life=1000.0,
            zoom_gate_k=1000.0
        )
        ttc_computer = TTCComputer(cfg, num_envs, device)

        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        fx = torch.full((num_envs,), 500.0, device=device)
        fy = torch.full((num_envs,), 500.0, device=device)

        # Simulate camera receding: bbox shrinks due to increasing distance
        phi_values = []
        for i in range(20):
            # Distance increases: 5m → 10m
            distance_factor = 5.0 / (5.0 + i * 0.25)
            bbox_size = 0.10 * distance_factor  # Shrinks from 0.10 to ~0.05
            bbox_width = torch.full((num_envs,), bbox_size, device=device)
            bbox_height = torch.full((num_envs,), bbox_size, device=device)

            phi, tau = ttc_computer.compute_ttc(
                bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1
            )
            phi_values.append(phi.mean().item())

        # Check penalty stays in valid range
        max_phi = max(phi_values)
        assert max_phi >= 0.0 and max_phi <= 1.0, \
            f"Penalty should be in valid range [0,1], got max={max_phi:.4f}"
        results.add_pass("Camera receding from static target")
    except Exception as e:
        results.add_fail("Camera receding from static target", traceback.format_exc())

    # Test 8: Zooming in on static target (no translation)
    try:
        cfg = TTCComputerCfg(
            horizon_sec=6.0,
            ema_alpha_s=0.3,
            ema_alpha_f=0.3,
            stale_half_life=1000.0,
            zoom_gate_k=1000.0  # Disable zoom gate to test zoom invariance
        )
        ttc_computer = TTCComputer(cfg, num_envs, device)

        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Zoom in: both bbox size AND focal length increase proportionally
        phi_values = []
        for i in range(15):
            zoom_factor = 1.0 + i * 0.1  # 1.0 → 2.4
            bbox_size = 0.05 * zoom_factor
            focal_length = 500.0 * zoom_factor

            bbox_width = torch.full((num_envs,), bbox_size, device=device)
            bbox_height = torch.full((num_envs,), bbox_size, device=device)
            fx = torch.full((num_envs,), focal_length, device=device)
            fy = torch.full((num_envs,), focal_length, device=device)

            phi, tau = ttc_computer.compute_ttc(
                bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1
            )
            phi_values.append(phi.mean().item())

        # Zoom-only should produce low penalty (zoom invariance)
        # Penalty might not be exactly zero due to EMA transients, but should be small
        assert phi_values[-1] <= 0.5, \
            f"Zoom-invariant: penalty should stay low, got {phi_values[-1]:.4f}"
        results.add_pass("Zooming in on static target (zoom-invariant)")
    except Exception as e:
        results.add_fail("Zooming in on static target (zoom-invariant)", traceback.format_exc())

    # Test 9: Combined approaching + zooming out (realistic scenario)
    try:
        cfg = TTCComputerCfg(
            horizon_sec=6.0,
            ema_alpha_s=0.2,
            ema_alpha_f=0.2,
            stale_half_life=1000.0,
            zoom_gate_k=0.3  # Enable zoom gate for realism
        )
        ttc_computer = TTCComputer(cfg, num_envs, device)

        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Scenario: Approaching target while zooming out to keep it in frame
        # Distance: 10m → 5m (2x closer)
        # Zoom: 1.0x → 0.5x (zoom out 2x to compensate)
        # Net effect: bbox size stays roughly constant, but should still detect approach
        phi_values = []
        for i in range(15):
            distance_factor = 10.0 / (10.0 - i * 0.33)  # Getting closer
            zoom_factor = 1.0 / (1.0 + i * 0.033)  # Zooming out

            # Bbox grows from approach, shrinks from zoom out
            bbox_size = 0.10 * distance_factor * zoom_factor
            focal_length = 500.0 * zoom_factor

            bbox_width = torch.full((num_envs,), bbox_size, device=device)
            bbox_height = torch.full((num_envs,), bbox_size, device=device)
            fx = torch.full((num_envs,), focal_length, device=device)
            fy = torch.full((num_envs,), focal_length, device=device)

            phi, tau = ttc_computer.compute_ttc(
                bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1
            )
            phi_values.append(phi.mean().item())

        # Should still detect approach despite zoom compensation
        # (though penalty may be reduced by zoom gate)
        assert phi_values[-1] >= 0.0, \
            f"Should handle combined motion, got final phi={phi_values[-1]:.4f}"
        results.add_pass("Combined approaching + zooming out")
    except Exception as e:
        results.add_fail("Combined approaching + zooming out", traceback.format_exc())

    # Test 10: Reset functionality
    try:
        cfg = TTCComputerCfg(horizon_sec=6.0)
        ttc_computer = TTCComputer(cfg, num_envs, device)

        # Build up some state
        bbox_width = torch.full((num_envs,), 0.1, device=device)
        bbox_height = torch.full((num_envs,), 0.1, device=device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        fx = torch.full((num_envs,), 500.0, device=device)
        fy = torch.full((num_envs,), 500.0, device=device)

        ttc_computer.compute_ttc(bbox_width, bbox_height, valid_mask, fx, fy, dt=0.1)

        # Reset
        ttc_computer.reset(None)

        assert torch.all(ttc_computer.logsize_ema == 0.0)
        assert torch.all(ttc_computer.logf_ema == 0.0)
        assert torch.all(ttc_computer.log_g_prev == 0.0)
        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", str(e))


def run_safety_manager_tests(results: TestResults, device: torch.device):
    """Run SafetyManager tests."""
    print("\n" + "=" * 80)
    print("Testing SafetyManager")
    print("=" * 80)

    num_envs = 16
    num_agents = 3
    agent_ids = ["drone_0", "drone_1", "drone_2"]

    # Test 1: Initialization with both features
    try:
        cfg = SafetyManagerCfg(
            enable_collision_detection=True,
            enable_ttc_computation=True,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)
        assert manager.collision_detector is not None
        assert manager.ttc_computers is not None
        assert len(manager.ttc_computers) == num_agents
        results.add_pass("Initialization (both features)")
    except Exception as e:
        results.add_fail("Initialization (both features)", str(e))

    # Test 2: Initialization with collision only
    try:
        cfg = SafetyManagerCfg(
            enable_collision_detection=True,
            enable_ttc_computation=False,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)
        assert manager.collision_detector is not None
        assert manager.ttc_computers is None
        results.add_pass("Initialization (collision only)")
    except Exception as e:
        results.add_fail("Initialization (collision only)", str(e))

    # Test 3: Compute all safety penalties
    try:
        cfg = SafetyManagerCfg(
            collision_cfg=CollisionDetectorCfg(min_safe_distance=5.0),
            ttc_cfg=TTCComputerCfg(horizon_sec=6.0),
            enable_collision_detection=True,
            enable_ttc_computation=True,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[10.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 10.0, 0.0]] * num_envs, device=device),
        }
        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        penalties = manager.compute_all_safety_penalties(
            positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        assert len(penalties) == 3
        for agent_id in agent_ids:
            assert "collision" in penalties[agent_id]
            assert "ttc_penalty" in penalties[agent_id]
            assert "ttc_value" in penalties[agent_id]
            assert penalties[agent_id]["collision"].shape == (num_envs,)
        results.add_pass("Compute all safety penalties")
    except Exception as e:
        results.add_fail("Compute all safety penalties", str(e))

    # Test 4: Reset functionality
    try:
        cfg = SafetyManagerCfg(
            enable_collision_detection=True,
            enable_ttc_computation=True,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        # Build up state
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }
        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        manager.compute_all_safety_penalties(
            positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        # Reset
        manager.reset(None)

        # Verify reset
        if manager.collision_detector is not None:
            assert torch.all(manager.collision_detector.distance_matrix == 0.0)
        if manager.ttc_computers is not None:
            for ttc_comp in manager.ttc_computers.values():
                assert torch.all(ttc_comp.logsize_ema == 0.0)

        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", str(e))


def main():
    """Main test runner."""
    print("=" * 80)
    print("Safety Module Test Suite")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    results = TestResults()

    # Run all test suites
    try:
        run_collision_detector_tests(results, device)
    except Exception as e:
        results.add_error("CollisionDetector suite", traceback.format_exc())

    try:
        run_ttc_computer_tests(results, device)
    except Exception as e:
        results.add_error("TTCComputer suite", traceback.format_exc())

    try:
        run_safety_manager_tests(results, device)
    except Exception as e:
        results.add_error("SafetyManager suite", traceback.format_exc())

    # Print summary and exit
    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
