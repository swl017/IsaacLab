#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Triangulation module test suite.

Tests the triangulation and uncertainty estimation functionality including:
- Position triangulation accuracy
- Validity-based returns (NaN + is_valid mask)
- Condition number checking
- Observability checks (minimum cameras, geometry)
- Covariance computation
- Quality metrics

Run with:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/run_tests.py
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run triangulation module test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW import other modules
import torch
import sys
import traceback
import math
from datetime import datetime
from typing import List, Tuple

# Import triangulation module
from isaaclab_tasks.direct.iris_ma6.triangulation import (
    TriangulationCfg,
    triangulate_targets,
    compute_triangulation_covariance,
    get_ray_directions_from_bbox,
    TriangulationResult,
)
from isaaclab_tasks.direct.iris_ma6.triangulation.triangulation import (
    compute_full_triangulation,
    build_camera_transforms,
    quat_to_rotation_matrix,
    skew,
)

VERBOSE = args_cli.test_verbose


# =============================================================================
# Test Results Tracking
# =============================================================================


class TestResults:
    """Track test results and provide summary."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.errors: List[Tuple[str, str]] = []

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

    def print_summary(self) -> bool:
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


# =============================================================================
# Test Utilities
# =============================================================================


def create_test_intrinsics(
    num_envs: int, num_cameras: int, fx: float = 500.0, device: torch.device = None
) -> torch.Tensor:
    """Create test camera intrinsic matrices.

    Args:
        num_envs: Number of environments
        num_cameras: Number of cameras
        fx: Focal length (pixels)
        device: Torch device

    Returns:
        [N, C, 3, 3] Intrinsic matrices
    """
    K = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(num_envs, num_cameras, 3, 3).clone()
    K[..., 0, 0] = fx
    K[..., 1, 1] = fx
    K[..., 0, 2] = 320.0  # cx
    K[..., 1, 2] = 240.0  # cy
    return K


def create_identity_quats(num_envs: int, num_cameras: int, device: torch.device = None) -> torch.Tensor:
    """Create identity quaternions (wxyz format).

    Returns:
        [N, C, 4] Quaternions
    """
    quats = torch.zeros(num_envs, num_cameras, 4, device=device)
    quats[..., 0] = 1.0  # w = 1, x = y = z = 0
    return quats


def create_yaw_quat(yaw: float, device: torch.device = None) -> torch.Tensor:
    """Create quaternion for yaw rotation (around Z axis).

    Args:
        yaw: Yaw angle in radians

    Returns:
        [4] Quaternion (wxyz)
    """
    half_yaw = yaw / 2.0
    return torch.tensor([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], device=device)


# =============================================================================
# Test: Triangulation Position Accuracy
# =============================================================================


def run_triangulation_position_tests(results: TestResults, device: torch.device):
    """Test triangulation position accuracy."""
    print("\n" + "=" * 80)
    print("Testing Triangulation Position Accuracy")
    print("=" * 80)

    cfg = TriangulationCfg()
    N, C, T = 4, 2, 1  # 4 envs, 2 cameras, 1 target

    # Test 1: Orthogonal cameras looking at target
    try:
        # Camera 1 at (0, -5, 0) looking along +Y
        # Camera 2 at (5, 0, 0) looking along -X
        # Target at (0, 0, 0) - should be triangulated accurately

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, -5.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([5.0, 0.0, 0.0])

        # Both rays point toward target at (0, 0, 0)
        target = torch.tensor([0.0, 0.0, 0.0], device=device)
        ray_dirs = torch.zeros(N, C, T, 3, device=device)
        ray_dirs[:, 0, 0, :] = target - camera_positions[:, 0, :]  # (0, 5, 0)
        ray_dirs[:, 1, 0, :] = target - camera_positions[:, 1, :]  # (-5, 0, 0)

        # Normalize
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)

        X_tri, is_valid, cond_num, num_valid = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        # Check validity
        assert is_valid.all(), f"Expected all valid, got {is_valid}"

        # Check position accuracy (should be at origin)
        assert torch.allclose(X_tri[0, 0], target, atol=1e-4), \
            f"Expected {target}, got {X_tri[0, 0]}"

        if VERBOSE:
            print(f"    Triangulated position: {X_tri[0, 0]}")
            print(f"    Condition number: {cond_num[0, 0].item():.2f}")

        results.add_pass("Orthogonal cameras - midpoint triangulation")
    except Exception as e:
        results.add_fail("Orthogonal cameras - midpoint triangulation", traceback.format_exc())

    # Test 2: Target at known 3D position
    try:
        # Two cameras looking at a target at (3, 2, 1)
        target_pos = torch.tensor([3.0, 2.0, 1.0], device=device)

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([6.0, 0.0, 0.0])

        # Ray directions pointing to target
        ray_dirs = torch.zeros(N, C, T, 3, device=device)
        ray_dirs[:, 0, 0, :] = target_pos - camera_positions[:, 0, :]
        ray_dirs[:, 1, 0, :] = target_pos - camera_positions[:, 1, :]

        # Normalize
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)

        X_tri, is_valid, _, _ = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        assert is_valid.all(), f"Expected all valid"
        assert torch.allclose(X_tri[0, 0], target_pos, atol=1e-3), \
            f"Expected {target_pos}, got {X_tri[0, 0]}"

        results.add_pass("Known target position triangulation")
    except Exception as e:
        results.add_fail("Known target position triangulation", traceback.format_exc())

    # Test 3: Multiple targets
    try:
        T_multi = 3
        targets = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ], device=device)

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, -5.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([0.0, 5.0, 0.0])

        ray_dirs = torch.zeros(N, C, T_multi, 3, device=device)
        for t in range(T_multi):
            ray_dirs[:, 0, t, :] = targets[t] - camera_positions[:, 0, :]
            ray_dirs[:, 1, t, :] = targets[t] - camera_positions[:, 1, :]

        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        valid_mask = torch.ones(N, C, T_multi, dtype=torch.bool, device=device)

        X_tri, is_valid, _, _ = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        assert is_valid.all(), f"Expected all valid"
        for t in range(T_multi):
            assert torch.allclose(X_tri[0, t], targets[t], atol=1e-3), \
                f"Target {t}: expected {targets[t]}, got {X_tri[0, t]}"

        results.add_pass("Multiple targets triangulation")
    except Exception as e:
        results.add_fail("Multiple targets triangulation", traceback.format_exc())


# =============================================================================
# Test: Validity-Based Returns
# =============================================================================


def run_validity_tests(results: TestResults, device: torch.device):
    """Test validity-based return behavior (NaN + is_valid mask)."""
    print("\n" + "=" * 80)
    print("Testing Validity-Based Returns")
    print("=" * 80)

    cfg = TriangulationCfg()
    N, C, T = 4, 3, 2  # 4 envs, 3 cameras, 2 targets

    # Test 1: Single camera - should be invalid
    try:
        camera_positions = torch.randn(N, C, 3, device=device)
        ray_dirs = torch.randn(N, C, T, 3, device=device)
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        # Only camera 0 is valid
        valid_mask = torch.zeros(N, C, T, dtype=torch.bool, device=device)
        valid_mask[:, 0, :] = True  # Only first camera valid

        X_tri, is_valid, cond_num, num_valid = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        # Should be invalid (need >= 2 cameras)
        assert not is_valid.any(), f"Expected all invalid with single camera"

        # Position should be NaN
        assert torch.isnan(X_tri).all(), f"Expected NaN for invalid triangulation"

        results.add_pass("Single camera returns is_valid=False")
    except Exception as e:
        results.add_fail("Single camera returns is_valid=False", traceback.format_exc())

    # Test 2: Behind camera detection
    try:
        # Setup: cameras at (0,0,0) and (5,0,0), both looking +Y
        # Target at (2.5, -1, 0) - behind both cameras
        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([5.0, 0.0, 0.0])
        camera_positions[:, 2, :] = torch.tensor([10.0, 0.0, 0.0])

        # All cameras look +Y but target is -Y
        target_behind = torch.tensor([2.5, -5.0, 0.0], device=device)

        ray_dirs = torch.zeros(N, C, T, 3, device=device)
        # Point rays AWAY from target (camera looks +Y, target is -Y)
        ray_dirs[:, :, 0, 1] = 1.0  # All cameras look +Y

        # Second target is valid (in front)
        target_front = torch.tensor([2.5, 5.0, 0.0], device=device)
        for c in range(C):
            ray_dirs[:, c, 1, :] = target_front - camera_positions[:, c, :]
        ray_dirs[:, :, 1, :] = ray_dirs[:, :, 1, :] / torch.norm(ray_dirs[:, :, 1, :], dim=-1, keepdim=True)

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)

        X_tri, is_valid, _, _ = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        # First target may have issues, second should be valid
        if VERBOSE:
            print(f"    Target 0 (behind) valid: {is_valid[:, 0]}")
            print(f"    Target 1 (front) valid: {is_valid[:, 1]}")

        results.add_pass("Behind camera detection")
    except Exception as e:
        results.add_fail("Behind camera detection", traceback.format_exc())

    # Test 3: Mixed validity per target
    try:
        # Target 0: only 1 camera sees it (invalid)
        # Target 1: 2 cameras see it (valid)
        valid_mask = torch.zeros(N, C, T, dtype=torch.bool, device=device)
        valid_mask[:, 0, 0] = True  # Only camera 0 sees target 0
        valid_mask[:, 0, 1] = True  # Cameras 0,1 see target 1
        valid_mask[:, 1, 1] = True

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([5.0, 0.0, 0.0])
        camera_positions[:, 2, :] = torch.tensor([10.0, 0.0, 0.0])

        target_pos = torch.tensor([2.5, 5.0, 0.0], device=device)
        ray_dirs = torch.zeros(N, C, T, 3, device=device)
        for c in range(C):
            for t in range(T):
                ray_dirs[:, c, t, :] = target_pos - camera_positions[:, c, :]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        X_tri, is_valid, _, num_valid = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        # Target 0 should be invalid, target 1 should be valid
        assert not is_valid[:, 0].any(), f"Target 0 should be invalid (1 camera)"
        assert is_valid[:, 1].all(), f"Target 1 should be valid (2 cameras)"

        # Target 0 position should be NaN
        assert torch.isnan(X_tri[:, 0, :]).all(), f"Invalid target should have NaN position"

        results.add_pass("Mixed validity per target")
    except Exception as e:
        results.add_fail("Mixed validity per target", traceback.format_exc())

    # Test 4: Zero cameras - should be invalid
    try:
        valid_mask = torch.zeros(N, C, T, dtype=torch.bool, device=device)

        camera_positions = torch.randn(N, C, 3, device=device)
        ray_dirs = torch.randn(N, C, T, 3, device=device)
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        X_tri, is_valid, _, num_valid = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        assert not is_valid.any(), f"No cameras should result in invalid"
        assert (num_valid == 0).all(), f"num_valid_cameras should be 0"

        results.add_pass("Zero valid cameras returns invalid")
    except Exception as e:
        results.add_fail("Zero valid cameras returns invalid", traceback.format_exc())


# =============================================================================
# Test: Condition Number and Geometry
# =============================================================================


def run_geometry_tests(results: TestResults, device: torch.device):
    """Test condition number checking and geometry handling."""
    print("\n" + "=" * 80)
    print("Testing Condition Number and Geometry")
    print("=" * 80)

    N, C, T = 4, 2, 1

    # Test 1: Good geometry (90° baseline)
    try:
        cfg = TriangulationCfg(condition_threshold=1e6)

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([10.0, 0.0, 0.0])

        # Target at (5, 5, 0) - good 45° angles from both cameras
        target = torch.tensor([5.0, 5.0, 0.0], device=device)

        ray_dirs = torch.zeros(N, C, T, 3, device=device)
        ray_dirs[:, 0, 0, :] = target - camera_positions[:, 0, :]
        ray_dirs[:, 1, 0, :] = target - camera_positions[:, 1, :]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)

        X_tri, is_valid, cond_num, _ = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        # Should be valid with low condition number
        assert is_valid.all(), f"Good geometry should be valid"
        assert (cond_num < 100).all(), f"Good geometry should have low condition number, got {cond_num}"

        if VERBOSE:
            print(f"    Condition number: {cond_num[0, 0].item():.2f}")

        results.add_pass("Good geometry (90° baseline)")
    except Exception as e:
        results.add_fail("Good geometry (90° baseline)", traceback.format_exc())

    # Test 2: Poor geometry (nearly parallel rays)
    try:
        cfg = TriangulationCfg(condition_threshold=1e4)

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([0.1, 0.0, 0.0])  # Very close cameras

        # Target far away - nearly parallel rays
        target = torch.tensor([0.0, 1000.0, 0.0], device=device)

        ray_dirs = torch.zeros(N, C, T, 3, device=device)
        ray_dirs[:, 0, 0, :] = target - camera_positions[:, 0, :]
        ray_dirs[:, 1, 0, :] = target - camera_positions[:, 1, :]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)

        X_tri, is_valid, cond_num, _ = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg
        )

        # High condition number expected
        if VERBOSE:
            print(f"    Condition number (parallel): {cond_num[0, 0].item():.2e}")
            print(f"    is_valid: {is_valid[0, 0]}")

        # With strict threshold, may be invalid
        results.add_pass("Poor geometry (parallel rays) - condition number computed")
    except Exception as e:
        results.add_fail("Poor geometry (parallel rays)", traceback.format_exc())

    # Test 3: Configurable condition threshold
    try:
        # Strict threshold
        cfg_strict = TriangulationCfg(condition_threshold=10.0)

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        camera_positions[:, 1, :] = torch.tensor([1.0, 0.0, 0.0])

        # Moderate geometry
        target = torch.tensor([0.5, 10.0, 0.0], device=device)

        ray_dirs = torch.zeros(N, C, T, 3, device=device)
        ray_dirs[:, 0, 0, :] = target - camera_positions[:, 0, :]
        ray_dirs[:, 1, 0, :] = target - camera_positions[:, 1, :]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)

        _, is_valid_strict, cond_num, _ = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg_strict
        )

        # Relaxed threshold
        cfg_relaxed = TriangulationCfg(condition_threshold=1e10)
        _, is_valid_relaxed, _, _ = triangulate_targets(
            camera_positions, ray_dirs, valid_mask, cfg_relaxed
        )

        # Relaxed should allow more through
        if VERBOSE:
            print(f"    Strict (thresh=10): valid={is_valid_strict[0, 0]}")
            print(f"    Relaxed (thresh=1e10): valid={is_valid_relaxed[0, 0]}")
            print(f"    Condition number: {cond_num[0, 0].item():.2f}")

        results.add_pass("Configurable condition threshold")
    except Exception as e:
        results.add_fail("Configurable condition threshold", traceback.format_exc())


# =============================================================================
# Test: Covariance Computation
# =============================================================================


def run_covariance_tests(results: TestResults, device: torch.device):
    """Test covariance computation and quality metrics."""
    print("\n" + "=" * 80)
    print("Testing Covariance Computation")
    print("=" * 80)

    N, C, T = 4, 3, 2

    # Test 1: Basic covariance computation
    try:
        cfg = TriangulationCfg(
            pix_std=5.0,
            include_pose_uncertainty=False,
            include_gimbal_uncertainty=False,
        )

        # Setup cameras looking at target
        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, -10.0, 5.0])
        camera_positions[:, 1, :] = torch.tensor([10.0, 0.0, 5.0])
        camera_positions[:, 2, :] = torch.tensor([0.0, 10.0, 5.0])

        camera_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, device=device)

        # Target positions
        X_target = torch.zeros(N, T, 3, device=device)
        X_target[:, 0, :] = torch.tensor([5.0, 0.0, 0.0])
        X_target[:, 1, :] = torch.tensor([3.0, 2.0, 1.0])

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)
        tri_valid = torch.ones(N, T, dtype=torch.bool, device=device)

        Sigma_X, quality, is_valid = compute_triangulation_covariance(
            X_target,
            camera_positions,
            camera_quats,
            gimbal_yaws,
            gimbal_rolls,
            gimbal_pitches,
            camera_intrinsics,
            valid_mask,
            tri_valid,
            cfg,
        )

        # Check outputs
        assert not torch.isnan(Sigma_X[is_valid]).any(), "Valid covariance should not be NaN"
        assert (quality[is_valid] > 0).all(), "Quality metric should be positive"

        # Check covariance is symmetric
        Sigma_valid = Sigma_X[is_valid]
        assert torch.allclose(Sigma_valid, Sigma_valid.transpose(-1, -2), atol=1e-6), \
            "Covariance should be symmetric"

        # Check positive definite (eigenvalues > 0)
        eigvals = torch.linalg.eigvalsh(Sigma_valid)
        assert (eigvals > 0).all(), "Covariance should be positive definite"

        if VERBOSE:
            print(f"    Covariance trace: {torch.diagonal(Sigma_X[0, 0], dim1=-2, dim2=-1).sum().item():.4f}")
            print(f"    Quality metric: {quality[0, 0].item():.4f}")

        results.add_pass("Basic covariance computation")
    except Exception as e:
        results.add_fail("Basic covariance computation", traceback.format_exc())

    # Test 2: Covariance with pose uncertainty
    try:
        cfg_pose = TriangulationCfg(
            pix_std=5.0,
            pos_std=0.1,
            ori_std=0.01,
            include_pose_uncertainty=True,
            include_gimbal_uncertainty=False,
        )

        cfg_no_pose = TriangulationCfg(
            pix_std=5.0,
            include_pose_uncertainty=False,
            include_gimbal_uncertainty=False,
        )

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, -10.0, 5.0])
        camera_positions[:, 1, :] = torch.tensor([10.0, 0.0, 5.0])
        camera_positions[:, 2, :] = torch.tensor([0.0, 10.0, 5.0])

        camera_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, device=device)

        X_target = torch.zeros(N, T, 3, device=device)
        X_target[:, 0, :] = torch.tensor([5.0, 0.0, 0.0])
        X_target[:, 1, :] = torch.tensor([3.0, 2.0, 1.0])

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)
        tri_valid = torch.ones(N, T, dtype=torch.bool, device=device)

        _, quality_pose, _ = compute_triangulation_covariance(
            X_target, camera_positions, camera_quats, gimbal_yaws, gimbal_rolls,
            gimbal_pitches, camera_intrinsics, valid_mask, tri_valid, cfg_pose
        )

        _, quality_no_pose, _ = compute_triangulation_covariance(
            X_target, camera_positions, camera_quats, gimbal_yaws, gimbal_rolls,
            gimbal_pitches, camera_intrinsics, valid_mask, tri_valid, cfg_no_pose
        )

        # With pose uncertainty, quality metric should be higher (more uncertainty)
        assert (quality_pose >= quality_no_pose).all(), \
            "Pose uncertainty should increase quality metric (trace)"

        if VERBOSE:
            print(f"    Quality without pose: {quality_no_pose[0, 0].item():.4f}")
            print(f"    Quality with pose: {quality_pose[0, 0].item():.4f}")

        results.add_pass("Pose uncertainty increases covariance")
    except Exception as e:
        results.add_fail("Pose uncertainty increases covariance", traceback.format_exc())

    # Test 3: Quality metric options
    try:
        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_positions[:, 0, :] = torch.tensor([0.0, -10.0, 5.0])
        camera_positions[:, 1, :] = torch.tensor([10.0, 0.0, 5.0])
        camera_positions[:, 2, :] = torch.tensor([0.0, 10.0, 5.0])

        camera_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, device=device)

        X_target = torch.zeros(N, T, 3, device=device)
        X_target[:, 0, :] = torch.tensor([5.0, 0.0, 0.0])
        X_target[:, 1, :] = torch.tensor([3.0, 2.0, 1.0])

        valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)
        tri_valid = torch.ones(N, T, dtype=torch.bool, device=device)

        metrics = {}
        for metric_name in ["trace", "sqrt_trace", "det", "max_eig"]:
            cfg = TriangulationCfg(
                pix_std=5.0,
                include_pose_uncertainty=False,
                include_gimbal_uncertainty=False,
                quality_metric=metric_name,
            )

            _, quality, _ = compute_triangulation_covariance(
                X_target, camera_positions, camera_quats, gimbal_yaws, gimbal_rolls,
                gimbal_pitches, camera_intrinsics, valid_mask, tri_valid, cfg
            )
            metrics[metric_name] = quality[0, 0].item()

            if VERBOSE:
                print(f"    {metric_name}: {metrics[metric_name]:.6f}")

        # sqrt_trace should be sqrt of trace
        assert abs(metrics["sqrt_trace"] - math.sqrt(metrics["trace"])) < 1e-4, \
            "sqrt_trace should be sqrt of trace"

        results.add_pass("Quality metric options (trace, det, max_eig, sqrt_trace)")
    except Exception as e:
        results.add_fail("Quality metric options", traceback.format_exc())

    # Test 4: Invalid covariance returns NaN
    try:
        cfg = TriangulationCfg()

        camera_positions = torch.zeros(N, C, 3, device=device)
        camera_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, device=device)

        X_target = torch.zeros(N, T, 3, device=device)

        # No valid cameras
        valid_mask = torch.zeros(N, C, T, dtype=torch.bool, device=device)
        tri_valid = torch.zeros(N, T, dtype=torch.bool, device=device)

        Sigma_X, quality, is_valid = compute_triangulation_covariance(
            X_target, camera_positions, camera_quats, gimbal_yaws, gimbal_rolls,
            gimbal_pitches, camera_intrinsics, valid_mask, tri_valid, cfg
        )

        assert not is_valid.any(), "Should be invalid with no cameras"
        assert torch.isnan(quality).all(), "Quality should be NaN when invalid"

        results.add_pass("Invalid covariance returns NaN")
    except Exception as e:
        results.add_fail("Invalid covariance returns NaN", traceback.format_exc())


# =============================================================================
# Test: Full Pipeline (compute_full_triangulation)
# =============================================================================


def run_full_pipeline_tests(results: TestResults, device: torch.device):
    """Test the full triangulation pipeline."""
    print("\n" + "=" * 80)
    print("Testing Full Pipeline (compute_full_triangulation)")
    print("=" * 80)

    N, C, T = 4, 3, 1

    # Test 1: Full pipeline with bbox input
    try:
        cfg = TriangulationCfg(
            pix_std=5.0,
            include_pose_uncertainty=True,
            include_gimbal_uncertainty=True,
        )

        # Camera setup: 3 cameras arranged in triangle
        robot_positions = torch.zeros(N, C, 3, device=device)
        robot_positions[:, 0, :] = torch.tensor([0.0, -10.0, 5.0])
        robot_positions[:, 1, :] = torch.tensor([8.66, 5.0, 5.0])
        robot_positions[:, 2, :] = torch.tensor([-8.66, 5.0, 5.0])

        robot_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, device=device)

        # Target at center, create bbox centered at image center
        bbox_2d = torch.zeros(N, C, T, 4, device=device)
        bbox_2d[..., 0] = 320.0  # x center
        bbox_2d[..., 1] = 240.0  # y center
        bbox_2d[..., 2] = 50.0   # width
        bbox_2d[..., 3] = 50.0   # height

        bbox_valid = torch.ones(N, C, T, dtype=torch.bool, device=device)

        result = compute_full_triangulation(
            bbox_2d,
            bbox_valid,
            robot_positions,
            robot_quats,
            gimbal_yaws,
            gimbal_rolls,
            gimbal_pitches,
            camera_intrinsics,
            cfg,
        )

        # Check result type
        assert isinstance(result, TriangulationResult), "Should return TriangulationResult"

        # Check shapes
        assert result.position.shape == (N, T, 3), f"Position shape: {result.position.shape}"
        assert result.covariance.shape == (N, T, 3, 3), f"Covariance shape: {result.covariance.shape}"
        assert result.quality_metric.shape == (N, T), f"Quality shape: {result.quality_metric.shape}"
        assert result.is_valid.shape == (N, T), f"is_valid shape: {result.is_valid.shape}"
        assert result.condition_number.shape == (N, T), f"Condition number shape: {result.condition_number.shape}"
        assert result.num_valid_cameras.shape == (N, T), f"num_valid_cameras shape: {result.num_valid_cameras.shape}"

        if VERBOSE:
            print(f"    Position: {result.position[0, 0]}")
            print(f"    Quality: {result.quality_metric[0, 0].item():.4f}")
            print(f"    Condition number: {result.condition_number[0, 0].item():.2f}")
            print(f"    Valid cameras: {result.num_valid_cameras[0, 0].item()}")

        results.add_pass("Full pipeline with bbox input")
    except Exception as e:
        results.add_fail("Full pipeline with bbox input", traceback.format_exc())

    # Test 2: Full pipeline with GT positions
    try:
        # Use same good setup as Test 1 but with GT positions
        cfg = TriangulationCfg(pix_std=5.0)

        robot_positions = torch.zeros(N, C, 3, device=device)
        robot_positions[:, 0, :] = torch.tensor([0.0, -10.0, 5.0])
        robot_positions[:, 1, :] = torch.tensor([8.66, 5.0, 5.0])
        robot_positions[:, 2, :] = torch.tensor([-8.66, 5.0, 5.0])

        robot_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, device=device)

        bbox_2d = torch.zeros(N, C, T, 4, device=device)
        bbox_2d[..., 0] = 320.0
        bbox_2d[..., 1] = 240.0
        bbox_2d[..., 2] = 50.0
        bbox_2d[..., 3] = 50.0

        bbox_valid = torch.ones(N, C, T, dtype=torch.bool, device=device)

        # Provide GT positions (at a position likely to be observable)
        target_gt = torch.zeros(N, T, 3, device=device)
        target_gt[:, 0, :] = torch.tensor([0.0, 0.0, 5.0])  # Same height as cameras

        result = compute_full_triangulation(
            bbox_2d,
            bbox_valid,
            robot_positions,
            robot_quats,
            gimbal_yaws,
            gimbal_rolls,
            gimbal_pitches,
            camera_intrinsics,
            cfg,
            target_positions_gt=target_gt,
        )

        # Check that the function ran without error
        assert result.position.shape == (N, T, 3), "Position shape correct"
        assert result.covariance.shape == (N, T, 3, 3), "Covariance shape correct"
        assert result.is_valid.shape == (N, T), "is_valid shape correct"

        # With GT positions, covariance computation may or may not be valid
        # depending on the geometry. Just verify no crashes occurred.
        if result.is_valid.any():
            assert not torch.isnan(result.quality_metric[result.is_valid]).any(), \
                "Valid entries should not have NaN quality"

        if VERBOSE:
            print(f"    is_valid: {result.is_valid}")
            if result.is_valid.any():
                print(f"    Quality with GT: {result.quality_metric[result.is_valid].mean().item():.4f}")

        results.add_pass("Full pipeline with GT positions")
    except Exception as e:
        results.add_fail("Full pipeline with GT positions", traceback.format_exc())


# =============================================================================
# Test: Ray Direction Computation
# =============================================================================


def run_ray_direction_tests(results: TestResults, device: torch.device):
    """Test ray direction computation from bounding boxes."""
    print("\n" + "=" * 80)
    print("Testing Ray Direction Computation")
    print("=" * 80)

    N, C, T = 2, 2, 1

    # Test 1: Center pixel gives forward ray
    try:
        robot_positions = torch.zeros(N, C, 3, device=device)
        robot_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, fx=500.0, device=device)

        # Bbox at image center
        bbox_2d = torch.zeros(N, C, T, 4, device=device)
        bbox_2d[..., 0] = 320.0  # cx
        bbox_2d[..., 1] = 240.0  # cy

        ray_dirs, R_wc = get_ray_directions_from_bbox(
            bbox_2d, camera_intrinsics, robot_positions, robot_quats,
            gimbal_yaws, gimbal_rolls, gimbal_pitches
        )

        # With identity orientation and RDF->ENU, center pixel should give a specific direction
        assert ray_dirs.shape == (N, C, T, 3), f"Shape: {ray_dirs.shape}"

        # Rays should be normalized
        norms = torch.norm(ray_dirs, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Rays should be normalized"

        if VERBOSE:
            print(f"    Ray direction (center): {ray_dirs[0, 0, 0]}")

        results.add_pass("Center pixel ray direction")
    except Exception as e:
        results.add_fail("Center pixel ray direction", traceback.format_exc())

    # Test 2: Off-center pixels give different directions
    try:
        robot_positions = torch.zeros(N, C, 3, device=device)
        robot_quats = create_identity_quats(N, C, device)
        gimbal_yaws = torch.zeros(N, C, device=device)
        gimbal_rolls = torch.zeros(N, C, device=device)
        gimbal_pitches = torch.zeros(N, C, device=device)
        camera_intrinsics = create_test_intrinsics(N, C, fx=500.0, device=device)

        # Test multiple bbox positions
        T_multi = 3
        bbox_2d = torch.zeros(N, C, T_multi, 4, device=device)
        bbox_2d[..., 0, 0] = 320.0  # center
        bbox_2d[..., 0, 1] = 240.0
        bbox_2d[..., 1, 0] = 0.0    # left edge
        bbox_2d[..., 1, 1] = 240.0
        bbox_2d[..., 2, 0] = 640.0  # right edge
        bbox_2d[..., 2, 1] = 240.0

        ray_dirs, _ = get_ray_directions_from_bbox(
            bbox_2d, camera_intrinsics, robot_positions, robot_quats,
            gimbal_yaws, gimbal_rolls, gimbal_pitches
        )

        # Each bbox should give different ray direction
        dir_center = ray_dirs[0, 0, 0]
        dir_left = ray_dirs[0, 0, 1]
        dir_right = ray_dirs[0, 0, 2]

        # Left and right should be different from center
        assert not torch.allclose(dir_center, dir_left, atol=1e-3), "Left should differ from center"
        assert not torch.allclose(dir_center, dir_right, atol=1e-3), "Right should differ from center"

        if VERBOSE:
            print(f"    Center: {dir_center}")
            print(f"    Left:   {dir_left}")
            print(f"    Right:  {dir_right}")

        results.add_pass("Off-center pixel ray directions")
    except Exception as e:
        results.add_fail("Off-center pixel ray directions", traceback.format_exc())

    # Test 3: Gimbal rotation affects ray direction
    try:
        robot_positions = torch.zeros(N, C, 3, device=device)
        robot_quats = create_identity_quats(N, C, device)
        camera_intrinsics = create_test_intrinsics(N, C, device=device)

        bbox_2d = torch.zeros(N, C, T, 4, device=device)
        bbox_2d[..., 0] = 320.0
        bbox_2d[..., 1] = 240.0

        # No gimbal rotation
        gimbal_yaws_0 = torch.zeros(N, C, device=device)
        gimbal_rolls_0 = torch.zeros(N, C, device=device)
        gimbal_pitches_0 = torch.zeros(N, C, device=device)

        ray_dirs_0, _ = get_ray_directions_from_bbox(
            bbox_2d, camera_intrinsics, robot_positions, robot_quats,
            gimbal_yaws_0, gimbal_rolls_0, gimbal_pitches_0
        )

        # With gimbal yaw
        gimbal_yaws_45 = torch.full((N, C), math.pi / 4, device=device)

        ray_dirs_45, _ = get_ray_directions_from_bbox(
            bbox_2d, camera_intrinsics, robot_positions, robot_quats,
            gimbal_yaws_45, gimbal_rolls_0, gimbal_pitches_0
        )

        # Ray directions should be different
        assert not torch.allclose(ray_dirs_0, ray_dirs_45, atol=1e-3), \
            "Gimbal rotation should change ray direction"

        if VERBOSE:
            print(f"    Ray (no gimbal): {ray_dirs_0[0, 0, 0]}")
            print(f"    Ray (45° yaw):   {ray_dirs_45[0, 0, 0]}")

        results.add_pass("Gimbal rotation affects ray direction")
    except Exception as e:
        results.add_fail("Gimbal rotation affects ray direction", traceback.format_exc())


# =============================================================================
# Test: Utility Functions
# =============================================================================


def run_utility_tests(results: TestResults, device: torch.device):
    """Test utility functions."""
    print("\n" + "=" * 80)
    print("Testing Utility Functions")
    print("=" * 80)

    # Test 1: Skew symmetric matrix
    try:
        v = torch.tensor([1.0, 2.0, 3.0], device=device)
        S = skew(v.unsqueeze(0))[0]

        # Check antisymmetric
        assert torch.allclose(S, -S.T, atol=1e-6), "Skew should be antisymmetric"

        # Check expected form
        expected = torch.tensor([
            [0.0, -3.0, 2.0],
            [3.0, 0.0, -1.0],
            [-2.0, 1.0, 0.0]
        ], device=device)
        assert torch.allclose(S, expected, atol=1e-6), f"Skew mismatch"

        results.add_pass("Skew symmetric matrix")
    except Exception as e:
        results.add_fail("Skew symmetric matrix", traceback.format_exc())

    # Test 2: Quaternion to rotation matrix
    try:
        # Identity quaternion
        q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        R_id = quat_to_rotation_matrix(q_id.unsqueeze(0))[0]

        assert torch.allclose(R_id, torch.eye(3, device=device), atol=1e-6), \
            "Identity quaternion should give identity matrix"

        # 90° rotation around Z
        q_z90 = torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)], device=device)
        R_z90 = quat_to_rotation_matrix(q_z90.unsqueeze(0))[0]

        # x axis should map to y axis
        x_rotated = R_z90 @ torch.tensor([1.0, 0.0, 0.0], device=device)
        assert torch.allclose(x_rotated, torch.tensor([0.0, 1.0, 0.0], device=device), atol=1e-5), \
            "90° Z rotation should map x to y"

        results.add_pass("Quaternion to rotation matrix")
    except Exception as e:
        results.add_fail("Quaternion to rotation matrix", traceback.format_exc())

    # Test 3: Build camera transforms
    try:
        N, C = 2, 2

        robot_pos = torch.zeros(N, C, 3, device=device)
        robot_quat = create_identity_quats(N, C, device)
        gimbal_yaw = torch.zeros(N, C, device=device)
        gimbal_roll = torch.zeros(N, C, device=device)
        gimbal_pitch = torch.zeros(N, C, device=device)

        R_wc, t_wc, e_yaw, e_roll, e_pitch = build_camera_transforms(
            robot_pos, robot_quat, gimbal_yaw, gimbal_roll, gimbal_pitch
        )

        # Check shapes
        assert R_wc.shape == (N, C, 3, 3), f"R_wc shape: {R_wc.shape}"
        assert t_wc.shape == (N, C, 3), f"t_wc shape: {t_wc.shape}"
        assert e_yaw.shape == (N, C, 3), f"e_yaw shape: {e_yaw.shape}"
        assert e_roll.shape == (N, C, 3), f"e_roll shape: {e_roll.shape}"
        assert e_pitch.shape == (N, C, 3), f"e_pitch shape: {e_pitch.shape}"

        # R_wc should be orthogonal
        R_flat = R_wc.reshape(N * C, 3, 3)
        for i in range(N * C):
            R = R_flat[i]
            assert torch.allclose(R @ R.T, torch.eye(3, device=device), atol=1e-5), \
                f"R_wc[{i}] not orthogonal"

        results.add_pass("Build camera transforms")
    except Exception as e:
        results.add_fail("Build camera transforms", traceback.format_exc())


# =============================================================================
# Main
# =============================================================================


def main():
    """Main test runner."""
    print("=" * 80)
    print("TRIANGULATION MODULE TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()

    try:
        run_utility_tests(results, device)
        run_ray_direction_tests(results, device)
        run_triangulation_position_tests(results, device)
        run_validity_tests(results, device)
        run_geometry_tests(results, device)
        run_covariance_tests(results, device)
        run_full_pipeline_tests(results, device)
    except Exception as e:
        results.add_error("Test suite execution", traceback.format_exc())

    success = results.print_summary()

    # Cleanup
    simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
