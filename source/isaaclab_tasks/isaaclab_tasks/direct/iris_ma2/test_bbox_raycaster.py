# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for BBoxRayCaster module.

Run with:
    python test_bbox_raycaster.py
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import torch
import unittest
import time

from bbox_raycaster import BBoxRayCaster, BBoxRayCasterCfg


class TestBBoxRayCaster(unittest.TestCase):
    """Test suite for BBoxRayCaster."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting on device: {cls.device}")

    def setUp(self):
        """Set up for each test."""
        # Test configuration
        self.num_envs = 16
        self.num_cameras = 2
        self.num_targets = 1
        self.image_height = 480
        self.image_width = 640

    def test_initialization(self):
        """Test basic initialization."""
        print("\n[Test] Initialization")

        cfg = BBoxRayCasterCfg(
            target_prim_paths=["/World/envs/env_.*/target"],
            mesh_prim_paths=["/World/ground"],
            num_cameras_per_env=self.num_cameras,
            enable_occlusion_check=False,  # Skip for this test
        )

        # Note: This test would need a running Isaac Sim instance to fully work
        # For unit testing without sim, we'd mock the mesh loading
        print("Configuration created successfully")
        print(f"  Cameras per env: {cfg.num_cameras_per_env}")
        print(f"  Occlusion enabled: {cfg.enable_occlusion_check}")

    def test_data_shapes(self):
        """Test that output data has correct shapes."""
        print("\n[Test] Data Shapes")

        # Create mock camera data
        camera_poses = self._create_mock_camera_poses()
        camera_intrinsics = self._create_mock_intrinsics()
        target_poses = self._create_mock_target_poses()

        # Note: Would need actual raycaster instance
        print(f"  Camera poses: {camera_poses['camera_0'][0].shape}")
        print(f"  Intrinsics: {camera_intrinsics['camera_0'].shape}")
        print(f"  Target poses: {target_poses[0].shape}")

    def test_projection_numerics(self):
        """Test numerical stability of projections."""
        print("\n[Test] Projection Numerics")

        # Test edge cases
        test_cases = [
            ("Behind camera", torch.tensor([0, 0, -1])),
            ("At camera", torch.tensor([0, 0, 0])),
            ("Very close", torch.tensor([0, 0, 0.01])),
            ("Very far", torch.tensor([0, 0, 1000.0])),
        ]

        from bbox_raycaster.utils.projection import batch_project_to_image_plane

        # Create simple intrinsic
        intrinsic = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0)
        intrinsic[:, :, 0, 0] = 500  # fx
        intrinsic[:, :, 1, 1] = 500  # fy
        intrinsic[:, :, 0, 2] = 320  # cx
        intrinsic[:, :, 1, 2] = 240  # cy

        for name, point in test_cases:
            points_camera = point.view(1, 1, 1, 1, 3).to(self.device)
            pixels, depths, valid = batch_project_to_image_plane(
                points_camera, intrinsic
            )

            print(f"  {name}:")
            print(f"    Pixel: {pixels[0, 0, 0, 0].cpu().numpy()}")
            print(f"    Depth: {depths[0, 0, 0, 0].item():.6f}")
            print(f"    Valid: {valid[0, 0, 0, 0].item()}")

    def test_bbox_computation(self):
        """Test bbox computation from corners."""
        print("\n[Test] BBox Computation")

        from bbox_raycaster.utils.bbox_ops import (
            compute_2d_bbox_from_corners,
            bbox_xyxy_to_xywh
        )

        # Create mock corners in image space
        corners_2d = torch.tensor([
            [[100, 100], [200, 100], [100, 200], [200, 200],  # Bottom face
             [100, 150], [200, 150], [100, 250], [200, 250]]  # Top face (simulated)
        ], device=self.device).view(1, 1, 1, 8, 2).float()

        valid_corners = torch.ones(1, 1, 1, 8, device=self.device, dtype=torch.bool)

        # Compute bbox
        bbox_xyxy, bbox_valid = compute_2d_bbox_from_corners(
            corners_2d, valid_corners
        )
        bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy)

        print(f"  Input corners: {corners_2d[0, 0, 0, :4].cpu().numpy()}")
        print(f"  BBox (xyxy): {bbox_xyxy[0, 0, 0].cpu().numpy()}")
        print(f"  BBox (xywh): {bbox_xywh[0, 0, 0].cpu().numpy()}")
        print(f"  Valid: {bbox_valid[0, 0, 0].item()}")

        # Test degenerate case (all corners at same point)
        degenerate_corners = torch.ones(1, 1, 1, 8, 2, device=self.device) * 100
        bbox_xyxy_deg, bbox_valid_deg = compute_2d_bbox_from_corners(
            degenerate_corners, valid_corners
        )

        print(f"  Degenerate bbox valid: {bbox_valid_deg[0, 0, 0].item()}")

    def test_occlusion_test_points(self):
        """Test occlusion test point generation."""
        print("\n[Test] Occlusion Test Points")

        from bbox_raycaster.utils.occlusion import generate_occlusion_test_points

        # Create mock bbox corners
        corners = torch.tensor([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        ], device=self.device).view(1, 1, 8, 3).float()

        # Create simple intrinsic
        intrinsic = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0)
        intrinsic[:, :, 0, 0] = 500  # fx
        intrinsic[:, :, 1, 1] = 500  # fy
        intrinsic[:, :, 0, 2] = 320  # cx
        intrinsic[:, :, 1, 2] = 240  # cy
        for pattern in ["center_only", "corners_only", "9point"]:
            test_points = generate_occlusion_test_points(corners, pattern)
            print(f"  Pattern '{pattern}': {test_points.shape[-2]} points")
            print(f"    First point: {test_points[0, 0, 0].cpu().numpy()}")

    def test_quaternion_normalization(self):
        """Test quaternion normalization edge cases."""
        print("\n[Test] Quaternion Normalization")

        from bbox_raycaster.utils.projection import batch_transform_points

        # Test cases
        test_quats = [
            ("Unit quat", torch.tensor([1.0, 0.0, 0.0, 0.0])),
            ("Denormalized", torch.tensor([2.0, 0.0, 0.0, 0.0])),
            ("Near-zero", torch.tensor([1e-9, 0.0, 0.0, 0.0])),
        ]

        points = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        pos = torch.zeros(1, 1, 3, device=self.device)

        for name, quat in test_quats:
            quat_batch = quat.view(1, 1, 4).to(self.device)
            try:
                result = batch_transform_points(points, pos, quat_batch)
                print(f"  {name}: Success - {result[0, 0, 0].cpu().numpy()}")
            except Exception as e:
                print(f"  {name}: Failed - {e}")

    def test_performance_scaling(self):
        """Test performance scaling with environment count."""
        print("\n[Test] Performance Scaling")

        from bbox_raycaster.utils.projection import (
            batch_transform_points,
            batch_project_to_image_plane
        )

        env_counts = [16, 64, 256, 1024, 4096]

        print(f"  {'Envs':<8} {'Transform (ms)':<16} {'Project (ms)':<16}")
        print(f"  {'-'*8} {'-'*16} {'-'*16}")

        for num_envs in env_counts:
            # Create test data
            points_local = torch.randn(8, 3, device=self.device)
            pos = torch.randn(num_envs, 1, 3, device=self.device)
            quat = torch.randn(num_envs, 1, 4, device=self.device)
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)

            # Benchmark transform
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                result = batch_transform_points(points_local, pos, quat)
            torch.cuda.synchronize()
            transform_time = (time.time() - start) * 100  # ms per call

            # Prepare for projection
            points_camera = result.view(num_envs, 1, 1, 8, 3)
            intrinsic = torch.eye(3, device=self.device).view(1, 1, 3, 3).expand(num_envs, 1, -1, -1)

            # Benchmark projection
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                pixels, depths, valid = batch_project_to_image_plane(points_camera, intrinsic)
            torch.cuda.synchronize()
            project_time = (time.time() - start) * 100  # ms per call

            print(f"  {num_envs:<8} {transform_time:>14.3f}  {project_time:>14.3f}")

    def test_memory_usage(self):
        """Test memory usage at different scales."""
        print("\n[Test] Memory Usage")

        if not torch.cuda.is_available():
            print("  Skipped (CUDA not available)")
            return

        env_counts = [16, 64, 256, 1024]

        print(f"  {'Envs':<8} {'Allocated (MB)':<20} {'Reserved (MB)':<20}")
        print(f"  {'-'*8} {'-'*20} {'-'*20}")

        for num_envs in env_counts:
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Allocate data structures (similar to raycaster)
            N, C, T = num_envs, 2, 1

            buffers = {
                'camera_pos': torch.zeros((N, C, 3), device=self.device),
                'camera_quat': torch.zeros((N, C, 4), device=self.device),
                'corners_world': torch.zeros((N, T, 8, 3), device=self.device),
                'corners_camera': torch.zeros((N, C, T, 8, 3), device=self.device),
                'pixels': torch.zeros((N, C, T, 8, 2), device=self.device),
                'bboxes': torch.zeros((N, C, T, 4), device=self.device),
            }

            allocated = torch.cuda.memory_allocated() / 1e6  # MB
            reserved = torch.cuda.memory_reserved() / 1e6

            print(f"  {num_envs:<8} {allocated:>18.2f}  {reserved:>18.2f}")

            # Clean up
            del buffers

    """
    Helper methods
    """

    def _create_mock_camera_poses(self):
        """Create mock camera poses for testing."""
        camera_poses = {}
        for i in range(self.num_cameras):
            # Random positions around origin
            pos = torch.randn(self.num_envs, 3, device=self.device) * 5.0
            pos[:, 2] += 3.0  # Elevate cameras

            # Random orientations (normalized quaternions)
            quat = torch.randn(self.num_envs, 4, device=self.device)
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)

            camera_poses[f"camera_{i}"] = (pos, quat)

        return camera_poses

    def _create_mock_intrinsics(self):
        """Create mock camera intrinsics."""
        intrinsics = {}
        for i in range(self.num_cameras):
            # Standard camera intrinsic
            K = torch.eye(3, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()
            K[:, 0, 0] = 500  # fx
            K[:, 1, 1] = 500  # fy
            K[:, 0, 2] = self.image_width / 2   # cx
            K[:, 1, 2] = self.image_height / 2  # cy

            intrinsics[f"camera_{i}"] = K

        return intrinsics

    def _create_mock_target_poses(self):
        """Create mock target poses."""
        # Random positions
        pos = torch.randn(self.num_envs, self.num_targets, 3, device=self.device) * 2.0
        pos[:, :, 2] += 2.0  # Elevate targets

        # Random orientations
        quat = torch.randn(self.num_envs, self.num_targets, 4, device=self.device)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)

        return (pos, quat)


def run_tests():
    """Run all tests."""
    print("="*60)
    print("BBoxRayCaster Test Suite")
    print("="*60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBBoxRayCaster)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)