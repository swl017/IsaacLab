# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Benchmark script for BBoxRayCaster module.

This script measures performance across different configurations to help
optimize deployment for your specific use case.

Run with:
    python benchmark_bbox_raycaster.py --num_envs 4096 --num_cameras 2
"""
from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app
import argparse
import time
import torch
from typing import Dict, List, Tuple

# Attempt to import bbox_raycaster utilities
# Note: Full benchmarking requires Isaac Sim running
try:
    from bbox_raycaster.utils.projection import (
        batch_transform_points,
        batch_transform_to_camera_frame,
        batch_project_to_image_plane,
    )
    from bbox_raycaster.utils.bbox_ops import (
        compute_2d_bbox_from_corners,
        bbox_xyxy_to_xywh,
        normalize_bboxes,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: bbox_raycaster utilities not available. Running limited benchmark.")


class BenchmarkConfig:
    """Configuration for benchmarking."""

    def __init__(
        self,
        num_envs: int = 4096,
        num_cameras: int = 2,
        num_targets: int = 1,
        num_corners: int = 8,
        image_height: int = 480,
        image_width: int = 640,
        device: str = "cuda:0",
        num_warmup: int = 10,
        num_iterations: int = 100,
    ):
        self.num_envs = num_envs
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.num_corners = num_corners
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations


class BenchmarkRunner:
    """Run benchmarks for BBoxRayCaster components."""

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.results: Dict[str, List[float]] = {}

        print(f"\nBenchmark Configuration:")
        print(f"  Environments: {config.num_envs}")
        print(f"  Cameras/env: {config.num_cameras}")
        print(f"  Targets/env: {config.num_targets}")
        print(f"  Image size: {config.image_height}x{config.image_width}")
        print(f"  Device: {config.device}")
        print(f"  Warmup iters: {config.num_warmup}")
        print(f"  Benchmark iters: {config.num_iterations}\n")

    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        if not UTILS_AVAILABLE:
            print("Skipping benchmarks - utilities not available")
            return

        print("="*70)
        print("BENCHMARK RESULTS")
        print("="*70)

        # Individual component benchmarks
        self.benchmark_transform_to_world()
        self.benchmark_transform_to_camera()
        self.benchmark_projection()
        self.benchmark_bbox_computation()
        self.benchmark_normalization()

        # End-to-end benchmark
        self.benchmark_full_pipeline()

        # Memory benchmark
        self.benchmark_memory_usage()

        # Print summary
        self.print_summary()

    def benchmark_transform_to_world(self):
        """Benchmark: Transform bbox corners from local to world frame."""
        if not UTILS_AVAILABLE:
            return

        print("\n[1] Transform to World Frame")
        print("-" * 70)

        # Prepare data
        corners_local = torch.randn(
            self.cfg.num_corners, 3, device=self.cfg.device
        )
        target_pos = torch.randn(
            self.cfg.num_envs, self.cfg.num_targets, 3,
            device=self.cfg.device
        )
        target_quat = torch.randn(
            self.cfg.num_envs, self.cfg.num_targets, 4,
            device=self.cfg.device
        )
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)

        # Warmup
        for _ in range(self.cfg.num_warmup):
            result = batch_transform_points(corners_local, target_pos, target_quat)

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(self.cfg.num_iterations):
            start = time.time()
            result = batch_transform_points(corners_local, target_pos, target_quat)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        self.results['transform_to_world'] = times
        self._print_stats("Transform to World", times)

    def benchmark_transform_to_camera(self):
        """Benchmark: Transform points from world to camera frame."""
        if not UTILS_AVAILABLE:
            return

        print("\n[2] Transform to Camera Frame")
        print("-" * 70)

        # Prepare data
        points_world = torch.randn(
            self.cfg.num_envs, self.cfg.num_cameras, self.cfg.num_targets,
            self.cfg.num_corners, 3, device=self.cfg.device
        )
        camera_pos = torch.randn(
            self.cfg.num_envs, self.cfg.num_cameras, 3,
            device=self.cfg.device
        )
        camera_quat = torch.randn(
            self.cfg.num_envs, self.cfg.num_cameras, 4,
            device=self.cfg.device
        )
        camera_quat = camera_quat / torch.norm(camera_quat, dim=-1, keepdim=True)

        # Warmup
        for _ in range(self.cfg.num_warmup):
            result = batch_transform_to_camera_frame(
                points_world, camera_pos, camera_quat
            )

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(self.cfg.num_iterations):
            start = time.time()
            result = batch_transform_to_camera_frame(
                points_world, camera_pos, camera_quat
            )
            torch.cuda.synchronize()
            times.append(time.time() - start)

        self.results['transform_to_camera'] = times
        self._print_stats("Transform to Camera", times)

    def benchmark_projection(self):
        """Benchmark: Project points to image plane."""
        if not UTILS_AVAILABLE:
            return

        print("\n[3] Project to Image Plane")
        print("-" * 70)

        # Prepare data
        points_camera = torch.randn(
            self.cfg.num_envs, self.cfg.num_cameras, self.cfg.num_targets,
            self.cfg.num_corners, 3, device=self.cfg.device
        )
        points_camera[..., 2] = torch.abs(points_camera[..., 2]) + 1.0  # Ensure positive depth

        intrinsics = torch.eye(3, device=self.cfg.device).unsqueeze(0).unsqueeze(0)
        intrinsics = intrinsics.expand(
            self.cfg.num_envs, self.cfg.num_cameras, -1, -1
        ).clone()
        intrinsics[:, :, 0, 0] = 500  # fx
        intrinsics[:, :, 1, 1] = 500  # fy
        intrinsics[:, :, 0, 2] = self.cfg.image_width / 2
        intrinsics[:, :, 1, 2] = self.cfg.image_height / 2

        # Warmup
        for _ in range(self.cfg.num_warmup):
            pixels, depths, valid = batch_project_to_image_plane(
                points_camera, intrinsics
            )

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(self.cfg.num_iterations):
            start = time.time()
            pixels, depths, valid = batch_project_to_image_plane(
                points_camera, intrinsics
            )
            torch.cuda.synchronize()
            times.append(time.time() - start)

        self.results['projection'] = times
        self._print_stats("Projection", times)

    def benchmark_bbox_computation(self):
        """Benchmark: Compute 2D bboxes from corners."""
        if not UTILS_AVAILABLE:
            return

        print("\n[4] Compute 2D Bounding Boxes")
        print("-" * 70)

        # Prepare data
        corners_2d = torch.rand(
            self.cfg.num_envs, self.cfg.num_cameras, self.cfg.num_targets,
            self.cfg.num_corners, 2, device=self.cfg.device
        ) * 640  # Random pixels

        valid_corners = torch.ones(
            self.cfg.num_envs, self.cfg.num_cameras, self.cfg.num_targets,
            self.cfg.num_corners, device=self.cfg.device, dtype=torch.bool
        )

        # Warmup
        for _ in range(self.cfg.num_warmup):
            bbox_xyxy, bbox_valid = compute_2d_bbox_from_corners(
                corners_2d, valid_corners
            )
            bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy)

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(self.cfg.num_iterations):
            start = time.time()
            bbox_xyxy, bbox_valid = compute_2d_bbox_from_corners(
                corners_2d, valid_corners
            )
            bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        self.results['bbox_computation'] = times
        self._print_stats("BBox Computation", times)

    def benchmark_normalization(self):
        """Benchmark: Normalize bboxes."""
        if not UTILS_AVAILABLE:
            return

        print("\n[5] Normalize Bounding Boxes")
        print("-" * 70)

        # Prepare data
        bboxes = torch.rand(
            self.cfg.num_envs, self.cfg.num_cameras, self.cfg.num_targets, 4,
            device=self.cfg.device
        ) * 320  # Random bbox in pixels

        image_shapes = torch.tensor(
            [[self.cfg.image_height, self.cfg.image_width]],
            device=self.cfg.device
        ).expand(self.cfg.num_envs, self.cfg.num_cameras, -1)

        # Warmup
        for _ in range(self.cfg.num_warmup):
            bboxes_norm = normalize_bboxes(bboxes, image_shapes)

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(self.cfg.num_iterations):
            start = time.time()
            bboxes_norm = normalize_bboxes(bboxes, image_shapes)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        self.results['normalization'] = times
        self._print_stats("Normalization", times)

    def benchmark_full_pipeline(self):
        """Benchmark: Full end-to-end pipeline."""
        if not UTILS_AVAILABLE:
            return

        print("\n[6] Full Pipeline (End-to-End)")
        print("-" * 70)

        # Prepare all data
        corners_local = torch.randn(self.cfg.num_corners, 3, device=self.cfg.device)
        target_pos = torch.randn(
            self.cfg.num_envs, self.cfg.num_targets, 3, device=self.cfg.device
        )
        target_quat = torch.randn(
            self.cfg.num_envs, self.cfg.num_targets, 4, device=self.cfg.device
        )
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)

        camera_pos = torch.randn(
            self.cfg.num_envs, self.cfg.num_cameras, 3, device=self.cfg.device
        )
        camera_quat = torch.randn(
            self.cfg.num_envs, self.cfg.num_cameras, 4, device=self.cfg.device
        )
        camera_quat = camera_quat / torch.norm(camera_quat, dim=-1, keepdim=True)

        intrinsics = torch.eye(3, device=self.cfg.device).unsqueeze(0).unsqueeze(0)
        intrinsics = intrinsics.expand(
            self.cfg.num_envs, self.cfg.num_cameras, -1, -1
        ).clone()
        intrinsics[:, :, 0, 0] = 500
        intrinsics[:, :, 1, 1] = 500
        intrinsics[:, :, 0, 2] = self.cfg.image_width / 2
        intrinsics[:, :, 1, 2] = self.cfg.image_height / 2

        image_shapes = torch.tensor(
            [[self.cfg.image_height, self.cfg.image_width]], device=self.cfg.device
        ).expand(self.cfg.num_envs, self.cfg.num_cameras, -1)

        def full_pipeline():
            # Step 1: Transform to world
            corners_world = batch_transform_points(
                corners_local, target_pos, target_quat
            )

            # Step 2: Expand and transform to camera
            corners_world_expanded = corners_world.unsqueeze(1).expand(
                -1, self.cfg.num_cameras, -1, -1, -1
            )
            corners_camera = batch_transform_to_camera_frame(
                corners_world_expanded, camera_pos, camera_quat
            )

            # Step 3: Project to image
            pixels, depths, valid = batch_project_to_image_plane(
                corners_camera, intrinsics
            )

            # Step 4: Compute bboxes
            bbox_xyxy, bbox_valid = compute_2d_bbox_from_corners(
                pixels, valid
            )
            bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy)

            # Step 5: Normalize
            bbox_norm = normalize_bboxes(bbox_xywh, image_shapes)

            return bbox_norm, bbox_valid

        # Warmup
        for _ in range(self.cfg.num_warmup):
            _ = full_pipeline()

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(self.cfg.num_iterations):
            start = time.time()
            _ = full_pipeline()
            torch.cuda.synchronize()
            times.append(time.time() - start)

        self.results['full_pipeline'] = times
        self._print_stats("Full Pipeline", times)

        # Calculate throughput
        total_operations = (
            self.cfg.num_envs * self.cfg.num_cameras * self.cfg.num_targets
        )
        avg_time_ms = sum(times) / len(times) * 1000
        throughput = total_operations / (avg_time_ms / 1000)
        print(f"  Throughput: {throughput:.0f} bbox/sec")
        print(f"  Per-env: {avg_time_ms / self.cfg.num_envs:.4f} ms")

    def benchmark_memory_usage(self):
        """Benchmark: Memory usage at different scales."""
        if not torch.cuda.is_available():
            print("\n[7] Memory Usage: Skipped (CUDA not available)")
            return

        print("\n[7] Memory Usage")
        print("-" * 70)

        original_num_envs = self.cfg.num_envs
        test_sizes = [256, 512, 1024, 2048, 4096, 8192]

        print(f"{'Envs':<10} {'Allocated (MB)':<20} {'Reserved (MB)':<20}")
        print("-" * 70)

        for num_envs in test_sizes:
            if num_envs > original_num_envs * 2:
                break  # Don't test sizes much larger than configured

            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Allocate buffers
            N, C, T = num_envs, self.cfg.num_cameras, self.cfg.num_targets

            buffers = {
                'corners_world': torch.zeros((N, T, 8, 3), device=self.cfg.device),
                'corners_camera': torch.zeros((N, C, T, 8, 3), device=self.cfg.device),
                'pixels': torch.zeros((N, C, T, 8, 2), device=self.cfg.device),
                'depths': torch.zeros((N, C, T, 8), device=self.cfg.device),
                'bboxes': torch.zeros((N, C, T, 4), device=self.cfg.device),
            }

            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6

            print(f"{num_envs:<10} {allocated:>18.2f}  {reserved:>18.2f}")

            del buffers

        torch.cuda.empty_cache()

    def print_summary(self):
        """Print summary of all benchmarks."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        print(f"\n{'Component':<25} {'Mean (ms)':<15} {'Std (ms)':<15} {'FPS':<10}")
        print("-" * 70)

        for name, times in self.results.items():
            mean_ms = sum(times) / len(times) * 1000
            std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
            fps = 1000 / mean_ms if mean_ms > 0 else 0

            display_name = name.replace('_', ' ').title()
            print(f"{display_name:<25} {mean_ms:>13.3f}  {std_ms:>13.3f}  {fps:>8.0f}")

        print("="*70)

    def _print_stats(self, name: str, times: List[float]):
        """Print statistics for a benchmark."""
        mean_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5

        print(f"  Mean: {mean_ms:.3f} ms")
        print(f"  Std:  {std_ms:.3f} ms")
        print(f"  Min:  {min_ms:.3f} ms")
        print(f"  Max:  {max_ms:.3f} ms")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark BBoxRayCaster")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--num_cameras", type=int, default=2, help="Number of cameras per env")
    parser.add_argument("--num_targets", type=int, default=1, help="Number of targets per env")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--num_iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--num_warmup", type=int, default=10, help="Warmup iterations")

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        num_envs=args.num_envs,
        num_cameras=args.num_cameras,
        num_targets=args.num_targets,
        device=args.device,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )

    # Run benchmarks
    runner = BenchmarkRunner(config)
    runner.run_all_benchmarks()


if __name__ == "__main__":
    main()