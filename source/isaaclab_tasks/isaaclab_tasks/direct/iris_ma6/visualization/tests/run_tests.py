#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test runner for iris_ma6 visualization module.

This script tests the visualization module components without requiring
a full simulation. It tests:
- Camera frustum computation (tensor operations only)
- Detection indicator color logic (tensor operations only)
- CustomVisualization initialization and warmup logic

Run with:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/visualization/tests/run_tests.py
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run visualization module test suite")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import other modules
import sys
import traceback
from datetime import datetime

import torch


class TestResults:
    """Track test results with pass/fail counts and detailed error reporting."""

    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []

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
        else:
            print("No tests run!")

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


def run_camera_frustum_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test camera frustum computation."""
    print("\n" + "=" * 80)
    print("Testing CameraFrustum")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.visualization.camera_frustum import (
        CameraFrustum,
        create_camera_cfg_tensor,
    )

    num_envs = 4

    # Test 1: CameraFrustum initialization
    try:
        frustum = CameraFrustum()
        # draw_interface may be None in headless mode - that's OK
        results.add_pass("CameraFrustum initialization")
    except Exception as e:
        results.add_fail("CameraFrustum initialization", traceback.format_exc())
        return  # Can't continue without initialization

    # Check if we're in headless mode (no draw_interface)
    headless_mode = frustum.draw_interface is None
    if headless_mode and verbose:
        print("    Running in headless mode - draw_interface not available")

    # Test 2: create_camera_cfg_tensor
    try:
        # Mock matches iris_ma6 production cfg (mrcal 1x calibration: 1920×1080, fx≈1053).
        class MockSpawn:
            focal_length = 11.493
            horizontal_aperture = 20.955
            clipping_range = (0.1, 1.0e5)

        class MockCameraCfg:
            width = 1920
            height = 1080
            spawn = MockSpawn()

        camera_cfg = MockCameraCfg()
        cfg_tensor = create_camera_cfg_tensor(camera_cfg, num_envs, device=device)

        assert cfg_tensor.shape == (num_envs, 6), f"Expected shape ({num_envs}, 6), got {cfg_tensor.shape}"
        assert cfg_tensor[0, 0] == 1920.0, f"Expected width 1920, got {cfg_tensor[0, 0]}"
        assert cfg_tensor[0, 1] == 1080.0, f"Expected height 1080, got {cfg_tensor[0, 1]}"
        assert abs(cfg_tensor[0, 2].item() - 11.493) < 1e-4, f"Expected focal_length 11.493, got {cfg_tensor[0, 2]}"

        if verbose:
            print(f"    Camera config tensor shape: {cfg_tensor.shape}")
            print(f"    Camera config tensor[0]: {cfg_tensor[0]}")

        results.add_pass("create_camera_cfg_tensor")
    except Exception as e:
        results.add_fail("create_camera_cfg_tensor", traceback.format_exc())

    # Test 3: Frustum computation with zoom level 1.0
    try:
        cfg_tensor = create_camera_cfg_tensor(camera_cfg, num_envs, device=device)
        zoom_level = torch.ones(num_envs, device=device)

        frustum_data = frustum._compute_camera_frustum_batched(cfg_tensor, zoom_level, device=device)

        assert "far_corners" in frustum_data
        assert frustum_data["far_corners"].shape == (num_envs, 4, 3)

        if verbose:
            print(f"    Far corners shape: {frustum_data['far_corners'].shape}")
            print(f"    Far plane: {frustum_data['far_plane']}")

        results.add_pass("Frustum computation (zoom=1.0)")
    except Exception as e:
        results.add_fail("Frustum computation (zoom=1.0)", traceback.format_exc())

    # Test 4: Frustum computation with zoom level 2.0 (should be narrower)
    try:
        zoom_level_1 = torch.ones(num_envs, device=device)
        zoom_level_2 = torch.ones(num_envs, device=device) * 2.0

        frustum_data_1 = frustum._compute_camera_frustum_batched(cfg_tensor, zoom_level_1, device=device)
        frustum_data_2 = frustum._compute_camera_frustum_batched(cfg_tensor, zoom_level_2, device=device)

        # Higher zoom should result in narrower FOV (smaller far corners spread)
        fov_1 = frustum_data_1["horizontal_fov_rad"][0].item()
        fov_2 = frustum_data_2["horizontal_fov_rad"][0].item()

        assert fov_2 < fov_1, f"Zoom 2.0 should have smaller FOV than zoom 1.0: {fov_2} >= {fov_1}"

        if verbose:
            print(f"    FOV at zoom 1.0: {torch.rad2deg(torch.tensor(fov_1)).item():.2f} deg")
            print(f"    FOV at zoom 2.0: {torch.rad2deg(torch.tensor(fov_2)).item():.2f} deg")

        results.add_pass("Frustum zoom scaling")
    except Exception as e:
        results.add_fail("Frustum zoom scaling", traceback.format_exc())

    # Test 5: Clear method exists
    try:
        frustum.clear()
        results.add_pass("CameraFrustum clear method")
    except Exception as e:
        results.add_fail("CameraFrustum clear method", traceback.format_exc())


def run_detection_indicator_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test detection indicator functionality."""
    print("\n" + "=" * 80)
    print("Testing DetectionIndicator")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.visualization.detection_indicator import DetectionIndicator

    num_envs = 4

    # Test 1: Initialization
    try:
        indicator = DetectionIndicator(num_envs, device)
        # draw_interface may be None in headless mode - that's OK
        assert indicator.num_envs == num_envs
        results.add_pass("DetectionIndicator initialization")
    except Exception as e:
        results.add_fail("DetectionIndicator initialization", traceback.format_exc())
        return

    # Check if we're in headless mode
    headless_mode = indicator.draw_interface is None
    if headless_mode and verbose:
        print("    Running in headless mode - draw_interface not available")

    # Test 2: Color logic - all detected
    try:
        detected_all = torch.ones(num_envs, dtype=torch.bool, device=device)
        colors = indicator.get_line_colors(detected_all)

        assert colors.shape == (num_envs, 4)
        # Green color: [0.0, 1.0, 0.0, 1.0]
        expected_green = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=device).expand(num_envs, -1)
        assert torch.allclose(colors, expected_green), f"Expected green, got {colors}"

        results.add_pass("Color logic - all detected (green)")
    except Exception as e:
        results.add_fail("Color logic - all detected (green)", traceback.format_exc())

    # Test 3: Color logic - none detected
    try:
        detected_none = torch.zeros(num_envs, dtype=torch.bool, device=device)
        colors = indicator.get_line_colors(detected_none)

        # Yellow color: [1.0, 1.0, 0.0, 1.0]
        expected_yellow = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=device).expand(num_envs, -1)
        assert torch.allclose(colors, expected_yellow), f"Expected yellow, got {colors}"

        results.add_pass("Color logic - none detected (yellow)")
    except Exception as e:
        results.add_fail("Color logic - none detected (yellow)", traceback.format_exc())

    # Test 4: Color logic - mixed detection
    try:
        detected_mixed = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
        colors = indicator.get_line_colors(detected_mixed)

        expected = torch.tensor([
            [0.0, 1.0, 0.0, 1.0],  # green (detected)
            [1.0, 1.0, 0.0, 1.0],  # yellow (not detected)
            [0.0, 1.0, 0.0, 1.0],  # green (detected)
            [1.0, 1.0, 0.0, 1.0],  # yellow (not detected)
        ], device=device)
        assert torch.allclose(colors, expected), f"Expected mixed colors, got {colors}"

        results.add_pass("Color logic - mixed detection")
    except Exception as e:
        results.add_fail("Color logic - mixed detection", traceback.format_exc())

    # Test 5: Clear method
    try:
        indicator.clear()
        results.add_pass("DetectionIndicator clear method")
    except Exception as e:
        results.add_fail("DetectionIndicator clear method", traceback.format_exc())


def run_custom_visualization_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test CustomVisualization wrapper."""
    print("\n" + "=" * 80)
    print("Testing CustomVisualization")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.visualization.custom_visualization import (
        CustomVisualization,
        MIN_FRAMES_BEFORE_VISUALIZATION,
    )

    num_envs = 4
    possible_agents = ["drone_0", "drone_1", "drone_2"]

    # Mock matches iris_ma6 production cfg (mrcal 1x calibration: 1920×1080, fx≈1053).
    class MockSpawn:
        focal_length = 11.493
        horizontal_aperture = 20.955
        clipping_range = (0.1, 1.0e5)

    class MockCameraCfg:
        width = 1920
        height = 1080
        spawn = MockSpawn()

    camera_cfg = MockCameraCfg()

    # Test 1: Initialization
    try:
        viz = CustomVisualization(num_envs, possible_agents, camera_cfg, device)
        assert len(viz.camera_frustum) == len(possible_agents)
        assert len(viz.detection_indicator) == len(possible_agents)
        assert viz._frame_count == 0
        assert not viz._visualization_enabled

        results.add_pass("CustomVisualization initialization")
    except Exception as e:
        results.add_fail("CustomVisualization initialization", traceback.format_exc())
        return

    # Test 2: Warmup period
    try:
        viz = CustomVisualization(num_envs, possible_agents, camera_cfg, device)

        # Before warmup
        assert not viz.is_ready, "Should not be ready before warmup"

        # During warmup
        for i in range(MIN_FRAMES_BEFORE_VISUALIZATION - 1):
            viz.step()
            assert not viz.is_ready, f"Should not be ready at frame {i+1}"

        # After warmup
        viz.step()
        assert viz.is_ready, "Should be ready after warmup period"

        if verbose:
            print(f"    Warmup period: {MIN_FRAMES_BEFORE_VISUALIZATION} frames")

        results.add_pass("Warmup period logic")
    except Exception as e:
        results.add_fail("Warmup period logic", traceback.format_exc())

    # Test 3: Per-agent visualizer creation
    try:
        viz = CustomVisualization(num_envs, possible_agents, camera_cfg, device)

        for agent_id in possible_agents:
            assert agent_id in viz.camera_frustum
            assert agent_id in viz.detection_indicator
            assert viz.camera_frustum[agent_id] is not None
            assert viz.detection_indicator[agent_id] is not None

        results.add_pass("Per-agent visualizer creation")
    except Exception as e:
        results.add_fail("Per-agent visualizer creation", traceback.format_exc())

    # Test 4: Update method called before warmup (should not crash)
    try:
        viz = CustomVisualization(num_envs, possible_agents, camera_cfg, device)

        # Create mock data
        camera_poses = {
            agent_id: (
                torch.randn(num_envs, 3, device=device),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1),
            )
            for agent_id in possible_agents
        }
        target_pos = torch.randn(num_envs, 3, device=device)
        bbox_empty = {agent_id: torch.zeros(num_envs, dtype=torch.bool, device=device) for agent_id in possible_agents}
        zoom_levels = {agent_id: torch.ones(num_envs, device=device) for agent_id in possible_agents}

        # This should not crash, just skip visualization
        viz.update(camera_poses, target_pos, bbox_empty, zoom_levels)

        results.add_pass("Update before warmup (no crash)")
    except Exception as e:
        results.add_fail("Update before warmup (no crash)", traceback.format_exc())

    # Test 5: Update method after warmup
    try:
        viz = CustomVisualization(num_envs, possible_agents, camera_cfg, device)

        # Complete warmup
        for _ in range(MIN_FRAMES_BEFORE_VISUALIZATION):
            viz.step()

        # Create mock data
        camera_poses = {
            agent_id: (
                torch.randn(num_envs, 3, device=device),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1),
            )
            for agent_id in possible_agents
        }
        target_pos = torch.randn(num_envs, 3, device=device)
        bbox_empty = {agent_id: torch.zeros(num_envs, dtype=torch.bool, device=device) for agent_id in possible_agents}
        zoom_levels = {agent_id: torch.ones(num_envs, device=device) for agent_id in possible_agents}

        # This should work now
        viz.update(camera_poses, target_pos, bbox_empty, zoom_levels)

        results.add_pass("Update after warmup")
    except Exception as e:
        results.add_fail("Update after warmup", traceback.format_exc())

    # Test 6: Multi-target bbox_empty handling
    try:
        viz = CustomVisualization(num_envs, possible_agents, camera_cfg, device)
        for _ in range(MIN_FRAMES_BEFORE_VISUALIZATION):
            viz.step()

        # Create multi-target bbox_empty (N, T) where T=3 targets
        num_targets = 3
        camera_poses = {
            agent_id: (
                torch.randn(num_envs, 3, device=device),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1),
            )
            for agent_id in possible_agents
        }
        target_pos = torch.randn(num_envs, 3, device=device)
        bbox_empty = {
            agent_id: torch.zeros(num_envs, num_targets, dtype=torch.bool, device=device)
            for agent_id in possible_agents
        }
        zoom_levels = {agent_id: torch.ones(num_envs, device=device) for agent_id in possible_agents}

        # Should handle multi-target case by taking first target
        viz.update(camera_poses, target_pos, bbox_empty, zoom_levels)

        results.add_pass("Multi-target bbox_empty handling")
    except Exception as e:
        results.add_fail("Multi-target bbox_empty handling", traceback.format_exc())


def main():
    """Main test runner."""
    print("=" * 80)
    print("VISUALIZATION MODULE TEST SUITE")
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
        run_camera_frustum_tests(results, device, verbose)
    except Exception as e:
        results.add_error("CameraFrustum test suite", traceback.format_exc())

    try:
        run_detection_indicator_tests(results, device, verbose)
    except Exception as e:
        results.add_error("DetectionIndicator test suite", traceback.format_exc())

    try:
        run_custom_visualization_tests(results, device, verbose)
    except Exception as e:
        results.add_error("CustomVisualization test suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
