#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Domain Randomization Module Test Suite

This test suite verifies the domain randomization components. The tests cover:
- Configuration instantiation and defaults
- Camera processor (crop/resize/intrinsics)
- Physics randomizer (sampling)
- Gimbal randomizer (offsets)
- Domain randomizer orchestration

Usage:
    # Run with Isaac Lab
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/domain_randomization/tests/run_tests.py
"""

import argparse

# IMPORTANT: AppLauncher must be initialized before other Isaac Lab imports
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run domain randomization tests")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--test-verbose",
    action="store_true",
    help="Enable verbose debug output",
)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW we can import other modules
import sys
import traceback
from datetime import datetime

import torch


# =============================================================================
# Test Results Tracking
# =============================================================================


class TestResults:
    """Tracks test results with pass/fail/error counts."""

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
        # Print first few lines of error
        error_lines = error.split("\n")[:3]
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
                print(f"  {error[:500]}")

        if self.errors:
            print("\n" + "-" * 80)
            print("TEST ERRORS:")
            print("-" * 80)
            for test_name, error in self.errors:
                print(f"\n{test_name}:")
                print(f"  {error[:500]}")

        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


# =============================================================================
# Configuration Tests
# =============================================================================


def run_config_tests(results: TestResults, device: torch.device):
    """Test configuration classes."""
    print("\n" + "=" * 80)
    print("Testing Configuration Classes")
    print("=" * 80)

    # Import configuration classes
    try:
        from isaaclab_tasks.direct.iris_ma6.domain_randomization import (
            CameraRandomizationCfg,
            DomainRandomizationCfg,
            GimbalRandomizationCfg,
            MassRandomizationCfg,
            MaterialRandomizationCfg,
            PhysicsRandomizationCfg,
        )

        results.add_pass("Import configuration classes")
    except Exception as e:
        results.add_error("Import configuration classes", traceback.format_exc())
        return

    # Test default instantiation
    try:
        cfg = DomainRandomizationCfg()
        assert cfg.enabled is True
        assert cfg.camera is not None
        assert cfg.physics is not None
        assert cfg.gimbal is not None
        results.add_pass("DomainRandomizationCfg default instantiation")
    except Exception as e:
        results.add_fail("DomainRandomizationCfg default instantiation", str(e))

    # Test camera config defaults
    try:
        cam_cfg = CameraRandomizationCfg()
        assert cam_cfg.render_width == 1920
        assert cam_cfg.render_height == 1080
        assert cam_cfg.fov_scale_range == (0.9, 1.0)
        assert cam_cfg.focal_length_range == (970.0, 1135.0)
        assert len(cam_cfg.discrete_resolutions) == 3
        results.add_pass("CameraRandomizationCfg defaults")
    except Exception as e:
        results.add_fail("CameraRandomizationCfg defaults", str(e))

    # Test physics config composition
    try:
        phys_cfg = PhysicsRandomizationCfg()
        assert phys_cfg.mass is not None
        assert phys_cfg.material is not None
        assert phys_cfg.mass.body_mass_scale_range == (0.9, 1.1)
        results.add_pass("PhysicsRandomizationCfg composition")
    except Exception as e:
        results.add_fail("PhysicsRandomizationCfg composition", str(e))

    # Test gimbal config defaults
    try:
        gimbal_cfg = GimbalRandomizationCfg()
        assert gimbal_cfg.yaw_offset_range == (-0.1, 0.1)
        assert gimbal_cfg.pitch_offset_range == (-0.05, 0.05)
        results.add_pass("GimbalRandomizationCfg defaults")
    except Exception as e:
        results.add_fail("GimbalRandomizationCfg defaults", str(e))

    # Test custom configuration
    try:
        custom_cfg = DomainRandomizationCfg(
            enabled=True,
            camera=CameraRandomizationCfg(
                fov_scale_range=(0.7, 1.0),
                focal_length_range=(900.0, 1100.0),
            ),
            physics=PhysicsRandomizationCfg(
                mass=MassRandomizationCfg(
                    body_mass_scale_range=(0.85, 1.15),
                ),
            ),
        )
        assert custom_cfg.camera.fov_scale_range == (0.7, 1.0)
        assert custom_cfg.physics.mass.body_mass_scale_range == (0.85, 1.15)
        results.add_pass("Custom configuration override")
    except Exception as e:
        results.add_fail("Custom configuration override", str(e))


# =============================================================================
# Camera Processor Tests
# =============================================================================


def run_camera_processor_tests(results: TestResults, device: torch.device):
    """Test camera processor functionality."""
    print("\n" + "=" * 80)
    print("Testing Camera Processor")
    print("=" * 80)

    try:
        from isaaclab_tasks.direct.iris_ma6.domain_randomization import (
            CameraProcessor,
            CameraRandomizationCfg,
        )

        results.add_pass("Import CameraProcessor")
    except Exception as e:
        results.add_error("Import CameraProcessor", traceback.format_exc())
        return

    num_envs = 16
    num_agents = 3
    cfg = CameraRandomizationCfg()

    # Test initialization
    try:
        processor = CameraProcessor(cfg, num_envs, num_agents, device)
        assert processor.num_envs == num_envs
        assert processor.num_agents == num_agents
        assert processor.crop_params.shape == (num_envs, num_agents, 4)
        assert processor.intrinsic_matrices.shape == (num_envs, num_agents, 3, 3)
        results.add_pass("CameraProcessor initialization")
    except Exception as e:
        results.add_fail("CameraProcessor initialization", str(e))
        return

    # Test default intrinsic matrices
    try:
        K = processor.get_intrinsic_matrices()
        # Check diagonal elements (focal lengths)
        assert torch.all(K[:, :, 0, 0] > 0), "f_x should be positive"
        assert torch.all(K[:, :, 1, 1] > 0), "f_y should be positive"
        # Check principal points are centered
        assert torch.all(K[:, :, 2, 2] == 1.0), "K[2,2] should be 1"
        results.add_pass("Default intrinsic matrices")
    except Exception as e:
        results.add_fail("Default intrinsic matrices", str(e))

    # Test randomization
    try:
        env_ids = torch.arange(8, device=device)
        processor.randomize(env_ids)

        # Check FOV scales are within range
        fov_scales = processor.get_fov_scales()
        assert torch.all(fov_scales[env_ids] >= cfg.fov_scale_range[0])
        assert torch.all(fov_scales[env_ids] <= cfg.fov_scale_range[1])
        results.add_pass("Camera parameter randomization")
    except Exception as e:
        results.add_fail("Camera parameter randomization", str(e))

    # Test crop parameters computation
    try:
        crop_params = processor.get_crop_params()
        crop_x = crop_params[:, :, 0]
        crop_y = crop_params[:, :, 1]
        crop_w = crop_params[:, :, 2]
        crop_h = crop_params[:, :, 3]

        # Check symmetric crops (centered principal point)
        expected_cx = (cfg.render_width - crop_w) / 2
        expected_cy = (cfg.render_height - crop_h) / 2
        assert torch.allclose(crop_x, expected_cx.floor(), atol=1.0)
        assert torch.allclose(crop_y, expected_cy.floor(), atol=1.0)
        results.add_pass("Symmetric crop computation")
    except Exception as e:
        results.add_fail("Symmetric crop computation", str(e))

    # Test image processing with grid_sample
    try:
        # Create synthetic images: (N, C, H, W)
        N = num_envs * num_agents
        images = torch.randn(N, 3, cfg.render_height, cfg.render_width, device=device)

        output_size = (360, 640)
        processed, intrinsics = processor.process_images_batched(images, output_size)

        assert processed.shape == (N, 3, output_size[0], output_size[1])
        assert intrinsics.shape == (N, 3, 3)
        results.add_pass("Image processing with grid_sample")
    except Exception as e:
        results.add_fail("Image processing with grid_sample", str(e))

    # Test intrinsic matrix consistency
    try:
        output_size = (360, 640)
        _, intrinsics = processor.process_images_batched(
            torch.randn(N, 3, cfg.render_height, cfg.render_width, device=device),
            output_size,
        )

        # Principal points should be centered
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        assert torch.allclose(cx, torch.tensor(output_size[1] / 2.0, device=device))
        assert torch.allclose(cy, torch.tensor(output_size[0] / 2.0, device=device))
        results.add_pass("Intrinsic matrix consistency")
    except Exception as e:
        results.add_fail("Intrinsic matrix consistency", str(e))

    # Test focal length scaling
    try:
        # After crop+resize: f_final = f_render * (output_w / crop_w)
        fov_scale = 0.5  # Crop to half size
        processor.fov_scales.fill_(fov_scale)
        processor._compute_all_transforms()

        crop_w = processor.crop_params[0, 0, 2].item()
        expected_crop_w = cfg.render_width * fov_scale

        output_w = 640
        expected_scale = output_w / expected_crop_w
        f_render = processor.focal_lengths[0, 0].item()
        expected_f_final = f_render * expected_scale

        K = processor._compute_intrinsics_for_output((360, 640))
        actual_f_final = K[0, 0, 0].item()

        assert abs(actual_f_final - expected_f_final) < 1.0, (
            f"Expected f={expected_f_final}, got {actual_f_final}"
        )
        results.add_pass("Focal length scaling")
    except Exception as e:
        results.add_fail("Focal length scaling", str(e))


# =============================================================================
# Physics Randomizer Tests
# =============================================================================


def run_physics_randomizer_tests(results: TestResults, device: torch.device):
    """Test physics randomizer functionality."""
    print("\n" + "=" * 80)
    print("Testing Physics Randomizer")
    print("=" * 80)

    try:
        from isaaclab_tasks.direct.iris_ma6.domain_randomization import (
            PhysicsRandomizationCfg,
            PhysicsRandomizer,
        )

        results.add_pass("Import PhysicsRandomizer")
    except Exception as e:
        results.add_error("Import PhysicsRandomizer", traceback.format_exc())
        return

    num_envs = 32
    cfg = PhysicsRandomizationCfg()

    # Test initialization
    try:
        randomizer = PhysicsRandomizer(cfg, num_envs, device)
        assert randomizer.num_envs == num_envs
        assert randomizer.mass_scales.shape == (num_envs,)
        assert randomizer.payload_masses.shape == (num_envs,)
        results.add_pass("PhysicsRandomizer initialization")
    except Exception as e:
        results.add_fail("PhysicsRandomizer initialization", str(e))
        return

    # Test mass sampling
    try:
        env_ids = torch.arange(16, device=device)
        randomizer.sample_mass_parameters(env_ids)

        mass_scales = randomizer.get_mass_scales(env_ids)
        mass_cfg = cfg.mass

        assert torch.all(mass_scales >= mass_cfg.body_mass_scale_range[0])
        assert torch.all(mass_scales <= mass_cfg.body_mass_scale_range[1])
        results.add_pass("Mass scale sampling")
    except Exception as e:
        results.add_fail("Mass scale sampling", str(e))

    # Test payload mass sampling
    try:
        payload_masses = randomizer.get_payload_masses(env_ids)
        mass_cfg = cfg.mass

        assert torch.all(payload_masses >= mass_cfg.payload_mass_range[0])
        assert torch.all(payload_masses <= mass_cfg.payload_mass_range[1])
        results.add_pass("Payload mass sampling")
    except Exception as e:
        results.add_fail("Payload mass sampling", str(e))

    # Test material sampling
    try:
        randomizer.sample_material_parameters(env_ids)

        static_friction, dynamic_friction = randomizer.get_friction_coefficients(env_ids)
        mat_cfg = cfg.material

        assert torch.all(static_friction >= mat_cfg.static_friction_range[0])
        assert torch.all(static_friction <= mat_cfg.static_friction_range[1])

        # Check consistency: dynamic <= static
        if mat_cfg.make_consistent:
            assert torch.all(dynamic_friction <= static_friction + 1e-6)
        results.add_pass("Material parameter sampling")
    except Exception as e:
        results.add_fail("Material parameter sampling", str(e))


# =============================================================================
# Gimbal Randomizer Tests
# =============================================================================


def run_gimbal_randomizer_tests(results: TestResults, device: torch.device):
    """Test gimbal randomizer functionality."""
    print("\n" + "=" * 80)
    print("Testing Gimbal Randomizer")
    print("=" * 80)

    try:
        from isaaclab_tasks.direct.iris_ma6.domain_randomization import (
            GimbalRandomizationCfg,
            GimbalRandomizer,
        )

        results.add_pass("Import GimbalRandomizer")
    except Exception as e:
        results.add_error("Import GimbalRandomizer", traceback.format_exc())
        return

    num_envs = 16
    num_agents = 3
    cfg = GimbalRandomizationCfg()

    # Test initialization
    try:
        randomizer = GimbalRandomizer(cfg, num_envs, num_agents, device)
        assert randomizer.num_envs == num_envs
        assert randomizer.num_agents == num_agents
        assert randomizer.joint_offsets.shape == (num_envs, num_agents, 3)
        results.add_pass("GimbalRandomizer initialization")
    except Exception as e:
        results.add_fail("GimbalRandomizer initialization", str(e))
        return

    # Test joint offset sampling
    try:
        env_ids = torch.arange(8, device=device)
        randomizer.randomize_joint_offsets(env_ids)

        offsets = randomizer.get_joint_offsets(env_ids)

        # Check yaw offsets
        yaw_offsets = offsets[:, :, 0]
        assert torch.all(yaw_offsets >= cfg.yaw_offset_range[0])
        assert torch.all(yaw_offsets <= cfg.yaw_offset_range[1])

        # Check pitch offsets
        pitch_offsets = offsets[:, :, 1]
        assert torch.all(pitch_offsets >= cfg.pitch_offset_range[0])
        assert torch.all(pitch_offsets <= cfg.pitch_offset_range[1])

        # Check roll offsets
        roll_offsets = offsets[:, :, 2]
        assert torch.all(roll_offsets >= cfg.roll_offset_range[0])
        assert torch.all(roll_offsets <= cfg.roll_offset_range[1])

        results.add_pass("Joint offset sampling")
    except Exception as e:
        results.add_fail("Joint offset sampling", str(e))

    # Test individual offset getters
    try:
        yaw = randomizer.get_yaw_offsets(env_ids)
        pitch = randomizer.get_pitch_offsets(env_ids)
        roll = randomizer.get_roll_offsets(env_ids)

        assert yaw.shape == (len(env_ids), num_agents)
        assert pitch.shape == (len(env_ids), num_agents)
        assert roll.shape == (len(env_ids), num_agents)
        results.add_pass("Individual offset getters")
    except Exception as e:
        results.add_fail("Individual offset getters", str(e))

    # Test dynamics sampling
    try:
        cfg_with_dynamics = GimbalRandomizationCfg(randomize_dynamics_per_episode=True)
        randomizer2 = GimbalRandomizer(cfg_with_dynamics, num_envs, num_agents, device)
        randomizer2.randomize_dynamics(env_ids)

        stiffness = randomizer2.get_stiffness_scales(env_ids)
        damping = randomizer2.get_damping_scales(env_ids)

        assert torch.all(stiffness >= cfg_with_dynamics.stiffness_scale_range[0])
        assert torch.all(stiffness <= cfg_with_dynamics.stiffness_scale_range[1])
        assert torch.all(damping >= cfg_with_dynamics.damping_scale_range[0])
        assert torch.all(damping <= cfg_with_dynamics.damping_scale_range[1])
        results.add_pass("Dynamics sampling")
    except Exception as e:
        results.add_fail("Dynamics sampling", str(e))


# =============================================================================
# Domain Randomizer Integration Tests
# =============================================================================


def run_domain_randomizer_tests(results: TestResults, device: torch.device):
    """Test domain randomizer orchestration."""
    print("\n" + "=" * 80)
    print("Testing Domain Randomizer (Integration)")
    print("=" * 80)

    try:
        from isaaclab_tasks.direct.iris_ma6.domain_randomization import (
            DomainRandomizationCfg,
            DomainRandomizer,
        )

        results.add_pass("Import DomainRandomizer")
    except Exception as e:
        results.add_error("Import DomainRandomizer", traceback.format_exc())
        return

    num_envs = 32
    num_agents = 3
    cfg = DomainRandomizationCfg()

    # Test initialization
    try:
        randomizer = DomainRandomizer(cfg, num_envs, num_agents, device)
        assert randomizer.num_envs == num_envs
        assert randomizer.num_agents == num_agents
        assert randomizer.camera_processor is not None
        assert randomizer.physics_randomizer is not None
        assert randomizer.gimbal_randomizer is not None
        results.add_pass("DomainRandomizer initialization")
    except Exception as e:
        results.add_fail("DomainRandomizer initialization", str(e))
        return

    # Test randomize_all
    try:
        env_ids = torch.arange(16, device=device)
        randomizer.randomize_all(env_ids)

        # Verify all components were randomized
        fov_scales = randomizer.get_fov_scales()
        assert not torch.all(fov_scales[env_ids] == 1.0), "FOV should be randomized"

        mass_scales = randomizer.get_mass_scales(env_ids)
        # Mass scales should vary (unless all happened to be 1.0)
        # Just check they exist
        assert mass_scales.shape == (len(env_ids),)

        gimbal_offsets = randomizer.get_gimbal_offsets(env_ids)
        assert gimbal_offsets.shape == (len(env_ids), num_agents, 3)

        results.add_pass("randomize_all orchestration")
    except Exception as e:
        results.add_fail("randomize_all orchestration", str(e))

    # Test image processing through randomizer
    try:
        N = num_envs * num_agents
        images = torch.randn(N, 3, cfg.camera.render_height, cfg.camera.render_width, device=device)

        processed, intrinsics = randomizer.process_images(images, output_size=(360, 640))

        assert processed.shape == (N, 3, 360, 640)
        assert intrinsics.shape == (N, 3, 3)
        results.add_pass("Image processing through randomizer")
    except Exception as e:
        results.add_fail("Image processing through randomizer", str(e))

    # Test state summary
    try:
        summary = randomizer.get_current_state_summary()
        assert "camera" in summary
        assert "physics" in summary
        assert "gimbal" in summary
        assert "fov_scale_mean" in summary["camera"]
        assert "mass_scale_mean" in summary["physics"]
        results.add_pass("State summary generation")
    except Exception as e:
        results.add_fail("State summary generation", str(e))

    # Test component access
    try:
        camera = randomizer.get_camera_processor()
        physics = randomizer.get_physics_randomizer()
        gimbal = randomizer.get_gimbal_randomizer()
        assert camera is not None
        assert physics is not None
        assert gimbal is not None
        results.add_pass("Component access")
    except Exception as e:
        results.add_fail("Component access", str(e))

    # Test disabled randomization
    try:
        disabled_cfg = DomainRandomizationCfg(enabled=False)
        disabled_randomizer = DomainRandomizer(disabled_cfg, num_envs, num_agents, device)

        # Save initial state
        initial_fov = disabled_randomizer.get_fov_scales().clone()

        # Try to randomize
        disabled_randomizer.randomize_all(env_ids)

        # Should be unchanged
        assert torch.all(disabled_randomizer.get_fov_scales() == initial_fov)
        results.add_pass("Disabled randomization")
    except Exception as e:
        results.add_fail("Disabled randomization", str(e))


# =============================================================================
# Main
# =============================================================================


def main():
    """Main test runner."""
    print("=" * 80)
    print("DOMAIN RANDOMIZATION MODULE TEST SUITE")
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
        run_config_tests(results, device)
    except Exception as e:
        results.add_error("Configuration test suite", traceback.format_exc())

    try:
        run_camera_processor_tests(results, device)
    except Exception as e:
        results.add_error("Camera processor test suite", traceback.format_exc())

    try:
        run_physics_randomizer_tests(results, device)
    except Exception as e:
        results.add_error("Physics randomizer test suite", traceback.format_exc())

    try:
        run_gimbal_randomizer_tests(results, device)
    except Exception as e:
        results.add_error("Gimbal randomizer test suite", traceback.format_exc())

    try:
        run_domain_randomizer_tests(results, device)
    except Exception as e:
        results.add_error("Domain randomizer test suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
