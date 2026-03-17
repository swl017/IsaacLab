# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main domain randomizer orchestrating all randomization components.

This module provides the DomainRandomizer class that coordinates:
- Camera parameter randomization (FOV, resolution, focal length)
- Physics property randomization (mass, friction, restitution)
- Gimbal joint randomization (offsets, dynamics)

The randomizer is designed to be standalone (no environment dependency)
for testing, with integration methods for environment use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .camera_processor import CameraProcessor
from .domain_randomization_cfg import DomainRandomizationCfg
from .gimbal_randomizer import GimbalRandomizer
from .physics_randomizer import PhysicsRandomizer

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


class DomainRandomizer:
    """Main orchestrator for all domain randomization.

    This class coordinates the three randomization components:
    - CameraProcessor: Computational camera randomization
    - PhysicsRandomizer: Mass and material randomization
    - GimbalRandomizer: Joint offset and dynamics randomization

    Example:
        ```python
        cfg = DomainRandomizationCfg()
        randomizer = DomainRandomizer(
            cfg=cfg,
            num_envs=256,
            num_agents=3,
            device="cuda",
        )

        # Randomize all parameters for some environments
        randomizer.randomize_all(env_ids=torch.arange(128))

        # Process camera images
        processed, intrinsics = randomizer.process_images(images, output_size=(360, 640))

        # Get gimbal offsets for control
        offsets = randomizer.get_gimbal_offsets()
        ```
    """

    def __init__(
        self,
        cfg: DomainRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
    ):
        """Initialize the domain randomizer.

        Args:
            cfg: Domain randomization configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: PyTorch device for tensor allocation.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = torch.device(device) if isinstance(device, str) else device

        # Initialize sub-components
        self.camera_processor = CameraProcessor(
            cfg=cfg.camera,
            num_envs=num_envs,
            num_agents=num_agents,
            device=self.device,
        )

        self.physics_randomizer = PhysicsRandomizer(
            cfg=cfg.physics,
            num_envs=num_envs,
            device=self.device,
        )

        self.gimbal_randomizer = GimbalRandomizer(
            cfg=cfg.gimbal,
            num_envs=num_envs,
            num_agents=num_agents,
            device=self.device,
        )

        # Set random seed if specified
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

    # =========================================================================
    # Main Randomization Methods
    # =========================================================================

    def randomize_all(self, env_ids: torch.Tensor | None = None):
        """Randomize all parameters for specified environments.

        This is the main entry point for per-episode randomization.
        Calls all component randomizers.

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.enabled:
            return

        self.randomize_camera(env_ids)
        self.randomize_physics(env_ids)
        self.randomize_gimbal(env_ids)

    def randomize_camera(self, env_ids: torch.Tensor | None = None):
        """Randomize camera parameters (FOV, focal length, resolution).

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.enabled or not self.cfg.camera.enabled:
            return

        self.camera_processor.randomize(env_ids)

    def randomize_physics(self, env_ids: torch.Tensor | None = None):
        """Randomize physics parameters (mass, materials).

        Samples new values but does NOT apply to simulation.
        Call apply_physics_randomization() with asset reference to apply.

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.enabled:
            return

        self.physics_randomizer.sample_mass_parameters(env_ids)
        self.physics_randomizer.sample_material_parameters(env_ids)

    def randomize_gimbal(self, env_ids: torch.Tensor | None = None):
        """Randomize gimbal parameters (joint offsets, dynamics).

        Samples new values. Joint offsets are retrieved via get_gimbal_offsets().
        Dynamics can be applied via apply_gimbal_dynamics().

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.enabled or not self.cfg.gimbal.enabled:
            return

        self.gimbal_randomizer.randomize_joint_offsets(env_ids)
        self.gimbal_randomizer.randomize_dynamics(env_ids)

    # =========================================================================
    # Application Methods (Require Asset References)
    # =========================================================================

    def apply_physics_randomization(
        self,
        asset: Articulation | RigidObject,
        env_ids: torch.Tensor | None = None,
        body_ids: list[int] | None = None,
    ):
        """Apply physics randomization to an asset.

        Applies previously sampled mass and material values to the asset.

        Args:
            asset: The Isaac Lab asset to modify.
            env_ids: Environment indices to apply. If None, applies to all.
            body_ids: Specific body indices. If None, modifies all bodies.
        """
        if not self.cfg.enabled:
            return

        self.physics_randomizer.apply_mass_randomization(asset, env_ids, body_ids)
        self.physics_randomizer.apply_material_randomization(asset, env_ids)

    def apply_gimbal_dynamics(
        self,
        articulation: Articulation,
        env_ids: torch.Tensor | None = None,
        gimbal_joint_ids: list[int] | None = None,
    ):
        """Apply gimbal dynamics randomization to an articulation.

        Applies previously sampled stiffness/damping scales.

        Args:
            articulation: The articulation asset containing gimbal joints.
            env_ids: Environment indices to apply. If None, applies to all.
            gimbal_joint_ids: Joint indices for gimbal. If None, uses [0, 1, 2].
        """
        if not self.cfg.enabled or not self.cfg.gimbal.enabled:
            return

        self.gimbal_randomizer.apply_dynamics_randomization(
            articulation, env_ids, gimbal_joint_ids
        )

    # =========================================================================
    # Camera Processing
    # =========================================================================

    def process_images(
        self,
        images: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process rendered images through the camera randomization pipeline.

        Applies crop and resize based on randomized FOV and resolution parameters.

        Args:
            images: Rendered images with shape (N, C, H, W) or (N, H, W, C).
            output_size: Output (H, W) size. If None, uses (360, 640) default.

        Returns:
            processed: Processed images with shape (N, C, H_out, W_out).
            intrinsics: Adjusted intrinsic matrices with shape (N, 3, 3).
        """
        return self.camera_processor.process_images_batched(images, output_size)

    # =========================================================================
    # Getter Methods
    # =========================================================================

    def get_gimbal_offsets(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get gimbal joint offsets to apply to position targets.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Joint offsets with shape (num_envs, num_agents, 3).
            Columns are yaw, pitch, roll offsets in radians.
        """
        return self.gimbal_randomizer.get_joint_offsets(env_ids)

    def get_intrinsic_matrices(self) -> torch.Tensor:
        """Get camera intrinsic matrices.

        Returns:
            Intrinsic matrices with shape (num_envs, num_agents, 3, 3).
        """
        return self.camera_processor.get_intrinsic_matrices()

    def get_fov_scales(self) -> torch.Tensor:
        """Get camera FOV scale factors.

        Returns:
            FOV scales with shape (num_envs, num_agents).
        """
        return self.camera_processor.get_fov_scales()

    def get_mass_scales(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get mass scale factors.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Mass scales with shape (num_envs,).
        """
        return self.physics_randomizer.get_mass_scales(env_ids)

    def get_payload_masses(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get payload masses.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Payload masses in kg with shape (num_envs,).
        """
        return self.physics_randomizer.get_payload_masses(env_ids)

    # =========================================================================
    # Component Access
    # =========================================================================

    def get_camera_processor(self) -> CameraProcessor:
        """Get the camera processor component.

        Returns:
            The CameraProcessor instance.
        """
        return self.camera_processor

    def get_physics_randomizer(self) -> PhysicsRandomizer:
        """Get the physics randomizer component.

        Returns:
            The PhysicsRandomizer instance.
        """
        return self.physics_randomizer

    def get_gimbal_randomizer(self) -> GimbalRandomizer:
        """Get the gimbal randomizer component.

        Returns:
            The GimbalRandomizer instance.
        """
        return self.gimbal_randomizer

    # =========================================================================
    # State Summary
    # =========================================================================

    def get_current_state_summary(self) -> dict:
        """Get a summary of current randomization state for logging.

        Returns:
            Dictionary with current parameter statistics.
        """
        return {
            "camera": {
                "fov_scale_mean": self.camera_processor.fov_scales.mean().item(),
                "fov_scale_std": self.camera_processor.fov_scales.std().item(),
                "focal_length_mean": self.camera_processor.focal_lengths.mean().item(),
            },
            "physics": {
                "mass_scale_mean": self.physics_randomizer.mass_scales.mean().item(),
                "mass_scale_std": self.physics_randomizer.mass_scales.std().item(),
                "payload_mass_mean": self.physics_randomizer.payload_masses.mean().item(),
            },
            "gimbal": {
                "yaw_offset_mean": self.gimbal_randomizer.joint_offsets[:, :, 0]
                .mean()
                .item(),
                "pitch_offset_mean": self.gimbal_randomizer.joint_offsets[:, :, 1]
                .mean()
                .item(),
                "roll_offset_mean": self.gimbal_randomizer.joint_offsets[:, :, 2]
                .mean()
                .item(),
            },
        }
