# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gimbal joint and dynamics randomization for domain randomization.

This module provides randomization of gimbal system properties including:
- Joint position offsets (yaw, pitch, roll biases)
- Joint dynamics (stiffness, damping, friction)

The randomizer stores offsets that the environment applies to gimbal
position targets, simulating mechanical tolerances and calibration errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .domain_randomization_cfg import GimbalRandomizationCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from ..controller.gimbal_rate_loop import GimbalRateLoop
    from ..controller.gimbal_rate_loop_cfg import GimbalRateLoopCfg


class GimbalRandomizer:
    """Gimbal joint and dynamics randomizer.

    This class samples and stores gimbal randomization parameters:
    - Joint offsets: Bias values added to gimbal position targets
    - Dynamics: Stiffness/damping scale factors for actuators

    The joint offsets simulate mechanical misalignment and calibration
    errors in real gimbal systems.

    Example:
        ```python
        cfg = GimbalRandomizationCfg()
        randomizer = GimbalRandomizer(cfg, num_envs=256, num_agents=3, device="cuda")

        # Randomize offsets for some environments
        randomizer.randomize_joint_offsets(env_ids=torch.arange(128))

        # Get offsets to apply to gimbal targets
        offsets = randomizer.get_joint_offsets()  # (num_envs, num_agents, 3)

        # In environment: apply offsets to position targets
        gimbal_targets_corrected = gimbal_targets + offsets
        ```
    """

    def __init__(
        self,
        cfg: GimbalRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
    ):
        """Initialize the gimbal randomizer.

        Args:
            cfg: Gimbal randomization configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: PyTorch device for tensor allocation.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-allocate storage for joint offsets
        # Shape: (num_envs, num_agents, 3) for yaw, pitch, roll
        self.joint_offsets = torch.zeros(
            num_envs, num_agents, 3, device=self.device, dtype=torch.float32
        )

        # Dynamics scale factors
        # Shape: (num_envs, num_agents) - same for all joints per gimbal
        self.stiffness_scales = torch.ones(
            num_envs, num_agents, device=self.device, dtype=torch.float32
        )
        self.damping_scales = torch.ones(
            num_envs, num_agents, device=self.device, dtype=torch.float32
        )
        self.friction_values = torch.zeros(
            num_envs, num_agents, device=self.device, dtype=torch.float32
        )

        # Initialize with default (non-randomized) values
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize all parameters with default (non-randomized) values."""
        # Zero offsets (no bias)
        self.joint_offsets.zero_()

        # Unit scale factors (no modification)
        self.stiffness_scales.fill_(1.0)
        self.damping_scales.fill_(1.0)
        self.friction_values.zero_()

    def randomize_joint_offsets(self, env_ids: torch.Tensor | None = None):
        """Randomize gimbal joint position offsets.

        Samples yaw, pitch, and roll offsets according to configuration.
        These offsets are added to gimbal position targets to simulate
        mechanical misalignment.

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.enabled or not self.cfg.randomize_offsets_per_episode:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        num_envs = len(env_ids)

        # Sample offsets for each joint
        # Yaw offset
        yaw_offsets = torch.empty(
            num_envs, self.num_agents, device=self.device
        ).uniform_(self.cfg.yaw_offset_range[0], self.cfg.yaw_offset_range[1])

        # Pitch offset
        pitch_offsets = torch.empty(
            num_envs, self.num_agents, device=self.device
        ).uniform_(self.cfg.pitch_offset_range[0], self.cfg.pitch_offset_range[1])

        # Roll offset
        roll_offsets = torch.empty(
            num_envs, self.num_agents, device=self.device
        ).uniform_(self.cfg.roll_offset_range[0], self.cfg.roll_offset_range[1])

        # Store offsets: order is yaw, pitch, roll
        self.joint_offsets[env_ids, :, 0] = yaw_offsets
        self.joint_offsets[env_ids, :, 1] = pitch_offsets
        self.joint_offsets[env_ids, :, 2] = roll_offsets

    def randomize_dynamics(self, env_ids: torch.Tensor | None = None):
        """Randomize gimbal joint dynamics (stiffness, damping, friction).

        Samples scale factors for stiffness and damping, and absolute
        friction values according to configuration.

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.enabled or not self.cfg.randomize_dynamics_per_episode:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        num_envs = len(env_ids)

        # Sample stiffness scale
        self.stiffness_scales[env_ids] = torch.empty(
            num_envs, self.num_agents, device=self.device
        ).uniform_(
            self.cfg.stiffness_scale_range[0], self.cfg.stiffness_scale_range[1]
        )

        # Sample damping scale
        self.damping_scales[env_ids] = torch.empty(
            num_envs, self.num_agents, device=self.device
        ).uniform_(self.cfg.damping_scale_range[0], self.cfg.damping_scale_range[1])

        # Sample friction
        self.friction_values[env_ids] = torch.empty(
            num_envs, self.num_agents, device=self.device
        ).uniform_(self.cfg.friction_range[0], self.cfg.friction_range[1])

    def apply_dynamics_randomization(
        self,
        articulation: Articulation,
        env_ids: torch.Tensor | None = None,
        gimbal_joint_ids: list[int] | None = None,
    ):
        """Apply dynamics randomization to gimbal joints in simulation.

        Modifies joint stiffness, damping, and friction for gimbal actuators.

        Args:
            articulation: The articulation asset containing gimbal joints.
            env_ids: Environment indices to apply. If None, applies to all.
            gimbal_joint_ids: Joint indices for yaw, pitch, roll joints.
                             If None, assumes joints 0, 1, 2.

        Note:
            This requires an articulation reference and modifies simulation state.
            For multi-agent setups, call once per agent with appropriate joint IDs.
        """
        if not self.cfg.enabled or not self.cfg.randomize_dynamics_per_episode:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if gimbal_joint_ids is None:
            gimbal_joint_ids = [0, 1, 2]  # Default: first 3 joints

        # Get default stiffness and damping
        default_stiffness = articulation.data.default_joint_stiffness[
            env_ids.unsqueeze(-1), gimbal_joint_ids
        ]
        default_damping = articulation.data.default_joint_damping[
            env_ids.unsqueeze(-1), gimbal_joint_ids
        ]

        # Apply scale factors (same scale for all 3 joints per gimbal)
        # For simplicity, use agent 0 scales (multi-agent needs per-agent articulations)
        scales_stiffness = self.stiffness_scales[env_ids, 0].unsqueeze(-1)
        scales_damping = self.damping_scales[env_ids, 0].unsqueeze(-1)

        new_stiffness = default_stiffness * scales_stiffness
        new_damping = default_damping * scales_damping

        # Write to simulation
        articulation.write_joint_stiffness_to_sim(
            new_stiffness, joint_ids=gimbal_joint_ids, env_ids=env_ids
        )
        articulation.write_joint_damping_to_sim(
            new_damping, joint_ids=gimbal_joint_ids, env_ids=env_ids
        )

        # Note: Joint friction requires different API, handled separately if needed

    def get_joint_offsets(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get current joint offsets for specified environments.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Joint offsets with shape (num_envs, num_agents, 3).
            Columns are yaw, pitch, roll offsets in radians.
        """
        if env_ids is None:
            return self.joint_offsets.clone()
        return self.joint_offsets[env_ids].clone()

    def get_yaw_offsets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get yaw joint offsets.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Yaw offsets with shape (num_envs, num_agents) in radians.
        """
        if env_ids is None:
            return self.joint_offsets[:, :, 0].clone()
        return self.joint_offsets[env_ids, :, 0].clone()

    def get_pitch_offsets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get pitch joint offsets.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Pitch offsets with shape (num_envs, num_agents) in radians.
        """
        if env_ids is None:
            return self.joint_offsets[:, :, 1].clone()
        return self.joint_offsets[env_ids, :, 1].clone()

    def get_roll_offsets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get roll joint offsets.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Roll offsets with shape (num_envs, num_agents) in radians.
        """
        if env_ids is None:
            return self.joint_offsets[:, :, 2].clone()
        return self.joint_offsets[env_ids, :, 2].clone()

    def get_stiffness_scales(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get current stiffness scale factors.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Stiffness scales with shape (num_envs, num_agents).
        """
        if env_ids is None:
            return self.stiffness_scales.clone()
        return self.stiffness_scales[env_ids].clone()

    def get_damping_scales(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get current damping scale factors.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Damping scales with shape (num_envs, num_agents).
        """
        if env_ids is None:
            return self.damping_scales.clone()
        return self.damping_scales[env_ids].clone()

    # ------------------------------------------------------------------
    # mas/035: Rate-loop τ randomization (replaces retired stiffness/damping DR)
    # ------------------------------------------------------------------

    def randomize_rate_loop_tau(
        self,
        rate_loop: "GimbalRateLoop",
        rate_loop_cfg: "GimbalRateLoopCfg",
        env_ids: torch.Tensor | None = None,
    ):
        """Sample per-env τ for the gimbal rate loop and write into the loop.

        Multiplicative scale factors are drawn from
        `rate_loop_cfg.tau_scale_range_yaw / tau_scale_range_pitch`. The
        nominal τ values come from `rate_loop_cfg.tau_yaw_s / tau_pitch_s`.
        Use this in place of the legacy `randomize_dynamics` path; that one
        is a no-op now that joint-PD scale ranges are retired.

        Args:
            rate_loop: The `GimbalRateLoop` instance to write into.
            rate_loop_cfg: Source config for τ nominals and scale ranges.
            env_ids: Environment indices to randomize. None randomizes all.
        """
        if not self.cfg.enabled or not self.cfg.randomize_dynamics_per_episode:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        n = len(env_ids)
        scale_yaw = torch.empty(n, device=self.device).uniform_(
            rate_loop_cfg.tau_scale_range_yaw[0],
            rate_loop_cfg.tau_scale_range_yaw[1],
        )
        scale_pitch = torch.empty(n, device=self.device).uniform_(
            rate_loop_cfg.tau_scale_range_pitch[0],
            rate_loop_cfg.tau_scale_range_pitch[1],
        )
        tau_yaw = scale_yaw * rate_loop_cfg.tau_yaw_s
        tau_pitch = scale_pitch * rate_loop_cfg.tau_pitch_s
        rate_loop.set_tau_per_env(tau_yaw, tau_pitch, env_ids=env_ids)
