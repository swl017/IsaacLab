# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gimbal controller with first-order dynamics and world-frame LOS stabilization.

Frame Convention:
- World: ENU (East-North-Up)
- Body: FLU (Forward-Left-Up)
- Gimbal Base: FLU, attached to drone body
- Joint Order: Yaw (outer) -> Roll (middle) -> Pitch (inner)
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_rotate_inverse

from .gimbal_controller_cfg import GimbalControllerCfg


class GimbalController:
    """Gimbal controller with first-order dynamics and auto roll stabilization.

    Provides world-frame line-of-sight (LOS) tracking by:
    1. Integrating rate commands to update target angles
    2. Computing stabilizing roll to keep horizon level
    3. Applying first-order lag dynamics to smooth joint motion
    """

    def __init__(
        self,
        cfg: GimbalControllerCfg,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize gimbal controller.

        Args:
            cfg: Gimbal controller configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        # State: current joint angles (after dynamics)
        self._yaw = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._roll = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._pitch = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # Target angles (before dynamics)
        self._yaw_target = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._pitch_target = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # Convert limits to tensors
        self._yaw_limits = cfg.yaw_limits
        self._pitch_limits = cfg.pitch_limits
        self._roll_limits = cfg.roll_limits

    @property
    def yaw(self) -> torch.Tensor:
        """Current gimbal yaw angle (N,) [rad]."""
        return self._yaw

    @property
    def roll(self) -> torch.Tensor:
        """Current gimbal roll angle (N,) [rad]."""
        return self._roll

    @property
    def pitch(self) -> torch.Tensor:
        """Current gimbal pitch angle (N,) [rad]."""
        return self._pitch

    def compute_control(
        self,
        gimbal_yaw_rate_cmd: torch.Tensor,
        gimbal_pitch_rate_cmd: torch.Tensor,
        q_body: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gimbal joint position targets.

        Args:
            gimbal_yaw_rate_cmd: (N,) yaw rate command, normalized [-1, 1].
            gimbal_pitch_rate_cmd: (N,) pitch rate command, normalized [-1, 1].
            q_body: (N, 4) body quaternion (wxyz).
            dt: Timestep [s].

        Returns:
            yaw_target: (N,) yaw joint target [rad].
            roll_target: (N,) roll joint target [rad] (stabilizing).
            pitch_target: (N,) pitch joint target [rad].
        """
        # Integrate rate commands to get target angles
        yaw_rate = gimbal_yaw_rate_cmd * self.cfg.max_gimbal_rate
        pitch_rate = gimbal_pitch_rate_cmd * self.cfg.max_gimbal_rate

        self._yaw_target = self._yaw_target + yaw_rate * dt
        self._pitch_target = self._pitch_target + pitch_rate * dt

        # Clamp to limits
        self._yaw_target = torch.clamp(
            self._yaw_target, self._yaw_limits[0], self._yaw_limits[1]
        )
        self._pitch_target = torch.clamp(
            self._pitch_target, self._pitch_limits[0], self._pitch_limits[1]
        )

        # Compute stabilizing roll
        if self.cfg.auto_stabilize_roll:
            roll_target = self._compute_stabilizing_roll(
                self._yaw_target, self._pitch_target, q_body
            )
        else:
            roll_target = torch.zeros_like(self._yaw_target)

        # Apply first-order lag dynamics with exact discretization
        alpha = 1.0 - torch.exp(torch.tensor(-dt / self.cfg.tau_gimbal, device=self.device))
        self._yaw = self._yaw + alpha * (self._yaw_target - self._yaw)
        self._pitch = self._pitch + alpha * (self._pitch_target - self._pitch)
        self._roll = self._roll + alpha * (roll_target - self._roll)

        # Clamp final outputs
        self._yaw = torch.clamp(self._yaw, self._yaw_limits[0], self._yaw_limits[1])
        self._pitch = torch.clamp(self._pitch, self._pitch_limits[0], self._pitch_limits[1])
        self._roll = torch.clamp(self._roll, self._roll_limits[0], self._roll_limits[1])

        return self._yaw, self._roll, self._pitch

    def _compute_stabilizing_roll(
        self,
        gimbal_yaw: torch.Tensor,
        gimbal_pitch: torch.Tensor,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Compute roll angle to keep horizon level in camera view.

        Args:
            gimbal_yaw: (N,) gimbal yaw angle [rad].
            gimbal_pitch: (N,) gimbal pitch angle [rad].
            q_body: (N, 4) body quaternion (wxyz).

        Returns:
            stabilizing_roll: (N,) roll angle [rad].
        """
        # World up vector in ENU
        world_up = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        world_up[:, 2] = 1.0

        # Transform world up to body frame (gimbal base frame)
        up_in_body = quat_rotate_inverse(q_body, world_up)

        # Rotate by inverse yaw to get up in yawed frame
        cos_yaw = torch.cos(gimbal_yaw)
        sin_yaw = torch.sin(gimbal_yaw)

        up_x_yawed = up_in_body[:, 0] * cos_yaw + up_in_body[:, 1] * sin_yaw
        up_y_yawed = -up_in_body[:, 0] * sin_yaw + up_in_body[:, 1] * cos_yaw
        up_z_yawed = up_in_body[:, 2]

        # Compute stabilizing roll using geometry
        cos_pitch = torch.cos(gimbal_pitch)

        # Roll angle that aligns camera up with world up (projected)
        stabilizing_roll = torch.atan2(-up_y_yawed, up_z_yawed / (cos_pitch + 1e-8))

        # Clamp to limits
        stabilizing_roll = torch.clamp(
            stabilizing_roll, self._roll_limits[0], self._roll_limits[1]
        )

        return stabilizing_roll

    def get_camera_direction_world(
        self,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Get camera pointing direction in world frame.

        Args:
            q_body: (N, 4) body quaternion (wxyz).

        Returns:
            direction: (N, 3) camera pointing direction in world frame.
        """
        # Camera points along +X in gimbal frame after all rotations
        # Start with forward vector [1, 0, 0]
        forward = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        forward[:, 0] = 1.0

        # Apply gimbal rotations: Yaw -> Roll -> Pitch
        direction = self._apply_gimbal_rotation(forward)

        # Transform from body frame to world frame
        from isaaclab.utils.math import quat_rotate
        direction_world = quat_rotate(q_body, direction)

        return direction_world

    def _apply_gimbal_rotation(self, vec: torch.Tensor) -> torch.Tensor:
        """Apply gimbal yaw-roll-pitch rotation to a vector.

        Args:
            vec: (N, 3) input vector in gimbal base frame.

        Returns:
            rotated: (N, 3) rotated vector.
        """
        # Rotation order: Yaw (Z) -> Roll (X) -> Pitch (Y)
        cos_yaw, sin_yaw = torch.cos(self._yaw), torch.sin(self._yaw)
        cos_roll, sin_roll = torch.cos(self._roll), torch.sin(self._roll)
        cos_pitch, sin_pitch = torch.cos(self._pitch), torch.sin(self._pitch)

        # Yaw rotation (around Z)
        x1 = vec[:, 0] * cos_yaw - vec[:, 1] * sin_yaw
        y1 = vec[:, 0] * sin_yaw + vec[:, 1] * cos_yaw
        z1 = vec[:, 2]

        # Roll rotation (around X in yawed frame)
        x2 = x1
        y2 = y1 * cos_roll - z1 * sin_roll
        z2 = y1 * sin_roll + z1 * cos_roll

        # Pitch rotation (around Y in yawed+rolled frame)
        x3 = x2 * cos_pitch + z2 * sin_pitch
        y3 = y2
        z3 = -x2 * sin_pitch + z2 * cos_pitch

        return torch.stack([x3, y3, z3], dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset gimbal state.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._yaw.zero_()
            self._roll.zero_()
            self._pitch.zero_()
            self._yaw_target.zero_()
            self._pitch_target.zero_()
        else:
            self._yaw[env_ids] = 0.0
            self._roll[env_ids] = 0.0
            self._pitch[env_ids] = 0.0
            self._yaw_target[env_ids] = 0.0
            self._pitch_target[env_ids] = 0.0

    def set_tau_gimbal(self, tau_gimbal: torch.Tensor | float):
        """Set gimbal time constant (for randomization).

        Args:
            tau_gimbal: Time constant [s].
        """
        if isinstance(tau_gimbal, torch.Tensor):
            self._tau_gimbal = tau_gimbal.to(self.device)
        else:
            self._tau_gimbal = tau_gimbal
