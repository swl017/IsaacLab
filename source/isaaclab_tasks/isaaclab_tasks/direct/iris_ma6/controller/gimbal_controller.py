# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gimbal controller with first-order dynamics and world-frame LOS stabilization.

Frame Convention:
- World: ENU (East-North-Up)
- Body: FLU (Forward-Left-Up), where +X = physics forward
- Visual Forward: Body +Y (mesh is rotated 90° CCW from physics frame)
- Gimbal Base: FLU, attached to drone body
- Joint Order: Yaw (outer) -> Roll (middle) -> Pitch (inner)

World-Frame Stabilization:
- Rate commands adjust world-frame pointing direction (azimuth, elevation)
- Body-frame joint angles are computed from world-frame targets + drone attitude
- Camera pointing direction is preserved in world frame when drone tilts

Yaw Offset:
- The body mesh has xformOp:orient = R_z(90°), so visual forward = body +Y.
- The controller works in body +X forward convention internally.
- The env code adds YAW_JOINT_OFFSET (π/2) when setting joint targets,
  so yaw_joint=0 in physics corresponds to visual forward (body +Y).
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import quat_rotate_inverse

from .gimbal_controller_cfg import GimbalControllerCfg

# Body mesh is rotated 90° CCW from physics frame (visual forward = body +Y).
# This offset is added to yaw joint targets in the env code to align gimbal
# yaw=0 with visual forward. Exported for use by the env.
YAW_JOINT_OFFSET = -math.pi / 2


class GimbalController:
    """Gimbal controller with world-frame LOS stabilization.

    Provides world-frame line-of-sight (LOS) tracking by:
    1. Storing target pointing direction in WORLD frame (azimuth, elevation)
    2. Computing body-frame joint angles that achieve world-frame pointing
    3. Applying first-order lag dynamics to smooth joint motion
    4. Auto-stabilizing roll to keep horizon level
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

        # State: current joint angles (after dynamics) in BODY frame
        self._yaw = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._roll = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._pitch = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # World-frame target angles (rate-integrated pointing direction)
        # Azimuth: yaw angle in world frame (0 = +X/East, positive CCW)
        # Elevation: pitch from horizon (0 = horizontal, negative = looking down)
        self._azimuth_world = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._elevation_world = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

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

    @property
    def azimuth_world(self) -> torch.Tensor:
        """World-frame azimuth angle (N,) [rad]."""
        return self._azimuth_world

    @property
    def elevation_world(self) -> torch.Tensor:
        """World-frame elevation angle (N,) [rad]."""
        return self._elevation_world

    def compute_control(
        self,
        gimbal_yaw_rate_cmd: torch.Tensor,
        gimbal_pitch_rate_cmd: torch.Tensor,
        q_body: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gimbal joint position targets for world-frame stabilization.

        Rate commands adjust WORLD-frame pointing direction (azimuth, elevation).
        Body-frame joint angles are then computed to achieve this world-frame
        orientation given the current drone attitude.

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
        # Integrate rate commands to update WORLD-frame pointing direction
        azimuth_rate = gimbal_yaw_rate_cmd * self.cfg.max_gimbal_rate
        elevation_rate = gimbal_pitch_rate_cmd * self.cfg.max_gimbal_rate

        self._azimuth_world = self._azimuth_world + azimuth_rate * dt
        self._elevation_world = self._elevation_world + elevation_rate * dt

        # Clamp world-frame angles to reasonable limits
        # Azimuth: wrap to [-pi, pi]
        self._azimuth_world = torch.atan2(
            torch.sin(self._azimuth_world), torch.cos(self._azimuth_world)
        )
        # Elevation: clamp to [-90°, +45°] (can't look straight up much)
        self._elevation_world = torch.clamp(
            self._elevation_world, self._pitch_limits[0], self._pitch_limits[1]
        )

        # Compute body-frame joint targets from world-frame pointing + drone attitude
        yaw_target, pitch_target = self._world_to_body_angles(
            self._azimuth_world, self._elevation_world, q_body
        )

        # Clamp to joint limits
        yaw_target = torch.clamp(yaw_target, self._yaw_limits[0], self._yaw_limits[1])
        pitch_target = torch.clamp(pitch_target, self._pitch_limits[0], self._pitch_limits[1])

        # Compute stabilizing roll
        if self.cfg.auto_stabilize_roll:
            roll_target = self._compute_stabilizing_roll(yaw_target, pitch_target, q_body)
        else:
            roll_target = torch.zeros_like(yaw_target)

        # Apply first-order lag dynamics with exact discretization
        alpha = 1.0 - torch.exp(torch.tensor(-dt / self.cfg.tau_gimbal, device=self.device))
        self._yaw = self._yaw + alpha * (yaw_target - self._yaw)
        self._pitch = self._pitch + alpha * (pitch_target - self._pitch)
        self._roll = self._roll + alpha * (roll_target - self._roll)

        # Clamp final outputs
        self._yaw = torch.clamp(self._yaw, self._yaw_limits[0], self._yaw_limits[1])
        self._pitch = torch.clamp(self._pitch, self._pitch_limits[0], self._pitch_limits[1])
        self._roll = torch.clamp(self._roll, self._roll_limits[0], self._roll_limits[1])

        return self._yaw, self._roll, self._pitch

    def _world_to_body_angles(
        self,
        azimuth_world: torch.Tensor,
        elevation_world: torch.Tensor,
        q_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world-frame pointing direction to body-frame gimbal angles.

        Args:
            azimuth_world: (N,) world-frame azimuth [rad].
            elevation_world: (N,) world-frame elevation [rad].
            q_body: (N, 4) body quaternion (wxyz).

        Returns:
            yaw_body: (N,) gimbal yaw in body frame [rad].
            pitch_body: (N,) gimbal pitch in body frame [rad].
        """
        # Compute desired pointing direction in world frame
        cos_az = torch.cos(azimuth_world)
        sin_az = torch.sin(azimuth_world)
        cos_el = torch.cos(elevation_world)
        sin_el = torch.sin(elevation_world)

        # World-frame pointing direction (ENU)
        # Azimuth 0 = +X (East), Elevation 0 = horizontal
        dir_world = torch.stack([
            cos_el * cos_az,  # X
            cos_el * sin_az,  # Y
            sin_el,           # Z
        ], dim=-1)

        # Transform to body frame
        dir_body = quat_rotate_inverse(q_body, dir_world)

        # Extract body-frame gimbal angles (yaw, pitch)
        # Yaw: rotation around body Z axis (positive = CCW = left)
        # atan2 gives angle from body +X forward convention.
        # The YAW_JOINT_OFFSET is applied later in env code when setting joint targets.
        yaw_body = torch.atan2(dir_body[:, 1], dir_body[:, 0])

        # Pitch: rotation around body Y axis (after yaw)
        # Project onto XZ plane in yawed frame
        xy_dist = torch.sqrt(dir_body[:, 0] ** 2 + dir_body[:, 1] ** 2)
        # Note: atan2 gives positive when target is above +X, but R_y(pitch) convention
        # expects negative pitch to rotate +X toward +Z (camera points up).
        # Negate to match the quaternion rotation convention used in visualization.
        pitch_body = -torch.atan2(dir_body[:, 2], xy_dist)

        return yaw_body, pitch_body

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

        # Rotate by gimbal yaw to get up in yawed frame
        # Controller yaw is in body +X forward convention (no offset needed).
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
        # Controller yaw is in body +X forward convention (no offset needed).
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
            self._azimuth_world.zero_()
            self._elevation_world.zero_()
        else:
            self._yaw[env_ids] = 0.0
            self._roll[env_ids] = 0.0
            self._pitch[env_ids] = 0.0
            self._azimuth_world[env_ids] = 0.0
            self._elevation_world[env_ids] = 0.0

    def set_tau_gimbal(self, tau_gimbal: torch.Tensor | float):
        """Set gimbal time constant (for randomization).

        Args:
            tau_gimbal: Time constant [s].
        """
        if isinstance(tau_gimbal, torch.Tensor):
            self._tau_gimbal = tau_gimbal.to(self.device)
        else:
            self._tau_gimbal = tau_gimbal
