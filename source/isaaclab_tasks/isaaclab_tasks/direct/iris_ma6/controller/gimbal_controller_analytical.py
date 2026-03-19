# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gimbal controller with decoupled LOS pointing and horizon stabilization.

Architecture — analytical decomposition with decoupled roll:

  yaw, pitch = _world_to_body_angles(azimuth, elevation, q_body)  [LOS: 2-DOF]
  roll       = _compute_stabilizing_roll(yaw, q_body)              [horizon: 1-DOF]

LOS pointing (yaw + pitch) is computed analytically from the world-frame target
direction projected into the body frame. Roll for horizon leveling is computed
independently. Roll saturation cannot affect LOS accuracy.

This follows the reference math Section 5 (LOS form): full 3-DOF stabilization
(omega_IG^G = 0) is "stronger than necessary". LOS only needs the 2 angular
velocity components perpendicular to the optical axis to vanish.

Frame Conventions:
- WORLD (ENU):        +X=East, +Y=North, +Z=Up
- BODY (FLU):         +X=Forward, +Y=Left, +Z=Up  (physics frame)
- VISUAL:             +Y=Forward (mesh rotated 90 deg CCW), compensated by YAW_JOINT_OFFSET
- GIMBAL BASE:        = Body frame (attached to drone body)
- CAMERA:             Final orientation after Yaw(Z)->Roll(X)->Pitch(Y)

Yaw Offset:
- The body mesh has xformOp:orient = R_z(90 deg), so visual forward = body +Y.
- The controller works in body +X forward convention internally.
- The env code adds YAW_JOINT_OFFSET (-pi/2) when setting joint targets.
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
)

from .gimbal_controller_cfg import GimbalControllerCfg

# Body mesh is rotated 90 deg CCW from physics frame (visual forward = body +Y).
# This offset is added to yaw joint targets in the env code so that controller
# yaw=0 maps to body +X (physics forward). Exported for use by the env.
YAW_JOINT_OFFSET = -math.pi / 2


class GimbalController:
    """Gimbal controller with decoupled LOS pointing and horizon stabilization.

    Analytical position control:
    - Yaw/pitch: direct atan2 decomposition from target direction (exact, no coupling)
    - Roll: independent horizon stabilization (saturation does not affect LOS)
    - Velocity: finite difference of positions (coherent with position)
    """

    def __init__(
        self,
        cfg: GimbalControllerCfg,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        # State: current joint angle targets in BODY frame
        self._yaw = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._roll = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._pitch = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # World-frame target angles (rate-integrated pointing direction)
        self._azimuth_world = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._elevation_world = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        self._yaw_limits = cfg.yaw_limits
        self._pitch_limits = cfg.pitch_limits
        self._roll_limits = cfg.roll_limits

    @property
    def yaw(self) -> torch.Tensor:
        return self._yaw

    @property
    def roll(self) -> torch.Tensor:
        return self._roll

    @property
    def pitch(self) -> torch.Tensor:
        return self._pitch

    @property
    def azimuth_world(self) -> torch.Tensor:
        return self._azimuth_world

    @property
    def elevation_world(self) -> torch.Tensor:
        return self._elevation_world

    def compute_control(
        self,
        gimbal_yaw_rate_cmd: torch.Tensor,
        gimbal_pitch_rate_cmd: torch.Tensor,
        q_body: torch.Tensor,
        dt: float,
        omega_body: torch.Tensor,
        joint_positions_actual: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Compute gimbal joint position and velocity targets.

        Uses analytical angle decomposition with decoupled roll:
          yaw, pitch = atan2 decomposition of target direction in body frame
          roll       = horizon stabilization from world-up projection

        Args:
            gimbal_yaw_rate_cmd: (N,) normalized [-1, 1].
            gimbal_pitch_rate_cmd: (N,) normalized [-1, 1].
            q_body: (N, 4) body quaternion (wxyz), body frame = FLU.
            dt: Physics timestep [s].
            omega_body: (N, 3) body angular velocity in body frame [rad/s] (from gyro).
                Not used by the analytical position loop (body rotation is already
                accounted for via q_body), but kept in the interface for compatibility.
            joint_positions_actual: (N, 3) actual joint positions [pitch, yaw, roll].

        Returns:
            pos_targets: (yaw, roll, pitch) joint position targets [rad].
            vel_targets: (yaw, roll, pitch) joint velocity targets [rad/s].
        """
        # -- 1. Feedback blend: correct internal state drift from actual joints --
        if self.cfg.feedback_blend > 0.0:
            actual_pitch = joint_positions_actual[:, 0]
            actual_yaw = joint_positions_actual[:, 1] - YAW_JOINT_OFFSET
            actual_roll = joint_positions_actual[:, 2]
            beta = self.cfg.feedback_blend
            self._yaw = self._yaw + beta * (actual_yaw - self._yaw)
            self._roll = self._roll + beta * (actual_roll - self._roll)
            self._pitch = self._pitch + beta * (actual_pitch - self._pitch)

        # -- 2. World-frame rate integration --
        azimuth_rate = gimbal_yaw_rate_cmd * self.cfg.max_gimbal_rate
        elevation_rate = gimbal_pitch_rate_cmd * self.cfg.max_gimbal_rate

        self._azimuth_world = self._azimuth_world + azimuth_rate * dt
        self._elevation_world = self._elevation_world + elevation_rate * dt

        self._azimuth_world = torch.atan2(
            torch.sin(self._azimuth_world), torch.cos(self._azimuth_world)
        )
        self._elevation_world = torch.clamp(
            self._elevation_world, self._pitch_limits[0], self._pitch_limits[1]
        )

        # Save previous positions for finite-difference velocity
        yaw_prev = self._yaw.clone()
        roll_prev = self._roll.clone()
        pitch_prev = self._pitch.clone()

        # -- 3. Compute yaw and pitch analytically (LOS: 2-DOF) --
        # Direct atan2 decomposition: exact at all angles, no coupling with roll
        yaw_new, pitch_new = self._world_to_body_angles(
            self._azimuth_world, self._elevation_world, q_body
        )

        # -- 4. Compute roll independently (horizon: 1-DOF) --
        # Roll saturation only affects horizon leveling, not LOS accuracy
        roll_new = self._compute_stabilizing_roll(yaw_new, q_body)

        # -- 5. Update state --
        self._yaw = yaw_new
        self._pitch = pitch_new
        self._roll = roll_new

        # -- 6. Clamp to joint limits --
        self._yaw = torch.clamp(self._yaw, self._yaw_limits[0], self._yaw_limits[1])
        self._pitch = torch.clamp(self._pitch, self._pitch_limits[0], self._pitch_limits[1])
        self._roll = torch.clamp(self._roll, self._roll_limits[0], self._roll_limits[1])

        # -- 7. Compute velocity as finite difference --
        # Position already accounts for body rotation (q_body used each step),
        # so the finite difference naturally captures stabilization rates.
        inv_dt = 1.0 / dt
        yaw_vel = (self._yaw - yaw_prev) * inv_dt
        roll_vel = (self._roll - roll_prev) * inv_dt
        pitch_vel = (self._pitch - pitch_prev) * inv_dt

        # -- 8. Output --
        pos_targets = (self._yaw, self._roll, self._pitch)
        vel_targets = (yaw_vel, roll_vel, pitch_vel)
        return pos_targets, vel_targets

    # ------------------------------------------------------------------
    # Analytical angle decomposition
    # ------------------------------------------------------------------

    def _world_to_body_angles(
        self,
        azimuth_world: torch.Tensor,
        elevation_world: torch.Tensor,
        q_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world-frame pointing to body-frame yaw/pitch.

        Projects the world-frame target direction into the body frame and
        extracts yaw (rotation in body XY plane) and pitch (elevation from
        body XY plane). Independent of roll.

        Frame: world (ENU) -> body (FLU) via q_body inverse rotation.
        """
        cos_az = torch.cos(azimuth_world)
        sin_az = torch.sin(azimuth_world)
        cos_el = torch.cos(elevation_world)
        sin_el = torch.sin(elevation_world)

        dir_world = torch.stack([cos_el * cos_az, cos_el * sin_az, sin_el], dim=-1)
        dir_body = quat_rotate_inverse(q_body, dir_world)

        yaw_body = torch.atan2(dir_body[:, 1], dir_body[:, 0])
        xy_dist = torch.sqrt(dir_body[:, 0] ** 2 + dir_body[:, 1] ** 2)
        pitch_body = -torch.atan2(dir_body[:, 2], xy_dist)

        return yaw_body, pitch_body

    def _compute_stabilizing_roll(
        self,
        gimbal_yaw: torch.Tensor,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Compute roll angle to keep horizon level.

        Projects world-up into the yawed gimbal frame and computes the roll
        needed to align the camera up-axis with the projected world-up.

        Frame: world up -> body frame -> yawed frame -> roll angle.
        roll = atan2(-up_y_yawed, up_z_yawed)
        """
        world_up = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        world_up[:, 2] = 1.0
        up_in_body = quat_rotate_inverse(q_body, world_up)

        cos_yaw = torch.cos(gimbal_yaw)
        sin_yaw = torch.sin(gimbal_yaw)

        up_y_yawed = -up_in_body[:, 0] * sin_yaw + up_in_body[:, 1] * cos_yaw
        up_z_yawed = up_in_body[:, 2]

        stabilizing_roll = torch.atan2(-up_y_yawed, up_z_yawed)
        return torch.clamp(stabilizing_roll, self._roll_limits[0], self._roll_limits[1])

    # ------------------------------------------------------------------
    # Utility methods (used by env code)
    # ------------------------------------------------------------------

    def _body_to_world_angles(
        self,
        yaw_body: torch.Tensor,
        pitch_body: torch.Tensor,
        q_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert body-frame yaw/pitch back to world-frame azimuth/elevation.

        Inverse of _world_to_body_angles (assumes roll=0).
        """
        cos_p = torch.cos(pitch_body)
        dir_body = torch.stack([
            cos_p * torch.cos(yaw_body),
            cos_p * torch.sin(yaw_body),
            -torch.sin(pitch_body),
        ], dim=-1)

        dir_world = quat_rotate(q_body, dir_body)

        azimuth_world = torch.atan2(dir_world[:, 1], dir_world[:, 0])
        xy_dist = torch.sqrt(dir_world[:, 0] ** 2 + dir_world[:, 1] ** 2)
        elevation_world = torch.atan2(dir_world[:, 2], xy_dist)

        return azimuth_world, elevation_world

    def get_camera_direction_world(
        self,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Get camera pointing direction in world frame.

        Applies Yaw(Z)->Roll(X)->Pitch(Y) to [1,0,0] in body frame,
        then transforms to world frame.
        """
        forward = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        forward[:, 0] = 1.0
        direction = self._apply_gimbal_rotation(forward)
        return quat_rotate(q_body, direction)

    def _apply_gimbal_rotation(self, vec: torch.Tensor) -> torch.Tensor:
        """Apply Yaw(Z)->Roll(X)->Pitch(Y) rotation to a vector in body frame."""
        cos_yaw, sin_yaw = torch.cos(self._yaw), torch.sin(self._yaw)
        cos_roll, sin_roll = torch.cos(self._roll), torch.sin(self._roll)
        cos_pitch, sin_pitch = torch.cos(self._pitch), torch.sin(self._pitch)

        # Yaw (Z)
        x1 = vec[:, 0] * cos_yaw - vec[:, 1] * sin_yaw
        y1 = vec[:, 0] * sin_yaw + vec[:, 1] * cos_yaw
        z1 = vec[:, 2]
        # Roll (X in yawed frame)
        x2 = x1
        y2 = y1 * cos_roll - z1 * sin_roll
        z2 = y1 * sin_roll + z1 * cos_roll
        # Pitch (Y in yawed+rolled frame)
        x3 = x2 * cos_pitch + z2 * sin_pitch
        y3 = y2
        z3 = -x2 * sin_pitch + z2 * cos_pitch

        return torch.stack([x3, y3, z3], dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset gimbal state."""
        init_yaw = self.cfg.initial_yaw
        if env_ids is None:
            self._yaw.fill_(init_yaw)
            self._roll.zero_()
            self._pitch.zero_()
            self._azimuth_world.fill_(init_yaw)
            self._elevation_world.zero_()
        else:
            self._yaw[env_ids] = init_yaw
            self._roll[env_ids] = 0.0
            self._pitch[env_ids] = 0.0
            self._azimuth_world[env_ids] = init_yaw
            self._elevation_world[env_ids] = 0.0
