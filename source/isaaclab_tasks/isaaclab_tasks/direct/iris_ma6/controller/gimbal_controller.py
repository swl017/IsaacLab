# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gimbal controller with gyro-based Jacobian feedforward stabilization.

Architecture (mirrors real gimbal controllers):
- POSITION LOOP: Computes desired joint angles from world-frame pointing direction.
  Uses simple Euler decomposition (yaw from atan2, pitch from atan2, roll from up-vector).
  Handles steady-state pointing. Approximate at large combined angles — the velocity
  feedforward compensates for the decomposition error dynamically.
- VELOCITY FEEDFORWARD: Projects body angular velocity (from gyro) onto each joint axis
  via the gimbal Jacobian, then negates to cancel body rotation. Exact at all angles.
  This is what makes real gimbals work — it directly cancels body rotation without
  relying on feedback error to accumulate.

Frame Conventions:
- WORLD (ENU):        +X=East, +Y=North, +Z=Up
- BODY (FLU):         +X=Forward, +Y=Left, +Z=Up  (physics frame)
- VISUAL:             +Y=Forward (mesh rotated 90° CCW), compensated by YAW_JOINT_OFFSET
- GIMBAL BASE:        = Body frame (attached to drone body)
- AFTER YAW R_z(ψ):  Body frame rotated ψ around Z
- AFTER YAW+ROLL:     Further rotated φ around the yawed X axis
- CAMERA:             Final orientation after Yaw(Z)→Roll(X)→Pitch(Y)

Gimbal Jacobian (maps joint velocities → body-frame angular velocity):
  J = [[ 0,      cos(ψ),    -sin(ψ)·cos(φ) ],
       [ 0,      sin(ψ),     cos(ψ)·cos(φ) ],
       [ 1,      0,           sin(φ)        ]]

  Joint axes in body frame:
    Yaw:   [0, 0, 1]                            (body Z)
    Roll:  [cos(ψ), sin(ψ), 0]                  (R_z(ψ) · [1,0,0])
    Pitch: [-sin(ψ)·cos(φ), cos(ψ)·cos(φ), sin(φ)]  (R_z(ψ)·R_x(φ) · [0,1,0])

  Inverse Jacobian (det(J) = cos(φ), singular at φ=±90°):
    J⁻¹ = [[ sy·sr/cr,  -cy·sr/cr,  1 ],
            [ cy,         sy,         0 ],
            [-sy/cr,      cy/cr,      0 ]]

  Feedforward: [ψ̇, φ̇, θ̇] = -J⁻¹ · ω_body

Yaw Offset:
- The body mesh has xformOp:orient = R_z(90°), so visual forward = body +Y.
- The controller works in body +X forward convention internally.
- The env code adds YAW_JOINT_OFFSET (-π/2) when setting joint targets.
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

from .gimbal_controller_cfg import GimbalControllerCfg

# Body mesh is rotated 90° CCW from physics frame (visual forward = body +Y).
# This offset is added to yaw joint targets in the env code so that controller
# yaw=0 maps to body +X (physics forward). Exported for use by the env.
YAW_JOINT_OFFSET = -math.pi / 2


class GimbalController:
    """Gimbal controller with gyro Jacobian feedforward.

    Two-path control:
    1. Position loop: world→body angle decomposition (steady-state pointing)
    2. Velocity feedforward: J⁻¹·(-ω_body) (transient stabilization)
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

        Args:
            gimbal_yaw_rate_cmd: (N,) normalized [-1, 1].
            gimbal_pitch_rate_cmd: (N,) normalized [-1, 1].
            q_body: (N, 4) body quaternion (wxyz), body frame = FLU.
            dt: Physics timestep [s].
            omega_body: (N, 3) body angular velocity in body frame [rad/s] (from gyro).
            joint_positions_actual: (N, 3) actual joint positions [pitch, yaw, roll].

        Returns:
            pos_targets: (yaw, roll, pitch) joint position targets [rad].
            vel_targets: (yaw, roll, pitch) joint velocity feedforward [rad/s].
        """
        # ── 1. World-frame rate integration ───────────────────────────
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

        # ── 2. Position loop: world→body angle decomposition ─────────
        # Simple decomposition: yaw from XY, pitch from Z, roll from up-vector.
        # This is approximate at large combined angles, but the velocity
        # feedforward handles the transient coupling exactly.
        yaw_target, pitch_target = self._world_to_body_angles(
            self._azimuth_world, self._elevation_world, q_body
        )
        if self.cfg.auto_stabilize_roll:
            roll_target = self._compute_stabilizing_roll(yaw_target, q_body)
        else:
            roll_target = torch.zeros_like(yaw_target)

        # Clamp to joint limits
        self._yaw = torch.clamp(yaw_target, self._yaw_limits[0], self._yaw_limits[1])
        self._pitch = torch.clamp(pitch_target, self._pitch_limits[0], self._pitch_limits[1])
        self._roll = torch.clamp(roll_target, self._roll_limits[0], self._roll_limits[1])

        # Back-propagate clamped angles to world-frame targets
        needs_backprop = (self._yaw != yaw_target) | (self._pitch != pitch_target)
        if needs_backprop.any():
            az_new, el_new = self._body_to_world_angles(self._yaw, self._pitch, q_body)
            self._azimuth_world = torch.where(needs_backprop, az_new, self._azimuth_world)
            self._elevation_world = torch.where(needs_backprop, el_new, self._elevation_world)

        # ── 3. Closed-loop feedback on world-frame state ──────────────
        beta = self.cfg.feedback_blend
        if beta > 0:
            pitch_actual = joint_positions_actual[:, 0]
            yaw_actual = joint_positions_actual[:, 1] - YAW_JOINT_OFFSET

            az_actual, el_actual = self._body_to_world_angles(yaw_actual, pitch_actual, q_body)
            az_diff = torch.atan2(
                torch.sin(az_actual - self._azimuth_world),
                torch.cos(az_actual - self._azimuth_world),
            )
            self._azimuth_world = self._azimuth_world + beta * az_diff
            self._elevation_world = self._elevation_world + beta * (el_actual - self._elevation_world)

            # Recompute position targets from corrected world state
            yaw_fb, pitch_fb = self._world_to_body_angles(
                self._azimuth_world, self._elevation_world, q_body
            )
            self._yaw = torch.clamp(yaw_fb, self._yaw_limits[0], self._yaw_limits[1])
            self._pitch = torch.clamp(pitch_fb, self._pitch_limits[0], self._pitch_limits[1])
            if self.cfg.auto_stabilize_roll:
                self._roll = torch.clamp(
                    self._compute_stabilizing_roll(self._yaw, q_body),
                    self._roll_limits[0], self._roll_limits[1],
                )

        # ── 4. Velocity feedforward: Jacobian-based gyro compensation ─
        # Computes joint velocities that exactly cancel body rotation.
        # Uses the inverse of the gimbal Jacobian: [ψ̇, φ̇, θ̇] = -J⁻¹ · ω_body
        yaw_vel, roll_vel, pitch_vel = self._compute_gyro_feedforward(
            omega_body, self._yaw, self._roll
        )

        pos_targets = (self._yaw, self._roll, self._pitch)
        vel_targets = (yaw_vel, roll_vel, pitch_vel)
        return pos_targets, vel_targets

    # ──────────────────────────────────────────────────────────────────
    # Position loop helpers
    # ──────────────────────────────────────────────────────────────────

    def _world_to_body_angles(
        self,
        azimuth_world: torch.Tensor,
        elevation_world: torch.Tensor,
        q_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world-frame pointing to body-frame yaw/pitch (roll=0 assumption).

        Frame: world (ENU) → body (FLU) via q_body inverse rotation.
        Yaw: atan2(dir_body_y, dir_body_x) — angle in body XY plane.
        Pitch: -atan2(dir_body_z, xy_dist) — elevation from body XY plane.
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

    def _body_to_world_angles(
        self,
        yaw_body: torch.Tensor,
        pitch_body: torch.Tensor,
        q_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert body-frame yaw/pitch back to world-frame azimuth/elevation.

        Inverse of _world_to_body_angles (assumes roll=0). Used for back-propagation.
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

    def _compute_stabilizing_roll(
        self,
        gimbal_yaw: torch.Tensor,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Compute roll angle to keep horizon level.

        Frame: world up → body frame → yawed frame → roll angle.
        roll = atan2(-up_y_yawed, up_z_yawed) where up is world-up in yawed frame.
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

    # ──────────────────────────────────────────────────────────────────
    # Velocity feedforward (Jacobian-based)
    # ──────────────────────────────────────────────────────────────────

    def _compute_gyro_feedforward(
        self,
        omega_body: torch.Tensor,
        gimbal_yaw: torch.Tensor,
        gimbal_roll: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint velocities that cancel body rotation via inverse Jacobian.

        For gimbal chain Yaw(Z)→Roll(X)→Pitch(Y):
          J⁻¹ = [[ sy·sr/cr,  -cy·sr/cr,  1 ],
                  [ cy,         sy,         0 ],
                  [-sy/cr,      cy/cr,      0 ]]

        Feedforward: [ψ̇, φ̇, θ̇] = -J⁻¹ · ω_body

        Args:
            omega_body: (N, 3) body angular velocity [ωx, ωy, ωz] in body frame (FLU).
            gimbal_yaw: (N,) current gimbal yaw ψ [rad].
            gimbal_roll: (N,) current gimbal roll φ [rad].

        Returns:
            yaw_vel, roll_vel, pitch_vel: (N,) each, joint velocity feedforward [rad/s].
        """
        cy = torch.cos(gimbal_yaw)
        sy = torch.sin(gimbal_yaw)
        cr = torch.cos(gimbal_roll)
        sr = torch.sin(gimbal_roll)

        # Avoid division by zero near gimbal lock (roll ≈ ±90°)
        inv_cr = 1.0 / (cr + 1e-6 * torch.sign(cr + 1e-8))

        wx = omega_body[:, 0]
        wy = omega_body[:, 1]
        wz = omega_body[:, 2]

        # J⁻¹ · ω_body (then negate for feedforward)
        yaw_ff = sy * sr * inv_cr * wx - cy * sr * inv_cr * wy + wz
        roll_ff = cy * wx + sy * wy
        pitch_ff = -sy * inv_cr * wx + cy * inv_cr * wy

        return -yaw_ff, -roll_ff, -pitch_ff

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    def get_camera_direction_world(
        self,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Get camera pointing direction in world frame.

        Applies Yaw(Z)→Roll(X)→Pitch(Y) to [1,0,0] in body frame,
        then transforms to world frame.
        """
        forward = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        forward[:, 0] = 1.0
        direction = self._apply_gimbal_rotation(forward)
        return quat_rotate(q_body, direction)

    def _apply_gimbal_rotation(self, vec: torch.Tensor) -> torch.Tensor:
        """Apply Yaw(Z)→Roll(X)→Pitch(Y) rotation to a vector in body frame."""
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
