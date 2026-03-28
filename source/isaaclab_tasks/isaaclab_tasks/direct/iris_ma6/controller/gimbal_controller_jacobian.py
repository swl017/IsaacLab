# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gimbal controller with unified Jacobian-inverse control law.

Architecture — unified J^{-1} pipeline (matches real gimbal stabilizers):

  q_dot_ref = J^{-1} * (omega_cmd - omega_body)
  q_ref     = q + q_dot_ref * dt

Where:
- omega_cmd = -K_pointing * att_error  (pointing/tracking rate command)
- omega_body                           (body angular velocity to reject)

Both tracking and stabilization flow through the same J^{-1}, producing
coherent position and velocity targets for the implicit actuator.

Frame Conventions:
- WORLD (ENU):        +X=East, +Y=North, +Z=Up
- BODY (FLU):         +X=Forward, +Y=Left, +Z=Up  (physics frame)
- VISUAL:             +Y=Forward (mesh rotated 90 deg CCW), compensated by YAW_JOINT_OFFSET
- GIMBAL BASE:        = Body frame (attached to drone body)
- AFTER YAW R_z(psi): Body frame rotated psi around Z
- AFTER YAW+ROLL:     Further rotated phi around the yawed X axis
- CAMERA:             Final orientation after Yaw(Z)->Roll(X)->Pitch(Y)

Gimbal Jacobian (maps joint velocities -> body-frame angular velocity):
  J = [[ 0,      cos(psi),    -sin(psi)*cos(phi) ],
       [ 0,      sin(psi),     cos(psi)*cos(phi) ],
       [ 1,      0,            sin(phi)           ]]

  Inverse Jacobian (det(J) = cos(phi), singular at phi=+/-90 deg):
    J^{-1} = [[ sy*sr/cr,  -cy*sr/cr,  1 ],
              [ cy,         sy,         0 ],
              [-sy/cr,      cy/cr,      0 ]]

  Unified control law:
    q_dot_ref = J^{-1} * (-K*att_error - omega_body)

Yaw Offset:
- The body mesh has xformOp:orient = R_z(90 deg), so visual forward = body +Y.
- The controller works in body +X forward convention internally.
- The env code adds YAW_JOINT_OFFSET (-pi/2) when setting joint targets.
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)

from .gimbal_controller_cfg import GimbalControllerCfg

# Body mesh is rotated 90 deg CCW from physics frame (visual forward = body +Y).
# This offset is added to yaw joint targets in the env code so that controller
# yaw=0 maps to body +X (physics forward). Exported for use by the env.
YAW_JOINT_OFFSET = -math.pi / 2


class GimbalController:
    """Gimbal controller with unified J^{-1} control law.

    Single-path control:
      q_dot_ref = J^{-1} * (omega_cmd - omega_body)
    where omega_cmd = -K_pointing * att_error.

    Produces coherent position and velocity targets from the same computation.
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

        # Per-env pointing gain (tensorized for parallel tuning)
        self._pointing_gain = torch.full(
            (num_envs, 1), cfg.pointing_gain, dtype=torch.float32, device=self.device
        )

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

        Uses the unified control law:
          q_dot_ref = J^{-1} * (omega_cmd - omega_body)
          q_ref     = q + q_dot_ref * dt

        Args:
            gimbal_yaw_rate_cmd: (N,) normalized [-1, 1].
            gimbal_pitch_rate_cmd: (N,) normalized [-1, 1].
            q_body: (N, 4) body quaternion (wxyz), body frame = FLU.
            dt: Physics timestep [s].
            omega_body: (N, 3) body angular velocity in body frame [rad/s] (from gyro).
            joint_positions_actual: (N, 3) actual joint positions [pitch, yaw, roll].

        Returns:
            pos_targets: (yaw, roll, pitch) joint position targets [rad].
            vel_targets: (yaw, roll, pitch) joint velocity targets [rad/s].
        """
        # -- 1. Read actual joint state from simulation --
        actual_pitch = joint_positions_actual[:, 0]
        actual_yaw = joint_positions_actual[:, 1] - YAW_JOINT_OFFSET
        actual_roll = joint_positions_actual[:, 2]

        # -- 2. Integrate world-frame LOS target (persistent setpoint) --
        # The world-frame target must persist across steps so the gimbal has a
        # fixed reference to stabilize against when the drone body tilts.
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

        # -- 3. Compute desired camera quaternion in body frame --
        q_desired_body = self._compute_desired_camera_quat(
            self._azimuth_world, self._elevation_world, q_body
        )

        # -- 4. Current gimbal quaternion from actual joint angles --
        q_current = self._gimbal_joints_to_quat(actual_yaw, actual_roll, actual_pitch)

        # -- 5. Quaternion error -> body-frame angular error --
        #   q_err = q_desired^{-1} * q_current
        #   att_error = 2 * sign(w) * [x, y, z]  (body-frame rotation error)
        q_err = quat_mul(quat_inv(q_desired_body), q_current)
        sign_w = torch.sign(q_err[:, 0]).unsqueeze(-1)
        sign_w = torch.where(sign_w == 0, torch.ones_like(sign_w), sign_w)
        att_error = 2.0 * sign_w * q_err[:, 1:4]  # (N, 3) in body frame

        # -- 6. Pointing rate command: omega_cmd = -K_pointing * att_error --
        omega_cmd = -self._pointing_gain * att_error

        # -- 7. Combined angular velocity through J^{-1} --
        #   q_dot_ref = J^{-1} * (omega_cmd - omega_body)
        omega_combined = omega_cmd - omega_body
        qdot_ref = self._compute_jacobian_inverse_times_omega(
            omega_combined, actual_yaw, actual_roll
        )

        # -- 8. Position targets = actual + joint rates * dt --
        self._yaw = actual_yaw + qdot_ref[:, 0] * dt
        self._roll = actual_roll + qdot_ref[:, 1] * dt
        self._pitch = actual_pitch + qdot_ref[:, 2] * dt

        # -- 9. Clamp to joint limits --
        self._yaw = torch.clamp(self._yaw, self._yaw_limits[0], self._yaw_limits[1])
        self._pitch = torch.clamp(self._pitch, self._pitch_limits[0], self._pitch_limits[1])
        self._roll = torch.clamp(self._roll, self._roll_limits[0], self._roll_limits[1])

        # -- 10. Output coherent position + velocity targets --
        pos_targets = (self._yaw, self._roll, self._pitch)
        vel_targets = (qdot_ref[:, 0], qdot_ref[:, 1], qdot_ref[:, 2])
        return pos_targets, vel_targets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_desired_camera_quat(
        self,
        azimuth_world: torch.Tensor,
        elevation_world: torch.Tensor,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Compute desired camera orientation as a quaternion in body frame.

        The desired camera frame has:
        - Forward (+X): pointing along (azimuth, elevation) in world frame
        - Up (+Z): aligned with world up (horizon stabilization)

        Frame: world (ENU) -> body (FLU) via q_body^{-1}.

        Args:
            azimuth_world: (N,) world-frame azimuth [rad].
            elevation_world: (N,) world-frame elevation [rad].
            q_body: (N, 4) body quaternion (wxyz).

        Returns:
            q_desired_body: (N, 4) desired camera quaternion in body frame (wxyz).
        """
        cos_az = torch.cos(azimuth_world)
        sin_az = torch.sin(azimuth_world)
        cos_el = torch.cos(elevation_world)
        sin_el = torch.sin(elevation_world)

        # Desired forward direction in world frame
        fwd_world = torch.stack([cos_el * cos_az, cos_el * sin_az, sin_el], dim=-1)
        world_up = torch.zeros_like(fwd_world)
        world_up[:, 2] = 1.0

        # Transform both to body frame
        fwd_body = quat_rotate_inverse(q_body, fwd_world)
        up_body = quat_rotate_inverse(q_body, world_up)

        # Build orthonormal camera frame in body coordinates
        cam_up = up_body - (up_body * fwd_body).sum(dim=-1, keepdim=True) * fwd_body
        cam_up = cam_up / (torch.norm(cam_up, dim=-1, keepdim=True) + 1e-8)
        cam_right = torch.cross(cam_up, fwd_body, dim=-1)

        # Build rotation matrix R = [fwd | right | up] as columns
        R00 = fwd_body[:, 0];  R01 = cam_right[:, 0]; R02 = cam_up[:, 0]
        R10 = fwd_body[:, 1];  R11 = cam_right[:, 1]; R12 = cam_up[:, 1]
        R20 = fwd_body[:, 2];  R21 = cam_right[:, 2]; R22 = cam_up[:, 2]

        # Full Shepperd's method: 4-branch quaternion extraction
        trace = R00 + R11 + R22

        # Compute all 4 branches
        # Branch 0: w largest (trace > 0)
        s0 = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 2.0  # 4w
        qw0 = s0 / 4.0
        qx0 = (R21 - R12) / s0
        qy0 = (R02 - R20) / s0
        qz0 = (R10 - R01) / s0

        # Branch 1: x largest (R00 > R11 and R00 > R22)
        s1 = torch.sqrt(torch.clamp(1.0 + R00 - R11 - R22, min=1e-8)) * 2.0  # 4x
        qw1 = (R21 - R12) / s1
        qx1 = s1 / 4.0
        qy1 = (R01 + R10) / s1
        qz1 = (R02 + R20) / s1

        # Branch 2: y largest (R11 > R22)
        s2 = torch.sqrt(torch.clamp(1.0 - R00 + R11 - R22, min=1e-8)) * 2.0  # 4y
        qw2 = (R02 - R20) / s2
        qx2 = (R01 + R10) / s2
        qy2 = s2 / 4.0
        qz2 = (R12 + R21) / s2

        # Branch 3: z largest
        s3 = torch.sqrt(torch.clamp(1.0 - R00 - R11 + R22, min=1e-8)) * 2.0  # 4z
        qw3 = (R10 - R01) / s3
        qx3 = (R02 + R20) / s3
        qy3 = (R12 + R21) / s3
        qz3 = s3 / 4.0

        # Select branch per element
        # Priority: trace > 0 -> branch 0, else largest diagonal
        use_0 = trace > 0
        use_1 = (~use_0) & (R00 > R11) & (R00 > R22)
        use_2 = (~use_0) & (~use_1) & (R11 > R22)
        # use_3 = everything else

        qw = torch.where(use_0, qw0, torch.where(use_1, qw1, torch.where(use_2, qw2, qw3)))
        qx = torch.where(use_0, qx0, torch.where(use_1, qx1, torch.where(use_2, qx2, qx3)))
        qy = torch.where(use_0, qy0, torch.where(use_1, qy1, torch.where(use_2, qy2, qy3)))
        qz = torch.where(use_0, qz0, torch.where(use_1, qz1, torch.where(use_2, qz2, qz3)))

        q_desired = torch.stack([qw, qx, qy, qz], dim=-1)
        q_desired = q_desired / (torch.norm(q_desired, dim=-1, keepdim=True) + 1e-8)

        return q_desired

    def _gimbal_joints_to_quat(
        self,
        yaw: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
    ) -> torch.Tensor:
        """Convert gimbal joint angles to a quaternion.

        Rotation order: Yaw(Z) -> Roll(X) -> Pitch(Y).

        Args:
            yaw: (N,) gimbal yaw [rad].
            roll: (N,) gimbal roll [rad].
            pitch: (N,) gimbal pitch [rad].

        Returns:
            q_gimbal: (N, 4) quaternion (wxyz).
        """
        N = yaw.shape[0]
        zero = torch.zeros(N, device=yaw.device)

        q_yaw = quat_from_euler_xyz(zero, zero, yaw)      # R_z(yaw)
        q_roll = quat_from_euler_xyz(roll, zero, zero)     # R_x(roll)
        q_pitch = quat_from_euler_xyz(zero, pitch, zero)   # R_y(pitch)

        # Compose: R_z(yaw) * R_x(roll) * R_y(pitch)
        q_gimbal = quat_mul(quat_mul(q_yaw, q_roll), q_pitch)
        return q_gimbal

    def _compute_jacobian_inverse_times_omega(
        self,
        omega: torch.Tensor,
        gimbal_yaw: torch.Tensor,
        gimbal_roll: torch.Tensor,
    ) -> torch.Tensor:
        """Compute J^{-1} * omega for the gimbal Jacobian.

        For gimbal chain Yaw(Z)->Roll(X)->Pitch(Y):
          J^{-1} = [[ sy*sr/cr,  -cy*sr/cr,  1 ],
                    [ cy,         sy,         0 ],
                    [-sy/cr,      cy/cr,      0 ]]

        Args:
            omega: (N, 3) angular velocity vector in body frame [rad/s].
            gimbal_yaw: (N,) current gimbal yaw psi [rad].
            gimbal_roll: (N,) current gimbal roll phi [rad].

        Returns:
            qdot: (N, 3) joint rates [yaw_dot, roll_dot, pitch_dot] [rad/s].
        """
        cy = torch.cos(gimbal_yaw)
        sy = torch.sin(gimbal_yaw)
        cr = torch.cos(gimbal_roll)
        sr = torch.sin(gimbal_roll)

        # Avoid division by zero near gimbal lock (roll ~ +/-90 deg)
        inv_cr = 1.0 / (cr + 1e-6 * torch.sign(cr + 1e-8))

        wx = omega[:, 0]
        wy = omega[:, 1]
        wz = omega[:, 2]

        yaw_dot = sy * sr * inv_cr * wx - cy * sr * inv_cr * wy + wz
        roll_dot = cy * wx + sy * wy
        pitch_dot = -sy * inv_cr * wx + cy * inv_cr * wy

        return torch.stack([yaw_dot, roll_dot, pitch_dot], dim=-1)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _world_to_body_angles(
        self,
        azimuth_world: torch.Tensor,
        elevation_world: torch.Tensor,
        q_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world-frame pointing to body-frame yaw/pitch (roll=0 assumption).

        Frame: world (ENU) -> body (FLU) via q_body inverse rotation.
        Yaw: atan2(dir_body_y, dir_body_x).
        Pitch: -atan2(dir_body_z, xy_dist).
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
