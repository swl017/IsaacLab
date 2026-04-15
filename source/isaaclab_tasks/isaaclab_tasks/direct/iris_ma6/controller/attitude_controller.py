# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quaternion-based P attitude controller for quadcopter.

This is the middle control loop that converts attitude errors to angular rate
setpoints, following the PX4 mc_att_control architecture.

Frame Convention:
- Body: FLU (Forward-Left-Up)
- Quaternion: wxyz (scalar first)

Control Law (P-only, matches PX4 mc_att_control):
    rate_setpoint = -Kp * attitude_error + R^T @ [0, 0, yaw_rate_world] * yaw_rate_des

`yaw_rate_des` is a WORLD-frame angular rate about the world z-axis (psi_dot).
It is projected into the body frame via the world z-axis expressed in the body
frame before being added to the body-rate setpoint.

The rate setpoint is then fed to the inner rate controller (RateController)
which computes actual torque commands via PID.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_mul, quat_inv, quat_rotate_inverse

from .attitude_controller_cfg import AttitudeControllerCfg


class AttitudeController:
    """Quaternion-based P attitude controller (outputs rate setpoint).

    Computes desired angular rate from attitude error using quaternion
    representation to avoid gimbal lock. The output rate setpoint is
    fed to the inner rate controller.

    Features:
    - Quaternion-based error avoids gimbal lock
    - Shortest path rotation via sign flip
    - Yaw rate feedforward
    - Rate limiting
    - Yaw deprioritization (reduced yaw tracking when roll/pitch error is large)
    """

    def __init__(
        self,
        cfg: AttitudeControllerCfg,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize attitude controller.

        Args:
            cfg: Attitude controller configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Convert gains to tensors
        self._Kp_att = torch.tensor(cfg.Kp_att, dtype=torch.float32, device=self.device)
        self._rate_limit = torch.tensor(cfg.rate_limit, dtype=torch.float32, device=self.device)
        self._yaw_weight = cfg.yaw_weight

    def compute_attitude_error(
        self,
        q_des: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attitude error from quaternion difference.

        Uses: q_err = q_des^(-1) * q_current
              attitude_error = 2 * sign(q_err.w) * q_err.xyz

        This formulation:
        - Avoids gimbal lock
        - Ensures shortest path rotation
        - Gives a 3D error vector [roll_err, pitch_err, yaw_err]

        Args:
            q_des: (N, 4) desired quaternion (wxyz).
            q_current: (N, 4) current quaternion (wxyz).

        Returns:
            attitude_error: (N, 3) attitude error [roll, pitch, yaw] [rad].
        """
        # Compute quaternion error: q_err = q_des^-1 * q_current
        q_des_inv = quat_inv(q_des)
        q_err = quat_mul(q_des_inv, q_current)

        # Ensure shortest path (flip sign if w < 0)
        sign_w = torch.sign(q_err[:, 0]).unsqueeze(-1)
        sign_w = torch.where(sign_w == 0, torch.ones_like(sign_w), sign_w)

        # Extract attitude error: 2 * sign(w) * [x, y, z]
        attitude_error = 2.0 * sign_w * q_err[:, 1:4]

        return attitude_error

    def compute_control(
        self,
        q_des: torch.Tensor,
        yaw_rate_des: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attitude control output (rate setpoint).

        Args:
            q_des: (N, 4) desired attitude quaternion (wxyz).
            yaw_rate_des: (N,) desired yaw rate [rad/s]. WORLD-frame psi_dot
                (rotation about world z-axis), matching PX4's yawspeed_setpoint.
            q_current: (N, 4) current attitude quaternion (wxyz).

        Returns:
            rate_setpoint: (N, 3) desired angular rate [roll, pitch, yaw] [rad/s].
        """
        # Compute attitude error
        attitude_error = self.compute_attitude_error(q_des, q_current)

        # P control law: rate_sp = -Kp * att_err  (body-frame rate setpoint)
        rate_setpoint = -self._Kp_att * attitude_error

        # Yaw rate feedforward (PX4 convention: yaw_rate_des is WORLD-frame psi_dot,
        # i.e. rotation about the world z-axis). Project the world z-axis into the
        # body frame and scale by yaw_rate_des, then add as a 3D body-rate vector.
        # Matches PX4 AttitudeControl.cpp:
        #     rate_setpoint += q.inversed().dcm_z() * _yawspeed_setpoint
        # (q.inversed().dcm_z() == world z-axis expressed in the body frame).
        world_z = torch.zeros_like(rate_setpoint)
        world_z[:, 2] = 1.0
        world_z_in_body = quat_rotate_inverse(q_current, world_z)  # (N, 3)
        rate_setpoint = rate_setpoint + world_z_in_body * yaw_rate_des.unsqueeze(-1)

        # Apply yaw weight (deprioritize yaw when roll/pitch error is large).
        # At level flight world_z_in_body ≈ [0, 0, 1], so this is identical to the
        # previous behavior; under tilt it scales the body-z component of the
        # combined P + world-yaw feedforward, which is the direct analogue.
        roll_pitch_error_mag = torch.norm(attitude_error[:, :2], dim=-1)
        yaw_scale = torch.clamp(1.0 - roll_pitch_error_mag / 0.5, min=self._yaw_weight, max=1.0)
        rate_setpoint[:, 2] = rate_setpoint[:, 2] * yaw_scale

        # Clamp rate setpoint to limits
        rate_setpoint = torch.clamp(rate_setpoint, -self._rate_limit, self._rate_limit)

        return rate_setpoint

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset controller state.

        Currently stateless, but placeholder for future extensions.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        pass  # Attitude controller is stateless (P-only)

    def set_gains(
        self,
        Kp_att: torch.Tensor | tuple | None = None,
    ):
        """Set control gains (for randomization).

        Args:
            Kp_att: Proportional gains [roll, pitch, yaw].
        """
        if Kp_att is not None:
            if isinstance(Kp_att, tuple):
                Kp_att = torch.tensor(Kp_att, dtype=torch.float32, device=self.device)
            self._Kp_att = Kp_att.to(self.device)
