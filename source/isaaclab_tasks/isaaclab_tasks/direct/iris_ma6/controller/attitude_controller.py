# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quaternion-based PD attitude controller for quadcopter.

Frame Convention:
- Body: FLU (Forward-Left-Up)
- Quaternion: wxyz (scalar first)
- +tau_x: Roll right (left wing up)
- +tau_y: Pitch down (nose down)
- +tau_z: Yaw left (CCW from above)
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_mul, quat_inv

from .attitude_controller_cfg import AttitudeControllerCfg
from .mixer import MixerMatrix


class AttitudeController:
    """Quaternion-based PD attitude controller.

    Computes body moments from attitude error using quaternion representation
    to avoid gimbal lock. The controller tracks a desired attitude quaternion
    for roll/pitch and a desired yaw rate.

    Control law:
        attitude_error = 2 * sign(q_err.w) * q_err.xyz
        tau_cmd = -Kp * attitude_error - Kd * (omega - omega_des)
    """

    def __init__(
        self,
        cfg: AttitudeControllerCfg,
        mixer: MixerMatrix,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize attitude controller.

        Args:
            cfg: Attitude controller configuration.
            mixer: Mixer matrix for thrust allocation.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._mixer = mixer

        # Convert gains to tensors
        self._Kp_att = torch.tensor(cfg.Kp_att, dtype=torch.float32, device=self.device)
        self._Kd_att = torch.tensor(cfg.Kd_att, dtype=torch.float32, device=self.device)
        self._tau_max = torch.tensor(cfg.tau_max, dtype=torch.float32, device=self.device)

        # Thrust limits for mixer allocation (prevents yaw commands from causing climb)
        self._thrust_min = cfg.thrust_min
        self._thrust_max = cfg.thrust_max

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
        # sign(q_err.w) ensures we rotate the short way
        sign_w = torch.sign(q_err[:, 0]).unsqueeze(-1)
        sign_w = torch.where(sign_w == 0, torch.ones_like(sign_w), sign_w)  # Handle w=0

        # Extract attitude error: 2 * sign(w) * [x, y, z]
        attitude_error = 2.0 * sign_w * q_err[:, 1:4]

        return attitude_error

    def compute_control(
        self,
        q_des: torch.Tensor,
        yaw_rate_des: torch.Tensor,
        q_current: torch.Tensor,
        omega_current: torch.Tensor,
        thrust_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attitude control output.

        Args:
            q_des: (N, 4) desired attitude quaternion (wxyz).
            yaw_rate_des: (N,) desired yaw rate [rad/s].
            q_current: (N, 4) current attitude quaternion (wxyz).
            omega_current: (N, 3) current body angular velocity [rad/s].
            thrust_cmd: (N,) total thrust command from velocity controller [N].

        Returns:
            omega_cmd: (N, 4) rotor speed commands [rad/s].
            tau_cmd: (N, 3) moment commands [tau_x, tau_y, tau_z] [Nm] (for debugging).
        """
        # Compute attitude error
        attitude_error = self.compute_attitude_error(q_des, q_current)

        # Desired angular velocity: [0, 0, yaw_rate_des]
        omega_des = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        omega_des[:, 2] = yaw_rate_des

        # Angular velocity error
        omega_error = omega_current - omega_des

        # PD control law: tau = -Kp * att_err - Kd * omega_err
        tau_cmd = -self._Kp_att * attitude_error - self._Kd_att * omega_error

        # Saturate moments
        tau_cmd = torch.clamp(tau_cmd, -self._tau_max, self._tau_max)

        # Allocate to rotor thrusts via mixer with proper limits
        # Using thrust_min > 0 prevents yaw commands from causing thrust imbalance
        thrusts = self._mixer.allocate(
            thrust_cmd, tau_cmd, thrust_min=self._thrust_min, thrust_max=self._thrust_max
        )

        # Convert thrusts to rotor speeds
        omega_cmd = self._mixer.thrust_to_omega(thrusts)

        return omega_cmd, tau_cmd

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset controller state.

        Currently stateless, but placeholder for future integral terms.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        pass  # Attitude controller is stateless (no integral term)

    def set_gains(
        self,
        Kp_att: torch.Tensor | tuple | None = None,
        Kd_att: torch.Tensor | tuple | None = None,
    ):
        """Set control gains (for randomization).

        Args:
            Kp_att: Proportional gains [roll, pitch, yaw].
            Kd_att: Derivative gains [roll, pitch, yaw].
        """
        if Kp_att is not None:
            if isinstance(Kp_att, tuple):
                Kp_att = torch.tensor(Kp_att, dtype=torch.float32, device=self.device)
            self._Kp_att = Kp_att.to(self.device)

        if Kd_att is not None:
            if isinstance(Kd_att, tuple):
                Kd_att = torch.tensor(Kd_att, dtype=torch.float32, device=self.device)
            self._Kd_att = Kd_att.to(self.device)
