# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity controller that converts velocity commands to attitude and thrust.

Frame Convention:
- World: ENU (East-North-Up)
- Body: FLU (Forward-Left-Up)
- Quaternion: wxyz (scalar first)

The controller computes desired acceleration, then converts to attitude by
aligning the body z-axis with the desired acceleration direction.
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import quat_from_euler_xyz, normalize

from .velocity_controller_cfg import VelocityControllerCfg


class VelocityController:
    """PI velocity controller with velocity-to-attitude conversion.

    Converts velocity commands to desired attitude and thrust commands
    for the inner attitude control loop.

    Control law:
        a_des = Kp * (v_des - v) + Ki * integral + g * z_hat
        thrust = m * |a_des|
        attitude = rotation aligning body z with a_des
    """

    def __init__(
        self,
        cfg: VelocityControllerCfg,
        mass: float,
        gravity: float,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize velocity controller.

        Args:
            cfg: Velocity controller configuration.
            mass: Drone mass [kg].
            gravity: Gravity magnitude [m/s^2].
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.mass = mass
        self.gravity = gravity
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Convert gains to tensors
        self._Kp_vel = torch.tensor(cfg.Kp_vel, dtype=torch.float32, device=self.device)
        self._Ki_vel = torch.tensor(cfg.Ki_vel, dtype=torch.float32, device=self.device)
        self._integral_limit = torch.tensor(cfg.integral_limit, dtype=torch.float32, device=self.device)

        # Convert max tilt to radians
        self._max_tilt_rad = math.radians(cfg.max_tilt)

        # Integral state
        self._vel_integral = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)

        # Gravity vector in world frame (ENU: +Z is up)
        self._g_vec = torch.tensor([0.0, 0.0, gravity], dtype=torch.float32, device=self.device)

    def compute_control(
        self,
        v_des: torch.Tensor,
        yaw_rate_des: torch.Tensor,
        v_current: torch.Tensor,
        q_current: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute velocity control output.

        Args:
            v_des: (N, 3) desired velocity in world frame [m/s].
            yaw_rate_des: (N,) desired yaw rate [rad/s].
            v_current: (N, 3) current velocity in world frame [m/s].
            q_current: (N, 4) current quaternion (wxyz).
            dt: Timestep [s].

        Returns:
            q_des: (N, 4) desired attitude quaternion (wxyz).
            thrust_cmd: (N,) total thrust command [N].
            yaw_rate_passthrough: (N,) yaw rate command [rad/s].
        """
        # Velocity error
        vel_error = v_des - v_current

        # PI control for desired acceleration
        # Proportional term
        a_des_p = self._Kp_vel * vel_error

        # Integral term with anti-windup
        self._vel_integral = self._vel_integral + vel_error * dt
        self._vel_integral = torch.clamp(
            self._vel_integral,
            -self._integral_limit,
            self._integral_limit,
        )
        a_des_i = self._Ki_vel * self._vel_integral

        # Total desired acceleration (gravity feedforward)
        a_des = a_des_p + a_des_i + self._g_vec

        # Compute desired attitude by aligning body z with a_des
        # This also clamps the attitude to max_tilt
        q_des, tilt_angle = self._compute_attitude_from_accel(a_des, q_current)

        # Compute thrust magnitude accounting for attitude clamping
        # If attitude is clamped, we need to reduce thrust to avoid excess vertical component
        # thrust * cos(tilt) = mass * g  ->  thrust = mass * g / cos(tilt)
        # But we also want to track desired vertical accel, so:
        # thrust = mass * a_des_z / cos(tilt), clamped to reasonable range
        cos_tilt = torch.cos(tilt_angle)
        cos_tilt = torch.clamp(cos_tilt, min=0.5)  # Avoid division by small values

        # Vertical acceleration component (gravity + desired vertical motion)
        a_vertical = a_des[:, 2]

        # Thrust to achieve vertical acceleration given current tilt
        thrust_cmd = self.mass * a_vertical / cos_tilt

        # Clamp thrust to reasonable range (0.5 to 2x hover thrust)
        hover_thrust = self.mass * self.gravity
        thrust_cmd = torch.clamp(thrust_cmd, min=0.5 * hover_thrust, max=2.0 * hover_thrust)

        return q_des, thrust_cmd, yaw_rate_des

    def _compute_attitude_from_accel(
        self,
        a_des: torch.Tensor,
        q_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute desired attitude quaternion from desired acceleration.

        The desired attitude aligns the body z-axis with the desired
        acceleration direction, with yaw maintained from current orientation.

        Args:
            a_des: (N, 3) desired acceleration in world frame [m/s^2].
            q_current: (N, 4) current quaternion (wxyz).

        Returns:
            q_des: (N, 4) desired attitude quaternion (wxyz).
            tilt_angle: (N,) total tilt angle from vertical [rad].
        """
        # Normalize desired acceleration to get body z direction (world frame)
        a_mag = torch.norm(a_des, dim=-1, keepdim=True)
        z_des_world = a_des / (a_mag + 1e-6)  # Avoid division by zero

        # Extract current yaw from quaternion
        yaw_current = self._extract_yaw(q_current)

        # CRITICAL FIX: Rotate z_des into yaw-aligned frame before computing roll/pitch
        # This ensures roll/pitch are computed relative to the drone's heading
        cos_yaw = torch.cos(yaw_current)
        sin_yaw = torch.sin(yaw_current)

        # Rotate z_des by -yaw around Z axis to get yaw-aligned frame
        # R(-yaw) = [[cos(yaw), sin(yaw), 0], [-sin(yaw), cos(yaw), 0], [0, 0, 1]]
        z_des_yaw_aligned = torch.stack([
            cos_yaw * z_des_world[:, 0] + sin_yaw * z_des_world[:, 1],
            -sin_yaw * z_des_world[:, 0] + cos_yaw * z_des_world[:, 1],
            z_des_world[:, 2],
        ], dim=-1)

        # Now compute roll and pitch in yaw-aligned frame
        # In FLU with yaw=0:
        # z_body_x = cos(roll)*sin(pitch)
        # z_body_y = -sin(roll)
        # z_body_z = cos(roll)*cos(pitch)

        # Roll from z_des_y (clamped to avoid asin domain issues)
        roll_des = -torch.asin(torch.clamp(z_des_yaw_aligned[:, 1], -1.0, 1.0))

        # Pitch from z_des_x and z_des_z
        pitch_des = torch.atan2(z_des_yaw_aligned[:, 0], z_des_yaw_aligned[:, 2])

        # Apply tilt limits
        roll_des = torch.clamp(roll_des, -self._max_tilt_rad, self._max_tilt_rad)
        pitch_des = torch.clamp(pitch_des, -self._max_tilt_rad, self._max_tilt_rad)

        # Compute total tilt angle (angle from vertical)
        # tilt = acos(cos(roll) * cos(pitch))
        tilt_angle = torch.acos(torch.clamp(
            torch.cos(roll_des) * torch.cos(pitch_des), -1.0, 1.0
        ))

        # Construct desired quaternion with current yaw
        q_des = quat_from_euler_xyz(roll_des, pitch_des, yaw_current)

        return q_des, tilt_angle

    def _extract_yaw(self, q: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle from quaternion (wxyz format).

        Args:
            q: (N, 4) quaternion (wxyz).

        Returns:
            yaw: (N,) yaw angle [rad].
        """
        # q = [w, x, y, z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Yaw from quaternion (ZYX Euler sequence)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return yaw

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset controller state.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._vel_integral.zero_()
        else:
            self._vel_integral[env_ids] = 0.0

    def set_gains(
        self,
        Kp_vel: torch.Tensor | tuple | None = None,
        Ki_vel: torch.Tensor | tuple | None = None,
    ):
        """Set control gains (for randomization).

        Args:
            Kp_vel: Proportional gains [x, y, z].
            Ki_vel: Integral gains [x, y, z].
        """
        if Kp_vel is not None:
            if isinstance(Kp_vel, tuple):
                Kp_vel = torch.tensor(Kp_vel, dtype=torch.float32, device=self.device)
            self._Kp_vel = Kp_vel.to(self.device)

        if Ki_vel is not None:
            if isinstance(Ki_vel, tuple):
                Ki_vel = torch.tensor(Ki_vel, dtype=torch.float32, device=self.device)
            self._Ki_vel = Ki_vel.to(self.device)

    def disable_integral_for_saturation(
        self,
        is_saturated: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        """Disable integral accumulation for saturated environments.

        Anti-windup: prevents integral from growing when actuators are saturated.

        Args:
            is_saturated: (N,) boolean mask of saturated environments.
            env_ids: Environment indices to consider. If None, consider all.
        """
        if env_ids is None:
            # Reset integral for saturated environments
            self._vel_integral[is_saturated] = 0.0
        else:
            mask = is_saturated[env_ids]
            self._vel_integral[env_ids[mask]] = 0.0
