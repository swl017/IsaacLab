# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PID angular rate controller for quadcopter.

This is the innermost control loop, converting angular rate setpoints to
body torque commands. It follows the PX4 mc_rate_control architecture.

Frame Convention:
- Body: FLU (Forward-Left-Up)
- +tau_x: Roll right (right wing down)
- +tau_y: Pitch down (nose down)
- +tau_z: Yaw left (CCW from above)

Control Law:
    tau_cmd = Kp * rate_error + Ki * integral - Kd * angular_accel

Note: D-term uses angular acceleration (measured), not rate error derivative.
"""

from __future__ import annotations

import torch

from .mixer import MixerMatrix
from .rate_controller_cfg import RateControllerCfg


class RateController:
    """PID angular rate controller with anti-windup.

    Converts angular rate setpoints to body torque commands, then allocates
    to individual rotor thrusts via the mixer matrix.

    Features:
    - PID control with angular acceleration feedback (not error derivative)
    - Saturation-aware anti-windup (inhibits integral when saturated)
    - Non-linear I-gain reduction for large errors
    """

    def __init__(
        self,
        cfg: RateControllerCfg,
        mixer: MixerMatrix,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize rate controller.

        Args:
            cfg: Rate controller configuration.
            mixer: Mixer matrix for thrust allocation.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._mixer = mixer

        # Convert gains to tensors
        self._Kp_rate = torch.tensor(cfg.Kp_rate, dtype=torch.float32, device=self.device)
        self._Ki_rate = torch.tensor(cfg.Ki_rate, dtype=torch.float32, device=self.device)
        self._Kd_rate = torch.tensor(cfg.Kd_rate, dtype=torch.float32, device=self.device)
        self._tau_max = torch.tensor(cfg.tau_max, dtype=torch.float32, device=self.device)
        self._integral_limit = torch.tensor(cfg.integral_limit, dtype=torch.float32, device=self.device)
        self._rate_limit = torch.tensor(cfg.rate_limit, dtype=torch.float32, device=self.device)

        # Scalar parameters
        self._antiwindup_limit = cfg.antiwindup_i_factor_limit
        self._thrust_min = cfg.thrust_min
        self._thrust_max = cfg.thrust_max

        # State: integral term
        self._rate_integral = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)

        # State: previous omega for angular acceleration computation
        self._prev_omega = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)

        # State: saturation flags from previous iteration
        self._saturation_positive = torch.zeros((num_envs, 3), dtype=torch.bool, device=self.device)
        self._saturation_negative = torch.zeros((num_envs, 3), dtype=torch.bool, device=self.device)

    def compute_control(
        self,
        rate_setpoint: torch.Tensor,
        omega_current: torch.Tensor,
        thrust_cmd: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rate control output.

        Args:
            rate_setpoint: (N, 3) desired angular rate [roll, pitch, yaw] [rad/s].
            omega_current: (N, 3) current body angular velocity [rad/s].
            thrust_cmd: (N,) total thrust command from velocity controller [N].
            dt: Control timestep [s].

        Returns:
            omega_cmd: (N, 4) rotor speed commands [rad/s].
            tau_cmd: (N, 3) moment commands [tau_x, tau_y, tau_z] [Nm].
            is_saturated: (N, 3) boolean mask indicating saturation per axis.
        """
        # Clamp rate setpoint to limits
        rate_sp = torch.clamp(rate_setpoint, -self._rate_limit, self._rate_limit)

        # Rate error
        rate_error = rate_sp - omega_current

        # Angular acceleration (backward difference)
        angular_accel = (omega_current - self._prev_omega) / dt
        self._prev_omega = omega_current.clone()

        # Update integral with anti-windup
        self._update_integral(rate_error, dt)

        # PID control law: tau = Kp * error + Ki * integral - Kd * accel
        tau_cmd = (
            self._Kp_rate * rate_error
            + self._Ki_rate * self._rate_integral
            - self._Kd_rate * angular_accel
        )

        # Saturate moments and track saturation
        is_saturated, tau_cmd = self._saturate_moments(tau_cmd)

        # Update saturation flags for next iteration's anti-windup
        self._saturation_positive = tau_cmd >= self._tau_max
        self._saturation_negative = tau_cmd <= -self._tau_max

        # Allocate to rotor thrusts via mixer
        thrusts = self._mixer.allocate(
            thrust_cmd, tau_cmd, thrust_min=self._thrust_min, thrust_max=self._thrust_max
        )

        # Convert thrusts to rotor speeds
        omega_cmd = self._mixer.thrust_to_omega(thrusts)

        return omega_cmd, tau_cmd, is_saturated

    def _update_integral(self, rate_error: torch.Tensor, dt: float):
        """Update integral term with anti-windup.

        Implements PX4-style anti-windup:
        1. Directional inhibit based on saturation feedback
        2. Non-linear I-gain reduction for large errors
        3. Hard integral clamping

        Args:
            rate_error: (N, 3) rate error [rad/s].
            dt: Control timestep [s].
        """
        # Create a copy of rate error for modification
        error_for_integral = rate_error.clone()

        # Directional anti-windup: prevent further winding when saturated
        # If saturated positive and error is positive, don't integrate
        error_for_integral = torch.where(
            self._saturation_positive & (rate_error > 0),
            torch.zeros_like(error_for_integral),
            error_for_integral,
        )
        # If saturated negative and error is negative, don't integrate
        error_for_integral = torch.where(
            self._saturation_negative & (rate_error < 0),
            torch.zeros_like(error_for_integral),
            error_for_integral,
        )

        # Non-linear I-gain reduction for large errors
        # i_factor = max(0, 1 - (error / limit)^2)
        error_ratio = rate_error / self._antiwindup_limit
        i_factor = torch.clamp(1.0 - error_ratio * error_ratio, min=0.0)

        # Integrate with variable gain
        delta_integral = i_factor * error_for_integral * dt
        new_integral = self._rate_integral + delta_integral

        # Clamp integral to limits
        self._rate_integral = torch.clamp(new_integral, -self._integral_limit, self._integral_limit)

    def _saturate_moments(
        self, tau_cmd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Saturate moment commands and return saturation flags.

        Args:
            tau_cmd: (N, 3) moment commands [Nm].

        Returns:
            is_saturated: (N, 3) boolean mask indicating saturation per axis.
            tau_clamped: (N, 3) clamped moment commands [Nm].
        """
        # Clamp moments
        tau_clamped = torch.clamp(tau_cmd, -self._tau_max, self._tau_max)

        # Detect saturation (any axis at limit)
        is_saturated = (
            (tau_cmd >= self._tau_max) | (tau_cmd <= -self._tau_max)
        )

        return is_saturated, tau_clamped

    def set_saturation_status(
        self,
        saturation_positive: torch.Tensor,
        saturation_negative: torch.Tensor,
    ):
        """Set saturation status from external source (e.g., mixer).

        This allows the mixer to communicate saturation back to the rate
        controller for improved anti-windup.

        Args:
            saturation_positive: (N, 3) positive saturation per axis.
            saturation_negative: (N, 3) negative saturation per axis.
        """
        self._saturation_positive = saturation_positive
        self._saturation_negative = saturation_negative

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset controller state.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._rate_integral.zero_()
            self._prev_omega.zero_()
            self._saturation_positive.zero_()
            self._saturation_negative.zero_()
        else:
            self._rate_integral[env_ids] = 0.0
            self._prev_omega[env_ids] = 0.0
            self._saturation_positive[env_ids] = False
            self._saturation_negative[env_ids] = False

    def set_gains(
        self,
        Kp_rate: torch.Tensor | tuple | None = None,
        Ki_rate: torch.Tensor | tuple | None = None,
        Kd_rate: torch.Tensor | tuple | None = None,
    ):
        """Set control gains (for randomization).

        Args:
            Kp_rate: Proportional gains [roll, pitch, yaw].
            Ki_rate: Integral gains [roll, pitch, yaw].
            Kd_rate: Derivative gains [roll, pitch, yaw].
        """
        if Kp_rate is not None:
            if isinstance(Kp_rate, tuple):
                Kp_rate = torch.tensor(Kp_rate, dtype=torch.float32, device=self.device)
            self._Kp_rate = Kp_rate.to(self.device)

        if Ki_rate is not None:
            if isinstance(Ki_rate, tuple):
                Ki_rate = torch.tensor(Ki_rate, dtype=torch.float32, device=self.device)
            self._Ki_rate = Ki_rate.to(self.device)

        if Kd_rate is not None:
            if isinstance(Kd_rate, tuple):
                Kd_rate = torch.tensor(Kd_rate, dtype=torch.float32, device=self.device)
            self._Kd_rate = Kd_rate.to(self.device)
