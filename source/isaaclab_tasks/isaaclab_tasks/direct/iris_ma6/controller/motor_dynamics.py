# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""First-order motor dynamics for quadcopter rotors."""

from __future__ import annotations

import torch

from .mixer import MixerMatrix
from .motor_dynamics_cfg import MotorDynamicsCfg


class MotorDynamics:
    """First-order motor dynamics with thrust and torque computation.

    Models motor speed response with first-order lag:
        d(omega)/dt = (omega_cmd - omega) / tau_motor

    Each rotor generates:
        Thrust: T_i = k_f * omega_i^2
        Reaction torque: Q_i = k_m * omega_i^2
    """

    def __init__(
        self,
        cfg: MotorDynamicsCfg,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize motor dynamics.

        Args:
            cfg: Motor dynamics configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Effective values: dispatched on cfg.model ("default" | "pegasus", ticket 040).
        k_f_eff = cfg.get_effective_k_f()
        k_m_eff = cfg.get_effective_k_m()
        omega_max_eff = cfg.get_effective_omega_max()
        tau_motor_eff = cfg.get_effective_tau_motor()

        # State: current rotor speeds (N, 4)
        self._omega = torch.full(
            (num_envs, 4),
            cfg.omega_min,  # Start at idle
            dtype=torch.float32,
            device=self.device,
        )

        # Create mixer for thrust/moment aggregation
        self._mixer = MixerMatrix(
            arm_length=cfg.arm_length,
            k_f=k_f_eff,
            k_m=k_m_eff,
            device=self.device,
        )

        # Precompute limits
        self._omega_min = cfg.omega_min
        self._omega_max = omega_max_eff
        self._tau_motor = torch.full(
            (num_envs,), tau_motor_eff, dtype=torch.float32, device=self.device
        )

        # Cache effective k_f for thrust_min/thrust_max in allocate_wrench()
        # (cfg.k_f stays bit-exact in default mode; Pegasus mode reads cfg.k_f_pegasus).
        self._k_f_eff = k_f_eff
        self._k_m_eff = k_m_eff

        # Ticket 040 — per-env multiplicative DR scales on rotor-thrust /
        # rotor-drag coefficients. Initialised to 1.0 (no DR). Written by
        # ``DroneController.randomize_gains`` only when the motor cfg's
        # ``model == "pegasus"`` AND ``GainRandomizationCfg.randomize_physics_pegasus``
        # is True. Applied at force/moment compute time so default-mode runs
        # observe bit-exact pre-040 wrenches.
        self._k_f_scale = torch.ones((num_envs,), dtype=torch.float32, device=self.device)
        self._k_m_scale = torch.ones((num_envs,), dtype=torch.float32, device=self.device)

    @property
    def omega(self) -> torch.Tensor:
        """Current rotor angular velocities (N, 4) [rad/s]."""
        return self._omega

    @property
    def mixer(self) -> MixerMatrix:
        """Get the mixer matrix instance."""
        return self._mixer

    def step(self, omega_cmd: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply first-order lag dynamics to update rotor speeds.

        Discrete-time: omega_new = omega + alpha * (omega_cmd - omega)
        where alpha = 1 - exp(-dt / tau_motor) for exact first-order response

        Args:
            omega_cmd: (N, 4) commanded rotor speeds [rad/s].
            dt: Simulation timestep [s].

        Returns:
            omega: (N, 4) updated rotor speeds [rad/s].
        """
        # Clamp command to valid range
        omega_cmd_clamped = torch.clamp(omega_cmd, min=self._omega_min, max=self._omega_max)

        # First-order lag: exact discretization alpha = 1 - exp(-dt/tau)
        # This gives exactly 63.2% response at t=tau
        if isinstance(self._tau_motor, torch.Tensor) and self._tau_motor.dim() >= 1:
            # Per-env tau: (N,) -> (N, 1) for broadcasting with (N, 4)
            alpha = 1.0 - torch.exp(-dt / self._tau_motor.unsqueeze(-1))
        else:
            alpha = 1.0 - torch.exp(torch.tensor(-dt / self._tau_motor, device=self.device))

        # Update rotor speeds
        self._omega = self._omega + alpha * (omega_cmd_clamped - self._omega)

        # Ensure output is within limits (should be, but safety)
        self._omega = torch.clamp(self._omega, min=self._omega_min, max=self._omega_max)

        return self._omega

    def compute_thrust_torque(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute thrust and reaction torque for each rotor.

        Uses: T = k_f * omega^2, Q = k_m * omega^2

        Returns:
            thrust: (N, 4) thrust per rotor [N].
            torque: (N, 4) reaction torque per rotor [Nm].
        """
        thrust = self._mixer.omega_to_thrust(self._omega)
        torque = self._mixer.omega_to_torque(self._omega)
        return thrust, torque

    def compute_body_wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total body force and moment from all rotors.

        Returns:
            force: (N, 3) body force [0, 0, F_z] [N].
            moment: (N, 3) body moment [tau_x, tau_y, tau_z] [Nm].
        """
        thrust, _ = self.compute_thrust_torque()

        # Use mixer to aggregate
        total_thrust, moments = self._mixer.aggregate(thrust)

        # Force is in +Z direction (body frame)
        force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force[:, 2] = total_thrust

        # Ticket 040 — apply per-env multiplicative DR scales on k_f / k_m.
        # F and τ_x / τ_y are linear in rotor thrust, so they scale by s_f.
        # τ_z is the rotor reaction-torque sum, which scales by s_m
        # independently of s_f (Q_i = k_m·ω_i² · sign, not derived from T_i).
        # In default mode (and when DR is off) both scales stay at 1.0, so
        # this is a bit-exact no-op.
        s_f = self._k_f_scale.unsqueeze(-1)  # (N, 1)
        force = force * s_f
        moments_scaled = torch.empty_like(moments)
        moments_scaled[:, 0] = moments[:, 0] * self._k_f_scale  # τ_x ∝ s_f
        moments_scaled[:, 1] = moments[:, 1] * self._k_f_scale  # τ_y ∝ s_f
        moments_scaled[:, 2] = moments[:, 2] * self._k_m_scale  # τ_z ∝ s_m

        return force, moments_scaled

    def allocate_wrench(
        self,
        thrust_cmd: torch.Tensor,
        moment_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """Allocate desired thrust and moment to rotor speed commands.

        Args:
            thrust_cmd: (N,) total thrust command [N].
            moment_cmd: (N, 3) moment command [tau_x, tau_y, tau_z] [Nm].

        Returns:
            omega_cmd: (N, 4) rotor speed commands [rad/s].
        """
        # Compute required thrusts (use effective k_f — Pegasus mode swaps it).
        thrust_min = self._k_f_eff * self._omega_min**2
        thrust_max = self._k_f_eff * self._omega_max**2

        thrusts = self._mixer.allocate(
            thrust_cmd=thrust_cmd,
            moment_cmd=moment_cmd,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
        )

        # Convert to angular velocities
        omega_cmd = self._mixer.thrust_to_omega(thrusts)

        # Clamp to limits
        omega_cmd = torch.clamp(omega_cmd, min=self._omega_min, max=self._omega_max)

        return omega_cmd

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset motor states to idle.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._omega.fill_(self._omega_min)
        else:
            self._omega[env_ids] = self._omega_min

    def reset_to_hover(
        self, mass: float, gravity: float, env_ids: torch.Tensor | None = None
    ):
        """Reset motor speeds to hover equilibrium.

        Initializes omega = sqrt(mass * g / (4 * k_f)) so the drone produces
        exactly hover thrust on the first sim step, avoiding a gravity dip.

        Args:
            mass: Drone mass [kg].
            gravity: Gravity magnitude [m/s^2].
            env_ids: Environment indices to reset. If None, reset all.
        """
        import math

        omega_hover = math.sqrt(mass * gravity / (4.0 * self._k_f_eff))
        omega_hover = max(self._omega_min, min(omega_hover, self._omega_max))
        if env_ids is None:
            self._omega.fill_(omega_hover)
        else:
            self._omega[env_ids] = omega_hover

    def set_tau_motor(self, tau_motor: torch.Tensor | float):
        """Set motor time constant (for randomization).

        Args:
            tau_motor: Time constant [s]. Can be scalar or (N,) tensor for per-env values.
        """
        if isinstance(tau_motor, torch.Tensor):
            self._tau_motor = tau_motor.to(self.device)
        else:
            self._tau_motor = tau_motor
