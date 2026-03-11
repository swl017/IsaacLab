# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aerodynamic effects with configurable fidelity levels.

Fidelity Levels:
- Level 0: Disabled (no aerodynamic effects)
- Level 1: Basic drag (quadratic drag on body)
- Level 2: Level 1 + Wind (constant + Dryden gust model)
- Level 3: Level 2 + Rotor effects (H-force, blade flapping)
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import quat_rotate_inverse

from .aerodynamics_cfg import AerodynamicsCfg


class AerodynamicEffects:
    """Configurable aerodynamic effects for quadcopter simulation.

    Computes aerodynamic forces and moments based on the configured
    fidelity level.
    """

    def __init__(
        self,
        cfg: AerodynamicsCfg,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize aerodynamic effects.

        Args:
            cfg: Aerodynamics configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._fidelity_level = cfg.fidelity_level

        # Precompute drag factor: 0.5 * rho * Cd * A
        self._drag_factor = 0.5 * cfg.rho * cfg.C_d * cfg.A

        # Wind state (for Dryden gust model)
        self._v_wind = torch.tensor(
            cfg.v_wind_mean, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(num_envs, 3).clone()

        # Gust state (filtered white noise)
        self._gust = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)

        # Convert wind mean to tensor
        self._v_wind_mean = torch.tensor(
            cfg.v_wind_mean, dtype=torch.float32, device=self.device
        )

    @property
    def fidelity_level(self) -> int:
        """Current fidelity level."""
        return self._fidelity_level

    def set_fidelity_level(self, level: int):
        """Set aerodynamic fidelity level.

        Args:
            level: Fidelity level (0-3).
        """
        if level < 0 or level > 3:
            raise ValueError(f"Fidelity level must be 0-3, got {level}")
        self._fidelity_level = level

    def compute_forces(
        self,
        v_body_world: torch.Tensor,
        omega_rotors: torch.Tensor,
        q_body: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute aerodynamic force and moment.

        Args:
            v_body_world: (N, 3) body velocity in world frame [m/s].
            omega_rotors: (N, 4) rotor angular velocities [rad/s].
            q_body: (N, 4) body quaternion (wxyz).
            dt: Timestep [s].

        Returns:
            F_aero: (N, 3) aerodynamic force in body frame [N].
            tau_aero: (N, 3) aerodynamic moment in body frame [Nm].
        """
        # Initialize outputs
        F_aero = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        tau_aero = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # Level 0: No effects
        if self._fidelity_level == 0:
            return F_aero, tau_aero

        # Compute relative velocity (body - wind)
        if self._fidelity_level >= 2:
            # Update wind with gust model
            self._update_gust(dt)
            v_wind = self._v_wind_mean.unsqueeze(0) + self._gust
        else:
            v_wind = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        v_rel_world = v_body_world - v_wind

        # Transform relative velocity to body frame
        v_rel_body = quat_rotate_inverse(q_body, v_rel_world)

        # Level 1+: Basic drag
        if self._fidelity_level >= 1:
            F_drag = self._compute_drag(v_rel_body)
            F_aero = F_aero + F_drag

        # Level 3: Rotor effects
        if self._fidelity_level >= 3:
            F_rotor, tau_rotor = self._compute_rotor_effects(v_rel_body, omega_rotors)
            F_aero = F_aero + F_rotor
            tau_aero = tau_aero + tau_rotor

        return F_aero, tau_aero

    def _compute_drag(self, v_rel_body: torch.Tensor) -> torch.Tensor:
        """Compute quadratic drag force.

        F_drag = -0.5 * rho * Cd * A * |v|^2 * v_hat

        Args:
            v_rel_body: (N, 3) relative velocity in body frame [m/s].

        Returns:
            F_drag: (N, 3) drag force in body frame [N].
        """
        v_mag = torch.norm(v_rel_body, dim=-1, keepdim=True)
        v_hat = v_rel_body / (v_mag + 1e-6)

        # Quadratic drag
        F_drag = -self._drag_factor * v_mag**2 * v_hat

        return F_drag

    def _update_gust(self, dt: float):
        """Update gust using first-order filtered white noise (Dryden-like model).

        Args:
            dt: Timestep [s].
        """
        # First-order low-pass filter on white noise
        # tau = 1 / (2 * pi * bandwidth)
        tau = 1.0 / (2 * math.pi * self.cfg.gust_bandwidth)
        alpha = dt / (dt + tau)

        # Generate white noise
        noise = torch.randn_like(self._gust) * self.cfg.sigma_gust

        # Filter
        self._gust = self._gust + alpha * (noise - self._gust)

    def _compute_rotor_effects(
        self,
        v_rel_body: torch.Tensor,
        omega_rotors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotor-specific aerodynamic effects.

        Includes:
        - H-force: drag due to rotor disk in forward flight
        - Blade flapping moment: pitch/roll moment from asymmetric inflow

        Args:
            v_rel_body: (N, 3) relative velocity in body frame [m/s].
            omega_rotors: (N, 4) rotor angular velocities [rad/s].

        Returns:
            F_rotor: (N, 3) rotor drag force in body frame [N].
            tau_rotor: (N, 3) blade flapping moment in body frame [Nm].
        """
        # Total rotor speed (sum across all rotors)
        omega_total = omega_rotors.sum(dim=-1, keepdim=True)

        # Horizontal velocity (XY plane in body frame)
        v_xy = v_rel_body[:, :2]

        # H-force: F_H = -k_H * omega_total * v_xy
        F_H_xy = -self.cfg.k_H * omega_total * v_xy
        F_rotor = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        F_rotor[:, :2] = F_H_xy

        # Blade flapping moment: tau_flap = -k_flap * v_xy
        # Moment about perpendicular axis to velocity
        tau_rotor = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        tau_rotor[:, 0] = -self.cfg.k_flap * v_rel_body[:, 1]  # Roll from lateral velocity
        tau_rotor[:, 1] = self.cfg.k_flap * v_rel_body[:, 0]   # Pitch from forward velocity

        return F_rotor, tau_rotor

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset aerodynamic state.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._gust.zero_()
        else:
            self._gust[env_ids] = 0.0

    def set_wind(
        self,
        v_wind_mean: torch.Tensor | tuple,
        env_ids: torch.Tensor | None = None,
    ):
        """Set mean wind velocity.

        Args:
            v_wind_mean: Mean wind velocity [m/s].
            env_ids: Environment indices. If None, set all.
        """
        if isinstance(v_wind_mean, tuple):
            v_wind_mean = torch.tensor(v_wind_mean, dtype=torch.float32, device=self.device)

        if env_ids is None:
            self._v_wind_mean = v_wind_mean.to(self.device)
        else:
            # Per-env wind (need to expand state)
            if self._v_wind_mean.dim() == 1:
                self._v_wind_mean = self._v_wind_mean.unsqueeze(0).expand(self.num_envs, 3).clone()
            self._v_wind_mean[env_ids] = v_wind_mean.to(self.device)
