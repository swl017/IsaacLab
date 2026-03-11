# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zoom controller with first-order dynamics for optical/mechanical zoom."""

from __future__ import annotations

import torch

from .zoom_controller_cfg import ZoomControllerCfg


class ZoomController:
    """Zoom controller with first-order dynamics.

    Simulates the transient dynamics of a mechanical/optical zoom system.
    The zoom level follows commanded rate with first-order lag.
    """

    def __init__(
        self,
        cfg: ZoomControllerCfg,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize zoom controller.

        Args:
            cfg: Zoom controller configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        # State: current zoom level
        self._zoom = torch.ones(num_envs, dtype=torch.float32, device=self.device)

        # Target zoom level (before dynamics)
        self._zoom_target = torch.ones(num_envs, dtype=torch.float32, device=self.device)

        # Store tau for potential per-env randomization
        self._tau_zoom = cfg.tau_zoom

    @property
    def zoom(self) -> torch.Tensor:
        """Current zoom level (N,)."""
        return self._zoom

    def compute_control(
        self,
        zoom_rate_cmd: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute zoom level with first-order lag dynamics.

        Args:
            zoom_rate_cmd: (N,) zoom rate command, normalized [-1, 1].
            dt: Timestep [s].

        Returns:
            zoom_level: (N,) current zoom level [1x to zoom_max].
        """
        # Scale rate command
        zoom_rate = zoom_rate_cmd * self.cfg.max_zoom_rate

        # Integrate to get target zoom
        self._zoom_target = self._zoom_target + zoom_rate * dt

        # Clamp target to valid range
        self._zoom_target = torch.clamp(
            self._zoom_target, self.cfg.zoom_min, self.cfg.zoom_max
        )

        # Apply first-order lag dynamics with exact discretization
        alpha = 1.0 - torch.exp(torch.tensor(-dt / self._tau_zoom, device=self.device))
        self._zoom = self._zoom + alpha * (self._zoom_target - self._zoom)

        # Clamp output
        self._zoom = torch.clamp(self._zoom, self.cfg.zoom_min, self.cfg.zoom_max)

        return self._zoom

    def get_fov(self, base_fov: float = 90.0) -> torch.Tensor:
        """Get current field of view based on zoom level.

        Assumes simple relationship: FOV = base_fov / zoom

        Args:
            base_fov: Field of view at 1x zoom [degrees].

        Returns:
            fov: (N,) field of view in degrees.
        """
        return base_fov / self._zoom

    def get_focal_length_multiplier(self) -> torch.Tensor:
        """Get focal length multiplier relative to base lens.

        Returns:
            multiplier: (N,) focal length multiplier (= zoom level).
        """
        return self._zoom

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset zoom state.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._zoom.fill_(1.0)
            self._zoom_target.fill_(1.0)
        else:
            self._zoom[env_ids] = 1.0
            self._zoom_target[env_ids] = 1.0

    def set_tau_zoom(self, tau_zoom: torch.Tensor | float):
        """Set zoom time constant (for randomization).

        Args:
            tau_zoom: Time constant [s]. Can be scalar or (N,) tensor.
        """
        if isinstance(tau_zoom, torch.Tensor):
            self._tau_zoom = tau_zoom.to(self.device)
        else:
            self._tau_zoom = tau_zoom

    def set_zoom(self, zoom_level: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Directly set zoom level (for initialization/teleop).

        Args:
            zoom_level: Zoom level to set.
            env_ids: Environment indices. If None, set all.
        """
        zoom_clamped = torch.clamp(zoom_level, self.cfg.zoom_min, self.cfg.zoom_max)
        if env_ids is None:
            self._zoom = zoom_clamped.clone()
            self._zoom_target = zoom_clamped.clone()
        else:
            self._zoom[env_ids] = zoom_clamped
            self._zoom_target[env_ids] = zoom_clamped
