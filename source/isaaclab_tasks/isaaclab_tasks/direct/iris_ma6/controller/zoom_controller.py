# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zoom controller with first-order dynamics for optical/mechanical zoom."""

from __future__ import annotations

import torch

from .zoom_controller_cfg import ZoomControllerCfg


# Measured zoom curve from the SIYI camera bench fit:
#   z_eff = 1 + a * (exp(b * (cmd - 1)) - 1)
# Source: /home/usrg/mas/src/scripts/camera_calibration/zoom_curve.json
# Trustworthy domain: cmd in [1.0, 5.0]. cmd > 5 is clamped because the 6x
# calibration uncertainty exceeds the trust threshold (mas/028).
ZOOM_CURVE_A: float = 0.32489
ZOOM_CURVE_B: float = 0.4767
ZOOM_CURVE_CMD_MAX: float = 5.0


def compute_z_eff(zoom_cmd: torch.Tensor) -> torch.Tensor:
    """Operator zoom command -> effective focal-length multiplier.

    Maps the operator-facing zoom command (1.0 = 1x, 5.0 = 5x) to the
    actual focal-length multiplier the real SIYI camera applies. The
    relationship is sub-linear: cmd=5 corresponds to z_eff ≈ 2.86, not 5.

    Inputs above ZOOM_CURVE_CMD_MAX (5.0) are clamped before the exponential
    is applied. cmd ∈ [1, 5] is the trust region of the underlying mrcal
    calibration.
    """
    cmd = torch.clamp(zoom_cmd, min=1.0, max=ZOOM_CURVE_CMD_MAX)
    return 1.0 + ZOOM_CURVE_A * (torch.exp(ZOOM_CURVE_B * (cmd - 1.0)) - 1.0)


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

        # Store tau and max_zoom_rate as per-env tensors for batch indexing
        self._tau_zoom = torch.full(
            (num_envs,), cfg.tau_zoom, dtype=torch.float32, device=self.device
        )
        self._max_zoom_rate = torch.full(
            (num_envs,), cfg.max_zoom_rate, dtype=torch.float32, device=self.device
        )

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
        # Scale rate command (supports per-env _max_zoom_rate)
        zoom_rate = zoom_rate_cmd * self._max_zoom_rate

        # Target = current actual zoom + rate * dt (no accumulated drift)
        self._zoom_target = self._zoom + zoom_rate * dt

        # Clamp target to valid range
        self._zoom_target = torch.clamp(
            self._zoom_target, self.cfg.zoom_min, self.cfg.zoom_max
        )

        # Apply first-order lag dynamics with exact discretization
        alpha = 1.0 - torch.exp(torch.as_tensor(-dt / self._tau_zoom, device=self.device))
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

        Applies the measured SIYI zoom curve so that zoom command 5.0 maps
        to ~2.86x effective focal multiplier (not 5.0x). Use the raw
        operator command via `self.zoom` when feeding observations.

        Returns:
            multiplier: (N,) effective focal length multiplier (z_eff).
        """
        return compute_z_eff(self._zoom)

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

    def set_max_zoom_rate(self, max_zoom_rate: torch.Tensor | float):
        """Set maximum zoom rate (for randomization).

        Args:
            max_zoom_rate: Max zoom rate [1/s]. Can be scalar or (N,) tensor.
        """
        if isinstance(max_zoom_rate, torch.Tensor):
            self._max_zoom_rate = max_zoom_rate.to(self.device)
        else:
            self._max_zoom_rate = max_zoom_rate

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
