# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-to-Collision (TTC) computation module for visual tracking."""

from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab.utils import configclass


@configclass
class TTCComputerCfg:
    """Configuration for time-to-collision computer."""

    horizon_sec: float = 6.0
    """TTC horizon in seconds. TTC values beyond this are clamped."""

    ema_alpha_s: float = 0.6
    """EMA smoothing factor for log-size (0=no smoothing, 1=no update)."""

    ema_alpha_f: float = 0.6
    """EMA smoothing factor for log-focal length (0=no smoothing, 1=no update)."""

    deriv_clip: float = 0.5
    """Clip derivative to this range to avoid numerical instability."""

    stale_half_life: float = 1.0
    """Half-life for staleness decay when detections become invalid (seconds)."""

    zoom_gate_k: float = 0.2
    """Gate aggressiveness for zoom motion. Lower values are more aggressive."""

    eps: float = 1e-6
    """Small epsilon for numerical stability."""

    ttc_penalty_scale: float = -10.0
    """Scale factor for TTC penalty in rewards."""


class TTCComputer:
    """
    Time-to-Collision (TTC) computer using zoom-invariant looming cues.

    This class implements a zoom-invariant approach to TTC estimation based on
    the rate of change of the logarithm of the ratio between bbox size and focal length.
    This makes the TTC estimate robust to zoom operations.

    The key insight is that for a camera approaching a target:
    - The bbox size 's' changes with distance
    - The focal length 'f' changes with zoom
    - The ratio s/f is zoom-invariant for a fixed-size target

    TTC is computed from: tau = -1 / (d/dt log(s/f))

    References:
        - Lee, D. N. (1976). A theory of visual control of braking based on
          information about time-to-collision. Perception, 5(4), 437-459.
    """

    def __init__(
        self,
        cfg: TTCComputerCfg,
        num_envs: int,
        device: torch.device,
    ):
        """
        Initialize the TTC computer.

        Args:
            cfg: Configuration for the TTC computer.
            num_envs: Number of parallel environments.
            device: Device for tensor computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # EMA buffers for log-size and log-focal
        self.logsize_ema = torch.zeros(num_envs, device=device)
        self.logf_ema = torch.zeros(num_envs, device=device)

        # Previous log(s/f) for derivative computation
        self.log_g_prev = torch.zeros(num_envs, device=device)

        # Time since last valid detection
        self.time_since_valid = torch.zeros(num_envs, device=device)

    def compute_ttc(
        self,
        bbox_width: torch.Tensor,
        bbox_height: torch.Tensor,
        valid_mask: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute zoom-invariant looming TTC penalty.

        Args:
            bbox_width: Bbox width in normalized coordinates [N].
            bbox_height: Bbox height in normalized coordinates [N].
            valid_mask: Boolean mask for valid detections [N].
            fx: Focal length x in pixels [N].
            fy: Focal length y in pixels [N].
            dt: Timestep (seconds).

        Returns:
            Tuple of:
            - phi: TTC penalty in [0, 1] where 1 means imminent collision [N].
            - tau: Estimated TTC in seconds [N].
        """
        # ---- 1) Build size and focal terms ----
        s = torch.sqrt(
            torch.clamp(bbox_width, min=0) * torch.clamp(bbox_height, min=0)
        )  # Geometric mean
        f_eff = torch.sqrt(
            torch.clamp(fx, min=self.cfg.eps) * torch.clamp(fy, min=self.cfg.eps)
        )
        valid = valid_mask & (s > 0)

        g_s = torch.log(s + self.cfg.eps)  # log size
        g_f = torch.log(f_eff + self.cfg.eps)  # log focal

        # ---- 2) EMA updates only where valid ----
        size_ema_new = self.cfg.ema_alpha_s * self.logsize_ema + (
            1.0 - self.cfg.ema_alpha_s
        ) * g_s
        focal_ema_new = self.cfg.ema_alpha_f * self.logf_ema + (
            1.0 - self.cfg.ema_alpha_f
        ) * g_f

        self.logsize_ema = torch.where(valid, size_ema_new, self.logsize_ema)
        self.logf_ema = torch.where(valid, focal_ema_new, self.logf_ema)

        # ---- 3) Zoom-invariant log-size: g = log(s/f) ----
        g_now = self.logsize_ema - self.logf_ema  # [N]

        # ---- 4) Derivative using previous g ----
        dg = (g_now - self.log_g_prev) / max(dt, 1e-6)
        self.log_g_prev = g_now.clone()

        # Robustify derivative
        dg = torch.clamp(dg, min=-self.cfg.deriv_clip, max=self.cfg.deriv_clip)

        # ---- 5) Zoom motion gate ----
        # Large |d ln f| means active zoom; soften penalty
        dlogf = (self.logf_ema - focal_ema_new).abs() / max(dt, 1e-6)
        w_zoom = torch.exp(-dlogf / max(self.cfg.zoom_gate_k, 1e-6))
        w_zoom = torch.clamp(w_zoom, 0.0, 1.0)

        # ---- 6) Looming TTC from zoom-invariant derivative ----
        tau = torch.full_like(dg, float("inf"))
        approaching = (dg < -1e-4) & valid
        tau = torch.where(approaching, -1.0 / torch.clamp(dg, max=-1e-4), tau)

        # ---- 7) Shape to [0,1] and apply staleness + gates ----
        H = float(self.cfg.horizon_sec)
        phi = (H - torch.clamp(tau, max=H)) / H
        phi = torch.clamp(phi, 0.0, 1.0)

        # Staleness decay
        self.time_since_valid = torch.where(
            valid,
            torch.zeros_like(self.time_since_valid),
            self.time_since_valid + dt,
        )
        stale_decay = torch.exp(
            -self.time_since_valid / max(self.cfg.stale_half_life, 1e-6)
        )

        # Combine gates
        gate = stale_decay * w_zoom
        phi = phi * gate

        return phi, tau

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset TTC buffers for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self.logsize_ema.zero_()
            self.logf_ema.zero_()
            self.log_g_prev.zero_()
            self.time_since_valid.zero_()
        else:
            self.logsize_ema[env_ids] = 0.0
            self.logf_ema[env_ids] = 0.0
            self.log_g_prev[env_ids] = 0.0
            self.time_since_valid[env_ids] = 0.0

    def reset_with_current_state(
        self,
        env_ids: torch.Tensor,
        bbox_width: torch.Tensor,
        bbox_height: torch.Tensor,
        valid_mask: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
    ):
        """
        Reset TTC buffers with current bbox and focal length state.

        This is useful for initializing the EMA buffers with valid initial values
        rather than zeros, which improves the first few TTC estimates.

        Args:
            env_ids: Environment indices to reset [K].
            bbox_width: Current bbox width [N].
            bbox_height: Current bbox height [N].
            valid_mask: Current validity mask [N].
            fx: Current focal length x [N].
            fy: Current focal length y [N].
        """
        if env_ids is None or env_ids.numel() == 0:
            return

        ridx = env_ids

        # Current scale and focal (for those envs)
        w = torch.clamp(bbox_width[ridx], min=0.0)
        h = torch.clamp(bbox_height[ridx], min=0.0)
        s = torch.sqrt(w * h)
        f_eff = torch.sqrt(
            torch.clamp(fx[ridx], min=self.cfg.eps)
            * torch.clamp(fy[ridx], min=self.cfg.eps)
        )

        valid = (valid_mask[ridx]) & (s > 0.0)

        # Log terms
        g_s = torch.log(s + self.cfg.eps)
        g_f = torch.log(f_eff + self.cfg.eps)
        g_now = g_s - g_f

        # Initialize EMAs/prev only where valid; zero where invalid
        self.logsize_ema[ridx] = torch.where(valid, g_s, torch.zeros_like(g_s))
        self.logf_ema[ridx] = torch.where(valid, g_f, torch.zeros_like(g_f))
        self.log_g_prev[ridx] = torch.where(valid, g_now, torch.zeros_like(g_now))
        self.time_since_valid[ridx] = 0.0
