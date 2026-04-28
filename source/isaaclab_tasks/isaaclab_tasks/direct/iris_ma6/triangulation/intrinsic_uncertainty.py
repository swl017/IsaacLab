# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-zoom intrinsic-calibration covariance for triangulation.

Source: mrcal calibration of the SIYI A8 mini at 1x/2x/4x/5x —
`/home/usrg/mas/datasets/camera_calibration/2026-04-17/<Nx>/intrinsics_summary.json`.

The σ values come from mrcal's per-parameter calibration uncertainty (1σ).
The uncertainty grows superlinearly with zoom command: at 1x it is ~2.5 %
of fx, at 5x it is ~6.5 %. Triangulation covariance must reflect this so
that high-zoom rays get appropriately down-weighted.

Outputs `Sigma_K = diag(σ_fx², σ_fy², σ_cx², σ_cy²)` per (env, camera). The
mrcal calibration also reports correlations between fx/fy and cx/cy, but
the cross-terms are an order of magnitude smaller than the diagonal and
the iris_ma5 reference (`triang_cov_reward_torch.py:298`) treats Sigma_K
as block-diagonal with these four parameters.
"""
from __future__ import annotations

import torch


# Anchor points: (cmd, σ_fx, σ_fy, σ_cx, σ_cy) in pixels at render resolution.
# Pulled directly from intrinsics_summary.json per zoom level. cmd > 5 is
# untrusted (calibration uncertainty blows up), cmd < 1 is unphysical.
_ZOOM_ANCHORS_CMD = (1.0, 2.0, 4.0, 5.0)
_ZOOM_ANCHORS_SIGMA_FX = (26.491, 25.446, 69.119, 203.954)
_ZOOM_ANCHORS_SIGMA_FY = (26.175, 25.154, 68.146, 202.284)
_ZOOM_ANCHORS_SIGMA_CX = (4.570, 5.584, 8.720, 32.392)
_ZOOM_ANCHORS_SIGMA_CY = (5.430, 6.410, 11.434, 32.967)


def _piecewise_linear(
    cmd: torch.Tensor, anchors_x: tuple[float, ...], anchors_y: tuple[float, ...]
) -> torch.Tensor:
    """Piecewise-linear interpolation with flat extrapolation outside the anchor range."""
    device = cmd.device
    dtype = cmd.dtype
    xs = torch.tensor(anchors_x, device=device, dtype=dtype)
    ys = torch.tensor(anchors_y, device=device, dtype=dtype)

    # Clamp to anchor range — flat extrapolation matches the trustworthy
    # calibration domain (cmd ∈ [1, 5]).
    cmd_clamped = torch.clamp(cmd, min=xs[0].item(), max=xs[-1].item())

    # Find the segment for each sample.
    idx = torch.bucketize(cmd_clamped, xs[1:-1].contiguous())  # [..]; values 0..len(xs)-2
    idx = torch.clamp(idx, 0, len(anchors_x) - 2)

    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]

    # Avoid div-by-zero at coincident anchors (won't happen for our table)
    span = (x1 - x0).clamp(min=1e-12)
    t = (cmd_clamped - x0) / span
    return y0 + t * (y1 - y0)


def compute_sigma_K_from_zoom(zoom_cmd: torch.Tensor) -> torch.Tensor:
    """Build per-camera intrinsic covariance from the policy zoom command.

    Args:
        zoom_cmd: [N, C] operator zoom command (1.0 = 1x, 5.0 = 5x).

    Returns:
        Sigma_K: [N, C, 4, 4] diagonal covariance over (fx, fy, cx, cy)
            in (pixel)² units at render resolution.
    """
    if zoom_cmd.dim() != 2:
        raise ValueError(f"zoom_cmd must be [N, C], got shape {tuple(zoom_cmd.shape)}")
    N, C = zoom_cmd.shape
    device = zoom_cmd.device
    dtype = zoom_cmd.dtype

    sigma_fx = _piecewise_linear(zoom_cmd, _ZOOM_ANCHORS_CMD, _ZOOM_ANCHORS_SIGMA_FX)
    sigma_fy = _piecewise_linear(zoom_cmd, _ZOOM_ANCHORS_CMD, _ZOOM_ANCHORS_SIGMA_FY)
    sigma_cx = _piecewise_linear(zoom_cmd, _ZOOM_ANCHORS_CMD, _ZOOM_ANCHORS_SIGMA_CX)
    sigma_cy = _piecewise_linear(zoom_cmd, _ZOOM_ANCHORS_CMD, _ZOOM_ANCHORS_SIGMA_CY)

    Sigma_K = torch.zeros(N, C, 4, 4, device=device, dtype=dtype)
    Sigma_K[..., 0, 0] = sigma_fx ** 2
    Sigma_K[..., 1, 1] = sigma_fy ** 2
    Sigma_K[..., 2, 2] = sigma_cx ** 2
    Sigma_K[..., 3, 3] = sigma_cy ** 2
    return Sigma_K


def compute_sigma_K_constant(
    num_envs: int,
    num_cameras: int,
    sigma_fx: float,
    sigma_fy: float | None = None,
    sigma_cx: float | None = None,
    sigma_cy: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a constant Sigma_K [N, C, 4, 4] (e.g. for testing or fixed cfg).

    Defaults σ_fy = σ_fx, σ_cx = σ_cy = 0.5 * σ_fx.
    """
    if sigma_fy is None:
        sigma_fy = sigma_fx
    if sigma_cx is None:
        sigma_cx = 0.5 * sigma_fx
    if sigma_cy is None:
        sigma_cy = 0.5 * sigma_fx

    Sigma_K = torch.zeros(num_envs, num_cameras, 4, 4, device=device, dtype=dtype)
    Sigma_K[..., 0, 0] = sigma_fx ** 2
    Sigma_K[..., 1, 1] = sigma_fy ** 2
    Sigma_K[..., 2, 2] = sigma_cx ** 2
    Sigma_K[..., 3, 3] = sigma_cy ** 2
    return Sigma_K
