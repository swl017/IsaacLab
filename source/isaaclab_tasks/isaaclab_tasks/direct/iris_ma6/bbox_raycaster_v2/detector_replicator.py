# file: detector_replicator.py
"""Detector replicator — calibrated bbox noise from offline YOLO calibration.

Applies Student's t localization noise with size-dependent scale to geometric
raycaster bboxes, replicating the error profile of a real object detector.
Noise parameters are loaded from a JSON file produced by the calibration script
(experiments/calibrate_bbox_noise.py).

This module is the single architectural home for all detector modeling:
- Ticket 009: calibrated Student's t localization noise (this file)
- Ticket 010: miss rate (FN) and false positive (FP) injection (future extension)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from isaaclab.utils import configclass


@dataclass
class NoiseModelParams:
    """Fitted noise model parameters from offline YOLO calibration.

    Loaded from JSON produced by experiments/calibrate_bbox_noise.py.
    """

    noise_distribution: str = "student_t"
    """Distribution type for localization noise."""

    # Localization noise: scale = a / target_size_px + b
    # Sampled as StudentT(df, loc=bias, scale=scale)
    center_noise_a: float = 0.0
    """Size-dependent noise term for bbox center [px^2]. scale = a / size + b."""

    center_noise_b: float = 7.0
    """Constant floor for bbox center noise [px]."""

    center_noise_df: float = 16.0
    """Student's t degrees of freedom for center noise. Lower = heavier tails."""

    size_noise_a: float = 0.0
    """Size-dependent noise term for bbox width/height [px^2]."""

    size_noise_b: float = 7.0
    """Constant floor for bbox size noise [px]."""

    size_noise_df: float = 12.0
    """Student's t degrees of freedom for size noise."""

    # Systematic bias (pixels)
    center_bias_x: float = 0.0
    """Systematic bias in bbox center X [px]."""

    center_bias_y: float = 0.0
    """Systematic bias in bbox center Y [px]."""

    # Fitting metadata
    fit_size_range: Tuple[float, float] = (20.0, 80.0)
    """Target size range [px] used during fitting."""

    @classmethod
    def from_json(cls, path: str) -> "NoiseModelParams":
        """Load parameters from a calibration JSON file.

        Args:
            path: Path to bbox_noise_params.json.

        Returns:
            NoiseModelParams with loaded values.
        """
        with open(path, "r") as f:
            data = json.load(f)

        fit_range = data.get("fit_size_range", [20.0, 80.0])

        return cls(
            noise_distribution=data.get("noise_distribution", "student_t"),
            center_noise_a=float(data.get("center_noise_a", 0.0)),
            center_noise_b=float(data.get("center_noise_b", 7.0)),
            center_noise_df=float(data.get("center_noise_df", 16.0)),
            size_noise_a=float(data.get("size_noise_a", 0.0)),
            size_noise_b=float(data.get("size_noise_b", 7.0)),
            size_noise_df=float(data.get("size_noise_df", 12.0)),
            center_bias_x=float(data.get("center_bias_x", 0.0)),
            center_bias_y=float(data.get("center_bias_y", 0.0)),
            fit_size_range=(float(fit_range[0]), float(fit_range[1])),
        )


@configclass
class DetectorReplicatorCfg:
    """Configuration for the detector replicator."""

    enabled: bool = True
    """Enable calibrated bbox noise (replaces delay system bbox_std)."""

    params_path: str = ""
    """Path to bbox_noise_params.json from calibration run."""

    apply_bias: bool = True
    """Apply systematic center bias from calibration."""


class DetectorReplicator:
    """Applies calibrated detector noise to geometric raycaster bboxes.

    Given clean GT bboxes from BBoxRayCasterV2, produces replicated bboxes
    that match the error distribution of a real YOLO detector. Noise is
    Student's t distributed with size-dependent scale.

    The replicator only adds noise to valid (non-empty) bboxes. Empty bboxes
    (occluded, out of FOV) remain zero.
    """

    def __init__(
        self,
        cfg: DetectorReplicatorCfg,
        device: str = "cuda:0",
    ):
        """Initialize the detector replicator.

        Args:
            cfg: Replicator configuration.
            device: Torch device.
        """
        self._cfg = cfg
        self._device = torch.device(device)
        self._params: Optional[NoiseModelParams] = None

        if cfg.enabled and cfg.params_path:
            self.load_params(cfg.params_path)

    @property
    def params(self) -> Optional[NoiseModelParams]:
        """Currently loaded noise model parameters."""
        return self._params

    def load_params(self, path: str):
        """Load noise model parameters from JSON.

        Args:
            path: Path to calibration JSON file.
        """
        self._params = NoiseModelParams.from_json(path)

    def apply(
        self,
        bboxes_xywh: torch.Tensor,
        bbox_empty: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply calibrated noise to bounding boxes.

        Args:
            bboxes_xywh: Clean GT bboxes (N, C, T, 4) as (cx, cy, w, h) in pixels.
            bbox_empty: Empty mask (N, C, T). True = invalid/occluded.
            noise_scale: Curriculum scale [0, 1]. 0 = no noise, 1 = full calibrated noise.

        Returns:
            Tuple of:
                - bboxes_replicated: (N, C, T, 4) noisy bboxes in xywh format
                - bbox_empty_replicated: (N, C, T) empty mask (same as input for ticket-009;
                  ticket-010 will extend with miss injection)
        """
        if self._params is None or noise_scale <= 0.0:
            return bboxes_xywh.clone(), bbox_empty.clone()

        p = self._params
        valid = ~bbox_empty  # (N, C, T)

        # Compute target size as geometric mean of w and h
        w = bboxes_xywh[..., 2]  # (N, C, T)
        h = bboxes_xywh[..., 3]  # (N, C, T)
        target_size = torch.sqrt(w * h).clamp(min=5.0)  # (N, C, T)

        # --- Center noise: Student's t with size-dependent scale ---
        center_scale = (p.center_noise_a / target_size + p.center_noise_b) * noise_scale
        center_noise = _sample_student_t(
            df=p.center_noise_df,
            scale=center_scale,
            shape=center_scale.shape,
            device=self._device,
        )  # (N, C, T)

        # --- Size noise: Student's t with size-dependent scale ---
        size_scale = (p.size_noise_a / target_size + p.size_noise_b) * noise_scale
        size_noise = _sample_student_t(
            df=p.size_noise_df,
            scale=size_scale,
            shape=size_scale.shape,
            device=self._device,
        )  # (N, C, T)

        # Build noise tensor (N, C, T, 4): [cx_noise, cy_noise, w_noise, h_noise]
        noise = torch.zeros_like(bboxes_xywh)
        noise[..., 0] = center_noise  # cx
        noise[..., 1] = _sample_student_t(  # cy (independent sample)
            df=p.center_noise_df,
            scale=center_scale,
            shape=center_scale.shape,
            device=self._device,
        )
        noise[..., 2] = size_noise  # w
        noise[..., 3] = _sample_student_t(  # h (independent sample)
            df=p.size_noise_df,
            scale=size_scale,
            shape=size_scale.shape,
            device=self._device,
        )

        # Add bias if configured
        if self._cfg.apply_bias:
            noise[..., 0] += p.center_bias_x * noise_scale
            noise[..., 1] += p.center_bias_y * noise_scale

        # Apply noise only to valid (non-empty) bboxes
        bboxes_replicated = bboxes_xywh.clone()
        valid_expanded = valid.unsqueeze(-1).expand_as(noise)  # (N, C, T, 4)
        bboxes_replicated[valid_expanded] += noise[valid_expanded]

        # Clamp width/height to be non-negative
        bboxes_replicated[..., 2:].clamp_(min=0.0)

        # Zero out empty bboxes (ensure they stay zero)
        empty_expanded = bbox_empty.unsqueeze(-1).expand_as(bboxes_replicated)
        bboxes_replicated[empty_expanded] = 0.0

        # For ticket-009, empty mask is unchanged (no miss injection yet)
        bbox_empty_replicated = bbox_empty.clone()

        return bboxes_replicated, bbox_empty_replicated


def _sample_student_t(
    df: float,
    scale: torch.Tensor,
    shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    """Sample from a Student's t distribution with given scale.

    Uses the standard transformation: t = z * sqrt(df / chi2(df))
    where z ~ N(0, scale) and chi2 ~ Chi2(df). This avoids
    torch.distributions overhead for batched sampling.

    Args:
        df: Degrees of freedom.
        scale: Scale parameter tensor (broadcast-compatible with shape).
        shape: Output shape.
        device: Torch device.

    Returns:
        Samples from Student's t(df, loc=0, scale=scale).
    """
    # For large df, Student's t ≈ Gaussian
    if df > 100.0:
        return torch.randn(shape, device=device) * scale

    # t = z * sqrt(df / chi2) where chi2 ~ Gamma(df/2, 2)
    z = torch.randn(shape, device=device)
    # chi2 = 2 * Gamma(df/2, 1)
    chi2 = 2.0 * torch.distributions.Gamma(df / 2.0, 1.0).sample(shape).to(device)
    t_samples = z * torch.sqrt(torch.tensor(df, device=device) / chi2)

    return t_samples * scale
