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
    center_noise_a: float = 3.32094098449108e-08
    """Size-dependent noise term for bbox center [px^2]. scale = a / size + b."""

    center_noise_b: float = 0.9903605416833782
    """Constant floor for bbox center noise [px]."""

    center_noise_df: float = 3.14485116541261
    """Student's t degrees of freedom for center noise. Lower = heavier tails."""

    size_noise_a: float = 52.57566163317716
    """Size-dependent noise term for bbox width/height [px^2]."""

    size_noise_b: float = 1.0712118528882864
    """Constant floor for bbox size noise [px]."""

    size_noise_df: float = 7.728841290635469
    """Student's t degrees of freedom for size noise."""

    # Systematic bias (pixels)
    center_bias_x: float = -0.236480712890625
    """Systematic bias in bbox center X [px]."""

    center_bias_y: float = -0.71551513671875
    """Systematic bias in bbox center Y [px]."""

    # Miss rate: p_miss = sigmoid(a * (threshold - target_size_px))
    # Dual sigmoids for sky vs ground background
    miss_sigmoid_a_sky: float = 0.024751038174665885
    """Miss sigmoid steepness for sky background."""

    miss_size_cutoff_sky: float = 33.0
    """Size cutoff below which sky targets are always missed (px)."""

    miss_size_threshold_sky: float = 1.2214178344000452e-12
    """Miss sigmoid threshold (px) for sky background."""

    miss_sigmoid_a_gnd: float = 0.013808565449099726
    """Miss sigmoid steepness for ground background."""

    miss_size_cutoff_gnd: float = 60.0
    """Size cutoff below which ground targets are always missed (px)."""

    miss_size_threshold_gnd: float = 45.56221154006872
    """Miss sigmoid threshold (px) for ground background. Lower = more aggressive miss."""

    # False positive rate
    fp_rate: float = 0.03469779341221618
    """False positive probability per camera per step."""

    fp_size_range: Tuple[float, float] = (10.0, 60.0)
    """Min/max FP bbox size in pixels (w and h sampled independently)."""

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

        fp_size = data.get("fp_size_range", [10.0, 120.0])

        # Backward compat: old JSON has single miss_sigmoid_a / miss_size_threshold_px
        # New JSON has per-background miss_sigmoid_a_sky/gnd, miss_size_threshold_sky/gnd
        miss_a_fallback = float(data.get("miss_sigmoid_a", 0.031))
        miss_thresh_fallback = float(data.get("miss_size_threshold_px", 65.0))

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
            miss_sigmoid_a_sky=float(data.get("miss_sigmoid_a_sky", miss_a_fallback)),
            miss_size_threshold_sky=float(data.get("miss_size_threshold_sky", miss_thresh_fallback)),
            miss_sigmoid_a_gnd=float(data.get("miss_sigmoid_a_gnd", miss_a_fallback)),
            miss_size_threshold_gnd=float(data.get("miss_size_threshold_gnd", miss_thresh_fallback)),
            fp_rate=float(data.get("fp_rate", 0.01)),
            fp_size_range=(float(fp_size[0]), float(fp_size[1])),
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

    apply_miss: bool = True
    """Enable probabilistic miss rate (FN). Requires enabled=True."""

    apply_fp: bool = True
    """Enable false positive injection. Requires enabled=True."""


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
        fp_fn_scale: float = 1.0,
        bg_is_ground: torch.Tensor | None = None,
        image_shapes: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply calibrated noise, miss rate, and FP to bounding boxes.

        Stage order: noise → miss → FP → zero-out empty.

        Args:
            bboxes_xywh: Clean GT bboxes (N, C, T, 4) as (cx, cy, w, h) in pixels.
            bbox_empty: Empty mask (N, C, T). True = invalid/occluded.
            noise_scale: Curriculum scale [0, 1] for localization noise.
            fp_fn_scale: Curriculum scale [0, 1] for miss rate and FP. 0 = no FN/FP.
            bg_is_ground: (N, C, T) bool. True = ground background. None = assume sky.
            image_shapes: (N, C, 2) as (height, width). Required for FP placement.

        Returns:
            Tuple of:
                - bboxes_replicated: (N, C, T, 4) noisy bboxes in xywh format
                - bbox_empty_replicated: (N, C, T) updated empty mask
        """
        bboxes_replicated = bboxes_xywh.clone()
        bbox_empty_replicated = bbox_empty.clone()

        if self._params is None:
            return bboxes_replicated, bbox_empty_replicated

        p = self._params
        valid = ~bbox_empty  # (N, C, T)

        # Compute target size as geometric mean of w and h
        w = bboxes_xywh[..., 2]  # (N, C, T)
        h = bboxes_xywh[..., 3]  # (N, C, T)
        target_size = torch.sqrt(w * h).clamp(min=5.0)  # (N, C, T)

        # === Stage 1: Localization noise ===
        if noise_scale > 0.0:
            # Center noise: Student's t with size-dependent scale
            center_scale = (p.center_noise_a / target_size + p.center_noise_b) * noise_scale
            size_scale = (p.size_noise_a / target_size + p.size_noise_b) * noise_scale

            noise = torch.zeros_like(bboxes_xywh)
            noise[..., 0] = _sample_student_t(p.center_noise_df, center_scale, center_scale.shape, self._device)
            noise[..., 1] = _sample_student_t(p.center_noise_df, center_scale, center_scale.shape, self._device)
            # Correlated w/h noise: single draw applied to both dimensions
            # to prevent impossible aspect ratios from independent sampling
            size_noise_shared = _sample_student_t(p.size_noise_df, size_scale, size_scale.shape, self._device)
            noise[..., 2] = 1.0 * size_noise_shared
            noise[..., 3] = 1.0 * size_noise_shared

            if self._cfg.apply_bias:
                noise[..., 0] += p.center_bias_x * noise_scale
                noise[..., 1] += p.center_bias_y * noise_scale

            valid_expanded = valid.unsqueeze(-1).expand_as(noise)
            bboxes_replicated[valid_expanded] += noise[valid_expanded]
            bboxes_replicated[..., 2:].clamp_(min=0.0)

        # === Stage 2: Miss rate (FN) ===
        if self._cfg.apply_miss and fp_fn_scale > 0.0:
            bboxes_replicated, bbox_empty_replicated = self._apply_miss(
                bboxes_replicated, bbox_empty_replicated, target_size, bg_is_ground, fp_fn_scale,
            )

        # === Stage 3: False positive ===
        if self._cfg.apply_fp and fp_fn_scale > 0.0 and image_shapes is not None:
            bboxes_replicated, bbox_empty_replicated = self._apply_fp(
                bboxes_replicated, bbox_empty_replicated, fp_fn_scale, image_shapes,
            )

        # === Stage 4: Zero-out empty bboxes ===
        empty_expanded = bbox_empty_replicated.unsqueeze(-1).expand_as(bboxes_replicated)
        bboxes_replicated[empty_expanded] = 0.0

        return bboxes_replicated, bbox_empty_replicated

    def _apply_miss(
        self,
        bboxes: torch.Tensor,
        bbox_empty: torch.Tensor,
        target_size: torch.Tensor,
        bg_is_ground: torch.Tensor | None,
        fp_fn_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply probabilistic miss rate conditioned on target size and background.

        Args:
            bboxes: (N, C, T, 4) replicated bboxes (post-noise).
            bbox_empty: (N, C, T) current empty mask.
            target_size: (N, C, T) geometric mean of w and h in pixels.
            bg_is_ground: (N, C, T) bool. True = ground. None = assume sky for all.
            fp_fn_scale: Curriculum scale [0, 1].

        Returns:
            Tuple of (bboxes, bbox_empty) with missed detections zeroed.
        """
        p = self._params

        # Compute miss probability with dual sigmoids
        if bg_is_ground is not None:
            # Ground: lower threshold = more aggressive miss
            # p_miss_gnd = torch.sigmoid(p.miss_sigmoid_a_gnd * (p.miss_size_threshold_gnd - target_size))
            p_miss_gnd = (2.0 * torch.sigmoid(p.miss_sigmoid_a_gnd * (p.miss_size_threshold_gnd - target_size))).clamp_(max=1.0)
            p_miss_gnd = torch.where(target_size < p.miss_size_cutoff_gnd, torch.ones_like(p_miss_gnd), p_miss_gnd)  # Floor for very small targets
            # p_miss_sky = torch.sigmoid(p.miss_sigmoid_a_sky * (p.miss_size_threshold_sky - target_size))
            p_miss_sky = 1.0 * torch.sigmoid(p.miss_sigmoid_a_sky * (p.miss_size_threshold_sky - target_size))
            p_miss_sky = torch.where(target_size < p.miss_size_cutoff_sky, torch.ones_like(p_miss_sky), p_miss_sky)  # Floor for very small targets
            p_miss = torch.where(bg_is_ground, p_miss_gnd, p_miss_sky)
        else:
            p_miss = torch.sigmoid(p.miss_sigmoid_a_sky * (p.miss_size_threshold_sky - target_size))

        # Scale by curriculum
        p_miss = p_miss * fp_fn_scale

        # Sample miss — only for currently valid (non-empty) detections
        missed = torch.bernoulli(p_miss).bool() & ~bbox_empty

        # Apply miss: zero bbox, mark empty
        bboxes = bboxes.clone()
        bboxes[missed.unsqueeze(-1).expand_as(bboxes)] = 0.0
        bbox_empty = bbox_empty | missed

        return bboxes, bbox_empty

    def _apply_fp(
        self,
        bboxes: torch.Tensor,
        bbox_empty: torch.Tensor,
        fp_fn_scale: float,
        image_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inject false positive bboxes with random location and size.

        For each camera, samples a Bernoulli with probability p_fp * fp_fn_scale.
        If triggered, writes a random bbox into target slot 0 regardless of current
        content (a FP can replace a missed or valid detection).

        Args:
            bboxes: (N, C, T, 4) replicated bboxes (post-noise, post-miss).
            bbox_empty: (N, C, T) current empty mask.
            fp_fn_scale: Curriculum scale [0, 1].
            image_shapes: (N, C, 2) as (height, width) in pixels.

        Returns:
            Tuple of (bboxes, bbox_empty) with FPs inserted.
        """
        p = self._params
        N, C, T, _ = bboxes.shape
        p_fp = p.fp_rate * fp_fn_scale

        if p_fp <= 0.0:
            return bboxes, bbox_empty

        # Sample FP trigger independently per camera-target pair (N, C, T)
        fp_mask = torch.bernoulli(torch.full((N, C, T), p_fp, device=self._device)).bool()

        if not fp_mask.any():
            return bboxes, bbox_empty

        bboxes = bboxes.clone()
        bbox_empty = bbox_empty.clone()

        # Image dimensions for bounds: (N, C, 2) → expand to (N, C, T)
        img_h = image_shapes[..., 0].float().unsqueeze(-1).expand(N, C, T)  # (N, C, T)
        img_w = image_shapes[..., 1].float().unsqueeze(-1).expand(N, C, T)  # (N, C, T)

        # Random center: uniform within image
        fp_cx = torch.rand(N, C, T, device=self._device) * img_w
        fp_cy = torch.rand(N, C, T, device=self._device) * img_h

        # Random size: uniform within configured range
        size_min, size_max = p.fp_size_range
        fp_w = torch.rand(N, C, T, device=self._device) * (size_max - size_min) + size_min
        fp_h = torch.rand(N, C, T, device=self._device) * (size_max - size_min) + size_min

        # Build FP bbox tensor (N, C, T, 4) and write where triggered
        fp_bbox = torch.stack([fp_cx, fp_cy, fp_w, fp_h], dim=-1)  # (N, C, T, 4)
        fp_mask_expanded = fp_mask.unsqueeze(-1).expand_as(bboxes)  # (N, C, T, 4)
        bboxes[fp_mask_expanded] = fp_bbox[fp_mask_expanded]
        bbox_empty[fp_mask] = False  # FP is a "valid" detection

        return bboxes, bbox_empty


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
