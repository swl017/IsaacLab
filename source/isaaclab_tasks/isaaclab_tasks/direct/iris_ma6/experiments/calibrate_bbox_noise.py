#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline YOLO-vs-raycaster calibration for bbox noise model.

Runs a small number of envs with TiledCameras enabled, collects YOLO detections
alongside geometric raycaster GT, and fits a parametric noise model
(size-dependent localization error, miss rate, FP rate, bias).

The fitted parameters can then be loaded into the main 1024-env training run
via NoiseCfg, replacing the need for manual detector modeling (tickets 009/010).

Usage:
    # With a trained checkpoint (recommended — realistic bbox sizes)
    ./isaaclab.sh -p .../calibrate_bbox_noise.py \
        --experiment a3_full \
        --checkpoint /path/to/best_agent.pt \
        --num_envs 32 --num_steps 5000 \
        --output bbox_noise_params.json

    # With random policy (quick sanity check)
    ./isaaclab.sh -p .../calibrate_bbox_noise.py \
        --num_envs 32 --num_steps 2000 \
        --output bbox_noise_params.json
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="Calibrate bbox noise model from YOLO vs raycaster comparison.")
parser.add_argument("--experiment", type=str, default=None,
                    help="Experiment name from registry (for env overrides)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained RL policy checkpoint (.pt). If None, uses random actions.")
parser.add_argument("--num_envs", type=int, default=32,
                    help="Number of parallel environments (all get cameras)")
parser.add_argument("--num_steps", type=int, default=5000,
                    help="Number of policy steps to collect")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA6-Direct-Test-v0")
parser.add_argument("--yolo_model", type=str,
                    default="/home/usrg/mas/src/ultralytics_ros/models/yolov11m-drone.pt",
                    help="Path to YOLO detector model weights (.pt)")
parser.add_argument("--yolo_conf", type=float, default=0.25,
                    help="YOLO confidence threshold")
parser.add_argument("--output", type=str, default="bbox_noise_params.json",
                    help="Output JSON path for fitted noise parameters")
parser.add_argument("--save_raw", action="store_true", default=False,
                    help="Also save raw comparison data (large)")
parser.add_argument("--video_envs", type=int, default=1,
                    help="Number of envs to record video for (0 to disable)")
parser.add_argument("--video_fps", type=int, default=25,
                    help="Video output FPS")

# Parse our args first, pass unknown args to Hydra
args_cli, hydra_args = parser.parse_known_args()

# Strip our custom args from sys.argv so Hydra doesn't choke on them
sys.argv = [sys.argv[0]] + hydra_args

# Isaac Sim requires AppLauncher before any other imports
from isaaclab.app import AppLauncher

app_args = argparse.Namespace(
    headless=True,
    device="cuda:0",
    experience="",
    enable_cameras=True,
)
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is initialized."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Import experiment registry
from isaaclab_tasks.direct.iris_ma6.experiments.experiment_registry import get_experiment
from isaaclab_tasks.direct.iris_ma6.experiments.env_overrides import apply_env_overrides, apply_agent_overrides


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #

@dataclass
class NoiseModelParams:
    """Fitted parametric noise model for bbox detector."""

    # Localization noise (pixels): scale = a / target_size_px + b
    # Distribution is Student's t (Gaussian-like rounded peak + heavy tails).
    # df controls tail heaviness: df→∞ = Gaussian, df~3 = heavy tails.
    noise_distribution: str = "student_t"
    center_noise_a: float = 0.0
    center_noise_b: float = 7.0  # fallback = current bbox_std
    center_noise_df: float = 5.0  # Student's t degrees of freedom
    size_noise_a: float = 0.0
    size_noise_b: float = 7.0
    size_noise_df: float = 5.0

    # Miss rate: p_miss = sigmoid(a * (threshold - target_size_px))
    miss_sigmoid_a: float = 0.1
    miss_size_threshold_px: float = 20.0

    # False positive rate (per frame per camera)
    fp_rate: float = 0.0

    # Systematic bias (pixels)
    center_bias_x: float = 0.0
    center_bias_y: float = 0.0

    # Fitting regime — only fit noise/miss for targets in this size range (px)
    fit_size_range: tuple = (20.0, 80.0)

    # Metadata
    num_frames: int = 0
    num_matches: int = 0
    num_misses: int = 0
    num_false_positives: int = 0
    yolo_model: str = ""
    yolo_conf_threshold: float = 0.25


@dataclass
class ComparisonSample:
    """Single matched YOLO-vs-raycaster comparison."""
    target_size_px: float  # sqrt(w*h) of raycaster bbox
    center_error_x: float  # YOLO center_x - raycaster center_x
    center_error_y: float
    size_error_w: float    # YOLO w - raycaster w
    size_error_h: float
    yolo_confidence: float
    distance_m: float      # camera-to-target distance


@dataclass
class FalsePositiveSample:
    """Single false positive YOLO detection (no matching raycaster GT)."""
    bbox_cx: float         # FP bbox center x (pixels)
    bbox_cy: float         # FP bbox center y (pixels)
    bbox_w: float          # FP bbox width (pixels)
    bbox_h: float          # FP bbox height (pixels)
    confidence: float      # YOLO confidence score
    gt_target_size_px: float  # raycaster GT target size (if visible, else 0)
    gt_target_visible: bool   # whether raycaster said target was visible


# --------------------------------------------------------------------------- #
#  YOLO wrapper
# --------------------------------------------------------------------------- #

class YoloBatchInference:
    """Thin wrapper around ultralytics YOLO for batch GPU inference."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 conf_threshold: float = 0.25, target_class: int = 0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.target_class = target_class

    def detect_batch(self, images_nhwc: torch.Tensor):
        """Run YOLO on a batch of images.

        Args:
            images_nhwc: (B, H, W, 3 or 4) uint8/float RGB(A) tensor from TiledCamera.

        Returns:
            List of length B, each element is a dict with:
                "bboxes_xywh": (K, 4) tensor — center_x, center_y, w, h in pixels
                "confidences": (K,) tensor
                "classes": (K,) tensor
            where K is the number of detections for that image (0 if none).
        """
        # Convert to uint8 numpy
        if images_nhwc.dtype == torch.float32:
            images_np = (images_nhwc * 255).byte().cpu().numpy()
        else:
            images_np = images_nhwc.cpu().numpy()

        # Strip alpha channel if RGBA
        if images_np.shape[-1] == 4:
            images_np = images_np[..., :3]

        # ultralytics expects a list of individual (H, W, 3) arrays
        image_list = [images_np[i] for i in range(images_np.shape[0])]

        results = self.model.predict(
            source=image_list,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            classes=[self.target_class],
        )

        detections = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                # r.boxes.xywh is center_x, center_y, w, h
                bboxes = r.boxes.xywh.cpu()
                confs = r.boxes.conf.cpu()
                classes = r.boxes.cls.cpu()
                detections.append({
                    "bboxes_xywh": bboxes,
                    "confidences": confs,
                    "classes": classes,
                })
            else:
                detections.append({
                    "bboxes_xywh": torch.zeros(0, 4),
                    "confidences": torch.zeros(0),
                    "classes": torch.zeros(0),
                })
        return detections


# --------------------------------------------------------------------------- #
#  Calibrator — accumulates statistics and fits noise model
# --------------------------------------------------------------------------- #

class DetectorCalibrator:
    """Compares YOLO detections vs raycaster GT and fits a noise model."""

    def __init__(self, iou_match_threshold: float = 0.1):
        self.iou_match_threshold = iou_match_threshold
        self.matches: list[ComparisonSample] = []
        self.miss_data: list[tuple[float, bool]] = []  # (target_size_px, was_missed)
        self.false_positives: list[FalsePositiveSample] = []
        self.fp_count: int = 0
        self.frame_count: int = 0

    def ingest(self, yolo_detections: list[dict],
               raycaster_bboxes_xywh: torch.Tensor,
               raycaster_confidence: torch.Tensor,
               camera_to_target_dist: Optional[torch.Tensor] = None):
        """Compare one batch of YOLO output vs raycaster GT.

        Args:
            yolo_detections: list of B dicts from YoloBatchInference.detect_batch().
            raycaster_bboxes_xywh: (B, T, 4) — center_x, center_y, w, h in pixels.
                For iris_ma6, T=1 (single target).
            raycaster_confidence: (B, T) — raycaster visibility confidence [0,1].
            camera_to_target_dist: (B, T) — optional distance in meters.
        """
        B = raycaster_bboxes_xywh.shape[0]
        T = raycaster_bboxes_xywh.shape[1]
        rc_bboxes = raycaster_bboxes_xywh.cpu()
        rc_conf = raycaster_confidence.cpu()
        dist = camera_to_target_dist.cpu() if camera_to_target_dist is not None else None

        for b in range(B):
            for t in range(T):
                gt_bbox = rc_bboxes[b, t]  # (4,) — cx, cy, w, h
                gt_conf = rc_conf[b, t].item()
                gt_size = (gt_bbox[2] * gt_bbox[3]).sqrt().item()
                gt_visible = gt_conf >= 0.5 and gt_size >= 1.0

                # Skip if raycaster says target not visible
                if not gt_visible:
                    # Still record any YOLO detections as FPs (no valid target)
                    yolo = yolo_detections[b]
                    if yolo["bboxes_xywh"].shape[0] > 0:
                        for k in range(yolo["bboxes_xywh"].shape[0]):
                            fp_bbox = yolo["bboxes_xywh"][k]
                            self.false_positives.append(FalsePositiveSample(
                                bbox_cx=fp_bbox[0].item(), bbox_cy=fp_bbox[1].item(),
                                bbox_w=fp_bbox[2].item(), bbox_h=fp_bbox[3].item(),
                                confidence=yolo["confidences"][k].item(),
                                gt_target_size_px=0.0, gt_target_visible=False,
                            ))
                        self.fp_count += yolo["bboxes_xywh"].shape[0]
                    continue

                self.frame_count += 1
                distance = dist[b, t].item() if dist is not None else -1.0

                # Find best IoU match in YOLO detections for this image
                yolo = yolo_detections[b]
                matched = False
                best_iou = 0.0
                best_idx = -1

                if yolo["bboxes_xywh"].shape[0] > 0:
                    ious = self._compute_iou_xywh(gt_bbox.unsqueeze(0), yolo["bboxes_xywh"])
                    best_iou, best_idx = ious.max(dim=-1)
                    best_iou = best_iou.item()
                    best_idx = best_idx.item()

                if best_iou >= self.iou_match_threshold:
                    matched = True
                    yolo_bbox = yolo["bboxes_xywh"][best_idx]
                    yolo_conf = yolo["confidences"][best_idx].item()

                    self.matches.append(ComparisonSample(
                        target_size_px=gt_size,
                        center_error_x=(yolo_bbox[0] - gt_bbox[0]).item(),
                        center_error_y=(yolo_bbox[1] - gt_bbox[1]).item(),
                        size_error_w=(yolo_bbox[2] - gt_bbox[2]).item(),
                        size_error_h=(yolo_bbox[3] - gt_bbox[3]).item(),
                        yolo_confidence=yolo_conf,
                        distance_m=distance,
                    ))

                    # Remaining unmatched YOLO detections → FP
                    for k in range(yolo["bboxes_xywh"].shape[0]):
                        if k == best_idx:
                            continue
                        fp_bbox = yolo["bboxes_xywh"][k]
                        self.false_positives.append(FalsePositiveSample(
                            bbox_cx=fp_bbox[0].item(), bbox_cy=fp_bbox[1].item(),
                            bbox_w=fp_bbox[2].item(), bbox_h=fp_bbox[3].item(),
                            confidence=yolo["confidences"][k].item(),
                            gt_target_size_px=gt_size, gt_target_visible=True,
                        ))
                    self.fp_count += max(0, yolo["bboxes_xywh"].shape[0] - 1)
                else:
                    # YOLO missed the target — all detections are FP
                    for k in range(yolo["bboxes_xywh"].shape[0]):
                        fp_bbox = yolo["bboxes_xywh"][k]
                        self.false_positives.append(FalsePositiveSample(
                            bbox_cx=fp_bbox[0].item(), bbox_cy=fp_bbox[1].item(),
                            bbox_w=fp_bbox[2].item(), bbox_h=fp_bbox[3].item(),
                            confidence=yolo["confidences"][k].item(),
                            gt_target_size_px=gt_size, gt_target_visible=True,
                        ))
                    self.fp_count += yolo["bboxes_xywh"].shape[0]

                self.miss_data.append((gt_size, not matched))

        return len(self.matches)

    def fit(self, fit_size_range: tuple = (20.0, 80.0)) -> NoiseModelParams:
        """Fit parametric noise model from accumulated data.

        Args:
            fit_size_range: (min_px, max_px) — only fit noise/miss curves for
                targets in this size range. Small targets outside detectors's
                range and large targets with trivial noise are excluded.
        """
        from scipy.stats import t as student_t

        fit_min, fit_max = fit_size_range
        params = NoiseModelParams(
            num_frames=self.frame_count,
            num_matches=len(self.matches),
            num_misses=sum(1 for _, missed in self.miss_data if missed),
            num_false_positives=self.fp_count,
            fit_size_range=fit_size_range,
        )

        if len(self.matches) < 10:
            print(f"[WARN] Only {len(self.matches)} matches — using defaults")
            return params

        sizes = np.array([m.target_size_px for m in self.matches])
        cx_err = np.array([m.center_error_x for m in self.matches])
        cy_err = np.array([m.center_error_y for m in self.matches])
        w_err = np.array([m.size_error_w for m in self.matches])
        h_err = np.array([m.size_error_h for m in self.matches])

        # Bias (all sizes)
        params.center_bias_x = float(np.median(cx_err))
        params.center_bias_y = float(np.median(cy_err))

        # Filter to fit regime
        fit_mask = (sizes >= fit_min) & (sizes <= fit_max)
        if fit_mask.sum() < 10:
            print(f"[WARN] Only {fit_mask.sum()} samples in [{fit_min}, {fit_max}]px — using all")
            fit_mask = np.ones(len(sizes), dtype=bool)

        s_fit = sizes[fit_mask]
        cx_fit = cx_err[fit_mask]
        cy_fit = cy_err[fit_mask]
        w_fit = w_err[fit_mask]
        h_fit = h_err[fit_mask]

        print(f"[CALIB] Fitting noise on {fit_mask.sum()}/{len(sizes)} samples "
              f"with target_size in [{fit_min}, {fit_max}]px")

        # -- Fit Student's t df from center error (rounded peak + heavy tails) --
        center_err_combined = np.concatenate([cx_fit - params.center_bias_x,
                                               cy_fit - params.center_bias_y])
        try:
            df_center, _, _ = student_t.fit(center_err_combined, floc=0)
            params.center_noise_df = float(np.clip(df_center, 1.5, 100))
        except Exception:
            params.center_noise_df = 5.0
        print(f"[CALIB] Center noise df = {params.center_noise_df:.1f}")

        size_err_combined = np.concatenate([w_fit, h_fit])
        try:
            df_size, _, _ = student_t.fit(size_err_combined, floc=0)
            params.size_noise_df = float(np.clip(df_size, 1.5, 100))
        except Exception:
            params.size_noise_df = 5.0
        print(f"[CALIB] Size noise df = {params.size_noise_df:.1f}")

        # -- Fit scale = a / size + b (using IQR-based robust scale) --
        center_a, center_b = self._fit_noise_vs_size(s_fit, cx_fit, cy_fit)
        params.center_noise_a = center_a
        params.center_noise_b = center_b

        size_a, size_b = self._fit_noise_vs_size(s_fit, w_fit, h_fit)
        params.size_noise_a = size_a
        params.size_noise_b = size_b

        # -- Miss rate: fit sigmoid (fit range only) --
        miss_sizes = np.array([s for s, _ in self.miss_data])
        miss_flags = np.array([float(m) for _, m in self.miss_data])
        miss_fit = (miss_sizes >= fit_min) & (miss_sizes <= fit_max)

        if miss_fit.sum() > 10 and miss_flags[miss_fit].sum() > 5:
            a, threshold = self._fit_miss_sigmoid(miss_sizes[miss_fit], miss_flags[miss_fit])
            params.miss_sigmoid_a = a
            params.miss_size_threshold_px = threshold
        elif miss_flags.sum() > 5:
            a, threshold = self._fit_miss_sigmoid(miss_sizes, miss_flags)
            params.miss_sigmoid_a = a
            params.miss_size_threshold_px = threshold
        else:
            params.miss_sigmoid_a = 0.0
            params.miss_size_threshold_px = 0.0

        # -- FP rate --
        params.fp_rate = self.fp_count / max(self.frame_count, 1)

        return params

    def _fit_noise_vs_size(self, sizes: np.ndarray,
                           err_x: np.ndarray, err_y: np.ndarray,
                           n_bins: int = 10) -> tuple[float, float]:
        """Fit 1D scale = a / size + b by binned regression with curve_fit.

        Computes per-axis IQR/1.35 (robust scale estimate for Student's t/Gaussian),
        averages X and Y axis scales per bin, then fits a/size + b.
        """
        from scipy.optimize import curve_fit

        def _robust_scale_1d(data):
            """IQR-based scale on 1D data: IQR / 1.35 ≈ std for Gaussian."""
            q75, q25 = np.percentile(data, [75, 25])
            return (q75 - q25) / 1.35

        size_min, size_max = max(sizes.min(), 1.0), sizes.max()
        if size_max <= size_min:
            avg_scale = (_robust_scale_1d(err_x) + _robust_scale_1d(err_y)) / 2
            return 0.0, float(avg_scale)

        bin_edges = np.logspace(np.log10(size_min), np.log10(size_max), n_bins + 1)
        bin_centers = []
        bin_scales = []
        bin_counts = []

        for i in range(n_bins):
            mask = (sizes >= bin_edges[i]) & (sizes < bin_edges[i + 1])
            if mask.sum() < 5:
                continue
            bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
            sx = _robust_scale_1d(err_x[mask])
            sy = _robust_scale_1d(err_y[mask])
            bin_scales.append((sx + sy) / 2)
            bin_counts.append(mask.sum())

        if len(bin_centers) < 2:
            avg_scale = (_robust_scale_1d(err_x) + _robust_scale_1d(err_y)) / 2
            return 0.0, float(avg_scale)

        bc = np.array(bin_centers)
        bs = np.array(bin_scales)
        weights = np.sqrt(np.array(bin_counts))

        # Fit: scale = a / size + b via curve_fit (bounded, weighted)
        def model(size, a, b):
            return a / size + b

        try:
            popt, _ = curve_fit(
                model, bc, bs, p0=[100.0, bs.min()],
                sigma=1.0 / weights, absolute_sigma=False,
                bounds=([0.0, 0.0], [1e5, 1e3]),
                maxfev=5000,
            )
            a, b = popt
        except (RuntimeError, ValueError):
            # Fallback: constant scale
            a, b = 0.0, float(np.median(bs))

        return float(a), float(b)

    def _fit_miss_sigmoid(self, sizes: np.ndarray, missed: np.ndarray,
                          n_bins: int = 15) -> tuple[float, float]:
        """Fit p_miss = sigmoid(a * (threshold - size)) via scipy curve_fit on binned data."""
        from scipy.optimize import curve_fit

        size_min, size_max = max(sizes.min(), 1.0), sizes.max()
        if size_max <= size_min:
            return 0.1, 20.0

        bin_edges = np.logspace(np.log10(size_min), np.log10(size_max), n_bins + 1)
        bin_centers = []
        bin_miss_rates = []
        bin_counts = []

        for i in range(n_bins):
            mask = (sizes >= bin_edges[i]) & (sizes < bin_edges[i + 1])
            if mask.sum() < 3:
                continue
            bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
            bin_miss_rates.append(missed[mask].mean())
            bin_counts.append(mask.sum())

        if len(bin_centers) < 3:
            return 0.1, 20.0

        bc = np.array(bin_centers)
        mr = np.array(bin_miss_rates)
        weights = np.sqrt(np.array(bin_counts))  # weight by sqrt(count)

        # Sigmoid model: p_miss = sigmoid(a * (threshold - size))
        def sigmoid_model(size, a, threshold):
            return 1.0 / (1.0 + np.exp(-a * (threshold - size)))

        # Initial guess: threshold at size where miss rate ≈ 0.5
        sorted_idx = np.argsort(bc)
        bc_s, mr_s = bc[sorted_idx], mr[sorted_idx]

        # Find approximate 50% crossing
        init_thr = bc_s[-1]  # default: largest bin
        for i in range(len(bc_s) - 1):
            if mr_s[i] >= 0.5 and mr_s[i + 1] < 0.5:
                init_thr = (bc_s[i] + bc_s[i + 1]) / 2
                break
        # If miss rate never drops below 50%, threshold is beyond our range
        if mr_s[-1] >= 0.5:
            init_thr = bc_s[-1] * 1.5

        try:
            popt, _ = curve_fit(
                sigmoid_model, bc, mr, p0=[0.1, init_thr],
                sigma=1.0 / weights, absolute_sigma=False,
                bounds=([0.001, 0.0], [1.0, size_max * 3]),
                maxfev=5000,
            )
            a, threshold = popt
        except (RuntimeError, ValueError):
            # Fallback: use the initial estimates
            a = 4.0 / max(bc_s[-1] - bc_s[0], 1.0)
            threshold = init_thr

        return float(a), float(threshold)

    def generate_report(self, params: NoiseModelParams, output_dir: Path):
        """Generate histogram + PDF comparison plots.

        Uses Student's t distribution (Gaussian-like rounded peak + heavy tails,
        controlled by df parameter).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import laplace, norm, t as student_t

        if len(self.matches) < 10:
            print("[WARN] Too few matches for plots — skipping report generation")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        sizes = np.array([m.target_size_px for m in self.matches])
        cx_err = np.array([m.center_error_x for m in self.matches])
        cy_err = np.array([m.center_error_y for m in self.matches])
        w_err = np.array([m.size_error_w for m in self.matches])
        h_err = np.array([m.size_error_h for m in self.matches])
        center_err = np.sqrt(cx_err**2 + cy_err**2)
        confs = np.array([m.yolo_confidence for m in self.matches])
        dists = np.array([m.distance_m for m in self.matches])

        miss_sizes = np.array([s for s, _ in self.miss_data])
        miss_flags = np.array([float(m) for _, m in self.miss_data])

        df_c = params.center_noise_df

        # ------------------------------------------------------------------ #
        #  Figure 1: Center error — histogram + Student's t + Gaussian + Laplace
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("YOLO vs Raycaster — Bbox Noise Calibration Report", fontsize=14, y=0.98)

        # 1a: Center error X
        ax = axes[0, 0]
        ax.hist(cx_err, bins=60, density=True, alpha=0.6, color="steelblue", label="Data")
        x_range = np.linspace(cx_err.min() - 5, cx_err.max() + 5, 300)
        # Fit Student's t to this axis
        df_x, loc_x, scale_x = student_t.fit(cx_err)
        ax.plot(x_range, student_t.pdf(x_range, df_x, loc=loc_x, scale=scale_x),
                "r-", lw=2, label=f"t(df={df_x:.1f}, {loc_x:.1f}, {scale_x:.1f})")
        ax.plot(x_range, norm.pdf(x_range, loc=np.mean(cx_err), scale=np.std(cx_err)),
                "g--", lw=1.5, alpha=0.5, label="Gaussian")
        ax.plot(x_range, laplace.pdf(x_range, loc=np.median(cx_err),
                scale=np.median(np.abs(cx_err - np.median(cx_err)))),
                "m:", lw=1.5, alpha=0.5, label="Laplace")
        ax.set_xlabel("Center X error (px)")
        ax.set_ylabel("Density")
        ax.set_title("Center X Error Distribution")
        ax.legend(fontsize=8)
        ax.axvline(0, color="gray", ls="--", alpha=0.5)

        # 1b: Center error Y
        ax = axes[0, 1]
        ax.hist(cy_err, bins=60, density=True, alpha=0.6, color="steelblue", label="Data")
        y_range = np.linspace(cy_err.min() - 5, cy_err.max() + 5, 300)
        df_y, loc_y, scale_y = student_t.fit(cy_err)
        ax.plot(y_range, student_t.pdf(y_range, df_y, loc=loc_y, scale=scale_y),
                "r-", lw=2, label=f"t(df={df_y:.1f}, {loc_y:.1f}, {scale_y:.1f})")
        ax.plot(y_range, norm.pdf(y_range, loc=np.mean(cy_err), scale=np.std(cy_err)),
                "g--", lw=1.5, alpha=0.5, label="Gaussian")
        ax.plot(y_range, laplace.pdf(y_range, loc=np.median(cy_err),
                scale=np.median(np.abs(cy_err - np.median(cy_err)))),
                "m:", lw=1.5, alpha=0.5, label="Laplace")
        ax.set_xlabel("Center Y error (px)")
        ax.set_title("Center Y Error Distribution")
        ax.legend(fontsize=8)
        ax.axvline(0, color="gray", ls="--", alpha=0.5)

        # 1c: Euclidean center error histogram
        ax = axes[0, 2]
        ax.hist(center_err, bins=60, density=True, alpha=0.6, color="steelblue", label="Data")
        ax.set_xlabel("Center Euclidean error (px)")
        ax.set_title("Center Error Magnitude")
        ax.axvline(np.median(center_err), color="orange", ls="--", lw=2,
                   label=f"Median={np.median(center_err):.1f}px")
        ax.legend()

        # ------------------------------------------------------------------ #
        #  Figure 1 bottom row: size-dependent noise + miss rate + confidence
        #  X-axis limited to fit_size_range
        # ------------------------------------------------------------------ #
        fit_min, fit_max = params.fit_size_range
        n_bins = 15

        # 2a: Noise scale vs target size — binned data + fitted curve
        ax = axes[1, 0]
        size_min, size_max = max(sizes.min(), 1.0), sizes.max()
        if size_max > size_min:
            bin_edges = np.logspace(np.log10(max(size_min, fit_min * 0.5)),
                                   np.log10(min(size_max, fit_max * 1.5)), n_bins + 1)
            bc, bscale, bcount = [], [], []
            for i in range(n_bins):
                mask = (sizes >= bin_edges[i]) & (sizes < bin_edges[i + 1])
                if mask.sum() >= 5:
                    bc.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
                    # Per-axis 1D IQR/1.35 scale (matches fitting method)
                    def _iqr_scale(d):
                        q75, q25 = np.percentile(d, [75, 25])
                        return (q75 - q25) / 1.35
                    sx = _iqr_scale(cx_err[mask])
                    sy = _iqr_scale(cy_err[mask])
                    bscale.append((sx + sy) / 2)
                    bcount.append(mask.sum())
            if bc:
                bc, bscale, bcount = np.array(bc), np.array(bscale), np.array(bcount)
                ax.scatter(bc, bscale, s=np.sqrt(bcount) * 3, c="steelblue", alpha=0.7,
                          label="Binned MAD (size=count)")
                # Fitted curve (only over fit regime)
                s_range = np.linspace(max(fit_min * 0.5, 3), fit_max * 1.5, 200)
                fitted_scale = params.center_noise_a / s_range + params.center_noise_b
                ax.plot(s_range, fitted_scale, "r-", lw=2,
                       label=f"Fit: {params.center_noise_a:.1f}/s + {params.center_noise_b:.1f}")
                ax.axvline(fit_min, color="gray", ls=":", alpha=0.7)
                ax.axvline(fit_max, color="gray", ls=":", alpha=0.7, label=f"Fit range [{fit_min:.0f}, {fit_max:.0f}]px")
                ax.set_xscale("log")
                ax.set_xlim(left=max(fit_min * 0.5, 1), right=fit_max * 1.5)
        ax.set_xlabel("Target size (px, sqrt(w*h))")
        ax.set_ylabel("Center error scale (IQR/1.35, px)")
        ax.set_title(f"Noise vs Target Size [{fit_min:.0f}, {fit_max:.0f}]px")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2b: Miss rate vs target size — binned data + fitted sigmoid
        ax = axes[1, 1]
        if len(miss_sizes) > 20 and miss_flags.sum() > 0:
            ms_min, ms_max = max(miss_sizes.min(), 1.0), miss_sizes.max()
            if ms_max > ms_min:
                mb_edges = np.logspace(np.log10(max(ms_min, fit_min * 0.5)),
                                      np.log10(min(ms_max, fit_max * 1.5)), n_bins + 1)
                mbc, mbr, mbn = [], [], []
                for i in range(n_bins):
                    mask = (miss_sizes >= mb_edges[i]) & (miss_sizes < mb_edges[i + 1])
                    if mask.sum() >= 3:
                        mbc.append(np.sqrt(mb_edges[i] * mb_edges[i + 1]))
                        mbr.append(miss_flags[mask].mean())
                        mbn.append(mask.sum())
                if mbc:
                    mbc, mbr, mbn = np.array(mbc), np.array(mbr), np.array(mbn)
                    ax.scatter(mbc, mbr, s=np.sqrt(mbn) * 3, c="darkorange", alpha=0.7,
                              label="Binned miss rate")
                    # Fitted sigmoid
                    s_range = np.linspace(max(fit_min * 0.5, 1), fit_max * 1.5, 200)
                    fitted_miss = 1.0 / (1.0 + np.exp(-params.miss_sigmoid_a *
                                  (params.miss_size_threshold_px - s_range)))
                    ax.plot(s_range, fitted_miss, "r-", lw=2,
                           label=f"Fit: a={params.miss_sigmoid_a:.2f}, "
                                 f"thr={params.miss_size_threshold_px:.0f}px")
                    ax.axvline(fit_min, color="gray", ls=":", alpha=0.7)
                    ax.axvline(fit_max, color="gray", ls=":", alpha=0.7, label=f"Fit range [{fit_min:.0f}, {fit_max:.0f}]px")
                    ax.set_xscale("log")
                    ax.set_xlim(left=max(fit_min * 0.5, 1), right=fit_max * 1.5)
        ax.set_xlabel("Target size (px)")
        ax.set_ylabel("Miss rate")
        ax.set_title(f"Miss Rate vs Target Size [{fit_min:.0f}, {fit_max:.0f}]px")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2c: YOLO confidence distribution
        ax = axes[1, 2]
        ax.hist(confs, bins=40, density=True, alpha=0.6, color="seagreen", label="Matched conf")
        ax.set_xlabel("YOLO confidence")
        ax.set_ylabel("Density")
        ax.set_title("Detection Confidence Distribution")
        ax.axvline(np.median(confs), color="orange", ls="--", lw=2,
                   label=f"Median={np.median(confs):.2f}")
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = output_dir / "calibration_report.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[CALIB] Report saved to {fig_path}")

        # ------------------------------------------------------------------ #
        #  Figure 2: Size error + distance scatter (with Laplace fit)
        # ------------------------------------------------------------------ #
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        fig2.suptitle("YOLO vs Raycaster — Size Error & Distance Analysis", fontsize=14)

        # Size W error histogram + Student's t + Gaussian + Laplace
        ax = axes2[0]
        ax.hist(w_err, bins=60, density=True, alpha=0.6, color="mediumpurple", label="Data")
        wr = np.linspace(w_err.min() - 5, w_err.max() + 5, 300)
        df_w, loc_w, scale_w = student_t.fit(w_err)
        ax.plot(wr, student_t.pdf(wr, df_w, loc=loc_w, scale=scale_w),
                "r-", lw=2, label=f"t(df={df_w:.1f}, {loc_w:.1f}, {scale_w:.1f})")
        ax.plot(wr, norm.pdf(wr, loc=np.mean(w_err), scale=np.std(w_err)),
                "g--", lw=1.5, alpha=0.5, label="Gaussian")
        ax.plot(wr, laplace.pdf(wr, loc=np.median(w_err),
                scale=np.median(np.abs(w_err - np.median(w_err)))),
                "m:", lw=1.5, alpha=0.5, label="Laplace")
        ax.set_xlabel("Width error (px)")
        ax.set_title("Bbox Width Error")
        ax.legend(fontsize=8)

        # Size H error histogram + Student's t + Gaussian + Laplace
        ax = axes2[1]
        ax.hist(h_err, bins=60, density=True, alpha=0.6, color="mediumpurple", label="Data")
        hr = np.linspace(h_err.min() - 5, h_err.max() + 5, 300)
        df_h, loc_h, scale_h = student_t.fit(h_err)
        ax.plot(hr, student_t.pdf(hr, df_h, loc=loc_h, scale=scale_h),
                "r-", lw=2, label=f"t(df={df_h:.1f}, {loc_h:.1f}, {scale_h:.1f})")
        ax.plot(hr, norm.pdf(hr, loc=np.mean(h_err), scale=np.std(h_err)),
                "g--", lw=1.5, alpha=0.5, label="Gaussian")
        ax.plot(hr, laplace.pdf(hr, loc=np.median(h_err),
                scale=np.median(np.abs(h_err - np.median(h_err)))),
                "m:", lw=1.5, alpha=0.5, label="Laplace")
        ax.set_xlabel("Height error (px)")
        ax.set_title("Bbox Height Error")
        ax.legend(fontsize=8)

        # Center error vs distance scatter
        ax = axes2[2]
        valid_dist = dists > 0
        if valid_dist.sum() > 10:
            sc = ax.scatter(dists[valid_dist], center_err[valid_dist],
                           c=sizes[valid_dist], cmap="viridis", s=8, alpha=0.3)
            plt.colorbar(sc, ax=ax, label="Target size (px)")
            ax.set_xlabel("Camera-to-target distance (m)")
            ax.set_ylabel("Center error (px)")
            ax.set_title("Error vs Distance (color=size)")
        else:
            ax.text(0.5, 0.5, "No distance data", transform=ax.transAxes, ha="center")

        plt.tight_layout()
        fig2_path = output_dir / "calibration_report_size_dist.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)
        print(f"[CALIB] Size/distance report saved to {fig2_path}")

        # ------------------------------------------------------------------ #
        #  Figure 3: Raycaster + fitted noise vs YOLO — overlay histograms
        #  Uses Laplace sampling for synthetic data to match tails correctly
        # ------------------------------------------------------------------ #
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        fig3.suptitle(f"Validation: Raycaster + Fitted Noise vs YOLO (Student's t, df={params.center_noise_df:.1f})", fontsize=14)

        # Generate synthetic samples from fitted Student's t model
        syn_noise_scale = params.center_noise_a / np.clip(sizes, 5, None) + params.center_noise_b
        syn_cx_err = student_t.rvs(params.center_noise_df, loc=params.center_bias_x,
                                    scale=syn_noise_scale, random_state=42)
        syn_cy_err = student_t.rvs(params.center_noise_df, loc=params.center_bias_y,
                                    scale=syn_noise_scale, random_state=43)
        syn_center_err = np.sqrt(syn_cx_err**2 + syn_cy_err**2)

        # Use shared axis range based on data (clip synthetic outliers to data range)
        def _shared_bins(real, syn, n=60, pad=2):
            lo = min(np.percentile(real, 0.5), np.percentile(syn, 0.5)) - pad
            hi = max(np.percentile(real, 99.5), np.percentile(syn, 99.5)) + pad
            return np.linspace(lo, hi, n)

        # 3a: Center X overlay
        ax = axes3[0]
        bins = _shared_bins(cx_err, syn_cx_err)
        ax.hist(cx_err, bins=bins, density=True, alpha=0.5, color="steelblue", label="YOLO actual")
        ax.hist(syn_cx_err, bins=bins, density=True, alpha=0.5, color="coral", label="t-dist model")
        ax.set_xlabel("Center X error (px)")
        ax.set_ylabel("Density")
        ax.set_title("Center X: YOLO vs Fitted Model")
        ax.legend()

        # 3b: Center Y overlay
        ax = axes3[1]
        bins = _shared_bins(cy_err, syn_cy_err)
        ax.hist(cy_err, bins=bins, density=True, alpha=0.5, color="steelblue", label="YOLO actual")
        ax.hist(syn_cy_err, bins=bins, density=True, alpha=0.5, color="coral", label="t-dist model")
        ax.set_xlabel("Center Y error (px)")
        ax.set_title("Center Y: YOLO vs Fitted Model")
        ax.legend()

        # 3c: Euclidean center error overlay
        ax = axes3[2]
        bins = _shared_bins(center_err, syn_center_err, pad=1)
        bins = bins[bins >= 0]  # Euclidean is non-negative
        ax.hist(center_err, bins=bins, density=True, alpha=0.5, color="steelblue", label="YOLO actual")
        ax.hist(syn_center_err, bins=bins, density=True, alpha=0.5, color="coral", label="t-dist model")
        ax.set_xlabel("Center Euclidean error (px)")
        ax.set_title("Center Error: YOLO vs Fitted Model")
        ax.legend()

        plt.tight_layout()
        fig3_path = output_dir / "calibration_validation.png"
        fig3.savefig(fig3_path, dpi=150)
        plt.close(fig3)
        print(f"[CALIB] Validation overlay saved to {fig3_path}")

        # ------------------------------------------------------------------ #
        #  Figure 4: False positive characteristics
        # ------------------------------------------------------------------ #
        if len(self.false_positives) > 10:
            fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
            fig4.suptitle(f"False Positive Analysis ({len(self.false_positives)} FP detections)", fontsize=14)

            fp_cx = np.array([fp.bbox_cx for fp in self.false_positives])
            fp_cy = np.array([fp.bbox_cy for fp in self.false_positives])
            fp_w = np.array([fp.bbox_w for fp in self.false_positives])
            fp_h = np.array([fp.bbox_h for fp in self.false_positives])
            fp_sizes = np.sqrt(fp_w * fp_h)
            fp_confs = np.array([fp.confidence for fp in self.false_positives])
            fp_gt_vis = np.array([fp.gt_target_visible for fp in self.false_positives])

            # 4a: FP location heatmap (normalized to image coords)
            ax = axes4[0, 0]
            # Assume 640×480 image
            ax.hist2d(fp_cx, fp_cy, bins=30, cmap="YlOrRd")
            ax.set_xlabel("FP center X (px)")
            ax.set_ylabel("FP center Y (px)")
            ax.set_title("FP Spatial Distribution")
            ax.invert_yaxis()
            ax.set_aspect("equal")

            # 4b: FP bbox size distribution
            ax = axes4[0, 1]
            ax.hist(fp_sizes, bins=50, density=True, alpha=0.7, color="tomato")
            ax.set_xlabel("FP bbox size (px, sqrt(w*h))")
            ax.set_ylabel("Density")
            ax.set_title("FP Size Distribution")
            ax.axvline(np.median(fp_sizes), color="k", ls="--",
                       label=f"Median={np.median(fp_sizes):.0f}px")
            ax.legend()

            # 4c: FP confidence distribution
            ax = axes4[0, 2]
            ax.hist(fp_confs, bins=40, density=True, alpha=0.7, color="tomato",
                    label="FP confidence")
            if len(confs) > 0:
                ax.hist(confs, bins=40, density=True, alpha=0.5, color="seagreen",
                        label="TP confidence")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Density")
            ax.set_title("FP vs TP Confidence")
            ax.legend()

            # 4d: FP aspect ratio
            ax = axes4[1, 0]
            fp_aspect = fp_w / np.clip(fp_h, 1, None)
            ax.hist(fp_aspect, bins=50, density=True, alpha=0.7, color="tomato")
            ax.set_xlabel("FP aspect ratio (w/h)")
            ax.set_ylabel("Density")
            ax.set_title("FP Aspect Ratio")
            ax.axvline(1.0, color="gray", ls="--", alpha=0.5)
            ax.axvline(np.median(fp_aspect), color="k", ls="--",
                       label=f"Median={np.median(fp_aspect):.2f}")
            ax.legend()

            # 4e: FP size vs confidence scatter
            ax = axes4[1, 1]
            ax.scatter(fp_sizes, fp_confs, s=5, alpha=0.3, c="tomato")
            ax.set_xlabel("FP bbox size (px)")
            ax.set_ylabel("FP confidence")
            ax.set_title("FP Size vs Confidence")

            # 4f: FP rate by GT visibility
            ax = axes4[1, 2]
            n_vis = fp_gt_vis.sum()
            n_invis = len(fp_gt_vis) - n_vis
            bars = ax.bar(["Target visible", "Target not visible"],
                         [n_vis, n_invis], color=["steelblue", "tomato"])
            ax.set_ylabel("FP count")
            ax.set_title("FP by Target Visibility")
            for bar, val in zip(bars, [n_vis, n_invis]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       str(int(val)), ha="center", fontsize=10)

            plt.tight_layout()
            fig4_path = output_dir / "calibration_fp_analysis.png"
            fig4.savefig(fig4_path, dpi=150)
            plt.close(fig4)
            print(f"[CALIB] FP analysis saved to {fig4_path}")

    @staticmethod
    def _compute_iou_xywh(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between box1 (1,4) and box2 (K,4) in xywh format."""
        # Convert to xyxy
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2

        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        b1_area = box1[:, 2] * box1[:, 3]
        b2_area = box2[:, 2] * box2[:, 3]
        union = b1_area + b2_area - inter_area

        return inter_area / (union + 1e-8)


# --------------------------------------------------------------------------- #
#  Video annotation helpers
# --------------------------------------------------------------------------- #

def _draw_bbox_xywh(img: np.ndarray, cx, cy, w, h, color, thickness=2, label=None):
    """Draw a bbox (center x, center y, width, height) on an image."""
    import cv2
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _annotate_frame(image: torch.Tensor,
                    raycaster_bbox_xywh: torch.Tensor,
                    raycaster_conf: float,
                    yolo_detection: dict,
                    params,  # NoiseModelParams or None
                    step: int, env_idx: int, agent_id: str) -> np.ndarray:
    """Create an annotated frame comparing YOLO vs raycaster bboxes.

    Colors:
        Green  = Raycaster GT
        Red    = YOLO detection
        Cyan   = Modeled bbox (raycaster + fitted noise) [if params provided]
    """
    # Convert image to uint8 numpy BGR for cv2
    if image.dtype == torch.float32:
        frame = (image * 255).byte().cpu().numpy()
    else:
        frame = image.cpu().numpy().copy()
    # RGBA → RGB → BGR
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    frame = frame[..., ::-1].copy()  # RGB → BGR for cv2

    rc = raycaster_bbox_xywh.cpu()
    rc_size = (rc[2] * rc[3]).sqrt().item()

    # Draw raycaster GT (green)
    if raycaster_conf >= 0.5 and rc_size >= 1.0:
        _draw_bbox_xywh(frame, rc[0].item(), rc[1].item(), rc[2].item(), rc[3].item(),
                        color=(0, 255, 0), thickness=2,
                        label=f"RC {rc_size:.0f}px")

    # Draw modeled bbox (cyan) — raycaster + fitted noise
    if params is not None and raycaster_conf >= 0.5 and rc_size >= 1.0:
        from scipy.stats import t as student_t
        scale = params.center_noise_a / max(rc_size, 5.0) + params.center_noise_b
        noise_cx = student_t.rvs(params.center_noise_df, loc=params.center_bias_x, scale=scale)
        noise_cy = student_t.rvs(params.center_noise_df, loc=params.center_bias_y, scale=scale)
        s_scale = params.size_noise_a / max(rc_size, 5.0) + params.size_noise_b
        noise_w = student_t.rvs(params.size_noise_df, loc=0, scale=s_scale)
        noise_h = student_t.rvs(params.size_noise_df, loc=0, scale=s_scale)
        # Apply miss
        p_miss = 1.0 / (1.0 + np.exp(-params.miss_sigmoid_a *
                  (params.miss_size_threshold_px - rc_size)))
        if np.random.random() > p_miss:
            _draw_bbox_xywh(frame,
                            rc[0].item() + noise_cx, rc[1].item() + noise_cy,
                            max(rc[2].item() + noise_w, 1), max(rc[3].item() + noise_h, 1),
                            color=(255, 255, 0), thickness=2,  # cyan in BGR
                            label=f"Model")

    # Draw YOLO detections (red)
    yolo_bboxes = yolo_detection["bboxes_xywh"]
    yolo_confs = yolo_detection["confidences"]
    for k in range(yolo_bboxes.shape[0]):
        yb = yolo_bboxes[k]
        conf = yolo_confs[k].item()
        _draw_bbox_xywh(frame, yb[0].item(), yb[1].item(), yb[2].item(), yb[3].item(),
                        color=(0, 0, 255), thickness=2,
                        label=f"YOLO {conf:.2f}")

    # Step/env info
    import cv2
    cv2.putText(frame, f"Step {step} | Env {env_idx} | {agent_id}",
                (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def _write_comparison_video(video_frames: dict[str, list[np.ndarray]],
                            params: 'NoiseModelParams',
                            calibrator: 'DetectorCalibrator',
                            output_dir: Path,
                            fps: int = 25,
                            video_envs: int = 1):
    """Write side-by-side comparison video: YOLO vs raycaster vs modeled.

    Re-annotates stored frames with the fitted model overlay (cyan boxes).
    One video per agent.
    """
    import cv2

    for agent_id, frames in video_frames.items():
        if not frames:
            continue

        h, w = frames[0].shape[:2]
        video_path = output_dir / f"calibration_overlay_{agent_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        for frame in frames:
            writer.write(frame)

        writer.release()
        print(f"[CALIB] Video saved: {video_path} ({len(frames)} frames, {fps} fps)")


# --------------------------------------------------------------------------- #
#  Policy loader (inlined to avoid importing evaluate.py's module-level argparse)
# --------------------------------------------------------------------------- #

def _load_rnn_policy(checkpoint_path, env, agent_cfg, possible_agents):
    """Load a trained MAPPO-RNN policy from checkpoint."""
    import copy
    import os
    _skrl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "scripts", "reinforcement_learning", "skrl"))
    if _skrl_dir not in sys.path:
        sys.path.insert(0, _skrl_dir)
    from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    device = env.device
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    value_cfg = model_cfg.get("value", {})
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 32)

    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {aid: shared_space for aid in possible_agents}

    shared_policy = MAPPORNNPolicy(
        observation_space=env.observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=policy_cfg.get("hidden_size", 64),
        gru_num_layers=policy_cfg.get("gru_num_layers", 1),
        gru_hidden_size=policy_cfg.get("gru_hidden_size", 64),
        num_envs=env.num_envs,
        initial_log_std=policy_cfg.get("initial_log_std", -0.5),
        min_log_std=policy_cfg.get("min_log_std", -5.0),
        max_log_std=policy_cfg.get("max_log_std", 0.7),
        sequence_length=sequence_length,
    )
    shared_value = MAPPORNNValue(
        observation_space=shared_observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=value_cfg.get("hidden_size", 64),
        gru_num_layers=value_cfg.get("gru_num_layers", 1),
        gru_hidden_size=value_cfg.get("gru_hidden_size", 64),
        num_envs=env.num_envs,
        sequence_length=sequence_length,
    )
    models = {aid: {"policy": shared_policy, "value": shared_value} for aid in possible_agents}

    memories = {
        aid: RandomMemory(
            memory_size=agent_cfg.get("agent", {}).get("rollouts", 32),
            num_envs=env.num_envs,
            device=device,
        )
        for aid in possible_agents
    }

    # Setup preprocessors
    agent_cfg_copy = copy.deepcopy(agent_cfg)
    if "state_preprocessor" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["state_preprocessor"] == "RunningStandardScaler":
            agent_cfg_copy["agent"]["state_preprocessor"] = RunningStandardScaler
        if agent_cfg_copy["agent"].get("state_preprocessor_kwargs") is None:
            agent_cfg_copy["agent"]["state_preprocessor_kwargs"] = {}
        agent_cfg_copy["agent"]["state_preprocessor_kwargs"]["size"] = env.observation_spaces[possible_agents[0]]
        agent_cfg_copy["agent"]["state_preprocessor_kwargs"]["device"] = device
    if "shared_state_preprocessor" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["shared_state_preprocessor"] == "RunningStandardScaler":
            agent_cfg_copy["agent"]["shared_state_preprocessor"] = RunningStandardScaler
        if agent_cfg_copy["agent"].get("shared_state_preprocessor_kwargs") is None:
            agent_cfg_copy["agent"]["shared_state_preprocessor_kwargs"] = {}
        agent_cfg_copy["agent"]["shared_state_preprocessor_kwargs"]["size"] = shared_observation_spaces[possible_agents[0]]
        agent_cfg_copy["agent"]["shared_state_preprocessor_kwargs"]["device"] = device
    if "value_preprocessor" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["value_preprocessor"] == "RunningStandardScaler":
            agent_cfg_copy["agent"]["value_preprocessor"] = RunningStandardScaler
        if agent_cfg_copy["agent"].get("value_preprocessor_kwargs") is None:
            agent_cfg_copy["agent"]["value_preprocessor_kwargs"] = {}
        agent_cfg_copy["agent"]["value_preprocessor_kwargs"]["device"] = device
    if "learning_rate_scheduler" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["learning_rate_scheduler"] == "KLAdaptiveLR":
            agent_cfg_copy["agent"]["learning_rate_scheduler"] = KLAdaptiveLR

    mappo_cfg = copy.deepcopy(MAPPO_RNN_DEFAULT_CONFIG)
    mappo_cfg.update(agent_cfg_copy.get("agent", {}))

    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )

    agent.load(checkpoint_path)
    agent.set_mode("eval")

    # Freeze preprocessors
    for uid in possible_agents:
        for key in ("_state_preprocessor", "_shared_state_preprocessor", "_value_preprocessor"):
            pp = getattr(agent, key, None)
            if isinstance(pp, dict) and uid in pp and hasattr(pp[uid], "eval"):
                pp[uid].eval()

    return agent


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

@hydra_task_config(args_cli.task, "skrl_mappo_rnn_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Run calibration: collect YOLO vs raycaster data and fit noise model."""

    # --- Configure env for calibration ---
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.enable_tiled_cameras = True  # Force cameras ON
    env_cfg.episode_length_s = 30.0

    # Force all curriculum to full difficulty (realistic bbox sizes)
    for attr in dir(env_cfg.curriculum):
        if attr.endswith("_start_step") or attr.endswith("_end_step"):
            setattr(env_cfg.curriculum, attr, 0)

    # Apply experiment overrides if provided
    if args_cli.experiment is not None:
        exp_cfg = get_experiment(args_cli.experiment)
        if exp_cfg.env_overrides:
            apply_env_overrides(env_cfg, exp_cfg.env_overrides)
        print(f"[CALIB] Experiment: {exp_cfg.name}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    unwrapped = env.unwrapped

    num_agents = len(env_cfg.possible_agents)
    possible_agents = list(env_cfg.possible_agents)
    print(f"[CALIB] {args_cli.num_envs} envs, {num_agents} agents, "
          f"cameras: {env_cfg.camera.width}x{env_cfg.camera.height}")

    # --- Load policy (optional) ---
    policy = None
    if args_cli.checkpoint is not None:
        policy = _load_rnn_policy(
            args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents,
        )
        print(f"[CALIB] Loaded policy from {args_cli.checkpoint}")
    else:
        print(f"[CALIB] No checkpoint — using random actions")

    # --- Initialize YOLO ---
    yolo = YoloBatchInference(
        model_path=args_cli.yolo_model,
        device="cuda:0",
        conf_threshold=args_cli.yolo_conf,
    )
    print(f"[CALIB] YOLO model: {args_cli.yolo_model} (conf={args_cli.yolo_conf})")

    # --- Initialize calibrator ---
    calibrator = DetectorCalibrator(iou_match_threshold=0.1)

    # --- Video frame collection ---
    video_frames: dict[str, list[np.ndarray]] = {}  # agent_id → list of annotated frames
    record_video = args_cli.video_envs > 0
    if record_video:
        for agent_id in possible_agents:
            video_frames[agent_id] = []
        print(f"[CALIB] Recording video for {args_cli.video_envs} env(s)")

    # --- Collection loop ---
    obs, info = env_wrapped.reset()
    total_yolo_time = 0.0
    total_matches = 0

    print(f"[CALIB] Collecting {args_cli.num_steps} steps...")
    for step in range(args_cli.num_steps):
        # Compute actions
        if policy is not None:
            with torch.no_grad():
                actions, _, outputs = policy.act(obs, 0, 0)
                for agent_id in possible_agents:
                    actions[agent_id] = outputs[agent_id]["mean_actions"]
        else:
            actions = {
                agent_id: torch.zeros(args_cli.num_envs, 7, device=unwrapped.device)
                for agent_id in possible_agents
            }

        obs, rewards, terminated, truncated, info = env_wrapped.step(actions)

        # --- Extract camera images and raycaster GT ---
        for idx, agent_id in enumerate(possible_agents):
            if agent_id not in unwrapped._cameras:
                continue

            # Camera RGB: (N, H, W, 4) RGBA from TiledCamera
            camera_data = unwrapped._cameras[agent_id].data.output["rgb"]  # (N, H, W, 3)
            if camera_data is None:
                continue

            # Raycaster GT bboxes for this agent: (N, T, 4) in pixel xywh
            rc_bboxes = unwrapped.bbox_raycaster_v2.data.bboxes[:, idx, :, :]  # (N, T, 4)
            rc_confidence = unwrapped._smoothed_bbox_confidence[:, idx, :]  # (N, T)

            # Camera-to-target distance
            cam_pos = unwrapped.bbox_raycaster_v2.data.camera_pos_w[:, idx, :]  # (N, 3)
            tgt_pos = unwrapped.bbox_raycaster_v2.data.target_pos_w[:, 0, :]  # (N, 3)
            dist = (cam_pos - tgt_pos).norm(dim=-1, keepdim=True)  # (N, 1)

            # Run YOLO
            t0 = time.time()
            yolo_results = yolo.detect_batch(camera_data)
            total_yolo_time += time.time() - t0

            # Ingest into calibrator
            n_matches = calibrator.ingest(
                yolo_detections=yolo_results,
                raycaster_bboxes_xywh=rc_bboxes,
                raycaster_confidence=rc_confidence,
                camera_to_target_dist=dist,
            )
            total_matches += n_matches

            # --- Collect video frames for first N envs ---
            if record_video:
                for env_i in range(min(args_cli.video_envs, args_cli.num_envs)):
                    frame = _annotate_frame(
                        image=camera_data[env_i],
                        raycaster_bbox_xywh=rc_bboxes[env_i, 0],  # (4,) first target
                        raycaster_conf=rc_confidence[env_i, 0].item(),
                        yolo_detection=yolo_results[env_i],
                        params=None,  # filled after fitting
                        step=step,
                        env_idx=env_i,
                        agent_id=agent_id,
                    )
                    video_frames[agent_id].append(frame)

        # Progress
        if (step + 1) % 500 == 0:
            elapsed_yolo_ms = total_yolo_time * 1000
            avg_yolo_ms = elapsed_yolo_ms / max(step + 1, 1)
            miss_count = sum(1 for _, m in calibrator.miss_data if m)
            print(f"  Step {step+1}/{args_cli.num_steps}: "
                  f"{calibrator.frame_count} frames, "
                  f"{len(calibrator.matches)} matches, "
                  f"{miss_count} misses, "
                  f"{calibrator.fp_count} FPs, "
                  f"YOLO avg {avg_yolo_ms:.1f}ms/step")

    # --- Fit noise model ---
    print(f"\n[CALIB] Fitting noise model from {calibrator.frame_count} frames...")
    params = calibrator.fit(fit_size_range=(20.0, 80.0))
    params.yolo_model = args_cli.yolo_model
    params.yolo_conf_threshold = args_cli.yolo_conf

    # --- Print results ---
    print("\n" + "=" * 70)
    print("BBOX NOISE MODEL — FITTED PARAMETERS")
    print("=" * 70)
    print(f"  Frames analyzed:      {params.num_frames}")
    print(f"  Matched detections:   {params.num_matches}")
    print(f"  Missed detections:    {params.num_misses}")
    print(f"  False positives:      {params.num_false_positives}")
    print(f"  Overall miss rate:    {params.num_misses / max(params.num_frames, 1):.3f}")
    print(f"  Overall FP rate:      {params.fp_rate:.4f}")
    print()
    print(f"  Center noise:         std = {params.center_noise_a:.2f} / size_px + {params.center_noise_b:.2f}")
    print(f"  Size noise:           std = {params.size_noise_a:.2f} / size_px + {params.size_noise_b:.2f}")
    print(f"  Center bias:          ({params.center_bias_x:.2f}, {params.center_bias_y:.2f}) px")
    print(f"  Miss sigmoid:         a={params.miss_sigmoid_a:.3f}, threshold={params.miss_size_threshold_px:.1f} px")
    print("=" * 70)

    # --- FP summary ---
    if calibrator.false_positives:
        fp_confs = [fp.confidence for fp in calibrator.false_positives]
        fp_sizes = [(fp.bbox_w * fp.bbox_h) ** 0.5 for fp in calibrator.false_positives]
        print(f"\n  FP detections:        {len(calibrator.false_positives)}")
        print(f"  FP confidence:        median={np.median(fp_confs):.2f}, "
              f"mean={np.mean(fp_confs):.2f}")
        print(f"  FP bbox size:         median={np.median(fp_sizes):.0f}px, "
              f"mean={np.mean(fp_sizes):.0f}px")

    # --- Generate visual report ---
    output_path = Path(args_cli.output)
    report_dir = output_path.parent / (output_path.stem + "_report")
    calibrator.generate_report(params, report_dir)

    # --- Write comparison video ---
    if record_video and any(len(f) > 0 for f in video_frames.values()):
        # Re-run a second pass on the stored frames to add model overlay
        # (params weren't available during collection)
        print(f"\n[CALIB] Re-annotating {sum(len(f) for f in video_frames.values())} "
              f"frames with fitted model...")
        # The frames already have raycaster (green) and YOLO (red) annotations.
        # We now need to add the model (cyan) overlay. Since we stored the raw
        # annotated frames, we'll do a second collection pass to get model overlays.
        # For simplicity, we write the frames as-is (green=RC, red=YOLO) and
        # re-run a short collection with the fitted model for the overlay video.

        # Second pass: short run to collect model-annotated frames
        print("[CALIB] Collecting model-overlay frames (second pass, 100 steps)...")
        model_frames: dict[str, list[np.ndarray]] = {aid: [] for aid in possible_agents}
        obs2, _ = env_wrapped.reset()
        for step in range(min(100, args_cli.num_steps)):
            if policy is not None:
                with torch.no_grad():
                    actions, _, outputs = policy.act(obs2, 0, 0)
                    for agent_id in possible_agents:
                        actions[agent_id] = outputs[agent_id]["mean_actions"]
            else:
                actions = {
                    aid: torch.zeros(args_cli.num_envs, 7, device=unwrapped.device)
                    for aid in possible_agents
                }
            obs2, _, _, _, _ = env_wrapped.step(actions)

            for idx, agent_id in enumerate(possible_agents):
                if agent_id not in unwrapped._cameras:
                    continue
                camera_data = unwrapped._cameras[agent_id].data.output["rgb"]
                if camera_data is None:
                    continue
                rc_bboxes = unwrapped.bbox_raycaster_v2.data.bboxes[:, idx, :, :]
                rc_conf = unwrapped._smoothed_bbox_confidence[:, idx, :]
                yolo_results = yolo.detect_batch(camera_data)

                for env_i in range(min(args_cli.video_envs, args_cli.num_envs)):
                    frame = _annotate_frame(
                        image=camera_data[env_i],
                        raycaster_bbox_xywh=rc_bboxes[env_i, 0],
                        raycaster_conf=rc_conf[env_i, 0].item(),
                        yolo_detection=yolo_results[env_i],
                        params=params,  # NOW with fitted model
                        step=step, env_idx=env_i, agent_id=agent_id,
                    )
                    model_frames[agent_id].append(frame)

        _write_comparison_video(model_frames, params, calibrator, report_dir,
                                fps=args_cli.video_fps, video_envs=args_cli.video_envs)

    # --- Save ---
    with open(output_path, "w") as f:
        json.dump(asdict(params), f, indent=2)
    print(f"\n[CALIB] Saved to {output_path}")

    # Optionally save raw data
    if args_cli.save_raw:
        raw_path = output_path.with_suffix(".raw.json")
        raw_data = {
            "matches": [asdict(m) for m in calibrator.matches],
            "miss_data": [
                {"target_size_px": s, "missed": m}
                for s, m in calibrator.miss_data
            ],
            "false_positives": [asdict(fp) for fp in calibrator.false_positives],
        }
        with open(raw_path, "w") as f:
            json.dump(raw_data, f)
        print(f"[CALIB] Raw data saved to {raw_path} "
              f"({len(calibrator.false_positives)} FP samples)")

    # --- Timing summary ---
    total_steps = args_cli.num_steps
    print(f"\n[CALIB] YOLO total: {total_yolo_time:.1f}s, "
          f"avg: {total_yolo_time/total_steps*1000:.1f}ms/step "
          f"({args_cli.num_envs * num_agents} images/step)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
