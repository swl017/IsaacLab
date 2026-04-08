#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified detector calibration: localization noise + FP/FN + background clutter.

Runs with gimbal locked to target and zero ego velocity (hover).
Collects YOLO-vs-raycaster comparisons and produces:
  - Noise model (Student's t localization error vs target size)
  - FN model (miss rate vs target size, conditioned on sky/ground background)
  - FP characteristics (spatial distribution, size, confidence)
  - Comparison video overlay

Usage:
    # Full calibration with Flight scene
    python calibrate_detector.py \\
        --experiment a1_with_aoi \\
        --num_envs 64 --num_steps 500 \\
        --output detector_calib.json

    # Control run without background clutter
    python calibrate_detector.py \\
        --experiment a1_with_aoi \\
        --num_envs 64 --num_steps 500 --no_flight_scene \\
        --output detector_calib_ground.json

    # Different YOLO model
    python calibrate_detector.py \\
        --experiment a1_with_aoi \\
        --yolo_model /home/usrg/mas/resource/dronecop9-2.pt \\
        --output detector_calib_cop9.json
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="Unified detector calibration (noise + FP/FN + clutter).")
parser.add_argument("--experiment", type=str, default="a1_with_aoi",
                    help="Experiment name from registry (for env overrides)")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Number of parallel environments (all get cameras)")
parser.add_argument("--num_steps", type=int, default=500,
                    help="Number of policy steps to collect")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA6-Direct-Test-v0")
parser.add_argument("--yolo_model", type=str,
                    default="/home/usrg/mas/src/ultralytics_ros/models/yolov11m-drone.pt",
                    help="Path to YOLO detector model weights (.pt)")
parser.add_argument("--yolo_conf", type=float, default=0.25,
                    help="YOLO confidence threshold")
parser.add_argument("--output", type=str, default="detector_calib.json",
                    help="Output JSON path")
parser.add_argument("--no_flight_scene", action="store_true", default=False,
                    help="Disable Flight scene for control run (plain ground)")
parser.add_argument("--iou_threshold", type=float, default=0.1,
                    help="IoU threshold for matching YOLO to raycaster GT")
parser.add_argument("--fit_size_min", type=float, default=20.0,
                    help="Min target size (px) for noise fitting")
parser.add_argument("--fit_size_max", type=float, default=80.0,
                    help="Max target size (px) for noise fitting")
parser.add_argument("--agent_filter_dist", type=float, default=80.0,
                    help="Pixel distance threshold for other-agent detection filtering")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained RL policy checkpoint (.pt). Uses policy xyz actions, zeroes gimbal/yaw.")
parser.add_argument("--video_envs", type=int, default=1,
                    help="Number of envs to record video for (0 to disable)")
parser.add_argument("--video_fps", type=int, default=25,
                    help="Video output FPS")

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher

app_args = argparse.Namespace(
    headless=True, device="cuda:0", experience="", enable_cameras=True,
)
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is initialized."""

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.direct.iris_ma6.experiments.experiment_registry import get_experiment
from isaaclab_tasks.direct.iris_ma6.experiments.env_overrides import apply_env_overrides


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #

@dataclass
class DetectionSample:
    """Single frame: full detection comparison (noise + FN + FP + clutter)."""
    step: int
    env_idx: int
    agent_id: str
    # Target geometry
    target_size_px: float
    distance_m: float
    elevation_deg: float
    background_is_sky: bool
    # Raycaster GT
    raycaster_confidence: float
    gt_bbox_cx: float
    gt_bbox_cy: float
    gt_bbox_w: float
    gt_bbox_h: float
    # YOLO result
    yolo_detected: bool
    yolo_confidence: float
    yolo_bbox_iou: float
    # Localization error (only if detected)
    center_error_x: float
    center_error_y: float
    size_error_w: float
    size_error_h: float


@dataclass
class FalsePositiveSample:
    """Single FP detection (not matching target or other agents)."""
    step: int
    env_idx: int
    agent_id: str
    bbox_cx: float
    bbox_cy: float
    bbox_w: float
    bbox_h: float
    confidence: float
    gt_target_visible: bool
    gt_target_size_px: float
    background_is_sky: bool


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
        if images_nhwc.dtype == torch.float32:
            images_np = (images_nhwc * 255).byte().cpu().numpy()
        else:
            images_np = images_nhwc.cpu().numpy()
        if images_np.shape[-1] == 4:
            images_np = images_np[..., :3]
        image_list = [images_np[i] for i in range(images_np.shape[0])]
        results = self.model.predict(
            source=image_list, conf=self.conf_threshold,
            device=self.device, verbose=False, classes=[self.target_class],
        )
        detections = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                detections.append({
                    "bboxes_xywh": r.boxes.xywh.cpu(),
                    "confidences": r.boxes.conf.cpu(),
                    "classes": r.boxes.cls.cpu(),
                })
            else:
                detections.append({
                    "bboxes_xywh": torch.zeros(0, 4),
                    "confidences": torch.zeros(0),
                    "classes": torch.zeros(0),
                })
        return detections


# --------------------------------------------------------------------------- #
#  Utilities
# --------------------------------------------------------------------------- #

def _compute_iou_xywh(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """IoU between box1 (1,4) and box2 (K,4) in xywh format."""
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    inter_area = ((torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0) *
                  (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0))
    return inter_area / (box1[:, 2] * box1[:, 3] + box2[:, 2] * box2[:, 3] - inter_area + 1e-8)


def _compute_elevation_deg(cam_pos: torch.Tensor, tgt_pos: torch.Tensor) -> torch.Tensor:
    """Elevation angle (deg below horizontal). Positive = looking down."""
    delta = tgt_pos - cam_pos
    return torch.rad2deg(torch.atan2(-delta[:, 2], delta[:, :2].norm(dim=-1)))


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    if n_total == 0:
        return 0.0, 1.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _query_background_is_sky(cam_pos, tgt_pos, static_mesh, margin=2.0):
    """Cast ray through target — inf = sky, finite = ground/building."""
    if static_mesh is None:
        return torch.ones(cam_pos.shape[0], dtype=torch.bool, device=cam_pos.device)
    from isaaclab.utils.warp import raycast_mesh
    ray_dir = tgt_pos - cam_pos
    ray_dir_norm = ray_dir / ray_dir.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    ray_starts = tgt_pos + ray_dir_norm * margin
    _, ray_dist, _, _ = raycast_mesh(
        ray_starts.unsqueeze(1), ray_dir_norm.unsqueeze(1), static_mesh,
        max_dist=1e4, return_distance=True,
    )
    return torch.isinf(ray_dist.squeeze(1))


def _project_points_to_pixels(points_w, cam_pos_w, cam_quat_w, intrinsic):
    """Project (N, P, 3) world points to (N, P, 2) pixel coords."""
    from isaaclab_tasks.direct.iris_ma6.bbox_raycaster_v2.utils.projection import (
        batch_transform_to_camera_frame, batch_project_to_image_plane,
    )
    N, P = points_w.shape[:2]
    pts = points_w.unsqueeze(1).unsqueeze(3)
    pts_cam = batch_transform_to_camera_frame(
        pts, cam_pos_w.unsqueeze(1), cam_quat_w.unsqueeze(1))
    pixels, _, valid = batch_project_to_image_plane(pts_cam, intrinsic.unsqueeze(1))
    return pixels[:, 0, :, 0, :], valid[:, 0, :, 0]


def _filter_agent_detections(yolo_bboxes, agent_pixels, agent_valid, dist_threshold=80.0):
    """Return bool mask: True = NOT an other-agent detection (keep it)."""
    K = yolo_bboxes.shape[0]
    if K == 0 or agent_valid.sum() == 0:
        return torch.ones(K, dtype=torch.bool)
    dists = torch.cdist(yolo_bboxes[:, :2].float(), agent_pixels[agent_valid].float())
    return dists.min(dim=1).values > dist_threshold


# --------------------------------------------------------------------------- #
#  Unified Detector Collector
# --------------------------------------------------------------------------- #

class DetectorCollector:
    """Collects all detector characterization data in one pass."""

    def __init__(self, iou_threshold: float = 0.1, background_type: str = "flight_scene",
                 agent_filter_dist: float = 80.0):
        self.iou_threshold = iou_threshold
        self.background_type = background_type
        self.agent_filter_dist = agent_filter_dist
        self.detections: list[DetectionSample] = []
        self.false_positives: list[FalsePositiveSample] = []

    def ingest(
        self,
        yolo_detections: list[dict],
        rc_bboxes_xywh: torch.Tensor,     # (B, T, 4)
        rc_confidence: torch.Tensor,       # (B, T)
        cam_pos: torch.Tensor,             # (B, 3)
        tgt_pos: torch.Tensor,             # (B, 3)
        bg_is_sky: torch.Tensor,           # (B,) bool
        other_agent_pixels: torch.Tensor,  # (B, P, 2)
        other_agent_valid: torch.Tensor,   # (B, P) bool
        step: int,
        agent_id: str,
    ):
        B, T = rc_bboxes_xywh.shape[:2]
        rc_bbox = rc_bboxes_xywh.cpu()
        rc_conf = rc_confidence.cpu()
        dist = (cam_pos - tgt_pos).norm(dim=-1).cpu()
        elev = _compute_elevation_deg(cam_pos, tgt_pos).cpu()
        sky = bg_is_sky.cpu()
        oa_px = other_agent_pixels.cpu()
        oa_val = other_agent_valid.cpu()

        for b in range(B):
            gt = rc_bbox[b, 0]  # first target
            gt_conf = rc_conf[b, 0].item()
            gt_size = (gt[2] * gt[3]).sqrt().item()
            gt_visible = gt_conf >= 0.5 and gt_size >= 1.0

            yolo = yolo_detections[b]
            n_yolo = yolo["bboxes_xywh"].shape[0]

            # Filter out other-agent detections
            if n_yolo > 0:
                keep = _filter_agent_detections(
                    yolo["bboxes_xywh"], oa_px[b], oa_val[b], self.agent_filter_dist)
                filt_bboxes = yolo["bboxes_xywh"][keep]
                filt_confs = yolo["confidences"][keep]
            else:
                filt_bboxes = yolo["bboxes_xywh"]
                filt_confs = yolo["confidences"]

            if not gt_visible:
                # Target not visible — any remaining detection is FP
                for k in range(filt_bboxes.shape[0]):
                    fb = filt_bboxes[k]
                    self.false_positives.append(FalsePositiveSample(
                        step=step, env_idx=b, agent_id=agent_id,
                        bbox_cx=fb[0].item(), bbox_cy=fb[1].item(),
                        bbox_w=fb[2].item(), bbox_h=fb[3].item(),
                        confidence=filt_confs[k].item(),
                        gt_target_visible=False, gt_target_size_px=0.0,
                        background_is_sky=bool(sky[b].item()),
                    ))
                continue

            # Target visible — match YOLO to GT
            detected = False
            best_conf = 0.0
            best_iou = 0.0
            best_idx = -1
            cx_err = cy_err = w_err = h_err = 0.0

            if filt_bboxes.shape[0] > 0:
                ious = _compute_iou_xywh(gt.unsqueeze(0), filt_bboxes)
                max_iou, max_idx_t = ious.max(dim=-1)
                best_iou = max_iou.item()
                best_idx = max_idx_t.item()
                if best_iou >= self.iou_threshold:
                    detected = True
                    yb = filt_bboxes[best_idx]
                    best_conf = filt_confs[best_idx].item()
                    cx_err = (yb[0] - gt[0]).item()
                    cy_err = (yb[1] - gt[1]).item()
                    w_err = (yb[2] - gt[2]).item()
                    h_err = (yb[3] - gt[3]).item()

            self.detections.append(DetectionSample(
                step=step, env_idx=b, agent_id=agent_id,
                target_size_px=gt_size,
                distance_m=dist[b].item(),
                elevation_deg=elev[b].item(),
                background_is_sky=bool(sky[b].item()),
                raycaster_confidence=gt_conf,
                gt_bbox_cx=gt[0].item(), gt_bbox_cy=gt[1].item(),
                gt_bbox_w=gt[2].item(), gt_bbox_h=gt[3].item(),
                yolo_detected=detected,
                yolo_confidence=best_conf,
                yolo_bbox_iou=best_iou,
                center_error_x=cx_err, center_error_y=cy_err,
                size_error_w=w_err, size_error_h=h_err,
            ))

            # Remaining unmatched detections → FP
            for k in range(filt_bboxes.shape[0]):
                if k == best_idx and detected:
                    continue
                fb = filt_bboxes[k]
                self.false_positives.append(FalsePositiveSample(
                    step=step, env_idx=b, agent_id=agent_id,
                    bbox_cx=fb[0].item(), bbox_cy=fb[1].item(),
                    bbox_w=fb[2].item(), bbox_h=fb[3].item(),
                    confidence=filt_confs[k].item(),
                    gt_target_visible=True, gt_target_size_px=gt_size,
                    background_is_sky=bool(sky[b].item()),
                ))

    # --- Properties ---
    @property
    def matches(self):
        return [d for d in self.detections if d.yolo_detected]

    @property
    def misses(self):
        return [d for d in self.detections if not d.yolo_detected]

    @property
    def num_frames(self):
        return len(self.detections)

    @property
    def miss_rate(self):
        return len(self.misses) / max(self.num_frames, 1)

    @property
    def fp_rate(self):
        return len(self.false_positives) / max(self.num_frames, 1)

    # --- Noise model fitting ---
    def fit_noise_model(self, fit_size_range=(20.0, 80.0)):
        """Fit Student's t noise model from matched detections."""
        from scipy.stats import t as student_t
        from scipy.optimize import curve_fit

        matched = self.matches
        if len(matched) < 10:
            print(f"[WARN] Only {len(matched)} matches — using defaults")
            return {}

        sizes = np.array([m.target_size_px for m in matched])
        cx_err = np.array([m.center_error_x for m in matched])
        cy_err = np.array([m.center_error_y for m in matched])
        w_err = np.array([m.size_error_w for m in matched])
        h_err = np.array([m.size_error_h for m in matched])

        # Fit Student's t df
        center_combined = np.concatenate([cx_err - np.median(cx_err),
                                           cy_err - np.median(cy_err)])
        try:
            df_center, _, scale_center = student_t.fit(center_combined, floc=0)
            df_center = float(np.clip(df_center, 1.5, 100))
        except Exception:
            df_center = 5.0

        size_combined = np.concatenate([w_err, h_err])
        try:
            df_size, _, scale_size = student_t.fit(size_combined, floc=0)
            df_size = float(np.clip(df_size, 1.5, 100))
        except Exception:
            df_size = 5.0

        # Fit scale = a / size + b from 1D per-axis IQR
        fit_min, fit_max = fit_size_range
        fit_mask = (sizes >= fit_min) & (sizes <= fit_max)
        s_fit = sizes[fit_mask] if fit_mask.sum() > 10 else sizes
        cx_fit = cx_err[fit_mask] if fit_mask.sum() > 10 else cx_err
        cy_fit = cy_err[fit_mask] if fit_mask.sum() > 10 else cy_err
        w_fit = w_err[fit_mask] if fit_mask.sum() > 10 else w_err
        h_fit = h_err[fit_mask] if fit_mask.sum() > 10 else h_err

        def _iqr_scale(data):
            q75, q25 = np.percentile(data, [75, 25])
            return (q75 - q25) / 1.35

        def _fit_scale_vs_size(s, err_a, err_b):
            """Fit scale = a / size + b from two error axes."""
            n_bins = 10
            if s.max() > s.min():
                bin_edges = np.logspace(np.log10(max(s.min(), 1)), np.log10(s.max()), n_bins + 1)
                bc, bs = [], []
                for i in range(n_bins):
                    mask = (s >= bin_edges[i]) & (s < bin_edges[i + 1])
                    if mask.sum() >= 5:
                        bc.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
                        bs.append((_iqr_scale(err_a[mask]) + _iqr_scale(err_b[mask])) / 2)
                if len(bc) >= 2:
                    bc_arr, bs_arr = np.array(bc), np.array(bs)
                    try:
                        popt, _ = curve_fit(lambda s, a, b: a / s + b, bc_arr, bs_arr,
                                            p0=[100.0, bs_arr.min()],
                                            bounds=([0, 0], [1e5, 1e3]), maxfev=5000)
                        return float(popt[0]), float(popt[1])
                    except Exception:
                        return 0.0, float(np.median(bs_arr))
                else:
                    return 0.0, float((_iqr_scale(err_a) + _iqr_scale(err_b)) / 2)
            else:
                return 0.0, float((_iqr_scale(err_a) + _iqr_scale(err_b)) / 2)

        center_a, center_b = _fit_scale_vs_size(s_fit, cx_fit, cy_fit)
        size_a, size_b = _fit_scale_vs_size(s_fit, w_fit, h_fit)

        return {
            "noise_distribution": "student_t",
            "center_noise_a": center_a,
            "center_noise_b": float(center_b),
            "center_noise_df": df_center,
            "size_noise_a": size_a,
            "size_noise_b": float(size_b),
            "size_noise_df": df_size,
            "center_bias_x": float(np.median(cx_err)),
            "center_bias_y": float(np.median(cy_err)),
            "fit_size_range": list(fit_size_range),
            "num_matches": len(matched),
        }

    # --- Dual sigmoid miss rate fitting ---
    def fit_miss_sigmoids(self, fit_size_range=(20.0, 80.0)):
        """Fit p_miss = sigmoid(a * (threshold - size)) separately for sky and ground.

        Returns dict with miss_sigmoid_a_sky, miss_size_threshold_sky,
        miss_sigmoid_a_gnd, miss_size_threshold_gnd (compatible with
        detector_replicator.py NoiseModelParams).
        """
        from scipy.optimize import curve_fit

        if not self.detections:
            return {}

        sizes = np.array([d.target_size_px for d in self.detections])
        missed = np.array([not d.yolo_detected for d in self.detections], dtype=float)
        is_sky = np.array([d.background_is_sky for d in self.detections])

        fit_min, fit_max = fit_size_range

        def _fit_sigmoid(sz, miss_flags, n_bins=15):
            """Fit sigmoid to binned miss rate data. Returns (a, threshold)."""
            size_min, size_max = max(sz.min(), 1.0), sz.max()
            if size_max <= size_min or len(sz) < 10:
                return 0.0, 0.0

            bin_edges = np.logspace(np.log10(size_min), np.log10(size_max), n_bins + 1)
            bc, mr, counts = [], [], []
            for i in range(n_bins):
                mask = (sz >= bin_edges[i]) & (sz < bin_edges[i + 1])
                if mask.sum() < 3:
                    continue
                bc.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
                mr.append(miss_flags[mask].mean())
                counts.append(mask.sum())

            if len(bc) < 3:
                return 0.0, 0.0

            bc = np.array(bc)
            mr = np.array(mr)
            weights = np.sqrt(np.array(counts))

            def sigmoid_model(size, a, threshold):
                return 1.0 / (1.0 + np.exp(-a * (threshold - size)))

            # Initial guess: threshold at ~50% crossing
            sorted_idx = np.argsort(bc)
            bc_s, mr_s = bc[sorted_idx], mr[sorted_idx]
            init_thr = bc_s[-1]
            for i in range(len(bc_s) - 1):
                if mr_s[i] >= 0.5 and mr_s[i + 1] < 0.5:
                    init_thr = (bc_s[i] + bc_s[i + 1]) / 2
                    break
            if mr_s[-1] >= 0.5:
                init_thr = bc_s[-1] * 1.5

            try:
                popt, _ = curve_fit(
                    sigmoid_model, bc, mr, p0=[0.1, init_thr],
                    sigma=1.0 / weights, absolute_sigma=False,
                    bounds=([0.001, 0.0], [1.0, size_max * 3]),
                    maxfev=5000,
                )
                return float(popt[0]), float(popt[1])
            except (RuntimeError, ValueError):
                a_fallback = 4.0 / max(bc_s[-1] - bc_s[0], 1.0)
                return float(a_fallback), float(init_thr)

        # Fit sky sigmoid
        sky_mask = is_sky & (sizes >= fit_min) & (sizes <= fit_max)
        if sky_mask.sum() > 10 and missed[sky_mask].sum() > 5:
            a_sky, thr_sky = _fit_sigmoid(sizes[sky_mask], missed[sky_mask])
        elif is_sky.sum() > 10:
            a_sky, thr_sky = _fit_sigmoid(sizes[is_sky], missed[is_sky])
        else:
            a_sky, thr_sky = 0.0, 0.0

        # Fit ground sigmoid
        gnd_mask = ~is_sky & (sizes >= fit_min) & (sizes <= fit_max)
        if gnd_mask.sum() > 10 and missed[gnd_mask].sum() > 5:
            a_gnd, thr_gnd = _fit_sigmoid(sizes[gnd_mask], missed[gnd_mask])
        elif (~is_sky).sum() > 10:
            a_gnd, thr_gnd = _fit_sigmoid(sizes[~is_sky], missed[~is_sky])
        else:
            # Fall back to sky params if insufficient ground data
            a_gnd, thr_gnd = a_sky, thr_sky

        # Combined (legacy backward compat)
        all_mask = (sizes >= fit_min) & (sizes <= fit_max)
        if all_mask.sum() > 10:
            a_all, thr_all = _fit_sigmoid(sizes[all_mask], missed[all_mask])
        else:
            a_all = (a_sky + a_gnd) / 2
            thr_all = (thr_sky + thr_gnd) / 2

        return {
            "miss_sigmoid_a_sky": a_sky,
            "miss_size_threshold_sky": thr_sky,
            "miss_sigmoid_a_gnd": a_gnd,
            "miss_size_threshold_gnd": thr_gnd,
            "miss_sigmoid_a": a_all,
            "miss_size_threshold_px": thr_all,
        }

    # --- FN binned miss rates ---
    def compute_fn_bins(self):
        """Compute miss rate binned by size (all/sky/ground), distance, elevation."""
        if not self.detections:
            return {}

        sizes = np.array([d.target_size_px for d in self.detections])
        dists = np.array([d.distance_m for d in self.detections])
        elevs = np.array([d.elevation_deg for d in self.detections])
        missed = np.array([not d.yolo_detected for d in self.detections])
        is_sky = np.array([d.background_is_sky for d in self.detections])

        def _bin(values, missed_arr, edges, label=None):
            bins = []
            for i in range(len(edges) - 1):
                mask = (values >= edges[i]) & (values < edges[i + 1])
                n = int(mask.sum())
                if n < 3:
                    continue
                n_miss = int(missed_arr[mask].sum())
                ci_lo, ci_hi = _wilson_ci(n_miss, n)
                entry = {"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                         "bin_center": float((edges[i] + edges[i + 1]) / 2),
                         "num_frames": n, "num_misses": n_miss,
                         "miss_rate": n_miss / n,
                         "ci_95_lo": ci_lo, "ci_95_hi": ci_hi}
                if label:
                    entry["background"] = label
                bins.append(entry)
            return bins

        size_edges = np.logspace(np.log10(max(5, sizes.min())),
                                 np.log10(min(200, sizes.max() * 1.1)), 16)
        dist_edges = np.linspace(max(5, dists.min()), min(50, dists.max() * 1.1), 11)
        elev_edges = np.linspace(max(0, elevs.min()), min(90, elevs.max() * 1.1), 10)

        n_sky = is_sky.sum()
        n_gnd = (~is_sky).sum()
        n_miss_sky = missed[is_sky].sum() if n_sky > 0 else 0
        n_miss_gnd = missed[~is_sky].sum() if n_gnd > 0 else 0

        return {
            "by_size": _bin(sizes, missed, size_edges),
            "by_size_sky": _bin(sizes[is_sky], missed[is_sky], size_edges, "sky") if n_sky > 3 else [],
            "by_size_ground": _bin(sizes[~is_sky], missed[~is_sky], size_edges, "ground") if n_gnd > 3 else [],
            "by_distance": _bin(dists, missed, dist_edges),
            "by_elevation": _bin(elevs, missed, elev_edges),
            "summary_sky_vs_ground": {
                "sky_frames": int(n_sky), "sky_misses": int(n_miss_sky),
                "sky_miss_rate": float(n_miss_sky / max(n_sky, 1)),
                "sky_ci_95": list(_wilson_ci(int(n_miss_sky), int(n_sky))),
                "ground_frames": int(n_gnd), "ground_misses": int(n_miss_gnd),
                "ground_miss_rate": float(n_miss_gnd / max(n_gnd, 1)),
                "ground_ci_95": list(_wilson_ci(int(n_miss_gnd), int(n_gnd))),
            },
        }

    # --- Report generation ---
    def generate_report(self, output_dir: Path, miss_sigmoids: dict | None = None):
        """Generate all plots: noise, FN, FP.

        Args:
            miss_sigmoids: Optional dict from fit_miss_sigmoids() with fitted
                sigmoid parameters to overlay on the FN report.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import t as student_t, laplace, norm

        output_dir.mkdir(parents=True, exist_ok=True)
        matched = self.matches

        # ================================================================== #
        #  Figure 1: Noise — localization error distributions (2x3)
        # ================================================================== #
        if len(matched) > 20:
            cx_err = np.array([m.center_error_x for m in matched])
            cy_err = np.array([m.center_error_y for m in matched])
            w_err = np.array([m.size_error_w for m in matched])
            h_err = np.array([m.size_error_h for m in matched])
            sizes = np.array([m.target_size_px for m in matched])
            confs = np.array([m.yolo_confidence for m in matched])

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"Localization Noise — {len(matched)} matches "
                         f"(gimbal locked, zero vel)", fontsize=13)

            # Top row: error distributions with Student's t + Gaussian + Laplace
            for ax, data, label in [(axes[0, 0], cx_err, "Center X"),
                                     (axes[0, 1], cy_err, "Center Y")]:
                ax.hist(data, bins=60, density=True, alpha=0.6, color="steelblue", label="Data")
                r = np.linspace(data.min() - 5, data.max() + 5, 300)
                df, loc, sc = student_t.fit(data)
                ax.plot(r, student_t.pdf(r, df, loc, sc), "r-", lw=2,
                        label=f"t(df={df:.1f})")
                ax.plot(r, norm.pdf(r, np.mean(data), np.std(data)),
                        "g--", lw=1.5, alpha=0.5, label="Gaussian")
                ax.plot(r, laplace.pdf(r, np.median(data),
                        np.median(np.abs(data - np.median(data)))),
                        "m:", lw=1.5, alpha=0.5, label="Laplace")
                ax.set_xlabel(f"{label} error (px)")
                ax.set_ylabel("Density")
                ax.set_title(f"{label} Error")
                ax.legend(fontsize=7)

            # Top right: size W/H error
            ax = axes[0, 2]
            ax.hist(w_err, bins=50, density=True, alpha=0.5, color="mediumpurple", label="Width")
            ax.hist(h_err, bins=50, density=True, alpha=0.5, color="coral", label="Height")
            df_w, loc_w, sc_w = student_t.fit(w_err)
            r_w = np.linspace(min(w_err.min(), h_err.min()) - 5,
                              max(w_err.max(), h_err.max()) + 5, 300)
            ax.plot(r_w, student_t.pdf(r_w, df_w, loc_w, sc_w), "m-", lw=2,
                    label=f"W: t(df={df_w:.1f})")
            ax.set_xlabel("Size error (px)")
            ax.set_title("Width/Height Error")
            ax.legend(fontsize=7)

            # Bottom left: noise vs target size
            ax = axes[1, 0]
            def _iqr(d):
                q75, q25 = np.percentile(d, [75, 25])
                return (q75 - q25) / 1.35
            n_bins = 12
            if sizes.max() > sizes.min():
                be = np.logspace(np.log10(max(sizes.min(), 1)), np.log10(sizes.max()), n_bins + 1)
                bc, bs, bn = [], [], []
                for i in range(n_bins):
                    m = (sizes >= be[i]) & (sizes < be[i + 1])
                    if m.sum() >= 5:
                        bc.append(np.sqrt(be[i] * be[i + 1]))
                        bs.append((_iqr(cx_err[m]) + _iqr(cy_err[m])) / 2)
                        bn.append(m.sum())
                if bc:
                    ax.scatter(bc, bs, s=[max(3, np.sqrt(n) * 2) for n in bn],
                              c="steelblue", alpha=0.7, label="Binned IQR/1.35")
                    ax.set_xscale("log")
            ax.set_xlabel("Target size (px)")
            ax.set_ylabel("1D noise scale (px)")
            ax.set_title("Noise vs Target Size")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Bottom center: confidence distribution
            ax = axes[1, 1]
            ax.hist(confs, bins=40, density=True, alpha=0.7, color="seagreen")
            ax.axvline(np.median(confs), color="orange", ls="--", lw=2,
                       label=f"Median={np.median(confs):.2f}")
            ax.set_xlabel("YOLO confidence")
            ax.set_title("Detection Confidence (hits)")
            ax.legend()

            # Bottom right: confidence vs target size scatter
            ax = axes[1, 2]
            ax.scatter(sizes, confs, s=5, alpha=0.3, c="seagreen")
            ax.set_xlabel("Target size (px)")
            ax.set_ylabel("Confidence")
            ax.set_title("Confidence vs Target Size")
            ax.set_xscale("log")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(output_dir / "noise_report.png", dpi=150)
            plt.close(fig)
            print(f"[CALIB] Noise report saved")

        # ================================================================== #
        #  Figure 2: FN — miss rate vs covariates (2x2)
        # ================================================================== #
        if self.num_frames > 20:
            binned = self.compute_fn_bins()
            summary = binned["summary_sky_vs_ground"]
            n_sky, n_gnd = summary["sky_frames"], summary["ground_frames"]

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"FN Analysis — {self.num_frames} frames, "
                         f"{self.miss_rate:.1%} miss\n"
                         f"Sky: {n_sky} ({summary['sky_miss_rate']:.1%} miss) | "
                         f"Ground: {n_gnd} ({summary['ground_miss_rate']:.1%} miss)",
                         fontsize=12)

            def _plot_bins(ax, bins, color, label):
                if not bins:
                    return
                c = [b["bin_center"] for b in bins]
                r = [b["miss_rate"] for b in bins]
                elo = [max(0, rv - b["ci_95_lo"]) for rv, b in zip(r, bins)]
                ehi = [max(0, b["ci_95_hi"] - rv) for rv, b in zip(r, bins)]
                ax.errorbar(c, r, yerr=[elo, ehi], fmt="o-", color=color,
                           capsize=3, markersize=5, lw=1.5, label=label, alpha=0.8)

            # Miss rate vs size (sky/ground overlay + fitted sigmoids)
            ax = axes[0, 0]
            _plot_bins(ax, binned["by_size"], "tomato", "All")
            _plot_bins(ax, binned["by_size_sky"], "deepskyblue", "Sky behind")
            _plot_bins(ax, binned["by_size_ground"], "saddlebrown", "Ground behind")
            if miss_sigmoids:
                s_range = np.linspace(5, 200, 200)
                for bg_key, color, label in [
                    ("sky", "deepskyblue", "Sky sigmoid"),
                    ("gnd", "saddlebrown", "Gnd sigmoid"),
                ]:
                    a = miss_sigmoids.get(f"miss_sigmoid_a_{bg_key}", 0)
                    thr = miss_sigmoids.get(f"miss_size_threshold_{bg_key}", 0)
                    if a > 0:
                        p_miss = 1.0 / (1.0 + np.exp(-a * (thr - s_range)))
                        ax.plot(s_range, p_miss, "--", color=color, lw=2, alpha=0.7,
                                label=f"{label} (a={a:.3f}, thr={thr:.0f})")
            ax.set_xlabel("Target size (px)")
            ax.set_ylabel("Miss rate")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            ax.set_title("Miss Rate vs Target Size (sky vs ground)")

            # Miss rate vs distance
            ax = axes[0, 1]
            _plot_bins(ax, binned["by_distance"], "tomato", "Miss rate (95% CI)")
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Miss rate")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_title("Miss Rate vs Distance")

            # Miss rate vs elevation
            ax = axes[1, 0]
            _plot_bins(ax, binned["by_elevation"], "tomato", "Miss rate (95% CI)")
            ax.set_xlabel("Elevation (deg below horizontal)")
            ax.set_ylabel("Miss rate")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_title("Miss Rate vs Elevation")

            # Sky vs ground bar
            ax = axes[1, 1]
            rates = [summary["sky_miss_rate"], summary["ground_miss_rate"]]
            ci_lo = [summary["sky_ci_95"][0], summary["ground_ci_95"][0]]
            ci_hi = [summary["sky_ci_95"][1], summary["ground_ci_95"][1]]
            elo = [r - lo for r, lo in zip(rates, ci_lo)]
            ehi = [hi - r for r, hi in zip(rates, ci_hi)]
            bars = ax.bar(["Sky", "Ground"], rates, color=["deepskyblue", "saddlebrown"],
                          alpha=0.7, yerr=[elo, ehi], capsize=8)
            for bar, r, n in zip(bars, rates, [n_sky, n_gnd]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{r:.1%}\n(n={n})", ha="center", fontsize=9)
            ax.set_ylabel("Miss rate")
            ax.set_ylim(0, 1.15)
            ax.set_title("Sky vs Ground Miss Rate")
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            fig.savefig(output_dir / "fn_report.png", dpi=150)
            plt.close(fig)
            print(f"[CALIB] FN report saved")

        # ================================================================== #
        #  Figure 3: FP characteristics (2x3)
        # ================================================================== #
        if len(self.false_positives) > 10:
            fp_cx = np.array([f.bbox_cx for f in self.false_positives])
            fp_cy = np.array([f.bbox_cy for f in self.false_positives])
            fp_w = np.array([f.bbox_w for f in self.false_positives])
            fp_h = np.array([f.bbox_h for f in self.false_positives])
            fp_sizes = np.sqrt(fp_w * fp_h)
            fp_confs = np.array([f.confidence for f in self.false_positives])
            fp_sky = np.array([f.background_is_sky for f in self.false_positives])

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"FP Analysis — {len(self.false_positives)} FP detections "
                         f"(agent-filtered, {self.background_type})", fontsize=13)

            # Spatial heatmap
            ax = axes[0, 0]
            ax.hist2d(fp_cx, fp_cy, bins=30, cmap="YlOrRd")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.set_title("FP Spatial Distribution")
            ax.invert_yaxis()
            ax.set_aspect("equal")

            # FP size
            ax = axes[0, 1]
            ax.hist(fp_sizes, bins=50, density=True, alpha=0.7, color="tomato")
            ax.axvline(np.median(fp_sizes), color="k", ls="--",
                       label=f"Median={np.median(fp_sizes):.0f}px")
            ax.set_xlabel("FP size (px)")
            ax.set_title("FP Size Distribution")
            ax.legend()

            # FP vs TP confidence
            ax = axes[0, 2]
            ax.hist(fp_confs, bins=40, density=True, alpha=0.7, color="tomato", label="FP")
            if matched:
                tp_confs = np.array([m.yolo_confidence for m in matched])
                ax.hist(tp_confs, bins=40, density=True, alpha=0.5, color="seagreen", label="TP")
            ax.set_xlabel("Confidence")
            ax.set_title("FP vs TP Confidence")
            ax.legend()

            # FP aspect ratio
            ax = axes[1, 0]
            fp_aspect = fp_w / np.clip(fp_h, 1, None)
            ax.hist(fp_aspect, bins=50, density=True, alpha=0.7, color="tomato")
            ax.axvline(np.median(fp_aspect), color="k", ls="--",
                       label=f"Median={np.median(fp_aspect):.2f}")
            ax.set_xlabel("Aspect ratio (w/h)")
            ax.set_title("FP Aspect Ratio")
            ax.legend()

            # FP size vs confidence
            ax = axes[1, 1]
            ax.scatter(fp_sizes, fp_confs, s=5, alpha=0.3, c="tomato")
            ax.set_xlabel("FP size (px)")
            ax.set_ylabel("Confidence")
            ax.set_title("FP Size vs Confidence")

            # FP sky vs ground
            ax = axes[1, 2]
            n_fp_sky = fp_sky.sum()
            n_fp_gnd = len(fp_sky) - n_fp_sky
            bars = ax.bar(["Sky", "Ground"], [n_fp_sky, n_fp_gnd],
                          color=["deepskyblue", "saddlebrown"], alpha=0.7)
            for bar, val in zip(bars, [n_fp_sky, n_fp_gnd]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(int(val)), ha="center")
            ax.set_ylabel("FP count")
            ax.set_title("FP by Background Type")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(output_dir / "fp_report.png", dpi=150)
            plt.close(fig)
            print(f"[CALIB] FP report saved")


# --------------------------------------------------------------------------- #
#  Video annotation
# --------------------------------------------------------------------------- #

def _draw_bbox_xywh(img, cx, cy, w, h, color, thickness=2, label=None):
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _annotate_frame(image, gt_bbox, gt_conf, yolo_det, iou_threshold,
                    target_size, distance, elevation, bg_sky,
                    step, env_idx, agent_id, noise_model=None):
    """Annotate frame: Green=GT, Red=YOLO, Cyan=Replicator model (if noise_model given)."""
    if image.dtype == torch.float32:
        frame = (image * 255).byte().cpu().numpy()
    else:
        frame = image.cpu().numpy().copy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    frame = frame[..., ::-1].copy()

    gt = gt_bbox.cpu()
    h, w = frame.shape[:2]
    visible = gt_conf >= 0.5 and target_size >= 1.0

    if visible:
        _draw_bbox_xywh(frame, gt[0].item(), gt[1].item(), gt[2].item(), gt[3].item(),
                        (0, 255, 0), 2, f"GT {target_size:.0f}px")

    # Replicator model overlay (cyan) — raycaster GT + fitted noise + miss
    if noise_model is not None and visible:
        from scipy.stats import t as student_t
        nm = noise_model
        scale = nm.get("center_noise_a", 0) / max(target_size, 5) + nm.get("center_noise_b", 10)
        df = nm.get("center_noise_df", 10)
        # Sample miss (use overall miss rate as proxy — proper implementation
        # would use the sky/ground sigmoid, but we don't have fn_bins here)
        p_miss = nm.get("_overall_miss_rate", 0.5)
        if np.random.random() > p_miss:
            # Detected by model — add noise
            ncx = student_t.rvs(df, loc=nm.get("center_bias_x", 0), scale=scale)
            ncy = student_t.rvs(df, loc=nm.get("center_bias_y", 0), scale=scale)
            nw = student_t.rvs(nm.get("size_noise_df", 10), loc=0, scale=nm.get("center_noise_b", 10))
            nh = student_t.rvs(nm.get("size_noise_df", 10), loc=0, scale=nm.get("center_noise_b", 10))
            mcx = gt[0].item() + ncx
            mcy = gt[1].item() + ncy
            mw = max(gt[2].item() + nw, 1)
            mh = max(gt[3].item() + nh, 1)
            _draw_bbox_xywh(frame, mcx, mcy, mw, mh,
                            (255, 255, 0), 2, "Model")  # cyan in BGR

    # Check YOLO match
    is_hit = False
    yolo_bboxes = yolo_det["bboxes_xywh"]
    if visible and yolo_bboxes.shape[0] > 0:
        ious = _compute_iou_xywh(gt.unsqueeze(0), yolo_bboxes)
        if ious.max().item() >= iou_threshold:
            is_hit = True

    for k in range(yolo_bboxes.shape[0]):
        yb = yolo_bboxes[k]
        _draw_bbox_xywh(frame, yb[0].item(), yb[1].item(), yb[2].item(), yb[3].item(),
                        (0, 0, 255), 2, f"YOLO {yolo_det['confidences'][k].item():.2f}")

    if visible:
        lbl = "HIT" if is_hit else "MISS"
        col = (0, 200, 0) if is_hit else (0, 0, 255)
        cv2.putText(frame, lbl, (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3, cv2.LINE_AA)

    bg = "SKY" if bg_sky else "GND"
    model_lbl = "  [+model]" if noise_model is not None else ""
    cv2.putText(frame, f"size={target_size:.0f}px  dist={distance:.1f}m  "
                f"elev={elevation:.0f}deg  bg={bg}{model_lbl}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Step {step} | Env {env_idx} | {agent_id}",
                (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend (bottom-right)
    legend_y = h - 50
    for lbl_text, lbl_color in [("GT", (0, 255, 0)), ("YOLO", (0, 0, 255)), ("Model", (255, 255, 0))]:
        if lbl_text == "Model" and noise_model is None:
            continue
        cv2.rectangle(frame, (w - 100, legend_y), (w - 85, legend_y + 10), lbl_color, -1)
        cv2.putText(frame, lbl_text, (w - 80, legend_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 15

    return frame


# --------------------------------------------------------------------------- #
#  Policy loader (inlined to avoid importing calibrate_bbox_noise's AppLauncher)
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
    """Unified detector calibration with gimbal lock and zero ego velocity."""

    background_type = "plain_ground" if args_cli.no_flight_scene else "flight_scene"

    # --- Configure env ---
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.enable_tiled_cameras = True
    env_cfg.debug_lock_gimbal_to_target = True
    env_cfg.episode_length_s = 30.0 if args_cli.checkpoint else 5.0
    if args_cli.no_flight_scene:
        env_cfg.use_flight_scene = False

    for attr in dir(env_cfg.curriculum):
        if attr.endswith("_start_step") or attr.endswith("_end_step"):
            setattr(env_cfg.curriculum, attr, 0)

    if args_cli.experiment is not None:
        exp_cfg = get_experiment(args_cli.experiment)
        if exp_cfg.env_overrides:
            apply_env_overrides(env_cfg, exp_cfg.env_overrides)
        print(f"[CALIB] Experiment: {exp_cfg.name}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    unwrapped = env.unwrapped

    num_agents = len(env_cfg.possible_agents)
    possible_agents = list(env_cfg.possible_agents)

    # --- Load policy (optional) ---
    policy = None
    vel_mode = "ZERO"
    if args_cli.checkpoint is not None:
        policy = _load_rnn_policy(
            args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents,
        )
        vel_mode = "POLICY-XYZ"
        print(f"[CALIB] Loaded policy from {args_cli.checkpoint}")
    print(f"[CALIB] {args_cli.num_envs} envs, {num_agents} agents, "
          f"bg={background_type}, gimbal=LOCKED, vel={vel_mode}")

    # --- Initialize ---
    yolo = YoloBatchInference(
        model_path=args_cli.yolo_model, device="cuda:0",
        conf_threshold=args_cli.yolo_conf,
    )
    print(f"[CALIB] YOLO: {args_cli.yolo_model} (conf={args_cli.yolo_conf})")

    collector = DetectorCollector(
        iou_threshold=args_cli.iou_threshold,
        background_type=background_type,
        agent_filter_dist=args_cli.agent_filter_dist,
    )

    record_video = args_cli.video_envs > 0

    # --- Collection loop ---
    obs, info = env_wrapped.reset()
    total_yolo_time = 0.0

    print(f"[CALIB] Collecting {args_cli.num_steps} steps (vel={vel_mode})...")
    for step in range(args_cli.num_steps):
        if policy is not None:
            # Use policy for xyz velocity, zero gimbal/yaw
            with torch.no_grad():
                raw_actions, _, outputs = policy.act(obs, 0, 0)
            actions = {}
            for aid in possible_agents:
                a = outputs[aid]["mean_actions"].clone()
                # Keep xyz (dims 0-2), zero yaw/gimbal/zoom (dims 3-6)
                a[:, 3:] = 0.0
                actions[aid] = a
        else:
            actions = {
                aid: torch.zeros(args_cli.num_envs, 7, device=unwrapped.device)
                for aid in possible_agents
            }
        obs, rewards, terminated, truncated, info = env_wrapped.step(actions)

        for idx, agent_id in enumerate(possible_agents):
            if agent_id not in unwrapped._cameras:
                continue

            camera_data = unwrapped._cameras[agent_id].data.output["rgb"]
            if camera_data is None:
                continue

            rc_bboxes = unwrapped.bbox_raycaster_v2.data.bboxes[:, idx, :, :]
            rc_conf = unwrapped._smoothed_bbox_confidence[:, idx, :]
            cam_pos = unwrapped.bbox_raycaster_v2.data.camera_pos_w[:, idx, :]
            cam_quat = unwrapped.bbox_raycaster_v2.data.camera_quat_w[:, idx, :]
            cam_intr = unwrapped.bbox_raycaster_v2.data.intrinsic_matrices[:, idx, :, :]
            tgt_pos = unwrapped.bbox_raycaster_v2.data.target_pos_w[:, 0, :]

            bg_is_sky = _query_background_is_sky(
                cam_pos, tgt_pos, unwrapped.bbox_raycaster_v2.static_mesh)

            # Project other agents
            oa_pos = [unwrapped._robots[oid].data.root_pos_w
                      for oid in possible_agents if oid != agent_id]
            if oa_pos:
                oa_pos_w = torch.stack(oa_pos, dim=1)
                oa_pixels, oa_valid = _project_points_to_pixels(
                    oa_pos_w, cam_pos, cam_quat, cam_intr)
            else:
                oa_pixels = torch.zeros(args_cli.num_envs, 0, 2, device=unwrapped.device)
                oa_valid = torch.zeros(args_cli.num_envs, 0, dtype=torch.bool, device=unwrapped.device)

            t0 = time.time()
            yolo_results = yolo.detect_batch(camera_data)
            total_yolo_time += time.time() - t0

            collector.ingest(
                yolo_detections=yolo_results,
                rc_bboxes_xywh=rc_bboxes,
                rc_confidence=rc_conf,
                cam_pos=cam_pos, tgt_pos=tgt_pos,
                bg_is_sky=bg_is_sky,
                other_agent_pixels=oa_pixels,
                other_agent_valid=oa_valid,
                step=step, agent_id=agent_id,
            )

        if (step + 1) % 100 == 0:
            n_match = len(collector.matches)
            n_miss = len(collector.misses)
            n_fp = len(collector.false_positives)
            print(f"  Step {step+1}/{args_cli.num_steps}: "
                  f"{collector.num_frames} frames, "
                  f"{n_match} hits, {n_miss} misses ({collector.miss_rate:.1%}), "
                  f"{n_fp} FPs, "
                  f"YOLO {total_yolo_time/(step+1)*1000:.1f}ms/step")

    # --- Results ---
    print(f"\n{'='*70}")
    print(f"DETECTOR CALIBRATION RESULTS ({background_type})")
    print(f"{'='*70}")
    print(f"  Frames:       {collector.num_frames}")
    print(f"  Matches:      {len(collector.matches)}")
    print(f"  Misses:       {len(collector.misses)} ({collector.miss_rate:.1%})")
    print(f"  FPs:          {len(collector.false_positives)} ({collector.fp_rate:.2f}/frame)")

    fn_bins = collector.compute_fn_bins()
    s = fn_bins.get("summary_sky_vs_ground", {})
    if s:
        print(f"  Sky miss:     {s['sky_miss_rate']:.1%} ({s['sky_frames']} frames)")
        print(f"  Ground miss:  {s['ground_miss_rate']:.1%} ({s['ground_frames']} frames)")

    noise = collector.fit_noise_model(
        fit_size_range=(args_cli.fit_size_min, args_cli.fit_size_max))
    if noise:
        print(f"  Noise df:     center={noise['center_noise_df']:.1f}, "
              f"size={noise['size_noise_df']:.1f}")
        print(f"  Center scale: a={noise['center_noise_a']:.1f}/size + "
              f"b={noise['center_noise_b']:.1f}")
        print(f"  Size scale:   a={noise['size_noise_a']:.1f}/size + "
              f"b={noise['size_noise_b']:.1f}")
        print(f"  Bias:         ({noise['center_bias_x']:.1f}, {noise['center_bias_y']:.1f}) px")

    miss_sigmoids = collector.fit_miss_sigmoids(
        fit_size_range=(args_cli.fit_size_min, args_cli.fit_size_max))
    if miss_sigmoids:
        print(f"  Miss (sky):   a={miss_sigmoids['miss_sigmoid_a_sky']:.4f}, "
              f"thr={miss_sigmoids['miss_size_threshold_sky']:.1f}px")
        print(f"  Miss (gnd):   a={miss_sigmoids['miss_sigmoid_a_gnd']:.4f}, "
              f"thr={miss_sigmoids['miss_size_threshold_gnd']:.1f}px")
    print(f"{'='*70}")

    # --- Build NoiseModelParams-compatible dict for detector_replicator ---
    replicator_params = {}
    if noise:
        replicator_params.update(noise)
    if miss_sigmoids:
        replicator_params.update(miss_sigmoids)
    replicator_params["fp_rate"] = collector.fp_rate
    replicator_params["fp_size_range"] = [10.0, 60.0]
    replicator_params["num_frames"] = collector.num_frames
    replicator_params["num_matches"] = len(collector.matches)
    replicator_params["num_misses"] = len(collector.misses)
    replicator_params["num_false_positives"] = len(collector.false_positives)
    replicator_params["yolo_model"] = args_cli.yolo_model
    replicator_params["yolo_conf_threshold"] = args_cli.yolo_conf

    # --- Plots ---
    output_path = Path(args_cli.output)
    report_dir = output_path.parent / (output_path.stem + "_report")
    collector.generate_report(report_dir, miss_sigmoids=miss_sigmoids)

    # --- Video (second pass with model overlay) ---
    if record_video and noise:
        report_dir.mkdir(parents=True, exist_ok=True)
        # Inject overall miss rate into noise dict for model overlay
        noise_for_video = dict(noise)
        noise_for_video["_overall_miss_rate"] = collector.miss_rate

        n_video_steps = min(100, args_cli.num_steps)
        print(f"\n[CALIB] Recording video with model overlay ({n_video_steps} steps)...")
        model_frames: dict[str, list[np.ndarray]] = {aid: [] for aid in possible_agents}
        obs2, _ = env_wrapped.reset()

        for step in range(n_video_steps):
            actions = {aid: torch.zeros(args_cli.num_envs, 7, device=unwrapped.device)
                       for aid in possible_agents}
            obs2, _, _, _, _ = env_wrapped.step(actions)

            for idx, agent_id in enumerate(possible_agents):
                if agent_id not in unwrapped._cameras:
                    continue
                camera_data = unwrapped._cameras[agent_id].data.output["rgb"]
                if camera_data is None:
                    continue
                rc_bboxes = unwrapped.bbox_raycaster_v2.data.bboxes[:, idx, :, :]
                rc_conf = unwrapped._smoothed_bbox_confidence[:, idx, :]
                cam_pos = unwrapped.bbox_raycaster_v2.data.camera_pos_w[:, idx, :]
                tgt_pos = unwrapped.bbox_raycaster_v2.data.target_pos_w[:, 0, :]
                bg_sky = _query_background_is_sky(
                    cam_pos, tgt_pos, unwrapped.bbox_raycaster_v2.static_mesh)
                yolo_results = yolo.detect_batch(camera_data)
                dist_v = (cam_pos - tgt_pos).norm(dim=-1)
                elev_v = _compute_elevation_deg(cam_pos, tgt_pos)

                for ei in range(min(args_cli.video_envs, args_cli.num_envs)):
                    gt_b = rc_bboxes[ei, 0]
                    gt_sz = (gt_b[2] * gt_b[3]).sqrt().item()
                    frame = _annotate_frame(
                        camera_data[ei], gt_b, rc_conf[ei, 0].item(),
                        yolo_results[ei], args_cli.iou_threshold,
                        gt_sz, dist_v[ei].item(), elev_v[ei].item(),
                        bg_sky[ei].item(), step, ei, agent_id,
                        noise_model=noise_for_video,
                    )
                    model_frames[agent_id].append(frame)

        for aid, frames in model_frames.items():
            if not frames:
                continue
            h, w = frames[0].shape[:2]
            vp = report_dir / f"detector_{aid}_{background_type}.mp4"
            wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"),
                                  args_cli.video_fps, (w, h))
            for f in frames:
                wr.write(f)
            wr.release()
            print(f"[CALIB] Video: {vp} ({len(frames)} frames)")

    # --- Save full calibration JSON ---
    output_data = {
        "metadata": {
            "background_type": background_type,
            "gimbal_locked": True,
            "ego_velocity": "policy_xyz" if args_cli.checkpoint else "zero",
            "checkpoint": args_cli.checkpoint,
            "yolo_model": args_cli.yolo_model,
            "yolo_conf_threshold": args_cli.yolo_conf,
            "iou_threshold": args_cli.iou_threshold,
            "agent_filter_dist_px": args_cli.agent_filter_dist,
            "num_envs": args_cli.num_envs,
            "num_steps": args_cli.num_steps,
            "num_agents": num_agents,
            "total_frames": collector.num_frames,
            "total_matches": len(collector.matches),
            "total_misses": len(collector.misses),
            "overall_miss_rate": collector.miss_rate,
            "total_fps": len(collector.false_positives),
            "fp_rate": collector.fp_rate,
            "experiment": args_cli.experiment,
        },
        "noise_model": noise,
        "miss_sigmoids": miss_sigmoids,
        "fn_binned_miss_rates": fn_bins,
        "fp_summary": {
            "count": len(collector.false_positives),
            "rate_per_frame": collector.fp_rate,
            "median_size_px": float(np.median([
                (f.bbox_w * f.bbox_h) ** 0.5 for f in collector.false_positives
            ])) if collector.false_positives else 0.0,
            "median_confidence": float(np.median([
                f.confidence for f in collector.false_positives
            ])) if collector.false_positives else 0.0,
        },
        "replicator_params": replicator_params,
        "raw_detections": [asdict(d) for d in collector.detections],
        "raw_false_positives": [asdict(f) for f in collector.false_positives],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[CALIB] Saved: {output_path}")

    # --- Save standalone replicator params JSON (loadable by NoiseModelParams.from_json) ---
    replicator_json_path = output_path.parent / (output_path.stem + "_replicator_params.json")
    with open(replicator_json_path, "w") as f:
        json.dump(replicator_params, f, indent=2)
    print(f"[CALIB] Replicator params: {replicator_json_path}")

    print(f"[CALIB] YOLO total: {total_yolo_time:.1f}s, "
          f"avg: {total_yolo_time/args_cli.num_steps*1000:.1f}ms/step")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()