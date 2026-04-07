#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Background clutter FN calibration — measure YOLO miss rate with gimbal locked to target.

With gimbal locked (`debug_lock_gimbal_to_target=True`), the target is always centered
in the camera frame. Any YOLO miss is therefore attributable to detector failure against
the background, not gimbal tracking error. This isolates the FN rate due to background
clutter (buildings, terrain, sky) from the FN rate due to gimbal shaking.

Run WITH and WITHOUT the Flight scene to isolate clutter effect:

    # With Flight scene (background clutter)
    python calibrate_clutter_fn.py --experiment a1_with_aoi \\
        --checkpoint /path/to/best_agent.pt \\
        --num_envs 16 --num_steps 500 \\
        --output clutter_fn_flight.json

    # Without Flight scene (control — plain ground)
    python calibrate_clutter_fn.py --experiment a1_with_aoi \\
        --checkpoint /path/to/best_agent.pt \\
        --num_envs 16 --num_steps 500 --no_flight_scene \\
        --output clutter_fn_ground.json
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="Background clutter FN calibration (gimbal locked to target).")
parser.add_argument("--experiment", type=str, default="a1_with_aoi",
                    help="Experiment name from registry (for env overrides)")
parser.add_argument("--checkpoint", type=str, default="/home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma6/2026-04-01_13-19-18_mappo_rnn_torch_928b9585f2_min_lr_tuning/checkpoints/best_agent.pt",
                    help="Path to trained RL policy checkpoint (.pt). If None, uses random actions.")
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
parser.add_argument("--output", type=str, default="clutter_fn_data.json",
                    help="Output JSON path")
parser.add_argument("--no_flight_scene", action="store_true", default=False,
                    help="Disable Flight scene for control run (plain ground)")
parser.add_argument("--iou_threshold", type=float, default=0.1,
                    help="IoU threshold for matching YOLO to raycaster GT")
parser.add_argument("--video_envs", type=int, default=1,
                    help="Number of envs to record video for (0 to disable)")
parser.add_argument("--video_fps", type=int, default=25,
                    help="Video output FPS")

# Parse our args first, pass unknown args to Hydra
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

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
import math
import os
import time
from dataclasses import dataclass, asdict
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
from isaaclab_tasks.direct.iris_ma6.experiments.env_overrides import apply_env_overrides, apply_agent_overrides


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #

@dataclass
class ClutterFNSample:
    """Single frame observation for clutter FN analysis."""
    step: int
    env_idx: int
    agent_id: str
    target_size_px: float
    distance_m: float
    elevation_deg: float        # pitch below horizontal (positive = looking down)
    raycaster_confidence: float
    raycaster_bbox_cx: float
    raycaster_bbox_cy: float
    raycaster_bbox_w: float
    raycaster_bbox_h: float
    yolo_detected: bool
    yolo_confidence: float      # 0.0 if miss
    yolo_bbox_iou: float        # 0.0 if miss
    background_type: str
    background_is_sky: bool     # True if ray through target hits nothing (sky behind target)


# --------------------------------------------------------------------------- #
#  YOLO wrapper (copied from calibrate_bbox_noise.py — cannot import due to
#  module-level AppLauncher side effect)
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
    return inter_area / (b1_area + b2_area - inter_area + 1e-8)


def _compute_elevation_deg(cam_pos: torch.Tensor, tgt_pos: torch.Tensor) -> torch.Tensor:
    """Elevation angle (degrees below horizontal) from camera to target.

    Args:
        cam_pos: (N, 3) camera world positions.
        tgt_pos: (N, 3) target world positions.

    Returns:
        (N,) elevation in degrees. Positive = looking down.
    """
    delta = tgt_pos - cam_pos
    horiz_dist = delta[:, :2].norm(dim=-1)
    vert_dist = -delta[:, 2]  # positive when target is below camera
    return torch.rad2deg(torch.atan2(vert_dist, horiz_dist))


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 1.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _query_background_is_sky(
    cam_pos: torch.Tensor,
    tgt_pos: torch.Tensor,
    static_mesh,
    margin: float = 2.0,
) -> torch.Tensor:
    """Cast rays through target and check if background is sky (no mesh hit).

    Args:
        cam_pos: (N, 3) camera positions in world frame.
        tgt_pos: (N, 3) target positions in world frame.
        static_mesh: wp.Mesh (ground + scene geometry) or None.
        margin: Start ray this many meters past the target to avoid hitting the target itself.

    Returns:
        (N,) bool tensor. True = sky behind target (ray missed all meshes).
    """
    if static_mesh is None:
        return torch.ones(cam_pos.shape[0], dtype=torch.bool, device=cam_pos.device)

    from isaaclab.utils.warp import raycast_mesh

    # Ray direction: camera → target, normalized
    ray_dir = tgt_pos - cam_pos
    dist_to_target = ray_dir.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    ray_dir_norm = ray_dir / dist_to_target

    # Start ray past the target (so we don't hit the target itself)
    ray_starts = tgt_pos + ray_dir_norm * margin

    # raycast_mesh expects (..., 3) with at least 2 leading dims for distance reshape
    # Unsqueeze to (N, 1, 3) so distance returns as (N, 1)
    ray_hits, ray_dist, _, _ = raycast_mesh(
        ray_starts.unsqueeze(1), ray_dir_norm.unsqueeze(1), static_mesh,
        max_dist=1e4, return_distance=True,
    )

    # inf distance = missed all meshes = sky. Squeeze (N, 1) → (N,)
    is_sky = torch.isinf(ray_dist.squeeze(1))
    return is_sky


def _project_points_to_pixels(
    points_w: torch.Tensor,     # (N, P, 3) — P points per env
    cam_pos_w: torch.Tensor,    # (N, 3)
    cam_quat_w: torch.Tensor,   # (N, 4) — camera orientation (wxyz)
    intrinsic: torch.Tensor,    # (N, 3, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-frame 3D points to pixel coordinates.

    Args:
        points_w: (N, P, 3) world positions of P points per env.
        cam_pos_w: (N, 3) camera positions.
        cam_quat_w: (N, 4) camera quaternions (wxyz).
        intrinsic: (N, 3, 3) intrinsic matrices.

    Returns:
        pixels: (N, P, 2) pixel coordinates (u, v).
        valid: (N, P) bool — True if point is in front of camera.
    """
    from isaaclab_tasks.direct.iris_ma6.bbox_raycaster_v2.utils.projection import (
        batch_transform_to_camera_frame,
        batch_project_to_image_plane,
    )
    N, P = points_w.shape[:2]
    # Reshape to (N, C=1, T=P, K=1, 3) for batch projection utils
    pts = points_w.unsqueeze(1).unsqueeze(3)       # (N, 1, P, 1, 3)
    cp = cam_pos_w.unsqueeze(1)                     # (N, 1, 3)
    cq = cam_quat_w.unsqueeze(1)                    # (N, 1, 4)
    intr = intrinsic.unsqueeze(1)                    # (N, 1, 3, 3)

    pts_cam = batch_transform_to_camera_frame(pts, cp, cq)       # (N, 1, P, 1, 3)
    pixels, depths, valid = batch_project_to_image_plane(pts_cam, intr)  # (N, 1, P, 1, 2)

    return pixels[:, 0, :, 0, :], valid[:, 0, :, 0]  # (N, P, 2), (N, P)


def _filter_agent_detections(
    yolo_bboxes_xywh: torch.Tensor,   # (K, 4)
    agent_pixels: torch.Tensor,        # (P, 2) — projected pixel centers of other agents
    agent_valid: torch.Tensor,         # (P,) bool
    dist_threshold_px: float = 80.0,
) -> torch.Tensor:
    """Return mask of YOLO detections that are NOT other agents.

    A YOLO detection is considered an "other agent" detection if its center
    is within dist_threshold_px of any projected agent position.

    Args:
        yolo_bboxes_xywh: (K, 4) YOLO detections.
        agent_pixels: (P, 2) pixel positions of other agents.
        agent_valid: (P,) validity mask.
        dist_threshold_px: Distance threshold for matching.

    Returns:
        (K,) bool — True if detection is NOT an other-agent detection (keep it).
    """
    K = yolo_bboxes_xywh.shape[0]
    if K == 0 or agent_valid.sum() == 0:
        return torch.ones(K, dtype=torch.bool)

    yolo_centers = yolo_bboxes_xywh[:, :2]  # (K, 2)
    valid_agents = agent_pixels[agent_valid]  # (P', 2)

    # Pairwise distance: (K, P')
    dists = torch.cdist(yolo_centers.float(), valid_agents.float())
    min_dist_to_agent = dists.min(dim=1).values  # (K,)

    return min_dist_to_agent > dist_threshold_px


def _draw_bbox_xywh(img: np.ndarray, cx, cy, w, h, color, thickness=2, label=None):
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
#  Clutter FN Collector
# --------------------------------------------------------------------------- #

class ClutterFNCollector:
    """Collects per-frame FN statistics with gimbal locked to target."""

    def __init__(self, iou_threshold: float = 0.1, background_type: str = "flight_scene"):
        self.iou_threshold = iou_threshold
        self.background_type = background_type
        self.samples: list[ClutterFNSample] = []

    def ingest_frame(
        self,
        yolo_detections: list[dict],
        raycaster_bboxes_xywh: torch.Tensor,   # (B, T, 4)
        raycaster_confidence: torch.Tensor,     # (B, T)
        cam_pos: torch.Tensor,                  # (B, 3)
        tgt_pos: torch.Tensor,                  # (B, 3)
        background_is_sky: torch.Tensor,        # (B,) bool
        other_agent_pixels: torch.Tensor,       # (B, P, 2) pixel coords of other agents
        other_agent_valid: torch.Tensor,        # (B, P) bool
        step: int,
        agent_id: str,
    ):
        """Process one batch of frames for one agent.

        Filters out YOLO detections that match other agent positions
        before performing target IoU matching.
        """
        B = raycaster_bboxes_xywh.shape[0]
        T = raycaster_bboxes_xywh.shape[1]
        rc_bboxes = raycaster_bboxes_xywh.cpu()
        rc_conf = raycaster_confidence.cpu()
        dist = (cam_pos - tgt_pos).norm(dim=-1).cpu()          # (B,)
        elev = _compute_elevation_deg(cam_pos, tgt_pos).cpu()  # (B,)
        is_sky = background_is_sky.cpu()                       # (B,)
        oa_pixels = other_agent_pixels.cpu()                   # (B, P, 2)
        oa_valid = other_agent_valid.cpu()                     # (B, P)

        for b in range(B):
            for t in range(T):
                gt_bbox = rc_bboxes[b, t]
                gt_conf = rc_conf[b, t].item()
                gt_size = (gt_bbox[2] * gt_bbox[3]).sqrt().item()

                if gt_conf < 0.5 or gt_size < 1.0:
                    continue

                # Filter out detections of other agents
                yolo = yolo_detections[b]
                detected = False
                best_conf = 0.0
                best_iou = 0.0

                if yolo["bboxes_xywh"].shape[0] > 0:
                    keep_mask = _filter_agent_detections(
                        yolo["bboxes_xywh"], oa_pixels[b], oa_valid[b],
                    )
                    filtered_bboxes = yolo["bboxes_xywh"][keep_mask]
                    filtered_confs = yolo["confidences"][keep_mask]

                    if filtered_bboxes.shape[0] > 0:
                        ious = _compute_iou_xywh(gt_bbox.unsqueeze(0), filtered_bboxes)
                        max_iou, max_idx = ious.max(dim=-1)
                        best_iou = max_iou.item()
                        if best_iou >= self.iou_threshold:
                            detected = True
                            best_conf = filtered_confs[max_idx.item()].item()

                self.samples.append(ClutterFNSample(
                    step=step,
                    env_idx=b,
                    agent_id=agent_id,
                    target_size_px=gt_size,
                    distance_m=dist[b].item(),
                    elevation_deg=elev[b].item(),
                    raycaster_confidence=gt_conf,
                    raycaster_bbox_cx=gt_bbox[0].item(),
                    raycaster_bbox_cy=gt_bbox[1].item(),
                    raycaster_bbox_w=gt_bbox[2].item(),
                    raycaster_bbox_h=gt_bbox[3].item(),
                    yolo_detected=detected,
                    yolo_confidence=best_conf,
                    yolo_bbox_iou=best_iou,
                    background_type=self.background_type,
                    background_is_sky=bool(is_sky[b].item()),
                ))

    @property
    def num_frames(self):
        return len(self.samples)

    @property
    def num_misses(self):
        return sum(1 for s in self.samples if not s.yolo_detected)

    @property
    def miss_rate(self):
        return self.num_misses / max(self.num_frames, 1)

    def compute_binned_miss_rates(self) -> dict:
        """Bin samples by size, distance, elevation and compute per-bin miss rates with Wilson CI.

        Also computes separate bins for sky vs ground background.
        """
        if not self.samples:
            return {"by_size": [], "by_distance": [], "by_elevation": [],
                    "by_size_sky": [], "by_size_ground": [],
                    "summary_sky_vs_ground": {}}

        sizes = np.array([s.target_size_px for s in self.samples])
        dists = np.array([s.distance_m for s in self.samples])
        elevs = np.array([s.elevation_deg for s in self.samples])
        missed = np.array([not s.yolo_detected for s in self.samples])
        is_sky = np.array([s.background_is_sky for s in self.samples])

        def _bin_miss_rate(values, missed, bin_edges, label=None):
            bins = []
            for i in range(len(bin_edges) - 1):
                mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
                n = mask.sum()
                if n < 3:
                    continue
                n_miss = missed[mask].sum()
                mr = n_miss / n
                ci_lo, ci_hi = _wilson_ci(int(n_miss), int(n))
                entry = {
                    "bin_lo": float(bin_edges[i]),
                    "bin_hi": float(bin_edges[i + 1]),
                    "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                    "num_frames": int(n),
                    "num_misses": int(n_miss),
                    "miss_rate": float(mr),
                    "ci_95_lo": float(ci_lo),
                    "ci_95_hi": float(ci_hi),
                    "background_type": self.background_type,
                }
                if label:
                    entry["background_behind"] = label
                bins.append(entry)
            return bins

        # Bin edges
        size_edges = np.logspace(np.log10(max(5.0, sizes.min())),
                                 np.log10(min(200.0, sizes.max() * 1.1)), 16)
        dist_edges = np.linspace(max(5.0, dists.min()), min(50.0, dists.max() * 1.1), 11)
        elev_edges = np.linspace(max(0.0, elevs.min()), min(90.0, elevs.max() * 1.1), 10)

        # Sky vs ground summary
        n_sky = is_sky.sum()
        n_ground = (~is_sky).sum()
        n_miss_sky = missed[is_sky].sum() if n_sky > 0 else 0
        n_miss_ground = missed[~is_sky].sum() if n_ground > 0 else 0
        sky_ci = _wilson_ci(int(n_miss_sky), int(n_sky)) if n_sky > 0 else (0.0, 1.0)
        gnd_ci = _wilson_ci(int(n_miss_ground), int(n_ground)) if n_ground > 0 else (0.0, 1.0)

        return {
            "by_size": _bin_miss_rate(sizes, missed, size_edges),
            "by_distance": _bin_miss_rate(dists, missed, dist_edges),
            "by_elevation": _bin_miss_rate(elevs, missed, elev_edges),
            "by_size_sky": _bin_miss_rate(sizes[is_sky], missed[is_sky], size_edges, "sky") if n_sky > 3 else [],
            "by_size_ground": _bin_miss_rate(sizes[~is_sky], missed[~is_sky], size_edges, "ground") if n_ground > 3 else [],
            "summary_sky_vs_ground": {
                "sky_frames": int(n_sky),
                "sky_misses": int(n_miss_sky),
                "sky_miss_rate": float(n_miss_sky / max(n_sky, 1)),
                "sky_ci_95": list(sky_ci),
                "ground_frames": int(n_ground),
                "ground_misses": int(n_miss_ground),
                "ground_miss_rate": float(n_miss_ground / max(n_ground, 1)),
                "ground_ci_95": list(gnd_ci),
            },
        }

    def generate_plots(self, output_dir: Path):
        """Generate miss-rate-vs-covariate plots with CI error bars."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(self.samples) < 20:
            print("[WARN] Too few samples for plots")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        binned = self.compute_binned_miss_rates()

        # --- Sky vs ground summary ---
        summary = binned["summary_sky_vs_ground"]
        n_sky = summary["sky_frames"]
        n_gnd = summary["ground_frames"]
        print(f"[CLUTTER] Background: sky={n_sky} frames ({summary['sky_miss_rate']:.1%} miss), "
              f"ground={n_gnd} frames ({summary['ground_miss_rate']:.1%} miss)")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Background Clutter FN Analysis — {self.background_type} "
                     f"({self.num_frames} frames, {self.miss_rate:.1%} miss rate)\n"
                     f"Sky: {n_sky} frames, {summary['sky_miss_rate']:.1%} miss | "
                     f"Ground: {n_gnd} frames, {summary['ground_miss_rate']:.1%} miss",
                     fontsize=12)

        # --- Top-left: Miss rate vs size — combined + sky + ground overlay ---
        ax = axes[0, 0]
        def _plot_bins(ax, bins, color, label, use_log=False):
            if not bins:
                return
            c = [b["bin_center"] for b in bins]
            r = [b["miss_rate"] for b in bins]
            lo = [b["ci_95_lo"] for b in bins]
            hi = [b["ci_95_hi"] for b in bins]
            err_lo = [rv - lv for rv, lv in zip(r, lo)]
            err_hi = [hv - rv for rv, hv in zip(r, hi)]
            ax.errorbar(c, r, yerr=[err_lo, err_hi], fmt="o-", color=color,
                       capsize=3, markersize=5, lw=1.5, label=label, alpha=0.8)

        _plot_bins(ax, binned["by_size"], "tomato", "All", use_log=True)
        _plot_bins(ax, binned["by_size_sky"], "deepskyblue", "Sky behind target")
        _plot_bins(ax, binned["by_size_ground"], "saddlebrown", "Ground/building behind")
        ax.set_xlabel("Target size (px)")
        ax.set_ylabel("Miss rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Miss Rate vs Target Size (sky vs ground)")

        # --- Top-right: Miss rate vs distance ---
        ax = axes[0, 1]
        _plot_bins(ax, binned["by_distance"], "tomato", "Miss rate (95% CI)")
        ax.set_xlabel("Camera-to-target distance (m)")
        ax.set_ylabel("Miss rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Miss Rate vs Distance")

        # --- Bottom-left: Miss rate vs elevation ---
        ax = axes[1, 0]
        _plot_bins(ax, binned["by_elevation"], "tomato", "Miss rate (95% CI)")
        ax.set_xlabel("Elevation angle (deg below horizontal)")
        ax.set_ylabel("Miss rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Miss Rate vs Elevation")

        # --- Bottom-right: Sky vs ground bar chart ---
        ax = axes[1, 1]
        labels = ["Sky\nbehind target", "Ground/building\nbehind target"]
        miss_rates = [summary["sky_miss_rate"], summary["ground_miss_rate"]]
        ci_lo_vals = [summary["sky_ci_95"][0], summary["ground_ci_95"][0]]
        ci_hi_vals = [summary["sky_ci_95"][1], summary["ground_ci_95"][1]]
        err_lo = [r - lo for r, lo in zip(miss_rates, ci_lo_vals)]
        err_hi = [hi - r for r, hi in zip(miss_rates, ci_hi_vals)]
        colors = ["deepskyblue", "saddlebrown"]
        bars = ax.bar(labels, miss_rates, color=colors, alpha=0.7,
                      yerr=[err_lo, err_hi], capsize=8)
        for bar, rate, n in zip(bars, miss_rates, [n_sky, n_gnd]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{rate:.1%}\n(n={n})", ha="center", fontsize=9)
        ax.set_ylabel("Miss rate")
        ax.set_ylim(0, 1.15)
        ax.set_title("Sky vs Ground Background Miss Rate")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig_path = output_dir / "clutter_fn_miss_rates.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[CLUTTER] Plot saved to {fig_path}")

        # --- Additional: hit confidence vs covariates ---
        hits = [s for s in self.samples if s.yolo_detected]
        if len(hits) > 20:
            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
            fig2.suptitle("Hit Confidence vs Covariates (gimbal-locked)", fontsize=13)

            hit_sizes = np.array([s.target_size_px for s in hits])
            hit_dists = np.array([s.distance_m for s in hits])
            hit_elevs = np.array([s.elevation_deg for s in hits])
            hit_confs = np.array([s.yolo_confidence for s in hits])

            for ax, (xdata, xlabel, use_log) in zip(axes2, [
                (hit_sizes, "Target size (px)", True),
                (hit_dists, "Distance (m)", False),
                (hit_elevs, "Elevation (deg)", False),
            ]):
                ax.scatter(xdata, hit_confs, s=5, alpha=0.3, c="seagreen")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("YOLO confidence")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                if use_log:
                    ax.set_xscale("log")

            plt.tight_layout()
            fig2_path = output_dir / "clutter_fn_hit_confidence.png"
            fig2.savefig(fig2_path, dpi=150)
            plt.close(fig2)
            print(f"[CLUTTER] Hit confidence plot saved to {fig2_path}")


# --------------------------------------------------------------------------- #
#  Video annotation
# --------------------------------------------------------------------------- #

def _annotate_clutter_frame(
    image: torch.Tensor,
    raycaster_bbox_xywh: torch.Tensor,
    raycaster_conf: float,
    yolo_detection: dict,
    iou_threshold: float,
    target_size_px: float,
    distance_m: float,
    elevation_deg: float,
    background_is_sky: bool,
    step: int,
    env_idx: int,
    agent_id: str,
) -> np.ndarray:
    """Annotate a frame for clutter FN visualization.

    Green = raycaster GT, Red = YOLO detection, large HIT/MISS label.
    """
    if image.dtype == torch.float32:
        frame = (image * 255).byte().cpu().numpy()
    else:
        frame = image.cpu().numpy().copy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    frame = frame[..., ::-1].copy()  # RGB → BGR

    rc = raycaster_bbox_xywh.cpu()
    h, w = frame.shape[:2]

    # Draw raycaster GT (green)
    is_visible = raycaster_conf >= 0.5 and target_size_px >= 1.0
    if is_visible:
        _draw_bbox_xywh(frame, rc[0].item(), rc[1].item(), rc[2].item(), rc[3].item(),
                        color=(0, 255, 0), thickness=2,
                        label=f"GT {target_size_px:.0f}px")

    # Check YOLO match
    is_hit = False
    yolo_bboxes = yolo_detection["bboxes_xywh"]
    yolo_confs = yolo_detection["confidences"]
    if is_visible and yolo_bboxes.shape[0] > 0:
        ious = _compute_iou_xywh(rc.unsqueeze(0), yolo_bboxes)
        best_iou, best_idx = ious.max(dim=-1)
        if best_iou.item() >= iou_threshold:
            is_hit = True

    # Draw all YOLO detections (red)
    for k in range(yolo_bboxes.shape[0]):
        yb = yolo_bboxes[k]
        conf = yolo_confs[k].item()
        _draw_bbox_xywh(frame, yb[0].item(), yb[1].item(), yb[2].item(), yb[3].item(),
                        color=(0, 0, 255), thickness=2,
                        label=f"YOLO {conf:.2f}")

    # HIT / MISS label (large, top-right)
    if is_visible:
        label = "HIT" if is_hit else "MISS"
        color = (0, 200, 0) if is_hit else (0, 0, 255)
        cv2.putText(frame, label, (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3, cv2.LINE_AA)

    # Covariate overlay (top-left)
    bg_label = "SKY" if background_is_sky else "GND"
    info = f"size={target_size_px:.0f}px  dist={distance_m:.1f}m  elev={elevation_deg:.0f}deg  bg={bg_label}"
    cv2.putText(frame, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Step/env info (bottom-left)
    cv2.putText(frame, f"Step {step} | Env {env_idx} | {agent_id}",
                (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# --------------------------------------------------------------------------- #
#  Policy loader (inlined — cannot import from calibrate_bbox_noise.py or evaluate.py)
# --------------------------------------------------------------------------- #

def _load_rnn_policy(checkpoint_path, env, agent_cfg, possible_agents):
    """Load a trained MAPPO-RNN policy from checkpoint."""
    import copy
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
        "scripts", "reinforcement_learning", "skrl")))
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
            num_envs=env.num_envs, device=device,
        ) for aid in possible_agents
    }

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
        possible_agents=possible_agents, models=models, memories=memories,
        observation_spaces=env.observation_spaces, action_spaces=env.action_spaces,
        device=device, cfg=mappo_cfg, shared_observation_spaces=shared_observation_spaces,
    )
    agent.load(checkpoint_path)
    agent.set_mode("eval")
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
    """Collect background clutter FN data with gimbal locked to target."""

    # --- Determine background type ---
    if args_cli.no_flight_scene:
        env_cfg.use_flight_scene = False
        background_type = "plain_ground"
    else:
        env_cfg.use_flight_scene = True
        background_type = "flight_scene"

    # --- Configure env ---
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.enable_tiled_cameras = True
    env_cfg.debug_lock_gimbal_to_target = True  # THE KEY DIFFERENCE
    env_cfg.episode_length_s = 30.0

    # Force curriculum to full difficulty
    for attr in dir(env_cfg.curriculum):
        if attr.endswith("_start_step") or attr.endswith("_end_step"):
            setattr(env_cfg.curriculum, attr, 0)

    if args_cli.experiment is not None:
        exp_cfg = get_experiment(args_cli.experiment)
        if exp_cfg.env_overrides:
            apply_env_overrides(env_cfg, exp_cfg.env_overrides)
        print(f"[CLUTTER] Experiment: {exp_cfg.name}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    unwrapped = env.unwrapped

    num_agents = len(env_cfg.possible_agents)
    possible_agents = list(env_cfg.possible_agents)
    print(f"[CLUTTER] {args_cli.num_envs} envs, {num_agents} agents, "
          f"background={background_type}, gimbal=LOCKED")

    # --- Load policy (optional) ---
    policy = None
    if args_cli.checkpoint is not None:
        policy = _load_rnn_policy(args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents)
        print(f"[CLUTTER] Loaded policy from {args_cli.checkpoint}")
    else:
        print(f"[CLUTTER] No checkpoint — using random actions")

    # --- Initialize YOLO ---
    yolo = YoloBatchInference(
        model_path=args_cli.yolo_model, device="cuda:0",
        conf_threshold=args_cli.yolo_conf,
    )
    print(f"[CLUTTER] YOLO model: {args_cli.yolo_model} (conf={args_cli.yolo_conf})")

    # --- Initialize collector ---
    collector = ClutterFNCollector(
        iou_threshold=args_cli.iou_threshold,
        background_type=background_type,
    )

    # --- Video frame collection ---
    record_video = args_cli.video_envs > 0
    video_frames: dict[str, list[np.ndarray]] = {}
    if record_video:
        video_frames = {aid: [] for aid in possible_agents}

    # --- Collection loop ---
    obs, info = env_wrapped.reset()
    total_yolo_time = 0.0

    print(f"[CLUTTER] Collecting {args_cli.num_steps} steps...")
    for step in range(args_cli.num_steps):
        if policy is not None:
            with torch.no_grad():
                actions, _, outputs = policy.act(obs, 0, 0)
                for agent_id in possible_agents:
                    actions[agent_id] = outputs[agent_id]["mean_actions"]
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
            rc_confidence = unwrapped._smoothed_bbox_confidence[:, idx, :]
            cam_pos = unwrapped.bbox_raycaster_v2.data.camera_pos_w[:, idx, :]
            cam_quat = unwrapped.bbox_raycaster_v2.data.camera_quat_w[:, idx, :]
            cam_intr = unwrapped.bbox_raycaster_v2.data.intrinsic_matrices[:, idx, :, :]
            tgt_pos = unwrapped.bbox_raycaster_v2.data.target_pos_w[:, 0, :]

            # Query background: cast ray through target, check if it hits scene geometry
            bg_is_sky = _query_background_is_sky(
                cam_pos, tgt_pos, unwrapped.bbox_raycaster_v2.static_mesh,
            )

            # Project other agents' positions to this camera's image plane
            other_agent_pos = []
            for other_idx, other_id in enumerate(possible_agents):
                if other_id == agent_id:
                    continue
                other_agent_pos.append(unwrapped._robots[other_id].data.root_pos_w)
            if other_agent_pos:
                # (N, P, 3) where P = num_other_agents
                oa_pos_w = torch.stack(other_agent_pos, dim=1)
                oa_pixels, oa_valid = _project_points_to_pixels(
                    oa_pos_w, cam_pos, cam_quat, cam_intr,
                )
            else:
                oa_pixels = torch.zeros(args_cli.num_envs, 0, 2, device=unwrapped.device)
                oa_valid = torch.zeros(args_cli.num_envs, 0, dtype=torch.bool, device=unwrapped.device)

            t0 = time.time()
            yolo_results = yolo.detect_batch(camera_data)
            total_yolo_time += time.time() - t0

            collector.ingest_frame(
                yolo_detections=yolo_results,
                raycaster_bboxes_xywh=rc_bboxes,
                raycaster_confidence=rc_confidence,
                cam_pos=cam_pos,
                tgt_pos=tgt_pos,
                background_is_sky=bg_is_sky,
                other_agent_pixels=oa_pixels,
                other_agent_valid=oa_valid,
                step=step,
                agent_id=agent_id,
            )

            # Video frames
            if record_video:
                dist_vec = (cam_pos - tgt_pos).norm(dim=-1)
                elev_vec = _compute_elevation_deg(cam_pos, tgt_pos)
                for env_i in range(min(args_cli.video_envs, args_cli.num_envs)):
                    gt_bbox = rc_bboxes[env_i, 0]
                    gt_size = (gt_bbox[2] * gt_bbox[3]).sqrt().item()
                    frame = _annotate_clutter_frame(
                        image=camera_data[env_i],
                        raycaster_bbox_xywh=gt_bbox,
                        raycaster_conf=rc_confidence[env_i, 0].item(),
                        yolo_detection=yolo_results[env_i],
                        iou_threshold=args_cli.iou_threshold,
                        target_size_px=gt_size,
                        distance_m=dist_vec[env_i].item(),
                        elevation_deg=elev_vec[env_i].item(),
                        background_is_sky=bg_is_sky[env_i].item(),
                        step=step, env_idx=env_i, agent_id=agent_id,
                    )
                    video_frames[agent_id].append(frame)

        # Progress
        if (step + 1) % 100 == 0:
            avg_yolo_ms = total_yolo_time / max(step + 1, 1) * 1000
            print(f"  Step {step+1}/{args_cli.num_steps}: "
                  f"{collector.num_frames} frames, "
                  f"{collector.num_misses} misses ({collector.miss_rate:.1%}), "
                  f"YOLO avg {avg_yolo_ms:.1f}ms/step")

    # --- Results ---
    print(f"\n[CLUTTER] Collection complete: {collector.num_frames} frames, "
          f"{collector.num_misses} misses ({collector.miss_rate:.1%})")

    # --- Plots ---
    output_path = Path(args_cli.output)
    report_dir = output_path.parent / (output_path.stem + "_report")
    collector.generate_plots(report_dir)

    # --- Video ---
    if record_video:
        report_dir.mkdir(parents=True, exist_ok=True)
        for agent_id, frames in video_frames.items():
            if not frames:
                continue
            h, w = frames[0].shape[:2]
            video_path = report_dir / f"clutter_fn_{agent_id}_{background_type}.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args_cli.video_fps, (w, h))
            for frame in frames:
                writer.write(frame)
            writer.release()
            print(f"[CLUTTER] Video saved: {video_path} ({len(frames)} frames)")

    # --- Save JSON ---
    binned = collector.compute_binned_miss_rates()
    output_data = {
        "metadata": {
            "background_type": background_type,
            "gimbal_locked": True,
            "yolo_model": args_cli.yolo_model,
            "yolo_conf_threshold": args_cli.yolo_conf,
            "iou_threshold": args_cli.iou_threshold,
            "num_envs": args_cli.num_envs,
            "num_steps": args_cli.num_steps,
            "num_agents": num_agents,
            "total_frames": collector.num_frames,
            "total_misses": collector.num_misses,
            "overall_miss_rate": collector.miss_rate,
            "experiment": args_cli.experiment,
        },
        "binned_miss_rates": binned,
        "raw_samples": [asdict(s) for s in collector.samples],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[CLUTTER] Saved to {output_path} ({len(collector.samples)} raw samples)")

    # --- Timing ---
    print(f"[CLUTTER] YOLO total: {total_yolo_time:.1f}s, "
          f"avg: {total_yolo_time / args_cli.num_steps * 1000:.1f}ms/step")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
