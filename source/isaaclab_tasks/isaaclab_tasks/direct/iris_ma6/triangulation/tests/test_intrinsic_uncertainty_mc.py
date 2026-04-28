#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Monte Carlo validation for the intrinsic-covariance (Sigma_K) block in
iris_ma6 triangulation. Backs ticket mas/029.

Sets up a 3-camera scenario (target at known position), draws Monte Carlo
samples by perturbing each camera's intrinsics (fx, fy, cx, cy) according
to a known Sigma_K, projects the target through the perturbed intrinsics,
adds zero pixel noise (we want intrinsic-only uncertainty isolated), then
triangulates with the NOMINAL intrinsics. The empirical 3D covariance of
the resulting positions should match the analytic covariance produced by
`compute_triangulation_covariance(..., Sigma_K_per_camera=Sigma_K,
include_intrinsics_uncertainty=True)`.

Pass criterion: |trace(Sigma_X_analytic) - trace(Sigma_X_empirical)| /
trace(Sigma_X_empirical) < 0.15 over ≥ 5000 samples. The 15% slack
accommodates Monte Carlo finite-sample noise; with 10k samples we
typically see < 5%.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MC validation for triangulation Sigma_K")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--n-samples", type=int, default=10000)
parser.add_argument("--zoom-cmd", type=float, default=4.0,
                    help="Operator zoom command at which to evaluate Sigma_K (1.0..5.0)")
parser.add_argument("--target-distance", type=float, default=80.0)
parser.add_argument("--test-verbose", action="store_true")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import sys
import traceback

import torch

from isaaclab_tasks.direct.iris_ma6.triangulation import (
    TriangulationCfg,
    compute_full_triangulation,
    compute_triangulation_covariance,
    compute_sigma_K_from_zoom,
    compute_sigma_K_constant,
    get_ray_directions_from_bbox,
    triangulate_targets,
)
from isaaclab_tasks.direct.iris_ma6.triangulation.triangulation import (
    build_camera_transforms,
)


VERBOSE = args_cli.test_verbose


def create_yaw_quat(yaw: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        device=device, dtype=torch.float32,
    )


def project_through_intrinsics(
    X_w: torch.Tensor,            # [3]
    cam_pos: torch.Tensor,        # [N, C, 3]
    R_wc: torch.Tensor,           # [N, C, 3, 3]
    K: torch.Tensor,              # [N, C, 3, 3]
):
    """Project a single world point through per-sample intrinsics."""
    N, C = cam_pos.shape[:2]
    X_rel = X_w.view(1, 1, 3) - cam_pos                                 # [N, C, 3]
    Xc = torch.matmul(R_wc.transpose(-1, -2), X_rel.unsqueeze(-1)).squeeze(-1)  # [N, C, 3]
    Z = Xc[..., 2].clamp(min=1e-6)
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    u = fx * (Xc[..., 0] / Z) + cx
    v = fy * (Xc[..., 1] / Z) + cy
    return torch.stack([u, v], dim=-1), Xc[..., 2]  # [N, C, 2], depth


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print("=" * 80)
    print("MC validation: triangulation Sigma_K (mas/029)")
    print("=" * 80)
    print(f"Device:           {device}")
    print(f"N samples:        {args_cli.n_samples}")
    print(f"Zoom cmd:         {args_cli.zoom_cmd}")
    print(f"Target distance:  {args_cli.target_distance} m")

    N = args_cli.n_samples
    C = 3

    # Three cameras on a circle of radius = target_distance, at +5 m altitude,
    # all yawed to face origin where the target sits. Three azimuths span 120°
    # so triangulation geometry is well-conditioned.
    radius = args_cli.target_distance
    altitude = 5.0
    angles_deg = [0.0, 60.0, 120.0]
    cam_xyz = []
    cam_yaws = []
    for ang_deg in angles_deg:
        ang = math.radians(ang_deg)
        cam_xyz.append([radius * math.cos(ang), radius * math.sin(ang), altitude])
        # Body yaw so that body +X points from camera toward origin.
        cam_yaws.append(math.atan2(-math.sin(ang), -math.cos(ang)))
    cam_pos_nominal = torch.tensor(cam_xyz, device=device, dtype=torch.float32)  # [C, 3]
    cam_quat_nominal = torch.stack(
        [create_yaw_quat(y, device) for y in cam_yaws], dim=0
    )  # [C, 4]

    # Target sits at the origin (center of the camera ring) at the same altitude.
    X_target = torch.tensor([0.0, 0.0, altitude], device=device, dtype=torch.float32)

    # Nominal gimbal angles: zero (all rotation is in the body yaw above)
    gimbal_yaw_nom = torch.zeros(1, C, device=device)
    gimbal_roll_nom = torch.zeros(1, C, device=device)
    gimbal_pitch_nom = torch.zeros(1, C, device=device)

    # Nominal intrinsics (1x render-resolution focal at 1920x1080, fx=1053 from mrcal 1x)
    fx_nominal = 1053.044591
    K_nominal = torch.zeros(C, 3, 3, device=device)
    K_nominal[:, 0, 0] = fx_nominal
    K_nominal[:, 1, 1] = fx_nominal
    K_nominal[:, 0, 2] = 960.0
    K_nominal[:, 1, 2] = 540.0
    K_nominal[:, 2, 2] = 1.0

    # Build Sigma_K from the requested zoom command
    zoom_cmd_tensor = torch.full((1, C), args_cli.zoom_cmd, device=device)
    Sigma_K_one = compute_sigma_K_from_zoom(zoom_cmd_tensor)  # [1, C, 4, 4]
    sigma_fx = Sigma_K_one[0, 0, 0, 0].sqrt().item()
    sigma_fy = Sigma_K_one[0, 0, 1, 1].sqrt().item()
    sigma_cx = Sigma_K_one[0, 0, 2, 2].sqrt().item()
    sigma_cy = Sigma_K_one[0, 0, 3, 3].sqrt().item()
    print(
        f"Sigma_K @ cmd={args_cli.zoom_cmd}: "
        f"σ_fx={sigma_fx:.2f}, σ_fy={sigma_fy:.2f}, "
        f"σ_cx={sigma_cx:.2f}, σ_cy={sigma_cy:.2f} px"
    )

    # ---------------- Monte Carlo sampling ----------------
    # Perturb intrinsics independently per (sample, camera). cx/cy/fy treated
    # as independent Gaussians per-axis (matches the diagonal Sigma_K we use).
    K_pert = K_nominal.unsqueeze(0).expand(N, C, 3, 3).clone()
    K_pert[..., 0, 0] += torch.randn(N, C, device=device) * sigma_fx
    K_pert[..., 1, 1] += torch.randn(N, C, device=device) * sigma_fy
    K_pert[..., 0, 2] += torch.randn(N, C, device=device) * sigma_cx
    K_pert[..., 1, 2] += torch.randn(N, C, device=device) * sigma_cy

    # Build camera transforms (no pose/gimbal perturbation here — we want
    # intrinsic-only uncertainty isolated for validation).
    cam_pos_batch = cam_pos_nominal.unsqueeze(0).expand(N, C, 3)
    cam_quat_batch = cam_quat_nominal.unsqueeze(0).expand(N, C, 4)
    R_wc, _, _, _, _ = build_camera_transforms(
        cam_pos_batch,
        cam_quat_batch,
        gimbal_yaw_nom.expand(N, C),
        gimbal_roll_nom.expand(N, C),
        gimbal_pitch_nom.expand(N, C),
    )

    # Project target through perturbed intrinsics
    pixels, depth = project_through_intrinsics(X_target, cam_pos_batch, R_wc, K_pert)

    # Build bbox tensors
    T = 1
    bbox_2d = torch.zeros(N, C, T, 4, device=device)
    bbox_2d[:, :, 0, 0] = pixels[:, :, 0]
    bbox_2d[:, :, 0, 1] = pixels[:, :, 1]
    bbox_2d[:, :, 0, 2] = 50.0
    bbox_2d[:, :, 0, 3] = 50.0
    bbox_valid = (depth > 1e-3).unsqueeze(-1)

    # Triangulate with NOMINAL intrinsics (system doesn't know the perturbation)
    cfg = TriangulationCfg(
        pix_std=1e-3,                   # near-zero pixel noise so Sigma_K dominates
        include_pose_uncertainty=False,
        include_gimbal_uncertainty=False,
        include_intrinsics_uncertainty=True,
        intrinsics_std=1.0,             # unused when Sigma_K_per_camera provided
        condition_threshold=1e10,
        min_cameras_required=2,
    )
    K_nominal_batch = K_nominal.unsqueeze(0).expand(N, C, 3, 3)
    ray_dirs, _ = get_ray_directions_from_bbox(
        bbox_2d, K_nominal_batch, cam_pos_batch, cam_quat_batch,
        gimbal_yaw_nom.expand(N, C), gimbal_roll_nom.expand(N, C), gimbal_pitch_nom.expand(N, C),
    )
    X_tri, tri_valid, _, _ = triangulate_targets(
        cam_pos_batch, ray_dirs, bbox_valid, cfg
    )

    # Empirical covariance of valid samples
    valid_flat = tri_valid[:, 0]
    samples = X_tri[valid_flat, 0, :]                       # [V, 3]
    n_valid = samples.shape[0]
    print(f"Valid samples: {n_valid} / {N}")
    if n_valid < 100:
        print("Too few valid samples to validate.")
        return 1

    mean = samples.mean(dim=0)
    centered = samples - mean
    Sigma_emp = (centered.unsqueeze(-1) * centered.unsqueeze(-2)).mean(dim=0)  # [3, 3]
    trace_emp = Sigma_emp.diagonal().sum().item()

    # Mean residual vs ground truth (sanity check that triangulation is unbiased)
    err_mean = (mean - X_target).norm().item()

    # ---------------- Analytical covariance ----------------
    # Use the nominal scenario (1 env) and Sigma_K_per_camera matching MC σ.
    valid_mask = torch.ones(1, C, T, dtype=torch.bool, device=device)
    tri_valid_one = torch.ones(1, T, dtype=torch.bool, device=device)
    Sigma_X_analytic, _, is_valid_an = compute_triangulation_covariance(
        X_target=X_target.view(1, 1, 3),
        robot_positions=cam_pos_nominal.unsqueeze(0),
        robot_quats=cam_quat_nominal.unsqueeze(0),
        gimbal_yaws=gimbal_yaw_nom,
        gimbal_rolls=gimbal_roll_nom,
        gimbal_pitches=gimbal_pitch_nom,
        camera_intrinsics=K_nominal.unsqueeze(0),
        valid_mask=valid_mask,
        triangulation_valid=tri_valid_one,
        cfg=cfg,
        Sigma_K_per_camera=Sigma_K_one,
    )
    if not is_valid_an[0, 0]:
        print("Analytical covariance returned invalid.")
        return 1
    Sigma_X_an = Sigma_X_analytic[0, 0]
    trace_an = Sigma_X_an.diagonal().sum().item()

    rel = abs(trace_an - trace_emp) / max(trace_emp, 1e-12)

    print()
    print(f"Empirical mean residual:  {err_mean:.4f} m")
    print(f"Empirical trace(Σ_X):     {trace_emp:.4f} m²")
    print(f"Analytical trace(Σ_X):    {trace_an:.4f} m²")
    print(f"Relative trace error:     {rel*100:.1f} %")

    if VERBOSE:
        print("\nEmpirical Σ_X:")
        print(Sigma_emp.cpu().numpy())
        print("\nAnalytical Σ_X:")
        print(Sigma_X_an.cpu().numpy())

    threshold = 0.15
    print(f"\nPass threshold: ≤ {threshold*100:.0f} % relative error")
    if rel <= threshold:
        print("✓ PASS")
        return 0
    else:
        print("✗ FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
