# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Triangulation module for iris_ma6.

This module provides multi-camera multi-target triangulation with uncertainty estimation.
Key features:
- Validity-based returns (NaN + is_valid mask) instead of sentinel fallback values
- Per-camera-target valid mask support [N, C, T]
- Condition number checking with configurable threshold
- Observability checks (minimum cameras, well-conditioned geometry)

Tensor Conventions:
- N: number of parallel environments
- C: number of cameras per environment
- T: number of targets per environment

Coordinate Frames:
- World frame: ENU (East-North-Up)
- Body frame: FLU (Forward-Left-Up)
- Camera frame: RDF (Right-Down-Forward) - OpenCV convention
- Quaternion: wxyz (scalar-first)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple

import isaaclab.utils.math as math_utils

from .triangulation_cfg import TriangulationCfg


# =============================================================================
# Data Structures
# =============================================================================


class TriangulationResult(NamedTuple):
    """Result of triangulation operation.

    Attributes:
        position: [N, T, 3] Triangulated position (NaN where invalid)
        covariance: [N, T, 3, 3] Covariance matrix (NaN where invalid)
        quality_metric: [N, T] Quality metric value (NaN where invalid)
        is_valid: [N, T] Validity mask
        condition_number: [N, T] Condition number of normal matrix
        num_valid_cameras: [N, T] Number of valid cameras per target
    """

    position: torch.Tensor
    covariance: torch.Tensor
    quality_metric: torch.Tensor
    is_valid: torch.Tensor
    condition_number: torch.Tensor
    num_valid_cameras: torch.Tensor


# =============================================================================
# Utility Functions
# =============================================================================


def skew(v: torch.Tensor) -> torch.Tensor:
    """Create skew-symmetric matrix from 3D vector(s).

    Args:
        v: [..., 3] tensor

    Returns:
        [..., 3, 3] skew-symmetric matrix
    """
    shape = v.shape[:-1]
    device = v.device
    dtype = v.dtype

    zeros = torch.zeros(*shape, device=device, dtype=dtype)

    skew_mat = torch.stack(
        [
            torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )

    return skew_mat


def rotation_matrix_from_euler(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor, order: str = "ZYX"
) -> torch.Tensor:
    """Create rotation matrix from Euler angles (batched).

    Args:
        roll, pitch, yaw: [...] tensors of angles in radians
        order: rotation order (e.g., 'ZYX' means Rz(yaw) @ Ry(pitch) @ Rx(roll))

    Returns:
        [..., 3, 3] rotation matrices
    """
    shape = roll.shape
    device = roll.device
    dtype = roll.dtype

    cx, sx = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cz, sz = torch.cos(yaw), torch.sin(yaw)

    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)

    # Rotation matrices
    Rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )

    Ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )

    Rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    m = {"X": Rx, "Y": Ry, "Z": Rz}
    R = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3).clone()

    for ax in order:
        R = R @ m[ax]

    return R


def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        quat: [..., 4] quaternion (w, x, y, z order)

    Returns:
        [..., 3, 3] rotation matrix
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    R = torch.stack(
        [
            torch.stack(
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                dim=-1,
            ),
            torch.stack(
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                dim=-1,
            ),
            torch.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
                dim=-1,
            ),
        ],
        dim=-2,
    )

    return R


# =============================================================================
# View Construction
# =============================================================================


def build_camera_transforms(
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    gimbal_yaw: torch.Tensor,
    gimbal_pitch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build camera transformation matrices from robot and gimbal state.

    The transformation chain fuses ego pose and gimbal orientation:
        R_wc = R_wb(q_wb) @ R_bg(alpha, beta) @ R_gc

    Args:
        robot_pos: [N, C, 3] Robot positions in world frame
        robot_quat: [N, C, 4] Robot orientations (wxyz quaternion)
        gimbal_yaw: [N, C] Gimbal yaw angles (radians)
        gimbal_pitch: [N, C] Gimbal pitch angles (radians)

    Returns:
        R_wc: [N, C, 3, 3] World-to-camera rotation matrices
        t_wc: [N, C, 3] Camera positions in world frame
        e_alpha: [N, C, 3] Gimbal yaw axis in camera frame
        e_beta: [N, C, 3] Gimbal pitch axis in camera frame
    """
    N, C = robot_pos.shape[:2]
    device = robot_pos.device
    dtype = robot_pos.dtype

    # World to body rotation from quaternion
    R_wb = quat_to_rotation_matrix(robot_quat)  # [N, C, 3, 3]

    # Body to gimbal: Z(yaw) then Y(pitch)
    zeros = torch.zeros(N, C, device=device, dtype=dtype)
    R_bg = rotation_matrix_from_euler(zeros, gimbal_pitch, gimbal_yaw, "ZYX")

    # Gimbal to camera (ENU to RDF - OpenCV convention)
    R_gc = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    ).T.expand(N, C, 3, 3)

    # Full transformation: world to camera
    R_wc = R_wb @ R_bg @ R_gc
    t_wc = robot_pos

    # Gimbal axes in camera frame (for Jacobian computation)
    R_bc = R_bg @ R_gc
    z_b = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(N, C, 3)
    e_alpha = torch.bmm(
        R_bc.view(N * C, 3, 3).transpose(-1, -2), z_b.view(N * C, 3, 1)
    ).view(N, C, 3)
    e_beta = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype).expand(N, C, 3)

    return R_wc, t_wc, e_alpha, e_beta


# =============================================================================
# Ray Direction Computation
# =============================================================================


def get_ray_directions_from_bbox(
    bbox_2d: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    gimbal_yaw: torch.Tensor,
    gimbal_pitch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ray directions from 2D bounding box centers.

    CRITICAL: Ray computation fuses ego pose and gimbal orientation.
    The ray direction in world frame depends on:
    1. Ego pose: Robot position and orientation quaternion
    2. Gimbal angles: Yaw and pitch
    3. Bbox center: Pixel coordinates from detection

    Args:
        bbox_2d: [N, C, T, 4] 2D bounding box (xywh format)
        camera_intrinsics: [N, C, 3, 3] Camera intrinsic matrices
        robot_pos: [N, C, 3] Robot positions
        robot_quat: [N, C, 4] Robot orientations (wxyz)
        gimbal_yaw: [N, C] Gimbal yaw angles
        gimbal_pitch: [N, C] Gimbal pitch angles

    Returns:
        ray_dirs: [N, C, T, 3] Ray directions in world frame (normalized)
        R_wc: [N, C, 3, 3] World-to-camera rotation matrices
    """
    N, C, T = bbox_2d.shape[:3]
    device = bbox_2d.device
    dtype = bbox_2d.dtype

    # Build camera transforms (fuses ego pose + gimbal)
    R_wc, _, _, _ = build_camera_transforms(robot_pos, robot_quat, gimbal_yaw, gimbal_pitch)

    # Extract intrinsic parameters
    fx = camera_intrinsics[..., 0, 0].unsqueeze(-1)  # [N, C, 1]
    fy = camera_intrinsics[..., 1, 1].unsqueeze(-1)
    cx = camera_intrinsics[..., 0, 2].unsqueeze(-1)
    cy = camera_intrinsics[..., 1, 2].unsqueeze(-1)

    # Convert bbox center to normalized image coordinates
    x_n = (bbox_2d[..., 0] - cx) / fx  # [N, C, T]
    y_n = (bbox_2d[..., 1] - cy) / fy

    # Form direction vectors in camera frame (RDF convention: Z forward)
    dirs_camera = torch.stack([x_n, y_n, torch.ones_like(x_n)], dim=-1)  # [N, C, T, 3]

    # Normalize direction vectors
    dirs_camera_norm = dirs_camera / (
        torch.norm(dirs_camera, dim=-1, keepdim=True) + 1e-12
    )

    # Rotate to world frame: d_w = R_wc @ d_c
    R_wc_exp = R_wc.unsqueeze(2).expand(N, C, T, 3, 3)
    dirs_world = torch.matmul(R_wc_exp, dirs_camera_norm.unsqueeze(-1)).squeeze(-1)

    return dirs_world, R_wc


# =============================================================================
# Position Triangulation
# =============================================================================


def triangulate_targets(
    camera_positions: torch.Tensor,
    ray_directions: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: TriangulationCfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triangulate target positions from multiple camera rays.

    Uses the midpoint method: minimizes sum of squared perpendicular distances
    to all valid rays.

    Args:
        camera_positions: [N, C, 3] Camera positions in world frame
        ray_directions: [N, C, T, 3] Ray directions (normalized) in world frame
        valid_mask: [N, C, T] Boolean mask of valid observations
        cfg: Triangulation configuration

    Returns:
        X_tri: [N, T, 3] Triangulated positions (NaN where invalid)
        is_valid: [N, T] Validity mask
        condition_number: [N, T] Condition number of normal matrix
        num_valid_cameras: [N, T] Number of valid cameras per target
    """
    N, C, T = ray_directions.shape[:3]
    device = ray_directions.device
    dtype = ray_directions.dtype

    # Initialize outputs with NaN (undefined values)
    X_tri = torch.full((N, T, 3), float("nan"), device=device, dtype=dtype)
    condition_number = torch.full((N, T), float("nan"), device=device, dtype=dtype)

    # Count valid cameras per target: [N, T]
    num_valid_cameras = valid_mask.sum(dim=1).to(dtype)

    # Normalize direction vectors
    dirs_norm = ray_directions / (torch.norm(ray_directions, dim=-1, keepdim=True) + 1e-12)

    # Apply valid mask: zero out invalid rays
    # Shape: [N, C, T, 1] for broadcasting
    mask_expanded = valid_mask.unsqueeze(-1).to(dtype)
    dirs_masked = dirs_norm * mask_expanded

    # Compute I - d @ d^T for each ray: [N, C, T, 3, 3]
    I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3).expand(N, C, T, 3, 3)
    dd_T = torch.matmul(dirs_masked.unsqueeze(-1), dirs_masked.unsqueeze(-2))
    I_minus_dd = (I - dd_T) * mask_expanded.unsqueeze(-1)

    # Sum over cameras: A = sum_c (I - d_c @ d_c^T) [N, T, 3, 3]
    A = I_minus_dd.sum(dim=1)

    # Expand camera positions: [N, C, 1, 3] -> [N, C, T, 3]
    pts_exp = camera_positions.unsqueeze(2).expand(N, C, T, 3)

    # Compute (I - d @ d^T) @ p for each ray: [N, C, T, 3]
    I_minus_dd_p = torch.matmul(I_minus_dd, pts_exp.unsqueeze(-1)).squeeze(-1)

    # Sum over cameras: b = sum_c (I - d_c @ d_c^T) @ p_c [N, T, 3]
    b = I_minus_dd_p.sum(dim=1)

    # Add regularization for numerical stability
    A_reg = A + cfg.regularization_eps * torch.eye(3, device=device, dtype=dtype)

    # Compute condition number via SVD
    svd_vals = torch.linalg.svdvals(A_reg)  # [N, T, 3]
    cond_num = svd_vals[..., 0] / (svd_vals[..., -1] + 1e-12)
    condition_number = cond_num

    # Determine validity
    is_solvable = num_valid_cameras >= cfg.min_cameras_required
    is_well_conditioned = cond_num < cfg.condition_threshold
    is_valid = is_solvable & is_well_conditioned

    # Solve only for valid entries
    valid_flat = is_valid.reshape(N * T)
    valid_indices = torch.where(valid_flat)[0]

    if valid_indices.numel() > 0:
        A_flat = A_reg.reshape(N * T, 3, 3)
        b_flat = b.reshape(N * T, 3)

        # Solve A @ X = b for valid entries
        X_solved = torch.linalg.solve(
            A_flat[valid_indices], b_flat[valid_indices].unsqueeze(-1)
        ).squeeze(-1)

        # Write results back
        X_tri_flat = X_tri.reshape(N * T, 3)
        X_tri_flat[valid_indices] = X_solved
        X_tri = X_tri_flat.reshape(N, T, 3)

    # Additional check: behind-camera detection
    # A point is invalid if it's behind ALL valid cameras
    if is_valid.any():
        # Vector from camera to triangulated point: [N, C, T, 3]
        cam_to_point = X_tri.unsqueeze(1) - camera_positions.unsqueeze(2)

        # Dot product with ray direction (positive = in front)
        dot_product = (cam_to_point * dirs_norm).sum(dim=-1)  # [N, C, T]

        # Point is in front of camera c if dot > 0
        is_in_front = dot_product > 0  # [N, C, T]

        # Point must be in front of at least one valid camera
        is_in_front_masked = is_in_front & valid_mask
        any_in_front = is_in_front_masked.any(dim=1)  # [N, T]

        # Update validity
        is_valid = is_valid & any_in_front

        # Set invalid positions to NaN
        X_tri = torch.where(is_valid.unsqueeze(-1), X_tri, torch.tensor(float("nan"), device=device, dtype=dtype))

    return X_tri, is_valid, condition_number, num_valid_cameras


# =============================================================================
# Jacobian Computation
# =============================================================================


def compute_projection_jacobian(
    Xc: torch.Tensor, K: torch.Tensor, min_z: float = 1e-6
) -> torch.Tensor:
    """Compute Jacobian of projection with respect to camera-frame point.

    Args:
        Xc: [..., 3] Points in camera frame
        K: [..., 3, 3] Camera intrinsic matrices
        min_z: Minimum Z value to prevent division by zero

    Returns:
        [..., 2, 3] Jacobian matrix du/dXc
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]

    X, Y, Z = Xc[..., 0], Xc[..., 1], Xc[..., 2]

    # Clamp Z to avoid division by zero
    Z = torch.where(torch.abs(Z) >= min_z, Z, torch.ones_like(Z) * min_z)
    Z2 = Z * Z

    Ju = torch.stack(
        [
            torch.stack([fx / Z, torch.zeros_like(Z), -fx * X / Z2], dim=-1),
            torch.stack([torch.zeros_like(Z), fy / Z, -fy * Y / Z2], dim=-1),
        ],
        dim=-2,
    )

    return Ju


# =============================================================================
# Covariance Computation
# =============================================================================


def compute_triangulation_covariance(
    X_target: torch.Tensor,
    robot_positions: torch.Tensor,
    robot_quats: torch.Tensor,
    gimbal_yaws: torch.Tensor,
    gimbal_pitches: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    triangulation_valid: torch.Tensor,
    cfg: TriangulationCfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute triangulation covariance for target positions.

    Uses first-order uncertainty propagation with multiple uncertainty sources:
    - Pixel detection noise
    - Camera position uncertainty
    - Camera orientation uncertainty
    - Gimbal angle uncertainty

    Args:
        X_target: [N, T, 3] Target positions (can be GT or triangulated)
        robot_positions: [N, C, 3] Camera positions
        robot_quats: [N, C, 4] Camera orientations (wxyz)
        gimbal_yaws: [N, C] Gimbal yaw angles
        gimbal_pitches: [N, C] Gimbal pitch angles
        camera_intrinsics: [N, C, 3, 3] Camera intrinsic matrices
        valid_mask: [N, C, T] Valid observation mask
        triangulation_valid: [N, T] Validity from triangulation step
        cfg: Triangulation configuration

    Returns:
        Sigma_X: [N, T, 3, 3] Covariance matrices (NaN where invalid)
        quality_metric: [N, T] Quality metric (NaN where invalid)
        is_valid: [N, T] Combined validity mask
    """
    N, C = robot_positions.shape[:2]
    T = X_target.shape[1]
    device = X_target.device
    dtype = X_target.dtype

    # Initialize outputs with NaN
    Sigma_X = torch.full((N, T, 3, 3), float("nan"), device=device, dtype=dtype)
    quality_metric = torch.full((N, T), float("nan"), device=device, dtype=dtype)

    # Build camera transforms
    R_wc, t_wc, e_alpha, e_beta = build_camera_transforms(
        robot_positions, robot_quats, gimbal_yaws, gimbal_pitches
    )

    # Build covariance matrices for uncertainty sources
    Sigma_pix = (
        torch.eye(2, device=device, dtype=dtype).view(1, 1, 2, 2).expand(N, C, 2, 2)
        * (cfg.pix_std**2)
    )

    Sigma_twb = None
    Sigma_phiwb = None
    Sigma_alpha = None
    Sigma_beta = None

    if cfg.include_pose_uncertainty:
        Sigma_twb = (
            torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(N, C, 3, 3)
            * (cfg.pos_std**2)
        )
        Sigma_phiwb = (
            torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(N, C, 3, 3)
            * (cfg.ori_std**2)
        )

    if cfg.include_gimbal_uncertainty:
        Sigma_alpha = (
            torch.ones(N, C, 1, 1, device=device, dtype=dtype) * (cfg.gimbal_std**2)
        )
        Sigma_beta = (
            torch.ones(N, C, 1, 1, device=device, dtype=dtype) * (cfg.gimbal_std**2)
        )

    # Compute Jacobians for all target-camera pairs
    # Expand target positions: [N, T, 1, 3]
    X_exp = X_target.unsqueeze(2)
    R_wc_exp = R_wc.unsqueeze(1)  # [N, 1, C, 3, 3]
    t_wc_exp = t_wc.unsqueeze(1)  # [N, 1, C, 3]
    K_exp = camera_intrinsics.unsqueeze(1)  # [N, 1, C, 3, 3]

    # Transform targets to camera frame: [N, T, C, 3]
    X_rel = X_exp - t_wc_exp
    Xc = torch.matmul(R_wc_exp.transpose(-1, -2), X_rel.unsqueeze(-1)).squeeze(-1)

    # Projection Jacobian: [N, T, C, 2, 3]
    Ju_Xc = compute_projection_jacobian(
        Xc, K_exp.squeeze(1).unsqueeze(1).expand(N, T, C, 3, 3), cfg.min_z_threshold
    )

    # Position Jacobian: J_X = du/dXc @ R_wc^T
    JX = torch.matmul(Ju_Xc, R_wc_exp.transpose(-1, -2))  # [N, T, C, 2, 3]

    # Apply valid mask to Jacobians
    mask_exp = valid_mask.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).to(dtype)  # [N, T, C, 1, 1]
    JX_masked = JX * mask_exp

    # Build nuisance parameter Jacobians
    J_blocks = []
    Sig_blocks = []

    if cfg.include_pose_uncertainty and Sigma_twb is not None:
        # Position: J_t = -du/dXc @ R_wc^T
        J_t = torch.matmul(Ju_Xc, -R_wc_exp.transpose(-1, -2)) * mask_exp
        J_blocks.append(J_t)
        Sig_blocks.append(Sigma_twb)

    if cfg.include_pose_uncertainty and Sigma_phiwb is not None:
        # Orientation: J_phi = -du/dXc @ [Xc]_x @ R_wc^T
        skew_Xc = skew(Xc)
        J_phi = torch.matmul(
            Ju_Xc, torch.matmul(-skew_Xc, R_wc_exp.transpose(-1, -2))
        ) * mask_exp
        J_blocks.append(J_phi)
        Sig_blocks.append(Sigma_phiwb)

    if cfg.include_gimbal_uncertainty and Sigma_alpha is not None:
        # Gimbal yaw
        e_alpha_exp = e_alpha.unsqueeze(1)  # [N, 1, C, 3]
        skew_Xc = skew(Xc)
        J_alpha = torch.matmul(
            Ju_Xc, torch.matmul(-skew_Xc, e_alpha_exp.unsqueeze(-1))
        ) * mask_exp
        J_blocks.append(J_alpha)
        Sig_blocks.append(Sigma_alpha)

    if cfg.include_gimbal_uncertainty and Sigma_beta is not None:
        # Gimbal pitch
        e_beta_exp = e_beta.unsqueeze(1)  # [N, 1, C, 3]
        skew_Xc = skew(Xc)
        J_beta = torch.matmul(
            Ju_Xc, torch.matmul(-skew_Xc, e_beta_exp.unsqueeze(-1))
        ) * mask_exp
        J_blocks.append(J_beta)
        Sig_blocks.append(Sigma_beta)

    # Concatenate nuisance Jacobians
    if len(J_blocks) > 0:
        J_theta = torch.cat(J_blocks, dim=-1)  # [N, T, C, 2, M]
        M = J_theta.shape[-1]

        # Build block-diagonal nuisance covariance
        Sigma_theta = torch.zeros(N, C, M, M, device=device, dtype=dtype)
        col_idx = 0
        for sig_block in Sig_blocks:
            block_size = sig_block.shape[-1]
            Sigma_theta[:, :, col_idx : col_idx + block_size, col_idx : col_idx + block_size] = sig_block
            col_idx += block_size
    else:
        J_theta = torch.zeros(N, T, C, 2, 0, device=device, dtype=dtype)
        Sigma_theta = torch.zeros(N, C, 0, 0, device=device, dtype=dtype)
        M = 0

    # Build stacked matrices for covariance computation
    JX_stacked = JX_masked.reshape(N, T, 2 * C, 3)

    # Block diagonal J_theta: [N, T, 2C, CM]
    J_theta_stacked = torch.zeros(N, T, 2 * C, C * M, device=device, dtype=dtype) if M > 0 else None

    if M > 0:
        for c in range(C):
            J_theta_stacked[:, :, 2 * c : 2 * c + 2, c * M : (c + 1) * M] = J_theta[:, :, c, :, :]

    # Block diagonal weight matrix: [N, 2C, 2C]
    W_i = torch.linalg.inv(Sigma_pix)  # [N, C, 2, 2]
    W_block = torch.zeros(N, 2 * C, 2 * C, device=device, dtype=dtype)
    for c in range(C):
        W_block[:, 2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = W_i[:, c]

    # Apply valid mask to weight matrix
    for c in range(C):
        # Get mask for this camera: [N, T]
        camera_mask = valid_mask[:, c, :]  # [N, T]
        # We need to zero out weights for invalid cameras
        # But this is per-target, so we handle it in the loop below

    W_block_exp = W_block.unsqueeze(1).expand(N, T, 2 * C, 2 * C)  # [N, T, 2C, 2C]

    # Block diagonal pixel covariance: [N, 2C, 2C]
    Sigma_z = torch.zeros(N, 2 * C, 2 * C, device=device, dtype=dtype)
    for c in range(C):
        Sigma_z[:, 2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = Sigma_pix[:, c]
    Sigma_z_exp = Sigma_z.unsqueeze(1).expand(N, T, 2 * C, 2 * C)  # [N, T, 2C, 2C]

    # Compute covariance for each target
    # Normal matrix: A = JX^T @ W @ JX [N, T, 3, 3]
    A = torch.matmul(torch.matmul(JX_stacked.transpose(-1, -2), W_block_exp), JX_stacked)

    # Add regularization
    A_reg = A + cfg.regularization_eps * torch.eye(3, device=device, dtype=dtype)

    # Check condition number
    svd_vals = torch.linalg.svdvals(A_reg)
    cond_num = svd_vals[..., 0] / (svd_vals[..., -1] + 1e-12)
    is_well_conditioned = cond_num < cfg.condition_threshold

    # Combined validity
    is_valid = triangulation_valid & is_well_conditioned

    # Compute residual covariance
    if M > 0:
        # Block diagonal nuisance covariance: [N, CM, CM]
        Sigma_theta_block = torch.zeros(N, C * M, C * M, device=device, dtype=dtype)
        for c in range(C):
            Sigma_theta_block[:, c * M : (c + 1) * M, c * M : (c + 1) * M] = Sigma_theta[:, c]
        Sigma_theta_exp = Sigma_theta_block.unsqueeze(1).expand(N, T, C * M, C * M)  # [N, T, CM, CM]

        # S = Sigma_z + J_theta @ Sigma_theta @ J_theta^T
        S_resid = Sigma_z_exp + torch.matmul(
            torch.matmul(J_theta_stacked, Sigma_theta_exp),
            J_theta_stacked.transpose(-1, -2),
        )
    else:
        S_resid = Sigma_z_exp.clone()  # [N, T, 2C, 2C]

    # Compute covariance: Sigma_X = A^{-1} @ JX^T @ W @ S @ W @ JX @ A^{-T}
    valid_flat = is_valid.reshape(N * T)
    valid_indices = torch.where(valid_flat)[0]

    if valid_indices.numel() > 0:
        A_reg_flat = A_reg.reshape(N * T, 3, 3)
        JX_stacked_flat = JX_stacked.reshape(N * T, 2 * C, 3)
        S_resid_flat = S_resid.reshape(N * T, 2 * C, 2 * C)
        W_block_flat = W_block_exp.reshape(N * T, 2 * C, 2 * C)

        # Compute only for valid entries
        A_inv = torch.linalg.inv(A_reg_flat[valid_indices])
        W_valid = W_block_flat[valid_indices]

        middle = torch.matmul(
            torch.matmul(JX_stacked_flat[valid_indices].transpose(-1, -2), W_valid),
            torch.matmul(S_resid_flat[valid_indices], torch.matmul(W_valid, JX_stacked_flat[valid_indices])),
        )

        Sigma_valid = torch.matmul(torch.matmul(A_inv, middle), A_inv.transpose(-1, -2))

        # Write back results
        Sigma_X_flat = Sigma_X.reshape(N * T, 3, 3)
        Sigma_X_flat[valid_indices] = Sigma_valid
        Sigma_X = Sigma_X_flat.reshape(N, T, 3, 3)

        # Compute quality metric
        quality_flat = quality_metric.reshape(N * T)

        if cfg.quality_metric == "trace":
            quality_flat[valid_indices] = torch.diagonal(Sigma_valid, dim1=-2, dim2=-1).sum(dim=-1)
        elif cfg.quality_metric == "sqrt_trace":
            quality_flat[valid_indices] = torch.sqrt(
                torch.diagonal(Sigma_valid, dim1=-2, dim2=-1).sum(dim=-1)
            )
        elif cfg.quality_metric == "det":
            quality_flat[valid_indices] = torch.linalg.det(Sigma_valid).abs().pow(1.0 / 3.0)
        elif cfg.quality_metric == "max_eig":
            eigvals = torch.linalg.eigvalsh(Sigma_valid)
            quality_flat[valid_indices] = eigvals[..., -1]

        quality_metric = quality_flat.reshape(N, T)

    # Additional validity check: positive definite covariance
    if is_valid.any():
        diag_vals = torch.diagonal(Sigma_X, dim1=-2, dim2=-1)  # [N, T, 3]
        is_positive_definite = (diag_vals > 0).all(dim=-1)
        is_valid = is_valid & is_positive_definite

        # Set invalid entries to NaN
        Sigma_X = torch.where(
            is_valid.unsqueeze(-1).unsqueeze(-1),
            Sigma_X,
            torch.tensor(float("nan"), device=device, dtype=dtype),
        )
        quality_metric = torch.where(
            is_valid, quality_metric, torch.tensor(float("nan"), device=device, dtype=dtype)
        )

    return Sigma_X, quality_metric, is_valid


# =============================================================================
# Main Entry Point
# =============================================================================


def compute_full_triangulation(
    bbox_2d: torch.Tensor,
    bbox_valid: torch.Tensor,
    robot_positions: torch.Tensor,
    robot_quats: torch.Tensor,
    gimbal_yaws: torch.Tensor,
    gimbal_pitches: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    cfg: TriangulationCfg,
    target_positions_gt: Optional[torch.Tensor] = None,
) -> TriangulationResult:
    """Perform full triangulation with uncertainty estimation.

    This is the main entry point for the triangulation module.

    Args:
        bbox_2d: [N, C, T, 4] 2D bounding boxes (xywh format)
        bbox_valid: [N, C, T] Valid detection mask
        robot_positions: [N, C, 3] Robot/camera positions
        robot_quats: [N, C, 4] Robot orientations (wxyz)
        gimbal_yaws: [N, C] Gimbal yaw angles
        gimbal_pitches: [N, C] Gimbal pitch angles
        camera_intrinsics: [N, C, 3, 3] Camera intrinsic matrices
        cfg: Triangulation configuration
        target_positions_gt: [N, T, 3] Optional GT positions for covariance
            (if None, uses triangulated positions)

    Returns:
        TriangulationResult containing all outputs with validity masks
    """
    # Step 1: Compute ray directions
    ray_dirs, _ = get_ray_directions_from_bbox(
        bbox_2d, camera_intrinsics, robot_positions, robot_quats, gimbal_yaws, gimbal_pitches
    )

    # Step 2: Triangulate positions
    X_tri, tri_valid, cond_num, num_valid_cams = triangulate_targets(
        robot_positions, ray_dirs, bbox_valid, cfg
    )

    # Step 3: Compute covariance
    # Use GT positions if provided, otherwise use triangulated positions
    X_for_cov = target_positions_gt if target_positions_gt is not None else X_tri

    # For covariance, we need valid triangulation OR valid GT
    cov_target_valid = tri_valid
    if target_positions_gt is not None:
        # If GT provided, all GT positions are valid for covariance computation
        cov_target_valid = torch.ones_like(tri_valid)

    Sigma_X, quality_metric, cov_valid = compute_triangulation_covariance(
        X_for_cov,
        robot_positions,
        robot_quats,
        gimbal_yaws,
        gimbal_pitches,
        camera_intrinsics,
        bbox_valid,
        cov_target_valid,
        cfg,
    )

    # Combined validity: triangulation valid AND covariance valid
    # (unless GT was provided, then just covariance validity matters)
    if target_positions_gt is not None:
        final_valid = cov_valid
    else:
        final_valid = tri_valid & cov_valid

    return TriangulationResult(
        position=X_tri,
        covariance=Sigma_X,
        quality_metric=quality_metric,
        is_valid=final_valid,
        condition_number=cond_num,
        num_valid_cameras=num_valid_cams,
    )
