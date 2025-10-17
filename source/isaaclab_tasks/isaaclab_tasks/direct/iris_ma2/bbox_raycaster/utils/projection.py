# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3D-to-2D projection utilities for batched bounding box raycasting."""

import torch
from typing import Optional

import isaaclab.utils.math as math_utils


def batch_transform_points(
    points_local: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """Transform points from local frame to world frame using poses.
    
    Args:
        points_local: Points in local frame. Shape (K, 3) or (T, K, 3).
        pos: Positions in world frame. Shape (N, T, 3).
        quat: Quaternions (w, x, y, z) in world frame. Shape (N, T, 4).
        out: Optional pre-allocated output tensor. Shape (N, T, K, 3).
        eps: Epsilon for quaternion normalization.
        
    Returns:
        Points in world frame. Shape (N, T, K, 3).
    """
    N, T = pos.shape[:2]
    
    # Handle different input shapes for points_local
    if points_local.ndim == 2:
        # Single set of points (K, 3) - expand to (T, K, 3)
        K = points_local.shape[0]
        points_local = points_local.unsqueeze(0).expand(T, -1, -1)
    else:
        # Already (T, K, 3)
        K = points_local.shape[1]
    
    # Normalize quaternions for numerical stability
    quat_norm = torch.norm(quat, dim=-1, keepdim=True)
    quat_normalized = quat / torch.clamp(quat_norm, min=eps)
    
    # Expand points to batch size: (N, T, K, 3)
    points_expanded = points_local.unsqueeze(0).expand(N, -1, -1, -1)
    
    # Rotate points: (N, T, K, 3)
    # Reshape for quat_apply: (N*T*K, 3) and (N*T*K, 4)
    points_flat = points_expanded.reshape(N * T * K, 3)
    quat_flat = quat_normalized.unsqueeze(2).expand(-1, -1, K, -1).reshape(N * T * K, 4)
    
    points_rotated_flat = math_utils.quat_apply(quat_flat, points_flat)
    points_rotated = points_rotated_flat.view(N, T, K, 3)
    
    # Add translation: (N, T, 1, 3) + (N, T, K, 3)
    pos_expanded = pos.unsqueeze(2)  # (N, T, 1, 3)
    
    if out is not None:
        torch.add(points_rotated, pos_expanded, out=out)
        return out
    else:
        return points_rotated + pos_expanded


def batch_transform_to_camera_frame(
    points_world: torch.Tensor,
    camera_pos: torch.Tensor,
    camera_quat: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """Transform points from world frame to camera frame.
    
    Args:
        points_world: Points in world frame. Shape (N, C, T, K, 3).
        camera_pos: Camera positions in world frame. Shape (N, C, 3).
        camera_quat: Camera quaternions (w, x, y, z) in world frame. Shape (N, C, 4).
        out: Optional pre-allocated output tensor. Shape (N, C, T, K, 3).
        eps: Epsilon for quaternion normalization.
        
    Returns:
        Points in camera frame. Shape (N, C, T, K, 3).
    """
    N, C, T, K = points_world.shape[:4]
    
    # Normalize quaternions
    quat_norm = torch.norm(camera_quat, dim=-1, keepdim=True)
    camera_quat_normalized = camera_quat / torch.clamp(quat_norm, min=eps)
    
    # Translate to camera origin: (N, C, 1, 1, 3)
    camera_pos_expanded = camera_pos.view(N, C, 1, 1, 3)
    points_rel = points_world - camera_pos_expanded
    
    # Rotate to camera frame using inverse quaternion
    # Reshape for batch rotation
    points_rel_flat = points_rel.reshape(N * C * T * K, 3)
    quat_flat = camera_quat_normalized.view(N, C, 1, 1, 4).expand(-1, -1, T, K, -1).reshape(N * C * T * K, 4)
    
    # Apply inverse rotation
    points_camera_flat = math_utils.quat_apply(
        math_utils.quat_inv(quat_flat),
        points_rel_flat
    )
    
    if out is not None:
        points_camera_flat.view(N, C, T, K, 3, out=out)
        return out
    else:
        return points_camera_flat.view(N, C, T, K, 3)


def batch_project_to_image_plane(
    points_camera: torch.Tensor,
    intrinsic_matrices: torch.Tensor,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D points in camera frame to 2D image plane.
    
    Args:
        points_camera: Points in camera frame. Shape (N, C, T, K, 3).
        intrinsic_matrices: Camera intrinsic matrices. Shape (N, C, 3, 3).
        eps: Epsilon to prevent division by zero.
        
    Returns:
        Tuple of:
            - pixels: Pixel coordinates (u, v). Shape (N, C, T, K, 2).
            - depths: Depth values (z-coordinate). Shape (N, C, T, K).
            - valid_mask: Boolean mask for valid projections. Shape (N, C, T, K).
    """
    N, C, T, K = points_camera.shape[:4]
    
    # Extract coordinates
    x = points_camera[..., 0]  # (N, C, T, K)
    y = points_camera[..., 1]
    z = points_camera[..., 2]
    
    # Check for points behind camera
    behind_camera = z <= eps
    
    # Safe depth (prevent division by zero)
    z_safe = torch.where(z > eps, z, torch.full_like(z, eps))
    
    # Extract intrinsic parameters
    fx = intrinsic_matrices[:, :, 0, 0].view(N, C, 1, 1)  # (N, C, 1, 1)
    fy = intrinsic_matrices[:, :, 1, 1].view(N, C, 1, 1)
    cx = intrinsic_matrices[:, :, 0, 2].view(N, C, 1, 1)
    cy = intrinsic_matrices[:, :, 1, 2].view(N, C, 1, 1)
    
    # Project to image plane
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    
    # Stack pixel coordinates
    pixels = torch.stack([u, v], dim=-1)  # (N, C, T, K, 2)
    
    # Valid projections are those in front of camera and finite
    valid_mask = ~behind_camera & torch.isfinite(pixels[..., 0]) & torch.isfinite(pixels[..., 1])
    
    return pixels, z, valid_mask


def check_points_in_fov(
    pixels: torch.Tensor,
    image_shapes: torch.Tensor
) -> torch.Tensor:
    """Check if projected points are within image bounds.
    
    Args:
        pixels: Pixel coordinates (u, v). Shape (N, C, T, K, 2).
        image_shapes: Image dimensions (height, width). Shape (N, C, 2).
        
    Returns:
        Boolean mask indicating points within FOV. Shape (N, C, T, K).
    """
    N, C = image_shapes.shape[:2]
    
    # Extract pixel coordinates
    u = pixels[..., 0]  # (N, C, T, K)
    v = pixels[..., 1]
    
    # Extract image dimensions
    img_h = image_shapes[:, :, 0].view(N, C, 1, 1)  # (N, C, 1, 1)
    img_w = image_shapes[:, :, 1].view(N, C, 1, 1)
    
    # Check bounds
    in_bounds_u = (u >= 0) & (u < img_w)
    in_bounds_v = (v >= 0) & (v < img_h)
    
    return in_bounds_u & in_bounds_v


def check_gimbal_lock(
    camera_quat: torch.Tensor,
    threshold: float = 0.99
) -> torch.Tensor:
    """Detect near-gimbal-lock orientations.
    
    Args:
        camera_quat: Camera quaternions. Shape (N, C, 4).
        threshold: Threshold for detecting gimbal lock (z-axis verticality).
        
    Returns:
        Boolean mask indicating potential gimbal lock. Shape (N, C).
    """
    # Convert to rotation matrix
    rot_mat = math_utils.quat_to_matrix(camera_quat)  # (N, C, 3, 3)
    
    # Check if camera z-axis (forward direction) is nearly vertical
    # Z-axis is the third column of rotation matrix
    z_axis = rot_mat[..., :, 2]  # (N, C, 3)
    z_vertical_component = torch.abs(z_axis[..., 2])  # Check world Z component
    
    near_gimbal_lock = z_vertical_component > threshold
    
    return near_gimbal_lock


def validate_projections(
    pixels: torch.Tensor,
    depths: torch.Tensor,
    projection_valid: torch.Tensor,
    image_shapes: torch.Tensor,
    min_depth: float = 0.01
) -> torch.Tensor:
    """Validate projected points.
    
    Args:
        pixels: Pixel coordinates. Shape (N, C, T, K, 2).
        depths: Depth values. Shape (N, C, T, K).
        projection_valid: Mask from projection step. Shape (N, C, T, K).
        image_shapes: Image dimensions. Shape (N, C, 2).
        min_depth: Minimum valid depth.
        
    Returns:
        Combined validity mask. Shape (N, C, T, K).
    """
    # Check depth
    valid_depth = depths > min_depth
    
    # Check FOV
    in_fov = check_points_in_fov(pixels, image_shapes)
    
    # Combine all validity checks
    valid = projection_valid & valid_depth & in_fov
    
    return valid