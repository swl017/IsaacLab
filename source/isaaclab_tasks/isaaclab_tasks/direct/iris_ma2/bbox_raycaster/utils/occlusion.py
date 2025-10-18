# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Occlusion detection using raycasting for batched bbox raycasting."""

import torch
import warp as wp
from typing import Literal, Optional

from isaaclab.utils.warp import raycast_mesh


def generate_occlusion_test_points(
    corners_world: torch.Tensor,
    pattern: Literal["9point", "corners_only", "center_only"],
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate test points from 3D bbox corners for occlusion detection.
    
    Args:
        corners_world: 8 corners in world frame. Shape (N, T, 8, 3).
        pattern: Pattern for test points.
        out: Optional pre-allocated output tensor.
        
    Returns:
        Test points in world frame. Shape depends on pattern:
            - "9point": (N, T, 9, 3)
            - "corners_only": (N, T, 4, 3)
            - "center_only": (N, T, 1, 3)
    """
    N, T = corners_world.shape[:2]
    
    if pattern == "center_only":
        # Only test center point
        center = corners_world.mean(dim=2, keepdim=True)  # (N, T, 1, 3)
        if out is not None:
            out.copy_(center)
            return out
        return center
    
    elif pattern == "corners_only":
        # Test 4 corners of the bottom face
        test_points = corners_world[:, :, :4, :]  # (N, T, 4, 3)
        if out is not None:
            out.copy_(test_points)
            return out
        return test_points
    
    elif pattern == "9point":
        # 4 corners + 4 edge midpoints + 1 center
        # Bottom face corners (indices 0-3)
        corners_bottom = corners_world[:, :, :4, :]  # (N, T, 4, 3)
        
        # Edge midpoints (average of adjacent corners)
        edge_01 = (corners_world[:, :, 0, :] + corners_world[:, :, 1, :]) / 2  # (N, T, 3)
        edge_12 = (corners_world[:, :, 1, :] + corners_world[:, :, 2, :]) / 2
        edge_23 = (corners_world[:, :, 2, :] + corners_world[:, :, 3, :]) / 2
        edge_30 = (corners_world[:, :, 3, :] + corners_world[:, :, 0, :]) / 2
        
        # Stack edge midpoints
        edges = torch.stack([edge_01, edge_12, edge_23, edge_30], dim=2)  # (N, T, 4, 3)
        
        # Center point
        center = corners_world.mean(dim=2, keepdim=True)  # (N, T, 1, 3)
        
        # Concatenate all test points
        test_points = torch.cat([corners_bottom, edges, center], dim=2)  # (N, T, 9, 3)
        
        if out is not None:
            out.copy_(test_points)
            return out
        return test_points
    
    else:
        raise ValueError(f"Unknown occlusion test pattern: {pattern}")


def batch_check_occlusion(
    camera_pos: torch.Tensor,
    test_points_world: torch.Tensor,
    target_positions: torch.Tensor,
    target_bbox_size: torch.Tensor,
    mesh: wp.Mesh,
    max_distance: float,
    visibility_threshold: float,
    tolerance_scale: float = 1.1,
    out_hits: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check occlusion using raycasting from cameras to target test points.
    
    Args:
        camera_pos: Camera positions in world frame. Shape (N, C, 3).
        test_points_world: Test points in world frame. Shape (N, T, K, 3).
        target_positions: Target center positions in world frame. Shape (N, T, 3).
        target_bbox_size: Target bbox dimensions. Shape (N, T, 3) or (T, 3).
        mesh: Warp mesh for raycasting.
        max_distance: Maximum raycasting distance.
        visibility_threshold: Fraction of points that must be visible.
        tolerance_scale: Tolerance multiplier for ray-target intersection.
        out_hits: Optional pre-allocated output tensor for hit positions.
        
    Returns:
        Tuple of:
            - visibility_mask: Boolean mask indicating visible targets. Shape (N, C, T).
            - visibility_ratio: Ratio of visible test points. Shape (N, C, T).
    """
    N, C = camera_pos.shape[:2]
    T, K = test_points_world.shape[1:3]
    
    # Expand camera positions: (N, C, 1, 1, 3) -> (N, C, T, K, 3)
    camera_pos_expanded = camera_pos.view(N, C, 1, 1, 3).expand(-1, -1, T, K, -1)
    
    # Expand test points: (N, 1, T, K, 3) -> (N, C, T, K, 3)
    test_points_expanded = test_points_world.unsqueeze(1).expand(-1, C, -1, -1, -1)
    
    # Compute ray directions
    ray_directions = test_points_expanded - camera_pos_expanded  # (N, C, T, K, 3)
    ray_distances = torch.norm(ray_directions, dim=-1, keepdim=True)  # (N, C, T, K, 1)
    
    # Normalize directions (avoid division by zero)
    eps = 1e-8
    ray_directions = ray_directions / torch.clamp(ray_distances, min=eps)
    
    # Flatten for raycasting
    batch_size = N * C * T * K
    ray_starts_flat = camera_pos_expanded.reshape(batch_size, 3)
    ray_directions_flat = ray_directions.reshape(batch_size, 3)
    
    # Perform raycasting
    ray_hits_flat, _, _, _ = raycast_mesh(
        ray_starts_flat,
        ray_directions_flat,
        mesh=mesh,
        max_dist=max_distance,
        return_distance=False,
        return_normal=False
    )
    
    # Reshape hit positions
    ray_hits = ray_hits_flat.view(N, C, T, K, 3) # (4, 2, 1, 1, 3)
    
    # Store hits for visualization if requested
    if out_hits is not None:
        out_hits.copy_(ray_hits)
    
    # Check if ray hits are close to target
    # Compute target bbox diagonal for tolerance
    if target_bbox_size.ndim == 2:
        # Shape (N, 3) -> expand to (N, T, 3) @TODO: What happens if T > 1?
        target_bbox_size = target_bbox_size.unsqueeze(1)
    
    bbox_diagonal = torch.norm(target_bbox_size, dim=-1) / 2.0  # (N, T)
    # max_dist_to_target = bbox_diagonal * tolerance_scale
    
    # max_dist_to_target = max_dist_to_target.unsqueeze(1)
    # max_dist_to_target = max_dist_to_target.unsqueeze(3)
    min_dist_to_target = torch.norm(
            test_points_expanded - camera_pos_expanded, dim=-1
        ) - bbox_diagonal.unsqueeze(1).unsqueeze(2).expand(-1, C, -1, -1) # (N, C, T, 1)
    
    # Expand target positions: (N, 1, T, 1, 3)
    target_pos_expanded = target_positions.unsqueeze(1).unsqueeze(3)
    
    # Compute distance from ray hit to target center
    dist_to_target = torch.norm(ray_hits - target_pos_expanded, dim=-1)  # (N, C, T, K)
    
    # Check if hit happens beyond the target
    hits_target = dist_to_target > min_dist_to_target
    
    # Check for invalid hits (nothing in between) (inf/nan from missing intersections)
    invalid_hit = (torch.isinf(ray_hits).any(dim=-1) | torch.isnan(ray_hits).any(dim=-1))
    
    # A point is visible if it hits the target (or no geometry)
    point_visible = hits_target | invalid_hit  # No hit means nothing blocking
    
    # Compute visibility ratio for each target
    visibility_ratio = point_visible.float().mean(dim=-1)  # (N, C, T)
    
    # Target is visible if visibility ratio exceeds threshold
    visibility_mask = visibility_ratio >= visibility_threshold
    
    return visibility_mask, visibility_ratio


def compute_bbox_diagonal(corners_local: torch.Tensor) -> torch.Tensor:
    """Compute the diagonal length of a bounding box.
    
    Args:
        corners_local: 8 corners in local frame. Shape (8, 3) or (T, 8, 3).
        
    Returns:
        Diagonal length. Shape () or (T,).
    """
    # Get min and max corners
    min_corner, _ = corners_local.min(dim=-2)
    max_corner, _ = corners_local.max(dim=-2)
    
    # Compute diagonal
    diagonal = torch.norm(max_corner - min_corner, dim=-1)
    
    return diagonal


def compute_bbox_size(corners_local: torch.Tensor) -> torch.Tensor:
    """Compute the size (width, height, depth) of a bounding box.
    
    Args:
        corners_local: 8 corners in local frame. Shape (8, 3) or (T, 8, 3).
        
    Returns:
        Bbox size (width, height, depth). Shape (3,) or (T, 3).
    """
    # Get min and max corners
    min_corner, _ = corners_local.min(dim=-2)
    max_corner, _ = corners_local.max(dim=-2)
    
    # Compute size
    size = max_corner - min_corner
    
    return size