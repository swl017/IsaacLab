# file: utils/occlusion.py

"""Occlusion detection using body-local raycasting for batched bbox raycasting."""

import torch
import warp as wp
from typing import Literal, Optional, Dict

from isaaclab.utils.warp import raycast_mesh
import isaaclab.utils.math as math_utils


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


def world_to_body_frame(
    points_world: torch.Tensor,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Transform points from world frame to body frame.
    
    Args:
        points_world: Points in world frame. Shape (..., 3).
        body_pos: Body position in world frame. Shape (..., 3).
        body_quat: Body quaternion (w, x, y, z). Shape (..., 4).
        eps: Epsilon for quaternion normalization.
        
    Returns:
        Points in body frame. Same shape as points_world.
    """
    # Store original shape
    orig_shape = points_world.shape
    points_flat = points_world.reshape(-1, 3)
    
    # Broadcast body pose to match points
    if body_pos.ndim == 1:
        body_pos = body_pos.unsqueeze(0).expand(points_flat.shape[0], -1)
    elif body_pos.shape[0] != points_flat.shape[0]:
        body_pos = body_pos.unsqueeze(0).expand(points_flat.shape[0], -1)
    
    if body_quat.ndim == 1:
        body_quat = body_quat.unsqueeze(0).expand(points_flat.shape[0], -1)
    elif body_quat.shape[0] != points_flat.shape[0]:
        body_quat = body_quat.unsqueeze(0).expand(points_flat.shape[0], -1)
    
    # Normalize quaternion
    quat_norm = torch.norm(body_quat, dim=-1, keepdim=True)
    body_quat = body_quat / torch.clamp(quat_norm, min=eps)
    
    # Translate to body origin
    points_rel = points_flat - body_pos
    
    # Rotate to body frame using inverse quaternion
    body_quat_inv = math_utils.quat_inv(body_quat)
    points_body = math_utils.quat_apply(body_quat_inv, points_rel)
    
    return points_body.reshape(orig_shape)


def batch_check_occlusion_body_local(
    camera_pos: torch.Tensor,
    camera_quat: torch.Tensor,
    test_points_world: torch.Tensor,
    target_positions: torch.Tensor,
    target_bbox_size: torch.Tensor,
    agent_poses: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    agent_meshes: Dict[str, wp.Mesh],
    static_mesh: Optional[wp.Mesh],
    max_distance: float,
    visibility_threshold: float,
    tolerance_scale: float = 1.1,
    current_agent_ids: Optional[list[str]] = None,
    out_hits: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check occlusion using per-environment body-local raycasting.
    
    This function tests occlusion by:
    1. Testing rays against static environment (ground) in world frame
    2. Testing rays against each agent's mesh in that agent's body frame
    
    Args:
        camera_pos: Camera positions in world frame. Shape (N, C, 3).
        camera_quat: Camera quaternions in world frame. Shape (N, C, 4).
        test_points_world: Test points in world frame. Shape (N, T, K, 3).
        target_positions: Target center positions in world frame. Shape (N, T, 3).
        target_bbox_size: Target bbox dimensions. Shape (N, T, 3) or (T, 3).
        agent_poses: Dict mapping agent_id -> (position, quaternion).
            position: (N, 3), quaternion: (N, 4)
        agent_meshes: Dict mapping agent_id -> wp.Mesh (in body-local coords).
        static_mesh: Warp mesh for static environment (ground, etc.) in world coords.
        max_distance: Maximum raycasting distance.
        visibility_threshold: Fraction of points that must be visible.
        tolerance_scale: Tolerance multiplier for ray-target intersection.
        current_agent_ids: List of agent IDs corresponding to camera indices.
        out_hits: Optional pre-allocated output tensor for hit positions.
        
    Returns:
        Tuple of:
            - visibility_mask: Boolean mask indicating visible targets. Shape (N, C, T).
            - visibility_ratio: Ratio of visible test points. Shape (N, C, T).
    """
    N, C = camera_pos.shape[:2]
    T, K = test_points_world.shape[1:3]
    device = camera_pos.device
    
    # Initialize visibility tracking
    point_visible = torch.ones((N, C, T, K), dtype=torch.bool, device=device)
    
    # Compute target bbox diagonal for tolerance
    if target_bbox_size.ndim == 2:
        target_bbox_size = target_bbox_size.unsqueeze(0).expand(N, -1, -1)
    bbox_diagonal = torch.norm(target_bbox_size, dim=-1) / 2.0  # (N, T)
    
    # === Per-environment, per-camera occlusion checking ===
    for env_idx in range(N):
        for cam_idx in range(C):
            cam_pos_w = camera_pos[env_idx, cam_idx]  # (3,)
            
            for target_idx in range(T):
                test_pts_w = test_points_world[env_idx, target_idx]  # (K, 3)
                target_pos_w = target_positions[env_idx, target_idx]  # (3,)
                target_radius = bbox_diagonal[env_idx, target_idx]  # scalar
                
                # Compute ray parameters
                ray_dirs_w = test_pts_w - cam_pos_w.unsqueeze(0)  # (K, 3)
                ray_distances = torch.norm(ray_dirs_w, dim=-1, keepdim=True)  # (K, 1)
                ray_dirs_w = ray_dirs_w / torch.clamp(ray_distances, min=1e-8)
                
                cam_to_target_dist = torch.norm(target_pos_w - cam_pos_w)  # scalar
                
                # --- 1. Test against static environment (world frame) ---
                if static_mesh is not None:
                    occluded_static = _raycast_static_env(
                        cam_pos_w, ray_dirs_w, static_mesh, 
                        cam_to_target_dist, target_radius, max_distance
                    )
                    point_visible[env_idx, cam_idx, target_idx] &= ~occluded_static
                
                # --- 2. Test against agent meshes (body-local frame) ---
                current_agent_id = current_agent_ids[cam_idx] if current_agent_ids else None
                
                for agent_id, agent_mesh in agent_meshes.items():
                    # if agent_id == current_agent_id:
                    #     continue
                    
                    if agent_id not in agent_poses:
                        continue
                    
                    agent_pos, agent_quat = agent_poses[agent_id]
                    occluded_agent = _raycast_agent_body_local(
                        cam_pos_w, ray_dirs_w, test_pts_w,
                        agent_pos[env_idx], agent_quat[env_idx],
                        agent_mesh, cam_to_target_dist, 
                        target_radius, max_distance
                    )
                    point_visible[env_idx, cam_idx, target_idx] &= ~occluded_agent
    
    # Compute visibility ratio
    visibility_ratio = point_visible.float().mean(dim=-1)  # (N, C, T)
    
    # Target is visible if visibility ratio exceeds threshold
    visibility_mask = visibility_ratio >= visibility_threshold
    
    return visibility_mask, visibility_ratio


def _raycast_static_env(
    camera_pos: torch.Tensor,      # (3,)
    ray_dirs: torch.Tensor,         # (K, 3)
    static_mesh: wp.Mesh,
    cam_to_target_dist: torch.Tensor,  # scalar
    target_radius: torch.Tensor,       # scalar
    max_distance: float
) -> torch.Tensor:
    """Raycast against static environment in world frame.
    
    Returns:
        occluded: Boolean tensor of shape (K,) indicating occluded rays.
    """
    K = ray_dirs.shape[0]
    
    # Prepare ray starts (all from camera)
    ray_starts = camera_pos.unsqueeze(0).expand(K, -1)  # (K, 3)
    
    # Perform raycasting
    ray_hits, _, _, _ = raycast_mesh(
        ray_starts,
        ray_dirs,
        mesh=static_mesh,
        max_dist=max_distance,
        return_distance=False,
        return_normal=False
    )
    
    # Check if hit is valid and closer than target
    valid_hit = ~(torch.isinf(ray_hits).any(dim=-1) | torch.isnan(ray_hits).any(dim=-1))
    
    if not valid_hit.any():
        return torch.zeros(K, dtype=torch.bool, device=camera_pos.device)
    
    hit_distances = torch.norm(ray_hits - camera_pos.unsqueeze(0), dim=-1)  # (K,)
    
    # Occluded if hit is significantly closer than target (with tolerance)
    min_clear_dist = cam_to_target_dist - target_radius
    occluded = valid_hit & (hit_distances < min_clear_dist)
    
    return occluded


def _raycast_agent_body_local(
    camera_pos_w: torch.Tensor,    # (3,) - world frame
    ray_dirs_w: torch.Tensor,      # (K, 3) - world frame
    test_points_w: torch.Tensor,   # (K, 3) - world frame
    agent_pos: torch.Tensor,       # (3,) - world frame
    agent_quat: torch.Tensor,      # (4,) - world frame
    agent_mesh: wp.Mesh,           # body-local coordinates
    cam_to_target_dist: torch.Tensor,  # scalar
    target_radius: torch.Tensor,       # scalar
    max_distance: float
) -> torch.Tensor:
    """Raycast against agent mesh in body-local frame.
    
    Args:
        camera_pos_w: Camera position in world frame.
        ray_dirs_w: Ray directions in world frame (normalized).
        test_points_w: Test points in world frame (for distance calculation).
        agent_pos: Agent position in world frame.
        agent_quat: Agent quaternion in world frame (w, x, y, z).
        agent_mesh: Agent mesh in body-local coordinates.
        cam_to_target_dist: Distance from camera to target.
        target_radius: Radius of target bounding sphere.
        max_distance: Maximum raycasting distance.
        
    Returns:
        occluded: Boolean tensor of shape (K,) indicating occluded rays.
    """
    K = ray_dirs_w.shape[0]
    device = camera_pos_w.device
    
    # === Transform camera and rays to agent body frame ===
    camera_pos_body = world_to_body_frame(
        camera_pos_w, agent_pos, agent_quat
    )  # (3,)
    
    # Transform ray directions (rotation only, no translation)
    # Normalize quaternion first
    agent_quat_normalized = agent_quat / torch.clamp(torch.norm(agent_quat), min=1e-8)
    agent_quat_inv = math_utils.quat_inv(agent_quat_normalized.unsqueeze(0))  # (1, 4)
    ray_dirs_body = math_utils.quat_apply(
        agent_quat_inv.expand(K, -1), ray_dirs_w
    )  # (K, 3)
    
    # === Perform raycasting in body frame ===
    ray_starts_body = camera_pos_body.unsqueeze(0).expand(K, -1)  # (K, 3)
    
    ray_hits_body, _, _, _ = raycast_mesh(
        ray_starts_body,
        ray_dirs_body,
        mesh=agent_mesh,
        max_dist=max_distance,
        return_distance=False,
        return_normal=False
    )
    
    # === Check if hits are valid and occluding ===
    valid_hit = ~(torch.isinf(ray_hits_body).any(dim=-1) | torch.isnan(ray_hits_body).any(dim=-1))
    
    if not valid_hit.any():
        return torch.zeros(K, dtype=torch.bool, device=device)
    
    # Compute hit distances in body frame
    hit_distances = torch.norm(ray_hits_body - camera_pos_body.unsqueeze(0), dim=-1)  # (K,)
    
    # Occluded if hit is significantly closer than target (with tolerance)
    min_clear_dist = cam_to_target_dist - target_radius
    occluded = valid_hit & (hit_distances < min_clear_dist)
    
    return occluded


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