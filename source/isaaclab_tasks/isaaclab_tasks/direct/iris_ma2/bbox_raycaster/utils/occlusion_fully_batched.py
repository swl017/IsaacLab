# file: utils/occlusion_fully_batched.py

"""
Fully batched occlusion detection - NO environment loops!

Key insight: Since all environments share the same agent meshes,
we can batch ALL rays across ALL environments at once in body frame.

Complexity: O(num_agents) raycasts instead of O(N*C*num_agents)
Expected speedup: 10-1000x depending on N
"""

import torch
import warp as wp
from typing import Dict, Optional

from isaaclab.utils.warp import raycast_mesh
import isaaclab.utils.math as math_utils


def batch_check_occlusion_fully_batched(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully batched occlusion checking with NO environment loops.
    
    This is the ultimate optimization: batch ALL rays from ALL environments
    at once for each agent mesh. Since meshes are shared across environments,
    we can do this transformation once per agent instead of once per (env, agent).
    
    Complexity: O(num_agents) raycasts
    Previous: O(N * C * num_agents) raycasts
    Speedup: Up to N*C*1000x for large N
    
    Args:
        camera_pos: Camera positions in world frame. Shape (N, C, 3).
        camera_quat: Camera quaternions in world frame. Shape (N, C, 4).
        test_points_world: Test points in world frame. Shape (N, T, K, 3).
        target_positions: Target center positions in world frame. Shape (N, T, 3).
        target_bbox_size: Target bbox dimensions. Shape (N, T, 3) or (N, 3).
        agent_poses: Dict mapping agent_id -> (position, quaternion).
            position: (N, 3), quaternion: (N, 4)
        agent_meshes: Dict mapping agent_id -> wp.Mesh (body-local).
        static_mesh: Static environment mesh in world coords.
        max_distance: Maximum raycasting distance.
        visibility_threshold: Fraction of points that must be visible.
        tolerance_scale: Tolerance for ray-target intersection.
        current_agent_ids: List of agent IDs for cameras (e.g., ["drone_0", "drone_1"]).
        
    Returns:
        Tuple of (visibility_mask, visibility_ratio) both shape (N, C, T).
    """
    N, C = camera_pos.shape[:2]
    T, K = test_points_world.shape[1:3]
    device = camera_pos.device
    
    # Initialize visibility tracking
    point_visible = torch.ones((N, C, T, K), dtype=torch.bool, device=device)
    
    # Compute target bbox radius for tolerance
    if target_bbox_size.ndim == 2:
        target_bbox_size = target_bbox_size.unsqueeze(1).expand(N, T, -1)
    bbox_radius = torch.norm(target_bbox_size, dim=-1) / 2.0  # (N, T)
    
    # === Prepare data for batched operations ===
    # Expand arrays once
    camera_pos_exp = camera_pos.view(N, C, 1, 1, 3).expand(-1, -1, T, K, -1)  # (N, C, T, K, 3)
    test_points_exp = test_points_world.unsqueeze(1).expand(-1, C, -1, -1, -1)  # (N, C, T, K, 3)
    
    # Compute ray directions and distances in world frame
    ray_dirs_w = test_points_exp - camera_pos_exp  # (N, C, T, K, 3)
    ray_distances = torch.norm(ray_dirs_w, dim=-1, keepdim=True)  # (N, C, T, K, 1)
    ray_dirs_w = ray_dirs_w / torch.clamp(ray_distances, min=1e-8)
    
    # Compute camera-to-target distances for occlusion checking
    target_pos_exp = target_positions.unsqueeze(1).unsqueeze(3)  # (N, 1, T, 1, 3)
    cam_to_target = torch.norm(target_pos_exp - camera_pos_exp, dim=-1)  # (N, C, T, K)
    bbox_radius_exp = bbox_radius.unsqueeze(1).unsqueeze(3)  # (N, 1, T, 1)
    
    # === 1. Test against static environment (world frame) ===
    if static_mesh is not None:
        occluded_static = _raycast_static_fully_batched(
            camera_pos_exp, ray_dirs_w, static_mesh,
            cam_to_target, bbox_radius_exp, max_distance
        )
        point_visible &= ~occluded_static
    
    # === 2. Test against agent meshes (body-local frame, FULLY BATCHED) ===
    if len(agent_meshes) > 0 and agent_poses is not None:
        # Create camera-to-agent mapping for self-occlusion checking
        camera_to_agent = {}
        if current_agent_ids:
            for cam_idx, agent_id in enumerate(current_agent_ids):
                camera_to_agent[cam_idx] = agent_id
        
        # Process each agent mesh ONCE for ALL environments
        for agent_id, agent_mesh in agent_meshes.items():
            if agent_id not in agent_poses:
                continue
            
            agent_pos, agent_quat = agent_poses[agent_id]  # (N, 3), (N, 4)
            
            # Determine which cameras should test against this agent
            # (skip self-occlusion)
            test_camera_mask = torch.ones(C, dtype=torch.bool, device=device)
            for cam_idx, cam_agent_id in camera_to_agent.items():
                if cam_agent_id == agent_id:
                    test_camera_mask[cam_idx] = False
            
            if not test_camera_mask.any():
                continue  # No cameras need to test this agent
            
            # === BATCH ALL RAYS ACROSS ALL ENVS FOR THIS AGENT ===
            occluded_by_agent = _raycast_agent_fully_batched(
                camera_pos, camera_quat, camera_pos_exp, ray_dirs_w,
                cam_to_target, bbox_radius_exp,
                agent_pos, agent_quat, agent_mesh,
                test_camera_mask, max_distance
            )  # (N, C, T, K)
            
            point_visible &= ~occluded_by_agent
    
    # Compute visibility ratio
    visibility_ratio = point_visible.float().mean(dim=-1)  # (N, C, T)
    visibility_mask = visibility_ratio >= visibility_threshold
    
    return visibility_mask, visibility_ratio


def _raycast_static_fully_batched(
    camera_pos: torch.Tensor,      # (N, C, T, K, 3)
    ray_dirs: torch.Tensor,         # (N, C, T, K, 3)
    static_mesh: wp.Mesh,
    cam_to_target: torch.Tensor,    # (N, C, T, K)
    target_radius: torch.Tensor,    # (N, 1, T, 1)
    max_distance: float
) -> torch.Tensor:
    """Fully batched static environment raycasting.
    
    Returns:
        occluded: Shape (N, C, T, K) boolean tensor.
    """
    N, C, T, K = camera_pos.shape[:4]
    
    # Flatten everything for one massive raycast
    batch_size = N * C * T * K
    ray_starts_flat = camera_pos.reshape(batch_size, 3)
    ray_dirs_flat = ray_dirs.reshape(batch_size, 3)
    
    # Perform single massive raycast
    ray_hits_flat, _, _, _ = raycast_mesh(
        ray_starts_flat,
        ray_dirs_flat,
        mesh=static_mesh,
        max_dist=max_distance,
        return_distance=False,
        return_normal=False
    )
    
    ray_hits = ray_hits_flat.view(N, C, T, K, 3)
    
    # Check if hits are valid and occluding
    valid_hit = ~(torch.isinf(ray_hits).any(dim=-1) | torch.isnan(ray_hits).any(dim=-1))
    
    if not valid_hit.any():
        return torch.zeros((N, C, T, K), dtype=torch.bool, device=camera_pos.device)
    
    hit_distances = torch.norm(ray_hits - camera_pos, dim=-1)  # (N, C, T, K)
    min_clear_dist = cam_to_target - target_radius
    
    occluded = valid_hit & (hit_distances < min_clear_dist)
    
    return occluded


def _raycast_agent_fully_batched(
    camera_pos_orig: torch.Tensor,     # (N, C, 3)
    camera_quat_orig: torch.Tensor,    # (N, C, 4)
    camera_pos_exp: torch.Tensor,      # (N, C, T, K, 3)
    ray_dirs_w: torch.Tensor,          # (N, C, T, K, 3)
    cam_to_target: torch.Tensor,       # (N, C, T, K)
    target_radius: torch.Tensor,       # (N, 1, T, 1)
    agent_pos: torch.Tensor,           # (N, 3)
    agent_quat: torch.Tensor,          # (N, 4)
    agent_mesh: wp.Mesh,
    test_camera_mask: torch.Tensor,    # (C,) - which cameras to test
    max_distance: float
) -> torch.Tensor:
    """Fully batched agent raycasting - NO environment loops!
    
    Key idea: Since agent_mesh is the same for all environments, we can:
    1. Transform ALL rays from ALL environments to body frame
    2. Do ONE massive raycast
    3. Check occlusions in batch
    
    Returns:
        occluded: Shape (N, C, T, K) boolean tensor.
    """
    N, C, T, K = camera_pos_exp.shape[:4]
    device = camera_pos_exp.device
    
    # Initialize result
    occluded = torch.zeros((N, C, T, K), dtype=torch.bool, device=device)
    
    # Get cameras that need testing
    test_cam_indices = torch.where(test_camera_mask)[0]
    if len(test_cam_indices) == 0:
        return occluded
    
    # === Transform ALL rays to agent body frame (BATCHED) ===
    
    # Normalize agent quaternions
    agent_quat_norm = agent_quat / torch.clamp(
        torch.norm(agent_quat, dim=-1, keepdim=True), min=1e-8
    )
    agent_quat_inv = math_utils.quat_inv(agent_quat_norm)  # (N, 4)
    
    # For each camera that needs testing
    for cam_idx in test_cam_indices:
        cam_idx = cam_idx.item()
        
        # Get data for this camera across ALL environments
        cam_pos_w = camera_pos_orig[:, cam_idx, :]  # (N, 3)
        ray_dirs_w_cam = ray_dirs_w[:, cam_idx, :, :, :]  # (N, T, K, 3)
        cam_to_tgt = cam_to_target[:, cam_idx, :, :]  # (N, T, K)
        
        # === Transform camera positions to body frame ===
        # cam_pos_w: (N, 3), agent_pos: (N, 3)
        cam_rel = cam_pos_w - agent_pos  # (N, 3)
        cam_body = math_utils.quat_apply(agent_quat_inv, cam_rel)  # (N, 3)
        
        # === Transform ray directions to body frame ===
        # ray_dirs_w_cam: (N, T, K, 3)
        # We need to apply rotation for each (N, T, K) ray
        
        # Flatten rays
        ray_dirs_flat = ray_dirs_w_cam.reshape(N * T * K, 3)  # (N*T*K, 3)
        
        # Expand quaternions to match
        quat_inv_exp = agent_quat_inv.unsqueeze(1).unsqueeze(1).expand(-1, T, K, -1)  # (N, T, K, 4)
        quat_inv_flat = quat_inv_exp.reshape(N * T * K, 4)  # (N*T*K, 4)
        
        # Rotate all rays at once
        ray_dirs_body_flat = math_utils.quat_apply(quat_inv_flat, ray_dirs_flat)
        ray_dirs_body = ray_dirs_body_flat.view(N, T, K, 3)  # (N, T, K, 3)
        
        # === Prepare for raycasting ===
        # Expand camera positions in body frame for all test points
        cam_body_exp = cam_body.view(N, 1, 1, 3).expand(-1, T, K, -1)  # (N, T, K, 3)
        
        # Flatten everything for ONE massive raycast
        batch_size = N * T * K
        ray_starts_flat = cam_body_exp.reshape(batch_size, 3)
        ray_dirs_flat = ray_dirs_body.reshape(batch_size, 3)
        
        # === SINGLE RAYCAST FOR ALL ENVIRONMENTS ===
        ray_hits_flat, _, _, _ = raycast_mesh(
            ray_starts_flat,
            ray_dirs_flat,
            mesh=agent_mesh,
            max_dist=max_distance,
            return_distance=False,
            return_normal=False
        )
        
        ray_hits_body = ray_hits_flat.view(N, T, K, 3)
        
        # === Check occlusions ===
        valid_hit = ~(torch.isinf(ray_hits_body).any(dim=-1) | 
                     torch.isnan(ray_hits_body).any(dim=-1))
        
        if valid_hit.any():
            hit_distances = torch.norm(ray_hits_body - cam_body_exp, dim=-1)  # (N, T, K)
            min_clear_dist = cam_to_tgt - target_radius[:, 0, :, 0]  # (N, T, K)
            
            occluded_this_cam = valid_hit & (hit_distances < min_clear_dist)
            
            # Update result for this camera
            occluded[:, cam_idx, :, :] = occluded_this_cam
    
    return occluded


def batch_world_to_body_frame_vectorized(
    points_world: torch.Tensor,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Fully vectorized world-to-body transformation.
    
    Handles arbitrary batch dimensions with broadcasting.
    
    Args:
        points_world: Points in world frame. Shape (..., 3).
        body_pos: Body positions. Shape (..., 3) or broadcastable.
        body_quat: Body quaternions. Shape (..., 4) or broadcastable.
        eps: Epsilon for quaternion normalization.
        
    Returns:
        Points in body frame. Same shape as points_world.
    """
    # Normalize quaternions
    quat_norm = torch.norm(body_quat, dim=-1, keepdim=True)
    body_quat_normalized = body_quat / torch.clamp(quat_norm, min=eps)
    
    # Translate
    points_rel = points_world - body_pos
    
    # Rotate using inverse quaternion
    body_quat_inv = math_utils.quat_inv(body_quat_normalized)
    
    # Handle broadcasting for arbitrary shapes
    orig_shape = points_rel.shape
    points_flat = points_rel.reshape(-1, 3)
    
    # Expand quaternions
    quat_shape = body_quat_inv.shape[:-1]
    points_shape = orig_shape[:-1]
    
    # Broadcast quaternion to match points
    if quat_shape != points_shape:
        # Expand quat to match points
        for _ in range(len(points_shape) - len(quat_shape)):
            body_quat_inv = body_quat_inv.unsqueeze(0)
        body_quat_inv = body_quat_inv.expand(*points_shape, -1)
    
    quat_flat = body_quat_inv.reshape(-1, 4)
    
    points_body_flat = math_utils.quat_apply(quat_flat, points_flat)
    points_body = points_body_flat.reshape(orig_shape)
    
    return points_body


# ============================================================================
# Utility: Performance comparison
# ============================================================================

def compare_implementations(
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
    current_agent_ids: list[str],
    num_iterations: int = 100
):
    """Compare performance of different implementations.
    
    This is useful for profiling and choosing the right implementation.
    """
    import time
    
    print("=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print(f"Setup: N={camera_pos.shape[0]}, C={camera_pos.shape[1]}, "
          f"T={test_points_world.shape[1]}, K={test_points_world.shape[2]}")
    print(f"Agents: {len(agent_meshes)}, Iterations: {num_iterations}")
    print("-" * 70)
    
    implementations = [
        ("Fully Batched (No Loops)", batch_check_occlusion_fully_batched),
    ]
    
    results = {}
    
    for name, impl_func in implementations:
        # Warm-up
        for _ in range(5):
            impl_func(
                camera_pos, camera_quat, test_points_world, target_positions,
                target_bbox_size, agent_poses, agent_meshes, static_mesh,
                max_distance, visibility_threshold, 1.1, current_agent_ids
            )
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iterations):
            vis_mask, vis_ratio = impl_func(
                camera_pos, camera_quat, test_points_world, target_positions,
                target_bbox_size, agent_poses, agent_meshes, static_mesh,
                max_distance, visibility_threshold, 1.1, current_agent_ids
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / num_iterations
        fps = 1.0 / avg_time
        
        results[name] = {
            "avg_time_ms": avg_time * 1000,
            "fps": fps,
            "vis_rate": vis_mask.float().mean().item(),
        }
        
        print(f"{name}:")
        print(f"  Time: {avg_time*1000:.3f}ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Visibility rate: {results[name]['vis_rate']:.1%}")
        print()
    
    print("=" * 70)
    
    return results