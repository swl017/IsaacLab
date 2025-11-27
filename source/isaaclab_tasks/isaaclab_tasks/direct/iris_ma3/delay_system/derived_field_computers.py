"""
Computation functions for derived fields.

This module contains functions that compute derived fields from raw sensor inputs.
All derived fields should be recomputed from delayed sensor data rather than
being delayed independently.
"""

from __future__ import annotations
import torch
from isaaclab.utils.math import quat_mul, quat_rotate_inverse


def compute_camera_orientation_from_gimbal(
    body_orientation_w: torch.Tensor,       # [N, 4] quaternion (w, x, y, z)
    joint_positions_b: torch.Tensor,        # [N, J] where J >= 2 (pitch, yaw)
    camera_offset_rotation_b: torch.Tensor, # [N, 4] quaternion
) -> torch.Tensor:
    """
    Compute camera orientation from body orientation and gimbal joint angles.

    The camera orientation is computed by composing:
    world → body → gimbal → camera_offset

    Args:
        body_orientation_w: Body orientation in world frame [N, 4] (w, x, y, z)
        joint_positions_b: Gimbal joint angles [N, J] (at least pitch, yaw)
        camera_offset_rotation_b: Camera mounting offset rotation [N, 4]

    Returns:
        camera_orientation_w: Camera orientation in world frame [N, 4]
    """
    # Extract gimbal angles (pitch=joint[0], yaw=joint[1])
    pitch = joint_positions_b[:, 0]  # [N]
    yaw = joint_positions_b[:, 1] if joint_positions_b.shape[1] > 1 else torch.zeros_like(pitch)

    # Convert Euler angles to quaternion
    # Gimbal rotation: pitch around Y, yaw around Z
    # q = quat_from_euler_xyz(roll=0, pitch=pitch, yaw=yaw)

    # Simplified gimbal quaternion (yaw around Z, then pitch around Y)
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    # Gimbal quaternion: first yaw (Z), then pitch (Y)
    gimbal_quat = torch.stack([
        cy * cp,           # w
        -sy * sp,          # x
        cy * sp,           # y
        sy * cp,           # z
    ], dim=-1)  # [N, 4]

    # Compose: world → body → gimbal
    body_gimbal_orientation = quat_mul(body_orientation_w, gimbal_quat)

    # Compose: world → body → gimbal → camera
    camera_orientation_w = quat_mul(body_gimbal_orientation, camera_offset_rotation_b)

    return camera_orientation_w


def compute_camera_position(
    body_position_w: torch.Tensor,          # [N, 3]
    body_orientation_w: torch.Tensor,       # [N, 4] quaternion
    camera_offset_position_b: torch.Tensor, # [N, 3]
) -> torch.Tensor:
    """
    Compute camera position from body pose and camera offset.

    Args:
        body_position_w: Body position in world frame [N, 3]
        body_orientation_w: Body orientation in world frame [N, 4]
        camera_offset_position_b: Camera offset position in body frame [N, 3]

    Returns:
        camera_position_w: Camera position in world frame [N, 3]
    """
    # Rotate camera offset from body frame to world frame
    # Note: quat_rotate_inverse actually rotates the vector by the quaternion
    # (the name is confusing, but it's the standard rotation operation)
    camera_offset_w = quat_rotate_inverse(body_orientation_w, camera_offset_position_b)

    # Add to body position
    camera_position_w = body_position_w + camera_offset_w

    return camera_position_w


def compute_ray_origins(
    camera_position_w: torch.Tensor,  # [N, 3]
    num_targets: int,
) -> torch.Tensor:
    """
    Compute ray origins from camera position.

    All rays originate from the camera position.

    Args:
        camera_position_w: Camera position in world frame [N, 3]
        num_targets: Number of targets (rays)

    Returns:
        ray_origins_w: Ray origins in world frame [N, T, 3]
    """
    N = camera_position_w.shape[0]

    # All rays originate from camera position
    # Expand to [N, T, 3]
    ray_origins_w = camera_position_w.unsqueeze(1).expand(N, num_targets, 3)

    return ray_origins_w


def compute_ray_directions_from_bbox(
    camera_orientation_w: torch.Tensor,  # [N, 4] quaternion
    camera_base_intrinsics: torch.Tensor,  # [N, 3, 3] - REQUIRED: base (unzoomed) camera intrinsics
    camera_zoom_level: torch.Tensor,     # [N]
    bboxes_2d: torch.Tensor,             # [N, T, 4] (x, y, w, h) in pixels
) -> torch.Tensor:
    """
    Compute 3D ray directions from 2D bounding box centers.

    This function unprojects 2D bbox centers to 3D normalized ray directions using
    the camera intrinsics matrix (with zoom applied) and then rotates to world frame.

    Args:
        camera_orientation_w: Camera orientation in world frame [N, 4] (w, x, y, z)
        camera_base_intrinsics: Base camera intrinsics matrix [N, 3, 3] (unzoomed)
            This should be the intrinsic calibration matrix K with fx, fy, cx, cy
        camera_zoom_level: Zoom level multiplier [N] (1.0 = no zoom)
            Applied to focal lengths (fx, fy) to compute zoomed intrinsics
        bboxes_2d: Bounding boxes in pixel coordinates [N, T, 4] (x, y, w, h)

    Returns:
        ray_directions_w: Normalized ray directions in world frame [N, T, 3]

    Note:
        - Zoom is applied by multiplying the base focal lengths (fx, fy) by zoom_level
        - Principal point (cx, cy) is not affected by zoom
        - All rays are normalized to unit vectors
    """
    N, T, _ = bboxes_2d.shape
    device = bboxes_2d.device

    # Ensure zoom_level is [N] not [N, 1]
    # FirstOrderLagSampler with state_dim=1 outputs [N, 1], which needs to be [N]
    def ensure_1d(tensor: torch.Tensor, expected_size: int) -> torch.Tensor:
        """Ensure tensor is 1D with expected size."""
        if tensor.dim() == 0:
            return tensor.unsqueeze(0).expand(expected_size)
        elif tensor.dim() == 1 and tensor.shape[0] == expected_size:
            return tensor
        else:
            squeezed = tensor.squeeze()
            if squeezed.dim() == 0:
                return squeezed.unsqueeze(0).expand(expected_size)
            elif squeezed.dim() == 1:
                if squeezed.shape[0] == expected_size:
                    return squeezed
                elif squeezed.shape[0] > expected_size:
                    return squeezed[:expected_size]
            return tensor.flatten()[:expected_size]

    camera_zoom_level = ensure_1d(camera_zoom_level, N)

    # Validate inputs
    assert camera_base_intrinsics.shape == (N, 3, 3), \
        f"camera_base_intrinsics must be [N, 3, 3], got {camera_base_intrinsics.shape}"
    assert camera_zoom_level.shape[0] == N, \
        f"camera_zoom_level shape mismatch: {camera_zoom_level.shape} vs N={N}"
    assert not torch.any(torch.isnan(camera_base_intrinsics)), \
        "camera_base_intrinsics contains NaN values"
    assert not torch.any(camera_base_intrinsics[:, 2, 2] == 0), \
        "camera_base_intrinsics[2,2] must be non-zero (should be 1.0)"

    # Compute bbox centers in pixel coordinates
    bbox_centers_px = bboxes_2d[..., :2] + bboxes_2d[..., 2:] / 2  # [N, T, 2]

    # Apply zoom to base intrinsics
    # Zoom affects focal lengths but not principal point
    K = camera_base_intrinsics.clone()
    K[:, 0, 0] = K[:, 0, 0] * camera_zoom_level  # fx *= zoom
    K[:, 1, 1] = K[:, 1, 1] * camera_zoom_level  # fy *= zoom
    # K[:, 0, 2] = cx (unchanged)
    # K[:, 1, 2] = cy (unchanged)
    # K[:, 2, 2] = 1.0 (unchanged)

    # Unproject pixel coordinates to normalized camera coordinates
    # Inverse of: [u, v, 1]^T = K * [X/Z, Y/Z, 1]^T
    # So: [X/Z, Y/Z, 1]^T = K^{-1} * [u, v, 1]^T

    # Homogeneous pixel coordinates [N, T, 3]
    pixels_hom = torch.ones(N, T, 3, device=device)
    pixels_hom[:, :, :2] = bbox_centers_px

    # Compute K^{-1} for each environment
    K_inv = torch.inverse(K)  # [N, 3, 3]

    # Unproject: [N, T, 3] = [N, 3, 3] @ [N, T, 3]
    # Need to do batch matrix multiplication
    # Reshape for bmm: [N*T, 1, 3] @ [N, 3, 3]^T = [N*T, 1, 3]

    # Expand K_inv for each target: [N, 1, 3, 3] -> [N, T, 3, 3]
    K_inv_expanded = K_inv.unsqueeze(1).expand(N, T, 3, 3)

    # Batch matrix-vector multiply
    rays_camera = torch.einsum('ntij,ntj->nti', K_inv_expanded, pixels_hom)  # [N, T, 3]

    # Normalize ray directions in camera frame
    rays_camera_normalized = rays_camera / (rays_camera.norm(dim=-1, keepdim=True) + 1e-8)

    # Rotate from camera frame to world frame
    # Need to apply quaternion rotation to each ray
    # quat_rotate_inverse rotates vectors by quaternion

    # Expand camera_orientation_w for each target: [N, 4] -> [N, T, 4]
    camera_quat_expanded = camera_orientation_w.unsqueeze(1).expand(N, T, 4)

    # Flatten for batch rotation
    rays_camera_flat = rays_camera_normalized.reshape(N * T, 3)
    camera_quat_flat = camera_quat_expanded.reshape(N * T, 4)

    rays_world_flat = quat_rotate_inverse(camera_quat_flat, rays_camera_flat)
    rays_world = rays_world_flat.reshape(N, T, 3)

    # Normalize (should already be normalized, but ensure it)
    rays_world_normalized = rays_world / (rays_world.norm(dim=-1, keepdim=True) + 1e-8)

    return rays_world_normalized


def compute_combined_angular_velocity(
    body_angular_velocity_w: torch.Tensor,  # [N, 3]
    body_angular_velocity_b: torch.Tensor,  # [N, 3]
    joint_velocities_b: torch.Tensor,       # [N, J]
    body_orientation_w: torch.Tensor,       # [N, 4] quaternion
    joint_positions_b: torch.Tensor,        # [N, J]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute combined angular velocity (body + gimbal).

    Args:
        body_angular_velocity_w: Body angular velocity in world frame [N, 3]
        body_angular_velocity_b: Body angular velocity in body frame [N, 3]
        joint_velocities_b: Gimbal joint velocities [N, J]
        body_orientation_w: Body orientation [N, 4] (for frame transformations)
        joint_positions_b: Gimbal joint positions [N, J] (for kinematics)

    Returns:
        combined_angular_velocity_w: Combined velocity in world frame [N, 3]
        combined_angular_velocity_b: Combined velocity in body frame [N, 3]
    """
    # Extract gimbal velocities (pitch_rate, yaw_rate)
    pitch_rate = joint_velocities_b[:, 0] if joint_velocities_b.shape[1] > 0 else torch.zeros(
        body_angular_velocity_b.shape[0], device=body_angular_velocity_b.device
    )
    yaw_rate = joint_velocities_b[:, 1] if joint_velocities_b.shape[1] > 1 else torch.zeros_like(pitch_rate)

    # Gimbal angular velocity in body frame
    # Assuming gimbal axes: yaw around body Z, pitch around body Y
    #
    # SIMPLIFICATION: This treats gimbal joint rates as if they contribute directly
    # to body-frame angular velocity components. This is a valid approximation when:
    # 1. Gimbal angles are small (linearization around zero)
    # 2. The gimbal axes are approximately aligned with body axes
    #
    # For large gimbal angles, the actual angular velocity would require proper
    # gimbal kinematics (e.g., using the gimbal Jacobian matrix).
    #
    # Current mapping:
    #   - yaw_rate (joint[1]) → body Z-axis rotation
    #   - pitch_rate (joint[0]) → body Y-axis rotation
    #
    # NOTE: At large yaw angles (e.g., yaw = -π/2), the pitch axis is no longer
    # aligned with body Y, so pitch_rate would contribute to body X (roll).
    gimbal_angular_velocity_b = torch.stack([
        torch.zeros_like(pitch_rate),  # X (no direct contribution in simplified model)
        pitch_rate,                     # Y (pitch axis assumed aligned with body Y)
        yaw_rate,                       # Z (yaw axis fixed to body Z)
    ], dim=-1)  # [N, 3]

    # Combined angular velocity in body frame
    # Angular velocities add linearly when expressed in the same frame
    combined_angular_velocity_b = body_angular_velocity_b + gimbal_angular_velocity_b

    # Transform to world frame
    combined_angular_velocity_w = quat_rotate_inverse(body_orientation_w, combined_angular_velocity_b)

    return combined_angular_velocity_w, combined_angular_velocity_b
