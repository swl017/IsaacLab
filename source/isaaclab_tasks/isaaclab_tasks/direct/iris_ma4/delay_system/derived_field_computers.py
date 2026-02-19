"""
Computation functions for derived fields.

This module contains functions that compute derived fields from raw sensor inputs.
All derived fields should be recomputed from delayed sensor data rather than
being delayed independently.
"""

from __future__ import annotations
import logging
import torch
from isaaclab.utils.math import quat_mul, quat_rotate_inverse, matrix_from_quat

logger = logging.getLogger(__name__)


def _sanitize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf quaternions with identity [1, 0, 0, 0].

    This guards against physics engine producing invalid orientations when
    a drone enters an extreme state (e.g. crash). Without this, a single
    NaN quaternion propagates through all downstream derived fields and
    crashes training.
    """
    bad = torch.isnan(quat).any(dim=-1, keepdim=True) | torch.isinf(quat).any(dim=-1, keepdim=True)
    if not bad.any():
        return quat
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0  # w=1, x=y=z=0
    return torch.where(bad, identity, quat)


def compute_camera_orientation_from_gimbal(
    body_orientation_w: torch.Tensor,       # [N, 4] quaternion (w, x, y, z)
    joint_positions_b: torch.Tensor,        # [N, J] where J >= 2 (pitch, yaw) or J >= 3 (pitch, yaw, roll)
    camera_offset_rotation_b: torch.Tensor, # [N, 4] quaternion
) -> torch.Tensor:
    """
    Compute camera orientation from body orientation and gimbal joint angles.

    The camera orientation is computed by composing:
    world → body → gimbal → camera_offset

    Gimbal rotation order: yaw (Z) → roll (X) → pitch (Y)
    This matches the gimbal_stabilizer.py convention (ZXY intrinsic rotation).

    Args:
        body_orientation_w: Body orientation in world frame [N, 4] (w, x, y, z)
        joint_positions_b: Gimbal joint angles [N, J] where:
            - joint[0] = pitch (rotation around Y axis)
            - joint[1] = yaw (rotation around Z axis)
            - joint[2] = roll (rotation around X axis, optional)
        camera_offset_rotation_b: Camera mounting offset rotation [N, 4]

    Returns:
        camera_orientation_w: Camera orientation in world frame [N, 4]
    """
    # Sanitize: replace NaN/Inf quaternions with identity to survive physics instabilities
    body_orientation_w = _sanitize_quaternion(body_orientation_w)

    # Extract gimbal angles (pitch=joint[0], yaw=joint[1], roll=joint[2])
    pitch = joint_positions_b[:, 0]  # [N]
    yaw = joint_positions_b[:, 1] if joint_positions_b.shape[1] > 1 else torch.zeros_like(pitch)
    roll = joint_positions_b[:, 2] if joint_positions_b.shape[1] > 2 else torch.zeros_like(pitch)

    # Compute half angles
    half_yaw = yaw * 0.5
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5

    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)
    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)

    # Gimbal quaternion: yaw (Z) * roll (X) * pitch (Y)
    # Using quaternion multiplication formula for ZXY intrinsic rotation
    # q_total = q_yaw * q_roll * q_pitch
    gimbal_quat = torch.stack([
        cy * cr * cp + sy * sr * sp,    # w
        cy * sr * cp - sy * cr * sp,    # x
        cy * cr * sp + sy * sr * cp,    # y
        sy * cr * cp - cy * sr * sp,    # z
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
    # Sanitize: replace NaN/Inf quaternions with identity to survive physics instabilities
    body_orientation_w = _sanitize_quaternion(body_orientation_w)

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
    
    # Convert 2D bbox center to normalized image coordinates
    fx = camera_base_intrinsics[..., 0, 0].unsqueeze(-1) * camera_zoom_level.unsqueeze(-1)  # (N, 1)
    fy = camera_base_intrinsics[..., 1, 1].unsqueeze(-1) * camera_zoom_level.unsqueeze(-1)  # (N, 1)
    cx = camera_base_intrinsics[..., 0, 2].unsqueeze(-1)
    cy = camera_base_intrinsics[..., 1, 2].unsqueeze(-1)

    def validate_bbox(bbox: torch.Tensor, width, height) -> torch.Tensor:
        """Validate arbitrary bounding boxes based on size and center criteria.
        Args:
            bboxes: Tensor of shape (N, 4) in normalized (x_center, y_center, width, height) format.
        Returns:
            Tensor of shape (N,) with boolean validity mask.
        """
        # Normalize one bbox
        # Extract dimensions
        img_h = height
        img_w = width
        
        # Prevent division by zero
        img_w_safe = torch.clamp(img_w, min=1e-6)
        img_h_safe = torch.clamp(img_h, min=1e-6)
        
        # Split bbox components (raw pixel values, before normalization)
        cx_raw, cy_raw, w_raw, h_raw = bbox.split(1, dim=-1)

        # Reject non-positive dimensions before normalization/clamping.
        # Sentinel values like -2e6 from detection failures must be caught here;
        # clamping would hide them and let invalid bboxes pass.
        positive_size = (w_raw > 0) & (h_raw > 0)

        # Normalize
        cx_norm = cx_raw / img_w_safe
        cy_norm = cy_raw / img_h_safe
        w_norm = w_raw / img_w_safe
        h_norm = h_raw / img_h_safe

        # Clamp to [0, 1] to handle numerical errors
        cx_norm = torch.clamp(cx_norm, 0.0, 1.0)
        cy_norm = torch.clamp(cy_norm, 0.0, 1.0)
        w_norm = torch.clamp(w_norm, 0.0, 1.0)
        h_norm = torch.clamp(h_norm, 0.0, 1.0)
        bbox_normalized = torch.cat([cx_norm, cy_norm, w_norm, h_norm], dim=-1) # (N, 4)

        # Check normalized size within valid range (strictly positive after raw check above)
        w_n = bbox_normalized[..., 2]
        h_n = bbox_normalized[..., 3]
        size_ok = (w_n > 0) & (w_n <= 1.0) & (h_n > 0) & (h_n <= 1.0)

        center_x = bbox_normalized[..., 0]
        center_y = bbox_normalized[..., 1]
        center_ok = (
            (center_x > 0.0) & (center_x < 1.0) &
            (center_y > 0.0) & (center_y < 1.0)
        )

        valid_mask = positive_size.squeeze(-1) & size_ok & center_ok

        return valid_mask

    is_bbox_valid = validate_bbox(bboxes_2d, 2*cx, 2*cy)  # [N, T]
    x_n = (bboxes_2d[..., 0] - cx) / fx # (N, T)
    y_n = (bboxes_2d[..., 1] - cy) / fy

    # Form direction vectors in camera frame
    dirs_camera = torch.stack([x_n, y_n, torch.ones_like(x_n)], dim=-1)  # [N, T, 3]

    # Normalize direction vectors
    dirs_camera_norm = dirs_camera / torch.norm(dirs_camera, dim=-1, keepdim=True) # [N, T, 3]

    # Rotate to world frame - treat dirs_camera_norm as batch of column vectors
    camera_rot_mat = matrix_from_quat(camera_orientation_w)  # [N, 3, 3]
    camera_rot_mat_exp = camera_rot_mat.unsqueeze(1).expand(N, dirs_camera_norm.shape[1], 3, 3)  # [N, T, 3, 3]
    ray_directions_w = torch.matmul(camera_rot_mat_exp, dirs_camera_norm.unsqueeze(-1)).squeeze(-1)  # [N, T, 3]

    ray_directions_w = torch.where(
        is_bbox_valid.unsqueeze(-1),
        ray_directions_w,
        torch.zeros_like(ray_directions_w)
    )

    # Check for NaN/Inf in ray directions
    if torch.isnan(ray_directions_w).any() or torch.isinf(ray_directions_w).any():
        # Find which environments have issues
        nan_mask = torch.isnan(ray_directions_w).any(dim=-1).any(dim=-1)  # [N]
        inf_mask = torch.isinf(ray_directions_w).any(dim=-1).any(dim=-1)  # [N]
        bad_envs = torch.nonzero(nan_mask | inf_mask).squeeze(-1).tolist()

        # Zero out NaN/Inf rays instead of crashing training.
        # This is a safety net — the quaternion sanitization and bbox validation
        # fixes above should prevent most NaN propagation, but if any slip through
        # we log a warning and zero the affected rays.
        logger.warning(
            f"ray_directions_w NaN/Inf in {len(bad_envs)} envs (zeroed out): {bad_envs[:5]}"
        )
        bad_mask = nan_mask | inf_mask  # [N]
        ray_directions_w[bad_mask] = 0.0

    return ray_directions_w

    # # Compute bbox centers in pixel coordinates
    # # bbox_centers_px = bboxes_2d[..., :2] + bboxes_2d[..., 2:] / 2  # [N, T, 2]
    # bbox_centers_px = bboxes_2d[..., :2] # [N, T, 2]

    # # Apply zoom to base intrinsics
    # # Zoom affects focal lengths but not principal point
    # K = camera_base_intrinsics.clone()
    # K[:, 0, 0] = K[:, 0, 0] * camera_zoom_level  # fx *= zoom
    # K[:, 1, 1] = K[:, 1, 1] * camera_zoom_level  # fy *= zoom
    # # K[:, 0, 2] = cx (unchanged)
    # # K[:, 1, 2] = cy (unchanged)
    # # K[:, 2, 2] = 1.0 (unchanged)

    # # Unproject pixel coordinates to normalized camera coordinates
    # # Inverse of: [u, v, 1]^T = K * [X/Z, Y/Z, 1]^T
    # # So: [X/Z, Y/Z, 1]^T = K^{-1} * [u, v, 1]^T

    # # Homogeneous pixel coordinates [N, T, 3]
    # pixels_hom = torch.ones(N, T, 3, device=device)
    # pixels_hom[:, :, :2] = bbox_centers_px

    # # Compute K^{-1} for each environment
    # K_inv = torch.inverse(K)  # [N, 3, 3]

    # # Unproject: [N, T, 3] = [N, 3, 3] @ [N, T, 3]
    # # Need to do batch matrix multiplication
    # # Reshape for bmm: [N*T, 1, 3] @ [N, 3, 3]^T = [N*T, 1, 3]

    # # Expand K_inv for each target: [N, 1, 3, 3] -> [N, T, 3, 3]
    # K_inv_expanded = K_inv.unsqueeze(1).expand(N, T, 3, 3)

    # # Batch matrix-vector multiply
    # rays_camera = torch.einsum('ntij,ntj->nti', K_inv_expanded, pixels_hom)  # [N, T, 3]

    # # Normalize ray directions in camera frame
    # rays_camera_normalized = rays_camera / (rays_camera.norm(dim=-1, keepdim=True) + 1e-8)

    # # Rotate from camera frame to world frame
    # # Need to apply quaternion rotation to each ray
    # # quat_rotate_inverse rotates vectors by quaternion

    # # Expand camera_orientation_w for each target: [N, 4] -> [N, T, 4]
    # camera_quat_expanded = camera_orientation_w.unsqueeze(1).expand(N, T, 4)

    # # Flatten for batch rotation
    # rays_camera_flat = rays_camera_normalized.reshape(N * T, 3)
    # camera_quat_flat = camera_quat_expanded.reshape(N * T, 4)

    # rays_world_flat = quat_rotate_inverse(camera_quat_flat, rays_camera_flat)
    # rays_world = rays_world_flat.reshape(N, T, 3)

    # # Normalize (should already be normalized, but ensure it)
    # rays_world_normalized = rays_world / (rays_world.norm(dim=-1, keepdim=True) + 1e-8)

    # return rays_world_normalized


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
    # Sanitize: replace NaN/Inf quaternions with identity to survive physics instabilities
    body_orientation_w = _sanitize_quaternion(body_orientation_w)

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
