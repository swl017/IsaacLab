# triang_cov_reward_torch.py
"""
PyTorch implementation of triangulation covariance computation.
Supports batched operations for N environments, T targets, C cameras.

Dimensions:
- N: number of parallel environments
- T: number of targets per environment  
- C: number of cameras per environment
"""

import torch
from typing import Optional, Tuple
import isaaclab.utils.math as math_utils

def skew(v: torch.Tensor) -> torch.Tensor:
    """
    Create skew-symmetric matrix from 3D vector(s).
    
    Args:
        v: [..., 3] tensor
    
    Returns:
        [..., 3, 3] skew-symmetric matrix
    """
    shape = v.shape[:-1]
    device = v.device
    
    zeros = torch.zeros(*shape, device=device)
    
    skew_mat = torch.stack([
        torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1)
    ], dim=-2)
    
    return skew_mat


def rotation_matrix_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor, 
                                order: str = 'ZYX') -> torch.Tensor:
    """
    Create rotation matrix from Euler angles (batched).
    
    Args:
        roll, pitch, yaw: [...] tensors of angles in radians
        order: rotation order (e.g., 'ZYX' means Rz(yaw) @ Ry(pitch) @ Rx(roll))
    
    Returns:
        [..., 3, 3] rotation matrices
    """
    shape = roll.shape
    device = roll.device
    
    cx, sx = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cz, sz = torch.cos(yaw), torch.sin(yaw)
    
    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)
    
    # Rotation matrices
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cx, -sx], dim=-1),
        torch.stack([zeros, sx, cx], dim=-1)
    ], dim=-2)
    
    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sy, zeros, cy], dim=-1)
    ], dim=-2)
    
    Rz = torch.stack([
        torch.stack([cz, -sz, zeros], dim=-1),
        torch.stack([sz, cz, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)
    
    m = {'X': Rx, 'Y': Ry, 'Z': Rz}
    R = torch.eye(3, device=device).expand(*shape, 3, 3).clone()
    
    for ax in order:
        R = R @ m[ax]
    
    return R


def proj_jacobian_wrt_Xc(Xc: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Jacobian of projection wrt camera-frame point.
    
    Args:
        Xc: [..., 3] points in camera frame
        K: [..., 3, 3] camera intrinsic matrices
    
    Returns:
        [..., 2, 3] Jacobian matrix
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    
    X, Y, Z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    
    # Avoid division by zero
    Z = torch.where(torch.abs(Z) >= 1e-12, Z, torch.ones_like(Z) * 1e-12)
    Z2 = Z * Z
    
    Ju = torch.stack([
        torch.stack([fx / Z, torch.zeros_like(Z), -fx * X / Z2], dim=-1),
        torch.stack([torch.zeros_like(Z), fy / Z, -fy * Y / Z2], dim=-1)
    ], dim=-2)
    
    return Ju


def build_views_from_env_state(
    robot_pos: torch.Tensor,  # [N, C, 3]
    robot_quat: torch.Tensor,  # [N, C, 4] (w, x, y, z)
    gimbal_yaw: torch.Tensor,  # [N, C]
    gimbal_pitch: torch.Tensor,  # [N, C]
    camera_intrinsics: torch.Tensor,  # [N, C, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build view parameters from environment state.
    
    Returns:
        R_wc: [N, C, 3, 3] world-to-camera rotations
        t_wc: [N, C, 3] camera positions
        e_alpha: [N, C, 3] gimbal yaw axis in camera frame
        e_beta: [N, C, 3] gimbal pitch axis in camera frame
    """
    N, C = robot_pos.shape[:2]
    device = robot_pos.device
    
    # Convert quaternion to rotation matrix (assuming w,x,y,z order)
    w, x, y, z = robot_quat[..., 0], robot_quat[..., 1], robot_quat[..., 2], robot_quat[..., 3]
    
    R_wb = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
    ], dim=-2)
    
    # Body to gimbal: Z(yaw) then Y(pitch)
    zeros = torch.zeros(N, C, device=device)
    R_bg = rotation_matrix_from_euler(zeros, gimbal_pitch, gimbal_yaw, 'ZYX')
    
    # Gimbal to camera (ENU to RDF)
    R_cg = torch.tensor([
        [0., -1., 0.],
        [0., 0., -1.],
        [1., 0., 0.]
    ], device=device)
    R_gc = R_cg.T.expand(N, C, 3, 3)
    
    # World to camera
    R_wc = R_wb @ R_bg @ R_gc
    t_wc = robot_pos
    
    # Gimbal axes in camera frame
    R_bc = R_bg @ R_gc
    z_b = torch.tensor([0., 0., 1.], device=device).expand(N, C, 3)
    e_alpha = torch.bmm(R_bc.view(N*C, 3, 3).transpose(-1, -2), 
                        z_b.view(N*C, 3, 1)).view(N, C, 3)
    e_beta = torch.tensor([-1., 0., 0.], device=device).expand(N, C, 3)
    
    return R_wc, t_wc, e_alpha, e_beta


def jacs_for_view_batch(
    X_w: torch.Tensor,  # [N, T, 3]
    R_wc: torch.Tensor,  # [N, C, 3, 3]
    t_wc: torch.Tensor,  # [N, C, 3]
    e_alpha: torch.Tensor,  # [N, C, 3]
    e_beta: torch.Tensor,  # [N, C, 3]
    K: torch.Tensor,  # [N, C, 3, 3]
    Sigma_pix: torch.Tensor,  # [N, C, 2, 2]
    Sigma_twb: Optional[torch.Tensor] = None,  # [N, C, 3, 3]
    Sigma_phiwb: Optional[torch.Tensor] = None,  # [N, C, 3, 3]
    Sigma_alpha: Optional[torch.Tensor] = None,  # [N, C, 1, 1]
    Sigma_beta: Optional[torch.Tensor] = None,  # [N, C, 1, 1]
    Sigma_K: Optional[torch.Tensor] = None,  # [N, C, 4, 4]
    include_pose: bool = True,
    include_gimbal: bool = True,
    include_intrinsics: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Jacobians for all cameras observing all targets.
    
    Returns:
        JX: [N, T, C, 2, 3] - Jacobian wrt X_w for each target-camera pair
        J_theta: [N, T, C, 2, M] - Jacobian wrt nuisance parameters
        Sigma_theta: [N, C, M, M] - covariance of nuisance parameters
        W_i: [N, C, 2, 2] - weight matrix (inverse pixel covariance)
    """
    N, T = X_w.shape[:2]
    C = R_wc.shape[1]
    device = X_w.device
    
    # Expand dimensions for broadcasting: X_w [N, T, 1, 3], cameras [N, 1, C, ...]
    X_w_exp = X_w.unsqueeze(2)  # [N, T, 1, 3]
    R_wc_exp = R_wc.unsqueeze(1)  # [N, 1, C, 3, 3]
    t_wc_exp = t_wc.unsqueeze(1)  # [N, 1, C, 3]
    K_exp = K.unsqueeze(1)  # [N, 1, C, 3, 3]
    
    # Transform to camera frame: [N, T, C, 3]
    X_rel = X_w_exp - t_wc_exp  # [N, T, C, 3]
    Xc = torch.matmul(R_wc_exp.transpose(-1, -2), X_rel.unsqueeze(-1)).squeeze(-1)
    
    # Projection Jacobian: [N, T, C, 2, 3]
    Ju_Xc = proj_jacobian_wrt_Xc(Xc, K_exp.squeeze(1).unsqueeze(1).expand(N, T, C, 3, 3))
    JX = torch.matmul(Ju_Xc, R_wc_exp.transpose(-1, -2))  # [N, T, C, 2, 3]
    
    # Process nuisance parameters (same for all targets in an environment)
    e_alpha_exp = e_alpha.unsqueeze(1)  # [N, 1, C, 3]
    e_beta_exp = e_beta.unsqueeze(1)  # [N, 1, C, 3]
    
    J_blocks = []
    Sig_blocks = []
    
    # Position uncertainty
    if include_pose and Sigma_twb is not None:
        J_t = torch.matmul(Ju_Xc, -R_wc_exp.transpose(-1, -2))  # [N, T, C, 2, 3]
        J_blocks.append(J_t)
        Sig_blocks.append(Sigma_twb)
    
    # Orientation uncertainty
    if include_pose and Sigma_phiwb is not None:
        skew_Xc = skew(Xc)  # [N, T, C, 3, 3]
        J_phi = torch.matmul(Ju_Xc, torch.matmul(-skew_Xc, R_wc_exp.transpose(-1, -2)))
        J_blocks.append(J_phi)
        Sig_blocks.append(Sigma_phiwb)
    
    # Gimbal yaw uncertainty
    if include_gimbal and Sigma_alpha is not None:
        skew_Xc = skew(Xc)  # [N, T, C, 3, 3]
        J_alpha = torch.matmul(Ju_Xc, torch.matmul(-skew_Xc, e_alpha_exp.unsqueeze(-1)))
        J_blocks.append(J_alpha)
        Sig_blocks.append(Sigma_alpha)
    
    # Gimbal pitch uncertainty
    if include_gimbal and Sigma_beta is not None:
        skew_Xc = skew(Xc)  # [N, T, C, 3, 3]
        J_beta = torch.matmul(Ju_Xc, torch.matmul(-skew_Xc, e_beta_exp.unsqueeze(-1)))
        J_blocks.append(J_beta)
        Sig_blocks.append(Sigma_beta)
    
    # Intrinsic uncertainty
    if include_intrinsics and Sigma_K is not None:
        X, Y, Z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
        Z = torch.where(torch.abs(Z) < 1e-12, torch.ones_like(Z) * 1e-12, Z)
        x_n = X / Z
        y_n = Y / Z
        
        J_K = torch.stack([
            torch.stack([x_n, torch.zeros_like(x_n), torch.ones_like(x_n), torch.zeros_like(x_n)], dim=-1),
            torch.stack([torch.zeros_like(y_n), y_n, torch.zeros_like(y_n), torch.ones_like(y_n)], dim=-1)
        ], dim=-2)  # [N, T, C, 2, 4]
        J_blocks.append(J_K)
        Sig_blocks.append(Sigma_K)
    
    # Concatenate Jacobians
    if len(J_blocks) > 0:
        J_theta = torch.cat(J_blocks, dim=-1)  # [N, T, C, 2, M]
        
        # Block diagonal covariance [N, C, M, M]
        M_total = J_theta.shape[-1]
        Sigma_theta = torch.zeros(N, C, M_total, M_total, device=device)
        col_idx = 0
        for sig_block in Sig_blocks:
            block_size = sig_block.shape[-1]
            Sigma_theta[:, :, col_idx:col_idx+block_size, col_idx:col_idx+block_size] = sig_block
            col_idx += block_size
    else:
        J_theta = torch.zeros(N, T, C, 2, 0, device=device)
        Sigma_theta = torch.zeros(N, C, 0, 0, device=device)
    
    # Weight matrix (inverse pixel covariance)
    W_i = torch.linalg.inv(Sigma_pix)  # [N, C, 2, 2]
    
    return JX, J_theta, Sigma_theta, W_i


def triangulation_covariance_multi_camera(
    X_w: torch.Tensor,  # [N, T, 3] - target positions
    robot_positions: torch.Tensor,  # [N, C, 3]
    robot_quats: torch.Tensor,  # [N, C, 4]
    gimbal_yaws: torch.Tensor,  # [N, C]
    gimbal_pitches: torch.Tensor,  # [N, C]
    camera_intrinsics: torch.Tensor,  # [N, C, 3, 3]
    Sigma_pix: torch.Tensor,  # [N, C, 2, 2]
    Sigma_twb: Optional[torch.Tensor] = None,  # [N, C, 3, 3]
    Sigma_phiwb: Optional[torch.Tensor] = None,  # [N, C, 3, 3]
    Sigma_alpha: Optional[torch.Tensor] = None,  # [N, C, 1, 1]
    Sigma_beta: Optional[torch.Tensor] = None,  # [N, C, 1, 1]
    Sigma_K: Optional[torch.Tensor] = None,  # [N, C, 4, 4]
    include_pose: bool = True,
    include_gimbal: bool = True,
    include_intrinsics: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute triangulation covariance for multiple targets and cameras (batched).
    
    Args:
        X_w: [N, T, 3] target positions (N envs, T targets per env)
        robot_positions: [N, C, 3] camera positions (C cameras per env)
        robot_quats: [N, C, 4] camera orientations
        gimbal_yaws: [N, C] gimbal yaw angles
        gimbal_pitches: [N, C] gimbal pitch angles
        camera_intrinsics: [N, C, 3, 3] camera intrinsic matrices
        Sigma_pix: [N, C, 2, 2] pixel noise covariance
        ...
    
    Returns:
        Sigma_X: [N, T, 3, 3] - covariance of triangulated position per target
        trace_cov: [N, T] - trace of covariance (scalar uncertainty metric)
    """
    N, T = X_w.shape[:2]
    C = robot_positions.shape[1]
    device = X_w.device
    
    # Build view parameters for all cameras
    R_wc, t_wc, e_alpha, e_beta = build_views_from_env_state(
        robot_positions, robot_quats, gimbal_yaws, gimbal_pitches, camera_intrinsics
    )
    
    # Compute Jacobians for all target-camera pairs
    JX, J_theta, Sigma_theta, W_i = jacs_for_view_batch(
        X_w, R_wc, t_wc, e_alpha, e_beta, camera_intrinsics,
        Sigma_pix, Sigma_twb, Sigma_phiwb, Sigma_alpha, Sigma_beta, Sigma_K,
        include_pose, include_gimbal, include_intrinsics
    )
    
    # JX: [N, T, C, 2, 3]
    # J_theta: [N, T, C, 2, M]
    # Sigma_theta: [N, C, M, M]
    # W_i: [N, C, 2, 2]
    
    # Get M (nuisance parameters per camera)
    M = J_theta.shape[-1]
    
    # Reshape for batch matrix operations
    # Stack cameras: [N, T, 2C, 3]
    JX_stacked = JX.reshape(N, T, 2*C, 3)
    
    # Create block diagonal J_theta: [N, T, 2C, CM]
    # Each camera's 2 measurements only depend on its own M parameters
    J_theta_stacked = torch.zeros(N, T, 2*C, C*M, device=device)
    for c in range(C):
        # Camera c's measurements (rows 2c:2c+2) depend on parameters [c*M:(c+1)*M]
        J_theta_stacked[:, :, 2*c:2*c+2, c*M:(c+1)*M] = J_theta[:, :, c, :, :]
    
    # Block diagonal weight matrix [N, 2C, 2C]
    W_block = torch.zeros(N, 2*C, 2*C, device=device)
    for c in range(C):
        W_block[:, 2*c:2*c+2, 2*c:2*c+2] = W_i[:, c]
    W_block_exp = W_block.unsqueeze(1)  # [N, 1, 2C, 2C]
    
    # Block diagonal pixel covariance [N, 2C, 2C]
    Sigma_z = torch.zeros(N, 2*C, 2*C, device=device)
    for c in range(C):
        Sigma_z[:, 2*c:2*c+2, 2*c:2*c+2] = Sigma_pix[:, c]
    Sigma_z_exp = Sigma_z.unsqueeze(1)  # [N, 1, 2C, 2C]
    
    # Block diagonal nuisance parameter covariance [N, C*M, C*M]
    Sigma_theta_block = torch.zeros(N, C*M, C*M, device=device)
    for c in range(C):
        Sigma_theta_block[:, c*M:(c+1)*M, c*M:(c+1)*M] = Sigma_theta[:, c]
    Sigma_theta_exp = Sigma_theta_block.unsqueeze(1)  # [N, 1, C*M, C*M]
    
    # Residual covariance: S = Sigma_z + J_theta @ Sigma_theta @ J_theta^T
    # [N, T, 2C, 2C]
    S_resid = Sigma_z_exp + torch.matmul(
        torch.matmul(J_theta_stacked, Sigma_theta_exp), 
        J_theta_stacked.transpose(-1, -2)
    )
    
    # Normal matrix: A = JX^T @ W @ JX [N, T, 3, 3]
    A = torch.matmul(
        torch.matmul(JX_stacked.transpose(-1, -2), W_block_exp), 
        JX_stacked
    )
    
    # Covariance: Sigma_X = A^{-1} @ JX^T @ W @ S @ W @ JX @ A^{-T}
    try:
        A_inv = torch.linalg.inv(A)
    except:
        # Use pseudo-inverse if singular
        A_inv = torch.linalg.pinv(A)
    
    middle = torch.matmul(
        torch.matmul(JX_stacked.transpose(-1, -2), W_block_exp),
        torch.matmul(S_resid, torch.matmul(W_block_exp, JX_stacked))
    )
    Sigma_X = torch.matmul(torch.matmul(A_inv, middle), A_inv.transpose(-1, -2))
    
    # Trace of covariance [N, T]
    trace_cov = torch.diagonal(Sigma_X, dim1=-2, dim2=-1).sum(dim=-1)
    
    return Sigma_X, trace_cov


def triangulation_covariance_simple(
    X_w: torch.Tensor,  # [N, T, 3]
    robot_positions: torch.Tensor,  # [N, C, 3]
    robot_quats: torch.Tensor,  # [N, C, 4]
    gimbal_yaws: torch.Tensor,  # [N, C]
    gimbal_pitches: torch.Tensor,  # [N, C]
    camera_intrinsics: torch.Tensor,  # [N, C, 3, 3]
    pixel_std: float = 0.5  # pixels
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified version using only pixel noise (fastest computation).
    
    Returns:
        Sigma_X: [N, T, 3, 3]
        trace_cov: [N, T]
    """
    N, C = robot_positions.shape[:2]
    device = X_w.device
    
    # Simple pixel covariance (isotropic) [N, C, 2, 2]
    Sigma_pix = torch.eye(2, device=device).unsqueeze(0).unsqueeze(0).expand(N, C, 2, 2) * (pixel_std ** 2)
    
    return triangulation_covariance_multi_camera(
        X_w, robot_positions, robot_quats, gimbal_yaws, gimbal_pitches,
        camera_intrinsics, Sigma_pix,
        Sigma_twb=None, Sigma_phiwb=None, Sigma_alpha=None, Sigma_beta=None,
        include_pose=False, include_gimbal=False, include_intrinsics=False
    )

def midpoint_method_batched(
    pts: torch.Tensor,
    dirs: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute midpoint of batched rays using least squares (batched).

    Args:
        pts: Points on each ray [N, C, 3]
        dirs: Direction vectors (unit) for each ray [N, C, T, 3]
        valid_mask: Per-camera validity mask [N, C]. Invalid cameras excluded from computation.
                   If None, all cameras are considered valid.

    Returns:
        X_mid: Midpoint of closest approach [N, T, 3]
        is_valid: Validity flag for each triangulation [N, T]. False if:
                  - Insufficient cameras (< 2 valid)
                  - Singular matrix (rank < 3)
                  - Result is behind all valid cameras
    """
    N, C, T = dirs.shape[:3]

    device = pts.device

    # Normalize direction vectors [N, C, T, 3]
    dirs_norm = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-12)

    # Identity matrix [N, C, T, 3, 3]
    I = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N, C, T, 3, 3)

    # Compute outer product d @ d^T [N, C, T, 3, 3]
    dd_T = torch.matmul(dirs_norm.unsqueeze(-1), dirs_norm.unsqueeze(-2))

    # Compute I - d @ d^T for each ray [N, C, T, 3, 3]
    I_minus_dd = I - dd_T

    # Apply valid_mask BEFORE summing over cameras
    # This prevents invalid cameras from contributing to triangulation
    if valid_mask is not None:
        # Expand mask: [N, C] -> [N, C, 1, 1, 1] for broadcasting with [N, C, T, 3, 3]
        mask_expanded = valid_mask.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1, 1]
        I_minus_dd = I_minus_dd * mask_expanded

    # Sum over cameras: A = sum(I - d @ d^T) [N, T, 3, 3]
    A = I_minus_dd.sum(dim=1)

    # Expand pts to match T dimension: [N, C, 3] -> [N, C, T, 3]
    pts_exp = pts.unsqueeze(-2).expand(N, C, T, 3)

    # Compute (I - d @ d^T) @ p for each ray [N, C, T, 3]
    I_minus_dd_p = torch.matmul(I_minus_dd, pts_exp.unsqueeze(-1)).squeeze(-1)

    # Apply valid_mask to the b vector computation as well
    if valid_mask is not None:
        # Mask already applied to I_minus_dd, so I_minus_dd_p is already masked
        pass

    # Sum over cameras: b = sum((I - d @ d^T) @ p) [N, T, 3]
    b = I_minus_dd_p.sum(dim=1)

    # Count valid cameras per batch for determining insufficient detection
    if valid_mask is not None:
        num_valid = valid_mask.sum(dim=1)  # [N]
        insufficient = num_valid < 2  # Need at least 2 cameras for triangulation
    else:
        insufficient = torch.zeros(N, dtype=torch.bool, device=device)

    # Compute fallback: mean of VALID camera positions only
    if valid_mask is not None:
        # Compute weighted mean: only include valid camera positions
        valid_pts = pts * valid_mask.unsqueeze(-1).float()  # [N, C, 3]
        num_valid_clamped = num_valid.clamp(min=1).unsqueeze(-1)  # [N, 1]
        mean_pts = valid_pts.sum(dim=1) / num_valid_clamped  # [N, 3]
    else:
        mean_pts = pts.mean(dim=1)  # [N, 3]

    # Expand to match target dimension: [N, 3] -> [N, T, 3]
    mean_pts = mean_pts.unsqueeze(1).expand(-1, T, -1)  # [N, T, 3]

    try:
        X_mid = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)  # [N, T, 3]
    except:
        # Fallback to mean if singular
        X_mid = mean_pts.clone()  # [N, T, 3]

    # Check for singular matrices per batch and use mean as fallback
    rank = torch.linalg.matrix_rank(A)  # [N, T]
    is_singular = (rank < 3) | insufficient.unsqueeze(-1)  # [N, T]

    # Use torch.where for per-environment fallback (not global .any())
    X_mid = torch.where(is_singular.unsqueeze(-1), mean_pts, X_mid)

    # ========== Behind Camera Check ==========
    # A triangulated point is "behind" a camera if the dot product of
    # (X_mid - camera_pos) with the ray direction is negative.
    # If the point is behind ALL valid cameras, the triangulation is invalid.

    # Expand X_mid and pts for per-camera check
    # X_mid: [N, T, 3] -> [N, 1, T, 3]
    # pts: [N, C, 3] -> [N, C, 1, 3]
    X_mid_exp = X_mid.unsqueeze(1)  # [N, 1, T, 3]
    pts_exp_check = pts.unsqueeze(2)  # [N, C, 1, 3]

    # Vector from camera to triangulated point [N, C, T, 3]
    cam_to_point = X_mid_exp - pts_exp_check  # [N, C, T, 3]

    # Dot product with normalized ray direction [N, C, T]
    dot_with_ray = (cam_to_point * dirs_norm).sum(dim=-1)  # [N, C, T]

    # A point is "behind" a camera if dot product < 0
    is_behind_camera = dot_with_ray < 0  # [N, C, T]

    # Apply valid_mask: only consider valid cameras for the behind check
    if valid_mask is not None:
        # Expand mask: [N, C] -> [N, C, T]
        mask_exp = valid_mask.unsqueeze(-1).expand(-1, -1, T)  # [N, C, T]
        # For invalid cameras, treat as "not behind" (True -> in front)
        # so they don't contribute to the "behind all cameras" check
        is_behind_camera = is_behind_camera & mask_exp

        # Count valid cameras that see the point in front
        in_front = (~is_behind_camera) & mask_exp  # [N, C, T]
        num_in_front = in_front.sum(dim=1)  # [N, T]

        # Invalid if no valid camera sees the point in front
        is_behind_all = num_in_front == 0  # [N, T]
    else:
        # All cameras valid: check if behind ALL cameras
        is_behind_all = is_behind_camera.all(dim=1)  # [N, T]

    # Use torch.where for per-environment fallback for behind-camera cases
    X_mid = torch.where(is_behind_all.unsqueeze(-1), mean_pts, X_mid)

    # Combined validity: valid if not singular AND not behind all cameras
    is_valid = ~(is_singular | is_behind_all)  # [N, T]

    return X_mid, is_valid

def get_ray_dir_from_bbox(
    bbox_2d: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    camera_rotations: torch.Tensor,
    camera_positions: torch.Tensor
) -> torch.Tensor:
    """
    Compute ray directions from 2D bounding box centers.

    Args:
        bbox_2d: 2D bounding box xywh [N, C, T, 4]
        camera_intrinsics: Camera intrinsic matrices [N, C, 3, 3]
        camera_rotations: Camera rotation quaternions [N, C, 4]
        camera_positions: Camera positions [N, C, 3]

    Returns:
        ray_dirs: Ray directions in world frame [N, C, T, 3]
    """
    N, C = bbox_2d.shape[:2]
    device = bbox_2d.device

    # Convert 2D bbox center to normalized image coordinates
    fx = camera_intrinsics[..., 0, 0].unsqueeze(-1)
    fy = camera_intrinsics[..., 1, 1].unsqueeze(-1)
    cx = camera_intrinsics[..., 0, 2].unsqueeze(-1)
    cy = camera_intrinsics[..., 1, 2].unsqueeze(-1)

    x_n = (bbox_2d[..., 0] - cx) / fx # (N, C, T)
    y_n = (bbox_2d[..., 1] - cy) / fy

    # Form direction vectors in camera frame
    dirs_camera = torch.stack([x_n, y_n, torch.ones_like(x_n)], dim=-1)  # [N, C, T, 3]

    # Normalize direction vectors
    dirs_camera_norm = dirs_camera / torch.norm(dirs_camera, dim=-1, keepdim=True) # [N, C, T, 3]

    # Rotate to world frame - treat dirs_camera_norm as batch of column vectors
    camera_rot_mat = math_utils.matrix_from_quat(camera_rotations)  # [N, C, 3, 3]
    camera_rot_mat_exp = camera_rot_mat.unsqueeze(2).expand(N, C, dirs_camera_norm.shape[2], 3, 3)  # [N, C, T, 3, 3]
    dirs_world = torch.matmul(camera_rot_mat_exp, dirs_camera_norm.unsqueeze(-1)).squeeze(-1)  # [N, C, T, 3]

    return dirs_world