# triang_cov_reward_torch.py
"""
PyTorch implementation of triangulation covariance computation.
Supports batched operations for multiple environments.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple

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


@dataclass
class Intrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float


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
        R = m[ax] @ R
    
    return R


def proj_jacobian_wrt_Xc(Xc: torch.Tensor, K: Intrinsics) -> torch.Tensor:
    """
    Jacobian of projection wrt camera-frame point.
    
    Args:
        Xc: [..., 3] points in camera frame
        K: camera intrinsics
    
    Returns:
        [..., 2, 3] Jacobian matrix
    """
    fx, fy = K.fx, K.fy
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
    robot_pos: torch.Tensor,  # [B, 3]
    robot_quat: torch.Tensor,  # [B, 4] (w, x, y, z)
    gimbal_yaw: torch.Tensor,  # [B]
    gimbal_pitch: torch.Tensor,  # [B]
    camera_intrinsics: List[Intrinsics],  # length B or 1
    Sigma_pix: torch.Tensor,  # [2, 2] or [B, 2, 2]
    Sigma_twb: Optional[torch.Tensor] = None,  # [3, 3] or [B, 3, 3]
    Sigma_phiwb: Optional[torch.Tensor] = None,  # [3, 3] or [B, 3, 3]
    Sigma_alpha: Optional[torch.Tensor] = None,  # [1, 1] or [B, 1, 1]
    Sigma_beta: Optional[torch.Tensor] = None,  # [1, 1] or [B, 1, 1]
    Sigma_K: Optional[torch.Tensor] = None,  # [4, 4] or [B, 4, 4]
    include_pose: bool = True,
    include_gimbal: bool = True,
    include_intrinsics: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Intrinsics]]:
    """
    Build view parameters from environment state.
    
    Returns:
        R_wc: [B, 3, 3] world-to-camera rotations
        t_wc: [B, 3] camera positions
        e_alpha: [B, 3] gimbal yaw axis in camera frame
        e_beta: [B, 3] gimbal pitch axis in camera frame
        K_list: list of intrinsics per batch
    """
    B = robot_pos.shape[0]
    device = robot_pos.device
    
    # Convert quaternion to rotation matrix (assuming w,x,y,z order)
    # Isaac Lab uses (w,x,y,z) quaternion format
    w, x, y, z = robot_quat[..., 0], robot_quat[..., 1], robot_quat[..., 2], robot_quat[..., 3]
    
    R_wb = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
    ], dim=-2)
    
    # Body to gimbal: Z(yaw) then Y(pitch)
    zeros = torch.zeros(B, device=device)
    R_bg = rotation_matrix_from_euler(zeros, gimbal_pitch, gimbal_yaw, 'ZYX')
    
    # Gimbal to camera (ENU to RDF)
    R_gc = torch.tensor([
        [0., -1., 0.],
        [0., 0., -1.],
        [1., 0., 0.]
    ], device=device).expand(B, 3, 3)
    
    # World to camera
    R_wc = R_wb @ R_bg @ R_gc
    t_wc = robot_pos
    
    # Gimbal axes in camera frame
    R_bc = R_bg @ R_gc
    z_b = torch.tensor([0., 0., 1.], device=device).expand(B, 3)
    e_alpha = torch.bmm(R_bc.transpose(-1, -2), z_b.unsqueeze(-1)).squeeze(-1)
    e_beta = torch.tensor([-1., 0., 0.], device=device).expand(B, 3)
    
    return R_wc, t_wc, e_alpha, e_beta, camera_intrinsics


def jacs_for_view_batch(
    X_w: torch.Tensor,  # [B, 3]
    R_wc: torch.Tensor,  # [B, 3, 3]
    t_wc: torch.Tensor,  # [B, 3]
    e_alpha: torch.Tensor,  # [B, 3]
    e_beta: torch.Tensor,  # [B, 3]
    K: Intrinsics,
    Sigma_pix: torch.Tensor,  # [B, 2, 2]
    Sigma_twb: Optional[torch.Tensor] = None,  # [B, 3, 3]
    Sigma_phiwb: Optional[torch.Tensor] = None,  # [B, 3, 3]
    Sigma_alpha: Optional[torch.Tensor] = None,  # [B, 1, 1]
    Sigma_beta: Optional[torch.Tensor] = None,  # [B, 1, 1]
    Sigma_K: Optional[torch.Tensor] = None,  # [B, 4, 4]
    include_pose: bool = True,
    include_gimbal: bool = True,
    include_intrinsics: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Jacobians for one view per batch element.
    
    Returns:
        JX: [B, 2, 3] - Jacobian wrt X_w
        J_theta: [B, 2, M] - Jacobian wrt nuisance parameters
        Sigma_theta: [B, M, M] - covariance of nuisance parameters
        W_i: [B, 2, 2] - weight matrix (inverse pixel covariance)
    """
    B = X_w.shape[0]
    device = X_w.device
    
    # Transform to camera frame
    Xc = torch.bmm(R_wc.transpose(-1, -2), (X_w - t_wc).unsqueeze(-1)).squeeze(-1)
    
    # Projection Jacobian
    Ju_Xc = proj_jacobian_wrt_Xc(Xc, K)  # [B, 2, 3]
    JX = torch.bmm(Ju_Xc, R_wc.transpose(-1, -2))  # [B, 2, 3]
    
    J_blocks = []
    Sig_blocks = []
    
    # Position uncertainty
    if include_pose and Sigma_twb is not None:
        J_t = torch.bmm(Ju_Xc, -R_wc.transpose(-1, -2))  # [B, 2, 3]
        J_blocks.append(J_t)
        Sig_blocks.append(Sigma_twb)
    
    # Orientation uncertainty
    if include_pose and Sigma_phiwb is not None:
        skew_Xc = skew(Xc)  # [B, 3, 3]
        J_phi = torch.bmm(Ju_Xc, torch.bmm(-skew_Xc, R_wc.transpose(-1, -2)))  # [B, 2, 3]
        J_blocks.append(J_phi)
        Sig_blocks.append(Sigma_phiwb)
    
    # Gimbal yaw uncertainty
    if include_gimbal and Sigma_alpha is not None:
        skew_Xc = skew(Xc)  # [B, 3, 3]
        J_alpha = torch.bmm(Ju_Xc, torch.bmm(-skew_Xc, e_alpha.unsqueeze(-1)))  # [B, 2, 1]
        J_blocks.append(J_alpha)
        Sig_blocks.append(Sigma_alpha)
    
    # Gimbal pitch uncertainty
    if include_gimbal and Sigma_beta is not None:
        skew_Xc = skew(Xc)  # [B, 3, 3]
        J_beta = torch.bmm(Ju_Xc, torch.bmm(-skew_Xc, e_beta.unsqueeze(-1)))  # [B, 2, 1]
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
        ], dim=-2)  # [B, 2, 4]
        J_blocks.append(J_K)
        Sig_blocks.append(Sigma_K)
    
    # Concatenate Jacobians
    if len(J_blocks) > 0:
        J_theta = torch.cat(J_blocks, dim=-1)  # [B, 2, M]
        
        # Block diagonal covariance
        Sigma_theta = torch.zeros(B, J_theta.shape[-1], J_theta.shape[-1], device=device)
        col_idx = 0
        for sig_block in Sig_blocks:
            block_size = sig_block.shape[-1]
            Sigma_theta[:, col_idx:col_idx+block_size, col_idx:col_idx+block_size] = sig_block
            col_idx += block_size
    else:
        J_theta = torch.zeros(B, 2, 0, device=device)
        Sigma_theta = torch.zeros(B, 0, 0, device=device)
    
    # Weight matrix (inverse pixel covariance)
    W_i = torch.linalg.inv(Sigma_pix)
    
    return JX, J_theta, Sigma_theta, W_i


def triangulation_covariance_multi_agent(
    X_w: torch.Tensor,  # [B, 3] - true/estimated target position
    robot_positions: List[torch.Tensor],  # list of [B, 3]
    robot_quats: List[torch.Tensor],  # list of [B, 4]
    gimbal_yaws: List[torch.Tensor],  # list of [B]
    gimbal_pitches: List[torch.Tensor],  # list of [B]
    camera_intrinsics: List[Intrinsics],
    Sigma_pix: torch.Tensor,  # [B, 2, 2] or [2, 2]
    Sigma_twb: Optional[torch.Tensor] = None,
    Sigma_phiwb: Optional[torch.Tensor] = None,
    Sigma_alpha: Optional[torch.Tensor] = None,
    Sigma_beta: Optional[torch.Tensor] = None,
    Sigma_K: Optional[torch.Tensor] = None,
    include_pose: bool = True,
    include_gimbal: bool = True,
    include_intrinsics: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute triangulation covariance for multiple agents (batched).
    
    Args:
        X_w: [B, 3] target position
        robot_positions: list of N agents' positions [B, 3]
        robot_quats: list of N agents' orientations [B, 4]
        gimbal_yaws: list of N agents' gimbal yaw angles [B]
        gimbal_pitches: list of N agents' gimbal pitch angles [B]
        camera_intrinsics: list of N Intrinsics objects
        Sigma_pix: pixel noise covariance [B, 2, 2] or [2, 2]
        ...
    
    Returns:
        Sigma_X: [B, 3, 3] - covariance of triangulated position
        trace_cov: [B] - trace of covariance (scalar uncertainty metric)
    """
    B = X_w.shape[0]
    N = len(robot_positions)
    device = X_w.device
    
    # Expand Sigma_pix if needed
    if Sigma_pix.dim() == 2:
        Sigma_pix = Sigma_pix.unsqueeze(0).expand(B, -1, -1)
    
    # Collect Jacobians from all views
    JX_list = []
    J_theta_list = []
    Sigma_theta_list = []
    W_list = []
    Sigma_z_list = []
    
    for i in range(N):
        # Build view parameters
        R_wc, t_wc, e_alpha, e_beta, _ = build_views_from_env_state(
            robot_positions[i], robot_quats[i], gimbal_yaws[i], gimbal_pitches[i],
            [camera_intrinsics[i]] * B, Sigma_pix,
            Sigma_twb, Sigma_phiwb, Sigma_alpha, Sigma_beta, Sigma_K,
            include_pose, include_gimbal, include_intrinsics
        )
        
        # Compute Jacobians
        JX, J_theta, Sigma_theta, W_i = jacs_for_view_batch(
            X_w, R_wc, t_wc, e_alpha, e_beta, camera_intrinsics[i],
            Sigma_pix, Sigma_twb, Sigma_phiwb, Sigma_alpha, Sigma_beta, Sigma_K,
            include_pose, include_gimbal, include_intrinsics
        )
        
        JX_list.append(JX)
        J_theta_list.append(J_theta)
        Sigma_theta_list.append(Sigma_theta)
        W_list.append(W_i)
        Sigma_z_list.append(Sigma_pix)
    
    # Stack along the view dimension: [B, 2N, ...]
    JX_full = torch.cat(JX_list, dim=1)  # [B, 2N, 3]
    
    # Block diagonal weight matrix
    W_full = torch.zeros(B, 2*N, 2*N, device=device)
    for i, W_i in enumerate(W_list):
        W_full[:, 2*i:2*i+2, 2*i:2*i+2] = W_i
    
    # Block diagonal pixel covariance
    Sigma_z_full = torch.zeros(B, 2*N, 2*N, device=device)
    for i in range(N):
        Sigma_z_full[:, 2*i:2*i+2, 2*i:2*i+2] = Sigma_z_list[i]
    
    # Block diagonal nuisance parameter Jacobian
    M_total = sum(j.shape[-1] for j in J_theta_list)
    J_theta_full = torch.zeros(B, 2*N, M_total, device=device)
    Sigma_theta_full = torch.zeros(B, M_total, M_total, device=device)
    
    row_idx = 0
    col_idx = 0
    for i in range(N):
        J_theta_i = J_theta_list[i]
        Sigma_theta_i = Sigma_theta_list[i]
        M_i = J_theta_i.shape[-1]
        
        J_theta_full[:, row_idx:row_idx+2, col_idx:col_idx+M_i] = J_theta_i
        Sigma_theta_full[:, col_idx:col_idx+M_i, col_idx:col_idx+M_i] = Sigma_theta_i
        
        row_idx += 2
        col_idx += M_i
    
    # Residual covariance: S = Sigma_z + J_theta @ Sigma_theta @ J_theta^T
    S_resid = Sigma_z_full + torch.bmm(torch.bmm(J_theta_full, Sigma_theta_full), 
                                        J_theta_full.transpose(-1, -2))
    
    # Normal matrix: A = JX^T @ W @ JX
    A = torch.bmm(torch.bmm(JX_full.transpose(-1, -2), W_full), JX_full)  # [B, 3, 3]
    
    # Covariance: Sigma_X = A^{-1} @ JX^T @ W @ S @ W @ JX @ A^{-T}
    try:
        A_inv = torch.linalg.inv(A)
    except:
        # Use pseudo-inverse if singular
        A_inv = torch.linalg.pinv(A)
    
    middle = torch.bmm(torch.bmm(JX_full.transpose(-1, -2), W_full), 
                       torch.bmm(S_resid, torch.bmm(W_full, JX_full)))
    Sigma_X = torch.bmm(torch.bmm(A_inv, middle), A_inv.transpose(-1, -2))
    
    # Trace of covariance
    trace_cov = torch.diagonal(Sigma_X, dim1=-2, dim2=-1).sum(dim=-1)
    
    return Sigma_X, trace_cov


def triangulation_covariance_simple(
    X_w: torch.Tensor,  # [B, 3]
    robot_positions: List[torch.Tensor],
    robot_quats: List[torch.Tensor],
    gimbal_yaws: List[torch.Tensor],
    gimbal_pitches: List[torch.Tensor],
    camera_intrinsics: List[Intrinsics],
    pixel_std: float = 0.5  # pixels
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified version using only pixel noise (fastest computation).
    
    Returns:
        Sigma_X: [B, 3, 3]
        trace_cov: [B]
    """
    B = X_w.shape[0]
    device = X_w.device
    
    # Simple pixel covariance (isotropic)
    Sigma_pix = torch.eye(2, device=device) * (pixel_std ** 2)
    
    return triangulation_covariance_multi_agent(
        X_w, robot_positions, robot_quats, gimbal_yaws, gimbal_pitches,
        camera_intrinsics, Sigma_pix,
        Sigma_twb=None, Sigma_phiwb=None, Sigma_alpha=None, Sigma_beta=None,
        include_pose=False, include_gimbal=False, include_intrinsics=False
    )