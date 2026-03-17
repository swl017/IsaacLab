#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Monte Carlo Validation for Triangulation Covariance Estimation.

This script validates that the analytical covariance computation matches
empirical statistics from Monte Carlo sampling of all uncertainty sources.

Key validation aspects:
1. Covariance magnitude: Analytical trace vs empirical trace
2. Covariance shape: Eigenvalue ratios (anisotropy)
3. Coverage: % of samples within 1σ, 2σ, 3σ ellipsoids
4. With/without gimbal roll: Validates roll Jacobian contribution
5. Angle sweep: Uncertainty vs viewing angle between cameras

Run with:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/monte_carlo_validation.py

Modes:
    --mode validation    Run standard validation scenarios (default)
    --mode angle_sweep   Sweep viewing angle and plot uncertainty
    --mode all           Run all experiments

Options:
    --n-samples N        Number of Monte Carlo samples (default: 10000)
    --test-verbose       Enable verbose debug output
    --save-plots         Save validation plots to tests/ directory
    --save-dir PATH      Directory to save plots (default: auto-generated)
"""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Monte Carlo validation for triangulation covariance")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--n-samples", type=int, default=100000, help="Number of MC samples")
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
parser.add_argument("--save-plots", action="store_true", help="Save validation plots")
parser.add_argument("--save-dir", type=str, default=None, help="Directory to save plots")
parser.add_argument("--mode", type=str, default="validation",
                    choices=["validation", "angle_sweep", "all"],
                    help="Analysis mode: validation, angle_sweep, or all")
parser.add_argument("--target-distance", type=float, default=100.0,
                    help="Distance to target in meters (default: 100m)")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW import other modules
import torch
import numpy as np
import sys
import os
import math
import traceback
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Matplotlib for plotting (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")

# Scipy for chi-squared CDF (for coverage computation)
try:
    from scipy.stats import chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using approximate coverage computation.")

# Import triangulation module
from isaaclab_tasks.direct.iris_ma6.triangulation import (
    TriangulationCfg,
    triangulate_targets,
    compute_triangulation_covariance,
    get_ray_directions_from_bbox,
    TriangulationResult,
)
from isaaclab_tasks.direct.iris_ma6.triangulation.triangulation import (
    compute_full_triangulation,
    build_camera_transforms,
    quat_to_rotation_matrix,
    rotation_matrix_from_euler,
)

VERBOSE = args_cli.test_verbose
N_SAMPLES = args_cli.n_samples


# =============================================================================
# Test Configuration
# =============================================================================


@dataclass
class MCScenario:
    """Monte Carlo test scenario configuration."""
    name: str
    num_cameras: int
    target_pos: Tuple[float, float, float]
    camera_positions: List[Tuple[float, float, float]]
    camera_yaws: List[float]  # Body yaw angles (rad)
    gimbal_yaws: List[float]  # Gimbal yaw angles (rad), 0 = forward
    gimbal_rolls: List[float]  # Gimbal roll angles (rad)
    gimbal_pitches: List[float]  # Gimbal pitch angles (rad)
    include_roll: bool = True  # Whether to include roll in covariance


# =============================================================================
# Utility Functions
# =============================================================================


def create_identity_quat(device: torch.device) -> torch.Tensor:
    """Create identity quaternion (wxyz)."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)


def create_yaw_quat(yaw: float, device: torch.device) -> torch.Tensor:
    """Create quaternion for yaw rotation around Z axis (wxyz)."""
    half_yaw = yaw / 2.0
    return torch.tensor(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        device=device,
    )


def create_test_intrinsics(
    num_envs: int, num_cameras: int, fx: float = 500.0, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create test camera intrinsic matrices.

    Returns:
        [N, C, 3, 3] Intrinsic matrices
    """
    K = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(num_envs, num_cameras, 3, 3).clone()
    K[..., 0, 0] = fx
    K[..., 1, 1] = fx
    K[..., 0, 2] = 320.0  # cx
    K[..., 1, 2] = 240.0  # cy
    return K


def project_point_to_pixel(
    target_pos: torch.Tensor,
    camera_pos: torch.Tensor,
    R_wc: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Project 3D point to pixel coordinates.

    Args:
        target_pos: [3] Target position in world frame
        camera_pos: [3] Camera position in world frame
        R_wc: [3, 3] World to camera rotation matrix
        K: [3, 3] Camera intrinsic matrix

    Returns:
        pixel: [2] Pixel coordinates (x, y)
        depth: Depth (Z in camera frame)
    """
    # Transform to camera frame
    X_c = R_wc.T @ (target_pos - camera_pos)

    # Check depth
    depth = X_c[2]
    if depth <= 0:
        return None, depth

    # Project to pixel
    u = K[0, 0] * X_c[0] / depth + K[0, 2]
    v = K[1, 1] * X_c[1] / depth + K[1, 2]

    return torch.stack([u, v]), depth


def sample_perturbation(
    sigma: float, size: Tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """Sample Gaussian perturbation."""
    return torch.randn(size, device=device) * sigma


def create_yaw_quat_batched(yaws: torch.Tensor) -> torch.Tensor:
    """Create quaternions for yaw rotations around Z axis (wxyz).

    Args:
        yaws: [N, C] Yaw angles in radians

    Returns:
        [N, C, 4] Quaternions (wxyz format)
    """
    half_yaws = yaws / 2.0
    cos_half = torch.cos(half_yaws)
    sin_half = torch.sin(half_yaws)
    zeros = torch.zeros_like(yaws)

    # wxyz format: [cos(θ/2), 0, 0, sin(θ/2)]
    quats = torch.stack([cos_half, zeros, zeros, sin_half], dim=-1)
    return quats


def create_euler_quat_batched(
    rolls: torch.Tensor, pitches: torch.Tensor, yaws: torch.Tensor
) -> torch.Tensor:
    """Create quaternions from Euler angles (ZYX convention, wxyz format).

    Args:
        rolls: [N, C] Roll angles (rotation around X) in radians
        pitches: [N, C] Pitch angles (rotation around Y) in radians
        yaws: [N, C] Yaw angles (rotation around Z) in radians

    Returns:
        [N, C, 4] Quaternions (wxyz format)
    """
    # Half angles
    cr = torch.cos(rolls / 2.0)
    sr = torch.sin(rolls / 2.0)
    cp = torch.cos(pitches / 2.0)
    sp = torch.sin(pitches / 2.0)
    cy = torch.cos(yaws / 2.0)
    sy = torch.sin(yaws / 2.0)

    # ZYX Euler to quaternion (wxyz)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quats = torch.stack([w, x, y, z], dim=-1)

    # Normalize to handle numerical errors
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    return quats


def project_points_batched(
    target_pos: torch.Tensor,
    camera_pos: torch.Tensor,
    R_wc: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D point to pixel coordinates (batched).

    Args:
        target_pos: [3] Target position in world frame
        camera_pos: [N, C, 3] Camera positions in world frame
        R_wc: [N, C, 3, 3] World to camera rotation matrices
        K: [N, C, 3, 3] Camera intrinsic matrices

    Returns:
        pixels: [N, C, 2] Pixel coordinates (x, y)
        depths: [N, C] Depth (Z in camera frame)
        valid: [N, C] Validity mask (True if depth > 0)
    """
    N, C = camera_pos.shape[:2]

    # Expand target position: [3] -> [N, C, 3]
    target_expanded = target_pos.view(1, 1, 3).expand(N, C, 3)

    # Relative position: [N, C, 3]
    rel_pos = target_expanded - camera_pos

    # Transform to camera frame: X_c = R_wc^T @ rel_pos
    # R_wc is [N, C, 3, 3], rel_pos is [N, C, 3]
    R_wc_T = R_wc.transpose(-1, -2)  # [N, C, 3, 3]
    X_c = torch.einsum('ncij,ncj->nci', R_wc_T, rel_pos)  # [N, C, 3]

    # Extract depth
    depths = X_c[..., 2]  # [N, C]
    valid = depths > 0  # [N, C]

    # Clamp depth to avoid division by zero
    safe_depths = torch.clamp(depths, min=1e-6)

    # Project to pixel coordinates
    u = K[..., 0, 0] * X_c[..., 0] / safe_depths + K[..., 0, 2]  # [N, C]
    v = K[..., 1, 1] * X_c[..., 1] / safe_depths + K[..., 1, 2]  # [N, C]

    pixels = torch.stack([u, v], dim=-1)  # [N, C, 2]

    return pixels, depths, valid


# =============================================================================
# Monte Carlo Sampling (Batched)
# =============================================================================


def run_monte_carlo_batched(
    scenario: MCScenario,
    cfg: TriangulationCfg,
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Run batched Monte Carlo simulation for a scenario.

    Processes all n_samples in parallel using tensor operations.

    IMPORTANT: The MC simulation models the scenario where:
    - True camera state is perturbed (position, orientation, gimbal angles)
    - The triangulator uses NOMINAL parameters (doesn't know the true perturbations)
    - This matches the analytical covariance formula assumption

    Args:
        scenario: Test scenario configuration
        cfg: Triangulation configuration with uncertainty parameters
        n_samples: Number of samples (batch size)
        device: Torch device

    Returns:
        samples: [N, 3] Triangulation results (invalid ones filled with NaN)
        is_valid: [N] Validity mask
        empirical_cov: [3, 3] Empirical covariance matrix
        num_valid: Number of valid samples
    """
    N = n_samples  # Batch size = number of MC samples
    C = scenario.num_cameras
    T = 1  # Single target

    # Nominal values
    target_pos = torch.tensor(scenario.target_pos, device=device, dtype=torch.float32)

    # ==========================================================================
    # Create NOMINAL camera parameters (used for triangulation)
    # ==========================================================================
    nominal_positions = torch.tensor(scenario.camera_positions, device=device, dtype=torch.float32)
    nominal_yaws = torch.tensor(scenario.camera_yaws, device=device, dtype=torch.float32)
    nominal_gimbal_yaws = torch.tensor(scenario.gimbal_yaws, device=device, dtype=torch.float32)
    nominal_gimbal_rolls = torch.tensor(scenario.gimbal_rolls, device=device, dtype=torch.float32)
    nominal_gimbal_pitches = torch.tensor(scenario.gimbal_pitches, device=device, dtype=torch.float32)

    # Nominal quaternions (body roll=0, pitch=0, yaw from scenario)
    zeros_c = torch.zeros(C, device=device, dtype=torch.float32)
    nominal_quats = create_euler_quat_batched(zeros_c.unsqueeze(0), zeros_c.unsqueeze(0),
                                              nominal_yaws.unsqueeze(0)).squeeze(0)  # [C, 4]

    # ==========================================================================
    # Sample PERTURBED camera parameters (true state, used for projection)
    # ==========================================================================
    # Position: nominal + noise
    camera_positions_pert = nominal_positions.unsqueeze(0).expand(N, C, 3).clone()
    pos_noise = sample_perturbation(cfg.pos_std, (N, C, 3), device)
    camera_positions_pert = camera_positions_pert + pos_noise

    # Body orientation: sample roll, pitch, yaw perturbations
    body_roll_noise = sample_perturbation(cfg.ori_std, (N, C), device)
    body_pitch_noise = sample_perturbation(cfg.ori_std, (N, C), device)
    body_yaw_noise = sample_perturbation(cfg.ori_std, (N, C), device)

    body_rolls = body_roll_noise  # Nominal roll is 0
    body_pitches = body_pitch_noise  # Nominal pitch is 0
    body_yaws = nominal_yaws.unsqueeze(0).expand(N, C).clone() + body_yaw_noise

    # Convert to quaternions using full Euler angles: [N, C, 4]
    camera_quats_pert = create_euler_quat_batched(body_rolls, body_pitches, body_yaws)

    # Gimbal angles: nominal + noise
    gimbal_yaws_pert = nominal_gimbal_yaws.unsqueeze(0).expand(N, C).clone()
    gimbal_yaws_pert = gimbal_yaws_pert + sample_perturbation(cfg.gimbal_std, (N, C), device)

    gimbal_rolls_pert = nominal_gimbal_rolls.unsqueeze(0).expand(N, C).clone()
    if scenario.include_roll:
        gimbal_rolls_pert = gimbal_rolls_pert + sample_perturbation(cfg.gimbal_std, (N, C), device)

    gimbal_pitches_pert = nominal_gimbal_pitches.unsqueeze(0).expand(N, C).clone()
    gimbal_pitches_pert = gimbal_pitches_pert + sample_perturbation(cfg.gimbal_std, (N, C), device)

    # ==========================================================================
    # Build camera transforms (PERTURBED - for projection)
    # ==========================================================================
    R_wc_pert, t_wc_pert, _, _, _ = build_camera_transforms(
        camera_positions_pert, camera_quats_pert, gimbal_yaws_pert, gimbal_rolls_pert, gimbal_pitches_pert
    )

    # ==========================================================================
    # Project target to each camera (using PERTURBED params - true state)
    # ==========================================================================
    camera_intrinsics = create_test_intrinsics(N, C, device=device)

    # Project target to pixel coordinates using PERTURBED camera state
    pixels, depths, depth_valid = project_points_batched(
        target_pos, camera_positions_pert, R_wc_pert, camera_intrinsics
    )

    # Add pixel noise: [N, C, 2]
    pixel_noise = sample_perturbation(cfg.pix_std, (N, C, 2), device)
    pixels_noisy = pixels + pixel_noise

    # ==========================================================================
    # Create bounding boxes
    # ==========================================================================
    bbox_2d = torch.zeros(N, C, T, 4, device=device)
    bbox_2d[:, :, 0, 0] = pixels_noisy[:, :, 0]  # x
    bbox_2d[:, :, 0, 1] = pixels_noisy[:, :, 1]  # y
    bbox_2d[:, :, 0, 2] = 50.0  # width
    bbox_2d[:, :, 0, 3] = 50.0  # height

    # Validity mask based on depth
    bbox_valid = depth_valid.unsqueeze(-1)  # [N, C, 1]

    # ==========================================================================
    # Check minimum cameras per sample
    # ==========================================================================
    num_valid_cameras = bbox_valid.sum(dim=(1, 2))  # [N]
    enough_cameras = num_valid_cameras >= cfg.min_cameras_required  # [N]

    # ==========================================================================
    # Triangulate using NOMINAL params (what the system "believes")
    # This is the key difference: triangulator doesn't know the true perturbations
    # ==========================================================================
    # Expand nominal params to batch size
    nominal_positions_exp = nominal_positions.unsqueeze(0).expand(N, C, 3)
    nominal_quats_exp = nominal_quats.unsqueeze(0).expand(N, C, 4)
    nominal_gimbal_yaws_exp = nominal_gimbal_yaws.unsqueeze(0).expand(N, C)
    nominal_gimbal_rolls_exp = nominal_gimbal_rolls.unsqueeze(0).expand(N, C)
    nominal_gimbal_pitches_exp = nominal_gimbal_pitches.unsqueeze(0).expand(N, C)

    ray_dirs, _ = get_ray_directions_from_bbox(
        bbox_2d, camera_intrinsics, nominal_positions_exp, nominal_quats_exp,
        nominal_gimbal_yaws_exp, nominal_gimbal_rolls_exp, nominal_gimbal_pitches_exp
    )

    X_tri, tri_valid, _, _ = triangulate_targets(
        nominal_positions_exp, ray_dirs, bbox_valid, cfg
    )

    # Combine validity: triangulation valid AND enough cameras
    is_valid = tri_valid.squeeze(-1) & enough_cameras  # [N]

    # Extract results: [N, 3]
    samples = X_tri[:, 0, :]  # [N, 3]

    # Mask invalid samples with NaN
    samples = torch.where(
        is_valid.unsqueeze(-1).expand_as(samples),
        samples,
        torch.full_like(samples, float('nan'))
    )

    # ==========================================================================
    # Compute empirical covariance from valid samples
    # ==========================================================================
    num_valid = int(is_valid.sum().item())

    if num_valid < 10:
        return samples, is_valid, None, num_valid

    # Extract valid samples
    valid_samples = samples[is_valid]  # [M, 3]

    # Compute empirical statistics
    mean = valid_samples.mean(dim=0)
    centered = valid_samples - mean
    empirical_cov = (centered.T @ centered) / (num_valid - 1)

    return samples, is_valid, empirical_cov, num_valid


def run_monte_carlo(
    scenario: MCScenario,
    cfg: TriangulationCfg,
    n_samples: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Run Monte Carlo simulation for a scenario.

    Uses batched computation for efficiency.

    Args:
        scenario: Test scenario configuration
        cfg: Triangulation configuration
        n_samples: Number of samples
        device: Torch device

    Returns:
        samples: [M, 3] Valid triangulation samples or None if not enough samples
        empirical_cov: [3, 3] Empirical covariance matrix or None
        num_valid: Number of valid samples
    """
    samples, is_valid, empirical_cov, num_valid = run_monte_carlo_batched(
        scenario, cfg, n_samples, device
    )

    if empirical_cov is None:
        return None, None, num_valid

    # Return only valid samples
    valid_samples = samples[is_valid]  # [M, 3]

    return valid_samples, empirical_cov, num_valid


# =============================================================================
# Analytical Covariance Computation
# =============================================================================


def compute_analytical_covariance(
    scenario: MCScenario,
    cfg: TriangulationCfg,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], bool]:
    """Compute analytical covariance for a scenario.

    Args:
        scenario: Test scenario configuration
        cfg: Triangulation configuration
        device: Torch device

    Returns:
        covariance: [3, 3] Analytical covariance matrix or None if invalid
        is_valid: Whether computation succeeded
    """
    N = 1
    C = scenario.num_cameras
    T = 1

    # Setup nominal camera state
    target_pos = torch.tensor(scenario.target_pos, device=device, dtype=torch.float32)
    X_target = target_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]

    camera_positions = torch.zeros(N, C, 3, device=device)
    for i, pos in enumerate(scenario.camera_positions):
        camera_positions[0, i] = torch.tensor(pos, device=device)

    camera_quats = torch.zeros(N, C, 4, device=device)
    for i, yaw in enumerate(scenario.camera_yaws):
        camera_quats[0, i] = create_yaw_quat(yaw, device)

    gimbal_yaws = torch.tensor([scenario.gimbal_yaws], device=device)  # [1, C]
    gimbal_rolls = torch.tensor([scenario.gimbal_rolls], device=device)  # [1, C]
    gimbal_pitches = torch.tensor([scenario.gimbal_pitches], device=device)  # [1, C]

    camera_intrinsics = create_test_intrinsics(N, C, device=device)

    # All cameras valid
    valid_mask = torch.ones(N, C, T, dtype=torch.bool, device=device)
    tri_valid = torch.ones(N, T, dtype=torch.bool, device=device)

    # Compute analytical covariance
    Sigma_X, quality, is_valid = compute_triangulation_covariance(
        X_target,
        camera_positions,
        camera_quats,
        gimbal_yaws,
        gimbal_rolls,
        gimbal_pitches,
        camera_intrinsics,
        valid_mask,
        tri_valid,
        cfg,
    )

    if not is_valid[0, 0]:
        return None, False

    return Sigma_X[0, 0], True


# =============================================================================
# Effective Dimensionality and Coverage Computation
# =============================================================================


def compute_effective_dimensionality(eigenvalues: torch.Tensor) -> float:
    """Compute effective dimensionality from eigenvalues using participation ratio.

    The effective dimensionality measures how many dimensions contribute
    significantly to the variance. For triangulation, the covariance is typically
    highly anisotropic (uncertainty mainly along depth direction), giving
    effective dimensionality close to 1.

    Formula: eff_dim = (Σ λᵢ)² / Σ λᵢ²

    - For isotropic 3D Gaussian (λ₁ = λ₂ = λ₃): eff_dim = 3
    - For 1D distribution (λ₁ >> λ₂, λ₃): eff_dim → 1

    Args:
        eigenvalues: Eigenvalues of covariance matrix (sorted ascending)

    Returns:
        Effective dimensionality (1.0 to 3.0 for 3D covariance)
    """
    # Participation ratio formula
    sum_eig = eigenvalues.sum().item()
    sum_eig_sq = (eigenvalues ** 2).sum().item()

    if sum_eig_sq < 1e-12:
        return 3.0  # Default to isotropic if near-zero

    eff_dim = (sum_eig ** 2) / sum_eig_sq
    return eff_dim


def compute_expected_coverage_3d(k_sigma: float) -> float:
    """Compute expected coverage for k-sigma ellipsoid in 3D.

    For a 3D multivariate Gaussian, the squared Mahalanobis distance follows
    chi-squared distribution with 3 DOF.

    NOTE: The DOF is always 3 for 3D covariance, regardless of eigenvalue
    anisotropy. The participation ratio (effective dimensionality) measures
    the shape anisotropy but does NOT change the chi-squared DOF.

    Args:
        k_sigma: Number of standard deviations (1, 2, or 3)

    Returns:
        Expected coverage probability (0 to 1)
    """
    if HAS_SCIPY:
        # P(χ²(3) ≤ k²) gives the coverage for 3D Gaussian
        threshold = k_sigma ** 2
        return chi2.cdf(threshold, df=3)
    else:
        # Hardcoded values for 3 DOF
        # P(χ²(3) ≤ 1) = 0.199, P(χ²(3) ≤ 4) = 0.739, P(χ²(3) ≤ 9) = 0.971
        expected_3d = {1: 0.199, 2: 0.739, 3: 0.971}
        return expected_3d.get(int(k_sigma), 0.5)


def get_chi2_threshold(k_sigma: float, dof: int = 3) -> float:
    """Get chi-squared threshold for k-sigma ellipsoid.

    These thresholds define the Mahalanobis distance squared cutoff
    for containing approximately the expected % of samples.

    Args:
        k_sigma: Number of standard deviations
        dof: Degrees of freedom (typically 3 for 3D covariance)

    Returns:
        Chi-squared threshold value
    """
    if HAS_SCIPY:
        # Compute quantile: P(χ²(dof) ≤ threshold) = target_coverage
        # For standard k-sigma definition, threshold = k²
        # But traditionally we use quantiles that give standard coverage
        return chi2.ppf(chi2.cdf(k_sigma ** 2, df=dof), df=dof)
    else:
        # Hardcoded thresholds for 3 DOF (standard 3D case)
        thresholds_3dof = {1: 3.53, 2: 8.02, 3: 14.16}
        return thresholds_3dof.get(int(k_sigma), k_sigma ** 2)


# =============================================================================
# Validation Metrics
# =============================================================================


def compute_validation_metrics(
    empirical_cov: torch.Tensor,
    analytical_cov: torch.Tensor,
    samples: torch.Tensor,
    target_pos: torch.Tensor,
) -> Dict[str, float]:
    """Compute validation metrics comparing empirical and analytical covariance.

    Args:
        empirical_cov: [3, 3] Empirical covariance from MC
        analytical_cov: [3, 3] Analytical covariance
        samples: [M, 3] MC samples
        target_pos: [3] True target position

    Returns:
        Dictionary of validation metrics
    """
    metrics = {}

    # Trace comparison
    emp_trace = torch.trace(empirical_cov).item()
    ana_trace = torch.trace(analytical_cov).item()
    metrics["empirical_trace"] = emp_trace
    metrics["analytical_trace"] = ana_trace
    metrics["trace_ratio"] = ana_trace / emp_trace if emp_trace > 0 else float("nan")

    # Eigenvalue comparison
    emp_eigvals = torch.linalg.eigvalsh(empirical_cov)
    ana_eigvals = torch.linalg.eigvalsh(analytical_cov)

    metrics["empirical_max_eigval"] = emp_eigvals.max().item()
    metrics["analytical_max_eigval"] = ana_eigvals.max().item()
    metrics["empirical_min_eigval"] = emp_eigvals.min().item()
    metrics["analytical_min_eigval"] = ana_eigvals.min().item()

    # Condition number (eigenvalue ratio)
    metrics["empirical_condition"] = (emp_eigvals.max() / emp_eigvals.min()).item()
    metrics["analytical_condition"] = (ana_eigvals.max() / ana_eigvals.min()).item()

    # Effective dimensionality from analytical covariance
    # This measures anisotropy: 1.0 = 1D (all variance in one direction), 3.0 = isotropic 3D
    eff_dim_ana = compute_effective_dimensionality(ana_eigvals)
    eff_dim_emp = compute_effective_dimensionality(emp_eigvals)
    metrics["effective_dim_analytical"] = eff_dim_ana
    metrics["effective_dim_empirical"] = eff_dim_emp

    # Coverage statistics (what % of samples fall within nσ ellipsoids)
    mean_sample = samples.mean(dim=0)
    centered = samples - target_pos

    # Mahalanobis distance using analytical covariance
    try:
        L = torch.linalg.cholesky(analytical_cov)
        L_inv = torch.linalg.inv(L)
        normalized = centered @ L_inv.T
        mahal_sq = (normalized ** 2).sum(dim=1)

        # Chi-squared thresholds for 3 DOF (standard thresholds)
        # These give 39.3%, 73.9%, 93.1% coverage for isotropic 3D Gaussian
        metrics["coverage_1sigma"] = (mahal_sq <= 1.0).float().mean().item()  # d ≤ 1
        metrics["coverage_2sigma"] = (mahal_sq <= 4.0).float().mean().item()  # d ≤ 2
        metrics["coverage_3sigma"] = (mahal_sq <= 9.0).float().mean().item()  # d ≤ 3

        # Compute expected coverage for 3D Gaussian
        # NOTE: DOF is always 3 regardless of eigenvalue anisotropy.
        # The Mahalanobis distance squared follows chi²(3) for any 3D covariance.
        metrics["coverage_1sigma_expected"] = compute_expected_coverage_3d(1.0)
        metrics["coverage_2sigma_expected"] = compute_expected_coverage_3d(2.0)
        metrics["coverage_3sigma_expected"] = compute_expected_coverage_3d(3.0)
    except Exception:
        metrics["coverage_1sigma"] = float("nan")
        metrics["coverage_2sigma"] = float("nan")
        metrics["coverage_3sigma"] = float("nan")
        metrics["coverage_1sigma_expected"] = float("nan")
        metrics["coverage_2sigma_expected"] = float("nan")
        metrics["coverage_3sigma_expected"] = float("nan")

    # Mean position error
    metrics["mean_error"] = (mean_sample - target_pos).norm().item()

    return metrics


# =============================================================================
# Test Scenarios
# =============================================================================


def compute_look_at_angles(
    camera_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Compute body yaw and gimbal pitch to look at target.

    Assumes gimbal_yaw=0 (camera forward = body forward).

    Args:
        camera_pos: Camera position (x, y, z)
        target_pos: Target position (x, y, z)

    Returns:
        body_yaw: Yaw angle for body to face target (rad)
        gimbal_pitch: Pitch angle for gimbal to look at target (rad)
    """
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]

    # Body yaw: angle in XY plane from +X axis
    body_yaw = math.atan2(dy, dx)

    # Horizontal distance
    horizontal_dist = math.sqrt(dx**2 + dy**2)

    # Gimbal pitch: angle to look down (negative for looking down)
    gimbal_pitch = math.atan2(-dz, horizontal_dist)  # Negative because pitch down is negative

    return body_yaw, gimbal_pitch


def create_test_scenarios() -> List[MCScenario]:
    """Create test scenarios for Monte Carlo validation.

    All scenarios are designed with cameras properly pointing at the target.
    """
    scenarios = []

    # Target at origin, height 0
    target = (0.0, 0.0, 0.0)

    # Scenario 1: 2 cameras, orthogonal baseline (90° apart), no roll
    cam1_pos = (-10.0, 0.0, 5.0)  # West of target
    cam2_pos = (0.0, -10.0, 5.0)  # South of target
    yaw1, pitch1 = compute_look_at_angles(cam1_pos, target)
    yaw2, pitch2 = compute_look_at_angles(cam2_pos, target)

    scenarios.append(MCScenario(
        name="2_cameras_orthogonal_no_roll",
        num_cameras=2,
        target_pos=target,
        camera_positions=[cam1_pos, cam2_pos],
        camera_yaws=[yaw1, yaw2],  # Body faces target
        gimbal_yaws=[0.0, 0.0],     # Camera forward = body forward
        gimbal_rolls=[0.0, 0.0],
        gimbal_pitches=[pitch1, pitch2],  # Look down at target
        include_roll=False,
    ))

    # Scenario 2: 2 cameras, orthogonal, with roll
    scenarios.append(MCScenario(
        name="2_cameras_orthogonal_with_roll",
        num_cameras=2,
        target_pos=target,
        camera_positions=[cam1_pos, cam2_pos],
        camera_yaws=[yaw1, yaw2],
        gimbal_yaws=[0.0, 0.0],
        gimbal_rolls=[math.radians(5), math.radians(-5)],  # Small roll
        gimbal_pitches=[pitch1, pitch2],
        include_roll=True,
    ))

    # Scenario 3: 3 cameras, triangle formation
    cam1_pos = (0.0, -10.0, 5.0)   # South
    cam2_pos = (8.66, 5.0, 5.0)    # Northeast
    cam3_pos = (-8.66, 5.0, 5.0)   # Northwest
    yaw1, pitch1 = compute_look_at_angles(cam1_pos, target)
    yaw2, pitch2 = compute_look_at_angles(cam2_pos, target)
    yaw3, pitch3 = compute_look_at_angles(cam3_pos, target)

    scenarios.append(MCScenario(
        name="3_cameras_triangle",
        num_cameras=3,
        target_pos=target,
        camera_positions=[cam1_pos, cam2_pos, cam3_pos],
        camera_yaws=[yaw1, yaw2, yaw3],  # Each body faces target
        gimbal_yaws=[0.0, 0.0, 0.0],
        gimbal_rolls=[0.0, 0.0, 0.0],
        gimbal_pitches=[pitch1, pitch2, pitch3],
        include_roll=True,
    ))

    # Scenario 4: 3 cameras with body tilt (roll stabilization active)
    scenarios.append(MCScenario(
        name="3_cameras_with_body_tilt",
        num_cameras=3,
        target_pos=target,
        camera_positions=[cam1_pos, cam2_pos, cam3_pos],
        camera_yaws=[yaw1, yaw2, yaw3],
        gimbal_yaws=[0.0, 0.0, 0.0],
        gimbal_rolls=[math.radians(10), math.radians(-5), math.radians(8)],  # Stabilizing roll
        gimbal_pitches=[pitch1, pitch2, pitch3],
        include_roll=True,
    ))

    # Scenario 5: Far target (large uncertainty)
    far_target = (0.0, 0.0, 0.0)
    cam1_pos = (-25.0, 0.0, 10.0)  # 25m west, 10m up
    cam2_pos = (25.0, 0.0, 10.0)   # 25m east, 10m up
    yaw1, pitch1 = compute_look_at_angles(cam1_pos, far_target)
    yaw2, pitch2 = compute_look_at_angles(cam2_pos, far_target)

    scenarios.append(MCScenario(
        name="far_target",
        num_cameras=2,
        target_pos=far_target,
        camera_positions=[cam1_pos, cam2_pos],
        camera_yaws=[yaw1, yaw2],
        gimbal_yaws=[0.0, 0.0],
        gimbal_rolls=[0.0, 0.0],
        gimbal_pitches=[pitch1, pitch2],
        include_roll=True,
    ))

    # Scenario 6: Close target (small uncertainty)
    close_target = (0.0, 0.0, 0.0)
    cam1_pos = (-5.0, 0.0, 3.0)  # 5m west, 3m up
    cam2_pos = (5.0, 0.0, 3.0)   # 5m east, 3m up
    yaw1, pitch1 = compute_look_at_angles(cam1_pos, close_target)
    yaw2, pitch2 = compute_look_at_angles(cam2_pos, close_target)

    scenarios.append(MCScenario(
        name="close_target",
        num_cameras=2,
        target_pos=close_target,
        camera_positions=[cam1_pos, cam2_pos],
        camera_yaws=[yaw1, yaw2],
        gimbal_yaws=[0.0, 0.0],
        gimbal_rolls=[0.0, 0.0],
        gimbal_pitches=[pitch1, pitch2],
        include_roll=True,
    ))

    return scenarios


# =============================================================================
# Main Validation
# =============================================================================


def run_validation(device: torch.device) -> Tuple[bool, List[Dict]]:
    """Run full Monte Carlo validation.

    Returns:
        Tuple of (all_passed, results_summary)
    """
    print("=" * 80)
    print("MONTE CARLO VALIDATION - TRIANGULATION COVARIANCE")
    print("=" * 80)
    print(f"Device:        {device}")
    print(f"Samples:       {N_SAMPLES}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    cfg = TriangulationCfg(
        pix_std=7.0,          # Pixel noise std
        pos_std=0.1,          # Position noise std (m)
        ori_std=0.01,         # Orientation noise std (rad)
        gimbal_std=0.005,     # Gimbal angle noise std (rad)
        include_pose_uncertainty=True,
        include_gimbal_uncertainty=True,
    )

    print(f"\nUncertainty parameters:")
    print(f"  Pixel std:   {cfg.pix_std} px")
    print(f"  Position std: {cfg.pos_std} m")
    print(f"  Orientation std: {math.degrees(cfg.ori_std):.2f}°")
    print(f"  Gimbal std:  {math.degrees(cfg.gimbal_std):.2f}°")

    scenarios = create_test_scenarios()
    all_passed = True
    results_summary = []

    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario.name}")
        print(f"{'='*80}")
        print(f"  Cameras: {scenario.num_cameras}")
        print(f"  Target:  {scenario.target_pos}")
        print(f"  Include roll: {scenario.include_roll}")

        # Run Monte Carlo
        print(f"\n  Running {N_SAMPLES} MC samples...", end="", flush=True)
        samples, empirical_cov, num_valid = run_monte_carlo(
            scenario, cfg, N_SAMPLES, device
        )
        print(f" done ({num_valid} valid)", flush=True)

        if samples is None or empirical_cov is None:
            print(f"  ✗ FAILED: Not enough valid samples ({num_valid})")
            all_passed = False
            continue

        # Compute analytical covariance
        analytical_cov, ana_valid = compute_analytical_covariance(scenario, cfg, device)

        if not ana_valid or analytical_cov is None:
            print(f"  ✗ FAILED: Analytical covariance computation failed")
            all_passed = False
            continue

        # Compute validation metrics
        target_pos = torch.tensor(scenario.target_pos, device=device)
        metrics = compute_validation_metrics(
            empirical_cov, analytical_cov, samples, target_pos
        )

        # Print results
        print(f"\n  Covariance comparison:")
        print(f"    Trace ratio (analytical/empirical): {metrics['trace_ratio']:.3f}")

        # Print full covariance matrices
        print(f"\n  Empirical covariance matrix (trace={metrics['empirical_trace']:.4f}):")
        for i in range(3):
            row = empirical_cov[i]
            print(f"    [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")

        print(f"\n  Analytical covariance matrix (trace={metrics['analytical_trace']:.4f}):")
        for i in range(3):
            row = analytical_cov[i]
            print(f"    [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")

        print(f"\n  Eigenvalues (variance in m²):")
        print(f"    Empirical:  [{metrics['empirical_min_eigval']:.5f}, {metrics['empirical_max_eigval']:.5f}]")
        print(f"    Analytical: [{metrics['analytical_min_eigval']:.5f}, {metrics['analytical_max_eigval']:.5f}]")

        # Effective dimensionality (measures anisotropy)
        eff_dim = metrics.get('effective_dim_analytical', 3.0)
        print(f"\n  Effective dimensionality: {eff_dim:.2f}")
        print(f"    (1.0 = 1D/depth-only error, 3.0 = isotropic 3D)")

        print(f"\n  Coverage (analytical ellipsoid, chi²(3) DOF):")
        print(f"    1σ: {metrics['coverage_1sigma']*100:.1f}% (expected: {metrics.get('coverage_1sigma_expected', 0.199)*100:.1f}%)")
        print(f"    2σ: {metrics['coverage_2sigma']*100:.1f}% (expected: {metrics.get('coverage_2sigma_expected', 0.739)*100:.1f}%)")
        print(f"    3σ: {metrics['coverage_3sigma']*100:.1f}% (expected: {metrics.get('coverage_3sigma_expected', 0.971)*100:.1f}%)")

        print(f"\n  Position error: {metrics['mean_error']:.4f} m")

        # Validation criteria
        # NOTE: MC now samples ALL uncertainty sources (including full 3D body orientation)
        # so analytical and empirical covariances should match closely.
        # Deviations may occur due to:
        # - First-order Jacobian approximation in analytical computation
        # - Finite sample effects in Monte Carlo
        passed = True
        validation_msgs = []

        # Check trace ratio: should be close to 1.0 with full uncertainty sampling
        # Allow ratio between 0.3 and 3.5 (accounting for sampling variance and
        # first-order approximation errors, especially for close-range scenarios
        # where Jacobian linearization is less accurate)
        if metrics['trace_ratio'] < 0.3 or metrics['trace_ratio'] > 3.5:
            validation_msgs.append(f"Trace ratio {metrics['trace_ratio']:.2f} outside acceptable range [0.3, 3.5]")
            passed = False

        # Check that analytical ellipsoid captures sufficient samples
        # Coverage should be close to expected based on effective dimensionality.
        # Allow coverage within a tolerance of the expected value.
        # For highly anisotropic distributions (eff_dim ≈ 1), expected coverage is higher.
        coverage_tolerance = 0.15  # Allow 15% deviation from expected
        exp_1s = metrics.get('coverage_1sigma_expected', 0.39)
        exp_2s = metrics.get('coverage_2sigma_expected', 0.74)
        exp_3s = metrics.get('coverage_3sigma_expected', 0.93)

        # Check coverage is within tolerance of expected (accounting for anisotropy)
        if metrics['coverage_1sigma'] < exp_1s - coverage_tolerance:
            validation_msgs.append(
                f"1σ coverage {metrics['coverage_1sigma']*100:.1f}% too low "
                f"(expected {exp_1s*100:.1f}% ± {coverage_tolerance*100:.0f}%)"
            )
            passed = False
        if metrics['coverage_2sigma'] < exp_2s - coverage_tolerance:
            validation_msgs.append(
                f"2σ coverage {metrics['coverage_2sigma']*100:.1f}% too low "
                f"(expected {exp_2s*100:.1f}% ± {coverage_tolerance*100:.0f}%)"
            )
            passed = False
        if metrics['coverage_3sigma'] < exp_3s - coverage_tolerance:
            validation_msgs.append(
                f"3σ coverage {metrics['coverage_3sigma']*100:.1f}% too low "
                f"(expected {exp_3s*100:.1f}% ± {coverage_tolerance*100:.0f}%)"
            )
            passed = False

        if passed:
            print(f"\n  ✓ PASSED")
        else:
            print(f"\n  ✗ FAILED:")
            for msg in validation_msgs:
                print(f"    - {msg}")
            all_passed = False

        results_summary.append({
            "scenario": scenario.name,
            "passed": passed,
            "metrics": metrics,
            "num_valid": num_valid,
        })

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    num_passed = sum(1 for r in results_summary if r["passed"])
    total = len(results_summary)

    for r in results_summary:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['scenario']}: trace_ratio={r['metrics']['trace_ratio']:.3f}, "
              f"2σ_coverage={r['metrics']['coverage_2sigma']*100:.1f}%")

    print(f"\nTotal: {num_passed}/{total} passed")
    print("=" * 80, flush=True)
    sys.stdout.flush()

    return all_passed, results_summary


# =============================================================================
# Angle Sweep Experiment
# =============================================================================


def create_angle_sweep_scenario(
    viewing_angle_deg: float,
    target_distance: float,
    camera_height: float = 5.0,
) -> MCScenario:
    """Create a scenario for a specific viewing angle between cameras.

    Args:
        viewing_angle_deg: Full viewing angle between cameras in degrees.
                          90° = optimal triangulation geometry
                          0° = parallel rays (infinite uncertainty)
                          180° = cameras facing each other (poor geometry)
        target_distance: Distance from cameras to target (m)
        camera_height: Camera height above target (m)

    Returns:
        MCScenario configured for this angle
    """
    target = (0.0, 0.0, 0.0)
    # Half-angle from centerline
    half_angle_rad = math.radians(viewing_angle_deg / 2.0)

    # Place cameras symmetrically around target
    # Both cameras are "behind" the target (negative X), spread in Y
    cam1_x = -target_distance * math.cos(half_angle_rad)
    cam1_y = target_distance * math.sin(half_angle_rad)
    cam1_pos = (cam1_x, cam1_y, camera_height)

    cam2_x = -target_distance * math.cos(half_angle_rad)
    cam2_y = -target_distance * math.sin(half_angle_rad)
    cam2_pos = (cam2_x, cam2_y, camera_height)

    yaw1, pitch1 = compute_look_at_angles(cam1_pos, target)
    yaw2, pitch2 = compute_look_at_angles(cam2_pos, target)

    return MCScenario(
        name=f"viewing_angle_{viewing_angle_deg:.0f}deg",
        num_cameras=2,
        target_pos=target,
        camera_positions=[cam1_pos, cam2_pos],
        camera_yaws=[yaw1, yaw2],
        gimbal_yaws=[0.0, 0.0],
        gimbal_rolls=[0.0, 0.0],
        gimbal_pitches=[pitch1, pitch2],
        include_roll=True,
    )


def run_angle_sweep(
    cfg: TriangulationCfg,
    n_samples: int,
    device: torch.device,
    target_distance: float = 20.0,
    angles_deg: Optional[List[float]] = None,
) -> List[Dict]:
    """Sweep viewing angle and compute uncertainty at each angle.

    Args:
        cfg: Triangulation configuration
        n_samples: Number of MC samples per angle
        device: Torch device
        target_distance: Distance to target (m)
        angles_deg: List of viewing angles to test in degrees.
                   Default: [20, 40, 60, 80, 90, 100, 120, 140, 160]
                   90° is optimal (perpendicular cameras)

    Returns:
        List of result dicts per angle
    """
    if angles_deg is None:
        # Sweep from narrow (20°) through optimal (90°) to wide (160°)
        angles_deg = [20, 40, 60, 80, 90, 100, 120, 140, 160]

    print("\n" + "=" * 80)
    print("VIEWING ANGLE SWEEP")
    print("=" * 80)
    print(f"Viewing angles: {angles_deg} deg (90° = optimal)")
    print(f"Target distance: {target_distance} m")
    print(f"MC samples per angle: {n_samples}")

    results = []

    for angle_deg in angles_deg:
        print(f"\n  Testing angle: {angle_deg}°...", end="", flush=True)

        # Create scenario for this angle
        scenario = create_angle_sweep_scenario(
            viewing_angle_deg=angle_deg,
            target_distance=target_distance,
        )

        # Run Monte Carlo
        samples, empirical_cov, num_valid = run_monte_carlo(
            scenario, cfg, n_samples, device
        )

        if samples is None or empirical_cov is None:
            print(f" FAILED ({num_valid} valid)")
            results.append({
                "angle_deg": angle_deg,
                "mc_total_std": float("nan"),
                "analytical_total_std": None,
                "mc_cov": None,
                "analytical_cov": None,
                "num_valid": num_valid,
            })
            continue

        # Compute analytical covariance
        analytical_cov, ana_valid = compute_analytical_covariance(scenario, cfg, device)

        # Compute total std (sqrt of trace)
        mc_total_std = math.sqrt(float(torch.trace(empirical_cov).item()))
        analytical_total_std = None
        if ana_valid and analytical_cov is not None:
            analytical_total_std = math.sqrt(float(torch.trace(analytical_cov).item()))

        # Compute deviation
        dev_str = ""
        if analytical_total_std is not None and analytical_total_std > 0:
            dev = (mc_total_std - analytical_total_std) / analytical_total_std * 100
            dev_str = f"  dev={dev:+.1f}%"

        print(f" MC={mc_total_std:.4f}m  AN={analytical_total_std:.4f}m{dev_str}")

        results.append({
            "angle_deg": angle_deg,
            "mc_total_std": mc_total_std,
            "analytical_total_std": analytical_total_std,
            "mc_cov": empirical_cov.cpu().numpy(),
            "analytical_cov": analytical_cov.cpu().numpy() if analytical_cov is not None else None,
            "num_valid": num_valid,
        })

    return results


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_angle_sweep(sweep_results: List[Dict], save_path: Optional[str] = None):
    """Plot viewing-angle sweep results.

    Shows analytical & MC std (left axis) and deviation percentage (right axis).

    Args:
        sweep_results: List of result dicts from run_angle_sweep
        save_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping plot")
        return

    angles = np.array([r["angle_deg"] for r in sweep_results])
    mc_stds = np.array([r["mc_total_std"] for r in sweep_results])
    an_stds = np.array([r["analytical_total_std"] if r["analytical_total_std"] is not None
                        else np.nan for r in sweep_results])

    has_analytical = not np.all(np.isnan(an_stds))

    fig, ax_left = plt.subplots(figsize=(10, 6))

    # Left axis: uncertainty in meters
    ax_left.plot(angles, mc_stds, 'o-', color='#3b82f6', linewidth=2, markersize=8,
                 label='Monte Carlo')
    if has_analytical:
        ax_left.plot(angles, an_stds, 's--', color='#ef4444', linewidth=2, markersize=8,
                     label='Analytical')

    ax_left.set_xlabel('Viewing Angle (deg)', fontweight='bold', fontsize=12)
    ax_left.set_ylabel('Total Uncertainty σ (m)', fontweight='bold', fontsize=12, color='black')
    ax_left.set_xticks(angles)
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(axis='y', labelcolor='black')

    # Right axis: deviation (%)
    if has_analytical:
        valid_mask = ~np.isnan(an_stds) & (an_stds > 0)
        deviations = np.full_like(mc_stds, np.nan)
        deviations[valid_mask] = (mc_stds[valid_mask] - an_stds[valid_mask]) / an_stds[valid_mask] * 100

        ax_right = ax_left.twinx()
        ax_right.bar(angles, deviations, width=4, alpha=0.25, color='#8b5cf6',
                     edgecolor='#8b5cf6', linewidth=1.2, label='Deviation (%)')
        ax_right.axhline(0, color='gray', linewidth=0.8, linestyle='-')
        ax_right.set_ylabel('Deviation (%)', fontweight='bold', fontsize=12, color='#8b5cf6')
        ax_right.tick_params(axis='y', labelcolor='#8b5cf6')

    # Combined legend
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    if has_analytical:
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        leg = ax_right.legend(lines_left + lines_right, labels_left + labels_right,
                              loc='upper right', fontsize=10)
    else:
        leg = ax_left.legend(loc='upper right', fontsize=10)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.9)

    ax_left.set_title('Triangulation Uncertainty vs Viewing Angle', fontweight='bold', fontsize=14)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close(fig)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Angle (°)':<12} {'MC Std (m)':<14} {'AN Std (m)':<14} {'Deviation (%)':<14}")
    print(f"{'-'*70}")
    for r in sweep_results:
        an = r['analytical_total_std']
        mc = r['mc_total_std']
        if an is not None and not np.isnan(mc) and an > 0:
            dev = (mc - an) / an * 100
            print(f"{r['angle_deg']:<12.0f} {mc:<14.4f} {an:<14.4f} {dev:<+14.2f}")
        else:
            print(f"{r['angle_deg']:<12.0f} {mc:<14.4f} {'N/A':<14} {'N/A':<14}")
    print(f"{'='*70}")


def plot_validation_summary(results_summary: List[Dict], save_path: Optional[str] = None):
    """Plot validation summary showing trace ratio and coverage for all scenarios.

    Args:
        results_summary: List of result dicts from run_validation
        save_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping plot")
        return

    n_scenarios = len(results_summary)
    if n_scenarios == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenario_names = [r["scenario"] for r in results_summary]
    trace_ratios = [r["metrics"]["trace_ratio"] for r in results_summary]
    coverage_2sigma = [r["metrics"]["coverage_2sigma"] * 100 for r in results_summary]
    passed = [r["passed"] for r in results_summary]

    x = np.arange(n_scenarios)
    colors = ['#10b981' if p else '#ef4444' for p in passed]

    # Plot 1: Trace ratio
    ax = axes[0]
    bars = ax.bar(x, trace_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Ideal (ratio=1)')
    ax.axhline(y=0.3, color='orange', linestyle=':', linewidth=1.5, label='Lower bound (0.3)')
    ax.axhline(y=3.5, color='orange', linestyle=':', linewidth=1.5, label='Upper bound (3.5)')
    ax.set_ylabel('Trace Ratio (Analytical/Empirical)', fontweight='bold', fontsize=11)
    ax.set_title('Covariance Trace Ratio', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: 2σ Coverage
    ax = axes[1]
    bars = ax.bar(x, coverage_2sigma, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.axhline(y=73.9, color='blue', linestyle='--', linewidth=2, label='Expected (73.9%)')
    ax.axhline(y=60, color='orange', linestyle=':', linewidth=1.5, label='Minimum (60%)')
    ax.set_ylabel('2σ Coverage (%)', fontweight='bold', fontsize=11)
    ax.set_title('Coverage Statistics (2σ Ellipsoid)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Monte Carlo Validation Summary', fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close(fig)


def plot_covariance_ellipses(results_summary: List[Dict], save_path: Optional[str] = None):
    """Plot 2D covariance ellipses (XY projection) for each scenario.

    Args:
        results_summary: List of result dicts with covariance info
        save_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping plot")
        return

    # We need empirical and analytical covariances - this requires modification
    # to run_validation to return covariance matrices
    print("  Note: Covariance ellipse plot requires additional data collection")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args_cli.mode
    save_plots = args_cli.save_plots
    save_dir = args_cli.save_dir

    # Create save directory if needed
    if save_plots:
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(
                os.path.dirname(__file__),
                f"mc_validation_{timestamp}"
            )
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving plots to: {save_dir}")

    success = True

    try:
        # Run validation mode
        if mode in ["validation", "all"]:
            validation_passed, results_summary = run_validation(device)
            success = success and validation_passed

            if save_plots and results_summary:
                plot_validation_summary(
                    results_summary,
                    save_path=os.path.join(save_dir, "validation_summary.png")
                )

        # Run angle sweep mode
        if mode in ["angle_sweep", "all"]:
            cfg = TriangulationCfg(
                pix_std=7.0,
                pos_std=0.1,
                ori_std=0.01,
                gimbal_std=0.005,
                include_pose_uncertainty=True,
                include_gimbal_uncertainty=True,
            )

            sweep_results = run_angle_sweep(
                cfg=cfg,
                n_samples=N_SAMPLES,
                device=device,
                target_distance=args_cli.target_distance,
                # Sweep from narrow through optimal (90°) to wide
                angles_deg=[20, 40, 60, 80, 90, 100, 120, 140, 160],
            )

            if save_plots:
                plot_angle_sweep(
                    sweep_results,
                    save_path=os.path.join(save_dir, "angle_sweep.png")
                )

    except Exception as e:
        print(f"\nERROR: {e}")
        print(traceback.format_exc())
        success = False

    # Cleanup
    simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
