# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the triangulation module."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TriangulationCfg:
    """Configuration for triangulation and uncertainty estimation.

    This configuration controls:
    - Uncertainty source standard deviations
    - Which uncertainty sources to include in covariance computation
    - Numerical stability parameters
    - Quality metric selection
    - Validation thresholds
    """

    # ==========================================================================
    # Uncertainty Source Standard Deviations
    # ==========================================================================

    pix_std: float = 7.0
    """Pixel detection noise standard deviation (pixels).

    Models the uncertainty in 2D bounding box center detection.
    Typical values: 3-10 pixels depending on detector quality.
    """

    pos_std: float = 0.1
    """Camera position uncertainty standard deviation (meters).

    Models uncertainty in robot/camera position from state estimation.
    Typical values: 0.05-0.2 m depending on localization quality.
    """

    ori_std: float = 0.001
    """Camera orientation uncertainty standard deviation (radians).

    Models uncertainty in robot/camera orientation from state estimation.
    Typical values: 0.001-0.01 rad depending on IMU/estimator quality.
    """

    gimbal_std: float = 0.001
    """Gimbal angle uncertainty standard deviation (radians).

    Models uncertainty in gimbal yaw/pitch angle encoders.
    Typical values: 0.001-0.005 rad depending on encoder resolution.
    """

    intrinsics_std: float = 10.0
    """Camera intrinsics uncertainty standard deviation.

    Models uncertainty in focal length and principal point.
    Only used if include_intrinsics_uncertainty is True.
    """

    # ==========================================================================
    # Uncertainty Source Inclusion Flags
    # ==========================================================================

    include_pose_uncertainty: bool = True
    """Include camera pose (position + orientation) uncertainty in covariance."""

    include_gimbal_uncertainty: bool = True
    """Include gimbal angle (yaw + pitch) uncertainty in covariance."""

    include_intrinsics_uncertainty: bool = False
    """Include camera intrinsics uncertainty in covariance.

    Usually disabled as intrinsics are well-calibrated and stable.
    """

    # ==========================================================================
    # Numerical Stability Parameters
    # ==========================================================================

    regularization_eps: float = 1e-6
    """Regularization epsilon for matrix inversion.

    Added to diagonal of normal matrix before inversion to improve stability.
    """

    condition_threshold: float = 1e6
    """Maximum acceptable condition number for normal matrix.

    If condition number exceeds this, triangulation is marked invalid.
    High condition number indicates poor geometry (e.g., collinear cameras).
    """

    min_z_threshold: float = 1e-6
    """Minimum Z value (depth) to avoid division by zero in projection.

    Points closer than this are clamped to prevent numerical instability.
    """

    # ==========================================================================
    # Validation Thresholds
    # ==========================================================================

    min_cameras_required: int = 2
    """Minimum number of valid cameras required for triangulation.

    With fewer cameras, triangulation is marked invalid.
    """

    max_trace_threshold: float = 1000.0
    """Maximum acceptable trace of covariance matrix.

    If trace exceeds this, triangulation is marked invalid.
    Indicates extremely high uncertainty.
    """

    # ==========================================================================
    # Quality Metric Selection
    # ==========================================================================

    quality_metric: Literal["trace", "det", "max_eig", "sqrt_trace"] = "trace"
    """Quality metric to compute from covariance matrix.

    Options:
    - "trace": Sum of variances (A-optimality) - default
    - "det": Determinant^(1/3) (D-optimality) - uncertainty volume
    - "max_eig": Maximum eigenvalue (E-optimality) - worst-case variance
    - "sqrt_trace": Square root of trace - standard deviation sum
    """

    # ==========================================================================
    # Temporal Filtering (Optional)
    # ==========================================================================

    use_temporal_filter: bool = False
    """Enable temporal covariance filtering (EMA smoothing).

    When enabled, covariance estimates are smoothed over time.
    """

    temporal_alpha: float = 0.1
    """EMA smoothing factor for temporal filtering.

    Higher values = faster adaptation to new measurements.
    Range: 0.0 (keep old) to 1.0 (use new only).
    """

    covariance_growth_rate: float = 1.05
    """Covariance inflation rate per timestep without observation.

    When no valid measurement is available, covariance is multiplied by this.
    Models increasing uncertainty over time without observations.
    """
