# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for CBF Safety Filter Module.

This module provides configuration dataclasses for:
- CPARewardShaper: Training-time CPA-based reward shaping
- RobustDeploymentFilter: Deployment-time hard CBF filter
- CBFDiagnostics: Training metrics tracking
- CBFManager: Unified facade coordinating all components
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class CPARewardShaperCfg:
    """Configuration for CPA-based reward shaper (training-time).

    The CPA (Closest Point of Approach) reward shaper computes a velocity-aware
    safety penalty using ground-truth positions and commanded velocities.
    This penalty provides dense gradient information about collision geometry
    during training.

    Ref: safety_spec.md Section 2.2, 2.3, 4.1
    """

    D_s: float = 2.0
    """Physical safety distance in meters. True minimum separation."""

    gamma: float = 2.0
    """CBF decay rate. Controls how fast barrier is allowed to shrink per step.
    Higher values mean stricter enforcement (less tolerance for approach)."""

    T: float = 1.0
    """CPA look-ahead horizon in seconds. Set approximately D_s / v_max.
    Predictions beyond this are clamped to avoid extrapolation artifacts."""

    lambda_cbf: float = 1.0
    """Penalty weight for reward composition. Tune so that at a mildly concerning
    configuration the penalty roughly equals one timestep of task reward."""

    epsilon: float = 1e-8
    """Numerical stability constant for velocity normalization when delta_v ≈ 0."""


@configclass
class RobustDeploymentFilterCfg:
    """Configuration for deployment-time CBF filter.

    The deployment filter uses simple distance-based CBF with inflated safety
    margins to handle communication delays and PX4 tracking lag. It provides
    formal safety guarantees under worst-case delay assumptions.

    Ref: safety_spec.md Section 2.4, 2.5, 4.2, 5.2
    """

    D_s: float = 2.0
    """Physical safety distance in meters (same as training)."""

    v_max: float = 15.0
    """Maximum expected agent velocity in m/s. Used to compute worst-case
    position drift during communication delay."""

    tau_delay_max: float = 0.2
    """Maximum communication delay in seconds. Worst-case observation staleness."""

    tau_px4: float = 0.3
    """PX4 velocity controller time constant in seconds. Additional tracking lag."""

    gamma_deploy: float = 1.0
    """CBF decay rate for deployment. More conservative (slower allowed approach)
    than training gamma to account for uncertainty."""

    num_iters: int = 2
    """Gauss-Seidel projection iterations. For 3 agents (2 constraints per agent),
    1-2 iterations are typically sufficient for convergence."""

    epsilon: float = 1e-8
    """Numerical stability constant for constraint normal normalization."""

    @property
    def D_deploy(self) -> float:
        """Inflated safety distance for deployment.

        D_deploy = D_s + v_max * (tau_delay_max + tau_px4)

        This accounts for worst-case position drift during total uncertainty
        (communication delay + tracking lag).
        """
        return self.D_s + self.v_max * (self.tau_delay_max + self.tau_px4)


@configclass
class CBFDiagnosticsCfg:
    """Configuration for CBF training diagnostics.

    Tracks safety metrics during training using ground-truth state.
    All metrics are computed from actual (not delayed) positions.
    """

    log_interval: int = 100
    """Steps between diagnostic logging. Higher values reduce overhead."""

    track_min_separation: bool = True
    """Track minimum pairwise separation across all agents."""

    track_collision_fraction: bool = True
    """Track fraction of environments with active collisions."""

    D_s: float = 2.0
    """Safety distance threshold for collision fraction metric."""


@configclass
class CBFManagerCfg:
    """Configuration for unified CBF safety manager.

    The CBFManager coordinates all safety components:
    - CPA reward shaper (training-time soft penalty)
    - Deployment filter (deployment-time hard constraint)
    - Diagnostics (training metrics)
    - Collision detection (episode termination)

    Ref: safety_spec.md Section 3, 6
    """

    cpa_cfg: CPARewardShaperCfg = CPARewardShaperCfg()
    """CPA reward shaper configuration."""

    deploy_cfg: RobustDeploymentFilterCfg = RobustDeploymentFilterCfg()
    """Deployment filter configuration."""

    diagnostics_cfg: CBFDiagnosticsCfg = CBFDiagnosticsCfg()
    """Diagnostics configuration."""

    enable_training_penalty: bool = True
    """Enable CPA penalty during training. When True, compute_training_penalty()
    returns the CPA barrier hinge loss for reward shaping."""

    enable_deployment_filter: bool = False
    """Enable hard CBF filter. When True, filter_actions() applies the
    distance-based CBF constraint. Should only be True at deployment."""

    enable_collision_termination: bool = True
    """Terminate episodes on actual collision (from GT). Provides strong signal
    that overall trajectory was unacceptable."""

    collision_distance: float = 2.0
    """Distance threshold for collision termination. Same as D_s by default."""
