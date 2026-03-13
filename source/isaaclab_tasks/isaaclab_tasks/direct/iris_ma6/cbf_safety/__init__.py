# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CBF Safety Filter Module for iris_ma6.

This module provides a two-layer CBF (Control Barrier Function) safety architecture:

**Training-time (CPA Reward Shaper)**:
- Velocity-aware CPA barrier using ground-truth state
- Soft penalty for reward shaping
- Dense gradient information about collision geometry

**Deployment-time (Robust Deployment Filter)**:
- Simple distance-based CBF with inflated margin
- Hard constraint using delayed observations
- Closed-form halfspace projection (no QP solver)

Usage:
    from isaaclab_tasks.direct.iris_ma6.cbf_safety import CBFManager, CBFManagerCfg

    cbf_manager = CBFManager(CBFManagerCfg(), num_envs, num_agents, device)

    # Training: compute penalty for reward shaping
    penalty = cbf_manager.compute_training_penalty(gt_positions, cmd_velocities, dt)

    # Deployment: filter actions through CBF
    v_safe, info = cbf_manager.filter_actions(v_nom, delayed_positions)

    # Check collisions for episode termination
    collided = cbf_manager.check_collisions(gt_positions)

Ref: safety_spec.md
"""

from .cbf_cfg import (
    CBFDiagnosticsCfg,
    CBFManagerCfg,
    CPARewardShaperCfg,
    RobustDeploymentFilterCfg,
)
from .cbf_diagnostics import CBFDiagnostics
from .cbf_manager import CBFManager
from .cpa_reward_shaper import CPARewardShaper
from .deploy_filter import RobustDeploymentFilter

__all__ = [
    # Configuration classes
    "CPARewardShaperCfg",
    "RobustDeploymentFilterCfg",
    "CBFDiagnosticsCfg",
    "CBFManagerCfg",
    # Core classes
    "CPARewardShaper",
    "RobustDeploymentFilter",
    "CBFDiagnostics",
    "CBFManager",
]
