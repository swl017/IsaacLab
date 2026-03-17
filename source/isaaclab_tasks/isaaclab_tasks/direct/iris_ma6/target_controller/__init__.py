# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-based target controller module.

This module provides target movement using the same DroneController
architecture as agents, producing realistic physics-based motion
instead of direct velocity writes (as in iris_ma5).

Key Features:
- DroneController integration for force/torque-based physics
- Multiple velocity generation modes (linear, circular, approach, evade)
- Behavior FSM for attacker state management
- Curriculum-scaled parameters
- Geofencing and altitude constraints

Usage:
    from isaaclab_tasks.direct.iris_ma6.target_controller import (
        TargetController,
        TargetControllerCfg,
        FSMState,
        VelocityMode,
    )

    cfg = TargetControllerCfg()
    controller = TargetController(
        cfg=cfg,
        mass=1.5,
        gravity=9.81,
        num_envs=4096,
        num_targets=10,
        device=device,
    )

    # In environment step:
    F_body, tau_body = controller.step(...)

Reference:
    - target_movement_spec.md: Full specification document
    - iris_ma6_env_spec.md: Environment specification
"""

from .target_controller import TargetController
from .target_controller_cfg import TargetControllerCfg, BehaviorProfile, BEHAVIOR_PROFILES
from .behavior_fsm import BehaviorFSM, FSMState, VelocityMode

__all__ = [
    # Main classes
    "TargetController",
    "TargetControllerCfg",
    "BehaviorFSM",
    # Configuration
    "BehaviorProfile",
    "BEHAVIOR_PROFILES",
    # Enums
    "FSMState",
    "VelocityMode",
]
