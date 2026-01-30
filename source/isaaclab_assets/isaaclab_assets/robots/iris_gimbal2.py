# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

IRIS_GIMBAL2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_gimbal2.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,  # Disabled - gyroscopic coupling may interfere with base control
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,  # Increased from 0 to allow torques to affect velocities
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,  # All joints start at 0
        },
        joint_vel={
            # Gimbal joints - keep at zero
            "pitch_joint": 0.0,
            "roll_joint": 0.0,
            "yaw_joint": 0.0,
            # Propeller joints - ZERO instead of 200 rad/s to avoid gyroscopic coupling
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
        },
    ),
    actuators={
        # KEEP gimbal actuators - these are real servo motors for gimbal control
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["pitch_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.14 * 3,  # ~9.42 rad/s max velocity
            stiffness=2e2,
            damping=1e0,
        ),
        "roll": ImplicitActuatorCfg(
            joint_names_expr=["roll_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.14 * 3,
            stiffness=2e2,
            damping=1e0,
        ),
        "yaw": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.14 * 3,
            stiffness=2e2,
            damping=1e0,
        ),
        # REMOVED propeller actuators - base body controlled via external forces
        # The propeller actuators would override our direct PhysX force application
    },
)
"""Configuration for the Iris quadcopter."""
