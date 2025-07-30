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
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
            # "cgo3_vertical_arm_joint": 0.0,
            # "cgo3_horizontal_arm_joint": 0.0,
            # "cgo3_camera_joint": 0.0,
        },
        joint_vel={
            "pitch_joint": 0.0,
            "roll_joint": 0.0,
            "yaw_joint": 0.0,
            "joint0": 200.0,
            "joint1": -200.0,
            "joint2": 200.0,
            "joint3": -200.0,
        },
    ),
    actuators={
        # "dummy": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     stiffness=0.0,
        #     damping=0.0,
        # ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["pitch_joint"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "roll": ImplicitActuatorCfg(
            joint_names_expr=["roll_joint"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "yaw": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)
"""Configuration for the Iris quadcopter."""
