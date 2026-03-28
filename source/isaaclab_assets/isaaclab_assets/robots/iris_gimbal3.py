# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

IRIS_GIMBAL3_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_gimbal3.usda",
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
            "pitch_joint": 0.0,
            "roll_joint": 0.0,
            "yaw_joint": -math.pi / 2,  # Gimbal yaw starts at -90° so camera points along body +X
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
        },
        joint_vel={
            # Gimbal joints
            "pitch_joint": 0.0,
            "roll_joint": 0.0,
            "yaw_joint": 0.0,
            # Propeller joints
            "joint0": 200.0,
            "joint1": -200.0,
            "joint2": 200.0,
            "joint3": -200.0,
        },
    ),
    actuators={
        # Gimbal servo actuators — tuned for tau = c/k = 0.05s time constant.
        # With k=1000, c=50: critically/overdamped response settling in ~50ms.
        # Effort limit generous to avoid saturation during fast transients.
        # Velocity limit 6*pi ≈ 18.85 rad/s (~1080 deg/s) for fast slewing.
        "roll": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=6.0 * math.pi,
            stiffness=1e3,
            damping=5e1,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["roll_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=6.0 * math.pi,
            stiffness=1e3,
            damping=5e1,
        ),
        "yaw": ImplicitActuatorCfg(
            joint_names_expr=["pitch_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=6.0 * math.pi,
            stiffness=1e3,
            damping=5e1,
        ),
        # Propeller joints - no physics actuators (visual spinning only)
        # Joint state is written directly from environment for visual effect
        # Thrust/torque applied directly to body via external forces
        "prop0": ImplicitActuatorCfg(
            joint_names_expr=["joint0"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "prop1": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "prop2": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "prop3": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Iris quadcopter."""
