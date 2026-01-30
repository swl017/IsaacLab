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

IRIS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris.usd",
        # usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_gimbal.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,  # Disabled - gyroscopic coupling may interfere with control
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
            ".*": 0.0,
        },
        joint_vel={
            # Set propeller joints to zero - we control via external forces, not joint motors
            # The 200 rad/s initial velocities were causing gyroscopic coupling issues
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
        },
    ),
    actuators={
        # REMOVED: Dummy actuators were overriding external force/torque commands
        # The actuators call set_dof_velocity_targets() which overwrites our control
        # For external wrench-based control, we don't need actuators on the joints
    },
)
"""Configuration for the Iris quadcopter."""
