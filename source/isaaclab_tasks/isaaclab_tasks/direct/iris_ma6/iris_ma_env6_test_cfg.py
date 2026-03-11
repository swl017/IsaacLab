# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Iris MA6 Test environment."""

from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import IRIS_GIMBAL2_CFG

from .controller import DroneControllerCfg


def _create_robot_cfg() -> ArticulationCfg:
    """Create robot config with gravity and gyroscopic forces enabled."""
    cfg = copy.deepcopy(IRIS_GIMBAL2_CFG)
    cfg.prim_path = "/World/envs/env_.*/{robot_name}"
    # Override rigid body properties to enable gravity and gyroscopic forces
    # Type ignore: spawn is UsdFileCfg at runtime which has rigid_props
    cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(  # type: ignore[union-attr]
        disable_gravity=False,  # Enable gravity for realistic testing
        max_depenetration_velocity=10.0,
        enable_gyroscopic_forces=True,  # Enable gyroscopic forces
    )
    return cfg


# Pre-create robot config with overridden physics properties
IRIS_GIMBAL2_TEST_CFG = _create_robot_cfg()


@configclass
class IrisMA6TestEnvCfg(DirectMARLEnvCfg):
    """Configuration for the Iris MA6 Test environment.

    This is a simplified test environment to validate the DroneController
    integration with 3 agents.

    Observation space per agent: 13D (pos, vel, quat, gimbal_yaw, gimbal_pitch, zoom)
    Action space per agent: 7D (vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate)
    """

    # ==========================================================================
    # Environment Meta
    # ==========================================================================

    num_agents: int = 3
    """Number of agents."""

    episode_length_s: float = 10.0
    """Episode length in seconds."""

    decimation: int = 4
    """Physics steps per control step (25 Hz policy at 100 Hz sim)."""

    # These are populated dynamically in __post_init__ based on num_agents.
    possible_agents: list[str] = ["drone_0", "drone_1", "drone_2"]
    """List of agent identifiers (auto-populated from num_agents)."""

    action_spaces: dict = {"drone_0": 7, "drone_1": 7, "drone_2": 7}
    """Action space dimensions per agent (auto-populated from num_agents)."""

    observation_spaces: dict = {"drone_0": 13, "drone_1": 13, "drone_2": 13}
    """Observation space dimensions per agent (13D: pos, vel, quat, gimbal_yaw, gimbal_pitch, zoom)."""

    state_space: int = -1
    """State space dimension. -1 means concatenate all observations."""

    debug_vis: bool = True
    """Enable debug visualization."""

    # ==========================================================================
    # Simulation
    # ==========================================================================

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_rigid_contact_count=2**23,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ==========================================================================
    # Assets
    # ==========================================================================

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_body.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                rigid_body_enabled=True,
            ),
            copy_from_source=False,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 3.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    """Target rigid body configuration."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(30.0, 30.0, 30.0),
        lookat=(0.0, 0.0, 0.0),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=50.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_GIMBAL2_TEST_CFG
    """Robot articulation configuration template with gravity and gyroscopic forces enabled."""

    # ==========================================================================
    # Controller Config
    # ==========================================================================

    drone_controller: DroneControllerCfg = DroneControllerCfg()
    """Drone controller configuration."""

    # ==========================================================================
    # Motion Limits
    # ==========================================================================

    max_lin_vel: float = 10.0
    """Maximum linear velocity (m/s)."""

    max_yaw_rate: float = math.radians(90.0)
    """Maximum yaw rate (rad/s)."""

    max_gimbal_rate: float = math.radians(360.0)
    """Maximum gimbal rate (rad/s)."""

    max_zoom_rate: float = 2.0
    """Maximum zoom rate (zoom levels per second)."""

    def __post_init__(self):
        """Populate agent-specific fields from num_agents."""
        if self.num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {self.num_agents}")

        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        self.action_spaces = {a: 7 for a in self.possible_agents}
        # Simplified observation: pos(3) + vel(3) + quat(4) + gimbal_yaw(1) + gimbal_pitch(1) + zoom(1) = 13D
        self.observation_spaces = {a: 13 for a in self.possible_agents}
