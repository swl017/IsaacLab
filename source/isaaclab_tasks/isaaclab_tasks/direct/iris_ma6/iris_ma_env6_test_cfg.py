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
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import IRIS_GIMBAL3_CFG

from .bbox_raycaster_v2 import BBoxRayCasterV2Cfg
from .controller import DroneControllerCfg
from .controller.tuning import TUNED_CONTROLLER_CFG


def _create_robot_cfg() -> ArticulationCfg:
    """Create robot config with gravity and gyroscopic forces enabled."""
    cfg = copy.deepcopy(IRIS_GIMBAL3_CFG)
    cfg.prim_path = "/World/envs/env_.*/{robot_name}"
    # Override rigid body properties to enable gravity and gyroscopic forces
    # Type ignore: spawn is UsdFileCfg at runtime which has rigid_props
    cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(  # type: ignore[union-attr]
        disable_gravity=False,  # Enable gravity for realistic testing
        max_depenetration_velocity=10.0,
        enable_gyroscopic_forces=False,  # Disabled - no gyroscopic coupling needed
    )
    return cfg


# Pre-create robot config with overridden physics properties
IRIS_GIMBAL2_TEST_CFG = _create_robot_cfg()


@configclass
class IrisMA6TestEnvCfg(DirectMARLEnvCfg):
    """Configuration for the Iris MA6 Test environment.

    This is a simplified test environment to validate the DroneController
    integration with 3 agents.

    Observation space per agent: 18D (pos, vel, quat, gimbal_yaw, gimbal_pitch, zoom, bbox, bbox_empty)
    Action space per agent: 7D (vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate)
    """

    # ==========================================================================
    # Environment Meta
    # ==========================================================================

    num_agents: int = 3
    """Number of agents."""

    episode_length_s: float = 30.0
    """Episode length in seconds."""

    decimation: int = 4
    """Physics steps per control step (25 Hz policy at 100 Hz sim)."""

    # These are populated dynamically in __post_init__ based on num_agents.
    possible_agents: list[str] = ["drone_0", "drone_1", "drone_2"]
    """List of agent identifiers (auto-populated from num_agents)."""

    action_spaces: dict = {"drone_0": 7, "drone_1": 7, "drone_2": 7}
    """Action space dimensions per agent (auto-populated from num_agents)."""

    observation_spaces: dict = {"drone_0": 18, "drone_1": 18, "drone_2": 18}
    """Observation space dimensions per agent (18D: pos, vel, quat, gimbal_yaw, gimbal_pitch, zoom, bbox, bbox_empty)."""

    state_space: int = -1
    """State space dimension. -1 means concatenate all observations."""

    debug_vis: bool = True
    """Enable debug visualization."""

    debug_frame_vis: bool = False
    """Enable visualization of coordinate frames for debugging."""

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
        eye=(18.0, 15.0, 16.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
        env_index=0,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=50.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_GIMBAL2_TEST_CFG
    """Robot articulation configuration template with gravity and gyroscopic forces enabled."""

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/{robot_name}/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            # ROS convention: forward=+Z, up=-Y
            # Maps camera forward → body +X (forward), camera up → body +Z (up)
            # Same quaternion value as frustum offset for consistency.
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    """Reference camera parameters used to build zoom-aware intrinsics for bbox_raycaster_v2."""

    bbox_raycaster_v2: BBoxRayCasterV2Cfg = BBoxRayCasterV2Cfg(
        target_prim_paths=["/World/envs/env_.*/target"],
        mesh_prim_paths=["/World/ground"],
        num_cameras_per_env=3,
        num_cameras_per_agent=1,
        load_agent_meshes=False,  # Disabled - causes shape bug with multi-env occlusion
        min_bbox_size=(0.01, 0.01),
        max_bbox_size=(0.95, 0.95),
        partial_detection_allowed=False,
        min_bbox_area_pixels=4.0,
        enable_occlusion_check=False,  # Disabled - shape bug with multi-env
        enable_self_occlusion=False,
        enable_inter_target_occlusion=False,
        occlusion_ray_pattern="center_only",
        occlusion_visibility_threshold=0.5,
        occlusion_ray_tolerance=1.1,
        max_distance=100.0,
        debug_vis=False,
        debug_memory=False,
    )
    """BBox raycaster V2 configuration used for camera/zoom/detection validation."""

    # ==========================================================================
    # Controller Config
    # ==========================================================================

    drone_controller: DroneControllerCfg = TUNED_CONTROLLER_CFG
    """Drone controller configuration. Loaded from auto-tuning results."""

    # ==========================================================================
    # Motion Limits
    # ==========================================================================

    max_lin_vel: float = 2.0
    """Maximum linear velocity (m/s). Kept low to avoid saturating attitude controller."""

    max_yaw_rate: float = math.radians(45.0)
    """Maximum yaw rate (rad/s)."""

    max_gimbal_rate: float = math.radians(180.0)
    """Maximum gimbal rate (rad/s)."""

    max_zoom_rate: float = 1.0
    """Maximum zoom rate (zoom levels per second)."""

    def __post_init__(self):
        """Populate agent-specific fields from num_agents."""
        if self.num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {self.num_agents}")

        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        self.action_spaces = {a: 7 for a in self.possible_agents}
        self.bbox_raycaster_v2.num_cameras_per_env = self.num_agents
        self.observation_spaces = {a: 18 for a in self.possible_agents}
