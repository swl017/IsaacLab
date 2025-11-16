# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import IRIS_GIMBAL2_CFG


class IrisMAEnvWindow(BaseEnvWindow):
    """Window manager for the Iris Multi-Agent environment."""

    def __init__(self, env, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class IrisMAEnvCfg(DirectMARLEnvCfg):
    # Environment configuration
    episode_length_s = 30.0
    decimation = 2
    debug_vis = True
    ui_window_class_type = IrisMAEnvWindow
    
    # Multi-agent setup
    possible_agents = ["drone_0", "drone_1"]
    action_spaces = {"drone_0": 7, "drone_1": 7}  # [vx, vy, vz, vyaw, gimbal_yaw, gimbal_pitch, zoom]
    observation_spaces = {"drone_0": 25, "drone_1": 25}
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=2 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    phys: PhysxCfg = PhysxCfg(
        gpu_max_rigid_patch_count=950272,
    )

    # Terrain
    terrain = TerrainImporterCfg(
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

    # Camera configurations for both drones
    camera_0_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone0/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={"class:target": (255, 36, 66, 255), "class:robot": (255, 36, 255, 255)},
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.5, -0.5, 0.5, -0.5), 
            convention="ros"
        ),
    )

    camera_1_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone1/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={"class:target": (255, 36, 66, 255), "class:robot": (255, 36, 255, 255)},
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.5, -0.5, 0.5, -0.5), 
            convention="ros"
        ),
    )

    # Target configuration
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                rigid_body_enabled=True,
            ),
            copy_from_source=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 3.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Frame transformers for each drone
    frame_transformer_0_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Drone0/body",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/Drone0/pitch_link"),
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/target"),
        ],
        debug_vis=True,
    )

    frame_transformer_1_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Drone1/body",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/Drone1/pitch_link"),
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/target"),
        ],
        debug_vis=True,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=20.0, replicate_physics=True)
    
    # Drone configurations
    drone_0_cfg: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/Drone0")
    drone_1_cfg: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/Drone1")
    
    # Control parameters
    thrust_to_weight = 4.0
    moment_scale = 10.0
    yaw_moment_scale = 1.0

    # Action weights for penalty
    action_weight = [1, 1, 5, 0.5, 0.03, 0.03, 0.01]
    action_delta_weight = [1, 1, 5, 0.5, 0.03, 0.03, 0.01]

    # Reward scales
    lin_vel_reward_scale = -0.1
    ang_vel_reward_scale = -0.02
    action_sum_reward_scale = -0.1
    action_delta_reward_scale = -0.01
    zoom_reward_scale = -0.5

    # Multi-agent specific rewards
    formation_reward_scale = 100.0  # Reward for maintaining optimal formation
    triangulation_reward_scale = 80.0  # Reward for good triangulation geometry
    bbox_center_reward_scale = 60.0
    bbox_size_reward_scale = 60.0
    coordination_reward_scale = 50.0  # Reward for coordination between drones

    # Physical limits
    max_target_speed = 15.0
    max_lin_vel = 15.0
    max_yaw_rate = 20.0
    max_lin_acc = 3.0
    max_ang_acc = 30.0
    
    # Target movement parameters
    target_acceleration_scale = 2.0
    target_velocity_damping = 0.95
    target_direction_change_prob = 0.01
    target_max_acceleration = 1.0

    # Formation parameters for triangulation
    optimal_baseline_distance = 8.0  # Optimal distance between drones for triangulation
    min_baseline_distance = 4.0     # Minimum safe distance between drones
    max_baseline_distance = 15.0    # Maximum useful distance for triangulation
    optimal_triangulation_angle = math.pi / 3  # 60 degrees for good triangulation