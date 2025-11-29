# file: iris_ma3_env_cfg.py

from __future__ import annotations

import gymnasium as gym
import torch
import math
import numpy as np
import copy
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import IRIS_GIMBAL2_CFG, IRIS_BODY_CFG
from isaaclab.markers import CUBOID_MARKER_CFG, FRAME_MARKER_CFG

from isaaclab_tasks.direct.iris_ma3 import bbox_raycaster

@configclass
class IrisMAEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 2.0 # NOTE: 20 -> 2 for quick testing
    decimation = 10
    
    # Define agents
    possible_agents = ["drone_0", "drone_1"]
    
    # Define spaces for each agent
    action_spaces = {
        "drone_0": 7,  # [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        "drone_1": 7,
    }
    observation_spaces = {
        "drone_0": 47,
        "drone_1": 47,
    }
    state_space = -1  # Concatenate all observations
    
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

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

    # Camera configuration template (dynamically applied to each agent)
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/{robot_name}/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={
            "class:target": (255, 36, 66, 255),
            "class:robot": (255, 36, 255, 255),
        },
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.7071068, 0, 0, -0.7071068), 
            convention="world"
        ),
    )

    # Target configuration
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_body.usda",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                rigid_body_enabled=True,
            ),
            copy_from_source=False,
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 3.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Viewer configuration
    viewer: ViewerCfg = ViewerCfg(
        eye=(15.0, 15.0, 15.0),
        lookat=(0.0, 0.0, 0.0),
        # origin_type="asset_body",
        # asset_name="Robot_0",
        # body_name="body",
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=25.0, replicate_physics=True)
    
    # Robot configuration template (dynamically applied to each agent)
    robot: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/{robot_name}")
    

    bbox_raycaster: bbox_raycaster.BBoxRayCasterCfg = bbox_raycaster.BBoxRayCasterCfg(
        target_prim_paths=[],  # Empty because we provide target poses directly via update()
        mesh_prim_paths=["/World/ground"],
        num_cameras_per_env=2,
        num_cameras_per_agent=1,
        load_agent_meshes=False,
        agent_mesh_simplification=1.1,
        use_collision_proxy=False,
        min_bbox_size=(0.01, 0.01),
        max_bbox_size=(0.90, 0.90),
        partial_detection_allowed=False,
        min_bbox_area_pixels=400.0,
        enable_occlusion_check=False,
        occlusion_ray_pattern="9point",
        occlusion_visibility_threshold=0.5,
        occlusion_ray_tolerance=1.1,
        max_distance=100.0,
        debug_vis=False,  # Disable debug vis in headless mode
        debug_memory=False,
    )

    # Control parameters
    thrust_to_weight = 4.0
    moment_scale = 10.0
    yaw_moment_scale = 1.0
    
    # Motion limits
    max_target_speed = 10.0
    max_target_ang_speed = math.radians(90.0)
    max_lin_vel = 20.0
    max_yaw_rate = math.radians(180.0)
    max_lin_acc = 3.0
    max_ang_acc = 15.0
    max_gimbal_yaw_angle = [math.radians(-200.0), math.radians(200.0)]
    max_gimbal_pitch_angle = [math.radians(-45.0), math.radians(10.0)]
    max_gimbal_roll_angle = [math.radians(-45.0), math.radians(45.0)]
    max_gimbal_angle_rate = math.radians(360.0)
    max_zoom_rate = 2.0
    max_zoom_level = 6.0
    max_no_detection_sec = episode_length_s / 2.0

    # Target movement parameters
    target_acceleration_scale = 2.0
    target_velocity_damping = 0.95
    target_direction_change_prob = 0.01
    target_max_acceleration = 3.0
    
    # Reward scales
    lin_vel_penalty_scale = -0.1
    ang_vel_penalty_scale = lin_vel_penalty_scale * 0.2
    action_sum_penalty_scale = -1.0
    action_weight = [1, 1, 1, 1, 1, 1, 1]
    action_delta_weight = [1, 1, 1, 1, 1, 1, 1]
    action_delta_penalty_scale = -0.05
    zoom_reward_scale = -0.5
    
    # Single-agent tracking rewards
    bbox_center_reward_scale = 60
    bbox_size_reward_scale = 60
    bbox_size_preferred = 0.10**2
    bbox_reward_shape_width = 0.35
    
    # Multi-agent coordination rewards
    triangulation_reward_scale = 5
    collision_penalty_scale = -100
    collision_min_safe_distance = 20.0
    ttc_penalty_scale = -10
    ttc_horizon = 5.0

    # Triangulation parameters
    pix_std = 7.0
    pos_std = 0.1
    ori_std = 0.01
    gimbal_std = 0.01
    intrinsics_std = 10.0

    # Curriculum
    curriculum_all_end_step: int = 200000
    curriculum_tracking_start_step: int = 10000 # Phase 1. Single-agent tracking
    curriculum_tracking_end_step: int = 80000
    curriculum_delay_start_step: int = 60000 # Phase 2. Delay and noise system
    curriculum_delay_end_step: int = 150000
    curriculum_coordination_start_step: int = 120000 # Phase 3. Triangulation
    curriculum_coordination_end_step: int = 200000
    curriculum_safety_start_step: int = 180000 # Phase 2&3. Min safe distance, TTC
    curriculum_safety_end_step: int = 230000
    curriculum_moving_target_start_step: int = 10000 # Phase 1&2. Target speed, acceleration
    curriculum_moving_target_end_step: int = 80000
    curriculum_dynamics_start_step: int = 40000 # Phase 0. Dynamics randomization
    curriculum_dynamics_end_step: int = 120000
    
    # Delay and Communication System Parameters
    enable_delay_system: bool = True
    enable_noise_in_observations: bool = True

    # Dynamics lag
    dynamics_time_constant: float = 0.1
    motion_time_constant: float = 0.1
    gimbal_time_constant: float = 0.01

    # Detection
    detection_fps: float = 20.0
    detection_mean_latency: float = 0.1
    detection_std_latency: float = 0.01
    detection_latency_bound: list = [0.05, 0.5]  # Minimum and maximum latency
    detection_failure_rate: float = 0.05

    # Communication
    comm_mean_latency: float = 0.1
    comm_std_latency: float = 0.01
    comm_latency_bound: list = [0.05, 0.5]  # Minimum and maximum latency
    comm_dropout_rate: float = 0.05
    # delay_buffer_size: int = 100
    max_time_since_comm: float = 10.0  # Default value when no comm received (seconds)
    max_time_since_detection: float = 10.0  # Normalize time since detection with this value

    # Reward decay
    detection_decay_time_constant: float = 1.0  # Time constant for exponential decay of detection confidence

    # # BBox detection parameters
    # bbox_fps: float = 30.0  # FPS throttle for bbox detections
    # bbox_latency_steps: int = 2  # Latency steps for bbox
    # bbox_dropout: float = 0.05  # Dropout probability for bbox

    play_sim_at_step: int = 180000
