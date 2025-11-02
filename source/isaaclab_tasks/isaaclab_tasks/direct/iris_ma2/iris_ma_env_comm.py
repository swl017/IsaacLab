# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math
import numpy as np
import copy
from typing import Dict
from dataclasses import dataclass

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
from isaaclab.utils.math import (
    subtract_frame_transforms, 
    euler_xyz_from_quat, 
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    quat_inv,
    quat_rotate,
    yaw_quat
)

# Pre-defined configs
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import IRIS_GIMBAL2_CFG, IRIS_BODY_CFG
from isaaclab.markers import CUBOID_MARKER_CFG, FRAME_MARKER_CFG

from .point_mass import PointMass
from .gimbal_stabilizer import GimbalStabilizer
from .bbox_generator import BBoxGenerator
from .camera_frustrum import CameraFrustrum, create_camera_cfg_tensor, project_2d_to_3d
from .triang_cov_reward_torch import (
    triangulation_covariance_multi_camera, 
    triangulation_covariance_simple, 
    midpoint_method_batched,
    get_ray_dir_from_bbox
)

import carb

DEBUG_DRAW = True
if DEBUG_DRAW:
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    except ImportError:
        try:
            from omni.isaac.debug_draw import _debug_draw as omni_debug_draw
        except ImportError:
            print("Warning: Debug draw module not available. Disabling debug visualization.")
            DEBUG_DRAW = False
            omni_debug_draw = None

from . import bbox_raycaster

# Import delay and communication system
from .delay_comm_system_optimized import DelayedObservationManager, FirstOrderLagBatched


@dataclass
class AgentStates:
    """Container for agent states used in computations.
    
    This class manages delayed and delayed+noisy states for all agents,
    avoiding duplicate computations and making the code more maintainable.
    """
    
    def __init__(self, num_envs: int, num_agents: int, device: str):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        
        # Delayed states (without noise) - used for rewards
        self.delayed_robot_pos = torch.zeros(num_envs, num_agents, 3, device=device)
        self.delayed_robot_quat = torch.zeros(num_envs, num_agents, 4, device=device)
        self.delayed_robot_lin_vel = torch.zeros(num_envs, num_agents, 3, device=device) # World frame
        self.delayed_robot_ang_vel = torch.zeros(num_envs, num_agents, 3, device=device) # Body frame
        self.delayed_robot_lin_acc = torch.zeros(num_envs, num_agents, 3, device=device) # Body frame
        self.delayed_gimbal_yaw = torch.zeros(num_envs, num_agents, device=device)
        self.delayed_gimbal_pitch = torch.zeros(num_envs, num_agents, device=device)
        self.delayed_zoom_level = torch.zeros(num_envs, num_agents, device=device)
        
        # Delayed + noisy states - used for observations
        self.delayed_noisy_robot_pos = torch.zeros(num_envs, num_agents, 3, device=device)
        self.delayed_noisy_robot_quat = torch.zeros(num_envs, num_agents, 4, device=device)
        self.delayed_noisy_robot_lin_vel = torch.zeros(num_envs, num_agents, 3, device=device)
        self.delayed_noisy_robot_ang_vel = torch.zeros(num_envs, num_agents, 3, device=device)
        self.delayed_noisy_robot_lin_acc = torch.zeros(num_envs, num_agents, 3, device=device)
        self.delayed_noisy_gimbal_yaw = torch.zeros(num_envs, num_agents, device=device)
        self.delayed_noisy_gimbal_pitch = torch.zeros(num_envs, num_agents, device=device)
        
        # Camera intrinsics based on zoom (delayed and delayed+noisy)
        self.camera_intrinsics_delayed = torch.zeros(num_envs, num_agents, 3, 3, device=device)
        self.camera_intrinsics_delayed_noisy = torch.zeros(num_envs, num_agents, 3, 3, device=device)
        
        # Ray directions
        self.ray_dir_delayed = torch.zeros(num_envs, num_agents, 1, 3, device=device)  # (N, C, T, 3)
        self.ray_dir_delayed_noisy = torch.zeros(num_envs, num_agents, 1, 3, device=device)
        
        # Bounding boxes
        self.bboxes_delayed = torch.zeros(num_envs, num_agents, 1, 4, device=device)  # (N, C, T, 4)
        self.bboxes_delayed_noisy = torch.zeros(num_envs, num_agents, 1, 4, device=device)
        self.valid_mask_delayed = torch.zeros(num_envs, num_agents, 1, device=device, dtype=torch.bool)
        self.valid_mask_delayed_noisy = torch.zeros(num_envs, num_agents, 1, device=device, dtype=torch.bool)
        self.time_since_detection_delayed = torch.zeros(num_envs, num_agents, 1, device=device)
        self.time_since_detection_delayed_noisy = torch.zeros(num_envs, num_agents, 1, device=device)
        
        # Camera poses in world frame (delayed and delayed+noisy)
        self.camera_pos_delayed = torch.zeros(num_envs, num_agents, 3, device=device)
        self.camera_quat_delayed = torch.zeros(num_envs, num_agents, 4, device=device)
        self.camera_pos_delayed_noisy = torch.zeros(num_envs, num_agents, 3, device=device)
        self.camera_quat_delayed_noisy = torch.zeros(num_envs, num_agents, 4, device=device)
        
        # Triangulation covariance results
        self.Sigma_X_delayed = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, 1, 3, 3
        ) * (-1.0)  # (N, T, 3, 3)
        self.trace_cov_delayed = torch.ones(self.num_envs, 1, device=self.device) * (-1.0)  # (N, T)
        self.Sigma_X_delayed_noisy = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, 1, 3, 3
        ) * (-1.0)
        self.trace_cov_delayed_noisy = torch.ones(self.num_envs, 1, device=self.device) * (-1.0)
        
        # ===== RECEIVED STATES FROM OTHER AGENTS (WITH COMM LATENCY) =====
        # These store the most recent received data from other agents
        # Structure: Dict[agent_id] -> tensor of shape [N, C-1, ...]
        # where C-1 is the number of other agents
        self.received_positions = {}          # Dict[agent_id] -> [N, C-1, 3]
        self.received_orientations = {}       # Dict[agent_id] -> [N, C-1, 4]
        self.received_linear_velocities = {}  # Dict[agent_id] -> [N, C-1, 3]
        self.received_angular_velocities = {} # Dict[agent_id] -> [N, C-1, 3]
        self.received_linear_accelerations = {} # Dict[agent_id] -> [N, C-1, 3]
        self.received_gimbal_yawpitch = {}    # Dict[agent_id] -> [N, C-1, 2]
        self.received_camera_pos = {}         # Dict[agent_id] -> [N, C-1, 3]
        self.received_camera_quat = {}        # Dict[agent_id] -> [N, C-1, 4]
        self.received_bboxes = {}             # Dict[agent_id] -> [N, C-1, T, 4]
        self.received_bbox_valid = {}         # Dict[agent_id] -> [N, C-1, T]
        self.received_ray_dirs = {}           # Dict[agent_id] -> [N, C-1, T, 3]
        self.received_timestamps = {}         # Dict[agent_id] -> [N, C-1] generation timestamps
        self.received_valid = {}              # Dict[agent_id] -> [N, C-1] bool flags (True if data received)
        self.received_age = {}                # Dict[agent_id] -> [N, C-1] age of received data in seconds
        self.received_time_since_detection = {} # Dict[agent_id] -> [N, C-1, T]
        
    def compute_camera_poses(
        self,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        gimbal_yaw: torch.Tensor,
        gimbal_pitch: torch.Tensor,
        gimbal_roll: torch.Tensor,
        camera_offset_rot: torch.Tensor,
        camera_offset_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute camera poses from robot and gimbal states.
        
        Args:
            robot_pos: (N, C, 3)
            robot_quat: (N, C, 4)
            gimbal_yaw: (N, C)
            gimbal_pitch: (N, C)
            gimbal_roll: (N, C)
            camera_offset_rot: (N, 4)
            camera_offset_pos: (N, 3)
            
        Returns:
            camera_pos: (N, C, 3)
            camera_quat: (N, C, 4)
        """
        N, C = robot_pos.shape[:2]
        
        # Expand camera offset for broadcasting
        camera_offset_rot_exp = camera_offset_rot.unsqueeze(1).expand(N, C, 4)
        camera_offset_pos_exp = camera_offset_pos.unsqueeze(1).expand(N, C, 3)
        
        # Compute gimbal quaternion
        gimbal_quat = quat_mul(
            quat_mul(
                quat_from_euler_xyz(
                    torch.zeros_like(gimbal_yaw),
                    torch.zeros_like(gimbal_yaw),
                    gimbal_yaw
                ),
                quat_from_euler_xyz(
                    torch.zeros_like(gimbal_pitch),
                    gimbal_pitch,
                    torch.zeros_like(gimbal_pitch)
                )
            ),
            quat_from_euler_xyz(gimbal_roll, torch.zeros_like(gimbal_roll), torch.zeros_like(gimbal_roll))
        )
        
        # Compute camera quaternion and position
        camera_quat = quat_mul(robot_quat, quat_mul(gimbal_quat, camera_offset_rot_exp))
        camera_pos = robot_pos + quat_rotate(robot_quat, camera_offset_pos_exp)
        
        return camera_pos, camera_quat
    
    def compute_intrinsics_with_zoom(
        self,
        base_intrinsics: torch.Tensor,
        zoom_level: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute camera intrinsics with zoom and optional noise.
        
        Args:
            base_intrinsics: (N, C, 3, 3) base intrinsic matrices
            zoom_level: (N, C) zoom levels
            noise: Optional (N, C, 4) noise for [fx, fy, cx, cy]
            
        Returns:
            intrinsics: (N, C, 3, 3) intrinsic matrices with zoom applied
        """
        intrinsics = base_intrinsics.clone()
        
        if noise is not None:
            intrinsics[:, :, 0, 0] = (intrinsics[:, :, 0, 0] + noise[:, :, 0]) * zoom_level
            intrinsics[:, :, 1, 1] = (intrinsics[:, :, 1, 1] + noise[:, :, 1]) * zoom_level
            # Currently not supported in IsaacSim 4.5
            # intrinsics[:, :, 0, 2] = intrinsics[:, :, 0, 2] + noise[:, :, 2]
            # intrinsics[:, :, 1, 2] = intrinsics[:, :, 1, 2] + noise[:, :, 3]
        else:
            intrinsics[:, :, 0, 0] *= zoom_level
            intrinsics[:, :, 1, 1] *= zoom_level
        
        return intrinsics


class IrisMAEnvWindow(BaseEnvWindow):
    """Window manager for the Multi-Agent Iris environment."""

    def __init__(self, env: IrisMAEnv, window_name: str = "IsaacLab MARL"):
        """Initialize the window."""
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class ResearchLoggingCfg:
    log_interval_steps: int = 1
    sample_envs: int = 64
    tqi_opt_area: float = 0.05
    tqi_area_sigma: float = 0.02
    tqi_threshold: float = 0.20
    beta_min_deg: float = 30.0
    beta_max_deg: float = 150.0
    pixel_pitch_m: float = 3.45e-6
    det_std_px: float = 0.5
    enable_step_logging: bool = DEBUG_DRAW

@configclass
class IrisMAEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    
    # Define agents
    possible_agents = ["drone_0", "drone_1"]
    
    # Define spaces for each agent
    action_spaces = {
        "drone_0": 7,  # [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        "drone_1": 7,
    }
    observation_spaces = {
        "drone_0": 32,
        "drone_1": 32,
    }
    state_space = -1
    
    debug_vis = True
    ui_window_class_type = IrisMAEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/100,
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

    # Camera configuration template
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
            rot=(0.5, -0.5, 0.5, -0.5), 
            convention="ros"
        ),
    )

    # Target configuration
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
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 3.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Viewer configuration
    viewer: ViewerCfg = ViewerCfg(
        eye=(100.0, 100.0, 100.0),
        lookat=(0.0, 0.0, 0.0),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=25.0, replicate_physics=True)
    
    # Robot configuration template
    robot: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/{robot_name}")
    
    research: ResearchLoggingCfg = ResearchLoggingCfg()

    bbox_raycaster: bbox_raycaster.BBoxRayCasterCfg = bbox_raycaster.BBoxRayCasterCfg(
        target_prim_paths=[],
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
        debug_vis=True,
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
    max_zoom_rate = [-2.0, 2.0]
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
    ttc_penalty_scale = -10
    min_safe_distance = 20.0
    ttc_horizon = 5.0

    # Triangulation parameters
    pix_std = 7.0
    pos_std = 0.1
    ori_std = 0.01
    gimbal_std = 0.01
    intrinsic_std = 10.0

    # Curriculum
    curriculum_all_end_step: int = 200000
    curriculum_tracking_start_step: int = 2 * 20000 # Phase 1. Single-agent tracking
    curriculum_tracking_end_step: int = 2 * 80000
    curriculum_delay_start_step: int = 2 * 60000 # Phase 2. Delay and noise system
    curriculum_delay_end_step: int = 2 * 150000
    curriculum_coordination_start_step: int = 2 * 120000 # Phase 3. Triangulation, collision, TTC
    curriculum_coordination_end_step: int = 2 * 200000
    curriculum_moving_target_start_step: int = 2 * 20000 # Phase 1&2. Target speed, acceleration
    curriculum_moving_target_end_step: int = 2 * 80000
    curriculum_dynamics_start_step: int = 2 * 40000 # Phase 0. Dynamics randomization
    curriculum_dynamics_end_step: int = 2 * 120000
    
    # Delay and Communication System Parameters
    enable_delay_system: bool = True
    enable_noise_in_observations: bool = True
    dynamics_time_constant: float = 0.1
    detection_fps: float = 20.0
    motion_time_constant: float = 0.1
    gimbal_time_constant: float = 0.01
    detection_mean_latency: float = 0.1
    detection_std_latency: float = 0.01
    comm_mean_delay: float = 0.1
    comm_std_delay: float = 0.01
    comm_dropout_rate: float = 0.05
    delay_buffer_size: int = 100
    max_time_since_comm: float = 10.0  # Default value when no comm received (seconds)
    max_time_since_detection: float = 10.0  # Normalize time since detection with this value
    detection_decay_time_constant: float = 1.0  # Time constant for exponential decay of detection confidence

    play_sim_at_step: int = 200000

class IrisMAEnv(DirectMARLEnv):
    cfg: IrisMAEnvCfg

    def __init__(self, cfg: IrisMAEnvCfg, render_mode: str | None = None, **kwargs):
        # Dynamically generate agent-specific robot and camera configs
        self.agent_robot_cfgs = {}
        self.agent_camera_cfgs = {}
        for agent_id in cfg.possible_agents:
            robot_index = agent_id.split("_")[-1]
            robot_name = f"Robot_{robot_index}"

            robot_cfg = copy.deepcopy(cfg.robot)
            robot_cfg.prim_path = robot_cfg.prim_path.format(robot_name=robot_name)
            self.agent_robot_cfgs[agent_id] = robot_cfg
            
            camera_cfg = copy.deepcopy(cfg.camera)
            camera_cfg.prim_path = camera_cfg.prim_path.format(robot_name=robot_name)
            self.agent_camera_cfgs[agent_id] = camera_cfg
            
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize per-agent action and control tensors
        self._actions = {
            agent: torch.zeros(self.num_envs, gym.spaces.flatdim(self.action_spaces[agent]), device=self.device)
            for agent in self.cfg.possible_agents
        }
        self._last_actions = {
            agent: torch.zeros(self.num_envs, gym.spaces.flatdim(self.action_spaces[agent]), device=self.device)
            for agent in self.cfg.possible_agents
        }
        self._thrust = {
            agent: torch.zeros(self.num_envs, 1, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self._moment = {
            agent: torch.zeros(self.num_envs, 1, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self._cmd_vel = {
            agent: torch.zeros(self.num_envs, 1, 6, device=self.device)
            for agent in self.cfg.possible_agents
        }

        # Target scale
        num_targets_per_env = 1
        self.base_target_scale = self.cfg.target_cfg.spawn.scale[0]
        self.target_scale = torch.ones(self.num_envs, num_targets_per_env, 3, device=self.device) * self.base_target_scale

        # Target movement
        self.target_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0)
        self.last_target_vel = self.target_vel.clone()
        self.target_acceleration = torch.zeros(self.num_envs, 6, device=self.device)
        self.target_desired_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0) * self.cfg.max_target_speed
        self.target_desired_vel[:, 3:5] = 0.0
        self.target_vel_change_timer = torch.zeros(self.num_envs, device=self.device)
        self.target_motion_type = torch.randint(0, 2, (self.num_envs,), device=self.device)
        self.target_motion_radius = torch.zeros(self.num_envs, device=self.device).uniform_(10.0, 50.0)
        self.target_motion_yaw_vel = torch.norm(self.target_desired_vel[:, :3], dim=-1) / self.target_motion_radius
        
        # Camera tracking for each agent
        self.camera_pos_world = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.camera_quat_world = {
            agent: torch.zeros(self.num_envs, 4, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.bboxes = {
            agent: torch.zeros(self.num_envs, 4, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.bbox_valid_mask = {
            agent: torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            for agent in self.cfg.possible_agents
        }
        
        # Gimbal tracking
        self.gimbal_dof_targets = {
            agent: torch.zeros(self.num_envs, self._robots[agent].num_joints, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # Zoom tracking
        self.zoom_level = {
            agent: torch.ones(self.num_envs, device=self.device) * 1.0
            for agent in self.cfg.possible_agents
        }
        self.base_focal_length = {
            agent: torch.full(
                (self.num_envs,),
                float(self.agent_camera_cfgs[agent].spawn.focal_length),
                device=self.device,
                dtype=torch.float32,
            )
            for agent in self.cfg.possible_agents
        }
        
        # Camera configuration batch for each agent
        self.camera_offset_wrt_body = [0.0, 0.0, 0.0]
        self.camera_cfg_batch = {
            agent_id: create_camera_cfg_tensor(agent_cfg, self.num_envs, device=self.device)
            for agent_id, agent_cfg in self.agent_camera_cfgs.items()
        }
        self.camera_intrinsics = {
            agent: bbox_raycaster.utils.create_intrinsic_matrix_tensor(self.camera_cfg_batch[agent])
            for agent in self.cfg.possible_agents
        }
        self.base_intrinsic_matrices = self.camera_intrinsics.copy()
        camera_offset_rot_single = torch.tensor(cfg.camera.offset.rot, device=self.device)
        self.camera_offset_rot_batch = camera_offset_rot_single.unsqueeze(0).expand(self.num_envs, -1)
        camera_offset_pos_single = torch.tensor(self.camera_offset_wrt_body, device=self.device)
        self.camera_offset_pos_batch = camera_offset_pos_single.unsqueeze(0).expand(self.num_envs, -1)

        # Action weights
        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device).repeat(self.num_envs, 1)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device).repeat(self.num_envs, 1)
        
        # Get body and joint indices for each robot
        self._body_ids = {}
        self.gimbal_joint_idx = {}
        self._robot_mass = {}
        self._robot_dynamics = {}
        self._stabilizers = {}
        self._gimbal_stabilizers = {}
        
        for agent in self.cfg.possible_agents:
            robot = self._robots[agent]
            self._body_ids[agent] = robot.find_bodies("body")[0]
            self.gimbal_joint_idx[agent] = {
                "yaw": robot.find_joints("yaw_joint")[0][0],
                "roll": robot.find_joints("roll_joint")[0][0],
                "pitch": robot.find_joints("pitch_joint")[0][0],
            }
            self._robot_mass[agent] = robot.root_physx_view.get_masses()[0].sum()
            self._robot_dynamics[agent] = FirstOrderLagBatched(
                num_envs=self.num_envs,
                state_dim=6, # [vx, vy, vz, wx, wy, wz]
                time_constant=self.cfg.dynamics_time_constant,
                dt=self.step_dt,
                device=self.device,
            )
            robot_weight = (self._robot_mass[agent] * torch.tensor(self.sim.cfg.gravity, device=self.device).norm()).item()
            self._stabilizers[agent] = PointMass(0, robot_weight, self.num_envs, self.cfg.robot.spawn.rigid_props.disable_gravity, self.device)
            self._gimbal_stabilizers[agent] = GimbalStabilizer(
                yaw_limits=self.cfg.max_gimbal_yaw_angle,
                pitch_limits=self.cfg.max_gimbal_pitch_angle,
                device=self.device
            )

        # Prepare data in [N, C, ...] format
        self.robot_positions_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.robot_quats_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 4, device=self.device
        )
        self.robot_velocity_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.robot_angular_velocity_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.robot_linear_acceleration_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.gimbal_yaws_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), device=self.device
        )
        self.gimbal_pitches_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), device=self.device
        )
        self.camera_intrinsics_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, 3, device=self.device
        )
        for agent_idx, agent_id in enumerate(self.cfg.possible_agents):
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            self.robot_quats_stacked[:, agent_idx, :] = default_quat.unsqueeze(0).expand(self.num_envs, -1)
            self.camera_intrinsics_stacked[:, agent_idx, :, :] = self.camera_intrinsics[agent_id]

        # Consecutive no-detection counters
        self.no_detection_counts = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 1, device=self.device, dtype=torch.int)

        # Define uncertainty covariances
        self.X_w_est = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.Sigma_X_invalid = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, 1, 3, 3
        ) * (-1.0)
        self.Sigma_X = self.Sigma_X_invalid.clone()
        self.trace_cov_invalid = torch.ones(self.num_envs, 1, device=self.device) * (-1.0)
        self.trace_cov = self.trace_cov_invalid.clone()
        self.is_tri_cov_valid = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

        # Pixel noise [N, C, 2, 2]
        self.Sigma_pix = torch.eye(2, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.cfg.possible_agents), 2, 2
        ) * (self.cfg.pix_std ** 2)
        # Position uncertainty [N, C, 3, 3]
        self.Sigma_twb = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.cfg.possible_agents), 3, 3
        ) * (self.cfg.pos_std ** 2)
        # Orientation uncertainty [N, C, 3, 3]
        self.Sigma_phiwb = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.cfg.possible_agents), 3, 3
        ) * (self.cfg.ori_std ** 2)
        # Gimbal uncertainties [N, C, 1, 1]
        self.Sigma_alpha = torch.ones(self.num_envs, len(self.cfg.possible_agents), 1, 1, 
                                device=self.device) * (self.cfg.gimbal_std ** 2)
        self.Sigma_beta = torch.ones(self.num_envs, len(self.cfg.possible_agents), 1, 1, 
                                device=self.device) * (self.cfg.gimbal_std ** 2)
        # Intrinsic uncertainties [N, C, 4, 4]
        self.Sigma_K = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.cfg.possible_agents), 4, 4
        ) * (self.cfg.intrinsic_std ** 2)

        # Sampled noise values
        self.sampled_noise_generator = torch.Generator(device=self.device).manual_seed(0)
        self.sampled_noise_pix = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 1, 4, device=self.device
        )
        self.sampled_noise_twb = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.sampled_noise_phiwb = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.sampled_noise_alpha = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 1, device=self.device
        )
        self.sampled_noise_beta = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 1, device=self.device
        )
        self.sampled_noise_K = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 4, device=self.device
        )
        self.sampled_noise_lin_vel = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.sampled_noise_ang_vel = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.sampled_noise_lin_acc = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.sampled_noise_control_gains = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), len(self._stabilizers[self.cfg.possible_agents[0]].default_gains), device=self.device
        )

        # Collision and TTC
        self.cam_to_cam_distance = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), len(self.cfg.possible_agents), 
            device=self.device)
        self.cam_to_target_distance = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 1, 
            device=self.device)
        self.cam_to_cam_ttc = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), len(self.cfg.possible_agents), 
            device=self.device)
        self.cam_to_target_ttc = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 1, 
            device=self.device)
 
        self.ttc_loom_logsize_ema  = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)
        self.ttc_loom_logf_ema     = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)
        self.ttc_loom_log_g_prev   = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)
        self.ttc_loom_time_since_valid = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)

        # Curriculum progress
        self.progress_all = 0
        self.progress_delay = 0
        self.progress_tracking = 0
        self.progress_coord = 0
        self.progress_move = 0
        self.progress_dynamics = 0

        self._research = {
            "cvtt_sec": torch.zeros(self.num_envs, device=self.device),
            "tqi_sum": torch.zeros(self.num_envs, device=self.device),
            "steps": torch.zeros(self.num_envs, device=self.device),
            "last_step_logged": -1,
        }
        perm = torch.randperm(self.num_envs, device=self.device)
        self._research_sample_ids = perm[:min(self.cfg.research.sample_envs, self.num_envs)]

        # Episode tracking
        self._episode_sums = {
            agent: {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in [
                    "action_sum", 
                    "action_delta", 
                    "bbox_center", 
                    "bbox_size", 
                    "triangulation", 
                    "collision",
                    "ttc_penalty",
                    ]
            }
            for agent in self.cfg.possible_agents
        }
        
        # Visualization
        self.set_debug_vis(self.cfg.debug_vis)
        if DEBUG_DRAW and omni_debug_draw is not None:
            self.camera_frustrum = CameraFrustrum()
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/myMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(3/255, 252/255, 248/255),
                        ),
                    ),
                }
            )
            self.tri_cov_visualizer = VisualizationMarkers(marker_cfg)
        
        # Initialize states container
        self.states = AgentStates(self.num_envs, len(self.cfg.possible_agents), self.device)
        self.agent_name_to_idx = {
            name: idx for idx, name in enumerate(self.cfg.possible_agents)
        }

        # Initialize Delay and Communication System
        if self.cfg.enable_delay_system:
            self.delay_manager = DelayedObservationManager(
                num_envs=self.num_envs,
                num_agents=len(self.cfg.possible_agents),
                dt=self.cfg.sim.dt * self.cfg.decimation,
                device=self.device,
                motion_time_constant=self.cfg.motion_time_constant,
                gimbal_time_constant=self.cfg.gimbal_time_constant,
                detection_mean_latency=self.cfg.detection_mean_latency,
                detection_std_latency=self.cfg.detection_std_latency,
                detection_fps=self.cfg.detection_fps,
                comm_mean_delay=self.cfg.comm_mean_delay,
                comm_std_delay=self.cfg.comm_std_delay,
                comm_dropout_rate=self.cfg.comm_dropout_rate,
                max_buffer_size=self.cfg.delay_buffer_size
            )
            self.base_dynamics_time_constant = torch.tensor(self.cfg.dynamics_time_constant, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_detection_fps = torch.tensor(self.cfg.detection_fps, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_motion_time_constant = torch.tensor(self.cfg.motion_time_constant, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_gimbal_time_constant = torch.tensor(self.cfg.gimbal_time_constant, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_detection_mean_latency = torch.tensor(self.cfg.detection_mean_latency, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_detection_std_latency = torch.tensor(self.cfg.detection_std_latency, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_comm_mean_delay = torch.tensor(self.cfg.comm_mean_delay, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_comm_std_delay = torch.tensor(self.cfg.comm_std_delay, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_comm_dropout_rate = torch.tensor(self.cfg.comm_dropout_rate, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            self.base_delay_buffer_size = torch.tensor(self.cfg.delay_buffer_size, device=self.device).repeat(self.num_envs, len(self.cfg.possible_agents))
            
            self.detection_period = 1.0 / self.cfg.detection_fps
            self.time_since_last_detection = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)
            self.last_detection_time = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)

            print(f"[IrisMAEnv] Comm system enabled:")
            print(f"  - Detection FPS: {self.cfg.detection_fps:.1f} Hz ({self.detection_period*1000:.1f}ms period)")
            print(f"  - Motion lag: {self.cfg.motion_time_constant*1000:.1f}ms")
            print(f"  - Gimbal lag: {self.cfg.gimbal_time_constant*1000:.1f}ms")
            print(f"  - Detection latency: {self.cfg.detection_mean_latency*1000:.1f}±{self.cfg.detection_std_latency*1000:.1f}ms")
            print(f"  - Comm delay: {self.cfg.comm_mean_delay*1000:.1f}±{self.cfg.comm_std_delay*1000:.1f}ms")
            print(f"  - Comm dropout: {self.cfg.comm_dropout_rate*100:.1f}%")
        else:
            self.delay_manager: DelayedObservationManager = None
            self.time_since_last_detection = None
            self.last_detection_time = None
            print("[IrisMAEnv] Comm system disabled")

        self._env_origins = self._compute_env_origins_grid(self.num_envs, self.cfg.scene.env_spacing)

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """Compute the origins of the environments in a grid."""
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device), 
            torch.arange(num_cols, device=self.device), 
            indexing="ij"
        )
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins
    
    def _setup_scene(self):
        """Setup the scene with multiple robots and shared target."""
        # Create robots for each agent
        self._robots = {}
        for agent_id, robot_cfg in self.agent_robot_cfgs.items():
            robot_cfg.spawn.semantic_tags = [("class", "robot")]
            self._robots[agent_id] = Articulation(robot_cfg)
        
        # Create cameras for each agent
        self._cameras = {}
        if DEBUG_DRAW:
            for agent_id, camera_cfg in self.agent_camera_cfgs.items():
                self._cameras[agent_id] = TiledCamera(camera_cfg)

        # Create terrain
        if self.cfg.terrain is not None:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        else:
            self._terrain = None
        
        # Create shared target
        self.cfg.target_cfg.spawn.semantic_tags = [("class", "target")]
        self.target = RigidObject(self.cfg.target_cfg)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Register articulations, rigid objects, and sensors
        for agent_id, robot in self._robots.items():
            robot_index = agent_id.split("_")[-1]
            self.scene.articulations[f"Robot_{robot_index}"] = robot
        
        self.scene.rigid_objects["target"] = self.target

        if DEBUG_DRAW:
            for agent_id, camera in self._cameras.items():
                robot_index = agent_id.split("_")[-1]
                self.scene.sensors[f"camera_{robot_index}"] = camera
        
        # Configure and initialize bbox raycaster
        self.cfg.bbox_raycaster.debug_vis = DEBUG_DRAW
        self.cfg.bbox_raycaster.target_prim_paths = self.scene.env_prim_paths
        
        self.bbox_raycaster = bbox_raycaster.BBoxRayCaster(
            cfg=self.cfg.bbox_raycaster,
            num_envs=self.num_envs,
            num_targets_per_env=1,
            device=self.device,
            agent_ids=self.cfg.possible_agents,
        )

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]):
        """Pre-process actions for all agents before physics step."""
        # Update delay manager time (must be first!)
        if self.cfg.enable_delay_system and self.delay_manager is not None:
            self.delay_manager.update_time()
        
        for agent_id, agent_actions in actions.items():
            if torch.isnan(agent_actions).any():
                continue
                
            robot = self._robots[agent_id]
            stabilizer = self._stabilizers[agent_id]
            
            # Process actions (implementation continues with existing logic...)
            self._actions[agent_id] = agent_actions.clone()
            
            # Extract actions
            vel_target_world = agent_actions[:, 0:3]
            yaw_rate_target_body = agent_actions[:, 3:4]
            gimbal_yaw_rate = agent_actions[:, 4:5]
            gimbal_pitch_rate = agent_actions[:, 5:6]
            zoom_rate = agent_actions[:, 6:7]
            
            # Velocity commands
            self._cmd_vel[agent_id][:, 0, 0:3] = vel_target_world * self.cfg.max_lin_vel
            self._cmd_vel[agent_id][:, 0, 5:6] = yaw_rate_target_body * self.cfg.max_yaw_rate
            
            # Update gimbal targets
            gimbal_idx = self.gimbal_joint_idx[agent_id]

            gimbal_yaw_desired = self.gimbal_dof_targets[agent_id][:, gimbal_idx["yaw"]] + gimbal_yaw_rate.squeeze(-1) * self.step_dt * self.cfg.max_gimbal_angle_rate
            gimbal_yaw_desired = self._stabilizers[agent_id].wrap_to_pi(gimbal_yaw_desired)
            gimbal_yaw_desired = torch.clamp(
                gimbal_yaw_desired,
                min=self.cfg.max_gimbal_yaw_angle[0],
                max=self.cfg.max_gimbal_yaw_angle[1]
            )
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["yaw"]] = gimbal_yaw_desired.clone()

            gimbal_pitch_desired = self.gimbal_dof_targets[agent_id][:, gimbal_idx["pitch"]] + gimbal_pitch_rate.squeeze(-1) * self.step_dt * self.cfg.max_gimbal_angle_rate
            gimbal_pitch_desired = self._stabilizers[agent_id].wrap_to_pi(gimbal_pitch_desired)
            gimbal_pitch_desired = torch.clamp(
                gimbal_pitch_desired,
                min=self.cfg.max_gimbal_pitch_angle[0],
                max=self.cfg.max_gimbal_pitch_angle[1]
            )
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["pitch"]] = gimbal_pitch_desired.clone()
            stabilizing_roll = self._gimbal_stabilizers[agent_id].compute_stabilizing_roll(
                gimbal_yaw=gimbal_yaw_desired,
                gimbal_pitch=gimbal_pitch_desired,
                drone_quat_world=robot.data.root_state_w[:, 3:7],
            )
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["roll"]] = stabilizing_roll * 0.9

            # Update zoom level
            self.zoom_level[agent_id] = torch.clamp(
                self.zoom_level[agent_id] + zoom_rate.squeeze(-1) * self.step_dt,
                min=1.0,
                max=self.cfg.max_zoom_level
            )
            
            # Apply joint targets
            robot.set_joint_position_target(self.gimbal_dof_targets[agent_id])
            # robot.set_joint_velocity_target(gimbal_yaw_rate.squeeze()*self.cfg.max_gimbal_angle_rate, gimbal_idx["yaw"])
            # robot.set_joint_velocity_target(gimbal_pitch_rate.squeeze()*self.cfg.max_gimbal_angle_rate, gimbal_idx["pitch"])

            # Compute forces using stabilizer
            i = self.agent_name_to_idx[agent_id]
            gains = {k: v.clone() for k, v in stabilizer.default_gains.items()}
            if self.cfg.enable_delay_system and self.delay_manager is not None:
                for gain_idx, key in enumerate(stabilizer.default_gains.keys()):
                    gains[key] *= self.sampled_noise_control_gains[:, i, gain_idx].clone()
            self._thrust[agent_id][:,0,:], self._moment[agent_id][:,0,:] = stabilizer.compute_control(
                cmd_lin_vel_w=self._cmd_vel[agent_id][:, 0, :3],
                cmd_yaw_vel=self._cmd_vel[agent_id][:, 0, 5],
                curr_quat_w=self.states.delayed_robot_quat[:, i, :],
                curr_lin_vel_w=self.states.delayed_robot_lin_vel[:, i, :],
                curr_ang_vel_b=self.states.delayed_robot_ang_vel[:, i, :],
                curr_lin_acc_b=self.states.delayed_robot_lin_acc[:, i, :],
                dt=self.step_dt,
                gains=gains,
            )
            self._thrust[agent_id][:,0,:] -= robot.data.root_lin_vel_b * 0.05  # Dampen linear velocity

            # # Compute new velocities
            agent_idx = self.cfg.possible_agents.index(agent_id)
            new_vel_lin_w = self.states.delayed_robot_lin_vel[:, agent_idx, :] * (1 - self.step_dt) + quat_rotate(
                self.states.delayed_robot_quat[:, agent_idx, :], self._thrust[agent_id][:, 0, :]) * self.step_dt
            new_vel_lin_w[:, :2] = torch.clamp(new_vel_lin_w[:, :2], min=-self.cfg.max_lin_vel, max=self.cfg.max_lin_vel)
            
            new_vel_ang_w = self.states.delayed_robot_ang_vel[:, agent_idx, :] * (1 - 3 * self.step_dt) + quat_rotate(
                self.states.delayed_robot_quat[:, agent_idx, :], self._moment[agent_id][:, 0, :]) * self.step_dt
            new_vel_ang_w[:, 0] = torch.zeros_like(new_vel_ang_w[:, 0])
            new_vel_ang_w[:, 1] = torch.zeros_like(new_vel_ang_w[:, 1])
            new_vel_ang_w[:, 2] = torch.clamp(new_vel_ang_w[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

            lagged_vel = self._robot_dynamics[agent_id].update(
                measured_state=torch.cat((new_vel_lin_w, new_vel_ang_w), dim=1),
                current_time=self._robot_dynamics[agent_id].timestamp+self.step_dt
            )

            # Write velocities and gimbal targets
            robot.write_root_velocity_to_sim(
                lagged_vel, 
                env_ids=robot._ALL_INDICES
            )
        # Randomly assign environments to linear (0) or circular (1) motion
        target_motion_assignment = torch.randint(0, 2, (self.num_envs,), device=self.device)
        linear_env_ids = torch.nonzero(target_motion_assignment == 0, as_tuple=False).squeeze(-1)
        circular_env_ids = torch.nonzero(target_motion_assignment == 1, as_tuple=False).squeeze(-1)

        # Update target movement based on assignment
        if len(linear_env_ids) > 0:
            self._update_target_motion(mode="linear", env_ids=linear_env_ids)
        if len(circular_env_ids) > 0:
            self._update_target_motion(mode="circular", env_ids=circular_env_ids)
        # Bounce targets that are too low
        up_vel = self.target_vel.clone()
        up_vel[..., 2] = torch.where(self.target_vel[..., 2] > 0, self.target_vel[..., 2], -self.target_vel[..., 2])
        target_lower_bound = torch.ones_like(self.target.data.root_state_w, device=self.device) * 1.0
        self.target_vel[..., 2] = torch.where(
            self.target.data.root_state_w[:, 2] < target_lower_bound[:, 2], 
            up_vel[..., 2], 
            self.target_vel[..., 2]
        )
        # Phase 1. Single-agent tracking
        # Phase 1&2. Target speed, acceleration
        # Phase 2. Delay and noise system
        # Phase 3. Triangulation, collision, TTC
        current_step = self.cfg.play_sim_at_step if DEBUG_DRAW else self.common_step_counter
        self.progress_all = self._linear_progress(0, self.cfg.curriculum_all_end_step, current_step)
        self.progress_delay = self._linear_progress(self.cfg.curriculum_delay_start_step, self.cfg.curriculum_delay_end_step, current_step)
        self.progress_tracking = self._linear_progress(self.cfg.curriculum_tracking_start_step, self.cfg.curriculum_tracking_end_step, current_step)
        self.progress_coord = self._linear_progress(self.cfg.curriculum_coordination_start_step, self.cfg.curriculum_coordination_end_step, current_step)
        self.progress_move = self._linear_progress(self.cfg.curriculum_moving_target_start_step, self.cfg.curriculum_moving_target_end_step, current_step)
        self.progress_dynamics = self._linear_progress(self.cfg.curriculum_dynamics_start_step, self.cfg.curriculum_dynamics_end_step, current_step)

        # Override in debug mode
        # progress = 1 if DEBUG_DRAW else progress
        # Apply target velocity
        
        self.target.write_root_com_velocity_to_sim(self.target_vel * self.progress_move)
        
        if self.delay_manager is not None:
            motion_time_constants = self.base_motion_time_constant.uniform_(0.0, self.cfg.motion_time_constant * self.progress_delay)
            gimbal_time_constants = self.base_gimbal_time_constant.uniform_(0.0, self.cfg.gimbal_time_constant * self.progress_delay)
            detection_fps = self.base_detection_fps.uniform_(self.cfg.detection_fps * self.progress_delay + (1.0/self.step_dt) * (1-self.progress_delay), 1.0/self.step_dt)
            detection_mean_latencies = self.base_detection_mean_latency.uniform_(0.0, self.cfg.detection_mean_latency * self.progress_delay)
            detection_std_latencies = self.base_detection_std_latency.uniform_(0.0, self.cfg.detection_std_latency * self.progress_delay)
            comm_mean_delays = self.base_comm_mean_delay.uniform_(0.0, self.cfg.comm_mean_delay * self.progress_delay)
            comm_std_delays = self.base_comm_std_delay.uniform_(0.0, self.cfg.comm_std_delay * self.progress_delay)
            comm_dropout_rates = self.base_comm_dropout_rate.uniform_(0.0, self.cfg.comm_dropout_rate * self.progress_delay)

            self.delay_manager.update_motion_time_constants(motion_time_constants)
            self.delay_manager.update_gimbal_time_constants(gimbal_time_constants)
            self.delay_manager.update_detection_fps(detection_fps)
            self.delay_manager.update_detection_latency(detection_mean_latencies, detection_std_latencies)
            self.delay_manager.update_comm_parameters(comm_mean_delays, comm_std_delays, comm_dropout_rates)


    def _apply_action(self):
        """Apply actions to the environment."""
        pass

    def _linear_progress(self, start: int, end: int, current=None):
        if end <= start: 
            return 1.0
        if current is None:
            current = self.common_step_counter
        x = (current - start) / (end - start)
        return 0.0 if x < 0 else (1.0 if x > 1 else float(x))

    def _compute_intermediate_values(self):
        """Compute intermediate values shared by both rewards and observations.
        
        This method is called at the start of _get_rewards() to prepare:
        1. Ground truth states for reward computation
        2. Delayed states (without noise) for reward computation
        3. Delayed+noisy states for observation computation
        """
        # Update time tracking
        if self.cfg.enable_delay_system and self.delay_manager is not None:
            self.time_since_last_detection += self.step_dt
        
        # Generate noise samples (used later for delayed+noisy states)
        if self.cfg.enable_noise_in_observations:
            self.sampled_noise_pix.normal_(mean=0.0, std=self.cfg.pix_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_twb.normal_(mean=0.0, std=self.cfg.pos_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_phiwb.normal_(mean=0.0, std=self.cfg.ori_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_alpha.normal_(mean=0.0, std=self.cfg.gimbal_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_beta.normal_(mean=0.0, std=self.cfg.gimbal_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_lin_vel.normal_(mean=0.0, std=0.1 * self.cfg.pos_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_ang_vel.normal_(mean=0.0, std=self.cfg.ori_std * self.cfg.pos_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_lin_acc.normal_(mean=0.0, std=100e-6 * self.progress_delay, generator=self.sampled_noise_generator)
        
        # ===== 1. COLLECT GROUND TRUTH STATES =====
        # These are used for reward computation
        robot_positions_gt = torch.zeros_like(self.robot_positions_stacked)
        robot_quats_gt = torch.zeros_like(self.robot_quats_stacked)
        robot_lin_vels_gt = torch.zeros_like(self.robot_velocity_stacked)
        robot_ang_vels_gt = torch.zeros_like(self.robot_angular_velocity_stacked)
        robot_lin_accs_gt = torch.zeros_like(self.robot_linear_acceleration_stacked)
        gimbal_yaws_gt = torch.zeros_like(self.gimbal_yaws_stacked)
        gimbal_pitches_gt = torch.zeros_like(self.gimbal_pitches_stacked)
        gimbal_rolls_gt = torch.zeros_like(self.gimbal_yaws_stacked)
        zoom_levels_gt = torch.zeros_like(self.gimbal_yaws_stacked)
        
        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            gimbal_idx = self.gimbal_joint_idx[agent_id]
            
            robot_positions_gt[:, i, :] = robot.data.root_pos_w.clone()
            robot_quats_gt[:, i, :] = robot.data.root_state_w[:, 3:7].clone()
            robot_lin_vels_gt[:, i, :] = robot.data.root_lin_vel_w.clone()
            robot_ang_vels_gt[:, i, :] = robot.data.root_ang_vel_b.clone()
            robot_lin_accs_gt[:, i, :] = quat_apply(robot.data.root_state_w[:, 3:7], robot.data.body_lin_acc_w[:, self._body_ids[agent_id][0], :])
            gimbal_yaws_gt[:, i] = robot.data.joint_pos[:, gimbal_idx["yaw"]].clone()
            gimbal_pitches_gt[:, i] = robot.data.joint_pos[:, gimbal_idx["pitch"]].clone()
            gimbal_rolls_gt[:, i] = robot.data.joint_pos[:, gimbal_idx["roll"]].clone()
            zoom_levels_gt[:, i] = self.zoom_level[agent_id].clone()

            width = self.camera_cfg_batch[agent_id][:, 0]
            horizontal_aperture = self.camera_cfg_batch[agent_id][:, 3]
            current_focal_length = (self.base_focal_length[agent_id] + self.sampled_noise_K[:, i, 0] / width * horizontal_aperture) * self.zoom_level[agent_id]
            self.camera_cfg_batch[agent_id][:, 2] = current_focal_length
        
        # Compute GT camera poses and intrinsics
        camera_pos_gt, camera_quat_gt = self.states.compute_camera_poses(
            robot_pos=robot_positions_gt,
            robot_quat=robot_quats_gt,
            gimbal_yaw=gimbal_yaws_gt,
            gimbal_pitch=gimbal_pitches_gt,
            gimbal_roll=gimbal_rolls_gt,
            camera_offset_rot=self.camera_offset_rot_batch,
            camera_offset_pos=self.camera_offset_pos_batch,
        )
        
        # Stack base intrinsics (N, C, 3, 3)
        base_intrinsics_stacked = torch.stack([
            self.base_intrinsic_matrices[agent_id] for agent_id in self.cfg.possible_agents
        ], dim=1)
        
        camera_intrinsics_gt = self.states.compute_intrinsics_with_zoom(
            base_intrinsics=base_intrinsics_stacked,
            zoom_level=zoom_levels_gt,
        )

        # Update bbox raycaster with GT states
        target_pos = self.target.data.root_pos_w
        target_quat = self.target.data.root_quat_w
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(1)
        if target_quat.ndim == 2:
            target_quat = target_quat.unsqueeze(1)
        
        camera_poses_gt = {
            str(agent_id): (camera_pos_gt[:, i, :], camera_quat_gt[:, i, :])
            for i, agent_id in enumerate(self.cfg.possible_agents)
        }
        camera_intrinsics_gt_dict = {
            str(agent_id): camera_intrinsics_gt[:, i, :, :]
            for i, agent_id in enumerate(self.cfg.possible_agents)
        }
        agent_poses_gt = {
            str(agent_id): (robot_positions_gt[:, i, :], robot_quats_gt[:, i, :])
            for i, agent_id in enumerate(self.cfg.possible_agents)
        }
        image_shapes = {
            str(agent_id): (self.agent_camera_cfgs[agent_id].height, self.agent_camera_cfgs[agent_id].width)
            for agent_id in self.cfg.possible_agents
        }
        
        self.bbox_raycaster.update(
            camera_poses=camera_poses_gt,
            camera_intrinsics=camera_intrinsics_gt_dict,
            target_poses=(target_pos, target_quat),
            agent_poses=agent_poses_gt,
            image_shapes=image_shapes,
            target_scale=self.target_scale
        )
        
        # Update per-agent bbox storage
        for i, agent_id in enumerate(self.cfg.possible_agents):
            self.bboxes[agent_id] = self.bbox_raycaster.data.bboxes[:, i, 0, :]
            self.bbox_valid_mask[agent_id] = self.bbox_raycaster.data.valid_mask[:, i, 0:1]
            self.camera_pos_world[agent_id] = camera_pos_gt[:, i, :]
            self.camera_quat_world[agent_id] = camera_quat_gt[:, i, :]
        
        # ===== 2. PROCESS DELAYED STATES (WITHOUT NOISE) FOR REWARDS =====
        if self.cfg.enable_delay_system and self.delay_manager is not None:
            for i, agent_id in enumerate(self.cfg.possible_agents):
                robot = self._robots[agent_id]
                gimbal_idx = self.gimbal_joint_idx[agent_id]
                
                # Get true states WITHOUT noise
                robot_pos_true = robot_positions_gt[:, i, :].clone()
                robot_quat_true = robot_quats_gt[:, i, :].clone()
                robot_lin_vel_true = robot_lin_vels_gt[:, i, :].clone()
                robot_ang_vel_true = robot_ang_vels_gt[:, i, :].clone()
                gimbal_yaw_true = robot.data.joint_pos[:, gimbal_idx["yaw"]].unsqueeze(-1)
                gimbal_pitch_true = robot.data.joint_pos[:, gimbal_idx["pitch"]].unsqueeze(-1)
                
                # Apply first-order lag (this simulates delays without adding noise)
                robot_pos_delayed, robot_quat_delayed, robot_lin_vel_delayed, robot_ang_vel_delayed = self.delay_manager.update_ego_motion(
                    self.agent_name_to_idx[agent_id],
                    robot_pos_true,
                    robot_quat_true,
                    robot_lin_vel_true,
                    robot_ang_vel_true
                )

                gimbal_yaw_delayed, gimbal_pitch_delayed = self.delay_manager.update_ego_gimbal(
                    self.agent_name_to_idx[agent_id],
                    gimbal_yaw_true,
                    gimbal_pitch_true
                )
                
                # Store delayed states (without noise)
                self.states.delayed_robot_pos[:, i, :] = robot_pos_delayed.clone()
                self.states.delayed_robot_quat[:, i, :] = robot_quat_delayed.clone()
                self.states.delayed_robot_lin_vel[:, i, :] = robot_lin_vel_delayed.clone()
                self.states.delayed_robot_ang_vel[:, i, :] = robot_ang_vel_delayed.clone()
                self.states.delayed_robot_lin_acc[:, i, :] = robot_lin_accs_gt[:, i, :].clone() # No delay for acceleration
                self.states.delayed_gimbal_yaw[:, i] = gimbal_yaw_delayed.squeeze(-1).clone()
                self.states.delayed_gimbal_pitch[:, i] = gimbal_pitch_delayed.squeeze(-1).clone()
                self.states.delayed_zoom_level[:, i] = self.zoom_level[agent_id].clone()
            
            # Compute delayed camera poses (without noise)
            gimbal_roll_delayed = gimbal_rolls_gt  # Roll is not delayed (assuming no control)
            camera_pos_delayed, camera_quat_delayed = self.states.compute_camera_poses(
                robot_pos=self.states.delayed_robot_pos,
                robot_quat=self.states.delayed_robot_quat,
                gimbal_yaw=self.states.delayed_gimbal_yaw,
                gimbal_pitch=self.states.delayed_gimbal_pitch,
                gimbal_roll=gimbal_roll_delayed,
                camera_offset_rot=self.camera_offset_rot_batch,
                camera_offset_pos=self.camera_offset_pos_batch,
            )
            self.states.camera_pos_delayed = camera_pos_delayed
            self.states.camera_quat_delayed = camera_quat_delayed
            
            # Compute delayed intrinsics (without noise)
            self.states.camera_intrinsics_delayed = self.states.compute_intrinsics_with_zoom(
                base_intrinsics=base_intrinsics_stacked,
                zoom_level=self.states.delayed_zoom_level,
            )
            
            for i, agent_id in enumerate(self.cfg.possible_agents):
                # Add current detection to the delay system (FPS-throttled internally)
                # - All frames are sent to the delay manager at each sim step
                # - Frames are captured at detection_fps rate (independent of bbox validity)
                # - Each captured frame stores bbox data and validity status
                self.delay_manager.add_detection(
                    i,
                    self.bbox_raycaster.data.bboxes[:, i, 0, :],  # Current bbox data [N, 4]
                    self.bbox_raycaster.data.valid_mask[:, i, 0]  # Current bbox validity [N]
                )

                # Retrieve delayed detection after latency
                # delayed_detection.data: ALL delayed frames?
                # delayed_detection.valid: whether a frame has arrived (after delay)
                # detection_valid_mask: whether the frame contains valid bbox
                delayed_detection, detection_valid_mask = self.delay_manager.get_delayed_detection(i)

                # Update states when a new frame arrives (delayed_detection.valid == True)
                # Keep previous states when no new frame (delayed_detection.valid == False)

                # Update bbox data: use new data if frame arrived, else keep old
                self.states.bboxes_delayed[:, i, 0, :] = torch.where(
                    delayed_detection.valid.unsqueeze(-1),  # [N] -> [N, 1]
                    delayed_detection.data,  # New bbox data [N, 4]
                    self.states.bboxes_delayed[:, i, 0, :]  # Previous bbox data [N, 4]
                )

                # Update validity mask: use new validity if frame arrived, else keep old
                self.states.valid_mask_delayed[:, i, 0] = torch.where(
                    delayed_detection.valid,  # [N]
                    detection_valid_mask.squeeze(-1),  # New validity [N, 1] -> [N]
                    self.states.valid_mask_delayed[:, i, 0]  # Previous validity [N]
                )
                self.states.bboxes_delayed[:, i, 0, :] = torch.where(
                    self.states.valid_mask_delayed[:, i, 0].unsqueeze(-1),  # [N] -> [N, 1]
                    self.states.bboxes_delayed[:, i, 0, :],  # Keep bbox if valid
                    torch.zeros_like(self.states.bboxes_delayed[:, i, 0, :])  # Zero bbox if invalid
                )

                # Update time since detection:
                # - If new frame arrived: reset to detection_age (timestamp age)
                # - If no new frame: increment by step_dt (time keeps increasing)
                current_time = self.delay_manager.current_time  # [N]
                detection_age = current_time - delayed_detection.timestamp  # [N]

                self.states.time_since_detection_delayed[:, i, 0] = torch.where(
                    torch.logical_and(delayed_detection.valid, detection_valid_mask.squeeze(-1)),  # [N] - did we receive a new frame AND does it have a valid bbox?
                    detection_age,  # Reset to age of received frame
                    self.states.time_since_detection_delayed[:, i, 0] + self.step_dt  # Increment time
                )
            
            self.states.ray_dir_delayed = get_ray_dir_from_bbox(
                self.states.bboxes_delayed,
                self.states.camera_intrinsics_delayed,
                self.states.camera_quat_delayed,
                self.states.camera_pos_delayed,
            )
            self.states.ray_dir_delayed = torch.where(
                self.states.valid_mask_delayed.unsqueeze(-1),  # [N, C, T] -> [N, C, T, 1]
                self.states.ray_dir_delayed,
                torch.zeros_like(self.states.ray_dir_delayed)
            )
            
            # ===== 2.5. BROADCAST AND RECEIVE AGENT STATES (INTER-AGENT COMMUNICATION) =====
            # Phase 1: Each agent broadcasts its delayed state to all other agents
            for i, agent_id in enumerate(self.cfg.possible_agents):
                agent_numeric_id = self.agent_name_to_idx[agent_id]
                
                # Prepare state dictionary with delayed states (WITHOUT noise)
                # These will be transmitted with communication latency and dropouts
                state_dict = {
                    'position': self.states.delayed_robot_pos[:, i, :],        # [N, 3]
                    'orientation': self.states.delayed_robot_quat[:, i, :],    # [N, 4]
                    'linear_velocity': self.states.delayed_robot_lin_vel[:, i, :],  # [N, 3]
                    'angular_velocity': self.states.delayed_robot_ang_vel[:, i, :],  # [N, 3]
                    'linear_acceleration': self.states.delayed_robot_lin_acc[:, i, :],  # [N, 3]
                    'gimbal_yaw': self.states.delayed_gimbal_yaw[:, i].unsqueeze(-1),  # [N, 1]
                    'gimbal_pitch': self.states.delayed_gimbal_pitch[:, i].unsqueeze(-1),  # [N, 1]
                    'camera_pos': self.states.camera_pos_delayed[:, i, :],     # [N, 3]
                    'camera_quat': self.states.camera_quat_delayed[:, i, :],   # [N, 4]
                    'bbox': self.states.bboxes_delayed[:, i, 0, :],            # [N, 4] (only first target)
                    'bbox_valid': self.states.valid_mask_delayed[:, i, 0].float(),  # [N]
                    'time_since_detection': self.states.time_since_detection_delayed[:, i, 0],  # [N]
                    'ray_dir': self.states.ray_dir_delayed[:, i, 0, :],        # [N, 3] (only first target)
                }
                
                # Broadcast to all other agents (with latency and dropouts)
                self.delay_manager.broadcast_state(
                    sender_id=agent_numeric_id,
                    state_dict=state_dict,
                    valid_mask=None  # All messages are valid
                )
            
            # Phase 2: Each agent receives states from other agents
            for i, agent_id in enumerate(self.cfg.possible_agents):
                agent_numeric_id = self.agent_name_to_idx[agent_id]
                
                # Receive messages from other agents
                received_data = self.delay_manager.receive_other_agent_states(
                    receiver_id=agent_numeric_id,
                    state_template=None
                )
                
                # Initialize storage for this agent's received data
                num_other_agents = len(self.cfg.possible_agents) - 1
                if agent_id not in self.states.received_positions:
                    # Initialize with default values
                    self.states.received_positions[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 3, device=self.device
                    )
                    self.states.received_orientations[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 4, device=self.device
                    )
                    self.states.received_orientations[agent_id][:, :, 0] = 1.0  # Identity quaternion
                    self.states.received_linear_velocities[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 3, device=self.device
                    )
                    self.states.received_angular_velocities[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 3, device=self.device
                    )
                    self.states.received_linear_accelerations[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 3, device=self.device
                    )
                    self.states.received_gimbal_yawpitch[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 2, device=self.device
                    )
                    self.states.received_camera_pos[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 3, device=self.device
                    )
                    self.states.received_camera_quat[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 4, device=self.device
                    )
                    self.states.received_camera_quat[agent_id][:, :, 0] = 1.0  # Identity quaternion
                    self.states.received_bboxes[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 1, 4, device=self.device
                    )
                    self.states.received_bbox_valid[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 1, device=self.device, dtype=torch.bool
                    )
                    self.states.received_ray_dirs[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, 1, 3, device=self.device
                    )
                    self.states.received_timestamps[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, device=self.device
                    )
                    self.states.received_valid[agent_id] = torch.zeros(
                        self.num_envs, num_other_agents, device=self.device, dtype=torch.bool
                    )
                    self.states.received_age[agent_id] = torch.full(
                        (self.num_envs, num_other_agents), self.cfg.max_time_since_comm, device=self.device
                    )
                    self.states.received_time_since_detection[agent_id] = torch.full(
                        (self.num_envs, num_other_agents, 1), self.cfg.max_time_since_detection, device=self.device
                    )
                
                # Process received data from each sender
                other_agent_idx = 0
                for sender_numeric_id in range(len(self.cfg.possible_agents)):
                    if sender_numeric_id == agent_numeric_id:
                        continue  # Skip self
                    
                    if sender_numeric_id in received_data:
                        # Extract timestamped data
                        data = received_data[sender_numeric_id]
                        
                        # Update received states for this sender
                        pos_data = data['position']
                        if pos_data.valid.any():
                            # Update positions where valid
                            self.states.received_positions[agent_id][:, other_agent_idx, :] = torch.where(
                                pos_data.valid.unsqueeze(-1),
                                pos_data.data,
                                self.states.received_positions[agent_id][:, other_agent_idx, :]
                            )
                        
                        ori_data = data['orientation']
                        if ori_data.valid.any():
                            self.states.received_orientations[agent_id][:, other_agent_idx, :] = torch.where(
                                ori_data.valid.unsqueeze(-1),
                                ori_data.data,
                                self.states.received_orientations[agent_id][:, other_agent_idx, :]
                            )
                        
                        lin_vel_data = data['linear_velocity']
                        if lin_vel_data.valid.any():
                            self.states.received_linear_velocities[agent_id][:, other_agent_idx, :] = torch.where(
                                lin_vel_data.valid.unsqueeze(-1),
                                lin_vel_data.data,
                                self.states.received_linear_velocities[agent_id][:, other_agent_idx, :]
                            )
                        
                        ang_vel_data = data['angular_velocity']
                        if ang_vel_data.valid.any():
                            self.states.received_angular_velocities[agent_id][:, other_agent_idx, :] = torch.where(
                                ang_vel_data.valid.unsqueeze(-1),
                                ang_vel_data.data,
                                self.states.received_angular_velocities[agent_id][:, other_agent_idx, :]
                            )

                        lin_acc_data = data['linear_acceleration']
                        if lin_acc_data.valid.any():
                            self.states.received_linear_accelerations[agent_id][:, other_agent_idx, :] = torch.where(
                                lin_acc_data.valid.unsqueeze(-1),
                                lin_acc_data.data,
                                self.states.received_linear_accelerations[agent_id][:, other_agent_idx, :]
                            )
                        
                        # Combine gimbal yaw and pitch
                        yaw_data = data['gimbal_yaw']
                        pitch_data = data['gimbal_pitch']
                        if yaw_data.valid.any() and pitch_data.valid.any():
                            gimbal_combined = torch.stack([yaw_data.data.squeeze(-1), pitch_data.data.squeeze(-1)], dim=-1)
                            self.states.received_gimbal_yawpitch[agent_id][:, other_agent_idx, :] = torch.where(
                                yaw_data.valid.unsqueeze(-1),
                                gimbal_combined,
                                self.states.received_gimbal_yawpitch[agent_id][:, other_agent_idx, :]
                            )
                        
                        cam_pos_data = data['camera_pos']
                        if cam_pos_data.valid.any():
                            self.states.received_camera_pos[agent_id][:, other_agent_idx, :] = torch.where(
                                cam_pos_data.valid.unsqueeze(-1),
                                cam_pos_data.data,
                                self.states.received_camera_pos[agent_id][:, other_agent_idx, :]
                            )
                        
                        cam_quat_data = data['camera_quat']
                        if cam_quat_data.valid.any():
                            self.states.received_camera_quat[agent_id][:, other_agent_idx, :] = torch.where(
                                cam_quat_data.valid.unsqueeze(-1),
                                cam_quat_data.data,
                                self.states.received_camera_quat[agent_id][:, other_agent_idx, :]
                            )
                        
                        bbox_data = data['bbox']
                        if bbox_data.valid.any():
                            self.states.received_bboxes[agent_id][:, other_agent_idx, 0, :] = torch.where(
                                bbox_data.valid.unsqueeze(-1),
                                bbox_data.data,
                                self.states.received_bboxes[agent_id][:, other_agent_idx, 0, :]
                            )
                        
                        bbox_valid_data = data['bbox_valid']
                        if bbox_valid_data.valid.any():
                            self.states.received_bbox_valid[agent_id][:, other_agent_idx, 0] = torch.where(
                                bbox_valid_data.valid,
                                bbox_valid_data.data.bool(),
                                self.states.received_bbox_valid[agent_id][:, other_agent_idx, 0]
                            )
                        tsd_valid_data = data['time_since_detection']
                        if tsd_valid_data.valid.any():
                            current_time = self.delay_manager.current_time
                            comm_delay = current_time - tsd_valid_data.timestamp
                            tsd_comm_delay_time = tsd_valid_data.data + comm_delay
                            self.states.received_time_since_detection[agent_id][:, other_agent_idx, 0] = torch.where(
                                tsd_valid_data.valid,
                                tsd_comm_delay_time,
                                self.states.received_time_since_detection[agent_id][:, other_agent_idx, 0]
                            )
                        
                        ray_dir_data = data['ray_dir']
                        if ray_dir_data.valid.any():
                            self.states.received_ray_dirs[agent_id][:, other_agent_idx, 0, :] = torch.where(
                                ray_dir_data.valid.unsqueeze(-1),
                                ray_dir_data.data,
                                self.states.received_ray_dirs[agent_id][:, other_agent_idx, 0, :]
                            )
                        
                        # Update metadata (use position data as representative)
                        current_time = self.delay_manager.current_time
                        self.states.received_timestamps[agent_id][:, other_agent_idx] = torch.where(
                            pos_data.valid,
                            pos_data.timestamp,
                            self.states.received_timestamps[agent_id][:, other_agent_idx]
                        )
                        self.states.received_valid[agent_id][:, other_agent_idx] = torch.where(
                            pos_data.valid,
                            torch.ones_like(pos_data.valid),
                            self.states.received_valid[agent_id][:, other_agent_idx]
                        )
                        
                        # Compute age of received data
                        age = current_time - self.states.received_timestamps[agent_id][:, other_agent_idx]
                        self.states.received_age[agent_id][:, other_agent_idx] = torch.where(
                            self.states.received_valid[agent_id][:, other_agent_idx],
                            age,
                            torch.full_like(age, self.cfg.max_time_since_comm)
                        )
                    
                    other_agent_idx += 1
            
            # ===== 3. COMPUTE DELAYED + NOISY STATES FOR OBSERVATIONS =====
            # Apply measurement noise to delayed states
            for i, agent_id in enumerate(self.cfg.possible_agents):
                # Add noise to delayed position
                self.states.delayed_noisy_robot_pos[:, i, :] = \
                    self.states.delayed_robot_pos[:, i, :] + self.sampled_noise_twb[:, i, :]
                self.states.delayed_noisy_robot_lin_vel[:, i, :] = \
                    self.states.delayed_robot_lin_vel[:, i, :] + self.sampled_noise_lin_vel[:, i, :]
                self.states.delayed_noisy_robot_ang_vel[:, i, :] = \
                    self.states.delayed_robot_ang_vel[:, i, :] + self.sampled_noise_ang_vel[:, i, :]
                self.states.delayed_noisy_robot_lin_acc[:, i, :] = \
                    self.states.delayed_robot_lin_acc[:, i, :] + self.sampled_noise_lin_acc[:, i, :]

                # Add noise to delayed orientation
                robot_roll, robot_pitch, robot_yaw = euler_xyz_from_quat(self.states.delayed_robot_quat[:, i, :])
                self.states.delayed_noisy_robot_quat[:, i, :] = quat_from_euler_xyz(
                    robot_roll + self.sampled_noise_phiwb[:, i, 0],
                    robot_pitch + self.sampled_noise_phiwb[:, i, 1],
                    robot_yaw + self.sampled_noise_phiwb[:, i, 2],
                )
                
                # Add noise to delayed gimbal angles
                self.states.delayed_noisy_gimbal_yaw[:, i] = \
                    self.states.delayed_gimbal_yaw[:, i] + self.sampled_noise_alpha[:, i, 0]
                self.states.delayed_noisy_gimbal_pitch[:, i] = \
                    self.states.delayed_gimbal_pitch[:, i] + self.sampled_noise_beta[:, i, 0]
            
            # Compute delayed+noisy camera poses
            camera_pos_delayed_noisy, camera_quat_delayed_noisy = self.states.compute_camera_poses(
                robot_pos=self.states.delayed_noisy_robot_pos,
                robot_quat=self.states.delayed_noisy_robot_quat,
                gimbal_yaw=self.states.delayed_noisy_gimbal_yaw,
                gimbal_pitch=self.states.delayed_noisy_gimbal_pitch,
                gimbal_roll=gimbal_roll_delayed,
                camera_offset_rot=self.camera_offset_rot_batch,
                camera_offset_pos=self.camera_offset_pos_batch,
            )
            self.states.camera_pos_delayed_noisy = camera_pos_delayed_noisy
            self.states.camera_quat_delayed_noisy = camera_quat_delayed_noisy
            
            # Compute delayed+noisy intrinsics
            self.states.camera_intrinsics_delayed_noisy = self.states.compute_intrinsics_with_zoom(
                base_intrinsics=base_intrinsics_stacked,
                zoom_level=self.states.delayed_zoom_level,
                noise=self.sampled_noise_K,
            )
            
            # Compute ray directions from delayed+noisy states
            # Use delayed+noisy bboxes (with pixel noise)
            self.states.bboxes_delayed_noisy = self.states.bboxes_delayed.clone()
            self.states.bboxes_delayed_noisy[:, :, :, :2] += self.sampled_noise_pix[:, :, :, :2].clone()  # x, y
            self.states.bboxes_delayed_noisy[:, :, :, 2:] += self.sampled_noise_pix[:, :, :, 2:].clone()  # w, h
            self.states.valid_mask_delayed_noisy = self.states.valid_mask_delayed.clone()
            self.states.time_since_detection_delayed_noisy = self.states.time_since_detection_delayed.clone()
            
            self.states.ray_dir_delayed_noisy = get_ray_dir_from_bbox(
                self.states.bboxes_delayed_noisy,
                self.states.camera_intrinsics_delayed_noisy,
                self.states.camera_quat_delayed_noisy,
                self.states.camera_pos_delayed_noisy,
            )
            self.states.ray_dir_delayed_noisy = torch.where(
                self.states.valid_mask_delayed_noisy.unsqueeze(-1),  # [N, C, T] -> [N, C, T, 1]
                self.states.ray_dir_delayed_noisy,
                torch.zeros_like(self.states.ray_dir_delayed_noisy)
            )
        else:
            # No delay system - just copy GT states
            self.states.delayed_robot_pos = robot_positions_gt.clone()
            self.states.delayed_robot_quat = robot_quats_gt.clone()
            self.states.delayed_robot_lin_vel = robot_lin_vels_gt.clone()
            self.states.delayed_robot_ang_vel = robot_ang_vels_gt.clone()
            self.states.delayed_robot_lin_acc = robot_lin_accs_gt.clone()
            self.states.delayed_gimbal_yaw = gimbal_yaws_gt.clone()
            self.states.delayed_gimbal_pitch = gimbal_pitches_gt.clone()
            self.states.delayed_zoom_level = zoom_levels_gt.clone()
            self.states.camera_pos_delayed = camera_pos_gt.clone()
            self.states.camera_quat_delayed = camera_quat_gt.clone()
            self.states.camera_intrinsics_delayed = camera_intrinsics_gt.clone()
            self.states.bboxes_delayed = self.bbox_raycaster.data.bboxes.clone()
            self.states.valid_mask_delayed = self.bbox_raycaster.data.valid_mask.clone()
            self.states.ray_dir_delayed = get_ray_dir_from_bbox(
                self.states.bboxes_delayed,
                self.states.camera_intrinsics_delayed,
                self.states.camera_quat_delayed,
                self.states.camera_pos_delayed,
            )
            self.states.ray_dir_delayed = torch.where(
                self.states.valid_mask_delayed.unsqueeze(-1),  # [N, C, T] -> [N, C, T, 1]
                self.states.ray_dir_delayed,
                torch.zeros_like(self.states.ray_dir_delayed)
            )
            
            # Delayed+noisy is same as delayed when delay system is off
            self.states.delayed_noisy_robot_pos = self.states.delayed_robot_pos.clone()
            self.states.delayed_noisy_robot_quat = self.states.delayed_robot_quat.clone()
            self.states.delayed_noisy_robot_lin_vel = self.states.delayed_robot_lin_vel.clone()
            self.states.delayed_noisy_robot_ang_vel = self.states.delayed_robot_ang_vel.clone()
            self.states.delayed_noisy_robot_lin_acc = self.states.delayed_robot_lin_acc.clone()
            self.states.delayed_noisy_gimbal_yaw = self.states.delayed_gimbal_yaw.clone()
            self.states.delayed_noisy_gimbal_pitch = self.states.delayed_gimbal_pitch.clone()
            self.states.camera_pos_delayed_noisy = self.states.camera_pos_delayed.clone()
            self.states.camera_quat_delayed_noisy = self.states.camera_quat_delayed.clone()
            self.states.camera_intrinsics_delayed_noisy = self.states.camera_intrinsics_delayed.clone()
            self.states.bboxes_delayed_noisy = self.states.bboxes_delayed.clone()
            self.states.valid_mask_delayed_noisy = self.states.valid_mask_delayed.clone()
            self.states.ray_dir_delayed_noisy = self.states.ray_dir_delayed.clone()

    def _compute_triangulation_covariance(self, X_w, robot_positions, robot_quats,
                                          gimbal_yaws, gimbal_pitches,
                                          camera_intrinsics, bbox_valid_mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        try:
            Sigma_X, trace_cov = triangulation_covariance_multi_camera(
                X_w=X_w,
                robot_positions=robot_positions,
                robot_quats=robot_quats,
                gimbal_yaws=gimbal_yaws,
                gimbal_pitches=gimbal_pitches,
                camera_intrinsics=camera_intrinsics,
                Sigma_pix=self.Sigma_pix * self.progress_coord,
                Sigma_twb=self.Sigma_twb * self.progress_coord,
                Sigma_phiwb=self.Sigma_phiwb * self.progress_coord,
                Sigma_alpha=self.Sigma_alpha * self.progress_coord,
                Sigma_beta=self.Sigma_beta * self.progress_coord,
                Sigma_K=self.Sigma_K * self.progress_coord,
                include_pose=True,
                include_gimbal=True,
                include_intrinsics=True
            )
            
            is_trace_valid = (trace_cov >= 0.0) & torch.isfinite(trace_cov)
            is_sigma_finite = torch.isfinite(Sigma_X).all(dim=(-2, -1))
            is_not_nan = ~torch.isnan(Sigma_X).any(dim=(-2, -1))
            is_all_bboxes_valid = bbox_valid_mask.all(dim=1)
            is_tri_cov_valid = is_trace_valid & is_sigma_finite & is_not_nan & is_all_bboxes_valid
            
            Sigma_X = torch.where(
                is_tri_cov_valid.unsqueeze(-1).unsqueeze(-1),
                Sigma_X,
                self.Sigma_X_invalid
            )
            trace_cov = torch.where(
                is_tri_cov_valid,
                trace_cov,
                self.trace_cov_invalid
            )
        except Exception as e:
            # print(f"Warning: Triangulation covariance (delayed) computation failed: {e}")
            is_tri_cov_valid = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            Sigma_X = self.Sigma_X_invalid.clone()
            trace_cov = self.trace_cov_invalid.clone()

        return Sigma_X, trace_cov, is_tri_cov_valid

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """Compute rewards for all agents."""
        # STEP 1: Compute all intermediate values (GT, delayed, delayed+noisy)
        self._compute_intermediate_values()
        
        rewards_dict = {}
        
        # STEP 2: Compute triangulation quality using DELAYED (without noise) values for rewards
        target_pos_stacked = self.target.data.root_state_w[:, :3].unsqueeze(1)  # [N, 1, 3]
        self.X_w_est = midpoint_method_batched(
                pts=torch.stack([self.states.camera_pos_delayed[:,i,:] for i, _ in enumerate(self.cfg.possible_agents)], dim=1),
                dirs=self.states.ray_dir_delayed,
            )
        self.states.Sigma_X_delayed, self.states.trace_cov_delayed, is_tri_cov_valid = self._compute_triangulation_covariance(
            X_w=self.X_w_est, # Use estimated target position instead of GT target_pos_stacked for reward computation
            robot_positions=self.states.delayed_robot_pos,
            robot_quats=self.states.delayed_robot_quat,
            gimbal_yaws=self.states.delayed_gimbal_yaw,
            gimbal_pitches=self.states.delayed_gimbal_pitch,
            camera_intrinsics=self.states.camera_intrinsics_delayed,
            bbox_valid_mask=self.states.valid_mask_delayed,
        )
        trace_cov_per_env = self.states.trace_cov_delayed[:, 0]
        triangulation_quality = torch.where(
            is_tri_cov_valid[:, 0],
            1.0 / torch.sqrt(trace_cov_per_env + 1e-12),
            torch.zeros(self.num_envs, device=self.device)
        )

        # STEP 3: Compute distances and collisions
        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            other_agent_idxs = list(range(len(self.cfg.possible_agents)))
            other_agent_idxs.pop(i)

            ego_pos = self.states.delayed_robot_pos[:, i, :].clone()
            ego_vel = self.states.delayed_robot_lin_vel[:, i, :].clone()
            target_pos = self.target.data.root_pos_w.clone()
            target_vel = self.target.data.root_lin_vel_w.clone()
            self.cam_to_target_distance[:, i, 0] = torch.norm(target_pos - ego_pos, dim=1)
            
            for j, other_agent_id in enumerate(self.cfg.possible_agents):
                if i >= j:
                    continue
                other_pos = self.states.delayed_robot_pos[:, j, :].clone()
                other_vel = self.states.delayed_robot_lin_vel[:, j, :].clone()

                distance = torch.norm(other_pos - ego_pos, dim=1)
                self.cam_to_cam_distance[:, i, j] = distance
                self.cam_to_cam_distance[:, j, i] = distance
                
                ttc_penalty = self.compute_ttc_from_pos_vel(
                    ego_pos=ego_pos,
                    ego_vel=ego_vel,
                    other_pos=other_pos,
                    other_vel=other_vel,
                )
                self.cam_to_cam_ttc[:, i, j] = ttc_penalty
                self.cam_to_cam_ttc[:, j, i] = ttc_penalty
        
        _, ttc_penalty_cam_to_target, self.cam_to_target_ttc[:, :, 0] = self.compute_looming_ttc_penalty_zoom_invariant(
                bbox_w_dn=self.states.bboxes_delayed_noisy[:, :, 0, 2],  # [N, C], bbox width
                bbox_h_dn=self.states.bboxes_delayed_noisy[:, :, 0, 3],  # [N, C], bbox height
                valid_dn=self.states.valid_mask_delayed_noisy[:, :, 0],  # [N, C], bool for valid_dn
                fx_delayed=self.states.camera_intrinsics_delayed[:, :, 0, 0],  # [N, C], fx
                fy_delayed=self.states.camera_intrinsics_delayed[:, :, 1, 1],  # [N, C], fy
                dt=self.step_dt,
                horizon_sec=self.cfg.ttc_horizon,
                ema_alpha_s=0.6,
                ema_alpha_f=0.6,
                deriv_clip=0.5,
                eps=1e-6,
                stale_half_life=0.2,
                zoom_gate_k=0.2,                         # gate aggressiveness for |d ln f|
                aggregate="max"                          # "max" (worst agent) or "mean"
        )

        self.cam_to_cam_distance[:, range(len(self.cfg.possible_agents)), range(len(self.cfg.possible_agents))] = float('inf')
        self.cam_to_cam_ttc[:, range(len(self.cfg.possible_agents)), range(len(self.cfg.possible_agents))] = float('inf')
        cam_to_cam_collision = self.cam_to_cam_distance < self.cfg.min_safe_distance * (0.5 + 0.5 * self.progress_coord)
        cam_to_target_collision = self.cam_to_target_distance < self.cfg.min_safe_distance * (0.5 + 0.5 * self.progress_coord)
        
        # STEP 4: Compute per-agent rewards
        for i, agent_id in enumerate(self.cfg.possible_agents):
            agent_cam_cfg = self.agent_camera_cfgs[agent_id]
            
            # Individual tracking rewards
            action_sum = torch.sum(torch.square(self.action_weight * self._actions[agent_id]), dim=1)
            action_delta = torch.sum(
                torch.square(self.action_delta_weight * (self._actions[agent_id] - self._last_actions[agent_id])), 
                dim=1
            )
            
            # Bounding box rewards
            self.bboxes[agent_id] = self.states.bboxes_delayed_noisy[:, i, 0, :]
            # Case 1) Bounding box valid mask based on boolean valid mask
            self.bbox_valid_mask[agent_id] = self.states.valid_mask_delayed_noisy[:, i]
            # self.bbox_valid_mask[agent_id] = torch.where(
            #     self.states.time_since_detection_delayed_noisy[:, i, 0] < self.cfg.max_time_since_detection,
            #     torch.ones_like(self.states.valid_mask_delayed_noisy[:, i, 0], dtype=torch.bool),
            #     torch.zeros_like(self.states.valid_mask_delayed_noisy[:, i, 0], dtype=torch.bool)
            # )
            # Case 2) Bounding box valid mask based on time since detection
            # detection_period = 1 / self.cfg.detection_fps * progress + (self.step_dt) * (1-progress)
            # detection_weight = torch.exp(-self.states.time_since_detection_delayed_noisy[:, i, 0] / (detection_period + self.cfg.detection_mean_latency))
            bbox = (torch.square((self.bboxes[agent_id][:, 0] + self.bboxes[agent_id][:, 2])/2 - agent_cam_cfg.width/2) / agent_cam_cfg.width**2 
                    + torch.square((self.bboxes[agent_id][:, 1] + self.bboxes[agent_id][:, 3])/2 - agent_cam_cfg.height/2) / agent_cam_cfg.height**2)
            bbox_center_mapped = torch.exp(-((bbox)**2)/self.cfg.bbox_reward_shape_width**2) * self.bbox_valid_mask[agent_id].squeeze(-1).float()
            # bbox_center_mapped = torch.exp(-((bbox)**2)/self.cfg.bbox_reward_shape_width**2) * detection_weight
            
            bbox_size = (self.bboxes[agent_id][:, 2] - self.bboxes[agent_id][:, 0]) * (self.bboxes[agent_id][:, 3] - self.bboxes[agent_id][:, 1]) / (agent_cam_cfg.width * agent_cam_cfg.height)
            bbox_size_mapped = torch.exp(-((bbox_size - self.cfg.bbox_size_preferred)**2)/self.cfg.bbox_reward_shape_width**2) * self.bbox_valid_mask[agent_id].squeeze(-1).float()
            # bbox_size_mapped = torch.exp(-((bbox_size - self.cfg.bbox_size_preferred)**2)/self.cfg.bbox_reward_shape_width**2) * detection_weight
            
            # Zoom penalty
            zoom_penalty = torch.square((self.zoom_level[agent_id] - 1.0)/10)

            # Collision penalty
            has_collision = torch.any(cam_to_cam_collision[:, i, :], dim=1) | torch.any(cam_to_target_collision[:, i, :], dim=1)
            collision_penalty = torch.where(
                has_collision,
                torch.ones(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )
            ttc_cam_to_cam_normalized = torch.clamp(
                (self.cam_to_cam_ttc[:, i, :] / self.cfg.ttc_horizon), min=0, max=1
            )
            ttc_penalty_cam_to_cam_all = 1.0 - ttc_cam_to_cam_normalized # (N, C) for agent i to all other agents including self...
            # Remove the i-th column (excluding self-comparison)
            ttc_penalty_cam_to_cam = ttc_penalty_cam_to_cam_all[:, other_agent_idxs]  # (N, C-1)

            # ttc_softmax_temp = 0.6 - 0.4 * self.progress_coord
            # ttc_penalty = self.combine_ttc_penalties(ttc_penalty_cam_to_cam, ttc_penalty_cam_to_target[:, i].unsqueeze(-1), temp=ttc_softmax_temp)

            ttc_penalty = torch.max(
                torch.max(ttc_penalty_cam_to_cam, dim=1).values,
                ttc_penalty_cam_to_target[:, i]
            )

            # Combine rewards
            rewards = {
                "action_sum": action_sum * self.cfg.action_sum_penalty_scale * self.step_dt,
                "action_delta": action_delta * self.cfg.action_delta_penalty_scale * self.step_dt,
                "bbox_center": bbox_center_mapped * self.cfg.bbox_center_reward_scale * self.step_dt,
                "bbox_size": bbox_size_mapped * self.cfg.bbox_size_reward_scale * self.step_dt,
                "triangulation": triangulation_quality * self.cfg.triangulation_reward_scale * self.step_dt * self.progress_coord,
                "collision": collision_penalty * self.cfg.collision_penalty_scale * self.step_dt * self.progress_coord,
                "ttc_penalty": ttc_penalty * self.cfg.ttc_penalty_scale * self.step_dt * self.progress_coord,
            }
            
            # Store for logging
            for key, value in rewards.items():
                self._episode_sums[agent_id][key] += value

            total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            
            # Check for NaN values
            for reward_name, reward_value in rewards.items():
                if torch.isnan(reward_value).any():
                    nan_envs = torch.nonzero(torch.isnan(reward_value)).flatten()
                    raise ValueError(f"NaN detected in {agent_id} reward '{reward_name}' at environments: {nan_envs.tolist()}")
            
            if torch.isnan(total_reward).any():
                nan_envs = torch.nonzero(torch.isnan(total_reward)).flatten()
                raise ValueError(f"NaN detected in {agent_id} total reward at environments: {nan_envs.tolist()}")
            
            rewards_dict[agent_id] = torch.sum(torch.stack(list(rewards.values())), dim=0)
            
            # Update last actions
            self._last_actions[agent_id] = self._actions[agent_id].clone()
    
        return rewards_dict

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents using delayed+noisy states."""
        observations = {}
        
        if self.cfg.enable_delay_system:
            # Compute triangulation covariance with DELAYED + NOISY states for observations
            self.X_w_est = midpoint_method_batched(
                pts=torch.stack([self.states.camera_pos_delayed_noisy[:,i,:] for i, _ in enumerate(self.cfg.possible_agents)], dim=1),
                dirs=self.states.ray_dir_delayed_noisy,
            )
            self.states.Sigma_X_delayed_noisy, self.states.trace_cov_delayed_noisy, is_tri_cov_valid = self._compute_triangulation_covariance(
                X_w=self.X_w_est,
                robot_positions=self.states.delayed_noisy_robot_pos,
                robot_quats=self.states.delayed_noisy_robot_quat,
                gimbal_yaws=self.states.delayed_noisy_gimbal_yaw,
                gimbal_pitches=self.states.delayed_noisy_gimbal_pitch,
                camera_intrinsics=self.states.camera_intrinsics_delayed_noisy,
                bbox_valid_mask=self.states.valid_mask_delayed_noisy,
            )
        else:
            # No delay system - use same as delayed
            self.states.Sigma_X_delayed_noisy = self.states.Sigma_X_delayed.clone()
            self.states.trace_cov_delayed_noisy = self.states.trace_cov_delayed.clone()
        
        # Extract Sigma diagonal for observations (using delayed+noisy covariance)
        Sigma_X_obs = self.states.Sigma_X_delayed_noisy
        Sigma_diag = torch.diagonal(Sigma_X_obs, dim1=-2, dim2=-1)  # (N, T, 3)
        Sigma_diag_sqrt = torch.sqrt(torch.clamp(Sigma_diag, min=0.0) + 1e-12)  # (N, T, 3)
        
        # Update stacked arrays with delayed+noisy states for observations
        self.robot_positions_stacked = self.states.delayed_noisy_robot_pos
        self.robot_quats_stacked = self.states.delayed_noisy_robot_quat
        self.robot_velocity_stacked = self.states.delayed_noisy_robot_lin_vel
        self.robot_angular_velocity_stacked = self.states.delayed_noisy_robot_ang_vel
        self.robot_linear_acceleration_stacked = self.states.delayed_noisy_robot_lin_acc
        self.gimbal_yaws_stacked = self.states.delayed_noisy_gimbal_yaw
        self.gimbal_pitches_stacked = self.states.delayed_noisy_gimbal_pitch
        
        for i, agent_id in enumerate(self.cfg.possible_agents):
            other_agent_idxs = list(range(len(self.cfg.possible_agents)))
            other_agent_idxs.pop(i)
            
            _, _, ego_yaw = euler_xyz_from_quat(self.robot_quats_stacked[:, i, :])
            
            # Use delayed+noisy bboxes for observations
            bbox_obs = self.states.bboxes_delayed_noisy[:, i, 0, :]  # (N, 4)
            bbox_valid = self.states.valid_mask_delayed_noisy[:, i, 0]  # (N,)
            time_since_detection = self.states.time_since_detection_delayed_noisy[:, i, 0]  # (N,)
            time_since_norm = torch.clamp(time_since_detection / self.cfg.max_time_since_detection, max=1.0).unsqueeze(-1)  # (N, 1)
            
            # Normalize bbox
            bbox_norm = self.bbox_raycaster.get_normalized_bboxes(bbox_obs.unsqueeze(1).unsqueeze(1))[:, 0, 0, :]  # (N, 4)
            bbox_norm = torch.where(
                bbox_valid.unsqueeze(-1),
                bbox_norm,
                torch.zeros_like(bbox_norm)
            )

            # ===== PHASE 4: USE RECEIVED STATES FOR OTHER AGENTS =====
            # Instead of using ground truth delayed+noisy states, use received states with comm latency
            if self.cfg.enable_delay_system and agent_id in self.states.received_positions:
                # Use received states from other agents (already have comm latency applied)
                other_positions = self.states.received_positions[agent_id]  # [N, C-1, 3]
                other_velocities = self.states.received_linear_velocities[agent_id]  # [N, C-1, 3]
                other_ray_dirs = self.states.received_ray_dirs[agent_id]  # [N, C-1, T, 3]
                other_bbox_valid = self.states.received_bbox_valid[agent_id]  # [N, C-1, T]
                other_bbox_age = self.states.received_time_since_detection[agent_id]  # [N, C-1, 1]
                
                # For received states that are not valid yet, use defaults
                # Default: position at [0,0,0], velocity at [0,0,0]
                default_pos = torch.zeros_like(other_positions)
                default_vel = torch.zeros_like(other_velocities)
                default_ray_dir = torch.zeros_like(other_ray_dirs)
                default_bbox_valid = torch.zeros_like(other_bbox_valid)
                default_time_since_detection = torch.full_like(other_bbox_age, self.cfg.max_time_since_detection)
                
                # Use received data where valid, otherwise use defaults
                valid_mask = self.states.received_valid[agent_id]  # [N, C-1]
                other_positions_obs = torch.where(
                    valid_mask.unsqueeze(-1),
                    other_positions,
                    default_pos
                )
                other_velocities_obs = torch.where(
                    valid_mask.unsqueeze(-1),
                    other_velocities,
                    default_vel
                )
                other_ray_dirs_obs = torch.where(
                    valid_mask.unsqueeze(-1).unsqueeze(-1),
                    other_ray_dirs,
                    default_ray_dir
                )
                other_bbox_valid_obs = torch.where(
                    valid_mask.unsqueeze(-1),
                    other_bbox_valid.float(),
                    default_bbox_valid.float()
                )
                other_bbox_age_obs = torch.where(
                    valid_mask.unsqueeze(-1),
                    other_bbox_age,
                    default_time_since_detection
                )
            else:
                # Fallback: use delayed+noisy states from all agents (no comm system)
                other_positions_obs = self.robot_positions_stacked[:, other_agent_idxs, :]
                other_velocities_obs = self.robot_velocity_stacked[:, other_agent_idxs, :]
                other_ray_dirs_obs = self.states.ray_dir_delayed_noisy[:, other_agent_idxs, :, :]
                other_bbox_valid_obs = self.states.valid_mask_delayed_noisy[:, other_agent_idxs, :]
                other_bbox_age_obs = self.states.time_since_detection_delayed_noisy[:, other_agent_idxs, :]
            
            obs = torch.cat([
                self.robot_positions_stacked[:, i, :],  # 3: pos
                ego_yaw.unsqueeze(-1),  # 1: yaw
                self.robot_velocity_stacked[:, i, :],  # 3: linear vel
                self.robot_angular_velocity_stacked[:, i, 2:3],  # 1: yaw rate
                self.robot_linear_acceleration_stacked[:, i, :],  # 3: linear acc
                self.gimbal_pitches_stacked[:, i].unsqueeze(-1),  # 1
                self.gimbal_yaws_stacked[:, i].unsqueeze(-1),  # 1
                bbox_norm,  # 4: normalized bbox
                bbox_valid.float().unsqueeze(-1),  # 1: validity
                # time_since_norm,  # 1: time since detection (normalized)
                self.zoom_level[agent_id].unsqueeze(-1),  # 1: zoom level
                other_ray_dirs_obs[:, :, 0, :].flatten(start_dim=1),  # 3*(C-1): ray dirs from other agents
                other_bbox_valid_obs[:, :, 0].flatten(start_dim=1),  # 1*(C-1): validity from other agents
                # other_bbox_age_obs.flatten(start_dim=1),  # 1*(C-1): time since detection from other agents
                other_positions_obs.flatten(start_dim=1),  # 3*(C-1): positions of other agents
                other_velocities_obs.flatten(start_dim=1),  # 3*(C-1): velocities of other agents
                Sigma_diag_sqrt[:, 0, :],  # 3: X, Y, Z std deviations
            ], dim=-1)
            
            # Check for NaN values
            if torch.isnan(obs).any():
                nan_envs = torch.nonzero(torch.isnan(obs)).flatten()
                raise ValueError(f"NaN detected in {agent_id} observation at environments: {nan_envs.tolist()}")

            observations[agent_id] = obs
        
        return observations

    def _get_states(self) -> torch.Tensor:
        """Get global state by concatenating all agent observations."""
        obs_list = []
        for agent_id in self.cfg.possible_agents:
            obs_list.append(self.obs_dict[agent_id])
        return torch.cat(obs_list, dim=-1)

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get termination and timeout flags for all agents."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        no_detection_too_long = self.no_detection_counts >= self.cfg.max_no_detection_sec / self.step_dt
        time_out = time_out
        terminated_dict = {}
        time_out_dict = {}
        
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            died = robot.data.root_pos_w[:, 2] < 5.0
            terminated_dict[agent_id] = died
            time_out_dict[agent_id] = time_out
        
        return terminated_dict, time_out_dict

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at specified indices."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[self.cfg.possible_agents[0]]._ALL_INDICES
        
        # Logging
        all_extras = {}
        for agent_id in self.cfg.possible_agents:
            for key, value in self._episode_sums[agent_id].items():
                episodic_sum_avg = torch.mean(value[env_ids])
                all_extras[f"Episode_Reward/{agent_id}_{key}"] = episodic_sum_avg / self.max_episode_length_s
                self._episode_sums[agent_id][key][env_ids] = 0.0
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(all_extras)

        formation_center = torch.zeros(len(env_ids), 3, device=self.device)
        formation_center[:, 0] = torch.zeros_like(formation_center[:, 0]).uniform_(-5.0, 5.0)
        formation_center[:, 1] = torch.zeros_like(formation_center[:, 1]).uniform_(0, 10.0)
        formation_center[:, 2] = torch.zeros_like(formation_center[:, 2]).uniform_(15.0, 20.0)

        gap_ref = torch.zeros(len(env_ids), 3, device=self.device)
        gap_ref[:, 0] = torch.zeros_like(gap_ref[:, 0]).uniform_(-5.0, 5.0)
        gap_ref[:, 1] = torch.zeros_like(gap_ref[:, 1]).uniform_(0, 10.0)
        gap_ref[:, 2] = torch.zeros_like(gap_ref[:, 2]).uniform_(5.0 * self.progress_move, 7.0 * self.progress_move)

        formation_rotation = quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device).uniform_(-math.pi * self.progress_move, math.pi * self.progress_move)
        )

        # Reset target
        # start = self.cfg.curriculum_tracking_start_step
        # end = self.cfg.curriculum_tracking_end_step
        # progress = (self.common_step_counter - start) / (end - start)
        # progress = 0.1 if (progress < 0.1 or start < 0) else (1 if progress > 1 else progress)
        # progress = 1 if DEBUG_DRAW else progress  # Always use full range in debug mode

        target_pos = formation_center.clone()
        target_pos[:, 2] += gap_ref[:, 2]
        target_pos[:, 0] += torch.zeros_like(target_pos[:, 0]).uniform_(4.0, 8.0 + 80.0 * self.progress_move)
        target_pos[:, 1] += torch.zeros_like(target_pos[:, 1]).uniform_(-3.0 - 30.0 * self.progress_move, 3 + 30.0 * self.progress_move)
        target_pos[:, 2] += torch.zeros_like(target_pos[:, 2]).uniform_(0.0, 5.0)
        target_pos = quat_apply(formation_rotation, target_pos) + self._env_origins[env_ids]
        
        default_target_state = self.target.data.default_root_state[env_ids].clone()
        default_target_state[:, :3] = target_pos
        default_target_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        )
        self.target.write_root_pose_to_sim(default_target_state[:, :7], env_ids)
        self.target.write_root_velocity_to_sim(default_target_state[:, 7:], env_ids)
        
        # Reset target velocity
        # Phase 1. Single-agent tracking
        # Phase 1&2. Target speed, acceleration
        # Phase 2. Delay and noise system
        # Phase 3. Triangulation, collision, TTC

        max_target_speed = self.cfg.max_target_speed * self.progress_move
        max_target_ang_speed = self.cfg.max_target_ang_speed * self.progress_move
        self.target_vel[env_ids, :3] = torch.zeros_like(self.target_vel[env_ids, :3]).uniform_(-max_target_speed, max_target_speed)
        self.target_vel[env_ids, 3:5] = torch.zeros_like(self.target_vel[env_ids, 3:5])
        self.target_vel[env_ids, 5] = torch.zeros_like(self.target_vel[env_ids, 5]).uniform_(-max_target_ang_speed, max_target_ang_speed)
        self.last_target_vel[env_ids] = self.target_vel[env_ids].clone()

        # Reset robots
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            robot.reset(env_ids)
            
            self._actions[agent_id][env_ids] = 0.0
            self._last_actions[agent_id][env_ids] = 0.0
            
            self.zoom_level[agent_id][env_ids] = torch.zeros_like(self.zoom_level[agent_id][env_ids]).uniform_(1.0, max(self.cfg.max_zoom_level / 2.0, 1.0))
            
            default_root_state = robot.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:2] = formation_center[:, 0:2] + gap_ref[:, 0:2] * (idx + 1) * (-1)**(idx + gap_ref[:, 0:2].int())
            default_root_state[:, 2] = formation_center[:, 2] + gap_ref[:, 2] * (-1)**(idx + gap_ref[:, 2].int())
            default_root_state[:, :3] = quat_apply(formation_rotation, default_root_state[:, :3])
            default_root_state[:, :3] += self._env_origins[env_ids]
            default_root_state[:, 3:7] = quat_from_euler_xyz(
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
            )

            joint_pos = robot.data.default_joint_pos[env_ids]
            gimbal_idx = self.gimbal_joint_idx[agent_id]
            gimbal_yaw, gimbal_pitch = self.point_to_region(
                default_root_state[:, :3],
                default_root_state[:, 3:7],
                target_pos + torch.zeros_like(target_pos).uniform_(-5.0 * self.progress_tracking, 5.0 * self.progress_tracking)
            )
            gimbal_yaw = self._stabilizers[agent_id].wrap_to_pi(gimbal_yaw)
            gimbal_pitch = self._stabilizers[agent_id].wrap_to_pi(gimbal_pitch)
            joint_pos[:, gimbal_idx["yaw"]] = torch.clamp(gimbal_yaw, min=self.cfg.max_gimbal_yaw_angle[0], max=self.cfg.max_gimbal_yaw_angle[1])
            joint_pos[:, gimbal_idx["pitch"]] = torch.clamp(gimbal_pitch, min=self.cfg.max_gimbal_pitch_angle[0], max=self.cfg.max_gimbal_pitch_angle[1])
            joint_pos[:, gimbal_idx["roll"]] = torch.zeros_like(joint_pos[:, gimbal_idx["roll"]])
            self.gimbal_dof_targets[agent_id][env_ids] = joint_pos

            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            
            self._stabilizers[agent_id].reset(env_ids)

        if self.cfg.enable_noise_in_observations:
            self.sampled_noise_K.normal_(mean=0.0, std=self.cfg.intrinsic_std * self.progress_delay, generator=self.sampled_noise_generator)
            self.sampled_noise_control_gains.uniform_(0.5 * self.progress_dynamics, 2.0 * self.progress_dynamics, generator=self.sampled_noise_generator)
            dynamics_time_constants = torch.zeros(self.num_envs, self._robot_dynamics[agent_id].state_dim, device=self.device).uniform_(
                0.0, self.cfg.dynamics_time_constant * self.progress_dynamics,
                generator=self.sampled_noise_generator
            )
            self._robot_dynamics[agent_id].update_time_constants(dynamics_time_constants, env_ids)

        # Reset Delay and Communication System
        if self.cfg.enable_delay_system and self.delay_manager is not None:
            self.delay_manager.reset(env_ids)
            self.time_since_last_detection[env_ids] = 0.0
            self.last_detection_time[env_ids] = 0.0
            
            # Reset filter states to current true values
            for agent_idx, agent_id in enumerate(self.cfg.possible_agents):
                robot = self._robots[agent_id]
                gimbal_idx = self.gimbal_joint_idx[agent_id]
                self._robot_dynamics[agent_id].filtered_state[env_ids] = self._robots[agent_id].data.root_state_w[env_ids, 7:].clone()
                
                self.delay_manager.position_filters[agent_idx].filtered_state[env_ids] = \
                    robot.data.root_pos_w[env_ids].clone()
                self.delay_manager.orientation_filters[agent_idx].filtered_quat[env_ids] = \
                    robot.data.root_state_w[env_ids, 3:7].clone()
                
                self.delay_manager.gimbal_yaw_filters[agent_idx].filtered_state[env_ids] = \
                    robot.data.joint_pos[env_ids, gimbal_idx["yaw"]].unsqueeze(-1).clone()
                self.delay_manager.gimbal_pitch_filters[agent_idx].filtered_state[env_ids] = \
                    robot.data.joint_pos[env_ids, gimbal_idx["pitch"]].unsqueeze(-1).clone()
        
        super()._reset_idx(env_ids)

    def point_to_region(self, drone_pos_w, drone_quat_w, target_pos):
        """Calculate gimbal angles to point at target."""
        target_vector_w = target_pos - drone_pos_w
        drone_quat_inv = quat_inv(drone_quat_w)
        target_vector_b = quat_rotate(drone_quat_inv, target_vector_w)
        
        gimbal_yaw = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])
        horizontal_distance = torch.sqrt(target_vector_b[:, 0]**2 + target_vector_b[:, 1]**2)
        gimbal_pitch = torch.atan2(-target_vector_b[:, 2], horizontal_distance)
        
        return gimbal_yaw, gimbal_pitch

    def compute_ttc_from_pos_vel(self, ego_pos, ego_vel, other_pos, other_vel, safe_distance: float | None = None) -> torch.Tensor:
        """Compute Time-To-Collision (TTC)."""
        rel_pos = other_pos - ego_pos
        rel_vel = other_vel - ego_vel
        
        distance = torch.norm(rel_pos, dim=1)
        rel_speed = torch.norm(rel_vel, dim=1)
        
        rel_pos_normalized = rel_pos / (distance.unsqueeze(-1) + 1e-8)
        closing_speed = -torch.sum(rel_vel * rel_pos_normalized, dim=1)
        
        ttc = torch.where(
            closing_speed > 0,
            (distance - (safe_distance if safe_distance is not None else 0)) / closing_speed,
            torch.full_like(distance, float('inf'))
        )
        
        return ttc

    def compute_looming_ttc_penalty_zoom_invariant(
        self,
        bbox_w_dn, bbox_h_dn, valid_dn,          # [N, A], delayed+noisy (causal)
        fx_delayed, fy_delayed,                  # [N, A], delayed intrinsics in pixels (causal, noiseless)
        dt,                                      # scalar
        horizon_sec=6.0,
        ema_alpha_s=0.6,                         # EMA for log-size
        ema_alpha_f=0.6,                         # EMA for log-focal
        deriv_clip=0.5,
        eps=1e-6,
        stale_half_life=1.0,
        zoom_gate_k=0.2,                         # gate aggressiveness for |d ln f|
        aggregate="max",
    ):
        N, A = bbox_w_dn.shape
        self._ensure_loom_buffers(N, A, bbox_w_dn.device)

        # ---- 1) Build size and focal terms ----
        s = torch.sqrt(torch.clamp(bbox_w_dn, min=0) * torch.clamp(bbox_h_dn, min=0))  # [N, A]
        f_eff = torch.sqrt(torch.clamp(fx_delayed, min=eps) * torch.clamp(fy_delayed, min=eps))
        valid = valid_dn & (s > 0)

        g_s  = torch.log(s + eps)        # log size
        g_f  = torch.log(f_eff + eps)    # log focal

        # Keep separate EMAs for size and focal
        if not hasattr(self, "ttc_loom_logsize_ema"):
            self.ttc_loom_logsize_ema = torch.zeros_like(g_s)
            self.ttc_loom_logf_ema    = torch.zeros_like(g_f)
            self.ttc_loom_log_g_prev  = torch.zeros_like(g_s)   # previous g = log(s/f)

        # EMA updates only where valid
        size_ema_new = ema_alpha_s * self.ttc_loom_logsize_ema + (1.0 - ema_alpha_s) * g_s
        focal_ema_new= ema_alpha_f * self.ttc_loom_logf_ema    + (1.0 - ema_alpha_f) * g_f
        self.ttc_loom_logsize_ema = torch.where(valid, size_ema_new, self.ttc_loom_logsize_ema)
        self.ttc_loom_logf_ema    = torch.where(valid, focal_ema_new, self.ttc_loom_logf_ema)

        # ---- 2) Zoom-invariant log-size: g = log(s/f) ----
        g_now = self.ttc_loom_logsize_ema - self.ttc_loom_logf_ema   # [N, A]

        # ---- 3) Derivative using previous g (CRITICAL) ----
        dg = (g_now - self.ttc_loom_log_g_prev) / max(dt, 1e-6)
        self.ttc_loom_log_g_prev = g_now.clone()

        # Robustify derivative
        dg = torch.clamp(dg, min=-deriv_clip, max=deriv_clip)

        # ---- 4) Optional gates ----
        # Zoom motion gate: large |d ln f| means active zoom; soften penalty
        dlogf = (self.ttc_loom_logf_ema - focal_ema_new).abs() / max(dt, 1e-6)  # use pre-update vs new for smoother gate
        w_zoom = torch.exp(-dlogf / max(zoom_gate_k, 1e-6))
        w_zoom = torch.clamp(w_zoom, 0.0, 1.0)

        # Aspect gate (optional)
        aspect = torch.log( (bbox_w_dn + eps) / (bbox_h_dn + eps) )

        # ---- 5) Looming TTC from zoom-invariant derivative ----
        tau = torch.full_like(dg, float("inf"))
        approaching = (dg < -1e-4) & valid
        tau = torch.where(approaching, -1.0 / torch.clamp(dg, max=-1e-4), tau)

        # ---- 6) Shape to [0,1] and apply staleness + gates ----
        H = float(horizon_sec)
        phi = (H - torch.clamp(tau, max=H)) / H
        phi = torch.clamp(phi, 0.0, 1.0)

        # staleness
        if not hasattr(self, "ttc_loom_time_since_valid"):
            self.ttc_loom_time_since_valid = torch.zeros_like(phi)
        self.ttc_loom_time_since_valid = torch.where(valid,
                                                torch.zeros_like(self.ttc_loom_time_since_valid),
                                                self.ttc_loom_time_since_valid + dt)
        stale_decay = torch.exp(-self.ttc_loom_time_since_valid / max(stale_half_life, 1e-6))

        # combine gates
        gate = stale_decay * w_zoom
        phi = phi * gate

        # ---- 7) Aggregate ----
        if aggregate == "max":
            team_phi = torch.max(phi, dim=1).values
        elif aggregate == "mean":
            denom = torch.clamp(valid.float().sum(dim=1), min=1.0)
            team_phi = (phi * valid.float()).sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown aggregate={aggregate}")

        return team_phi, phi, tau

    @torch.no_grad()
    def _ensure_loom_buffers(self, N: int, A: int, device):
        # Zoom-invariant looming TTC buffers
        if not hasattr(self, "ttc_loom_logsize_ema"):
            self.ttc_loom_logsize_ema   = torch.zeros((N, A), device=device)
        if not hasattr(self, "ttc_loom_logf_ema"):
            self.ttc_loom_logf_ema      = torch.zeros((N, A), device=device)
        if not hasattr(self, "ttc_loom_log_g_prev"):
            self.ttc_loom_log_g_prev    = torch.zeros((N, A), device=device)
        if not hasattr(self, "ttc_loom_time_since_valid"):
            self.ttc_loom_time_since_valid = torch.zeros((N, A), device=device)

    def combine_ttc_penalties(self, phi_camcam, phi_loom, temp=0.25):
        t = max(float(temp), 1e-6)
        x = torch.stack([phi_camcam, phi_loom], dim=-1)  # [N, C-1 + 1] (should have been C-1+T)
        m = torch.amax(x / t, dim=-1, keepdim=True)
        y = m + t * torch.log(torch.clamp(torch.sum(torch.exp((x / t) - m), dim=-1, keepdim=True), min=1e-20))
        return y.squeeze(-1)  # [N, A]

    @torch.no_grad()
    def reset_ttc_buffers(
        self,
        reset_env_ids: torch.LongTensor,
        bbox_w_dn: torch.Tensor,             # [N, A] delayed+noisy
        bbox_h_dn: torch.Tensor,             # [N, A] delayed+noisy
        valid_dn: torch.Tensor,              # [N, A] bool, delayed+noisy
        fx_delayed: torch.Tensor,            # [N, A] delayed intrinsics (pixels)
        fy_delayed: torch.Tensor,            # [N, A] delayed intrinsics (pixels)
        eps: float = 1e-6,
    ):
        """
        Reset zoom-invariant looming TTC buffers for the specified environments.
        Call this on both full and partial resets right after observations are refreshed.

        Args:
            reset_env_ids: 1D LongTensor of env indices to reset (can be empty)
            bbox_w_dn, bbox_h_dn, valid_dn: delayed+noisy bboxes and validity mask
            fx_delayed, fy_delayed: delayed intrinsics (per agent), in pixels
        """
        if reset_env_ids is None or reset_env_ids.numel() == 0:
            return

        N, A = bbox_w_dn.shape
        device = bbox_w_dn.device
        self._ensure_loom_buffers(N, A, device)

        ridx = reset_env_ids

        # Current scale and focal (for those envs)
        w = torch.clamp(bbox_w_dn[ridx], min=0.0)                  # [R, A]
        h = torch.clamp(bbox_h_dn[ridx], min=0.0)                  # [R, A]
        s = torch.sqrt(w * h)                                      # [R, A]
        f_eff = torch.sqrt(torch.clamp(fx_delayed[ridx], min=eps) *
                        torch.clamp(fy_delayed[ridx], min=eps)) # [R, A]

        valid = (valid_dn[ridx]) & (s > 0.0)

        # Log terms
        g_s   = torch.log(s + eps)                                 # log size
        g_f   = torch.log(f_eff + eps)                             # log focal
        g_now = g_s - g_f                                          # log(s / f_eff)

        # Aspect (for optional gating elsewhere)
        aspect_log = torch.log((w + eps) / (h + eps))              # log(w/h)

        # Initialize EMAs/prev only where valid; zero where invalid

        self.ttc_loom_logsize_ema[ridx] = torch.where(valid, g_s, torch.zeros_like(g_s))
        self.ttc_loom_logf_ema[ridx]    = torch.where(valid, g_f, torch.zeros_like(g_f))
        self.ttc_loom_log_g_prev[ridx]  = torch.where(valid, g_now, torch.zeros_like(g_now))
        self.ttc_loom_time_since_valid[ridx] = torch.zeros_like(self.ttc_loom_time_since_valid[ridx])

    def _update_target_motion(self, mode: str = "default", env_ids: torch.Tensor | None = None):
        """Update target motion."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.target_vel_change_timer += self.step_dt
        change_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.target_direction_change_prob
        
        if mode == "linear" or mode == "default":
            new_desired_vel = torch.zeros_like(self.target_desired_vel)
            new_desired_vel[:, :3] = torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0
            new_desired_vel[:, :3] = new_desired_vel[:, :3] / (torch.norm(new_desired_vel[:, :3], dim=-1, keepdim=True) + 1e-8)
            new_desired_vel[:, :3] *= torch.rand(self.num_envs, 1, device=self.device) * self.cfg.max_target_speed
            new_desired_vel[:, 3:5] = 0.0
            new_desired_vel[:, 5] = torch.rand(self.num_envs, device=self.device) * self.cfg.max_target_speed * 0.1
            
            self.target_desired_vel[env_ids] = torch.where(
                change_mask.unsqueeze(-1).expand(-1, 6)[env_ids],
                new_desired_vel[env_ids],
                self.target_desired_vel[env_ids]
            )
            
            self.target_vel_change_timer = torch.where(
                change_mask, 
                torch.zeros_like(self.target_vel_change_timer), 
                self.target_vel_change_timer
            )
            
            vel_error = self.target_desired_vel[env_ids] - self.target_vel[env_ids]
            self.target_acceleration[env_ids] = vel_error * self.cfg.target_acceleration_scale
            
            self.target_acceleration[env_ids] = torch.clamp(
                self.target_acceleration[env_ids], 
                -self.cfg.target_max_acceleration * self.progress_move, 
                self.cfg.target_max_acceleration * self.progress_move
            )
        
        if mode == "circular" or mode == "default":
            t = self.common_step_counter * self.step_dt
            self.target_motion_radius = torch.where(
                change_mask,
                (torch.rand_like(self.target_motion_radius) * 30.0 + 40.0) * self.progress_move + 3.0,
                self.target_motion_radius
            )
            radius = self.target_motion_radius
            
            self.target_motion_yaw_vel = torch.where(
                change_mask,
                (torch.rand_like(self.target_motion_yaw_vel) * self.cfg.max_target_ang_speed / radius) * self.progress_move,
                self.target_motion_yaw_vel
            )
            angular_speed = self.target_motion_yaw_vel
            
            center = self._env_origins
            desired_x = center[:, 0] + radius * torch.cos(angular_speed * t)
            desired_y = center[:, 1] + radius * torch.sin(angular_speed * t)
            desired_z = center[:, 2] + (torch.rand(self.num_envs, device=self.device) * 10.0 + 30.0) * self.progress_move

            desired_pos = torch.stack([desired_x, desired_y, desired_z], dim=1)
            pos_error = desired_pos - self.target.data.root_state_w[:, :3]
            
            kp = 1.0
            self.target_desired_vel[env_ids, :2] = kp * pos_error[env_ids, :2]
            self.target_desired_vel[env_ids, 2] = torch.where(
                change_mask[env_ids],
                kp * pos_error[env_ids, 2],
                self.target_desired_vel[env_ids, 2]
            )
            
            speed = torch.norm(self.target_desired_vel[env_ids, :3], dim=1, keepdim=True)
            speed_clamped = torch.clamp(speed, max=self.cfg.max_target_speed * self.progress_move)
            self.target_desired_vel[env_ids, :3] = self.target_desired_vel[env_ids, :3] / (speed + 1e-6) * speed_clamped # Keep the linear speed norm within limits
            self.target_desired_vel[env_ids, 3:] = 0.0 # No flipping in pitch and roll
            
            vel_error = self.target_desired_vel[env_ids] - self.target_vel[env_ids]
            self.target_acceleration[env_ids] = vel_error * self.cfg.target_acceleration_scale

            self.target_acceleration[env_ids] = torch.clamp(
                self.target_acceleration[env_ids], 
                -self.cfg.target_max_acceleration * self.progress_move, 
                self.cfg.target_max_acceleration * self.progress_move
            )
        
        self.target_vel += self.target_acceleration * self.step_dt
        self.target_vel *= self.cfg.target_velocity_damping
        
        self.target_vel[:, :3] = torch.clamp(
            self.target_vel[:, :3],
            -self.cfg.max_target_speed * self.progress_move,
            self.cfg.max_target_speed * self.progress_move
        )
        # Check for NaN values in target velocity
        if torch.isnan(self.target_vel).any():
            nan_envs = torch.nonzero(torch.isnan(self.target_vel).any(dim=1)).flatten()
            raise ValueError(f"NaN detected in target velocity at environments: {nan_envs.tolist()}")

    def _update_research_metrics(self):
        """Update research metrics."""
        p0 = self._robots["drone_0"].data.root_pos_w
        p1 = self._robots["drone_1"].data.root_pos_w
        pt = self.target.data.root_pos_w

        d01 = torch.norm(p0 - p1, dim=1)

        v0 = pt - p0
        v1 = pt - p1
        cosb = torch.sum(v0 * v1, dim=1) / (torch.norm(v0, dim=1) * torch.norm(v1, dim=1) + 1e-8)
        cosb = torch.clamp(cosb, -1.0, 1.0)
        beta = torch.acos(cosb)
        beta_deg = torch.rad2deg(beta)
        w_geom = torch.sin(beta) ** 2

        def bbox_area_frac(b, cam_cfg):
            area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
            return area / (cam_cfg.width * cam_cfg.height)

        a0 = bbox_area_frac(self.bboxes["drone_0"], self.agent_camera_cfgs["drone_0"])
        a1 = bbox_area_frac(self.bboxes["drone_1"], self.agent_camera_cfgs["drone_1"])
        valid0 = self.bbox_valid_mask["drone_0"][:, 0]
        valid1 = self.bbox_valid_mask["drone_1"][:, 0]
        both_valid = valid0 & valid1

        a = torch.minimum(a0, a1)
        mu = self.cfg.research.tqi_opt_area
        sig = self.cfg.research.tqi_area_sigma
        w_scale = torch.exp(-((a - mu) ** 2) / (sig ** 2 + 1e-12))
        tqi = both_valid.float() * w_geom * w_scale

        self._research["cvtt_sec"] += (tqi >= self.cfg.research.tqi_threshold).float() * self.step_dt
        self._research["tqi_sum"] += tqi
        self._research["steps"] += 1

        f0_m = (self.base_focal_length["drone_0"] * self.zoom_level["drone_0"]) * 1e-3
        f1_m = (self.base_focal_length["drone_1"] * self.zoom_level["drone_1"]) * 1e-3
        pitch = self.cfg.research.pixel_pitch_m
        sig_px = self.cfg.research.det_std_px
        sig_th0 = sig_px * pitch / (f0_m + 1e-12)
        sig_th1 = sig_px * pitch / (f1_m + 1e-12)

        d0 = torch.norm(v0, dim=1) + 1e-6
        d1 = torch.norm(v1, dim=1) + 1e-6
        fim_scalar = w_geom * (1.0 / (d0 * d0 * sig_th0 * sig_th0) + 1.0 / (d1 * d1 * sig_th1 * sig_th1))
        ig_approx = torch.log(fim_scalar + 1e-12)

        if not self.cfg.research.enable_step_logging:
            return
        if (self._research["last_step_logged"] == -1) or \
           ((self.common_step_counter - self._research["last_step_logged"]) >= self.cfg.research.log_interval_steps):

            ids = self._research_sample_ids
            def mean_std(x):
                return torch.mean(x).item(), torch.std(x).item()

            m_d, s_d = mean_std(d01)
            m_b, s_b = mean_std(beta_deg)
            m_tqi, s_tqi = mean_std(tqi)
            m_ig, s_ig = mean_std(ig_approx)
            valid_rate0 = valid0.float().mean().item()
            valid_rate1 = valid1.float().mean().item()
            both_valid_rate = both_valid.float()[ids].mean().item()

            p50_ang = torch.quantile(beta_deg[ids], 0.5).item()
            p90_ang = torch.quantile(beta_deg[ids], 0.9).item()

            if "step" not in self.extras:
                self.extras["step"] = {}
            self.extras["step"].update({
                "Step/baseline_m_mean": m_d,
                "Step/baseline_m_std": s_d,
                "Step/angle_deg_mean": m_b,
                "Step/angle_deg_std": s_b,
                "Step/angle_deg_p50": p50_ang,
                "Step/angle_deg_p90": p90_ang,
                "Step/tqi_mean": m_tqi,
                "Step/tqi_std": s_tqi,
                "Step/ig_approx_mean": m_ig,
                "Step/ig_approx_std": s_ig,
                "Step/valid_rate_d0": valid_rate0,
                "Step/valid_rate_d1": valid_rate1,
                "Step/both_valid_rate_sample": both_valid_rate,
                "Step/curriculum_alpha": float(
                    max(0.0, min(1.0, (self.common_step_counter - self.cfg.curriculum_coordination_start_step) /
                        (self.cfg.curriculum_coordination_end_step - self.cfg.curriculum_coordination_start_step + 1e-6))))
            })
            
            if self.cfg.enable_delay_system and self.delay_manager is not None:
                delay_stats = self.delay_manager.get_statistics()
                self.extras["step"].update({
                    "Delay/comm_buffer_mean": delay_stats['comm_buffer_sizes'].float().mean().item(),
                    "Delay/comm_buffer_max": delay_stats['comm_buffer_sizes'].max().item(),
                    "Delay/det_buffer_mean": sum(
                        delay_stats['detection_buffer_sizes'][i].float().mean().item() 
                        for i in range(len(self.cfg.possible_agents))
                    ) / len(self.cfg.possible_agents),
                })
            
            self._research["last_step_logged"] = self.common_step_counter

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""

        """Apply camera intrinsics based on zoom level."""
        if DEBUG_DRAW:
            for i, agent_id in enumerate(self.cfg.possible_agents):
                current_focal_length = self.base_focal_length[agent_id] * self.zoom_level[agent_id]
                self._cameras[agent_id].set_intrinsic_matrices_batched(
                    self.camera_intrinsics[agent_id], 
                    current_focal_length
                )
                """
                @file: source/isaaclab/isaaclab/sensors/camera/camera.py (modified)
                ️@description: Added method to set different focal lengths for envs in batch.
                def set_intrinsic_matrices_batched(
                    self, matrices: torch.Tensor, focal_length: torch.tensor, env_ids: Sequence[int] | None = None
                ):
                    # resolve env_ids
                    if env_ids is None:
                        env_ids = self._ALL_INDICES
                    # convert matrices to numpy tensors
                    if isinstance(matrices, torch.Tensor):
                        matrices = matrices.cpu().numpy()
                    else:
                        matrices = np.asarray(matrices, dtype=float)
                    # iterate over env_ids
                    for i, intrinsic_matrix in zip(env_ids, matrices):
                        # extract parameters from matrix
                        f_x = intrinsic_matrix[0, 0]
                        c_x = intrinsic_matrix[0, 2]
                        f_y = intrinsic_matrix[1, 1]
                        c_y = intrinsic_matrix[1, 2]
                        # get viewport parameters
                        height, width = self.image_shape
                        height, width = float(height), float(width)
                        # resolve parameters for usd camera
                        params = {
                            "focal_length": focal_length[i].item(),
                            "horizontal_aperture": width * focal_length[i].item() / f_x,
                            "vertical_aperture": height * focal_length[i].item() / f_y,
                            "horizontal_aperture_offset": (c_x - width / 2) / f_x,
                            "vertical_aperture_offset": (c_y - height / 2) / f_y,
                        }

                        # TODO: Adjust to handle aperture offsets once supported by omniverse
                        #   Internal ticket from rendering team: OM-42611
                        if params["horizontal_aperture_offset"] > 1e-4 or params["vertical_aperture_offset"] > 1e-4:
                            omni.log.warn("Camera aperture offsets are not supported by Omniverse. These parameters are ignored.")

                        # change data for corresponding camera index
                        sensor_prim = self._sensor_prims[i]
                        # set parameters for camera
                        for param_name, param_value in params.items():
                            # convert to camel case (CC)
                            param_name = to_camel_case(param_name, to="CC")
                            # get attribute from the class
                            param_attr = getattr(sensor_prim, f"Get{param_name}Attr")
                            # set value
                            # note: We have to do it this way because the camera might be on a different
                            #   layer (default cameras are on session layer), and this is the simplest
                            #   way to set the property on the right layer.
                            omni.usd.set_prop_val(param_attr(), param_value)
                    # update the internal buffers
                    self._update_intrinsic_matrices(env_ids)
                """

        if DEBUG_DRAW and hasattr(self, 'draw_interface'):
            self.draw_interface.clear_lines()
        
            for agent_id in self.cfg.possible_agents:
                line_colors_yellow = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1) # (N, 4)
                line_colors_green = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                line_colors_red = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                
                is_valid = self.bbox_valid_mask[agent_id]
                agent_idx = self.cfg.possible_agents.index(agent_id)
                if self.bbox_raycaster.data.valid_bbox_visibility_mask is not None:
                    is_visible = self.bbox_raycaster.data.valid_bbox_visibility_mask[:, agent_idx, 0].unsqueeze(-1)
                else:
                    is_visible = torch.ones_like(is_valid) > 0

                # line_colors = torch.where(
                #     is_valid, 
                #     line_colors_green,
                #     torch.where(
                #         is_visible,
                #         line_colors_yellow,
                #         line_colors_red
                #     )
                # ).tolist()
                line_colors = torch.where(
                    is_valid, 
                    line_colors_green,
                    line_colors_yellow,
                ).tolist()
                # time_since_detection = self.states.time_since_detection_delayed_noisy[:, i, 0]  # (N,)
                # time_since_norm = torch.clamp(time_since_detection / self.cfg.max_time_since_detection, max=1.0).unsqueeze(-1)  # (N, 1)
                # freshness = time_since_norm
                # line_colors = (line_colors_red * (1.0 - freshness) +
                #                line_colors_green * freshness).tolist()
                # line_colors = torch.stack([
                #     1.0 - freshness,  # R: staler is redder
                #     freshness,        # G: fresher is greener
                #     torch.zeros_like(freshness),  # B
                #     torch.ones_like(freshness)    # A
                # ], dim=-1).tolist()
                
                line_thicknesses = [1.0] * self.camera_pos_world[agent_id].shape[0]
                self.draw_interface.draw_lines(
                    self.camera_pos_world[agent_id].tolist(), 
                    self.target.data.root_pos_w.tolist(), 
                    line_colors, 
                    line_thicknesses
                )
                
                if hasattr(self, 'camera_frustrum'):
                    self.camera_frustrum.draw_frustrum(
                        camera_position=self.camera_pos_world[agent_id],
                        camera_orientation=self.camera_quat_world[agent_id],
                        camera_intrinsics=self.camera_cfg_batch[agent_id],
                        zoom_level=self.zoom_level[agent_id],
                        device=self.device
                    )

            # Visualize triangulation covariance ellipsoids
            if self.cfg.enable_delay_system and self.states is not None:
                ray_dir_for_vis = self.states.ray_dir_delayed
            else:
                ray_dir_for_vis = get_ray_dir_from_bbox(
                    self.bbox_raycaster.data.bboxes,
                    torch.stack([self.camera_intrinsics[agent_id] for agent_id in self.cfg.possible_agents], dim=1),
                    torch.stack([self.camera_quat_world[agent_id] for agent_id in self.cfg.possible_agents], dim=1),
                    torch.stack([self.camera_pos_world[agent_id] for agent_id in self.cfg.possible_agents], dim=1),
                )
            
            # Use delayed covariance for visualization
            Sigma_X_vis = self.states.Sigma_X_delayed_noisy
            Sigma_diag = torch.diagonal(Sigma_X_vis, dim1=-2, dim2=-1)
            Sigma_diag_sqrt = torch.sqrt(torch.clamp(Sigma_diag, min=0.0) + 1e-12)
            if Sigma_diag_sqrt.ndim == 3 and Sigma_diag_sqrt.shape[1] >= 1:
                scales = Sigma_diag_sqrt[:, 0, :]
            else:
                scales = Sigma_diag_sqrt.squeeze(1)

            self.tri_cov_visualizer.visualize(
                translations=self.X_w_est.squeeze(1),
                scales=scales * 2.5,
            )
            
            self.bbox_raycaster.visualize()