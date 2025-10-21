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

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
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
    quat_from_euler_xyz,
    quat_mul,
    quat_inv,
    quat_rotate,
)

# Pre-defined configs
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import IRIS_GIMBAL2_CFG, IRIS_BODY_CFG
from isaaclab.markers import CUBOID_MARKER_CFG, FRAME_MARKER_CFG

# from point_mass import PointMass
# from bbox_generator import BBoxGenerator
# from camera_frustrum import CameraFrustrum, create_camera_cfg_tensor, project_2d_to_3d
from .point_mass import PointMass
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

# import bbox_raycaster
from . import bbox_raycaster

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
    log_interval_steps: int = 1              # throttle step logs
    sample_envs: int = 64                     # subsample envs for percentiles
    tqi_opt_area: float = 0.05                # bbox area fraction (5%) is “ideal” size in image
    tqi_area_sigma: float = 0.02              # width of Gaussian around the ideal area
    tqi_threshold: float = 0.20               # threshold for CVTT accumulation
    beta_min_deg: float = 30.0                # "good" triangulation angle window (for ref only)
    beta_max_deg: float = 150.0
    pixel_pitch_m: float = 3.45e-6            # sensor pixel pitch (user-tunable)
    det_std_px: float = 0.5                   # detector 1-sigma in pixels (user-tunable)
    enable_step_logging: bool = DEBUG_DRAW          # allow disabling for speed

@configclass
class IrisMAEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 30.0
    decimation = 2
    
    # Define agents - The environment is now built from this list
    possible_agents = ["drone_0", "drone_1"]
    
    # Define spaces for each agent
    action_spaces = {
        "drone_0": 7,  # [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        "drone_1": 7,
    }
    observation_spaces = {
        "drone_0": 26,
        "drone_1": 26,
    }
    state_space = -1  # Concatenate all observations
    
    debug_vis = True
    ui_window_class_type = IrisMAEnvWindow

    # simulation
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
            rot=(0.5, -0.5, 0.5, -0.5), 
            convention="ros"
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

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=25.0, replicate_physics=True)
    
    # Robot configuration template (dynamically applied to each agent)
    robot: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/{robot_name}")
    
    research: ResearchLoggingCfg = ResearchLoggingCfg()

    bbox_raycaster: bbox_raycaster.BBoxRayCasterCfg = bbox_raycaster.BBoxRayCasterCfg(
        target_prim_paths=[], # Will be set dynamically in _setup_scene
        mesh_prim_paths=["/World/ground"],
        
        # Camera configuration
        num_cameras_per_env=2,
        num_cameras_per_agent=1,
        
        load_agent_meshes=False,
        agent_mesh_simplification=1.1,  # Use 70% of triangles for balance
        use_collision_proxy=False,      # Use full meshes, not boxes
        
        # Validation thresholds
        min_bbox_size=(0.02, 0.02),
        max_bbox_size=(0.90, 0.90),
        partial_detection_allowed=False,
        min_bbox_area_pixels=400.0,
        
        # Occlusion detection
        enable_occlusion_check=False,
        occlusion_ray_pattern="9point",  # Start simple, can upgrade to "9point"
        occlusion_visibility_threshold=0.5,
        occlusion_ray_tolerance=1.1,
        
        # Performance
        max_distance=100.0,
        
        # Debug
        debug_vis=True,
        debug_memory=False,
    )

    # Control parameters
    thrust_to_weight = 4.0
    moment_scale = 10.0
    yaw_moment_scale = 1.0
    
    # Motion limits
    max_target_speed = 0.0
    max_lin_vel = 10.0
    max_yaw_rate = 20.0
    max_lin_acc = 3.0
    max_ang_acc = 30.0
    
    # Target movement parameters
    target_acceleration_scale = 2.0
    target_velocity_damping = 0.95
    target_direction_change_prob = 0.01
    target_max_acceleration = 1.0
    
    # Reward scales (per agent)
    lin_vel_penalty_scale = -0.1
    ang_vel_penalty_scale = lin_vel_penalty_scale * 0.2
    action_sum_penalty_scale = -1.0
    action_weight = [1, 1, 5, 0.5, 0.03, 0.03, 0.01]
    action_delta_penalty_scale = -0.01
    action_delta_weight = [1, 1, 5, 0.5, 0.03, 0.03, 0.01]
    zoom_reward_scale = -0.5
    
    # Single-agent tracking rewards
    bbox_center_reward_scale = 60
    bbox_size_reward_scale = 60
    
    # Multi-agent coordination rewards
    triangulation_reward_scale = 5  # Reward for good triangulation geometry
    collision_penalty_scale = -100   # Penalty for getting too close to each other
    min_safe_distance = 10.0

    # Triangulation parameters
    pix_std = 7.0  # pixel standard deviation
    pos_std = 0.1  # position uncertainty
    ori_std = 0.02  # rads orientation uncertainty
    gimbal_std = 0.02  # rads gimbal uncertainty
    intrinsic_std = 10.0  # pixel uncertainty in focal length and principal point

    # Step at which to start and end introducing coordination rewards
    curriculum_coordination_start_step: int = 30000
    curriculum_coordination_end_step: int = 40000
    curriculum_coordination_duration_steps: int = 5000
    curriculum_tracking_start_step: int = -1
    curriculum_tracking_end_step: int = 10000

class IrisMAEnv(DirectMARLEnv):
    cfg: IrisMAEnvCfg

    def __init__(self, cfg: IrisMAEnvCfg, render_mode: str | None = None, **kwargs):
        # -- Dynamically generate agent-specific robot and camera configs --
        # This must be done BEFORE super().__init__ because _setup_scene is called from there.
        self.agent_robot_cfgs = {}
        self.agent_camera_cfgs = {}
        for agent_id in cfg.possible_agents:
            robot_index = agent_id.split("_")[-1]
            robot_name = f"Robot_{robot_index}"

            # Create and store robot config
            robot_cfg = copy.deepcopy(cfg.robot)
            robot_cfg.prim_path = robot_cfg.prim_path.format(robot_name=robot_name)
            self.agent_robot_cfgs[agent_id] = robot_cfg
            
            # Create and store camera config
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
        self.base_target_scale = self.cfg.target_cfg.spawn.scale[0] # 1 / 0.06 # Base scale for target size (6cm cube scaled up by to match 1m in env)
        self.target_scale = torch.ones(self.num_envs, num_targets_per_env, 3, device=self.device) * self.base_target_scale

        # Target movement
        self.target_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0)
        self.last_target_vel = self.target_vel.clone()
        self.target_acceleration = torch.zeros(self.num_envs, 6, device=self.device)
        self.target_desired_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0)
        self.target_vel_change_timer = torch.zeros(self.num_envs, device=self.device)
        
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
        self.bboxes_normalized = {
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
        self.camera_cfg_batch = {
            agent_id: create_camera_cfg_tensor(agent_cfg, self.num_envs, device=self.device)
            for agent_id, agent_cfg in self.agent_camera_cfgs.items()
        } # Shape (N, 6) per agent
        self.camera_intrinsics = {
            agent: bbox_raycaster.utils.create_intrinsic_matrix_tensor(self.camera_cfg_batch[agent])
            for agent in self.cfg.possible_agents
        } # Shape (N, 3, 3) per agent
        self.base_intrinsic_matrices = self.camera_intrinsics.copy()
        camera_offset_rot_single = torch.tensor(cfg.camera.offset.rot, device=self.device)
        self.camera_offset_rot_batch = camera_offset_rot_single.unsqueeze(0).expand(self.num_envs, -1)
        camera_offset_pos_single = torch.tensor(cfg.camera.offset.pos, device=self.device)
        self.camera_offset_pos_batch = camera_offset_pos_single.unsqueeze(0).expand(self.num_envs, -1)

        # Action weights
        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device).repeat(self.num_envs, 1)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device).repeat(self.num_envs, 1)
        
        # Get body and joint indices for each robot
        self._body_ids = {}
        self.gimbal_joint_idx = {}
        self._robot_mass = {}
        self._stabilizers = {}
        
        for agent in self.cfg.possible_agents:
            robot = self._robots[agent]
            self._body_ids[agent] = robot.find_bodies("body")[0]
            self.gimbal_joint_idx[agent] = {
                "yaw": robot.find_joints("yaw_joint")[0][0],
                "roll": robot.find_joints("roll_joint")[0][0],
                "pitch": robot.find_joints("pitch_joint")[0][0],
            }
            self._robot_mass[agent] = robot.root_physx_view.get_masses()[0].sum()
            robot_weight = (self._robot_mass[agent] * torch.tensor(self.sim.cfg.gravity, device=self.device).norm()).item()
            self._stabilizers[agent] = PointMass(0, robot_weight, self.num_envs, self.device)
        

        # Prepare data in [N, C, ...] format where N=num_envs, C=num_cameras
        self.robot_positions_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self.robot_quats_stacked = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 4, device=self.device
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
        # Initialize with default values
        for agent_idx, agent_id in enumerate(self.cfg.possible_agents):
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            self.robot_quats_stacked[:, agent_idx, :] = default_quat.unsqueeze(0).expand(self.num_envs, -1)
            self.camera_intrinsics_stacked[:, agent_idx, :, :] = self.camera_intrinsics[agent_id]

        # Define uncertainty covariances
        # Target position uncertainty [N, T, 3, 3]
        self.Sigma_X_invalid = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, 1, 3, 3
        ) * (-1.0)
        self.Sigma_X = self.Sigma_X_invalid.clone()
        # trace_cov: [N, T]
        self.trace_cov_invalid = torch.ones(self.num_envs, 1, device=self.device) * (-1.0)
        self.trace_cov = self.trace_cov_invalid.clone()
        # Tri cov computation validity mask: [N, T]
        self.is_tri_cov_valid = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

        # Pixel noise [N, C, 2, 2]
        self.Sigma_pix = torch.eye(2, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.cfg.possible_agents), 2, 2
        ) * (self.cfg.pix_std ** 2)
        # Position uncertainty (e.g., from GPS/state estimator) [N, C, 3, 3]
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
        # Intrinsic uncertainties [N, C, 4, 4] (fx, fy, cx, cy)
        self.Sigma_K = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.cfg.possible_agents), 4, 4
        ) * (self.cfg.intrinsic_std ** 2)

        self.baseline_distances = torch.ones(
            self.num_envs, len(self.cfg.possible_agents), len(self.cfg.possible_agents), 1, 
            device=self.device) * (-1.0)

        self._research = {
            "cvtt_sec": torch.zeros(self.num_envs, device=self.device),
            "tqi_sum": torch.zeros(self.num_envs, device=self.device),
            "steps": torch.zeros(self.num_envs, device=self.device),
            "last_step_logged": -1,
        }
        # choose a stable random subsample once (for percentiles / hist)
        perm = torch.randperm(self.num_envs, device=self.device)
        self._research_sample_ids = perm[:min(self.cfg.research.sample_envs, self.num_envs)]

        # Episode tracking
        self._episode_sums = {
            agent: {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in [
                    # "lin_vel", 
                    "action_sum", 
                    # "action_delta", 
                    "bbox_center", 
                    "bbox_size", 
                    # "zoom", 
                    "triangulation", 
                    # "collision",
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
        # Create robots for each agent from the dynamically generated configs
        self._robots = {}
        for agent_id, robot_cfg in self.agent_robot_cfgs.items():
            robot_cfg.spawn.semantic_tags = [("class", "robot")]
            self._robots[agent_id] = Articulation(robot_cfg)
        
        # Create cameras for each agent from the dynamically generated configs
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
        # This gives: ["/World/envs/env_0", "/World/envs/env_1", ...]
        # BBoxRayCaster will:
        #   1. Append "/target" to each path
        #   2. Extract bbox from all environments
        #   3. Automatically detect if all targets identical
        #   4. Optimize storage accordingly
        
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
        for agent_id, agent_actions in actions.items():
            if torch.isnan(agent_actions).any():
                continue
                
            robot = self._robots[agent_id]
            stabilizer = self._stabilizers[agent_id]
            
            # Clamp and store actions
            self._actions[agent_id] = agent_actions.clone().clamp(-1.0, 1.0)
            # self._actions[agent_id] = torch.zeros_like(self._actions[agent_id])
            
            # Compute command velocities
            self._cmd_vel[agent_id][:, 0, :3] = self._actions[agent_id][:, :3] * self.cfg.max_lin_vel
            self._cmd_vel[agent_id][:, 0, 5] = self._actions[agent_id][:, 3] * self.cfg.max_yaw_rate
            
            # Compute stabilizing control
            self._thrust[agent_id][:,0,:], self._moment[agent_id][:,0,:] = stabilizer.compute_control(
                self._cmd_vel[agent_id][:, 0, :3], 
                self._cmd_vel[agent_id][:, 0, 5],
                robot.data.root_state_w[:, 3:7],
                robot.data.root_lin_vel_w, 
                robot.data.root_ang_vel_b, 
                self.step_dt
            )
            
            # Update gimbal targets
            roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_state_w[:, 3:7])
            curr_quat_w_yaw = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)
            roll_b, pitch_b, _ = euler_xyz_from_quat(quat_mul(robot.data.root_state_w[:, 3:7], quat_inv(curr_quat_w_yaw)))
            
            gimbal_idx = self.gimbal_joint_idx[agent_id]
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["yaw"]] += self._actions[agent_id][:, 4] * self.step_dt * 2 * math.pi
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["yaw"]] = torch.clamp(
                self.gimbal_dof_targets[agent_id][:, gimbal_idx["yaw"]], -math.pi*2/3, math.pi*2/3
            )
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["roll"]] = self._stabilizers[agent_id].wrap_to_pi(-roll_b)
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["pitch"]] += self._actions[agent_id][:, 5] * self.step_dt * 2 * math.pi
            self.gimbal_dof_targets[agent_id][:, gimbal_idx["pitch"]] = torch.clamp(
                self.gimbal_dof_targets[agent_id][:, gimbal_idx["pitch"]], -math.pi*1/3, math.pi*1/3
            )
            # gimbal_yaw, gimbal_pitch = self.point_to_region(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], self.target.data.root_state_w[:, :3])

            # Update zoom level
            delta_zoom = self._actions[agent_id][:, 6] * self.step_dt * 5.0
            self.zoom_level[agent_id] += delta_zoom
            self.zoom_level[agent_id] = torch.clamp(self.zoom_level[agent_id], 1.0, 10.0)
            

        # Update target movement (shared)
        self._update_target_movement(mode="circular")
        
        # Bounce targets that are too low
        up_vel = self.target_vel.clone()
        up_vel[..., 2] = torch.where(self.target_vel[..., 2] > 0, self.target_vel[..., 2], -self.target_vel[..., 2])
        target_lower_bound = torch.ones_like(self.target.data.root_state_w, device=self.device) * 1.0
        self.target_vel[..., 2] = torch.where(
            self.target.data.root_state_w[:, 2] < target_lower_bound[:, 2], 
            up_vel[..., 2], 
            self.target_vel[..., 2]
        )
        
    def _apply_action(self):
        """Apply actions to all robots in the simulation."""
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            stabilizer = self._stabilizers[agent_id]
            
            # Compute new velocities
            new_vel_lin_w = robot.data.root_lin_vel_w * (1 - self.step_dt) + quat_rotate(
                robot.data.root_state_w[:, 3:7], self._thrust[agent_id][:, 0, :]) * self.step_dt
            new_vel_lin_w[:, :2] = torch.clamp(new_vel_lin_w[:, :2], min=-self.cfg.max_lin_vel, max=self.cfg.max_lin_vel)
            
            new_vel_ang_w = robot.data.root_ang_vel_w * (1 - 3 * self.step_dt) + quat_rotate(
                robot.data.root_state_w[:, 3:7], self._moment[agent_id][:, 0, :]) * self.step_dt
            new_vel_ang_w[:, 0] = torch.zeros_like(new_vel_ang_w[:, 0])
            new_vel_ang_w[:, 1] = torch.zeros_like(new_vel_ang_w[:, 1])
            new_vel_ang_w[:, 2] = torch.clamp(new_vel_ang_w[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
            
            # Write velocities and gimbal targets
            robot.write_root_velocity_to_sim(
                torch.cat((new_vel_lin_w, new_vel_ang_w), dim=1), 
                env_ids=robot._ALL_INDICES
            )
            robot.set_joint_position_target(self.gimbal_dof_targets[agent_id])
        
        # Apply target velocity
        self.target.write_root_com_velocity_to_sim(self.target_vel)
        
        

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents."""
        camera_poses = {}
        camera_intrinsics = {}
        image_shapes = {}
        agent_poses = {}

        # Compute agent states and camera parameters
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]

            # Update camera parameters for each agent
            current_focal_length = self.base_focal_length[agent_id] * self.zoom_level[agent_id]
            self.camera_cfg_batch[agent_id][:, 2] = current_focal_length

            # Compute camera pose for bounding box generation
            robot_pos = robot.data.root_pos_w
            robot_quat = robot.data.root_state_w[:, 3:7]
            agent_poses[agent_id] = (robot_pos, robot_quat)

            gimbal_idx = self.gimbal_joint_idx[agent_id]
            gimbal_yaw = robot.data.joint_pos[:, gimbal_idx["yaw"]]
            gimbal_roll = robot.data.joint_pos[:, gimbal_idx["roll"]]
            gimbal_pitch = robot.data.joint_pos[:, gimbal_idx["pitch"]]
            
            gimbal_quat = quat_mul(
                quat_mul(
                    quat_from_euler_xyz(torch.zeros_like(gimbal_yaw), torch.zeros_like(gimbal_yaw), gimbal_yaw),
                    quat_from_euler_xyz(torch.zeros_like(gimbal_pitch), gimbal_pitch, torch.zeros_like(gimbal_pitch))
                ),
                quat_from_euler_xyz(gimbal_roll, torch.zeros_like(gimbal_roll), torch.zeros_like(gimbal_roll))
            )
            
            self.camera_quat_world[agent_id] = quat_mul(robot_quat, quat_mul(gimbal_quat, self.camera_offset_rot_batch))
            camera_offset = torch.tensor((0, 0, 0.1), device=self.device).expand(self.num_envs, -1)
            self.camera_pos_world[agent_id] = robot_pos + quat_rotate(self.camera_quat_world[agent_id], camera_offset)
            
            agent_cam_cfg = self.agent_camera_cfgs[agent_id]
            # Generate bounding boxes
            # bbox_gen = BBoxGenerator(
            #     camera_width=agent_cam_cfg.width, 
            #     camera_height=agent_cam_cfg.height,
            #     focal_length=agent_cam_cfg.spawn.focal_length, 
            #     horizontal_aperture=agent_cam_cfg.spawn.horizontal_aperture
            # )
            
            # self.bboxes[agent_id], self.bbox_valid_mask[agent_id][:, 0] = bbox_gen.generate_2d_bbox(
            #     self.target.data.root_state_w[:, :3], 
            #     self.target.data.root_state_w[:, 3:7],
            #     self.camera_pos_world[agent_id],
            #     self.camera_quat_world[agent_id],
            #     min_bbox_size=int(0.01 * agent_cam_cfg.width),
            #     focal_length=current_focal_length
            # )
            
            # # Normalize bounding boxes
            # self.bboxes_normalized[agent_id] = self.bboxes[agent_id].clone()
            # valid = self.bbox_valid_mask[agent_id].squeeze(-1)
            # self.bboxes_normalized[agent_id][:, 0] = torch.where(valid, self.bboxes[agent_id][:, 0] / agent_cam_cfg.width, torch.ones_like(self.bboxes[agent_id][:, 0]) * (-1.0))
            # self.bboxes_normalized[agent_id][:, 1] = torch.where(valid, self.bboxes[agent_id][:, 1] / agent_cam_cfg.height, torch.ones_like(self.bboxes[agent_id][:, 1]) * (-1.0))
            # self.bboxes_normalized[agent_id][:, 2] = torch.where(valid, self.bboxes[agent_id][:, 2] / agent_cam_cfg.width, torch.ones_like(self.bboxes[agent_id][:, 2]) * (-1.0))
            # self.bboxes_normalized[agent_id][:, 3] = torch.where(valid, self.bboxes[agent_id][:, 3] / agent_cam_cfg.height, torch.ones_like(self.bboxes[agent_id][:, 3]) * (-1.0))
        
            camera_poses[agent_id] = (self.camera_pos_world[agent_id], self.camera_quat_world[agent_id])

            # Apply zoom to intrinsics
            # Apply zoom to intrinsics using the fixed base matrices
            self.camera_intrinsics[agent_id] = self.base_intrinsic_matrices[agent_id].clone()
            self.camera_intrinsics[agent_id][:, 0, 0] *= self.zoom_level[agent_id]
            self.camera_intrinsics[agent_id][:, 1, 1] *= self.zoom_level[agent_id]
            camera_intrinsics[agent_id] = self.camera_intrinsics[agent_id]

            # Image shapes
            image_shapes[agent_id] = (self.agent_camera_cfgs[agent_id].height, self.agent_camera_cfgs[agent_id].width)

        if DEBUG_DRAW:
            for i, agent_id in enumerate(self.cfg.possible_agents):
                current_focal_length = self.base_focal_length[agent_id] * self.zoom_level[agent_id]
                self._cameras[agent_id].set_intrinsic_matrices_batched(
                    self.camera_intrinsics[agent_id], 
                    current_focal_length
                )
                """
                @file: source/isaaclab/isaaclab/sensors/camera/camera.py (modified)
                ️@description: Added method to set intrinsic matrices in batch.
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
        # Get target poses with correct dimensions
        target_pos = self.target.data.root_pos_w  # May be (N, 3)
        target_quat = self.target.data.root_quat_w  # May be (N, 4)
        
        # Add target dimension if missing
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(1)  # (N, 3) -> (N, 1, 3)
        if target_quat.ndim == 2:
            target_quat = target_quat.unsqueeze(1)  # (N, 4) -> (N, 1, 4)
        
        target_poses = (target_pos, target_quat)

        self.bbox_raycaster.update(
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            target_poses=target_poses,
            agent_poses=agent_poses,
            image_shapes=image_shapes,
            target_scale=self.target_scale
        )

        # ===== TRIANGULATION COVARIANCE COMPUTATION =====
        # Stack robot positions and orientations [N, C, 3] and [N, C, 4]
        # Use pre-allocated stacked tensors for efficiency
        for agent_idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            joint_idx = self.gimbal_joint_idx[agent_id]
            self.robot_positions_stacked[:, agent_idx, :] = robot.data.root_pos_w
            self.robot_quats_stacked[:, agent_idx, :] = robot.data.root_state_w[:, 3:7]
            self.gimbal_yaws_stacked[:, agent_idx] = robot.data.joint_pos[:, joint_idx["yaw"]]
            self.gimbal_pitches_stacked[:, agent_idx] = robot.data.joint_pos[:, joint_idx["pitch"]]
            self.camera_intrinsics_stacked[:, agent_idx, :, :] = self.camera_intrinsics[agent_id]
        
        # Target positions [N, T, 3] where T=1 (one target per environment)
        target_pos_batch = self.target.data.root_state_w[:, :3].unsqueeze(1)  # [N, 1, 3]

        # Update Sigma_K based on zoom level
        for agent_idx, agent_id in enumerate(self.cfg.possible_agents):
            self.Sigma_K[:, agent_idx, :, :] = torch.eye(4, device=self.device).expand(self.num_envs, 4, 4) * (
                self.cfg.intrinsic_std * self.zoom_level[agent_id].unsqueeze(-1).unsqueeze(-1)) ** 2

        try:
            # Compute triangulation covariance using simplified version (pixel noise only)
            # self.Sigma_X, trace_cov = triangulation_covariance_simple(
            #     X_w=target_pos_batch,
            #     robot_positions=self.robot_positions_stacked,
            #     robot_quats=self.robot_quats_stacked,
            #     gimbal_yaws=self.gimbal_yaws_stacked,
            #     gimbal_pitches=self.gimbal_pitches_stacked,
            #     camera_intrinsics=self.camera_intrinsics_stacked,
            #     pixel_std=50.0
            # )
            # Full covariance computation 
            self.Sigma_X, self.trace_cov = triangulation_covariance_multi_camera(
                X_w=target_pos_batch,
                robot_positions=self.robot_positions_stacked,
                robot_quats=self.robot_quats_stacked,
                gimbal_yaws=self.gimbal_yaws_stacked,
                gimbal_pitches=self.gimbal_pitches_stacked,
                camera_intrinsics=self.camera_intrinsics_stacked,
                Sigma_pix=self.Sigma_pix,
                Sigma_twb=self.Sigma_twb,
                Sigma_phiwb=self.Sigma_phiwb,
                Sigma_alpha=self.Sigma_alpha,
                Sigma_beta=self.Sigma_beta,
                Sigma_K=self.Sigma_K,
                include_pose=True,
                include_gimbal=True,
                include_intrinsics=True
            )
            # Check validity for all targets: trace_cov should be non-negative and finite
            # Sigma_X: [N, T, 3, 3], trace_cov: [N, T]
            is_trace_valid = (self.trace_cov >= 0.0) & torch.isfinite(self.trace_cov)  # [N, T]
            is_sigma_finite = torch.isfinite(self.Sigma_X).all(dim=(-2, -1))  # [N, T]
            is_not_nan = ~torch.isnan(self.Sigma_X).any(dim=(-2, -1))  # [N, T]
            is_bbox_valid = self.bbox_raycaster.data.valid_mask.all(dim=1)  # [N, C] -> [N, T]
            self.is_tri_cov_valid = is_trace_valid & is_sigma_finite & is_not_nan & is_bbox_valid  # [N, T]
            self.Sigma_X = torch.where(
                self.is_tri_cov_valid.unsqueeze(-1).unsqueeze(-1),
                self.Sigma_X,
                self.Sigma_X_invalid
            )
            Sigma_X_diag = torch.diagonal(self.Sigma_X, dim1=-2, dim2=-1)  # [N, T, 3]
            Sigma_diag_sqrt = torch.sqrt(torch.clamp(Sigma_X_diag, min=0.0) + 1e-12)  # (N, T, 3)
            self.trace_cov = torch.where(
                self.is_tri_cov_valid,
                self.trace_cov,
                self.trace_cov_invalid
            )
        except Exception as e:
            # Fallback if computation fails
            # print(f"Warning: Triangulation covariance computation failed: {e}")
            self.is_tri_cov_valid = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            self.Sigma_X = self.Sigma_X_invalid.clone()
            self.trace_cov = self.trace_cov_invalid.clone()
            Sigma_diag_sqrt = torch.ones(self.num_envs, 1, 3, device=self.device) * (-1.0)


        observations = {}
        
        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            # This logic is specific to a 2-agent scenario
            other_agent_id = next(aid for aid in self.cfg.possible_agents if aid != agent_id)
            other_robot = self._robots[other_agent_id]
                        
            self.bboxes[agent_id] = self.bbox_raycaster.data.bboxes_xyxy[:, i, 0, :]
            self.bbox_valid_mask[agent_id][:, 0] = self.bbox_raycaster.data.valid_mask[:, i, 0]
            self.bboxes_normalized[agent_id] = self.bbox_raycaster.data.bboxes_normalized[:, i, 0, :] # xywh

            obs = torch.cat([
                robot.data.root_state_w[:, :10],  # 10: pos, quat, lin_vel
                robot.data.root_ang_vel_b[:, 2:3],  # 1: yaw rate
                robot.data.body_lin_acc_w[:, 0],  # 3: acceleration
                robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]:self.gimbal_joint_idx[agent_id]["pitch"] + 1],  # 1
                robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]:self.gimbal_joint_idx[agent_id]["yaw"] + 1],  # 1
                self.bboxes_normalized[agent_id],  # 4: normalized bbox
                self.bbox_valid_mask[agent_id].float(),  # 1: validity
                self.zoom_level[agent_id].unsqueeze(-1),  # 1: zoom level
                self.trace_cov[:, 0:1],  # 1: trace of covariance
                Sigma_diag_sqrt[:, 0, :],  # 3: X, Y, Z (world) std deviations
            ], dim=-1)
            
            # Check for NaN values in total obs
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

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """Compute rewards for all agents."""
        rewards_dict = {}

        # Calculate the curriculum progress factor (alpha) from 0.0 to 1.0
        start = self.cfg.curriculum_coordination_start_step
        end = self.cfg.curriculum_coordination_end_step

        # self.common_step_counter is from the base DirectMARLEnv class
        progress = (self.common_step_counter - start) / (end - start)
        progress = 0 if progress < 0 else (1 if progress > 1 else progress)
        curriculum_alpha = torch.tensor(progress, device=self.device)
        # curriculum_alpha = torch.ones(self.num_envs, device=self.device)
        
        trace_cov_per_env = self.trace_cov[:, 0]  # [N]
        triangulation_quality = torch.where(
            self.is_tri_cov_valid[:, 0],
            1.0 / torch.sqrt(trace_cov_per_env + 1e-12),
            torch.zeros(self.num_envs, device=self.device)
        )
            
        # ===== GEOMETRIC METRICS FOR DEBUGGING =====
        # Compute baseline distance between drones
        # baseline_distances: shape (N, C, C, 1)
        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            for j, other_agent_id in enumerate(self.cfg.possible_agents):
                if i >= j: # Compute only upper triangle
                    continue
                ego_robot = self._robots[ego_agent_id]
                other_robot = self._robots[other_agent_id]
                
                ego_pos = ego_robot.data.root_pos_w  # [N, 3]
                other_pos = other_robot.data.root_pos_w  # [N, 3]
                
                distance = torch.norm(ego_pos - other_pos, dim=1, keepdim=False)  # [N,]
                self.baseline_distances[:, i, j, 0] = distance  # Store distance
                self.baseline_distances[:, j, i, 0] = distance  # Symmetric
        # Collision table: shape (N, C, C, 1)
        collision_table = torch.where(
            (self.baseline_distances < self.cfg.min_safe_distance) & (self.baseline_distances > 0),
            torch.zeros_like(self.baseline_distances), # Zero for collision
            torch.ones_like(self.baseline_distances), # One for no collision
        )

        
        # ===== PER-AGENT REWARDS =====
        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            agent_cam_cfg = self.agent_camera_cfgs[agent_id]
            
            # Individual tracking rewards
            lin_vel = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)
            action_sum = torch.sum(torch.square(self.action_weight * self._actions[agent_id]), dim=1)
            action_delta = torch.sum(
                torch.square(self.action_delta_weight * (self._actions[agent_id] - self._last_actions[agent_id])), 
                dim=1
            )
            
            # Bounding box rewards
            bbox = (torch.square((self.bboxes[agent_id][:, 0] + self.bboxes[agent_id][:, 2])/2 - agent_cam_cfg.width/2) / agent_cam_cfg.width**2 
                    + torch.square((self.bboxes[agent_id][:, 1] + self.bboxes[agent_id][:, 3])/2 - agent_cam_cfg.height/2) / agent_cam_cfg.height**2)
            bbox_center_mapped = (1 - torch.tanh(bbox / 0.8)) * self.bbox_valid_mask[agent_id].squeeze(-1).float()
            
            bbox_size = (self.bboxes[agent_id][:, 2] - self.bboxes[agent_id][:, 0]) * (self.bboxes[agent_id][:, 3] - self.bboxes[agent_id][:, 1]) / (agent_cam_cfg.width * agent_cam_cfg.height)
            bbox_size_mapped = (1 - torch.tanh((bbox_size-0.2**2) / 0.8)) * self.bbox_valid_mask[agent_id].squeeze(-1).float()
            
            # Zoom penalty
            zoom_penalty = torch.square((self.zoom_level[agent_id] - 1.0)/10)

            # Collision penalty
            collision_penalty = torch.where(
                torch.prod(collision_table[:, i, :, 0], dim=1) < 1.0,
                torch.ones(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )

            # Combine rewards
            rewards = {
                # "lin_vel": lin_vel * self.cfg.lin_vel_penalty_scale * self.step_dt,
                "action_sum": action_sum * self.cfg.action_sum_penalty_scale * self.step_dt,
                #"action_delta": action_delta * self.cfg.action_delta_penalty_scale * self.step_dt,
                "bbox_center": bbox_center_mapped * self.cfg.bbox_center_reward_scale * self.step_dt,
                "bbox_size": bbox_size_mapped * self.cfg.bbox_size_reward_scale * self.step_dt,
                #"zoom": zoom_penalty * self.cfg.zoom_reward_scale * self.step_dt,
                "triangulation": triangulation_quality * self.cfg.triangulation_reward_scale * self.step_dt # * curriculum_alpha,
                # "collision": collision_penalty * self.cfg.collision_penalty_scale * self.step_dt * curriculum_alpha,
            }
            
            # Store for logging
            for key, value in rewards.items():
                self._episode_sums[agent_id][key] += value

            total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            
            # Check for NaN values in individual rewards
            for reward_name, reward_value in rewards.items():
                if torch.isnan(reward_value).any():
                    nan_envs = torch.nonzero(torch.isnan(reward_value)).flatten()
                    raise ValueError(f"NaN detected in {agent_id} reward '{reward_name}' at environments: {nan_envs.tolist()}")
            
            # Check for NaN values in total reward
            if torch.isnan(total_reward).any():
                nan_envs = torch.nonzero(torch.isnan(total_reward)).flatten()
                raise ValueError(f"NaN detected in {agent_id} total reward at environments: {nan_envs.tolist()}")
            
            # Total reward
            rewards_dict[agent_id] = torch.sum(torch.stack(list(rewards.values())), dim=0)
            
            # Update last actions
            self._last_actions[agent_id] = self._actions[agent_id].clone()
    
        return rewards_dict

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get termination and timeout flags for all agents."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated_dict = {}
        time_out_dict = {}
        
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            died = robot.data.root_pos_w[:, 2] < 0.7
            terminated_dict[agent_id] = died
            time_out_dict[agent_id] = time_out
        
        return terminated_dict, time_out_dict

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at specified indices."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[self.cfg.possible_agents[0]]._ALL_INDICES # Use any agent to get all indices
        
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

        if self.cfg.curriculum_tracking_start_step < 0:
            curriculum_tracking_state = 0
            for agent_id in self.cfg.possible_agents:
                curriculum_tracking_state += 1 if all_extras[f"Episode_Reward/{agent_id}_bbox_center"] > 50 else 0
            if curriculum_tracking_state == len(self.cfg.possible_agents):
                self.cfg.curriculum_tracking_start_step = self.common_step_counter
                self.cfg.curriculum_tracking_end_step += self.cfg.curriculum_tracking_start_step
                self.cfg.curriculum_coordination_start_step = self.cfg.curriculum_tracking_end_step
                self.cfg.curriculum_coordination_end_step = self.cfg.curriculum_coordination_start_step + self.cfg.curriculum_coordination_duration_steps

        formation_center = torch.zeros(len(env_ids), 3, device=self.device)
        formation_center[:, 0] = torch.zeros_like(formation_center[:, 0]).uniform_(-5.0, 5.0)
        formation_center[:, 1] = torch.zeros_like(formation_center[:, 1]).uniform_(0, 10.0)
        formation_center[:, 2] = torch.zeros_like(formation_center[:, 2]).uniform_(15.0, 20.0)

        gap_ref = torch.zeros(len(env_ids), 3, device=self.device)
        gap_ref[:, 0] = torch.zeros_like(gap_ref[:, 0]).uniform_(-5.0, 5.0)
        gap_ref[:, 1] = torch.zeros_like(gap_ref[:, 1]).uniform_(0, 10.0)
        gap_ref[:, 2] = torch.zeros_like(gap_ref[:, 2]).uniform_(5.0, 7.0)

        # Reset target position (centered between drones)
        target_pos = torch.zeros(len(env_ids), 3, device=self.device)
        start = self.cfg.curriculum_tracking_start_step
        end = self.cfg.curriculum_tracking_end_step
        progress = (self.common_step_counter - start) / (end - start)
        progress = 0.1 if (progress < 0.1 or start < 0) else (1 if progress > 1 else progress)
        # progress = 1 if DEBUG_DRAW else progress  # Always use full range in debug mode
        target_pos[:, 0] = torch.zeros_like(target_pos[:, 0]).uniform_(40.0 * progress, 80.0 * progress) + formation_center[:, 0]
        target_pos[:, 1] = torch.zeros_like(target_pos[:, 1]).uniform_(-20.0 * progress, 20.0 * progress) + formation_center[:, 1]
        target_pos[:, 2] = torch.zeros_like(target_pos[:, 2]).uniform_(0.0, 5.0) + formation_center[:, 2]
        target_pos += self._env_origins[env_ids]
        
        self.target.data.root_state_w[env_ids, :3] = target_pos
        self.target.data.root_state_w[env_ids, 7:] = torch.zeros_like(self.target.data.root_state_w[env_ids, 7:])
        self.target.write_root_pose_to_sim(self.target.data.root_state_w[env_ids, :7], env_ids)
        
        # Randomize target scale
        # self.target_scale[env_ids, :, 0] = torch.zeros_like(self.target_scale[env_ids, :, 0]).uniform_(
        #     0.5 * self.base_target_scale, 1.5 * self.base_target_scale)
        # self.target_scale[env_ids, :, 1] = self.target_scale[env_ids, :, 0]
        # self.target_scale[env_ids, :, 2] = torch.zeros_like(self.target_scale[env_ids, :, 2]).uniform_(
        #     0.5 * self.base_target_scale, 1.0 * self.base_target_scale)
        # Reset target velocity
        self.target_vel[env_ids] = torch.zeros_like(self.target_vel[env_ids]).uniform_(-1.0, 1.0)
        self.last_target_vel[env_ids] = self.target_vel[env_ids].clone()

        # Reset robots
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            robot.reset(env_ids)
            
            # Reset actions
            self._actions[agent_id][env_ids] = 0.0
            self._last_actions[agent_id][env_ids] = 0.0
            
            # Reset zoom level
            self.zoom_level[agent_id][env_ids] = torch.zeros_like(self.zoom_level[agent_id][env_ids]).uniform_(1.0, 2.0)
            
            # Set initial positions with offset for each drone based on its index
            default_root_state = robot.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = self._env_origins[env_ids].clone()
            default_root_state[:, 0:2] += formation_center[:, 0:2] + gap_ref[:, 0:2] * (idx + 1) * (-1)**(idx + gap_ref[:, 0:2].int())
            default_root_state[:, 2] += formation_center[:, 2] + gap_ref[:, 2] * (-1)**(idx + gap_ref[:, 2].int())
            default_root_state[:, 3:7] = quat_from_euler_xyz(
                                            torch.zeros(len(env_ids), device=self.device),
                                            torch.zeros(len(env_ids), device=self.device),
                                            torch.zeros(len(env_ids), device=self.device))
                                            # torch.zeros(len(env_ids), device=self.device).uniform_(-math.pi*1/6, math.pi*1/6))

            # Reset gimbal targets
            joint_pos = robot.data.default_joint_pos[env_ids]
            gimbal_idx = self.gimbal_joint_idx[agent_id]
            gimbal_yaw, gimbal_pitch = self.point_to_region(
                                            default_root_state[:, :3],
                                            default_root_state[:, 3:7],
                                            target_pos)
                                            # target_pos + torch.zeros_like(target_pos).uniform_(-5.0, 5.0))
            joint_pos[:, gimbal_idx["yaw"]] = gimbal_yaw
            joint_pos[:, gimbal_idx["pitch"]] = gimbal_pitch
            joint_pos[:, gimbal_idx["roll"]] = torch.zeros_like(joint_pos[:, gimbal_idx["roll"]])
            self.gimbal_dof_targets[agent_id][env_ids] = joint_pos

            # Write to simulation
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            
            # Reset stabilizers
            self._stabilizers[agent_id].reset(env_ids)
        
        # Reset episode length buffer
        super()._reset_idx(env_ids)

    def point_to_region(self, drone_pos_w, drone_quat_w, target_pos):
        # Vector from drone to target in world frame
        target_vector_w = target_pos - drone_pos_w
        
        # Transform target vector to drone body frame
        drone_quat_inv = quat_inv(drone_quat_w)
        target_vector_b = quat_rotate(drone_quat_inv, target_vector_w)
        
        # Calculate gimbal angles in drone body frame
        # Yaw: rotation around Z-axis (horizontal pointing)
        gimbal_yaw = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])
        
        # Pitch: rotation around Y-axis (vertical pointing)
        horizontal_distance = torch.sqrt(target_vector_b[:, 0]**2 + target_vector_b[:, 1]**2)
        gimbal_pitch = torch.atan2(-target_vector_b[:, 2], horizontal_distance)
        
        return gimbal_yaw, gimbal_pitch

    def _update_target_movement(self, mode: str = "linear"):
        """Update target movement with smooth acceleration-based motion."""
        self.target_vel_change_timer += self.step_dt
        
        # Randomly change desired velocity
        change_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.target_direction_change_prob
        
        if mode == "linear":
            # Generate new random desired velocities
            new_desired_vel = torch.zeros_like(self.target_desired_vel)
            new_desired_vel[:, :3] = torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0
            new_desired_vel[:, 3:] = torch.rand(self.num_envs, 3, device=self.device) * 0.2 - 0.1
            new_desired_vel[:, :3] *= self.cfg.max_target_speed * 0.3
            
            # Update desired velocity where mask is true
            self.target_desired_vel = torch.where(
                change_mask.unsqueeze(-1).expand(-1, 6),
                new_desired_vel,
                self.target_desired_vel
            )
            
            # Reset timer for environments that changed direction
            self.target_vel_change_timer = torch.where(
                change_mask, 
                torch.zeros_like(self.target_vel_change_timer), 
                self.target_vel_change_timer
            )
            
            # Compute acceleration towards desired velocity
            vel_error = self.target_desired_vel - self.target_vel
            self.target_acceleration = vel_error * self.cfg.target_acceleration_scale
            
            # Clamp acceleration
            self.target_acceleration = torch.clamp(
                self.target_acceleration, 
                -self.cfg.target_max_acceleration, 
                self.cfg.target_max_acceleration
            )
        elif mode == "circular":
            t = self.common_step_counter * self.step_dt
            radius = torch.rand_like(change_mask.float()) * 20.0 + 50.0
            angular_speed = torch.rand_like(change_mask.float()) * 0.5 + 0.2
            
            center = self._env_origins
            desired_x = center[:, 0] + radius * torch.cos(angular_speed * t)
            desired_y = center[:, 1] + radius * torch.sin(angular_speed * t)
            desired_z = 10.0 + 5.0 * torch.sin(0.5 * angular_speed * t)

            desired_pos = torch.stack([desired_x, desired_y, desired_z], dim=1)
            pos_error = desired_pos - self.target.data.root_state_w[:, :3]
            
            # Proportional control to compute desired velocity
            kp = 1.0
            self.target_desired_vel[:, :3] = kp * pos_error
            
            # Clamp desired velocity
            speed = torch.norm(self.target_desired_vel[:, :3], dim=1, keepdim=True)
            speed_clamped = torch.clamp(speed, max=self.cfg.max_target_speed)
            self.target_desired_vel[:, :3] = self.target_desired_vel[:, :3] / (speed + 1e-6) * speed_clamped
            
            # No rotation in circular mode
            self.target_desired_vel[:, 3:] = 0.0
            
            # Compute acceleration towards desired velocity
            vel_error = self.target_desired_vel - self.target_vel
            self.target_acceleration = vel_error * self.cfg.target_acceleration_scale
            
            # Clamp acceleration
            self.target_acceleration = torch.clamp(
                self.target_acceleration, 
                -self.cfg.target_max_acceleration, 
                self.cfg.target_max_acceleration
            )
        
        # Update velocity with acceleration and damping
        self.target_vel += self.target_acceleration * self.step_dt
        self.target_vel *= self.cfg.target_velocity_damping
        
        # Clamp final velocity
        self.target_vel[:, :3] = torch.clamp(
            self.target_vel[:, :3], 
            -self.cfg.max_target_speed, 
            self.cfg.max_target_speed
        )

    def _update_research_metrics(self):
        # --- geometry ---
        p0 = self._robots["drone_0"].data.root_pos_w
        p1 = self._robots["drone_1"].data.root_pos_w
        pt = self.target.data.root_pos_w

        d01 = torch.norm(p0 - p1, dim=1)  # baseline (m)

        v0 = pt - p0
        v1 = pt - p1
        cosb = torch.sum(v0 * v1, dim=1) / (torch.norm(v0, dim=1) * torch.norm(v1, dim=1) + 1e-8)
        cosb = torch.clamp(cosb, -1.0, 1.0)
        beta = torch.acos(cosb)                               # rad
        beta_deg = torch.rad2deg(beta)
        w_geom = torch.sin(beta) ** 2                         # sin^2 beta

        # --- bbox area fraction & validity ---
        def bbox_area_frac(b, cam_cfg):
            area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
            return area / (cam_cfg.width * cam_cfg.height)

        a0 = bbox_area_frac(self.bboxes["drone_0"], self.agent_camera_cfgs["drone_0"])
        a1 = bbox_area_frac(self.bboxes["drone_1"], self.agent_camera_cfgs["drone_1"])
        valid0 = self.bbox_valid_mask["drone_0"][:, 0]
        valid1 = self.bbox_valid_mask["drone_1"][:, 0]
        both_valid = valid0 & valid1

        # --- TQI ---
        a = torch.minimum(a0, a1)  # conservative: use smaller footprint
        mu = self.cfg.research.tqi_opt_area
        sig = self.cfg.research.tqi_area_sigma
        w_scale = torch.exp(-((a - mu) ** 2) / (sig ** 2 + 1e-12))
        tqi = both_valid.float() * w_geom * w_scale

        # --- CVTT accumulation ---
        self._research["cvtt_sec"] += (tqi >= self.cfg.research.tqi_threshold).float() * self.step_dt
        self._research["tqi_sum"] += tqi
        self._research["steps"] += 1

        # --- Noise proxy & IG_approx ---
        # Focal length used in your bbox generation is in mm; convert to meters
        f0_m = (self.base_focal_length["drone_0"] * self.zoom_level["drone_0"]) * 1e-3
        f1_m = (self.base_focal_length["drone_1"] * self.zoom_level["drone_1"]) * 1e-3
        pitch = self.cfg.research.pixel_pitch_m
        sig_px = self.cfg.research.det_std_px
        sig_th0 = sig_px * pitch / (f0_m + 1e-12)     # radians
        sig_th1 = sig_px * pitch / (f1_m + 1e-12)

        d0 = torch.norm(v0, dim=1) + 1e-6
        d1 = torch.norm(v1, dim=1) + 1e-6
        fim_scalar = w_geom * (1.0 / (d0 * d0 * sig_th0 * sig_th0) + 1.0 / (d1 * d1 * sig_th1 * sig_th1))
        ig_approx = torch.log(fim_scalar + 1e-12)

        # ====== STEP LOGGING (throttled) ======
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

            # A couple of percentiles from the subsample (keeps tensors small)
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
            self._research["last_step_logged"] = self.common_step_counter

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if DEBUG_DRAW and hasattr(self, 'draw_interface'):
            self.draw_interface.clear_lines()
        
            for agent_id in self.cfg.possible_agents:
                # Draw line to target (color indicates visibility)
                line_colors_yellow = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                line_colors_green = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                line_colors_red = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                
                # Red if invalid (occluded), yellow if invalid (other reason), green if valid
                is_valid = self.bbox_valid_mask[agent_id]
                agent_idx = self.cfg.possible_agents.index(agent_id)
                if self.bbox_raycaster.data.valid_bbox_visibility_mask is not None:
                    is_visible = self.bbox_raycaster.data.valid_bbox_visibility_mask[:, agent_idx, 0].unsqueeze(-1)
                else:
                    is_visible = torch.ones_like(is_valid) > 0

                line_colors = torch.where(
                    is_valid, 
                    line_colors_green,
                    torch.where(
                        is_visible,
                        line_colors_yellow,  # Invalid but not occluded (e.g., out of FOV)
                        line_colors_red      # Occluded
                    )
                ).tolist()
                
                line_thicknesses = [1.0] * self.camera_pos_world[agent_id].shape[0]
                self.draw_interface.draw_lines(
                    self.camera_pos_world[agent_id].tolist(), 
                    self.target.data.root_pos_w.tolist(), 
                    line_colors, 
                    line_thicknesses
                )
                
                # Draw camera frustum
                if hasattr(self, 'camera_frustrum'):
                    self.camera_frustrum.draw_frustrum(
                        camera_position=self.camera_pos_world[agent_id],
                        camera_orientation=self.camera_quat_world[agent_id],
                        camera_intrinsics=self.camera_cfg_batch[agent_id],
                        zoom_level=self.zoom_level[agent_id],
                        device=self.device
                    )

            # Visualize triangulation covariance ellipsoids
            ray_dir = get_ray_dir_from_bbox(
                self.bbox_raycaster.data.bboxes,
                self.camera_intrinsics_stacked,
                torch.stack([self.camera_quat_world[agent_id] for agent_id in self.cfg.possible_agents], dim=1),
                torch.stack([self.camera_pos_world[agent_id] for agent_id in self.cfg.possible_agents], dim=1),
            )
            X_w_est = midpoint_method_batched(
                pts=torch.stack([self.camera_pos_world[agent_id] for agent_id in self.cfg.possible_agents], dim=1),
                dirs=ray_dir,
            )
            # Extract diagonal of Sigma_X (N, T, 3, 3) -> (N, T, 3)
            Sigma_diag = torch.diagonal(self.Sigma_X, dim1=-2, dim2=-1)  # (N, T, 3)
            # Numerical stability and square root to get std-dev per axis
            Sigma_diag_sqrt = torch.sqrt(torch.clamp(Sigma_diag, min=0.0) + 1e-12)  # (N, T, 3)
            # Reduce to (N, 3). For single target (T=1) pick first target; otherwise take first target as default.
            if Sigma_diag_sqrt.ndim == 3 and Sigma_diag_sqrt.shape[1] >= 1:
                scales = Sigma_diag_sqrt[:, 0, :]  # (N, 3)
            else:
                scales = Sigma_diag_sqrt.squeeze(1)  # fallback to (N, 3)

            # Visualize triangulation covariances using per-axis std-dev scales
            self.tri_cov_visualizer.visualize(
                translations=X_w_est.squeeze(1),  # (N, T, 3) -> (N, 3)
                scales=scales * 2.5, # 95% confidence interval approx.
            )
            
            # Visualize raycaster debug info (occlusion rays, etc.)
            self.bbox_raycaster.visualize()
