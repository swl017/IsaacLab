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

from isaaclab_tasks.direct.iris_ma3.iris_ma_env3_cfg import IrisMAEnvCfg
from isaaclab_tasks.direct.iris_ma3.bbox_raycaster import BBoxRayCaster, BBoxRayCasterCfg
from isaaclab_tasks.direct.iris_ma3.triangulation.triang_cov_reward_torch import (
    triangulation_covariance_multi_camera,
    triangulation_covariance_simple,
    midpoint_method_batched,
    get_ray_dir_from_bbox
)
from isaaclab_tasks.direct.iris_ma3.delay_system import MultiAgentDelaySystem, FirstOrderLag
from isaaclab_tasks.direct.iris_ma3.controller import PointMass, GimbalStabilizer
from isaaclab_tasks.direct.iris_ma3.safety import SafetyManager, SafetyManagerCfg, CollisionDetectorCfg, TTCComputerCfg
from isaaclab_tasks.direct.iris_ma3.randomization import Randomizer
from isaaclab_tasks.direct.iris_ma3.visualization import CustomVisualization

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

        # Call parent init - this will call _setup_scene() which creates self._robots
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        C = len(cfg.possible_agents) # C as in "Cameras"
        T = 1 # number of targets

        # Get body and joint indices for each robot (AFTER super().__init__())
        self._body_ids = {}
        self.gimbal_joint_idx = {}
        self._robot_mass = {}
        self._robot_dynamics = {}
        self._robot_controllers = {}
        self._gimbal_stabilizers = {}

        for agent in self.cfg.possible_agents:
            robot = self._robots[agent]
            self._body_ids[agent] = robot.find_bodies("body")[0][0]
            self.gimbal_joint_idx[agent] = {
                "yaw": robot.find_joints("yaw_joint")[0][0],
                "roll": robot.find_joints("roll_joint")[0][0],
                "pitch": robot.find_joints("pitch_joint")[0][0],
            }
            self._robot_mass[agent] = robot.root_physx_view.get_masses()[0].sum()
            self._robot_dynamics[agent] = FirstOrderLag(
                num_envs=self.num_envs,
                state_dim=6, # [vx, vy, vz, wx, wy, wz]
                time_constant=self.cfg.dynamics_time_constant,
                dt=self.step_dt,
                device=self.device,
            )
            robot_weight = (self._robot_mass[agent] * torch.tensor(self.sim.cfg.gravity, device=self.device).norm()).item()
            self._robot_controllers[agent] = PointMass(
                mass=self._robot_mass[agent], 
                weight=robot_weight, 
                num_envs=self.num_envs, 
                disable_gravity=self.cfg.robot.spawn.rigid_props.disable_gravity, 
                device=self.device
            )
            self._gimbal_stabilizers[agent] = GimbalStabilizer(
                yaw_limits=self.cfg.max_gimbal_yaw_angle,
                pitch_limits=self.cfg.max_gimbal_pitch_angle,
                device=self.device
            )

        # Initialize GT action variables
        self._actions = {agent_id: torch.zeros(self.num_envs, len(self.cfg.action_weight), device=self.device) for agent_id in cfg.possible_agents}
        self._last_actions = {agent_id: torch.zeros(self.num_envs, len(self.cfg.action_weight), device=self.device) for agent_id in cfg.possible_agents}
        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device)

        self.cmd_vel = torch.zeros(self.num_envs, len(self.cfg.possible_agents), len(self.cfg.action_weight), device=self.device) # [N, C, 7]
        self.last_cmd_vel = torch.zeros_like(self.cmd_vel)  # [N, C, 7]
        self.cmd_gimbal_yaw = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)  # [N, C]
        self.cmd_gimbal_pitch = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)  # [N, C]
        self.zoom_level = torch.ones(self.num_envs, len(self.cfg.possible_agents), device=self.device)  # [N, C]

        # Initialize MultiAgentDelaySystem
        self.delay_system = MultiAgentDelaySystem(
            possible_agents=cfg.possible_agents,
            num_envs=self.num_envs,
            num_joints_per_agent={agent: 3 for agent in cfg.possible_agents},  # yaw, pitch, roll
            num_targets_per_agent={agent: 1 for agent in cfg.possible_agents},
            dt=self.step_dt,
            device=self.device,
            # Noise parameters
            enable_noise=cfg.enable_noise_in_observations,
            position_noise_std=cfg.pos_std,
            orientation_noise_std=cfg.ori_std,
            linear_velocity_noise_std=0.1 * cfg.pos_std,  # 10% of position noise
            angular_velocity_noise_std=cfg.ori_std * cfg.pos_std,  # Product formula
            linear_acceleration_noise_std=100e-6,
            gimbal_noise_std=cfg.gimbal_std,
            bbox_noise_std=cfg.pix_std,
            zoom_noise_std=0.1,  # Zoom noise std
            noise_seed=0  # Fixed seed for reproducibility
        )

        # Set camera config for state manager
        width = self.cfg.camera.width
        height = self.cfg.camera.height
        focal_length = self.cfg.camera.spawn.focal_length
        horizontal_aperture = self.cfg.camera.spawn.horizontal_aperture
        vertical_aperture = horizontal_aperture * (height / width)

        # Get camera offset from config
        offset_pos = torch.tensor(self.cfg.camera.offset.pos, device=self.device)
        offset_rot = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self.device) # torch.tensor(self.cfg.camera.offset.rot, device=self.device)

        # Set camera configs for all agents
        for agent_id in cfg.possible_agents:
            self.delay_system.set_camera_configs(
                agent_id,
                width=width,
                height=height,
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=vertical_aperture,
                offset_position_b=offset_pos,
                offset_rotation_b=offset_rot
            )

        # Initialize BBoxRayCaster (lazy - will be created on first use)
        self.bbox_raycaster: BBoxRayCaster = None
        self._bbox_raycaster_initialized = False

        # Initialize BBoxRayCaster on first use
        self._initialize_bbox_raycaster()

        # Initialize targets
        num_targets_per_env = 1
        self.base_target_scale = self.cfg.target_cfg.spawn.scale[0]
        self.target_scale = torch.ones(self.num_envs, num_targets_per_env, 3, device=self.device) * self.base_target_scale

        # Initialize SafetyManager for collision detection and TTC computation
        self.safety_manager = SafetyManager(
            cfg=SafetyManagerCfg(
                enable_collision_detection=True,
                enable_ttc_computation=True,
                collision_cfg=CollisionDetectorCfg(
                    min_safe_distance=self.cfg.collision_min_safe_distance,
                    collision_penalty_scale=1.0,
                ),
                ttc_cfg=TTCComputerCfg(
                    horizon_sec=self.cfg.ttc_horizon if hasattr(self.cfg, 'ttc_horizon') else 6.0,
                    ema_alpha_s=0.6,
                    ema_alpha_f=0.6,
                    deriv_clip=0.5,
                    stale_half_life=1.0,
                    zoom_gate_k=0.2,
                    ttc_penalty_scale=1.0,
                ),
            ),
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
        )

        # Randomizer for initial states (gimbal-aware formation and target sampling)
        self.randomizer = Randomizer(
            self.num_envs,
            self.device,
            gimbal_pitch_limits=self.cfg.max_gimbal_pitch_angle  # Pass gimbal constraints
        )

        # Tune randomizer for desired target distances
        # For targets at 15-20m, we need tight formations
        self.randomizer.initial_states.cfg.max_agent_separation = 3.0  # Down from 12.0
        self.randomizer.initial_states.cfg.min_agent_separation = 2.0  # Down from 3.0
        self.randomizer.initial_states.cfg.z_variation_range = (0.0, 1.0)  # Down from (0.0, 5.0)

        # Adjust target distance parameters
        self.randomizer.targets.cfg.distance_scale_factor = 2.0  # Down from 2.5

        # With these settings:
        # - Max formation spread: ~6m
        # - Min target distance: ~12-15m
        # - Target range: 15-20m ✓

        # Curriculum parameters
        self.progress_delay = 0.0  # Will be updated during training
        self.progress_coord = 0.0  # For triangulation covariance scaling

        # Triangulation covariance placeholders
        self.Sigma_X_invalid = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, 1, 3, 3
        ) * (-1.0)
        self.trace_cov_invalid = torch.ones(self.num_envs, 1, device=self.device) * (-1.0)


        # Uncertainty matrices for triangulation covariance computation
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
        # Camera intrinsics uncertainties [N, C, 4, 4]
        self.Sigma_K = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.cfg.possible_agents), 4, 4
        ) * (self.cfg.intrinsics_std ** 2)

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
                    "ttc",
                    ]
            }
            for agent in self.cfg.possible_agents
        }

        # Visualization
        self.set_debug_vis(self.cfg.debug_vis)
        if DEBUG_DRAW and omni_debug_draw is not None:
            self.visualization = CustomVisualization(self.num_envs, self.cfg.possible_agents, self.device)

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
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

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

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]):
        """Pre-process actions for all agents before physics step. Runs at decimated rate."""
        current_step = self.cfg.play_sim_at_step if DEBUG_DRAW else self.common_step_counter
        self.progress_all = self._linear_progress(0, self.cfg.curriculum_all_end_step, current_step)
        self.progress_delay = self._linear_progress(self.cfg.curriculum_delay_start_step, self.cfg.curriculum_delay_end_step, current_step)
        self.progress_tracking = self._linear_progress(self.cfg.curriculum_tracking_start_step, self.cfg.curriculum_tracking_end_step, current_step)
        # self.progress_coord = self._linear_progress(self.cfg.curriculum_coordination_start_step, self.cfg.curriculum_coordination_end_step, current_step)
        self.progress_coord = 1.0 if DEBUG_DRAW else self._linear_progress(self.cfg.curriculum_coordination_start_step, self.cfg.curriculum_coordination_end_step, current_step)
        self.progress_safety = self._linear_progress(self.cfg.curriculum_safety_start_step, self.cfg.curriculum_safety_end_step, current_step)
        self.progress_move = self._linear_progress(self.cfg.curriculum_moving_target_start_step, self.cfg.curriculum_moving_target_end_step, current_step)
        self.progress_dynamics = self._linear_progress(self.cfg.curriculum_dynamics_start_step, self.cfg.curriculum_dynamics_end_step, current_step)

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            # Clip and store actions
            action = torch.clamp(actions[agent_id], min=-1.0, max=1.0)
            self._actions[agent_id] = action.clone()

            # Scale actions
            self.cmd_vel[:, idx, 0:3] = action[:, 0:3] * self.cfg.max_lin_vel
            self.cmd_vel[:, idx, 3] = action[:, 3] * self.cfg.max_yaw_rate
            self.cmd_vel[:, idx, 4:6] = action[:, 4:6] * self.cfg.max_gimbal_angle_rate
            self.cmd_vel[:, idx, 6] = action[:, 6] * self.cfg.max_zoom_rate


            # Get current gimbal angles
            robot = self._robots[agent_id]
            gimbal_yaw = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]].clone()
            gimbal_pitch = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]].clone()

            self.cmd_gimbal_yaw[:, idx] = gimbal_yaw + self.cmd_vel[:, idx, 4] * self.cfg.sim.dt
            self.cmd_gimbal_pitch[:, idx] = gimbal_pitch + self.cmd_vel[:, idx, 5] * self.cfg.sim.dt


    def _apply_action(self):
        """Apply actions to the environment. Runs at simulation rate, not at decimated rate."""
        # action: [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Robot root control
            force, moment = self._robot_controllers[agent_id].compute_control_quat_exact(
                cmd_lin_vel_w=self.cmd_vel[:, idx, 0:3],
                cmd_yaw_vel_w=self.cmd_vel[:, idx, 3],
                curr_quat_w=robot.data.root_quat_w.clone(),
                curr_lin_vel_w=robot.data.root_lin_vel_w.clone(),
                curr_ang_vel_b=robot.data.root_ang_vel_b.clone(),
                curr_lin_acc_b=robot.data.body_lin_acc_w[:, self._body_ids[agent_id]].clone(),
                dt=self.cfg.sim.dt
            )  # linear and angular velocities
            
            robot.set_external_force_and_torque(
                forces=force.unsqueeze(1),
                torques=moment.unsqueeze(1),
                body_ids=self._body_ids[agent_id]
            )

            # robot.write_root_velocity_to_sim(
            #     torch.cat([
            #         self.cmd_vel[:, idx, 0:3],  # linear velocity
            #         torch.zeros(self.num_envs, 2, device=self.device),  # x,y angular velocity
            #         self.cmd_vel[:, idx, 3:4],  # z angular velocity (yaw rate)
            #     ], dim=-1)
            # )

            # Gimbal control
            gimbal_yaw = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]].clone()
            gimbal_pitch = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]].clone()

            gimbal_yaw_target = self.cmd_gimbal_yaw[:, idx]
            gimbal_pitch_target = self.cmd_gimbal_pitch[:, idx] # TODO: Compensate for drone attitude: horizontal stabilized frame

            gimbal_roll_stabilizing = self._gimbal_stabilizers[agent_id].compute_stabilizing_roll(gimbal_yaw, gimbal_pitch, robot.data.root_quat_w)

            robot.set_joint_position_target(
                target=torch.stack([self.cmd_gimbal_yaw[:, idx], gimbal_roll_stabilizing, self.cmd_gimbal_pitch[:, idx]], dim=-1),
                joint_ids=[
                    self.gimbal_joint_idx[agent_id]["yaw"],
                    self.gimbal_joint_idx[agent_id]["roll"],
                    self.gimbal_joint_idx[agent_id]["pitch"],
                ]
            )

            self.zoom_level[:, idx] += self.cmd_vel[:, idx, 6] * self.cfg.max_zoom_rate * self.cfg.sim.dt  # zoom level
            self.zoom_level[:, idx] = torch.clamp(self.zoom_level[:, idx], min=1.0, max=self.cfg.max_zoom_level)

    def _linear_progress(self, start: int, end: int, current=None):
        if end <= start:
            return 1.0
        if current is None:
            current = self.common_step_counter
        x = (current - start) / (end - start)
        return 0.0 if x < 0 else (1.0 if x > 1 else float(x))

    def _initialize_bbox_raycaster(self):
        """Lazy initialization of BBoxRayCaster after targets are spawned."""
        if self._bbox_raycaster_initialized:
            return

        # Create a modified config with actual target paths
        bbox_cfg = copy.deepcopy(self.cfg.bbox_raycaster)
        bbox_cfg.debug_vis = DEBUG_DRAW
        target_prim_paths = self.scene.env_prim_paths # [f"/World/envs/env_{i}/target" for i in range(self.num_envs)]
        bbox_cfg.target_prim_paths = target_prim_paths

        self.bbox_raycaster = BBoxRayCaster(
            cfg=bbox_cfg,
            num_envs=self.num_envs,
            num_targets_per_env=1,
            device=self.device,
            agent_ids=self.cfg.possible_agents
        )
        self._bbox_raycaster_initialized = True

    def _compute_intermediate_values(self):
        """Compute intermediate values shared by both rewards and observations.

        This method prepares:
        1. GT states → Delayed states → Delayed+noisy states
        2. Bounding box detections from GT states
        3. Delayed and noisy bbox detections
        4. Communication between agents
        """

        # ========================================
        # CRITICAL: Update simulation time first!
        # ========================================
        # This advances the internal clock in MultiAgentObservationPipeline
        # so that delayed data (with FPS throttling and latency) can become available.
        # Without this, delayed bboxes will NEVER show valid=True!
        self.delay_system.update_time()

        # Update curriculum scaling
        self.delay_system.set_noise_progress_scale(self.progress_delay)

        # ========== Update GT states for all agents ==========
        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Get body angular velocity including gimbal (combined angular velocity)
            body_link_ang_vel_w = robot.data.body_link_ang_vel_w[:, self.gimbal_joint_idx[agent_id]["pitch"], :]

            gimbal_joint_indices = torch.tensor([
                self.gimbal_joint_idx[agent_id]["roll"],
                self.gimbal_joint_idx[agent_id]["pitch"],
                self.gimbal_joint_idx[agent_id]["yaw"],
            ], device=self.device)

            # Update GT states (automatically creates delayed and delayed+noisy)
            self.delay_system.update_gt_states(
                agent_id=agent_id,
                body_position_w=robot.data.root_pos_w,
                body_orientation_w=robot.data.root_quat_w,
                body_linear_velocity_w=robot.data.root_lin_vel_w,
                body_angular_velocity_w=robot.data.root_ang_vel_w,
                body_linear_acceleration_w=robot.data.body_lin_acc_w[:, self._body_ids[agent_id], :],
                body_combined_angular_velocity_w=body_link_ang_vel_w,  # Combined angular velocity
                joint_positions_b=robot.data.joint_pos[:, gimbal_joint_indices],  # [roll, pitch, yaw]
                zoom_level=self.zoom_level[:, i]  # [N,]
            )

        # ========== Get GT bounding boxes using BBoxRayCaster ==========
        # Collect GT camera poses and intrinsics
        gt_camera_poses = {}
        gt_camera_intrinsics = {}

        for agent_id in self.cfg.possible_agents:
            gt_state = self.delay_system.gt_states.agents[agent_id]
            gt_camera_poses[agent_id] = (
                gt_state.data.camera_position_w,
                gt_state.data.camera_orientation_w
            )
            gt_camera_intrinsics[agent_id] = gt_state.data.camera_base_intrinsics.clone()
            gt_camera_intrinsics[agent_id][:, 0, 0] *= gt_state.data.camera_zoom_level  # fx
            gt_camera_intrinsics[agent_id][:, 1, 1] *= gt_state.data.camera_zoom_level  # fy

        # Get target GT pose
        target_pos = self.target.data.root_pos_w
        target_quat = self.target.data.root_quat_w
        # Placeholder for multiple targets pos (N, T, 3) and quat (N, T, 4)
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(1)
        if target_quat.ndim == 2:
            target_quat = target_quat.unsqueeze(1)
        
        # Collect agent body poses for occlusion checking
        agent_body_poses = {
            agent_id: (robot.data.root_pos_w, robot.data.root_quat_w)
            for agent_id, robot in self._robots.items()
        }
        image_shapes = {
            str(agent_id): (self.agent_camera_cfgs[agent_id].height, self.agent_camera_cfgs[agent_id].width)
            for agent_id in self.cfg.possible_agents
        }
        # Update bbox raycaster with GT data
        self.bbox_raycaster.update(
            camera_poses=gt_camera_poses,
            camera_intrinsics=gt_camera_intrinsics,
            target_poses=(target_pos, target_quat),
            agent_poses=agent_body_poses,
            image_shapes=image_shapes,
            target_scale=self.target_scale
        )

        # ========== Update detections in state manager ==========
        for i, agent_id in enumerate(self.cfg.possible_agents):
            # Get GT bboxes from raycaster [N, 1, 4] (single target)
            gt_bbox = self.bbox_raycaster.data.bboxes[:, i:i+1, :, :]  # [N, 1, T, 4]
            bbox_valid = self.bbox_raycaster.data.valid_mask[:, i:i+1, :]  # [N, 1, T]

            # Update detections (applies FPS throttle, latency, dropout, pixel noise)
            self.delay_system.update_detections(
                agent_id=agent_id,
                bboxes_2d_gt=gt_bbox.squeeze(1),  # [N, T, 4] -> [N, 1, 4]
            )
        # Debug: Show GT bbox state (v2.2: validate using raycaster)
        gt_bbox_debug = self.delay_system.gt_states.agents[self.cfg.possible_agents[0]].data.bboxes_2d
        gt_bbox_valid_debug = self.bbox_raycaster.validate_bbox(gt_bbox_debug[:, 0, :])
        print(f"bbox_gt_states:")
        print(f"  bboxes_2d: {gt_bbox_debug}")
        print(f"  bboxes_2d_valid (raycaster): {gt_bbox_valid_debug.tolist()}")
        # NOTE: broadcast_state() removed in v2.2 - communication delays handled internally

    def _compute_triangulation_covariance(self, X_w, robot_positions, robot_quats,
                                          gimbal_yaws, gimbal_pitches,
                                          camera_intrinsics, bbox_valid_mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print(f"[DEBUG] _compute_triangulation_covariance")
        print(f"  X_w: {X_w.shape} ({X_w.tolist()})")
        print(f"  X_w_gt: {self.target.data.root_pos_w.shape} ({self.target.data.root_pos_w.tolist()})")
        print(f"  robot_positions: {robot_positions.shape} ({robot_positions.tolist()})")
        print(f"  robot_quats: {robot_quats.shape} ({robot_quats.tolist()})")
        print(f"  gimbal_yaws: {gimbal_yaws.shape} ({gimbal_yaws.tolist()})")
        print(f"  gimbal_pitches: {gimbal_pitches.shape} ({gimbal_pitches.tolist()})")
        print(f"  camera_intrinsics: {camera_intrinsics.shape} ({camera_intrinsics.tolist()})")
        print(f"  Sigma_pix: {self.Sigma_pix.shape} ({self.Sigma_pix.tolist()})")
        print(f"  Sigma_twb: {self.Sigma_twb.shape} ({self.Sigma_twb.tolist()})")
        print(f"  Sigma_phiwb: {self.Sigma_phiwb.shape} ({self.Sigma_phiwb.tolist()})")
        print(f"  Sigma_alpha: {self.Sigma_alpha.shape} ({self.Sigma_alpha.tolist()})")
        print(f"  Sigma_beta: {self.Sigma_beta.shape} ({self.Sigma_beta.tolist()})")
        print(f"  Sigma_K: {self.Sigma_K.shape} ({self.Sigma_K.tolist()})")
        try:
            # progress_coord = (0.1 + self.progress_coord)/1.1 if self.cfg.enable_delay_system else 1.0
            progress_coord = self.progress_coord
            Sigma_X, trace_cov = triangulation_covariance_multi_camera(
                X_w=X_w,
                robot_positions=robot_positions,
                robot_quats=robot_quats,
                gimbal_yaws=gimbal_yaws,
                gimbal_pitches=gimbal_pitches,
                camera_intrinsics=camera_intrinsics,
                Sigma_pix=self.Sigma_pix * progress_coord,
                Sigma_twb=self.Sigma_twb * progress_coord,
                Sigma_phiwb=self.Sigma_phiwb * progress_coord,
                Sigma_alpha=self.Sigma_alpha * progress_coord,
                Sigma_beta=self.Sigma_beta * progress_coord,
                Sigma_K=self.Sigma_K * progress_coord,
                include_pose=True,
                include_gimbal=True,
                include_intrinsics=True
            )
            print(f"  Sigma_X: {Sigma_X.shape} {Sigma_X}")
            print(f"  trace_cov: {trace_cov.shape} {trace_cov}")
            
            is_trace_valid = (trace_cov >= 0.0) & torch.isfinite(trace_cov)
            is_sigma_finite = torch.isfinite(Sigma_X).all(dim=(-2, -1))
            is_not_nan = ~torch.isnan(Sigma_X).any(dim=(-2, -1))
            is_all_bboxes_valid = bbox_valid_mask.all(dim=1).unsqueeze(-1)  # [N,] -> [N, 1]
            is_diag_positive = torch.diagonal(Sigma_X, dim1=-2, dim2=-1).min(dim=-1).values > 0.0 # NOTE: Temporary fix for suspicous Sigma computation
            is_tri_cov_valid = is_trace_valid & is_sigma_finite & is_not_nan & is_all_bboxes_valid & is_diag_positive
            print(f"  is_trace_valid: {is_trace_valid.tolist()}")
            print(f"  is_sigma_finite: {is_sigma_finite.tolist()}")
            print(f"  is_not_nan: {is_not_nan.tolist()}")
            print(f"  is_all_bboxes_valid: {is_all_bboxes_valid.tolist()}")
            print(f"  is_tri_cov_valid: {is_tri_cov_valid.tolist()}")
            
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
            print(f"Warning: Triangulation covariance (delayed) computation failed: {e}")
            is_tri_cov_valid = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            Sigma_X = self.Sigma_X_invalid.clone()
            trace_cov = self.trace_cov_invalid.clone()

        return Sigma_X, trace_cov, is_tri_cov_valid

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        print(f"[DEBUG] _get_rewards")
        self._compute_intermediate_values()
        rewards_dict = {}

        # ========== Compute triangulation PER AGENT (for rewards) ==========
        # v2.2 API: Use get_all_states_for_rewards() which returns Dict[agent_id -> AgentStates]
        self.triangulation_results = {}  # {agent_id: (Sigma_X, trace_cov, is_valid)}

        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            print(f"Rewards for agent {ego_agent_id}:")

            # Get all agent states from ego perspective (clean, no noise)
            all_states = self.delay_system.get_all_states_for_rewards(ego_agent_id)
            print(f"  all_states keys: {list(all_states.keys())}")

            # Build camera geometry from all agent states
            robot_positions_list = []
            robot_quats_list = []
            gimbal_yaws_list = []
            gimbal_pitches_list = []
            camera_intrinsics_list = []
            bbox_valid_list = []
            ray_origin_list = []

            # Iterate over ALL agents (ego first, then others in order)
            for agent_id in self.cfg.possible_agents:
                state = all_states[agent_id]
                robot_positions_list.append(state.data.body_position_w)  # [N, 3]
                robot_quats_list.append(state.data.body_orientation_w)  # [N, 4]
                gimbal_yaws_list.append(state.data.joint_positions_b[:, 0:1])  # [N, 1]
                gimbal_pitches_list.append(state.data.joint_positions_b[:, 1:2])  # [N, 1]
                camera_intrinsics_updated = state.data.camera_base_intrinsics.clone()
                # Apply zoom level to intrinsics
                zoom_level = state.data.camera_zoom_level.clone()  # [N,]
                camera_intrinsics_updated[:, 0, 0] *= zoom_level  # fx
                camera_intrinsics_updated[:, 1, 1] *= zoom_level  # fy
                camera_intrinsics_list.append(camera_intrinsics_updated)  # [N, 3, 3]
                # Validate bbox AFTER delay using raycaster (v2.2 pattern)
                bbox_valid = self.bbox_raycaster.validate_bbox(state.data.bboxes_2d[:, 0, :])  # [N]
                bbox_valid_list.append(bbox_valid.squeeze(-1))
                ray_origin_list.append(state.data.camera_position_w)  # [N, 3]

            # Stack into [N, C, ...] format
            robot_positions = torch.stack(robot_positions_list, dim=1)  # [N, C, 3]
            robot_quats = torch.stack(robot_quats_list, dim=1)  # [N, C, 4]
            gimbal_yaws = torch.stack(gimbal_yaws_list, dim=1)  # [N, C, 1]
            gimbal_pitches = torch.stack(gimbal_pitches_list, dim=1)  # [N, C, 1]
            camera_intrinsics = torch.stack(camera_intrinsics_list, dim=1)  # [N, C, 3, 3]
            bbox_valid_mask = torch.stack(bbox_valid_list, dim=1)  # [N, C]
            ray_origins = torch.stack(ray_origin_list, dim=1)  # [N, C, 3]
            print(f"  Agent {ego_agent_id}:")
            print(f"    bbox_valid_mask: {bbox_valid_mask.tolist()}")

            # Triangulate target position from this agent's perspective
            X_w_gt = self.target.data.root_pos_w.unsqueeze(1)  # [N, T, 3]

            # Compute triangulation covariance for THIS AGENT
            Sigma_X, trace_cov, is_tri_cov_valid = self._compute_triangulation_covariance(
                X_w=X_w_gt,
                robot_positions=ray_origins,
                robot_quats=robot_quats,
                gimbal_yaws=gimbal_yaws.squeeze(-1),
                gimbal_pitches=gimbal_pitches.squeeze(-1),
                camera_intrinsics=camera_intrinsics,
                bbox_valid_mask=bbox_valid_mask
            )
            std_dev = torch.sqrt(torch.diagonal(Sigma_X, dim1=-2, dim2=-1))  # [N, 1, 3]
            std_dev = torch.where(
                is_tri_cov_valid.unsqueeze(-1),
                std_dev,
                torch.ones_like(std_dev) * 1e6  # Large value for invalid
            )
            self.triangulation_results[ego_agent_id] = (X_w_gt, Sigma_X, std_dev, trace_cov, is_tri_cov_valid)

        # ========== Compute safety penalties for all agents ==========
        # Collect data for SafetyManager using v2.2 view scheme API
        agent_positions = {}
        agent_bboxes = {}
        agent_bbox_valid = {}
        agent_focal_lengths = {}

        for agent_id in self.cfg.possible_agents:
            # Get ego's own delayed state from view scheme
            ego_states = self.delay_system.get_all_states_for_rewards(agent_id)
            ego_state = ego_states[agent_id]  # Ego's own delayed state
            agent_positions[agent_id] = ego_state.data.body_position_w  # [N, 3]
            agent_bboxes[agent_id] = ego_state.data.bboxes_2d[:, 0, :]  # [N, 4]
            # Validate bbox AFTER delay using raycaster (v2.2 pattern)
            agent_bbox_valid[agent_id] = self.bbox_raycaster.validate_bbox(
                ego_state.data.bboxes_2d[:, 0, :]
            ).squeeze(-1)  # [N]
            # Extract fx and fy from camera intrinsics
            agent_focal_lengths[agent_id] = (
                ego_state.data.camera_base_intrinsics[:, 0, 0] * ego_state.data.camera_zoom_level,  # fx [N]
                ego_state.data.camera_base_intrinsics[:, 1, 1] * ego_state.data.camera_zoom_level,  # fy [N]
            )

        # Compute all safety penalties at once
        safety_penalties = self.safety_manager.compute_all_safety_penalties(
            agent_positions=agent_positions,
            agent_bboxes=agent_bboxes,
            agent_bbox_valid=agent_bbox_valid,
            agent_focal_lengths=agent_focal_lengths,
            agent_ids=self.cfg.possible_agents,
            dt=self.step_dt,
        )

        # ========== Compute rewards for each agent ==========
        for i, agent_id in enumerate(self.cfg.possible_agents):
            # Get ego's own delayed state from view scheme (v2.2 API)
            ego_states = self.delay_system.get_all_states_for_rewards(agent_id)
            delayed_state = ego_states[agent_id]  # Ego's own delayed state

            # Unpack triangulation results for this agent
            X_w_gt, Sigma_X, std_dev, trace_cov, is_tri_cov_valid = self.triangulation_results[agent_id]

            # Compute action penalties
            action_sum = torch.sum(torch.square(self.action_weight * self._actions[agent_id]), dim=1)
            action_delta = torch.sum(
                torch.square(self.action_delta_weight * (self._actions[agent_id] - self._last_actions[agent_id])),
                dim=1
            )

            # Bbox rewards (from delayed detections)
            bbox_center = delayed_state.data.bboxes_2d[:, 0, 0:2]  # [N, 2] (x, y)
            bbox_size = delayed_state.data.bboxes_2d[:, 0, 2:4]  # [N, 2] (w, h)
            # Validate bbox AFTER delay using raycaster (v2.2 pattern)
            bbox_valid = self.bbox_raycaster.validate_bbox(delayed_state.data.bboxes_2d[:, 0, :]).squeeze(-1)  # [N]

            # Map to rewards (centered and sized appropriately)
            bbox_center_dist = torch.norm(bbox_center - 0.5, dim=1)  # Distance from center [N] # TODO: Check if normalized
            bbox_center_mapped = torch.exp(-10.0 * bbox_center_dist) * bbox_valid.float()

            bbox_area = bbox_size[:, 0] * bbox_size[:, 1]  # [N]
            bbox_size_mapped = torch.exp(-torch.abs(bbox_area - 0.2)) * bbox_valid.float()

            # Triangulation quality reward (from this agent's perspective)
            triangulation_quality = torch.where(
                is_tri_cov_valid[:, 0],
                torch.exp(-trace_cov[:, 0]),
                torch.zeros_like(trace_cov[:, 0])
            )

            # Extract safety penalties from SafetyManager
            collision_penalty = -safety_penalties[agent_id]["collision"]  # Negate for penalty
            ttc_penalty = safety_penalties[agent_id]["ttc_penalty"]

            # Combine rewards
            rewards = {
                "action_sum": action_sum * self.cfg.action_sum_penalty_scale * self.step_dt,
                "action_delta": action_delta * self.cfg.action_delta_penalty_scale * self.step_dt,
                "bbox_center": bbox_center_mapped * self.cfg.bbox_center_reward_scale * self.step_dt,
                "bbox_size": bbox_size_mapped * self.cfg.bbox_size_reward_scale * self.step_dt,
                "triangulation": triangulation_quality * self.cfg.triangulation_reward_scale * self.step_dt * self.progress_coord,
                "collision": collision_penalty * self.cfg.collision_penalty_scale * self.step_dt * self.progress_coord,
                "ttc": ttc_penalty * self.cfg.ttc_penalty_scale * self.step_dt * self.progress_coord,
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

            if torch.isnan(std_dev).any():
                print(f"std_dev(reward): {std_dev}")
                nan_envs = torch.nonzero(torch.isnan(std_dev).any(dim=(-2, -1))).flatten()
                raise ValueError(f"NaN detected in {agent_id} triangulation std_dev(reward) at environments: {nan_envs.tolist()}")

            rewards_dict[agent_id] = total_reward

            # Update last actions
            self._last_actions[agent_id] = self._actions[agent_id].clone()
            self.last_cmd_vel[:, i, :] = self.cmd_vel[:, i, :].clone()

        return rewards_dict

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents using delayed+noisy states (v2.2 API)."""
        print(f"[DEBUG] _get_observations")

        # ========== Compute triangulation PER AGENT (for observations) ==========
        # v2.2 API: Use get_all_states_for_observations() which returns Dict[agent_id -> AgentStates]
        self.triangulation_results_noisy = {}

        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            # Get all agent states from ego perspective (noisy pipeline)
            all_states = self.delay_system.get_all_states_for_observations(ego_agent_id)

            # Build camera geometry from all agent states
            robot_positions_list = []
            robot_quats_list = []
            gimbal_yaws_list = []
            gimbal_pitches_list = []
            camera_intrinsics_list = []
            bbox_valid_list = []
            ray_origin_list = []
            ray_dirs_list = []

            # Iterate over ALL agents
            for agent_id in self.cfg.possible_agents:
                state = all_states[agent_id]
                robot_positions_list.append(state.data.body_position_w)  # [N, 3]
                robot_quats_list.append(state.data.body_orientation_w)  # [N, 4]
                gimbal_yaws_list.append(state.data.joint_positions_b[:, 0:1])  # [N, 1]
                gimbal_pitches_list.append(state.data.joint_positions_b[:, 1:2])  # [N, 1]
                camera_intrinsics_updated = state.data.camera_base_intrinsics.clone()
                # Apply zoom level to intrinsics
                zoom_level = state.data.camera_zoom_level.clone()  # [N,]
                camera_intrinsics_updated[:, 0, 0] *= zoom_level  # fx
                camera_intrinsics_updated[:, 1, 1] *= zoom_level  # fy
                camera_intrinsics_list.append(camera_intrinsics_updated)  # [N, 3, 3]
                # Validate bbox AFTER delay using raycaster (v2.2 pattern)
                bbox_valid = self.bbox_raycaster.validate_bbox(state.data.bboxes_2d[:, 0, :])  # [N, 1]
                print(f"  Agent {ego_agent_id} sees bbox from {agent_id}: {state.data.bboxes_2d[:, 0, :].tolist()}, valid: {bbox_valid.squeeze(-1).tolist()}")
                bbox_valid_list.append(bbox_valid.squeeze(-1))
                ray_origin_list.append(state.data.camera_position_w)  # [N, 3]
                ray_dirs_list.append(state.data.camera_ray_directions_w[:, 0, :])  # [N, 3]

            # Stack
            robot_positions = torch.stack(robot_positions_list, dim=1)  # [N, C, 3]
            robot_quats = torch.stack(robot_quats_list, dim=1)  # [N, C, 4]
            gimbal_yaws = torch.stack(gimbal_yaws_list, dim=1)  # [N, C, 1]
            gimbal_pitches = torch.stack(gimbal_pitches_list, dim=1)  # [N, C, 1]
            camera_intrinsics = torch.stack(camera_intrinsics_list, dim=1)  # [N, C, 3, 3]
            bbox_valid_mask = torch.stack(bbox_valid_list, dim=1)  # [N, C]
            ray_origins = torch.stack(ray_origin_list, dim=1)  # [N, C, 3]
            ray_dirs = torch.stack(ray_dirs_list, dim=1)  # [N, C, 3]

            # Triangulate from noisy perspective
            # midpoint_method_batched expects pts [N, C, 3] and dirs [N, C, T, 3]
            # We need to add a target dimension to ray_dirs
            ray_dirs_expanded = ray_dirs.unsqueeze(2)  # [N, C, 1, 3]
            X_w_triangulated, is_triangulation_valid = midpoint_method_batched(
                pts=ray_origins,
                dirs=ray_dirs_expanded,
                valid_mask=bbox_valid_mask  # [N, C] - exclude invalid detections from triangulation
            )  # X_w_triangulated: [N, 1, 3], is_triangulation_valid: [N, 1]

            # Compute covariance
            Sigma_X, trace_cov, is_cov_valid = self._compute_triangulation_covariance(
                X_w=X_w_triangulated,
                robot_positions=ray_origins,
                robot_quats=robot_quats,
                gimbal_yaws=gimbal_yaws.squeeze(-1),
                gimbal_pitches=gimbal_pitches.squeeze(-1),
                camera_intrinsics=camera_intrinsics,
                bbox_valid_mask=bbox_valid_mask
            )

            # Combine validity flags: triangulation valid AND covariance valid
            is_valid = is_triangulation_valid & is_cov_valid  # [N, 1]

            # Compute standard deviations
            std_dev = torch.sqrt(torch.diagonal(Sigma_X, dim1=-2, dim2=-1))  # [N, 1, 3]
            std_dev = torch.where(
                is_valid.unsqueeze(-1),
                std_dev,
                torch.ones_like(std_dev) * 1e6  # Large value for invalid
            )

            self.triangulation_results_noisy[ego_agent_id] = (X_w_triangulated, Sigma_X, std_dev, trace_cov, is_valid)

        # ========== Build observations for each agent ==========
        observations = {}

        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            # v2.2 API: Get all states for this ego agent
            all_states = self.delay_system.get_all_states_for_observations(ego_agent_id)
            ego_state = all_states[ego_agent_id]  # Ego's own noisy delayed state

            X_w_tri, Sigma_X, std_tri, trace_cov, is_tri_cov_valid = self.triangulation_results_noisy[ego_agent_id]
            # std_tri = torch.sqrt(torch.diagonal(Sigma_X, dim1=-2, dim2=-1))  # [N, 1, 3]
            # std_tri = torch.where(
            #     is_tri_cov_valid.unsqueeze(-1),
            #     std_tri,
            #     torch.ones_like(std_tri) * 1e6  # Large value for invalid
            # )

            # Extract ego observations from noisy delayed state
            pos = ego_state.data.body_position_w  # [N, 3]
            roll, pitch, yaw = euler_xyz_from_quat(ego_state.data.body_orientation_w)  # Each [N]
            yaw = yaw.unsqueeze(-1)  # [N, 1]
            lin_vel = ego_state.data.body_linear_velocity_w  # [N, 3]
            yaw_rate = ego_state.data.body_angular_velocity_w[:, 2:3]  # [N, 1]
            lin_acc = ego_state.data.body_linear_acceleration_w  # [N, 3]
            gimbal_pitch = ego_state.data.joint_positions_b[:, 1:2]  # [N, 1]
            gimbal_yaw = ego_state.data.joint_positions_b[:, 0:1]  # [N, 1]
            combined_ang_vel = ego_state.data.body_combined_angular_velocity_w  # [N, 3]

            # Detection observations - validate bbox using raycaster (v2.2 pattern)
            bbox = ego_state.data.bboxes_2d[:, 0, :]  # [N, 4]
            bbox_valid = self.bbox_raycaster.validate_bbox(bbox).unsqueeze(-1)  # [N, 1]
            # Compute time_since_detection from timestamps (v2.2: current_time - detection_timestamp)
            # current_time and timestamp_detection are both [N] tensors (per-environment)
            time_since_detection = (self.delay_system.current_time - ego_state.data.timestamp_detection).unsqueeze(-1)  # [N, 1]
            zoom = ego_state.data.camera_zoom_level.unsqueeze(-1)  # [N, 1]
            ray_dir = ego_state.data.camera_ray_directions_w[:, 0, :]  # [N, 3]

            # Received states from other agents (v2.2: all states already in all_states dict)
            other_positions = []
            other_lin_vels = []
            other_combined_ang_vels = []
            other_bbox_valids = []
            other_data_age = []  # Age of position/velocity data (for collision/TTC safety)
            other_detection_age = []  # Age of bbox detection
            other_ray_dirs = []

            for other_agent_id in self.cfg.possible_agents:
                if other_agent_id == ego_agent_id:
                    continue

                # v2.2: All agent states are available in all_states dict
                other_state = all_states[other_agent_id]
                other_positions.append(other_state.data.body_position_w)  # [N, 3]
                other_lin_vels.append(other_state.data.body_linear_velocity_w)  # [N, 3]
                other_combined_ang_vels.append(other_state.data.body_combined_angular_velocity_w)  # [N, 3]
                # Validate bbox using raycaster (v2.2 pattern)
                other_bbox = other_state.data.bboxes_2d[:, 0, :]  # [N, 4]
                other_bbox_valids.append(self.bbox_raycaster.validate_bbox(other_bbox).unsqueeze(-1))  # [N, 1]
                # Compute detection age from timestamps (v2.2: current_time - timestamp_detection)
                detection_age = (self.delay_system.current_time - other_state.data.timestamp_detection).unsqueeze(-1)  # [N, 1]
                other_data_age.append(detection_age)
                other_detection_age.append(detection_age)
                other_ray_dirs.append(other_state.data.camera_ray_directions_w[:, 0, :])  # [N, 3]

            # Concatenate all observations
            obs = torch.cat([
                pos,  # 3
                yaw,  # 1
                lin_vel,  # 3
                yaw_rate,  # 1
                lin_acc,  # 3
                gimbal_pitch,  # 1
                gimbal_yaw,  # 1
                combined_ang_vel,  # 3
                bbox,  # 4 ## 20
                bbox_valid,  # 1
                time_since_detection,  # 1
                zoom,  # 1
                ray_dir,  # 3 ## 26
                *other_positions,  # 3 * (C-1)
                *other_lin_vels,  # 3 * (C-1)
                *other_combined_ang_vels,  # 3 * (C-1)
                *other_bbox_valids,  # 1 * (C-1) ## 26 + 10*(C-1)
                *other_ray_dirs,  # 3 * (C-1) ## 26 + 15*(C-1)
                *other_data_age,  # 1 * (C-1) (state data age for collision/TTC safety)
                *other_detection_age,  # 1 * (C-1) (bbox detection age)
                X_w_tri[:, 0, :],  # 3*T (triangulation estimate)
                std_tri[:, 0, :],  # 3*T (triangulation std deviations)
            ], dim=-1) ## 26 + 15*(C-1) + 6*T

            # Check for NaN values
            if torch.isnan(obs).any():
                std_dev_reward = self.triangulation_results[ego_agent_id][2]
                std_dev_obs = self.triangulation_results_noisy[ego_agent_id][2]
                print(f"std_dev_reward for {ego_agent_id}: {std_dev_reward.tolist()}")
                print(f"std_dev_obs for {ego_agent_id}: {std_dev_obs.tolist()}")
                nan_envs = torch.nonzero(torch.isnan(obs).any(dim=1)).flatten()
                raise ValueError(f"NaN detected in {ego_agent_id} observation at environments: {nan_envs.tolist()}")

            observations[ego_agent_id] = obs

        return observations

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

    def _build_initial_gt_states_for_reset(self, env_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Build initial GT states dict for delay system reset.

        This creates a minimal structure that matches what delay_system.reset() expects:
        - Dict[agent_id, object with .data attribute]
        - The .data object has fields like body_position_w, body_orientation_w, etc.

        This is used to properly initialize FirstOrderLag samplers with actual GT values
        instead of zeros, avoiding the initial transient convergence period.

        Args:
            env_ids: Indices of environments being reset

        Returns:
            Dict[agent_id, InitialStateWrapper] containing initial GT values
        """
        from types import SimpleNamespace

        initial_gt_states = {}

        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]

            # Create a namespace to hold initial data
            data = SimpleNamespace()

            # Motion states (from robot root state)
            data.body_position_w = robot.data.root_pos_w.clone()
            data.body_linear_velocity_w = robot.data.root_lin_vel_w.clone()
            data.body_linear_velocity_b = robot.data.root_lin_vel_b.clone()
            data.body_angular_velocity_w = robot.data.root_ang_vel_w.clone()
            data.body_angular_velocity_b = robot.data.root_ang_vel_b.clone()
            # Combined angular velocity (body + gimbal) - init to body angular velocity
            data.body_combined_angular_velocity_w = robot.data.root_ang_vel_w.clone()
            data.body_combined_angular_velocity_b = robot.data.root_ang_vel_b.clone()
            # Accelerations - init to zeros (will be computed from dynamics)
            data.body_linear_acceleration_w = torch.zeros_like(robot.data.root_pos_w)
            data.body_linear_acceleration_b = torch.zeros_like(robot.data.root_pos_w)
            data.body_angular_acceleration_b = torch.zeros_like(robot.data.root_pos_w)

            # Orientation states (from robot quaternion)
            data.body_orientation_w = robot.data.root_quat_w.clone()
            # Camera orientation - use body orientation as initial (will be updated by derived fields)
            data.camera_orientation_w = robot.data.root_quat_w.clone()

            # Joint states - only include gimbal joints (yaw, roll, pitch)
            # The delay system was configured with num_joints=3 for the gimbal
            gimbal_joint_ids = [
                self.gimbal_joint_idx[agent_id]["yaw"],
                self.gimbal_joint_idx[agent_id]["roll"],
                self.gimbal_joint_idx[agent_id]["pitch"],
            ]
            data.joint_positions_b = robot.data.joint_pos[:, gimbal_joint_ids].clone()
            data.joint_velocities_b = robot.data.joint_vel[:, gimbal_joint_ids].clone()
            data.joint_accelerations_b = torch.zeros(self.num_envs, 3, device=self.device)

            # Zoom level - shape [N, 1] to match sampler expectation
            data.camera_zoom_level = torch.ones(self.num_envs, 1, device=self.device)

            # Wrap in a SimpleNamespace with .data attribute
            wrapper = SimpleNamespace(data=data)
            initial_gt_states[agent_id] = wrapper

        return initial_gt_states

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at specified indices."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[self.cfg.possible_agents[0]]._ALL_INDICES

        # NOTE: delay_system.reset() is called AFTER robot states are written (see below)
        # This allows us to initialize FirstOrderLag samplers with actual GT values

        # Reset dynamics filters
        for agent_id in self.cfg.possible_agents:
            self._robot_dynamics[agent_id].reset(env_ids=env_ids)

        # Reset SafetyManager (collision detector and TTC computer)
        self.safety_manager.reset(env_ids=env_ids)

        # Reset last actions and cmd_vel
        for agent_id in self.cfg.possible_agents:
            self._last_actions[agent_id][env_ids].zero_()
        self.last_cmd_vel[env_ids].zero_()
        self.zoom_level[env_ids].fill_(1.0)

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

        # =====================================================================
        # Gimbal-Aware Formation and Target Sampling
        # =====================================================================
        # Step 1: Generate random formation (positions, orientations, velocities)
        formation_data = self.randomizer.initial_states.get_random_formation(
            num_agents=len(self.cfg.possible_agents),
            num_envs=len(env_ids),
            scale_factor=1.0,  # TODO: Hook to curriculum
        )
        # formation_data shape: [len(env_ids), num_agents, 13]
        #   [:, :, 0:3] - positions
        #   [:, :, 3:7] - orientations (quaternions)
        #   [:, :, 7:13] - velocities

        # Step 2: Sample feasible target positions (guaranteed gimbal feasibility)
        target_positions = self.randomizer.targets.sample_target_position(
            formation_data=formation_data,
            scale_factor=1.0,  # TODO: Hook to curriculum
        )
        # target_positions shape: [len(env_ids), 3]
        # NOTE: OVERRIDE FOR TESTING
        target_positions = torch.tensor([10.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)

        # Step 3: Apply terrain origins
        terrain_offset = self._terrain.env_origins[env_ids]

        # Offset target positions
        target_pos = target_positions + terrain_offset

        # Step 4: Set target states
        target_state = self.target.data.default_root_state[env_ids].clone()
        target_state[:, 0:3] = target_pos
        target_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.rand(len(env_ids), device=self.device) * 2 * math.pi  # Random yaw
        )
        self.target.write_root_pose_to_sim(target_state[:, :7], env_ids)
        self.target.write_root_velocity_to_sim(target_state[:, 7:], env_ids)

        # Step 5: Set agent states and compute gimbal angles
        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Extract agent state from formation
            agent_positions = formation_data[:, i, 0:3]     # [len(env_ids), 3]
            agent_orientations = formation_data[:, i, 3:7]  # [len(env_ids), 4]
            agent_velocities = formation_data[:, i, 7:13]   # [len(env_ids), 6]

            # # NOTE: OVERRIDE FOR TESTING
            agent_positions = torch.tensor([0.0, i * 5.0, 1.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
            agent_orientations = quat_from_euler_xyz(
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device)
            )
            agent_velocities = torch.zeros(len(env_ids), 6, device=self.device)

            # Construct root state
            agent_root_state = robot.data.default_root_state[env_ids].clone()
            agent_root_state[:, 0:3] = agent_positions + terrain_offset
            agent_root_state[:, 3:7] = agent_orientations
            agent_root_state[:, 7:13] = agent_velocities

            # Set robot root state
            robot.write_root_pose_to_sim(agent_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(agent_root_state[:, 7:13], env_ids)

            # Compute gimbal angles (now guaranteed feasible)
            gimbal_yaw, gimbal_roll, gimbal_pitch = \
                self._gimbal_stabilizers[agent_id].compute_stabilized_angles_from_target_point(
                    target_point_world=target_pos,
                    drone_position_world=agent_root_state[:, 0:3],
                    drone_quat_world=agent_orientations
                )

            # Set gimbal joint positions
            gimbal_joint_ids = [
                    self.gimbal_joint_idx[agent_id]["yaw"],
                    self.gimbal_joint_idx[agent_id]["roll"],
                    self.gimbal_joint_idx[agent_id]["pitch"],
                ]
            joint_pos = torch.stack([gimbal_yaw, gimbal_pitch, gimbal_roll], dim=-1)
            joint_vel = robot.data.default_joint_vel[env_ids][:,gimbal_joint_ids].clone()

            robot.write_joint_state_to_sim(
                position=joint_pos,
                velocity=joint_vel,
                joint_ids=gimbal_joint_ids,
                env_ids=env_ids
            )

        # =====================================================================
        # Reset Delay System with Initial GT States
        # =====================================================================
        # Build initial GT states from robot data for proper FirstOrderLag initialization
        # This prevents the initial transient where filtered values slowly converge from zeros
        initial_gt_states = self._build_initial_gt_states_for_reset(env_ids)
        self.delay_system.reset(env_ids=env_ids, initial_gt_states=initial_gt_states)

        # eye=self._robots[self.cfg.possible_agents[0]].data.root_pos_w[env_ids[0]].tolist()+[5.0, 5.0, 5.0]
        # lookat=self.target.data.root_pos_w[env_ids[0]].tolist()
        # self.viewport_camera_controller.update_view_location(
        #     eye=self._robots[self.cfg.possible_agents[0]].data.root_pos_w[env_ids[0]].tolist(),
        #     lookat=self.target.data.root_pos_w[env_ids[0]].tolist()
        # )

        super()._reset_idx(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not DEBUG_DRAW:
            return

        self.visualization.camera_frustum[self.cfg.possible_agents[0]].draw_interface.clear_lines()
        self.visualization.detection_indicator[self.cfg.possible_agents[0]].draw_interface.clear_lines()
        # print(f"Triangulation: ")
        for i, agent_id in enumerate(self.cfg.possible_agents):
            camera_intrinsics_updated = self.delay_system.gt_states.agents[agent_id].data.camera_base_intrinsics.clone()
            camera_intrinsics_updated[:, 0, 0] *= self.zoom_level[:, i]
            camera_intrinsics_updated[:, 1, 1] *= self.zoom_level[:, i]
            self._cameras[agent_id].set_intrinsic_matrices_batched(
                camera_intrinsics_updated, 
                self.cfg.camera.spawn.focal_length * self.zoom_level[:, i]
            )
            
            self.visualization.camera_frustum[agent_id].draw_frustum(
                camera_position=self.delay_system.gt_states.agents[agent_id].data.camera_position_w,
                camera_orientation=self.delay_system.gt_states.agents[agent_id].data.camera_orientation_w,
                camera_intrinsics=camera_intrinsics_updated,
                camera_cfg=self.cfg.camera,
                zoom_level=self.delay_system.gt_states.agents[agent_id].data.camera_zoom_level,
                device=self.device
            )

            # Visualize GT bboxes using raycaster
            # gt_bbox = self.delay_system.gt_states.agents[agent_id].data.bboxes_2d[:, 0, :]  # [N, 4]
            # gt_bbox_valid = self.bbox_raycaster.validate_bbox(gt_bbox)  # [N, 1]
            # self.visualization.detection_indicator[agent_id].draw_indicator(
            #     start_points=self.delay_system.gt_states.agents[agent_id].data.camera_position_w,
            #     end_points=self.target.data.root_pos_w,
            #     detected=gt_bbox_valid.squeeze(-1)  # [N]
            # )

            # Visualize delayed bboxes using raycaster
            delayed_states = self.delay_system.get_all_states_for_observations(agent_id)
            delayed_bbox = delayed_states[agent_id].data.bboxes_2d[:, 0, :]  # [N, 4]
            delayed_bbox_valid = self.bbox_raycaster.validate_bbox(delayed_bbox)  # [N, 1]
            self.visualization.detection_indicator[agent_id].draw_indicator(
                start_points=delayed_states[agent_id].data.camera_position_w,
                end_points=self.target.data.root_pos_w,
                detected=delayed_bbox_valid.squeeze(-1)  # [N]
            )

            # Either visualize using GT+delayed states or noisy+delayed states
            # X_w_tri, Sigma_X, std_dev, trace_cov, is_tri_cov_valid = self.triangulation_results[agent_id]
            X_w_tri, Sigma_X, std_dev, trace_cov, is_tri_cov_valid = self.triangulation_results_noisy[agent_id]
            valid_envs = torch.nonzero(
                is_tri_cov_valid.squeeze(-1), as_tuple=False
            ).squeeze(-1).tolist()  # python list of env ids with valid tri cov

            if len(valid_envs) > 0:
                scale = torch.sqrt(torch.clamp(std_dev, min=0.0)) * 2.5
                self.visualization.tri_cov_visualizer[agent_id].visualize(
                    translations=X_w_tri[valid_envs, 0, :], # [N, 3]
                    scales=scale[valid_envs, 0, :], # [N, 3]
                )
                # print(f" Agent {agent_id}:")
                # print(f"  Estimated Position: {X_w_tri[0,0,:].cpu().numpy()}")
                # print(f"  Std Deviations: {std_tri[0,0,:].cpu().numpy()}")

                # Visualize camera ray through bbox center noisy+delay
                states = self.delay_system.get_all_states_for_observations(agent_id)
                bbox = states[agent_id].data.bboxes_2d[:, 0, :]  # [N, 4]
                bbox_valid = self.bbox_raycaster.validate_bbox(bbox)  # [N]
                self.visualization.detection_indicator[agent_id].draw_indicator(
                    start_points=states[agent_id].data.camera_position_w[valid_envs],
                    end_points=X_w_tri[valid_envs, 0, :],
                    detected=bbox_valid[valid_envs]  # [N]
                )
            else:
                self.visualization.tri_cov_visualizer[agent_id].visualize(
                    translations=torch.zeros((0, 3), device=self.device), # [N, 3]
                    scales=torch.zeros((0, 3), device=self.device), # [N, 3]
                )
