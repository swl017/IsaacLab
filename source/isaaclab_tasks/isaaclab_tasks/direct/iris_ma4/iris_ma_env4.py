# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-Agent Iris Environment V4.

This environment uses the DelaySystemV2 architecture adapted from the quadcopter
delay system, providing:
- Field-based delay pipeline with DataBus architecture
- Dual pipeline (clean for rewards, noisy for observations)
- Perspective-aware delays (fast ego, slow inter-agent)
- Scalability to arbitrary number of agents
"""

from __future__ import annotations

import gymnasium as gym
import torch
import math
import numpy as np
import copy
from typing import Dict, Optional, Any

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

# Configuration
from isaaclab_tasks.direct.iris_ma4.iris_ma_env4_cfg import IrisMAEnvCfg

# BBox raycaster (from iris_ma4 or iris_ma3)
from isaaclab_tasks.direct.iris_ma4.bbox_raycaster import BBoxRayCaster, BBoxRayCasterCfg

# Triangulation utilities
from isaaclab_tasks.direct.iris_ma4.triangulation.triang_cov_reward_torch import (
    triangulation_covariance_multi_camera,
    triangulation_covariance_simple,
    midpoint_method_batched,
    get_ray_dir_from_bbox
)

# NEW: Import DelaySystemV2 wrapper instead of original delay system
from isaaclab_tasks.direct.iris_ma4.delay_system_v2 import (
    MultiAgentDelaySystemV2,
    MultiAgentDelaySystemV2Cfg,
)

# Controller and safety modules
from isaaclab_tasks.direct.iris_ma4.controller import PointMass, PointMassCfg, GimbalStabilizer, GimbalStabilizerCfg
from isaaclab_tasks.direct.iris_ma4.safety import SafetyManager, SafetyManagerCfg, CollisionDetectorCfg, TTCComputerCfg
from isaaclab_tasks.direct.iris_ma4.randomization import Randomizer, DistanceBasedFormationCfg
from isaaclab_tasks.direct.iris_ma4.visualization import CustomVisualization
from isaaclab_tasks.direct.iris_ma4.target_movement import TargetMovement

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


class IrisMAEnvV4(DirectMARLEnv):
    """Multi-Agent Iris Environment V4 using DelaySystemV2 architecture."""

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
        C = len(cfg.possible_agents)  # C as in "Cameras"
        T = 1  # number of targets

        # Get body and joint indices for each robot (AFTER super().__init__())
        self._body_ids = {}
        self.gimbal_joint_idx = {}
        self._robot_mass = {}
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
            robot_weight = (self._robot_mass[agent] * torch.tensor(self.sim.cfg.gravity, device=self.device).norm()).item()
            self._robot_controllers[agent] = PointMass(
                cfg=self.cfg.point_mass,
                mass=self._robot_mass[agent],
                num_envs=self.num_envs,
                device=self.device,
                weight=robot_weight,
                disable_gravity=self.cfg.robot.spawn.rigid_props.disable_gravity,
            )
            self._gimbal_stabilizers[agent] = GimbalStabilizer(
                cfg=self.cfg.gimbal,
                device=self.device,
            )

        # Initialize GT action variables
        self._actions = {agent_id: torch.zeros(self.num_envs, len(self.cfg.action_weight), device=self.device) for agent_id in cfg.possible_agents}
        self._last_actions = {agent_id: torch.zeros(self.num_envs, len(self.cfg.action_weight), device=self.device) for agent_id in cfg.possible_agents}
        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device)

        self.cmd_vel = torch.zeros(self.num_envs, len(self.cfg.possible_agents), len(self.cfg.action_weight), device=self.device)  # [N, C, 7]
        self.last_cmd_vel = torch.zeros_like(self.cmd_vel)  # [N, C, 7]
        self.cmd_gimbal_yaw = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)  # [N, C]
        self.cmd_gimbal_pitch = torch.zeros(self.num_envs, len(self.cfg.possible_agents), device=self.device)  # [N, C]
        self.zoom_level = torch.ones(self.num_envs, len(self.cfg.possible_agents), device=self.device)  # [N, C]

        # =====================================================================
        # Initialize MultiAgentDelaySystemV2 (NEW: using V2 architecture)
        # =====================================================================
        delay_cfg = MultiAgentDelaySystemV2Cfg(
            dt=self.step_dt,
            # Motion filtering
            motion_time_constant=cfg.motion_time_constant,
            orientation_time_constant=cfg.motion_time_constant,
            joint_time_constant=cfg.gimbal_time_constant,
            zoom_time_constant=0.1,
            # Detection
            detection_fps_mean=cfg.detection_fps,
            detection_fps_std=5.0,
            detection_latency_mean=cfg.detection_mean_latency,
            detection_latency_std=cfg.detection_std_latency,
            detection_dropout_rate=cfg.detection_failure_rate,
            # Ego communication (fast local processing)
            ego_comm_time_constant=0.005,
            # Inter-agent communication (slow network)
            inter_agent_comm_fps_mean=30.0,
            inter_agent_comm_fps_std=5.0,
            inter_agent_comm_latency_mean=cfg.comm_mean_latency,
            inter_agent_comm_latency_std=cfg.comm_std_latency,
            inter_agent_comm_dropout_rate=cfg.comm_dropout_rate,
            # Noise parameters
            enable_noise=cfg.enable_noise_in_observations,
            position_noise_std=cfg.pos_std,
            orientation_noise_std=cfg.ori_std,
            linear_velocity_noise_std=0.1 * cfg.pos_std,
            angular_velocity_noise_std=cfg.ori_std * cfg.pos_std,
            linear_acceleration_noise_std=100e-6,
            gimbal_noise_std=cfg.gimbal_std,
            bbox_noise_std=cfg.pix_std,
            zoom_noise_std=0.1,
        )

        self.delay_system = MultiAgentDelaySystemV2(
            cfg=delay_cfg,
            possible_agents=cfg.possible_agents,
            num_envs=self.num_envs,
            num_joints_per_agent={agent: 3 for agent in cfg.possible_agents},  # yaw, pitch, roll
            num_targets_per_agent={agent: 1 for agent in cfg.possible_agents},
            device=self.device,
        )

        # Set camera config for delay system
        width = self.cfg.camera.width
        height = self.cfg.camera.height
        focal_length = self.cfg.camera.spawn.focal_length
        horizontal_aperture = self.cfg.camera.spawn.horizontal_aperture
        vertical_aperture = horizontal_aperture * (height / width)

        # Cache image dimensions for bbox normalization
        self._img_dims = torch.tensor([width, height], device=self.device, dtype=torch.float32)

        # Get camera offset from config
        offset_pos = torch.tensor(self.cfg.camera.offset.pos, device=self.device)
        offset_rot = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self.device)

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
                    horizon_sec=self.cfg.ttc_horizon,
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

        # Randomizer for initial states (distance-based formation generation)
        # Primary curriculum factor is distance to target
        formation_cfg = DistanceBasedFormationCfg(
            distance_min=10.0,   # Easy: 10m at scale_factor=0
            distance_max=80.0,   # Hard: 80m at scale_factor=1
            min_agent_separation=5.0,
            max_agent_separation=30.0,
            formation_height_min=10.0,
            formation_height_max=30.0,
            target_height_offset_min=-2.0,
            target_height_offset_max=2.0,
            z_variation_min=0.0,
            z_variation_max=3.0,
            line_max_z_component=0.3,
        )
        self.randomizer = Randomizer(
            num_envs=self.num_envs,
            num_agents=len(cfg.possible_agents),
            device=self.device,
            cfg=formation_cfg,
            max_lin_vel=self.cfg.max_lin_vel,
            max_yaw_rate=self.cfg.max_yaw_rate,
        )

        # Curriculum parameters
        current_step = self.cfg.play_sim_at_step if DEBUG_DRAW else self.common_step_counter
        curr = self.cfg.curriculum
        self.progress_all = self._linear_progress(0, curr.all_end_step, current_step)
        self.progress_delay = self._linear_progress(curr.delay_start_step, curr.delay_end_step, current_step)
        self.progress_tracking = self._linear_progress(curr.tracking_start_step, curr.tracking_end_step, current_step)
        self.progress_coord = 1.0 if DEBUG_DRAW else self._linear_progress(curr.coordination_start_step, curr.coordination_end_step, current_step)
        self.progress_safety = self._linear_progress(curr.safety_start_step, curr.safety_end_step, current_step)
        self.progress_move = self._linear_progress(curr.moving_target_start_step, curr.moving_target_end_step, current_step)
        self.progress_dynamics = self._linear_progress(curr.dynamics_start_step, curr.dynamics_end_step, current_step)
        self.current_max_zoom = curr.get_max_zoom_level(current_step)

        # Triangulation covariance placeholders
        self.Sigma_X_invalid = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, 1, 3, 3
        ) * (-1.0)
        self.trace_cov_invalid = torch.ones(self.num_envs, 1, device=self.device) * (-1.0)

        # Uncertainty matrices for triangulation covariance computation
        # Use .repeat() instead of .expand() so each env can have independent values
        # (expand creates views that share memory, repeat creates actual copies)
        C = len(self.cfg.possible_agents)
        self.Sigma_pix = (
            torch.eye(2, device=self.device) * (self.cfg.pix_std ** 2)
        ).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, C, 1, 1)
        self.Sigma_twb = (
            torch.eye(3, device=self.device) * (self.cfg.pos_std ** 2)
        ).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, C, 1, 1)
        self.Sigma_phiwb = (
            torch.eye(3, device=self.device) * (self.cfg.ori_std ** 2)
        ).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, C, 1, 1)
        self.Sigma_alpha = torch.ones(self.num_envs, C, 1, 1, device=self.device) * (self.cfg.gimbal_std ** 2)
        self.Sigma_beta = torch.ones(self.num_envs, C, 1, 1, device=self.device) * (self.cfg.gimbal_std ** 2)
        self.Sigma_K = (
            torch.eye(4, device=self.device) * (self.cfg.intrinsics_std ** 2)
        ).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, C, 1, 1)

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

        # FIX: Detection dropout tracking for debugging
        # Tracks per-episode statistics on detection failures
        self._detection_stats = {
            "total_steps": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "valid_triangulations": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "invalid_all_agents": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),  # Steps where ALL agents had detection failures
            "pair_valid_count": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),  # Steps with at least 2 valid detections
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

        # Create target movement controller
        self.target_movement = TargetMovement(
            cfg=self.cfg.target_movement,
            num_envs=self.num_envs,
            device=self.device,
        )

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
        curr = self.cfg.curriculum
        self.progress_all = self._linear_progress(0, curr.all_end_step, current_step)
        self.progress_delay = self._linear_progress(curr.delay_start_step, curr.delay_end_step, current_step)
        self.progress_tracking = self._linear_progress(curr.tracking_start_step, curr.tracking_end_step, current_step)
        self.progress_coord = 1.0 if DEBUG_DRAW else self._linear_progress(curr.coordination_start_step, curr.coordination_end_step, current_step)
        self.progress_safety = self._linear_progress(curr.safety_start_step, curr.safety_end_step, current_step)
        self.progress_move = self._linear_progress(curr.moving_target_start_step, curr.moving_target_end_step, current_step)
        self.progress_dynamics = self._linear_progress(curr.dynamics_start_step, curr.dynamics_end_step, current_step)
        self.current_max_zoom = curr.get_max_zoom_level(current_step)

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            # Clip and store actions
            action = torch.clamp(actions[agent_id], min=-1.0, max=1.0)
            self._actions[agent_id] = action.clone()

            # Scale actions
            # [0 vx, 1 vy, 2 vz, 3 yaw_rate, 4 gimbal_yaw_rate, 5 gimbal_pitch_rate, 6 zoom_rate]
            self.cmd_vel[:, idx, 0:2] = action[:, 0:2] * self.cfg.max_lin_vel
            self.cmd_vel[:, idx, 2] = action[:, 2] * self.cfg.max_lin_vel / 2.0
            self.cmd_vel[:, idx, 3] = action[:, 3] * self.cfg.max_yaw_rate
            self.cmd_vel[:, idx, 4:6] = action[:, 4:6] * self.cfg.gimbal.max_rate
            self.cmd_vel[:, idx, 6] = action[:, 6] * self.cfg.max_zoom_rate
            """ Debug logs """
            # carb.log_warn(f"cmd_vel:\n{self.cmd_vel[:, idx, :].cpu().numpy()}")

            # Get current gimbal angles
            robot = self._robots[agent_id]
            gimbal_yaw = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]].clone()
            gimbal_pitch = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]].clone()

            # Integrate gimbal rate commands to get gimbal angle targets
            # cmd_vel[:, idx, 4] = gimbal_yaw_rate, cmd_vel[:, idx, 5] = gimbal_pitch_rate
            self.cmd_gimbal_yaw[:, idx] = torch.clamp(self.cmd_gimbal_yaw[:, idx] + self.cmd_vel[:, idx, 4] * self.step_dt, min=self.cfg.gimbal.yaw_limits[0], max=self.cfg.gimbal.yaw_limits[1])
            self.cmd_gimbal_pitch[:, idx] = torch.clamp(self.cmd_gimbal_pitch[:, idx] + self.cmd_vel[:, idx, 5] * self.step_dt, min=self.cfg.gimbal.pitch_limits[0], max=self.cfg.gimbal.pitch_limits[1])

    def _apply_action(self):
        """Apply actions to the environment. Runs at simulation rate, not at decimated rate."""
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
            )

            robot.set_external_force_and_torque(
                forces=force.unsqueeze(1),
                torques=moment.unsqueeze(1),
                body_ids=self._body_ids[agent_id]
            )

            # Gimbal control
            gimbal_yaw = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]].clone()
            gimbal_pitch = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]].clone()

            gimbal_roll_stabilizing = self._gimbal_stabilizers[agent_id].compute_stabilizing_roll(gimbal_yaw, gimbal_pitch, robot.data.root_quat_w)

            # Gimbal joint position targets using [pitch, yaw, roll] convention
            # This aligns with compute_camera_orientation_from_gimbal() expectations
            robot.set_joint_position_target(
                target=torch.stack([
                    self.cmd_gimbal_pitch[:, idx],   # pitch target → pitch joint
                    self.cmd_gimbal_yaw[:, idx],     # yaw target → yaw joint
                    gimbal_roll_stabilizing          # roll stabilizing → roll joint
                ], dim=-1),
                joint_ids=[
                    self.gimbal_joint_idx[agent_id]["pitch"],
                    self.gimbal_joint_idx[agent_id]["yaw"],
                    self.gimbal_joint_idx[agent_id]["roll"],
                ]
            )

            """ Debug logs """
            # gimbal_targets = torch.stack([
            #         self.cmd_gimbal_pitch[:, idx],
            #         self.cmd_gimbal_yaw[:, idx],
            #         gimbal_roll_stabilizing], dim=-1)
            # carb.log_warn(f"gimbal targets (pitch, yaw, roll): {gimbal_targets.cpu().numpy().round(2)}")

            # gimbal_actual = torch.stack([
            #         robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]],
            #         robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]],
            #         robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["roll"]]], dim=-1)
            # carb.log_warn(f"gimbal actual (pitch, yaw, roll): {gimbal_actual.cpu().numpy().round(2)}")

            self.zoom_level[:, idx] += self.cmd_vel[:, idx, 6] * self.cfg.max_zoom_rate * self.cfg.sim.dt
            self.zoom_level[:, idx] = torch.clamp(self.zoom_level[:, idx], min=1.0, max=self.cfg.max_zoom_level)

        # Update target movement
        self._update_target_movement()

    def _update_target_movement(self):
        """Update target movement each physics step."""
        velocity = self.target_movement.step(
            current_position=self.target.data.root_pos_w,
            env_origins=self._terrain.env_origins,
            curriculum_progress=self.progress_move,
            dt=self.cfg.sim.dt,
        )
        self.target.write_root_velocity_to_sim(velocity)
        # carb.log_warn(f"Target velocity command: {velocity.cpu().numpy()}")
        # carb.log_warn(f"Target velocity: {self.target.data.root_lin_vel_w.cpu().numpy()}")

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

        bbox_cfg = copy.deepcopy(self.cfg.bbox_raycaster)
        # NOTE: Do NOT enable debug_vis for BBoxRayCaster - it causes GPU crashes
        # due to timing issues with Vulkan/USD prim population. The useful visualization
        # (camera frustums, detection indicators, tri_cov markers) is handled by
        # CustomVisualization class instead.
        bbox_cfg.debug_vis = False
        target_prim_paths = self.scene.env_prim_paths
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
        self.delay_system.update_time()

        # Update curriculum scaling
        self.delay_system.set_noise_progress_scale(self.progress_delay)

        # ========== Update GT states for all agents ==========
        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Get body angular velocity including gimbal (combined angular velocity)
            body_link_ang_vel_w = robot.data.body_link_ang_vel_w[:, self.gimbal_joint_idx[agent_id]["pitch"], :]

            # Joint positions use [pitch, yaw, roll] convention to match
            # compute_camera_orientation_from_gimbal() expectations
            gimbal_joint_indices = torch.tensor([
                self.gimbal_joint_idx[agent_id]["pitch"],
                self.gimbal_joint_idx[agent_id]["yaw"],
                self.gimbal_joint_idx[agent_id]["roll"],
            ], device=self.device)

            # Update GT states (automatically creates delayed and delayed+noisy)
            self.delay_system.update_gt_states(
                agent_id=agent_id,
                body_position_w=robot.data.root_pos_w,
                body_orientation_w=robot.data.root_quat_w,
                body_linear_velocity_w=robot.data.root_lin_vel_w,
                body_angular_velocity_w=robot.data.root_ang_vel_w,
                body_linear_acceleration_w=robot.data.body_lin_acc_w[:, self._body_ids[agent_id], :],
                body_combined_angular_velocity_w=body_link_ang_vel_w,
                joint_positions_b=robot.data.joint_pos[:, gimbal_joint_indices],
                zoom_level=self.zoom_level[:, i]
            )

        # ========== Get GT bounding boxes using BBoxRayCaster ==========
        gt_camera_poses = {}
        gt_camera_intrinsics = {}

        for agent_id in self.cfg.possible_agents:
            gt_state = self.delay_system.gt_states.agents[agent_id]
            gt_camera_poses[agent_id] = (
                gt_state.data.camera_position_w,
                gt_state.data.camera_orientation_w
            )
            gt_camera_intrinsics[agent_id] = gt_state.data.camera_base_intrinsics.clone()
            gt_camera_intrinsics[agent_id][:, 0, 0] *= gt_state.data.camera_zoom_level
            gt_camera_intrinsics[agent_id][:, 1, 1] *= gt_state.data.camera_zoom_level

        # Get target GT pose
        target_pos = self.target.data.root_pos_w
        target_quat = self.target.data.root_quat_w
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(1)
        if target_quat.ndim == 2:
            target_quat = target_quat.unsqueeze(1)

        agent_body_poses = {
            agent_id: (robot.data.root_pos_w, robot.data.root_quat_w)
            for agent_id, robot in self._robots.items()
        }
        image_shapes = {
            str(agent_id): (self.agent_camera_cfgs[agent_id].height, self.agent_camera_cfgs[agent_id].width)
            for agent_id in self.cfg.possible_agents
        }

        self.bbox_raycaster.update(
            camera_poses=gt_camera_poses,
            camera_intrinsics=gt_camera_intrinsics,
            target_poses=(target_pos, target_quat),
            agent_poses=agent_body_poses,
            image_shapes=image_shapes,
            target_scale=self.target_scale
        )

        # ========== Update detections in delay system ==========
        for i, agent_id in enumerate(self.cfg.possible_agents):
            gt_bbox = self.bbox_raycaster.data.bboxes[:, i:i+1, :, :]
            self.delay_system.update_detections(
                agent_id=agent_id,
                bboxes_2d_gt=gt_bbox.squeeze(1),
            )

    def _compute_triangulation_covariance(self, X_w, robot_positions, robot_quats,
                                          gimbal_yaws, gimbal_pitches,
                                          camera_intrinsics, bbox_valid_mask, is_triangulation_valid) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            progress_coord = self.progress_coord
            Sigma_X, trace_cov = triangulation_covariance_multi_camera(
                X_w=X_w,
                robot_positions=robot_positions,
                robot_quats=robot_quats,
                gimbal_yaws=gimbal_yaws,
                gimbal_pitches=gimbal_pitches,
                camera_intrinsics=camera_intrinsics,
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

            is_trace_valid = (trace_cov >= 0.0) & torch.isfinite(trace_cov)
            is_sigma_finite = torch.isfinite(Sigma_X).all(dim=(-2, -1))
            is_not_nan = ~torch.isnan(Sigma_X).any(dim=(-2, -1))
            # FIX: Changed from ALL agents valid to at least 2 valid (pair requirement)
            # This prevents cascading failures from 5% detection dropout
            num_valid_agents = bbox_valid_mask.sum(dim=1).unsqueeze(-1)
            is_pair_valid = num_valid_agents >= 2  # At least 2 agents for triangulation
            is_diag_positive = torch.diagonal(Sigma_X, dim1=-2, dim2=-1).min(dim=-1).values > 0.0
            is_tri_cov_valid = is_trace_valid & is_sigma_finite & is_not_nan & is_pair_valid & is_diag_positive & is_triangulation_valid

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
            print(f"Warning: Triangulation covariance computation failed: {e}")
            is_tri_cov_valid = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            Sigma_X = self.Sigma_X_invalid.clone()
            trace_cov = self.trace_cov_invalid.clone()

        return Sigma_X, trace_cov, is_tri_cov_valid

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        self._compute_intermediate_values()
        rewards_dict = {}

        # ========== Compute triangulation PER AGENT (for rewards) ==========
        self.triangulation_results = {}

        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            # Get all agent states from ego perspective (clean, no noise)
            all_states = self.delay_system.get_all_states_for_rewards(ego_agent_id)

            # Build camera geometry from all agent states
            robot_positions_list = []
            robot_quats_list = []
            gimbal_yaws_list = []
            gimbal_pitches_list = []
            camera_intrinsics_list = []
            bbox_valid_list = []
            ray_origin_list = []

            for agent_id in self.cfg.possible_agents:
                state = all_states[agent_id]
                robot_positions_list.append(state.data.body_position_w)
                robot_quats_list.append(state.data.body_orientation_w)
                # joint_positions_b uses [pitch, yaw, roll] convention
                gimbal_pitches_list.append(state.data.joint_positions_b[:, 0:1])  # [0] = pitch
                gimbal_yaws_list.append(state.data.joint_positions_b[:, 1:2])     # [1] = yaw
                camera_intrinsics_updated = state.data.camera_base_intrinsics.clone()
                zoom_level = state.data.camera_zoom_level.clone()
                camera_intrinsics_updated[:, 0, 0] *= zoom_level
                camera_intrinsics_updated[:, 1, 1] *= zoom_level
                camera_intrinsics_list.append(camera_intrinsics_updated)
                bbox_valid = self.bbox_raycaster.validate_bbox(state.data.bboxes_2d[:, 0, :])
                bbox_valid_list.append(bbox_valid.squeeze(-1))
                ray_origin_list.append(state.data.camera_position_w)

            robot_positions = torch.stack(robot_positions_list, dim=1)
            robot_quats = torch.stack(robot_quats_list, dim=1)
            gimbal_pitches = torch.stack(gimbal_pitches_list, dim=1)
            gimbal_yaws = torch.stack(gimbal_yaws_list, dim=1)
            camera_intrinsics = torch.stack(camera_intrinsics_list, dim=1)
            bbox_valid_mask = torch.stack(bbox_valid_list, dim=1)
            ray_origins = torch.stack(ray_origin_list, dim=1)

            X_w_gt = self.target.data.root_pos_w.unsqueeze(1)

            Sigma_X, trace_cov, is_tri_cov_valid = self._compute_triangulation_covariance(
                X_w=X_w_gt,
                robot_positions=ray_origins,
                robot_quats=robot_quats,
                gimbal_yaws=gimbal_yaws.squeeze(-1),
                gimbal_pitches=gimbal_pitches.squeeze(-1),
                camera_intrinsics=camera_intrinsics,
                bbox_valid_mask=bbox_valid_mask,
                is_triangulation_valid=torch.ones(self.num_envs, 1, device=self.device, dtype=torch.bool)
            )
            std_dev = torch.sqrt(torch.diagonal(Sigma_X, dim1=-2, dim2=-1))
            std_dev = torch.where(
                is_tri_cov_valid.unsqueeze(-1),
                std_dev,
                torch.ones_like(std_dev) * 1e6
            )
            self.triangulation_results[ego_agent_id] = (X_w_gt, Sigma_X, std_dev, trace_cov, is_tri_cov_valid)

            # FIX: Track detection dropout statistics (only for first agent to avoid double counting)
            if i == 0:
                num_valid_detections = bbox_valid_mask.sum(dim=1)  # [N]
                self._detection_stats["total_steps"] += 1.0
                self._detection_stats["valid_triangulations"] += is_tri_cov_valid.squeeze(-1).float()
                self._detection_stats["invalid_all_agents"] += (num_valid_detections == 0).float()
                self._detection_stats["pair_valid_count"] += (num_valid_detections >= 2).float()

        # ========== Compute safety penalties for all agents ==========
        agent_positions = {}
        agent_bboxes = {}
        agent_bbox_valid = {}
        agent_focal_lengths = {}

        for agent_id in self.cfg.possible_agents:
            ego_states = self.delay_system.get_all_states_for_rewards(agent_id)
            ego_state = ego_states[agent_id]
            agent_positions[agent_id] = ego_state.data.body_position_w
            agent_bboxes[agent_id] = ego_state.data.bboxes_2d[:, 0, :]
            agent_bbox_valid[agent_id] = self.bbox_raycaster.validate_bbox(
                ego_state.data.bboxes_2d[:, 0, :]
            ).squeeze(-1)
            agent_focal_lengths[agent_id] = (
                ego_state.data.camera_base_intrinsics[:, 0, 0] * ego_state.data.camera_zoom_level,
                ego_state.data.camera_base_intrinsics[:, 1, 1] * ego_state.data.camera_zoom_level,
            )

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
            ego_states = self.delay_system.get_all_states_for_rewards(agent_id)
            delayed_state = ego_states[agent_id]

            X_w_gt, Sigma_X, std_dev, trace_cov, is_tri_cov_valid = self.triangulation_results[agent_id]

            action_sum = torch.sum(torch.square(self.action_weight * self._actions[agent_id]), dim=1)
            action_delta = torch.sum(
                torch.square(self.action_delta_weight * (self._actions[agent_id] - self._last_actions[agent_id])),
                dim=1
            )

            # Extract raw bbox values (in pixel coordinates) and normalize to [0, 1]
            bbox_center_raw = delayed_state.data.bboxes_2d[:, 0, 0:2]
            bbox_size_raw = delayed_state.data.bboxes_2d[:, 0, 2:4]
            bbox_valid = self.bbox_raycaster.validate_bbox(delayed_state.data.bboxes_2d[:, 0, :]).squeeze(-1)

            # Normalize using cached image dimensions
            bbox_center = bbox_center_raw / self._img_dims
            bbox_size = bbox_size_raw / self._img_dims

            bbox_center_dist = torch.norm(bbox_center - 0.5, dim=1)
            bbox_center_mapped = torch.exp(-10.0 * bbox_center_dist) * bbox_valid.float()

            bbox_area = bbox_size[:, 0] * bbox_size[:, 1]
            bbox_size_mapped = torch.exp(-torch.abs(bbox_area - 0.2)) * bbox_valid.float()

            triangulation_quality = torch.where(
                is_tri_cov_valid[:, 0],
                1.0 / (1.0 + trace_cov[:, 0] / 7.0),
                torch.zeros_like(trace_cov[:, 0])
            )

            collision_penalty = -safety_penalties[agent_id]["collision"]
            ttc_penalty = safety_penalties[agent_id]["ttc_penalty"]

            rewards = {
                "action_sum": action_sum * self.cfg.action_sum_penalty_scale * self.step_dt,
                "action_delta": action_delta * self.cfg.action_delta_penalty_scale * self.step_dt,
                "bbox_center": bbox_center_mapped * self.cfg.bbox_center_reward_scale * self.step_dt,
                "bbox_size": bbox_size_mapped * self.cfg.bbox_size_reward_scale * self.step_dt,
                "triangulation": triangulation_quality * self.cfg.triangulation_reward_scale * self.step_dt * self.progress_coord,
                "collision": collision_penalty * self.cfg.collision_penalty_scale * self.step_dt * self.progress_safety,
                "ttc": ttc_penalty * self.cfg.ttc_penalty_scale * self.step_dt * self.progress_safety,
            }

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

            rewards_dict[agent_id] = total_reward
            self._last_actions[agent_id] = self._actions[agent_id].clone()
            self.last_cmd_vel[:, i, :] = self.cmd_vel[:, i, :].clone()

        return rewards_dict

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents using delayed+noisy states."""
        self.triangulation_results_noisy = {}

        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            all_states = self.delay_system.get_all_states_for_observations(ego_agent_id)

            robot_positions_list = []
            robot_quats_list = []
            gimbal_yaws_list = []
            gimbal_pitches_list = []
            camera_intrinsics_list = []
            bbox_valid_list = []
            ray_origin_list = []
            ray_dirs_list = []

            for agent_id in self.cfg.possible_agents:
                state = all_states[agent_id]
                robot_positions_list.append(state.data.body_position_w)
                robot_quats_list.append(state.data.body_orientation_w)
                # joint_positions_b uses [pitch, yaw, roll] convention
                gimbal_pitches_list.append(state.data.joint_positions_b[:, 0:1])  # [0] = pitch
                gimbal_yaws_list.append(state.data.joint_positions_b[:, 1:2])     # [1] = yaw
                camera_intrinsics_updated = state.data.camera_base_intrinsics.clone()
                zoom_level = state.data.camera_zoom_level.clone()
                camera_intrinsics_updated[:, 0, 0] *= zoom_level
                camera_intrinsics_updated[:, 1, 1] *= zoom_level
                camera_intrinsics_list.append(camera_intrinsics_updated)
                bbox_valid = self.bbox_raycaster.validate_bbox(state.data.bboxes_2d[:, 0, :])
                bbox_valid_list.append(bbox_valid.squeeze(-1))
                ray_origin_list.append(state.data.camera_position_w)
                ray_dirs_list.append(state.data.camera_ray_directions_w[:, 0, :])

            robot_positions = torch.stack(robot_positions_list, dim=1)
            robot_quats = torch.stack(robot_quats_list, dim=1)
            gimbal_pitches = torch.stack(gimbal_pitches_list, dim=1)
            gimbal_yaws = torch.stack(gimbal_yaws_list, dim=1)
            camera_intrinsics = torch.stack(camera_intrinsics_list, dim=1)
            bbox_valid_mask = torch.stack(bbox_valid_list, dim=1)
            ray_origins = torch.stack(ray_origin_list, dim=1)
            ray_dirs = torch.stack(ray_dirs_list, dim=1)

            ray_dirs_expanded = ray_dirs.unsqueeze(2)
            X_w_triangulated, is_triangulation_valid = midpoint_method_batched(
                pts=ray_origins,
                dirs=ray_dirs_expanded,
                valid_mask=bbox_valid_mask
            )

            Sigma_X, trace_cov, is_cov_valid = self._compute_triangulation_covariance(
                X_w=X_w_triangulated,
                robot_positions=ray_origins,
                robot_quats=robot_quats,
                gimbal_yaws=gimbal_yaws.squeeze(-1),
                gimbal_pitches=gimbal_pitches.squeeze(-1),
                camera_intrinsics=camera_intrinsics,
                bbox_valid_mask=bbox_valid_mask,
                is_triangulation_valid=is_triangulation_valid
            )

            is_valid = is_triangulation_valid & is_cov_valid

            std_dev = torch.sqrt(torch.diagonal(Sigma_X, dim1=-2, dim2=-1))
            std_dev = torch.where(
                is_valid.unsqueeze(-1),
                std_dev,
                torch.ones_like(std_dev) * 1e6
            )

            self.triangulation_results_noisy[ego_agent_id] = (X_w_triangulated, Sigma_X, std_dev, trace_cov, is_valid)

        # ========== Build observations for each agent ==========
        observations = {}

        for i, ego_agent_id in enumerate(self.cfg.possible_agents):
            all_states = self.delay_system.get_all_states_for_observations(ego_agent_id)
            ego_state = all_states[ego_agent_id]

            X_w_tri, Sigma_X, std_tri, trace_cov, is_tri_cov_valid = self.triangulation_results_noisy[ego_agent_id]

            # FIX: Mask invalid triangulation coordinates to prevent NaN/Inf in observations
            # Use ego position as fallback when triangulation is invalid
            X_w_tri_masked = torch.where(
                is_tri_cov_valid.unsqueeze(-1),
                X_w_tri,
                torch.zeros_like(ego_state.data.body_position_w).unsqueeze(1)  # Set to zero as fallback
            )

            pos = ego_state.data.body_position_w
            roll, pitch, yaw = euler_xyz_from_quat(ego_state.data.body_orientation_w)
            yaw = yaw.unsqueeze(-1)
            lin_vel = ego_state.data.body_linear_velocity_w
            yaw_rate = ego_state.data.body_angular_velocity_w[:, 2:3]
            lin_acc = ego_state.data.body_linear_acceleration_w
            # joint_positions_b uses [pitch, yaw, roll] convention
            gimbal_pitch = ego_state.data.joint_positions_b[:, 0:1]  # [0] = pitch
            gimbal_yaw = ego_state.data.joint_positions_b[:, 1:2]    # [1] = yaw
            combined_ang_vel = ego_state.data.body_combined_angular_velocity_w

            bbox_raw = ego_state.data.bboxes_2d[:, 0, :]
            bbox_valid = self.bbox_raycaster.validate_bbox(bbox_raw).unsqueeze(-1)
            # Normalize bbox to [0, 1]: [x_center, y_center, width, height] / [w, h, w, h]
            bbox = bbox_raw / self._img_dims.repeat(2)
            time_since_detection = (self.delay_system.current_time - ego_state.data.timestamp_detection).unsqueeze(-1)
            zoom = ego_state.data.camera_zoom_level.unsqueeze(-1)
            ray_dir = ego_state.data.camera_ray_directions_w[:, 0, :]

            other_positions = []
            other_lin_vels = []
            other_combined_ang_vels = []
            other_bbox_valids = []
            other_data_age = []
            other_detection_age = []
            other_ray_dirs = []

            for other_agent_id in self.cfg.possible_agents:
                if other_agent_id == ego_agent_id:
                    continue

                other_state = all_states[other_agent_id]
                other_positions.append(other_state.data.body_position_w)
                other_lin_vels.append(other_state.data.body_linear_velocity_w)
                other_combined_ang_vels.append(other_state.data.body_combined_angular_velocity_w)
                other_bbox = other_state.data.bboxes_2d[:, 0, :]
                other_bbox_valids.append(self.bbox_raycaster.validate_bbox(other_bbox).unsqueeze(-1))
                detection_age = (self.delay_system.current_time - other_state.data.timestamp_detection).unsqueeze(-1)
                other_data_age.append(detection_age)
                other_detection_age.append(detection_age)
                other_ray_dirs.append(other_state.data.camera_ray_directions_w[:, 0, :])

            obs = torch.cat([
                pos,
                yaw,
                lin_vel,
                yaw_rate,
                lin_acc,
                gimbal_pitch,
                gimbal_yaw,
                combined_ang_vel,
                bbox,
                bbox_valid,
                time_since_detection,
                zoom,
                ray_dir,
                *other_positions,
                *other_lin_vels,
                *other_combined_ang_vels,
                *other_bbox_valids,
                *other_ray_dirs,
                *other_data_age,
                *other_detection_age,
                X_w_tri_masked[:, 0, :],  # FIX: Use masked triangulation position
                std_tri[:, 0, :],
            ], dim=-1)

            # Check for NaN/Inf values
            if torch.isnan(obs).any() or torch.isinf(obs).any():
                nan_envs = torch.nonzero((torch.isnan(obs) | torch.isinf(obs)).any(dim=1)).flatten()
                raise ValueError(f"NaN/Inf in {ego_agent_id} obs at envs {nan_envs.tolist()}")

            observations[ego_agent_id] = obs

        return observations

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get termination and timeout flags for all agents."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated_dict = {}
        time_out_dict = {}

        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            died = (robot.data.root_pos_w[:, 2] < 2.0) | (robot.data.root_pos_w[:, 2] > 50.0)
            terminated_dict[agent_id] = died
            time_out_dict[agent_id] = time_out

        return terminated_dict, time_out_dict

    def _build_initial_gt_states_for_reset(
        self,
        env_ids: torch.Tensor,
        reset_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """Build initial GT states dict for delay system reset."""
        from types import SimpleNamespace

        initial_gt_states = {}

        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            data = SimpleNamespace()

            if reset_states and agent_id in reset_states:
                rs = reset_states[agent_id]
                position = rs['position']
                orientation = rs['orientation']
                linear_velocity_w = rs['linear_velocity']
                angular_velocity_w = rs['angular_velocity']
                joint_velocities = rs['joint_velocities']

                from isaaclab.utils.math import quat_rotate_inverse
                linear_velocity_b = quat_rotate_inverse(orientation, linear_velocity_w)
                angular_velocity_b = quat_rotate_inverse(orientation, angular_velocity_w)

                data.body_position_w = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_position_w[env_ids] = position

                data.body_linear_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_linear_velocity_w[env_ids] = linear_velocity_w

                data.body_linear_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_linear_velocity_b[env_ids] = linear_velocity_b

                data.body_angular_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_angular_velocity_w[env_ids] = angular_velocity_w

                data.body_angular_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_angular_velocity_b[env_ids] = angular_velocity_b

                pitch_rate = joint_velocities[:, 0]
                yaw_rate = joint_velocities[:, 1] if joint_velocities.shape[1] > 1 else torch.zeros_like(pitch_rate)

                gimbal_ang_vel_b = torch.stack([
                    torch.zeros_like(pitch_rate),
                    pitch_rate,
                    yaw_rate,
                ], dim=-1)

                combined_angular_velocity_b = angular_velocity_b + gimbal_ang_vel_b

                from isaaclab.utils.math import quat_rotate
                combined_angular_velocity_w = quat_rotate(orientation, combined_angular_velocity_b)

                data.body_combined_angular_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_combined_angular_velocity_w[env_ids] = combined_angular_velocity_w
                data.body_combined_angular_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_combined_angular_velocity_b[env_ids] = combined_angular_velocity_b

                data.body_linear_acceleration_w = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_linear_acceleration_b = torch.zeros(self.num_envs, 3, device=self.device)
                data.body_angular_acceleration_b = torch.zeros(self.num_envs, 3, device=self.device)

                data.body_orientation_w = torch.zeros(self.num_envs, 4, device=self.device)
                data.body_orientation_w[:, 0] = 1.0
                data.body_orientation_w[env_ids] = rs['orientation']

                data.joint_positions_b = torch.zeros(self.num_envs, 3, device=self.device)
                data.joint_positions_b[env_ids] = rs['joint_positions']

                data.joint_velocities_b = torch.zeros(self.num_envs, 3, device=self.device)
                data.joint_velocities_b[env_ids] = rs['joint_velocities']

                # Camera offset rotation (identity quaternion - no rotation offset)
                data.camera_offset_rotation_b = torch.zeros(self.num_envs, 4, device=self.device)
                data.camera_offset_rotation_b[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0]

                # Compute camera orientation from body + gimbal angles
                from isaaclab_tasks.direct.iris_ma4.delay_system.derived_field_computers import (
                    compute_camera_orientation_from_gimbal
                )
                data.camera_orientation_w = compute_camera_orientation_from_gimbal(
                    body_orientation_w=data.body_orientation_w,
                    joint_positions_b=data.joint_positions_b,
                    camera_offset_rotation_b=data.camera_offset_rotation_b,
                )

                data.joint_accelerations_b = torch.zeros(self.num_envs, 3, device=self.device)

            else:
                import warnings
                warnings.warn(
                    f"_build_initial_gt_states_for_reset: Fallback branch reached for {agent_id}."
                )
                data.body_position_w = robot.data.root_pos_w.clone()
                data.body_linear_velocity_w = robot.data.root_lin_vel_w.clone()
                data.body_linear_velocity_b = robot.data.root_lin_vel_b.clone()
                data.body_angular_velocity_w = robot.data.root_ang_vel_w.clone()
                data.body_angular_velocity_b = robot.data.root_ang_vel_b.clone()
                data.body_combined_angular_velocity_w = robot.data.root_ang_vel_w.clone()
                data.body_combined_angular_velocity_b = robot.data.root_ang_vel_b.clone()
                data.body_linear_acceleration_w = torch.zeros_like(robot.data.root_pos_w)
                data.body_linear_acceleration_b = torch.zeros_like(robot.data.root_pos_w)
                data.body_angular_acceleration_b = torch.zeros_like(robot.data.root_pos_w)

                data.body_orientation_w = robot.data.root_quat_w.clone()

                # joint_positions_b uses [pitch, yaw, roll] convention
                gimbal_joint_ids = [
                    self.gimbal_joint_idx[agent_id]["pitch"],
                    self.gimbal_joint_idx[agent_id]["yaw"],
                    self.gimbal_joint_idx[agent_id]["roll"],
                ]
                data.joint_positions_b = robot.data.joint_pos[:, gimbal_joint_ids].clone()
                data.joint_velocities_b = robot.data.joint_vel[:, gimbal_joint_ids].clone()
                data.joint_accelerations_b = torch.zeros(self.num_envs, 3, device=self.device)

                # Camera offset rotation (identity quaternion - no rotation offset)
                data.camera_offset_rotation_b = torch.zeros(self.num_envs, 4, device=self.device)
                data.camera_offset_rotation_b[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0]

                # Compute camera orientation from body + gimbal angles
                from isaaclab_tasks.direct.iris_ma4.delay_system.derived_field_computers import (
                    compute_camera_orientation_from_gimbal
                )
                data.camera_orientation_w = compute_camera_orientation_from_gimbal(
                    body_orientation_w=data.body_orientation_w,
                    joint_positions_b=data.joint_positions_b,
                    camera_offset_rotation_b=data.camera_offset_rotation_b,
                )

            data.camera_zoom_level = torch.ones(self.num_envs, 1, device=self.device)
            if reset_states and agent_id in reset_states and 'zoom_level' in reset_states[agent_id]:
                data.camera_zoom_level[env_ids, 0] = reset_states[agent_id]['zoom_level']

            wrapper = SimpleNamespace(data=data)
            initial_gt_states[agent_id] = wrapper

        return initial_gt_states

    def _update_sigma_matrices(self, env_ids: torch.Tensor, sampled_noise: Dict[str, torch.Tensor]):
        """Update triangulation covariance Sigma matrices with sampled noise values.

        This is called on reset to update each environment's noise variance based on
        the sampled values from the delay system. This provides domain randomization
        where each environment has different noise characteristics.

        Args:
            env_ids: Environment indices that were reset.
            sampled_noise: Dictionary of sampled noise std values from delay system.
                Keys: "position", "orientation", "gimbal", "bbox", etc.
                Values: Tensor of shape [len(env_ids)] with sampled std per env.
        """
        C = len(self.cfg.possible_agents)
        K = len(env_ids)

        # Get sampled std values (shape: [K])
        pix_std = sampled_noise["bbox"]
        pos_std = sampled_noise["position"]
        ori_std = sampled_noise["orientation"]
        gimbal_std = sampled_noise["gimbal"]

        # Update Sigma_pix: [N, C, 2, 2] - diagonal matrix scaled by pix_std^2
        # Variance = std^2, broadcast [K] -> [K, C, 2, 2]
        pix_var = (pix_std ** 2).view(K, 1, 1, 1)  # [K, 1, 1, 1]
        eye2 = torch.eye(2, device=self.device)  # [2, 2]
        self.Sigma_pix[env_ids] = pix_var * eye2.view(1, 1, 2, 2).expand(K, C, 2, 2)

        # Update Sigma_twb: [N, C, 3, 3] - diagonal matrix scaled by pos_std^2
        pos_var = (pos_std ** 2).view(K, 1, 1, 1)
        eye3 = torch.eye(3, device=self.device)
        self.Sigma_twb[env_ids] = pos_var * eye3.view(1, 1, 3, 3).expand(K, C, 3, 3)

        # Update Sigma_phiwb: [N, C, 3, 3] - diagonal matrix scaled by ori_std^2
        ori_var = (ori_std ** 2).view(K, 1, 1, 1)
        self.Sigma_phiwb[env_ids] = ori_var * eye3.view(1, 1, 3, 3).expand(K, C, 3, 3)

        # Update Sigma_alpha, Sigma_beta: [N, C, 1, 1] - scalar gimbal_std^2
        gimbal_var = (gimbal_std ** 2).view(K, 1, 1, 1)
        self.Sigma_alpha[env_ids] = gimbal_var.expand(K, C, 1, 1)
        self.Sigma_beta[env_ids] = gimbal_var.expand(K, C, 1, 1)

        # Note: Sigma_K (camera intrinsics) is kept constant for now
        # Could be extended to support per-env sampling if needed

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at specified indices."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[self.cfg.possible_agents[0]]._ALL_INDICES

        self.safety_manager.reset(env_ids=env_ids)

        for agent_id in self.cfg.possible_agents:
            self._last_actions[agent_id][env_ids].zero_()
        self.last_cmd_vel[env_ids].zero_()
        self.zoom_level[env_ids].fill_(1.0)

        all_extras = {}
        for agent_id in self.cfg.possible_agents:
            for key, value in self._episode_sums[agent_id].items():
                episodic_sum_avg = torch.mean(value[env_ids])
                all_extras[f"Episode_Reward/{agent_id}_{key}"] = episodic_sum_avg / self.max_episode_length_s
                self._episode_sums[agent_id][key][env_ids] = 0.0

        # FIX: Log detection dropout statistics for debugging
        total_steps = self._detection_stats["total_steps"][env_ids]
        # Avoid division by zero
        valid_total = torch.clamp(total_steps, min=1.0)
        all_extras["Detection/valid_triangulation_rate"] = torch.mean(
            self._detection_stats["valid_triangulations"][env_ids] / valid_total
        )
        all_extras["Detection/pair_valid_rate"] = torch.mean(
            self._detection_stats["pair_valid_count"][env_ids] / valid_total
        )
        all_extras["Detection/all_invalid_rate"] = torch.mean(
            self._detection_stats["invalid_all_agents"][env_ids] / valid_total
        )
        # Reset detection stats for these envs
        for key in self._detection_stats:
            self._detection_stats[key][env_ids] = 0.0

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(all_extras)

        # Generate formation and target positions using distance-based generator
        # Primary curriculum factor is distance to target (not gimbal constraints)
        formation_scale = self.progress_tracking  # 0.0 -> 1.0 as training progresses

        # Select formation type based on curriculum progress
        # Early: planar (simplest, all agents same height)
        # Mid: grid (2D grid with curriculum-scaled height variation)
        # Late: line (full 3D formations with height variation)
        if formation_scale < 0.3:
            formation_type = "planar"
        elif formation_scale < 0.7:
            formation_type = "grid"
        else:
            formation_type = "line"

        # Single call generates both formation and target
        # Distance to target is the PRIMARY difficulty factor
        formation_result = self.randomizer.generate_formation_and_target(
            env_ids=torch.arange(len(env_ids), device=self.device),
            scale_factor=formation_scale,
            formation_type=formation_type,
        )

        formation_data = formation_result.agent_root_states
        target_positions = formation_result.target_position
        target_positions0 = torch.tensor([10.0, 0.0, 0.5], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)

        terrain_offset = self._terrain.env_origins[env_ids]
        target_pos = target_positions + terrain_offset
        """ Debug mode """
        # target_pos = target_positions0 + terrain_offset

        target_state = self.target.data.default_root_state[env_ids].clone()
        target_state[:, 0:3] = target_pos
        target_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        )
        self.target.write_root_pose_to_sim(target_state[:, :7], env_ids)
        self.target.write_root_velocity_to_sim(target_state[:, 7:], env_ids)

        # Reset target movement module
        self.target_movement.reset(env_ids)

        reset_states = {}

        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            agent_positions = formation_data[:, i, 0:3]
            agent_orientations = formation_data[:, i, 3:7]
            agent_velocities = formation_data[:, i, 7:13]

            # HARDCODED OVERRIDE - Uncomment below to use fixed positions for quick testing
            """ Debug mode """
            # agent_positions = torch.tensor([0.0, i * 5.0, 1.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
            # agent_orientations = quat_from_euler_xyz(
            #     torch.zeros(len(env_ids), device=self.device),
            #     torch.zeros(len(env_ids), device=self.device),
            #     torch.zeros(len(env_ids), device=self.device)
            # )
            # agent_velocities = torch.zeros(len(env_ids), 6, device=self.device)

            agent_root_state = robot.data.default_root_state[env_ids].clone()
            agent_root_state[:, 0:3] = agent_positions + terrain_offset
            agent_root_state[:, 3:7] = agent_orientations
            agent_root_state[:, 7:13] = agent_velocities

            robot.write_root_pose_to_sim(agent_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(agent_root_state[:, 7:13], env_ids)

            gimbal_yaw, gimbal_roll, gimbal_pitch = \
                self._gimbal_stabilizers[agent_id].compute_stabilized_angles_from_target_point(
                    target_point_world=target_pos,
                    drone_position_world=agent_root_state[:, 0:3],
                    drone_quat_world=agent_orientations
                )

            # Verify gimbal angles are within configured limits
            if __debug__:
                assert (gimbal_pitch >= self.cfg.gimbal.pitch_limits[0]).all(), \
                    f"Initial gimbal pitch {gimbal_pitch.min():.3f} below min {self.cfg.gimbal.pitch_limits[0]:.3f}"
                assert (gimbal_pitch <= self.cfg.gimbal.pitch_limits[1]).all(), \
                    f"Initial gimbal pitch {gimbal_pitch.max():.3f} above max {self.cfg.gimbal.pitch_limits[1]:.3f}"
                assert (gimbal_yaw >= self.cfg.gimbal.yaw_limits[0]).all(), \
                    f"Initial gimbal yaw {gimbal_yaw.min():.3f} below min {self.cfg.gimbal.yaw_limits[0]:.3f}"
                assert (gimbal_yaw <= self.cfg.gimbal.yaw_limits[1]).all(), \
                    f"Initial gimbal yaw {gimbal_yaw.max():.3f} above max {self.cfg.gimbal.yaw_limits[1]:.3f}"

            # joint_positions_b uses [pitch, yaw, roll] convention for consistency
            gimbal_joint_ids = [
                self.gimbal_joint_idx[agent_id]["pitch"],
                self.gimbal_joint_idx[agent_id]["yaw"],
                self.gimbal_joint_idx[agent_id]["roll"],
            ]
            joint_pos = torch.stack([gimbal_pitch, gimbal_yaw, gimbal_roll], dim=-1)
            joint_vel = robot.data.default_joint_vel[env_ids][:, gimbal_joint_ids].clone()

            robot.write_joint_state_to_sim(
                position=joint_pos,
                velocity=joint_vel,
                joint_ids=gimbal_joint_ids,
                env_ids=env_ids
            )
            robot.set_joint_position_target(
                target=torch.stack([gimbal_pitch, gimbal_yaw, gimbal_roll], dim=-1),
                # target=torch.stack([torch.ones_like(gimbal_yaw)*1, torch.ones_like(gimbal_roll)*0.2, torch.ones_like(gimbal_pitch)*0.5], dim=-1),
                joint_ids=gimbal_joint_ids,
                env_ids=env_ids
            )
            self.cmd_gimbal_yaw[env_ids, i] = gimbal_yaw.clone()
            self.cmd_gimbal_pitch[env_ids, i] = gimbal_pitch.clone()

            # Randomize zoom level between 1.0 and current_max_zoom
            random_zoom = 1.0 + torch.rand(len(env_ids), device=self.device) * (10.0 - 1.0)
            self.zoom_level[env_ids, i] = random_zoom
            
            reset_states[agent_id] = {
                'position': agent_root_state[:, 0:3].clone(),
                'orientation': agent_root_state[:, 3:7].clone(),
                'linear_velocity': agent_root_state[:, 7:10].clone(),
                'angular_velocity': agent_root_state[:, 10:13].clone(),
                'joint_positions': joint_pos.clone(),
                'joint_velocities': joint_vel.clone(),
                'zoom_level': self.zoom_level[env_ids, i].clone(),
            }

        initial_gt_states = self._build_initial_gt_states_for_reset(env_ids, reset_states=reset_states)

        # Reset delay system with noise resampling based on curriculum progress
        # Each environment gets its noise std sampled from Uniform(0, sigma_max * progress)
        sampled_noise = self.delay_system.reset(
            env_ids=env_ids,
            initial_gt_states=initial_gt_states,
            noise_progress=self.progress_delay,  # Use delay curriculum progress
        )

        # Update Sigma matrices with sampled noise values for triangulation covariance
        self._update_sigma_matrices(env_ids, sampled_noise)

        super()._reset_idx(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not DEBUG_DRAW:
            return

        # Increment frame counter for visualization warmup (prevents GPU crashes)
        self.visualization.step()

        # Safety check: ensure triangulation results exist before visualization
        if not hasattr(self, 'triangulation_results_noisy') or not self.triangulation_results_noisy:
            return

        self.visualization.camera_frustum[self.cfg.possible_agents[0]].draw_interface.clear_lines()
        self.visualization.detection_indicator[self.cfg.possible_agents[0]].draw_interface.clear_lines()

        for i, agent_id in enumerate(self.cfg.possible_agents):
            # Safety check: skip if this agent's triangulation results don't exist yet
            if agent_id not in self.triangulation_results_noisy:
                continue

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

            delayed_states = self.delay_system.get_all_states_for_observations(agent_id)
            delayed_bbox = delayed_states[agent_id].data.bboxes_2d[:, 0, :]
            delayed_bbox_valid = self.bbox_raycaster.validate_bbox(delayed_bbox)
            self.visualization.detection_indicator[agent_id].draw_indicator(
                start_points=delayed_states[agent_id].data.camera_position_w,
                end_points=self.target.data.root_pos_w,
                detected=delayed_bbox_valid.squeeze(-1)
            )

            X_w_tri, Sigma_X, std_dev, trace_cov, is_tri_cov_valid = self.triangulation_results_noisy[agent_id]
            valid_envs = torch.nonzero(
                is_tri_cov_valid.squeeze(-1), as_tuple=False
            ).squeeze(-1).tolist()

            if len(valid_envs) > 0:
                scale = torch.sqrt(torch.clamp(std_dev, min=0.0)) * 2.5
                # Use safe visualization method that checks if prim is ready
                self.visualization.visualize_tri_cov(
                    agent_id=agent_id,
                    translations=X_w_tri[valid_envs, 0, :],
                    scales=scale[valid_envs, 0, :],
                )
            else:
                # Use safe visualization method that checks if prim is ready
                self.visualization.visualize_tri_cov(
                    agent_id=agent_id,
                    translations=torch.zeros(1, 3, device=self.device),
                    scales=torch.zeros(1, 3, device=self.device),
                )
