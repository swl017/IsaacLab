# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math
import numpy as np
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
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
from isaaclab_assets import IRIS_GIMBAL2_CFG
from isaaclab.markers import CUBOID_MARKER_CFG, FRAME_MARKER_CFG

from .point_mass import PointMass
from .bbox_generator import BBoxGenerator
from .camera_frustrum import CameraFrustrum, create_camera_cfg_tensor, project_2d_to_3d

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
class IrisMAEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 30.0
    decimation = 2
    
    # Define agents
    possible_agents = ["drone_0", "drone_1"]
    
    # Define spaces for each agent
    action_spaces = {
        "drone_0": 7,  # [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        "drone_1": 7,
    }
    observation_spaces = {
        "drone_0": 26,  # Added formation-related observations
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

    # Camera configuration (shared between agents)
    camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot_.*/pitch_link/camera",
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

    # Target configuration (shared between agents)
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

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=15.0, replicate_physics=True)
    
    # Robot configurations for each agent
    robot_0: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_1: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    
    # Control parameters
    thrust_to_weight = 4.0
    moment_scale = 10.0
    yaw_moment_scale = 1.0
    
    # Motion limits
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
    
    # Reward scales (per agent)
    lin_vel_reward_scale = -0.1
    ang_vel_reward_scale = -0.02
    action_sum_reward_scale = -0.1
    action_weight = [1, 1, 5, 0.5, 0.03, 0.03, 0.01]
    action_delta_reward_scale = -0.01
    action_delta_weight = [1, 1, 5, 0.5, 0.03, 0.03, 0.01]
    zoom_reward_scale = -0.5
    
    # Single-agent tracking rewards
    bbox_center_reward_scale = 30
    bbox_size_reward_scale = 30
    
    # Multi-agent coordination rewards
    triangulation_reward_scale = 50  # Reward for good triangulation geometry
    baseline_reward_scale = 40       # Reward for optimal baseline distance
    collision_penalty_scale = -100   # Penalty for getting too close to each other
    
    # Triangulation parameters
    optimal_baseline_distance = 3.0  # Optimal distance between drones for triangulation
    min_safe_distance = 1.5          # Minimum safe distance between drones
    optimal_triangulation_angle = 90.0  # Optimal angle in degrees


class IrisMAEnv(DirectMARLEnv):
    cfg: IrisMAEnvCfg

    def __init__(self, cfg: IrisMAEnvCfg, render_mode: str | None = None, **kwargs):
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
        
        # Target movement (shared)
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
        self.base_focal_length = self.cfg.camera_cfg.spawn.focal_length
        
        # Camera configuration
        self.camera_cfg_batch = create_camera_cfg_tensor(self.cfg.camera_cfg, self.num_envs, device=self.device)
        camera_offset_rot_single = torch.tensor(cfg.camera_cfg.offset.rot, device=self.device)
        self.camera_offset_rot_batch = camera_offset_rot_single.unsqueeze(0).expand(self.num_envs, -1)
        camera_offset_pos_single = torch.tensor(cfg.camera_cfg.offset.pos, device=self.device)
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
        
        # Episode tracking
        self._episode_sums = {
            agent: {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in ["lin_vel", "action_sum", "action_delta", "bbox_center", "bbox_size", "zoom", 
                           "triangulation", "baseline", "collision"]
            }
            for agent in self.cfg.possible_agents
        }
        
        # Visualization
        self.set_debug_vis(self.cfg.debug_vis)
        if DEBUG_DRAW and omni_debug_draw is not None:
            self.camera_frustrum = CameraFrustrum()
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        
        self.step_count = 0
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
        self.cfg.robot_0.spawn.semantic_tags = [("class", "robot")]
        self.cfg.robot_1.spawn.semantic_tags = [("class", "robot")]
        self._robots["drone_0"] = Articulation(self.cfg.robot_0)
        self._robots["drone_1"] = Articulation(self.cfg.robot_1)
        
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
        
        # Register articulations
        self.scene.articulations["robot_0"] = self._robots["drone_0"]
        self.scene.articulations["robot_1"] = self._robots["drone_1"]
        self.scene.rigid_objects["target"] = self.target
        
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
            
            # Update zoom level
            delta_zoom = self._actions[agent_id][:, 6] * self.step_dt * 5.0
            self.zoom_level[agent_id] += delta_zoom
            self.zoom_level[agent_id] = torch.clamp(self.zoom_level[agent_id], 1.0, 10.0)
            
            # Update camera parameters for each agent
            current_focal_length = self.base_focal_length * self.zoom_level[agent_id]
            
            # Compute camera pose for bounding box generation
            robot_pos = robot.data.root_pos_w
            robot_quat = robot.data.root_state_w[:, 3:7]
            
            gimbal_yaw = robot.data.joint_pos[:, gimbal_idx["yaw"]]
            gimbal_roll = robot.data.joint_pos[:, gimbal_idx["roll"]]
            gimbal_pitch = robot.data.joint_pos[:, gimbal_idx["pitch"]]
            
            gimbal_quat = quat_mul(
                quat_mul(
                    quat_from_euler_xyz(torch.zeros_like(gimbal_yaw), torch.zeros_like(gimbal_yaw), gimbal_yaw),
                    quat_from_euler_xyz(gimbal_roll, torch.zeros_like(gimbal_roll), torch.zeros_like(gimbal_roll))
                ),
                quat_from_euler_xyz(torch.zeros_like(gimbal_pitch), gimbal_pitch, torch.zeros_like(gimbal_pitch))
            )
            
            self.camera_quat_world[agent_id] = quat_mul(robot_quat, quat_mul(gimbal_quat, self.camera_offset_rot_batch))
            camera_offset = torch.tensor((0, 0, 0.1), device=self.device).expand(self.num_envs, -1)
            self.camera_pos_world[agent_id] = robot_pos + quat_rotate(self.camera_quat_world[agent_id], camera_offset)
            
            # Generate bounding boxes
            bbox_gen = BBoxGenerator(
                camera_width=self.cfg.camera_cfg.width, 
                camera_height=self.cfg.camera_cfg.height,
                focal_length=self.cfg.camera_cfg.spawn.focal_length, 
                horizontal_aperture=self.cfg.camera_cfg.spawn.horizontal_aperture
            )
            
            self.bboxes[agent_id], self.bbox_valid_mask[agent_id][:, 0] = bbox_gen.generate_2d_bbox(
                self.target.data.root_state_w[:, :3], 
                self.target.data.root_state_w[:, 3:7],
                self.camera_pos_world[agent_id], 
                self.camera_quat_world[agent_id], 
                min_bbox_size=int(0.1*self.cfg.camera_cfg.width),
                focal_length=current_focal_length
            )
            
            # Normalize bounding boxes
            self.bboxes_normalized[agent_id] = self.bboxes[agent_id].clone()
            valid = self.bbox_valid_mask[agent_id].squeeze(-1)
            self.bboxes_normalized[agent_id][:, 0] = torch.where(valid, self.bboxes[agent_id][:, 0] / self.cfg.camera_cfg.width, torch.ones_like(self.bboxes[agent_id][:, 0]) * (-1.0))
            self.bboxes_normalized[agent_id][:, 1] = torch.where(valid, self.bboxes[agent_id][:, 1] / self.cfg.camera_cfg.height, torch.ones_like(self.bboxes[agent_id][:, 1]) * (-1.0))
            self.bboxes_normalized[agent_id][:, 2] = torch.where(valid, self.bboxes[agent_id][:, 2] / self.cfg.camera_cfg.width, torch.ones_like(self.bboxes[agent_id][:, 2]) * (-1.0))
            self.bboxes_normalized[agent_id][:, 3] = torch.where(valid, self.bboxes[agent_id][:, 3] / self.cfg.camera_cfg.height, torch.ones_like(self.bboxes[agent_id][:, 3]) * (-1.0))
        
        # Update target movement (shared)
        self._update_target_movement()
        
        # Bounce targets that are too low
        up_vel = self.target_vel.clone()
        up_vel[..., 2] = torch.where(self.target_vel[..., 2] > 0, self.target_vel[..., 2], -self.target_vel[..., 2])
        target_lower_bound = torch.ones_like(self.target.data.root_state_w, device=self.device) * 1.0
        self.target_vel[..., 2] = torch.where(
            self.target.data.root_state_w[:, 2] < target_lower_bound[:, 2], 
            up_vel[..., 2], 
            self.target_vel[..., 2]
        )
        
        self.step_count += 1

    def _apply_action(self):
        """Apply actions to all robots in the simulation."""
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            stabilizer = self._stabilizers[agent_id]
            
            # Compute new velocities
            new_vel_lin_w = robot.data.root_lin_vel_w * (1 - self.step_dt) + quat_rotate(
                robot.data.root_state_w[:, 3:7], self._thrust[agent_id][:, 0, :]) * self.step_dt
            new_vel_lin_w[:, 2] = torch.clamp(new_vel_lin_w[:, 2], min=-self.cfg.max_lin_vel, max=self.cfg.max_lin_vel)
            
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
        
        # Debug visualization
        if DEBUG_DRAW and hasattr(self, 'draw_interface'):
            self.draw_interface.clear_lines()
            for agent_id in self.cfg.possible_agents:
                line_colors_yellow = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                line_colors_green = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                line_colors = torch.where(self.bbox_valid_mask[agent_id], line_colors_green, line_colors_yellow).tolist()
                line_thicknesses = [5.0] * self.camera_pos_world[agent_id].shape[0]
                self.draw_interface.draw_lines(
                    self.camera_pos_world[agent_id].tolist(), 
                    self.target.data.root_pos_w.tolist(), 
                    line_colors, 
                    line_thicknesses
                )

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents."""
        observations = {}
        
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            other_agent_id = "drone_1" if agent_id == "drone_0" else "drone_0"
            other_robot = self._robots[other_agent_id]
            
            # Relative position to other drone
            relative_pos = other_robot.data.root_pos_w - robot.data.root_pos_w
            relative_dist = torch.norm(relative_pos, dim=1, keepdim=True)
            relative_dir = relative_pos / (relative_dist + 1e-6)
            
            # Triangulation angle (angle between camera-target lines)
            to_target_self = self.target.data.root_pos_w - robot.data.root_pos_w
            to_target_other = self.target.data.root_pos_w - other_robot.data.root_pos_w
            cos_angle = torch.sum(to_target_self * to_target_other, dim=1, keepdim=True) / (
                torch.norm(to_target_self, dim=1, keepdim=True) * torch.norm(to_target_other, dim=1, keepdim=True) + 1e-6
            )
            triangulation_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
            
            obs = torch.cat([
                robot.data.root_state_w[:, :10],  # 10: pos, quat, lin_vel
                robot.data.root_ang_vel_b[:, 2:3],  # 1: yaw rate
                robot.data.body_lin_acc_w[:, 0],  # 3: acceleration
                robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]:self.gimbal_joint_idx[agent_id]["pitch"] + 1],  # 1
                robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]:self.gimbal_joint_idx[agent_id]["yaw"] + 1],  # 1
                self.bboxes_normalized[agent_id],  # 4: normalized bbox
                self.bbox_valid_mask[agent_id].float(),  # 1: validity
                relative_dir,  # 3: direction to other drone
                relative_dist / 10.0,  # 1: normalized distance to other drone
                triangulation_angle / math.pi,  # 1: normalized triangulation angle
            ], dim=-1)
            
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
        
        # Get robot positions for triangulation calculations
        robot_positions = {
            agent: self._robots[agent].data.root_pos_w for agent in self.cfg.possible_agents
        }
        
        # Calculate baseline distance between drones
        baseline_distance = torch.norm(
            robot_positions["drone_0"] - robot_positions["drone_1"], 
            dim=1
        )
        
        # Calculate triangulation angle
        to_target_0 = self.target.data.root_pos_w - robot_positions["drone_0"]
        to_target_1 = self.target.data.root_pos_w - robot_positions["drone_1"]
        cos_angle = torch.sum(to_target_0 * to_target_1, dim=1) / (
            torch.norm(to_target_0, dim=1) * torch.norm(to_target_1, dim=1) + 1e-6
        )
        triangulation_angle_rad = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        triangulation_angle_deg = torch.rad2deg(triangulation_angle_rad)
        
        # Optimal triangulation angle reward (peak at 90 degrees)
        optimal_angle = self.cfg.optimal_triangulation_angle
        triangulation_quality = torch.exp(-torch.square(triangulation_angle_deg - optimal_angle) / (30.0**2))
        
        # Optimal baseline reward (peak at optimal_baseline_distance)
        baseline_quality = torch.exp(-torch.square(baseline_distance - self.cfg.optimal_baseline_distance) / (1.5**2))
        
        # Collision penalty
        collision_penalty = torch.where(
            baseline_distance < self.cfg.min_safe_distance,
            torch.ones_like(baseline_distance) * self.cfg.collision_penalty_scale,
            torch.zeros_like(baseline_distance)
        )
        
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            
            # Individual tracking rewards
            lin_vel = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)
            action_sum = torch.sum(torch.square(self.action_weight * self._actions[agent_id]), dim=1)
            action_delta = torch.sum(
                torch.square(self.action_delta_weight * (self._actions[agent_id] - self._last_actions[agent_id])), 
                dim=1
            )
            
            # Bounding box rewards
            bbox = (torch.square((self.bboxes[agent_id][:, 0] + self.bboxes[agent_id][:, 2])/2 - self.cfg.camera_cfg.width/2) / self.cfg.camera_cfg.width**2 
                    + torch.square((self.bboxes[agent_id][:, 1] + self.bboxes[agent_id][:, 3])/2 - self.cfg.camera_cfg.height/2) / self.cfg.camera_cfg.height**2)
            bbox_center_mapped = (1 - torch.tanh(bbox / 0.8)) * self.bbox_valid_mask[agent_id].squeeze(-1).float()
            
            bbox_size = (self.bboxes[agent_id][:, 2] - self.bboxes[agent_id][:, 0]) * (self.bboxes[agent_id][:, 3] - self.bboxes[agent_id][:, 1]) / (self.cfg.camera_cfg.width * self.cfg.camera_cfg.height)
            bbox_size_mapped = (1 - torch.tanh((bbox_size-0.2**2) / 0.8)) * self.bbox_valid_mask[agent_id].squeeze(-1).float()
            
            # Zoom penalty
            zoom_penalty = torch.square((self.zoom_level[agent_id] - 1.0)/10)
            
            # Combine rewards
            rewards = {
                "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
                "action_sum": action_sum * self.cfg.action_sum_reward_scale * self.step_dt,
                "action_delta": action_delta * self.cfg.action_delta_reward_scale * self.step_dt,
                "bbox_center": bbox_center_mapped * self.cfg.bbox_center_reward_scale * self.step_dt,
                "bbox_size": bbox_size_mapped * self.cfg.bbox_size_reward_scale * self.step_dt,
                "zoom": zoom_penalty * self.cfg.zoom_reward_scale * self.step_dt,
                "triangulation": triangulation_quality * self.cfg.triangulation_reward_scale * self.step_dt,
                "baseline": baseline_quality * self.cfg.baseline_reward_scale * self.step_dt,
                "collision": collision_penalty * self.step_dt,
            }
            
            # Store for logging
            for key, value in rewards.items():
                self._episode_sums[agent_id][key] += value
            
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
            env_ids = self._robots["drone_0"]._ALL_INDICES
        
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
        
        # Reset robots
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            robot.reset(env_ids)
            
            # Reset actions
            self._actions[agent_id][env_ids] = 0.0
            self._last_actions[agent_id][env_ids] = 0.0
            
            # Reset zoom level
            self.zoom_level[agent_id][env_ids] = 1.0
            
            # Reset gimbal targets
            joint_pos = robot.data.default_joint_pos[env_ids]
            gimbal_idx = self.gimbal_joint_idx[agent_id]
            joint_pos[:, gimbal_idx["yaw"]] = torch.zeros_like(joint_pos[:, gimbal_idx["yaw"]]).uniform_(-math.pi*2/3, math.pi*2/3)
            joint_pos[:, gimbal_idx["pitch"]] = torch.zeros_like(joint_pos[:, gimbal_idx["pitch"]]).uniform_(-math.pi*1/3, math.pi*1/3)
            joint_pos[:, gimbal_idx["roll"]] = torch.zeros_like(joint_pos[:, gimbal_idx["roll"]])
            self.gimbal_dof_targets[agent_id][env_ids] = joint_pos
            
            # Set initial positions with offset for each drone
            default_root_state = robot.data.default_root_state[env_ids]
            if idx == 0:
                # First drone on left
                default_root_state[:, 0] = -2.0
                default_root_state[:, 1] = torch.zeros_like(default_root_state[:, 1]).uniform_(-1.0, 1.0)
            else:
                # Second drone on right
                default_root_state[:, 0] = 2.0
                default_root_state[:, 1] = torch.zeros_like(default_root_state[:, 1]).uniform_(-1.0, 1.0)
            
            default_root_state[:, :3] += self._env_origins[env_ids]
            default_root_state[:, 2] += torch.ones_like(default_root_state[:, 2]) * 3.0
            
            # Write to simulation
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            
            # Reset stabilizers
            self._stabilizers[agent_id].reset(env_ids)
        
        # Reset target position (centered between drones)
        target_pos = torch.zeros(len(env_ids), 3, device=self.device)
        target_pos[:, 0] = torch.zeros_like(target_pos[:, 0]).uniform_(4.0, 8.0)
        target_pos[:, 1] = torch.zeros_like(target_pos[:, 1]).uniform_(-2.0, 2.0)
        target_pos[:, 2] = torch.zeros_like(target_pos[:, 2]).uniform_(2.0, 4.0)
        target_pos += self._env_origins[env_ids]
        
        self.target.data.root_state_w[env_ids, :3] = target_pos
        self.target.data.root_state_w[env_ids, 7:] = torch.zeros_like(self.target.data.root_state_w[env_ids, 7:])
        self.target.write_root_pose_to_sim(self.target.data.root_state_w[env_ids, :7], env_ids)
        
        # Reset target velocity
        self.target_vel[env_ids] = torch.zeros_like(self.target_vel[env_ids]).uniform_(-1.0, 1.0)
        self.last_target_vel[env_ids] = self.target_vel[env_ids].clone()
        
        # Reset episode length buffer
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

    def _update_target_movement(self):
        """Update target movement with smooth acceleration-based motion."""
        self.target_vel_change_timer += self.step_dt
        
        # Randomly change desired velocity
        change_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.target_direction_change_prob
        
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
        
        # Update velocity with acceleration and damping
        self.target_vel += self.target_acceleration * self.step_dt
        self.target_vel *= self.cfg.target_velocity_damping
        
        # Clamp final velocity
        self.target_vel[:, :3] = torch.clamp(
            self.target_vel[:, :3], 
            -self.cfg.max_target_speed, 
            self.cfg.max_target_speed
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        pass