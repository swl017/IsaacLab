# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import TiledCamera, FrameTransformer
from isaaclab.utils.math import (
    subtract_frame_transforms, 
    euler_xyz_from_quat, 
    quat_from_euler_xyz,
    quat_mul,
    quat_inv,
    quat_rotate,
)

from isaaclab.markers import CUBOID_MARKER_CFG

# Import components from iris_gimbal2_zoom
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'iris_gimbal2_zoom'))
from point_mass import PointMass
from bbox_generator import BBoxGenerator
from camera_frustrum import CameraFrustrum, create_camera_cfg_tensor

from .iris_ma_env_cfg import IrisMaEnvCfg

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


class IrisMaEnv(DirectMARLEnv):
    """Multi-agent environment for two gimbaled drones performing triangulation tracking."""
    
    cfg: IrisMaEnvCfg

    def __init__(self, cfg: IrisMaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize drone-specific components after scene setup
        self._initialize_drone_components()

        # Actions for both drones (vx, vy, vz, vyaw, gimbal_yaw, gimbal_pitch, zoom)
        self._actions = {
            "drone_0": torch.zeros(self.num_envs, 7, device=self.device),
            "drone_1": torch.zeros(self.num_envs, 7, device=self.device)
        }
        self._last_actions = {
            "drone_0": torch.zeros(self.num_envs, 7, device=self.device),
            "drone_1": torch.zeros(self.num_envs, 7, device=self.device)
        }

        # Control commands for both drones
        self._cmd_vel = {
            "drone_0": torch.zeros(self.num_envs, 1, 6, device=self.device),
            "drone_1": torch.zeros(self.num_envs, 1, 6, device=self.device)
        }
        self._thrust = {
            "drone_0": torch.zeros(self.num_envs, 1, 3, device=self.device),
            "drone_1": torch.zeros(self.num_envs, 1, 3, device=self.device)
        }
        self._moment = {
            "drone_0": torch.zeros(self.num_envs, 1, 3, device=self.device),
            "drone_1": torch.zeros(self.num_envs, 1, 3, device=self.device)
        }

        # Target motion
        self.target_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0)
        self.last_target_vel = self.target_vel.clone()
        self.target_acceleration = torch.zeros(self.num_envs, 6, device=self.device)
        self.target_desired_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0)
        self.target_vel_change_timer = torch.zeros(self.num_envs, device=self.device)


        # Formation and triangulation tracking
        self.formation_baseline_distance = torch.zeros(self.num_envs, device=self.device)
        self.triangulation_angle = torch.zeros(self.num_envs, device=self.device)
        self.triangulation_quality = torch.zeros(self.num_envs, device=self.device)

        # Desired positions for formation
        self._desired_formation_offset = torch.tensor([
            [-4.0, -4.0, 0.0],  # drone_0 offset from target
            [4.0, 4.0, 0.0]     # drone_1 offset from target
        ], device=self.device)

        # Action weights
        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device).repeat(self.num_envs, 1)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device).repeat(self.num_envs, 1)

        # Base focal length for zoom
        self.base_focal_length = self.cfg.camera_0_cfg.spawn.focal_length

        # Curriculum learning
        self.curriculum = torch.tensor(-1, device=self.device).repeat(self.num_envs, 1)
        self.last_curriculum = self.curriculum.clone()

        # Episode tracking
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "formation", "triangulation", "bbox_center_0", "bbox_size_0",
                "bbox_center_1", "bbox_size_1", "coordination", "zoom_0", "zoom_1"
            ]
        }

        # Environment origins
        self.step_count = 0
        self._env_origins = self._compute_env_origins_grid(self.num_envs, self.cfg.scene.env_spacing)


        # Visualization setup
        self.set_debug_vis(self.cfg.debug_vis)
        if DEBUG_DRAW and omni_debug_draw is not None:
            self.camera_frustrum = CameraFrustrum()
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """Compute the origins of the environments in a grid based on configured spacing."""
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
        """Set up the scene with two drones and a target."""
                # Camera and bbox data for both drones
        self.camera_data = {}
        self.bbox_data = {}
        self.gimbal_data = {}
        self.zoom_data = {}
        
        for agent in ["drone_0", "drone_1"]:
            self.camera_data[agent] = {
                "pos_world": torch.zeros(self.num_envs, 3, device=self.device),
                "quat_world": torch.zeros(self.num_envs, 4, device=self.device)
            }
            self.bbox_data[agent] = {
                "bboxes": torch.zeros(self.num_envs, 4, device=self.device),
                "bboxes_normalized": torch.zeros(self.num_envs, 4, device=self.device),
                "valid_mask": torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            }
            self.gimbal_data[agent] = {
                "dof_targets": None,  # Will be initialized after scene setup
                "joint_indices": {}
            }
            self.zoom_data[agent] = {
                "level": torch.ones(self.num_envs, device=self.device) * 1.0
            }

        # These will be initialized in _setup_scene
        self._drones = {}
        self._cameras = {}
        self.frame_transformers = {}
        self.target = None
        self._stabilizers = {}
        self.camera_cfg_batch = {}
        self.camera_offset_rot_batch = {}
        self.camera_offset_pos_batch = {}

        # Physics properties (will be set after scene setup)
        self._robot_mass = None
        self._gravity_magnitude = None
        self._robot_weight = None

        # Configure semantic tags and create drones
        self.cfg.drone_0_cfg.spawn.semantic_tags = [("class", "robot")]
        self.cfg.drone_1_cfg.spawn.semantic_tags = [("class", "robot")]
        
        self._drones["drone_0"] = Articulation(self.cfg.drone_0_cfg)
        self._drones["drone_1"] = Articulation(self.cfg.drone_1_cfg)

        # Terrain
        if self.cfg.terrain is not None:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        else:
            self._terrain = None

        # Cameras (only for debug visualization)
        if DEBUG_DRAW:
            self._cameras["drone_0"] = TiledCamera(self.cfg.camera_0_cfg)
            self._cameras["drone_1"] = TiledCamera(self.cfg.camera_1_cfg)

        # Target
        self.cfg.target_cfg.spawn.semantic_tags = [("class", "target")]
        self.target = RigidObject(self.cfg.target_cfg)

        # Frame transformers
        self.frame_transformers["drone_0"] = FrameTransformer(self.cfg.frame_transformer_0_cfg)
        self.frame_transformers["drone_1"] = FrameTransformer(self.cfg.frame_transformer_1_cfg)

        # Clone and register with scene
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        self.scene.articulations["drone_0"] = self._drones["drone_0"]
        self.scene.articulations["drone_1"] = self._drones["drone_1"]
        
        for camera_name, camera in self._cameras.items():
            self.scene.sensors[f"camera_{camera_name}"] = camera
            
        self.scene.rigid_objects["target"] = self.target
        
        for ft_name, ft in self.frame_transformers.items():
            self.scene.sensors[f"frame_transformer_{ft_name}"] = ft

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)



    def _initialize_drone_components(self):
        """Initialize drone-specific components after scene is set up."""
        # Get mass and physics properties
        self._robot_mass = self._drones["drone_0"].root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Initialize stabilizers and gimbal data for both drones
        for agent in ["drone_0", "drone_1"]:
            self._stabilizers[agent] = PointMass(0, self._robot_weight, self.num_envs, self.device)
            
            # Get joint indices for gimbal control
            drone = self._drones[agent]
            gimbal_joints = {
                "yaw": drone.find_joints("yaw_joint")[0][0],
                "roll": drone.find_joints("roll_joint")[0][0], 
                "pitch": drone.find_joints("pitch_joint")[0][0]
            }
            self.gimbal_data[agent]["joint_indices"] = gimbal_joints
            
            # Initialize gimbal targets
            self.gimbal_data[agent]["dof_targets"] = torch.zeros(self.num_envs, drone.num_joints, device=self.device)
            self.gimbal_data[agent]["dof_targets"][:, gimbal_joints["yaw"]] = 0
            self.gimbal_data[agent]["dof_targets"][:, gimbal_joints["roll"]] = 0
            self.gimbal_data[agent]["dof_targets"][:, gimbal_joints["pitch"]] = 0

        # Initialize camera configurations
        for i, agent in enumerate(["drone_0", "drone_1"]):
            camera_cfg = getattr(self.cfg, f"camera_{i}_cfg")
            self.camera_cfg_batch[agent] = create_camera_cfg_tensor(camera_cfg, self.num_envs, device=self.device)
            
            camera_offset_rot_single = torch.tensor(camera_cfg.offset.rot, device=self.device)
            self.camera_offset_rot_batch[agent] = camera_offset_rot_single.unsqueeze(0).expand(self.num_envs, -1)
            
            camera_offset_pos_single = torch.tensor(camera_cfg.offset.pos, device=self.device)
            self.camera_offset_pos_batch[agent] = camera_offset_pos_single.unsqueeze(0).expand(self.num_envs, -1)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        """Process actions for both drones before physics step."""
        # Clamp and store actions
        for agent in ["drone_0", "drone_1"]:
            if torch.isnan(actions[agent]).any():
                continue
            self._actions[agent] = actions[agent].clone().clamp(-1.0, 1.0)
            
            # Convert actions to velocity commands
            self._cmd_vel[agent][:, 0, :3] = self._actions[agent][:, :3] * self.cfg.max_lin_vel
            self._cmd_vel[agent][:, 0, 5] = self._actions[agent][:, 3] * self.cfg.max_yaw_rate

            # Compute thrust and moment using stabilizer
            drone = self._drones[agent]
            self._thrust[agent][:, 0, :], self._moment[agent][:, 0, :] = self._stabilizers[agent].compute_control(
                self._cmd_vel[agent][:, 0, :3], 
                self._cmd_vel[agent][:, 0, 5], 
                drone.data.root_state_w[:, 3:7],
                drone.data.root_lin_vel_w, 
                drone.data.root_ang_vel_b, 
                self.step_dt
            )

            # Update gimbal control
            self._update_gimbal_control(agent)
            
            # Update zoom level
            self._update_zoom_level(agent)

        # Update target movement
        self._update_target_movement()

    def _update_gimbal_control(self, agent: str):
        """Update gimbal control for a specific drone."""
        drone = self._drones[agent]
        gimbal_joints = self.gimbal_data[agent]["joint_indices"]
        
        # Get current orientation
        roll, pitch, yaw = euler_xyz_from_quat(drone.data.root_state_w[:, 3:7])
        curr_quat_w_yaw = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)
        roll_b, _, _ = euler_xyz_from_quat(quat_mul(drone.data.root_state_w[:, 3:7], quat_inv(curr_quat_w_yaw)))
        
        # Update gimbal targets
        gimbal_targets = self.gimbal_data[agent]["dof_targets"]
        gimbal_targets[:, gimbal_joints["yaw"]] += self._actions[agent][:, 4] * self.step_dt * 2 * math.pi
        gimbal_targets[:, gimbal_joints["yaw"]] = torch.clamp(
            gimbal_targets[:, gimbal_joints["yaw"]], -math.pi*2/3, math.pi*2/3
        )
        gimbal_targets[:, gimbal_joints["roll"]] = self._stabilizers[agent].wrap_to_pi(-roll_b)
        gimbal_targets[:, gimbal_joints["pitch"]] += self._actions[agent][:, 5] * self.step_dt * 2 * math.pi
        gimbal_targets[:, gimbal_joints["pitch"]] = torch.clamp(
            gimbal_targets[:, gimbal_joints["pitch"]], -math.pi*1/3, math.pi*1/3
        )

    def _update_zoom_level(self, agent: str):
        """Update zoom level for a specific drone."""
        delta_zoom = self._actions[agent][:, 6] * self.step_dt * 5.0
        self.zoom_data[agent]["level"] += delta_zoom
        self.zoom_data[agent]["level"] = torch.clamp(self.zoom_data[agent]["level"], 1.0, 10.0)
        
        # Update camera focal length
        current_focal_length = self.base_focal_length * self.zoom_data[agent]["level"]
        self.camera_cfg_batch[agent][:, 2] = current_focal_length
        
        # Update actual camera if exists
        camera = self._cameras.get(agent)
        if camera is not None:
            self._update_camera_intrinsics(agent, camera)

    def _update_camera_intrinsics(self, agent: str, camera: TiledCamera):
        """Update camera intrinsics for zoom."""
        cfg_batch = self.camera_cfg_batch[agent]
        width = cfg_batch[:, 0]
        height = cfg_batch[:, 1] 
        focal_length_mm = cfg_batch[:, 2]
        horizontal_aperture = cfg_batch[:, 3]
        
        f_x = (width * focal_length_mm) / horizontal_aperture
        f_y = f_x
        c_x = width / 2.0
        c_y = height / 2.0
        
        intrinsic_matrices = torch.zeros(self.num_envs, 3, 3, device=self.device)
        intrinsic_matrices[:, 0, 0] = f_x
        intrinsic_matrices[:, 0, 2] = c_x
        intrinsic_matrices[:, 1, 1] = f_y
        intrinsic_matrices[:, 1, 2] = c_y
        intrinsic_matrices[:, 2, 2] = 1.0
        
        camera.set_intrinsic_matrices(intrinsic_matrices, focal_length=self.base_focal_length)

    def _update_target_movement(self):
        """Update target movement with smooth acceleration-based motion."""
        self.target_vel_change_timer += self.step_dt
        
        # Randomly change desired velocity
        change_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.target_direction_change_prob
        
        new_desired_vel = torch.zeros_like(self.target_desired_vel)
        new_desired_vel[:, :3] = torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0
        new_desired_vel[:, 3:] = torch.rand(self.num_envs, 3, device=self.device) * 0.2 - 0.1
        new_desired_vel[:, :3] *= self.cfg.max_target_speed * 0.3
        
        self.target_desired_vel = torch.where(
            change_mask.unsqueeze(-1).expand(-1, 6),
            new_desired_vel,
            self.target_desired_vel
        )
        
        self.target_vel_change_timer = torch.where(
            change_mask, torch.zeros_like(self.target_vel_change_timer), self.target_vel_change_timer
        )
        
        # Compute acceleration and update velocity
        vel_error = self.target_desired_vel - self.target_vel
        self.target_acceleration = vel_error * self.cfg.target_acceleration_scale
        self.target_acceleration = torch.clamp(
            self.target_acceleration, -self.cfg.target_max_acceleration, self.cfg.target_max_acceleration
        )
        
        self.target_vel += torch.where(
            self.curriculum > 0, 
            self.target_acceleration * self.step_dt, 
            torch.zeros_like(self.target_acceleration)
        )
        self.target_vel *= torch.where(self.curriculum > 0, self.cfg.target_velocity_damping, 1.0)
        
        # Clamp final velocity
        self.target_vel[:, :3] = torch.clamp(
            self.target_vel[:, :3], -self.cfg.max_target_speed, self.cfg.max_target_speed
        )

        # Prevent target from going too low
        up_vel = self.target_vel.clone()
        up_vel[..., 2] = torch.where(self.target_vel[..., 2] > 0, self.target_vel[..., 2], -self.target_vel[..., 2])
        target_lower_bound = torch.ones_like(self.target.data.root_state_w, device=self.device) * 1.0
        self.target_vel[..., 2] = torch.where(
            self.target.data.root_state_w[:, 2] < target_lower_bound[:, 2], 
            up_vel[..., 2], 
            self.target_vel[..., 2]
        )

    def _apply_action(self):
        """Apply computed actions to both drones and target."""
        for agent in ["drone_0", "drone_1"]:
            drone = self._drones[agent]
            
            # Apply velocity control
            new_vel_lin_w = drone.data.root_lin_vel_w * (1 - self.step_dt) + quat_rotate(
                drone.data.root_state_w[:, 3:7], self._thrust[agent][:, 0, :]
            ) * self.step_dt
            new_vel_lin_w[:, 2] = torch.clamp(new_vel_lin_w[:, 2], -self.cfg.max_lin_vel, self.cfg.max_lin_vel)
            
            new_vel_ang_w = drone.data.root_ang_vel_w * (1 - 3 * self.step_dt) + quat_rotate(
                drone.data.root_state_w[:, 3:7], self._moment[agent][:, 0, :]
            ) * self.step_dt
            new_vel_ang_w[:, 0] = torch.zeros_like(new_vel_ang_w[:, 0])
            new_vel_ang_w[:, 1] = torch.zeros_like(new_vel_ang_w[:, 1])
            new_vel_ang_w[:, 2] = torch.clamp(new_vel_ang_w[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
            
            drone.write_root_velocity_to_sim(
                torch.cat((new_vel_lin_w, new_vel_ang_w), dim=1), env_ids=drone._ALL_INDICES
            )
            
            # Apply gimbal control
            drone.set_joint_position_target(self.gimbal_data[agent]["dof_targets"])

        # Apply target movement
        self.target.write_root_com_velocity_to_sim(self.target_vel)

        # Update frame transformers
        for ft in self.frame_transformers.values():
            ft.update(dt=self.cfg.sim.dt)

        # Update camera positions and compute bboxes
        self._update_camera_data()
        self._compute_bboxes()
        self._compute_formation_metrics()

        # Debug visualization
        if DEBUG_DRAW and hasattr(self, 'draw_interface'):
            self._draw_debug_info()

    def _update_camera_data(self):
        """Update camera positions and orientations for both drones."""
        for agent in ["drone_0", "drone_1"]:
            # Use frame transformer data for camera position
            self.camera_data[agent]["pos_world"] = self.frame_transformers[agent].data.target_pos_w[:, 0]
            
            # Compute camera orientation from robot and gimbal
            drone = self._drones[agent]
            robot_quat = drone.data.root_state_w[:, 3:7]
            gimbal_joints = self.gimbal_data[agent]["joint_indices"]
            
            gimbal_yaw = drone.data.joint_pos[:, gimbal_joints["yaw"]]
            gimbal_roll = drone.data.joint_pos[:, gimbal_joints["roll"]]
            gimbal_pitch = drone.data.joint_pos[:, gimbal_joints["pitch"]]
            
            gimbal_quat = quat_mul(
                quat_mul(
                    quat_from_euler_xyz(torch.zeros_like(gimbal_yaw), torch.zeros_like(gimbal_yaw), gimbal_yaw),
                    quat_from_euler_xyz(gimbal_roll, torch.zeros_like(gimbal_roll), torch.zeros_like(gimbal_roll))
                ),
                quat_from_euler_xyz(torch.zeros_like(gimbal_pitch), gimbal_pitch, torch.zeros_like(gimbal_pitch))
            )
            
            self.camera_data[agent]["quat_world"] = quat_mul(
                robot_quat, quat_mul(gimbal_quat, self.camera_offset_rot_batch[agent])
            )

    def _compute_bboxes(self):
        """Compute bounding boxes for target in both drone cameras."""
        for agent in ["drone_0", "drone_1"]:
            camera_cfg = getattr(self.cfg, f"camera_{agent.split('_')[1]}_cfg")
            bbox_gen = BBoxGenerator(
                camera_width=camera_cfg.width, 
                camera_height=camera_cfg.height,
                focal_length=camera_cfg.spawn.focal_length, 
                horizontal_aperture=camera_cfg.spawn.horizontal_aperture
            )
            
            current_focal_length = self.base_focal_length * self.zoom_data[agent]["level"]
            bboxes, valid_mask = bbox_gen.generate_2d_bbox(
                self.target.data.root_state_w[:, :3], 
                self.target.data.root_state_w[:, 3:7],
                self.camera_data[agent]["pos_world"], 
                self.camera_data[agent]["quat_world"], 
                min_bbox_size=int(0.1 * camera_cfg.width),
                focal_length=current_focal_length
            )
            
            self.bbox_data[agent]["bboxes"] = bboxes
            self.bbox_data[agent]["valid_mask"][:, 0] = valid_mask
            
            # Normalize bboxes
            normalized = bboxes.clone()
            normalized[:, 0] = torch.where(valid_mask, bboxes[:, 0] / camera_cfg.width, torch.ones_like(bboxes[:, 0]) * (-1.0))
            normalized[:, 1] = torch.where(valid_mask, bboxes[:, 1] / camera_cfg.height, torch.ones_like(bboxes[:, 1]) * (-1.0))
            normalized[:, 2] = torch.where(valid_mask, bboxes[:, 2] / camera_cfg.width, torch.ones_like(bboxes[:, 2]) * (-1.0))
            normalized[:, 3] = torch.where(valid_mask, bboxes[:, 3] / camera_cfg.height, torch.ones_like(bboxes[:, 3]) * (-1.0))
            self.bbox_data[agent]["bboxes_normalized"] = normalized

    def _compute_formation_metrics(self):
        """Compute formation and triangulation quality metrics."""
        # Compute baseline distance between drones
        drone_0_pos = self._drones["drone_0"].data.root_pos_w
        drone_1_pos = self._drones["drone_1"].data.root_pos_w
        self.formation_baseline_distance = torch.linalg.norm(drone_1_pos - drone_0_pos, dim=1)
        
        # Compute triangulation angle (angle between drone-target vectors)
        target_pos = self.target.data.root_state_w[:, :3]
        vec_0_to_target = target_pos - drone_0_pos
        vec_1_to_target = target_pos - drone_1_pos
        
        # Normalize vectors
        vec_0_norm = torch.linalg.norm(vec_0_to_target, dim=1, keepdim=True)
        vec_1_norm = torch.linalg.norm(vec_1_to_target, dim=1, keepdim=True)
        vec_0_normalized = vec_0_to_target / (vec_0_norm + 1e-6)
        vec_1_normalized = vec_1_to_target / (vec_1_norm + 1e-6)
        
        # Compute angle between vectors
        dot_product = torch.sum(vec_0_normalized * vec_1_normalized, dim=1)
        self.triangulation_angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        
        # Compute triangulation quality (closer to optimal angle is better)
        angle_error = torch.abs(self.triangulation_angle - self.cfg.optimal_triangulation_angle)
        self.triangulation_quality = 1.0 - torch.tanh(angle_error / (math.pi / 6))

    def _draw_debug_info(self):
        """Draw debug visualization lines and frustums."""
        self.draw_interface.clear_lines()
        
        # Draw lines from cameras to target
        for agent in ["drone_0", "drone_1"]:
            valid_mask = self.bbox_data[agent]["valid_mask"]
            line_colors_yellow = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
            line_colors_green = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
            line_colors = torch.where(valid_mask, line_colors_green, line_colors_yellow).tolist()
            line_thicknesses = [5.0] * self.num_envs
            
            self.draw_interface.draw_lines(
                self.camera_data[agent]["pos_world"].tolist(), 
                self.target.data.root_pos_w.tolist(), 
                line_colors, 
                line_thicknesses
            )
            
            # Draw camera frustum
            self.camera_frustrum.draw_frustrum(
                self.camera_data[agent]["pos_world"], 
                self.camera_data[agent]["quat_world"],
                self.camera_cfg_batch[agent], 
                self.zoom_data[agent]["level"], 
                self.device
            )

        # Draw formation baseline
        drone_0_pos = self._drones["drone_0"].data.root_pos_w
        drone_1_pos = self._drones["drone_1"].data.root_pos_w
        line_colors_blue = [[0.0, 0.0, 1.0, 1.0]] * self.num_envs
        line_thicknesses = [3.0] * self.num_envs
        self.draw_interface.draw_lines(
            drone_0_pos.tolist(), 
            drone_1_pos.tolist(), 
            line_colors_blue, 
            line_thicknesses
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations for both drones."""
        observations = {}
        
        target_pos = self.target.data.root_state_w[:, :3]
        
        for agent in ["drone_0", "drone_1"]:
            drone = self._drones[agent]
            other_agent = "drone_1" if agent == "drone_0" else "drone_0"
            other_drone = self._drones[other_agent]
            
            # Relative target position in drone body frame
            target_pos_b, _ = subtract_frame_transforms(
                drone.data.root_state_w[:, :3], 
                drone.data.root_state_w[:, 3:7], 
                target_pos
            )
            
            # Relative other drone position in drone body frame
            other_drone_pos_b, _ = subtract_frame_transforms(
                drone.data.root_state_w[:, :3], 
                drone.data.root_state_w[:, 3:7], 
                other_drone.data.root_pos_w
            )
            
            # Gimbal positions
            gimbal_joints = self.gimbal_data[agent]["joint_indices"]
            gimbal_pitch = drone.data.joint_pos[:, gimbal_joints["pitch"]]
            gimbal_yaw = drone.data.joint_pos[:, gimbal_joints["yaw"]]
            
            # Formation metrics
            formation_error = torch.abs(self.formation_baseline_distance - self.cfg.optimal_baseline_distance)
            triangulation_quality = self.triangulation_quality
            
            observations[agent] = torch.cat([
                drone.data.root_state_w[:, :10],  # position, orientation, linear and angular velocity (10)
                drone.data.root_ang_vel_b[:, 2:3],  # yaw rate (1) 
                drone.data.body_lin_acc_w[:, 0],  # linear acceleration (3)
                gimbal_pitch.unsqueeze(1),  # gimbal pitch (1)
                gimbal_yaw.unsqueeze(1),  # gimbal yaw (1)
                self.bbox_data[agent]["bboxes_normalized"],  # normalized bbox (4)
                self.bbox_data[agent]["valid_mask"].float(),  # bbox valid (1)
                target_pos_b,  # target position in body frame (3)
                other_drone_pos_b,  # other drone position in body frame (3)
                formation_error.unsqueeze(1),  # formation error (1)
                triangulation_quality.unsqueeze(1),  # triangulation quality (1)
            ], dim=-1)
        
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Compute rewards for both drones with focus on formation and triangulation."""
        rewards = {}
        
        # Compute shared formation and triangulation rewards
        formation_error = torch.abs(self.formation_baseline_distance - self.cfg.optimal_baseline_distance)
        formation_reward = (1 - torch.tanh(formation_error / 2.0)) * self.cfg.formation_reward_scale * self.step_dt
        
        triangulation_reward = self.triangulation_quality * self.cfg.triangulation_reward_scale * self.step_dt
        
        # Check if both drones can see the target (coordination reward)
        both_can_see = (
            self.bbox_data["drone_0"]["valid_mask"][:, 0] & 
            self.bbox_data["drone_1"]["valid_mask"][:, 0]
        ).float()
        coordination_reward = both_can_see * self.cfg.coordination_reward_scale * self.step_dt
        
        for agent in ["drone_0", "drone_1"]:
            drone = self._drones[agent]
            
            # Individual drone penalties
            lin_vel = torch.sum(torch.square(drone.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(drone.data.root_ang_vel_b), dim=1)
            action_sum = torch.sum(torch.square(self.action_weight * self._actions[agent]), dim=1)
            action_delta = torch.sum(torch.square(
                self.action_delta_weight * (self._actions[agent] - self._last_actions[agent])
            ), dim=1)
            
            # Bbox tracking rewards
            bbox = self.bbox_data[agent]["bboxes"]
            camera_cfg = getattr(self.cfg, f"camera_{agent.split('_')[1]}_cfg")
            bbox_center_error = (
                torch.square((bbox[:, 0] + bbox[:, 2])/2 - camera_cfg.width/2) / camera_cfg.width**2 +
                torch.square((bbox[:, 1] + bbox[:, 3])/2 - camera_cfg.height/2) / camera_cfg.height**2
            )
            bbox_center_reward = (1 - torch.tanh(bbox_center_error / 0.8)) * self.bbox_data[agent]["valid_mask"][:, 0].float()
            
            bbox_size = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / (camera_cfg.width * camera_cfg.height)
            bbox_size_reward = (1 - torch.tanh((bbox_size - 0.2**2) / 0.8)) * self.bbox_data[agent]["valid_mask"][:, 0].float()
            
            # Zoom penalty
            zoom_penalty = torch.square((self.zoom_data[agent]["level"] - 1.0) / 10) * self.cfg.zoom_reward_scale * self.step_dt
            
            # Combine all rewards for this agent
            agent_rewards = {
                "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
                "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
                "action_sum": action_sum * self.cfg.action_sum_reward_scale * self.step_dt,
                "action_delta": action_delta * self.cfg.action_delta_reward_scale * self.step_dt,
                "bbox_center": bbox_center_reward * self.cfg.bbox_center_reward_scale * self.step_dt,
                "bbox_size": bbox_size_reward * self.cfg.bbox_size_reward_scale * self.step_dt,
                "zoom": zoom_penalty,
                "formation": formation_reward,
                "triangulation": triangulation_reward,
                "coordination": coordination_reward,
            }
            
            rewards[agent] = torch.sum(torch.stack(list(agent_rewards.values())), dim=0)
            
            # Update episode sums for logging
            for key, value in agent_rewards.items():
                if key in ["formation", "triangulation", "coordination"]:
                    # Shared rewards only logged once
                    if agent == "drone_0":
                        self._episode_sums[key] += value
                else:
                    # Individual rewards
                    self._episode_sums[f"{key}_{agent.split('_')[1]}"] += value
        
        # Store last actions
        for agent in ["drone_0", "drone_1"]:
            self._last_actions[agent] = self._actions[agent].clone()
            
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute termination and timeout conditions for both drones."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if either drone crashed or went too high
        died_0 = self._drones["drone_0"].data.root_pos_w[:, 2] < 0.7
        died_1 = self._drones["drone_1"].data.root_pos_w[:, 2] < 0.7
        
        # Check if drones are too close (collision avoidance)
        too_close = self.formation_baseline_distance < self.cfg.min_baseline_distance
        
        died = torch.logical_or(torch.logical_or(died_0, died_1), too_close)
        
        terminated = {"drone_0": died, "drone_1": died}
        time_outs = {"drone_0": time_out, "drone_1": time_out}
        
        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments for specified indices."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._drones["drone_0"]._ALL_INDICES

        # Log final metrics
        if len(env_ids) > 0:
            final_formation_error = torch.abs(
                self.formation_baseline_distance[env_ids] - self.cfg.optimal_baseline_distance
            ).mean()
            final_triangulation_quality = self.triangulation_quality[env_ids].mean()
            
            # Reset episode sums and log
            extras = {}
            for key in self._episode_sums.keys():
                if len(self._episode_sums[key][env_ids]) > 0:
                    episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                    extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s

            if "bbox_center_0" in self._episode_sums and "bbox_center_1" in self._episode_sums:
                bbox_reward_0 = torch.mean(self._episode_sums["bbox_center_0"][env_ids]) if len(self._episode_sums["bbox_center_0"][env_ids]) > 0 else torch.tensor(0.0)
                bbox_reward_1 = torch.mean(self._episode_sums["bbox_center_1"][env_ids]) if len(self._episode_sums["bbox_center_1"][env_ids]) > 0 else torch.tensor(0.0)
                avg_bbox_reward = (bbox_reward_0 + bbox_reward_1) / 2
                self.curriculum = torch.where(
                    avg_bbox_reward > 20, torch.ones_like(self.curriculum), self.last_curriculum
                )
            
            for key in self._episode_sums.keys():
                self._episode_sums[key][env_ids] = 0.0
            
            # Set up logging for both agents
            for agent in ["drone_0", "drone_1"]:
                if agent not in self.extras:
                    self.extras[agent] = {}
                self.extras[agent]["log"] = extras.copy()
            
            # Add formation and triangulation metrics
            formation_metrics = {
                "Metrics/final_formation_error": final_formation_error.item(),
                "Metrics/final_triangulation_quality": final_triangulation_quality.item(),
            }
            
            for agent in ["drone_0", "drone_1"]:
                self.extras[agent]["log"].update(formation_metrics)

        # Reset drones
        for agent in ["drone_0", "drone_1"]:
            self._drones[agent].reset(env_ids)
            self._stabilizers[agent].reset(env_ids)
        
        # Reset parent
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset actions
        for agent in ["drone_0", "drone_1"]:
            self._actions[agent][env_ids] = 0.0
            self._last_actions[agent][env_ids] = 0.0

        # Reset target position
        self._reset_target_position(env_ids)
        
        # Reset drone positions in formation
        self._reset_drone_formation(env_ids)
        
        # Reset zoom levels
        for agent in ["drone_0", "drone_1"]:
            self.zoom_data[agent]["level"][env_ids] = 1.0

        # Update curriculum
        if len(env_ids) > 0 and "bbox_center_0" in self._episode_sums and "bbox_center_1" in self._episode_sums:
            bbox_reward_0 = torch.mean(self._episode_sums["bbox_center_0"][env_ids]) if len(self._episode_sums["bbox_center_0"][env_ids]) > 0 else torch.tensor(0.0)
            bbox_reward_1 = torch.mean(self._episode_sums["bbox_center_1"][env_ids]) if len(self._episode_sums["bbox_center_1"][env_ids]) > 0 else torch.tensor(0.0)
            avg_bbox_reward = (bbox_reward_0 + bbox_reward_1) / 2
            self.curriculum = torch.where(
                avg_bbox_reward > 20, torch.ones_like(self.curriculum), self.last_curriculum
            )
        self.last_curriculum = self.curriculum.clone()
        self.last_target_vel = self.target_vel.clone()

    def _reset_target_position(self, env_ids: torch.Tensor):
        """Reset target to a random position visible by both drones."""
        # Place target in a position that can be seen by both drones
        target_pos = torch.zeros(len(env_ids), 3, device=self.device)
        target_pos[:, 0] = torch.zeros_like(target_pos[:, 0]).uniform_(2.0, 8.0)
        target_pos[:, 1] = torch.zeros_like(target_pos[:, 1]).uniform_(-2.0, 2.0)
        target_pos[:, 2] = torch.zeros_like(target_pos[:, 2]).uniform_(2.0, 5.0)
        target_pos[:, :3] += self._env_origins[env_ids, :3]
        
        self.target.data.root_state_w[env_ids, :3] = target_pos
        self.target.data.root_state_w[env_ids, 7:] = torch.zeros_like(self.target.data.root_state_w[env_ids, 7:])
        self.target.write_root_pose_to_sim(self.target.data.root_state_w[env_ids, :7], env_ids)

    def _reset_drone_formation(self, env_ids: torch.Tensor):
        """Reset drones to optimal formation positions."""
        target_pos = self.target.data.root_state_w[env_ids, :3]
        
        for i, agent in enumerate(["drone_0", "drone_1"]):
            drone = self._drones[agent]
            
            # Reset joint positions
            joint_pos = drone.data.default_joint_pos[env_ids]
            gimbal_joints = self.gimbal_data[agent]["joint_indices"]
            
            joint_pos[:, gimbal_joints["yaw"]] = torch.zeros_like(
                joint_pos[:, gimbal_joints["yaw"]]
            ).uniform_(-math.pi * 2/3, math.pi * 2/3)
            joint_pos[:, gimbal_joints["pitch"]] = torch.zeros_like(
                joint_pos[:, gimbal_joints["pitch"]]
            ).uniform_(-math.pi * 1/3, math.pi * 1/3)
            joint_pos[:, gimbal_joints["roll"]] = torch.zeros_like(joint_pos[:, gimbal_joints["roll"]])
            
            self.gimbal_data[agent]["dof_targets"][env_ids] = joint_pos
            
            # Position drones in optimal formation around target
            formation_offset = self._desired_formation_offset[i]
            drone_pos = target_pos + formation_offset.unsqueeze(0).expand(len(env_ids), -1)
            
            # Add some randomization to initial positions
            drone_pos += torch.zeros_like(drone_pos).uniform_(-1.0, 1.0)
            drone_pos[:, 2] = torch.clamp(drone_pos[:, 2], 1.0, 6.0)  # Keep reasonable altitude
            
            # Reset drone state
            joint_vel = drone.data.default_joint_vel[env_ids]
            default_root_state = drone.data.default_root_state[env_ids]
            default_root_state[:, :3] = drone_pos
            default_root_state[:, 2] += 3.0  # Additional height offset
            
            drone.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            drone.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization for markers."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization markers."""
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self.target.data.root_state_w[:, :3])