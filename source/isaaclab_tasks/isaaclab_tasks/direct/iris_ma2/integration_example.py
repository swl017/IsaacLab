# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Integration example showing how to use BBoxRayCaster in IrisMAEnv.

This example demonstrates:
1. Configuration setup
2. Initialization in environment
3. Update during observation collection
4. Accessing bbox data per agent
"""
from isaaclab.app import AppLauncher
# launch omniverse app
simulation_app = AppLauncher(headless=True).app

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.rigid_object.rigid_object import RigidObject
import isaaclab.sim as sim_utils
import torch
from typing import Any

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from bbox_raycaster import BBoxRayCaster, BBoxRayCasterCfg

# Assuming your existing environment configuration
from iris_ma_env import IrisMAEnvCfg
import copy


class IrisMAEnvCfgWithBBox(IrisMAEnvCfg):
    """Extended configuration with bbox raycaster."""

    # Add bbox raycaster configuration
    bbox_raycaster: BBoxRayCasterCfg = BBoxRayCasterCfg(
        # Target configuration
        target_prim_paths=["/World/envs/env_.*/target"],
        
        # Mesh configuration for occlusion detection
        mesh_prim_paths=[
            "/World/ground",
            # Add other static obstacles if needed
            # "/World/envs/env_.*/obstacles"
        ],
        
        # Camera configuration
        num_cameras_per_env=2,  # Match number of agents
        
        # Validation thresholds
        min_bbox_size=(0.02, 0.02),  # 2% of image size minimum
        max_bbox_size=(0.90, 0.90),  # 90% of image size maximum
        partial_detection_allowed=False,  # All corners must be visible
        min_bbox_area_pixels=16.0,  # At least 4x4 pixels
        
        # Occlusion detection
        enable_occlusion_check=True,
        occlusion_ray_pattern="9point",  # Most accurate
        occlusion_visibility_threshold=0.5,  # 50% of points must be visible
        occlusion_ray_tolerance=1.1,
        
        # Performance
        max_distance=100.0,
        
        # Debug (disable in training)
        debug_vis=False,
        debug_vis_corners=False,
        debug_vis_occlusion_rays=False,
        debug_memory=False,
    )


class IrisMAEnvWithBBox(DirectMARLEnv):
    """Multi-agent Iris environment with bbox raycasting."""

    cfg: IrisMAEnvCfgWithBBox

    def __init__(self, cfg: IrisMAEnvCfgWithBBox, render_mode: str | None = None, **kwargs):
        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)
        self.camera_cfg_batch = {
            agent_id: create_camera_cfg_tensor(agent_cfg, self.num_envs, device=self.device)
            for agent_id, agent_cfg in self.agent_camera_cfgs.items()
        }
        self.zoom_level = {
            agent: torch.ones(self.num_envs, device=self.device) * 1.0
            for agent in self.cfg.possible_agents
        }
        self.base_focal_length = self.cfg.camera.spawn.focal_length


    def _setup_scene(self):
        self.agent_robot_cfgs = {}
        self.agent_camera_cfgs = {}
        self.bboxes = {}
        self.bboxes_normalized = {}
        self.bbox_valid_mask = {}
        # Initialize bbox storage tensors
        num_agents = len(self.cfg.possible_agents)
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

            self.bboxes[agent_id] = torch.zeros(self.num_envs, 4, device=self.device)
            self.bboxes_normalized[agent_id] = torch.zeros(self.num_envs, 4, device=self.device)
            self.bbox_valid_mask[agent_id] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._robots = {}
        for agent_id, robot_cfg in self.agent_robot_cfgs.items():
            robot_cfg.spawn.semantic_tags = [("class", "robot")]
            self._robots[agent_id] = Articulation(robot_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        self.cfg.terrain.env_spacing = 15.0
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Register articulations, rigid objects, and sensors
        for agent_id, robot in self._robots.items():
            robot_index = agent_id.split("_")[-1]
            self.scene.articulations[f"Robot_{robot_index}"] = robot
        
        self.scene.rigid_objects["target"] = self.target

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Initialize bbox raycaster
        self.bbox_raycaster = BBoxRayCaster(
            cfg=cfg.bbox_raycaster,
            num_envs=self.num_envs,
            num_targets_per_env=1,  # One target per environment
            device=self.device
        )

        # Log initialization
        print(f"[IrisMAEnv] Initialized BBoxRayCaster:")
        print(self.bbox_raycaster)

    def _get_observations(self) -> dict:
        """Get observations including bbox data.
        
        This method is called every step to collect observations for all agents.
        """
        # Update camera poses and intrinsics
        camera_poses = {}
        camera_intrinsics = {}
        image_shapes = {}

        for agent_id in self.cfg.possible_agents:
            # Get camera world pose from robot
            camera_pos_w, camera_quat_w = self._get_camera_world_pose(agent_id)
            camera_poses[agent_id] = (camera_pos_w, camera_quat_w)

            # Get camera intrinsics (accounting for zoom)
            intrinsic_matrix = self._get_camera_intrinsics(agent_id)
            camera_intrinsics[agent_id] = intrinsic_matrix

            # Get image shape
            camera_cfg = self.agent_camera_cfgs[agent_id]
            image_shapes[agent_id] = (camera_cfg.height, camera_cfg.width)

        # Get target poses
        target_poses = (
            self.target.data.root_pos_w,  # (N, T, 3)
            self.target.data.root_quat_w  # (N, T, 4)
        )

        # Update bbox raycaster
        self.bbox_raycaster.update(
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            target_poses=target_poses,
            image_shapes=image_shapes
        )

        # Extract bbox data per agent and store in environment
        for i, agent_id in enumerate(self.cfg.possible_agents):
            # Extract for this specific camera (agent)
            self.bboxes[agent_id] = self.bbox_raycaster.data.bboxes[:, i, 0, :]  # (N, 4)
            self.bboxes_normalized[agent_id] = self.bbox_raycaster.data.bboxes_normalized[:, i, 0, :]  # (N, 4)
            self.bbox_valid_mask[agent_id] = self.bbox_raycaster.data.valid_mask[:, i, 0]  # (N,)

            # Optional: Store occlusion visibility ratio
            if self.cfg.bbox_raycaster.enable_occlusion_check:
                visibility_ratio = self.bbox_raycaster.data.occlusion_visibility_ratio[:, i, 0]
                # You can use this for additional reward shaping or logging

        # Continue with regular observation collection
        obs = self._compute_observations()

        return obs

    def _compute_observations(self) -> dict:
        """Compute observations for all agents including bbox data."""
        obs = {}

        for agent_id in self.cfg.possible_agents:
            # Get robot state
            robot = self._robots[agent_id]
            robot_pos = robot.data.root_pos_w
            robot_vel = robot.data.root_lin_vel_w
            # ... other robot states

            # Get target state
            target_pos = self.target.data.root_pos_w[:, 0, :]  # (N, 3)
            # ... other target states

            # Get bbox data for this agent
            bbox = self.bboxes_normalized[agent_id]  # (N, 4) - normalized
            bbox_valid = self.bbox_valid_mask[agent_id].float().unsqueeze(-1)  # (N, 1)

            # Combine into observation vector
            obs[agent_id] = torch.cat([
                robot_pos,
                robot_vel,
                target_pos,
                bbox,  # Normalized bbox (cx, cy, w, h)
                bbox_valid,  # 1.0 if valid, 0.0 if invalid/occluded
                # ... other observations
            ], dim=-1)

        return obs

    def _get_camera_world_pose(self, agent_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get camera pose in world frame for a specific agent.
        
        This accounts for:
        1. Robot base pose
        2. Gimbal orientation (yaw, roll, pitch joints)
        3. Camera offset from gimbal
        
        Returns:
            Tuple of (position, quaternion) both with shape (N, 3) and (N, 4)
        """
        robot = self._robots[agent_id]
        
        # Get gimbal link pose (pitch_link where camera is attached)
        # This depends on your robot structure
        pitch_link_idx = robot.find_bodies("pitch_link")[0][0]
        pitch_link_pose = robot.data.body_pos_w[:, pitch_link_idx, :]
        pitch_link_quat = robot.data.body_quat_w[:, pitch_link_idx, :]

        # Apply camera offset
        camera_cfg = self.agent_camera_cfgs[agent_id]
        offset_pos = torch.tensor(camera_cfg.offset.pos, device=self.device)
        offset_quat = torch.tensor(camera_cfg.offset.rot, device=self.device)

        # Ensure pitch_link_quat has correct batch dimension
        if pitch_link_quat.shape[0] != self.num_envs:
            pitch_link_quat = pitch_link_quat.expand(self.num_envs, -1)
        if pitch_link_pose.shape[0] != self.num_envs:
            pitch_link_pose = pitch_link_pose.expand(self.num_envs, -1)

        # Transform offset to world frame
        import isaaclab.utils.math as math_utils
        offset_pos_world = math_utils.quat_apply(
            pitch_link_quat,
            offset_pos.expand(self.num_envs, -1)
        )
        camera_pos_w = pitch_link_pose + offset_pos_world

        # Combine quaternions
        camera_quat_w = math_utils.quat_mul(
            pitch_link_quat,
            offset_quat.expand(self.num_envs, -1)
        )

        return camera_pos_w, camera_quat_w

    def _get_camera_intrinsics(self, agent_id: str) -> torch.Tensor:
        """Get camera intrinsic matrix accounting for zoom.
        
        Returns:
            Intrinsic matrix with shape (N, 3, 3)
        """
        # Get base intrinsic from camera config
        base_intrinsic = self.camera_cfg_batch[agent_id]  # (N, 3, 3)

        # Apply zoom factor
        zoom = self.zoom_level[agent_id]  # (N,)
        
        # Update focal lengths
        intrinsic = base_intrinsic.clone()
        intrinsic[:, 0, 0] *= zoom  # fx
        intrinsic[:, 1, 1] *= zoom  # fy

        return intrinsic

    def _compute_rewards(self) -> dict:
        """Compute rewards including bbox-based rewards."""
        rewards = {}

        for agent_id in self.cfg.possible_agents:
            # Base reward computation
            # ... existing reward logic

            # Bbox-based rewards
            bbox_valid = self.bbox_valid_mask[agent_id]  # (N,)
            bbox_norm = self.bboxes_normalized[agent_id]  # (N, 4)

            # Reward for keeping target in view
            target_in_view_reward = bbox_valid.float() * 1.0

            # Reward for centered target
            cx, cy = bbox_norm[:, 0], bbox_norm[:, 1]
            center_error = torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
            centering_reward = torch.exp(-5.0 * center_error) * bbox_valid.float()

            # Reward for appropriate bbox size (not too small, not too large)
            w, h = bbox_norm[:, 2], bbox_norm[:, 3]
            bbox_area = w * h
            optimal_area = 0.15  # 15% of image
            size_error = torch.abs(bbox_area - optimal_area)
            size_reward = torch.exp(-10.0 * size_error) * bbox_valid.float()

            # Combine rewards
            bbox_reward = (
                0.3 * target_in_view_reward +
                0.4 * centering_reward +
                0.3 * size_reward
            )

            # Add to total reward
            rewards[agent_id] = self._compute_base_reward(agent_id) + 0.5 * bbox_reward

        return rewards


def create_camera_cfg_tensor(
    camera_cfg: Any,
    num_envs: int,
    device: str
) -> torch.Tensor:
    """Helper to create batched camera intrinsic matrix.
    
    Args:
        camera_cfg: Camera configuration object
        num_envs: Number of environments
        device: Device for tensor
        
    Returns:
        Intrinsic matrix with shape (num_envs, 3, 3)
    """
    # Extract camera parameters
    focal_length = camera_cfg.spawn.focal_length
    h_aperture = camera_cfg.spawn.horizontal_aperture
    v_aperture = camera_cfg.spawn.vertical_aperture or (
        h_aperture * camera_cfg.height / camera_cfg.width
    )
    h_offset = getattr(camera_cfg.spawn, 'horizontal_aperture_offset', 0.0)
    v_offset = getattr(camera_cfg.spawn, 'vertical_aperture_offset', 0.0)

    # Compute intrinsic parameters
    fx = camera_cfg.width * focal_length / h_aperture
    fy = camera_cfg.height * focal_length / v_aperture
    cx = h_offset * fx + camera_cfg.width / 2
    cy = v_offset * fy + camera_cfg.height / 2

    # Create intrinsic matrix
    intrinsic = torch.eye(3, device=device).unsqueeze(0).expand(num_envs, -1, -1).clone()
    intrinsic[:, 0, 0] = fx
    intrinsic[:, 1, 1] = fy
    intrinsic[:, 0, 2] = cx
    intrinsic[:, 1, 2] = cy

    return intrinsic


# Example usage in training script
if __name__ == "__main__":
    # from isaaclab.app import AppLauncher

    # # Launch Isaac Sim
    # app_launcher = AppLauncher(headless=True)
    # simulation_app = app_launcher.app

    # Create environment
    cfg = IrisMAEnvCfgWithBBox()
    env = IrisMAEnvWithBBox(cfg)

    # Run episode
    env.reset()
    for _ in range(1000):
        actions = {agent: torch.randn(env.num_envs, env.action_spaces[agent]) 
                   for agent in env.cfg.possible_agents}
        obs, rewards, dones, infos = env.step(actions)

        # Access bbox data
        for agent_id in env.cfg.possible_agents:
            bboxes = env.bboxes[agent_id]
            valid = env.bbox_valid_mask[agent_id]
            
            # Log or use for visualization
            print(f"{agent_id}: {valid.sum().item()}/{env.num_envs} valid detections")

    simulation_app.close()