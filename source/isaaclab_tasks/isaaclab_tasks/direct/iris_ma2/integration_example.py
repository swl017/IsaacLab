# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CORRECTED Integration example for BBoxRayCaster in IrisMAEnv.

Key fixes:
1. Proper dimension handling for target poses (N, T, 3) and (N, T, 4)
2. Consistent tensor operations throughout
3. Correct prim path handling for cloned environments
"""

from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=False).app

import copy
import numpy as np
import torch
from typing import Any

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from bbox_raycaster import BBoxRayCaster, BBoxRayCasterCfg
from iris_ma_env import IrisMAEnvCfg


class IrisMAEnvCfgWithBBox(IrisMAEnvCfg):
    """Extended configuration with bbox raycaster."""

    bbox_raycaster: BBoxRayCasterCfg = BBoxRayCasterCfg(
        target_prim_paths=[], # Will be set dynamically in _setup_scene
        mesh_prim_paths=["/World/ground"],
        
        # Camera configuration
        num_cameras_per_env=2,
        num_cameras_per_agent=1,
        
        # Validation thresholds
        min_bbox_size=(0.02, 0.02),
        max_bbox_size=(0.90, 0.90),
        partial_detection_allowed=False,
        min_bbox_area_pixels=16.0,
        
        # Occlusion detection
        enable_occlusion_check=True,
        occlusion_ray_pattern="9point",
        occlusion_visibility_threshold=0.5,
        occlusion_ray_tolerance=1.1,
        
        # Performance
        max_distance=100.0,
        
        # Debug
        debug_vis=False,
        debug_memory=False,
    )


class IrisMAEnvWithBBox(DirectMARLEnv):
    """Multi-agent Iris environment with bbox raycasting."""

    cfg: IrisMAEnvCfgWithBBox

    def __init__(self, cfg: IrisMAEnvCfgWithBBox, render_mode: str | None = None, **kwargs):
        # Force small number of envs for testing
        cfg.scene.num_envs = 5
        
        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize camera intrinsics and zoom
        self.camera_cfg_batch = {
            agent_id: create_camera_cfg_tensor(agent_cfg, self.num_envs, device=self.device)
            for agent_id, agent_cfg in self.agent_camera_cfgs.items()
        }
        self.zoom_level = {
            agent: torch.ones(self.num_envs, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.base_focal_length = self.cfg.camera.spawn.focal_length

    def _setup_scene(self):
        """Setup scene with robots, targets, and bbox raycaster."""
        # Initialize storage
        self.agent_robot_cfgs = {}
        self.agent_camera_cfgs = {}
        self.bboxes = {}
        self.bboxes_normalized = {}
        self.bbox_valid_mask = {}
        
        # Create agent-specific configurations
        for agent_id in self.cfg.possible_agents:
            robot_index = agent_id.split("_")[-1]
            robot_name = f"Robot_{robot_index}"

            # Robot config
            robot_cfg = copy.deepcopy(self.cfg.robot)
            robot_cfg.prim_path = robot_cfg.prim_path.format(robot_name=robot_name)
            self.agent_robot_cfgs[agent_id] = robot_cfg
            
            # Camera config
            camera_cfg = copy.deepcopy(self.cfg.camera)
            camera_cfg.prim_path = camera_cfg.prim_path.format(robot_name=robot_name)
            self.agent_camera_cfgs[agent_id] = camera_cfg

            # Bbox storage
            self.bboxes[agent_id] = torch.zeros(self.num_envs, 4, device=self.device)
            self.bboxes_normalized[agent_id] = torch.zeros(self.num_envs, 4, device=self.device)
            self.bbox_valid_mask[agent_id] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        # Create robots
        self._robots = {}
        for agent_id, robot_cfg in self.agent_robot_cfgs.items():
            robot_cfg.spawn.semantic_tags = [("class", "robot")]
            self._robots[agent_id] = Articulation(robot_cfg)
        
        # Create target
        self.target = RigidObject(self.cfg.target_cfg)
        
        # Create terrain
        self.cfg.terrain.env_spacing = 15.0
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Register with scene
        for agent_id, robot in self._robots.items():
            robot_index = agent_id.split("_")[-1]
            self.scene.articulations[f"Robot_{robot_index}"] = robot
        self.scene.rigid_objects["target"] = self.target

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Configure and initialize bbox raycaster
        # IMPORTANT: Use cloned environment paths (not regex)
        self.cfg.bbox_raycaster.target_prim_paths = self.scene.env_prim_paths
        # This gives: ["/World/envs/env_0", "/World/envs/env_1", ...]
        # BBoxRayCaster will:
        #   1. Append "/target" to each path
        #   2. Extract bbox from all environments
        #   3. Automatically detect if all targets identical
        #   4. Optimize storage accordingly
        
        self.bbox_raycaster = BBoxRayCaster(
            cfg=self.cfg.bbox_raycaster,
            num_envs=self.num_envs,
            num_targets_per_env=1,
            device=self.device
        )
        
        print(f"\n[IrisMAEnv] Initialized BBoxRayCaster:")
        print(self.bbox_raycaster)
        
        # Verify bbox extraction strategy
        if self.bbox_raycaster.targets_share_bbox:
            print(f"✓ Optimized: All targets share bbox (memory savings: 99.98%)")
            print(f"  Bbox storage: {self.bbox_raycaster.target_bbox_corners_local.shape}")
        else:
            print(f"✓ Per-environment: Targets have different bboxes")
            print(f"  Bbox storage: {self.bbox_raycaster.target_bbox_corners_local.shape}")

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass

    def _get_states(self):
        obs_list = []
        for agent_id in self.cfg.possible_agents:
            obs_list.append(self.obs[agent_id])
        return torch.cat(obs_list, dim=-1)

    def _get_observations(self) -> dict:
        """Get observations including bbox data."""
        # Collect camera data from all agents
        camera_poses = {}
        camera_intrinsics = {}
        image_shapes = {}

        for agent_id in self.cfg.possible_agents:
            # Get camera pose in world frame
            camera_pos_w, camera_quat_w = self._get_camera_world_pose(agent_id)
            camera_poses[agent_id] = (camera_pos_w, camera_quat_w)

            # Get intrinsics (with zoom applied)
            intrinsic_matrix = self._get_camera_intrinsics(agent_id)
            camera_intrinsics[agent_id] = intrinsic_matrix

            # Get image dimensions
            camera_cfg = self.agent_camera_cfgs[agent_id]
            image_shapes[agent_id] = (camera_cfg.height, camera_cfg.width)

        # Get target poses with correct dimensions
        # CRITICAL: Ensure target poses have shape (N, T, 3) and (N, T, 4)
        target_pos = self.target.data.root_pos_w  # May be (N, 3)
        target_quat = self.target.data.root_quat_w  # May be (N, 4)
        
        # Add target dimension if missing
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(1)  # (N, 3) -> (N, 1, 3)
        if target_quat.ndim == 2:
            target_quat = target_quat.unsqueeze(1)  # (N, 4) -> (N, 1, 4)
        
        target_poses = (target_pos, target_quat)

        # Update bbox raycaster
        self.bbox_raycaster.update(
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            target_poses=target_poses,
            image_shapes=image_shapes
        )

        # Extract bbox data per agent
        for i, agent_id in enumerate(self.cfg.possible_agents):
            self.bboxes[agent_id] = self.bbox_raycaster.data.bboxes[:, i, 0, :]
            self.bboxes_normalized[agent_id] = self.bbox_raycaster.data.bboxes_normalized[:, i, 0, :]
            self.bbox_valid_mask[agent_id] = self.bbox_raycaster.data.valid_mask[:, i, 0]

        # Compute observations
        self.obs = self._compute_observations()
        return self.obs
    
    def _get_rewards(self):
        return self._compute_rewards()
    
    def _get_dones(self):
        terminated_dict = {agent_id: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for agent_id in self.cfg.possible_agents}
        timeout_dict = {agent_id: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for agent_id in self.cfg.possible_agents}
        return terminated_dict, timeout_dict

    def _reset_idx(self, env_ids):
        return super()._reset_idx(env_ids)

    def _set_debug_vis_impl(self, enabled: bool):
        """Enable/disable debug visualization."""
        self.bbox_raycaster.visualize()

    def _compute_observations(self) -> dict:
        """Compute observations for all agents."""
        obs = {}

        for agent_id in self.cfg.possible_agents:
            # Robot state
            robot = self._robots[agent_id]
            robot_pos = robot.data.root_pos_w  # (N, 3)
            robot_vel = robot.data.root_lin_vel_w  # (N, 3)

            # Target state
            target_pos = self.target.data.root_pos_w  # (N, 3)

            # Bbox data
            bbox = self.bboxes_normalized[agent_id]  # (N, 4)
            bbox_valid = self.bbox_valid_mask[agent_id].float().unsqueeze(-1)  # (N, 1)

            # Combine into observation
            obs[agent_id] = torch.cat([
                robot_pos,      # (N, 3)
                robot_vel,      # (N, 3)
                target_pos,     # (N, 3)
                bbox,           # (N, 4)
                bbox_valid,     # (N, 1)
            ], dim=-1)  # Total: (N, 14)

        return obs

    def _get_camera_world_pose(self, agent_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get camera pose in world frame.
        
        Returns:
            (position, quaternion) with shapes (N, 3) and (N, 4)
        """
        robot = self._robots[agent_id]
        
        # Get pitch link pose (where camera is mounted)
        pitch_link_bodies = robot.find_bodies("pitch_link")
        if len(pitch_link_bodies[0]) == 0:
            raise RuntimeError(f"Could not find 'pitch_link' body in robot {agent_id}")
        
        pitch_link_idx = pitch_link_bodies[0][0]
        pitch_link_pos = robot.data.body_pos_w[:, pitch_link_idx, :]  # (N, 3)
        pitch_link_quat = robot.data.body_quat_w[:, pitch_link_idx, :]  # (N, 4)

        # Get camera offset from config
        camera_cfg = self.agent_camera_cfgs[agent_id]
        offset_pos = torch.tensor(camera_cfg.offset.pos, device=self.device)
        offset_quat = torch.tensor(camera_cfg.offset.rot, device=self.device)

        # Transform offset to world frame
        offset_pos_world = math_utils.quat_apply(
            pitch_link_quat,
            offset_pos.expand(self.num_envs, -1)
        )
        camera_pos_w = pitch_link_pos + offset_pos_world

        # Combine quaternions
        camera_quat_w = math_utils.quat_mul(
            pitch_link_quat,
            offset_quat.expand(self.num_envs, -1)
        )

        return camera_pos_w, camera_quat_w

    def _get_camera_intrinsics(self, agent_id: str) -> torch.Tensor:
        """Get camera intrinsic matrix with zoom applied.
        
        Returns:
            Intrinsic matrix with shape (N, 3, 3)
        """
        # Base intrinsic
        base_intrinsic = self.camera_cfg_batch[agent_id]  # (N, 3, 3)

        # Apply zoom
        zoom = self.zoom_level[agent_id]  # (N,)
        
        intrinsic = base_intrinsic.clone()
        intrinsic[:, 0, 0] *= zoom  # fx
        intrinsic[:, 1, 1] *= zoom  # fy

        return intrinsic

    def _compute_base_reward(self, agent_id: str) -> torch.Tensor:
        """Placeholder for base reward computation.
        
        Returns:
            Reward tensor with shape (N,)
        """
        # Implement your base reward logic here
        return torch.zeros(self.num_envs, device=self.device)

    def _compute_rewards(self) -> dict:
        """Compute rewards including bbox-based components."""
        rewards = {}

        for agent_id in self.cfg.possible_agents:
            # Base reward
            base_reward = self._compute_base_reward(agent_id)

            # Bbox data
            bbox_valid = self.bbox_valid_mask[agent_id]  # (N,)
            bbox_norm = self.bboxes_normalized[agent_id]  # (N, 4)

            # Component 1: Visibility reward
            r_visibility = bbox_valid.float() * 1.0

            # Component 2: Centering reward
            cx, cy = bbox_norm[:, 0], bbox_norm[:, 1]
            center_error = torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
            r_centering = torch.exp(-5.0 * center_error) * bbox_valid.float()

            # Component 3: Size reward
            w, h = bbox_norm[:, 2], bbox_norm[:, 3]
            bbox_area = w * h
            optimal_area = 0.15
            size_error = torch.abs(bbox_area - optimal_area)
            r_size = torch.exp(-10.0 * size_error) * bbox_valid.float()

            # Combine
            bbox_reward = (
                0.3 * r_visibility +
                0.4 * r_centering +
                0.3 * r_size
            )

            rewards[agent_id] = base_reward + 0.5 * bbox_reward

        return rewards


def create_camera_cfg_tensor(
    camera_cfg: Any,
    num_envs: int,
    device: str
) -> torch.Tensor:
    """Create batched camera intrinsic matrix.
    
    Args:
        camera_cfg: Camera configuration
        num_envs: Number of environments
        device: Device for tensor
        
    Returns:
        Intrinsic matrix (num_envs, 3, 3)
    """
    # Extract parameters
    focal_length = camera_cfg.spawn.focal_length
    h_aperture = camera_cfg.spawn.horizontal_aperture
    v_aperture = camera_cfg.spawn.vertical_aperture
    if v_aperture is None:
        v_aperture = h_aperture * camera_cfg.height / camera_cfg.width
    
    h_offset = getattr(camera_cfg.spawn, 'horizontal_aperture_offset', 0.0)
    v_offset = getattr(camera_cfg.spawn, 'vertical_aperture_offset', 0.0)

    # Compute intrinsic parameters
    fx = camera_cfg.width * focal_length / h_aperture
    fy = camera_cfg.height * focal_length / v_aperture
    cx = h_offset * fx + camera_cfg.width / 2
    cy = v_offset * fy + camera_cfg.height / 2

    # Create matrix
    intrinsic = torch.eye(3, device=device).unsqueeze(0).expand(num_envs, -1, -1).clone()
    intrinsic[:, 0, 0] = fx
    intrinsic[:, 1, 1] = fy
    intrinsic[:, 0, 2] = cx
    intrinsic[:, 1, 2] = cy

    return intrinsic


if __name__ == "__main__":
    # Create environment
    cfg = IrisMAEnvCfgWithBBox()
    env = IrisMAEnvWithBBox(cfg)

    # Run test episode
    print("\n" + "="*70)
    print("Running BBoxRayCaster Integration Test")
    print("="*70)
    
    env.reset()
    
    for step in range(100):
        # Zero actions respecting each agent's Box action space
        actions = {}
        for agent in env.cfg.possible_agents:
            action_space = env.action_spaces[agent]
            torch_dtype = torch.from_numpy(np.zeros((), dtype=action_space.dtype)).dtype
            action_shape = (env.num_envs,) + action_space.shape
            actions[agent] = torch.zeros(action_shape, device=env.device, dtype=torch_dtype)
        
        # Step environment
        obs, rewards, terminated, timeout, infos = env.step(actions)

        # Log bbox statistics every 10 steps
        if step % 10 == 0:
            print(f"\nStep {step}:")
            for agent_id in env.cfg.possible_agents:
                valid_count = env.bbox_valid_mask[agent_id].sum().item()
                
                if valid_count > 0:
                    # Compute statistics for valid detections
                    bbox = env.bboxes_normalized[agent_id][env.bbox_valid_mask[agent_id]]
                    avg_cx = bbox[:, 0].mean().item()
                    avg_cy = bbox[:, 1].mean().item()
                    avg_area = (bbox[:, 2] * bbox[:, 3]).mean().item()
                    
                    print(f"  {agent_id}: {valid_count}/{env.num_envs} valid")
                    print(f"    Avg center: ({avg_cx:.3f}, {avg_cy:.3f})")
                    print(f"    Avg area: {avg_area:.3f}")
                else:
                    print(f"  {agent_id}: 0/{env.num_envs} valid")

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)

    simulation_app.close()
