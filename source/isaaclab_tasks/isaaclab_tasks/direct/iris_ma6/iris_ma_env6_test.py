# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Iris MA6 Test Environment for validating DroneController integration.

This is a simplified multi-agent environment with 3 drones to test the iris_ma6
controller module. It demonstrates basic usage of DroneController with
velocity commands, gimbal control, and zoom control.

Control Architecture:
    Policy (25Hz) -> DroneController -> Forces/Torques + Gimbal Targets + Zoom Level
"""

from __future__ import annotations

import copy
import torch
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import TiledCamera

from .bbox_raycaster_v2 import BBoxRayCasterV2
from .bbox_raycaster_v2.utils.projection import create_intrinsic_matrix_tensor
from .controller import DroneController
from .controller.gimbal_controller import YAW_JOINT_OFFSET
from .delay_system_v3.derived_field_computers import (
    compute_camera_orientation_from_gimbal,
    compute_camera_position,
)
from .iris_ma_env6_test_cfg import IrisMA6TestEnvCfg
from .visualization import CustomVisualization
from .visualization.frame_visualizer import FRAME_LINKS


class IrisMA6TestEnv(DirectMARLEnv):
    """Simplified test environment for iris_ma6 DroneController validation.

    This environment provides:
    - 3 agents with velocity + gimbal + zoom control
    - Simple distance-based rewards
    - Minimal observations for testing

    Observation space per agent: 18D
        - pos (3): World position
        - vel (3): World velocity
        - quat (4): Orientation quaternion (wxyz)
        - gimbal_yaw (1): Current gimbal yaw angle
        - gimbal_pitch (1): Current gimbal pitch angle
        - zoom (1): Current zoom level
        - bbox (4): Primary target normalized bbox (cx, cy, w, h)
        - bbox_empty (1): 1 if bbox is empty, 0 otherwise

    Action space per agent: 7D
        - vx, vy, vz (3): Velocity commands in world frame
        - yaw_rate (1): Yaw rate command
        - gimbal_yaw_rate (1): Gimbal yaw rate command
        - gimbal_pitch_rate (1): Gimbal pitch rate command
        - zoom_rate (1): Zoom rate command
    """

    cfg: IrisMA6TestEnvCfg

    def __init__(self, cfg: IrisMA6TestEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the test environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments passed to DirectMARLEnv.
        """
        # Dynamically generate agent-specific robot and camera configs BEFORE super().__init__
        # This is required because the scene setup needs the configs
        self.agent_robot_cfgs: Dict[str, object] = {}
        self.agent_camera_cfgs: Dict[str, object] = {}
        for agent_id in cfg.possible_agents:
            robot_index = agent_id.split("_")[-1]
            robot_name = f"Robot_{robot_index}"

            # Create and store robot config
            robot_cfg = copy.deepcopy(cfg.robot)
            robot_cfg.prim_path = robot_cfg.prim_path.format(robot_name=robot_name)
            self.agent_robot_cfgs[agent_id] = robot_cfg

            # Create and store camera config (attached to pitch_link/gimbal tip)
            camera_cfg = copy.deepcopy(cfg.camera)
            camera_cfg.prim_path = camera_cfg.prim_path.format(robot_name=robot_name)
            self.agent_camera_cfgs[agent_id] = camera_cfg

        # Call parent constructor
        super().__init__(cfg, render_mode, **kwargs)

        # Store references to robots (populated after scene setup)
        self._robots: Dict[str, Articulation] = {}
        for agent_id in cfg.possible_agents:
            robot_index = agent_id.split("_")[-1]
            self._robots[agent_id] = self.scene.articulations[f"Robot_{robot_index}"]

        # Find gimbal joint indices for each robot
        self.gimbal_joint_idx: Dict[str, Dict[str, int]] = {}
        for agent_id in cfg.possible_agents:
            robot = self._robots[agent_id]
            self.gimbal_joint_idx[agent_id] = {
                "yaw": robot.find_joints("yaw_joint")[0][0],
                "roll": robot.find_joints("roll_joint")[0][0],
                "pitch": robot.find_joints("pitch_joint")[0][0],
            }

        # DEBUG: Print joint mapping for first agent
        first_agent = cfg.possible_agents[0]
        first_robot = self._robots[first_agent]
        print("\n=== GIMBAL JOINT MAPPING ===")
        print(f"  All joint names: {first_robot.joint_names}")
        print(f"  pitch_joint -> idx {self.gimbal_joint_idx[first_agent]['pitch']}")
        print(f"  yaw_joint   -> idx {self.gimbal_joint_idx[first_agent]['yaw']}")
        print(f"  roll_joint  -> idx {self.gimbal_joint_idx[first_agent]['roll']}")
        print("  USD kinematic chain: body -> yaw(Z) -> roll(X) -> pitch(Y)")
        print("=" * 30)

        # NOTE: Propeller visual spinning disabled - write_joint_state_to_sim overwrites
        # joint state including causing instability. To enable visual
        # spinning, the USD asset needs to be modified to make propeller rigid bodies
        # kinematic with zero mass/inertia.

        # Find body IDs for force application
        self._body_ids: Dict[str, list] = {}
        for agent_id in cfg.possible_agents:
            robot = self._robots[agent_id]
            body_ids, _ = robot.find_bodies("body")
            self._body_ids[agent_id] = body_ids

        # Find body indices for frame visualization links (body, yaw_link, roll_link, pitch_link)
        self._frame_link_ids: Dict[str, Dict[str, int]] = {}
        for agent_id in cfg.possible_agents:
            robot = self._robots[agent_id]
            self._frame_link_ids[agent_id] = {}
            for link_name in FRAME_LINKS:
                ids, _ = robot.find_bodies(link_name)
                self._frame_link_ids[agent_id][link_name] = ids[0]

        # Create per-agent DroneControllers
        self._controllers: Dict[str, DroneController] = {}
        for agent_id in cfg.possible_agents:
            robot = self._robots[agent_id]
            # Get mass from physics - sum all body masses for one environment
            all_masses = robot.root_physx_view.get_masses()
            # Shape is (num_envs, num_bodies) - sum bodies for first env
            mass = all_masses[0].sum().item()
            self._controllers[agent_id] = DroneController(
                cfg=self.cfg.drone_controller,
                mass=mass,
                gravity=9.81,
                num_envs=self.num_envs,
                device=self.device,
            )

        # Action buffers
        self._actions: Dict[str, torch.Tensor] = {}
        num_agents = len(cfg.possible_agents)
        self.cmd_vel = torch.zeros(self.num_envs, num_agents, 7, device=self.device)

        # Zoom level tracking (controller manages internal state, we track for observations)
        self.zoom_level = torch.ones(self.num_envs, num_agents, device=self.device)

        # Initial gimbal angles
        self.cmd_gimbal_yaw = torch.zeros(self.num_envs, num_agents, device=self.device)
        self.cmd_gimbal_pitch = torch.zeros(self.num_envs, num_agents, device=self.device)

        self._camera_offset_position_b = torch.tensor(
            self.cfg.camera.offset.pos, dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        # Camera offset rotation for frustum visualization.
        # Frustum uses +Z forward convention. This rotation, combined with the
        # gimbal yaw joint (which includes YAW_JOINT_OFFSET = -π/2), maps
        # frustum +Z → body +Y (visual forward).
        # Derivation: R_z(-90°) * R_y(-90°) maps +Z → +Y.
        self._camera_offset_rotation_b = torch.tensor(
            [0.7071, -0.7071, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        camera_cfg_batch = torch.tensor(
            [
                self.cfg.camera.width,
                self.cfg.camera.height,
                self.cfg.camera.spawn.focal_length,
                self.cfg.camera.spawn.horizontal_aperture,
                self.cfg.camera.spawn.clipping_range[0],
                self.cfg.camera.spawn.clipping_range[1],
            ],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self._camera_intrinsics_base = create_intrinsic_matrix_tensor(camera_cfg_batch)
        self._camera_image_shape = (self.cfg.camera.height, self.cfg.camera.width)
        bbox_cfg = copy.deepcopy(self.cfg.bbox_raycaster_v2)
        bbox_cfg.target_prim_paths = list(self.scene.env_prim_paths)
        self.bbox_raycaster_v2 = BBoxRayCasterV2(
            cfg=bbox_cfg,
            num_envs=self.num_envs,
            num_targets_per_env=1,
            device=self.device,
            agent_ids=self.cfg.possible_agents,
        )

        # Visualization is created lazily via _set_debug_vis_impl when debug_vis is enabled
        self._visualization: CustomVisualization | None = None

        # Cache for visualization data (updated each step, used by debug callback)
        self._vis_camera_poses: Dict[str, tuple] = {}
        self._vis_target_pos: torch.Tensor | None = None
        self._vis_bbox_empty: Dict[str, torch.Tensor] = {}
        self._vis_zoom_levels: Dict[str, torch.Tensor] = {}

        # Enable debug visualization if configured
        if self.cfg.debug_vis:
            self.set_debug_vis(True)

    def _setup_scene(self):
        """Setup the scene with multiple robots, cameras, and shared target."""
        # Create robots for each agent from the dynamically generated configs
        for agent_id, robot_cfg in self.agent_robot_cfgs.items():
            robot = Articulation(robot_cfg)
            robot_index = agent_id.split("_")[-1]
            self.scene.articulations[f"Robot_{robot_index}"] = robot

        # Create TiledCamera sensors for each agent (attached to gimbal tip/pitch_link)
        # These cameras are aligned with the frustum visualization direction
        self._cameras: Dict[str, TiledCamera] = {}
        for agent_id, camera_cfg in self.agent_camera_cfgs.items():
            self._cameras[agent_id] = TiledCamera(camera_cfg)

        # Create terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create shared target
        self.target = RigidObject(self.cfg.target_cfg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Register rigid objects
        self.scene.rigid_objects["target"] = self.target

        # Register TiledCamera sensors in scene
        for agent_id, camera in self._cameras.items():
            robot_index = agent_id.split("_")[-1]
            self.scene.sensors[f"camera_{robot_index}"] = camera

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]):
        """Pre-process actions for all agents before physics step.

        Runs at decimated rate (policy frequency).

        Args:
            actions: Dictionary mapping agent_id to action tensor (N, 7).
        """
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            # Clip and store actions
            action = actions[agent_id]
            action = torch.clamp(action, min=-1.0, max=1.0)
            self._actions[agent_id] = action.clone()

            # Scale actions from [-1, 1] to physical units
            # [0 vx, 1 vy, 2 vz, 3 yaw_rate, 4 gimbal_yaw_rate, 5 gimbal_pitch_rate, 6 zoom_rate]
            self.cmd_vel[:, idx, 0:3] = action[:, 0:3] * self.cfg.max_lin_vel
            self.cmd_vel[:, idx, 3] = action[:, 3] * self.cfg.max_yaw_rate
            # Gimbal and zoom controllers expect normalized [-1, 1] input and scale internally
            self.cmd_vel[:, idx, 4] = action[:, 4]  # Gimbal yaw rate (normalized)
            self.cmd_vel[:, idx, 5] = action[:, 5]  # Gimbal pitch rate (normalized)
            self.cmd_vel[:, idx, 6] = action[:, 6]  # Zoom rate (normalized)

    def _apply_action(self):
        """Apply actions to the environment.

        Runs at simulation rate (not decimated).
        """
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            controller = self._controllers[agent_id]

            # Extract commands for this agent
            v_cmd = self.cmd_vel[:, idx, 0:3]
            yaw_rate_cmd = self.cmd_vel[:, idx, 3]
            gimbal_yaw_rate = self.cmd_vel[:, idx, 4]
            gimbal_pitch_rate = self.cmd_vel[:, idx, 5]
            zoom_rate = self.cmd_vel[:, idx, 6]

            # Step the DroneController
            # Returns: F_body, tau_body, (gimbal_yaw, gimbal_roll, gimbal_pitch), zoom_level
            F_body, tau_body, gimbal_targets, zoom_level = controller.step_policy(
                v_cmd=v_cmd,
                yaw_rate_cmd=yaw_rate_cmd,
                gimbal_yaw_rate_cmd=gimbal_yaw_rate,
                gimbal_pitch_rate_cmd=gimbal_pitch_rate,
                zoom_rate_cmd=zoom_rate,
                q_body=robot.data.root_quat_w,
                v_body=robot.data.root_lin_vel_w,
                omega_body=robot.data.root_ang_vel_b,
                sim_dt=self.cfg.sim.dt * self.cfg.decimation,
            )

            # Apply forces in body frame (Isaac Lab expects local frame)
            robot.set_external_force_and_torque(
                forces=F_body.unsqueeze(1),
                torques=tau_body.unsqueeze(1),
                body_ids=self._body_ids[agent_id],
            )

            # Apply gimbal targets
            # Add YAW_JOINT_OFFSET (π/2) to yaw: the body mesh is rotated 90° CCW
            # (visual forward = body +Y), so yaw_joint=0 in physics must correspond
            # to body +Y. The controller works in body +X forward convention.
            gimbal_yaw, gimbal_roll, gimbal_pitch = gimbal_targets
            gimbal_yaw_joint = gimbal_yaw + YAW_JOINT_OFFSET
            robot.set_joint_position_target(
                target=torch.stack([gimbal_pitch, gimbal_yaw_joint, gimbal_roll], dim=-1),
                joint_ids=[
                    self.gimbal_joint_idx[agent_id]["pitch"],
                    self.gimbal_joint_idx[agent_id]["yaw"],
                    self.gimbal_joint_idx[agent_id]["roll"],
                ],
            )

            # DEBUG: Print gimbal info for first agent, first env, every 100 steps
            if idx == 0 and hasattr(self, '_debug_counter'):
                self._debug_counter += 1
                if self._debug_counter % 100 == 0:
                    # Get actual joint positions
                    actual_pitch = robot.data.joint_pos[0, self.gimbal_joint_idx[agent_id]["pitch"]].item()
                    actual_yaw = robot.data.joint_pos[0, self.gimbal_joint_idx[agent_id]["yaw"]].item()
                    actual_roll = robot.data.joint_pos[0, self.gimbal_joint_idx[agent_id]["roll"]].item()

                    import math
                    print(f"\n=== GIMBAL DEBUG (step {self._debug_counter}) ===")
                    print(f"  CMD rates:  yaw={gimbal_yaw_rate[0].item():.3f}, pitch={gimbal_pitch_rate[0].item():.3f}")
                    print(f"  TARGETS:    yaw={math.degrees(gimbal_yaw[0].item()):.1f}°, "
                          f"roll={math.degrees(gimbal_roll[0].item()):.1f}°, "
                          f"pitch={math.degrees(gimbal_pitch[0].item()):.1f}°")
                    print(f"  ACTUAL:     yaw={math.degrees(actual_yaw):.1f}°, "
                          f"roll={math.degrees(actual_roll):.1f}°, "
                          f"pitch={math.degrees(actual_pitch):.1f}°")
                    print(f"  Joint IDs:  pitch={self.gimbal_joint_idx[agent_id]['pitch']}, "
                          f"yaw={self.gimbal_joint_idx[agent_id]['yaw']}, "
                          f"roll={self.gimbal_joint_idx[agent_id]['roll']}")
                    # Camera axis decomposition for pitch-roll diagnosis
                    if agent_id in self._cameras:
                        try:
                            cam = self._cameras[agent_id]
                            poses, quats = cam._view.get_world_poses()
                            q = quats[0]  # [w, x, y, z] prim quat in world
                            print(f"  CAM PRIM quat:  {[round(x, 4) for x in q.tolist()]}")
                            print(f"  BODY quat:      {[round(x, 4) for x in robot.data.root_quat_w[0].tolist()]}")

                            # Decompose prim quat to camera axes in world frame
                            # OpenGL camera: forward=-Z, up=+Y, right=+X
                            w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
                            # Rotation matrix from quat (maps camera frame to world)
                            R00 = 1 - 2*(y*y + z*z); R01 = 2*(x*y - w*z); R02 = 2*(x*z + w*y)
                            R10 = 2*(x*y + w*z); R11 = 1 - 2*(x*x + z*z); R12 = 2*(y*z - w*x)
                            R20 = 2*(x*z - w*y); R21 = 2*(y*z + w*x); R22 = 1 - 2*(x*x + y*y)

                            # Camera axes in world frame
                            cam_fwd = [-R02, -R12, -R22]   # -Z column
                            cam_up = [R01, R11, R21]        # +Y column
                            cam_right = [R00, R10, R20]     # +X column
                            print(f"  CAM forward(w): [{cam_fwd[0]:.3f}, {cam_fwd[1]:.3f}, {cam_fwd[2]:.3f}]")
                            print(f"  CAM up(w):      [{cam_up[0]:.3f}, {cam_up[1]:.3f}, {cam_up[2]:.3f}]")
                            print(f"  CAM right(w):   [{cam_right[0]:.3f}, {cam_right[1]:.3f}, {cam_right[2]:.3f}]")

                            # Gimbal world-frame angles for context
                            gimbal_ctrl = self._controllers[agent_id].gimbal_controller
                            print(f"  WORLD az/el:    az={math.degrees(gimbal_ctrl.azimuth_world[0].item()):.1f}°, "
                                  f"el={math.degrees(gimbal_ctrl.elevation_world[0].item()):.1f}°")
                            print(f"  STAB roll:      {math.degrees(actual_roll):.1f}° (target={math.degrees(gimbal_roll[0].item()):.1f}°)")
                        except Exception as e:
                            print(f"  CAM DEBUG err: {e}")
            elif idx == 0 and not hasattr(self, '_debug_counter'):
                self._debug_counter = 0

            # Store zoom level for observations
            self.zoom_level[:, idx] = zoom_level

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents.

        Returns:
            Dictionary mapping agent_id to observation tensor (N, 18).
        """
        camera_poses = {}
        camera_intrinsics = {}
        image_shapes = {}
        agent_poses = {}

        target_pos = self.target.data.root_pos_w
        target_quat = self.target.data.root_quat_w
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(1)
        if target_quat.ndim == 2:
            target_quat = target_quat.unsqueeze(1)

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Use computed camera pose (like iris_ma5) for consistent frustum visualization
            # This ensures gimbal stabilization is properly reflected in frustum orientation
            # The computed orientation is in world convention with +Z forward
            gimbal_joint_pos = torch.stack(
                [
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]],
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]],
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["roll"]],
                ],
                dim=-1,
            )
            camera_poses[agent_id] = (
                compute_camera_position(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    self._camera_offset_position_b,
                ),
                compute_camera_orientation_from_gimbal(
                    robot.data.root_quat_w,
                    gimbal_joint_pos,
                    self._camera_offset_rotation_b,
                ),
            )

            intrinsic = self._camera_intrinsics_base.clone()
            intrinsic[:, 0, 0] *= self.zoom_level[:, idx]
            intrinsic[:, 1, 1] *= self.zoom_level[:, idx]
            camera_intrinsics[agent_id] = intrinsic
            image_shapes[agent_id] = self._camera_image_shape
            agent_poses[agent_id] = (robot.data.root_pos_w, robot.data.root_quat_w)

            # Update TiledCamera intrinsics based on zoom level (runs at policy frequency)
            if agent_id in self._cameras:
                self._cameras[agent_id].set_intrinsic_matrices_batched(
                    intrinsic,
                    self.cfg.camera.spawn.focal_length * self.zoom_level[:, idx],
                )

            # Debug: compare frustum vs TiledCamera orientation (every 200 steps, first agent only)
            if idx == 0 and hasattr(self, '_debug_counter') and self._debug_counter % 200 == 0 and self._debug_counter > 0:
                try:
                    from isaaclab.utils.math import quat_rotate
                    frustum_quat = camera_poses[agent_id][1][0]  # first env
                    frustum_fwd = quat_rotate(frustum_quat.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=self.device))[0]
                    print(f"\n=== CAMERA vs FRUSTUM DEBUG (step {self._debug_counter}) ===")
                    print(f"  Frustum quat:       {[round(x, 4) for x in frustum_quat.tolist()]}")
                    print(f"  Frustum fwd (+Z):   {[round(x, 4) for x in frustum_fwd.tolist()]}")
                    print(f"  Body quat:          {[round(x, 4) for x in robot.data.root_quat_w[0].tolist()]}")
                    if agent_id in self._cameras:
                        cam = self._cameras[agent_id]
                        try:
                            cam.update(dt=0.0)
                            cam_quat_world = cam.data.quat_w_world[0]
                            cam_fwd = quat_rotate(cam_quat_world.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=self.device))[0]
                            print(f"  Camera quat(world): {[round(x, 4) for x in cam_quat_world.tolist()]}")
                            print(f"  Camera fwd(+X):     {[round(x, 4) for x in cam_fwd.tolist()]}")
                        except Exception as e:
                            print(f"  Camera data error: {e}")
                            # Read pose directly from USD prim
                            try:
                                poses, quats = cam._view.get_world_poses()
                                print(f"  Camera prim quat:   {[round(x, 4) for x in quats[0].tolist()]}")
                            except Exception as e2:
                                print(f"  Camera prim error: {e2}")
                except Exception as e:
                    print(f"  Debug error: {e}")

        self.bbox_raycaster_v2.update(
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            target_poses=(target_pos, target_quat),
            agent_poses=agent_poses,
            image_shapes=image_shapes,
        )

        obs = {}
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Get gimbal angles from joints
            gimbal_yaw = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]]
            gimbal_pitch = robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]]

            bbox = self.bbox_raycaster_v2.data.bboxes_normalized[:, idx, 0, :]
            bbox_empty = self.bbox_raycaster_v2.data.bbox_empty[:, idx, 0].float().unsqueeze(-1)

            # Concatenate observation: pos(3) + vel(3) + quat(4) + gimbal_yaw(1) + gimbal_pitch(1) + zoom(1) + bbox(4) + bbox_empty(1)
            obs[agent_id] = torch.cat(
                [
                    robot.data.root_pos_w,  # (N, 3)
                    robot.data.root_lin_vel_w,  # (N, 3)
                    robot.data.root_quat_w,  # (N, 4)
                    gimbal_yaw.unsqueeze(-1),  # (N, 1)
                    gimbal_pitch.unsqueeze(-1),  # (N, 1)
                    self.zoom_level[:, idx : idx + 1],  # (N, 1)
                    bbox,  # (N, 4)
                    bbox_empty,  # (N, 1)
                ],
                dim=-1,
            )

        # Cache data for debug visualization callback
        self._vis_camera_poses = camera_poses
        self._vis_target_pos = self.target.data.root_pos_w
        self._vis_bbox_empty = {
            agent_id: self.bbox_raycaster_v2.data.bbox_empty[:, idx, 0]
            for idx, agent_id in enumerate(self.cfg.possible_agents)
        }
        self._vis_zoom_levels = {
            agent_id: self.zoom_level[:, idx]
            for idx, agent_id in enumerate(self.cfg.possible_agents)
        }

        return obs

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """Get rewards for all agents.

        Simple reward: negative distance to target for each agent.

        Returns:
            Dictionary mapping agent_id to reward tensor (N,).
        """
        rewards = {}
        target_pos = self.target.data.root_pos_w

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            drone_pos = robot.data.root_pos_w

            # Simple distance-based reward
            dist = torch.norm(drone_pos - target_pos, dim=-1)
            rewards[agent_id] = -dist * 0.1

        return rewards

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get termination and truncation flags for all agents.

        Returns:
            Tuple of (terminated, truncated) dictionaries.
        """
        terminated = {}
        truncated = {}

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]

            # Terminate if drone goes too low (crashed)
            pos_z = robot.data.root_pos_w[:, 2]
            crashed = pos_z < 0.5

            terminated[agent_id] = crashed
            truncated[agent_id] = time_out & ~crashed

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at specified indices.

        Args:
            env_ids: Environment indices to reset.
        """
        super()._reset_idx(env_ids)

        # Reset each robot to initial position
        num_reset = len(env_ids)

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Calculate initial position in a formation around origin
            # Agents positioned in a triangle formation
            angle = idx * 2 * 3.14159 / len(self.cfg.possible_agents)
            offset_x = 5.0 * torch.cos(torch.tensor(angle))
            offset_y = 5.0 * torch.sin(torch.tensor(angle))

            # Reset root state
            root_pos = self._terrain.env_origins[env_ids].clone()
            root_pos[:, 0] += offset_x
            root_pos[:, 1] += offset_y
            root_pos[:, 2] = 3.0  # Initial height

            root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_reset, 4)
            root_vel = torch.zeros(num_reset, 6, device=self.device)

            robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
            robot.write_root_velocity_to_sim(root_vel, env_ids)

            # Reset joint positions (gimbal to neutral)
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            # Reset controller state
            self._controllers[agent_id].reset(env_ids)

        # Reset target position
        target_pos = self._terrain.env_origins[env_ids].clone()
        target_pos[:, 2] = 3.5  # Target height
        target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_reset, 4)
        target_vel = torch.zeros(num_reset, 6, device=self.device)

        self.target.write_root_pose_to_sim(torch.cat([target_pos, target_quat], dim=-1), env_ids)
        self.target.write_root_velocity_to_sim(target_vel, env_ids)

        # Reset command buffers
        self.cmd_vel[env_ids] = 0.0
        self.zoom_level[env_ids] = 1.0
        self.cmd_gimbal_yaw[env_ids] = 0.0
        self.cmd_gimbal_pitch[env_ids] = 0.0

    # ==================================================================================
    # Debug Visualization
    # ==================================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function creates or destroys the visualization objects based on the
        debug_vis flag. Called by set_debug_vis() from DirectMARLEnv.

        Args:
            debug_vis: Whether to enable debug visualization.
        """
        if debug_vis:
            # Create visualization if not exists
            if self._visualization is None:
                self._visualization = CustomVisualization(
                    num_envs=self.num_envs,
                    possible_agents=self.cfg.possible_agents,
                    camera_cfg=self.cfg.camera,
                    device=self.device,
                )
        else:
            # Destroy visualization
            if self._visualization is not None:
                # Clear any drawn lines before destroying
                for agent_id in self.cfg.possible_agents:
                    self._visualization.camera_frustum[agent_id].clear()
                # Hide frame marker USD prims
                self._visualization.frame_visualizer.set_visibility(False)
                self._visualization = None

    def _debug_vis_callback(self, event):
        """Debug visualization callback called each frame.

        This draws camera frustums and detection indicators for all agents.
        Called automatically via the post-update event subscription when
        debug visualization is enabled.

        Args:
            event: Event data from the simulation app (unused).
        """
        if self._visualization is None:
            return

        # Skip if visualization data not yet populated
        if not self._vis_camera_poses or self._vis_target_pos is None:
            return

        # Update frame counter for warmup (prevents GPU crashes)
        self._visualization.step()

        # Draw visualizations
        self._visualization.update(
            camera_poses=self._vis_camera_poses,
            target_pos=self._vis_target_pos,
            bbox_empty=self._vis_bbox_empty,
            zoom_levels=self._vis_zoom_levels,
        )

        # Draw frame axes on body and gimbal links
        link_poses: Dict[str, tuple] = {}
        for link_name in FRAME_LINKS:
            all_pos = []
            all_quat = []
            for agent_id in self.cfg.possible_agents:
                robot = self._robots[agent_id]
                body_idx = self._frame_link_ids[agent_id][link_name]
                all_pos.append(robot.data.body_pos_w[:, body_idx])
                all_quat.append(robot.data.body_quat_w[:, body_idx])
            link_poses[link_name] = (
                torch.cat(all_pos, dim=0),
                torch.cat(all_quat, dim=0),
            )
        self._visualization.update_frames(link_poses)
