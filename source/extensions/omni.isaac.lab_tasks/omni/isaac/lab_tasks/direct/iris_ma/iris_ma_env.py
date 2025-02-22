# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectMARLEnv
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, matrix_from_quat, matrix_from_euler
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, euler_xyz_from_quat
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

from .iris_ma_env_cfg import IrisEnvCfg


class IrisEnv(DirectMARLEnv):
    cfg: IrisEnvCfg

    def __init__(self, cfg: IrisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = 6 #cfg.action_spaces["right_hand"]

        # buffers for position targets
        self.right_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.right_hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.num_hand_dofs*2, device=self.device)
        self.right_hand_thrust = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_moment = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_gimbal = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_previous_thrust = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_previous_moment = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_previous_gimbal = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_vehicle_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_target_pixels = torch.zeros(self.num_envs, 2, device=self.device)
        self.right_hand_target_visible = torch.zeros(self.num_envs, 2, device=self.device) # [right visible, left visible]
        self.right_hand_target_relative_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_left_hand_relative_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_hand_ray_target_distance = torch.zeros(self.num_envs, 1, device=self.device)

        self.left_hand_thrust = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_moment = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_gimbal = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_previous_thrust = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_previous_moment = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_previous_gimbal = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_vehicle_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_target_pixels = torch.zeros(self.num_envs, 2, device=self.device)
        self.left_hand_target_visible = torch.zeros(self.num_envs, 2, device=self.device) # [left visible, right visible]
        self.left_hand_target_relative_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_right_hand_relative_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_hand_ray_target_distance = torch.zeros(self.num_envs, 1, device=self.device)

        # Goal position
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "right_hand_ray_target_distance",
                "left_hand_ray_target_distance",
                "right_hand_target_distance",
                "left_hand_target_distance",            
                ]
        }
        # Get specific body indices
        # self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self.right_hand.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
    
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.right_hand = Articulation(self.cfg.right_robot_cfg)
        self.right_hand_body_id = self.right_hand.find_bodies("body")[0]
        self.left_hand = Articulation(self.cfg.left_robot_cfg)
        self.left_hand_body_id = self.left_hand.find_bodies("body")[0]
        # self.object = RigidObject(self.cfg.object_cfg)
        self.target_pos_w

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["right_robot"] = self.right_hand
        self.scene.articulations["left_robot"] = self.left_hand
        self.scene.rigid_objects["object"] = self.object

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self.right_hand_thrust[:, 2] = self.cfg.thrust_to_weight * self._robot_weight * self._actions[:, 0]
        self.right_hand_moment[:, :] = self.cfg.moment_scale * self._actions[:, 1:4]
        self.right_hand_gimbal = self.compute_stabilizing_commands(self, self.right_hand_vehicle_rpy)
        self.right_hand_gimbal[:, 1] += self.cfg.gimbal_scale * self._actions[:, 4]
        self.right_hand_gimbal[:, 2] += self.cfg.gimbal_scale * self._actions[:, 5]
        self.left_hand_thrust[:, 2] = self.cfg.thrust_to_weight * self._robot_weight * self._actions[:, self.num_hand_dofs]
        self.left_hand_moment[:, :] = self.cfg.moment_scale * self._actions[:, self.num_hand_dofs+1:self.num_hand_dofs+4]
        self.left_hand_gimbal = self.compute_stabilizing_commands(self, self.left_hand_vehicle_rpy)
        self.left_hand_gimbal[:, 1] += self.cfg.gimbal_scale * self._actions[:, self.num_hand_dofs+4]
        self.left_hand_gimbal[:, 2] += self.cfg.gimbal_scale * self._actions[:, self.num_hand_dofs+5]
        # self.target_pos_w += self.target_vel_w * self.step_dt

    def _apply_action(self):
        self.right_hand.set_external_force_and_torque(self.right_hand_thrust, self.right_hand_moment, body_ids=self.right_hand_body_id)
        self.left_hand.set_external_force_and_torque(self.left_hand_thrust, self.left_hand_moment, body_ids=self.left_hand_body_id)
        self.right_hand.set_joint_position_target(
            self.right_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self.left_hand.set_joint_position_target(
            self.left_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self.right_hand_thrust[:,:] = self.right_hand_previous_thrust[:,:]
        self.right_hand_moment[:,:] = self.right_hand_previous_moment[:,:]
        self.right_hand_gimbal[:,:] = self.right_hand_previous_gimbal[:,:]
        self.left_hand_thrust[:,:] = self.left_hand_previous_thrust[:,:]
        self.left_hand_moment[:,:] = self.left_hand_previous_moment[:,:]
        self.left_hand_gimbal[:,:] = self.left_hand_previous_gimbal[:,:]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        right_hand_desired_pos_b, _ = subtract_frame_transforms(
            self.right_hand.data.root_state_w[:, :3], self.right_hand.data.root_state_w[:, 3:7], self.target_pos_w
        )
        left_hand_desired_pos_b, _ = subtract_frame_transforms(
            self.left_hand.data.root_state_w[:, :3], self.left_hand.data.root_state_w[:, 3:7], self.target_pos_w
        )
        self.right_hand_vehicle_rpy = euler_xyz_from_quat(self.right_hand.data.root_quat_w)
        self.left_hand_vehicle_rpy = euler_xyz_from_quat(self.left_hand.data.root_quat_w)
        self.right_hand_left_hand_relative_pos = self.left_hand.data.root_pos_w - self.right_hand.data.root_pos_w
        self.left_hand_right_hand_relative_pos = -self.right_hand_left_hand_relative_pos
        right_hand_ray_target_distance, is_in_front = self.compute_ray_target_distance(self.right_hand.data.root_quat_w, self.right_hand_gimbal, self.right_hand_target_relative_pos)
        self.right_hand_ray_target_distance = right_hand_ray_target_distance if right_hand_ray_target_distance[:,0] < 2 and is_in_front else 100
        left_hand_ray_target_distance, is_in_front = self.compute_ray_target_distance(self.left_hand.data.root_quat_w, self.left_hand_gimbal, self.left_hand_target_relative_pos)
        self.left_hand_ray_target_distance = left_hand_ray_target_distance if left_hand_ray_target_distance[:,0] < 2 and is_in_front else 100

        observations = {
            "right_hand": torch.cat(
                (
                    self.right_hand.data.root_quat_w,
                    self.cfg.vel_obs_scale * self.right_hand.data.root_lin_vel_w,
                    self.right_hand.data.root_ang_vel_w,
                    self.right_hand.data.projected_gravity_b,
                    self.right_hand.data.root_quat_w,
                    self.right_hand.data.root_lin_vel_w,
                    self.right_hand.data.root_ang_vel_w,
                    self.right_hand.data.projected_gravity_b,
                    right_hand_desired_pos_b,
                    self._actions[:,:self.num_hand_dofs],
                    self.right_hand_left_hand_relative_pos,
                    self.right_hand_ray_target_distance,
                ),
                dim=-1,
            ),
            "left_hand": torch.cat(
                (
                    self.left_hand.data.root_quat_w,
                    self.cfg.vel_obs_scale * self.left_hand.data.root_lin_vel_w,
                    self.left_hand.data.root_ang_vel_w,
                    self.left_hand.data.projected_gravity_b,
                    self.left_hand.data.root_quat_w,
                    self.left_hand.data.root_lin_vel_w,
                    self.left_hand.data.root_ang_vel_w,
                    self.left_hand.data.projected_gravity_b,
                    left_hand_desired_pos_b,
                    self._actions[:,:self.num_hand_dofs],
                    self.left_hand_right_hand_relative_pos,
                    self.left_hand_ray_target_distance,
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self.target_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "right_hand_ray_target_distance": self.right_hand_ray_target_distance * self.cfg.ray_target_distance_scale * self.step_dt,
            "left_hand_ray_target_distance": self.left_hand_ray_target_distance * self.cfg.ray_target_distance_scale * self.step_dt,
            "right_hand_target_distance": torch.norm(self.right_hand.data.root_com_pos_w - self.target_pos_w, dim=-1) * self.cfg.target_distance_scale * self.step_dt,
            "left_hand_target_distance": torch.norm(self.left_hand.data.root_com_pos_w - self.target_pos_w, dim=-1) * self.cfg.target_distance_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
                self.right_hand.data.root_pos_w[:, 2] < 0.1, 
                self.right_hand.data.root_pos_w[:, 2] > 20.0
        )
        died = torch.logical_or(
                died,
                torch.norm(self.right_hand_left_hand_relative_pos, dim=-1) < 2.0
        )
        died = torch.logical_or(
                died,
                torch.norm(self.right_hand.data.root_com_pos_w - self.target_pos_w, dim=-1) < 2.0
        )
        died = torch.logical_or(
                died,
                torch.norm(self.left_hand.data.root_com_pos_w - self.target_pos_w, dim=-1) < 2.0
        )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.right_hand._ALL_INDICES
            env_ids = self.left_hand._ALL_INDICES

        # Logging
        right_hand_final_distance_to_target = torch.linalg.norm(
            self.target_pos_w[env_ids] - self.right_hand.data.root_pos_w[env_ids], dim=1
        ).mean()
        left_hand_final_distance_to_target = torch.linalg.norm(
            self.target_pos_w[env_ids] - self.left_hand.data.root_pos_w[env_ids], dim=1
        ).mean()
        right_hand_ray_target_distance = torch.linalg.norm(
            self.right_hand_ray_target_distance[env_ids], dim=1
        ).mean()
        left_hand_ray_target_distance = torch.linalg.norm(
            self.left_hand_ray_target_distance[env_ids], dim=1
        ).mean()
        final_distance_to_goal = torch.linalg.norm(
            self.target_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/right_hand_ray_target_distance"] = right_hand_ray_target_distance.item()
        extras["Metrics/right_hand_final_distance_to_target"] = right_hand_final_distance_to_target.item()
        extras["Metrics/left_hand_ray_target_distance"] = left_hand_ray_target_distance.item()
        extras["Metrics/left_hand_final_distance_to_target"] = left_hand_final_distance_to_target.item()

        self.extras["log"].update(extras)

        self.right_hand.reset(env_ids)
        self.left_hand.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self.target_pos_w[env_ids, :2] = torch.zeros_like(self.target_pos_w[env_ids, :2]).uniform_(-10.0, 10.0)
        self.target_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self.target_pos_w[env_ids, 2] = torch.zeros_like(self.target_pos_w[env_ids, 2]).uniform_(0.5, 5)
        self.target_vel_w[env_ids] = torch.ones_like(self.target_vel_w[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self.right_hand.data.default_joint_pos[env_ids]
        joint_vel = self.right_hand.data.default_joint_vel[env_ids]
        default_root_state = self.right_hand.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.right_hand.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.right_hand.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.right_hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        joint_pos = self.left_hand.data.default_joint_pos[env_ids]
        joint_vel = self.left_hand.data.default_joint_vel[env_ids]
        default_root_state = self.left_hand.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.left_hand.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.left_hand.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.left_hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self.target_pos_w)

    def compute_ray_target_distance(self, vehicle_quat_w, gimbal_rpy, target_relative_pos) -> tuple[torch.Tensor, bool]:
        """
        Calculate the distance between a ray (from drone + gimbal) and a target point.
        
        Args:
            drone_quaternion (np.array): [qw, qx, qy, qz] quaternion representing drone orientation
            gimbal_rpy (np.array): [roll, pitch, yaw] gimbal angles in radians
            target_position (np.array): [x, y, z] target position in world frame
            
        Returns:
            float: Shortest distance from ray to target
        """
        # Convert drone quaternion to rotation matrix
        vehicle_rotation = matrix_from_quat(vehicle_quat_w)
        gimbal_rotation = matrix_from_euler(gimbal_rpy, 'xyz')
        
        # Combined rotation (drone orientation + gimbal angles)
        total_rotation = vehicle_rotation @ gimbal_rotation
        
        # Ray direction in world frame (assuming camera points along body x-axis)
        ray_direction = total_rotation @ self.x_unit_tensor
        ray_direction_norm = torch.sqrt(torch.sum(ray_direction * ray_direction, dim=1))
        ray_direction = ray_direction / ray_direction_norm[:, None]        
        # Ray origin is drone position (assumed to be at [0,0,0] in world frame)
        ray_origin = torch.zeros_like(target_relative_pos, device=self.device)
        
        # Vector from ray origin to target
        to_target = target_relative_pos - ray_origin
        
        # Calculate perpendicular distance using cross product
        # d = ||(p - o) × v|| / ||v||
        # where p is target point, o is ray origin, v is ray direction
        distance = torch.norm(torch.cross(to_target, ray_direction, dim=-1), dim=-1)
        
        # Calculate dot product between ray direction and vector to target
        # If dot product is positive, target is in front of ray
        dot_product = torch.sum(ray_direction * to_target, dim=-1)  # [B]
        is_target_in_front = True if dot_product > 0 else False # [B]
        
        return distance, is_target_in_front


    def compute_stabilizing_commands(self, vehicle_rpy):
        """Compute joint commands to counteract vehicle rotation"""
        # For perfect stabilization, we want to counter the vehicle rotation
        # Note: The signs might need to be inverted depending on joint conventions
        stabilizing_rpy = torch.zeros(self.num_envs, 1, 3, device=self.device)
        stabilizing_rpy[:, 0, 0] = vehicle_rpy[0][:]
        stabilizing_rpy[:, 0, 1] = vehicle_rpy[1][:]
        stabilizing_rpy[:, 0, 2] = -np.pi / 2 + vehicle_rpy[2][:]
        # stabilizing_roll = vehicle_rpy[0][:]
        # stabilizing_pitch = vehicle_rpy[1][:]
        # stabilizing_yaw = -np.pi/2 + vehicle_rpy[2][:]

        return stabilizing_rpy

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


