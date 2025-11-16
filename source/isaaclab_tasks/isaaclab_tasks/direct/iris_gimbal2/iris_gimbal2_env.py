# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
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
import math

##
# Pre-defined configs
##
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab_assets import IRIS_GIMBAL2_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .stabilizer import DroneStabilizingController
from .point_mass import PointMass

import carb
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG 
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
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

import isaaclab.sim as sim_utils
import numpy as np

from .bbox_generator import BBoxGenerator
from .camera_frustrum import CameraFrustrum, create_camera_cfg_tensor, project_2d_to_3d

class IrisGimbal2EnvWindow(BaseEnvWindow):
    """Window manager for the Iris environment."""

    def __init__(self, env: IrisGimbal2Env, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class IrisGimbal2EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 30.0
    decimation = 2
    action_space = 6
    observation_space = 21
    state_space = 0
    debug_vis = True

    ui_window_class_type = IrisGimbal2EnvWindow

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

    # terrain = None
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
        debug_vis= False,
    )

    camera_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/pitch_link/camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=[
                "rgb",
                "semantic_segmentation",
            ],
            colorize_semantic_segmentation=True,
            colorize_instance_segmentation=True,
            colorize_instance_id_segmentation=True,
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

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
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

    frame_transformer_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/Robot/pitch_link"),
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/target"),
        ],
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    # robot
    robot: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 4.0
    moment_scale = 10.0
    yaw_moment_scale = 1.0

    # reward scales
    lin_vel_reward_scale = -0.01
    ang_vel_reward_scale = -0.02
    action_sum_reward_scale = -0.1
    action_weight = [1, 1, 2, 0.1, 0.03, 0.03]
    action_delta_reward_scale = -0.01
    action_delta_weight = [1, 1, 2, 0.1, 0.03, 0.03]
    distance_to_goal_reward_scale = 60.0
    yaw_reward_scale = -10
    yaw_rate_reward_scale = -0.001

    bbox_center_reward_scale = 60
    bbox_size_reward_scale = 60
    
    max_target_speed = 15.0
    max_lin_vel = 15.0
    max_yaw_rate = 20.0
    max_lin_acc = 3.0
    max_ang_acc = 30.0
    
    # target movement parameters
    target_acceleration_scale = 2.0
    target_velocity_damping = 0.95
    target_direction_change_prob = 0.01
    target_max_acceleration = 1.0

class IrisGimbal2Env(DirectRLEnv):
    cfg: IrisGimbal2EnvCfg

    def __init__(self, cfg: IrisGimbal2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._cmd_vel = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self.target_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0)
        self.last_target_vel = self.target_vel.clone()
        
        # Smooth movement variables
        self.target_acceleration = torch.zeros(self.num_envs, 6, device=self.device)
        self.target_desired_vel = torch.zeros(self.num_envs, 6, device=self.device).uniform_(-1.0, 1.0)
        self.target_vel_change_timer = torch.zeros(self.num_envs, device=self.device)

        self.camera_pos_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.camera_quat_world = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw_w = torch.zeros(self.num_envs, 1, device=self.device)
        self.bboxes = torch.zeros(self.num_envs, 4, device=self.device)  # (num_envs, 4 corners, 2D coordinates)
        self.bboxes_normalized = torch.zeros(self.num_envs, 4, device=self.device)  # (num_envs, 4 corners, 2D coordinates)
        self.bbox_valid_mask = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        self.camera_cfg_batch = create_camera_cfg_tensor(self.cfg.camera_cfg, self.num_envs, device=self.device)
        camera_offset_rot_single = torch.tensor(cfg.camera_cfg.offset.rot, device=self.device)
        self.camera_offset_rot_batch = camera_offset_rot_single.unsqueeze(0).expand(self.num_envs, -1)
        camera_offset_pos_single = torch.tensor(cfg.camera_cfg.offset.pos, device=self.device)
        self.camera_offset_pos_batch = camera_offset_pos_single.unsqueeze(0).expand(self.num_envs, -1)

        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device).repeat(self.num_envs, 1)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device).repeat(self.num_envs, 1)

        self.curriculum = torch.tensor(-1, device=self.device).repeat(self.num_envs, 1)
        self.last_curriculum = self.curriculum.clone()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "action_sum",
                "action_delta",
                # "distance_to_goal",
                # "yaw",
                # "yaw_rate",
                "bbox_center",
                "bbox_size",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self.gimbal_yaw_joint_idx = self._robot.find_joints("yaw_joint")[0][0]
        self.gimbal_roll_joint_idx = self._robot.find_joints("roll_joint")[0][0]
        self.gimbal_pitch_joint_idx = self._robot.find_joints("pitch_joint")[0][0]
        self.gimbal_dof_targets = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self.gimbal_dof_targets[:, self.gimbal_yaw_joint_idx] = 0
        self.gimbal_dof_targets[:, self.gimbal_roll_joint_idx] = 0
        self.gimbal_dof_targets[:, self.gimbal_pitch_joint_idx] = 0
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._moment_of_inertia = self._robot.root_physx_view.get_inertias()[0]
        self.target_speed = torch.zeros_like(self._desired_pos_w)
        self.max_lin_vel = self.cfg.max_lin_vel
        self.max_yaw_rate = self.cfg.max_yaw_rate   
        self.max_lin_acc = self.cfg.max_lin_acc
        self.max_ang_acc = self.cfg.max_ang_acc

        self._stabilizer = PointMass(0, self._robot_weight, self.num_envs, self.device)
        # self._stabilizer = DroneStabilizingController(self._robot_mass, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizer")
        frame_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self.frame_visualizer = VisualizationMarkers(frame_cfg)
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
        marker_cfg.markers["cuboid"].semantic_tags = [("class", "1")]
        # -- goal pose
        marker_cfg.prim_path = "/Visuals/FrameVisualizer/goal_position"
        self.goal_visualizer = VisualizationMarkers(marker_cfg)
        if DEBUG_DRAW and omni_debug_draw is not None:
            self.camera_frustrum = CameraFrustrum()
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        self.step_count = 0
        self._env_origins = self._compute_env_origins_grid(self.num_envs, self.cfg.scene.env_spacing)

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """Compute the origins of the environments in a grid based on configured spacing."""
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        # create a grid of origins
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device), torch.arange(num_cols, device=self.device), indexing="ij"
        )
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins
    
    def _setup_scene(self):
        self.cfg.robot.spawn.semantic_tags = [("class", "robot")]
        self._robot = Articulation(self.cfg.robot)

        if self.cfg.terrain is not None:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        else:
            self._terrain = None

        self._camera = None
        if DEBUG_DRAW:
            self._camera = TiledCamera(self.cfg.camera_cfg)

        self.cfg.target_cfg.spawn.semantic_tags = [("class", "target")]
        self.target = RigidObject(self.cfg.target_cfg)

        self.frame_transformer = FrameTransformer(self.cfg.frame_transformer_cfg)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self._robot
        if self._camera is not None:
            self.scene.sensors["tiled_camera"] = self._camera
        self.scene.rigid_objects["target"] = self.target

        # self.frame_transformer = FrameTransformer(self.scene["frame_transformer"])
        self.scene.sensors["frame_transformer"] = self.frame_transformer
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        if torch.isnan(actions).any():
            return
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._cmd_vel[:, 0, :3] = self._actions[:, :3] * self.max_lin_vel
        # self._cmd_vel[:, 0, 2] = torch.zeros_like(self._actions[:, 2])
        self._cmd_vel[:, 0, 5] = self._actions[:, 3] * self.max_yaw_rate
        self._cmd_vel_lin = torch.zeros_like(self._cmd_vel[:, 0, :3])
        # self._cmd_vel_lin[:, 0] = torch.ones_like(self._cmd_vel)[:,0]
        self._cmd_vel_ang = torch.zeros_like(self._cmd_vel[:, 0, 5])

        self._thrust[:,0,:], self._moment[:,0,:] = self._stabilizer.compute_control(
            self._cmd_vel[:, 0, :3], self._cmd_vel[:, 0, 5], 
            self._robot.data.root_state_w[:, 3:7],
            self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_b, self.step_dt
        )

        self.new_vel_lin_w = self._robot.data.root_lin_vel_w * (1 - self.step_dt) + quat_rotate(
            self._robot.data.root_state_w[:, 3:7], self._thrust[:, 0, :]) * self.step_dt
        self.new_vel_lin_w[:, 2] = torch.clamp(self.new_vel_lin_w[:, 2], min=-self.max_lin_vel, max=self.max_lin_vel)
        self.new_vel_ang_w = self._robot.data.root_ang_vel_w * (1 - 3 * self.step_dt) + quat_rotate(
            self._robot.data.root_state_w[:, 3:7], self._moment[:, 0, :]) * self.step_dt
        self.new_vel_ang_w[:, 0] = torch.zeros_like(self.new_vel_ang_w[:, 0])
        self.new_vel_ang_w[:, 1] = torch.zeros_like(self.new_vel_ang_w[:, 1])
        self.new_vel_ang_w[:, 2] = torch.clamp(self.new_vel_ang_w[:, 2], -self.max_yaw_rate, self.max_yaw_rate)
        
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_state_w[:, 3:7])
        curr_quat_w_yaw = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)
        roll_b, pitch_b, _ = euler_xyz_from_quat(quat_mul(self._robot.data.root_state_w[:, 3:7], quat_inv(curr_quat_w_yaw)))
        self.gimbal_dof_targets[:, self.gimbal_yaw_joint_idx] += self._actions[:, 4] * self.step_dt * 2 * math.pi
        self.gimbal_dof_targets[:, self.gimbal_yaw_joint_idx] = torch.clamp(self.gimbal_dof_targets[:, self.gimbal_yaw_joint_idx], -math.pi*2/3, math.pi*2/3)
        self.gimbal_dof_targets[:, self.gimbal_roll_joint_idx] = self._stabilizer.wrap_to_pi(-roll_b)
        self.gimbal_dof_targets[:, self.gimbal_pitch_joint_idx] += self._actions[:, 5] * self.step_dt * 2 * math.pi
        self.gimbal_dof_targets[:, self.gimbal_pitch_joint_idx] = torch.clamp(self.gimbal_dof_targets[:, self.gimbal_pitch_joint_idx], -math.pi*1/3, math.pi*1/3)

        # Compute actual camera position based on robot state and gimbal orientation
        # Get robot base position and orientation
        robot_pos = self._robot.data.root_pos_w
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        # Get gimbal joint positions
        gimbal_yaw = self._robot.data.joint_pos[:, self.gimbal_yaw_joint_idx]
        gimbal_roll = self._robot.data.joint_pos[:, self.gimbal_roll_joint_idx] 
        gimbal_pitch = self._robot.data.joint_pos[:, self.gimbal_pitch_joint_idx]
        
        # Create gimbal orientation quaternion (yaw -> roll -> pitch)
        gimbal_quat = quat_mul(
            quat_mul(
                quat_from_euler_xyz(torch.zeros_like(gimbal_yaw), torch.zeros_like(gimbal_yaw), gimbal_yaw),
                quat_from_euler_xyz(gimbal_roll, torch.zeros_like(gimbal_roll), torch.zeros_like(gimbal_roll))
            ),
            quat_from_euler_xyz(torch.zeros_like(gimbal_pitch), gimbal_pitch, torch.zeros_like(gimbal_pitch))
        )

        self.camera_quat_world = quat_mul(robot_quat, quat_mul(gimbal_quat, self.camera_offset_rot_batch))
        
        # Apply camera offset from configuration (pos=(0.2, 0.0, -0.1))
        # camera_offset = torch.tensor(self.cfg.camera_cfg.offset.pos, device=self.device).expand(self.num_envs, -1)
        camera_offset = torch.tensor((0, 0, 0.1), device=self.device).expand(self.num_envs, -1)
        # self.camera_pos_world = robot_pos + quat_rotate(self.camera_quat_world, camera_offset)
        self.camera_pos_world = self.frame_transformer.data.target_pos_w[:,0]
        
        bbox_gen = BBoxGenerator(camera_width=self.cfg.camera_cfg.width, camera_height=self.cfg.camera_cfg.height,
                                focal_length=self.cfg.camera_cfg.spawn.focal_length, horizontal_aperture= self.cfg.camera_cfg.spawn.horizontal_aperture)
        self.bboxes, self.bbox_valid_mask[:, 0] = bbox_gen.generate_2d_bbox(
            self.target.data.root_state_w[:, :3], self.target.data.root_state_w[:, 3:7],
            self.frame_transformer.data.target_pos_w[:,0], self.camera_quat_world, min_bbox_size=int(0.1*self.cfg.camera_cfg.width),
        )
        self.bboxes_normalized = self.bboxes.clone()
        self.bboxes_normalized[:, 0] = torch.where(self.bbox_valid_mask.squeeze(-1), self.bboxes[:, 0] / self.cfg.camera_cfg.width, torch.ones_like(self.bboxes[:, 0]) * (-1.0))
        self.bboxes_normalized[:, 1] = torch.where(self.bbox_valid_mask.squeeze(-1), self.bboxes[:, 1] / self.cfg.camera_cfg.height, torch.ones_like(self.bboxes[:, 1]) * (-1.0))
        self.bboxes_normalized[:, 2] = torch.where(self.bbox_valid_mask.squeeze(-1), self.bboxes[:, 2] / self.cfg.camera_cfg.width, torch.ones_like(self.bboxes[:, 2]) * (-1.0))
        self.bboxes_normalized[:, 3] = torch.where(self.bbox_valid_mask.squeeze(-1), self.bboxes[:, 3] / self.cfg.camera_cfg.height, torch.ones_like(self.bboxes[:, 3]) * (-1.0))
        # carb.log_warn(f"body pose: {[f'{x:.2f}' for x in self._robot.data.root_state_w[0, :3].cpu().tolist()]}")
        # carb.log_warn(f"camera pose: {[f'{x:.2f}' for x in self.camera_pos_world[0].cpu().tolist()]}, target pose: {[f'{x:.2f}' for x in self.target.data.root_state_w[0, :3].cpu().tolist()]}")
        # carb.log_warn(f"camera ftp: {[f'{x:.2f}' for x in self.frame_transformer.data.target_pos_w[0, 0].tolist()]}")
        # r, p, y = euler_xyz_from_quat(self.frame_transformer.data.target_quat_w[:1, 0])
        # carb.log_warn(f"camera ftq: {[f'{x:.2f}' for x in [self._stabilizer.wrap_to_pi(r).tolist()[0], self._stabilizer.wrap_to_pi(p).tolist()[0], self._stabilizer.wrap_to_pi(y).tolist()[0]]]}")
        # carb.log_warn(f"gimbal yaw: {self._robot.data.joint_pos[0, self.gimbal_yaw_joint_idx].item():.2f}, gimbal roll: {self._robot.data.joint_pos[0, self.gimbal_roll_joint_idx].item():.2f}, gimbal pitch: {self._robot.data.joint_pos[0, self.gimbal_pitch_joint_idx].item():.2f}")
        # bb = self.bboxes[0].cpu().tolist()
        # carb.log_warn(f"bboxes: [f'{(bb[0]+bb[2])/2:.0f}', f'{(bb[1]+bb[3])/2:.0f}], valid_mask: {self.bbox_valid_mask[0].cpu().tolist()}]")

        # Smooth target movement with continuous acceleration
        # if self.step_count > 14000:
        self._update_target_movement()

        # Bounce up when target is too low
        up_vel = self.target_vel.clone()
        up_vel[..., 2] = -self.target_vel[..., 2]
        target_lower_bound = torch.ones_like(self.target.data.root_state_w, device=self.device) * 1.0
        self.target_vel[..., 2] = torch.where(self.target.data.root_state_w[:, 2] < target_lower_bound[:, 2], up_vel[..., 2], self.target_vel[..., 2])
        self.step_count += 1

    def _apply_action(self):
        # self._robot.write_root_link_velocity_to_sim(self._cmd_vel[self._body_id, 0, :6], env_ids=self._body_id)
        # self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        self._robot.write_root_velocity_to_sim(
            torch.cat((self.new_vel_lin_w, self.new_vel_ang_w), dim=1), env_ids=self._robot._ALL_INDICES
        )
        self._robot.set_joint_position_target(self.gimbal_dof_targets)

        self.target.write_root_com_velocity_to_sim(self.target_vel)

        # self.frame_visualizer.visualize(self.camera_pos_world, self.camera_quat_world)
        robot_pos_w = self.frame_transformer.data.source_pos_w
        robot_quat_w = self.frame_transformer.data.source_quat_w
        camera_pos_w = self.frame_transformer.data.target_pos_w[:, 0]
        camera_quat_w = self.frame_transformer.data.target_quat_w[:, 0]
        self.frame_transformer.update(dt=self.cfg.sim.dt)
        # self.frame_visualizer.visualize(
        #     torch.cat([robot_pos_w, camera_pos_w], dim=0), torch.cat([robot_quat_w, camera_quat_w], dim=0)
        # )
        self.goal_visualizer.visualize(self._desired_pos_w)
        if DEBUG_DRAW and hasattr(self, 'draw_interface'):
            self.draw_interface.clear_lines()
            # line_colors_yellow = [[1.0, 1.0, 0.0, 1.0]] * self.camera_pos_world.shape[0]
            # line_colors_green = [[0.0, 1.0, 0.0, 1.0]] * self.camera_pos_world.shape[0]
            line_colors_yellow = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
            line_colors_green = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
            line_colors = torch.where(self.bbox_valid_mask, line_colors_green, line_colors_yellow).tolist()
            line_thicknesses = [5.0] * self.camera_pos_world.shape[0]
            self.draw_interface.draw_lines(self.camera_pos_world.tolist(), self.target.data.root_pos_w.tolist(), line_colors, line_thicknesses)
            self.camera_frustrum.draw_frustrum(self.camera_pos_world, self.camera_quat_world, self.camera_cfg_batch, self.device)
        # self.frame_visualizer.visualize(self._robot.data.default_root_state[:, :3], self._robot.data.root_state_w[:, 3:7], marker_indices=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_state_w[:, :10], # 10
                self._robot.data.root_ang_vel_b[:, 2:3], # 1
                self._robot.data.body_lin_acc_w[:,0], # 3
                self._robot.data.joint_pos[:, self.gimbal_pitch_joint_idx:self.gimbal_pitch_joint_idx + 1], # 1
                self._robot.data.joint_pos[:, self.gimbal_yaw_joint_idx:self.gimbal_yaw_joint_idx + 1], # 1
                self.bboxes_normalized, # 4
                self.bbox_valid_mask.float(), # 1
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        action_sum = torch.sum(torch.square(self.action_weight * self._actions), dim=1)
        action_delta = torch.sum(torch.square(self.action_delta_weight * (self._actions - self._last_actions)), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh((distance_to_goal) / 0.8)
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_state_w[:, 3:7])
        roll = self._stabilizer.wrap_to_pi(roll)
        pitch = self._stabilizer.wrap_to_pi(pitch)
        yaw = self._stabilizer.wrap_to_pi(yaw)
        stabilize = torch.square(roll) + torch.square(pitch) #torch.square(roll / (45.0 / 180.0 * torch.pi)) + torch.square(pitch / (45.0 / 180.0 * torch.pi))
        yaw_error = torch.square(self._desired_yaw_w[:, 0] - yaw)
        yaw_rate = torch.square(self._robot.data.root_ang_vel_b[:, 2])
        bbox = (torch.square((self.bboxes[:, 0] + self.bboxes[:, 2])/2 - self.cfg.camera_cfg.width/2) / self.cfg.camera_cfg.width**2 
                + torch.square((self.bboxes[:, 1] + self.bboxes[:, 3])/2 - self.cfg.camera_cfg.height/2) / self.cfg.camera_cfg.height**2)
        bbox_center_mapped = (1 - torch.tanh(bbox / 0.8)) * self.bbox_valid_mask.squeeze(-1).float()
        bbox_size = (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1]) / (self.cfg.camera_cfg.width * self.cfg.camera_cfg.height)
        bbox_size_mapped = (1 - torch.tanh((bbox_size-0.2**2) / 0.8)) * self.bbox_valid_mask.squeeze(-1).float()
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "action_sum": action_sum * self.cfg.action_sum_reward_scale * self.step_dt,
            "action_delta": action_delta * self.cfg.action_delta_reward_scale * self.step_dt,
            # "yaw": yaw_error * self.cfg.yaw_reward_scale * self.step_dt,
            # "yaw_rate": yaw_rate * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "bbox_center": bbox_center_mapped * self.cfg.bbox_center_reward_scale * self.step_dt,
            "bbox_size": bbox_size_mapped * self.cfg.bbox_size_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        self._last_actions = self._actions.clone()
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self._robot.data.root_pos_w[:, 2] < 0.7
        # died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.7, self._robot.data.root_pos_w[:, 2] > 10.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        final_bbox_center = torch.mean(
            (self.bboxes[env_ids, 0] + self.bboxes[env_ids, 2]) / 2 / self.cfg.camera_cfg.width
            + (self.bboxes[env_ids, 1] + self.bboxes[env_ids, 3]) / 2 / self.cfg.camera_cfg.height
        )
        final_bbox_size = torch.mean(
            (self.bboxes[env_ids, 2] - self.bboxes[env_ids, 0]) * (self.bboxes[env_ids, 3] - self.bboxes[env_ids, 1])
            / (self.cfg.camera_cfg.width * self.cfg.camera_cfg.height)
        )
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
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/final_bbox_center"] = final_bbox_center.item()
        extras["Metrics/final_bbox_size"] = final_bbox_size.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        default_root_state = self._robot.data.default_root_state[env_ids]
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(default_root_state[:, :2])
        self._desired_pos_w[env_ids, 2] = torch.ones_like(default_root_state[:, 2]) * 3.5
        bbox_reward = (1 - torch.tanh(final_bbox_center / 0.8)) * self.cfg.bbox_center_reward_scale
        curriculum_rate = 0.01
        self.curriculum = torch.where(bbox_reward > 20, torch.ones_like(self.curriculum), self.last_curriculum)
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(2.0, 5.0)
        self._desired_pos_w[env_ids, 1] = torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(-1.0, 1.0)
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.5, 4.5)
        self._desired_pos_w[env_ids, :3] += self._env_origins[env_ids, :3]
        self._desired_yaw_w[env_ids, 0] = torch.zeros_like(self._desired_yaw_w[env_ids, 0]).uniform_(-3.14, 3.14)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, self.gimbal_yaw_joint_idx] = torch.zeros_like(joint_pos[:, self.gimbal_yaw_joint_idx]).uniform_(-3.14 * 2 / 3, 3.14 * 2 / 3)
        joint_pos[:, self.gimbal_pitch_joint_idx] = torch.zeros_like(joint_pos[:, self.gimbal_pitch_joint_idx]).uniform_(-3.14*1/3, 3.14*1/3)
        joint_pos[:, self.gimbal_roll_joint_idx] = torch.zeros_like(joint_pos[:, self.gimbal_roll_joint_idx])
        self.gimbal_dof_targets[env_ids] = joint_pos
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :2] = torch.zeros_like(default_root_state[:, :2])
        default_root_state[:, :3] += self._env_origins[env_ids]
        default_root_state[:, 2] += torch.ones_like(default_root_state[:, 2]) * 3.0
        # default_root_state[:, 2] += torch.zeros_like(default_root_state[:, 2]).uniform_(1.5, 4.5)
        # default_root_state[:, 3:7] = quat_from_euler_xyz(
        #     torch.zeros_like(default_root_state[:, 3]), 
        #     torch.zeros_like(default_root_state[:, 4]), 
        #     torch.ones_like(default_root_state[:, 5]) * 3.14
        # )
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- Project target from 2d to 3d position --- #
        robot_pos = default_root_state[:, :3]
        robot_quat = default_root_state[:, 3:7]
        gimbal_yaw = joint_pos[:, self.gimbal_yaw_joint_idx]
        gimbal_roll = joint_pos[:, self.gimbal_roll_joint_idx] 
        gimbal_pitch = joint_pos[:, self.gimbal_pitch_joint_idx]
        
        # Create gimbal orientation quaternion (yaw -> roll -> pitch)
        gimbal_quat = quat_mul(
            quat_mul(
                quat_from_euler_xyz(torch.zeros_like(gimbal_yaw), torch.zeros_like(gimbal_yaw), gimbal_yaw),
                quat_from_euler_xyz(gimbal_roll, torch.zeros_like(gimbal_roll), torch.zeros_like(gimbal_roll))
            ),
            quat_from_euler_xyz(torch.zeros_like(gimbal_pitch), gimbal_pitch, torch.zeros_like(gimbal_pitch))
        )

        camera_quat_world = quat_mul(robot_quat, quat_mul(gimbal_quat, self.camera_offset_rot_batch[env_ids]))
        camera_offset = torch.tensor((0, 0, 0.1), device=self.device).expand(len(env_ids), -1)
        camera_pos_world = robot_pos + quat_rotate(camera_quat_world, camera_offset)
        # --- sample random pixels --- #
        sampled_box_center = torch.zeros(len(env_ids), 2, device=self.device)
        sampled_box_center[:, 0] = torch.randint(0, self.cfg.camera_cfg.width, (len(env_ids),), device=self.device).float()
        sampled_box_center[:, 1] = torch.randint(0, self.cfg.camera_cfg.height, (len(env_ids),), device=self.device).float()
        sampled_distances = torch.zeros(len(env_ids), 1, device=self.device).uniform_(8.0, 12.0)
        target_pos = project_2d_to_3d(self.camera_cfg_batch[env_ids], sampled_box_center, sampled_distances, camera_pos_world, camera_quat_world, device=self.device)

        self.target.data.root_state_w[env_ids, :3] = target_pos
        self.target.data.root_state_w[env_ids, 7:] = torch.zeros_like(self.target.data.root_state_w[env_ids, 7:])
        self.target.write_root_pose_to_sim(self.target.data.root_state_w[env_ids, :7], env_ids)
        new_target_vel = self.last_target_vel[env_ids] * (1+curriculum_rate)
        self.target_vel[env_ids] = torch.where(new_target_vel > self.cfg.max_target_speed,
                                                new_target_vel,
                                                self.last_target_vel[env_ids])
        
        self._last_actions[env_ids] = torch.zeros_like(self._actions[env_ids])
        self._stabilizer.reset(env_ids)
        self.target_quat = quat_from_euler_xyz(
            torch.zeros_like(self._desired_yaw_w[:, 0]), 
            torch.zeros_like(self._desired_yaw_w[:, 0]), 
            self._desired_yaw_w[:, 0]
        )
        self.last_curriculum = self.curriculum.clone()
        self.last_target_vel = self.target_vel.clone()

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
            # self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
                # self.frame_visualizer.set_visibility(False)

    def _update_target_movement(self):
        """Update target movement with smooth acceleration-based motion."""
        # Update timer for direction changes
        self.target_vel_change_timer += self.step_dt
        
        # Randomly change desired velocity with some probability
        change_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.target_direction_change_prob
        
        # Generate new random desired velocities for targets that need to change
        new_desired_vel = torch.zeros_like(self.target_desired_vel)
        new_desired_vel[:, :3] = torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0  # [-1, 1]
        new_desired_vel[:, 3:] = torch.rand(self.num_envs, 3, device=self.device) * 0.2 - 0.1  # small angular velocities
        
        # Scale by max speed
        # if self.step_count > 14000:
        new_desired_vel[:, :3] *= self.cfg.max_target_speed * 0.3  # use 30% of max speed for smoother motion
        
        # Update desired velocity where mask is true
        self.target_desired_vel = torch.where(
            change_mask.unsqueeze(-1).expand(-1, 6),
            new_desired_vel,
            self.target_desired_vel
        )
        
        # Reset timer for environments that changed direction
        self.target_vel_change_timer = torch.where(change_mask, torch.zeros_like(self.target_vel_change_timer), self.target_vel_change_timer)
        
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
        self.target_vel += torch.where(self.curriculum > 0, self.target_acceleration * self.step_dt, torch.zeros_like(self.target_acceleration))
        self.target_vel *= torch.where(self.curriculum > 0, self.cfg.target_velocity_damping, 1.0)
        # self.target_vel += self.target_acceleration * self.step_dt
        # self.target_vel *= self.cfg.target_velocity_damping
        
        # Clamp final velocity
        self.target_vel[:, :3] = torch.clamp(
            self.target_vel[:, :3], 
            -self.cfg.max_target_speed, 
            self.cfg.max_target_speed
        )

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
