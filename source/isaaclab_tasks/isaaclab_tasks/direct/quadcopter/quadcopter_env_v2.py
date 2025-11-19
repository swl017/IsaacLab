# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG, IRIS_CFG, IRIS_GIMBAL2_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaaclab_tasks.direct.iris_ma3.controller.point_mass import PointMass
from isaaclab_tasks.direct.iris_ma3.controller.gimbal_stabilizer import GimbalStabilizer
from isaaclab_tasks.direct.iris_ma3.visualization.camera_frustum import CameraFrustum

# Debug draw for camera frustum visualization
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


class QuadcopterArticulation(Articulation):
    """Custom articulation for quadcopters with hybrid control.

    Hybrid Control Architecture:
    - Base body: Controlled via external forces/torques (PhysX API)
    - Gimbal joints: Controlled via standard actuator API (position control)

    The propeller actuators have been removed from the robot configuration
    to allow direct force/torque control of the base body without conflicts.
    Only gimbal actuators remain, which use the standard Articulation API.
    """

    def write_data_to_sim(self):
        """Write both external wrenches and actuator commands to simulation.

        This uses the standard Articulation behavior:
        1. External forces applied to base body for thrust/moment control
        2. Actuator API used for gimbal joint position control
        3. Propeller actuators have been removed from robot config
        """
        # Apply external wrench to base body
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,  # Apply at center of mass
                indices=self._ALL_INDICES,
                is_global=False,  # Body frame
            )

        # Apply actuator model to compute joint commands for gimbal
        self._apply_actuator_model()

        # Set actuation forces (computed by actuator model)
        self.root_physx_view.set_dof_actuation_forces(
            self._joint_effort_target_sim, self._ALL_INDICES
        )

        # Position and velocity targets for implicit actuators (gimbal joints)
        if self._has_implicit_actuators:
            self.root_physx_view.set_dof_position_targets(
                self._joint_pos_target_sim, self._ALL_INDICES
            )
            self.root_physx_view.set_dof_velocity_targets(
                self._joint_vel_target_sim, self._ALL_INDICES
            )


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
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
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 10
    action_space = 6  # [vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch]
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Controller gains - Iteration 5 (balanced PD control with saturation)
    controller_gains = {
        "kp_x": 8.0,      # Velocity tracking proportional gain
        "ki_x": 1.5,      # Velocity tracking integral gain
        "kd_x": 0.8,      # Velocity tracking derivative gain
        "kp_att": 1.2,   # Attitude proportional gain (balanced, not too aggressive)
        "kd_att": 0.8,    # Attitude derivative gain (ratio 0.15 for critical damping)
        "kp_yaw": 8.0,    # Yaw rate tracking gain
        "kd_yaw": 1.5,    # Yaw derivative gain (deprecated for rate tracking)
    }

    max_linear_speed = 10.0  # m/s
    max_yaw_rate = math.radians(360)  # rad/s

    # Gimbal limits (radians)
    max_gimbal_yaw_angle = 3.14159  # ±180°
    max_gimbal_pitch_angle = 1.5708  # ±90°
    max_gimbal_angle_rate = math.radians(360)

    # Camera configuration (for visualization when not headless)
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.7071068, 0, 0, -0.7071068),
            # rot=(-0.2353829, -0.3812674, -0.4696361, -0.7607049),  # ROS camera convention (FLU to camera frame)
            # rot=(0.5, -0.5, 0.5, -0.5),  # ROS camera convention (FLU to camera frame)
            convention="world"
        ),
    )

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions buffer (velocity commands for PointMass controller + gimbal commands)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Camera (created in _setup_scene if not headless)
        self._camera = None

        # Camera frustum visualization (for debug visualization)
        if DEBUG_DRAW and omni_debug_draw is not None:
            self.camera_frustum = CameraFrustum()
        else:
            self.camera_frustum = None

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id, body_names = self._robot.find_bodies("body")
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        # Initialize controller with correct mass and weight
        self._robot_controller = PointMass(
            mass=self._robot_mass,
            weight=self._robot_weight,
            num_envs=self.num_envs,
            disable_gravity=True,  # Disable gravity compensation(Gravity is already disabled in the robot config)
            device=self.device
        )

        # Override controller gains with custom values from config
        self._controller_gains = {
            "kp_x": torch.tensor([self.cfg.controller_gains["kp_x"]], device=self.device).expand(self.num_envs),
            "ki_x": torch.tensor([self.cfg.controller_gains["ki_x"]], device=self.device).expand(self.num_envs),
            "kd_x": torch.tensor([self.cfg.controller_gains["kd_x"]], device=self.device).expand(self.num_envs),
            "kp_y": torch.tensor([self.cfg.controller_gains["kp_x"]], device=self.device).expand(self.num_envs),
            "kd_y": torch.tensor([self.cfg.controller_gains["kd_x"]], device=self.device).expand(self.num_envs),
            "kp_z": torch.tensor([self.cfg.controller_gains["kp_x"]], device=self.device).expand(self.num_envs),
            "kp_att": torch.tensor([self.cfg.controller_gains["kp_att"]], device=self.device).expand(self.num_envs),
            "kd_att": torch.tensor([self.cfg.controller_gains["kd_att"]], device=self.device).expand(self.num_envs),
            "kp_yaw": torch.tensor([self.cfg.controller_gains["kp_yaw"]], device=self.device).expand(self.num_envs),
            "kd_yaw": torch.tensor([self.cfg.controller_gains["kd_yaw"]], device=self.device).expand(self.num_envs),
        }

        # Initialize gimbal control system
        # Find gimbal joint indices
        yaw_joint_ids, _ = self._robot.find_joints("yaw_joint")
        pitch_joint_ids, _ = self._robot.find_joints("pitch_joint")
        roll_joint_ids, _ = self._robot.find_joints("roll_joint")

        self.gimbal_joint_idx = {
            "yaw": yaw_joint_ids[0] if len(yaw_joint_ids) > 0 else None,
            "pitch": pitch_joint_ids[0] if len(pitch_joint_ids) > 0 else None,
            "roll": roll_joint_ids[0] if len(roll_joint_ids) > 0 else None,
        }

        # Check if gimbal joints exist
        self.has_gimbal = all(idx is not None for idx in self.gimbal_joint_idx.values())

        if self.has_gimbal:
            # Initialize gimbal stabilizer
            self._gimbal_stabilizer = GimbalStabilizer(
                device=self.device,
                yaw_limits=[-self.cfg.max_gimbal_yaw_angle, self.cfg.max_gimbal_yaw_angle],
                pitch_limits=[-self.cfg.max_gimbal_pitch_angle, self.cfg.max_gimbal_pitch_angle],
            )
        else:
            print("[WARNING] Gimbal joints not found, gimbal control disabled")

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.cfg.robot.spawn.rigid_props.disable_gravity = True
        # Use custom articulation class that skips actuator API calls
        # This prevents set_dof_actuation_forces() from overriding our external forces
        self._robot = QuadcopterArticulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Create camera if debug visualization is enabled
        if DEBUG_DRAW:
            self._camera = TiledCamera(self.cfg.camera)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Register camera in scene if created
        if self._camera is not None:
            self.scene.sensors["camera"] = self._camera

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions for use in _apply_action().

        Actions are now interpreted as velocity commands by the PointMass controller:
        - For base control: [vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch]
        - Legacy thrust/moment parsing removed (not used with PointMass controller)
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        """Apply actions to the robot.

        Actions: [vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch]
        - vx, vy, vz: Velocity commands for base body (world frame)
        - yaw_rate: Yaw rate command for base body
        - gimbal_yaw: Gimbal yaw position rate command
        - gimbal_pitch: Gimbal pitch position rate command
        """
        # Get current robot state
        curr_quat_w = self._robot.data.root_quat_w.clone()
        curr_lin_vel_w = self._robot.data.root_lin_vel_w.clone()
        curr_ang_vel_b = self._robot.data.root_ang_vel_b.clone()
        curr_lin_acc_w = self._robot.data.body_lin_acc_w[:, self._body_id[0]].clone()

        # Convert linear acceleration from world to body frame
        from isaaclab.utils.math import quat_rotate_inverse
        curr_lin_acc_b = quat_rotate_inverse(curr_quat_w, curr_lin_acc_w)

        # Compute control forces and moments using PointMass controller
        force, moment = self._robot_controller.compute_control_quat_exact(
            cmd_lin_vel_w=self._actions[:, 0:3] * self.cfg.max_linear_speed,
            cmd_yaw_vel_w=self._actions[:, 3] * self.cfg.max_yaw_rate,
            curr_quat_w=curr_quat_w,
            curr_lin_vel_w=curr_lin_vel_w,
            curr_ang_vel_b=curr_ang_vel_b,
            curr_lin_acc_b=curr_lin_acc_b,
            dt=self.cfg.sim.dt,
            gains=self._controller_gains
        )

        # Apply external forces and torques to base body
        force_reshaped = force.unsqueeze(1)  # (num_envs, 1, 3)
        moment_reshaped = moment.unsqueeze(1)  # (num_envs, 1, 3)

        self._robot.set_external_force_and_torque(
            forces=force_reshaped,
            torques=moment_reshaped,
            body_ids=[self._body_id[0]],
            env_ids=None
        )

        # Gimbal control (if gimbal joints exist)
        if self.has_gimbal:
            # Extract gimbal commands from actions
            cmd_gimbal_yaw = self._actions[:, 4]
            cmd_gimbal_pitch = self._actions[:, 5]

            # Get current gimbal joint positions
            gimbal_yaw_current = self._robot.data.joint_pos[:, self.gimbal_joint_idx["yaw"]].clone()
            gimbal_pitch_current = self._robot.data.joint_pos[:, self.gimbal_joint_idx["pitch"]].clone()

            # Compute target positions (rate control)
            gimbal_yaw_target = gimbal_yaw_current + cmd_gimbal_yaw * self.cfg.max_gimbal_angle_rate * self.cfg.sim.dt
            gimbal_pitch_target = gimbal_pitch_current + cmd_gimbal_pitch * self.cfg.max_gimbal_angle_rate * self.cfg.sim.dt

            # Compute stabilizing roll to keep horizon level
            gimbal_roll_stabilizing = self._gimbal_stabilizer.compute_stabilizing_roll(
                gimbal_yaw_current,
                gimbal_pitch_current,
                curr_quat_w
            )

            # self._robot.write_joint_position_to_sim(
            #     position=gimbal_roll_stabilizing.unsqueeze(-1),
            #     joint_ids=[self.gimbal_joint_idx["roll"]],
            # )

            # Apply gimbal position targets
            self._robot.set_joint_position_target(
                target=torch.stack([gimbal_yaw_target, gimbal_roll_stabilizing, gimbal_pitch_target], dim=-1),
                joint_ids=[
                    self.gimbal_joint_idx["yaw"],
                    self.gimbal_joint_idx["roll"],
                    self.gimbal_joint_idx["pitch"],
                ]
            )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
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
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Reset controller integral state to prevent windup carryover
        self._robot_controller.reset_integral(env_ids)
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Add random initial attitude perturbation (±2 degrees roll/pitch)
        # This allows us to quickly see if the controller can stabilize
        import isaaclab.utils.math as math_utils
        num_resets = len(env_ids)
        # Random roll and pitch: uniform in [-2°, +2°] = [-0.0349, +0.0349] rad
        random_roll = torch.zeros(num_resets, device=self.device).uniform_(-0.0349, 0.0349)
        random_pitch = torch.zeros(num_resets, device=self.device).uniform_(-0.0349, 0.0349)
        random_yaw = torch.zeros(num_resets, device=self.device)  # Keep yaw at 0

        # Convert Euler angles to quaternion
        random_quat = math_utils.quat_from_euler_xyz(random_roll, random_pitch, random_yaw)

        # Apply perturbation to default orientation (quaternion multiplication)
        default_quat = default_root_state[:, 3:7]
        perturbed_quat = math_utils.quat_mul(random_quat, default_quat)
        default_root_state[:, 3:7] = perturbed_quat

        # CRITICAL FIX: The IRIS USD file has baked-in velocities on all rigid bodies
        # (propellers spinning at 7-13 rad/s, base at ~0.01 rad/s). These cause immediate
        # gyroscopic instability. We must explicitly zero ALL velocities.
        # For articulations, body velocities are determined by root + joint velocities.

        # Zero root velocity (override USD default)
        zero_root_vel = torch.zeros((len(env_ids), 6), device=self.device)

        # Zero joint velocities (override USD default)
        num_joints = self._robot.num_joints
        zero_joint_vel = torch.zeros((len(env_ids), num_joints), device=self.device)

        # Initialize gimbal joints to neutral position (if gimbal exists)
        if self.has_gimbal:
            # Set gimbal joints to zero position (neutral/forward-looking)
            joint_pos[:, self.gimbal_joint_idx["yaw"]] = 0.0
            joint_pos[:, self.gimbal_joint_idx["pitch"]] = 0.0
            joint_pos[:, self.gimbal_joint_idx["roll"]] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(zero_root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, zero_joint_vel, None, env_ids)

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
        """Debug visualization callback - draws camera frustum."""
        # update the markers
        # self.goal_pos_visualizer.visualize(self._desired_pos_w) # DEPRECATED

        # Camera frustum visualization (if camera exists and debug draw is available)
        if DEBUG_DRAW:
            # Clear previous lines
            self.camera_frustum.draw_interface.clear_lines()

            # Compute camera pose from robot state and gimbal angles
            camera_position_w, camera_orientation_w = self._compute_camera_pose()

            # Get camera intrinsics (constant for this environment)
            # Format: [fx, fy, cx, cy] (camera intrinsic matrix diagonal and principal point)
            width = self.cfg.camera.width
            height = self.cfg.camera.height
            focal_length = self.cfg.camera.spawn.focal_length
            horizontal_aperture = self.cfg.camera.spawn.horizontal_aperture
            vertical_aperture = horizontal_aperture * (height / width)

            # Focal lengths in pixels
            f_x = focal_length / horizontal_aperture * width
            f_y = focal_length / vertical_aperture * height
            c_x = width / 2.0
            c_y = height / 2.0

            # Create batched intrinsics tensor [N, 4] containing [fx, fy, cx, cy]
            camera_intrinsics = torch.stack([
                torch.full((self.num_envs,), f_x, device=self.device),
                torch.full((self.num_envs,), f_y, device=self.device),
                torch.full((self.num_envs,), c_x, device=self.device),
                torch.full((self.num_envs,), c_y, device=self.device),
            ], dim=1)

            # Zoom level (fixed at 1.0 for now, can be made dynamic later)
            zoom_level = torch.ones(self.num_envs, device=self.device)

            # Draw camera frustum
            self.camera_frustum.draw_frustum(
                camera_position=camera_position_w,
                camera_orientation=camera_orientation_w,
                camera_intrinsics=camera_intrinsics,
                camera_cfg=self.cfg.camera,
                zoom_level=zoom_level,
                device=self.device
            )

    def _compute_camera_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute camera position and orientation in world frame.

        Returns:
            camera_position_w: [N, 3] camera position in world frame
            camera_orientation_w: [N, 4] camera orientation quaternion in world frame
        """
        # Get robot base state
        robot_pos_w = self._robot.data.root_pos_w  # [N, 3]
        robot_quat_w = self._robot.data.root_quat_w  # [N, 4] (w, x, y, z)

        # Get gimbal joint angles
        if self.has_gimbal:
            gimbal_yaw = self._robot.data.joint_pos[:, self.gimbal_joint_idx["yaw"]]  # [N]
            gimbal_pitch = self._robot.data.joint_pos[:, self.gimbal_joint_idx["pitch"]]  # [N]
            gimbal_roll = self._robot.data.joint_pos[:, self.gimbal_joint_idx["roll"]]  # [N]
        else:
            gimbal_yaw = torch.zeros(self.num_envs, device=self.device)
            gimbal_pitch = torch.zeros(self.num_envs, device=self.device)
            gimbal_roll = torch.zeros(self.num_envs, device=self.device)

        # Camera offset from gimbal base (in gimbal base frame)
        # This matches the offset in camera config
        from isaaclab.utils.math import quat_mul, quat_rotate, quat_from_euler_xyz

        camera_offset_pos = torch.tensor(self.cfg.camera.offset.pos, device=self.device)  # [3]
        camera_offset_quat = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self.device) # torch.tensor(self.cfg.camera.offset.rot, device=self.device)  # [4]
        # camera_offset_quat =  # torch.tensor(self.cfg.camera.offset.rot, device=self.device)  # [4]

        # Compute gimbal orientation (yaw -> roll -> pitch sequence)
        # Note: Isaac Lab quaternion convention is (w, x, y, z)
        gimbal_quat_b = quat_from_euler_xyz(gimbal_roll, gimbal_pitch, gimbal_yaw)  # [N, 4]

        # Compute camera orientation in robot base frame
        # camera_quat_b = gimbal_quat_b * camera_offset_quat
        camera_quat_b = quat_mul(gimbal_quat_b, camera_offset_quat.unsqueeze(0).expand(self.num_envs, -1))

        # Transform camera orientation to world frame
        camera_quat_w = quat_mul(robot_quat_w, camera_quat_b)

        # Compute camera position in robot base frame
        # First apply gimbal rotation to offset, then add to robot position
        camera_offset_in_gimbal = quat_rotate(gimbal_quat_b, camera_offset_pos.unsqueeze(0).expand(self.num_envs, -1))

        # Transform to world frame
        camera_pos_w = robot_pos_w + quat_rotate(robot_quat_w, camera_offset_in_gimbal)

        return camera_pos_w, camera_quat_w
