# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG, IRIS_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaaclab_tasks.direct.iris_ma3.controller.point_mass import PointMass


class QuadcopterArticulation(Articulation):
    """Custom articulation for quadcopters that skips actuator API calls.

    For quadcopters controlled purely via external forces/torques, we don't want
    actuator-related PhysX API calls to override our control commands.

    This class overrides write_data_to_sim() to ONLY apply external wrenches,
    skipping the set_dof_actuation_forces() call that happens in the base class.
    """

    def write_data_to_sim(self):
        """Write external wrenches to simulation WITHOUT actuator commands.

        Overrides the base Articulation.write_data_to_sim() to skip:
        - self._apply_actuator_model()
        - self.root_physx_view.set_dof_actuation_forces(...)
        - self.root_physx_view.set_dof_position_targets(...)
        - self.root_physx_view.set_dof_velocity_targets(...)

        This ensures external forces/torques are the ONLY control input.
        """
        # DEBUG: Verify custom write_data_to_sim is being called
        import os
        if os.environ.get("DEBUG_EXTERNAL_WRENCH") == "1" and self.has_external_wrench:
            print(f"[CUSTOM QuadcopterArticulation write_data_to_sim] Applying external wrench")
            print(f"  First torque: {self._external_torque_b.view(-1, 3)[0]}")

        # Write external wrench (same as parent class)
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,  # Apply at center of mass
                indices=self._ALL_INDICES,
                is_global=False,  # Body frame
            )

        # INTENTIONALLY SKIP all actuator-related API calls
        # These were overriding our external force/torque commands


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
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
    robot: ArticulationCfg = IRIS_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Controller gains - Iteration 5 (balanced PD control with saturation)
    controller_gains = {
        "kp_x": 8.0,      # Velocity tracking proportional gain
        "ki_x": 1.5,      # Velocity tracking integral gain
        "kd_x": 0.8,      # Velocity tracking derivative gain
        "kp_att": 20.0,   # Attitude proportional gain (balanced, not too aggressive)
        "kd_att": 3.0,    # Attitude derivative gain (ratio 0.15 for critical damping)
        "kp_yaw": 8.0,    # Yaw rate tracking gain
        "kd_yaw": 1.5,    # Yaw derivative gain (deprecated for rate tracking)
    }

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

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
        print(f"\n{'='*70}")
        print(f"[DIAGNOSTIC] Robot body structure:")
        print(f"  Body names found: {body_names}")
        print(f"  Body IDs: {self._body_id}")
        print(f"  Total num_bodies: {self._robot.num_bodies}")
        print(f"  Total num_instances: {self._robot.num_instances}")
        print(f"{'='*70}\n")
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

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.cfg.robot.spawn.rigid_props.disable_gravity = True
        # Use custom articulation class that skips actuator API calls
        # This prevents set_dof_actuation_forces() from overriding our external forces
        self._robot = QuadcopterArticulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        # ACCELERATION SENSOR TEST MODE
        # Apply fixed force and verify measured acceleration matches F/m
        TEST_ACCELERATION_SENSOR = True

        # DEBUG: Log controller inputs and outputs
        curr_quat_w = self._robot.data.root_quat_w.clone()
        curr_lin_vel_w = self._robot.data.root_lin_vel_w.clone()
        curr_ang_vel_b = self._robot.data.root_ang_vel_b.clone()
        curr_lin_acc_w = self._robot.data.body_lin_acc_w[:, self._body_id[0]].clone()

        # Convert linear acceleration from world to body frame
        from isaaclab.utils.math import quat_rotate_inverse
        curr_lin_acc_b = quat_rotate_inverse(curr_quat_w, curr_lin_acc_w)

        if TEST_ACCELERATION_SENSOR:
            # Apply FIXED force to test acceleration sensor accuracy
            # Expected: a = F/m (in body frame)

            # Test 1: Apply 10N forward force (X-axis in body frame)
            test_force = torch.zeros((self.num_envs, 3), device=self.device)
            test_force[:, 0] = 10.0  # 10 N forward

            # Zero moment for pure translational test
            test_moment = torch.zeros((self.num_envs, 3), device=self.device)

            # Expected acceleration: a = F/m - g (if gravity not disabled)
            # Note: curr_lin_acc_b should read this acceleration
            expected_acc_x = 10.0 / self._robot_mass  # ~6.17 m/s² for 1.619 kg

            force = test_force
            moment = test_moment

            # Log every 10 steps to track acceleration behavior
            if self.common_step_counter % 10 == 0:
                print(f"\n=== ACCELERATION TEST - Step {self.common_step_counter} ===")
                print(f"Applied force (body): {force[0]}")
                print(f"Robot mass: {self._robot_mass:.3f} kg")
                print(f"Expected acc (X-axis): {expected_acc_x:.3f} m/s²")
                print(f"Measured acc (body): {curr_lin_acc_b[0]}")
                print(f"Measured acc (world): {curr_lin_acc_w[0]}")
                print(f"Current velocity (world): {curr_lin_vel_w[0]}")
                print(f"Current orientation (quat): {curr_quat_w[0]}")

                # Check if acceleration matches expectation
                acc_error = curr_lin_acc_b[0, 0] - expected_acc_x
                print(f"Acceleration error (X): {acc_error:.3f} m/s² ({acc_error/expected_acc_x*100:.1f}%)")

        else:
            # Normal controller operation
            force, moment = self._robot_controller.compute_control_quat_exact(
                    cmd_lin_vel_w=self._actions[:, 0:3],  # Actions are already in m/s (clamped to [-1,1])
                    cmd_yaw_vel=self._actions[:, 3],      # Yaw rate in rad/s (clamped to [-1,1])
                    curr_quat_w=curr_quat_w,
                    curr_lin_vel_w=curr_lin_vel_w,
                    curr_ang_vel_b=curr_ang_vel_b,
                    curr_lin_acc_b=curr_lin_acc_b,
                    dt=self.cfg.sim.dt,
                    gains=self._controller_gains
                )

        # Log data every 50 steps for env 0
        if self.common_step_counter % 50 == 0:
            from isaaclab.utils.math import euler_xyz_from_quat
            euler_tensor = euler_xyz_from_quat(curr_quat_w)
            euler = euler_tensor[0].cpu().numpy()
            print(f"\n=== Step {self.common_step_counter} ===")
            print(f"Actions (cmd): {self._actions[0]}")
            print(f"Attitude (r,p,y): [{euler[0]*57.3:.1f}, {euler[1]*57.3:.1f}, {euler[2]*57.3:.1f}] deg")
            print(f"Current quat: {curr_quat_w[0]}")
            print(f"Ang Vel (body): {curr_ang_vel_b[0]}")
            print(f"Lin Vel (world): {curr_lin_vel_w[0]}")
            print(f"Force (body): {force[0]}")
            print(f"Moment (body): {moment[0]}")
            print(f"Gains: kp_att={self._controller_gains['kp_att'][0]:.2f}, kd_att={self._controller_gains['kd_att'][0]:.2f}")
            print(f"Robot mass: {self._robot_mass:.3f} kg, weight: {self._robot_weight:.3f} N")

        # EXPERIMENTAL: Test if direct velocity control works (bypass external forces)
        # This will help determine if the issue is specific to external force application
        # or if there's a more fundamental problem with controlling this robot
        # RESULT: Direct velocity control also didn't work - actuators were overriding commands
        # SOLUTION: Removed actuators from IRIS_CFG to allow external force control

        USE_DIRECT_VELOCITY_CONTROL = False  # Disabled - using external forces now

        if USE_DIRECT_VELOCITY_CONTROL:
            # Try to damp angular velocity directly by setting it to zero
            current_lin_vel = self._robot.data.root_lin_vel_w.clone()
            current_ang_vel = self._robot.data.root_ang_vel_b.clone()

            # Apply damping: reduce angular velocity by 10% each step
            damped_ang_vel = current_ang_vel * 0.9

            # Set velocities directly (bypassing forces/torques)
            root_velocities = torch.cat([current_lin_vel, damped_ang_vel], dim=1)
            self._robot.write_root_velocity_to_sim(root_velocities)

            if self.common_step_counter % 50 == 0:
                print(f"\n[DIRECT VELOCITY CONTROL TEST]:")
                print(f"  Current ang_vel: {current_ang_vel[0]} rad/s")
                print(f"  Setting to: {damped_ang_vel[0]} rad/s (90% of current)")
                print(f"  Next step should show ~10% reduction if this works")
        else:
            # DIRECT PHYSX API APPROACH - Bypass Articulation abstraction entirely
            # Apply forces/torques directly via PhysX without going through set_external_force_and_torque()
            # This ensures actuator API calls can't override our commands
            # Note: No moment sign inversion needed - the 180° reference handles the frame flip

            # PhysX API requires force/torque data for ALL bodies (num_envs * num_bodies)
            # We only want to apply to base body (body 0), so create zero arrays and fill base entries
            num_bodies = self._robot.num_bodies
            num_envs = self.num_envs

            # Create arrays for all bodies (num_envs * num_bodies, 3)
            all_forces = torch.zeros((num_envs * num_bodies, 3), device=self.device)
            all_torques = torch.zeros((num_envs * num_bodies, 3), device=self.device)

            # Fill in forces/torques for base body only (indices 0, 5, 10, 15, ...)
            base_body_indices = torch.arange(num_envs, device=self.device) * num_bodies
            all_forces[base_body_indices] = force
            all_torques[base_body_indices] = moment

            # Apply to ALL bodies (PhysX requires this), but only base bodies have non-zero values
            self._robot.root_physx_view.apply_forces_and_torques_at_position(
                force_data=all_forces,  # (num_envs * num_bodies, 3)
                torque_data=all_torques,  # (num_envs * num_bodies, 3)
                position_data=None,  # Apply at center of mass
                indices=self._robot._ALL_INDICES,  # All body indices
                is_global=False,  # Body frame
            )

            # DIAGNOSTIC: Verify forces are being applied
            if self.common_step_counter % 50 == 0:
                print(f"\n[DIAGNOSTIC - Direct PhysX API]:")
                print(f"  Applied force: {force[0]}")
                print(f"  Applied torque: {moment[0]}")
                print(f"  has_external_wrench: {self._robot.has_external_wrench}")
                print(f"  _external_force_b shape: {self._robot._external_force_b.shape}")
                print(f"  _external_force_b[0]: {self._robot._external_force_b[0]}")
                print(f"  _external_torque_b[0]: {self._robot._external_torque_b[0]}")

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

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(zero_root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, zero_joint_vel, None, env_ids)

        # DIAGNOSTIC: Log acceleration values immediately after reset
        print(f"\n=== RESET DIAGNOSTIC - Env IDs: {env_ids[:5]} ===")
        print(f"Reset {len(env_ids)} environments")
        print(f"Root velocity after reset: {self._robot.data.root_lin_vel_w[env_ids[0]] if len(env_ids) > 0 else 'N/A'}")
        print(f"Root ang velocity after reset: {self._robot.data.root_ang_vel_b[env_ids[0]] if len(env_ids) > 0 else 'N/A'}")
        # Note: Acceleration might not update until next physics step
        if len(env_ids) > 0:
            curr_lin_acc_w_reset = self._robot.data.body_lin_acc_w[env_ids[0], self._body_id[0]]
            from isaaclab.utils.math import quat_rotate_inverse
            curr_quat_w_reset = self._robot.data.root_quat_w[env_ids[0]]
            curr_lin_acc_b_reset = quat_rotate_inverse(curr_quat_w_reset.unsqueeze(0), curr_lin_acc_w_reset.unsqueeze(0))
            print(f"Acceleration (world) after reset: {curr_lin_acc_w_reset}")
            print(f"Acceleration (body) after reset: {curr_lin_acc_b_reset[0]}")

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
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
