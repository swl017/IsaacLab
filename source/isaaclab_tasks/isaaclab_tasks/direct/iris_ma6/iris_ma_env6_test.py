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
from isaaclab.utils.math import euler_xyz_from_quat, quat_mul, quat_rotate, quat_rotate_inverse, wrap_to_pi

from .bbox_raycaster_v2 import BBoxRayCasterV2
from .bbox_raycaster_v2.utils.projection import create_intrinsic_matrix_tensor
from .cbf_safety import CBFManager
from .controller import DroneController
from .controller.gimbal_controller import YAW_JOINT_OFFSET
from .delay_system_v3 import MultiAgentDelaySystemV3, AgentStates
from .initial_states import InitialStates
from .delay_system_v3.derived_field_computers import compute_combined_angular_velocity
from .iris_ma_env6_test_cfg import IrisMA6TestEnvCfg
from .target_controller import TargetController
from .triangulation import TriangulationResult, compute_full_triangulation
from .visualization import CustomVisualization
from .visualization.frame_visualizer import FRAME_LINKS

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


def body_to_world_gimbal_angles(
    yaw_body: torch.Tensor,
    pitch_body: torch.Tensor,
    q_body: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert body-frame gimbal yaw/pitch to world-frame azimuth/elevation.

    Mirrors GimbalController._body_to_world_angles() math.

    Args:
        yaw_body: (N,) body-frame gimbal yaw (rad). Must already have YAW_JOINT_OFFSET subtracted.
        pitch_body: (N,) body-frame gimbal pitch (rad).
        q_body: (N, 4) body orientation quaternion (wxyz).

    Returns:
        Tuple of (azimuth_world, elevation_world), each (N,) in radians.
    """
    dir_world = gimbal_ray_direction_world(yaw_body, pitch_body, q_body)
    azimuth = torch.atan2(dir_world[..., 1], dir_world[..., 0])
    xy_dist = torch.sqrt(dir_world[..., 0] ** 2 + dir_world[..., 1] ** 2)
    elevation = torch.atan2(dir_world[..., 2], xy_dist)
    return azimuth, elevation


def gimbal_ray_direction_world(
    yaw_body: torch.Tensor,
    pitch_body: torch.Tensor,
    q_body: torch.Tensor,
) -> torch.Tensor:
    """Compute world-frame camera ray direction from body-frame gimbal angles.

    Args:
        yaw_body: (N,) body-frame gimbal yaw (rad). Must already have YAW_JOINT_OFFSET subtracted.
        pitch_body: (N,) body-frame gimbal pitch (rad).
        q_body: (N, 4) body orientation quaternion (wxyz).

    Returns:
        (N, 3) unit direction vector in world frame.
    """
    cos_p = torch.cos(pitch_body)
    dir_body = torch.stack(
        [
            cos_p * torch.cos(yaw_body),
            cos_p * torch.sin(yaw_body),
            -torch.sin(pitch_body),
        ],
        dim=-1,
    )
    return quat_rotate(q_body, dir_body)


class IrisMA6TestEnv(DirectMARLEnv):
    """Simplified test environment for iris_ma6 DroneController validation.

    This environment provides:
    - 3 agents with velocity + gimbal + zoom control
    - Simple distance-based rewards
    - Minimal observations for testing

    Observation space per agent: 26D ego + 14D per other agent [+ 6D triangulation]
        Ego (26D):
        - pos (3): World position
        - vel (3): World velocity
        - quat (4): Orientation quaternion (wxyz)
        - ang_vel_b (3): Body angular velocity (gyro)
        - lin_acc_b (3): Body linear acceleration (accelerometer)
        - gimbal_azimuth_world (1): World-frame gimbal azimuth
        - gimbal_elevation_world (1): World-frame gimbal elevation
        - gimbal_azimuth_rate (1): World-frame gimbal azimuth rate
        - gimbal_elevation_rate (1): World-frame gimbal elevation rate
        - zoom (1): Current zoom level
        - bbox (4): Primary target normalized bbox (cx, cy, w, h)
        - bbox_empty (1): 1 if bbox is empty, 0 otherwise
        Inter-agent (14D per other agent):
        - position (3), linear_velocity (3), gimbal_azimuth_world (1),
          gimbal_elevation_world (1), gimbal_azimuth_rate (1),
          gimbal_elevation_rate (1), zoom (1), bbox_empty (1), data_age (1), bbox_age (1)

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
        # Auto-disable tiled cameras when rendering is not available.
        # Check both the ENABLE_CAMERAS env var and carb settings (which reflect
        # --enable_cameras CLI flag resolved by AppLauncher).
        # try:
        #     import carb.settings
        #     rendering_enabled = carb.settings.get_settings().get_as_bool(
        #         "/physics/fabricUpdateTransformations"
        #     )
        #     if not rendering_enabled:
        #         cfg.enable_tiled_cameras = False
        #         print("[IrisMA6TestEnv] Tiled cameras disabled (rendering not enabled)")
        #     else:
        #         cfg.enable_tiled_cameras = True
        #         print("[IrisMA6TestEnv] Tiled cameras enabled")
        # except ImportError:
        #     print("[IrisMA6TestEnv] Tiled cameras disabled (config)")

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
            if cfg.enable_tiled_cameras:
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
        self._last_actions: Dict[str, torch.Tensor] = {
            agent_id: torch.zeros(self.num_envs, len(self.cfg.action_weight), device=self.device)
            for agent_id in cfg.possible_agents
        }
        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device)
        num_agents = len(cfg.possible_agents)
        self.cmd_vel = torch.zeros(self.num_envs, num_agents, 7, device=self.device)

        # State caches (populated in _apply_action, consumed by rewards/dones/obs)
        self._root_pos_w: Dict[str, torch.Tensor] = {}
        self._root_quat_w: Dict[str, torch.Tensor] = {}
        self._root_lin_vel_w: Dict[str, torch.Tensor] = {}
        self._root_ang_vel_b: Dict[str, torch.Tensor] = {}
        self._root_lin_acc_w: Dict[str, torch.Tensor] = {}
        self._root_lin_acc_b: Dict[str, torch.Tensor] = {}
        self._gimbal_joint_pos: Dict[str, torch.Tensor] = {}
        self._target_pos_w: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w: torch.Tensor = torch.zeros(self.num_envs, 4, device=self.device)

        # Zoom level tracking (controller manages internal state, we track for observations)
        self.zoom_level = torch.ones(self.num_envs, num_agents, device=self.device)

        # Per-env max linear velocity (randomized via curriculum)
        self._max_lin_vel = torch.full(
            (self.num_envs,), self.cfg.max_lin_vel, device=self.device
        )

        # Initial gimbal angles
        self.cmd_gimbal_yaw = torch.zeros(self.num_envs, num_agents, device=self.device)
        self.cmd_gimbal_pitch = torch.zeros(self.num_envs, num_agents, device=self.device)

        self._camera_offset_position_b = torch.tensor(
            self.cfg.camera.offset.pos, dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        # Camera offset rotation for frustum visualization.
        # R_z(-90°) * R_x(-90°) maps frustum +Z → body +X (physics forward).
        # Same quaternion as TiledCamera ROS offset. YAW_JOINT_OFFSET is
        # stripped from joint positions before this offset is applied, so
        # pitch rotations correctly tilt the frustum. See §5.4.
        self._camera_offset_rotation_b = torch.tensor(
            [0.5, -0.5, 0.5, -0.5], dtype=torch.float32, device=self.device
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

        # Initialize CBF safety manager for collision avoidance
        self.cbf_manager = CBFManager(
            cfg=self.cfg.cbf_safety,
            num_envs=self.num_envs,
            num_agents=len(cfg.possible_agents),
            device=self.device,
        )

        # Initialize delay system for realistic observation delays
        if self.cfg.enable_delay_system:
            self._delay_system = MultiAgentDelaySystemV3(
                cfg=self.cfg.delay_system,
                possible_agents=self.cfg.possible_agents,
                num_envs=self.num_envs,
                num_joints=3,  # pitch, yaw, roll
                num_targets=1,
                device=self.device,
            )
            # Start with no delay — curriculum will ramp it up
            self._delay_system.set_delay_mode("none", progress=0.0)
        else:
            self._delay_system = None

        # Simulation time tracking for delay system
        self._sim_time = torch.zeros(self.num_envs, device=self.device)

        # Visualization is created lazily via _set_debug_vis_impl when debug_vis is enabled
        self._visualization: CustomVisualization | None = None

        # Cache for visualization data (updated each step, used by debug callback)
        self._vis_camera_poses: Dict[str, tuple] = {}
        self._vis_target_pos: torch.Tensor | None = None
        self._vis_bbox_empty: Dict[str, torch.Tensor] = {}
        self._vis_zoom_levels: Dict[str, torch.Tensor] = {}

        # Temporal smoothing for occlusion detection to prevent oscillation
        # Uses exponential moving average (EMA) on bbox_confidence
        self._occlusion_ema_alpha = 1.0  # Lower = more smoothing (0.3 = 70% history, 30% new)
        self._smoothed_bbox_confidence: torch.Tensor = torch.ones(
            self.num_envs, num_agents, 1, device=self.device
        )  # (N, C, T) - starts at 1.0 (visible)
        self._smoothed_bbox_empty_threshold = 0.4  # Below this confidence = occluded

        # Image dimensions for bbox normalization
        self._img_dims = torch.tensor(
            [self.cfg.camera.width, self.cfg.camera.height],
            device=self.device,
            dtype=torch.float32,
        )

        # Per-step reward components (for visualization/debugging)
        self._step_rewards: Dict[str, Dict[str, torch.Tensor]] = {}

        # Episode reward tracking for logging
        self._episode_sums = {
            agent: {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in [
                    "action_sum",
                    "action_delta",
                    "bbox_center",
                    "bbox_size",
                    "triangulation",
                    "cbf_penalty",
                    "collision",
                    "altitude",
                    "est_error_gt",
                    "est_error_e2e",
                ]
            }
            for agent in self.cfg.possible_agents
        }

        # Detection dropout tracking for debugging
        self._detection_stats = {
            "total_steps": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "invalid_all_agents": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "pair_valid_count": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        # Collision event counter (per-episode)
        self._collision_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Curriculum progress factors (updated each step in _get_rewards)
        self.progress_coord = 0.0
        self.progress_safety = 0.0

        # Triangulation module initialization
        # Dual pipeline: GT for rewards, triangulated for observations
        self._triangulation_result_gt: TriangulationResult | None = None
        self._triangulation_result_obs: TriangulationResult | None = None

        # Cache for camera intrinsics per agent (updated each step with zoom)
        self._camera_intrinsics_cache: Dict[str, torch.Tensor] = {}

        # Initialize initial states generator for curriculum-driven reset randomization
        if self.cfg.enable_initial_states_randomization:
            self._initial_states = InitialStates(
                cfg=self.cfg.initial_states,
                num_envs=self.num_envs,
                num_agents=len(cfg.possible_agents),
                device=self.device,
            )
        else:
            self._initial_states = None

        # Curriculum progress tracking (updated externally, used by initial_states)
        self.progress_tracking = 0.0
        self.progress_moving_target = 0.0
        self.progress_agent_velocity = 0.0
        self.progress_delay = 0.0
        self.progress_dynamics = 0.0

        # Initialize target controller for physics-based target movement
        if self.cfg.enable_target_controller:
            # Get target mass from physics
            target_masses = self.target.root_physx_view.get_masses()
            target_mass = target_masses[0].sum().item()

            # Set control_dt to match simulation dt
            target_cfg = self.cfg.target_controller
            target_cfg.control_dt = self.cfg.sim.dt

            self._target_controller = TargetController(
                cfg=target_cfg,
                mass=target_mass,
                gravity=9.81,
                num_envs=self.num_envs,
                num_targets=1,
                device=self.device,
            )
        else:
            self._target_controller = None

        # Facility position for approach mode (at each environment's origin)
        # Clone env_origins so targets approach their local environment center
        self._facility_position = self._terrain.env_origins.clone()

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
        if self.cfg.enable_tiled_cameras:
            for agent_id, camera_cfg in self.agent_camera_cfgs.items():
                self._cameras[agent_id] = TiledCamera(camera_cfg)

        # Create terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create shared target (RigidObject with gravity enabled)
        # RigidObject is used because iris_body.usda has no joints (propellers removed)
        self.target = RigidObject(self.cfg.target_cfg)
        # Register target in scene so it gets proper lifecycle management
        # (write_data_to_sim / update are handled by scene automatically)
        self.scene.rigid_objects["target"] = self.target

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Register TiledCamera sensors in scene
        for agent_id, camera in self._cameras.items():
            robot_index = agent_id.split("_")[-1]
            self.scene.sensors[f"camera_{robot_index}"] = camera

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Optionally load aesthetic Flight scene (Y-up USD, rotated 90° around X to Z-up)
        if self.cfg.use_flight_scene:
            import omni.usd
            from pxr import UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            xform = UsdGeom.Xform.Define(stage, "/World/FlightScene")
            # Order matters: USD applies ops top-to-bottom → translate, rotate, scale
            ox, oy, oz = self.cfg.flight_scene_offset
            xform.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
            xform.AddRotateXOp().Set(90.0)  # USD in Y-up convention → Rotate the USD to match Z-up
            s = self.cfg.flight_scene_scale
            xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))
            xform.GetPrim().GetReferences().AddReference(self.cfg.flight_scene_usd)

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
            self.cmd_vel[:, idx, 0:3] = action[:, 0:3] * self._max_lin_vel.unsqueeze(-1)
            self.cmd_vel[:, idx, 3] = action[:, 3] * self.cfg.max_yaw_rate
            # Gimbal and zoom controllers expect normalized [-1, 1] input and scale internally
            self.cmd_vel[:, idx, 4] = action[:, 4]  # Gimbal yaw rate (normalized)
            self.cmd_vel[:, idx, 5] = action[:, 5]  # Gimbal pitch rate (normalized)
            self.cmd_vel[:, idx, 6] = action[:, 6]  # Zoom rate (normalized)

        self.decimated_step = 0

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

            # Read current gimbal joint positions [pitch, yaw, roll]
            gimbal_jp = torch.stack(
                [
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]],
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]],
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["roll"]],
                ],
                dim=-1,
            )

            # Step the DroneController
            # _apply_action runs every sim step (sim.dt = 0.01s), not every policy
            # step (sim.dt * decimation = 0.04s). The drone cascade needs the full
            # decimated dt for its inner-loop subdivision (velocity→attitude→rate).
            # The gimbal receives sim.dt separately for correct rate integration.
            F_body, tau_body, gimbal_pos_targets, gimbal_vel_targets, zoom_level = controller.step_policy(
                v_cmd=v_cmd,
                yaw_rate_cmd=yaw_rate_cmd,
                gimbal_yaw_rate_cmd=gimbal_yaw_rate,
                gimbal_pitch_rate_cmd=gimbal_pitch_rate,
                zoom_rate_cmd=zoom_rate,
                q_body=robot.data.root_quat_w,
                v_body=robot.data.root_lin_vel_w,
                omega_body=robot.data.root_ang_vel_b,
                sim_dt=self.cfg.sim.dt * self.cfg.decimation,
                gimbal_joint_positions=gimbal_jp,
                physics_dt=self.cfg.sim.dt,
            )

            # Apply forces in body frame (Isaac Lab expects local frame)
            robot.set_external_force_and_torque(
                forces=F_body.unsqueeze(1),
                torques=tau_body.unsqueeze(1),
                body_ids=self._body_ids[agent_id],
            )

            use_implicit_gimbal_control = True  # Set to False to bypass implicit control and write joint state directly
            if use_implicit_gimbal_control:
                # Apply gimbal position targets
                # Add YAW_JOINT_OFFSET (-π/2) to yaw: controller yaw=0 means body +X
                # (physics forward). The offset shifts the physical joint so the
                # combined chain (joint + camera offset) points along body +X.
                gimbal_yaw, gimbal_roll, gimbal_pitch = gimbal_pos_targets
                gimbal_yaw_joint = gimbal_yaw + YAW_JOINT_OFFSET
                robot.set_joint_position_target(
                    target=torch.stack([gimbal_pitch, gimbal_yaw_joint, gimbal_roll], dim=-1),
                    joint_ids=[
                        self.gimbal_joint_idx[agent_id]["pitch"],
                        self.gimbal_joint_idx[agent_id]["yaw"],
                        self.gimbal_joint_idx[agent_id]["roll"],
                    ],
                )

                # Apply gimbal velocity feedforward targets
                # Makes the actuator's damping term assistive during transients:
                # τ_d = d*(v_target - v) pushes toward the expected velocity
                # instead of opposing motion (when v_target=0).
                gimbal_yaw_vel, gimbal_roll_vel, gimbal_pitch_vel = gimbal_vel_targets
                robot.set_joint_velocity_target(
                    target=torch.stack([gimbal_pitch_vel, gimbal_yaw_vel, gimbal_roll_vel], dim=-1),
                    joint_ids=[
                        self.gimbal_joint_idx[agent_id]["pitch"],
                        self.gimbal_joint_idx[agent_id]["yaw"],
                        self.gimbal_joint_idx[agent_id]["roll"],
                    ],
                )
            else:
                # Apply gimbal joint state directly (bypass implicit actuator lag).
                # write_joint_state_to_sim teleports joints to exact positions at
                # the start of the physics step. We must ALSO set the actuator
                # targets to match, otherwise the implicit PD controller (still
                # active in PhysX) applies force during the step based on stale
                # targets, pulling joints away from the desired position.
                # With targets == state, PD force = k*(pos-pos) + d*(vel-vel) = 0.
                gimbal_yaw, gimbal_roll, gimbal_pitch = gimbal_pos_targets
                gimbal_yaw_joint = gimbal_yaw + YAW_JOINT_OFFSET
                gimbal_yaw_vel, gimbal_roll_vel, gimbal_pitch_vel = gimbal_vel_targets
                gimbal_joint_ids = [
                    self.gimbal_joint_idx[agent_id]["pitch"],
                    self.gimbal_joint_idx[agent_id]["yaw"],
                    self.gimbal_joint_idx[agent_id]["roll"],
                ]
                pos_cmd = torch.stack([gimbal_pitch, gimbal_yaw_joint, gimbal_roll], dim=-1)
                vel_cmd = torch.stack([gimbal_pitch_vel, gimbal_yaw_vel, gimbal_roll_vel], dim=-1)
                robot.write_joint_state_to_sim(
                    position=pos_cmd, velocity=vel_cmd, joint_ids=gimbal_joint_ids,
                )
                # Match actuator targets so PD applies zero force during step
                robot.set_joint_position_target(target=pos_cmd, joint_ids=gimbal_joint_ids)
                robot.set_joint_velocity_target(target=vel_cmd, joint_ids=gimbal_joint_ids)

            # Store zoom level for observations
            self.zoom_level[:, idx] = zoom_level

        # Apply target controller if enabled
        if self._target_controller is not None:
            # Target data is kept current by scene.update() (target registered in scene)

            # Get current target state (add target dimension)
            target_pos = self.target.data.root_pos_w.unsqueeze(1)  # [N, 1, 3]
            target_vel = self.target.data.root_lin_vel_w.unsqueeze(1)  # [N, 1, 3]
            target_quat = self.target.data.root_quat_w.unsqueeze(1)  # [N, 1, 4]
            target_omega = self.target.data.root_ang_vel_b.unsqueeze(1)  # [N, 1, 3]

            # Get agent positions for evasion logic
            agent_positions = torch.stack([
                self._robots[agent_id].data.root_pos_w
                for agent_id in self.cfg.possible_agents
            ], dim=1)  # [N, num_agents, 3]

            # Interceptor roles (1 = INTERCEPT for all agents)
            agent_roles = torch.ones(
                self.num_envs, len(self.cfg.possible_agents),
                dtype=torch.long, device=self.device
            )

            # Step the target controller
            dt = self.cfg.sim.dt
            F_body_target, tau_body_target = self._target_controller.step(
                current_position=target_pos,
                current_velocity=target_vel,
                current_quat=target_quat,
                current_angular_vel=target_omega,
                facility_position=self._facility_position,
                interceptor_positions=agent_positions,
                interceptor_roles=agent_roles,
                curriculum_progress=self.progress_moving_target,
                dt=dt,
                env_origins=self._terrain.env_origins,
            )

            # Apply forces to target (squeeze target dimension, add body dimension)
            self.target.set_external_force_and_torque(
                forces=F_body_target.squeeze(1).unsqueeze(1),  # [N, 1, 3]
                torques=tau_body_target.squeeze(1).unsqueeze(1),  # [N, 1, 3]
                body_ids=[0],  # Root body
            )

        # Acquire and process states on the final substep (used by rewards/dones/obs after decimation)
        if self.decimated_step == self.cfg.decimation - 1:
            self._update_state_cache()

        self.decimated_step += 1

    def _update_state_cache(self):
        """Acquire robot/target states and update derived quantities (camera poses, bbox, vis cache).

        Called on the last decimation substep and after resets.
        """
        self._target_pos_w = self.target.data.root_pos_w
        self._target_quat_w = self.target.data.root_quat_w

        # Update delay system time
        self._sim_time += self.cfg.sim.dt * self.cfg.decimation
        if self._delay_system is not None:
            self._delay_system.set_time(self._sim_time)

        camera_poses = {}
        camera_intrinsics = {}
        image_shapes = {}
        agent_poses = {}

        target_pos = self._target_pos_w
        target_quat = self._target_quat_w
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(1)
        if target_quat.ndim == 2:
            target_quat = target_quat.unsqueeze(1)

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            self._root_pos_w[agent_id] = robot.data.root_pos_w
            self._root_quat_w[agent_id] = robot.data.root_quat_w
            self._root_lin_vel_w[agent_id] = robot.data.root_lin_vel_w
            self._root_ang_vel_b[agent_id] = robot.data.root_ang_vel_b
            # Linear acceleration (world and body frame — mimics IMU accelerometer)
            self._root_lin_acc_w[agent_id] = robot.data.body_lin_acc_w[:, self._body_ids[agent_id][0]]
            self._root_lin_acc_b[agent_id] = quat_rotate_inverse(
                self._root_quat_w[agent_id], self._root_lin_acc_w[agent_id]
            )
            self._gimbal_joint_pos[agent_id] = torch.stack(
                [
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]],
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]],
                    robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["roll"]],
                ],
                dim=-1,
            )

            root_pos = self._root_pos_w[agent_id]
            root_quat = self._root_quat_w[agent_id]

            # Use pitch_link body pose directly from physics simulation.
            # This ensures exact match with TiledCamera which is attached to pitch_link.
            # Previously used compute_camera_orientation_from_gimbal() which analytically
            # computed the orientation from joint angles, but this could drift from the
            # actual physics-computed link transforms under motion.
            robot = self._robots[agent_id]
            pitch_link_idx = self._frame_link_ids[agent_id]["pitch_link"]
            camera_pos_w = robot.data.body_pos_w[:, pitch_link_idx]  # (N, 3)
            camera_quat_physics = robot.data.body_quat_w[:, pitch_link_idx]  # (N, 4)

            # Apply camera offset rotation to convert from pitch_link frame to camera frame.
            # The offset (0.5, -0.5, 0.5, -0.5) is R_z(-90°) * R_x(-90°) which maps:
            # - Camera +Z (optical axis) -> Body +X (forward)
            # - Camera +Y (down) -> Body -Z (down)
            # - Camera +X (right) -> Body +Y (left, then negated = right)
            camera_quat_w = quat_mul(camera_quat_physics, self._camera_offset_rotation_b)
            camera_poses[agent_id] = (camera_pos_w, camera_quat_w)

            intrinsic = self._camera_intrinsics_base.clone()
            intrinsic[:, 0, 0] *= self.zoom_level[:, idx]
            intrinsic[:, 1, 1] *= self.zoom_level[:, idx]
            camera_intrinsics[agent_id] = intrinsic
            image_shapes[agent_id] = self._camera_image_shape
            agent_poses[agent_id] = (root_pos, root_quat)

            # Update TiledCamera intrinsics based on zoom level
            if agent_id in self._cameras:
                self._cameras[agent_id].set_intrinsic_matrices_batched(
                    intrinsic,
                    self.cfg.camera.spawn.focal_length * self.zoom_level[:, idx],
                )

        self.bbox_raycaster_v2.update(
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            target_poses=(target_pos, target_quat),
            agent_poses=agent_poses,
            image_shapes=image_shapes,
        )

        # Apply temporal smoothing to bbox_confidence to prevent oscillation
        # EMA: smoothed = alpha * new + (1 - alpha) * old
        raw_confidence = self.bbox_raycaster_v2.data.bbox_confidence  # (N, C, T)
        self._smoothed_bbox_confidence = (
            self._occlusion_ema_alpha * raw_confidence
            + (1.0 - self._occlusion_ema_alpha) * self._smoothed_bbox_confidence
        )

        # Update delay system ground truth for each agent
        if self._delay_system is not None:
            for idx, agent_id in enumerate(self.cfg.possible_agents):
                # Create AgentStates and populate with current ground truth
                gt_states = AgentStates(
                    num_envs=self.num_envs,
                    num_joints=3,  # pitch, yaw, roll
                    num_targets=1,
                    device=self.device,
                )
                gt_data = gt_states.data

                # Body motion
                gt_data.body_position_w = self._root_pos_w[agent_id]
                gt_data.body_orientation_w = self._root_quat_w[agent_id]
                gt_data.body_linear_velocity_w = self._root_lin_vel_w[agent_id]
                gt_data.body_angular_velocity_w = self._root_ang_vel_b[agent_id]
                gt_data.body_angular_velocity_b = self._root_ang_vel_b[agent_id]
                gt_data.body_linear_acceleration_w = self._root_lin_acc_w[agent_id]
                gt_data.body_linear_acceleration_b = self._root_lin_acc_b[agent_id]

                # Joint states (gimbal) - order is pitch, yaw, roll
                gt_data.joint_positions_b = self._gimbal_joint_pos[agent_id]
                # Joint velocities from commanded gimbal rates [pitch_rate, yaw_rate, 0]
                joint_vel = torch.zeros_like(self._gimbal_joint_pos[agent_id])
                joint_vel[:, 0] = self.cmd_vel[:, idx, 5] * self.cfg.max_gimbal_rate  # pitch rate
                joint_vel[:, 1] = self.cmd_vel[:, idx, 4] * self.cfg.max_gimbal_rate  # yaw rate
                gt_data.joint_velocities_b = joint_vel

                # Combined angular velocity (body + gimbal)
                combined_w, combined_b = compute_combined_angular_velocity(
                    body_angular_velocity_w=self._root_ang_vel_b[agent_id],  # Note: currently stored as body-frame
                    body_angular_velocity_b=self._root_ang_vel_b[agent_id],
                    joint_velocities_b=joint_vel,
                    body_orientation_w=self._root_quat_w[agent_id],
                    joint_positions_b=self._gimbal_joint_pos[agent_id],
                )
                gt_data.body_combined_angular_velocity_w = combined_w
                gt_data.body_combined_angular_velocity_b = combined_b

                # Gimbal world-frame angles (from joint positions + body orientation)
                az, el = body_to_world_gimbal_angles(
                    yaw_body=gt_data.joint_positions_b[:, 1] - YAW_JOINT_OFFSET,
                    pitch_body=gt_data.joint_positions_b[:, 0],
                    q_body=gt_data.body_orientation_w,
                )
                gt_data.gimbal_azimuth_world = az
                gt_data.gimbal_elevation_world = el

                # Gimbal world-frame rates (from action commands)
                max_rate = self.cfg.max_gimbal_rate
                gt_data.gimbal_azimuth_rate = self.cmd_vel[:, idx, 4] * max_rate
                gt_data.gimbal_elevation_rate = self.cmd_vel[:, idx, 5] * max_rate

                # Camera geometry
                cam_pos_w, cam_ori_w = camera_poses[agent_id]
                gt_data.camera_position_w = cam_pos_w
                gt_data.camera_orientation_w = cam_ori_w
                gt_data.camera_zoom_level = self.zoom_level[:, idx]
                gt_data.camera_base_intrinsics = self._camera_intrinsics_base.clone()

                # Detection (bbox) - shape (N, T, 4)
                # CRITICAL: Use pixel bboxes, not normalized - triangulation expects pixels
                gt_data.bboxes_2d = self.bbox_raycaster_v2.data.bboxes[
                    :, idx, :, :
                ]

                # Timestamps
                gt_data.timestamp_sim_walltime = self._sim_time
                gt_data.timestamp_motion = self._sim_time.clone()
                gt_data.timestamp_detection = self._sim_time.clone()

                # Update delay system
                self._delay_system.update_ground_truth(agent_id, gt_states)

        # Cache data for debug visualization callback
        self._vis_camera_poses = camera_poses
        self._vis_target_pos = self._target_pos_w
        # Use smoothed confidence for visualization to reduce flickering
        self._vis_bbox_empty = {
            agent_id: self._smoothed_bbox_confidence[:, idx, 0] < self._smoothed_bbox_empty_threshold
            for idx, agent_id in enumerate(self.cfg.possible_agents)
        }
        self._vis_zoom_levels = {
            agent_id: self.zoom_level[:, idx]
            for idx, agent_id in enumerate(self.cfg.possible_agents)
        }

        # Update camera intrinsics cache for triangulation
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            self._camera_intrinsics_cache[agent_id] = camera_intrinsics[agent_id]

    def _compute_zoomed_intrinsics(
        self,
        base_intrinsics: torch.Tensor,
        zoom_level: torch.Tensor,
    ) -> torch.Tensor:
        """Compute camera intrinsics with zoom applied.

        Args:
            base_intrinsics: Base camera intrinsics [N, 3, 3]
            zoom_level: Zoom level multiplier [N]

        Returns:
            Zoomed intrinsics [N, 3, 3] with fx, fy scaled by zoom
        """
        intrinsics = base_intrinsics.clone()
        intrinsics[:, 0, 0] *= zoom_level  # fx
        intrinsics[:, 1, 1] *= zoom_level  # fy
        return intrinsics

    def _build_gt_states(self) -> Dict[AgentID, AgentStates]:
        """Build GT states dictionary when delay system is disabled.

        Returns:
            Dictionary mapping agent_id to AgentStates with current GT values.
        """
        gt_states = {}
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            states = AgentStates(
                num_envs=self.num_envs,
                num_joints=3,
                num_targets=1,
                device=self.device,
            )
            data = states.data

            # Body motion
            data.body_position_w = self._root_pos_w[agent_id]
            data.body_orientation_w = self._root_quat_w[agent_id]

            # Joint states
            data.joint_positions_b = self._gimbal_joint_pos[agent_id]

            # Gimbal world-frame angles
            az, el = body_to_world_gimbal_angles(
                yaw_body=data.joint_positions_b[:, 1] - YAW_JOINT_OFFSET,
                pitch_body=data.joint_positions_b[:, 0],
                q_body=data.body_orientation_w,
            )
            data.gimbal_azimuth_world = az
            data.gimbal_elevation_world = el

            # Gimbal world-frame rates (from action commands)
            max_rate = self.cfg.max_gimbal_rate
            data.gimbal_azimuth_rate = self.cmd_vel[:, idx, 4] * max_rate
            data.gimbal_elevation_rate = self.cmd_vel[:, idx, 5] * max_rate

            # Body velocity (needed for inter-agent obs in GT path)
            data.body_linear_velocity_w = self._root_lin_vel_w[agent_id]

            # Camera geometry
            robot = self._robots[agent_id]
            pitch_link_idx = self._frame_link_ids[agent_id]["pitch_link"]
            data.camera_position_w = robot.data.body_pos_w[:, pitch_link_idx]
            data.camera_base_intrinsics = self._camera_intrinsics_base.clone()
            data.camera_zoom_level = self.zoom_level[:, idx]

            # Detection - use pixel bboxes (not normalized) for triangulation
            data.bboxes_2d = self.bbox_raycaster_v2.data.bboxes[:, idx, :, :]

            gt_states[agent_id] = states
        return gt_states

    def _compute_triangulation(
        self,
        states: Dict[AgentID, AgentStates],
        use_gt_target: bool = True,
    ) -> TriangulationResult:
        """Compute triangulation and covariance for all targets.

        Dual-pipeline architecture:
        - use_gt_target=True: For rewards - covariance computed at GT position
        - use_gt_target=False: For observations - uses midpoint triangulation

        Args:
            states: Agent states dictionary (reward_states or delayed_states).
                Must be provided - no fallback to GT.
            use_gt_target: If True, use GT target position for covariance computation.
                          If False, use triangulated position (midpoint method).

        Returns:
            TriangulationResult with position, covariance, quality_metric, validity
        """
        # Extract camera positions from states
        camera_positions = torch.stack([
            states[agent_id].data.camera_position_w
            for agent_id in self.cfg.possible_agents
        ], dim=1)  # [N, C, 3]

        # Extract robot orientations from states
        robot_quats = torch.stack([
            states[agent_id].data.body_orientation_w
            for agent_id in self.cfg.possible_agents
        ], dim=1)  # [N, C, 4]

        # Gimbal angles from joint positions (order: pitch, yaw, roll)
        # CRITICAL: Subtract YAW_JOINT_OFFSET from raw joint yaw to get logical gimbal yaw.
        # The physical joint has offset -π/2 applied, so joint_yaw=-π/2 means camera forward.
        # Triangulation expects gimbal_yaw=0 to mean camera forward.
        gimbal_pitches = torch.stack([
            states[agent_id].data.joint_positions_b[:, 0]
            for agent_id in self.cfg.possible_agents
        ], dim=1)  # [N, C]

        gimbal_yaws = torch.stack([
            states[agent_id].data.joint_positions_b[:, 1] - YAW_JOINT_OFFSET
            for agent_id in self.cfg.possible_agents
        ], dim=1)  # [N, C]

        gimbal_rolls = torch.stack([
            states[agent_id].data.joint_positions_b[:, 2]
            for agent_id in self.cfg.possible_agents
        ], dim=1)  # [N, C]

        # Compute zoomed camera intrinsics from states
        camera_intrinsics = torch.stack([
            self._compute_zoomed_intrinsics(
                states[agent_id].data.camera_base_intrinsics,
                states[agent_id].data.camera_zoom_level
            )
            for agent_id in self.cfg.possible_agents
        ], dim=1)  # [N, C, 3, 3]

        # Get bbox from states [N, C, T, 4] (pixel xywh format)
        bbox_2d = torch.stack([
            states[agent_id].data.bboxes_2d
            for agent_id in self.cfg.possible_agents
        ], dim=1)

        # Bbox validity from states (non-zero bbox)
        bbox_valid = bbox_2d.abs().sum(dim=-1) > 1e-6  # [N, C, T]

        # Target position (GT): [N, T, 3]
        target_pos_gt = self._target_pos_w.unsqueeze(1) if use_gt_target else None

        # Call triangulation
        result = compute_full_triangulation(
            bbox_2d=bbox_2d,
            bbox_valid=bbox_valid,
            robot_positions=camera_positions,
            robot_quats=robot_quats,
            gimbal_yaws=gimbal_yaws,
            gimbal_rolls=gimbal_rolls,
            gimbal_pitches=gimbal_pitches,
            camera_intrinsics=camera_intrinsics,
            cfg=self.cfg.triangulation,
            target_positions_gt=target_pos_gt,
        )

        return result

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """Get rewards for all agents.

        Reward composition (6 components, migrated from iris_ma5):
        - action_sum: Penalty for action magnitude (weighted per dimension)
        - action_delta: Penalty for action changes (smoothness)
        - bbox_center: Reward for centering target in image
        - bbox_size: Reward for appropriate bbox size (~20% of image area)
        - triangulation: Covariance-based quality (multiple modes)
        - cbf_penalty: CPA barrier violation (replaces iris_ma5 collision+ttc)

        CBF penalty always uses GT positions for safety.
        Triangulation uses GT target position for covariance computation.

        Returns:
            Dictionary mapping agent_id to reward tensor (N,).
        """
        rewards_dict = {}

        # Update curriculum progress
        # current_step = self.common_step_counter if not DEBUG_DRAW else self.cfg.debug_initial_step
        current_step = self.common_step_counter
        curr = self.cfg.curriculum
        self.progress_coord = self._linear_progress(
            curr.coordination_start_step, curr.coordination_end_step, current_step
        )
        self.progress_safety = self._linear_progress(
            curr.safety_start_step, curr.safety_end_step, current_step
        )

        # Update curriculum progress for initial states randomization
        self.progress_tracking = curr.get_progress(
            current_step, curr.tracking_start_step, curr.tracking_end_step
        )

        self.progress_moving_target = curr.get_progress(
            current_step, curr.moving_target_start_step, curr.moving_target_end_step
        )
        self.progress_agent_velocity = curr.get_agent_velocity_progress(current_step)

        # Update delay system curriculum (none → fixed → random, noise/dropout ramp)
        if self._delay_system is not None:
            delay_mode = curr.get_delay_mode(current_step)
            if delay_mode == "none":
                self.progress_delay = 0.0
                self._delay_system.set_delay_mode("none", progress=0.0)
            elif delay_mode == "fixed":
                self.progress_delay = curr.get_fixed_delay_progress(current_step)
                self._delay_system.set_delay_mode("fixed", progress=self.progress_delay)
            else:  # "random"
                self.progress_delay = curr.get_random_delay_progress(current_step)
                self._delay_system.set_delay_mode("random", progress=self.progress_delay)

            # Ramp noise (phase 2: 80k-100k)
            self._delay_system.set_noise_scale(curr.get_noise_progress(current_step))

            # Ramp dropout (phase 5: 160k-200k)
            dropout_progress = curr.get_dropout_progress(current_step)
            self._delay_system.set_dropout_rate(
                dropout_progress * self.cfg.delay_system_params.dropout_prob
            )

        # Get states for reward computation
        if self._delay_system is not None:
            if self.cfg.use_noisy_rewards:
                reward_states = self._delay_system.get_all_states_for_observations(
                    ego_agent_id=self.cfg.possible_agents[0]
                )
            else:
                reward_states = self._delay_system.get_all_states_for_rewards(
                    ego_agent_id=self.cfg.possible_agents[0]
                )
        else:
            reward_states = self._build_gt_states()

        # Track detection dropout statistics
        # bbox_valid per agent for target 0: count how many agents see the target
        num_valid_detections = torch.zeros(self.num_envs, device=self.device)
        for agent_id in self.cfg.possible_agents:
            bbox_raw = reward_states[agent_id].data.bboxes_2d[:, 0, :]  # [N, 4]
            agent_valid = bbox_raw.abs().sum(dim=-1) > 1e-6  # [N]
            num_valid_detections += agent_valid.float()
        self._detection_stats["total_steps"] += 1.0
        self._detection_stats["invalid_all_agents"] += (num_valid_detections == 0).float()
        self._detection_stats["pair_valid_count"] += (num_valid_detections >= 2).float()

        # Compute triangulation for all active task reward levels
        if self.cfg.enable_triangulation:
            task_level = self.cfg.task_reward_level

            # Level 1 (FIM): covariance at GT target position
            self._triangulation_result_gt = self._compute_triangulation(
                states=reward_states, use_gt_target=True
            )

            # Level 2 (GT-anchored estimation error): triangulate with GT drone positions
            if task_level >= 2 or self.cfg.curriculum_task_levels:
                gt_states = self._build_gt_states()
                self._tri_result_l2 = self._compute_triangulation(
                    states=gt_states, use_gt_target=False
                )
            else:
                self._tri_result_l2 = None

            # Level 3 (E2E estimation error): triangulate with delayed drone positions
            if task_level >= 3 or self.cfg.curriculum_task_levels:
                delayed_states_e2e = self._get_delayed_states_for_e2e()
                self._tri_result_l3 = self._compute_triangulation(
                    states=delayed_states_e2e, use_gt_target=False
                )
            else:
                self._tri_result_l3 = None

        # Stack GT positions for CBF computation: (E, N, 3)
        gt_positions = torch.stack(
            [self._root_pos_w[agent_id] for agent_id in self.cfg.possible_agents],
            dim=1,
        )
        cmd_velocities = self.cmd_vel[:, :, 0:3]
        dt = self.cfg.sim.dt * self.cfg.decimation

        # Check collisions for sharp penalty (complements continuous CPA + termination)
        collided = self.cbf_manager.check_collisions(gt_positions)  # (E,) bool

        # Compute CBF penalty from GT state
        cbf_penalty_all = self.cbf_manager.compute_training_penalty(
            gt_positions=gt_positions,
            commanded_velocities=cmd_velocities,
            dt=dt,
        )  # (E,)

        # Compute rewards for each agent
        for i, agent_id in enumerate(self.cfg.possible_agents):
            delayed_state = reward_states[agent_id]

            # ---- Action penalties ----
            action_sum = torch.sum(
                torch.square(self.action_weight * self._actions[agent_id]), dim=1
            )
            action_delta = torch.sum(
                torch.square(
                    self.action_delta_weight
                    * (self._actions[agent_id] - self._last_actions[agent_id])
                ),
                dim=1,
            )

            # ---- BBox rewards (using delayed/observed state) ----
            bbox_center_raw = delayed_state.data.bboxes_2d[:, 0, 0:2]  # [N, 2] pixel center
            bbox_size_raw = delayed_state.data.bboxes_2d[:, 0, 2:4]    # [N, 2] pixel w,h
            bbox_raw = delayed_state.data.bboxes_2d[:, 0, :]  # [N, 4]
            bbox_valid = bbox_raw.abs().sum(dim=-1) > 1e-6  # [N]

            if self.cfg.use_omnidirectional_cameras:
                bbox_valid = torch.ones_like(bbox_valid)

            # Normalize to [0, 1]
            bbox_center = bbox_center_raw / self._img_dims
            bbox_size = bbox_size_raw / self._img_dims

            # Center reward: exp(-10 * dist_from_center) * valid
            bbox_center_dist = torch.norm(bbox_center - 0.5, dim=1)
            bbox_center_mapped = torch.exp(-10.0 * bbox_center_dist) * bbox_valid.float()

            # Size reward: exp(-|area - 0.2|) * valid (ideal bbox area = 20% of image)
            bbox_area = bbox_size[:, 0] * bbox_size[:, 1]
            bbox_size_mapped = torch.exp(-torch.abs(bbox_area - 0.2)) * bbox_valid.float()

            # ---- Triangulation / task reward (3 switchable levels) ----
            if self.cfg.enable_triangulation and self._triangulation_result_gt is not None:
                task_level = self.cfg.task_reward_level

                # Level 1: FIM (always computed for logging/curriculum)
                fim_reward = self._compute_fim_reward()

                # Level 2: GT-anchored estimation error
                if self._tri_result_l2 is not None:
                    est_error_gt_reward = self._compute_estimation_error_reward(
                        self._tri_result_l2
                    )
                else:
                    est_error_gt_reward = torch.zeros(self.num_envs, device=self.device)

                # Level 3: E2E estimation error
                if self._tri_result_l3 is not None:
                    est_error_e2e_reward = self._compute_estimation_error_reward(
                        self._tri_result_l3
                    )
                else:
                    est_error_e2e_reward = torch.zeros(self.num_envs, device=self.device)

                # Blend based on curriculum or fixed level
                if self.cfg.curriculum_task_levels:
                    l2_prog, l3_prog = self.cfg.curriculum.get_task_level_progress(
                        current_step
                    )
                    triangulation_quality = (
                        (1.0 - l2_prog) * fim_reward
                        + l2_prog * (1.0 - l3_prog) * est_error_gt_reward
                        + l2_prog * l3_prog * (
                            (1.0 - self.cfg.e2e_weight) * est_error_gt_reward
                            + self.cfg.e2e_weight * est_error_e2e_reward
                        )
                    )
                else:
                    if task_level == 1:
                        triangulation_quality = fim_reward
                    elif task_level == 2:
                        triangulation_quality = est_error_gt_reward
                    elif task_level == 3:
                        triangulation_quality = (
                            (1.0 - self.cfg.e2e_weight) * est_error_gt_reward
                            + self.cfg.e2e_weight * est_error_e2e_reward
                        )
                    else:
                        raise ValueError(
                            f"Unknown task_reward_level: {task_level}"
                        )
            else:
                triangulation_quality = torch.zeros(self.num_envs, device=self.device)
                fim_reward = torch.zeros(self.num_envs, device=self.device)
                est_error_gt_reward = torch.zeros(self.num_envs, device=self.device)
                est_error_e2e_reward = torch.zeros(self.num_envs, device=self.device)

            # Altitude penalty: penalize flying below threshold
            pos_z = self._root_pos_w[agent_id][:, 2]
            too_low = (pos_z < self.cfg.altitude_min_threshold).float()

            step_dt = self.cfg.sim.dt * self.cfg.decimation
            rewards = {
                "action_sum": action_sum * self.cfg.action_sum_penalty_scale * step_dt,
                "action_delta": action_delta * self.cfg.action_delta_penalty_scale * step_dt,
                "bbox_center": bbox_center_mapped * self.cfg.bbox_center_reward_scale * step_dt,
                "bbox_size": bbox_size_mapped * self.cfg.bbox_size_reward_scale * step_dt,
                "triangulation": (
                    triangulation_quality
                    * self.cfg.triangulation_reward_scale
                    * step_dt
                    * self.progress_coord
                ),
                "cbf_penalty": (
                    -self.cbf_manager.lambda_cbf
                    * cbf_penalty_all
                    * self.progress_safety
                ),
                "collision": collided.float() * self.cfg.collision_penalty_scale * step_dt,
                "altitude": too_low * self.cfg.altitude_penalty_scale * step_dt,
                "est_error_gt": est_error_gt_reward,
                "est_error_e2e": est_error_e2e_reward,
            }

            # ---- NaN sanitization (zero out, don't terminate) ----
            for key in rewards:
                nan_mask = torch.isnan(rewards[key]) | torch.isinf(rewards[key])
                if nan_mask.any():
                    rewards[key] = torch.where(
                        nan_mask, torch.zeros_like(rewards[key]), rewards[key]
                    )

            # ---- Store per-step rewards for visualization ----
            self._step_rewards[agent_id] = {k: v.clone() for k, v in rewards.items()}

            # ---- Episode sum tracking ----
            for key, value in rewards.items():
                self._episode_sums[agent_id][key] += value

            # ---- Total reward (exclude diagnostic keys from sum) ----
            _diagnostic_keys = {"est_error_gt", "est_error_e2e"}
            total_reward = torch.sum(
                torch.stack([v for k, v in rewards.items() if k not in _diagnostic_keys]),
                dim=0,
            )
            rewards_dict[agent_id] = total_reward

            # Update last actions
            self._last_actions[agent_id] = self._actions[agent_id].clone()

        return rewards_dict

    def _linear_progress(self, start: int, end: int, current=None):
        """Calculate linear curriculum progress for a given phase.

        Args:
            start: Phase start step.
            end: Phase end step.
            current: Current step (defaults to common_step_counter).

        Returns:
            Progress value between 0.0 and 1.0.
        """
        if end <= start:
            return 1.0
        if current is None:
            current = self.common_step_counter
        x = (current - start) / (end - start)
        return 0.0 if x < 0 else (1.0 if x > 1 else float(x))

    def _compute_fim_reward(self) -> torch.Tensor:
        """Level 1: FIM reward from covariance trace.

        Uses the already-computed triangulation result (use_gt_target=True)
        to extract quality_metric (trace of covariance) and map it to a
        reward via sqrt(10 / trace).

        Returns:
            [N] tensor of FIM-based triangulation quality reward.
        """
        tri_result = self._triangulation_result_gt
        quality = tri_result.quality_metric[:, 0]  # trace of covariance
        is_valid = tri_result.is_valid[:, 0]
        safe_trace = torch.clamp(quality, min=1e-6)
        safe_trace = torch.where(
            torch.isnan(safe_trace),
            torch.full_like(safe_trace, 1e6),
            safe_trace,
        )
        return torch.where(
            is_valid,
            torch.clamp(torch.sqrt(10.0 / safe_trace), max=100.0),
            torch.zeros_like(quality),
        )

    def _compute_estimation_error_reward(
        self, tri_result: "TriangulationResult"
    ) -> torch.Tensor:
        """Compute reward from estimation error: scale * exp(-temp * ||error||).

        Maps estimation error to [0, scale] range using exponential.
        Used by both Level 2 (GT-anchored) and Level 3 (E2E).

        Args:
            tri_result: Triangulation result computed with use_gt_target=False,
                       so result.position contains the estimated target position.

        Returns:
            [N] tensor of estimation error reward.
        """
        est_pos = tri_result.position[:, 0, :]  # [N, 3] estimated target position
        gt_pos = self._target_pos_w             # [N, 3] GT target position
        is_valid = tri_result.is_valid[:, 0]    # [N]

        error = torch.norm(est_pos - gt_pos, dim=-1)  # [N]
        # Replace NaN errors with large value (invalid triangulation)
        error = torch.where(
            torch.isnan(error), torch.full_like(error, 100.0), error
        )

        reward = self.cfg.estimation_error_scale * torch.exp(
            -self.cfg.estimation_error_temp * error
        )
        return torch.where(is_valid, reward, torch.zeros_like(reward))

    def _get_delayed_states_for_e2e(self) -> Dict[str, "AgentStates"]:
        """Get fully delayed states for Level 3 E2E reward.

        Always uses the delay pipeline regardless of use_noisy_rewards config,
        since Level 3 is defined as triangulation with delayed drone positions.

        Returns:
            Dictionary mapping agent_id to delayed AgentStates.
        """
        if self._delay_system is not None:
            return self._delay_system.get_all_states_for_observations(
                ego_agent_id=self.cfg.possible_agents[0]
            )
        else:
            # Fallback to GT when no delay system is configured
            return self._build_gt_states()

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents.

        When delay system is enabled, observations use delayed states.
        Otherwise, uses ground truth states directly.

        Observation structure per agent:
        - Ego (26D): pos(3), vel(3), quat(4), ang_vel_b(3), lin_acc_b(3),
          gimbal_az_w(1), gimbal_el_w(1), gimbal_az_rate(1), gimbal_el_rate(1),
          zoom(1), bbox(4), bbox_empty(1)
        - Inter-agent (14D per other): pos(3), vel(3), gimbal_az_w(1),
          gimbal_el_w(1), gimbal_az_rate(1), gimbal_el_rate(1),
          zoom(1), bbox_empty(1), data_age(1), bbox_age(1)
        - Optional triangulation tail (6D): tri_pos(3), tri_std(3)

        Returns:
            Dictionary mapping agent_id to observation tensor.
        """
        obs = {}

        if self._delay_system is not None:
            # Use delayed states for each agent
            for idx, agent_id in enumerate(self.cfg.possible_agents):
                # Get delayed states from ego's perspective
                delayed_states = self._delay_system.get_all_states_for_observations(
                    ego_agent_id=agent_id
                )

                # Get ego's delayed self-state
                ego_states = delayed_states[agent_id]
                ego_data = ego_states.data

                # Bbox from delayed states (shape: N, T, 4) - pixel format
                bbox_pixel = ego_data.bboxes_2d[:, 0, :]  # First target
                bbox_empty = (bbox_pixel.abs().sum(dim=-1) < 1e-6).float().unsqueeze(-1)

                # Normalize bbox to [0, 1] for observation (xywh format)
                img_h, img_w = self._camera_image_shape
                bbox = bbox_pixel.clone()
                bbox[:, 0] /= img_w  # x center
                bbox[:, 1] /= img_h  # y center
                bbox[:, 2] /= img_w  # width
                bbox[:, 3] /= img_h  # height

                # Ego gimbal body-frame joints (yaw=0 means forward)
                ego_gimbal_yaw_body = (ego_data.joint_positions_b[:, 1] - YAW_JOINT_OFFSET).unsqueeze(-1)
                ego_gimbal_pitch_body = ego_data.joint_positions_b[:, 0:1]
                # Ego camera ray direction in world frame
                ego_ray_w = gimbal_ray_direction_world(
                    ego_gimbal_yaw_body.squeeze(-1),
                    ego_gimbal_pitch_body.squeeze(-1),
                    ego_data.body_orientation_w,
                )

                # Ego bbox age-of-information
                ego_bbox_aoi = (self._sim_time - ego_data.timestamp_detection).unsqueeze(-1)

                # Build ego observation (30D)
                ego_obs = torch.cat(
                    [
                        ego_data.body_position_w,  # (N, 3)
                        ego_data.body_linear_velocity_w,  # (N, 3)
                        wrap_to_pi(torch.stack(euler_xyz_from_quat(ego_data.body_orientation_w), dim=-1)),  # (N, 3) roll, pitch, yaw in [-pi, pi]
                        ego_data.body_angular_velocity_b,  # (N, 3) — gyro
                        ego_data.body_linear_acceleration_b,  # (N, 3) — accelerometer
                        ego_gimbal_yaw_body,  # (N, 1) — body-frame, 0=forward
                        ego_gimbal_pitch_body,  # (N, 1) — body-frame
                        ego_ray_w,  # (N, 3) — world-frame camera pointing direction
                        ego_data.body_combined_angular_velocity_w,  # (N, 3) — camera sweep rate
                        ego_bbox_aoi,  # (N, 1) — bbox age-of-information
                        ego_data.camera_zoom_level.unsqueeze(-1),  # (N, 1)
                        bbox,  # (N, 4) - normalized
                        bbox_empty,  # (N, 1)
                    ],
                    dim=-1,
                )

                # Build inter-agent observations (13D per other agent)
                other_obs_parts = []
                for other_id in self.cfg.possible_agents:
                    if other_id == agent_id:
                        continue
                    other_data = delayed_states[other_id].data

                    # Other agent's bbox_empty
                    other_bbox_pixel = other_data.bboxes_2d[:, 0, :]
                    other_bbox_empty = (
                        other_bbox_pixel.abs().sum(dim=-1) < 1e-6
                    ).float().unsqueeze(-1)

                    # Age of Information
                    data_age = (self._sim_time - other_data.timestamp_motion).unsqueeze(-1)
                    bbox_age = (self._sim_time - other_data.timestamp_detection).unsqueeze(-1)

                    # Other agent's camera ray direction in world frame
                    other_ray_w = gimbal_ray_direction_world(
                        other_data.joint_positions_b[:, 1] - YAW_JOINT_OFFSET,
                        other_data.joint_positions_b[:, 0],
                        other_data.body_orientation_w,
                    )

                    other_obs_parts.append(
                        torch.cat(
                            [
                                other_data.body_position_w,  # (N, 3)
                                other_data.body_linear_velocity_w,  # (N, 3)
                                other_ray_w,  # (N, 3) — world-frame camera pointing
                                other_data.body_combined_angular_velocity_w,  # (N, 3) — camera sweep rate
                                other_data.camera_zoom_level.unsqueeze(-1),  # (N, 1)
                                other_bbox_empty,  # (N, 1)
                                data_age,  # (N, 1)
                                bbox_age,  # (N, 1)
                            ],
                            dim=-1,
                        )
                    )

                obs[agent_id] = torch.cat([ego_obs] + other_obs_parts, dim=-1)

        else:
            # Use ground truth states directly
            # Build GT states for inter-agent access
            gt_all = self._build_gt_states()

            for idx, agent_id in enumerate(self.cfg.possible_agents):
                bbox = self.bbox_raycaster_v2.data.bboxes_normalized[:, idx, 0, :]
                bbox_empty = (bbox.abs().sum(dim=-1) < 1e-6).float().unsqueeze(-1)

                ego_gt = gt_all[agent_id].data

                # Ego gimbal body-frame joints (yaw=0 means forward)
                ego_gimbal_yaw_body = (ego_gt.joint_positions_b[:, 1] - YAW_JOINT_OFFSET).unsqueeze(-1)
                ego_gimbal_pitch_body = ego_gt.joint_positions_b[:, 0:1]
                # Ego camera ray direction in world frame
                ego_ray_w = gimbal_ray_direction_world(
                    ego_gimbal_yaw_body.squeeze(-1),
                    ego_gimbal_pitch_body.squeeze(-1),
                    self._root_quat_w[agent_id],
                )

                # Ego observation (30D) — GT path: bbox AoI = 0 (no delay)
                ego_obs = torch.cat(
                    [
                        self._root_pos_w[agent_id],  # (N, 3)
                        self._root_lin_vel_w[agent_id],  # (N, 3)
                        wrap_to_pi(torch.stack(euler_xyz_from_quat(self._root_quat_w[agent_id]), dim=-1)),  # (N, 3) roll, pitch, yaw in [-pi, pi]
                        self._root_ang_vel_b[agent_id],  # (N, 3) — gyro
                        self._root_lin_acc_b[agent_id],  # (N, 3) — accelerometer
                        ego_gimbal_yaw_body,  # (N, 1) — body-frame, 0=forward
                        ego_gimbal_pitch_body,  # (N, 1) — body-frame
                        ego_ray_w,  # (N, 3) — world-frame camera pointing direction
                        ego_gt.body_combined_angular_velocity_w,  # (N, 3) — camera sweep rate
                        torch.zeros(self.num_envs, 1, device=self.device),  # (N, 1) — bbox AoI (GT = 0)
                        self.zoom_level[:, idx : idx + 1],  # (N, 1)
                        bbox,  # (N, 4)
                        bbox_empty,  # (N, 1)
                    ],
                    dim=-1,
                )

                # Inter-agent observations (13D per other, GT = no delay, ages = 0)
                other_obs_parts = []
                zero_age = torch.zeros(self.num_envs, 1, device=self.device)
                for other_idx, other_id in enumerate(self.cfg.possible_agents):
                    if other_id == agent_id:
                        continue
                    other_gt = gt_all[other_id].data

                    other_bbox = self.bbox_raycaster_v2.data.bboxes_normalized[:, other_idx, 0, :]
                    other_bbox_empty = (other_bbox.abs().sum(dim=-1) < 1e-6).float().unsqueeze(-1)


                    # Other agent's camera ray direction in world frame
                    other_ray_w = gimbal_ray_direction_world(
                        other_gt.joint_positions_b[:, 1] - YAW_JOINT_OFFSET,
                        other_gt.joint_positions_b[:, 0],
                        other_gt.body_orientation_w,
                    )

                    other_obs_parts.append(
                        torch.cat(
                            [
                                other_gt.body_position_w,  # (N, 3)
                                other_gt.body_linear_velocity_w,  # (N, 3)
                                other_ray_w,  # (N, 3) — world-frame camera pointing
                                other_gt.body_combined_angular_velocity_w,  # (N, 3) — camera sweep rate
                                self.zoom_level[:, other_idx : other_idx + 1],  # (N, 1)
                                other_bbox_empty,  # (N, 1)
                                zero_age,  # (N, 1) — GT, no delay
                                zero_age,  # (N, 1) — GT, no delay
                            ],
                            dim=-1,
                        )
                    )

                obs[agent_id] = torch.cat([ego_obs] + other_obs_parts, dim=-1)

        # Append triangulation tail if enabled (observation pipeline uses triangulated position)
        if self.cfg.enable_triangulation:
            # Get states for triangulation
            if self._delay_system is not None:
                # Use delayed states from first agent's perspective
                tri_states = self._delay_system.get_all_states_for_observations(
                    ego_agent_id=self.cfg.possible_agents[0]
                )
            else:
                tri_states = self._build_gt_states()

            # Compute triangulation WITHOUT GT (uses midpoint method for position)
            self._triangulation_result_obs = self._compute_triangulation(
                states=tri_states, use_gt_target=False
            )

            result = self._triangulation_result_obs
            is_valid = result.is_valid[:, 0]  # [N]

            # Triangulated position (zero fallback when invalid, matching MA5)
            tri_pos = torch.where(
                is_valid.unsqueeze(-1).expand(-1, 3),
                result.position[:, 0, :],  # [N, 3] - triangulated
                torch.zeros(self.num_envs, 3, device=self.device),  # zero fallback
            )

            # Standard deviation from covariance diagonal (or -1 as invalid marker)
            # Handle NaN values in covariance
            cov = result.covariance[:, 0, :, :]  # [N, 3, 3]
            cov_diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # [N, 3]

            # Replace NaN with large values before sqrt
            cov_diag_safe = torch.where(
                torch.isnan(cov_diag),
                torch.ones_like(cov_diag),  # Will become -1 due to invalid mask
                cov_diag,
            )
            cov_diag_safe = torch.clamp(cov_diag_safe, min=1e-12)

            tri_std = torch.where(
                is_valid.unsqueeze(-1).expand(-1, 3),
                torch.sqrt(cov_diag_safe),
                torch.full_like(cov_diag, -1.0),  # Invalid marker
            )

            # Append triangulation tail to each agent's observation
            for agent_id in self.cfg.possible_agents:
                obs[agent_id] = torch.cat(
                    [
                        obs[agent_id],
                        tri_pos,   # [N, 3] - triangulated position
                        tri_std,   # [N, 3] - uncertainty std_dev
                    ],
                    dim=-1,
                )

        return obs

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get termination and truncation flags for all agents.

        Termination conditions:
        - Crashed: drone goes below z=2.0m
        - Collision: inter-agent distance < collision_distance (GT-based)

        Returns:
            Tuple of (terminated, truncated) dictionaries.
        """
        terminated = {}
        truncated = {}

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check collisions using GT positions
        gt_positions = torch.stack(
            [self._root_pos_w[agent_id] for agent_id in self.cfg.possible_agents],
            dim=1,
        )
        collided = self.cbf_manager.check_collisions(gt_positions)  # (E,)

        # Track collision events per episode
        self._collision_count += collided.float()

        for agent_id in self.cfg.possible_agents:
            # Terminate if drone goes too low (crashed) OR collision
            pos_z = self._root_pos_w[agent_id][:, 2]
            crashed = pos_z < 0.5
            # died = crashed | collided
            died = crashed

            terminated[agent_id] = died
            truncated[agent_id] = time_out & ~terminated[agent_id]

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at specified indices.

        Uses InitialStates module for curriculum-driven randomization if enabled,
        otherwise falls back to hardcoded triangle formation.

        Args:
            env_ids: Environment indices to reset.
        """
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        self.progress_dynamics = self._linear_progress(
            self.cfg.curriculum.dynamics_start_step,
            self.cfg.curriculum.dynamics_end_step,
        )

        if self._initial_states is not None:
            # Generate randomized initial states via InitialStates module
            result = self._initial_states.generate(
                env_ids=torch.arange(num_reset, device=self.device),
                curriculum_progress=self.progress_tracking,
            )

            # Apply agent states
            for idx, agent_id in enumerate(self.cfg.possible_agents):
                robot = self._robots[agent_id]

                # Root pose: position + orientation
                # Add terrain origin offset
                agent_pos = result.agent_positions[:, idx] + self._terrain.env_origins[env_ids]
                agent_quat = result.agent_orientations[:, idx]
                root_pose = torch.cat([agent_pos, agent_quat], dim=-1)

                # Root velocity: linear + angular
                root_vel = torch.cat([
                    result.agent_linear_velocities[:, idx],
                    result.agent_angular_velocities[:, idx],
                ], dim=-1)

                robot.write_root_pose_to_sim(root_pose, env_ids)
                robot.write_root_velocity_to_sim(root_vel, env_ids)

                # Apply gimbal joint states
                # Result has [yaw, roll, pitch], need to set via joint indices
                gimbal_angles = result.gimbal_joint_positions[:, idx]  # [N, 3] = [yaw, roll, pitch]

                # Build full joint position tensor from defaults
                joint_pos = robot.data.default_joint_pos[env_ids].clone()
                joint_vel = torch.zeros_like(joint_pos)

                # Set gimbal joints with YAW_JOINT_OFFSET applied to yaw
                yaw_idx = self.gimbal_joint_idx[agent_id]["yaw"]
                roll_idx = self.gimbal_joint_idx[agent_id]["roll"]
                pitch_idx = self.gimbal_joint_idx[agent_id]["pitch"]

                joint_pos[:, yaw_idx] = gimbal_angles[:, 0] + YAW_JOINT_OFFSET
                joint_pos[:, roll_idx] = gimbal_angles[:, 1]
                joint_pos[:, pitch_idx] = gimbal_angles[:, 2]

                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

                # Reset controller state and sync gimbal internal state
                self._controllers[agent_id].reset(env_ids)
                # Sync gimbal controller internal state to initial body-frame angles
                self._controllers[agent_id]._gimbal._yaw[env_ids] = gimbal_angles[:, 0]
                self._controllers[agent_id]._gimbal._pitch[env_ids] = gimbal_angles[:, 2]

                # Compute world-frame azimuth/elevation from the ACTUAL initial
                # body-frame gimbal angles (not always the target direction).
                # When curriculum randomizes gimbal, the initial angles may NOT
                # point at the target.  Using _body_to_world_angles ensures
                # _azimuth_world/_elevation_world are consistent with _yaw/_pitch,
                # so the first compute_control() produces no gimbal movement.
                azimuth_world, elevation_world = (
                    self._controllers[agent_id]._gimbal._body_to_world_angles(
                        gimbal_angles[:, 0],   # body-frame yaw
                        gimbal_angles[:, 2],   # body-frame pitch
                        result.agent_orientations[:, idx],  # body quaternion
                    )
                )

                self._controllers[agent_id]._gimbal._azimuth_world[env_ids] = azimuth_world
                self._controllers[agent_id]._gimbal._elevation_world[env_ids] = elevation_world

                # Set zoom level (both env tracking tensor AND controller internal state).
                # The controller.reset() above resets zoom to 1.0, so we must
                # sync the random initial zoom into the controller's internal
                # _zoom and _zoom_target to prevent the first _apply_action from
                # overwriting it back to 1.0.
                self.zoom_level[env_ids, idx] = result.zoom_levels[:, idx]
                self._controllers[agent_id]._zoom.set_zoom(
                    result.zoom_levels[:, idx], env_ids
                )

                # Curriculum-scaled tau_zoom: near-instant at progress=0,
                # ramp to configured value at full dynamics progress.
                # When progress > 0, randomize_gains() will overwrite with
                # per-env randomized values centered on this nominal.
                tau_zoom_curriculum = max(
                    self.cfg.drone_controller.zoom.tau_zoom * self.progress_dynamics, 1e-4
                )
                self._controllers[agent_id]._zoom.set_tau_zoom(tau_zoom_curriculum)

            # Apply target states
            target_pos = result.target_positions + self._terrain.env_origins[env_ids]
            target_quat = result.target_orientations
            target_pose = torch.cat([target_pos, target_quat], dim=-1)

            self.target.write_root_pose_to_sim(target_pose, env_ids)
            self.target.write_root_velocity_to_sim(result.target_velocities, env_ids)

        else:
            # Fallback to hardcoded triangle formation
            self._reset_idx_hardcoded(env_ids)

        # Reset command buffers
        self.cmd_vel[env_ids] = 0.0
        if self._initial_states is None:
            self.zoom_level[env_ids] = 1.0
        self.cmd_gimbal_yaw[env_ids] = 0.0
        self.cmd_gimbal_pitch[env_ids] = 0.0

        # Reset CBF manager state
        self.cbf_manager.reset(env_ids)

        # Reset target controller state
        if self._target_controller is not None:
            self._target_controller.reset(env_ids)

        # Reset delay system and randomize per-agent delay parameters
        if self._delay_system is not None:
            self._delay_system.reset(env_ids)
            self._delay_system.randomize_per_agent_params(env_ids)

        # Scale max_lin_vel with agent velocity curriculum (decoupled from target motion):
        # 3 m/s at progress=0, full (10 m/s) at progress=1
        self._max_lin_vel[env_ids] = self.cfg.max_lin_vel_min + self.progress_agent_velocity * (
            self.cfg.max_lin_vel - self.cfg.max_lin_vel_min
        )

        # Randomize controller gains (curriculum-gated to dynamics phase)
        if self.progress_dynamics > 0.0:
            tau_zoom_curriculum = max(
                self.cfg.drone_controller.zoom.tau_zoom * self.progress_dynamics, 1e-4
            )
            for agent_id in self.cfg.possible_agents:
                self._controllers[agent_id].randomize_gains(
                    env_ids, self.progress_dynamics, self.cfg.gain_randomization,
                    tau_zoom_nominal=tau_zoom_curriculum,
                )

            # Randomize max linear velocity (same dynamics curriculum gate)
            gain_cfg = self.cfg.gain_randomization
            if gain_cfg.randomize_max_lin_vel:
                low = 1.0 - self.progress_dynamics * (1.0 - gain_cfg.max_lin_vel_scale_range[0])
                high = 1.0 + self.progress_dynamics * (gain_cfg.max_lin_vel_scale_range[1] - 1.0)
                M = len(env_ids)
                scale = torch.empty(M, device=self.device).uniform_(low, high)
                self._max_lin_vel[env_ids] = self.cfg.max_lin_vel * scale

        self._sim_time[env_ids] = 0.0

        # Reset smoothed occlusion confidence to visible state
        self._smoothed_bbox_confidence[env_ids] = 1.0

        # Log episode reward sums and reset tracking
        all_extras = {}
        for agent_id in self.cfg.possible_agents:
            self._last_actions[agent_id][env_ids].zero_()
            for key, value in self._episode_sums[agent_id].items():
                episodic_sum_avg = torch.mean(value[env_ids])
                all_extras[f"Episode_Reward/{agent_id}_{key}"] = (
                    episodic_sum_avg / self.max_episode_length_s
                )
                self._episode_sums[agent_id][key][env_ids] = 0.0

        # Log detection dropout statistics
        total_steps = torch.clamp(self._detection_stats["total_steps"][env_ids], min=1.0)
        all_extras["Detection/all_invalid_rate"] = torch.mean(
            self._detection_stats["invalid_all_agents"][env_ids] / total_steps
        )
        all_extras["Detection/pair_valid_rate"] = torch.mean(
            self._detection_stats["pair_valid_count"][env_ids] / total_steps
        )

        # Log collision count and fraction (collision steps / episode length)
        all_extras["Safety/collision_per_env"] = torch.mean(self._collision_count[env_ids])
        all_extras["Safety/collision_fraction"] = torch.mean(
            self._collision_count[env_ids] / total_steps
        )

        # Reset detection stats and collision count for these envs
        for key in self._detection_stats:
            self._detection_stats[key][env_ids] = 0.0
        self._collision_count[env_ids] = 0.0

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(all_extras)

        # Clear triangulation results (will be recomputed on first step)
        self._triangulation_result_gt = None
        self._triangulation_result_obs = None

        # Point viewer camera at target when the tracked env resets
        if self.viewport_camera_controller is not None and self.cfg.viewer.env_index in env_ids:
            target_pos_w = self.target.data.root_pos_w[self.cfg.viewer.env_index]  # [3]
            robot_pos_w = self._robots[self.cfg.possible_agents[0]].data.root_pos_w[self.cfg.viewer.env_index]  # [3]
            lookat_offset = (target_pos_w - robot_pos_w).detach().cpu().numpy() * 1.5
            import numpy as np
            angle = np.arctan2(-lookat_offset[1], -lookat_offset[0])
            eye = np.array([10 * np.cos(angle), 10 * np.sin(angle), 1.0])
            self.viewport_camera_controller.update_view_location(eye=eye)

        # Populate state caches so _get_observations works on first reset
        self._update_state_cache()

    def _reset_idx_hardcoded(self, env_ids: torch.Tensor):
        """Hardcoded reset logic (fallback when initial_states is disabled).

        Places agents in a triangle formation around origin with target at 10m.

        Args:
            env_ids: Environment indices to reset.
        """
        num_reset = len(env_ids)

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Calculate initial position in a formation around origin
            # Agents positioned in a triangle formation
            angle = (idx - 1) * 2 * 3.14159 / len(self.cfg.possible_agents)
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
        target_pos[:, 0] += 10.0
        target_pos[:, 2] = 3.5  # Target height
        target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_reset, 4)
        target_vel = torch.zeros(num_reset, 6, device=self.device)

        self.target.write_root_pose_to_sim(torch.cat([target_pos, target_quat], dim=-1), env_ids)
        self.target.write_root_velocity_to_sim(target_vel, env_ids)

        # Reset zoom level for hardcoded mode
        self.zoom_level[env_ids] = 1.0

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
        if self.cfg.debug_frame_vis:
            self._visualization.update_frames(link_poses)

        # Draw covariance ellipsoids for triangulation uncertainty
        # GT path: center on actual target position (not midpoint triangulation)
        # Covariance is evaluated at GT position, so ellipsoid shows uncertainty there
        if self.cfg.enable_triangulation and self._triangulation_result_gt is not None:
            result = self._triangulation_result_gt
            self._visualization.update_covariance_ellipsoids(
                translations=self._target_pos_w,  # GT target position — stable
                covariance=result.covariance[:, 0, :, :],  # [N, 3, 3] - first target
                is_valid=result.is_valid[:, 0],  # [N] - first target
            )
        # Draw observed triangulation (gray) alongside GT (cyan)
        if self.cfg.enable_triangulation and self._triangulation_result_obs is not None:
            result_obs = self._triangulation_result_obs
            self._visualization.update_covariance_ellipsoids_obs(
                translations=result_obs.position[:, 0, :],  # [N, 3] - first target
                covariance=result_obs.covariance[:, 0, :, :],  # [N, 3, 3] - first target
                is_valid=result_obs.is_valid[:, 0],  # [N] - first target
            )
