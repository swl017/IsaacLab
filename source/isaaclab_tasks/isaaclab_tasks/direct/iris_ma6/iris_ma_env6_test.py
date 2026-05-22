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
import numpy as np
import torch
from typing import Dict, List

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import euler_xyz_from_quat, quat_mul, quat_rotate, quat_rotate_inverse, wrap_to_pi, yaw_quat

from .bbox_raycaster_v2 import BBoxRayCasterV2
from .bbox_raycaster_v2.utils.projection import create_intrinsic_matrix_tensor
from .cbf_safety import CBFManager
from .domain_randomization import DomainRandomizer
from .controller import DroneController
from .controller.gimbal_controller import YAW_JOINT_OFFSET
from .controller.zoom_controller import compute_z_eff
from .curriculum.progress_helper import sample_per_env_progress
from .delay_system_v3 import MultiAgentDelaySystemV3, AgentStates
from .initial_states import InitialStates
from .delay_system_v3.derived_field_computers import compute_combined_angular_velocity, compute_ray_directions_from_bbox
from .iris_ma_env6_test_cfg import (
    _CRITIC_PRIVILEGED_FIELD_REGISTRY,
    IrisMA6TestEnvCfg,
)
from .target_controller import TargetController
from .triangulation import (
    TriangulationResult,
    compute_full_triangulation,
    compute_sigma_K_from_zoom,
    compute_sigma_drift,
)
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


def az_el_to_gimbal_angles(
    azimuth_world: torch.Tensor,
    elevation_world: torch.Tensor,
    q_body: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert world-frame azimuth/elevation to body-frame gimbal (yaw, roll, pitch).

    Inverse of body_to_world_gimbal_angles / gimbal_ray_direction_world.

    Args:
        azimuth_world: (N,) desired world-frame azimuth (rad).
        elevation_world: (N,) desired world-frame elevation (rad).
        q_body: (N, 4) body orientation quaternion (wxyz).

    Returns:
        Tuple of (yaw_body, roll_body, pitch_body), each (N,) in radians.
        roll_body is 0 (horizon-level auto-stabilization).
    """
    # Desired direction in world frame
    cos_el = torch.cos(elevation_world)
    dir_world = torch.stack(
        [
            cos_el * torch.cos(azimuth_world),
            cos_el * torch.sin(azimuth_world),
            torch.sin(elevation_world),
        ],
        dim=-1,
    )
    # Rotate into body frame
    dir_body = quat_rotate_inverse(q_body, dir_world)
    # Invert the gimbal_ray_direction_world mapping:
    #   dir_body = [cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), -sin(pitch)]
    yaw_body = torch.atan2(dir_body[..., 1], dir_body[..., 0])
    pitch_body = -torch.asin(dir_body[..., 2].clamp(-1.0, 1.0))
    roll_body = torch.zeros_like(yaw_body)  # auto-stabilize horizon
    return yaw_body, roll_body, pitch_body


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

        # ------------------------------------------------------------------
        # Asymmetric actor-critic plumbing: when any ``enable_critic_*`` flag
        # is True, ``cfg.state_space`` has been set to a positive int (in
        # cfg.__post_init__), and ``DirectMARLEnv._configure_env_spaces`` has
        # wrapped it in a gym Box at ``self.state_space``. The skrl trainer's
        # MAPPO setup (train_mappo_rnn_hydra.py) reads
        # ``env.shared_observation_spaces`` (a dict, not the singular
        # ``state_space``) to size the centralized critic network and its
        # running-stats preprocessor; mirror the state_space Box into
        # per-agent entries here so that path picks up the privileged-tail
        # dimension instead of falling back to the auto-concat of actor obs
        # (which is smaller and triggers a tensor-size mismatch in the
        # preprocessor at first record_transition).
        # ------------------------------------------------------------------
        _asymmetric_critic_enabled = (
            getattr(self.cfg, "enable_critic_continuous_zoom", False)
            or getattr(self.cfg, "enable_critic_gt_target", False)
            # Ticket 037: privileged env-param tail is also asymmetric.
            or bool(getattr(self.cfg, "critic_privileged_fields", []))
        )
        if _asymmetric_critic_enabled and self.state_space is not None:
            self.shared_observation_spaces = {
                a: self.state_space for a in self.cfg.possible_agents
            }

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

        # Single batched DroneController for all agents.
        # Layout: rows [0, N) = agent 0, [N, 2N) = agent 1, etc.
        # All agents share the same URDF so mass is identical.
        first_robot = self._robots[cfg.possible_agents[0]]
        all_masses = first_robot.root_physx_view.get_masses()
        mass = all_masses[0].sum().item()
        self._controller = DroneController(
            cfg=self.cfg.drone_controller,
            mass=mass,
            gravity=9.81,
            num_envs=self.num_envs * len(cfg.possible_agents),
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

        # Action smoothness tracking (per-dimension RMS of action deltas)
        num_action_dims = len(self.cfg.action_weight)
        self._action_delta_sq_acc: Dict[str, torch.Tensor] = {
            agent_id: torch.zeros(self.num_envs, num_action_dims, device=self.device)
            for agent_id in cfg.possible_agents
        }
        self._action_smoothness_steps = torch.zeros(self.num_envs, device=self.device)

        # State caches (populated in _apply_action, consumed by rewards/dones/obs)
        self._root_pos_w: Dict[str, torch.Tensor] = {}
        self._root_quat_w: Dict[str, torch.Tensor] = {}
        self._root_lin_vel_w: Dict[str, torch.Tensor] = {}
        self._root_ang_vel_b: Dict[str, torch.Tensor] = {}
        self._root_lin_acc_w: Dict[str, torch.Tensor] = {}
        self._root_lin_acc_b: Dict[str, torch.Tensor] = {}
        self._gimbal_joint_pos: Dict[str, torch.Tensor] = {}
        self._gimbal_joint_vel: Dict[str, torch.Tensor] = {}
        self._gimbal_az_rate_world: Dict[str, torch.Tensor] = {}
        self._gimbal_el_rate_world: Dict[str, torch.Tensor] = {}
        self._cached_gt_states: Dict[AgentID, AgentStates] | None = None
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

        # Setup detector replicator for calibrated bbox noise
        if self.cfg.calibrated_bbox_noise.enabled:
            self.bbox_raycaster_v2.setup_detector_replicator(self.cfg.calibrated_bbox_noise)

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

        # Pre-allocated AgentStates buffers — reused each step to avoid ~50 torch.zeros() per step
        self._delay_gt_buffers: Dict[AgentID, AgentStates] = {
            agent_id: AgentStates(self.num_envs, 3, 1, self.device)
            for agent_id in self.cfg.possible_agents
        }
        self._gt_state_cache: Dict[AgentID, AgentStates] = {
            agent_id: AgentStates(self.num_envs, 3, 1, self.device)
            for agent_id in self.cfg.possible_agents
        }

        # Pre-allocated DR intrinsics tensor (N, A, 3, 3) — avoids per-agent clone each step
        self._dr_intrinsics_per_agent = self._camera_intrinsics_base.unsqueeze(1).expand(
            -1, num_agents, -1, -1
        ).clone().contiguous()

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
                    "target_proximity",
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

        # Tracking-lost truncation: consecutive steps with zero valid detections
        self._all_lost_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tracking_lost_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._num_valid_detections = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._detection_reacquire_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tracking_lost_timeout_steps = int(
            self.cfg.tracking_lost_timeout_s / (self.cfg.sim.dt * self.cfg.decimation)
        )

        # Curriculum progress factors. Reward-gating progress signals are
        # per-env (N,) tensors written in _reset_idx and held constant for
        # the episode — avoids mid-episode reward-function changes that bias
        # GAE bootstrapped value targets. progress_coord is a step-at-
        # episode-boundary (0 → 1 once); progress_safety is a per-env
        # ramp value sampled at episode start.
        self.progress_coord = torch.zeros(self.num_envs, device=self.device)
        self.progress_safety = torch.zeros(self.num_envs, device=self.device)

        # Triangulation module initialization
        # Dual pipeline: GT for rewards, triangulated for observations
        self._triangulation_result_gt: TriangulationResult | None = None
        self._triangulation_result_obs: TriangulationResult | None = None

        # Per-agent bbox-non-empty flag, cached during _get_rewards from the same
        # source the policy/reward pipeline uses (delayed if delay system on, else GT).
        # Read by get_aux_supervision() for the per-agent half of the policy tri head's
        # validity mask. Shape: [num_envs, num_agents], bool. None until first step.
        self._per_agent_bbox_nonempty: torch.Tensor | None = None

        # Auxiliary-supervision caches for the policy tri head (ticket 031). Populated
        # in _get_rewards and intentionally NOT cleared by _reset_idx so that
        # get_aux_supervision() — called externally after env.step() — sees the data
        # from the just-completed _get_rewards even when some envs reset in this step.
        # Shapes: position (N, A, 3) float32, valid (N, A) bool. None until first step.
        self._aux_target_position_w_cache: torch.Tensor | None = None
        self._aux_target_valid_cache: torch.Tensor | None = None

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

        # Initialize domain randomizer for sim-to-real transfer
        if self.cfg.domain_randomization.enabled:
            self._domain_randomizer = DomainRandomizer(
                cfg=self.cfg.domain_randomization,
                num_envs=self.num_envs,
                num_agents=len(cfg.possible_agents),
                device=self.device,
            )
        else:
            self._domain_randomizer = None

        # DR buffers: per-env intrinsic scale, gimbal offsets, target scale
        self._dr_intrinsic_scale = torch.ones(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._dr_gimbal_offsets = torch.zeros(
            self.num_envs, len(cfg.possible_agents), 3, device=self.device
        )
        # Target z-scale for bbox size randomization: (N, 1, 3) = (x=1, y=1, z=sampled)
        self._dr_target_scale = torch.ones(
            self.num_envs, 1, 3, device=self.device
        )

        # Curriculum progress tracking (updated externally, used by initial_states)
        self.progress_tracking = 0.0
        self.progress_moving_target = 0.0
        self.progress_agent_velocity = 0.0
        self.progress_delay = 0.0
        self.progress_dynamics = 0.0
        self.progress_zoom_tau = 0.0  # mas/037: decoupled from progress_dynamics
        # Ticket 037 Slice 7 — per-axis scalar progresses (mean across (env, agent)
        # for logging back-compat; the per-(env, agent) draws live in
        # `_eff_progress_*` further below). Fall back to `progress_dynamics`
        # when the per-axis curriculum (start, end) is unset.
        self.progress_gimbal_rate_tau = 0.0
        self.progress_drone_gains = 0.0
        self.progress_max_lin_vel_scale = 0.0
        self.progress_camera_fov = 0.0
        self.progress_gimbal_mech_offsets = 0.0
        self.progress_mass_inertia = 0.0
        self.progress_gimbal_stiff_damp = 0.0
        self.progress_target_scale = 0.0
        self._curriculum_noise_scale = 0.0
        self._curriculum_fp_fn_scale = 0.0

        # Ticket 034: env-owned RNG for per-(env, agent) curriculum effective-progress
        # sampling. Seeded from cfg.seed so two training runs with the same seed
        # produce identical curriculum-jitter samples. Stays on the env's device.
        # An explicit cfg.seed is required — silently defaulting causes
        # non-reproducible runs that look reproducible in their logs.
        if self.cfg.seed is None:
            raise ValueError(
                "IrisMA6TestEnvCfg.seed must be set explicitly for ticket-034 "
                "per-(env, agent) curriculum sampling to be reproducible. "
                "Pass --seed <int> at the trainer / set seed in your hydra cfg."
            )
        self._curriculum_generator = torch.Generator(device=self.device)
        self._curriculum_generator.manual_seed(int(self.cfg.seed))

        # Ticket 034: per-(env, agent) effective-progress tensors. Written in
        # _reset_idx via curriculum.progress_helper.sample_per_env_progress;
        # held constant for an episode; read by step-time hooks in _get_rewards
        # and by reset-time consumers below.
        # Slice 1: agent_velocity. Slice 2: dynamics, gimbal_dead_time.
        # Slice 3: zoom_dead_time. Slice 4: noise, dropout, delay, burst_dropout.
        self._eff_progress_agent_velocity = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_dynamics = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_gimbal_dead_time = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_zoom_dead_time = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_noise = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_dropout = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_delay = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_burst_dropout = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        # Ticket 037 Slice 7 — per-axis decoupling of `_eff_progress_dynamics`.
        # Each of these 8 tensors is per-(env, agent), independently sampled
        # at reset from its own curriculum schedule. When
        # `enable_axis_independence=False` (default), all 8 fall back to the
        # bundled `dynamics_*` schedule (= _eff_progress_dynamics value),
        # preserving bit-exact t034 behavior. Consumers read from these
        # per-axis tensors instead of the old `_eff_progress_dynamics`.
        self._eff_progress_gimbal_rate_tau = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_drone_gains = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_max_lin_vel_scale = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_camera_fov = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_gimbal_mech_offsets = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_mass_inertia = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_gimbal_stiff_damp = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )
        self._eff_progress_target_scale = torch.zeros(
            self.num_envs, len(cfg.possible_agents), device=self.device
        )

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

        # Offset ground plane downward so it doesn't occlude the scene
        import omni.usd
        from pxr import Gf, UsdGeom
        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath(self.cfg.terrain.prim_path + "/terrain")
        if ground_prim.IsValid():
            ground_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, -20.0))

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

    def _batch_idx(self, env_ids: torch.Tensor, agent_idx: int) -> torch.Tensor:
        """Map env_ids to batched controller row indices for a given agent.

        Layout: agent 0 → rows [0, N), agent 1 → [N, 2N), etc.
        """
        return env_ids + agent_idx * self.num_envs

    # ------------------------------------------------------------------ Ticket 037
    def _axis_progress(self, axis: str) -> float:
        """Scalar per-axis dynamics progress; falls back to `progress_dynamics`
        when `enable_axis_independence=False` (bit-exact t034 behavior).

        ``axis`` is one of: gimbal_rate_tau, drone_gains, max_lin_vel_scale,
        camera_fov, gimbal_mech_offsets, mass_inertia, gimbal_stiff_damp,
        target_scale.
        """
        if not self.cfg.enable_axis_independence:
            return self.progress_dynamics
        return getattr(self, f"progress_{axis}")

    def _axis_eff_progress(self, axis: str) -> torch.Tensor:
        """Per-(env, agent) effective progress for the axis; falls back to
        `_eff_progress_dynamics` when `enable_axis_independence=False`.

        Returns ``Tensor[N, A]`` either way.
        """
        if not self.cfg.enable_axis_independence:
            return self._eff_progress_dynamics
        return getattr(self, f"_eff_progress_{axis}")

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
            if self.cfg.enable_asymmetric_z_envelope:
                # Ticket 039 — match PX4 MPC_Z_VEL_MAX_{UP,DN}. xy keeps the
                # per-env curriculum+DR cap; z uses scalar PX4 bounds with a
                # sign-dependent scale (action>0 climb, action<0 descend).
                self.cmd_vel[:, idx, 0:2] = action[:, 0:2] * self._max_lin_vel.unsqueeze(-1)
                z = action[:, 2]
                z_scale = torch.where(
                    z >= 0,
                    torch.full_like(z, self.cfg.max_vel_z_up),
                    torch.full_like(z, self.cfg.max_vel_z_dn),
                )
                self.cmd_vel[:, idx, 2] = z * z_scale
            else:
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
        Uses a single batched controller call for all agents to reduce
        GPU kernel launch overhead (ticket-017).
        """
        N = self.num_envs
        A = len(self.cfg.possible_agents)
        agents = list(enumerate(self.cfg.possible_agents))

        # --- Gather inputs into (N*A, ...) batched tensors ---
        # cmd_vel is (N, A, 7) — reshape agent dim into batch dim
        v_cmd_batch = torch.cat([self.cmd_vel[:, i, 0:3] for i in range(A)], dim=0)  # (N*A, 3)
        yaw_rate_batch = torch.cat([self.cmd_vel[:, i, 3] for i in range(A)], dim=0)  # (N*A,)
        gimbal_yaw_rate_batch = torch.cat([self.cmd_vel[:, i, 4] for i in range(A)], dim=0)
        gimbal_pitch_rate_batch = torch.cat([self.cmd_vel[:, i, 5] for i in range(A)], dim=0)
        zoom_rate_batch = torch.cat([self.cmd_vel[:, i, 6] for i in range(A)], dim=0)

        # Robot state: quaternion, linear vel, angular vel, gimbal joint positions
        q_parts, v_parts, omega_parts, gimbal_jp_parts = [], [], [], []
        for idx, agent_id in agents:
            robot = self._robots[agent_id]
            q_parts.append(robot.data.root_quat_w)
            v_parts.append(robot.data.root_lin_vel_w)
            omega_parts.append(robot.data.root_ang_vel_b)
            gimbal_jp_parts.append(torch.stack([
                robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]],
                robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]],
                robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["roll"]],
            ], dim=-1))

        q_batch = torch.cat(q_parts, dim=0)            # (N*A, 4)
        v_batch = torch.cat(v_parts, dim=0)             # (N*A, 3)
        omega_batch = torch.cat(omega_parts, dim=0)     # (N*A, 3)
        gimbal_jp_batch = torch.cat(gimbal_jp_parts, dim=0)  # (N*A, 3)

        # --- Single batched controller call ---
        # _apply_action runs every sim step (sim.dt = 0.01s), not every policy
        # step (sim.dt * decimation = 0.04s). The drone cascade needs the full
        # decimated dt for its inner-loop subdivision (velocity→attitude→rate).
        # The gimbal receives sim.dt separately for correct rate integration.
        F_body_batch, tau_body_batch, gimbal_pos_batch, gimbal_vel_batch, zoom_batch = (
            self._controller.step_policy(
                v_cmd=v_cmd_batch,
                yaw_rate_cmd=yaw_rate_batch,
                gimbal_yaw_rate_cmd=gimbal_yaw_rate_batch,
                gimbal_pitch_rate_cmd=gimbal_pitch_rate_batch,
                zoom_rate_cmd=zoom_rate_batch,
                q_body=q_batch,
                v_body=v_batch,
                omega_body=omega_batch,
                sim_dt=self.cfg.sim.dt * self.cfg.decimation,
                gimbal_joint_positions=gimbal_jp_batch,
                physics_dt=self.cfg.sim.dt,
            )
        )

        # Unpack batched gimbal outputs — each is (N*A,)
        gimbal_yaw_all, gimbal_roll_all, gimbal_pitch_all = gimbal_pos_batch
        gimbal_yaw_vel_all, gimbal_roll_vel_all, gimbal_pitch_vel_all = gimbal_vel_batch

        # --- Scatter outputs back per-agent (Isaac Sim API requires per-Articulation calls) ---
        for idx, agent_id in agents:
            robot = self._robots[agent_id]
            s = idx * N  # slice start

            # Apply forces in body frame (Isaac Lab expects local frame)
            robot.set_external_force_and_torque(
                forces=F_body_batch[s:s + N].unsqueeze(1),
                torques=tau_body_batch[s:s + N].unsqueeze(1),
                body_ids=self._body_ids[agent_id],
            )

            # DR gimbal offsets: simulate mechanical misalignment (yaw, pitch, roll)
            dr_yaw_off = self._dr_gimbal_offsets[:, idx, 0]
            dr_pitch_off = self._dr_gimbal_offsets[:, idx, 1]
            dr_roll_off = self._dr_gimbal_offsets[:, idx, 2]

            if not self.cfg.debug_lock_gimbal_to_target:
                # Apply gimbal position targets
                # Add YAW_JOINT_OFFSET (-π/2) to yaw: controller yaw=0 means body +X
                # (physics forward). The offset shifts the physical joint so the
                # combined chain (joint + camera offset) points along body +X.
                gimbal_yaw = gimbal_yaw_all[s:s + N]
                gimbal_roll = gimbal_roll_all[s:s + N]
                gimbal_pitch = gimbal_pitch_all[s:s + N]
                gimbal_yaw_joint = gimbal_yaw + YAW_JOINT_OFFSET + dr_yaw_off
                robot.set_joint_position_target(
                    target=torch.stack([gimbal_pitch + dr_pitch_off, gimbal_yaw_joint, gimbal_roll + dr_roll_off], dim=-1),
                    joint_ids=[
                        self.gimbal_joint_idx[agent_id]["pitch"],
                        self.gimbal_joint_idx[agent_id]["yaw"],
                        self.gimbal_joint_idx[agent_id]["roll"],
                    ],
                )

                # Apply gimbal velocity feedforward targets
                gimbal_yaw_vel = gimbal_yaw_vel_all[s:s + N]
                gimbal_roll_vel = gimbal_roll_vel_all[s:s + N]
                gimbal_pitch_vel = gimbal_pitch_vel_all[s:s + N]
                robot.set_joint_velocity_target(
                    target=torch.stack([gimbal_pitch_vel, gimbal_yaw_vel, gimbal_roll_vel], dim=-1),
                    joint_ids=[
                        self.gimbal_joint_idx[agent_id]["pitch"],
                        self.gimbal_joint_idx[agent_id]["yaw"],
                        self.gimbal_joint_idx[agent_id]["roll"],
                    ],
                )
            else:
                # point to region
                los_to_target = self._target_pos_w - robot.data.root_pos_w  # (N, 3)
                desired_az = torch.atan2(los_to_target[..., 1], los_to_target[..., 0])
                desired_el = torch.atan2(los_to_target[..., 2], torch.linalg.norm(los_to_target[..., 0:2], dim=-1))

                # Turn desired az/el into body-frame gimbal angles using inverse of camera offset rotation
                gimbal_yaw, gimbal_roll, gimbal_pitch = az_el_to_gimbal_angles(
                    desired_az, desired_el, robot.data.root_quat_w
                )

                gimbal_yaw_joint = gimbal_yaw + YAW_JOINT_OFFSET + dr_yaw_off
                gimbal_joint_ids = [
                    self.gimbal_joint_idx[agent_id]["pitch"],
                    self.gimbal_joint_idx[agent_id]["yaw"],
                    self.gimbal_joint_idx[agent_id]["roll"],
                ]
                pos_cmd = torch.stack([gimbal_pitch + dr_pitch_off, gimbal_yaw_joint, gimbal_roll + dr_roll_off], dim=-1)
                robot.set_joint_position_target(target=pos_cmd, joint_ids=gimbal_joint_ids)

            # Store zoom level for observations
            self.zoom_level[:, idx] = zoom_batch[s:s + N]

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
            self._gimbal_joint_vel[agent_id] = torch.stack(
                [
                    robot.data.joint_vel[:, self.gimbal_joint_idx[agent_id]["pitch"]],
                    robot.data.joint_vel[:, self.gimbal_joint_idx[agent_id]["yaw"]],
                    robot.data.joint_vel[:, self.gimbal_joint_idx[agent_id]["roll"]],
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

            # Compute zoomed+DR intrinsics in pre-allocated tensor (no clone).
            # zoom_level here is the operator zoom command; map it to the
            # measured effective focal multiplier (SIYI zoom curve) before
            # multiplying fx/fy.
            dr_scale = self._dr_intrinsic_scale[:, idx]
            z_eff = compute_z_eff(self.zoom_level[:, idx])
            combined_scale = z_eff * dr_scale
            self._dr_intrinsics_per_agent[:, idx] = self._camera_intrinsics_base
            self._dr_intrinsics_per_agent[:, idx, 0, 0] *= combined_scale
            self._dr_intrinsics_per_agent[:, idx, 1, 1] *= combined_scale
            camera_intrinsics[agent_id] = self._dr_intrinsics_per_agent[:, idx]
            image_shapes[agent_id] = self._camera_image_shape
            agent_poses[agent_id] = (root_pos, root_quat)

            # Update TiledCamera intrinsics based on effective focal multiplier
            if agent_id in self._cameras:
                self._cameras[agent_id].set_intrinsic_matrices_batched(
                    self._dr_intrinsics_per_agent[:, idx],
                    self.cfg.camera.spawn.focal_length * z_eff,
                )

        self.bbox_raycaster_v2.update(
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            target_poses=(target_pos, target_quat),
            agent_poses=agent_poses,
            image_shapes=image_shapes,
            target_scale=self._dr_target_scale,
        )

        # Apply calibrated detector noise, miss rate, and FP injection
        if self.cfg.calibrated_bbox_noise.enabled:
            self.bbox_raycaster_v2.apply_detector_replicator(
                noise_scale=self._curriculum_noise_scale,
                fp_fn_scale=self._curriculum_fp_fn_scale,
                sim_time=self._sim_time,
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
                # Reuse pre-allocated AgentStates buffer (all fields overwritten below)
                gt_states = self._delay_gt_buffers[agent_id]
                gt_data = gt_states.data

                # Body motion (copy_ keeps pre-allocated buffers bound)
                gt_data.body_position_w.copy_(self._root_pos_w[agent_id])
                gt_data.body_orientation_w.copy_(self._root_quat_w[agent_id])
                gt_data.body_linear_velocity_w.copy_(self._root_lin_vel_w[agent_id])
                gt_data.body_angular_velocity_w.copy_(self._root_ang_vel_b[agent_id])
                gt_data.body_angular_velocity_b.copy_(self._root_ang_vel_b[agent_id])
                gt_data.body_linear_acceleration_w.copy_(self._root_lin_acc_w[agent_id])
                gt_data.body_linear_acceleration_b.copy_(self._root_lin_acc_b[agent_id])

                # Joint states (gimbal) - order is pitch, yaw, roll
                gt_data.joint_positions_b.copy_(self._gimbal_joint_pos[agent_id])
                # Actual joint velocities from simulation (not from policy LOS rate commands)
                gt_data.joint_velocities_b.copy_(self._gimbal_joint_vel[agent_id])

                # Combined angular velocity (body + gimbal)
                combined_w, combined_b = compute_combined_angular_velocity(
                    body_angular_velocity_w=self._root_ang_vel_b[agent_id],  # Note: currently stored as body-frame
                    body_angular_velocity_b=self._root_ang_vel_b[agent_id],
                    joint_velocities_b=self._gimbal_joint_vel[agent_id],
                    body_orientation_w=self._root_quat_w[agent_id],
                    joint_positions_b=self._gimbal_joint_pos[agent_id],
                )
                gt_data.body_combined_angular_velocity_w.copy_(combined_w)
                gt_data.body_combined_angular_velocity_b.copy_(combined_b)

                # Gimbal world-frame angles (from joint positions + body orientation)
                az, el = body_to_world_gimbal_angles(
                    yaw_body=gt_data.joint_positions_b[:, 1] - YAW_JOINT_OFFSET,
                    pitch_body=gt_data.joint_positions_b[:, 0],
                    q_body=gt_data.body_orientation_w,
                )
                gt_data.gimbal_azimuth_world.copy_(az)
                gt_data.gimbal_elevation_world.copy_(el)

                # Gimbal world-frame LOS rates derived from combined angular velocity
                # (body rate + joint rates via kinematics, preserving independent noise sources)
                # For spherical coords: d/dt(az) = omega_z - tan(el)*(omega_x*cos(az) + omega_y*sin(az))
                #                        d/dt(el) = omega_x*sin(az) - omega_y*cos(az)
                omega = combined_w  # (N, 3) total camera angular velocity in world frame
                gt_data.gimbal_azimuth_rate.copy_(
                    omega[:, 2]
                    - torch.tan(el) * (omega[:, 0] * torch.cos(az) + omega[:, 1] * torch.sin(az))
                )
                gt_data.gimbal_elevation_rate.copy_(
                    omega[:, 0] * torch.sin(az) - omega[:, 1] * torch.cos(az)
                )
                self._gimbal_az_rate_world[agent_id] = gt_data.gimbal_azimuth_rate.clone()
                self._gimbal_el_rate_world[agent_id] = gt_data.gimbal_elevation_rate.clone()

                # Camera geometry
                cam_pos_w, cam_ori_w = camera_poses[agent_id]
                gt_data.camera_position_w.copy_(cam_pos_w)
                gt_data.camera_orientation_w.copy_(cam_ori_w)
                gt_data.camera_zoom_level.copy_(self.zoom_level[:, idx])
                # DR intrinsics: copy base then scale in-place on gt_data
                gt_data.camera_base_intrinsics.copy_(self._camera_intrinsics_base)
                dr_scale = self._dr_intrinsic_scale[:, idx]
                gt_data.camera_base_intrinsics[:, 0, 0] *= dr_scale
                gt_data.camera_base_intrinsics[:, 1, 1] *= dr_scale
                # effective_hfov = 2 * atan(image_width / (2 * fx_effective))
                # where fx_effective = fx_nominal * dr_scale * z_eff(zoom_cmd)
                img_w = self._camera_image_shape[1]
                z_eff_gt = compute_z_eff(self.zoom_level[:, idx])
                fx_eff = self._camera_intrinsics_base[:, 0, 0] * dr_scale * z_eff_gt
                gt_data.camera_effective_hfov.copy_(2.0 * torch.atan2(
                    torch.tensor(img_w * 0.5, device=self.device), fx_eff
                ))

                # Detection (bbox) - shape (N, T, 4)
                # CRITICAL: Use pixel bboxes, not normalized - triangulation expects pixels
                gt_data.bboxes_2d.copy_(self.bbox_raycaster_v2.data.bboxes[
                    :, idx, :, :
                ])

                # Camera ray directions (needed by GT observation path)
                gt_data.camera_ray_directions_w.copy_(compute_ray_directions_from_bbox(
                    camera_orientation_w=gt_data.camera_orientation_w,
                    camera_base_intrinsics=gt_data.camera_base_intrinsics,
                    camera_zoom_level=gt_data.camera_zoom_level,
                    bboxes_2d=gt_data.bboxes_2d,
                ))

                # Timestamps — sub-step jitter for continuous AoI
                gt_data.timestamp_sim_walltime.copy_(self._sim_time)
                if self.cfg.continuous_aoi_jitter:
                    step_dt = self.cfg.sim.dt * self.cfg.decimation
                    gt_data.timestamp_motion.copy_(self._sim_time + torch.empty_like(self._sim_time).uniform_(-step_dt, 0))
                    gt_data.timestamp_detection.copy_(self._sim_time + torch.empty_like(self._sim_time).uniform_(-step_dt, 0))
                else:
                    gt_data.timestamp_motion.copy_(self._sim_time)
                    gt_data.timestamp_detection.copy_(self._sim_time)

                # Pass replicated (noisy) bboxes if detector replicator is active
                replicated_bboxes = None
                if (
                    self.cfg.calibrated_bbox_noise.enabled
                    and self.bbox_raycaster_v2.data.bboxes_replicated is not None
                ):
                    replicated_bboxes = self.bbox_raycaster_v2.data.bboxes_replicated[
                        :, idx, :, :
                    ]

                # Update delay system
                self._delay_system.update_ground_truth(
                    agent_id, gt_states, replicated_bboxes=replicated_bboxes
                )

            # Cache GT states dict so _build_gt_states() can skip recomputation
            # Buffers remain valid until next _update_state_cache() call
            self._cached_gt_states = {
                agent_id: self._delay_gt_buffers[agent_id]
                for agent_id in self.cfg.possible_agents
            }

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
        # (camera_intrinsics values are views into _dr_intrinsics_per_agent; clone to decouple)
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            self._camera_intrinsics_cache[agent_id] = camera_intrinsics[agent_id].clone()

    def _compute_zoomed_intrinsics(
        self,
        base_intrinsics: torch.Tensor,
        zoom_level: torch.Tensor,
    ) -> torch.Tensor:
        """Compute camera intrinsics with measured zoom curve applied.

        Args:
            base_intrinsics: Base camera intrinsics [N, 3, 3]
            zoom_level: Operator zoom command [N] (1.0 = 1x).
                Internally mapped to z_eff via the SIYI zoom curve.

        Returns:
            Zoomed intrinsics [N, 3, 3] with fx, fy scaled by z_eff(zoom_level)
        """
        z_eff = compute_z_eff(zoom_level)
        intrinsics = base_intrinsics.clone()
        intrinsics[:, 0, 0] *= z_eff  # fx
        intrinsics[:, 1, 1] *= z_eff  # fy
        return intrinsics

    def _build_gt_states(self) -> Dict[AgentID, AgentStates]:
        """Build GT states dictionary.

        When the delay system is active, returns the cached GT states computed
        during _update_state_cache() to avoid redundant recomputation.
        Otherwise, builds from scratch using current simulation state.

        Returns:
            Dictionary mapping agent_id to AgentStates with current GT values.
        """
        if self._cached_gt_states is not None:
            return self._cached_gt_states

        gt_states = {}
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            # Reuse pre-allocated AgentStates buffer (zero and repopulate)
            states = self._gt_state_cache[agent_id]
            states.zero_()
            data = states.data

            # Body motion (copy_ keeps pre-allocated buffers bound)
            data.body_position_w.copy_(self._root_pos_w[agent_id])
            data.body_orientation_w.copy_(self._root_quat_w[agent_id])

            # Joint states
            data.joint_positions_b.copy_(self._gimbal_joint_pos[agent_id])

            # Gimbal world-frame angles
            az, el = body_to_world_gimbal_angles(
                yaw_body=data.joint_positions_b[:, 1] - YAW_JOINT_OFFSET,
                pitch_body=data.joint_positions_b[:, 0],
                q_body=data.body_orientation_w,
            )
            data.gimbal_azimuth_world.copy_(az)
            data.gimbal_elevation_world.copy_(el)

            # Gimbal world-frame rates (pre-computed via finite difference in _update_state_cache)
            data.gimbal_azimuth_rate.copy_(self._gimbal_az_rate_world[agent_id])
            data.gimbal_elevation_rate.copy_(self._gimbal_el_rate_world[agent_id])

            # Body velocity (needed for inter-agent obs in GT path)
            data.body_linear_velocity_w.copy_(self._root_lin_vel_w[agent_id])

            # Camera geometry
            robot = self._robots[agent_id]
            pitch_link_idx = self._frame_link_ids[agent_id]["pitch_link"]
            data.camera_position_w.copy_(robot.data.body_pos_w[:, pitch_link_idx])
            data.camera_orientation_w.copy_(robot.data.body_quat_w[:, pitch_link_idx])
            # DR intrinsics: copy base then scale in-place on data
            data.camera_base_intrinsics.copy_(self._camera_intrinsics_base)
            dr_scale = self._dr_intrinsic_scale[:, idx]
            data.camera_base_intrinsics[:, 0, 0] *= dr_scale
            data.camera_base_intrinsics[:, 1, 1] *= dr_scale
            data.camera_zoom_level.copy_(self.zoom_level[:, idx])
            # effective_hfov = 2 * atan(image_width / (2 * fx_effective))
            # fx_effective = fx_nominal * dr_scale * z_eff(zoom_cmd)
            img_w = self._camera_image_shape[1]
            z_eff_b = compute_z_eff(self.zoom_level[:, idx])
            fx_eff = self._camera_intrinsics_base[:, 0, 0] * dr_scale * z_eff_b
            data.camera_effective_hfov.copy_(2.0 * torch.atan2(
                torch.tensor(img_w * 0.5, device=self.device), fx_eff
            ))

            # Detection - use pixel bboxes (not normalized) for triangulation
            data.bboxes_2d.copy_(self.bbox_raycaster_v2.data.bboxes[:, idx, :, :])

            # Ray directions from bbox (camera-to-target through bbox center)
            data.camera_ray_directions_w.copy_(compute_ray_directions_from_bbox(
                camera_orientation_w=data.camera_orientation_w,
                camera_base_intrinsics=data.camera_base_intrinsics,
                camera_zoom_level=data.camera_zoom_level,
                bboxes_2d=data.bboxes_2d,
            ))

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

        # Per-camera intrinsic covariance from the measured per-zoom mrcal
        # std-devs (mas/029). Build only when the cfg requests it.
        Sigma_K_per_camera = None
        if self.cfg.triangulation.include_intrinsics_uncertainty:
            zoom_per_camera = torch.stack(
                [
                    states[agent_id].data.camera_zoom_level
                    for agent_id in self.cfg.possible_agents
                ],
                dim=1,
            )  # [N, C]
            Sigma_K_per_camera = compute_sigma_K_from_zoom(zoom_per_camera)

        obs_age_per_camera = torch.stack(
            [
                (self._sim_time - states[agent_id].data.timestamp_detection).clamp_min(0.0)
                for agent_id in self.cfg.possible_agents
            ],
            dim=1,
        )  # [N, C]
        vel_mag_per_camera = torch.stack(
            [
                torch.linalg.vector_norm(states[agent_id].data.body_linear_velocity_w, dim=-1)
                for agent_id in self.cfg.possible_agents
            ],
            dim=1,
        )  # [N, C]
        ang_vel_mag_per_camera = torch.stack(
            [
                torch.linalg.vector_norm(states[agent_id].data.body_combined_angular_velocity_w, dim=-1)
                for agent_id in self.cfg.possible_agents
            ],
            dim=1,
        )  # [N, C]
        pos_std_per_camera, ori_std_per_camera = compute_sigma_drift(
            obs_age_per_camera,
            vel_mag_per_camera,
            ang_vel_mag_per_camera,
            self.cfg.triangulation,
        )

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
            pos_std_per_camera=pos_std_per_camera,
            ori_std_per_camera=ori_std_per_camera,
            Sigma_K_per_camera=Sigma_K_per_camera,
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
        current_step = self.cfg.debug_initial_step if self.cfg.use_debug_initial_step else self.common_step_counter
        # current_step = self.common_step_counter
        curr = self.cfg.curriculum
        # Reward-gating curriculum signals (progress_coord, progress_safety) and
        # reset-only scalars (progress_tracking / progress_moving_target /
        # progress_agent_velocity) are computed in _reset_idx — see Phase A+D
        # of the dynamics-curriculum follow-up. Recomputing them per step here
        # is unnecessary: the per-env reward-gating values are held for the
        # episode (avoids GAE bootstrap bias), and the scalar reset-only
        # values are read by _reset_idx itself.

        # Update delay system curriculum (none → fixed → random, noise/dropout ramp).
        # Ticket 034 — Slice 4: pass per-(env, agent) effective-progress tensors
        # to the wrapper. The scalar mirrors (self.progress_delay,
        # self._curriculum_noise_scale, self._curriculum_fp_fn_scale) are kept
        # for logging back-compat — they reflect the mean across (env, agent).
        if self._delay_system is not None:
            delay_mode = curr.get_delay_mode(current_step)
            if delay_mode == "none":
                self.progress_delay = 0.0
                # Pass zero tensor so the wrapper's per-(env, agent) state
                # is cleared too (otherwise stale "fixed/random" values persist).
                self._delay_system.set_delay_mode(
                    "none", progress=torch.zeros_like(self._eff_progress_delay)
                )
            elif delay_mode == "fixed":
                self.progress_delay = float(self._eff_progress_delay.mean().item())
                self._delay_system.set_delay_mode(
                    "fixed", progress=self._eff_progress_delay
                )
            else:  # "random"
                self.progress_delay = float(self._eff_progress_delay.mean().item())
                self._delay_system.set_delay_mode(
                    "random", progress=self._eff_progress_delay
                )

            # Ramp noise (phase 2: 80k-100k)
            self._curriculum_noise_scale = float(self._eff_progress_noise.mean().item())
            self._delay_system.set_noise_scale(self._eff_progress_noise)

            # Ramp FP/FN (co-located with noise by default).
            # FP/FN axis is consumed downstream by the replicator (out of
            # ticket-034 scope); keep the scalar curriculum value here.
            self._curriculum_fp_fn_scale = curr.get_fp_fn_progress(current_step)

            # Ramp dropout (phase 5: 160k-180k)
            dropout_rate_per_env_agent = (
                self._eff_progress_dropout * self.cfg.delay_system_params.dropout_prob
            )
            self._delay_system.set_dropout_rate(dropout_rate_per_env_agent)

            # Ramp burst dropout (phase 6: 200k-220k)
            if self.cfg.delay_system_params.burst_dropout_enabled:
                p_onset_per_env_agent = (
                    self._eff_progress_burst_dropout
                    * self.cfg.delay_system_params.burst_p_onset
                )
                self._delay_system.set_burst_params(
                    p_onset=p_onset_per_env_agent,
                    p_recovery=self.cfg.delay_system_params.burst_p_recovery,
                )

        # mas/035: ramp gimbal rate-loop τ during the dynamics phase (180k-220k).
        # progress=0 → pass-through (instant gimbal); progress=1 → measured τ.
        # Ticket 034 — Slice 2: pass per-(env, agent) effective progress
        # flattened to the batched-controller layout
        # (agent_a → rows [a*N, (a+1)*N); see _batch_idx). `.t().reshape(-1)`
        # produces [agent_0_env_0, ..., agent_0_env_N-1, agent_1_env_0, ...].
        # Ticket 037 Slice 7: read from gimbal_rate_tau axis (falls back to
        # `_eff_progress_dynamics` when `enable_axis_independence=False`).
        self._controller.gimbal_rate_loop.set_progress(
            self._axis_eff_progress("gimbal_rate_tau").t().reshape(-1)
        )

        # mas/036: ramp gimbal command-to-first-move dead-time scale.
        # 0 → no delay at the rate-loop input; 1 → full measured Gaussian
        # sampled per env at episode reset. Independent of τ ramp above.
        # Ticket 034 — Slice 2: per-(env, agent) effective progress (same
        # flatten as above).
        self._controller.gimbal_rate_loop.set_dead_time_curriculum_scale(
            self._eff_progress_gimbal_dead_time.t().reshape(-1)
        )

        # mas/037: ramp zoom command-to-first-move dead-time scale (siyi_a8
        # mode only — no-op when cfg.zoom.model == "first_order"). Mirrors
        # mas/036 hook above; per-env τ_d sample takes effect on next reset.
        # Ticket 034 — Slice 3: pass per-(env, agent) effective progress.
        self._controller.zoom_controller.set_dead_time_curriculum_scale(
            self._eff_progress_zoom_dead_time.t().reshape(-1)
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
        per_agent_nonempty = torch.zeros(
            self.num_envs, self.cfg.num_agents, dtype=torch.bool, device=self.device
        )
        for agent_idx, agent_id in enumerate(self.cfg.possible_agents):
            bbox_raw = reward_states[agent_id].data.bboxes_2d[:, 0, :]  # [N, 4]
            agent_valid = bbox_raw.abs().sum(dim=-1) > 1e-6  # [N]
            per_agent_nonempty[:, agent_idx] = agent_valid
            num_valid_detections += agent_valid.float()
        self._num_valid_detections = num_valid_detections
        # Cache for get_aux_supervision() — read by the trainer wrapper after env.step().
        self._per_agent_bbox_nonempty = per_agent_nonempty
        self._detection_stats["total_steps"] += 1.0
        self._detection_stats["invalid_all_agents"] += (num_valid_detections == 0).float()
        self._detection_stats["pair_valid_count"] += (num_valid_detections >= 2).float()

        # Compute triangulation for all active task reward levels
        # (reward-only; obs tail is gated separately by cfg.enable_triangulation)
        task_level = self.cfg.task_reward_level

        # Level 1 (FIM): covariance at GT target position
        self._triangulation_result_gt = self._compute_triangulation(
            states=reward_states, use_gt_target=True
        )

        # Cache aux-head supervision for the policy tri head (ticket 031). These
        # are NOT cleared in _reset_idx, so get_aux_supervision() — called externally
        # after env.step() returns — always sees the data from the most recent
        # _get_rewards, even if some envs reset in this same step.
        #
        # Source of truth for the GT target world position is `self._target_pos_w`
        # (snapshot of `self.target.data.root_pos_w`, updated each step in
        # `_update_state_cache`). The `_triangulation_result_gt.position` field is
        # the *triangulated estimate* (X_tri), which is NaN when the geometric
        # solve is invalid — the wrong source for supervising the head.
        gt_pos_target = self._target_pos_w                                             # (N, 3)
        scene_valid = self._triangulation_result_gt.is_valid[:, 0]                     # (N,)
        self._aux_target_position_w_cache = (
            gt_pos_target.unsqueeze(1)
            .expand(-1, self.cfg.num_agents, -1)
            .contiguous()
            .to(dtype=torch.float32)
        )                                                                              # (N, A, 3)
        self._aux_target_valid_cache = (
            self._per_agent_bbox_nonempty & scene_valid.unsqueeze(-1)
        )                                                                              # (N, A) bool

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
            action_delta_per_dim = torch.square(
                self.action_delta_weight
                * (self._actions[agent_id] - self._last_actions[agent_id])
            )  # [N, 7]
            action_delta = torch.sum(action_delta_per_dim, dim=1)

            # Accumulate per-dimension squared deltas for RMS logging
            self._action_delta_sq_acc[agent_id] += action_delta_per_dim

            # ---- BBox rewards (using delayed state without noise) ----
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
            if self._triangulation_result_gt is not None:
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

            # Altitude penalty: continuous penalty proportional to how far below threshold
            pos_z = self._root_pos_w[agent_id][:, 2]
            altitude_deficit = torch.clamp(self.cfg.altitude_min_threshold - pos_z, min=0.0)

            # Target proximity penalty: continuous penalty for flying too close to target
            dist_to_target = torch.norm(
                self._root_pos_w[agent_id][:, :3] - self._target_pos_w[:, :3], dim=-1
            )
            target_proximity_deficit = torch.clamp(
                self.cfg.target_proximity_threshold - dist_to_target, min=0.0
            )

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
                "altitude": altitude_deficit * self.cfg.altitude_penalty_scale * step_dt,
                "target_proximity": (
                    target_proximity_deficit
                    * self.cfg.target_proximity_penalty_scale
                    * step_dt
                    * self.progress_safety
                ),
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

        # Increment smoothness step counter (once per step, outside agent loop)
        self._action_smoothness_steps += 1.0

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

    def get_aux_supervision(self) -> Dict[str, torch.Tensor]:
        """Per-agent supervision for the policy triangulation head (ticket 031).

        Pure READ on env state cached during ``_get_rewards``. Safe to call
        multiple times within a sim step. Returns the GT target world position
        broadcast across agents, plus a per-agent validity mask combining
        the scene-level FIM ``is_valid`` flag with per-agent bbox-non-empty.

        Reads from ``_aux_target_position_w_cache`` /
        ``_aux_target_valid_cache``, which are populated by ``_get_rewards``
        and **intentionally not cleared by ``_reset_idx``** so that callers
        invoking this method after ``env.step()`` returns always see the
        most recent supervision — even on steps where some envs reset.

        Returns:
            ``{
                "tri_target_position_w": [num_envs, num_agents, 3] float32,
                "tri_target_valid":      [num_envs, num_agents]     bool,
            }``

        If called before the first ``_get_rewards`` (e.g. during reset), both
        tensors are zero-filled and ``tri_target_valid`` is all False.
        """
        N = self.num_envs
        A = self.cfg.num_agents
        device = self.device

        if self._aux_target_position_w_cache is None or self._aux_target_valid_cache is None:
            return {
                "tri_target_position_w": torch.zeros(N, A, 3, dtype=torch.float32, device=device),
                "tri_target_valid": torch.zeros(N, A, dtype=torch.bool, device=device),
            }

        return {
            "tri_target_position_w": self._aux_target_position_w_cache,
            "tri_target_valid": self._aux_target_valid_cache,
        }

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents.

        When delay system is enabled, observations use delayed states.
        Otherwise, uses ground truth states directly.

        Observation structure per agent:
        - Ego (31D): pos(3), vel(3), rpy(3), ang_vel_b(3), lin_acc_b(3),
          gimbal_yaw_body(1), gimbal_pitch_body(1), ray_w(3), combined_ang_vel_w(3),
          bbox_aoi(1), zoom(1), effective_hfov(1), bbox(4), bbox_empty(1)
        - Inter-agent (16D per other): pos(3), vel(3), ray_w(3), combined_ang_vel_w(3),
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
                # Ego camera ray direction in world frame (bbox-based, from camera to target)
                ego_ray_w = ego_data.camera_ray_directions_w[:, 0, :]

                # Ego bbox age-of-information
                ego_bbox_aoi = (self._sim_time - ego_data.timestamp_detection).unsqueeze(-1)

                # Build ego observation (31D)
                ego_obs = torch.cat(
                    [
                        ego_data.body_position_w,  # (N, 3)
                        ego_data.body_linear_velocity_w,  # (N, 3)
                        wrap_to_pi(torch.stack(euler_xyz_from_quat(ego_data.body_orientation_w), dim=-1)),  # (N, 3) roll, pitch, yaw in [-pi, pi]
                        ego_data.body_angular_velocity_b,  # (N, 3) — gyro
                        ego_data.body_linear_acceleration_b,  # (N, 3) — accelerometer
                        ego_gimbal_yaw_body,  # (N, 1) — body-frame, 0=forward
                        ego_gimbal_pitch_body,  # (N, 1) — body-frame
                        ego_ray_w,  # (N, 3) — world-frame ray to target through bbox center
                        ego_data.body_combined_angular_velocity_w,  # (N, 3) — camera sweep rate
                        ego_bbox_aoi,  # (N, 1) — bbox age-of-information
                        ego_data.camera_zoom_level.unsqueeze(-1),  # (N, 1)
                        ego_data.camera_effective_hfov.unsqueeze(-1),  # (N, 1) — effective HFOV (rad)
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

                    # Other agent's camera ray direction in world frame (bbox-based)
                    other_ray_w = other_data.camera_ray_directions_w[:, 0, :]

                    other_obs_parts.append(
                        torch.cat(
                            [
                                other_data.body_position_w,  # (N, 3)
                                other_data.body_linear_velocity_w,  # (N, 3)
                                other_ray_w,  # (N, 3) — world-frame ray to target
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
                # Ego camera ray direction in world frame (bbox-based, from camera to target)
                ego_ray_w = ego_gt.camera_ray_directions_w[:, 0, :]

                # Ego observation (31D) — GT path: bbox AoI = 0 (no delay)
                ego_obs = torch.cat(
                    [
                        self._root_pos_w[agent_id],  # (N, 3)
                        self._root_lin_vel_w[agent_id],  # (N, 3)
                        wrap_to_pi(torch.stack(euler_xyz_from_quat(self._root_quat_w[agent_id]), dim=-1)),  # (N, 3) roll, pitch, yaw in [-pi, pi]
                        self._root_ang_vel_b[agent_id],  # (N, 3) — gyro
                        self._root_lin_acc_b[agent_id],  # (N, 3) — accelerometer
                        ego_gimbal_yaw_body,  # (N, 1) — body-frame, 0=forward
                        ego_gimbal_pitch_body,  # (N, 1) — body-frame
                        ego_ray_w,  # (N, 3) — world-frame ray to target through bbox center
                        ego_gt.body_combined_angular_velocity_w,  # (N, 3) — camera sweep rate
                        torch.zeros(self.num_envs, 1, device=self.device),  # (N, 1) — bbox AoI (GT = 0)
                        self.zoom_level[:, idx : idx + 1],  # (N, 1)
                        ego_gt.camera_effective_hfov.unsqueeze(-1),  # (N, 1) — effective HFOV (rad)
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


                    # Other agent's camera ray direction in world frame (bbox-based)
                    other_ray_w = other_gt.camera_ray_directions_w[:, 0, :]

                    other_obs_parts.append(
                        torch.cat(
                            [
                                other_gt.body_position_w,  # (N, 3)
                                other_gt.body_linear_velocity_w,  # (N, 3)
                                other_ray_w,  # (N, 3) — world-frame ray to target
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

    def _get_states(self) -> torch.Tensor:
        """Build the centralized critic state for the asymmetric MAPPO critic.

        Called by the parent ``DirectMARLEnv.state()`` whenever
        ``cfg.state_space > 0`` (set in cfg.__post_init__ when any
        ``enable_critic_*`` flag is True). The skrl MAPPO trainer invokes
        ``self.env.state()`` once before each step and once after, and
        injects the result into ``infos["shared_states"]`` /
        ``infos["shared_next_states"]`` before ``record_transition`` — so
        time-alignment is automatic.

        State layout (sized to match cfg.state_space):
            concat( actor_obs[a] for a in agents )                              # actor concat
              ++  concat( [zoom_internal[a], zoom_target[a]]                    # zoom tail (per-agent)
                           for a in agents )                                    # if enable_critic_continuous_zoom
              ++  target_pos_w (3)                                              # GT target position (global)
                                                                                # if enable_critic_gt_target
              ++  target_lin_vel_w (3)                                          # GT target velocity (global)
                                                                                # if enable_critic_gt_target_velocity

        The actor's per-agent obs is unchanged (still sees quantized
        published zoom that the deployed hardware will expose, and only
        delayed/noisy bbox-derived target signals). The critic sees the
        concat-of-actor-obs plus the privileged tails enabled — sharpening
        V estimates and temporal credit assignment without affecting what
        the actor (and therefore the deployed policy) can use.
        """
        cfg_zoom = getattr(self.cfg, "enable_critic_continuous_zoom", False)
        cfg_target = getattr(self.cfg, "enable_critic_gt_target", False)
        cfg_target_vel = getattr(self.cfg, "enable_critic_gt_target_velocity", False)
        # Ticket 037 — privileged env-param tail.
        cfg_priv = bool(getattr(self.cfg, "critic_privileged_fields", []))

        if not (cfg_zoom or cfg_target or cfg_priv):
            # Fallback: same as DirectMARLEnv's auto-concat path.
            return torch.cat(
                [self.obs_dict[a].reshape(self.num_envs, -1) for a in self.cfg.possible_agents],
                dim=-1,
            )

        N = self.num_envs
        parts: List[torch.Tensor] = [
            torch.cat([self.obs_dict[a] for a in self.cfg.possible_agents], dim=-1),
        ]

        # Privileged zoom tail (per-agent: zoom_internal + zoom_target)
        if cfg_zoom:
            # Per-agent batched controller layout is agent-major:
            # ``zoom_internal[a*N : (a+1)*N]`` is agent ``a``'s slice. Same
            # convention as the scatter at iris_ma_env6_test.py L809:
            # ``self.zoom_level[:, idx] = zoom_batch[idx*N : (idx+1)*N]``.
            z_int = self._controller._zoom.zoom_internal       # (N*A,)
            z_tgt = self._controller._zoom._zoom_target        # (N*A,)
            zoom_parts: List[torch.Tensor] = []
            for a_idx in range(len(self.cfg.possible_agents)):
                s, e = a_idx * N, (a_idx + 1) * N
                zoom_parts.append(z_int[s:e].unsqueeze(-1))    # (N, 1)
                zoom_parts.append(z_tgt[s:e].unsqueeze(-1))    # (N, 1)
            parts.append(torch.cat(zoom_parts, dim=-1))        # (N, 2*A)

        # Privileged GT target tail (global; one target in v0)
        if cfg_target:
            # Same source the aux-head supervision uses (snapshot of
            # self.target.data.root_pos_w in _update_state_cache).
            parts.append(self._target_pos_w.float())            # (N, 3)
            if cfg_target_vel:
                parts.append(self.target.data.root_lin_vel_w.float())  # (N, 3)

        # Ticket 037 — privileged env-param tail (cfg-driven).
        # Per-agent fields are agent-major: all of a0's K_pa values
        # first, then a1's, etc. Shared fields come last.
        if cfg_priv:
            parts.append(self._get_critic_privileged_obs())     # (N, K_priv)

        return torch.cat(parts, dim=-1)

    def _get_critic_privileged_obs(self) -> torch.Tensor:
        """Ticket 037 — build the privileged env-param tail for the centralized critic.

        Returns shape ``(num_envs, K)`` where
        ``K = per_agent_dim × num_agents + shared_dim``.

        Per-agent fields are concatenated agent-major (a0's K_pa fields first,
        then a1's, etc.) matching the existing zoom-tail convention; shared
        fields (one value per env) come after. Each field is pre-normalized
        to ``[-1, +1]`` using its cfg-known native range (linear map), so the
        critic sees a bounded, well-conditioned input from training step 0
        without depending on ``RunningStandardScaler`` convergence.

        Setting ``cfg.critic_privileged_fields = []`` (default) bypasses this
        method entirely via the guard in :meth:`_get_states`, so behavior is
        bit-exact t034 in that case. See doc/critic_obs_design.md §3 for
        field semantics and the storage map at §3.4.

        Called from :meth:`_get_states` once per env.state() invocation; the
        skrl MAPPO trainer reads it before and after each step, so it must
        be cheap (no allocations beyond the output concat).
        """
        cfg = self.cfg
        N = self.num_envs
        A = len(cfg.possible_agents)

        def _norm(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
            """Linearly map ``[low, high]`` → ``[-1, +1]`` per-element.

            Degenerate range (``|high - low| ≈ 0``, e.g. randomization disabled
            for that axis) returns zeros — the critic sees "no signal" rather
            than NaN or div-by-zero.
            """
            span = high - low
            if abs(span) < 1e-12:
                return torch.zeros_like(x)
            return 2.0 * (x - low) / span - 1.0

        def _batched_to_na(x: torch.Tensor) -> torch.Tensor:
            """Batched-controller layout ``(N*A,)`` → ``(N, A)`` agent-major.

            The batched controllers use the convention agent_a → rows
            ``[a*N, (a+1)*N)`` (see :meth:`_batch_idx`). ``.reshape(A, N).t()``
            inverts that to ``(N, A)``.
            """
            return x.reshape(A, N).t().contiguous()

        def _read(name: str) -> torch.Tensor:
            """Read one field, return pre-normalized tensor (shape (N, A) for
            per_agent, (N,) for shared)."""
            # --- Drone controller gain scales (current/nominal, axis-uniform) ---
            if name == "vel_gain_scale":
                return _norm(
                    _batched_to_na(self._controller.get_vel_gain_scale()),
                    *cfg.gain_randomization.scale_range,
                )
            if name == "att_gain_scale":
                return _norm(
                    _batched_to_na(self._controller.get_att_gain_scale()),
                    *cfg.gain_randomization.scale_range,
                )
            if name == "rate_gain_scale":
                return _norm(
                    _batched_to_na(self._controller.get_rate_gain_scale()),
                    *cfg.gain_randomization.scale_range,
                )
            if name == "motor_gain_scale":
                return _norm(
                    _batched_to_na(self._controller.get_motor_gain_scale()),
                    *cfg.gain_randomization.scale_range,
                )
            if name == "zoom_tau_scale":
                return _norm(
                    _batched_to_na(self._controller.get_zoom_tau_scale()),
                    *cfg.gain_randomization.zoom_scale_range,
                )
            if name == "zoom_max_rate_scale":
                return _norm(
                    _batched_to_na(self._controller.get_zoom_max_rate_scale()),
                    *cfg.gain_randomization.scale_range,
                )

            # --- Drone dynamics ---
            if name == "max_lin_vel_per_env":
                # Currently stored per-env at iris_ma_env6_test.py:2632
                # (mean-collapsed across agents); broadcast to per-agent for
                # the critic obs. Ticket 038 will extend _max_lin_vel to
                # (N, A) — this code path then yields genuine per-agent signal
                # automatically (the read becomes self._max_lin_vel directly
                # without unsqueeze/expand).
                #
                # Actual range under full curriculum:
                #   curriculum: _max_lin_vel ∈ [max_lin_vel_min, max_lin_vel]
                #   × randomize scale ∈ [max_lin_vel_scale_range[0],
                #                        max_lin_vel_scale_range[1]]
                # So the bound is [min × scale_lo, max × scale_hi].
                per_env = self._max_lin_vel  # (N,)
                scale_lo, scale_hi = cfg.gain_randomization.max_lin_vel_scale_range
                low = cfg.max_lin_vel_min * scale_lo
                high = cfg.max_lin_vel * scale_hi
                normed = _norm(per_env, low, high)
                return normed.unsqueeze(-1).expand(N, A).contiguous()

            # --- Gimbal rate-loop effective τ ---
            if name == "gimbal_rate_tau_yaw_s":
                grl_cfg = cfg.drone_controller.gimbal_rate_loop
                return _norm(
                    _batched_to_na(self._controller.gimbal_rate_loop.tau_yaw_effective),
                    0.0,
                    grl_cfg.tau_yaw_s * grl_cfg.tau_scale_range_yaw[1],
                )
            if name == "gimbal_rate_tau_pitch_s":
                grl_cfg = cfg.drone_controller.gimbal_rate_loop
                return _norm(
                    _batched_to_na(self._controller.gimbal_rate_loop.tau_pitch_effective),
                    0.0,
                    grl_cfg.tau_pitch_s * grl_cfg.tau_scale_range_pitch[1],
                )

            # --- Dead times ---
            if name == "gimbal_dead_time_s":
                return _norm(
                    _batched_to_na(self._controller.gimbal_rate_loop.dead_time_seconds),
                    0.0,
                    cfg.drone_controller.gimbal_rate_loop.dead_time_max_s,
                )
            if name == "zoom_dead_time_s":
                return _norm(
                    _batched_to_na(self._controller._zoom.dead_time_seconds),
                    0.0,
                    cfg.drone_controller.zoom.dead_time_max_s,
                )

            # --- Camera FOV ---
            if name == "fov_scale":
                fov_raw = self._domain_randomizer.get_fov_scales()  # (N, A)
                return _norm(fov_raw, *cfg.domain_randomization.camera.fov_scale_range)

            # --- Gimbal mechanical offsets ---
            if name == "gimbal_yaw_offset":
                return _norm(
                    self._domain_randomizer.gimbal_randomizer.get_yaw_offsets(),
                    *cfg.domain_randomization.gimbal.yaw_offset_range,
                )
            if name == "gimbal_pitch_offset":
                return _norm(
                    self._domain_randomizer.gimbal_randomizer.get_pitch_offsets(),
                    *cfg.domain_randomization.gimbal.pitch_offset_range,
                )
            if name == "gimbal_roll_offset":
                return _norm(
                    self._domain_randomizer.gimbal_randomizer.get_roll_offsets(),
                    *cfg.domain_randomization.gimbal.roll_offset_range,
                )

            # --- Gimbal joint dynamics ---
            if name == "gimbal_stiffness_scale":
                return _norm(
                    self._domain_randomizer.gimbal_randomizer.get_stiffness_scales(),
                    *cfg.domain_randomization.gimbal.stiffness_scale_range,
                )
            if name == "gimbal_damping_scale":
                return _norm(
                    self._domain_randomizer.gimbal_randomizer.get_damping_scales(),
                    *cfg.domain_randomization.gimbal.damping_scale_range,
                )

            # --- Robot body mass (per-env, broadcast to agents) ---
            if name == "body_mass_scale_or_add":
                per_env = self._domain_randomizer.physics_randomizer.get_mass_scales()  # (N,)
                normed = _norm(
                    per_env,
                    *cfg.domain_randomization.physics.mass.body_mass_scale_range,
                )
                return normed.unsqueeze(-1).expand(N, A).contiguous()

            # --- Detection pipeline (env-derived: _eff_progress_* × cfg base) ---
            if name == "detection_latency_s":
                base = cfg.delay_system_params.ego_detection_latency_mean
                return _norm(self._eff_progress_delay * base, 0.0, base)
            if name == "bbox_noise_std":
                base = cfg.delay_system_params.noise_bbox_std
                return _norm(self._eff_progress_noise * base, 0.0, base)
            if name == "detection_dropout_prob":
                base = cfg.delay_system_params.dropout_prob
                return _norm(self._eff_progress_dropout * base, 0.0, base)
            if name == "burst_p_onset":
                base = cfg.delay_system_params.burst_p_onset
                return _norm(self._eff_progress_burst_dropout * base, 0.0, base)

            # --- Target physics (shared per env) ---
            if name == "target_scale_xy":
                return _norm(self._dr_target_scale[:, 0, 0], *cfg.target_xy_scale_range)
            if name == "target_scale_z":
                return _norm(self._dr_target_scale[:, 0, 2], *cfg.target_z_scale_range)

            raise ValueError(f"unknown critic_privileged_field: {name!r}")

        # Split fields by scope, preserving the user's declared order
        # within each group.
        per_agent_fields = [
            f for f in cfg.critic_privileged_fields
            if _CRITIC_PRIVILEGED_FIELD_REGISTRY[f]["scope"] == "per_agent"
        ]
        shared_fields = [
            f for f in cfg.critic_privileged_fields
            if _CRITIC_PRIVILEGED_FIELD_REGISTRY[f]["scope"] == "shared"
        ]

        parts: List[torch.Tensor] = []
        if per_agent_fields:
            # Stack per-agent reads → (N, A, K_pa).
            pa_stack = torch.stack([_read(f) for f in per_agent_fields], dim=-1)
            # Agent-major flatten: a0's K_pa first, then a1's, …
            parts.append(pa_stack.reshape(N, A * len(per_agent_fields)))
        if shared_fields:
            parts.append(torch.stack([_read(f) for f in shared_fields], dim=-1))

        return torch.cat(parts, dim=-1) if parts else torch.empty(N, 0, device=self.device)

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get termination and truncation flags for all agents.

        Termination conditions:
        - Crashed: drone goes below z=0.5m

        Truncation conditions:
        - Timeout: episode_length >= max_episode_length
        - Tracking lost: all agents blind for tracking_lost_timeout_s (consecutive)

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

        # Tracking-lost truncation with debounced reset.
        # Requires tracking_reacquire_steps consecutive valid frames to reset
        # the lost counter, preventing noise-induced single valid frames from
        # defeating the timeout.
        tracking_lost = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.enable_tracking_truncation:
            all_blind = self._num_valid_detections == 0
            in_grace = self.episode_length_buf < self.cfg.tracking_truncation_grace_steps

            # Reacquire counter: counts consecutive valid frames; resets on blind/grace
            self._detection_reacquire_counter = torch.where(
                all_blind | in_grace,
                torch.zeros_like(self._detection_reacquire_counter),
                self._detection_reacquire_counter + 1,
            )

            # Lost counter: increments on blind, holds on sporadic valid frames,
            # resets only after sustained reacquisition
            reacquired = self._detection_reacquire_counter >= self.cfg.tracking_reacquire_steps
            self._all_lost_counter = torch.where(
                in_grace,
                torch.zeros_like(self._all_lost_counter),
                torch.where(
                    all_blind,
                    self._all_lost_counter + 1,
                    torch.where(
                        reacquired,
                        torch.zeros_like(self._all_lost_counter),
                        self._all_lost_counter,
                    ),
                ),
            )

            tracking_lost = self._all_lost_counter >= self._tracking_lost_timeout_steps
            self._tracking_lost_count += tracking_lost.float()

        # Any termination/truncation applies to all agents (cooperative MARL)
        any_crashed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent_id in self.cfg.possible_agents:
            pos_z = self._root_pos_w[agent_id][:, 2]
            any_crashed |= pos_z < 0.5

        for agent_id in self.cfg.possible_agents:
            terminated[agent_id] = any_crashed
            truncated[agent_id] = (time_out | tracking_lost) & ~terminated[agent_id]

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at specified indices.

        Uses InitialStates module for curriculum-driven randomization if enabled,
        otherwise falls back to hardcoded triangle formation.

        Args:
            env_ids: Environment indices to reset.
        """
        super()._reset_idx(env_ids)

        # Invalidate GT state cache — stale until next _update_state_cache()
        self._cached_gt_states = None

        num_reset = len(env_ids)

        current_step = self.cfg.debug_initial_step if self.cfg.use_debug_initial_step else self.common_step_counter
        self.progress_dynamics = self._linear_progress(
            self.cfg.curriculum.dynamics_start_step,
            self.cfg.curriculum.dynamics_end_step,
            current_step,
        )
        # mas/037: zoom τ₁ ramp is decoupled from gimbal `dynamics_*` because
        # zoom and gimbal have different timescales and the policy benefits
        # from a slower zoom-τ ramp specifically. Bootstrap (< zoom_tau_start)
        # keeps τ₁ near-instant; full ramp at zoom_tau_end.
        self.progress_zoom_tau = self._linear_progress(
            self.cfg.curriculum.zoom_tau_start_step,
            self.cfg.curriculum.zoom_tau_end_step,
            current_step,
        )

        # Ticket 037 Slice 7 — per-axis dynamics progresses. Each method falls
        # back to dynamics_* if the per-axis (start, end) is None (default),
        # so behavior is bit-exact to pre-decoupling when no per-axis cfg is
        # set. When enable_axis_independence=True AND per-axis cfgs are set,
        # each axis ramps on its own schedule.
        curr = self.cfg.curriculum
        self.progress_gimbal_rate_tau   = curr.get_gimbal_rate_tau_progress(current_step)
        self.progress_drone_gains       = curr.get_drone_gains_progress(current_step)
        self.progress_max_lin_vel_scale = curr.get_max_lin_vel_scale_progress(current_step)
        self.progress_camera_fov        = curr.get_camera_fov_progress(current_step)
        self.progress_gimbal_mech_offsets = curr.get_gimbal_mech_offsets_progress(current_step)
        self.progress_mass_inertia      = curr.get_mass_inertia_progress(current_step)
        self.progress_gimbal_stiff_damp = curr.get_gimbal_stiff_damp_progress(current_step)
        self.progress_target_scale      = curr.get_target_scale_progress(current_step)

        # Step-at-episode-boundary: triangulation reward turns on when the env
        # starts an episode at or after coordination_start_step. Held constant
        # for the episode (no mid-episode reward-function change). The legacy
        # coordination_end_step is intentionally ignored — set start==end if
        # you want a strict step semantic, or revert to ramping by reading
        # _linear_progress here instead.
        coord_active = float(current_step >= self.cfg.curriculum.coordination_start_step)
        self.progress_coord[env_ids] = coord_active

        # Phase A — per-env reward-gating ramp (safety → cbf_penalty +
        # target_proximity). Same rationale as progress_coord: hold the
        # value constant within an episode so GAE bootstrap targets are
        # unbiased.
        self.progress_safety[env_ids] = self._linear_progress(
            self.cfg.curriculum.safety_start_step,
            self.cfg.curriculum.safety_end_step,
            current_step,
        )

        # Phase D — reset-only scalar curricula (only consumed in _reset_idx).
        # Computed here instead of _get_rewards because the consumers are
        # right below.
        curr = self.cfg.curriculum
        self.progress_tracking = curr.get_progress(
            current_step, curr.tracking_start_step, curr.tracking_end_step
        )
        self.progress_moving_target = curr.get_progress(
            current_step, curr.moving_target_start_step, curr.moving_target_end_step
        )
        self.progress_agent_velocity = curr.get_agent_velocity_progress(current_step)

        # Ticket 034 — Slice 1: per-(env, agent) effective agent_velocity progress.
        # Each (env, agent) draws eff_p ~ Uniform(0, progress_agent_velocity) so the
        # env distribution always retains positive mass on the slow regime, even at
        # progress_agent_velocity = 1.
        self._eff_progress_agent_velocity[env_ids] = sample_per_env_progress(
            global_progress=self.progress_agent_velocity,
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
            env_ids=env_ids,
            generator=self._curriculum_generator,
        )

        # Ticket 034 — Slice 2: gimbal rate-loop τ (progress_dynamics) and
        # gimbal dead-time scale. Same anti-forgetting sampler — at full
        # progress_dynamics the env distribution still contains near-zero
        # gimbal-τ (pass-through) and near-zero dead-time envs.
        self._eff_progress_dynamics[env_ids] = sample_per_env_progress(
            global_progress=self.progress_dynamics,
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            env_ids=env_ids,
            device=self.device,
            generator=self._curriculum_generator,
        )

        # Ticket 037 Slice 7 — per-axis dynamics sampling. Each draws its
        # own (env, agent) sample from Uniform(0, progress_axis), where
        # progress_axis falls back to progress_dynamics if the per-axis
        # curriculum (start, end) is unset. Result: with all per-axis
        # cfgs at default (None), these 8 tensors equal `_eff_progress_dynamics`
        # bit-exactly under the same `_curriculum_generator` RNG state...
        # NOT: the generator advances per call, so successive draws diverge.
        # For bit-exact t034 reproduction, set `enable_axis_independence=False`
        # which routes all consumers back to `_eff_progress_dynamics` and
        # the 8 below remain unused (zero-init). See _read_axis_eff_progress
        # for the dispatch.
        def _sample_per_axis(global_p: float) -> torch.Tensor:
            return sample_per_env_progress(
                global_progress=global_p,
                num_envs=self.num_envs,
                num_agents=len(self.cfg.possible_agents),
                env_ids=env_ids,
                device=self.device,
                generator=self._curriculum_generator,
            )

        if self.cfg.enable_axis_independence:
            self._eff_progress_gimbal_rate_tau[env_ids]     = _sample_per_axis(self.progress_gimbal_rate_tau)
            self._eff_progress_drone_gains[env_ids]         = _sample_per_axis(self.progress_drone_gains)
            self._eff_progress_max_lin_vel_scale[env_ids]   = _sample_per_axis(self.progress_max_lin_vel_scale)
            self._eff_progress_camera_fov[env_ids]          = _sample_per_axis(self.progress_camera_fov)
            self._eff_progress_gimbal_mech_offsets[env_ids] = _sample_per_axis(self.progress_gimbal_mech_offsets)
            self._eff_progress_mass_inertia[env_ids]        = _sample_per_axis(self.progress_mass_inertia)
            self._eff_progress_gimbal_stiff_damp[env_ids]   = _sample_per_axis(self.progress_gimbal_stiff_damp)
            self._eff_progress_target_scale[env_ids]        = _sample_per_axis(self.progress_target_scale)
        # When enable_axis_independence=False, the consumer sites read from
        # `_eff_progress_dynamics` directly via _read_axis_eff_progress;
        # the 8 per-axis tensors remain at their zero-init values and the
        # generator RNG state is preserved (no extra draws).
        self._eff_progress_gimbal_dead_time[env_ids] = sample_per_env_progress(
            global_progress=self.cfg.curriculum.get_gimbal_dead_time_progress(current_step),
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
            env_ids=env_ids,
            generator=self._curriculum_generator,
        )

        # Slice 3: zoom dead-time per-(env, agent) effective progress.
        self._eff_progress_zoom_dead_time[env_ids] = sample_per_env_progress(
            global_progress=self.cfg.curriculum.get_zoom_dead_time_progress(current_step),
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
            env_ids=env_ids,
            generator=self._curriculum_generator,
        )

        # Slice 4: observability axes (noise, dropout, delay mode, burst).
        # Each draws eff_p ~ Uniform(0, axis_progress) per-(env, agent) so
        # the env distribution always contains clean-obs envs alongside
        # high-corruption envs.
        self._eff_progress_noise[env_ids] = sample_per_env_progress(
            global_progress=curr.get_noise_progress(current_step),
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
            env_ids=env_ids,
            generator=self._curriculum_generator,
        )
        self._eff_progress_dropout[env_ids] = sample_per_env_progress(
            global_progress=curr.get_dropout_progress(current_step),
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
            env_ids=env_ids,
            generator=self._curriculum_generator,
        )
        # Delay-mode progress depends on which mode the curriculum is in;
        # _get_rewards reads `curr.get_delay_mode(current_step)` and picks
        # the corresponding progress getter. For the reset-time per-(env, agent)
        # sample we use the same axis: fixed-delay progress drives the
        # `fixed` phase, random-delay progress drives the `random` phase,
        # and `none` produces zeros.
        delay_mode = curr.get_delay_mode(current_step)
        if delay_mode == "none":
            global_progress_delay = 0.0
        elif delay_mode == "fixed":
            global_progress_delay = curr.get_fixed_delay_progress(current_step)
        else:
            global_progress_delay = curr.get_random_delay_progress(current_step)
        self._eff_progress_delay[env_ids] = sample_per_env_progress(
            global_progress=global_progress_delay,
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
            env_ids=env_ids,
            generator=self._curriculum_generator,
        )
        self._eff_progress_burst_dropout[env_ids] = sample_per_env_progress(
            global_progress=curr.get_burst_dropout_progress(current_step),
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
            env_ids=env_ids,
            generator=self._curriculum_generator,
        )

        # Push the freshly-sampled per-(env, agent) scales to the batched
        # controller BEFORE the per-agent reset loop below, otherwise
        # `_controller.reset()` → `_sample_dead_time()` uses the stale scalar
        # value (initial cfg value at first reset; previous step's value on
        # subsequent resets) and writes 0 dead times. The same step-time
        # hooks in _get_rewards will re-push these values every step, but
        # the reset path needs them up front. Flatten (N, A) → (N*A,) into
        # the batched-controller layout.
        # Ticket 037 Slice 7: read from gimbal_rate_tau axis (bit-exact
        # equal to _eff_progress_dynamics when enable_axis_independence=False).
        self._controller.gimbal_rate_loop.set_progress(
            self._axis_eff_progress("gimbal_rate_tau").t().reshape(-1)
        )
        self._controller.gimbal_rate_loop.set_dead_time_curriculum_scale(
            self._eff_progress_gimbal_dead_time.t().reshape(-1)
        )
        self._controller.zoom_controller.set_dead_time_curriculum_scale(
            self._eff_progress_zoom_dead_time.t().reshape(-1)
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

                # Compute world-frame azimuth/elevation from the ACTUAL initial
                # body-frame gimbal angles (not always the target direction).
                # When curriculum randomizes gimbal, the initial angles may NOT
                # point at the target. Seeding _azimuth_world/_elevation_world
                # from the body→world conversion ensures the first
                # compute_control() produces no gimbal movement (mas/035
                # smooth-reset invariant).
                azimuth_world, elevation_world = (
                    self._controller._gimbal._body_to_world_angles(
                        gimbal_angles[:, 0],   # body-frame yaw
                        gimbal_angles[:, 2],   # body-frame pitch
                        result.agent_orientations[:, idx],  # body quaternion
                    )
                )

                # Reset controller state via public API; the new mas/035 reset
                # signature seeds the world-frame setpoint instead of the env
                # poking into private fields.
                # Use batch indices: agent idx maps env_ids into the batched controller
                batch_ids = self._batch_idx(env_ids, idx)
                self._controller.reset(
                    batch_ids,
                    gimbal_az_initial=azimuth_world,
                    gimbal_el_initial=elevation_world,
                )
                # Sync the controller's body-frame joint state too. (The
                # _yaw/_pitch attributes are convenience caches and are
                # rewritten on the first compute_control(); seeding them
                # here just keeps reset-time observations consistent.)
                self._controller._gimbal._yaw[batch_ids] = gimbal_angles[:, 0]
                self._controller._gimbal._pitch[batch_ids] = gimbal_angles[:, 2]

                # Set zoom level (both env tracking tensor AND controller internal state).
                # The controller.reset() above resets zoom to 1.0, so we must
                # sync the random initial zoom into the controller's internal
                # _zoom and _zoom_target to prevent the first _apply_action from
                # overwriting it back to 1.0.
                #
                # The env-side tracking tensor mirrors the controller's
                # *published* zoom (the .zoom property): in first_order mode
                # this is identical to internal state; in siyi_a8 mode it
                # applies the publish-path quantization. Reading from the
                # property here keeps the first observation post-reset
                # consistent with what compute_control() will return on the
                # next step (otherwise the policy/RNN sees a one-step
                # discontinuity at every episode start).
                self._controller._zoom.set_zoom(
                    result.zoom_levels[:, idx], batch_ids
                )
                self.zoom_level[env_ids, idx] = self._controller._zoom.zoom[batch_ids]

                # Curriculum-scaled tau_zoom: near-instant (1e-4) at progress=0
                # gives the policy fast bbox/zoom learning during bootstrap,
                # ramping to the configured nominal at full dynamics progress.
                # During the dynamics phase, randomize_gains() multiplies this
                # live curriculum-gated value by a random scale (in-place), so
                # the per-env distribution remains centered on the curriculum
                # value. _nominal_gains["tau_zoom"] (Bug 4 fix) stays at
                # cfg.zoom.tau_zoom and is intentionally NOT used as the base
                # for zoom randomization — see drone_controller.randomize_gains.
                # mas/037: zoom τ₁ uses its own curriculum (zoom_tau_*) so the
                # ramp is slower than gimbal `dynamics_*`. Bootstrap stays
                # near-instant (1e-4) until zoom_tau_start_step; ramps to the
                # configured tau_zoom by zoom_tau_end_step.
                tau_zoom_curriculum = max(
                    self.cfg.drone_controller.zoom.tau_zoom * self.progress_zoom_tau, 1e-4
                )
                self._controller._zoom._tau_zoom[batch_ids] = tau_zoom_curriculum

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

        # Scale max_lin_vel with agent velocity curriculum (decoupled from target motion).
        # Ticket 034: replaces the deterministic `progress_agent_velocity * (...)` mapping
        # with the per-(env, agent) effective progress from progress_helper. The axis is
        # per-env (one max_lin_vel per env, shared across agents), so the agent-axis is
        # collapsed via mean. At any progress, the distribution of _max_lin_vel over envs
        # always retains slow-regime envs (down to max_lin_vel_min) — anti-forgetting.
        eff_p_av = self._eff_progress_agent_velocity[env_ids].mean(dim=-1)  # (len(env_ids),)
        self._max_lin_vel[env_ids] = self.cfg.max_lin_vel_min + eff_p_av * (
            self.cfg.max_lin_vel - self.cfg.max_lin_vel_min
        )

        # Randomize controller gains (curriculum-gated to drone-gains axis;
        # ticket 037 Slice 7 decoupled from progress_dynamics).
        p_drone_gains = self._axis_progress("drone_gains")
        if p_drone_gains > 0.0:
            for idx in range(len(self.cfg.possible_agents)):
                batch_ids = self._batch_idx(env_ids, idx)
                self._controller.randomize_gains(
                    batch_ids, p_drone_gains, self.cfg.gain_randomization,
                )

            # Randomize max linear velocity (decoupled axis: max_lin_vel_scale).
            # Ticket 034: composes multiplicatively with the curriculum value set
            # above, instead of replacing it. Previously this branch overwrote the
            # curriculum's `_max_lin_vel` with `cfg.max_lin_vel * Uniform(0.8, 1.2)`,
            # which collapsed all envs into the fast band [8, 12] m/s. Multiplying
            # on top preserves the per-env curriculum distribution while still
            # adding the ±20 % gain-randomization spread.
            gain_cfg = self.cfg.gain_randomization
            p_vel_scale = self._axis_progress("max_lin_vel_scale")
            if gain_cfg.randomize_max_lin_vel and p_vel_scale > 0.0:
                low = 1.0 - p_vel_scale * (1.0 - gain_cfg.max_lin_vel_scale_range[0])
                high = 1.0 + p_vel_scale * (gain_cfg.max_lin_vel_scale_range[1] - 1.0)
                M = len(env_ids)
                scale = torch.empty(M, device=self.device).uniform_(low, high)
                self._max_lin_vel[env_ids] = self._max_lin_vel[env_ids] * scale

        # ---- Domain Randomization (curriculum-gated to dynamics phase) ----
        if self._domain_randomizer is not None:
            # Sample new randomized parameters for resetting envs
            self._domain_randomizer.randomize_all(env_ids)

            # Camera intrinsics: FOV scale perturbation (camera_fov axis,
            # ticket 037 Slice 7 decoupled).
            # fov_scale < 1 means narrower FOV → larger effective focal length (1/fov_scale)
            # Curriculum gating: interpolate from 1.0 (no perturbation) to full perturbation
            fov_scales = self._domain_randomizer.get_fov_scales()  # (N, num_agents)
            raw_scale = 1.0 / fov_scales[env_ids]  # range [1.0, 2.0] for fov [0.5, 1.0]
            p_fov = self._axis_progress("camera_fov")
            self._dr_intrinsic_scale[env_ids] = 1.0 + p_fov * (raw_scale - 1.0)

            # Gimbal mechanical offsets (gimbal_mech_offsets axis, decoupled).
            # Offsets are (M, num_agents, 3) with [yaw, pitch, roll]
            gimbal_offsets = self._domain_randomizer.get_gimbal_offsets(env_ids)
            p_mech = self._axis_progress("gimbal_mech_offsets")
            self._dr_gimbal_offsets[env_ids] = p_mech * gimbal_offsets

            # Mass/inertia (mass_inertia axis, decoupled).
            if self._axis_progress("mass_inertia") > 0.0:
                # Physics: apply randomized mass/inertia to simulation assets
                # Controller retains nominal mass — intentional model mismatch for robustness
                for agent_id in self.cfg.possible_agents:
                    self._domain_randomizer.apply_physics_randomization(
                        self._robots[agent_id], env_ids
                    )

            # Gimbal joint dynamics (gimbal_stiff_damp axis, decoupled).
            if self._axis_progress("gimbal_stiff_damp") > 0.0:
                # Apply randomized stiffness/damping to actuators
                for agent_id in self.cfg.possible_agents:
                    gimbal_joint_ids = [
                        self.gimbal_joint_idx[agent_id]["yaw"],
                        self.gimbal_joint_idx[agent_id]["roll"],
                        self.gimbal_joint_idx[agent_id]["pitch"],
                    ]
                    self._domain_randomizer.apply_gimbal_dynamics(
                        self._robots[agent_id], env_ids, gimbal_joint_ids
                    )

            # Target scale randomization for bbox size enrichment
            # (target_scale axis, decoupled).
            # x=y (uniform), z >= xy, curriculum-gated: 1.0 at progress=0, full at progress=1
            M = len(env_ids)
            xy_lo, xy_hi = self.cfg.target_xy_scale_range
            z_lo, z_hi = self.cfg.target_z_scale_range
            raw_xy = torch.empty(M, device=self.device).uniform_(xy_lo, xy_hi)
            raw_z = torch.empty(M, device=self.device).uniform_(z_lo, z_hi)
            raw_z = torch.max(raw_z, raw_xy)  # z >= xy
            p_target = self._axis_progress("target_scale")
            gated_xy = 1.0 + p_target * (raw_xy - 1.0)
            gated_z = 1.0 + p_target * (raw_z - 1.0)
            self._dr_target_scale[env_ids, 0, 0] = gated_xy  # x
            self._dr_target_scale[env_ids, 0, 1] = gated_xy  # y = x
            self._dr_target_scale[env_ids, 0, 2] = gated_z   # z >= xy

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

        # Log action delta RMS and reset accumulators
        smoothness_steps = torch.clamp(self._action_smoothness_steps[env_ids], min=1.0)
        for agent_id in self.cfg.possible_agents:
            per_dim_rms = torch.sqrt(
                self._action_delta_sq_acc[agent_id][env_ids]
                / smoothness_steps.unsqueeze(-1)
            )  # [len(env_ids), 7]
            # Log aggregate RMS only (per-dim available via parity eval)
            all_extras[f"Action_Smoothness/{agent_id}_total_rms"] = torch.mean(
                torch.norm(per_dim_rms, dim=-1)
            )
            self._action_delta_sq_acc[agent_id][env_ids] = 0.0
        self._action_smoothness_steps[env_ids] = 0.0

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

        # Log tracking-lost truncation statistics
        all_extras["Termination/tracking_lost_fraction"] = torch.mean(
            self._tracking_lost_count[env_ids].clamp(max=1.0)
        )

        # Reset detection stats, collision count, and tracking-lost counter
        for key in self._detection_stats:
            self._detection_stats[key][env_ids] = 0.0
        self._collision_count[env_ids] = 0.0
        self._all_lost_counter[env_ids] = 0
        self._detection_reacquire_counter[env_ids] = 0
        self._tracking_lost_count[env_ids] = 0.0

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(all_extras)

        # Clear triangulation results (will be recomputed on first step)
        self._triangulation_result_gt = None
        self._triangulation_result_obs = None

        # # Point viewer camera at target when the tracked env resets
        # if self.viewport_camera_controller is not None and self.cfg.viewer.env_index in env_ids:
        #     target_pos_w = self.target.data.root_pos_w[self.cfg.viewer.env_index]  # [3]
        #     robot_pos_w = self._robots[self.cfg.possible_agents[0]].data.root_pos_w[self.cfg.viewer.env_index]  # [3]
        #     lookat_offset = (target_pos_w - robot_pos_w).detach().cpu().numpy() * 1.5
        #     import numpy as np
        #     angle = np.arctan2(-lookat_offset[1], -lookat_offset[0])
        #     eye = np.array([10 * np.cos(angle), 10 * np.sin(angle), 1.0])
        #     self.viewport_camera_controller.update_view_location(eye=eye)

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

            # Reset controller state (use batch indices)
            batch_ids = self._batch_idx(env_ids, idx)
            self._controller.reset(batch_ids)

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
        if self._triangulation_result_gt is not None:
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
