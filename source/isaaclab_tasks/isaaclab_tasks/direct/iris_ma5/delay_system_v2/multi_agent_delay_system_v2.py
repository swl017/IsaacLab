# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-agent delay system using V2 field-based architecture."""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Any
import copy

from isaaclab.envs.common import AgentID
import logging
from isaaclab.utils.math import quat_rotate_inverse, normalize, quat_mul, quat_from_angle_axis

# Import local delay system V2 components (self-contained)
from .delay_system_v2 import DelaySystemV2
from .delay_cfg import (
    DelaySystemCfgV2,
    FieldDelayCfg,
    DistributionCfg,
    NoiseCfg,
)

from .multi_agent_delay_system_v2_cfg import MultiAgentDelaySystemV2Cfg
from .agent_states import AgentStates, AgentStatesData
from .derived_field_computers import (
    compute_camera_orientation_from_gimbal,
    compute_camera_position,
    compute_ray_origins,
    compute_ray_directions_from_bbox,
    compute_combined_angular_velocity,
)

logger = logging.getLogger(__name__)


class MultiAgentDelaySystemV2:
    """Multi-agent delay system using V2 field-based architecture.

    This class wraps the quadcopter's DelaySystemV2 to provide a multi-agent
    delay system with:
    - Dual pipelines: Clean (for rewards) + Noisy (for observations)
    - Per-agent field storage with perspective-aware delays
    - Ego (fast) vs Inter-agent (slow) communication delays
    - Noise injection before delays
    - API compatibility with iris_ma3 delay system

    Field Naming Convention:
        "{agent_id}.{field_name}.{perspective}"
        Examples:
        - "agent_0.body_position_w.ego" - Fast path for ego's own view
        - "agent_0.body_position_w.other" - Slow path for other agents' view

    Example usage:
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=["agent_0", "agent_1"],
            num_envs=512,
            num_joints_per_agent={"agent_0": 3, "agent_1": 3},
            num_targets_per_agent={"agent_0": 1, "agent_1": 1},
            device="cuda:0",
        )

        # Per-step updates
        delay_system.update_time()
        for agent_id in agents:
            delay_system.update_gt_states(agent_id, body_position_w=pos, ...)
            delay_system.update_detections(agent_id, bboxes_2d_gt=bboxes)

        # Get states for reward/observation computation
        reward_states = delay_system.get_all_states_for_rewards("agent_0")
        obs_states = delay_system.get_all_states_for_observations("agent_0")
    """

    # Raw sensor fields (delayed independently)
    RAW_MOTION_FIELDS = [
        "body_position_w",
        "body_linear_velocity_w",
        "body_angular_velocity_w",
        "body_linear_acceleration_w",
    ]
    RAW_ORIENTATION_FIELDS = [
        "body_orientation_w",
    ]
    RAW_JOINT_FIELDS = [
        "joint_positions_b",
        "joint_velocities_b",
    ]
    RAW_DETECTION_FIELDS = [
        "bboxes_2d",
    ]
    RAW_ZOOM_FIELDS = [
        "camera_zoom_level",
    ]
    # NOTE: timestamp_detection is NOT a separate pipeline field. It is concatenated
    # as the last dimension of bboxes_2d so it shares the exact same DelayPipeline
    # (identical staleness, latency, and dropout behavior).

    # Static fields (no delay, passthrough)
    STATIC_FIELDS = [
        "camera_offset_position_b",
        "camera_offset_rotation_b",
        "camera_base_intrinsics",
    ]

    def __init__(
        self,
        cfg: MultiAgentDelaySystemV2Cfg,
        possible_agents: List[AgentID],
        num_envs: int,
        num_joints_per_agent: Dict[AgentID, int],
        num_targets_per_agent: Dict[AgentID, int],
        device: torch.device,
    ):
        """Initialize multi-agent delay system.

        Args:
            cfg: Configuration for the delay system.
            possible_agents: List of agent IDs.
            num_envs: Number of parallel environments.
            num_joints_per_agent: Number of gimbal joints per agent.
            num_targets_per_agent: Number of detection targets per agent.
            device: Device for tensor allocation.
        """
        self.cfg = cfg
        self._possible_agents = possible_agents
        self._num_envs = num_envs
        self._num_joints_per_agent = num_joints_per_agent
        self._num_targets_per_agent = num_targets_per_agent
        self._device = device

        # Camera configuration storage
        self._camera_configs: Dict[AgentID, Dict[str, Any]] = {}

        # Noise progress scale (for curriculum learning) - deprecated, use per-env noise
        self._noise_progress_scale = 1.0 if cfg.enable_noise else 0.0

        # Per-environment noise standard deviations (sampled on reset)
        # Each environment gets its own noise level for domain randomization
        self._per_env_noise_std = {
            "position": torch.zeros(num_envs, device=device),
            "orientation": torch.zeros(num_envs, device=device),
            "linear_velocity": torch.zeros(num_envs, device=device),
            "angular_velocity": torch.zeros(num_envs, device=device),
            "linear_acceleration": torch.zeros(num_envs, device=device),
            "gimbal": torch.zeros(num_envs, device=device),
            "bbox": torch.zeros(num_envs, device=device),
            "zoom": torch.zeros(num_envs, device=device),
        }

        # Build field dimensions and configs
        self._field_dims = self._build_field_dims()
        self._field_configs_ego, self._field_configs_other = self._build_field_configs()

        # Create V2 delay system configuration
        v2_cfg_ego = DelaySystemCfgV2(
            dt=cfg.dt,
            use_enhanced_mode=True,
            field_configs=self._field_configs_ego,
        )
        v2_cfg_other = DelaySystemCfgV2(
            dt=cfg.dt,
            use_enhanced_mode=True,
            field_configs=self._field_configs_other,
        )

        # Build field dims for both perspectives
        ego_field_dims = {k: v for k, v in self._field_dims.items() if k.endswith(".ego")}
        other_field_dims = {k: v for k, v in self._field_dims.items() if k.endswith(".other")}

        # Create TWO delay systems for each perspective (clean/noisy x ego/other)
        # But actually, we need clean and noisy for both ego and other
        # Simplify: use one DelaySystemV2 for ego, one for other, each handles clean/noisy internally
        # Actually DelaySystemV2 doesn't separate clean/noisy - it has dual pipelines internally
        # So we need: clean_ego, noisy_ego, clean_other, noisy_other = 4 delay systems

        # Clean pipeline (for rewards) - ego perspective
        self._delay_clean_ego = DelaySystemV2(
            cfg=v2_cfg_ego,
            num_envs=num_envs,
            device=str(device),
            field_dims=ego_field_dims,
        )
        # Clean pipeline (for rewards) - other agent perspective
        self._delay_clean_other = DelaySystemV2(
            cfg=v2_cfg_other,
            num_envs=num_envs,
            device=str(device),
            field_dims=other_field_dims,
        )
        # Noisy pipeline (for observations) - ego perspective
        self._delay_noisy_ego = DelaySystemV2(
            cfg=v2_cfg_ego,
            num_envs=num_envs,
            device=str(device),
            field_dims=ego_field_dims,
        )
        # Noisy pipeline (for observations) - other agent perspective
        self._delay_noisy_other = DelaySystemV2(
            cfg=v2_cfg_other,
            num_envs=num_envs,
            device=str(device),
            field_dims=other_field_dims,
        )

        # Ground truth states storage (before any delays)
        self._gt_states: Dict[AgentID, AgentStates] = {}
        for agent_id in possible_agents:
            self._gt_states[agent_id] = AgentStates(
                num_envs=num_envs,
                num_joints=num_joints_per_agent[agent_id],
                num_targets=num_targets_per_agent[agent_id],
                device=device,
            )

        # Current simulation time
        self._current_time = torch.zeros(num_envs, device=device)

        # Initialize delay pipelines with default values
        # This ensures data is available immediately after construction
        self._initialize_pipelines_from_gt_states()

    @property
    def current_time(self) -> torch.Tensor:
        """Current simulation time per environment [N]."""
        return self._current_time

    @property
    def gt_states(self) -> "GTStatesAccessor":
        """Access to ground truth states (before delays).

        Returns an accessor object for compatibility with iris_ma3 API.
        """
        return GTStatesAccessor(self._gt_states)

    def _build_field_dims(self) -> Dict[str, int]:
        """Build field dimensions dictionary for all agents and perspectives."""
        dims = {}

        for agent_id in self._possible_agents:
            num_joints = self._num_joints_per_agent[agent_id]
            num_targets = self._num_targets_per_agent[agent_id]

            for perspective in ["ego", "other"]:
                # Motion fields (3D vectors)
                for field in self.RAW_MOTION_FIELDS:
                    dims[f"{agent_id}.{field}.{perspective}"] = 3

                # Orientation fields (quaternion)
                for field in self.RAW_ORIENTATION_FIELDS:
                    dims[f"{agent_id}.{field}.{perspective}"] = 4

                # Joint fields
                for field in self.RAW_JOINT_FIELDS:
                    dims[f"{agent_id}.{field}.{perspective}"] = num_joints

                # Detection fields (flattened bbox + coupled timestamp)
                for field in self.RAW_DETECTION_FIELDS:
                    dims[f"{agent_id}.{field}.{perspective}"] = num_targets * 4 + 1  # [T*4] bbox + [1] timestamp

                # Zoom field
                for field in self.RAW_ZOOM_FIELDS:
                    dims[f"{agent_id}.{field}.{perspective}"] = 1

        return dims

    def _build_field_configs(self) -> tuple[Dict[str, FieldDelayCfg], Dict[str, FieldDelayCfg]]:
        """Build field configurations for ego and other perspectives."""
        cfg = self.cfg

        # Ego perspective configs (fast, minimal delay)
        ego_configs = {}
        # Other perspective configs (slow, full communication delay)
        other_configs = {}

        for agent_id in self._possible_agents:
            # === Ego perspective (fast local processing) ===
            # Motion fields: first-order lag only
            motion_cfg_ego = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.ego_comm_time_constant,
            )
            for field in self.RAW_MOTION_FIELDS:
                ego_configs[f"{agent_id}.{field}.ego"] = motion_cfg_ego

            # Orientation fields: first-order lag
            orientation_cfg_ego = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.ego_comm_time_constant,
            )
            for field in self.RAW_ORIENTATION_FIELDS:
                ego_configs[f"{agent_id}.{field}.ego"] = orientation_cfg_ego

            # Joint fields: first-order lag
            joint_cfg_ego = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.ego_comm_time_constant,
            )
            for field in self.RAW_JOINT_FIELDS:
                ego_configs[f"{agent_id}.{field}.ego"] = joint_cfg_ego

            # Detection fields: staleness + latency + dropout (same for ego)
            # NOTE: first_order_lag MUST remain disabled for detection fields because
            # the last dimension is a concatenated timestamp that must not be smoothed.
            detection_cfg_ego = FieldDelayCfg(
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="uniform",
                    mean=cfg.detection_fps_mean,
                    half_range=cfg.detection_fps_std,
                ),
                latency_enabled=True,
                latency=DistributionCfg(
                    type="normal",
                    mean=cfg.detection_latency_mean,
                    std=cfg.detection_latency_std,
                ),
                dropout_enabled=True,
                dropout_prob=cfg.detection_dropout_rate,
            )
            for field in self.RAW_DETECTION_FIELDS:
                ego_configs[f"{agent_id}.{field}.ego"] = detection_cfg_ego

            # Zoom field: first-order lag
            zoom_cfg_ego = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.ego_comm_time_constant,
            )
            for field in self.RAW_ZOOM_FIELDS:
                ego_configs[f"{agent_id}.{field}.ego"] = zoom_cfg_ego

            # === Other perspective (slow inter-agent communication) ===
            # Motion fields: first-order lag + staleness + latency + dropout
            motion_cfg_other = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.motion_time_constant,
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="uniform",
                    mean=cfg.inter_agent_comm_fps_mean,
                    half_range=cfg.inter_agent_comm_fps_std,
                ),
                latency_enabled=True,
                latency=DistributionCfg(
                    type="normal",
                    mean=cfg.inter_agent_comm_latency_mean,
                    std=cfg.inter_agent_comm_latency_std,
                ),
                dropout_enabled=True,
                dropout_prob=cfg.inter_agent_comm_dropout_rate,
            )
            for field in self.RAW_MOTION_FIELDS:
                other_configs[f"{agent_id}.{field}.other"] = motion_cfg_other

            # Orientation fields: same as motion
            orientation_cfg_other = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.orientation_time_constant,
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="uniform",
                    mean=cfg.inter_agent_comm_fps_mean,
                    half_range=cfg.inter_agent_comm_fps_std,
                ),
                latency_enabled=True,
                latency=DistributionCfg(
                    type="normal",
                    mean=cfg.inter_agent_comm_latency_mean,
                    std=cfg.inter_agent_comm_latency_std,
                ),
                dropout_enabled=True,
                dropout_prob=cfg.inter_agent_comm_dropout_rate,
            )
            for field in self.RAW_ORIENTATION_FIELDS:
                other_configs[f"{agent_id}.{field}.other"] = orientation_cfg_other

            # Joint fields
            joint_cfg_other = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.joint_time_constant,
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="uniform",
                    mean=cfg.inter_agent_comm_fps_mean,
                    half_range=cfg.inter_agent_comm_fps_std,
                ),
                latency_enabled=True,
                latency=DistributionCfg(
                    type="normal",
                    mean=cfg.inter_agent_comm_latency_mean,
                    std=cfg.inter_agent_comm_latency_std,
                ),
                dropout_enabled=True,
                dropout_prob=cfg.inter_agent_comm_dropout_rate,
            )
            for field in self.RAW_JOINT_FIELDS:
                other_configs[f"{agent_id}.{field}.other"] = joint_cfg_other

            # Detection fields: combined detector + communication delays
            # NOTE: first_order_lag MUST remain disabled for detection fields because
            # the last dimension is a concatenated timestamp that must not be smoothed.
            detection_cfg_other = FieldDelayCfg(
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="uniform",
                    mean=cfg.detection_fps_mean,
                    half_range=cfg.detection_fps_std,
                ),
                latency_enabled=True,
                latency=DistributionCfg(
                    type="normal",
                    mean=cfg.detection_latency_mean + cfg.inter_agent_comm_latency_mean,
                    std=(cfg.detection_latency_std**2 + cfg.inter_agent_comm_latency_std**2)**0.5,
                ),
                dropout_enabled=True,
                dropout_prob=min(1.0, cfg.detection_dropout_rate + cfg.inter_agent_comm_dropout_rate),
            )
            for field in self.RAW_DETECTION_FIELDS:
                other_configs[f"{agent_id}.{field}.other"] = detection_cfg_other

            # Zoom field
            zoom_cfg_other = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.zoom_time_constant,
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="uniform",
                    mean=cfg.inter_agent_comm_fps_mean,
                    half_range=cfg.inter_agent_comm_fps_std,
                ),
                latency_enabled=True,
                latency=DistributionCfg(
                    type="normal",
                    mean=cfg.inter_agent_comm_latency_mean,
                    std=cfg.inter_agent_comm_latency_std,
                ),
                dropout_enabled=True,
                dropout_prob=cfg.inter_agent_comm_dropout_rate,
            )
            for field in self.RAW_ZOOM_FIELDS:
                other_configs[f"{agent_id}.{field}.other"] = zoom_cfg_other

        return ego_configs, other_configs

    def set_camera_configs(
        self,
        agent_id: AgentID,
        width: int,
        height: int,
        focal_length: float,
        horizontal_aperture: float,
        vertical_aperture: float,
        offset_position_b: torch.Tensor,
        offset_rotation_b: torch.Tensor,
    ):
        """Set camera configuration for an agent.

        Args:
            agent_id: Agent identifier.
            width: Image width in pixels.
            height: Image height in pixels.
            focal_length: Focal length in mm.
            horizontal_aperture: Horizontal aperture in mm.
            vertical_aperture: Vertical aperture in mm.
            offset_position_b: Camera offset position in body frame [3].
            offset_rotation_b: Camera offset rotation in body frame [4] (w,x,y,z).
        """
        # Compute intrinsics matrix K
        fx = focal_length * width / horizontal_aperture
        fy = focal_length * height / vertical_aperture
        cx = width / 2.0
        cy = height / 2.0

        # Create intrinsics matrix [N, 3, 3]
        K = torch.zeros(self._num_envs, 3, 3, device=self._device)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0

        # Store config
        self._camera_configs[agent_id] = {
            "width": width,
            "height": height,
            "focal_length": focal_length,
            "horizontal_aperture": horizontal_aperture,
            "vertical_aperture": vertical_aperture,
            "offset_position_b": offset_position_b.to(self._device),
            "offset_rotation_b": offset_rotation_b.to(self._device),
            "intrinsics": K,
        }

        # Update GT states with camera config
        gt_state = self._gt_states[agent_id]
        gt_state.data.camera_offset_position_b[:] = offset_position_b.to(self._device)
        gt_state.data.camera_offset_rotation_b[:] = offset_rotation_b.to(self._device)
        gt_state.data.camera_base_intrinsics[:] = K

    def update_time(self, dt: Optional[float] = None):
        """Advance simulation time for all delay systems.

        Args:
            dt: Time step in seconds. Uses cfg.dt if None.
        """
        dt_actual = dt or self.cfg.dt
        self._current_time += dt_actual

        # Step all delay systems
        self._delay_clean_ego.step(dt_actual)
        self._delay_clean_other.step(dt_actual)
        self._delay_noisy_ego.step(dt_actual)
        self._delay_noisy_other.step(dt_actual)

    def update_gt_states(
        self,
        agent_id: AgentID,
        body_position_w: torch.Tensor,
        body_orientation_w: torch.Tensor,
        body_linear_velocity_w: torch.Tensor,
        body_angular_velocity_w: torch.Tensor,
        body_linear_acceleration_w: torch.Tensor,
        body_combined_angular_velocity_w: torch.Tensor,
        joint_positions_b: torch.Tensor,
        zoom_level: torch.Tensor,
    ) -> torch.Tensor:
        """Update ground truth states for an agent.

        This stores the raw states and feeds them to the delay pipelines.
        Detects NaN/Inf in simulator states and sanitizes them to prevent
        pipeline corruption. Returns a mask of environments with bad states
        so the caller can terminate those episodes.

        Args:
            agent_id: Agent identifier.
            body_position_w: Body position in world frame [N, 3].
            body_orientation_w: Body orientation (quaternion) in world frame [N, 4].
            body_linear_velocity_w: Linear velocity in world frame [N, 3].
            body_angular_velocity_w: Angular velocity in world frame [N, 3].
            body_linear_acceleration_w: Linear acceleration in world frame [N, 3].
            body_combined_angular_velocity_w: Combined angular velocity [N, 3].
            joint_positions_b: Gimbal joint positions [N, J].
            zoom_level: Camera zoom level [N].

        Returns:
            nan_env_mask: Boolean tensor [N] indicating environments with NaN/Inf
                in any state. These environments should be terminated by the caller.
        """
        # Check all input states for NaN/Inf before processing
        states_to_check = {
            "body_position_w": body_position_w,
            "body_orientation_w": body_orientation_w,
            "body_linear_velocity_w": body_linear_velocity_w,
            "body_angular_velocity_w": body_angular_velocity_w,
            "body_linear_acceleration_w": body_linear_acceleration_w,
            "body_combined_angular_velocity_w": body_combined_angular_velocity_w,
            "joint_positions_b": joint_positions_b,
        }
        zoom_expanded = zoom_level.unsqueeze(-1) if zoom_level.dim() == 1 else zoom_level
        states_to_check["zoom_level"] = zoom_expanded

        nan_env_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        for field_name, state in states_to_check.items():
            bad = torch.isnan(state) | torch.isinf(state)
            if bad.any():
                bad_envs = bad.any(dim=-1) if state.dim() > 1 else bad
                nan_env_mask = nan_env_mask | bad_envs
                bad_count = bad_envs.sum().item()
                logger.warning(
                    f"NaN/Inf in simulator state '{field_name}' for agent '{agent_id}' "
                    f"in {bad_count} envs"
                )

        # Sanitize NaN environments to prevent pipeline corruption
        if nan_env_mask.any():
            body_position_w = body_position_w.clone()
            body_orientation_w = body_orientation_w.clone()
            body_linear_velocity_w = body_linear_velocity_w.clone()
            body_angular_velocity_w = body_angular_velocity_w.clone()
            body_linear_acceleration_w = body_linear_acceleration_w.clone()
            body_combined_angular_velocity_w = body_combined_angular_velocity_w.clone()
            joint_positions_b = joint_positions_b.clone()
            zoom_level = zoom_level.clone()

            body_position_w[nan_env_mask] = 0.0
            body_orientation_w[nan_env_mask] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self._device
            )
            body_linear_velocity_w[nan_env_mask] = 0.0
            body_angular_velocity_w[nan_env_mask] = 0.0
            body_linear_acceleration_w[nan_env_mask] = 0.0
            body_combined_angular_velocity_w[nan_env_mask] = 0.0
            joint_positions_b[nan_env_mask] = 0.0
            zoom_level[nan_env_mask] = 1.0

        # Store GT states
        gt_state = self._gt_states[agent_id]
        gt_state.data.body_position_w[:] = body_position_w
        gt_state.data.body_orientation_w[:] = body_orientation_w
        gt_state.data.body_linear_velocity_w[:] = body_linear_velocity_w
        gt_state.data.body_angular_velocity_w[:] = body_angular_velocity_w
        gt_state.data.body_linear_acceleration_w[:] = body_linear_acceleration_w
        gt_state.data.body_combined_angular_velocity_w[:] = body_combined_angular_velocity_w
        gt_state.data.joint_positions_b[:] = joint_positions_b
        gt_state.data.camera_zoom_level[:] = zoom_level
        gt_state.data.timestamp_motion[:] = self._current_time

        # Compute body-frame velocities
        gt_state.data.body_linear_velocity_b[:] = quat_rotate_inverse(
            body_orientation_w, body_linear_velocity_w
        )
        gt_state.data.body_angular_velocity_b[:] = quat_rotate_inverse(
            body_orientation_w, body_angular_velocity_w
        )
        gt_state.data.body_combined_angular_velocity_b[:] = quat_rotate_inverse(
            body_orientation_w, body_combined_angular_velocity_w
        )
        gt_state.data.body_linear_acceleration_b[:] = quat_rotate_inverse(
            body_orientation_w, body_linear_acceleration_w
        )

        # Compute derived camera fields
        if agent_id in self._camera_configs:
            cam_cfg = self._camera_configs[agent_id]
            gt_state.data.camera_position_w[:] = compute_camera_position(
                body_position_w,
                body_orientation_w,
                cam_cfg["offset_position_b"].expand(self._num_envs, -1),
            )
            gt_state.data.camera_orientation_w[:] = compute_camera_orientation_from_gimbal(
                body_orientation_w,
                joint_positions_b,
                cam_cfg["offset_rotation_b"].expand(self._num_envs, -1),
            )

        # Store to delay pipelines (both perspectives)
        self._store_to_pipelines(agent_id, "ego", body_position_w, body_orientation_w,
                                  body_linear_velocity_w, body_angular_velocity_w,
                                  body_linear_acceleration_w, joint_positions_b, zoom_level)
        self._store_to_pipelines(agent_id, "other", body_position_w, body_orientation_w,
                                  body_linear_velocity_w, body_angular_velocity_w,
                                  body_linear_acceleration_w, joint_positions_b, zoom_level)

        return nan_env_mask

    def _store_to_pipelines(
        self,
        agent_id: AgentID,
        perspective: str,
        body_position_w: torch.Tensor,
        body_orientation_w: torch.Tensor,
        body_linear_velocity_w: torch.Tensor,
        body_angular_velocity_w: torch.Tensor,
        body_linear_acceleration_w: torch.Tensor,
        joint_positions_b: torch.Tensor,
        zoom_level: torch.Tensor,
    ):
        """Store states to clean and noisy pipelines.

        Uses per-environment noise standard deviations for domain randomization.
        Each environment has its own noise level sampled on reset.
        """
        cfg = self.cfg

        # Select appropriate delay systems based on perspective
        if perspective == "ego":
            delay_clean = self._delay_clean_ego
            delay_noisy = self._delay_noisy_ego
        else:
            delay_clean = self._delay_clean_other
            delay_noisy = self._delay_noisy_other

        # Get per-env noise std and reshape for broadcasting: [N] -> [N, 1]
        position_noise_std = self._per_env_noise_std["position"].unsqueeze(-1)
        lin_vel_noise_std = self._per_env_noise_std["linear_velocity"].unsqueeze(-1)
        ang_vel_noise_std = self._per_env_noise_std["angular_velocity"].unsqueeze(-1)
        lin_acc_noise_std = self._per_env_noise_std["linear_acceleration"].unsqueeze(-1)

        # Motion fields with per-env noise
        fields_data = {
            "body_position_w": (body_position_w, position_noise_std),
            "body_linear_velocity_w": (body_linear_velocity_w, lin_vel_noise_std),
            "body_angular_velocity_w": (body_angular_velocity_w, ang_vel_noise_std),
            "body_linear_acceleration_w": (body_linear_acceleration_w, lin_acc_noise_std),
        }

        for field_name, (data, noise_std) in fields_data.items():
            full_name = f"{agent_id}.{field_name}.{perspective}"
            # Clean pipeline
            delay_clean.store(full_name, data)
            # Noisy pipeline (with per-env noise)
            if cfg.enable_noise:
                # noise_std is [N, 1], broadcasts to [N, 3]
                noisy_data = data + torch.randn_like(data) * noise_std
                delay_noisy.store(full_name, noisy_data)
            else:
                delay_noisy.store(full_name, data)

        # Orientation field (rotation-based noise for quaternions)
        # Instead of additive noise + normalize (which is geometrically incorrect),
        # we compose the original quaternion with a small random rotation.
        # The perturbation angle is sampled from N(0, noise_std) in radians,
        # applied around a uniformly random axis on the unit sphere.
        full_name = f"{agent_id}.body_orientation_w.{perspective}"
        delay_clean.store(full_name, body_orientation_w)
        if cfg.enable_noise:
            N = body_orientation_w.shape[0]
            ori_noise_std = self._per_env_noise_std["orientation"]  # [N]

            # Sample random rotation axis (uniform on unit sphere)
            random_axis = torch.randn(N, 3, device=self._device)
            random_axis = normalize(random_axis)  # [N, 3]

            # Sample perturbation angle from N(0, noise_std) in radians
            perturbation_angle = torch.randn(N, device=self._device) * ori_noise_std  # [N]

            # Create perturbation quaternion and compose with original
            perturbation_quat = quat_from_angle_axis(perturbation_angle, random_axis)  # [N, 4]
            noisy_quat = quat_mul(body_orientation_w, perturbation_quat)  # [N, 4]
            noisy_quat = normalize(noisy_quat)

            delay_noisy.store(full_name, noisy_quat)
        else:
            delay_noisy.store(full_name, body_orientation_w)

        # Joint fields (gimbal)
        full_name = f"{agent_id}.joint_positions_b.{perspective}"
        delay_clean.store(full_name, joint_positions_b)
        if cfg.enable_noise:
            gimbal_noise_std = self._per_env_noise_std["gimbal"].unsqueeze(-1)  # [N, 1]
            noisy_joints = joint_positions_b + torch.randn_like(joint_positions_b) * gimbal_noise_std
            delay_noisy.store(full_name, noisy_joints)
        else:
            delay_noisy.store(full_name, joint_positions_b)

        # Placeholder for joint velocities (set to zero for now)
        full_name = f"{agent_id}.joint_velocities_b.{perspective}"
        zero_joint_vel = torch.zeros_like(joint_positions_b)
        delay_clean.store(full_name, zero_joint_vel)
        delay_noisy.store(full_name, zero_joint_vel)

        # Zoom field
        full_name = f"{agent_id}.camera_zoom_level.{perspective}"
        zoom_expanded = zoom_level.unsqueeze(-1) if zoom_level.dim() == 1 else zoom_level
        delay_clean.store(full_name, zoom_expanded)
        if cfg.enable_noise:
            zoom_noise_std = self._per_env_noise_std["zoom"].unsqueeze(-1)  # [N, 1]
            noisy_zoom = zoom_expanded + torch.randn_like(zoom_expanded) * zoom_noise_std
            noisy_zoom = torch.clamp(noisy_zoom, min=1.0)
            delay_noisy.store(full_name, noisy_zoom)
        else:
            delay_noisy.store(full_name, zoom_expanded)

    def update_detections(
        self,
        agent_id: AgentID,
        bboxes_2d_gt: torch.Tensor,
    ):
        """Update detection (bounding box) data for an agent.

        Uses per-environment noise standard deviations for domain randomization.

        Args:
            agent_id: Agent identifier.
            bboxes_2d_gt: Ground truth bounding boxes [N, T, 4] (x, y, w, h).
        """
        cfg = self.cfg

        # Store GT
        gt_state = self._gt_states[agent_id]
        gt_state.data.bboxes_2d[:] = bboxes_2d_gt
        gt_state.data.timestamp_detection[:] = self._current_time

        # Compute ray directions from GT
        if agent_id in self._camera_configs:
            cam_cfg = self._camera_configs[agent_id]
            num_targets = self._num_targets_per_agent[agent_id]

            gt_state.data.camera_ray_origins_w[:] = compute_ray_origins(
                gt_state.data.camera_position_w, num_targets
            )
            gt_state.data.camera_ray_directions_w[:] = compute_ray_directions_from_bbox(
                gt_state.data.camera_orientation_w,
                cam_cfg["intrinsics"],
                gt_state.data.camera_zoom_level,
                bboxes_2d_gt,
            )

        # Flatten bboxes for storage: [N, T, 4] -> [N, T*4]
        bboxes_flat = bboxes_2d_gt.flatten(start_dim=1)

        # Timestamp: [N] -> [N, 1] for concatenation
        timestamp_data = self._current_time.unsqueeze(-1)

        # Store to both perspectives
        # Timestamp is concatenated as the last dimension of the detection tensor
        # so it shares the exact same DelayPipeline (identical staleness/latency/dropout).
        for perspective in ["ego", "other"]:
            if perspective == "ego":
                delay_clean = self._delay_clean_ego
                delay_noisy = self._delay_noisy_ego
            else:
                delay_clean = self._delay_clean_other
                delay_noisy = self._delay_noisy_other

            full_name = f"{agent_id}.bboxes_2d.{perspective}"

            # Clean: concat bbox + timestamp -> [N, T*4+1]
            clean_concat = torch.cat([bboxes_flat, timestamp_data], dim=-1)
            delay_clean.store(full_name, clean_concat)

            # Noisy: add noise ONLY to bbox dimensions, NOT to timestamp
            if cfg.enable_noise:
                bbox_noise_std = self._per_env_noise_std["bbox"].unsqueeze(-1)  # [N, 1]
                noisy_bbox = bboxes_flat + torch.randn_like(bboxes_flat) * bbox_noise_std
                noisy_concat = torch.cat([noisy_bbox, timestamp_data], dim=-1)
                delay_noisy.store(full_name, noisy_concat)
            else:
                delay_noisy.store(full_name, clean_concat)

    def get_all_states_for_rewards(self, ego_agent_id: AgentID) -> Dict[AgentID, AgentStates]:
        """Get clean delayed states for reward computation.

        Args:
            ego_agent_id: The ego agent requesting the states.

        Returns:
            Dictionary mapping agent IDs to their delayed states (clean, no noise).
        """
        return self._build_agent_states(ego_agent_id, use_clean=True)

    def get_all_states_for_observations(self, ego_agent_id: AgentID) -> Dict[AgentID, AgentStates]:
        """Get noisy delayed states for observation computation.

        Args:
            ego_agent_id: The ego agent requesting the states.

        Returns:
            Dictionary mapping agent IDs to their delayed states (with noise).
        """
        return self._build_agent_states(ego_agent_id, use_clean=False)

    def _build_agent_states(
        self, ego_agent_id: AgentID, use_clean: bool
    ) -> Dict[AgentID, AgentStates]:
        """Build AgentStates from delayed fields.

        Args:
            ego_agent_id: The ego agent's perspective.
            use_clean: If True, use clean pipeline (rewards). If False, use noisy (observations).

        Returns:
            Dictionary mapping agent IDs to their delayed states.
        """
        result = {}

        for agent_id in self._possible_agents:
            # Determine perspective and select appropriate delay system
            if agent_id == ego_agent_id:
                perspective = "ego"
                delay_sys = self._delay_clean_ego if use_clean else self._delay_noisy_ego
            else:
                perspective = "other"
                delay_sys = self._delay_clean_other if use_clean else self._delay_noisy_other

            # Create new AgentStates
            num_joints = self._num_joints_per_agent[agent_id]
            num_targets = self._num_targets_per_agent[agent_id]
            agent_states = AgentStates(
                num_envs=self._num_envs,
                num_joints=num_joints,
                num_targets=num_targets,
                device=self._device,
            )

            # Retrieve delayed fields
            def get_field(field_name: str) -> torch.Tensor:
                full_name = f"{agent_id}.{field_name}.{perspective}"
                if use_clean:
                    return delay_sys.get_delayed_clean(full_name)
                else:
                    return delay_sys.get_delayed_noisy(full_name)

            # Motion fields
            agent_states.data.body_position_w[:] = get_field("body_position_w")
            agent_states.data.body_linear_velocity_w[:] = get_field("body_linear_velocity_w")
            agent_states.data.body_angular_velocity_w[:] = get_field("body_angular_velocity_w")
            agent_states.data.body_linear_acceleration_w[:] = get_field("body_linear_acceleration_w")

            # Orientation (normalize after filtering with zero/NaN quaternion protection)
            body_orientation = get_field("body_orientation_w")
            body_orientation_norm = normalize(body_orientation)
            # Detect invalid quaternions: zero norm, NaN, or Inf
            quat_norm = body_orientation.norm(dim=-1, keepdim=True)
            invalid_quat_mask = (
                torch.isnan(quat_norm) | torch.isinf(quat_norm) | (quat_norm < 1e-6)
            )  # [N, 1]
            if invalid_quat_mask.any():
                identity_quat = torch.zeros_like(body_orientation_norm)
                identity_quat[..., 0] = 1.0  # w=1 for identity
                body_orientation_norm = torch.where(
                    invalid_quat_mask, identity_quat, body_orientation_norm
                )
            agent_states.data.body_orientation_w[:] = body_orientation_norm

            # Joint fields
            agent_states.data.joint_positions_b[:] = get_field("joint_positions_b")
            agent_states.data.joint_velocities_b[:] = get_field("joint_velocities_b")

            # Zoom field (with protection against zero values)
            zoom = get_field("camera_zoom_level")
            zoom = zoom.squeeze(-1) if zoom.dim() > 1 else zoom
            # Protect against zero zoom level (would cause division by zero in ray computation)
            zoom = torch.clamp(zoom, min=0.1)
            agent_states.data.camera_zoom_level[:] = zoom

            # Detection field with coupled timestamp: [N, T*4+1]
            detection_data = get_field("bboxes_2d")
            bboxes_flat = detection_data[:, :-1]  # [N, T*4]
            detection_timestamp = detection_data[:, -1]  # [N]
            agent_states.data.bboxes_2d[:] = bboxes_flat.reshape(self._num_envs, num_targets, 4)

            # Compute body-frame velocities from world-frame
            agent_states.data.body_linear_velocity_b[:] = quat_rotate_inverse(
                agent_states.data.body_orientation_w,
                agent_states.data.body_linear_velocity_w,
            )
            agent_states.data.body_angular_velocity_b[:] = quat_rotate_inverse(
                agent_states.data.body_orientation_w,
                agent_states.data.body_angular_velocity_w,
            )
            agent_states.data.body_linear_acceleration_b[:] = quat_rotate_inverse(
                agent_states.data.body_orientation_w,
                agent_states.data.body_linear_acceleration_w,
            )

            # Compute combined angular velocity
            combined_w, combined_b = compute_combined_angular_velocity(
                agent_states.data.body_angular_velocity_w,
                agent_states.data.body_angular_velocity_b,
                agent_states.data.joint_velocities_b,
                agent_states.data.body_orientation_w,
                agent_states.data.joint_positions_b,
            )
            agent_states.data.body_combined_angular_velocity_w[:] = combined_w
            agent_states.data.body_combined_angular_velocity_b[:] = combined_b

            # Compute derived camera fields
            if agent_id in self._camera_configs:
                cam_cfg = self._camera_configs[agent_id]
                offset_pos = cam_cfg["offset_position_b"].expand(self._num_envs, -1)
                offset_rot = cam_cfg["offset_rotation_b"].expand(self._num_envs, -1)

                # Store camera config in agent_states
                agent_states.data.camera_offset_position_b[:] = offset_pos
                agent_states.data.camera_offset_rotation_b[:] = offset_rot
                agent_states.data.camera_base_intrinsics[:] = cam_cfg["intrinsics"]

                # Compute camera position
                agent_states.data.camera_position_w[:] = compute_camera_position(
                    agent_states.data.body_position_w,
                    agent_states.data.body_orientation_w,
                    offset_pos,
                )

                # Compute camera orientation
                agent_states.data.camera_orientation_w[:] = compute_camera_orientation_from_gimbal(
                    agent_states.data.body_orientation_w,
                    agent_states.data.joint_positions_b,
                    offset_rot,
                )

                # Compute ray origins
                agent_states.data.camera_ray_origins_w[:] = compute_ray_origins(
                    agent_states.data.camera_position_w,
                    num_targets,
                )

                # Compute ray directions from bboxes
                agent_states.data.camera_ray_directions_w[:] = compute_ray_directions_from_bbox(
                    agent_states.data.camera_orientation_w,
                    cam_cfg["intrinsics"],
                    agent_states.data.camera_zoom_level,
                    agent_states.data.bboxes_2d,
                )

            # Timestamps
            # timestamp_motion: Use current time (motion data is continuously streamed)
            agent_states.data.timestamp_motion[:] = self._current_time
            # timestamp_detection: Extracted from the coupled detection pipeline above.
            # This guarantees the timestamp reflects exactly when the bbox was captured,
            # with identical staleness, latency, and dropout as the bbox data.
            agent_states.data.timestamp_detection[:] = detection_timestamp

            result[agent_id] = agent_states

        return result

    def reset(
        self,
        env_ids: torch.Tensor,
        initial_gt_states: Optional[Dict[AgentID, Any]] = None,
        noise_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Reset delay system for specified environments.

        Resamples per-environment noise standard deviations for domain randomization.
        Each environment gets noise sampled from Uniform(0, sigma_max * noise_progress).

        Args:
            env_ids: Environment indices to reset.
            initial_gt_states: Optional initial GT states to warm-start filters.
            noise_progress: Curriculum progress in [0, 1] for noise scaling.
                0 = no noise, 1 = full noise range.

        Returns:
            Dictionary of sampled noise std values for the reset environments.
            Keys: "position", "orientation", "gimbal", "bbox", etc.
            Values: Tensor of shape [len(env_ids)] with sampled std per env.
        """
        # Reset simulation time for these environments
        self._current_time[env_ids] = 0.0

        # Resample noise standard deviations for reset environments
        sampled_noise = self.resample_noise_std(env_ids, progress=noise_progress)

        # Reset all delay systems
        self._delay_clean_ego.reset(env_ids)
        self._delay_clean_other.reset(env_ids)
        self._delay_noisy_ego.reset(env_ids)
        self._delay_noisy_other.reset(env_ids)

        # Reset GT states
        for agent_id in self._possible_agents:
            gt_state = self._gt_states[agent_id]

            # Reset to zeros or initial values if provided
            if initial_gt_states and agent_id in initial_gt_states:
                init_state = initial_gt_states[agent_id]
                if hasattr(init_state, 'data'):
                    # Copy initial data
                    gt_state.data.body_position_w[env_ids] = init_state.data.body_position_w[env_ids]
                    gt_state.data.body_orientation_w[env_ids] = init_state.data.body_orientation_w[env_ids]
                    gt_state.data.body_linear_velocity_w[env_ids] = init_state.data.body_linear_velocity_w[env_ids]
                    gt_state.data.body_angular_velocity_w[env_ids] = init_state.data.body_angular_velocity_w[env_ids]
                    gt_state.data.joint_positions_b[env_ids] = init_state.data.joint_positions_b[env_ids]
                    if hasattr(init_state.data, 'camera_zoom_level'):
                        gt_state.data.camera_zoom_level[env_ids] = init_state.data.camera_zoom_level[env_ids].squeeze()
                    if hasattr(init_state.data, 'body_linear_acceleration_w'):
                        gt_state.data.body_linear_acceleration_w[env_ids] = init_state.data.body_linear_acceleration_w[env_ids]
                    else:
                        gt_state.data.body_linear_acceleration_w[env_ids] = 0.0
            else:
                # Reset to default values
                gt_state.data.body_position_w[env_ids] = 0.0
                gt_state.data.body_orientation_w[env_ids, 0] = 1.0  # w=1 for identity
                gt_state.data.body_orientation_w[env_ids, 1:] = 0.0
                gt_state.data.body_linear_velocity_w[env_ids] = 0.0
                gt_state.data.body_angular_velocity_w[env_ids] = 0.0
                gt_state.data.body_linear_acceleration_w[env_ids] = 0.0
                gt_state.data.joint_positions_b[env_ids] = 0.0
                gt_state.data.camera_zoom_level[env_ids] = 1.0
                gt_state.data.bboxes_2d[env_ids] = 0.0

            # Reset timestamps
            gt_state.data.timestamp_sim_walltime[env_ids] = 0.0
            gt_state.data.timestamp_motion[env_ids] = 0.0
            gt_state.data.timestamp_detection[env_ids] = 0.0

        # CRITICAL: Push initial GT states to delay pipelines
        # This ensures the delay systems have data available for get_delayed_* calls
        self._initialize_pipelines_from_gt_states(env_ids)

        return sampled_noise

    def set_noise_progress_scale(self, progress: float):
        """Set noise progress scale for curriculum learning.

        Args:
            progress: Progress value in [0, 1]. 0 = no noise, 1 = full noise.

        Note: This is deprecated in favor of per-env noise sampling via resample_noise_std().
        Kept for backward compatibility.
        """
        self._noise_progress_scale = progress if self.cfg.enable_noise else 0.0

    def resample_noise_std(
        self,
        env_ids: torch.Tensor,
        progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Resample noise standard deviations for specified environments.

        For curriculum learning, noise is sampled uniformly:
            sigma ~ Uniform(0, sigma_max * progress)

        This provides domain randomization where each environment has its own
        noise level, with the maximum noise scaling with curriculum progress.

        Args:
            env_ids: Environment indices to resample.
            progress: Curriculum progress in [0, 1]. Scales maximum noise.

        Returns:
            Dictionary mapping field names to sampled std values for the reset
            environments. Shape: [len(env_ids)] per field.
        """
        if not self.cfg.enable_noise:
            # Return zeros if noise disabled
            for key in self._per_env_noise_std:
                self._per_env_noise_std[key][env_ids] = 0.0
            return {k: self._per_env_noise_std[k][env_ids].clone()
                    for k in self._per_env_noise_std}

        num_reset = len(env_ids)
        cfg = self.cfg

        # Map field names to max std values from config
        max_std_map = {
            "position": cfg.position_noise_std,
            "orientation": cfg.orientation_noise_std,
            "linear_velocity": cfg.linear_velocity_noise_std,
            "angular_velocity": cfg.angular_velocity_noise_std,
            "linear_acceleration": cfg.linear_acceleration_noise_std,
            "gimbal": cfg.gimbal_noise_std,
            "bbox": cfg.bbox_noise_std,
            "zoom": cfg.zoom_noise_std,
        }

        sampled = {}
        for key, max_std in max_std_map.items():
            # Sample uniformly from [0, max_std * progress]
            max_value = max_std * progress
            samples = torch.rand(num_reset, device=self._device) * max_value
            self._per_env_noise_std[key][env_ids] = samples
            sampled[key] = samples.clone()

        return sampled

    def get_noise_std(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Get current noise standard deviations for specified environments.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Dictionary mapping noise field names to their current std values.
        """
        if env_ids is None:
            return {k: v.clone() for k, v in self._per_env_noise_std.items()}
        return {k: v[env_ids].clone() for k, v in self._per_env_noise_std.items()}

    # ==========================================================================
    # Curriculum Learning API for Phase 3/4
    # ==========================================================================

    def set_delay_mode(self, mode: str, progress: float = 1.0):
        """Set delay mode for all pipelines (curriculum learning).

        This method controls how latency delays are applied:
        - 'none': No latency delay (Phase 0-2)
        - 'fixed': Fixed deterministic delay, all envs same (Phase 3)
        - 'random': Random delay sampled per environment (Phase 4+)

        The progress parameter ramps the delay magnitude:
        - In 'fixed' mode: delay = config_mean * progress
        - In 'random' mode: delay ~ N(config_mean * progress, config_std * progress)

        Args:
            mode: Delay mode - 'none', 'fixed', or 'random'
            progress: Curriculum progress [0, 1] for ramping delay magnitude/variance
        """
        # Apply to all 4 delay systems
        self._delay_clean_ego.set_all_delay_modes(mode, progress)
        self._delay_clean_other.set_all_delay_modes(mode, progress)
        self._delay_noisy_ego.set_all_delay_modes(mode, progress)
        self._delay_noisy_other.set_all_delay_modes(mode, progress)

    def set_dropout_enabled(self, enabled: bool, progress: float = 0.0):
        """Enable/disable dropout with curriculum scaling.

        Args:
            enabled: Whether dropout is enabled
            progress: Curriculum progress [0, 1] for ramping dropout rate
                When enabled, dropout_rate = config_rate * progress
        """
        if enabled and progress > 0:
            # Ramp dropout rate based on progress
            # Use original config dropout rates scaled by progress
            cfg = self.cfg
            detection_dropout = cfg.detection_dropout_rate * progress
            comm_dropout = cfg.inter_agent_comm_dropout_rate * progress

            # Apply scaled dropout to all noisy pipelines
            self._delay_noisy_ego.set_all_dropout_rates(detection_dropout)
            self._delay_noisy_other.set_all_dropout_rates(min(1.0, detection_dropout + comm_dropout))
        else:
            # Disable dropout
            self._delay_noisy_ego.set_all_dropout_rates(0.0)
            self._delay_noisy_other.set_all_dropout_rates(0.0)

        # Clean pipelines never have dropout
        self._delay_clean_ego.set_all_dropout_rates(0.0)
        self._delay_clean_other.set_all_dropout_rates(0.0)

    def _initialize_pipelines_from_gt_states(self, env_ids: torch.Tensor | None = None):
        """Initialize delay pipelines with current GT states.

        This is called during reset to ensure the delay systems have data
        available for immediate retrieval. Without this, the first call to
        get_delayed_* would fail with KeyError.

        Args:
            env_ids: Environment indices that were reset. If provided, seeds the
                dropout held data for noisy pipelines so that hold-last-value
                dropout returns the initial GT state rather than zeros after reset.
        """
        for agent_id in self._possible_agents:
            gt_state = self._gt_states[agent_id]

            # Push GT states to delay pipelines (both perspectives)
            self._store_to_pipelines(
                agent_id, "ego",
                gt_state.data.body_position_w,
                gt_state.data.body_orientation_w,
                gt_state.data.body_linear_velocity_w,
                gt_state.data.body_angular_velocity_w,
                gt_state.data.body_linear_acceleration_w,
                gt_state.data.joint_positions_b,
                gt_state.data.camera_zoom_level,
            )
            self._store_to_pipelines(
                agent_id, "other",
                gt_state.data.body_position_w,
                gt_state.data.body_orientation_w,
                gt_state.data.body_linear_velocity_w,
                gt_state.data.body_angular_velocity_w,
                gt_state.data.body_linear_acceleration_w,
                gt_state.data.joint_positions_b,
                gt_state.data.camera_zoom_level,
            )

            # Push bbox data with coupled timestamp_detection
            bboxes_flat = gt_state.data.bboxes_2d.flatten(start_dim=1)  # [N, T*4]
            timestamp_data = self._current_time.unsqueeze(-1)  # [N, 1]
            concat_data = torch.cat([bboxes_flat, timestamp_data], dim=-1)  # [N, T*4+1]
            for perspective in ["ego", "other"]:
                if perspective == "ego":
                    delay_clean = self._delay_clean_ego
                    delay_noisy = self._delay_noisy_ego
                else:
                    delay_clean = self._delay_clean_other
                    delay_noisy = self._delay_noisy_other

                full_name = f"{agent_id}.bboxes_2d.{perspective}"
                delay_clean.store(full_name, concat_data)
                delay_noisy.store(full_name, concat_data)

        # Step delay systems to process the initial data
        # This ensures data is available for immediate retrieval
        dt = self.cfg.dt
        self._delay_clean_ego.step(dt)
        self._delay_clean_other.step(dt)
        self._delay_noisy_ego.step(dt)
        self._delay_noisy_other.step(dt)

        # Seed dropout held data with GT initial state for the reset environments.
        # Without this, dropout_held_data stays at 0.0 after reset (set by pipeline.reset()),
        # causing hold-last-value dropout to return zeros instead of a valid initial snapshot.
        # Only affects envs whose dropout_held_data was already initialized (episode 2+).
        if env_ids is not None:
            self._delay_noisy_ego.seed_dropout_held_data(env_ids)
            self._delay_noisy_other.seed_dropout_held_data(env_ids)


class GTStatesAccessor:
    """Accessor class for ground truth states (for API compatibility)."""

    def __init__(self, gt_states: Dict[AgentID, AgentStates]):
        self._gt_states = gt_states

    @property
    def agents(self) -> Dict[AgentID, AgentStates]:
        """Access to all agents' GT states."""
        return self._gt_states

    def __getitem__(self, agent_id: AgentID) -> AgentStates:
        """Get GT states for a specific agent."""
        return self._gt_states[agent_id]
