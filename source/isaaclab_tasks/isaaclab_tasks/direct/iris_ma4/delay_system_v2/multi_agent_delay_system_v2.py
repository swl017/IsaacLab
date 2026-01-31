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
from isaaclab.utils.math import quat_rotate_inverse, normalize

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
    # Timestamp fields (delayed identically to their associated data)
    RAW_TIMESTAMP_FIELDS = [
        "timestamp_detection",  # Timestamp when detection data was captured
    ]
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

        # Noise progress scale (for curriculum learning)
        self._noise_progress_scale = 1.0 if cfg.enable_noise else 0.0

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

                # Detection fields (flattened)
                for field in self.RAW_DETECTION_FIELDS:
                    dims[f"{agent_id}.{field}.{perspective}"] = num_targets * 4  # [T, 4] flattened

                # Zoom field
                for field in self.RAW_ZOOM_FIELDS:
                    dims[f"{agent_id}.{field}.{perspective}"] = 1

                # Timestamp fields (scalar values)
                for field in self.RAW_TIMESTAMP_FIELDS:
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
            detection_cfg_ego = FieldDelayCfg(
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="normal",
                    mean=cfg.detection_fps_mean,
                    std=cfg.detection_fps_std,
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

            # Timestamp fields: same delay as detection, but NO first-order lag
            # Timestamps should not be smoothed, but should be delayed identically
            timestamp_cfg_ego = FieldDelayCfg(
                first_order_lag_enabled=False,  # No smoothing for timestamps
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="normal",
                    mean=cfg.detection_fps_mean,
                    std=cfg.detection_fps_std,
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
            for field in self.RAW_TIMESTAMP_FIELDS:
                ego_configs[f"{agent_id}.{field}.ego"] = timestamp_cfg_ego

            # === Other perspective (slow inter-agent communication) ===
            # Motion fields: first-order lag + staleness + latency + dropout
            motion_cfg_other = FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=cfg.motion_time_constant,
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="normal",
                    mean=cfg.inter_agent_comm_fps_mean,
                    std=cfg.inter_agent_comm_fps_std,
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
                    type="normal",
                    mean=cfg.inter_agent_comm_fps_mean,
                    std=cfg.inter_agent_comm_fps_std,
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
                    type="normal",
                    mean=cfg.inter_agent_comm_fps_mean,
                    std=cfg.inter_agent_comm_fps_std,
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
            detection_cfg_other = FieldDelayCfg(
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="normal",
                    mean=cfg.detection_fps_mean,
                    std=cfg.detection_fps_std,
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
                    type="normal",
                    mean=cfg.inter_agent_comm_fps_mean,
                    std=cfg.inter_agent_comm_fps_std,
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

            # Timestamp fields: same delay as detection, but NO first-order lag
            # Timestamps should not be smoothed, but should be delayed identically
            timestamp_cfg_other = FieldDelayCfg(
                first_order_lag_enabled=False,  # No smoothing for timestamps
                staleness_enabled=True,
                sample_rate=DistributionCfg(
                    type="normal",
                    mean=cfg.detection_fps_mean,
                    std=cfg.detection_fps_std,
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
            for field in self.RAW_TIMESTAMP_FIELDS:
                other_configs[f"{agent_id}.{field}.other"] = timestamp_cfg_other

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
    ):
        """Update ground truth states for an agent.

        This stores the raw states and feeds them to the delay pipelines.

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
        """
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
        """Store states to clean and noisy pipelines."""
        cfg = self.cfg
        scale = self._noise_progress_scale

        # Select appropriate delay systems based on perspective
        if perspective == "ego":
            delay_clean = self._delay_clean_ego
            delay_noisy = self._delay_noisy_ego
        else:
            delay_clean = self._delay_clean_other
            delay_noisy = self._delay_noisy_other

        # Motion fields
        fields_data = {
            "body_position_w": (body_position_w, cfg.position_noise_std),
            "body_linear_velocity_w": (body_linear_velocity_w, cfg.linear_velocity_noise_std),
            "body_angular_velocity_w": (body_angular_velocity_w, cfg.angular_velocity_noise_std),
            "body_linear_acceleration_w": (body_linear_acceleration_w, cfg.linear_acceleration_noise_std),
        }

        for field_name, (data, noise_std) in fields_data.items():
            full_name = f"{agent_id}.{field_name}.{perspective}"
            # Clean pipeline
            delay_clean.store(full_name, data)
            # Noisy pipeline (with noise)
            if cfg.enable_noise and noise_std > 0 and scale > 0:
                noisy_data = data + torch.randn_like(data) * noise_std * scale
                delay_noisy.store(full_name, noisy_data)
            else:
                delay_noisy.store(full_name, data)

        # Orientation field (special handling for quaternions)
        full_name = f"{agent_id}.body_orientation_w.{perspective}"
        delay_clean.store(full_name, body_orientation_w)
        if cfg.enable_noise and cfg.orientation_noise_std > 0 and scale > 0:
            # Add noise to quaternion and normalize
            noisy_quat = body_orientation_w + torch.randn_like(body_orientation_w) * cfg.orientation_noise_std * scale
            noisy_quat = normalize(noisy_quat)
            delay_noisy.store(full_name, noisy_quat)
        else:
            delay_noisy.store(full_name, body_orientation_w)

        # Joint fields
        full_name = f"{agent_id}.joint_positions_b.{perspective}"
        delay_clean.store(full_name, joint_positions_b)
        if cfg.enable_noise and cfg.gimbal_noise_std > 0 and scale > 0:
            noisy_joints = joint_positions_b + torch.randn_like(joint_positions_b) * cfg.gimbal_noise_std * scale
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
        if cfg.enable_noise and cfg.zoom_noise_std > 0 and scale > 0:
            noisy_zoom = zoom_expanded + torch.randn_like(zoom_expanded) * cfg.zoom_noise_std * scale
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

        Args:
            agent_id: Agent identifier.
            bboxes_2d_gt: Ground truth bounding boxes [N, T, 4] (x, y, w, h).
        """
        cfg = self.cfg
        scale = self._noise_progress_scale

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

        # Store to both perspectives
        for perspective in ["ego", "other"]:
            if perspective == "ego":
                delay_clean = self._delay_clean_ego
                delay_noisy = self._delay_noisy_ego
            else:
                delay_clean = self._delay_clean_other
                delay_noisy = self._delay_noisy_other

            full_name = f"{agent_id}.bboxes_2d.{perspective}"
            delay_clean.store(full_name, bboxes_flat)

            if cfg.enable_noise and cfg.bbox_noise_std > 0 and scale > 0:
                noisy_bbox = bboxes_flat + torch.randn_like(bboxes_flat) * cfg.bbox_noise_std * scale
                delay_noisy.store(full_name, noisy_bbox)
            else:
                delay_noisy.store(full_name, bboxes_flat)

            # Store timestamp_detection alongside bboxes (same delay pipeline)
            # This ensures the timestamp is delayed identically to the detection data
            timestamp_field_name = f"{agent_id}.timestamp_detection.{perspective}"
            timestamp_data = self._current_time.unsqueeze(-1)  # [N] -> [N, 1]
            delay_clean.store(timestamp_field_name, timestamp_data)
            delay_noisy.store(timestamp_field_name, timestamp_data)  # No noise for timestamps

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

            # Orientation (normalize after filtering with zero-quaternion protection)
            body_orientation = get_field("body_orientation_w")
            body_orientation_norm = normalize(body_orientation)
            # Detect invalid quaternions (zero or near-zero norm) and replace with identity
            quat_norm = body_orientation.norm(dim=-1, keepdim=True)
            invalid_quat_mask = quat_norm < 1e-6  # [N, 1]
            if invalid_quat_mask.any():
                # Create identity quaternion [1, 0, 0, 0] for invalid entries
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

            # Detection field (unflatten)
            bboxes_flat = get_field("bboxes_2d")
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
            # timestamp_detection: Retrieve from delay pipeline (delayed with detection data)
            # This ensures the timestamp reflects when the detection data was captured
            delayed_timestamp = get_field("timestamp_detection")
            agent_states.data.timestamp_detection[:] = delayed_timestamp.squeeze(-1)

            result[agent_id] = agent_states

        return result

    def reset(
        self,
        env_ids: torch.Tensor,
        initial_gt_states: Optional[Dict[AgentID, Any]] = None,
    ):
        """Reset delay system for specified environments.

        Args:
            env_ids: Environment indices to reset.
            initial_gt_states: Optional initial GT states to warm-start filters.
        """
        # Reset simulation time for these environments
        self._current_time[env_ids] = 0.0

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
        self._initialize_pipelines_from_gt_states()

    def set_noise_progress_scale(self, progress: float):
        """Set noise progress scale for curriculum learning.

        Args:
            progress: Progress value in [0, 1]. 0 = no noise, 1 = full noise.
        """
        self._noise_progress_scale = progress if self.cfg.enable_noise else 0.0

    def _initialize_pipelines_from_gt_states(self):
        """Initialize delay pipelines with current GT states.

        This is called during reset to ensure the delay systems have data
        available for immediate retrieval. Without this, the first call to
        get_delayed_* would fail with KeyError.
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

            # Push bbox data and timestamp_detection
            bboxes_flat = gt_state.data.bboxes_2d.flatten(start_dim=1)
            timestamp_data = self._current_time.unsqueeze(-1)  # [N] -> [N, 1]
            for perspective in ["ego", "other"]:
                if perspective == "ego":
                    delay_clean = self._delay_clean_ego
                    delay_noisy = self._delay_noisy_ego
                else:
                    delay_clean = self._delay_clean_other
                    delay_noisy = self._delay_noisy_other

                # Store bboxes
                full_name = f"{agent_id}.bboxes_2d.{perspective}"
                delay_clean.store(full_name, bboxes_flat)
                delay_noisy.store(full_name, bboxes_flat)

                # Store timestamp_detection (initialized to 0, same as current_time at init)
                timestamp_field_name = f"{agent_id}.timestamp_detection.{perspective}"
                delay_clean.store(timestamp_field_name, timestamp_data)
                delay_noisy.store(timestamp_field_name, timestamp_data)

        # Step delay systems to process the initial data
        # This ensures data is available for immediate retrieval
        dt = self.cfg.dt
        self._delay_clean_ego.step(dt)
        self._delay_clean_other.step(dt)
        self._delay_noisy_ego.step(dt)
        self._delay_noisy_other.step(dt)


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
