# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-agent delay system wrapper V3.

This module provides the high-level multi-agent interface for the delay system,
with full curriculum learning support. It wraps the UnifiedDelaySystem and
provides convenient methods for multi-agent state management.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple, Literal, Any

from isaaclab.envs.common import AgentID
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .delay_cfg_v3 import MultiAgentDelayCfgV3, NoiseCfg
from .delay_system_v3 import UnifiedDelaySystem
from .agent_states import AgentStates, AgentStatesData
from .derived_field_computers import compute_ray_origins, compute_ray_directions_from_bbox


# Standard field definitions
BODY_FIELDS = [
    ("body_position_w", (3,)),
    ("body_orientation_w", (4,)),  # quaternion wxyz
    ("body_linear_velocity_w", (3,)),
    ("body_angular_velocity_w", (3,)),
    ("body_angular_velocity_b", (3,)),
    ("body_linear_acceleration_b", (3,)),
]

JOINT_FIELDS = [
    ("joint_positions_b", None),  # Shape depends on num_joints
    ("joint_velocities_b", None),
]

CAMERA_FIELDS = [
    ("camera_position_w", (3,)),
    ("camera_orientation_w", (4,)),
    ("camera_zoom_level", (1,)),  # zoom level scalar per env
]

GIMBAL_WORLD_FIELDS = [
    ("gimbal_azimuth_world", (1,)),
    ("gimbal_elevation_world", (1,)),
    ("gimbal_azimuth_rate", (1,)),
    ("gimbal_elevation_rate", (1,)),
]

DETECTION_FIELDS = [
    ("bboxes_2d", None),  # Shape depends on num_targets: (num_targets, 4)
]


class MultiAgentDelaySystemV3:
    """Multi-agent delay system with curriculum learning support.

    This system:
    1. Manages state storage and delay for multiple agents
    2. Provides separate paths for rewards (clean) and observations (noisy)
    3. Supports curriculum learning with mode/progress control

    **Key API methods:**
    - update_ground_truth(): Store current ground truth states
    - get_all_states_for_rewards(): Get clean delayed states for reward computation
    - get_all_states_for_observations(): Get noisy delayed states for observations
    - set_delay_mode(): Control curriculum delay mode
    """

    def __init__(
        self,
        cfg: MultiAgentDelayCfgV3,
        possible_agents: List[AgentID],
        num_envs: int,
        num_joints: int,
        num_targets: int,
        device: torch.device,
    ):
        """Initialize multi-agent delay system.

        Args:
            cfg: Configuration for the delay system.
            possible_agents: List of possible agent IDs.
            num_envs: Number of parallel environments.
            num_joints: Number of gimbal joints per agent.
            num_targets: Number of detection targets per agent.
            device: Torch device.
        """
        self._cfg = cfg
        self._possible_agents = list(possible_agents)
        self._num_envs = num_envs
        self._num_joints = num_joints
        self._num_targets = num_targets
        self._device = device

        # Create underlying unified delay system
        self._delay_system = UnifiedDelaySystem(
            cfg=cfg.delay_cfg,
            num_envs=num_envs,
            device=device,
        )

        # Ground truth states (no delay)
        self._gt_states: Dict[AgentID, AgentStates] = {}
        for agent_id in possible_agents:
            self._gt_states[agent_id] = AgentStates(
                num_envs, num_joints, num_targets, device
            )

        # Cached delayed states for rewards and observations
        self._cached_reward_states: Dict[AgentID, Dict[AgentID, AgentStates]] = {}
        self._cached_obs_states: Dict[AgentID, Dict[AgentID, AgentStates]] = {}

        # Register fields
        self._register_fields()

        # Noise configuration
        self._noise_cfg = cfg.noise
        self._noise_scale = 0.0  # Start with no noise; curriculum ramps via set_noise_scale()

        # Current time
        self._t_current = torch.zeros(num_envs, device=device)

        # Curriculum state
        self._mode: Literal["none", "fixed", "random"] = "none"
        self._progress: float = 1.0

        # Per-agent heterogeneity scales (sampled at episode reset)
        self._per_agent_latency_scale: Dict[AgentID, float] = {
            a: 1.0 for a in possible_agents
        }
        self._per_agent_noise_scale: Dict[AgentID, float] = {
            a: 1.0 for a in possible_agents
        }
        self._per_agent_dropout_offset: Dict[AgentID, float] = {
            a: 0.0 for a in possible_agents
        }
        # Base dropout rate from curriculum (before per-agent offsets)
        self._base_dropout_rate: float = 0.0

    def _register_fields(self):
        """Register all fields for delay processing."""
        for agent_id in self._possible_agents:
            # Body fields
            for field_name, shape in BODY_FIELDS:
                key = f"{agent_id}.{field_name}"
                self._delay_system.register_field(key, shape)

            # Joint fields
            for field_name, _ in JOINT_FIELDS:
                key = f"{agent_id}.{field_name}"
                shape = (self._num_joints,)
                self._delay_system.register_field(key, shape)

            # Gimbal world-frame fields
            for field_name, shape in GIMBAL_WORLD_FIELDS:
                key = f"{agent_id}.{field_name}"
                self._delay_system.register_field(key, shape)

            # Camera fields
            for field_name, shape in CAMERA_FIELDS:
                key = f"{agent_id}.{field_name}"
                self._delay_system.register_field(key, shape)

            # Detection fields
            for field_name, _ in DETECTION_FIELDS:
                key = f"{agent_id}.{field_name}"
                shape = (self._num_targets, 4)
                self._delay_system.register_field(key, shape)

    @property
    def possible_agents(self) -> List[AgentID]:
        """List of possible agent IDs."""
        return self._possible_agents

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    @property
    def delay_system(self) -> UnifiedDelaySystem:
        """Underlying unified delay system."""
        return self._delay_system

    @property
    def gt_states(self) -> Dict[AgentID, AgentStates]:
        """Ground truth states (no delay)."""
        return self._gt_states

    def set_time(self, t: torch.Tensor):
        """Set current simulation time.

        Args:
            t: Current time tensor.
        """
        self._t_current = t.to(self._device)
        self._delay_system.set_time(t)

    def update_ground_truth(self, agent_id: AgentID, states: AgentStates):
        """Update ground truth state for an agent.

        This stores the current state and pushes it through the delay system.

        Args:
            agent_id: Agent identifier.
            states: Current ground truth state.
        """
        # Store ground truth
        self._gt_states[agent_id] = states

        # Get noise standard deviations (per-agent scaled)
        pos_noise = self._get_noise_std("position", agent_id)
        vel_noise = self._get_noise_std("velocity", agent_id)
        ori_noise = self._get_noise_std("orientation", agent_id)
        ang_vel_noise = self._get_noise_std("angular_velocity", agent_id)
        acc_noise = self._get_noise_std("acceleration", agent_id)

        # Store fields in delay system
        data = states.data
        prefix = f"{agent_id}."

        # Body fields with appropriate noise
        # NOTE: Quaternion orientation stored with noise_std=0 — rotation noise
        # applied properly via axis-angle perturbation in _add_noise_to_states()
        self._delay_system.store(
            prefix + "body_position_w",
            data.body_position_w,
            noise_std=pos_noise,
        )
        self._delay_system.store(
            prefix + "body_orientation_w",
            data.body_orientation_w,
            noise_std=0.0,
        )
        self._delay_system.store(
            prefix + "body_linear_velocity_w",
            data.body_linear_velocity_w,
            noise_std=vel_noise,
        )
        self._delay_system.store(
            prefix + "body_angular_velocity_w",
            data.body_angular_velocity_w,
            noise_std=ang_vel_noise,
        )
        self._delay_system.store(
            prefix + "body_angular_velocity_b",
            data.body_angular_velocity_b,
            noise_std=ang_vel_noise,
        )
        self._delay_system.store(
            prefix + "body_linear_acceleration_b",
            data.body_linear_acceleration_b,
            noise_std=acc_noise,
        )

        # Joint fields
        self._delay_system.store(
            prefix + "joint_positions_b",
            data.joint_positions_b,
            noise_std=ori_noise,
        )
        self._delay_system.store(
            prefix + "joint_velocities_b",
            data.joint_velocities_b,
            noise_std=vel_noise,
        )

        # Gimbal world-frame fields
        self._delay_system.store(
            prefix + "gimbal_azimuth_world",
            data.gimbal_azimuth_world.unsqueeze(-1),
            noise_std=ori_noise,
        )
        self._delay_system.store(
            prefix + "gimbal_elevation_world",
            data.gimbal_elevation_world.unsqueeze(-1),
            noise_std=ori_noise,
        )
        self._delay_system.store(
            prefix + "gimbal_azimuth_rate",
            data.gimbal_azimuth_rate.unsqueeze(-1),
            noise_std=vel_noise,
        )
        self._delay_system.store(
            prefix + "gimbal_elevation_rate",
            data.gimbal_elevation_rate.unsqueeze(-1),
            noise_std=vel_noise,
        )

        # Camera fields
        self._delay_system.store(
            prefix + "camera_position_w",
            data.camera_position_w,
            noise_std=pos_noise,
        )
        self._delay_system.store(
            prefix + "camera_orientation_w",
            data.camera_orientation_w,
            noise_std=0.0,  # Rotation noise applied in _add_noise_to_states()
        )
        # Zoom level (no noise on discrete camera setting)
        self._delay_system.store(
            prefix + "camera_zoom_level",
            data.camera_zoom_level.unsqueeze(-1),  # Convert [N] to [N, 1]
            noise_std=0.0,
        )

        # Detection fields (bbox)
        # Flatten bbox for storage: (N, T, 4) -> (N, T*4)
        # Actually, we keep it as (N, T, 4) since we registered it with that shape
        bbox_noise = self._get_noise_std("bbox", agent_id)
        self._delay_system.store(
            prefix + "bboxes_2d",
            data.bboxes_2d,
            noise_std=bbox_noise,  # Pixel noise on bbox center/dimensions
        )

    def _get_noise_std(
        self, noise_type: str, agent_id: Optional[AgentID] = None
    ) -> float:
        """Get noise standard deviation for a field type.

        Args:
            noise_type: Type of noise ("position", "velocity", "orientation", "angular_velocity", "acceleration", "bbox").
            agent_id: Optional agent ID for per-agent noise scaling.

        Returns:
            Noise standard deviation scaled by curriculum and per-agent factor.
        """
        if not self._noise_cfg.enabled:
            return 0.0

        if noise_type == "position":
            base_std = self._noise_cfg.position_std.value
        elif noise_type == "velocity":
            base_std = self._noise_cfg.velocity_std.value
        elif noise_type == "orientation":
            base_std = self._noise_cfg.orientation_std.value
        elif noise_type == "angular_velocity":
            base_std = self._noise_cfg.angular_velocity_std.value
        elif noise_type == "acceleration":
            base_std = self._noise_cfg.acceleration_std.value
        elif noise_type == "bbox":
            base_std = self._noise_cfg.bbox_std.value
        else:
            base_std = 0.0

        per_agent_scale = 1.0
        if agent_id is not None and self._cfg.per_agent_randomization:
            per_agent_scale = self._per_agent_noise_scale.get(agent_id, 1.0)

        return base_std * self._noise_scale * per_agent_scale

    def get_all_states_for_rewards(
        self, ego_agent_id: AgentID
    ) -> Dict[AgentID, AgentStates]:
        """Get states for reward computation with configurable delay/noise.

        Behavior controlled by cfg.reward_state_cfg:
        - use_delay=False, use_noise=False: Pure ground-truth (privileged)
        - use_delay=True, use_noise=False: Delayed, clean (default)
        - use_delay=False, use_noise=True: GT with noise
        - use_delay=True, use_noise=True: Full noisy delayed

        Args:
            ego_agent_id: The agent computing rewards.

        Returns:
            Dictionary mapping agent IDs to their states.
        """
        reward_cfg = self._cfg.reward_state_cfg

        if not reward_cfg.use_delay:
            # No delay - return ground truth (optionally with noise)
            return self._build_gt_states(add_noise=reward_cfg.use_noise)
        else:
            # Apply delay pipeline
            return self._build_all_states(ego_agent_id, use_noise=reward_cfg.use_noise)

    def get_all_states_for_observations(
        self, ego_agent_id: AgentID
    ) -> Dict[AgentID, AgentStates]:
        """Get noisy delayed states for observation computation.

        Returns states with:
        - Observation noise applied
        - Dropout applied (some detections may be stale)
        - Delay applied based on ego/other perspective

        Args:
            ego_agent_id: The agent computing observations.

        Returns:
            Dictionary mapping agent IDs to their delayed states.
        """
        return self._build_all_states(ego_agent_id, use_noise=True)

    def _build_all_states(
        self, ego_agent_id: AgentID, use_noise: bool
    ) -> Dict[AgentID, AgentStates]:
        """Build delayed states for all agents from ego's perspective.

        Args:
            ego_agent_id: The requesting agent.
            use_noise: Whether to use noisy path.

        Returns:
            Dictionary of delayed states.
        """
        result: Dict[AgentID, AgentStates] = {}

        for agent_id in self._possible_agents:
            perspective: Literal["ego", "other"] = (
                "ego" if agent_id == ego_agent_id else "other"
            )
            result[agent_id] = self._build_agent_states(
                agent_id, perspective, use_noise
            )

        return result

    def _build_agent_states(
        self,
        agent_id: AgentID,
        perspective: Literal["ego", "other"],
        use_noise: bool,
    ) -> AgentStates:
        """Build delayed states for a single agent.

        Args:
            agent_id: Agent identifier.
            perspective: "ego" or "other".
            use_noise: Whether to use noisy path.

        Returns:
            AgentStates with delayed data.
        """
        # Create new state container
        states = AgentStates(
            self._num_envs,
            self._num_joints,
            self._num_targets,
            self._device,
        )

        prefix = f"{agent_id}."
        data = states.data

        # Allow dropout only for observations, not rewards
        allow_dropout = use_noise

        # Body position
        pos, pos_ts = self._delay_system.get_delayed(
            prefix + "body_position_w", perspective, use_noise, allow_dropout
        )
        data.body_position_w = pos

        # Body orientation
        ori, ori_ts = self._delay_system.get_delayed(
            prefix + "body_orientation_w", perspective, use_noise, allow_dropout
        )
        data.body_orientation_w = ori

        # Body velocity
        lin_vel, _ = self._delay_system.get_delayed(
            prefix + "body_linear_velocity_w", perspective, use_noise, allow_dropout
        )
        data.body_linear_velocity_w = lin_vel

        ang_vel, _ = self._delay_system.get_delayed(
            prefix + "body_angular_velocity_w", perspective, use_noise, allow_dropout
        )
        data.body_angular_velocity_w = ang_vel

        ang_vel_b, _ = self._delay_system.get_delayed(
            prefix + "body_angular_velocity_b", perspective, use_noise, allow_dropout
        )
        data.body_angular_velocity_b = ang_vel_b

        lin_acc_b, _ = self._delay_system.get_delayed(
            prefix + "body_linear_acceleration_b", perspective, use_noise, allow_dropout
        )
        data.body_linear_acceleration_b = lin_acc_b

        # Joint states
        joint_pos, _ = self._delay_system.get_delayed(
            prefix + "joint_positions_b", perspective, use_noise, allow_dropout
        )
        data.joint_positions_b = joint_pos

        joint_vel, _ = self._delay_system.get_delayed(
            prefix + "joint_velocities_b", perspective, use_noise, allow_dropout
        )
        data.joint_velocities_b = joint_vel

        # Gimbal world-frame angles
        az, _ = self._delay_system.get_delayed(
            prefix + "gimbal_azimuth_world", perspective, use_noise, allow_dropout
        )
        data.gimbal_azimuth_world = az.squeeze(-1)

        el, _ = self._delay_system.get_delayed(
            prefix + "gimbal_elevation_world", perspective, use_noise, allow_dropout
        )
        data.gimbal_elevation_world = el.squeeze(-1)

        az_rate, _ = self._delay_system.get_delayed(
            prefix + "gimbal_azimuth_rate", perspective, use_noise, allow_dropout
        )
        data.gimbal_azimuth_rate = az_rate.squeeze(-1)

        el_rate, _ = self._delay_system.get_delayed(
            prefix + "gimbal_elevation_rate", perspective, use_noise, allow_dropout
        )
        data.gimbal_elevation_rate = el_rate.squeeze(-1)

        # Camera states
        cam_pos, _ = self._delay_system.get_delayed(
            prefix + "camera_position_w", perspective, use_noise, allow_dropout
        )
        data.camera_position_w = cam_pos

        cam_ori, _ = self._delay_system.get_delayed(
            prefix + "camera_orientation_w", perspective, use_noise, allow_dropout
        )
        data.camera_orientation_w = cam_ori

        # Bounding boxes
        bbox, bbox_ts = self._delay_system.get_delayed(
            prefix + "bboxes_2d", perspective, use_noise, allow_dropout
        )
        data.bboxes_2d = bbox

        # Store detection timestamp
        data.timestamp_detection = bbox_ts

        # Store motion timestamp (use position timestamp)
        data.timestamp_motion = pos_ts

        # Copy static fields from ground truth (these don't change at runtime)
        gt = self._gt_states[agent_id]
        data.camera_offset_position_b = gt.data.camera_offset_position_b.clone()
        data.camera_offset_rotation_b = gt.data.camera_offset_rotation_b.clone()
        data.camera_base_intrinsics = gt.data.camera_base_intrinsics.clone()

        # Retrieve delayed zoom level
        zoom, _ = self._delay_system.get_delayed(
            prefix + "camera_zoom_level", perspective, use_noise, allow_dropout
        )
        data.camera_zoom_level = zoom.squeeze(-1)  # Convert [N, 1] back to [N]

        # Compute derived ray geometry from delayed states
        # Ray origins = camera position (expanded to match num_targets)
        data.camera_ray_origins_w = compute_ray_origins(
            camera_position_w=data.camera_position_w,
            num_targets=self._num_targets,
        )

        # Ray directions = unprojected bbox centers in world frame
        data.camera_ray_directions_w = compute_ray_directions_from_bbox(
            camera_orientation_w=data.camera_orientation_w,
            camera_base_intrinsics=data.camera_base_intrinsics,
            camera_zoom_level=data.camera_zoom_level,
            bboxes_2d=data.bboxes_2d,
        )

        return states

    def _build_gt_states(self, add_noise: bool = False) -> Dict[AgentID, AgentStates]:
        """Build ground-truth states for all agents (no delay).

        Args:
            add_noise: If True, add observation noise to GT states.

        Returns:
            Dictionary of ground-truth states (optionally with noise).
        """
        result: Dict[AgentID, AgentStates] = {}
        for agent_id in self._possible_agents:
            if add_noise:
                result[agent_id] = self._clone_with_noise(self._gt_states[agent_id], agent_id)
            else:
                result[agent_id] = self._clone_agent_states(self._gt_states[agent_id])
        return result

    def _clone_agent_states(self, states: AgentStates) -> AgentStates:
        """Create a deep copy of agent states.

        Args:
            states: Source states to clone.

        Returns:
            Cloned states with all tensors copied.
        """
        cloned = AgentStates(
            self._num_envs,
            self._num_joints,
            self._num_targets,
            self._device,
        )

        # Copy all data fields that are tensors
        src_data = states.data
        dst_data = cloned.data

        # Body fields
        dst_data.body_position_w = src_data.body_position_w.clone()
        dst_data.body_orientation_w = src_data.body_orientation_w.clone()
        dst_data.body_linear_velocity_w = src_data.body_linear_velocity_w.clone()
        dst_data.body_angular_velocity_w = src_data.body_angular_velocity_w.clone()
        dst_data.body_angular_velocity_b = src_data.body_angular_velocity_b.clone()
        dst_data.body_linear_acceleration_b = src_data.body_linear_acceleration_b.clone()

        # Joint fields
        dst_data.joint_positions_b = src_data.joint_positions_b.clone()
        dst_data.joint_velocities_b = src_data.joint_velocities_b.clone()

        # Gimbal world-frame fields
        dst_data.gimbal_azimuth_world = src_data.gimbal_azimuth_world.clone()
        dst_data.gimbal_elevation_world = src_data.gimbal_elevation_world.clone()
        dst_data.gimbal_azimuth_rate = src_data.gimbal_azimuth_rate.clone()
        dst_data.gimbal_elevation_rate = src_data.gimbal_elevation_rate.clone()

        # Camera fields
        dst_data.camera_position_w = src_data.camera_position_w.clone()
        dst_data.camera_orientation_w = src_data.camera_orientation_w.clone()
        dst_data.camera_offset_position_b = src_data.camera_offset_position_b.clone()
        dst_data.camera_offset_rotation_b = src_data.camera_offset_rotation_b.clone()
        dst_data.camera_base_intrinsics = src_data.camera_base_intrinsics.clone()
        dst_data.camera_zoom_level = src_data.camera_zoom_level.clone()
        dst_data.camera_ray_directions_w = src_data.camera_ray_directions_w.clone()
        dst_data.camera_ray_origins_w = src_data.camera_ray_origins_w.clone()

        # Detection fields
        dst_data.bboxes_2d = src_data.bboxes_2d.clone()

        # Timestamps
        dst_data.timestamp_motion = src_data.timestamp_motion.clone()
        dst_data.timestamp_detection = src_data.timestamp_detection.clone()
        dst_data.timestamp_sim_walltime = src_data.timestamp_sim_walltime.clone()

        return cloned

    def _clone_with_noise(
        self, states: AgentStates, agent_id: Optional[AgentID] = None
    ) -> AgentStates:
        """Clone states and add observation noise.

        Args:
            states: Source states to clone.
            agent_id: Optional agent ID for per-agent noise scaling.

        Returns:
            Cloned states with observation noise added.
        """
        cloned = self._clone_agent_states(states)

        # Get noise standard deviations (per-agent scaled)
        pos_noise = self._get_noise_std("position", agent_id)
        vel_noise = self._get_noise_std("velocity", agent_id)
        ori_noise = self._get_noise_std("orientation", agent_id)
        ang_vel_noise = self._get_noise_std("angular_velocity", agent_id)
        acc_noise = self._get_noise_std("acceleration", agent_id)

        data = cloned.data
        N = data.body_position_w.shape[0]
        device = data.body_position_w.device

        # Position noise (additive Gaussian)
        if pos_noise > 0:
            data.body_position_w = data.body_position_w + torch.randn_like(
                data.body_position_w
            ) * pos_noise
            data.camera_position_w = data.camera_position_w + torch.randn_like(
                data.camera_position_w
            ) * pos_noise

        # Linear velocity noise (additive Gaussian)
        if vel_noise > 0:
            data.body_linear_velocity_w = data.body_linear_velocity_w + torch.randn_like(
                data.body_linear_velocity_w
            ) * vel_noise
            data.joint_velocities_b = data.joint_velocities_b + torch.randn_like(
                data.joint_velocities_b
            ) * vel_noise

        # Angular velocity noise (separate scale, rad/s)
        if ang_vel_noise > 0:
            data.body_angular_velocity_w = data.body_angular_velocity_w + torch.randn_like(
                data.body_angular_velocity_w
            ) * ang_vel_noise
            data.body_angular_velocity_b = data.body_angular_velocity_b + torch.randn_like(
                data.body_angular_velocity_b
            ) * ang_vel_noise

        # Linear acceleration noise (separate scale, m/s²)
        if acc_noise > 0:
            data.body_linear_acceleration_b = data.body_linear_acceleration_b + torch.randn_like(
                data.body_linear_acceleration_b
            ) * acc_noise

        # Orientation noise — proper axis-angle perturbation preserving unit quaternion
        if ori_noise > 0:
            # Body orientation: random small rotation
            axis = torch.randn(N, 3, device=device)
            axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
            angle = torch.randn(N, device=device) * ori_noise
            noise_quat = quat_from_angle_axis(angle, axis)
            data.body_orientation_w = quat_mul(noise_quat, data.body_orientation_w)

            # Camera orientation: independent random small rotation
            axis_cam = torch.randn(N, 3, device=device)
            axis_cam = axis_cam / (axis_cam.norm(dim=-1, keepdim=True) + 1e-8)
            angle_cam = torch.randn(N, device=device) * ori_noise
            noise_quat_cam = quat_from_angle_axis(angle_cam, axis_cam)
            data.camera_orientation_w = quat_mul(noise_quat_cam, data.camera_orientation_w)

            # Joint positions: additive (gimbal angles in radians, not quaternions)
            data.joint_positions_b = data.joint_positions_b + torch.randn_like(
                data.joint_positions_b
            ) * ori_noise

            # Gimbal world-frame angles: additive noise (radians)
            data.gimbal_azimuth_world = data.gimbal_azimuth_world + torch.randn(
                N, device=device
            ) * ori_noise
            data.gimbal_elevation_world = data.gimbal_elevation_world + torch.randn(
                N, device=device
            ) * ori_noise

        # Gimbal world-frame rates: additive noise (use velocity noise scale)
        if vel_noise > 0:
            data.gimbal_azimuth_rate = data.gimbal_azimuth_rate + torch.randn(
                N, device=device
            ) * vel_noise
            data.gimbal_elevation_rate = data.gimbal_elevation_rate + torch.randn(
                N, device=device
            ) * vel_noise

        return cloned

    def get_detection_aoi(
        self, ego_agent_id: AgentID, target_agent_id: AgentID, use_noise: bool = True
    ) -> torch.Tensor:
        """Get Age-of-Information for detection of a target by ego.

        Args:
            ego_agent_id: The observing agent.
            target_agent_id: The target agent.
            use_noise: Whether to use noisy path.

        Returns:
            AoI tensor of shape (num_envs,).
        """
        perspective: Literal["ego", "other"] = (
            "ego" if target_agent_id == ego_agent_id else "other"
        )
        return self._delay_system.get_aoi(
            f"{target_agent_id}.bboxes_2d", perspective, use_noise
        )

    def set_delay_mode(
        self, mode: Literal["none", "fixed", "random"], progress: float = 1.0
    ):
        """Set delay mode for curriculum learning with per-agent scaling.

        When per_agent_randomization is enabled, each agent's latency progress
        is scaled by its independently sampled latency_scale factor.

        Args:
            mode: Delay mode ('none', 'fixed', 'random').
            progress: Curriculum progress [0, 1].
        """
        self._mode = mode
        self._progress = max(0.0, min(1.0, progress))

        if self._cfg.per_agent_randomization:
            # Apply per-agent latency scaling
            for agent_id in self._possible_agents:
                scale = self._per_agent_latency_scale[agent_id]
                effective_progress = min(progress * scale, 1.0)
                self._delay_system.set_field_delay_mode(
                    str(agent_id), mode, effective_progress
                )
        else:
            self._delay_system.set_delay_mode(mode, progress)

    def set_dropout_rate(self, rate: float):
        """Set dropout rate for curriculum control with per-agent offsets.

        Args:
            rate: Base dropout probability [0, 1].
        """
        self._base_dropout_rate = rate

        if self._cfg.per_agent_randomization:
            for agent_id in self._possible_agents:
                offset = self._per_agent_dropout_offset[agent_id]
                effective_rate = min(rate + offset, 1.0)
                self._delay_system.set_field_dropout_rate(str(agent_id), effective_rate)
        else:
            self._delay_system.set_dropout_rate(rate)

    def set_noise_scale(self, scale: float):
        """Set noise scale for curriculum control.

        Per-agent noise scaling is applied in _get_noise_std() at query time.

        Args:
            scale: Noise scale [0, 1].
        """
        self._noise_scale = max(0.0, min(1.0, scale))

    def randomize_per_agent_params(self, env_ids: Optional[torch.Tensor] = None):
        """Sample independent delay parameters per agent at episode reset.

        Each agent gets its own latency scale, noise scale, and dropout offset,
        creating heterogeneous delay characteristics across the team.

        Args:
            env_ids: Environment indices being reset (unused currently since
                     per-agent params are shared across envs).
        """
        if not self._cfg.per_agent_randomization:
            return

        cfg = self._cfg.per_agent_cfg
        for agent_id in self._possible_agents:
            self._per_agent_latency_scale[agent_id] = (
                torch.empty(1).uniform_(*cfg.latency_scale_range).item()
            )
            self._per_agent_noise_scale[agent_id] = (
                torch.empty(1).uniform_(*cfg.noise_scale_range).item()
            )
            self._per_agent_dropout_offset[agent_id] = (
                torch.empty(1).uniform_(*cfg.dropout_offset_range).item()
            )

        # Re-apply current curriculum state with new per-agent scales
        self.set_delay_mode(self._mode, self._progress)
        self.set_dropout_rate(self._base_dropout_rate)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset system for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._delay_system.reset(env_ids)

        if env_ids is None:
            self._t_current.zero_()
        else:
            self._t_current[env_ids] = 0.0

    def step(self, reset_env_ids: Optional[torch.Tensor] = None):
        """Called each simulation step.

        Args:
            reset_env_ids: Environments that were reset this step.
        """
        self._delay_system.step(reset_env_ids)
