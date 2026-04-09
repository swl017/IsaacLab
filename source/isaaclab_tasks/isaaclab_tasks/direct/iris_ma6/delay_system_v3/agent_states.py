# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent state data structures for multi-agent delay system."""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.common import AgentID


class AgentStatesData:
    """Data container for a single agent's state across all environments.

    All tensors are on the device specified during initialization.
    Shape conventions:
        - N: number of environments
        - J: number of joints (gimbal)
        - T: number of targets
    """
    num_envs: int
    num_joints: int
    num_targets: int
    device: torch.device

    # Timestamps
    timestamp_sim_walltime: torch.Tensor  # [N] in seconds
    timestamp_motion: torch.Tensor  # [N] in seconds
    timestamp_detection: torch.Tensor  # [N] in seconds

    # Body motion (world frame)
    body_position_w: torch.Tensor  # [N, 3]
    body_orientation_w: torch.Tensor  # [N, 4] (w, x, y, z)
    body_linear_velocity_w: torch.Tensor  # [N, 3]
    body_linear_velocity_b: torch.Tensor  # [N, 3]
    body_angular_velocity_w: torch.Tensor  # [N, 3]
    body_angular_velocity_b: torch.Tensor  # [N, 3]
    body_combined_angular_velocity_w: torch.Tensor  # Combined angular velocity (body + gimbal) [N, 3]
    body_combined_angular_velocity_b: torch.Tensor  # Combined angular velocity in body frame [N, 3]
    body_linear_acceleration_w: torch.Tensor  # [N, 3]
    body_linear_acceleration_b: torch.Tensor  # [N, 3]
    body_angular_acceleration_b: torch.Tensor  # [N, 3]

    # Joint states (gimbal)
    joint_positions_b: torch.Tensor  # [N, J]
    joint_velocities_b: torch.Tensor  # [N, J]
    joint_accelerations_b: torch.Tensor  # [N, J]

    # Gimbal world-frame angles (derived from joint positions + body orientation)
    gimbal_azimuth_world: torch.Tensor  # [N] world-frame azimuth (rad)
    gimbal_elevation_world: torch.Tensor  # [N] world-frame elevation (rad)
    gimbal_azimuth_rate: torch.Tensor  # [N] world-frame azimuth rate (rad/s)
    gimbal_elevation_rate: torch.Tensor  # [N] world-frame elevation rate (rad/s)

    # Camera geometry (derived from body + gimbal)
    camera_offset_position_b: torch.Tensor  # [N, 3]
    camera_offset_rotation_b: torch.Tensor  # [N, 4] (w, x, y, z)
    camera_base_intrinsics: torch.Tensor  # [N, 3, 3] - base camera intrinsics matrix K (unzoomed)
    camera_position_w: torch.Tensor  # [N, 3]
    camera_orientation_w: torch.Tensor  # [N, 4] (w, x, y, z)
    camera_ray_directions_w: torch.Tensor  # [N, T, 3]
    camera_ray_origins_w: torch.Tensor  # [N, T, 3]
    camera_zoom_level: torch.Tensor  # zoom level [N]
    camera_effective_hfov: torch.Tensor  # effective horizontal FOV in radians [N]

    # Detection (bbox)
    bboxes_2d: torch.Tensor  # xywh format [N, T, 4]


class AgentStates:
    """Container for agent state data with initialization helpers.

    This class provides:
    - Data storage via AgentStatesData
    - Proper tensor initialization with correct shapes
    - Optional timestamp tracking
    """
    data: AgentStatesData
    timestamps: Optional[Any]  # Will be AgentStatesTimestamps when delay system is used

    def __init__(self, num_envs: int, num_joints: int, num_targets: int, device: torch.device):
        """Initialize agent states with zeros.

        Args:
            num_envs: Number of parallel environments
            num_joints: Number of gimbal joints (typically 3: yaw, pitch, roll)
            num_targets: Number of detection targets
            device: Device for tensor allocation
        """
        self.data = AgentStatesData()
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.num_targets = num_targets
        self.device = device
        self.timestamps = None  # Initialized by delay system if needed
        self._initialize_agent_state()

    def _initialize_agent_state(self):
        """Initialize all state tensors with zeros/identity values."""
        N = self.num_envs
        J = self.num_joints
        T = self.num_targets
        device = self.device

        # Timestamps
        self.data.timestamp_sim_walltime = torch.zeros(N, device=device)
        self.data.timestamp_motion = torch.zeros(N, device=device)
        self.data.timestamp_detection = torch.zeros(N, device=device)

        # Body motion
        self.data.body_position_w = torch.zeros(N, 3, device=device)
        self.data.body_orientation_w = math_utils.default_orientation(N, device=str(device))
        self.data.body_linear_velocity_w = torch.zeros(N, 3, device=device)
        self.data.body_linear_velocity_b = torch.zeros(N, 3, device=device)
        self.data.body_angular_velocity_w = torch.zeros(N, 3, device=device)
        self.data.body_angular_velocity_b = torch.zeros(N, 3, device=device)
        self.data.body_combined_angular_velocity_w = torch.zeros(N, 3, device=device)
        self.data.body_combined_angular_velocity_b = torch.zeros(N, 3, device=device)
        self.data.body_linear_acceleration_w = torch.zeros(N, 3, device=device)
        self.data.body_linear_acceleration_b = torch.zeros(N, 3, device=device)
        self.data.body_angular_acceleration_b = torch.zeros(N, 3, device=device)

        # Joint states
        self.data.joint_positions_b = torch.zeros(N, J, device=device)
        self.data.joint_velocities_b = torch.zeros(N, J, device=device)
        self.data.joint_accelerations_b = torch.zeros(N, J, device=device)

        # Gimbal world-frame angles
        self.data.gimbal_azimuth_world = torch.zeros(N, device=device)
        self.data.gimbal_elevation_world = torch.zeros(N, device=device)
        self.data.gimbal_azimuth_rate = torch.zeros(N, device=device)
        self.data.gimbal_elevation_rate = torch.zeros(N, device=device)

        # Camera geometry
        self.data.camera_offset_position_b = torch.zeros(N, 3, device=device)
        self.data.camera_offset_rotation_b = math_utils.default_orientation(N, device=str(device))
        # Initialize camera intrinsics with proper structure [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.data.camera_base_intrinsics = torch.zeros(N, 3, 3, device=device)
        self.data.camera_base_intrinsics[:, 2, 2] = 1.0  # Bottom-right element must be 1.0
        self.data.camera_position_w = torch.zeros(N, 3, device=device)
        self.data.camera_orientation_w = math_utils.default_orientation(N, device=str(device))
        self.data.camera_ray_directions_w = torch.zeros(N, T, 3, device=device)
        self.data.camera_ray_origins_w = torch.zeros(N, T, 3, device=device)
        self.data.camera_zoom_level = torch.zeros(N, device=device)
        self.data.camera_effective_hfov = torch.zeros(N, device=device)  # radians

        # Detection
        self.data.bboxes_2d = torch.zeros(N, T, 4, device=device)
        # NOTE: bboxes_2d_valid_mask has been REMOVED from the delay pipeline.
        # Users should validate bboxes AFTER delay processing using:
        #     bbox_valid = bbox_raycaster.validate_bbox(bbox)  # [N, T]
        # This avoids confusion between frame-level validity and detection validity.

    def zero_(self):
        """Zero all state tensors in-place without re-allocation.

        WRITE method — call once per step before populating with new data.
        """
        d = self.data
        # Timestamps
        d.timestamp_sim_walltime.zero_()
        d.timestamp_motion.zero_()
        d.timestamp_detection.zero_()
        # Body motion
        d.body_position_w.zero_()
        d.body_orientation_w[:] = 0.0
        d.body_orientation_w[:, 0] = 1.0  # w=1 identity quaternion
        d.body_linear_velocity_w.zero_()
        d.body_linear_velocity_b.zero_()
        d.body_angular_velocity_w.zero_()
        d.body_angular_velocity_b.zero_()
        d.body_combined_angular_velocity_w.zero_()
        d.body_combined_angular_velocity_b.zero_()
        d.body_linear_acceleration_w.zero_()
        d.body_linear_acceleration_b.zero_()
        d.body_angular_acceleration_b.zero_()
        # Joint states
        d.joint_positions_b.zero_()
        d.joint_velocities_b.zero_()
        d.joint_accelerations_b.zero_()
        # Gimbal world-frame angles
        d.gimbal_azimuth_world.zero_()
        d.gimbal_elevation_world.zero_()
        d.gimbal_azimuth_rate.zero_()
        d.gimbal_elevation_rate.zero_()
        # Camera geometry
        d.camera_offset_position_b.zero_()
        d.camera_offset_rotation_b[:] = 0.0
        d.camera_offset_rotation_b[:, 0] = 1.0
        d.camera_position_w.zero_()
        d.camera_orientation_w[:] = 0.0
        d.camera_orientation_w[:, 0] = 1.0
        d.camera_base_intrinsics.zero_()
        d.camera_base_intrinsics[:, 2, 2] = 1.0
        d.camera_ray_directions_w.zero_()
        d.camera_ray_origins_w.zero_()
        d.camera_zoom_level.zero_()
        d.camera_effective_hfov.zero_()
        # Detection
        d.bboxes_2d.zero_()


class MultiAgentStates:
    """Container for multiple agents' states.

    Provides convenience methods for accessing states from different
    agent perspectives (ego vs other agents).
    """
    agent_states: Dict[AgentID, AgentStates]

    def __init__(
        self,
        agent_ids: List[AgentID],
        num_envs: int,
        num_joints: int,
        num_targets: int,
        device: torch.device
    ):
        """Initialize states for all agents.

        Args:
            agent_ids: List of agent identifiers
            num_envs: Number of parallel environments
            num_joints: Number of gimbal joints per agent
            num_targets: Number of detection targets per agent
            device: Device for tensor allocation
        """
        self.agent_states = {}
        for agent_id in agent_ids:
            self.agent_states[agent_id] = AgentStates(num_envs, num_joints, num_targets, device)

    def get_all_agent_states_for_ego(self, ego_agent: AgentID) -> Dict[AgentID, AgentStates]:
        """Get all agent states from a specific ego agent's perspective.

        Args:
            ego_agent: The ego agent requesting the states

        Returns:
            Dictionary mapping agent IDs to their states
        """
        agent_states = {}

        for agent_id, states in self.agent_states.items():
            if agent_id == ego_agent:
                agent_states[agent_id] = states
                continue
            agent_states[agent_id] = self.get_delayed_agent_states(
                ego_agent=ego_agent, other_agent=agent_id
            )

        return agent_states

    def get_delayed_agent_states(self, ego_agent: AgentID, other_agent: AgentID) -> AgentStates:
        """Get delayed states of another agent from ego's perspective.

        This is a placeholder - actual implementation depends on the delay system.

        Args:
            ego_agent: The requesting agent
            other_agent: The agent whose states are requested

        Returns:
            Delayed states of other_agent
        """
        # Default implementation returns the stored states directly
        # The MultiAgentDelaySystemV2 overrides this behavior
        return self.agent_states[other_agent]
