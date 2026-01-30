from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.common import AgentID
from isaaclab.utils.buffers.delay_buffer import DelayBuffer


class AgentStatesData:
    num_envs: int
    num_joints: int
    num_targets: int
    device: torch.device
    timestamp_sim_walltime: torch.Tensor # [N] in seconds
    timestamp_motion: torch.Tensor # [N] in seconds
    timestamp_detection: torch.Tensor # [N] in seconds
    body_position_w: torch.Tensor # [N, 3]
    body_orientation_w: torch.Tensor # [N, 4] (w, x, y, z)
    body_linear_velocity_w: torch.Tensor # [N, 3]
    body_linear_velocity_b: torch.Tensor # [N, 3]
    body_angular_velocity_w: torch.Tensor # [N, 3]
    body_angular_velocity_b: torch.Tensor # [N, 3]
    body_combined_angular_velocity_w: torch.Tensor  # Combined angular velocity (body + gimbal) [N, 3]
    body_combined_angular_velocity_b: torch.Tensor  # Combined angular velocity in body frame [N, 3]
    body_linear_acceleration_w: torch.Tensor # [N, 3]
    body_linear_acceleration_b: torch.Tensor # [N, 3]
    body_angular_acceleration_b: torch.Tensor # [N, 3]
    joint_positions_b: torch.Tensor # [N, J]
    joint_velocities_b: torch.Tensor # [N, J]
    joint_accelerations_b: torch.Tensor # [N, J]
    camera_offset_position_b: torch.Tensor # [N, 3]
    camera_offset_rotation_b: torch.Tensor # [N, 4] (w, x, y, z)
    camera_base_intrinsics: torch.Tensor # [N, 3, 3] - base camera intrinsics matrix K (unzoomed)
    camera_position_w: torch.Tensor # [N, 3]
    camera_orientation_w: torch.Tensor # [N, 4] (w, x, y, z)
    camera_ray_directions_w: torch.Tensor # [N, T, 3]
    camera_ray_origins_w: torch.Tensor # [N, T, 3]
    camera_zoom_level: torch.Tensor # zoom level [N]
    bboxes_2d: torch.Tensor # xywh format [N, T, 4]

class AgentStates:
    data: AgentStatesData
    timestamps: Optional[Any]  # Will be AgentStatesTimestamps when delay system is used

    def __init__(self, num_envs: int, num_joints: int, num_targets: int, device: torch.device):
        self.data = AgentStatesData()
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.num_targets = num_targets
        self.device = device
        self.timestamps = None  # Initialized by delay system if needed
        self._initialize_agent_state()

    def _initialize_agent_state(self):
        N = self.num_envs
        J = self.num_joints
        T = self.num_targets
        device = self.device

        self.data.timestamp_sim_walltime = torch.zeros(N, device=device)
        self.data.timestamp_motion = torch.zeros(N, device=device)
        self.data.timestamp_detection = torch.zeros(N, device=device)
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
        self.data.joint_positions_b = torch.zeros(N, J, device=device)
        self.data.joint_velocities_b = torch.zeros(N, J, device=device)
        self.data.joint_accelerations_b = torch.zeros(N, J, device=device)
        self.data.camera_offset_position_b = torch.zeros(N, 3, device=device)
        self.data.camera_offset_rotation_b = torch.zeros(N, 4, device=device)
        # Initialize camera intrinsics with proper structure [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.data.camera_base_intrinsics = torch.zeros(N, 3, 3, device=device)
        self.data.camera_base_intrinsics[:, 2, 2] = 1.0  # Bottom-right element must be 1.0
        self.data.camera_position_w = torch.zeros(N, 3, device=device)
        self.data.camera_orientation_w = math_utils.default_orientation(N, device=str(device))
        self.data.camera_ray_directions_w = torch.zeros(N, T, 3, device=device)
        self.data.camera_ray_origins_w = torch.zeros(N, T, 3, device=device)
        self.data.camera_zoom_level = torch.zeros(N, device=device)
        self.data.bboxes_2d = torch.zeros(N, T, 4, device=device)
        # NOTE: bboxes_2d_valid_mask has been REMOVED from the delay pipeline.
        # Users should validate bboxes AFTER delay processing using:
        #     bbox_valid = bbox_raycaster.validate_bbox(bbox)  # [N, T]
        # This avoids confusion between frame-level validity and detection validity.

class MultiAgentStates:
    agent_states: Dict[AgentID, AgentStates]

    def __init__(self, agent_ids: List[AgentID], num_envs: int, num_joints: int, num_targets: int,
                 device: torch.device):
        self.agent_states = {}
        for agent_id in agent_ids:
            self.agent_states[agent_id] = AgentStates(num_envs, num_joints, num_targets, device)

    def get_all_agent_states_for_ego(self, ego_agent: AgentID) -> Dict[AgentID, AgentStates]:
        agent_states = {}

        for agent_id, states in self.agent_states.items():
            if agent_id == ego_agent:
                agent_states[agent_id] = states
                continue
            agent_states[agent_id] = self.get_delayed_agent_states(ego_agent=ego_agent, other_agent=agent_id)

        return agent_states
    
    def get_delayed_agent_states(self, ego_agent: AgentID, other_agent: AgentID) -> AgentStates:
        pass