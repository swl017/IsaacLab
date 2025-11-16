"""
file: delayed_states.py
description: Defines and handles 1. states 2. delays(lag, latency, throttle buffer) 3. noise needed for computing observations
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.common import AgentID
from isaaclab.utils.buffers.delay_buffer import DelayBuffer


class DelayBufferCustom(DelayBuffer):
    """Custom DelayBuffer with separated append/retrieve operations.
    
    Extends Isaac Lab's DelayBuffer to separate write (append) from read (get_delayed) operations,
    preventing buffer corruption when reading without new data.
    """
    
    def append(self, data: torch.Tensor):
        """Append new data to the buffer without retrieving.

        Args:
            data: The input data. Shape is (batch_size, ...).
        """
        self._circular_buffer.append(data)

    def get_delayed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve delayed data from the buffer without appending new data.

        Returns delayed data based on the current time lag setting.
        If the buffer doesn't have enough history to satisfy the delay, returns invalid mask.

        If the buffer is empty (no data appended yet), raises RuntimeError.
        Users should check if buffer has data before calling this method, or handle the error.

        Returns:
            Tuple of (delayed_data, sufficient_history_mask) where:
            - delayed_data: The delayed version of the data from the stored buffer. Shape is (batch_size, ...).
            - sufficient_history_mask: Boolean mask [N] indicating which envs have enough history.
                                       True = delay can be satisfied, False = not enough history yet.

        Raises:
            RuntimeError: If the buffer is empty (no data has been appended yet).
        """
        # Check if buffer is empty - raise informative error
        if self._circular_buffer._buffer is None or torch.any(self._circular_buffer._num_pushes == 0):
            raise RuntimeError(
                "Cannot retrieve delayed data from empty buffer. "
                "Please append data first using append() or compute() method."
            )

        # Check if we have enough history to satisfy the delay
        # We need at least (time_lag + 1) items in the buffer
        sufficient_history = self._circular_buffer._num_pushes > self._time_lags

        delayed_data = self._circular_buffer[self._time_lags]
        return delayed_data.clone(), sufficient_history


class CommManager:
    """Communication manager using Isaac Lab's DelayBuffer with dropout support.

    This implementation uses discrete timestep delays instead of continuous time latencies.
    Messages are delayed by an integer number of steps drawn from a Gaussian distribution.
    Random dropouts are applied when messages are sent.

    Returns raw tensors with validity masks instead of TimestampedData structures.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        mean_delay_steps: float,
        std_delay_steps: float,
        dropout_rate: float,
        device: torch.device,
        max_history: int = 50
    ):
        """Initialize the communication manager.

        Args:
            num_envs: Number of parallel environments.
            num_agents: Number of agents in the system.
            mean_delay_steps: Mean delay in timesteps (will be rounded to int).
            std_delay_steps: Standard deviation of delay in timesteps.
            dropout_rate: Probability of message dropout (0.0 to 1.0).
            device: Device to allocate tensors on.
            max_history: Maximum number of timesteps to buffer (should be >= max expected delay).
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.max_history = max_history

        # Delay parameters [N, num_agents] - per sender
        self.mean_delay_steps = torch.full((num_envs, num_agents), mean_delay_steps, device=device)
        self.std_delay_steps = torch.full((num_envs, num_agents), std_delay_steps, device=device)
        self.dropout_rate = torch.full((num_envs, num_agents), dropout_rate, device=device)

        # Create DelayBuffers for each sender->receiver link
        # Structure: buffers[sender_id][receiver_id] = Dict[data_key -> DelayBuffer]
        self.buffers: Dict[int, Dict[int, Dict[str, DelayBuffer]]] = {}

        # Store data shapes for each key (to avoid accessing protected members)
        self.data_shapes: Dict[int, Dict[int, Dict[str, Tuple[int, ...]]]] = {}

        # Dropout masks - store which messages were dropped
        # Structure: dropout_buffers[sender_id][receiver_id] = DelayBuffer for dropout mask
        self.dropout_buffers: Dict[int, Dict[int, DelayBuffer]] = {}

        # Valid masks - store which messages were sent
        # Structure: valid_buffers[sender_id][receiver_id] = DelayBuffer for valid mask
        self.valid_buffers: Dict[int, Dict[int, DelayBuffer]] = {}

        # Statistics
        self.message_count = torch.zeros(num_envs, num_agents, num_agents, dtype=torch.long, device=device)
        self.dropout_count = torch.zeros(num_envs, num_agents, num_agents, dtype=torch.long, device=device)

        # Initialize nested dictionaries
        for sender_id in range(num_agents):
            self.buffers[sender_id] = {}
            self.data_shapes[sender_id] = {}
            self.dropout_buffers[sender_id] = {}
            self.valid_buffers[sender_id] = {}
            for receiver_id in range(num_agents):
                if sender_id != receiver_id:
                    self.buffers[sender_id][receiver_id] = {}
                    self.data_shapes[sender_id][receiver_id] = {}
                    # Create dropout buffer (stores boolean dropout mask)
                    self.dropout_buffers[sender_id][receiver_id] = DelayBufferCustom(
                        history_length=max_history,
                        batch_size=num_envs,
                        device=str(device)
                    )
                    # Create valid buffer (stores boolean valid mask)
                    self.valid_buffers[sender_id][receiver_id] = DelayBufferCustom(
                        history_length=max_history,
                        batch_size=num_envs,
                        device=str(device)
                    )

    def update_delay_params(
        self,
        mean_delay_steps: Optional[torch.Tensor] = None,
        std_delay_steps: Optional[torch.Tensor] = None,
        dropout_rate: Optional[torch.Tensor] = None
    ):
        """Update delay and dropout parameters.

        Args:
            mean_delay_steps: New mean delay in timesteps [N, num_agents]. If None, unchanged.
            std_delay_steps: New std delay in timesteps [N, num_agents]. If None, unchanged.
            dropout_rate: New dropout rate [N, num_agents]. If None, unchanged.
        """
        if mean_delay_steps is not None:
            self.mean_delay_steps = mean_delay_steps
        if std_delay_steps is not None:
            self.std_delay_steps = std_delay_steps
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate

    def send_message(
        self,
        sender_id: int,
        data: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ):
        """Send a message from one agent to all other agents.

        Args:
            sender_id: ID of the sending agent.
            data: Dictionary of data tensors to send. Each tensor shape: [N, ...].
            valid_mask: Optional validity mask [N]. If None, all messages are valid.
        """
        if valid_mask is None:
            valid_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Send to each receiver
        for receiver_id in range(self.num_agents):
            if receiver_id == sender_id:
                continue

            # Generate random delays for this link [N] - integers
            delays_float = torch.normal(
                mean=self.mean_delay_steps[:, sender_id],
                std=self.std_delay_steps[:, sender_id]
            ).clamp(min=0.0, max=float(self.max_history))
            delays_int = delays_float.round().long()  # [N]

            # Generate dropout mask [N] - True means DROPPED
            dropout_mask = torch.rand(self.num_envs, device=self.device) < self.dropout_rate[:, sender_id]

            # Final mask: valid AND not dropped
            send_mask = valid_mask & ~dropout_mask  # [N]

            # Set delays in all buffers for this link
            self.dropout_buffers[sender_id][receiver_id].set_time_lag(delays_int)
            self.valid_buffers[sender_id][receiver_id].set_time_lag(delays_int)

            # Store dropout status (True = dropped, False = not dropped)
            dropout_tensor = dropout_mask.float().unsqueeze(-1)  # [N, 1]
            self.dropout_buffers[sender_id][receiver_id].append(dropout_tensor)

            # Store valid status
            valid_tensor = send_mask.float().unsqueeze(-1)  # [N, 1]
            self.valid_buffers[sender_id][receiver_id].append(valid_tensor)

            # Store each data field
            for key, tensor in data.items():
                # Create buffer for this data field if it doesn't exist
                if key not in self.buffers[sender_id][receiver_id]:
                    self.buffers[sender_id][receiver_id][key] = DelayBufferCustom(
                        history_length=self.max_history,
                        batch_size=self.num_envs,
                        device=str(self.device)
                    )
                    # Store the shape (excluding batch dimension)
                    self.data_shapes[sender_id][receiver_id][key] = tensor.shape[1:]

                # Set same delay for data buffer
                self.buffers[sender_id][receiver_id][key].set_time_lag(delays_int)

                # Store data (even if dropped - will be masked on retrieval)
                self.buffers[sender_id][receiver_id][key].append(tensor)

            # Update statistics
            self.message_count[:, sender_id, receiver_id] += valid_mask.long()
            self.dropout_count[:, sender_id, receiver_id] += (valid_mask & dropout_mask).long()

    def receive_messages(
        self,
        receiver_id: int
    ) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Receive delayed messages from all other agents.

        Args:
            receiver_id: ID of the receiving agent.

        Returns:
            Dictionary mapping sender_id -> {data_key -> (data_tensor, valid_mask)}
            where valid_mask indicates which environments received valid (non-dropped) data.
        """
        received_data = {}

        for sender_id in range(self.num_agents):
            if sender_id == receiver_id:
                continue

            # Check if any data has been sent on this link
            if not self.buffers[sender_id][receiver_id]:
                continue

            # Retrieve dropout status (1.0 = dropped, 0.0 = not dropped)
            dropout_tensor, dropout_sufficient = self.dropout_buffers[sender_id][receiver_id].get_delayed()
            is_dropped = dropout_tensor.squeeze(-1) > 0.5  # [N]

            # Retrieve valid status (1.0 = valid, 0.0 = invalid)
            valid_tensor, valid_sufficient = self.valid_buffers[sender_id][receiver_id].get_delayed()
            is_valid = valid_tensor.squeeze(-1) > 0.5  # [N]

            # Final validity: valid AND not dropped AND sufficient history
            # All buffers should have sufficient history together
            sufficient_history = dropout_sufficient & valid_sufficient
            final_valid = is_valid & ~is_dropped & sufficient_history  # [N]

            # Retrieve all data fields
            data_dict = {}
            for key, buffer in self.buffers[sender_id][receiver_id].items():
                # Get delayed data (read-only operation)
                delayed_data, data_sufficient = buffer.get_delayed()

                # Data is only valid if buffer has sufficient history
                data_valid = final_valid & data_sufficient

                # Store as (data, valid_mask) tuple
                data_dict[key] = (delayed_data, data_valid)

            if data_dict:
                received_data[sender_id] = data_dict

        return received_data

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get communication statistics.

        Returns:
            Dictionary with statistics:
                - message_count: Total messages sent [N, num_agents, num_agents]
                - dropout_count: Total messages dropped [N, num_agents, num_agents]
                - dropout_rate: Effective dropout rate [N, num_agents, num_agents]
        """
        dropout_rate_effective = torch.where(
            self.message_count > 0,
            self.dropout_count.float() / self.message_count.float(),
            torch.zeros_like(self.dropout_count, dtype=torch.float32)
        )

        return {
            'message_count': self.message_count.clone(),
            'dropout_count': self.dropout_count.clone(),
            'dropout_rate_effective': dropout_rate_effective
        }

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset communication buffers for specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, resets all.
        """
        # Convert env_ids to proper format for DelayBuffer.reset()
        if env_ids is None:
            env_ids_list = None  # Reset all
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_list = env_ids.cpu().tolist()
            env_ids_tensor = env_ids

        # Reset all buffers
        for sender_id in range(self.num_agents):
            for receiver_id in range(self.num_agents):
                if sender_id == receiver_id:
                    continue

                # Reset dropout and valid buffers
                self.dropout_buffers[sender_id][receiver_id].reset(env_ids_list)
                self.valid_buffers[sender_id][receiver_id].reset(env_ids_list)

                # Reset data buffers
                for buffer in self.buffers[sender_id][receiver_id].values():
                    buffer.reset(env_ids_list)

        # Reset statistics
        self.message_count[env_ids_tensor] = 0
        self.dropout_count[env_ids_tensor] = 0

class AgentStatesData:
    num_envs: int
    num_joints: int
    num_targets: int
    device: torch.device
    body_position_w: torch.Tensor
    body_orientation_w: torch.Tensor
    body_linear_velocity_w: torch.Tensor
    body_linear_velocity_b: torch.Tensor
    body_angular_velocity_w: torch.Tensor
    body_angular_velocity_b: torch.Tensor
    body_combined_angular_velocity_w: torch.Tensor  # Combined angular velocity (body + gimbal)
    body_combined_angular_velocity_b: torch.Tensor  # Combined angular velocity in body frame
    body_linear_acceleration_w: torch.Tensor
    body_linear_acceleration_b: torch.Tensor
    body_angular_acceleration_b: torch.Tensor
    joint_positions_b: torch.Tensor
    joint_velocities_b: torch.Tensor
    joint_accelerations_b: torch.Tensor
    camera_width: torch.Tensor
    camera_height: torch.Tensor
    camera_image_rbg: torch.Tensor
    camera_focal_length: torch.Tensor
    camera_horizontal_aperture: torch.Tensor
    camera_vertical_aperture: torch.Tensor
    camera_offset_position_b: torch.Tensor
    camera_offset_rotation_b: torch.Tensor
    camera_intrinsics: torch.Tensor
    camera_position_w: torch.Tensor
    camera_orientation_w: torch.Tensor
    camera_ray_directions_w: torch.Tensor
    camera_ray_origins_w: torch.Tensor
    camera_zoom_level: torch.Tensor
    bboxes_2d: torch.Tensor # xywh format
    bboxes_2d_valid_mask: torch.Tensor # boolean mask indicating valid bounding boxes
    bboxes_2d_age: torch.Tensor # time elapsed since detection was captured (seconds) [N, T]

class AgentStates:
    """Defines the agent states to be used in the task.

    Attributes:
        body_position_w (torch.Tensor): Body position in world frame.
        body_orientation_w (torch.Tensor): Body orientation in world frame. (w, x, y, z)
        body_linear_velocity_w (torch.Tensor): Body linear velocity in world frame.
        body_linear_velocity_b (torch.Tensor): Body linear velocity in body frame.
        body_angular_velocity_w (torch.Tensor): Body angular velocity in world frame.
        body_angular_velocity_b (torch.Tensor): Body angular velocity in body frame.
        body_combined_angular_velocity_w (torch.Tensor): Combined angular velocity (body + gimbal) in world frame.
        body_combined_angular_velocity_b (torch.Tensor): Combined angular velocity (body + gimbal) in body frame.
        body_linear_acceleration_w (torch.Tensor): Body linear acceleration in world frame.
        body_linear_acceleration_b (torch.Tensor): Body linear acceleration in body frame.
        body_angular_acceleration_b (torch.Tensor): Body angular acceleration in body frame.
        joint_positions_b (torch.Tensor): Joint positions in body frame.
        joint_velocities_b (torch.Tensor): Joint velocities in body frame.
        joint_accelerations_b (torch.Tensor): Joint accelerations in body frame.
    """
    data: AgentStatesData

    def __init__(self, num_envs: int, num_joints: int, num_targets: int, device: torch.device):
        N = num_envs
        J = num_joints # convention: 0 roll, 1 pitch, 2 yaw
        T = num_targets

        # Initialize data container
        self.data = AgentStatesData()

        self.data.num_envs = num_envs
        self.data.num_joints = num_joints
        self.data.num_targets = num_targets
        self.data.device = device
        self.data.body_position_w = torch.zeros((N, 3), device=device)
        self.data.body_orientation_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)
        self.data.body_linear_velocity_w = torch.zeros((N, 3), device=device)
        self.data.body_linear_velocity_b = torch.zeros((N, 3), device=device)
        self.data.body_angular_velocity_w = torch.zeros((N, 3), device=device)
        self.data.body_angular_velocity_b = torch.zeros((N, 3), device=device)
        self.data.body_combined_angular_velocity_w = torch.zeros((N, 3), device=device)
        self.data.body_combined_angular_velocity_b = torch.zeros((N, 3), device=device)
        self.data.body_linear_acceleration_w = torch.zeros((N, 3), device=device)
        self.data.body_linear_acceleration_b = torch.zeros((N, 3), device=device)
        self.data.body_angular_acceleration_b = torch.zeros((N, 3), device=device)
        self.data.joint_positions_b = torch.zeros((N, J), device=device)
        self.data.joint_velocities_b = torch.zeros((N, J), device=device)
        self.data.joint_accelerations_b = torch.zeros((N, J), device=device)
        self.data.camera_width = torch.zeros((N,), device=device)
        self.data.camera_height = torch.zeros((N,), device=device)
        self.data.camera_image_rbg = torch.zeros((N, 3, 0, 0), device=device, dtype=torch.uint8)
        self.data.camera_focal_length = torch.zeros((N,), device=device)
        self.data.camera_horizontal_aperture = torch.zeros((N,), device=device)
        self.data.camera_vertical_aperture = torch.zeros((N,), device=device)
        self.data.camera_offset_position_b = torch.zeros((N, 3), device=device)
        self.data.camera_offset_rotation_b = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)
        self.data.camera_intrinsics = torch.zeros((N, 3, 3), device=device)
        self.data.camera_position_w = torch.zeros((N, 3), device=device)
        self.data.camera_orientation_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)
        self.data.camera_ray_directions_w = torch.zeros((N, T, 3), device=device)
        self.data.camera_ray_origins_w = torch.zeros((N, T, 3), device=device)
        self.data.camera_zoom_level = torch.ones((N,), device=device)
        self.data.bboxes_2d = torch.zeros((N, T, 4), device=device)
        self.data.bboxes_2d_valid_mask = torch.zeros((N, T), device=device, dtype=torch.bool)
        self.data.bboxes_2d_age = torch.zeros((N, T), device=device)

    def set_camera_configs(
        self,
        width: Optional[torch.Tensor] | Optional[int],
        height: Optional[torch.Tensor] | Optional[int],
        focal_length: Optional[torch.Tensor] | Optional[float],
        horizontal_aperture: Optional[torch.Tensor] | Optional[float],
        offset_position_b: Optional[torch.Tensor] | Optional[float],
        offset_rotation_b: Optional[torch.Tensor] | Optional[float],
        vertical_aperture: Optional[torch.Tensor] | Optional[float] = None,
    ):
        """Sets the camera configuration parameters.

        Args:
            width (Optional[torch.Tensor] | Optional[int]): Camera image width.
            height (Optional[torch.Tensor] | Optional[int]): Camera image height.
            focal_length (Optional[torch.Tensor] | Optional[float]): Camera focal length.
            horizontal_aperture (Optional[torch.Tensor] | Optional[float]): Camera horizontal aperture.
            offset_position_b (Optional[torch.Tensor] | Optional[float]): Camera offset position in body frame.
            offset_rotation_b (Optional[torch.Tensor] | Optional[float]): Camera offset rotation in body frame.
            intrinsics (Optional[torch.Tensor] | Optional[float]| None): Camera intrinsic matrix.
        """
        self.data.camera_width = width if isinstance(width, torch.Tensor) else torch.full((self.data.num_envs,), width, device=self.data.device)
        self.data.camera_height = height if isinstance(height, torch.Tensor) else torch.full((self.data.num_envs,), height, device=self.data.device)
        self.data.camera_focal_length = focal_length if isinstance(focal_length, torch.Tensor) else torch.full((self.data.num_envs,), focal_length, device=self.data.device)
        self.data.camera_horizontal_aperture = horizontal_aperture if isinstance(horizontal_aperture, torch.Tensor) else torch.full((self.data.num_envs,), horizontal_aperture, device=self.data.device)
        if vertical_aperture is None:
            self.data.camera_vertical_aperture = self.data.camera_horizontal_aperture * (self.data.camera_height / self.data.camera_width)
        else:
            self.data.camera_vertical_aperture = vertical_aperture if isinstance(vertical_aperture, torch.Tensor) else torch.full((self.data.num_envs,), vertical_aperture, device=self.data.device)
        # Handle offset position (shape: [N, 3] or [3])
        if isinstance(offset_position_b, torch.Tensor):
            if offset_position_b.ndim == 2 and offset_position_b.shape[0] == self.data.num_envs:
                self.data.camera_offset_position_b = offset_position_b.clone()
            else:
                # Single position - broadcast to all envs
                self.data.camera_offset_position_b = offset_position_b.reshape(1, -1).repeat(self.data.num_envs, 1)
        else:
            self.data.camera_offset_position_b = torch.tensor([offset_position_b] * 3, device=self.data.device).unsqueeze(0).repeat(self.data.num_envs, 1)

        # Handle offset rotation (shape: [N, 4] or [4])
        if isinstance(offset_rotation_b, torch.Tensor):
            if offset_rotation_b.ndim == 2 and offset_rotation_b.shape[0] == self.data.num_envs:
                self.data.camera_offset_rotation_b = offset_rotation_b.clone()
            else:
                # Single quaternion - broadcast to all envs
                self.data.camera_offset_rotation_b = offset_rotation_b.reshape(1, -1).repeat(self.data.num_envs, 1)
        else:
            self.data.camera_offset_rotation_b = torch.tensor([offset_rotation_b] * 4, device=self.data.device).unsqueeze(0).repeat(self.data.num_envs, 1)

    def update_intrinsic_matrix(self, zoom_level: torch.Tensor, env_idxs: Optional[torch.Tensor] = None):
        """Updates the camera intrinsic matrix based on focal length and image size."""
        if env_idxs is None:
            env_idxs = torch.arange(self.data.num_envs, device=self.data.device)
        fx = zoom_level * (self.data.camera_width * self.data.camera_focal_length) / self.data.camera_horizontal_aperture
        vertical_aperture = self.data.camera_horizontal_aperture * (self.data.camera_height / self.data.camera_width)
        fy = zoom_level * (self.data.camera_height * self.data.camera_focal_length) / vertical_aperture
        cx = self.data.camera_width / 2.0
        cy = self.data.camera_height / 2.0

        self.data.camera_intrinsics[env_idxs, 0, 0] = fx[env_idxs]
        self.data.camera_intrinsics[env_idxs, 0, 2] = cx[env_idxs]
        self.data.camera_intrinsics[env_idxs, 1, 1] = fy[env_idxs]
        self.data.camera_intrinsics[env_idxs, 1, 2] = cy[env_idxs]
        self.data.camera_intrinsics[env_idxs, 2, 2] = 1.0
    
    def update_camera_pose(self, body_pos: torch.Tensor, body_quat: torch.Tensor, joint_pos: torch.Tensor, env_idxs: Optional[torch.Tensor] = None):
        """Updates the camera pose in world frame based on body and joint states.

        Args:
            body_pos: Body position in world frame of shape (N, 3).
            body_quat: Body orientation in world frame of shape (N, 4).
            joint_pos: Joint positions in body frame of shape (N, J). ZYX Euler angles.
            env_idxs: Indices of environments to update. If None, updates all.
        """
        if env_idxs is None:
            env_idxs = torch.arange(self.data.num_envs, device=self.data.device)

        gimbal_quat_b = self.quat_from_gimbal_zxy(
            roll=joint_pos[env_idxs, 0],
            pitch=joint_pos[env_idxs, 1],
            yaw=joint_pos[env_idxs, 2],
        )  # (n, 4)

        # Compose all rotations: world -> body -> gimbal -> camera
        self.data.camera_orientation_w[env_idxs] = math_utils.quat_mul(
            math_utils.quat_mul(body_quat[env_idxs], gimbal_quat_b),
            self.data.camera_offset_rotation_b[env_idxs],
        )  # (n, 4)

        # Transform camera offset position through the composed rotation
        self.data.camera_position_w[env_idxs] = body_pos[env_idxs] + math_utils.quat_apply(
            body_quat[env_idxs], # Simlification, assuming offset is applied at body level
            self.data.camera_offset_position_b[env_idxs]
        )  # (n, 3)

    def update_bboxes_2d(self, bboxes_2d: torch.Tensor, valid_mask: torch.Tensor, env_idxs: Optional[torch.Tensor] = None):
        """Updates the 2D bounding boxes and their validity mask.

        Args:
            bboxes_2d: 2D bounding boxes of shape (N, T, 4). xywh format.
            valid_mask: Validity mask of shape (N, T).
            env_idxs: Indices of environments to update. If None, updates all.
        """
        if env_idxs is None:
            env_idxs = torch.arange(self.data.num_envs, device=self.data.device)
        # Expand mask to match bbox shape [N, T, 4]
        valid_mask_expanded = valid_mask[env_idxs].unsqueeze(-1)  # [N, T] -> [N, T, 1]
        self.data.bboxes_2d[env_idxs] = torch.where(
            valid_mask_expanded, bboxes_2d[env_idxs], torch.ones_like(bboxes_2d[env_idxs]) * (-1.0)
        )
        self.data.bboxes_2d_valid_mask[env_idxs] = valid_mask[env_idxs]

    def update_camera_rays_to_bboxes(self, bboxes_2d: torch.Tensor, camera_intrinsics: torch.Tensor, camera_position: torch.Tensor, camera_orientation: torch.Tensor, env_idxs: Optional[torch.Tensor] = None):
        """Updates the camera ray directions and origins based on bounding boxes and camera matrices.
        
        Args:
            bboxes_2d: 2D bounding boxes of shape (N, T, 4). xywh format.
            camera_intrinsics: Camera intrinsic matrices of shape (N, 3, 3).
            camera_position: Camera position in world frame of shape (N, 3).
            camera_orientation: Camera orientation in world frame of shape (N, 4).
            env_idxs: Indices of environments to update. If None, updates all.
        """
        if env_idxs is None:
            env_idxs = torch.arange(self.data.num_envs, device=self.data.device)
        N, T, _ = bboxes_2d.shape
        n = env_idxs.shape[0]

        fx = camera_intrinsics[env_idxs, 0, 0].unsqueeze(-1) # (n, 1)
        fy = camera_intrinsics[env_idxs, 1, 1].unsqueeze(-1)
        cx = camera_intrinsics[env_idxs, 0, 2].unsqueeze(-1)
        cy = camera_intrinsics[env_idxs, 1, 2].unsqueeze(-1)

        bboxes_xywh = bboxes_2d[env_idxs, ..., :4]  # (n, T, 4)
        x_n = (bboxes_xywh[..., 0] - cx) / fx # (n, T)
        y_n = (bboxes_xywh[..., 1] - cy) / fy

        dirs_camera = torch.stack([x_n, y_n, torch.ones_like(x_n)], dim=-1)  # (n, T, 3)
        dirs_camera_norm = dirs_camera / torch.norm(dirs_camera, dim=-1, keepdim=True)  # Normalize

        camera_rot_mat = math_utils.matrix_from_quat(camera_orientation[env_idxs])  # (n, 3, 3)
        """
        Purpose: rotate T vectors by N rotation matrices
            camera_rot_mat:    (N, 3, 3)     - N rotation matrices
                                n  i  j

            dirs_camera_norm:  (N, T, 3)     - N×T direction vectors
                                n  t  j

            Operation: For each environment n and target t, apply rotation matrix:
                    dirs_world[n,t,i] = Σ_j R[n,i,j] * v[n,t,j]

        Result:           (N, T, 3)      - N×T rotated vectors
                            n  t  i"""
        self.data.camera_ray_directions_w[env_idxs] = torch.einsum('nij,ntj->nti', camera_rot_mat, dirs_camera_norm)  # (n, T, 3)
        self.data.camera_ray_origins_w[env_idxs] = camera_position[env_idxs].unsqueeze(1).expand(-1, T, -1)  # (n, T, 3)

    @torch.jit.script
    def quat_from_gimbal_zxy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Convert gimbal angles to quaternion for R = R_z(yaw) * R_x(roll) * R_y(pitch).
        R_z(ψ) = [ cos(ψ)  -sin(ψ)   0 ]
                 [ sin(ψ)   cos(ψ)   0 ]
                 [   0        0      1 ]
        R_x(φ) = [ 1      0         0    ]
                 [ 0   cos(φ)  -sin(φ)   ]
                 [ 0   sin(φ)   cos(φ)   ]
        R_y(θ) = [  cos(θ)   0   sin(θ) ]
                 [    0      1     0    ]
                 [ -sin(θ)   0   cos(θ) ]
        R = (R_z * R_x) * R_y 
          = [ cψ·cθ - sψ·sφ·sθ    -sψ·cφ     cψ·sθ + sψ·sφ·cθ ]
            [ sψ·cθ + cψ·sφ·sθ     cψ·cφ     sψ·sθ - cψ·sφ·cθ ]
            [     -cφ·sθ             sφ           cφ·cθ       ]
        
        Args:
            yaw: Rotation around z-axis (in radians). Shape is (N,).
            roll: Rotation around x-axis (in radians). Shape is (N,).
            pitch: Rotation around y-axis (in radians). Shape is (N,).
        
        Returns:
            The quaternion in (w, x, y, z). Shape is (N, 4).
        """
        # Half angles - using clear naming convention
        cz = torch.cos(yaw * 0.5)
        sz = torch.sin(yaw * 0.5)
        cx = torch.cos(roll * 0.5)
        sx = torch.sin(roll * 0.5)
        cy = torch.cos(pitch * 0.5)
        sy = torch.sin(pitch * 0.5)
        
        # ZXY extrinsic: R = R_z(yaw) * R_x(roll) * R_y(pitch)
        qw = cz * cx * cy - sz * sx * sy
        qx = cz * sx * cy - sz * cx * sy
        qy = cz * cx * sy + sz * sx * cy
        qz = cz * sx * sy + sz * cx * cy
        
        return torch.stack([qw, qx, qy, qz], dim=-1)
    
    def reset(self):
        """Resets all states to zero.
        """
        self.data.body_position_w.zero_()
        self.data.body_orientation_w.fill_(0.0)
        self.data.body_orientation_w[:, 0] = 1.0  # w component
        self.data.body_linear_velocity_w.zero_()
        self.data.body_linear_velocity_b.zero_()
        self.data.body_angular_velocity_w.zero_()
        self.data.body_angular_velocity_b.zero_()
        self.data.body_combined_angular_velocity_w.zero_()
        self.data.body_combined_angular_velocity_b.zero_()
        self.data.body_linear_acceleration_w.zero_()
        self.data.body_linear_acceleration_b.zero_()
        self.data.body_angular_acceleration_b.zero_()
        self.data.joint_positions_b.zero_()
        self.data.joint_velocities_b.zero_()
        self.data.joint_accelerations_b.zero_()
        self.data.camera_width.zero_()
        self.data.camera_height.zero_()
        self.data.camera_image_rbg.zero_()
        self.data.camera_focal_length.zero_()
        self.data.camera_horizontal_aperture.zero_()
        self.data.camera_vertical_aperture.zero_()
        self.data.camera_offset_position_b.zero_()
        self.data.camera_offset_rotation_b.fill_(0.0)
        self.data.camera_offset_rotation_b[:, 0] = 1.0  # w component
        self.data.camera_intrinsics.zero_()
        self.data.camera_position_w.zero_()
        self.data.camera_orientation_w.fill_(0.0)
        self.data.camera_orientation_w[:, 0] = 1.0  # w component
        self.data.camera_ray_directions_w.zero_()
        self.data.camera_ray_origins_w.zero_()
        self.data.camera_zoom_level.fill_(1.0)
        self.data.bboxes_2d.zero_()
        self.data.bboxes_2d_valid_mask.zero_()

class DelayedAgentStates(AgentStates):
    """Defines the delayed states for a single agent.
    Handles delays such as lag, latency, and throttle buffer.
    Inherits from AgentStates.
    """

    def __init__(self, num_envs: int, num_joints: int, num_targets: int, device: torch.device):
        super().__init__(num_envs, num_joints, num_targets, device)

    def compute_lag(self):
        pass

    def compute_latency(self):
        pass

    def compute_throttle_buffer(self):
        pass

    def update(self):
        pass

class MultiAgentStates:
    """Defines the states for multiple agents.
    """

    def __init__(self, possible_agents: list[AgentID], num_envs: int, num_joints_per_agent: dict[AgentID, int], num_targets_per_agent: dict[AgentID, int], device: torch.device):
        self.agents: Dict[AgentID, AgentStates] = {}
        for agent_id in possible_agents:
            self.agents[agent_id] = AgentStates(
                num_envs=num_envs,
                num_joints=num_joints_per_agent[agent_id],
                num_targets=num_targets_per_agent[agent_id],
                device=device,
            )


class FirstOrderLag:
    """First-order lag filter"""
    
    def __init__(self, num_envs: int, state_dim: int, time_constant: torch.Tensor | float,
                 dt: float, device: torch.device | str, initial_state: Optional[torch.Tensor] = None):
        """
        Args:
            num_envs: Number of environments.
            state_dim: Dimension of the state.
            time_constant: Time constant for the filter. (N, 1) or (N, state_dim) tensor or float.
            dt: Time step.
            device: Device to allocate tensors on.
            initial_state: Initial state of the filter.
        """
        self.num_envs = num_envs
        self.state_dim = state_dim
        if isinstance(time_constant, float):
            self.tau = torch.full((num_envs,state_dim), time_constant, device=device)
        else:
            self.tau = time_constant
        self.dt = dt
        self.device = device
        self.alpha = dt / (dt + self.tau) # (N, 1) or (N, state_dim)

        self.filtered_state = initial_state.clone() if initial_state is not None else \
                              torch.zeros(num_envs, state_dim, device=device)
    
    def update_time_constants(self, time_constant: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.tau[env_ids] = time_constant[env_ids]
        self.alpha[env_ids] = self.dt / (self.dt + time_constant[env_ids])

    def update(self, measured_state: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        self.filtered_state = self.filtered_state + self.alpha * (measured_state - self.filtered_state)
        return self.filtered_state.clone()
    
    def reset(self, env_ids: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if state is not None:
            self.filtered_state[env_ids] = state
        else:
            self.filtered_state[env_ids] = 0.0

class QuaternionFirstOrderLag:
    """Quaternion SLERP filter."""
    
    def __init__(self, num_envs: int, time_constant: torch.Tensor, dt: float,
                 device: torch.device, initial_quat: Optional[torch.Tensor] = None):
        self.num_envs = num_envs
        self.tau = time_constant
        self.dt = dt
        self.device = device
        self.alpha = dt / (dt + time_constant)
        
        if initial_quat is not None:
            self.filtered_quat = initial_quat.clone()
        else:
            self.filtered_quat = torch.zeros(num_envs, 4, device=device)
            self.filtered_quat[:, 0] = 1.0
        
        self.timestamp = torch.zeros(num_envs, device=device)

    def update_time_constants(self, time_constant: torch.Tensor):
        self.tau = time_constant
        self.alpha = self.dt / (self.dt + time_constant)

    def update(self, measured_quat: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        self.filtered_quat = self._slerp(self.filtered_quat, measured_quat, self.alpha)
        if isinstance(current_time, torch.Tensor):
            self.timestamp = current_time.clone()
        else:
            self.timestamp.fill_(current_time)
        return self.filtered_quat.clone()
    
    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        q0 = q0 / (torch.norm(q0, dim=-1, keepdim=True) + 1e-8)
        q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
        
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.abs(dot)
        
        mask = (dot > 0.9995).squeeze(-1)
        theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_theta = torch.sin(theta)
        sin_theta = torch.where(sin_theta < 1e-8, torch.ones_like(sin_theta), sin_theta)
        
        w0 = torch.sin((1 - t) * theta) / sin_theta
        w1 = torch.sin(t * theta) / sin_theta
        result_slerp = w0 * q0 + w1 * q1
        
        result_lerp = (1 - t) * q0 + t * q1
        result_lerp = result_lerp / (torch.norm(result_lerp, dim=-1, keepdim=True) + 1e-8)
        
        return torch.where(mask.unsqueeze(-1), result_lerp, result_slerp)
    
    def reset(self, env_ids: Optional[torch.Tensor] = None, quat: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if quat is not None:
            self.filtered_quat[env_ids] = quat
        else:
            self.filtered_quat[env_ids] = 0
            self.filtered_quat[env_ids, 0] = 1.0
        self.timestamp[env_ids] = 0.0

class ChannelBuffer:
    """Unified buffer with realistic channel impairments.

    Applies impairments in physically realistic order:
    1. Throttle: Rate limiting (FPS, bandwidth) - determines if sample is taken
    2. Dropout: Packet loss / detection failure - determines if sample succeeds
    3. Latency: Variable transmission/processing delay - determines when sample arrives

    Only valid (non-dropped, throttled) data is stored in the latency buffer,
    making this memory-efficient and physically realistic.
    """

    def __init__(
        self,
        num_envs: int,
        data_shape: Tuple[int, ...],
        throttle_period: torch.Tensor | float,
        dropout_rate: torch.Tensor | float,
        mean_latency: torch.Tensor | float,
        std_latency: torch.Tensor | float,
        dt: float,
        device: torch.device,
        max_history: int = 100
    ):
        """Initialize unified channel buffer.

        Args:
            num_envs: Number of parallel environments.
            data_shape: Shape of data (excluding batch dimension).
            throttle_period: Minimum period between updates in seconds [N] or scalar.
                           Set to 0 to disable throttling.
            dropout_rate: Probability of dropping data (0.0 to 1.0) [N] or scalar.
                         Set to 0 to disable dropout.
            mean_latency: Mean latency in seconds [N] or scalar.
                         Set to 0 for zero-latency (immediate).
            std_latency: Std deviation of latency in seconds [N] or scalar.
            dt: Simulation timestep in seconds.
            device: Device to allocate tensors on.
            max_history: Maximum history length for latency buffer.
        """
        self.num_envs = num_envs
        self.data_shape = data_shape
        self.dt = dt
        self.device = device
        self.max_history = max_history

        # Convert parameters to tensors if needed
        if isinstance(throttle_period, float):
            self.throttle_period = torch.full((num_envs,), throttle_period, device=device)
        else:
            self.throttle_period = throttle_period

        if isinstance(dropout_rate, float):
            self.dropout_rate = torch.full((num_envs,), dropout_rate, device=device)
        else:
            self.dropout_rate = dropout_rate

        if isinstance(mean_latency, float):
            self.mean_latency = torch.full((num_envs,), mean_latency, device=device)
        else:
            self.mean_latency = mean_latency

        if isinstance(std_latency, float):
            self.std_latency = torch.full((num_envs,), std_latency, device=device)
        else:
            self.std_latency = std_latency

        # Throttle state
        self.last_throttle_time = torch.full((num_envs,), -float('inf'), device=device)

        # Latest valid data (shared across all stages)
        self.latest_data = torch.zeros(num_envs, *data_shape, device=device)
        self.has_valid_data = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Time-based availability tracking
        self.data_added_time = torch.full((num_envs,), -float('inf'), device=device)  # When data was last added
        self.data_available_time = torch.full((num_envs,), float('inf'), device=device)  # When data will be available

        # Latency buffer (Isaac Lab's DelayBuffer)
        self.latency_buffer = DelayBufferCustom(
            history_length=max_history,
            batch_size=num_envs,
            device=str(device)
        )

        # Statistics
        self.total_attempts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.throttled_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.dropout_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update_params(
        self,
        throttle_period: Optional[torch.Tensor] = None,
        dropout_rate: Optional[torch.Tensor] = None,
        mean_latency: Optional[torch.Tensor] = None,
        std_latency: Optional[torch.Tensor] = None
    ):
        """Update channel parameters.

        Args:
            throttle_period: New throttle period [N]. If None, unchanged.
            dropout_rate: New dropout rate [N]. If None, unchanged.
            mean_latency: New mean latency [N]. If None, unchanged.
            std_latency: New std latency [N]. If None, unchanged.
        """
        if throttle_period is not None:
            self.throttle_period = throttle_period
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate
        if mean_latency is not None:
            self.mean_latency = mean_latency
        if std_latency is not None:
            self.std_latency = std_latency

    def compute(
        self,
        data: torch.Tensor,
        current_time: torch.Tensor | float,
        has_source_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process data through channel with impairments.

        Applies: Throttle → Dropout → Latency in that order.

        Args:
            data: New data [N, *data_shape].
            current_time: Current simulation time [N] or scalar.
            has_source_data: Boolean mask [N] indicating whether each environment has
                meaningful source data available to transmit. True means source has data
                (e.g., sensor detected target, measurement is valid). Independent of
                channel impairments. If None, assumes all environments have data.

        Returns:
            Tuple of (delayed_data, valid_mask) where:
            - delayed_data: Output data with all impairments applied [N, *data_shape]
            - valid_mask: Boolean mask [N] indicating successful transmission. True means
              has_source_data=True AND passed throttle AND not dropped by channel.
        """
        # Convert current_time to tensor if needed
        if isinstance(current_time, float):
            current_time_tensor = torch.full((self.num_envs,), current_time, device=self.device)
        else:
            current_time_tensor = current_time

        # Initialize source data availability
        if has_source_data is None:
            has_source_data = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Track total attempts
        self.total_attempts += 1

        # ===== STAGE 1: THROTTLE (Rate Limiting) =====
        # Check if enough time has passed since last throttled transmission
        time_since_last = current_time_tensor - self.last_throttle_time
        throttle_mask = time_since_last >= self.throttle_period  # [N]

        # Combine with source data availability
        throttle_mask = throttle_mask & has_source_data

        # Update throttle statistics
        self.throttled_count += (~throttle_mask & has_source_data).long()

        # ===== STAGE 2: DROPOUT (Packet Loss) =====
        # Only apply dropout to throttled attempts
        dropout_decision = torch.rand(self.num_envs, device=self.device) < self.dropout_rate
        dropout_mask = dropout_decision & throttle_mask  # Only drop throttled data

        # Final validity: throttled AND not dropped
        valid_mask = throttle_mask & ~dropout_mask  # [N]

        # Update dropout statistics
        self.dropout_count += dropout_mask.long()
        self.success_count += valid_mask.long()

        # ===== STAGE 3: LATENCY (Variable Delay) =====
        # Only buffer valid (non-dropped, throttled) data
        if valid_mask.any():
            # Generate random latencies for valid data
            latencies_sec = torch.normal(
                mean=self.mean_latency,
                std=self.std_latency
            ).clamp(min=0.0)

            # Convert to integer timesteps
            delay_steps = (latencies_sec / self.dt).round().long().clamp(min=0, max=self.max_history)

            # Set delays in latency buffer
            self.latency_buffer.set_time_lag(delay_steps)

            # Update latest data for valid samples
            self.latest_data = torch.where(
                valid_mask.view(-1, *([1]*len(self.data_shape))),
                data,
                self.latest_data
            )

            # Append new data to latency buffer (sample-and-hold with latest_data)
            self.latency_buffer.append(self.latest_data)

            # Update validity tracking
            self.has_valid_data = self.has_valid_data | valid_mask

            # Update time-based availability tracking
            self.data_added_time = torch.where(
                valid_mask,
                current_time_tensor,
                self.data_added_time
            )
            self.data_available_time = torch.where(
                valid_mask,
                current_time_tensor + latencies_sec,
                self.data_available_time
            )

            # Update throttle timestamp for successful transmissions
            self.last_throttle_time = torch.where(
                valid_mask,
                current_time_tensor,
                self.last_throttle_time
            )

        # Retrieve delayed data from latency buffer (read-only operation)
        # If buffer is empty (e.g., 100% dropout, no data ever appended), return latest_data
        try:
            delayed_data, sufficient_history = self.latency_buffer.get_delayed()
        except RuntimeError:
            # Buffer is empty, return latest_data (will be zeros initially)
            delayed_data = self.latest_data.clone()
            sufficient_history = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Return delayed data and validity mask
        # Note: valid_mask indicates NEW valid data this timestep (whether it passed impairments)
        # The delayed_data contains historically valid data (delayed)
        return delayed_data, valid_mask

    def get_delayed(self, current_time: Optional[torch.Tensor | float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get delayed data without updating buffer state (read-only).

        This method retrieves delayed data without applying any new impairments.
        It's designed for receiving/reading data that was already sent through
        the channel via compute().

        Args:
            current_time: Current simulation time [N] or scalar. If None, always returns data if available.

        Returns:
            Tuple of (delayed_data, data_available_mask, data_age) where:
            - delayed_data: Delayed data [N, *data_shape]
            - data_available_mask: Whether delayed data is available (valid data + time elapsed) [N]
            - data_age: Time elapsed since data was captured in seconds [N]. Zero if no data or current_time is None.
        """
        # Use get_delayed() to retrieve without modifying buffer
        # If buffer is empty, return latest_data (zeros initially)
        try:
            delayed_data, sufficient_history = self.latency_buffer.get_delayed()
        except RuntimeError:
            # Buffer is empty, return latest_data
            delayed_data = self.latest_data.clone()
            sufficient_history = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Check time-based availability and compute data age
        if current_time is not None:
            # Convert current_time to tensor if needed
            if isinstance(current_time, float):
                current_time_tensor = torch.full((self.num_envs,), current_time, device=self.device)
            else:
                current_time_tensor = current_time

            # Data is available if: has_valid_data AND (time >= data_available_time)
            time_elapsed = current_time_tensor >= self.data_available_time
            data_available = self.has_valid_data & time_elapsed

            # Compute data age: time since data was captured
            # For environments with no data, age is zero
            data_age = torch.where(
                self.has_valid_data,
                current_time_tensor - self.data_added_time,
                torch.zeros_like(current_time_tensor)
            )
        else:
            # No time check - use has_valid_data only
            data_available = self.has_valid_data.clone()
            # No current_time provided - cannot compute age
            data_age = torch.zeros(self.num_envs, device=self.device)

        return delayed_data, data_available, data_age

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get channel statistics.

        Returns:
            Dictionary with statistics:
                - total_attempts: Total number of compute calls [N]
                - throttled_count: Number throttled (rate limited) [N]
                - dropout_count: Number dropped (packet loss) [N]
                - success_count: Number successfully transmitted [N]
                - throttle_rate: Effective throttle rate [N]
                - dropout_rate_effective: Effective dropout rate [N]
                - success_rate: Overall success rate [N]
        """
        throttle_rate = torch.where(
            self.total_attempts > 0,
            self.throttled_count.float() / self.total_attempts.float(),
            torch.zeros_like(self.throttled_count, dtype=torch.float32)
        )

        dropout_rate_eff = torch.where(
            self.total_attempts > 0,
            self.dropout_count.float() / self.total_attempts.float(),
            torch.zeros_like(self.dropout_count, dtype=torch.float32)
        )

        success_rate = torch.where(
            self.total_attempts > 0,
            self.success_count.float() / self.total_attempts.float(),
            torch.zeros_like(self.success_count, dtype=torch.float32)
        )

        return {
            'total_attempts': self.total_attempts.clone(),
            'throttled_count': self.throttled_count.clone(),
            'dropout_count': self.dropout_count.clone(),
            'success_count': self.success_count.clone(),
            'throttle_rate': throttle_rate,
            'dropout_rate_effective': dropout_rate_eff,
            'success_rate': success_rate
        }

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset buffer for specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, resets all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids_list = None
        else:
            env_ids_list = env_ids.cpu().tolist()

        # Reset throttle state
        self.last_throttle_time[env_ids] = -float('inf')

        # Reset data
        self.latest_data[env_ids] = 0.0
        self.has_valid_data[env_ids] = False

        # Reset time-based availability tracking
        self.data_added_time[env_ids] = -float('inf')
        self.data_available_time[env_ids] = float('inf')

        # Reset latency buffer
        self.latency_buffer.reset(env_ids_list)

        # Reset statistics
        self.total_attempts[env_ids] = 0
        self.throttled_count[env_ids] = 0
        self.dropout_count[env_ids] = 0
        self.success_count[env_ids] = 0


class MultiAgentCommChannel:
    """Multi-agent communication channel with realistic impairments.

    Uses ChannelBuffer to simulate realistic inter-agent communication with:
    - Bandwidth throttling (message rate limiting)
    - Packet dropout (communication loss)
    - Variable latency (transmission delays)

    Each sender-receiver link has independent ChannelBuffers per data field.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        mean_latency: float,
        std_latency: float,
        throttle_period: float = 0.0,
        dropout_rate: float = 0.0,
        dt: float = 0.01,
        device: torch.device = torch.device('cpu'),
        max_history: int = 100
    ):
        """Initialize multi-agent communication channel.

        Args:
            num_envs: Number of parallel environments.
            num_agents: Number of agents in the system.
            mean_latency: Mean communication delay in seconds.
            std_latency: Std deviation of delay in seconds.
            throttle_period: Minimum period between messages in seconds (0 = no throttling).
            dropout_rate: Probability of packet loss (0.0 to 1.0).
            dt: Simulation timestep in seconds.
            device: Device to allocate tensors on.
            max_history: Maximum history length for latency buffer.
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.dt = dt
        self.device = device
        self.max_history = max_history

        # Communication parameters (per sender)
        self.mean_latency = torch.full((num_envs, num_agents), mean_latency, device=device)
        self.std_latency = torch.full((num_envs, num_agents), std_latency, device=device)
        self.throttle_period = torch.full((num_envs, num_agents), throttle_period, device=device)
        self.dropout_rate = torch.full((num_envs, num_agents), dropout_rate, device=device)

        # Structure: channel_buffers[sender_id][receiver_id][data_key] -> ChannelBuffer
        self.channel_buffers: Dict[int, Dict[int, Dict[str, ChannelBuffer]]] = {}
        self.data_shapes: Dict[int, Dict[int, Dict[str, Tuple[int, ...]]]] = {}

        # Initialize nested structure
        for sender_id in range(num_agents):
            self.channel_buffers[sender_id] = {}
            self.data_shapes[sender_id] = {}
            for receiver_id in range(num_agents):
                if sender_id != receiver_id:
                    self.channel_buffers[sender_id][receiver_id] = {}
                    self.data_shapes[sender_id][receiver_id] = {}

    def update_comm_params(
        self,
        mean_latency: Optional[torch.Tensor] = None,
        std_latency: Optional[torch.Tensor] = None,
        throttle_period: Optional[torch.Tensor] = None,
        dropout_rate: Optional[torch.Tensor] = None
    ):
        """Update communication parameters.

        Args:
            mean_latency: New mean latency [N, num_agents]. If None, unchanged.
            std_latency: New std latency [N, num_agents]. If None, unchanged.
            throttle_period: New throttle period [N, num_agents]. If None, unchanged.
            dropout_rate: New dropout rate [N, num_agents]. If None, unchanged.
        """
        if mean_latency is not None:
            self.mean_latency = mean_latency
        if std_latency is not None:
            self.std_latency = std_latency
        if throttle_period is not None:
            self.throttle_period = throttle_period
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate

        # Update all existing buffers
        for sender_id in range(self.num_agents):
            for receiver_id in range(self.num_agents):
                if sender_id == receiver_id:
                    continue
                for channel_buffer in self.channel_buffers[sender_id][receiver_id].values():
                    channel_buffer.update_params(
                        throttle_period=self.throttle_period[:, sender_id] if throttle_period is not None else None,
                        dropout_rate=self.dropout_rate[:, sender_id] if dropout_rate is not None else None,
                        mean_latency=self.mean_latency[:, sender_id] if mean_latency is not None else None,
                        std_latency=self.std_latency[:, sender_id] if std_latency is not None else None
                    )

    def send_message(
        self,
        sender_id: int,
        data: Dict[str, torch.Tensor],
        current_time: torch.Tensor | float,
        has_source_data: Optional[torch.Tensor] = None
    ):
        """Send message from sender to all receivers.

        Args:
            sender_id: ID of sending agent.
            data: Dictionary of data tensors {key: [N, ...]}.
            current_time: Current simulation time.
            has_source_data: Boolean mask [N] indicating whether each environment has
                meaningful source data to transmit. True means source has data available
                (e.g., sensor has valid measurement, target in range). Independent of
                channel impairments. If None, assumes all environments have data.
        """
        if has_source_data is None:
            has_source_data = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Send to each receiver
        for receiver_id in range(self.num_agents):
            if receiver_id == sender_id:
                continue

            # Process each data field
            for key, tensor in data.items():
                # Create ChannelBuffer on first use
                if key not in self.channel_buffers[sender_id][receiver_id]:
                    data_shape = tensor.shape[1:]
                    self.channel_buffers[sender_id][receiver_id][key] = ChannelBuffer(
                        num_envs=self.num_envs,
                        data_shape=data_shape,
                        throttle_period=self.throttle_period[:, sender_id],
                        dropout_rate=self.dropout_rate[:, sender_id],
                        mean_latency=self.mean_latency[:, sender_id],
                        std_latency=self.std_latency[:, sender_id],
                        dt=self.dt,
                        device=self.device,
                        max_history=self.max_history
                    )
                    self.data_shapes[sender_id][receiver_id][key] = data_shape

                # Send through channel (applies throttle, dropout, latency)
                # Note: We don't need the returned delayed_data here during send
                _ = self.channel_buffers[sender_id][receiver_id][key].compute(
                    data=tensor,
                    current_time=current_time,
                    has_source_data=has_source_data
                )

    def receive_messages(
        self,
        receiver_id: int,
        current_time: Optional[torch.Tensor | float] = None
    ) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Receive delayed messages from all senders.

        Args:
            receiver_id: ID of receiving agent.
            current_time: Current simulation time [N] or scalar. If None, no time-based checking.

        Returns:
            Dict mapping sender_id -> {data_key -> (data, valid_mask, data_age)}
            where valid_mask indicates if the buffer has ever received valid data,
            and data_age is the time elapsed since data was captured (in seconds).
        """
        received_data = {}

        for sender_id in range(self.num_agents):
            if sender_id == receiver_id:
                continue

            # Check if any data exists on this link
            if not self.channel_buffers[sender_id][receiver_id]:
                continue

            # Retrieve all data fields
            data_dict = {}
            for key, channel_buffer in self.channel_buffers[sender_id][receiver_id].items():
                # Get delayed data (read-only operation) - pass current_time for time-based availability
                delayed_data, valid_mask, data_age = channel_buffer.get_delayed(current_time)
                data_dict[key] = (delayed_data, valid_mask, data_age)

            if data_dict:
                received_data[sender_id] = data_dict

        return received_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics across all links.

        Returns:
            Dictionary mapping link names to their statistics.
        """
        stats = {}
        for sender_id in range(self.num_agents):
            for receiver_id in range(self.num_agents):
                if sender_id == receiver_id:
                    continue
                link_stats = {}
                for key, channel_buffer in self.channel_buffers[sender_id][receiver_id].items():
                    link_stats[key] = channel_buffer.get_statistics()
                if link_stats:
                    stats[f"link_{sender_id}_to_{receiver_id}"] = link_stats
        return stats

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset all channel buffers.

        Args:
            env_ids: Indices of environments to reset. If None, resets all.
        """
        for sender_id in range(self.num_agents):
            for receiver_id in range(self.num_agents):
                if sender_id == receiver_id:
                    continue
                for channel_buffer in self.channel_buffers[sender_id][receiver_id].values():
                    channel_buffer.reset(env_ids)


class MultiAgentObservationPipeline:
    """Multi-agent observation pipeline with realistic sensor modeling.

    Manages all observation-related delays and noise:
    - Ego motion filtering (IMU/state estimation lag)
    - Gimbal actuation dynamics
    - Vision detection with FPS throttling, processing latency, and failure rate
    - Inter-agent communication channel

    Uses composition of ChannelBuffer for unified impairment modeling.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        dt: float,
        device: torch.device,
        # Motion filter parameters
        motion_time_constant: float = 0.1,
        gimbal_time_constant: float = 0.03,
        # Detection parameters
        detection_fps: float = 30.0,
        detection_mean_latency: float = 0.05,
        detection_std_latency: float = 0.02,
        detection_failure_rate: float = 0.0,
        # Communication parameters
        comm_mean_delay: float = 0.1,
        comm_std_delay: float = 0.03,
        comm_throttle_period: float = 0.0,
        comm_dropout_rate: float = 0.05,
        max_buffer_size: int = 100
    ):
        """Initialize multi-agent observation pipeline.

        Args:
            num_envs: Number of parallel environments.
            num_agents: Number of agents in the system.
            dt: Simulation timestep in seconds.
            device: Device to allocate tensors on.
            motion_time_constant: Time constant for motion filters in seconds.
            gimbal_time_constant: Time constant for gimbal filters in seconds.
            detection_fps: Detection update frequency in Hz.
            detection_mean_latency: Mean detection processing delay in seconds.
            detection_std_latency: Std deviation of detection delay in seconds.
            detection_failure_rate: Detection dropout rate (0.0 to 1.0).
            comm_mean_delay: Mean communication delay in seconds.
            comm_std_delay: Std deviation of communication delay in seconds.
            comm_throttle_period: Communication bandwidth limit in seconds (0 = no limit).
            comm_dropout_rate: Communication packet loss rate (0.0 to 1.0).
            max_buffer_size: Maximum buffer size for latency buffers.
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.dt = dt
        self.device = device
        self.max_buffer_size = max_buffer_size

        # ===== MOTION FILTERS =====
        self.position_filters = {}
        self.orientation_filters = {}
        self.linear_velocity_filters = {}
        self.angular_velocity_filters = {}

        for agent_id in range(num_agents):
            self.position_filters[agent_id] = FirstOrderLag(
                num_envs, 3, motion_time_constant, dt, device
            )
            self.orientation_filters[agent_id] = QuaternionFirstOrderLag(
                num_envs,
                torch.full((num_envs,), motion_time_constant/10.0, device=device),
                dt, device
            )
            self.linear_velocity_filters[agent_id] = FirstOrderLag(
                num_envs, 3, motion_time_constant/5.0, dt, device
            )
            self.angular_velocity_filters[agent_id] = FirstOrderLag(
                num_envs, 3, motion_time_constant/100.0, dt, device
            )

        # ===== GIMBAL FILTERS =====
        self.gimbal_yaw_filters = {}
        self.gimbal_pitch_filters = {}

        for agent_id in range(num_agents):
            self.gimbal_yaw_filters[agent_id] = FirstOrderLag(
                num_envs, 1, gimbal_time_constant, dt, device
            )
            self.gimbal_pitch_filters[agent_id] = FirstOrderLag(
                num_envs, 1, gimbal_time_constant, dt, device
            )

        # ===== DETECTION SYSTEM (using ChannelBuffer) =====
        self.detection_channels: Dict[int, ChannelBuffer] = {}

        # Detection parameters [N, num_agents]
        self.detection_fps = torch.tensor(detection_fps, device=device).repeat(num_envs, num_agents)
        self.detection_throttle_period = torch.where(
            self.detection_fps > 0,
            1.0 / self.detection_fps,
            torch.zeros_like(self.detection_fps)
        )
        self.detection_mean_latency = torch.tensor(detection_mean_latency, device=device).repeat(num_envs, num_agents)
        self.detection_std_latency = torch.tensor(detection_std_latency, device=device).repeat(num_envs, num_agents)
        self.detection_failure_rate = torch.tensor(detection_failure_rate, device=device).repeat(num_envs, num_agents)

        # ===== COMMUNICATION SYSTEM (using MultiAgentCommChannel) =====
        self.comm_channel = MultiAgentCommChannel(
            num_envs=num_envs,
            num_agents=num_agents,
            mean_latency=comm_mean_delay,
            std_latency=comm_std_delay,
            throttle_period=comm_throttle_period,
            dropout_rate=comm_dropout_rate,
            dt=dt,
            device=device,
            max_history=max_buffer_size
        )

        # Time tracking
        self.current_time = torch.zeros(num_envs, device=device)

    def update_time(self, time_increment: Optional[float] = None):
        """Update simulation time.

        Args:
            time_increment: Time increment in seconds. If None, uses dt.
        """
        if time_increment is None:
            time_increment = self.dt
        self.current_time += time_increment

    # ===== MOTION FILTERING =====
    def update_ego_motion(
        self,
        agent_id: int,
        true_position: torch.Tensor,
        true_orientation: torch.Tensor,
        true_linear_velocity: torch.Tensor,
        true_angular_velocity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter ego motion states.

        Args:
            agent_id: Agent ID.
            true_position: True position [N, 3].
            true_orientation: True orientation quaternion [N, 4].
            true_linear_velocity: True linear velocity [N, 3].
            true_angular_velocity: True angular velocity [N, 3].

        Returns:
            Tuple of (filtered_pos, filtered_quat, filtered_lin_vel, filtered_ang_vel).
        """
        filtered_pos = self.position_filters[agent_id].update(true_position, self.current_time)
        filtered_quat = self.orientation_filters[agent_id].update(true_orientation, self.current_time)
        filtered_lin_vel = self.linear_velocity_filters[agent_id].update(true_linear_velocity, self.current_time)
        filtered_ang_vel = self.angular_velocity_filters[agent_id].update(true_angular_velocity, self.current_time)
        return filtered_pos, filtered_quat, filtered_lin_vel, filtered_ang_vel

    def update_ego_gimbal(
        self,
        agent_id: int,
        true_yaw: torch.Tensor,
        true_pitch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter gimbal angles.

        Args:
            agent_id: Agent ID.
            true_yaw: True yaw angle [N] or [N, 1].
            true_pitch: True pitch angle [N] or [N, 1].

        Returns:
            Tuple of (filtered_yaw, filtered_pitch).
        """
        if true_yaw.dim() == 1:
            true_yaw = true_yaw.unsqueeze(-1)
        if true_pitch.dim() == 1:
            true_pitch = true_pitch.unsqueeze(-1)

        filtered_yaw = self.gimbal_yaw_filters[agent_id].update(true_yaw, self.current_time)
        filtered_pitch = self.gimbal_pitch_filters[agent_id].update(true_pitch, self.current_time)
        return filtered_yaw, filtered_pitch

    # ===== DETECTION SYSTEM =====
    def add_detection(
        self,
        agent_id: int,
        detection_data: torch.Tensor,
        has_detection: torch.Tensor
    ):
        """Add detection data with FPS throttling, dropout, and latency.

        Args:
            agent_id: Agent ID.
            detection_data: Detection data [N, ...] (e.g., bboxes).
            has_detection: Boolean mask [N] indicating whether each environment has a
                valid detection from the sensor. True means sensor detected target
                (target in frame, not occluded). Independent of channel impairments.
        """
        # Create ChannelBuffer on first use
        if agent_id not in self.detection_channels:
            self.detection_channels[agent_id] = ChannelBuffer(
                num_envs=self.num_envs,
                data_shape=detection_data.shape[1:],
                throttle_period=self.detection_throttle_period[:, agent_id],
                dropout_rate=self.detection_failure_rate[:, agent_id],
                mean_latency=self.detection_mean_latency[:, agent_id],
                std_latency=self.detection_std_latency[:, agent_id],
                dt=self.dt,
                device=self.device,
                max_history=self.max_buffer_size
            )

        # Send through detection channel (handles FPS, dropout, latency automatically)
        _ = self.detection_channels[agent_id].compute(
            data=detection_data,
            current_time=self.current_time,
            has_source_data=has_detection
        )

    def get_delayed_detection(
        self,
        agent_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get delayed detection with validity mask and data age.

        Args:
            agent_id: Agent ID.

        Returns:
            Tuple of (detection_data, valid_mask, data_age) where:
            - detection_data: Detection data [N, 4]
            - valid_mask: Boolean mask indicating data availability [N]
            - data_age: Time elapsed since detection was captured in seconds [N]
        """
        if agent_id not in self.detection_channels:
            # No detections yet - return zeros
            return (
                torch.zeros(self.num_envs, 4, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )

        # Get delayed detection (read-only) - pass current_time for time-based availability check
        delayed_data, valid_mask, data_age = self.detection_channels[agent_id].get_delayed(self.current_time)
        return delayed_data, valid_mask, data_age

    # ===== COMMUNICATION =====
    def broadcast_state(
        self,
        sender_id: int,
        state_dict: Dict[str, torch.Tensor],
        has_source_data: Optional[torch.Tensor] = None
    ):
        """Broadcast state to all other agents.

        Args:
            sender_id: Sending agent ID.
            state_dict: Dictionary of state data {key: [N, ...]}.
            has_source_data: Boolean mask [N] indicating whether each environment has
                meaningful source data to transmit. True means source has data available.
                Independent of channel impairments. If None, assumes all have data.
        """
        self.comm_channel.send_message(
            sender_id=sender_id,
            data=state_dict,
            current_time=self.current_time,
            has_source_data=has_source_data
        )

    def receive_other_agent_states(
        self,
        receiver_id: int
    ) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Receive states from all other agents.

        Args:
            receiver_id: Receiving agent ID.

        Returns:
            Dict mapping sender_id -> {data_key -> (data, valid_mask, data_age)} where:
            - data: State data [N, ...]
            - valid_mask: Boolean mask indicating data availability [N]
            - data_age: Time elapsed since data was captured in seconds [N]
        """
        return self.comm_channel.receive_messages(receiver_id=receiver_id, current_time=self.current_time)

    # ===== PARAMETER UPDATES =====
    def update_detection_params(
        self,
        detection_fps: Optional[torch.Tensor] = None,
        detection_mean_latency: Optional[torch.Tensor] = None,
        detection_std_latency: Optional[torch.Tensor] = None,
        detection_failure_rate: Optional[torch.Tensor] = None
    ):
        """Update detection parameters.

        Args:
            detection_fps: New detection FPS [N, num_agents].
            detection_mean_latency: New mean latency [N, num_agents].
            detection_std_latency: New std latency [N, num_agents].
            detection_failure_rate: New failure rate [N, num_agents].
        """
        if detection_fps is not None:
            self.detection_fps = detection_fps
            self.detection_throttle_period = torch.where(
                detection_fps > 0,
                1.0 / detection_fps,
                torch.zeros_like(detection_fps)
            )
        if detection_mean_latency is not None:
            self.detection_mean_latency = detection_mean_latency
        if detection_std_latency is not None:
            self.detection_std_latency = detection_std_latency
        if detection_failure_rate is not None:
            self.detection_failure_rate = detection_failure_rate

        # Update existing channels
        for agent_id, channel in self.detection_channels.items():
            channel.update_params(
                throttle_period=self.detection_throttle_period[:, agent_id] if detection_fps is not None else None,
                dropout_rate=self.detection_failure_rate[:, agent_id] if detection_failure_rate is not None else None,
                mean_latency=self.detection_mean_latency[:, agent_id] if detection_mean_latency is not None else None,
                std_latency=self.detection_std_latency[:, agent_id] if detection_std_latency is not None else None
            )

    def update_comm_params(
        self,
        mean_latency: Optional[torch.Tensor] = None,
        std_latency: Optional[torch.Tensor] = None,
        throttle_period: Optional[torch.Tensor] = None,
        dropout_rate: Optional[torch.Tensor] = None
    ):
        """Update communication parameters.

        Args:
            mean_latency: New mean latency [N, num_agents].
            std_latency: New std latency [N, num_agents].
            throttle_period: New throttle period [N, num_agents].
            dropout_rate: New dropout rate [N, num_agents].
        """
        self.comm_channel.update_comm_params(
            mean_latency=mean_latency,
            std_latency=std_latency,
            throttle_period=throttle_period,
            dropout_rate=dropout_rate
        )

    # ===== STATISTICS & RESET =====
    def get_statistics(self) -> Dict[str, Any]:
        """Get all pipeline statistics.

        Returns:
            Dictionary with detection and communication statistics.
        """
        stats = {
            'current_time': self.current_time.clone(),
            'detection_stats': {},
            'comm_stats': self.comm_channel.get_statistics()
        }

        for agent_id, channel in self.detection_channels.items():
            stats['detection_stats'][f'agent_{agent_id}'] = channel.get_statistics()

        return stats

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset all components.

        Args:
            env_ids: Indices of environments to reset. If None, resets all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset filters
        for agent_id in range(self.num_agents):
            self.position_filters[agent_id].reset(env_ids)
            self.orientation_filters[agent_id].reset(env_ids)
            self.linear_velocity_filters[agent_id].reset(env_ids)
            self.angular_velocity_filters[agent_id].reset(env_ids)
            self.gimbal_yaw_filters[agent_id].reset(env_ids)
            self.gimbal_pitch_filters[agent_id].reset(env_ids)

        # Reset detection channels
        for channel in self.detection_channels.values():
            channel.reset(env_ids)

        # Reset communication
        self.comm_channel.reset(env_ids)

        # Reset time
        self.current_time[env_ids] = 0.0


class MultiAgentStateManager:
    """Extended multi-agent observation pipeline with integrated state storage.

    This class extends MultiAgentObservationPipeline to internally manage:
    - Ground truth (GT) states: Direct simulation states
    - Delayed states: GT + motion lag + detection latency (no noise) → for rewards
    - Delayed+noisy states: Delayed + sensor noise → for observations
    - Received states: States from other agents via communication channel

    The goal is to simplify the main environment module by centralizing all state
    processing logic inside this pipeline.
    """

    def __init__(
        self,
        possible_agents: list[str],
        num_envs: int,
        num_joints_per_agent: Dict[str, int],
        num_targets_per_agent: Dict[str, int],
        dt: float,
        device: torch.device,
        # Motion filter parameters
        motion_time_constant: float = 0.1,
        gimbal_time_constant: float = 0.03,
        # Detection parameters
        detection_fps: float = 30.0,
        detection_mean_latency: float = 0.05,
        detection_std_latency: float = 0.02,
        detection_failure_rate: float = 0.0,
        # Communication parameters
        comm_mean_delay: float = 0.1,
        comm_std_delay: float = 0.03,
        comm_throttle_period: float = 0.0,
        comm_dropout_rate: float = 0.05,
        # Noise parameters
        enable_noise: bool = True,
        position_noise_std: float = 0.01,
        orientation_noise_std: float = 0.01,
        linear_velocity_noise_std: Optional[float] = None,
        angular_velocity_noise_std: Optional[float] = None,
        linear_acceleration_noise_std: float = 100e-6,
        gimbal_noise_std: float = 0.01,
        zoom_noise_std: float = 0.01,
        bbox_noise_std: float = 1.0,
        noise_seed: Optional[int] = 0,
        max_buffer_size: int = 100
    ):
        """Initialize multi-agent state manager.

        Args:
            possible_agents: List of agent IDs (e.g., ["drone_0", "drone_1"]).
            num_envs: Number of parallel environments.
            num_joints_per_agent: Dict mapping agent_id -> number of joints.
            num_targets_per_agent: Dict mapping agent_id -> number of targets.
            dt: Simulation timestep in seconds.
            device: Device to allocate tensors on.
            motion_time_constant: Time constant for motion filters in seconds.
            gimbal_time_constant: Time constant for gimbal filters in seconds.
            detection_fps: Detection update frequency in Hz.
            detection_mean_latency: Mean detection processing delay in seconds.
            detection_std_latency: Std deviation of detection delay in seconds.
            detection_failure_rate: Detection dropout rate (0.0 to 1.0).
            comm_mean_delay: Mean communication delay in seconds.
            comm_std_delay: Std deviation of communication delay in seconds.
            comm_throttle_period: Communication bandwidth limit in seconds (0 = no limit).
            comm_dropout_rate: Communication packet loss rate (0.0 to 1.0).
            enable_noise: Whether to add noise to delayed states for observations.
            position_noise_std: Position noise standard deviation in meters.
            orientation_noise_std: Orientation noise standard deviation in radians.
            linear_velocity_noise_std: Linear velocity noise std in m/s.
            angular_velocity_noise_std: Angular velocity noise std in rad/s.
            linear_acceleration_noise_std: Linear acceleration noise std in m/s^2.
            gimbal_noise_std: Gimbal angle noise std in radians.
            zoom_noise_std: Zoom level noise std (relative).
            bbox_noise_std: Bbox noise std in pixels.
            noise_seed: Random seed for noise generator. If None, uses random seed.
            max_buffer_size: Maximum buffer size for latency buffers.
        """
        self.possible_agents = possible_agents
        self.num_envs = num_envs
        self.num_agents = len(possible_agents)
        self.dt = dt
        self.device = device
        self.enable_noise = enable_noise

        # Noise parameters
        self.position_noise_std = position_noise_std
        self.orientation_noise_std = orientation_noise_std
        # Compute linear and angular velocity noise from formulas if not provided (matching iris_ma_env3.py)
        self.linear_velocity_noise_std = linear_velocity_noise_std if linear_velocity_noise_std is not None else 0.1 * position_noise_std
        self.angular_velocity_noise_std = angular_velocity_noise_std if angular_velocity_noise_std is not None else orientation_noise_std * position_noise_std
        self.linear_acceleration_noise_std = linear_acceleration_noise_std
        self.gimbal_noise_std = gimbal_noise_std
        self.zoom_noise_std = zoom_noise_std
        self.bbox_noise_std = bbox_noise_std

        # Noise generator (seeded for reproducibility)
        if noise_seed is not None:
            self.noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
        else:
            self.noise_generator = torch.Generator(device=device)

        # Pre-allocated noise tensors (reused each step for efficiency)
        self.sampled_noise_pos = torch.zeros(num_envs, self.num_agents, 3, device=device)
        self.sampled_noise_ori = torch.zeros(num_envs, self.num_agents, 3, device=device)
        self.sampled_noise_lin_vel = torch.zeros(num_envs, self.num_agents, 3, device=device)
        self.sampled_noise_ang_vel = torch.zeros(num_envs, self.num_agents, 3, device=device)
        self.sampled_noise_lin_acc = torch.zeros(num_envs, self.num_agents, 3, device=device)
        self.sampled_noise_gimbal = torch.zeros(num_envs, self.num_agents, 3, device=device)
        self.sampled_noise_zoom = torch.zeros(num_envs, self.num_agents, device=device)

        # Pre-allocated bbox noise tensors (will be resized as needed)
        # Start with reasonable max size assumption
        max_targets = max(num_targets_per_agent.values()) if num_targets_per_agent else 1
        self.sampled_noise_bbox = torch.zeros(num_envs, self.num_agents, max_targets, 4, device=device)

        # Curriculum/progress scaling for noise (default to 1.0 = full noise)
        self.noise_progress_scale = 1.0

        # Create agent ID to index mapping
        self.agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(possible_agents)}

        # Initialize the base observation pipeline
        self.obs_pipeline = MultiAgentObservationPipeline(
            num_envs=num_envs,
            num_agents=self.num_agents,
            dt=dt,
            device=device,
            motion_time_constant=motion_time_constant,
            gimbal_time_constant=gimbal_time_constant,
            detection_fps=detection_fps,
            detection_mean_latency=detection_mean_latency,
            detection_std_latency=detection_std_latency,
            detection_failure_rate=detection_failure_rate,
            comm_mean_delay=comm_mean_delay,
            comm_std_delay=comm_std_delay,
            comm_throttle_period=comm_throttle_period,
            comm_dropout_rate=comm_dropout_rate,
            max_buffer_size=max_buffer_size
        )

        # ===== STATE STORAGE =====
        # Ground truth states (direct from simulation)
        self.gt_states = MultiAgentStates(
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent=num_joints_per_agent,
            num_targets_per_agent=num_targets_per_agent,
            device=device
        )

        # Delayed states (lag + latency, no noise) - for rewards
        self.delayed_states = MultiAgentStates(
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent=num_joints_per_agent,
            num_targets_per_agent=num_targets_per_agent,
            device=device
        )

        # Delayed + noisy states (lag + latency + noise) - for observations
        self.delayed_noisy_states = MultiAgentStates(
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent=num_joints_per_agent,
            num_targets_per_agent=num_targets_per_agent,
            device=device
        )

        # Received states storage (from communication channel)
        # Dict[receiver_agent_id][sender_agent_id] -> AgentStates
        self.received_states = {
            receiver_id: {} for receiver_id in possible_agents
        }

    def set_camera_configs(self, agent_id: str, **kwargs):
        """Set camera configuration for an agent.

        Args:
            agent_id: Agent ID.
            **kwargs: Camera configuration parameters (width, height, focal_length, etc.)
        """
        self.gt_states.agents[agent_id].set_camera_configs(**kwargs)
        self.delayed_states.agents[agent_id].set_camera_configs(**kwargs)
        self.delayed_noisy_states.agents[agent_id].set_camera_configs(**kwargs)

    def update_time(self, time_increment: Optional[float] = None):
        """Update simulation time.

        Args:
            time_increment: Time increment in seconds. If None, uses dt.
        """
        self.obs_pipeline.update_time(time_increment)

    def update_gt_states(
        self,
        agent_id: str,
        body_position_w: torch.Tensor,
        body_orientation_w: torch.Tensor,
        body_linear_velocity_w: torch.Tensor,
        body_angular_velocity_w: torch.Tensor,
        body_combined_angular_velocity_w: Optional[torch.Tensor] = None,
        body_linear_acceleration_w: Optional[torch.Tensor] = None,
        joint_positions_b: Optional[torch.Tensor] = None,
        joint_velocities_b: Optional[torch.Tensor] = None,
        zoom_level: Optional[torch.Tensor] = None,
        env_idxs: Optional[torch.Tensor] = None
    ):
        """Update ground truth states for an agent.

        This method stores GT states and triggers delayed state processing through
        the observation pipeline (motion filters, detection latency, etc.).

        Args:
            agent_id: Agent ID.
            body_position_w: Body position in world frame [N, 3].
            body_orientation_w: Body orientation quaternion in world frame [N, 4].
            body_linear_velocity_w: Linear velocity in world frame [N, 3].
            body_angular_velocity_w: Angular velocity in world frame [N, 3].
            body_combined_angular_velocity_w: Combined angular velocity (body + gimbal) in world frame [N, 3].
                This represents the angular velocity of the gimbal pitch link (or other specified body link)
                which incorporates both the robot body rotation and gimbal joint rotations.
            body_linear_acceleration_w: Linear acceleration in world frame [N, 3].
            joint_positions_b: Joint positions (gimbal angles) [N, J].
            joint_velocities_b: Joint velocities [N, J].
            zoom_level: Camera zoom level [N].
            env_idxs: Environment indices to update. If None, updates all.
        """
        if env_idxs is None:
            env_idxs = torch.arange(self.num_envs, device=self.device)

        agent_state = self.gt_states.agents[agent_id]

        # Store GT states
        agent_state.data.body_position_w[env_idxs] = body_position_w[env_idxs]
        agent_state.data.body_orientation_w[env_idxs] = body_orientation_w[env_idxs]
        agent_state.data.body_linear_velocity_w[env_idxs] = body_linear_velocity_w[env_idxs]
        agent_state.data.body_angular_velocity_w[env_idxs] = body_angular_velocity_w[env_idxs]

        if body_combined_angular_velocity_w is not None:
            agent_state.data.body_combined_angular_velocity_w[env_idxs] = body_combined_angular_velocity_w[env_idxs]
            # Also compute body frame version
            agent_state.data.body_combined_angular_velocity_b[env_idxs] = math_utils.quat_rotate_inverse(
                body_orientation_w[env_idxs], body_combined_angular_velocity_w[env_idxs]
            )

        if body_linear_acceleration_w is not None:
            agent_state.data.body_linear_acceleration_w[env_idxs] = body_linear_acceleration_w[env_idxs]

        if joint_positions_b is not None:
            agent_state.data.joint_positions_b[env_idxs] = joint_positions_b[env_idxs]

        if joint_velocities_b is not None:
            agent_state.data.joint_velocities_b[env_idxs] = joint_velocities_b[env_idxs]

        if zoom_level is not None:
            agent_state.data.camera_zoom_level[env_idxs] = zoom_level[env_idxs]

        # Update camera pose from body + gimbal
        agent_state.update_camera_pose(
            body_pos=agent_state.data.body_position_w,
            body_quat=agent_state.data.body_orientation_w,
            joint_pos=agent_state.data.joint_positions_b,
            env_idxs=env_idxs
        )

        # Update camera intrinsics from zoom
        agent_state.update_intrinsic_matrix(
            zoom_level=agent_state.data.camera_zoom_level,
            env_idxs=env_idxs
        )

        # ===== PROCESS DELAYED STATES (NO NOISE) =====
        agent_idx = self.agent_id_to_idx[agent_id]

        # Apply motion filters (lag only, no noise)
        filtered_pos, filtered_quat, filtered_lin_vel, filtered_ang_vel = \
            self.obs_pipeline.update_ego_motion(
                agent_id=agent_idx,
                true_position=body_position_w,
                true_orientation=body_orientation_w,
                true_linear_velocity=body_linear_velocity_w,
                true_angular_velocity=body_angular_velocity_w
            )

        # Apply gimbal filters
        if joint_positions_b is not None:
            gimbal_yaw = joint_positions_b[:, 2:3] if joint_positions_b.shape[1] > 2 else joint_positions_b[:, 0:1]
            gimbal_pitch = joint_positions_b[:, 1:2] if joint_positions_b.shape[1] > 1 else torch.zeros_like(gimbal_yaw)

            filtered_gimbal_yaw, filtered_gimbal_pitch = self.obs_pipeline.update_ego_gimbal(
                agent_id=agent_idx,
                true_yaw=gimbal_yaw,
                true_pitch=gimbal_pitch
            )

            # Reconstruct filtered joint positions
            if joint_positions_b.shape[1] == 3:
                filtered_joint_pos = torch.cat([
                    joint_positions_b[:, 0:1],  # Roll (unfiltered for now)
                    filtered_gimbal_pitch,
                    filtered_gimbal_yaw
                ], dim=-1)
            else:
                filtered_joint_pos = joint_positions_b.clone()
        else:
            filtered_joint_pos = None

        # Store delayed states (no noise)
        delayed_state = self.delayed_states.agents[agent_id]
        delayed_state.data.body_position_w[env_idxs] = filtered_pos[env_idxs]
        delayed_state.data.body_orientation_w[env_idxs] = filtered_quat[env_idxs]
        delayed_state.data.body_linear_velocity_w[env_idxs] = filtered_lin_vel[env_idxs]
        delayed_state.data.body_angular_velocity_w[env_idxs] = filtered_ang_vel[env_idxs]

        # Combined angular velocity (body + gimbal) - store in delayed states
        # This is a direct measurement from articulation, so we use filtered values as-is
        if body_combined_angular_velocity_w is not None:
            delayed_state.data.body_combined_angular_velocity_w[env_idxs] = body_combined_angular_velocity_w[env_idxs]
            # Compute body frame version using filtered quaternion
            delayed_state.data.body_combined_angular_velocity_b[env_idxs] = math_utils.quat_rotate_inverse(
                filtered_quat[env_idxs], body_combined_angular_velocity_w[env_idxs]
            )

        if filtered_joint_pos is not None:
            delayed_state.data.joint_positions_b[env_idxs] = filtered_joint_pos[env_idxs]

        if zoom_level is not None:
            delayed_state.data.camera_zoom_level[env_idxs] = zoom_level[env_idxs]

        # Update delayed camera pose
        delayed_state.update_camera_pose(
            body_pos=delayed_state.data.body_position_w,
            body_quat=delayed_state.data.body_orientation_w,
            joint_pos=delayed_state.data.joint_positions_b,
            env_idxs=env_idxs
        )

        delayed_state.update_intrinsic_matrix(
            zoom_level=delayed_state.data.camera_zoom_level,
            env_idxs=env_idxs
        )

        # ===== PROCESS DELAYED + NOISY STATES =====
        if self.enable_noise and self.noise_progress_scale > 0.0:
            # Sample noise once for this agent (using seeded generator)
            # Note: Follows the same noise formulas as iris_ma_env3.py for consistency

            # Position noise (scaled by progress)
            self.sampled_noise_pos[:, agent_idx, :].normal_(
                mean=0.0,
                std=self.position_noise_std * self.noise_progress_scale,
                generator=self.noise_generator
            )
            noisy_pos = filtered_pos + self.sampled_noise_pos[:, agent_idx, :]

            # Orientation noise (small angle approximation, scaled by progress)
            self.sampled_noise_ori[:, agent_idx, :].normal_(
                mean=0.0,
                std=self.orientation_noise_std * self.noise_progress_scale,
                generator=self.noise_generator
            )
            noise_quat = math_utils.quat_from_euler_xyz(
                self.sampled_noise_ori[:, agent_idx, 0],
                self.sampled_noise_ori[:, agent_idx, 1],
                self.sampled_noise_ori[:, agent_idx, 2]
            )
            noisy_quat = math_utils.quat_mul(filtered_quat, noise_quat)

            # Linear velocity noise (10% of position noise, as in original)
            self.sampled_noise_lin_vel[:, agent_idx, :].normal_(
                mean=0.0,
                std=0.1 * self.position_noise_std * self.noise_progress_scale,
                generator=self.noise_generator
            )
            noisy_lin_vel = filtered_lin_vel + self.sampled_noise_lin_vel[:, agent_idx, :]

            # Angular velocity noise (ori_std * pos_std, as in original)
            self.sampled_noise_ang_vel[:, agent_idx, :].normal_(
                mean=0.0,
                std=self.orientation_noise_std * self.position_noise_std * self.noise_progress_scale,
                generator=self.noise_generator
            )
            noisy_ang_vel = filtered_ang_vel + self.sampled_noise_ang_vel[:, agent_idx, :]

            # Combined angular velocity noise (same std as angular velocity)
            if body_combined_angular_velocity_w is not None:
                # Reuse the same noise tensor as angular velocity for consistency
                noisy_combined_ang_vel = body_combined_angular_velocity_w + self.sampled_noise_ang_vel[:, agent_idx, :]
            else:
                noisy_combined_ang_vel = None

            # Linear acceleration noise
            if body_linear_acceleration_w is not None:
                self.sampled_noise_lin_acc[:, agent_idx, :].normal_(
                    mean=0.0,
                    std=self.linear_acceleration_noise_std * self.noise_progress_scale,
                    generator=self.noise_generator
                )
                noisy_lin_acc = body_linear_acceleration_w + self.sampled_noise_lin_acc[:, agent_idx, :]
            else:
                noisy_lin_acc = None

            # Gimbal noise
            if filtered_joint_pos is not None:
                self.sampled_noise_gimbal[:, agent_idx, :filtered_joint_pos.shape[1]].normal_(
                    mean=0.0,
                    std=self.gimbal_noise_std * self.noise_progress_scale,
                    generator=self.noise_generator
                )
                noisy_joint_pos = filtered_joint_pos + self.sampled_noise_gimbal[:, agent_idx, :filtered_joint_pos.shape[1]]
            else:
                noisy_joint_pos = None

            # Zoom noise
            if zoom_level is not None:
                self.sampled_noise_zoom[:, agent_idx].normal_(
                    mean=0.0,
                    std=self.zoom_noise_std * self.noise_progress_scale,
                    generator=self.noise_generator
                )
                noisy_zoom = zoom_level + self.sampled_noise_zoom[:, agent_idx]
                noisy_zoom = torch.clamp(noisy_zoom, min=1.0, max=10.0)  # Reasonable zoom range
            else:
                noisy_zoom = None
        else:
            # No noise - use filtered states directly (clone to avoid reference issues)
            noisy_pos = filtered_pos.clone()
            noisy_quat = filtered_quat.clone()
            noisy_lin_vel = filtered_lin_vel.clone()
            noisy_ang_vel = filtered_ang_vel.clone()
            noisy_combined_ang_vel = body_combined_angular_velocity_w.clone() if body_combined_angular_velocity_w is not None else None
            noisy_lin_acc = body_linear_acceleration_w.clone() if body_linear_acceleration_w is not None else None
            noisy_joint_pos = filtered_joint_pos.clone() if filtered_joint_pos is not None else None
            noisy_zoom = zoom_level.clone() if zoom_level is not None else None

        # Store delayed + noisy states
        noisy_state = self.delayed_noisy_states.agents[agent_id]
        noisy_state.data.body_position_w[env_idxs] = noisy_pos[env_idxs]
        noisy_state.data.body_orientation_w[env_idxs] = noisy_quat[env_idxs]
        noisy_state.data.body_linear_velocity_w[env_idxs] = noisy_lin_vel[env_idxs]
        noisy_state.data.body_angular_velocity_w[env_idxs] = noisy_ang_vel[env_idxs]

        if noisy_combined_ang_vel is not None:
            noisy_state.data.body_combined_angular_velocity_w[env_idxs] = noisy_combined_ang_vel[env_idxs]
            # Compute body frame version using noisy quaternion
            noisy_state.data.body_combined_angular_velocity_b[env_idxs] = math_utils.quat_rotate_inverse(
                noisy_quat[env_idxs], noisy_combined_ang_vel[env_idxs]
            )

        if noisy_lin_acc is not None:
            noisy_state.data.body_linear_acceleration_w[env_idxs] = noisy_lin_acc[env_idxs]

        if noisy_joint_pos is not None:
            noisy_state.data.joint_positions_b[env_idxs] = noisy_joint_pos[env_idxs]

        if noisy_zoom is not None:
            noisy_state.data.camera_zoom_level[env_idxs] = noisy_zoom[env_idxs]

        # Update noisy camera pose
        noisy_state.update_camera_pose(
            body_pos=noisy_state.data.body_position_w,
            body_quat=noisy_state.data.body_orientation_w,
            joint_pos=noisy_state.data.joint_positions_b,
            env_idxs=env_idxs
        )

        noisy_state.update_intrinsic_matrix(
            zoom_level=noisy_state.data.camera_zoom_level,
            env_idxs=env_idxs
        )

    def update_detections(
        self,
        agent_id: str,
        bboxes_2d_gt: torch.Tensor,
        valid_mask_gt: torch.Tensor,
        env_idxs: Optional[torch.Tensor] = None
    ):
        """Update bbox detections with FPS throttling, dropout, and latency.

        This method processes GT bboxes through the detection channel and updates
        both delayed and delayed+noisy states with the processed detections.

        Args:
            agent_id: Agent ID.
            bboxes_2d_gt: Ground truth 2D bboxes [N, T, 4] (xywh format).
            valid_mask_gt: GT validity mask [N, T].
            env_idxs: Environment indices to update. If None, updates all.
        """
        if env_idxs is None:
            env_idxs = torch.arange(self.num_envs, device=self.device)

        agent_idx = self.agent_id_to_idx[agent_id]

        # Store GT bboxes
        self.gt_states.agents[agent_id].update_bboxes_2d(
            bboxes_2d=bboxes_2d_gt,
            valid_mask=valid_mask_gt,
            env_idxs=env_idxs
        )

        # Process through detection channel (adds FPS throttle, dropout, latency)
        # For simplicity, we'll process each target separately
        num_targets = bboxes_2d_gt.shape[1]
        for t in range(num_targets):
            bbox_t = bboxes_2d_gt[:, t, :]  # [N, 4]
            valid_t = valid_mask_gt[:, t]   # [N]

            # Add detection (applies FPS throttle, dropout, latency)
            self.obs_pipeline.add_detection(
                agent_id=agent_idx,
                detection_data=bbox_t,
                has_detection=valid_t
            )

        # Retrieve delayed detections (no noise)
        delayed_bboxes = []
        delayed_valid = []
        delayed_ages = []
        for t in range(num_targets):
            bbox_delayed, valid_delayed, data_age = self.obs_pipeline.get_delayed_detection(agent_idx)
            delayed_bboxes.append(bbox_delayed)
            delayed_valid.append(valid_delayed)
            delayed_ages.append(data_age)

        delayed_bboxes_stacked = torch.stack(delayed_bboxes, dim=1)  # [N, T, 4]
        delayed_valid_stacked = torch.stack(delayed_valid, dim=1)    # [N, T]
        delayed_ages_stacked = torch.stack(delayed_ages, dim=1)      # [N, T]

        # Update delayed states
        self.delayed_states.agents[agent_id].update_bboxes_2d(
            bboxes_2d=delayed_bboxes_stacked,
            valid_mask=delayed_valid_stacked,
            env_idxs=env_idxs
        )

        # Store detection ages in delayed states
        self.delayed_states.agents[agent_id].data.bboxes_2d_age[env_idxs] = delayed_ages_stacked[env_idxs]

        # Add bbox noise for observations
        if self.enable_noise:
            # Use seeded generator and progress scaling for consistency
            num_targets = delayed_bboxes_stacked.shape[1]

            # Ensure bbox noise tensor is large enough
            if self.sampled_noise_bbox.shape[2] < num_targets:
                self.sampled_noise_bbox = torch.zeros(
                    self.num_envs, self.num_agents, num_targets, 4,
                    device=self.device
                )

            # Sample bbox noise (pixel-space noise)
            self.sampled_noise_bbox[:, agent_idx, :num_targets, :].normal_(
                mean=0.0,
                std=self.bbox_noise_std * self.noise_progress_scale,
                generator=self.noise_generator
            )
            noisy_bboxes = delayed_bboxes_stacked + self.sampled_noise_bbox[:, agent_idx, :num_targets, :]
        else:
            noisy_bboxes = delayed_bboxes_stacked

        # Update delayed + noisy states
        self.delayed_noisy_states.agents[agent_id].update_bboxes_2d(
            bboxes_2d=noisy_bboxes,
            valid_mask=delayed_valid_stacked,
            env_idxs=env_idxs
        )

        # Store detection ages in delayed+noisy states (same age, different bbox due to noise)
        self.delayed_noisy_states.agents[agent_id].data.bboxes_2d_age[env_idxs] = delayed_ages_stacked[env_idxs]

        # Update camera rays for both delayed and delayed+noisy states
        for state_manager, bboxes in [
            (self.delayed_states, delayed_bboxes_stacked),
            (self.delayed_noisy_states, noisy_bboxes)
        ]:
            agent_state = state_manager.agents[agent_id]
            agent_state.update_camera_rays_to_bboxes(
                bboxes_2d=bboxes,
                camera_intrinsics=agent_state.data.camera_intrinsics,
                camera_position=agent_state.data.camera_position_w,
                camera_orientation=agent_state.data.camera_orientation_w,
                env_idxs=env_idxs
            )

    def broadcast_state(
        self,
        sender_id: str,
        state_keys: list[str] = None,
        has_source_data: Optional[torch.Tensor] = None
    ):
        """Broadcast delayed states to all other agents via communication channel.

        Args:
            sender_id: Sending agent ID.
            state_keys: List of state keys to broadcast. If None, broadcasts all.
            has_source_data: Boolean mask [N] indicating valid data. If None, assumes all valid.
        """
        agent_idx = self.agent_id_to_idx[sender_id]
        delayed_state = self.delayed_states.agents[sender_id]

        # Build state dict to broadcast
        if state_keys is None:
            state_keys = [
                'position', 'orientation', 'linear_velocity', 'angular_velocity',
                'combined_angular_velocity',
                'camera_position', 'camera_orientation', 'camera_intrinsics',
                'bboxes_2d', 'bboxes_2d_valid_mask', 'camera_ray_directions_w'
            ]

        state_dict = {}
        for key in state_keys:
            if key == 'position':
                state_dict[key] = delayed_state.data.body_position_w
            elif key == 'orientation':
                state_dict[key] = delayed_state.data.body_orientation_w
            elif key == 'linear_velocity':
                state_dict[key] = delayed_state.data.body_linear_velocity_w
            elif key == 'angular_velocity':
                state_dict[key] = delayed_state.data.body_angular_velocity_w
            elif key == 'combined_angular_velocity':
                state_dict[key] = delayed_state.data.body_combined_angular_velocity_w
            elif key == 'camera_position':
                state_dict[key] = delayed_state.data.camera_position_w
            elif key == 'camera_orientation':
                state_dict[key] = delayed_state.data.camera_orientation_w
            elif key == 'camera_intrinsics':
                state_dict[key] = delayed_state.data.camera_intrinsics
            elif key == 'bboxes_2d':
                state_dict[key] = delayed_state.data.bboxes_2d
            elif key == 'bboxes_2d_valid_mask':
                state_dict[key] = delayed_state.data.bboxes_2d_valid_mask.float()
            elif key == 'camera_ray_directions_w':
                state_dict[key] = delayed_state.data.camera_ray_directions_w

        # Broadcast through communication channel
        self.obs_pipeline.broadcast_state(
            sender_id=agent_idx,
            state_dict=state_dict,
            has_source_data=has_source_data
        )

    def receive_other_agent_states(
        self,
        receiver_id: str
    ) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Receive states from all other agents through communication channel.

        Args:
            receiver_id: Receiving agent ID.

        Returns:
            Dict mapping sender_agent_id -> {state_key -> (data, valid_mask, data_age)}
        """
        agent_idx = self.agent_id_to_idx[receiver_id]

        # Receive from communication channel (returns with agent indices)
        received_data = self.obs_pipeline.receive_other_agent_states(receiver_id=agent_idx)

        # Convert agent indices back to agent IDs
        received_states = {}
        for sender_idx, data_dict in received_data.items():
            sender_id = self.possible_agents[sender_idx]
            received_states[sender_id] = data_dict

        return received_states

    def get_gt_states(self, agent_id: str) -> AgentStates:
        """Get ground truth states for an agent.

        Args:
            agent_id: Agent ID.

        Returns:
            AgentStates object with GT states.
        """
        return self.gt_states.agents[agent_id]

    def get_delayed_states(self, agent_id: str) -> AgentStates:
        """Get delayed states (no noise) for an agent - use for reward computation.

        Args:
            agent_id: Agent ID.

        Returns:
            AgentStates object with delayed states (lag + latency, no noise).
        """
        return self.delayed_states.agents[agent_id]

    def get_delayed_noisy_states(self, agent_id: str) -> AgentStates:
        """Get delayed + noisy states for an agent - use for observations.

        Args:
            agent_id: Agent ID.

        Returns:
            AgentStates object with delayed + noisy states (lag + latency + noise).
        """
        return self.delayed_noisy_states.agents[agent_id]

    def get_all_gt_states(self) -> MultiAgentStates:
        """Get all GT states.

        Returns:
            MultiAgentStates object with all GT states.
        """
        return self.gt_states

    def get_all_delayed_states(self) -> MultiAgentStates:
        """Get all delayed states (no noise).

        Returns:
            MultiAgentStates object with all delayed states.
        """
        return self.delayed_states

    def get_all_delayed_noisy_states(self) -> MultiAgentStates:
        """Get all delayed + noisy states.

        Returns:
            MultiAgentStates object with all delayed + noisy states.
        """
        return self.delayed_noisy_states

    def set_noise_progress_scale(self, progress: float):
        """Set noise curriculum/progress scaling factor.

        This scales all noise standard deviations by the given factor, allowing
        for curriculum learning where noise gradually increases during training.

        Args:
            progress: Scaling factor for noise (0.0 to 1.0 typical range).
                     0.0 = no noise, 1.0 = full noise as specified in init.
        """
        self.noise_progress_scale = progress

    def update_detection_params(self, **kwargs):
        """Update detection parameters. See MultiAgentObservationPipeline.update_detection_params."""
        self.obs_pipeline.update_detection_params(**kwargs)

    def update_comm_params(self, **kwargs):
        """Update communication parameters. See MultiAgentObservationPipeline.update_comm_params."""
        self.obs_pipeline.update_comm_params(**kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics. See MultiAgentObservationPipeline.get_statistics."""
        return self.obs_pipeline.get_statistics()

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset all components for specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, resets all.
        """
        # Reset the underlying pipeline
        self.obs_pipeline.reset(env_ids)

        # Note: We don't reset state storage here as it will be overwritten
        # by the next update_gt_states call from the environment
