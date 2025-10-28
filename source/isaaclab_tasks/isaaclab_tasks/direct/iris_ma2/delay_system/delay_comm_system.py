# delay_comm_system.py
"""
Comprehensive delay and communication system for multi-agent robotic environments.

Features:
- Timestamped data structures
- First-order lag filters for position, orientation, and gimbal angles
- Random latency for detections
- Communication delays (latency)
- Communication dropouts (packet loss)
"""

import torch
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import math


@dataclass
class TimestampedData:
    """Container for data with timestamp information."""
    data: torch.Tensor
    timestamp: torch.Tensor  # [N] - timestamp for each environment
    valid: torch.Tensor  # [N] - validity mask for each environment
    
    def clone(self):
        """Deep copy of timestamped data."""
        return TimestampedData(
            data=self.data.clone(),
            timestamp=self.timestamp.clone(),
            valid=self.valid.clone()
        )


class FirstOrderLag:
    """
    First-order lag filter for smooth state estimation.
    Simulates navigation filter delays in real robotic systems.
    
    The filter follows: x_filtered(t) = x_filtered(t-1) + alpha * (x_measured(t) - x_filtered(t-1))
    where alpha = dt / (dt + tau), and tau is the time constant.
    """
    
    def __init__(
        self, 
        num_envs: int,
        state_dim: int,
        time_constant: float,
        dt: float,
        device: torch.device,
        initial_state: Optional[torch.Tensor] = None
    ):
        """
        Args:
            num_envs: Number of parallel environments
            state_dim: Dimension of the state vector
            time_constant: Time constant tau (seconds). Smaller = faster response
            dt: Timestep (seconds)
            device: Torch device
            initial_state: Initial state [N, state_dim]. If None, starts at zeros
        """
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.tau = time_constant
        self.dt = dt
        self.device = device
        
        # Compute filter coefficient
        self.alpha = dt / (dt + time_constant)
        
        # Initialize filtered state
        if initial_state is not None:
            self.filtered_state = initial_state.clone()
        else:
            self.filtered_state = torch.zeros(num_envs, state_dim, device=device)
        
        # Store timestamp
        self.timestamp = torch.zeros(num_envs, device=device)
    
    def update(self, measured_state: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        """
        Update the filter with new measurement.
        
        Args:
            measured_state: New measurement [N, state_dim]
            current_time: Current simulation time [N] or scalar
            
        Returns:
            filtered_state: Filtered state [N, state_dim]
        """
        # Update filtered state
        self.filtered_state = self.filtered_state + self.alpha * (measured_state - self.filtered_state)
        
        # Update timestamp
        if isinstance(current_time, torch.Tensor):
            self.timestamp = current_time.clone()
        else:
            self.timestamp.fill_(current_time)
        
        return self.filtered_state.clone()
    
    def reset(self, env_ids: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None):
        """
        Reset filter for specific environments.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all
            state: State to reset to [len(env_ids), state_dim]. If None, reset to zeros
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if state is not None:
            self.filtered_state[env_ids] = state
        else:
            self.filtered_state[env_ids] = 0.0
        
        self.timestamp[env_ids] = 0.0


class QuaternionFirstOrderLag:
    """
    First-order lag filter specifically for quaternion orientations.
    Uses SLERP (Spherical Linear Interpolation) for proper quaternion filtering.
    """
    
    def __init__(
        self, 
        num_envs: int,
        time_constant: float,
        dt: float,
        device: torch.device,
        initial_quat: Optional[torch.Tensor] = None
    ):
        """
        Args:
            num_envs: Number of parallel environments
            time_constant: Time constant tau (seconds)
            dt: Timestep (seconds)
            device: Torch device
            initial_quat: Initial quaternion [N, 4] (w, x, y, z). If None, identity
        """
        self.num_envs = num_envs
        self.tau = time_constant
        self.dt = dt
        self.device = device
        
        # Compute filter coefficient (SLERP parameter)
        self.alpha = dt / (dt + time_constant)
        
        # Initialize filtered quaternion
        if initial_quat is not None:
            self.filtered_quat = initial_quat.clone()
        else:
            # Identity quaternion
            self.filtered_quat = torch.zeros(num_envs, 4, device=device)
            self.filtered_quat[:, 0] = 1.0  # w = 1
        
        self.timestamp = torch.zeros(num_envs, device=device)
    
    def update(self, measured_quat: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        """
        Update filter using SLERP.
        
        Args:
            measured_quat: New measurement [N, 4] (w, x, y, z)
            current_time: Current simulation time [N] or scalar
            
        Returns:
            filtered_quat: Filtered quaternion [N, 4]
        """
        # SLERP interpolation
        self.filtered_quat = self._slerp(self.filtered_quat, measured_quat, self.alpha)
        
        # Update timestamp
        if isinstance(current_time, torch.Tensor):
            self.timestamp = current_time.clone()
        else:
            self.timestamp.fill_(current_time)
        
        return self.filtered_quat.clone()
    
    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        """
        Spherical linear interpolation between quaternions.
        
        Args:
            q0: Starting quaternion [N, 4]
            q1: Ending quaternion [N, 4]
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated quaternion [N, 4]
        """
        # Normalize quaternions
        q0 = q0 / (torch.norm(q0, dim=-1, keepdim=True) + 1e-8)
        q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
        
        # Compute dot product
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        
        # If dot product is negative, negate one quaternion (take shorter path)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.abs(dot)
        
        # If quaternions are very close, use linear interpolation
        DOT_THRESHOLD = 0.9995
        mask = (dot > DOT_THRESHOLD).squeeze(-1)
        
        # SLERP computation
        theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_theta = torch.sin(theta)
        
        # Avoid division by zero
        sin_theta = torch.where(sin_theta < 1e-8, torch.ones_like(sin_theta), sin_theta)
        
        w0 = torch.sin((1 - t) * theta) / sin_theta
        w1 = torch.sin(t * theta) / sin_theta
        
        result_slerp = w0 * q0 + w1 * q1
        
        # Linear interpolation for near-parallel quaternions
        result_lerp = (1 - t) * q0 + t * q1
        result_lerp = result_lerp / (torch.norm(result_lerp, dim=-1, keepdim=True) + 1e-8)
        
        # Combine based on mask
        result = torch.where(mask.unsqueeze(-1), result_lerp, result_slerp)
        
        return result
    
    def reset(self, env_ids: Optional[torch.Tensor] = None, quat: Optional[torch.Tensor] = None):
        """Reset filter for specific environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if quat is not None:
            self.filtered_quat[env_ids] = quat
        else:
            # Identity quaternion
            self.filtered_quat[env_ids] = torch.zeros(len(env_ids), 4, device=self.device)
            self.filtered_quat[env_ids, 0] = 1.0
        
        self.timestamp[env_ids] = 0.0


class LatencyBuffer:
    """
    Buffer for adding random latency to detection data.
    Simulates processing delays in detection/tracking systems.
    """
    
    def __init__(
        self,
        num_envs: int,
        data_shape: Tuple[int, ...],
        mean_latency: float,
        std_latency: float,
        dt: float,
        device: torch.device,
        max_buffer_size: int = 100
    ):
        """
        Args:
            num_envs: Number of parallel environments
            data_shape: Shape of data per environment (e.g., (num_targets, 4) for bboxes)
            mean_latency: Mean latency in seconds
            std_latency: Standard deviation of latency in seconds
            dt: Timestep (seconds)
            device: Torch device
            max_buffer_size: Maximum buffer entries per environment
        """
        self.num_envs = num_envs
        self.data_shape = data_shape
        self.mean_latency = mean_latency
        self.std_latency = std_latency
        self.dt = dt
        self.device = device
        self.max_buffer_size = max_buffer_size
        
        # Buffer stores list of (data, timestamp, valid) tuples for each environment
        self.buffer = [[] for _ in range(num_envs)]
    
    def add_measurement(
        self, 
        data: torch.Tensor, 
        current_time: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ):
        """
        Add new measurement to buffer with random latency.
        
        Args:
            data: Measurement data [N, *data_shape]
            current_time: Current simulation time [N] or scalar
            valid_mask: Validity mask [N]. If None, all valid
        """
        if valid_mask is None:
            valid_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Generate random latencies for each environment
        latencies = torch.normal(
            mean=self.mean_latency,
            std=self.std_latency,
            size=(self.num_envs,),
            device=self.device
        ).clamp(min=0.0)  # Ensure non-negative
        
        # Compute delivery times
        if isinstance(current_time, torch.Tensor):
            delivery_times = current_time + latencies
        else:
            delivery_times = torch.full((self.num_envs,), current_time, device=self.device) + latencies
        
        # Add to buffer for each environment
        for env_id in range(self.num_envs):
            if valid_mask[env_id]:
                self.buffer[env_id].append({
                    'data': data[env_id].clone(),
                    'delivery_time': delivery_times[env_id].item(),
                    'generation_time': current_time[env_id].item() if isinstance(current_time, torch.Tensor) else current_time
                })
                
                # Maintain buffer size
                if len(self.buffer[env_id]) > self.max_buffer_size:
                    self.buffer[env_id].pop(0)
    
    def get_delayed_measurement(
        self, 
        current_time: torch.Tensor
    ) -> TimestampedData:
        """
        Retrieve measurements that should be delivered at current time.
        
        Args:
            current_time: Current simulation time [N] or scalar
            
        Returns:
            TimestampedData with delayed measurements
        """
        result_data = torch.zeros(self.num_envs, *self.data_shape, device=self.device)
        result_timestamp = torch.zeros(self.num_envs, device=self.device)
        result_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for env_id in range(self.num_envs):
            if len(self.buffer[env_id]) == 0:
                continue
            
            current_t = current_time[env_id].item() if isinstance(current_time, torch.Tensor) else current_time
            
            # Find measurements ready for delivery
            ready_measurements = [m for m in self.buffer[env_id] if m['delivery_time'] <= current_t]
            
            if ready_measurements:
                # Get most recent ready measurement
                latest = max(ready_measurements, key=lambda m: m['generation_time'])
                
                result_data[env_id] = latest['data']
                result_timestamp[env_id] = latest['generation_time']
                result_valid[env_id] = True
                
                # Remove delivered measurements
                self.buffer[env_id] = [m for m in self.buffer[env_id] 
                                      if m['delivery_time'] > current_t]
        
        return TimestampedData(
            data=result_data,
            timestamp=result_timestamp,
            valid=result_valid
        )
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Clear buffer for specific environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        for env_id in env_ids.cpu().numpy():
            self.buffer[env_id].clear()


class CommManager:
    """
    Communication manager handling delays and dropouts between agents.
    Simulates realistic multi-agent communication constraints.
    """
    
    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        mean_comm_delay: float,
        std_comm_delay: float,
        dropout_rate: float,
        dt: float,
        device: torch.device,
        max_buffer_size: int = 100
    ):
        """
        Args:
            num_envs: Number of parallel environments
            num_agents: Number of agents
            mean_comm_delay: Mean communication delay in seconds
            std_comm_delay: Standard deviation of communication delay
            dropout_rate: Probability of packet dropout [0, 1]
            dt: Timestep (seconds)
            device: Torch device
            max_buffer_size: Maximum buffer entries per communication link
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.mean_comm_delay = mean_comm_delay
        self.std_comm_delay = std_comm_delay
        self.dropout_rate = dropout_rate
        self.dt = dt
        self.device = device
        self.max_buffer_size = max_buffer_size
        
        # Buffer for each communication link: [env_id][sender_id][receiver_id]
        self.comm_buffers = [
            [[[] for _ in range(num_agents)] for _ in range(num_agents)]
            for _ in range(num_envs)
        ]
        
        # Track last received data timestamp for each link
        self.last_received_time = torch.zeros(
            num_envs, num_agents, num_agents, device=device
        )
    
    def send_message(
        self,
        sender_id: int,
        data: Dict[str, torch.Tensor],
        current_time: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ):
        """
        Send message from one agent to all others (broadcast).
        
        Args:
            sender_id: ID of sending agent
            data: Dictionary of data tensors {key: [N, ...]}
            current_time: Current simulation time [N] or scalar
            valid_mask: Validity mask [N]. If None, all valid
        """
        if valid_mask is None:
            valid_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Generate random communication delays
        comm_delays = torch.normal(
            mean=self.mean_comm_delay,
            std=self.std_comm_delay,
            size=(self.num_envs,),
            device=self.device
        ).clamp(min=0.0)
        
        # Generate dropout mask (True = packet lost)
        dropout_mask = torch.rand(self.num_envs, device=self.device) < self.dropout_rate
        
        # Compute delivery times
        if isinstance(current_time, torch.Tensor):
            delivery_times = current_time + comm_delays
            gen_times = current_time
        else:
            delivery_times = torch.full((self.num_envs,), current_time, device=self.device) + comm_delays
            gen_times = torch.full((self.num_envs,), current_time, device=self.device)
        
        # Send to all receivers (except self)
        for receiver_id in range(self.num_agents):
            if receiver_id == sender_id:
                continue
            
            for env_id in range(self.num_envs):
                # Check if valid and not dropped
                if valid_mask[env_id] and not dropout_mask[env_id]:
                    # Clone data for this environment
                    env_data = {key: tensor[env_id].clone() for key, tensor in data.items()}
                    
                    self.comm_buffers[env_id][sender_id][receiver_id].append({
                        'data': env_data,
                        'delivery_time': delivery_times[env_id].item(),
                        'generation_time': gen_times[env_id].item(),
                        'sender_id': sender_id
                    })
                    
                    # Maintain buffer size
                    if len(self.comm_buffers[env_id][sender_id][receiver_id]) > self.max_buffer_size:
                        self.comm_buffers[env_id][sender_id][receiver_id].pop(0)
    
    def receive_messages(
        self,
        receiver_id: int,
        current_time: torch.Tensor,
        data_template: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[int, TimestampedData]:
        """
        Receive all messages for a specific agent.
        
        Args:
            receiver_id: ID of receiving agent
            current_time: Current simulation time [N] or scalar
            data_template: Template for data structure. If None, use first received message
            
        Returns:
            Dictionary mapping sender_id to TimestampedData
        """
        received_data = {}
        
        for sender_id in range(self.num_agents):
            if sender_id == receiver_id:
                continue
            
            # Initialize result containers
            if data_template is not None:
                result_data = {
                    key: torch.zeros(self.num_envs, *tensor.shape[1:], device=self.device)
                    for key, tensor in data_template.items()
                }
            else:
                result_data = None
            
            result_timestamp = torch.zeros(self.num_envs, device=self.device)
            result_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
            # Check each environment
            for env_id in range(self.num_envs):
                buffer = self.comm_buffers[env_id][sender_id][receiver_id]
                
                if len(buffer) == 0:
                    continue
                
                current_t = current_time[env_id].item() if isinstance(current_time, torch.Tensor) else current_time
                
                # Find messages ready for delivery
                ready_messages = [m for m in buffer if m['delivery_time'] <= current_t]
                
                if ready_messages:
                    # Get most recent ready message
                    latest = max(ready_messages, key=lambda m: m['generation_time'])
                    
                    # Initialize result_data structure if needed
                    if result_data is None:
                        result_data = {
                            key: torch.zeros(self.num_envs, *tensor.shape, device=self.device)
                            for key, tensor in latest['data'].items()
                        }
                    
                    # Copy data
                    for key, value in latest['data'].items():
                        result_data[key][env_id] = value
                    
                    result_timestamp[env_id] = latest['generation_time']
                    result_valid[env_id] = True
                    self.last_received_time[env_id, sender_id, receiver_id] = current_t
                    
                    # Remove delivered messages
                    self.comm_buffers[env_id][sender_id][receiver_id] = [
                        m for m in buffer if m['delivery_time'] > current_t
                    ]
            
            if result_data is not None:
                received_data[sender_id] = {
                    key: TimestampedData(
                        data=tensor,
                        timestamp=result_timestamp.clone(),
                        valid=result_valid.clone()
                    ) for key, tensor in result_data.items()
                }
        
        return received_data
    
    def get_comm_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Get communication statistics for monitoring.
        
        Returns:
            Dictionary with statistics:
                - buffer_sizes: [N, num_agents, num_agents] current buffer sizes
                - last_received: [N, num_agents, num_agents] time since last message
        """
        # Current buffer sizes
        buffer_sizes = torch.zeros(
            self.num_envs, self.num_agents, self.num_agents,
            dtype=torch.long, device=self.device
        )
        
        for env_id in range(self.num_envs):
            for sender_id in range(self.num_agents):
                for receiver_id in range(self.num_agents):
                    buffer_sizes[env_id, sender_id, receiver_id] = len(
                        self.comm_buffers[env_id][sender_id][receiver_id]
                    )
        
        return {
            'buffer_sizes': buffer_sizes,
            'last_received_time': self.last_received_time.clone()
        }
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Clear buffers for specific environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        for env_id in env_ids.cpu().numpy():
            for sender_id in range(self.num_agents):
                for receiver_id in range(self.num_agents):
                    self.comm_buffers[env_id][sender_id][receiver_id].clear()
        
        self.last_received_time[env_ids] = 0.0


class DelayedObservationManager:
    """
    High-level manager for all delayed observations in a multi-agent system.
    Combines first-order lags, latency buffers, and communication management.
    """
    
    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        dt: float,
        device: torch.device,
        # Motion delays (first-order lag)
        motion_time_constant: float = 0.1,  # 100ms lag for position/orientation
        # Gimbal delays (faster first-order lag)
        gimbal_time_constant: float = 0.03,  # 30ms lag for gimbal angles
        # Detection delays (random latency)
        detection_mean_latency: float = 0.05,  # 50ms mean
        detection_std_latency: float = 0.02,   # 20ms std
        # Communication delays
        comm_mean_delay: float = 0.1,    # 100ms mean delay
        comm_std_delay: float = 0.03,    # 30ms std delay
        comm_dropout_rate: float = 0.05  # 5% packet loss
    ):
        """
        Args:
            num_envs: Number of parallel environments
            num_agents: Number of agents per environment
            dt: Timestep (seconds)
            device: Torch device
            motion_time_constant: Time constant for position/orientation filtering
            gimbal_time_constant: Time constant for gimbal angle filtering
            detection_mean_latency: Mean detection latency
            detection_std_latency: Std of detection latency
            comm_mean_delay: Mean communication delay between agents
            comm_std_delay: Std of communication delay
            comm_dropout_rate: Communication dropout probability
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.dt = dt
        self.device = device
        
        # Create first-order lag filters for each agent's ego observations
        self.position_filters = {}
        self.orientation_filters = {}
        self.gimbal_yaw_filters = {}
        self.gimbal_pitch_filters = {}
        
        for agent_id in range(num_agents):
            # Position filter (3D)
            self.position_filters[agent_id] = FirstOrderLag(
                num_envs=num_envs,
                state_dim=3,
                time_constant=motion_time_constant,
                dt=dt,
                device=device
            )
            
            # Orientation filter (quaternion)
            self.orientation_filters[agent_id] = QuaternionFirstOrderLag(
                num_envs=num_envs,
                time_constant=motion_time_constant,
                dt=dt,
                device=device
            )
            
            # Gimbal yaw filter (scalar)
            self.gimbal_yaw_filters[agent_id] = FirstOrderLag(
                num_envs=num_envs,
                state_dim=1,
                time_constant=gimbal_time_constant,
                dt=dt,
                device=device
            )
            
            # Gimbal pitch filter (scalar)
            self.gimbal_pitch_filters[agent_id] = FirstOrderLag(
                num_envs=num_envs,
                state_dim=1,
                time_constant=gimbal_time_constant,
                dt=dt,
                device=device
            )
        
        # Detection latency buffer for each agent
        self.detection_buffers = {}
        self.detection_mean_latency = detection_mean_latency
        self.detection_std_latency = detection_std_latency
        
        # Communication manager
        self.comm_manager = CommManager(
            num_envs=num_envs,
            num_agents=num_agents,
            mean_comm_delay=comm_mean_delay,
            std_comm_delay=comm_std_delay,
            dropout_rate=comm_dropout_rate,
            dt=dt,
            device=device
        )
        
        # Current simulation time
        self.current_time = torch.zeros(num_envs, device=device)
    
    def update_time(self, time_increment: Optional[float] = None):
        """Update simulation time."""
        if time_increment is None:
            time_increment = self.dt
        self.current_time += time_increment
    
    def update_ego_motion(
        self,
        agent_id: int,
        true_position: torch.Tensor,
        true_orientation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update ego agent's motion with first-order lag.
        
        Args:
            agent_id: Agent identifier
            true_position: True position [N, 3]
            true_orientation: True orientation quaternion [N, 4] (w, x, y, z)
            
        Returns:
            filtered_position: [N, 3]
            filtered_orientation: [N, 4]
        """
        filtered_pos = self.position_filters[agent_id].update(true_position, self.current_time)
        filtered_quat = self.orientation_filters[agent_id].update(true_orientation, self.current_time)
        
        return filtered_pos, filtered_quat
    
    def update_ego_gimbal(
        self,
        agent_id: int,
        true_yaw: torch.Tensor,
        true_pitch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update ego agent's gimbal angles with first-order lag.
        
        Args:
            agent_id: Agent identifier
            true_yaw: True gimbal yaw [N] or [N, 1]
            true_pitch: True gimbal pitch [N] or [N, 1]
            
        Returns:
            filtered_yaw: [N, 1]
            filtered_pitch: [N, 1]
        """
        # Ensure 2D shape
        if true_yaw.dim() == 1:
            true_yaw = true_yaw.unsqueeze(-1)
        if true_pitch.dim() == 1:
            true_pitch = true_pitch.unsqueeze(-1)
        
        filtered_yaw = self.gimbal_yaw_filters[agent_id].update(true_yaw, self.current_time)
        filtered_pitch = self.gimbal_pitch_filters[agent_id].update(true_pitch, self.current_time)
        
        return filtered_yaw, filtered_pitch
    
    def add_detection(
        self,
        agent_id: int,
        detection_data: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ):
        """
        Add detection data with random latency.
        
        Args:
            agent_id: Agent identifier
            detection_data: Detection data [N, ...]
            valid_mask: Validity mask [N]
        """
        # Create buffer if doesn't exist
        if agent_id not in self.detection_buffers:
            self.detection_buffers[agent_id] = LatencyBuffer(
                num_envs=self.num_envs,
                data_shape=detection_data.shape[1:],
                mean_latency=self.detection_mean_latency,
                std_latency=self.detection_std_latency,
                dt=self.dt,
                device=self.device
            )
        
        self.detection_buffers[agent_id].add_measurement(
            detection_data,
            self.current_time,
            valid_mask
        )
    
    def get_delayed_detection(self, agent_id: int) -> TimestampedData:
        """
        Get delayed detection data for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            TimestampedData with delayed detection
        """
        if agent_id not in self.detection_buffers:
            # Return empty data if no buffer exists
            return TimestampedData(
                data=torch.zeros(self.num_envs, 0, device=self.device),
                timestamp=torch.zeros(self.num_envs, device=self.device),
                valid=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
        
        return self.detection_buffers[agent_id].get_delayed_measurement(self.current_time)
    
    def broadcast_state(
        self,
        sender_id: int,
        state_dict: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ):
        """
        Broadcast agent state to all other agents with communication delays/dropouts.
        
        Args:
            sender_id: ID of broadcasting agent
            state_dict: State data dictionary {key: [N, ...]}
            valid_mask: Validity mask [N]
        """
        self.comm_manager.send_message(
            sender_id,
            state_dict,
            self.current_time,
            valid_mask
        )
    
    def receive_other_agent_states(
        self,
        receiver_id: int,
        state_template: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[int, Dict[str, TimestampedData]]:
        """
        Receive states from all other agents.
        
        Args:
            receiver_id: ID of receiving agent
            state_template: Template for expected data structure
            
        Returns:
            Dictionary mapping sender_id to state dictionary
        """
        return self.comm_manager.receive_messages(
            receiver_id,
            self.current_time,
            state_template
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about delays and communication."""
        comm_stats = self.comm_manager.get_comm_statistics()
        
        stats = {
            'current_time': self.current_time.clone(),
            'comm_buffer_sizes': comm_stats['buffer_sizes'],
            'comm_last_received': comm_stats['last_received_time'],
        }
        
        # Add detection buffer sizes
        detection_buffer_sizes = {}
        for agent_id, buffer in self.detection_buffers.items():
            sizes = torch.tensor([len(buffer.buffer[i]) for i in range(self.num_envs)],
                                device=self.device, dtype=torch.long)
            detection_buffer_sizes[agent_id] = sizes
        
        stats['detection_buffer_sizes'] = detection_buffer_sizes
        
        return stats
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset all delay systems for specific environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset motion filters
        for agent_id in range(self.num_agents):
            self.position_filters[agent_id].reset(env_ids)
            self.orientation_filters[agent_id].reset(env_ids)
            self.gimbal_yaw_filters[agent_id].reset(env_ids)
            self.gimbal_pitch_filters[agent_id].reset(env_ids)
        
        # Reset detection buffers
        for buffer in self.detection_buffers.values():
            buffer.reset(env_ids)
        
        # Reset communication manager
        self.comm_manager.reset(env_ids)
        
        # Reset time for these environments
        self.current_time[env_ids] = 0.0
