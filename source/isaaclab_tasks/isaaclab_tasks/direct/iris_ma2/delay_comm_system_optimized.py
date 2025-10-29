# delay_comm_system_optimized.py
"""
Delay and communication system using vectorized tensor operations.
"""

import torch
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class TimestampedData:
    """Container for data with timestamp information."""
    data: torch.Tensor
    timestamp: torch.Tensor  # [N] - timestamp for each environment
    valid: torch.Tensor  # [N] - validity mask
    
    def clone(self):
        return TimestampedData(
            data=self.data.clone(),
            timestamp=self.timestamp.clone(),
            valid=self.valid.clone()
        )


# FirstOrderLag and QuaternionFirstOrderLag remain unchanged (already optimized)
class FirstOrderLag:
    """First-order lag filter (already optimized)."""
    
    def __init__(self, num_envs: int, state_dim: int, time_constant: float,
                 dt: float, device: torch.device, initial_state: Optional[torch.Tensor] = None):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.tau = time_constant
        self.dt = dt
        self.device = device
        self.alpha = dt / (dt + time_constant)
        
        self.filtered_state = initial_state.clone() if initial_state is not None else \
                              torch.zeros(num_envs, state_dim, device=device)
        self.timestamp = torch.zeros(num_envs, device=device)
    
    def update_time_constants(self, time_constant: float):
        self.tau = time_constant
        self.alpha = self.dt / (self.dt + time_constant)

    def update(self, measured_state: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        self.filtered_state = self.filtered_state + self.alpha * (measured_state - self.filtered_state)
        if isinstance(current_time, torch.Tensor):
            self.timestamp = current_time.clone()
        else:
            self.timestamp.fill_(current_time)
        return self.filtered_state.clone()
    
    def reset(self, env_ids: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if state is not None:
            self.filtered_state[env_ids] = state
        else:
            self.filtered_state[env_ids] = 0.0
        self.timestamp[env_ids] = 0.0


class QuaternionFirstOrderLag:
    """Quaternion SLERP filter (already optimized)."""
    
    def __init__(self, num_envs: int, time_constant: float, dt: float,
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

    def update_time_constants(self, time_constant: float):
        self.tau = time_constant
        self.alpha = self.dt / (self.dt + time_constant)

    def update(self, measured_quat: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        self.filtered_quat = self._slerp(self.filtered_quat, measured_quat, self.alpha)
        if isinstance(current_time, torch.Tensor):
            self.timestamp = current_time.clone()
        else:
            self.timestamp.fill_(current_time)
        return self.filtered_quat.clone()
    
    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
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


class LatencyBufferOptimized:
    """
    OPTIMIZED: Fixed-size tensor circular buffer.
    """
    
    def __init__(self, num_envs: int, data_shape: Tuple[int, ...], mean_latency: float,
                 std_latency: float, dt: float, device: torch.device, max_buffer_size: int = 100):
        self.num_envs = num_envs
        self.data_shape = data_shape
        self.mean_latency = mean_latency
        self.std_latency = std_latency
        self.dt = dt
        self.device = device
        self.max_buffer_size = max_buffer_size
        
        # Circular buffer tensors [N, max_buffer_size, *data_shape]
        self.buffer_data = torch.zeros(num_envs, max_buffer_size, *data_shape, device=device)
        self.buffer_delivery_time = torch.full((num_envs, max_buffer_size), float('inf'), device=device)
        self.buffer_gen_time = torch.zeros(num_envs, max_buffer_size, device=device)
        self.write_ptr = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.buffer_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    def update_latency_params(self, mean_latency: float, std_latency: float):
        self.mean_latency = mean_latency
        self.std_latency = std_latency

    def add_measurement(self, data: torch.Tensor, current_time: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None,
                       shared_latencies: Optional[torch.Tensor] = None):
        if valid_mask is None:
            valid_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Use shared latencies if provided, otherwise generate random latencies [N]
        if shared_latencies is not None:
            latencies = shared_latencies
        else:
            latencies = torch.normal(
                mean=self.mean_latency, std=self.std_latency,
                size=(self.num_envs,), device=self.device
            ).clamp(min=0.0)
        
        # Delivery times [N]
        if isinstance(current_time, torch.Tensor):
            delivery_times, gen_times = current_time + latencies, current_time
        else:
            current_time_tensor = torch.full((self.num_envs,), current_time, device=self.device)
            delivery_times, gen_times = current_time_tensor + latencies, current_time_tensor
        
        # Vectorized write
        env_indices = torch.arange(self.num_envs, device=self.device)
        write_positions = self.write_ptr
        
        self.buffer_data[env_indices, write_positions] = torch.where(
            valid_mask.view(-1, *([1]*len(self.data_shape))), data,
            self.buffer_data[env_indices, write_positions]
        )
        self.buffer_delivery_time[env_indices, write_positions] = torch.where(
            valid_mask, delivery_times, self.buffer_delivery_time[env_indices, write_positions]
        )
        self.buffer_gen_time[env_indices, write_positions] = torch.where(
            valid_mask, gen_times, self.buffer_gen_time[env_indices, write_positions]
        )
        
        # Update pointers
        self.write_ptr = torch.where(valid_mask, (self.write_ptr + 1) % self.max_buffer_size, self.write_ptr)
        self.buffer_count = torch.where(
            valid_mask, torch.clamp(self.buffer_count + 1, max=self.max_buffer_size), self.buffer_count
        )
    
    def get_delayed_measurement(self, current_time: torch.Tensor) -> TimestampedData:
        current_t = current_time if isinstance(current_time, torch.Tensor) else \
                    torch.full((self.num_envs,), current_time, device=self.device)
        
        # Find ready messages [N, max_buffer_size]
        ready_mask = self.buffer_delivery_time <= current_t.unsqueeze(1)
        
        # Most recent ready message per env
        gen_times_masked = torch.where(ready_mask, self.buffer_gen_time,
                                       torch.full_like(self.buffer_gen_time, -float('inf')))
        most_recent_idx = torch.argmax(gen_times_masked, dim=1)  # [N]
        has_ready = ready_mask.any(dim=1)  # [N]
        
        # Extract data
        env_indices = torch.arange(self.num_envs, device=self.device)
        result_data = self.buffer_data[env_indices, most_recent_idx]
        result_timestamp = self.buffer_gen_time[env_indices, most_recent_idx]
        
        # Clean up delivered messages
        self.buffer_delivery_time = torch.where(
            ready_mask, torch.full_like(self.buffer_delivery_time, float('inf')), self.buffer_delivery_time
        )
        delivered_count = ready_mask.sum(dim=1)
        self.buffer_count = torch.clamp(self.buffer_count - delivered_count, min=0)
        
        return TimestampedData(data=result_data, timestamp=result_timestamp, valid=has_ready)
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.buffer_delivery_time[env_ids] = float('inf')
        self.write_ptr[env_ids] = 0
        self.buffer_count[env_ids] = 0


class CommManagerOptimized:
    """
    OPTIMIZED: Tensor-based communication with vectorized ops.
    """
    
    def __init__(self, num_envs: int, num_agents: int, mean_comm_delay: float,
                 std_comm_delay: float, dropout_rate: float, dt: float,
                 device: torch.device, max_buffer_size: int = 100):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.mean_comm_delay = mean_comm_delay
        self.std_comm_delay = std_comm_delay
        self.dropout_rate = dropout_rate
        self.dt = dt
        self.device = device
        self.max_buffer_size = max_buffer_size
        
        # Buffers per link [N, num_agents, num_agents, max_buffer_size, ...]
        self.buffers = {}  # Per data key
        self.buffer_delivery_time = torch.full(
            (num_envs, num_agents, num_agents, max_buffer_size), float('inf'), device=device
        )
        self.buffer_gen_time = torch.zeros(num_envs, num_agents, num_agents, max_buffer_size, device=device)
        self.write_ptr = torch.zeros(num_envs, num_agents, num_agents, dtype=torch.long, device=device)
        self.buffer_count = torch.zeros(num_envs, num_agents, num_agents, dtype=torch.long, device=device)
        self.last_received_time = torch.zeros(num_envs, num_agents, num_agents, device=device)
    
    def update_comm_params(self, mean_comm_delay: float, std_comm_delay: float, dropout_rate: float):
        self.mean_comm_delay = mean_comm_delay
        self.std_comm_delay = std_comm_delay
        self.dropout_rate = dropout_rate

    def send_message(self, sender_id: int, data: Dict[str, torch.Tensor], current_time: torch.Tensor,
                    valid_mask: Optional[torch.Tensor] = None):
        """OPTIMIZED: Vectorized send."""
        if valid_mask is None:
            valid_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        gen_times = current_time if isinstance(current_time, torch.Tensor) else \
                   torch.full((self.num_envs,), current_time, device=self.device)
        
        # Loop over receivers (unavoidable, but inner ops vectorized)
        for receiver_id in range(self.num_agents):
            if receiver_id == sender_id:
                continue
            
            # Generate delays and dropouts [N] - INDEPENDENT per receiver
            comm_delays = torch.normal(
                mean=self.mean_comm_delay, std=self.std_comm_delay,
                size=(self.num_envs,), device=self.device
            ).clamp(min=0.0)
            dropout_mask = torch.rand(self.num_envs, device=self.device) < self.dropout_rate
            
            delivery_times = gen_times + comm_delays
            send_mask = valid_mask & ~dropout_mask
            
            write_pos = self.write_ptr[:, sender_id, receiver_id]
            env_idx = torch.arange(self.num_envs, device=self.device)
            
            # Store data (create buffers on first use)
            for key, tensor in data.items():
                if key not in self.buffers:
                    data_shape = tensor.shape[1:]
                    self.buffers[key] = torch.zeros(
                        self.num_envs, self.num_agents, self.num_agents,
                        self.max_buffer_size, *data_shape, device=self.device
                    )
                
                # Vectorized write
                self.buffers[key][env_idx, sender_id, receiver_id, write_pos] = torch.where(
                    send_mask.view(-1, *([1]*(len(tensor.shape)-1))), tensor,
                    self.buffers[key][env_idx, sender_id, receiver_id, write_pos]
                )
            
            # Write metadata
            self.buffer_delivery_time[env_idx, sender_id, receiver_id, write_pos] = torch.where(
                send_mask, delivery_times, self.buffer_delivery_time[env_idx, sender_id, receiver_id, write_pos]
            )
            self.buffer_gen_time[env_idx, sender_id, receiver_id, write_pos] = torch.where(
                send_mask, gen_times, self.buffer_gen_time[env_idx, sender_id, receiver_id, write_pos]
            )
            
            # Update pointers
            self.write_ptr[:, sender_id, receiver_id] = torch.where(
                send_mask, (write_pos + 1) % self.max_buffer_size, write_pos
            )
            self.buffer_count[:, sender_id, receiver_id] = torch.where(
                send_mask, torch.clamp(self.buffer_count[:, sender_id, receiver_id] + 1,
                                      max=self.max_buffer_size),
                self.buffer_count[:, sender_id, receiver_id]
            )
    
    def receive_messages(self, receiver_id: int, current_time: torch.Tensor,
                        data_template: Optional[Dict[str, torch.Tensor]] = None
                        ) -> Dict[int, Dict[str, TimestampedData]]:
        """OPTIMIZED: Vectorized receive."""
        current_t = current_time if isinstance(current_time, torch.Tensor) else \
                   torch.full((self.num_envs,), current_time, device=self.device)
        
        received_data = {}
        
        for sender_id in range(self.num_agents):
            if sender_id == receiver_id:
                continue
            
            # Get buffers for this link
            delivery_times = self.buffer_delivery_time[:, sender_id, receiver_id]  # [N, max_buffer_size]
            gen_times = self.buffer_gen_time[:, sender_id, receiver_id]
            
            # Find ready messages
            ready_mask = delivery_times <= current_t.unsqueeze(1)
            gen_times_masked = torch.where(ready_mask, gen_times, torch.full_like(gen_times, -float('inf')))
            most_recent_idx = torch.argmax(gen_times_masked, dim=1)  # [N]
            has_ready = ready_mask.any(dim=1)
            
            # Extract data
            env_idx = torch.arange(self.num_envs, device=self.device)
            result_dict = {}
            for key in self.buffers.keys():
                result_data = self.buffers[key][env_idx, sender_id, receiver_id, most_recent_idx]
                result_timestamp = gen_times[env_idx, most_recent_idx]
                result_dict[key] = TimestampedData(
                    data=result_data, timestamp=result_timestamp, valid=has_ready
                )
            
            if result_dict:
                received_data[sender_id] = result_dict
            
            # Clean up
            self.buffer_delivery_time[:, sender_id, receiver_id] = torch.where(
                ready_mask, torch.full_like(delivery_times, float('inf')), delivery_times
            )
            delivered_count = ready_mask.sum(dim=1)
            self.buffer_count[:, sender_id, receiver_id] = torch.clamp(
                self.buffer_count[:, sender_id, receiver_id] - delivered_count, min=0
            )
            self.last_received_time[:, sender_id, receiver_id] = torch.where(
                has_ready, current_t, self.last_received_time[:, sender_id, receiver_id]
            )
        
        return received_data
    
    def get_comm_statistics(self) -> Dict[str, torch.Tensor]:
        return {
            'buffer_sizes': self.buffer_count.clone(),
            'last_received_time': self.last_received_time.clone()
        }
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.buffer_delivery_time[env_ids] = float('inf')
        self.write_ptr[env_ids] = 0
        self.buffer_count[env_ids] = 0
        self.last_received_time[env_ids] = 0.0


class DelayedObservationManager:
    """High-level manager using optimized components."""
    
    def __init__(self, num_envs: int, num_agents: int, dt: float, device: torch.device,
                 motion_time_constant: float = 0.1, gimbal_time_constant: float = 0.03,
                 detection_mean_latency: float = 0.05, detection_std_latency: float = 0.02,
                 detection_fps: float = 30.0,
                 comm_mean_delay: float = 0.1, comm_std_delay: float = 0.03,
                 comm_dropout_rate: float = 0.05, max_buffer_size: int = 100):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.dt = dt
        self.device = device
        self.max_buffer_size = max_buffer_size
        
        self.detection_fps = detection_fps
        self.detection_period = 1.0 / detection_fps if detection_fps > 0 else 0.0
        self.last_detection_time = {}  # Dict[agent_id -> Tensor[N]]
        
        # Create filters
        self.position_filters = {}
        self.orientation_filters = {}
        self.linear_velocity_filters = {}
        self.angular_velocity_filters = {}
        self.gimbal_yaw_filters = {}
        self.gimbal_pitch_filters = {}
        
        for agent_id in range(num_agents):
            self.position_filters[agent_id] = FirstOrderLag(
                num_envs, 3, motion_time_constant, dt, device
            )
            self.orientation_filters[agent_id] = QuaternionFirstOrderLag(
                num_envs, motion_time_constant/10.0, dt, device
            )
            self.linear_velocity_filters[agent_id] = FirstOrderLag(
                num_envs, 3, motion_time_constant/5.0, dt, device
            )
            self.angular_velocity_filters[agent_id] = FirstOrderLag(
                num_envs, 3, motion_time_constant/100.0, dt, device
            )
            self.gimbal_yaw_filters[agent_id] = FirstOrderLag(
                num_envs, 1, gimbal_time_constant, dt, device
            )
            self.gimbal_pitch_filters[agent_id] = FirstOrderLag(
                num_envs, 1, gimbal_time_constant, dt, device
            )
        
        # buffers
        self.detection_buffers = {}
        self.detection_valid_mask_buffers = {}
        self.detection_mean_latency = detection_mean_latency
        self.detection_std_latency = detection_std_latency
        
        # communication
        self.comm_manager = CommManagerOptimized(
            num_envs, num_agents, comm_mean_delay, comm_std_delay,
            comm_dropout_rate, dt, device, self.max_buffer_size
        )
        
        self.current_time = torch.zeros(num_envs, device=device)
    
    def update_time(self, time_increment: Optional[float] = None):
        if time_increment is None:
            time_increment = self.dt
        self.current_time += time_increment
    
    def update_ego_motion(self, agent_id: int, true_position: torch.Tensor,
                         true_orientation: torch.Tensor, true_linear_velocity: torch.Tensor,
                         true_angular_velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filtered_pos = self.position_filters[agent_id].update(true_position, self.current_time)
        filtered_quat = self.orientation_filters[agent_id].update(true_orientation, self.current_time)
        filtered_lin_vel = self.linear_velocity_filters[agent_id].update(true_linear_velocity, self.current_time)
        filtered_ang_vel = self.angular_velocity_filters[agent_id].update(true_angular_velocity, self.current_time)
        return filtered_pos, filtered_quat, filtered_lin_vel, filtered_ang_vel
    
    def update_ego_gimbal(self, agent_id: int, true_yaw: torch.Tensor,
                         true_pitch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if true_yaw.dim() == 1:
            true_yaw = true_yaw.unsqueeze(-1)
        if true_pitch.dim() == 1:
            true_pitch = true_pitch.unsqueeze(-1)
        
        filtered_yaw = self.gimbal_yaw_filters[agent_id].update(true_yaw, self.current_time)
        filtered_pitch = self.gimbal_pitch_filters[agent_id].update(true_pitch, self.current_time)
        return filtered_yaw, filtered_pitch
    
    def update_motion_time_constants(self, motion_time_constant: float):
        for agent_id in range(self.num_agents):
            self.position_filters[agent_id].update_time_constants(motion_time_constant)
            self.orientation_filters[agent_id].update_time_constants(motion_time_constant / 10.0)
            self.linear_velocity_filters[agent_id].update_time_constants(motion_time_constant / 5.0)
            self.angular_velocity_filters[agent_id].update_time_constants(motion_time_constant / 100.0)

    def update_gimbal_time_constants(self, gimbal_time_constant: float):
        for agent_id in range(self.num_agents):
            self.gimbal_yaw_filters[agent_id].update_time_constants(gimbal_time_constant)
            self.gimbal_pitch_filters[agent_id].update_time_constants(gimbal_time_constant)

    def update_detection_fps(self, detection_fps: float):
        self.detection_fps = detection_fps
        self.detection_period = 1.0 / detection_fps if detection_fps > 0 else 0.0

    def update_detection_latency(self, mean_latency: float, std_latency: float):
        self.detection_mean_latency = mean_latency
        self.detection_std_latency = std_latency
        for buffer in self.detection_buffers.values():
            buffer.update_latency_params(mean_latency, std_latency)

    def update_comm_parameters(self, mean_comm_delay: float, std_comm_delay: float, dropout_rate: float):
        self.comm_manager.update_comm_params(mean_comm_delay, std_comm_delay, dropout_rate)

    def add_detection(self, agent_id: int, detection_data: torch.Tensor,
                     valid_mask: torch.Tensor):
        # Initialize last detection time for this agent if needed
        if agent_id not in self.last_detection_time:
            self.last_detection_time[agent_id] = torch.full(
                (self.num_envs,), -float('inf'), device=self.device
            )
        
        # Check if enough time has passed since last detection (FPS throttling)
        # fps_mask determines which frames to capture, independent of detection validity
        time_since_last = self.current_time - self.last_detection_time[agent_id]
        fps_mask = time_since_last >= self.detection_period  # [N]
        
        if agent_id not in self.detection_buffers:
            self.detection_buffers[agent_id] = LatencyBufferOptimized(
                self.num_envs, detection_data.shape[1:], self.detection_mean_latency,
                self.detection_std_latency, self.dt, self.device, self.max_buffer_size
            )
        if agent_id not in self.detection_valid_mask_buffers:
            self.detection_valid_mask_buffers[agent_id] = LatencyBufferOptimized(
                self.num_envs, (1,), self.detection_mean_latency,
                self.detection_std_latency, self.dt, self.device, self.max_buffer_size
            )

        # Generate shared latencies for both bbox and valid_mask
        # This ensures they arrive at the same time (synchronized)
        shared_latencies = torch.normal(
            mean=self.detection_mean_latency,
            std=self.detection_std_latency,
            size=(self.num_envs,),
            device=self.device
        ).clamp(min=0.0)

        # Add detection data for all frames that pass FPS throttling
        # Use shared latencies to ensure bbox and valid_mask arrive together
        self.detection_buffers[agent_id].add_measurement(
            detection_data, self.current_time, fps_mask, shared_latencies
        )

        # Store detection validity for the captured frames
        # This tracks: "for frames we captured, was the bbox within image bounds?"
        # Use SAME latencies to ensure synchronization
        self.detection_valid_mask_buffers[agent_id].add_measurement(
            valid_mask.unsqueeze(-1), self.current_time, fps_mask, shared_latencies
        )

        # Update last detection time for environments that passed FPS throttling
        self.last_detection_time[agent_id] = torch.where(
            fps_mask,
            self.current_time,
            self.last_detection_time[agent_id]
        )

    def get_delayed_detection(self, agent_id: int) -> Tuple[TimestampedData, torch.Tensor]:
        if agent_id not in self.detection_buffers:
            detection = TimestampedData(
                data=torch.zeros(self.num_envs, 4, device=self.device),
                timestamp=torch.zeros(self.num_envs, device=self.device),
                valid=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
        else:
            detection = self.detection_buffers[agent_id].get_delayed_measurement(self.current_time)
        
        if agent_id not in self.detection_valid_mask_buffers:
            valid_mask = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        else:
            valid_mask = self.detection_valid_mask_buffers[agent_id].get_delayed_measurement(self.current_time).data
        
        return detection, valid_mask
    
    def broadcast_state(self, sender_id: int, state_dict: Dict[str, torch.Tensor],
                       valid_mask: Optional[torch.Tensor] = None):
        self.comm_manager.send_message(sender_id, state_dict, self.current_time, valid_mask)
    
    def receive_other_agent_states(self, receiver_id: int,
                                   state_template: Optional[Dict[str, torch.Tensor]] = None
                                   ) -> Dict[int, Dict[str, TimestampedData]]:
        return self.comm_manager.receive_messages(receiver_id, self.current_time, state_template)
    
    def get_statistics(self) -> Dict[str, Any]:
        comm_stats = self.comm_manager.get_comm_statistics()
        stats = {
            'current_time': self.current_time.clone(),
            'comm_buffer_sizes': comm_stats['buffer_sizes'],
            'comm_last_received': comm_stats['last_received_time'],
        }
        detection_buffer_sizes = {
            agent_id: buffer.buffer_count.clone()
            for agent_id, buffer in self.detection_buffers.items()
        }
        stats['detection_buffer_sizes'] = detection_buffer_sizes
        return stats
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        for agent_id in range(self.num_agents):
            self.position_filters[agent_id].reset(env_ids)
            self.orientation_filters[agent_id].reset(env_ids)
            self.linear_velocity_filters[agent_id].reset(env_ids)
            self.angular_velocity_filters[agent_id].reset(env_ids)
            self.gimbal_yaw_filters[agent_id].reset(env_ids)
            self.gimbal_pitch_filters[agent_id].reset(env_ids)
        
        for buffer in self.detection_buffers.values():
            buffer.reset(env_ids)

        for buffer in self.detection_valid_mask_buffers.values():
            buffer.reset(env_ids)

        for agent_id in self.last_detection_time.keys():
            self.last_detection_time[agent_id][env_ids] = -float('inf')
        
        self.comm_manager.reset(env_ids)
        self.current_time[env_ids] = 0.0