# delayed_states.py API Documentation

**Version:** 1.0
**Last Updated:** 2025-11-12

## Overview

`delayed_states.py` provides a comprehensive framework for modeling realistic sensor delays, communication impairments, and state filtering in multi-agent robotics simulations. The module is designed for GPU-accelerated parallel environments and supports:

- **Channel Impairments**: Throttling (FPS limiting), dropout (packet loss), and variable latency
- **Multi-Agent Communication**: Full-mesh topology with per-link independent impairments
- **Motion Filtering**: First-order lag filters and quaternion SLERP for state estimation
- **Data Validity Tracking**: Semantic separation of source availability and channel success
- **Data Age Tracking**: Time elapsed since data capture for fusion algorithms

---

## Table of Contents

1. [Core Components](#core-components)
   - [DelayBufferCustom](#delaybuffercustom)
   - [ChannelBuffer](#channelbuffer)
   - [MultiAgentCommChannel](#multiagentcommchannel)
   - [FirstOrderLag](#firstorderlag)
   - [QuaternionFirstOrderLag](#quaternionfirstorderlag)
   - [MultiAgentObservationPipeline](#multiagentobservationpipeline)
2. [Usage Examples](#usage-examples)
3. [Key Concepts](#key-concepts)
4. [Performance Considerations](#performance-considerations)

---

## Core Components

### DelayBufferCustom

**Extends:** `isaaclab.utils.buffers.delay_buffer.DelayBuffer`

Custom delay buffer with separated append/retrieve operations to prevent buffer corruption.

#### Constructor

```python
DelayBufferCustom(
    history_length: int,
    batch_size: int,
    device: str
)
```

**Parameters:**
- `history_length` (int): Maximum number of timesteps to buffer
- `batch_size` (int): Number of parallel environments
- `device` (str): Device string (e.g., "cuda:0", "cpu")

#### Methods

##### `append(data: torch.Tensor) -> None`

Append new data to the buffer without retrieving.

**Parameters:**
- `data` (torch.Tensor): Input data of shape `[N, ...]`

**Returns:** None

**Example:**
```python
buffer = DelayBufferCustom(history_length=100, batch_size=4, device="cuda")
data = torch.randn(4, 3, device="cuda")
buffer.append(data)
```

##### `get_delayed() -> Tuple[torch.Tensor, torch.Tensor]`

Retrieve delayed data without appending new data (read-only operation).

**Returns:**
- `delayed_data` (torch.Tensor): Delayed data of shape `[N, ...]`
- `sufficient_history_mask` (torch.Tensor): Boolean mask `[N]` indicating which environments have enough history

**Raises:**
- `RuntimeError`: If buffer is empty (no data appended yet)

**Example:**
```python
delayed_data, has_history = buffer.get_delayed()
if has_history[0]:
    print(f"Environment 0 data: {delayed_data[0]}")
```

---

### ChannelBuffer

Unified buffer with realistic channel impairments applied in physically realistic order:
**Throttle → Dropout → Latency**

#### Constructor

```python
ChannelBuffer(
    num_envs: int,
    data_shape: Tuple[int, ...],
    throttle_period: torch.Tensor | float,
    dropout_rate: torch.Tensor | float,
    mean_latency: torch.Tensor | float,
    std_latency: torch.Tensor | float,
    dt: float,
    device: torch.device,
    max_history: int = 100
)
```

**Parameters:**
- `num_envs` (int): Number of parallel environments
- `data_shape` (Tuple[int, ...]): Shape of data excluding batch dimension
- `throttle_period` (Tensor | float): Minimum period between updates in seconds `[N]` or scalar. Set to 0 to disable.
- `dropout_rate` (Tensor | float): Probability of dropping data (0.0 to 1.0) `[N]` or scalar
- `mean_latency` (Tensor | float): Mean latency in seconds `[N]` or scalar
- `std_latency` (Tensor | float): Std deviation of latency in seconds `[N]` or scalar
- `dt` (float): Simulation timestep in seconds
- `device` (torch.device): Device to allocate tensors on
- `max_history` (int, optional): Maximum history length for latency buffer. Default: 100

**Example:**
```python
buffer = ChannelBuffer(
    num_envs=4,
    data_shape=(3,),  # 3D position
    throttle_period=0.1,  # 10 Hz max rate
    dropout_rate=0.05,    # 5% packet loss
    mean_latency=0.05,    # 50ms mean delay
    std_latency=0.02,     # 20ms std
    dt=0.01,              # 100 Hz simulation
    device=torch.device("cuda")
)
```

#### Methods

##### `compute(data, current_time, has_source_data=None) -> Tuple[torch.Tensor, torch.Tensor]`

Process data through channel with impairments.

**Parameters:**
- `data` (torch.Tensor): New data `[N, *data_shape]`
- `current_time` (Tensor | float): Current simulation time `[N]` or scalar
- `has_source_data` (Optional[torch.Tensor]): Boolean mask `[N]` indicating whether each environment has meaningful source data available. True = sensor detected target, measurement is valid. Independent of channel impairments. If None, assumes all have data.

**Returns:**
- `delayed_data` (torch.Tensor): Output data with all impairments applied `[N, *data_shape]`
- `valid_mask` (torch.Tensor): Boolean mask `[N]` indicating successful transmission. True = has_source_data AND passed throttle AND not dropped.

**Example:**
```python
# Detection with per-environment validity
detection = torch.randn(4, 4, device=device)  # 4 envs, 4D bbox
has_detection = torch.tensor([True, False, True, True], device=device)

delayed_det, success_mask = buffer.compute(
    data=detection,
    current_time=current_time,
    has_source_data=has_detection
)

# success_mask[0] = True if env 0 had detection AND passed channel
# success_mask[1] = False (no source data)
```

##### `get_delayed(current_time=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Get delayed data without updating buffer state (read-only).

**Parameters:**
- `current_time` (Optional[Tensor | float]): Current simulation time `[N]` or scalar. If None, always returns data if available.

**Returns:**
- `delayed_data` (torch.Tensor): Delayed data `[N, *data_shape]`
- `data_available_mask` (torch.Tensor): Whether delayed data is available (valid data + time elapsed) `[N]`
- `data_age` (torch.Tensor): Time elapsed since data was captured in seconds `[N]`. Zero if no data or current_time is None.

**Example:**
```python
delayed_data, available, age = buffer.get_delayed(current_time)

# Use age for time-weighted fusion
weights = torch.exp(-age / 1.0)  # Exponential decay with 1s time constant
fused = (delayed_data * weights.unsqueeze(-1)).sum(0) / weights.sum()
```

##### `update_params(...) -> None`

Update channel parameters at runtime.

**Parameters:**
- `throttle_period` (Optional[torch.Tensor]): New throttle period `[N]`
- `dropout_rate` (Optional[torch.Tensor]): New dropout rate `[N]`
- `mean_latency` (Optional[torch.Tensor]): New mean latency `[N]`
- `std_latency` (Optional[torch.Tensor]): New std latency `[N]`

**Example:**
```python
# Increase latency for specific environments
new_latency = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device)
buffer.update_params(mean_latency=new_latency)
```

##### `get_statistics() -> Dict[str, torch.Tensor]`

Get channel statistics.

**Returns:**
Dictionary with:
- `total_attempts` (torch.Tensor): Total compute calls `[N]`
- `throttled_count` (torch.Tensor): Number throttled `[N]`
- `dropout_count` (torch.Tensor): Number dropped `[N]`
- `success_count` (torch.Tensor): Successfully transmitted `[N]`
- `throttle_rate` (torch.Tensor): Effective throttle rate `[N]`
- `dropout_rate_effective` (torch.Tensor): Effective dropout rate `[N]`
- `success_rate` (torch.Tensor): Overall success rate `[N]`

##### `reset(env_ids=None) -> None`

Reset buffer for specified environments.

**Parameters:**
- `env_ids` (Optional[torch.Tensor]): Indices of environments to reset. If None, resets all.

---

### MultiAgentCommChannel

Multi-agent communication channel with realistic impairments using ChannelBuffer for each sender-receiver link.

#### Constructor

```python
MultiAgentCommChannel(
    num_envs: int,
    num_agents: int,
    mean_latency: float,
    std_latency: float,
    throttle_period: float = 0.0,
    dropout_rate: float = 0.0,
    dt: float = 0.01,
    device: torch.device = torch.device('cpu'),
    max_history: int = 100
)
```

**Parameters:**
- `num_envs` (int): Number of parallel environments
- `num_agents` (int): Number of agents in the system
- `mean_latency` (float): Mean communication delay in seconds
- `std_latency` (float): Std deviation of delay in seconds
- `throttle_period` (float, optional): Minimum period between messages in seconds. Default: 0.0 (no throttling)
- `dropout_rate` (float, optional): Probability of packet loss (0.0 to 1.0). Default: 0.0
- `dt` (float, optional): Simulation timestep in seconds. Default: 0.01
- `device` (torch.device, optional): Device to allocate tensors on. Default: cpu
- `max_history` (int, optional): Maximum history length. Default: 100

**Example:**
```python
comm = MultiAgentCommChannel(
    num_envs=4,
    num_agents=3,
    mean_latency=0.1,   # 100ms mean delay
    std_latency=0.03,   # 30ms std
    throttle_period=0.05,  # 20 Hz max message rate
    dropout_rate=0.02,     # 2% packet loss
    dt=0.01,
    device=torch.device("cuda")
)
```

#### Methods

##### `send_message(sender_id, data, current_time, has_source_data=None) -> None`

Send message from sender to all receivers.

**Parameters:**
- `sender_id` (int): ID of sending agent
- `data` (Dict[str, torch.Tensor]): Dictionary of data tensors `{key: [N, ...]}`
- `current_time` (Tensor | float): Current simulation time
- `has_source_data` (Optional[torch.Tensor]): Boolean mask `[N]` indicating source data availability. If None, assumes all have data.

**Example:**
```python
# Agent 0 broadcasts position and velocity
sender_id = 0
state = {
    "position": torch.randn(4, 3, device=device),
    "velocity": torch.randn(4, 3, device=device)
}
has_data = torch.tensor([True, True, False, True], device=device)

comm.send_message(sender_id, state, current_time, has_data)
```

##### `receive_messages(receiver_id, current_time=None) -> Dict[int, Dict[str, Tuple]]`

Receive delayed messages from all senders.

**Parameters:**
- `receiver_id` (int): ID of receiving agent
- `current_time` (Optional[Tensor | float]): Current simulation time. If None, no time-based checking.

**Returns:**
Dictionary mapping `sender_id -> {data_key -> (data, valid_mask, data_age)}` where:
- `data` (torch.Tensor): Received data `[N, ...]`
- `valid_mask` (torch.Tensor): Boolean mask `[N]` indicating data availability
- `data_age` (torch.Tensor): Time elapsed since data was captured in seconds `[N]`

**Example:**
```python
# Agent 1 receives messages
receiver_id = 1
received = comm.receive_messages(receiver_id, current_time)

for sender_id, data_dict in received.items():
    if "position" in data_dict:
        pos, valid, age = data_dict["position"]
        # Use only valid data with age < 1 second
        fresh_mask = valid & (age < 1.0)
        fresh_positions = pos[fresh_mask]
```

##### `update_comm_params(...) -> None`

Update communication parameters for all links.

**Parameters:**
- `mean_latency` (Optional[torch.Tensor]): New mean latency `[N, num_agents]`
- `std_latency` (Optional[torch.Tensor]): New std latency `[N, num_agents]`
- `throttle_period` (Optional[torch.Tensor]): New throttle period `[N, num_agents]`
- `dropout_rate` (Optional[torch.Tensor]): New dropout rate `[N, num_agents]`

##### `get_statistics() -> Dict[str, Any]`

Get communication statistics across all links.

**Returns:**
Dictionary mapping link names to their statistics.

##### `reset(env_ids=None) -> None`

Reset all channel buffers.

---

### FirstOrderLag

First-order lag filter for scalar/vector states.

#### Constructor

```python
FirstOrderLag(
    num_envs: int,
    state_dim: int,
    time_constant: torch.Tensor | float,
    dt: float,
    device: torch.device | str,
    initial_state: Optional[torch.Tensor] = None
)
```

**Parameters:**
- `num_envs` (int): Number of environments
- `state_dim` (int): Dimension of the state
- `time_constant` (Tensor | float): Time constant for the filter `[N, 1]` or `[N, state_dim]` or scalar
- `dt` (float): Time step
- `device` (torch.device | str): Device to allocate tensors on
- `initial_state` (Optional[torch.Tensor]): Initial state of the filter `[N, state_dim]`

**Example:**
```python
# Position filter with 0.1s time constant
pos_filter = FirstOrderLag(
    num_envs=4,
    state_dim=3,
    time_constant=0.1,
    dt=0.01,
    device="cuda"
)
```

#### Methods

##### `update(measured_state, current_time) -> torch.Tensor`

Update filter with new measurement.

**Parameters:**
- `measured_state` (torch.Tensor): New measurement `[N, state_dim]`
- `current_time` (torch.Tensor): Current time `[N]` or scalar

**Returns:**
- `filtered_state` (torch.Tensor): Filtered state `[N, state_dim]`

**Example:**
```python
true_position = torch.randn(4, 3, device=device)
filtered_pos = pos_filter.update(true_position, current_time)
```

##### `update_time_constants(time_constant, env_ids=None) -> None`

Update time constants for specific environments.

##### `reset(env_ids=None, state=None) -> None`

Reset filter state for specified environments.

---

### QuaternionFirstOrderLag

Quaternion SLERP (Spherical Linear Interpolation) filter for orientation.

#### Constructor

```python
QuaternionFirstOrderLag(
    num_envs: int,
    time_constant: torch.Tensor,
    dt: float,
    device: torch.device,
    initial_quat: Optional[torch.Tensor] = None
)
```

**Parameters:**
- `num_envs` (int): Number of environments
- `time_constant` (torch.Tensor): Time constant `[N]`
- `dt` (float): Time step
- `device` (torch.device): Device
- `initial_quat` (Optional[torch.Tensor]): Initial quaternion `[N, 4]` (w, x, y, z)

**Example:**
```python
quat_filter = QuaternionFirstOrderLag(
    num_envs=4,
    time_constant=torch.full((4,), 0.05, device=device),
    dt=0.01,
    device=device
)
```

#### Methods

##### `update(measured_quat, current_time) -> torch.Tensor`

Update quaternion filter with SLERP.

**Parameters:**
- `measured_quat` (torch.Tensor): New quaternion `[N, 4]` (w, x, y, z)
- `current_time` (torch.Tensor): Current time

**Returns:**
- `filtered_quat` (torch.Tensor): Filtered quaternion `[N, 4]`

**Note:** Automatically handles antipodal quaternions (shortest path).

---

### MultiAgentObservationPipeline

Complete multi-agent observation pipeline with motion filtering, detection system, and communication.

#### Constructor

```python
MultiAgentObservationPipeline(
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
)
```

**Parameters:**
- `num_envs` (int): Number of parallel environments
- `num_agents` (int): Number of agents
- `dt` (float): Simulation timestep in seconds
- `device` (torch.device): Device
- `motion_time_constant` (float, optional): Motion filter time constant. Default: 0.1
- `gimbal_time_constant` (float, optional): Gimbal filter time constant. Default: 0.03
- `detection_fps` (float, optional): Detection update frequency in Hz. Default: 30.0
- `detection_mean_latency` (float, optional): Mean detection delay. Default: 0.05
- `detection_std_latency` (float, optional): Std of detection delay. Default: 0.02
- `detection_failure_rate` (float, optional): Detection dropout rate. Default: 0.0
- `comm_mean_delay` (float, optional): Mean communication delay. Default: 0.1
- `comm_std_delay` (float, optional): Std of communication delay. Default: 0.03
- `comm_throttle_period` (float, optional): Communication bandwidth limit. Default: 0.0
- `comm_dropout_rate` (float, optional): Communication packet loss. Default: 0.05
- `max_buffer_size` (int, optional): Maximum buffer size. Default: 100

**Example:**
```python
pipeline = MultiAgentObservationPipeline(
    num_envs=4,
    num_agents=3,
    dt=0.01,
    device=torch.device("cuda"),
    detection_fps=30.0,
    comm_dropout_rate=0.02
)
```

#### Methods

##### `update_time(time_increment=None) -> None`

Update simulation time.

**Parameters:**
- `time_increment` (Optional[float]): Time increment in seconds. If None, uses dt.

##### `update_ego_motion(agent_id, true_position, true_orientation, true_linear_velocity, true_angular_velocity)`

Filter ego motion states.

**Returns:**
Tuple of `(filtered_pos, filtered_quat, filtered_lin_vel, filtered_ang_vel)`

##### `update_ego_gimbal(agent_id, true_yaw, true_pitch)`

Filter gimbal angles.

**Returns:**
Tuple of `(filtered_yaw, filtered_pitch)`

##### `add_detection(agent_id, detection_data, has_detection) -> None`

Add detection data with FPS throttling, dropout, and latency.

**Parameters:**
- `agent_id` (int): Agent ID
- `detection_data` (torch.Tensor): Detection data `[N, ...]`
- `has_detection` (torch.Tensor): Boolean mask `[N]` indicating sensor has valid detection

##### `get_delayed_detection(agent_id) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Get delayed detection with validity mask and data age.

**Returns:**
- `detection_data` (torch.Tensor): Detection data `[N, 4]`
- `valid_mask` (torch.Tensor): Boolean mask `[N]`
- `data_age` (torch.Tensor): Time elapsed since detection `[N]`

##### `broadcast_state(sender_id, state_dict, has_source_data=None) -> None`

Broadcast state to all other agents.

**Parameters:**
- `sender_id` (int): Sending agent ID
- `state_dict` (Dict[str, torch.Tensor]): State data `{key: [N, ...]}`
- `has_source_data` (Optional[torch.Tensor]): Source data availability mask `[N]`

##### `receive_other_agent_states(receiver_id) -> Dict[int, Dict[str, Tuple]]`

Receive states from all other agents.

**Returns:**
Dict mapping `sender_id -> {data_key -> (data, valid_mask, data_age)}`

##### `update_detection_params(...) -> None`

Update detection parameters at runtime.

##### `update_comm_params(...) -> None`

Update communication parameters at runtime.

##### `get_statistics() -> Dict[str, Any]`

Get all pipeline statistics.

##### `reset(env_ids=None) -> None`

Reset all components.

---

## Usage Examples

### Example 1: Basic Channel Buffer

```python
import torch
from delayed_states import ChannelBuffer

# Create channel buffer
buffer = ChannelBuffer(
    num_envs=4,
    data_shape=(3,),
    throttle_period=0.1,  # 10 Hz
    dropout_rate=0.05,
    mean_latency=0.05,
    std_latency=0.02,
    dt=0.01,
    device=torch.device("cuda")
)

# Simulate loop
for t in range(1000):
    current_time = t * 0.01

    # New measurement
    measurement = torch.randn(4, 3, device="cuda")
    has_measurement = torch.tensor([True, True, False, True], device="cuda")

    # Process through channel
    delayed_data, success_mask = buffer.compute(
        measurement, current_time, has_measurement
    )

    # Get delayed data (read-only)
    data, available, age = buffer.get_delayed(current_time)

    # Use data where available and age < 0.5s
    fresh_mask = available & (age < 0.5)
    if fresh_mask.any():
        fresh_data = data[fresh_mask]
        # Process fresh_data...

# Get statistics
stats = buffer.get_statistics()
print(f"Success rate: {stats['success_rate']}")
```

### Example 2: Multi-Agent Communication

```python
from delayed_states import MultiAgentCommChannel

# Create communication channel
comm = MultiAgentCommChannel(
    num_envs=4,
    num_agents=3,
    mean_latency=0.1,
    std_latency=0.03,
    dropout_rate=0.02,
    dt=0.01,
    device=torch.device("cuda")
)

# Agent 0 sends state
sender_id = 0
state = {
    "position": torch.randn(4, 3, device="cuda"),
    "velocity": torch.randn(4, 3, device="cuda")
}
comm.send_message(sender_id, state, current_time)

# Agent 1 receives
receiver_id = 1
received = comm.receive_messages(receiver_id, current_time)

for sender, data_dict in received.items():
    if "position" in data_dict:
        pos, valid, age = data_dict["position"]
        print(f"From agent {sender}: position age = {age[0]:.3f}s")
```

### Example 3: Complete Observation Pipeline

```python
from delayed_states import MultiAgentObservationPipeline

# Create pipeline
pipeline = MultiAgentObservationPipeline(
    num_envs=4,
    num_agents=2,
    dt=0.01,
    device=torch.device("cuda"),
    detection_fps=30.0,
    comm_dropout_rate=0.02
)

# Main loop
for step in range(1000):
    # Update time
    pipeline.update_time()

    # Filter ego motion for agent 0
    true_pos = get_true_position(agent_id=0)
    true_quat = get_true_orientation(agent_id=0)
    true_lin_vel = get_true_linear_velocity(agent_id=0)
    true_ang_vel = get_true_angular_velocity(agent_id=0)

    filt_pos, filt_quat, filt_vel, filt_ang = pipeline.update_ego_motion(
        agent_id=0,
        true_position=true_pos,
        true_orientation=true_quat,
        true_linear_velocity=true_lin_vel,
        true_angular_velocity=true_ang_vel
    )

    # Add detection
    detection = get_detection(agent_id=0)
    has_detection = get_detection_validity(agent_id=0)
    pipeline.add_detection(0, detection, has_detection)

    # Get delayed detection
    delayed_det, det_valid, det_age = pipeline.get_delayed_detection(0)

    # Broadcast to other agents
    if det_valid.any():
        pipeline.broadcast_state(0, {"detection": delayed_det}, det_valid)

    # Receive from others
    received = pipeline.receive_other_agent_states(receiver_id=1)
    for sender, data_dict in received.items():
        if "detection" in data_dict:
            det, valid, age = data_dict["detection"]
            # Fuse detections...
```

---

## Key Concepts

### 1. Data Validity Semantics

The system distinguishes between two types of validity:

**Source Data Availability (`has_source_data`):**
- Independent of channel impairments
- Examples: sensor detected target, measurement is valid, state exists
- User-provided boolean mask

**Channel Transmission Success:**
- Affected by throttle, dropout, and latency
- Determined by the channel impairments

**Output Validity (`valid_mask`):**
```
valid_mask = has_source_data AND passed_throttle AND not_dropped
```

### 2. Data Age Tracking

All `get_delayed()` and `receive_messages()` operations return data age:
- Computed as: `current_time - data_added_time`
- Enables time-weighted fusion
- Helps detect stale data
- Zero if no data available or current_time is None

**Example use cases:**
```python
# Time-weighted averaging
weights = torch.exp(-age / tau)
fused = (data * weights.unsqueeze(-1)).sum(0) / weights.sum()

# Staleness detection
stale_mask = age > max_age
fresh_data = data[~stale_mask]

# Exponential moving average with age
alpha = 1 - torch.exp(-age / tau)
filtered = (1 - alpha) * old_value + alpha * new_value
```

### 3. Channel Impairment Order

Impairments are applied in physically realistic order:

1. **Throttle (FPS limiting)**: Determines if sample is taken
2. **Dropout (Packet loss)**: Determines if sample succeeds
3. **Latency (Delay)**: Determines when sample arrives

This order ensures that:
- Dropped packets don't consume bandwidth
- Latency only applies to successfully transmitted data
- Statistics accurately reflect real-world behavior

### 4. Per-Environment Control

All parameters support per-environment tensors:
```python
# Different latencies per environment
mean_latency = torch.tensor([0.05, 0.1, 0.05, 0.2], device=device)
buffer = ChannelBuffer(..., mean_latency=mean_latency, ...)

# Dynamic updates
new_dropout = torch.tensor([0.0, 0.0, 0.5, 0.5], device=device)
buffer.update_params(dropout_rate=new_dropout)
```

---

## Performance Considerations

### 1. Memory Usage

- Buffers use fixed-size circular buffers (max_history)
- Memory scales as: `O(num_envs * max_history * data_size)`
- No dynamic allocation during runtime
- GPU-friendly memory layout

### 2. Computational Efficiency

- All operations are batched across environments
- Minimal CPU-GPU transfers
- Uses Isaac Lab's optimized DelayBuffer internally
- Lazy buffer initialization (created on first use)

### 3. Best Practices

**Buffer Sizing:**
```python
# Rule of thumb: max_history should cover max expected delay
max_expected_delay = mean_latency + 3 * std_latency  # 99.7% coverage
max_history = int(max_expected_delay / dt) + 10  # Add margin
```

**Batch Operations:**
```python
# Good: Process all agents in parallel
for agent_id in range(num_agents):
    data = get_data(agent_id)  # [N, ...]
    buffer.compute(data, current_time)

# Avoid: Per-environment loops
# for env_id in range(num_envs):  # DON'T DO THIS
#     data = get_data(env_id)  # [1, ...]
```

**Reset Strategy:**
```python
# Reset only done environments (more efficient)
done_env_ids = torch.where(dones)[0]
if len(done_env_ids) > 0:
    pipeline.reset(done_env_ids)
```

---

## References

- Isaac Lab Documentation: [https://isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab)
- Test Suite: See [TEST_SUMMARY.md](tests/TEST_SUMMARY.md)
- Quick Reference: See [QUICK_START.md](tests/QUICK_START.md)

---

**Last Updated:** 2025-11-12
**Version:** 1.0
**Tested:** ✅ 68/68 tests passing
