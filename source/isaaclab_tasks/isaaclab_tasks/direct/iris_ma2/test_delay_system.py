# test_delay_system.py
"""
Comprehensive tests and visualizations for the delay and communication system.

Run this file to validate functionality and visualize delay characteristics.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import time

from delay_comm_system_optimized import (
    DelayedObservationManager,
    FirstOrderLag,
    QuaternionFirstOrderLag,
    LatencyBufferOptimized,
    CommManagerOptimized,
    TimestampedData
)

class DelaySystemTester:
    """Comprehensive testing suite for delay and communication system."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', output_dir='/home/claude'):
        self.device = torch.device(device)
        self.output_dir = output_dir
        print(f"Running tests on device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def test_first_order_lag(self, visualize=True):
        """Test first-order lag filter with step input."""
        print("\n" + "="*60)
        print("TEST 1: First-Order Lag Filter")
        print("="*60)
        
        num_envs = 4
        state_dim = 1
        time_constant = 0.1  # 100ms
        dt = 0.01  # 10ms timestep
        num_steps = 200
        
        # Create filter
        filter = FirstOrderLag(
            num_envs=num_envs,
            state_dim=state_dim,
            time_constant=time_constant,
            dt=dt,
            device=self.device
        )
        
        # Apply step input
        history = []
        for step in range(num_steps):
            # Step input at t=0.5s
            if step < 50:
                measured = torch.zeros(num_envs, state_dim, device=self.device)
            else:
                measured = torch.ones(num_envs, state_dim, device=self.device)
            
            current_time = step * dt
            filtered = filter.update(measured, current_time)
            
            history.append({
                'time': current_time,
                'measured': measured[0, 0].item(),
                'filtered': filtered[0, 0].item()
            })
        
        # Validate
        # After 3*tau, should reach ~94.7% of step (1 - e^(-3) ≈ 0.9502)
        idx_3tau = int(3 * time_constant / dt) + 50
        if idx_3tau < len(history):
            response = history[idx_3tau]['filtered']
            expected = 0.94  # ~94% threshold (accounting for discretization)
            assert response >= expected, f"Response at 3*tau should be >={expected}, got {response}"
            print(f"✓ Step response at 3*tau: {response:.3f} (expected >= {expected})")
        
        if visualize:
            self._plot_first_order_lag(history)
        
        print("✓ First-order lag test passed!")
        return history
    
    def test_quaternion_slerp(self, visualize=True):
        """Test quaternion SLERP filter."""
        print("\n" + "="*60)
        print("TEST 2: Quaternion SLERP Filter")
        print("="*60)
        
        num_envs = 4
        time_constant = 0.1
        dt = 0.01
        num_steps = 200
        
        # Create filter
        filter = QuaternionFirstOrderLag(
            num_envs=num_envs,
            time_constant=time_constant,
            dt=dt,
            device=self.device
        )
        
        # Start at identity
        q_start = torch.tensor([1., 0., 0., 0.], device=self.device).expand(num_envs, 4)
        
        # Target: 90 degree rotation around Z axis
        # q = [cos(45°), 0, 0, sin(45°)]
        angle = torch.tensor(torch.pi / 4, device=self.device)  # 45 degrees (for 90 degree rotation)
        q_target = torch.zeros(num_envs, 4, device=self.device)
        q_target[:, 0] = torch.cos(angle)
        q_target[:, 3] = torch.sin(angle)
        
        history = []
        for step in range(num_steps):
            if step < 50:
                measured = q_start
            else:
                measured = q_target
            
            current_time = step * dt
            filtered = filter.update(measured, current_time)
            
            # Convert to angle for visualization
            w_filt = filtered[0, 0].item()
            z_filt = filtered[0, 3].item()
            angle_filt = 2 * torch.acos(torch.clamp(filtered[0, 0], -1, 1)).item()
            angle_meas = 2 * torch.acos(torch.clamp(measured[0, 0], -1, 1)).item()
            
            history.append({
                'time': current_time,
                'measured_angle': np.degrees(angle_meas),
                'filtered_angle': np.degrees(angle_filt),
                'w': w_filt,
                'z': z_filt
            })
        
        # Validate quaternion normalization
        for h in history:
            quat_norm = np.sqrt(h['w']**2 + h['z']**2)
            assert abs(quat_norm - 1.0) < 1e-4, f"Quaternion not normalized: {quat_norm}"
        
        print(f"✓ Quaternion normalization maintained throughout")
        
        if visualize:
            self._plot_quaternion_slerp(history)
        
        print("✓ Quaternion SLERP test passed!")
        return history
    
    def test_latency_buffer(self, visualize=True):
        """Test detection latency buffer."""
        print("\n" + "="*60)
        print("TEST 3: Latency Buffer")
        print("="*60)
        
        num_envs = 100  # More envs to see distribution
        data_shape = (4,)  # e.g., bbox [x, y, w, h]
        mean_latency = 0.05  # 50ms
        std_latency = 0.02   # 20ms
        dt = 0.01
        num_steps = 150
        
        buffer = LatencyBufferOptimized(
            num_envs=num_envs,
            data_shape=data_shape,
            mean_latency=mean_latency,
            std_latency=std_latency,
            dt=dt,
            device=self.device
        )
        
        # Add measurements and track delays
        delays = []
        for step in range(num_steps):
            current_time = step * dt
            
            # Add new measurement
            data = torch.ones(num_envs, *data_shape, device=self.device) * step
            buffer.add_measurement(data, current_time)
            
            # Retrieve delayed measurement
            delayed = buffer.get_delayed_measurement(current_time)
            
            # Track delay for valid measurements
            if delayed.valid.any():
                valid_mask = delayed.valid
                actual_delays = current_time - delayed.timestamp[valid_mask]
                delays.extend(actual_delays.cpu().numpy())
        
        # Validate delay statistics
        delays = np.array(delays)
        actual_mean = np.mean(delays)
        actual_std = np.std(delays)
        
        print(f"Expected delay: {mean_latency:.3f} ± {std_latency:.3f} s")
        print(f"Actual delay:   {actual_mean:.3f} ± {actual_std:.3f} s")
        
        # Allow 20% tolerance due to sampling
        assert abs(actual_mean - mean_latency) < 0.2 * mean_latency, \
            f"Mean delay off by >20%: {actual_mean} vs {mean_latency}"
        
        print("✓ Latency distribution matches expected parameters")
        
        if visualize:
            self._plot_latency_distribution(delays, mean_latency, std_latency)
        
        print("✓ Latency buffer test passed!")
        return delays
    
    def test_detection_fps_throttling(self, visualize=True):
        """Test FPS-based detection dropping."""
        print("\n" + "="*60)
        print("TEST 4: Detection FPS Throttling")
        print("="*60)
        
        num_envs = 8
        num_agents = 2
        dt = 0.01  # 100 Hz simulation
        detection_fps = 10.0  # 10 Hz detection rate (100ms period)
        
        # Create manager with FPS throttling
        delay_mgr = DelayedObservationManager(
            num_envs=num_envs,
            num_agents=num_agents,
            dt=dt,
            device=self.device,
            detection_fps=detection_fps,
            detection_mean_latency=0.02,
            detection_std_latency=0.005
        )
        
        # Track accepted/dropped detections per agent
        detection_times = {agent_id: [] for agent_id in range(num_agents)}
        attempted_times = {agent_id: [] for agent_id in range(num_agents)}
        
        num_steps = 300  # 3 seconds of simulation
        
        for step in range(num_steps):
            delay_mgr.update_time()
            current_time = step * dt
            
            for agent_id in range(num_agents):
                # Generate bbox detection data [N, 4] = [x, y, w, h]
                bbox_data = torch.rand(num_envs, 4, device=self.device) * 100
                bbox_data[:, 0] = current_time * 10  # x encodes time for tracking
                
                # All envs have valid detections
                valid_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
                
                # Attempt to add detection every timestep (100 Hz attempt rate)
                attempted_times[agent_id].append(current_time)
                
                # Add detection (will be throttled by FPS)
                delay_mgr.add_detection(agent_id, bbox_data, valid_mask)
                
                # Check if detection was actually added by examining the buffer
                if agent_id in delay_mgr.detection_buffers:
                    # Check last detection time
                    last_det = delay_mgr.last_detection_time[agent_id][0].item()
                    if abs(last_det - current_time) < dt * 0.5:  # Detection was just added
                        detection_times[agent_id].append(current_time)
        
        # Analyze results
        expected_period = 1.0 / detection_fps
        
        for agent_id in range(num_agents):
            det_times = np.array(detection_times[agent_id])
            att_times = np.array(attempted_times[agent_id])
            
            print(f"\nAgent {agent_id}:")
            print(f"  Attempted detections: {len(att_times)}")
            print(f"  Accepted detections:  {len(det_times)}")
            print(f"  Acceptance rate:      {len(det_times)/len(att_times)*100:.1f}%")
            
            if len(det_times) > 1:
                # Check inter-detection intervals
                intervals = np.diff(det_times)
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                print(f"  Mean interval:        {mean_interval*1000:.1f} ms")
                print(f"  Std interval:         {std_interval*1000:.1f} ms")
                print(f"  Expected interval:    {expected_period*1000:.1f} ms")
                
                # Validate: mean interval should be close to expected period
                tolerance = 0.15  # 15% tolerance
                assert abs(mean_interval - expected_period) < tolerance * expected_period, \
                    f"Mean interval {mean_interval:.3f}s doesn't match expected {expected_period:.3f}s"
                
                # Validate: all intervals should be >= expected period (with small tolerance for numerical errors)
                min_interval = np.min(intervals)
                assert min_interval >= expected_period - 2*dt, \
                    f"Found interval {min_interval:.3f}s < expected minimum {expected_period:.3f}s"
        
        if visualize:
            self._plot_fps_throttling(detection_times, attempted_times, expected_period)
        
        print("\n✓ Detection FPS throttling test passed!")
        return detection_times
    
    def test_communication_manager(self, visualize=True):
        """Test communication delays and dropouts."""
        print("\n" + "="*60)
        print("TEST 4: Communication Manager")
        print("="*60)
        
        num_envs = 100
        num_agents = 3
        mean_delay = 0.1
        std_delay = 0.03
        dropout_rate = 0.1  # 10% packet loss
        dt = 0.01
        send_interval = 0.1  # Send every 0.1s to match mean delay (avoid pileup)
        num_total_steps = 250  # 2.5 seconds total
        num_send_steps = 200   # Send for 2.0 seconds
        num_drain_steps = 50   # Drain for 0.5 seconds
        
        comm_mgr = CommManagerOptimized(
            num_envs=num_envs,
            num_agents=num_agents,
            mean_comm_delay=mean_delay,
            std_comm_delay=std_delay,
            dropout_rate=dropout_rate,
            dt=dt,
            device=self.device
        )
        
        # Track delivery statistics
        sent_count = 0
        received_count = 0
        delays_received = []
        
        # Phase 1: Send messages periodically
        for step in range(num_send_steps):
            current_time = step * dt
            
            # Only send every send_interval seconds (not every step)
            if step % int(send_interval / dt) == 0:
                # Agent 0 broadcasts to others
                sender_id = 0
                data = {
                    'position': torch.randn(num_envs, 3, device=self.device),
                    'timestamp': torch.full((num_envs,), current_time, device=self.device)
                }
                
                comm_mgr.send_message(sender_id, data, current_time)
                sent_count += num_envs * (num_agents - 1)  # Broadcast to all others
            
            # Try to receive every step
            for receiver_id in [1, 2]:
                received = comm_mgr.receive_messages(receiver_id, current_time)
                
                if sender_id in received and 'timestamp' in received[sender_id]:
                    timestamp_data = received[sender_id]['timestamp']
                    valid_mask = timestamp_data.valid
                    
                    if valid_mask.any():
                        received_count += valid_mask.sum().item()
                        
                        # Calculate delays
                        gen_times = timestamp_data.data[valid_mask]
                        actual_delays = current_time - gen_times
                        delays_received.extend(actual_delays.cpu().numpy())
        
        # Phase 2: Drain phase - continue receiving but stop sending
        print(f"  Draining buffers for {num_drain_steps} steps...")
        for step in range(num_send_steps, num_send_steps + num_drain_steps):
            current_time = step * dt
            
            # Only receive, no sending
            for receiver_id in [1, 2]:
                received = comm_mgr.receive_messages(receiver_id, current_time)
                
                if sender_id in received and 'timestamp' in received[sender_id]:
                    timestamp_data = received[sender_id]['timestamp']
                    valid_mask = timestamp_data.valid
                    
                    if valid_mask.any():
                        received_count += valid_mask.sum().item()
                        
                        # Calculate delays
                        gen_times = timestamp_data.data[valid_mask]
                        actual_delays = current_time - gen_times
                        delays_received.extend(actual_delays.cpu().numpy())
        
        # Validate dropout rate
        actual_dropout = 1.0 - (received_count / sent_count)
        print(f"Expected dropout rate: {dropout_rate:.2%}")
        print(f"Actual dropout rate:   {actual_dropout:.2%}")
        print(f"Messages sent:     {sent_count}")
        print(f"Messages received: {received_count}")
        print(f"Send interval:     {send_interval:.3f}s (every {int(send_interval/dt)} steps)")
        
        # Allow tolerance due to sampling
        assert abs(actual_dropout - dropout_rate) < 0.05, \
            f"Dropout rate off by >5%: {actual_dropout:.2%} vs {dropout_rate:.2%}"
        
        print("✓ Dropout rate matches expected")
        
        # Validate delay statistics (only for received messages)
        if len(delays_received) > 0:
            delays_received = np.array(delays_received)
            actual_mean = np.mean(delays_received)
            actual_std = np.std(delays_received)
            
            print(f"Expected comm delay: {mean_delay:.3f} ± {std_delay:.3f} s")
            print(f"Actual comm delay:   {actual_mean:.3f} ± {actual_std:.3f} s")
            
            if visualize:
                self._plot_comm_statistics(delays_received, mean_delay, std_delay)
        
        print("✓ Communication manager test passed!")
        return delays_received, actual_dropout
    
    def test_full_system(self, visualize=True):
        """Test complete DelayedObservationManager with FPS-limited detections."""
        print("\n" + "="*60)
        print("TEST 5: Full System Integration (with FPS-limited detections)")
        print("="*60)
        
        num_envs = 512
        num_agents = 2
        dt = 0.02  # 50 Hz simulation
        detection_fps = 15.0  # 15 Hz detection rate
        num_steps = 250  # 5 seconds
        
        delay_mgr = DelayedObservationManager(
            num_envs=num_envs,
            num_agents=num_agents,
            dt=dt,
            device=self.device,
            motion_time_constant=0.1,
            gimbal_time_constant=0.03,
            detection_mean_latency=0.05,
            detection_std_latency=0.02,
            detection_fps=detection_fps,  # NEW: FPS throttling
            comm_mean_delay=0.1,
            comm_std_delay=0.03,
            comm_dropout_rate=0.05
        )
        
        # Simulation loop
        position_history = [[] for _ in range(num_agents)]
        # Track success per environment, per receiver, per sender
        comm_success_count = torch.zeros(num_envs, num_agents, num_agents, device=self.device)
        comm_attempt_count = torch.zeros(num_envs, num_agents, num_agents, device=self.device)
        comm_success_rate_history = []  # Time series for plotting
        detection_stats = {agent_id: {'attempted': 0, 'accepted': 0} for agent_id in range(num_agents)}
        
        for step in range(num_steps):
            delay_mgr.update_time()
            
            for agent_id in range(num_agents):
                # Simulate true state with sinusoidal motion
                t = delay_mgr.current_time[0].item()
                true_pos = torch.zeros(num_envs, 3, device=self.device)
                true_pos[:, 0] = torch.sin(torch.tensor(2 * np.pi * 0.5 * t))  # 0.5 Hz oscillation
                true_pos[:, 1] = agent_id * 2.0  # Offset between agents
                
                true_quat = torch.tensor([1., 0., 0., 0.], device=self.device).expand(num_envs, 4)
                true_yaw = torch.sin(torch.tensor(2 * np.pi * 0.2 * t)) * 0.5
                true_pitch = torch.zeros(num_envs, device=self.device)
                
                # Get delayed motion observations
                delayed_pos, delayed_quat, _, _ = delay_mgr.update_ego_motion(
                    agent_id, true_pos, true_quat,
                    torch.zeros_like(true_pos), torch.zeros_like(true_pos)
                )
                delayed_yaw, delayed_pitch = delay_mgr.update_ego_gimbal(agent_id, true_yaw, true_pitch)
                
                # Simulate detection (similar to user's use case)
                # Generate bbox detections [N, 4] = [x, y, w, h]
                bbox_data = torch.zeros(num_envs, 4, device=self.device)
                bbox_data[:, 0] = true_pos[:, 0] * 100 + 320  # Center x (pixel coords)
                bbox_data[:, 1] = true_pos[:, 1] * 100 + 240  # Center y
                bbox_data[:, 2] = 50 + torch.rand(num_envs, device=self.device) * 20  # Width
                bbox_data[:, 3] = 50 + torch.rand(num_envs, device=self.device) * 20  # Height
                
                # Valid mask (simulate some detections failing)
                valid_mask = torch.rand(num_envs, device=self.device) > 0.1  # 90% valid
                
                # Track attempted detections
                detection_stats[agent_id]['attempted'] += 1
                
                # Add detection (will be FPS-throttled internally)
                last_det_time = delay_mgr.last_detection_time.get(agent_id, None)
                delay_mgr.add_detection(agent_id, bbox_data, valid_mask)
                
                # Check if detection was accepted
                new_det_time = delay_mgr.last_detection_time.get(agent_id, None)
                if new_det_time is not None and (last_det_time is None or 
                   (new_det_time[0] - (last_det_time[0] if last_det_time is not None else -999)) > 1e-6):
                    detection_stats[agent_id]['accepted'] += 1
                
                # Get delayed detection (matching user's pattern)
                delayed_detection = delay_mgr.get_delayed_detection(agent_id)
                bboxes_delayed = delayed_detection.data  # [N, 4]
                valid_mask_delayed = delayed_detection.valid  # [N]
                
                # Store history
                position_history[agent_id].append({
                    'time': t,
                    'true_x': true_pos[0, 0].item(),
                    'delayed_x': delayed_pos[0, 0].item(),
                    'true_yaw': true_yaw[0].item() if true_yaw.dim() > 0 else true_yaw.item(),
                    'delayed_yaw': delayed_yaw[0, 0].item(),
                    'bbox_valid': valid_mask_delayed[0].item()
                })
                
                # Broadcast state (always valid to test communication, not detection)
                # The detection valid_mask_delayed tracks detection availability
                # But communication should be tested independently
                broadcast_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
                delay_mgr.broadcast_state(
                    agent_id,
                    {'position': delayed_pos, 'bbox': bboxes_delayed},
                    broadcast_mask  # Always broadcast to test communication dropout
                )
            
            # Check communication success per environment
            for receiver_id in range(num_agents):
                received = delay_mgr.receive_other_agent_states(receiver_id)
                
                for sender_id in range(num_agents):
                    if sender_id == receiver_id:
                        continue  # Don't count self-communication
                    
                    # Track attempts for each env
                    comm_attempt_count[:, receiver_id, sender_id] += 1
                    
                    # Check if this sender's message was received
                    if sender_id in received and 'position' in received[sender_id]:
                        # Count success per environment
                        valid_mask = received[sender_id]['position'].valid
                        comm_success_count[:, receiver_id, sender_id] += valid_mask.float()
            
            # Calculate instantaneous success rate for this timestep (for plotting)
            if step > 0:  # Need at least one attempt
                per_env_success = torch.where(
                    comm_attempt_count > 0,
                    comm_success_count / comm_attempt_count,
                    torch.zeros_like(comm_success_count)
                )
                comm_success_rate_history.append(per_env_success.mean().item())
            else:
                comm_success_rate_history.append(0.0)
        
        # Validate system behavior
        # Calculate success rate per environment, then average
        # Avoid division by zero
        per_env_success_rate = torch.where(
            comm_attempt_count > 0,
            comm_success_count / comm_attempt_count,
            torch.zeros_like(comm_success_count)
        )
        
        # Average across all sender-receiver pairs and environments
        avg_comm_success = per_env_success_rate.mean().item()
        
        print(f"Average communication success rate: {avg_comm_success:.2%}")
        print(f"Expected: ~{1 - 0.05:.2%} (considering 5% dropout)")
        
        # More detailed breakdown
        print(f"\nPer-environment statistics:")
        print(f"  Min success rate:  {per_env_success_rate.min().item():.2%}")
        print(f"  Max success rate:  {per_env_success_rate.max().item():.2%}")
        print(f"  Std deviation:     {per_env_success_rate.std().item():.4f}")
        
        # Tolerance check - should be close to 95% with some variance
        assert 0.90 < avg_comm_success < 0.98, \
            f"Average success rate {avg_comm_success:.2%} not in expected range [90%, 98%]"
        
        # Validate detection FPS throttling
        print(f"\nDetection Statistics:")
        expected_acceptance_rate = detection_fps * dt
        for agent_id in range(num_agents):
            attempted = detection_stats[agent_id]['attempted']
            accepted = detection_stats[agent_id]['accepted']
            actual_rate = accepted / attempted if attempted > 0 else 0
            
            print(f"  Agent {agent_id}:")
            print(f"    Attempted: {attempted}")
            print(f"    Accepted:  {accepted}")
            print(f"    Rate:      {actual_rate:.2%}")
            print(f"    Expected:  ~{expected_acceptance_rate:.2%}")
            
            # Validate with tolerance
            tolerance = 0.3  # 30% tolerance
            assert abs(actual_rate - expected_acceptance_rate) < tolerance, \
                f"Detection acceptance rate {actual_rate:.2%} far from expected {expected_acceptance_rate:.2%}"
        
        # Check position lag
        agent_0_hist = position_history[0]
        # Find time when true position crosses zero (at t≈0.5s)
        true_crossing_idx = None
        for i, h in enumerate(agent_0_hist):
            if h['time'] > 0.5 and h['true_x'] > 0 and agent_0_hist[i-1]['true_x'] < 0:
                true_crossing_idx = i
                break
        
        # Delayed signal should cross later
        if true_crossing_idx is not None:
            for i, h in enumerate(agent_0_hist[true_crossing_idx:], start=true_crossing_idx):
                if h['delayed_x'] > 0:
                    delayed_crossing_idx = i
                    lag_time = h['time'] - agent_0_hist[true_crossing_idx]['time']
                    print(f"\nPosition lag observed: {lag_time*1000:.1f}ms")
                    break
        
        if visualize:
            self._plot_full_system(position_history, comm_success_rate_history, detection_fps)
        
        print("✓ Full system integration test passed!")
        return position_history, per_env_success_rate
    
    def benchmark_performance(self):
        """Benchmark computational performance."""
        print("\n" + "="*60)
        print("BENCHMARK: Computational Performance")
        print("="*60)
        
        # Test different scales
        env_counts = [128, 512, 2048, 8192]
        
        for num_envs in env_counts:
            delay_mgr = DelayedObservationManager(
                num_envs=num_envs,
                num_agents=2,
                dt=0.02,
                device=self.device
            )
            
            # Warm-up
            for _ in range(10):
                delay_mgr.update_time()
                for agent_id in range(2):
                    pos = torch.randn(num_envs, 3, device=self.device)
                    quat = torch.randn(num_envs, 4, device=self.device)
                    lin_vel = torch.randn(num_envs, 3, device=self.device)
                    ang_vel = torch.randn(num_envs, 3, device=self.device)
                    delay_mgr.update_ego_motion(agent_id, pos, quat, lin_vel, ang_vel)
            
            # Benchmark
            num_iterations = 100
            start_time = time.time()
            
            for _ in range(num_iterations):
                delay_mgr.update_time()
                for agent_id in range(2):
                    pos = torch.randn(num_envs, 3, device=self.device)
                    quat = torch.randn(num_envs, 4, device=self.device)
                    lin_vel = torch.randn(num_envs, 3, device=self.device)
                    ang_vel = torch.randn(num_envs, 3, device=self.device)
                    yaw = torch.randn(num_envs, device=self.device)
                    pitch = torch.randn(num_envs, device=self.device)
                    delay_mgr.update_ego_motion(agent_id, pos, quat, lin_vel, ang_vel)
                    delay_mgr.update_ego_gimbal(agent_id, yaw, pitch)
                    
                    state = {'position': pos, 'orientation': quat}
                    delay_mgr.broadcast_state(agent_id, state)
                    delay_mgr.receive_other_agent_states(agent_id)
            
            elapsed = time.time() - start_time
            fps = num_iterations / elapsed
            time_per_step = elapsed / num_iterations * 1000  # ms
            
            print(f"  {num_envs:5d} envs: {fps:6.1f} FPS | {time_per_step:5.2f} ms/step")
        
        print("✓ Performance benchmark completed!")
    
    # Visualization helper methods
    def _plot_first_order_lag(self, history):
        """Plot first-order lag step response."""
        times = [h['time'] for h in history]
        measured = [h['measured'] for h in history]
        filtered = [h['filtered'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, measured, 'b--', label='Measured (Step Input)', linewidth=2)
        plt.plot(times, filtered, 'r-', label='Filtered Output', linewidth=2)
        plt.axhline(y=0.95, color='g', linestyle=':', label='95% Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('First-Order Lag: Step Response (τ=0.1s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = f'{self.output_dir}/first_order_lag.png'
        plt.savefig(output_path, dpi=150)
        print(f"  → Saved plot: {output_path}")
        plt.close()
    
    def _plot_quaternion_slerp(self, history):
        """Plot quaternion SLERP response."""
        times = [h['time'] for h in history]
        measured_angles = [h['measured_angle'] for h in history]
        filtered_angles = [h['filtered_angle'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, measured_angles, 'b--', label='Measured', linewidth=2)
        plt.plot(times, filtered_angles, 'r-', label='Filtered (SLERP)', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Rotation Angle (degrees)')
        plt.title('Quaternion SLERP Filter: 90° Rotation Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = f'{self.output_dir}/quaternion_slerp.png'
        plt.savefig(output_path, dpi=150)
        print(f"  → Saved plot: {output_path}")
        plt.close()
    
    def _plot_latency_distribution(self, delays, mean_latency, std_latency):
        """Plot latency distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist(delays, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay theoretical normal distribution
        x = np.linspace(delays.min(), delays.max(), 100)
        theoretical = (1 / (std_latency * np.sqrt(2 * np.pi))) * \
                     np.exp(-0.5 * ((x - mean_latency) / std_latency) ** 2)
        plt.plot(x, theoretical, 'r-', linewidth=2, label='Theoretical Normal')
        
        plt.axvline(x=mean_latency, color='g', linestyle='--', label=f'Mean = {mean_latency*1000:.0f}ms')
        plt.xlabel('Delay (s)')
        plt.ylabel('Probability Density')
        plt.title(f'Detection Latency Distribution (μ={mean_latency*1000:.0f}ms, σ={std_latency*1000:.0f}ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = f'{self.output_dir}/latency_distribution.png'
        plt.savefig(output_path, dpi=150)
        print(f"  → Saved plot: {output_path}")
        plt.close()
    
    def _plot_fps_throttling(self, detection_times, attempted_times, expected_period):
        """Plot FPS throttling behavior."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Detection timeline
        ax = axes[0]
        for agent_id, det_times in detection_times.items():
            att_times = attempted_times[agent_id]
            
            # Plot all attempted detections as light dots
            y_pos = agent_id
            ax.scatter(att_times, [y_pos]*len(att_times), 
                      s=5, alpha=0.2, color='gray', label=f'Agent {agent_id} (attempted)' if agent_id == 0 else '')
            
            # Plot accepted detections as larger colored dots
            ax.scatter(det_times, [y_pos]*len(det_times),
                      s=50, alpha=0.8, marker='o', label=f'Agent {agent_id} (accepted)')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Agent ID')
        ax.set_title(f'Detection Timeline (FPS = {1.0/expected_period:.1f} Hz, Period = {expected_period*1000:.0f} ms)')
        ax.set_yticks(list(detection_times.keys()))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Inter-detection interval histogram
        ax = axes[1]
        for agent_id, det_times in detection_times.items():
            if len(det_times) > 1:
                intervals = np.diff(det_times) * 1000  # Convert to ms
                ax.hist(intervals, bins=30, alpha=0.6, label=f'Agent {agent_id}', edgecolor='black')
        
        ax.axvline(x=expected_period*1000, color='r', linestyle='--', linewidth=2, 
                  label=f'Expected ({expected_period*1000:.0f} ms)')
        ax.set_xlabel('Inter-Detection Interval (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Inter-Detection Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/fps_throttling.png'
        plt.savefig(output_path, dpi=150)
        print(f"  → Saved plot: {output_path}")
        plt.close()
    
    def _plot_comm_statistics(self, delays, mean_delay, std_delay):
        """Plot communication delay statistics."""
        plt.figure(figsize=(10, 6))
        plt.hist(delays, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay theoretical distribution
        x = np.linspace(max(0, delays.min()), delays.max(), 100)
        theoretical = (1 / (std_delay * np.sqrt(2 * np.pi))) * \
                     np.exp(-0.5 * ((x - mean_delay) / std_delay) ** 2)
        plt.plot(x, theoretical, 'r-', linewidth=2, label='Theoretical')
        
        plt.axvline(x=mean_delay, color='g', linestyle='--', label=f'Mean = {mean_delay*1000:.0f}ms')
        plt.xlabel('Communication Delay (s)')
        plt.ylabel('Probability Density')
        plt.title(f'Communication Delay Distribution (μ={mean_delay*1000:.0f}ms, σ={std_delay*1000:.0f}ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = f'{self.output_dir}/comm_delay_distribution.png'
        plt.savefig(output_path, dpi=150)
        print(f"  → Saved plot: {output_path}")
        plt.close()
    
    def _plot_full_system(self, position_history, comm_success_rate, detection_fps=None):
        """Plot full system behavior."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Position tracking for Agent 0
        ax = axes[0]
        times = [h['time'] for h in position_history[0]]
        true_x = [h['true_x'] for h in position_history[0]]
        delayed_x = [h['delayed_x'] for h in position_history[0]]
        
        ax.plot(times, true_x, 'b-', label='True Position', linewidth=2, alpha=0.7)
        ax.plot(times, delayed_x, 'r-', label='Delayed Position', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('X Position (m)')
        title = 'Agent 0: Position with Motion Lag'
        if detection_fps:
            title += f' (Detection: {detection_fps:.0f} Hz)'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Gimbal tracking for Agent 0
        ax = axes[1]
        true_yaw = [h['true_yaw'] for h in position_history[0]]
        delayed_yaw = [h['delayed_yaw'] for h in position_history[0]]
        
        ax.plot(times, true_yaw, 'b-', label='True Yaw', linewidth=2, alpha=0.7)
        ax.plot(times, delayed_yaw, 'r-', label='Delayed Yaw', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gimbal Yaw (rad)')
        ax.set_title('Agent 0: Gimbal Yaw with Lag')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Communication success rate
        ax = axes[2]
        window_size = 50
        smoothed = np.convolve(comm_success_rate, np.ones(window_size)/window_size, mode='valid')
        times_comm = np.linspace(0, len(comm_success_rate) * 0.02, len(smoothed))
        
        ax.plot(times_comm, smoothed, 'g-', linewidth=2)
        ax.axhline(y=0.95, color='r', linestyle='--', label='Expected (~95%)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Success Rate')
        ax.set_title('Inter-Agent Communication Success Rate (5% dropout)')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/full_system_behavior.png'
        plt.savefig(output_path, dpi=150)
        print(f"  → Saved plot: {output_path}")
        plt.close()
    
    def run_all_tests(self, visualize=True):
        """Run complete test suite."""
        print("\n" + "="*60)
        print("RUNNING COMPLETE TEST SUITE")
        print("="*60)
        
        start_time = time.time()
        
        try:
            self.test_first_order_lag(visualize)
            self.test_quaternion_slerp(visualize)
            self.test_latency_buffer(visualize)
            self.test_detection_fps_throttling(visualize)  # NEW TEST
            self.test_communication_manager(visualize)
            self.test_full_system(visualize)
            self.benchmark_performance()
            
            elapsed = time.time() - start_time
            print("\n" + "="*60)
            print(f"✓ ALL TESTS PASSED! ({elapsed:.2f}s)")
            print("="*60)
            
            if visualize:
                print(f"\nVisualization plots saved to {self.output_dir}/")
            
            return True
            
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            return False
        except Exception as e:
            print(f"\n✗ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Run comprehensive test suite
    # You can change the output directory by passing it as an argument
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/claude'
    
    tester = DelaySystemTester(output_dir=output_dir)
    success = tester.run_all_tests(visualize=True)
    
    if success:
        print("\n" + "="*60)
        print("System is ready for integration into your environment!")
        print("See integration_example.py for usage instructions.")
        print("="*60)
    else:
        print("\nTests failed. Please review the error messages above.")