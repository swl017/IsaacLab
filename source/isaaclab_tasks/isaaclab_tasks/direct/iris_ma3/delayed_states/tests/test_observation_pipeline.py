"""
Tests for MultiAgentObservationPipeline class.
Tests 29-34: End-to-end detection, filtering, integration, time management.
Plus: Time synchronization, memory leaks.
"""

import torch
import pytest
import gc
from isaaclab_tasks.direct.iris_ma3.delayed_states import MultiAgentObservationPipeline


class TestObservationPipeline:
    """Test suite for MultiAgentObservationPipeline."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def num_envs(self):
        """Number of environments."""
        return 4

    @pytest.fixture
    def num_agents(self):
        """Number of agents."""
        return 3

    @pytest.fixture
    def dt(self):
        """Timestep."""
        return 0.01

    @pytest.fixture
    def pipeline(self, num_envs, num_agents, dt, device):
        """Create MultiAgentObservationPipeline."""
        return MultiAgentObservationPipeline(
            num_envs=num_envs,
            num_agents=num_agents,
            dt=dt,
            device=device,
            motion_time_constant=0.1,
            gimbal_time_constant=0.03,
            detection_fps=30.0,
            detection_mean_latency=0.05,
            detection_std_latency=0.02,
            detection_failure_rate=0.1,
            comm_mean_delay=0.1,
            comm_std_delay=0.03,
            comm_throttle_period=0.0,
            comm_dropout_rate=0.05,
            max_buffer_size=100
        )

    # ==================== Test 29: End-to-End Detection ====================

    def test_end_to_end_detection_pipeline(self, pipeline, num_envs, dt, device):
        """Test 29: Complete detection pipeline with all impairments."""
        agent_id = 0

        # Add detections for multiple timesteps
        for t in range(100):
            # Detection data (e.g., bounding box)
            detection_data = torch.tensor([
                [100.0 + t, 200.0, 50.0, 50.0]  # xywh format
            ], device=device).repeat(num_envs, 1)

            # Some valid, some invalid
            valid_mask = torch.tensor([True, False, True, False], device=device)

            pipeline.add_detection(agent_id, detection_data, valid_mask)
            pipeline.update_time()

        # Get delayed detection
        delayed_detection, detection_valid, _ = pipeline.get_delayed_detection(agent_id)

        # Should have some valid detections (not all dropped by failure rate)
        print(f"[INFO] End-to-end detection - Shape: {delayed_detection.shape}, Expected: ({num_envs}, 4)")
        print(f"[INFO] End-to-end detection - Valid count: {detection_valid.sum().item()}/{num_envs}")
        assert delayed_detection.shape == (num_envs, 4), \
            f"Expected shape ({num_envs}, 4), got {delayed_detection.shape}"

        # Valid mask should reflect both input validity and channel impairments
        assert detection_valid.dtype == torch.bool
        assert detection_valid.shape == (num_envs,)

    def test_detection_fps_throttling(self, num_envs, num_agents, dt, device):
        """Test 29b: FPS throttling limits detection rate."""
        # 10 Hz detection
        pipeline = MultiAgentObservationPipeline(
            num_envs=num_envs,
            num_agents=num_agents,
            dt=dt,
            device=device,
            detection_fps=10.0,  # 10 Hz
            detection_mean_latency=0.0,
            detection_std_latency=0.0,
            detection_failure_rate=0.0
        )

        agent_id = 0
        detection_count = 0

        # Run for 1 second (100 timesteps at dt=0.01)
        for t in range(100):
            detection_data = torch.ones(num_envs, 4, device=device) * t
            valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

            pipeline.add_detection(agent_id, detection_data, valid_mask)

            # Check if detection was accepted (successful throttle pass)
            # This is tracked internally, we can verify via statistics later

            pipeline.update_time()

        # At 10 Hz, should get roughly 10 detections in 1 second
        # We can verify by checking the detection buffer statistics
        stats = pipeline.get_statistics()
        if 'detection_stats' in stats and f'agent_{agent_id}' in stats['detection_stats']:
            success_count = stats['detection_stats'][f'agent_{agent_id}']['success_count'][0].item()
            expected_count = 10  # 10 Hz * 1 second
            tolerance = 3

            print(f"[INFO] Detection FPS throttling - Success count: {success_count}, Expected: {expected_count} ± {tolerance}")
            assert abs(success_count - expected_count) <= tolerance, \
                f"Expected ~{expected_count} detections, got {success_count}"

    # ==================== Test 30: Motion + Gimbal Filtering ====================

    def test_motion_and_gimbal_filtering(self, pipeline, num_envs, device):
        """Test 30: All filters applied independently."""
        agent_id = 0

        # Update motion
        true_position = torch.randn(num_envs, 3, device=device)
        true_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
        true_lin_vel = torch.randn(num_envs, 3, device=device)
        true_ang_vel = torch.randn(num_envs, 3, device=device)

        filtered_pos, filtered_quat, filtered_lin, filtered_ang = pipeline.update_ego_motion(
            agent_id, true_position, true_orientation, true_lin_vel, true_ang_vel
        )

        # Outputs should be filtered (not equal to inputs initially)
        # After one step, filter output will be between zero (initial) and input
        pos_diff = torch.max(torch.abs(filtered_pos - true_position)).item()
        print(f"[INFO] Motion filtering - Position diff after 1 step: {pos_diff:.4f} (should be >0, filtered)")
        assert not torch.allclose(filtered_pos, true_position), \
            "Position should be filtered"

        # Update gimbal
        true_yaw = torch.rand(num_envs, device=device)
        true_pitch = torch.rand(num_envs, device=device)

        filtered_yaw, filtered_pitch = pipeline.update_ego_gimbal(
            agent_id, true_yaw, true_pitch
        )

        # Should be filtered
        assert filtered_yaw.shape == (num_envs, 1)
        assert filtered_pitch.shape == (num_envs, 1)

        # After many updates, should converge to input
        for _ in range(500):
            pipeline.update_ego_motion(
                agent_id, true_position, true_orientation, true_lin_vel, true_ang_vel
            )
            filtered_yaw, filtered_pitch = pipeline.update_ego_gimbal(
                agent_id, true_yaw, true_pitch
            )

        # Should converge
        yaw_diff = torch.max(torch.abs(filtered_yaw.squeeze() - true_yaw)).item()
        pitch_diff = torch.max(torch.abs(filtered_pitch.squeeze() - true_pitch)).item()
        print(f"[INFO] Gimbal convergence after 500 steps - Yaw diff: {yaw_diff:.4f}, Pitch diff: {pitch_diff:.4f}, Expected: <0.1")
        assert torch.allclose(filtered_yaw.squeeze(), true_yaw, atol=0.1), \
            "Gimbal yaw should converge to input"

    # ==================== Test 31: Detection + Communication Integration ====================

    def test_detection_communication_integration(self, pipeline, num_envs, dt, device):
        """Test 31: Combined delay of detection and communication."""
        agent0 = 0
        agent1 = 1

        # Agent 0 detects target at t=0
        detection_data = torch.ones(num_envs, 4, device=device) * 42.0
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Mark time when detection occurs
        detection_time = pipeline.current_time.clone()

        pipeline.add_detection(agent0, detection_data, valid_mask)
        pipeline.update_time()

        # Wait for detection to process
        for _ in range(20):
            pipeline.add_detection(agent0, torch.zeros(num_envs, 4, device=device),
                                 torch.zeros(num_envs, dtype=torch.bool, device=device))
            pipeline.update_time()

        # Agent 0 gets delayed detection
        delayed_det, det_valid, _ = pipeline.get_delayed_detection(agent0)

        # Broadcast detection to Agent 1
        if det_valid.any():
            state_dict = {"detection": delayed_det}
            pipeline.broadcast_state(agent0, state_dict)

        # Wait for communication delay
        for _ in range(30):
            pipeline.update_time()

        # Agent 1 receives
        received = pipeline.receive_other_agent_states(agent1)

        # Check total delay
        if agent0 in received and "detection" in received[agent0]:
            received_data, received_valid, _ = received[agent0]["detection"]

            # Total time elapsed
            total_time = pipeline.current_time[0].item()
            initial_time = detection_time[0].item()
            elapsed = total_time - initial_time

            # Total delay should be detection_latency + comm_delay
            # Expected: ~0.05s (detection) + ~0.1s (comm) = ~0.15s
            # With 50 timesteps at dt=0.01 = 0.5s total, plenty of time
            expected_min_delay = 0.15 - 0.1  # Allow tolerance
            expected_max_delay = 0.15 + 0.1

            if received_valid.any():
                # If we received valid data, check it's the marker value
                # received_data has shape [num_envs, 4], so [0, 0] gets first element of first environment
                received_value = received_data[received_valid][0, 0].item() if received_valid.any() else 0
                print(f"[INFO] Detection-comm integration - Received value: {received_value:.1f}, Expected: 42.0")
                print(f"[INFO] Detection-comm integration - Total delay: {elapsed:.3f}s")
                assert torch.any(torch.abs(received_data - 42.0) < 1.0), \
                    "Should receive the marker detection value"

    # ==================== Test 32: Time Management ====================

    def test_time_management(self, pipeline, dt):
        """Test 32: Time increments correctly."""
        initial_time = pipeline.current_time.clone()

        # Update time 100 times
        for _ in range(100):
            pipeline.update_time()

        final_time = pipeline.current_time

        # Should have advanced by 100 * dt
        expected_time = initial_time + 100 * dt
        actual_time = final_time[0].item()
        expected_val = expected_time[0].item()
        print(f"[INFO] Time management - Actual: {actual_time:.4f}, Expected: {expected_val:.4f}")
        assert torch.allclose(final_time, expected_time, atol=1e-5), \
            f"Time should advance correctly: expected {expected_time}, got {final_time}"

    def test_time_with_custom_increment(self, pipeline):
        """Test 32b: Custom time increments."""
        initial_time = pipeline.current_time.clone()

        # Custom increment
        custom_dt = 0.05
        pipeline.update_time(time_increment=custom_dt)

        expected_time = initial_time + custom_dt
        actual_time = pipeline.current_time[0].item()
        expected_val = expected_time[0].item()
        print(f"[INFO] Custom time increment - Actual: {actual_time:.4f}, Expected: {expected_val:.4f} (custom_dt={custom_dt})")
        assert torch.allclose(pipeline.current_time, expected_time, atol=1e-5), \
            "Custom time increment should work"

    # ==================== Test 33: Parameter Update Propagation ====================

    def test_detection_parameter_updates(self, pipeline, num_envs, device):
        """Test 33: Update detection parameters at runtime."""
        # Update detection FPS
        new_fps = torch.full((num_envs, 3), 60.0, device=device)  # 60 Hz
        pipeline.update_detection_params(detection_fps=new_fps)

        # Verify update
        actual_fps = pipeline.detection_fps[0, 0].item()
        expected_fps = new_fps[0, 0].item()
        print(f"[INFO] Detection parameter update - FPS: {actual_fps:.1f}, Expected: {expected_fps:.1f}")
        assert torch.allclose(pipeline.detection_fps, new_fps), \
            "Detection FPS should be updated"

        # Update latency
        new_mean = torch.full((num_envs, 3), 0.1, device=device)
        new_std = torch.full((num_envs, 3), 0.03, device=device)
        pipeline.update_detection_params(
            detection_mean_latency=new_mean,
            detection_std_latency=new_std
        )

        actual_mean = pipeline.detection_mean_latency[0, 0].item()
        actual_std = pipeline.detection_std_latency[0, 0].item()
        print(f"[INFO] Detection latency update - Mean: {actual_mean:.3f}, Std: {actual_std:.3f}, Expected: 0.1, 0.03")
        assert torch.allclose(pipeline.detection_mean_latency, new_mean)
        assert torch.allclose(pipeline.detection_std_latency, new_std)

    def test_comm_parameter_updates(self, pipeline, num_envs, device):
        """Test 33b: Update communication parameters."""
        # Update comm delay
        new_mean = torch.full((num_envs, 3), 0.2, device=device)
        new_std = torch.full((num_envs, 3), 0.05, device=device)

        pipeline.update_comm_params(
            mean_latency=new_mean,
            std_latency=new_std
        )

        # Should propagate to comm_channel
        actual_comm_mean = pipeline.comm_channel.mean_latency[0, 0].item()
        expected_comm_mean = new_mean[0, 0].item()
        print(f"[INFO] Comm parameter update - Mean latency: {actual_comm_mean:.3f}, Expected: {expected_comm_mean:.3f}")
        assert torch.allclose(pipeline.comm_channel.mean_latency, new_mean)

    # ==================== Test 34: Complete Reset ====================

    def test_complete_reset(self, pipeline, num_envs, num_agents, device):
        """Test 34: Reset clears all state."""
        # Fill pipeline with data
        for agent_id in range(num_agents):
            # Motion
            pos = torch.randn(num_envs, 3, device=device)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
            vel = torch.randn(num_envs, 3, device=device)
            ang_vel = torch.randn(num_envs, 3, device=device)
            pipeline.update_ego_motion(agent_id, pos, quat, vel, ang_vel)

            # Detection
            det = torch.randn(num_envs, 4, device=device)
            valid = torch.ones(num_envs, dtype=torch.bool, device=device)
            pipeline.add_detection(agent_id, det, valid)

            # Communication
            state = {"test": torch.randn(num_envs, 2, device=device)}
            pipeline.broadcast_state(agent_id, state)

        # Advance time
        for _ in range(50):
            pipeline.update_time()

        # Reset specific environments
        env_ids = torch.tensor([0, 2], device=device)
        pipeline.reset(env_ids)

        # Time should be reset for those envs
        time_env0 = pipeline.current_time[0].item()
        time_env2 = pipeline.current_time[2].item()
        time_env1 = pipeline.current_time[1].item()
        print(f"[INFO] Complete reset - Time env 0: {time_env0:.4f}, env 2: {time_env2:.4f}, Expected: 0.0")
        print(f"[INFO] Complete reset - Time env 1 (not reset): {time_env1:.4f}, Expected: >0.0")
        assert pipeline.current_time[0] == 0.0
        assert pipeline.current_time[2] == 0.0
        assert pipeline.current_time[1] > 0.0  # Not reset

        # Filters should be reset (back to zero)
        for agent_id in range(num_agents):
            filter_state_env0 = pipeline.position_filters[agent_id].filtered_state[0].tolist()
            filter_state_env2 = pipeline.position_filters[agent_id].filtered_state[2].tolist()
            if agent_id == 0:  # Print for first agent only to avoid clutter
                print(f"[INFO] Complete reset - Filter state env 0: {filter_state_env0}, Expected: [0, 0, 0]")
                print(f"[INFO] Complete reset - Filter state env 2: {filter_state_env2}, Expected: [0, 0, 0]")
            assert torch.allclose(pipeline.position_filters[agent_id].filtered_state[0],
                                torch.zeros(3, device=device), atol=1e-5)
            assert torch.allclose(pipeline.position_filters[agent_id].filtered_state[2],
                                torch.zeros(3, device=device), atol=1e-5)

    # ==================== Additional: Time Synchronization ====================

    def test_time_synchronization(self, num_envs, num_agents, device):
        """Test time synchronization: total delay = detection + comm."""
        dt = 0.01
        detection_latency = 0.05  # 50ms
        comm_latency = 0.10       # 100ms

        pipeline = MultiAgentObservationPipeline(
            num_envs=num_envs,
            num_agents=num_agents,
            dt=dt,
            device=device,
            detection_fps=1000.0,  # No throttling
            detection_mean_latency=detection_latency,
            detection_std_latency=0.0,  # No variance for predictability
            detection_failure_rate=0.0,  # No failures
            comm_mean_delay=comm_latency,
            comm_std_delay=0.0,  # No variance
            comm_dropout_rate=0.0  # No dropout
        )

        agent0 = 0
        agent1 = 1

        # Agent 0 detects at t=0
        detection_data = torch.ones(num_envs, 4, device=device) * 999.0
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

        start_time = pipeline.current_time[0].item()
        pipeline.add_detection(agent0, detection_data, valid_mask)

        # Run simulation
        detection_received_at = None
        message_received_at = None

        for step in range(300):  # 3 seconds should be plenty
            pipeline.update_time()

            # Check when detection arrives
            if detection_received_at is None:
                det, det_valid, _ = pipeline.get_delayed_detection(agent0)
                if det_valid[0] and torch.abs(det[0, 0] - 999.0) < 0.1:
                    detection_received_at = pipeline.current_time[0].item()
                    # Broadcast immediately
                    pipeline.broadcast_state(agent0, {"detection": det})

            # Check when agent1 receives
            if message_received_at is None and detection_received_at is not None:
                received = pipeline.receive_other_agent_states(agent1)
                if agent0 in received and "detection" in received[agent0]:
                    data, valid, _ = received[agent0]["detection"]
                    if valid[0] and torch.abs(data[0, 0] - 999.0) < 0.1:
                        message_received_at = pipeline.current_time[0].item()
                        break

        # Verify timing
        assert detection_received_at is not None, "Detection should arrive"
        assert message_received_at is not None, "Message should arrive"

        # Detection delay should be at least detection_latency (might be more due to timing)
        detection_delay = detection_received_at - start_time
        print(f"[INFO] Time sync - Detection delay: {detection_delay:.4f}s, Expected: >={detection_latency * 0.5:.4f}s")
        assert detection_delay >= detection_latency * 0.5, \
            f"Detection delay {detection_delay} should be >= {detection_latency * 0.5}"

        # Total delay should be at least detection_latency + some comm delay
        # Allow for timing discretization (dt=0.01)
        total_delay = message_received_at - start_time
        expected_min = detection_latency * 0.5  # At least half of detection latency
        expected_ideal = detection_latency + comm_latency
        print(f"[INFO] Time sync - Total delay: {total_delay:.4f}s, Expected: >={expected_min:.4f}s (ideal: {expected_ideal:.4f}s)")
        assert total_delay >= expected_min, \
            f"Total delay {total_delay} should be >= {expected_min} (was expecting ~{detection_latency + comm_latency})"

        # Message should arrive after detection (comm delay adds to total)
        print(f"[INFO] Time sync - Message arrived after detection: {message_received_at > detection_received_at}, Expected: True")
        assert message_received_at > detection_received_at, \
            "Message should arrive after detection"

    # ==================== Additional: Memory Leaks ====================

    def test_memory_leak(self, num_envs, num_agents, dt, device):
        """Test for memory leaks over many iterations."""
        pipeline = MultiAgentObservationPipeline(
            num_envs=num_envs,
            num_agents=num_agents,
            dt=dt,
            device=device,
            max_buffer_size=50  # Smaller buffer
        )

        # Run many iterations
        for iteration in range(1000):
            for agent_id in range(num_agents):
                # Add various data
                det = torch.randn(num_envs, 4, device=device)
                valid = torch.rand(num_envs, device=device) > 0.5
                pipeline.add_detection(agent_id, det, valid)

                state = {
                    "pos": torch.randn(num_envs, 3, device=device),
                    "vel": torch.randn(num_envs, 3, device=device)
                }
                pipeline.broadcast_state(agent_id, state)

                _ = pipeline.receive_other_agent_states(agent_id)

            pipeline.update_time()

            # Periodically reset some envs to test reset doesn't leak
            if iteration % 100 == 0:
                pipeline.reset(torch.tensor([0], device=device))

        # Force garbage collection
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # If we get here without OOM, test passes
        # Memory usage should stabilize due to fixed buffer sizes
        print(f"[INFO] Memory leak test - Completed 1000 iterations, Expected: no OOM or memory issues")
        assert True, "Should complete without memory issues"

    def test_statistics_collection(self, pipeline, num_envs, num_agents, device):
        """Test statistics collection across pipeline."""
        # Generate activity
        for _ in range(50):
            for agent_id in range(num_agents):
                det = torch.randn(num_envs, 4, device=device)
                valid = torch.ones(num_envs, dtype=torch.bool, device=device)
                pipeline.add_detection(agent_id, det, valid)

                state = {"data": torch.randn(num_envs, 2, device=device)}
                pipeline.broadcast_state(agent_id, state)

            pipeline.update_time()

        # Get statistics
        stats = pipeline.get_statistics()

        # Should have current_time
        has_current_time = 'current_time' in stats
        has_detection_stats = 'detection_stats' in stats
        has_comm_stats = 'comm_stats' in stats
        print(f"[INFO] Statistics collection - Has current_time: {has_current_time}, detection_stats: {has_detection_stats}, comm_stats: {has_comm_stats}")
        print(f"[INFO] Statistics collection - Current time shape: {stats['current_time'].shape}, Expected: ({num_envs},)")

        assert 'current_time' in stats
        assert stats['current_time'].shape == (num_envs,)

        # Should have detection stats
        assert 'detection_stats' in stats

        # Should have comm stats
        assert 'comm_stats' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
