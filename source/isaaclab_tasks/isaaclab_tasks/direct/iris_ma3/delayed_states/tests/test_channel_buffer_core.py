"""
Tests for ChannelBuffer class.
Tests 1-4, 10-17: Core impairments (throttle, dropout, latency) and edge cases.
"""

import torch
import pytest
from isaaclab_tasks.direct.iris_ma3.delayed_states import ChannelBuffer


class TestChannelBufferCore:
    """Test suite for ChannelBuffer core functionality."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def num_envs(self):
        """Number of test environments."""
        return 8

    @pytest.fixture
    def data_shape(self):
        """Data shape (excluding batch dimension)."""
        return (3,)

    @pytest.fixture
    def dt(self):
        """Simulation timestep."""
        return 0.01

    # ==================== Tests 1-4: Core Requirements ====================

    def test_throttle_only(self, num_envs, data_shape, dt, device):
        """Test 1: Throttle (FPS limiting) only."""
        throttle_period = 0.2  # 5 Hz
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=throttle_period,  # 5 Hz
            dropout_rate=0.0,
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        current_time = 0.0
        throttle_passes = 0
        total_attempts = 100

        for i in range(total_attempts):
            data = torch.randn(num_envs, *data_shape, device=device)
            _, valid_mask = buffer.compute(data, current_time)

            if valid_mask.any():
                throttle_passes += valid_mask.sum().item()

            current_time += dt

        # With 10 Hz throttle and dt=0.01, we should get roughly 10 passes
        # (100 attempts * 0.01s = 1.0s total time, at 10 Hz = ~10 passes per env)
        # 5 Hz means ~5 passes per env
        expected_passes = int(total_attempts * dt / throttle_period) * num_envs
        tolerance = num_envs * 2  # Allow some tolerance

        print(f"[INFO] Throttle passes: {throttle_passes}, Expected: {expected_passes} ± {tolerance}")

        assert abs(throttle_passes - expected_passes) <= tolerance, \
            f"Expected ~{expected_passes} throttle passes, got {throttle_passes}"

    def test_dropout_only(self, num_envs, data_shape, dt, device):
        """Test 2: Dropout (packet loss) only."""
        torch.manual_seed(42)  # For reproducibility

        # 30% dropout rate
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,  # No throttling
            dropout_rate=0.3,
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        total_attempts = 1000
        success_count = 0

        for i in range(total_attempts):
            data = torch.randn(num_envs, *data_shape, device=device)
            _, valid_mask = buffer.compute(data, float(i) * dt)
            success_count += valid_mask.sum().item()

        # Expected success rate: 70% (1 - dropout_rate)
        expected_success = int(total_attempts * num_envs * 0.7)
        # Allow 5% tolerance
        tolerance = int(total_attempts * num_envs * 0.05)

        print(f"[INFO] Dropout successes: {success_count}, Expected: {expected_success} ± {tolerance}")

        assert abs(success_count - expected_success) <= tolerance, \
            f"Expected ~{expected_success} successes, got {success_count}"

    def test_latency_only(self, num_envs, data_shape, dt, device):
        """Test 3: Latency (variable delays) only."""
        # Mean latency of 5 timesteps (0.05s)
        mean_latency_sec = 0.05
        std_latency_sec = 0.01

        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,
            dropout_rate=0.0,
            mean_latency=mean_latency_sec,
            std_latency=std_latency_sec,
            dt=dt,
            device=device,
            max_history=20
        )

        # Send unique identifiable data
        marker_value = 42.0
        data = torch.full((num_envs, *data_shape), marker_value, device=device)
        buffer.compute(data, 0.0)

        # Advance time and check delayed output
        # After mean_latency, we should start seeing the marker value
        timesteps_to_check = int(mean_latency_sec / dt) + 10

        marker_appeared = False
        time_marker_added = 0.0
        for t in range(1, timesteps_to_check):
            dummy_data = torch.zeros(num_envs, *data_shape, device=device)
            delayed_data, _ = buffer.compute(dummy_data, float(t) * dt)

            # Check if marker value appeared
            if torch.any(torch.abs(delayed_data - marker_value) < 1e-5):
                marker_appeared = True
                time_marker_added = float(t) * dt
                break


        print(f"[INFO] Marker appeared: {marker_appeared} after {time_marker_added:.3f} seconds")
        assert marker_appeared, "Marker value should appear after latency period"

    def test_combined_impairments(self, num_envs, data_shape, dt, device):
        """Test 4: Throttle + Dropout + Latency combined."""
        torch.manual_seed(123)

        throttle_period=0.05  # 20 Hz
        dropout_rate=0.2      # 20% dropout
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=throttle_period,  # 20 Hz
            dropout_rate=dropout_rate,      # 20% dropout
            mean_latency=0.03,     # 30ms mean
            std_latency=0.01,      # 10ms std
            dt=dt,
            device=device,
            max_history=50
        )

        total_attempts = 500
        for i in range(total_attempts):
            data = torch.randn(num_envs, *data_shape, device=device)
            delayed_data, valid_mask = buffer.compute(data, float(i) * dt)

        # Get statistics
        stats = buffer.get_statistics()

        # Verify all statistics are present
        assert 'total_attempts' in stats
        assert 'throttled_count' in stats
        assert 'dropout_count' in stats
        assert 'success_count' in stats

        # Total attempts should match
        assert torch.all(stats['total_attempts'] == total_attempts)

        # Success count should be less than total (some throttled + some dropped)
        assert torch.all(stats['success_count'] < stats['total_attempts'])

        print(f"[INFO] Combined impairments - Throttled: {stats['throttled_count'][0].item()}, "
              f"Dropped: {stats['dropout_count'][0].item()}, "
              f"Success: {stats['success_count'][0].item()}/{total_attempts}, "
              f"Success rate: {stats['success_rate'][0].item():.2%}"
              f"Expected (dt/throttle * (1 - dropout)): ~{(dt/throttle_period)*(1-dropout_rate):.2%}")

        # Success rate should be roughly: (1 / throttle_period) * (1 - dropout_rate)
        # But since throttle depends on timing, just check it's reasonable
        assert torch.all(stats['success_rate'] > 0.1)  # At least 10% success
        assert torch.all(stats['success_rate'] < 0.9)  # Less than 90% success

    # ==================== Tests 10-12: Edge Cases ====================

    def test_throttle_zero_period(self, num_envs, data_shape, dt, device):
        """Test 10: Zero throttle period (no limiting)."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,  # No throttling
            dropout_rate=0.0,
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        # All attempts should pass
        passes = 0
        for i in range(100):
            data = torch.randn(num_envs, *data_shape, device=device)
            _, valid_mask = buffer.compute(data, float(i) * dt)
            passes += valid_mask.sum().item()
            assert torch.all(valid_mask), "All data should pass with zero throttle period"

        print(f"[INFO] Zero throttle period - Passes: {passes}/{100 * num_envs} (100% expected)")

    def test_throttle_infinite_period(self, num_envs, data_shape, dt, device):
        """Test 10b: Very large throttle period (only first sample)."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=1000.0,  # Effectively infinite
            dropout_rate=0.0,
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        first_passed = False
        subsequent_passes = 0

        for i in range(100):
            data = torch.randn(num_envs, *data_shape, device=device)
            _, valid_mask = buffer.compute(data, float(i) * dt)

            if i == 0:
                first_passed = valid_mask.any()
            elif valid_mask.any():
                subsequent_passes += 1

        print(f"[INFO] Infinite throttle - First passed: {first_passed}, Subsequent passes: {subsequent_passes}/99 (0 expected)")

        assert first_passed, "First sample should always pass"
        assert subsequent_passes == 0, "No subsequent samples should pass with infinite throttle"

    def test_dropout_zero_rate(self, num_envs, data_shape, dt, device):
        """Test 11: 0% dropout (all pass)."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,
            dropout_rate=0.0,  # No dropout
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        for i in range(100):
            data = torch.randn(num_envs, *data_shape, device=device)
            _, valid_mask = buffer.compute(data, float(i) * dt)
            assert torch.all(valid_mask), "All data should pass with zero dropout"

    def test_dropout_full_rate(self, num_envs, data_shape, dt, device):
        """Test 11b: 100% dropout (all dropped)."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,
            dropout_rate=1.0,  # 100% dropout
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        dropped = 0
        for i in range(100):
            data = torch.randn(num_envs, *data_shape, device=device)
            _, valid_mask = buffer.compute(data, float(i) * dt)
            dropped += (~valid_mask).sum().item()
            assert torch.all(~valid_mask), "All data should be dropped with 100% dropout"

        print(f"[INFO] Full dropout - Dropped: {dropped}/{100 * num_envs} (100% expected)")

    def test_latency_zero(self, num_envs, data_shape, dt, device):
        """Test 12: Zero latency (immediate delivery)."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,
            dropout_rate=0.0,
            mean_latency=0.0,  # Zero latency
            std_latency=0.0,
            dt=dt,
            device=device
        )

        # Send specific data - with zero latency, it should be available immediately
        test_data = torch.randn(num_envs, *data_shape, device=device)
        delayed_data, valid_mask = buffer.compute(test_data, 0.0)

        # With zero latency (delay=0), delayed_data should be test_data immediately
        # Note: After first append, delay buffer returns the appended data
        max_diff = torch.max(torch.abs(delayed_data - test_data)).item()
        print(f"[INFO] Zero latency - Max difference: {max_diff:.2e} (should be ~0)")

        assert torch.allclose(delayed_data, test_data, atol=1e-5), \
            f"Zero latency should deliver data immediately, expected {test_data[0]}, got {delayed_data[0]}"

    def test_latency_clamping(self, num_envs, data_shape, dt, device):
        """Test 12b: Latency clamped to max_history."""
        max_history = 10
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,
            dropout_rate=0.0,
            mean_latency=100.0,  # Way larger than max_history
            std_latency=0.0,
            dt=dt,
            device=device,
            max_history=max_history
        )

        # Should not crash and should clamp to max_history
        for i in range(20):
            data = torch.randn(num_envs, *data_shape, device=device)
            _, _ = buffer.compute(data, float(i) * dt)

    # ==================== Test 13: Impairment Order ====================

    def test_impairment_order(self, device):
        """Test 13: Verify correct order: Throttle → Dropout → Latency."""
        torch.manual_seed(999)

        num_envs = 100  # More envs for statistical significance
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(1,),
            throttle_period=0.1,  # 10 Hz
            dropout_rate=0.5,     # 50% dropout
            mean_latency=0.0,     # Zero for simplicity
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        current_time = 0.0
        total_attempts = 200

        for i in range(total_attempts):
            data = torch.randn(num_envs, 1, device=device)
            _, _ = buffer.compute(data, current_time)
            current_time += 0.01

        stats = buffer.get_statistics()

        # Verify: total = success + throttled + dropped
        # success_count: data that passed all stages
        # throttled_count: data that was blocked by throttle
        # dropout_count: data that passed throttle but was dropped

        total_attempts_per_env = stats['total_attempts'][0].item()
        throttled = stats['throttled_count'][0].item()
        dropped = stats['dropout_count'][0].item()
        success = stats['success_count'][0].item()

        # throttled + (dropped + success) should equal total_attempts
        # because dropout only applies to non-throttled data
        assert success + throttled + dropped == total_attempts_per_env, \
            f"Counts don't add up: {success} + {throttled} + {dropped} != {total_attempts_per_env}"

        # With 10 Hz and dt=0.01, we expect ~20 throttle passes out of 200
        # Of those ~20, half should be dropped (50% dropout rate)
        expected_throttle_passes = int(total_attempts * 0.01 / 0.1)
        expected_throttled = total_attempts - expected_throttle_passes
        tolerance = max(10, int(expected_throttled * 0.1))  # 10% tolerance or minimum 10
        assert abs(throttled - expected_throttled) <= tolerance, \
            f"Throttle count incorrect: expected ~{expected_throttled}, got {throttled}"

    # ==================== Test 14: Valid Mask Semantics ====================

    def test_valid_mask_semantics(self, num_envs, data_shape, dt, device):
        """Test 14: Input validity vs channel impairments."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,
            dropout_rate=0.0,  # No channel impairments
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        # Input mask: alternating valid/invalid
        input_valid = torch.tensor([True, False, True, False, True, False, True, False],
                                   device=device)

        data = torch.randn(num_envs, *data_shape, device=device)
        _, output_valid = buffer.compute(data, 0.0, has_source_data=input_valid)

        # Output should match input (no channel impairments)
        assert torch.all(output_valid == input_valid), \
            "Output validity should match input when no channel impairments"

    def test_input_validity_with_dropout(self, device):
        """Test 14b: Input validity interacts with dropout."""
        torch.manual_seed(777)

        buffer = ChannelBuffer(
            num_envs=4,
            data_shape=(1,),
            throttle_period=0.0,
            dropout_rate=1.0,  # 100% dropout
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        # Even with 100% dropout, invalid inputs shouldn't count as "dropped"
        input_valid = torch.tensor([True, False, True, False], device=device)

        for _ in range(10):
            data = torch.randn(4, 1, device=device)
            _, output_valid = buffer.compute(data, 0.0, has_source_data=input_valid)

            # Output should be all False (invalid inputs + dropout)
            assert torch.all(~output_valid), \
                "All outputs should be invalid (input invalid OR dropped)"

    # ==================== Test 15: Dynamic Parameter Updates ====================

    def test_dynamic_parameter_update(self, num_envs, data_shape, dt, device):
        """Test 15: Update parameters mid-run."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.1,  # Start with 10 Hz
            dropout_rate=0.1,     # 10% dropout
            mean_latency=0.05,
            std_latency=0.01,
            dt=dt,
            device=device
        )

        # Run 50 timesteps with initial parameters
        for i in range(50):
            data = torch.randn(num_envs, *data_shape, device=device)
            buffer.compute(data, float(i) * dt)

        stats_before = buffer.get_statistics()

        # Update parameters
        new_throttle = torch.full((num_envs,), 0.05, device=device)  # 20 Hz
        new_dropout = torch.full((num_envs,), 0.5, device=device)    # 50% dropout
        buffer.update_params(
            throttle_period=new_throttle,
            dropout_rate=new_dropout
        )

        # Run another 50 timesteps
        for i in range(50, 100):
            data = torch.randn(num_envs, *data_shape, device=device)
            buffer.compute(data, float(i) * dt)

        stats_after = buffer.get_statistics()

        # Total attempts should have increased
        assert torch.all(stats_after['total_attempts'] > stats_before['total_attempts'])

        # Success rate in second half should be lower (higher dropout)
        # This is approximate due to randomness
        assert torch.all(stats_after['success_rate'] < 0.6), \
            "Success rate should decrease with higher dropout"

    # ==================== Test 16: Statistics Accuracy ====================

    def test_statistics_accuracy(self, device):
        """Test 16: Verify statistics over many runs."""
        torch.manual_seed(42)

        num_envs = 10
        throttle_period = 0.05  # 20 Hz
        dropout_rate = 0.3      # 30%
        dt = 0.01
        total_runs = 1000

        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(2,),
            throttle_period=throttle_period,
            dropout_rate=dropout_rate,
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        current_time = 0.0
        for i in range(total_runs):
            data = torch.randn(num_envs, 2, device=device)
            buffer.compute(data, current_time)
            current_time += dt

        stats = buffer.get_statistics()

        # Verify statistics structure
        assert stats['total_attempts'][0] == total_runs
        assert torch.all(stats['throttled_count'] + stats['dropout_count'] +
                        stats['success_count'] == total_runs)

        # Expected success rate: ~(1 - throttle_rate) * (1 - dropout_rate)
        # With throttle_period=0.05 and dt=0.01, throttle_rate ≈ 1 - 0.01/0.05 = 0.8
        # So expected success ≈ 0.2 * 0.7 = 0.14
        expected_success_rate = (dt / throttle_period) * (1 - dropout_rate)

        # Allow 20% relative tolerance due to randomness
        tolerance = expected_success_rate * 0.2
        actual_success_rate = stats['success_rate'][0].item()

        print(f"[INFO] Statistics accuracy - Success rate: {actual_success_rate:.3f}, "
              f"Expected: {expected_success_rate:.3f} ± {tolerance:.3f}, "
              f"Throttled: {stats['throttled_count'][0].item()}, "
              f"Dropped: {stats['dropout_count'][0].item()}, "
              f"Success: {stats['success_count'][0].item()}")

        assert abs(actual_success_rate - expected_success_rate) <= tolerance, \
            f"Success rate {actual_success_rate} differs from expected {expected_success_rate}"

    # ==================== Test 17: Reset Behavior ====================

    def test_reset_specific_envs(self, num_envs, data_shape, dt, device):
        """Test 17: Reset specific environments."""
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.0,
            dropout_rate=0.0,
            mean_latency=0.0,
            std_latency=0.0,
            dt=dt,
            device=device
        )

        # Fill buffer with data
        for i in range(20):
            data = torch.randn(num_envs, *data_shape, device=device)
            buffer.compute(data, float(i) * dt)

        # Get statistics before reset
        stats_before = buffer.get_statistics()

        # Reset environments [0, 2, 4]
        env_ids = torch.tensor([0, 2, 4], device=device)
        buffer.reset(env_ids)

        # Get statistics after reset
        stats_after = buffer.get_statistics()

        # Reset environments should have zero statistics
        assert stats_after['total_attempts'][0] == 0
        assert stats_after['total_attempts'][2] == 0
        assert stats_after['total_attempts'][4] == 0

        # Non-reset environments should maintain statistics
        assert stats_after['total_attempts'][1] == stats_before['total_attempts'][1]
        assert stats_after['total_attempts'][3] == stats_before['total_attempts'][3]

        # has_valid_data should be False for reset envs immediately after reset
        # Check by looking at the internal state
        assert not buffer.has_valid_data[0], "Reset env 0 should have False has_valid_data"
        assert not buffer.has_valid_data[2], "Reset env 2 should have False has_valid_data"
        assert not buffer.has_valid_data[4], "Reset env 4 should have False has_valid_data"

        # Non-reset envs should still have valid data
        assert buffer.has_valid_data[1], "Non-reset env 1 should have True has_valid_data"
        assert buffer.has_valid_data[3], "Non-reset env 3 should have True has_valid_data"

    # ==================== Additional: Determinism Test ====================

    def test_determinism(self, num_envs, data_shape, dt, device):
        """Test deterministic behavior with same seed."""
        torch.manual_seed(12345)

        buffer1 = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.05,
            dropout_rate=0.3,
            mean_latency=0.02,
            std_latency=0.01,
            dt=dt,
            device=device
        )

        torch.manual_seed(12345)  # Reset seed

        buffer2 = ChannelBuffer(
            num_envs=num_envs,
            data_shape=data_shape,
            throttle_period=0.05,
            dropout_rate=0.3,
            mean_latency=0.02,
            std_latency=0.01,
            dt=dt,
            device=device
        )

        # Same operations with same seed
        torch.manual_seed(999)
        for i in range(50):
            data = torch.randn(num_envs, *data_shape)
            buffer1.compute(data.to(device), float(i) * dt)

        torch.manual_seed(999)  # Reset for second buffer
        for i in range(50):
            data = torch.randn(num_envs, *data_shape)
            buffer2.compute(data.to(device), float(i) * dt)

        # Statistics should match
        stats1 = buffer1.get_statistics()
        stats2 = buffer2.get_statistics()

        assert torch.all(stats1['success_count'] == stats2['success_count']), \
            "Deterministic execution should produce identical results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
