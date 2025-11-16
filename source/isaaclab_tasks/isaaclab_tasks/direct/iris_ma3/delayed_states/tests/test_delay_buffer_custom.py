"""
Tests for DelayBufferCustom class.
Tests 7-9: Buffer corruption prevention, sample-and-hold, variable delays.
"""

import torch
import pytest
from isaaclab_tasks.direct.iris_ma3.delayed_states import DelayBufferCustom


class TestDelayBufferCustom:
    """Test suite for DelayBufferCustom."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def buffer(self, device):
        """Create a DelayBufferCustom instance."""
        return DelayBufferCustom(
            history_length=10,
            batch_size=4,
            device=device
        )

    def test_buffer_corruption_prevention(self, buffer, device):
        """Test 7: Verify get_delayed() doesn't append data."""
        # Append initial data
        data1 = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
        buffer.append(data1)

        # Get buffer state
        initial_buffer_state = buffer._circular_buffer._num_pushes.clone()

        # Call get_delayed() multiple times
        for _ in range(10):
            result, _ = buffer.get_delayed()

        # Verify buffer wasn't modified
        final_buffer_state = buffer._circular_buffer._num_pushes
        assert torch.all(initial_buffer_state == final_buffer_state), \
            "get_delayed() should not modify buffer state"

        # Verify result is consistent
        result1, _ = buffer.get_delayed()
        result2, _ = buffer.get_delayed()
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        print(f"[INFO] Buffer corruption test - Multiple get_delayed() calls max diff: {max_diff:.2e} (should be 0)")

        assert torch.allclose(result1, result2), \
            "Multiple get_delayed() calls should return same data"

    def test_append_only_writes(self, buffer, device):
        """Test 7b: Verify append() only writes without returning."""
        data = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)

        # append() returns None, not data
        result = buffer.append(data)
        print(f"[INFO] Append return value: {result}, Expected: None")
        assert result is None, "append() should not return data"

        # Data should be in buffer
        buffer.set_time_lag(0)
        retrieved, _ = buffer.get_delayed()
        max_diff = torch.max(torch.abs(retrieved - data)).item()
        print(f"[INFO] Append-retrieve roundtrip max diff: {max_diff:.2e} (should be 0)")
        assert torch.allclose(retrieved, data), "Data should be retrievable after append"

    def test_sample_and_hold_behavior(self, buffer, device):
        """Test 8: Sample-and-hold with sparse data."""
        buffer.set_time_lag(0)  # No delay

        # Append data at t=0, t=1, t=5
        times = [0, 1, 5]
        data_values = [10.0, 20.0, 50.0]

        for t, val in zip(times, data_values):
            data = torch.full((4, 1), val, device=device)
            buffer.append(data)

        # At this point, we've appended 3 times, so buffer has data at indices 0, 1, 2
        # With zero delay, get_delayed() returns most recent (index 2, value 50.0)
        result, _ = buffer.get_delayed()
        expected = torch.full((4, 1), 50.0, device=device)
        actual_value = result[0, 0].item()
        expected_value = expected[0, 0].item()
        print(f"[INFO] Sample-and-hold - Actual: {actual_value}, Expected: {expected_value} (last appended)")
        assert torch.allclose(result, expected), \
            "Should return last appended value"

    def test_variable_delay_per_environment(self, device):
        """Test 9: Different delays per environment."""
        buffer = DelayBufferCustom(
            history_length=10,
            batch_size=4,
            device=device
        )

        # Append sequence of data
        for i in range(6):
            data = torch.full((4, 1), float(i), device=device)
            buffer.append(data)

        # Set different delays per environment: [0, 1, 2, 5]
        delays = torch.tensor([0, 1, 2, 5], device=device)
        buffer.set_time_lag(delays)

        # Get delayed data
        result, sufficient_history = buffer.get_delayed()

        # Check sufficient history for each environment
        # We have 6 items, so:
        # env 0: delay=0 needs 1 item -> sufficient
        # env 1: delay=1 needs 2 items -> sufficient
        # env 2: delay=2 needs 3 items -> sufficient
        # env 3: delay=5 needs 6 items -> sufficient
        print(f"[INFO] Variable delays - Sufficient history: {sufficient_history.tolist()}, Expected: [True, True, True, True]")
        assert torch.all(sufficient_history), "All envs should have sufficient history"

        # Expected: most recent is 5, so:
        # env 0: delay=0 -> value=5
        # env 1: delay=1 -> value=4
        # env 2: delay=2 -> value=3
        # env 3: delay=5 -> value=0
        expected = torch.tensor([[5.0], [4.0], [3.0], [0.0]], device=device)
        result_list = result.squeeze().tolist()
        expected_list = expected.squeeze().tolist()
        print(f"[INFO] Variable delays - Actual: {result_list}, Expected: {expected_list}")

        assert torch.allclose(result, expected), \
            f"Variable delays failed. Expected {expected}, got {result}"

    def test_delay_clamping_to_history(self, device):
        """Test 9b: Delay larger than available history."""
        buffer = DelayBufferCustom(
            history_length=10,
            batch_size=2,
            device=device
        )

        # Append only 3 timesteps
        for i in range(3):
            data = torch.full((2, 1), float(i), device=device)
            buffer.append(data)

        # Request delay of 10 (larger than 3 appends)
        buffer.set_time_lag(10)

        # Should return data but with sufficient_history=False since we need 11 items but only have 3
        result, sufficient_history = buffer.get_delayed()
        expected = torch.full((2, 1), 0.0, device=device)

        result_value = result[0, 0].item()
        expected_value = expected[0, 0].item()
        print(f"[INFO] Delay clamping - Actual: {result_value}, Expected: {expected_value} (oldest data)")
        print(f"[INFO] Delay clamping - Sufficient history: {sufficient_history.tolist()}, Expected: [False, False]")

        assert torch.allclose(result, expected), \
            "Should return oldest available data (clamped)"
        assert not torch.any(sufficient_history), \
            "Should indicate insufficient history when delay > buffer contents"

    def test_reset_functionality(self, buffer, device):
        """Test buffer reset."""
        # Fill buffer
        for i in range(5):
            data = torch.full((4, 1), float(i), device=device)
            buffer.append(data)

        # Reset specific environments
        buffer.reset([0, 2])

        # Check that those environments were reset
        # After reset, num_pushes should be 0 for those envs
        pushes = buffer._circular_buffer._num_pushes
        pushes_list = pushes.tolist()
        print(f"[INFO] Reset test - Num pushes after reset: {pushes_list}, Expected: [0, >0, 0, >0]")
        assert pushes[0] == 0, "Env 0 should be reset"
        assert pushes[2] == 0, "Env 2 should be reset"
        assert pushes[1] > 0, "Env 1 should not be reset"
        assert pushes[3] > 0, "Env 3 should not be reset"

    def test_determinism(self, device):
        """Test deterministic behavior with same operations."""
        # Create two identical buffers
        buffer1 = DelayBufferCustom(history_length=10, batch_size=4, device=device)
        buffer2 = DelayBufferCustom(history_length=10, batch_size=4, device=device)

        # Same operations
        for i in range(5):
            data = torch.full((4, 2), float(i), device=device)
            buffer1.append(data)
            buffer2.append(data)

        buffer1.set_time_lag(2)
        buffer2.set_time_lag(2)

        # Results should be identical
        result1, sufficient1 = buffer1.get_delayed()
        result2, sufficient2 = buffer2.get_delayed()

        max_diff = torch.max(torch.abs(result1 - result2)).item()
        sufficient_match = torch.all(sufficient1 == sufficient2).item()
        print(f"[INFO] Determinism test - Max diff between buffers: {max_diff:.2e} (should be 0)")
        print(f"[INFO] Determinism test - Sufficient history flags match: {sufficient_match}, Expected: True")

        assert torch.allclose(result1, result2), \
            "Deterministic operations should produce identical results"
        assert torch.all(sufficient1 == sufficient2), \
            "Sufficient history flags should be identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
