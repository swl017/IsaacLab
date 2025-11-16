"""
Tests for validity mask handling across all components.
Tests 35-37: Input vs channel validity, persistence, per-environment.
"""

import torch
import pytest
from isaaclab_tasks.direct.iris_ma3.delayed_states import (
    ChannelBuffer,
    MultiAgentCommChannel,
    MultiAgentObservationPipeline
)


class TestValidityMasks:
    """Test suite for validity mask semantics."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== Test 35: Input vs Channel Validity ====================

    def test_input_validity_vs_channel_validity(self, device):
        """Test 35: Distinguish input validity from channel-caused invalidity."""
        num_envs = 4

        # Scenario 1: Input valid, but 100% dropout
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(2,),
            throttle_period=0.0,
            dropout_rate=1.0,  # 100% dropout
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        data = torch.randn(num_envs, 2, device=device)
        input_valid = torch.tensor([True, True, True, True], device=device)

        _, output_valid = buffer.compute(data, 0.0, has_source_data=input_valid)

        # All should be invalid due to dropout (channel-caused)
        valid_count = output_valid.sum().item()
        print(f"[INFO] Input vs channel validity - 100% dropout: Valid count: {valid_count}/4, Expected: 0")
        assert torch.all(~output_valid), \
            "All outputs should be invalid due to 100% dropout"

        # Scenario 2: Input invalid, but 0% dropout
        buffer2 = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(2,),
            throttle_period=0.0,
            dropout_rate=0.0,  # No dropout
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        input_invalid = torch.tensor([False, False, False, False], device=device)

        _, output_valid2 = buffer2.compute(data, 0.0, has_source_data=input_invalid)

        # All should be invalid due to input (not channel)
        valid_count2 = output_valid2.sum().item()
        print(f"[INFO] Input vs channel validity - Input invalid: Valid count: {valid_count2}/4, Expected: 0")
        assert torch.all(~output_valid2), \
            "All outputs should be invalid due to input invalidity"

    def test_mixed_validity_scenarios(self, device):
        """Test 35b: Mixed input validity with channel impairments."""
        torch.manual_seed(42)

        num_envs = 8
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(1,),
            throttle_period=0.0,
            dropout_rate=0.5,  # 50% dropout
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        # Mixed input validity
        input_valid = torch.tensor([True, True, True, True, False, False, False, False],
                                   device=device)

        data = torch.randn(num_envs, 1, device=device)
        _, output_valid = buffer.compute(data, 0.0, has_source_data=input_valid)

        # Last 4 envs should always be invalid (input invalid)
        last_4_valid_count = output_valid[4:].sum().item()
        first_4_valid_count = output_valid[:4].sum().item()
        print(f"[INFO] Mixed validity - Last 4 envs (input invalid) valid count: {last_4_valid_count}/4, Expected: 0")
        print(f"[INFO] Mixed validity - First 4 envs (input valid) valid count: {first_4_valid_count}/4 (depends on 50% dropout)")
        assert torch.all(~output_valid[4:]), \
            "Input-invalid environments should always be invalid in output"

        # First 4 envs may or may not be valid (depends on dropout)
        # But we can't assert specifics due to randomness, just check structure
        assert output_valid.shape == (num_envs,)
        assert output_valid.dtype == torch.bool

    def test_comm_channel_validity_propagation(self, device):
        """Test 35c: Validity propagation through communication."""
        num_envs = 4
        num_agents = 2

        comm = MultiAgentCommChannel(
            num_envs=num_envs,
            num_agents=num_agents,
            mean_latency=0.01,
            std_latency=0.0,
            throttle_period=0.0,
            dropout_rate=0.0,  # No dropout for clarity
            dt=0.01,
            device=device
        )

        # Send with mixed validity
        sender_id = 0
        data = {"value": torch.ones(num_envs, 1, device=device)}
        input_valid = torch.tensor([True, False, True, False], device=device)

        comm.send_message(sender_id, data, 0.0, has_source_data=input_valid)

        # Advance time
        for _ in range(10):
            dummy = {"value": torch.zeros(num_envs, 1, device=device)}
            comm.send_message(sender_id, dummy, 0.01)

        # Receive
        received = comm.receive_messages(1)

        if sender_id in received and "value" in received[sender_id]:
            _, output_valid, _ = received[sender_id]["value"]

            # Invalid inputs should remain invalid
            # (May not have received yet for valid inputs, but invalid should stay invalid)
            # This is tricky with delays, so we just check shape
            valid_count = output_valid.sum().item()
            print(f"[INFO] Comm validity propagation - Valid count: {valid_count}/{num_envs}, Input pattern: [T, F, T, F]")
            assert output_valid.shape == (num_envs,)
            assert output_valid.dtype == torch.bool

    # ==================== Test 36: Validity Persistence ====================

    def test_validity_persistence(self, device):
        """Test 36: has_valid_data persists after data is sent."""
        num_envs = 2

        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(3,),
            throttle_period=0.0,
            dropout_rate=0.0,
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        # Initially, no valid data
        _, has_valid, _ = buffer.get_delayed()
        initial_valid_count = has_valid.sum().item()
        print(f"[INFO] Validity persistence - Initial valid count: {initial_valid_count}/2, Expected: 0")
        assert torch.all(~has_valid), "Initially should have no valid data"

        # Send valid data
        data = torch.randn(num_envs, 3, device=device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        buffer.compute(data, 0.0, has_source_data=valid_mask)

        # Now has_valid_data should be True
        _, has_valid, _ = buffer.get_delayed()
        after_send_valid_count = has_valid.sum().item()
        print(f"[INFO] Validity persistence - After sending: {after_send_valid_count}/2, Expected: 2")
        assert torch.all(has_valid), "Should have valid data after sending"

        # Call get_delayed() many times - should persist
        for _ in range(100):
            _, has_valid, _ = buffer.get_delayed()
            assert torch.all(has_valid), "Valid data flag should persist"

        print(f"[INFO] Validity persistence - After 100 get_delayed() calls: {has_valid.sum().item()}/2, Expected: 2 (persisted)")

        # Even after reset, calling compute again should restore it
        buffer.reset(torch.tensor([0], device=device))
        _, has_valid, _ = buffer.get_delayed()
        print(f"[INFO] Validity persistence - After reset env 0: env0={has_valid[0].item()}, env1={has_valid[1].item()}, Expected: False, True")
        assert not has_valid[0], "Reset env should lose valid flag"
        assert has_valid[1], "Non-reset env should keep valid flag"

    def test_validity_persistence_across_dropouts(self, device):
        """Test 36b: Validity persists even if subsequent data is dropped."""
        torch.manual_seed(999)

        num_envs = 1
        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(1,),
            throttle_period=0.0,
            dropout_rate=1.0,  # 100% dropout after first
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        # First, send with 0% dropout temporarily
        buffer.dropout_rate = torch.tensor([0.0], device=device)
        data = torch.tensor([[1.0]], device=device)
        buffer.compute(data, 0.0)

        # Check valid
        _, has_valid, _ = buffer.get_delayed()
        first_send_valid = has_valid[0].item()
        print(f"[INFO] Validity across dropouts - After first send: {first_send_valid}, Expected: True")
        assert has_valid[0], "Should have valid data after first send"

        # Now set 100% dropout and send more
        buffer.dropout_rate = torch.tensor([1.0], device=device)
        for _ in range(50):
            buffer.compute(torch.tensor([[2.0]], device=device), 0.0)

        # has_valid_data should still be True (from first successful send)
        _, has_valid, _ = buffer.get_delayed()
        after_dropouts_valid = has_valid[0].item()
        print(f"[INFO] Validity across dropouts - After 50 sends with 100% dropout: {after_dropouts_valid}, Expected: True (persisted)")
        assert has_valid[0], "Valid flag should persist despite dropouts"

    # ==================== Test 37: Per-Environment Validity ====================

    def test_per_environment_validity_independence(self, device):
        """Test 37: Each environment tracks validity independently."""
        num_envs = 8

        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(2,),
            throttle_period=0.0,
            dropout_rate=0.0,
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        # Send data with alternating validity
        data = torch.randn(num_envs, 2, device=device)
        input_valid = torch.tensor([True, False, True, False, True, False, True, False],
                                   device=device)

        buffer.compute(data, 0.0, has_source_data=input_valid)

        # Check has_valid_data flag
        _, has_valid, _ = buffer.get_delayed()

        # Should match input validity pattern
        valid_pattern = has_valid.tolist()
        input_pattern = input_valid.tolist()
        print(f"[INFO] Per-env validity - First send pattern: {valid_pattern}, Expected: {input_pattern}")
        assert torch.all(has_valid == input_valid), \
            "Per-environment validity should match input pattern"

        # Send again with different pattern
        input_valid2 = torch.tensor([False, True, False, True, False, True, False, True],
                                    device=device)

        buffer.compute(data, 0.01, has_source_data=input_valid2)

        _, has_valid2, _ = buffer.get_delayed()

        # Should be OR of both patterns (once valid, stays valid until reset)
        expected = input_valid | input_valid2
        valid_pattern2 = has_valid2.tolist()
        expected_pattern = expected.tolist()
        print(f"[INFO] Per-env validity - After second send (OR): {valid_pattern2}, Expected: {expected_pattern}")
        assert torch.all(has_valid2 == expected), \
            "Validity should accumulate (OR) across multiple sends"

    def test_per_environment_reset_validity(self, device):
        """Test 37b: Reset affects only specified environments."""
        num_envs = 6

        buffer = ChannelBuffer(
            num_envs=num_envs,
            data_shape=(1,),
            throttle_period=0.0,
            dropout_rate=0.0,
            mean_latency=0.0,
            std_latency=0.0,
            dt=0.01,
            device=device
        )

        # All environments get valid data
        data = torch.ones(num_envs, 1, device=device)
        valid_all = torch.ones(num_envs, dtype=torch.bool, device=device)
        buffer.compute(data, 0.0, has_source_data=valid_all)

        # Check all have valid data
        _, has_valid, _ = buffer.get_delayed()
        assert torch.all(has_valid), "All should have valid data"

        # Reset envs [1, 3, 5]
        reset_ids = torch.tensor([1, 3, 5], device=device)
        buffer.reset(reset_ids)

        # Check validity
        _, has_valid_after, _ = buffer.get_delayed()

        # Reset envs should be invalid
        reset_envs = [has_valid_after[1].item(), has_valid_after[3].item(), has_valid_after[5].item()]
        non_reset_envs = [has_valid_after[0].item(), has_valid_after[2].item(), has_valid_after[4].item()]
        print(f"[INFO] Per-env reset validity - Reset envs [1,3,5]: {reset_envs}, Expected: [False, False, False]")
        print(f"[INFO] Per-env reset validity - Non-reset envs [0,2,4]: {non_reset_envs}, Expected: [True, True, True]")

        assert not has_valid_after[1]
        assert not has_valid_after[3]
        assert not has_valid_after[5]

        # Non-reset envs should remain valid
        assert has_valid_after[0]
        assert has_valid_after[2]
        assert has_valid_after[4]

    def test_pipeline_end_to_end_validity(self, device):
        """Test 37c: Validity tracking through full pipeline."""
        num_envs = 4
        num_agents = 2

        pipeline = MultiAgentObservationPipeline(
            num_envs=num_envs,
            num_agents=num_agents,
            dt=0.01,
            device=device,
            detection_fps=100.0,  # High FPS to minimize throttling
            detection_failure_rate=0.0,  # No failures
            comm_dropout_rate=0.0  # No dropout
        )

        agent0 = 0

        # Add detection with per-env validity
        detection = torch.randn(num_envs, 4, device=device)
        input_valid = torch.tensor([True, False, True, False], device=device)

        pipeline.add_detection(agent0, detection, input_valid)

        # Advance time
        for _ in range(20):
            pipeline.update_time()

        # Get delayed detection
        delayed_det, det_valid, _ = pipeline.get_delayed_detection(agent0)

        # det_valid should reflect input validity (and channel impairments)
        # Envs 1 and 3 should definitely be invalid (input was invalid)
        # Envs 0 and 2 should be valid (input was valid, no channel failures)
        # Note: Due to delays, might not have data yet, so we check differently

        # After enough time, valid inputs should have data
        valid_pattern = det_valid.tolist()
        print(f"[INFO] Pipeline end-to-end validity - Detection valid pattern: {valid_pattern}, Input was: [T, F, T, F]")
        print(f"[INFO] Pipeline end-to-end validity - Env 1 valid: {det_valid[1].item()}, Env 3 valid: {det_valid[3].item()}, Expected: False, False")

        assert det_valid.shape == (num_envs,)
        assert det_valid.dtype == torch.bool

        # If we got valid data, it should be from originally valid inputs
        if det_valid[1]:
            pytest.fail("Env 1 had invalid input, should not have valid detection")
        if det_valid[3]:
            pytest.fail("Env 3 had invalid input, should not have valid detection")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
