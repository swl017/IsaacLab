"""
Tests for FirstOrderLag and QuaternionFirstOrderLag filters.
Tests 24-28: Filter convergence, time constant updates, SLERP, reset.
"""

import torch
import pytest
import math
from isaaclab_tasks.direct.iris_ma3.delayed_states import FirstOrderLag, QuaternionFirstOrderLag


class TestFilters:
    """Test suite for lag filters."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def num_envs(self):
        """Number of environments."""
        return 4

    @pytest.fixture
    def dt(self):
        """Timestep."""
        return 0.01

    # ==================== Test 24: FirstOrderLag Convergence ====================

    def test_first_order_lag_convergence(self, num_envs, dt, device):
        """Test 24: Step response converges exponentially."""
        time_constant = 0.1
        state_dim = 3

        filter_obj = FirstOrderLag(
            num_envs=num_envs,
            state_dim=state_dim,
            time_constant=time_constant,
            dt=dt,
            device=device
        )

        # Step input: 0 → 1
        target = torch.ones(num_envs, state_dim, device=device)
        current_time = 0.0

        # Track convergence
        history = []
        for t in range(500):
            filtered = filter_obj.update(target, current_time)
            history.append(filtered[0, 0].item())  # Track first env, first dim
            current_time += dt

        # After 5 time constants, should be ~99.3% converged
        time_5tau = int(5 * time_constant / dt)
        final_value = history[time_5tau]

        print(f"[INFO] First-order lag convergence at 5τ - Actual: {final_value:.4f}, Expected: >0.99")
        assert final_value > 0.99, \
            f"Should converge to >99% after 5τ, got {final_value}"

        # Check exponential behavior: response should be roughly 1 - exp(-t/τ)
        # At t=τ, should be ~63.2% converged
        time_1tau = int(time_constant / dt)
        value_at_tau = history[time_1tau]
        expected_at_tau = 1 - math.exp(-1)  # ≈ 0.632

        print(f"[INFO] First-order lag convergence at τ - Actual: {value_at_tau:.4f}, Expected: {expected_at_tau:.4f} ± 0.1")
        assert abs(value_at_tau - expected_at_tau) < 0.1, \
            f"At t=τ, expected ~{expected_at_tau}, got {value_at_tau}"

    def test_first_order_lag_affects_convergence_rate(self, num_envs, dt, device):
        """Test 24b: Time constant affects convergence speed."""
        state_dim = 1

        # Fast filter (small τ)
        fast_filter = FirstOrderLag(
            num_envs=num_envs,
            state_dim=state_dim,
            time_constant=0.05,
            dt=dt,
            device=device
        )

        # Slow filter (large τ)
        slow_filter = FirstOrderLag(
            num_envs=num_envs,
            state_dim=state_dim,
            time_constant=0.5,
            dt=dt,
            device=device
        )

        target = torch.ones(num_envs, state_dim, device=device)
        current_time = 0.0

        # Update both for 100 steps
        for _ in range(100):
            fast_result = fast_filter.update(target, current_time)
            slow_result = slow_filter.update(target, current_time)
            current_time += dt

        # Fast filter should be closer to target
        fast_value = fast_result[0, 0].item()
        slow_value = slow_result[0, 0].item()
        print(f"[INFO] Convergence rate - Fast filter: {fast_value:.4f}, Slow filter: {slow_value:.4f} (fast > slow expected)")
        assert fast_result[0, 0] > slow_result[0, 0], \
            "Fast filter should converge faster than slow filter"

    # ==================== Test 25: Time Constant Update ====================

    def test_update_time_constant(self, num_envs, dt, device):
        """Test 25: Dynamically update time constant."""
        filter_obj = FirstOrderLag(
            num_envs=num_envs,
            state_dim=1,
            time_constant=0.1,
            dt=dt,
            device=device
        )

        # Converge partway
        target = torch.ones(num_envs, 1, device=device)
        for _ in range(50):
            filter_obj.update(target, 0.0)

        value_before = filter_obj.filtered_state[0, 0].item()

        # Update time constant to much larger value
        new_tau = torch.full((num_envs, 1), 1.0, device=device)
        filter_obj.update_time_constants(new_tau)

        # Continue updating
        for _ in range(50):
            filter_obj.update(target, 0.0)

        value_after = filter_obj.filtered_state[0, 0].item()

        # Should have slowed down convergence
        # (value_after should not be much larger than value_before)
        progress = value_after - value_before
        print(f"[INFO] Time constant update - Progress after update: {progress:.4f}, Expected: <0.3 (slowed convergence)")
        print(f"[INFO] Time constant update - Value before: {value_before:.4f}, Value after: {value_after:.4f}")
        assert progress < 0.3, \
            f"Large time constant should slow convergence, but got progress {progress}"

    # ==================== Test 26: Quaternion SLERP ====================

    def test_quaternion_slerp_interpolation(self, num_envs, dt, device):
        """Test 26: SLERP interpolation follows geodesic."""
        time_constant = 0.1

        quat_filter = QuaternionFirstOrderLag(
            num_envs=num_envs,
            time_constant=torch.full((num_envs,), time_constant, device=device),
            dt=dt,
            device=device
        )

        # Start at identity
        # Target: 180° rotation around Z axis
        # q = [cos(θ/2), 0, 0, sin(θ/2)] for rotation θ around Z
        theta = math.pi  # 180 degrees
        target_quat = torch.tensor([
            [math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)]
        ], device=device).repeat(num_envs, 1)

        # Normalize
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)

        current_time = 0.0

        # Interpolate
        for _ in range(200):
            result = quat_filter.update(target_quat, current_time)
            current_time += dt

            # Verify quaternion remains normalized
            norm = torch.norm(result, dim=-1)
            assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4), \
                "Quaternion should remain normalized during SLERP"

        # Final result should be close to target
        final_quat = quat_filter.filtered_quat[0]
        dot_product = torch.sum(final_quat * target_quat[0])

        # Dot product should be close to 1 (or -1, equivalent rotations)
        dot_value = abs(dot_product.item())
        print(f"[INFO] Quaternion SLERP - |Dot product|: {dot_value:.4f}, Expected: ~1.0 ± 0.1")
        assert abs(abs(dot_product.item()) - 1.0) < 0.1, \
            f"Should converge to target quaternion, dot={dot_product}"

    def test_quaternion_shortest_path(self, num_envs, dt, device):
        """Test 27: Antipodal handling (shortest path)."""
        time_constant = 0.05

        quat_filter = QuaternionFirstOrderLag(
            num_envs=num_envs,
            time_constant=torch.full((num_envs,), time_constant, device=device),
            dt=dt,
            device=device,
            initial_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
        )

        # Target is negation of current (antipodal)
        # These represent the same rotation, so shouldn't cause motion
        target_quat = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)

        # Update once
        result = quat_filter.update(target_quat, 0.0)

        # Should have flipped target internally to avoid long path
        # Result should still be close to identity (not much change)
        # The filter should recognize these are the same rotation
        dot_with_identity = torch.sum(result[0] * torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

        # Should remain close to identity
        dot_value = abs(dot_with_identity.item())
        print(f"[INFO] Quaternion shortest path - |Dot with identity|: {dot_value:.4f}, Expected: >0.9")
        assert abs(dot_with_identity.item()) > 0.9, \
            "Antipodal quaternions should be handled by shortest path"

    # ==================== Test 28: Filter Reset ====================

    def test_first_order_lag_reset(self, num_envs, dt, device):
        """Test 28: Reset specific environments."""
        filter_obj = FirstOrderLag(
            num_envs=num_envs,
            state_dim=2,
            time_constant=0.1,
            dt=dt,
            device=device
        )

        # Build up some state
        target = torch.ones(num_envs, 2, device=device)
        for _ in range(50):
            filter_obj.update(target, 0.0)

        # State should be non-zero
        assert torch.all(filter_obj.filtered_state > 0.5)

        # Reset environments [0, 2]
        env_ids = torch.tensor([0, 2], device=device)
        filter_obj.reset(env_ids)

        # Those environments should be zero
        reset_env0 = filter_obj.filtered_state[0].tolist()
        reset_env2 = filter_obj.filtered_state[2].tolist()
        retained_env1 = filter_obj.filtered_state[1].tolist()
        retained_env3 = filter_obj.filtered_state[3].tolist()
        print(f"[INFO] Filter reset - Reset envs [0,2]: {reset_env0}, {reset_env2}, Expected: [0, 0]")
        print(f"[INFO] Filter reset - Retained envs [1,3]: {retained_env1}, {retained_env3}, Expected: >0.5")

        assert torch.allclose(filter_obj.filtered_state[0], torch.zeros(2, device=device))
        assert torch.allclose(filter_obj.filtered_state[2], torch.zeros(2, device=device))

        # Others should retain state
        assert torch.all(filter_obj.filtered_state[1] > 0.5)
        assert torch.all(filter_obj.filtered_state[3] > 0.5)

    def test_quaternion_filter_reset(self, num_envs, dt, device):
        """Test 28b: Reset quaternion filter."""
        quat_filter = QuaternionFirstOrderLag(
            num_envs=num_envs,
            time_constant=torch.full((num_envs,), 0.1, device=device),
            dt=dt,
            device=device
        )

        # Build up some rotation
        target = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=device).repeat(num_envs, 1)
        for _ in range(50):
            quat_filter.update(target, 0.0)

        # Reset env 1
        quat_filter.reset(torch.tensor([1], device=device))

        # Should be back to identity
        expected_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        reset_quat = quat_filter.filtered_quat[1].tolist()
        retained_quat = quat_filter.filtered_quat[0].tolist()
        print(f"[INFO] Quaternion filter reset - Reset env 1: {reset_quat}, Expected: [1, 0, 0, 0]")
        print(f"[INFO] Quaternion filter reset - Retained env 0: {retained_quat}, Expected: not identity")

        assert torch.allclose(quat_filter.filtered_quat[1], expected_identity, atol=1e-5), \
            "Reset quaternion should return to identity"

        # Env 0 should retain its state
        assert not torch.allclose(quat_filter.filtered_quat[0], expected_identity, atol=1e-1), \
            "Non-reset quaternion should maintain state"

    def test_quaternion_filter_reset_with_custom_state(self, num_envs, dt, device):
        """Test 28c: Reset to custom quaternion."""
        quat_filter = QuaternionFirstOrderLag(
            num_envs=num_envs,
            time_constant=torch.full((num_envs,), 0.1, device=device),
            dt=dt,
            device=device
        )

        # Reset env 0 to specific quaternion
        custom_quat = torch.tensor([[0.707, 0.0, 0.707, 0.0]], device=device)
        quat_filter.reset(torch.tensor([0], device=device), quat=custom_quat)

        # Should match custom quaternion
        result_quat = quat_filter.filtered_quat[0].tolist()
        expected_quat = custom_quat[0].tolist()
        print(f"[INFO] Quaternion custom reset - Actual: {result_quat}, Expected: {expected_quat}")

        assert torch.allclose(quat_filter.filtered_quat[0], custom_quat[0], atol=1e-5), \
            "Should reset to custom quaternion"

    # ==================== Additional Tests ====================

    def test_first_order_lag_determinism(self, num_envs, dt, device):
        """Test deterministic filter behavior."""
        # Two identical filters
        filter1 = FirstOrderLag(num_envs, 3, 0.1, dt, device)
        filter2 = FirstOrderLag(num_envs, 3, 0.1, dt, device)

        # Same inputs
        torch.manual_seed(999)
        max_diffs = []
        for _ in range(100):
            input_data = torch.randn(num_envs, 3, device=device)
            result1 = filter1.update(input_data, 0.0)
            result2 = filter2.update(input_data, 0.0)
            max_diffs.append(torch.max(torch.abs(result1 - result2)).item())

            assert torch.allclose(result1, result2), \
                "Deterministic filters should produce identical results"

        max_diff_overall = max(max_diffs)
        print(f"[INFO] Filter determinism - Max diff over 100 updates: {max_diff_overall:.2e} (should be 0)")

    def test_quaternion_normalization_maintained(self, num_envs, dt, device):
        """Test that quaternions remain normalized through many updates."""
        quat_filter = QuaternionFirstOrderLag(
            num_envs=num_envs,
            time_constant=torch.full((num_envs,), 0.05, device=device),
            dt=dt,
            device=device
        )

        # Random target quaternions
        torch.manual_seed(123)
        norm_errors = []
        for _ in range(500):
            # Generate random unit quaternions
            target = torch.randn(num_envs, 4, device=device)
            target = target / torch.norm(target, dim=-1, keepdim=True)

            result = quat_filter.update(target, 0.0)

            # Check normalization
            norms = torch.norm(result, dim=-1)
            norm_errors.append(torch.max(torch.abs(norms - 1.0)).item())
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
                f"Quaternions must remain normalized, got norms {norms}"

        max_norm_error = max(norm_errors)
        print(f"[INFO] Quaternion normalization - Max norm error over 500 updates: {max_norm_error:.2e} (should be <1e-4)")

    def test_slerp_vs_lerp_fallback(self, device):
        """Test 46: SLERP vs LERP fallback for small angles."""
        num_envs = 1
        dt = 0.01

        quat_filter = QuaternionFirstOrderLag(
            num_envs=num_envs,
            time_constant=torch.tensor([0.5], device=device),  # Slow for small increments
            dt=dt,
            device=device,
            initial_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        )

        # Target very close to identity (small angle)
        small_angle = 0.001  # radians
        target = torch.tensor([[
            math.cos(small_angle / 2),
            0.0,
            0.0,
            math.sin(small_angle / 2)
        ]], device=device)

        # One update
        result = quat_filter.update(target, 0.0)

        # Should use LERP fallback for small angles (dot > 0.9995)
        # Result should still be valid and normalized
        norm = torch.norm(result)
        norm_value = norm.item()
        print(f"[INFO] SLERP vs LERP fallback - Quaternion norm: {norm_value:.6f}, Expected: 1.0 ± 1e-4")
        assert torch.allclose(norm, torch.tensor(1.0, device=device), atol=1e-4), \
            "LERP fallback should maintain normalization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
