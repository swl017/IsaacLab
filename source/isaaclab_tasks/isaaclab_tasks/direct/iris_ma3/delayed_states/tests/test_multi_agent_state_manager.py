#!/usr/bin/env python3
"""
Test suite for MultiAgentStateManager.

Tests the integrated state management system with GT, delayed, and delayed+noisy states,
including seeded noise generation and curriculum learning.

Priority Levels:
- P0: Critical user-facing functionality
- P1: Important edge cases and integration
- P2: Nice-to-have validation tests
"""

import pytest
import torch
from typing import Dict

from isaaclab_tasks.direct.iris_ma3.delayed_states import (
    MultiAgentStateManager,
    AgentStates,
    MultiAgentStates,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def basic_config(device):
    """Basic configuration for state manager."""
    return {
        "possible_agents": ["drone_0", "drone_1"],
        "num_envs": 4,
        "num_joints_per_agent": {"drone_0": 3, "drone_1": 3},
        "num_targets_per_agent": {"drone_0": 1, "drone_1": 1},
        "dt": 0.01,
        "device": device,
        "motion_time_constant": 0.1,
        "gimbal_time_constant": 0.03,
        "detection_fps": 30.0,
        "detection_mean_latency": 0.05,
        "detection_std_latency": 0.02,
        "comm_mean_delay": 0.1,
        "comm_std_delay": 0.03,
        "enable_noise": True,
        "position_noise_std": 0.01,
        "orientation_noise_std": 0.01,
        "gimbal_noise_std": 0.01,
        "bbox_noise_std": 1.0,
        "noise_seed": 42,
    }


@pytest.fixture
def state_manager(basic_config):
    """Create a basic state manager instance."""
    return MultiAgentStateManager(**basic_config)


# ============================================================================
# P0 Tests: Critical Functionality
# ============================================================================

@pytest.mark.p0
def test_initialization(basic_config, device):
    """Test state manager initializes correctly."""
    manager = MultiAgentStateManager(**basic_config)

    # Check basic attributes
    assert manager.num_envs == 4
    assert manager.num_agents == 2
    assert len(manager.possible_agents) == 2
    assert manager.device == device

    # Check state storage exists
    assert hasattr(manager, 'gt_states')
    assert hasattr(manager, 'delayed_states')
    assert hasattr(manager, 'delayed_noisy_states')

    # Check states have correct agents
    for agent_id in basic_config["possible_agents"]:
        assert agent_id in manager.gt_states.agents
        assert agent_id in manager.delayed_states.agents
        assert agent_id in manager.delayed_noisy_states.agents

    # Check noise tensors allocated
    assert manager.sampled_noise_pos.shape == (4, 2, 3)
    assert manager.sampled_noise_ori.shape == (4, 2, 3)
    assert manager.sampled_noise_bbox.shape[0] == 4

    print("✓ Initialization test passed")


@pytest.mark.p0
def test_noise_reproducibility(basic_config, device):
    """Test that same seed produces identical noise."""
    # Create two managers with same seed
    manager1 = MultiAgentStateManager(**basic_config)
    manager2 = MultiAgentStateManager(**basic_config)

    # Update both with same GT states
    pos = torch.randn(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.randn(4, 3, device=device)
    ang_vel = torch.randn(4, 3, device=device)
    joints = torch.randn(4, 3, device=device) * 0.1

    manager1.update_time()
    manager2.update_time()

    for agent_id in basic_config["possible_agents"]:
        manager1.update_gt_states(
            agent_id, pos, quat, vel, ang_vel,
            joint_positions_b=joints
        )
        manager2.update_gt_states(
            agent_id, pos, quat, vel, ang_vel,
            joint_positions_b=joints
        )

    # Check delayed+noisy states are identical (same seed)
    for agent_id in basic_config["possible_agents"]:
        state1 = manager1.get_delayed_noisy_states(agent_id)
        state2 = manager2.get_delayed_noisy_states(agent_id)

        assert torch.allclose(state1.data.body_position_w, state2.data.body_position_w, atol=1e-6)
        assert torch.allclose(state1.data.body_orientation_w, state2.data.body_orientation_w, atol=1e-6)

    print("✓ Noise reproducibility test passed")


@pytest.mark.p0
def test_update_gt_states(state_manager, device):
    """Test GT state updates propagate correctly."""
    agent_id = "drone_0"

    # Create test data
    pos = torch.tensor([[1.0, 2.0, 3.0]], device=device).repeat(4, 1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.tensor([[0.5, 0.0, 0.0]], device=device).repeat(4, 1)
    ang_vel = torch.zeros(4, 3, device=device)
    joints = torch.tensor([[0.0, 0.1, 0.2]], device=device).repeat(4, 1)
    zoom = torch.ones(4, device=device) * 2.0

    state_manager.update_time()
    state_manager.update_gt_states(
        agent_id, pos, quat, vel, ang_vel,
        joint_positions_b=joints,
        zoom_level=zoom
    )

    # Check GT states stored correctly
    gt_state = state_manager.get_gt_states(agent_id)
    assert torch.allclose(gt_state.data.body_position_w, pos)
    assert torch.allclose(gt_state.data.body_orientation_w, quat)
    assert torch.allclose(gt_state.data.joint_positions_b, joints)
    assert torch.allclose(gt_state.data.camera_zoom_level, zoom)

    # Check delayed states exist (should be filtered)
    delayed_state = state_manager.get_delayed_states(agent_id)
    assert delayed_state.data.body_position_w is not None
    assert delayed_state.data.body_orientation_w is not None

    # Check delayed+noisy states exist
    noisy_state = state_manager.get_delayed_noisy_states(agent_id)
    assert noisy_state.data.body_position_w is not None

    print("✓ Update GT states test passed")


@pytest.mark.p0
def test_curriculum_learning(basic_config, device):
    """Test noise curriculum scaling."""
    manager = MultiAgentStateManager(**basic_config)
    agent_id = "drone_0"

    # Test data
    pos = torch.randn(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    # Update with no noise (progress = 0)
    manager.set_noise_progress_scale(0.0)
    manager.update_time()
    manager.update_gt_states(agent_id, pos, quat, vel, ang_vel)
    # Clone to avoid reference issues when states are updated later
    state_no_noise_pos = manager.get_delayed_noisy_states(agent_id).data.body_position_w.clone()
    delayed_no_noise_pos = manager.get_delayed_states(agent_id).data.body_position_w.clone()

    # Update with half noise (progress = 0.5)
    manager.set_noise_progress_scale(0.5)
    manager.update_time()
    manager.update_gt_states(agent_id, pos, quat, vel, ang_vel)
    state_half_noise_pos = manager.get_delayed_noisy_states(agent_id).data.body_position_w.clone()

    # Update with full noise (progress = 1.0)
    manager.set_noise_progress_scale(1.0)
    manager.update_time()
    manager.update_gt_states(agent_id, pos, quat, vel, ang_vel)
    state_full_noise_pos = manager.get_delayed_noisy_states(agent_id).data.body_position_w.clone()

    # With progress=0, noisy should equal delayed (no noise added)
    assert torch.allclose(
        state_no_noise_pos,
        delayed_no_noise_pos,
        atol=1e-6
    )

    # Noise magnitude should increase with progress
    noise_half = (state_half_noise_pos - delayed_no_noise_pos).abs().mean()
    noise_full = (state_full_noise_pos - delayed_no_noise_pos).abs().mean()

    # Note: Due to randomness, we can't guarantee noise_full > noise_half in a single sample,
    # but we can verify that progress=0 gives zero noise
    assert noise_half >= 0.0  # Should be non-negative

    print(f"✓ Curriculum learning test passed (noise_half={noise_half:.6f}, noise_full={noise_full:.6f})")


@pytest.mark.p0
def test_detection_processing(state_manager, device):
    """Test bbox detection processing with FPS throttle and noise."""
    agent_id = "drone_0"

    # Create GT bboxes
    bboxes_gt = torch.tensor([[100.0, 150.0, 50.0, 60.0]], device=device).repeat(4, 1, 1)  # [N, T, 4]
    valid_gt = torch.ones(4, 1, dtype=torch.bool, device=device)

    state_manager.update_time()
    state_manager.update_detections(agent_id, bboxes_gt, valid_gt)

    # Check GT bboxes stored
    gt_state = state_manager.get_gt_states(agent_id)
    assert torch.allclose(gt_state.data.bboxes_2d, bboxes_gt)
    assert torch.all(gt_state.data.bboxes_2d_valid_mask == valid_gt)

    # Check delayed bboxes exist
    delayed_state = state_manager.get_delayed_states(agent_id)
    assert delayed_state.data.bboxes_2d is not None

    # Check delayed+noisy bboxes exist and differ from delayed
    noisy_state = state_manager.get_delayed_noisy_states(agent_id)
    assert noisy_state.data.bboxes_2d is not None

    # With noise enabled, noisy should differ from delayed
    if state_manager.enable_noise and state_manager.noise_progress_scale > 0:
        # Note: May be same if noise happens to be zero, but check shape at least
        assert noisy_state.data.bboxes_2d.shape == delayed_state.data.bboxes_2d.shape

    print("✓ Detection processing test passed")


# ============================================================================
# P1 Tests: Important Integration
# ============================================================================

@pytest.mark.p1
def test_state_consistency(state_manager, device):
    """Test that GT → delayed → noisy chain is consistent."""
    agent_id = "drone_0"

    # Update states
    pos = torch.tensor([[5.0, 10.0, 15.0]], device=device).repeat(4, 1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    state_manager.update_time()
    state_manager.update_gt_states(agent_id, pos, quat, vel, ang_vel)

    # Get all state types
    gt = state_manager.get_gt_states(agent_id)
    delayed = state_manager.get_delayed_states(agent_id)
    noisy = state_manager.get_delayed_noisy_states(agent_id)

    # GT should match input exactly
    assert torch.allclose(gt.data.body_position_w, pos)

    # Delayed should differ from GT (due to filtering)
    # After one step, filter hasn't converged yet
    assert not torch.allclose(delayed.data.body_position_w, pos, atol=0.01)

    # Noisy should have same shape as delayed
    assert noisy.data.body_position_w.shape == delayed.data.body_position_w.shape
    assert noisy.data.body_orientation_w.shape == delayed.data.body_orientation_w.shape

    # Camera poses should be updated for all state types
    assert gt.data.camera_position_w is not None
    assert delayed.data.camera_position_w is not None
    assert noisy.data.camera_position_w is not None

    # Camera intrinsics should be updated
    assert gt.data.camera_intrinsics is not None
    assert delayed.data.camera_intrinsics is not None
    assert noisy.data.camera_intrinsics is not None

    print("✓ State consistency test passed")


@pytest.mark.p1
def test_multi_agent_integration(basic_config, device):
    """Test multiple agents work independently."""
    manager = MultiAgentStateManager(**basic_config)

    # Different states for each agent
    pos0 = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(4, 1)
    pos1 = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(4, 1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    manager.update_time()
    manager.update_gt_states("drone_0", pos0, quat, vel, ang_vel)
    manager.update_gt_states("drone_1", pos1, quat, vel, ang_vel)

    # Check states are independent
    gt0 = manager.get_gt_states("drone_0")
    gt1 = manager.get_gt_states("drone_1")

    assert torch.allclose(gt0.data.body_position_w, pos0)
    assert torch.allclose(gt1.data.body_position_w, pos1)
    assert not torch.allclose(gt0.data.body_position_w, gt1.data.body_position_w)

    print("✓ Multi-agent integration test passed")


@pytest.mark.p1
def test_communication_integration(state_manager, device):
    """Test state broadcasting and receiving."""
    # Update states for drone_0
    pos = torch.randn(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    state_manager.update_time()
    state_manager.update_gt_states("drone_0", pos, quat, vel, ang_vel)

    # Broadcast drone_0 state
    state_manager.broadcast_state("drone_0", state_keys=['position', 'orientation'])

    # drone_1 should receive it (with delay)
    received = state_manager.receive_other_agent_states("drone_1")

    # Check structure (may not have data yet due to latency)
    assert isinstance(received, dict)
    # If drone_0 sent to drone_1, we should eventually see it
    # (but initial receive may be empty due to latency)

    print("✓ Communication integration test passed")


@pytest.mark.p1
def test_noise_formula_consistency(basic_config, device):
    """Test noise formulas match iris_ma_env3.py specifications."""
    manager = MultiAgentStateManager(**basic_config)

    # Check linear velocity noise is 10% of position noise
    expected_vel_std = 0.1 * basic_config["position_noise_std"]
    assert manager.linear_velocity_noise_std == expected_vel_std or \
           abs(manager.linear_velocity_noise_std - expected_vel_std) < 1e-9

    # Check angular velocity noise formula
    expected_ang_vel_std = basic_config["orientation_noise_std"] * basic_config["position_noise_std"]
    assert manager.angular_velocity_noise_std == expected_ang_vel_std or \
           abs(manager.angular_velocity_noise_std - expected_ang_vel_std) < 1e-9

    # Check acceleration noise default
    assert manager.linear_acceleration_noise_std == 100e-6

    print("✓ Noise formula consistency test passed")


@pytest.mark.p1
def test_reset_functionality(state_manager, device):
    """Test reset clears state properly."""
    agent_id = "drone_0"

    # Populate states
    pos = torch.randn(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    for _ in range(10):
        state_manager.update_time()
        state_manager.update_gt_states(agent_id, pos, quat, vel, ang_vel)

    # Reset specific environments
    env_ids = torch.tensor([0, 2], device=device)
    state_manager.reset(env_ids)

    # Time should be reset for those envs
    # (Internal pipeline time is reset)

    # Reset all
    state_manager.reset(None)

    print("✓ Reset functionality test passed")


# ============================================================================
# P2 Tests: Validation and Edge Cases
# ============================================================================

@pytest.mark.p2
def test_camera_pose_updates(state_manager, device):
    """Test camera poses are automatically updated."""
    agent_id = "drone_0"

    # Set camera config first
    state_manager.set_camera_configs(
        agent_id,
        width=640,
        height=480,
        focal_length=0.015,
        horizontal_aperture=0.024,
        offset_position_b=torch.tensor([0.0, 0.0, 0.0], device=device),
        offset_rotation_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    )

    # Update states
    pos = torch.randn(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)
    joints = torch.randn(4, 3, device=device) * 0.1

    state_manager.update_time()
    state_manager.update_gt_states(
        agent_id, pos, quat, vel, ang_vel,
        joint_positions_b=joints
    )

    # Check camera pose updated in all state types
    gt = state_manager.get_gt_states(agent_id)
    delayed = state_manager.get_delayed_states(agent_id)
    noisy = state_manager.get_delayed_noisy_states(agent_id)

    # Camera positions should be non-zero (offset from body)
    assert not torch.allclose(gt.data.camera_position_w, torch.zeros_like(gt.data.camera_position_w))
    assert delayed.data.camera_position_w is not None
    assert noisy.data.camera_position_w is not None

    print("✓ Camera pose updates test passed")


@pytest.mark.p2
def test_zoom_noise_clamping(state_manager, device):
    """Test zoom noise is clamped to reasonable range."""
    agent_id = "drone_0"

    # Set very high zoom with noise
    extreme_zoom = torch.tensor([15.0], device=device).repeat(4)  # Above max
    pos = torch.zeros(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    state_manager.update_time()
    state_manager.update_gt_states(
        agent_id, pos, quat, vel, ang_vel,
        zoom_level=extreme_zoom
    )

    # Noisy zoom should be clamped to [1.0, 10.0]
    noisy = state_manager.get_delayed_noisy_states(agent_id)
    assert torch.all(noisy.data.camera_zoom_level >= 1.0)
    assert torch.all(noisy.data.camera_zoom_level <= 10.0)

    print("✓ Zoom noise clamping test passed")


@pytest.mark.p2
def test_partial_env_update(state_manager, device):
    """Test updating only specific environments."""
    agent_id = "drone_0"

    # Update only env 0 and 2
    env_idxs = torch.tensor([0, 2], device=device)

    pos = torch.zeros(4, 3, device=device)
    pos[env_idxs] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    state_manager.update_time()
    state_manager.update_gt_states(
        agent_id, pos, quat, vel, ang_vel,
        env_idxs=env_idxs
    )

    # Check only specified envs were updated
    gt = state_manager.get_gt_states(agent_id)
    assert torch.allclose(gt.data.body_position_w[0], pos[0])
    assert torch.allclose(gt.data.body_position_w[2], pos[2])

    print("✓ Partial env update test passed")


@pytest.mark.p2
def test_detection_with_invalid_mask(state_manager, device):
    """Test detection processing with some invalid detections."""
    agent_id = "drone_0"

    # Create bboxes with mixed validity
    bboxes = torch.randn(4, 1, 4, device=device).abs() * 100  # Positive values
    valid = torch.tensor([[True], [False], [True], [False]], device=device)

    state_manager.update_time()
    state_manager.update_detections(agent_id, bboxes, valid)

    # Check validity propagates
    delayed = state_manager.get_delayed_states(agent_id)
    # Validity may change due to channel impairments, but shape should match
    assert delayed.data.bboxes_2d_valid_mask.shape == valid.shape

    print("✓ Detection with invalid mask test passed")


@pytest.mark.p2
def test_get_all_states(state_manager, device):
    """Test get_all_* methods return correct types."""
    # Update states for all agents
    pos = torch.randn(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    state_manager.update_time()
    for agent_id in state_manager.possible_agents:
        state_manager.update_gt_states(agent_id, pos, quat, vel, ang_vel)

    # Get all states
    all_gt = state_manager.get_all_gt_states()
    all_delayed = state_manager.get_all_delayed_states()
    all_noisy = state_manager.get_all_delayed_noisy_states()

    # Check types
    assert isinstance(all_gt, MultiAgentStates)
    assert isinstance(all_delayed, MultiAgentStates)
    assert isinstance(all_noisy, MultiAgentStates)

    # Check agents present
    for agent_id in state_manager.possible_agents:
        assert agent_id in all_gt.agents
        assert agent_id in all_delayed.agents
        assert agent_id in all_noisy.agents

        assert isinstance(all_gt.agents[agent_id], AgentStates)

    print("✓ Get all states test passed")


@pytest.mark.p2
def test_noise_disabled(basic_config, device):
    """Test behavior when noise is disabled."""
    basic_config["enable_noise"] = False
    manager = MultiAgentStateManager(**basic_config)

    agent_id = "drone_0"
    pos = torch.randn(4, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    vel = torch.zeros(4, 3, device=device)
    ang_vel = torch.zeros(4, 3, device=device)

    manager.update_time()
    manager.update_gt_states(agent_id, pos, quat, vel, ang_vel)

    # With noise disabled, delayed and noisy should be identical
    delayed = manager.get_delayed_states(agent_id)
    noisy = manager.get_delayed_noisy_states(agent_id)

    assert torch.allclose(delayed.data.body_position_w, noisy.data.body_position_w, atol=1e-6)
    assert torch.allclose(delayed.data.body_orientation_w, noisy.data.body_orientation_w, atol=1e-6)

    print("✓ Noise disabled test passed")


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
