# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CollisionDetector class."""

import pytest
import torch

from isaaclab_tasks.direct.iris_ma3.safety import CollisionDetector, CollisionDetectorCfg


@pytest.fixture
def device():
    """Fixture for device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def basic_config():
    """Fixture for basic collision detector configuration."""
    return CollisionDetectorCfg(
        min_safe_distance=5.0,
        enable_velocity_based=False,
        lookahead_time=1.0,
        collision_penalty_scale=-100.0,
    )


@pytest.fixture
def detector(basic_config, device):
    """Fixture for collision detector instance."""
    num_envs = 16
    num_agents = 3
    return CollisionDetector(basic_config, num_envs, num_agents, device)


class TestCollisionDetectorInit:
    """Test collision detector initialization."""

    def test_init_creates_buffers(self, detector, device):
        """Test that initialization creates required buffers."""
        assert detector.distance_matrix.shape == (16, 3, 3)
        assert detector.collision_matrix.shape == (16, 3, 3)
        assert detector.distance_matrix.device.type == device.type
        assert detector.collision_matrix.dtype == torch.bool

    def test_init_buffers_zero(self, detector):
        """Test that buffers are initialized to zero/false."""
        assert torch.all(detector.distance_matrix == 0.0)
        assert torch.all(~detector.collision_matrix)


class TestDistanceComputation:
    """Test distance computation."""

    def test_compute_distances_basic(self, detector, device):
        """Test basic distance computation."""
        # Create test positions
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[3.0, 0.0, 0.0]] * 16, device=device),
            "agent_2": torch.tensor([[0.0, 4.0, 0.0]] * 16, device=device),
        }

        distances = detector.compute_distances(positions)

        # Check shape
        assert distances.shape == (16, 3, 3)

        # Check known distances (should be same across all envs)
        # agent_0 to agent_1: 3.0
        assert torch.allclose(distances[:, 0, 1], torch.tensor(3.0, device=device), atol=1e-5)
        # agent_0 to agent_2: 4.0
        assert torch.allclose(distances[:, 0, 2], torch.tensor(4.0, device=device), atol=1e-5)
        # agent_1 to agent_2: sqrt(3^2 + 4^2) = 5.0
        assert torch.allclose(distances[:, 1, 2], torch.tensor(5.0, device=device), atol=1e-5)

    def test_compute_distances_symmetry(self, detector, device):
        """Test that distance matrix is symmetric."""
        positions = {
            "agent_0": torch.randn(16, 3, device=device),
            "agent_1": torch.randn(16, 3, device=device),
            "agent_2": torch.randn(16, 3, device=device),
        }

        distances = detector.compute_distances(positions)

        # Check symmetry: d[i,j] = d[j,i]
        assert torch.allclose(distances[:, 0, 1], distances[:, 1, 0], atol=1e-5)
        assert torch.allclose(distances[:, 0, 2], distances[:, 2, 0], atol=1e-5)
        assert torch.allclose(distances[:, 1, 2], distances[:, 2, 1], atol=1e-5)

    def test_compute_distances_diagonal_infinity(self, detector, device):
        """Test that diagonal elements are infinity."""
        positions = {
            "agent_0": torch.randn(16, 3, device=device),
            "agent_1": torch.randn(16, 3, device=device),
            "agent_2": torch.randn(16, 3, device=device),
        }

        distances = detector.compute_distances(positions)

        # Check diagonal is infinity
        assert torch.all(torch.isinf(distances[:, 0, 0]))
        assert torch.all(torch.isinf(distances[:, 1, 1]))
        assert torch.all(torch.isinf(distances[:, 2, 2]))

    def test_compute_distances_zero_distance(self, detector, device):
        """Test zero distance between agents at same location."""
        # All agents at origin
        positions = {
            "agent_0": torch.zeros(16, 3, device=device),
            "agent_1": torch.zeros(16, 3, device=device),
            "agent_2": torch.zeros(16, 3, device=device),
        }

        distances = detector.compute_distances(positions)

        # All off-diagonal elements should be zero
        assert torch.allclose(distances[:, 0, 1], torch.tensor(0.0, device=device), atol=1e-5)
        assert torch.allclose(distances[:, 0, 2], torch.tensor(0.0, device=device), atol=1e-5)
        assert torch.allclose(distances[:, 1, 2], torch.tensor(0.0, device=device), atol=1e-5)


class TestCollisionDetection:
    """Test collision detection."""

    def test_detect_collisions_no_collision(self, detector, device):
        """Test collision detection when agents are far apart."""
        # Agents separated by 10 meters (> min_safe_distance of 5.0)
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[10.0, 0.0, 0.0]] * 16, device=device),
            "agent_2": torch.tensor([[0.0, 10.0, 0.0]] * 16, device=device),
        }

        collisions = detector.detect_collisions(positions)

        # No collisions should be detected
        # (except diagonal which we don't check)
        assert torch.all(~collisions[:, 0, 1])
        assert torch.all(~collisions[:, 0, 2])
        assert torch.all(~collisions[:, 1, 2])

    def test_detect_collisions_with_collision(self, detector, device):
        """Test collision detection when agents are too close."""
        # Agents separated by 3 meters (< min_safe_distance of 5.0)
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[3.0, 0.0, 0.0]] * 16, device=device),
            "agent_2": torch.tensor([[0.0, 10.0, 0.0]] * 16, device=device),
        }

        collisions = detector.detect_collisions(positions)

        # agent_0 and agent_1 should collide
        assert torch.all(collisions[:, 0, 1])
        assert torch.all(collisions[:, 1, 0])  # Symmetric

        # Others should not collide
        assert torch.all(~collisions[:, 0, 2])
        assert torch.all(~collisions[:, 1, 2])

    def test_detect_collisions_threshold(self, detector, device):
        """Test collision detection at exact threshold."""
        # Agents at exactly min_safe_distance
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[5.0, 0.0, 0.0]] * 16, device=device),
            "agent_2": torch.tensor([[10.0, 0.0, 0.0]] * 16, device=device),
        }

        collisions = detector.detect_collisions(positions)

        # At exactly threshold, should NOT collide (< not <=)
        assert torch.all(~collisions[:, 0, 1])


class TestCollisionPenalties:
    """Test collision penalty computation."""

    def test_compute_penalties_no_collision(self, detector, device):
        """Test penalties when no collisions occur."""
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[10.0, 0.0, 0.0]] * 16, device=device),
            "agent_2": torch.tensor([[0.0, 10.0, 0.0]] * 16, device=device),
        }

        penalties = detector.compute_collision_penalties(
            positions, ["agent_0", "agent_1", "agent_2"]
        )

        # No penalties
        assert torch.allclose(penalties["agent_0"], torch.tensor(0.0, device=device))
        assert torch.allclose(penalties["agent_1"], torch.tensor(0.0, device=device))
        assert torch.allclose(penalties["agent_2"], torch.tensor(0.0, device=device))

    def test_compute_penalties_one_collision(self, detector, device):
        """Test penalties when one pair collides."""
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[3.0, 0.0, 0.0]] * 16, device=device),  # Collides with agent_0
            "agent_2": torch.tensor([[0.0, 10.0, 0.0]] * 16, device=device),
        }

        penalties = detector.compute_collision_penalties(
            positions, ["agent_0", "agent_1", "agent_2"]
        )

        # agent_0 and agent_1 should have -1 penalty
        assert torch.allclose(penalties["agent_0"], torch.tensor(-1.0, device=device))
        assert torch.allclose(penalties["agent_1"], torch.tensor(-1.0, device=device))
        # agent_2 should have no penalty
        assert torch.allclose(penalties["agent_2"], torch.tensor(0.0, device=device))

    def test_compute_penalties_multiple_collisions(self, detector, device):
        """Test penalties when one agent collides with multiple others."""
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[3.0, 0.0, 0.0]] * 16, device=device),  # Collides with agent_0
            "agent_2": torch.tensor([[0.0, 3.0, 0.0]] * 16, device=device),  # Collides with agent_0
        }

        penalties = detector.compute_collision_penalties(
            positions, ["agent_0", "agent_1", "agent_2"]
        )

        # agent_0 collides with both others: -2
        assert torch.allclose(penalties["agent_0"], torch.tensor(-2.0, device=device))
        # agent_1 and agent_2 each collide with one: -1
        assert torch.allclose(penalties["agent_1"], torch.tensor(-1.0, device=device))
        assert torch.allclose(penalties["agent_2"], torch.tensor(-1.0, device=device))


class TestVelocityBasedPrediction:
    """Test velocity-based collision prediction."""

    def test_predict_collisions_disabled(self, detector, device):
        """Test that prediction returns current collisions when disabled."""
        positions = {
            "agent_0": torch.zeros(16, 3, device=device),
            "agent_1": torch.ones(16, 3, device=device) * 10.0,
            "agent_2": torch.ones(16, 3, device=device) * 20.0,
        }
        velocities = {
            "agent_0": torch.ones(16, 3, device=device),
            "agent_1": -torch.ones(16, 3, device=device),
            "agent_2": torch.zeros(16, 3, device=device),
        }

        # First detect current collisions
        current_collisions = detector.detect_collisions(positions)

        # Predict should return same as current (feature disabled)
        predicted = detector.predict_collisions(positions, velocities)

        assert torch.all(predicted == current_collisions)

    def test_predict_collisions_enabled(self, device):
        """Test collision prediction with velocity enabled."""
        cfg = CollisionDetectorCfg(
            min_safe_distance=5.0,
            enable_velocity_based=True,
            lookahead_time=1.0,
        )
        det = CollisionDetector(cfg, num_envs=16, num_agents=2, device=device)

        # Agents far apart but approaching
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[10.0, 0.0, 0.0]] * 16, device=device),
        }
        velocities = {
            "agent_0": torch.tensor([[3.0, 0.0, 0.0]] * 16, device=device),  # Moving right
            "agent_1": torch.tensor([[-3.0, 0.0, 0.0]] * 16, device=device),  # Moving left
        }

        # Current: no collision (distance=10)
        current = det.detect_collisions(positions)
        assert torch.all(~current[:, 0, 1])

        # Future (t+1): collision (distance=10-6=4 < 5)
        predicted = det.predict_collisions(positions, velocities)
        assert torch.all(predicted[:, 0, 1])


class TestClosestAgentDistance:
    """Test closest agent distance computation."""

    def test_get_closest_distance(self, detector, device):
        """Test getting closest agent distance."""
        positions = {
            "agent_0": torch.tensor([[0.0, 0.0, 0.0]] * 16, device=device),
            "agent_1": torch.tensor([[5.0, 0.0, 0.0]] * 16, device=device),
            "agent_2": torch.tensor([[10.0, 0.0, 0.0]] * 16, device=device),
        }

        detector.compute_distances(positions)

        # agent_0's closest is agent_1 at distance 5.0
        closest = detector.get_closest_agent_distance("agent_0", ["agent_0", "agent_1", "agent_2"])
        assert torch.allclose(closest, torch.tensor(5.0, device=device), atol=1e-5)

        # agent_1's closest is agent_0 at distance 5.0
        closest = detector.get_closest_agent_distance("agent_1", ["agent_0", "agent_1", "agent_2"])
        assert torch.allclose(closest, torch.tensor(5.0, device=device), atol=1e-5)

        # agent_2's closest is agent_1 at distance 5.0
        closest = detector.get_closest_agent_distance("agent_2", ["agent_0", "agent_1", "agent_2"])
        assert torch.allclose(closest, torch.tensor(5.0, device=device), atol=1e-5)


class TestReset:
    """Test reset functionality."""

    def test_reset_all(self, detector, device):
        """Test resetting all environments."""
        # Set some values
        detector.distance_matrix.fill_(10.0)
        detector.collision_matrix.fill_(True)

        # Reset all
        detector.reset()

        assert torch.all(detector.distance_matrix == 0.0)
        assert torch.all(~detector.collision_matrix)

    def test_reset_partial(self, detector, device):
        """Test resetting specific environments."""
        # Set some values
        detector.distance_matrix.fill_(10.0)
        detector.collision_matrix.fill_(True)

        # Reset only envs [0, 5, 10]
        env_ids = torch.tensor([0, 5, 10], device=device)
        detector.reset(env_ids)

        # Check reset envs are zero/false
        assert torch.all(detector.distance_matrix[env_ids] == 0.0)
        assert torch.all(~detector.collision_matrix[env_ids])

        # Check other envs unchanged
        other_ids = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15], device=device)
        assert torch.all(detector.distance_matrix[other_ids] == 10.0)
        assert torch.all(detector.collision_matrix[other_ids])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
