# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SafetyManager class."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.direct.iris_ma3.safety import (
    SafetyManager,
    SafetyManagerCfg,
    CollisionDetectorCfg,
    TTCComputerCfg,
)


@pytest.fixture
def device():
    """Fixture for device selection."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def num_envs():
    """Fixture for number of environments."""
    return 16


@pytest.fixture
def num_agents():
    """Fixture for number of agents."""
    return 3


@pytest.fixture
def agent_ids():
    """Fixture for agent identifiers."""
    return ["drone_0", "drone_1", "drone_2"]


@pytest.fixture
def default_cfg():
    """Fixture for default SafetyManager configuration."""
    return SafetyManagerCfg(
        collision_cfg=CollisionDetectorCfg(
            min_safe_distance=5.0,
            enable_velocity_based=False,
            collision_penalty_scale=-10.0,
        ),
        ttc_cfg=TTCComputerCfg(
            horizon_sec=6.0,
            ema_alpha_s=0.6,
            ema_alpha_f=0.6,
            ttc_penalty_scale=-10.0,
        ),
        enable_collision_detection=True,
        enable_ttc_computation=True,
    )


@pytest.fixture
def safety_manager(default_cfg, num_envs, num_agents, device):
    """Fixture for SafetyManager instance."""
    return SafetyManager(default_cfg, num_envs, num_agents, device)


# ========================================
# Initialization Tests
# ========================================


class TestInitialization:
    """Test SafetyManager initialization."""

    def test_init_with_both_features(self, default_cfg, num_envs, num_agents, device):
        """Test initialization with both collision and TTC enabled."""
        manager = SafetyManager(default_cfg, num_envs, num_agents, device)

        assert manager.num_envs == num_envs
        assert manager.num_agents == num_agents
        assert manager.device == device
        assert manager.collision_detector is not None
        assert manager.ttc_computers is not None
        assert len(manager.ttc_computers) == num_agents

    def test_init_collision_only(self, num_envs, num_agents, device):
        """Test initialization with only collision detection enabled."""
        cfg = SafetyManagerCfg(
            enable_collision_detection=True,
            enable_ttc_computation=False,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        assert manager.collision_detector is not None
        assert manager.ttc_computers is None

    def test_init_ttc_only(self, num_envs, num_agents, device):
        """Test initialization with only TTC computation enabled."""
        cfg = SafetyManagerCfg(
            enable_collision_detection=False,
            enable_ttc_computation=True,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        assert manager.collision_detector is None
        assert manager.ttc_computers is not None
        assert len(manager.ttc_computers) == num_agents

    def test_init_no_features(self, num_envs, num_agents, device):
        """Test initialization with both features disabled."""
        cfg = SafetyManagerCfg(
            enable_collision_detection=False,
            enable_ttc_computation=False,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        assert manager.collision_detector is None
        assert manager.ttc_computers is None


# ========================================
# Collision Penalty Tests
# ========================================


class TestCollisionPenalties:
    """Test collision penalty computation."""

    def test_compute_collision_penalties_no_collision(self, safety_manager, num_envs, agent_ids, device):
        """Test collision penalties when agents are far apart."""
        # Place agents far apart (>5.0 units)
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[10.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 10.0, 0.0]] * num_envs, device=device),
        }

        penalties = safety_manager.compute_collision_penalties(positions, agent_ids)

        assert len(penalties) == 3
        for agent_id in agent_ids:
            assert penalties[agent_id].shape == (num_envs,)
            assert torch.all(penalties[agent_id] == 0.0)

    def test_compute_collision_penalties_with_collision(self, safety_manager, num_envs, agent_ids, device):
        """Test collision penalties when agents are too close."""
        # Place agents close together (<5.0 units)
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }

        penalties = safety_manager.compute_collision_penalties(positions, agent_ids)

        assert len(penalties) == 3
        for agent_id in agent_ids:
            assert penalties[agent_id].shape == (num_envs,)
            # Each agent collides with 2 others
            assert torch.all(penalties[agent_id] == -2.0)

    def test_compute_collision_penalties_disabled(self, num_envs, num_agents, agent_ids, device):
        """Test collision penalties when collision detection is disabled."""
        cfg = SafetyManagerCfg(
            enable_collision_detection=False,
            enable_ttc_computation=True,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[1.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 1.0, 0.0]] * num_envs, device=device),
        }

        penalties = manager.compute_collision_penalties(positions, agent_ids)

        # Should return zero penalties when disabled
        assert len(penalties) == 3
        for agent_id in agent_ids:
            assert torch.all(penalties[agent_id] == 0.0)


# ========================================
# TTC Penalty Tests
# ========================================


class TestTTCPenalties:
    """Test TTC penalty computation."""

    def test_compute_ttc_penalties_no_valid(self, safety_manager, num_envs, agent_ids, device):
        """Test TTC penalties with no valid detections."""
        # All invalid detections
        bboxes = {
            agent_id: torch.zeros(num_envs, 4, device=device) for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.zeros(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        ttc_results = safety_manager.compute_ttc_penalties(
            bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        assert len(ttc_results) == 3
        for agent_id in agent_ids:
            phi, tau = ttc_results[agent_id]
            assert phi.shape == (num_envs,)
            assert tau.shape == (num_envs,)
            # No valid detections should give zero penalty
            assert torch.all(phi == 0.0)

    def test_compute_ttc_penalties_growing_bbox(self, safety_manager, num_envs, agent_ids, device):
        """Test TTC penalties with growing bbox (approaching target)."""
        # Simulate approaching target (growing bbox)
        bboxes_t0 = {
            agent_id: torch.tensor(
                [[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device
            )  # (x, y, w, h)
            for agent_id in agent_ids
        }
        valid_t0 = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        # First step with small bbox
        ttc_t0 = safety_manager.compute_ttc_penalties(
            bboxes_t0, valid_t0, focal_lengths, agent_ids, dt=0.1
        )

        # Second step with larger bbox
        bboxes_t1 = {
            agent_id: torch.tensor(
                [[0.5, 0.5, 0.15, 0.15]] * num_envs, device=device
            )  # Bbox grew
            for agent_id in agent_ids
        }
        ttc_t1 = safety_manager.compute_ttc_penalties(
            bboxes_t1, valid_t0, focal_lengths, agent_ids, dt=0.1
        )

        # Growing bbox should produce positive TTC penalty (approaching)
        for agent_id in agent_ids:
            phi_t1, tau_t1 = ttc_t1[agent_id]
            assert phi_t1.shape == (num_envs,)
            assert tau_t1.shape == (num_envs,)
            # Should have some penalty for approaching
            assert torch.all(phi_t1 >= 0.0)
            assert torch.all(phi_t1 <= 1.0)

    def test_compute_ttc_penalties_disabled(self, num_envs, num_agents, agent_ids, device):
        """Test TTC penalties when TTC computation is disabled."""
        cfg = SafetyManagerCfg(
            enable_collision_detection=True,
            enable_ttc_computation=False,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        ttc_results = manager.compute_ttc_penalties(
            bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        # Should return zero penalties and infinite TTC when disabled
        assert len(ttc_results) == 3
        for agent_id in agent_ids:
            phi, tau = ttc_results[agent_id]
            assert torch.all(phi == 0.0)
            assert torch.all(tau == float("inf"))


# ========================================
# Unified Penalty Tests
# ========================================


class TestUnifiedPenalties:
    """Test unified safety penalty computation."""

    def test_compute_all_safety_penalties(self, safety_manager, num_envs, agent_ids, device):
        """Test computing all safety penalties at once."""
        # Setup data
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[10.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 10.0, 0.0]] * num_envs, device=device),
        }
        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        penalties = safety_manager.compute_all_safety_penalties(
            positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        # Verify structure
        assert len(penalties) == 3
        for agent_id in agent_ids:
            assert "collision" in penalties[agent_id]
            assert "ttc_penalty" in penalties[agent_id]
            assert "ttc_value" in penalties[agent_id]

            assert penalties[agent_id]["collision"].shape == (num_envs,)
            assert penalties[agent_id]["ttc_penalty"].shape == (num_envs,)
            assert penalties[agent_id]["ttc_value"].shape == (num_envs,)

    def test_compute_all_safety_penalties_with_collision(
        self, safety_manager, num_envs, agent_ids, device
    ):
        """Test unified penalties with collisions present."""
        # Setup collision scenario
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),  # Close
            "drone_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),  # Close
        }
        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        penalties = safety_manager.compute_all_safety_penalties(
            positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        # Verify collision penalties are negative
        for agent_id in agent_ids:
            assert torch.all(penalties[agent_id]["collision"] < 0.0)


# ========================================
# State Access Tests
# ========================================


class TestStateAccess:
    """Test state access methods."""

    def test_get_distance_matrix(self, safety_manager, num_envs, agent_ids, device):
        """Test getting distance matrix."""
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[3.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 4.0, 0.0]] * num_envs, device=device),
        }

        # Compute distances first
        safety_manager.compute_collision_penalties(positions, agent_ids)

        # Get distance matrix
        dist_matrix = safety_manager.get_distance_matrix()

        assert dist_matrix is not None
        assert dist_matrix.shape == (num_envs, 3, 3)
        assert torch.allclose(
            dist_matrix[:, 0, 1], torch.tensor(3.0, device=device), atol=1e-5
        )

    def test_get_collision_matrix(self, safety_manager, num_envs, agent_ids, device):
        """Test getting collision matrix."""
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }

        # Compute collisions first
        safety_manager.compute_collision_penalties(positions, agent_ids)

        # Get collision matrix
        coll_matrix = safety_manager.get_collision_matrix()

        assert coll_matrix is not None
        assert coll_matrix.shape == (num_envs, 3, 3)
        assert coll_matrix.dtype == torch.bool

    def test_get_matrices_disabled(self, num_envs, num_agents, device):
        """Test getting matrices when collision detection is disabled."""
        cfg = SafetyManagerCfg(
            enable_collision_detection=False,
            enable_ttc_computation=True,
        )
        manager = SafetyManager(cfg, num_envs, num_agents, device)

        assert manager.get_distance_matrix() is None
        assert manager.get_collision_matrix() is None


# ========================================
# Reset Tests
# ========================================


class TestReset:
    """Test reset functionality."""

    def test_reset_all(self, safety_manager, num_envs, agent_ids, device):
        """Test resetting all environments."""
        # Setup initial state
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }
        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        # Compute penalties to build up state
        safety_manager.compute_all_safety_penalties(
            positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        # Reset all
        safety_manager.reset(None)

        # Verify collision detector was reset
        if safety_manager.collision_detector is not None:
            assert torch.all(safety_manager.collision_detector.distance_matrix == 0.0)
            assert torch.all(safety_manager.collision_detector.collision_matrix == False)

        # Verify TTC computers were reset
        if safety_manager.ttc_computers is not None:
            for ttc_comp in safety_manager.ttc_computers.values():
                assert torch.all(ttc_comp.logsize_ema == 0.0)
                assert torch.all(ttc_comp.logf_ema == 0.0)

    def test_reset_partial(self, safety_manager, num_envs, agent_ids, device):
        """Test resetting specific environments."""
        # Setup initial state
        positions = {
            "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_1": torch.tensor([[2.0, 0.0, 0.0]] * num_envs, device=device),
            "drone_2": torch.tensor([[0.0, 2.0, 0.0]] * num_envs, device=device),
        }
        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        # Compute penalties to build up state
        safety_manager.compute_all_safety_penalties(
            positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        # Reset only first 4 environments
        env_ids = torch.arange(4, device=device)
        safety_manager.reset(env_ids)

        # Verify partial reset (only first 4 envs should be zero)
        if safety_manager.ttc_computers is not None:
            for ttc_comp in safety_manager.ttc_computers.values():
                assert torch.all(ttc_comp.logsize_ema[:4] == 0.0)
                # Rest should still have values (may be zero if no computation yet)

    def test_reset_ttc_with_state(self, safety_manager, num_envs, agent_ids, device):
        """Test resetting TTC with current state."""
        env_ids = torch.arange(num_envs, device=device)
        bboxes = {
            agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
            for agent_id in agent_ids
        }
        valid_masks = {
            agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
            for agent_id in agent_ids
        }
        focal_lengths = {
            agent_id: (
                torch.full((num_envs,), 500.0, device=device),
                torch.full((num_envs,), 500.0, device=device),
            )
            for agent_id in agent_ids
        }

        safety_manager.reset_ttc_with_state(
            env_ids, bboxes, valid_masks, focal_lengths, agent_ids
        )

        # Verify TTC buffers were initialized with valid values (not zero)
        if safety_manager.ttc_computers is not None:
            for ttc_comp in safety_manager.ttc_computers.values():
                # EMAs should be initialized with log values
                assert not torch.all(ttc_comp.logsize_ema == 0.0)
                assert not torch.all(ttc_comp.logf_ema == 0.0)


# ========================================
# Integration Tests
# ========================================


class TestIntegration:
    """Integration tests for SafetyManager."""

    def test_multi_step_simulation(self, safety_manager, num_envs, agent_ids, device):
        """Test multi-step simulation with realistic scenario."""
        # Simulate 10 steps of agents approaching each other
        for step in range(10):
            # Agents gradually move closer
            distance = 10.0 - step * 0.5  # Start at 10, move to 5
            positions = {
                "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
                "drone_1": torch.tensor(
                    [[distance, 0.0, 0.0]] * num_envs, device=device
                ),
                "drone_2": torch.tensor(
                    [[0.0, distance, 0.0]] * num_envs, device=device
                ),
            }

            # Bboxes gradually grow (approaching target)
            bbox_size = 0.05 + step * 0.01
            bboxes = {
                agent_id: torch.tensor(
                    [[0.5, 0.5, bbox_size, bbox_size]] * num_envs, device=device
                )
                for agent_id in agent_ids
            }
            valid_masks = {
                agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
                for agent_id in agent_ids
            }
            focal_lengths = {
                agent_id: (
                    torch.full((num_envs,), 500.0, device=device),
                    torch.full((num_envs,), 500.0, device=device),
                )
                for agent_id in agent_ids
            }

            penalties = safety_manager.compute_all_safety_penalties(
                positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
            )

            # Verify penalties increase over time
            for agent_id in agent_ids:
                assert penalties[agent_id]["collision"].shape == (num_envs,)
                assert penalties[agent_id]["ttc_penalty"].shape == (num_envs,)

                # At final step, should have collision penalties
                if step == 9:
                    assert torch.all(penalties[agent_id]["collision"] < 0.0)

    def test_reset_during_simulation(self, safety_manager, num_envs, agent_ids, device):
        """Test resetting specific environments during simulation."""
        # Run a few steps
        for _ in range(5):
            positions = {
                "drone_0": torch.tensor([[0.0, 0.0, 0.0]] * num_envs, device=device),
                "drone_1": torch.tensor([[3.0, 0.0, 0.0]] * num_envs, device=device),
                "drone_2": torch.tensor([[0.0, 3.0, 0.0]] * num_envs, device=device),
            }
            bboxes = {
                agent_id: torch.tensor([[0.5, 0.5, 0.1, 0.1]] * num_envs, device=device)
                for agent_id in agent_ids
            }
            valid_masks = {
                agent_id: torch.ones(num_envs, dtype=torch.bool, device=device)
                for agent_id in agent_ids
            }
            focal_lengths = {
                agent_id: (
                    torch.full((num_envs,), 500.0, device=device),
                    torch.full((num_envs,), 500.0, device=device),
                )
                for agent_id in agent_ids
            }

            safety_manager.compute_all_safety_penalties(
                positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
            )

        # Reset half the environments
        reset_ids = torch.arange(num_envs // 2, device=device)
        safety_manager.reset(reset_ids)

        # Continue simulation - should work without errors
        penalties = safety_manager.compute_all_safety_penalties(
            positions, bboxes, valid_masks, focal_lengths, agent_ids, dt=0.1
        )

        assert len(penalties) == 3
        for agent_id in agent_ids:
            assert penalties[agent_id]["collision"].shape == (num_envs,)
