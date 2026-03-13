# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPA Reward Shaper module for training-time safety penalty.

Implements predictive Closest Point of Approach (CPA) barrier for velocity-aware
collision avoidance reward shaping. Operates on ground-truth state only.

Ref: safety_spec.md Section 2.2, 2.3, 4.1
"""

from __future__ import annotations

import torch

from .cbf_cfg import CPARewardShaperCfg


class CPARewardShaper:
    """Predictive CPA-based safety penalty for reward shaping.

    The CPA barrier is velocity-aware, penalizing approach trajectories that would
    lead to collision while allowing parallel flight at close range. This provides
    better gradient signal than pure distance-based barriers.

    Key properties:
    - Uses ground-truth positions (no delay, no noise)
    - Uses commanded velocities (clean signal in simulator)
    - Discrete-time formulation avoids numerical differentiation
    - Fully batched for GPU efficiency

    Usage:
        shaper = CPARewardShaper(cfg, num_envs, num_agents, device)
        penalty = shaper.compute_penalty(gt_positions, cmd_velocities, dt)
        reward = task_reward - cfg.lambda_cbf * penalty
    """

    def __init__(
        self,
        cfg: CPARewardShaperCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the CPA reward shaper.

        Args:
            cfg: Configuration for the CPA reward shaper.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Device for tensor computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Pre-compute constants
        self.D_s_sq = cfg.D_s**2

        # Generate upper triangular indices for unique pairs (i < j)
        # This avoids computing duplicate pairs and self-pairs
        self._pair_indices = self._generate_pair_indices()

    def _generate_pair_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate indices for all unique agent pairs (i < j).

        Returns:
            Tuple of (i_indices, j_indices) for upper triangular pairs.
        """
        i_list = []
        j_list = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                i_list.append(i)
                j_list.append(j)

        i_indices = torch.tensor(i_list, dtype=torch.long, device=self.device)
        j_indices = torch.tensor(j_list, dtype=torch.long, device=self.device)
        return i_indices, j_indices

    def _compute_cpa_barrier(
        self,
        dp: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute CPA barrier value for relative position and velocity.

        Args:
            dp: Relative position delta_p = p_i - p_j, shape (E, P, 3).
            dv: Relative velocity delta_v = v_i - v_j, shape (E, P, 3).

        Returns:
            h_cpa: CPA barrier value = d_CPA^2 - D_s^2, shape (E, P).
            tau: Time of closest approach, clamped to [0, T], shape (E, P).
        """
        # Time of closest approach: tau* = -dp·dv / ||dv||^2
        # Clamp to [0, T] to avoid negative times and extrapolation
        dv_sq = (dv * dv).sum(dim=-1) + self.cfg.epsilon  # (E, P)
        tau_star = -(dp * dv).sum(dim=-1) / dv_sq  # (E, P)
        tau = tau_star.clamp(0.0, self.cfg.T)  # (E, P)

        # Position at closest approach: dp_cpa = dp + tau * dv
        dp_cpa = dp + tau.unsqueeze(-1) * dv  # (E, P, 3)

        # CPA distance squared
        d_cpa_sq = (dp_cpa * dp_cpa).sum(dim=-1)  # (E, P)

        # Barrier value: h = d_CPA^2 - D_s^2
        # h > 0: safe trajectory (miss distance > D_s)
        # h = 0: grazing collision
        # h < 0: predicted collision
        h_cpa = d_cpa_sq - self.D_s_sq  # (E, P)

        return h_cpa, tau

    def compute_penalty(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute CPA barrier penalty using discrete-time condition.

        The penalty is a hinge loss on the discrete-time CBF condition:
        violation = max(0, (1 - gamma*dt)*h_current - h_next)

        Args:
            positions: Ground-truth positions (E, N, 3).
            velocities: Commanded velocities (E, N, 3).
            dt: Timestep duration in seconds.

        Returns:
            penalty: Sum of violations per environment (E,).
        """
        i_idx, j_idx = self._pair_indices
        num_pairs = len(i_idx)

        if num_pairs == 0:
            # Single agent, no pairwise penalty
            return torch.zeros(self.num_envs, device=self.device)

        # Extract positions and velocities for all pairs
        # positions[:, i_idx]: (E, P, 3) positions of first agent in each pair
        # positions[:, j_idx]: (E, P, 3) positions of second agent in each pair
        p_i = positions[:, i_idx]  # (E, P, 3)
        p_j = positions[:, j_idx]  # (E, P, 3)
        v_i = velocities[:, i_idx]  # (E, P, 3)
        v_j = velocities[:, j_idx]  # (E, P, 3)

        # Relative position and velocity
        dp = p_i - p_j  # (E, P, 3)
        dv = v_i - v_j  # (E, P, 3)

        # Current CPA barrier
        h_current, _ = self._compute_cpa_barrier(dp, dv)  # (E, P)

        # Predicted next-step relative position (assuming constant velocity)
        dp_next = dp + dv * dt  # (E, P, 3)

        # Next-step CPA barrier
        # Note: velocities are assumed constant (commanded velocity)
        h_next, _ = self._compute_cpa_barrier(dp_next, dv)  # (E, P)

        # Discrete-time CBF condition:
        # h_next >= (1 - gamma*dt) * h_current
        # Violation: relu((1 - gamma*dt)*h_current - h_next)
        required = (1.0 - self.cfg.gamma * dt) * h_current
        violation = torch.relu(required - h_next)  # (E, P)

        # Sum violations across all pairs
        penalty = violation.sum(dim=-1)  # (E,)

        return penalty

    def compute_penalty_simple(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Compute simplified CPA margin penalty (continuous-time approximation).

        Alternative formulation using continuous-time margin hinge:
        penalty = max(0, -m) where m = h_dot + gamma*h

        This is simpler but requires computing h_dot analytically.

        Args:
            positions: Ground-truth positions (E, N, 3).
            velocities: Commanded velocities (E, N, 3).

        Returns:
            penalty: Sum of margin violations per environment (E,).
        """
        i_idx, j_idx = self._pair_indices
        num_pairs = len(i_idx)

        if num_pairs == 0:
            return torch.zeros(self.num_envs, device=self.device)

        p_i = positions[:, i_idx]
        p_j = positions[:, j_idx]
        v_i = velocities[:, i_idx]
        v_j = velocities[:, j_idx]

        dp = p_i - p_j
        dv = v_i - v_j

        # CPA barrier value
        h_cpa, tau = self._compute_cpa_barrier(dp, dv)

        # For simple approximation, use distance-based h_dot when tau = 0
        # h = ||dp||^2 - D_s^2, h_dot = 2*dp·dv
        # This is exact when drones are at closest approach (tau=0)
        h_dot = 2.0 * (dp * dv).sum(dim=-1)  # (E, P)

        # CBF margin: m = h_dot + gamma*h
        # Positive margin means safe, negative means approaching unsafely
        margin = h_dot + self.cfg.gamma * h_cpa  # (E, P)

        # Penalty for negative margin
        penalty_per_pair = torch.relu(-margin)  # (E, P)
        penalty = penalty_per_pair.sum(dim=-1)  # (E,)

        return penalty

    def get_cpa_info(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get detailed CPA information for debugging.

        Args:
            positions: Ground-truth positions (E, N, 3).
            velocities: Commanded velocities (E, N, 3).

        Returns:
            Dictionary with CPA metrics:
            - h_cpa: CPA barrier values (E, P)
            - tau: Time to closest approach (E, P)
            - d_cpa: CPA distances (E, P)
            - min_h_cpa: Minimum h across pairs (E,)
        """
        i_idx, j_idx = self._pair_indices

        if len(i_idx) == 0:
            return {
                "h_cpa": torch.zeros(self.num_envs, 0, device=self.device),
                "tau": torch.zeros(self.num_envs, 0, device=self.device),
                "d_cpa": torch.zeros(self.num_envs, 0, device=self.device),
                "min_h_cpa": torch.zeros(self.num_envs, device=self.device),
            }

        p_i = positions[:, i_idx]
        p_j = positions[:, j_idx]
        v_i = velocities[:, i_idx]
        v_j = velocities[:, j_idx]

        dp = p_i - p_j
        dv = v_i - v_j

        h_cpa, tau = self._compute_cpa_barrier(dp, dv)
        d_cpa = torch.sqrt(h_cpa + self.D_s_sq)  # d_CPA = sqrt(h + D_s^2)

        return {
            "h_cpa": h_cpa,
            "tau": tau,
            "d_cpa": d_cpa,
            "min_h_cpa": h_cpa.min(dim=-1).values if h_cpa.shape[1] > 0 else torch.zeros(self.num_envs, device=self.device),
        }

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset state for specified environments.

        The CPA reward shaper is stateless, so this is a no-op.
        Interface provided for consistency with other components.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        pass
