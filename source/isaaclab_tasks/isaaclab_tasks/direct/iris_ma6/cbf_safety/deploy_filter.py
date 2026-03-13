# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust Deployment Filter module for deployment-time CBF safety.

Implements simple distance-based CBF with inflated margin for delay robustness.
Uses closed-form halfspace projection (no QP solver needed).

Ref: safety_spec.md Section 2.4, 2.5, 4.2
"""

from __future__ import annotations

import torch
from typing import Dict

from .cbf_cfg import RobustDeploymentFilterCfg


class RobustDeploymentFilter:
    """Simple distance-based CBF with inflated margin.

    Designed for delayed, noisy observations at deployment. Uses the simplest
    possible barrier with a margin that covers worst-case position error from
    communication delay.

    Key properties:
    - Uses delayed positions (latest received, may be stale)
    - Inflated safety distance D_deploy accounts for delay and tracking lag
    - Closed-form halfspace projection (no QP solver)
    - Gauss-Seidel iteration for multiple pairwise constraints

    Usage:
        filter = RobustDeploymentFilter(cfg, num_envs, num_agents, device)
        v_safe, info = filter.filter(v_nom, delayed_positions, delayed_velocities)
    """

    def __init__(
        self,
        cfg: RobustDeploymentFilterCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the deployment filter.

        Args:
            cfg: Configuration for the deployment filter.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Device for tensor computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Compute inflated safety distance
        self.D_deploy = cfg.D_deploy
        self.D_deploy_sq = self.D_deploy**2

        # Pre-allocate diagnostic buffers
        self.filter_active = torch.zeros(
            num_envs, num_agents, dtype=torch.bool, device=device
        )

    def _project_halfspace(
        self,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form projection onto halfspace {v : a^T v >= b}.

        The CBF constraint is linear: a^T v >= b
        If v violates the constraint (a^T v < b), project onto the boundary:
        v_safe = v + ((b - a^T v) / ||a||^2) * a

        Args:
            v: Velocity to project (E, 3).
            a: Constraint normal (E, 3).
            b: Constraint bound (E,).

        Returns:
            v_safe: Projected velocity (E, 3).
            violated: Boolean mask of which environments had violation (E,).
        """
        # Compute constraint value: a^T v
        a_dot_v = (a * v).sum(dim=-1)  # (E,)

        # Check constraint violation
        violated = a_dot_v < b  # (E,) bool

        # Compute projection scale
        a_norm_sq = (a * a).sum(dim=-1) + self.cfg.epsilon  # (E,)
        scale = torch.relu(b - a_dot_v) / a_norm_sq  # (E,)

        # Apply correction only where violated
        correction = scale.unsqueeze(-1) * a  # (E, 3)
        v_safe = v + correction  # (E, 3)

        return v_safe, violated

    def filter(
        self,
        v_nom: torch.Tensor,
        positions: torch.Tensor,
        neighbor_velocities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Hard CBF-QP via iterative Gauss-Seidel projection.

        For each agent i, the CBF constraint with agent j is:
        2 * dp_ij^T * (v_i - v_j) + gamma * h_ij >= 0

        where h_ij = ||p_i - p_j||^2 - D_deploy^2.

        Rearranged as a^T v_i >= b:
        a = 2 * dp_ij
        b = 2 * dp_ij^T * v_j - gamma * h_ij

        Args:
            v_nom: Nominal velocities from policy (E, N, 3).
            positions: Latest received (delayed) positions (E, N, 3).
            neighbor_velocities: Latest received velocities (E, N, 3).
                If None, assumes zero (most conservative).

        Returns:
            v_safe: Filtered safe velocities (E, N, 3).
            info: Dictionary with diagnostics.
        """
        v_safe = v_nom.clone()

        # Reset filter active flags
        self.filter_active.zero_()

        # If no neighbor velocities provided, assume zero (conservative)
        if neighbor_velocities is None:
            neighbor_velocities = torch.zeros_like(v_nom)

        # Gauss-Seidel iteration over all pairwise constraints
        for _ in range(self.cfg.num_iters):
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i == j:
                        continue

                    # Relative position: dp = p_i - p_j
                    dp = positions[:, i] - positions[:, j]  # (E, 3)

                    # Distance-based barrier: h = ||dp||^2 - D_deploy^2
                    h = (dp * dp).sum(dim=-1) - self.D_deploy_sq  # (E,)

                    # Constraint: 2*dp^T*(v_i - v_j) + gamma*h >= 0
                    # Rearranged: (2*dp)^T * v_i >= -2*dp^T*v_j - gamma*h
                    a = 2.0 * dp  # (E, 3)
                    v_j = neighbor_velocities[:, j]  # (E, 3)
                    b = -(a * v_j).sum(dim=-1) - self.cfg.gamma_deploy * h  # (E,)

                    # Project v_i onto the safe halfspace
                    v_safe_i, violated = self._project_halfspace(
                        v_safe[:, i], a, b
                    )
                    v_safe[:, i] = v_safe_i

                    # Track which agents had constraints active
                    self.filter_active[:, i] |= violated

        # Compute diagnostics
        info = {
            "deploy_cbf/filter_active_fraction": self.filter_active.float().mean().item(),
            "deploy_cbf/agents_filtered": self.filter_active.any(dim=0).sum().item(),
        }

        return v_safe, info

    def check_constraint_satisfaction(
        self,
        velocities: torch.Tensor,
        positions: torch.Tensor,
        neighbor_velocities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Check if velocities satisfy all CBF constraints.

        Args:
            velocities: Velocities to check (E, N, 3).
            positions: Positions (E, N, 3).
            neighbor_velocities: Neighbor velocities (E, N, 3).

        Returns:
            satisfied: Boolean tensor (E, N, N) where [e, i, j] indicates
                whether agent i satisfies constraint with agent j in env e.
        """
        if neighbor_velocities is None:
            neighbor_velocities = torch.zeros_like(velocities)

        satisfied = torch.ones(
            self.num_envs, self.num_agents, self.num_agents,
            dtype=torch.bool, device=self.device
        )

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue

                dp = positions[:, i] - positions[:, j]
                h = (dp * dp).sum(dim=-1) - self.D_deploy_sq

                v_i = velocities[:, i]
                v_j = neighbor_velocities[:, j]
                dv = v_i - v_j

                # Constraint: 2*dp^T*dv + gamma*h >= 0
                constraint_val = 2.0 * (dp * dv).sum(dim=-1) + self.cfg.gamma_deploy * h
                satisfied[:, i, j] = constraint_val >= -self.cfg.epsilon

        return satisfied

    def get_barrier_values(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Get current barrier values for all agent pairs.

        Args:
            positions: Positions (E, N, 3).

        Returns:
            h_values: Barrier values (E, N, N). Diagonal is inf.
        """
        # Compute all pairwise barriers
        # h_ij = ||p_i - p_j||^2 - D_deploy^2
        diff = positions[:, :, None, :] - positions[:, None, :, :]  # (E, N, N, 3)
        dist_sq = (diff * diff).sum(dim=-1)  # (E, N, N)
        h_values = dist_sq - self.D_deploy_sq

        # Set diagonal to inf
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        h_values = torch.where(
            eye.unsqueeze(0),
            torch.full_like(h_values, float("inf")),
            h_values
        )

        return h_values

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self.filter_active.zero_()
        else:
            self.filter_active[env_ids] = False
