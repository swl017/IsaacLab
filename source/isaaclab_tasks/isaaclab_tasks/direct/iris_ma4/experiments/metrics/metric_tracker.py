# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric tracking for evaluation rollouts.

Tracks the 7 IROS 2026 paper metrics during evaluation:
    1. trace(Sigma_X) — triangulation uncertainty
    2. Triangulation RMSE — localization error vs ground truth
    3. Task success rate — fraction of episodes meeting quality threshold
    4. Visibility ratio — fraction of timesteps with all-agent detection
    5. Collision rate — inter-agent collisions per episode
    6. Convergence speed — steps to reach trace threshold
    7. Time-to-first-lock — steps to first valid triangulation
"""

from __future__ import annotations

from typing import Dict, List

import torch


class MetricTracker:
    """Tracks per-step metrics during evaluation rollouts and computes aggregates.

    Usage::

        tracker = MetricTracker(num_envs=256, num_agents=2, device="cuda:0")

        obs, info = env.reset()
        while completed < target_episodes:
            obs, rewards, terminated, truncated, info = env.step(actions)
            tracker.step(trace_sigma=..., triangulated_pos=..., ...)

            done_envs = torch.where(terminated | truncated)[0]
            if done_envs.numel() > 0:
                tracker.record_episode_end(done_envs)
                completed += done_envs.numel()

        results = tracker.compute_final_metrics()
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        success_rmse_threshold: float = 2.0,
        success_time_fraction: float = 0.8,
        convergence_trace_threshold: float = 10.0,
        first_lock_trace_threshold: float = 50.0,
    ):
        """Initialize the metric tracker.

        Args:
            num_envs: Number of parallel environments.
            num_agents: Number of agents.
            device: Torch device.
            success_rmse_threshold: RMSE threshold (m) for a step to count as "successful".
            success_time_fraction: Fraction of successful steps required for episode success.
            convergence_trace_threshold: trace(Sigma_X) threshold for convergence.
            first_lock_trace_threshold: trace(Sigma_X) threshold for first lock.
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.success_rmse_threshold = success_rmse_threshold
        self.success_time_fraction = success_time_fraction
        self.convergence_trace_threshold = convergence_trace_threshold
        self.first_lock_trace_threshold = first_lock_trace_threshold

        # Per-env accumulators (reset on episode end)
        self._step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._trace_sum = torch.zeros(num_envs, device=device)
        self._trace_valid_count = torch.zeros(num_envs, device=device)
        self._rmse_sum = torch.zeros(num_envs, device=device)
        self._rmse_valid_count = torch.zeros(num_envs, device=device)
        self._visibility_steps = torch.zeros(num_envs, device=device)
        self._collision_count = torch.zeros(num_envs, device=device)
        self._first_lock_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self._converged_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self._success_steps = torch.zeros(num_envs, device=device)

        # Completed episode stats
        self._completed_episodes: List[Dict[str, float]] = []

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset trackers for given environment indices."""
        self._step_count[env_ids] = 0
        self._trace_sum[env_ids] = 0.0
        self._trace_valid_count[env_ids] = 0.0
        self._rmse_sum[env_ids] = 0.0
        self._rmse_valid_count[env_ids] = 0.0
        self._visibility_steps[env_ids] = 0.0
        self._collision_count[env_ids] = 0.0
        self._first_lock_step[env_ids] = -1
        self._converged_step[env_ids] = -1
        self._success_steps[env_ids] = 0.0

    def step(
        self,
        trace_sigma: torch.Tensor,
        triangulated_pos: torch.Tensor,
        gt_target_pos: torch.Tensor,
        bbox_valid_mask: torch.Tensor,
        collision_flags: torch.Tensor,
        tri_valid: torch.Tensor,
    ) -> None:
        """Record one step of metrics.

        Args:
            trace_sigma: [N, 1] trace of triangulation covariance.
            triangulated_pos: [N, 1, 3] or [N, 3] triangulated target position.
            gt_target_pos: [N, 3] ground truth target position.
            bbox_valid_mask: [N, C] boolean, valid detections per agent.
            collision_flags: [N] boolean, inter-agent collision this step.
            tri_valid: [N, 1] boolean, triangulation valid this step.
        """
        self._step_count += 1

        # trace(Sigma_X)
        trace_val = trace_sigma.squeeze(-1)  # [N]
        valid = tri_valid.squeeze(-1).bool()  # [N]
        self._trace_sum += torch.where(valid, trace_val, torch.zeros_like(trace_val))
        self._trace_valid_count += valid.float()

        # RMSE
        if triangulated_pos.dim() == 3:
            tri_pos = triangulated_pos.squeeze(1)  # [N, 3]
        else:
            tri_pos = triangulated_pos
        rmse = (tri_pos - gt_target_pos).norm(dim=-1)  # [N]
        self._rmse_sum += torch.where(valid, rmse, torch.zeros_like(rmse))
        self._rmse_valid_count += valid.float()

        # Visibility: all agents have valid bbox
        all_visible = bbox_valid_mask.all(dim=-1)  # [N]
        self._visibility_steps += all_visible.float()

        # Collisions
        self._collision_count += collision_flags.float()

        # First lock: first step where trace < threshold and valid
        first_lock_mask = (
            (self._first_lock_step == -1) & valid & (trace_val < self.first_lock_trace_threshold)
        )
        self._first_lock_step = torch.where(first_lock_mask, self._step_count, self._first_lock_step)

        # Convergence: first step where trace < convergence threshold
        converge_mask = (
            (self._converged_step == -1) & valid & (trace_val < self.convergence_trace_threshold)
        )
        self._converged_step = torch.where(converge_mask, self._step_count, self._converged_step)

        # Success: step counts where RMSE < threshold
        success_this_step = valid & (rmse < self.success_rmse_threshold)
        self._success_steps += success_this_step.float()

    def record_episode_end(self, env_ids: torch.Tensor) -> None:
        """Record metrics for completed episodes and reset those envs."""
        for idx in env_ids:
            i = idx.item()
            steps = self._step_count[i].item()
            if steps == 0:
                continue

            valid_count = self._trace_valid_count[i].item()
            self._completed_episodes.append({
                "trace_sigma_mean": (
                    self._trace_sum[i].item() / max(valid_count, 1)
                ),
                "rmse_mean": (
                    self._rmse_sum[i].item() / max(self._rmse_valid_count[i].item(), 1)
                ),
                "visibility_ratio": self._visibility_steps[i].item() / steps,
                "collision_count": self._collision_count[i].item(),
                "first_lock_step": self._first_lock_step[i].item(),
                "convergence_step": self._converged_step[i].item(),
                "success": (
                    (self._success_steps[i].item() / steps) >= self.success_time_fraction
                ),
                "episode_length": steps,
                "tri_valid_ratio": valid_count / steps,
            })
        self.reset(env_ids)

    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics across all completed episodes.

        Returns:
            Dictionary with the 7 paper metrics plus metadata.
        """
        if not self._completed_episodes:
            return {"num_episodes": 0}

        n = len(self._completed_episodes)
        return {
            "trace_sigma_mean": sum(e["trace_sigma_mean"] for e in self._completed_episodes) / n,
            "triangulation_rmse": sum(e["rmse_mean"] for e in self._completed_episodes) / n,
            "task_success_rate": sum(1 for e in self._completed_episodes if e["success"]) / n,
            "visibility_ratio": sum(e["visibility_ratio"] for e in self._completed_episodes) / n,
            "collision_rate": sum(e["collision_count"] for e in self._completed_episodes) / n,
            "convergence_speed": _mean_positive(
                [e["convergence_step"] for e in self._completed_episodes]
            ),
            "time_to_first_lock": _mean_positive(
                [e["first_lock_step"] for e in self._completed_episodes]
            ),
            "tri_valid_ratio": sum(e["tri_valid_ratio"] for e in self._completed_episodes) / n,
            "num_episodes": n,
        }

    def get_raw_episodes(self) -> List[Dict[str, float]]:
        """Return all completed episode metrics (for per-episode analysis)."""
        return list(self._completed_episodes)


def _mean_positive(values: list) -> float:
    """Mean of positive values only (skip -1 sentinel)."""
    positive = [v for v in values if v >= 0]
    return sum(positive) / len(positive) if positive else -1.0
