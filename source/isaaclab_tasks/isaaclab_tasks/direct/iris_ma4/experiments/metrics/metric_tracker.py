# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric tracking for evaluation rollouts.

Tracks IROS 2026 paper metrics during evaluation:
    1. trace(Sigma_X) — triangulation uncertainty
    2. Triangulation RMSE — localization error vs ground truth
    3. Task success rate — hierarchical: track maintenance AND accuracy
    4. Visibility ratio — mean fraction of agents with valid detection per step
    5. Collision rate — inter-agent collisions per episode
    6. Convergence speed — steps to reach trace threshold
    7. Time-to-first-lock — steps to first valid triangulation
    8. Triangulation valid ratio — time fraction with successful triangulation
    9. Track loss count — valid-to-invalid transitions (fragmentation)
   10. Max track gap — longest consecutive invalid sequence (steps)
   11. Track maintenance rate — fraction of episodes meeting tri_valid threshold

Legacy metric (backward compatibility):
    - task_success_rate_legacy — old flat RMSE-only success definition

Success definition (hierarchical):
    Level 0 — Track Maintenance: tri_valid_ratio >= threshold (default 0.5)
    Level 1 — Localization Accuracy: among valid steps, RMSE < threshold for >= 80%
    Overall success = Level 0 AND Level 1

All metrics are reported as mean +/- std across episodes.
"""

from __future__ import annotations

import math
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
        track_maintenance_threshold: float = 0.5,
    ):
        """Initialize the metric tracker.

        Args:
            num_envs: Number of parallel environments.
            num_agents: Number of agents.
            device: Torch device.
            success_rmse_threshold: RMSE threshold (m) for a step to count as "accurate".
            success_time_fraction: Fraction of valid steps with accurate RMSE required
                for Level 1 (accuracy) success.
            convergence_trace_threshold: trace(Sigma_X) threshold for convergence.
            first_lock_trace_threshold: trace(Sigma_X) threshold for first lock.
            track_maintenance_threshold: Minimum tri_valid_ratio for Level 0
                (track maintenance) success. Default 0.5 = at least 50% of episode
                must have valid triangulation.
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.success_rmse_threshold = success_rmse_threshold
        self.success_time_fraction = success_time_fraction
        self.convergence_trace_threshold = convergence_trace_threshold
        self.first_lock_trace_threshold = first_lock_trace_threshold
        self.track_maintenance_threshold = track_maintenance_threshold

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

        # Track continuity accumulators
        self._prev_tri_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._track_loss_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._current_gap_length = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._max_track_gap = torch.zeros(num_envs, device=device, dtype=torch.long)

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
        self._prev_tri_valid[env_ids] = False
        self._track_loss_count[env_ids] = 0
        self._current_gap_length[env_ids] = 0
        self._max_track_gap[env_ids] = 0

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

        # Visibility: mean fraction of agents with valid bbox per step
        # E.g., with 2 agents: 0.0 (none), 0.5 (one), 1.0 (both)
        # Better for N>2 agents since triangulation needs >=2 valid detections
        agent_visibility_frac = bbox_valid_mask.float().mean(dim=-1)  # [N]
        self._visibility_steps += agent_visibility_frac

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

        # Success: step counts where RMSE < threshold (used for both legacy and Level 1)
        success_this_step = valid & (rmse < self.success_rmse_threshold)
        self._success_steps += success_this_step.float()

        # Track continuity: detect valid→invalid transitions (track loss events)
        track_lost = self._prev_tri_valid & ~valid
        self._track_loss_count += track_lost.long()

        # Track gap: consecutive invalid steps
        self._current_gap_length = torch.where(
            ~valid,
            self._current_gap_length + 1,
            torch.zeros_like(self._current_gap_length),
        )
        self._max_track_gap = torch.max(self._max_track_gap, self._current_gap_length)

        self._prev_tri_valid = valid.clone()

    def record_episode_end(self, env_ids: torch.Tensor) -> None:
        """Record metrics for completed episodes and reset those envs."""
        for idx in env_ids:
            i = idx.item()
            steps = self._step_count[i].item()
            if steps == 0:
                continue

            valid_count = self._trace_valid_count[i].item()
            tri_valid_ratio = valid_count / steps

            # Legacy success: (valid AND accurate steps) / total_steps >= threshold
            legacy_success_frac = self._success_steps[i].item() / steps
            legacy_success = legacy_success_frac >= self.success_time_fraction

            # Hierarchical success
            # Level 0: Track maintenance — tri_valid_ratio >= threshold
            level0_pass = tri_valid_ratio >= self.track_maintenance_threshold

            # Level 1: Accuracy — among valid steps, RMSE < threshold for >= fraction
            # _success_steps counts steps that are both valid AND RMSE < threshold
            level1_frac = self._success_steps[i].item() / max(valid_count, 1)
            level1_pass = level1_frac >= self.success_time_fraction if level0_pass else False

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
                "success": level0_pass and level1_pass,
                "success_legacy": legacy_success,
                "track_maintenance_pass": level0_pass,
                "accuracy_pass": level1_pass,
                "track_loss_count": self._track_loss_count[i].item(),
                "max_track_gap": self._max_track_gap[i].item(),
                "episode_length": steps,
                "tri_valid_ratio": tri_valid_ratio,
            })
        self.reset(env_ids)

    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics across all completed episodes.

        Returns mean, std, median, and clipped (5th–95th percentile) stats
        for each metric, plus metadata.

        Returns:
            Dictionary with ``{metric}_mean``, ``{metric}_std``,
            ``{metric}_median``, ``{metric}_p5``, ``{metric}_p95``,
            ``{metric}_clipped_mean``, ``{metric}_clipped_std`` pairs
            plus metadata.
        """
        if not self._completed_episodes:
            return {"num_episodes": 0}

        n = len(self._completed_episodes)
        eps = self._completed_episodes

        def _mean_std(values: List[float]):
            m = sum(values) / len(values)
            var = sum((v - m) ** 2 for v in values) / len(values)
            return m, math.sqrt(var)

        def _mean_std_positive(values: List[float]):
            pos = [v for v in values if v >= 0]
            if not pos:
                return -1.0, 0.0
            return _mean_std(pos)

        def _robust_stats(values: List[float]):
            """Compute median, percentiles, and clipped (5th-95th) mean/std."""
            sv = sorted(values)
            n_v = len(sv)
            median = sv[n_v // 2] if n_v % 2 == 1 else (sv[n_v // 2 - 1] + sv[n_v // 2]) / 2
            p5 = sv[max(int(n_v * 0.05), 0)]
            p95 = sv[min(int(n_v * 0.95), n_v - 1)]
            clipped = [v for v in values if p5 <= v <= p95]
            if clipped:
                clip_m, clip_s = _mean_std(clipped)
            else:
                clip_m, clip_s = _mean_std(values)
            return median, p5, p95, clip_m, clip_s

        def _robust_stats_positive(values: List[float]):
            """Robust stats excluding negative sentinel values (-1)."""
            pos = [v for v in values if v >= 0]
            if not pos:
                return -1.0, -1.0, -1.0, -1.0, 0.0
            return _robust_stats(pos)

        trace_m, trace_s = _mean_std([e["trace_sigma_mean"] for e in eps])
        rmse_m, rmse_s = _mean_std([e["rmse_mean"] for e in eps])
        vis_m, vis_s = _mean_std([e["visibility_ratio"] for e in eps])
        coll_m, coll_s = _mean_std([e["collision_count"] for e in eps])
        conv_m, conv_s = _mean_std_positive([e["convergence_step"] for e in eps])
        lock_m, lock_s = _mean_std_positive([e["first_lock_step"] for e in eps])
        triv_m, triv_s = _mean_std([e["tri_valid_ratio"] for e in eps])
        loss_m, loss_s = _mean_std([e["track_loss_count"] for e in eps])
        gap_m, gap_s = _mean_std([e["max_track_gap"] for e in eps])

        # Robust stats (median, percentiles, clipped mean/std)
        # Robust stats for key plotting metrics (trace, rmse, visibility, tri_valid)
        trace_med, trace_p5, trace_p95, trace_cm, trace_cs = _robust_stats(
            [e["trace_sigma_mean"] for e in eps])
        rmse_med, rmse_p5, rmse_p95, rmse_cm, rmse_cs = _robust_stats(
            [e["rmse_mean"] for e in eps])
        vis_med, vis_p5, vis_p95, vis_cm, vis_cs = _robust_stats(
            [e["visibility_ratio"] for e in eps])
        triv_med, triv_p5, triv_p95, triv_cm, triv_cs = _robust_stats(
            [e["tri_valid_ratio"] for e in eps])
        # Median-only for secondary metrics
        coll_med, _, _, _, _ = _robust_stats([e["collision_count"] for e in eps])
        conv_med, _, _, _, _ = _robust_stats_positive([e["convergence_step"] for e in eps])
        lock_med, _, _, _, _ = _robust_stats_positive([e["first_lock_step"] for e in eps])
        loss_med, _, _, _, _ = _robust_stats([e["track_loss_count"] for e in eps])
        gap_med, _, _, _, _ = _robust_stats([e["max_track_gap"] for e in eps])

        success_rate = sum(1 for e in eps if e["success"]) / n
        legacy_success_rate = sum(1 for e in eps if e["success_legacy"]) / n
        track_maintenance_rate = sum(1 for e in eps if e["track_maintenance_pass"]) / n
        accuracy_rate = sum(1 for e in eps if e["accuracy_pass"]) / n

        return {
            # --- Mean / Std (original) ---
            "trace_sigma_mean": trace_m,
            "trace_sigma_std": trace_s,
            "triangulation_rmse_mean": rmse_m,
            "triangulation_rmse_std": rmse_s,
            "task_success_rate": success_rate,
            "task_success_rate_legacy": legacy_success_rate,
            "track_maintenance_rate": track_maintenance_rate,
            "accuracy_rate": accuracy_rate,
            "visibility_mean": vis_m,
            "visibility_std": vis_s,
            "collision_rate_mean": coll_m,
            "collision_rate_std": coll_s,
            "convergence_speed_mean": conv_m,
            "convergence_speed_std": conv_s,
            "time_to_first_lock_mean": lock_m,
            "time_to_first_lock_std": lock_s,
            "tri_valid_ratio_mean": triv_m,
            "tri_valid_ratio_std": triv_s,
            "track_loss_count_mean": loss_m,
            "track_loss_count_std": loss_s,
            "max_track_gap_mean": gap_m,
            "max_track_gap_std": gap_s,
            # --- Median ---
            "trace_sigma_median": trace_med,
            "triangulation_rmse_median": rmse_med,
            "visibility_median": vis_med,
            "collision_rate_median": coll_med,
            "convergence_speed_median": conv_med,
            "time_to_first_lock_median": lock_med,
            "tri_valid_ratio_median": triv_med,
            "track_loss_count_median": loss_med,
            "max_track_gap_median": gap_med,
            # --- Percentiles (5th, 95th) ---
            "trace_sigma_p5": trace_p5,
            "trace_sigma_p95": trace_p95,
            "triangulation_rmse_p5": rmse_p5,
            "triangulation_rmse_p95": rmse_p95,
            "visibility_p5": vis_p5,
            "visibility_p95": vis_p95,
            "tri_valid_ratio_p5": triv_p5,
            "tri_valid_ratio_p95": triv_p95,
            # --- Clipped mean/std (5th–95th percentile) ---
            "trace_sigma_clipped_mean": trace_cm,
            "trace_sigma_clipped_std": trace_cs,
            "triangulation_rmse_clipped_mean": rmse_cm,
            "triangulation_rmse_clipped_std": rmse_cs,
            "visibility_clipped_mean": vis_cm,
            "visibility_clipped_std": vis_cs,
            "tri_valid_ratio_clipped_mean": triv_cm,
            "tri_valid_ratio_clipped_std": triv_cs,
            # --- Metadata ---
            "num_episodes": n,
        }

    def get_raw_episodes(self) -> List[Dict[str, float]]:
        """Return all completed episode metrics (for per-episode analysis)."""
        return list(self._completed_episodes)
