# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric tracking for iris_ma6 evaluation rollouts.

Tracks paper metrics during evaluation:
    1. trace(Sigma_X) — triangulation uncertainty
    2. Triangulation RMSE — localization error vs ground truth
    3. Task success rate — hierarchical: track maintenance AND accuracy
    4. Visibility ratio — mean fraction of agents with valid detection per step
    5. Collision rate — inter-agent collisions per episode
    6. Convergence speed — steps to reach trace threshold
    7. Time-to-first-lock — steps to first valid triangulation
    8. Triangulation valid ratio — time fraction with successful triangulation
    9. CBF violation rate — fraction of steps with CBF penalty > 0
   10. Min separation — minimum pairwise agent distance per episode
   11. Track loss count — valid-to-invalid transitions (fragmentation)
   12. Max track gap — longest consecutive invalid sequence (steps)

Success definition (hierarchical):
    Level 0 — Track Maintenance: tri_valid_ratio >= threshold (default 0.5)
    Level 1 — Localization Accuracy: among valid steps, RMSE < threshold for >= 80%
    Overall success = Level 0 AND Level 1
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
        self._trace_sq_sum = torch.zeros(num_envs, device=device)
        self._trace_valid_count = torch.zeros(num_envs, device=device)
        self._rmse_sum = torch.zeros(num_envs, device=device)
        self._rmse_sq_sum = torch.zeros(num_envs, device=device)
        self._rmse_valid_count = torch.zeros(num_envs, device=device)
        self._visibility_steps = torch.zeros(num_envs, device=device)
        self._visibility_sq_sum = torch.zeros(num_envs, device=device)
        self._collision_count = torch.zeros(num_envs, device=device)
        self._first_lock_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self._converged_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self._success_steps = torch.zeros(num_envs, device=device)

        # Track continuity accumulators
        self._prev_tri_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._track_loss_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._current_gap_length = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._max_track_gap = torch.zeros(num_envs, device=device, dtype=torch.long)

        # CBF safety metrics (new for iris_ma6)
        self._cbf_violation_count = torch.zeros(num_envs, device=device)
        self._min_separation = torch.full((num_envs,), float("inf"), device=device)

        # Completed episode stats
        self._completed_episodes: List[Dict[str, float]] = []

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset trackers for given environment indices."""
        self._step_count[env_ids] = 0
        self._trace_sum[env_ids] = 0.0
        self._trace_sq_sum[env_ids] = 0.0
        self._trace_valid_count[env_ids] = 0.0
        self._rmse_sum[env_ids] = 0.0
        self._rmse_sq_sum[env_ids] = 0.0
        self._rmse_valid_count[env_ids] = 0.0
        self._visibility_steps[env_ids] = 0.0
        self._visibility_sq_sum[env_ids] = 0.0
        self._collision_count[env_ids] = 0.0
        self._first_lock_step[env_ids] = -1
        self._converged_step[env_ids] = -1
        self._success_steps[env_ids] = 0.0
        self._prev_tri_valid[env_ids] = False
        self._track_loss_count[env_ids] = 0
        self._current_gap_length[env_ids] = 0
        self._max_track_gap[env_ids] = 0
        self._cbf_violation_count[env_ids] = 0.0
        self._min_separation[env_ids] = float("inf")

    def step(
        self,
        trace_sigma: torch.Tensor,
        triangulated_pos: torch.Tensor,
        gt_target_pos: torch.Tensor,
        bbox_valid_mask: torch.Tensor,
        collision_flags: torch.Tensor,
        tri_valid: torch.Tensor,
        cbf_penalty: torch.Tensor | None = None,
        min_pairwise_dist: torch.Tensor | None = None,
    ) -> None:
        """Record one step of metrics.

        Args:
            trace_sigma: [N] or [N, 1] trace of triangulation covariance.
            triangulated_pos: [N, 1, 3] or [N, 3] triangulated target position.
            gt_target_pos: [N, 3] ground truth target position.
            bbox_valid_mask: [N, C] boolean, valid detections per agent.
            collision_flags: [N] boolean, inter-agent collision this step.
            tri_valid: [N] or [N, 1] boolean, triangulation valid this step.
            cbf_penalty: [N] CBF penalty value (>0 means violation).
            min_pairwise_dist: [N] minimum pairwise agent distance.
        """
        self._step_count += 1

        # trace(Sigma_X)
        trace_val = trace_sigma.squeeze(-1) if trace_sigma.dim() > 1 else trace_sigma
        valid = tri_valid.squeeze(-1).bool() if tri_valid.dim() > 1 else tri_valid.bool()
        self._trace_sum += torch.where(valid, trace_val, torch.zeros_like(trace_val))
        self._trace_sq_sum += torch.where(valid, trace_val ** 2, torch.zeros_like(trace_val))
        self._trace_valid_count += valid.float()

        # RMSE
        if triangulated_pos.dim() == 3:
            tri_pos = triangulated_pos.squeeze(1)
        else:
            tri_pos = triangulated_pos
        rmse = (tri_pos - gt_target_pos).norm(dim=-1)
        self._rmse_sum += torch.where(valid, rmse, torch.zeros_like(rmse))
        self._rmse_sq_sum += torch.where(valid, rmse ** 2, torch.zeros_like(rmse))
        self._rmse_valid_count += valid.float()

        # Visibility
        agent_visibility_frac = bbox_valid_mask.float().mean(dim=-1)
        self._visibility_steps += agent_visibility_frac
        self._visibility_sq_sum += agent_visibility_frac ** 2

        # Collisions
        self._collision_count += collision_flags.float()

        # First lock
        first_lock_mask = (
            (self._first_lock_step == -1) & valid & (trace_val < self.first_lock_trace_threshold)
        )
        self._first_lock_step = torch.where(first_lock_mask, self._step_count, self._first_lock_step)

        # Convergence
        converge_mask = (
            (self._converged_step == -1) & valid & (trace_val < self.convergence_trace_threshold)
        )
        self._converged_step = torch.where(converge_mask, self._step_count, self._converged_step)

        # Success steps
        success_this_step = valid & (rmse < self.success_rmse_threshold)
        self._success_steps += success_this_step.float()

        # Track continuity
        track_lost = self._prev_tri_valid & ~valid
        self._track_loss_count += track_lost.long()
        self._current_gap_length = torch.where(
            ~valid,
            self._current_gap_length + 1,
            torch.zeros_like(self._current_gap_length),
        )
        self._max_track_gap = torch.max(self._max_track_gap, self._current_gap_length)

        # CBF metrics (new for iris_ma6)
        if cbf_penalty is not None:
            self._cbf_violation_count += (cbf_penalty > 1e-6).float()
        if min_pairwise_dist is not None:
            self._min_separation = torch.min(self._min_separation, min_pairwise_dist)

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

            # Hierarchical success
            level0_pass = tri_valid_ratio >= self.track_maintenance_threshold
            level1_frac = self._success_steps[i].item() / max(valid_count, 1)
            level1_pass = level1_frac >= self.success_time_fraction if level0_pass else False

            self._completed_episodes.append({
                "trace_sigma_mean": (
                    self._trace_sum[i].item() / max(valid_count, 1)
                ),
                "trace_sigma_sum": self._trace_sum[i].item(),
                "trace_sigma_sq_sum": self._trace_sq_sum[i].item(),
                "trace_valid_count": valid_count,
                "rmse_mean": (
                    self._rmse_sum[i].item() / max(self._rmse_valid_count[i].item(), 1)
                ),
                "rmse_sum": self._rmse_sum[i].item(),
                "rmse_sq_sum": self._rmse_sq_sum[i].item(),
                "rmse_valid_count": self._rmse_valid_count[i].item(),
                "visibility_ratio": self._visibility_steps[i].item() / steps,
                "visibility_sum": self._visibility_steps[i].item(),
                "visibility_sq_sum": self._visibility_sq_sum[i].item(),
                "collision_count": self._collision_count[i].item(),
                "first_lock_step": self._first_lock_step[i].item(),
                "convergence_step": self._converged_step[i].item(),
                "success": level0_pass and level1_pass,
                "track_maintenance_pass": level0_pass,
                "accuracy_pass": level1_pass,
                "track_loss_count": self._track_loss_count[i].item(),
                "max_track_gap": self._max_track_gap[i].item(),
                "episode_length": steps,
                "tri_valid_ratio": tri_valid_ratio,
                "cbf_violation_rate": self._cbf_violation_count[i].item() / steps,
                "min_separation": self._min_separation[i].item(),
            })
        self.reset(env_ids)

    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics across all completed episodes."""
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

        def _pooled_mean_std(sum_key: str, sq_sum_key: str, count_key: str):
            total_sum = sum(e[sum_key] for e in eps)
            total_sq_sum = sum(e[sq_sum_key] for e in eps)
            total_count = sum(e[count_key] for e in eps)
            if total_count == 0:
                return 0.0, 0.0
            mean = total_sum / total_count
            var = total_sq_sum / total_count - mean ** 2
            return mean, math.sqrt(max(var, 0.0))

        def _robust_stats(values: List[float]):
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

        # Step-level metrics
        trace_m, trace_s = _pooled_mean_std("trace_sigma_sum", "trace_sigma_sq_sum", "trace_valid_count")
        rmse_m, rmse_s = _pooled_mean_std("rmse_sum", "rmse_sq_sum", "rmse_valid_count")
        vis_m, vis_s = _pooled_mean_std("visibility_sum", "visibility_sq_sum", "episode_length")

        # Episode-level metrics
        coll_m, coll_s = _mean_std([e["collision_count"] for e in eps])
        conv_m, conv_s = _mean_std_positive([e["convergence_step"] for e in eps])
        lock_m, lock_s = _mean_std_positive([e["first_lock_step"] for e in eps])
        triv_m, triv_s = _mean_std([e["tri_valid_ratio"] for e in eps])
        loss_m, loss_s = _mean_std([e["track_loss_count"] for e in eps])
        gap_m, gap_s = _mean_std([e["max_track_gap"] for e in eps])

        # CBF metrics (new for iris_ma6)
        cbf_viol_m, cbf_viol_s = _mean_std([e["cbf_violation_rate"] for e in eps])
        min_sep_vals = [e["min_separation"] for e in eps if e["min_separation"] < 1e10]
        min_sep_m, min_sep_s = _mean_std(min_sep_vals) if min_sep_vals else (float("inf"), 0.0)

        # Robust stats for key metrics
        trace_med, trace_p5, trace_p95, trace_cm, trace_cs = _robust_stats(
            [e["trace_sigma_mean"] for e in eps])
        rmse_med, rmse_p5, rmse_p95, rmse_cm, rmse_cs = _robust_stats(
            [e["rmse_mean"] for e in eps])
        vis_med, vis_p5, vis_p95, vis_cm, vis_cs = _robust_stats(
            [e["visibility_ratio"] for e in eps])
        triv_med, triv_p5, triv_p95, triv_cm, triv_cs = _robust_stats(
            [e["tri_valid_ratio"] for e in eps])

        success_rate = sum(1 for e in eps if e["success"]) / n
        track_maintenance_rate = sum(1 for e in eps if e["track_maintenance_pass"]) / n
        accuracy_rate = sum(1 for e in eps if e["accuracy_pass"]) / n

        return {
            # --- Mean / Std ---
            "trace_sigma_mean": trace_m,
            "trace_sigma_std": trace_s,
            "triangulation_rmse_mean": rmse_m,
            "triangulation_rmse_std": rmse_s,
            "task_success_rate": success_rate,
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
            # --- CBF metrics (new) ---
            "cbf_violation_rate_mean": cbf_viol_m,
            "cbf_violation_rate_std": cbf_viol_s,
            "min_separation_mean": min_sep_m,
            "min_separation_std": min_sep_s,
            # --- Median ---
            "trace_sigma_median": trace_med,
            "triangulation_rmse_median": rmse_med,
            "visibility_median": vis_med,
            "tri_valid_ratio_median": triv_med,
            # --- Percentiles ---
            "trace_sigma_p5": trace_p5,
            "trace_sigma_p95": trace_p95,
            "triangulation_rmse_p5": rmse_p5,
            "triangulation_rmse_p95": rmse_p95,
            "visibility_p5": vis_p5,
            "visibility_p95": vis_p95,
            "tri_valid_ratio_p5": triv_p5,
            "tri_valid_ratio_p95": triv_p95,
            # --- Clipped mean/std ---
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
