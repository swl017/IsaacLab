# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-(env, agent) tracker for cooperative single-agent track-loss and re-acquisition events.

Ticket 050 (cooperative track re-acquisition), Slice A — instrumentation.

A *peer-assisted deficit* for agent i is a maximal run of steps where agent i has no actionable
track of the target while at least one peer j does. The tracker detects these deficits, tags their
onset cause, and records whether (and how fast) agent i re-acquires the target — the gradient the
later slices (reward / channel) must exploit.

Effective-track signal — a reachable-set-within-FOV validation gate (no staleness time constant):

    d_i(t) = (bbox_i nonempty) AND ( v_max * bbox_age_i  <  k_fov * range_i * tan(fov_eff_half_i) )

At AoI=0 the reachable set is a point and the gate reduces to "is the detection in frame"
(FOV-exit loss); as AoI grows it becomes the probabilistic version (dropout-staleness loss).

Calling contract (see CONTEXT.md §4.4):
- update(...):           WRITE -- advances per-agent state machines. Once per sim step
                          (idempotent within a step via a ``_last_update_time`` guard).
- episode_summary(ids):  READ  -- per-episode aggregate scalars for the given envs.
- reset(ids):            WRITE -- clears state + accumulators for the given envs (all if None).
"""

from __future__ import annotations

import torch

from .cooperation_metrics_cfg import ReacquisitionTrackerCfg

# Per-agent state machine
_IDLE = 0
_DEFICIT = 1
_HOLD = 2

# Onset cause buckets
_CAUSE_COLD = 0      # deficit present at episode start (cold acquisition from deficit)
_CAUSE_FAR = 1       # empty, target in frame but too small -> recovery needs zoom-in
_CAUSE_EDGE = 2      # empty, target out of frame -> recovery needs gimbal slew
_CAUSE_DROPOUT = 3   # nonempty but stale (gate failed by age) -> packet/detection dropout
_CAUSE_FOV = 4       # empty, far/edge not distinguished
_N_CAUSES = 5

_MID_LOSS_CAUSES = (_CAUSE_FAR, _CAUSE_EDGE, _CAUSE_DROPOUT, _CAUSE_FOV)


class ReacquisitionTracker:
    """Detects peer-assisted track-loss / re-acquisition events per (env, agent)."""

    def __init__(
        self,
        cfg: ReacquisitionTrackerCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        step_dt: float,
    ):
        self._cfg = cfg
        self._num_envs = num_envs
        self._num_agents = num_agents
        self._device = device
        self._step_dt = float(step_dt)

        self._tau_min_steps = max(1, int(round(cfg.tau_min_s / self._step_dt)))
        self._tau_hold_steps = max(1, int(round(cfg.tau_hold_s / self._step_dt)))

        self._last_update_time = -1.0e30

        N, A, dev = num_envs, num_agents, device
        # Per-agent state machine + counters
        self._state = torch.zeros(N, A, dtype=torch.long, device=dev)
        self._deficit_len = torch.zeros(N, A, dtype=torch.long, device=dev)
        self._hold_len = torch.zeros(N, A, dtype=torch.long, device=dev)
        self._deficit_qualified = torch.zeros(N, A, dtype=torch.bool, device=dev)
        self._deficit_cause = torch.zeros(N, A, dtype=torch.long, device=dev)
        self._reacq_time = torch.zeros(N, A, dtype=torch.long, device=dev)  # steps onset->reacquire
        self._deficit_dist_onset = torch.zeros(N, A, dtype=torch.float, device=dev)

        # Per-episode accumulators
        self._ep_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self._ep_cause_counts = torch.zeros(N, A, _N_CAUSES, dtype=torch.long, device=dev)
        self._ep_success = torch.zeros(N, A, dtype=torch.long, device=dev)
        self._ep_reacq_step_sum = torch.zeros(N, A, dtype=torch.float, device=dev)
        self._ep_team_track_steps = torch.zeros(N, dtype=torch.long, device=dev)

        # Diagnostics
        self._ep_recov_count = torch.zeros(N, A, dtype=torch.long, device=dev)
        self._ep_recov_dist_delta_sum = torch.zeros(N, A, dtype=torch.float, device=dev)
        self._ep_recov_bearing_align_sum = torch.zeros(N, A, dtype=torch.float, device=dev)
        self._ep_recov_align_count = torch.zeros(N, A, dtype=torch.long, device=dev)
        # Diagnostics captured at the re-acquisition instant (DEFICIT->HOLD), committed on hold-confirm
        self._reacq_distdelta_pending = torch.zeros(N, A, dtype=torch.float, device=dev)
        self._reacq_align_pending = torch.zeros(N, A, dtype=torch.float, device=dev)
        self._reacq_haspeer_pending = torch.zeros(N, A, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ helpers

    def _effective_track(
        self,
        detected_nonempty: torch.Tensor,
        bbox_age: torch.Tensor,
        target_range: torch.Tensor,
        fov_eff_half: torch.Tensor,
        v_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reachable-set-within-FOV gate. Returns (d, gate_fail_by_staleness), both [N, A] bool."""
        reach = v_max.unsqueeze(-1) * bbox_age  # [N, A] metres the target could have moved
        gate = self._cfg.k_fov * target_range * torch.tan(fov_eff_half)  # [N, A]
        gate_ok = reach < gate
        d = detected_nonempty & gate_ok
        gate_fail_staleness = detected_nonempty & (~gate_ok)
        return d, gate_fail_staleness

    def _classify_onset_cause(
        self,
        bbox_empty: torch.Tensor,
        gate_fail_staleness: torch.Tensor,
        target_pixel_inbounds: torch.Tensor | None,
        is_first_step: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-(env, agent) onset cause enum [N, A] long."""
        cause = torch.full_like(self._deficit_cause, _CAUSE_FOV)
        # nonempty but stale -> dropout (mutually exclusive with empty)
        cause = torch.where(gate_fail_staleness, torch.full_like(cause, _CAUSE_DROPOUT), cause)
        if self._cfg.tag_far_edge and target_pixel_inbounds is not None:
            far = bbox_empty & target_pixel_inbounds
            edge = bbox_empty & (~target_pixel_inbounds)
            cause = torch.where(far, torch.full_like(cause, _CAUSE_FAR), cause)
            cause = torch.where(edge, torch.full_like(cause, _CAUSE_EDGE), cause)
        # cold overrides: deficit present at episode start
        cold = is_first_step.unsqueeze(-1).expand_as(cause)
        cause = torch.where(cold, torch.full_like(cause, _CAUSE_COLD), cause)
        return cause

    def _add_cause_counts(self, mask: torch.Tensor, cause: torch.Tensor) -> None:
        """Scatter-add 1 into ``_ep_cause_counts[n, a, cause]`` for entries where mask is True."""
        idx = cause.clamp(0, _N_CAUSES - 1).unsqueeze(-1)  # [N, A, 1]
        onehot = torch.zeros_like(self._ep_cause_counts)
        onehot.scatter_(2, idx, 1)
        self._ep_cause_counts += onehot * mask.long().unsqueeze(-1)

    def _nearest_peer_distance(self, agent_pos_w: torch.Tensor) -> torch.Tensor:
        """Min distance from each agent to any peer. agent_pos_w [N, A, 3] -> [N, A]."""
        diff = agent_pos_w.unsqueeze(2) - agent_pos_w.unsqueeze(1)  # [N, A, A, 3]
        dist = diff.norm(dim=-1)  # [N, A, A]
        eye = torch.eye(self._num_agents, dtype=torch.bool, device=self._device)
        dist = dist.masked_fill(eye, float("inf"))
        return dist.min(dim=2).values  # [N, A]

    def _bearing_alignment(
        self, agent_bearing_w: torch.Tensor, d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Best bearing alignment of each agent to ANY peer that currently has a valid track.

        For agent i: ``max_{j != i, d_j} cos(bearing_i, bearing_j)``. Returns (align[N, A],
        has_valid_peer[N, A]); align is 0 where the agent has no valid peer.
        """
        b = torch.nn.functional.normalize(agent_bearing_w, dim=-1)  # [N, A, 3]
        cos = torch.einsum("nik,njk->nij", b, b)  # [N, A(i), A(j)]
        eye = torch.eye(self._num_agents, dtype=torch.bool, device=self._device)
        valid_peer = d.unsqueeze(1).expand(-1, self._num_agents, -1) & (~eye)  # [N, A(i), A(j)]
        cos = cos.masked_fill(~valid_peer, float("-inf"))
        align = cos.max(dim=2).values  # [N, A]
        has_peer = valid_peer.any(dim=2)  # [N, A]
        align = torch.where(has_peer, align, torch.zeros_like(align))
        return align, has_peer

    # ------------------------------------------------------------------ WRITE

    def update(
        self,
        *,
        detected_nonempty: torch.Tensor,  # [N, A] bool
        bbox_age: torch.Tensor,           # [N, A] seconds
        target_range: torch.Tensor,       # [N, A] metres
        fov_eff_half: torch.Tensor,       # [N, A] radians (half-FOV)
        v_max: torch.Tensor,              # [N] m/s
        t: float,
        bbox_empty: torch.Tensor | None = None,
        target_pixel_inbounds: torch.Tensor | None = None,
        agent_pos_w: torch.Tensor | None = None,
        agent_bearing_w: torch.Tensor | None = None,
    ) -> None:
        """Advance the per-agent deficit/recovery state machines by one step.

        WRITE — call exactly once per sim step. Idempotent within a step.
        """
        t = float(t)
        if abs(t - self._last_update_time) < 1e-6:
            return
        self._last_update_time = t

        d, gate_fail_staleness = self._effective_track(
            detected_nonempty, bbox_age, target_range, fov_eff_half, v_max
        )
        if bbox_empty is None:
            bbox_empty = ~detected_nonempty

        team_count = d.long().sum(dim=1)  # [N]
        peer_has = (team_count.unsqueeze(-1) - d.long()) >= 1  # [N, A]
        in_pa_deficit = (~d) & peer_has

        is_first_step = self._ep_steps == 0  # [N]
        self._ep_team_track_steps += (team_count >= 2).long()

        st = self._state
        idle0 = st == _IDLE
        def0 = st == _DEFICIT
        hold0 = st == _HOLD
        next_state = st.clone()

        cause_new = self._classify_onset_cause(
            bbox_empty, gate_fail_staleness, target_pixel_inbounds, is_first_step
        )
        peer_dist = (
            self._nearest_peer_distance(agent_pos_w)
            if (self._cfg.record_diagnostics and agent_pos_w is not None)
            else None
        )
        if self._cfg.record_diagnostics and agent_bearing_w is not None:
            align_vals, align_haspeer = self._bearing_alignment(agent_bearing_w, d)
        else:
            align_vals, align_haspeer = None, None

        # ---- DEFICIT transitions ----
        recovered = def0 & d
        peer_gone = def0 & (~d) & (~peer_has)
        stay_def = def0 & (~d) & peer_has

        next_state = torch.where(recovered, torch.full_like(st, _HOLD), next_state)
        self._hold_len = torch.where(recovered, torch.ones_like(self._hold_len), self._hold_len)
        self._reacq_time = torch.where(recovered, self._deficit_len, self._reacq_time)
        # Capture recovery-instant diagnostics into pending buffers (committed on hold-confirm)
        if peer_dist is not None:
            self._reacq_distdelta_pending = torch.where(
                recovered, self._deficit_dist_onset - peer_dist, self._reacq_distdelta_pending
            )
        if align_vals is not None:
            self._reacq_align_pending = torch.where(recovered, align_vals, self._reacq_align_pending)
            self._reacq_haspeer_pending = torch.where(recovered, align_haspeer, self._reacq_haspeer_pending)
        next_state = torch.where(peer_gone, torch.full_like(st, _IDLE), next_state)
        self._deficit_len = torch.where(stay_def, self._deficit_len + 1, self._deficit_len)

        # ---- HOLD transitions ----
        held = hold0 & d
        hold_lost = hold0 & (~d)
        self._hold_len = torch.where(held, self._hold_len + 1, self._hold_len)
        confirmed = held & (self._hold_len >= self._tau_hold_steps)
        succ = confirmed & self._deficit_qualified
        self._ep_success += succ.long()
        self._ep_reacq_step_sum += torch.where(
            succ, self._reacq_time.float(), torch.zeros_like(self._ep_reacq_step_sum)
        )
        if self._cfg.record_diagnostics:
            self._ep_recov_count += succ.long()
            self._ep_recov_dist_delta_sum += torch.where(
                succ, self._reacq_distdelta_pending, torch.zeros_like(self._ep_recov_dist_delta_sum)
            )
            align_commit = succ & self._reacq_haspeer_pending
            self._ep_recov_align_count += align_commit.long()
            self._ep_recov_bearing_align_sum += torch.where(
                align_commit, self._reacq_align_pending, torch.zeros_like(self._ep_recov_bearing_align_sum)
            )
        next_state = torch.where(confirmed, torch.full_like(st, _IDLE), next_state)

        hl_peer = hold_lost & peer_has
        hl_nopeer = hold_lost & (~peer_has)
        next_state = torch.where(hl_peer, torch.full_like(st, _DEFICIT), next_state)
        next_state = torch.where(hl_nopeer, torch.full_like(st, _IDLE), next_state)

        # ---- IDLE -> DEFICIT ----
        enter = idle0 & in_pa_deficit
        next_state = torch.where(enter, torch.full_like(st, _DEFICIT), next_state)

        # Fresh deficit init (newly entered from IDLE or from a failed HOLD recovery)
        fresh = enter | hl_peer
        self._deficit_len = torch.where(fresh, torch.ones_like(self._deficit_len), self._deficit_len)
        self._deficit_qualified = torch.where(
            fresh, torch.zeros_like(self._deficit_qualified), self._deficit_qualified
        )
        self._deficit_cause = torch.where(fresh, cause_new, self._deficit_cause)
        if peer_dist is not None:
            self._deficit_dist_onset = torch.where(fresh, peer_dist, self._deficit_dist_onset)

        # ---- qualification pass (uniform over all agents currently in DEFICIT) ----
        in_def_now = next_state == _DEFICIT
        newly_qual = in_def_now & (~self._deficit_qualified) & (self._deficit_len >= self._tau_min_steps)
        if bool(newly_qual.any()):
            self._add_cause_counts(newly_qual, self._deficit_cause)
            self._deficit_qualified = torch.where(
                newly_qual, torch.ones_like(self._deficit_qualified), self._deficit_qualified
            )

        self._state = next_state
        self._ep_steps += 1

    # ------------------------------------------------------------------ READ

    def episode_summary(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-episode aggregate scalars (0-dim tensors) for ``env_ids``, ready for extras["log"]."""
        e = env_ids
        zero = torch.zeros((), device=self._device)
        steps = self._ep_steps[e].clamp(min=1).float()
        counts = self._ep_cause_counts[e].float()  # [M, A, 5]

        per_env_cause = counts.sum(dim=1)  # [M, 5] summed over agents
        mid = per_env_cause[:, list(_MID_LOSS_CAUSES)].sum(dim=1)  # [M]
        cold = per_env_cause[:, _CAUSE_COLD]  # [M]
        qual_total = per_env_cause.sum(dim=1)  # [M] all causes
        success = self._ep_success[e].sum(dim=1).float()  # [M]
        reacq_sum = self._ep_reacq_step_sum[e].sum(dim=1)  # [M]
        team_maint = self._ep_team_track_steps[e].float() / steps  # [M]

        qsum = qual_total.sum()
        ssum = success.sum()
        out = {
            "Coop/track_loss_event_rate": mid.mean(),
            "Coop/cold_deficit_rate": cold.mean(),
            "Coop/reacq_success_rate": (success.sum() / qsum) if bool(qsum > 0) else zero,
            "Coop/time_to_reacq_mean": (reacq_sum.sum() / ssum * self._step_dt) if bool(ssum > 0) else zero,
            "Coop/team_track_maintenance": team_maint.mean(),
            "Coop/event_rate_far": per_env_cause[:, _CAUSE_FAR].mean(),
            "Coop/event_rate_edge": per_env_cause[:, _CAUSE_EDGE].mean(),
            "Coop/event_rate_dropout": per_env_cause[:, _CAUSE_DROPOUT].mean(),
            "Coop/event_rate_fov": per_env_cause[:, _CAUSE_FOV].mean(),
        }
        if self._cfg.record_diagnostics:
            rc = self._ep_recov_count[e].sum().float()
            ac = self._ep_recov_align_count[e].sum().float()
            out["Coop/reacq_dist_delta_mean"] = (
                self._ep_recov_dist_delta_sum[e].sum() / rc if bool(rc > 0) else zero
            )
            out["Coop/reacq_bearing_align_mean"] = (
                self._ep_recov_bearing_align_sum[e].sum() / ac if bool(ac > 0) else zero
            )
        return out

    def episode_values(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-env RAW episode quantities (not reduced), for correct cross-episode aggregation in
        the eval pipeline. Each value is a [M] tensor over ``env_ids``. Call before ``reset``.
        """
        e = env_ids
        counts = self._ep_cause_counts[e].float()  # [M, A, 5]
        per_env_cause = counts.sum(dim=1)  # [M, 5]
        return {
            "mid_events": per_env_cause[:, list(_MID_LOSS_CAUSES)].sum(dim=1),
            "cold_events": per_env_cause[:, _CAUSE_COLD],
            "qual_total": per_env_cause.sum(dim=1),
            "success": self._ep_success[e].sum(dim=1).float(),
            "reacq_step_sum": self._ep_reacq_step_sum[e].sum(dim=1),
            "team_track_steps": self._ep_team_track_steps[e].float(),
            "ep_steps": self._ep_steps[e].float(),
            "far": per_env_cause[:, _CAUSE_FAR],
            "edge": per_env_cause[:, _CAUSE_EDGE],
            "dropout": per_env_cause[:, _CAUSE_DROPOUT],
            "fov": per_env_cause[:, _CAUSE_FOV],
            "recov_count": self._ep_recov_count[e].sum(dim=1).float(),
            "recov_dist_delta_sum": self._ep_recov_dist_delta_sum[e].sum(dim=1),
            "recov_align_count": self._ep_recov_align_count[e].sum(dim=1).float(),
            "recov_align_sum": self._ep_recov_bearing_align_sum[e].sum(dim=1),
        }

    # ------------------------------------------------------------------ WRITE (reset)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear state machine + per-episode accumulators for ``env_ids`` (all envs if None)."""
        if env_ids is None:
            env_ids = slice(None)
        self._state[env_ids] = _IDLE
        self._deficit_len[env_ids] = 0
        self._hold_len[env_ids] = 0
        self._deficit_qualified[env_ids] = False
        self._deficit_cause[env_ids] = 0
        self._reacq_time[env_ids] = 0
        self._deficit_dist_onset[env_ids] = 0.0
        self._ep_steps[env_ids] = 0
        self._ep_cause_counts[env_ids] = 0
        self._ep_success[env_ids] = 0
        self._ep_reacq_step_sum[env_ids] = 0.0
        self._ep_team_track_steps[env_ids] = 0
        self._ep_recov_count[env_ids] = 0
        self._ep_recov_dist_delta_sum[env_ids] = 0.0
        self._ep_recov_bearing_align_sum[env_ids] = 0.0
        self._ep_recov_align_count[env_ids] = 0
        self._reacq_distdelta_pending[env_ids] = 0.0
        self._reacq_align_pending[env_ids] = 0.0
        self._reacq_haspeer_pending[env_ids] = False
