# cooperation_metrics — Module Context

## Purpose
Instrument single-agent, peer-assisted **track-loss** and **re-acquisition** events (ticket 050,
Slice A). Produces the gate metric proving the cooperation trigger exists, and the baseline
re-acquisition numbers the later slices must improve. Measurement-only — no obs/reward/dynamics change.

## Inputs (per sim step, via `update`)
- `detected_nonempty` [N, A] bool — per-agent valid (non-empty) detection (reward-path signal).
- `bbox_age` [N, A] s — detection age-of-information (delay system AoI).
- `target_range` [N, A] m — range to target (privileged GT range recommended; metric-only).
- `fov_eff_half` [N, A] rad — effective half-FOV (from zoom).
- `v_max` [N] m/s — curriculum max target speed.
- optional: `bbox_empty`, `target_pixel_inbounds` (far/edge tagging), `agent_pos_w`,
  `agent_bearing_w` (diagnostics — per-agent bearing; alignment reduces over ANY valid peer).

## Outputs (per episode, via `episode_summary(env_ids)`)
Dict of 0-dim scalar tensors under the `Coop/` namespace: `track_loss_event_rate` (GATE metric),
`cold_deficit_rate`, `reacq_success_rate`, `time_to_reacq_mean`, `team_track_maintenance`,
per-cause `event_rate_{far,edge,dropout,fov}`, and diagnostics `reacq_dist_delta_mean`,
`reacq_bearing_align_mean`. Routed to `extras["log"]` (training TB) and to `experiments/` (eval).

## Effective-track gate
`d_i = (bbox nonempty) AND ( v_max·bbox_age < k_fov·range·tan(fov_eff_half) )`. At AoI=0 reduces to
"is the detection in frame" (FOV-exit); as AoI grows, the probabilistic dropout-staleness version.

## Dependencies
None (standalone; torch + `isaaclab.utils.configclass` only). Consumed by `iris_ma_env6_test.py`
(`_get_rewards`, `_reset_idx`) and `experiments/` (eval).

## Key Files
- `cooperation_metrics_cfg.py` — `ReacquisitionTrackerCfg`.
- `reacquisition_tracker.py` — `ReacquisitionTracker` (per-agent IDLE/DEFICIT/HOLD state machine).
- `tests/run_tests.py` — standalone suite.

## Calling Contract (§4.4)
- `update(...)`: **WRITE**, once per sim step, idempotent within a step (`_last_update_time` guard).
  Place in `_get_rewards` after `_per_agent_bbox_nonempty` is computed.
- `episode_summary(env_ids)`: **READ**, safe to call multiple times; call in `_reset_idx`.
- `reset(env_ids)`: **WRITE**, call in `_reset_idx` after `episode_summary`.
- Stateful invariant: a peer-assisted deficit is a maximal run where agent i has no track while a
  peer does; deficits count only past `tau_min_s`; re-acquisition counts only if held past `tau_hold_s`.
