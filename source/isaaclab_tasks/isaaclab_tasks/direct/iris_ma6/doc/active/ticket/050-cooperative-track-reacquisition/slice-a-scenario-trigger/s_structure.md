# Stage S — Structure Outline (Slice A)

## New files
- `cooperation_metrics/__init__.py` — exports `ReacquisitionTrackerCfg`, `ReacquisitionTracker`.
- `cooperation_metrics/CONTEXT.md` — ICM L2 contract (+ §4.4 calling contract).
- `cooperation_metrics/cooperation_metrics_cfg.py`
  - `@configclass ReacquisitionTrackerCfg`: enable, k_fov, use_gt_range, tau_min_s, tau_hold_s,
    tag_far_edge, record_diagnostics.
- `cooperation_metrics/reacquisition_tracker.py`
  - `ReacquisitionTracker.__init__(cfg, num_envs, num_agents, device, step_dt)`
  - `update(*, detected_nonempty[N,A], bbox_age[N,A], target_range[N,A], fov_eff_half[N,A],
    v_max[N], t, bbox_empty=None, target_pixel_inbounds=None, agent_pos_w=None,
    ego_bearing_w=None, peer_bearing_w=None) -> None`  [WRITE once/step, idempotent]
  - `episode_summary(env_ids) -> dict[str, 0-dim Tensor]`  [READ] — `Coop/*`
  - `reset(env_ids=None) -> None`  [WRITE]
  - internal: `_effective_track`, `_classify_onset_cause`, `_add_cause_counts`,
    `_nearest_peer_distance`, `_bearing_alignment`; per-agent IDLE/DEFICIT/HOLD state machine.
- `cooperation_metrics/tests/`: `__init__.py`, `run_tests.py`, `README.md`, `test_result.txt`, `error_log.txt`.
- `track_loss_scenario_cfg.py` (Slice 3): `@configclass TrackLossScenarioCfg` ceiling overrides.

## Modified files (Slices 2-4)
- `iris_ma_env6_test_cfg.py` [add] `cooperation_metrics` field (S2); `enable_track_loss_scenario` +
  `track_loss_scenario` + `__post_init__` overlay (S3).
- `iris_ma_env6_test.py` [modify] `__init__` instantiate tracker; `_gather_reacq_signals` (add);
  `_get_rewards` call `update`; `_reset_idx` `episode_summary`→`extras["log"]` + `reset` (S2).
- `experiments/metrics/metric_tracker.py` [add] reuse tracker for eval re-acq fields (S4).
- `curriculum/curriculum_cfg.py` — no structural change; fields overlaid by TrackLossScenarioCfg.
- `ARCHITECTURE.md` [add] cooperation_metrics node.

Note: eval (S4) reuses `ReacquisitionTracker` (one metric definition, two call sites) — hence the
tracker is a standalone module, not env-private.
