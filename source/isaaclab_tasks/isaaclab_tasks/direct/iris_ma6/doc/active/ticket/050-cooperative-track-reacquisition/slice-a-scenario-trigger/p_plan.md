# Stage P — Implementation Plan (Slice A)

4 vertical slices, each end-to-end testable; metric exists before the scenario that must move it.
Default-off throughout (baseline bit-exact until each flag is confirmed).

## Slice 1: Standalone re-acquisition tracker  [DONE 2026-06-12]
1.1 cfg; 1.2 class+state; 1.3 `_effective_track` (reachable-set-within-FOV gate); 1.4
`_classify_onset_cause`; 1.5 `update` (WRITE, idempotent, deficit/recovery state machine); 1.6
`episode_summary`+`reset`; 1.7 `__init__`/`CONTEXT.md`; 1.8 tests→green.
Test checkpoint: `run_tests.py` — 12/12 PASS (gate, flicker filters, cold/mid/dropout tagging,
peer-assisted precondition, failed-hold, idempotency, team maintenance, reset, batch).

## Slice 2: Wire tracker into env (metric observable on existing scenario)
2.1 cfg field; 2.2 `__init__` instantiate (guard on enable); 2.3 `_gather_reacq_signals`
(bbox_age via get_detection_aoi, GT range, fov_eff_half from camera_effective_hfov/2, v_max);
2.4 `_get_rewards` call update; 2.5 `_reset_idx` summary→extras["log"] + reset.
Test checkpoint: short run with `cooperation_metrics.enable=True` shows `Info / Coop/*` in TB;
enable=False ⇒ scalars absent + reward/obs bit-exact.

## Slice 3: Scenario ceiling raise (manufactures events — the GATE)
3.1 `TrackLossScenarioCfg`; 3.2 `enable_track_loss_scenario` + field; 3.3 `__post_init__` overlay
ceiling onto initial_states/target_controller/curriculum (floor untouched); 3.4 keep dropout
independent per-(env,agent); 3.5 config smoke test.
Test checkpoint: short run flag-on shows `Coop/track_loss_event_rate` ≥ 0.5 qualifying mid-loss
events/episode (cold present), split by cause; flag-off ≡ baseline. [Slice-A GATE]

## Slice 4: Eval-side metric (reuse tracker)
4.1 metric_tracker.py emits track_loss_event_rate, reacq_success_rate, time_to_reacq_mean,
team_track_maintenance (+per-cause) via the tracker; 4.2 evaluate.py feeds per-step signals.
Test checkpoint: evaluate.py on t047 agent_400000.pt (flag on) reports re-acq metrics; expected
LOW reacq_success_rate (C2 baseline for Slice B).
