# PR — Ticket 050 / Slice A: Cooperation trigger + re-acquisition instrumentation

Prerequisite slice of the cooperative-track-reacquisition arc. Makes the scenario manufacture
single-agent peer-assisted track-loss and instruments re-acquisition. NO reward/obs change.
Verified: events occur at 2.05/episode (gate >=0.5); C2 baseline reacq_success_rate=0.385.

## Changes by slice
- **Slice 1 — cooperation_metrics/ (new module):** ReacquisitionTracker, per-(env,agent)
  IDLE/DEFICIT/HOLD state machine. Effective-track = reachable-set-within-FOV gate
  (v_max*bbox_age < k*R*tan(fov_half)); no staleness constant; unifies FOV-exit + dropout loss.
  Coop/* episode scalars + episode_values() per-env raw counts. cfg/__init__/CONTEXT/tests.
- **Slice 2 — env wiring (iris_ma_env6_test{,_cfg}.py):** cooperation_metrics cfg field
  (enable=False default); __init__ instantiate; _gather_reacq_signals(); _get_rewards update();
  _reset_idx summary->extras["log"] + reset. Flag-off = bit-exact.
- **Slice 3 — scenario ceiling raise (track_loss_scenario_cfg.py + env):** TrackLossScenarioCfg
  + enable_track_loss_scenario. Overlay applied in env __init__ BEFORE super() (NOT __post_init__,
  which Hydra from_dict bypasses). Ceiling-only; floors intact (= t045/t046 difficulty-revert).
- **Slice 4 — eval (experiments/evaluate.py):** --coop_metrics/--track_loss_scenario flags;
  per-episode buffer (collect_episode_values, default-off) aggregated into results["reacquisition"].

## Deviation from plan
Slice-3 overlay moved __post_init__ -> env __init__ (Hydra from_dict bypass). See
[[project_hydra_post_init_bypass]].

## Files
New: cooperation_metrics/ (module+tests), track_loss_scenario_cfg.py
Modified: iris_ma_env6_test.py, iris_ma_env6_test_cfg.py, experiments/evaluate.py

## Verification
- run_tests.py: 14/14. smoke_env_wiring.py + smoke_scenario_gate.py: PASS.
- Eval (t048 wide): track_loss_event_rate=2.05 (gate PASS), reacq_success_rate=0.385 (C2 baseline).
  See eval_baseline_results.md.

## Deferred (not this PR)
Reward (Slice B); peer-bearing channel + EKF (Slice C); architecture (Slice D); far/edge split
(needs target_pixel_inbounds); privileged-critic GT-target.
