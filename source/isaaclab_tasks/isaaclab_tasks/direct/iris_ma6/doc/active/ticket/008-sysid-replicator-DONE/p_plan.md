# Stage P — Implementation Plan (008-sysid-replicator)

## Slice 1: PX4 target loading and resampling

End-to-end: load PX4 CSVs → resample → produce TargetTimeseries that can be printed and inspected.

- Step 1.1: Copy PX4 SITL CSVs from `/home/usrg/IsaacPX4/sysid_output/` into `iris_ma6/sysid_output/` (5 drone tests only, exclude gimbal CSVs)
- Step 1.2: Implement `PX4TargetLoader.__init__`, `_load_csv`, `_resample_to_100hz` in `sysid_replicator.py` (AppLauncher boilerplate + argparse + dataclasses at top)
- Step 1.3: Implement `_compute_att_error`, `_compute_rate_error`, `_detect_command_onset`, `get_target`
- Step 1.4: Implement `main()` stub that loads targets and prints summary (test names, durations, sample counts, signal ranges) — verifies the full loading pipeline without Isaac Sim env

**Test checkpoint:** Run `./isaaclab.sh -p .../sysid_replicator.py --sysid-dir .../sysid_output --dry-run` — prints per-test target summary. Verify: 5 tests loaded, all at 100 Hz, att_error/rate_error shapes correct, no NaN.

## Slice 2: iris_ma6 response generation (velocity + hover + new tests)

End-to-end: for a single gain set, run all 5 tests through DroneController and collect timeseries.

- Step 2.1: Implement `SysidReplicator.__init__` (compose ParallelTuner, store targets/weights)
- Step 2.2: Implement `_run_velocity_test` and `_run_hover_test` (delegate to ParallelTuner methods with extended durations matching PX4 data)
- Step 2.3: Implement `_run_yaw_step_test` — yaw_rate_cmd=0.5 for 2s then zero, recording yaw rate + att_error + rate_error
- Step 2.4: Implement `_run_impulse_recovery_test` — vel_cmd=5.0 for 1s then zero, recording vel_x + att_error + rate_error
- Step 2.5: Wire `evaluate_batch` to run all 5 tests, return per-env histories (no scoring yet)
- Step 2.6: Update `main()` to run evaluate_batch with current TUNED_CONTROLLER_CFG gains (single env) and print signal shapes/ranges

**Test checkpoint:** Run with `--num-trials 1` using current gains. Verify: all 5 tests produce histories with expected shapes (T matches PX4 duration at 100 Hz), no NaN, velocity reaches ~5 m/s for vel_step_5.

## Slice 3: Scoring and gain search

End-to-end: score 1024 candidates against PX4 targets, find best match.

- Step 3.1: Implement `_compute_timeseries_mse` — resample/trim to common length, weighted MSE (vel_x/yaw 60%, att_error 25%, rate_error 15%)
- Step 3.2: Implement `_compute_metric_comparison` — extract settling_time, ss_error, damping_ratio from both sides, compute diff_pct
- Step 3.3: Wire scoring into `evaluate_batch` so it returns (scores, histories, metric_comparisons)
- Step 3.4: Update `main()` to run full 1024-candidate sweep, find best, print results summary + mismatch table
- Step 3.5: Save `replicator_metrics.json` to `sysid_output/analysis/`

**Test checkpoint:** Run with `--num-trials 1024`. Verify: best score is finite, mismatch table shows per-metric deltas, at least some metrics improve vs current gains.

## Slice 4: Output — plots, config export, integration

End-to-end: produce all deliverables (PDFs, gain config, re-export).

- Step 4.1: Implement `generate_comparison_plots` — per-test PDF with iris_ma6 best vs PX4 SITL overlay (velocity/yaw, att_error, rate_error rows)
- Step 4.2: Implement `export_px4_matched_config` — write `tuning_results/px4_matched.py` with `PX4_MATCHED_CONTROLLER_CFG`
- Step 4.3: Add `from .px4_matched import PX4_MATCHED_CONTROLLER_CFG` to `tuning_results/__init__.py`
- Step 4.4: Final `main()` wiring — generate plots, export config, print completion summary

**Test checkpoint:** Run full pipeline. Verify: 5 PDF files in `sysid_output/analysis/`, `px4_matched.py` exists and is importable, `__init__.py` re-exports `PX4_MATCHED_CONTROLLER_CFG`.
