# Stage Q — Questions (008-sysid-replicator)

## Assumptions — confirmed

1. **PX4 SITL CSV location**: Copy CSVs into `iris_ma6/sysid_output/` (local to module).
2. **Tests to match**: All 5 — hover, vel_step_5, vel_step_10, yaw_step, vel_impulse_recovery.
3. **Gimbal out of scope**: Confirmed. Mark for separate ticket.
4. **Quaternion reorder**: CSVs are xyzw, DroneController is wxyz. Reorder on load.
5. **Cascade setpoints**: Use intermediate cascade signals (att_sp vs actual, rate_sp vs actual) in matching — not just final outputs.
6. **Separate file**: New file (e.g. `sysid_replicator.py`), reuse ParallelTuner infrastructure by import. Do not modify `auto_tune.py`.
7. **Step response via sim**: Use DroneController.step_policy() in Isaac Sim env (same as auto_tune.py).
8. **Test durations**: Extend iris_ma6 test durations to match PX4 SITL recording lengths.
9. **Yaw step test**: Add yaw step test to iris_ma6 response generator.

## Architectural decisions — resolved

10. **Scoring**: Hybrid (C) — primary score is timeseries MSE, secondary is metric-based comparison for mismatch table.
11. **Weights**: velocity 60% / attitude 25% / rate 15%. Keep `TuningScoreWeights` as reference only.
12. **Search budget**: 1024 candidates, random search. Upgrade to multi-stage refinement if insufficient.
13. **Velocity component**: Match X velocity and yaw only.
14. **Timebase**: Interpolate PX4 125 Hz → iris_ma6 100 Hz. Future-proofing for real-world drone replication.
15. **Output gains**: Separate file (not overwrite TUNED_CONTROLLER_CFG).
16. **Plots**: PDF, overlay iris_ma6 best vs PX4 SITL on same axes, save to `sysid_output/analysis/`.
