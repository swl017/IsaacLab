## Ticket: Sysid replicator — tune iris_ma6 controller gains to match PX4 SITL response

**What**: Build a replicator that sweeps iris_ma6 DroneController gain sets, collects step responses, and scores each against the PX4 SITL response recorded by ticket-007. The output is the gain set whose iris_ma6 response best matches PX4 SITL — closing the dynamics gap identified in the sim-to-sim comparison.

**Why**: Ticket-007 showed 7/8 metrics exceed the ±20% mismatch threshold between iris_ma6's DroneController and PX4 SITL (settling time 55% off, SS error 50% off, hover drift 58-73% off). The policy was trained on iris_ma6's dynamics — if those dynamics don't match PX4, the policy will fail at deployment. Rather than tuning PX4 to match iris_ma6 (impractical for sim-to-real), we tune iris_ma6 to match PX4 SITL. This is a new version of `auto_tune.py` where the scoring target is PX4 SITL data instead of hand-crafted criteria.

**Evidence**:
- Ticket-007 comparison table: hover_drift +58%, vel_5_settling +55%, vel_5_ss_error +50%, damping_ratio -91%
- PX4 SITL response CSVs with cascade setpoints (att_sp, rate_sp, thrust_sp) available in `sysid_output/`
- Existing `auto_tune.py` already supports parallel gain sweep with real DroneController + step response extraction
- The gap is too large for gain randomization (±20%) to cover — need to shift the nominal gains

**Scope**:
1. **PX4 SITL target loader**: Load the recorded PX4 SITL CSVs (from ticket-007 sysid_node output) as the target response to match
2. **iris_ma6 response generator**: For each candidate gain set, run DroneController.step_policy() on the same step commands as sysid_node (hover, vel_step_5, vel_step_10, yaw_step) and collect timeseries (velocity, attitude, rate vs time). Reuse auto_tune.py's parallel env infrastructure.
3. **Matching score function**: Score each gain set by how well iris_ma6's response matches PX4 SITL's response. Metrics: timeseries MSE on velocity, attitude error (actual vs setpoint), rate error (actual vs setpoint). Weight velocity tracking highest.
4. **Gain search**: Random search over the same parameter space as auto_tune.py (12 gains: Kp/Ki_vel, Kp_att, Kp/Ki/Kd_rate). Output: best-matching gain set + comparison plots (iris_ma6 best vs PX4 SITL overlay).
5. **Output**: Updated `TUNED_CONTROLLER_CFG` with PX4-matched gains, comparison PDFs, mismatch table showing improvement.

**Scope boundary**:
- Do NOT modify PX4 parameters or the sysid_node
- Do NOT modify the RL policy or reward function
- Do NOT change the controller architecture (keep 4-loop cascade)
- Do NOT change the motor dynamics model or aerodynamic model — tune gains only
- Gimbal matching is out of scope (gimbal is controlled by `los_rate_controller.py`, not iris_ma6's cascade)

**Affected modules**:
- `controller/tuning/auto_tune.py` — fork for replicator mode. Do not make modifications
- `controller/tuning/tuning_results/__init__.py` — updated gains after replication
- `sysid_output/*.csv` — PX4 SITL target data (read-only input)

**Key references**:
- PX4 SITL data: `sysid_output/` CSVs from ticket-007 (includes att_sp, rate_sp, thrust_sp columns)
- Existing tuner: `controller/tuning/auto_tune.py` (ParallelTuner, parallel env, step response extraction)
- Current best gains: `controller/tuning/tuning_results/__init__.py` (aero3, Cd=0.03)
- Ticket-007 analysis: `sysid_output/analysis/sysid_metrics.json` (mismatch quantification)
- Ticket-005 §2.1: decision rule and methodology

**Acceptance criteria**:
1. Replicator runs parallel gain sweep using iris_ma6 DroneController
2. Each candidate is scored against PX4 SITL timeseries (not hand-crafted criteria)
3. Best-matching gains produce iris_ma6 response that visually overlays PX4 SITL in comparison plots
4. Mismatch table shows improvement: majority of metrics within ±20% threshold
5. Updated `TUNED_CONTROLLER_CFG` exported

**Flow**: Full QRISPY
