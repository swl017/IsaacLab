## Ticket: Cascaded controller tuning with oscillation criteria and step response visualization

**What**: Extend the existing `controller/tuning/auto_tune.py` with oscillation detection metrics and step response plotting, then re-tune the cascaded PID controller to reduce velocity tracking oscillations.

**Why**: Policy evaluation (ticket-001) revealed that iris_ma6 agents exhibit oscillatory velocity tracking — agents overshoot/undershoot at 2-7 m/s instead of smoothly tracking a 5 m/s target. This causes diverging agent-target distance (14→70m over 20s), dropping visibility (90→30%), and suboptimal viewing angles (plateaus ~65° vs iris_ma5's stable 90°). The root cause is the cascaded PID controller (velocity→attitude→rate→motor) which introduces ~70ms signal path latency. The current auto-tuner evaluates settling time, overshoot, and steady-state error but lacks oscillation criteria and has no visual step response output for manual inspection.

**Evidence**:
- iris_ma6 trajectory speed panel shows oscillatory tracking: agents swing between 2-7 m/s while target moves at 4-5 m/s (see `experiments/scripts/outputs/trajectory.pdf`, panels g/h)
- iris_ma5 uses a holonomic point-mass controller with forced-level attitude — body doesn't tilt, gimbal operates on stable base, resulting in smooth velocity tracking
- iris_ma6 uses full PX4-style cascade: velocity cmd → desired attitude (tilt) → rate setpoint → motor torques. Each loop adds latency.
- Current tuner (`controller/tuning/auto_tune.py`) evaluates: hover drift, velocity settling time, overshoot %, steady-state error, attitude recovery time — but does NOT measure oscillation frequency, damping ratio, or number of zero-crossings in the velocity error signal

**Scope**:
- Add oscillation metrics to `TuningMetrics`: damping ratio, oscillation count (zero-crossings of velocity error), peak-to-peak amplitude of steady-state oscillation, frequency of dominant oscillation mode
- Add step response plotting: generate per-trial PDF showing velocity command vs actual velocity, attitude angles, and motor commands over time — enables visual inspection of ringing, overshoot, and settling behavior
- Update scoring function to penalize oscillatory responses (low damping ratio, high zero-crossing count)
- Re-run tuning with updated criteria
- Validate: re-run ~~policy evaluation (ticket-001 pipeline)~~ training with new controller gains and compare timeseries/trajectory plots

**Scope boundary**:
- Do NOT modify the controller architecture (keep 4-loop cascade)
- Do NOT modify the RL policy or reward function
- Do NOT change the motor dynamics model (tau_motor=10ms is a physical parameter)
- Do NOT add feedforward terms or model-predictive control — this is PID tuning only

**Affected modules**:
- `controller/tuning/auto_tune.py` — add oscillation metrics and step response plotting
- `controller/tuning/README.md` — document new metrics and outputs
- `iris_ma_env6_test_cfg.py` — update default controller gains after tuning
- `controller/*_cfg.py` — updated gain values

**Key references**:
- Current tuner: `controller/tuning/auto_tune.py` (ParallelTuner, TuningMetrics, ParameterSet)
- Current tuner docs: `controller/tuning/README.md`
- Controller cascade: `controller/drone_controller.py:170-295` (step_policy)
- Velocity controller: `controller/velocity_controller.py` (Kp_vel, Ki_vel, max_tilt=45°)
- Rate controller: `controller/rate_controller.py` (Kp_rate, Ki_rate, Kd_rate, tau_max)
- Motor dynamics: `controller/motor_dynamics.py` (tau_motor=10ms)
- Policy evaluation pipeline: `experiments/evaluate.py` + `experiments/scripts/plot_trajectory.py`

**Acceptance criteria**:
1. `TuningMetrics` includes oscillation count, damping ratio, and steady-state oscillation amplitude
2. Auto-tuner generates per-trial step response PDF (velocity, attitude, motor commands vs time)
3. Scoring function penalizes oscillatory responses (configurable weight)
4. Re-tuned gains produce visibly smoother velocity tracking in step response plots
5. Policy evaluation with new gains shows improved trajectory tracking: agent-target distance growth rate reduced, viewing angle closer to 90°

**Flow**: Full QRISPY
