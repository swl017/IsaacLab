# EKF2 state-stream lag characterization

What lives here: a measurement pipeline that records Pegasus ground-truth
state and the matching PX4 EKF2 estimate (delivered through MAVROS) under a
fixed maneuver suite, fits a per-channel time delay, and writes a JSON +
PDF that downstream tickets consume. This is the body-state equivalent of
mas/036 (gimbal dead-time) and mas/037 (zoom response) — same characterization
pattern, different channel.

The motivation, scope boundary, and validation gates live in
[ticket 041](../../../doc/active/ticket/041-px4-ekf-state-lag-measurement/ticket.md).
The data contract — channel list, time-base reconciliation rules, fit-method
selection, and the JSON schema — lives in
[doc/ekf_state_lag_spec.md](../../../doc/ekf_state_lag_spec.md). Read those
first; this README is the lab-notebook entry point.

## Running

1. Bring up Pegasus SITL + PX4 (Iris airframe, single vehicle, namespace
   `px4_1`) with lockstep enabled. MAVROS bridge up and arming complete via
   `offboard_py`. The existing tmux session in
   [tmux/isaac_sim.tmuxp.yaml](../../../../../../../tmux/isaac_sim.tmuxp.yaml)
   covers the standard launch.
2. From this directory: `./run_measurement.sh`. The script verifies topics,
   verifies `/clock` is advancing, snapshots the PX4 EKF2 params (best-effort
   via MAVROS get-param), records 6 maneuvers × 3 trials, then runs the fit.
3. Inspect `ekf_state_lag.json` (the canonical fit) and
   `ekf_lag_report.pdf` (one page per channel).

The position-channel sanity gate is the most important read: a measured
`position_xyz.delay_mean_s` of ~0.110 s ± 0.030 confirms that `use_sim_time`
+ lockstep are flowing correctly through the whole stack. Anything outside
that range means the time-base is broken, not the EKF — fix it before
trusting the rest of the channels.

## What downstream consumes

A future "apply EKF2 state delay to iris_ma6 observations / controller
feedback" ticket will read this JSON's `channels.<name>.delay_mean_s` and
plumb the per-channel delay into the iris_ma6 training-time observation
path. The schema is locked at `schema_version: 1`; breaking changes bump
the integer. Per-channel `status: "skipped"` blocks signal that the
measurement is missing or invalid — consumers should fall back to a
conservative default (e.g., `EKF2_GPS_DELAY` for GPS-driven channels) and
log the gap.

## 2026-05-25 measurement — headline finding

**In Pegasus lockstep SITL, the consumer-side EKF-channel lag is ≤ 10 ms
on position, velocity (world & body), and attitude_yaw** (below the 5 ms
resolution of the 200 Hz fit grid). attitude_pitch reports 16.7 ms (the
only channel with marginal but measurable non-zero lag, peak=0.95).

Two mechanisms together produce this:

1. **PX4 EKF lag-compensation.** PX4's EKF2 runs its Kalman filter at a
   delayed-fusion horizon (110 ms behind for GPS-driven channels) but
   integrates raw IMU forward to t=NOW via `OutputPredictor` before
   emitting. The `vehicle_local_position` it publishes is the *predicted*
   state at the latest IMU sample, stamped with that latest IMU time —
   so the value AND its stamp both refer to NOW, not to t−110 ms.
   See `PX4-Autopilot/src/modules/ekf2/EKF/output_predictor.cpp` and
   `EKF2.cpp:1410 PublishLocalPosition(now)`.

2. **Lockstep + use_sim_time collapses the publish-rate freshness gap.**
   With `/clock` paused between simulator ticks, no sim-time elapses
   between PX4 emit and recorder receive. The median `recv_stamp_s −
   Header.stamp` is exactly 0 ms on every topic. In wall-clock real-time,
   a 30 Hz mavros publisher would give the subscriber ~17 ms median
   freshness gap; in lockstep, that gap is below the /clock tick.

This is the *real* result and a correction to the ticket's naive
prediction of "position ~ EKF2_GPS_DELAY = 110 ms". The 110 ms is the
INTERNAL fusion delay, not the OUTPUT-vs-TRUTH delay that an iris_ma6
observation consumer sees.

**Implication for sim2sim parity (ticket 040):** the iris_ma6 training
path does NOT need an EKF-delay buffer for Pegasus SITL deployment. The
residual sim2sim gap after 040 lands cannot hide inside the EKF channel.

**Limitation — this number does NOT generalize to real hardware.**
Real-hardware deployment runs without lockstep: `/clock` is wall-time,
mavros publishes at ~30 Hz with ~5–10 ms MAVLink-serial transport, and
the subscriber will see a 15–25 ms median freshness gap **plus** any
non-zero estimation lag if real sensor noise degrades PX4's
lag-compensation prediction. SITL is the load-bearing floor; the
real-hardware re-measurement is a follow-up — and should be done with
wall-clock subscribers (no use_sim_time) so the freshness gap is
visible in `recv − stamp`.

**No observable first-order dynamics** — but the EKF has them by design.
PX4 specifies `EKF2_TAU_POS = EKF2_TAU_VEL = 0.25 s` (the output
predictor's smoothing-filter TC, [ekf2_params.c:1125,1136]). This TC
only manifests when there's a substantial discrepancy between the
IMU-integrated output and the corrected delayed-fusion state — i.e.,
when IMU bias drift / sensor noise makes the dead-reckoned output
disagree with the GPS-corrected delayed state. In lockstep SITL the
IMU is simulator-perfect, so the two paths agree and the 250 ms TC
has nothing to smooth. A first-order-plus-delay fit on the recorded
data finds T ≈ 0–10 ms across all fittable channels (essentially at
the 5 ms measurement grid), improving RMSE by < 3% over a zero-delay
model. Real hardware should re-test specifically for first-order
settling with T ≈ 0.25 s after disturbances.

3 channels (attitude_roll, body_rate_xyz, linear_acceleration_xyz) were
SKIPPED because the existing maneuver suite does not excite them with
clean step inputs:
- **attitude_roll**: vel_step_x produces only ~4 mrad roll transients.
  A dedicated roll-doublet maneuver would fix this.
- **body_rate_xyz**: chirp excites pitch rate well but xcorr cannot fit a
  constant delay to a frequency-swept signal. Step or doublet maneuvers
  on the rate channel are needed.
- **linear_acceleration_xyz**: similar — `state/accel` is inertial-frame
  truth, `mavros/imu/data.linear_acceleration` is body-frame + gravity;
  the maneuvers do not produce a high-SNR aligned signal in this pair.

These are a known follow-up; the spec's primary finding (EKF output is
lag-compensated) is solid on the 5 channels that fit successfully.
