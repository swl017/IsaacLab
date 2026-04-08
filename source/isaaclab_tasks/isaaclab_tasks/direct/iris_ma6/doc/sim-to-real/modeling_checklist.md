# Sim-to-Real Modeling Checklist

Items requiring real-world data collection to calibrate simulation parameters.

| # | Item | Sim Parameter | Status | Notes |
|---|------|--------------|--------|-------|
| 1 | Burst dropout characterization | `burst_p_onset`, `burst_p_recovery`, `burst_good_dropout_prob`, `burst_bad_dropout_prob` | Pending | Measure packet loss patterns on target wireless link (WiFi / telemetry radio). Fit Gilbert-Elliott transition probabilities from real traces. Currently using defaults (p_on=0.01, p_rec=0.1). |
| 2 | Drone step response | Motor dynamics, PID gains, drag coefficients | Pending | Fly step inputs on real Iris and record attitude/position response. Fit transfer function to replicate rise time, overshoot, settling time in sim. |
| 3 | Camera intrinsics with uncertainties | `camera_base_intrinsics`, focal length, distortion | Pending | Calibrate with checkerboard. Record per-parameter uncertainty (std) for domain randomization ranges. |
| 4 | Gimbal response characteristics | Gimbal controller gains, max rates, backlash | Pending | Command step inputs to gimbal, measure actual angle response. Characterize latency, rate limits, deadband. |
