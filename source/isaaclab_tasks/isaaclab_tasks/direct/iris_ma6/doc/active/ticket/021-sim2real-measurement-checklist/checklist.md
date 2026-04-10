# Sim-to-Real Measurement Checklist

**Purpose**: Ground every DR range in measured real-world data. Each item maps to a specific DR parameter or model validation point in iris_ma6.

**Hardware**: Iris quadrotor with 2-axis gimbal + zoom camera, PX4 autopilot

---

## Category A: Flight Dynamics (PX4 Logs)

### A1. Thrust-to-Weight Characterization

- [ ] **What**: Hover throttle percentage at known takeoff weight
- [ ] **Why**: Validates Isaac Sim thrust model; sets thrust DR baseline
- [ ] **How to collect**:
  1. Weigh drone with full payload (gimbal + camera + battery) on kitchen scale
  2. Fly hover in calm conditions (indoor or windless), 30s minimum
  3. Log `actuator_outputs` and `vehicle_local_position` from PX4 `.ulg`
  4. Repeat at 3 battery levels: full, 50%, 20% (voltage sag effect)
- [ ] **Extract**: Mean hover throttle, throttle variance, thrust-per-motor at hover
- [ ] **Maps to**: Aerodynamics `C_d`, `A`; mass DR baseline; battery sag model (not yet implemented)

### A2. Velocity Step Response

- [ ] **What**: Velocity tracking dynamics (rise time, overshoot, steady-state error)
- [ ] **Why**: Validates controller + drag model; calibrates velocity PID and drag feedforward
- [ ] **How to collect**:
  1. Command velocity steps in offboard mode: 0 -> 3, 0 -> 5, 0 -> 8, 0 -> 10 m/s
  2. Hold each step for 5s, return to hover for 5s between steps
  3. Do in X, Y axes separately (2 flights)
  4. Log `vehicle_local_position` (pos + vel), `vehicle_attitude`, `actuator_outputs`
  5. Repeat 3x for statistical significance
- [ ] **Extract**: Rise time (10-90%), overshoot %, steady-state error, settling time
- [ ] **Maps to**: Controller gain DR (currently ±20%); drag coefficients; Ticket-004 feedforward

### A3. Attitude Step Response

- [ ] **What**: Roll/pitch step response characteristics
- [ ] **Why**: Validates attitude controller time constants in sim
- [ ] **How to collect**:
  1. In stabilized/acro mode, command roll/pitch steps of 10, 20, 30 degrees
  2. Log `vehicle_attitude`, `vehicle_angular_velocity`, `actuator_outputs` at max rate
  3. 5 repetitions per angle per axis
- [ ] **Extract**: Attitude time constant (tau_roll, tau_pitch), rate limits (p_max, q_max), damping ratio
- [ ] **Maps to**: Controller attitude gains; rate limit DR

### A4. Yaw Response

- [ ] **What**: Yaw rate step response
- [ ] **Why**: Yaw dynamics are typically slower and less damped than roll/pitch
- [ ] **How to collect**:
  1. Command yaw rate steps: 30, 60, 90 deg/s in offboard mode
  2. Log `vehicle_attitude`, `vehicle_angular_velocity`
  3. 5 repetitions per rate
- [ ] **Extract**: Yaw time constant (tau_yaw), max yaw rate, steady-state yaw rate error
- [ ] **Maps to**: `max_yaw_rate` action limit; controller yaw gain DR

### A5. Motor RPM Characterization (Optional, High Value)

- [ ] **What**: Motor speed vs. command mapping, motor time constant
- [ ] **Why**: The interception project identified 27 parameters from motor logs; this is the gold standard
- [ ] **How to collect**:
  1. If ESC telemetry available: log individual motor RPM during flight maneuvers
  2. Otherwise: use a propeller tachometer on a test stand with stepped PWM commands
  3. Sweep PWM from idle to max in 5% increments, hold 2s each
- [ ] **Extract**: RPM-vs-PWM curve, motor time constant (tau_motor), min/max RPM
- [ ] **Maps to**: Isaac Sim motor model validation (if modeled), actuator time constant DR

---

## Category B: Drag and Aerodynamics

### B1. Steady-State Drag Force

- [ ] **What**: Velocity-dependent drag at multiple airspeeds
- [ ] **Why**: Validates `C_d * A` product; determines if quadratic drag model is sufficient
- [ ] **How to collect**:
  1. Fly constant-velocity segments at 2, 4, 6, 8, 10 m/s in calm air
  2. Hold each for 10s minimum (need steady state)
  3. Log `vehicle_local_position`, `vehicle_attitude` (pitch angle = proxy for drag)
  4. Indoor flight preferred (no wind)
- [ ] **Extract**: Average pitch angle at each speed (pitch ≈ atan(drag/weight)); compute effective Cd*A
- [ ] **Maps to**: `AerodynamicsCfg.C_d` (nominal 0.03), `AerodynamicsCfg.A` (nominal 0.1 m^2)

### B2. Wind Disturbance Characterization

- [ ] **What**: Position/velocity tracking error under known wind conditions
- [ ] **Why**: Calibrates `sigma_gust` and `v_wind_mean` for aero level 2+
- [ ] **How to collect**:
  1. Fly hover + velocity tracking outdoors in measured wind (use anemometer)
  2. Record wind speed/direction at 1Hz concurrent with PX4 logs
  3. Log position error, velocity error, attitude disturbance
  4. 3 flights at different wind speeds (calm, moderate, gusty)
- [ ] **Extract**: Position error std vs. wind speed; gust bandwidth from PSD of position error
- [ ] **Maps to**: `AerodynamicsCfg.v_wind_mean`, `sigma_gust`, `gust_bandwidth`

---

## Category C: Sensor Latency Pipeline

### C1. IMU-to-Policy Latency (Proprioceptive)

- [ ] **What**: End-to-end latency from IMU measurement to policy input
- [ ] **Why**: Validates delay system `ego_motion_latency` (currently 5ms mean, 2ms std)
- [ ] **How to collect**:
  1. On the real system, instrument the ROS2 pipeline:
     - Timestamp at IMU driver publish
     - Timestamp at policy node receive
  2. Log 1000+ samples during active flight
  3. Use `ros2 topic delay` or custom latency logger
- [ ] **Extract**: Mean, std, 95th/99th percentile latency; distribution shape
- [ ] **Maps to**: `delay_system_params.ego_motion_latency_ms_mean/std`

### C2. Detection Inference Latency

- [ ] **What**: YOLO/detector inference time on deployment GPU
- [ ] **Why**: Validates `ego_detection_latency` (currently 100ms mean, 15ms std)
- [ ] **How to collect**:
  1. Run detector on target hardware with representative images
  2. Time 500+ inference calls (include preprocessing)
  3. Measure under load (other processes running as in deployment)
- [ ] **Extract**: Mean, std, min, max inference time; distribution
- [ ] **Maps to**: `delay_system_params.ego_detection_latency_ms_mean/std`

### C3. Inter-Agent Communication Latency

- [ ] **What**: Round-trip time for state exchange between drones
- [ ] **Why**: Validates `other_agent_latency` (currently 500ms mean, 80ms std)
- [ ] **How to collect**:
  1. Deploy 2+ drones with ROS2 DDS communication
  2. Publish timestamped messages, measure one-way latency at receiver
  3. Test at various distances (10m, 30m, 50m, 100m)
  4. Test with and without obstacles (signal attenuation)
  5. Log 1000+ samples per condition
- [ ] **Extract**: Mean, std, 95th percentile; packet loss rate; burst loss statistics
- [ ] **Maps to**: `delay_system_params.other_agent_latency_ms_mean/std`; dropout parameters

### C4. Detection FPS and Dropout Rate

- [ ] **What**: Actual detection rate and missed detection probability
- [ ] **Why**: Validates staleness model (currently 25 FPS ± 5) and dropout (5% base)
- [ ] **How to collect**:
  1. Run full pipeline on deployment hardware
  2. Log every frame: was target detected? confidence? bbox?
  3. Ground truth from motion capture or manual annotation
  4. 5-minute continuous operation minimum
- [ ] **Extract**: Mean detection FPS, FPS variance, miss rate, false positive rate, burst miss statistics (consecutive misses)
- [ ] **Maps to**: Detection staleness params; `dropout_base_probability`; burst dropout params

---

## Category D: Camera and Gimbal

### D1. Camera Intrinsic Calibration

- [ ] **What**: Focal length, principal point, distortion coefficients per zoom level
- [ ] **Why**: Validates `CameraRandomizationCfg.focal_length_range` (800-1200 px)
- [ ] **How to collect**:
  1. Standard checkerboard calibration (OpenCV `calibrateCamera`)
  2. Calibrate at each discrete zoom level (1x, 2x, 4x, etc.)
  3. Use 20+ images per zoom level from diverse angles
- [ ] **Extract**: f_x, f_y per zoom level; principal point offset; distortion k1-k5
- [ ] **Maps to**: `CameraRandomizationCfg.focal_length_range`, `fov_scale_range`

### D2. Gimbal Joint Calibration

- [ ] **What**: Gimbal zero-offset, range of motion, backlash
- [ ] **Why**: Validates `GimbalRandomizationCfg` offset ranges (±0.1 rad yaw, ±0.05 pitch)
- [ ] **How to collect**:
  1. Command gimbal to known angles, measure actual angle with protractor/IMU
  2. Sweep full range in yaw and pitch, 5-degree increments
  3. Measure hysteresis: sweep forward vs backward
  4. Repeat on 3+ gimbal units if available (manufacturing variance)
- [ ] **Extract**: Zero offset per axis (mean, std across units), max range, backlash magnitude
- [ ] **Maps to**: `GimbalRandomizationCfg.yaw_offset_range`, `pitch_offset_range`

### D3. Gimbal Dynamic Response

- [ ] **What**: Gimbal angular velocity limits, settling time, overshoot
- [ ] **Why**: Validates gimbal stiffness/damping DR (currently ±20%)
- [ ] **How to collect**:
  1. Command step inputs to gimbal yaw/pitch at various amplitudes
  2. Log gimbal angle vs. time at high rate (>100 Hz)
  3. Command max-rate slews to measure velocity limits
- [ ] **Extract**: Max angular rate, rise time, overshoot, effective stiffness/damping
- [ ] **Maps to**: `GimbalRandomizationCfg.stiffness_scale_range`, `damping_scale_range`

---

## Category E: Mass and Payload

### E1. Drone Mass Budget

- [ ] **What**: Mass of each component and total takeoff weight
- [ ] **Why**: Sets nominal mass in sim; validates mass DR range
- [ ] **How to collect**:
  1. Weigh separately: frame, motors (x4), props (x4), battery, gimbal, camera, cables, misc
  2. Weigh complete assembled drone
  3. Weigh with each payload variant (different cameras, batteries)
- [ ] **Extract**: Component masses, total mass, payload variation range
- [ ] **Maps to**: `MassRandomizationCfg.body_mass_scale_range`, `payload_mass_range`

### E2. Center of Gravity with Gimbal

- [ ] **What**: CoG shift as gimbal rotates
- [ ] **Why**: Gimbal mass redistribution affects trim; not currently DR'd
- [ ] **How to collect**:
  1. Balance drone on a knife edge or hang from string (3-axis CoG)
  2. Measure CoG at gimbal neutral, max yaw left/right, max pitch up/down
- [ ] **Extract**: CoG position (x,y,z) at 5+ gimbal configurations; max CoG shift magnitude
- [ ] **Maps to**: Potential new DR parameter; validates inertia recomputation

---

## Category F: Controller Gains (PX4 Parameters)

### F1. PX4 Controller Gain Extraction

- [ ] **What**: Actual PID gains used on the real drone
- [ ] **Why**: Sets nominal gains for sim controller; determines if ±20% DR is appropriate
- [ ] **How to collect**:
  1. `param show MC_*` on PX4 shell (or from `.ulg` parameters section)
  2. Record all velocity, attitude, and rate PID gains
  3. If auto-tuned, record the auto-tune result
- [ ] **Extract**: All MC_PITCHRATE_*, MC_ROLLRATE_*, MC_YAWRATE_*, MPC_XY_VEL_*, MPC_Z_VEL_* gains
- [ ] **Maps to**: `GainRandomizationCfg` nominal values; ±20% range validation

### F2. Gain Sensitivity Analysis

- [ ] **What**: Flight performance under intentionally perturbed gains
- [ ] **Why**: Validates that ±20% DR range is survivable (not too aggressive)
- [ ] **How to collect**:
  1. Fly with nominal gains, record tracking performance
  2. Perturb velocity P gain by +20%, -20%, fly same trajectory
  3. Perturb rate D gain by +20%, -20%
  4. Check: does the drone remain stable? How much does tracking degrade?
- [ ] **Extract**: Tracking error vs. gain perturbation; stability margin
- [ ] **Maps to**: Confirms or adjusts ±20% gain DR range

---

## Data Collection Protocol

### Equipment Needed
- Kitchen scale (0.1g resolution) for mass measurements
- Anemometer for wind measurements
- Protractor or digital angle gauge for gimbal calibration
- Calibration checkerboard (A2 or larger)
- PX4 `.ulg` log enabled at max rate (`SDLOG_PROFILE = 1` for high-rate logging)
- ROS2 latency logging node
- Laptop with `pyulog` installed for log parsing

### Flight Test Procedure
1. **Pre-flight**: Record ambient temperature, wind (anemometer), battery voltage
2. **Arm and log**: Ensure PX4 logging active, ROS2 bag recording
3. **Execute maneuver**: Follow specific protocol per measurement item
4. **Post-flight**: Download `.ulg`, record final battery voltage
5. **Parse**: Use `pyulog` (`ulog2csv`) to extract topic CSVs

### Log Parsing Tools
```bash
# Install pyulog
pip install pyulog

# Convert .ulg to CSV
ulog2csv flight.ulg

# Extract specific topics
ulog2csv flight.ulg -m vehicle_local_position,vehicle_attitude,actuator_outputs

# Quick flight summary
ulog_info flight.ulg
```

### Minimum Data Requirements
| Category | Minimum Flights | Minimum Duration | Repetitions |
|----------|----------------|------------------|-------------|
| A (Dynamics) | 6 | 60s each | 3x per condition |
| B (Drag/Wind) | 4 | 30s per speed | 3x |
| C (Latency) | 2 | 5 min continuous | 1000+ samples |
| D (Camera/Gimbal) | 0 (bench test) | N/A | 20+ images |
| E (Mass) | 0 (bench test) | N/A | 3x weighings |
| F (Gains) | 3 | 60s each | 3x |

---

## Priority Order

**Phase 1 (Before first real-world test)**:
1. E1 (mass budget) — 30 min bench work
2. F1 (PX4 gains) — 5 min param dump
3. D1 (camera calibration) — 1 hr bench work
4. D2 (gimbal calibration) — 1 hr bench work

**Phase 2 (First flight campaign, calm conditions)**:
5. A1 (hover thrust) — 1 flight
6. A2 (velocity steps) — 2 flights
7. B1 (drag characterization) — 1 flight
8. C1-C2 (ego latency) — instrument pipeline during above flights

**Phase 3 (Multi-drone flights)**:
9. C3 (inter-agent comms) — 2-drone flight
10. C4 (detection FPS/dropout) — operational scenario
11. A3-A4 (attitude/yaw response) — 2 flights

**Phase 4 (Outdoor / stress testing)**:
12. B2 (wind characterization) — 3 outdoor flights
13. F2 (gain sensitivity) — 3 flights with perturbed gains
14. E2 (CoG with gimbal) — bench test
15. A5 (motor RPM, if ESC telemetry available)