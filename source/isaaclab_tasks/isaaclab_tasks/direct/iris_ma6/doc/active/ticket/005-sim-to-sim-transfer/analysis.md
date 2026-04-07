# Sim-to-Real Domain Shift Analysis for iris_ma6

## Source

Based on ChatGPT analysis (domain shift decomposition for active triangulation MARL),
cross-referenced against iris_ma6 actual implementation as of 2026-04-06.

## 1. Domain Shift Decomposition

The ChatGPT analysis identifies 5 domain shift categories. For each, we assess what
iris_ma6 already covers and what's missing.

### 1.1 Geometric / Calibration Shift

**What matters**: Camera intrinsics, zoom-focal mapping, gimbal axis alignment,
body-gimbal-camera extrinsics. Errors here corrupt ray directions, which directly
break triangulation regardless of detection quality.

| Parameter | iris_ma6 status | Gap? |
|-----------|----------------|------|
| Camera intrinsics (f_x, f_y) | `CameraRandomizationCfg.focal_length_range=(800, 1200)` px exists in config but **DomainRandomizer is NOT integrated into the env** — focal length is fixed at `camera.spawn.focal_length` during training | **Gap: dead code** |
| Image width and height | Fixed at render resolution; `target_width/height_range` exists in config but not active (same integration gap) | **Gap: dead code** |
| Zoom-focal mapping | `ZoomController` uses linear 1/zoom model | **Gap**: real zoom lenses are nonlinear |
| Gimbal axis alignment | `GimbalRandomizationCfg`: yaw ±5.7°, pitch ±2.9°, roll ±1.1° offset — exists in config but **not integrated** (same gap as camera) | **Gap: dead code** |
| Body-gimbal-camera extrinsic | `GimbalRandomizationCfg.mount_offset`: ±1cm position, ±1.1° rotation — `enabled=False` AND not integrated | **Gap: dead code + disabled** |
| Camera principal point drift | Not modeled — crops are always centered | **Gap** |
| Lens distortion | Not modeled | **Gap** (minor — bbox extraction doesn't use raw pixels) |
| Rolling shutter | Not modeled | **Gap** (minor — Isaac Sim renders global shutter) |

**Critical finding**: The entire `DomainRandomizer` module (camera, physics mass/material,
gimbal dynamics randomization) exists as standalone code with tests, but
`iris_ma_env6_test.py` does NOT call `DomainRandomizer.randomize_all()` anywhere.
Only **gain randomization** (via `DroneController.randomize_gains()`) and
**initial state randomization** (via `InitialStatesGenerator`) are actually active
during training. Camera intrinsics, physics mass, gimbal dynamics, and material
properties are all fixed throughout training.

**Assessment**: The domain randomization module needs to be **integrated into the env's
`_reset_idx`**. This is a prerequisite for sim-to-sim transfer. The code is written and
tested — it just needs to be wired in.

### 1.2 Temporal Shift

**What matters**: Detector latency, communication delay, timestamp misalignment,
frame drop. The ChatGPT analysis rates this as the highest-priority shift for this
system — "ray timing mismatch" is more dangerous than visual mismatch.

| Parameter | iris_ma6 status | Gap? |
|-----------|----------------|------|
| Detector latency | Ego detection: mean=50ms, std=20ms (Normal) | Covered |
| Communication delay | Other agent: mean=100ms, std=80ms (Normal) | Covered |
| Per-agent heterogeneity | `per_agent_randomization=True`, scale 0.5-2.0x | Covered |
| Burst dropout | **Not implemented** — Bernoulli i.i.d. per step | **Gap** |
| Heavy-tail latency | **Not implemented** — Normal with clipping | **Gap** |
| Packet loss pattern | i.i.d. dropout, probability=5% | Partial (no burst/correlated loss) |
| Timestamp misalignment | Tracked per-field via AoI; delay system maintains correct timestamps | Covered |
| Staleness (FPS limiting) | 20-30 FPS uniform, per-episode | Covered |

**Assessment**: Core delay pipeline is strong (advance/query split, per-agent, curriculum-gated).
Two gaps: (1) burst/correlated dropout (real WiFi drops packets in bursts, not i.i.d.), and
(2) heavy-tail latency (occasional 500ms+ spikes from GC, OS scheduling, network congestion).

### 1.3 Dynamics / Actuation Shift

**What matters**: Drone velocity response lag, yaw/gimbal rate saturation,
controller bandwidth mismatch, wind, battery voltage.

| Parameter | iris_ma6 status | Gap? |
|-----------|----------------|------|
| Motor time constant | `tau_motor=10ms`, randomized ±20% | Covered |
| Velocity controller gains | Randomized ±20% (Kp_vel, Ki_vel) | Covered |
| Attitude/rate gains | Randomized ±20% each | Covered |
| Max tilt angle | Fixed 45° | Could randomize ±5° |
| Gimbal rate limit | Fixed `max_gimbal_rate=π rad/s` | Could randomize |
| Zoom dynamics | `tau_zoom` randomized 0.01-1.0x (wide), curriculum-gated | Covered |
| Aerodynamic drag | Configurable fidelity 0-3 | Covered |
| Wind | Mean wind + Dryden gust model (level 2+) | Covered |
| Battery voltage | Not modeled | **Gap** (affects thrust ceiling) |
| Rotor degradation | Not modeled | **Gap** (minor) |

**Assessment**: Good coverage via gain randomization and aerodynamics model. Ticket-003
(current work) is directly improving this. Drag feedforward (ticket-004) addresses
the SS error gap. Battery/rotor degradation are minor.

### 1.4 Perception Shift

**What matters**: Lighting, background clutter, target appearance, bbox noise,
partial occlusion, false positives/negatives.

| Parameter | iris_ma6 status | Gap? |
|-----------|----------------|------|
| Bbox center/size noise | `bbox_std=7.0` pixels, per-agent scaled | Covered |
| Occlusion detection | Raycasting with visibility threshold, agent mesh occlusion | Covered |
| `bbox_empty` validity flag | Set for occluded/OOF/too-small targets | Covered |
| Target appearance variation | Not modeled (raycaster uses geometry, not vision) | N/A (no detector in loop) |
| Lighting variation | Not modeled | N/A (no detector in loop) |
| False positives | **Not modeled** — detections are geometry-based | **Gap** |
| False negatives (miss rate) | **Not modeled** — deterministic validity check | **Gap** |
| Detection confidence | **Not in observation** — only binary `bbox_empty` flag | **Gap** |
| Partial detection | `partial_detection_allowed=False` — all-or-nothing | **Gap** (real detectors have partial boxes) |

**Assessment**: iris_ma6 uses geometric bbox extraction (raycasting), not a learned detector.
This sidesteps visual domain shift entirely — the sim-to-real gap becomes "how different
is the real detector's output distribution from the geometric model?" The gaps (false
positives, miss rate, confidence) need to be bridged by a **bbox noise model** calibrated
from real detector statistics.

**Q: Should we plug in YOLO batch inference during training?**

No — not for training. The geometric raycaster runs at ~0.01ms/env (GPU raycasting) vs
YOLO at ~5-30ms/image (batch GPU inference), giving 10-100x slower training. With 1024
envs at 100Hz, YOLO-in-the-loop is infeasible. The practical approach:

- **Training**: Geometric raycaster + calibrated bbox noise model (miss rate, FP rate,
  confidence-dependent localization noise). Fast, deterministic, sufficient if the noise
  model is calibrated from real detector statistics.
- **Sim-to-sim validation** (PegasusSimulator): YOLO batch inference on 1-3 envs.
  Throughput isn't a constraint at evaluation scale. This validates that the noise model
  was adequate.
- **Only retrain with YOLO** if sim-to-sim validation reveals that the noise model can't
  capture detector failure modes (e.g., systematic bias for small targets, confidence-
  dependent localization error). This is a Phase 2 discovery, not something to build
  preemptively.

### 1.5 Multi-Agent Coordination Shift

**What matters**: Stale teammate state, asymmetric link quality,
per-agent performance differences, communication topology.

| Parameter | iris_ma6 status | Gap? |
|-----------|----------------|------|
| Stale teammate state | Modeled via delay system (latency + staleness + dropout) | Covered |
| Per-agent link quality | `per_agent_randomization=True` with latency/dropout offsets | Covered |
| Asymmetric agent dynamics | Per-agent gain randomization | Covered |
| Communication topology | **Fully connected assumed** — all agents see all others | **Gap** |
| Agent failure/dropout | Not modeled (all agents always present) | **Gap** (minor for 3 agents) |

**Assessment**: Good coverage for 3-agent scenario. Communication topology gap is
minor with 3 agents (fully connected is realistic for short-range links).

---

## 2. Priority Ranking for sim-to-sim Transfer

The ChatGPT analysis argues: **"ray timing + calibration > visual realism"**. We agree.
For sim-to-sim transfer (iris_ma6 → PegasusSimulator + PX4), the priorities are:

| Priority | Category | Action Required | Status |
|----------|----------|-----------------|--------|
| ~~P0~~ | ~~Action/observation interface~~ | ~~Map iris_ma6 7D actions → PX4 offboard~~ | **Done** (`mas_policy`: observation_assembler.py, action_publisher.py, policy_loader.py, cbf_filter.py — full 25Hz RNN inference pipeline with SKRL checkpoint loading) |
| **P0** | DomainRandomizer integration | Wire `DomainRandomizer.randomize_all()` into `_reset_idx` | Ticket-006 |
| **P0.5** | Light system identification | Step response comparison: iris_ma6 vs PX4 SITL (see §2.1 below) | Ticket-007 **Done** (7/8 metrics exceed ±20%: settling +55%, SS error +50%, hover drift +58-73%) |
| **P1** | Controller dynamics match | Tune iris_ma6 gains to match PX4 SITL response | Ticket-008 (sysid replicator — sweep gains, score against PX4 SITL data) |
| **P1** | Temporal model calibration | Measure PegasusSim latency distributions, update delay_cfg | Not started |
| **P2** | Bbox noise model | Calibrate raycaster noise from YOLO detector statistics | Ticket-009 |
| **P2** | Burst dropout | Gilbert-Elliott correlated dropout model in delay system | Ticket-011 |
| **P3** | Calibration biases | Enable gimbal mount offset randomization, add zoom nonlinearity | Not started |
| **P3** | False positive/negative model | Probabilistic miss rate, FP, detection confidence in bbox pipeline | Ticket-010 (depends on ticket-009 for calibration data) |

### 2.1 Light System Identification (P0.5)

A lightweight sysid step between training and sim-to-sim deployment, designed to
be reusable for sim-to-real later. The goal: quantify the dynamics mismatch between
iris_ma6's DroneController and PX4's actual cascade, then either correct the model
or confirm the mismatch is within the gain randomization range.

**What to measure** (in PegasusSimulator with PX4 SITL):

| Test | Command | Record | Compare against |
|------|---------|--------|-----------------|
| Velocity step 5 m/s | Offboard vel setpoint [5,0,0] | velocity, attitude, rates vs time | iris_ma6 tuner's vel_5 test |
| Velocity step 10 m/s | Offboard vel setpoint [10,0,0] | velocity, attitude, rates vs time | iris_ma6 tuner's vel_10 test |
| Attitude recovery | Perturb 20° roll/pitch via offboard attitude | attitude error, rate error vs time | iris_ma6 tuner's att test |
| Hover drift | Offboard vel setpoint [0,0,0] at 2m | position drift over 10s | iris_ma6 tuner's hover test |
| Yaw step | Offboard yaw_rate 0.5 rad/s for 2s | yaw angle vs time | Not yet in tuner |
| Gimbal slew | Command max gimbal rate | gimbal angle vs time | Gimbal tuning sweep results |

**What to extract**:
1. **Settling time, overshoot, damping ratio** for each test — same metrics as auto_tune.py
2. **Effective tau_motor** — fit first-order lag to velocity step onset
3. **Effective Cd** — fit drag coefficient from steady-state tilt at 5/10 m/s
4. **Effective max_tilt** — observe actual max attitude during aggressive commands
5. **Effective gimbal lag** — measure gimbal response time

**How to use the results**:
- If iris_ma6 metrics match PX4 SITL within ±20% (gain randomization range): **proceed
  with deployment** — the policy was trained with enough randomization to cover the gap.
- If mismatch > 20%: update iris_ma6 controller defaults or expand gain randomization
  range, retrain, re-validate.
- **For sim-to-real**: repeat the same tests on the real drone. The test scripts and
  metrics are identical — only the data source changes (PX4 SITL → real PX4 telemetry).

**Implementation**: The auto_tune.py step response plotter already generates the
reference curves. The sysid script would:
1. Send offboard commands to PX4 via MAVROS (reuse `offboard_py`)
2. Record `/mavros/local_position/odom` + `/mavros/imu/data` + gimbal state
3. Compute the same oscillation metrics (damping, ZC, amplitude, frequency)
4. Generate comparison plots: iris_ma6 reference vs PX4 SITL actual
5. Output a mismatch summary table

This is a measurement script (~200 lines), not a training change.

---

### 2.2 Selected Controller Gains (Ticket-003 Output)

From aero level 3 tuning (Cd=0.03), selected gains stored in
`controller/tuning/tuning_results/__init__.py`:

```python
TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(2.0416, 2.0416, 1.5355),
        Ki_vel=(1.3005, 1.3005, 0.7767),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(5.2060, 5.2060, 1.9334),
    ),
    rate=RateControllerCfg(
        Kp_rate=(0.3932, 0.3932, 0.3956),
        Ki_rate=(0.1805, 0.1805, 0.0900),
        Kd_rate=(0.01451, 0.01451, 0.00000),
    ),
)
```

Pattern: low Kp_vel (~2.0, avoids overdriving attitude loop), high Kp_rate (~0.39,
compensates motor lag), high Kd_rate (~0.015, damps motor lag oscillation), high
Ki_vel (~1.3, compensates drag at speed). These become the reference curves for sysid.

---

## 3. Gap Summary: What iris_ma6 Has vs Needs

### Already strong (no action needed for sim-to-sim):
- Per-agent delay/staleness/dropout with curriculum
- Gain randomization with anti-forgetting
- AoI and data age in observations
- bbox_empty validity flags
- RNN policy (implicit temporal alignment)
- Dual observation/reward paths

### Critical: DomainRandomizer not integrated (code exists, not wired)
The following modules exist with configs and tests but `iris_ma_env6_test.py` does NOT
call them during training. Only gain randomization and initial state randomization are active.

| Module | Code exists | Called from env? | Effect |
|--------|------------|-----------------|--------|
| Camera intrinsics randomization | `CameraProcessor.randomize()` | **No** | Fixed f_x, f_y throughout training |
| Camera resolution randomization | `CameraProcessor.randomize()` | **No** | Fixed resolution throughout training |
| Physics mass randomization | `PhysicsRandomizer.sample_mass_parameters()` | **No** | Fixed mass throughout training |
| Physics material randomization | `PhysicsRandomizer.sample_material_parameters()` | **No** | Fixed friction/restitution |
| Gimbal dynamics randomization | `GimbalRandomizer.randomize_dynamics()` | **No** | Fixed stiffness/damping |
| Gimbal offset randomization | `GimbalRandomizer.randomize_joint_offsets()` | **No** | No calibration bias |
| Gain randomization | `DroneController.randomize_gains()` | **Yes** | Active, curriculum-gated |
| Initial state randomization | `InitialStatesGenerator.generate()` | **Yes** | Active, curriculum-gated |
| Delay/noise/dropout | `MultiAgentDelaySystemV3` | **Yes** | Active, curriculum-gated |

**Action required**: Wire `DomainRandomizer.randomize_all()` into `_reset_idx` with
curriculum gating. The code is written — this is an integration task, not a development task.

(User's comment: Only the mock camera parameters(intrinsics and resolution) are used for bbox raycaster. The actual RGB image doesn't have to be processed.)

### Additional gaps for sim-to-sim:
1. **Burst dropout** — real wireless drops packets in bursts (5-20 consecutive), not i.i.d.
2. **Heavy-tail latency** — occasional 200-500ms spikes (GC, network congestion)
3. **Probabilistic detection model** — false positives, miss rate, partial detections
4. **Detection confidence** — add confidence score to observation (currently binary)
5. **Zoom-focal nonlinearity** — real zoom lens has nonlinear focal length mapping

### Not needed for sim-to-sim (only for sim-to-real):
- Visual appearance randomization (raycaster bypasses vision)
- Texture/lighting variation
- Lens distortion
- Rolling shutter
- Battery voltage modeling
