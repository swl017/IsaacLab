# IRIS C-UAS: Ontology-Guided MARL Pipeline

## From Agent Capabilities to Architecture

**Version**: 0.2 (Updated to reflect iris_ma6 implementation)
**Last Updated**: 2026-03-17

---

## 1. Implementation Status Overview

This document maps ontology concepts to the actual iris_ma6 codebase. Components are marked with status indicators:
- **[IMPLEMENTED]**: Working code in iris_ma6 test environment
- **[PARTIAL]**: Partially implemented or conceptually present
- **[PLANNED]**: Designed in env spec but not yet implemented

### Current Implementation (iris_ma6 Test Environment)
| Component | Status | Module |
|-----------|--------|--------|
| Controller cascade | [IMPLEMENTED] | `controller/` |
| 2D bbox detection | [IMPLEMENTED] | `bbox_raycaster_v2/` |
| 3D triangulation | [IMPLEMENTED] | `triangulation/` |
| Communication delay | [IMPLEMENTED] | `delay_system_v3/` |
| CBF safety | [IMPLEMENTED] | `cbf_safety/` |
| Initial state randomization | [IMPLEMENTED] | `initial_states/` |
| Debug visualization | [IMPLEMENTED] | `visualization/` |

### Planned Features (Full iris_ma6 Spec)
| Component | Status | Reference |
|-----------|--------|-----------|
| Role switching (observe↔intercept) | [PLANNED] | env spec §2 |
| Attacker system | [PLANNED] | env spec §4 |
| Multi-target tracking | [PLANNED] | env spec §5 |
| Full observation pipeline (227D) | [PLANNED] | env spec §3 |

---

## 2. Ontology Structure

The ontology formalizes the domain knowledge your agents operate within. Below is a class hierarchy with key properties, mapped to actual implementations.

### 2.1 Class Hierarchy

```
Thing
├── Environment
│   ├── IrisMA6TestEnv [IMPLEMENTED]
│   │   └── iris_ma_env6_test.py
│   └── IrisMA6EnvCfg [IMPLEMENTED]
│       └── iris_ma_env6_test_cfg.py
│
├── Agent [IMPLEMENTED - 3 agents in test env, 6 planned]
│   ├── hasRole: {Observer} [IMPLEMENTED] / {Observer, Interceptor} [PLANNED]
│   ├── hasController: DroneController [IMPLEMENTED]
│   │   └── controller/drone_controller.py
│   ├── hasCamera: TiledCamera + BBoxRayCasterV2 [IMPLEMENTED]
│   │   └── bbox_raycaster_v2/bbox_raycaster.py
│   └── hasSafety: CBFManager [IMPLEMENTED]
│       └── cbf_safety/cbf_manager.py
│
├── Controller [IMPLEMENTED] (DroneController cascade)
│   ├── VelocityController (25Hz, PI outer loop)
│   │   └── controller/velocity_controller.py
│   ├── AttitudeController (100Hz, P middle loop)
│   │   └── controller/attitude_controller.py
│   ├── RateController (100Hz, PID inner loop)
│   │   └── controller/rate_controller.py
│   ├── MotorDynamics (τ=20ms, 1st order)
│   │   └── controller/motor_dynamics.py
│   ├── MixerMatrix (X-configuration)
│   │   └── controller/mixer.py
│   ├── GimbalController (τ=50ms)
│   │   └── controller/gimbal_controller.py
│   └── ZoomController (τ=100ms)
│       └── controller/zoom_controller.py
│
├── Perception [IMPLEMENTED]
│   ├── BBoxRayCasterV2 (GPU-batched 2D detection)
│   │   ├── hasTargetPrimPaths: list[str]
│   │   ├── hasBboxOutput: [N, C, T, 4] tensor
│   │   ├── hasBboxValidMask: [N, C, T] tensor
│   │   ├── hasOcclusionDetection: bool
│   │   └── bbox_raycaster_v2/
│   ├── TriangulationResult (3D estimation + uncertainty)
│   │   ├── hasPosition: [N, T, 3] tensor
│   │   ├── hasCovariance: [N, T, 3, 3] tensor
│   │   ├── hasIsValid: [N, T] tensor
│   │   └── triangulation/triangulation.py
│   └── MultiAgentDelaySystemV3 (comm delays)
│       ├── hasLatency: LatencyCfg (μ=100ms)
│       ├── hasStaleness: StalenessCfg (20-30 FPS)
│       ├── hasDropout: DropoutCfg
│       └── delay_system_v3/
│
├── Safety [IMPLEMENTED]
│   ├── CBFManager (orchestrator)
│   │   └── cbf_safety/cbf_manager.py
│   ├── CPARewardShaper (training-time penalty)
│   │   ├── usesGroundTruthPositions: bool = True
│   │   └── cbf_safety/cpa_reward_shaper.py
│   └── RobustDeploymentFilter (deployment-time filter)
│       ├── usesDelayedObservations: bool = True
│       └── cbf_safety/deploy_filter.py
│
├── Target [PARTIAL]
│   ├── hasEstimatedPosition: Position [IMPLEMENTED via triangulation]
│   ├── hasCovariance: TriangulationResult.covariance [IMPLEMENTED]
│   ├── hasDetectionStatus: bbox_valid_mask [IMPLEMENTED]
│   ├── hasTrackStatus: {Tracked, Degraded, Lost} [PLANNED - enum not implemented]
│   ├── hasVelocity: Vector3 [PLANNED]
│   └── hasAliveStatus: bool [PLANNED]
│
├── Formation [PARTIAL]
│   ├── hasMember: Agent [num_agents]
│   │   └── IrisMA6TestEnvCfg.num_agents (default: 3, planned: 6)
│   ├── hasBaseline: float (computed from agent positions)
│   │   └── Implicit in triangulation geometry
│   └── providesTriangulation: bool
│       └── Requires ≥2 agents with valid bbox detections
│
├── InitialStates [IMPLEMENTED]
│   ├── hasCurriculumProgress: float [0, 1]
│   ├── hasAgentPlacement: cylinder-based
│   ├── hasTargetDistance: [30-200m]
│   └── initial_states/initial_states.py
│
├── MissionPhase [PLANNED - see env spec §8]
│   ├── Search (no target acquired)
│   ├── Track (target under active triangulation)
│   ├── Engage (interceptor committed)
│   └── PostEngagement (BDA)
│
├── AttackerManager [PLANNED - see env spec §4]
│   ├── AttackerBehaviorCfg
│   ├── AttackerWaveGenerator
│   ├── AttackerController
│   └── InterceptResolver
│
├── Zone [PLANNED]
│   ├── hasType: {ProtectedAsset, EngagementZone, BufferZone}
│   ├── hasROE: EngagementRule [1..*]
│   └── hasCivilianRisk: {None, Low, High}
│
├── EngagementRule [PLANNED]
│   ├── requiresMinObservers: int
│   ├── requiresMaxCEP: float (m)
│   └── permitsKineticEngagement: bool
│
└── Visualization [IMPLEMENTED]
    ├── CameraFrustum (zoom-aware rendering)
    ├── DetectionIndicator (LOS lines with status coloring)
    ├── CovarianceEllipsoid (uncertainty visualization)
    └── visualization/custom_visualization.py
```

### 2.2 Inference Rules — Implementation Status

| Rule | Description | Status | Implementation |
|------|-------------|--------|----------------|
| **Triangulation feasibility** | Formation provides triangulation iff ≥2 members have valid detections | [IMPLEMENTED] | `triangulation/triangulation.py` computes `is_valid` from camera pair validity |
| **Track status** | Tracked if triangulation valid, Degraded if only 1 detection, Lost otherwise | [PARTIAL] | `bbox_valid_mask` provides raw data; enum not implemented |
| **Engagement authorization** | Authorized iff CEP ≤ requiresMaxCEP | [PLANNED] | Would use `TriangulationCfg.quality_metric` |
| **Formation redundancy** | redundancy = count(tracking agents) - 2 | [IMPLICIT] | Can be computed from bbox_valid_mask sum |
| **Role switch safety** | Block switch if would break triangulation before authorization | [DIFFERENT] | CBF provides safety filtering via CPA, not role gating |

#### Implemented Rule: Triangulation Feasibility

```python
# triangulation/triangulation.py - simplified logic
def compute_full_triangulation(
    camera_positions: Tensor,  # [N, C, 3]
    ray_directions: Tensor,    # [N, C, T, 3]
    valid_mask: Tensor,        # [N, C, T] - from bbox_valid_mask
    ...
) -> TriangulationResult:
    # Count valid observations per target
    num_valid = valid_mask.sum(dim=1)  # [N, T]

    # Triangulation requires ≥2 valid observations
    is_valid = num_valid >= 2  # [N, T]

    # Compute position and covariance only for valid targets
    ...
    return TriangulationResult(position=pos, covariance=cov, is_valid=is_valid)
```

#### Planned Rule: Engagement Authorization

```
IF   Target(?t) ∧ hasEstimationCovariance(?t, ?cov)
     ∧ CEP(?cov, ?cep) ∧ Zone(?z) ∧ hasROE(?z, ?roe)
     ∧ requiresMaxCEP(?roe, ?maxcep) ∧ ?cep ≤ ?maxcep
     ∧ permitsKineticEngagement(?roe, true)
THEN engagementAuthorized(?t, true)
```

This rule would use:
- `TriangulationResult.covariance` for CEP computation
- `TriangulationCfg.quality_metric` (trace, det, max_eig, sqrt_trace)
- Zone/ROE system (not yet implemented)

### 2.3 Why This Needs to Be an Ontology, Not Just Code

You could hard-code the inference rules as if-else statements. The ontology provides three benefits:

1. **Composability**: When future versions add new roles (e.g., jammer drone), you add a class and rules — you don't rewrite the decision tree.

2. **Queryability**: "Which agents are currently tracking target T?" is a SPARQL query, not a custom function. The LLM can query the ontology in the outer loop.

3. **Provenance for explainability**: When the operator asks "why wasn't engagement authorized?", the reasoner's inference chain is the answer:
   `CEP(12.3m) > requiresMaxCEP(10.0m) → engagementAuthorized = false`

---

## 3. MDP Formulations

### 3.1 Test Environment MDP [IMPLEMENTED]

The current iris_ma6 test environment implements a simplified MDP for core module validation.

**State** (per agent, 18D base + 6D triangulation):

| Feature | Dims | Source |
|---------|------|--------|
| position (world frame) | 3 | Robot articulation |
| velocity (world frame) | 3 | Robot articulation |
| orientation (quaternion wxyz) | 4 | Robot articulation |
| gimbal_yaw | 1 | Gimbal joint |
| gimbal_pitch | 1 | Gimbal joint |
| zoom_level | 1 | Zoom controller |
| bbox (cx, cy, w, h) | 4 | BBoxRayCasterV2 |
| bbox_empty | 1 | BBoxRayCasterV2 |
| **Triangulation (optional):** | | |
| triangulated_position | 3 | TriangulationResult |
| triangulation_std | 3 | sqrt(diag(covariance)) |

**Total**: 18D (base) or 24D (with triangulation)

**Action** (per agent, 7D continuous):

| Action | Dims | Range | Target |
|--------|------|-------|--------|
| vx, vy, vz | 3 | [-1, 1] × max_lin_vel | VelocityController |
| yaw_rate | 1 | [-1, 1] × max_yaw_rate | VelocityController |
| gimbal_yaw_rate | 1 | [-1, 1] × max_gimbal_rate | GimbalController |
| gimbal_pitch_rate | 1 | [-1, 1] × max_gimbal_rate | GimbalController |
| zoom_rate | 1 | [-1, 1] × max_zoom_rate | ZoomController |

**Control Hierarchy**:

```
Policy output (25Hz, decimation=4)
    ↓ [vx, vy, vz, yaw_rate] scaled by max_lin_vel, max_yaw_rate
VelocityController (PI, 25Hz)
    ↓ [attitude_cmd (roll, pitch), thrust_cmd]
AttitudeController (P, 100Hz)
    ↓ [rate_setpoint (p, q, r)]
RateController (PID, 100Hz)
    ↓ [torque_cmd (3D)]
MixerMatrix (X-configuration)
    ↓ [motor_cmds (4×)]
MotorDynamics (1st order, τ=20ms)
    ↓ [rotor_forces, rotor_torques]
Physics simulation (100Hz)
```

### 3.2 Full Spec MDP [PLANNED]

The full iris_ma6 design (see `iris_ma6_env_spec.md`) extends the test environment:

**Observation Space (227D)**:
- Ego (28D): pos, yaw, vel, yaw_rate, accel, gimbal, body_omega, bbox, bbox_empty, time_since_detection, zoom, ray_dir, **role**, **facility_direction**
- Allies (75D): 5 × 15D (pos, vel, omega, bbox_empty, ray_dir, data_age, detection_age)
- Targets (120D): 10 × 12D (rel_pos, rel_vel, loc_std, alive, aoi, threat_level)
- Defense (4D): num_alive_targets, num_intercepting, min_dist, episode_progress

**Action Space (8D)**:
- Platform (4D): vx, vy, vz, yaw_rate
- Gimbal+zoom (3D): gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate
- **Role (1D)**: role_action (negative=observe, positive=intercept)

**Role Transition Mechanism** (from env spec §2.2):

```python
role[i] = OBSERVE if a_role < 0 else INTERCEPT
```

### 3.3 Upper-Level Semi-MDP [PLANNED]

The role-switching decision operates at a longer timescale (options-level):

**State** (formation-level):
- n_obs, n_int: count of observers and interceptors
- CEP(t): current circular error probable
- dCEP/dt: rate of CEP improvement
- d_target: distance to nearest target
- phase: current MissionPhase
- engAuth: engagement authorization status

**Options** (discrete, temporally extended):

| Option | Precondition | Termination |
|--------|--------------|-------------|
| `ContinueTrack` | phase = Track | CEP converges or timeout |
| `SwitchOne` | engAuth = true ∧ redundancy ≥ 1 | Drone completes role transition |
| `SwitchAll` | engAuth = true ∧ CEP < killRadius | All observers become interceptors |

---

## 4. Role-Switch Trigger Options [PLANNED]

Three possible role-switch triggers for iris_ma6 full implementation:

### Option A: Explicit Operator Signal

```
trigger = operator_command
```

The upper-level MDP is disabled; the human directly invokes `SwitchOne` or `SwitchAll`. The ontology still enforces safety rules — even if the operator commands a switch, the system blocks it if `engagementAuthorized = false` and presents the reason.

**Recommended as starting point for iris_ma6.**

### Option B: Estimation Error Threshold

```
trigger = (CEP(t) < τ)
```

A fixed threshold τ (e.g., `requiresMaxCEP` from the ROE). Simple, interpretable, directly derived from the ontology's `EngagementRule.requiresMaxCEP`. But it's brittle:
- Doesn't account for whether CEP is still improving
- Doesn't account for target dynamics
- Doesn't account for opportunity cost

### Option C: RL-Optimized Joint Conditions

```
trigger = π_meta(s_meta) = SwitchOne   [learned policy]
```

The upper-level Semi-MDP policy considers CEP, its derivative, target dynamics, formation redundancy, time pressure, and cost — all jointly. The ontology constrains which actions are *legal*; the RL policy optimizes *which legal action is best*.

**Recommended deployment approach: Start with A, use B as baseline, train C in simulation, deploy C with A as override.**

---

## 5. LLM Integration Points [ASPIRATIONAL]

> **Note**: This section describes future integration. The LLM never touches the real-time loop. It operates on three timescales.

### 5.1 Pre-Mission (minutes to hours before)

**Mission planning assistance:**

```
Operator: "We have 6 drones available. Intel says 2 threats approaching
           from the north, possibly with decoys. Terrain is hilly with
           civilian structures to the south."

LLM: [queries ontology for available assets, zone definitions, ROE]
     [reasons about allocation]

     "Given 2 confirmed threats + possible decoys, I'd suggest:
      - 2 formations of 3 (2 observers + 1 reserve/interceptor each)
      - Northern engagement zone, keeping the southern civilian zone
        as no-kinetic
      - The hilly terrain will cause LOS breaks — consider wider baselines

      Shall I update the ontology instances for this mission config?"
```

### 5.2 During Mission — Async Advisory (seconds to minutes)

The LLM monitors ontology state and provides non-time-critical analysis:

```
LLM: "Target T2's evasion pattern is consistent with a pre-programmed
      waypoint-following UAS — it's not reacting to your observers.
      The triangulation should remain stable during intercept approach."
```

### 5.3 Post-Mission (minutes to hours after)

**After-action ontology refinement:**

```
LLM: "In 3 of the last 5 engagements, the CEP threshold was met but
      the intercept still missed. Analysis shows the target accelerated
      during the switch transition period (~2.3s average).

      Recommend:
      1. Add ontological property: Target.hasAccelerationEstimate
      2. Modify Rule 3: account for predicted CEP at intercept time
      3. Adjust R_meta: add term for target acceleration penalty"
```

---

## 6. Implementation Roadmap

| Phase | Description | Status | Key Deliverables |
|-------|-------------|--------|------------------|
| **Phase 1** | iris_ma5: Observation-only triangulation | [COMPLETE] | 2-3 observer coordination, triangulation, delay system |
| **Phase 2** | iris_ma6 test: Core modules integration | [IN PROGRESS] | Controller cascade, bbox v2, triangulation v2, CBF safety, initial states |
| **Phase 3** | iris_ma6 full: Role switching + attackers | [PLANNED] | 8D action space, attacker system, 227D observations |
| **Phase 4** | Adversarial: Learned attackers | [FUTURE] | Self-play, population-based training |

### Phase 2 Module Status

| Module | Location | Test Status |
|--------|----------|-------------|
| DroneController | `controller/` | ✅ 39 tests passing |
| BBoxRayCasterV2 | `bbox_raycaster_v2/` | ✅ Tested |
| Triangulation | `triangulation/` | ✅ Tested |
| DelaySystemV3 | `delay_system_v3/` | ✅ Tested |
| CBFSafety | `cbf_safety/` | ✅ Tested |
| InitialStates | `initial_states/` | ✅ Tested |
| Visualization | `visualization/` | ✅ Tested |

---

## 7. Time Constants and Rates

All values from actual implementation in `controller/controller_cfg.py` and `iris_ma_env6_test_cfg.py`:

| Component | Time Constant | Rate | Configuration Source |
|-----------|---------------|------|---------------------|
| Simulation | 10ms | 100Hz | `SimulationCfg.dt=1/100` |
| Policy loop | 40ms | 25Hz | `decimation=4` |
| Motor dynamics | 20ms | - | `MotorDynamicsCfg.tau=0.02` |
| Gimbal response | 50ms | - | `GimbalControllerCfg.tau=0.05` |
| Zoom response | 100ms | - | `ZoomControllerCfg.tau=0.1` |
| Rate controller | 10ms | 100Hz | Inner loop |
| Attitude controller | 10ms | 100Hz | Middle loop |
| Velocity controller | 40ms | 25Hz | Outer loop (matches policy) |
| Delay latency (mean) | 100ms | - | `LatencyCfg.mean=0.1` |
| Detection staleness | 33-50ms | 20-30 FPS | `StalenessCfg` |

### Runtime Component Mapping

| Component | Runs at | Latency | Deterministic? |
|-----------|---------|---------|----------------|
| Lower-level MDP (drone control) | Onboard / edge | ~10ms | Policy fixed after training |
| Ontology reasoner | Mission computer | ~10ms per query | Yes, fully |
| Upper-level Semi-MDP (role switch) | Mission computer | ~1s decision cycle | Policy fixed after training |
| LLM advisory | Cloud / base station | ~2-10s | No — advisory only |

---

## 8. Module Reference Table

| Ontology Concept | Python Module | Key Classes | Status |
|------------------|---------------|-------------|--------|
| Agent.Controller | `controller/` | `DroneController`, `DroneControllerCfg` | [IMPLEMENTED] |
| Agent.Gimbal | `controller/` | `GimbalController`, `ZoomController` | [IMPLEMENTED] |
| Agent.Motor | `controller/` | `MotorDynamics`, `MixerMatrix` | [IMPLEMENTED] |
| Agent.Aerodynamics | `controller/` | `AerodynamicEffects` | [IMPLEMENTED] |
| Perception.Detection | `bbox_raycaster_v2/` | `BBoxRayCasterV2`, `BBoxRayCasterV2Data` | [IMPLEMENTED] |
| Perception.Triangulation | `triangulation/` | `compute_full_triangulation()`, `TriangulationResult` | [IMPLEMENTED] |
| Perception.Delay | `delay_system_v3/` | `MultiAgentDelaySystemV3`, `DelayPipelineV3` | [IMPLEMENTED] |
| Safety.Training | `cbf_safety/` | `CPARewardShaper` | [IMPLEMENTED] |
| Safety.Deployment | `cbf_safety/` | `RobustDeploymentFilter` | [IMPLEMENTED] |
| Safety.Manager | `cbf_safety/` | `CBFManager` | [IMPLEMENTED] |
| InitialStates | `initial_states/` | `InitialStates`, `InitialStatesGenerator` | [IMPLEMENTED] |
| Visualization.Frustum | `visualization/` | `CameraFrustum` | [IMPLEMENTED] |
| Visualization.Detection | `visualization/` | `DetectionIndicator` | [IMPLEMENTED] |
| Visualization.Covariance | `visualization/` | `CovarianceEllipsoid` | [IMPLEMENTED] |
| Target.Attacker | `attacker/` | `AttackerManager` | [PLANNED] |
| MissionPhase | N/A | State machine | [PLANNED] |
| EngagementRule | N/A | ROE system | [PLANNED] |
| Zone | N/A | Zone definitions | [PLANNED] |

---

## 9. Related Documentation

| Document | Description | Path |
|----------|-------------|------|
| Environment Specification | Full iris_ma6 design | `doc/iris_ma6_env_spec.md` |
| Controller Specification | Controller architecture | `doc/controller_spec.md` |
| Frame Conventions | Coordinate systems | `doc/frame_conventions.md` |
| BBox Specification | Detection module | `doc/bbox_spec.md` |
| Triangulation Specification | 3D estimation | `doc/triangulation_spec.md` |
| Safety Specification | CBF system | `doc/safety_spec.md` |
| Initial States Specification | Randomization | `doc/initial_states_spec.md` |
| Visualization Specification | Debug rendering | `doc/visualization_spec.md` |
