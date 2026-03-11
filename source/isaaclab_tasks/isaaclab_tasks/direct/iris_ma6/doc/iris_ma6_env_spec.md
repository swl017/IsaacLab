# iris_ma6 Environment Specification
## Observation + Interception for Facility Defense

**Version**: Draft 0.1
**Base**: iris_ma5 (V5) — IROS 2026 submission
**Scope**: 6 defenders vs N scripted attackers, facility defense with dynamic observe→intercept role transition

**Related Documents**:
- [frame_conventions.md](frame_conventions.md) — Coordinate frames, quaternion conventions, gimbal angles
- [controller_spec.md](controller_spec.md) — Controller architecture and parameters
- [bbox_spec.md](bbox_spec.md) — Object detection module, occlusion, and FP/FN modeling

---

## 0) Design Philosophy

iris_ma5 is an observation-only triangulation environment. iris_ma6 extends it into an **integrated observation-interception defense system**.

Core extension principles:

1. **Maximize V5 module reuse**: Triangulation, delay system, covariance, and safety modules are retained as submodules
2. **Role transition as the key contribution**: The policy learns to trigger observe→intercept transitions when uncertainty is sufficiently low
3. **Phased attacker complexity**: Initially scripted attackers with parametric behaviors + randomization; later transition to learned adversarial attackers (see §16)
4. **Progressive scaling**: Complexity is managed via 2v1 → 3v2 → 6v5 → 6v10 curriculum

---

## 1) Problem Formulation

### 1.1 Mission Definition

- **Protected facility**: A defense zone of radius `r_facility` located at the environment center
- **Defender drones**: `C = 6`, each equipped with a gimbaled zoom camera (same as V5)
- **Attacker drones**: `T ∈ {1, 2, ..., 10}`, scripted behavior attempting to reach the facility
- **Win condition**: All attacker drones intercepted before reaching the facility
- **Lose condition**: One or more attacker drones enter the facility radius

### 1.2 Dec-POMDP Extension

The V5 Dec-POMDP tuple is extended:

$$\langle \mathcal{S}, \{A^i\}, \{O^i\}, f, \{O^i\}, R, \gamma, \mathcal{T}, \mathcal{M} \rangle$$

Additional elements:
- $\mathcal{T} = \{t_1, ..., t_T\}$: Set of attacker drones (variable size; removed upon interception)
- $\mathcal{M}: \{1,...,C\} \to \{\texttt{OBSERVE}, \texttt{INTERCEPT}\}$: Agent role mapping (determined by policy)

Global state:

$$s_k = \left(\{x_k^{(i)}\}_{i=1}^C,\ \{x_k^{(t)}, \text{alive}_k^{(t)}\}_{t=1}^T,\ p_{\text{facility}}\right)$$

### 1.3 Agent States

**Defender drone** (V5 extension): $x^{(i)} \in \mathbb{R}^{17}$

$$x^{(i)} = [\underbrace{p, v, q, \omega}_{\text{V5: 16D}},\ \underbrace{\alpha, \beta, z}_{\text{gimbal+zoom: already in V5}},\ \underbrace{\texttt{role}}_{\text{NEW: 1D}}]$$

- `role ∈ {0, 1}`: 0 = OBSERVE, 1 = INTERCEPT (discrete, determined by policy output)

**Attacker drone**: $x^{(t)} \in \mathbb{R}^{7}$

$$x^{(t)} = [p^{(t)},\ v^{(t)},\ \texttt{alive}^{(t)}]$$

- No camera/gimbal (scripted, so no internal state needed)
- `alive ∈ {0, 1}`: Set to 0 upon interception

---

## 2) Action Space

### 2.1 Continuous Actions (V5 Extension)

Per agent, 8-dimensional continuous action:

$$a = [\underbrace{a_{vx}, a_{vy}, a_{vz}, a_{\dot\psi}}_{\text{platform (4D)}},\ \underbrace{a_{\dot\alpha}, a_{\dot\beta}, a_{\dot z}}_{\text{gimbal+zoom (3D)}},\ \underbrace{a_{\text{role}}}_{\text{NEW (1D)}}] \in [-1, 1]^8$$

### 2.2 Role Decision Mechanism

`a_role` is a continuous output converted to a discrete role via thresholding:

$$\texttt{role}_k^{(i)} = \begin{cases} \texttt{OBSERVE} & \text{if } a_{\text{role}} < 0 \\ \texttt{INTERCEPT} & \text{if } a_{\text{role}} \geq 0 \end{cases}$$

Action interpretation changes with role:

| Action Dimension | OBSERVE Mode | INTERCEPT Mode |
|------------------|--------------|----------------|
| `a_vx, a_vy, a_vz` | Optimize triangulation geometry | Pursuit maneuver toward target |
| `a_ψ̇` | Maintain observation attitude | Adjust heading for pursuit |
| `a_α̇, a_β̇` | Target centering (same as V5) | Target centering (same) |
| `a_ż` | Zoom optimization (same as V5) | Zoom out preferred (wide FoV) |

> **Implementation note**: The velocity command interpretation is not physically altered based on role. The same velocity controller (PointMass) is applied; behavioral differentiation is induced through reward shaping.

### 2.3 Target Assignment

In OBSERVE mode, operation is the same as V5 (currently single target), but multi-target environments require **observation target selection**.

**Option A — Implicit allocation (recommended, initial):**
- Target selection occurs naturally through gimbal pointing. Whichever target the gimbal is aimed at constitutes the assignment.
- BBoxRayCaster computes bboxes for all alive targets; the target closest to image center is automatically matched as the current observation subject.

**Option B — Explicit allocation (future extension):**
- An additional discrete action selects the target index (action space: `a_target ∈ {1, ..., T}`)

Target assignment in INTERCEPT mode:
- Automatic assignment to the nearest alive target (heuristic)
- Or the policy implicitly selects via per-target threat levels included in the observation

---

## 3) Observation Space

### 3.1 Design Principles

The V5 fixed-size observation vector is preserved, with multi-target information appended.
Variable target count is handled via **fixed slots + zero-padding** (set-based encoding is deferred to future architecture research).

### 3.2 Observation Vector Structure

**Dimension formula**: `ego(28D) + allies(15D × 5) + targets(12D × T_max) + defense(4D)`

For `T_max = 10`, `C = 6`: 28 + 75 + 120 + 4 = **227D**

#### 3.2.1 Ego Features (28D) — V5 26D + 2D Added

| Feature | Dims | Source | Note |
|---------|------|--------|------|
| position $p^w$ | 3 | V5 | |
| yaw $\psi$ | 1 | V5 | |
| linear velocity $v^w$ | 3 | V5 | |
| yaw rate $\dot\psi$ | 1 | V5 | |
| linear acceleration | 3 | V5 | |
| gimbal pitch, yaw | 2 | V5 | |
| body angular velocity | 3 | V5 | |
| ego bbox (primary target) | 4 | V5 | Bbox of the currently observed target |
| bbox empty | 1 | V6 | 1 = zero-filled / no detection, 0 = non-empty bbox |
| time since detection | 1 | V5 | |
| zoom level | 1 | V5 | |
| ray direction | 3 | V5 | |
| **current role** | **1** | **NEW** | 0=observe, 1=intercept |
| **facility direction** | **1** | **NEW** | Bearing angle to facility (ego-relative) |

**Total**: 26 + 2 = **28D**

#### 3.2.2 Per Other-Agent Features (15D × 5) — Same as V5

V5 other-agent observation repeated for `C-1 = 5` agents. Identical structure:

| Feature | Dims |
|---------|------|
| position | 3 |
| linear velocity | 3 |
| angular velocity | 3 |
| bbox empty (primary target) | 1 |
| ray direction | 3 |
| data age (AoI) | 1 |
| detection age | 1 |

**Total**: 15D × 5 = **75D**

#### 3.2.3 Per-Target Features (12D × T_max) — NEW

For each attacker drone (valid only if alive; zero-padded if dead):

| Feature | Dims | Description |
|---------|------|-------------|
| relative position | 3 | Triangulation estimate or prediction (ego-relative coordinates) |
| relative velocity | 3 | Estimated velocity (finite difference or prediction) |
| localization std | 3 | $\sqrt{\text{diag}(\Sigma_X^{(t)})}$, triangulation uncertainty for this target |
| alive flag | 1 | 0 = eliminated, 1 = active |
| AoI (localization) | 1 | Time elapsed since last valid triangulation of this target |
| threat level | 1 | $\texttt{threat}^{(t)} = \frac{v_{\text{approach}}^{(t)}}{\|p^{(t)} - p_{\text{fac}}\|}$ (approach speed / distance to facility) |

**Total**: 12D × 10 = **120D**

> **Ordering rule**: Targets are sorted by threat level in descending order and placed into slots. This mitigates the permutation problem and ensures the most dangerous target always occupies the first slot.

#### 3.2.4 Global Defense Features (4D) — NEW

| Feature | Dims | Description |
|---------|------|-------------|
| num alive targets | 1 | Currently alive attacker count (normalized: / T_max) |
| num intercepting allies | 1 | Allies currently in INTERCEPT mode (normalized: / C) |
| min target-facility dist | 1 | Distance of the nearest attacker to the facility |
| episode progress | 1 | Current step / max steps |

**Total**: **4D**

---

## 4) Scripted Attacker System (Phase A)

> **Note**: This section describes the initial scripted attacker implementation. For learned adversarial attackers (Phase B), see §16.

### 4.1 AttackerManager Module

New module replacing V5's `TargetMovement`: `attacker/attacker_manager.py`

```
AttackerManager
├── AttackerBehaviorCfg       # Behavior parameter definitions
├── AttackerWaveGenerator     # Attack wave generation (timing, spawn, count)
├── AttackerController        # Per-attacker velocity controller
└── InterceptResolver         # Interception adjudication logic
```

### 4.2 Attacker Behavior Model

Each attacker drone operates via a finite state machine:

```
APPROACH → [interceptor detected] → EVADE → [evasion success] → APPROACH
                                           → [intercepted]     → DEAD
         → [facility reached]     → BREACH (episode failure)
```

#### 4.2.1 APPROACH: Facility Approach

Base velocity:
$$v_d^{(t)} = v_{\text{max}}^{(t)} \cdot \frac{p_{\text{fac}} - p^{(t)}}{\|p_{\text{fac}} - p^{(t)}\|}$$

Path variation (randomized):
- **Direct**: Straight-line approach
- **Offset**: Indirect approach via intermediate waypoints (waypoints randomly placed around the facility)
- **Low-altitude**: Low-altitude approach (increases detection difficulty)

#### 4.2.2 EVADE: Evasion Maneuver

Triggered when an interceptor enters radius `d_evade_trigger`:

$$v_{\text{evade}} = v_{\text{max}}^{(t)} \cdot \hat{n}_{\perp}$$

where $\hat{n}_{\perp}$ is a random direction perpendicular to the interceptor→attacker vector.

Evasion intensity parameters:
- `evasion_agility ∈ [0, 1]`: Acceleration multiplier for evasion maneuvers
- `evasion_duration`: Duration of evasion maneuver (returns to APPROACH after)
- `evasion_probability ∈ [0, 1]`: Probability of attempting evasion (some attackers fly straight without evading)

#### 4.2.3 Behavior Profiles (Domain Randomization)

At episode reset, each attacker drone is assigned a random profile:

| Profile | Speed | Evasion Agility | Path | Proportion |
|---------|-------|-----------------|------|------------|
| **Kamikaze** | High (12 m/s) | None (0.0) | Direct | 20% |
| **Standard** | Medium (7 m/s) | Medium (0.5) | Offset | 40% |
| **Evasive** | Medium (7 m/s) | High (0.9) | Offset | 25% |
| **Stealth** | Low (3 m/s) | Medium (0.5) | Low-altitude | 15% |

### 4.3 Attack Wave Configuration

Attacker drones do not all appear at once; they spawn in waves:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `num_waves` | 1–3 | Total number of waves |
| `drones_per_wave` | 2–5 | Drones per wave |
| `wave_interval` | 5–15 s | Time interval between waves |
| `spawn_radius` | 500–1000 m | Spawn distance from facility |
| `spawn_arc` | 60°–360° | Spawn azimuth range (narrow = concentrated attack, wide = dispersed attack) |
| `spawn_altitude` | 20–80 m | Spawn altitude range |

### 4.4 Intercept Adjudication (InterceptResolver)

Intercept success condition:

$$\texttt{intercept}^{(i,t)} = \begin{cases} 1 & \text{if } \texttt{role}^{(i)} = \texttt{INTERCEPT} \land \|p^{(i)} - p^{(t)}\| < d_{\text{capture}} \\ 0 & \text{otherwise} \end{cases}$$

Post-intercept processing:
- Attacker drone: `alive = 0`, physics deactivated
- Defender drone survival: Determined by probability `p_survive`

$$\texttt{alive\_defender}^{(i)} = \begin{cases} 1 & \text{with prob. } p_{\text{survive}} \\ 0 & \text{with prob. } 1 - p_{\text{survive}} \end{cases}$$

- If survived: Immediately reverts to OBSERVE role, available for next target assignment
- If lost: Agent deactivated (actions ignored, observation zero-filled)

| Parameter | Default | Range |
|-----------|---------|-------|
| `d_capture` | 5.0 m | 3–10 m |
| `p_survive` | 0.8 | 0.5–1.0 |

---

## 5) Triangulation System (V5 Reuse)

### 5.1 Multi-Target Triangulation

V5's `midpoint_method_batched` and `triangulation_covariance_multi_camera` are called per-target:

```python
for t in range(T_max):
    if alive[t]:
        # Select only OBSERVE-mode agents currently detecting this target
        valid_cameras = [i for i in range(C) 
                        if role[i] == OBSERVE and bbox_empty[i][t] == 0]
        
        X_tri[t], valid[t] = midpoint_method_batched(
            camera_pos[valid_cameras], ray_dirs[valid_cameras][t])
        
        Sigma_X[t] = triangulation_covariance_multi_camera(
            camera_pos[valid_cameras], ..., X_gt[t])
```

### 5.2 BBoxRayCaster Extension

V5's BBoxRayCaster projects a single target's 8 corners. In the multi-target environment:

- Bbox computed for each camera × each alive target → tensor shape: `[N_env, C, T_max, 4]`
- Inter-target occlusion: Handling cases where a near target occludes a far target (depth ordering)

#### 5.2.1 Occlusion Implementation Plan

**Phase 1 (Initial)**: No occlusion handling — same as V5 default. All targets in FoV produce valid bboxes regardless of depth ordering.

**Phase 2 (Occlusion-Aware Detection)**: Implement depth-based occlusion masking:

1. **Depth ordering per camera**: For each camera, compute distances to all alive targets
   $$d^{(c,t)} = \|p^{(t)} - p_{\text{cam}}^{(c)}\|$$

2. **Bbox overlap detection**: For each pair of targets $(t_1, t_2)$ in same camera's FoV:
   - Compute IoU between projected bboxes
   - If IoU > $\tau_{\text{occlude}}$ (e.g., 0.3) and $d^{(c,t_1)} < d^{(c,t_2)}$, mark $t_2$ as occluded

3. **Occlusion mask tensor**: `occluded[N_env, C, T_max]` — boolean mask indicating occluded targets
   - Occluded targets: `bbox_empty = 1`, `bbox = [0, 0, 0, 0]`
   - Triangulation excludes occluded observations

4. **Partial occlusion handling**: If IoU is moderate (0.1–0.3), reduce bbox confidence rather than full invalidation
   $$\text{bbox\_confidence}^{(c,t)} = 1 - \text{IoU} \cdot \mathbb{1}[d^{(c,t)} > d^{(c,t_{\text{occluder}})}]$$

**Implementation location**: `bbox_raycaster/occlusion_handler.py`

**Validation criteria**:
- Near target fully occluding far target → far target `bbox_empty = 1`
- Camera repositioning reveals occluded target → `bbox_empty` returns to 0
- Triangulation covariance increases when observations are lost to occlusion

### 5.3 Detection-to-Target Association

When multiple targets fall within a single camera's FoV, matching which bbox belongs to which target is required:

- **In simulation, GT association is used**: Each bbox's source target is known
- Real deployment would require re-identification / tracking, but GT suffices for the training environment

---

## 6) Delay System Extension

### 6.1 V5 DelaySystemV2 Reuse

Inter-agent communication delay retains the V5 structure:
- ego path: Fast self-state
- other path: Slow teammate state via communication

### 6.2 Target Information Delay

Delay is also applied to shared target localization results:

| Information Path | Staleness | Latency | Dropout |
|-----------------|-----------|---------|---------|
| ego → ego bbox | 50 Hz (onboard detection) | 10 ms | 2% |
| triangulation result → all agents | 25 Hz (fusion rate) | 20–100 ms | 5% |
| target alive status | instant | 0 ms | 0% (critical) |

### 6.3 AoI Extension

In addition to V5's inter-agent AoI, **per-target AoI** is maintained:

$$\Delta_{\text{loc}}^{(t)} = k_{\text{current}} - k_{\text{last\_valid\_tri}}^{(t)}$$

This value populates the per-target `AoI (localization)` field in the observation vector.

---

## 7) Reward Function

### 7.1 Role-Dependent Reward Decomposition

Total per-agent reward:

$$r_k^{(i)} = r_{\text{defense}} + r_{\text{role}} + r_{\text{safety}} + r_{\text{action}}$$

### 7.2 Defense Reward (Shared Across Team)

Episode-level outcomes distributed per step:

**Facility defense reward** (shared by all agents):
$$r_{\text{defense}} = s_{\text{def}} \cdot \Delta t \cdot \sum_{t=1}^{T} \frac{\texttt{alive}^{(t)}}{T} \cdot \frac{d^{(t)}_{\text{fac}}}{d^{(t)}_{\text{fac,init}}}$$

Intuition: Positive reward when attackers remain far from the facility (or are eliminated).

**Intercept success bonus** (for the intercepting agent):
$$r_{\text{kill}} = s_{\text{kill}} \cdot \mathbb{1}[\text{intercept success at step } k]$$

**Facility breach penalty** (shared by all agents):
$$r_{\text{breach}} = -s_{\text{breach}} \cdot \mathbb{1}[\text{any target reached facility}]$$

### 7.3 Role-Dependent Reward

#### OBSERVE Mode ($\texttt{role}^{(i)} = 0$):

Uses V5 observation rewards directly, computed for the observed target:

$$r_{\text{obs}}^{(i)} = r_{\text{center}}^{(i)} + r_{\text{size}}^{(i)} + r_{\text{tri}}^{(i)}$$

- $r_{\text{center}}$: Target bbox centering (V5 §11.2)
- $r_{\text{size}}$: Bbox size shaping (V5 §11.2)
- $r_{\text{tri}}$: Triangulation quality — **based on covariance of observed targets** (V5 §11.3.1)

Additional observation reward:

$$r_{\text{coverage}} = s_{\text{cov}} \cdot \Delta t \cdot \frac{\text{num targets with } \Delta_{\text{loc}} < \tau_{\text{fresh}}}{T_{\text{alive}}}$$

Intuition: Reward for maintaining fresh localization across many targets.

#### INTERCEPT Mode ($\texttt{role}^{(i)} = 1$):

Tracking/interception rewards replace observation rewards:

**Approach reward** (reward for closing distance to target):
$$r_{\text{approach}} = s_{\text{app}} \cdot \Delta t \cdot \text{clip}\left(\frac{d_{k-1}^{(i,t)} - d_k^{(i,t)}}{v_{\max} \cdot \Delta t},\ -1,\ 1\right)$$

**Uncertainty-gated intercept**: Intercept transition is only rewarded when uncertainty is sufficiently low:
$$r_{\text{intercept\_ready}} = s_{\text{ready}} \cdot \Delta t \cdot \mathbb{1}\left[\sqrt{\text{tr}(\Sigma_X^{(t)})} < \sigma_{\text{threshold}}\right] \cdot \mathbb{1}[\texttt{role} = \texttt{INTERCEPT}]$$

Intuition: Switching to intercept mode without sufficient localization forfeits this reward → **induces observe-first strategy**.

### 7.4 Safety Reward (V5 Extension)

V5 inter-agent collision + TTC are retained:

$$r_{\text{coll}} = s_{\text{coll}} \cdot \Delta t \cdot \phi_{\text{coll}}^{(\text{agent-agent})} \cdot \text{progress}_{\text{safety}}$$

> **Note**: Interceptor↔attacker proximity is an interception event, not a collision. Safety collisions apply only between friendly agents.

### 7.5 Action Penalty (Same as V5)

$$r_{\text{action}} = r_{\text{act}} + r_{\Delta\text{act}}$$

Same as V5 §11.1, extended to 8 dimensions (including role action).

### 7.6 Reward Scale Guidelines

| Term | Scale | Gating | Rationale |
|------|-------|--------|-----------|
| $r_{\text{defense}}$ | 1.0 | always | Core objective |
| $r_{\text{kill}}$ | 10.0 | on intercept | Sparse, large bonus |
| $r_{\text{breach}}$ | -50.0 | on breach | Immediate episode termination |
| $r_{\text{tri}}$ | 0.5 | progress_coord | V5 level |
| $r_{\text{center}}$ | 0.3 | OBSERVE only | V5 level |
| $r_{\text{approach}}$ | 0.5 | INTERCEPT only | Approach shaping |
| $r_{\text{intercept\_ready}}$ | 0.3 | INTERCEPT + low uncertainty | Uncertainty gate |
| $r_{\text{coverage}}$ | 0.3 | OBSERVE only | Multi-target coverage |
| $r_{\text{coll}}$ | -2.0 | progress_safety | V5 level |
| $r_{\text{action}}$ | -0.01 | always | V5 level |

---

## 8) Episode Lifecycle

### 8.1 Initialization (`_reset_idx`)

1. Facility position: Fixed at environment origin
2. Defender drone placement: Randomly placed within 50–100 m radius of facility (V5 formation generator extended)
3. Attack wave 1 spawn: Randomly placed at spawn_radius with random azimuth/altitude
4. All defender drones initial role: OBSERVE
5. All per-target AoI: ∞ (unobserved state)

### 8.2 Episode Progression

```
Every control step k:
  1. _pre_physics_step:
     - Update curriculum progress
     - Action preprocessing (same as V5)
     - Role decision (a_role thresholding)
     - Wave spawning check (time-based)
  
  2. _apply_action:
     - Defender drones: V5 force/torque + gimbal (same)
     - Attacker drones: Scripted velocity → acceleration controller
  
  3. _compute_intermediate_values:
     - Delay system update (same as V5)
     - Multi-target detection (BBoxRayCaster × T)
     - Multi-target triangulation (OBSERVE agents only)
     - Intercept adjudication (InterceptResolver)
     - Attacker state update (alive, evasion FSM)
  
  4. _get_rewards: See §7 above
  
  5. _get_observations: See §3 above
  
  6. _check_termination:
     - breach: any target reaches facility → done, penalty
     - all killed: all targets dead → done, bonus
     - timeout: max_steps reached → done
     - all defenders lost: all defender drones destroyed → done, penalty
```

### 8.3 Termination Conditions

| Condition | done | Reward |
|-----------|------|--------|
| All attackers eliminated | True | $+r_{\text{all\_clear}}$ |
| Any attacker reaches facility | True | $-r_{\text{breach}}$ |
| Timeout (30–60 s) | True | Penalty proportional to remaining attacker count |
| All defenders lost | True | $-r_{\text{breach}}$ |

---

## 9) Curriculum

### 9.1 V5 Curriculum Compatibility

The V5 linear progress ramp structure is retained:

$$\text{progress}(k; k_s, k_e) = \text{clip}\left(\frac{k - k_s}{k_e - k_s},\ 0,\ 1\right)$$

### 9.2 Extended Curriculum Dimensions

5 dimensions added to V5's 7 for a total of 12:

| # | Dimension | V5 | Schedule | Description |
|---|-----------|-----|----------|-------------|
| 1 | delay/noise | ✓ | 0–100k | Same as V5 |
| 2 | tracking difficulty | ✓ | 0–60k | Same as V5 |
| 3 | triangulation reward | ✓ | 20k–100k | Same as V5 |
| 4 | safety penalties | ✓ | 40k–120k | Same as V5 |
| 5 | moving target speed | ✓ | 20k–100k | Same as V5 → transitions to attacker speed |
| 6 | dynamics randomization | ✓ | 80k–200k | Same as V5 |
| 7 | zoom curriculum | ✓ | 0–60k | Same as V5 |
| 8 | **num attackers** | NEW | 50k–200k | 1 → T_max |
| 9 | **attacker evasion** | NEW | 100k–250k | 0 → 1 (evasion_agility scale) |
| 10 | **wave complexity** | NEW | 100k–250k | 1 wave → num_waves |
| 11 | **intercept reward enable** | NEW | 60k–150k | Intercept reward activation |
| 12 | **defender loss enable** | NEW | 150k–300k | p_survive < 1 activation |

### 9.3 4-Phase Training Plan

| Phase | Steps | Focus | Key Curriculum Activations |
|-------|-------|-------|---------------------------|
| **Phase 1** | 0–60k | Single-target observation | V5 ego motion + tracking. 1 attacker, stationary, no evasion |
| **Phase 2** | 40k–120k | Triangulation + moving target | V5 coordination. Attacker moves toward facility. OBSERVE only |
| **Phase 3** | 80k–200k | Intercept transition learning | Intercept reward activated. 1–3 attackers, mild evasion |
| **Phase 4** | 150k–350k | Full scenario | Multi-wave, 5–10 attackers, strong evasion, defender loss, full delay/noise |

---

## 10) Randomization

### 10.1 V5 Randomization Retained

- Defender drone initial formation (planar/grid/line)
- Gimbal feasibility check
- Physics parameters (mass, drag, etc.)

### 10.2 Additional Randomization

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| num attackers per episode | Uniform(curriculum_min, curriculum_max) | 1–10 |
| attacker behavior profile | Categorical (§4.2.3) | — |
| spawn direction | Uniform | 0°–360° |
| spawn radius | Uniform | 500–1000 m |
| spawn altitude | Uniform | 20–80 m |
| wave timing | Uniform | 5–15 s interval |
| spawn arc width | Uniform | 60°–360° |
| attacker max speed | Uniform per profile | 3–12 m/s |
| evasion agility | Uniform per profile × curriculum | 0–1 |
| facility radius | Fixed (early curriculum) → Uniform (late) | 10–30 m |
| `p_survive` | Fixed 1.0 (early) → curriculum (late) | 0.5–1.0 |
| `d_capture` | Fixed | 5 m |

---

## 11) Evaluation Metrics

### 11.1 V5 Metrics Retained (Per-Target Extension)

| Metric | V5 | Extension |
|--------|-----|-----------|
| RMSE | ✓ | Averaged per target |
| Tri Valid | ✓ | Averaged per target |
| Visibility | ✓ | Averaged per target |
| Convergence | ✓ | Averaged per target |
| Collision (agent-agent) | ✓ | Same |

### 11.2 New Defense Metrics

| # | Metric | Unit | Description |
|---|--------|------|-------------|
| 1 | **Defense Success Rate** | % | Fraction of episodes where all attackers are eliminated |
| 2 | **Breach Rate** | % | Fraction of episodes where ≥1 attacker reaches the facility |
| 3 | **Kill Count** | count/ep | Eliminations per episode |
| 4 | **Intercept RMSE** | m | Localization error at the moment of interception |
| 5 | **Time-to-Intercept** | s | Time from target spawn to elimination |
| 6 | **Observe-to-Intercept Latency** | steps | Uncertainty level at the moment of role transition |
| 7 | **Defender Survival Rate** | % | Fraction of defender drones alive at episode end |
| 8 | **Coverage Freshness** | s | Mean localization AoI across all alive targets |
| 9 | **Role Transition Count** | count/ep | Number of OBSERVE↔INTERCEPT transitions |
| 10 | **Uncertainty at Intercept** | m | $\sqrt{\text{tr}(\Sigma_X)}$ at intercept moment |

### 11.3 Hierarchical Task Success (Extended)

**Level 0 (Track Maintenance)**: Same as V5 — tri_valid ≥ 50% for alive targets

**Level 1 (Localization Accuracy)**: Same as V5 — RMSE < 2.0m for ≥ 80% of valid steps

**Level 2 (Defense) — NEW**:
$$\text{pass}_2 = \left(\frac{\text{killed\_targets}}{T} \geq 0.8\right) \land (\text{breach} = 0)$$

**Overall Success**:
$$\text{success} = \text{pass}_0 \land \text{pass}_1 \land \text{pass}_2$$

---

## 12) Training Configuration

### 12.1 Architecture

MAPPO-RNN retained (validated in V5). Key changes:

- **Input dim**: 227D (V5: 47D)
- **Hidden size**: 256 (V5: 64) — scaled proportionally to observation increase
- **GRU hidden**: 256 (V5: 64)
- **Action dim**: 8 (V5: 7)
- **Shared policy**: All defender agents share the same policy (same as V5)

### 12.2 Critic (CTDE)

Centralized critic shared observation:
- Concatenation of all agents' ego features
- GT positions/velocities of all targets (privileged information)
- Global defense features

Estimated critic obs dim: $\sim$ 28 × 6 + 6 × 10 + 4 = **232D**

### 12.3 Training Parameter Guidelines

| Parameter | Value | Note |
|-----------|-------|------|
| num_envs | 2048–4096 | Reduced from V5 due to increased per-env compute |
| total_timesteps | 300k–500k | Sufficient training through Phase 4 |
| sequence_length | 32–64 | Longer context may be needed for multi-target |
| mini_batch_size | Requires tuning | Increased obs dim impacts GPU memory |
| episode_max_steps | 750–1500 | 30–60 s @ 25 Hz control rate |
| γ | 0.99 | Same as V5 |
| GAE λ | 0.95 | Same as V5 |

---

## 13) Implementation Roadmap

### 13.1 Milestone Plan

#### Phase A: Scripted Attackers (Current Scope)

| Milestone | Goal | Implementation | Validation Criteria |
|-----------|------|----------------|---------------------|
| **M1** | V5 → V6 base structure | Environment class, config dataclass, attacker spawn/remove | 6 defenders + 1 attacker episode runs |
| **M2** | Scripted attacker | AttackerManager, behavior FSM, wave generator | Episodes complete with varied attacker profiles |
| **M3** | Multi-target detection | BBoxRayCaster multi-target extension, per-target triangulation | Simultaneous triangulation of 2+ targets succeeds |
| **M4** | Role transition | 8D action space, role reward, intercept resolver | observe→intercept→kill sequence learned in 1v1 |
| **M5** | Curriculum integration | 12-dim curriculum, 4-phase schedule | Defense success rate > 50% in 6v5 |
| **M6** | Full scale | 6v10, multi-wave, domain randomization | Defense success rate > 30% in 6v10 |

#### Phase B: Adversarial MARL (Future — see §16)

| Milestone | Goal | Implementation | Validation Criteria |
|-----------|------|----------------|---------------------|
| **M7** | Adversarial training infra | Self-play framework, attacker policy class, alternating training | Attacker policy trains alongside frozen defender |
| **M8** | Co-evolution | Joint defender-attacker training, population-based methods | Nash equilibrium metrics, defense robustness |
| **M9** | Transfer evaluation | Evaluate defenders against held-out attacker strategies | Generalization to unseen attack patterns |

### 13.2 V5 Module Reuse Map

| V5 Module | Role in V6 | Modification Level |
|-----------|------------|-------------------|
| `iris_ma_env5.py` | Base class or fork | **Major** — Full episode logic overhaul |
| `point_mass.py` | Defender drone controller | None |
| `gimbal_stabilizer.py` | Gimbal control | None |
| `multi_agent_delay_system_v2.py` | Inter-agent delay | **Minor** — Add per-target delay channels |
| `bbox_raycaster.py` | Target detection | **Medium** — Multi-target loop |
| `triang_cov_reward_torch.py` | Triangulation + covariance | **Minor** — Per-target call wrapping |
| `safety_manager.py` | Friendly collision | None |
| `ttc_computer.py` | TTC computation | **Minor** — Interceptor exception handling |
| `target_movement.py` | **Replaced** | Replaced by AttackerManager |
| `curriculum_cfg.py` | Curriculum | **Medium** — 5 additional dimensions |
| `experiment_registry.py` | Experiment management | **Medium** — New ablation groups added |
| `metric_tracker.py` | Evaluation | **Medium** — Defense metrics added |

---

## 14) Ablation Study Plan

### 14.1 Experiment Groups

| Group | Focus | Variants |
|-------|-------|----------|
| **B1** | Role transition mechanism | `b1_learned_role` (this design), `b1_heuristic_role` (fixed uncertainty threshold), `b1_always_observe` (observation only, no interception) |
| **B2** | Uncertainty threshold | `b2_sigma_1m`, `b2_sigma_2m`, `b2_sigma_5m`, `b2_no_gate` (no gate) |
| **B3** | Attacker difficulty | `b3_easy` (slow, no evasion), `b3_medium` (mixed), `b3_hard` (fast, evasive) |
| **B4** | Agent scaling | `b4_3v2`, `b4_4v3`, `b4_6v5`, `b4_6v10` |
| **B5** | V5 ablation revalidation | `b5_no_aoi`, `b5_no_delay`, `b5_angular_only` — Whether V5 conclusions hold in the defense scenario |

### 14.2 Key Research Questions

1. **Does the policy learn to trigger observe→intercept transitions based on uncertainty?**
   → Compare B1, B2. Validate via "Uncertainty at Intercept" metric.

2. **Does observation quality directly impact intercept success rate?**
   → Examine correlation between uncertainty threshold and defense success rate in B2 results.

3. **Are V5's delay-aware / AoI / multi-source covariance contributions amplified in the defense scenario?**
   → Re-evaluate V5 ablations with defense metrics in B5.

4. **How do resource recycling (p_survive) strategies emerge under numerical inferiority?**
   → Analyze sequential intercept behavior in B4 agent scaling results.

---

## 15) Key Implementation Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Multi-target triangulation compute cost | Per-env throughput degradation | Parallel batched processing per target; compute only for OBSERVE agents |
| Training instability with 227D observation | Convergence failure | Observation normalization, increased hidden size, fine-grained curriculum tuning |
| Sparse intercept reward | Credit assignment difficulty | Approach shaping + intercept_ready reward for dense signal |
| Generalization to variable target count | Overfitting to specific T | Curriculum-based progressive T increase + per-episode randomization |
| Role oscillation (rapid repeated switching) | Unstable behavior | Hysteresis or cooldown on role transitions |
| GPU memory (C=6, T=10, N=4096) | OOM | Reduce to N=2048 or compress observations |

---

## 16) Adversarial MARL Roadmap (TODO)

### 16.1 Motivation

Scripted attackers (§4) provide diverse but ultimately predictable behaviors. Learned adversarial attackers enable:
- **Emergent attack strategies**: Attackers discover defender blind spots
- **Robustness via co-evolution**: Defenders trained against adaptive opponents generalize better
- **Research contribution**: Adversarial multi-agent observation→interception is a novel problem setting

### 16.2 Phased Development Approach

#### Phase 1: Scripted Attackers with Observe→Intercept (Current — M1–M6)

**Status**: Primary development focus

- Defenders learn role transition (OBSERVE ↔ INTERCEPT) against parametric scripted attackers
- Attacker behaviors: Kamikaze, Standard, Evasive, Stealth profiles (§4.2.3)
- Domain randomization provides behavioral diversity
- **Deliverable**: Trained defender policy with >50% defense success on 6v5

#### Phase 2: Learned Attackers (Future — M7–M9)

**Objective**: Replace scripted attackers with learned policies while retaining defender's observe→intercept capability

##### 16.2.1 Attacker Policy Architecture

**Observation space** (per attacker):
- Ego state: position, velocity (6D)
- Facility direction and distance (2D)
- Nearest defender: relative position, velocity (6D)
- Nearest interceptor: relative position, role indicator (4D)
- Alive teammates count (1D)

**Total**: ~19D per attacker

**Action space** (per attacker):
- 3D velocity command (same as defender platform actions)
- Evasion intensity (1D, continuous)

**Total**: 4D

##### 16.2.2 Training Paradigm Options

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Alternating training** | Freeze defenders, train attackers for N steps; swap | Simple implementation | Oscillation, forgetting |
| **Self-play with population** | Maintain population of defenders and attackers | Better diversity | Higher compute |
| **Asymmetric MARL** | Joint training with separate objectives | End-to-end | Harder to balance |
| **Regret minimization** | Game-theoretic equilibrium seeking | Principled | Complex implementation |

**Recommended initial approach**: Alternating training with experience replay from historical opponents.

##### 16.2.3 Reward Function for Attackers

$$r_{\text{attacker}}^{(t)} = r_{\text{approach\_fac}} + r_{\text{survival}} + r_{\text{evasion\_success}} + r_{\text{breach}}$$

| Term | Scale | Description |
|------|-------|-------------|
| $r_{\text{approach\_fac}}$ | 0.5 | Progress toward facility (distance reduction shaping) |
| $r_{\text{survival}}$ | 0.1 | Per-step bonus for remaining alive |
| $r_{\text{evasion\_success}}$ | 1.0 | Bonus for escaping pursuit after triggering evasion |
| $r_{\text{breach}}$ | +50.0 | Terminal bonus for reaching facility |
| $r_{\text{intercepted}}$ | -5.0 | Penalty upon interception |

##### 16.2.4 Curriculum for Adversarial Training

| Stage | Defender | Attacker | Duration |
|-------|----------|----------|----------|
| **S1** | Frozen (from Phase 1) | Learn against fixed defender | 100k steps |
| **S2** | Fine-tune | Frozen (from S1) | 50k steps |
| **S3** | Co-train | Co-train | 200k steps |
| **S4** | Population-based | Population-based | 300k+ steps |

### 16.3 Implementation Requirements

#### 16.3.1 Environment Modifications

- **Dual policy support**: Environment must accept separate action tensors for defenders and attackers
- **Attacker observation computation**: New observation pipeline for attacker agents
- **Attacker reward computation**: Separate reward function (opposing defender rewards)
- **Mixed agent types**: Support heterogeneous agent configurations

#### 16.3.2 Training Infrastructure

- **Multi-policy training**: SKRL/RLlib support for multiple policies
- **Policy freezing**: Ability to freeze one population during training
- **Historical opponent sampling**: Replay buffer of past policy checkpoints
- **Equilibrium metrics**: Compute exploitability, response quality

#### 16.3.3 New Files

```
iris_ma6/
├── attacker/
│   ├── learned_attacker_policy.py      # Attacker network architecture
│   ├── attacker_obs_builder.py         # Attacker observation computation
│   └── attacker_reward.py              # Attacker reward function
├── adversarial/                         # NEW directory
│   ├── adversarial_trainer.py          # Alternating/self-play training loop
│   ├── population_manager.py           # Historical opponent management
│   ├── equilibrium_metrics.py          # Nash gap, exploitability
│   └── adversarial_cfg.py              # Adversarial training configuration
```

### 16.4 Ablation Studies for Adversarial Phase

| Group | Focus | Variants |
|-------|-------|----------|
| **C1** | Training paradigm | `c1_alternating`, `c1_population`, `c1_joint` |
| **C2** | Attacker capability | `c2_observe_blind` (attackers can't see defenders), `c2_full_obs` |
| **C3** | Transfer | `c3_scripted_to_learned` (defender trained on scripted, tested on learned) |

### 16.5 Research Questions

1. **Do defenders trained against learned attackers generalize better to novel attack strategies?**
   → Compare transfer performance: scripted-trained vs adversarial-trained defenders

2. **Does the observe→intercept transition strategy remain effective against adaptive attackers?**
   → Analyze role transition patterns when attackers learn to exploit observation delays

3. **What emergent attack/defense strategies arise from co-evolution?**
   → Qualitative analysis of trained behaviors, coordination patterns

### 16.6 Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training instability in self-play | Non-convergence, cycling | Population-based diversity, replay from historical checkpoints |
| Attacker degeneracy (trivial strategies) | Defenders don't improve | Minimum attacker diversity constraints, behavioral rewards |
| Compute cost (2x agents) | Slower iteration | Start with small scale (3v2), progressive scaling |
| Forgetting during alternating training | Performance regression | Experience replay, elastic weight consolidation |

---

## Appendix A: V5 → V6 Key Changes Summary

| Aspect | V5 | V6 (Phase A: Scripted) | V6 (Phase B: Adversarial) |
|--------|-----|------------------------|---------------------------|
| Agents | 2–3 observers | 6 observer/interceptors | 6 observer/interceptors |
| Targets | 1, scripted motion | 1–10, scripted attack FSM | 1–10, learned policies |
| Action dim | 7 (velocity + gimbal + zoom) | 8 (+role) | 8 (+role) |
| Obs dim | 47–62 | ~227 | ~227 (defenders) / ~19 (attackers) |
| Role | Fixed (OBSERVE) | Dynamic (OBSERVE ↔ INTERCEPT) | Dynamic (OBSERVE ↔ INTERCEPT) |
| Objective | Minimize triangulation uncertainty | Facility defense | Facility defense (adversarial) |
| Target behavior | Linear / circular motion | Goal-directed attack + evasion | Learned attack strategies |
| Episode termination | Timeout only | Breach / all-killed / timeout / all-defenders-lost | Same |
| Curriculum | 7 dimensions | 12 dimensions, 4 phases | + adversarial stages (S1–S4) |
| Key metric | RMSE, tri_valid | Defense success rate, kill count | + Nash equilibrium, robustness |

## Appendix B: Expected File Structure

```
iris_ma6/
├── iris_ma_env6.py                    # Main environment (fork from V5)
├── iris_ma_env6_cfg.py                # Config dataclass
├── controller/
│   ├── point_mass.py                  # V5 reuse
│   └── gimbal_stabilizer.py           # V5 reuse
├── delay_system_v2/                   # V5 reuse + minor extension
│   ├── multi_agent_delay_system_v2.py
│   ├── delay_pipeline.py
│   └── derived_field_computers.py
├── bbox_raycaster/
│   └── bbox_raycaster.py             # V5 extension (multi-target)
├── triangulation/
│   └── triang_cov_reward_torch.py    # V5 reuse (per-target wrapper)
├── safety/
│   ├── safety_manager.py             # V5 reuse
│   ├── collision_detector.py         # V5 reuse
│   └── ttc_computer.py              # V5 minor extension
├── attacker/                          # NEW
│   ├── attacker_manager.py
│   ├── attacker_behavior_cfg.py
│   ├── attacker_wave_generator.py
│   ├── attacker_controller.py
│   └── intercept_resolver.py
├── curriculum/
│   └── curriculum_cfg.py             # V5 extension (12 dims)
├── randomization/
│   ├── randomizer.py                 # V5 extension
│   └── distance_based_generator.py   # V5 reuse
├── experiments/
│   ├── experiment_registry.py        # B1–B5 experiments added
│   ├── experiment_cfg.py
│   ├── evaluate.py                   # Defense metrics added
│   └── metrics/
│       ├── metric_tracker.py         # V5 extension
│       ├── timeseries_tracker.py     # V5 extension
│       └── trajectory_recorder.py    # V5 extension
└── adversarial/                       # Phase B: Adversarial MARL (TODO)
    ├── learned_attacker_policy.py    # Attacker network architecture
    ├── attacker_obs_builder.py       # Attacker observation computation
    ├── attacker_reward.py            # Attacker reward function
    ├── adversarial_trainer.py        # Alternating/self-play training loop
    ├── population_manager.py         # Historical opponent management
    ├── equilibrium_metrics.py        # Nash gap, exploitability
    └── adversarial_cfg.py            # Adversarial training configuration
```
