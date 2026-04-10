# Literature Survey: Observation, Reward, Curriculum, and DR Design for Multi-Agent Drone RL

**Date**: 2026-04-10
**Context**: Ticket-022 observation redesign for iris_ma6. Survey of works with concrete ablation results.

---

## 1. Observation Frame: Ego-Centric / Body-Frame vs World-Frame

### Aerial_To_Aerial_Interception (TU Delft PATS-X, 2025)
- **Codebase**: `/home/usrg/source/Aerial_To_Aerial_Interception`
- **Default obs**: `rel_pos+vel_body` (8D): direction(3) + magnitude(1) + vel_direction(3) + vel_magnitude(1)
- **Finding**: Body-frame relative position and velocity is "most effective"; temporal history adds no benefit beyond noise filtering
- **Result**: 99.1% capture rate, 0.85s median time-to-first-interception (motor-level)
- **Key file**: `src/utils/observations.py`

### Dionigi 2024 — "The Power of Input" (arXiv: 2410.07686)
- **Ablation (hovering, combined position error)**:
  - `{e_W, R, u}` (world-frame error + rotation matrix + prev action): **2.7 cm**
  - `{e_B, u}` (body-frame error + prev action): **3.0 cm**
  - `{e_B, omega, u}`: **2.7 cm**
  - `{e_W, R, omega, u}` (adding angular velocity): **4.6 cm** (worse)
  - Adding velocity `{e_W, v_W, R, u}`: **6.7 cm** (degraded from 5.5 cm without)
  - Configs without rotation matrix: **failed to converge**
- **Sim-to-real**: body-frame `{e_B, omega, u}` achieved **1.21 m/s** vs world-frame `{e_W, R, u}` at **1.06 m/s**
- **Key takeaway**: More information does NOT always help. Rotation matrix essential. Body-frame transfers better.

### Zhao 2024 — "What Matters in Zero-Shot Sim-to-Real" (arXiv: 2412.11764, RA-L)
- Rotation matrix representation: **best tracking** across all speeds
- Quaternion: slight degradation (4D discontinuity)
- Excluding linear velocity: **significant performance drop**
- Including previous action: **slight performance decrease**
- Adding time vector to critic: **greatly enhances tracking accuracy**
- **Five key factors**: (1) rotation matrix + velocity in actor, (2) time vector in critic, (3) action difference regularization, (4) system ID with selective randomization, (5) large batch sizes

### Summary for iris_ma6
- Heading-frame (yaw-only rotation) redesign is well-supported by literature
- [cos psi, sin psi] over Euler yaw avoids wrapping discontinuity (Zhao confirms rot-mat > Euler)
- Be selective about observation features: convergence angle and baseline features justified because they match FIM reward objective

---

## 2. Reward Term Ablations

### Zhao 2024 — Action Smoothness Penalties (real-world figure-eight, normal speed)
| Penalty type | Tracking error (m) |
|---|---|
| Action difference `\|u_t - u_{t-1}\|_2` | **0.028** (best) |
| Action magnitude `\|u_t\|_2` | 0.066 |
| Jerk penalty | 0.047 (diverged at fast speed) |
| Snap penalty | diverged at normal+fast |
| Low-pass filter | diverged at all speeds |
| Action clipping | 0.077 |

**Conclusion**: Action-difference >> action magnitude >> snap/jerk. Higher-order penalties destabilize.

### Aerial_To_Aerial_Interception — Reward Structure
- **Active reward**: `effective_gain = (prev_distance - current_distance) - evader_displacement`
- Rate penalty: `RATE_PENALTY = 0.001`
- Smoothing gamma sweep: 0 to 90 (13 configurations tested)
- Boundary shaping: `BOUNDARY_PENALTY_WEIGHT = 0.25`
- Out-of-bounds: `-5.0`, Capture bonus: `-2.0` (negative = reward)

### Safe Heterogeneous Multi-Agent RL (arXiv: 2601.08327, 2026)
- Progressive reward ablation:
  - R1 (distance only): **failed** -- reward stays near zero
  - R2 (distance + goal): clear convergence
  - R3 (+ collision): similar to R2 but **lower variance**
  - R4 (+ communication diversity): highest reward but **higher variance**

### MonoRace (arXiv: 2601.15222, 2025)
- 8 reward terms: progress, gate bonus, angular rate penalty, offset penalty, perception penalty, motor change penalty, low-action penalty, crash penalty
- Motor change penalty added post-hoc after observing bang-bang behavior

### Curriculum Racing (arXiv: 2602.24030, 2026)
- Without obstacle-avoidance reward: success drops **100% -> 36.7%**

### iris_ma6 Reward Weight Analysis and Recommendation
Current per-step magnitudes (step_dt = 0.04):

| Term | Scale | Typical per-step | Role |
|---|---|---|---|
| bbox_center | +60.0 | 0 to 2.4 | Primary tracking |
| bbox_size | +60.0 | 0 to 2.4 | Distance regulation |
| triangulation | +5.0 x progress | 0 to 0.2 | Task objective |
| action_sum | -10.0 | -0.02 to -0.4 | Energy |
| action_delta | -5.0 | -0.01 to -0.2 | Smoothness |
| collision | -100.0 | 0 or -4.0 (spike) | Hard constraint |
| altitude | -100.0 | 0 or continuous | Hard constraint |
| target_prox | -50.0 x progress | 0 or continuous | Soft constraint |
| CBF | -1.0 x progress | continuous | Soft constraint |

**Issue**: BBox rewards dominate triangulation by ~12x. Once tracking is learned, formation geometry signal may be too weak.

**Suggested rebalancing** (based on Zhao + MARL literature):
```
triangulation_reward_scale: 5.0  -> 20.0   (4x, closer to bbox magnitude)
action_sum_penalty_scale:  -10.0 -> -5.0    (halve, less important per Zhao)
action_delta_penalty_scale: -5.0 -> -10.0   (double, more important per Zhao)
```

---

## 3. Curriculum Learning

### CRUISE (arXiv: 2510.22570, 2025) — Multi-Drone Racing
| Scenario | CRUISE | VANILLA (no curriculum) |
|---|---|---|
| Ring Track, 2 drones | 4.4 m/s, 100% | 2.8 m/s, 50% |
| Ring Track, 3+ drones | 91-93.5% | **0% (total failure)** |
| Figure-8, 3 drones | 98% | 43.3% |
| Figure-8, 4 drones | 97% | 30% |

**Strongest evidence**: Without curriculum, VANILLA achieves 0% with 3+ agents.

### Curriculum Racing (arXiv: 2602.24030, 2026)
- One-step learning (no curriculum): **0% success** across all tracks
- Multi-stage curriculum: **100% success**
- GRU ablation: Without GRU, success drops to **70-80%** (from 100%)

### OPEN Multi-UAV Pursuit-Evasion (arXiv: 2409.15866, 2024)
- Full OPEN: **100% capture, 329.6 steps**
- Without Adaptive Environment Generator: **72.7%, 602.2 steps**
- Baseline MAPPO: **61.7%, 541.0 steps**

### Implication for iris_ma6
- Current two-phase curriculum (progress_coord 0->50k, progress_safety 40k->80k) is well-designed
- Consider a third phase that increases triangulation weight once tracking proficiency is achieved

---

## 4. Domain Randomization

### Zhao 2024 — DR vs System ID (with accurate calibration, normal speed)
| Configuration | Tracking error (m) |
|---|---|
| SysID only (accurate mass) | **0.028** |
| SysID + DR 10% | 0.041 (worse) |
| SysID + DR 30% | 0.066 (much worse) |
| Offset +30% mass (no DR) | diverged |

**Key insight**: DR only helps for inaccurately calibrated parameters. Accurate SysID alone outperforms blind DR.

### Swift (Kaufmann et al., Nature 2023)
- Explicitly does NOT use DR for dynamics
- Uses **residual model fine-tuning** instead: train in sim, deploy, collect real data, learn residual correction `f_corrected = f_nominal + f_residual`, retrain
- DR baseline achieved **0% track completion** in realistic conditions
- Residual approach maintained ~100% completion

### Aerial_To_Aerial_Interception — DR Sweep
- Levels tested: **0%, 10%, 20%, 30%** across 38 parameters
- Motor policies need 10-20% DR for hardware transfer
- Acceleration policies did NOT transfer regardless of DR level
- CTBR policies transfer better with less DR

### Learning on the Fly (UZH RPG, arXiv: 2508.21065, RA-L 2026)
- DATT (DR baseline): 20M steps, 0.231 m hovering error
- Proposed (differentiable adaptation): 4.5M steps + 3 adaptation steps, **0.105 m** (55% improvement)

### AirGym (arXiv: 2504.15129, 2025)
- With wind+DR: **0.043 m error, 60% success**
- Without DR: **0.073 m, 30% success**

### Implication for iris_ma6
- **Selective DR** of uncertain parameters > uniform DR of everything
- Priority DR targets for iris_ma6: cascade controller parameters (gains, time constants, saturations), since the velocity-command action space relies on a modeled cascade controller
- Accurate system identification of known parameters (mass, inertia) should precede DR

---

## 5. Gavin2024 — Most Directly Relevant Work

**Paper**: "Multi-Agent RL based Drone Guidance for N-View Triangulation" (ICUAS 2024)
**PDF**: `/home/usrg/IsaacPX4/2026-IROS/bib/pdf/Gavin2024.pdf`

### Observation (Section IV-C)
- **World-frame absolute positions only**: `o_i = (p_i, p_1, ..., p_{i-1}, p_{i+1}, ..., p_n, p_T)`
- No velocity, orientation, camera rays, gimbal state, or uncertainty
- Omnidirectional cameras, no restricted FOV, no occlusion, no noise
- Action: position increment in [-1,1]^3, scaled to 20cm sphere (holonomic, ~2m/s)

### Reward (Eq. 15)
```
r_i = 1 / sqrt(Tr(Sigma_X))   if safe conditions met
    = -r_penalty                otherwise (collision/ground/proximity)
```
- `Sigma_X` = full 3x3 covariance matrix from analytical Jacobian propagation (Eq. 9-10)
- Uncertainty is **only in the reward, NOT in the observation**
- Policy must implicitly learn geometry-uncertainty relationship from reward signal alone

### Uncertainty Model (Section IV-D)
- Analytical model via implicit function theorem: `J_X Sigma_X J_X^T = J_v Sigma_v J_v^T`
- Input noise: Gaussian on polar (theta) and azimuth (phi) angles, sigma = 0.003 rad
- Covariance propagated through triangulation Jacobians (Eq. 12-14)
- Validated against Monte Carlo simulation (Fig. 2)

### Results (Table II)
| Metric | 2 Trackers (train) | 2 Trackers (Bullet) | 3 Trackers (train) | 3 Trackers (Bullet) |
|---|---|---|---|---|
| Mean angle (deg) | 91.14 | 102.0 | 96.56 | 85.44 |
| Std dev (deg) | 9.15 | 9.98 | 13.64 | 26.34 |
| Crash rate (%) | 12.26 | 22.0 | 6.90 | 14.34 |

- 2 trackers converge to ~91 deg (optimal for 2-view triangulation)
- 3 trackers show higher angle variance and crash rates in realistic sim

### Limitations (relevant to iris_ma6 design decisions)
- No drone dynamics model (holonomic point mass)
- Perfect omnidirectional cameras (no FOV, no gimbal)
- Perfect position knowledge (no delay, no noise, no partial observability)
- Fixed target only in training; moving target tested but not trained on
- No curriculum, no domain randomization

### Key takeaway for iris_ma6
- `1/sqrt(Tr(Sigma))` validated as sufficient reward for driving formation geometry
- iris_ma6's decision to feed scalar uncertainty into actor obs is justified by the much harder POMDP (delay, noise, restricted FOV, gimbal) where the policy can't easily infer uncertainty from geometry alone
- Gavin2024's simplistic obs works only under perfect-information assumptions

---

## 6. Action Space and Sim-to-Real Transfer

### Kaufmann & Bauersfeld (ICRA 2022, arXiv: 2202.10796) — Action Space Ablation
- Body-rate + thrust (CTBR): most robust to sim-to-real transfer
- Single rotor thrust (SRT): most sensitive to control delays
- Linear velocity (LV): does not represent dynamic constraints correctly

### Aerial_To_Aerial_Interception — Control Abstraction Comparison
| Level | Actions | Sim Performance | Hardware Transfer |
|---|---|---|---|
| Motor RPM (4D) | Direct rotor speeds | Best (99.1%, 0.85s) | Transfers with 10-20% DR |
| CTBR (4D) | Thrust + body rates | Good | Better transfer than motor |
| Acceleration (3D) | Desired linear accel | Fast in sim | **Did NOT transfer** |

### Implication for iris_ma6
- iris_ma6 uses **velocity commands** with a modeled cascade controller (attitude -> body rate -> thrust)
- This is higher abstraction than CTBR. Transfer risk is mitigated because the cascade controller dynamics are modeled in sim (policy trains with controller-in-the-loop)
- **Priority for sim2real**: DR of cascade controller parameters (gains, time constants, saturations)
- **Residual model fine-tuning** (Swift approach) is applicable: collect real flight data, learn residual correction to cascade controller model, retrain. Implementable in IsaacLab by injecting learned residual into post-physics step

---

## 7. Additional Relevant Works

### Reactive Aerobatic Flight (arXiv: 2505.24396, 2025)
- Body-frame observations, progressive curriculum
- DR: drag +/-50%, control input +/-20%, latency 4-36ms

### OPEN Multi-UAV Pursuit-Evasion (arXiv: 2409.15866, 2024)
- Ego-centric per-agent: quaternion, linear velocity, real/predicted relative position
- Real-world zero-shot on Crazyflie quadrotors

### PUARL — Perception Uncertainty-Aware RL (MobiCom 2024)
- Addresses perception uncertainty from relative position measurement noise in multi-agent pursuit
- Directly applicable to triangulation uncertainty problem

---

## References

1. Aerial_To_Aerial_Interception — TU Delft PATS-X (local: `/home/usrg/source/Aerial_To_Aerial_Interception`)
2. Dionigi et al. 2024 — arXiv: 2410.07686
3. Zhao et al. 2024 — arXiv: 2412.11764 (RA-L)
4. Kaufmann et al. 2023 — Nature (Swift)
5. Pfeiffer et al. 2025 — arXiv: 2601.15222 (MonoRace)
6. CRUISE 2025 — arXiv: 2510.22570
7. Curriculum Racing 2026 — arXiv: 2602.24030
8. OPEN 2024 — arXiv: 2409.15866
9. Safe Heterogeneous MARL 2026 — arXiv: 2601.08327
10. UZH RPG 2026 — arXiv: 2508.21065 (Learning on the Fly)
11. Kaufmann & Bauersfeld ICRA 2022 — arXiv: 2202.10796
12. AirGym 2025 — arXiv: 2504.15129
13. Gavin et al. ICUAS 2024 — DOI: 10.1109/ICUAS60882.2024.10556867
14. PUARL — MobiCom 2024 — ACM DL: 10.1145/3636534.3694724
