# Experiment: 2026-04-05_20-31-03_mappo_rnn_torch_2be3b9a3d9_control_gain_and_fidelity

**Commit**: `2be3b9a3d9`
**Date**: 2026-04-05
**Experiment ID**: N/A (manual run, pre-experiment-registry)
**Base**: `928b9585f2`<2026-04-01_13-19-18_mappo_rnn_torch_928b9585f2_min_lr_tuning>

**Status**: Aborted at 264k steps (of 400k target). Last checkpoint: agent_240000.pt.

---

## 1. Hypothesis
The 928b baseline trained on a simplified dynamics model (aerodynamics level 0, no drag, PX4-style
gains). Sysid-replicator analysis (ticket-008) showed 55% dynamics mismatch between iris_ma6 and
PX4 SITL. This run tests two simultaneous changes:

1. **PX4-matched controller gains**: Replace hand-tuned PX4-style gains with sysid-replicator output
   (optimized to match PX4 SITL step response via timeseries MSE).
2. **Full aerodynamic fidelity**: Enable level-3 aerodynamics (drag + rotor effects + blade flapping)
   with calibrated C_d=0.03 and gusts disabled (sigma_gust=0.0).

Additionally, action penalty weights were significantly increased (+15x action_sum, +15x action_delta)
and zoom action weight raised from 0.3 to 1.0, based on preliminary A8 ablation findings.

The hypothesis was that PX4-matched gains + realistic aerodynamics would improve sim-to-sim transfer
fidelity without degrading training stability, assuming the policy can adapt to the stiffer dynamics.

## 2. Configuration Delta
Changes from baseline `928b9585f2`:

```python
# Controller gains (sysid-replicator output)
velocity:
    Kp_vel: (1.5, 1.5, 3.0) -> (2.0416, 2.0416, 1.5355)
    Ki_vel: (0.25, 0.25, 0.3) -> (1.3005, 1.3005, 0.7767)
attitude:
    Kp_att: (6.5, 6.5, 2.8) -> (5.2060, 5.2060, 1.9334)
rate:
    Kp_rate: (0.15, 0.15, 0.2)   -> (0.3932, 0.3932, 0.3956)
    Ki_rate: (0.2, 0.2, 0.15)    -> (0.1805, 0.1805, 0.0900)
    Kd_rate: (0.003, 0.003, 0.001) -> (0.01451, 0.01451, 0.0)

# Aerodynamics
fidelity_level: 0 -> 3          # Full aero model
C_d: 1.0 -> 0.03               # Calibrated drag coefficient
sigma_gust: 2.0 -> 0.0         # Gusts disabled for controlled test

# Action penalties (from A8 preliminary findings)
action_sum_penalty_scale: -2.0 -> -30.0     # 15x increase
action_delta_penalty_scale: -1.0 -> -15.0   # 15x increase
action_weight[6] (zoom): 0.3 -> 1.0
action_delta_weight[6] (zoom): 0.3 -> 1.0
```

## 3. Results (264k steps, aborted)

### 3.1 Final Metrics
| Metric | Value | 928b (400k) | Delta |
|--------|------:|------:|------:|
| Mean Reward | 2501 | 3788 | **-1287** |
| Max Reward | 5680 | 5473 | +207 |
| Learning Rate | 0.000548 | 0.000100 | +448% |
| Policy Std Dev | 0.140 | 1.210 | **-1.07** |
| Value Loss | 0.000205 | 0.001011 | -79.7% |
| Policy Loss | -0.000136 | -0.002348 | weaker |
| pair_valid_rate | 0.677 | 0.861 | **-0.184** |
| triangulation (d0) | 24.0 | 44.4 | **-20.4** |
| bbox_center (d0) | 13.7 | 22.6 | **-8.9** |
| bbox_size (d0) | 35.1 | 44.4 | **-9.3** |
| collision_fraction | 0.0002 | 0.0008 | -75% |
| tracking_lost_fraction | 0.014 | 0.005 | **+180%** |

### 3.2 Head-to-Head vs 928b (baseline)

#### Policy Std Dev -- collapsed to ~0.14 (928b converged to 1.21)
| Step | 928b | 2be3 | Delta |
|-----:|------:|------:|------:|
| 4k | 0.644 | 0.429 | -0.215 |
| 20k | 0.308 | 0.110 | **-0.198** |
| 52k | 0.808 | 0.111 | **-0.697** |
| 84k | 1.207 | 0.110 | **-1.097** |
| 120k | 1.224 | 0.118 | **-1.106** |
| 200k | 1.215 | 0.155 | **-1.060** |
| 264k | -- | 0.140 | -- |

**Critical finding**: The policy standard deviation collapsed to ~0.11 by step 20k and never
recovered. The 928b baseline followed the expected trajectory of gradually increasing sigma
(0.3 -> 0.6 -> 1.0 -> 1.21) as the KL-adaptive scheduler explored. Here, sigma was crushed
immediately and stayed at 0.11-0.16 throughout training.

This is a **premature convergence / exploration collapse** -- the policy locked into a narrow
action distribution early, preventing the broad exploration needed for multi-agent coordination.

#### Learning Rate -- hyperactive oscillations
| Step | 928b | 2be3 | Pattern |
|-----:|------:|------:|---------|
| 4k | 0.00608 | 0.00880 | Both hot |
| 20k | 0.00030 | 0.00042 | Both dropped |
| 84k | 0.00044 | 0.00121 | Diverging |
| 164k | 0.00042 | **0.00321** | 2be3 spiking |
| 196k | 0.00028 | **0.00625** | 2be3 at max |
| 228k | 0.00010 | 0.00223 | 2be3 volatile |
| 264k | -- | 0.00055 | Still oscillating |

The KL-adaptive scheduler shows extreme oscillation in the 2be3 run (LR swings between 0.0005
and 0.007), indicating the policy is chronically undershooting the KL target. The low sigma
means small LR changes produce outsized KL effects, creating a feedback loop:
low sigma -> undershoot KL target -> scheduler raises LR -> oversized policy update ->
KL spike -> scheduler crashes LR -> undershoot again.

#### Mean Reward -- stagnation after 120k
| Phase | Steps | 928b | 2be3 | Delta | Notes |
|-------|------:|------:|------:|------:|-------|
| Early | 4k-20k | 1673->3244 | -1550->3347 | +103 | Both recover from init |
| Exploration | 36k-68k | 2989->3058 | 3113->2968 | -90 | Similar |
| Peak climb | 84k-116k | 3344->3958 | 3303->3958 | 0 | **Identical peaks** |
| Curriculum shift | 124k-132k | 2957->-- | 3328->-- | +371 | 2be3 hit less hard |
| Degradation | 148k-180k | 3662->3634 | 3177->2785 | **-849** | 2be3 declining |
| Collapse | 196k-228k | 3166->-- | 2274->2281 | **-885** | 2be3 stagnant |
| Terminal | 264k | -- | 2501 | -- | No recovery trend |

The 2be3 run matched the 928b baseline perfectly through 116k steps (both peaked at 3958).
After the curriculum shift at ~124k, 2be3 declined continuously. By 200k, reward was 2274
with no upward trend, justifying the abort.

#### Task Metrics -- pair_valid_rate collapse
| Step | 928b | 2be3 | Delta |
|-----:|------:|------:|------:|
| 20k | 0.950 | 0.938 | -0.012 |
| 84k | 0.890 | 0.837 | -0.053 |
| 116k | -- | 0.904 | peak |
| 164k | 0.830 | **0.781** | -0.049 |
| 196k | 0.770 | **0.667** | **-0.103** |
| 264k | -- | 0.677 | no recovery |

pair_valid_rate dropped to 0.67 by 196k and never recovered. The stiffer dynamics and
compressed sigma meant the policy couldn't adapt its formation geometry during the curriculum
ramp phase.

#### tracking_lost_fraction -- rising trend
```
4k:   0.40 (warmup)
20k:  0.01 (learned basic tracking)
68k:  0.04 (curriculum ramp -- rising)
116k: 0.00 (noise onset masks termination)
148k: 0.005
196k: 0.008
228k: 0.014  <- rising
264k: 0.014  <- stuck high
```
928b's tracking_lost_fraction was 0.005 at completion. The 2be3 run shows a persistent upward
trend, suggesting the policy is losing targets more frequently under the stiffer dynamics.

#### Action smoothness -- better (but from rigid policy)
| Step | 928b | 2be3 | Delta |
|-----:|------:|------:|------:|
| 20k | -- | 0.436 | -- |
| 100k | -- | 0.515 | -- |
| 200k | -- | 0.613 | -- |
| 264k | -- | 0.566 | -- |

Action RMS remained moderate (~0.5-0.6). The 15x action penalty increase was effective at
controlling action magnitude, but this came at the cost of exploration (policy sigma collapsed).

## 4. Analysis

### Why this run failed: confounded changes + exploration collapse

Three simultaneous changes made diagnosis difficult:
1. **Sysid-matched gains** (stiffer velocity loop: Kp_vel 1.5->2.04, Ki_vel 0.25->1.30)
2. **Level-3 aerodynamics** (drag forces, rotor effects change hover equilibrium)
3. **15x action penalty** (dramatically shrinks feasible action space)

The root cause of failure is **exploration collapse** (sigma ~0.14 vs expected ~1.21). The most
likely culprit is the 15x action penalty increase combined with stiffer gains:

- The stiffer gains amplify small actions into larger state changes.
- The heavier action penalty punishes larger actions.
- Together, they create a narrow basin where only tiny actions are rewarded.
- The policy converges to tiny sigma immediately (by 20k steps).
- Once sigma is locked low, the KL-adaptive scheduler cannot recover exploration.

**The sysid-matched gains themselves may be fine** -- the run tracked the baseline perfectly
through 116k (before post-curriculum dynamics became dominant). The problem is that the action
penalty magnitude was not re-calibrated for the new gain structure.

### Positive signal: identical early performance
The fact that both runs had indistinguishable reward trajectories through 116k suggests the
new gains and aerodynamics did not harm basic learning. The divergence only occurred after
curriculum transitions demanded adaptation, which the low-sigma policy could not provide.

### Why pair_valid_rate collapsed
With sigma ~0.14 (vs 1.21 in baseline), the policy has ~8.6x less action variance.
When curriculum introduces observation noise and delay, the policy cannot explore alternative
formation geometries to maintain valid detection pairs. It gets stuck in its early-training
formation pattern, which is not robust to observation degradation.

## 5. Decision
- **Abort confirmed at 264k** -- no recovery mechanism with sigma stuck at 0.14.
- **Do NOT use this run's checkpoints** -- the low-sigma policy is fundamentally underexplored.
- **Decouple the changes for next experiments**:
  - Test sysid gains + level-3 aero **without** action penalty changes (use 928b action weights).
  - Test action penalty tuning separately (A8 ablation already in progress).
  - Only combine gains + penalties after both are independently validated.
- The A8 weight_config_a (5x penalty) and weight_config_b (15x penalty) ablations running on
  2026-04-05 will determine the correct action penalty scale.
- **Sysid gains remain promising** -- the 0-116k parity with baseline suggests they can work
  with appropriate action penalty calibration.
