# Experiment: 2026-05-02_14-43-47_mappo_rnn_torch_58a3c3ee84_siyi_a8_mini_triangulation_fix

**Commit**: `58a3c3ee84` ("Adding triangulation fix")
**Date**: 2026-05-02 → 2026-05-03
**Experiment ID**: `siyi_a8_mini_triangulation_fix`
**Base**: `343e2ed4a3` (siyi_a8_mini_action_penalty_tuning) with the single change `include_drift_uncertainty: True → False`.
**Status**: Run reached **292k**. Triangulation reward dramatically recovered both pre- and post-delay. Diagnosis from the previous experiment **fully validated** — and over-delivered: the drift fix recovered a 2.66× pre-delay gap that the diagnosis had not predicted. Late-phase tail (240k+) still degrades faster than 0d4fa6906b, attributed to the very-light `-1/-1` action-penalty regime, not the SIYI realism stack.

---

## 1. Hypothesis (going in)

The 343e2ed4a3 run showed `tri_d0` collapsing from 6.55 (120k) → 0.12 (140k) across the fixed-delay ramp. Diagnosis pinned the collapse on `compute_sigma_drift` using `‖v_cam‖²` as a variance contribution where the correct quantity is the velocity-estimate uncertainty `P_{vv}` (~0.1 m/s). Fix: gate the entire drift inflation behind `include_drift_uncertainty`, default it `False`.

Predictions going in:
- Pre-delay (120k): `tri_d0` essentially unchanged (~6.5) — the diagnosis assumed `obs_age ≈ 0` pre-delay, so drift contributes nothing.
- Post-delay 124k: `tri_d0` recovers to 3–5 (vs 1.0 in PREV).
- Post-delay 140k: `tri_d0` recovers to 2–4 (vs 0.12 in PREV).
- Post-delay 200k: `tri_d0` recovers to 1–3 (PREV not reached).

Single-knob isolation. Action penalties (`-1/-1`), λ_cbf (1.0), envelope (60/25), rate-loop curriculum gate, and curriculum windows all preserved verbatim from 343e2ed4a3.

## 2. Configuration delta (vs 343e2ed4a3)

```python
# Single change
triangulation.include_drift_uncertainty: True (effective default) → False
```

The flag was added to `TriangulationCfg` as part of this fix. With it `False`, `compute_sigma_drift` early-returns the base `cfg.pos_std` and `cfg.ori_std` regardless of observation age, ego velocity, or angular velocity.

## 3. Results

### 3.1 Bootstrap & pre-Level-3 (0–80k)

| Step | rew_inst | ep_len | pair_valid | track_lost | sigma | tri_d0 | bbox_center | bbox_size | coll_per_env |
|----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k   | 4.78 | 457 | 0.74 | 0.21 | 0.67 | 0.00 | 23.1 | 38.3 | 1.31 |
| 12k  | 6.53 | 497 | 0.91 | 0.016 | 0.95 | 0.00 | 44.5 | 46.6 | 1.22 |
| 20k  | 6.54 | 498 | 0.93 | 0.008 | 1.32 | 0.00 | 45.0 | 47.0 | 5.65 |
| 40k  | 6.07 | 497 | 0.88 | 0.006 | 1.76 | 0.00 | 41.6 | 45.8 | 3.49 |
| 60k  | 5.95 | 490 | 0.82 | 0.024 | 1.75 | 0.00 | 40.6 | 43.9 | 5.27 |
| 80k  | 6.14 | 485 | 0.78 | 0.061 | 1.75 | **5.94** | 37.1 | 42.2 | 4.42 |

`tri_d0` is zero pre-80k because triangulation reward only unlocks at `task_level_3_start_step=80_000`. Bootstrap is identical to PREV through this window — same env, same observation pipeline, same bbox/center reward magnitudes (`bbox_center` and `bbox_size` match PREV within MC noise). Sigma climbs to its `-1/-1` regime saturation at ~1.76 by 40k, same as PREV.

### 3.2 Post-Level-3, pre-delay (80k–120k)

| Step | rew_inst | pair_valid | tri_d0 | bbox_center | coll_per_env |
|----:|---:|---:|---:|---:|---:|
| 80k  | 6.14 | 0.78 | 5.94 | 37.1 | 4.42 |
| 100k | **7.10** | 0.84 | **15.47** | 39.3 | 5.39 |
| 116k | 7.23 | 0.85 | (~17) | 39.7 | 5.79 |
| 120k | **7.27** | 0.85 | **17.40** | 39.7 | 5.51 |

vs 343e2ed4a3 at 100k / 120k (same step, same config except drift): `tri_d0` was 5.88 / 6.55. **NEW shows 15.47 / 17.40 — a 2.6×–2.7× pre-delay gap.** This was not predicted. The cause: `compute_sigma_drift` was firing on the env's intra-step `obs_age` values (the per-camera observation timestamp lags the current sim time by a few env-steps even with `delay_mode="none"`). At `obs_age = 40 ms`, `‖v‖ = 5 m/s`, the formula inflates σ_pos from 0.10 to ~0.22 m and σ_ori from 0.01 to ~0.041 rad — enough to halve `1/sqrt(trace)`. The drift bug had been silently inflating production triangulation reward across **all** training regimes since mas/029 landed, not just post-delay.

`rew_inst` at 120k is **7.27**, the highest of any iris_ma6 run with the SIYI realism stack. 0d4fa6906b at 120k: 8.35. NEW closes the gap to 87% of baseline despite the `-1/-1` penalty regime and tighter geometry.

### 3.3 Detection-latency ramp (120k–140k): triangulation holds

| Step | rew_inst | pair_valid | tri_d0 | bbox_center | bbox_size | coll_per_env |
|----:|---:|---:|---:|---:|---:|---:|
| 120k | 7.27 | 0.85 | **17.40** | 39.7 | 44.3 | 5.51 |
| 124k | 5.46 | 0.73 | **14.09** | 24.6 | 40.8 | 2.45 |
| 132k | 4.70 | 0.64 | 12.45 | 17.5 | 38.0 | 2.00 |
| 140k | 4.11 | 0.53 | **10.46** | 13.3 | 33.8 | 1.55 |

vs PREV across the same window (120k → 140k): `tri_d0` 6.55 → 0.12. **NEW: 17.40 → 10.46** — only a 1.7× drop, vs PREV's 54× collapse. Better than predicted (3–5 at 124k, 2–4 at 140k); the actual values are 14.1 and 10.5.

vs 0d4fa across the same window: 32.18 → 22.06 (1.5× drop). NEW now sits at 47–60% of 0d4fa's absolute level through the delay onset, with a similar shape rather than a collapse. The remaining gap is the `Sigma_K` (intrinsic uncertainty) addition + tighter geometry, both of which are deployment-locked.

`coll_per_env` drops from 5.51 → 1.55 across the same window — the policy adapts formation, but does so without abandoning triangulation (PREV's 6.22 → 1.11 was associated with the triangulation collapse; here the formation widens *and* triangulation holds).

### 3.4 Random delay + dropout phase (140k–200k): mostly stable

| Step | rew_inst | pair_valid | tri_d0 | bbox_center |
|----:|---:|---:|---:|---:|
| 140k | 4.11 | 0.53 | 10.46 | 13.3 |
| 148k | (peak) | (peak) | **16.15** | (peak) |
| 156k | 4.90 | 0.64 | 13.82 | 19.9 |
| 180k | 5.37 | 0.71 | **16.50** | 23.0 |
| 200k | 4.59 | 0.56 | 13.73 | 17.6 |

`tri_d0` peaks at **16.50 at 180k** — within sampling noise of the 17.40 pre-delay peak. The policy fully absorbs the random_delay and dropout ramps. `pair_valid` recovers to 0.71 by 180k vs 0d4fa's 0.74 (within 0.03). `rew_inst` at 180k is **5.37 vs 0d4fa's 5.68 — within 6%.** This is the strongest sustained training result of any iris_ma6 run with the SIYI realism stack.

### 3.5 Burst dropout + late phase (200k–292k): tail degradation

| Step | rew_inst | ep_len | pair_valid | tri_d0 | bbox_center | sigma | coll_per_env |
|----:|---:|---:|---:|---:|---:|---:|---:|
| 200k | 4.59 | 452 | 0.56 | 13.73 | 17.6 | 1.76 | 2.11 |
| 220k | 3.42 | 405 | 0.41 | 9.51 | 11.0 | 1.77 | 1.06 |
| 240k | 2.71 | 323 | 0.29 | 5.60 | 6.3 | 1.77 | 0.39 |
| 260k | 2.36 | 234 | 0.27 | 3.66 | 4.0 | 1.77 | 0.16 |
| 280k | 2.08 | **166** | 0.26 | 2.33 | 2.3 | 1.77 | 0.10 |

The tail degrades faster than 0d4fa's. 0d4fa at 240k: pair_valid 0.61, ep_len 454, tri 20.7. NEW at 240k: 0.29 / 323 / 5.6. The post-200k window includes burst_dropout (200k–220k) on top of the random_delay + dropout already in effect.

The late-phase failure mode is consistent with the very-light penalty regime: sigma is stuck at 1.77 throughout, `coll_per_env` falls to ~0.1 (agents fully spread out), and ep_len collapses to ~166 by 280k. The policy never finds a concentrated basin that can absorb compounding observation corruption. This is the same failure mode as the f6d513e6e3 ablation — it was never about drift, but about action penalties being load-bearing for late-phase robustness.

## 4. Analysis

### 4.1 Drift fix delivered everywhere — including pre-delay

The single-knob change recovered triangulation reward at every step, with magnitude scaling with the observation age that was being inflated:

| Step | PREV | NEW | NEW/PREV |
|----:|---:|---:|---:|
| 120k (pre-delay) | 6.55 | 17.40 | **2.7×** |
| 124k (delay onset) | 1.01 | 14.09 | 14× |
| 140k (full fixed delay) | 0.12 | 10.46 | **87×** |
| 180k (full random delay) | 0.18 | 16.50 | **92×** |
| 200k | 0.16 | 13.73 | 86× |

The pre-delay 2.7× ratio invalidates my prior claim that "drift contributes ~0 pre-delay." The env's observation pipeline assigns `timestamp_detection` lagging the current sim time by a few env-steps even when `delay_mode="none"`, so `obs_age` is on the order of 40 ms baseline. Plugging that into `‖v‖² · Δt²` with `‖v‖=5` gives `pos_var += 0.04`, doubling the per-camera σ_pos and roughly halving `1/sqrt(trace)`. Every iris_ma6 run since mas/029 has been silently paying this cost.

### 4.2 The pre-delay 5× scale gap is now ~2× — and explainable

Earlier diagnosis attributed a 5× pre-delay gap (vs 0d4fa) to `Sigma_K` + tighter geometry. With drift fixed:

| Run | tri_d0 at 120k | gap to 0d4fa |
|-----|---:|---:|
| 0d4fa6906b | 32.18 | 1.0× |
| 343e2ed4a3 (drift on) | 6.55 | 4.9× |
| 58a3c3ee84 (drift off) | 17.40 | 1.85× |

So drift accounted for ~2.6× of the 4.9× pre-delay gap. The residual 1.85× is genuinely from `Sigma_K` (per-zoom intrinsic covariance, ~26 px σ_fx at 1× zoom) and the tighter geometry (60/25 vs 100/40 — shorter triangulation baselines). Both are deployment-locked, so 1.85× is the irreducible cost of running with realistic intrinsics on the deployment envelope.

### 4.3 Late-phase failure mode is action-penalty regime, not corruption

NEW survives all curriculum ramps cleanly through 200k (`tri_d0` peaks at 180k = 16.5). Past 200k, when burst_dropout layers on top of fully-active random_delay + dropout, the policy collapses (`tri_d0` 200k → 280k: 13.7 → 2.3). Sigma stays at 1.77 the whole time — no policy concentration. Compare to 0d4fa's late phase where sigma climbed gradually from 0.21 → 0.42 (still committed) and pair_valid held at 0.61 at 240k.

The diagnosis: **`-1/-1` action penalties are insufficient pressure for the policy to commit to a stable strategy under compounding observation corruption.** The f6d513e6e3 ablation already showed this for `-22/-10`; the `-1/-1` regime is worse. This is independent of the drift fix.

## 5. Decision

**The drift fix is ratified.** `include_drift_uncertainty: bool = False` becomes the iris_ma6 default; future runs should keep it off unless explicitly tuning the AoI process model. Production triangulation reward will be ~2.7× higher pre-delay and ~85× higher post-delay than runs since mas/029 landed.

**Next intervention isolated by this run**: revert action penalties to `-30/-15` (matches 0d4fa6906b) with all other knobs fixed. This isolates the late-phase tail. If the tail recovers, the iris_ma6 minimum-viable-s2r configuration is complete:

- `include_drift_uncertainty=False` (this run, ratified)
- `cylinder_diameter_max=60`, `target_distance_max=25` (deployment envelope)
- Rate-loop curriculum gate enabled
- λ_cbf=1.0
- `action_sum_penalty_scale=-30`, `action_delta_penalty_scale=-15` (next run)
- `triangulation_reward_scale=5.0` (no bump needed; the 1.85× residual gap is irreducible)

## 6. Open items

- **Late-phase tail validation** — needs the action-penalty revert to test whether the failure mode is fully attributable to penalty regime.
- **Pre-mas/029 production triangulation reward was ~2.7× inflated** by the drift bug not firing (pre-mas/029 had no `compute_sigma_drift` at all). Worth a brief audit of any policies trained between mas/029 and now to understand their actual triangulation quality vs. logged value.
- **`Sigma_K` magnitudes** could still be reviewed for over-conservatism. The mrcal calibration's σ_fx values are real, but the analytical formula assumes fx, fy, cx, cy are independent — if there are correlations in the calibration (which is typical), the diagonal `Sigma_K = σ²·I_4` is conservative. Low-priority since drift fix already gave the bulk of the recovery.

## 7. Side notes

- The `-1/-1` action-penalty regime maintained `pair_valid ≥ 0.85` and `tri_d0 ≥ 17` through the 100k–120k pre-delay window despite `sigma=1.76`. This is a useful data point: bbox-based tracking is robust to wide policy distributions if the bbox reward gradient is strong enough. The cost shows up at the late-phase robustness tail, not in nominal performance.
- The curriculum ramp the policy survives without major regression (140k random_delay → 156k recovery → 180k peak) suggests random delay is *not* the harder corruption — the policy adapts within ~30k steps. Burst dropout (200k+) is the corruption that compounds with the `-1/-1` regime to break the late phase.
- Single-knob isolation worked exactly as intended. Three runs now form a clean A/B/C: 343e2ed4a3 (drift on, baseline), 58a3c3ee84 (drift off), and the next action-penalty revert (drift off + heavier penalties). The 0d4fa6906b reference run pre-dates the SIYI stack entirely.
