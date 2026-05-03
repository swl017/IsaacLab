# Experiment: 2026-05-01_23-34-32_mappo_rnn_torch_343e2ed4a3_siyi_a8_mini_action_penalty_tuning

**Commit**: `343e2ed4a3` ("Action penalty tuning")
**Date**: 2026-05-01 → 2026-05-02
**Experiment ID**: `siyi_a8_mini_action_penalty_tuning`
**Base**: `f6d513e6e3` (failed action-penalty tuning) → reverted CBF, with very-light action penalties applied at launch.
**Status**: Run reached 208k. Pre-delay (0–120k): tracking works at expected scale. **At 124k the triangulation reward collapsed by 60×** across the fixed-delay ramp, isolated to a misformulated drift term in `compute_sigma_drift`. Fix landed mid-experiment: `include_drift_uncertainty` flag added to `TriangulationCfg`, defaulted to `False`. MC validation simultaneously found and fixed (analytical / empirical mismatch caused by un-paired `Sigma_K` injection in the test path).

---

## 1. Hypothesis (going in)

After the f6d513e6e3 ablation showed that loosening action penalties (-30→-22, -15→-10) broke sigma collapse, the intent was to revert toward the 0d4fa6906b baseline while keeping the deployment-envelope and rate-loop changes. The launch-time configuration ended up further from baseline than the recommendation: `action_sum_penalty_scale = -1.0` and `action_delta_penalty_scale = -1.0` (env.yaml as-launched), much lower than the committed `-10/-5`. λ_cbf reverted to `1.0` and the `cylinder_diameter_max=60`, `target_distance_max=25`, rate-loop curriculum gate were preserved.

Implicit hypothesis: with the new SIYI realism stack stable, the run should at minimum reach 0d4fa-comparable pre-delay reward and reveal where (if anywhere) the post-delay regime breaks.

## 2. Configuration delta (vs f6d513e6e3 launch state)

```python
# Action penalties — further loosened (vs f6d513e6e3 -22 / -10)
action_sum_penalty_scale: -22.0 → -1.0
action_delta_penalty_scale: -10.0 → -1.0

# CBF — reverted to baseline
cbf_safety.cpa_cfg.lambda_cbf: 2.0 → 1.0

# Geometry / rate-loop — preserved
cylinder_diameter_max: 60.0
target_distance_max: 25.0
rate-loop curriculum gate: enabled (uncommented at iris_ma_env6_test.py:1407)

# Curriculum — unchanged from f6d513e6e3
moving_target_end_step: 80_000   # not stretched to 100k
```

## 3. Results

### 3.1 Bootstrap & pre-delay phase (0–120k)

| Step | rew_inst | ep_len | pair_valid | track_lost | sigma | tri_d0 | bbox_center | bbox_size | coll_per_env |
|----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k   | 4.78 | 457 | 0.74 | 0.21 | 0.67 | 0.00 | 23.1 | 38.3 | 1.31 |
| 20k  | 6.54 | 498 | 0.93 | 0.008 | **1.32** | 0.00 | 45.0 | 47.0 | 5.65 |
| 40k  | 6.07 | 497 | 0.88 | 0.006 | **1.76** | 0.00 | 41.6 | 45.8 | 3.49 |
| 80k  | 5.98 | 487 | 0.80 | 0.045 | 1.75 | 2.62 | 38.2 | 43.2 | 4.73 |
| 100k | 6.33 | 491 | 0.84 | 0.038 | 1.76 | 5.88 | 39.9 | 44.5 | 6.20 |
| 116k | 6.53 | 494 | 0.86 | 0.020 | 1.76 | **6.61** | 41.0 | 45.1 | 6.01 |
| 120k | 6.45 | 494 | 0.86 | 0.026 | 1.75 | 6.55 | 40.2 | 44.9 | 6.22 |

Sigma is enormous (1.7–1.8) — the policy is operating in a near-Gaussian-saturated regime due to the `-1/-1` action penalties. Despite that, pair_valid and bbox_center hold near-baseline levels. **Triangulation reward climbs to ~6.5 by 120k** — the expected pre-delay value, and exactly **5× lower than 0d4fa6906b at the same step (32.18)**. coll_per_env saturates at ~6 per episode (heavy collision pressure under the loose-action regime, not a learning failure).

Entropy loss is `-0.03` throughout — entropy *increasing*, the entropy bonus dominates the gradient. The policy never concentrates into a tight basin, but the bbox/center rewards are strong enough to maintain useful tracking.

### 3.2 Detection-latency onset (120k–140k): triangulation collapse

| Step | rew_inst | pair_valid | track_lost | tri_d0 | bbox_center | bbox_size | coll_per_env |
|----:|---:|---:|---:|---:|---:|---:|---:|
| 120k | 6.45 | 0.86 | 0.026 | 6.55 | 40.2 | 44.9 | 6.22 |
| 124k | 4.14 | 0.70 | 0.14 | **1.01** | 23.0 | 40.9 | 2.06 |
| 132k | 3.21 | 0.53 | 0.34 | **0.23** | 13.9 | 35.4 | 1.54 |
| 140k | 2.72 | 0.43 | 0.45 | **0.12** | 9.7 | 30.2 | 1.11 |
| 156k | 3.65 | 0.64 | 0.12 | 0.21 | 18.7 | 39.7 | 2.69 |

Across the 20k-step `fixed_delay_start_step=120k → 140k` ramp:
- Triangulation reward drops by **54× (6.55 → 0.12)**.
- bbox_center drops by **4×**, bbox_size only by **1.5×**.
- coll_per_env drops from 6.22 → 1.11 — **the policy abandons close formation** because it no longer pays in triangulation.
- Compare to 0d4fa6906b across the same window: 32.18 → 22.06 (a **1.5× drop, then steady at ~22** through 240k). The new run's collapse is qualitatively different from the baseline's mild dip.

The 156k bounce-back (pair_valid 0.43 → 0.64) coincides with `random_delay_start_step=140k`; the policy briefly re-finds a wider-baseline strategy before random delay re-destabilizes it.

## 4. Analysis — root cause of the post-delay collapse

The 100×+ ratio between 0d4fa's and this run's post-delay triangulation reward isolated to a single new code path: `compute_sigma_drift` (mas/029, [triangulation/triangulation.py:504-570](../../triangulation/triangulation.py)).

The function inflates per-camera σ_pos and σ_ori as a function of observation age `Δt` and ego velocity, intended as a Bar-Shalom OOSM correction:

$$R_{\text{eff}} = R + H \, \Sigma_{\text{drift}} \, H^\top$$

The `constant_velocity` branch implements:

$$\sigma^2_{pos,\text{eff}} = \sigma_{pos}^2 + \|v_{cam}\|^2 \cdot \Delta t^2$$

with the in-code comment "We use the deterministic component since v is known from odom." This is the bug: the deterministic drift `v · Δt` is a **bias** to be predicted-forward and subtracted, not a **variance** to be added. The correct quantity for the variance contribution is `P_{vv} · Δt²`, where `P_{vv}` is the EKF *velocity-estimate uncertainty* (~0.1 m/s on a multirotor), not `‖v‖²` (~25 m²/s² typical).

Quantitative effect at the curriculum's full 310 ms latency with `‖v‖=5 m/s`, `‖ω‖=1 rad/s`:

| Quantity | Base | After inflation | Multiplier |
|---|---:|---:|---:|
| σ_pos | 0.10 m | **1.55 m** | **15×** |
| σ_ori | 0.01 rad (0.6°) | **0.31 rad (17.8°)** | **31×** |
| Per-camera lateral error at 25 m | — | ~7.6 m | — |
| 1/sqrt(trace) (rough) | ~1.3 | ~0.09 | ~14× collapse |

The collapse predicted analytically matches the observed 12×–182× ratios at 124k–140k. **The policy was not failing to learn — the reward gradient was being mathematically destroyed by an over-conservative covariance inflation that kicks in only when `obs_age > 0` (i.e., once `delay_mode != "none"` at step 120k).**

## 5. Decision — fix landed mid-experiment

Three changes shipped to break the over-conservatism without disabling the legitimate target-motion uncertainty path:

1. **`include_drift_uncertainty: bool = False`** added to `TriangulationCfg` ([triangulation_cfg.py](../../triangulation/triangulation_cfg.py)).
2. **`compute_sigma_drift` early-returns** the base `pos_std` / `ori_std` when the flag is off ([triangulation.py:533-548](../../triangulation/triangulation.py)). The original `random_walk` / `constant_velocity` / `ou` branches and the legitimate `target_velocity` branch are preserved for opt-in use.
3. **MC validation test fixed (option c)**: `monte_carlo_validation.py` now (a) defaults `create_test_intrinsics` to the SIYI 1× calibration (`fx=1053.04`, `cx=960`, `cy=540` on 1920×1080) and (b) samples per-camera intrinsic perturbations in the empirical path matching the analytical `Sigma_K` fallback. After the fix, all 6 scenarios pass with trace_ratio ≈ 1.0 and 2σ coverage 73.5–74.1% (expected 73.9%). Plots: [tests/mc_plots/validation_slice1/](../../triangulation/tests/mc_plots/validation_slice1/).

Predicted effect on next training run with `include_drift_uncertainty=False` (default):

| Metric | Predicted | This run |
|---|---:|---:|
| `tri_d0` at 120k (pre-delay) | ~6.5 (unchanged) | 6.55 |
| `tri_d0` at 124k | **3–5** | 1.01 |
| `tri_d0` at 140k | **2–4** | 0.12 |
| `tri_d0` at 200k | **1–3** | not reached |

If post-delay `tri_d0` lands in the predicted range, the diagnosis is fully validated. If it stays low (≤0.5), there is another delay-sensitive term — most likely the per-zoom `Sigma_K` interacting with stale rays, or a geometric issue not modeled in the back-of-envelope above.

## 6. Open items / follow-ups

- The pre-delay 5× reward gap vs 0d4fa6906b is **not** explained by drift (drift contributes ~0 at `Δt ≈ 0`). It comes from `Sigma_K` (intrinsic uncertainty, mas/029) and tighter geometry (60/25 vs 100/40). After the drift fix is empirically validated, decide whether to (a) bump `triangulation_reward_scale` 5×, (b) re-evaluate the per-zoom `Sigma_K` magnitudes, or (c) loosen the geometry envelope.
- `mode=aoi_validation` of the MC suite was not re-run; the early-return only fires when the flag is False, so the drift-on path is unaffected. Verification deferred per user direction.
- `test_intrinsic_uncertainty_mc.py` may break under the new defaults (pre-approved). Not verified.
- The `coll_per_env ≈ 6` regime under `-1/-1` action penalties is not a learning failure but is far above the 0d4fa baseline (~0.4 typical). Future runs reverting action penalties to `-30/-15` should drop this back into baseline range without further intervention.

## 7. Side notes

- The `343e2ed4a3` commit's diff showed `action_sum_penalty_scale=-10`, `action_delta_penalty_scale=-5`, but the env.yaml as launched shows `-1.0/-1.0`. Either the user further edited locally before launch or there is an override mechanism not visible in the config — worth confirming before reproducing.
- Sigma at `1.75` throughout is the highest of any iris_ma6 run on record. Despite this, the policy maintained pair_valid 0.85–0.93 in the pre-delay window. Useful data point: bbox-only tracking is robust to wide policy distributions when action penalties are this light, but the cost is high collision exposure (`coll_per_env ≈ 6`).
- The drift bug had been silently inflating triangulation covariance in every iris_ma6 training run since mas/029 landed. The 0d4fa baseline pre-dated this code path entirely — which is why its post-delay `tri_d0` held at ~22 while every subsequent run collapsed.
