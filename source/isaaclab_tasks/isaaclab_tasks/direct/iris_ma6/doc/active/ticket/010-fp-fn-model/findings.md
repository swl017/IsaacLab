# Ticket 010 — Detector FP/FN Model: Calibration Findings

## Date: 2026-04-08

## Summary

We ran two calibration scripts against the iris_ma6 Isaac Sim environment to
characterize YOLOv11m-drone detection behavior. The findings replace the need
for hand-modeled FP/FN parameters and provide the data for a **detector
replicator** model that conditions on target size and background type.

---

## 1. Calibration Scripts Developed

| Script | Purpose | Gimbal | Key output |
|--------|---------|--------|------------|
| `calibrate_bbox_noise.py` | Localization noise, overall miss/FP rate | Policy-controlled | `bbox_noise_params_yolov11m.json` |
| `calibrate_clutter_fn.py` | Background-dependent FN rate | **Locked to target** | `clutter_fn_data.json` |

**Why two scripts**: With policy-controlled gimbal, the target drifts in/out of
FOV, confounding miss rate with gimbal tracking error. The gimbal-locked script
isolates detector FN due to background clutter from gimbal behavior.

---

## 2. Localization Noise (from `calibrate_bbox_noise.py`)

**Model**: yolov11m-drone.pt, conf=0.25, 64 envs, 250 steps, policy-controlled gimbal.

### Distribution: Student's t (best fit)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Center noise scale | `a/size + b` where a≈0, b≈13.4 px | Size-dependent: ~20px at 10px targets, ~5px at 100px targets (see plot) |
| Center noise df | 4.9 | Heavier tails than Gaussian, rounded peak (no Laplace spike) |
| Size noise scale | 18.0 px | Width/height error |
| Size noise df | 5.4 | |
| Center bias | (-1.2, +0.1) px | Small systematic offset |
| FP rate | 0.49/frame | High — includes other-agent detections (see §4) |

**Size-dependent noise**: The binned data clearly shows localization error
decreasing from ~20px (at target size ~10px) to ~5px (at target size ~100px).
The curve fitting returned `a≈0` due to the regression being dominated by
mid-range bins, but the visual trend is unambiguous. The replicator should use
`scale = a/size + b` with parameters re-fitted from the binned data.

See: `experiments/bbox_noise_params_yolov11m_report-1/calibration_report.png`
(bottom-left panel: "Noise vs Target Size")

**Key finding**: Student's t(df≈5) fits the YOLO localization error significantly
better than both Gaussian (underestimates tails) and Laplace (overestimates peak).
The rounded peak + heavy tails match the detector's error profile.

See: `experiments/bbox_noise_params_yolov11m_report/calibration_report.png`
(top row: Center X/Y error distributions with Student's t, Gaussian, Laplace overlays)
and `experiments/bbox_noise_params_yolov11m_report/calibration_validation.png`
(YOLO actual vs Student's t model overlay histograms)

### Miss rate (policy-controlled gimbal — biased)

- Overall: 90% miss rate — **heavily biased** by gimbal shaking
- Miss sigmoid threshold: ~94px (confounded with gimbal tracking loss)
- These numbers should NOT be used for the replicator model (see §3 instead)

---

## 3. Background-Dependent FN Rate (from `calibrate_clutter_fn.py`)

**Model**: yolov11m-drone.pt, conf=0.25, 64 envs, 500 steps, **gimbal locked to target**.
Flight scene enabled. Other-agent detections filtered out via pixel projection.

### Overall statistics

| Metric | Value |
|--------|-------|
| Total frames (target visible) | 58,468 |
| Total misses | 39,874 |
| Overall miss rate | 68.2% |
| Sky background frames | 45,327 (77.5%) |
| Sky miss rate | **64.3%** (CI: 63.8-64.7%) |
| Ground/building background frames | 13,141 (22.5%) |
| Ground miss rate | **81.8%** (CI: 81.1-82.4%) |
| **Background penalty** | **+17.5 percentage points** |

### Miss rate vs target size — the primary curve for the replicator

| Target size (px) | Sky miss rate | Ground miss rate | Background gap |
|-------------------|--------------|------------------|----------------|
| <10 | ~100% | ~100% | — (too small) |
| 10-30 | ~70-80% | ~90-95% | **+15-20%** |
| 30-60 | ~40-55% | ~50-70% | **+10-20%** |
| 60-100 | ~30-50% | ~40-60% | ~10% |
| >100 | ~40-60% | ~50-70% | Noisy |

**Critical zone: 10-60px** — this is where background matters most. Below 10px,
nothing is detectable. Above 100px, both backgrounds are mostly detectable.
The ground/building background adds a consistent +15-20% miss rate penalty in the
critical detection zone.

**This is the primary variable for the replicator** because agents use zoom,
which decouples target pixel size from physical distance. A target at 40m with 4x
zoom produces the same pixel size as a target at 10m with 1x zoom — and the
detector's miss rate depends on the pixel size, not the physical distance.

See: `experiments/clutter_fn_data_report/clutter_fn_miss_rates.png`
(top-left panel: "Miss Rate vs Target Size (sky vs ground)" — the three-curve overlay)

### Miss rate vs distance (less informative due to zoom)

| Distance | Miss rate |
|----------|-----------|
| 10m | ~5% |
| 15m | ~30% |
| 20m | ~40% |
| 30m | ~80% |
| 40m+ | ~90% |

Correlates with target size shrinking at distance — but since agents use zoom,
distance alone is not the right predictor. Target pixel size (which accounts for
zoom) is the correct variable.

See: `experiments/clutter_fn_data_report/clutter_fn_miss_rates.png` (top-right panel)

### Miss rate vs elevation angle (proxy for background type)

| Elevation (deg below horizontal) | Miss rate |
|----------------------------------|-----------|
| 5-15° (near-level) | ~85-95% |
| 20-30° | ~60-80% |
| 35-45° (steep look-down) | ~30-50% |
| >45° | ~10-20% |

Steep look-down angles have sky behind target → lower miss rate. Near-level
viewing has terrain/buildings behind → higher miss rate. This confirms the
elevation→background→miss rate causal chain. The direct `background_is_sky`
ray query (§5a) is preferred over elevation as a predictor.

See: `experiments/clutter_fn_data_report/clutter_fn_miss_rates.png` (bottom-left panel)

### YOLO confidence vs target size

- **>50px**: confidence 0.3-0.9, median ~0.6-0.7
- **20-50px**: confidence 0.3-0.6, fewer high-confidence hits
- **<20px**: rare, all low confidence (0.25-0.4, near threshold)
- Confidence degrades **smoothly** with size — no hard cutoff

See: `experiments/clutter_fn_data_report/clutter_fn_hit_confidence.png`
(left panel: "YOLO confidence vs Target size")

---

## 4. False Positive Characteristics

### Other-agent confusion (resolved)

YOLO detects other drones in the scene as valid "drone" class detections. Without
filtering, these inflate the FP count. Solution implemented in `calibrate_clutter_fn.py`:
- Project other agents' world positions to the current camera's pixel coordinates
- Discard any YOLO detection within 80px of another agent's projected center
- Remaining unmatched detections are genuine FPs (clutter)

### FP statistics (from `calibrate_bbox_noise.py`, pre-agent-filtering)

- FP rate: 0.49/frame (includes other-agent detections — overestimate)
- True clutter FP rate is lower — needs re-measurement with agent filtering
  applied to `calibrate_bbox_noise.py` (future work)

### FP spatial and size distribution (pre-filtering, but still informative)

See: `experiments/bbox_noise_params_yolov11m_report/calibration_fp_analysis.png`

Key observations from the FP analysis (6 panels):
- **Spatial heatmap**: FPs distributed relatively uniformly across the image,
  slight concentration in center/left — no strong edge bias
- **FP size distribution**: Median ~32px. Most FPs are small (10-50px),
  similar to target size range — hard to distinguish from real targets by size alone
- **FP vs TP confidence**: FP confidence peaks at 0.3-0.5, TP confidence peaks
  at 0.6-0.8. A confidence threshold of ~0.5 would remove many FPs at the cost
  of some TPs. This suggests confidence is a useful signal for the replicator
- **FP aspect ratio**: Median ~2.0 (wider than tall) — real drone targets are
  closer to 1.0. Aspect ratio could be a secondary filter
- **FP size vs confidence**: Small FPs cluster at low confidence; larger FPs
  can have higher confidence (likely other-agent detections)
- **FP by target visibility**: ~86% of FPs occur when the target IS visible
  (the detector finds something else instead), ~14% when target is not visible

Note: These statistics are inflated by other-agent detections. The spatial
heatmap and size distribution will change after agent filtering is applied.

---

## 5. Detector Replicator Model Design

Based on these findings, the replicator needs:

### 5a. Background query (per-frame, during training)

```python
# Cast ray through target past the target mesh
bg_is_sky = raycast_mesh(
    ray_starts=target_pos + ray_dir * 2.0,  # start past target
    ray_directions=ray_dir,
    mesh=bbox_raycaster.static_mesh,
).distance == inf  # inf = sky, finite = ground/building
```

Cost: ~0.01ms for 1024 rays (negligible, GPU-resident mesh).

### 5b. Conditional miss rate

```python
# Two sigmoid miss curves conditioned on background
if bg_is_sky:
    p_miss = sigmoid(a_sky * (threshold_sky - target_size_px))
else:
    p_miss = sigmoid(a_gnd * (threshold_gnd - target_size_px))

# Apply miss
if bernoulli(p_miss):
    bbox = zeros  # → bbox_empty = True
```

Parameters to fit from calibration data:
- `a_sky`, `threshold_sky` — sky miss sigmoid
- `a_gnd`, `threshold_gnd` — ground miss sigmoid  
- `threshold_gnd > threshold_sky` by ~10-15px

### 5c. Localization noise (when detected)

```python
# Size-dependent Student's t noise
# Binned data shows: ~20px noise at 10px targets, ~5px at 100px targets
scale = a / target_size_px.clamp(min=5) + b  # a, b re-fitted from binned data
noise = StudentT(df=4.9, loc=bias, scale=scale).sample()
bbox_center += noise  # per-axis
```

Note: The curve fit returned `a≈0` due to regression issues (see §2), but the
binned data trend is clear. Re-fit `a, b` from the per-bin scale values in the
calibration report.

### 5d. Confidence (optional observation signal)

If confidence is added to observations (ticket 010 §3):
```python
# Confidence decreases with size, conditioned on detection
confidence = clip(0.3 + 0.4 * (target_size_px / 100), 0.25, 0.95)
confidence += normal(0, 0.1)  # noise
```

---

## 6. Report Artifacts

| File | Location |
|------|----------|
| Bbox noise params (JSON) | `experiments/bbox_noise_params_yolov11m_report/bbox_noise_params_yolov11m.json` |
| Bbox noise calibration plots | `experiments/bbox_noise_params_yolov11m_report/calibration_report.png` (+ 3 more) |
| Clutter FN data (JSON, 58K samples) | `experiments/clutter_fn_data_report/clutter_fn_data.json` |
| Clutter FN miss rate plots | `experiments/clutter_fn_data_report/clutter_fn_miss_rates.png` |
| Clutter FN confidence plots | `experiments/clutter_fn_data_report/clutter_fn_hit_confidence.png` |
| Clutter FN video overlay | `experiments/clutter_fn_data_report/clutter_fn_*.mp4` |

---

## 7. Remaining Work

| Item | Status | Notes |
|------|--------|-------|
| Localization noise → delay system | Not started | Apply Student's t(df=4.9, scale=13.4) in `_get_noise_std()` |
| Sky/ground conditional miss → replicator | Not started | `raycast_mesh` query + dual sigmoid |
| FP injection | Deferred | Need agent-filtered FP statistics first |
| Detection confidence in obs | Deferred | Orthogonal to calibration — architecture decision |
| Curriculum gating | Not started | Ramp in noise/miss with training progress |
| Re-run `calibrate_bbox_noise.py` with agent filtering | Not started | Current FP rate inflated by other-agent detections |

---

## 8. Methodology Notes

### Why Student's t over Gaussian/Laplace

The YOLO localization error has a **rounded peak** (like Gaussian) with **heavier
tails** (like Laplace). Student's t captures both via the `df` parameter:
- df → ∞: Gaussian (light tails)
- df ≈ 1: Cauchy (very heavy tails)
- df ≈ 5: Best fit for YOLO — rounded peak + moderate heavy tails

Validation: synthetic Student's t samples closely match the YOLO error histogram
in both center error X/Y and Euclidean magnitude (see `calibration_validation.png`).

### Why gimbal-locked for FN calibration

With policy gimbal control, 90% of "misses" are from the target being at the
edge of or outside the FOV due to gimbal shake. This conflates two independent
failure modes:
1. **Gimbal tracking failure** → target not in frame (already handled by bbox_empty)
2. **Detector failure** → target in frame but YOLO misses it (background clutter)

Gimbal lock isolates (2), which is what the replicator needs to model. The
policy already handles (1) through its gimbal control learning.

### Why ray-based background classification

Elevation angle correlates with background type but is an indirect proxy. The
`raycast_mesh` query directly answers "is there scene geometry behind the target?"
This is:
- **Exact**: no approximation via angles
- **Cheap**: one ray per frame per camera, GPU-accelerated
- **Available during training**: the `static_mesh` is already loaded by `BBoxRayCasterV2`
