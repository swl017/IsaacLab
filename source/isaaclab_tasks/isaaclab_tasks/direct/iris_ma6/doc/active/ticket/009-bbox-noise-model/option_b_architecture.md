# Option B: Offline YOLO Calibration → Detector Replicator in Training

## Problem

Tickets 009 (bbox noise model) and 010 (FP/FN model) require hand-modeling the
detector's noise profile: size-dependent localization error, miss rate, FP rate,
confidence. This is labor-intensive and fragile — the model must be re-calibrated
whenever the detector changes.

YOLO-in-the-loop for all envs is infeasible (~160x throughput loss).
Running YOLO on a subset of envs during training was the original Option B, but
**TiledCamera cannot be created for a subset of envs** — it validates
`view.count == num_envs` at init and allocates memory for all envs upfront.

## Revised Approach: Two-Phase Pipeline

Instead of running YOLO during training, split into two phases:

```
Phase 1: Offline calibration (~10 min, one-time)
  64 envs + cameras ON → YOLO vs raycaster → fit NoiseModelParams → save JSON

Phase 2: Full training (unchanged, zero overhead)
  1024 envs + cameras OFF → raycaster + detector_replicator(JSON params) → policy
```

This is simpler, faster, and decouples calibration from training entirely.

### Why Offline Beats Online

| | Online (original Option B) | Offline (revised) |
|---|---|---|
| TiledCamera subset | **Not supported** — Isaac Sim requires cameras for all envs | N/A — separate small run |
| Training overhead | 1.5-1.8x slowdown | **Zero** |
| Complexity | YOLO in training loop, async threading | Standalone script |
| Re-calibration | Automatic but coupled | Re-run 10-min script when detector changes |
| Engineering effort | ~400 lines in env + delay system | ~350 lines standalone script (done) |

## Architecture

### Phase 1: Calibration Run

```
calibrate_bbox_noise.py (standalone script)
│
├── IrisMA6TestEnv            (64 envs, enable_tiled_cameras=True)
│   ├── BBoxRayCasterV2       → GT pixel bboxes (64, 2, 1, 4)
│   └── TiledCamera × 2      → RGB images (64, H, W, 3) per agent
│
├── YoloBatchInference        → YOLO detections per image
│   └── ultralytics YOLO(yolov11m-drone.pt or dronecop9-2.pt)
│
├── DetectorCalibrator        → accumulates YOLO-vs-raycaster stats
│   ├── .ingest()             → IoU matching, error recording
│   └── .fit()                → binned regression → NoiseModelParams
│
└── Output: bbox_noise_params.json + report PNGs
```

**Script**: `experiments/calibrate_bbox_noise.py`

**Usage**:
```bash
# With trained policy (realistic bbox size distribution)
python calibrate_bbox_noise.py \
    --experiment a1_with_aoi \
    --checkpoint /path/to/best_agent.pt \
    --num_envs 64 --num_steps 250 \
    --output bbox_noise_params_yolov11m.json

# Quick sanity check with random policy
python calibrate_bbox_noise.py \
    --num_envs 32 --num_steps 2000 \
    --output bbox_noise_params.json
```

### Phase 2: Apply Fitted Parameters via Detector Replicator

Load the JSON and apply via `detector_replicator` inside `bbox_raycaster_v2`.
The replicator is the single architectural home for all detector modeling:
- **Ticket 009**: calibrated Student's t localization noise
- **Ticket 010**: miss rate (FN) and false positive (FP) injection

#### Dual output from bbox_raycaster_v2

`bbox_raycaster_v2` outputs both pure GT bboxes and replicated bboxes:
- **GT bboxes** → reward path (both privileged and perception-aligned). Rewards must never be corrupted by detector stochasticity.
- **Replicated bboxes** → observation path only. The policy learns to cope with noisy/unreliable detections through its observations.

#### Localization noise (detector_replicator, ticket 009)

```python
# Inside bbox_raycaster_v2/detector_replicator
def apply_noise(self, bboxes, bbox_empty, target_size_px, noise_scale):
    """Apply calibrated Student's t noise to detected bboxes.
    
    Only applied where bbox_empty=False (valid detections).
    """
    p = self._calibrated_params
    scale = p.center_noise_a / target_size_px.clamp(min=5.0) + p.center_noise_b
    scale *= noise_scale  # curriculum gating
    noise = StudentT(df=p.center_noise_df, loc=p.center_bias, scale=scale).sample()
    
    # Only add noise to valid detections
    noisy_bboxes = bboxes.clone()
    noisy_bboxes[~bbox_empty] += noise[~bbox_empty]
    return noisy_bboxes
```

#### Miss injection (detector_replicator, ticket 010)

```python
# Inside bbox_raycaster_v2/detector_replicator
def apply_miss(self, bboxes, bbox_empty, target_size_px, bg_is_sky):
    """Probabilistic miss based on target size and background type.
    
    Only applied where bbox_empty=False (skip already-empty).
    Uses dual sigmoids conditioned on background (sky vs ground).
    """
    p = self._calibrated_params
    if bg_is_sky:
        p_miss = sigmoid(p.miss_a_sky * (p.miss_threshold_sky - target_size_px))
    else:
        p_miss = sigmoid(p.miss_a_gnd * (p.miss_threshold_gnd - target_size_px))
    
    missed = torch.bernoulli(p_miss).bool() & ~bbox_empty
    bboxes_out = bboxes.clone()
    bboxes_out[missed] = 0.0
    bbox_empty_out = bbox_empty | missed
    return bboxes_out, bbox_empty_out
```

#### Data Flow (per policy step, calibrated training)

```
bbox_raycaster_v2.update(...)
    → data.bboxes            (1024, 2, 1, 4) pixel     [GT, clean]
    → data.bbox_empty        (1024, 2, 1)               [GT occlusion]

bbox_raycaster_v2.apply_detector_replicator(params, noise_scale)
    # 1. Apply miss rate (ticket 010) — only where bbox_empty=False
    #    bg_is_sky from raycast_mesh query through target
    #    Missed → zero bbox, set bbox_empty_replicated=True
    # 2. Apply noise (ticket 009) — only where bbox_empty_replicated=False
    #    Student's t(df~12-16, scale=a/size+b) + bias
    # 3. Zero out bboxes where bbox_empty_replicated=True (bug fix)
    → data.bboxes_replicated       (1024, 2, 1, 4)     [noisy + FN]
    → data.bbox_empty_replicated   (1024, 2, 1)         [includes FN misses]

For each agent:
    # Reward path: GT bboxes (clean)
    gt_data.bboxes_2d = raycaster.data.bboxes[:, idx, :, :]

    # Obs path: replicated bboxes (noisy, with FN)
    obs_data.bboxes_2d = raycaster.data.bboxes_replicated[:, idx, :, :]

    delay_system.update_ground_truth(agent_id, gt_states, obs_states)

_get_rewards():
    delay_system.get_states_for_rewards()  → GT bboxes (clean, delayed)

_get_observations():
    delay_system.get_states_for_observations()  → replicated bboxes (noisy, delayed)
```

---

## Calibration Results (2026-04-07)

### Run Configuration

| Parameter | Value |
|-----------|-------|
| YOLO model | yolov11m-drone.pt (39 MB, single-class "drone") |
| YOLO confidence threshold | 0.25 |
| Num envs | 64 |
| Num agents | 2 |
| Num steps | 250 |
| Policy | a1_with_aoi (trained checkpoint) |
| Camera resolution | 640 × 480 |
| Fit size range | [20, 80] px |

### Fitted Parameters

```json
{
  "noise_distribution": "student_t",
  "center_noise_a": "~0 (size-independent in [20,80]px range — needs refit from binned data)",
  "center_noise_b": "13.4 px (authoritative, from findings.md)",
  "center_noise_df": "12-16 (authoritative, from report-5 with more matches)",
  "size_noise_a": "~0",
  "size_noise_b": "18.0 px",
  "size_noise_df": "12.2",
  "miss_sigmoid_a": "0.031 (ticket-010 scope)",
  "miss_size_threshold_px": "65.6 (ticket-010 scope)",
  "fp_rate": "0.49/frame (inflated by other-agent detections — needs re-measurement)",
  "center_bias_x": "-1.2 px",
  "center_bias_y": "+0.1 px"
}
```

**Parameter sources**: Two calibration runs produced different values. The authoritative
sources are:
- **df~12-16**: from `bbox_noise_params_yolov11m_report-5/` (more matches, more reliable)
- **noise scale ~13.4px**: from `findings.md` (supersedes earlier 10.5px estimate)
- **bias (-1.2, +0.1)px**: from findings.md

### Key Findings

**Detection statistics** (20,897 visible frames analyzed):
- **5,043 matched** (24%) — YOLO found the target with IoU ≥ 0.1
- **15,854 missed** (76%) — target visible to raycaster but YOLO missed it
- **4,402 false positives** — spurious detections (FP rate = 0.21/frame, pre-agent-filtering)

**Distribution model**: Student's t fits significantly better than both Gaussian
and Laplace for localization error. The t-distribution captures the rounded peak
(like Gaussian) with heavier tails (like Laplace) via the df parameter:

| Distribution | Peak shape | Tail weight | Fit quality |
|-------------|------------|-------------|-------------|
| Gaussian | Rounded | Light | Underestimates tails |
| Laplace | Too sharp (spike) | Heavy | Overestimates peak |
| **Student's t (df≈12-16)** | **Rounded** | **Moderate-heavy** | **Best fit** |

**Localization noise** (center error):
- Scale ≈ 13.4 px base, size-dependent: ~20px at 10px targets, ~5px at 100px targets
- Binned data shows clear size-dependent trend despite regression returning a≈0
- Bias: (-1.2, +0.1) px — small systematic offset
- df ≈ 12-16 — moderately heavy tails (heavier than Gaussian, lighter than Laplace)

**Size noise** (bbox width/height error):
- Scale ≈ 18.0 px
- df ≈ 12 — slightly heavier tails than center error

**Miss rate** (see findings.md §3 for background-dependent curves):
- Overall: 68.2% (gimbal-locked calibration)
- Sky background: 64.3%, Ground background: 81.8% (+17.5% penalty)
- Critical zone: 10-60px target size
- Sigmoid parameters to be fitted per-background (ticket-010 scope)

**False positive rate**: 0.49/frame (inflated by other-agent detections).
True clutter FP rate needs re-measurement with agent filtering applied.

### Comparison with Previous Assumptions

| Parameter | Old (hand-tuned) | Calibrated | Change |
|-----------|-----------------|------------|--------|
| `bbox_std` | 7.0 px (Gaussian) | 13.4 px (Student's t, df=12-16) | +91%, heavier tails |
| Miss rate | Not modeled | dual sigmoid (sky/ground) | **New** (ticket-010) |
| FP rate | Not modeled | ~0.49/frame (needs filtering) | **New** (ticket-010) |
| Bias | Not modeled | (-1.2, +0.1) px | **New** (small) |

### Visual Report

Three report figures generated in `bbox_noise_params_yolov11m_report/`:

1. **calibration_report.png** — 2×3 grid:
   - Top: Center X/Y error distributions (Student's t vs Gaussian vs Laplace PDFs)
   - Bottom: Noise vs target size [20,80]px, miss rate vs target size, confidence distribution

2. **calibration_report_size_dist.png** — 1×3:
   - Width/height error distributions + error vs camera-to-target distance scatter

3. **calibration_validation.png** — 1×3:
   - Overlay histograms: synthetic samples from fitted Student's t model vs actual YOLO errors
   - Good match for center X and Euclidean error; slight mismatch in center Y tails

---

## Fitted Noise Model

### NoiseModelParams

```python
@dataclass
class NoiseModelParams:
    noise_distribution: str = "student_t"

    # Localization: scale = a / target_size_px + b
    # Sampled as StudentT(df, loc=bias, scale=scale)
    center_noise_a: float     # size-dependent term
    center_noise_b: float     # constant floor (px)
    center_noise_df: float    # Student's t degrees of freedom

    size_noise_a: float
    size_noise_b: float
    size_noise_df: float

    # Miss rate: p_miss = sigmoid(a * (threshold - target_size_px))
    # Ticket-010 extends with dual sigmoids (sky/ground)
    miss_sigmoid_a: float
    miss_size_threshold_px: float

    # False positive rate (per frame per camera)
    fp_rate: float

    # Systematic bias (pixels)
    center_bias_x: float
    center_bias_y: float

    # Fitting regime
    fit_size_range: tuple     # (min_px, max_px) used for fitting
```

### Calibration Methodology

1. **IoU matching**: For each frame where raycaster reports a visible target
   (confidence > 0.5, size > 1px), find the best-IoU YOLO detection.
   Match if IoU ≥ 0.1. Unmatched GT = miss. Unmatched YOLO = false positive.

2. **Distribution fitting**: Fit Student's t via `scipy.stats.t.fit()` on the
   center error (zero-mean) to get df. This gives a rounded peak (no Laplace
   spike) with controllable tail heaviness.

3. **Localization scale**: Bin matched pairs by target size (log-spaced, [20,80]px).
   Per-bin 1D IQR/1.35 scale (average of X and Y axes). Fit `scale = a/size + b`
   via weighted `scipy.optimize.curve_fit`.

4. **Miss rate**: Bin by target size. Per-bin miss rate.
   Fit sigmoid `p = σ(a*(threshold - size))` via weighted `scipy.optimize.curve_fit`.

5. **FP rate**: Total false positives / total frames.

6. **Bias**: Median center error across all matches.

## What This Replaces

| Ticket | Manual work | Detector replicator replacement |
|--------|-------------|---------------------|
| 009 (bbox noise model) | Collect data in PegasusSim, fit offline, update NoiseCfg | 10-min calibration script → JSON → replicator |
| 010 §1 (miss rate) | Hand-model sigmoid | Auto-fitted from YOLO miss data (dual sky/ground sigmoids) |
| 010 §2 (FP model) | Hand-tune FP rate | Measured directly from YOLO |
| 010 §3 (detection confidence in obs) | **Deferred** — requires principled model from calibration data |
| 010 §4 (curriculum gating) | **Not replaced** — still needed, but params come from JSON |

**Tickets 009 and 010 scopes 1-2 are fully replaced by the calibration script.**

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| `YoloBatchInference` | **Done** | `experiments/calibrate_bbox_noise.py` |
| `DetectorCalibrator` | **Done** | `experiments/calibrate_bbox_noise.py` |
| `NoiseModelParams` | **Done** | `experiments/calibrate_bbox_noise.py` |
| Calibration script | **Done** — validated | `experiments/calibrate_bbox_noise.py` |
| Visual report generation | **Done** | 3 PNG figures per run |
| Calibration run | **Done** | `experiments/bbox_noise_params_yolov11m_report/` |
| `detector_replicator` in `bbox_raycaster_v2` | Not started | `bbox_raycaster_v2/detector_replicator.py` |
| Dual bbox output from raycaster | Not started | `bbox_raycaster_v2/bbox_raycaster_v2_data.py` |
| `CalibratedBBoxNoiseCfg` in env cfg | Not started | `iris_ma_env6_test_cfg.py` |
| Dual bbox wiring in env (GT→rewards, replicated→obs) | Not started | `iris_ma_env6_test.py` |
| Delay system `bbox_std=0` when replicator enabled | Not started | `iris_ma_env6_test.py` init |

## Implementation Order (Remaining)

1. **Create `detector_replicator`** in `bbox_raycaster_v2` — Student's t noise, cfg, toggle (~80 lines)
2. **Add dual bbox output** to `BBoxRayCasterV2Data` — `bboxes_replicated`, `bbox_empty_replicated` (~20 lines)
3. **Add `CalibratedBBoxNoiseCfg`** — config + JSON loading (~30 lines)
4. **Wire dual bboxes in env** — GT to reward path, replicated to obs path; set delay `bbox_std=0` (~30 lines)
5. **Validation** — compare policy performance with/without calibrated noise

**Total new code in training pipeline**: ~160 lines (replicator + data + cfg + env wiring).

## Integration into env_cfg

```python
# In iris_ma_env6_test_cfg.py
@configclass
class CalibratedBBoxNoiseCfg:
    enabled: bool = False
    """Load calibrated noise params from JSON and apply via detector_replicator."""

    params_path: str = ""
    """Path to bbox_noise_params.json from calibration run."""

    apply_bias: bool = True
    """Apply systematic center bias from calibration."""
```

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| ultralytics not in Isaac Sim conda env | `pip install ultralytics` — no conflicts with env_isaaclab |
| YOLO detects wrong objects (not target drone) | Single-class model, filter by class_id=0 |
| 68% miss rate seems high | Expected — sim camera distances produce small targets; real deployment may differ |
| Stale calibration after detector update | Re-run 10-min script — automated, no manual fitting |
| Rendering 64×2 cameras is slow | Only during calibration, not training |
| Dual bbox storage in delay system | May need dual fields in AgentStates or separate store calls — design in Stage S |

## Open Questions

1. **Should noise be size-dependent in training?** Current calibration shows ~constant
   scale across [20,80]px (a≈0). However, binned data from findings.md clearly shows
   ~20px noise at 10px targets, ~5px at 100px targets. Re-fit `a, b` from per-bin
   scale values with a wider size range.

2. **When to re-calibrate?** After YOLO model update, after camera resolution change,
   or after significant env changes (new target model, new zoom range).

3. **Dual bbox storage mechanism**: How does the delay system store both GT and replicated
   bboxes? Options: (a) dual fields in AgentStates, (b) separate store calls,
   (c) single store with a flag for which path to use. Resolve in Stage S.
