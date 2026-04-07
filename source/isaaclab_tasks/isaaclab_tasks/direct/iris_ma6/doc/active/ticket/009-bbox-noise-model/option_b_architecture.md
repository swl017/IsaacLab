# Option B: Offline YOLO Calibration → Parameterized Noise in Training

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
  32 envs + cameras ON → YOLO vs raycaster → fit NoiseModelParams → save JSON

Phase 2: Full training (unchanged, zero overhead)
  1024 envs + cameras OFF → raycaster + calibrated noise from JSON → policy
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
├── IrisMA6TestEnv            (32 envs, enable_tiled_cameras=True)
│   ├── BBoxRayCasterV2       → GT pixel bboxes (32, 3, 1, 4)
│   └── TiledCamera × 3      → RGB images (32, H, W, 3) per agent
│
├── YoloBatchInference        → YOLO detections per image
│   └── ultralytics YOLO(yolov11m-drone.pt)
│
├── DetectorCalibrator        → accumulates YOLO-vs-raycaster stats
│   ├── .ingest()             → IoU matching, error recording
│   └── .fit()                → binned regression → NoiseModelParams
│
└── Output: bbox_noise_params.json
```

**Script**: `experiments/calibrate_bbox_noise.py`

**Usage**:
```bash
# With trained policy (realistic bbox size distribution)
./isaaclab.sh -p .../calibrate_bbox_noise.py \
    --experiment a3_full \
    --checkpoint /path/to/best_agent.pt \
    --num_envs 32 --num_steps 5000 \
    --output bbox_noise_params.json

# Quick sanity check with random policy
./isaaclab.sh -p .../calibrate_bbox_noise.py \
    --num_envs 32 --num_steps 2000 \
    --output bbox_noise_params.json
```

**Expected timing**: 32 envs × 3 agents = 96 images/step.
YOLO batch at ~2ms/image → ~200ms/step. 5000 steps ≈ 17 minutes.

### Phase 2: Apply Fitted Parameters in Training

Load the JSON and apply to the delay system. Two integration points:

#### 2a. Size-dependent noise in delay system

Extend `_get_noise_std()` in `delay_system_v3/multi_agent_wrapper.py:360`:

```python
# Current
elif noise_type == "bbox":
    base_std = self._noise_cfg.bbox_std.value

# With calibrated params loaded from JSON
elif noise_type == "bbox":
    if self._calibrated_params is not None and target_size_px is not None:
        p = self._calibrated_params
        base_std = p.center_noise_a / target_size_px.clamp(min=5.0) + p.center_noise_b
    else:
        base_std = self._noise_cfg.bbox_std.value
```

#### 2b. Miss injection (probabilistic bbox dropout)

```python
def _apply_detection_miss(self, bbox: torch.Tensor,
                          target_size_px: torch.Tensor) -> torch.Tensor:
    """Zero out bbox with probability p_miss(target_size)."""
    if self._calibrated_params is None:
        return bbox
    p = self._calibrated_params
    p_miss = torch.sigmoid(p.miss_sigmoid_a * (p.miss_size_threshold - target_size_px))
    mask = torch.bernoulli(1.0 - p_miss).unsqueeze(-1)
    return bbox * mask  # missed detections become zero → bbox_empty
```

Both are applied after raycaster GT is written to `gt_data.bboxes_2d` and before
the delay system processes it. Curriculum-gated via existing `noise_scale`.

### Data Flow (per policy step, calibrated training)

```
bbox_raycaster_v2.update(...)
    → bboxes: (1024, 3, 1, 4) pixel         [GT geometry]

For each agent:
    gt_data.bboxes_2d = raycaster.bboxes[:, idx, :, :]

    # NEW: apply calibrated noise + miss
    target_size = sqrt(bbox_w * bbox_h)
    noise_std = calibrated.center_noise_a / target_size + calibrated.center_noise_b
    gt_data.bboxes_2d += N(0, noise_std)     [size-dependent noise]
    gt_data.bboxes_2d *= bernoulli(1-p_miss)  [probabilistic miss]

    delay_system.update_ground_truth(agent_id, gt_states)
```

## Fitted Noise Model

### NoiseModelParams

```python
@dataclass
class NoiseModelParams:
    # Localization: std = a / target_size_px + b
    center_noise_a: float    # size-dependent term (larger for small targets)
    center_noise_b: float    # constant floor
    size_noise_a: float
    size_noise_b: float

    # Miss rate: p_miss = sigmoid(a * (threshold - target_size_px))
    miss_sigmoid_a: float    # steepness of transition
    miss_size_threshold_px: float  # size below which miss rate rises

    # False positive rate (per frame per camera)
    fp_rate: float

    # Systematic bias (pixels)
    center_bias_x: float
    center_bias_y: float

    # Metadata
    num_frames: int
    num_matches: int
    num_misses: int
    num_false_positives: int
```

### What gets fitted (replacing tickets 009 + 010)

| Parameter | Manual (ticket 009/010) | Auto-calibrated |
|-----------|------------------------|-----------------|
| bbox_std | Fixed 7.0 px | `a/size + b` fitted from YOLO data |
| Miss rate | Hand-modeled sigmoid | Sigmoid fitted from YOLO miss data |
| FP rate | Hand-tuned constant | Measured directly |
| Center bias | Not modeled | Measured directly |
| Size noise | Fixed scalar | `a/size + b` fitted from YOLO data |

### Calibration methodology

1. **IoU matching**: For each frame where raycaster reports a visible target
   (confidence > 0.5, size > 1px), find the best-IoU YOLO detection.
   Match if IoU ≥ 0.1. Unmatched GT = miss. Unmatched YOLO = false positive.

2. **Localization noise**: Bin matched pairs by target size (log-spaced).
   Per-bin, compute std of center error. Fit `std = a/size + b` via least squares.

3. **Miss rate**: Bin by target size. Per-bin miss rate.
   Fit sigmoid `p = σ(a*(threshold - size))` to find threshold and steepness.

4. **FP rate**: Total false positives / total frames.

5. **Bias**: Mean center error across all matches.

## Integration into env_cfg

```python
# In iris_ma_env6_test_cfg.py
@configclass
class CalibratedBBoxNoiseCfg:
    enabled: bool = False
    """Load calibrated noise params from JSON."""

    params_path: str = ""
    """Path to bbox_noise_params.json from calibration run."""

    apply_miss: bool = True
    """Apply probabilistic miss based on calibrated miss rate."""

    apply_fp: bool = False
    """Apply false positive injection (deferred — requires obs changes)."""
```

## What This Replaces

| Ticket | Manual work | Option B replacement |
|--------|-------------|---------------------|
| 009 (bbox noise model) | Collect data in PegasusSim, fit offline, update NoiseCfg | 10-min calibration script → JSON |
| 010 §1 (miss rate) | Hand-model sigmoid | Auto-fitted from YOLO miss data |
| 010 §2 (FP model) | Hand-tune FP rate | Measured directly from YOLO |
| 010 §3 (detection confidence in obs) | **Not replaced** — orthogonal observation design choice |
| 010 §4 (curriculum gating) | **Not replaced** — still needed, but params come from JSON |

**Tickets 009 and 010 scopes 1-2 are fully replaced by the calibration script.**

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| `YoloBatchInference` | **Done** (in calibration script) | `experiments/calibrate_bbox_noise.py` |
| `DetectorCalibrator` | **Done** (in calibration script) | `experiments/calibrate_bbox_noise.py` |
| `NoiseModelParams` | **Done** (in calibration script) | `experiments/calibrate_bbox_noise.py` |
| Calibration script | **Done** — ready to test | `experiments/calibrate_bbox_noise.py` |
| Size-dependent noise in delay system | Not started | `delay_system_v3/multi_agent_wrapper.py` |
| Miss injection in delay system | Not started | `delay_system_v3/multi_agent_wrapper.py` |
| `CalibratedBBoxNoiseCfg` in env cfg | Not started | `iris_ma_env6_test_cfg.py` |
| JSON loading in env init | Not started | `iris_ma_env6_test.py` |

## Implementation Order

1. **Run calibration script** — get actual numbers, validate YOLO works in Isaac Sim env
2. **Inspect fitted params** — sanity check: are the numbers reasonable?
3. **Extend delay system** — size-dependent noise + miss injection (~50 lines)
4. **Add CalibratedBBoxNoiseCfg** — config + JSON loading (~30 lines)
5. **Wire into env** — apply calibrated noise before delay system update (~20 lines)
6. **Validation** — compare policy performance with/without calibrated noise

**Total new code in training pipeline**: ~100 lines (delay system + env + cfg).
The calibration script is standalone and already complete.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| ultralytics not in Isaac Sim conda env | `pip install ultralytics` before first run |
| YOLO detects wrong objects (not target drone) | Single-class model, filter by class_id=0 |
| Calibration data from 32 envs not representative | Use trained policy for diverse bbox sizes; 5000 steps × 96 cameras = 480K frames |
| Stale calibration after detector update | Re-run 10-min script — automated, no manual fitting |
| Rendering 32×3 cameras is slow | Only during calibration, not training. ~17 min one-time cost |

## Open Questions

1. **When to re-calibrate?** After YOLO model update, after camera resolution change,
   or after significant env changes (new target model, new zoom range). Could automate
   as a CI step.

2. **Should FP injection be in Phase 1?** FP requires generating spurious bboxes
   at random image locations, which changes the observation semantics. Defer to
   after basic noise + miss are validated.

3. **Detection confidence in observations?** Ticket 010 §3 proposes adding confidence
   to the obs vector. This is orthogonal to calibration — it's an architecture decision.
   The calibrated miss rate already captures the main effect (small targets → more misses).
