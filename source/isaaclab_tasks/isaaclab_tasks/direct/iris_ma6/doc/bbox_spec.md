# Object Detection Module Specification

**Applies to**: iris_ma6 (V6) multi-target observation/interception environment
**Base implementation**: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/bbox_raycaster`
**Purpose**: Convert per-camera geometric visibility into training-time detection outputs with configurable realism

---

## 1) Scope

The object detection module is the perception layer that sits between:

- `BBoxRayCaster`: geometric projection, FoV validation, and mesh-based occlusion checks
- Downstream consumers: triangulation, target assignment, reward shaping, and observations

For V6, the detector must support:

- Multi-camera, multi-target detection tensors with shape `[N_env, C, T_max, ...]`
- Deterministic ground-truth association in simulation
- Optional inter-target occlusion masking
- Detection corruption for false negatives and false positives
- Explicit empty-box signaling via `bbox_empty`
- Confidence outputs so downstream modules can distinguish weak detections from clean detections

This document defines the detector contract, corruption model, and rollout phases.

---

## 2) Design Goals

1. Reuse V5 geometry: Keep corner projection, bbox validation, and environment/agent occlusion in `bbox_raycaster`
2. Separate geometry from realism: Realistic misses and spurious detections should be injected after geometric visibility is computed
3. Preserve batched tensors: No Python-side per-env object lists in the core update path
4. Keep GT association available: Realistic detection noise is added, but target identity remains known in simulation unless explicitly disabled for future research
5. Support curriculum: Start with clean detections, then progressively add occlusion, false negatives, and false positives

---

## 3) Module Placement

Recommended package structure:

```text
iris_ma6/object_detection/
├── object_detector.py
├── object_detector_cfg.py
├── object_detector_data.py
├── false_detection_sampler.py
└── occlusion_handler.py
```

Responsibilities:

- `bbox_raycaster/`: geometric candidates and visibility
- `occlusion_handler.py`: inter-target occlusion refinement for overlapping targets in the same image
- `false_detection_sampler.py`: false-negative and false-positive injection
- `object_detector.py`: orchestration and final output tensor assembly

---

## 4) Inputs and Outputs

### 4.1 Inputs

Per step, the detector consumes:

- `bboxes_geom[N_env, C, T_max, 4]`: projected bboxes from `BBoxRayCaster`
- `bbox_empty_geom[N_env, C, T_max]`: geometric emptiness mask after FoV and size checks
- `visibility_ratio[N_env, C, T_max]`: occlusion visibility ratio from mesh raycasting
- `camera_pos_w[N_env, C, 3]`
- `target_pos_w[N_env, T_max, 3]`
- `target_alive[N_env, T_max]`
- Optional image metadata: height, width, focal length, zoom level

### 4.2 Outputs

The detector publishes:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `bbox` | `[N_env, C, T_max, 4]` | Final bbox in normalized `xywh` |
| `bbox_empty` | `[N_env, C, T_max]` | Empty flag: `1` if bbox is empty/zero-filled, `0` otherwise |
| `bbox_confidence` | `[N_env, C, T_max]` | Confidence in `[0, 1]` |
| `bbox_source` | `[N_env, C, T_max]` | Enum: GT, occluded-drop, false-negative, false-positive, invalid |
| `occluded` | `[N_env, C, T_max]` | Inter-target occlusion mask |
| `false_positive_mask` | `[N_env, C, T_max]` | Marks slots populated by synthetic detections |
| `miss_reason` | `[N_env, C, T_max]` | Enum for diagnostics: none, FoV, size, static-occlusion, target-occlusion, FN-sampled |

`bbox_source` and `miss_reason` are not required in the policy observation, but should exist in the module data for debugging and evaluation.

---

## 5) Detection Pipeline

### 5.1 Stage A: Geometric Detection

Use the V5 `BBoxRayCaster.update(...)` output directly:

1. Project 3D target corners into each camera
2. Compute 2D bbox per camera-target pair
3. Apply FoV, size, and raycast-based environment/agent occlusion checks

Result:

- `bbox_geom`
- `bbox_empty_geom`
- `visibility_ratio`

### 5.2 Stage B: Inter-Target Occlusion

V5 raycasting handles static-scene and agent-body occlusion, but not target-vs-target overlap. V6 adds an image-space occlusion pass:

1. For each environment and camera, compute target depth
   $$d^{(c,t)} = \|p^{(t)} - p_{\mathrm{cam}}^{(c)}\|$$
2. For all alive targets with `bbox_empty_geom = 0`, compute pairwise bbox IoU
3. If two bboxes overlap and one target is clearly nearer, suppress or down-weight the farther one

Recommended rule:

- If `IoU > tau_occlude_hard` and `d_near + margin < d_far`, set `occluded=True`, zero-fill the farther bbox, and set `bbox_empty=1`
- If `tau_occlude_soft < IoU <= tau_occlude_hard`, keep the detection but reduce confidence

Recommended defaults:

- `tau_occlude_soft = 0.10`
- `tau_occlude_hard = 0.30`
- `depth_margin_m = 1.0`

Confidence after soft occlusion:

$$
\mathrm{conf}_{occ}^{(c,t)} = \mathrm{conf}_{geom}^{(c,t)} \cdot (1 - \lambda_{occ} \cdot \mathrm{IoU})
$$

with `lambda_occ = 0.5` initially.

### 5.3 Stage C: False Negative Injection

False negatives model missed detections even when a target is geometrically visible.

For each non-empty GT detection, sample:

$$
z_{fn}^{(c,t)} \sim \mathrm{Bernoulli}(p_{fn}^{(c,t)})
$$

If `z_fn = 1`:

- `bbox = [0, 0, 0, 0]`
- `bbox_empty = 1`
- `bbox_confidence = 0`
- `miss_reason = FN_SAMPLED`

False-negative probability should be state-dependent:

$$
p_{fn}^{(c,t)} = \mathrm{clip}(p_{0,fn} + w_d f_d + w_v f_v + w_s f_s + w_o f_o,\ 0,\ p_{fn,max})
$$

Suggested factors:

- `f_d`: normalized distance penalty
- `f_v = 1 - visibility_ratio`: partial occlusion penalty
- `f_s`: penalty for very small bbox area
- `f_o`: off-center penalty based on image-center distance

Recommended curriculum:

- Phase 0: `p_0_fn = 0`
- Phase 1: constant `p_0_fn = 0.02`
- Phase 2: state-dependent `p_fn` with cap `p_fn_max = 0.20`

### 5.4 Stage D: False Positive Injection

False positives model detector hallucinations. These are synthetic bboxes with no real target behind them.

Sampling policy per environment-camera pair:

$$
n_{fp}^{(c)} \sim \mathrm{Bernoulli}(p_{fp,frame})
$$

If sampled, place one synthetic detection into an unused or dedicated detection slot.

Initial V6 rule:

- Reuse the target-slot layout `[N_env, C, T_max, 4]`
- Only inject FPs into slots where `target_alive=False` or `bbox_empty=1`
- Mark them with `false_positive_mask=True`
- Exclude them from triangulation and GT reward terms

False-positive bbox generation:

- `cx, cy`: uniform over image, optionally biased toward image edges
- `w, h`: sampled from a bounded prior matching plausible target sizes
- `confidence`: low-to-medium, e.g. `Uniform(0.15, 0.55)`

Recommended constraints:

- `max_fp_per_camera = 1` initially
- `p_fp_frame = 0.01` to `0.05`
- Never overwrite a valid GT detection

### 5.5 Stage E: Final Selection

Final emptiness is:

$$
\mathrm{bbox\_empty} = \neg \left((\neg \mathrm{bbox\_empty\_geom} \land \neg \mathrm{occluded} \land \neg z_{fn}) \lor \mathrm{false\_positive}\right)
$$

Downstream rule:

- Empty bbox tensors are always written as `[0, 0, 0, 0]`
- Downstream modules must use `bbox_empty`, not floating-point zero checks, to decide whether a bbox exists
- Triangulation uses only detections with `bbox_empty=0` and `false_positive_mask=False`
- Observation features may include both `bbox_empty` and `bbox_confidence`
- Evaluation should separately report GT recall and FP rate

---

## 6) Confidence Model

Confidence starts from geometric quality and is then degraded by occlusion/noise:

$$
\mathrm{conf}_{geom}^{(c,t)} =
\mathbb{1}[\neg \mathrm{bbox\_empty\_geom}]
\cdot
\mathrm{clip}(w_{vis}\,r_{vis} + w_{area}\,r_{area} + w_{center}\,r_{center}, 0, 1)
$$

Where:

- `r_vis = visibility_ratio`
- `r_area`: score for bbox area within target range
- `r_center`: score for central image placement

Then:

- hard occlusion or FN: confidence `= 0`
- soft occlusion: multiplicative reduction
- false positives: sampled directly from FP confidence prior

Recommended initial weights:

- `w_vis = 0.5`
- `w_area = 0.3`
- `w_center = 0.2`

---

## 7) Data Association Policy

For simulation training, target identity remains known:

- GT-backed detections stay in their target index slot
- False positives are flagged explicitly and must not be interpreted as real targets

This is intentionally simpler than real-world MOT/re-identification. A future research extension may replace fixed target slots with a detection list plus association stage.

---

## 8) Configuration Surface

Suggested config:

```python
@configclass
class ObjectDetectorCfg:
    enable_inter_target_occlusion: bool = False
    tau_occlude_soft: float = 0.10
    tau_occlude_hard: float = 0.30
    depth_margin_m: float = 1.0

    enable_false_negatives: bool = False
    false_negative_base_prob: float = 0.0
    false_negative_max_prob: float = 0.20
    fn_distance_weight: float = 0.25
    fn_visibility_weight: float = 0.35
    fn_size_weight: float = 0.25
    fn_offcenter_weight: float = 0.15

    enable_false_positives: bool = False
    false_positive_frame_prob: float = 0.02
    max_false_positives_per_camera: int = 1
    false_positive_confidence_range: tuple[float, float] = (0.15, 0.55)
    false_positive_size_range: tuple[float, float] = (0.02, 0.20)

    emit_debug_tensors: bool = True
```

The detector config should wrap, not duplicate, `BBoxRayCasterCfg`.

---

## 9) Observation and Triangulation Integration

### 9.1 Observation

Per-target observation fields should use final detector outputs:

- `ego bbox (primary target)`: selected from final `bbox`
- `bbox empty`: from final `bbox_empty`
- optional future extension: append `bbox_confidence`

### 9.2 Triangulation

Triangulation consumes only non-false-positive detections:

```python
tri_input = (bbox_empty == 0) & (~false_positive_mask) & target_alive
```

Expected effects:

- False negatives reduce camera count and increase covariance
- False positives should not corrupt triangulation because they are filtered before fusion
- Inter-target occlusion reduces multi-view availability for farther targets

---

## 10) Rollout Phases

### Phase 0: Clean Geometry

- `enable_inter_target_occlusion = False`
- `enable_false_negatives = False`
- `enable_false_positives = False`

Matches current V5-style behavior while supporting multi-target tensor shapes.

### Phase 1: Occlusion-Aware Detection

- Enable inter-target occlusion
- Keep FP/FN disabled

Goal: ensure depth ordering and overlap masking behave correctly.

### Phase 2: Missed Detection Realism

- Enable false negatives
- Keep false positives low or disabled

Goal: train robustness to intermittent observation loss.

### Phase 3: Full Detection Corruption

- Enable false negatives and false positives
- Tune rates through curriculum or domain randomization

Goal: approximate deployment-time perception imperfections.

---

## 11) Validation Criteria

### 11.1 Geometric and Occlusion

- Near target fully overlapping far target in one camera causes the far target bbox to become empty
- Repositioning the camera restores the previously occluded target
- Non-overlapping targets remain unaffected

### 11.2 False Negatives

- With `false_negative_base_prob = 0`, output matches deterministic geometry
- Increasing `p_fn` reduces empirical recall monotonically
- FN events increase target AoI and triangulation covariance as expected

### 11.3 False Positives

- FP detections appear only in allowed slots
- FP confidence stays within configured range
- FP detections never contribute to triangulation
- Logged FP/frame rate matches the configured sampling distribution within tolerance

### 11.4 Metrics

Track per camera and per episode:

- precision
- recall
- false positive rate per frame
- false negative rate on geometrically valid targets
- mean confidence for TP vs FP
- triangulation covariance vs number of valid cameras

---

## 12) Non-Goals for Initial V6

The initial detector spec does not require:

- learned image-based detection
- appearance embeddings or target re-identification
- NMS across arbitrary proposal sets
- persistent FP tracks across many frames

Those can be added later if the research scope shifts from privileged simulation detection toward raw-vision perception.
