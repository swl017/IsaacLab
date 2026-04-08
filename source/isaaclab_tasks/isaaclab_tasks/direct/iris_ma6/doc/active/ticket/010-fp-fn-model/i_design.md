## Design Document: Detector Replicator — FP/FN Model

### Problem statement

The existing detector replicator applies calibrated localization noise but produces deterministic detections — a target is either detected or not, with no stochastic miss rate and no false positives. Real YOLO detectors miss small/distant targets (miss rate 15-80% depending on size and background) and produce spurious detections from ground clutter (~0.5 FP/frame). The policy needs to learn to cope with this unreliability during training.

### Proposed approach

Extend the existing `DetectorReplicator` in `bbox_raycaster_v2` with two new capabilities applied as post-processing stages after noise:

**Stage order in `apply()`**: noise → miss rate → false positive → zero-out empty

1. **Miss rate (FN)**: For each valid detection (where `bbox_empty=False`), sample a Bernoulli miss with probability conditioned on (a) target bbox size in pixels and (b) background type (sky vs ground). Background is classified by casting a ray from the camera through the target and checking whether it hits the static mesh (ground) or passes through (sky). When missed: zero the bbox and set `bbox_empty_replicated=True`. Skip when `bbox_empty` is already True.

2. **False positive (FP)**: Per camera per step, sample a Bernoulli with probability `p_fp`. If triggered, generate a spurious bbox with random center location (uniform in image) and random size (uniform in configurable range). The FP bbox is written into the replicated output and `bbox_empty_replicated` is set to False. FPs enter the delay system as normal detections — they are indistinguishable from true positives in the observation path.

3. **Zero-out fix**: After all stages, explicitly zero bbox values wherever `bbox_empty_replicated=True`. This fixes the existing bug where noisy values leak through on empty frames.

**Background classification** uses the existing `static_mesh` (wp.Mesh) on `BBoxRayCasterV2`. A ray is cast from each camera position through the target position using `raycast_mesh()`. If the ray hits the static mesh, the background is ground (cluttered); if it misses (distance = inf), the background is sky (clean). This requires no new mesh loading — the static mesh is already loaded for occlusion checks.

**Curriculum gating**: Add `fp_fn_start_step` and `fp_fn_end_step` to `curriculum_cfg.py`. The FP/FN effect scales linearly from 0 to 1 over this range, multiplied into both `p_miss` and `p_fp`. Default placement: co-located with noise phase (100k-120k), since FP/FN is another form of observation degradation.

### Key interfaces and data flow

```
bbox_raycaster_v2.update(...)
    → data.bboxes          (N,C,T,4)  GT clean
    → data.bbox_empty      (N,C,T)    GT occlusion

bbox_raycaster_v2.apply_detector_replicator(noise_scale, fp_fn_scale)
    1. Clone GT bboxes
    2. Apply Student's t noise (existing, ticket-009)
    3. Classify background per camera-target pair:
       ray_start = camera_pos_w             (N,C,3)
       ray_dir   = normalize(target_pos - camera_pos)  (N,C,T,3)
       hit_dist  = raycast_mesh(ray_start, ray_dir, static_mesh)
       bg_is_ground = hit_dist < inf       (N,C,T) bool
    4. Compute miss probability:
       target_size = sqrt(w * h)            (N,C,T) pixels
       p_miss_sky  = sigmoid(a_sky * (thresh_sky - target_size))
       p_miss_gnd  = sigmoid(a_gnd * (thresh_gnd - target_size))
       p_miss = where(bg_is_ground, p_miss_gnd, p_miss_sky) * fp_fn_scale
       missed = bernoulli(p_miss) & ~bbox_empty
    5. Apply miss: zero bbox, set bbox_empty_replicated=True
    6. Generate FP: bernoulli(p_fp * fp_fn_scale) per camera
       fp_center = uniform(0, img_w) × uniform(0, img_h)
       fp_size   = uniform(fp_size_min, fp_size_max) for w, h
       Write FP bbox where triggered, set bbox_empty_replicated=False
    7. Zero-out: where bbox_empty_replicated, set bbox values to 0

    → data.bboxes_replicated       (N,C,T,4)  noisy + FN + FP
    → data.bbox_empty_replicated   (N,C,T)    updated mask

Env _update_state_cache():
    raycaster.update(...)
    raycaster.apply_detector_replicator(noise_scale, fp_fn_scale)
    for agent:
        gt_states.bboxes_2d    = data.bboxes[:, idx]            # reward path
        noisy_states.bboxes_2d = data.bboxes_replicated[:, idx] # obs path
    delay_system.update_ground_truth(..., replicated_bboxes=noisy_bboxes)

Reward path: delay_system.get_states_for_rewards() → GT bboxes (clean)
Obs path:    delay_system.get_states_for_observations() → replicated bboxes (noisy+FN+FP)
```

### Configuration extension

```python
# Extend NoiseModelParams with miss/FP fields
miss_sigmoid_a_sky: float       # Steepness for sky background
miss_size_threshold_sky: float  # Threshold (px) for sky
miss_sigmoid_a_gnd: float       # Steepness for ground background
miss_size_threshold_gnd: float  # Threshold (px) for ground
fp_rate: float                  # FP probability per camera per step
fp_size_range: (float, float)   # Min/max FP bbox size (pixels)

# Extend DetectorReplicatorCfg
apply_miss: bool = True         # Enable miss rate model
apply_fp: bool = True           # Enable false positive injection

# Extend CurriculumCfg
fp_fn_start_step: int = 100000  # Co-located with noise phase
fp_fn_end_step: int = 120000
```

Parameters loaded from the same calibration JSON (ticket-009/010 findings).

### Idempotency

Add `_last_update_time` guard to `apply_detector_replicator()` to prevent double-sampling noise/miss/FP within the same sim step. This addresses the gap identified in R research (item 1 in gaps).

### What this does NOT include

- Detection confidence in observations (deferred)
- Observation dimension changes
- SKRL config changes
- Changes to raycaster core (mesh intersection, occlusion)
- Changes to delay system pipeline stages
- Partial detection (truncated/non-axis-aligned bboxes)
- Calibration data collection (ticket-009 scope)
- Online YOLO during training

### Open risks

1. **`raycast_mesh` performance**: One additional ray per camera-target pair per step for background classification. The same function is already called for occlusion with 1-9 rays per pair, so one extra ray is negligible. But if occlusion is disabled (`enable_occlusion_check=False`), the static mesh may not be loaded — need to handle this case (default to sky if no mesh).

2. **FP bbox target index**: The raycaster output has shape (N, C, T, 4) where T=num_targets. FP must overwrite a target slot. With T=1, FP overwrites the real detection — a missed-then-FP'd frame shows a random bbox instead of empty. With T>1, FP could use an unused target slot. Current env uses T=1.

3. **FP and miss interaction**: If a real detection is first missed (FN), then a FP is generated in the same step, the FP replaces the zeroed-out bbox. This is correct behavior — the detector "missed" the real target but "hallucinated" something else. The policy sees a bbox that doesn't correspond to the real target.

4. **Calibration parameter availability**: Ticket-009 calibration provides overall miss rate and FP rate. The sky/ground split comes from ticket-010 findings. If these aren't yet fitted as dual sigmoids, placeholder parameters suffice.