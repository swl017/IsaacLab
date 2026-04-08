## Implementation Plan — Ticket 010: FP/FN Model

### Slice 1: Miss rate model in DetectorReplicator

- Step 1.1: Extend `NoiseModelParams` with miss rate fields (`miss_sigmoid_a_sky/gnd`, `miss_size_threshold_sky/gnd`) and update `from_json()` with backward-compatible defaults in `detector_replicator.py`
- Step 1.2: Add `apply_miss: bool = True` to `DetectorReplicatorCfg` in `detector_replicator.py`
- Step 1.3: Implement `_apply_miss(bboxes, bbox_empty, target_size, bg_is_ground, fp_fn_scale)` in `DetectorReplicator` — dual sigmoid conditioned on background, Bernoulli sampling, zero-out on miss
- Step 1.4: Modify `apply()` signature to accept `fp_fn_scale` and `bg_is_ground`; call `_apply_miss()` after noise stage
- Step 1.5: Add `_classify_background()` to `BBoxRayCasterV2` — raycast from camera through target on `static_mesh`, return `bg_is_ground` (N,C,T) bool
- Step 1.6: Modify `apply_detector_replicator()` to accept `fp_fn_scale`, call `_classify_background()`, pass results to replicator
- Step 1.7: Add `_last_replicator_time` idempotency guard to `apply_detector_replicator()`

- **Test checkpoint**: Unit test — construct replicator with known miss params, pass bboxes with `bg_is_ground=True` and `bg_is_ground=False`, verify miss rates differ and scale with `fp_fn_scale`. Verify `bbox_empty=True` inputs are not double-missed. Verify idempotency guard prevents double-call.

### Slice 2: False positive injection

- Step 2.1: Extend `NoiseModelParams` with FP fields (`fp_rate`, `fp_size_range`) and update `from_json()` in `detector_replicator.py`
- Step 2.2: Add `apply_fp: bool = True` to `DetectorReplicatorCfg` in `detector_replicator.py`
- Step 2.3: Implement `_apply_fp(bboxes, bbox_empty, fp_fn_scale, image_shapes)` in `DetectorReplicator` — Bernoulli per camera, random center/size, write into target slot
- Step 2.4: Modify `apply()` to accept `image_shapes`, call `_apply_fp()` after miss stage
- Step 2.5: Ensure zero-out stage at end of `apply()` zeros bboxes where `bbox_empty_replicated=True` (bug fix)

- **Test checkpoint**: Unit test — set `fp_rate=1.0`, verify every frame gets a FP bbox. Set `fp_rate=0.0`, verify no FPs. Verify FP bbox center is within image bounds and size within configured range. Verify zero-out: bboxes are 0 where `bbox_empty_replicated=True`.

### Slice 3: Curriculum gating and env integration

- Step 3.1: Add `fp_fn_start_step`, `fp_fn_end_step`, `get_fp_fn_progress()` to `curriculum_cfg.py`
- Step 3.2: In `iris_ma_env6_test.py` `_get_rewards()` curriculum section: compute `fp_fn_progress` via `curriculum.get_fp_fn_progress(current_step)`
- Step 3.3: In `iris_ma_env6_test.py` `_update_state_cache()`: pass `fp_fn_scale=self._fp_fn_progress` to `apply_detector_replicator()`
- Step 3.4: Pass `image_shapes` from raycaster data to `apply_detector_replicator()` (needed for FP placement bounds)

- **Test checkpoint**: Verify curriculum ramp: at step 0, `fp_fn_progress=0.0` (no miss/FP); at step 110k, `fp_fn_progress=0.5`; at step 120k+, `fp_fn_progress=1.0`. Run env for a few steps with replicator enabled and verify no crashes, observation vector unchanged in dimension.
