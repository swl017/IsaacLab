## Implementation Plan

### Slice 1: DetectorReplicator core + data fields

- Step 1.1: Create `bbox_raycaster_v2/detector_replicator.py` with `NoiseModelParams` (dataclass, JSON load/save), `DetectorReplicatorCfg` (@configclass: enabled, params_path, apply_bias), and `DetectorReplicator` class with `__init__`, `load_params()`, `apply(bboxes, bbox_empty, noise_scale) -> (bboxes_replicated, bbox_empty_replicated)`
- Step 1.2: Add replicated fields to `bbox_raycaster_v2/bbox_raycaster_v2_data.py`: `bboxes_replicated`, `bboxes_xyxy_replicated`, `bbox_empty_replicated` — initialized to clones of GT in `__init__`/reset
- Step 1.3: Add `apply_detector_replicator(noise_scale)` method to `bbox_raycaster_v2.py` — calls replicator, writes to `_data.bboxes_replicated` etc.
- Step 1.4: Update `bbox_raycaster_v2/__init__.py` with exports
- Test checkpoint: Standalone test — create DetectorReplicator with known params JSON, feed synthetic bboxes, verify output shape, verify noise is Student's t distributed (check mean/std/kurtosis), verify bbox_empty masking (noise only on valid bboxes), verify noise_scale=0 produces clean output

### Slice 2: field_storage + wrapper plumbing

- Step 2.1: Add `noisy_data: Optional[torch.Tensor] = None` parameter to `field_storage.py:store()` — when provided, store directly as `_noisy[field_name]` instead of generating Gaussian noise
- Step 2.2: Modify `multi_agent_wrapper.py:update_ground_truth()` — accept optional `replicated_bboxes: Optional[torch.Tensor] = None`; pass as `noisy_data` to the bboxes_2d store call; keep `noise_std=0.0` when replicated_bboxes is provided
- Test checkpoint: Unit test — store with `noisy_data`, verify `get_raw()` returns clean data and `get_noisy()` returns the pre-noised data; verify existing codepath (noise_std>0, no noisy_data) still works

### Slice 3: Env integration + config

- Step 3.1: Add `calibrated_bbox_noise: DetectorReplicatorCfg` to `iris_ma_env6_test_cfg.py` (default: disabled)
- Step 3.2: In `iris_ma_env6_test.py.__init__()` — if enabled: create DetectorReplicator on BBoxRayCasterV2, load params from JSON; set delay system `noise_bbox_std=0`
- Step 3.3: In `iris_ma_env6_test.py._update_state_cache()` — after `bbox_raycaster_v2.update()`: call `apply_detector_replicator(noise_scale)`; pass `replicated_bboxes` to `delay_system.update_ground_truth()`
- Step 3.4: Update `bbox_raycaster_v2/CONTEXT.md` — add detector_replicator to key files, update outputs
- Test checkpoint: Run env with `calibrated_bbox_noise.enabled=True` and a test JSON, verify: (1) observations contain noisy bboxes, (2) rewards use clean GT bboxes, (3) no crash with `--num_envs 4`, (4) noise_scale=0 curriculum start produces clean obs
