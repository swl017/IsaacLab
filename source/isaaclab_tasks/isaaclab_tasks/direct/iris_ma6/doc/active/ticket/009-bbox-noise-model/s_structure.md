## Structure Outline

### New files

- `bbox_raycaster_v2/detector_replicator.py`
  - `NoiseModelParams` — dataclass: center_noise_a/b/df, size_noise_a/b/df, center_bias_x/y, fit_size_range
  - `DetectorReplicatorCfg` — @configclass: enabled, params_path, apply_bias
  - `DetectorReplicator`
    - `__init__(cfg: DetectorReplicatorCfg, num_envs: int, device: str)`
    - `load_params(path: str)` → loads JSON into NoiseModelParams
    - `apply(bboxes: Tensor, bbox_empty: Tensor, noise_scale: float) -> Tuple[Tensor, Tensor]`
      — returns (bboxes_replicated, bbox_empty_replicated)
      — computes target_size from bbox w*h, applies Student's t noise with scale=a/size+b, adds bias
      — only noises where bbox_empty=False
      — noise_scale for curriculum gating (0=no noise, 1=full calibrated noise)

- `bbox_raycaster_v2/detector_replicator_cfg.py` (ALTERNATIVE: keep cfg + dataclass in detector_replicator.py if small enough)
  - Merge into `detector_replicator.py` — single file is sufficient at ~80 lines

### Modified files

- `bbox_raycaster_v2/bbox_raycaster_v2_data.py`
  - [add] `bboxes_replicated: torch.Tensor` — (N,C,T,4) xywh, noisy
  - [add] `bboxes_xyxy_replicated: torch.Tensor` — (N,C,T,4) xyxy, noisy
  - [add] `bbox_empty_replicated: torch.Tensor` — (N,C,T) bool, includes FN (ticket-010 future)
  - [modify] `__init__` or `reset` — initialize new fields to match existing shapes

- `bbox_raycaster_v2/bbox_raycaster_v2.py`
  - [add] `apply_detector_replicator(noise_scale: float)` — calls DetectorReplicator.apply() on self._data.bboxes, writes results to self._data.bboxes_replicated / bbox_empty_replicated

- `bbox_raycaster_v2/__init__.py`
  - [add] export `DetectorReplicator`, `DetectorReplicatorCfg`, `NoiseModelParams`

- `bbox_raycaster_v2/CONTEXT.md`
  - [modify] add detector_replicator to key files, update outputs section

- `delay_system_v3/field_storage.py`
  - [modify] `store(field_name, data, timestamp, noise_std, noisy_data=None)` — when noisy_data is provided, store it directly as `_noisy[field_name]` instead of generating Gaussian noise

- `delay_system_v3/multi_agent_wrapper.py`
  - [modify] `update_ground_truth()` — accept optional `replicated_bboxes` parameter; pass as `noisy_data` to `store()` for the bboxes_2d field; set `noise_std=0.0` when replicated bboxes provided

- `iris_ma_env6_test_cfg.py`
  - [add] `calibrated_bbox_noise: DetectorReplicatorCfg` — config for detector replicator (enabled, params_path, apply_bias)

- `iris_ma_env6_test.py`
  - [modify] `__init__` — if cfg.calibrated_bbox_noise.enabled: create DetectorReplicator, load JSON params; set delay_system bbox_std to 0
  - [modify] `_update_state_cache()` — after raycaster.update(): call raycaster.apply_detector_replicator(noise_scale); pass replicated bboxes to delay_system.update_ground_truth() via new parameter
  - [existing] `_get_rewards()` — no change (uses get_raw → clean GT bboxes)
  - [existing] `_get_observations()` — no change (uses get_delayed with use_noise=True → replicated bboxes from _noisy dict)
