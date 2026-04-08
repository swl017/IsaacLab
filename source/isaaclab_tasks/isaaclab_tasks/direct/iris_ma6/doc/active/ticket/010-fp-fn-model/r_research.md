## Codebase Research Report — Ticket 010

### Module inventory

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| `bbox_raycaster_v2/` | 12 | 3760 | GPU-accelerated 2D bbox extraction with occlusion, detector replicator |
| `delay_system_v3/` | 13 | 6632 | Multi-agent communication latency, noise, dropout simulation |
| `curriculum/` | 1 | ~200 | Training difficulty scheduling (7 sequential phases) |
| `iris_ma_env6_test.py` | 1 | ~2500 | Main environment orchestration |
| `iris_ma_env6_test_cfg.py` | 1 | ~800 | Environment configuration |

---

### bbox_raycaster_v2 — relevant structures

**BBoxRayCasterV2** (main class):
- `update(camera_poses, camera_intrinsics, target_poses, agent_poses, image_shapes, target_scale)` → fills `data` with GT bboxes
- `setup_detector_replicator(cfg: DetectorReplicatorCfg)` → configures noise replicator
- `apply_detector_replicator(noise_scale: float)` → writes replicated bboxes to `data.bboxes_replicated`, `data.bboxes_xyxy_replicated`, `data.bbox_empty_replicated`
- `get_normalized_bboxes(bboxes_xywh)` → returns (N, C, T, 4) normalized to [0,1]

**BBoxRayCasterV2Data** (output container, all tensors):

| Field | Shape | Description |
|-------|-------|-------------|
| `bboxes` | (N, C, T, 4) | GT bboxes (cx, cy, w, h) pixels |
| `bboxes_xyxy` | (N, C, T, 4) | GT bboxes (x_min, y_min, x_max, y_max) pixels |
| `bboxes_normalized` | (N, C, T, 4) | GT bboxes normalized [0,1] |
| `bbox_empty` | (N, C, T) | True = invalid/occluded |
| `bbox_confidence` | (N, C, T) | Confidence [0,1] from occlusion tests |
| `bboxes_replicated` | (N, C, T, 4) | GT + calibrated noise (xywh) |
| `bboxes_xyxy_replicated` | (N, C, T, 4) | Replicated in xyxy format |
| `bbox_empty_replicated` | (N, C, T) | Empty mask for replicated bboxes |
| `occlusion_visibility_ratio` | (N, C, T) | Fraction of visible test points |
| `ray_dir_w` | (N, C, T, 3) | Ray directions camera→target |

**DetectorReplicator** (existing, ticket-009):
- Constructor: `DetectorReplicator(cfg: DetectorReplicatorCfg, device: str)`
- `load_params(path: str)` → loads from JSON
- `apply(bboxes_xywh, bbox_empty, noise_scale)` → returns (replicated_bboxes, empty_mask)
- Applies Student's t noise to center (cx, cy) and size (w, h) independently
- Only applies to non-empty bboxes
- NOT idempotent — each call samples fresh noise

**NoiseModelParams** (dataclass):
- `center_noise_a/b/df`: size-dependent scale = a/target_size + b, Student's t df
- `size_noise_a/b/df`: same for bbox width/height
- `center_bias_x/y`: systematic offset (pixels)
- `fit_size_range`: (min_px, max_px) calibration regime
- No miss rate or FP fields currently

**DetectorReplicatorCfg**:
- `enabled: bool = True`
- `params_path: str = ""`
- `apply_bias: bool = True`

**Mesh handles** (accessible after init):
- `self.static_mesh: wp.Mesh | None` — environment geometry (ground plane, terrain)
- `self.agent_meshes: Dict[str, wp.Mesh]` — per-agent drone meshes
- Both are Warp mesh objects, usable for `raycast_mesh()` queries
- Static mesh constructed from `mesh_prim_paths` config; infinite plane if prim is Plane type
- Meshes are loaded once at init, read-only thereafter

**Raycasting function** (from isaaclab):
- `isaaclab.utils.warp.raycast_mesh(ray_starts, ray_directions, mesh)` → (distances, normals, face_ids)
- Already used by occlusion check in `utils/occlusion_fully_batched.py`

---

### delay_system_v3 — relevant structures

**FieldStorage** (dual-path storage):
- Stores `_raw[field_name]` (ground truth) and `_noisy[field_name]` (with noise) separately
- Single `_timestamps[field_name]` shared by both
- `store(field_name, data, noise_std, noisy_data=None)`:
  - If `noisy_data` provided → stored directly as `_noisy` (pre-noised injection)
  - Elif `noise_std > 0` → `_noisy = data + randn() * noise_std`
  - Else → `_noisy = data.clone()`

**MultiAgentDelaySystemV3.update_ground_truth()** — bbox path:
```python
if replicated_bboxes is not None:
    self._delay_system.store(
        prefix + "bboxes_2d",
        data.bboxes_2d,              # Clean GT
        noise_std=0.0,
        noisy_data=replicated_bboxes  # Pre-noised from detector replicator
    )
else:
    bbox_noise = self._get_noise_std("bbox", agent_id)
    self._delay_system.store(
        prefix + "bboxes_2d",
        data.bboxes_2d,
        noise_std=bbox_noise,        # Gaussian noise
    )
```

**Two retrieval paths**:

| Method | use_noise | allow_dropout | Used for |
|--------|-----------|---------------|----------|
| `get_states_for_rewards()` | False (default) | False | Reward computation |
| `get_states_for_observations()` | True | True | Observation vector |

- `use_noise=False` → reads from `_raw` (GT)
- `use_noise=True` → reads from `_noisy` (replicated/noisy)
- `allow_dropout=False` → skips dropout stage (rewards always receive data)
- `allow_dropout=True` → applies dropout masking (obs may miss frames)

**Pipeline stages** (applied to both raw and noisy independently):
1. Staleness (FPS limiting)
2. Latency (communication delay buffer)
3. First-order lag (smoothing)
4. Dropout (packet loss — obs only)

**NoiseCfg.bbox_std**: `DistributionCfg(type="constant", value=7.0, min_value=0.0)`
- Default 7.0 pixels, Gaussian
- Curriculum-scaled via `set_noise_scale(scale)` → `base_std * scale * per_agent_scale`

**AgentStates.bboxes_2d**: shape `[N, T, 4]`, pixel coordinates (xywh)
- `bboxes_2d_valid_mask` has been REMOVED — external validation required

---

### curriculum — relevant structures

**Seven sequential phases** with linear ramp between `start_step` and `end_step`:

| Phase | Start | End | Controls |
|-------|-------|-----|----------|
| Agent velocity | 20k | 40k | `max_lin_vel` ramp |
| Tracking geometry | 20k | 60k | Spawn randomization |
| Safety CBF | 20k | 40k | Collision penalty scale |
| Target motion | 40k | 80k | Target controller difficulty |
| Coordination | 60k | 100k | Triangulation reward scale |
| Noise | 100k | 120k | `set_noise_scale()` [0→1] |
| Fixed delay | 120k | 140k | Latency means, no variance |
| Random delay | 140k | 160k | Latency variance + staleness |
| Dropout | 160k | 180k | Dropout probability [0→5%] |
| Dynamics DR | 180k | 200k | Camera/mass/gimbal randomization |

**Progress computation**: `get_progress(step, start, end)` → linear [0.0, 1.0]

**No existing FP/FN phase** — noise phase (100k-120k) is the closest analogue.

**Task reward levels** (separate curriculum):
- Level 1: FIM proxy (orientation-based)
- Level 2: GT-anchored triangulation error (steps 40k-80k)
- Level 3: E2E triangulation error (steps 80k-120k)

---

### Environment integration — relevant wiring

**_update_state_cache()** — bbox flow:
1. `bbox_raycaster_v2.update(...)` → fills GT bboxes
2. If `calibrated_bbox_noise.enabled`: `bbox_raycaster_v2.apply_detector_replicator(noise_scale)` → fills replicated bboxes
3. For each agent: builds `AgentStates`, sets `bboxes_2d` from raycaster data
4. If replicator enabled: passes `replicated_bboxes` to `update_ground_truth()`
5. Delay system stores both GT (_raw) and replicated (_noisy) versions

**_get_rewards()** — bbox usage:
- Calls `delay_system.get_all_states_for_rewards()` with `use_noise=False, allow_dropout=False`
- Extracts `bboxes_2d` from returned states → these are **GT (clean) delayed** bboxes
- Computes `bbox_center_reward` and `bbox_size_reward` from normalized GT bboxes
- CBF penalty: always from GT positions (not from delay system)

**_get_observations()** — bbox usage:
- Calls `delay_system.get_all_states_for_observations()` with `use_noise=True, allow_dropout=True`
- Extracts `bboxes_2d` from returned states → these are **replicated (noisy) delayed + dropout** bboxes
- Normalizes for observation vector (4D bbox + 1D bbox_empty per agent)
- Bbox validity: `bbox_valid = (bbox_pixel.abs().sum(dim=-1) > 1e-6)`

**Observation vector** (per agent, 30D ego + 16D per other + 6D triangulation):
- Bbox fields in ego obs: `bbox_normalized` (4D), `bbox_empty` (1D), `bbox_aoi` (1D)
- Bbox fields in other-agent obs: `bbox_empty` (1D), `bbox_age` (1D)

---

### Conventions observed

1. **Dual-path already exists**: FieldStorage stores raw + noisy separately. Reward path reads raw (GT), obs path reads noisy. The detector replicator injects pre-noised data as the noisy version.
2. **Raycaster produces both GT and replicated outputs**: `data.bboxes` vs `data.bboxes_replicated` already exist as separate fields.
3. **bbox_empty zeroing**: After `update()`, `_sync_public_outputs()` zeros bboxes where `bbox_empty=True`. The replicated path in `apply_detector_replicator()` also zeros empty bboxes.
4. **Curriculum gating pattern**: `progress = get_progress(step, start, end)` → multiply into effect. Used consistently for safety, coordination, noise, delay, dropout.
5. **Noise scale multiplication**: `set_noise_scale(scale)` multiplies base noise std. Scale 0→1 during curriculum.
6. **Per-agent heterogeneity**: delay, noise, dropout can vary per agent via `per_agent_randomization`.
7. **Mesh accessibility**: `self.static_mesh` (wp.Mesh) is accessible on BBoxRayCasterV2 after init. Raycasting utility `raycast_mesh()` already imported and used.
8. **No FP generation exists anywhere**: No mechanism to inject spurious bboxes.
9. **No background-dependent logic exists**: No sky-vs-ground queries on the mesh.
10. **Bbox format**: Stored as xywh (center_x, center_y, width, height) in pixels. Also available as xyxy and normalized.

---

### Gaps and inconsistencies

1. **DetectorReplicator `apply()` is not idempotent**: Each call samples fresh noise. No `_last_update_time` guard. Contrast with delay system's `advance()` which has an idempotency guard.
2. **`bbox_empty_replicated` copies `bbox_empty` directly**: No miss rate logic — replicated empty mask is identical to GT empty mask. The field exists but does not differ from GT.
3. **NoiseModelParams has no miss/FP fields**: Only localization noise parameters. No `miss_sigmoid_a`, `miss_size_threshold`, `p_fp`, or background-type conditioning.
4. **No `raycast_mesh` for background classification**: The static mesh is used for occlusion testing only. No utility for "cast ray through target, check if it hits ground or sky."
5. **Curriculum has no FP/FN phase**: Noise phase (100k-120k) controls `set_noise_scale()`. No equivalent for miss rate or FP rate.
6. **Reward path reads raw, obs path reads noisy — but no explicit "replicated" vs "Gaussian noise" distinction in delay system config**: The `replicated_bboxes` kwarg is an implementation detail of `update_ground_truth()`, not a config-level choice.
7. **`bbox_raycaster_v2.__init__` exports DetectorReplicator and NoiseModelParams**: These are already public API.
8. **Target size available at replicator time**: `apply()` internally computes `target_size = sqrt(w * h)` from the bbox. This can be reused for miss rate conditioning.
9. **`data.bbox_confidence`** exists and is computed from occlusion visibility ratio, but is NOT used in observations or rewards. It is only used for temporal EMA smoothing in `_update_state_cache()`.