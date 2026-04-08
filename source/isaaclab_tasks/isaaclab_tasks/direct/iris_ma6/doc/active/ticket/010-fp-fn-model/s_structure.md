## Structure Outline — Ticket 010: FP/FN Model

### Modified files

#### 1. `bbox_raycaster_v2/detector_replicator.py`

```python
# --- Existing, no change ---
@dataclass
class NoiseModelParams:
    # ... existing noise fields unchanged ...

    # [add] Miss rate fields
    miss_sigmoid_a_sky: float = 0.031
    """Miss sigmoid steepness for sky background."""

    miss_size_threshold_sky: float = 65.0
    """Miss sigmoid threshold (px) for sky background."""

    miss_sigmoid_a_gnd: float = 0.031
    """Miss sigmoid steepness for ground background."""

    miss_size_threshold_gnd: float = 50.0
    """Miss sigmoid threshold (px) for ground background. Lower = more aggressive miss."""

    fp_rate: float = 0.01
    """False positive probability per camera per step."""

    fp_size_range: Tuple[float, float] = (10.0, 60.0)
    """Min/max FP bbox size in pixels (w and h sampled independently)."""

    @classmethod
    def from_json(cls, path: str) -> "NoiseModelParams":
        # [modify] Load new miss/FP fields from JSON (with backward-compatible defaults)


# --- Existing, no change ---
@configclass
class DetectorReplicatorCfg:
    enabled: bool = True
    params_path: str = ""
    apply_bias: bool = True

    # [add]
    apply_miss: bool = True
    """Enable probabilistic miss rate (FN). Requires enabled=True."""

    apply_fp: bool = True
    """Enable false positive injection. Requires enabled=True."""


# --- Existing class ---
class DetectorReplicator:

    def __init__(self, cfg, device):
        # [existing] — no change

    # [existing] — no change
    @property
    def params(self) -> Optional[NoiseModelParams]: ...

    # [existing] — no change
    def load_params(self, path: str): ...

    # [modify] Add fp_fn_scale param, add miss+FP stages, add bg_is_ground param
    def apply(
        self,
        bboxes_xywh: torch.Tensor,       # (N, C, T, 4)
        bbox_empty: torch.Tensor,         # (N, C, T)
        noise_scale: float = 1.0,
        fp_fn_scale: float = 1.0,         # [add] Curriculum scale for miss/FP [0,1]
        bg_is_ground: torch.Tensor | None = None,  # [add] (N, C, T) bool, from raycast_mesh
        image_shapes: torch.Tensor | None = None,   # [add] (N, C, 2) for FP placement
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply noise, miss rate, and FP to bounding boxes.

        Stage order: noise → miss → FP → zero-out empty.
        """

    # [add] New private method
    def _apply_miss(
        self,
        bboxes: torch.Tensor,             # (N, C, T, 4)
        bbox_empty: torch.Tensor,          # (N, C, T)
        target_size: torch.Tensor,         # (N, C, T)
        bg_is_ground: torch.Tensor,        # (N, C, T) bool
        fp_fn_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply probabilistic miss. Returns (bboxes, bbox_empty) with misses zeroed."""

    # [add] New private method
    def _apply_fp(
        self,
        bboxes: torch.Tensor,             # (N, C, T, 4)
        bbox_empty: torch.Tensor,          # (N, C, T)
        fp_fn_scale: float,
        image_shapes: torch.Tensor,        # (N, C, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inject false positive bboxes. Returns (bboxes, bbox_empty) with FPs inserted."""


# [existing] — no change
def _sample_student_t(df, scale, shape, device) -> torch.Tensor: ...
```

#### 2. `bbox_raycaster_v2/bbox_raycaster_v2.py`

```python
class BBoxRayCasterV2:
    # [existing] __init__ — no change
    # [existing] update() — no change
    # [existing] setup_detector_replicator() — no change

    # [modify] Add fp_fn_scale param, add background classification
    def apply_detector_replicator(
        self,
        noise_scale: float = 1.0,
        fp_fn_scale: float = 1.0,   # [add]
    ):
        """Apply calibrated noise, miss rate, and FP injection.

        Classifies background (sky vs ground) via raycast_mesh on static_mesh,
        then delegates to DetectorReplicator.apply().
        """

    # [add] New private method
    def _classify_background(self) -> torch.Tensor:
        """Classify background as sky or ground for each camera-target pair.

        Casts a ray from camera through target position. If ray hits static_mesh,
        background is ground; if it misses, background is sky.

        Returns:
            bg_is_ground: (N, C, T) bool. True = ground background.
        """

    # [add] Idempotency guard field
    _last_replicator_time: float  # Set in __init__, checked in apply_detector_replicator
```

#### 3. `curriculum/curriculum_cfg.py`

```python
@configclass
class CurriculumCfg:
    # [existing fields] — no change

    # [add] FP/FN curriculum phase (co-located with noise)
    fp_fn_start_step: int = 100000
    """Step to start introducing false positives and miss rate."""

    fp_fn_end_step: int = 120000
    """Step when FP/FN rates reach calibrated values."""

    # [add] Convenience method
    def get_fp_fn_progress(self, current_step: int) -> float:
        """Get progress within FP/FN phase [0, 1]."""
        return self.get_progress(current_step, self.fp_fn_start_step, self.fp_fn_end_step)
```

#### 4. `iris_ma_env6_test.py`

```python
class IrisMA6TestEnv:
    # [modify] In _get_rewards() curriculum section:
    #   Add: fp_fn_progress = curriculum.get_fp_fn_progress(current_step)

    # [modify] In _update_state_cache():
    #   Change: apply_detector_replicator(noise_scale) →
    #           apply_detector_replicator(noise_scale, fp_fn_scale=fp_fn_progress)
```

#### 5. `iris_ma_env6_test_cfg.py`

```python
# [existing] — no structural change needed
# DetectorReplicatorCfg already imported and used
# apply_miss and apply_fp fields added to DetectorReplicatorCfg (item 1)
```

### New files

None. All changes extend existing files.

### Files NOT changed

- `bbox_raycaster_v2_data.py` — `bboxes_replicated`, `bboxes_xyxy_replicated`, `bbox_empty_replicated` already exist
- `bbox_raycaster_v2_cfg.py` — no new config needed at raycaster level
- `delay_system_v3/` — no changes; dual-path storage via `replicated_bboxes` kwarg already works
- `__init__.py` — no new exports needed
