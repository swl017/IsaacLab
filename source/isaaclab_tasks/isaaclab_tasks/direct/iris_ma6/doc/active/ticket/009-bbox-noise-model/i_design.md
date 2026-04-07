## Design Document: Calibrated Bbox Noise Model

### Problem statement

The bbox noise in training is a fixed 7.0 px Gaussian applied uniformly to center and size,
regardless of target size, distance, or detector characteristics. Offline YOLO calibration
(completed) shows: (1) noise is size-dependent — small targets are noisier, (2) the error
distribution is Student's t (df~12-16), not Gaussian — heavier tails, rounded peak,
(3) the actual scale is ~13.4 px (nearly double the hand-tuned value), (4) systematic bias
of ~1 px exists. The current noise model underestimates both magnitude and tail weight, and
ignores size dependence entirely.

### Proposed approach

Replace the fixed-scalar Gaussian bbox noise with a calibrated Student's t noise model
whose scale varies with target bbox size. The calibration parameters are loaded from a JSON
file produced by the offline calibration script (`experiments/calibrate_bbox_noise.py`,
already complete).

**Where noise is applied**: Inside a new `detector_replicator` submodule in `bbox_raycaster_v2`.
This is the correct location because: (a) target bbox size is directly available,
(b) noise and miss rate (ticket-010) must be co-located — missed detections should not have
noise applied, (c) the replicator is the single architectural home for all detector modeling
(noise + FN + FP), (d) it decouples detector noise from communication delay modeling.

**Dual output**: After `update()`, `bbox_raycaster_v2` produces two sets of bboxes:
- **GT bboxes** (`data.bboxes`): pure ground-truth from raycasting, unchanged
- **Replicated bboxes** (`data.bboxes_replicated`): GT + calibrated noise from `detector_replicator`

The env wires GT bboxes to the reward path (both privileged and perception-aligned) and
replicated bboxes to the observation path. This ensures rewards are never corrupted by
detector stochasticity — the policy is *guided* by physical truth but learns to *cope*
with noisy detections through its observations.

**Delay system bbox_std**: Set to 0.0 when calibrated noise is enabled. The delay system
no longer applies its own bbox noise. All other delay stages (latency, staleness, dropout)
remain unchanged.

**Why not post-delay noise** (previous "simpler alternative" — revoked): Applying noise in
`_get_observations()` on delayed-but-clean bboxes is incompatible with ticket-010's miss rate
injection. When a detection is missed (FN), the bbox is zeroed — applying noise post-delay
to a zeroed bbox produces a small noisy bbox instead of empty. Noise and miss rate must be
co-located at the detector replicator level, applied as a unit before delay system ingestion.

### Key interfaces and data flow

```
bbox_raycaster_v2.update(...)
    → data.bboxes              (N,C,T,4) pure GT         [unchanged]
    → data.bboxes_xyxy         (N,C,T,4) pure GT xyxy    [unchanged]
    → data.bbox_empty          (N,C,T)   occlusion flags  [unchanged]

bbox_raycaster_v2.apply_detector_replicator(params, noise_scale)
    → data.bboxes_replicated      (N,C,T,4) xywh with Student's t noise  [NEW]
    → data.bboxes_xyxy_replicated (N,C,T,4) xyxy with noise              [NEW]
    → data.bbox_empty_replicated  (N,C,T)   includes FN misses           [NEW, ticket-010]

Env _update_state_cache():
    raycaster.update(...)                                  # GT bboxes
    raycaster.apply_detector_replicator(params, scale)     # replicated bboxes

    for agent_id:
        # Reward path: GT bboxes → delay system gt store
        gt_data.bboxes_2d = raycaster.data.bboxes[:, idx, :, :]

        # Obs path: replicated bboxes → delay system obs store
        obs_data.bboxes_2d = raycaster.data.bboxes_replicated[:, idx, :, :]

    delay_system.update_ground_truth(...)  # stores both gt and obs bbox fields

Env _get_rewards():
    delay_system.get_states_for_rewards()  → uses GT bboxes (clean, delayed)

Env _get_observations():
    delay_system.get_states_for_observations()  → uses replicated bboxes (noisy, delayed)
```

### Configuration

```python
@dataclass
class CalibratedBBoxNoiseCfg:
    enabled: bool = False
    """Enable calibrated bbox noise (replaces delay system bbox_std)."""

    params_path: str = ""
    """Path to JSON from calibration script."""

    apply_bias: bool = True
    """Apply systematic center bias from calibration."""
```

Parameters loaded from JSON at env init, stored as a `NoiseModelParams` dataclass.
Curriculum control via existing `noise_scale` (0→1) multiplied into the Student's t
scale parameter.

### What this does NOT include

- Miss rate / false positive injection (ticket-010 — extends the detector replicator)
- Detection confidence in observations (deferred)
- Changes to the raycaster's geometric computation
- Changes to delay system pipeline stages (latency, staleness, dropout)
- Refitting the noise model (separate session — current `a≈0` will be corrected using per-bin scale values)
- Online YOLO calibration during training (rejected — TiledCamera can't subset envs)

### Open risks

1. **Model fit quality**: Current fit shows `a≈0` (size-independent) which contradicts
   the visual data. The design accommodates size-dependent noise (`a/size + b`), but
   the actual parameters need refitting with a wider size range or different binning.
   The binned data from findings.md shows clear trend: ~20px noise at 10px targets,
   ~5px at 100px targets. Re-fit from per-bin scale values.
2. **Student's t sampling performance**: `torch.distributions.StudentT.sample()` may be
   slower than `torch.randn_like()`. Need to benchmark; if too slow, can use the
   Gaussian approximation for df>30 or a cached lookup table.
3. **Interaction with delay system dropout**: When dropout holds a stale bbox, calibrated
   noise was already applied to the original frame. On the next non-dropped frame, fresh
   noise is applied. This is correct behavior (each detection has independent noise).
4. **Dual bbox storage**: The delay system needs to store both GT and replicated bboxes,
   or the env needs to maintain separate data paths. The exact mechanism (dual fields
   in AgentStates vs. separate store calls) is deferred to Stage S.
