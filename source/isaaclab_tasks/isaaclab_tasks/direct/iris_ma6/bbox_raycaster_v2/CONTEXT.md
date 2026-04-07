# BBox RayCaster V2 Module

## Purpose
GPU-accelerated 2D bounding box extraction using Warp-based raycasting.
Computes pixel-space bounding boxes of targets as seen from each camera,
with built-in occlusion handling via ray intersection tests.

Includes a **detector replicator** submodule that applies calibrated noise
(Student's t, size-dependent scale) to replicate real YOLO detector error
profiles. Produces dual outputs: clean GT bboxes and replicated (noisy) bboxes.

## Inputs
- Camera poses: position and orientation per camera
- Camera intrinsics: focal length, resolution, principal point
- Target poses: position and size of each target
- Agent poses: positions of other agents (for occlusion checks)
- Calibration JSON: noise model parameters from offline YOLO calibration (for replicator)

## Outputs
- `BBoxRayCasterV2Data` dataclass containing:
  - Bounding boxes: `(N, C, T, 4)` as [center_x, center_y, width, height] (xywh) and [x_min, y_min, x_max, y_max] (xyxy)
  - Empty flag: `(N, C, T)` boolean (True if target not visible)
  - Replicated bboxes: `(N, C, T, 4)` with calibrated detector noise (when replicator enabled)
  - Replicated empty flag: `(N, C, T)` (same as GT for ticket-009; ticket-010 adds FN misses)

## Dependencies
None (standalone module). Requires NVIDIA Warp for GPU raycasting.

## Key Files
- `bbox_raycaster_v2.py` - Core raycasting, bbox extraction, and detector replicator integration
- `bbox_raycaster_v2_cfg.py` - Configuration (ray density, max range)
- `bbox_raycaster_v2_data.py` - Output data container (GT + replicated fields)
- `detector_replicator.py` - Calibrated noise: NoiseModelParams, DetectorReplicatorCfg, DetectorReplicator
- `utils/` - Helper utilities for ray generation and intersection
- `doc/` - Additional documentation

## Calling Contract
- `update()`: **WRITE**. Call once per step from `_update_state_cache()`. Computes GT bboxes.
- `apply_detector_replicator(noise_scale)`: **WRITE**. Call once per step after `update()`.
  Produces replicated bboxes. Safe to skip (replicated fields will be None).
- `data.bboxes` / `data.bboxes_xyxy`: **READ**. Clean GT, safe to read any time after `update()`.
- `data.bboxes_replicated` / `data.bboxes_xyxy_replicated`: **READ**. Noisy, safe after `apply_detector_replicator()`.

## Spec
`doc/bbox_spec.md` (if available in `doc/` directory).
