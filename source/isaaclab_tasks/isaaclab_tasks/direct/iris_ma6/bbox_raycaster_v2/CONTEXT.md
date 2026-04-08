# BBox RayCaster V2 Module

## Purpose
GPU-accelerated 2D bounding box extraction using Warp-based raycasting.
Computes pixel-space bounding boxes of targets as seen from each camera,
with built-in occlusion handling via ray intersection tests.

Includes a **detector replicator** submodule that applies calibrated noise
(Student's t, size-dependent scale), probabilistic miss rate (FN, conditioned on
target size and sky/ground background), and false positive injection to replicate
real YOLO detector behavior. Produces dual outputs: clean GT bboxes (for rewards)
and replicated bboxes (noisy + FN + FP, for observations).

## Inputs
- Camera poses: position and orientation per camera
- Camera intrinsics: focal length, resolution, principal point
- Target poses: position and size of each target
- Agent poses: positions of other agents (for occlusion checks)
- Noise model params: baked into NoiseModelParams dataclass defaults (calibrated via experiments/calibrate_detector.py)

## Outputs
- `BBoxRayCasterV2Data` dataclass containing:
  - Bounding boxes: `(N, C, T, 4)` as [center_x, center_y, width, height] (xywh) and [x_min, y_min, x_max, y_max] (xyxy)
  - Empty flag: `(N, C, T)` boolean (True if target not visible)
  - Replicated bboxes: `(N, C, T, 4)` with noise + FN + FP (when replicator enabled)
  - Replicated empty flag: `(N, C, T)` includes FN misses (differs from GT empty)
  - Background classification: `bg_is_ground` `(N, C, T)` bool. True = ground/terrain behind target, False = sky. Uses ray direction (Z < 0) + static mesh raycast.

## Dependencies
None (standalone module). Requires NVIDIA Warp for GPU raycasting.

## Key Files
- `bbox_raycaster_v2.py` - Core raycasting, bbox extraction, and detector replicator integration
- `bbox_raycaster_v2_cfg.py` - Configuration (ray density, max range)
- `bbox_raycaster_v2_data.py` - Output data container (GT + replicated fields)
- `detector_replicator.py` - Detector modeling: NoiseModelParams (noise + miss + FP params), DetectorReplicatorCfg, DetectorReplicator (noise→miss→FP→zero-out pipeline)
- `utils/` - Helper utilities for ray generation and intersection
- `doc/` - Additional documentation

## Calling Contract
- `update()`: **WRITE**. Call once per step from `_update_state_cache()`. Computes GT bboxes.
- `apply_detector_replicator(noise_scale, fp_fn_scale, sim_time)`: **WRITE**. Call once per step
  after `update()`. Applies noise→miss→FP→zero-out. Idempotent within same sim_time via guard.
  Classifies background (sky/ground) via raycast_mesh on static_mesh for miss rate conditioning.
- `data.bboxes` / `data.bboxes_xyxy`: **READ**. Clean GT, safe to read any time after `update()`.
- `data.bboxes_replicated` / `data.bboxes_xyxy_replicated`: **READ**. Noisy+FN+FP, safe after `apply_detector_replicator()`.
- `data.bbox_empty_replicated`: **READ**. Differs from `bbox_empty` when FN misses are active.
- `data.bg_is_ground`: **READ**. Background classification, safe after `apply_detector_replicator()`.

## Spec
`doc/bbox_spec.md` (if available in `doc/` directory).
