# BBox RayCaster V2 Module

## Purpose
GPU-accelerated 2D bounding box extraction using Warp-based raycasting.
Computes pixel-space bounding boxes of targets as seen from each camera,
with built-in occlusion handling via ray intersection tests.

## Inputs
- Camera poses: position and orientation per camera
- Camera intrinsics: focal length, resolution, principal point
- Target poses: position and size of each target
- Agent poses: positions of other agents (for occlusion checks)

## Outputs
- `BBoxRayCasterV2Data` dataclass containing:
  - Bounding boxes: `(N, C, T, 4)` as [u_min, v_min, u_max, v_max]
  - Empty flag: `(N, C, T)` boolean (True if target not visible)

## Dependencies
None (standalone module). Requires NVIDIA Warp for GPU raycasting.

## Key Files
- `bbox_raycaster_v2.py` - Core raycasting and bbox extraction logic
- `bbox_raycaster_v2_cfg.py` - Configuration (ray density, max range)
- `bbox_raycaster_v2_data.py` - Output data container
- `utils/` - Helper utilities for ray generation and intersection
- `doc/` - Additional documentation

## Spec
`doc/bbox_spec.md` (if available in `doc/` directory).
