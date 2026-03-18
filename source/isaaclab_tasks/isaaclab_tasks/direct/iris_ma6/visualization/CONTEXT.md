# Visualization Module

## Purpose
Debug visualization for the multi-agent surveillance environment.
Renders camera frustums, detection indicators, triangulation covariance
ellipsoids, and coordinate frames in Isaac Sim's viewport.

## Inputs
- Camera poses and intrinsics (for frustum rendering)
- Target positions and `bbox_empty` flags (for detection indicators)
- Covariance matrices `(N, T, 3, 3)` (for uncertainty ellipsoids)
- Zoom levels (for frustum FOV adjustment)
- Coordinate frames (for axis visualization)

## Outputs
- Isaac Sim visualization markers (rendered in viewport, no tensor output)

## Dependencies
None (standalone module). Uses Isaac Sim marker/debug-draw APIs.

## Key Files
- `custom_visualization.py` - Top-level visualization manager
- `camera_frustum.py` - Camera frustum wireframe rendering
- `detection_indicator.py` - Target detection status markers
- `covariance_ellipsoid.py` - 3D covariance ellipsoid rendering
- `frame_visualizer.py` - Coordinate frame axis rendering
- `tests/` - Visualization tests

## Spec
`doc/visualization_spec.md`
