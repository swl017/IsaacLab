# BBoxRayCasterV2 Configuration Reference

This document provides a complete reference for all configuration options in `BBoxRayCasterV2Cfg`.

## Configuration Dataclass

```python
from bbox_raycaster_v2 import BBoxRayCasterV2Cfg

cfg = BBoxRayCasterV2Cfg(
    target_prim_paths=["/World/envs/env_.*/target"],
    mesh_prim_paths=["/World/ground"],
    num_cameras_per_env=3,
    enable_occlusion_check=True,
    # ... additional options
)
```

## Required Parameters

### target_prim_paths
```python
target_prim_paths: list[str]
```
**Required.** List of USD prim paths for targets to detect.

Example:
```python
target_prim_paths=["/World/envs/env_.*/target"]
# Matches: /World/envs/env_0/target, /World/envs/env_1/target, ...
```

### mesh_prim_paths
```python
mesh_prim_paths: list[str]
```
**Required.** List of mesh paths for static environment occlusion (ground, obstacles).

Example:
```python
mesh_prim_paths=["/World/ground", "/World/envs/env_.*/obstacles"]
```

### num_cameras_per_env
```python
num_cameras_per_env: int
```
**Required.** Number of cameras per environment (typically equals num_agents).

---

## Agent Mesh Configuration

### load_agent_meshes
```python
load_agent_meshes: bool = True
```
Whether to load agent meshes for drone-to-drone occlusion detection.

When enabled:
- Loads each agent's mesh in body-local coordinates
- Tests for occlusions between agents (one drone blocking another's view)
- Includes ALL mesh components (body, propellers, arms)

### agent_mesh_simplification
```python
agent_mesh_simplification: float = 1.0
```
Mesh simplification factor. Range: [0.0, 1.0]

| Value | Effect |
|-------|--------|
| 1.0 | No simplification (most accurate, slowest) |
| 0.5 | Keep ~50% of triangles |
| 0.1 | Keep ~10% of triangles (fastest) |

### use_collision_proxy
```python
use_collision_proxy: bool = False
```
Use simple box collision proxies instead of full meshes.

- **True**: Uses axis-aligned bounding boxes (0.5m x 0.5m x 0.2m)
- **False**: Uses detailed mesh geometry

Useful when mesh loading fails or for debugging.

---

## Camera Configuration

### num_cameras_per_agent
```python
num_cameras_per_agent: int = 1
```
Number of cameras per agent. Usually 1 for single-camera drones.

---

## Validation Thresholds

### min_bbox_size
```python
min_bbox_size: tuple[float, float] = (0.01, 0.01)
```
Minimum bbox size as fraction of image dimensions.

Format: `(width_fraction, height_fraction)`

Example: `(0.01, 0.01)` = minimum 1% of image width/height

### max_bbox_size
```python
max_bbox_size: tuple[float, float] = (0.95, 0.95)
```
Maximum bbox size as fraction of image dimensions.

Bboxes larger than this are marked invalid (target too close).

### partial_detection_allowed
```python
partial_detection_allowed: bool = False
```
Whether to allow partial detections (some corners out of FOV).

- **False**: All 8 bbox corners must be visible
- **True**: Bboxes computed even with some corners outside FOV

### min_bbox_area_pixels
```python
min_bbox_area_pixels: float = 4.0
```
Minimum bbox area in pixels to be valid. Prevents degenerate bboxes.

---

## Occlusion Detection

### enable_occlusion_check
```python
enable_occlusion_check: bool = True
```
Whether to perform raycasting-based occlusion checking.

Detects occlusions from:
1. Static environment (ground, obstacles)
2. Dynamic agents (other drones, if `load_agent_meshes=True`)

### occlusion_ray_pattern
```python
occlusion_ray_pattern: Literal["9point", "corners_only", "center_only"] = "center_only"
```
Pattern for occlusion test points on target surface.

| Pattern | Points | Description |
|---------|--------|-------------|
| `"center_only"` | 1 | Center point only (fastest) |
| `"corners_only"` | 4 | 4 bottom corners |
| `"9point"` | 9 | 4 corners + 4 edge midpoints + center (most accurate) |

**Recommendation**: Use `"9point"` for robust occlusion detection.

### occlusion_visibility_threshold
```python
occlusion_visibility_threshold: float = 0.5
```
Fraction of test points that must be visible.

With `"9point"` pattern and threshold `0.5`: at least 5/9 points must be visible.

Lower values = more sensitive to occlusion (stricter).

### occlusion_ray_tolerance
```python
occlusion_ray_tolerance: float = 1.1
```
Tolerance multiplier for ray-target intersection.

- Multiplies target's bbox radius
- Larger values = more permissive (fewer false occlusions)
- Recommended: 1.1 - 1.3

### enable_self_occlusion
```python
enable_self_occlusion: bool = True
```
Whether the observing agent's own body can occlude camera view.

When enabled:
- Camera's own drone body is checked for occlusion
- Uses `self_occlusion_min_hit_distance_m` to avoid false positives

### self_occlusion_min_hit_distance_m
```python
self_occlusion_min_hit_distance_m: float = 0.05
```
Ignore self-hits closer than this distance (meters).

Prevents camera mount false positives where rays hit very close geometry.

---

## Inter-Target Occlusion (Image-Space)

### enable_inter_target_occlusion
```python
enable_inter_target_occlusion: bool = True
```
Apply image-space IoU-based occlusion after geometric detection.

When two targets overlap in 2D:
- Computes IoU (Intersection over Union)
- Applies soft/hard thresholds based on depth

### inter_target_iou_soft_threshold
```python
inter_target_iou_soft_threshold: float = 0.10
```
IoU threshold where soft confidence decay begins.

When IoU > soft_threshold: confidence linearly decreases.

### inter_target_iou_hard_threshold
```python
inter_target_iou_hard_threshold: float = 0.30
```
IoU threshold where farther target becomes empty.

When IoU > hard_threshold: farther target is marked as `bbox_empty=True`.

### inter_target_depth_margin_m
```python
inter_target_depth_margin_m: float = 1.0
```
Minimum depth separation required before one target can occlude another.

If two targets are within this margin, neither occludes the other.

---

## Performance Options

### max_distance
```python
max_distance: float = 100.0
```
Maximum raycasting distance in meters.

Targets beyond this distance skip occlusion checking. Set based on environment size.

---

## Numerical Stability

### projection_epsilon
```python
projection_epsilon: float = 1e-6
```
Epsilon for preventing division by zero in projections.

### quat_normalize_epsilon
```python
quat_normalize_epsilon: float = 1e-8
```
Epsilon for quaternion normalization.

### gimbal_lock_threshold
```python
gimbal_lock_threshold: float = 0.99
```
Threshold for detecting gimbal lock (z-axis verticality).

### warn_gimbal_lock
```python
warn_gimbal_lock: bool = False
```
Whether to warn when gimbal lock is detected.

---

## Debug Options

### debug_vis
```python
debug_vis: bool = False
```
Enable debug visualization in viewport.

Shows bounding boxes, rays, and other debug information.

### debug_vis_corners
```python
debug_vis_corners: bool = False
```
Visualize projected bbox corners (requires `debug_vis=True`).

### debug_vis_occlusion_rays
```python
debug_vis_occlusion_rays: bool = False
```
Visualize occlusion rays and hit points (requires `debug_vis=True`).

### debug_memory
```python
debug_memory: bool = False
```
Track and print GPU memory usage at various pipeline stages.

---

## Example Configurations

### Training Configuration (Recommended)
```python
bbox_raycaster_v2 = BBoxRayCasterV2Cfg(
    target_prim_paths=["/World/envs/env_.*/target"],
    mesh_prim_paths=["/World/ground"],
    num_cameras_per_env=3,

    # Agent mesh - full occlusion detection
    load_agent_meshes=True,
    enable_occlusion_check=True,
    enable_self_occlusion=True,
    enable_inter_target_occlusion=True,

    # Robust occlusion with 9 test points
    occlusion_ray_pattern="9point",
    occlusion_visibility_threshold=0.3,  # 30% points visible
    occlusion_ray_tolerance=1.2,
    self_occlusion_min_hit_distance_m=0.05,

    # Validation
    min_bbox_size=(0.01, 0.01),
    max_bbox_size=(0.95, 0.95),
    partial_detection_allowed=False,
    min_bbox_area_pixels=4.0,

    # Performance
    max_distance=100.0,

    # Debug off for training
    debug_vis=False,
    debug_memory=False,
)
```

### Debug Configuration
```python
bbox_raycaster_v2 = BBoxRayCasterV2Cfg(
    target_prim_paths=["/World/envs/env_.*/target"],
    mesh_prim_paths=["/World/ground"],
    num_cameras_per_env=3,

    # Same as training...
    load_agent_meshes=True,
    enable_occlusion_check=True,

    # Debug visualization ON
    debug_vis=True,
    debug_vis_corners=True,
    debug_vis_occlusion_rays=True,
    debug_memory=True,
)
```

### Fast Configuration (Reduced Accuracy)
```python
bbox_raycaster_v2 = BBoxRayCasterV2Cfg(
    target_prim_paths=["/World/envs/env_.*/target"],
    mesh_prim_paths=["/World/ground"],
    num_cameras_per_env=3,

    # Simplified meshes
    load_agent_meshes=True,
    agent_mesh_simplification=0.3,  # Keep 30% triangles

    # Faster occlusion (less accurate)
    occlusion_ray_pattern="center_only",  # 1 test point
    enable_self_occlusion=False,
    enable_inter_target_occlusion=False,

    # Relaxed validation
    partial_detection_allowed=True,
)
```

### No Occlusion Configuration
```python
bbox_raycaster_v2 = BBoxRayCasterV2Cfg(
    target_prim_paths=["/World/envs/env_.*/target"],
    mesh_prim_paths=[],  # No meshes needed
    num_cameras_per_env=3,

    # Disable all occlusion
    load_agent_meshes=False,
    enable_occlusion_check=False,
    enable_self_occlusion=False,
    enable_inter_target_occlusion=False,
)
```
