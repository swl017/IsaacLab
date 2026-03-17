# Occlusion Detection Pipeline

This document describes the occlusion detection algorithms implemented in BBoxRayCasterV2.

## Overview

The occlusion detection system determines when targets are hidden from camera view by:
1. **Static environment** (ground plane, obstacles)
2. **Other agents** (drones blocking each other's view)
3. **Self-occlusion** (camera's own drone body)
4. **Inter-target occlusion** (one target hiding another in image space)

## Occlusion Detection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Occlusion Detection Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ Generate Test    │  K test points on target surface          │
│  │ Points (N,T,K,3) │  Pattern: center_only, corners_only, 9pt  │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Static Mesh Raycast (World Frame)                     │   │
│  │                                                          │   │
│  │    For each ray (camera → test_point):                   │   │
│  │    • Cast ray in world coordinates                       │   │
│  │    • If hit_distance < target_distance: OCCLUDED         │   │
│  │                                                          │   │
│  │    Complexity: O(1) raycast call                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. Agent Mesh Raycast (Body-Local Frame)                 │   │
│  │                                                          │   │
│  │    For each agent mesh:                                  │   │
│  │    • Transform ALL rays to agent's body frame            │   │
│  │    • Cast rays against body-local mesh                   │   │
│  │    • Transform hits back, check if occluding             │   │
│  │                                                          │   │
│  │    Complexity: O(num_agents) raycast calls               │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. Compute Visibility Ratio                              │   │
│  │                                                          │   │
│  │    visibility_ratio = visible_points / total_points      │   │
│  │    visibility_mask = ratio >= threshold                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. Inter-Target Occlusion (Image-Space IoU)              │   │
│  │                                                          │   │
│  │    For overlapping bboxes in 2D:                         │   │
│  │    • Compute IoU (Intersection over Union)               │   │
│  │    • If IoU > threshold AND depth difference:            │   │
│  │      - Soft: Reduce confidence                           │   │
│  │      - Hard: Mark farther target as empty                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Test Point Generation

Test points are generated on the target's bounding box surface:

### Pattern: `center_only` (K=1)
```
    +-------+
    |       |
    |   ●   |  ← Center point only
    |       |
    +-------+
```

### Pattern: `corners_only` (K=4)
```
    ●-------●
    |       |
    |       |  ← 4 bottom corners
    |       |
    ●-------●
```

### Pattern: `9point` (K=9)
```
    ●---●---●
    |       |
    ●   ●   ●  ← 4 corners + 4 edge midpoints + center
    |       |
    ●---●---●
```

**Recommendation**: Use `9point` for robust occlusion detection. The `center_only` pattern can miss partial occlusions and cause oscillation.

---

## Static Mesh Raycasting

Static meshes (ground plane, obstacles) are tested in world coordinates.

### Algorithm

```python
def _raycast_static_fully_batched(
    camera_pos,      # (N, C, T, K, 3) - expanded camera positions
    ray_dirs,        # (N, C, T, K, 3) - ray directions (camera → target)
    static_mesh,     # wp.Mesh in world frame
    cam_to_target,   # (N, C, T, K) - distance from camera to target
    target_radius,   # (N, 1, T, 1) - target bbox radius for tolerance
    max_distance
):
    # Flatten all rays into one batch
    batch_size = N * C * T * K
    ray_starts_flat = camera_pos.reshape(batch_size, 3)
    ray_dirs_flat = ray_dirs.reshape(batch_size, 3)

    # Single massive raycast
    ray_hits_flat = raycast_mesh(
        ray_starts_flat, ray_dirs_flat,
        mesh=static_mesh,
        max_dist=max_distance
    )

    # Unflatten results
    ray_hits = ray_hits_flat.view(N, C, T, K, 3)

    # Check if hits are in front of target
    valid_hit = ~(isinf(ray_hits) | isnan(ray_hits))
    hit_distance = norm(ray_hits - camera_pos)
    min_clear_distance = cam_to_target - target_radius

    # Occluded if hit is closer than target
    occluded = valid_hit & (hit_distance < min_clear_distance)

    return occluded  # (N, C, T, K)
```

---

## Agent Mesh Raycasting (Body-Local)

Agent meshes are stored in body-local coordinates for efficient batched processing.

### Key Insight

Since all environments share the same agent mesh geometry:
- **Traditional**: Transform mesh to world frame for each environment → O(N * mesh_vertices)
- **Body-local**: Transform rays to body frame → O(N * num_rays)

For large N (thousands of environments), this is dramatically faster.

### Algorithm

```python
def _raycast_agent_fully_batched(
    camera_pos_exp,   # (N, C, T, K, 3)
    ray_dirs_w,       # (N, C, T, K, 3) - world frame directions
    cam_to_target,    # (N, C, T, K)
    target_radius,    # (N, 1, T, 1)
    agent_pos,        # (N, 3) - agent body position
    agent_quat,       # (N, 4) - agent body orientation
    agent_mesh,       # wp.Mesh in body-local frame
    test_camera_mask, # (C,) - which cameras to test
    self_camera_mask, # (C,) - which cameras belong to this agent
    max_distance,
    self_occlusion_min_hit_distance_m
):
    occluded = zeros((N, C, T, K), bool)

    # Compute inverse quaternion for world → body transform
    agent_quat_inv = quat_inv(agent_quat)  # (N, 4)

    for cam_idx in cameras_to_test:
        # Get data for this camera across ALL environments
        cam_pos_w = camera_pos_exp[:, cam_idx, 0, 0, :]  # (N, 3)
        ray_dirs_w_cam = ray_dirs_w[:, cam_idx]          # (N, T, K, 3)

        # === Transform camera position to body frame ===
        cam_rel = cam_pos_w - agent_pos  # (N, 3)
        cam_body = quat_apply(agent_quat_inv, cam_rel)  # (N, 3)

        # === Transform ray directions to body frame ===
        ray_dirs_flat = ray_dirs_w_cam.reshape(N * T * K, 3)
        quat_inv_flat = agent_quat_inv.unsqueeze(1).unsqueeze(1).expand(-1, T, K, -1)
        quat_inv_flat = quat_inv_flat.reshape(N * T * K, 4)
        ray_dirs_body_flat = quat_apply(quat_inv_flat, ray_dirs_flat)

        # === Single raycast for ALL environments ===
        ray_starts = cam_body.expand(T * K, -1, -1).reshape(N * T * K, 3)
        ray_hits_body = raycast_mesh(ray_starts, ray_dirs_body_flat, agent_mesh)

        # Check occlusions
        hit_distance = norm(ray_hits_body - cam_body_exp)
        min_clear_dist = cam_to_target - target_radius

        occluded_this_cam = valid_hit & (hit_distance < min_clear_dist)

        # Self-occlusion: ignore hits too close (camera mount)
        if is_self_camera[cam_idx]:
            occluded_this_cam &= hit_distance >= self_occlusion_min_hit_distance_m

        occluded[:, cam_idx] = occluded_this_cam

    return occluded
```

### Self-Occlusion Handling

When the camera belongs to the same agent as the mesh being tested:

1. **Enable check**: `enable_self_occlusion=True`
2. **Minimum distance filter**: `self_occlusion_min_hit_distance_m=0.05`
3. **Logic**: Ignore hits closer than 5cm to avoid camera mount false positives

```
Camera mounted on drone body:

    ┌─────────────────┐
    │     Drone       │
    │   ┌─────┐       │
    │   │ Cam │ ← Rays from here
    │   └─────┘       │
    │                 │
    └─────────────────┘
           │
    ┌──────┴──────┐
    │  Camera     │
    │  Mount      │ ← Hits here ignored (< 5cm)
    └─────────────┘
           │
           ▼
    Propeller arm ← Hits here count (> 5cm)
```

---

## Inter-Target Occlusion (Image-Space)

After geometric raycasting, inter-target occlusion handles cases where:
- Two targets overlap in 2D projection
- One target is closer than the other

### Algorithm

```python
def apply_inter_target_occlusion(
    bboxes_xyxy,        # (N, C, T, 4)
    bbox_empty,         # (N, C, T)
    bbox_confidence,    # (N, C, T)
    camera_pos_w,       # (N, C, 3)
    target_pos_w,       # (N, T, 3)
    soft_iou_threshold,  # 0.10
    hard_iou_threshold,  # 0.30
    depth_margin_m       # 1.0
):
    # For each pair of targets (i, j):
    for i in range(T):
        for j in range(i+1, T):
            # Compute IoU in image space
            iou = compute_iou(bboxes_xyxy[:, :, i], bboxes_xyxy[:, :, j])

            # Compute depths
            depth_i = distance(camera_pos_w, target_pos_w[:, i])
            depth_j = distance(camera_pos_w, target_pos_w[:, j])

            # Determine which is closer
            closer_is_i = depth_i < depth_j
            depth_diff = abs(depth_i - depth_j)

            # Apply occlusion only if sufficient depth separation
            apply_mask = depth_diff > depth_margin_m

            # Soft threshold: reduce confidence
            if iou > soft_iou_threshold:
                confidence_reduction = (iou - soft_iou) / (hard_iou - soft_iou)
                farther_confidence *= (1 - confidence_reduction)

            # Hard threshold: mark as empty
            if iou > hard_iou_threshold:
                farther_bbox_empty = True
```

### IoU Calculation

```
    Target A (closer)          Target B (farther)
    ┌─────────────┐
    │             │
    │     ┌───────┼─────┐
    │     │ INTER │     │
    │     │ SECT  │     │
    └─────┼───────┘     │
          │             │
          └─────────────┘

IoU = Intersection Area / Union Area
    = |A ∩ B| / |A ∪ B|
```

---

## Visibility Ratio and Threshold

After all occlusion checks, the visibility ratio determines final detection status:

```python
# Count visible test points
visible_points = ~occluded_by_static & ~occluded_by_agents  # (N, C, T, K)
visibility_ratio = visible_points.float().mean(dim=-1)       # (N, C, T)

# Apply threshold
visibility_mask = visibility_ratio >= occlusion_visibility_threshold

# Example with 9-point pattern, threshold=0.3:
# - 3/9 points visible → ratio=0.33 → VISIBLE
# - 2/9 points visible → ratio=0.22 → OCCLUDED
```

### Threshold Selection

| Threshold | Effect |
|-----------|--------|
| 0.1 | Very strict (almost any occlusion marks target as hidden) |
| 0.3 | Moderate (30% of target must be visible) |
| 0.5 | Balanced (50% must be visible) |
| 0.9 | Very permissive (only full occlusion hides target) |

**Recommendation**: Use `0.3` for most scenarios.

---

## Temporal Smoothing

To prevent oscillation at occlusion boundaries, the environment applies EMA (Exponential Moving Average) smoothing:

```python
# In environment's _update_state_cache():

# Get raw confidence from raycaster
raw_confidence = self.bbox_raycaster_v2.data.bbox_confidence  # (N, C, T)

# Apply EMA smoothing
alpha = 0.3  # Smoothing factor (lower = more smoothing)
self._smoothed_bbox_confidence = (
    alpha * raw_confidence +
    (1 - alpha) * self._smoothed_bbox_confidence
)

# Use smoothed value for empty decision
bbox_empty = self._smoothed_bbox_confidence < threshold  # e.g., 0.4
```

### Why Smoothing?

At occlusion boundaries, small movements cause rapid switching:

**Without smoothing:**
```
Frame 1: Target visible   → bbox_empty=False
Frame 2: Target occluded  → bbox_empty=True
Frame 3: Target visible   → bbox_empty=False
Frame 4: Target occluded  → bbox_empty=True
... (oscillation continues)
```

**With smoothing (α=0.3):**
```
Frame 1: raw=1.0, smoothed=1.0  → visible
Frame 2: raw=0.0, smoothed=0.7  → visible (smoothed still high)
Frame 3: raw=0.0, smoothed=0.49 → visible
Frame 4: raw=0.0, smoothed=0.34 → OCCLUDED (crossed threshold)
... (stable transition)
```

### Reset Behavior

On environment reset, the smoothed confidence is reset to 1.0:

```python
def _reset_idx(self, env_ids):
    self._smoothed_bbox_confidence[env_ids] = 1.0
```

This ensures new episodes start with full visibility (no stale occlusion state).

---

## Performance Characteristics

### Raycasting Complexity

| Component | Raycast Calls | Total Rays |
|-----------|---------------|------------|
| Static mesh | 1 | N × C × T × K |
| Agent meshes | num_agents | N × C × T × K per agent |
| **Total** | 1 + num_agents | - |

### Typical Performance

With N=256, C=3, T=1, K=9, num_agents=3:

| Operation | Time |
|-----------|------|
| Static raycast | ~0.5ms |
| Agent raycasts (3x) | ~1.5ms |
| Visibility computation | ~0.1ms |
| Inter-target occlusion | ~0.1ms |
| **Total** | ~2.2ms |

### Memory Usage

| Buffer | Shape | Size (fp32) |
|--------|-------|-------------|
| test_points | (N, T, K, 3) | N × T × K × 12 bytes |
| ray_hits | (N, C, T, K, 3) | N × C × T × K × 12 bytes |
| visibility_ratio | (N, C, T) | N × C × T × 4 bytes |

For N=256, C=3, T=1, K=9: ~112 KB total

---

## Troubleshooting

### Target Always Occluded

1. Check if `enable_occlusion_check=True`
2. Verify mesh paths are correct
3. Check `occlusion_visibility_threshold` (lower values are stricter)
4. Enable `debug_vis_occlusion_rays=True` to visualize

### Target Never Occluded

1. Verify mesh is loaded (check logs for "Loaded mesh...")
2. Check `max_distance` is large enough
3. Verify agent meshes are in body-local coordinates
4. Check `occlusion_ray_pattern` (center_only may miss partial occlusions)

### Oscillation Between Detected/Occluded

1. Increase `occlusion_ray_pattern` to `"9point"`
2. Decrease `occlusion_visibility_threshold` to 0.3
3. Enable temporal smoothing in environment (EMA filter)
4. Increase `occlusion_ray_tolerance` to 1.2+

### Self-Occlusion False Positives

1. Increase `self_occlusion_min_hit_distance_m` (default 0.05m)
2. Check camera mount geometry in USD
3. Verify camera is not inside mesh geometry

### Performance Issues

1. Reduce `agent_mesh_simplification` to 0.3-0.5
2. Use `occlusion_ray_pattern="center_only"` if accuracy allows
3. Disable `enable_inter_target_occlusion` if not needed
4. Reduce `max_distance` to match environment size
