# BBoxRayCasterV2 Architecture

This document describes the architecture, data flow, and key algorithms of the BBoxRayCasterV2 module.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BBoxRayCasterV2                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Camera     │    │   Target     │    │    Agent     │    │   Static   │ │
│  │   Poses      │    │   Poses      │    │   Meshes     │    │   Mesh     │ │
│  │  (N, C, 7)   │    │  (N, T, 7)   │    │  (body-local)│    │  (world)   │ │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └─────┬──────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   │                   │       │
│  ┌─────────────────────────────────────────────┴───────────────────┴──────┐ │
│  │                         update() Pipeline                              │ │
│  │                                                                        │ │
│  │  1. Stack camera data → (N, C, 3), (N, C, 4), (N, C, 3, 3)            │ │
│  │  2. Normalize quaternions                                              │ │
│  │  3. Transform bbox corners: local → world frame                        │ │
│  │  4. Transform corners: world → camera frame                            │ │
│  │  5. Project to image plane: 3D → 2D pixels                            │ │
│  │  6. Compute 2D bboxes from projected corners                          │ │
│  │  7. Validate detections (size, FOV, corners)                          │ │
│  │  8. Check occlusions (static + agent meshes)                          │ │
│  │  9. Apply inter-target occlusion (image-space IoU)                    │ │
│  │ 10. Sync outputs to data container                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌──────────────────┐                               │
│                          │ BBoxRayCasterV2  │                               │
│                          │      Data        │                               │
│                          │                  │                               │
│                          │ • bboxes (N,C,T,4)│                              │
│                          │ • bbox_empty     │                               │
│                          │ • bbox_confidence│                               │
│                          │ • visibility_ratio│                              │
│                          └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Initialization

During `__init__`, the raycaster:

1. **Loads static mesh** from USD scene (ground plane, obstacles)
2. **Loads agent meshes** in body-local coordinates for each drone
3. **Extracts target bboxes** from USD prims (one-time extraction)
4. **Allocates buffers** for all intermediate and output tensors

```python
# Mesh loading hierarchy
Robot_0/
├── body/           # Main fuselage mesh
│   └── mesh
├── prop_0/         # Propeller meshes
│   └── mesh
├── prop_1/
│   └── mesh
├── prop_2/
│   └── mesh
└── prop_3/
    └── mesh

# All meshes combined into single warp.Mesh in body-local coordinates
```

### 2. Update Pipeline

Each `update()` call processes a frame in 10 steps:

#### Step 1-2: Input Processing
```python
# Stack camera data from dict to batched tensors
camera_pos: (N, C, 3)      # World positions
camera_quat: (N, C, 4)     # Quaternions (w,x,y,z)
intrinsics: (N, C, 3, 3)   # Camera matrices

# Normalize quaternions for numerical stability
quat_normalized = quat / norm(quat)
```

#### Step 3: World Frame Transform
```python
# Transform target bbox corners from local to world frame
corners_local: (8, 3) or (N, 8, 3)  # 8 corners of 3D bbox
corners_world: (N, T, 8, 3)         # After rotation + translation
```

#### Step 4: Camera Frame Transform
```python
# Transform corners from world to each camera's frame
corners_camera: (N, C, T, 8, 3)

# For each camera:
#   1. Translate: corners - camera_pos
#   2. Rotate: quat_inv(camera_quat) * relative_pos
```

#### Step 5: Image Projection
```python
# Project 3D camera-frame points to 2D pixels
pixels: (N, C, T, 8, 2)    # 2D pixel coordinates
depths: (N, C, T, 8)       # Z-depth of each corner

# Projection: pixel = K @ (point / point.z)
# where K is the 3x3 intrinsic matrix
```

#### Step 6: Bbox Computation
```python
# Compute 2D bbox from projected corners
bboxes_xyxy: (N, C, T, 4)  # (x_min, y_min, x_max, y_max)
bboxes_xywh: (N, C, T, 4)  # (center_x, center_y, width, height)
bboxes_normalized: (N, C, T, 4)  # Values in [0, 1]
```

#### Step 7: Validation
```python
# Check various validity conditions
corners_ok: All 8 corners visible (or allow partial)
size_ok: min_size <= bbox <= max_size
area_ok: bbox_area >= min_bbox_area_pixels

valid_mask = corners_ok & size_ok & area_ok
```

#### Step 8: Occlusion Check (Raycasting)
```python
# Generate test points on target surface
test_points: (N, T, K, 3)  # K = 1, 4, or 9 points

# Cast rays from each camera to each test point
# Check for hits against:
#   1. Static mesh (ground, obstacles) - world frame
#   2. Agent meshes (other drones) - body-local frame

visibility_ratio: (N, C, T)  # Fraction of visible points
occluded = visibility_ratio < threshold
```

#### Step 9: Inter-Target Occlusion
```python
# Image-space IoU-based occlusion
# If two targets overlap in 2D and one is closer:
#   - Soft threshold: Reduce confidence
#   - Hard threshold: Mark farther target as empty
```

#### Step 10: Output Sync
```python
# Finalize outputs
bbox_empty = ~valid_mask
bboxes[bbox_empty] = 0  # Zero out invalid boxes
bbox_confidence = visibility_ratio * valid_mask
```

## Key Components

### BBoxRayCasterV2 (Main Class)

The main class orchestrates the entire pipeline:

```python
class BBoxRayCasterV2:
    """Batched bounding box raycaster for multi-agent environments."""

    # Configuration
    cfg: BBoxRayCasterV2Cfg
    num_envs: int
    num_cameras: int
    num_targets: int

    # Meshes for occlusion
    static_mesh: wp.Mesh        # Ground, obstacles (world frame)
    agent_meshes: Dict[str, wp.Mesh]  # Per-agent (body-local frame)

    # Pre-computed target data
    target_bbox_corners_local: torch.Tensor  # (8, 3) or (N, 8, 3)
    target_bbox_sizes: torch.Tensor          # (3,) or (N, 3)

    # Output container
    data: BBoxRayCasterV2Data
```

### BBoxRayCasterV2Data (Output Container)

Holds all output tensors:

```python
@dataclass
class BBoxRayCasterV2Data:
    # Core outputs
    bboxes: torch.Tensor          # (N, C, T, 4) - xywh format
    bboxes_xyxy: torch.Tensor     # (N, C, T, 4) - xyxy format
    bboxes_normalized: torch.Tensor  # (N, C, T, 4) - [0,1] range
    bbox_empty: torch.Tensor      # (N, C, T) - True if invalid
    bbox_confidence: torch.Tensor # (N, C, T) - [0,1] confidence

    # Occlusion data
    occlusion_visibility_ratio: torch.Tensor  # (N, C, T)
    self_occlusion_mask: torch.Tensor         # (N, C, T)
    occluded: torch.Tensor                    # (N, C, T)
```

### Occlusion Detection Module

The fully batched occlusion detection operates in body-local coordinates:

```python
def batch_check_occlusion_fully_batched(
    camera_pos, camera_quat,      # (N, C, 3), (N, C, 4)
    test_points_world,            # (N, T, K, 3)
    target_positions,             # (N, T, 3)
    target_bbox_size,             # (N, T, 3)
    agent_poses,                  # Dict[agent_id -> (pos, quat)]
    agent_meshes,                 # Dict[agent_id -> wp.Mesh]
    static_mesh,                  # wp.Mesh (world frame)
    ...
) -> (visibility_mask, visibility_ratio, self_occlusion_mask):

    # 1. Test against static environment (world frame)
    #    - Single raycast for ALL (N*C*T*K) rays

    # 2. Test against each agent mesh (body-local frame)
    #    For each agent:
    #      - Transform rays to agent's body frame
    #      - Single raycast for ALL rays
    #      - Check if hits are in front of target

    # Complexity: O(num_agents) raycasts instead of O(N*C*num_agents)
```

## Mesh Loading

### Agent Mesh Loading

Agent meshes are loaded from USD in body-local coordinates:

```python
def _load_agent_mesh(self, agent_id: str) -> wp.Mesh:
    # 1. Find ALL mesh prims under the robot (recursive)
    #    Includes: body, propellers, arms, etc.

    # 2. Get body transform as reference frame
    body_world_transform = get_world_transform_matrix(body_prim)

    # 3. For each mesh prim:
    #    a. Get mesh world transform
    #    b. Transform vertices to world frame
    #    c. Transform from world to body-local frame
    #    d. Handle quad faces by triangulating

    # 4. Combine all meshes into single warp.Mesh
    combined_mesh = convert_to_warp_mesh(all_vertices, all_faces)

    return combined_mesh
```

### Why Body-Local Coordinates?

Using body-local coordinates enables a key optimization:

1. **All environments share the same mesh geometry** (same robot model)
2. **Only the body pose differs** between environments
3. **Rays can be batched across ALL environments** with one transform

```
Traditional approach: O(N * C * num_agents) raycasts
Body-local approach:  O(num_agents) raycasts
Speedup: Up to N*C times faster!
```

## Temporal Smoothing

To prevent oscillation at occlusion boundaries, the environment applies temporal smoothing:

```python
# Exponential Moving Average (EMA) filter
alpha = 0.3  # Smoothing factor

# In _update_state_cache():
smoothed_confidence = (
    alpha * raw_confidence +
    (1 - alpha) * smoothed_confidence
)

# Final empty decision uses smoothed value
bbox_empty = smoothed_confidence < threshold  # e.g., 0.4
```

This prevents rapid switching between detected/occluded states when targets are near occlusion boundaries.

## Performance Considerations

### Memory Usage

| Tensor | Shape | Size (N=256, C=3, T=1, fp32) |
|--------|-------|------------------------------|
| corners_world | (N, T, 8, 3) | 24 KB |
| corners_camera | (N, C, T, 8, 3) | 72 KB |
| pixels | (N, C, T, 8, 2) | 48 KB |
| bboxes | (N, C, T, 4) | 12 KB |
| visibility_ratio | (N, C, T) | 3 KB |

Total: ~160 KB for core buffers (excluding meshes)

### Raycasting Performance

With 9-point occlusion pattern:
- Rays per update: N * C * T * K = 256 * 3 * 1 * 9 = 6,912 rays
- Static mesh: 1 raycast call
- Agent meshes: 3 raycast calls (one per agent)
- Total: 4 raycast calls per update

Typical performance: 1-5ms per update on RTX 4090
