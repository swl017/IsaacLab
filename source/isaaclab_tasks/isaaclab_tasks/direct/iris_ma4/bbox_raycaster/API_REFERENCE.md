# BBoxRayCaster API Reference

Quick reference for the BBoxRayCaster module API.

---

## Classes

### `BBoxRayCaster`

Main class for batched bounding box extraction with occlusion detection.

#### Constructor

```python
BBoxRayCaster(
    cfg: BBoxRayCasterCfg,
    num_envs: int,
    num_targets_per_env: int,
    device: str
)
```

**Parameters:**
- `cfg`: Configuration instance
- `num_envs`: Number of parallel environments
- `num_targets_per_env`: Number of targets per environment
- `device`: Device string (e.g., "cuda:0")

#### Methods

##### `update()`

```python
update(
    camera_poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
    camera_intrinsics: dict[str, torch.Tensor],
    target_poses: tuple[torch.Tensor, torch.Tensor],
    image_shapes: dict[str, tuple[int, int]] | None = None
)
```

Main update method. Computes bounding boxes for current frame.

**Parameters:**
- `camera_poses`: Dict mapping agent_id to (position, quaternion)
  - Position: `(N, 3)` - world frame
  - Quaternion: `(N, 4)` - (w, x, y, z) format
- `camera_intrinsics`: Dict mapping agent_id to intrinsic matrix `(N, 3, 3)`
- `target_poses`: Tuple of (position, quaternion)
  - Position: `(N, T, 3)`
  - Quaternion: `(N, T, 4)`
- `image_shapes`: Optional dict mapping agent_id to (height, width)

**Returns:** None (updates `self.data`)

##### `visualize()`

```python
visualize()
```

Update debug visualization markers (only if `debug_vis=True`).

#### Properties

##### `data`

```python
@property
def data(self) -> BBoxRayCasterData
```

Access to the data container with all outputs.

---

## Data Structures

### `BBoxRayCasterData`

Container for all raycaster data.

#### Fields

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `bboxes` | `(N, C, T, 4)` | float32 | BBoxes in pixels: (cx, cy, w, h) |
| `bboxes_normalized` | `(N, C, T, 4)` | float32 | Normalized to [0,1] |
| `valid_mask` | `(N, C, T)` | bool | True if detection valid |
| `camera_pos_w` | `(N, C, 3)` | float32 | Camera positions (world) |
| `camera_quat_w` | `(N, C, 4)` | float32 | Camera quaternions (w,x,y,z) |
| `target_pos_w` | `(N, T, 3)` | float32 | Target positions (world) |
| `target_quat_w` | `(N, T, 4)` | float32 | Target quaternions (w,x,y,z) |
| `intrinsic_matrices` | `(N, C, 3, 3)` | float32 | Camera intrinsics |
| `image_shapes` | `(N, C, 2)` | int64 | Image (height, width) |
| `occlusion_visibility_ratio` | `(N, C, T)` | float32 | Visibility [0,1] (if enabled) |

**Debug Fields** (only if `debug_vis=True`):
- `projected_corners_2d`: `(N, C, T, 8, 2)` - Corner pixels
- `corners_valid_mask`: `(N, C, T, 8)` - Corner validity
- `occlusion_test_points_w`: `(N, T, K, 3)` - Test points
- `occlusion_ray_hits_w`: `(N, C, T, K, 3)` - Ray hits

**Notation:**
- `N` = num_envs
- `C` = num_cameras_per_env
- `T` = num_targets_per_env
- `K` = num_occlusion_test_points

---

## Configuration

### `BBoxRayCasterCfg`

Configuration dataclass for BBoxRayCaster.

#### Parameters

##### Required

```python
target_prim_paths: list[str]
mesh_prim_paths: list[str]
num_cameras_per_env: int
```

##### Validation

```python
min_bbox_size: tuple[float, float] = (0.01, 0.01)
max_bbox_size: tuple[float, float] = (0.95, 0.95)
partial_detection_allowed: bool = False
min_bbox_area_pixels: float = 4.0
```

##### Occlusion

```python
enable_occlusion_check: bool = True
occlusion_ray_pattern: Literal["9point", "corners_only", "center_only"] = "9point"
occlusion_visibility_threshold: float = 0.5
occlusion_ray_tolerance: float = 1.1
```

##### Performance

```python
max_distance: float = 100.0
projection_epsilon: float = 1e-6
quat_normalize_epsilon: float = 1e-8
gimbal_lock_threshold: float = 0.99
warn_gimbal_lock: bool = False
```

##### Debug

```python
debug_vis: bool = False
debug_vis_corners: bool = False
debug_vis_occlusion_rays: bool = False
debug_memory: bool = False
visualizer_cfg: VisualizationMarkersCfg = ...
```

---

## Utility Functions

### Projection Utilities (`utils.projection`)

#### `batch_transform_points()`

```python
batch_transform_points(
    points_local: torch.Tensor,  # (K, 3) or (T, K, 3)
    pos: torch.Tensor,            # (N, T, 3)
    quat: torch.Tensor,           # (N, T, 4)
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor  # (N, T, K, 3)
```

Transform points from local to world frame.

#### `batch_transform_to_camera_frame()`

```python
batch_transform_to_camera_frame(
    points_world: torch.Tensor,   # (N, C, T, K, 3)
    camera_pos: torch.Tensor,     # (N, C, 3)
    camera_quat: torch.Tensor,    # (N, C, 4)
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor  # (N, C, T, K, 3)
```

Transform points from world to camera frame.

#### `batch_project_to_image_plane()`

```python
batch_project_to_image_plane(
    points_camera: torch.Tensor,      # (N, C, T, K, 3)
    intrinsic_matrices: torch.Tensor, # (N, C, 3, 3)
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# Returns: (pixels, depths, valid_mask)
```

Project 3D points to 2D image plane.

**Returns:**
- `pixels`: `(N, C, T, K, 2)` - (u, v) coordinates
- `depths`: `(N, C, T, K)` - z-coordinates
- `valid_mask`: `(N, C, T, K)` - validity

#### `check_points_in_fov()`

```python
check_points_in_fov(
    pixels: torch.Tensor,        # (N, C, T, K, 2)
    image_shapes: torch.Tensor   # (N, C, 2)
) -> torch.Tensor  # (N, C, T, K)
```

Check if pixels are within image bounds.

#### `check_gimbal_lock()`

```python
check_gimbal_lock(
    camera_quat: torch.Tensor,  # (N, C, 4)
    threshold: float = 0.99
) -> torch.Tensor  # (N, C)
```

Detect potential gimbal lock orientations.

---

### BBox Operations (`utils.bbox_ops`)

#### `get_bbox_corners_local()`

```python
get_bbox_corners_local(
    prim_path: str,
    device: str
) -> torch.Tensor  # (8, 3)
```

Extract 8 corners of 3D bounding box from prim.

#### `compute_2d_bbox_from_corners()`

```python
compute_2d_bbox_from_corners(
    corners_2d: torch.Tensor,     # (N, C, T, 8, 2)
    valid_corners: torch.Tensor,  # (N, C, T, 8)
    min_area_pixels: float = 4.0
) -> tuple[torch.Tensor, torch.Tensor]
# Returns: (bbox_xyxy, valid_bbox)
```

Compute 2D bboxes from projected corners.

**Returns:**
- `bbox_xyxy`: `(N, C, T, 4)` - (min_x, min_y, max_x, max_y)
- `valid_bbox`: `(N, C, T)` - validity

#### `bbox_xyxy_to_xywh()`

```python
bbox_xyxy_to_xywh(
    bbox_xyxy: torch.Tensor  # (N, C, T, 4)
) -> torch.Tensor  # (N, C, T, 4)
```

Convert bbox from xyxy to xywh format.

#### `normalize_bboxes()`

```python
normalize_bboxes(
    bboxes: torch.Tensor,        # (N, C, T, 4)
    image_shapes: torch.Tensor,  # (N, C, 2)
    epsilon: float = 1e-7
) -> torch.Tensor  # (N, C, T, 4)
```

Normalize bboxes to [0, 1] range.

#### `validate_bbox_sizes()`

```python
validate_bbox_sizes(
    bboxes_norm: torch.Tensor,  # (N, C, T, 4)
    min_size: tuple[float, float],
    max_size: tuple[float, float]
) -> torch.Tensor  # (N, C, T)
```

Check if bbox sizes are within valid range.

---

### Occlusion Detection (`utils.occlusion`)

#### `generate_occlusion_test_points()`

```python
generate_occlusion_test_points(
    corners_world: torch.Tensor,  # (N, T, 8, 3)
    pattern: Literal["9point", "corners_only", "center_only"],
    out: Optional[torch.Tensor] = None
) -> torch.Tensor  # (N, T, K, 3)
```

Generate test points for occlusion detection.

**Returns:** Test points, where K depends on pattern:
- `"9point"`: K=9
- `"corners_only"`: K=4
- `"center_only"`: K=1

#### `batch_check_occlusion()`

```python
batch_check_occlusion(
    camera_pos: torch.Tensor,         # (N, C, 3)
    test_points_world: torch.Tensor,  # (N, T, K, 3)
    target_positions: torch.Tensor,   # (N, T, 3)
    target_bbox_size: torch.Tensor,   # (N, T, 3) or (T, 3)
    mesh: wp.Mesh,
    max_distance: float,
    visibility_threshold: float,
    tolerance_scale: float = 1.1,
    out_hits: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]
# Returns: (visibility_mask, visibility_ratio)
```

Check occlusion using raycasting.

**Returns:**
- `visibility_mask`: `(N, C, T)` - boolean visibility
- `visibility_ratio`: `(N, C, T)` - fraction visible [0,1]

---

## Usage Patterns

### Basic Usage

```python
# Initialize
raycaster = BBoxRayCaster(cfg, num_envs, num_targets_per_env, device)

# Update
raycaster.update(camera_poses, camera_intrinsics, target_poses)

# Access results
bboxes = raycaster.data.bboxes_normalized  # (N, C, T, 4)
valid = raycaster.data.valid_mask           # (N, C, T)

# Per-agent extraction
for i, agent_id in enumerate(agents):
    agent_bbox = bboxes[:, i, 0, :]  # (N, 4)
    agent_valid = valid[:, i, 0]      # (N,)
```

### With Observations

```python
def _get_observations(self):
    # Update raycaster
    self.bbox_raycaster.update(...)
    
    # Extract bboxes
    for i, agent_id in enumerate(self.agents):
        bbox = self.bbox_raycaster.data.bboxes_normalized[:, i, 0, :]
        valid = self.bbox_raycaster.data.valid_mask[:, i, 0]
        
        # Add to observation
        obs[agent_id] = torch.cat([
            robot_state,
            bbox,
            valid.float().unsqueeze(-1)
        ], dim=-1)
```

### With Rewards

```python
def _compute_rewards(self):
    rewards = {}
    
    for i, agent_id in enumerate(self.agents):
        bbox = self.bbox_raycaster.data.bboxes_normalized[:, i, 0, :]
        valid = self.bbox_raycaster.data.valid_mask[:, i, 0]
        
        # Visibility reward
        r_vis = valid.float() * 2.0
        
        # Centering reward
        cx, cy = bbox[:, 0], bbox[:, 1]
        center_err = torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
        r_center = torch.exp(-5.0 * center_err) * valid.float()
        
        rewards[agent_id] = r_vis + r_center
    
    return rewards
```

---

## Performance Characteristics

| Configuration | Time (ms) | Memory (MB) |
|---------------|-----------|-------------|
| 4096 envs, 2 cams, no occlusion | 0.8 | 200 |
| 4096 envs, 2 cams, center_only | 2.0 | 250 |
| 4096 envs, 2 cams, 9point | 3.2 | 350 |
| 8192 envs, 4 cams, 9point | 6.5 | 700 |

*Benchmarked on NVIDIA RTX 4090*

---

## Error Handling

### Common Exceptions

```python
RuntimeError: Invalid mesh prim path
→ Check that mesh_prim_paths exist in scene

ValueError: Expected N cameras, got M
→ Ensure camera_poses dict has correct number of entries

RuntimeError: Corner transformation failed
→ Check target poses are valid (not NaN/Inf)
```

### Validation

```python
# Check if detections are working
num_valid = raycaster.data.valid_mask.sum()
if num_valid == 0:
    print("Warning: No valid detections!")
    # Check camera poses, intrinsics, target positions
```

---

## Tips & Best Practices

1. **Always use normalized bboxes** in observations/rewards
2. **Check valid_mask** before using bbox data
3. **Start with occlusion disabled** for debugging
4. **Use visibility_ratio** for gradual penalties
5. **Profile before optimizing** - measure first

---

For complete examples, see:
- `integration_example.py` - Full integration
- `USAGE_GUIDE.md` - Detailed use cases
- `README.md` - Quick start guide