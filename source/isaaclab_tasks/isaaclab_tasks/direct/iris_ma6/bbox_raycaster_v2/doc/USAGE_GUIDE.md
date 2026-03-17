# BBoxRayCasterV2 Usage Guide

This guide demonstrates how to integrate BBoxRayCasterV2 into your multi-agent environment.

## Quick Start

```python
from bbox_raycaster_v2 import BBoxRayCasterV2, BBoxRayCasterV2Cfg

# 1. Create configuration
cfg = BBoxRayCasterV2Cfg(
    target_prim_paths=["/World/envs/env_.*/target"],
    mesh_prim_paths=["/World/ground"],
    num_cameras_per_env=3,
    enable_occlusion_check=True,
    occlusion_ray_pattern="9point",
)

# 2. Initialize raycaster
raycaster = BBoxRayCasterV2(
    cfg=cfg,
    num_envs=256,
    num_targets_per_env=1,
    device="cuda:0",
    agent_ids=["drone_0", "drone_1", "drone_2"]
)

# 3. Update each frame
raycaster.update(
    camera_poses=camera_poses,      # Dict[agent_id -> (pos, quat)]
    camera_intrinsics=intrinsics,   # Dict[agent_id -> K matrix]
    target_poses=target_poses,      # (pos_tensor, quat_tensor)
    agent_poses=agent_poses         # Dict[agent_id -> (pos, quat)]
)

# 4. Access results
bboxes = raycaster.data.bboxes           # (N, C, T, 4) - xywh format
bbox_empty = raycaster.data.bbox_empty   # (N, C, T) - True if invalid
confidence = raycaster.data.bbox_confidence  # (N, C, T) - [0, 1]
```

## Integration with DirectMARLEnv

### Environment Configuration

```python
# In your_env_cfg.py

@configclass
class YourEnvCfg(DirectMARLEnvCfg):
    # ... other config ...

    bbox_raycaster_v2: BBoxRayCasterV2Cfg = BBoxRayCasterV2Cfg(
        target_prim_paths=["/World/envs/env_.*/target"],
        mesh_prim_paths=["/World/ground"],
        num_cameras_per_env=3,
        num_cameras_per_agent=1,

        # Occlusion settings
        load_agent_meshes=True,
        enable_occlusion_check=True,
        enable_self_occlusion=True,
        enable_inter_target_occlusion=True,
        occlusion_ray_pattern="9point",
        occlusion_visibility_threshold=0.3,

        # Validation
        min_bbox_size=(0.01, 0.01),
        max_bbox_size=(0.95, 0.95),
        partial_detection_allowed=False,
    )

    def __post_init__(self):
        # Update cameras based on num_agents
        self.bbox_raycaster_v2.num_cameras_per_env = self.num_agents
```

### Environment Initialization

```python
# In your_env.py

class YourEnv(DirectMARLEnv):
    cfg: YourEnvCfg

    def __init__(self, cfg: YourEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize bbox raycaster after scene setup
        self._setup_bbox_raycaster()

        # Temporal smoothing buffers
        self._occlusion_ema_alpha = 0.3
        self._smoothed_bbox_confidence = torch.ones(
            self.num_envs, self.cfg.num_agents, 1,
            device=self.device
        )
        self._smoothed_bbox_empty_threshold = 0.4

    def _setup_bbox_raycaster(self):
        """Initialize bbox raycaster with current scene."""
        agent_ids = [f"drone_{i}" for i in range(self.cfg.num_agents)]

        self.bbox_raycaster_v2 = BBoxRayCasterV2(
            cfg=self.cfg.bbox_raycaster_v2,
            num_envs=self.num_envs,
            num_targets_per_env=1,
            device=self.device,
            agent_ids=agent_ids
        )

        # Pre-compute base intrinsics from camera config
        self._base_intrinsic = self._compute_intrinsic_matrix(
            focal_length=self.cfg.camera.spawn.focal_length,
            h_aperture=self.cfg.camera.spawn.horizontal_aperture,
            width=self.cfg.camera.width,
            height=self.cfg.camera.height
        )

    def _compute_intrinsic_matrix(self, focal_length, h_aperture, width, height):
        """Compute camera intrinsic matrix from camera parameters."""
        # Focal length in pixels
        fx = focal_length * width / h_aperture
        fy = fx  # Square pixels

        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0

        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], device=self.device)

        return K
```

### Update Loop Integration

```python
def _update_state_cache(self, dt=None):
    """Update internal state caches including bbox detection."""

    # === Update bbox raycaster ===

    # 1. Build camera poses dict
    camera_poses = {}
    for i, agent_id in enumerate(self.cfg.possible_agents):
        camera_pos = self._robots[agent_id].data.body_state_w[
            :, self._camera_body_idx, :3
        ]  # (N, 3)
        camera_quat = self._robots[agent_id].data.body_state_w[
            :, self._camera_body_idx, 3:7
        ]  # (N, 4)
        camera_poses[agent_id] = (camera_pos, camera_quat)

    # 2. Build camera intrinsics dict (with zoom support)
    camera_intrinsics = {}
    for i, agent_id in enumerate(self.cfg.possible_agents):
        # Apply zoom to intrinsics
        zoom_level = self._zoom_levels[:, i]  # (N,)
        K = self._base_intrinsic.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        K[:, 0, 0] *= zoom_level  # fx
        K[:, 1, 1] *= zoom_level  # fy
        camera_intrinsics[agent_id] = K

    # 3. Build target poses
    target_pos = self._target.data.root_pos_w.unsqueeze(1)  # (N, 1, 3)
    target_quat = self._target.data.root_quat_w.unsqueeze(1)  # (N, 1, 4)
    target_poses = (target_pos, target_quat)

    # 4. Build agent poses for occlusion
    agent_poses = {}
    for agent_id in self.cfg.possible_agents:
        body_pos = self._robots[agent_id].data.root_pos_w  # (N, 3)
        body_quat = self._robots[agent_id].data.root_quat_w  # (N, 4)
        agent_poses[agent_id] = (body_pos, body_quat)

    # 5. Update raycaster
    self.bbox_raycaster_v2.update(
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
        target_poses=target_poses,
        agent_poses=agent_poses,
        image_shapes={
            agent_id: (self.cfg.camera.height, self.cfg.camera.width)
            for agent_id in self.cfg.possible_agents
        }
    )

    # 6. Apply temporal smoothing
    raw_confidence = self.bbox_raycaster_v2.data.bbox_confidence  # (N, C, T)
    self._smoothed_bbox_confidence = (
        self._occlusion_ema_alpha * raw_confidence +
        (1.0 - self._occlusion_ema_alpha) * self._smoothed_bbox_confidence
    )

def _get_observations(self) -> dict[str, torch.Tensor]:
    """Build observations for each agent."""
    observations = {}

    for i, agent_id in enumerate(self.cfg.possible_agents):
        # Get bbox for this camera
        bbox = self.bbox_raycaster_v2.data.bboxes_normalized[:, i, 0, :]  # (N, 4)

        # Use smoothed empty decision
        bbox_empty = (
            self._smoothed_bbox_confidence[:, i, 0] < self._smoothed_bbox_empty_threshold
        ).float().unsqueeze(-1)  # (N, 1)

        # Build observation vector
        obs = torch.cat([
            # ... other observations ...
            bbox,           # (N, 4) - normalized xywh
            bbox_empty,     # (N, 1) - 1.0 if empty
        ], dim=-1)

        observations[agent_id] = obs

    return observations

def _reset_idx(self, env_ids: torch.Tensor):
    """Reset environments."""
    super()._reset_idx(env_ids)

    # Reset temporal smoothing for these environments
    self._smoothed_bbox_confidence[env_ids] = 1.0
```

## Working with Zoom

BBoxRayCasterV2 supports dynamic zoom by modifying camera intrinsics:

```python
def _apply_zoom_to_intrinsics(self, base_K: torch.Tensor, zoom: torch.Tensor):
    """Scale intrinsic matrix by zoom factor.

    Args:
        base_K: Base intrinsic matrix (N, 3, 3)
        zoom: Zoom level per camera (N,) - 1.0 = no zoom, 2.0 = 2x zoom

    Returns:
        Zoomed intrinsic matrix (N, 3, 3)
    """
    K_zoomed = base_K.clone()
    K_zoomed[:, 0, 0] *= zoom  # Scale fx
    K_zoomed[:, 1, 1] *= zoom  # Scale fy
    # Note: cx, cy remain unchanged (principal point)
    return K_zoomed
```

## Multiple Targets

For environments with multiple targets per environment:

```python
# Configure for T targets
cfg = BBoxRayCasterV2Cfg(
    target_prim_paths=[
        "/World/envs/env_.*/target_0",
        "/World/envs/env_.*/target_1",
        "/World/envs/env_.*/target_2",
    ],
    # ...
)

raycaster = BBoxRayCasterV2(
    cfg=cfg,
    num_envs=256,
    num_targets_per_env=3,  # T=3
    device="cuda:0"
)

# Target poses shape: (N, T, 3) and (N, T, 4)
target_pos = torch.stack([t.data.root_pos_w for t in targets], dim=1)
target_quat = torch.stack([t.data.root_quat_w for t in targets], dim=1)

# Results have T dimension
bboxes = raycaster.data.bboxes  # (N, C, 3, 4)
```

## Accessing Detailed Results

```python
# After update()...

# Bounding boxes
bboxes_xywh = raycaster.data.bboxes           # (N, C, T, 4) - center_x, center_y, w, h
bboxes_xyxy = raycaster.data.bboxes_xyxy      # (N, C, T, 4) - x_min, y_min, x_max, y_max
bboxes_norm = raycaster.data.bboxes_normalized # (N, C, T, 4) - values in [0, 1]

# Validity masks
bbox_empty = raycaster.data.bbox_empty         # (N, C, T) - True if invalid

# Detailed empty reasons
compute_empty = raycaster.data.bbox_compute_empty_mask   # Bbox computation failed
corners_empty = raycaster.data.bbox_corners_empty_mask   # Corners outside FOV
size_empty = raycaster.data.bbox_size_empty_mask         # Size out of bounds
visibility_empty = raycaster.data.bbox_visibility_empty_mask  # Occluded

# Occlusion data
visibility_ratio = raycaster.data.occlusion_visibility_ratio  # (N, C, T) - [0, 1]
self_occlusion = raycaster.data.self_occlusion_mask           # (N, C, T) - self-blocked
inter_occluded = raycaster.data.occluded                      # (N, C, T) - by other target

# Confidence
confidence = raycaster.data.bbox_confidence    # (N, C, T) - combined confidence
```

## Debug Visualization

Enable debug visualization to see occlusion rays:

```python
cfg = BBoxRayCasterV2Cfg(
    # ...
    debug_vis=True,
    debug_vis_corners=True,
    debug_vis_occlusion_rays=True,
)

# In render loop
raycaster.visualize()
```

## Performance Tips

1. **Reduce test points for speed**: Use `occlusion_ray_pattern="center_only"` during initial testing

2. **Simplify agent meshes**: Set `agent_mesh_simplification=0.3` for faster raycasting

3. **Limit max distance**: Set `max_distance` to match your environment size

4. **Disable unused features**:
   ```python
   enable_self_occlusion=False     # If camera never sees own body
   enable_inter_target_occlusion=False  # If targets don't overlap
   ```

5. **Use collision proxies for testing**:
   ```python
   use_collision_proxy=True  # Simple box instead of full mesh
   ```

## Common Patterns

### Reward Shaping Based on Detection

```python
def _compute_reward(self):
    # Reward for keeping target in view
    bbox_visible = ~self.bbox_raycaster_v2.data.bbox_empty  # (N, C, T)
    visibility_reward = bbox_visible.float().mean(dim=(1, 2))  # (N,)

    # Reward for centering target
    bbox_norm = self.bbox_raycaster_v2.data.bboxes_normalized  # (N, C, T, 4)
    center_x, center_y = bbox_norm[..., 0], bbox_norm[..., 1]
    centering_error = ((center_x - 0.5)**2 + (center_y - 0.5)**2).mean(dim=(1, 2))
    centering_reward = 1.0 - centering_error.clamp(0, 1)

    return visibility_reward * 1.0 + centering_reward * 0.5
```

### Termination on Loss of View

```python
def _get_dones(self):
    # Terminate if target lost for too long
    bbox_empty = self.bbox_raycaster_v2.data.bbox_empty.all(dim=1)  # (N, T)
    self._frames_without_detection[bbox_empty.any(dim=1)] += 1
    self._frames_without_detection[~bbox_empty.any(dim=1)] = 0

    terminated = self._frames_without_detection > 100  # 100 frames
    return terminated, torch.zeros_like(terminated)
```

### Multi-Agent Coordination

```python
def _compute_triangulation_reward(self):
    # Get bbox confidence for all cameras
    confidence = self.bbox_raycaster_v2.data.bbox_confidence  # (N, C, T)

    # Reward when multiple cameras see target
    cameras_seeing_target = (confidence > 0.5).sum(dim=1)  # (N, T)
    triangulation_possible = cameras_seeing_target >= 2

    return triangulation_possible.float().mean(dim=1)  # (N,)
```
