# BBoxRayCaster Usage Guide

This guide provides practical examples and tips for using the BBoxRayCaster module in your multi-agent RL environment.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Configuration Tuning](#configuration-tuning)
3. [Common Use Cases](#common-use-cases)
4. [Reward Shaping](#reward-shaping)
5. [Debugging](#debugging)
6. [Performance Optimization](#performance-optimization)
7. [Common Issues](#common-issues)

---

## Basic Setup

### Step 1: Add to Environment Configuration

```python
from bbox_raycaster import BBoxRayCasterCfg

class MyEnvCfg(DirectMARLEnvCfg):
    bbox_raycaster: BBoxRayCasterCfg = BBoxRayCasterCfg(
        target_prim_paths=["/World/envs/env_.*/target"],
        mesh_prim_paths=["/World/ground"],
        num_cameras_per_env=2,  # Match your agent count
    )
```

### Step 2: Initialize in Environment

```python
class MyEnv(DirectMARLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        self.bbox_raycaster = BBoxRayCaster(
            cfg=cfg.bbox_raycaster,
            num_envs=self.num_envs,
            num_targets_per_env=1,
            device=self.device
        )
        
        # Storage for per-agent bboxes
        self.bboxes = {}
        self.bbox_valid = {}
```

### Step 3: Update in Observation Collection

```python
def _get_observations(self):
    # Collect camera data
    camera_poses = {
        agent_id: self._get_camera_pose(agent_id)
        for agent_id in self.agents
    }
    
    camera_intrinsics = {
        agent_id: self._get_camera_intrinsics(agent_id)
        for agent_id in self.agents
    }
    
    # Update raycaster
    self.bbox_raycaster.update(
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
        target_poses=(
            self.target.data.root_pos_w,
            self.target.data.root_quat_w
        )
    )
    
    # Extract per-agent bboxes
    for i, agent_id in enumerate(self.agents):
        self.bboxes[agent_id] = self.bbox_raycaster.data.bboxes_normalized[:, i, 0, :]
        self.bbox_valid[agent_id] = self.bbox_raycaster.data.valid_mask[:, i, 0]
    
    # Continue with observation construction...
```

---

## Configuration Tuning

### For Training Efficiency

Prioritize speed over accuracy during training:

```python
BBoxRayCasterCfg(
    # Validation
    partial_detection_allowed=True,  # Allow partial views
    min_bbox_size=(0.01, 0.01),     # Very permissive
    max_bbox_size=(0.99, 0.99),
    
    # Occlusion - use fastest pattern
    enable_occlusion_check=True,
    occlusion_ray_pattern="center_only",  # Fastest
    occlusion_visibility_threshold=0.3,   # More permissive
    
    # No debug overhead
    debug_vis=False,
    debug_memory=False,
)
```

### For Evaluation/Testing

Prioritize accuracy for evaluation:

```python
BBoxRayCasterCfg(
    # Strict validation
    partial_detection_allowed=False,  # Require full visibility
    min_bbox_size=(0.02, 0.02),
    max_bbox_size=(0.90, 0.90),
    
    # Accurate occlusion
    enable_occlusion_check=True,
    occlusion_ray_pattern="9point",  # Most accurate
    occlusion_visibility_threshold=0.6,  # Strict
    
    # Debug if needed
    debug_vis=True,
)
```

### For Dense Environments

When environments have many obstacles:

```python
BBoxRayCasterCfg(
    # Include all relevant meshes
    mesh_prim_paths=[
        "/World/ground",
        "/World/envs/env_.*/obstacles",
        "/World/envs/env_.*/walls",
    ],
    
    # Robust occlusion checking
    occlusion_ray_pattern="9point",
    occlusion_visibility_threshold=0.5,
    occlusion_ray_tolerance=1.2,  # More tolerance for complex geometry
)
```

---

## Common Use Cases

### Use Case 1: Target Tracking with Gimbaled Camera

Track a moving target while maintaining it centered:

```python
def _compute_rewards(self):
    rewards = {}
    
    for agent_id in self.agents:
        # Get bbox data
        bbox = self.bboxes[agent_id]  # (N, 4) - (cx, cy, w, h) normalized
        valid = self.bbox_valid[agent_id]  # (N,)
        
        # Reward for keeping target in view
        visibility_reward = valid.float() * 2.0
        
        # Reward for centering (cx, cy should be ~0.5)
        cx, cy = bbox[:, 0], bbox[:, 1]
        center_error = torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
        centering_reward = torch.exp(-5.0 * center_error) * valid.float()
        
        # Reward for maintaining good bbox size (not too small/large)
        w, h = bbox[:, 2], bbox[:, 3]
        area = w * h
        optimal_area = 0.15  # 15% of image
        size_error = torch.abs(area - optimal_area)
        size_reward = torch.exp(-10.0 * size_error) * valid.float()
        
        rewards[agent_id] = (
            0.3 * visibility_reward +
            0.5 * centering_reward +
            0.2 * size_reward
        )
    
    return rewards
```

### Use Case 2: Zoom Control Based on Target Distance

Automatically adjust zoom to maintain optimal bbox size:

```python
def _compute_zoom_action(self, agent_id: str):
    """Compute desired zoom level based on current bbox size."""
    bbox = self.bboxes[agent_id]
    valid = self.bbox_valid[agent_id]
    
    # Target area: 20% of image
    target_area = 0.20
    
    # Current area
    w, h = bbox[:, 2], bbox[:, 3]
    current_area = w * h
    
    # Compute zoom adjustment
    # If bbox too small -> zoom in
    # If bbox too large -> zoom out
    area_ratio = current_area / target_area
    
    # Clamp to prevent extreme zooms
    zoom_factor = torch.clamp(area_ratio, 0.5, 2.0)
    
    # Only adjust when target is valid
    zoom_factor = torch.where(valid, zoom_factor, torch.ones_like(zoom_factor))
    
    return zoom_factor
```

### Use Case 3: Multi-Target Tracking

Track multiple targets and select the best one:

```python
# For environments with multiple targets per env:
# Initialize with num_targets_per_env > 1

def _select_best_target(self, agent_id: str, agent_idx: int):
    """Select the best target to track based on visibility and position."""
    # Get all targets for this agent
    bboxes = self.bbox_raycaster.data.bboxes[:, agent_idx, :, :]  # (N, T, 4)
    valid = self.bbox_raycaster.data.valid_mask[:, agent_idx, :]   # (N, T)
    
    # Compute score for each target
    cx, cy = bboxes[..., 0], bboxes[..., 1]
    w, h = bboxes[..., 2], bboxes[..., 3]
    
    # Score based on centrality and size
    center_score = 1.0 - torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    size_score = w * h
    
    # Combined score (only for valid targets)
    score = (center_score + size_score) * valid.float()
    
    # Select best target
    best_target_idx = torch.argmax(score, dim=-1)  # (N,)
    
    # Extract bbox for best target
    batch_indices = torch.arange(self.num_envs, device=self.device)
    best_bbox = bboxes[batch_indices, best_target_idx]
    best_valid = valid[batch_indices, best_target_idx]
    
    return best_bbox, best_valid
```

### Use Case 4: Occlusion-Aware Navigation

Penalize actions that lead to occlusions:

```python
def _compute_occlusion_penalty(self, agent_id: str, agent_idx: int):
    """Penalize when target becomes occluded."""
    # Current visibility
    current_valid = self.bbox_valid[agent_id]
    
    # Visibility ratio (how much of target is visible)
    visibility_ratio = self.bbox_raycaster.data.occlusion_visibility_ratio[:, agent_idx, 0]
    
    # Strong penalty for occlusion
    occlusion_penalty = torch.where(
        current_valid,
        torch.zeros_like(visibility_ratio),
        -5.0 * torch.ones_like(visibility_ratio)
    )
    
    # Gradual penalty as visibility decreases
    visibility_penalty = -2.0 * (1.0 - visibility_ratio)
    
    return occlusion_penalty + visibility_penalty
```

---

## Reward Shaping

### Complete Reward Function Example

```python
def _compute_rewards(self):
    rewards = {}
    
    for i, agent_id in enumerate(self.agents):
        # Base rewards (existing)
        base_reward = self._compute_base_reward(agent_id)
        
        # Bbox data
        bbox = self.bboxes[agent_id]
        valid = self.bbox_valid[agent_id]
        visibility_ratio = self.bbox_raycaster.data.occlusion_visibility_ratio[:, i, 0]
        
        # Component rewards
        r_visibility = self._reward_visibility(valid, visibility_ratio)
        r_centering = self._reward_centering(bbox, valid)
        r_size = self._reward_size(bbox, valid)
        r_occlusion = self._reward_no_occlusion(valid, visibility_ratio)
        
        # Weighted combination
        rewards[agent_id] = (
            1.0 * base_reward +
            0.5 * r_visibility +
            0.3 * r_centering +
            0.2 * r_size +
            0.4 * r_occlusion
        )
    
    return rewards

def _reward_visibility(self, valid, visibility_ratio):
    """Reward for keeping target visible."""
    return valid.float() * 2.0 + visibility_ratio * 1.0

def _reward_centering(self, bbox, valid):
    """Reward for keeping target centered."""
    cx, cy = bbox[:, 0], bbox[:, 1]
    error = torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    return torch.exp(-5.0 * error) * valid.float()

def _reward_size(self, bbox, valid):
    """Reward for maintaining optimal bbox size."""
    w, h = bbox[:, 2], bbox[:, 3]
    area = w * h
    optimal = 0.15
    error = torch.abs(area - optimal)
    return torch.exp(-10.0 * error) * valid.float()

def _reward_no_occlusion(self, valid, visibility_ratio):
    """Penalty for occlusions."""
    penalty = torch.where(valid, torch.zeros_like(visibility_ratio), -3.0)
    gradual = -1.0 * (1.0 - visibility_ratio)
    return penalty + gradual
```

---

## Debugging

### Enable Debug Visualization

```python
BBoxRayCasterCfg(
    debug_vis=True,
    debug_vis_corners=True,
    debug_vis_occlusion_rays=True,
)

# In your update loop
self.bbox_raycaster.update(...)
self.bbox_raycaster.visualize()  # Update markers
```

### Log Detection Statistics

```python
def _log_bbox_stats(self):
    """Log bbox detection statistics for debugging."""
    for i, agent_id in enumerate(self.agents):
        valid = self.bbox_valid[agent_id]
        num_valid = valid.sum().item()
        
        if self.cfg.bbox_raycaster.enable_occlusion_check:
            vis_ratio = self.bbox_raycaster.data.occlusion_visibility_ratio[:, i, 0]
            avg_vis = vis_ratio[valid].mean().item() if valid.any() else 0.0
        else:
            avg_vis = 1.0 if valid.any() else 0.0
        
        print(f"{agent_id}: {num_valid}/{self.num_envs} valid "
              f"(avg visibility: {avg_vis:.2f})")
```

### Check for Invalid Bboxes

```python
def _check_bbox_validity(self, agent_id: str):
    """Debug helper to understand why bboxes are invalid."""
    bbox = self.bboxes[agent_id]
    valid = self.bbox_valid[agent_id]
    
    # Check different failure modes
    invalid_envs = ~valid
    
    if invalid_envs.any():
        # Size issues
        w, h = bbox[:, 2], bbox[:, 3]
        too_small = (w < 0.01) | (h < 0.01)
        too_large = (w > 0.95) | (h > 0.95)
        
        print(f"\n{agent_id} Invalid Bboxes:")
        print(f"  Too small: {(invalid_envs & too_small).sum().item()}")
        print(f"  Too large: {invalid_envs & too_large).sum().item()}")
        
        if self.cfg.bbox_raycaster.enable_occlusion_check:
            # Check occlusion
            print(f"  Occluded: (check visibility ratio)")
```

---

## Performance Optimization

### Profile Your Setup

```bash
python benchmark_bbox_raycaster.py --num_envs 4096 --num_cameras 2
```

### Optimization Checklist

1. **Disable occlusion during training** (if not critical):
   ```python
   enable_occlusion_check=False  # ~3-4x faster
   ```

2. **Use fastest occlusion pattern**:
   ```python
   occlusion_ray_pattern="center_only"  # ~2x faster than "9point"
   ```

3. **Reduce camera count** (if possible):
   - 2 cameras: baseline
   - 4 cameras: ~2x slower

4. **Pre-allocate all buffers** (already done in module)

5. **Avoid Python loops** - use batched operations

### Memory Optimization

For environments with limited GPU memory:

```python
BBoxRayCasterCfg(
    # Disable debug features
    debug_vis=False,
    debug_vis_corners=False,
    debug_vis_occlusion_rays=False,
    
    # Use simpler occlusion
    occlusion_ray_pattern="corners_only",  # Uses less memory than "9point"
)
```

---

## Common Issues

### Issue 1: All Bboxes Invalid

**Symptoms:** `valid_mask` is all `False`

**Diagnosis:**
```python
# Check camera poses
print("Camera positions:", camera_poses[agent_id][0][:5])
print("Camera quaternions:", camera_poses[agent_id][1][:5])

# Check target poses
print("Target positions:", target_poses[0][:5])

# Check intrinsics
print("Intrinsics:", camera_intrinsics[agent_id][:1])
```

**Solutions:**
1. Verify camera is looking at target
2. Check quaternion convention (ROS vs OpenGL)
3. Ensure targets are in FOV
4. Temporarily set `partial_detection_allowed=True`

### Issue 2: Bbox Positions Incorrect

**Symptoms:** Bboxes don't match visual target position

**Solutions:**
1. Verify camera offset convention matches configuration
2. Check if gimbal joint states are applied correctly
3. Ensure target bbox was extracted from correct prim

### Issue 3: Performance Degradation

**Symptoms:** Update takes >5ms

**Solutions:**
1. Profile with `debug_memory=True`
2. Reduce occlusion checking complexity
3. Check for memory allocations in update loop
4. Verify no Python loops over environments

### Issue 4: Memory Leaks

**Symptoms:** GPU memory grows over time

**Solutions:**
1. Ensure no tensor accumulation in reward computation
2. Detach tensors used for logging
3. Clear unused debug buffers

---

## Best Practices

1. **Always normalize bboxes** - Use `bboxes_normalized` in observations/rewards
2. **Handle invalid cases** - Check `valid_mask` before using bbox data
3. **Use visibility ratio** - Gradual occlusion signal is better than binary
4. **Profile first** - Measure before optimizing
5. **Start simple** - Disable occlusion initially, add complexity gradually

---

For more examples, see `integration_example.py` in the repository.