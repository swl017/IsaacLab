# Initial States Specification

## Purpose

This module provides configurable initialization of agent and target states for the iris_ma6 multi-agent drone environment. It supports both controlled positioning and curriculum-driven randomization for reinforcement learning training.

## Scope

The module handles initial state configuration for:
- Agent positions, orientations, and velocities
- Target position, orientation, and velocity
- Gimbal joint angles (yaw, pitch)
- Zoom levels
- "Designated observer" selection (one agent per environment guaranteed to view target)

## Design Philosophy

### Curriculum Sampling Strategy
**To prevent catastrophic forgetting**, curriculum-controlled parameters use the sampling strategy:

```
value ~ Uniform(min, min + progress * (max - min))
```

This ensures that as training progresses:
- At `progress=0`: Only easy samples (min values)
- At `progress=0.5`: Easy to medium samples (min to midpoint)
- At `progress=1`: Full range (min to max)

**Key insight**: All previous difficulty levels remain in the sampling distribution, so the policy doesn't forget how to handle easier scenarios.

### Designated Observer Concept
Each environment has exactly one "designated observer" - an agent that:
1. Has its gimbal pointing at the target
2. (Optionally) has its body facing the target direction
3. Ensures at least one detection per reset

---

## Configuration Parameters

### Cylinder Configuration (Agent Placement)

| Parameter | Default | Sampling | Description |
|-----------|---------|----------|-------------|
| `cylinder_diameter_min` | 30.0 m | Lower bound | Minimum cylinder diameter |
| `cylinder_diameter_max` | 100.0 m | `uniform(min, min+p*(max-min))` | Maximum cylinder diameter |
| `cylinder_height_min` | 10.0 m | Fixed | Minimum base height above ground |
| `cylinder_height_max` | 50.0 m | Fixed | Maximum base height above ground |
| `cylinder_height_range` | 20.0 m | Fixed | Vertical spread within cylinder |
| `agent_clearance` | 10.0 m | Fixed | Minimum distance between any two agents |

### Target Configuration

| Parameter | Default | Sampling | Description |
|-----------|---------|----------|-------------|
| `target_distance_min` | 30.0 m | Lower bound | Minimum distance from cylinder center |
| `target_distance_max` | 200.0 m | `uniform(min, min+p*(max-min))` | Maximum distance from cylinder center |
| `target_height_offset` | [-10, +10] m | `uniform(-10, 10)` | Height offset from cylinder center |

### Velocity Configuration

| Parameter | Default | Sampling | Description |
|-----------|---------|----------|-------------|
| `agent_max_velocity` | 10.0 m/s | Upper bound | Maximum agent velocity magnitude |
| `agent_velocity_scale_max` | 1.0 | `uniform(0, p*scale_max)` | Velocity scale at progress=1 |
| `target_max_velocity` | 10.0 m/s | Upper bound | Maximum target velocity magnitude |
| `target_velocity_scale_max` | 1.0 | `uniform(0, p*scale_max)` | Velocity scale at progress=1 |
| `max_yaw_rate` | 45 deg/s | Fixed | Maximum initial angular velocity |

**Velocity Sampling:**
```
agent_velocity = uniform(0, progress * agent_velocity_scale_max) * agent_max_velocity
target_velocity = uniform(0, progress * target_velocity_scale_max) * target_max_velocity
```

### Gimbal Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gimbal_yaw_limits` | [-180, +180] deg | Yaw joint limits |
| `gimbal_pitch_limits` | [-45, +45] deg | Pitch joint limits |
| `gimbal_curriculum_mode` | "gradual" | How non-observers randomize (see below) |

**Gimbal Curriculum Modes:**
- `"gradual"`: Interpolate between all-pointing and random based on progress
  - At progress=p, with probability (1-p) agent points at target, otherwise random
- `"threshold"`: Switch to random at specific progress threshold
- `"always_pointing"`: All agents always point at target (for early debugging)

### Zoom Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zoom_min` | 1.0 | Minimum zoom level |
| `zoom_max` | 30.0 | Maximum zoom level |
| `zoom_initial_range` | [1.0, 10.0] | Initial random sampling range |

### Designated Observer Selection

| Mode | Description |
|------|-------------|
| `"random"` | Random agent each reset |
| `"fixed"` | Always agent 0 |
| `"rotating"` | Cycle through agents across resets |

---

## Generation Algorithm

### Step 1: Sample Curriculum-Controlled Parameters
```python
# Sample from ranges that grow with progress (prevents forgetting)
diameter = uniform(diameter_min, diameter_min + progress * (diameter_max - diameter_min))
target_distance = uniform(distance_min, distance_min + progress * (distance_max - distance_min))
velocity_scale = uniform(0, progress * velocity_scale_max)
```

### Step 2: Generate Cylinder Center
- Random XY position within environment origin offset
- Random Z within [cylinder_height_min, cylinder_height_max]

### Step 3: Place Agents in Cylinder
- Use rejection sampling to place agents within cylinder (diameter from Step 1)
- Ensure minimum `agent_clearance` between all pairs
- Fallback to evenly-distributed placement if rejection sampling fails

### Step 4: Generate Target Position
- Distance from Step 1 (curriculum-sampled)
- Random bearing from cylinder center
- Height = cylinder center height + uniform(height_offset_min, height_offset_max)

### Step 5: Generate Target Velocity
- Random direction (unit vector)
- Magnitude = velocity_scale (from Step 1) * target_max_velocity

### Step 6: Select Designated Observer
- Per configured mode (random/fixed/rotating)
- Store index in result for downstream use

### Step 7: Generate Agent Body Orientations
- Designated observer: faces target (with noise)
- Other agents: random yaw (configurable)
- All agents: zero roll and pitch (level flight)

### Step 8: Generate Agent Velocities
- Random direction
- Magnitude = velocity_scale (from Step 1) * agent_max_velocity
- Angular velocity: uniform(-max_yaw_rate * progress, +max_yaw_rate * progress)

### Step 9: Generate Gimbal States
- Designated observer: compute angles to point at target
- Other agents (curriculum-based):
  - Sample coin flip: rand() < (1 - progress)
  - If true: point at target
  - If false: random within gimbal limits
- Roll always 0 (auto-stabilized by controller)

### Step 10: Generate Zoom Levels
- Random within `zoom_initial_range`

---

## Result Structure

```python
@dataclass
class InitialStatesResult:
    # Agent states [num_envs, num_agents, ...]
    agent_positions: torch.Tensor        # [N, A, 3]
    agent_orientations: torch.Tensor     # [N, A, 4] quaternion (wxyz)
    agent_linear_velocities: torch.Tensor   # [N, A, 3]
    agent_angular_velocities: torch.Tensor  # [N, A, 3]

    # Target states [num_envs, ...]
    target_positions: torch.Tensor       # [N, 3]
    target_orientations: torch.Tensor    # [N, 4] quaternion (wxyz)
    target_velocities: torch.Tensor      # [N, 6] (linear + angular)

    # Gimbal states [num_envs, num_agents, 3] = [pitch, yaw, roll]
    gimbal_joint_positions: torch.Tensor # [N, A, 3]

    # Zoom states
    zoom_levels: torch.Tensor            # [N, A]

    # Metadata
    designated_observer_idx: torch.Tensor # [N] agent index
    cylinder_centers: torch.Tensor       # [N, 3]
    distances_to_target: torch.Tensor    # [N, A]
```

---

## Environment Integration

### Initialization
```python
self._initial_states = InitialStates(
    cfg=self.cfg.initial_states,
    num_envs=self.num_envs,
    num_agents=len(self.cfg.possible_agents),
    device=self.device,
)
```

### Reset
```python
def _reset_idx(self, env_ids: torch.Tensor):
    result = self._initial_states.generate(
        env_ids=env_ids,
        curriculum_progress=self._curriculum_progress,
    )

    # Apply agent states to robots
    for idx, agent_id in enumerate(self.cfg.possible_agents):
        robot = self._robots[agent_id]
        # Apply root pose and velocity
        # Apply gimbal joint states
        # Set zoom level

    # Apply target states
```

---

## Curriculum Schedule Example

With `uniform(min, min + progress * (max - min))` sampling:

| Progress | Distance Range | Velocity Range | Gimbal Pointing Prob | Cylinder Diameter Range |
|----------|----------------|----------------|----------------------|-------------------------|
| 0.0 | [30, 30] m | [0, 0] m/s | 100% | [30, 30] m |
| 0.25 | [30, 72.5] m | [0, 2.5] m/s | 75% | [30, 47.5] m |
| 0.5 | [30, 115] m | [0, 5] m/s | 50% | [30, 65] m |
| 0.75 | [30, 157.5] m | [0, 7.5] m/s | 25% | [30, 82.5] m |
| 1.0 | [30, 200] m | [0, 10] m/s | 0% (observer only) | [30, 100] m |

**Note**: At progress=1.0, easy samples (30m, 0 m/s) are still possible, preventing forgetting.

---

## Notes

### Gimbal Angle Computation
Gimbal angles are computed to point the camera (attached to pitch_link) at the target:
1. Compute direction from agent to target in world frame
2. Transform to body frame using inverse of agent orientation
3. Extract yaw (rotation in body XY plane)
4. Extract pitch (elevation angle)
5. Account for `YAW_JOINT_OFFSET = -pi/2`

### Controller State Sync
After applying gimbal joint positions via `write_joint_state_to_sim`, the DroneController's internal gimbal state must be synchronized to prevent jumps on first control step.

### Rejection Sampling Fallback
If rejection sampling fails to place all agents within max attempts, fall back to even angular distribution around cylinder center with random radius and height.

### Why Uniform(min, min + p*(max-min)) Instead of Interpolation?
Traditional curriculum uses fixed difficulty that increases over time:
```
difficulty = min + progress * (max - min)  # WRONG: forgets easy cases
```

Our approach maintains the full range of easier difficulties:
```
difficulty = uniform(min, min + progress * (max - min))  # CORRECT: no forgetting
```

This is essential for RL where the policy must remain competent across all difficulty levels.
