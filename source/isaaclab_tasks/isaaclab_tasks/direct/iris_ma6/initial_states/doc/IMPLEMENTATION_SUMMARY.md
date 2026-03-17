# Initial States Module - Implementation Summary

## Purpose

This module provides configurable initialization of agent and target states for the iris_ma6 multi-agent drone environment. It supports curriculum-driven randomization that prevents catastrophic forgetting during reinforcement learning training.

## Design Philosophy

### Curriculum Sampling Strategy

Traditional curriculum learning uses fixed difficulty that increases over time:
```python
difficulty = min + progress * (max - min)  # WRONG: forgets easy cases
```

Our approach maintains the full range of easier difficulties:
```python
difficulty = uniform(min, min + progress * (max - min))  # CORRECT: no forgetting
```

This is essential for RL where the policy must remain competent across all difficulty levels.

### Designated Observer Concept

Each environment has exactly one "designated observer" - an agent that:
1. Has its gimbal pointing at the target (guaranteed)
2. Optionally has its body facing toward the target
3. Ensures at least one detection per reset

This provides a stable learning signal even at high curriculum difficulty.

---

## Module Architecture

### File Structure

```
initial_states/
├── __init__.py                    # Exports: InitialStatesCfg, InitialStatesResult, InitialStates
├── initial_states_cfg.py          # Configuration dataclasses
├── initial_states_generator.py    # Core generation algorithm
├── initial_states.py              # Thin wrapper for environment integration
├── doc/                           # Documentation
└── tests/                         # Unit tests (24 tests)
```

### Class Hierarchy

```
InitialStates (wrapper)
    └── InitialStatesGenerator (core logic)
            └── InitialStatesCfg (configuration)
                    └── InitialStatesResult (output)
```

---

## Configuration Parameters

### Cylinder Placement (`InitialStatesCfg`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cylinder_diameter_min` | 30.0 m | Minimum cylinder diameter at progress=0 |
| `cylinder_diameter_max` | 100.0 m | Maximum cylinder diameter at progress=1 |
| `cylinder_height_min` | 10.0 m | Minimum base height above ground |
| `cylinder_height_max` | 50.0 m | Maximum base height above ground |
| `cylinder_height_range` | 20.0 m | Vertical spread within cylinder |
| `agent_clearance` | 10.0 m | Minimum distance between agents |
| `max_placement_retries` | 50 | Max rejection sampling attempts |

### Target Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_distance_min` | 30.0 m | Minimum distance at progress=0 |
| `target_distance_max` | 200.0 m | Maximum distance at progress=1 |
| `target_height_offset_min` | -10.0 m | Height offset range (min) |
| `target_height_offset_max` | +10.0 m | Height offset range (max) |

### Velocity Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent_max_velocity` | 10.0 m/s | Maximum agent initial velocity |
| `agent_velocity_scale_max` | 1.0 | Scale at progress=1 |
| `target_max_velocity` | 10.0 m/s | Maximum target initial velocity |
| `target_velocity_scale_max` | 1.0 | Scale at progress=1 |
| `max_yaw_rate` | 45 deg/s | Maximum initial yaw rate |

### Gimbal Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gimbal_yaw_min/max` | [-180, +180] deg | Yaw joint limits |
| `gimbal_pitch_min/max` | [-45, +45] deg | Pitch joint limits |
| `gimbal_curriculum_mode` | "gradual" | Randomization mode |
| `gimbal_randomization_threshold` | 0.5 | Threshold for "threshold" mode |

**Gimbal Curriculum Modes:**
- `"gradual"`: With probability (1-progress), agent points at target
- `"threshold"`: Point until threshold, then random
- `"always_pointing"`: All agents always point at target (debugging)

### Zoom Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zoom_min/max` | [1.0, 30.0] | Zoom level limits |
| `zoom_initial_min/max` | [1.0, 10.0] | Initial sampling range |

### Designated Observer Selection

| Mode | Description |
|------|-------------|
| `"random"` | Random agent each reset (default) |
| `"fixed"` | Always agent 0 |
| `"rotating"` | Cycle through agents |

---

## Generation Algorithm (10 Steps)

### Step 1: Sample Curriculum-Controlled Parameters
```python
diameter = uniform(min, min + progress * (max - min))
target_distance = uniform(min, min + progress * (max - min))
velocity_scale = uniform(0, progress * scale_max)
```

### Step 2: Generate Cylinder Centers
- Random XY offset from env origin: [-5, +5] meters
- Random Z height: [height_min, height_max]

### Step 3: Place Agents in Cylinder
- Use rejection sampling with minimum `agent_clearance`
- Uniform distribution within cylinder disk
- Fallback to even angular distribution if sampling fails

### Step 4: Generate Target Position
- Random bearing from cylinder center
- Distance from curriculum sampling
- Height offset: uniform(-10, +10) meters from cylinder center

### Step 5: Generate Target Velocity
- Random unit direction
- Magnitude: velocity_scale × max_velocity
- No angular velocity

### Step 6: Select Designated Observer
- Per configured mode (random/fixed/rotating)

### Step 7: Generate Agent Orientations
- Designated observer: face target (with noise)
- Others: random yaw or face target (configurable)
- All agents: zero roll and pitch (level flight)

### Step 8: Generate Agent Velocities
- Random direction
- Magnitude: velocity_scale × max_velocity
- Angular velocity: yaw rate scaled by progress

### Step 9: Generate Gimbal States
- Designated observer: compute angles to point at target
- Others: based on curriculum mode
  - "gradual": probability (1-progress) to point at target
  - "threshold": point until threshold, then random
  - "always_pointing": always point at target

### Step 10: Generate Zoom Levels
- Random within [zoom_initial_min, zoom_initial_max]

---

## Gimbal Angle Computation

The gimbal angles are computed to point the camera at the target in body frame:

```python
def _compute_gimbal_angles_to_target(agent_pos, agent_quat, target_pos):
    # Direction to target in world frame
    dir_world = target_pos - agent_pos
    dir_world = normalize(dir_world)

    # Transform to body frame
    dir_body = quat_rotate_inverse(agent_quat, dir_world)

    # Extract gimbal angles
    yaw = atan2(dir_body.y, dir_body.x)
    xy_dist = sqrt(dir_body.x² + dir_body.y²)
    pitch = -atan2(dir_body.z, xy_dist)

    return yaw, pitch
```

**Important**: The `YAW_JOINT_OFFSET = -π/2` is applied when writing to joint state, not in this computation.

---

## Environment Integration

### Configuration (`iris_ma_env6_test_cfg.py`)

```python
from .initial_states import InitialStatesCfg

@configclass
class IrisMA6TestEnvCfg(DirectMARLEnvCfg):
    initial_states: InitialStatesCfg = InitialStatesCfg()
    enable_initial_states_randomization: bool = True
```

### Initialization (`iris_ma_env6_test.py::__init__`)

```python
if self.cfg.enable_initial_states_randomization:
    self._initial_states = InitialStates(
        cfg=self.cfg.initial_states,
        num_envs=self.num_envs,
        num_agents=len(self.cfg.possible_agents),
        device=self.device,
    )
else:
    self._initial_states = None

self._curriculum_progress = 0.0  # Updated during training
```

### Reset (`iris_ma_env6_test.py::_reset_idx`)

```python
def _reset_idx(self, env_ids: torch.Tensor):
    super()._reset_idx(env_ids)

    if self._initial_states is not None:
        result = self._initial_states.generate(
            env_ids=torch.arange(len(env_ids), device=self.device),
            curriculum_progress=self._curriculum_progress,
        )

        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]

            # Apply root pose (with terrain offset)
            agent_pos = result.agent_positions[:, idx] + self._terrain.env_origins[env_ids]
            agent_quat = result.agent_orientations[:, idx]
            robot.write_root_pose_to_sim(torch.cat([agent_pos, agent_quat], dim=-1), env_ids)

            # Apply root velocity
            robot.write_root_velocity_to_sim(torch.cat([
                result.agent_linear_velocities[:, idx],
                result.agent_angular_velocities[:, idx],
            ], dim=-1), env_ids)

            # Apply gimbal joints (with YAW_JOINT_OFFSET)
            gimbal_angles = result.gimbal_joint_positions[:, idx]
            joint_pos[:, yaw_idx] = gimbal_angles[:, 0] + YAW_JOINT_OFFSET
            joint_pos[:, roll_idx] = gimbal_angles[:, 1]
            joint_pos[:, pitch_idx] = gimbal_angles[:, 2]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            # Sync controller internal state
            self._controllers[agent_id].reset(env_ids)
            self._controllers[agent_id]._gimbal._yaw[env_ids] = gimbal_angles[:, 0]
            self._controllers[agent_id]._gimbal._pitch[env_ids] = gimbal_angles[:, 2]

            # Sync world-frame angles for LOS stabilization
            dir_to_target = result.target_positions - result.agent_positions[:, idx]
            azimuth_world = torch.atan2(dir_to_target[:, 1], dir_to_target[:, 0])
            xy_dist = torch.sqrt(dir_to_target[:, 0]**2 + dir_to_target[:, 1]**2)
            elevation_world = torch.atan2(dir_to_target[:, 2], xy_dist)
            self._controllers[agent_id]._gimbal._azimuth_world[env_ids] = azimuth_world
            self._controllers[agent_id]._gimbal._elevation_world[env_ids] = elevation_world

            # Set zoom level
            self.zoom_level[env_ids, idx] = result.zoom_levels[:, idx]

        # Apply target states
        target_pos = result.target_positions + self._terrain.env_origins[env_ids]
        self.target.write_root_pose_to_sim(torch.cat([target_pos, result.target_orientations], dim=-1), env_ids)
        self.target.write_root_velocity_to_sim(result.target_velocities, env_ids)
    else:
        self._reset_idx_hardcoded(env_ids)  # Fallback
```

---

## Critical Implementation Notes

### 1. YAW_JOINT_OFFSET

The gimbal yaw joint has a mesh rotation offset of `-π/2` radians. This offset is applied when writing to simulation, not in angle computation:

```python
joint_pos[:, yaw_idx] = gimbal_angles[:, 0] + YAW_JOINT_OFFSET  # -π/2
```

### 2. World-Frame vs Body-Frame Angles

The gimbal controller uses two coordinate systems:
- **Body-frame angles** (`_yaw`, `_pitch`): Gimbal position relative to drone body
- **World-frame angles** (`_azimuth_world`, `_elevation_world`): LOS direction in world frame

On reset, both must be synchronized:
- Body-frame: Set to computed gimbal angles
- World-frame: Computed from agent-to-target direction (NOT from body-frame angles)

### 3. Rejection Sampling Fallback

With 3 agents and 10m clearance in a 30m diameter cylinder, rejection sampling usually succeeds. If it fails (> max_retries), agents are placed evenly around the cylinder at 70% radius.

### 4. Curriculum Progress Update

The `_curriculum_progress` variable must be updated externally (e.g., by curriculum manager or training script):

```python
env._curriculum_progress = 0.5  # Set to training progress [0, 1]
```

---

## Curriculum Schedule Example

With default parameters:

| Progress | Diameter Range | Target Distance | Velocity Range | Gimbal Pointing Prob |
|----------|----------------|-----------------|----------------|----------------------|
| 0.0 | [30, 30] m | [30, 30] m | [0, 0] m/s | 100% |
| 0.25 | [30, 47.5] m | [30, 72.5] m | [0, 2.5] m/s | 75% |
| 0.5 | [30, 65] m | [30, 115] m | [0, 5] m/s | 50% |
| 0.75 | [30, 82.5] m | [30, 157.5] m | [0, 7.5] m/s | 25% |
| 1.0 | [30, 100] m | [30, 200] m | [0, 10] m/s | 0% (observer only) |

---

## Testing

Run the test suite:
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/initial_states/tests/run_tests.py
```

All 24 tests should pass:
- **Configuration tests** (2): Default and custom values
- **Generator tests** (7): Initialization, generation, bounds validation
- **Designated observer tests** (3): Selection modes, gimbal pointing accuracy
- **Gimbal curriculum tests** (2): Progress-based randomization
- **Velocity tests** (4): Magnitude bounds, curriculum scaling
- **Zoom tests** (2): Range validation, variation
- **Integration tests** (4): Wrapper, partial env_ids, config updates

---

## Common Issues

### Agents Not Pointing at Target
**Cause**: World-frame angles not synchronized on reset.
**Fix**: Compute azimuth/elevation from agent-to-target direction, not from body-frame gimbal angles.

### Agents Too Close
**Cause**: Small cylinder diameter with large clearance requirement.
**Fix**: Increase `cylinder_diameter_min` or decrease `agent_clearance`.

### Target Too Far
**Cause**: Large `target_distance_max` value.
**Fix**: Adjust distance parameters for camera/detection range.

### NaN Values
**Cause**: Division by zero in direction normalization.
**Fix**: All direction computations include epsilon: `norm + 1e-8`.
