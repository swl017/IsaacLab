# Formation Generator Usage Guide

## Overview
The `InitialStatesRandomizer.get_random_formation()` method generates random formations for multi-agent systems with guaranteed minimum separation and configurable patterns.

## Basic API

```python
root_states = randomizer.get_random_formation(
    num_agents=len(self.cfg.possible_agents),
    num_envs=None,  # defaults to self.num_envs
    formation_type=None,  # "line", "grid", or None for random
    scale_factor=1.0,  # future curriculum hook
    env_ids=None  # specific env indices or None for all
)

# Returns: torch.Tensor [num_envs, num_agents, 13]
#   [:, :, 0:3]  - positions (x, y, z)
#   [:, :, 3:7]  - orientations (quaternion w, x, y, z)
#   [:, :, 7:13] - velocities (all zeros)
```

## Integration Example: Environment Reset

### Example 1: Reset All Agents in Formation

```python
def _reset_idx(self, env_ids: torch.Tensor):
    """Reset specific environments."""

    # Generate formation for resetting environments
    new_robot_states = self.randomizer.initial_states.get_random_formation(
        num_agents=len(self.cfg.possible_agents),
        env_ids=env_ids
    )

    # Apply to each robot
    for i, agent_id in enumerate(self.cfg.possible_agents):
        robot = self._robots[agent_id]

        # Extract state for this agent
        agent_root_state = new_robot_states[:, i, :]  # [len(env_ids), 13]

        # Write positions and orientations
        robot.write_root_pose_to_sim(agent_root_state[:, 0:7], env_ids)

        # Write velocities (zeros)
        robot.write_root_velocity_to_sim(agent_root_state[:, 7:13], env_ids)
```

### Example 2: With Additional Customization

```python
def _reset_idx(self, env_ids: torch.Tensor):
    """Reset with custom gimbal initialization."""

    # Get formation
    formation_type = "line" if self.training_phase == 1 else "grid"
    scale = self.curriculum.get_formation_scale()  # future

    new_robot_states = self.randomizer.initial_states.get_random_formation(
        num_agents=len(self.cfg.possible_agents),
        formation_type=formation_type,
        scale_factor=scale,
        env_ids=env_ids
    )

    # Get target position for gimbal initialization
    target_positions = self.target.data.root_pos_w[env_ids]

    for i, agent_id in enumerate(self.cfg.possible_agents):
        robot = self._robots[agent_id]
        agent_state = new_robot_states[:, i, :]

        # Set root state
        robot.write_root_pose_to_sim(agent_state[:, 0:7], env_ids)
        robot.write_root_velocity_to_sim(agent_state[:, 7:13], env_ids)

        # Initialize gimbal to point at target
        gimbal_yaw, gimbal_pitch = self.point_to_region(
            agent_state[:, 0:3],  # robot position
            agent_state[:, 3:7],  # robot orientation
            target_positions
        )

        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]] = gimbal_yaw
        joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]] = gimbal_pitch

        robot.write_joint_state_to_sim(joint_pos, None, None, env_ids)
```

### Example 3: Split Formation by Agent Role

```python
def _reset_idx(self, env_ids: torch.Tensor):
    """Different formation for leader vs followers."""

    # Generate base formation
    formation = self.randomizer.initial_states.get_random_formation(
        num_agents=len(self.cfg.possible_agents),
        env_ids=env_ids
    )

    for i, agent_id in enumerate(self.cfg.possible_agents):
        robot = self._robots[agent_id]
        agent_state = formation[:, i, :].clone()

        # Leader stays in formation center
        if agent_id == "agent_0":
            formation_center = formation[:, :, 0:3].mean(dim=1)  # [len(env_ids), 3]
            agent_state[:, 0:3] = formation_center

        # Apply state
        robot.write_root_pose_to_sim(agent_state[:, 0:7], env_ids)
        robot.write_root_velocity_to_sim(agent_state[:, 7:13], env_ids)
```

## Configuration

Modify formation behavior via config:

```python
from isaaclab_tasks.direct.iris_ma3.randomization import InitialStatesRandomizerCfg

# In your environment config
randomizer_cfg = InitialStatesRandomizerCfg(
    # Formation types available for random selection
    formation_types=["line", "grid"],
    default_formation_type="random",  # or specify "line"/"grid"

    # Agent separation constraints
    min_agent_separation=3.0,  # minimum distance between agents
    max_agent_separation=12.0,  # maximum formation spread

    # Formation center bounds (where formation spawns)
    formation_center_bounds={
        "x_min": -5.0, "x_max": 5.0,
        "y_min": -10.0, "y_max": 10.0,
        "z_min": 15.0, "z_max": 20.0,
    },

    # Vertical variation within formation
    allow_vertical_variation=True,
    z_variation_range=(0.0, 5.0),

    # Agent orientations
    orientation_mode="formation_aligned",  # or "random_yaw"
    orientation_noise_std=0.087,  # ~5 degrees

    # Formation rotation range
    orientation_bounds={
        "yaw_min": -3.14,
        "yaw_max": 3.14,
    },
)
```

## Formation Types

### Line Formation
- Agents arranged along arbitrary 3D line (not just Y-axis)
- Random distances between agents
- Line direction randomized per environment
- Respects minimum height constraints

**Best for**: Surveillance, perimeter patrol, sweep operations

### Grid Formation
- 2D grid in XY plane
- Automatically determines rows/cols from num_agents
- Random spacing per environment
- Optional Z-axis variation

**Best for**: Area coverage, search tasks, formation flying

## Advanced Usage

### Future Curriculum Integration

```python
# In your curriculum module (future)
class FormationCurriculum:
    def get_formation_config(self, progress: float):
        """Return formation parameters based on training progress."""
        return {
            'formation_type': 'line' if progress < 0.5 else 'grid',
            'scale_factor': 0.5 + 1.5 * progress,  # 0.5 -> 2.0
        }

# In environment
def _reset_idx(self, env_ids: torch.Tensor):
    progress = self.curriculum.get_progress()
    config = self.curriculum.get_formation_config(progress)

    formation = self.randomizer.initial_states.get_random_formation(
        num_agents=len(self.cfg.possible_agents),
        formation_type=config['formation_type'],
        scale_factor=config['scale_factor'],
        env_ids=env_ids
    )
    # ... apply formation
```

### Debugging & Visualization

```python
# Print formation info
formation = self.randomizer.initial_states.get_random_formation(...)
positions = formation[:, :, 0:3]

print(f"Formation center: {positions.mean(dim=1)[0]}")
print(f"Formation spread: {positions.std(dim=1)[0]}")

# Check distances
for env_idx in [0]:  # first env
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            dist = (positions[env_idx, i] - positions[env_idx, j]).norm()
            print(f"Distance agent_{i} to agent_{j}: {dist:.2f}m")
```

## Key Features

✅ **Fully vectorized**: Generates formations for all environments in parallel
✅ **Collision-free**: Guaranteed minimum separation between agents
✅ **Curriculum-ready**: `scale_factor` parameter for future integration
✅ **Flexible**: Line and grid patterns with random variations
✅ **Safe**: Validates ground clearance and boundaries
✅ **Clean API**: Single call returns complete robot states

## Comparison to Previous Implementation

| Aspect | Old (iris_ma2) | New (iris_ma3) |
|--------|----------------|----------------|
| **Formation variety** | Single alternating pattern | Line + Grid with variations |
| **Agent spacing** | Hard-coded | Randomized per environment |
| **Scalability** | Fixed agent count | Works for any num_agents |
| **API** | Inline calculation in reset | Clean method call |
| **Collision safety** | None | Enforced with retries |
| **Code reuse** | Copy-paste needed | Centralized, reusable |
| **Testing** | Manual | Automated test suite |
| **Curriculum** | Tangled logic | Clean hook via scale_factor |
