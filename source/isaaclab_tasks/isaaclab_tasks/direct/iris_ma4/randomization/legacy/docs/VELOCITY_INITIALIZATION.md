# Velocity Initialization in Formation Generator

## Overview
The formation generator now initializes agents with **small random velocities** instead of zero velocities. This provides more realistic initial conditions and helps with training dynamics.

## Implementation

### Velocity Ranges

**Linear Velocity** (x, y, z components):
```python
range = [-max_lin_vel/5 * scale_factor, max_lin_vel/5 * scale_factor]
```

**Yaw Rate** (rotation around Z-axis):
```python
range = [-max_yaw_rate * scale_factor, max_yaw_rate * scale_factor]
```

**Roll/Pitch Rates**: Set to zero (agents maintain level flight)

### Code Location
See [initial_states.py:162-173](source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/randomization/initial_states.py#L162-L173)

```python
# Linear velocity (indices 7:10)
root_states[type_envs, :, 7:10] = math_utils.sample_uniform(
    lower=-self.iris_cfg.max_lin_vel/5 * scale_factor,
    upper=self.iris_cfg.max_lin_vel/5 * scale_factor,
    size=(len(type_envs), num_agents, 3),
    device=self.device
)

# Yaw rate only (index 12)
root_states[type_envs, :, 12:13] = math_utils.sample_uniform(
    lower=-self.iris_cfg.max_yaw_rate * scale_factor,
    upper=self.iris_cfg.max_yaw_rate * scale_factor,
    size=(len(type_envs), num_agents, 1),
    device=self.device
)
```

## Rationale

### Why Small Initial Velocities?

1. **Realism**: Real drones rarely start from perfect hovering state
2. **Training robustness**: Agents learn to handle non-zero initial conditions
3. **Curriculum support**: Velocity magnitude scales with `scale_factor`
4. **Exploration**: Adds diversity to initial states for better coverage

### Why 1/5 of Max Velocity?

- **Safety**: Small enough to avoid immediate collisions
- **Stability**: Doesn't overwhelm control during first timesteps
- **Learning**: Provides gentle perturbation without chaos
- **Tunable**: Can be adjusted via curriculum `scale_factor`

## Validation

The `_validate_formation()` method now checks velocity bounds:

```python
def _validate_formation(self, root_states: torch.Tensor):
    # ... position checks ...

    # Check velocities are within expected bounds
    lin_vel = velocities[:, :, 0:3]
    ang_vel = velocities[:, :, 3:6]

    max_lin_vel = lin_vel.abs().max()
    max_ang_vel = ang_vel.abs().max()

    expected_max_lin = self.iris_cfg.max_lin_vel / 5
    expected_max_ang = self.iris_cfg.max_yaw_rate

    if max_lin_vel > expected_max_lin * 1.1:  # 10% tolerance
        print(f"Warning: Linear velocity too high: {max_lin_vel:.2f}")
```

## Test Results

From [test_velocity_bounds.py](test_velocity_bounds.py):

### Scale Factor = 0.5
```
Linear Velocity:
  Expected range: [-0.5000, 0.5000]
  Actual range:   [-0.4996, 0.4987]
  Mean: -0.0057 (centered around zero ✓)
  Std:  0.2932

Yaw Rate:
  Expected range: [-0.5000, 0.5000]
  Actual range:   [-0.4966, 0.4982]
  Mean: -0.0051 (centered around zero ✓)
```

### Scale Factor = 1.0
```
Linear Velocity:
  Expected range: [-1.0000, 1.0000]
  Actual range:   [-0.9897, 0.9995]
  Std:  0.5794

Yaw Rate:
  Expected range: [-1.0000, 1.0000]
  Actual range:   [-0.9959, 0.9972]
  Std:  0.5606
```

### Scale Factor = 2.0
```
Linear Velocity:
  Expected range: [-2.0000, 2.0000]
  Actual range:   [-1.9957, 1.9865]
  Std:  1.1535

Yaw Rate:
  Expected range: [-2.0000, 2.0000]
  Actual range:   [-1.9974, 1.9921]
  Std:  1.1637
```

### Scaling Verification
```
Standard deviation ratios (should be ~2.0):
  std(1.0) / std(0.5) = 1.95 ✓
  std(2.0) / std(1.0) = 2.02 ✓
```

## Curriculum Integration (Future)

When curriculum module is ready:

```python
# Early training: Small velocities
scale = 0.5
formation = randomizer.get_random_formation(
    num_agents=3,
    scale_factor=scale  # velocities: ±0.5 m/s linear, ±0.5 rad/s yaw
)

# Late training: Larger velocities
scale = 2.0
formation = randomizer.get_random_formation(
    num_agents=3,
    scale_factor=scale  # velocities: ±2.0 m/s linear, ±2.0 rad/s yaw
)
```

## State Vector Structure

Complete root state tensor `[num_envs, num_agents, 13]`:

| Indices | Content | Initialization |
|---------|---------|----------------|
| 0:3 | Position (x, y, z) | Random formation pattern |
| 3:7 | Orientation (quat w,x,y,z) | Formation-aligned or random |
| 7:10 | Linear velocity (vx, vy, vz) | **Uniform random (scaled)** |
| 10:13 | Angular velocity (wx, wy, wz) | **wz random (scaled), wx=wy=0** |

## Example Output

For a 3-agent formation with `scale_factor=1.0` and `max_lin_vel=5.0 m/s`:

```python
# Agent 0
position:     [2.3, -5.1, 18.5]
orientation:  [1.0, 0.0, 0.0, 0.0]  # facing +X
linear_vel:   [0.87, -0.34, 0.12]   # ±1.0 m/s range
angular_vel:  [0.0, 0.0, -0.42]     # yaw rate ±1.0 rad/s

# Agent 1
position:     [2.8, 1.2, 19.3]
orientation:  [0.998, 0.0, 0.0, 0.062]  # slight yaw noise
linear_vel:   [-0.45, 0.78, -0.23]
angular_vel:  [0.0, 0.0, 0.15]
```

## Comparison to Previous Implementation

| Aspect | Old (iris_ma2) | New (iris_ma3) |
|--------|----------------|----------------|
| **Linear velocity** | All zeros | Random ±(max_vel/5 × scale) |
| **Angular velocity** | All zeros | Yaw rate random ±(max_yaw × scale) |
| **Curriculum support** | N/A | Via `scale_factor` parameter |
| **Validation** | None | Automatic bounds checking |
| **Realism** | Static start | Dynamic start |

## Benefits for Training

1. **Faster convergence**: Agents don't need to learn "how to move from rest"
2. **Better generalization**: Experience diverse initial conditions from episode 1
3. **Robust policies**: Trained on perturbations from the start
4. **Curriculum-friendly**: Smoothly increase difficulty via `scale_factor`
5. **Sim-to-real transfer**: More realistic initial conditions

## Notes

- Roll and pitch rates are kept at zero to maintain level flight attitude
- Velocities are uniformly distributed (not Gaussian) for bounded support
- The `/5` factor for linear velocity is conservative; can be adjusted if needed
- Yaw rate uses full `max_yaw_rate` range (no division by 5)
- All velocities scale linearly with `scale_factor` for curriculum compatibility
