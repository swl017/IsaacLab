# Point Mass Controller API Reference

## Overview

The `PointMass` controller provides quaternion-based attitude and velocity control for quadcopters in Isaac Lab environments. It implements exact quaternion mathematics to avoid gimbal lock and provides robust control across the full SO(3) rotation space.

**Module:** `isaaclab_tasks.direct.iris_ma3.controller.point_mass`

---

## Class: `PointMass`

### Constructor

```python
PointMass(
    mass: float = 1.0,
    weight: float | None = None,
    num_envs: int = 1,
    disable_gravity: bool = True,
    device: str = 'cuda:0'
)
```

**Parameters:**

- **mass** (`float`, default: `1.0`)
  Mass of the robot in kilograms.

- **weight** (`float | None`, default: `None`)
  Weight (force) in Newtons. If `None`, calculated as `mass * 9.81`.

- **num_envs** (`int`, default: `1`)
  Number of parallel environments for vectorized computation.

- **disable_gravity** (`bool`, default: `True`)
  If `True`, gravity compensation is NOT applied. If `False`, gravity is compensated in the body frame.

- **device** (`str`, default: `'cuda:0'`)
  PyTorch device for tensor operations (`'cuda:0'`, `'cpu'`, etc.).

**Example:**
```python
from isaaclab_tasks.direct.iris_ma3.controller.point_mass import PointMass

controller = PointMass(
    mass=1.619,
    num_envs=16,
    disable_gravity=False,
    device='cuda:0'
)
```

---

## Methods

### `compute_control_quat_exact`

```python
compute_control_quat_exact(
    cmd_lin_vel_w: torch.Tensor,
    cmd_yaw_vel: torch.Tensor,
    curr_quat_w: torch.Tensor,
    curr_lin_vel_w: torch.Tensor,
    curr_ang_vel_b: torch.Tensor,
    curr_lin_acc_b: torch.Tensor,
    dt: float = 0.02,
    gains: dict = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

**Primary control method** using exact quaternion logarithm map for attitude error computation.

**Parameters:**

- **cmd_lin_vel_w** (`torch.Tensor`, shape: `(N, 3)`)
  Commanded linear velocity in **world frame** (m/s).

- **cmd_yaw_vel** (`torch.Tensor`, shape: `(N,)` or `(N, 1)`)
  Commanded yaw angular velocity (rad/s).

- **curr_quat_w** (`torch.Tensor`, shape: `(N, 4)`)
  Current orientation quaternion in **world frame**, format: `(w, x, y, z)`.

- **curr_lin_vel_w** (`torch.Tensor`, shape: `(N, 3)`)
  Current linear velocity in **world frame** (m/s).

- **curr_ang_vel_b** (`torch.Tensor`, shape: `(N, 3)`)
  Current angular velocity in **body frame** (rad/s).

- **curr_lin_acc_b** (`torch.Tensor`, shape: `(N, 3)`)
  Current linear acceleration in **body frame** (m/s²).

- **dt** (`float`, default: `0.02`)
  Time step for integration (seconds). Used for integral term accumulation.

- **gains** (`dict | None`, default: `None`)
  Custom control gains. If `None`, uses `self.default_gains`. See [Gain Parameters](#gain-parameters).

**Returns:**

- **force** (`torch.Tensor`, shape: `(N, 3)`)
  Control force in **body frame** (N).

- **moment** (`torch.Tensor`, shape: `(N, 3)`)
  Control moment in **body frame** (N·m).

**Mathematical Approach:**

This method uses the **exact quaternion logarithm map**:

- For quaternion error `q_error = q_desired * q_current^(-1)`:
  - Exact rotation vector: `error = 2 * θ * axis / sin(θ/2)`
  - Where `θ = 2 * acos(w)` is the rotation angle
  - Handles singularities via Taylor series for small angles

**Control Structure:**

1. **Attitude Control (Hybrid):**
   - Roll/Pitch: Position control (target 0° for level flight)
   - Yaw: Rate tracking (follows commanded yaw rate)

2. **Velocity Control:**
   - PID control with integral anti-windup
   - Gravity compensation (if enabled)
   - Thrust saturation

**Example:**
```python
force, moment = controller.compute_control_quat_exact(
    cmd_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]], device='cuda:0'),  # 1 m/s forward
    cmd_yaw_vel=torch.tensor([0.0], device='cuda:0'),                 # No yaw rotation
    curr_quat_w=robot.data.root_quat_w,
    curr_lin_vel_w=robot.data.root_lin_vel_w,
    curr_ang_vel_b=robot.data.root_ang_vel_b,
    curr_lin_acc_b=robot.data.root_lin_acc_b,
    dt=0.02,
)
```

---

### `compute_control`

```python
compute_control(
    cmd_lin_vel_w: torch.Tensor,
    cmd_yaw_vel: torch.Tensor,
    curr_quat_w: torch.Tensor,
    curr_lin_vel_w: torch.Tensor,
    curr_ang_vel_b: torch.Tensor,
    curr_lin_acc_b: torch.Tensor,
    dt: float = 0.02,
    gains: dict = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

**Alternative control method** using small-angle quaternion approximation.

**Parameters:** Same as [`compute_control_quat_exact`](#compute_control_quat_exact).

**Returns:** Same as [`compute_control_quat_exact`](#compute_control_quat_exact).

**Differences from `compute_control_quat_exact`:**

- Uses **small-angle approximation**: `error ≈ 2 * [x, y, z]` components
- Maintains current yaw (extracts and preserves yaw from current orientation)
- Faster computation but less accurate for large angles
- Valid for angles < ~15-20°

**When to use:**
- High-frequency control loops where performance is critical
- Small expected attitude errors
- When exact quaternion math overhead is not justified

---

### `reset_integral`

```python
reset_integral(env_ids: torch.Tensor = None)
```

Reset velocity error integral state for specified environments.

**Parameters:**

- **env_ids** (`torch.Tensor | None`, default: `None`)
  Indices of environments to reset. If `None`, resets all environments.

**Usage:**

Call this method when environments are reset to prevent integral windup from carrying over between episodes.

**Example:**
```python
# Reset all environments
controller.reset_integral()

# Reset specific environments
env_ids = torch.tensor([0, 3, 7], device='cuda:0')
controller.reset_integral(env_ids)
```

---

### `reset`

```python
reset(env_ids=None)
```

Reset controller state for specified environments.

**Parameters:**

- **env_ids** (`torch.Tensor | None`, default: `None`)
  Indices of environments to reset. If `None`, resets all environments.

**Internal State Reset:**
- Clears desired yaw tracking state (used in `compute_control`)
- Does NOT reset integral state (use `reset_integral` for that)

---

### `wrap_to_pi`

```python
wrap_to_pi(angle: torch.Tensor) -> torch.Tensor
```

Wrap angles to the range `[-π, π]`.

**Parameters:**

- **angle** (`torch.Tensor`)
  Input angles in radians (any shape).

**Returns:**

- **wrapped** (`torch.Tensor`)
  Wrapped angles in `[-π, π]` (same shape as input).

**Example:**
```python
angles = torch.tensor([3.5, -4.0, 0.5], device='cuda:0')
wrapped = controller.wrap_to_pi(angles)
# Result: tensor([-2.7832, -0.8584,  0.5000])
```

---

## Gain Parameters

### Default Gains

The controller uses the following default gains (stored in `self.default_gains`):

```python
{
    # Velocity Control (PID)
    "kp_x": 3.0,   # Position proportional gain (all axes)
    "ki_x": 0.5,   # Integral gain (velocity tracking)
    "kd_x": 0.1,   # Derivative gain (acceleration damping)
    "kp_y": 3.0,   # Y-axis proportional gain
    "kd_y": 0.1,   # Y-axis derivative gain
    "kp_z": 2.0,   # Z-axis proportional gain

    # Attitude Control (PD)
    "kp_att": 20.0,  # Roll/pitch proportional gain
    "kd_att": 3.0,   # Roll/pitch derivative gain
    "kp_yaw": 1.2,   # Yaw rate proportional gain
    "kd_yaw": 0.5,   # Yaw rate derivative gain (deprecated)
}
```

### Custom Gains

Override default gains by passing a custom dictionary:

```python
custom_gains = {
    "kp_att": torch.tensor([25.0], device='cuda:0').expand(num_envs),
    "kd_att": torch.tensor([4.0], device='cuda:0').expand(num_envs),
    "kp_x": torch.tensor([4.0], device='cuda:0').expand(num_envs),
    # ... other gains ...
}

force, moment = controller.compute_control_quat_exact(
    # ... state parameters ...
    gains=custom_gains
)
```

**Gain Tuning Guidelines:**

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `kp_att` | 15.0 - 30.0 | Higher → Faster attitude response, may cause oscillation |
| `kd_att` | 2.0 - 5.0 | Higher → More damping, smoother but slower |
| `kd_att / kp_att` | 0.15 - 0.25 | Critical damping ratio for stability |
| `kp_x` | 2.0 - 5.0 | Higher → Faster velocity tracking |
| `ki_x` | 0.3 - 1.0 | Higher → Eliminates steady-state error, may cause windup |
| `kp_yaw` | 1.0 - 2.0 | Yaw rate tracking responsiveness |

---

## Attributes

### Control Limits

- **`max_moment`** (`torch.Tensor`, shape: `(N, 3)`)
  Maximum moment limits `[roll, pitch, yaw]` in N·m.
  Default: `[5.0, 5.0, 2.0]`

- **`_integral_max`** (`float`)
  Maximum integral accumulation in N·s.
  Default: `5.0`

### State

- **`_vel_error_integral`** (`torch.Tensor`, shape: `(N, 3)`)
  Accumulated velocity error integral for anti-windup PID control.

- **`weight_tensor`** (`torch.Tensor`, shape: `(N, 3)`)
  Gravity force vector `[0, 0, -weight]` in world frame.

---

## Coordinate Frames

**World Frame (Inertial):**
- X: Forward (North)
- Y: Left (West)
- Z: Up

**Body Frame:**
- X: Forward (nose direction)
- Y: Left (left wing)
- Z: Up (through rotors)

**Quaternion Convention:**
- Format: `(w, x, y, z)` where `w` is the scalar part
- Represents rotation from world frame to body frame
- Unit quaternion: `w² + x² + y² + z² = 1`

---

## Error Handling

The controller handles numerical edge cases:

1. **Quaternion Normalization:** Not enforced (assumes input quaternions are unit)
2. **Small Angle Singularity:** Taylor series approximation for `θ < 1e-4`
3. **Integral Windup:** Clamped to `±_integral_max` with back-calculation anti-windup
4. **Thrust Saturation:** Z-axis force clamped to `[-1.0 * weight, 3.0 * weight]`

---

## Performance Considerations

- **Vectorized Operations:** All computations are batched across `num_envs`
- **GPU Acceleration:** Native PyTorch on CUDA for minimal overhead
- **Computational Cost:**
  - `compute_control`: ~0.1-0.2 ms per step (16 envs on RTX 3090)
  - `compute_control_quat_exact`: ~0.2-0.3 ms per step (exact math overhead)

---

## Version History

- **v1.0** (Current): Exact quaternion logarithm map, hybrid attitude control
- **v0.9**: Small-angle quaternion approximation
- **v0.8**: Initial implementation with Euler angles

---

## See Also

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Integration examples and tutorials
- [Isaac Lab Math Utils](https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.math) - Quaternion utilities
