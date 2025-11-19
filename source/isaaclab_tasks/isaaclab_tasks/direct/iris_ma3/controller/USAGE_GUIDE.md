# Point Mass Controller Usage Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Integration with Isaac Lab](#integration-with-isaac-lab)
3. [Common Usage Patterns](#common-usage-patterns)
4. [Gain Tuning](#gain-tuning)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Basic Setup

```python
import torch
from isaaclab_tasks.direct.iris_ma3.controller.point_mass import PointMass

# Create controller for 16 parallel environments
controller = PointMass(
    mass=1.619,           # Iris quadcopter mass (kg)
    num_envs=16,
    disable_gravity=False,  # Enable gravity compensation
    device='cuda:0'
)

# Get robot state from Isaac Lab environment
robot = env.scene["robot"]

# Compute control commands
force, moment = controller.compute_control_quat_exact(
    cmd_lin_vel_w=actions[:, :3],           # Commanded velocity from RL policy
    cmd_yaw_vel=actions[:, 3],              # Commanded yaw rate
    curr_quat_w=robot.data.root_quat_w,     # Current orientation
    curr_lin_vel_w=robot.data.root_lin_vel_w,
    curr_ang_vel_b=robot.data.root_ang_vel_b,
    curr_lin_acc_b=robot.data.root_lin_acc_b,
    dt=env.step_dt,
)

# Apply forces to simulation (via PhysX)
robot.root_physx_view.apply_forces_and_torques_at_position(
    force_data=force_all_bodies,   # See "Force Application" section
    torque_data=moment_all_bodies,
    indices=robot._ALL_INDICES,
    is_global=False,  # Body frame
)
```

---

## Integration with Isaac Lab

### Complete Environment Integration

This example shows full integration with a `DirectRLEnv`:

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.iris_ma3.controller.point_mass import PointMass

class QuadcopterEnv(DirectRLEnv):
    def __init__(self, cfg: DirectRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Initialize controller
        self._robot_controller = PointMass(
            mass=self._robot_mass,
            num_envs=self.num_envs,
            disable_gravity=False,
            device=self.device
        )

    def _setup_scene(self):
        # Add robot to scene
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Get robot mass from USD
        self._robot_mass = self._robot.root_physx_view.get_masses()[0, 0].item()

        # Cache body ID for force application
        self._body_id = self._robot.find_bodies("body")[0][0]

        super()._setup_scene()

    def _apply_action(self):
        """Apply control forces based on commanded actions."""
        # Actions: [vx, vy, vz, yaw_rate]
        cmd_vel = self._actions[:, :3]  # Linear velocity
        cmd_yaw_rate = self._actions[:, 3]

        # Compute control forces/moments
        force, moment = self._robot_controller.compute_control_quat_exact(
            cmd_lin_vel_w=cmd_vel,
            cmd_yaw_vel=cmd_yaw_rate,
            curr_quat_w=self._robot.data.root_quat_w,
            curr_lin_vel_w=self._robot.data.root_lin_vel_w,
            curr_ang_vel_b=self._robot.data.root_ang_vel_b,
            curr_lin_acc_b=self._robot.data.root_lin_acc_b,
            dt=self.step_dt,
        )

        # Apply to PhysX (see Force Application section)
        self._apply_forces_to_physx(force, moment)

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        super()._reset_idx(env_ids)

        # Reset controller integral state
        self._robot_controller.reset_integral(env_ids)
```

---

## Common Usage Patterns

### 1. Velocity Tracking

Track desired velocities while maintaining level attitude:

```python
# Hover in place
cmd_vel = torch.zeros((num_envs, 3), device='cuda:0')
cmd_yaw_rate = torch.zeros(num_envs, device='cuda:0')

# Move forward at 2 m/s
cmd_vel = torch.tensor([[2.0, 0.0, 0.0]], device='cuda:0').expand(num_envs, 3)

# Climb at 1 m/s while moving forward
cmd_vel = torch.tensor([[1.0, 0.0, 1.0]], device='cuda:0').expand(num_envs, 3)

# Yaw rotation at 30°/s (0.52 rad/s)
cmd_yaw_rate = torch.full((num_envs,), 0.52, device='cuda:0')

force, moment = controller.compute_control_quat_exact(
    cmd_lin_vel_w=cmd_vel,
    cmd_yaw_vel=cmd_yaw_rate,
    # ... other parameters ...
)
```

### 2. Custom Gains Per Environment

Use different gains for different environments (e.g., domain randomization):

```python
# Create per-environment gains
custom_gains = {
    "kp_att": torch.tensor([20.0, 25.0, 18.0, 22.0], device='cuda:0'),  # 4 envs
    "kd_att": torch.tensor([3.0, 3.5, 2.8, 3.2], device='cuda:0'),
    # ... other gains with shape (num_envs,) ...
}

force, moment = controller.compute_control_quat_exact(
    # ... state parameters ...
    gains=custom_gains
)
```

### 3. Force Application to PhysX

**Critical:** PhysX requires force/torque data for ALL bodies, not just the base:

```python
def _apply_forces_to_physx(self, force, moment):
    """Apply control forces to PhysX articulation.

    Args:
        force: (num_envs, 3) - Force in body frame
        moment: (num_envs, 3) - Moment in body frame
    """
    num_bodies = self._robot.num_bodies  # e.g., 5 (base + 4 rotors)
    num_envs = self.num_envs

    # Create arrays for all bodies (num_envs * num_bodies, 3)
    all_forces = torch.zeros((num_envs * num_bodies, 3), device=self.device)
    all_moments = torch.zeros((num_envs * num_bodies, 3), device=self.device)

    # Fill in forces/moments for base body only (indices 0, 5, 10, 15, ...)
    base_body_indices = torch.arange(num_envs, device=self.device) * num_bodies
    all_forces[base_body_indices] = force
    all_moments[base_body_indices] = moment

    # Apply to PhysX
    self._robot.root_physx_view.apply_forces_and_torques_at_position(
        force_data=all_forces,
        torque_data=all_moments,
        position_data=None,  # Apply at center of mass
        indices=self._robot._ALL_INDICES,
        is_global=False,  # Body frame
    )
```

### 4. Observation Collection

Collect controller-relevant observations for RL:

```python
def _get_observations(self) -> dict:
    obs = {
        # State observations
        "robot_quat": self._robot.data.root_quat_w,           # (N, 4)
        "robot_lin_vel": self._robot.data.root_lin_vel_w,     # (N, 3)
        "robot_ang_vel": self._robot.data.root_ang_vel_b,     # (N, 3)

        # Velocity error (useful for RL policy)
        "vel_error": self._cmd_vel - self._robot.data.root_lin_vel_w,

        # Attitude error (compute from quaternion)
        "attitude_error": self._compute_attitude_error(),
    }
    return obs

def _compute_attitude_error(self):
    """Helper to compute attitude error for observations."""
    from isaaclab.utils.math import euler_xyz_from_quat

    roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)

    # Error from level (0° roll/pitch)
    roll_error = -roll  # Negative because we want 0
    pitch_error = -pitch

    return torch.stack([roll_error, pitch_error, yaw], dim=1)
```

---

## Gain Tuning

### Step-by-Step Tuning Process

**1. Start with Conservative Gains**

```python
controller = PointMass(mass=1.619, num_envs=1, device='cuda:0')

# Override with very conservative gains
conservative_gains = {
    "kp_att": torch.tensor([5.0], device='cuda:0'),
    "kd_att": torch.tensor([1.0], device='cuda:0'),
    "kp_x": torch.tensor([1.0], device='cuda:0'),
    "kd_x": torch.tensor([0.05], device='cuda:0'),
}
```

**2. Tune Attitude Control First**

Increase `kp_att` until you see oscillation, then back off 20%:

```python
# Test sequence
kp_att_values = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

for kp in kp_att_values:
    gains["kp_att"] = torch.tensor([kp], device='cuda:0')
    gains["kd_att"] = torch.tensor([kp * 0.15], device='cuda:0')  # Maintain ratio

    # Run test and observe stability
    # If oscillation appears, use previous value
```

**3. Tune Velocity Control**

Once attitude is stable, tune velocity tracking:

```python
# Increase kp_x until velocity tracking is responsive
kp_x_values = [1.0, 2.0, 3.0, 4.0, 5.0]

for kp in kp_x_values:
    gains["kp_x"] = torch.tensor([kp], device='cuda:0')
    gains["kd_x"] = torch.tensor([kp * 0.03], device='cuda:0')

    # Test step response to velocity command
```

**4. Add Integral Term (If Needed)**

For steady-state error elimination:

```python
# Start with small ki_x
gains["ki_x"] = torch.tensor([0.2], device='cuda:0')

# Increase gradually if steady-state error persists
# Watch for integral windup (oscillation or overshoot)
```

### Gain Stability Criteria

| Symptom | Solution |
|---------|----------|
| High-frequency oscillation | Reduce `kp_att`, increase `kd_att` |
| Slow response | Increase `kp_att`, maintain `kd_att / kp_att` ratio |
| Overshooting | Reduce `kp_x`, increase `kd_x` |
| Steady-state error | Increase `ki_x` (check anti-windup) |
| Yaw drift | Increase `kp_yaw` |

---

## Troubleshooting

### Problem: Drone Flips on Startup

**Symptoms:**
- Immediate tumbling when simulation starts
- Angular velocities exceed 100 rad/s

**Solutions:**

1. **Check initial orientation:**
```python
# In _reset_idx(), ensure quaternion is valid
default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # Identity
self._robot.write_root_pose_to_sim(poses, env_ids)
```

2. **Verify force application:**
```python
# Ensure forces are applied to correct body in correct frame
print(f"Force magnitude: {torch.norm(force, dim=1).mean():.2f} N")
print(f"Moment magnitude: {torch.norm(moment, dim=1).mean():.2f} Nm")
```

3. **Check gains are not too aggressive:**
```python
# Temporarily reduce gains
gains["kp_att"] *= 0.5
gains["kd_att"] *= 0.5
```

### Problem: Drone Drifts Horizontally

**Symptoms:**
- Maintains altitude but drifts in X/Y
- Velocity tracking error persists

**Solutions:**

1. **Enable integral term:**
```python
gains["ki_x"] = torch.tensor([0.5], device='cuda:0').expand(num_envs)
```

2. **Check velocity observations:**
```python
# Verify velocities are in correct frame
print(f"Vel error: {cmd_vel - curr_lin_vel_w}")
```

3. **Increase position gains:**
```python
gains["kp_x"] *= 1.5
```

### Problem: Oscillation Around 0°/360° Roll

**Symptoms:**
- Roll oscillates between 359° and 1°
- Moments saturate at ±5 Nm
- Otherwise stable

**Root Cause:**
- Quaternion double-cover issue when targeting 0° roll
- USD body frame has Z-down convention

**Current Status:**
- This is a known issue documented in the debugging history
- Oscillation is small (~1.5 rad/s) and does not prevent flight
- RL policies can learn to compensate

**Workarounds:**
1. **Accept the oscillation** - it's flyable for RL training
2. **Modify USD file** - flip body frame at source (advanced)
3. **Implement deadband** - ignore errors < 5° (may reduce control authority)

### Problem: Integral Windup

**Symptoms:**
- Overshoot when approaching target
- Oscillation with increasing amplitude
- Slow settling time

**Solutions:**

1. **Reduce integral gain:**
```python
gains["ki_x"] *= 0.5
```

2. **Check anti-windup is working:**
```python
# Monitor integral state
print(f"Integral: {controller._vel_error_integral.abs().max():.2f}")
# Should not exceed controller._integral_max (default: 5.0)
```

3. **Reset integral on large errors:**
```python
# In environment reset
controller.reset_integral(env_ids)
```

---

## Advanced Topics

### 1. Adaptive Gain Scheduling

Adjust gains based on flight regime:

```python
def get_adaptive_gains(velocity, altitude):
    """Compute gains based on current flight state."""
    base_gains = controller.default_gains.copy()

    # Higher gains at high speed for stability
    speed = torch.norm(velocity[:, :2], dim=1)
    kp_att_factor = 1.0 + 0.2 * (speed / 5.0).clamp(0, 1)

    base_gains["kp_att"] = base_gains["kp_att"] * kp_att_factor
    base_gains["kd_att"] = base_gains["kd_att"] * kp_att_factor

    # Lower gains near ground for soft landing
    altitude_factor = (altitude / 2.0).clamp(0.5, 1.0)
    base_gains["kp_z"] = base_gains["kp_z"] * altitude_factor

    return base_gains
```

### 2. Emergency Recovery

Detect and recover from unstable states:

```python
def check_stability(ang_vel, quat):
    """Check if drone is in recoverable state."""
    # Check angular velocity
    ang_vel_mag = torch.norm(ang_vel, dim=1)
    is_spinning = ang_vel_mag > 10.0  # rad/s

    # Check orientation (upside down = ~180° roll/pitch)
    from isaaclab.utils.math import euler_xyz_from_quat
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    is_inverted = (torch.abs(roll) > 2.5) | (torch.abs(pitch) > 2.5)

    # Recovery needed
    needs_recovery = is_spinning | is_inverted

    return needs_recovery

# In _apply_action()
if check_stability(ang_vel, quat).any():
    # Reset affected environments
    env_ids = torch.where(check_stability(ang_vel, quat))[0]
    self._reset_idx(env_ids)
```

### 3. Multi-Robot Coordination

Use separate controller instances for formation flight:

```python
# Leader-follower formation
leader_controller = PointMass(mass=1.619, num_envs=1, device='cuda:0')
follower_controllers = [
    PointMass(mass=1.619, num_envs=1, device='cuda:0')
    for _ in range(3)
]

# Leader tracks user command
leader_force, leader_moment = leader_controller.compute_control_quat_exact(
    cmd_lin_vel_w=user_command,
    # ... state ...
)

# Followers track leader with offset
for i, follower in enumerate(follower_controllers):
    offset = formation_offsets[i]  # e.g., [2.0, 2.0, 0.0]
    target_vel = leader_vel + compute_formation_correction(offset)

    force, moment = follower.compute_control_quat_exact(
        cmd_lin_vel_w=target_vel,
        # ... state ...
    )
```

### 4. Performance Optimization

For high-frequency control (>100 Hz):

```python
# Use small-angle approximation for speed
force, moment = controller.compute_control(  # Faster than compute_control_quat_exact
    cmd_lin_vel_w=cmd_vel,
    cmd_yaw_vel=cmd_yaw_rate,
    # ... state ...
)

# Batch multiple steps
for _ in range(decimation):
    controller.compute_control(...)  # Reuse same gains dict

# Minimize data transfers
with torch.no_grad():  # Disable gradient computation
    force, moment = controller.compute_control_quat_exact(...)
```

---

## See Also

- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/) - Environment setup
- [Debugging History](../../../../../../ROOT_CAUSE_FOUND.md) - External force application troubleshooting
