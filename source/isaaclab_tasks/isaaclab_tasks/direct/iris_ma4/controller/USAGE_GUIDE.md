# Point Mass Controller Usage Guide

## Overview

This guide documents the **proven stable method** for controlling quadcopters with and without gimbals in Isaac Lab. The implementation is based on extensive testing and debugging, culminating in the working solution in [quadcopter_env_v2.py](../quadcopter/quadcopter_env_v2.py).

**Key Achievements:**
- ✅ Stable velocity control for base body (vx, vy, vz, yaw_rate)
- ✅ Independent gimbal control (yaw, pitch) with automatic roll stabilization
- ✅ No conflicts between base and gimbal actuators
- ✅ Tested with 4096 parallel environments
- ✅ Compatible with RL training pipelines

**Architecture:**
- **Base Body**: External forces/torques via PhysX (PointMass controller)
- **Gimbal**: Position control via Articulation API (ImplicitActuators)
- **Hybrid Approach**: Both control methods coexist without conflicts

## Table of Contents

1. [Quick Start](#quick-start)
2. [Integration with Isaac Lab](#integration-with-isaac-lab)
3. [Common Usage Patterns](#common-usage-patterns)
4. [Gain Tuning](#gain-tuning)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Basic Setup (Base Body Only)

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

# Apply forces using Isaac Lab's standard API (RECOMMENDED)
force_reshaped = force.unsqueeze(1)  # (num_envs, 1, 3)
moment_reshaped = moment.unsqueeze(1)  # (num_envs, 1, 3)

robot.set_external_force_and_torque(
    forces=force_reshaped,
    torques=moment_reshaped,
    body_ids=[0],  # Apply to base body only
    env_ids=None   # All environments
)
# Forces are applied in write_data_to_sim() before physics step
```

### Hybrid Setup (Base Body + Gimbal Control)

**Recommended for quadcopters with gimbals** - This is the proven stable approach:

```python
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.iris_ma3.controller.point_mass import PointMass
from isaaclab_tasks.direct.iris_ma3.controller.gimbal_stabilizer import GimbalStabilizer

# Custom articulation class for hybrid control
class QuadcopterArticulation(Articulation):
    """Hybrid control: External forces for base, actuators for gimbal."""

    def write_data_to_sim(self):
        # Apply external wrench to base body
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_INDICES,
                is_global=False,
            )

        # Apply actuator model for gimbal joints
        self._apply_actuator_model()
        self.root_physx_view.set_dof_actuation_forces(
            self._joint_effort_target_sim, self._ALL_INDICES
        )

        # Position targets for gimbal (implicit actuators)
        if self._has_implicit_actuators:
            self.root_physx_view.set_dof_position_targets(
                self._joint_pos_target_sim, self._ALL_INDICES
            )
            self.root_physx_view.set_dof_velocity_targets(
                self._joint_vel_target_sim, self._ALL_INDICES
            )

# In your environment
def _apply_action(self):
    # Get current robot state
    curr_quat_w = self._robot.data.root_quat_w.clone()
    curr_lin_vel_w = self._robot.data.root_lin_vel_w.clone()
    curr_ang_vel_b = self._robot.data.root_ang_vel_b.clone()
    curr_lin_acc_w = self._robot.data.body_lin_acc_w[:, body_id].clone()

    # Convert acceleration to body frame
    from isaaclab.utils.math import quat_rotate_inverse
    curr_lin_acc_b = quat_rotate_inverse(curr_quat_w, curr_lin_acc_w)

    # Base body control
    force, moment = self._robot_controller.compute_control_quat_exact(
        cmd_lin_vel_w=self._actions[:, 0:3] * self.cfg.max_linear_speed,
        cmd_yaw_vel=self._actions[:, 3] * self.cfg.max_yaw_rate,
        curr_quat_w=curr_quat_w,
        curr_lin_vel_w=curr_lin_vel_w,
        curr_ang_vel_b=curr_ang_vel_b,
        curr_lin_acc_b=curr_lin_acc_b,
        dt=self.cfg.sim.dt,
        gains=self._controller_gains  # Optional: custom gains
    )

    # Apply external forces to base body
    self._robot.set_external_force_and_torque(
        forces=force.unsqueeze(1),
        torques=moment.unsqueeze(1),
        body_ids=[0],
        env_ids=None
    )

    # Gimbal control (rate-based position commands)
    if self.has_gimbal:
        cmd_gimbal_yaw = self._actions[:, 4]
        cmd_gimbal_pitch = self._actions[:, 5]

        gimbal_yaw_current = self._robot.data.joint_pos[:, self.gimbal_joint_idx["yaw"]]
        gimbal_pitch_current = self._robot.data.joint_pos[:, self.gimbal_joint_idx["pitch"]]

        # Rate control: increment position
        gimbal_yaw_target = gimbal_yaw_current + cmd_gimbal_yaw * max_rate * dt
        gimbal_pitch_target = gimbal_pitch_current + cmd_gimbal_pitch * max_rate * dt

        # Stabilizing roll (keeps horizon level)
        gimbal_roll_stabilizing = self._gimbal_stabilizer.compute_stabilizing_roll(
            gimbal_yaw_current, gimbal_pitch_current, self._robot.data.root_quat_w
        )

        # Apply gimbal position targets
        self._robot.set_joint_position_target(
            target=torch.stack([gimbal_yaw_target, gimbal_roll_stabilizing, gimbal_pitch_target], dim=-1),
            joint_ids=[self.gimbal_joint_idx["yaw"], self.gimbal_joint_idx["roll"], self.gimbal_joint_idx["pitch"]]
        )
```

---

## Why Use Isaac Lab's API? (Critical Background)

### The Actuator Override Problem

When integrating external force-based controllers (like PointMass) with Isaac Lab articulations, you may encounter a **critical issue**: **articulation actuators override external forces**.

#### Root Cause

Isaac Lab's articulation system calls these methods in sequence during `write_data_to_sim()`:

1. `apply_forces_and_torques_at_position()` - Applies external wrenches
2. `_apply_actuator_model()` - Computes joint forces from actuator models
3. **`set_dof_actuation_forces()`** - **Overwrites** forces on actuated joints

If your robot has propeller/rotor actuators (e.g., `DirectActuatorCfg` on propeller joints), step 3 will **override the external forces** you applied in step 1, causing the controller to have no effect.

#### Attempted Solutions (What NOT to Do)

**❌ Bad Approach 1: Direct PhysX API calls in `_apply_action()`**

```python
# DON'T DO THIS - causes race conditions and timing issues
def _apply_action(self):
    force, moment = controller.compute_control(...)

    # Calling PhysX API directly bypasses Isaac Lab's buffer management
    self._robot.root_physx_view.apply_forces_and_torques_at_position(
        force_data=force_all_bodies,
        torque_data=moment_all_bodies,
        indices=self._robot._ALL_INDICES,
        is_global=False
    )
    # Problem: write_data_to_sim() is called LATER and may override these forces!
```

**Why this fails:**
- `apply_forces_and_torques_at_position()` is called in `_apply_action()`
- But `write_data_to_sim()` is called **after** by the environment loop
- `set_dof_actuation_forces()` in `write_data_to_sim()` may override your forces
- Physics pipeline timing is critical - forces must be applied at the right moment

**❌ Bad Approach 2: Custom `write_data_to_sim()` without setting `has_external_wrench` flag**

```python
# DON'T DO THIS - flag stays False, forces never applied
class CustomArticulation(Articulation):
    def write_data_to_sim(self):
        # This check will FAIL because flag is False!
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(...)
```

**Why this fails:**
- `has_external_wrench` flag is only set to `True` when you call `set_external_force_and_torque()`
- If you bypass this API, the flag stays `False`
- Your custom `write_data_to_sim()` skips force application entirely

#### ✅ Correct Solution: Use Isaac Lab's Standard API

**Step 1: Remove propeller actuators from robot config**

```python
# In your robot configuration (e.g., iris_gimbal2.py)
IRIS_CFG = ArticulationCfg(
    actuators={
        # REMOVE propeller actuators - they cause set_dof_actuation_forces() conflicts
        # "propellers": DirectActuatorCfg(...),  # DELETE THIS

        # KEEP gimbal actuators if needed (servos, different joints)
        "gimbal_yaw": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint"],
            stiffness=2e3,
            damping=1e2,
        ),
    },
)
```

**Step 2: Use `set_external_force_and_torque()` in your environment**

```python
def _apply_action(self):
    # Compute control forces
    force, moment = self._controller.compute_control(...)  # (num_envs, 3)

    # Reshape to (num_envs, num_bodies, 3) - we apply only to base body
    force_reshaped = force.unsqueeze(1)  # (num_envs, 1, 3)
    moment_reshaped = moment.unsqueeze(1)  # (num_envs, 1, 3)

    # Use standard Isaac Lab API
    # This sets has_external_wrench=True and populates internal buffers
    self._robot.set_external_force_and_torque(
        forces=force_reshaped,
        torques=moment_reshaped,
        body_ids=[0],  # Base body ID
        env_ids=None   # All environments
    )
    # Forces will be applied in write_data_to_sim() before physics step
```

**Step 3: Use standard or custom `write_data_to_sim()`**

```python
# Option A: Use base class (if no gimbal)
# No custom write_data_to_sim() needed - base class handles everything

# Option B: Custom class for hybrid control (base + gimbal) - RECOMMENDED FOR GIMBALS
class QuadcopterArticulation(Articulation):
    """Hybrid control: External forces for base, actuators for gimbal.

    This is the proven stable approach for quadcopters with gimbals.
    Tested and validated in quadcopter_env_v2.py.
    """

    def write_data_to_sim(self):
        # Apply external wrench to base body (flag is now True!)
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_INDICES,
                is_global=False,
            )

        # Apply gimbal actuators - safe because propeller actuators removed
        self._apply_actuator_model()
        self.root_physx_view.set_dof_actuation_forces(
            self._joint_effort_target_sim, self._ALL_INDICES
        )

        # Position targets for gimbal joints (implicit actuators)
        if self._has_implicit_actuators:
            self.root_physx_view.set_dof_position_targets(
                self._joint_pos_target_sim, self._ALL_INDICES
            )
            self.root_physx_view.set_dof_velocity_targets(
                self._joint_vel_target_sim, self._ALL_INDICES
            )
```

### Why This Works

1. **Removed propeller actuators** → `set_dof_actuation_forces()` only affects gimbal joints (if any)
2. **Called `set_external_force_and_torque()`** → Sets `has_external_wrench=True` and populates buffers
3. **`write_data_to_sim()` applies forces** → At the correct time in physics pipeline
4. **No race conditions** → Forces and actuator commands applied in correct sequence
5. **Hybrid architecture** → Base body controlled by external forces, gimbal by actuators - no conflicts

**Gimbal Control Benefits:**
- **Rate-based commands**: Incremental position updates prevent jumps
- **Automatic stabilization**: GimbalStabilizer keeps horizon level
- **Decoupled from base**: Gimbal motion doesn't disturb base body control
- **Standard API**: Uses `set_joint_position_target()` - no custom PhysX calls needed

### Troubleshooting Checklist

If your controller has no effect, check:

- [ ] `has_external_wrench` flag is `True` (print in diagnostics)
- [ ] Propeller actuators removed from robot config
- [ ] Called `set_external_force_and_torque()` in `_apply_action()`
- [ ] Not calling `apply_forces_and_torques_at_position()` directly
- [ ] `write_data_to_sim()` applies forces when flag is True

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

### 4. Gimbal Control Integration

Control gimbal angles independently from base body motion:

```python
from isaaclab_tasks.direct.iris_ma3.controller.gimbal_stabilizer import GimbalStabilizer

# Initialize gimbal stabilizer
gimbal_stabilizer = GimbalStabilizer(
    device=device,
    yaw_limits=[-math.pi, math.pi],      # ±180°
    pitch_limits=[-math.pi/2, math.pi/2], # ±90°
)

def _apply_gimbal_control(self):
    """Apply gimbal position commands with automatic stabilization."""
    # Get gimbal commands from actions
    cmd_gimbal_yaw = self._actions[:, 4]    # [-1, 1] normalized
    cmd_gimbal_pitch = self._actions[:, 5]

    # Get current gimbal positions
    gimbal_yaw_current = self._robot.data.joint_pos[:, self.gimbal_joint_idx["yaw"]]
    gimbal_pitch_current = self._robot.data.joint_pos[:, self.gimbal_joint_idx["pitch"]]

    # Rate control: increment position targets
    max_rate = math.radians(360)  # 360°/s
    dt = self.step_dt

    gimbal_yaw_target = gimbal_yaw_current + cmd_gimbal_yaw * max_rate * dt
    gimbal_pitch_target = gimbal_pitch_current + cmd_gimbal_pitch * max_rate * dt

    # Compute stabilizing roll (keeps horizon level during yaw/pitch motion)
    gimbal_roll_stabilizing = gimbal_stabilizer.compute_stabilizing_roll(
        gimbal_yaw_current,
        gimbal_pitch_current,
        self._robot.data.root_quat_w  # Base body orientation
    )

    # Apply targets to gimbal joints
    self._robot.set_joint_position_target(
        target=torch.stack([gimbal_yaw_target, gimbal_roll_stabilizing, gimbal_pitch_target], dim=-1),
        joint_ids=[
            self.gimbal_joint_idx["yaw"],
            self.gimbal_joint_idx["roll"],
            self.gimbal_joint_idx["pitch"],
        ]
    )
```

**Key Points:**
- **Rate control**: Prevents large jumps, smooth motion
- **Automatic roll stabilization**: Keeps camera horizon level
- **Decoupled**: Gimbal doesn't affect base body dynamics
- **Joint IDs**: Ensure correct mapping from robot config

### 5. Observation Collection

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

        # Gimbal state (if applicable)
        "gimbal_angles": self._robot.data.joint_pos[:, [
            self.gimbal_joint_idx["yaw"],
            self.gimbal_joint_idx["pitch"],
            self.gimbal_joint_idx["roll"],
        ]],
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

### Problem: Gimbal Not Moving

**Symptoms:**
- Gimbal joints stay at initial position
- No response to gimbal commands
- Base body control works fine

**Solutions:**

1. **Check gimbal joint indices:**
```python
# Verify joint IDs are found
print(f"Gimbal joint IDs: {self.gimbal_joint_idx}")
# Should not be None
assert all(idx is not None for idx in self.gimbal_joint_idx.values())
```

2. **Verify actuator config:**
```python
# In robot config, ensure gimbal has actuators
actuators = {
    "gimbal_yaw": ImplicitActuatorCfg(
        joint_names_expr=["yaw_joint"],
        stiffness=2e3,
        damping=1e2,
    ),
    # ... pitch and roll ...
}
```

3. **Check custom articulation class:**
```python
# Ensure write_data_to_sim() calls actuator API
def write_data_to_sim(self):
    # ... external wrench ...
    self._apply_actuator_model()  # REQUIRED
    self.root_physx_view.set_dof_actuation_forces(...)
    if self._has_implicit_actuators:  # REQUIRED for position control
        self.root_physx_view.set_dof_position_targets(...)
```

### Problem: Gimbal Disturbs Base Body

**Symptoms:**
- Drone tilts when gimbal moves
- Base body velocity tracking degrades during gimbal motion
- Oscillations correlate with gimbal movement

**Root Cause:**
- Gimbal inertia affects base body (physics coupling)
- This is expected physical behavior

**Solutions:**

1. **Increase controller gains** to compensate:
```python
# Slightly higher attitude gains help reject disturbances
gains["kp_att"] *= 1.2
gains["kd_att"] *= 1.2
```

2. **Use feedforward compensation** (advanced):
```python
# Predict gimbal-induced torque and pre-compensate
gimbal_accel = (gimbal_vel_target - gimbal_vel_current) / dt
compensation_moment = gimbal_inertia * gimbal_accel
moment_total = moment_from_controller + compensation_moment
```

3. **Accept minor coupling** - typically < 5% impact on tracking

### Problem: Camera Horizon Not Level

**Symptoms:**
- Camera tilts when gimbal yaws
- Horizon rotates during pitch motion
- Roll stabilization not working

**Solutions:**

1. **Verify GimbalStabilizer is used:**
```python
from isaaclab_tasks.direct.iris_ma3.controller.gimbal_stabilizer import GimbalStabilizer

# Must be initialized
self._gimbal_stabilizer = GimbalStabilizer(device=device, ...)

# Must be called in _apply_action()
roll_stabilizing = self._gimbal_stabilizer.compute_stabilizing_roll(
    gimbal_yaw, gimbal_pitch, robot_quat_w
)
```

2. **Check roll joint index:**
```python
# Roll stabilization only works if roll joint exists
assert self.gimbal_joint_idx["roll"] is not None
```

3. **Verify target application order:**
```python
# Order matters: [yaw, roll, pitch]
target = torch.stack([yaw_target, roll_stab, pitch_target], dim=-1)
joint_ids = [yaw_idx, roll_idx, pitch_idx]
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
- **[quadcopter_env_v2.py](../quadcopter/quadcopter_env_v2.py)** - **Reference implementation** of stable hybrid control (base + gimbal)
