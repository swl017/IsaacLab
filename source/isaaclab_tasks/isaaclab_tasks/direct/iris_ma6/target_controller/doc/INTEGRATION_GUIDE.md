# Target Controller Integration Guide

This guide explains how to integrate the `TargetController` module into an Isaac Lab environment for physics-based target movement.

## Overview

The `TargetController` provides physics-realistic target movement using the same `DroneController` architecture as agents. Instead of direct velocity writes (`write_root_velocity_to_sim()`), it generates forces and torques applied via `set_external_force_and_torque()`.

**Key Benefits:**
- Physically realistic dynamics (inertia, acceleration limits)
- Consistent behavior with agent drones
- Support for attacker behavior modes (approach, evade)
- Curriculum-scaled difficulty

## Prerequisites

1. Target must be a `RigidObject` with physics enabled
2. Target mass must be known (read from physics asset)
3. Environment must call `step()` at simulation frequency

## Integration Steps

### Step 1: Import the Module

```python
from .target_controller import TargetController, TargetControllerCfg
```

### Step 2: Add Configuration

In your environment config class:

```python
from .target_controller import TargetControllerCfg

@configclass
class MyEnvCfg(DirectMARLEnvCfg):
    # ... other config ...

    # Target controller configuration
    target_controller: TargetControllerCfg = TargetControllerCfg(
        control_dt=0.01,  # Match your simulation dt
        max_speed_start=3.0,
        max_speed_end=12.0,
        use_attacker_mode=True,
    )

    # Enable/disable target controller
    enable_target_controller: bool = True
```

### Step 3: Initialize in Environment

In your environment `__init__`:

```python
def __init__(self, cfg: MyEnvCfg, **kwargs):
    super().__init__(cfg, **kwargs)

    # ... other initialization ...

    # Initialize target controller if enabled
    if self.cfg.enable_target_controller:
        # Get target mass from physics
        target_masses = self.target.root_physx_view.get_masses()
        target_mass = target_masses[0].sum().item()

        # Set control_dt to match simulation dt
        target_cfg = self.cfg.target_controller
        target_cfg.control_dt = self.cfg.sim.dt

        self._target_controller = TargetController(
            cfg=target_cfg,
            mass=target_mass,
            gravity=9.81,
            num_envs=self.num_envs,
            num_targets=1,  # Adjust for your setup
            device=self.device,
        )
    else:
        self._target_controller = None
```

### Step 4: Apply Actions in `_apply_action`

In your `_apply_action` method, call the target controller to get forces/torques:

```python
def _apply_action(self):
    # ... agent action application ...

    # Apply target controller if enabled
    if self._target_controller is not None:
        # Get current target state
        target_pos = self.target.data.root_pos_w.unsqueeze(1)  # [N, 1, 3]
        target_vel = self.target.data.root_lin_vel_w.unsqueeze(1)  # [N, 1, 3]
        target_quat = self.target.data.root_quat_w.unsqueeze(1)  # [N, 1, 4]
        target_omega = self.target.data.root_ang_vel_b.unsqueeze(1)  # [N, 1, 3]

        # Get agent positions for evasion logic
        agent_positions = torch.stack([
            self._robots[agent_id].data.root_pos_w
            for agent_id in self.cfg.possible_agents
        ], dim=1)  # [N, num_agents, 3]

        # Interceptor roles (1 = INTERCEPT for all agents)
        agent_roles = torch.ones(
            self.num_envs, len(self.cfg.possible_agents),
            dtype=torch.long, device=self.device
        )

        # Step the target controller
        dt = self.cfg.sim.dt
        F_body, tau_body = self._target_controller.step(
            current_position=target_pos,
            current_velocity=target_vel,
            current_quat=target_quat,
            current_angular_vel=target_omega,
            facility_position=self._facility_position,  # [N, 3]
            interceptor_positions=agent_positions,
            interceptor_roles=agent_roles,
            curriculum_progress=self._curriculum_progress,
            dt=dt,
            env_origins=self._terrain.env_origins,
        )

        # Apply forces to target (squeeze target dimension)
        self.target.set_external_force_and_torque(
            forces=F_body.squeeze(1).unsqueeze(1),  # [N, 1, 3]
            torques=tau_body.squeeze(1).unsqueeze(1),  # [N, 1, 3]
            body_ids=[0],  # Root body
        )
```

### Step 5: Reset Handler

In your `_reset_idx` method:

```python
def _reset_idx(self, env_ids: torch.Tensor):
    super()._reset_idx(env_ids)

    # ... other reset logic ...

    # Reset target controller
    if self._target_controller is not None:
        self._target_controller.reset(env_ids)
```

### Step 6: Curriculum Integration (Optional)

To scale difficulty over training:

```python
def set_curriculum_progress(self, progress: float):
    """Update curriculum progress (0.0 to 1.0)."""
    self._curriculum_progress = progress

    # Curriculum affects:
    # - Target max speed: 3 -> 12 m/s
    # - Geofence size: 50 -> 200 m
    # - Update intervals: 6-12s -> 2-4s (faster direction changes)
    # - Evasion agility: 0 -> 1 (after evasion_start_progress)
```

## Configuration Reference

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `control_dt` | 0.01 | Control loop timestep [s] |
| `max_speed_start` | 3.0 | Max speed at curriculum 0 [m/s] |
| `max_speed_end` | 12.0 | Max speed at curriculum 1 [m/s] |
| `max_tilt` | 30.0 | Max tilt angle [deg] |

### Velocity Mode Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_attacker_mode` | True | Use approach/evade vs linear/circular |
| `default_velocity_mode` | "linear" | Default mode: linear, circular, approach |
| `linear_weight` | 0.5 | Probability of linear vs circular |

### Behavior Profiles

| Profile | Speed | Evasion | Path Type |
|---------|-------|---------|-----------|
| kamikaze | 1.5x | 0% | Direct only |
| standard | 1.0x | 50% | Mostly offset |
| evasive | 1.0x | 90% | Mostly offset |
| stealth | 0.6x | 50% | Mostly low-alt |

### Geofencing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `geofence_min_size` | 50.0 | Half-width at progress 0 [m] |
| `geofence_max_size` | 200.0 | Half-width at progress 1 [m] |
| `min_altitude` | 5.0 | Minimum altitude [m] |
| `max_altitude` | 100.0 | Maximum altitude [m] |

## FSM States

The target controller uses a Finite State Machine for attacker behavior:

```
    ┌────────────┐
    │  APPROACH  │ ◄─── Initial state
    └─────┬──────┘
          │
          │ Interceptor detected
          │ within evade_trigger_distance
          ▼
    ┌────────────┐
    │   EVADE    │
    └─────┬──────┘
          │
          │ Evasion timer expires
          │
          ▼
    ┌────────────┐
    │  APPROACH  │ ◄─── Return to approach
    └────────────┘
          │
          │ Intercepted (distance < capture_distance)
          ▼
    ┌────────────┐
    │    DEAD    │
    └────────────┘
```

## Accessing Target State

```python
# Check if targets are alive
alive = self._target_controller.alive  # [N_env, N_target]

# Get FSM state
fsm_state = self._target_controller.fsm_state  # [N_env, N_target]

# Check for intercept events
intercepted = self._target_controller.check_intercept(
    current_position=target_pos,
    interceptor_positions=agent_positions,
    interceptor_roles=agent_roles,
    capture_distance=5.0,
)
```

## Troubleshooting

### Target not moving
1. Check that `enable_target_controller` is True
2. Verify `set_external_force_and_torque` is called each step
3. Ensure target mass is correctly read from physics

### Erratic movement
1. Lower controller gains in `TargetControllerCfg`
2. Increase `motor_tau` for smoother response
3. Reduce `max_tilt` for more stable flight

### Targets ignoring geofence
1. Verify `env_origins` is passed to `step()`
2. Check `geofence_min_size` and `geofence_max_size` values

### Evasion not triggering
1. Ensure `curriculum_progress > evasion_start_progress` (default 0.3)
2. Check interceptor distance vs `evade_trigger_distance`
3. Verify `interceptor_roles` contains 1 (INTERCEPT)

## Example: Minimal Integration

```python
from isaaclab.envs import DirectRLEnv
from .target_controller import TargetController, TargetControllerCfg

class MinimalTargetEnv(DirectRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Read target mass
        mass = self.target.root_physx_view.get_masses()[0].sum().item()

        # Create controller
        self._target_ctrl = TargetController(
            cfg=TargetControllerCfg(control_dt=self.cfg.sim.dt),
            mass=mass,
            gravity=9.81,
            num_envs=self.num_envs,
            num_targets=1,
            device=self.device,
        )

        self._curriculum = 0.0

    def _apply_action(self):
        # Apply agent actions...

        # Step target controller
        F, tau = self._target_ctrl.step(
            current_position=self.target.data.root_pos_w.unsqueeze(1),
            current_velocity=self.target.data.root_lin_vel_w.unsqueeze(1),
            current_quat=self.target.data.root_quat_w.unsqueeze(1),
            current_angular_vel=self.target.data.root_ang_vel_b.unsqueeze(1),
            facility_position=torch.zeros(self.num_envs, 3, device=self.device),
            interceptor_positions=torch.zeros(self.num_envs, 1, 3, device=self.device),
            interceptor_roles=torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device),
            curriculum_progress=self._curriculum,
            dt=self.cfg.sim.dt,
        )

        # Apply forces
        self.target.set_external_force_and_torque(
            forces=F.squeeze(1).unsqueeze(1),
            torques=tau.squeeze(1).unsqueeze(1),
            body_ids=[0],
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._target_ctrl.reset(env_ids)
```

## Related Documentation

- [target_movement_spec.md](../../doc/target_movement_spec.md): Full specification
- [../tests/README.md](../tests/README.md): Test suite documentation
- [controller_spec.md](../../doc/controller_spec.md): DroneController architecture
