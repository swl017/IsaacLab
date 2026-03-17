# Target Movement Specification for iris_ma6

## Physics-Based Target Motion using DroneController

**Version**: 1.0
**Base**: iris_ma6 DroneController module
**Reference**: iris_ma5 TargetMovement module
**Scope**: Physics-realistic target movement system using the same cascaded control architecture as agents

**Related Documents**:
- [iris_ma6_env_spec.md](iris_ma6_env_spec.md) - Main environment specification (Section 4: Attacker System)
- [controller_spec.md](controller_spec.md) - DroneController architecture
- [frame_conventions.md](frame_conventions.md) - Coordinate frames and quaternion conventions

---

## 1. Executive Summary

### 1.1 Motivation

The iris_ma5 `TargetMovement` module uses direct velocity writes (`target.write_root_velocity_to_sim()`) which bypasses physics simulation and creates unrealistic instant velocity changes. This specification defines `TargetController` for iris_ma6, which wraps the existing `DroneController` to generate velocity commands that are translated through the full PX4-style cascaded control loop into physics-realistic forces and torques.

### 1.2 Key Differences from iris_ma5

| Aspect | iris_ma5 (TargetMovement) | iris_ma6 (TargetController) |
|--------|---------------------------|----------------------------|
| Motion method | `write_root_velocity_to_sim()` | `set_external_force_and_torque()` |
| Velocity changes | Instantaneous | Physically realistic (inertia, lag) |
| Control architecture | Direct velocity → sim | Velocity cmd → DroneController → F/tau |
| Motor dynamics | None | First-order motor lag (tau=10ms) |
| Attitude | Fixed (no tilt) | Tilts to accelerate (max 30 deg) |
| Behavior modes | Linear, Circular | Linear, Circular, Approach, Evade |
| Integration | Standalone module | Wraps DroneController |

### 1.3 Design Goals

1. **Reuse DroneController**: Leverage the existing, tested cascaded control loop
2. **Realistic target dynamics**: Targets exhibit physical inertia and control limitations
3. **Behavior composability**: Support both iris_ma5 modes and iris_ma6 attacker FSM
4. **Simplified interface**: Hide DroneController complexity behind velocity command API
5. **Curriculum compatibility**: Support progressive difficulty scaling

---

## 2. Module Architecture

### 2.1 Class Hierarchy

```
TargetController
├── DroneController (composition, not inheritance)
│   ├── VelocityController    # v_cmd → q_des, thrust_cmd
│   ├── AttitudeController    # q_des → rate_setpoint
│   ├── RateController        # rate_setpoint → tau_cmd
│   ├── MotorDynamics         # omega_cmd → F_body, tau_body
│   └── (Gimbal/Zoom: unused for targets)
├── VelocityGenerator
│   ├── LinearModeGenerator    # Random direction changes
│   ├── CircularModeGenerator  # Orbital flight paths
│   ├── ApproachModeGenerator  # Goal-directed approach
│   └── EvadeModeGenerator     # Reactive evasion
└── BehaviorFSM (for attacker integration)
    ├── APPROACH state
    ├── EVADE state
    └── DEAD state
```

### 2.2 Data Flow

```
                 ┌─────────────────────────────────────────────┐
                 │             TargetController                │
                 │                                             │
  Behavior Mode  │  ┌─────────────────┐   ┌─────────────────┐  │  F_body, tau_body
  ────────────►  │  │ VelocityGenerator│   │  DroneController │  │  ──────────────►
                 │  │                 │──►│                 │  │  (to physics)
  Current State  │  │ Linear/Circular │   │ Cascaded Control│  │
  ────────────►  │  │ Approach/Evade  │   │ v→q→rate→motor │  │
                 │  └─────────────────┘   └─────────────────┘  │
                 │         ▲                      ▲            │
                 │         │                      │            │
                 │    TargetControllerCfg    DroneControllerCfg│
                 └─────────────────────────────────────────────┘
```

### 2.3 Integration with Environment

```python
# In iris_ma_env6.py

# =========================================================================
# Setup (in __init__):
# =========================================================================
self._target_controller = TargetController(
    cfg=self.cfg.target_controller,
    drone_controller_cfg=self.cfg.target_drone_controller,  # Simplified gains
    mass=target_mass,  # Read from physics asset
    gravity=self.cfg.sim.gravity[2],
    num_targets=self.cfg.max_targets,
    num_envs=self.num_envs,
    device=self.device,
)

# =========================================================================
# Reset (in _reset_idx):
# =========================================================================
self._target_controller.reset(env_ids)

# =========================================================================
# Step (in _apply_action):
# =========================================================================
F_body, tau_body = self._target_controller.step(
    current_position=self.target.data.root_pos_w,
    current_velocity=self.target.data.root_lin_vel_w,
    current_quat=self.target.data.root_quat_w,
    current_angular_vel=self.target.data.root_ang_vel_b,
    facility_position=self._facility_pos,
    interceptor_positions=interceptor_pos,  # For evasion
    interceptor_roles=interceptor_roles,     # OBSERVE/INTERCEPT
    curriculum_progress=self._curriculum_progress,
    dt=self.cfg.sim.dt,
)

# Apply forces to target
self.target.set_external_force_and_torque(
    forces=F_body.unsqueeze(1),
    torques=tau_body.unsqueeze(1),
    body_ids=self._target_body_ids,
)
```

### 2.4 File Structure

```
iris_ma6/
├── target_controller/
│   ├── __init__.py
│   ├── target_controller.py       # Main TargetController class
│   ├── target_controller_cfg.py   # Configuration dataclass
│   ├── velocity_generators/
│   │   ├── __init__.py
│   │   ├── base_generator.py      # Abstract base class
│   │   ├── linear_mode.py         # Random direction mode
│   │   ├── circular_mode.py       # Orbital flight mode
│   │   ├── approach_mode.py       # Goal-directed approach
│   │   └── evade_mode.py          # Reactive evasion
│   ├── behavior_fsm.py            # FSM state management
│   └── tests/
│       ├── __init__.py
│       ├── run_tests.py
│       ├── test_linear_mode.py
│       ├── test_circular_mode.py
│       ├── test_approach_mode.py
│       ├── test_evade_mode.py
│       ├── test_controller_integration.py
│       └── README.md
```

---

## 3. Velocity Generation Modes

### 3.1 Linear Mode (iris_ma5 Compatible)

Random direction flight with periodic direction changes. Ported from iris_ma5.

**State Variables:**
```python
desired_velocity: torch.Tensor     # [N_env, N_target, 3] target velocity
update_timer: torch.Tensor         # [N_env, N_target] time since last update
next_update_time: torch.Tensor     # [N_env, N_target] time until next update
```

**Algorithm:**
```python
def compute_linear_velocity(
    self,
    env_ids: torch.Tensor,
    target_ids: torch.Tensor,
    curriculum_progress: float,
    dt: float,
) -> torch.Tensor:
    """Compute desired velocity for linear mode.

    Args:
        env_ids: Environment indices.
        target_ids: Target indices within environments.
        curriculum_progress: Curriculum progress (0-1).
        dt: Timestep [s].

    Returns:
        v_cmd: [N, 3] velocity command.
    """
    # Check for direction updates
    self.update_timer[env_ids, target_ids] += dt
    update_mask = self.update_timer[env_ids, target_ids] >= self.next_update_time[env_ids, target_ids]

    if update_mask.any():
        update_env_ids = env_ids[update_mask]
        update_target_ids = target_ids[update_mask]
        n = len(update_env_ids)

        # Generate random direction (uniformly on sphere)
        direction = torch.randn(n, 3, device=self.device)
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

        # Random speed within curriculum-scaled limits
        max_speed = self._get_max_speed(curriculum_progress)
        speed = torch.rand(n, 1, device=self.device) * max_speed

        self.desired_velocity[update_env_ids, update_target_ids] = direction * speed

        # Reset timer with curriculum-scaled interval
        self.update_timer[update_env_ids, update_target_ids] = 0.0
        self.next_update_time[update_env_ids, update_target_ids] = self._sample_update_interval(
            n, curriculum_progress
        )

    return self.desired_velocity[env_ids, target_ids]
```

**Reference**: [iris_ma5/target_movement/target_movement.py:299-350](../../../iris_ma5/target_movement/target_movement.py)

### 3.2 Circular Mode (iris_ma5 Compatible)

Orbital flight around a center point. Ported from iris_ma5.

**State Variables:**
```python
circular_center: torch.Tensor       # [N_env, N_target, 3] orbit center
circular_radius: torch.Tensor       # [N_env, N_target] orbit radius
circular_angular_speed: torch.Tensor # [N_env, N_target] angular velocity (rad/s)
circular_phase: torch.Tensor        # [N_env, N_target] current phase angle
circular_height: torch.Tensor       # [N_env, N_target] orbit altitude
```

**Algorithm:**
```python
def compute_circular_velocity(
    self,
    env_ids: torch.Tensor,
    target_ids: torch.Tensor,
    current_position: torch.Tensor,
    curriculum_progress: float,
    dt: float,
) -> torch.Tensor:
    """Compute desired velocity for circular orbit mode.

    Args:
        env_ids: Environment indices.
        target_ids: Target indices.
        current_position: [N, 3] current target positions.
        curriculum_progress: Curriculum progress (0-1).
        dt: Timestep [s].

    Returns:
        v_cmd: [N, 3] velocity command.
    """
    # Update phase
    self.circular_phase[env_ids, target_ids] += self.circular_angular_speed[env_ids, target_ids] * dt

    # Desired position on orbit
    phase = self.circular_phase[env_ids, target_ids]
    radius = self.circular_radius[env_ids, target_ids]
    center = self.circular_center[env_ids, target_ids]
    height = self.circular_height[env_ids, target_ids]

    desired_pos = torch.stack([
        center[:, 0] + radius * torch.cos(phase),
        center[:, 1] + radius * torch.sin(phase),
        center[:, 2] + height,
    ], dim=1)

    # Proportional control to track orbit
    pos_error = desired_pos - current_position
    kp = 1.0  # Position tracking gain

    velocity_cmd = kp * pos_error

    # Clamp to max speed
    speed = torch.norm(velocity_cmd, dim=1, keepdim=True)
    max_speed = self._get_max_speed(curriculum_progress)
    velocity_cmd = velocity_cmd * torch.clamp(max_speed / (speed + 1e-6), max=1.0)

    return velocity_cmd
```

**Reference**: [iris_ma5/target_movement/target_movement.py:359-396](../../../iris_ma5/target_movement/target_movement.py)

### 3.3 Approach Mode (Attacker)

Goal-directed approach toward facility with optional path variation. From iris_ma6 env spec §4.2.1.

**State Variables:**
```python
approach_path_type: torch.Tensor    # [N_env, N_target] 0=direct, 1=offset, 2=low-altitude
approach_waypoints: torch.Tensor    # [N_env, N_target, W, 3] intermediate waypoints (W=max waypoints)
current_waypoint_idx: torch.Tensor  # [N_env, N_target] current waypoint index
```

**Path Variants:**

| Path Type | ID | Description | Speed Modifier | Altitude |
|-----------|-----|-------------|----------------|----------|
| Direct | 0 | Straight line to facility | 1.0x | Normal |
| Offset | 1 | Via intermediate waypoints | 0.9x | Normal |
| Low-altitude | 2 | Ground-hugging approach | 0.7x | 5-10m |

**Algorithm:**
```python
def compute_approach_velocity(
    self,
    env_ids: torch.Tensor,
    target_ids: torch.Tensor,
    current_position: torch.Tensor,
    facility_position: torch.Tensor,
    curriculum_progress: float,
) -> torch.Tensor:
    """Compute approach velocity toward facility.

    Args:
        env_ids: Environment indices.
        target_ids: Target indices.
        current_position: [N, 3] current positions.
        facility_position: [N, 3] facility positions.
        curriculum_progress: Curriculum progress (0-1).

    Returns:
        v_cmd: [N, 3] velocity command.
    """
    n = len(env_ids)
    v_cmd = torch.zeros(n, 3, device=self.device)

    path_types = self.approach_path_type[env_ids, target_ids]

    # Direct approach: straight to facility
    direct_mask = path_types == 0
    if direct_mask.any():
        direction = facility_position[direct_mask] - current_position[direct_mask]
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        speed = self._get_approach_speed(curriculum_progress) * 1.0
        v_cmd[direct_mask] = direction * speed

    # Offset approach: via waypoints
    offset_mask = path_types == 1
    if offset_mask.any():
        offset_env_ids = env_ids[offset_mask]
        offset_target_ids = target_ids[offset_mask]
        waypoint_idx = self.current_waypoint_idx[offset_env_ids, offset_target_ids]

        # Get current target waypoint
        target_pos = self._get_waypoint_or_facility(
            offset_env_ids, offset_target_ids, waypoint_idx, facility_position[offset_mask]
        )

        # Check waypoint reached
        dist_to_waypoint = torch.norm(current_position[offset_mask] - target_pos, dim=1)
        reached = dist_to_waypoint < self.cfg.waypoint_reach_threshold

        # Advance to next waypoint
        self.current_waypoint_idx[offset_env_ids[reached], offset_target_ids[reached]] += 1

        direction = target_pos - current_position[offset_mask]
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        speed = self._get_approach_speed(curriculum_progress) * 0.9
        v_cmd[offset_mask] = direction * speed

    # Low-altitude approach
    low_alt_mask = path_types == 2
    if low_alt_mask.any():
        direction = facility_position[low_alt_mask] - current_position[low_alt_mask]
        direction[:, 2] = 0  # No vertical component in horizontal direction
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        speed = self._get_approach_speed(curriculum_progress) * 0.7
        v_cmd[low_alt_mask] = direction * speed

        # Altitude regulation toward low altitude
        target_alt = self.cfg.low_altitude_target  # 7.5m default
        alt_error = target_alt - current_position[low_alt_mask, 2]
        v_cmd[low_alt_mask, 2] = torch.clamp(alt_error * 0.5, -2.0, 2.0)

    return v_cmd
```

**Reference**: [iris_ma6_env_spec.md §4.2.1](iris_ma6_env_spec.md)

### 3.4 Evade Mode (Attacker)

Reactive evasion maneuver when interceptor detected nearby. From iris_ma6 env spec §4.2.2.

**State Variables:**
```python
evasion_direction: torch.Tensor     # [N_env, N_target, 3] evasion direction (perpendicular to threat)
evasion_timer: torch.Tensor         # [N_env, N_target] remaining evasion time
evasion_agility: torch.Tensor       # [N_env, N_target] evasion intensity (0-1)
```

**Algorithm:**
```python
def compute_evade_velocity(
    self,
    env_ids: torch.Tensor,
    target_ids: torch.Tensor,
    current_position: torch.Tensor,
    interceptor_positions: torch.Tensor,
    interceptor_roles: torch.Tensor,
    curriculum_progress: float,
    dt: float,
) -> torch.Tensor:
    """Compute evasive velocity away from nearest interceptor.

    Args:
        env_ids: Environment indices.
        target_ids: Target indices.
        current_position: [N, 3] current positions.
        interceptor_positions: [N, num_defenders, 3] defender positions.
        interceptor_roles: [N, num_defenders] roles (0=OBSERVE, 1=INTERCEPT).
        curriculum_progress: Curriculum progress (0-1).
        dt: Timestep [s].

    Returns:
        v_cmd: [N, 3] velocity command.
    """
    n = len(env_ids)

    # Filter to only INTERCEPT mode defenders
    intercept_mask = interceptor_roles == 1  # [N, num_defenders]

    # Compute distance to each interceptor
    pos_expanded = current_position.unsqueeze(1)  # [N, 1, 3]
    dist_to_interceptors = torch.norm(pos_expanded - interceptor_positions, dim=-1)  # [N, num_defenders]

    # Mask out non-interceptors with large distance
    dist_to_interceptors[~intercept_mask] = 1e6

    # Find nearest interceptor
    nearest_dist, nearest_idx = dist_to_interceptors.min(dim=1)

    # Gather nearest interceptor position
    batch_indices = torch.arange(n, device=self.device)
    nearest_pos = interceptor_positions[batch_indices, nearest_idx]  # [N, 3]

    # Threat vector: from interceptor toward target
    threat_vec = current_position - nearest_pos
    threat_vec_norm = threat_vec / (torch.norm(threat_vec, dim=1, keepdim=True) + 1e-6)

    # Perpendicular direction for evasion (random left/right)
    random_sign = (torch.rand(n, device=self.device) > 0.5).float() * 2 - 1
    perp_vec = torch.stack([
        -threat_vec_norm[:, 1] * random_sign,
        threat_vec_norm[:, 0] * random_sign,
        torch.zeros(n, device=self.device),
    ], dim=1)

    # Blend evasion direction with escape direction
    escape_weight = 0.3  # 30% escape, 70% perpendicular
    evasion_direction = (1 - escape_weight) * perp_vec + escape_weight * threat_vec_norm
    evasion_direction = evasion_direction / (torch.norm(evasion_direction, dim=1, keepdim=True) + 1e-6)

    # Apply evasion agility scaling with curriculum
    agility = self.evasion_agility[env_ids, target_ids] * self._get_agility_scale(curriculum_progress)
    max_speed = self._get_max_speed(curriculum_progress)

    v_cmd = evasion_direction * max_speed * agility.unsqueeze(1)

    # Update evasion timer
    self.evasion_timer[env_ids, target_ids] -= dt

    return v_cmd
```

**Reference**: [iris_ma6_env_spec.md §4.2.2](iris_ma6_env_spec.md)

---

## 4. Configuration Parameters

### 4.1 TargetControllerCfg

```python
@configclass
class TargetControllerCfg:
    """Configuration for physics-based target movement."""

    # ==========================================================================
    # DroneController Override (Simplified for targets)
    # ==========================================================================

    use_simplified_controller: bool = True
    """Use simplified controller gains (less aggressive than agents)."""

    velocity_kp: tuple[float, float, float] = (2.0, 2.0, 2.0)
    """Velocity P gains for targets [x, y, z]. Lower than agents (4.5) for smoother motion."""

    velocity_ki: tuple[float, float, float] = (0.1, 0.1, 0.1)
    """Velocity I gains for targets [x, y, z]."""

    max_tilt: float = 30.0
    """Maximum tilt angle for targets [deg]. Lower than agents (45) for stability."""

    # ==========================================================================
    # Speed Control (Curriculum-Scaled)
    # ==========================================================================

    max_speed_start: float = 3.0
    """Maximum speed at curriculum progress 0 [m/s]."""

    max_speed_end: float = 12.0
    """Maximum speed at curriculum progress 1 [m/s]."""

    max_acceleration: float = 5.0
    """Maximum acceleration [m/s^2]."""

    # ==========================================================================
    # Linear Mode Parameters
    # ==========================================================================

    linear_weight: float = 0.5
    """Probability of linear mode vs circular mode (for iris_ma5 compatibility)."""

    update_interval_min_start: float = 6.0
    """Minimum direction change interval at progress 0 [s]."""

    update_interval_min_end: float = 2.0
    """Minimum direction change interval at progress 1 [s]."""

    update_interval_max_start: float = 12.0
    """Maximum direction change interval at progress 0 [s]."""

    update_interval_max_end: float = 4.0
    """Maximum direction change interval at progress 1 [s]."""

    # ==========================================================================
    # Circular Mode Parameters
    # ==========================================================================

    circular_radius_min: float = 15.0
    """Minimum orbit radius [m]."""

    circular_radius_max: float = 60.0
    """Maximum orbit radius [m]."""

    circular_height_min: float = 15.0
    """Minimum orbit altitude above center [m]."""

    circular_height_max: float = 40.0
    """Maximum orbit altitude above center [m]."""

    max_angular_speed: float = 0.3
    """Maximum angular speed for circular orbit [rad/s]."""

    # ==========================================================================
    # Approach Mode Parameters (Attacker)
    # ==========================================================================

    approach_speed_base: float = 7.0
    """Base approach speed [m/s]."""

    approach_path_direct_weight: float = 0.3
    """Probability of direct approach path."""

    approach_path_offset_weight: float = 0.4
    """Probability of offset approach path."""

    approach_path_low_alt_weight: float = 0.3
    """Probability of low-altitude approach path."""

    offset_waypoint_distance: float = 100.0
    """Distance of offset waypoints from direct path [m]."""

    max_waypoints: int = 3
    """Maximum number of waypoints for offset path."""

    waypoint_reach_threshold: float = 5.0
    """Distance to consider waypoint reached [m]."""

    low_altitude_target: float = 7.5
    """Target altitude for low-altitude approach [m]."""

    # ==========================================================================
    # Evasion Mode Parameters (Attacker)
    # ==========================================================================

    evade_trigger_distance: float = 50.0
    """Distance at which evasion triggers [m]."""

    evade_duration_min: float = 2.0
    """Minimum evasion duration [s]."""

    evade_duration_max: float = 5.0
    """Maximum evasion duration [s]."""

    evasion_agility_min: float = 0.3
    """Minimum evasion agility (speed multiplier)."""

    evasion_agility_max: float = 1.0
    """Maximum evasion agility (speed multiplier)."""

    # ==========================================================================
    # Geofencing
    # ==========================================================================

    geofence_min_size: float = 50.0
    """Geofence half-width at progress 0 [m]."""

    geofence_max_size: float = 200.0
    """Geofence half-width at progress 1 [m]."""

    geofence_margin: float = 10.0
    """Distance from boundary to start slowing [m]."""

    geofence_bounce_factor: float = 0.8
    """Velocity reduction factor when bouncing off geofence."""

    # ==========================================================================
    # Altitude Constraints
    # ==========================================================================

    min_altitude: float = 5.0
    """Minimum altitude above ground [m]."""

    max_altitude: float = 100.0
    """Maximum altitude above ground [m]."""

    altitude_bounce_velocity: float = 1.0
    """Vertical velocity applied when hitting altitude bounds [m/s]."""

    # ==========================================================================
    # Behavior Profiles (Attacker Domain Randomization)
    # ==========================================================================

    behavior_kamikaze_weight: float = 0.2
    """Probability of kamikaze profile (fast, no evasion)."""

    behavior_standard_weight: float = 0.4
    """Probability of standard profile (medium speed, medium evasion)."""

    behavior_evasive_weight: float = 0.25
    """Probability of evasive profile (medium speed, high evasion)."""

    behavior_stealth_weight: float = 0.15
    """Probability of stealth profile (slow, low altitude)."""
```

### 4.2 Behavior Profile Presets

Following iris_ma6_env_spec.md §4.2.3:

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BehaviorProfile:
    """Attacker behavior profile definition."""
    name: str
    speed_multiplier: float                   # Relative to max_speed
    evasion_agility: float                    # 0 = no evasion, 1 = maximum
    path_type_weights: Tuple[float, float, float]  # (direct, offset, low_alt)

BEHAVIOR_PROFILES = {
    "kamikaze": BehaviorProfile(
        name="kamikaze",
        speed_multiplier=1.5,      # Fast: 12 m/s at full curriculum
        evasion_agility=0.0,       # No evasion
        path_type_weights=(1.0, 0.0, 0.0),  # Direct only
    ),
    "standard": BehaviorProfile(
        name="standard",
        speed_multiplier=1.0,      # Medium: 7 m/s
        evasion_agility=0.5,       # Medium evasion
        path_type_weights=(0.2, 0.6, 0.2),  # Mostly offset
    ),
    "evasive": BehaviorProfile(
        name="evasive",
        speed_multiplier=1.0,      # Medium: 7 m/s
        evasion_agility=0.9,       # High evasion
        path_type_weights=(0.1, 0.7, 0.2),  # Mostly offset
    ),
    "stealth": BehaviorProfile(
        name="stealth",
        speed_multiplier=0.6,      # Slow: 3 m/s
        evasion_agility=0.5,       # Medium evasion
        path_type_weights=(0.0, 0.3, 0.7),  # Mostly low-altitude
    ),
}
```

---

## 5. DroneController Integration

### 5.1 Controller Simplification for Targets

Targets use a modified DroneControllerCfg with reduced gains for smoother, less aggressive motion:

```python
def create_target_drone_controller_cfg() -> DroneControllerCfg:
    """Create simplified DroneControllerCfg for targets.

    Key differences from agent configuration:
    - Lower velocity gains: Smoother acceleration response
    - Reduced max tilt: More stable flight, less aggressive maneuvering
    - Slower motor time constant: Less responsive but more stable
    - Disabled aerodynamics: Simplified dynamics

    Returns:
        DroneControllerCfg configured for target movement.
    """
    return DroneControllerCfg(
        velocity=VelocityControllerCfg(
            Kp_vel=(2.0, 2.0, 2.0),      # Agent: (4.5, 4.5, 3.4)
            Ki_vel=(0.1, 0.1, 0.1),      # Agent: (0.6, 0.6, 0.3)
            integral_limit=(2.0, 2.0, 1.0),
            max_tilt=30.0,               # Agent: 45.0
            max_lin_vel=15.0,            # Agent: 30.0
        ),
        attitude=AttitudeControllerCfg(
            Kp_att=(4.0, 4.0, 2.0),      # Agent: (6.5, 6.5, 2.8)
            rate_limit=(180.0, 180.0, 180.0),  # deg/s
            yaw_weight=0.4,
        ),
        rate=RateControllerCfg(
            Kp_rate=(0.10, 0.10, 0.15),  # Agent: (0.15, 0.15, 0.2)
            Ki_rate=(0.10, 0.10, 0.05),  # Agent: (0.2, 0.2, 0.1)
            Kd_rate=(0.002, 0.002, 0.0),
            integral_limit=(0.2, 0.2, 0.1),
            tau_max=(15.0, 15.0, 6.0),   # Nm
            yaw_weight=0.4,
        ),
        motor=MotorDynamicsCfg(
            k_f=1.2e-05,
            k_m=1.4e-07,
            tau_motor=0.015,             # Agent: 0.01 (slower response)
            omega_min=50.0,
            omega_max=5000.0,
            arm_length=0.22,
        ),
        gimbal=GimbalControllerCfg(),    # Unused but required
        zoom=ZoomControllerCfg(),        # Unused but required
        aerodynamics=AerodynamicsCfg(
            enable=False,                # Disabled for simplicity
        ),
        control_dt=0.01,                 # 100 Hz inner loop
    )
```

### 5.2 step_policy Wrapper

The TargetController wraps DroneController.step_policy with a simplified interface:

```python
def step(
    self,
    current_position: torch.Tensor,
    current_velocity: torch.Tensor,
    current_quat: torch.Tensor,
    current_angular_vel: torch.Tensor,
    facility_position: torch.Tensor,
    interceptor_positions: torch.Tensor,
    interceptor_roles: torch.Tensor,
    curriculum_progress: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Step target controller and return body forces/torques.

    This is the main interface for target movement. It:
    1. Determines velocity command based on FSM state and mode
    2. Applies geofencing and altitude constraints
    3. Passes velocity through DroneController for physics-based forces

    Args:
        current_position: [N_env, N_target, 3] positions in world frame.
        current_velocity: [N_env, N_target, 3] velocities in world frame.
        current_quat: [N_env, N_target, 4] orientations (wxyz).
        current_angular_vel: [N_env, N_target, 3] angular velocities in body frame.
        facility_position: [N_env, 3] facility positions.
        interceptor_positions: [N_env, N_defender, 3] defender positions.
        interceptor_roles: [N_env, N_defender] defender roles (0=OBSERVE, 1=INTERCEPT).
        curriculum_progress: Curriculum progress (0-1).
        dt: Simulation timestep [s].

    Returns:
        F_body: [N_env, N_target, 3] body frame forces [N].
        tau_body: [N_env, N_target, 3] body frame torques [Nm].
    """
    # Flatten for per-target processing
    batch_size = self.num_envs * self.num_targets
    pos_flat = current_position.view(batch_size, 3)
    vel_flat = current_velocity.view(batch_size, 3)
    quat_flat = current_quat.view(batch_size, 4)
    omega_flat = current_angular_vel.view(batch_size, 3)

    # Generate velocity commands based on FSM state and mode
    v_cmd = self._compute_velocity_command(
        current_position=pos_flat,
        facility_position=facility_position,
        interceptor_positions=interceptor_positions,
        interceptor_roles=interceptor_roles,
        curriculum_progress=curriculum_progress,
        dt=dt,
    )

    # Apply geofencing velocity modification
    v_cmd = self._apply_geofence_constraint(
        v_cmd, pos_flat, curriculum_progress
    )

    # Apply altitude constraint
    v_cmd = self._apply_altitude_constraint(
        v_cmd, pos_flat
    )

    # Pass through DroneController (no gimbal/zoom for targets)
    F_body_flat, tau_body_flat, _, _ = self._drone_controller.step_policy(
        v_cmd=v_cmd,
        yaw_rate_cmd=torch.zeros(batch_size, device=self.device),
        gimbal_yaw_rate_cmd=torch.zeros(batch_size, device=self.device),
        gimbal_pitch_rate_cmd=torch.zeros(batch_size, device=self.device),
        zoom_rate_cmd=torch.zeros(batch_size, device=self.device),
        q_body=quat_flat,
        v_body=vel_flat,
        omega_body=omega_flat,
        sim_dt=dt,
    )

    # Reshape back to [N_env, N_target, 3]
    F_body = F_body_flat.view(self.num_envs, self.num_targets, 3)
    tau_body = tau_body_flat.view(self.num_envs, self.num_targets, 3)

    return F_body, tau_body
```

### 5.3 Geofencing and Altitude Constraints

```python
def _apply_geofence_constraint(
    self,
    v_cmd: torch.Tensor,
    current_position: torch.Tensor,
    curriculum_progress: float,
) -> torch.Tensor:
    """Apply geofencing velocity constraints.

    Args:
        v_cmd: [N, 3] velocity command.
        current_position: [N, 3] current positions.
        curriculum_progress: Curriculum progress (0-1).

    Returns:
        Modified velocity command with geofence constraints.
    """
    # Compute geofence size based on curriculum
    geofence_size = self.cfg.geofence_min_size + curriculum_progress * (
        self.cfg.geofence_max_size - self.cfg.geofence_min_size
    )

    # Position relative to environment origin
    relative_pos = current_position - self._env_origins

    # Apply bounce at boundaries
    for axis in [0, 1]:  # X and Y only
        too_low = relative_pos[:, axis] < -geofence_size
        too_high = relative_pos[:, axis] > geofence_size

        # Bounce: reverse and reduce velocity
        v_cmd[too_low, axis] = torch.abs(v_cmd[too_low, axis]) * self.cfg.geofence_bounce_factor
        v_cmd[too_high, axis] = -torch.abs(v_cmd[too_high, axis]) * self.cfg.geofence_bounce_factor

    return v_cmd

def _apply_altitude_constraint(
    self,
    v_cmd: torch.Tensor,
    current_position: torch.Tensor,
) -> torch.Tensor:
    """Apply altitude constraints.

    Args:
        v_cmd: [N, 3] velocity command.
        current_position: [N, 3] current positions.

    Returns:
        Modified velocity command with altitude constraints.
    """
    altitude = current_position[:, 2] - self._env_origins[:, 2]

    # Minimum altitude
    too_low = altitude < self.cfg.min_altitude
    v_cmd[too_low, 2] = torch.clamp(
        v_cmd[too_low, 2],
        min=self.cfg.altitude_bounce_velocity,
    )

    # Maximum altitude
    too_high = altitude > self.cfg.max_altitude
    v_cmd[too_high, 2] = torch.clamp(
        v_cmd[too_high, 2],
        max=-self.cfg.altitude_bounce_velocity,
    )

    return v_cmd
```

---

## 6. AttackerManager FSM Integration

### 6.1 FSM States

```python
class FSMState(IntEnum):
    """Attacker finite state machine states."""
    APPROACH = 0    # Moving toward facility
    EVADE = 1       # Evading interceptor
    DEAD = 2        # Intercepted (deactivated)
    BREACH = 3      # Reached facility (episode failure)
```

### 6.2 State Transition Diagram

```
           ┌──────────────────────────────────────────────────────────┐
           │                                                          │
           │                     [spawned]                            │
           │                         │                                │
           │                         ▼                                │
           │                    ┌─────────┐                           │
           │       ┌───────────►│ APPROACH├────────┐                  │
           │       │            └─────────┘        │                  │
           │       │                 │             │                  │
           │ [evasion_timer <= 0]   │         [dist_to_facility       │
           │       │                │          < r_facility]          │
           │       │    [interceptor in                │              │
           │       │     INTERCEPT mode &&             │              │
           │       │     dist < d_evade]               │              │
           │       │                │                  │              │
           │       │                ▼                  ▼              │
           │   ┌───────┐                          ┌────────┐          │
           │   │ EVADE │◄──────────               │ BREACH │          │
           │   └───────┘                          └────────┘          │
           │       │                                   │              │
           │       │ [dist < d_capture &&              │              │
           │       │  interceptor.role == INTERCEPT]   │              │
           │       │                                   │              │
           │       ▼                                   ▼              │
           │   ┌──────┐                        [Episode Failure]      │
           │   │ DEAD │                                               │
           │   └──────┘                                               │
           │                                                          │
           └──────────────────────────────────────────────────────────┘
```

### 6.3 Mode Selection Logic

```python
def _compute_velocity_command(
    self,
    current_position: torch.Tensor,
    facility_position: torch.Tensor,
    interceptor_positions: torch.Tensor,
    interceptor_roles: torch.Tensor,
    curriculum_progress: float,
    dt: float,
) -> torch.Tensor:
    """Compute velocity command based on FSM state and behavior mode.

    Args:
        current_position: [N, 3] current positions.
        facility_position: [N_env, 3] facility positions.
        interceptor_positions: [N_env, N_defender, 3] defender positions.
        interceptor_roles: [N_env, N_defender] defender roles.
        curriculum_progress: Curriculum progress (0-1).
        dt: Timestep [s].

    Returns:
        v_cmd: [N, 3] velocity command.
    """
    n = current_position.shape[0]
    v_cmd = torch.zeros(n, 3, device=self.device)

    # Get alive targets only
    alive_mask = self.alive.view(-1)

    if not alive_mask.any():
        return v_cmd

    alive_indices = torch.where(alive_mask)[0]

    # Expand facility position to per-target
    facility_expanded = facility_position.unsqueeze(1).expand(-1, self.num_targets, -1).reshape(n, 3)

    # Separate by FSM state
    fsm_flat = self.fsm_state.view(-1)
    approach_mask = (fsm_flat[alive_indices] == FSMState.APPROACH)
    evade_mask = (fsm_flat[alive_indices] == FSMState.EVADE)

    approach_indices = alive_indices[approach_mask]
    evade_indices = alive_indices[evade_mask]

    # ------------------------------------------------------------------
    # Check evasion triggers for APPROACH targets
    # ------------------------------------------------------------------
    if len(approach_indices) > 0:
        dist_to_interceptors = self._compute_interceptor_distances(
            approach_indices, current_position, interceptor_positions, interceptor_roles
        )

        # Check if evasion should trigger (and agility > 0)
        agility = self.evasion_agility.view(-1)[approach_indices]
        should_evade = (dist_to_interceptors < self.cfg.evade_trigger_distance) & (agility > 0)

        if should_evade.any():
            evade_trigger_indices = approach_indices[should_evade]
            self._transition_to_evade(evade_trigger_indices)

            # Update masks
            approach_indices = approach_indices[~should_evade]

    # ------------------------------------------------------------------
    # Check evasion completion for EVADE targets
    # ------------------------------------------------------------------
    if len(evade_indices) > 0:
        evasion_complete = self.evasion_timer.view(-1)[evade_indices] <= 0

        if evasion_complete.any():
            return_indices = evade_indices[evasion_complete]
            self._transition_to_approach(return_indices)
            evade_indices = evade_indices[~evasion_complete]

    # ------------------------------------------------------------------
    # Compute velocities by state
    # ------------------------------------------------------------------
    # APPROACH state: use approach or observation mode
    if len(approach_indices) > 0:
        mode = self.velocity_mode.view(-1)[approach_indices]

        # Linear mode
        linear_mask = mode == VelocityMode.LINEAR
        if linear_mask.any():
            linear_indices = approach_indices[linear_mask]
            v_cmd[linear_indices] = self._velocity_generators['linear'].compute(
                indices=linear_indices,
                current_position=current_position[linear_indices],
                curriculum_progress=curriculum_progress,
                dt=dt,
            )

        # Circular mode
        circular_mask = mode == VelocityMode.CIRCULAR
        if circular_mask.any():
            circular_indices = approach_indices[circular_mask]
            v_cmd[circular_indices] = self._velocity_generators['circular'].compute(
                indices=circular_indices,
                current_position=current_position[circular_indices],
                curriculum_progress=curriculum_progress,
                dt=dt,
            )

        # Approach mode (attacker behavior)
        approach_mode_mask = mode == VelocityMode.APPROACH
        if approach_mode_mask.any():
            approach_mode_indices = approach_indices[approach_mode_mask]
            v_cmd[approach_mode_indices] = self._velocity_generators['approach'].compute(
                indices=approach_mode_indices,
                current_position=current_position[approach_mode_indices],
                facility_position=facility_expanded[approach_mode_indices],
                curriculum_progress=curriculum_progress,
            )

    # EVADE state: use evasion mode
    if len(evade_indices) > 0:
        v_cmd[evade_indices] = self._velocity_generators['evade'].compute(
            indices=evade_indices,
            current_position=current_position[evade_indices],
            interceptor_positions=interceptor_positions,
            interceptor_roles=interceptor_roles,
            curriculum_progress=curriculum_progress,
            dt=dt,
        )

    return v_cmd
```

---

## 7. Curriculum Schedule

### 7.1 Target-Specific Curriculum Dimensions

| Dimension | Start (p=0) | End (p=1) | Schedule (steps) | Description |
|-----------|-------------|-----------|------------------|-------------|
| `max_speed` | 3.0 m/s | 12.0 m/s | 20k-100k | Target maximum speed |
| `evasion_agility` | 0.0 | 1.0 | 100k-250k | Evasion intensity multiplier |
| `geofence_size` | 50m | 200m | 0-100k | Operating area half-width |
| `update_interval` | 6-12s | 2-4s | 0-60k | Direction change frequency |

### 7.2 Curriculum Functions

```python
def _get_max_speed(self, curriculum_progress: float) -> float:
    """Get curriculum-scaled maximum speed.

    Args:
        curriculum_progress: Progress value (0-1).

    Returns:
        Maximum speed [m/s].
    """
    return self.cfg.max_speed_start + curriculum_progress * (
        self.cfg.max_speed_end - self.cfg.max_speed_start
    )

def _get_agility_scale(self, curriculum_progress: float) -> float:
    """Get curriculum-scaled evasion agility multiplier.

    Evasion starts disabled and ramps up after 30% progress.

    Args:
        curriculum_progress: Progress value (0-1).

    Returns:
        Agility scale (0-1).
    """
    evasion_start_progress = 0.3  # Start at 30% progress

    if curriculum_progress < evasion_start_progress:
        return 0.0

    scaled_progress = (curriculum_progress - evasion_start_progress) / (1.0 - evasion_start_progress)
    return scaled_progress

def _sample_update_interval(self, n: int, curriculum_progress: float) -> torch.Tensor:
    """Sample direction change interval based on curriculum.

    Args:
        n: Number of samples.
        curriculum_progress: Progress value (0-1).

    Returns:
        [n] tensor of update intervals [s].
    """
    interval_min = self.cfg.update_interval_min_start + curriculum_progress * (
        self.cfg.update_interval_min_end - self.cfg.update_interval_min_start
    )
    interval_max = self.cfg.update_interval_max_start + curriculum_progress * (
        self.cfg.update_interval_max_end - self.cfg.update_interval_max_start
    )

    return torch.rand(n, device=self.device) * (interval_max - interval_min) + interval_min

def _get_geofence_size(self, curriculum_progress: float) -> float:
    """Get curriculum-scaled geofence size.

    Args:
        curriculum_progress: Progress value (0-1).

    Returns:
        Geofence half-width [m].
    """
    return self.cfg.geofence_min_size + curriculum_progress * (
        self.cfg.geofence_max_size - self.cfg.geofence_min_size
    )
```

### 7.3 Alignment with iris_ma6 Environment Curriculum

The target movement curriculum aligns with iris_ma6_env_spec.md §9.2:

| iris_ma6 Dimension | Target Movement Impact |
|--------------------|----------------------|
| #5 moving_target_speed (20k-100k) | Maps to `max_speed` scaling |
| #9 attacker_evasion (100k-250k) | Maps to `evasion_agility` scaling |
| #8 num_attackers (50k-200k) | Handled by environment, not target controller |

---

## 8. Testing Plan

### 8.1 Test Directory Structure

Following CLAUDE.md test guidelines:

```
target_controller/
└── tests/
    ├── __init__.py
    ├── run_tests.py                      # Standalone test runner
    ├── test_initialization.py            # Config loading, buffer allocation
    ├── test_linear_mode.py               # Linear velocity generation
    ├── test_circular_mode.py             # Circular orbit tracking
    ├── test_approach_mode.py             # Goal-directed approach variants
    ├── test_evade_mode.py                # Reactive evasion
    ├── test_drone_controller_integration.py  # Force/torque output validation
    ├── test_geofencing.py                # Boundary constraint behavior
    ├── test_altitude_constraints.py      # Min/max altitude enforcement
    ├── test_curriculum.py                # Curriculum scaling functions
    ├── test_fsm_transitions.py           # FSM state transitions
    └── README.md                         # Test documentation
```

### 8.2 Test Categories

1. **Initialization**
   - Configuration loading
   - Buffer allocation shapes
   - DroneController creation
   - Behavior profile assignment

2. **Linear Mode**
   - Direction changes at intervals
   - Velocity magnitude clamping
   - Curriculum-scaled intervals

3. **Circular Mode**
   - Orbit radius tracking
   - Phase updates
   - Position error correction

4. **Approach Mode**
   - Direct path computation
   - Waypoint navigation
   - Low-altitude constraints

5. **Evade Mode**
   - Trigger detection
   - Perpendicular direction computation
   - Timer countdown

6. **DroneController Integration**
   - Force/torque validity (no NaN/Inf)
   - No gimbal/zoom side effects
   - Attitude response to velocity command

7. **Geofencing**
   - Boundary detection
   - Velocity bounce behavior
   - Curriculum-scaled area

8. **Curriculum**
   - Speed scaling correctness
   - Agility ramping
   - Interval interpolation

### 8.3 Validation Criteria

- Forces/torques are non-NaN, non-Inf
- Velocity commands respect max_speed limits
- Altitude constraints are enforced
- FSM transitions occur at correct conditions
- Evasion triggers within distance threshold
- Curriculum scaling matches expected progressions

---

## 9. Implementation Roadmap

### Phase 1: Core Module (Estimated: Week 1)

1. Create `target_controller.py` with DroneController composition
2. Implement `TargetControllerCfg` dataclass
3. Port linear and circular modes from iris_ma5
4. Add geofencing and altitude constraints
5. Create unit tests for Phase 1 features

**Deliverable**: TargetController with linear/circular modes working

### Phase 2: Attacker Modes (Estimated: Week 2)

1. Implement approach mode with path variants
2. Implement evade mode with trigger logic
3. Add behavior profile randomization
4. Integrate FSM state management
5. Create unit tests for attacker modes

**Deliverable**: Full velocity generation with all 4 modes

### Phase 3: Environment Integration (Estimated: Week 3)

1. Integrate TargetController into iris_ma6 environment
2. Connect with AttackerManager FSM
3. Add curriculum schedule hooks
4. Validate end-to-end behavior
5. Performance optimization

**Deliverable**: Complete integration with iris_ma6 environment

---

## 10. Key Implementation Notes

### 10.1 Mass and Inertia

Targets use the same drone model as agents. Mass should be read from the physics asset at environment initialization:

```python
# In environment __init__
target_mass = self.target.root_physx_view.get_masses()[0].sum().item()

self._target_controller = TargetController(
    cfg=self.cfg.target_controller,
    mass=target_mass,  # Critical for correct thrust computation
    gravity=abs(self.cfg.sim.gravity[2]),
    num_targets=self.cfg.max_targets,
    num_envs=self.num_envs,
    device=self.device,
)
```

### 10.2 Multi-Target Support

The design supports multiple targets per environment. Tensor shape conventions:

```python
# Primary shapes:
# - current_position: [N_env, N_target, 3]
# - v_cmd: [N_env, N_target, 3]
# - F_body: [N_env, N_target, 3]
# - fsm_state: [N_env, N_target]
# - alive: [N_env, N_target]

# For DroneController (requires flattening):
# - v_cmd_flat: [N_env * N_target, 3]
# - F_body_flat: [N_env * N_target, 3]
```

### 10.3 Yaw Control

Unlike agents, targets do not need yaw control for gimbal pointing. Yaw rate command is always set to zero, allowing natural yaw drift from aerodynamics or disturbances:

```python
# In step():
F_body, tau_body, _, _ = self._drone_controller.step_policy(
    v_cmd=v_cmd,
    yaw_rate_cmd=torch.zeros(batch_size, device=self.device),  # Always zero
    gimbal_yaw_rate_cmd=torch.zeros(batch_size, device=self.device),
    gimbal_pitch_rate_cmd=torch.zeros(batch_size, device=self.device),
    zoom_rate_cmd=torch.zeros(batch_size, device=self.device),
    ...
)
```

### 10.4 Reset Behavior

On environment reset, the TargetController must reset:
- DroneController internal states (integrals, motor speeds)
- Velocity generator states (timers, phases, waypoints)
- FSM states (back to APPROACH)
- Behavior profiles (re-randomize)

```python
def reset(self, env_ids: torch.Tensor):
    """Reset target controller state for specified environments.

    Args:
        env_ids: Environment indices to reset.
    """
    # Reset DroneController
    for target_idx in range(self.num_targets):
        flat_ids = env_ids * self.num_targets + target_idx
        self._drone_controller.reset(flat_ids)

    # Reset velocity generators
    for generator in self._velocity_generators.values():
        generator.reset(env_ids)

    # Reset FSM to APPROACH
    self.fsm_state[env_ids] = FSMState.APPROACH

    # Reset alive status
    self.alive[env_ids] = True

    # Randomize behavior profiles
    self._assign_behavior_profiles(env_ids)
```

### 10.5 Dead Target Handling

Dead targets (intercepted or reached facility) should:
- Return zero velocity command
- Return zero force/torque
- Not participate in FSM transitions

```python
# In _compute_velocity_command:
alive_mask = self.alive.view(-1)

if not alive_mask.any():
    return torch.zeros(n, 3, device=self.device)

alive_indices = torch.where(alive_mask)[0]
# ... only process alive_indices
```

---

## Appendix A: Reference Files

| File | Purpose |
|------|---------|
| [iris_ma5/target_movement/target_movement.py](../../../iris_ma5/target_movement/target_movement.py) | Linear/circular mode reference |
| [iris_ma5/target_movement/target_movement_cfg.py](../../../iris_ma5/target_movement/target_movement_cfg.py) | Configuration reference |
| [controller/drone_controller.py](../controller/drone_controller.py) | DroneController to wrap |
| [controller/velocity_controller.py](../controller/velocity_controller.py) | Velocity-to-attitude conversion |
| [iris_ma6_env_spec.md](iris_ma6_env_spec.md) | Section 4: Attacker System |

## Appendix B: Comparison with iris_ma5 step() Interface

**iris_ma5 TargetMovement.step()**:
```python
def step(
    self,
    current_position: torch.Tensor,  # [N, 3]
    env_origins: torch.Tensor,        # [N, 3]
    curriculum_progress: float,
    dt: float,
    env_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:  # Returns velocity [N, 6]
```

**iris_ma6 TargetController.step()**:
```python
def step(
    self,
    current_position: torch.Tensor,    # [N_env, N_target, 3]
    current_velocity: torch.Tensor,    # [N_env, N_target, 3] -- NEW
    current_quat: torch.Tensor,        # [N_env, N_target, 4] -- NEW
    current_angular_vel: torch.Tensor, # [N_env, N_target, 3] -- NEW
    facility_position: torch.Tensor,   # [N_env, 3] -- NEW
    interceptor_positions: torch.Tensor,  # [N_env, N_defender, 3] -- NEW
    interceptor_roles: torch.Tensor,      # [N_env, N_defender] -- NEW
    curriculum_progress: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:  # Returns (F_body, tau_body)
```

Key differences:
1. Additional state inputs for DroneController (velocity, quaternion, angular velocity)
2. Attacker-specific inputs (facility, interceptors)
3. Returns forces/torques instead of velocities
4. Multi-target support built-in
