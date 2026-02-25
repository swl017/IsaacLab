# Safety Module Architecture

## Overview

The `safety` module provides collision detection and time-to-collision (TTC) computation for multi-agent drone environments. It enables safe coordination by monitoring both inter-agent distances and camera-target approach dynamics.

## Design Philosophy

### Separation of Concerns

The module is structured around three main components:

1. **CollisionDetector**: Handles agent-agent collision detection
2. **TTCComputer**: Computes camera-target time-to-collision
3. **SafetyManager**: Unified interface coordinating both subsystems

This separation enables:
- Independent testing and validation of each component
- Flexible configuration (enable/disable features independently)
- Clear responsibility boundaries
- Reusability across different environments

### Batched Computation

All computations are fully batched across environments for GPU efficiency:
- Distance matrices computed via broadcasting
- TTC EMA filters operate on full batches
- No Python loops over environments

### Stateful Design

Both collision detection and TTC computation maintain internal state:
- **CollisionDetector**: Caches distance and collision matrices
- **TTCComputer**: Maintains EMA buffers for smoothing

This enables:
- Efficient incremental updates
- Smooth temporal behavior via filtering
- Environment-specific reset capability

## Module Structure

```
safety/
├── __init__.py                 # Module exports
├── collision_detector.py       # Inter-agent collision detection
├── ttc_computer.py            # Time-to-collision computation
├── safety_manager.py          # Unified safety management
├── ARCHITECTURE.md            # This file
└── API_REFERENCE.md          # Detailed API documentation
```

## Component Details

### 1. CollisionDetector

**Purpose**: Detect and prevent inter-agent collisions

**Key Features**:
- Pairwise distance computation between all agents
- Configurable minimum safe distance threshold
- Optional velocity-based collision prediction
- Per-agent penalty computation

**State**:
- `distance_matrix [N, A, A]`: Pairwise distances
- `collision_matrix [N, A, A]`: Boolean collision flags

**Typical Workflow**:
```python
# Update positions and detect collisions
collision_matrix = detector.detect_collisions(agent_positions)

# Get penalties for reward function
penalties = detector.compute_collision_penalties(agent_positions, agent_ids)
```

### 2. TTCComputer

**Purpose**: Estimate time-to-collision for camera-target tracking

**Key Features**:
- Zoom-invariant TTC estimation using looming cues
- Exponential moving average (EMA) smoothing
- Staleness decay for invalid detections
- Zoom motion gating

**Algorithm**:

The TTC computer implements a zoom-invariant approach based on the tau-dot theory of visual perception:

1. **Compute log-size and log-focal**:
   ```
   s = sqrt(bbox_width * bbox_height)  # Geometric mean of bbox dimensions
   f = sqrt(fx * fy)                    # Effective focal length
   g_s = log(s)
   g_f = log(f)
   ```

2. **Apply EMA smoothing**:
   ```
   g_s_smooth = alpha * g_s_prev + (1-alpha) * g_s
   g_f_smooth = alpha * g_f_prev + (1-alpha) * g_f
   ```

3. **Compute zoom-invariant quantity**:
   ```
   g = g_s_smooth - g_f_smooth = log(s/f)
   ```

   This ratio is invariant to zoom because:
   - Zooming increases both bbox size AND focal length proportionally
   - Their ratio remains constant for a fixed-size target at fixed distance

4. **Compute TTC from rate of change**:
   ```
   dg/dt = d/dt log(s/f)
   tau = -1 / (dg/dt)  # TTC in seconds
   ```

5. **Apply gates and penalties**:
   - Staleness gate: Reduce confidence for old detections
   - Zoom gate: Reduce penalty during active zoom operations
   - Map to [0,1] penalty: phi = (H - min(tau, H)) / H

**State**:
- `logsize_ema [N]`: EMA of log(bbox size)
- `logf_ema [N]`: EMA of log(focal length)
- `log_g_prev [N]`: Previous log(s/f) for derivative
- `time_since_valid [N]`: Time since last valid detection

**Mathematical Foundation**:

The approach is based on Lee's tau-dot theory (1976), extended for zoom-invariance:

- **Original tau-dot**: tau = -theta / (d(theta)/dt) where theta is visual angle
- **Zoom-invariant**: tau = -log(s/f) / d(log(s/f))/dt
- **Key insight**: log(s/f) is independent of zoom operations

**Typical Workflow**:
```python
# Compute TTC for one agent
phi, tau = ttc_computer.compute_ttc(
    bbox_width, bbox_height, valid_mask,
    fx, fy, dt
)

# phi: penalty in [0,1], tau: TTC in seconds
```

#### Staleness Decay Mechanism

**Purpose**: Gradually reduce TTC penalty when the target is no longer being detected.

**The Problem It Solves**:

Without staleness decay, when the camera loses track of the target (occlusion, target leaves FOV, detection failure), the last computed penalty persists indefinitely. This creates several issues:

1. **Unfair Penalties**: Agent penalized even with no visual contact
2. **Training Instability**: Discontinuous reward signals when detection toggles
3. **Exploitable Behavior**: Agent could look away to maintain low penalty

**How It Works**:

```python
# Track time since last valid detection
self.time_since_valid = torch.where(
    valid,
    torch.zeros_like(self.time_since_valid),  # Reset when valid
    self.time_since_valid + dt,                # Accumulate when invalid
)

# Exponential decay based on staleness
stale_decay = torch.exp(-self.time_since_valid / stale_half_life)
phi = phi * stale_decay  # Reduce penalty over time
```

**Decay Characteristics**:
- **Half-life parameter** controls decay rate (default: 1.0s)
- After 1 half-life: penalty reduced to 50%
- After 2 half-lives: penalty reduced to 25%
- After 3 half-lives: penalty reduced to 12.5%
- Approaches zero asymptotically

**Why Gradual Instead of Immediate Zeroing?**

| Aspect | Immediate Zeroing | Gradual Decay |
|--------|------------------|---------------|
| **Training Stability** | Discontinuous rewards | Smooth gradients |
| **Temporal Context** | Ignores recent history | Preserves collision risk |
| **Sensor Noise** | 1-frame dropout = lost target | Distinguishes glitch from loss |
| **Agent Behavior** | Can exploit by looking away | Maintains caution briefly |

**Real-World Scenario**:
```
Frame 100: Target visible, TTC=2s → penalty=0.67
Frame 101: Detection fails → Target likely still at ~2s distance
  With zeroing: penalty=0.00 (instant drop!)
  With decay:   penalty=0.65 (smooth transition)
Frame 110: Still no detection (0.1s elapsed)
  With decay:   penalty=0.61 (gradual fade)
Frame 150: Still no detection (0.5s elapsed, half-life)
  With decay:   penalty=0.33 (50% decay)
```

**Analogy**: Like eyes adjusting to darkness - vision fades gradually, not instantly.

#### Zoom Gate Mechanism

**Purpose**: Reduce TTC penalty during active zoom operations to prevent false collision warnings.

**The Problem It Solves**:

Although the algorithm is designed to be zoom-invariant via the `log(size/focal)` ratio, transient numerical effects occur during active zooming:

1. **EMA Lag**: Separate filters for size and focal create temporary mismatches
2. **Derivative Spikes**: Rapid focal changes can create artifact derivatives
3. **False Positives**: Zoom transients can trigger false collision warnings

**How It Works**:

```python
# Detect zoom motion from focal length changes
dlogf = (self.logf_ema - focal_ema_new).abs() / dt  # Rate of focal change

# Gate strength: high zoom rate → low gate value → suppress penalty
w_zoom = torch.exp(-dlogf / zoom_gate_k)
w_zoom = torch.clamp(w_zoom, 0.0, 1.0)

# Apply gate to penalty
phi = phi * w_zoom  # Reduced during zoom
```

**Gate Behavior**:

| Zoom Activity | dlogf | w_zoom | Effect on Penalty |
|---------------|-------|--------|-------------------|
| No zoom (constant f) | ≈0 | ≈1.0 | **Unaffected** |
| Slow zoom | Small | 0.8-1.0 | Slightly reduced |
| Fast zoom | Large | 0.0-0.3 | **Strongly suppressed** |

**Parameter Tuning**:
- **`zoom_gate_k`**: Controls gate aggressiveness
  - Small value (0.1-0.2): Aggressive suppression, more zoom tolerance
  - Large value (0.5-1.0): Mild suppression, maintain sensitivity
  - Default: 0.2 (balanced)

**Real-World Scenario**:
```
Scenario: Operator zooms in rapidly (focal: 400 → 800 in 0.5s)

Without zoom gate:
t=0.0s: f=400, penalty=0.10 (baseline)
t=0.1s: f=480, penalty=0.45 (spike from derivative artifact!)
t=0.2s: f=560, penalty=0.52 (false collision warning!)
t=0.5s: f=800, penalty=0.15 (settles after zoom)

With zoom gate (k=0.2):
t=0.0s: f=400, penalty=0.10 (baseline)
t=0.1s: f=480, penalty=0.08 (suppressed during zoom)
t=0.2s: f=560, penalty=0.06 (still suppressed)
t=0.5s: f=800, penalty=0.12 (smooth transition)
```

**Why Not Just Fix Zoom-Invariance?**

The zoom-invariance property (`log(s/f)` ratio) is mathematically sound in continuous time, but discretization introduces artifacts:

1. **Separate EMA Filters**: Size and focal have independent smoothing → temporary mismatch
2. **Discrete Derivatives**: Finite differences amplify high-frequency noise
3. **Asynchronous Updates**: Sensor update rates may differ

The zoom gate acts as a **safety net** for these numerical edge cases while preserving the zoom-invariant design.

**Test Evidence**: See `test_zooming_in_on_static_target()` which validates that pure zoom operations (with gate disabled) still produce low penalties, confirming the underlying zoom-invariance works. The gate only activates during transient zoom changes.

### 3. SafetyManager

**Purpose**: Unified interface for all safety computations

**Key Features**:
- Single entry point for safety penalties
- Coordinates collision and TTC subsystems
- Flexible enable/disable of features
- Batch operations across all agents

**Design Pattern**: Facade

The SafetyManager acts as a facade, providing:
- Simplified API for environment integration
- Consistent interface across features
- Automatic state management
- Bulk operations for efficiency

**Typical Workflow**:
```python
# Compute all safety penalties at once
penalties = safety_manager.compute_all_safety_penalties(
    agent_positions=positions,
    agent_bboxes=bboxes,
    agent_bbox_valid=valid_masks,
    agent_focal_lengths=focal_lengths,
    agent_ids=agent_ids,
    dt=timestep
)

# Returns: Dict[agent_id, Dict[penalty_type, tensor]]
# Example: penalties["drone_0"]["collision"] -> [N]
#          penalties["drone_0"]["ttc_penalty"] -> [N]
```

## Data Flow

### Collision Detection Path

```
agent_positions
    |
    v
CollisionDetector.compute_distances()
    |
    v
distance_matrix [N, A, A]
    |
    v
CollisionDetector.detect_collisions()
    |
    v
collision_matrix [N, A, A]
    |
    v
CollisionDetector.compute_collision_penalties()
    |
    v
penalties per agent [N]
```

### TTC Computation Path

```
bbox_dimensions, focal_lengths, valid_mask
    |
    v
TTCComputer.compute_ttc()
    |
    ├──> log(size), log(focal)
    ├──> EMA smoothing
    ├──> Compute derivative
    ├──> TTC = -1 / derivative
    └──> Apply gates & penalties
    |
    v
(phi [N], tau [N])
```

### Integrated Safety Path

```
SafetyManager.compute_all_safety_penalties()
    ├──> CollisionDetector (agent positions)
    └──> TTCComputer (bbox data)
    |
    v
Unified penalty dict per agent
```

## Configuration System

### Hierarchical Configuration

```python
SafetyManagerCfg
├── collision_cfg: CollisionDetectorCfg
│   ├── min_safe_distance: float
│   ├── enable_velocity_based: bool
│   ├── lookahead_time: float
│   └── collision_penalty_scale: float
└── ttc_cfg: TTCComputerCfg
    ├── horizon_sec: float
    ├── ema_alpha_s: float
    ├── ema_alpha_f: float
    ├── deriv_clip: float
    ├── stale_half_life: float
    ├── zoom_gate_k: float
    └── ttc_penalty_scale: float
```

### Configuration Best Practices

1. **Collision Detection**:
   - `min_safe_distance`: Set to 2-3x agent radius
   - `lookahead_time`: Typically 0.5-2.0 seconds

2. **TTC Computation**:
   - `horizon_sec`: Match to typical approach duration (4-8s)
   - `ema_alpha_s/f`: Higher = more smoothing (0.5-0.8 typical)
   - `deriv_clip`: Limit to prevent noise amplification (0.5-1.0)
   - `zoom_gate_k`: Lower = more aggressive zoom gating (0.1-0.3)
   - `stale_half_life`: Time for 50% confidence decay when target lost (0.5-2.0s)

## Integration with Environment

### Initialization

```python
from isaaclab_tasks.direct.iris_ma3.safety import SafetyManager, SafetyManagerCfg

# In __init__
self.safety_manager = SafetyManager(
    cfg=SafetyManagerCfg(
        collision_cfg=CollisionDetectorCfg(min_safe_distance=5.0),
        ttc_cfg=TTCComputerCfg(horizon_sec=6.0),
    ),
    num_envs=self.num_envs,
    num_agents=len(self.cfg.possible_agents),
    device=self.device,
)
```

### Reward Computation

```python
# In _get_rewards()
penalties = self.safety_manager.compute_all_safety_penalties(
    agent_positions={agent_id: robot.data.root_pos_w for agent_id, robot in self._robots.items()},
    agent_bboxes={agent_id: state.data.bboxes_2d[:, 0, :] for agent_id, state in delayed_states.items()},
    agent_bbox_valid={agent_id: state.data.bboxes_2d_valid_mask[:, 0] for agent_id, state in delayed_states.items()},
    agent_focal_lengths={agent_id: (intrinsics[:, 0, 0], intrinsics[:, 1, 1]) for agent_id, intrinsics in camera_intrinsics.items()},
    agent_ids=self.cfg.possible_agents,
    dt=self.step_dt,
)

# Use penalties in reward computation
for agent_id in self.cfg.possible_agents:
    collision_reward = penalties[agent_id]["collision"] * collision_scale
    ttc_reward = penalties[agent_id]["ttc_penalty"] * ttc_scale
```

### Reset

```python
# In _reset_idx()
self.safety_manager.reset(env_ids)

# Optional: Initialize TTC with current state for better initial estimates
self.safety_manager.reset_ttc_with_state(
    env_ids=env_ids,
    agent_bboxes=current_bboxes,
    agent_bbox_valid=current_valid,
    agent_focal_lengths=current_focal_lengths,
    agent_ids=self.cfg.possible_agents,
)
```

## Performance Considerations

### GPU Efficiency

- All operations are fully batched
- No Python loops over environments or agents
- Distance computation uses broadcasting (O(1) kernel launches)
- EMA updates are element-wise operations

### Memory Usage

Per environment:
- CollisionDetector: O(A²) for distance/collision matrices
- TTCComputer: O(A) for EMA buffers
- Total: O(N × A²) where N = num_envs, A = num_agents

Typical memory for N=4096, A=3:
- Collision: 4096 × 3 × 3 × 4 bytes = 148 KB
- TTC: 4096 × 3 × 4 floats × 4 bytes = 197 KB
- Total: ~350 KB (negligible)

### Computational Complexity

- Collision detection: O(N × A²) per step
- TTC computation: O(N × A) per step
- Typical cost: <1% of total environment step time

## Testing Strategy

### Unit Tests

Each component should be tested independently:

1. **CollisionDetector**:
   - Distance computation accuracy
   - Collision threshold behavior
   - Velocity-based prediction
   - Reset functionality

2. **TTCComputer**:
   - TTC estimation accuracy
   - EMA smoothing behavior
   - Zoom invariance property
   - Gate functionality

3. **SafetyManager**:
   - Integration correctness
   - Configuration propagation
   - Batch operations

### Integration Tests

Test the complete safety pipeline:
- Multi-agent collision scenarios
- Camera-target approach scenarios
- Combined safety penalty computation
- Reset and state management

## Future Extensions

### Potential Enhancements

1. **Collision Avoidance**:
   - Velocity field guidance
   - Repulsive potential fields
   - Trajectory prediction

2. **TTC Improvements**:
   - Multi-target TTC
   - Occlusion-aware TTC
   - Adaptive horizon based on velocity

3. **Additional Safety Metrics**:
   - Inter-agent communication quality
   - Formation maintenance
   - Geofence violations

4. **Performance Optimizations**:
   - Sparse collision checking (only nearby agents)
   - Adaptive TTC update rates
   - JIT compilation of critical paths

## References

1. Lee, D. N. (1976). "A theory of visual control of braking based on information about time-to-collision." Perception, 5(4), 437-459.

2. Regan, D., & Hamstra, S. J. (1993). "Dissociation of discrimination thresholds for time to contact and for rate of angular expansion." Vision Research, 33(4), 447-462.

3. Sun, H. J., & Frost, B. J. (1998). "Computation of different optical variables of looming objects in pigeon nucleus rotundus neurons." Nature Neuroscience, 1(4), 296-303.
