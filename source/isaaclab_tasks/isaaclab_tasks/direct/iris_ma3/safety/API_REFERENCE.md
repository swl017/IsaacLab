# Safety Module API Reference

Complete API reference for the safety module components.

## Table of Contents

- [CollisionDetector](#collisiondetector)
- [TTCComputer](#ttccomputer)
- [SafetyManager](#safetymanager)
- [Configuration Classes](#configuration-classes)

---

## CollisionDetector

### Class: `CollisionDetector`

Collision detector for multi-agent drone environments.

#### Constructor

```python
def __init__(
    self,
    cfg: CollisionDetectorCfg,
    num_envs: int,
    num_agents: int,
    device: torch.device,
)
```

**Parameters**:
- `cfg` (CollisionDetectorCfg): Configuration for the collision detector
- `num_envs` (int): Number of parallel environments
- `num_agents` (int): Number of agents per environment
- `device` (torch.device): Device for tensor computations

**Attributes**:
- `distance_matrix` (torch.Tensor): Pairwise distance matrix [N, A, A]
- `collision_matrix` (torch.Tensor): Boolean collision flags [N, A, A]

---

#### Method: `compute_distances`

Compute pairwise distances between all agents.

```python
def compute_distances(
    self,
    agent_positions: Dict[str, torch.Tensor]
) -> torch.Tensor
```

**Parameters**:
- `agent_positions` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to position tensor [N, 3]

**Returns**:
- torch.Tensor: Distance matrix [N, A, A] where element [n, i, j] is the distance between agent i and agent j in environment n

**Notes**:
- Diagonal elements are set to infinity (self-distance)
- Updates internal `distance_matrix` attribute

**Example**:
```python
positions = {
    "drone_0": robot0.data.root_pos_w,  # [N, 3]
    "drone_1": robot1.data.root_pos_w,  # [N, 3]
}
distances = detector.compute_distances(positions)  # [N, 2, 2]
```

---

#### Method: `detect_collisions`

Detect collisions between agents based on minimum safe distance.

```python
def detect_collisions(
    self,
    agent_positions: Dict[str, torch.Tensor]
) -> torch.Tensor
```

**Parameters**:
- `agent_positions` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to position tensor [N, 3]

**Returns**:
- torch.Tensor: Collision matrix [N, A, A] where element [n, i, j] is True if agents i and j are in collision in environment n

**Notes**:
- Automatically calls `compute_distances` internally
- Updates internal `collision_matrix` attribute
- Collision threshold is `cfg.min_safe_distance`

**Example**:
```python
collisions = detector.detect_collisions(positions)  # [N, 2, 2]
# collisions[0, 0, 1] == True means drone_0 and drone_1 collided in env 0
```

---

#### Method: `compute_collision_penalties`

Compute collision penalties for each agent.

```python
def compute_collision_penalties(
    self,
    agent_positions: Dict[str, torch.Tensor],
    agent_ids: list[str],
) -> Dict[str, torch.Tensor]
```

**Parameters**:
- `agent_positions` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to position tensor [N, 3]
- `agent_ids` (list[str]): List of agent identifiers in order

**Returns**:
- Dict[str, torch.Tensor]: Dictionary mapping agent_id to collision penalty tensor [N]

**Notes**:
- Penalty is -1.0 for each collision with another agent
- No self-collision penalty
- Penalty values are negative (for use in reward functions)

**Example**:
```python
penalties = detector.compute_collision_penalties(
    positions, ["drone_0", "drone_1"]
)
# penalties["drone_0"]: [N] with values in {0, -1, -2, ...}
```

---

#### Method: `predict_collisions`

Predict collisions based on current positions and velocities.

```python
def predict_collisions(
    self,
    agent_positions: Dict[str, torch.Tensor],
    agent_velocities: Dict[str, torch.Tensor],
) -> torch.Tensor
```

**Parameters**:
- `agent_positions` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to position tensor [N, 3]
- `agent_velocities` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to velocity tensor [N, 3]

**Returns**:
- torch.Tensor: Predicted collision matrix [N, A, A] at `lookahead_time`

**Notes**:
- Only active if `cfg.enable_velocity_based = True`
- Otherwise returns current collision matrix
- Predicts positions at time `t + lookahead_time`

**Example**:
```python
velocities = {
    "drone_0": robot0.data.root_lin_vel_w,
    "drone_1": robot1.data.root_lin_vel_w,
}
future_collisions = detector.predict_collisions(positions, velocities)
```

---

#### Method: `get_closest_agent_distance`

Get the distance to the closest agent for a specific agent.

```python
def get_closest_agent_distance(
    self,
    agent_id: str,
    agent_ids: list[str]
) -> torch.Tensor
```

**Parameters**:
- `agent_id` (str): Agent identifier
- `agent_ids` (list[str]): List of all agent identifiers

**Returns**:
- torch.Tensor: Distance to closest agent [N]

**Notes**:
- Requires `compute_distances` to be called first
- Returns minimum distance excluding self

**Example**:
```python
closest_dist = detector.get_closest_agent_distance(
    "drone_0", ["drone_0", "drone_1", "drone_2"]
)  # [N]
```

---

#### Method: `reset`

Reset collision detection state for specified environments.

```python
def reset(self, env_ids: torch.Tensor | None = None)
```

**Parameters**:
- `env_ids` (torch.Tensor | None): Environment indices to reset [K]. If None, reset all.

**Example**:
```python
# Reset all environments
detector.reset()

# Reset specific environments
detector.reset(torch.tensor([0, 5, 10], device=device))
```

---

## TTCComputer

### Class: `TTCComputer`

Time-to-collision computer using zoom-invariant looming cues.

#### Constructor

```python
def __init__(
    self,
    cfg: TTCComputerCfg,
    num_envs: int,
    device: torch.device,
)
```

**Parameters**:
- `cfg` (TTCComputerCfg): Configuration for the TTC computer
- `num_envs` (int): Number of parallel environments
- `device` (torch.device): Device for tensor computations

**Attributes**:
- `logsize_ema` (torch.Tensor): EMA of log(bbox size) [N]
- `logf_ema` (torch.Tensor): EMA of log(focal length) [N]
- `log_g_prev` (torch.Tensor): Previous log(s/f) for derivative [N]
- `time_since_valid` (torch.Tensor): Time since last valid detection [N]

---

#### Method: `compute_ttc`

Compute zoom-invariant looming TTC penalty.

```python
def compute_ttc(
    self,
    bbox_width: torch.Tensor,
    bbox_height: torch.Tensor,
    valid_mask: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

**Parameters**:
- `bbox_width` (torch.Tensor): Bbox width in normalized coordinates [N]
- `bbox_height` (torch.Tensor): Bbox height in normalized coordinates [N]
- `valid_mask` (torch.Tensor): Boolean mask for valid detections [N]
- `fx` (torch.Tensor): Focal length x in pixels [N]
- `fy` (torch.Tensor): Focal length y in pixels [N]
- `dt` (float): Timestep in seconds

**Returns**:
- Tuple[torch.Tensor, torch.Tensor]:
  - `phi`: TTC penalty in [0, 1] where 1 means imminent collision [N]
  - `tau`: Estimated TTC in seconds [N]

**Algorithm**:
1. Compute geometric mean of bbox dimensions: `s = sqrt(w * h)`
2. Compute effective focal length: `f = sqrt(fx * fy)`
3. Apply EMA smoothing to log(s) and log(f)
4. Compute zoom-invariant quantity: `g = log(s) - log(f) = log(s/f)`
5. Compute TTC: `Ä = -1 / (dg/dt)`
6. Map to penalty: `Æ = (H - min(Ä, H)) / H`
7. Apply staleness and zoom gates

**Notes**:
- Returns Æ=0 when Ä > horizon_sec
- Returns Æ=1 when Ä H 0 (imminent collision)
- EMA smoothing reduces noise sensitivity
- Zoom gate reduces penalty during active zoom

**Example**:
```python
phi, tau = ttc_computer.compute_ttc(
    bbox_width=bboxes[:, 2],   # [N]
    bbox_height=bboxes[:, 3],  # [N]
    valid_mask=valid,           # [N]
    fx=intrinsics[:, 0, 0],    # [N]
    fy=intrinsics[:, 1, 1],    # [N]
    dt=0.02,                    # 50 Hz
)
# phi: [N], values in [0, 1]
# tau: [N], values in [0, inf] seconds
```

---

#### Method: `reset`

Reset TTC buffers for specified environments.

```python
def reset(self, env_ids: torch.Tensor | None = None)
```

**Parameters**:
- `env_ids` (torch.Tensor | None): Environment indices to reset [K]. If None, reset all.

**Notes**:
- Sets all EMA buffers to zero
- Resets derivative tracking

**Example**:
```python
# Reset all
ttc_computer.reset()

# Reset specific environments
ttc_computer.reset(torch.tensor([0, 1, 2], device=device))
```

---

#### Method: `reset_with_current_state`

Reset TTC buffers with current bbox and focal length state.

```python
def reset_with_current_state(
    self,
    env_ids: torch.Tensor,
    bbox_width: torch.Tensor,
    bbox_height: torch.Tensor,
    valid_mask: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
)
```

**Parameters**:
- `env_ids` (torch.Tensor): Environment indices to reset [K]
- `bbox_width` (torch.Tensor): Current bbox width [N]
- `bbox_height` (torch.Tensor): Current bbox height [N]
- `valid_mask` (torch.Tensor): Current validity mask [N]
- `fx` (torch.Tensor): Current focal length x [N]
- `fy` (torch.Tensor): Current focal length y [N]

**Notes**:
- Initializes EMA buffers with current values (not zeros)
- Improves first few TTC estimates after reset
- Only updates specified environments

**Example**:
```python
ttc_computer.reset_with_current_state(
    env_ids=torch.tensor([0, 5], device=device),
    bbox_width=current_bboxes[:, 2],
    bbox_height=current_bboxes[:, 3],
    valid_mask=current_valid,
    fx=current_intrinsics[:, 0, 0],
    fy=current_intrinsics[:, 1, 1],
)
```

---

## SafetyManager

### Class: `SafetyManager`

Unified safety manager for multi-agent drone environments.

#### Constructor

```python
def __init__(
    self,
    cfg: SafetyManagerCfg,
    num_envs: int,
    num_agents: int,
    device: torch.device,
)
```

**Parameters**:
- `cfg` (SafetyManagerCfg): Configuration for the safety manager
- `num_envs` (int): Number of parallel environments
- `num_agents` (int): Number of agents per environment
- `device` (torch.device): Device for tensor computations

**Attributes**:
- `collision_detector` (CollisionDetector | None): Collision detector instance
- `ttc_computers` (Dict[int, TTCComputer] | None): TTC computer per agent

---

#### Method: `compute_collision_penalties`

Compute collision penalties for all agents.

```python
def compute_collision_penalties(
    self,
    agent_positions: Dict[str, torch.Tensor],
    agent_ids: list[str],
) -> Dict[str, torch.Tensor]
```

**Parameters**:
- `agent_positions` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to position tensor [N, 3]
- `agent_ids` (list[str]): List of agent identifiers

**Returns**:
- Dict[str, torch.Tensor]: Dictionary mapping agent_id to collision penalty tensor [N]

**Notes**:
- Returns zeros if collision detection is disabled
- Otherwise delegates to `CollisionDetector.compute_collision_penalties`

**Example**:
```python
penalties = safety_manager.compute_collision_penalties(
    agent_positions={
        "drone_0": robot0.data.root_pos_w,
        "drone_1": robot1.data.root_pos_w,
    },
    agent_ids=["drone_0", "drone_1"],
)
# penalties["drone_0"]: [N]
```

---

#### Method: `compute_ttc_penalties`

Compute TTC penalties for all agents.

```python
def compute_ttc_penalties(
    self,
    agent_bboxes: Dict[str, torch.Tensor],
    agent_bbox_valid: Dict[str, torch.Tensor],
    agent_focal_lengths: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    agent_ids: list[str],
    dt: float,
) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]
```

**Parameters**:
- `agent_bboxes` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to bbox tensor [N, 4] (x, y, w, h)
- `agent_bbox_valid` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to validity mask [N]
- `agent_focal_lengths` (Dict[str, tuple]): Dictionary mapping agent_id to (fx, fy) tensors [N]
- `agent_ids` (list[str]): List of agent identifiers
- `dt` (float): Timestep in seconds

**Returns**:
- Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Dictionary mapping agent_id to (phi, tau) where:
  - `phi`: TTC penalty in [0, 1] [N]
  - `tau`: Estimated TTC in seconds [N]

**Notes**:
- Returns (zeros, inf) if TTC computation is disabled
- Otherwise delegates to individual `TTCComputer` instances

**Example**:
```python
ttc_results = safety_manager.compute_ttc_penalties(
    agent_bboxes={
        "drone_0": bbox0,  # [N, 4]
        "drone_1": bbox1,  # [N, 4]
    },
    agent_bbox_valid={
        "drone_0": valid0,  # [N]
        "drone_1": valid1,  # [N]
    },
    agent_focal_lengths={
        "drone_0": (fx0, fy0),  # Each [N]
        "drone_1": (fx1, fy1),  # Each [N]
    },
    agent_ids=["drone_0", "drone_1"],
    dt=0.02,
)
# ttc_results["drone_0"] = (phi, tau)  # Each [N]
```

---

#### Method: `compute_all_safety_penalties`

Compute all safety penalties (collision + TTC) for all agents.

```python
def compute_all_safety_penalties(
    self,
    agent_positions: Dict[str, torch.Tensor],
    agent_bboxes: Dict[str, torch.Tensor],
    agent_bbox_valid: Dict[str, torch.Tensor],
    agent_focal_lengths: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    agent_ids: list[str],
    dt: float,
) -> Dict[str, Dict[str, torch.Tensor]]
```

**Parameters**:
- `agent_positions` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to position tensor [N, 3]
- `agent_bboxes` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to bbox tensor [N, 4]
- `agent_bbox_valid` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to validity mask [N]
- `agent_focal_lengths` (Dict[str, tuple]): Dictionary mapping agent_id to (fx, fy) [N]
- `agent_ids` (list[str]): List of agent identifiers
- `dt` (float): Timestep in seconds

**Returns**:
- Dict[str, Dict[str, torch.Tensor]]: Nested dictionary mapping agent_id to penalty dictionary:
  - `"collision"`: Collision penalty [N]
  - `"ttc_penalty"`: TTC penalty Æ [N]
  - `"ttc_value"`: TTC estimate Ä [N]

**Notes**:
- Single method for all safety computations
- Recommended entry point for most use cases
- Efficient batch processing

**Example**:
```python
penalties = safety_manager.compute_all_safety_penalties(
    agent_positions=positions,
    agent_bboxes=bboxes,
    agent_bbox_valid=valid_masks,
    agent_focal_lengths=focal_lengths,
    agent_ids=["drone_0", "drone_1"],
    dt=0.02,
)

# Access individual penalties
collision_penalty = penalties["drone_0"]["collision"]  # [N]
ttc_penalty = penalties["drone_0"]["ttc_penalty"]      # [N]
ttc_value = penalties["drone_0"]["ttc_value"]          # [N]
```

---

#### Method: `get_distance_matrix`

Get the current pairwise distance matrix between agents.

```python
def get_distance_matrix(self) -> torch.Tensor | None
```

**Returns**:
- torch.Tensor | None: Distance matrix [N, A, A] or None if collision detection is disabled

**Example**:
```python
distances = safety_manager.get_distance_matrix()  # [N, A, A]
```

---

#### Method: `get_collision_matrix`

Get the current collision matrix between agents.

```python
def get_collision_matrix(self) -> torch.Tensor | None
```

**Returns**:
- torch.Tensor | None: Collision matrix [N, A, A] or None if collision detection is disabled

**Example**:
```python
collisions = safety_manager.get_collision_matrix()  # [N, A, A]
```

---

#### Method: `reset`

Reset safety manager state for specified environments.

```python
def reset(self, env_ids: torch.Tensor | None = None)
```

**Parameters**:
- `env_ids` (torch.Tensor | None): Environment indices to reset [K]. If None, reset all.

**Notes**:
- Resets both collision detector and all TTC computers
- Call after episode termination

**Example**:
```python
# In _reset_idx()
self.safety_manager.reset(env_ids)
```

---

#### Method: `reset_ttc_with_state`

Reset TTC computers with current bbox and focal length state.

```python
def reset_ttc_with_state(
    self,
    env_ids: torch.Tensor,
    agent_bboxes: Dict[str, torch.Tensor],
    agent_bbox_valid: Dict[str, torch.Tensor],
    agent_focal_lengths: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    agent_ids: list[str],
)
```

**Parameters**:
- `env_ids` (torch.Tensor): Environment indices to reset [K]
- `agent_bboxes` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to bbox tensor [N, 4]
- `agent_bbox_valid` (Dict[str, torch.Tensor]): Dictionary mapping agent_id to validity mask [N]
- `agent_focal_lengths` (Dict[str, tuple]): Dictionary mapping agent_id to (fx, fy) [N]
- `agent_ids` (list[str]): List of agent identifiers

**Notes**:
- Initializes TTC EMA buffers with current state
- Improves TTC estimates immediately after reset
- Optional but recommended

**Example**:
```python
# In _reset_idx() after getting current state
self.safety_manager.reset_ttc_with_state(
    env_ids=env_ids,
    agent_bboxes=current_bboxes,
    agent_bbox_valid=current_valid,
    agent_focal_lengths=current_focal_lengths,
    agent_ids=self.cfg.possible_agents,
)
```

---

## Configuration Classes

### Class: `CollisionDetectorCfg`

Configuration for collision detector.

**Attributes**:

```python
@configclass
class CollisionDetectorCfg:
    min_safe_distance: float = 0.5
    """Minimum safe distance between agents (in meters)."""

    enable_velocity_based: bool = False
    """Whether to enable velocity-based collision prediction."""

    lookahead_time: float = 1.0
    """Time horizon for velocity-based collision prediction (seconds)."""

    collision_penalty_scale: float = -100.0
    """Scale factor for collision penalty in rewards."""
```

**Example**:
```python
cfg = CollisionDetectorCfg(
    min_safe_distance=5.0,       # 5 meters
    enable_velocity_based=True,
    lookahead_time=2.0,          # 2 seconds
    collision_penalty_scale=-200.0,
)
```

---

### Class: `TTCComputerCfg`

Configuration for time-to-collision computer.

**Attributes**:

```python
@configclass
class TTCComputerCfg:
    horizon_sec: float = 6.0
    """TTC horizon in seconds. TTC values beyond this are clamped."""

    ema_alpha_s: float = 0.6
    """EMA smoothing factor for log-size (0=no smoothing, 1=no update)."""

    ema_alpha_f: float = 0.6
    """EMA smoothing factor for log-focal length (0=no smoothing, 1=no update)."""

    deriv_clip: float = 0.5
    """Clip derivative to this range to avoid numerical instability."""

    stale_half_life: float = 1.0
    """Half-life for staleness decay when detections become invalid (seconds)."""

    zoom_gate_k: float = 0.2
    """Gate aggressiveness for zoom motion. Lower values are more aggressive."""

    eps: float = 1e-6
    """Small epsilon for numerical stability."""

    ttc_penalty_scale: float = -10.0
    """Scale factor for TTC penalty in rewards."""
```

**Example**:
```python
cfg = TTCComputerCfg(
    horizon_sec=8.0,           # 8 second horizon
    ema_alpha_s=0.7,           # More smoothing
    ema_alpha_f=0.7,
    deriv_clip=0.3,            # Tighter clipping
    stale_half_life=1.5,       # Longer decay
    zoom_gate_k=0.15,          # More aggressive gating
    ttc_penalty_scale=-20.0,
)
```

---

### Class: `SafetyManagerCfg`

Configuration for unified safety manager.

**Attributes**:

```python
@configclass
class SafetyManagerCfg:
    collision_cfg: CollisionDetectorCfg = CollisionDetectorCfg()
    """Configuration for collision detection."""

    ttc_cfg: TTCComputerCfg = TTCComputerCfg()
    """Configuration for TTC computation."""

    enable_collision_detection: bool = True
    """Whether to enable collision detection."""

    enable_ttc_computation: bool = True
    """Whether to enable TTC computation."""
```

**Example**:
```python
cfg = SafetyManagerCfg(
    collision_cfg=CollisionDetectorCfg(min_safe_distance=5.0),
    ttc_cfg=TTCComputerCfg(horizon_sec=8.0),
    enable_collision_detection=True,
    enable_ttc_computation=True,
)
```

---

## Complete Usage Example

```python
from isaaclab_tasks.direct.iris_ma3.safety import SafetyManager, SafetyManagerCfg
from isaaclab_tasks.direct.iris_ma3.safety import CollisionDetectorCfg, TTCComputerCfg

# In environment __init__
self.safety_manager = SafetyManager(
    cfg=SafetyManagerCfg(
        collision_cfg=CollisionDetectorCfg(
            min_safe_distance=5.0,
            collision_penalty_scale=-100.0,
        ),
        ttc_cfg=TTCComputerCfg(
            horizon_sec=6.0,
            ttc_penalty_scale=-10.0,
        ),
        enable_collision_detection=True,
        enable_ttc_computation=True,
    ),
    num_envs=self.num_envs,
    num_agents=len(self.cfg.possible_agents),
    device=self.device,
)

# In _get_rewards()
# Prepare data
agent_positions = {
    agent_id: robot.data.root_pos_w
    for agent_id, robot in self._robots.items()
}
agent_bboxes = {
    agent_id: delayed_state.data.bboxes_2d[:, 0, :]
    for agent_id, delayed_state in delayed_states.items()
}
agent_bbox_valid = {
    agent_id: delayed_state.data.bboxes_2d_valid_mask[:, 0]
    for agent_id, delayed_state in delayed_states.items()
}
agent_focal_lengths = {
    agent_id: (
        delayed_state.data.camera_intrinsics[:, 0, 0],
        delayed_state.data.camera_intrinsics[:, 1, 1],
    )
    for agent_id, delayed_state in delayed_states.items()
}

# Compute all safety penalties
safety_penalties = self.safety_manager.compute_all_safety_penalties(
    agent_positions=agent_positions,
    agent_bboxes=agent_bboxes,
    agent_bbox_valid=agent_bbox_valid,
    agent_focal_lengths=agent_focal_lengths,
    agent_ids=self.cfg.possible_agents,
    dt=self.step_dt,
)

# Use in reward computation
for agent_id in self.cfg.possible_agents:
    collision_reward = (
        safety_penalties[agent_id]["collision"]
        * self.cfg.collision_penalty_scale
        * self.step_dt
        * self.progress_safety  # Curriculum scaling
    )
    ttc_reward = (
        safety_penalties[agent_id]["ttc_penalty"]
        * self.cfg.ttc_penalty_scale
        * self.step_dt
        * self.progress_safety  # Curriculum scaling
    )

    total_reward = ... + collision_reward + ttc_reward

# In _reset_idx()
self.safety_manager.reset(env_ids)

# Optional: Initialize with current state
self.safety_manager.reset_ttc_with_state(
    env_ids=env_ids,
    agent_bboxes=current_bboxes,
    agent_bbox_valid=current_valid,
    agent_focal_lengths=current_focal_lengths,
    agent_ids=self.cfg.possible_agents,
)
```
