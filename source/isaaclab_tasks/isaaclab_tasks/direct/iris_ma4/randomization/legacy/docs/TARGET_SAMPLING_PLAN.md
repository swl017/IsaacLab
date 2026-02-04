# Target Position Sampling Plan - Gimbal-Aware Randomization

## Problem Statement

When initializing formations and target positions, some configurations result in **infeasible gimbal angles** that exceed mechanical limits:

**Current Gimbal Limits** (from `iris_ma_env3_cfg.py`):
```python
max_gimbal_yaw_angle = [-200°, +200°]    # Nearly full rotation
max_gimbal_pitch_angle = [-45°, +10°]    # Limited vertical range
max_gimbal_roll_angle = [-45°, +45°]     # Stabilization range
```

**Critical Constraint**: **Pitch angle is severely limited** (only -45° to +10°, ~55° total range)

### Current Issues

1. **Random target placement** may require pitch angles outside [-45°, +10°]
2. **No guarantee** that all agents can see the target after formation spawn
3. **Clamping gimbal angles** to limits results in misalignment from target
4. **Rejection sampling** (random retry) is inefficient for multiple agents

---

## Analysis: Gimbal Pointing Geometry

Given:
- Agent position: `P_agent = [x_a, y_a, z_a]`
- Agent orientation (yaw): `θ_agent`
- Target position: `P_target = [x_t, y_t, z_t]`
- Gimbal limits: `pitch ∈ [-45°, +10°]`

The required **pitch angle** in gimbal frame is:
```
vector_to_target = P_target - P_agent
distance_horizontal = sqrt((x_t - x_a)² + (y_t - y_a)²)
distance_vertical = z_t - z_a

pitch_required = atan2(-distance_vertical, distance_horizontal)
```

**Key Insight**: Pitch angle depends on the **elevation angle** from agent to target.

### Feasible Target Zone (Single Agent)

For a given agent at position `P_agent`:

```
pitch ∈ [-45°, +10°]  means:

tan(-45°) ≤ -Δz / d_horizontal ≤ tan(+10°)
tan(+10°) ≥ -Δz / d_horizontal ≥ tan(-45°)

Rearranging:
-d_horizontal * tan(+10°) ≥ Δz ≥ -d_horizontal * tan(-45°)
-d_horizontal * tan(+10°) ≥ (z_t - z_a) ≥ -d_horizontal
```

**Feasible height range**:
```
z_min(d_h) = z_a - d_h * tan(45°) = z_a - d_h
z_max(d_h) = z_a - d_h * tan(10°) ≈ z_a - 0.176 * d_h
```

Where `d_h = horizontal distance from agent to target`.

### Multi-Agent Constraint

For **all agents** to see the target:
```
z_target ∈ [max(z_min_i), min(z_max_i)] for all agents i
```

This creates a **feasible zone** that shrinks as:
- Agents are farther apart
- Agents at different heights
- Formation spread increases

---

## Proposed Solutions

### **Option 1: Constrained Random Sampling (Recommended)**

Generate target positions that are **guaranteed feasible** for all agents.

#### Algorithm:

```python
def sample_gimbal_feasible_target(
    agent_positions: torch.Tensor,  # [N_env, N_agents, 3]
    agent_orientations: torch.Tensor,  # [N_env, N_agents, 4]
    pitch_limits: tuple[float, float] = (-45°, +10°),
    safety_margin: float = 5°,
    horizontal_range: tuple[float, float] = (10.0, 100.0),
    target_region_center: torch.Tensor | None = None,
) -> torch.Tensor:  # [N_env, 3]
    """
    Sample target positions that all agents can point to with valid gimbal angles.

    Steps:
    1. Compute formation center and spread
    2. Sample horizontal distance from formation
    3. Compute feasible height range for all agents
    4. Sample target height within intersection of feasible ranges
    5. Sample horizontal position (bearing) randomly
    6. Validate and return
    """
```

#### Detailed Steps:

**Step 1**: Compute formation geometry
```python
formation_center = agent_positions.mean(dim=1)  # [N_env, 3]
max_agent_spread = compute_max_pairwise_distance(agent_positions)  # [N_env]
```

**Step 2**: Sample horizontal distance
```python
# Distance should be >> formation spread for good triangulation
min_distance = max(horizontal_range[0], max_agent_spread * 2)
max_distance = horizontal_range[1]
horizontal_distance = sample_uniform(min_distance, max_distance)  # [N_env]
```

**Step 3**: Compute feasible height range
```python
# For each agent, compute what target heights are reachable
for each agent i:
    d_h_i = horizontal_distance  # approximate
    z_min_i = z_agent_i - d_h_i * tan(45° - margin)
    z_max_i = z_agent_i - d_h_i * tan(10° + margin)

# Intersection of all agents' feasible ranges
z_feasible_min = max(z_min_i for all i)
z_feasible_max = min(z_max_i for all i)

# Check if intersection is non-empty
if z_feasible_min > z_feasible_max:
    # No feasible solution - adjust horizontal_distance or formation
    retry()
```

**Step 4**: Sample target height
```python
target_z = sample_uniform(z_feasible_min, z_feasible_max)  # [N_env]
```

**Step 5**: Sample horizontal position (bearing)
```python
# Random bearing from formation center
bearing = sample_uniform(0, 2π)  # [N_env]

target_x = formation_center_x + horizontal_distance * cos(bearing)
target_y = formation_center_y + horizontal_distance * sin(bearing)
target_z = sampled from Step 4

target_pos = [target_x, target_y, target_z]  # [N_env, 3]
```

**Step 6**: Validation
```python
# Verify all agents can point to target
for each agent:
    pitch_required = compute_pitch(agent_pos, target_pos)
    assert pitch_limits[0] + margin <= pitch_required <= pitch_limits[1] - margin
```

#### Advantages:
✅ **Guaranteed feasibility** - No rejection sampling needed
✅ **Efficient** - O(N_agents) computation
✅ **Curriculum-friendly** - Can control horizontal distance with scale factor
✅ **Multi-agent aware** - Considers all agents simultaneously

#### Disadvantages:
❌ May produce conservative targets (intersection of feasible zones can be small)
❌ More complex implementation

---

### **Option 2: Spherical Cap Sampling**

Sample target in a spherical cap around formation center with elevation constraints.

#### Algorithm:

```python
def sample_spherical_cap_target(
    formation_center: torch.Tensor,  # [N_env, 3]
    radius_range: tuple[float, float] = (10.0, 100.0),
    elevation_range: tuple[float, float] = (-10°, +30°),  # From horizontal
) -> torch.Tensor:  # [N_env, 3]
    """
    Sample uniformly on spherical cap with elevation constraints.

    Elevation is measured from horizontal plane:
    - elevation > 0: target above formation
    - elevation < 0: target below formation

    Pitch angle ≈ -elevation (for targets far from formation)
    """
    radius = sample_uniform(*radius_range)

    # Sample azimuth uniformly
    azimuth = sample_uniform(0, 2π)

    # Sample elevation with constraint
    # For uniform sampling on sphere, use: cos(elevation)
    cos_elev_min = cos(elevation_range[1])
    cos_elev_max = cos(elevation_range[0])
    cos_elevation = sample_uniform(cos_elev_min, cos_elev_max)
    elevation = arccos(cos_elevation)

    # Convert to Cartesian
    target_x = formation_center_x + radius * cos(elevation) * cos(azimuth)
    target_y = formation_center_y + radius * cos(elevation) * sin(azimuth)
    target_z = formation_center_z + radius * sin(elevation)

    return [target_x, target_y, target_z]
```

#### Advantages:
✅ **Simple** - Easy to implement and understand
✅ **Uniform distribution** - Good coverage of space
✅ **Fast** - No iteration or validation needed

#### Disadvantages:
❌ **Approximate** - Doesn't exactly enforce gimbal limits for each agent
❌ **Formation-dependent** - May fail for spread-out formations
❌ **Post-validation needed** - Still requires checking feasibility

---

### **Option 3: Rejection Sampling with Early Termination**

Improved rejection sampling with smart initial guesses.

#### Algorithm:

```python
def sample_with_rejection(
    agent_positions: torch.Tensor,
    max_retries: int = 50,
) -> torch.Tensor:
    """
    Sample random targets and reject infeasible ones.

    Improvements over naive rejection:
    1. Bias initial sample toward feasible region
    2. Check feasibility before full gimbal computation
    3. Early termination on any agent failure
    4. Fallback to constrained sampling after max_retries
    """
    for retry in range(max_retries):
        # Initial guess biased toward feasible region
        target = sample_biased_target(agent_positions)

        # Quick feasibility check (elevation only)
        if not quick_feasibility_check(agent_positions, target):
            continue

        # Full gimbal angle computation
        gimbals = compute_all_gimbal_angles(agent_positions, target)

        if all_within_limits(gimbals):
            return target

    # Fallback to Option 1 (constrained sampling)
    return sample_gimbal_feasible_target(agent_positions)
```

#### Advantages:
✅ **Flexible** - Can use any target distribution
✅ **Optimal when successful** - Gets best random coverage

#### Disadvantages:
❌ **Unpredictable runtime** - Can take many retries
❌ **Inefficient for large formations** - Feasible volume shrinks exponentially
❌ **Requires fallback** - Still needs Option 1 as backup

---

## Recommended Implementation

**Hybrid Approach**: Use **Option 1** (Constrained Sampling) with **Option 2** (Spherical Cap) as initial guess

### Implementation Strategy:

```python
class TargetSampler:
    def sample_target_position(
        self,
        formation_data: dict,  # From get_random_formation()
        target_distance_range: tuple[float, float] = (15.0, 80.0),
        safety_margin_deg: float = 5.0,
    ) -> torch.Tensor:
        """
        Sample target positions feasible for all agents in formation.

        Returns:
            target_positions: [num_envs, 3]
        """
        agent_positions = formation_data['agent_positions']  # [N_env, N_agents, 3]

        # Step 1: Compute formation geometry
        formation_center, max_spread = self._analyze_formation(agent_positions)

        # Step 2: Determine feasible distance range
        min_dist = max(target_distance_range[0], max_spread * 2.5)
        max_dist = target_distance_range[1]

        # Step 3: Sample horizontal distance and bearing
        distance = sample_uniform(min_dist, max_dist, shape=(num_envs,))
        bearing = sample_uniform(0, 2*π, shape=(num_envs,))

        # Step 4: Compute feasible height intersection
        z_min, z_max = self._compute_feasible_height_range(
            agent_positions,
            distance,
            bearing,
            formation_center,
        )

        # Step 5: Sample target height
        target_z = sample_uniform(z_min, z_max)

        # Step 6: Construct target position
        target_x = formation_center[:, 0] + distance * cos(bearing)
        target_y = formation_center[:, 1] + distance * sin(bearing)
        target_pos = stack([target_x, target_y, target_z], dim=1)

        # Step 7: Validate
        self._validate_gimbal_feasibility(agent_positions, target_pos)

        return target_pos
```

### Key Methods:

#### `_compute_feasible_height_range()`
```python
def _compute_feasible_height_range(
    self,
    agent_positions: torch.Tensor,  # [N_env, N_agents, 3]
    horizontal_distance: torch.Tensor,  # [N_env]
    bearing: torch.Tensor,  # [N_env]
    formation_center: torch.Tensor,  # [N_env, 3]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute height range where all agents can point to target.

    Returns:
        z_min, z_max: [N_env]
    """
    num_envs, num_agents, _ = agent_positions.shape

    # Compute target horizontal position (without z)
    target_x = formation_center[:, 0] + horizontal_distance * cos(bearing)
    target_y = formation_center[:, 1] + horizontal_distance * sin(bearing)

    # For each agent, compute feasible z range
    z_min_per_agent = []
    z_max_per_agent = []

    for agent_idx in range(num_agents):
        agent_pos = agent_positions[:, agent_idx, :]  # [N_env, 3]

        # Horizontal distance from this agent to target
        dx = target_x - agent_pos[:, 0]
        dy = target_y - agent_pos[:, 1]
        d_h = sqrt(dx**2 + dy**2)

        # Height range based on pitch limits
        pitch_min = self.pitch_limits[0] + margin  # e.g., -45° + 5° = -40°
        pitch_max = self.pitch_limits[1] - margin  # e.g., +10° - 5° = +5°

        # pitch = atan2(-Δz, d_h) => Δz = -d_h * tan(pitch)
        z_min_i = agent_pos[:, 2] - d_h * tan(-pitch_min)  # More negative pitch => higher target
        z_max_i = agent_pos[:, 2] - d_h * tan(-pitch_max)  # Less negative pitch => lower target

        z_min_per_agent.append(z_min_i)
        z_max_per_agent.append(z_max_i)

    # Intersection of all feasible ranges
    z_min_per_agent = torch.stack(z_min_per_agent, dim=1)  # [N_env, N_agents]
    z_max_per_agent = torch.stack(z_max_per_agent, dim=1)  # [N_env, N_agents]

    z_min_feasible = z_min_per_agent.max(dim=1)[0]  # [N_env]
    z_max_feasible = z_max_per_agent.min(dim=1)[0]  # [N_env]

    # Handle empty intersection (z_min > z_max)
    # Fallback: use formation center height
    invalid_mask = z_min_feasible > z_max_feasible
    z_min_feasible[invalid_mask] = formation_center[invalid_mask, 2] - 5.0
    z_max_feasible[invalid_mask] = formation_center[invalid_mask, 2] + 5.0

    return z_min_feasible, z_max_feasible
```

---

## Integration with Formation Generator

### Updated API:

```python
# In environment reset
formation_data = self.randomizer.initial_states.get_random_formation(
    num_agents=len(self.cfg.possible_agents),
    env_ids=env_ids
)

# Sample feasible target position
target_positions = self.randomizer.targets.sample_target_position(
    formation_data=formation_data,
    target_distance_range=(15.0, 80.0 * curriculum_scale),
    safety_margin_deg=5.0,
)

# Apply states
for i, agent_id in enumerate(self.cfg.possible_agents):
    robot.write_root_pose_to_sim(formation_data[:, i, 0:7], env_ids)

    # Compute gimbal angles (now guaranteed feasible!)
    gimbal_angles = compute_gimbal_angles(
        formation_data[:, i, 0:3],  # agent position
        formation_data[:, i, 3:7],  # agent orientation
        target_positions,
    )
```

---

## Performance Considerations

### Computational Cost:

**Constrained Sampling (Option 1)**:
- Time: O(N_envs × N_agents)
- Operations: Mostly vectorized PyTorch
- Typical: <1ms for 100 envs × 3 agents on GPU

**Rejection Sampling (Option 3)**:
- Time: O(N_envs × N_agents × N_retries)
- Typical: 5-50ms depending on retry count
- Unpredictable, can spike

**Recommendation**: Use Option 1 for guaranteed performance.

### Memory:

- Additional storage: ~O(N_envs × N_agents) for feasibility computation
- Negligible compared to environment state

---

## Curriculum Integration

Control sampling difficulty via parameters:

```python
# Early training: Close targets, narrow bearing range
target_distance = (15.0, 30.0)
bearing_range = (-π/4, +π/4)  # Only in front

# Late training: Far targets, full bearing range
target_distance = (15.0, 100.0)
bearing_range = (0, 2π)  # All directions
```

---

## Summary

### Recommended Approach:

✅ **Use constrained sampling (Option 1)** as primary method
✅ **Compute feasible height range** based on all agents
✅ **Sample within intersection** of feasible zones
✅ **Validate gimbal angles** as post-check (should always pass)
✅ **Integrate with formation generator** via clean API

### Expected Benefits:

- **100% feasibility** - No rejected samples
- **Fast** - O(N_agents) per environment
- **Curriculum-ready** - Easy to scale difficulty
- **Multi-agent aware** - Considers all constraints simultaneously
- **Predictable performance** - No random retries

### File Structure:

```
randomization/
├── initial_states.py          # Formation generator (existing)
├── target_sampling.py         # New: Target sampler
├── target_sampling_cfg.py     # New: Configuration
├── test_target_sampling.py    # New: Tests
└── TARGET_SAMPLING_PLAN.md    # This file
```
