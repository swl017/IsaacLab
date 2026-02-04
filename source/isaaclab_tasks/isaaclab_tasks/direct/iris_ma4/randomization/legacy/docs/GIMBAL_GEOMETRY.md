# Gimbal Geometry and Feasible Target Zones

## Gimbal Angle Definition

### Pitch Angle (Critical Constraint)

```
              Up (Z+)
               |
               |  Target (above)
               | /
               |/ pitch < 0 (looking up)
    Agent ----●---- Horizontal
               |\
               | \ pitch > 0 (looking down)
               |  \
               |   Target (below)
```

**Sign Convention**:
- `pitch < 0`: Looking **upward** (target above horizontal)
- `pitch = 0`: Looking **horizontal**
- `pitch > 0`: Looking **downward** (target below horizontal)

**Current Limits**: `pitch ∈ [-45°, +10°]`
- Can look up to 45° above horizontal
- Can look down to 10° below horizontal

### Geometry Formula

Given:
- Agent position: `P_a = (x_a, y_a, z_a)`
- Target position: `P_t = (x_t, y_t, z_t)`

Compute:
```python
# Vector from agent to target
Δx = x_t - x_a
Δy = y_t - y_a
Δz = z_t - z_a

# Horizontal distance (in XY plane)
d_horizontal = sqrt(Δx² + Δy²)

# Pitch angle
pitch = atan2(-Δz, d_horizontal)
```

**Note**: The negative sign on `Δz` follows the gimbal convention where:
- Negative pitch → target higher than agent (`Δz > 0`)
- Positive pitch → target lower than agent (`Δz < 0`)

---

## Single Agent: Feasible Target Zone

For a single agent at position `(0, 0, z_a)`, the feasible target zone is a **cone** defined by pitch limits.

### Vertical Cross-Section (XZ plane)

```
        Z
        ^
        |           Pitch = -45° boundary
        |          /
     +  |         /
        |        /  Feasible
     z_a|-------● Agent     Zone
        |        \
        |         \
     -  |          \ Pitch = +10° boundary
        |           \
        +------------------------> X (horizontal distance)
                d_h
```

### Height Bounds at Distance `d_h`

```python
# For pitch ∈ [-45°, +10°]:
z_min(d_h) = z_a - d_h * tan(45°) = z_a - d_h
z_max(d_h) = z_a - d_h * tan(10°) ≈ z_a - 0.176 * d_h
```

**Example**: Agent at `z_a = 20m`, target at `d_h = 50m`
```
z_min = 20 - 50 * 1.0   = -30m  (far below agent)
z_max = 20 - 50 * 0.176 =  11.2m (slightly below agent)

Feasible range: z ∈ [-30m, 11.2m]
```

### 3D Visualization

The feasible zone is a **truncated cone**:

```
                    Top boundary (pitch = +10°)
                   ___----''''----___
                ,''                  '',
              ,'                        ',
             /           ● Agent          \
            ;            z_a               ;
            |                              |
             \                            /
              '.                        .'
                ''---....____....---''
                  Bottom boundary (pitch = -45°)
```

**Properties**:
- **Narrow cone** near agent
- **Widens** as distance increases
- **Asymmetric**: Can look down more than up (wait, reversed in this case!)
  - Actually: Can look **up** 45°, down only 10°
  - Zone extends **below** agent more than above

---

## Multi-Agent: Intersection of Feasible Zones

For **N agents**, the feasible target zone is the **intersection** of N cones.

### Example: 2 Agents at Different Heights

```
        Z
        ^
        |
     25 |     ● Agent 2 (z = 25m)
        |    /|\ Cone 2
        |   / | \
     20 | ● Agent 1 (z = 20m)
        |/|\  |  \
        / | \ |   \
       /  |  \|    \
      /   |  ⊗------\  Intersection Zone (feasible for both)
     /    | /  \     \
    /_____|/____\_____\________> X
                d_h
```

**Computation**:
```python
# Agent 1 constraints
z_min_1 = z_a1 - d_h * tan(45°)
z_max_1 = z_a1 - d_h * tan(10°)

# Agent 2 constraints
z_min_2 = z_a2 - d_h * tan(45°)
z_max_2 = z_a2 - d_h * tan(10°)

# Intersection
z_min_feasible = max(z_min_1, z_min_2)
z_max_feasible = min(z_max_1, z_max_2)

# Check if non-empty
if z_min_feasible > z_max_feasible:
    # No feasible solution!
    # Need to increase d_h or reduce height difference
```

### Effect of Formation Parameters

**Vertical spread** (agents at different heights):
```
Larger Δz_agents → Smaller intersection
```

**Horizontal spread** (agents apart in XY):
```
Larger d_h → Larger intersection (cones widen)
```

**Critical Distance**:

For agents with height difference `Δz_agents`, minimum distance where intersection exists:

```python
# Cone slope difference
slope_diff = tan(45°) - tan(10°) ≈ 1.0 - 0.176 = 0.824

# Minimum distance for non-empty intersection
d_h_min = Δz_agents / slope_diff ≈ 1.21 * Δz_agents
```

**Example**: Agents at `z=20m` and `z=25m` (Δz = 5m)
```
d_h_min ≈ 1.21 * 5 = 6.05m

If target is closer than 6m, some agents cannot see it!
```

---

## Practical Examples

### Example 1: Line Formation (Horizontal)

```
Agents:       ●-------●-------●
              1       2       3
z_a:          20      20      20  (same height)
x_a:         -10       0     +10

Target at:    x=50, y=0, z=15
```

**Analysis**:
```python
# All agents at same height → same z constraints
d_h ≈ sqrt((50-x_a)² + 0²)  # varies per agent
d_h_1 = 60m, d_h_2 = 50m, d_h_3 = 60m

# Pitch for agent 2 (closest)
pitch_2 = atan2(-(15-20), 50) = atan2(5, 50) = 5.7°
✓ Within [-45°, +10°]

# Pitch for agent 1,3 (farther)
pitch_1,3 = atan2(5, 60) = 4.8°
✓ Within limits

All agents can see target!
```

### Example 2: Line Formation (Vertical)

```
Agents (vertical line):
              ● 3 (z=25)
              |
              ● 2 (z=20)
              |
              ● 1 (z=15)

x_a = 0 for all

Target at: x=30, y=0, z=18
```

**Analysis**:
```python
d_h = 30m for all agents

# Agent 1 (z=15, target at z=18)
pitch_1 = atan2(-(18-15), 30) = atan2(-3, 30) = -5.7°
✓ Within [-45°, +10°]

# Agent 2 (z=20, target at z=18)
pitch_2 = atan2(-(18-20), 30) = atan2(2, 30) = 3.8°
✓ Within limits

# Agent 3 (z=25, target at z=18)
pitch_3 = atan2(-(18-25), 30) = atan2(7, 30) = 13.1°
✗ EXCEEDS +10° limit!

Agent 3 cannot see target!
```

**Fix**: Increase distance or raise target
```python
# Option 1: Move target farther
d_h = 50m
pitch_3 = atan2(7, 50) = 8.0°
✓ Now feasible

# Option 2: Raise target
z_target = 20m
pitch_3 = atan2(-(20-25), 30) = atan2(5, 30) = 9.5°
✓ Feasible
```

### Example 3: Grid Formation (3 agents)

```
Top view:         Side view:

    ● 2                ● 1,2,3 (z=20)
                       |
● 1   ● 3            Target (z=18)
                       |
```

**Setup**:
```python
Agent 1: (-5, -5, 20)
Agent 2: (-5, +5, 20)
Agent 3: (+5, -5, 20)

Target: (40, 0, 18)
```

**Analysis**:
```python
# Distance varies per agent
d_h_1 = sqrt((40-(-5))² + (0-(-5))²) = sqrt(45² + 5²) = 45.3m
d_h_2 = sqrt((40-(-5))² + (0-5)²)    = sqrt(45² + 5²) = 45.3m
d_h_3 = sqrt((40-5)² + (0-(-5))²)    = sqrt(35² + 5²) = 35.4m

# Pitch angles
pitch_1 = atan2(-(18-20), 45.3) = atan2(2, 45.3) = 2.5°  ✓
pitch_2 = atan2(2, 45.3) = 2.5°  ✓
pitch_3 = atan2(2, 35.4) = 3.2°  ✓

All feasible!
```

---

## Design Guidelines

### For Target Sampling:

1. **Ensure minimum distance**: `d_h > max_formation_spread * 2`
2. **Compute height intersection**: Use all agents' constraints
3. **Add safety margin**: Reduce limits by 5° (e.g., [-40°, +5°])
4. **Validate**: Always check actual pitch angles

### For Formation Design:

1. **Limit vertical spread**: Keep `Δz_agents < d_h * 0.8`
2. **Place agents at similar heights**: Reduces intersection shrinkage
3. **Horizontal formations preferred**: Easier to satisfy pitch constraints

### For Curriculum:

**Easy** (Early training):
- Small formation spread (< 10m)
- Agents at same height
- Target distance: 20-40m
- Target near formation height

**Hard** (Late training):
- Large formation spread (20-50m)
- Agents at varying heights (Δz up to 10m)
- Target distance: 50-100m
- Target at any feasible height

---

## Mathematical Summary

### Key Equations:

**Pitch angle**:
```
pitch = atan2(-(z_t - z_a), sqrt((x_t - x_a)² + (y_t - y_a)²))
```

**Feasible height at distance d_h**:
```
z_min(d_h) = z_a - d_h * tan(-pitch_max)
z_max(d_h) = z_a - d_h * tan(-pitch_min)
```

**Multi-agent intersection**:
```
Z_feasible = [max(z_min_i), min(z_max_i)] for all agents i
```

**Minimum target distance**:
```
d_h_min ≈ Δz_agents / (tan(pitch_max) - tan(pitch_min))
```

For `pitch ∈ [-45°, +10°]`:
```
d_h_min ≈ 1.21 * Δz_agents
```

---

## Conclusion

The **pitch angle constraint** is the dominant factor in target placement feasibility. By:

1. Computing the intersection of feasible zones for all agents
2. Sampling targets within this intersection
3. Maintaining sufficient horizontal distance

We can **guarantee** that all agents can point their gimbals at the target within mechanical limits, avoiding infeasible configurations entirely.
