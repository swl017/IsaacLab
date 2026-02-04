# Configuration Tuning Guide

Guide for tuning formation and target sampling parameters to achieve desired behavior.

---

## Problem: Targets Too Far Away

**Symptom:** Targets spawn far from formations even with low `target_distance_max`

**Root Cause:** Formation spread × distance_scale_factor exceeds configured max distance

---

## Understanding the Relationship

```
Required Minimum Distance = max_formation_spread × distance_scale_factor

Where:
  max_formation_spread = worst-case distance between furthest agents
  distance_scale_factor = safety multiplier (default 2.5)
```

**Example:**
- 3 agents in line with 12m spacing
- `max_spread = 2 × 12 = 24 meters`
- `min_distance = 24 × 2.5 = 60 meters`
- Even if `target_distance_max = 20m`, targets will be at ~60m!

---

## Quick Fixes

### Fix 1: Tighter Formations (Recommended)

Add to `iris_ma_env3.py` `__init__()`:

```python
# After: self.randomizer = Randomizer(...)

# Reduce formation spread
self.randomizer.initial_states.cfg.max_agent_separation = 3.0  # Default: 12.0
self.randomizer.initial_states.cfg.min_agent_separation = 2.0  # Default: 3.0
self.randomizer.initial_states.cfg.z_variation_range = (0.0, 1.0)  # Default: (0.0, 5.0)
```

**Result:** Formation spread ~6m → min distance ~15m → fits in [15, 20]m range ✓

### Fix 2: Smaller Distance Scale

Add to `iris_ma_env3.py` `__init__()`:

```python
# After: self.randomizer = Randomizer(...)

# Allow targets closer to formation
self.randomizer.targets.cfg.distance_scale_factor = 1.0  # Default: 2.5
```

**Result:** Same formation spread but targets can be 1× (not 2.5×) away

⚠️ **Warning:** Lower scale factors may place targets inside formations for large spreads

### Fix 3: Increase Max Distance

Modify `target_sampling.py` `TargetSamplerCfg`:

```python
@configclass
class TargetSamplerCfg:
    target_distance_min: float = 15.0
    target_distance_max: float = 60.0  # Increase from 20.0
```

**Result:** Wider target distance range accommodates larger formations

---

## Configuration Matrix

### For Close Targets (15-25m)

```python
# Formation Config
max_agent_separation = 3.0
min_agent_separation = 2.0
z_variation_range = (0.0, 1.0)

# Target Config
target_distance_min = 15.0
target_distance_max = 25.0
distance_scale_factor = 2.0

# Result: Compact formations, close targets
# Max formation spread: ~6m
# Target distance: 15-25m
```

### For Medium Targets (20-50m)

```python
# Formation Config
max_agent_separation = 6.0
min_agent_separation = 3.0
z_variation_range = (0.0, 2.0)

# Target Config
target_distance_min = 20.0
target_distance_max = 50.0
distance_scale_factor = 2.5

# Result: Medium formations, medium targets
# Max formation spread: ~12m
# Target distance: 30-50m
```

### For Far Targets (30-80m)

```python
# Formation Config
max_agent_separation = 12.0  # Default
min_agent_separation = 3.0   # Default
z_variation_range = (0.0, 5.0)  # Default

# Target Config
target_distance_min = 30.0
target_distance_max = 80.0
distance_scale_factor = 2.5  # Default

# Result: Large formations, far targets
# Max formation spread: ~24m
# Target distance: 60-80m
```

---

## Diagnostic Commands

### Check Current Configuration

```python
# In iris_ma_env3.py or debug script:

print("=== Formation Config ===")
print(f"Min separation: {self.randomizer.initial_states.cfg.min_agent_separation}m")
print(f"Max separation: {self.randomizer.initial_states.cfg.max_agent_separation}m")
print(f"Z variation: {self.randomizer.initial_states.cfg.z_variation_range}")

print("\n=== Target Config ===")
print(f"Distance range: [{self.randomizer.targets.cfg.target_distance_min}, "
      f"{self.randomizer.targets.cfg.target_distance_max}]m")
print(f"Distance scale factor: {self.randomizer.targets.cfg.distance_scale_factor}")

# Estimate worst case
max_spread = self.randomizer.initial_states.cfg.max_agent_separation * 2  # For 3 agents
min_target_dist = max_spread * self.randomizer.targets.cfg.distance_scale_factor

print(f"\n=== Estimated Behavior ===")
print(f"Max formation spread: ~{max_spread:.1f}m")
print(f"Min target distance: ~{min_target_dist:.1f}m")

if min_target_dist > self.randomizer.targets.cfg.target_distance_max:
    print(f"⚠️  WARNING: Targets will exceed configured max distance!")
    print(f"   Actual distance: ~{min_target_dist:.1f}m")
    print(f"   Configured max: {self.randomizer.targets.cfg.target_distance_max}m")
else:
    print(f"✓ Configuration compatible")
```

### Monitor Actual Distances

```python
# In iris_ma_env3.py _reset_idx(), after target sampling:

# Compute actual distances
agent_positions = formation_data[:, :, 0:3]
formation_centers = agent_positions.mean(dim=1)
target_distances = torch.norm(target_positions - formation_centers, dim=-1)

# Log statistics
if env_ids[0] == 0:  # Only log for env 0
    print(f"Target distances: min={target_distances.min():.1f}m, "
          f"max={target_distances.max():.1f}m, "
          f"mean={target_distances.mean():.1f}m")
```

---

## Advanced: Dynamic Configuration

### Curriculum-Based Scaling

```python
class IrisMAEnv(DirectMARLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Start with tight formations
        self.randomizer.initial_states.cfg.max_agent_separation = 3.0
        self.randomizer.targets.cfg.target_distance_max = 25.0

        self.training_progress = 0.0  # 0.0 to 1.0

    def update_curriculum(self, success_rate: float):
        """Gradually increase difficulty."""
        if success_rate > 0.85:
            self.training_progress = min(1.0, self.training_progress + 0.01)

            # Scale up formations and distances
            self.randomizer.initial_states.cfg.max_agent_separation = 3.0 + 9.0 * self.training_progress
            self.randomizer.targets.cfg.target_distance_max = 25.0 + 55.0 * self.training_progress
```

### Per-Environment Variation

```python
# Sample different formation sizes per environment
def _reset_idx(self, env_ids):
    # Randomly vary formation size per environment
    for env_id in env_ids:
        scale = 0.5 + 1.5 * torch.rand(1).item()  # 0.5 to 2.0

        # This affects only this specific reset
        formation_data = self.randomizer.initial_states.get_random_formation(
            num_agents=3,
            num_envs=1,
            scale_factor=scale
        )
```

---

## Common Scenarios

### Scenario 1: "I want targets always between 15-20m"

```python
# Ensure formation never exceeds ~6m spread
self.randomizer.initial_states.cfg.max_agent_separation = 3.0
self.randomizer.initial_states.cfg.z_variation_range = (0.0, 1.0)

# Set distance parameters
self.randomizer.targets.cfg.target_distance_min = 15.0
self.randomizer.targets.cfg.target_distance_max = 20.0
self.randomizer.targets.cfg.distance_scale_factor = 2.0  # Lower for tighter formations
```

### Scenario 2: "I want larger formations but targets still close"

```python
# Larger formations OK
self.randomizer.initial_states.cfg.max_agent_separation = 8.0

# But allow targets closer to formation
self.randomizer.targets.cfg.distance_scale_factor = 1.2  # Much lower!
self.randomizer.targets.cfg.target_distance_min = 15.0
self.randomizer.targets.cfg.target_distance_max = 30.0
```

⚠️ **Risk:** With large formations and low scale factor, targets may be inside formation

### Scenario 3: "I want maximum variety"

```python
# Wide formation range
self.randomizer.initial_states.cfg.max_agent_separation = 12.0

# Wide target range
self.randomizer.targets.cfg.target_distance_min = 20.0
self.randomizer.targets.cfg.target_distance_max = 80.0
self.randomizer.targets.cfg.distance_scale_factor = 2.5
```

---

## Validation

After changing configuration, run:

```bash
# Test formation generator
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/randomization/test_formation.py

# Test target sampler (will show warnings if config incompatible)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/randomization/test_target_sampling.py
```

Check for warnings like:
```
Warning: Formation spread (24.0m) × scale_factor (2.5) exceeds target_distance_max (20.0m).
Consider reducing formation_spacing or increasing target_distance_max.
```

---

## Summary

**Golden Rule:**
```
target_distance_max >= (max_agent_separation × num_agents) × distance_scale_factor
```

For 3 agents:
```
target_distance_max >= 2 × max_agent_separation × distance_scale_factor
```

**Example:**
- Want `target_distance_max = 20m`?
- With `distance_scale_factor = 2.0`?
- Then `max_agent_separation ≤ 20 / (2 × 2.0) = 5.0m`

---

## Quick Reference Table

| Desired Target Distance | Max Agent Separation | Distance Scale Factor |
|-------------------------|----------------------|----------------------|
| 15-20m | 3.0m | 2.0 |
| 20-30m | 5.0m | 2.0 |
| 25-40m | 6.0m | 2.5 |
| 30-50m | 8.0m | 2.5 |
| 40-80m | 12.0m | 2.5 |

All values assume 3 agents in worst-case configuration (line formation).
