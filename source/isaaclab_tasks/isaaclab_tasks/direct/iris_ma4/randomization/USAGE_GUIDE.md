# Randomizer Module - Usage Guide

Practical guide for integrating gimbal-aware formation and target sampling into your multi-agent environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Integration with Environment](#integration-with-environment)
- [Curriculum Learning](#curriculum-learning)
- [Configuration and Tuning](#configuration-and-tuning)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Quick Start

### Installation

The randomizer module is part of the iris_ma3 task package. No additional installation required.

```python
from isaaclab_tasks.direct.iris_ma3.randomization import Randomizer
import torch
import math

# Create randomizer
randomizer = Randomizer(
    num_envs=100,
    device=torch.device("cuda"),
    gimbal_pitch_limits=(math.radians(-45), math.radians(10))
)

# Generate formation
formation = randomizer.initial_states.get_random_formation(num_agents=3)

# Sample feasible targets
targets = randomizer.targets.sample_target_position(formation)
```

**Output:**
- `formation`: [100, 3, 13] tensor with positions, orientations, velocities
- `targets`: [100, 3] tensor with target positions

**Guarantee:** All gimbal angles will be within [-45°, +10°] limits (with 5° safety margin).

---

## Basic Usage

### 1. Creating the Randomizer

```python
import torch
import math
from isaaclab_tasks.direct.iris_ma3.randomization import Randomizer

# Match your environment configuration
randomizer = Randomizer(
    num_envs=self.num_envs,
    device=self.device,
    gimbal_pitch_limits=self.cfg.max_gimbal_pitch_angle  # IMPORTANT: Must match!
)
```

**Critical:** The `gimbal_pitch_limits` must exactly match your environment's gimbal configuration!

### 2. Generating Formations

```python
# Random formation type (50% line, 50% grid)
formation = randomizer.initial_states.get_random_formation(num_agents=3)

# Specific formation type
line_formation = randomizer.initial_states.get_random_formation(
    num_agents=3,
    formation_type="line"
)

grid_formation = randomizer.initial_states.get_random_formation(
    num_agents=4,  # Must be perfect square or near-square for best results
    formation_type="grid"
)
```

**Formation Data Structure:**
```python
formation.shape  # [num_envs, num_agents, 13]

positions = formation[:, :, 0:3]      # [num_envs, num_agents, 3]
orientations = formation[:, :, 3:7]   # [num_envs, num_agents, 4] (quaternions)
linear_vel = formation[:, :, 7:10]    # [num_envs, num_agents, 3]
angular_vel = formation[:, :, 10:13]  # [num_envs, num_agents, 3]
```

### 3. Sampling Targets

```python
# Sample targets for formation
targets = randomizer.targets.sample_target_position(formation)

# Or from positions only
positions = formation[:, :, 0:3]
targets = randomizer.targets.sample_target_position(positions)
```

### 4. Validating Feasibility (Optional)

```python
validation = randomizer.targets.validate_target_feasibility(
    agent_positions=formation[:, :, 0:3],
    agent_orientations=formation[:, :, 3:7],
    target_positions=targets
)

# Should always be 100% when using the constrained sampler
assert validation['all_feasible'].all(), "Unexpected infeasibility!"
print(f"Feasibility: {validation['all_feasible'].float().mean() * 100:.1f}%")
```

---

## Integration with Environment

### Complete _reset_idx() Implementation

```python
def _reset_idx(self, env_ids: torch.Tensor):
    """Reset environments at given indices."""

    # Step 1: Generate formation (positions + orientations + velocities)
    formation_data = self.randomizer.initial_states.get_random_formation(
        num_agents=len(self.cfg.possible_agents),
        env_ids=env_ids,
        scale_factor=1.0,  # Future: get from curriculum
    )
    # formation_data shape: [len(env_ids), num_agents, 13]

    # Step 2: Sample feasible target positions
    target_positions = self.randomizer.targets.sample_target_position(
        formation_data=formation_data,
        scale_factor=1.0,  # Future: get from curriculum
    )
    # target_positions shape: [len(env_ids), 3]

    # Step 3: Set target states
    target_state = self.target.data.default_root_state[env_ids].clone()
    target_state[:, 0:3] = target_positions
    target_state[:, 3:7] = quat_from_euler_xyz(
        torch.zeros(len(env_ids), device=self.device),
        torch.zeros(len(env_ids), device=self.device),
        torch.rand(len(env_ids), device=self.device) * 2 * math.pi  # Random yaw
    )
    self.target.write_root_pose_to_sim(target_state[:, :7], env_ids)
    self.target.write_root_velocity_to_sim(target_state[:, 7:], env_ids)

    # Step 4: Set agent states and compute gimbal angles
    for i, agent_id in enumerate(self.cfg.possible_agents):
        robot = self._robots[agent_id]
        agent_state = formation_data[:, i, :]  # [len(env_ids), 13]

        # Set root state
        robot.write_root_pose_to_sim(agent_state[:, 0:7], env_ids)
        robot.write_root_velocity_to_sim(agent_state[:, 7:13], env_ids)

        # Compute gimbal angles (now guaranteed feasible!)
        gimbal_yaw, gimbal_roll, gimbal_pitch = \
            self._gimbal_stabilizers[agent_id].compute_stabilized_angles_from_target_point(
                target_point_world=target_positions,
                drone_position_world=agent_state[:, 0:3],
                drone_quat_world=agent_state[:, 3:7]
            )

        # Set gimbal joint positions
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]] = gimbal_yaw
        joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]] = gimbal_pitch
        joint_pos[:, self.gimbal_joint_idx[agent_id]["roll"]] = gimbal_roll

        robot.write_joint_state_to_sim(
            position=joint_pos,
            velocity=robot.data.default_joint_vel[env_ids],
            joint_ids=None,
            env_ids=env_ids
        )

    # Call parent reset
    super()._reset_idx(env_ids)
```

### Environment __init__() Setup

```python
def __init__(self, cfg: IrisMAEnvCfg, render_mode: str | None = None, **kwargs):
    # ... existing initialization ...

    # Create randomizer with gimbal constraints
    self.randomizer = Randomizer(
        self.num_envs,
        self.device,
        gimbal_pitch_limits=self.cfg.max_gimbal_pitch_angle
    )

    # ... rest of initialization ...
```

---

## Curriculum Learning

### Basic Curriculum Integration

```python
class FormationCurriculum:
    """Simple curriculum that scales difficulty with training progress."""

    def __init__(self):
        self.progress = 0.0  # 0.0 to 1.0

    def update(self, success_rate: float):
        """Update curriculum based on performance."""
        if success_rate > 0.8:  # If doing well, increase difficulty
            self.progress = min(1.0, self.progress + 0.01)
        elif success_rate < 0.5:  # If struggling, decrease difficulty
            self.progress = max(0.0, self.progress - 0.01)

    def get_scale_factor(self) -> float:
        """Get scale factor for randomization."""
        # Start easy (0.5), end hard (2.0)
        return 0.5 + 1.5 * self.progress
```

### Using Curriculum in Environment

```python
def __init__(self, cfg, **kwargs):
    super().__init__(cfg, **kwargs)
    self.curriculum = FormationCurriculum()

def _reset_idx(self, env_ids: torch.Tensor):
    # Get curriculum scale
    scale = self.curriculum.get_scale_factor()

    # Generate scaled formation
    formation = self.randomizer.initial_states.get_random_formation(
        num_agents=len(self.cfg.possible_agents),
        scale_factor=scale,  # Affects formation size and velocities
        env_ids=env_ids
    )

    # Sample scaled targets
    targets = self.randomizer.targets.sample_target_position(
        formation,
        scale_factor=scale  # Affects target distance
    )

    # ... rest of reset logic ...

def post_physics_step(self):
    # ... existing logic ...

    # Update curriculum periodically
    if self.common_step_counter % 1000 == 0:
        success_rate = self.compute_success_rate()
        self.curriculum.update(success_rate)
```

### Advanced Multi-Stage Curriculum

```python
class MultiStageCurriculum:
    """Curriculum with distinct stages."""

    STAGES = [
        {"name": "easy", "scale": 0.5, "min_success": 0.85},
        {"name": "medium", "scale": 1.0, "min_success": 0.80},
        {"name": "hard", "scale": 1.5, "min_success": 0.75},
        {"name": "expert", "scale": 2.0, "min_success": 0.70},
    ]

    def __init__(self):
        self.current_stage = 0
        self.success_history = []

    def update(self, success_rate: float):
        self.success_history.append(success_rate)

        # Check if ready for next stage (based on recent performance)
        if len(self.success_history) >= 100:
            recent_success = sum(self.success_history[-100:]) / 100
            min_required = self.STAGES[self.current_stage]["min_success"]

            if recent_success >= min_required and self.current_stage < len(self.STAGES) - 1:
                self.current_stage += 1
                print(f"Advancing to stage: {self.STAGES[self.current_stage]['name']}")

    def get_scale_factor(self) -> float:
        return self.STAGES[self.current_stage]["scale"]
```

---

## Configuration and Tuning

### Adjusting Target Distance

```python
# Make targets closer or farther
self.randomizer.targets.cfg.target_distance_min = 20.0  # meters
self.randomizer.targets.cfg.target_distance_max = 100.0  # meters

# Ensure target distance scales with formation size
self.randomizer.targets.cfg.distance_scale_factor = 3.0  # 3× formation spread
```

### Adjusting Formation Parameters

```python
# Modify formation randomizer config
self.randomizer.initial_states.cfg.formation_spacing_min = 5.0  # meters
self.randomizer.initial_states.cfg.formation_spacing_max = 10.0  # meters
self.randomizer.initial_states.cfg.min_agent_separation = 4.0  # meters

# Altitude range
self.randomizer.initial_states.cfg.ground_clearance_min = 10.0  # meters
self.randomizer.initial_states.cfg.ground_clearance_max = 30.0  # meters

# Height variation within formation
self.randomizer.initial_states.cfg.z_variation_range = (-5.0, 5.0)  # meters
```

### Adjusting Safety Margins

```python
# Tighter constraints (less conservative)
self.randomizer.targets.cfg.pitch_safety_margin = math.radians(3.0)  # 3° margin

# Looser constraints (more conservative)
self.randomizer.targets.cfg.pitch_safety_margin = math.radians(10.0)  # 10° margin
```

---

## Common Patterns

### Pattern 1: Resetting Specific Environments

```python
def _reset_idx(self, env_ids: torch.Tensor):
    """Reset only environments in env_ids."""

    # Generate formation for specific environments
    formation = self.randomizer.initial_states.get_random_formation(
        num_agents=3,
        num_envs=len(env_ids),  # Override number of environments
        env_ids=env_ids  # Pass environment indices
    )

    targets = self.randomizer.targets.sample_target_position(
        formation,
        num_envs=len(env_ids)
    )

    # Apply states only to env_ids
    # ... (use env_ids as indices)
```

### Pattern 2: Fixed Formation Type per Environment

```python
def __init__(self, cfg, **kwargs):
    super().__init__(cfg, **kwargs)

    # Assign formation types
    self.formation_types = ["line"] * (self.num_envs // 2) + \
                          ["grid"] * (self.num_envs - self.num_envs // 2)

def _reset_idx(self, env_ids: torch.Tensor):
    # Reset each formation type separately
    for formation_type in ["line", "grid"]:
        mask = torch.tensor([self.formation_types[i] == formation_type
                            for i in env_ids.cpu().numpy()], device=self.device)
        type_env_ids = env_ids[mask]

        if len(type_env_ids) > 0:
            formation = self.randomizer.initial_states.get_random_formation(
                num_agents=3,
                num_envs=len(type_env_ids),
                formation_type=formation_type
            )
            # ... process formation ...
```

### Pattern 3: Caching Randomizations

```python
def __init__(self, cfg, **kwargs):
    super().__init__(cfg, **kwargs)

    # Pre-generate randomizations (useful for evaluation)
    self.cached_formations = []
    self.cached_targets = []

    for _ in range(1000):  # Generate 1000 scenarios
        formation = self.randomizer.initial_states.get_random_formation(
            num_agents=3,
            num_envs=1
        )
        target = self.randomizer.targets.sample_target_position(formation)
        self.cached_formations.append(formation)
        self.cached_targets.append(target)

    self.cache_idx = 0

def _reset_idx(self, env_ids: torch.Tensor):
    # Use cached randomizations
    for i, env_id in enumerate(env_ids):
        formation = self.cached_formations[self.cache_idx % len(self.cached_formations)]
        target = self.cached_targets[self.cache_idx % len(self.cached_targets)]
        self.cache_idx += 1

        # Apply to environment...
```

### Pattern 4: Domain Randomization Integration

```python
def _reset_idx(self, env_ids: torch.Tensor):
    # Generate formation and targets
    formation = self.randomizer.initial_states.get_random_formation(...)
    targets = self.randomizer.targets.sample_target_position(formation)

    # Add domain randomization
    # Randomize mass
    for agent_id in self.cfg.possible_agents:
        robot = self._robots[agent_id]
        mass_scale = 0.8 + 0.4 * torch.rand(len(env_ids), device=self.device)
        # Apply mass randomization...

    # Randomize wind
    wind_velocity = torch.randn(len(env_ids), 3, device=self.device) * 2.0
    # Apply wind...

    # Set formation and targets
    # ...
```

---

## Troubleshooting

### Issue: Many Fallback Warnings

```
Warning: X environments have infeasible height ranges. Using fallback.
```

**Causes:**
1. Formation vertical spread too large for target distance
2. Gimbal pitch limits too restrictive
3. Target distance too close

**Solutions:**
```python
# Increase minimum target distance
self.randomizer.targets.cfg.target_distance_min = 30.0

# Reduce formation height variation
self.randomizer.initial_states.cfg.z_variation_range = (-1.0, 1.0)

# Increase distance scaling
self.randomizer.targets.cfg.distance_scale_factor = 3.5
```

### Issue: Targets Too Close/Far

**Solution:**
```python
# Adjust distance range
self.randomizer.targets.cfg.target_distance_min = 25.0
self.randomizer.targets.cfg.target_distance_max = 75.0

# Or check formation spread
formation = self.randomizer.initial_states.get_random_formation(3)
positions = formation[:, :, 0:3]
spread = positions.std(dim=1).norm(dim=-1).mean()
print(f"Average formation spread: {spread:.2f}m")
```

### Issue: Gimbal Angles Still Out of Bounds

**Cause:** Mismatch between `gimbal_pitch_limits` and environment config

**Solution:**
```python
# ALWAYS use environment config
self.randomizer = Randomizer(
    self.num_envs,
    self.device,
    gimbal_pitch_limits=self.cfg.max_gimbal_pitch_angle  # MUST MATCH!
)

# Verify limits match
print(f"Env gimbal limits: {self.cfg.max_gimbal_pitch_angle}")
print(f"Randomizer limits: ({self.randomizer.targets.cfg.pitch_limit_min}, "
      f"{self.randomizer.targets.cfg.pitch_limit_max})")
```

### Issue: Agents Spawning Outside Boundaries

**Solution:**
```python
# Adjust spawn area
self.randomizer.initial_states.cfg.spawn_area_x_range = (-50.0, 50.0)
self.randomizer.initial_states.cfg.spawn_area_y_range = (-50.0, 50.0)

# Adjust ground clearance
self.randomizer.initial_states.cfg.ground_clearance_min = 5.0
self.randomizer.initial_states.cfg.ground_clearance_max = 40.0
```

### Issue: Agents Colliding at Spawn

**Solution:**
```python
# Increase minimum separation
self.randomizer.initial_states.cfg.min_agent_separation = 5.0

# Increase formation spacing
self.randomizer.initial_states.cfg.formation_spacing_min = 5.0
self.randomizer.initial_states.cfg.formation_spacing_max = 12.0
```

---

## Best Practices

### 1. Always Match Gimbal Limits

```python
# ✓ CORRECT
self.randomizer = Randomizer(
    self.num_envs,
    self.device,
    gimbal_pitch_limits=self.cfg.max_gimbal_pitch_angle
)

# ✗ WRONG - Hardcoded limits may not match environment
self.randomizer = Randomizer(
    self.num_envs,
    self.device,
    gimbal_pitch_limits=(math.radians(-45), math.radians(10))
)
```

### 2. Use Validation During Development

```python
def _reset_idx(self, env_ids: torch.Tensor):
    formation = self.randomizer.initial_states.get_random_formation(...)
    targets = self.randomizer.targets.sample_target_position(formation)

    # Validate during development/testing
    if self.cfg.debug_mode:
        validation = self.randomizer.targets.validate_target_feasibility(
            formation[:, :, 0:3],
            formation[:, :, 3:7],
            targets
        )
        assert validation['all_feasible'].all(), \
            f"Found {validation['violations'].sum()} violations!"
```

### 3. Scale Gradually for Curriculum

```python
# ✓ CORRECT - Gradual scaling
scale = 0.5 + 1.5 * curriculum_progress  # 0.5 → 2.0

# ✗ WRONG - Abrupt changes
scale = 2.0 if curriculum_progress > 0.5 else 0.5
```

### 4. Log Randomization Statistics

```python
def _reset_idx(self, env_ids: torch.Tensor):
    formation = self.randomizer.initial_states.get_random_formation(...)
    targets = self.randomizer.targets.sample_target_position(formation)

    # Log statistics for analysis
    if self.common_step_counter % 1000 == 0:
        positions = formation[:, :, 0:3]
        spread = positions.std(dim=1).norm(dim=-1).mean()
        target_dist = torch.norm(targets - positions.mean(dim=1), dim=-1).mean()

        self.logger.log({
            "randomization/formation_spread": spread,
            "randomization/target_distance": target_dist,
            "randomization/scale_factor": self.curriculum.get_scale_factor()
        })
```

### 5. Test with Edge Cases

```python
# Test with different agent counts
for num_agents in [1, 2, 3, 5, 10]:
    formation = randomizer.initial_states.get_random_formation(num_agents)
    assert formation.shape == (num_envs, num_agents, 13)

# Test with different formation types
for formation_type in ["line", "grid"]:
    formation = randomizer.initial_states.get_random_formation(
        num_agents=4,
        formation_type=formation_type
    )
    # Verify feasibility...

# Test with extreme scale factors
for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
    formation = randomizer.initial_states.get_random_formation(
        num_agents=3,
        scale_factor=scale
    )
    targets = randomizer.targets.sample_target_position(formation, scale_factor=scale)
    # Verify all constraints met...
```

---

## Performance Tips

### 1. Minimize Randomizer Calls

```python
# ✓ EFFICIENT - Generate once per reset
def _reset_idx(self, env_ids: torch.Tensor):
    formation = self.randomizer.initial_states.get_random_formation(...)
    targets = self.randomizer.targets.sample_target_position(formation)
    # Use formation and targets...

# ✗ INEFFICIENT - Multiple calls
def _reset_idx(self, env_ids: torch.Tensor):
    for env_id in env_ids:
        formation = self.randomizer.initial_states.get_random_formation(
            num_agents=3,
            num_envs=1
        )  # Slow! One at a time
```

### 2. Batch Environment Resets

```python
# ✓ EFFICIENT - Batch processing
def _reset_idx(self, env_ids: torch.Tensor):
    formation = self.randomizer.initial_states.get_random_formation(
        num_agents=3,
        num_envs=len(env_ids)  # Process all at once
    )
```

### 3. Avoid Validation in Production

```python
# Validation is useful for debugging but adds overhead
# Remove in production training
if self.cfg.validate_randomization:  # Debug flag
    validation = self.randomizer.targets.validate_target_feasibility(...)
```

---

## Testing Your Integration

Run the provided test suites to verify correct integration:

```bash
# Test formation generator
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/randomization/test_formation.py

# Test target sampler
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/randomization/test_target_sampling.py
```

Expected output: **ALL TESTS PASSED! ✓**

---

## Next Steps

- Read [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for step-by-step environment integration
- Check [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation
- See example integration in `iris_ma_env3.py`
- Implement curriculum learning for progressive training difficulty

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review test files for usage examples
3. Verify gimbal limits match environment configuration
4. Check logs for fallback warnings
