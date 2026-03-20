# Unified Randomization & Curriculum Architecture for iris_ma6

**Status**: Backlog — planned for next iris_ma iteration
**Created**: 2026-03-20

## Context

Randomization and curriculum scaling are currently scattered across 6+ config classes (`DomainRandomizationCfg`, `DelaySystemKeyParams`, `InitialStatesCfg`, `GainRandomizationCfg`, `TargetControllerCfg`, `CurriculumCfg`) with inconsistent distribution formats (`(min, max)` tuples, `mean/std`, `DistributionCfg` dataclass) and ad-hoc curriculum coupling (manual `get_progress()` calls bridged by env code). This makes it hard to audit, experiment with, or add new parameters.

**Goal**: A single declarative config system where every randomizable parameter specifies its distribution, sampling frequency, and curriculum schedule in one place.

---

## 1. Core Data Classes

### 1.1 `DistributionSpec` — Unified distribution descriptor

Replaces all existing distribution formats. Intentionally flat (no nested dicts) for `@configclass` serialization and Hydra override support.

```python
@configclass
class DistributionSpec:
    type: Literal["uniform", "normal", "log_uniform", "constant", "discrete"] = "uniform"
    # uniform / log_uniform
    low: float = 0.0
    high: float = 1.0
    # normal
    mean: float = 0.0
    std: float = 1.0
    # constant
    value: float = 0.0
    # discrete
    choices: list[float] = []
    # clipping
    min_clip: float | None = None
    max_clip: float | None = None
```

### 1.2 `CurriculumSchedule` — When/how curriculum modulates a parameter

Two modulation modes:
- **`range`**: Distribution shrinks toward `center` at progress=0, expands to full range at progress=1. Anti-catastrophic-forgetting pattern from `InitialStatesCfg`.
- **`magnitude`**: All distribution params scale linearly with progress. At progress=0 → effective 0. Perfect for noise/delay ramp-up.

```python
@configclass
class CurriculumSchedule:
    mode: Literal["range", "magnitude", "none"] = "none"
    start_step: int = 0
    end_step: int = 200000
    center: float | None = None  # For "range" mode; defaults to midpoint
```

### 1.3 `RandomizableParam` — Atomic unit

```python
@configclass
class RandomizableParam:
    name: str = ""                    # dot-path key, e.g. "physics.body_mass_scale"
    enabled: bool = True
    distribution: DistributionSpec = DistributionSpec()
    frequency: Literal["per_step", "per_episode", "per_env_reset", "per_agent", "prestartup"] = "per_episode"
    curriculum: CurriculumSchedule = CurriculumSchedule()
    tensor_shape_suffix: tuple[int, ...] = ()  # extra dims beyond (num_envs,)
```

### 1.4 Category Configs → `UnifiedRandomizationCfg`

9 category dataclasses, each containing `RandomizableParam` fields:

```python
@configclass
class UnifiedRandomizationCfg:
    enabled: bool = True
    physics: PhysicsParamsCfg       # 8 params
    controller: ControllerParamsCfg # 8 params
    initial_states: InitialStateParamsCfg  # 7 params
    target_motion: TargetMotionParamsCfg   # 5 params
    delay: DelayParamsCfg           # 5 params
    noise: NoiseParamsCfg           # 4 params
    camera: CameraParamsCfg         # 2 params
    gimbal: GimbalParamsCfg         # 4 params
    rewards: RewardParamsCfg        # 3 params
```

---

## 2. Complete Parameter Registry (~46 parameters)

| Category | Parameter | Distribution | Frequency | Curriculum | Mode |
|----------|-----------|-------------|-----------|------------|------|
| **Physics** | body_mass_scale | U(0.9, 1.1) | per_episode | 160k-200k | range(1.0) |
| | body_mass_additive | U(-0.05, 0.05) | per_episode | 160k-200k | range(0.0) |
| | payload_mass | U(0.0, 0.2) | per_episode | 160k-200k | range(0.0) |
| | gimbal_mass_scale | U(0.95, 1.05) | per_episode | 160k-200k | range(1.0) |
| | static_friction | U(0.7, 1.3) | per_episode | 160k-200k | range(1.0) |
| | dynamic_friction | U(0.5, 1.0) | per_episode | 160k-200k | range(0.75) |
| | restitution | U(0.0, 0.3) | per_episode | 160k-200k | range(0.0) |
| | body_scale | U(0.95, 1.05) | prestartup | none | — |
| **Controller** | velocity_kp_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| | velocity_ki_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| | attitude_kp_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| | rate_kp_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| | rate_ki_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| | rate_kd_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| | motor_tau_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| | max_lin_vel_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1.0) |
| **Initial States** | cylinder_diameter | U(20, 100) | per_episode | 0-60k | range(20) |
| | cylinder_height | U(10, 50) | per_episode | 0-60k | range(10) |
| | vertical_spread | U(2, 10) | per_episode | 0-60k | range(2) |
| | target_distance | U(10, 40) | per_episode | 0-60k | range(10) |
| | agent_velocity_scale | U(0, 1) | per_episode | 0-60k | magnitude |
| | orientation_noise | N(0, 0.2) | per_episode | none | — |
| | zoom_initial_max | U(1, 10) | per_episode | 0-60k | range(1) |
| **Target** | max_speed | U(1, 5) | per_episode | 20k-80k | range(1) |
| | max_acceleration | C(5.0) | per_episode | none | — |
| | update_interval | U(2, 6) | per_episode | 20k-80k | range(6) |
| | circular_radius | U(15, 60) | per_episode | none | — |
| | linear_weight | C(0.5) | per_episode | none | — |
| **Delay** | ego_motion_fol_tau | C(0.005) | per_episode | none | — |
| | ego_detection_latency | N(0.1, 0.015) | per_episode | 100k-130k | magnitude |
| | other_agent_latency | N(0.5, 0.08) | per_episode | 100k-130k | magnitude |
| | staleness_fps | U(20, 30) | per_episode | 130k-160k | magnitude |
| | dropout_prob | C(0.05) | per_step | 160k-200k | magnitude |
| **Noise** | position_std | C(0.1) | per_step | 80k-100k | magnitude |
| | velocity_std | C(0.05) | per_step | 80k-100k | magnitude |
| | orientation_std | C(0.01) | per_step | 80k-100k | magnitude |
| | bbox_std | C(7.0) | per_step | 80k-100k | magnitude |
| **Camera** | fov_scale | U(0.5, 1.0) | per_episode | none | — |
| | focal_length | U(800, 1200) | per_episode | none | — |
| **Gimbal** | yaw_offset | U(-0.1, 0.1) | per_episode | 160k-200k | range(0) |
| | pitch_offset | U(-0.05, 0.05) | per_episode | 160k-200k | range(0) |
| | stiffness_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1) |
| | damping_scale | U(0.8, 1.2) | per_episode | 160k-200k | range(1) |
| **Rewards** | safety_scale | C(1.0) | per_step | 0-20k | magnitude |
| | tracking_scale | C(1.0) | per_step | 0-60k | magnitude |
| | coordination_scale | C(1.0) | per_step | 60k-100k | magnitude |

*U=uniform, N=normal, C=constant*

---

## 3. Sampling Engine (`UnifiedRandomizer`)

```python
class UnifiedRandomizer:
    def __init__(cfg: UnifiedRandomizationCfg, num_envs, num_agents, device): ...

    # Called once per step from env
    def set_step(step: int):
        """Update training step; auto-resamples all per_step params."""

    # Called from _reset_idx()
    def sample_on_reset(env_ids: Tensor):
        """Resample all per_episode/per_env_reset/per_agent params for given envs."""

    # Consumers pull values
    def get_scalar(name: str, env_ids: Tensor | None) -> Tensor:
        """Get sampled values. Shape: (num_envs,) or (len(env_ids),)."""

    def get_progress(name: str) -> float:
        """Get current curriculum progress for a parameter."""

    def get_all_params_summary() -> dict:
        """For logging/debugging."""
```

**Curriculum modulation logic** (inside `_do_sample`):
- Compute `progress = clamp((step - start) / (end - start), 0, 1)`
- **range mode**: `effective_low = center + p * (low - center)`, `effective_high = center + p * (high - center)`
- **magnitude mode**: `effective_value = p * value` (or `p * mean`, `p * std`)
- Sample from the effective distribution on GPU

---

## 4. Environment Integration

### 4.1 Reset flow (`_reset_idx`)
```python
self._randomizer.sample_on_reset(env_ids)  # One call resamples everything

# Consumers pull what they need:
self._initial_states.generate(
    cylinder_diameter=self._randomizer.get_scalar("initial_states.cylinder_diameter", env_ids),
    target_distance=self._randomizer.get_scalar("initial_states.target_distance", env_ids),
    ...
)
self._controllers[agent].apply_gain_scales(
    vel_kp_scale=self._randomizer.get_scalar("controller.velocity_kp_scale", env_ids),
    ...
)
```

### 4.2 Step flow (`_get_rewards` / `_get_observations`)
```python
self._randomizer.set_step(self.common_step_counter)  # Resamples per_step params

noise_pos = self._randomizer.get_scalar("noise.position_std")  # Already curriculum-scaled
safety_w = self._randomizer.get_scalar("reward.safety_scale")
```

### 4.3 Relationship to existing modules

The unified randomizer **does NOT replace** existing modules. It replaces how they get their parameter values:
- `CurriculumCfg` → simplified to phase boundary constants only (no `get_progress()` methods needed)
- `GainRandomizationCfg` → absorbed into `ControllerParamsCfg`
- `DomainRandomizationCfg` → absorbed into `PhysicsParamsCfg`, `CameraParamsCfg`, `GimbalParamsCfg`
- `InitialStatesGenerator`, `DroneController.randomize_gains()`, `MultiAgentDelaySystemV3` remain as **appliers** — they receive values from the randomizer instead of computing their own

---

## 5. Implementation Steps

### Phase 1: Core framework (new files)
1. Create `iris_ma6/unified_randomization/` module
2. `distribution_spec.py` — `DistributionSpec`, `CurriculumSchedule`, `RandomizableParam`
3. `param_registry.py` — all `*ParamsCfg` categories + `UnifiedRandomizationCfg`
4. `unified_randomizer.py` — `UnifiedRandomizer` sampling engine
5. `CONTEXT.md` for module
6. Tests for core sampling and curriculum modulation

### Phase 2: Integration (modify existing files)
1. Add `randomization: UnifiedRandomizationCfg` to `IrisMA6TestEnvCfg`
2. Instantiate `UnifiedRandomizer` in env `__init__`
3. Replace `_linear_progress()` / `curriculum.get_progress()` calls with `_randomizer.get_scalar()`
4. Refactor `InitialStatesGenerator.generate()` to accept explicit per-parameter values
5. Wire controller gain randomization through unified randomizer
6. Wire delay/noise curriculum through unified randomizer

### Phase 3: Cleanup
1. Deprecate `GainRandomizationCfg`, `DomainRandomizationCfg`
2. Simplify `CurriculumCfg` to phase boundary constants only

---

## 6. Key Design Decisions

1. **Delay mode transitions** (none/fixed/random): The randomizer exposes a computed `delay_mode` property based on which delay parameters have non-zero progress, preserving the existing mode semantics.

2. **InitialStatesGenerator refactor**: Currently takes a single `curriculum_progress` float. Needs refactoring to accept individual parameter values (cylinder_diameter, target_distance, etc.) since they may have different curriculum phases.

3. **Hydra compatibility**: Flat `@configclass` fields enable overrides like `randomization.physics.body_mass_scale.distribution.low=0.85`.

4. **No changes to curriculum timeline**: The well-established phase boundaries (0k, 20k, 60k, 80k, 100k, 130k, 160k, 200k) are preserved — each `RandomizableParam` simply references the same step values.

---

## 7. Critical Files

| File | Role |
|------|------|
| `iris_ma6/iris_ma_env6_test.py` | Main env — all curriculum/randomization calls live here |
| `iris_ma6/iris_ma_env6_test_cfg.py` | Env config — add `UnifiedRandomizationCfg` |
| `iris_ma6/curriculum/curriculum_cfg.py` | Current phase definitions — simplify |
| `iris_ma6/domain_randomization/domain_randomization_cfg.py` | Current scattered config — absorb |
| `iris_ma6/delay_system_v3/delay_cfg_v3.py` | Has own `DistributionCfg` — new system supersedes |
| `iris_ma6/initial_states/initial_states_cfg.py` | Per-param min/max — refactor to accept values |
| `iris_ma6/controller/gain_randomization_cfg.py` | Gain scales — absorb |

---

## 8. Verification

1. **Unit tests**: Sample each distribution type, verify curriculum modulation (range + magnitude modes), verify per_step vs per_episode sampling frequency
2. **Integration test**: Run env for 100 steps, verify that `get_all_params_summary()` returns sensible values at different training steps (0, 100k, 200k)
3. **Regression**: Ensure that with default `UnifiedRandomizationCfg` values matching current hardcoded values, training behavior is unchanged
4. **Hydra override**: Verify that `randomization.noise.position_std.distribution.value=0.2` correctly overrides the default