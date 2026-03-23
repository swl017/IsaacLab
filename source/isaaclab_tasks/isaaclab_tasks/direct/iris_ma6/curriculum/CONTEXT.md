# Curriculum Module

## Purpose
Curriculum learning configuration for progressive training difficulty.
Defines how environment parameters (spawn range, target speed, delay magnitude,
randomization intensity) scale with training progress.

## Inputs
- Training progress metric (e.g., episode count, mean reward)

## Outputs
- Curriculum progress: `float` in [0.0, 1.0] consumed by other modules
- Parameter schedules for initial_states, target_controller, domain_randomization
- Agent velocity progress (decoupled from target motion): `get_agent_velocity_progress()`

## Dependencies
None (minimal module, orchestrated by the environment).

## Key Files
- `curriculum_cfg.py` - Configuration dataclass defining schedules and thresholds
- `doc/` - Additional documentation

## Spec
None (minimal configuration-only module).
