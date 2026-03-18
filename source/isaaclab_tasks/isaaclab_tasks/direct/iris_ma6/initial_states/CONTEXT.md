# Initial States Module

## Purpose
Curriculum-driven randomization of agent and target initial configurations
at episode reset. Generates spatially valid spawn positions that respect
inter-agent separation and formation constraints, scaling difficulty with
curriculum progress.

## Inputs
- Curriculum progress: `float` in [0.0, 1.0]
- Environment indices: `(M,)` tensor of envs to reset
- Number of agents, number of targets

## Outputs
- `InitialStatesResult` dataclass containing:
  - Agent positions: `(N, A, 3)`
  - Agent velocities: `(N, A, 3)`
  - Target position: `(N, 3)`
  - Gimbal angles: `(N, A, J)` initial joint angles

## Dependencies
None (standalone module).

## Key Files
- `initial_states.py` - Top-level interface
- `initial_states_generator.py` - Core sampling and validation logic
- `initial_states_cfg.py` - Configuration (spawn radius, altitude range, separation)
- `tests/` - Unit tests
- `doc/initial_states_spec.md` - Specification document

## Spec
`doc/initial_states_spec.md`
