# Curriculum Module

## Purpose
Curriculum learning configuration for progressive training difficulty.
Defines how environment parameters (spawn range, target speed, delay magnitude,
randomization intensity) scale with training progress.

## Inputs
- Training step counter (`common_step_counter`)

## Outputs
- Per-phase progress values: `float` in [0.0, 1.0] consumed by env and other modules
- Delay mode string: `"none"` / `"fixed"` / `"random"`

## Dependencies
None (minimal module, orchestrated by the environment).

## Key Files
- `curriculum_cfg.py` — Phase definitions, step boundaries, progress getters
- `doc/curriculum_wiring.md` — Reference table mapping each phase to the env variable it controls and where it's applied

## Phases (13 total, in chronological order)
Agent Velocity → Safety → Tracking → Target Motion → Coordination →
Task Levels → Noise → FP/FN → Fixed Delay → Random Delay → Dropout →
Dynamics → Burst Dropout

## Spec
None (configuration-only module).
