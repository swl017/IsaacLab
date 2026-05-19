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
- `progress_helper.py` — Per-(env, agent) effective-progress sampler. Maps a
  scalar global `p ∈ [0, 1]` to a `Tensor[E, A]` of values drawn i.i.d. from
  `Uniform(0, p)`. Enforces the anti-forgetting invariant (ticket 034): the
  per-env distribution always contains positive mass on the easy regime,
  including at `p = 1`. Stateless; consumed by env `_reset_idx` once per axis.
- `doc/curriculum_wiring.md` — Reference table mapping each phase to the env variable it controls and where it's applied

## Phases (13 total, in chronological order)
Agent Velocity → Safety → Tracking → Target Motion → Coordination →
Task Levels → Noise → FP/FN → Fixed Delay → Random Delay → Dropout →
Dynamics → Burst Dropout

## Spec
None (configuration-only module).
