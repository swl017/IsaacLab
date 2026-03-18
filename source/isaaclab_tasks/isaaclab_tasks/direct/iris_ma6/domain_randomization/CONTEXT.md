# Domain Randomization Module

## Purpose
Sim-to-real domain randomization for camera, physics, and gimbal parameters.
Applies per-environment randomized perturbations at episode reset to improve
transfer robustness of trained policies.

## Inputs
- Environment indices: `(M,)` tensor of envs to randomize
- Curriculum progress (optional, for progressive randomization)

## Outputs
Cached randomized parameters per environment:
- Camera: focal length scale, noise level, latency offset
- Physics: mass scale, inertia perturbation, drag coefficients
- Gimbal: bias offsets, rate limits, backlash

## Dependencies
None (standalone module).

## Key Files
- `domain_randomizer.py` - Top-level randomizer orchestrating sub-randomizers
- `camera_processor.py` - Camera intrinsic/extrinsic randomization
- `physics_randomizer.py` - Mass, inertia, drag randomization
- `gimbal_randomizer.py` - Gimbal bias and limit randomization
- `domain_randomization_cfg.py` - Configuration (ranges, distributions)
- `tests/` - Unit tests
- `doc/domain_randomization_spec.md` - Specification document

## Spec
`doc/domain_randomization_spec.md`
