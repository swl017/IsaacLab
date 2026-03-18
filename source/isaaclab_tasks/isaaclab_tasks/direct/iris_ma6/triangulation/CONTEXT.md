# Triangulation Module

## Purpose
Multi-camera multi-target triangulation with uncertainty quantification.
Fuses bearing measurements from multiple cameras to estimate 3D target
positions and their covariance, used for observation-quality rewards.

## Inputs
- Camera ray directions: `(N, C, T, 3)` unit vectors per camera per target
- Ray origins: `(N, C, T, 3)` camera positions
- Validity mask: `(N, C, T)` boolean indicating valid detections

## Outputs
- `TriangulationResult` dataclass containing:
  - Estimated position: `(N, T, 3)`
  - Covariance matrix: `(N, T, 3, 3)`
  - Is valid: `(N, T)` boolean (requires >= 2 valid views)

## Dependencies
None (standalone module).

## Key Files
- `triangulation.py` - Core least-squares triangulation and covariance computation
- `triangulation_cfg.py` - Configuration (min views, max condition number)
- `tests/` - Unit tests for triangulation accuracy and edge cases

## Spec
`doc/triangulation_spec.md`
