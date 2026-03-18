# CBF Safety Module

## Purpose
Two-layer collision avoidance system:
1. **Training time**: Closest Point of Approach (CPA) penalty for reward shaping
2. **Deployment time**: Hard safety filter using Control Barrier Functions (CBF)

## Inputs
### Training
- Ground-truth positions: `(N, A, 3)`
- Commanded velocities: `(N, A, 3)`
- Simulation timestep `dt`

### Deployment
- Nominal (RL-commanded) velocities: `(N, A, 3)`
- Delayed positions and velocities from delay system

## Outputs
### Training
- Collision penalty per environment: `(N,)`
- Diagnostic metrics (min distances, violation counts)

### Deployment
- Safe velocities: `(N, A, 3)` satisfying CBF constraints

## Dependencies
None (standalone module).

## Key Files
- `cbf_manager.py` - Top-level manager selecting training vs deploy mode
- `cpa_reward_shaper.py` - CPA-based reward penalty computation
- `deploy_filter.py` - Hard CBF safety filter for deployment
- `cbf_diagnostics.py` - Logging and diagnostic utilities
- `cbf_cfg.py` - Configuration (safety radius, penalty gains, CBF parameters)

## Spec
`doc/safety_spec.md`
