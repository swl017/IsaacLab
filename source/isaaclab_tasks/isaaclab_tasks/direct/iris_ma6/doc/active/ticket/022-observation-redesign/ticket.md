## Ticket: Observation space redesign — heading frame + geometry features

**What**: Redesign the actor observation space to use the heading frame (vehicle-1 frame, yaw-only rotation from world) for ego-centric spatial reasoning, add pre-computed triangulation geometry features, and clean up frame inconsistencies. The critic observation remains in world frame.

**Why**: Analysis of the Aerial_To_Aerial_Interception codebase (TU Delft PATS-X) showed that ego-centric observations improved learning speed by 47% over world-frame equivalents. The current iris_ma6 observation has several issues:
1. **World-frame inter-agent state** forces the policy to learn coordinate subtraction and rotation — pure linear algebra that should be pre-computed
2. **Euler angle yaw** has a wrapping discontinuity at ±π that corrupts RNN hidden state
3. **No triangulation geometry features** — the policy must discover convergence angle and baseline quality from raw vectors, yet these are exactly what the FIM reward optimizes
4. **Missing gimbal roll joint** — the auto-stabilizing roll can saturate under aggressive banking, but the policy has no visibility into this
5. **Gimbal yaw joint** is redundant with the camera ray direction once the ray is in heading frame

**Spec**: [doc/observation_redesign_spec.md](../observation_redesign_spec.md) — full specification including frame definitions, per-feature rationale, dimension tables, computation pseudocode, and migration plan.

**Summary of changes**:

| Block | Current | Proposed | Change |
|-------|---------|----------|--------|
| Ego | 31D (world pos, world vel, Euler [φ,θ,ψ], body ω, body a, gimbal yaw+pitch, world ray, world cam_ω, aoi, zoom, hfov, bbox, empty) | 29D (heading vel, [φ,θ], [cos ψ, sin ψ], body ω, body a, heading ray, gimbal pitch+roll joints, world cam_ω, aoi, zoom, hfov, bbox, empty) | -2D |
| Inter-agent (per other) | 16D (world pos, world vel, world ray, world cam_ω, zoom, empty, ages×2) | 19D (heading Δpos, heading Δvel, heading ray, world cam_ω, convergence angle, baseline mag, baseline-ray angle, zoom, empty, ages×2) | +3D |
| Tri tail (actor) | 6D (world pos, world std) | 4D (heading Δpos, scalar uncertainty) | -2D |
| **Total (2 agents)** | **53D** | **52D** | **-1D** |

**Scope boundary**:
- DO change: `_get_observations()` in `iris_ma_env6_test.py` (both delay and GT paths)
- DO change: observation docstring and dimension constants
- DO change: SKRL config observation dimensions (`skrl_mappo_rnn_cfg.yaml`)
- DO add: `rotate_to_heading_frame()` utility function
- DO add: geometry feature computation (convergence angle, baseline magnitude, baseline-ray angle)
- DO add: gimbal roll joint to ego observation
- DO NOT change: reward structure, curriculum, network architecture, action space
- DO NOT change: critic observation (separate ticket if needed)
- DO NOT change: triangulation module internals (only how its output is represented in obs)
- DO NOT change: delay system (frame rotation happens after delay output)

**Affected modules**:
- `iris_ma_env6_test.py` — `_get_observations()` (lines 1672-1910)
- `iris_ma_env6_test_cfg.py` — observation dimension constants, `action_weight` / `action_delta_weight` (dims unchanged but verify)
- `agents/skrl_mappo_rnn_cfg.yaml` — observation space size
- New utility: heading frame rotation function (in env file or shared utils)

**Acceptance criteria**:
- All observations match spec dimensions (ego 29D, inter-agent 19D, tri tail 4D)
- Heading-frame rotation identity test: `rotate_to_heading_frame(v_w, ψ=0) == v_w`
- Geometry features: convergence angle = 0 for parallel rays, π/2 for perpendicular
- Triangulation round-trip: `Rz(ψ) @ tri_pos_v1 + ego_pos_w ≈ tri_pos_w` (atol=1e-5)
- Training runs to convergence; compare learning curves against current observation baseline
- No NaN/Inf in observations under delay system with noise, dropout, and burst dropout enabled

**Dependencies**:
- None (self-contained observation change)
- Pairs well with: ticket-006 (domain randomization integration) for drag DR, ticket-021 (sim2real measurements)

**Flow**: Full QRISPY — this is a breaking change to observation space requiring fresh training and comparison experiments.
