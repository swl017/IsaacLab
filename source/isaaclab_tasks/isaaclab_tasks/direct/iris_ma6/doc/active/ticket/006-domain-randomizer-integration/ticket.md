## Ticket: Integrate DomainRandomizer into iris_ma_env6_test._reset_idx

**What**: Wire the existing `DomainRandomizer` module into the environment's `_reset_idx` so that camera intrinsics, physics mass/material, and gimbal dynamics are randomized per-episode during training. Currently these are all fixed — the `DomainRandomizer` code exists with configs and tests but is never called from the env.

**Why**: The sim-to-sim transfer analysis (ticket-005) identified this as the P0 gap. Without camera/physics/gimbal randomization, the policy is trained with zero robustness to:
- Camera focal length variation (real cameras differ from sim default)
- Mass/inertia variation (payload, battery, manufacturing tolerances)
- Gimbal mechanical tolerances (axis alignment, stiffness, damping)
- Material properties (friction, restitution on landing)

The `DomainRandomizer` module already implements all of this with curriculum-compatible APIs. Only the wiring into `_reset_idx` is missing.

**Note**: Only mock camera parameters (intrinsics + resolution) are used by the bbox raycaster — no actual RGB image processing is needed. The camera randomization affects the intrinsic matrix used for bbox projection and ray computation, not rendering.

**Scope**:
- Call `DomainRandomizer.randomize_all(env_ids)` from `_reset_idx`
- Add curriculum gating: ramp randomization ranges with `progress_dynamics` (same phase as gain randomization, steps 180k-200k)
- Pass randomized intrinsic matrices to bbox raycaster and delay system
- Pass randomized mass to `DroneController` (affects thrust-to-weight ratio)
- Pass randomized gimbal dynamics to gimbal actuators (stiffness, damping)
- Enable gimbal mount offset randomization (`enabled=True`)
- Add `DomainRandomizerCfg` field to `IrisMA6TestEnvCfg`

**Scope boundary**:
- Do NOT modify the `DomainRandomizer` module itself (already complete)
- Do NOT add new randomization types (burst dropout, heavy-tail latency — separate tickets)
- Do NOT change the camera rendering pipeline (no RGB processing)
- Do NOT change the RL policy or reward function

**Affected modules**:
- `iris_ma_env6_test.py` — add DomainRandomizer instantiation in `__init__`, call in `_reset_idx`
- `iris_ma_env6_test_cfg.py` — add `DomainRandomizerCfg` field
- `ARCHITECTURE.md` — update dependency graph (env → domain_randomization)

**Key references**:
- DomainRandomizer: `domain_randomization/domain_randomizer.py` (randomize_all, apply_physics, apply_gimbal_dynamics)
- Camera processor: `domain_randomization/camera_processor.py` (randomize, get_intrinsic_matrices)
- Physics randomizer: `domain_randomization/physics_randomizer.py` (sample + apply)
- Gimbal randomizer: `domain_randomization/gimbal_randomizer.py` (offsets + dynamics)
- Config: `domain_randomization/domain_randomization_cfg.py` (all ranges and defaults)
- Sim-to-sim analysis: `doc/active/ticket/005-sim-to-sim-transfer/analysis.md`

**Acceptance criteria**:
1. `DomainRandomizer.randomize_all()` called in `_reset_idx` for each reset batch
2. Randomized intrinsics flow through to bbox raycaster (verify bbox changes with different focal lengths)
3. Randomized mass affects controller thrust-to-weight (verify hover altitude drift with ±10% mass)
4. Gimbal offsets active (verify pointing bias with offset randomization)
5. Curriculum gating: randomization ranges = 0 at progress=0, full at progress=1
6. No training regression: pair_valid_rate and bbox metrics unchanged at early curriculum (progress < 0.8)

**Follow-up**:
- [ ] Visually verify target z-scale randomization in bbox using the teleop script (run at full dynamics progress, confirm bbox height changes across resets)

**Flow**: Light
