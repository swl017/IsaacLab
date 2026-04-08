## Ticket: Camera Intrinsic Observability — What Should the Policy Know?

**What**: Decide whether and how to expose camera intrinsic parameters (focal length / FOV) to the policy's observation vector, now that domain randomization varies these per episode (ticket-006). Currently the agent sees bbox pixels and zoom level but has no information about the underlying camera model, making distance ambiguous under DR.

**Why**: With DR active (ticket-006), the mapping from bbox pixel size to target distance is no longer fixed. A 50x30 px bbox at zoom=1.0 could mean:
- Normal target at 20m with nominal intrinsics
- 2x-scaled target at 40m with nominal intrinsics
- Normal target at 10m with narrow FOV (0.5x fov_scale)

Without intrinsic information, the agent cannot disambiguate distance from a single camera view. This affects:
1. **Triangulation geometry** — FIM reward depends on formation spacing relative to target distance; wrong distance estimate degrades coordination
2. **Zoom decisions** — is the bbox small because the target is far or because FOV is narrow?
3. **Formation spacing** — maintaining good triangulation baseline requires knowing approximate target range

This is an architectural decision that shapes what the policy can learn vs. what it must be robust to. Getting it wrong either limits asymptotic performance or creates a brittle policy.

**Analysis**:

### What matches real deployment?

| Parameter | Known at deployment? | Recommendation |
|-----------|---------------------|----------------|
| Camera focal length / FOV | **Yes** — factory-calibrated, fixed per camera | **Explicit** in obs |
| Zoom level | **Yes** — commanded by policy | Already explicit (1D) |
| Target 3D size | **No** — genuinely unknown | **Implicit** (keep out of obs) |
| DR mass / gimbal offsets | **No** — unknown model error | **Implicit** (keep out of obs) |

### Camera intrinsics: should be explicit

- At deployment, the agent KNOWS its camera calibration (focal length, principal point)
- During training, DR varies intrinsics to simulate different cameras
- If intrinsics are NOT in observations, the agent must learn a single policy that works across all FOV values — this is "blind-to-intrinsics" training
- Blind-to-intrinsics forces robustness but caps performance: the agent can't exploit its known camera geometry
- With intrinsics in observations, the agent learns to USE calibration data, which is strictly more capable

### Target 3D size: should be implicit

- At deployment, target size is genuinely unknown (different targets, unknown geometry)
- DR target scale randomization (ticket-006) intentionally breaks the monocular size-distance coupling
- The agent must learn to rely on multi-view triangulation geometry rather than single-view size heuristics
- This is a core benefit of target scale DR

### Representation options

| Option | Dim | Pros | Cons |
|--------|-----|------|------|
| `dr_intrinsic_scale` (relative to nominal) | 1D | Compact, direct | Deployment needs conversion from calibration |
| `effective_hfov` (radians) | 1D | Physical, deployment-ready | Scale varies, may need normalization |
| `zoom * dr_intrinsic_scale` (combined) | 1D | Single number captures total scaling | Redundant with existing zoom obs |
| `(fx, fy)` in pixels | 2D | Complete, raw | Too low-level, fx=fy for square pixels |
| Separate `dr_intrinsic_scale` + existing `zoom` | 1D new | Agent can reason about zoom vs. camera independently | Two scaling factors to combine |

**Recommended**: Add `dr_intrinsic_scale` (1D) to ego observations. The agent already has `zoom` — adding the camera-specific scale lets it compute effective focal length. At deployment, this value is derived from the known camera calibration relative to the training nominal.

**Scope**:
- Add 1D camera intrinsic scale to ego observation (obs_dim 30 → 31)
- Wire through delay system (register, store, retrieve, noise)
- Update obs_dim in cfg and __post_init__
- Update SKRL config for new obs dimension
- Do NOT add target scale to observations (intentionally implicit)
- Do NOT add mass/gimbal DR parameters to observations (unknown at deployment)

**Scope boundary**:
- Do NOT change reward function
- Do NOT change action space
- Do NOT change inter-agent observation structure (other agents don't broadcast their intrinsics — they could, but that's a separate decision)

**Key question for review**:
Should inter-agent observations include the other agent's intrinsic scale? In deployment, agents could share calibration data over the comm link. This would let agent A reason about agent B's camera geometry for triangulation. Cost: +1D per other agent. Benefit: better cooperative triangulation planning.

**Acceptance criteria**:
1. Ego obs includes camera intrinsic scale (1D), wired through delay system
2. At progress_dynamics=0 (no DR), intrinsic scale = 1.0 for all envs
3. At progress_dynamics=1, intrinsic scale varies per env (matches _dr_intrinsic_scale)
4. Training with intrinsics in obs matches or exceeds blind-to-intrinsics baseline
5. No regression in early training metrics (bbox_center, pair_valid_rate at 0-20k steps)

**Dependencies**: ticket-006 (domain randomizer integration) — completed

**Flow**: Medium
