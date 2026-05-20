## Ticket 037 — Critic-only privileged observations for MAPPO-RNN (Phase 1)

**Status**: Open
**Created**: 2026-05-20
**Discovered in**: Ticket 036 (policy adaptation probe — robust verdict on `agent_400000.pt`).
**Affected modules**: `iris_ma_env6_test.py` (shared obs construction), `mappo_rnn.py` (`MAPPORNNValue`), `iris_ma_env6_test_cfg.py` (shared obs dim).
**Out of scope**: actor-side aux sysid head (Phase 2, recorded below).

---

### What

Add per-(env, agent) privileged information to the **centralized critic only**. The actor observation is unchanged. Actor stays deployment-realizable; critic gains a sufficient statistic for the per-env nuisance variance, sharpening advantage estimates fed to the policy gradient.

### Why

Ticket 036 closed with a **robust** verdict — `agent_400000.pt`'s GRU does not encode env identity (Probe 2 R² < 0 on all 7 non-degenerate `_eff_progress_*` axes; Probe 3 detrended signal −0.21 nat). Two consequences:

1. Value-loss floor is set by `Var[return | obs]` rather than `Var[return | obs, env_params]`. The gap is large at t034's full per-env randomization — the policy currently leaves +39 % `action_sum` and +35 % `action_delta` regression vs fix2 on the table because advantages cannot attribute return variance to env-difficulty terms.
2. The actor's GRU has no signal pulling it to encode env identity in the hidden state.

Phase 1 addresses (1). Phase 2 (deferred) addresses (2).

**Expected outcome of Phase 1 alone**: V-loss drops; σ collapses modestly (cleaner advantage → less exploration pressure); `action_sum` / `action_delta` improves but does not necessarily reach fix2 levels. Re-run probe 036 against the new 400 k checkpoint — Probe 2 R² is **not** expected to improve materially (no new gradient pulling toward env-id encoding), unless the actor already had latent capacity Probe 2 missed.

### Critical design question — what does the critic see?

This is the load-bearing decision and the ticket should not be implemented until it is resolved with a written analysis. The two extreme positions:

- **A. `_eff_progress_*` (8 scalars × A)**. Already cached on env as `[N, A]` tensors. In `[0, 1]`, curriculum-aligned, zero new plumbing.
- **B. Actual jittered physical parameters**. `max_lin_vel`, `tau_gimbal`, gimbal dead-time steps, zoom dead-time steps, controller gains, mass, inertia, FOV scale, gimbal mechanical offsets, target scale, latency_s, noise_std, dropout_prob, burst_dropout_prob. ~12–18 scalars depending on what's bundled vs separable.

**Bundling concern (the real issue)**: `progress_dynamics` currently co-ramps **at least seven physically-independent quantities** at [iris_ma_env6_test.py:2637–2708](../../../../iris_ma_env6_test.py):

1. Gimbal rate-loop τ
2. Drone controller gains (`randomize_gains`)
3. Max-linear-velocity multiplicative scale (±20 %)
4. Camera FOV scale (intrinsic randomization)
5. Gimbal mechanical offsets (yaw/pitch/roll misalignment)
6. Robot mass / inertia
7. Gimbal joint stiffness / damping
8. Target object scale (xy, z)

In real hardware these are **independent random variables** — gimbal mass does not correlate with drone controller gains. Training under bundled co-randomization is a **distributional artifact**: a robust policy that exploits the correlation will deploy worse than the metrics suggest, because real deployments lie off the training manifold (one axis jittered, others nominal).

This bias is invisible to the actor (which doesn't see the params) but **directly shapes what the critic learns** if we feed it the bundled `_eff_progress_dynamics`. The critic would learn `V(s, eff_progress_dynamics)` — which conflates seven distinct hardness sources. If we instead feed the actual params, the critic can attribute V to each independently — but at the cost of normalizers, more obs dims, and plumbing across `DroneController`, `GimbalRateLoop`, `ZoomController`, `DomainRandomizer`, `DelaySystem`.

**A third option falls out of this**:

- **C. Unbundled `_eff_progress_*` + targeted decoupling**. Keep the curriculum-progress abstraction but split `progress_dynamics` into separate axes for the seven sub-randomizations above. This is a non-trivial env change (curriculum cfg + sample paths), but produces both a better-conditioned critic input AND a less-correlated training distribution (improving sim-to-real on its own merits).

### Status

- §1 lit review: **complete** (key citations verified against ar5iv; doc at [doc/critic_obs_design.md](../../critic_obs_design.md)).
- §2 comparison matrix: **complete** with refined dim count (48 for A=2, not 57).
- §3 decision: **LOCKED on Option B (independent-draw view, env decoupled)**. Concrete 23 per-agent + 2 shared field list in [critic_obs_design.md §3.1](../../critic_obs_design.md).
- Implementation slices: **see below**, ready to start.

### Workflow

1. **Literature review** (≤ 1 day). Survey published privileged-critic / asymmetric AC / privileged-distillation work for the obs-choice question. Read with a critic-vs-actor-symmetry lens; note what each paper feeds the critic and why. Minimum list:
   - Pinto et al., "Asymmetric Actor Critic for Image-Based Robot Learning" (2018) — original asymmetric AC; full simulator state to critic.
   - Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain" (ETH 2020) — privileged learning, teacher critic sees terrain heightmap + contact info.
   - Kumar et al., "RMA: Rapid Motor Adaptation" (2021) — env-encoder over physical params (mass, friction, motor strength, payload, etc.) as a privileged sub-network, not the critic obs per se but the closest precedent for the "actual params vs progress" question.
   - Yu et al., "The Surprising Effectiveness of PPO in Cooperative MARL" / MAPPO (2022) — centralized-critic obs choices in MARL.
   - Miki et al., "Learning Robust Perceptive Locomotion for Quadrupedal Robots" (ScienceRobotics 2022) — teacher-student with privileged proprioception.
   - Margolis & Agrawal, "Walk These Ways" (2023) — multiplicity-of-good-policies framing, conditional command obs.
   - Hwangbo et al., "Learning Agile and Dynamic Motor Skills" (ScienceRobotics 2019) — actuator-net-based sim2real, critic obs choices.
   
   Output: §3 of [doc/critic_obs_design.md](../../critic_obs_design.md) — what each paper feeds, what they justify, common patterns.

2. **In-depth comparison** of options A / B / C across these axes:
   - **Information sufficiency for V(s, env_params)**. Asymptotic V-loss floor under each choice.
   - **Bundling / independence**. Does the choice preserve the independence structure of real-world parameter variation? (B and C: yes; A: no.)
   - **Sim-to-real correlation risk**. If the policy learns to exploit a training-time correlation that doesn't hold at deployment, how does each choice surface or mask the risk?
   - **Dimensionality and normalization**. Obs dim, scale range, whether RunningStandardScaler is needed.
   - **Plumbing cost**. Module touchpoints, code changes.
   - **Curriculum alignment**. Does the critic input distribution shift smoothly as the curriculum ramps?
   - **Downstream probe-ability**. Can we re-run ticket 036 Probe 2 post-Phase-1 to check if the critic's V is now env-aware? (For B and C: directly. For A: yes but only against bundled scalars.)

   Output: §4 of `doc/critic_obs_design.md` — decision matrix.

3. **Decision and recommendation**. Lock the obs choice. Default tentative recommendation pre-analysis: **C with a phased rollout** — start with A (cheapest) as a control run, then unbundle `progress_dynamics` and re-train under C if the bundling-attribution analysis shows it matters for sim-to-real. **This default must be confirmed or overruled by the §1–2 analysis**, not assumed.

4. **Implementation** (Slice plan, gated on §3):
   - 4.1 — `shared_observation_spaces` in [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py): extend per-agent shared obs by the privileged dims (cfg-flagged, default off so existing checkpoints load).
   - 4.2 — Critic obs construction: concat `_eff_progress_*` (or chosen alternative) into the shared-state tensor before it reaches `MAPPORNNValue`.
   - 4.3 — Cfg flag `enable_critic_privileged_obs: bool = False` on `IrisMA6TestEnvCfg` to gate the dim change; behind-flag retraining requires a fresh log dir.
   - 4.4 — Document the contract in the env CONTEXT.md.

5. **Training run** (~24 h on the dev box at 400 k steps, matching t034's settings exactly except the new cfg flag).

6. **Evaluation against t034 baseline**:
   - **V-loss trajectory** (tensorboard `loss/critic`) — expect lower asymptotic floor.
   - **σ trajectory** (tensorboard `policy/sigma` or `policy/log_std`) — expect modest decrease vs t034's 0.28 plateau.
   - **Action smoothness regression eval**: `action_sum`, `action_delta` at matched steps {120 k, 200 k, 400 k}. Expect partial recovery toward fix2 levels.
   - **Task-quality non-regression**: pair_valid_rate, bbox_center, collisions, track_lost — must not drop ≥ 2 % vs t034 at 400 k.
   - **Re-run ticket 036 probe** against the new 400 k checkpoint. Expectations:
     - Probe 1 ratio comparable to t034 (the reset-settling artifact is policy-agnostic).
     - Probe 2 R² may climb slightly but not above 0.1 (no gradient toward actor-side env-id encoding without Phase 2).
     - Probe 3 detrended signal comparable to t034's −0.21.
   - **Visibility regression eval**: matched-step ratios at steps {39 k, 99 k, 199 k, 399 k} must stay ≥ 0.80 (t034 acceptance bar).

7. **Doc** the design decision and outcome in [doc/experiments/2026-XX-XX_037_critic_privileged_obs.md](../../experiments/) using the standard experiment template.

### Implementation slices (locked on Option B per §3 decision)

Sized for incremental review; each slice is independently testable. Estimated total: ~3 days of engineering + ~24 h training.

#### Slice 1 — Inventory & accessor audit (½ day, no code change)

For each of the 25 fields in [critic_obs_design.md §3.1](../../critic_obs_design.md), locate the canonical owner module and write down:

- Where the value lives (tensor name, shape, layout `[num_envs, num_agents]` or `[num_envs]` or `[num_envs, num_agents, k]`).
- Whether a getter already exists; if not, what to add.
- Whether the stored layout matches our `(num_envs, num_agents)` convention or is per-vehicle / per-batched-controller-row (the batched controller layout reshapes `(num_envs, num_agents)` ↔ `(num_envs * num_agents,)` — getters must return the un-flattened shape).
- For zoom-related fields (`zoom_tau_scale`, `zoom_max_rate_scale`, `zoom_dead_time_s`): the BatchedController convention is non-trivial; double-check.

**Output**: a `field_registry.md` table (or directly into [critic_obs_design.md](../../critic_obs_design.md) appendix) mapping each of the 25 fields to its accessor.

#### Slice 2 — Cfg surface (¼ day)

In [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py):

```python
# New cfg fields
critic_privileged_fields: list[str] = field(default_factory=list)
# Empty list = no privileged obs (t034 bit-exact reproduction).
# Full list per design doc §3.1 enables the 48-dim privileged channel.

enable_axis_independence: bool = False
# Decouple progress_dynamics into 8 independent curriculum axes
# (per ticket 037 decoupling sub-scope). False = bit-exact t034.
```

Add a constant `_CRITIC_PRIVILEGED_FIELD_REGISTRY` listing valid field names + dim + per-field-norm range, drawn from [critic_obs_design.md §3.1](../../critic_obs_design.md). Validate `critic_privileged_fields` against this registry at cfg post-init.

#### Slice 3 — Accessor getters per module (1 day)

Add getter methods identified in Slice 1. Convention: each returns a tensor of shape `(num_envs, num_agents)` (or `(num_envs,)` for the 2 shared fields). Pre-normalization happens in Slice 4, not in the getters — getters return native units.

Touchpoints:

- [controller/drone_controller.py](../../../../controller/drone_controller.py): cache the 6 gain scales in `randomize_gains` (currently the scales are local-variable-and-discarded; need to store them to per-(env, agent) tensors `self._gain_scales` of shape `(num_envs, num_agents, 6)`). Add `get_gain_scales(self) -> Tensor`.
- [controller/gimbal_rate_loop.py](../../../../controller/gimbal_rate_loop.py): expose `get_tau()` and `get_dead_time_s()`.
- [controller/zoom_controller.py](../../../../controller/zoom_controller.py): expose `get_dead_time_s()`.
- [domain_randomization/domain_randomizer.py](../../../../domain_randomization/domain_randomizer.py): expose `get_fov_scales()` (already exists), `get_gimbal_mech_offsets()` (already exists per code), `get_gimbal_mass_scales()`, `get_gimbal_stiffness_damping()`.
- [domain_randomization/physics_randomizer.py](../../../../domain_randomization/physics_randomizer.py): expose `get_body_mass_scale_or_addition()` — returning whichever of `mass_scales` or `mass_additions` is the active mode.
- [delay_system_v3/delay_pipeline_v3.py](../../../../delay_system_v3/delay_pipeline_v3.py): expose `get_latency_s()`, `get_noise_std()`, `get_dropout_prob()`, `get_burst_dropout_prob()` (all per (env, agent)).
- [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py): expose `get_max_lin_vel_per_env_per_agent()` (currently `self._max_lin_vel` is shape `(num_envs,)` — verify whether it should be per-agent; if not, broadcast in privileged obs construction).
- [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py): expose `get_target_scale()` returning the per-env shared `(num_envs, 2)` `[xy, z]`.

**Test for each getter**: assert the returned value equals the actual physics-applied value (via a property test that reads the asset post-apply and compares).

#### Slice 4 — Privileged obs assembly (½ day)

In [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py):

```python
def _get_privileged_obs(self) -> Tensor:
    """Build the privileged obs tensor (num_envs, num_agents, K_per_agent + 2 shared)
    for the centralized critic. Returns native-units → pre-normalized to [-1, +1]
    using cfg-known ranges from _CRITIC_PRIVILEGED_FIELD_REGISTRY."""
```

Iterate over `cfg.critic_privileged_fields`, look up each in the registry, call the accessor, pre-normalize, concat. Returns `(num_envs, num_agents * 23 + 2)` for the full field list.

Concat order: per-agent fields in the order of `_CRITIC_PRIVILEGED_FIELD_REGISTRY`; shared fields appended last. **Deterministic ordering** matters for checkpoint compat.

#### Slice 5 — Shared obs wiring (¼ day)

Locate the shared-obs construction in [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) (the path that populates `shared_observation_spaces`). Append the privileged tensor (cfg-gated).

Update `shared_observation_spaces` dim:

```python
shared_obs_dim = base_shared_dim + (
    len(cfg.critic_privileged_fields) * num_envs_field_dim_contribution
)
```

where the per-field dim contribution comes from the registry.

#### Slice 6 — SKRL preprocessor handling (¼ day)

In [agents/skrl_mappo_rnn_cfg.yaml](../../../../agents/skrl_mappo_rnn_cfg.yaml) and the agent setup in [train_mappo_rnn_hydra.py](../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py):

- `shared_state_preprocessor` size auto-derives from `env.shared_observation_spaces` — verify this still works with the extended dim.
- Since we pre-normalize in the env (Slice 4), the `RunningStandardScaler` will see [-1, +1] inputs for the privileged dims; it'll re-normalize but the impact will be small.

Add a sanity test: when `critic_privileged_fields = []`, the preprocessor's input dim equals t034's exactly.

#### Slice 7 — Env decoupling sub-scope (1 day)

Separate from the critic-obs work but tightly coupled: split `progress_dynamics` into 8 independent curriculum axes.

In [curriculum/curriculum_cfg.py](../../../../curriculum/curriculum_cfg.py) (or wherever curriculum cfg lives):

```python
# Replace the single (dynamics_start_step, dynamics_end_step) with:
gimbal_rate_tau_start_step / _end_step
drone_controller_gains_start_step / _end_step
max_lin_vel_scale_start_step / _end_step
camera_fov_start_step / _end_step
gimbal_mech_offsets_start_step / _end_step
mass_inertia_start_step / _end_step
gimbal_stiff_damp_start_step / _end_step
target_scale_start_step / _end_step

# Backwards-compat: keep `dynamics_start_step` / `_end_step` as an alias
# that sets all 8 to the same values when `enable_axis_independence=False`.
```

In [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) `_reset_idx`: replace the 7 sites currently gated on `self.progress_dynamics > 0.0` and `self._eff_progress_dynamics` with the corresponding per-axis progress reads. Add 7 new `_eff_progress_*` tensors of shape `(num_envs, num_agents)`.

**Regression test**: run a 1k-step short rollout with `enable_axis_independence=False` and verify per-step env state matches t034 within float tolerance.

#### Slice 8 — Unit tests (½ day)

- `test_privileged_obs_dim.py`: dim equals expected count for various `critic_privileged_fields` configurations.
- `test_privileged_obs_normalization.py`: every output value is in [-1, +1].
- `test_privileged_obs_correctness.py`: round-trip — denormalize via the registry's range and assert equality to the actual physics-applied value.
- `test_axis_independence.py`: with `enable_axis_independence=False`, the 8 new `_eff_progress_*` tensors are pairwise identical at every reset (alias path validates).
- `test_t034_bitexact.py`: with `critic_privileged_fields=[]` and `enable_axis_independence=False`, the shared_obs tensor and all env dynamics match t034 at every step in a fixed-seed rollout.

#### Slice 9 — Training run (~24 h wall-clock)

Mirror the t034 cfg exactly, plus:

```yaml
critic_privileged_fields: [<full 25-field list per design doc §3.1>]
enable_axis_independence: true
```

400 k steps, seed=0 (matching t034), same `skrl_mappo_rnn_cfg.yaml`. Log dir name: `037_t034_critic_priv_obs_axis_indep`.

#### Slice 10 — Evaluation (½ day, post-train)

Per ticket §6 (this section, above the Implementation slices):

- V-loss trajectory vs t034.
- σ trajectory vs t034.
- Action smoothness regression: `action_sum`, `action_delta` at {120k, 200k, 400k}.
- Task-quality non-regression: pair_valid_rate, bbox_center, collisions, track_lost.
- Visibility regression ratios at {39k, 99k, 199k, 399k}.
- Re-run [adaptation_probe.py](../../../../experiments/adaptation_probe.py) against `037` 400 k. Expected: probes 1+2+3 still robust verdict (Phase 1 doesn't add actor-side env-id encoding).

#### Slice 11 — Writeup (½ day)

`doc/experiments/2026-XX-XX_037_critic_privileged_obs.md` using the standard experiment template, with the §6 metrics table. If outcomes are surprising (e.g., actor metrics move beyond Phase 1 expectation), open Phase 2 ticket prematurely.

### Acceptance criteria

- §1–3 written and committed to [doc/critic_obs_design.md](../../critic_obs_design.md) **before** any code change to the env or model.
- Cfg-flagged implementation: `enable_critic_privileged_obs = False` reproduces t034 behavior bit-exactly (regression: same 400 k checkpoint metrics).
- 400 k training run completes; tensorboard scalars logged, checkpoints saved.
- V-loss at 400 k is ≥ 15 % below t034's V-loss at 400 k. (Soft target — exact threshold TBD post lit review; the magnitude depends on how much of the return variance is attributable to env_params.)
- All four visibility ratios at {39 k, 99 k, 199 k, 399 k} ≥ 0.80 vs the new run's matched-step references.
- Task-quality metrics at 400 k within ±2 % of t034.
- Ticket 036's probe re-run report: comparable to t034 on all three probes (no actor-side change expected).

### Scope boundary

- **DO**: extend shared obs (critic input only), keep actor obs unchanged, ship behind a cfg flag.
- **DO**: complete the design doc before coding — this is the load-bearing decision and is too easy to get wrong by default.
- **DO**: **decouple `progress_dynamics`** into separate curriculum axes for the 7 physically-independent quantities currently bundled (see [iris_ma_env6_test.py:2637–2708](../../../../iris_ma_env6_test.py)). Each becomes its own per-(env, agent) `_eff_progress_*` axis sampled independently. This is now in scope alongside the critic-obs work because: (a) the bundling masks sim-to-real failure modes regardless of what the critic sees, (b) without decoupling, the critic-obs choice is forced toward option A by default, and (c) the env-side change and the critic-side change are co-dependent design decisions.
- **DO**: re-run ticket 036's probe against the new checkpoint to confirm/refute the "Phase 1 doesn't make actor adaptive" prediction.
- **DO**: use an extensible cfg surface — `critic_privileged_fields: list[str]` rather than a single bool — so future privileged fields (e.g., aux sysid head outputs in Phase 2) can be added without re-architecting.
- **DO NOT**: change actor obs.
- **DO NOT**: add an auxiliary loss head on the actor (Phase 2; see below).
- **DO NOT**: claim adaptive behavior. Phase 1 alone is value-function variance reduction.

### Decoupling sub-scope (new — within Phase 1)

The current `progress_dynamics` co-ramps at least 7 physically-independent quantities. Sim-to-real benefit and critic-obs interpretability both improve when these are independent curriculum axes. Per-axis breakdown to expose as separate `_eff_progress_*`:

1. `_eff_progress_gimbal_rate_tau` — gimbal rate-loop τ (currently the headline meaning of `_eff_progress_dynamics`)
2. `_eff_progress_drone_controller_gains` — drone position/velocity/attitude controller gains
3. `_eff_progress_max_lin_vel_scale` — multiplicative ±20 % scale on per-env max linear velocity
4. `_eff_progress_camera_fov` — FOV scale perturbation (intrinsic randomization)
5. `_eff_progress_gimbal_mech_offsets` — mechanical yaw/pitch/roll misalignment
6. `_eff_progress_mass_inertia` — robot mass and inertia
7. `_eff_progress_gimbal_stiff_damp` — gimbal joint stiffness / damping
8. `_eff_progress_target_scale` — target-object xy and z scale (also currently gated by `progress_dynamics`)

Curriculum cfg gets 8 new `(start_step, end_step)` pairs replacing the single `dynamics_*` pair. **Backwards-compat path**: keep `progress_dynamics` as an alias that sets all 8 to the same schedule by default, so existing runs reproduce; new runs set them independently.

Acceptance for the decoupling sub-scope alone:
- All 8 axes are independently sample-able via `_curriculum_generator`.
- `enable_axis_independence: bool = False` (cfg-flagged) — when False, all 8 axes share the original `progress_dynamics` schedule (bit-exact reproduction of t034). When True, they're set independently per the new cfg fields.
- Unit test: under `enable_axis_independence=False`, the 8 new `_eff_progress_*` tensors are pairwise identical at every reset.

### Risk

- **Low** on training stability — adding obs to the critic alone is a well-trodden pattern (asymmetric AC). KL stays in band; PPO clip behavior unchanged.
- **Medium** on choosing the wrong obs. If §1–2 are skipped and we default to bundled `_eff_progress_*`, the critic learns a value function tied to a training-time correlation that doesn't exist in deployment — masking sim-to-real failure modes the actor may inherit at training time. This is the reason §1–3 are gates.
- **Low** that Phase 1 alone produces no measurable improvement — even cleaner advantages should reduce variance somewhere. Worst case: V-loss drops, σ unchanged, actor regression unchanged → confirms the robust-policy ceiling and motivates Phase 2.

### Phase 2 — actor-side aux sysid head (future reference)

Recorded here so the path forward is clear. **NOT in scope for ticket 037**.

If Phase 1 ships cleanly but the actor's regression metrics (σ, action_sum) don't recover toward fix2, the next intervention is **auxiliary supervision on the actor's GRU output** that forces env-identity encoding. Pattern mirrors the triangulation head at [doc/policy_triangulation_head_spec.md](../../policy_triangulation_head_spec.md):

- New `MAPPORNNPolicyWithSysidHead` subclass: post-GRU hidden state → small MLP head → `μ_eff_progress, log σ²_eff_progress` over the same 8 (or 12–18) latent axes chosen in Phase 1.
- Train with Gaussian NLL against the env's ground-truth `_eff_progress_*` (or actual params); auxiliary loss weight ≈ 0.01 (tune by ablation).
- Re-run ticket 036 Probe 2 across training checkpoints. **Acceptance**: R²_mlp climbs above 0.5 on at least 4 of 7 axes by training-end. This is the direct test that the GRU now encodes env identity — the precondition for env-conditional behavior.
- After confirming Phase 2 trains a representation-aware actor, evaluate whether downstream `action_sum` / `action_delta` improves beyond Phase 1. If yes — adaptive behavior achieved. If no — the actor has the representation but is still finding a robust-action policy optimal, motivating a stronger intervention (e.g., privileged-distillation teacher-student).

Phase 2 opens as a separate ticket post-Phase-1 evaluation. Its motivation depends on Phase 1's results.

### Affected files

To be confirmed by §1–3 design doc. Tentative:

- NEW: [doc/critic_obs_design.md](../../critic_obs_design.md) (literature review + decision)
- EDIT: [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py) (cfg flag, shared obs dim, `critic_privileged_fields`, `enable_axis_independence`)
- EDIT: [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) (shared obs construction; 8 new `_eff_progress_*` tensors for decoupling sub-scope)
- EDIT: [curriculum/](../../../../curriculum/) module — 8 new `(start_step, end_step)` schedules replacing `dynamics_*`; alias path for backwards-compat.
- EDIT: [agents/skrl_mappo_rnn_cfg.yaml](../../../../agents/skrl_mappo_rnn_cfg.yaml) (shared obs preprocessor size — may auto-derive)
- NEW: [doc/experiments/2026-XX-XX_037_critic_privileged_obs.md](../../experiments/) (post-run writeup)

### References

- Ticket 036 (probe results; closed with robust verdict).
- Ticket 034 (per-env jitter curriculum; the t034 baseline this builds on).
- `agent_400000.pt` at `logs/skrl/iris_ma6/2026-05-18_11-45-51_..._ticket034_per_env_jitter/checkpoints/`.
- Pinto et al. 2018 (asymmetric AC); Lee et al. 2020 (ETH quadruped); Kumar et al. 2021 (RMA); Yu et al. 2022 (MAPPO).

**Flow**: Medium. Design doc (§1–3) is the hard part; implementation (§4–5) is mechanical; evaluation (§6) reuses existing infra.
