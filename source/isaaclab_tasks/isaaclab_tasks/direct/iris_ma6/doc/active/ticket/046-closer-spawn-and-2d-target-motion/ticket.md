## Ticket 046 — Closer spawn + 2D target motion (task-difficulty reduction, Phase 1)

**Status**: Proposed
**Created**: 2026-06-02
**Type**: Cfg patch + one feature flag + A/B validation.
**Target setup**: iris_ma6 training, post-t045 (slew clip ON + envelope-down). Pegasus SITL + PX4 SITL transfer remains the load-bearing case.
**Deliverable**:
1. Cfg patch lowering spawn geometry: `initial_states.cylinder_diameter_max`, `target_distance_max`, `target_height_offset_max`. No flag — these are scalar value changes.
2. One new feature flag `target_controller.enable_z_motion: bool = True` (default preserves prior behavior). When False, the target moves in the horizontal plane only (z velocity clamped to 0 in `target_controller._apply_constraints`).
3. Two Slice-1 experiments (`baseline`: current cfg at full spawn radius + 3D target motion; `treatment`: closer spawn + 2D target motion). 200k × seed=42.
4. One-page experiment writeup in `doc/experiments/`.
5. Cfg-default flips (gated on Slice-1 acceptance).

**What**: The t045 200k A/B + 9-call probe sweep showed the PX4-strict slew clip is the binding constraint for at least 90% of policy steps on every channel × every checkpoint × every curriculum stage (126/126 cells at p90 = δ_max). Envelope tuning (t045 lowered `max_lin_vel` 10→5) was exhausted as a lever — the slew bandwidth budget is consumed by bbox tracking regardless of envelope, leaving no headroom for the cooperative bearing maneuvers triangulation needs (stuck at 21 vs baseline 40, −47%).

The remaining levers are **task-difficulty reduction** (this ticket) and **reward retune** (ticket 047, depends on 046). 046 targets the two largest "physical agility demands" that compete with triangulation for slew bandwidth:

1. **Spawn far-apart** — at `cylinder_diameter_max = 100` and `target_distance_max = 40`, agents can be 50m apart with the target another 40m away → translation chase dominates the slew budget before any bearing maneuver is possible. Closer placement keeps the target near-permanently in view, freeing slew bandwidth for cooperative geometry.
2. **3D target motion** — the linear-mode FSM generates targets via `direction = torch.randn(n, 3)` followed by velocity scaling, so the target moves on a unit-3-sphere of directions. Vertical motion forces agents to expend slew on z corrections, doubly so under the asymmetric z envelope (`max_vel_z_up = 3.0`, `max_vel_z_dn = 1.5`). Restricting to 2D motion eliminates this z chase, matching what most real-world target-following deployments care about (the target is typically a ground or low-altitude vehicle).

**Why a flag for z=0 but not for spawn**: spawn values are dialed-in via cfg defaults (they were tuned in t034 anti-forgetting work) and don't need a regression escape hatch — the previous values can be restored by overriding the cfg if needed. The z=0 behavior, however, is a *capability change* on the target controller — `enable_z_motion=False` is the new behavior, `True` preserves backward compat for any consumer (other tickets / sweeps) that still wants 3D motion.

**Single insertion point for z=0**: [target_controller.py:240](../../../target_controller/target_controller.py#L240) — `v_cmd = self._apply_constraints(v_cmd, pos_flat, curriculum_progress)`. Add `v_cmd[:, 2] = 0.0` inside `_apply_constraints` gated on the flag. No changes to the four velocity generators (`linear_mode.py`, `circular_mode.py`, `approach_mode.py`, `evade_mode.py`) — they continue to emit 3D velocities; the orchestrator zeros the z component before applying.

**Why not retune rewards in the same ticket**: separation of concerns. 046 isolates the *task-difficulty* contribution. 047 (depends on 046) layers reward retune on top. Combined-change tickets are harder to attribute when one fails: if 046+047-as-one regresses, we don't know if the spawn change or the reward change is the culprit. Sequential A/Bs surface the contributions cleanly.

### Patch summary

1. **`iris_ma_env6_test_cfg.py`** — spawn-geometry value changes (no flags):
   ```python
   # Before                                                After
   initial_states.cylinder_diameter_max = 100.0           → 30.0
   initial_states.target_distance_max   = 40.0            → 15.0
   initial_states.target_height_offset_max = 4.0          → 2.0
   ```
   Curriculum mechanism unchanged — the curriculum still ramps from `*_min` to `*_max`, just with a smaller max. Verifying:
   - `cylinder_diameter_min = 20.0 → cylinder_diameter_max = 30.0`: agents within 20-30m spread (was 20-100m). Effective shrink: 3.3× on the dynamic range.
   - `target_distance_min = 10.0 → target_distance_max = 15.0`: target 10-15m from cylinder center (was 10-40m). Effective shrink: 5.7× on the dynamic range.

2. **`target_controller/target_controller_cfg.py`** — one new flag:
   ```python
   enable_z_motion: bool = True
   """Ticket 046 — when True (default, backward-compat), the target velocity
   command preserves its z component as generated by the active behavior FSM
   (linear / circular / approach / evade). When False, ``_apply_constraints``
   zeros ``v_cmd[:, 2]`` so the target moves in the horizontal plane only —
   used by the t046 task-difficulty reduction to eliminate vertical chase
   that consumes agent slew bandwidth."""
   ```

3. **`target_controller/target_controller.py`** — single insertion in `_apply_constraints`:
   ```python
   # In _apply_constraints, after existing speed/altitude clamps:
   if not self.cfg.enable_z_motion:
       v_cmd[:, 2] = 0.0
   ```
   No changes to any velocity_generators/*.py. The generators still emit (N, 3) velocities; the orchestrator zeros z.

4. **`experiments/experiment_registry.py`** — two Slice-1 entries:
   - `validation_task_geom_baseline`: cfg unchanged (current defaults). Comparison anchor; the t045 in-flight log can serve as a partial baseline through 200k if a fresh seed-42 run is too expensive, but a clean baseline is preferred for the writeup.
   - `validation_task_geom_treatment`: cfg overrides for the three spawn values + `target_controller.enable_z_motion=False`.

5. **No env-side architectural changes.** Slew clip, prev-action obs, reset bug fix, and t044/045 trackers are preserved as-is. This is a cfg + 2-line behavior-flag ticket.

### Slices

**Slice 0 — None.** No empirical pre-flight needed; the t045 probe already characterized the policy's slew-bound regime. We move directly to implementation.

**Slice 1 — patch + bit-exact regression unit test.**
- Cfg patch + flag wiring.
- Unit test under `tests/test_target_z_motion_v1.py`:
  - With `enable_z_motion=True` (default): bit-exact regression. Target velocity z-component matches pre-patch.
  - With `enable_z_motion=False`: `v_cmd[:, 2] == 0.0` for all targets across all behavior modes (linear / circular / approach / evade). Verify by stepping the env, observing target velocity.
  - Spawn geometry: with the new cfg defaults, initial target/agent positions sample from the smaller cylinders; verify `(target_pos - cylinder_center).xy.norm() ≤ target_distance_max`.

**Slice 2 — short A/B training run.**
- Two configs, same RNG seed (42), 200k steps:
  - `validation_task_geom_baseline`: current cfg defaults.
  - `validation_task_geom_treatment`: closer spawn + z=0.
- Metrics:
  - **Visibility (load-bearing)**: `Info / Detection/pair_valid_rate` (target ≥ 0.95), `Info / Detection/all_invalid_rate` (target near zero).
  - **Triangulation recovery**: `Info / Episode_Reward/drone_X_triangulation` (target ≥ 32, mid-bar between t045's 21 and baseline 40).
  - **Slew saturation**: `Info / Action_Smoothness/slew_sat_drone_X_*` (predicted to *drop* on at least the vz and yaw channels, since target z chase + agile bearing slewing are removed).
  - **Task quality preservation**: `bbox_center`, `bbox_size`, `tracking_lost_fraction` — should stay within ±5% of t045 (the prior treatment), or improve since the task is now easier.
  - **Training health**: KL, LR, value_loss within healthy bands.
- Acceptance bar:
  - `pair_valid_rate ≥ 0.95` (visibility near-permanent — the design intent of closer spawn).
  - `triangulation ≥ 32` (≥ 80% of t043 prev_action_only baseline, breaking out of t045's 21 plateau).
  - At least 2 channels show `slew_sat ≤ 0.70` (visible relief on the slew bandwidth budget).
  - Reward ≥ t045 reward (not worse than the prior treatment).
  - No new safety regression (`collision_fraction ≤ 2× t043 baseline`).

**Slice 3 — cfg-default flip (conditional on Slice 2 success).**
- Land the four overrides into `IrisMA6TestEnvCfg` + `TargetControllerCfg` defaults.
- Update `feature_list.json`, `progress.txt`, and `iris_ma6_env_spec.md` to reflect the new spawn + z-clamp posture.

### Method (training-time validation)

1. Apply patch (cfg values + flag wiring + 2-line `_apply_constraints` change). Slice-1 unit test passes.
2. Register Slice-1 experiments. No cfg-default flips yet.
3. Launch `validation_task_geom_treatment` (200k × seed=42). Optionally launch `_baseline` if a fresh seed-42 baseline is wanted (the t045 in-flight log is the partial alternative).
4. Decision rule (read at 200k):
   - All acceptance bars passed → trigger Slice 3 cfg-default flip. Update session docs.
   - Visibility passed but triangulation didn't recover → confirms reward-shape is the residual bottleneck → trigger Ticket 047 (reward retune on top of 046's cfg).
   - Visibility didn't pass (pair_valid < 0.95) → the spawn values are still too generous → revisit Slice 1 with tighter values (e.g., diameter_max = 20, distance_max = 10).
   - Saturation didn't drop → the slew bandwidth is structurally insufficient even for visibility-only tracking → escalate to Ticket 049 (loosen slew, deferred).

### Acceptance criteria

| Criterion | Result |
|---|---|
| Cfg patch landed (spawn values + `enable_z_motion` flag) | pending |
| Single-point z-clamp in `target_controller._apply_constraints` (no generator changes) | pending |
| `enable_z_motion=True` produces bit-exact pre-patch target velocity z | unit test (Slice 1) |
| `enable_z_motion=False` zeros target velocity z across all 4 behavior modes | unit test (Slice 1) |
| Spawn samples within new cylinder dimensions for fixed seed | unit test (Slice 1) |
| Slice-2 `pair_valid_rate ≥ 0.95` at 200k | tensorboard |
| Slice-2 `triangulation ≥ 32` at 200k | tensorboard |
| Slice-2 ≥ 2 channels with `slew_sat ≤ 0.70` | tensorboard |
| Slice-2 reward ≥ t045 reward | tensorboard |
| `collision_fraction ≤ 2× t043 baseline` (load-bearing safety) | tensorboard |
| `doc/experiments/<date>_ticket046_closer_spawn_2d_target.md` writeup landed | written |

### Scope boundary

- **DO**: lower spawn geometry (`cylinder_diameter_max`, `target_distance_max`, `target_height_offset_max`) to the values listed. Add `enable_z_motion` flag with the single-point implementation in `_apply_constraints`. A/B + writeup. Flip defaults if Slice 2 passes.
- **DO NOT**: retune rewards. Defer to ticket 047 entirely. The whole point of separating 046 from 047 is to attribute the triangulation recovery (or lack of it) to *task-difficulty* alone, without entangling reward changes.
- **DO NOT**: loosen the slew clip. Deferred to ticket 048, gated on 046+047 establishing a smooth-behaving baseline. The user's principle: prove the PX4-strict envelope works at the right task difficulty first, then loosen with paired PX4 SITL parameter retune.
- **DO NOT**: change the velocity generators (`linear_mode.py`, etc.). Single-point z-clamp in the orchestrator preserves their interface and makes the flag behavior easy to reason about / unit-test.
- **DO NOT**: change initial_states curriculum mechanism. Only the `*_max` values change; the curriculum still ramps from `*_min` to the new `*_max`.
- **DO NOT**: revert t043 (prev_action_obs), t044 (slew clip + reset bug fix + new trackers), or t045 (envelope downshift). 046 builds on the t045 cfg state.

### Risk

Low.

1. **Triangulation still doesn't recover** under closer spawn + 2D target → confirms the bottleneck is reward-shape, not task-geometry. This is a *welcome* negative result — it gates Ticket 047 with high confidence. Mitigation: ticket 047 is already designed as the immediate follow-up.

2. **`pair_valid_rate` overshoots** (e.g., 0.99+) — visibility too easy → the policy may never face challenging bearing geometries → triangulation reward stays low because the FIM is degenerate (parallel bearings on a stationary target). Diagnostic: if pair_valid is high but triangulation is low, the issue is geometric not visibility — slightly relax spawn (e.g., diameter_max = 40 instead of 30). The proposed value 30 is a midpoint; tunable.

3. **2D target motion makes the task uninterestingly easy** for sim-to-real → real targets (vehicles, people) often move along terrain features that approximate 2D motion, so this is *more* realistic, not less. The `enable_z_motion=True` escape hatch preserves the 3D regime for any sweep / ablation that wants it (e.g., aerial target ablation, separate ticket).

4. **Spawn shrink invalidates t034 anti-forgetting expectations** — the anti-forgetting work assumed a 20-100m diameter range. Shrinking to 20-30m compresses the per-(env, agent) progress sampler's spawn diversity. Mitigation: the anti-forgetting wider-distribution principle still holds within the new range; t034's mechanism is preserved.

5. **Existing checkpoints not directly comparable** — pre-046 checkpoints were trained against far-apart spawn + 3D target. Loading them into 046's env will produce policies that work but underperform (overfit to the old regime). Forward-only; not blocking.

### Coupling

- **Ticket 045** (envelope downshift, landed) — 046 builds on the t045 cfg state (max_lin_vel=5, action_slew_vel_xy=0.040, target_controller.max_lin_vel=2.5).
- **Ticket 044** (slew clip, landed) — unchanged. The slew clip is the *binding* constraint that 046 is calibrating the task around.
- **Ticket 047** (reward retune, immediate follow-up if Slice 2 doesn't fully recover triangulation).
- **Ticket 002** (general reward weight retuning) — superseded by 047 for the specific bbox_center vs triangulation rebalance; 002 stays open for other reward-shape work.
- **Ticket 034** (per-(env, agent) effective-progress sampler) — anti-forgetting mechanism unchanged; only the cylinder dimensions inside which it samples shrink.
- **Initial-states module** — value-only changes; module logic unchanged.
- **Target-controller module** — adds one flag + one 2-line clamp in the orchestrator. Generators untouched.

### Affected files

**Edits (Slice 1)**:
- [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py) — spawn-geometry value changes (3 fields).
- [target_controller/target_controller_cfg.py](../../../target_controller/target_controller_cfg.py) — new `enable_z_motion: bool = True` field.
- [target_controller/target_controller.py](../../../target_controller/target_controller.py) — 2-line clamp inside `_apply_constraints` gated on the flag.
- [experiments/experiment_registry.py](../../../experiments/experiment_registry.py) — register `validation_task_geom_{baseline, treatment}`.

**Edits (Slice 3, conditional cfg-default flip)**:
- Same `iris_ma_env6_test_cfg.py` and `target_controller_cfg.py` files — flip the defaults to the treatment values.

**New**:
- `tests/test_target_z_motion_v1.py` — Slice-1 unit test (target velocity z clamp + spawn geometry).
- `doc/experiments/<date>_ticket046_closer_spawn_2d_target.md` — Slice-2 writeup.

### References

- [iris_ma_env6_test_cfg.py:567](../../../iris_ma_env6_test_cfg.py#L567) — `max_lin_vel` (post-t045 = 5.0).
- [initial_states/initial_states_cfg.py:54](../../../initial_states/initial_states_cfg.py#L54) — `cylinder_diameter_min / max`.
- [initial_states/initial_states_cfg.py:102](../../../initial_states/initial_states_cfg.py#L102) — `target_distance_min / max`.
- [initial_states/initial_states_cfg.py:113](../../../initial_states/initial_states_cfg.py#L113) — `target_height_offset_min / max`.
- [target_controller/target_controller.py:240](../../../target_controller/target_controller.py#L240) — `v_cmd = self._apply_constraints(...)` — single insertion point for z-clamp.
- [t045 ticket](../045-task-difficulty-calibration/ticket.md) — parent decision context.
- [t045 writeup placeholder](../../../experiments/<TBD>_ticket045_task_difficulty_calibration.md) — post-training synthesis (saturation + triangulation result).
- [t043 prev_action_only baseline](../../../experiments/2026-05-26_ticket043_prev_action_lowpass.md) — triangulation reference point (40.1 at 200k).
- [iris_ma6/CLAUDE.md](../../../CLAUDE.md) — test-writing conventions for Slice 1.

**Flow**: Low. Three small file edits + one unit test + one 200k training run. Estimated 2 days: (a) Slice-1 patch + tests (3-4 hours); (b) Slice-2 launch (~24 wall-h); (c) analyze + writeup; (d) conditional Slice-3 cfg-default flip + doc updates.
