## Ticket 045 — Task-difficulty calibration to match PX4-strict slew envelope

**Status**: Proposed
**Created**: 2026-05-28
**Type**: Cfg tuning + A/B validation (no env-side architectural change).
**Target setup**: iris_ma6 training (MAPPO-RNN baseline), branched from `09762f259f` (the t044 slew-clip commit). Sim-to-sim PegasusSimulator + PX4 SITL is the load-bearing transfer case.
**Deliverable**:
1. Experiment-registry-only patch (no cfg-default flips yet — defaults flip only after Slice 1 validates). The treatment experiment's `env_overrides` capture three coordinated changes:
   - Lower `max_lin_vel`: 10 → 5 m/s.
   - Proportionally scale `action_slew_vel_xy`: 0.020 → 0.040 (preserves PX4-strict on the *physical-acceleration* axis).
   - Lower `target_controller.max_speed_end`: 5.0 → 2.5 m/s (matches the agent envelope so the target doesn't outrun pursuers — folded into immediate scope per 2026-05-28 decision).
2. Two Slice-1 experiments (`baseline`: t044 cfg unchanged at 10 m/s; `treatment`: coordinated lowered envelope). 200k × seed=42.
3. One-page experiment writeup in `doc/experiments/`.
4. Cfg-default flip (only after Slice 1 passes acceptance bars).

**What**: The t044 200k A/B (in-flight at 204k as of 2026-05-28) confirmed the slew clip ships an architecturally clean smoothness win (action_delta −69%, total_rms −44% vs t043 prev_action_only baseline, reward parity at −1%) BUT three task-quality metrics regressed past the ticket's acceptance bars:

| Metric | t044 slew_on @ 204k | t043 baseline @ 200k | Δ | Verdict |
|---|---|---|---|---|
| triangulation | 20.6 | 40.1 | **−48.6%** | catastrophic |
| tracking_lost | 5.0% | 1.4% | **+253%** | hard fail |
| collision_fraction | 0.41% | 0.18% | **+122%** | at the 2× safety ceiling |
| pair_valid_rate | 0.815 | 0.855 | −4.8% | at the ±5% edge |

And per-channel slew saturation is **92–99% across ALL 7 channels** — not just velocity (predicted by Slice 0) but also yaw_rate, gimbal, and zoom (which Slice 0 said were well within δ at the un-clipped policy). Interpretation: the policy *adapts* to compensate for velocity-channel starvation by pushing harder on yaw + gimbal to keep the bbox centered. Cross-channel adaptation that the Slice-0 probe (on a slew-untrained policy) couldn't predict.

**Diagnosis**: the issue isn't the slew clip per se — it's that the **velocity envelope (`max_lin_vel = 10 m/s`) is too large for what PX4-strict acceleration (5 m/s²) can deliver in a single policy step**. At δ_xy = 0.020 in action units and `max_lin_vel = 10`, the physical Δvxy cap is 0.2 m/s per 0.04 s step = 5 m/s² (matches `MPC_ACC_HOR_MAX`). To reach the full 10 m/s envelope from rest, the policy needs **50 policy steps = 2 wall-seconds at full saturation**. Episodes are 20 s, so a quarter of every episode is spent just *accelerating to top speed*. There's no headroom for the bearing-change maneuvers that multi-agent triangulation requires.

**Hypothesis**: lower `max_lin_vel` to a value that's reachable in ≤1 wall-second under PX4-strict accel. At `max_lin_vel = 5`, full envelope reached in 25 steps = 1 s; the policy then has 19 s of episode budget for tracking + cooperative maneuvering rather than just acceleration. **This is exactly the user's stated principle: "what we can achieve in reality."** Real Iris quadrotors flown under PX4 position control rarely cruise above 5–7 m/s; the 10 m/s setting was a sim-only agility budget that doesn't transfer.

**Why the δ scaling**: the slew clip is in action units. `action_slew_vel_xy = 0.020` → `Δvxy_physical = action_slew_vel_xy × max_lin_vel = 0.020 × 10 = 0.2 m/s/step = 5 m/s²`. To preserve **PX4-strict on the physical-acceleration axis** (the load-bearing alignment, not the arbitrary action-axis number) when `max_lin_vel` halves, the action-space δ must double: `0.040 × 5 = 0.2 m/s/step = 5 m/s²` ✓. Without this scaling, halving `max_lin_vel` ALSO halves the achievable physical acceleration → the policy is even more constrained, not less.

**Target-speed reduction folded into Slice 1** (2026-05-28 update): at `max_lin_vel = 5` and the original `target_max_speed_end = 5`, agents have *zero* speed headroom over the target — pursuit becomes impossible at curriculum end. Folding the target-speed halving (5 → 2.5) into Slice 1 preserves the 2:1 agent-vs-target ratio that the t043/t040/042 baseline trained against (`10:5 → 5:2.5`). This is a *single coordinated downshift of the task envelope*, not a multi-experiment bisect: agent envelope, slew δ, and target envelope all scale together by the same factor (× 0.5 on velocities, × 2 on action-unit δ to hold physical accel constant). Failure attribution is cleaner — if Slice 1 fails, the task is over-constrained on a non-envelope axis (reward shape or curriculum), not because we mis-bisected.

**Out-of-scope alternatives considered**:
- *Reward retuning* (soften bbox_center, raise action_delta) — defer to ticket 002. Slice 4's job is to make the architectural constraint achievable; reward weights tune the trade-off *within* that constraint.
- *Soften DR aggressiveness* (FOV/mass/gimbal ranges) — independent axis, separate ticket.
- *Loosen slew clip* (raise δ above PX4-strict on the physical axis) — explicitly rejected by the user as breaking the sim-to-real alignment goal.
- *Reduce `target_controller.max_acceleration`* (currently 2.0 m/s²) — the target's own internal accel limit. With peak target speed halving 5 → 2.5, time-to-peak shrinks from 2.5 → 1.25 s; this stays inside the existing curriculum. Leave alone in v1.

### Patch summary

1. **`experiments/experiment_registry.py`** — add two Slice-1 entries:
   - `validation_task_difficulty_baseline`: cfg unchanged at `max_lin_vel = 10`, `action_slew_vel_xy = 0.020`, `target_controller.max_speed_end = 5.0`. This is the t044 treatment — provided for reproducibility, but doesn't need to be re-launched since the t044 in-flight run's TB log IS the baseline data.
   - `validation_task_difficulty_treatment`: cfg overrides to `max_lin_vel = 5`, `action_slew_vel_xy = 0.040`, `target_controller.max_speed_end = 2.5`. The shipping candidate.

2. **No cfg-default flips yet** — per the 2026-05-28 decision, defaults flip only after Slice 1 validates. Reduces rollback friction if the experiment shows unexpected regression on a non-envelope axis.

3. **No env-side changes.** The slew clip, prev-action obs, reset bug fix, and trackers from tickets 043+044 are all preserved as-is. This is a pure experiment-registration ticket; v1 cfg-default flip is a follow-up edit gated on Slice 1 acceptance.

**Coordinated downshift summary** (the three knobs in `treatment`):

| Knob | Before | After | Ratio | Why |
|---|---|---|---|---|
| `max_lin_vel` | 10.0 m/s | 5.0 m/s | ×0.5 | "What we can achieve in reality" — PX4-flown Iris quads cruise ≤7 m/s |
| `action_slew_vel_xy` | 0.020 | 0.040 | ×2.0 | Holds physical PX4 accel constant: `δ × max_lin_vel = 5 m/s² = MPC_ACC_HOR_MAX` |
| `target_controller.max_speed_end` | 5.0 m/s | 2.5 m/s | ×0.5 | Preserves 2:1 agent-target speed ratio (prevents target outrunning agents) |

Untouched (PX4-aligned independent of `max_lin_vel`):
- `action_slew_vel_z = 0.053`, `max_vel_z_up = 3.0`, `max_vel_z_dn = 1.5`
- `action_slew_yaw_rate = 0.30`, `max_yaw_rate = π/4`
- `action_slew_gimbal_*_rate = 0.40`, gimbal native limits
- `action_slew_zoom_rate = 0.20`

`max_lin_vel_min` (curriculum start) left at 3.0 — the curriculum range compresses to [3.0, 5.0] (1.67×) but the mechanism is unchanged.

### Slices

**Slice 0 — t044 run termination.**
- The in-flight t044 run is at step 204k / 400k. Reward has stabilized at ~3219 (−1% of baseline), saturation at 92–99% everywhere, triangulation stuck at −49%. Another 196k steps will not change the architectural verdict.
- **Action**: kill the run. Save the 204k checkpoint as the t044 baseline reference for the Slice-1 comparison.

**Slice 1 — coordinated envelope downshift A/B.**
- Three coordinated config overrides (`max_lin_vel`, `action_slew_vel_xy`, `target_controller.max_speed_end`). One 200k × seed=42 run on the treatment.
- Comparison anchor: t044 @ 200k (already done; no fresh baseline run needed).
- Acceptance bar (read at step 200k):
  - `triangulation` recovered to ≥ 80% of t043 baseline value (≥ 32, currently 20.6 in t044).
  - `tracking_lost_fraction` ≤ 2× t043 baseline (≤ 0.028, currently 0.050 in t044).
  - `collision_fraction` ≤ 2× t043 baseline (≤ 0.0036, currently 0.0041 in t044 → borderline; expected to drop with smaller velocity envelope and the same slew bandwidth budget).
  - `slew_sat_*` per-channel ≤ 0.70 on at least 3 channels (currently all 7 saturate at 0.92–0.99). This is the diagnostic that the velocity-envelope reduction relieved the cross-channel saturation pressure.
  - `total_rms` smoothness remains ≥ 30% better than t043 baseline (i.e., the t044 smoothness win is preserved — we're calibrating task difficulty, not unwinding the slew clip).
  - Reward within ±5% of t043 baseline (i.e., not worse than t044's −1%).
- Failure modes:
  - Triangulation doesn't recover → cooperative-coordination signal is reward-side, not envelope-side. Defer to ticket 002 reward retuning OR add Slice 2 (target-speed reduction).
  - Saturation drops on velocity but stays high on yaw/gimbal → cross-channel saturation was *not* compensation; it's intrinsic to the policy under any agile bbox-tracking task. Likely requires reward retuning.
  - All saturation drops but reward also drops below 95% baseline → the velocity envelope reduction crippled the agents' ability to keep up with the target. Bisect (max_lin_vel ∈ {5, 7, 8}) in Slice 2 alt.

**Slice 2 — cfg-default flip (conditional on Slice 1 success).**
- Land the three coordinated overrides directly into `IrisMA6TestEnvCfg` as the new shipping defaults.
- Update the t043+t044 wrap-up note in `feature_list.json` and `progress.txt`.
- No re-validation needed (Slice 1 is the validation).

**Slice 3 — bbox_center reward soften (last resort, spawns ticket 002 follow-up).**
- Out of 045 scope. If Slice 1 doesn't recover triangulation even with the coordinated envelope downshift, the issue is reward-shaped (bbox_center demand exceeds what the velocity envelope can deliver). Reward retuning is ticket 002's domain.

### Method (training-time validation)

1. **Kill t044.** Save the 204k checkpoint as the Slice-1 baseline comparison anchor.
2. **Register Slice-1 experiments** in `experiments/experiment_registry.py`. No cfg-default flips.
3. **Launch Slice-1 treatment** via `++experiment=validation_task_difficulty_treatment` (or whatever the runner convention is). 200k × seed=42.
4. **Decision rule (read at 200k)**:
   - All acceptance bars passed → trigger Slice 2 (cfg-default flip). Update progress + feature_list. Land the three knobs into `IrisMA6TestEnvCfg`.
   - Mixed results (triangulation partial, saturation partially drops) → diagnose which axis still binds (target_proximity? curriculum_dynamics? specific channel saturation?) and spawn a tighter follow-up ticket.
   - Catastrophic (worse than t044) → keep cfg defaults at 10 m/s, investigate. Most likely cause given the coordinated downshift: reward-shape problem (bbox_center demand doesn't scale with envelope), defer to ticket 002.

### Acceptance criteria

| Criterion | Result |
|---|---|
| Cfg patch landed: `max_lin_vel = 5.0`, `action_slew_vel_xy = 0.040` | pending |
| Slice-1 treatment run completed at 200k | pending |
| `triangulation` ≥ 80% of t043 baseline (≥ 32, t044 at 20.6) | tensorboard |
| `tracking_lost_fraction` ≤ 2× t043 baseline (≤ 0.028) | tensorboard |
| `collision_fraction` ≤ 2× t043 baseline | tensorboard |
| `slew_sat_*` ≤ 0.70 on at least 3 channels | tensorboard |
| Smoothness win preserved (`total_rms` ≥ 30% smaller than t043 baseline) | tensorboard |
| Reward within ±5% of t043 baseline | tensorboard |
| `doc/experiments/<date>_ticket045_task_difficulty_calibration.md` writeup landed | written |

### Scope boundary

- **DO**: lower `max_lin_vel` to 5 m/s; scale `action_slew_vel_xy` proportionally to preserve PX4-strict physical acceleration alignment.
- **DO**: leave other PX4-aligned δ values, the slew clip, prev-action obs, t043/044 trackers all unchanged.
- **DO**: ship the new defaults if Slice 1 passes; document the reasoning ("what we can achieve in reality").
- **DO NOT**: change the slew clip itself (δ values, flag default, code path). The slew clip is correct; the task envelope was wrong.
- **DO NOT**: retune reward weights. Defer to ticket 002. (Slice 4 of *this* ticket is explicitly out of scope.)
- **DO NOT**: change DR aggressiveness, curriculum schedules, or CBF settings. Independent axes.
- **DO NOT**: revert ticket 043 (prev_action_obs ON) or ticket 044 (slew clip + reset bug fix + new metrics).
- **DO NOT**: launch the Slice-1 *baseline* — the t044 204k log already serves as the comparison anchor. Only `treatment` needs a fresh run.

### Risk

Low.

1. **Target outruns agents** — mitigated by design (target speed halved alongside agent envelope, preserving the 2:1 ratio that the baseline trained against). Residual risk: if target *acceleration* (currently 2.0 m/s²) is the bottleneck rather than peak speed, the same triangulation failure mode resurfaces in a different guise. Diagnostic: `Info / Episode_Reward/drone_0_target_proximity` should land near baseline (-1.5 ish). If it spikes more negative, the target is escaping by accel, not by peak speed → Slice 3 (separate ticket) on target_max_acceleration.

2. **PX4 alignment shifts to *cruise* (3 m/s²) not *max* (5 m/s²)** if δ_xy stays at 0.040 but `max_lin_vel` drops further in some future iteration. Documented; the v1 patch preserves the 5 m/s² figure.

3. **Curriculum effective range compresses**. `max_lin_vel_min = 3.0` → `max_lin_vel = 5.0` is a 1.7× range (was 3.3× at 3.0 → 10.0). The agent-velocity curriculum's anti-forgetting Uniform(0, p) sampling will produce per-env `_max_lin_vel` values uniformly in [3.0, 5.0] at end-of-curriculum, instead of [3.0, 10.0]. Compresses the experimental diversity, but the curriculum mechanism still works.

4. **Existing checkpoints become incompatible.** Any pre-045 checkpoint trained at `max_lin_vel = 10` will produce actions calibrated to a 10 m/s envelope; loading into the new cfg would clip vel commands by 2×. Forward-only feature; document.

5. **Slice-0 saturation predictions become stale.** The Slice-0 probe was run against the t043 prev_action_only policy at `max_lin_vel = 10`. The new `max_lin_vel = 5` regime will produce a different |Δa| distribution. If Slice 1 needs re-tuning of δ (Case B or C from the t044 ticket framework), a new Slice-0-style probe at `max_lin_vel = 5` would be the principled re-sizing step. Out of v1 scope.

### Coupling

- **Ticket 044** (per-channel slew clip, in-flight) — 045 is the calibration follow-up. The slew clip itself ships unchanged; only the velocity envelope it operates within changes.
- **Ticket 043** (prev_action_obs, landed) — unchanged. The prev-action obs continues to carry the applied filtered command in physical units; with the new `max_lin_vel = 5`, the channel's dynamic range halves on vel dims (matches deployment-realistic MAVROS publish rates).
- **Ticket 042** (per-channel EKF lag, landed) — orthogonal. Lag is on the obs path; envelope change is on the action path.
- **Ticket 002** (reward weight retuning) — possible follow-up if Slice 1+2 don't fully recover triangulation. Document the trigger.
- **Curriculum module** — `max_lin_vel_min` and `max_lin_vel` form the curriculum range. The compression (3.3× → 1.7×) is documented but doesn't require curriculum code changes.

### Affected files

**Edits (Slice 1, this ticket)**:
- [experiments/experiment_registry.py](../../../experiments/experiment_registry.py) — add `validation_task_difficulty_{baseline, treatment}`.

**Edits (Slice 2, conditional cfg-default flip)**:
- [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py) — `max_lin_vel: 10.0 → 5.0`, `action_slew_vel_xy: 0.020 → 0.040`, `target_controller.max_speed_end: 5.0 → 2.5` (the latter via setting the default `TargetControllerCfg()` differently).

**New**:
- `doc/experiments/<date>_ticket045_task_difficulty_calibration.md` — Slice-1 writeup, populated post-run.

### References

- [iris_ma_env6_test_cfg.py:567](../../../iris_ma_env6_test_cfg.py#L567) — `max_lin_vel = 10.0`.
- [iris_ma_env6_test_cfg.py:570](../../../iris_ma_env6_test_cfg.py#L570) — `max_lin_vel_min = 3.0`.
- [target_controller_cfg.py:131](../../../target_controller/target_controller_cfg.py#L131) — `max_speed_end = 5.0` (target peak; Slice-2 lever).
- [PX4-Autopilot/.../mc_pos_control_params.c:614](../../../../../../../../PX4-Autopilot/src/modules/mc_pos_control/mc_pos_control_params.c#L614) — `MPC_ACC_HOR = 3.0 m/s²` (cruise); `MPC_ACC_HOR_MAX = 5.0 m/s²` (max).
- [t044 ticket](../044-action-slew-rate-clip/ticket.md) — parent ticket; v1 Slice-0 results table.
- [t043 writeup](../../../experiments/2026-05-26_ticket043_prev_action_lowpass.md) — the prev_action_only comparison anchor.

**Flow**: Low. Two-line cfg change; one 200k training run; one writeup. Estimated 1.5 days: (a) Slice-0 kill + cfg patch (10 min); (b) Slice-1 launch (~24 wall-h); (c) analyze + writeup; (d) optional Slice 2.
