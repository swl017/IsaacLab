## Ticket 047 — Reward retune: visibility + smoothness redirect (task-difficulty reduction, Phase 2)

**Status**: In Progress (Slice 0 — empirical baseline landed 2026-06-02 from t046 partial run)
**Created**: 2026-06-02
**Type**: Reward-weight cfg patch + warm-start fast-screen + cold-start validation.
**Target setup**: iris_ma6 training, post-046 (closer spawn + 2D target motion + post-045 envelope + post-044 slew clip). Pegasus SITL + PX4 SITL transfer remains the load-bearing case.
**Depends on**: Ticket 046's partial run. The t046 treatment run (2026-06-02_00-29-06_…_ticket046_closer_spawn_2d_target) died at step ~150k due to a Claude-background-wrapper SIGHUP, but the partial TB through 144k is sufficient to fix the empirical baseline.
**Deliverable**:
1. Cfg patch retuning four reward scales (see §"Proposed weights").
2. Fast-screen Slice 1 (warm-start +30k × 4 candidates) before any cold-start full run.
3. Cold-start Slice 2 (200-400k × seed=42) on the surviving candidate.
4. One-page experiment writeup in `doc/experiments/`.
5. Cfg-default flips (gated on Slice-2 acceptance).

---

### Slice 0 — Empirical baseline (DONE)

#### Per-component reward contribution at t046 step 144k

Per-agent episode-summed weighted contributions (the env logs *already-weighted* values into `Info / Episode_Reward/drone_X_<term>`):

| Component | scale | drone_0 | drone_1 | mean | per-step (mean) |
|---|---:|---:|---:|---:|---:|
| `bbox_center` | ×60 | 31.7 | 18.1 | **24.9** | +0.10 |
| `bbox_size` | ×60 | 41.4 | 39.7 | **40.5** | +0.16 |
| `triangulation` | ×5 | 27.1 | 27.1 | **27.1** | +0.11 |
| `action_delta` | ×−12 | −4.9 | −4.5 | −4.7 | −0.019 |
| `action_sum` | ×−8 | −7.6 | −10.3 | −9.0 | −0.036 |
| `target_proximity` | ×−50 | −4.9 | −5.2 | −5.0 | −0.020 |
| `collision` | ×−100 | −0.51 | −0.51 | −0.51 | −0.002 |
| `cbf_penalty` | (built-in) | −0.052 | −0.052 | −0.05 | −0.0002 |
| `altitude` | ×−100 | −1e-5 | −5e-4 | ~0 | ~0 |

Total reward mean = **2927** (Reward / Total reward (mean) @ 144k). Per-step mean = 6.19.

**Per-agent ratio** of the three "task" reward heads: bbox_center : bbox_size : triangulation ≈ **25 : 40 : 27** — *not* the 10:1 imbalance the ticket's prior text assumed. Triangulation is **already near-parity** with bbox_center on a per-agent basis. The dominant head is **bbox_size** (40), not bbox_center.

The raw signals (before weighting) are very different in scale:
- bbox_center raw: ≈ 0.002 per step (small visibility signal, multiplied 60× to dominate)
- bbox_size raw: ≈ 0.0027 per step
- triangulation raw: ≈ 0.022 per step (10× larger raw; needs only 5× weight to match)

#### Action smoothness saturation (the *load-bearing* failure mode)

| channel | drone_0 | drone_1 |
|---|---:|---:|
| `slew_sat_vx` | **0.954** | **0.958** |
| `slew_sat_vy` | **0.961** | **0.966** |
| `slew_sat_vz` | **0.991** | **0.991** |
| `slew_sat_yaw_rate` | **0.968** | **0.964** |
| `slew_sat_gim_yaw` | 0.868 | 0.677 |
| `slew_sat_gim_pitch` | 0.921 | 0.802 |
| `slew_sat_zoom` | **0.978** | **0.977** |

All four velocity channels at ≥ 95% saturation. The policy is hammering the slew clip on every step. `cmd_vel_delta` ≈ 0.24 on both drones (the per-step physical command magnitude change, near the slew limit).

#### Detection / visibility

- `Detection/pair_valid_rate` = **0.744** (target ≥ 0.95). Closer spawn improved visibility but not enough.
- `Detection/all_invalid_rate` = 0.033 (≈3% of steps with no drone seeing the target).

#### Training health

- `Learning / KL` = 0.036 (healthy; KL-adaptive bands intact).
- `Learning / LR` = 3e-4 (at max; LR-scheduler hasn't reduced).
- `Loss / Value loss` = 0.028 (drone_0/1 same).
- `Loss / Policy loss` = 0.0021.
- `Loss / Entropy loss` = −0.143.
- `Policy / Standard deviation` = **2.014** on both drones (uncoverged — 7-8× a typical converged value of ~0.25).

**Stage of training:** ~37.6% of 400k. Per the agent-velocity curriculum (0-40k cold-start phase, target-motion 40-120k, observability 120k+), step 144k is just past the observability-onset phase. High `Policy / std` is partly the curriculum sliding the reward landscape under the policy.

---

### Re-stated problem (informed by Slice 0)

The original framing ("bbox vs triangulation imbalance, retune 60:5 → 30:25") was wrong on the empirical numbers. The actual t046 picture:

1. **Visibility (`pair_valid_rate` = 0.74) is the load-bearing problem.** A deployable policy needs ≥ 0.9 here. Single-agent tracking is *under*-prioritized in the slew-bandwidth budget, not over-prioritized.
2. **Every velocity channel saturates the slew clip** at 95-99%. The policy has no fine-grained control on any axis — it picks a direction and rides the slew bound. Reward-side smoothness penalty (`action_delta_penalty_scale = −12`) contributes only ~4% of total reward magnitude (∼5 per agent / ∼82 total) — too weak to push back on the slew-clip's pull.
3. **bbox_size is the largest task-head contribution** (40 per agent vs bbox_center 25, tri 27). Under the closer t046 spawn, the bbox is reliably *large enough* anyway (target near-permanently in frame whenever visible), so bbox_size carries less marginal information than its weight implies — it's eating slew budget for low marginal task value.
4. **Triangulation is competitive with bbox_center** per-agent already (27 vs 25). Modestly raising it to lead (~35-40 contribution) makes sense; a 5×-style hike would distort the curriculum.

### Proposed weight set (grounded in Slice 0 contributions)

| Reward scale | Now | Proposed | Δ | Rationale |
|---|---:|---:|---:|---|
| `bbox_center_reward_scale` | **60** | **60** | × 1.0 | Hold. bbox_center is the visibility signal; halving it (original ticket plan) would worsen `pair_valid_rate` further. |
| `bbox_size_reward_scale` | **60** | **30** | × 0.5 | Halve. Under closer spawn the bbox is already large when visible; this head was the *largest* contributor (40) but carries low marginal info. Frees slew budget. |
| `triangulation_reward_scale` | **5** | **8** | × 1.6 | Modest lift. Current contribution 27; target ~43 (lead position) without distorting the curriculum. Avoids the 5× hike's risk of over-correction. |
| `action_delta_penalty_scale` | **−12** | **−24** | × 2.0 | Double. Reward-side smoothness pressure that can actually compete with the slew-clip's saturation pull (contribution moves from ~−5 to ~−10). The slew clip is architectural; this is the reward-side complement. |

**Target post-retune contributions** (rough projection assuming the policy reaches similar raw signal levels):
- bbox_center: 25 (held)
- bbox_size: 20 (halved from 40)
- triangulation: 43 (1.6× lifted from 27)
- action_delta: −10 (doubled from −5)
- Total task heads: 25 + 20 + 43 = 88, **triangulation leads**.
- Smoothness pressure: −10 = ~11% of total task reward (vs current ~6%).

**Why halve bbox_size, not bbox_center**: bbox_center is what keeps the target in frame; halving it under the visibility-bottleneck regime is the wrong direction. bbox_size is partially redundant with bbox_center (a centered target is usually a decent-sized target), and it's the head with the *least* marginal info under the closer t046 spawn. Lowest-regret cut.

**Why 1.6× tri (not 5×)**: original ticket's 5× was based on the false 10:1 imbalance. Empirically, tri only needs to lead by ~50% to be the dominant task head. 1.6× gets there without the value-loss spike and curriculum disruption a 5× would cause.

**Why double action_delta**: the smoothness penalty contributes only ~4% of total reward magnitude currently. If we want it to be a *competing* pressure against the slew-clip pull (which is binding at 95%+), it needs to be material — doubling moves it to ~8%, still not dominant but actually noticeable to the optimizer.

**What's NOT changing**: `collision_penalty_scale = −100`, `target_proximity_penalty_scale = −50`, `altitude_penalty_scale = −100`, `action_sum_penalty_scale = −8`, `cbf` penalty scales, `est_error_*` reward scales. All safety- and physics-side terms stay fixed so the A/B is clean.

### Patch summary

1. **`iris_ma_env6_test_cfg.py`** — four reward-scale value changes:
   ```python
   # Before                                                After
   bbox_center_reward_scale: float = 60.0                → 60.0   (no change)
   bbox_size_reward_scale:   float = 60.0                → 30.0
   triangulation_reward_scale: float = 5.0               → 8.0
   action_delta_penalty_scale: float = -12.0             → -24.0
   ```

2. **`experiments/experiment_registry.py`** — register fast-screen candidates as separate experiments (see §"Candidate set" below).

3. **No env-side architectural changes.** Slew clip, prev-action obs, spawn geometry, target z-clamp, envelope downshift — all preserved.

### Slices

#### Slice 0 — DONE
Empirical baseline pulled from t046 partial run (TB events at step 4k → 144k). All numbers above.

#### Slice 1 — Fast-screen via warm-start (NEW)

**Why warm-start, not cold-start**: a 200-400k cold-start costs 1.5 days × candidate. The t046 partial run gives us `agent_120000.pt`, a policy that already absorbed the t046 cfg (closer spawn + 2D target) for 120k steps and sits at curriculum progress ≈ 30%. Loading this checkpoint into each candidate's reward weights and running +30k more lets the policy adapt to the *new* reward gradient (which is the actual A/B variable) without re-paying the 0-30% curriculum cost.

**Why +30k not +20k**: the policy needs ~15-20k to settle on the new gradient (smoothness metrics like `cmd_vel_delta` and `slew_sat_*` move within ~5k once the optimizer commits to the new advantage estimates), but `pair_valid_rate` and `triangulation` need another ~10-15k to express. +30k = enough headroom for both axes; ~3 wall-h at ~3 it/s.

**Risk**: warm-start can hide cold-start instability — a candidate whose reward shape is too steep at early curriculum may look fine when nudged from a t046-trained policy but fail from scratch. **Mitigation**: any candidate that *clearly wins* the warm-start screen goes to a full cold-start Slice 2 for final validation. Multiple candidates can be screened in parallel (one GPU at a time, but each is only ~3h).

**Candidate set** (4 × ~3h ≈ 12 wall-h total):

| Candidate | bbox_c | bbox_s | tri | action_delta | Hypothesis |
|---|---:|---:|---:|---:|---|
| **A** (proposed default) | 60 | 30 | 8 | −24 | Hold bbox_c, halve bbox_s, modest tri lift, double smoothness. Lowest-regret retune. |
| **B** (tri-led) | 60 | 30 | 15 | −24 | Same as A but stronger tri push (×3 not ×1.6). Tests whether the modest tri lift in A is too timid. |
| **C** (smoothness-only) | 60 | 60 | 5 | −36 | No reward retune; **triple** action_delta penalty. Isolates whether smoothness-side pressure alone can dent slew saturation. |
| **D** (visibility-first) | 90 | 30 | 8 | −24 | Boost bbox_c 1.5× to push `pair_valid_rate` ≥ 0.9. Tests "fix visibility first; triangulation follows once the target stays in frame." |

Each candidate launches as `validation_reward_retune_<X>_warmstart` in the registry, with `--checkpoint logs/.../ticket046_closer_spawn_2d_target/checkpoints/agent_120000.pt`. The trainer's `--checkpoint` arg + `_log_dir_from_checkpoint` plumbing (already in `run_experiment.py`) puts the warm-start log in a new sub-dir so TBs don't collide.

**Screen acceptance bars at +20k → +30k from warm-start** (so absolute step = 140k → 150k):
- `slew_sat_drone_X_<vx,vy,vz,yaw>` drops by ≥ 5 percentage points on at least 2 channels (e.g., 0.96 → 0.91).
- `pair_valid_rate` ≥ 0.80 (relaxed from 0.95 for a +30k screen) or unchanged from baseline.
- `Episode_Reward/drone_X_triangulation` doesn't *collapse* (stays ≥ 20 per agent).
- `Policy / Standard deviation` flat or trending down (sanity check for healthy optimization).
- No NaN, no value_loss explosion, KL stays in [0.02, 0.05].

**Decision**:
- 1 clear winner → Slice 2 (full cold-start) with that candidate.
- 2-3 close → run a second screen round narrowing the weight magnitudes, OR commit to 2 cold-starts.
- No movement on any → reward retune isn't the lever; escalate to Ticket 048 (loosen slew).

#### Slice 2 — Cold-start full validation (surviving candidate only)

- 1 config, seed=42, **400k** steps (full curriculum, comparable to t045/t046 baselines).
- Launch under tmux: `tmux new-session -d -s t047_<candidate> '<launch command>'`.
- Read metrics at 200k (acceptance gate) and 400k (final).
- Acceptance bar at 400k:
  - `pair_valid_rate ≥ 0.90` (deployment-quality visibility).
  - `triangulation ≥ 40` (t043 baseline parity or better).
  - At least 3 velocity channels with `slew_sat ≤ 0.85` (visible slew-budget relief).
  - `collision_fraction ≤ 2× t043 baseline` (load-bearing safety).
  - Total reward ≥ 0.9 × t046 baseline (no regression).

#### Slice 3 — cfg-default flip (conditional on Slice 2 success)

Land the winning candidate's four weight values into `IrisMA6TestEnvCfg` defaults. Update `feature_list.json`, `progress.txt`, reward-related comments in `iris_ma6_env_spec.md`. Close out Ticket 002 (general reward weight retuning) as superseded for this axis.

### Acceptance criteria

| Criterion | Result |
|---|---|
| Slice 0 empirical baseline pulled | done (2026-06-02) |
| Stale slew docstring on `action_slew_vel_xy` fixed | done (2026-06-02) |
| t047 cfg defaults updated to candidate A (`bbox_size=30`, `tri=8`, `ad=−24`) | pending |
| 4 warm-start fast-screen experiments registered (`validation_reward_retune_{A,B,C,D}_warmstart`) | pending |
| Fast-screen Slice 1: smoothness drop ≥ 5pp on ≥ 2 channels for winning candidate | pending |
| Cold-start Slice 2: `pair_valid_rate ≥ 0.90` at 400k | pending |
| Cold-start Slice 2: `triangulation ≥ 40` at 400k | pending |
| Cold-start Slice 2: `slew_sat ≤ 0.85` on ≥ 3 velocity channels | pending |
| Writeup at `doc/experiments/<date>_ticket047_reward_retune.md` | pending |

### Scope boundary

- **DO**: change the four reward scales listed. Run the warm-start fast-screen. Run one cold-start full validation. Flip defaults if Slice 2 passes.
- **DO NOT**: change any safety/proximity/altitude scale, or the action_sum scale. Those are calibrated against orthogonal failure modes; entangling them with this A/B muddies attribution.
- **DO NOT**: change spawn geometry, target z-motion, or envelope (those are 046/045's domain).
- **DO NOT**: loosen the slew clip. Deferred to Ticket 048, gated on 047's outcome.
- **DO NOT**: change curriculum, DR, EKF lag, network width, or anything else.
- **DO NOT**: re-launch any 400k cold-start as a Claude background task. Long runs go in tmux (per the 2026-06-02 lesson — see `~/.claude/projects/.../memory/feedback_long_training_tmux.md`).

### Risk

Low to medium.

1. **Warm-start hides cold-start failure**: a candidate that adapts smoothly from `agent_120000.pt` may still diverge from scratch under a steep new reward gradient. Mitigation: Slice 2 cold-start validation is mandatory before any cfg-default flip.

2. **Halving bbox_size collapses bbox_center too**: the two heads are correlated (a centered target tends to be a sized target). Halving size could induce a smaller bbox overall, dropping pair_valid further. Mitigation: warm-start screen surfaces this fast — if pair_valid drops below 0.7 on any candidate, that candidate is dead. Fallback: revert bbox_size to 45 (×0.75 instead of ×0.5).

3. **Doubled action_delta penalty over-dampens learning**: pushing the policy toward zero action_delta could starve exploration and freeze the policy at low-reward initial conditions. Mitigation: warm-start should show value_loss spike + entropy collapse if this happens. Fallback: 1.5× (−18) instead of 2× (−24).

4. **Tri 5 → 8 is too timid**: triangulation moves to 35 but plateaus there because the policy still can't free slew bandwidth (saturation unchanged). Diagnostic: if slew_sat doesn't drop with candidate A but does with C (smoothness-only), the lever is action_delta, not tri weight — escalate that.

5. **No reward retune helps**: all 4 candidates fail to dent saturation in the warm-start screen → confirms the slew clip itself is the binding constraint and reward shape has no further leverage. Escalates Ticket 048 (loosen slew) with high confidence.

6. **bbox_center asymmetry (drone_0 = 32, drone_1 = 18) persists**: a non-reward-side issue (e.g., designated-observer asymmetry from `initial_states`) that t047's reward changes can't fix. Note for follow-up; not blocking this ticket.

### Coupling

- **Ticket 046** (closer spawn + 2D target, landed Slice 1, Slice 2 partial): t047 builds on t046's cfg state. 046's Slice 2 acceptance was *not* fully met (pair_valid 0.74 < 0.95, slew_sat unchanged), but the partial readout is enough to redirect t047's plan as captured above.
- **Ticket 044** (slew clip): unchanged. The binding constraint t047 is reward-shaping around.
- **Ticket 045** (envelope downshift): preserved.
- **Ticket 048** (loosen slew clip, deferred): t047's failure-mode #5 is the trigger.
- **Ticket 002** (general reward weight retuning): absorbed for the visibility/smoothness axis once 047 lands. The rest of 002's scope stays open.
- **Ticket 037** (task_reward_level / asymmetric critic): preserved; reward *structure* unchanged, only scales.

### Affected files

**Edits (Slice 1)**:
- [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py) — four reward-scale value changes (no new fields, no flags).
- [experiments/experiment_registry.py](../../../experiments/experiment_registry.py) — register `validation_reward_retune_{A,B,C,D}_warmstart`.

**Edits (Slice 3, conditional cfg-default flip)**:
- Same `iris_ma_env6_test_cfg.py` — keep the four scales at the winning candidate's values.

**New**:
- `doc/experiments/<date>_ticket047_reward_retune.md` — Slice-1 + Slice-2 writeup.

### References

- [iris_ma_env6_test_cfg.py:816](../../../iris_ma_env6_test_cfg.py#L816) — `triangulation_reward_scale = 5.0` (current).
- [iris_ma_env6_test_cfg.py:836](../../../iris_ma_env6_test_cfg.py#L836) — `bbox_center_reward_scale = 60.0` (current).
- [iris_ma_env6_test_cfg.py:839](../../../iris_ma_env6_test_cfg.py#L839) — `bbox_size_reward_scale = 60.0` (current).
- [iris_ma_env6_test_cfg.py:833](../../../iris_ma_env6_test_cfg.py#L833) — `action_delta_penalty_scale = -12.0` (current).
- [Ticket 046](../046-closer-spawn-and-2d-target-motion/ticket.md) — partial-run baseline data.
- [Ticket 044](../044-action-slew-rate-clip/ticket.md) — original slew clip rationale.
- [t046 partial-run TB](../../../../../../../logs/skrl/iris_ma6/2026-06-02_00-29-06_mappo_rnn_torch_ticket046_closer_spawn_2d_target/) — empirical baseline source.
- [feedback_long_training_tmux.md](~/.claude/projects/-home-usrg-IsaacPX4-IsaacLab/memory/feedback_long_training_tmux.md) — launch protocol for Slice 2.

**Flow**: Medium. Four cfg changes + 4 warm-start screens (~12 wall-h) + 1 cold-start full validation (~24-36 wall-h). Estimated 2-3 days end-to-end.
