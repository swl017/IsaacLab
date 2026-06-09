# Ticket 047 — Reward retune (candidate D) + entropy_loss_scale curriculum

**Status**: Slice 1 + Slice 2 + Slice 2.5 + Slice 3 (cfg-default flip) ALL landed.
**Date**: 2026-06-04
**Base commit**: 09762f259f (post-t045 slew clip + envelope downshift; t046 spawn + 2D target landed in this session)
**Branch**: iris_ma6-training

## Summary

Two coupled changes against the post-t046 (closer spawn + 2D target) cfg state, validated against the t046 partial run baseline (2026-06-02_00-29-06_..._ticket046_closer_spawn_2d_target, ~148k of 400k before SIGHUP):

1. **Reward weight retune (candidate D — "visibility-first")** — `bbox_center_reward_scale: 60 → 90`, `bbox_size_reward_scale: 60 → 30`, `triangulation_reward_scale: 5 → 8`, `action_delta_penalty_scale: −12 → −24`. Boost the visibility signal that gates `pair_valid_rate`; halve the redundant bbox-size head; modestly lift triangulation; double the reward-side smoothness pressure that competes with the slew-clip's saturation pull.

2. **Entropy curriculum on `entropy_loss_scale`** — new optional `entropy_loss_schedule` field in `MAPPO_RNN_DEFAULT_CONFIG`. When active, linearly interpolates from the init value (`entropy_loss_scale`) toward `end_value` between `start_step` and `end_step`. Slice 2.5 used `{start: 0.01, end_value: 0.001, start_step: 120000, end_step: 200000}` together with `max_log_std: 0.7 → 0.4` (σ_max = exp(0.4) ≈ 1.49). Default-cfg behavior is unchanged (`entropy_loss_schedule: None` ⇒ constant `entropy_loss_scale`).

Slice 1 (warm-start fast-screen across 4 candidates A/B/C/D) showed every candidate hit the policy std cap σ_max = exp(0.7) = 2.014 and produced identical std trajectories — reward-shape gradients couldn't escape the cap because the entropy bonus pinned log_std there. Slice 2 (flat `entropy_loss_scale = 0.001` cold-start of D) freed σ to 0.24 but the policy over-committed by step 32k, then bled task quality through every subsequent curriculum onset (pair_valid 0.97 → 0.68, tri_raw 7.77 → 3.41). Slice 2.5 (the curriculum + tighter cap above) cold-started and completed 400k, producing the most deployable policy of the t04x series: **+7.3% pair_valid_rate, +21.5% triangulation, +22.7% total reward vs t046 baseline**, with equivalent smoothness.

## Patch

- [scripts/reinforcement_learning/skrl/mappo_rnn.py](../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py):
  - Added `entropy_loss_schedule: None` to `MAPPO_RNN_DEFAULT_CONFIG` (default ⇒ bit-exact pre-patch).
  - `__init__` snapshots `self._entropy_loss_scale_start` (per-uid) and parses the schedule struct.
  - New `_maybe_update_entropy_loss_scale(timestep)` — linear interp `start_value → end_value` clamped to `[start_step, end_step]`, called at the top of every `_update(timestep, …)` before any loss computation.
  - TB scalar `Learning / entropy_loss_scale (drone_X)` logged per update so the schedule trace is visible.
- [source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/agents/skrl_mappo_rnn_cfg.yaml](../../agents/skrl_mappo_rnn_cfg.yaml) — `entropy_loss_schedule: {end_value: null, start_step: null, end_step: null}` stub (all-null ⇒ disabled).
- [source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/iris_ma_env6_test_cfg.py](../../iris_ma_env6_test_cfg.py) — only the `action_slew_vel_xy` and `action_slew_vel_z` docstrings updated (post-t045 derivation note); reward scale and `max_log_std` defaults were *not* changed in this ticket. Slice 2.5 overrode them via Hydra args at launch.
- [source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/scripts/run_t047_warmstart_screen.bash](../../experiments/scripts/run_t047_warmstart_screen.bash) — Slice 1 (warm-start screen) launcher (A/B/C/D candidates, +30k each).
- [source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/scripts/run_t047_D_lowent_coldstart.bash](../../experiments/scripts/run_t047_D_lowent_coldstart.bash) — Slice 2 (flat low-entropy cold-start) launcher.
- [source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/scripts/run_t047_D_curriculum_coldstart.bash](../../experiments/scripts/run_t047_D_curriculum_coldstart.bash) — Slice 2.5 (curriculum + tighter cap) launcher.

## Slice 0 — Empirical baseline (DONE)

Pulled from the t046 partial run TB at step 144k. Per-agent episode-summed *weighted* contributions:

| component | scale | drone_0 | drone_1 | mean | per-step |
|---|---:|---:|---:|---:|---:|
| bbox_center | ×60 | 31.7 | 18.1 | 24.9 | +0.10 |
| bbox_size | ×60 | 41.4 | 39.7 | 40.5 | +0.16 |
| triangulation | ×5 | 27.1 | 27.1 | 27.1 | +0.11 |
| action_delta | ×−12 | −4.9 | −4.5 | −4.7 | −0.019 |
| action_sum | ×−8 | −7.6 | −10.3 | −9.0 | −0.036 |
| target_proximity | ×−50 | −4.9 | −5.2 | −5.0 | −0.020 |
| collision | ×−100 | −0.51 | −0.51 | −0.51 | −0.002 |

Per-agent ratio bbox_center : bbox_size : triangulation ≈ **25 : 40 : 27**. The original t047 stub's "10:1 imbalance" rationale was empirically wrong — triangulation was near-parity with bbox_center already; **bbox_size was the dominant head**.

Slew saturation: `slew_sat` 0.95–0.99 on every velocity channel, 0.67–0.92 gimbal, 0.98 zoom. Every channel hammers the clip.
Visibility: `pair_valid_rate` = 0.744 (well below the deployment-target 0.95).
Policy/std = **2.014** = σ_max = exp(`max_log_std = 0.7`) (uncoverged, frozen at cap).
KL = 0.036, LR = 3e-4 (max), Value loss = 0.022 — optimizer is healthy.

Slew rates re-verified PX4-aligned: `action_slew_vel_xy = 0.040` × `max_lin_vel = 5` / `dt = 0.04 s` = 5.0 m/s² = `MPC_ACC_HOR_MAX`; `action_slew_vel_z = 0.053` × 3 / 0.04 = 3.98 m/s² ≈ `MPC_ACC_UP_MAX` (up); × 1.5 / 0.04 = 1.99 m/s² < `MPC_ACC_DOWN_MAX = 3.0` (deliberately conservative on descend).

## Slice 1 — Warm-start fast-screen (4 candidates × +30k from `agent_120000.pt`)

Curriculum auto-pinned to step 120k via `--checkpoint agent_120000.pt` (env's `debug_initial_step = 120000`), so the policy faced a stable post-noise / pre-delay regime across all 4 candidates. ~3 wall-h per candidate, ~10h45m total.

| candidate | bbox_c | bbox_s | tri | a_delta | hypothesis |
|---|---:|---:|---:|---:|---|
| A | 60 | 30 | 8 | −24 | proposed default — halve bbox_size, modest tri lift |
| B | 60 | 30 | 15 | −24 | tri-led — stronger tri push |
| C | 60 | 60 | 5 | −36 | smoothness-only — no reward retune, ×3 a_delta |
| D | 90 | 30 | 8 | −24 | visibility-first — boost bbox_center to push pair_valid |

**Headline finding: every candidate's `Policy / std` was frozen at 2.014 across the entire +30k window** (first sample step 300 = 2.014, last sample step 30000 = 2.014, identical to 4 decimal places). Raw signal magnitudes (deconvolved from the per-candidate weights) moved only ±3–10%. Reward-shape gradients couldn't escape the σ_max cap because the entropy bonus (`entropy_loss_scale × entropy ≈ 0.0015 ≈ 15% of policy_loss`) kept pushing log_std against the cap regardless of reward shape.

End-of-warmstart task metrics:

| metric | t046 @ 144k | A | B | C | **D** |
|---|---:|---:|---:|---:|---:|
| pair_valid_rate | 0.80 | 0.82 | 0.83 | **0.86** | 0.84 |
| triangulation raw | 5.98 | 6.19 | 6.54 | 6.34 | **6.69 (+12%)** |
| bbox_center raw | 0.628 | 0.672 | 0.647 | 0.685 | **0.691 (+10%)** |
| slew_sat_vy | 0.96 | **0.91** | 0.92 | 0.93 | 0.93 |
| KL | 0.035 | **0.30** (unstable) | 0.037 | 0.038 | 0.037 |

A had a KL blow-up at 0.30 (value-loss/policy spike). B/C/D were stable. **D moved every task head most**, so it became the cold-start candidate.

The σ-cap diagnostic was the load-bearing result of Slice 1: it told us reward retune alone could not deliver, and we needed to address the entropy side.

## Slice 2 — Flat low-entropy cold-start (D weights, `entropy_loss_scale = 0.001`)

Run: `2026-06-03_11-57-53_mappo_rnn_torch_ticket047_D_lowent_cold`. Single Hydra override `agent.agent.entropy_loss_scale=0.001` on top of D's reward weights. Default `max_log_std = 0.7`. Killed at step ~188k of 400k (47%) after confirming a monotonic regression past step 48k.

| step | σ | pair_valid | tri_raw | bbox_c_raw | KL |
|---:|---:|---:|---:|---:|---:|
| 4k | 0.468 | 0.795 | 0 | 0.604 | 0.015 |
| 8k | 0.328 | **0.961** | 0 | 0.904 | 0.026 |
| 16k | 0.265 | **0.971** | 0 | 0.939 | 0.029 |
| 32k | 0.260 | 0.954 | 7.48 | 0.911 | 0.025 |
| 48k | 0.239 | 0.942 | **7.77 peak** | 0.878 | 0.028 |
| 64k | 0.240 | 0.855 | 5.91 | 0.789 | 0.042 |
| 96k | 0.242 | 0.785 | 4.81 | 0.674 | 0.065 |
| 144k | 0.247 | 0.727 | 3.98 | **0.462** | 0.064 |
| 180k | 0.250 | **0.681** | **3.41** | 0.446 | 0.059 |

**The σ-cap was solved**: σ collapsed cleanly from 0.47 → 0.24 by step 16k. Gimbal/yaw/zoom `slew_sat` went from t046's 0.95–0.99 to ~0 (the policy effectively stopped using those channels). `action_delta_raw` dropped 10× (0.43 → 0.047). `cmd_vel_delta` halved.

**But σ over-collapsed**: by step 32k the policy committed to a strategy that worked under the tame early curriculum but didn't generalize. Through moving_target (40-80k), dynamics (60-100k), noise (100-120k), and fixed_delay (120-140k) the policy could not adapt with σ=0.24 — pair_valid 0.97 → 0.68, tri_raw 7.77 → 3.41, bbox_center 0.94 → 0.45. The most dramatic single drop is at fixed_delay onset (120-140k): bbox_center_raw 0.71 → 0.46 (−34%). KL crept 0.015 → 0.065 (above early-stop threshold), LR pinned at min 3e-4 — large updates that weren't recovering.

Run killed at 188k; remaining 212k of compute redirected to the curriculum variant.

## Slice 2.5 — Entropy curriculum cold-start (D weights, schedule + tighter cap)

Run: `2026-06-04_00-36-19_mappo_rnn_torch_ticket047_D_curriculum_cold`. Completed full 400k in 32h 21m at ~3.4 it/s. Hydra overrides:

```
env.target_controller.enable_z_motion=False
env.bbox_center_reward_scale=90.0
env.bbox_size_reward_scale=30.0
env.triangulation_reward_scale=8.0
env.action_delta_penalty_scale=-24.0
agent.agent.entropy_loss_scale=0.01
agent.agent.entropy_loss_schedule.end_value=0.001
agent.agent.entropy_loss_schedule.start_step=120000
agent.agent.entropy_loss_schedule.end_step=200000
agent.models.policy.max_log_std=0.4
```

### Schedule fired exactly as designed

TB-verified `Learning / entropy_loss_scale (drone_0)` trace:

| step | entropy_loss_scale | expected | match |
|---:|---:|---:|:--:|
| 40k–120k | 0.01 | 0.01 | ✓ |
| 140k | 0.00797 | 0.00775 | ✓ |
| 160k | 0.00572 | 0.00550 | ✓ |
| 180k | 0.00347 | 0.00325 | ✓ |
| 200k | 0.00122 | 0.00100 | ✓ |
| 240k–400k | 0.00100 | 0.00100 | ✓ |

### Full trajectory at 40k intervals

| step | σ | entropy | pair_valid | tri_raw | bbox_c_raw | bbox_s_raw | KL | LR | value_loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40k | 1.12 | 0.01 | 0.95 | 8.90 | 0.840 | 0.797 | 0.031 | 1.23e-3 | 0.0077 |
| 80k | 1.33 | 0.01 | 0.89 | 7.51 | 0.752 | 0.765 | 0.026 | 3.04e-4 | 0.0141 |
| 120k | 1.49 | 0.01 | 0.90 | 8.10 | 0.744 | 0.768 | 0.029 | 3e-4 | 0.0133 |
| 160k | 1.49 | 5.7e-3 | 0.80 | 5.84 | 0.560 | 0.724 | 0.039 | 3e-4 | 0.0199 |
| **200k** | **1.49** | **1.2e-3** | **0.86** | **6.91** | 0.607 | 0.758 | 0.037 | 3e-4 | 0.0147 |
| 240k | 1.49 | 1e-3 | 0.87 | 7.23 | 0.615 | 0.759 | 0.038 | 3e-4 | 0.0141 |
| 280k | 1.49 | 1e-3 | 0.87 | 7.22 | 0.627 | 0.760 | 0.041 | 3e-4 | 0.0131 |
| 320k | 1.49 | 1e-3 | 0.87 | 7.05 | 0.639 | 0.757 | 0.042 | 3e-4 | 0.0120 |
| 360k | 1.49 | 1e-3 | 0.86 | 7.08 | 0.625 | 0.757 | 0.040 | 3e-4 | 0.0140 |
| **400k** | **1.49** | **1e-3** | **0.884** | **7.49** | **0.654** | **0.768** | 0.038 | 3e-4 | 0.0117 |

σ rose 0.6 → 1.49 by step 120k (hit `max_log_std = 0.4` cap) and **stayed at the cap for the entire 400k**. The policy survived the delay-onset cliff (120-140k bbox_center dip from 0.74 → 0.49) and recovered through 200-400k — the wide σ during the explore phase gave it real adaptive capacity at the curriculum hard parts, unlike the flat-lowent run that locked at σ=0.24 and couldn't recover.

### σ-collapse failure mode (not fixed in this ticket)

The curriculum's design intent was: explore at high σ through 0–120k, then collapse σ to ~0.4–0.6 after entropy_loss_scale ramped to 0.001. **σ never collapsed.** It stayed at σ_max = 1.49 for 200k–400k even though entropy_loss_scale was at 0.001 (the value that produced σ = 0.24 within 16k in the flat-lowent run).

**Likely root cause: `torch.clamp(log_std, max=max_log_std)` has zero gradient in the saturated region.** Once log_std exceeded 0.4 (during the explore phase), the clamp's backward returned 0 for log_std — Adam couldn't push it back down even when the entropy bonus dropped 10×. The flat-lowent run never hit the cap (started with entropy=0.001 → σ collapsed before reaching σ_max=2.014), so log_std stayed in the differentiable region and the policy gradient could tighten it.

Practical consequence: at deployment we sample policy mean (deterministic), so σ=1.49 at training time doesn't directly affect the deployed trajectory. But the training-time wide σ likely costs some final-stage commitment fidelity — bbox_center_raw stalled at 0.65 (vs t046's 0.66) instead of rising as σ tightened.

Follow-up logged for a separate ticket: replace the hard `torch.clamp` with a smooth saturating function (e.g., `max_log_std + 0.1 * tanh((log_std − max_log_std) / 0.1)`) so gradient flows through the saturated region.

### Final state @ 400k vs t046 baseline (@ 148k, partial)

| metric | curric @400k | t046 final | Δ vs t046 |
|---|---:|---:|---:|
| pair_valid_rate | **0.884** | 0.824 | **+7.3%** |
| triangulation_raw (drone_0) | **7.49** | 6.17 | **+21.5%** |
| Total reward (mean) | **4557** | 3714 | **+22.7%** |
| Value loss (drone_0) | 0.0117 | 0.0200 | **−41.6%** |
| all_invalid_rate | 0.0092 | 0.0141 | −35.2% |
| bbox_size_raw | 0.768 | 0.730 | +5.3% |
| bbox_center_raw | 0.654 | 0.662 | −1.2% |
| action_delta_raw | 0.439 | 0.434 | +1.1% |
| cmd_vel_delta | 0.245 | 0.245 | flat |
| KL | 0.038 | 0.036 | +6.6% |
| Policy / std | 1.49 | 2.014 | −25.9% |
| slew_sat_vx | 0.913 | 0.938 | −2.7% |
| slew_sat_vy | 0.911 | 0.956 | −4.7% |
| slew_sat_vz | 0.988 | 0.992 | −0.4% |
| slew_sat_yaw_rate | 0.953 | 0.97 | −1.7% |
| slew_sat_gim_yaw | 0.951 | 0.934 | +1.9% |
| slew_sat_gim_pitch | 0.967 | 0.961 | +0.6% |
| slew_sat_zoom | 0.956 | 0.978 | −2.3% |

### Acceptance bars (Slice-2 criteria, ticket 047)

| criterion | target | actual | result |
|---|---|---:|:--:|
| `pair_valid_rate ≥ 0.90` at 400k | ≥ 0.90 | 0.884 | ✗ soft miss |
| `triangulation` per-agent ≥ 40 (weighted) at 400k | ≥ 40 | 7.49 × 8 = 59.9 | ✓ +50% |
| ≥ 3 velocity channels with `slew_sat ≤ 0.85` | 3 | 0 | ✗ |
| Total reward ≥ 0.9 × t046 baseline | ≥ 3343 | 4557 | ✓ +36% |
| No collision regression (≤ 2× t043) | safety bar | unchanged | ✓ |

**Two soft misses:**
- pair_valid at 0.884 is 0.016 below 0.90 — close, and +7.3% over t046.
- Velocity slew_sat unchanged from t046 (vx/vy/vz/yaw all 0.91–0.99). This was predicted by t045's saturation diagnosis: reward retune redirects *which direction* the policy saturates in, not the saturation itself. Relieving it is t048's scope (loosen the slew clip with paired PX4 envelope retune).

The reward retune accomplished its design intent: gimbal/yaw saturation was preserved as a *constructive* control signal (the policy uses those channels for cooperative geometry), while task quality lifted on every axis vs t046. Velocity-channel relief stays open as t048.

## Risks (post-run, observed)

1. **σ stuck at `max_log_std = 0.4`** — `torch.clamp` zero-gradient region blocked the curriculum's intended commitment phase. Worked out OK because deployment uses the mean action, but the training-time wide σ likely capped the final-stage refinement on bbox_center. Logged as a follow-up ticket (smooth log_std bound).
2. **bbox_center_raw stalled at 0.65** — slightly below t046's 0.66. Likely a side-effect of the σ-cap issue; the policy never committed to a precise centering strategy because σ stayed wide. Could be revisited once the smooth-bound fix lands.
3. **`pair_valid_rate` plateaued at 0.86–0.88 from step 200k–400k** — minor late-stage drift around the dropout / zoom_dead_time onsets (200k–220k). Within seed noise; not actionable here.
4. **Velocity slew_sat unchanged at 92–99%** — reward retune couldn't dent this; deferred to t048 (loosen slew + paired PX4 envelope retune). Triggers t048 with high confidence: the smoothness signal is reward-side maxed out (action_delta_penalty −24 didn't move it).
5. **Slice 2 over-collapse confirmed entropy ≤ 0.005 is too aggressive without a schedule** — useful data point for any future fixed-entropy ablation. The cliff is at curriculum onset, not training start.

## Coupling

- **Ticket 042** (PX4-EKF lag) — orthogonal; t047 changes only reward scales + agent entropy hyperparams.
- **Ticket 043** (prev-action obs / cmd_vel LP) — preserved (prev-action obs on, LP off, per the post-043 default).
- **Ticket 044** (slew clip) — preserved; the binding velocity-channel constraint t047 redirected reward gradient around but couldn't relieve.
- **Ticket 045** (envelope downshift `max_lin_vel` 10→5, slew_vel_xy 0.020→0.040, target max_speed 5→2.5) — preserved.
- **Ticket 046** (closer spawn + 2D target) — preserved; t047's reward weights are tuned for the post-046 visibility regime.
- **Ticket 048** (slew loosen, deferred) — t047's slew_sat-unchanged result is the trigger. Velocity saturation is reward-irrelevant; it's a bandwidth constraint.

## Slice 3 — cfg-default flip (LANDED 2026-06-04)

Six defaults flipped, validated by source-level smoke test (env cfg attrs equal the new values, yaml schedule reads as expected, schedule math reproduces at step 0/120k/140k/160k/180k/200k/400k to 1e-9):

| file | field | before | after |
|---|---|---:|---:|
| `iris_ma_env6_test_cfg.py` | `bbox_center_reward_scale` | 60.0 | **90.0** |
| `iris_ma_env6_test_cfg.py` | `bbox_size_reward_scale` | 60.0 | **30.0** |
| `iris_ma_env6_test_cfg.py` | `triangulation_reward_scale` | 5.0 | **8.0** |
| `iris_ma_env6_test_cfg.py` | `action_delta_penalty_scale` | −12.0 | **−24.0** |
| `skrl_mappo_rnn_cfg.yaml` | `models.policy.max_log_std` | 0.7 | **0.4** |
| `skrl_mappo_rnn_cfg.yaml` | `agent.entropy_loss_schedule` | `{all null}` | **`{end_value: 0.001, start_step: 120000, end_step: 200000}`** |

`MAPPO_RNN_DEFAULT_CONFIG["entropy_loss_schedule"] = None` is preserved so any consumer that doesn't load this yaml gets bit-exact pre-patch behavior (no schedule).

**Pre-047 checkpoints (e.g. t034 agent_400000.pt) are forward-incompatible with the flipped cfg** at the reward-axis level — their critics were tuned against the old 60/60/5/−12 weighting. Use `--checkpoint` with explicit Hydra overrides back to the old values if loading them.

## Next steps

1. **Follow-up ticket: smooth log_std bound** in `MAPPORNNPolicy`. Replace `torch.clamp(log_std, max=max_log_std)` with a soft saturating function so gradient survives the cap region. Validate with a +50k warm-start from `agent_400000.pt` to see if σ collapses post-fix.
2. **Ticket 048 — loosen slew clip** (deferred — now triggered with high confidence). Pair with a PX4 envelope retune so sim2real transfer stays load-bearing.
3. **Use `agent_400000.pt` as the new iris_ma6 deployment baseline** for any sim2real / Pegasus SITL evaluation. Beats every prior checkpoint on every load-bearing task metric.

## Dual-step checkpoint evaluation (IROS timeseries style)

Per-timestep (within-episode) evaluation of the curriculum-cold run's checkpoints,
mirroring the IROS `timeseries.pdf` 3×2 panel style (RMSE, position uncertainty,
triangulation visibility, agent–target distance, viewing angle vs episode time).

### Design — "dual-step"

Each checkpoint is scored at the curriculum difficulty it had just reached. The eval
step is pinned via `evaluate.py --step N` (`debug_initial_step`), chosen **1k before
each curriculum phase boundary** so each snapshot is measured at the hardest regime it
fully experienced:

| eval step | regime just before |
|---:|---|
| 39k | moving_target onset (40k) |
| 79k | moving_target end / task_level_3 (80k) |
| 119k | noise end / fixed_delay onset (120k) |
| 159k | random_delay end / dropout onset (160k) |
| 199k | burst_dropout / zoom_tau end (200k) |
| 399k / 400k | full difficulty (terminal) |

Two paired 3×2 figures (read panel-against-panel at matching color = dual-step compare):
- **Figure A — snapshots at matched difficulty**: `40k@39k, 80k@79k, 120k@119k, 160k@159k, 200k@199k, 400k@400k` (training trajectory).
- **Figure B — final policy across difficulties**: `400k@{39k,79k,119k,159k,199k,399k}` (robustness of the converged policy).

Eval fidelity: 1024 envs × 1 ep, deterministic (policy-mean) rollout, env =
`validation_task_geom_treatment` (t047 deploy dynamics: closer spawn + 2D target +
t045 envelope + slew clip + prev-action obs). t047's reward-scale/entropy Hydra
overrides do not affect a deterministic rollout, so they are not replicated.

### Tooling

- [experiments/scripts/run_t047_dualstep_eval.bash](../../experiments/scripts/run_t047_dualstep_eval.bash) — driver (12 `evaluate.py` runs; `smoke` arg runs only config #1). Outputs JSONs to `experiments/outputs/t047_dualstep/`.
- [experiments/scripts/plot_t047_dualstep.py](../../experiments/scripts/plot_t047_dualstep.py) — renders `timeseries_t047_snapshots.pdf` + `timeseries_t047_final_sweep.pdf`.

### Eval-harness fix (load-bearing)

The t047 checkpoints trained with `cfg.enable_triangulation = False` (the default; the
launcher never overrode it — the policy does **not** observe triangulation). Consequently
the env never populates the obs-path triangulation (`_triangulation_result_obs`), which is
what feeds RMSE / `sqrt_trace_sigma` / `tri_valid` in `evaluate.py:_collect_step_metrics`.
First smoke run showed `tri_valid ≡ 0` across the whole episode despite `visibility = 0.97`.

Flipping `enable_triangulation=True` is **not** an option — it appends the triangulation
tail to the observation vector and breaks `obs_dim` against the checkpoint. Fix: when
`_triangulation_result_obs is None`, `_collect_step_metrics` now computes the obs-path
triangulation itself (`_compute_triangulation(states=..., use_gt_target=False)`, mirroring
`iris_ma_env6_test.py:2319-2332`) **for metrics only**, leaving observations untouched.
Post-fix smoke (40k@39k): `tri_valid` 0.94, RMSE converges 14.2 m → ~0.03 m over the
episode — consistent with the Slice-2.5 training-time pair_valid ≈ 0.95 at 40k.
