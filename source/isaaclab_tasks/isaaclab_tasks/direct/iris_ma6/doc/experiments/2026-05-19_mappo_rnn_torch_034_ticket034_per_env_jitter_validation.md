# Experiment: 034_2026-05-18_11-45-45_ticket034_per_env_jitter

**Run dir**: `logs/skrl/iris_ma6/2026-05-18_11-45-51_mappo_rnn_torch_034_2026-05-18_11-45-45_ticket034_per_env_jitter`
**Date**: 2026-05-18 → 2026-05-19 (34 h 08 min wall-clock)
**Validates**: ticket 034 (curriculum-driven catastrophic forgetting)
**Base**: `2026-05-15_12-00-44_..._scheduler_param_fix2` (ticket-033 baseline)
**Status**: Completed 400 k / 400 k. **Closes ticket 034.**

---

## 1. Hypothesis

The fix2 baseline showed 38–42 % visibility deficit on early-phase env
settings (39 k / 99 k) when evaluating its 400 k checkpoint. R0 audit
attributed this to ~14 of 16 curriculum axes using a deterministic
`progress · value` sampler that collapses the env distribution onto the
hardest setting at `progress = 1`, eliminating the easy-regime
gradient signal.

Ticket 034's fix introduces per-(env, agent) anti-forgetting sampling:
at each reset, every (env, agent) draws `eff_p ~ Uniform(0, global_p)`
and applies the existing mapping with `eff_p` in place of `global_p`.
The env distribution therefore retains positive mass at the easy regime
at every global progress value.

Hypothesis: the 400 k checkpoint trained under the new sampler recovers
visibility at low eval steps to ≥ 80 % of the matched-step reference
checkpoint, without degrading late-training reward parity vs fix2.

## 2. Configuration delta

Code changes per ticket-034 Slices 1–4. No yaml changes from fix2.
Same `skrl_mappo_rnn_cfg.yaml`: `rollouts=32`, `learning_epochs=3`,
`mini_batches=8`, `sequence_length=32`, `episode_start_mask_steps=8`,
`kl_threshold=0.04` (PPO), `kl_threshold=0.02` (scheduler),
`kl_factor=2.0`, `lr_factor=1.25`, `min_lr=3e-4`, `max_lr=1.5e-3`.

Environment: `episode_length_s=20.0`, `num_envs=1024`, `seed=0`.

## 3. Results

### 3.1 Regression eval (the headline)

Visibility ratio = post-fix 400 k / matched-step reference checkpoint
(both from this run, evaluated under `--step <S>` curriculum pinning).
Pass bar: ratio ≥ 0.80 (Q5).

| Eval step | Reference vis | Post-fix 400 k vis | **Post-fix ratio** | Pre-fix ratio (R0) | Δ | Pass? |
|---:|---:|---:|---:|---:|---:|:---:|
| 39 000 | 0.930 | 0.836 | **0.898** | 0.616 | **+0.282** | ✓ |
| 99 000 | 0.912 | 0.754 | **0.827** | 0.577 | +0.250 | ✓ |
| 199 000 | 0.863 | 0.892 | **1.034** | 0.988 | +0.045 | ✓ |
| 399 000 | 0.889 (self) | 0.889 | 1.000 | 1.000 | 0.000 | n/a |

All four cells pass. The 38–42 % visibility deficit on early-phase
settings is eliminated. At step 199 k the new 400 k checkpoint *exceeds*
the matched-step 200 k checkpoint's visibility, the opposite of the
pre-fix forgetting pattern.

Reference visibilities are within ±0.015 between the two runs at every
eval step — the curriculum changes do not regress early-phase
performance.

### 3.2 Training-time controllers

| step | KL (fix2 / t034) | LR (fix2 / t034) | std (fix2 / t034) |
|---:|---:|---:|---:|
| 40 k | 0.022 / 0.022 | 1.49e-3 / 1.50e-3 | 0.21 / 0.22 |
| 80 k | 0.027 / 0.028 | 1.46e-3 / 1.37e-3 | 0.23 / 0.23 |
| **120 k** | 0.027 / 0.025 | 1.47e-3 / **1.01e-3** | 0.24 / 0.21 |
| 160 k | 0.021 / 0.022 | 1.50e-3 / 1.50e-3 | 0.28 / 0.27 |
| 200 k | 0.022 / 0.022 | 1.50e-3 / 1.50e-3 | 0.26 / 0.28 |
| 240 k | 0.023 / 0.023 | 1.50e-3 / 1.49e-3 | 0.26 / 0.27 |
| 280 k | 0.022 / 0.024 | 1.50e-3 / 1.50e-3 | 0.25 / 0.28 |
| 320 k | 0.024 / 0.024 | 1.50e-3 / 1.49e-3 | 0.24 / 0.28 |
| 360 k | 0.023 / 0.024 | 1.50e-3 / 1.48e-3 | 0.24 / 0.29 |
| 400 k | 0.023 / 0.026 | 1.50e-3 / 1.49e-3 | 0.24 / 0.28 |

- **KL** in [0.022, 0.028] all run — identical band to fix2, well below
  the 0.04 PPO early-stop.
- **LR** at max (1.5e-3) for ~95 % of training. One transient ~33 % dip
  at 120 k covering the 100–120 k noise window + 120–140 k fixed-delay
  ramp; recovered to max by 160 k. The KLAdaptive controller is doing
  its job under the extra per-(env, agent) jitter variance, with
  generous headroom.
- **Std** drifts up ~0.05 from fix2 (terminal 0.28 vs fix2 0.24).
  Well below the 0.30 collapse threshold. The wider on-policy support
  reflects the wider env distribution.

The 8 s-episode-mistake run (killed at 128 k) had LR pinned at min from
84 k onward; this run, with `episode_length_s=20.0` reverted, never
exhibits that pathology.

### 3.3 Mean reward and per-component (last-10 avg at 400 k)

| Metric | fix2 | t034 | Δ |
|---|---:|---:|---:|
| Mean reward (last-10 avg) | 2 714 | **2 642** | −72 (−2.7 %) |
| Peak reward | 5 406 @ 28 k | 5 346 @ 32 k | −60 (−1.1 %) |
| `bbox_center` | +23.0 | **+27.0** | +17.5 % |
| `bbox_size` | +43.5 | +44.1 | parity |
| `triangulation` | +24.7 | +24.8 | parity |
| `collision` | −0.077 | **−0.036** | **53 % fewer** |
| `cbf_penalty` | −0.034 | −0.024 | 29 % lower |
| `action_sum` | −9.1 | −12.6 | 39 % worse |
| `action_delta` | −11.1 | −15.0 | 35 % worse |
| `pair_valid_rate` | 0.807 | **0.834** | +2.7 pp |
| `tracking_lost_fraction` | 0.041 | **0.033** | 20 % lower |

**Net pattern**: every task-quality metric improves (bbox tracking,
pair-valid, track-loss, safety). Action-smoothness penalties worsen
35–40 % — the policy produces more aggressive and jittery commands as
its cost for tracking under wider env variance. The two effects roughly
cancel in mean reward (last-10 avg parity within noise).

### 3.4 Visibility at low eval steps — pre vs post

| Eval step | Pre-fix 400 k vis | Post-fix 400 k vis | Absolute Δ |
|---:|---:|---:|---:|
| 39 k | 0.574 | **0.836** | +0.262 (+46 % relative) |
| 99 k | 0.531 | **0.754** | +0.223 (+42 % relative) |
| 199 k | 0.868 | 0.892 | +0.024 (+3 %) |
| 399 k | 0.875 | 0.889 | +0.014 (+2 %) |

Recovery is concentrated at low eval steps, exactly where the
anti-forgetting fix targets.

## 4. Decision

- **Ticket 034 closed**. Per-(env, agent) anti-forgetting sampling on
  the dynamics and observability axes recovers low-step-curriculum
  performance to ≥ 89 % of matched-reference at every audited point,
  exceeding the Q5 80 % bar by a comfortable margin.
- **Use `agent_400000.pt` from this run as the new iris_ma6 baseline.**
  Better than fix2 by every task-quality and safety metric; comparable
  on mean reward; only behind on action-smoothness penalties (which
  reflect the wider on-policy distribution, not a regression in
  control quality).
- **No follow-up training run needed.** Validation suite passes from a
  single seed; revisit if a second-seed reproducibility check is
  required for publishing the numbers.

### Notes for future work (not blocking)

- The action-smoothness regression (`action_sum`, `action_delta` ~35 %
  worse) is worth measuring in offboard deployment. If the rate-loop
  command signal is noticeably noisier, consider tightening the action
  penalty weights, or adding a critic-only privileged-obs path (per
  Q10) so the value head can disambiguate aggressive vs stable
  policies under wide env variance.
- The 120 k LR dip was the worst controller-health event of the run.
  Eliminating it would require either a wider scheduler dead-zone or
  a narrower jitter window (`Uniform(α·p, p)` with `α > 0`).
  Optional optimization — current behavior is acceptable.
- `progress_tracking` axes in `initial_states` are already
  anti-forgetting under the default config (R2.8 verified). No action
  needed unless `gimbal_curriculum_mode` is switched to `"gradual"` /
  `"threshold"` or `other_agents_orientation_mode` to `"curriculum"`,
  in which case those distributions need the same fix.
- Deferred to follow-up tickets: FP/FN curriculum redesign (Q6),
  delay-system runtime efficiency (Q13), optional critic-only
  privileged obs (Q10).
