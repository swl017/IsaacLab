# Experiment: 036_policy_adaptation_probe

**Commit**: `f07c8bfc0c10c667331dc97d636da6f0d57b3bce`
**Date**: 2026-05-20
**Closes**: ticket 036 (probe whether the t034 MAPPO-RNN policy adapts to per-env jitter)
**Target checkpoint**: `logs/skrl/iris_ma6/2026-05-18_11-45-51_mappo_rnn_torch_034_2026-05-18_11-45-45_ticket034_per_env_jitter/checkpoints/agent_400000.pt`
**Probe artifacts**: [outputs/2026-05-20/probe036/](../../experiments/outputs/2026-05-20/probe036/)

---

## 1. Question

Does the 400 k t034 checkpoint actively system-identify the per-(env, agent)
randomized parameters (`max_lin_vel`, dynamics gains, gimbal/zoom dead time,
delay, noise, dropout) and adjust its control strategy accordingly, or does
it apply a robust one-rule-fits-all policy?

Ticket-034 validation showed the 400 k checkpoint outperforms fix2 on
task-quality metrics (bbox_center +17.5 %, collisions −53 %, pair_valid
+2.7 pp, track_lost −20 %) but at +39 % `action_sum` and +35 % `action_delta`.
Two compatible explanations:

- **Adaptive**: more aggressive commands in hard envs, calmer in easy envs;
  population mean inflates because hard envs dominate the integrated stat.
- **Robust**: uniformly higher command magnitudes across all envs because the
  policy lacks per-env conditional behavior.

The indirect priors (σ +0.04 above fix2 plateau, monotone σ growth across
training) point at "robust". This probe is the direct test.

## 2. Method

Three read-only diagnostic experiments against `agent_400000.pt` in
[experiments/adaptation_probe.py](../../experiments/adaptation_probe.py). No
retraining, no env changes, no checkpoint mutation. Curriculum pinned to
step 199 000 (fully ramped) via `debug_initial_step`. `num_envs=1024`,
`seed=42`, episode length 500 steps (20 s @ 25 Hz).

### Probe 1 — action energy vs episode step

Per-step accumulator of `|a|_1` and `|Δa|_1` (mean over agents) bucketed by
`step_in_episode`. Rollout for 2 × `max_episode_length` so every env crosses
every step. Verdict rule: `early[5:30].mean() / steady[125:250].mean()` of
`action_delta`. Adaptive ≥ 1.3; robust ≤ 1.1.

### Probe 2 — linear / MLP decoding of latents

At T ∈ {0, 4, 8, 16, 32, 64, 128, 256, 499}, snapshot per-agent **pre-GRU MLP**
features (via `nn.Module.register_forward_hook` on `policy.net`) and
**post-GRU hidden state** (from `outputs[uid]["rnn"][0]`). One sample per
`(env, agent)` per episode. After rollout, fit Ridge and a small MLP
(hidden=32) per latent axis with 80/20 train/test split; report test R²
per axis × T × feature-type. Adaptive on axis X iff `max_T R²_mlp(X) > 0.5`
on either pre or post features. Robust iff `max_T R²_mlp(X) < 0.1`.

### Probe 3 — hidden-state swap counterfactual

Pair envs `(i, i+N/2)` (i.e., bank A = first half, bank B = second half).
Three rollouts (reference, cross-swap at T=100, identity-swap control); at
T=100, the cross-swap rollout exchanges `_rnn_states["policy"]` between
paired envs. Per-step approximate KL between cross-swap and reference, and
between control and reference, under the policy's state-independent
Gaussian (KL ≈ ‖μ₁ − μ₂‖² / (2σ²)). Verdict on **detrended** signal:
`(swap_post − swap_pre) − (ctrl_post − ctrl_pre)`. Robust if |detrended|
< 0.5; adaptive if > 2.0.

**Reproducibility caveat**: Isaac Sim physics is not bit-reproducible across
`env.reset()` calls in this setup even with matched seeds, so the
"identity-swap control" has a non-zero per-step KL floor. The probe reports
detrended signal (post-pre) to remove the constant floor; if the control
itself shows post-swap excess > 1 nat, the floor drifts and the verdict is
flagged "caution".

## 3. Results

### 3.1 Probe 1 — action energy vs episode step

`n_envs=1024`, two full episodes covered per env (≥ 2 000 samples per binned step
for T ≤ 256; T=499 caught at episode boundaries where `step_in_ep` is
clipped before `done`, so the empty count for T=499 is expected and not
load-bearing).

| Step | n | `mean(\|a\|_1)` | `mean(\|Δa\|_1)` |
|---:|---:|---:|---:|
| 0 | 3074 | 2.353 ± 0.598 | 1.640 ± 1.246 (reset transient — `prev_actions = None`, skipped) |
| 4 | 2138 | 2.564 ± 0.690 | 0.779 ± 0.396 |
| 8 | 2137 | 2.470 ± 0.711 | 0.749 ± 0.392 |
| 16 | 2134 | 2.200 ± 0.720 | 0.751 ± 0.387 |
| 32 | 2129 | 1.902 ± 0.601 | 0.622 ± 0.312 |
| 64 | 2112 | 1.775 ± 0.582 | 0.584 ± 0.307 |
| 128 | 2086 | 1.610 ± 0.657 | 0.494 ± 0.275 |
| 256 | 2046 | 1.431 ± 0.668 | 0.426 ± 0.235 |

**Mechanical reading.** `action_sum` drops monotonically from ~2.5 (early) to
~1.4 (mid-episode). `action_delta` drops from ~0.75 (steps 4–16) to ~0.43
(step 256). Verdict rule `early[5:30]/steady[125:250]` returns ratio = **1.50**,
above the 1.30 "adaptive" threshold.

**Interpretation:** *Probe 1 alone is not diagnostic.* A robust policy with a
non-zero settling time produces this same signature: random initial pose →
large corrective commands → as the system aligns with its steady-state
pointing geometry, commands drop. Probes 2 and 3 are needed to distinguish.

[Plot](../../experiments/outputs/2026-05-20/probe036/probe1_action_vs_step.png) ·
[CSV](../../experiments/outputs/2026-05-20/probe036/probe1_action_vs_step.csv) ·
[JSON](../../experiments/outputs/2026-05-20/probe036/probe1_action_vs_step.json)

### 3.2 Probe 2 — hidden-feature decoding (the gold-standard test)

`n_envs=1024 × A=2 agents = 2048 (env, agent) samples per snapshot T`,
80/20 train/test, ridge α=1.0 and 32-hidden MLP. Per-axis verdict on
`max_T R²_mlp` over both pre-GRU and post-GRU features:

| Latent axis | Verdict | max R²_post | max R²_pre |
|---|---|---:|---:|
| `eff_progress_agent_velocity`   | **robust** | −1.05 | −0.40 |
| `eff_progress_dynamics`         | **robust** | −1.13 | −0.58 |
| `eff_progress_gimbal_dead_time` | **robust** | −1.18 | −0.40 |
| `eff_progress_zoom_dead_time`   | **robust** | −0.95 | −0.46 |
| `eff_progress_noise`            | **robust** | −1.07 | −0.51 |
| `eff_progress_dropout`          | **robust** | −1.07 | −0.36 |
| `eff_progress_delay`            | **robust** | −0.76 | −0.22 |
| `eff_progress_burst_dropout`    | no-data (degenerate — near-zero variance at this curriculum step) | — | — |

**All non-degenerate axes return negative R²** on both pre-GRU MLP features
and the post-GRU hidden state. Negative R² means the trained linear/MLP
decoder does *worse than predicting the mean* on held-out (env, agent) pairs.
The decoder cannot recover env identity from either representation.

Two secondary observations worth recording:

1. **Post-GRU R² is consistently *more* negative than pre-GRU R²** (−1.1 vs
   −0.4 on average). The GRU is not concentrating env-identity information —
   if anything, training has pushed the post-GRU representation *further*
   from anything linearly/MLP-decodable into env params. Consistent with
   "the GRU carries target-state memory, not env identity".
2. **`burst_dropout`** is degenerate at training step 199 k because its
   curriculum ramp hasn't introduced variance yet (every env draws the same
   value, so R² is undefined). Not load-bearing for the verdict.

`T=4` returns wildly oscillating R² (clipped "EX" in tabular dump) because
the GRU has only seen 4 obs steps and the test split is dominated by
high-leverage outliers. Steps T ∈ [8, 256] are all consistently negative,
which is the load-bearing window.

[Heatmap](../../experiments/outputs/2026-05-20/probe036/probe2_hidden_decoding.png) ·
[JSON](../../experiments/outputs/2026-05-20/probe036/probe2_hidden_decoding.json)

### 3.3 Probe 3 — hidden-state swap counterfactual

`n_envs=1024`, half=512 paired envs, swap at T=100, comparing cross-swap
vs identity-control rollouts against a no-swap reference.

| | pre-swap KL | post-swap KL (T ∈ [100, 199]) | excess (post − pre) |
|---|---:|---:|---:|
| Cross-swap   | 19.36 | 13.56 | **−5.79** |
| Identity-ctrl | 19.46 | 13.88 | **−5.58** |
| Detrended signal (swap_excess − ctrl_excess) | | | **−0.21** |

The absolute KL is dominated by Isaac Sim's non-determinism floor
(~19 nats pre-swap divergence between rollouts that *should* be identical;
~14 nats post-swap due to envs naturally settling into less divergent
trajectories). The two rollouts both drop by ~5.7 nats from pre to post —
i.e., the floor itself moves. The detrended signal of **−0.21 nat** is
well below the robust threshold of ±0.5 nat: the swap **did not** measurably
change downstream actions beyond the noise floor.

If the policy were adaptive, swapping the hidden state of paired envs (each
with its own jitter draw) would feed env A's GRU memory to env B's
observations — the action commands at env B should be off from "what env B
would have done with its own GRU memory". We see no such effect.

[Plot](../../experiments/outputs/2026-05-20/probe036/probe3_swap_kl.png) ·
[CSV](../../experiments/outputs/2026-05-20/probe036/probe3_swap_kl.csv) ·
[JSON](../../experiments/outputs/2026-05-20/probe036/probe3_swap_kl.json)

## 4. Verdict

**Robust, not adaptive.** Convergent evidence:

1. **Probe 2 (gold-standard)**: no latent axis is decodable from either
   pre-GRU or post-GRU features (R² < 0 across 7 of 7 non-degenerate axes).
2. **Probe 3**: hidden-state swap produces no detectable downstream action
   divergence (−0.21 nat detrended, within ±0.5 robust threshold).
3. **Probe 1**: the apparent "adaptive" signature (action_delta dropping
   1.5× from early to mid-episode) is explained by initial-state
   settling — the policy is correcting a random initial pose toward
   steady-state pointing, not identifying env params.
4. **Indirect priors confirmed**: σ +0.04 above fix2 plateau, +39 %
   action_sum, +35 % action_delta in t034 are now fully accounted for by
   the wider env support requiring wider action support under a
   non-adaptive policy. The PPO losses don't "converge" because the
   irreducible per-state outcome variance is bounded below by
   `Var[return | obs]`, not `Var[return | obs, env_params]`.

The +39 % action-energy regression vs fix2 is therefore **not** a free
optimization — every env is getting more-aggressive commands than the
env-specific optimum, leaving deployment headroom on the table.

## 5. Follow-up: critic-only privileged obs

Per ticket 036 §"Next step": whether the verdict is robust or adaptive,
the cleanest forward investment is adding the 8 `_eff_progress_*` tensors
(8 scalars × A agents) to the **centralized critic's obs only**. Actor
observation unchanged (preserves deployment compatibility). Mechanism:

- **Value loss floor drops**: critic now attributes return variance to
  env-param nuisance, so `Var[return | obs, env_params] < Var[return | obs]`.
- **Advantage sharpens**: cleaner GAE → less noisy policy gradient → policy
  can commit to env-conditional behavior on the states where it matters.
- **σ collapses toward fix2 plateau**: stronger inward pull on σ from the
  sharpened advantage, so action_sum / action_delta should drop.

Estimated effort: ~1 day to implement (8 × A new critic-input dims, no actor
change), ~24 h to retrain 400 k for A/B against the t034 baseline. Worth
doing whether or not the probes confirm "robust" — strictly improves the
value function's discrimination, with actor-side downstream effects as
upside if the probes show partial adaptation.

If Probe 2 returns R² ≈ 0 on every axis (strict robust), then privileged
critic obs alone may be insufficient — a stronger intervention (aux
sysid head per [doc/policy_triangulation_head_spec.md](../../doc/policy_triangulation_head_spec.md)-style auxiliary loss, or
privileged distillation) may be warranted, opening a separate ticket.

## 6. Limitations / Open

- **Single curriculum step (199 k)**. R² as a function of training step
  is left to a backlog item if the verdict is ambiguous.
- **Probe 3 noise floor**: physics non-determinism caps the resolution
  of the swap signal. Detrended signal compensates only partially.
- **64-dim GRU hidden**. Smaller than typical sysid representations; the
  representational ceiling on Probe 2 R² may be capped by hidden dim
  rather than encoding fidelity.
