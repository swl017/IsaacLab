## Ticket 036 — Probe whether the iris_ma6 MAPPO-RNN policy adapts to randomized env params

**Status**: Open
**Created**: 2026-05-19
**Discovered in**: Ticket-034 validation run analysis (post-fix `action_sum` +39 %, `action_delta` +35 % vs fix2 baseline).
**Target checkpoint**: `logs/skrl/iris_ma6/2026-05-18_11-45-51_..._ticket034_per_env_jitter/checkpoints/agent_400000.pt`
**Affected modules**: `iris_ma_env6_test.py` (read access to `_eff_progress_*` per-env tensors), `experiments/evaluate.py` (probe harness host).

---

### What

Determine empirically whether the GRU-based MAPPO-RNN policy actively system-identifies the per-(env, agent) randomized parameters (`max_lin_vel`, controller gains, gimbal/zoom dynamics, latency, noise scale, dropout rate, etc.) and adjusts its control strategy accordingly, or whether it applies a robust one-rule-fits-all controller that ignores the latent params.

### Why

Ticket 034's validation run shows the new 400 k checkpoint outperforming fix2 on every task-quality metric (bbox_center +17.5 %, collisions −53 %, pair_valid +2.7 pp, track_lost −20 %) but with substantially higher action-smoothness penalties (`action_sum` +39 %, `action_delta` +35 %). Two compatible explanations:

- **Adaptation**: the policy is inferring env params and commanding env-appropriate action magnitudes. Population-mean action energy stays roughly constant; the +39 % is concentrated in the high-mass / high-τ / high-latency envs and the policy is correctly commanding more aggressive recovery in those envs to maintain tracking.
- **Robust (no adaptation)**: the policy uses average-effort control regardless of env params, trading uniformly higher commanded magnitudes for the ability to satisfy task objectives across the full distribution. No per-env adjustment.

These have very different implications for follow-on engineering. Robust-only means an explicit adaptation mechanism (critic-only privileged obs per Q10, or aux sysid head, or privileged distillation) is a high-leverage forward investment. Adaptive means the current architecture is doing what we want and the +39 % is an unavoidable cost. The two cannot be distinguished from training-time tensorboard scalars alone.

Settling this also re-frames the action-smoothness concern. If the policy is adaptive, action_sum/action_delta are *correctly* higher in hard envs and *correctly* lower in easy envs — the population mean is high because hard envs dominate the average. If it's robust, every env (including easy ones) is getting unnecessarily aggressive commands and the policy is leaving deployment headroom on the table.

### Indirect evidence: policy std drift

Independent of the action-energy regression, the policy std (σ) trajectory
across training provides a second line of evidence pointing at non-adaptation:

| Run | σ at 28k | σ at 200k | σ at 400k | Δ over training |
|-----|---------:|----------:|----------:|----------------:|
| fix2 (pre-034 baseline) | 0.170 | 0.262 | 0.244 | +0.07 then collapses slightly |
| t034 (validated run)    | 0.170 | 0.281 | 0.281 | +0.11, asymptotes around 0.28 |

σ in t034 plateaus **+0.04 higher** than fix2 and continues to drift upward
through ~360 k, asymptoting around 0.28 in the last few checkpoints. The
drift is monotonic in distance-from-start (not a transient response to a
phase boundary).

**Mechanical interpretation.** σ is governed by the equilibrium between
two opposing forces in the PPO objective:

- **Entropy bonus** (`entropy_loss_scale=0.01 × H(π)`): gradient pushes
  σ UP, rewards exploration. Constant pull.
- **Policy gradient** (`E[ratio_t · A_t]`): when advantages reward specific
  actions, the policy peaks → σ DOWN. The clipping mask zeroes this pull
  for samples whose `ratio_t` leaves `[0.8, 1.2]`, so the inward pull
  weakens as σ shrinks (fewer "in-trust-region" samples to provide gradient).

σ stabilizes where these forces balance.

**Adaptive policy prediction**: σ should *collapse per env* once the GRU
hidden state identifies the env params and the policy commits to
env-specific deterministic actions. Population-level σ at training-end
should be **lower** than fix2 (which has a narrower training distribution
and therefore less to be uncertain about).

**Robust policy prediction**: σ should *grow* with the breadth of the
training distribution, because the policy must cover wider action
support to satisfy task objectives across env params. Population-level σ
should be **higher** than fix2.

**Observed: σ higher than fix2 by +0.04, asymptoting around 0.28.** This
is the robust-policy prediction. The +0.04 has a mechanical explanation:
ticket-034's wider env distribution requires a wider action distribution
to be optimal *if the policy doesn't have per-env conditional behavior*.

Combined with the +39 % `action_sum` and +35 % `action_delta` observations,
this gives two converging indirect signals for the robust-not-adaptive
hypothesis. Probes 1–3 below are the definitive tests, but the priors
heading in already favor "robust."

This also explains a closely-related observation people sometimes flag as
a training pathology: policy loss, value loss, and entropy loss don't
"converge" by the end of t034's 400 k run. They aren't supposed to.
Convergence to a fixed point is a supervised-learning expectation; on-policy
actor-critic stabilizes the losses into a band whose *floor* is set by
irreducible per-state outcome variance, not by training time. With wider
env support that floor is higher; with no privileged obs the critic
cannot attribute return variance to env-params, so value-loss residual is
permanent. None of these losses approach zero — they shouldn't.

The ticket-036 framing therefore extends naturally: probe #2 (linear
probe on GRU hidden state) is exactly the test of whether the residual
variance is recoverable via the GRU memory (i.e. the policy *could* be
adapting but isn't being trained to) or is genuinely orthogonal to it
(i.e. the GRU isn't carrying env identity at all).

### Probes

Three diagnostic experiments, cheap to run, no retraining:

**Probe 1 — Action energy vs episode step.**
Replay the 400 k checkpoint with a fixed env config (single mass, single `max_lin_vel`, single τ). Log `action_sum_per_step` and `action_delta_per_step` averaged across the rollout's envs. Plot as a function of step-in-episode.

- **Adaptive policy signature**: action variance is HIGH at steps 0–N (≈ 30 steps to fill the GRU history; "probing" the env) and DROPS to a stable lower level after. Hidden state has identified env, policy commits to env-appropriate control.
- **Robust policy signature**: action variance is STATIONARY across all episode steps from step 1. Hidden state is being used (if at all) only for target-velocity tracking, not env identification.

Cheapest probe. ~30 min wall-clock. Single eval invocation with a per-step action logger added to the trajectory recorder.

**Probe 2 — Linear probe on GRU hidden state (gold standard).**
Roll out N=1024 episodes spanning the full per-(env, agent) jitter distribution. At each step T ∈ {0, 4, 8, 16, 32, 64, 128, 256, 499}, save `(h_T, ground_truth_latent_params)` pairs where the latent params are `env._eff_progress_*` (a Tensor[N, A] direct read).

Fit a linear regression `latent_params = W · h_T + b` (Ridge / LASSO; or a tiny MLP for nonlinear sysid). Plot test R² as a function of T.

- **Adaptive signature**: R² rises from ~0 at T=0 to substantial value (> 0.5 on at least some axes) by T=32+. Hidden state encodes env identity.
- **Robust signature**: R² stays near 0 throughout. Hidden state is not encoding env identity; it's either empty or carrying only target-state info.

Gold-standard test. ~1 h to run + fit. Definitive answer per axis (mass, max_lin_vel, latency, noise, etc.).

**Probe 3 — Hidden-state-swap counterfactual.**
Roll out two parallel episodes, env_A and env_B, with very different jitter draws. At step T = 100, swap their GRU hidden states. Continue rollout. Measure action-trajectory KL divergence between (swapped env_A rollout) and (non-swapped env_A rollout).

- **Adaptive signature**: divergence is large — the policy's actions depend on env-conditional info in the hidden state.
- **Robust signature**: divergence is small — hidden state is mostly env-agnostic; swapping doesn't change behavior much.

Counterfactual test; complements Probe 2.

### Acceptance criteria

The ticket closes when **Probe 1 has been run and reported** (action vs episode-step plot + verdict). Probes 2 and 3 are recommended but not strictly required for closure; they're the definitive answer if Probe 1 is ambiguous.

Output: an entry in `iris_ma6/doc/experiments/` documenting the probe results, with a clear verdict (adaptive / robust / partially-adaptive) and per-axis breakdown if Probe 2 is run.

### Scope boundary

- **Out of scope**: actually implementing an adaptation mechanism. If this ticket concludes robust-only, that motivates a follow-up ticket for privileged-critic obs or aux sysid head — but those are separate work.
- **Out of scope**: retraining. All probes use the ticket-034-validated 400 k checkpoint as-is.
- **Out of scope**: comparison to fix2's checkpoint (different env distribution; not apples-to-apples for this question). Use only the 034 checkpoint.

### Implementation sketch

Probe 1 is a ~50 LOC addition to `experiments/evaluate.py` (or a separate `experiments/adaptation_probe.py`):

- Take a single eval-step (e.g. 199 000) so the curriculum is fully ramped.
- num_envs=512, run 2 × max_episode_length steps to ensure each env completes ≥ 1 full episode.
- Per step, accumulate `(step_in_episode, action_sum, action_delta)` keyed by env_id.
- At rollout end, group by step_in_episode and compute mean / std / p5 / p95 across envs.
- Dump CSV + a markdown table sampled at steps {0, 4, 8, 16, 32, 64, 128, 256, 499}.

Probe 2 needs a hook to dump the GRU hidden state after each forward pass — skrl exposes `policy.rnn_states` per agent on the rollout; can be read after each `agent.act(...)`.

### Why this isn't part of ticket 034

Ticket 034's pass criterion was visibility-ratio recovery, which was met. The action-energy regression is a *secondary* observation that doesn't impact the ticket's correctness, and the probe to characterize it is independent of the curriculum work. Splitting it out keeps the deployment-readiness investigation focused.

---

### Next step (after probes confirm/refute adaptation): add critic-only privileged obs

Whether or not the probes confirm non-adaptation, the cleanest forward
investment is **adding critic-only privileged observations** — the
mechanism deferred from ticket 034 (Q10 answer) and the one strongly
suggested by the priors above. The architecture:

- **Actor** (per-agent policy): observation unchanged. Continues to see
  only deployment-realizable quantities (the 31D ego + 16D × (N−1)
  inter-agent obs). Deployment compatibility preserved — no change to
  the offboard observation contract.
- **Centralized critic** (value head, MAPPO already has it asymmetric per
  ticket 033): obs extended with per-(env, agent) effective-progress
  tensors that the env already maintains. Specifically the 8 currently-
  cached tensors:
  - `_eff_progress_agent_velocity`, `_eff_progress_dynamics`,
  - `_eff_progress_gimbal_dead_time`, `_eff_progress_zoom_dead_time`,
  - `_eff_progress_noise`, `_eff_progress_dropout`,
  - `_eff_progress_delay`, `_eff_progress_burst_dropout`.
  Per-agent these are 8 scalars; flattened per env they're 8 × A obs.

Mechanism:

1. **Value loss drops substantially.** The critic can now attribute return
   variance to the env-param nuisance, so the value-loss floor (currently
   set by `Var[return | observed_state]` integrated over the unobserved
   env params) drops to the residual `Var[return | observed_state, env_params]`,
   which is smaller.
2. **Advantage signal sharpens.** Cleaner GAE → less noisy policy
   gradient → policy can commit to env-conditional behavior on the
   states where it matters.
3. **σ collapses to a per-env-conditional value.** With sharper
   advantages and a value function that respects env-params, the policy
   gradient pull on σ strengthens. Population σ drops toward the fix2
   plateau (or below). `action_sum` / `action_delta` should drop in turn
   if the policy is genuinely able to use the (privileged-critic-derived)
   signal to behave env-specifically.
4. **The probe outcome decides where the gains accrue.** If probes show
   the actor-side GRU is already encoding env identity (Probe 2 R² > 0.5
   at T=32+), then the bottleneck was just the value head not exploiting
   it → privileged obs gives an immediate value-loss reduction and
   advantage sharpening. If the GRU isn't encoding env identity at all
   (R² ≈ 0), then privileged obs alone may not be enough — a stronger
   intervention (aux sysid head, privileged distillation) might be
   warranted instead.

The change is small (8 × A new critic-input dims; no actor change;
existing `_eff_progress_*` tensors already live on the env). Implementing
behind a cfg flag (`enable_critic_continuous_zoom`-style toggle) lets it
ship without breaking existing checkpoints.

Estimated effort: ~1 day to implement, ~24 h to retrain at 400 k for
A/B comparison against the t034 baseline. Worth doing whether or not
adaptation is confirmed — it strictly improves the value function's
discrimination power, and the actor-side downstream effects are an upside.
