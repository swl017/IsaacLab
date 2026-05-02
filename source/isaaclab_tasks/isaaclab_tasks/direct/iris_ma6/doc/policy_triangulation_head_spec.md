# Policy Triangulation Head Specification

**Project:** `iris_ma6` — Multi-Drone Active Triangulation
**Scope:** Auxiliary actor head for distributed target position + uncertainty estimation
**Status:** Draft v1
**Last Updated:** 2026-05-02

**Related Documents:**
- [triangulation_spec.md](triangulation_spec.md) — Geometric triangulation module (the head's training-time supervisor)
- [iris_ma6_env_spec.md](iris_ma6_env_spec.md) — Environment overview
- [observation_redesign_spec.md](observation_redesign_spec.md) — Planned actor obs structure (orthogonal to this head)
- [frame_conventions.md](frame_conventions.md) — Coordinate systems
- Implementation plan: [active/ticket/031-policy-triangulation-head/ticket.md](active/ticket/031-policy-triangulation-head/ticket.md)

---

## 1. Purpose

A per-agent auxiliary regression head that predicts the target's 3D world-frame position and per-axis uncertainty (`μ_xyz, log σ²_xyz`) from each agent's actor observation. The head shares the encoder backbone (MLP + GRU, inherited from `MAPPORNNPolicy`) with the action head and is trained with a Gaussian negative-log-likelihood (NLL) loss against the ground-truth target position. The auxiliary loss is masked **per-agent** by `(this agent's bbox is non-empty) AND (scene's _triangulation_result_gt.is_valid)`.

Implementation lives at [scripts/reinforcement_learning/skrl/mappo_with_aux.py](../../../../../scripts/reinforcement_learning/skrl/mappo_with_aux.py) (alongside `mappo_rnn.py`), trained via [scripts/reinforcement_learning/skrl/train_mappo_with_aux_hydra.py](../../../../../scripts/reinforcement_learning/skrl/train_mappo_with_aux_hydra.py). The head subclasses `MAPPORNNPolicy`; the trainer subclasses `MAPPO_RNN` and overrides `init`, `record_transition`, and `_update` to plumb supervision and add the auxiliary loss term.

### 1.1 Two intents

**Intent 1 — Implicit target reasoning (1a).** The auxiliary supervision shapes the encoder. The action head consumes the same encoder hidden state that produces (μ, σ), so target-reasoning features get gradient pressure from both the policy loss *and* the supervision loss. The action head **never directly sees (μ, σ)** — no skip connection. Why this and not the explicit/skip-connection variant: avoids the bootstrap problem (the action head would otherwise condition on uncalibrated head outputs early in training) and prevents the policy loss from corrupting the head's calibration via shared parameters.

**Intent 2 — Distributed estimator at deployment.** The geometric triangulator ([triangulation/triangulation.py](../triangulation/triangulation.py)) is a centralized fusion that requires all agents' rays + poses delivered to one node per step. By contrast, this head runs on each agent independently, consuming only that agent's actor observation, and emits a Gaussian measurement (μ_i, σ_i) that downstream subsystems can consume directly. **At deployment, the geometric triangulator is not run online** — the head replaces it. The geometric pipeline remains a *training-time* artifact (it produces the supervision targets and the validity mask).

These two intents converge on a single supervision strategy: NLL against the GT target position, with σ implicitly calibrated to "how wrong the head's μ is likely to be" under the deployment-realistic delay/noise pipeline.

---

## 2. Loss Intuition

The head's output is interpreted as a Gaussian distribution over the target's world position, with diagonal covariance:

```
p(target | obs) = N(target | μ(obs), diag(σ²(obs)))
```

where μ ∈ ℝ³ and σ² ∈ ℝ³₊ are produced by the head from the encoder hidden state. The head outputs `log σ²` (not σ² directly) so the parameter space is unconstrained; `σ² = exp(log σ²)` is computed at the loss site.

**Loss = negative log-likelihood of the GT target under the predicted Gaussian:**

```
−log p(GT | μ, σ²)  =  0.5 · Σᵢ [ (GTᵢ − μᵢ)² / σ²ᵢ  +  log σ²ᵢ ]   (+ const)
```

The two terms are in tension, and the equilibrium is what makes σ a calibrated uncertainty:

| Term | Effect | Failure if alone |
|------|--------|------------------|
| `(GT − μ)² / σ²` | Inverse-variance-weighted MSE: confident-but-wrong predictions are amplified; uncertain predictions are softened | Without the second term, the head would push σ → ∞ to make any prediction "fine" |
| `log σ²` | Penalty on inflating σ | Without the first term, the head would push σ → 0 with no incentive for μ to be accurate |

**Equilibrium.** Per axis, taking ∂/∂σ²ᵢ of the loss and setting to zero gives `σ²ᵢ = (GTᵢ − μᵢ)²`. So the optimal σᵢ at each input is the actual squared error of the head's μ at that input. Marginalized over the data distribution given an input's features: σ²ᵢ converges to the **conditional variance of the residual** — i.e., the irreducible uncertainty in the target's position given everything the head can observe.

This is the mechanism that makes σ a deployment-trustworthy quantity (Intent 2): it tells consumers "given what this agent can see right now, here is the expected per-axis squared error of my prediction." If the agent's bbox is fresh and well-localized, σ shrinks. If the bbox is stale, occluded, or absent, σ grows — provided the head was trained on samples where that condition occurred (which is the role of the validity mask, see §4.3).

**Numerical stability.** `log σ²` is clamped to `[−10, 4]` before the `exp()` to prevent (a) division by zero when σ² is too small at init and (b) gradient explosion when σ² is unboundedly large. This range admits σ ∈ [≈ 7e-3, ≈ 7] meters per axis, which spans the relevant regime for this environment without truncating useful values.

**Implementation form** (matches workflow step 2 of the ticket):
```python
per_dim_nll = 0.5 * ((target - mu).pow(2) * (-log_var).exp() + log_var)  # [B, 3]
nll = per_dim_nll.sum(dim=-1)                                            # [B]
aux_loss = (nll * mask).sum() / mask.sum().clamp_min(1.0)
total_loss = ppo_loss + loss_scale * aux_loss
```

`(-log_var).exp()` is `1/σ²`; the additive `log_var` is the σ-inflation regularizer. The mask is `_triangulation_result_gt.is_valid` (see §4.3).

---

## 3. Architecture

```
            obs (per-agent actor obs, shape [B, S, obs_dim])
                       │
                       ▼
              ┌──────────────────┐
              │  Encoder MLP     │  pre-GRU MLP (matches MAPPORNNPolicy)
              └──────┬───────────┘
                     │
                     ▼
              ┌──────────────────┐
              │  GRU layers      │  shared with MAPPORNNPolicy structure
              └──────┬───────────┘
                     │ h  (post-GRU hidden state)
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐ ┌─────────┐  ┌─────────────┐
   │ action  │ │ log_std │  │ tri head    │  hidden = [128, elu]
   │ head    │ │ (param) │  │ → (μ, log σ²)
   │ → mean  │ │         │  │   6 outputs │
   └─────────┘ └─────────┘  └─────────────┘
        │            │            │
        └────────────┴────────────┘
                     ▼
        Gaussian action distribution
        (mean, log_std) + auxiliary (μ, log σ²)
```

**Critical**: the action head's input is `h` only — **not** `[h, μ, σ]`. No skip connection. This is enforced by a regression test (`test_no_skip_connection.py`).

The encoder structure (MLP + GRU) is inherited from [scripts/reinforcement_learning/skrl/mappo_rnn.py](../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py)'s `MAPPORNNPolicy`. The tri head reads `h` at the same point the action head reads from — the post-GRU hidden tensor — ensuring both heads see identical features.

---

## 4. Design Choices

### 4.1 Supervision target = GT target position

Pulled from `_triangulation_result_gt.position[:, 0, :]` ([iris_ma_env6_test.py:1448](../iris_ma_env6_test.py#L1448)). At `task_reward_level = 1` the FIM is computed with `use_gt_target=True`, so the `.position` field is the GT target world position used as the FIM Taylor-expansion point — not a triangulated estimate. This makes it the right supervision label.

### 4.2 World frame, diagonal covariance

Both `μ_xyz` and `log σ²_xyz` are predicted in world coordinates. Covariance is diagonal — 3 numbers, not a full Cholesky. Justified because [visualization/covariance_ellipsoid.py:137-152](../visualization/covariance_ellipsoid.py#L137-L152) consumes only the diagonal of the 3×3 covariance (renders an axis-aligned ellipsoid). A full Cholesky parameterization would be discarded by the only consumer.

### 4.3 Per-agent validity mask + episode-start composition

The auxiliary loss is masked per-sample by three conditions composed with logical AND:

```
mask[env, agent, t] = (this agent's bbox is non-empty)
                    AND (_triangulation_result_gt.is_valid)
                    AND (episode_step >= episode_start_mask_steps)
```

The first two are about *observability*; the third about *RNN warm-up* and matches the existing policy/value/entropy loss masking in `MAPPO_RNN` ([mappo_rnn.py:819-845](../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L819-L845)). Without the third term, the aux loss would train against samples where the GRU hidden state is still settling — inconsistent with how the policy loss is weighted, and pollutes the head's calibration during the warm-up window. (`burn_in_steps` is currently 0 in the v0 cfg — out of scope; if enabled later, the same slicing applied to policy outputs must apply to the aux head outputs and supervision.) Rationale for the first two terms:

- **Decentralized parity with a centralized estimator.** The head is the deployment-time replacement for the centralized triangulator (Intent 2). A centralized estimator cannot localize the target from a single agent without that agent's bbox; the per-agent mask enforces the same constraint on the head. Training only on samples where the agent has a real measurement keeps the head's σ honest.
- **Avoid confabulation in the no-evidence regime.** Without per-agent masking, an agent with empty bbox would still be supervised whenever any *other* agent saw the target (via inter-agent obs channels). The head would learn to produce confident-looking μ from inter-agent cues alone — at deployment, when those inter-agent cues come through delayed comms, this becomes a hallucinated target.
- **Fail loud, not quiet.** When the mask is False at deployment, the head is in untrained territory. Its outputs are arbitrary by design. Downstream consumers must gate on either predicted σ or the geometric `is_valid` (separately published) before consuming μ.
- **Out-of-scope: cooperative extrapolation.** Using inter-agent observations to localize *despite* an agent's own empty bbox is a v2 ambition. Not in this spec.

**Note**: the per-agent bbox-emptiness signal is already computed at [iris_ma_env6_test.py:1777](../iris_ma_env6_test.py#L1777) (`bbox_empty = (bbox_pixel.abs().sum(dim=-1) < 1e-6).float()`). Reuse it rather than recomputing.

### 4.4 Architecture: shared encoder, no skip connection (Intent 1a)

Action head consumes encoder hidden state `h` only. The (μ, σ) head consumes the same `h`. The action head does NOT receive (μ, σ) as inputs. Avoids:
- **Bootstrap problem** — early in training, σ is uncalibrated and μ is random; the action head conditioning on garbage delays convergence.
- **Calibration corruption** — under a skip connection, the policy loss flows back through (μ, σ) into the encoder via two paths (direct via `h`, and via `[μ, σ]`). The σ-inflation term in NLL is no longer the only force on σ; the policy loss can drive σ to whatever value makes actions easier, breaking calibration.

The alternative architecture (1b — explicit skip connection from head outputs into action head) is intentionally deferred. If the head turns out to be well-calibrated and the policy under-utilizes the encoder's target features, 1b becomes a candidate change in a follow-up ticket.

### 4.5 Per-agent actor head, no critic head

Each agent's actor model has its own head (no parameter sharing across agents — they have separate actor instances under MAPPO). Supervision is the same shared GT target broadcast across agents. The MAPPO critic gets no auxiliary head; the critic already has access to the global state and does not need a per-agent estimator.

### 4.6 Input parity (Intent 2 deployment requirement)

The head's `forward` is a pure function of one agent's actor obs — the same tensor that drives the action head. Enforced:
- **By construction** — the head is a sub-module of the per-agent actor model. It never receives a multi-agent tensor.
- **By test** — `test_input_parity.py` runs the head on a per-agent obs of shape `[B, obs_dim_per_agent]` in isolation and asserts the head produces `[B, 6]` without needing other agents' tensors.

### 4.7 Calibration parity (Intent 2 deployment requirement)

σ must be trustworthy under the same delay/noise pipeline that runs at deployment. NLL-on-GT achieves this automatically because the training input passes through the delay system already (`DelaySystem.get_states_for_observations`, [iris_ma_env6_test.py:1763-1770](../iris_ma_env6_test.py#L1763-L1770)). The acceptance criterion in the ticket explicitly evaluates calibration **under the active delay pipeline** to enforce this.

### 4.8 Operating regime: `task_reward_level = 1`

Supervision uses only `_triangulation_result_gt`, which is unconditionally computed at line 1448. `_tri_result_l2` and `_tri_result_l3` are `None` at this level ([iris_ma_env6_test.py:1452-1468](../iris_ma_env6_test.py#L1452-L1468)). Forward-compatible: bumping `task_reward_level` later does not change the supervision pipeline.

---

## 5. Output Contract

Each agent's head emits:

| Field | Shape | Frame | Meaning |
|-------|-------|-------|---------|
| `μ_xyz` | `[3]` | World | Mean of predicted Gaussian over target position |
| `log σ²_xyz` | `[3]` | World | Per-axis log-variance, clamped to `[-10, 4]` |

A consumer interprets the pair as a Gaussian measurement `N(target | μ, diag(exp(log σ²)))`. Recommended consumer behaviors:

- **Inverse-variance fusion across agents:** `μ_fused = (Σ Σᵢ⁻¹)⁻¹ Σ Σᵢ⁻¹ μᵢ`, valid under the Gaussian-independence assumption.
- **Best-agent selection:** pick the agent with smallest `tr(Σᵢ)`.
- **Kalman update:** use `(μᵢ, diag(exp(log σ²ᵢ)))` directly as a measurement update.
- **Fallback gate:** if all agents have `tr(Σ) > τ_threshold`, hand off to a default behavior (hover, search pattern, etc.).

The geometric `is_valid` flag may also be published separately to support deterministic gating, since predicted σ is a calibrated-uncertainty estimate but not a hard observability indicator.

### 5.1 Deployment statefulness (GRU)

Because the encoder backbone is `MLP + GRU`, the head's outputs at time *t* depend on the agent's observation history through the GRU hidden state. Implications for the deployment runtime:

- **GRU state is per-agent and persistent across the inference loop.** Each drone maintains its own GRU hidden state, initialized to zeros at episode start, and feeds it forward at every inference step. Calling the network statelessly (re-initializing GRU at every step) is **incorrect** — it discards the temporal accumulation that the head was trained on, and σ calibration breaks.
- **At t=0, σ may be larger than at t > k.** The head has only the current obs to condition on at t=0; over time, the GRU integrates more evidence and σ tightens. This is the desired behavior; downstream consumers should not be surprised by σ varying smoothly over the episode.
- **A drone that crashes and respawns must zero its own GRU state.** Mirrors the training-time reset behavior at episode termination ([mappo_rnn.py:385-397](../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L385-L397)).

---

## 6. SKRL Integration Pattern

The head and its loss integrate with SKRL via the canonical extension hooks — no monkey-patching, no forking. All four pieces follow patterns already used by [scripts/reinforcement_learning/skrl/mappo_rnn.py](../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py) (the v0 trainer base).

### 6.1 Auxiliary outputs via `compute()`'s `extra_dict`

SKRL's `Model.compute()` returns `(output, log_std, extra_dict)`. The third element is the canonical place for auxiliary outputs. `MAPPORNNPolicy.compute()` already populates `{"rnn": [hidden_states]}`; we extend it:

```python
def compute(self, inputs, role):
    output, hidden_states = self.compute_base(inputs)
    policy_output = self.policy_layer(output)
    tri_out = self.tri_head(output)  # output = post-GRU hidden state
    return policy_output, self.log_std_parameter, {
        "rnn": [hidden_states],
        "tri_mu":      tri_out[..., :3],
        "tri_log_var": tri_out[..., 3:].clamp(-10.0, 4.0),
    }
```

The trainer reads `policy_outputs["tri_mu"]` and `policy_outputs["tri_log_var"]` after `policy.act(...)` returns.

### 6.2 Supervision plumbing: env method → trainer wrapper → `infos` → memory tensor

Three steps, mirroring the existing `shared_states` pattern at [train_mappo_rnn_hydra.py:230-231](../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py#L230-L231):

**Step 1 — Env exposes a public method:**
```python
# In iris_ma_env6_test.py
def get_aux_supervision(self) -> dict[str, torch.Tensor]:
    """Pure READ on env state. Returns per-agent supervision for the policy tri head."""
    return {
        "tri_target_position_w": self._triangulation_result_gt.position[:, 0, :].unsqueeze(1).expand(-1, self.num_agents, -1).contiguous(),
        "tri_target_valid":      self._compose_per_agent_validity(),
    }
```

**Step 2 — Trainer wrapper writes to `infos`** (in `train_mappo_with_aux_hydra.py`'s training loop):
```python
next_states, rewards, terminated, truncated, infos = self.env.step(actions)
aux = self.env.unwrapped.get_aux_supervision()
infos["tri_target_position_w"] = aux["tri_target_position_w"]
infos["tri_target_valid"] = aux["tri_target_valid"]
self.agents.record_transition(..., infos=infos, ...)
```

**Step 3 — Custom MAPPO subclass registers + extracts**:
1. In `init(trainer_cfg)`:
   ```python
   for uid in self.possible_agents:
       self.memories[uid].create_tensor(name="tri_target_position_w", size=3, dtype=torch.float32)
       self.memories[uid].create_tensor(name="tri_target_valid",      size=1, dtype=torch.float32)
   ```
2. In `record_transition(infos=...)`:
   ```python
   for agent_idx, uid in enumerate(self.possible_agents):
       self.memories[uid].add_samples(
           tri_target_position_w=infos["tri_target_position_w"][:, agent_idx, :],
           tri_target_valid=infos["tri_target_valid"][:, agent_idx:agent_idx+1].float(),
       )
   ```
3. In `_update(...)`, retrieve and apply the **same `reshape_to_sequences` helper** that produces the policy minibatches ([mappo_rnn.py:549-570](../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L549-L570)) — guarantees alignment between aux supervision and policy outputs.

**Why the env doesn't write `infos` directly**: SKRL's runner logs `extras` to tensorboard but not `infos`. The trainer-wrapper-injects-infos pattern is the existing convention here; we follow it. May be revisited if the runner-side logging story changes.

**Silent-failure mode**: SKRL's `Memory.add_samples(**kwargs)` ignores keys not registered via `create_tensor` (no error, no warning). `test_memory_tensor_registration.py` and `test_preflight_smoke.py` together guard against this.

### 6.3 Loss injection via `_update()` override

Subclass `MAPPO_RNN` and re-implement (or hook into) the per-minibatch loop in [mappo_rnn.py:417-959](../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L417). After the `policy.act(...)` call (~L785-787), read `policy_outputs["tri_mu"]` and `policy_outputs["tri_log_var"]`. Compose the per-agent validity mask with the existing `episode_mask`:

```python
episode_mask = (sampled_episode_step.squeeze(-1) >= self._episode_start_mask_steps[uid])
combined_mask = sampled_tri_target_valid.squeeze(-1).bool() & episode_mask

per_dim_nll = 0.5 * ((sampled_tri_target_position_w - tri_mu).pow(2) * (-tri_log_var).exp() + tri_log_var)
per_dim_nll = per_dim_nll.clamp(max=cfg.get("aux_nll_clamp", 1000.0))   # numerical guard
nll = per_dim_nll.sum(dim=-1)
denom = combined_mask.float().sum().clamp_min(1.0)
aux_loss = (nll * combined_mask.float()).sum() / denom

total_loss = policy_loss + entropy_loss + value_loss + aux_loss_scale * aux_loss
self.scaler.scale(total_loss).backward()
```

The bit-exact regression test (`aux_loss_scale = 0.0` vs `tri_head.enabled = false`) catches any divergence from `MAPPO_RNN`'s original loss math. The NLL clamp guards against the rare case where `log_var` approaches the lower clamp boundary while `(target − mu)` is large at init.

### 6.4 Logged metrics

The `_update` override also logs aux-specific scalars via `self.track_data(...)` so they appear in tensorboard/wandb alongside policy/value losses:

| Scalar | Definition |
|--------|------------|
| `aux/nll` | masked-mean NLL (the loss value before `aux_loss_scale`) |
| `aux/predicted_rmse` | `sqrt(((tri_target − tri_mu) ** 2 * combined_mask).sum() / (3 * denom))` — mean per-axis RMSE on valid samples |
| `aux/calibration_2sigma_coverage` | empirical fraction of GT inside the predicted ±2σ-per-axis box, restricted to valid samples |
| `aux/effective_sample_fraction` | `combined_mask.float().mean()` — drops as the curriculum grows delays |

### 6.5 Calling Contract (per CLAUDE.md §"Stateful Component Rules")

| Method | Type | Frequency | Idempotent | Lifecycle hook |
|--------|------|-----------|------------|----------------|
| `MAPPOWithAuxPolicy.compute(obs, ...)` | READ | Once per actor forward pass | Yes (pure function of weights + obs) | Inside skrl rollout collection and during gradient steps |
| Env writes `infos["tri_target_position_w"]` | WRITE | Once per env step | Guarded by being co-located with the existing `_triangulation_result_gt` write | `_post_physics_step` at [iris_ma_env6_test.py:1448](../iris_ma_env6_test.py#L1448) |
| Env writes `infos["tri_target_valid"]` | WRITE | Once per env step | Same as above | Same as above |
| `MAPPOWithAux.record_transition(infos=...)` | WRITE (memory) | Once per env step | Routes per-agent slices to per-agent memory | skrl trainer hook |
| `MAPPOWithAux._update` aux loss step | WRITE (gradients) | Once per gradient step | N/A (training step) | skrl trainer hook |

The head's `compute` is a pure READ. The env's WRITE of supervision tensors happens at the same lifecycle hook as the existing geometric triangulation compute — no new stateful surface in the env. The trainer's WRITE of memory and gradients happens entirely inside skrl's existing lifecycle.

---

## 7. Out of Scope

Explicitly NOT addressed in this spec (deferred to follow-up tickets if/when needed):

- **Skip connection (Intent 1b)** — feeding (μ, σ) forward into the action head as features. Deferred.
- **Cooperative extrapolation** — relaxing the validity mask to enable single-agent-sees scenarios. Deferred.
- **Critic-side auxiliary head.** Critic has global state; no clear added value.
- **Full Cholesky covariance.** Requires upgrading [visualization/covariance_ellipsoid.py](../visualization/covariance_ellipsoid.py) to consume off-diagonal terms.
- **Non-stationary supervision (L2/L3).** Considered and rejected — see ticket §"Why these intents converge on NLL-on-GT."
- **Recurrent or multi-step head.** Single-step feedforward only.
- **Bootstrapping the head with pre-computed geometric estimates** (e.g., warm-starting μ from `_tri_result_l2`). Could be useful but adds coupling to higher task levels; left for a follow-up if convergence speed is a problem.

---

## 8. Acceptance References

For the test suite, training-run gates, and bit-exact regression criteria, see [active/ticket/031-policy-triangulation-head/ticket.md §"Acceptance criteria"](active/ticket/031-policy-triangulation-head/ticket.md). This spec defines *what to build*; the ticket defines *what to verify*.