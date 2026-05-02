## Ticket 031 — Policy head for distributed target estimation (position + uncertainty)

**Status**: Open
**Created**: 2026-05-02
**Target env**: v0 ([iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) + [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py)).
**Operating regime**: `task_reward_level = 1` ([iris_ma_env6_test_cfg.py:496-499](../../../../iris_ma_env6_test_cfg.py#L496-L499)).
**Trainer base**: subclass `MAPPO_RNN` ([scripts/reinforcement_learning/skrl/mappo_rnn.py](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py)) — the actively-used variant for v0 (`skrl_mappo_rnn_cfg_entry_point`).

**What**: Add an auxiliary regression head to the (parameter-shared) actor network that predicts the **target's 3D world-frame position** and **diagonal log-variance** (6 outputs: `μ_xyz, log σ²_xyz`) from each agent's actor observation. Train it with a Gaussian NLL against the GT target position, masked by `(this agent's bbox is non-empty) AND (_triangulation_result_gt.is_valid) AND (episode_step >= episode_start_mask_steps)`. Architecture: the head shares the encoder backbone (MLP + GRU) with the action head; the action head consumes the encoder hidden state, **not** the head's outputs (no skip connection — implicit reasoning, see Intent 1). At deployment, each agent runs the same shared network on its own obs to produce (μ_i, σ_i) — replacing the centralized geometric triangulator (see Intent 2).

### Two primary intents

#### Intent 1 — Train the policy to reason about target + uncertainty (implicit, via shared encoder)

The auxiliary supervision shapes the encoder. The action head consumes the same encoder hidden state that produces (μ, σ), so target-reasoning features get gradient pressure from both the policy loss *and* the supervision loss. The action head **never directly sees (μ, σ)** — no skip connection. The encoder is forced to learn target-aware features; the action head learns to consume them implicitly. This is the conservative choice: no bootstrap problem (action head never conditions on uncalibrated head outputs), no risk of policy loss corrupting the head's calibration via shared parameters in a multi-output skip architecture.

#### Intent 2 — Replace the centralized triangulator at deployment with a distributed estimator

At deployment, **each agent's policy network is a self-contained estimator** that emits (μ_i, σ_i) using only that agent's per-agent actor observation. The geometric triangulator ([_compute_triangulation](../../../../iris_ma_env6_test.py#L1198), [triangulation/triangulation.py](../../../../triangulation/triangulation.py)) becomes a **training-time-only artifact**. Hard constraints this imposes:

- **Input parity.** The head's `forward` must be a pure function of one agent's actor obs (the same obs that drives the action head). No GT, no centralized scene state, no other agent's head output. Inter-agent observations are allowed only via the comms-delayed channels already in the actor obs.
- **Calibration parity.** The head's σ must be trustworthy under the same delay/noise pipeline that runs at deployment. NLL-on-GT achieves this automatically because the training input passes through the delay system already (`DelaySystem.get_states_for_observations`, [iris_ma_env6_test.py:1763-1770](../../../../iris_ma_env6_test.py#L1763-L1770)) — σ calibrates against the actual head error in those conditions.
- **Per-agent independence.** Each agent's head runs forward independently. No head depends on another agent's head output. (Cooperative fusion, if desired, happens *outside* the head — downstream.)
- **Output contract.** Each agent emits (μ_i, σ_i) as a well-typed Gaussian measurement: 3D mean in world coordinates, diagonal covariance `diag(exp(log σ²_xyz))`. Consumers can pick the agent with smallest σ, fuse via inverse-variance weighting, hand off to a Kalman filter as a measurement update, or fall back to a default behavior when σ is too large.

### Why these two intents converge on NLL-on-GT (rather than supervising on geometric-pipeline error)

Considered earlier: supervise σ against `(_tri_result_l3.position - GT)²` ("predict the centralized triangulator's error"). Rejected because:
- **Intent 1 misalignment.** The policy reasons about *the head's* μ via the encoder. So σ should answer "how wrong is the head likely to be?" — not "how wrong is some external estimator the policy never sees?"
- **Intent 2 misalignment.** L3 is the centralized triangulator we are *replacing* at deployment. Calibrating σ against an estimator that won't run online is misaligned with the deployment-relevant quantity, which is `(head_μ - GT)`.
- **Operating regime.** At `task_reward_level = 1`, [_tri_result_l2 / _tri_result_l3](../../../../iris_ma_env6_test.py#L1452-L1468) are not even computed (they're gated on `task_level >= 2 or curriculum_task_levels`). NLL-on-GT uses only `_triangulation_result_gt`, which is unconditionally available.

NLL-on-GT trains the head to be a calibrated estimator of the truth, end-to-end-correctly under whatever delays/noise the training rollouts contain. That's exactly what both intents need.

### Background — what already exists in v0

- **Trainer in use**: `MAPPO_RNN` from [scripts/reinforcement_learning/skrl/mappo_rnn.py](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py) (1097 lines, custom subclass of skrl's `MAPPO`). Models are `MAPPORNNPolicy` and `MAPPORNNValue` — instantiated **directly in Python** (not via skrl's YAML model_instantiator). Training entry: [scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py).
- **No triangulation in observations.** Actor obs is 31D ego + 16D × (num_agents − 1) inter-agent. The optional 6D tri tail is gated off (`enable_triangulation: bool = False`, [iris_ma_env6_test_cfg.py:409](../../../../iris_ma_env6_test_cfg.py#L409)). Critic likewise has no triangulation channel.
- **Geometric triangulation lives in [triangulation/triangulation.py](../../../../triangulation/triangulation.py)**, returning `TriangulationResult(position [N,T,3], covariance [N,T,3,3], quality_metric, is_valid, condition_number, num_valid_cameras)` ([triangulation.py:44-61](../../../../triangulation/triangulation.py#L44-L61)).
- **At `task_reward_level = 1`, only `_triangulation_result_gt` is computed** ([iris_ma_env6_test.py:1448](../../../../iris_ma_env6_test.py#L1448)) with `use_gt_target=True`. So `.position` is the GT target world position (used as the FIM Taylor-expansion point), `.covariance` is the FIM-proxy covariance, and `.is_valid` reflects whether the geometric solve would have succeeded. `_tri_result_l2 / _tri_result_l3` are `None` in this regime.
- **Per-agent bbox emptiness is already computed** at [iris_ma_env6_test.py:1777](../../../../iris_ma_env6_test.py#L1777): `bbox_empty = (bbox_pixel.abs().sum(dim=-1) < 1e-6).float()`. This is the source for the per-agent half of the validity mask.
- **Parameter-sharing across agents.** [train_mappo_rnn_hydra.py:298-329](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py#L298-L329) instantiates `shared_policy = MAPPORNNPolicy(...)` once and maps the same Python reference to every agent in `models[agent_id]`. There is **one** policy network used by all agents — gradients from each agent's loss flow into the same parameters. Per-agent independence is preserved at inference (each agent runs forward on its own obs); shared params are a training-time property only.
- **SKRL extension hooks confirmed**:
  - `Model.compute()` returns `(output, log_std, extra_dict)` — auxiliary outputs go in `extra_dict` ([mappo_rnn.py:1081-1083](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L1081-L1083) already uses this pattern with `{"rnn": [hidden_states]}`).
  - `MAPPO_RNN._update()` is the override point for adding auxiliary loss to the gradient step ([mappo_rnn.py:417-959](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L417)).
  - Memory tensors must be **explicitly registered** via `memory.create_tensor(name=..., size=..., dtype=...)` in `init()`. Custom keys in `infos` are NOT auto-captured (silent failure mode).
  - `record_transition(infos=...)` is the env→memory ingest path; override to call `memory.add_samples(tri_target_position_w=infos["tri_target_position_w"][:, agent_idx, :], ...)`.
  - The trainer wrapper currently injects `infos["shared_states"]` after `env.step()` ([train_mappo_rnn_hydra.py:230-231](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py#L230-L231)) by calling `self.env.state()`. This ticket follows the same pattern: env exposes `get_aux_supervision()`; trainer wrapper writes the result into `infos`.
  - `MAPPO_RNN` already masks losses by `episode_step >= episode_start_mask_steps` ([mappo_rnn.py:819-845](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L819-L845)); aux loss must compose with this same mask. `burn_in_steps` is currently unused (defaults to 0) — out of scope.

### Design choices (decided up front; do not re-litigate during implementation)

1. **Supervision target = GT target position.** Pulled from `_triangulation_result_gt.position[:, 0, :]` (correct because `use_gt_target=True` makes that field the GT eval point). NLL provides calibrated σ implicitly.
2. **Frame = world.** Both `μ_xyz` and `log σ²_xyz` in world coordinates. No heading-frame transform.
3. **Covariance form = diagonal.** Output `log σ²_xyz` (3 numbers) → `Σ = diag(exp(log σ²))`. Justified because [visualization/covariance_ellipsoid.py:137-152](../../../../visualization/covariance_ellipsoid.py#L137-L152) only consumes the diagonal of the 3×3 covariance.
4. **Auxiliary loss mask** = `(agent's bbox non-empty) AND (_triangulation_result_gt.is_valid) AND (episode_step >= episode_start_mask_steps)`. The first two enforce the no-confabulation property (decentralized head, like a centralized estimator, cannot estimate from a single agent without a bbox). The third composes with `MAPPO_RNN`'s existing episode-start masking ([mappo_rnn.py:819-845](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L819-L845)) so the aux loss skips the same warm-up steps as policy/value/entropy losses. Downstream consumers at deployment must gate the head's μ on either predicted σ or `is_valid`.
5. **Architecture = shared encoder, no skip connection (Intent 1a).** Action head consumes the post-GRU hidden state. The (μ, σ) head consumes the same hidden state. The action head does NOT receive (μ, σ) as direct inputs. Avoids the bootstrap problem and keeps the head's calibration honest. Encoder is **MLP + GRU** (matching `MAPPORNNPolicy`'s structure).
6. **One shared head across agents, no critic head.** With parameter-sharing ([train_mappo_rnn_hydra.py:298-329](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py#L298-L329)), there is exactly one `tri_head` module — the same parameters generate (μ_i, σ_i) for every agent's obs. Each agent's NLL contributes gradients to the same params via that agent's `_update` inner loop. The head learns to be input-invariant across agents (each (env, t) yields N obs all paired with the same GT target — a useful generalization signal). No critic-side head.
7. **Input parity (Intent 2).** The head's `forward` is a pure function of one agent's actor obs — the very same tensor `obs[agent_id]` that drives the action head. Enforced by construction (the head is a sub-module of the per-agent actor model) and by an explicit test (see step 6).
8. **Operating regime = `task_reward_level = 1`.** Supervision uses only `_triangulation_result_gt`. Forward-compatible with higher levels — flipping `task_reward_level` later does not change the supervision pipeline, since `_triangulation_result_gt` is unconditionally computed.
9. **SKRL integration = subclass `MAPPO_RNN` + custom model class** — no monkey-patching, no forking. Auxiliary outputs flow via `compute()`'s `extra_dict`; supervision tensors flow via `infos` dict + memory's registered tensor API.

### Workflow

1. **Custom actor model + custom MAPPO subclass** — add [scripts/reinforcement_learning/skrl/mappo_with_aux.py](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_with_aux.py) (alongside `mappo_rnn.py`):
   - **`MAPPOWithAuxPolicy`** — subclasses `MAPPORNNPolicy`. Adds a `tri_head` module: `MLP(128, elu) → 6` consuming the post-GRU hidden state (same point `policy_layer` reads from). `compute(...)` returns:
     ```python
     return policy_output, self.log_std_parameter, {
         "rnn": [hidden_states],
         "tri_mu":      tri_out[..., :3],
         "tri_log_var": tri_out[..., 3:].clamp(-10.0, 4.0),
     }
     ```
     **Do NOT override `act()`** — rely on `GaussianMixin.act()` to wrap `compute()` and produce `log_prob`; auxiliary outputs flow through `extra_dict` only.
   - **`MAPPOWithAux`** — subclasses `MAPPO_RNN`. Overrides:
     - `init(trainer_cfg)`: call `super().init()`, then for each agent register memory tensors `tri_target_position_w` (size 3) and `tri_target_valid` (size 1) via `memory.create_tensor(...)`.
     - `record_transition(states, actions, ..., infos)`: call `super().record_transition(...)`, then for each agent extract `infos["tri_target_position_w"][:, agent_idx, :]` (shape `(num_envs, 3)`) and `infos["tri_target_valid"][:, agent_idx:agent_idx+1]` (shape `(num_envs, 1)`) and pass to that agent's `memory.add_samples(...)`.
     - `_update(timestep, timesteps)`: re-implement (or hook into) the per-minibatch loop. After the `policy.act(...)` call (~L785-787), read `policy_outputs["tri_mu"]` and `policy_outputs["tri_log_var"]`. Retrieve `tri_target_position_w` and `tri_target_valid` from memory using the **same `reshape_to_sequences` helper** that the existing code uses (L549-570) — guarantees alignment with policy outputs. Compute aux NLL (formula in step 2). Add `aux_loss_scale * aux_loss` to total loss before `.backward()` at L901. Log `aux/nll`, `aux/predicted_rmse`, `aux/calibration_2sigma_coverage`, `aux/effective_sample_fraction` via `self.track_data(...)`.
   - **`MAPPOWithAuxValue`** — identical to `MAPPORNNValue`, no aux head (per design choice 6). Re-export for symmetry.
   - **`MAPPO_WITH_AUX_DEFAULT_CONFIG`** — copy `MAPPO_RNN_DEFAULT_CONFIG`, add `aux_loss_scale: 0.1`, `aux_log_var_clamp: [-10.0, 4.0]`, `aux_nll_clamp: 1000.0`.
   - **No skip connection** — the action layer's input dim equals the post-GRU hidden dim, NOT hidden_dim + 6. Enforced by test.

2. **Auxiliary loss formula** (computed inside the overridden `_update`, after sequence reshape and burn-in slicing if applicable):
   ```python
   # mu, log_var: same shape as policy outputs after reshape_to_sequences (e.g., [N*num_seq, seq_len, 3])
   # tri_target, tri_valid: pulled from memory via the SAME reshape helper that produces the policy minibatches
   mu          = policy_outputs["tri_mu"]
   log_var     = policy_outputs["tri_log_var"]                  # already clamped at compute()
   tri_target  = sampled_tri_target_position_w                  # post-reshape, post-burn-in
   tri_valid   = sampled_tri_target_valid.squeeze(-1).bool()    # bool mask

   # Compose with the same episode-start mask that policy/value losses use (line 819-845 of mappo_rnn.py)
   episode_mask = (sampled_episode_step.squeeze(-1) >= self._episode_start_mask_steps[uid])
   combined_mask = tri_valid & episode_mask                     # bool, same shape as policy outputs

   per_dim_nll = 0.5 * ((tri_target - mu).pow(2) * (-log_var).exp() + log_var)
   per_dim_nll = per_dim_nll.clamp(max=cfg.get("aux_nll_clamp", 1000.0))   # numerical guard
   nll = per_dim_nll.sum(dim=-1)                                # per-sample
   denom = combined_mask.float().sum().clamp_min(1.0)
   aux_loss = (nll * combined_mask.float()).sum() / denom

   total_loss = policy_loss + entropy_loss + value_loss + aux_loss_scale * aux_loss
   ```
   When `combined_mask.sum() == 0` for a minibatch, `aux_loss == 0` (well-defined, no NaN). When `aux_loss_scale == 0`, the aux contribution is zero — wired-but-off regression gate.

3. **Plumb supervision via env-exposed method + trainer-side injection** — mirrors the existing `shared_states` pattern ([train_mappo_rnn_hydra.py:230-231](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py#L230-L231)). Two pieces:

   **(3a) Env-side**: add a public method `IrisMA6TestEnv.get_aux_supervision() -> dict[str, torch.Tensor]` to [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py). Returns:
   - `"tri_target_position_w"`: shape `[N, A, 3]`. `_triangulation_result_gt.position[:, 0, :]` (GT target world position) broadcast across the agent dimension.
   - `"tri_target_valid"`: shape `[N, A]`. Computed as `(per_agent_bbox_nonempty[N, A]) AND (_triangulation_result_gt.is_valid[:, 0:1])`. Per-agent bbox emptiness reuses the existing logic at [iris_ma_env6_test.py:1777](../../../../iris_ma_env6_test.py#L1777).

   The method is a pure READ on env state (idempotent within a sim step). It is *not* called from inside `_post_physics_step` — it is called by the trainer wrapper after `env.step()`, the same way `env.state()` is called for `shared_states`.

   **(3b) Trainer-side**: in [train_mappo_with_aux_hydra.py](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_with_aux_hydra.py)'s training loop, after `env.step(...)` and before `agents.record_transition(...)`:
   ```python
   aux = env.unwrapped.get_aux_supervision()
   infos["tri_target_position_w"] = aux["tri_target_position_w"]
   infos["tri_target_valid"] = aux["tri_target_valid"]
   ```
   `MAPPOWithAux.record_transition` then indexes `infos[key][:, agent_idx, ...]` per agent and routes to that agent's memory.

   **Why not write directly inside the env's `step()`**: `infos` is not surfaced to tensorboard logging by skrl's runner (it logs `extras` instead). The existing pattern is for the trainer wrapper to inject supervision into `infos`; we follow it. (May be revisited in a later stage if the runner-side logging makes a different path preferable.)

4. **New YAML config** — add [agents/skrl_mappo_rnn_aux_cfg.yaml](../../../../agents/skrl_mappo_rnn_aux_cfg.yaml). Header comment: "Based on skrl_mappo_rnn_cfg.yaml as of 2026-05-02." Adds `tri_head` and `aux_loss_scale`:
   ```yaml
   models:
     policy:
       # ... fields from skrl_mappo_rnn_cfg.yaml, unchanged ...
       tri_head:
         enabled: true
         hidden: [128]
         log_var_clamp: [-10.0, 4.0]
   agent:
     # ... fields from skrl_mappo_rnn_cfg.yaml, unchanged ...
     aux_loss_scale: 0.1
     aux_nll_clamp: 1000.0
   ```
   `enabled: false` removes the head entirely (no params, no auxiliary loss). `aux_loss_scale: 0.0` keeps the head wired but contributes zero gradient to shared params — the bit-exact regression gate. The original `skrl_mappo_rnn_cfg.yaml` is left untouched so pre-aux runs remain reproducible.

5. **New training entry point** — add [scripts/reinforcement_learning/skrl/train_mappo_with_aux_hydra.py](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_with_aux_hydra.py), structurally mirroring `train_mappo_rnn_hydra.py`. Differences:
   - Imports `MAPPOWithAux, MAPPO_WITH_AUX_DEFAULT_CONFIG, MAPPOWithAuxPolicy, MAPPOWithAuxValue` from `mappo_with_aux`.
   - Decorates `main` with `@hydra_task_config(args_cli.task, "skrl_mappo_rnn_aux_cfg_entry_point")`.
   - In the training loop, after `env.step(...)`, calls `env.unwrapped.get_aux_supervision()` and writes the result into `infos` (see step 3b).
   - Otherwise identical lifecycle.

6. **Spec doc** — already drafted at [doc/policy_triangulation_head_spec.md](../../../policy_triangulation_head_spec.md). Update to reflect:
   - Per-agent masking (§4.3).
   - SKRL integration pattern (new section, references this ticket's workflow).
   - Architecture diagram now showing MLP + GRU encoder.
   - File layout: `scripts/reinforcement_learning/skrl/mappo_with_aux.py` and `train_mappo_with_aux_hydra.py`.

7. **Tests — split by dependency** (pure-PyTorch tests skip AppLauncher for fast iteration):

   **(7a) Pure-PyTorch unit tests** at [scripts/reinforcement_learning/skrl/tests/policy_tri_head/](../../../../../../../../../scripts/reinforcement_learning/skrl/tests/policy_tri_head/) — runner: `run_tests.py` (no AppLauncher, ~5s):
   - `test_actor_with_tri_head_shape.py`: build `MAPPOWithAuxPolicy`, forward with seq dim, assert `policy_output` / `tri_mu` / `tri_log_var` shapes. Assert `log_var` is within the configured clamp range after random-weight forward.
   - `test_no_skip_connection.py`: inspect `policy_layer.in_features` — assert it equals the post-GRU hidden dim, NOT hidden_dim + 6. Regression guard against turning Intent 1a into 1b.
   - `test_input_parity.py` (Intent 2): pass per-agent obs of shape `[B, S, obs_dim_per_agent]` in isolation; assert `tri_mu` shape `[B, S, 3]` without requiring other agents' tensors.
   - `test_aux_loss_masking.py`: synthetic batch with mixed mask. Confirm `aux_loss` equals masked-mean NLL over valid samples. With `mask.sum()==0`, return `aux_loss == 0` with no NaN. Also test the `episode_step >= episode_start_mask_steps` composition: when `episode_step < threshold`, sample's mask becomes False even if `tri_target_valid` is True.
   - `test_invalid_triangulation_no_grad.py`: batch with `mask` all-False; backward through the auxiliary loss; assert `aux_loss == 0` AND gradient norm on `tri_head` params is zero.
   - `test_memory_tensor_registration.py`: instantiate `MAPPOWithAux` with mock env, call `init(trainer_cfg)`, assert each per-agent memory has `tri_target_position_w` (size 3) and `tri_target_valid` (size 1) tensors registered via `memory.tensors_keys`.
   - `test_loss_scale_zero_regression.py`: with `aux_loss_scale = 0.0`, one full update step produces **bit-exact** gradients on shared parameters (encoder MLP + GRU + policy_layer + value_layer) vs a baseline run with `tri_head.enabled = false`. Same seed; RNG equalized at model-construction time (re-seed immediately before `policy_layer` init in the head-enabled run so the action layer's weights match bit-exactly). The bit-exact comparison excludes the head's own params (they don't exist in baseline).

   **(7b) Env-integration tests** at [iris_ma6/tests/policy_tri_head/](../../../../tests/policy_tri_head/) — runner: `run_integration_tests.py` (AppLauncher-prefixed, ~30s):
   - `test_world_frame_supervision.py`: launch env mock with `tri_pos_w = (10, 5, 0)`, `_triangulation_result_gt.is_valid = True`, `agent_0.bbox_nonempty = True`, `agent_1.bbox_nonempty = False`. Call `env.get_aux_supervision()`; assert `tri_target_position_w[:, 0, :] == (10, 5, 0)` for all agents (broadcast), `tri_target_valid[:, 0] == True`, `tri_target_valid[:, 1] == False`. With `_triangulation_result_gt.is_valid = False`, assert `tri_target_valid` is False for all agents regardless of bbox.
   - `test_preflight_smoke.py`: launch v0 env with `Isaac-Iris-MA6-Direct-Test-v0` task and `skrl_mappo_rnn_aux_cfg.yaml` (with `aux_loss_scale = 0.0`). Run **at least 128 env steps** with the full `MAPPOWithAux` trainer (rollout collection + at least one `_update` call). Assert: (a) no exception thrown across the 128 steps, (b) no NaN/Inf in any model parameter or gradient after the gradient update, (c) `tri_target_position_w` and `tri_target_valid` tensors in memory are populated with non-zero entries (not silently dropped due to unregistered key — catches the silent-failure mode of `add_samples`), (d) gradient norms on `policy_layer` + `value_layer` are non-zero (PPO loss flowing). End-to-end smoke test before any full training run.

### Scope boundary

- DO: actor-side per-agent auxiliary head; world-frame; diagonal log-variance; GT-target supervision; **per-agent** validity mask (own bbox non-empty AND scene `is_valid`).
- DO: Intent 1a — shared encoder, no skip connection from head outputs to action head.
- DO: Intent 2 — head is a pure function of per-agent actor obs; deployable as the standalone distributed estimator.
- DO: subclass `MAPPO_RNN` and `MAPPORNNPolicy`. Place new files under `scripts/reinforcement_learning/skrl/`.
- DO: keep `tri_head.enabled` and `aux_loss_scale` configurable.
- DO: write `infos["tri_target_*"]` in the same lifecycle hook as `_triangulation_result_gt` (line 1448, no new compute).
- DO NOT: feed the head's output back into the action head's input. (That would be Intent 1b, intentionally rejected.)
- DO NOT: add a critic-side head, an RNN-based head **separate from the encoder** (the GRU is shared with the action head), or a hierarchical head.
- DO NOT: enable the optional 6D tri tail observation (`enable_triangulation`). Whether the actor *also* gets a triangulation observation is an orthogonal question; this ticket is about the head, not the input.
- DO NOT: use `_tri_result_l2 / _tri_result_l3` as supervision targets. Both are `None` at `task_reward_level = 1` and using them would couple this ticket to higher task levels.
- DO NOT: switch to full-Cholesky covariance unless [visualization/covariance_ellipsoid.py](../../../../visualization/covariance_ellipsoid.py) is upgraded to consume off-diagonal terms.
- DO NOT: relax the per-agent validity mask to enable cooperative extrapolation (e.g., letting an empty-bbox agent be supervised because some other agent saw the target). Out of scope.
- DO NOT: change the geometric triangulator's lifecycle. It continues to run during training (it's the supervision source). The "training-time only" framing applies to *deployment*, not to *training*.
- DO NOT: monkey-patch skrl internals or vendor a copy of skrl files. Use the canonical subclass + override pattern only.

### Acceptance criteria

- All eight tests in step 7 pass.
- Single training run (Iris-MA6-v_test, default 2 agents, default config, `task_reward_level = 1`) with `tri_head.enabled=true, aux_loss_scale=0.1` for 200k env-steps. Compare to a paired baseline run with `tri_head.enabled=false` (head removed entirely; same seed):
  - Aux NLL on a held-out validation rollout decreases monotonically over training (no divergence, no NaN).
  - Mean episode return at 200k is **within ±5%** of the baseline. Do-no-harm gate. Improvements often surface later; not required at this gate.
  - On `tri_target_valid==True` validation samples evaluated **under the active delay pipeline** (Intent 2 calibration parity), predicted-position RMSE in world frame is bounded — concretely, ≤ 1.5× the empirical std of the training distribution's GT target positions. Sanity floor: the head must beat the trivial "predict the dataset mean" baseline.
  - Calibration spot-check on `tri_target_valid==True` validation samples (under delay): empirical fraction of GT positions falling inside the head's predicted ±2σ-per-axis box is in `[0.85, 1.0]` (Gaussian ideal: ≈0.95³ ≈ 0.86 for 3 independent axes). Loose bounds, just to flag gross miscalibration.
- Ablation: `aux_loss_scale = 0.0` (head wired, zero gradient) produces **bit-exact** gradients on the **shared parameters** (encoder MLP + GRU + action head + value head) vs the head-removed baseline (`tri_head.enabled = false`). Both runs use the same seed; the head's param-init RNG draws are equalized by re-seeding immediately before the action head's init in the head-enabled run. Wired-but-off regression gate.

### Risk

Medium. Three load-bearing pieces:
1. **`_update()` override** — re-implementing the inner per-minibatch loop without diverging from `MAPPO_RNN`'s sequence/burn-in masking semantics. Plan: make the override call `super()` where possible and only inject the aux loss term; if `super()` doesn't expose the right hook, copy the inner loop verbatim and inject. The bit-exact regression gate catches any divergence.
2. **Calibration under delay (Intent 2)** — the calibration spot-check must run under the delay pipeline that deployment will use, not in a noise-free eval. If σ ends up calibrated only in the no-delay regime, the deployment guarantee fails silently. The calibration test in step 7 is the contract; the acceptance criterion explicitly evaluates under the active delay pipeline.
3. **Memory tensor registration** — skrl's memory does NOT auto-capture `infos` keys. Forgetting to register or to call `add_samples` is a silent failure (the supervision tensors stay zeros, NLL trains the head to predict zeros). The `test_memory_tensor_registration.py` test guards against this.

### Coupling

- Independent of mas/035, mas/036 (gimbal rate-loop / dead-time) — those are sim2real fidelity work; this is policy architecture.
- Independent of ticket 029 (delay system redesign) — supervision is taken from `_triangulation_result_gt` which uses GT target position as the FIM eval point, not the delay pipeline.
- Adjacent to the planned 4D actor tri tail in [observation_redesign_spec.md §4.1](../../../observation_redesign_spec.md#L153). Orthogonal: the head and the input observation are independent variables. Either can be ablated against the other; do not couple them in this ticket.
- **Forward-compatible with `task_reward_level >= 2`**: supervision uses only `_triangulation_result_gt`, which is unconditionally computed. Bumping the task level later does not require touching this head.
- **Forward-compatible with non-RNN MAPPO**: if the trainer base ever switches back to plain `MAPPO`, `MAPPOWithAux` can be ported by swapping its parent class and adjusting the encoder (drop GRU). Tests are largely shape-agnostic.

### Affected files

**New, under `scripts/reinforcement_learning/skrl/`** (alongside `mappo_rnn.py`):
- NEW: [scripts/reinforcement_learning/skrl/mappo_with_aux.py](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_with_aux.py) — `MAPPOWithAuxPolicy`, `MAPPOWithAuxValue`, `MAPPOWithAux`, `MAPPO_WITH_AUX_DEFAULT_CONFIG`.
- NEW: [scripts/reinforcement_learning/skrl/train_mappo_with_aux_hydra.py](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_with_aux_hydra.py) — training entry point, mirroring `train_mappo_rnn_hydra.py`.

**New, pure-PyTorch tests (no AppLauncher) — under `scripts/reinforcement_learning/skrl/tests/policy_tri_head/`**:
- NEW: `scripts/reinforcement_learning/skrl/tests/__init__.py` (if not present)
- NEW: `scripts/reinforcement_learning/skrl/tests/policy_tri_head/__init__.py`
- NEW: `scripts/reinforcement_learning/skrl/tests/policy_tri_head/run_tests.py` (no AppLauncher)
- NEW: `.../test_actor_with_tri_head_shape.py`
- NEW: `.../test_no_skip_connection.py`
- NEW: `.../test_input_parity.py`
- NEW: `.../test_aux_loss_masking.py`
- NEW: `.../test_invalid_triangulation_no_grad.py`
- NEW: `.../test_memory_tensor_registration.py`
- NEW: `.../test_loss_scale_zero_regression.py`
- NEW: `.../README.md`

**New, env-integration tests (AppLauncher) — under `iris_ma6/tests/policy_tri_head/`**:
- NEW: `iris_ma6/tests/policy_tri_head/__init__.py`
- NEW: `iris_ma6/tests/policy_tri_head/run_integration_tests.py` (AppLauncher-prefixed per [CLAUDE.md](../../../../../../../../../../CLAUDE.md))
- NEW: `iris_ma6/tests/policy_tri_head/test_world_frame_supervision.py`
- NEW: `iris_ma6/tests/policy_tri_head/test_preflight_smoke.py` (≥128-step rollout + 1 update)
- NEW: `iris_ma6/tests/policy_tri_head/README.md`

**Edits**:
- NEW: [agents/skrl_mappo_rnn_aux_cfg.yaml](../../../../agents/skrl_mappo_rnn_aux_cfg.yaml) — based on `skrl_mappo_rnn_cfg.yaml` with `tri_head` block under `models.policy` and `aux_loss_scale` / `aux_nll_clamp` under `agent`. Header comment dates the basis. Original `skrl_mappo_rnn_cfg.yaml` left untouched.
- EDIT: [iris_ma6/__init__.py](../../../../__init__.py) — register `skrl_mappo_rnn_aux_cfg_entry_point` for `Isaac-Iris-MA6-Direct-Test-v0` (and optionally for MVT-v0 / Test-v1 for symmetry; engineer choice).
- EDIT: [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) — add public method `get_aux_supervision() -> dict[str, torch.Tensor]` that reads `_triangulation_result_gt` and the existing per-agent `bbox_empty` (line 1777, line 1862) and returns `{"tri_target_position_w": [N,A,3], "tri_target_valid": [N,A]}`. Pure READ; no new compute, no new lifecycle hook.
- EDIT: [doc/policy_triangulation_head_spec.md](../../../policy_triangulation_head_spec.md) — reflect per-agent + episode-start masking, SKRL integration pattern, GRU statefulness in deployment contract.

### References

- [scripts/reinforcement_learning/skrl/mappo_rnn.py](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py) — base class for `MAPPOWithAux`; existing custom MAPPO subclass.
- [scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py) — template for `train_mappo_with_aux_hydra.py`.
- [triangulation/CONTEXT.md](../../../../triangulation/CONTEXT.md) — triangulation module contract.
- [iris_ma_env6_test.py:1448](../../../../iris_ma_env6_test.py#L1448) — `_triangulation_result_gt` write site (source of GT supervision).
- [iris_ma_env6_test.py:1777](../../../../iris_ma_env6_test.py#L1777) — per-agent `bbox_empty` computation (source of per-agent half of validity mask).
- [iris_ma_env6_test_cfg.py:496-499](../../../../iris_ma_env6_test_cfg.py#L496-L499) — `task_reward_level = 1` (operating regime).
- [visualization/covariance_ellipsoid.py:137-152](../../../../visualization/covariance_ellipsoid.py#L137-L152) — diag-only covariance consumption (motivates diagonal head).
- skrl Model.compute() contract: `~/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/skrl/models/torch/base.py:313-331`.
- skrl Memory.create_tensor() contract: `~/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/skrl/memories/torch/base.py:131-185`.

**Flow**: Medium. Three load-bearing pieces (custom MAPPO subclass, calibration-under-delay, memory tensor registration). All others are mechanical. Estimated 2-3 commits, with vertical slices: (1) model class + shape/no-skip/input-parity/memory-registration tests; (2) supervision plumbing + masking/world-frame/no-grad tests; (3) trainer override + bit-exact regression test; full training run as a separate validation pass.
