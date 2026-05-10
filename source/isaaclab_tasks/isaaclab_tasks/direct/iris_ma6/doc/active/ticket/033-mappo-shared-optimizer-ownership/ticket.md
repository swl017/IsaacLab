## Ticket 033 — MAPPO-RNN optimizer ownership for shared and heterogeneous agents

**Status**: Implemented (pending end-to-end validation in a real iris_ma6 training run)
**Created**: 2026-05-09
**Implemented**: 2026-05-09
**Target trainer**: MAPPO-RNN training path for iris_ma6:
[mappo_rnn.py](../../../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py) and
[train_mappo_rnn_hydra.py](../../../../../../../../../scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py).
**Observed in run**:
`/home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma6/2026-05-08_01-25-46_mappo_rnn_torch_2f8907ebed_siyi_a8_mini_siyi_zoom_continuous_critic_on`

**What**: Fix MAPPO-RNN optimizer/scheduler ownership so shared model parameters are stepped exactly once per update, while preserving support for future heterogeneous agents with separate physics, observations, actions, and tasks.

**Why**: The current trainer creates one shared actor and one shared centralized critic, then assigns the same Python model objects to every agent:

- `shared_policy = MAPPORNNPolicy(...)`
- `shared_value = MAPPORNNValue(...)`
- `models[agent_id] = {"policy": shared_policy, "value": shared_value}`

But `MAPPO_RNN.__init__` creates one optimizer and one KL scheduler per `agent_id`, each wrapping that agent's `policy` and `value` parameters. When policy/value objects are shared, this means multiple independent Adam optimizers and KL schedulers own the same parameter tensors. The `_update()` loop then steps the same shared parameters once for `drone_0` and again for `drone_1` inside the same MAPPO update.

This violates the optimizer invariant:

> A parameter tensor must be owned by exactly one optimizer step per update.

Consequences seen in the 2026-05-08 run:

- `drone_0` and `drone_1` log different learning-rate curves even though they update the same shared policy/value weights.
- Adam moments and step counters diverge per agent for the same parameter tensors.
- KLAdaptiveLR decisions are made independently per agent, then both schedulers mutate shared parameters.
- Policy standard deviation shows regular peaks and late upward drift, especially after curriculum hardening phases.
- Policy/value losses become harder to interpret because each per-agent metric is both a local loss and a partial update to a shared model.

This ticket should not hardcode "one global optimizer for all agents." The future direction includes heterogeneous agents, so the correct rule is:

> Build one optimizer/scheduler per unique parameter ownership group, not per agent id.

### Applicability — when this bug actually triggers

This pathology is **unique to the homogeneous-shared-model case**, where multiple
`uid` entries point to the same Python model object (the current iris_ma6 setup,
where `models["drone_0"]["policy"] is models["drone_1"]["policy"]`).

It does **not** apply to:

- Single-agent training (only one `uid`, only one optimizer).
- Heterogeneous multi-agent training where each `uid` owns its own policy/value
  objects with disjoint parameter tensors. In that case the stock per-`uid`
  optimizer pattern is already correct: each Adam optimizer owns a unique
  parameter set, each KLAdaptiveLR controls only its own LR, and per-agent
  metrics are unambiguously interpretable.

Implications for the fix:

- The fix is not "always aggregate across all agents." Forcing aggregation in a
  heterogeneous setup would either be a no-op (different parameter sets just
  get summed into a meaningless joint loss) or actively wrong (e.g. mixing
  losses with different scales/objectives across agents with different tasks).
- The correct trigger is **parameter-tensor identity overlap across `uid`s**.
  When two `uid`s share at least one trainable parameter tensor, their
  contributions must be combined into a single optimizer step on that shared
  group. When parameter sets are disjoint, the per-`uid` path is preserved
  unchanged.
- For the current iris_ma6 homogeneous setup, this collapses to "one optimizer
  for both drones." For a future heterogeneous setup it falls through to the
  existing per-`uid` behavior with no behavioral change. For the hybrid case,
  parameter-id overlap detection is what makes the grouping decision automatic
  rather than configuration-driven.

### Current behavior to preserve

- Homogeneous shared policy/value remains supported.
- Existing per-agent memories remain supported.
- Existing centralized critic path remains supported, including `shared_observation_spaces` and `env.state()`.
- Existing recurrent sequence training, `episode_start_mask_steps`, value preprocessing, advantage normalization, KL early stopping, and checkpointing semantics remain intact.

### Desired ownership semantics

#### Case 1 — Homogeneous fully shared models

Example:

```python
models["drone_0"]["policy"] is models["drone_1"]["policy"]
models["drone_0"]["value"]  is models["drone_1"]["value"]
```

Expected:

- One policy/value parameter group.
- One optimizer.
- One KL scheduler.
- Losses from all contributing agents are aggregated.
- One backward/grad-clip/optimizer-step per MAPPO update for that shared group.
- One scheduler step using an aggregate KL over contributing agents.

#### Case 2 — Heterogeneous fully separate models

Example:

```python
models["camera_drone"]["policy"] is not models["interceptor_drone"]["policy"]
```

Expected:

- Separate parameter ownership groups.
- Separate optimizers and schedulers are valid.
- Each group steps only the parameters it uniquely owns.
- Per-group KL and loss logging remains meaningful.

#### Case 3 — Hybrid partially shared models

Example:

- Shared perception encoder.
- Separate actor heads.
- Shared or separate critics.

Expected:

- No parameter appears in more than one optimizer.
- If any parameters are shared across agents or heads, either:
  - use a single optimizer for the coupled group and aggregate all relevant losses, or
  - split only truly disjoint parameter sets into separate optimizers.
- Avoid overlapping optimizers such as "one optimizer for shared encoder + head A" and "another optimizer for shared encoder + head B"; that still double-steps the encoder.

### Implementation direction

1. **Detect unique parameter groups by parameter identity.**
   - Gather trainable parameters for each agent's policy/value pair.
   - Use `id(parameter)` or object identity to detect overlap.
   - Build ownership groups such that no parameter id appears in more than one group.

2. **Map agents to ownership groups.**
   - Each agent contributes rollout data, losses, and KL samples to one or more model groups.
   - Fully shared current iris_ma6 should produce one group containing both `drone_0` and `drone_1`.
   - Fully heterogeneous agents should produce one group per agent.

3. **Aggregate losses before stepping a shared group.**
   - Compute each agent's PPO policy loss, value loss, entropy loss, and KL using that agent's memory.
   - For each ownership group, sum or mean the contributing losses.
   - Backpropagate once for the group.
   - Clip gradients once over that group's unique parameters.
   - Step the group's optimizer once.

4. **Aggregate scheduler signal consistently.**
   - For KLAdaptiveLR, use the mean KL over the group contributors and minibatches.
   - Log both:
     - per-agent KL/loss diagnostics, and
     - per-optimizer-group LR/KL diagnostics.

5. **Checkpoint by ownership without breaking existing loading.**
   - Avoid saving multiple optimizer states for the same shared parameters.
   - Preserve model state dict compatibility with current checkpoints where feasible.
   - If checkpoint format changes, provide a migration/read path for existing `agent_*.pt` checkpoints.

6. **Keep heterogeneous-agent path explicit.**
   - Do not force all agents into one optimizer.
   - Do not assume same observation/action shapes.
   - Do not assume same reward/task objective.
   - The only invariant is no overlapping optimizer ownership of the same parameter tensor.

### Tests

Add targeted tests around the optimizer construction and update path. These can be pure-PyTorch tests using tiny mock models and fake memories where possible; AppLauncher is not required for the core invariant.

1. **Shared model ownership test**
   - Create two agents that reference the same policy and same value model.
   - Initialize MAPPO-RNN.
   - Assert exactly one optimizer owns each shared parameter id.
   - Assert only one scheduler exists for the shared ownership group.

2. **Separate model ownership test**
   - Create two agents with separate policy/value models.
   - Assert two disjoint optimizers are created.
   - Assert no parameter id appears in more than one optimizer.

3. **Hybrid shared-encoder test**
   - Create two agent models with a shared encoder and separate heads.
   - Assert the shared encoder parameters are not present in two optimizers.
   - Assert the grouping either couples the whole overlapping set or otherwise rejects ambiguous overlap with a clear error.

4. **Single-step update test**
   - For shared models, run one synthetic update.
   - Assert Adam `state[p]["step"]` increments once per optimizer update, not once per agent id.
   - Assert model parameters change after the aggregate loss step.

5. **Scheduler consistency test**
   - Feed controlled KL values for two shared agents.
   - Assert the scheduler uses the aggregate KL once.
   - Assert only one LR value is logged for the shared optimizer group, plus optional per-agent diagnostic KL.

6. **Regression test for heterogeneous agents**
   - Create separate models for two agents.
   - Run one update.
   - Assert each optimizer's step counter increments independently and no parameter state is shared.

### Acceptance criteria

- No trainable parameter id appears in more than one optimizer's parameter groups.
- Current iris_ma6 homogeneous shared-policy setup creates one optimizer/scheduler group for the shared actor/critic parameters.
- Future heterogeneous setup with disjoint policies creates disjoint optimizers/schedulers.
- Shared-policy update aggregates both agents' losses before one optimizer step.
- KLAdaptiveLR is stepped once per optimizer group using aggregate KL from contributing agents.
- TensorBoard or SKRL logs make ownership clear:
  - per-agent losses remain available for diagnosis,
  - per-optimizer-group LR/std/KL are logged for optimizer behavior.
- Existing checkpoints can still be loaded for evaluation, and either:
  - old training checkpoints can resume, or
  - resume incompatibility is documented explicitly with a migration path.

### Scope boundary

- DO: fix optimizer/scheduler ownership and update semantics.
- DO: support homogeneous shared models, heterogeneous separate models, and hybrid partial-sharing without overlapping optimizer ownership.
- DO: add explicit diagnostics so future metric interpretation is not ambiguous.
- DO: keep the existing MAPPO-RNN algorithmic behavior unchanged except for removing double-optimizer stepping of shared parameters.
- DO NOT: tune rewards, curriculum, entropy scale, learning-rate thresholds, or model sizes in this ticket.
- DO NOT: change iris_ma6 environment reward semantics.
- DO NOT: switch to a new RL library or rewrite the trainer wholesale.
- DO NOT: assume all future agents are homogeneous.

### Risk

Medium-high. The bug fix touches the core update path, and the recurrent MAPPO implementation already carries several shape-sensitive contracts:

- sequence reshaping and RNN hidden-state reset,
- episode-start loss masking,
- value preprocessing,
- time-limit bootstrapping,
- KL early stopping,
- shared centralized critic state.

The safest implementation is to first isolate optimizer ownership/grouping behind small helper functions with pure-PyTorch tests, then change the `_update()` stepping order while preserving the existing per-agent loss computation as much as possible.

### Notes from run analysis

The 2026-05-08 run's best checkpoint is `agent_40000.pt` / `best_agent.pt`, while final reward at 400k is lower. Later policy standard deviation increases mostly on lateral velocity, yaw, gimbal, and zoom dimensions. Curriculum hardening explains part of the late behavior, but the shared-parameter double-optimizer issue makes LR/std/loss curves materially less trustworthy. This ticket addresses that trainer-level confound before further MAPPO-RNN hyperparameter interpretation.

### Stock skrl MAPPO behavior

Checked on 2026-05-09 in this IsaacLab environment:

`/home/usrg/IsaacPX4/IsaacLab/_isaac_sim/kit/python/lib/python3.10/site-packages/skrl/multi_agents/torch/mappo/mappo.py`

Stock `skrl.multi_agents.torch.mappo.MAPPO` also builds optimizers and schedulers per `agent_id`, not as a single global optimizer:

```python
self.optimizers = {}
self.schedulers = {}

for uid in self.possible_agents:
    policy = self.policies[uid]
    value = self.values[uid]
    if policy is value:
        optimizer = torch.optim.Adam(policy.parameters(), lr=self._learning_rate[uid])
    else:
        optimizer = torch.optim.Adam(
            itertools.chain(policy.parameters(), value.parameters()),
            lr=self._learning_rate[uid],
        )
    self.optimizers[uid] = optimizer
```

Its `_update()` loop then iterates over `self.possible_agents` and calls `zero_grad()`, `backward()`, `step()`, and scheduler update through `self.optimizers[uid]` / `self.schedulers[uid]`.

Implication:

- Stock skrl MAPPO is fine for the common heterogeneous/separate-model case where each `uid` owns disjoint model parameters.
- Stock skrl MAPPO does not appear to deduplicate optimizer ownership when multiple agents point to the same Python model object.
- Passing the same shared model object under multiple `uid` entries can therefore create the same overlapping-optimizer problem.
- The desired iris_ma6 fix should be a deliberate extension over stock skrl behavior: optimizer ownership should follow unique parameter sets, not `agent_id`.

### skrl v2.0.0 status (checked 2026-05-09)

skrl v2.0.0 was released on 2026-04-08. The release notes highlight a NVIDIA Warp
backend, MuJoCo Playground / ManiSkill support, and a new
observation-vs-privileged-state distinction — none of which touch the multi-agent
optimizer construction path.

The v2 source on the `develop` branch was inspected to confirm whether the
shared-parameter-double-optimizer issue had been addressed upstream. It has not:

- `skrl/multi_agents/torch/mappo/mappo.py` (v2) still iterates
  `for uid in self.possible_agents` and creates one `torch.optim.Adam` per `uid`.
- The only existing deduplication is the within-uid `if policy is value` shortcut,
  which collapses the policy/value pair for a single agent. It does not detect or
  collapse sharing *across* uids.
- There is no `id(parameter)`-based grouping, no set-membership "already-seen-this-model"
  guard, no cross-uid loss aggregation prior to `backward()`/`step()`, and no comment
  or runtime warning about the same Python model object being registered under
  multiple `uid` entries.
- IPPO (`skrl/multi_agents/torch/ippo/ippo.py`) follows the same per-uid optimizer
  pattern in v2; the same caveat applies.

Implication for this ticket:

- We cannot defer this fix to an upstream skrl release. v2.0.0 leaves the behavior
  unchanged from v1.4.x for shared-model setups.
- The fix described in this ticket should be implemented in our local
  `mappo_rnn.py`. If iris_ma6 ever switches to stock `skrl.MAPPO` (MLP path), the
  same patch logic — ownership keyed by unique parameter set rather than `uid` —
  will need to be applied or wrapped there as well.

### Implementation summary (2026-05-09)

The fix lives entirely in our local `mappo_rnn.py` and a new sibling helper
module `mappo_rnn_groups.py`. The public surface (`MAPPO_RNN`,
`MAPPO_RNN_DEFAULT_CONFIG`, `MAPPORNNPolicy`, `MAPPORNNValue`) is unchanged,
so no caller — including `train_mappo_rnn_hydra.py`, `train_mappo_rnn.py`,
`play_iris_mappo_rnn.py`, and `mappo_with_aux.py` — needs to be touched.

Files changed / added:

- `scripts/reinforcement_learning/skrl/mappo_rnn_groups.py` (new) —
  parameter-ownership grouping helper. Pure-PyTorch, dependency-free apart
  from torch, so it is unit-testable without AppLauncher / Isaac Sim.
- `scripts/reinforcement_learning/skrl/mappo_rnn.py` —
  * `__init__` now builds ownership groups, then constructs **one**
    `torch.optim.Adam` and **one** `KLAdaptiveLR` per group. Per-uid
    `self.optimizers[uid]` / `self.schedulers[uid]` aliases are kept for
    back-compat with skrl's `checkpoint_modules` plumbing.
  * `_update` is restructured into three explicit phases:
    Phase A keeps the existing per-uid GAE + sequence batching unchanged
    and stashes the prepared state in `per_uid_state[uid]`.
    Phase B iterates per ownership group: for each (epoch, minibatch) it
    sums per-uid losses on the un-stepped weights, runs one backward, one
    grad-clip over the group's unique parameters, one `optimizer.step()`,
    and one `scheduler.step()` per epoch using aggregate KL.
    Phase C preserves per-uid loss / std diagnostics so per-agent curves
    remain interpretable.
  * Per-group LR and KL are tracked separately under
    `Learning / Learning rate (group: ...)` and `Learning / KL (group: ...)`.
- `scripts/reinforcement_learning/skrl/tests/test_mappo_rnn_groups.py` (new)
  — 13 pure-PyTorch unit tests for the grouping helper, including a
  reproduction of the Adam-state divergence bug in isolation.
- `scripts/reinforcement_learning/skrl/tests/test_mappo_rnn_init.py` (new)
  — 4 integration tests that build `MAPPO_RNN` against tiny torch models
  and verify optimizer/scheduler aliasing and Adam-state cardinality.
- `scripts/reinforcement_learning/skrl/tests/test_mappo_rnn_update.py`
  (new) — 3 end-to-end tests that drive `MAPPO_RNN` through one rollout of
  synthetic transitions and trigger `_update`. The headline test checks
  that the shared optimizer steps exactly `learning_epochs * mini_batches`
  times after one MAPPO update, **not double**, proving the bug is gone in
  the full update path. The heterogeneous variant confirms each per-uid
  optimizer steps independently when parameter sets are disjoint.
- `scripts/reinforcement_learning/skrl/tests/README.md` (new) — overview
  of the test suite layout, run commands, and links back to this ticket.

Test results on 2026-05-09 (CUDA, torch 2.5.1+cu118):
`groups 13/13`, `init 4/4`, `update 3/3` — all green.

Acceptance criteria checked:

- ✅ No trainable parameter id appears in more than one optimizer's
  parameter groups (asserted via `assert_groups_partition_params` at
  `__init__` and verified by `test_mappo_rnn_groups.py`).
- ✅ Current iris_ma6 homogeneous shared-policy setup creates one
  optimizer/scheduler for the shared actor/critic parameters
  (`test_mappo_rnn_init.py::test_homogeneous_shared_collapses_to_one_optimizer`).
- ✅ Future heterogeneous setup with disjoint policies creates disjoint
  optimizers/schedulers
  (`test_mappo_rnn_init.py::test_heterogeneous_disjoint_keeps_per_uid_optimizers`,
  `test_mappo_rnn_update.py::test_heterogeneous_each_optimizer_steps_independently`).
- ✅ Shared-policy update aggregates both agents' losses before one
  optimizer step
  (`test_mappo_rnn_update.py::test_shared_optimizer_step_counter_after_one_update`).
- ✅ KLAdaptiveLR is stepped once per optimizer group using aggregate KL
  from contributing agents (Phase B scheduler block in `_update`).
- ✅ Logs make ownership clear: per-agent losses preserved
  (`Loss / Policy loss (drone_0)` etc.), plus per-group LR/KL
  (`Learning / Learning rate (group: drone_0+drone_1)`,
  `Learning / KL (group: drone_0+drone_1)`).
- ⚠️ Checkpoint compatibility: `checkpoint_modules[uid]["optimizer"]`
  still points at the (now shared) optimizer object. For the homogeneous
  shared case skrl will save the same optimizer state under each uid's
  key (redundant but harmless); loading old per-uid checkpoints into a
  shared group will load each uid's optimizer state into the same shared
  optimizer in turn, with the last write winning. For heterogeneous
  disjoint groups the layout is unchanged. Treat resume from old shared
  checkpoints as approximate, not bit-identical — the prior optimizer
  state was itself wrong about the parameter's update history.

Pending validation:

- Run the next iris_ma6 MAPPO-RNN training and confirm that:
  * `Learning / Learning rate (group: drone_0+drone_1)` is a single curve
    (no longer two divergent per-uid LR curves).
  * `Policy / Standard deviation (drone_0)` and `(drone_1)` are now
    identical (same shared policy → same stddev), where previously they
    diverged.
  * Late-training std drift on lateral velocity / yaw / gimbal / zoom
    behaves more cleanly than the 2026-05-08 run — direction expected,
    magnitude TBD pending the run.

### Post-mortem of 2026-05-09 first-validation run + SUM→MEAN follow-up

The first validation run with the initial fix
(`logs/skrl/iris_ma6/2026-05-09_22-12-18_mappo_rnn_torch_1abe3cf54b_mappo_rnn_shared_model_fix`)
confirmed the structural fix worked — `drone_0` and `drone_1` policy std
became bit-identical, and only one LR/KL curve was logged per group — but
training collapsed:

- Reward peaked at 4362 at 24k (better bootstrap than the 2026-05-08 buggy
  run's 3957), then degraded: 3288 at 44k → 1997 at 64k → 1001 at 104k →
  28 at 124k → -200 plateau through 264k.
- Episode length followed: 498 (24k) → 251 (124k) → 141 (264k).
- `Termination/tracking_lost_fraction` rose from 0.005 (24k) to 0.78 (124k)
  to 0.98 (264k); `Detection/pair_valid_rate` collapsed 0.91 → 0.13 → 0.07.
- KLAdaptive pinned LR at `min_lr=1e-4` from 124k onward; aggregate KL
  (group) hovered at 0.025–0.04 against `kl_threshold=0.02`.
- Per-uid logged losses were 5–25× larger than the pre-fix run's at
  comparable steps (mostly during the curriculum-window failure).

Root cause: the initial fix aggregated per-uid losses by **SUM** in Phase B
of `_update`. For a homogeneous shared group with N=2 uids this silently
multiplied the optimizer-visible scaling of `value_loss_scale` and
`entropy_loss_scale` by N, and shifted the per-MB KL that KLAdaptiveLR
consumes into a regime the threshold was not tuned for. Concretely:

```
SUM:  total_loss = Σ_uid (policy_loss_uid + value_loss_scale * value_loss_uid
                         + entropy_loss_scale * entropy_loss_uid)
                 = Σ policy_loss + (N * value_loss_scale) * mean_value_loss
                                 + (N * entropy_loss_scale) * mean_entropy_loss
```

So with `value_loss_scale=1.0`, `entropy_loss_scale=0.01`, and N=2, the
optimizer was effectively running with 2.0 / 0.02 — the value head over-
weighted the policy head's gradient share, and the entropy bonus was
doubled (consistent with the observed late-training std drift back up to
0.297 once policy gradients went to zero).

Fix follow-up (2026-05-09, same session):

- Phase B in `mappo_rnn.py::_update` now aggregates per-uid losses by
  **MEAN** instead of SUM:

  ```python
  total_loss = sum(
      c["policy_loss"] + c["value_loss"] + c["entropy_loss"]
      for c in components_per_uid.values()
  ) / len(group.uids)
  ```

- For singleton (heterogeneous) groups, `len(group.uids) == 1` and MEAN
  reduces to identity — zero behavior change for disjoint agents. For
  shared groups (N ≥ 2), every loss-scaled hyperparameter recovers its
  configured magnitude, KL-per-MB recovers the pre-fix per-uid magnitude,
  and KLAdaptiveLR sees the signal it was tuned against.
- This matches the canonical MAPPO formulation
  L_MAPPO = (1/N) Σ_i L_PPO_i.
- All 20 tests still green after the change: groups 13/13, init 4/4,
  update 3/3.

Why this is the principled choice over the alternatives:

| Option | Behavior | Verdict |
|--------|----------|---------|
| Retune LR / scales / threshold | per-N maintenance debt; LR knob means different things in shared vs hetero setups | rejected |
| Keep SUM, halve scales for shared groups | same maintenance debt, surprising semantics | rejected |
| Disable KL early-stop for groups | symptom-suppression, not root-cause | rejected |
| Revert to per-uid optimizers | that is the bug | rejected |
| **SUM → MEAN** | hyperparameter-invariant under N, matches MAPPO literature, single-line change | **accepted** |

Pending second-validation run: same agent.yaml, same seed, with the MEAN
fix in place. Expected:

- Per-uid `Loss / Policy loss`, `Loss / Value loss`, `Loss / Entropy loss`
  drop back to the pre-fix per-uid magnitudes (5–25× smaller than the
  first-validation run).
- KL-per-MB drops back to the pre-fix per-uid range (~0.005–0.015 instead
  of 0.025–0.040 aggregate); KLAdaptive raises rather than pins LR.
- Bootstrap stays as good as the SUM run (cleaner gradients are still
  cleaner — that benefit comes from the optimizer/scheduler unification,
  not from sum-vs-mean).
- Curriculum window 40k–120k: policy adapts at roughly the pre-fix rate;
  `Termination/tracking_lost_fraction` does not exceed 0.1, episode
  length stays >450, reward stays >2000 through 200k.

Open thought (separate from this ticket): bootstrap-phase std collapses
from 0.348 → 0.169 by 24k in both pre-fix and post-fix runs; this rapid
commitment is a generic feature of the current `entropy_loss_scale=0.01`
+ `min_log_std=-5.0` (σ_floor ≈ 0.007) combination, not caused by the
fix. If late-curriculum re-exploration ever proves brittle, raising
`min_log_std` to about -2.0 (σ_floor ≈ 0.135) would prevent over-
commitment without distorting the policy mean. Tracked separately, not
in scope for this ticket.

### skrl v2 RNN support for multi-agent algorithms (checked 2026-05-09)

skrl v2 still does **not** ship a recurrent variant of MAPPO (or IPPO). The
`skrl/multi_agents/torch/mappo/` directory on the v2 `develop` branch contains
only `__init__.py`, `mappo.py`, and `mappo_cfg.py`; the analogous `ippo/`
directory contains only `__init__.py`, `ippo.py`, and `ippo_cfg.py`. The
single-agent PPO ships an explicit `ppo_rnn.py` sibling, but that pattern has
not been extended to the multi-agent algorithms in v2.0.0.

Implication: the original motivation for our local `mappo_rnn.py` (RNN support
for MAPPO) remains. The optimizer-ownership fix in this ticket therefore
continues to live in our custom trainer and is not made obsolete by upgrading
the skrl pin.
