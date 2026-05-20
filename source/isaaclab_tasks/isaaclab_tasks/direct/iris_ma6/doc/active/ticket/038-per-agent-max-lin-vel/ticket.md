## Ticket 038 — Extend `_max_lin_vel` to per-(env, agent)

**Status**: Open
**Created**: 2026-05-20
**Discovered in**: Ticket 037 Slice 1 inventory audit (critic-only privileged obs).
**Affected modules**: `iris_ma_env6_test.py` (`_max_lin_vel` storage), `_pre_physics_step` (action scaling).
**Blocked on**: nothing — independent of ticket 037 but motivated by it.

---

### What

`self._max_lin_vel` is currently shape `(N,)` — one value per env, applied uniformly to all agents within an env. At reset, the per-agent draws from `_eff_progress_agent_velocity` (shape `(N, A)`) are collapsed via `.mean(dim=-1)` ([iris_ma_env6_test.py:2632](../../../../iris_ma_env6_test.py#L2632)) before being written into `_max_lin_vel`.

Extend the storage to `(N, A)` so each agent has its own `max_lin_vel` derived from its own `_eff_progress_agent_velocity[env, agent]` draw, and update consumers (action scaling at [iris_ma_env6_test.py:740](../../../../iris_ma_env6_test.py#L740)) to use the per-agent value.

### Why

1. **Sim-to-real fidelity**: real drones in a multi-agent setup have *independent* max-velocity envelopes (mechanical, motor, payload-dependent per vehicle). The per-env collapse is a sim-only artifact, not a real-world correlation.
2. **Per-(env, agent) curriculum jitter (ticket 034)**: the curriculum samples `_eff_progress_agent_velocity` per-(env, agent). Collapsing to per-env discards information the curriculum has already paid to sample. The bug is silent — all agents in an env see the same max_lin_vel even though their progress draws differ.
3. **Critic privileged obs (ticket 037 §3.4 field #7)**: with `_max_lin_vel` per-env, the critic's `max_lin_vel_per_env` field is identical for all agents in an env. Per-agent storage gives the critic genuine per-(env, agent) signal — likely small in magnitude but conceptually correct.

### Scope

- **Storage**: `self._max_lin_vel` from `(N,)` → `(N, A)`.
- **Reset path** ([iris_ma_env6_test.py:2627–2658](../../../../iris_ma_env6_test.py#L2627-L2658)):
  - Remove the `.mean(dim=-1)` collapse at L2632.
  - Compute `_max_lin_vel[env_ids]` per-(env, agent) using `_eff_progress_agent_velocity[env_ids]` shape `(M, A)` directly.
  - The `randomize_max_lin_vel` block at L2653-2658 already draws `scale` per env; extend to per-(env, agent) `(M, A)` and apply to `_max_lin_vel[env_ids]`.
- **Action scaling** ([iris_ma_env6_test.py:740](../../../../iris_ma_env6_test.py#L740)):
  - Current: `self.cmd_vel[:, idx, 0:3] = action[:, 0:3] * self._max_lin_vel.unsqueeze(-1)`
  - New: `self.cmd_vel[:, idx, 0:3] = action[:, 0:3] * self._max_lin_vel[:, idx].unsqueeze(-1)`
- **Initial-states module**: check whether `initial_states_generator.py` reads `_max_lin_vel` to scale velocity initial conditions — if yes, update the read path too.
- **Cfg flag**: `enable_per_agent_max_lin_vel: bool = False` for bit-exact t034 reproduction. Default False; new runs opt in.

### Acceptance criteria

- With `enable_per_agent_max_lin_vel = False`, `_max_lin_vel` reproduces the current per-env-collapsed value bit-exactly. Existing checkpoints continue to evaluate without regression.
- With `enable_per_agent_max_lin_vel = True`, each `(env, agent)` has its own `_max_lin_vel` value derived from its own `_eff_progress_agent_velocity` draw + independent randomization.
- Unit test: 1024 envs, 2 agents, set `_eff_progress_agent_velocity[0, 0] = 0.0` and `_eff_progress_agent_velocity[0, 1] = 1.0` — assert `_max_lin_vel[0, 0] != _max_lin_vel[0, 1]` after reset, and the action scaling at L740 uses the per-agent value.
- Regression: random-seed 1k-step rollout with flag=False matches t034 fixed-seed rollout in `cmd_vel[:, :, 0:3]` to float-precision.

### Scope boundary

- DO: extend `_max_lin_vel` to per-(env, agent); update all consumers.
- DO NOT: change actor obs (max_lin_vel is not in the actor obs; this is purely physics-side).
- DO NOT: change initial-state spawn ranges (they're a separate config path).

### Risk

- **Low**: the per-env collapse appears to be a vestige rather than an intentional design (no comments suggest why agents within an env should share a max_lin_vel). Extending is straightforward.
- **Watch**: any code path that broadcasts `_max_lin_vel` assuming `(N,)` shape will need an update. Slice 1 of this ticket = enumerate all read sites.

### Files affected

- EDIT: [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) — storage, reset path, action scaling
- EDIT: [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py) — add `enable_per_agent_max_lin_vel: bool` flag
- POTENTIAL EDIT: [initial_states/initial_states_generator.py](../../../../initial_states/initial_states_generator.py) (if it reads `_max_lin_vel`)

### Relation to ticket 037

Ticket 037 currently uses `_max_lin_vel` per-env (broadcast across agents in the critic obs); after this ticket lands, that broadcast becomes a genuine per-(env, agent) signal with no change to ticket 037's code — the same tensor read at [iris_ma_env6_test.py:_get_states] just returns the per-(env, agent) shape and the existing per-agent indexing works.

### Effort

~½ day to identify all `_max_lin_vel` read sites + extend + add cfg flag + regression test.
