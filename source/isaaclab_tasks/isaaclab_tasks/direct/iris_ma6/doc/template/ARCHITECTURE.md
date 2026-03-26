# {{PROJECT_NAME}} Architecture

{{PROJECT_DESCRIPTION}}

## Module Dependency Graph

```
┌──────────────────────────────────────────────────────────┐
│            {{ENV_FILE}} (Main Environment)                │
│            Orchestrates all modules below                │
└──────────────────┬───────────────────────────────────────┘
                   │ imports & calls
    ┌──────────────┼──────────────────────────────────┐
    │              │              │                    │
    ▼              ▼              ▼                    ▼
module_a       module_b       module_c            module_d
```

<!-- Update this diagram as modules are added or dependencies change. -->

## Directed Dependencies

List cross-module dependencies (outside the main environment):

```
module_x ──→ module_y    (reason for dependency)
```

All other modules should be standalone. The environment file is the sole integration point.

## Data Flow Per Step

```
Policy Output (action_dim per agent)
       │
       ▼
_pre_physics_step()
  ├─ [action processing, safety filters]
       │
       ▼
_physics_step()
  ├─ [force application, substeps]
       │
       ▼
_post_physics_step()
  ├─ [state updates, sensor processing]
       │
       ▼
_get_rewards()                         _get_observations()
  ├─ [reward terms]                      ├─ [observation assembly]
       │
       ▼
_get_dones()
  ├─ [termination conditions]

_reset_idx(env_ids)
  ├─ [state resets, randomization]
```

## Key Data Containers

| Container | Module | Shape | Description |
|-----------|--------|-------|-------------|
| <!-- Add rows as modules are implemented --> | | | |

## Module Isolation

**Standalone** (no internal dependencies):
<!-- List modules here -->

**Has dependencies**:
<!-- List modules with their dependencies here -->

## Stateful Mutation Rule

Methods that advance internal state (buffer appends, counter increments, RNG samples) must only be called from **ONE lifecycle hook per step**. Read-only accessors are safe to call from multiple hooks. All stateful methods must be idempotent within a single sim step (guarded by `_last_update_time` or equivalent).

See individual module `CONTEXT.md` files for per-method WRITE/READ/CONFIG annotations.

## File Conventions

- `*_cfg.py` — Configuration dataclasses
- `CONTEXT.md` — Module routing contract (inputs, outputs, dependencies)
- `tests/` — Per-module test suites
- `doc/*_spec.md` — Authoritative specifications
