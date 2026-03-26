# {{ModuleName}} Module

## Purpose
{{MODULE_DESCRIPTION}}

## Inputs
- Input description: `(shape)` — where it comes from
<!-- Add all inputs with tensor shapes and value ranges -->

## Outputs
- Output description: `(shape)` — who consumes it
<!-- Add all outputs with tensor shapes and value ranges -->

## Dependencies
None (standalone module).
<!-- Or list dependencies: "controller/ (reuses DroneController)" -->

## Key Files
- `{{MODULE_NAME}}.py` — Core implementation
- `{{MODULE_NAME}}_cfg.py` — Configuration dataclass
<!-- Add other key files -->

## Calling Contract
<!-- Required for stateful modules. Delete this section for stateless modules. -->

- `method_name()`: **WRITE**. Call exactly once per step from `_lifecycle_hook()`.
  Description of what it mutates. Idempotent via `_last_update_time` guard.
- `query_method()`: **READ**. Safe to call multiple times per step.
- `set_param()`: **CONFIG**. Call from curriculum/reset for parameter changes.
- `reset(env_ids)`: Call from `_reset_idx()`.

## Spec
`doc/{{MODULE_NAME}}_spec.md`
<!-- Or "None (self-documented via docstrings)." -->
