# {{ModuleName}} Module

## Purpose
{{MODULE_DESCRIPTION}}

## Inputs
- Input description: `(shape)` — where it comes from

## Outputs
- Output description: `(shape)` — who consumes it

## Dependencies
None (standalone module).

## Key Files
- `{{MODULE_NAME}}.py` — Core implementation
- `{{MODULE_NAME}}_cfg.py` — Configuration dataclass

## Calling Contract
- `update()`: **WRITE**. Call once per step from `_post_physics_step()`.
- `compute()`: **READ**. Safe to call multiple times per step.
- `reset(env_ids)`: Call from `_reset_idx()`.

## Spec
`doc/{{MODULE_NAME}}_spec.md`
