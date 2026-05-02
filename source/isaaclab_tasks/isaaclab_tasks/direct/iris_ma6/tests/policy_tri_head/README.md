# Policy Tri Head — Env-Integration Tests

Tests for ticket 031 that exercise the **real** `IrisMA6TestEnv` and the
end-to-end training pipeline. AppLauncher-prefixed (slow) — the fast pure-
PyTorch unit tests live separately at
[scripts/reinforcement_learning/skrl/tests/policy_tri_head/](../../../../../../../../../scripts/reinforcement_learning/skrl/tests/policy_tri_head/).

## Test files (slice 2)

- `test_world_frame_supervision.py` — launches a small v0 env, reads
  `env.get_aux_supervision()`, and asserts:
  - `tri_target_position_w` is the GT target world position broadcast across
    agents.
  - `tri_target_valid` correctly composes per-agent bbox-non-empty AND scene
    `_triangulation_result_gt.is_valid`.
- `test_preflight_smoke.py` — runs ≥128 env steps with the full
  `MAPPOWithAux` + `AuxInjectingTrainer` pipeline at `aux_loss_scale=0.0` and
  asserts:
  - No exception across the run.
  - No NaN/Inf in any model parameter or gradient after the gradient update.
  - `tri_target_position_w` and `tri_target_valid` tensors in memory are
    populated with non-zero entries (catches `Memory.add_samples` silently
    dropping unregistered keys).
  - Gradient norm on `policy_layer` + `value_layer` is non-zero (the standard
    PPO loss is flowing).

Slice 3 will tighten `test_preflight_smoke.py` to also require non-zero
gradient on the `tri_head` parameters when `aux_loss_scale > 0`.

## Running

```bash
# All integration tests (recommended)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/tests/policy_tri_head/run_integration_tests.py

# Verbose mode
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/tests/policy_tri_head/run_integration_tests.py --test-verbose
```

These tests launch Isaac Sim — expect ~30s startup overhead even before any
test runs.
