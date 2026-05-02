# Policy Tri Head — Pure-PyTorch Test Suite

Tests for `mappo_with_aux.py` (model class + standalone loss helper) that do
**not** require Isaac Sim. Fast iteration loop (~5s) for ticket 031.

Env-integration tests live separately at
[iris_ma6/tests/policy_tri_head/](../../../../source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/tests/policy_tri_head/).

## Test files (slice 1)

- `test_actor_with_tri_head_shape.py` — `MAPPOWithAuxPolicy.compute()` returns
  the right shapes; `tri_log_var` lies inside the configured clamp.
- `test_no_skip_connection.py` — `policy_layer.in_features == gru_hidden_size`,
  not `gru_hidden_size + 6`. Regression guard against accidentally turning
  Intent 1a into Intent 1b.
- `test_input_parity.py` — model forwards a single agent's obs in isolation
  (no leakage of multi-agent shape requirements).
- `test_aux_loss_masking.py` — `compute_aux_nll_loss` cases:
  1. mixed mask matches the masked-mean NLL.
  2. all-False mask returns 0 with no NaN.
  3. episode-step mask zeros samples where `episode_step <
     episode_start_mask_steps`, even when `tri_valid=True`.
  4. NLL clamp caps absurd per-dim values.
- `test_invalid_triangulation_no_grad.py` — backward through the loss with
  an all-False mask leaves `tri_head` parameters' gradient norm at zero.

Slice 2 will add `test_memory_tensor_registration.py`. Slice 3 will add
`test_loss_scale_zero_regression.py`.

## Running

```bash
# All tests (recommended)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/tests/policy_tri_head/run_tests.py

# Verbose mode
./isaaclab.sh -p scripts/reinforcement_learning/skrl/tests/policy_tri_head/run_tests.py --test-verbose
```

The runner does not launch Isaac Sim, so plain `python` works too as long
as `torch` and `skrl` are importable in the current environment.

## Convention

- One `run_*_tests(results, device, verbose=False)` entry function per test
  file, called from `run_tests.py`.
- Pass/fail recorded via the shared `TestResults` helper in `run_tests.py`.
- No pytest. Exit code 0 ⇔ all pass.
