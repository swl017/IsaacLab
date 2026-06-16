# information_reward Test Suite

Tests for `InformationReward` (ticket 050, Slice B) — the FIM-with-prior team quality + per-agent
difference (counterfactual) reward.

## Test Files
- **run_tests.py**: standalone runner (no pytest; Isaac Sim compatible).
  - Definedness at 0 / 1 / N bearings (no NaN — the key fix vs LS triangulation).
  - Two well-separated bearings collapse Σ (quality jump); two collinear stay poor (GDOP).
  - `r_diff` (marginal information) ranks orthogonal contributor above redundant one.
  - Invalid mask ignores the bearing; in_plane_only mode; batched multi-env shapes.

## Running
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/information_reward/tests/run_tests.py
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p .../run_tests.py   # CPU
```

## Expected
All pass on CPU and GPU. compute() is pure/READ-only; reward-side only (no obs/dynamics).
`test_result.txt` overwritten each run; `error_log.txt` appended.
