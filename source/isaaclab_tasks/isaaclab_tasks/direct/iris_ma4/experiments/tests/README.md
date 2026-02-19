# Experiments Sub-Module Tests

Run tests with:
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/experiments/tests/run_tests.py
```

## Test Categories

1. **Registry tests** — experiment lookup, registration, group filtering, suites
2. **Override tests** — env config dotted-path overrides, agent dict overrides, ablation flags
3. **Metrics tests** — MetricTracker accumulation, edge cases, convergence detection
4. **MLP model tests** — forward pass shapes, no RNN specification, 3D input handling
5. **AoI indices tests** — observation index computation for AoI masking
