# Experiments Sub-Module Documentation

## Overview

The `experiments/` sub-module provides infrastructure for running systematic ablation
experiments for the IROS 2026 paper on multi-agent active visual triangulation.

## Key Files

| File | Purpose |
|------|---------|
| `experiment_cfg.py` | `ExperimentCfg` and `ExperimentSuiteCfg` dataclasses |
| `experiment_registry.py` | Central registry of all named experiments (A1-A4, baselines, sweeps) |
| `env_overrides.py` | `apply_env_overrides()` and `apply_agent_overrides()` utilities |
| `run_experiment.py` | CLI entry point for training a single named experiment |
| `run_suite.py` | CLI entry point for batch-running a suite of experiments |
| `evaluate.py` | Post-training evaluation with 7 paper metrics |
| `metrics/metric_tracker.py` | `MetricTracker` class for all 7 metrics |
| `baselines/greedy_policy.py` | Scripted equiangular orbit policy |
| `models/mappo_mlp.py` | MLP-only policy/value models for A2 ablation |
| `models/mappo_mlp_agent.py` | `MAPPO_MLP` agent (inherits from `MAPPO_RNN`) |

## Usage

```bash
# List experiments
./isaaclab.sh -p .../experiments/run_experiment.py --list

# Run single experiment
./isaaclab.sh -p .../experiments/run_experiment.py --experiment a1_no_delay --seed 42

# Run suite (dry run)
./isaaclab.sh -p .../experiments/run_suite.py --suite iros2026_must --dry_run

# Evaluate checkpoint
./isaaclab.sh -p .../experiments/evaluate.py --experiment a1_stochastic_delay \
    --checkpoint /path/to/agent.pt --output results.json
```

## Experiment Groups

- **A1**: Delay modeling ablation (no delay, stochastic, AoI toggle, delay sweep)
- **A2**: Architecture ablation (MAPPO-RNN vs MAPPO-MLP)
- **A3**: Dual-path ablation (clean vs noisy reward pipeline)
- **A4**: Reward ablation (multi-source covariance vs angular-only vs heuristic vs image-only)
- **baseline**: Gavin2024-style (MLP, no delay, angular-noise-only covariance) and greedy orbit
- **agent_sweep**: N=1,2,3,4 agent count
- **noise_sweep**: sigma_pixel=0,1,2,4
