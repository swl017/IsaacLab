## Ticket: End-to-end policy evaluation for iris_ma6

**What**: Run the iris_ma6 `experiments/evaluate.py` on the trained `a1_with_aoi` policy checkpoint and produce timeseries + trajectory JSON data that can be fed into the existing IROS plotting scripts to generate `timeseries.pdf` and `trajectory.pdf`.

**Why**: The iris_ma6 experiments module has never been validated end-to-end. The `evaluate.py` script exists and appears structurally complete, but it has not been run against a real checkpoint. The IROS paper deadline requires these figures to be regenerated from the iris_ma6 policy (the iris_ma5 figures at `2026-IROS/figures/plots_2026-02-28-episode100sec/timeseries.pdf` and `2026-IROS/figures/trajectory_2026-03-01/.../trajectory.pdf` are the reference format).

**Scope boundary**:
- Do NOT modify the plotting scripts in `2026-IROS/figures/` (`plot_timeseries.py`, `plot_trajectory.py`). Those are the downstream consumers — we produce compatible JSON.
- Do NOT modify the training pipeline, reward function, or environment logic.
- Do NOT add MLP/frame-stacking support (iris_ma6 is MAPPO-RNN only).
- Do NOT add new experiments to the registry.

**Affected modules**:
- `iris_ma6/experiments/evaluate.py` — main evaluation entry point
- `iris_ma6/experiments/metrics/metric_tracker.py` — episode-level metrics
- `iris_ma6/experiments/metrics/timeseries_tracker.py` — per-timestep metrics
- `iris_ma6/experiments/metrics/trajectory_recorder.py` — XYZ path recording
- `iris_ma6/experiments/env_overrides.py` — experiment config application
- `iris_ma6/experiments/experiment_registry.py` — `a1_with_aoi` entry

**Policy checkpoint**: `logs/skrl/iris_ma6/2026-04-01_13-19-18_mappo_rnn_torch_928b9585f2_min_lr_tuning/`
- Contains `agent_drone_0_final.pt`, `agent_drone_1_final.pt`, `checkpoints/best_agent.pt`

**Acceptance criteria**:
1. `evaluate.py --experiment a1_with_aoi --checkpoint <path>/best_agent.pt --num_envs 4096 --num_episodes 1 --output eval.json` runs to completion without error.
2. The output JSON contains a `"timeseries"` key with per-timestep data (rmse, sqrt_trace, visibility, distance, viewing_angle) compatible with `plot_timeseries.py`.
3. The output JSON contains a `"trajectories"` key with agent/target XYZ paths compatible with `plot_trajectory.py`.
4. Running `plot_timeseries.py` on the output produces a `timeseries.pdf` with the same subplot layout as the reference (3x2 grid: RMSE, uncertainty, visibility, distance, viewing angle, legend).
5. Running `plot_trajectory.py` on the output produces a `trajectory.pdf` with the same layout as the reference (3D paths + viewing angle/RMSE panel).

**Flow**: Full QRISPY
