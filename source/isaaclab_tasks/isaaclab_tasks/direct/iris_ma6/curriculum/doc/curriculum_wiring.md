# Curriculum Wiring Reference

Maps each curriculum phase to the variable it controls and where it's applied in the environment.

**Source of truth**: `curriculum_cfg.py` (phase definitions) + `iris_ma_env6_test.py` (application).
**Last verified**: 2026-04-09.

| Phase | Steps | CurriculumCfg Getter | Env Variable | Controls | Env Location |
|---|---|---|---|---|---|
| Agent Velocity | 20k–40k | `get_agent_velocity_progress()` | `progress_agent_velocity` | `_max_lin_vel` per env | `_reset_idx()` |
| Safety | 20k–40k | `get_progress(safety_*)` | `progress_safety` | CBF penalty scale | `_get_rewards()` |
| Tracking | 20k–60k | `get_progress(tracking_*)` | `progress_tracking` | Initial state randomization range | `_reset_idx()` |
| Target Motion | 40k–80k | `get_progress(moving_target_*)` | `progress_moving_target` | Target speed/maneuverability | target controller |
| Coordination | 60k–100k | `get_progress(coordination_*)` | `progress_coord` | Triangulation reward weight | `_get_rewards()` |
| Task Levels | 40k–120k | `get_task_level_progress()` | `l2_prog, l3_prog` | Reward blend FIM→GT→E2E | `_get_rewards()` |
| Noise | 100k–120k | `get_noise_progress()` | `_curriculum_noise_scale` | `delay_system.set_noise_scale()` | `_get_rewards()` |
| FP/FN | 100k–120k | `get_fp_fn_progress()` | `_curriculum_fp_fn_scale` | `apply_detector_replicator(fp_fn_scale=)` | `_get_rewards()` + `_post_physics_step()` |
| Fixed Delay | 120k–140k | `get_fixed_delay_progress()` | `progress_delay` | `set_delay_mode("fixed", progress)` | `_get_rewards()` |
| Random Delay | 140k–160k | `get_random_delay_progress()` | `progress_delay` | `set_delay_mode("random", progress)` | `_get_rewards()` |
| Dropout | 160k–180k | `get_dropout_progress()` | `dropout_progress` | `set_dropout_rate(progress * prob)` | `_get_rewards()` |
| Dynamics | 180k–200k | `get_progress(dynamics_*)` | `progress_dynamics` | Gain randomization, DR, zoom tau | `_reset_idx()` |
| Burst Dropout | 200k–220k | `get_burst_dropout_progress()` | `burst_progress` | `set_burst_params(p_onset * progress)` | `_get_rewards()` |

## Notes

- All curriculum updates happen in `_get_rewards()` (called once per step), except `progress_dynamics` and `progress_tracking` which are updated in `_reset_idx()`.
- `all_end_step` (400k) is informational — last active phase ends at 220k.
- Burst dropout only active when `delay_system_params.burst_dropout_enabled=True` (default: False).
