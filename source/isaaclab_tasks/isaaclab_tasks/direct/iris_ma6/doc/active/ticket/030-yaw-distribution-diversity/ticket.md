## Ticket: Diversify drone yaw initial distribution

**What**: Sample agent body yaw uniformly over `[-π, π]` at reset for **all** agents (including the designated observer). Gimbals continue to point at the target as before — body yaw and gimbal yaw are decoupled.

**Why**: The current default (`other_agents_orientation_mode="face_target"` with `orientation_noise_std=0.2 rad`) keeps body yaw within ~±11° of target bearing. This under-samples the off-axis regime the policy must handle at deployment. Since gimbals always point at the target (`gimbal_curriculum_mode="always_pointing"`), body yaw can be fully randomized without harming the bbox signal at t=0.

**Change**:
- Set body yaw to `uniform(-π, π)` for all agents at reset, independent of target bearing.
- Keep gimbal pointing at target unchanged.

**Affected files**:
- [initial_states/initial_states_generator.py](../../../../initial_states/initial_states_generator.py) — yaw generation branch
- [initial_states/initial_states_cfg.py](../../../../initial_states/initial_states_cfg.py) — defaults (or just the env-level override)
- [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py) — `initial_states=InitialStatesCfg(...)` override

**Acceptance criteria**:
- Empirical yaw histogram across reset is uniform on `[-π, π]`.
- Gimbal still points at target at t=0; bbox non-empty for designated observer.
- No camera jerk on reset (preserves existing zero-velocity + hover-motor init behavior).

**Flow**: Lite — one-line config change plus a histogram test.
