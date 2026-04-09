## Ticket: Batch DroneController across agents (A1)

**What**: Replace N per-agent `DroneController(num_envs=1024)` instances with a single `DroneController(num_envs=1024*N_agents)`. Reshape inputs from `(N, ...)` per agent to `(N*A, ...)` before calling, reshape outputs back after.

**Why**: `_apply_action()` runs at 100 Hz (simulation rate) — 4x more frequent than any other hot-path method. Each `controller.step_policy()` call launches GPU kernels for the full cascaded PID (velocity -> attitude -> rate -> motor). With 2 agents this doubles the kernel launch overhead; with 3 agents it triples. Batching eliminates N-1 redundant kernel launches per physics step.

**Scope boundary**:
- Modify controller instantiation in `__init__` and action application in `_apply_action()`
- The Isaac Sim API calls (`set_external_force_and_torque`, `set_joint_position_target`) remain per-robot (separate Articulation objects) — only the controller math is batched
- Controller `reset()` must handle per-env resets correctly: reset rows `[env_id * A + agent_idx]` for each agent
- Gain randomization must work with the new `(N*A, ...)` state shape

**Affected modules**:
- `controller/drone_controller.py` — constructor, reset, state shapes
- `iris_ma_env6_test.py` — `__init__` (instantiation), `_apply_action()`, `_reset_idx()`

**Key constraint**: Controller has internal state (motor dynamics `_motor.omega`, gimbal integrator, rate controller integral). With batching, these become `(N*A, ...)` shaped — agent states are different rows in the batch. This is natural but reset indexing must be correct.

**Estimated impact**: 15-20% wall-clock reduction per step.

**Flow**: Direct implementation. Benchmark with `torch.cuda.Event` timing before/after.

**Parent**: 016-cut-training-time (research.md §2.1, §4 A1)
