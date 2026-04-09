## Ticket: Pre-allocate AgentStates buffers (A2)

**What**: Pre-allocate persistent `AgentStates` objects in `__init__` and reuse them each step instead of creating fresh instances. Eliminates ~50 `torch.zeros()` GPU allocations per policy step.

**Why**: Every decimation step (25 Hz), the delay system update loop (`iris_ma_env6_test.py:913-974`) creates `AgentStates(num_envs=1024, num_joints=3, num_targets=1)` per agent. Each `AgentStates.__init__` allocates ~25 tensors via `torch.zeros()`. With 2 agents: 50 GPU allocations per step. Similarly, `_build_gt_states()` re-creates AgentStates each time it's called (up to 3x per step for triangulation levels).

**Scope boundary**:
- Pre-allocate `self._delay_gt_buffers: dict[str, AgentStates]` in `__init__`
- Pre-allocate `self._gt_state_cache: dict[str, AgentStates]` for `_build_gt_states()`
- Add a `reset_data()` or `zero_()` method to `AgentStates` that zeros fields without re-allocation
- Pre-allocate `self._dr_intrinsics_per_agent` tensor `(N, A, 3, 3)` instead of cloning `_camera_intrinsics_base` per agent each step
- Do NOT change the AgentStates data schema or field semantics

**Affected modules**:
- `delay_system_v3/agent_states.py` — add `reset_data()` method
- `iris_ma_env6_test.py` — `__init__`, `_update_state_cache()` (lines 913-974), `_build_gt_states()`, intrinsics handling (lines 872-877)

**Estimated impact**: 5-8% wall-clock reduction per step.

**Flow**: Direct implementation. Can be done independently of ticket 017.

**Parent**: 016-cut-training-time (research.md §2.2, §2.6, §4 A2)
