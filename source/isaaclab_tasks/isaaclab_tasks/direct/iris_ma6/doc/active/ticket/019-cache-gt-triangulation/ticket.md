## Ticket: Cache GT states and deduplicate triangulation (A3)

**What**: Cache the ground-truth agent states computed during `_update_state_cache()` and reuse them in `_get_rewards()` instead of rebuilding. Avoid redundant `_build_gt_states()` and `_compute_triangulation()` calls.

**Why**: In `_get_rewards()` (lines 1318-1343), triangulation is computed at up to 3 levels. Level 2 calls `_build_gt_states()` (line 1329) which loops over all agents, recomputing gimbal world-frame angles, combined angular velocity, and camera geometry — all of which were already computed in the delay system update block (lines 913-974). The same data is rebuilt from scratch instead of being reused.

**Scope boundary**:
- After the delay system update in `_update_state_cache()`, store the GT states dict as `self._cached_gt_states`
- Make `_build_gt_states()` return `self._cached_gt_states` when the delay system is active (the delay update already computes identical data)
- Cache `body_to_world_gimbal_angles()` results per agent in `_update_state_cache()` so both the delay update and `_build_gt_states()` read from cache
- Do NOT change triangulation logic or reward semantics

**Affected modules**:
- `iris_ma_env6_test.py` — `_update_state_cache()`, `_build_gt_states()`, `_get_rewards()`

**Estimated impact**: 3-5% wall-clock reduction per step.

**Flow**: Direct implementation. Independent of tickets 017 and 018.

**Parent**: 016-cut-training-time (research.md §2.3, §4 A3)
