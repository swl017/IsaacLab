# Ticket 016: Cut Training Time — Research Report

**Date**: 2026-04-09
**Scope**: Computation optimization + curriculum compression for iris_ma6

---

## 1. Current Training Profile

| Parameter | Value |
|-----------|-------|
| Total timesteps | 400,000 |
| Physics rate | 100 Hz (dt=0.01s) |
| Policy rate | 25 Hz (decimation=4) |
| Episode length | 20s (500 policy steps) |
| Parallel envs | 1024 |
| Agents | 2 (expandable to 3) |
| Rollouts | 32 |
| Learning epochs | 6 |
| Mini-batches | 8 |
| Architecture | MAPPO-RNN (GRU-64) |

**Baseline (928b)**: Still learning at 400k — policy loss -0.0023, triangulation 44.4 and rising.

---

## 2. Computation Bottlenecks

### 2.1 CRITICAL: Per-agent for-loops in `_apply_action()` (100 Hz)

**File**: `iris_ma_env6_test.py:637-746`

Runs at **simulation rate** (every physics step, 4x more frequent than policy). Loops over agents to:
- `torch.stack` 3 gimbal joint positions per agent (line 649-656)
- Call `controller.step_policy()` per agent (line 663) — the most expensive operation
- Call `robot.set_external_force_and_torque()` per agent (line 678)
- Call `robot.set_joint_position_target()` per agent (line 696)

With 2 agents: 2 separate controller forward passes per physics step.
With 3 agents: 3 separate passes.

**Root cause**: Each agent has its own `DroneController` instance (475-line cascaded controller). The controller is batch-vectorized over envs (N=1024) but called once per agent.

### 2.2 CRITICAL: Fresh AgentStates allocation per agent per step

**File**: `iris_ma_env6_test.py:913-974`

Every decimation step (25 Hz), creates `AgentStates(num_envs=1024, ...)` per agent. Each AgentStates allocates ~25 `torch.zeros()` tensors. With 2 agents: **50 GPU allocations per policy step**.

Additionally populates 20+ fields, computes `body_to_world_gimbal_angles()` (trig ops), `compute_combined_angular_velocity()`, and LOS rates per agent.

### 2.3 MAJOR: Redundant `_build_gt_states()` calls

**File**: `iris_ma_env6_test.py:1318-1343`

Triangulation computed at 3 levels in `_get_rewards()`:
- Level 1: uses `reward_states` (already fetched)
- Level 2: calls `_build_gt_states()` (line 1329) — **rebuilds all agent states from scratch**
- Level 3: calls `_get_delayed_states_for_e2e()`

`_build_gt_states()` itself loops over agents, recomputes gimbal angles, combined angular velocity — all of which were already computed in `_update_state_cache()` (lines 913-974).

### 2.4 MAJOR: O(N^2) observation construction

**File**: `iris_ma_env6_test.py:1640-1728`

For each ego agent, loops over other agents building inter-agent obs via `torch.cat` of 8 tensors. With 2 agents: 2 ego iterations x 1 other = 2 inner loops. With 3 agents: 3 x 2 = 6 inner loops.

Each ego observation involves `torch.cat` of 14 tensors (ego) + N-1 `torch.cat` of 8 tensors (others) + a final `torch.cat` to assemble.

### 2.5 MODERATE: Detection stats loop, GT position stacking

**File**: `iris_ma_env6_test.py:1306-1316`

Detection validity counted via per-agent loop with dictionary lookups. GT positions stacked via list comprehension (line 1346-1348). Both trivially vectorizable.

### 2.6 MODERATE: Camera intrinsics clone per agent

**File**: `iris_ma_env6_test.py:872-877, 979`

`self._camera_intrinsics_base.clone()` called per agent at each decimation step. Could use pre-allocated `(N, A, 3, 3)` buffer with in-place multiply.

---

## 3. Curriculum & Training Time Analysis

### 3.1 Current Curriculum Timeline (220k active + 180k post)

```
Phase               Start    End    Duration
─────────────────────────────────────────────
Free warmup         0k       20k    20k
Agent velocity      20k      40k    20k
Safety (CBF)        20k      40k    20k
Tracking/formation  20k      60k    40k
Moving target       40k      80k    40k
Coordination        60k      100k   40k
Noise + FP/FN       100k     120k   20k
Fixed delay         120k     140k   20k
Random delay        140k     160k   20k
IID dropout         160k     180k   20k
Dynamics DR         180k     200k   20k
Burst dropout       200k     220k   20k
─── curriculum ends ─────────────────────────
Post-curriculum     220k     400k   180k (45% of budget!)
```

### 3.2 Key Experiment Findings

**From no-curriculum ablation (a07987a412)**:
- Policy masters step-0 difficulty in **8-16k steps** (entropy turns positive at 16k)
- 20k warmup wastes ~4-8k steps of compute
- Value loss drops to 0.0002 by 20k — value function starts overfitting if curriculum doesn't ramp

**From compressed curriculum (7e46515b39)**:
- 20k/phase compressed schedule works **better** than slower variants
- Triangulation ramps faster: 0 -> 47 by 120k (vs slow curriculum: 0 -> 44 by 164k)
- Prevents value function overfitting to any single difficulty regime

**From baseline (928b9585f2)**:
- **Post-curriculum (220k-400k)**: reward goes from 3365 -> 3788 (+423, 12.6% improvement)
- **Triangulation**: 39.8 (240k) -> 44.4 (400k) — still climbing but rate slowing
- **Policy loss**: -0.0023 at 400k — still improving but diminishing returns
- **Dynamics phase (180-200k)**: Biggest instability, takes ~60k steps to recover

**From action penalty ablation (40aab / a9_bbox)**:
- High action penalties (-30/-15) cause exploration collapse (sigma -> 0.16)
- Conservative penalties (-2/-1) maintain healthy sigma ~1.21
- Detector replicator + burst dropout interact poorly with high penalties

### 3.3 Where Learning Actually Happens

| Phase | Steps | Learning Signal |
|-------|-------|----------------|
| Warmup | 0-16k | pair_valid_rate 0.65->0.95, entropy convergence |
| Geometry | 20-60k | Smooth ramp, no cliff |
| Target motion | 40-80k | Concentration at 60-80k |
| Coordination | 60-100k | Triangulation 1.1 -> 40.7 (rapid) |
| Noise | 100-120k | No cliff, pair_valid stays >0.83 |
| Delay | 120-140k | Dip at 120k, recovery by 140k |
| Dynamics | 180-200k | **Recovery phase, not learning** — value loss spike |
| Post-curriculum 220-300k | Steady improvement (~+300 reward) |
| Post-curriculum 300-400k | Diminishing returns (~+100 reward) |

---

## 4. Optimization Opportunities

### Track A: Computation (wall-clock per step)

| ID | Optimization | Frequency | Impact | Effort |
|----|-------------|-----------|--------|--------|
| A1 | Batch controller across agents | 100 Hz | **HIGH** | Medium |
| A2 | Pre-allocate AgentStates buffers | 25 Hz | **MEDIUM** | Low |
| A3 | Cache GT states, deduplicate triangulation | 25 Hz | **MEDIUM** | Low |
| A4 | Vectorize observation construction | 25 Hz | LOW-MED | Medium |
| A5 | Vectorize state cache reads | 25 Hz | LOW | Low |

**A1 detail**: Create a single `DroneController(num_envs=N*A)` instead of A separate instances with `num_envs=N`. Reshape all inputs from `(N, A, ...)` to `(N*A, ...)` before calling, reshape outputs back after. The Isaac Sim API calls (`set_external_force_and_torque`, `set_joint_position_target`) must remain per-robot (separate Articulation objects), but these are thin buffer writes — the expensive math is in the controller.

**A2 detail**: Pre-allocate `self._gt_state_buffers[agent_id] = AgentStates(...)` once in `__init__`. Overwrite field data each step instead of re-allocating. Eliminates ~50 `torch.zeros()` GPU allocations per policy step.

**A3 detail**: After the delay system update (lines 913-1021), store the GT states dict. When `_build_gt_states()` is called for L2 triangulation, return the cached version. Saves one full GT state rebuild + one triangulation pass per step.

### Track B: Curriculum (total training steps)

| ID | Optimization | Saves | Risk |
|----|-------------|-------|------|
| B1 | Compress warmup: 20k -> 12k | 8k steps (2%) | Low |
| B2 | Compress post-curriculum: 180k -> 100k | 80k steps (20%) | Low (checkpoint safety) |
| B3 | Gentle dynamics ramp: 20k -> 30k steps | Enables shorter recovery | Low |
| B4 | Merge adjacent phases (tighter overlap) | 20k steps (5%) | Medium |

**B2 detail**: The post-curriculum phase currently runs 220k-400k (180k steps = 45% of budget). Evidence shows:
- 220k-300k: meaningful improvement (+300 reward, triangulation climbing)
- 300k-400k: diminishing returns (~+100 reward, triangulation gain slowing)

Proposal: total 320k steps (post-curriculum 220k-320k = 100k steps). Save checkpoints at 280k/300k/320k as safety net.

---

## 5. Proposed Compressed Schedule

Combining B1+B2+B3 (conservative variant):

```
Phase               Start    End    Duration  (was)
─────────────────────────────────────────────────────
Free warmup         0k       12k    12k       (20k)
Agent velocity      12k      32k    20k       (same)
Safety              12k      32k    20k       (same)
Tracking            12k      52k    40k       (same)
Moving target       32k      72k    40k       (same)
Coordination        52k      92k    40k       (same)
Noise + FP/FN       92k      112k   20k       (same)
Fixed delay         112k     132k   20k       (same)
Random delay        132k     152k   20k       (same)
IID dropout         152k     172k   20k       (same)
Dynamics DR         162k     192k   30k       (was 20k, gentler)
Burst dropout       192k     212k   20k       (same)
─── curriculum ends ─────────────────────────────────
Post-curriculum     212k     320k   108k      (was 180k)
─────────────────────────────────────────────────────
Total: 320k (was 400k) — 20% reduction
```

---

## 6. Priority Matrix

| Priority | Item | Type | Est. Impact | Effort |
|----------|------|------|-------------|--------|
| **P0** | B2: Shorten post-curriculum | Config | 20% fewer steps | Trivial |
| **P1** | A1: Batch controller | Code | 15-20% faster steps | 1-2 days |
| **P2** | A2: Pre-allocate AgentStates | Code | 5-8% faster steps | Half day |
| **P2** | A3: Cache GT states/triangulation | Code | 3-5% faster steps | Half day |
| **P2** | B3: Gentle dynamics ramp | Config | Enables shorter post-curriculum | Trivial |
| **P3** | B1: Compress warmup | Config | 2% fewer steps | Trivial |
| **P3** | A4: Vectorize observations | Code | 3-5% faster steps (more with 3 agents) | Medium |
| **P4** | A5: Vectorize state cache | Code | 1-2% faster steps | Low |

**Combined estimated savings**: 35-45% wall-clock reduction (20% fewer steps + 15-20% faster per-step).

---

## 7. Verification Plan

1. **Computation benchmarks**: Use `torch.cuda.Event` timing around each lifecycle method before/after optimization. Compare average step time over 1000 steps.
2. **Training parity**: Run 320k compressed schedule, compare 928b metrics at matching curriculum milestones (not absolute step counts).
3. **Checkpoint safety**: Save at 280k/300k/320k. If triangulation gain from 280k->320k < 2.0, the compression is validated.