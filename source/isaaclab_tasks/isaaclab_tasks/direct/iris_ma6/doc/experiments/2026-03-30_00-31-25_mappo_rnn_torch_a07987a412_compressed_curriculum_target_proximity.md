# Experiment: 2026-03-30_00-31-25_mappo_rnn_torch_a07987a412_compressed_curriculum_target_proximity

**Commit**: `a07987a412`
**Date**: 2026-03-30
**Experiment ID**: `a1_with_aoi`
**Base**: `4f7d9d0c34`<2026-03-28_21-29-36_mappo_rnn_torch_4f7d9d0c34_gimbal_controller_fix>

**BUG**: Run with `current_step = self.cfg.debug_initial_step` where `debug_initial_step=0`.
The curriculum never advanced — all progress factors stayed at 0 the entire run.
This is effectively a **no-curriculum ablation** at step-0 difficulty.

Ran to 288k.

---

## 1. Hypothesis (intended, not tested)
Compressed curriculum (180k total, 20k per phase) to prevent value function overfitting
to any single difficulty regime. Also added target proximity penalty.

## 2. Configuration Delta (intended)
- Compressed curriculum: 280k → 180k total, 20k per phase
- Added target proximity penalty
- `debug_initial_step=0` **bug**: curriculum progress frozen at 0

## 2b. Actual Configuration (due to bug)
- Static target (1 m/s, direction change every 6-12s)
- Easiest initial positions (progress=0: tight formation, near-planar)
- No safety/CBF penalty ramp (progress_safety=0)
- No coordination/triangulation reward (progress_coordination=0)
- No noise, delay, dropout, or dynamics randomization
- Agent max_lin_vel at minimum (3 m/s)

## 3. Results (288k, no-curriculum ablation)

### 3.1 Training Curves
**Metric: `pair_valid_rate`**
- Converges to 0.96 by 8k and holds 0.96-0.98 for the entire 288k run.
- No cliff, no decline, no oscillation. Perfectly flat.

**Metric: `drone_0_bbox_center`**
- Converges to 54-55 by 20k. Holds at 54-55 for the entire run.
- Saturated near maximum — the task is trivially easy.
- Compare: best curriculum run (4f7d9d0c34) peaked at 51.

**Metric: `Total timesteps (mean)`**
- 498 throughout. No episode length issues.

**Metric: `drone_0_triangulation`**
- Zero forever. Coordination reward never activates (progress=0).

**Metric: `drone_0_cbf_penalty`**
- Zero forever. Safety never ramps (progress=0).

**Metric: `drone_0_collision`**
- -0.2 to -1.3, oscillating. Collisions persist without CBF.
- Tight formations (progress=0) + no preventive CBF signal = persistent collisions.

**Metric: `drone_0_action_sum`**
- Converges to -1.8 by 20k. 4x lower than curriculum run (-7.3).
- The policy barely moves — near-zero actions suffice for a slow target.

**Metric: `Reward / Total reward (mean)`**
- Converges to ~3150 by 20k. Flat for 270k more steps.
- Compare: 4f7d9d0c34 peaked at 4277 (with coordination reward stacking on top).

**Metric: `Policy / Standard deviation`**
- Drops to 0.34 by 20k, stabilizes at 0.46-0.49 for the rest.
- Compare: curriculum run reached 1.08. The easy task allows low exploration.

### 3.2 Loss Analysis (key insight)
**Metric: `Entropy loss`**
- Negative (learning) during 0-12k: -0.009 → -0.001.
- **Turns positive at 16k** (0.001) and stays positive (0.001-0.003) through 288k.
- Positive entropy loss = policy is actively reducing entropy (becoming more deterministic).
- This is the convergence signal: the policy found the optimal strategy and is committing.

**Metric: `Value loss`**
- Drops to 0.0002 by 20k. Stays below 0.002 the entire run.
- Near-zero value loss = the value function accurately predicts returns in a stationary MDP.
- Compare: curriculum run had value loss 0.003-0.006 (non-stationary MDP).

**Metric: `Policy loss`**
- Slightly negative (-0.001 to -0.004) throughout.
- Policy is making small, incremental improvements within the easy task.

### 3.2 Behavior Observations
Not tested.

## 4. Analysis

### What this reveals about curriculum design:
- **The policy masters step-0 difficulty in 8-16k steps.** pair_valid_rate hits 0.96 by 8k,
  bbox_center saturates by 20k, entropy loss turns positive at 16k. Everything after
  is wasted compute — 270k steps of grinding with zero improvement.
- **Entropy loss sign is the convergence signal.** Negative = still learning the current
  difficulty. Positive = ready for harder challenges. The curriculum should start ramping
  when entropy loss turns positive (~16k in this run).
- **Policy std stays at 0.47 without curriculum** (vs 1.08 with). The curriculum is what
  drives exploration higher — each difficulty increase forces the policy to explore again.
  Low std at easy difficulty is healthy and expected.
- **Value loss near-zero confirms overfitting risk.** At 0.0002, the value function has
  memorized the step-0 return distribution. If curriculum starts after this point, the
  value predictions become suddenly wrong, causing curriculum shock. Better to start
  the curriculum while the value function is still plastic (before 20k).
- **CBF must ramp early.** Without safety penalty, collisions persist (-0.5 to -1.0)
  even at easy difficulty with tight formations. The current design (safety 0-20k)
  is correct — agents need CBF signal from the start.
- **Collision penalty without CBF is insufficient.** The -100 collision penalty is reactive
  (only fires on contact). CBF provides a preventive gradient that keeps agents apart.
  Both are needed.

### Curriculum start timing recommendation:
- **20k offset**: Start all ramps at 20k. The policy has converged (entropy positive at 16k)
  and is ready for difficulty, but the value function hasn't fully overfitted yet (value
  loss still slightly declining at 20k). This is the sweet spot.
- Exception: safety/CBF should start at 0 — agents need collision avoidance from the
  beginning since tight formations cause collisions immediately.

## 5. Decision / Todo
- Fix `debug_initial_step` bug (set `current_step` from actual training step, not cfg).
- Apply 20k curriculum offset: all ramps start at 20k except safety (start at 0).
- Re-run with corrected curriculum. Expected: faster convergence to full difficulty,
  no late-stage decline from value function overfitting.
