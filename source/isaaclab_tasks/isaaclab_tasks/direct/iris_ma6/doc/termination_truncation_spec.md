# Termination and Truncation Spec for iris_ma6

## 1. Motivation

The current termination conditions (altitude crash at z < 0.5m, timeout) leave a gap:
episodes where all drones survive but **no agent detects the target** continue to
run, burning training compute with zero useful reward signal.

Experiment `9db8a11b47_altitude_penalty` (2026-03-27) shows this failure mode:
- Episodes collapse to ~52 steps at 24k due to altitude crashes.
- Even when drones survive, the 140k curriculum collapse produces long stretches
  where `all_invalid_rate` reaches 20-48% — extended periods of blind flight that
  generate no learning signal.

Terminating when all agents lose track for a sustained duration would:
1. Free training compute for fresh resets with learnable initial conditions.
2. Sharpen the reward signal by cutting dead-weight episodes.
3. Create an explicit incentive: **maintain at least one detection at all times.**

## 2. Existing Termination and Truncation Conditions

| Condition | Type | Trigger | Current Config |
|-----------|------|---------|----------------|
| Altitude crash | Terminated | Any agent `z < 0.5m` | Hard floor safety net |
| Timeout | Truncated | `episode_length >= max_episode_length` | 499 steps (19.96s) |
| Collision | Disabled | `collided` (commented out in `_get_dones`) | Was penalty-only |

Detection tracking already exists in `_get_rewards()` (lines 1128-1137):
```python
num_valid_detections = sum(agent_valid.float() for each agent)
self._detection_stats["invalid_all_agents"] += (num_valid_detections == 0).float()
```

## 3. Proposed: All-Lost Tracking Termination and Truncation

### 3.1 Definition

Terminate an episode when **no agent has a valid detection** for `N` consecutive
steps. A "valid detection" means the delayed bounding box for target 0 is non-zero
(`bbox.abs().sum() > 1e-6`), using the same check as the existing detection stats.

### 3.2 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_tracking_termination` | `bool` | `True` | Master switch |
| `tracking_lost_timeout_s` | `float` | `1.0` | Seconds of all-agents-blind before termination |
| `tracking_termination_grace_steps` | `int` | `50` | Steps after reset before the counter starts (lets policy establish initial detections) |

Derived: `tracking_lost_timeout_steps = int(tracking_lost_timeout_s / (sim_dt * decimation))`
At default config (dt=0.01, decimation=4): `1.0 / 0.04 = 25 steps`.

### 3.3 State

New per-environment counter in `__init__`:
```
_all_lost_counter: Tensor[num_envs]  # consecutive steps with zero detections
```

Reset to 0 in `_reset_idx()`.

### 3.4 Logic (in `_get_dones`)

```
# Count valid detections this step (reuse from _get_rewards or recompute)
all_blind = (num_valid_detections == 0)

# Grace period: don't count during first N steps after reset
in_grace = (episode_length_buf < tracking_termination_grace_steps)

# Update counter
_all_lost_counter = where(all_blind & ~in_grace,
                          _all_lost_counter + 1,
                          0)

# Truncate
tracking_lost = (_all_lost_counter >= tracking_lost_timeout_steps)
```

This is a **truncated** condition (not terminated), because the agents are still
alive and could theoretically recover detections — we are cutting the episode
short as a training efficiency decision. The value function should bootstrap
V(s_last) rather than learning V=0, preserving the policy's incentive to keep
flying even when detections are temporarily lost.

### 3.5 Curriculum Gating

The termination should **not** activate before the policy has had a chance to learn
basic detection. Two options:

**Option A — Grace-steps only (recommended):**
The `tracking_termination_grace_steps` parameter (default 50 = 2s) provides a
warm-up window per episode. Combined with `gimbal_curriculum_mode="always_pointing"`
(gimbals start on target), this gives the policy time to establish detections
before the clock starts. No global curriculum gate needed.

**Option B — Curriculum-gated activation:**
Only enable after `progress_tracking > threshold` (e.g., 0.5, ~30k steps).
This avoids punishing the policy before it can reasonably track. Adds one more
config knob but is safer for early training.

Recommendation: start with Option A. If early training shows premature terminations,
add Option B.

## 4. Integration

### 4.1 Config (`iris_ma_env6_test_cfg.py`)

Add to the safety/termination section:
```python
enable_tracking_termination: bool = True
tracking_lost_timeout_s: float = 1.0
tracking_termination_grace_steps: int = 50
```

### 4.2 Environment (`iris_ma_env6_test.py`)

**`__init__`**: Initialize `_all_lost_counter` tensor.

**`_get_rewards`**: Expose `num_valid_detections` as `self._num_valid_detections`
so `_get_dones` can read it without recomputing.

**`_get_dones`**: Add tracking-lost check after existing crash/collision checks.
Apply as `truncated[agent_id] |= tracking_lost` for all agents (if any one
agent's episode is over, all are over in this cooperative MARL setup).
Ensure mutual exclusion: `truncated &= ~terminated` (crash takes priority).

**`_reset_idx`**: Reset `_all_lost_counter[env_ids] = 0`.

### 4.3 Logging

Log to extras at episode end:
- `Termination/tracking_lost_fraction`: fraction of resets caused by tracking loss
  (vs crash, vs timeout). Useful for diagnosing whether the termination fires
  too often or too rarely.

### 4.4 Calling Contract

| Method | Type | Frequency | Notes |
|--------|------|-----------|-------|
| Counter update in `_get_dones` | WRITE | Once per step | Uses `_num_valid_detections` set by `_get_rewards` in the same step. `_get_rewards` is always called before `_get_dones`. |
| Counter reset in `_reset_idx` | WRITE | On reset | Must zero counter for reset env_ids |

## 5. Design Decisions

**Why consecutive steps, not cumulative?**
A cumulative counter (total blind steps in episode) would penalize transient
occlusions that the policy recovers from. Consecutive steps only terminate when the
policy has truly lost the target with no recovery.

**Why truncated, not terminated?**
The agents are still alive and could recover detections. This is an arbitrary
time-limit decision (like the 499-step timeout), not a true task failure.
Using `truncated` lets the value function bootstrap V(s_last) instead of
learning V=0, which preserves the policy's incentive to keep flying and
reacquire the target rather than giving up once detections are lost.

**Why 1 second default?**
- Too short (< 0.5s / 12 steps): transient noise or delay-induced dropouts
  would cause false terminations, especially after the noise curriculum.
- Too long (> 3s): defeats the purpose — 3s of blind flight is already 75 wasted
  steps.
- 1 second (25 steps) balances robustness to noise with early termination of
  truly lost episodes.

**Why grace steps instead of curriculum gating?**
Per-episode grace is simpler, has no global training-step dependency, and works
correctly even when curriculum is disabled (e.g., evaluation). It also handles
the cold-start naturally: gimbals start pointed at target, so detections are
typically valid from step 1. The 50-step grace is a safety margin for the delay
system to propagate initial detections.
