# Chapter 5: Practical Debugging Guide for iris_ma6

> **Reading time**: ~35 minutes.
> **Prerequisites**: Chapters 1-4 (the theory), plus access to your Tensorboard logs.
> **Format**: Field manual -- less math, more "see X → do Y."

---

## 5.1 Your Tensorboard Dashboard: What Each Curve Means

You have **5 standard PPO curves** (per agent) and **~15 task-specific curves**. Here is every one, what it tells you, and what "healthy" looks like.

### Standard PPO Curves

| Curve | What it measures | Healthy range | Alarm signal |
|---|---|---|---|
| **Learning / Learning rate** | KL-adaptive scheduler output | 0.0002 - 0.003, oscillating | Monotonic decay to floor; stuck at min_lr |
| **Policy / Standard deviation** | Mean σ across 7 action dims | 0.3 - 1.5; rises then stabilizes | Collapse to < 0.1 (entropy death) or runaway > 2.0 |
| **Loss / Policy loss** | Negative of clipped surrogate, averaged | −0.005 to +0.005 | Sustained > +0.01 (policy can't improve); spikes > 0.015 |
| **Loss / Value loss** | MSE between V_ϕ prediction and GAE target | 0.0001 - 0.005 | Sustained growth > 0.01 (value divergence) |
| **Loss / Entropy loss** | $-c_e \cdot H[\pi]$ | −0.025 to +0.003 | Positive and growing (entropy collapsing) |

### Task-Specific: Reward Decomposition

These are the most important curves for understanding *what the drones are doing*. Each is an **episode-averaged, time-normalized** reward component.

| Curve | What it measures | Good sign | Bad sign |
|---|---|---|---|
| **Episode_Reward/bbox_center** | Target centered in camera frame | Rising toward max (~60 × dt) | Flat or declining after curriculum shift |
| **Episode_Reward/bbox_size** | Target occupies ~20% of image | Rising toward max | Saturating far below max (drones too far/close) |
| **Episode_Reward/triangulation** | Multi-camera localization quality | Rising as coordination curriculum engages (60k+) | Flat after 100k (agents not coordinating) |
| **Episode_Reward/action_sum** | Action magnitude penalty | Small negative, stable | Growing more negative (jerky high-energy control) |
| **Episode_Reward/action_delta** | Action smoothness penalty | Small negative, stable | Growing more negative (oscillating commands) |
| **Episode_Reward/cbf_penalty** | Continuous collision barrier violation | Near zero | Large negative spikes (near-miss events) |
| **Episode_Reward/collision** | Discrete collision events | Zero | Any non-zero values |
| **Episode_Reward/altitude** | Below-minimum-altitude penalty | Zero | Non-zero (drones diving below 2m) |
| **Episode_Reward/target_proximity** | Too-close-to-target penalty | Zero | Non-zero (drones crowding the target) |

### Task-Specific: Tracking Performance

| Curve | What it measures | Target | Alarm |
|---|---|---|---|
| **Detection/pair_valid_rate** | Fraction of steps where 2+ agents see target | > 0.9 | Declining below 0.8 |
| **Detection/all_invalid_rate** | Fraction of steps where no agent sees target | < 0.05 | Rising above 0.1 |
| **Termination/tracking_lost_fraction** | Fraction of episode in lost-tracking state | < 0.05 | Rising above 0.1 |

### Task-Specific: Safety

| Curve | What it measures | Target | Alarm |
|---|---|---|---|
| **Safety/collision_fraction** | Collision steps / total steps | 0 | Any sustained non-zero |
| **Safety/collision_per_env** | Raw collision count per episode | 0 | > 0.5 (multiple collisions per episode) |
| **Action_Smoothness/total_rms** | RMS of action changes | Stable, moderate | Growing (control oscillation) |

---

## 5.2 Reward Decomposition: The Skill You Need Most

Total reward going up doesn't mean the policy is doing what you want. Total reward going down doesn't mean the policy is failing. **You must look at the individual reward terms.**

### Example scenario: Total reward rises but tracking degrades

| Step | Total reward | bbox_center | triangulation | action_sum |
|---:|---:|---:|---:|---:|
| 50k | 3000 | 45 | 2 | -8 |
| 100k | 3200 | 40 | 3 | **-3** |

Total reward rose by 200. But bbox_center *dropped* by 5. The improvement came entirely from action_sum becoming less negative (smoother, lower-energy control). The policy learned to be lazy -- it reduced penalties without improving tracking.

**Diagnosis**: The action penalties are too strong relative to the tracking rewards. The policy found it easier to reduce action magnitude than to improve tracking.

**Fix**: Reduce `action_sum_penalty_scale` from -10 to -5, or increase `bbox_center_reward_scale` from 60 to 80.

### Example scenario: Reward drops at curriculum transition

| Step | Total | bbox_center | triangulation | cbf_penalty | Notes |
|---:|---:|---:|---:|---:|---|
| 18k | 3100 | 50 | 0 | 0 | Before safety curriculum |
| 22k | 2800 | 48 | 0 | **-250** | Safety curriculum onset (20k) |
| 30k | 3050 | 47 | 0 | -50 | Adapted to safety constraints |

Total reward dropped 300 at step 22k. Looks alarming. But the decomposition shows: bbox_center barely changed, and the entire drop came from the new cbf_penalty. The policy hasn't degraded -- it just hasn't learned the new constraint yet. By 30k it adapts, and cbf_penalty shrinks.

**Diagnosis**: This is healthy. The curriculum introduced a new penalty, and the policy needs time to learn collision avoidance.

**Non-action**: Wait. If cbf_penalty doesn't shrink by 40k, then investigate.

---

## 5.3 Failure Mode Catalog

### Failure Mode 1: LR Collapse

**What you see in Tensorboard**:
- Learning rate decays monotonically to min_lr floor
- Policy std freezes (no exploration changes)
- Value loss starts rising after LR hits floor
- Total reward plateaus then slowly degrades
- pair_valid_rate declines

**Causal chain**: Covered in Chapter 4. KL exceeds threshold repeatedly → cascading LR cuts → policy frozen.

**Leading indicator**: LR. It crashes *before* task metrics degrade (typically 5k-10k steps earlier).

**Fixes (in order of preference)**:
1. Raise `min_lr` to 3e-4
2. Reduce `learning_epochs` from 6 to 4
3. Raise `kl_threshold` from 0.02 to 0.03
4. Check if a curriculum phase change coincided with the crash

---

### Failure Mode 2: Entropy Collapse (σ → 0)

**What you see in Tensorboard**:
- Policy std drops monotonically below 0.2 and keeps falling
- Entropy loss becomes positive (entropy is very low)
- Policy loss becomes strongly negative (exploiting one strategy)
- Total reward rises fast initially, then hard-plateaus
- pair_valid_rate might be decent but never improves further

**Causal chain**: The policy found one okay strategy and committed to it too hard. With σ ≈ 0.1, the policy is nearly deterministic. It can't explore alternatives. The entropy bonus ($c_e = 0.01$) is too weak to counteract the exploitation gradient.

**Leading indicator**: Policy std. If it drops below 0.2 by step 20k and shows no sign of rebounding, the policy is locking in.

**Fixes**:
1. Increase `entropy_loss_scale` from 0.01 to 0.03 or 0.05
2. Raise `min_log_std` (prevent σ from going below ~0.3)
3. Reduce reward scale for the dominant reward term (usually bbox_center) so the exploitation gradient weakens

**Distinguishing from healthy Phase 1**: In healthy training, σ drops to ~0.33 then rebounds by step 30k-40k. In entropy collapse, σ drops to < 0.2 and stays there. Check at step 40k: if σ < 0.25, it's likely collapse.

---

### Failure Mode 3: Value Function Divergence

**What you see in Tensorboard**:
- Value loss grows monotonically (0.001 → 0.01 → 0.05 → ...)
- Policy loss becomes erratic (large positive and negative swings)
- Total reward oscillates wildly between updates
- LR may be healthy (this is not a scheduler problem)

**Causal chain**: The value function can't learn the return distribution. This corrupts advantage estimates, which corrupts the policy gradient. Common causes:
- Reward structure changed too fast (aggressive curriculum)
- Value network too small for the complexity of the global state
- `value_loss_scale` too low (value gradient overwhelmed by policy gradient)

**Leading indicator**: Value loss. If it consistently rises over 10k+ steps rather than oscillating, the critic is falling behind.

**Fixes**:
1. Check if curriculum transitions are too abrupt. Extend ramp durations.
2. Increase `value_loss_scale` from 1.0 to 2.0 (give the critic more gradient budget)
3. Slow the curriculum: the value function needs time to catch up after each difficulty increase
4. Consider separate learning rates for actor and critic (SKRL supports this by using separate optimizers)

---

### Failure Mode 4: Reward Hacking

**What you see in Tensorboard**:
- Total reward rises, but...
- pair_valid_rate is low or declining
- tracking_lost_fraction is rising
- One reward term dominates the total (e.g., action_sum penalty goes to near-zero)
- Collision metrics might be rising

**Causal chain**: The policy found a strategy that maximizes total reward without actually accomplishing the task. Examples:
- Flying far away to avoid collisions and altitude penalties (zero negative rewards), at the cost of poor tracking (reduced positive rewards, but not enough to offset)
- Holding gimbal still to minimize action_delta penalty, even though the target is moving
- Hovering at one safe position (low action_sum) rather than actively tracking

**Leading indicator**: Divergence between total reward and task metrics (pair_valid_rate, tracking_lost_fraction). If reward rises but task metrics don't improve or get worse, the policy is gaming the reward.

**Fixes**:
1. Identify which reward term the policy is gaming (the one showing disproportionate improvement)
2. Reduce scale of the gamed term, increase scale of the task-critical terms
3. Add a direct penalty for the undesired behavior (e.g., if drones fly too far, add a range penalty)
4. Check reward term interactions -- sometimes reducing one penalty creates a loophole in another

---

### Failure Mode 5: Agent Symmetry (Both Drones Do the Same Thing)

**What you see in Tensorboard**:
- Both agents' reward curves track identically
- triangulation reward is low (good triangulation needs diverse viewpoints)
- bbox_center rewards are similar for both agents
- pair_valid_rate might be high (both see the target), but from the same angle

**Causal chain**: Without explicit role differentiation, both drones learn the same "best individual" strategy. This is locally optimal for each agent but globally suboptimal (two identical viewpoints give poor triangulation baseline).

**Leading indicator**: triangulation reward staying flat despite coordination curriculum engaging (60k+). Also compare the two agents' action distributions -- if their std devs and policy losses track identically, they haven't differentiated.

**Fixes**:
1. Increase `triangulation_reward_scale` (stronger incentive for geometric diversity)
2. Add observation features that break symmetry (agent index, relative position to partner)
3. Use asymmetric initialization (start one drone at a different position)
4. Wait -- sometimes differentiation emerges late (after 100k+ steps). Check if it's actually a problem before fixing.

---

### Failure Mode 6: Curriculum Too Fast

**What you see in Tensorboard**:
- Reward drops sharply at a curriculum boundary and *never recovers*
- Value loss spikes and stays elevated
- LR may crash (Failure Mode 1 triggered by curriculum)
- pair_valid_rate drops at the same step

**Causal chain**: The difficulty increased faster than the policy could adapt. The value function's predictions are wrong for the new difficulty, advantages are noisy, and the policy gradient is unreliable.

**Leading indicator**: Value loss spiking at a curriculum boundary and not recovering within 10k steps. Also: total reward dropping > 20% at a boundary and not recovering.

**Diagnosis step**: Check which curriculum phase triggered the drop. Map the step number to the curriculum table:

| Steps | Phase | What changed |
|---|---|---|
| 20k-40k | Agent velocity ramp + safety intro | Faster drones, collision penalty |
| 40k-80k | Target dynamics | Harder-to-track target |
| 60k-100k | Coordination | Triangulation reward scales up |
| 100k-120k | Noise + FP/FN | Observation quality degrades |
| 120k-140k | Fixed delay | Communication latency |
| 140k-160k | Random delay | Latency variance |
| 160k-180k | Dropout | Packet loss |
| 180k-200k | Dynamics randomization | Gain variation |
| 200k-220k | Burst dropout | Sustained outages |

**Fixes**:
1. Extend the offending phase's ramp (e.g., 20k → 40k becomes 20k → 60k)
2. Reduce the phase's maximum difficulty
3. Add a "warmup" LR reset at the curriculum boundary
4. Increase `min_lr` so the policy can adapt even when the scheduler cuts LR

---

## 5.4 The Decision Tree

When you open Tensorboard and something looks wrong, follow this flowchart:

```
START: Something looks wrong in Tensorboard
│
├─ Is Learning Rate at the floor?
│  ├─ YES → Failure Mode 1 (LR Collapse)
│  │        Check: Did it coincide with a curriculum step?
│  │        Fix: min_lr, learning_epochs, kl_threshold
│  │
│  └─ NO → Continue...
│
├─ Is Policy Std < 0.2 and still dropping?
│  ├─ YES → Failure Mode 2 (Entropy Collapse)
│  │        Fix: entropy_loss_scale, min_log_std
│  │
│  └─ NO → Continue...
│
├─ Is Value Loss growing monotonically?
│  ├─ YES → Failure Mode 3 (Value Divergence)
│  │        Check: Is there a curriculum transition nearby?
│  │        Fix: Slow curriculum, increase value_loss_scale
│  │
│  └─ NO → Continue...
│
├─ Is Total Reward rising but task metrics (pair_valid, tracking_lost) not improving?
│  ├─ YES → Failure Mode 4 (Reward Hacking)
│  │        Check: Which reward term is improving disproportionately?
│  │        Fix: Rebalance reward scales
│  │
│  └─ NO → Continue...
│
├─ Is Reward dropping sharply at a specific step?
│  ├─ YES → Check the curriculum table (Section 5.3, FM 6)
│  │        Is it recovering within 10k steps?
│  │        ├─ YES → Normal curriculum transition (wait)
│  │        └─ NO  → Failure Mode 6 (Curriculum Too Fast)
│  │                  Fix: Extend ramp, reduce difficulty, raise min_lr
│  │
│  └─ NO → Continue...
│
├─ Are both agents' curves identical?
│  ├─ YES → Failure Mode 5 (Agent Symmetry)
│  │        Check: Is triangulation reward growing?
│  │        Fix: Increase triangulation_reward_scale
│  │
│  └─ NO → Training is probably healthy.
│          Check individual reward terms for optimization opportunities.
```

---

## 5.5 Quick Reference: Hyperparameter Cheat Sheet

When to adjust what:

| Symptom | Primary knob | Direction | Secondary knob |
|---|---|---|---|
| LR crashes to floor | `min_lr` | Raise (1e-4 → 3e-4) | `learning_epochs` (reduce) |
| σ collapses to zero | `entropy_loss_scale` | Raise (0.01 → 0.05) | `min_log_std` (raise) |
| σ explodes beyond 2.0 | `entropy_loss_scale` | Lower (0.01 → 0.005) | `max_log_std` (lower) |
| Value loss diverges | Curriculum ramp duration | Extend | `value_loss_scale` (raise) |
| Policy loss spikes | `grad_norm_clip` | Lower (0.3 → 0.2) | `ratio_clip` (lower to 0.15) |
| Reward hacking | Gamed reward term scale | Reduce | Task reward scale (raise) |
| Drones don't coordinate | `triangulation_reward_scale` | Raise (5 → 15) | Coordination curriculum onset (earlier) |
| Jerky control | `action_delta_penalty_scale` | Raise (-5 → -10) | -- |
| Drones crash | `collision_penalty_scale` | Raise (-100 → -200) | Safety curriculum onset (earlier) |
| Slow learning | `learning_rate` | Raise (3e-4 → 1e-3) | `learning_epochs` (raise) |

---

## 5.6 The 30-Second Tensorboard Check

When you start a training run, check these 4 things at 5k-step intervals:

1. **LR**: Is it oscillating (healthy) or monotonically decaying (trouble)?
2. **σ**: Is it rebounding after initial drop (healthy) or still falling (entropy collapse)?
3. **Value loss**: Is it stable or slowly growing (fine) or spiking (curriculum issue)?
4. **pair_valid_rate**: Is it above 0.9 (great) or below 0.8 (tracking problem)?

If all four are fine, the run is on track. If any one is off, go to the decision tree in Section 5.4.

---

## 5.7 Curriculum Transition Survival Guide

Your curriculum has ~9 phase transitions over 220k steps. Each transition is a potential instability point. Here's what to expect and when to intervene:

| Transition | Step | Expected impact | Watch for | Intervene if... |
|---|---|---|---|---|
| Safety intro | 20k | Reward dips ~10% from cbf_penalty | LR crash (FM1) | Reward not recovering by 30k |
| Agent velocity | 20k-40k | Slight bbox_center dip (faster drones = harder tracking) | Entropy collapse from faster dynamics | σ < 0.25 at 30k |
| Target dynamics | 40k | bbox_center dip, value loss spike | Value divergence (FM3) | Value loss > 0.005 at 60k |
| Coordination | 60k | triangulation reward starts growing | Agent symmetry (FM5) | triangulation flat at 80k |
| Noise | 100k | pair_valid_rate dips, bbox rewards dip | tracking_lost rising | pair_valid < 0.7 at 120k |
| Fixed delay | 120k | All metrics dip moderately | LR crash from distribution shift | LR at floor by 130k |
| Random delay | 140k | Further metric degradation | Compounding with previous phase | pair_valid < 0.6 at 160k |
| Dropout | 160k | pair_valid drops (by design) | tracking_lost spike | tracking_lost > 0.3 at 180k |
| Dynamics | 180k | Action smoothness degrades | Value divergence | Value loss > 0.01 at 200k |
| Burst dropout | 200k | pair_valid drops further | Cascading failures from accumulated difficulty | Any FM1/FM3 symptoms |

**General rule**: After any transition, give the policy 1.5-2x the ramp duration to stabilize before concluding it has failed. A 20k ramp (20k-40k) needs until ~50k to fully adapt. A 20k ramp (160k-180k) might need until 210k because multiple difficulties are compounding.

---

## 5.8 Quiz

**Q1**: You open Tensorboard and see: LR = 0.0008 (stable), σ = 1.1 (stable), value loss = 0.003 (rising slowly), total reward = 3800 (plateau for 20k steps), pair_valid_rate = 0.92 (good), triangulation = 1.2 (flat for 20k steps). What's the likely issue? What do you check first?

<details>
<summary>Answer</summary>

The issue is likely **agent symmetry** (Failure Mode 5) or **insufficient coordination incentive**.

Evidence:
- PPO metrics are healthy (LR stable, σ stable)
- Tracking is good (pair_valid = 0.92)
- But triangulation reward is flat despite the coordination curriculum being active (if this is past 60k)
- The policy has plateaued -- it found a good tracking strategy but isn't optimizing for geometric diversity

Check first: Are both agents' bbox_center and bbox_size curves nearly identical? If yes, both drones are watching the target from similar angles, giving poor triangulation baseline.

Fix: Increase `triangulation_reward_scale` from 5 to 10-15 to make coordination a stronger gradient signal relative to individual tracking rewards.
</details>

**Q2**: At step 105k, you see: total reward dropped 15%, pair_valid_rate dropped from 0.91 to 0.78, but LR is still 0.001 and σ is still 1.2. What happened?

<details>
<summary>Answer</summary>

The noise + FP/FN curriculum phase just kicked in (100k-120k). The detections became unreliable (noisy bounding boxes, false positives and false negatives).

The PPO machinery is fine (LR and σ are healthy). The policy just hasn't adapted to the noisier observations yet. The drop in pair_valid_rate is *expected* -- with FP/FN injection, even a good policy will have more "missed" detections.

Check: Is pair_valid recovering by 115k? If yes, the policy is learning to cope with noise. If not, the noise ramp might be too aggressive -- extend the 100k-120k phase to 100k-140k.

The key diagnostic: total reward dropped but LR didn't crash. This means the policy is actively learning (not frozen). Give it time.
</details>

**Q3**: You see action_delta penalty growing steadily more negative (-2 → -5 → -8 over 50k steps), while bbox_center reward is also growing. Total reward is rising. Is this a problem?

<details>
<summary>Answer</summary>

**Maybe**. It depends on the magnitudes.

The policy is learning better tracking (bbox_center rising) at the cost of jerkier control (action_delta penalty growing). If the bbox_center gains outweigh the action_delta costs in total reward, the optimizer is doing exactly what you told it to: maximize total reward.

**When it's fine**: The action_delta cost is small relative to the tracking gains, and the actual control commands are physically feasible (not saturating actuators).

**When it's a problem**: The control jerkiness will transfer badly to the real drone. In simulation, jerky commands are free. On hardware, they cause mechanical stress, oscillation, and sensor blur. If you plan sim-to-real transfer, you should increase `action_delta_penalty_scale` to force smoother control even if it costs some tracking performance.

Check: Look at Action_Smoothness/total_rms. If it's growing beyond what the real hardware can handle (you need to know your actuator bandwidth to judge this), increase the penalty.
</details>

---

**This completes the 5-chapter series.** You now have:
- **Ch 1-2**: The theoretical foundation (policy gradients, PPO)
- **Ch 3**: How theory manifests in your training logs
- **Ch 4**: Deep analysis of one specific failure mode
- **Ch 5**: Practical recipes for everything else you'll encounter

Go train some drones.
