# Teacher-Student Training for Observation-Robust Multi-Agent Policies

**Context**: iris_ma6 multi-agent drone observation environment  
**Problem**: Difficulty training FP/FN and burst dropout resilience via curriculum alone  
**Date**: 2026-04-14

---

## Table of Contents

1. [Problem Diagnosis](#1-problem-diagnosis)
2. [Why Curriculum Alone Fails](#2-why-curriculum-alone-fails)
3. [Approach 1: Asymmetric Actor-Critic (AAC)](#3-approach-1-asymmetric-actor-critic)
4. [Approach 2: Teacher-Student Distillation](#4-approach-2-teacher-student-distillation)
5. [Approach 3: Observation Masking with Auxiliary Losses](#5-approach-3-observation-masking-with-auxiliary-losses)
6. [Approach 4: Information Bottleneck](#6-approach-4-information-bottleneck)
7. [Multi-Agent Considerations](#7-multi-agent-considerations)
8. [Proposed Implementation for iris_ma6](#8-proposed-implementation-for-iris_ma6)
9. [References](#9-references)

---

## 1. Problem Diagnosis

### 1.1 Current Architecture

The iris_ma6 environment trains a shared MAPPO-RNN policy (64-hidden MLP + 64-hidden GRU)
for 2-3 drone agents performing cooperative target observation. Each agent observes:

- **Ego (31D)**: position, velocity, orientation, IMU, gimbal angles, camera ray,
  bbox (normalized xywh), bbox_empty flag, zoom, HFOV, age-of-information
- **Inter-agent (16D per other)**: position, velocity, camera ray, sweep rate, zoom,
  bbox_empty, data_age, bbox_age
- **Optional triangulation tail (6D)**: estimated target position + uncertainty

The curriculum introduces perception degradation across 6 phases over 220k steps:

```
Step:    0k   20k   40k   60k   80k  100k  120k  140k  160k  180k  200k  220k
         |     |     |     |     |     |     |     |     |     |     |     |
AgentVel:[--ramp--]full--------------------------------------------------
Safety:  [--ramp--]full--------------------------------------------------
Tracking:[--------ramp--------]full--------------------------------------
Target:        [----------ramp----------]full----------------------------
Coord:                  [----ramp----]full-------------------------------
Noise:                                [--ramp--]full---------------------
FP/FN:                                [--ramp--]full---------------------  (currently disabled)
FixDelay:                                   [--ramp--]full---------------
RndDelay:                                         [--ramp--]full---------
Dropout:                                                [rmp]full--------
Burst:                                                       [ramp]full--
```

### 1.2 Specific Failure Modes

**Problem 1: Reward signal destruction when FP/FN activates.**
When a false negative (miss) occurs, the bbox observation becomes zero and `bbox_empty=1`.
The bbox-tracking reward components (`bbox_center`, `bbox_size`) produce **zero reward**
rather than negative reward:

```python
# iris_ma_env6_test.py, reward computation
bbox_valid = bbox_raw.abs().sum(dim=-1) > 1e-6   # False when missed
bbox_center_mapped = exp(-10 * center_dist) * bbox_valid.float()  # -> 0.0 when missed
bbox_size_mapped = exp(-|area - 0.2|) * bbox_valid.float()        # -> 0.0 when missed
```

The policy trained on 100k steps of clean observations interprets the sudden reward
drop as "my actions got worse" rather than "my observations got corrupted." This triggers:

```
V_clean(s) >> V_noisy(s)  -->  large negative advantages  -->  policy collapse
```

**Problem 2: Value function aliasing under partial observability.**
The value network sees the **same noisy observations** as the policy (state_space=-1
means shared_obs = concat of all agent obs). When FP/FN or burst dropout corrupts
observations, the value function produces high-variance estimates because it cannot
distinguish "target is lost" from "target moved away." This inflates gradient variance
and destabilizes PPO updates.

**Problem 3: Stale inter-agent data during burst dropout.**
The Gilbert-Elliott burst dropout model (BurstDropoutSampler) freezes **all** inter-agent
data channels simultaneously. During a burst (mean 10 steps = 400ms at 25Hz), the ego
agent sees frozen positions/velocities with growing `data_age`, but the frozen values
may be arbitrarily stale. The policy has no learned fallback because it trained on
clean inter-agent data for 200k steps before burst dropout begins.

**Problem 4: FP/FN curriculum destabilizes training.**
The FP/FN curriculum was enabled in prior experiment runs and caused training
instability, confirming the gradient variance problem described above. The current
run has `fp_fn_background_start_step`, `fp_fn_ramp_start_step`, and `fp_fn_end_step`
set to 4,000,000 (effectively disabled) as a workaround. The two-phase design
(background at 10% from 15k-100k, ramp to full from 100k-120k) is sound in principle
but requires a stable value baseline (e.g., AAC) to survive the transition.

### 1.3 Root Cause Summary

The fundamental issue is that **curriculum-based noise injection destroys the learning
signal that the policy depends on**. The policy must simultaneously:

1. Maintain previously-learned tracking behavior
2. Develop new behaviors for corrupted observations (hold-last, search, graceful degradation)
3. Continue learning coordination under noisy triangulation

This creates conflicting gradient objectives. The value function cannot provide a stable
baseline because it suffers from the same observation corruption as the policy.

---

## 2. Why Curriculum Alone Fails

### 2.1 The Gradient Variance Problem

In standard PPO with GAE, the policy gradient is:

$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|o) \cdot \hat{A}(o, a)\right]$$

where the advantage estimate is:

$$\hat{A}(o, a) = \sum_{t=0}^{T} (\gamma\lambda)^t \delta_t, \quad \delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)$$

When observation noise increases suddenly (curriculum transition), **both** $r_t$ and
$V(o_t)$ become noisy:

- $r_t$ drops because bbox rewards become zero during misses
- $V(o_t)$ becomes unreliable because the value function was fit to clean observations
- The advantage $\hat{A}$ has high variance, producing noisy gradient updates

**Theoretical bound** (Belmega et al., ICML 2025): For a symmetric critic under partial
observability, the mean-squared Bellman error decomposes as:

$$\text{MSBE} = \underbrace{\text{TD error}}_{\text{temporal}} + \underbrace{\text{approx error}}_{\text{capacity}} + \underbrace{\text{aliasing error}}_{\text{partial obs}}$$

The aliasing term arises because multiple true states $s$ map to the same observation
$o$. When FP/FN is active, many states (target visible but missed, target occluded,
target out-of-frame) all produce `bbox_empty=1`, making aliasing severe.

### 2.2 Catastrophic Forgetting During Transitions

Each curriculum phase transition creates a non-stationary MDP. The policy gradient from
the new phase can destructively interfere with previously-learned behavior (Kirkpatrick
et al., 2017). While on-policy RL naturally minimizes forgetting compared to supervised
fine-tuning (MIT, 2025), the effect is not eliminated, especially when the reward
landscape changes dramatically (as it does when bbox rewards drop to zero).

### 2.3 The Teacher-Student Decomposition Insight

The key theoretical insight is that teacher-student methods **decouple perception
learning from control learning**:

| Aspect | Curriculum-Only | Teacher-Student |
|--------|----------------|-----------------|
| Control learning | Must learn from noisy rewards | Learns from teacher's stable actions |
| Perception robustness | Gradual noise exposure | Teacher provides noise-free targets |
| Value estimation | Noisy (same corrupted obs) | Stable (privileged critic) |
| Credit assignment | Confounded by obs noise | Clean (teacher knows true state) |
| Gradient variance | High during transitions | Low (stable baseline) |

---

## 3. Approach 1: Asymmetric Actor-Critic (AAC)

### 3.1 Core Idea

During training, give the **critic (value function)** access to privileged ground-truth
state while the **actor (policy)** sees only the noisy, delayed, dropout-affected
observations. At deployment, the critic is discarded.

```
Training:
  Actor:  pi_theta(a | o_noisy)             -- noisy obs only
  Critic: V_phi(s_privileged)               -- GT state (no noise, no delay, no dropout)

Deployment:
  Actor:  pi_theta(a | o_noisy)             -- unchanged
  Critic: [discarded]
```

### 3.2 Mathematical Formulation

The policy gradient becomes:

$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|o) \cdot \hat{A}^{\text{priv}}(s, a)\right]$$

where the advantage is computed using the **privileged** value function:

$$\hat{A}^{\text{priv}}(s, a) = \sum_{t=0}^{T} (\gamma\lambda)^t \delta_t^{\text{priv}}, \quad \delta_t^{\text{priv}} = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

**Key property**: The aliasing term in the MSBE vanishes because the privileged critic
observes the full state $s$, not the partial observation $o$. The value estimate
$V_\phi(s)$ is low-variance regardless of observation corruption.

### 3.3 Why It Helps with FP/FN and Burst Dropout

1. **Stable value baseline**: When a false negative zeroes the bbox reward, the
   privileged critic still knows the true target position and produces a correct value
   estimate. The advantage correctly attributes the reward drop to observation noise
   rather than to the policy's actions.

2. **Proper credit assignment during bursts**: During a burst dropout, the actor's
   inter-agent observations freeze. The privileged critic sees all agents' true
   positions and can correctly evaluate the joint state, even though the actor is
   operating on stale data.

3. **Curriculum-immune value function**: The critic's accuracy does not degrade during
   curriculum transitions because it never depends on the noisy observation pipeline.

4. **Faster noise curriculum**: With a stable value baseline absorbing gradient
   variance, the noise/FP/FN curriculum can be ramped much faster (estimated 2-5x)
   because the actor's gradient updates remain well-directed.

### 3.4 Privileged State Design for iris_ma6

The privileged state should include everything the reward function uses plus
information that disambiguates observation aliasing:

```
Privileged State (per ego agent, estimated 43D for 3 agents):

  Ego GT (15D):
    [0-2]   gt_body_position_w          (3)  -- true position (no noise)
    [3-5]   gt_body_linear_velocity_w   (3)  -- true velocity
    [6-8]   gt_roll_pitch_yaw           (3)  -- true orientation
    [9-10]  gt_gimbal_yaw_pitch_body    (2)  -- true gimbal angles
    [11-13] gt_camera_ray_w             (3)  -- true ray direction
    [14]    gt_zoom_level               (1)  -- true zoom

  Target GT (6D):
    [15-17] gt_target_position_w        (3)  -- TRUE target position (never exposed to actor)
    [18-20] gt_target_velocity_w        (3)  -- true target velocity

  Per Other Agent GT (11D x (num_agents-1) = 22D for 3 agents):
    [0-2]   gt_other_position_w         (3)  -- true other position
    [3-5]   gt_other_velocity_w         (3)  -- true other velocity
    [6-8]   gt_other_camera_ray_w       (3)  -- true other ray
    [9]     gt_other_zoom               (1)  -- true other zoom
    [10]    gt_other_bbox_empty_true     (1)  -- TRUE detection state (occlusion only, no FP/FN)

  Total: 15 + 6 + 22 = 43D (for 3 agents)
```

**Important**: The privileged state includes `gt_target_position_w` which the actor
never sees. This is the key asymmetry -- the critic can evaluate how well the team
is tracking even when the actor has no detection.

### 3.5 Implementation in iris_ma6

**Environment changes** (iris_ma_env6_test.py):

```python
def _get_states(self) -> Dict[str, torch.Tensor]:
    """Build privileged state for asymmetric critic.

    Uses ground-truth positions, velocities, and detection states.
    No delay, no noise, no FP/FN, no dropout applied.
    """
    states = {}
    for idx, agent_id in enumerate(self.cfg.possible_agents):
        ego_parts = [
            self._root_pos_w[agent_id],           # (N, 3) GT position
            self._root_lin_vel_w[agent_id],        # (N, 3) GT velocity
            wrap_to_pi(torch.stack(
                euler_xyz_from_quat(self._root_quat_w[agent_id]), dim=-1
            )),                                     # (N, 3) GT orientation
            self._gimbal_joint_pos[:, idx, 1:2] - YAW_JOINT_OFFSET,  # (N, 1) GT gimbal yaw
            self._gimbal_joint_pos[:, idx, 0:1],   # (N, 1) GT gimbal pitch
            # GT camera ray (from bbox raycaster, pre-noise)
            self.bbox_raycaster_v2.data.camera_ray_directions_w[:, idx, 0, :],  # (N, 3)
            self.zoom_level[:, idx:idx+1],         # (N, 1) GT zoom
        ]

        target_parts = [
            self._target_pos_w,                    # (N, 3) GT target position
            self._target_vel_w,                    # (N, 3) GT target velocity
        ]

        other_parts = []
        for other_idx, other_id in enumerate(self.cfg.possible_agents):
            if other_id == agent_id:
                continue
            # GT occlusion-only bbox_empty (no FP/FN noise)
            gt_bbox = self.bbox_raycaster_v2.data.bboxes_normalized[:, other_idx, 0, :]
            gt_bbox_empty = (gt_bbox.abs().sum(dim=-1) < 1e-6).float().unsqueeze(-1)

            other_parts.extend([
                self._root_pos_w[other_id],        # (N, 3)
                self._root_lin_vel_w[other_id],    # (N, 3)
                self.bbox_raycaster_v2.data.camera_ray_directions_w[:, other_idx, 0, :],  # (N, 3)
                self.zoom_level[:, other_idx:other_idx+1],  # (N, 1)
                gt_bbox_empty,                     # (N, 1) occlusion only
            ])

        states[agent_id] = torch.cat(ego_parts + target_parts + other_parts, dim=-1)
    return states
```

**Config changes** (iris_ma_env6_test_cfg.py):

```python
state_space: int = 43   # Privileged state dim (15 ego + 6 target + 11*2 others)
```

**Training harness** (run_experiment.py):
The existing code already creates the value network with `shared_observation_spaces`
dimensionality. Setting `state_space=43` will cause the SKRL wrapper to expose a
43D shared observation space, and `MAPPORNNValue` will be instantiated with this size.

**No changes needed to SKRL's MAPPO_RNN** -- it already supports separate observation
spaces for policy and value via `observation_spaces` vs `shared_observation_spaces`.

### 3.6 Theoretical Guarantees

- **Unbiased policy gradient**: The AAC policy gradient is unbiased because the
  privileged critic does not constrain the actor's information set. The actor
  optimizes the same objective as standard PPO; only the baseline changes.
- **Lower variance**: Var($\hat{A}^{\text{priv}}$) $\leq$ Var($\hat{A}^{\text{sym}}$)
  because the privileged value function has lower approximation error.
- **No training-deployment gap in the actor**: The actor trains and deploys on the
  same observation space. The gap is only in value estimation quality, which
  is irrelevant at deployment.

### 3.7 Limitations

- The critic may become **overly optimistic** about what the actor can achieve,
  leading to high advantage estimates for states where the actor is fundamentally
  limited by partial observability.
- Does not directly teach the actor **what to do** during observation dropout --
  it only provides better gradient signal. The actor must still discover dropout-robust
  behaviors through exploration.
- If the observation is so degenerate that no policy can succeed (e.g., 100% dropout
  for extended periods), AAC cannot help.

### 3.8 Key References

- Pinto et al. (2017), "Asymmetric Actor Critic for Image-Based Robot Learning" -- RSS
- Belmega et al. (2025), "A Theoretical Justification for Asymmetric Actor-Critic" -- ICML
- OpenAI et al. (2020), "Learning Dexterous In-Hand Manipulation" -- used AAC with domain randomization
- Weihs et al. (2025), "Informed Asymmetric Actor-Critic: Leveraging Privileged Information"

---

## 4. Approach 2: Teacher-Student Distillation

### 4.1 Core Idea

Decompose the problem into two stages:
1. **Stage 1**: Train a **teacher** policy with privileged (GT) observations using standard RL
2. **Stage 2**: Train a **student** policy with realistic (noisy/delayed/dropout) observations
   by imitating the teacher via supervised learning (behavioral cloning or DAgger)

This separates the two hard problems: learning **what to do** (teacher, Stage 1) from
learning **to perceive** (student, Stage 2).

### 4.2 Stage 1: Teacher Training

The teacher is trained with ground-truth observations (delay system disabled):

```
Teacher observation (same 63D structure, but all GT):
  - Ego: GT position, velocity, orientation, gimbal, ray, bbox (no noise/FP/FN)
  - Inter-agent: GT positions, velocities, rays (no delay, no dropout)
  - Triangulation: GT-anchored (no estimation error)

Teacher training:
  pi_teacher(a | o_GT)  trained via PPO/MAPPO with standard rewards
  ~200k steps to convergence
```

The teacher learns optimal tracking, coordination, and triangulation behavior without
perception challenges. This is the "cheating" baseline that defines the performance
ceiling.

### 4.3 Stage 2: Student Distillation

#### 4.3.1 Behavioral Cloning (BC)

The simplest approach: collect teacher rollouts and train the student to match:

$$\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o_t^{\text{noisy}}, s_t) \sim \mathcal{D}} \left[ \| \pi_{\text{student}}(o_t^{\text{noisy}}) - \pi_{\text{teacher}}(s_t) \|^2 \right]$$

**Problem**: Distribution shift. The student trains on data from the teacher's state
distribution, but its own errors cause it to visit different states. Compounding
error grows as $O(T^2)$ where $T$ is the episode length (Ross et al., 2011).

#### 4.3.2 DAgger (Dataset Aggregation)

DAgger addresses distribution shift by collecting data under the **student's** policy
but labeling with the **teacher's** actions:

```
Algorithm: DAgger for iris_ma6

Initialize: D = {} (empty dataset)
Train teacher pi_T with GT observations

For iteration i = 1..N:
  1. Roll out student pi_S in full-noise environment
     - Collect trajectory: {(o_t^noisy, s_t)}
  2. Query teacher for action labels:
     - a_t^* = pi_T(s_t)  at each state visited by student
  3. Aggregate: D = D + {(o_t^noisy, a_t^*)}
  4. Train student: pi_S = argmin_{pi} sum_{D} ||pi(o^noisy) - a^*||^2
```

**Guarantee**: DAgger achieves $O(1/N)$ regret relative to the best policy in the
student's policy class, vs $O(T)$ compounding error for pure BC.

#### 4.3.3 On-Policy Distillation (Rudin et al., 2022)

The approach used in ANYmal parkour and subsequent locomotion works. Instead of
collecting a dataset, the student is trained online:

```
For each training step:
  1. Student rolls out: a_t ~ pi_S(o_t^noisy), collect reward r_t
  2. Teacher labels: a_t^* = pi_T(s_t)  (privileged state at same timestep)
  3. Loss: L = (1-alpha) * L_PPO(pi_S) + alpha * ||pi_S(o_t^noisy) - a_t^*||^2
  4. Optional: anneal alpha from 1.0 -> 0.0 over training
```

This combines imitation (stable signal from teacher) with RL (adapts to student's
own distribution and can exceed teacher in some states).

### 4.4 Student-Informed Teacher Training (Messikommer et al., ICLR 2025)

**Key insight**: A standard teacher may learn behaviors that are **fundamentally
unimitable** by the student. For example, a teacher with perfect target knowledge
might learn aggressive pursuit maneuvers that require knowing the target's exact
future trajectory -- information the student with noisy/missing bbox can never
reconstruct.

**Solution**: Joint training with an imitability constraint:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}}(\pi_T) + \lambda \cdot D_{\text{KL}}(\pi_T(\cdot|s) \| \pi_S(\cdot|o))$$

The KL term penalizes the teacher for taking actions the student cannot predict.
This forces the teacher to learn behaviors that are:
1. High-reward (from the RL objective)
2. Recoverable from partial observations (from the imitability constraint)

**Why this matters for iris_ma6**: A standard teacher might learn to rely on continuous
detection, taking aggressive gimbal slews that only work with uninterrupted bbox.
The student-informed teacher would learn more conservative, recoverable behaviors
that degrade gracefully when bbox goes empty.

### 4.5 Implementation Strategies for iris_ma6

#### Strategy A: Sequential (simplest, ~2x training time)

```
Phase 1: Train teacher (200k steps)
  - Delay system disabled
  - No FP/FN, no dropout, no burst
  - Standard MAPPO-RNN
  - Save teacher checkpoint

Phase 2: Distill to student (200k steps)
  - Full noise pipeline from step 0 (no curriculum needed!)
  - Loss = (1-alpha) * L_PPO + alpha * L_DAgger
  - Alpha annealing: 1.0 -> 0.0 over 200k steps
  - Teacher queried at each step with GT state
```

**Advantage**: Student trains under deployment conditions from step 0. No curriculum
transitions. The teacher's action labels provide a stable learning signal regardless
of observation quality.

#### Strategy B: Progressive Distillation (3 stages)

```
Stage 1: Teacher with GT obs (100k steps)
Stage 2: Student-v1 with noise+delay, no FP/FN (100k steps, DAgger from teacher)
Stage 3: Student-v2 with full noise+FP/FN+dropout (100k steps, DAgger from teacher)
         + RL fine-tuning (100k steps)
```

Each stage starts from the previous student's weights, progressively adapting to
harder observation conditions.

#### Strategy C: Concurrent Student-Informed (most sophisticated)

```
For each training step:
  1. Teacher rollout with GT obs, student rollout with noisy obs (same env)
  2. Teacher loss = L_RL + lambda * D_KL(pi_T || pi_S)
  3. Student loss = L_DAgger(pi_S, pi_T)
  4. Both networks update
```

### 4.6 Practical Considerations

- **Teacher architecture**: Can be simpler (MLP, no GRU) since GT observations have
  no temporal ambiguity. But using the same architecture simplifies distillation.
- **Action space matching**: Teacher and student must have identical action spaces
  (7D: vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch, zoom_rate).
- **Stochastic vs deterministic teacher**: For DAgger, use the teacher's **mean**
  action (not sampled) to provide clean targets. For KL-based distillation, use the
  full distribution.
- **Deployment**: Only the student network is deployed. Teacher is training-only.
- **Checkpoint management**: Need to save/load teacher separately from student.

### 4.7 Limitations

- 2-3x total training time (teacher training + distillation)
- Teacher's performance ceiling limits the student
- DAgger requires querying the teacher at every student step (2x compute per step)
- If the teacher's policy class cannot solve the task, distillation fails
- Student-informed approach adds a sensitive hyperparameter ($\lambda$)

### 4.8 Key References

- Chen et al. (2020), "Learning by Cheating" -- CoRL (two-stage decomposition)
- Ross et al. (2011), "A Reduction of Imitation Learning to No-Regret Online Learning" -- AISTATS (DAgger)
- Rudin et al. (2022), "Learning to Walk in Minutes Using Massively Parallel Deep RL" -- CoRL
- Hoeller, Rudin et al. (2024), "ANYmal Parkour" -- Science Robotics (on-policy distillation)
- Messikommer et al. (2025), "Student-Informed Teacher Training" -- ICLR Spotlight
- Chane-Sane et al. (2024), "SoloParkour" -- CoRL

---

## 5. Approach 3: Observation Masking with Auxiliary Losses

### 5.1 Core Idea

Train with stochastic observation dropout from the start, and add auxiliary
losses that force the RNN hidden state to maintain useful predictions even
when observations are missing.

### 5.2 Masked Training

Instead of curriculum-ramping FP/FN, apply random observation masking from step 0:

$$o_{\text{masked}} = o \odot m, \quad m_i \sim \text{Bernoulli}(1 - p_{\text{drop}})$$

The mask $m$ is concatenated to the observation so the policy knows which channels
are valid. The iris_ma6 `bbox_empty` flag already serves this purpose for detection
channels.

**Key insight from Skand et al. (CoRL 2024)**: Simple random masking during training
produces policies that are robust to sensor failure at deployment, without
requiring complex teacher-student setups. The critical design choice is **masking
during training**, not just testing.

### 5.3 Auxiliary Reconstruction Loss

The main limitation of pure masking is that the policy has no explicit incentive
to maintain an internal state estimate through dropout periods. An auxiliary loss
addresses this:

$$\mathcal{L}_{\text{aux}} = \mathbb{E}\left[ \mathbb{1}[\text{bbox\_empty}] \cdot \| f_{\text{pred}}(h_t) - \text{bbox}_{\text{GT}} \|^2 \right]$$

where $h_t$ is the GRU hidden state and $f_{\text{pred}}$ is a small MLP head that
predicts the bbox from the hidden state. This loss is only active when the bbox is
empty, forcing the GRU to maintain a bbox prediction through dropout gaps.

**Architecture**:

```
Observation -> [MLP encoder] -> [GRU] -> h_t -> [Policy head] -> actions
                                   |
                                   +---> [Aux head (small MLP)] -> predicted_bbox
                                                |
                                                v
                                         L_aux = MSE(pred, GT) * bbox_empty
```

The auxiliary head is discarded at deployment. Its only purpose is to shape the
GRU hidden state during training.

### 5.4 Temporal Attention for Burst Robustness

For burst dropout (correlated multi-step dropout), a temporal attention mechanism
can learn to weight reliable observations more heavily:

```python
# Masked Sensory-Temporal Attention (MSTA)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, seq_len):
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_seq, mask_seq):
        # h_seq: (batch, seq_len, hidden)
        # mask_seq: (batch, seq_len, 1) - 0 for dropped, 1 for valid
        Q, K, V = self.query(h_seq), self.key(h_seq), self.value(h_seq)
        attn = (Q @ K.T) / sqrt(d)
        # Mask out dropped timesteps
        attn = attn.masked_fill(mask_seq.squeeze(-1) == 0, -inf)
        attn = softmax(attn, dim=-1)
        return attn @ V
```

This replaces the standard GRU's implicit temporal weighting with an explicit
attention mechanism that can skip over burst dropout gaps.

### 5.5 Implementation for iris_ma6

**Minimal change** (auxiliary loss only, ~30 lines):

1. Add a small MLP head (64 -> 32 -> 4) on the GRU hidden state output
2. During training, compute MSE between predicted bbox and GT bbox
3. Weight by `bbox_empty` flag (only active during misses)
4. Add to PPO loss: `L_total = L_PPO + 0.1 * L_aux`

**Moderate change** (masked training + auxiliary loss):

1. Apply random bbox masking from step 0 (p_mask = 0.1 initially)
2. Curriculum-ramp p_mask to match calibrated FP/FN rates
3. Add auxiliary bbox prediction loss as above
4. Add inter-agent data masking with same schedule

### 5.6 Limitations

- Auxiliary losses add hyperparameters (loss weight, head architecture)
- Reconstruction targets require GT data (available in sim only)
- Does not address the value function aliasing problem (combine with AAC for best results)
- Pure masking without auxiliary loss may lead to "give up" behavior during dropout

### 5.7 Key References

- Skand et al. (2024), "Simple Masked Training Strategies for Sensor-Failure Robustness" -- CoRL
- Li et al. (2024), "Masked Sensory-Temporal Attention for Sensor Generalization"
- Liu et al. (2024), "MMP: Towards Robust Multi-Modal Learning with Masked Modality Projection"

---

## 6. Approach 4: Information Bottleneck

### 6.1 Core Idea

Compress observations through a stochastic bottleneck that retains only task-relevant
information, filtering out noise. The Variational Information Bottleneck (VIB) adds
a KL penalty that pushes the encoder toward a compact, noise-invariant representation.

### 6.2 Mathematical Formulation

Standard policy: $\pi(a | o)$

VIB policy:
$$o \xrightarrow{\text{encoder}} q(z|o) \xrightarrow{\text{sample}} z \xrightarrow{\text{policy}} \pi(a|z)$$

Training objective:
$$\max_{\theta, \phi} \mathbb{E}_{z \sim q_\phi(z|o)}[R(\pi_\theta(z))] - \beta \cdot D_{\text{KL}}(q_\phi(z|o) \| p(z))$$

where $p(z) = \mathcal{N}(0, I)$ is a standard Gaussian prior. The $\beta$ parameter
controls the compression-performance tradeoff.

**Encoder**:
$$q_\phi(z|o) = \mathcal{N}(\mu_\phi(o), \sigma_\phi^2(o))$$

**Reparameterization**: $z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$

### 6.3 Why It Helps with Noisy Observations

1. **Noise filtering**: The KL penalty penalizes representations that encode observation
   noise. The encoder learns to extract signal (target bearing, team geometry) while
   ignoring noise (bbox jitter, FP artifacts).

2. **Graceful degradation**: When bbox is empty, the encoder's uncertainty ($\sigma$)
   increases naturally, signaling to the policy that the observation is unreliable.

3. **Burst robustness**: The bottleneck encourages temporal smoothing in the latent
   space because the KL penalty prevents abrupt representation changes.

### 6.4 Implementation for iris_ma6

```python
class VIBEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2 * latent_dim),  # mean + log_var
        )

    def forward(self, o):
        params = self.net(o)
        mu, log_var = params.chunk(2, dim=-1)
        std = (0.5 * log_var).exp()
        z = mu + std * torch.randn_like(std)
        kl = 0.5 * (mu**2 + std**2 - log_var - 1).sum(-1)
        return z, kl

# Modified policy network:
# obs -> VIBEncoder -> z -> GRU -> policy head
# L_total = L_PPO + beta * mean(kl)
```

### 6.5 Limitations

- $\beta$ is sensitive: too high loses task-relevant info, too low provides no filtering
- Adds training instability from stochastic encoder (on top of PPO stochasticity)
- May conflict with RNN hidden state (GRU already provides temporal smoothing)
- Best suited as a **complement** to AAC or distillation, not standalone

### 6.6 Key References

- Alemi et al. (2017), "Deep Variational Information Bottleneck" -- ICLR
- Lu et al. (2024), "Multimodal Information Bottleneck for Deep RL with Multiple Sensors"

---

## 7. Multi-Agent Considerations

### 7.1 Parameter Sharing in iris_ma6

All agents share a single policy network and single value network. This means:

- **One teacher, one student**: Teacher and student share weights across agents.
  Distillation is parameter-efficient.
- **Symmetric roles**: All agents have identical observation/action structure,
  so the policy learns role-agnostic behavior.
- **Shared critic = CTDE**: The value function concatenates all agents' states
  (Centralized Training, Decentralized Execution). With AAC, the centralized
  critic sees all agents' GT states.

### 7.2 Communication Dropout as a Multi-Agent Problem

Burst dropout in iris_ma6 is **directional**: channel B->A can fail while A->B remains
active. This creates asymmetric information states:

```
Agent A knows: own state + stale B state (burst) + own bbox
Agent B knows: own state + fresh A state + own bbox
```

The policy must handle this asymmetry despite parameter sharing. The `data_age` and
`bbox_age` observations encode the asymmetry, but the policy must learn to interpret
these signals correctly.

**With AAC**: The privileged critic sees both agents' true states and knows the
communication state (which channels are in burst). This allows correct value
estimation even under asymmetric information.

**With distillation**: The teacher operates without communication constraints, so it
doesn't experience asymmetry. The student must learn to handle it through DAgger
(visiting states with asymmetric information and receiving teacher's GT-informed
actions as labels).

### 7.3 Triangulation Under FP/FN

Multi-agent triangulation requires at least 2 agents with valid bbox detections.
When FP/FN is active, the probability that $\geq 2$ agents detect the target drops:

$$P(\text{triangulation valid}) = 1 - \sum_{k=0}^{1} \binom{n}{k} (1-p_{\text{miss}})^k p_{\text{miss}}^{n-k}$$

For 3 agents with 20% miss rate: $P \approx 0.896$ (10% chance of failed triangulation).
For 3 agents with 50% miss rate: $P \approx 0.500$ (half the time, no triangulation).

The teacher can learn optimal triangulation behavior (maximize observation geometry)
which the student inherits through distillation. The student then adapts this behavior
to handle intermittent triangulation failures.

### 7.4 Selective Communication Filtering

More advanced multi-agent approaches use **attention-based communication** that
learns to filter unreliable messages:

$$m_{j \to i}^{\text{filtered}} = \text{Attention}(h_i, h_j, o_j) \cdot g(\text{data\_age}_{ji})$$

where $g$ is a learned gating function that downweights stale data. This is more
expressive than the current approach of passing raw `data_age` as an observation
feature. However, it requires architectural changes beyond the current GRU-based
policy.

---

## 8. Proposed Implementation for iris_ma6

### 8.1 Recommended Approach: AAC + Curriculum Re-enable (Priority 1)

Based on the analysis, the **highest-impact, lowest-cost** approach is:

1. **Asymmetric Actor-Critic** for stable value estimation
2. **Re-enable FP/FN curriculum** now that the value baseline is stable

The auxiliary bbox prediction loss (Section 5.3) is deprioritized: the GRU already
has the representational capacity to maintain bbox predictions through dropout via
backpropagation-through-time. When bbox returns after a burst, the tracking reward
gradient propagates back through the GRU hidden states spanning the gap. The real
bottleneck is value function aliasing, not hidden state quality. The auxiliary loss
may provide marginal benefit but should be tested empirically only after AAC is
validated.

This avoids the complexity and doubled training time of full teacher-student
distillation while addressing the core gradient variance problem.

#### Implementation Plan

**Step 1: Add `_get_states()` to environment** (~80 lines)

Add a method returning the privileged state vector (43D for 3 agents). This is
returned via the `shared_observation_spaces` mechanism that SKRL already supports.

```python
# iris_ma_env6_test.py
def _get_states(self) -> Dict[str, torch.Tensor]:
    """Privileged state for asymmetric critic (training only)."""
    # Build GT state: ego(15) + target(6) + others(11*2) = 43D
    ...
```

**Step 2: Update config** (~5 lines)

```python
# iris_ma_env6_test_cfg.py
state_space: int = 43  # Privileged state (was -1 = concat obs)
```

**Step 3: Add auxiliary loss to MAPPO_RNN** (~40 lines)

```python
# mappo_rnn.py: MAPPORNNPolicy
class MAPPORNNPolicy(Model):
    def __init__(self, ..., aux_bbox_dim=4):
        ...
        # Auxiliary bbox prediction head
        self.aux_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 32),
            nn.ELU(),
            nn.Linear(32, aux_bbox_dim),
        )

    def compute_aux_loss(self, hidden_states, gt_bboxes, bbox_empty_mask):
        """Compute auxiliary bbox prediction loss (training only)."""
        pred_bbox = self.aux_head(hidden_states)
        loss = ((pred_bbox - gt_bboxes) ** 2 * bbox_empty_mask).mean()
        return loss
```

**Step 4: Re-enable FP/FN curriculum with faster ramp** (~5 lines)

```python
# curriculum_cfg.py
fp_fn_background_start_step: int = 15000    # Was 4000000 (disabled)
fp_fn_ramp_start_step: int = 100000         # Was 4000000
fp_fn_end_step: int = 120000                # Was 4000000
```

**Step 5: Enable burst dropout curriculum** (already configured, just verify)

Burst dropout at 200k-220k should work better with AAC because the value function
remains stable through the transition.

#### Expected Behavior

```
Step:    0k   20k   40k   60k   80k  100k  120k  140k  160k  180k  200k  220k
         |     |     |     |     |     |     |     |     |     |     |     |
Policy:  [learns tracking with stable GT-informed value baseline]
FP/FN bg:     [--- flat 0.1 ---]
FP/FN:                         [ramp 0.1->1.0]
Burst:                                                       [ramp]
         |                     |               |                   |
         v                     v               v                   v
  Value fn stable      FP/FN ramp starts    Full FP/FN        Burst onset
  throughout           (critic absorbs       reached           (critic stable,
  (sees GT state)       variance)            (actor adapted)    actor adapts)
```

### 8.2 Fallback: Full Teacher-Student Distillation (Priority 2)

If AAC alone is insufficient (actor cannot discover dropout-robust behaviors through
exploration alone), add explicit distillation:

**Phase 1: Train teacher** (200k steps, delay_system disabled)

```bash
# Train teacher with GT observations
python run_experiment.py task=Isaac-IrisMA6 \
    env.delay_system_enabled=false \
    env.state_space=-1 \
    agent.experiment.experiment_name=teacher_gt
```

**Phase 2: Distill student** (200k steps, full noise pipeline)

```python
# Modified training loop
for step in range(200_000):
    # Student rollout with noisy observations
    obs_noisy = env.get_observations()
    actions_student = student_policy(obs_noisy)

    # Teacher labels from GT state
    state_gt = env.get_states()  # Privileged
    actions_teacher = teacher_policy(state_gt).detach()

    # Combined loss
    distill_loss = MSE(actions_student, actions_teacher)
    rl_loss = PPO_loss(student_policy, obs_noisy, rewards)
    total_loss = (1 - alpha) * rl_loss + alpha * distill_loss

    # Anneal alpha: 1.0 -> 0.0
    alpha = max(0.0, 1.0 - step / 150_000)
```

### 8.3 Advanced Option: Student-Informed Teacher (Priority 3)

If the teacher learns behaviors the student cannot imitate (aggressive tracking that
requires continuous detection), add the imitability constraint:

```python
# Joint teacher-student training
teacher_rl_loss = compute_ppo_loss(teacher, state_gt, rewards)
student_imitation_loss = MSE(student(obs_noisy), teacher(state_gt).detach())

# Imitability: penalize teacher for unpredictable actions
with torch.no_grad():
    student_pred = student(obs_noisy)
imitability_loss = MSE(teacher(state_gt), student_pred.detach())

teacher_loss = teacher_rl_loss + lambda_imit * imitability_loss
student_loss = student_imitation_loss
```

### 8.4 Implementation Priority and Effort Estimates

| Priority | Approach | Effort | Files Changed | Training Cost |
|----------|----------|--------|---------------|---------------|
| **1** | AAC (privileged critic) | ~130 lines | env, cfg, run_experiment | 1x (same 400k steps) |
| **1b** | + Re-enable FP/FN curriculum | ~5 lines | curriculum_cfg.py | 1x |
| **1c** | + Auxiliary bbox loss (optional) | +~40 lines | mappo_rnn.py | 1x (test empirically after AAC) |
| **2** | Teacher-student distillation | ~200 lines | new training script | 2x (teacher + student) |
| **3** | Student-informed teacher | +~50 lines | training script | 2.5x (concurrent training) |
| **4** | VIB encoder | ~60 lines | mappo_rnn.py | 1.1x (small overhead) |

### 8.5 Experimental Validation Plan

**Experiment 1: AAC baseline**
- Train with AAC (privileged critic), FP/FN disabled
- Compare value loss and gradient variance against symmetric critic
- Expected: lower value loss, smoother training curves

**Experiment 2: AAC + FP/FN curriculum**
- Enable FP/FN background at 15k, full ramp at 100k-120k
- Monitor: triangulation quality, bbox tracking reward, value loss stability
- Expected: policy maintains tracking through FP/FN onset

**Experiment 3: AAC + burst dropout**
- Full curriculum including burst dropout at 200k-220k
- Monitor: inter-agent coordination during burst episodes
- Expected: policy develops hold-last-position behavior

**Experiment 4: AAC + auxiliary loss**
- Add auxiliary bbox prediction loss
- Monitor: GRU hidden state quality during dropout (via aux loss value)
- Expected: faster adaptation to FP/FN, better bbox prediction in dropout

**Experiment 5: Full distillation (if needed)**
- Train teacher, distill to student with DAgger
- Compare against AAC-only and curriculum-only baselines
- Expected: highest final performance, but 2x training time

---

## 9. References

### Core Teacher-Student / Privileged Learning

1. Pinto, L., et al. (2017). "Asymmetric Actor Critic for Image-Based Robot Learning." RSS.
2. Chen, D., et al. (2020). "Learning by Cheating." CoRL.
3. Ross, S., et al. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS.
4. Rudin, N., et al. (2022). "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning." CoRL.
5. Hoeller, D., Rudin, N., et al. (2024). "ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots." Science Robotics.
6. Messikommer, N., et al. (2025). "Student-Informed Teacher Training." ICLR (Spotlight).
7. Chane-Sane, E., et al. (2024). "SoloParkour: Constrained Visual Locomotion." CoRL.

### Theoretical Foundations

8. Belmega, E.V., et al. (2025). "A Theoretical Justification for Asymmetric Actor-Critic Algorithms." ICML.
9. Weihs, L., et al. (2025). "Informed Asymmetric Actor-Critic: Leveraging Privileged Information." arXiv.
10. OpenAI, et al. (2020). "Learning Dexterous In-Hand Manipulation." IJRR.

### Observation Robustness

11. Skand, V., et al. (2024). "Simple Masked Training Strategies Yield Control Policies Robust to Sensor Failure." CoRL.
12. Li, X., et al. (2024). "Masked Sensory-Temporal Attention for Sensor Generalization in Quadruped Locomotion." arXiv.
13. Liu, Y., et al. (2024). "MMP: Towards Robust Multi-Modal Learning with Masked Modality Projection." arXiv.

### Information-Theoretic

14. Alemi, A., et al. (2017). "Deep Variational Information Bottleneck." ICLR.
15. Lu, J., et al. (2024). "Multimodal Information Bottleneck for Deep Reinforcement Learning with Multiple Sensors." Neural Networks.

### Multi-Agent Communication

16. Yu, C., et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS.
17. Sun, Y., et al. (2025). "ROCO: Role-Oriented Communication for Efficient Multi-Agent RL." Expert Systems with Applications.
18. Guan, S., et al. (2024). "Context-Aware Communication for Multi-Agent RL." arXiv.

### Curriculum Learning

19. Bengio, Y., et al. (2009). "Curriculum Learning." ICML.
20. Narvekar, S., et al. (2020). "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey." JMLR.