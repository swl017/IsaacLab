# Chapter 3: MAPPO & Your Training Dynamics

> **Reading time**: ~50 minutes.
> **Prerequisites**: Chapters 1-2
> **Source code**: SKRL MAPPO at `_isaac_sim/.../skrl/multi_agents/torch/mappo/mappo.py`
> **Your analysis**: Re-read alongside `ppo_mappo_algorithm.md`

---

## 3.1 From PPO to MAPPO: What Changes?

MAPPO (Multi-Agent PPO) is not a fundamentally new algorithm. It's PPO applied to each agent independently, with one key architectural change: **the value function sees global information**.

### Your setup: 2 drones, 7 actions each

Each drone has:
- **Local observations** $o_t^i$: its own position, velocity, gimbal angles, target bearing from its own sensors
- **Global state** $s_t$: concatenation of all agents' observations (what both drones see)
- **Action** $a_t^i \in \mathbb{R}^7$: velocity commands (vx, vy, vz), yaw rate, gimbal yaw/pitch rates, zoom rate
- **Policy** $\pi_{\theta_i}(a^i | o^i)$: maps local observations to action distribution
- **Value function** $V_{\phi_i}(s)$: maps global state to expected return

### CTDE: Centralized Training, Decentralized Execution

This is the core idea of MAPPO, and it's worth understanding deeply.

**The problem with fully decentralized learning**: If drone_0's value function only sees $o^0$, it can't distinguish between "drone_1 is in a good position to help" and "drone_1 is far away and useless." The advantages $\hat{A}_t$ would be noisy because the value function is missing information about the other agent. Noisy advantages mean noisy policy gradients mean slow, unstable learning.

**The solution**: During training (in simulation), we have access to everything. Let the value function see the global state $s_t = [o^0_t, o^1_t]$. This gives it the best possible estimate of future returns, which gives the cleanest possible advantage signal.

$$V_{\phi_i}(s_t) = V_{\phi_i}(o^0_t, o^1_t)$$

**At deployment**: The value function is not needed. Only the policy runs, and it only needs local observations:

$$a^i_t \sim \pi_{\theta_i}(a^i | o^i_t)$$

**Optimal control analogy**: This is like using a full-state observer (Luenberger observer, Kalman filter) during controller design to get the best possible cost-to-go estimate, but deploying an output-feedback controller that only uses sensor measurements.

**In SKRL code** (`mappo.py` lines 344-348, 508, 538):
```python
# Value function receives shared_states (global)
values, _, _ = self.values[uid].act(
    {"states": self._shared_state_preprocessor[uid](shared_states)}, role="value"
)

# Policy receives local states only
_, next_log_prob, _ = policy.act(
    {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
)

# Value loss uses shared_states
predicted_values, _, _ = value.act({"states": sampled_shared_states}, role="value")
```

---

## 3.2 Independent Learning in MAPPO

Despite the name "multi-agent," MAPPO treats each agent's learning as independent. The outer loop in `_update` iterates over agents:

**In SKRL code** (`mappo.py` line 450):
```python
for uid in self.possible_agents:  # ["drone_0", "drone_1"]
    policy = self.policies[uid]
    value = self.values[uid]
    memory = self.memories[uid]
    # ... run the full PPO update for this agent ...
```

Each agent has its own:
- Policy network and value network
- Optimizer (with its own learning rate)
- Memory buffer (with its own rollouts)
- KL-adaptive scheduler

This is why your training logs show separate curves for drone_0 and drone_1. They can diverge -- one drone might learn faster than the other.

### Why not share parameters?

Some MAPPO implementations share policy parameters between agents (the policy takes an agent ID as additional input). Your setup uses **independent parameters**. Trade-offs:

| Approach | Pros | Cons |
|---|---|---|
| Shared parameters | Half the parameters; agents automatically symmetric; faster convergence for symmetric tasks | Can't specialize (e.g., leader/follower roles); gradient interference between agents |
| Independent parameters | Agents can specialize; no gradient interference | More parameters; may learn redundant representations |

For your drone tracking task, independent parameters make sense: the two drones need to coordinate positions (don't both track the same side), which benefits from role specialization.

---

## 3.3 RNN Processing in MAPPO

Your config uses GRU (Gated Recurrent Unit) with hidden size 64 and sequence length 32. This adds a temporal dimension to the policy.

### Why RNNs?

In a POMDP (partially observable MDP), the current observation $o_t$ doesn't contain enough information to determine the optimal action. The agent needs *memory* of past observations. An RNN hidden state $h_t$ serves as a compressed memory:

$$h_t = \text{GRU}(o_t, h_{t-1})$$
$$a_t \sim \pi_\theta(a | h_t)$$

In your drone task, partial observability arises from:
- Communication delays between agents
- Target motion prediction (need to infer velocity from position history)
- Gimbal state history (the current gimbal angle alone doesn't tell you how fast you were tracking)

### How RNN training works in PPO

Standard PPO trains on independent $(s, a, r)$ tuples. With RNNs, you need to train on *sequences* to propagate hidden states:

1. The rollout buffer stores 32 consecutive timesteps (your `sequence_length = 32`)
2. During training, mini-batches are sequences, not random samples
3. The GRU hidden state is propagated through each sequence during the forward pass
4. BPTT (Backpropagation Through Time) computes gradients through the sequence

This means each "sample" in the mini-batch is a 32-step sequence, not a single timestep. The computational cost is roughly 32x a non-RNN PPO update.

---

## 3.4 Your Training Run: Phase-by-Phase

Now let's walk through your `7e46515b39` training run with the understanding from Chapters 1-2. This is from your `ppo_mappo_algorithm.md`, but now you should understand *why* each number behaves the way it does.

### Phase 1: Rapid Value Learning (steps 4k-20k)

| Metric | Step 4k | Step 16k | What's happening |
|---|---|---|---|
| Value Loss | 0.0378 | 0.0003 | $V_\phi$ learning the return baseline (100x reduction!) |
| Policy Loss | -0.0003 | -0.0046 | Policy improving: $L^{\text{CLIP}} > 0$ |
| Std Dev | 0.559 | 0.335 | Policy narrowing: found good actions, exploiting them |
| LR | 0.00297 | 0.00087 | Scheduler cutting: small σ → large KL per update |
| Reward | 1656 | 3086 | Rapidly learning basic behaviors |

**Why value loss drops 100x**: Initially $V_\phi \approx 0$ but actual returns are ~1656. The MSE loss $(1656 - 0)^2$ is enormous. After a few updates, $V_\phi$ learns to predict ~3000, and the residual drops to ~10, giving loss $\sim 10^2 / 3000^2 \approx 0.00001$.

**Remember: the value loss in your Tensorboard is in normalized units, not raw reward units** (see Ch 2 §2.4 for details). A value loss of 0.003 means the critic's RMSE is $\sqrt{0.003} \approx 0.055$ *standard deviations* of the return distribution. In raw reward units, that's $0.055 \times \sigma_\text{return}$, which is ~80-100 for your return scale. This normalization is what keeps the value loss curve in a stable range (0.0001 - 0.01) even as raw rewards quadruple across training.

**Why σ drops**: The policy gradient pushes the mean $\mu$ toward good actions. Simultaneously, the gradient of $L^{\text{CLIP}}$ with respect to $\sigma$ is negative when the policy is clearly improving (tighter distribution = higher log-probability of the mean action = higher $r_t$ for those actions). The entropy bonus ($c_e = 0.01$) pushes back, but too weakly.

**Why LR drops**: From Chapter 1, $D_{\text{KL}} \propto (\Delta\mu)^2 / \sigma^2$. With σ=0.335 (small) and rapidly changing μ (strong advantages from imperfect $V_\phi$), the KL per update is large. The scheduler cuts LR to slow the policy down.

### Phase 2: Exploration Recovery (steps 20k-92k)

| Metric | Step 20k | Step 60k | Step 92k | What's happening |
|---|---|---|---|---|
| Std Dev | 0.337 | 0.760 | 1.214 | Entropy bonus slowly winning over exploitation |
| Policy Loss | -0.001 | +0.003 | +0.001 | Exploration dilutes good-action probability |
| LR | 0.001 | 0.001 | 0.001 | Stable: larger σ absorbs KL |
| Reward | 3120 | 2781 | 3420 | Dip then recovery: exploration cost then payoff |

**The σ rebound explained**: Once $V_\phi$ is accurate, advantages $\hat{A}_t$ become small (the value function's predictions are close to actual returns). The policy gradient weakens. Meanwhile, the entropy bonus gradient is *constant* ($-c_e = -0.01$ pushing $\ln\sigma$ up every step). When the policy gradient weakens enough, the entropy gradient dominates, and σ begins to rise.

This is not a failure -- it's the algorithm working as designed. The policy found a decent strategy (reward ~3100) but is now exploring to find better ones.

**The reward dip at step 60k**: Higher σ means more random actions. The policy mean might still point toward good actions, but the noise from exploration causes worse trajectories on average. Think of it as adding dithering to a controller -- performance degrades temporarily, but you're probing the landscape for better operating points.

**The recovery by step 92k**: The exploration discovered better strategies. Reward exceeds the Phase 1 plateau (3420 > 3086). The policy mean shifted to better actions that wouldn't have been found with the tight σ=0.335 distribution.

### Phase 3: Peak Performance (steps 92k-124k)

| Metric | Step 92k | Step 120k | What's happening |
|---|---|---|---|
| Std Dev | 1.214 | 1.211 | Equilibrium: entropy push ≈ policy gradient pull |
| LR | 0.001 | 0.002 | Rising: large σ → small KL → scheduler increases LR |
| Reward | 3420 | 4269 | Best performance: good μ + adequate exploration |

**Why σ stabilizes**: The entropy bonus pushes σ up. The policy gradient pushes σ down (tighter distribution = higher probability of known-good actions). These forces balance at σ ≈ 1.21. Additionally, `max_log_std = 0.7` provides a ceiling at $e^{0.7} \approx 2.01$.

**Why LR rises**: With σ ≈ 1.21, the same μ shift produces much less KL divergence: $D_{\text{KL}} \propto (\Delta\mu)^2 / (1.21)^2$ vs $(\Delta\mu)^2 / (0.335)^2$ in Phase 1. That's $13x$ less KL for the same parameter change. The scheduler sees consistently low KL and ramps LR up.

### Phase 4: Curriculum Shock (steps 124k-156k)

This is the critical transition. At step ~124k, the curriculum increases difficulty.

| Metric | Step 120k | Step 124k | Step 144k | What's happening |
|---|---|---|---|---|
| Reward | 4269 | 3320 | 3862 | Sharp drop: harder task |
| Value Loss | 0.001 | 0.006 | 0.005 | $V_\phi$ predictions are wrong for new difficulty |
| Policy Loss | +0.001 | -0.002 | +0.014 | Violently conflicting gradient signals |
| LR | 0.002 | 0.001 | 0.0002 | Crashing: KL spikes from distribution shift |

**Why the value function is suddenly wrong**: $V_\phi$ learned to predict returns under the old difficulty level. The curriculum increases difficulty, changing the reward distribution. $V_\phi$ now over-predicts returns (it thinks things will go better than they actually do). The large $(V_\phi(s) - V^{\text{targ}})^2$ error shows up as spiking value loss.

**Why policy loss spikes to +0.014**: The advantages are computed relative to the *wrong* value baseline. Some actions that were good under the old curriculum are bad under the new one, but the stale value function still assigns them positive advantages. The policy tries to increase these actions, which is counterproductive. The clipping mechanism prevents catastrophe, but the large magnitude of the loss (+0.014, highest in the run) shows severe signal noise.

**Why LR crashes**: The distribution shift causes large KL between consecutive updates (the policy is trying to adapt rapidly to the new curriculum). The KL-adaptive scheduler interprets this as "updates are too aggressive" and cuts LR. This is actually protective -- without it, the noisy gradients from stale advantages could destroy the policy.

### Phase 5: Frozen Policy (steps 156k-224k)

| Metric | Step 172k | Step 200k | What's happening |
|---|---|---|---|
| LR | 0.00005 | 0.000001 | Near zero: cascading KL cuts |
| Value Loss | 0.004 | 0.011 | Rising: $V_\phi$ can't track changing returns with frozen policy |
| Reward | 3935 | 2509 | Declining: can't adapt to continuing curriculum changes |

**The vicious cycle**: LR is so low that the policy can't change. But the environment (curriculum) keeps changing. The value function tries to track the changing returns, but the policy it's evaluating is static. The value function becomes inaccurate, which makes advantages noisy, which makes the small gradient updates even less useful.

**Why the scheduler doesn't recover**: The scheduler increases LR when KL drops below $\tau/2 = 0.01$. With LR ≈ 10⁻⁶, the policy can barely change at all, so KL ≈ 0. The scheduler does increase LR (note LR=0.000001 at step 200k recovers to 0.00006 by step 212k). But 1.5x increase from 10⁻⁶ is still tiny. It would take $\log_{1.5}(300) \approx 14$ consecutive low-KL updates to reach LR=3×10⁻⁴. But as soon as LR gets high enough for the policy to change, KL spikes again, and the scheduler cuts back down.

This is the **collapse cascade** that your `lr_collapse_analysis.md` diagnoses in detail. Chapter 4 dissects the math.

---

## 3.5 The Causal Feedback Diagram

All the quantities in your training logs form a coupled dynamical system. Here's the diagram from your analysis, annotated with the mechanisms from Chapters 1-2:

```
                ┌──────────────────────────────────────┐
                │                                      │
                ▼                                      │
        ┌──────────────┐                               │
 ┌─────►│  Policy σ    │◄──── Entropy gradient          │
 │      │              │      (-0.01 per step,          │
 │      │              │       constant upward push     │
 │      │              │       on ln σ)                 │
 │      └──────┬───────┘                               │
 │             │                                        │
 │             │ σ enters KL formula:                    │
 │             │ D_KL ∝ (Δμ)² / σ²                     │
 │             │ Large σ → small KL → LR rises          │
 │             ▼                                        │
 │      ┌──────────────┐     ┌───────────────┐         │
 │      │KL Divergence │────►│  Learning Rate │─────────┘
 │      │              │     │               │    LR scales the
 │      │ KL > 2τ: cut │     │ Δθ = -α·∇L   │    gradient step,
 │      │ KL < τ/2: up │     │               │    which changes
 │      └──────────────┘     └───────┬───────┘    σ via entropy
 │                                   │            and policy loss
 │             Larger α → bigger Δμ  │
 │             → bigger advantages   │
 │                                   ▼
 │      ┌──────────────┐     ┌───────────────┐
 │      │ Value Loss   │◄───►│  Policy Loss  │
 │      │              │     │               │
 │      │ V_ϕ accuracy │     │ -L^CLIP       │
 │      │ determines   │     │ Clip prevents │
 │      │ advantage    │     │ ratio > 1.2   │
 │      │ quality      │     │ or < 0.8      │
 │      └──────┬───────┘     └───────┬───────┘
 │             │                     │
 │             │  Better V_ϕ →       │ Better policy →
 │             │  cleaner Â_t →      │ higher rewards →
 │             │  better policy      │ V_ϕ must re-learn
 │             │  updates            │
 │             ▼                     ▼
 │      ┌──────────────┐     ┌───────────────┐
 └──────│  Advantages  │     │ Total Reward   │
        │  Â_t = GAE   │     │               │
        └──────────────┘     └───────────────┘
```

### Key feedback loops:

1. **Stabilizing loop (healthy)**: Large σ → small KL → LR rises → stronger gradient → policy improves → repeat. This is the loop operating during Phase 2-3.

2. **Destabilizing loop (collapse)**: Distribution shift → large KL → LR cut → can't adapt → more KL from next shift → more LR cuts. This is the loop operating during Phase 4-5.

3. **Exploration loop**: Small advantages (good $V_\phi$) → entropy gradient dominates → σ rises → more exploration → discovers better strategies → reward increases.

4. **Exploitation loop**: Large advantages (early training) → policy gradient dominates → σ drops → concentrates on known-good actions → fast improvement but limited exploration.

The training dynamics are the result of these loops competing with each other. The "art" of hyperparameter tuning is setting the coefficients ($c_e$, $\epsilon$, `kl_threshold`, `grad_norm_clip`) so that the stabilizing loops dominate.

---

## 3.6 Quiz

**Q1**: Your MAPPO has separate memories for drone_0 and drone_1. Each stores its own rollouts. But both drones operate in the same environment. Why is it important that each agent's memory contains transitions from its own perspective?

<details>
<summary>Answer</summary>

Each agent's policy update needs $(o^i_t, a^i_t, r^i_t, \log\pi_{\theta_i}(a^i_t|o^i_t))$ tuples. The observations are agent-specific (each drone sees from its own sensors). The actions are agent-specific (each drone has its own gimbal, throttle, etc.). The log-probabilities are agent-specific (computed from that agent's policy).

If you mixed data from both agents into one buffer, the log-probabilities wouldn't match -- the importance sampling ratio $\pi_\theta(a|s) / \pi_{\theta_\text{old}}(a|s)$ would be meaningless because the actions came from a different agent's policy.

The rewards *could* be shared (cooperative task), but the observations and log-probs must be agent-specific.
</details>

**Q2**: During Phase 2 (exploration recovery), the policy loss is positive (+0.003 at step 60k). Your policy is getting *worse* at individual updates, yet reward eventually recovers to 3420. How is this possible?

<details>
<summary>Answer</summary>

The policy loss being positive means $L^{\text{CLIP}} < 0$ on average. This happens because increasing σ dilutes the probability of *all* specific actions, including the good ones. So the ratio $r_t$ for previously good actions decreases below 1.0 (the policy is less likely to take them now), making $r_t \cdot \hat{A}_t$ smaller than $\hat{A}_t$ even when $\hat{A}_t > 0$.

But this doesn't mean the policy is getting worse. The policy *mean* $\mu$ might still be improving (pointing toward better actions). The wider σ means the policy is *exploring around* that mean, trying nearby actions. Some will be better, some worse.

Reward recovers because exploration discovers states and actions that the tight σ=0.335 policy never tried. The mean $\mu$ shifts to these better strategies over time. The positive policy loss is the *per-update cost* of exploration; the reward recovery is the *cumulative benefit*.

This is analogous to adding process noise in an EKF: it makes individual estimates worse, but prevents the filter from becoming overconfident and missing the true state.
</details>

**Q3**: At step 124k (curriculum shift), value loss spikes from 0.001 to 0.006 and LR crashes from 0.002 to 0.001. Which caused which? What is the causal chain?

<details>
<summary>Answer</summary>

The causal chain:

1. **Curriculum shift** changes the reward distribution (harder task → different returns)
2. **Value function** is still calibrated for the old distribution → **value loss spikes** (it under/over-predicts)
3. **Inaccurate value function** → **noisy advantages** $\hat{A}_t$ (because $\hat{A}_t = G_t - V_\phi(s_t)$ and $V_\phi$ is wrong)
4. **Noisy, large advantages** → **large policy gradient** → **large Δμ per update**
5. **Large Δμ** → **large KL** between old and new policy
6. **Large KL** → **scheduler cuts LR**

So value loss spiking and LR crashing are both *effects* of the curriculum shift, but the value loss spike *precedes* and *causes* the LR crash through the advantage → gradient → KL → scheduler chain.

This is why your analysis doc correctly concludes "LR is the leading indicator" -- it crashes first, before downstream metrics (pair_valid_rate, tracking_lost) degrade.
</details>

---

## 3.7 Appendix: Curriculum Design Principles

The Phase 4 curriculum shock in §3.4 is a concrete example of a more general principle: **curriculum design is an exercise in aligning the direction and speed of environment changes with the policy's current learning capacity.** This appendix formalizes that principle into a design framework.

### A.0 The Bootstrap Principle (precondition)

**Before any phase-matching can apply, the policy must learn at least one stable signal.** A randomly-initialized policy has no learned response to anything. It needs at least one perfectly reliable signal in the observation→reward chain to learn cause and effect. Curriculum stages that corrupt this bootstrap signal must wait until the bootstrap skill is established (typically 12-20k steps).

**Why this matters**: The phase-matching framework (§A.2-A.3) assumes the policy is *already learning* and asks "what kind of difficulty change is compatible with the current learning state?" But if the policy hasn't bootstrapped yet, there is no "learning state" to match against — the framework simply doesn't apply.

**Empirical evidence**: The 5ce56 and 91f5da experiments ramped observation corruption (noise, dropout, FN) from step 0 with the intent of preventing σ collapse. Both failed catastrophically. The policy never reached pair_valid > 0.62 (compared to 0.95+ in successful runs). The diagnosis: with bbox occasionally zeroed by FN from the first step, a randomly-initialized policy has no reliable mapping from "what I see" to "what I should do" — there's nothing to bootstrap from.

**The principle**:

> **At step 0, at least one signal in the observation→reward chain must be perfectly reliable so a randomly-initialized policy can learn cause and effect. Stages that corrupt this signal (observation noise, dropout, FN, FP, sensor failures) must wait until base bootstrap is complete.**

**Practical rule**: During the bootstrap window (0-15k for iris_ma6), the observation→reward signal must be **identical to the deployment-target distribution minus all sensor corruption**. Physics randomization is OK (it perturbs dynamics, not observations). Reward shaping is OK (it perturbs the reward target, not the observation). What is *not* OK is anything that breaks the observation→action mapping the policy is trying to learn from random initialization.

**Diagnostic**: If `pair_valid_rate` (or your task-equivalent core metric) is not above 0.85 by step 15-20k, the bootstrap has failed. No amount of subsequent curriculum tuning will recover it — the policy will just drift downward from a bad initial fit.

### A.1 Classifying Curriculum Stages by Their Effect on the Optimal Policy

A curriculum stage perturbs the environment, which shifts the optimal policy $\pi^*$. The nature of the shift falls into one of four categories:

#### Shrinking curricula (narrow the acceptable action range)

The new environment removes "forbidden zones" from action space. The optimal policy becomes **more concentrated**. σ should *decrease* to match.

Examples from iris_ma6:
- Safety constraints (CBF penalty, altitude penalty, target proximity)
- Action magnitude penalties (action_sum, action_delta)
- Collision penalties

**Effect on σ**: downward pressure. The policy loss gradient pushes log_std down because narrow, specific actions have high advantage.

#### Expanding curricula (require access to a wider action range)

The new environment demands the policy *use* more of action space to cope. σ should *increase*, or at least not decrease.

**Important sub-distinction**: expanding stages split into two categories that look similar but behave very differently in early training:

**Physics-side expand** (perturbs dynamics, observation→action mapping preserved):
- Domain randomization (mass, inertia, friction, controller gains)
- Faster agent velocities (need larger control efforts)
- Faster target dynamics
- Wind/disturbance forces

These are **safe in Phase 1 and during bootstrap** (§A.0) because the observation→reward chain stays intact. A randomized mass means "pitch forward" still moves the agent forward; only the magnitude varies. The policy can still learn cause and effect from random initialization. Physics-side perturbations produce *graceful degradation*: the action that was 80% correct under nominal dynamics is still 70% correct under randomized dynamics.

**Observation-side expand** (corrupts the observation channel):
- Sensor noise on positions/velocities/bboxes
- Detection dropout (random frame loss)
- False negatives (bbox zeroed)
- False positives (spurious detections)
- Communication delays
- Burst dropout (correlated blackouts)

These are **fatal in Phase 1 and must wait until bootstrap is complete** (§A.0). They don't perturb the action→state mapping — they break the observation channel itself. When a bbox is zeroed by a false negative, there's no "graceful degradation" — the entire downstream computation breaks. A randomly-initialized policy has no learned response to "no detection," so it cannot bootstrap from this.

**The asymmetry**: Physics DR adds *bounded noise* to a *learned function*. Observation corruption *deletes information* from the *input space*. The first is recoverable by averaging; the second creates input states the policy has never seen and has no response for.

**Effect on σ**: Both put upward pressure on σ in steady state. Physics-side does so by demanding broader action exploration to be robust to dynamics variation. Observation-side does so by demanding broader exploration to handle corrupted inputs. **But observation-side cannot apply this pressure during bootstrap** because the policy never establishes a baseline strategy to perturb.

#### Shifting curricula (move the optimum laterally without changing its width)

The new environment keeps the spread roughly constant but moves *where* the optimal actions lie.

Examples from iris_ma6:
- Triangulation reward introduction (different viewpoints needed from different agents)
- Coordination penalties (role differentiation between drones)
- Moving target acquisition (target bearing shifts over time)

**Effect on σ**: mostly neutral, but the policy mean $\mu$ must move through policy space. Requires sufficient exploration budget to escape the old local optimum.

#### Complex curricula (combinations of the above)

Real curriculum stages often combine effects. For example, "faster target dynamics" is both expand (need wider action range) and shift (optimal actions are different). Classify by the dominant effect.

### A.2 The Training Phase Framework

From §3.4, training has natural phases:
- **Phase 1** (0-20k): σ narrows as policy exploits obvious advantages
- **Phase 2** (20k-92k): σ rebounds as entropy bonus dominates over weakened advantages
- **Phase 3** (92k+): σ stabilizes at equilibrium; policy exploits refined strategies

These phases have **different exploration budgets** (measured by the size of σ and the ability of the policy to visit new states):

| Phase | σ range | Exploration budget | Best-suited curriculum type |
|---|---|---|---|
| Early Phase 1 (0-10k) | 0.5 → 0.4 | Low (narrowing fast) | Shrink (safety) |
| Late Phase 1 (10k-20k) | 0.35 | Minimum | None (let policy stabilize) |
| Early Phase 2 (20k-40k) | 0.35 → 0.5 | Growing | Expand (dynamics, velocity) |
| Mid Phase 2 (40k-80k) | 0.5 → 1.0 | Moderate | Shift (coordination, new rewards) |
| Late Phase 2 (80k-92k) | 1.0 → 1.2 | High | Shift, complex |
| Phase 3 (92k+) | 1.2 (stable) | High but committed | Refinement, not new stages |

**Important caveat: σ ceiling is action-space-scale-dependent.**

The σ values in the table above (0.35 → 1.21) are from the 928b training run, which used a narrow action space (45 deg/s yaw, 180 deg/s gimbal, 1× zoom rate). The Phase 2 σ ceiling of ~1.2 is **not a universal constant** — it depends on the action scale.

In a wider action space (e.g., 90 deg/s yaw, 360 deg/s gimbal, 4× zoom rate, as used in the 3461+ runs), the same sigma in normalized space corresponds to 2-4× larger physical effects. The policy discovers early that *small normalized actions suffice*, and the exploitation gradient is correspondingly stronger. Phase 2 σ caps at ~0.5-0.6 instead of 1.2.

**Why this matters for curriculum design**: the σ ceiling determines the exploration budget available for late-phase shifts. With σ ≈ 0.5 in Phase 2, the policy cannot execute a shift that requires escaping a deep local optimum (e.g., observation corruption that invalidates the learned strategy). The framework's recommendation to introduce shifts in mid-Phase 2 assumes σ ≈ 1.0+ at that point. **In wide action spaces, you must either:**
1. Move shifts to later (when σ has had more time to grow) — but σ may never reach 1.0+
2. Reduce shift magnitude (so it fits within the available exploration budget)
3. Increase entropy coefficient $c_e$ to push σ harder (the only direct lever)
4. Compute σ in **physical units** rather than normalized units when reasoning about exploration capacity:

$$\sigma_{\text{physical}} = \sigma_{\text{normalized}} \cdot s_i$$

For yaw rate at the 928b ceiling: $1.21 \cdot 0.785 = 0.95$ rad/s. For yaw rate at the wide-action 3461 ceiling: $0.5 \cdot 1.571 = 0.79$ rad/s. These are physically comparable — the wide-action policy is *not* under-exploring in absolute terms. It just looks lower in normalized units.

**The practical implication**: when porting a curriculum from one action space to another, **think in physical units**. A curriculum stage that requires "σ > 1.0 in Phase 2" in a narrow action space may require "σ > 0.4 in Phase 2" in a wide action space — the same physical exploration budget.

### A.3 The Design Principle: Match Stage Type to Training Phase

**Shrink stages** should be introduced when the policy has low exploration needs:
- Early Phase 1: the policy is already collapsing onto a narrow optimum; shrinking is compatible
- Late Phase 2: the policy has found good strategies and can afford to tighten
- **Not in mid-Phase 2**: would fight the entropy-driven σ growth

**Expand stages** should be introduced when the policy has room to grow:
- Early Phase 2: σ is rebounding; expanding is aligned with the natural dynamic
- Can be combined with shrink stages in Phase 2 to balance the forces (your 20k-40k window does this intentionally -- velocity expand + safety shrink)

**Shift stages** are the riskiest and require maximum exploration budget:
- Mid-to-late Phase 2: σ is near its peak, providing maximum exploration capacity to escape the old optimum
- **Never in Phase 1**: the policy has collapsed and cannot escape the old local optimum
- **Never in Phase 3**: the policy has committed to an equilibrium and won't move without a huge perturbation

**Your iris_ma6 curriculum follows this framework well** (even if accidentally). The coordination curriculum activates at 60k -- mid Phase 2 -- exactly when the policy has the most exploration budget to discover new multi-agent strategies. If it had been introduced at 10k (deep Phase 1), the two drones would have converged to symmetric strategies and never escaped.

### A.4 Stacking Multiple Stages: Compatibility Rules

When multiple curriculum stages overlap, their effects on σ and the policy optimum combine. Design rules:

1. **No more than 2-3 stages active simultaneously**. Each additional stage adds another dimension of perturbation to the value function, increasing the risk of divergence.

2. **Shrink and expand can be combined in Phase 2**. They apply opposing pressures on σ, which can produce a stable equilibrium. Your 20k-40k window combines velocity expand + safety shrink + tracking-randomization expand -- three stages, two expanding and one shrinking. The net effect is a slight σ decrease (shrink slightly dominates) or stays flat, which is compatible with the natural Phase 1→2 transition.

3. **Two shifts should not overlap**. Each shift tries to move the policy to a new optimum. Two simultaneous shifts create conflicting gradients and can leave the policy stuck between them. If you need multiple shifts, stagger them (shift A during 40k-60k, shift B during 60k-80k).

4. **Shifts should not overlap with shrinks**. The shift needs exploration to escape the old optimum; the shrink reduces exploration. They fight each other.

5. **Shifts can overlap with expands**. The expand increases exploration capacity, which the shift consumes. They're compatible.

### A.5 Ramp Speed: The Value Function Tracking Heuristic

From §3.4 Phase 4, the cascade begins when the value function cannot track the changing reward distribution fast enough. The ramp speed determines how much $V_\phi$ falls behind per update.

**Heuristic**: the ramp duration should be at least **2x the value function settling time**.

The settling time is the number of gradient steps needed for $V_\phi$ to adapt to a step change in the reward distribution. You can measure it empirically:
1. Take a checkpoint at step $k$
2. Apply an instantaneous reward change (e.g., flip on a new reward term at full weight)
3. Train for $N$ steps and measure when value loss returns to baseline
4. That's the settling time

For iris_ma6, empirically this is ~5k-10k steps per update. So a curriculum ramp should span at least 10k-20k steps to let the value function follow smoothly.

**Your curriculum phases are all 20k-40k long**, which satisfies the 2x-settling-time heuristic. Shorter ramps risk triggering the Phase 4 cascade; longer ramps are safer but waste training budget.

#### Ramp speed is a Goldilocks problem (not "longer is better")

The 2x-settling-time heuristic gives a *lower* bound on ramp duration. It does **not** mean "longer is always better." Excessively long ramps have a hidden failure mode:

**The slow-ramp failure mode**: If a ramp is much longer than the value function settling time, the policy and value function adapt incrementally to each tiny difficulty increment. The gradient signal at any moment is small (because the next-step difficulty is barely different from the current). The policy *drifts along the ramp* without ever experiencing a strong enough gradient to commit to a robust strategy for the new regime.

When the ramp completes, the policy has technically been trained on the full difficulty, but only at a single moment in time. It has not experienced sustained training under the full difficulty. If subsequent perturbations cause it to forget the late-ramp adaptation, there's no reservoir of training data to recover from.

**Empirical evidence**: The 36700 experiment used a 100k FP/FN ramp (5x slower than the original 20k). The policy slowly drifted along the ramp, but when noise hit at 60k, the policy collapsed because it had only experienced low-rate FP/FN — never strong enough to develop robust bbox-empty handling. The long ramp gave V_φ time to settle, but also gave the policy time to forget bbox-empty handling between low-rate exposures.

**The Goldilocks range**: Empirically, ramp duration should sit in $[2T_s, 5T_s]$ where $T_s$ is the value function settling time. Below $2T_s$, the value function lags and KL spikes (Phase 4 cascade). Above $5T_s$, the gradient signal becomes too weak per step and the policy drifts without committing.

For iris_ma6 with $T_s \approx 5$-10k, this gives a ramp duration window of **10k–50k**. The 928b/36700 schedules at 20k-40k sit in this sweet spot. The 91f5da schedules at 40k-100k for several stages are above the Goldilocks range and contribute to the bootstrap failure (the gradient signal during 0-30k is fragmented across seven slow ramps, none of them strong enough to drive learning).

**Diagnostic for over-slow ramps**: If a curriculum stage is in its ramp window but the corresponding metric (e.g., value loss, target reward term) is *not* changing, the ramp is too slow. The stage should produce a measurable signal — if it doesn't, the policy isn't learning from it.

### A.6 Monitoring Signals for Curriculum Health

When you add or modify a curriculum stage, check these signals in Tensorboard:

| Signal | Healthy | Alarm |
|---|---|---|
| **Value loss** | Small bump (1.5-3x baseline), recovers within 5k steps | Large spike (>5x), persists beyond 10k steps |
| **LR** | Small dip (10-30%), recovers within 5k steps | Crash to floor, stays there |
| **KL divergence** | Stays in dead zone [0.01, 0.04] | Sustained > 0.04 for multiple epochs |
| **Reward of the affected term** | Smooth trajectory, no discontinuity | Jump or cliff at the boundary |
| **Policy std σ** | Continues its natural phase trajectory | Sharp reversal (e.g., was rising, suddenly drops) |

If any signal is in the alarm column, the ramp is too fast, the stage is wrong for the current phase, or multiple stages are conflicting. Extend the ramp, delay the stage to a more compatible phase, or remove a conflicting concurrent stage.

### A.7 The Checklist for Adding a New Curriculum Stage

Before implementing a new curriculum stage, answer these questions:

1. **Bootstrap safety**: Does this stage corrupt the observation→reward chain? If yes, it must start *after* bootstrap is complete (≥15-20k). Verify the policy reaches `pair_valid_rate > 0.85` (or task-equivalent core metric) before this stage activates (per §A.0).
2. **Classification**: Is this stage shrink, expand, shift, or complex? If expand, is it physics-side or observation-side (per §A.1)?
3. **Phase**: What training phase will it activate in? What is the expected σ at that time? Compute σ in **physical units** (per §A.2 caveat) — not just normalized.
4. **Compatibility**: Does the stage type match the training phase (per §A.3)?
5. **Stacking**: What other stages are active in the same window? Are their effects compatible (per §A.4)? Is the total active stage count ≤ 3?
6. **Ramp speed**: Is the ramp duration in the Goldilocks range $[2T_s, 5T_s]$ (per §A.5)? Not just longer than $2T_s$ — also not longer than $5T_s$.
7. **Value loss prediction**: What's the expected value loss bump at the ramp onset? Is it within the healthy range (per §A.6)?
8. **Escape hatch**: If the stage fails, what's the rollback? Can you disable it mid-run and resume training?

Treat these questions as a **pre-flight check**. Most curriculum problems are traceable to one of them being skipped — especially questions 1 and 5, which are the most commonly violated.

### A.8 Worked Example: Adding a New Sensor Dropout Stage

Suppose you want to add a new curriculum stage: "sensor dropout" -- randomly drop 20% of observation entries to simulate sensor failures.

1. **Classification**: Expand (the policy needs to be more robust, wider action distribution to handle uncertainty). Also slight shift (optimal actions under noisy observations differ from clean observations).
2. **Phase**: Where to insert? This is a robustness curriculum; sensor dropout is most compatible with Phase 2 (moderate σ) or early Phase 3.
3. **Compatibility**: Expand + shift is best in mid-to-late Phase 2. Let's target 100k-120k.
4. **Stacking**: Check what else is active at 100k. Your existing curriculum has noise (100k-120k) already in that window. Adding another noise-type stage doubles the challenge. Consider staggering to 120k-140k instead, or reducing the dropout maximum to 10%.
5. **Ramp speed**: Settling time ~5k steps. Minimum ramp 10k steps. Use 20k ramp (120k-140k).
6. **Value loss prediction**: 20% dropout is a significant perturbation. Expected bump ~3-4x baseline. Borderline healthy.
7. **Escape hatch**: Make the dropout rate a config parameter so it can be set to 0 on the fly.

**Decision**: Schedule the new dropout stage at 120k-140k, with max dropout 15% (conservative), ramped linearly over 20k steps. Monitor value loss, KL, and pair_valid_rate for the first 10k steps after activation.

---

**Key takeaway**: Curriculum design is not a post-hoc addition to an RL setup -- it's an integral part of the optimization problem. A well-designed curriculum is a smooth trajectory through policy space; a poorly-designed curriculum is a series of discontinuities the policy may fail to bridge. Before implementing any new stage, classify its effect on the optimal policy and match it to a compatible training phase.

---

**Next**: [Chapter 4 -- The KL-Adaptive LR & Collapse Analysis](ch4_kl_adaptive_lr_collapse.md)
