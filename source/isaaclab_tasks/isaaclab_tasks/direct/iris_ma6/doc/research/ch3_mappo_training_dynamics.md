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

**Why value loss drops 100x**: Initially $V_\phi \approx 0$ but actual returns are ~1656. The MSE loss $(1656 - 0)^2$ is enormous. After a few updates, $V_\phi$ learns to predict ~3000, and the residual drops to ~10, giving loss $\sim 10^2 / 3000^2 \approx 0.00001$. (The value preprocessor normalizes, so the actual scale is different, but the ratio is similar.)

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

**Next**: [Chapter 4 -- The KL-Adaptive LR & Collapse Analysis](ch4_kl_adaptive_lr_collapse.md)
