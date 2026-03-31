# PPO & MAPPO: Rigorous Mathematical Treatment with Training Log Analysis

> **Training run**: `2026-03-30_22-25-56_mappo_rnn_torch_7e46515b39_compressed_curriculum_target_proximity_20k`
> **Hyperparameters**: γ=0.99, λ=0.95, ε_clip=0.2, ε_value=0.2, c_e=0.01, c_v=1.0, KL threshold=0.02, grad_norm_clip=0.3, LR_base=0.0003, mini_batches=8, learning_epochs=6, rollouts=32, sequence_length=32

---

## 1. Proximal Policy Optimization (PPO)

### 1.1 Setup

We have a Markov Decision Process (S, A, P, R, γ) where an agent with policy π_θ(a|s) collects trajectories τ = (s₀, a₀, r₀, s₁, …). The goal is to maximize the expected discounted return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

### 1.2 Policy Gradient Foundation

The policy gradient theorem gives:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]$$

where $\hat{A}_t$ is the **advantage estimate** — how much better action $a_t$ was than the average action under the current value function.

### 1.3 Generalized Advantage Estimation (GAE)

This run uses γ = 0.99, λ = 0.95. The TD residual is:

$$\delta_t = r_t + \gamma \, V_\phi(s_{t+1}) - V_\phi(s_t)$$

GAE accumulates these with exponential decay:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \, \delta_{t+l} = \delta_t + (\gamma\lambda)\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \cdots$$

- λ = 0 gives the 1-step TD advantage (low variance, high bias).
- λ = 1 gives Monte Carlo advantage (high variance, low bias).
- λ = 0.95 is a high-variance compromise — appropriate for multi-agent settings where the environment is noisy.

### 1.4 The PPO Clipped Surrogate Objective

PPO avoids destructive large policy updates by clipping the importance-sampling ratio. Let θ_old be the parameters at collection time. The ratio is:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

The **clipped surrogate objective** (ε = 0.2 in this run):

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;\; \text{clip}\big(r_t(\theta),\; 1{-}\epsilon,\; 1{+}\epsilon\big)\,\hat{A}_t\Big)\right]$$

This means:
- If $\hat{A}_t > 0$ (good action): r_t is capped at 1 + ε = 1.2. The policy can increase the probability of this action, but only by 20%.
- If $\hat{A}_t < 0$ (bad action): r_t is floored at 1 − ε = 0.8. The policy can decrease the probability, but only by 20%.

### 1.5 Value Function Loss

The value network V_φ(s) is trained via clipped MSE (ε_v = 0.2):

$$L^{V}(\phi) = \mathbb{E}_t\!\left[\max\!\Big(\big(V_\phi(s_t) - V_t^{\text{targ}}\big)^2,\;\; \big(\text{clip}(V_\phi(s_t),\, V_{\phi_{\text{old}}}(s_t) \pm \epsilon_v) - V_t^{\text{targ}}\big)^2\Big)\right]$$

where $V_t^{\text{targ}} = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$ is the bootstrapped target.

### 1.6 Entropy Bonus

For a Gaussian policy $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s),\, \sigma_\theta^2)$ with d-dimensional action space:

$$H[\pi_\theta(\cdot|s)] = \frac{d}{2}\ln(2\pi e) + \sum_{i=1}^{d} \ln \sigma_i$$

The entropy bonus $L^H$ encourages exploration by preventing σ → 0.

### 1.7 Combined Loss

The total loss minimized per update epoch (c_e = 0.01, c_v = 1.0):

$$L(\theta, \phi) \;=\; -L^{\text{CLIP}}(\theta) \;+\; c_v \cdot L^{V}(\phi) \;-\; c_e \cdot H[\pi_\theta]$$

$$\boxed{L \;=\; -L^{\text{CLIP}} \;+\; 1.0 \cdot L^{V} \;-\; 0.01 \cdot H}$$

Signs: we **maximize** the clipped objective and entropy (hence negative signs in the minimization), and **minimize** the value loss.

---

## 2. MAPPO (Multi-Agent PPO)

### 2.1 Extension to Multi-Agent

This setup has 2 agents (drone_0, drone_1). MAPPO uses the **centralized training, decentralized execution (CTDE)** paradigm:

- **Decentralized policy**: Each agent i has $\pi_{\theta_i}(a_t^i \mid o_t^i, h_t^i)$ conditioned on its **local observation** $o_t^i$ and RNN hidden state $h_t^i$.
- **Centralized value function**: $V_{\phi_i}(s_t)$ takes the **global state** s_t (shared observations from all agents).

Each agent's PPO update is **independent** — agent i computes its own advantages, its own clipped ratio, its own entropy. The key insight is that the value function uses richer global information to reduce variance in advantage estimation, while the policy remains decentralized for execution.

### 2.2 Parameter Sharing

Both drones share the same network architecture (mappo_rnn, hidden=64, GRU hidden=64). The SKRL MAPPO implementation trains separate parameter sets per agent, which is why the logs show separate loss/LR/std curves for drone_0 and drone_1.

### 2.3 RNN Sequence Processing

The config uses sequence_length = 32 with GRU. During updates:
1. Rollout buffer of 32 timesteps is split into mini_batches = 8 chunks.
2. Each mini-batch processes sequences, propagating GRU hidden states.
3. learning_epochs = 6 passes over the full rollout buffer per update.

---

## 3. Training Log Analysis — How the Quantities Interact

Using run `7e46515b39_compressed_curriculum_target_proximity_20k`:

### 3.1 Phase 1: Early Training (steps 4k–20k) — Rapid Value Learning

| Step | LR | Policy Loss | Value Loss | Entropy Loss | Std Dev | Mean Reward |
|------:|------:|------:|------:|------:|------:|------:|
| 4k | 0.00297 | −0.000328 | 0.03775 | −0.00941 | 0.559 | 1656 |
| 8k | 0.00188 | −0.002437 | 0.000278 | −0.00446 | 0.452 | 2717 |
| 16k | 0.00087 | −0.004554 | 0.000298 | +0.000916 | 0.335 | 3086 |

**What's happening mathematically:**

**Value loss** drops 100× from 0.0377 to 0.0003. This is the value network V_φ rapidly learning the baseline return. Since $\hat{A}_t = R_t - V_\phi(s_t)$, initially V_φ ≈ 0 while R_t ≈ 1656, giving massive TD residuals. As V_φ improves, L^V plummets.

**Policy loss is negative** (−0.0005 → −0.0045). Recall L^CLIP is maximized, so the reported loss −L^CLIP being negative means the clipped objective is positive — the policy is successfully increasing probability of advantageous actions. The negative sign means:

$$r_t(\theta)\,\hat{A}_t > 0 \implies \text{policy is moving in the direction of positive advantages}$$

**Std dev drops** from 0.559 → 0.335. The policy is collapsing toward exploitation too fast — σ shrinks, meaning ln σ decreases, entropy $H = \frac{d}{2}\ln(2\pi e) + \sum \ln\sigma_i$ decreases. The entropy loss term −c_e · H grows (becomes less negative / more positive) in the total loss, but at scale 0.01 it's too weak to prevent this.

**LR drops aggressively** from 0.00297 → 0.00087. The KL-adaptive scheduler detects:

$$D_{\text{KL}}(\pi_{\theta_\text{old}} \,\|\, \pi_\theta) > \text{kl\_threshold} = 0.02$$

When KL divergence exceeds the threshold, the scheduler **reduces LR** to slow down policy changes. The math: with small σ and rapidly changing μ, the KL between old and new Gaussians is:

$$D_{\text{KL}} = \sum_i \left[\ln\frac{\sigma_i'}{\sigma_i} + \frac{\sigma_i^2 + (\mu_i - \mu_i')^2}{2\,{\sigma_i'}^2} - \frac{1}{2}\right]$$

Small σ amplifies the (μ − μ')² / 2σ'² term, causing KL to spike even for moderate mean shifts.

**Reward rises** 1656 → 3086. The agent learns a reasonable baseline policy quickly.

### 3.2 Phase 2: Exploration Recovery (steps 20k–92k) — Entropy Rebound

| Step | LR | Policy Loss | Value Loss | Entropy Loss | Std Dev | Mean Reward |
|------:|------:|------:|------:|------:|------:|------:|
| 20k | 0.00095 | −0.000929 | 0.000247 | +0.00123 | 0.337 | 3120 |
| 48k | 0.00114 | +0.001448 | 0.000809 | −0.00117 | 0.547 | 2947 |
| 60k | 0.00096 | +0.002783 | 0.001323 | −0.00765 | 0.760 | 2781 |
| 92k | 0.00141 | +0.001292 | 0.002753 | −0.00913 | 1.214 | 3420 |

**Critical dynamics:**

**Std dev rises** from 0.335 → 1.214 (3.6× increase). This is a consequence of max_log_std = 0.7 (so σ_max = e^0.7 ≈ 2.01) and the learned log-std parameters moving toward higher values. The entropy bonus gradient is:

$$\frac{\partial}{\partial \ln\sigma_i}\big(-c_e \cdot H\big) = -c_e$$

This constant gradient pushes ln σ upward at every step. During Phase 1, the policy gradient dominated and compressed σ. Now, with a decent baseline policy learned, the advantage signals $\hat{A}_t$ become smaller in magnitude (value function is more accurate), so the relative strength of the entropy gradient increases. The policy "relaxes" and explores more.

**Policy loss becomes positive** (+0.001 to +0.003). Now −L^CLIP > 0 means L^CLIP < 0. This happens because with increasing σ, the policy assigns more probability to a wider range of actions. For previously high-probability actions that happened to have $\hat{A}_t > 0$, the ratio $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ may actually decrease (the wider distribution dilutes probability at the old action), so $r_t \hat{A}_t$ can be negative even when $\hat{A}_t > 0$.

**Reward temporarily drops** 3120 → 2781 (step 60k). This is the **exploration cost** — wider σ means more random actions, temporarily reducing performance. But this exploration is essential: the agent is searching a broader action space.

**LR increases** from 0.00087 back to 0.00141. The KL-adaptive scheduler detects that with larger σ, the KL divergence between policy updates is smaller:

$$D_{\text{KL}} \propto \frac{(\Delta\mu)^2}{2\,\sigma^2}$$

Larger σ absorbs the same Δμ with less KL divergence, so the scheduler **raises LR** to allow faster learning. This is the adaptive mechanism working as intended.

**Reward recovers and exceeds** the Phase 1 plateau: 2781 → 3420 (step 92k). The exploration found better strategies.

### 3.3 Phase 3: Peak Performance (steps 92k–124k) — Exploitation

| Step | LR | Policy Loss | Value Loss | Entropy Loss | Std Dev | Mean Reward |
|------:|------:|------:|------:|------:|------:|------:|
| 100k | 0.00131 | +0.000616 | 0.004219 | −0.01083 | 1.215 | 3708 |
| 108k | 0.00153 | +0.000950 | 0.001758 | −0.00694 | 1.209 | 4246 |
| 120k | 0.00183 | +0.000912 | 0.001240 | −0.00634 | 1.211 | 4269 |

**Std dev stabilizes** at ~1.21. It has hit an equilibrium where:

$$\underbrace{-c_e \cdot \frac{\partial H}{\partial \ln\sigma}}_{\text{entropy push (constant }+0.01\text{)}} \;\approx\; \underbrace{\frac{\partial L^{\text{CLIP}}}{\partial \ln\sigma}}_{\text{policy gradient pull (toward lower }\sigma\text{)}}$$

The max_log_std = 0.7 also provides a soft ceiling (e^0.7 ≈ 2.01, and mean std is 1.21, so individual dimensions are near this limit).

**LR peaks** at 0.00183. The scheduler sees consistently low KL (large σ ≈ stable), and ramps LR toward the configured base of 0.003.

**Peak reward** 4269 at step 120k. This is the best the agent achieves — the right balance of exploration (high σ) and exploitation (good μ).

### 3.4 Phase 4: Curriculum Shift & Instability (steps 124k–156k)

| Step | LR | Policy Loss | Value Loss | Entropy Loss | Std Dev | Mean Reward |
|------:|------:|------:|------:|------:|------:|------:|
| 124k | 0.00070 | −0.001993 | 0.005627 | −0.01691 | 1.212 | 3320 |
| 140k | 0.00019 | +0.004163 | 0.003586 | −0.01740 | 1.208 | 3926 |
| 144k | 0.00016 | **+0.014098** | 0.004702 | −0.02059 | 1.209 | 3862 |

**Reward drops sharply** 4269 → 3320 at step 124k. This corresponds to a **curriculum stage transition** — the environment increased difficulty (e.g., closer target proximity requirement). The value function's predictions are suddenly wrong for the new task distribution.

**Value loss spikes** from 0.001240 → 0.005627. The value function trained on the old curriculum under-estimates the new returns, so:

$$L^V = \big(V_\phi(s_t) - V_t^{\text{targ}}\big)^2 \;\text{increases}$$

**Policy loss spikes to +0.014** at step 144k. This is the largest policy loss in the entire run. The clipped objective is strongly negative, meaning the policy is taking actions that the advantage function evaluates as poor under the new curriculum. The clip at ε = 0.2 prevents catastrophic updates, but the signal is strong.

**LR crashes** 0.00183 → 0.00016. The KL-adaptive scheduler aggressively reduces LR because the distribution shift caused large KL divergences between consecutive updates. This is a **stabilization mechanism** — without it, the large policy gradients from the curriculum shift could destroy the learned policy.

### 3.5 Phase 5: Late Training & Decay (steps 156k–224k)

| Step | LR | Policy Loss | Value Loss | Entropy Loss | Std Dev | Mean Reward |
|------:|------:|------:|------:|------:|------:|------:|
| 172k | 0.00005 | +0.003992 | 0.003820 | −0.01883 | 1.215 | 3935 |
| 192k | 0.00003 | +0.000712 | 0.009809 | −0.02670 | 1.215 | 3455 |
| 200k | 0.000001 | +0.000292 | 0.010974 | −0.01796 | 1.216 | 2509 |
| 224k | 0.00004 | −0.000457 | 0.005433 | −0.01680 | 1.215 | 3408 |

**LR is near zero** (~0.00001–0.00005). The KL-adaptive scheduler has reduced it so aggressively that the policy is essentially frozen. At LR = 0.000001 (step 200k), the effective parameter update is:

$$\Delta\theta = -\alpha \cdot \nabla_\theta L \;\approx\; -10^{-6} \cdot \nabla_\theta L \;\approx\; 0$$

**Value loss rises** 0.003 → 0.011. With frozen policy parameters, the value network can't track the changing return distribution (curriculum is still shifting), so its predictions degrade. This is a **divergence** — the value function is being trained (it has its own parameters), but the policy it's evaluating is static, creating a non-stationary target.

**Reward oscillates** between 2509 and 3935. The policy can't adapt, and the curriculum continues changing, leading to inconsistent performance. The drop to 2509 at step 200k corresponds to LR = 0.000001, the absolute minimum.

**Late recovery** to 3408 (step 224k) — LR bounces slightly to 0.00004 as KL stays low (frozen policy → zero KL), and the scheduler tentatively increases LR. But it's not enough for meaningful learning.

---

## 4. The Feedback Loop: How All Quantities Steer Each Other

### Causal Diagram

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
            ┌──────────────┐                           │
     ┌─────►│  Policy σ    │◄──── Entropy gradient     │
     │      └──────┬───────┘      (-c_e per step)      │
     │             │                                    │
     │             │ Affects KL between updates          │
     │             ▼                                    │
     │      ┌──────────────┐     ┌───────────────┐     │
     │      │KL Divergence │────►│  Learning Rate │─────┘
     │      └──────────────┘     └───────┬───────┘
     │                                   │
     │             Scales gradient step   │
     │                                   ▼
     │      ┌──────────────┐     ┌───────────────┐
     │      │ Value Loss   │◄───►│  Policy Loss  │
     │      └──────┬───────┘     └───────┬───────┘
     │             │                     │
     │             │  Better V → better  │ Better policy →
     │             │  advantage estimates│ higher rewards
     │             ▼                     ▼
     │      ┌──────────────┐     ┌───────────────┐
     └──────│  Advantages  │     │ Total Reward   │
            └──────────────┘     └───────────────┘
```

### 4a. Learning Rate ↔ Everything

The KL-adaptive scheduler (kl_threshold = 0.02) is the **master regulator**:

$$\alpha_{t+1} = \begin{cases} \alpha_t \;/\; 1.5 & \text{if } D_{\text{KL}} > 2 \times 0.02 \\[4pt] \alpha_t \;\times\; 1.5 & \text{if } D_{\text{KL}} < 0.02 \;/\; 2 \\[4pt] \alpha_t & \text{otherwise} \end{cases}$$

From the logs: LR ranged from **0.000001** (step 200k, policy frozen) to **0.00297** (step 4k, aggressive early learning). The KL divergence between Gaussian policies:

$$D_{\text{KL}} = \sum_i \left[\ln\frac{\sigma_i^{\text{new}}}{\sigma_i^{\text{old}}} + \frac{(\sigma_i^{\text{old}})^2 + (\mu_i^{\text{old}} - \mu_i^{\text{new}})^2}{2\,(\sigma_i^{\text{new}})^2} - \frac{1}{2}\right]$$

is dominated by $(\Delta\mu)^2 / 2\sigma^2$. So:
- **Small σ + large μ changes → high KL → LR drops** (Phase 1: σ=0.33, LR crashed to 0.0009)
- **Large σ + small μ changes → low KL → LR rises** (Phase 3: σ=1.21, LR rose to 0.0018)
- **Curriculum shift → distribution shift → high KL → LR crashes** (Phase 4: LR → 0.0002)

### 4b. Entropy Loss ↔ Policy Std Dev

The entropy loss reported is −c_e · H[π_θ]. For Gaussian policy with std σ across d dimensions:

$$\text{Entropy Loss} = -0.01 \times \left(\frac{d}{2}\ln(2\pi e) + \sum_i \ln\sigma_i\right)$$

From the logs:
- Step 16k: std=0.335, entropy loss = +0.0009 (positive — entropy is low, the loss term is small, minimizer doesn't care).
- Step 120k: std=1.211, entropy loss = −0.0063 (negative — entropy is high, large magnitude — this contributes meaningfully to the loss).

When entropy loss magnitude grows, it means the policy has high entropy. The 0.01 coefficient is small, so this term mainly prevents complete collapse rather than driving exploration.

### 4c. Policy Loss ↔ Total Reward

Policy loss = −L^CLIP. When the policy improves (better actions get higher probability), L^CLIP > 0 and the reported loss is negative. From the logs:

- **Steps 4k–16k**: Policy loss negative (−0.005), reward rising rapidly. Policy is effectively exploiting clear advantage signals.
- **Steps 40k–120k**: Policy loss positive (+0.001 to +0.003), reward rising slowly. The policy is exploring (high σ), so individual updates aren't always improving expected reward, but the cumulative effect enables discovering better strategies.
- **Step 144k**: Policy loss = +0.014 (spike), reward dropping. Curriculum shift made the old policy's actions disadvantageous under the new evaluation.

### 4d. Value Loss ↔ Advantage Quality ↔ Policy Learning

Value loss determines how accurate the advantage estimates are. From the logs:

$$\text{Advantage variance} \;\propto\; L^V$$

- **Step 8k**: Value loss = 0.0003, advantage estimates are excellent, policy loss is clearly negative (−0.0024) → clean policy improvement.
- **Step 192k**: Value loss = 0.0098, advantages are noisy, policy loss is small (+0.0007) → policy can't learn effectively because the signal is buried in noise.

This creates a **vicious cycle** in late training: frozen policy → value function can't stabilize → noisy advantages → no useful policy gradient → policy stays frozen.

### 4e. Total Reward as the Integrated Outcome

Total reward is not a direct input to the PPO update — it's the **outcome**. The actual inputs are per-step rewards r_t used to compute TD targets. But reward reflects the quality of the policy-value system:

| Phase | Mean Reward Trend | Root Cause |
|-------|-------------------|------------|
| 4k→16k | 1656→3086 (+86%) | Value function learns baseline; policy exploits obvious advantages |
| 16k→60k | 3086→2781 (−10%) | Exploration cost: σ tripling from 0.33→0.76 |
| 60k→120k | 2781→4269 (+53%) | Exploration payoff: better strategies discovered |
| 120k→124k | 4269→3320 (−22%) | Curriculum shift: harder task |
| 124k→172k | 3320→3935 (+19%) | Partial recovery under new curriculum |
| 172k→200k | 3935→2509 (−36%) | LR collapse: policy frozen, can't adapt |
| 200k→224k | 2509→3408 (+36%) | Minor LR recovery allows slight adaptation |

---

## 5. Key Takeaway: The KL-Adaptive LR is the Bottleneck

The most significant finding from this run: the **KL-adaptive scheduler overcorrected** after the curriculum shift at ~124k steps. It reduced LR from 0.0018 to 0.000001, effectively freezing the policy for the final 100k steps. The policy never recovered to its peak of 4269.

Mathematically, the issue is that after a curriculum shift, the optimal policy under the new curriculum may be far from the current policy. But the KL constraint $D_{\text{KL}} < 0.02$ combined with tiny LR means the policy can only move in infinitesimal steps:

$$\|\theta_{t+1} - \theta_t\| \;\leq\; \alpha \cdot \frac{\|\nabla L\|}{\text{grad\_norm\_clip}}$$

With α = 10⁻⁶ and grad_norm_clip = 0.3, the maximum parameter change per step is ~3 × 10⁻⁷, which requires O(10⁶) steps to make meaningful progress — far beyond the remaining 100k timesteps.

---

## Appendix: Full Metric Trajectories (drone_0)

### A.1 Learning Rate

| Step | LR |
|------:|------:|
| 4000 | 0.002966 |
| 8000 | 0.001882 |
| 12000 | 0.001381 |
| 16000 | 0.000873 |
| 20000 | 0.000949 |
| 24000 | 0.000922 |
| 28000 | 0.000823 |
| 32000 | 0.001062 |
| 36000 | 0.000926 |
| 40000 | 0.000791 |
| 44000 | 0.000837 |
| 48000 | 0.001139 |
| 52000 | 0.001093 |
| 56000 | 0.000927 |
| 60000 | 0.000960 |
| 64000 | 0.001110 |
| 68000 | 0.001070 |
| 72000 | 0.001366 |
| 76000 | 0.001387 |
| 80000 | 0.001424 |
| 84000 | 0.000861 |
| 88000 | 0.001276 |
| 92000 | 0.001410 |
| 96000 | 0.001351 |
| 100000 | 0.001309 |
| 104000 | 0.001746 |
| 108000 | 0.001525 |
| 112000 | 0.001702 |
| 116000 | 0.001695 |
| 120000 | 0.001825 |
| 124000 | 0.000702 |
| 128000 | 0.000664 |
| 132000 | 0.000511 |
| 136000 | 0.000400 |
| 140000 | 0.000194 |
| 144000 | 0.000163 |
| 148000 | 0.000217 |
| 152000 | 0.000191 |
| 156000 | 0.000087 |
| 160000 | 0.000064 |
| 164000 | 0.000081 |
| 168000 | 0.000074 |
| 172000 | 0.000047 |
| 176000 | 0.000001 |
| 180000 | 0.000008 |
| 184000 | 0.000027 |
| 188000 | 0.000060 |
| 192000 | 0.000026 |
| 196000 | 0.000008 |
| 200000 | 0.000001 |
| 204000 | 0.000006 |
| 208000 | 0.000008 |
| 212000 | 0.000064 |
| 216000 | 0.000051 |
| 220000 | 0.000063 |
| 224000 | 0.000036 |

### A.2 Policy Loss

| Step | Policy Loss |
|------:|------:|
| 4000 | −0.000328 |
| 8000 | −0.002437 |
| 12000 | −0.002324 |
| 16000 | −0.004554 |
| 20000 | −0.000929 |
| 24000 | −0.001038 |
| 28000 | +0.000856 |
| 32000 | +0.001322 |
| 36000 | +0.000930 |
| 40000 | +0.002222 |
| 44000 | +0.001219 |
| 48000 | +0.001448 |
| 52000 | +0.001258 |
| 56000 | +0.001387 |
| 60000 | +0.002783 |
| 64000 | +0.000867 |
| 68000 | +0.002622 |
| 72000 | +0.001568 |
| 76000 | +0.001226 |
| 80000 | +0.000344 |
| 84000 | +0.002811 |
| 88000 | +0.000981 |
| 92000 | +0.001292 |
| 96000 | +0.000610 |
| 100000 | +0.000616 |
| 104000 | +0.000390 |
| 108000 | +0.000950 |
| 112000 | +0.000198 |
| 116000 | +0.000612 |
| 120000 | +0.000912 |
| 124000 | −0.001993 |
| 128000 | +0.001586 |
| 132000 | +0.001527 |
| 136000 | +0.002129 |
| 140000 | +0.004163 |
| 144000 | +0.014098 |
| 148000 | +0.013918 |
| 152000 | +0.012313 |
| 156000 | +0.007822 |
| 160000 | +0.006212 |
| 164000 | +0.004828 |
| 168000 | +0.003284 |
| 172000 | +0.003992 |
| 176000 | +0.003940 |
| 180000 | +0.003948 |
| 184000 | +0.002980 |
| 188000 | +0.001988 |
| 192000 | +0.000712 |
| 196000 | −0.000878 |
| 200000 | +0.000292 |
| 204000 | +0.000387 |
| 208000 | −0.001084 |
| 212000 | −0.001315 |
| 216000 | −0.001033 |
| 220000 | +0.000281 |
| 224000 | −0.000457 |

### A.3 Value Loss

| Step | Value Loss |
|------:|------:|
| 4000 | 0.037746 |
| 8000 | 0.000278 |
| 12000 | 0.000194 |
| 16000 | 0.000298 |
| 20000 | 0.000247 |
| 24000 | 0.000274 |
| 28000 | 0.000414 |
| 32000 | 0.000415 |
| 36000 | 0.000463 |
| 40000 | 0.000538 |
| 44000 | 0.000657 |
| 48000 | 0.000809 |
| 52000 | 0.000749 |
| 56000 | 0.001059 |
| 60000 | 0.001323 |
| 64000 | 0.001416 |
| 68000 | 0.001866 |
| 72000 | 0.001248 |
| 76000 | 0.001183 |
| 80000 | 0.002437 |
| 84000 | 0.003855 |
| 88000 | 0.003680 |
| 92000 | 0.002753 |
| 96000 | 0.002637 |
| 100000 | 0.004219 |
| 104000 | 0.002149 |
| 108000 | 0.001758 |
| 112000 | 0.002031 |
| 116000 | 0.001720 |
| 120000 | 0.001240 |
| 124000 | 0.005627 |
| 128000 | 0.003256 |
| 132000 | 0.003326 |
| 136000 | 0.002886 |
| 140000 | 0.003586 |
| 144000 | 0.004702 |
| 148000 | 0.004337 |
| 152000 | 0.004535 |
| 156000 | 0.004352 |
| 160000 | 0.005893 |
| 164000 | 0.004572 |
| 168000 | 0.004189 |
| 172000 | 0.003820 |
| 176000 | 0.003717 |
| 180000 | 0.004022 |
| 184000 | 0.005024 |
| 188000 | 0.005189 |
| 192000 | 0.009809 |
| 196000 | 0.007496 |
| 200000 | 0.010974 |
| 204000 | 0.010668 |
| 208000 | 0.010928 |
| 212000 | 0.009825 |
| 216000 | 0.007020 |
| 220000 | 0.004490 |
| 224000 | 0.005433 |

### A.4 Entropy Loss

| Step | Entropy Loss |
|------:|------:|
| 4000 | −0.009412 |
| 8000 | −0.004455 |
| 12000 | −0.001455 |
| 16000 | +0.000916 |
| 20000 | +0.001234 |
| 24000 | +0.001541 |
| 28000 | +0.002578 |
| 32000 | +0.001176 |
| 36000 | +0.000839 |
| 40000 | +0.000398 |
| 44000 | −0.000464 |
| 48000 | −0.001171 |
| 52000 | −0.002419 |
| 56000 | −0.003669 |
| 60000 | −0.007651 |
| 64000 | −0.004931 |
| 68000 | −0.005481 |
| 72000 | −0.003788 |
| 76000 | −0.003650 |
| 80000 | −0.005947 |
| 84000 | −0.013232 |
| 88000 | −0.013595 |
| 92000 | −0.009128 |
| 96000 | −0.007980 |
| 100000 | −0.010825 |
| 104000 | −0.006528 |
| 108000 | −0.006942 |
| 112000 | −0.007853 |
| 116000 | −0.007111 |
| 120000 | −0.006340 |
| 124000 | −0.016913 |
| 128000 | −0.014155 |
| 132000 | −0.013208 |
| 136000 | −0.012384 |
| 140000 | −0.017396 |
| 144000 | −0.020593 |
| 148000 | −0.018450 |
| 152000 | −0.018228 |
| 156000 | −0.018453 |
| 160000 | −0.023463 |
| 164000 | −0.019333 |
| 168000 | −0.017694 |
| 172000 | −0.018825 |
| 176000 | −0.018797 |
| 180000 | −0.019280 |
| 184000 | −0.020973 |
| 188000 | −0.017752 |
| 192000 | −0.026696 |
| 196000 | −0.013757 |
| 200000 | −0.017963 |
| 204000 | −0.019568 |
| 208000 | −0.023142 |
| 212000 | −0.024248 |
| 216000 | −0.020957 |
| 220000 | −0.013896 |
| 224000 | −0.016798 |

### A.5 Policy Standard Deviation

| Step | Std Dev |
|------:|------:|
| 4000 | 0.559032 |
| 8000 | 0.451798 |
| 12000 | 0.376587 |
| 16000 | 0.335228 |
| 20000 | 0.336768 |
| 24000 | 0.357531 |
| 28000 | 0.376171 |
| 32000 | 0.429429 |
| 36000 | 0.467151 |
| 40000 | 0.484132 |
| 44000 | 0.508214 |
| 48000 | 0.546933 |
| 52000 | 0.606411 |
| 56000 | 0.678625 |
| 60000 | 0.759995 |
| 64000 | 0.781701 |
| 68000 | 0.814403 |
| 72000 | 0.855745 |
| 76000 | 0.901908 |
| 80000 | 0.974150 |
| 84000 | 1.075314 |
| 88000 | 1.189692 |
| 92000 | 1.214088 |
| 96000 | 1.213581 |
| 100000 | 1.215070 |
| 104000 | 1.214070 |
| 108000 | 1.209352 |
| 112000 | 1.211413 |
| 116000 | 1.212139 |
| 120000 | 1.211177 |
| 124000 | 1.212106 |
| 128000 | 1.210532 |
| 132000 | 1.208128 |
| 136000 | 1.207860 |
| 140000 | 1.207958 |
| 144000 | 1.209367 |
| 148000 | 1.213544 |
| 152000 | 1.216384 |
| 156000 | 1.216744 |
| 160000 | 1.216443 |
| 164000 | 1.216173 |
| 168000 | 1.215366 |
| 172000 | 1.215238 |
| 176000 | 1.215140 |
| 180000 | 1.215098 |
| 184000 | 1.214990 |
| 188000 | 1.214674 |
| 192000 | 1.214585 |
| 196000 | 1.215317 |
| 200000 | 1.215982 |
| 204000 | 1.216500 |
| 208000 | 1.216697 |
| 212000 | 1.216542 |
| 216000 | 1.215892 |
| 220000 | 1.214906 |
| 224000 | 1.214597 |

### A.6 Total Reward (Mean)

| Step | Mean Reward |
|------:|------:|
| 4000 | 1656.22 |
| 8000 | 2716.79 |
| 12000 | 2981.23 |
| 16000 | 3086.29 |
| 20000 | 3119.88 |
| 24000 | 3118.74 |
| 28000 | 3114.57 |
| 32000 | 3119.96 |
| 36000 | 3081.78 |
| 40000 | 3075.36 |
| 44000 | 2999.19 |
| 48000 | 2947.07 |
| 52000 | 2887.12 |
| 56000 | 2776.87 |
| 60000 | 2780.80 |
| 64000 | 2733.76 |
| 68000 | 2766.60 |
| 72000 | 2943.48 |
| 76000 | 2886.80 |
| 80000 | 2741.12 |
| 84000 | 3013.35 |
| 88000 | 3242.97 |
| 92000 | 3419.88 |
| 96000 | 3477.14 |
| 100000 | 3708.08 |
| 104000 | 4173.37 |
| 108000 | 4246.09 |
| 112000 | 4170.63 |
| 116000 | 4204.44 |
| 120000 | 4269.37 |
| 124000 | 3319.84 |
| 128000 | 3842.35 |
| 132000 | 3956.76 |
| 136000 | 3935.13 |
| 140000 | 3926.10 |
| 144000 | 3861.86 |
| 148000 | 3920.18 |
| 152000 | 3869.32 |
| 156000 | 3802.08 |
| 160000 | 3880.92 |
| 164000 | 3936.92 |
| 168000 | 3981.46 |
| 172000 | 3934.57 |
| 176000 | 3934.78 |
| 180000 | 3932.28 |
| 184000 | 3883.92 |
| 188000 | 3628.69 |
| 192000 | 3454.95 |
| 196000 | 2681.94 |
| 200000 | 2509.06 |
| 204000 | 2637.98 |
| 208000 | 2891.82 |
| 212000 | 3204.15 |
| 216000 | 3363.60 |
| 220000 | 3394.42 |
| 224000 | 3407.62 |
