# Reward State Design: Ground Truth vs. Observed States

**Project:** `iris_ma6` — Multi-Drone Active Triangulation  
**Scope:** Reward function state dependency design  
**Status:** Draft v2  
**Last Updated:** 2026-03-10

---

## 1. Problem Statement

The IRIS system is a Dec-POMDP with communication delays: agents receive observations $o_i^t$ that are delayed versions of the true state $s^t$. Two interrelated design questions arise:

1. **State source:** Should reward components be computed from $s^t$ (ground truth) or $o^t$ (delayed observations)?
2. **Reward target:** Should the task reward measure geometric proxy (FIM), estimation quality (triangulation error), or end-to-end system performance?

iris_ma5 computes all rewards from delayed observations using FIM as the task metric. This document examines alternatives for iris_ma6.

---

## 2. Formal Setup

### 2.1 Dec-POMDP Definition

The standard Dec-POMDP tuple $\langle \mathcal{I}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{T}, R, \{\mathcal{O}_i\}, \mathcal{Z}, \gamma \rangle$ defines the reward as:

$$R : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$$

The reward is a function of the **true state**, not the observation. This is definitional. The objective is:

$$J(\boldsymbol{\pi}) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t\, R(s^t, \mathbf{a}^t)\right]$$

Each agent's policy $\pi_i(a_i \mid \tau_i^t)$ conditions on its local action-observation history $\tau_i^t$, which is a sufficient statistic for its belief over the true state.

### 2.2 The Two State Source Formulations

**Formulation A — GT reward (standard Dec-POMDP):**

$$r^t = R(s^t, \mathbf{a}^t)$$

The policy optimizes for true-state quality. The agent must implicitly estimate $s^t$ from $\tau_i^t$ to take good actions. The centralized critic $V(s)$ uses privileged state access (CTDE).

**Formulation B — Delayed-observation reward (iris_ma5):**

$$r^t = \tilde{R}(o^t, \mathbf{a}^t)$$

The policy optimizes for observable quality. The reward is in the same information space as the policy's input. The critic $V(o)$ doesn't need privileged access.

These are different optimization problems. Neither is incorrect — they optimize for different objectives.

---

## 3. Credit Assignment Analysis

### 3.1 Delayed Rewards Have Easier Credit Assignment

This is the opposite of what one might initially assume. The logic:

**Delayed reward — consistent information frame.** The agent observes $o^{t-\tau}$, takes action $a^t$, and receives reward $\tilde{R}(o^{t-\tau})$. The observation and the reward are in the **same information space**. The critic $V(o)$ predicts future rewards from observations, and the advantage $\hat{A}^t = \tilde{r}^t + \gamma V(o^{t+1-\tau}) - V(o^{t-\tau})$ compares consecutive observations in a consistent frame. The delay shifts everything uniformly — observation, reward, and value — so the temporal difference correctly identifies whether the action improved the situation *as the agent sees it*.

**GT reward — mismatched information frames.** The agent observes $o^{t-\tau}$, takes action $a^t$, and receives reward $R(s^t)$. The reward refers to a state the agent cannot see. The critic estimates $V(s)$ from privileged information, but the policy gradient $\nabla_\theta \log \pi(a \mid o^{t-\tau}) \cdot \hat{A}^t$ multiplies an observation-space policy by a state-space advantage. The policy must learn: "when I saw observation $o$ (delayed), and the true-state advantage was $A$, what should I have done differently?" — even though the mapping from $o$ to $s$ is exactly the implicit estimation problem the agent struggles with.

### 3.2 Summary

| Property | GT reward | Delayed reward |
|---|---|---|
| Credit assignment | Harder — reward refers to unobservable state | Easier — reward matches observation |
| Objective correctness | Optimizes physical reality | Optimizes observable proxy |
| Critic design | Needs privileged access (CTDE) | Can use same observations as actor |
| What policy learns | Delay compensation + task (harder) | Task in observation space (easier) |
| Information consistency | Reward and observation in different spaces | Reward and observation in same space |

The GT argument is about **what you're optimizing for** (physical reality), not about making learning easier. It may make learning harder, and whether that cost is worth the benefit is empirical.

### 3.3 When Delayed Rewards Are Defensible

The observation-based formulation has a genuine advantage when the **information constraint is binding** — when the GT-optimal behavior is not achievable given the observation limitations.

If the optimal triangulation geometry requires knowing the target's position within 0.5 m, but the delay introduces 3 m of uncertainty, no policy can reliably maintain GT-optimal formation. A GT-reward policy might overshoot toward the optimal geometry but oscillate because it can't observe when it's achieved. A delayed-reward policy finds the best formation *maintainable* given its information — potentially suboptimal in GT terms but more stable.

This is analogous to the certainty-equivalence principle in control theory: sometimes optimizing for the expected state produces worse outcomes than optimizing for the observable state, because the former ignores the cost of estimation error.

### 3.4 When GT Rewards Are Clearly Better

GT rewards are preferable when:

- **The reward measures a physical event.** Collision is physical reality regardless of observation. Safety rewards should always use GT.
- **The delay is small relative to task dynamics.** The GT-optimal and observation-optimal formations are nearly identical, but GT rewards avoid the observation pipeline's distortion.
- **The reward component measures something the agent controls directly.** Formation geometry is a function of the agents' true positions. Using delayed positions introduces noise from a process (comm delay) that the policy can't influence.

---

## 4. Task Reward Components for IRIS

### 4.1 Quantities Available in Simulation

| Symbol | Description | Computed from |
|---|---|---|
| $\mathbf{p}_T^{true}$ | Target true position | Sim GT |
| $\mathbf{p}_i^{true}$ | Drone $i$ true position | Sim GT |
| $\mathbf{p}_i^{delayed}$ | Drone $i$ position as known to the system | Delayed observation |
| $\beta_i^{meas}$ | Bearing measurement from drone $i$ to target | Physics: $f(\mathbf{p}_i^{true}, \mathbf{p}_T^{true}) + \text{noise}$ |
| $\hat{\mathbf{p}}_T^{GT}$ | Triangulation using GT drone positions | $\text{triangulate}(\mathbf{p}_i^{true}, \beta_i^{meas})$ |
| $\hat{\mathbf{p}}_T^{delayed}$ | Triangulation using delayed drone positions | $\text{triangulate}(\mathbf{p}_i^{delayed}, \beta_i^{meas})$ |
| $\text{FIM}^{GT}$ | Fisher Information from GT geometry | $f(\mathbf{p}_i^{true}, \mathbf{p}_T^{true})$ |
| $\text{FIM}^{delayed}$ | Fisher Information from perceived geometry | $f(\mathbf{p}_i^{delayed}, \mathbf{p}_T^{delayed})$ |

Note: bearing measurements $\beta_i^{meas}$ are always physical — they're taken at the true positions with sensor noise. The physical measurement is the same regardless of what position knowledge the triangulation algorithm uses to interpret it.

### 4.2 Three Levels of Task Reward

Each measures something different about the system's quality:

#### Level 1: Formation Geometry Proxy (FIM)

$$r_{FIM} = f(\text{FIM}(\mathbf{p}_i, \mathbf{p}_T))$$

Measures the geometric potential for accurate triangulation — baseline angles, spatial diversity, condition number of the measurement geometry. This is a proxy: good FIM implies good estimation is *possible*, but doesn't guarantee it.

**Advantages:** Smooth, well-behaved, differentiable w.r.t. positions. No bearing measurement noise. Easy to optimize.

**Disadvantage:** Proxy gap. A formation with excellent FIM where delayed self-positions corrupt the triangulation would get high reward despite poor actual estimation. Doesn't capture the real estimation chain.

#### Level 2: GT-Anchored Estimation Error

$$r_{est}^{GT} = -\|\hat{\mathbf{p}}_T^{GT} - \mathbf{p}_T^{true}\|$$

Runs the actual triangulation algorithm using **true** drone positions to anchor bearing rays. Compares the resulting target estimate to ground truth.

**What it isolates:** Pure formation quality. "If we had perfect self-localization, how good would this formation be for finding the target?" Removes the confound of delayed self-position knowledge.

**Why this is good for training:** The policy controls formation geometry. This reward measures the downstream effect of that geometry on actual estimation, passing through the real bearing noise and triangulation algorithm, but without penalizing the policy for delay-induced position errors it can't control.

**Disadvantage:** Noisier than FIM (bearing measurement noise propagates). Each evaluation is a stochastic sample, not a deterministic geometric quantity.

#### Level 3: End-to-End Estimation Error

$$r_{est}^{E2E} = -\|\hat{\mathbf{p}}_T^{delayed} - \mathbf{p}_T^{true}\|$$

Runs the actual triangulation algorithm using **delayed** drone positions. This is the real system performance — what the system actually delivers at deployment.

**What it measures:** Everything — formation geometry + bearing noise + delay-induced self-position error. The most honest metric.

**Advantage:** Directly optimizes deployment performance. If delayed self-positions degrade triangulation, the policy learns about it and can potentially adapt (e.g., smoother maneuvers produce less position drift during the delay, so triangulation is more accurate even if geometry is slightly worse).

**Disadvantage:** The policy is rewarded/penalized for two coupled factors — formation geometry AND delay-induced position error — making it harder to learn which aspect of the action affected the reward. Also noisiest of the three options.

### 4.3 Component-Wise State Source Recommendation

| Reward component | GT state | Delayed state | Recommendation |
|---|---|---|---|
| **Safety penalty** (CBF margin, collision) | Collision is physical reality; delayed obs misses emerging dangers | No compelling argument | **GT** (clear) |
| **FIM** (if used) | Measures true geometric potential | Measures perceived geometric potential | **GT** preferred — policy controls true geometry |
| **GT-anchored estimation error** | By definition uses GT positions | N/A | **GT** (by construction) |
| **End-to-end estimation error** | N/A | By definition uses delayed positions | **Delayed** (by construction) |
| **Control effort** | Action is known exactly | Same | **Either** (no difference) |

The interesting question is not GT vs. delayed for a fixed metric, but **which metric to use** — each one naturally determines its own state source.

---

## 5. Reward Composition

### 5.1 Recommended Configuration

```python
def compute_reward(self):
    # Primary task reward: GT-anchored estimation error
    # Isolates formation quality without delay confound
    p_est_gt = self.triangulate(self.true_states)
    est_error_gt = (p_est_gt - self.true_target_pos).norm(dim=-1)

    # Auxiliary: end-to-end estimation error
    # Teaches policy about delay-induced degradation
    p_est_delayed = self.triangulate(self.delayed_states)
    est_error_e2e = (p_est_delayed - self.true_target_pos).norm(dim=-1)

    # Safety: always GT
    safety_penalty = self.cbf_penalty(self.true_states, actions)

    reward = (
        - lambda_est * est_error_gt       # formation quality (primary)
        - lambda_e2e * est_error_e2e       # delay awareness (auxiliary)
        - lambda_cbf * safety_penalty      # safety
    )
```

**Rationale for the combination:**

- The GT-anchored error ($\lambda_{est}$) is the main task signal. It tells the policy: "your formation geometry produced this estimation quality." The policy learns to create good triangulation geometry, evaluated through the actual estimation pipeline (bearing noise, triangulation algorithm), but without being punished for comm delay it can't control.
- The end-to-end error ($\lambda_{e2e}$) is a secondary signal that teaches delay awareness. Formations that are hard to exploit with stale self-positions are penalized. The policy learns: "even if the geometry is good, if my motion during the delay window corrupts the position anchoring, the estimate degrades."
- The balance $\lambda_{est} / \lambda_{e2e}$ controls the trade-off between geometric optimality and delay robustness.

### 5.2 Curriculum Approach

Connect to the existing curriculum learning framework:

| Training phase | Task reward | Rationale |
|---|---|---|
| Early | FIM only | Smooth, easy to optimize. Policy learns basic formation geometry. |
| Mid | GT-anchored estimation error | Noisier but real. Policy refines formations based on actual triangulation performance. |
| Late | GT-anchored + end-to-end | Adds delay awareness. Policy learns delay-robust formations. |

This progressively moves the policy from optimizing a smooth proxy to optimizing the real deployment objective, using the earlier phases to provide a good initialization for the harder later objectives.

---

## 6. Ablation Design

### 6.1 Experiment Matrix

**State source ablation (fixed metric: FIM):**

| Experiment | Task reward | Safety reward | Critic input | Tests |
|---|---|---|---|---|
| A | $\text{FIM}^{delayed}$ | Termination only | Delayed | iris_ma5 baseline |
| B | $\text{FIM}^{delayed}$ | CBF penalty (GT) | Delayed + GT | Safety reward GT effect |
| C | $\text{FIM}^{GT}$ | CBF penalty (GT) | GT | Full GT effect |

**Reward target ablation (fixed state source: GT where applicable):**

| Experiment | Task reward | Tests |
|---|---|---|
| D | $\text{FIM}^{GT}$ | Geometric proxy |
| E | $-\|\hat{\mathbf{p}}_T^{GT} - \mathbf{p}_T^{true}\|$ | Estimation quality (geometry-isolated) |
| F | $-\|\hat{\mathbf{p}}_T^{delayed} - \mathbf{p}_T^{true}\|$ | End-to-end system performance |
| G | Weighted E + F | Geometry + delay robustness |

**Delay severity interaction (run best config from above at each delay level):**

| Delay | 0 steps | 2 steps | 4 steps |
|---|---|---|---|
| GT vs delayed reward gap | Expected: small | Expected: moderate | Expected: large (or reversed) |

### 6.2 Evaluation Protocol

All experiments evaluated on the same deployment metrics regardless of training reward:

- $\|\hat{\mathbf{p}}_T^{delayed} - \mathbf{p}_T^{true}\|$ — end-to-end estimation error (the metric that matters)
- $\text{FIM}^{GT}$ — formation geometry quality
- Collision rate (GT)
- Formation quality variance (stability)
- Training convergence speed

The key comparisons:
- **A vs C:** Does GT state source matter? (iris_ma5 → iris_ma6 main hypothesis)
- **D vs E:** Does actual estimation error beat FIM as a reward signal?
- **E vs F:** Does end-to-end awareness help or hurt?
- **Delay × state source interaction:** Does the GT advantage disappear at high delay?

### 6.3 Hypotheses

**State source (A vs C):**
- C outperforms A on collision rate (clear — GT safety rewards are better).
- C outperforms A on mean estimation quality at low delay.
- At high delay, the gap narrows or reverses for the task component, because the information constraint becomes binding.

**Reward target (D vs E vs F):**
- E outperforms D because it captures the actual estimation chain, not just a proxy.
- F underperforms E in isolation because the reward is noisier and conflates controllable (geometry) and uncontrollable (delay) factors.
- G (weighted E+F) outperforms both E and F alone, because the auxiliary E2E signal teaches delay-robust behavior without overwhelming the geometric signal.

**Delay interaction:**
- At 0 delay: all formulations perform similarly (no information constraint).
- At 4 steps delay: delayed-reward policy may show lower variance (doesn't chase unverifiable optima); GT-reward policy may show higher mean (optimizes for reality) or oscillate (can't verify achievement).

---

## 7. Framing as Contribution

The ablation investigates **how reward information structure interacts with observation delay in cooperative MARL for active perception tasks**. This is a Type 4 (Empirical Insight) contribution:

- **Existing assumption:** Dec-POMDP rewards are on true states (standard formulation), and FIM is a sufficient proxy for estimation quality.
- **Practical deviations:** (a) Many MARL implementations compute rewards from observations; (b) FIM doesn't capture the full estimation pipeline.
- **Research questions:** (1) Does the GT vs. delayed state source matter, and under what delay regimes? (2) Does actual estimation error outperform FIM as a reward signal? (3) How do these choices interact with observation delay severity?
- **Methodology:** Controlled ablation across state sources, reward targets, and delay conditions (0, 2, 4 timesteps), with all experiments evaluated on identical deployment metrics.

The connection to the curriculum learning work is direct: the curriculum paper showed how task property analysis guides learning schedule design; this investigation shows how information structure analysis guides reward function design. Same methodological principle — understanding the task properties to make better engineering decisions.

---

## 8. References

- Bernstein, D.S. et al. "The Complexity of Decentralized Control of Markov Decision Processes." Mathematics of Operations Research, 2002.
- Brunke, L. et al. "Safe Learning in Robotics." Annual Review of Control, Robotics, and Autonomous Systems, 2022.
- Eck, A. et al. "Potential-Based Reward Shaping for POMDPs." AAMAS, 2016.
- Ng, A.Y. et al. "Policy Invariance Under Reward Transformations." ICML, 1999.
- Oliehoek, F.A. & Amato, C. "A Concise Introduction to Decentralized POMDPs." Springer, 2016.
- OpenAI et al. "Solving Rubik's Cube with a Robot Hand." arXiv:1910.07113, 2019.
- Rashid, T. et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." ICML, 2018.
- Tishby, N. & Polani, D. "Information Theory of Decisions and Actions." Perception-Action Cycle, Springer, 2011.
- Tobin, J. et al. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." IROS, 2017.
- Yu, C. et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS, 2022.