# Chapter 1: From Optimal Control to Reinforcement Learning

> **Reading time**: ~45 minutes. Best absorbed with pencil and paper.
> **Companion readings**: Sutton & Barto Ch 3, 6.1-6.3, 13.1-13.3

---

## 1.1 The Rosetta Stone: Optimal Control vs RL

You already know optimal control. RL solves the same problem with different assumptions.
Here is the mapping between the two worlds:

| Optimal Control | Reinforcement Learning | Relationship |
|---|---|---|
| State $x_t$ | State $s_t$ | Same concept |
| Control input $u_t$ | Action $a_t$ | Same concept |
| Dynamics $x_{t+1} = f(x_t, u_t)$ | Transition $P(s_{t+1} \mid s_t, a_t)$ | RL doesn't assume $f$ is known |
| Stage cost $\ell(x_t, u_t)$ | Reward $r_t = R(s_t, a_t)$ | Flipped sign: minimize cost vs maximize reward |
| Cost-to-go $J^*(x_t) = \min_{u_{t:T}} \sum_{k=t}^{T} \gamma^k \ell_k$ | Value function $V^\pi(s_t) = \mathbb{E}\left[\sum_{k=t}^{T} \gamma^k r_k\right]$ | Same structure, opposite sign, plus expectation |
| Control law $u_t = K(x_t)$ | Policy $a_t \sim \pi_\theta(a \mid s)$ | RL policies are typically stochastic |
| Bellman equation (HJB) | Bellman equation | Identical structure |
| Riccati equation (LQR) | Analytical solution for linear-quadratic case | Same thing |

The two **fundamental differences** are:

1. **Unknown dynamics.** In optimal control you have $f(x,u)$ and can compute $\partial f / \partial u$. In RL, you don't have $f$ -- you can only *sample* transitions by running the system. This is why RL needs trial-and-error instead of computing gradients analytically.

2. **Stochastic policies.** Optimal control laws are typically deterministic: $u = K(x)$. RL policies are distributions $\pi(a|s)$ -- they output a probability over actions, not a single action. This seems wasteful, but it enables *exploration* (trying new things to discover better strategies) and makes the optimization landscape smoother.

> **For your intuition**: Think of RL as "optimal control where you've lost the dynamics model and have to learn everything from rollouts." The math is the same Bellman structure; the algorithms are different because the information is different.

---

## 1.2 The Bellman Equation: Same Equation, Two Worlds

In optimal control (discrete time, infinite horizon, discount $\gamma$), the cost-to-go satisfies:

$$J^*(x) = \min_u \left[\ell(x,u) + \gamma \, J^*(f(x,u))\right]$$

In RL, the value function under policy $\pi$ satisfies:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}\left[R(s,a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot|s,a)}[V^\pi(s')]\right]$$

These are the same equation. The differences:
- **Sign flip**: minimize cost vs maximize reward
- **Expectation**: because the policy is stochastic and transitions may be stochastic
- **$\pi$ superscript**: the value is policy-dependent; we evaluate $V$ *for a given policy* rather than for the optimal one

The **optimal** value function satisfies:

$$V^*(s) = \max_a \left[R(s,a) + \gamma \, \mathbb{E}_{s'}[V^*(s')]\right]$$

which is exactly the HJB equation in discrete time. If you could solve this, you'd be done. The problem: $s$ might be a 50-dimensional observation vector, and you'd need to evaluate $V^*$ at every point in that space.

> **Sutton & Barto**: Read Chapter 3 (Sections 3.1-3.6) now. It formalizes the MDP framework. You'll find this very familiar from optimal control -- the notation is just different.

---

## 1.3 Why Not Just Solve the Bellman Equation?

In optimal control for linear systems, you solve the Riccati equation and get $J^*(x) = x^T P x$. Done.

For nonlinear systems, you might use dynamic programming on a grid, or iterative methods. This works when:
- The state space is low-dimensional (can be discretized)
- The dynamics are known (can propagate forward)

In your drone problem, neither holds:
- **State space**: ~50 dimensional observations per agent (positions, velocities, gimbal angles, target state, communication, etc.)
- **Dynamics**: the full Isaac Sim physics engine -- no analytical model

Even if you discretized each dimension into 100 bins, you'd have $100^{50} = 10^{100}$ states. This is the **curse of dimensionality**.

The solution: **function approximation**. Instead of storing $V(s)$ in a table, parameterize it as a neural network $V_\phi(s)$ with parameters $\phi$. Instead of searching over all possible control laws, parameterize the policy as a neural network $\pi_\theta(a|s)$ with parameters $\theta$.

Now the problem becomes: **optimize $\theta$ and $\phi$ to maximize expected reward.**

This is deep RL. The "deep" just means "the function approximator is a neural network."

---

## 1.4 Two Families of Approaches

There are two ways to optimize the policy. Understanding both helps you see why PPO is designed the way it is.

### Value-based methods (the DP path)

Learn $V^*(s)$ or $Q^*(s,a)$ first, then derive the policy from it:

$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

This is the RL analogue of solving HJB first, then computing the optimal control. Examples: Q-learning, DQN.

**Problem**: This only works for discrete actions ($\arg\max$ over a finite set). Your drones have 7 continuous action dimensions (vx, vy, vz, yaw_rate, gimbal_yaw, gimbal_pitch, zoom_rate). You can't enumerate all possible actions.

### Policy gradient methods (the direct optimization path)

Optimize $\pi_\theta$ directly by gradient ascent on the expected return:

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

where $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \gamma^t r_t\right]$.

This is the RL analogue of **sensitivity analysis** in optimal control: compute $dJ/d\theta$ and move $\theta$ in the direction that improves $J$.

**PPO is a policy gradient method.** It's in this family.

The fundamental question is: how do you compute $\nabla_\theta J(\theta)$ when you don't know the dynamics?

---

## 1.5 The Policy Gradient Theorem

This is the single most important result in policy gradient RL. It answers the question above.

### The problem

We want $\nabla_\theta J(\theta)$ where:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right]$$

The expectation is over trajectories $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ sampled by running policy $\pi_\theta$ in the environment. The distribution of $\tau$ depends on both $\pi_\theta$ (which we control) and $P(s'|s,a)$ (the dynamics, which we don't know).

Naively, computing this gradient seems to require $\partial P / \partial \theta$, which doesn't exist (the dynamics don't depend on our policy parameters).

### The theorem

The **policy gradient theorem** (Sutton et al., 1999) shows:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t \, \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \Psi_t\right]$$

where $\Psi_t$ is some measure of "how good was action $a_t$." The dynamics $P(s'|s,a)$ have vanished from the gradient expression. The gradient only depends on:

1. $\nabla_\theta \log \pi_\theta(a_t|s_t)$ -- how changing $\theta$ changes the log-probability of the action we took (we know this, it's our neural network)
2. $\Psi_t$ -- how good that action was (we can estimate this from observed rewards)

### Intuition from optimal control

Think of $\nabla_\theta \log \pi_\theta(a|s)$ as the **sensitivity of the policy** to parameter changes. It's analogous to $\partial u / \partial \theta$ in sensitivity analysis. The theorem says: weight this sensitivity by the quality of the action, and you get the gradient of the objective. No dynamics model needed.

### What is $\Psi_t$?

Different choices give different algorithms:

| Choice of $\Psi_t$ | Name | Properties |
|---|---|---|
| $\sum_{k=t}^T \gamma^{k-t} r_k$ (total future return) | REINFORCE | Unbiased but very high variance |
| $\sum_{k=t}^T \gamma^{k-t} r_k - V^\pi(s_t)$ (return minus baseline) | REINFORCE with baseline | Unbiased, lower variance |
| $Q^\pi(s_t, a_t) - V^\pi(s_t)$ (advantage) | Advantage actor-critic | Even lower variance, but needs $Q^\pi$ |
| $\hat{A}_t^{\text{GAE}}$ | GAE (what PPO uses) | Tunable bias-variance via $\lambda$ |

The best choice is the **advantage function** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, which measures: "how much better is action $a$ compared to the average action from state $s$?"

Using the advantage instead of raw returns dramatically reduces variance because it removes the contribution of the state value (which is common to all actions from that state).

> **Sutton & Barto**: Read Chapter 13, Sections 13.1-13.3. This derives the policy gradient theorem carefully. Pay particular attention to the "baseline" discussion in 13.4 -- it directly motivates why we use advantages.

---

## 1.6 Temporal Difference Learning: Learning $V$ from Fragments

To compute advantages, we need the value function $V^\pi(s)$. How do we learn it?

### Monte Carlo: wait for the full return

Run a complete episode. Observe the actual return $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$. Update:

$$V(s_t) \leftarrow V(s_t) + \alpha \, (G_t - V(s_t))$$

**Problem**: You must wait until the episode ends. In your drone environment, episodes are hundreds of steps. High variance because $G_t$ accumulates noise from all future steps.

### Temporal Difference (TD): bootstrap from the next step

Instead of waiting for the full return, use the Bellman equation to estimate it:

$$G_t \approx r_t + \gamma \, V(s_{t+1})$$

Update:

$$V(s_t) \leftarrow V(s_t) + \alpha \, \underbrace{(r_t + \gamma \, V(s_{t+1}) - V(s_t))}_{\delta_t \text{ (TD error)}}$$

The **TD error** $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ measures "how surprised were we by this transition?"

- $\delta_t > 0$: the actual reward + next state value was *better* than expected. We should increase $V(s_t)$.
- $\delta_t < 0$: worse than expected. Decrease $V(s_t)$.

### Optimal control analogy

The TD error is a **Bellman residual**. In optimal control, if $J^*$ perfectly satisfies the Bellman equation, then:

$$\ell(x,u) + \gamma J^*(f(x,u)) - J^*(x) = 0$$

The TD error is the sampled version of this residual. TD learning drives this residual toward zero using stochastic gradient descent.

> **Sutton & Barto**: Read Chapter 6, Sections 6.1-6.3. This introduces TD learning. The key insight is bootstrapping: using your own estimate $V(s_{t+1})$ to improve your estimate $V(s_t)$.

---

## 1.7 Generalized Advantage Estimation (GAE)

Now we can connect everything. We need the advantage $A^\pi(s_t, a_t)$ for the policy gradient. We have a learned value function $V_\phi(s)$. GAE combines multiple TD errors to estimate the advantage with a tunable bias-variance tradeoff.

### The 1-step advantage estimate

$$\hat{A}_t^{(1)} = \delta_t = r_t + \gamma \, V_\phi(s_{t+1}) - V_\phi(s_t)$$

This is **low variance** (uses only one step of randomness) but **high bias** (heavily depends on $V_\phi$ being accurate -- if $V_\phi$ is wrong, so is the advantage).

### The $n$-step advantage estimate

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)$$

More steps = more actual rewards = less bias, but more variance.

### The infinite-step (Monte Carlo) advantage

$$\hat{A}_t^{(\infty)} = \sum_{k=0}^{T-t} \gamma^k r_{t+k} - V_\phi(s_t) = G_t - V_\phi(s_t)$$

No bias (uses actual returns), but maximum variance.

### GAE: exponentially weighted combination

GAE takes an exponentially weighted average of all $n$-step advantages:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1-\lambda)\left[\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots\right]$$

This simplifies to:

$$\boxed{\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \, \delta_{t+l}}$$

where $\delta_k = r_k + \gamma V_\phi(s_{k+1}) - V_\phi(s_k)$ is the TD error at step $k$.

### The $\lambda$ knob

| $\lambda$ | Behavior | Bias | Variance |
|---|---|---|---|
| 0 | $\hat{A}_t = \delta_t$ (1-step TD) | High (trusts $V_\phi$ completely) | Low |
| 1 | $\hat{A}_t = G_t - V_\phi(s_t)$ (Monte Carlo) | Low (uses actual returns) | High |
| 0.95 (your setting) | Weighted blend, favoring longer horizons | Low-medium | Medium-high |

**Your setting** ($\lambda = 0.95$) leans toward Monte Carlo. This makes sense in multi-agent settings where the environment is noisy -- you'd rather have unbiased (but noisy) advantages than biased ones that systematically misjudge the situation.

### Optimal control analogy

GAE is analogous to choosing between 1-step vs $n$-step model predictive control. With a perfect model (perfect $V_\phi$), 1-step is enough. With an imperfect model, you want to rely more on actual observed costs and less on the model's predictions. $\lambda$ is the "how much do you trust your model" knob.

---

## 1.8 The Actor-Critic Architecture

Putting it all together, we have two neural networks:

1. **Actor** (policy): $\pi_\theta(a|s)$ -- outputs a distribution over actions given state
2. **Critic** (value function): $V_\phi(s)$ -- estimates the expected return from state $s$

The training loop:
1. **Collect data**: run the actor in the environment, record $(s_t, a_t, r_t, s_{t+1})$ tuples
2. **Compute advantages**: use the critic + GAE to compute $\hat{A}_t$ for each timestep
3. **Update actor**: policy gradient using $\hat{A}_t$ (how to do this well is Chapter 2 -- PPO)
4. **Update critic**: minimize TD error (reduce $\|V_\phi(s_t) - V_t^{\text{target}}\|^2$)
5. **Repeat**

This is exactly what your MAPPO training does. The actor outputs 7-dimensional Gaussian actions for each drone. The critic estimates expected return from the global state.

---

## 1.9 Summary: What You Already Know, Reframed

| You already know... | Which in RL is called... |
|---|---|
| Cost-to-go $J^*(x)$ | Optimal value function $V^*(s)$ |
| Bellman/HJB equation | Bellman equation (same thing) |
| Sensitivity analysis $dJ/d\theta$ | Policy gradient $\nabla_\theta J(\theta)$ |
| Model predictive control horizon | Bias-variance tradeoff ($\lambda$ in GAE) |
| LQR (solve Riccati analytically) | Tabular DP (solve Bellman exactly) -- both limited to simple systems |
| Nonlinear MPC with function approximation | Deep RL with neural network approximation |
| Controller $u = K(x)$ | Deterministic policy $a = \mu_\theta(s)$ |
| Randomized control (dithering for ID) | Stochastic policy $a \sim \pi_\theta(a|s)$ (exploration) |

The conceptual leap from optimal control to RL is small. The algorithmic leap is large, because you've lost the dynamics model and must learn everything from data.

---

## Quiz 1

Test your understanding before moving to Chapter 2.

**Q1**: Your drone policy outputs actions from $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma^2 I)$. You observe that action $a_t = [3, 0, 0, ...]$ from state $s_t$ led to a reward 500 higher than expected ($\hat{A}_t = 500$). Qualitatively, what does the policy gradient do to $\mu_\theta(s_t)$?

<details>
<summary>Answer</summary>

The policy gradient is:

$$\nabla_\theta J \propto \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t$$

For a Gaussian policy, $\nabla_\mu \log \pi = (a_t - \mu) / \sigma^2$. Since $\hat{A}_t > 0$, the gradient pushes $\mu_\theta(s_t)$ *toward* $a_t = [3, 0, 0, ...]$. In words: "that action was better than average, so make it more likely next time."

If $\hat{A}_t$ were negative, the gradient would push $\mu$ *away* from $a_t$.
</details>

**Q2**: You have $\lambda = 0.95$ and $\gamma = 0.99$. The TD errors for 4 consecutive steps are $\delta_0 = 10, \delta_1 = -5, \delta_2 = 3, \delta_3 = 1$. Compute $\hat{A}_0^{\text{GAE}}$ (to 2 decimal places).

<details>
<summary>Answer</summary>

$$\hat{A}_0 = \delta_0 + (\gamma\lambda)\delta_1 + (\gamma\lambda)^2\delta_2 + (\gamma\lambda)^3\delta_3$$

$\gamma\lambda = 0.99 \times 0.95 = 0.9405$

$$\hat{A}_0 = 10 + 0.9405 \times (-5) + 0.9405^2 \times 3 + 0.9405^3 \times 1$$

$$= 10 - 4.7025 + 2.6524 + 0.8318 = 8.78$$

Note how the negative TD error at step 1 ($\delta_1 = -5$) is discounted by $\gamma\lambda$, so it partially cancels the large positive $\delta_0$. With $\lambda = 0$, you'd get $\hat{A}_0 = 10$ (ignoring future information). With $\lambda = 1$, you'd weight all future TD errors equally (up to $\gamma$ discount), giving more variance.
</details>

**Q3**: In your iris_ma6 training, the value function uses the **global state** (observations from both drones) while the policy uses only **local observations** (one drone's view). From an optimal control perspective, why is this a good idea?

<details>
<summary>Answer</summary>

This is the **CTDE** (Centralized Training, Decentralized Execution) paradigm.

In optimal control terms: the value function is like a *centralized observer* that sees the full state. This gives it the best possible estimate of future returns, which means the advantage estimates $\hat{A}_t = G_t - V(s_t)$ have lower variance. Lower variance advantages → more stable policy gradients → better training.

The policy must be decentralized (each drone only sees its own sensors) because at deployment time, there's no central observer. But during training, we can use any information we want to improve the training signal.

It's analogous to using a high-fidelity simulation model for controller design, but deploying a controller that only uses onboard sensors.
</details>

**Q4**: If the value function $V_\phi$ is *perfect* (zero Bellman residual everywhere), what happens to GAE?

<details>
<summary>Answer</summary>

If $V_\phi = V^\pi$ exactly, then all TD errors become:

$$\delta_t = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

by definition of the advantage. So $\hat{A}_t^{\text{GAE}} = \sum_l (\gamma\lambda)^l A^\pi(s_{t+l}, a_{t+l})$.

For $\lambda = 0$: $\hat{A}_t = A^\pi(s_t, a_t)$ -- the true 1-step advantage. No bias.

For any $\lambda$: GAE becomes a (discounted) sum of true advantages over future timesteps. This is still a valid advantage estimate but includes "credit" from future actions -- not just the current action's quality.

Key insight: when $V_\phi$ is perfect, $\lambda$ doesn't matter for bias (there is none). In practice, $V_\phi$ is imperfect, and that's when $\lambda$ trades off bias vs variance.
</details>

---

**Next**: [Chapter 2 -- PPO: The Algorithm](ch2_ppo_algorithm.md)
