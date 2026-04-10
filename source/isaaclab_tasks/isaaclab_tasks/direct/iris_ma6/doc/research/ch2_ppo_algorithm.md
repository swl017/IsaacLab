# Chapter 2: PPO -- The Algorithm

> **Reading time**: ~60 minutes.
> **Companion readings**: Sutton & Barto Ch 13.4-13.5
> **Source code**: SKRL PPO at `_isaac_sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/ppo/ppo.py`

---

## 2.1 Why Not Just Use the Policy Gradient Directly?

From Chapter 1, the policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_t\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]$$

The simplest algorithm (REINFORCE) does:

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

This works in theory, but in practice it's **catastrophically unstable** for two reasons:

### Problem 1: Step size sensitivity

In optimal control, you know that Newton's method with unit step size can diverge on non-quadratic problems. You use line search or trust regions to ensure each step improves the objective.

Policy gradient has the same problem, but *worse*. The objective $J(\theta)$ is not a simple function of $\theta$ -- it depends on the *entire trajectory distribution*, which changes when $\theta$ changes. A step that looks good based on current data might completely destroy the policy.

**Concrete example from your drone training**: At step 16k, the policy has σ=0.335 (tight distribution). The policy gradient says "increase probability of action [3.2, 0, 1.5, ...]." A small step in $\theta$ could shift the mean by 0.5 in normalized space, which at σ=0.335 means the KL divergence between old and new policy is $(0.5/0.335)^2 \approx 2.2$ per dimension. Over 7 dimensions, that's $D_{\text{KL}} \approx 15$. The new policy is almost unrecognizable from the old one.

### Problem 2: Data is from the old policy

You collect a rollout buffer of 32 timesteps using $\pi_{\theta_{\text{old}}}$. Then you do 6 epochs of gradient updates. After the first epoch, $\theta$ has changed -- but you're still using data from $\theta_{\text{old}}$. If the policy drifts too far, the data becomes misleading.

This is the **off-policy problem**: your training data came from a different policy than the one you're currently optimizing. The further the new policy drifts from the data-collection policy, the more the gradient estimates degrade.

### The trust region idea

The solution: **constrain how much the policy can change per update**. This is exactly what trust regions do in nonlinear optimization. TRPO (Trust Region Policy Optimization, 2015) formalized this:

$$\max_\theta \; \mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \hat{A}_t\right] \quad \text{subject to} \quad D_{\text{KL}}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$$

This is a constrained optimization problem. TRPO solves it with conjugate gradient + line search, which is expensive and hard to implement.

**PPO's insight** (Schulman, 2017): replace the hard constraint with a clipped objective that achieves approximately the same effect, but is trivially simple to implement with standard gradient descent.

> **Sutton & Barto**: Read Chapter 13.4-13.5. Section 13.4 covers REINFORCE. Section 13.5 introduces the baseline (which you already know as the advantage). These are the building blocks PPO improves on.

---

## 2.2 The Importance Sampling Ratio

Before understanding the clip, you need the **importance sampling ratio**. When we reuse data collected under $\pi_{\theta_\text{old}}$ to optimize $\pi_\theta$, we correct for the distribution mismatch:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

- $r_t = 1$: the new policy assigns the same probability to action $a_t$ as the old one.
- $r_t > 1$: the new policy makes $a_t$ more likely.
- $r_t < 1$: the new policy makes $a_t$ less likely.

The **surrogate objective** uses this ratio:

$$L^{\text{surrogate}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]$$

This is a first-order approximation to the expected return under $\pi_\theta$, using data from $\pi_{\theta_\text{old}}$.

**In SKRL code** (`ppo.py` lines 476-477):
```python
ratio = torch.exp(next_log_prob - sampled_log_prob)
surrogate = sampled_advantages * ratio
```

The ratio is computed in log space for numerical stability: $r_t = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_\text{old}}(a_t|s_t))$.

---

## 2.3 The Clipped Surrogate Objective

PPO clips the ratio $r_t$ to prevent the policy from changing too much. With $\epsilon = 0.2$ (your config):

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\!\Big(r_t(\theta)\,\hat{A}_t, \;\; \text{clip}(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon)\,\hat{A}_t\Big)\right]$$

This is best understood case-by-case:

### Case 1: Good action ($\hat{A}_t > 0$)

The policy gradient wants to *increase* $r_t$ (make this action more likely). But the clip limits it:

$$L_t = \min\!\big(r_t \cdot \hat{A}_t, \;\; \min(r_t, 1.2) \cdot \hat{A}_t\big)$$

If $r_t < 1.2$: both terms agree, gradient flows normally.
If $r_t \geq 1.2$: the clipped term becomes $1.2 \cdot \hat{A}_t$ (constant), so the gradient is zero. The policy stops trying to increase the probability further.

**Intuition**: "You found a good action. You can increase its probability by up to 20%, but no more in a single update."

### Case 2: Bad action ($\hat{A}_t < 0$)

The policy gradient wants to *decrease* $r_t$ (make this action less likely). The clip limits it:

$$L_t = \min\!\big(r_t \cdot \hat{A}_t, \;\; \max(r_t, 0.8) \cdot \hat{A}_t\big)$$

Note the sign flip: since $\hat{A}_t < 0$, minimizing $r_t \cdot \hat{A}_t$ means maximizing $|r_t \cdot \hat{A}_t|$, which means minimizing $r_t$.

If $r_t > 0.8$: gradient flows normally, pushing $r_t$ down.
If $r_t \leq 0.8$: clipped, gradient is zero. The policy stops decreasing the probability.

**Intuition**: "You found a bad action. You can decrease its probability by up to 20%, but no more."

### The `min` combines both clips

The `min` operation ensures the **pessimistic** bound is used. This means:
- For good actions, we take the *lower bound* on how much we can increase probability.
- For bad actions, we take the *lower bound* on how much benefit we get from decreasing probability.

The net effect: the clip acts as a **soft trust region**, preventing any single update from moving the policy too far from where the data was collected.

**In SKRL code** (`ppo.py` lines 476-482):
```python
ratio = torch.exp(next_log_prob - sampled_log_prob)
surrogate = sampled_advantages * ratio
surrogate_clipped = sampled_advantages * torch.clip(
    ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
)
policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
```

The negative sign is because PyTorch optimizers *minimize*, and we want to *maximize* the clipped objective.

---

## 2.4 Value Function Loss

The critic $V_\phi(s)$ is trained to predict the expected return. The target comes from GAE:

$$V_t^{\text{target}} = \hat{A}_t + V_{\phi_\text{old}}(s_t)$$

This is the advantage (how much better than baseline) plus the old baseline. Equivalently, it's the estimated actual return.

The loss:

$$L^V(\phi) = \frac{1}{|\mathcal{B}|} \sum_{t \in \mathcal{B}} \left(V_\phi(s_t) - V_t^{\text{target}}\right)^2$$

**Value clipping** (optional, `clip_predicted_values=True` with $\epsilon_v = 0.2$):

$$V_\phi^{\text{clipped}} = V_{\phi_\text{old}}(s_t) + \text{clip}\big(V_\phi(s_t) - V_{\phi_\text{old}}(s_t),\; -\epsilon_v,\; +\epsilon_v\big)$$

$$L^V = \max\left[(V_\phi - V^{\text{targ}})^2, \; (V_\phi^{\text{clipped}} - V^{\text{targ}})^2\right]$$

This prevents the value function from changing too much in a single update -- the same trust-region idea applied to the critic. In your config, `clip_predicted_values=False`, so you use the simpler unclipped version.

**In SKRL code** (`ppo.py` lines 485-491):
```python
predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

if self._clip_predicted_values:
    predicted_values = sampled_values + torch.clip(
        predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
    )
value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
```

---

## 2.5 Entropy Bonus

For a Gaussian policy with $d$ action dimensions and standard deviations $\sigma_1, \ldots, \sigma_d$:

$$H[\pi_\theta(\cdot|s)] = \frac{d}{2}\ln(2\pi e) + \sum_{i=1}^d \ln \sigma_i$$

The entropy is high when $\sigma$ is large (broad distribution, exploring many actions) and low when $\sigma$ is small (narrow distribution, exploiting a specific action).

The entropy bonus adds $c_e \cdot H$ to the objective (or equivalently, subtracts from the loss):

$$L^H = -c_e \cdot H[\pi_\theta]$$

With your $c_e = 0.01$, this is a gentle nudge toward exploration. The gradient with respect to $\ln\sigma_i$ is simply $-c_e = -0.01$ -- a constant push upward on the log-std. This prevents the policy from collapsing to a deterministic point (σ → 0), which would kill exploration forever.

**In SKRL code** (`ppo.py` lines 470-473):
```python
if self._entropy_loss_scale:
    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
else:
    entropy_loss = 0
```

---

## 2.6 The Combined Loss

All three terms are summed (with your coefficients):

$$L(\theta, \phi) = \underbrace{-L^{\text{CLIP}}(\theta)}_{\text{policy loss}} + \underbrace{c_v \cdot L^V(\phi)}_{\text{value loss}} + \underbrace{(-c_e \cdot H[\pi_\theta])}_{\text{entropy loss}}$$

$$= -L^{\text{CLIP}} + 1.0 \cdot L^V - 0.01 \cdot H$$

**In SKRL code** (`ppo.py` line 495):
```python
self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()
```

A single backward pass computes gradients for all three terms jointly. The policy parameters $\theta$ receive gradients from the policy loss and entropy loss. The value parameters $\phi$ receive gradients from the value loss. (If policy and value share parameters, all three affect all parameters.)

### Why joint optimization?

In your setup, the policy and value networks are separate (you have separate `policy` and `value` models), but they share a single optimizer. The combined loss means a single `optimizer.step()` updates both. This is simpler than alternating between policy and value updates, and empirically works well.

---

## 2.7 The Full Update Loop

Here's the complete flow for one PPO update, with SKRL code references:

### Step 0: Collect rollouts (happens before `_update`)

The training loop calls `act()` → `record_transition()` repeatedly for `rollouts` (32) timesteps. Each step stores $(s_t, a_t, r_t, \log\pi_{\theta_\text{old}}(a_t|s_t), V_\phi(s_t))$ in the replay buffer.

**In SKRL code** (`ppo.py` lines 342-346):
```python
self._rollout += 1
if not self._rollout % self._rollouts and timestep >= self._learning_starts:
    self.set_mode("train")
    self._update(timestep, timesteps)
    self.set_mode("eval")
```

### Step 1: Compute GAE advantages

Using the stored values and rewards, compute $\hat{A}_t$ for every timestep in the buffer.

**In SKRL code** (`ppo.py` lines 360-405, called at 407-428):

The `compute_gae` function iterates backward through the buffer:

```python
for i in reversed(range(memory_size)):
    next_values = values[i + 1] if i < memory_size - 1 else last_values
    advantage = (
        rewards[i]
        - values[i]
        + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
    )
    advantages[i] = advantage
```

This implements the recursive form:

$$\hat{A}_t = (r_t - V(s_t)) + \gamma \cdot \mathbb{1}[\text{not done}] \cdot (V(s_{t+1}) + \lambda \cdot \hat{A}_{t+1})$$

which is equivalent to $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ from Chapter 1.

Note that `not_dones` zeroes out the propagation at episode boundaries -- if the episode terminated, we don't bootstrap from the "next state" of a new episode.

After computing advantages, they are **normalized** (line 403):
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

This is critical for training stability. Without normalization, the magnitude of advantages would vary wildly between updates (early training has huge advantages, late training has small ones), making the effective step size unpredictable.

### Step 2: Multiple learning epochs

For each of the 6 learning epochs, iterate over 8 mini-batches:

**In SKRL code** (`ppo.py` lines 438-531):
```python
for epoch in range(self._learning_epochs):      # 6 epochs
    kl_divergences = []
    for (sampled_states, ...) in sampled_batches:  # 8 mini-batches
        # ... compute ratio, clip, losses ...
        self.optimizer.zero_grad()
        self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()
        # gradient clipping
        if self._grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(..., self._grad_norm_clip)
        self.scaler.step(self.optimizer)
```

Each mini-batch is a random subset of the rollout buffer. The 6 epochs mean each data point is used ~6 times. This is the "multiple epochs of SGD" that makes PPO sample-efficient.

### Step 3: Gradient clipping

**In SKRL code** (`ppo.py` lines 502-509):
```python
if self._grad_norm_clip > 0:
    self.scaler.unscale_(self.optimizer)
    nn.utils.clip_grad_norm_(
        itertools.chain(self.policy.parameters(), self.value.parameters()),
        self._grad_norm_clip
    )
```

With `grad_norm_clip = 0.3`, the total gradient norm is capped at 0.3. This prevents any single mini-batch from causing a catastrophically large parameter update. It's a safety net on top of the clipping.

### Step 4: KL-adaptive learning rate update

After each epoch, update the LR based on KL divergence:

**In SKRL code** (`ppo.py` lines 521-530):
```python
if self._learning_rate_scheduler:
    if isinstance(self.scheduler, KLAdaptiveLR):
        kl = torch.tensor(kl_divergences, device=self.device).mean()
        self.scheduler.step(kl.item())
```

Note: the KL is computed *approximately* using the ratio, not the exact Gaussian KL formula:

**In SKRL code** (`ppo.py` lines 460-463):
```python
with torch.no_grad():
    ratio = next_log_prob - sampled_log_prob
    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
```

This uses the approximation $D_{\text{KL}} \approx \mathbb{E}[(e^r - 1) - r]$ where $r = \log(\pi_\theta / \pi_{\theta_\text{old}})$. This is the second-order Taylor expansion of KL divergence: $D_{\text{KL}} \approx \frac{1}{2}\mathbb{E}[r^2]$ for small $r$, but the exact formula $(e^r - 1 - r)$ is more accurate for larger $r$.

### Step 5: Logging

**In SKRL code** (`ppo.py` lines 533-544):
```python
self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
```

These are exactly the numbers in your `ppo_mappo_algorithm.md` training logs.

---

## 2.8 Summary: PPO's Four Lines of Defense Against Catastrophic Updates

| Defense | Mechanism | Your config |
|---|---|---|
| 1. Ratio clipping | Limits probability ratio to $[0.8, 1.2]$ | $\epsilon = 0.2$ |
| 2. Gradient norm clipping | Limits total gradient magnitude | `grad_norm_clip = 0.3` |
| 3. KL-adaptive learning rate | Reduces LR when policy changes too fast | `kl_threshold = 0.02` |
| 4. Multiple mini-batches | Each update uses a fraction of the data, reducing per-step noise | `mini_batches = 8` |

Each defense addresses a different failure mode:
- Clipping prevents the *direction* of the update from being too aggressive.
- Gradient clipping prevents the *magnitude* of the update from being too large.
- KL-adaptive LR provides *global* step size control based on actual policy change.
- Mini-batches reduce *variance* of the gradient estimate.

Together, they make PPO remarkably robust -- which is why it's the default algorithm for complex control problems like your multi-agent drones.

---

## 2.9 Quizzes

**Q1**: At the start of training (step 4k), the ratio $r_t$ for a particular action is 1.35, and $\hat{A}_t = 200$ (good action). What value does the clipped surrogate objective use for this sample?

<details>
<summary>Answer</summary>

With $\epsilon = 0.2$:
- Unclipped: $r_t \cdot \hat{A}_t = 1.35 \times 200 = 270$
- Clipped: $\text{clip}(1.35, 0.8, 1.2) \cdot \hat{A}_t = 1.2 \times 200 = 240$
- $L_t = \min(270, 240) = 240$

The clip activates and caps the objective. The gradient for this sample is zero (the minimum selected the constant clipped term), preventing the policy from increasing this action's probability further.

This sample already moved the policy 35% toward this action ($r_t = 1.35$). The clip says "enough, you've already moved 20% beyond what we trust."
</details>

**Q2**: The reported "Policy loss" in your training logs is the *negative* of $L^{\text{CLIP}}$, averaged over epochs and mini-batches. At step 8k, the policy loss is $-0.0024$. What does this mean about the policy update?

<details>
<summary>Answer</summary>

Policy loss = $-L^{\text{CLIP}}$. If the loss is $-0.0024$, then $L^{\text{CLIP}} = +0.0024 > 0$.

This means the clipped surrogate objective is positive: on average, the policy is successfully increasing probability of actions with positive advantages and decreasing probability of actions with negative advantages. The policy is improving.

At step 60k, the policy loss is $+0.0028$, meaning $L^{\text{CLIP}} = -0.0028 < 0$. The policy is not improving on average -- the increasing σ (exploration) is diluting the probability of previously good actions. This is the temporary "exploration cost" before the policy discovers better strategies.
</details>

**Q3**: Why does PPO reuse each rollout for 6 epochs instead of collecting fresh data each time? What is the tradeoff?

<details>
<summary>Answer</summary>

**Sample efficiency**: Collecting rollouts is expensive (requires running the physics simulator). Reusing data 6 times extracts more learning from each rollout, reducing the total simulator time needed.

**Tradeoff**: After epoch 1, the policy has changed. Epochs 2-6 are increasingly off-policy -- the data was collected by $\pi_{\theta_\text{old}}$ but we're optimizing $\pi_\theta$, which is drifting from $\theta_\text{old}$. The ratio $r_t$ drifts away from 1.0, and the clip fires more often, providing less gradient signal.

This is exactly why the KL divergence *accumulates* across epochs (as discussed in your `lr_collapse_analysis.md`). With 6 epochs, the total KL is roughly 6x the single-epoch KL. Reducing to 4 epochs (your fix) trades some sample efficiency for less KL accumulation.

The PPO literature suggests 3-4 epochs is often sufficient. Beyond that, the policy has extracted most of the useful information from the rollout, and further epochs mainly accumulate KL without proportional benefit.
</details>

**Q4**: Suppose the advantage normalization step was removed. What would happen to training?

<details>
<summary>Answer</summary>

Without normalization, the magnitude of advantages would depend on the absolute scale of rewards.

Early training: rewards are ~1600, advantages might be ~500. The effective gradient step is proportional to $\alpha \cdot \hat{A}_t / \sigma^2$. With $\hat{A}_t \sim 500$ and $\sigma \sim 0.5$, the gradient is enormous.

Late training: rewards saturate at ~4200, advantages shrink to ~50. The effective gradient step drops 10x.

The result: erratic early training (gradients too large) followed by sluggish late training (gradients too small). The learning rate would need constant manual tuning.

Normalization to mean=0, std=1 decouples the gradient magnitude from the reward scale. A "2 standard deviation above average" action always gets the same gradient magnitude, regardless of whether rewards are 1600 or 4200.
</details>

---

**Next**: [Chapter 3 -- MAPPO & Your Training Dynamics](ch3_mappo_training_dynamics.md)
