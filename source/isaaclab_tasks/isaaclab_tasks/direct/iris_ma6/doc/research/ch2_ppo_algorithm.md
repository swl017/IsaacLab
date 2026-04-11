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

## 2.10 Appendix: Why the Surrogate Objective Looks the Way It Does

In §2.2 we wrote:

$$L^{\text{surrogate}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]$$

and called it a "first-order approximation." This appendix explains *what* it approximates and *why* it's only first-order.

### The real objective

What we actually want to maximize is the expected return under the **new** policy:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right]$$

The expectation is over trajectories sampled by $\pi_\theta$. But we don't have data from $\pi_\theta$ -- we only have data from $\pi_{\theta_\text{old}}$. So we need to express $J(\theta)$ in terms of quantities we can measure from the old rollouts.

### The exact identity (Kakade & Langford, 2002)

There is a beautiful exact result called the **policy improvement lemma**:

$$J(\theta) - J(\theta_\text{old}) = \mathbb{E}_{s \sim \rho_\theta,\, a \sim \pi_\theta(\cdot|s)}\left[A^{\pi_{\theta_\text{old}}}(s, a)\right]$$

In words: the *improvement* from $\theta_\text{old}$ to $\theta$ equals the expected advantage (computed under the old value function) of the actions taken by the new policy, at states the new policy visits.

Two things to notice:
1. The expectation is over $s \sim \rho_\theta$ -- the **state distribution** induced by the new policy. Different policies visit different states, and this matters.
2. The action distribution is $\pi_\theta(\cdot|s)$ -- also the new policy.
3. The advantage uses the *old* value function $V^{\pi_{\theta_\text{old}}}$, which is what we have.

This is the *real* objective we'd love to optimize. The problem: $\rho_\theta$ is intractable. To compute it, we'd need to roll out the new policy in the environment -- which is exactly what we're trying to avoid. Each gradient step would require fresh rollouts, defeating the purpose of reusing the buffer for multiple epochs.

### The surrogate's approximation

The surrogate replaces $\rho_\theta$ with $\rho_{\theta_\text{old}}$ (the *old* policy's state distribution, which we can sample from -- it's the buffer we just collected):

$$L^{\text{surrogate}}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_\text{old}},\, a \sim \pi_\theta(\cdot|s)}\left[A^{\pi_{\theta_\text{old}}}(s, a)\right]$$

Then it uses **importance sampling** to express the inner expectation over $\pi_\theta$ in terms of samples from $\pi_{\theta_\text{old}}$:

$$\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[f(a)\right] = \mathbb{E}_{a \sim \pi_{\theta_\text{old}}(\cdot|s)}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} f(a)\right] = \mathbb{E}_{a \sim \pi_{\theta_\text{old}}(\cdot|s)}\left[r_t(\theta) f(a)\right]$$

(This step is *exact* -- importance sampling has no error if the support is the same.)

Plugging in $f(a) = A^{\pi_{\theta_\text{old}}}(s,a)$:

$$L^{\text{surrogate}}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_\text{old}},\, a \sim \pi_{\theta_\text{old}}(\cdot|s)}\left[r_t(\theta) \cdot A^{\pi_{\theta_\text{old}}}(s, a)\right]$$

Replacing the true advantage with our GAE estimate $\hat{A}_t$ and the joint expectation with empirical averaging over the rollout buffer:

$$L^{\text{surrogate}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]$$

That's where the formula in §2.2 comes from.

### Why "first-order"

The only approximation we made is replacing $\rho_\theta$ with $\rho_{\theta_\text{old}}$. How big is the error?

When $\theta = \theta_\text{old}$, the two distributions are identical and the error is zero. As $\theta$ drifts from $\theta_\text{old}$, $\rho_\theta$ drifts from $\rho_{\theta_\text{old}}$. The relationship is approximately:

$$\rho_\theta(s) \approx \rho_{\theta_\text{old}}(s) + \mathcal{O}(\|\theta - \theta_\text{old}\|)$$

So the error in the surrogate is:

$$L^{\text{surrogate}}(\theta) - (J(\theta) - J(\theta_\text{old})) = \mathcal{O}(\|\theta - \theta_\text{old}\|^2)$$

The surrogate is correct **to first order in $\theta - \theta_\text{old}$**. Its gradient at $\theta = \theta_\text{old}$ exactly matches the true policy gradient. The error grows quadratically as $\theta$ drifts.

### Why this matters for PPO

The first-order accuracy is *only* valid near $\theta_\text{old}$. As the policy drifts, the surrogate becomes a less and less accurate proxy for the real objective. Maximizing it without restraint would push $\theta$ to a point where the surrogate value is high but the *true* objective $J(\theta)$ is actually low (because the state distribution has shifted in unexpected ways).

This is the fundamental motivation for the trust region:

> **The surrogate is trustworthy only when the new policy is close to the old one. So constrain the policies to stay close.**

TRPO enforces this with a hard KL constraint:

$$\max_\theta L^{\text{surrogate}}(\theta) \quad \text{s.t.} \quad D_\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$$

PPO enforces it implicitly via clipping. When $|r_t - 1| > \epsilon$, the clip says "we don't trust the surrogate this far from the data; flatten the gradient." This is a soft, sample-wise version of the trust region: we restrict not the global KL, but the per-sample importance ratio.

The KL-adaptive learning rate scheduler (Chapter 4) is yet another safety mechanism: it monitors the *actual* KL drift after each update and reduces the step size if the policy is moving too fast. Together, the three mechanisms (clip, KL scheduler, gradient norm clip) keep $\theta$ in the regime where the surrogate is a faithful proxy for the real objective.

### Summary

| Object | Formula | Status |
|---|---|---|
| Real objective | $J(\theta) - J(\theta_\text{old}) = \mathbb{E}_{s \sim \rho_\theta, a \sim \pi_\theta}[A^{\pi_{\theta_\text{old}}}]$ | Exact, but intractable ($\rho_\theta$ unknown) |
| Surrogate objective | $L^{\text{surrogate}}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_\text{old}}, a \sim \pi_{\theta_\text{old}}}[r_t(\theta) \hat{A}_t]$ | Tractable; first-order accurate |
| Approximation | $\rho_\theta \to \rho_{\theta_\text{old}}$ | Error $\mathcal{O}(\|\theta - \theta_\text{old}\|^2)$ |
| Clipped surrogate | $L^{\text{CLIP}}(\theta) = \mathbb{E}_t[\min(r_t \hat{A}_t, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t)]$ | Tractable; pessimistic; trust-region-like |

The clip and the KL scheduler exist because the surrogate is a *first-order* approximation. If we had the exact objective, we could maximize it freely. Because we only have a local approximation, we must stay local.

---

## 2.11 Appendix: Two Small But Confusing Details About the Value Loss

### Question 1: What is $\mathcal{B}$ in the value loss formula?

Recall the value loss from §2.4:

$$L^V(\phi) = \frac{1}{|\mathcal{B}|} \sum_{t \in \mathcal{B}} \left(V_\phi(s_t) - V_t^{\text{target}}\right)^2$$

$\mathcal{B}$ is just a **mini-batch**: a subset of timesteps from the rollout buffer. The notation $|\mathcal{B}|$ is the mini-batch size, and $t \in \mathcal{B}$ means "for each timestep in the mini-batch." So the expression says "average squared error over the mini-batch."

**Concrete example from your config**:
- `rollouts = 32` (timesteps of data per update) × `num_envs` (say 2048 parallel environments) = 65,536 total samples in the rollout buffer
- `mini_batches = 8` → each mini-batch is 65,536 / 8 = 8,192 samples
- So $|\mathcal{B}| = 8{,}192$ and the sum runs over those 8,192 timesteps

**Why mini-batches at all?** Two reasons:
1. **Memory**: A full rollout buffer won't fit in GPU memory for the backward pass.
2. **Gradient noise as regularization**: Mini-batch SGD introduces small stochastic perturbations that help escape local optima and regularize the optimization landscape. Full-batch gradient descent is theoretically cleaner but often performs worse in practice for neural networks.

In the SKRL code (`ppo.py` line 431), `self.memory.sample_all(names=..., mini_batches=self._mini_batches)` splits the buffer into 8 mini-batches. The inner loop in `_update()` iterates over them:

```python
for (sampled_states, sampled_actions, ...) in sampled_batches:  # 8 mini-batches
    ...
    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
```

`F.mse_loss` computes exactly the $\frac{1}{|\mathcal{B}|}\sum_{t \in \mathcal{B}}(\cdot)^2$ formula -- it's the mean squared error over the mini-batch.

**Mini-batch vs epoch terminology**:
- 1 **mini-batch** = one gradient step = one `loss.backward(); optimizer.step()`
- 1 **epoch** = one full pass through all mini-batches = 8 mini-batches in your config
- 1 **update** = `learning_epochs` epochs = 6 full passes = 48 gradient steps total (8 × 6)
- 1 **iteration** / rollout cycle = collect 32 timesteps → do 1 update → repeat

So each time the `_update()` function runs, it performs 48 gradient steps, and $\mathcal{B}$ refers to the particular mini-batch each of those 48 steps operates on.

### Question 2: If clipping is PPO's core idea, why is value clipping optional?

Good question -- and the answer reveals a subtlety about what "clipping" means and where it actually matters.

**The clip on the policy is fundamental.** Without it, PPO degenerates into vanilla policy gradient, and the multi-epoch off-policy training becomes unstable. The policy clip is the *defining feature* of PPO. It cannot be turned off without changing the algorithm.

**The clip on the value function is a separate, optional heuristic.** It was added in the original OpenAI baselines implementation of PPO, but the paper itself (Schulman et al., 2017) doesn't include it. It's an *implementation detail* that the community later debated.

The two clips serve different purposes:

| Clip | Purpose | Required? |
|---|---|---|
| **Policy clip** on $r_t(\theta)$ | Keep the importance sampling ratio bounded so the first-order surrogate remains trustworthy. Prevents the off-policy gradient from diverging. | **Yes** -- this is what makes PPO, PPO. |
| **Value clip** on $V_\phi(s)$ | Prevent the value function from overshooting its previous prediction by more than $\epsilon_v$. A trust-region-like restriction on the critic. | **No** -- it's an optional stability trick. |

### Why is the policy clip essential but the value clip optional?

The reasons come from the structure of each loss:

#### 1. The value loss is already a proper supervised objective

The value loss $(V_\phi(s) - V^\text{target})^2$ is a standard regression loss. It has a well-defined optimum ($V_\phi(s) = V^\text{target}$) and a convex shape around that optimum (for a linear model; for neural networks the shape is more complex but still well-behaved). Standard SGD on a regression loss is *stable* as long as the learning rate is reasonable. There's no importance-sampling issue, no off-policy drift, no catastrophic failure mode that clipping must prevent.

In contrast, the policy surrogate has an importance sampling ratio $r_t(\theta)$ that can explode or implode multiplicatively. A 10x ratio means the gradient for that sample is 10x larger than normal. Without clipping, a single outlier sample can swing the gradient wildly. The policy clip is a numerical safety mechanism that has no analogue in the value loss.

#### 2. The value clip doesn't correspond to a trust region in the usual sense

The policy clip is a *pessimistic* bound: we take the `min` of the clipped and unclipped terms. For positive advantages, the clip *lowers* the objective (making us less optimistic about gains); for negative advantages, it *raises* the objective (making us less optimistic about losses). Either way, the clip is **conservative** -- it penalizes over-confident updates.

The value clip uses `max` of two squared errors:

$$L^V = \max\left[(V_\phi - V^\text{targ})^2,\; (V_\phi^\text{clipped} - V^\text{targ})^2\right]$$

This is also pessimistic in a sense: we take whichever error is *worse*, then minimize that. The effect is: if the unclipped prediction would improve beyond $V_{\phi_\text{old}} \pm \epsilon_v$ in a single step, the gradient is suppressed.

But this isn't solving a fundamental stability problem -- it's just preventing the critic from updating "too fast." You could achieve the same effect with a lower learning rate for the value network, or with stronger gradient norm clipping, or just by being patient. There are *many* ways to slow down the critic, and value clipping is one of them.

#### 3. Empirical evidence is mixed

Engstrom et al. (2020), "Implementation Matters in Deep Policy Gradients," ran careful ablations on PPO's "code-level" optimizations. They found:

- **Policy clip**: essential. Removing it breaks PPO.
- **Advantage normalization**: important. Removing it destabilizes training.
- **Value clip**: minor effect. Sometimes slightly helpful, sometimes slightly harmful, depending on the environment.
- **Orthogonal weight initialization**: important. Removing it hurts.

The value clip turned out to be one of the weaker contributions. It was added by the OpenAI baselines authors for symmetry with the policy clip (the thinking: "if we clip one, we should clip the other"), but the theoretical justification is much weaker.

#### 4. Your config doesn't use it

In your SKRL setup, `clip_predicted_values = False`. This means:

```python
value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
```

Just a plain MSE loss. No clipping. This works fine in practice for your drone task. If you ever see symptoms of **value function divergence** (growing value loss, wild advantage estimates), one fix would be to enable value clipping. But for stable training, it's not needed -- and adding it would slow down value learning slightly, which could hurt if the value function is struggling to keep up with a moving target (e.g., during curriculum shifts).

### Summary

| Question | Answer |
|---|---|
| What is $\mathcal{B}$? | A mini-batch -- a subset of timesteps from the rollout buffer. $|\mathcal{B}|$ is the mini-batch size. |
| Why is the policy clip essential? | It solves a fundamental stability problem: importance sampling ratios can explode, and the first-order surrogate is only accurate near $\theta_\text{old}$. |
| Why is the value clip optional? | The value loss is already a well-behaved regression objective. The clip is a heuristic speed limit, easily replaced by other stability measures (learning rate, gradient clipping). |
| Should *you* enable value clipping? | Probably not, unless you observe value function divergence. Your current `clip_predicted_values = False` is the mainstream default. |

**Key takeaway**: "PPO clips stuff" is a slogan, but only one of those clips is load-bearing. The policy clip defines the algorithm. The value clip is a bonus feature you can take or leave.

---

## 2.12 Appendix: Does a Clipped Sample Mean "Giving Up" On It?

Yes, exactly. When a sample gets clipped, PPO effectively says "I've learned enough from this one, move on." This appendix makes that intuition precise.

### What happens mechanically

With positive advantage, $r_t = 1.35$, $\epsilon = 0.2$:

$$L_t = \min(r_t \cdot \hat{A}_t,\; \text{clip}(r_t, 0.8, 1.2) \cdot \hat{A}_t) = \min(1.35 \hat{A}_t,\; 1.2 \hat{A}_t) = 1.2 \hat{A}_t$$

The selected term is $1.2 \cdot \hat{A}_t$, which is a **constant** in $\theta$ (the clip has no $\theta$ dependence). Its gradient is:

$$\frac{\partial}{\partial \theta}\big(1.2 \cdot \hat{A}_t\big) = 0$$

PyTorch autodiff computes exactly this zero. When `optimizer.step()` runs, this sample contributes nothing to the parameter update. It's as if the sample had been deleted from the mini-batch.

### The intuition: "given up" is the right mental model

PPO is literally giving up on squeezing more gradient out of this sample. Why?

**Because the sample came from $\pi_{\theta_\text{old}}$ and the current policy has already drifted 35% away from the data.** The first-order surrogate (see §2.10) is only trustworthy near $\theta_\text{old}$. At $r_t = 1.35$, we're already past the trust region. Any further update based on this sample would be extrapolating an approximation that's no longer valid.

Think of it as a **local validity certificate**:

- $r_t \in [0.8, 1.2]$: "This sample still represents what the current policy does. Use it."
- $r_t \notin [0.8, 1.2]$: "This sample is from a policy too different from the current one. Stop trusting it."

The gradient is zero not because the sample is *uninformative* -- it's because using it would be *dishonest*. The advantage estimate $\hat{A}_t$ was computed using the old value function, and its accuracy decays as the policy moves away from where the estimate was valid.

### An important subtlety: only one direction is clipped

The clip is **asymmetric per sample**. For a positive-advantage sample:

- If $r_t > 1.2$: gradient is zero (can't increase probability further)
- If $r_t < 0.8$: gradient is **not** clipped. PPO still allows this sample to *push* the probability back up.

That asymmetry is the `min` operation doing its job:

$$L_t = \min(r_t \hat{A}_t, \text{clip}(r_t, 0.8, 1.2) \hat{A}_t)$$

At $r_t = 0.5$ with $\hat{A}_t > 0$:
- $r_t \hat{A}_t = 0.5 \hat{A}_t$
- $\text{clip}(0.5, 0.8, 1.2) \hat{A}_t = 0.8 \hat{A}_t$
- $\min = 0.5 \hat{A}_t$ (unclipped wins)

Gradient is nonzero, pushing $r_t$ back up toward 1. PPO says: "The current policy has drifted *away* from a good action. Pull it back."

By symmetry, for a negative-advantage sample:
- If $r_t < 0.8$: gradient is zero (can't decrease probability further)
- If $r_t > 1.2$: gradient is **not** clipped -- PPO still lets the sample push probability down

**So "giving up" is accurate only for the direction the policy has already moved *past* the trust region.** PPO is willing to undo bad drift, but not to compound good drift beyond the clip.

### Connection to "per-sample early stopping"

There's a useful analogy: PPO's clipping acts like a **per-sample early stopping** mechanism. Each sample contributes gradient signal until its importance ratio leaves the trust region, then it retires. Over the course of 6 epochs, samples progressively drop out as the policy drifts:

| Epoch | Typical fraction of samples still contributing gradient |
|---:|---:|
| 1 | ~100% (just collected, $r_t \approx 1$) |
| 2 | ~95% |
| 3 | ~85% |
| 4 | ~70% |
| 5 | ~55% |
| 6 | ~40% |

(These numbers vary with the environment and learning rate, but the qualitative pattern is universal.)

This is why reducing `learning_epochs` from 6 to 4 (your fix in `lr_collapse_analysis.md`) loses very little signal: epochs 5-6 are already mostly clipped out. You're skipping the epochs where most samples have already "given up." It also explains why epochs 5-6 primarily contribute to **KL accumulation** rather than useful learning -- the samples still producing gradient are the ones that drifted the furthest, and their gradients push the policy even further from $\theta_\text{old}$.

### A second-order intuition: the clip as a self-regulating learning rate

Here's another way to see it. The effective learning rate for each sample is:

$$\alpha_{\text{effective},t} = \alpha \cdot \mathbb{1}[\text{sample } t \text{ is not clipped}]$$

As the policy drifts, more samples get clipped, and the effective batch size shrinks. The optimizer is effectively taking smaller and smaller steps until the rollout is exhausted. PPO has a built-in mechanism for gradually reducing its update magnitude as it extracts more learning from a fixed dataset -- without needing an explicit LR schedule (though the KL-adaptive scheduler provides an additional layer on top).

### Summary

- "Clipped sample = zero gradient" = correct.
- "Giving up on the sample" = correct.
- **Why**: the sample's advantage estimate is only valid near $\theta_\text{old}$, and the policy has drifted too far to trust it.
- **Asymmetry**: PPO gives up on *further exploitation* of a drifted sample, but still corrects *backward drift* from good actions (or *forward drift* from bad ones).
- **Emergent behavior**: samples self-retire as they become unreliable, which is why PPO can run many epochs on the same data without exploding.
- **Practical consequence**: reducing `learning_epochs` from 6 to 4 loses little signal because most samples have already clipped out by epoch 5.

---

**Next**: [Chapter 3 -- MAPPO & Your Training Dynamics](ch3_mappo_training_dynamics.md)
