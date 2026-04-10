# Chapter 4: The KL-Adaptive LR Scheduler & Collapse Analysis

> **Reading time**: ~40 minutes.
> **Prerequisites**: Chapters 1-3
> **Your analysis**: This chapter is a companion to `lr_collapse_analysis.md` -- read them together
> **Source code**: `_isaac_sim/.../skrl/resources/schedulers/torch/kl_adaptive.py`

---

## 4.1 KL Divergence for Gaussian Policies: Full Derivation

Your policies are diagonal Gaussians:

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$$

where $d=7$ (your action dimensions) and $\sigma_i = \exp(\text{log\_std}_i)$ are state-independent learned parameters.

### The general KL formula for Gaussians

For two $d$-dimensional diagonal Gaussians $p = \mathcal{N}(\mu_p, \Sigma_p)$ and $q = \mathcal{N}(\mu_q, \Sigma_q)$ where $\Sigma = \text{diag}(\sigma^2_1, \ldots, \sigma^2_d)$:

$$D_{\text{KL}}(p \| q) = \frac{1}{2}\left[\sum_{i=1}^d \left(\frac{\sigma_{p,i}^2}{\sigma_{q,i}^2} + \frac{(\mu_{q,i} - \mu_{p,i})^2}{\sigma_{q,i}^2} - 1 + \ln\frac{\sigma_{q,i}^2}{\sigma_{p,i}^2}\right)\right]$$

### Derivation sketch

KL divergence is:

$$D_{\text{KL}}(p \| q) = \mathbb{E}_{x \sim p}\left[\ln \frac{p(x)}{q(x)}\right]$$

For Gaussians, $\ln p(x) = -\frac{1}{2}\sum_i \left[\ln(2\pi\sigma_{p,i}^2) + \frac{(x_i - \mu_{p,i})^2}{\sigma_{p,i}^2}\right]$.

Substituting and taking the expectation (using $\mathbb{E}_{x \sim p}[(x_i - \mu_{p,i})^2] = \sigma_{p,i}^2$ and $\mathbb{E}_{x \sim p}[(x_i - \mu_{q,i})^2] = \sigma_{p,i}^2 + (\mu_{p,i} - \mu_{q,i})^2$), you get the formula above.

### Within one PPO update

Let $p = \pi_{\theta_\text{old}}$ and $q = \pi_\theta$. Within a single update:
- The log-std parameters $\sigma$ change slowly (they're shared across all states)
- The mean $\mu_\theta(s)$ changes faster (it's a function of the neural network)

So $\sigma_p \approx \sigma_q$, and the KL simplifies to:

$$D_{\text{KL}} \approx \frac{1}{2}\sum_{i=1}^d \frac{(\mu_{\theta,i}(s) - \mu_{\theta_\text{old},i}(s))^2}{\sigma_i^2}$$

The variance ratio and log terms approximately cancel. What remains is the **squared mean shift, divided by variance, summed over dimensions**.

### Key insight

$$D_{\text{KL}} \propto \frac{(\Delta\mu)^2}{\sigma^2}$$

This has two critical consequences:

1. **Small σ amplifies KL**: If σ=0.335 (Phase 1), a mean shift of 0.1 gives $\frac{(0.1)^2}{(0.335)^2} = 0.089$ per dimension. Over 7 dimensions: $D_{\text{KL}} \approx 0.62$. That's 30x above the threshold of 0.02!

2. **Large σ absorbs KL**: If σ=1.21 (Phase 3), the same mean shift gives $\frac{(0.1)^2}{(1.21)^2} = 0.0068$ per dimension. Over 7 dimensions: $D_{\text{KL}} \approx 0.048$. Just barely above threshold.

This is why the learning rate is low during Phase 1 (small σ → high KL → scheduler cuts) and high during Phase 3 (large σ → low KL → scheduler raises).

---

## 4.2 The KL-Adaptive Scheduler: Complete Mechanics

Here is the full scheduler from SKRL (`kl_adaptive.py` lines 71-99):

```python
def step(self, kl=None):
    if kl is not None:
        for group in self.optimizer.param_groups:
            if kl > self.kl_threshold * self._kl_factor:      # kl > 2τ
                group["lr"] = max(group["lr"] / self._lr_factor, self.min_lr)  # divide by 1.5
            elif kl < self.kl_threshold / self._kl_factor:    # kl < τ/2
                group["lr"] = min(group["lr"] * self._lr_factor, self.max_lr)  # multiply by 1.5
            # else: dead zone, no change
```

With your config (`kl_threshold = 0.02`, `kl_factor = 2`, `lr_factor = 1.5`):

$$\alpha_{k+1} = \begin{cases}
\alpha_k / 1.5 & \text{if } D_{\text{KL}} > 0.04 \quad \text{(too much change)} \\
\alpha_k \times 1.5 & \text{if } D_{\text{KL}} < 0.01 \quad \text{(too little change)} \\
\alpha_k & \text{if } 0.01 \leq D_{\text{KL}} \leq 0.04 \quad \text{(dead zone)}
\end{cases}$$

Subject to: $\alpha_{\min} \leq \alpha_k \leq \alpha_{\max}$

### The dead zone

The dead zone $[0.01, 0.04]$ is important. It's a region of "acceptable" KL values where the scheduler does nothing. Without it, the scheduler would oscillate: cut LR → KL drops → raise LR → KL rises → cut again.

In your successful 928b run, KL typically sits in the dead zone (~0.015), and the LR oscillates gently between 0.0002 and 0.001. The scheduler acts as a thermostat.

### The asymmetry

Cutting is more aggressive than raising:
- One cut: $\alpha \rightarrow \alpha / 1.5 = 0.67\alpha$ (33% reduction)
- One raise: $\alpha \rightarrow \alpha \times 1.5$ (50% increase)

But cuts happen when things are *going wrong* (KL too high, policy changing too fast). Raises happen when things are *calm* (KL too low, policy barely changing). In turbulent situations, there are many more cuts than raises, creating a strong downward bias.

### When is the scheduler called?

**In SKRL code** (`ppo.py` lines 521-530, `mappo.py` lines 573-583):

The scheduler is called **once per epoch**, not once per mini-batch. With 6 learning epochs, the scheduler adjusts LR 6 times per update. But the KL it receives is the *average across mini-batches within that epoch*.

Each epoch processes all mini-batches, accumulating KL. If the first 3 mini-batches each add KL ≈ 0.015, the epoch-average might be 0.015. But after 6 epochs, the *cumulative* policy drift from $\theta_\text{old}$ could be much larger. The scheduler only sees the per-epoch snapshot, not the cumulative drift.

---

## 4.3 How Action Scaling Amplifies KL

This is the core of your `lr_collapse_analysis.md`. Let me re-derive it from first principles.

### The chain from action scale to KL

Your policy outputs normalized actions $a \in [-1, 1]^7$. These are scaled to physical commands:

| Dimension | Description | Scale $s_i$ (failed run) |
|---|---|---|
| 0-2 | vx, vy, vz | 10 m/s |
| 3 | yaw_rate | 1.571 rad/s |
| 4 | gimbal_yaw | 6.28 rad/s |
| 5 | gimbal_pitch | 6.28 rad/s |
| 6 | zoom_rate | 4.0 /s |

The physical command is $u_i = a_i \times s_i$. A small change in the policy mean $\Delta\mu_i$ in normalized space produces:
- Physical change: $\Delta u_i = \Delta\mu_i \times s_i$
- State change: $\Delta x \propto \Delta u_i$ (from dynamics)
- Reward change: $\Delta r \propto \frac{\partial r}{\partial x} \cdot \Delta x$

The advantage signal $\hat{A}_t$ is the sum of reward changes. Larger action scales $s_i$ mean each unit change in $\mu_i$ produces a larger reward change, hence a larger advantage.

### From larger advantages to larger KL

The policy gradient update is:

$$\Delta\mu \propto \alpha \cdot \frac{\hat{A}_t}{\sigma^2}$$

(The $1/\sigma^2$ comes from $\nabla_\mu \log\pi = (a - \mu)/\sigma^2$ for a Gaussian.)

The resulting KL is:

$$D_{\text{KL}} \propto \frac{(\Delta\mu)^2}{\sigma^2} \propto \frac{\alpha^2 \cdot \hat{A}_t^2}{\sigma^6}$$

Since $\hat{A}_t$ scales with the action scales $s_i$, doubling the action scale roughly doubles the advantages, which **quadruples** the KL:

$$D_{\text{KL}}^{\text{wide}} \approx \left(\frac{s^{\text{wide}}}{s^{\text{narrow}}}\right)^2 \cdot D_{\text{KL}}^{\text{narrow}}$$

With 4 of 7 dimensions having 2-4x larger scales, the effective KL amplification is roughly **2-3x**, pushing it from the dead zone (~0.015) well above 2τ = 0.04.

---

## 4.4 The Collapse Cascade: A Positive Feedback Loop

Here is the step-by-step cascade for your failed `3461` run:

### Trigger: Curriculum onset at step 20k

The curriculum increases difficulty. The reward distribution shifts. In the narrow-action-space run (928b), this produces a moderate KL increase that stays within the dead zone.

### Step 1: Amplified KL (first epoch)

In the wider-action-space run (3461), the same curriculum shift produces 2-3x larger advantages (because the actions have more physical effect). The per-mini-batch KL is $\sim 0.02$, right at the threshold edge.

### Step 2: Accumulated KL (across 6 epochs)

The policy drifts further from $\theta_\text{old}$ with each epoch. After 6 epochs:

$$D_{\text{KL}}^{\text{total}} \approx \sum_{e=1}^{6} D_{\text{KL}}^{(e)} \approx 6 \times 0.02 = 0.12$$

The scheduler sees the epoch-average KL. Even if individual epoch-averages are $\sim 0.02$ (in the dead zone), the cumulative drift means the later epochs have *measured* KL $> 0.04$, triggering a cut.

### Step 3: First LR cut

$$\alpha: 0.001 \rightarrow 0.001/1.5 = 0.000667$$

### Step 4: Next update -- curriculum shift is ongoing

The curriculum doesn't shift instantaneously. It's a ramp over thousands of steps. The reward distribution is still changing. Even with the reduced LR, the advantages from the distribution shift are still large. KL again exceeds 0.04.

### Step 5: Cascading cuts

Each update triggers another cut. After $n$ consecutive cuts:

$$\alpha = 0.001 \times (1/1.5)^n$$

| Cuts ($n$) | LR | Time to reach |
|---:|---:|---|
| 0 | 0.001000 | Step ~16k |
| 1 | 0.000667 | |
| 2 | 0.000444 | |
| 3 | 0.000296 | |
| 4 | 0.000198 | |
| 5 | 0.000132 | Step ~20k |

**Five consecutive cuts reduce the LR by 87% in just ~4k steps.** This matches the observed drop from 0.001 to 0.000119 between steps 16k and 20k.

### Step 6: Floor reached

At $\alpha_{\min} = 10^{-4}$, the policy gradient is:

$$\Delta\theta = -10^{-4} \cdot \nabla L$$

With `grad_norm_clip = 0.3`, the maximum parameter change per step is $3 \times 10^{-5}$. The policy is effectively frozen.

### Step 7: Attempted recovery fails

When LR is at the floor, the policy barely changes, so KL ≈ 0. The scheduler raises LR: $10^{-4} \times 1.5 = 1.5 \times 10^{-4}$. But one raise is not enough. You need $\log_{1.5}(10) \approx 5.7$ consecutive raises to get back to $10^{-3}$. As soon as LR is high enough for the policy to change meaningfully, KL spikes again, and the scheduler cuts back down. The system oscillates at the floor.

### Why this is a positive feedback loop

```
Curriculum shift
    → Large advantages
    → Large Δμ per update
    → Large KL
    → LR cut
    → Can't adapt to curriculum
    → Policy becomes increasingly wrong for current difficulty
    → Even larger advantages when LR briefly recovers
    → Even larger KL
    → More LR cuts
    → Permanent floor
```

The fundamental issue: the scheduler treats all KL increases the same. It can't distinguish between "the policy is changing too aggressively" (bad KL -- should cut) and "the environment changed, and the policy needs to adapt" (good KL -- should not cut).

---

## 4.5 The Fixes and Why They Work

### Fix 1: `min_lr = 3e-4` (raise the floor)

The current floor is $10^{-4}$. Raising it to $3 \times 10^{-4}$:

**Why 3e-4?** In the successful 928b run, the LR oscillated between 0.0002 and 0.001. A floor of 3e-4 sits comfortably in this range. The policy at $3 \times 10^{-4}$ can still make meaningful updates:

$$\Delta\theta_{\text{min}} = -3 \times 10^{-4} \cdot \nabla L$$

That's 3x more than the old floor. The gradient step is proportional to $\alpha \cdot \hat{A}_t / \sigma^2$, so the policy can adapt 3x faster to curriculum changes.

**Won't this cause instability?** No, because the clipping mechanism (ε=0.2) and gradient norm clipping (0.3) still bound the per-step change. The LR floor only matters when the scheduler has reduced LR below the floor. The normal operating range ($\alpha \approx 0.001$) is unaffected.

### Fix 2: `learning_epochs = 4` (reduce accumulated KL)

Reducing from 6 to 4 epochs cuts the cumulative KL by ~33%:

$$D_{\text{KL}}^{\text{4 epochs}} \approx \frac{4}{6} \cdot D_{\text{KL}}^{\text{6 epochs}} = 0.67 \cdot D_{\text{KL}}^{\text{6 epochs}}$$

If the 6-epoch KL was 0.05 (above threshold), the 4-epoch KL might be 0.033 (in the dead zone). This prevents the cascade from starting.

**Won't this hurt sample efficiency?** PPO literature (Schulman 2017, Andrychowicz 2020) shows diminishing returns past 3-4 epochs. Epochs 5-6 contribute little new learning because:
1. The clip fires more often (the ratio $r_t$ has drifted far from 1.0)
2. The advantages are stale (computed from $V_{\phi_\text{old}}$ which is now several updates old)
3. The gradient signal is dominated by already-corrected actions

### Combined effect

With both fixes:
1. **Prevention**: 4 epochs reduces the chance of exceeding the KL threshold
2. **Recovery**: Even if KL does spike, the floor at 3e-4 keeps the policy viable

This is defense in depth -- two independent mechanisms that each reduce the probability of collapse.

---

## 4.6 Designing Your Own KL Schedule: A Framework

Now that you understand the machinery, here's how to think about tuning these parameters for future experiments.

### The key inequality

For the scheduler to stay in the dead zone:

$$\frac{\tau}{2} \leq \underbrace{\frac{\alpha^2}{2\sigma^2} \sum_i \left(\frac{\hat{A}_t}{\sigma_i}\right)^2}_{D_{\text{KL}} \text{ per update}} \leq 2\tau$$

The quantities you can control:
- $\alpha$: learning rate (set by scheduler, bounded by min/max)
- $\tau$: KL threshold (config parameter)
- Number of epochs (affects cumulative KL)
- $\sigma_{\min}, \sigma_{\max}$: policy std bounds

The quantities determined by the environment:
- $\hat{A}_t$: advantage magnitude (depends on reward structure and value function accuracy)
- $\sigma$: policy std (evolves during training)

### Rules of thumb

| If you observe... | Try... | Why |
|---|---|---|
| LR crashing to floor early | Raise `min_lr`, reduce `learning_epochs` | Prevent cascade |
| LR stuck at max, policy oscillating | Lower `max_lr`, increase `kl_threshold` | Slow down updates |
| KL always in dead zone, LR never adjusts | Narrow the dead zone: reduce `kl_factor` from 2 to 1.5 | Make scheduler more responsive |
| Training stable but slow | Raise `learning_rate`, add more `learning_epochs` | More aggressive updates while stable |
| Curriculum shifts cause instability | Raise `min_lr`, reduce `learning_epochs`, consider warmup | Protect against distribution shifts |

### When to use a different scheduler entirely

The KL-adaptive scheduler works well for stationary environments. For non-stationary environments (like yours with curriculum learning), consider:

1. **Linear decay**: $\alpha_t = \alpha_0 \cdot (1 - t/T)$. Simple, predictable, no collapse cascade. But doesn't adapt to training dynamics.

2. **Cosine annealing**: $\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\pi t / T))$. Smooth decay with periodic warm restarts. Good for curriculum shifts (the restarts naturally coincide with difficulty increases).

3. **KL-adaptive with reset**: The current scheduler + a mechanism that resets LR to the initial value when a curriculum stage change is detected. This combines adaptivity with robustness to distribution shifts.

---

## 4.7 Quiz

**Q1**: You have `kl_threshold = 0.02`, `kl_factor = 2`, `lr_factor = 1.5`, `min_lr = 1e-4`, `max_lr = 1e-2`. Starting from LR = 0.001, the measured KL values for 5 consecutive epochs are: [0.05, 0.03, 0.008, 0.005, 0.009]. What is the LR after each epoch?

<details>
<summary>Answer</summary>

Dead zone: [0.01, 0.04].

| Epoch | KL | Action | LR after |
|---:|---:|---|---:|
| 1 | 0.05 | > 0.04 → cut | 0.001 / 1.5 = 0.000667 |
| 2 | 0.03 | In dead zone → hold | 0.000667 |
| 3 | 0.008 | < 0.01 → raise | 0.000667 × 1.5 = 0.001000 |
| 4 | 0.005 | < 0.01 → raise | 0.001000 × 1.5 = 0.001500 |
| 5 | 0.009 | < 0.01 → raise | 0.001500 × 1.5 = 0.002250 |

After the initial cut from the high-KL epoch, the policy stabilizes (KL drops into and below the dead zone). The scheduler recovers LR over 3 consecutive raises. This is the **healthy** scenario where the scheduler self-corrects.

Compare this to the collapse scenario: if all 5 epochs had KL > 0.04, the LR would be $0.001 / 1.5^5 = 0.000132$. That's the cascade.
</details>

**Q2**: You're designing a new experiment with even wider action scales (8x for gimbal). You expect KL to be ~4x higher than the current setup. Without changing the scheduler, what will happen? Name three different fixes you could apply, and rank them by how "surgical" they are (most targeted fix first).

<details>
<summary>Answer</summary>

**What will happen**: The 4x KL amplification means KL will consistently exceed 2τ = 0.04. The scheduler will cascade LR to the floor almost immediately. The policy will be frozen from the start. Training will fail.

**Three fixes, most surgical first**:

1. **Raise `kl_threshold` to 0.08** (4x higher). Most surgical because it directly compensates for the 4x KL amplification. The dead zone shifts to [0.04, 0.16], exactly covering the new KL range. No other hyperparameters need to change.

2. **Raise `min_lr` to 1e-3 and reduce `learning_epochs` to 3**. This is less targeted -- it doesn't prevent the cascade, but ensures the policy can still learn at the floor. It changes the training dynamics (fewer epochs = less sample efficiency), so it has side effects.

3. **Normalize advantages per action dimension** (code change). Instead of a single advantage $\hat{A}_t$, compute per-dimension advantages that account for the action scale. This removes the root cause (action scaling affecting KL) rather than compensating for its effects. Most robust long-term, but requires code changes, not just config changes.
</details>

**Q3**: The SKRL code computes approximate KL as `((torch.exp(ratio) - 1) - ratio).mean()` where `ratio = next_log_prob - sampled_log_prob`. Show that for small `ratio` (small policy change), this approximates $\frac{1}{2}r^2$ (half the squared log-ratio), which is the standard KL approximation.

<details>
<summary>Answer</summary>

Let $r = \log(\pi_\theta / \pi_{\theta_\text{old}}) = \text{next\_log\_prob} - \text{sampled\_log\_prob}$.

The SKRL formula is $f(r) = e^r - 1 - r$.

Taylor expand $e^r$ around $r = 0$:

$$e^r = 1 + r + \frac{r^2}{2} + \frac{r^3}{6} + \ldots$$

$$f(r) = (1 + r + \frac{r^2}{2} + \ldots) - 1 - r = \frac{r^2}{2} + \frac{r^3}{6} + \ldots$$

For small $r$: $f(r) \approx \frac{r^2}{2}$.

This is the **second-order approximation** to KL divergence: $D_{\text{KL}}(p \| q) \approx \frac{1}{2}\mathbb{E}[(\log(p/q))^2]$.

The exact formula $e^r - 1 - r$ is always non-negative (it's the Bregman divergence of $e^x$), so it's a valid KL approximation even for large $r$, unlike the $r^2/2$ approximation which can underestimate.

This is numerically more stable than computing the exact Gaussian KL, because it works for any policy distribution (not just Gaussian) and doesn't require access to the distribution parameters -- only the log-probabilities.
</details>

---

## 4.8 Connections Back to Your System

You now have the full picture. Let's map the abstract concepts back to concrete numbers from your training:

| Concept | In your system |
|---|---|
| Policy σ | Starts at 0.559, drops to 0.335, rebounds to 1.21, stabilizes |
| KL divergence | $\propto (\Delta\mu)^2 / \sigma^2$; small σ → high KL → LR cuts |
| Dead zone [0.01, 0.04] | During Phase 2-3, KL sits here; scheduler barely moves |
| Collapse cascade | 5 consecutive cuts: 0.001 → 0.000132 in 4k steps |
| Action scale amplification | 2-4x wider scales → 2-3x higher KL → cascade triggers |
| Fix: min_lr = 3e-4 | Floor within the natural LR oscillation range of healthy training |
| Fix: learning_epochs = 4 | 33% less KL accumulation; keeps KL in dead zone |

When you re-read `lr_collapse_analysis.md` now, every equation should connect to something you derived or understood in these four chapters. The analysis is not a collection of facts -- it's a specific instance of the general PPO/MAPPO dynamics you now understand.

---

## 4.9 Further Reading

If you want to go deeper:

### Papers
1. **Schulman et al., 2017** - "Proximal Policy Optimization Algorithms" (the PPO paper)
2. **Schulman et al., 2016** - "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (the GAE paper)
3. **Yu et al., 2022** - "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (the MAPPO paper)
4. **Engstrom et al., 2020** - "Implementation Matters in Deep Policy Gradients" (why code details like advantage normalization matter enormously)
5. **Andrychowicz et al., 2020** - "What Matters In On-Policy Reinforcement Learning?" (systematic hyperparameter study; directly relevant to your tuning decisions)

### Sutton & Barto (in your research folder)
- **Ch 3**: MDP foundations (already familiar from optimal control)
- **Ch 6.1-6.3**: TD learning
- **Ch 13.1-13.5**: Policy gradient methods, REINFORCE, baselines
- **Ch 12**: Eligibility traces (the theoretical foundation for GAE's λ parameter)

### SKRL source files for reference
- PPO agent: `_isaac_sim/.../skrl/agents/torch/ppo/ppo.py` (544 lines)
- MAPPO agent: `_isaac_sim/.../skrl/multi_agents/torch/mappo/mappo.py` (606 lines)
- KL-adaptive scheduler: `_isaac_sim/.../skrl/resources/schedulers/torch/kl_adaptive.py` (100 lines)

---

**You've completed the four chapters.** Go back and re-read your `ppo_mappo_algorithm.md` and `lr_collapse_analysis.md` -- they should now read like specific case studies of the general theory, not opaque technical documents.
