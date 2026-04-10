# LR Collapse Analysis: Why Wider Action Space Breaks the KL-Adaptive Scheduler

> **Failed run**: `2026-04-09_19-06-29_a9_bbox_size_baseline_3461991973_seed42`
> **Reference run**: `928b9585f2` (min_lr_tuning, same penalties, narrower action space)
> **Fix applied**: `min_lr=3e-4`, `learning_epochs=4`

---

## 1. The KL-Adaptive Learning Rate Scheduler

SKRL's `KLAdaptiveLR` adjusts the learning rate after each update based on the KL divergence between the old and new policy:

$$D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) = \mathbb{E}_{s \sim \mathcal{D}}\left[\mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}}\left[\log \frac{\pi_{\theta_{\text{old}}}(a|s)}{\pi_\theta(a|s)}\right]\right]$$

The scheduler rule (with `kl_threshold` $\tau = 0.02$):

$$\alpha_{k+1} = \begin{cases} \alpha_k / 1.5 & \text{if } D_{\text{KL}} > 2\tau = 0.04 \\[4pt] \alpha_k \times 1.5 & \text{if } D_{\text{KL}} < \tau / 2 = 0.01 \\[4pt] \alpha_k & \text{otherwise (dead zone: } [0.01, 0.04]\text{)} \end{cases}$$

subject to $\alpha_k \geq \alpha_{\min}$.

The dead zone $[\tau/2, 2\tau]$ is meant to absorb normal training noise. The problem is what happens when $D_{\text{KL}}$ consistently exceeds $2\tau$.

---

## 2. KL Divergence for Gaussian Policies

Our MAPPO policy outputs a diagonal Gaussian:

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \, \text{diag}(\sigma^2))$$

where $\sigma$ is a state-independent learnable parameter (log_std). For two diagonal Gaussians with shared covariance (the typical case within one update), the KL divergence decomposes over action dimensions:

$$D_{\text{KL}} = \frac{1}{2} \sum_{i=1}^{d} \frac{(\mu_i^{\text{new}} - \mu_i^{\text{old}})^2}{\sigma_i^2}$$

(The $\log(\sigma/\sigma') + (\sigma'^2 - \sigma^2)/(2\sigma^2)$ terms are near-zero within one update since $\sigma$ changes slowly.)

**Key insight**: KL scales as $(\Delta\mu)^2 / \sigma^2$. The mean shift per gradient step depends on the learning rate and the reward gradient.

---

## 3. How Action Scaling Amplifies KL

### The action-to-physics mapping

The policy outputs normalized actions $a \in [-1, 1]^7$. These are scaled to physical commands:

$$u_i = a_i \times s_i$$

where $s_i$ is the physical scale for dimension $i$. The key differences:

| Dimension | 928b scale $s_i$ | 3461 scale $s_i$ | Ratio |
|-----------|------------------:|------------------:|------:|
| vx, vy, vz | 10 m/s | 10 m/s | 1.0x |
| yaw_rate | 0.785 rad/s (45 deg/s) | 1.571 rad/s (90 deg/s) | **2.0x** |
| gimbal_yaw | 3.14 rad/s (180 deg/s) | 6.28 rad/s (360 deg/s) | **2.0x** |
| gimbal_pitch | 3.14 rad/s (180 deg/s) | 6.28 rad/s (360 deg/s) | **2.0x** |
| zoom_rate | 1.0 /s | 4.0 /s | **4.0x** |

### The reward sensitivity chain

A small policy mean shift $\Delta\mu_i$ in normalized space produces a physical command change $\Delta u_i = \Delta\mu_i \times s_i$, which causes a state change $\Delta x \propto \Delta u_i$, which produces a reward change:

$$\Delta r \propto \frac{\partial r}{\partial x} \cdot \frac{\partial x}{\partial u_i} \cdot s_i \cdot \Delta\mu_i$$

The reward gradient w.r.t. the policy mean is therefore:

$$\frac{\partial r}{\partial \mu_i} \propto s_i \cdot \frac{\partial r}{\partial x} \cdot \frac{\partial x}{\partial u_i}$$

**Larger $s_i$ means steeper reward gradients in policy space.** A unit change in $\mu_i$ produces a 2-4x larger physical effect, hence 2-4x larger reward change, hence 2-4x larger advantage signal $\hat{A}_t$.

### The gradient → KL → LR chain

One PPO gradient step on mini-batch $\mathcal{B}$ updates the mean by:

$$\Delta\mu \propto \alpha \cdot \frac{1}{|\mathcal{B}|} \sum_{t \in \mathcal{B}} \frac{\hat{A}_t}{\sigma^2} \cdot \nabla_\theta \mu_\theta(s_t)$$

The resulting KL (from Section 2):

$$D_{\text{KL}} \propto \sum_i \frac{(\Delta\mu_i)^2}{\sigma_i^2} \propto \alpha^2 \cdot \sum_i \frac{\hat{A}_{t,i}^2}{\sigma_i^4}$$

Since $\hat{A}_t$ is amplified by the action scales (wider action space → larger reward changes → larger advantages), we get:

$$D_{\text{KL}}^{3461} \approx \left(\frac{\bar{s}^{3461}}{\bar{s}^{928b}}\right)^2 \cdot D_{\text{KL}}^{928b}$$

where $\bar{s}$ is the effective (weighted) action scale. With 4 of 7 dimensions having 2-4x larger scales, the effective amplification is roughly **2-3x in KL**, which pushes it from the dead zone ($D_{\text{KL}} \approx 0.015$ in 928b) well above $2\tau = 0.04$.

---

## 4. The Collapse Cascade

At curriculum onset (step 20k), the reward distribution shifts (new difficulty → new advantage statistics). In 928b this created moderate KL (within the dead zone). In 3461:

**Step 1**: Curriculum onset at 20k creates a reward distribution shift.

**Step 2**: The wider action scales amplify this shift into larger per-update $\hat{A}_t$.

**Step 3**: Each of the 6 learning epochs accumulates KL:

$$D_{\text{KL}}^{\text{total}} \approx \sum_{e=1}^{6} D_{\text{KL}}^{(e)}$$

After epoch 6, the total KL exceeds $2\tau = 0.04$.

**Step 4**: Scheduler divides LR by 1.5: $\alpha \leftarrow \alpha / 1.5$.

**Step 5**: Next update: the curriculum shift is still present (it's a 20k-step ramp, not a one-time event). The reward distribution is non-stationary. KL again exceeds threshold.

**Step 6**: Repeated divisions: $\alpha \leftarrow \alpha / 1.5^n$. After just 5 consecutive over-threshold updates:

$$\alpha = 0.001 \times (1/1.5)^5 = 0.001 \times 0.132 = 0.000132 \approx \alpha_{\min}$$

This explains the observed 0.001 → 0.000119 drop between step 16k and 20k — approximately 5 consecutive scheduler cuts.

**Step 7**: At $\alpha_{\min} = 10^{-4}$, the policy gradient is too small to adapt to increasing difficulty. pair_valid_rate degrades monotonically.

---

## 5. Why the Fixes Work

### Fix 1: `min_lr = 3e-4` (raise the floor)

The floor determines the minimum adaptation rate. With $\alpha_{\min} = 3 \times 10^{-4}$:

$$\Delta\mu_{\min} \propto 3 \times 10^{-4} \cdot \hat{A}_t / \sigma^2$$

This is 3x larger than with $\alpha_{\min} = 10^{-4}$. The policy can still adapt to curriculum difficulty even after a KL spike floors the scheduler. The scheduler will **try** to raise LR when KL drops below $\tau/2$, so the floor just prevents permanent freeze — it doesn't prevent the scheduler from working normally above the floor.

928b's LR oscillated in the range $[0.0002, 0.001]$ during active training. A floor of $3 \times 10^{-4}$ sits within this natural range.

### Fix 2: `learning_epochs = 4` (reduce cumulative KL)

The KL accumulates across epochs. With 6 epochs, the policy drifts far from the collection policy. Reducing to 4:

$$D_{\text{KL}}^{\text{total}}(4\text{ epochs}) \approx \frac{4}{6} \cdot D_{\text{KL}}^{\text{total}}(6\text{ epochs}) = 0.67 \cdot D_{\text{KL}}^{\text{total}}$$

This 33% reduction can bring the KL from above $2\tau$ back into the dead zone, preventing the cascade entirely. PPO's sample efficiency typically plateaus after 3-4 epochs (further epochs over-fit to the rollout buffer), so the cost is minimal.

### Combined effect

With both fixes, the system has two lines of defense:
1. **Reduced KL per update** (fewer epochs) → scheduler stays in dead zone more often
2. **Higher floor** → even when the scheduler does cut, the policy retains sufficient learning capacity

---

## 6. Numerical Verification from Training Logs

### 928b (working) at step 20k:
- LR: 0.001575 (healthy, in normal oscillation range)
- Policy std: 0.644
- Reward: 3225
- pair_valid_rate: 0.953

### 3461 (failed) at step 20k:
- LR: **0.000119** (near floor after cascade)
- Policy std: 0.702 (slightly higher — more exploration, but useless without LR)
- Reward: 3084 (still OK — damage hasn't propagated yet)
- pair_valid_rate: 0.965 (still OK)

The LR is the **leading indicator**. It collapses before any task metric degrades. By step 28k, the downstream effects appear: pair_valid drops (0.96→0.91), tracking_lost rises (0.003→0.023). By step 60k, the cascade is complete: pair_valid=0.34, tracking_lost=0.46.

This confirms the fix should target the LR mechanism, not the curriculum or reward structure.

---

## 7. Risk Assessment

| Scenario | Probability | Mitigation |
|----------|:-----------:|------------|
| min_lr=3e-4 is too high → noisy late training | Low | 928b oscillated 0.0002-0.001; 3e-4 is conservative |
| 4 epochs insufficient for sample efficiency | Low | PPO literature shows diminishing returns past 3-4 epochs |
| KL still exceeds threshold despite both fixes | Medium | Next lever: `kl_threshold` 0.02→0.03 |
| Sigma divergence from wider dead zone | Low | Only triggered if kl_threshold is raised; not applicable with current fix |