# CBF Safety Filter Module Specification

**Project:** `iris_ma6` — Multi-Drone Active Triangulation  
**Module:** `cbf_safety_filter`  
**Author:** Seungwook  
**Status:** Draft v2  
**Last Updated:** 2026-03-10

---

## 1. Motivation and Design Philosophy

### 1.1 The Problem

Aerospace-domain stakeholders are reluctant to ship learning-based controllers that directly command low-level actuators. The IRIS architecture addresses this by having the MAPPO-RNN policy output **velocity commands** to PX4, which handles all lower-level flight control. This creates a natural abstraction boundary for inserting a safety layer.

The system operates under realistic constraints: communication delays (0, 2, 4 timestep observation delay), velocity measurement noise at deployment, and high-speed flight where analytical safety margins interact poorly with information staleness.

### 1.2 Design Philosophy: Separation of Concerns

The core insight, developed through analysis of how CBF formulations degrade under comm delay, is that **training-time safety shaping and deployment-time safety filtering serve fundamentally different roles and should use different mechanisms.**

- **Training-time:** The simulator has ground-truth access. Use a velocity-aware, minimally conservative barrier to give the policy the best possible gradient signal about collision geometry.
- **Deployment-time:** Only delayed, noisy observations are available. Use a simple, robust barrier with inflated margins that tolerates information staleness.

The MAPPO-RNN policy, trained with delays in the loop and GT-based safety rewards, becomes the **primary** collision avoidance mechanism — its RNN hidden state learns implicit state estimation and delay compensation. The deployment CBF filter is a **safety net**, not the main controller.

### 1.3 Dec-POMDP Framing

The system is a Dec-POMDP. Rewards are functions of the **true state** $s^t$, not the observation $o_i^{t-\tau}$. The agent's job is to maximize rewards it cannot directly observe, using delayed observations as input. This is the standard, correct POMDP formulation.

| Component | Uses delayed observations | Uses ground truth |
|---|---|---|
| Policy network input | ✓ | |
| RNN hidden state | ✓ (learns to estimate GT) | |
| Critic input (CTDE) | | ✓ (privileged) |
| Task reward (formation quality) | | ✓ |
| Safety penalty (CBF margin) | | ✓ |
| Hard CBF filter (deployment) | ✓ (only delayed available) | |
| Episode termination (collision) | | ✓ |

**Rationale for GT rewards:** Computing rewards from delayed observations introduces a temporal shift — the policy gets penalized for dangers already resolved and misses emerging threats. GT rewards provide immediate, accurate credit assignment. The RNN learns to compensate for the observation gap.

**Change from iris_ma5:** iris_ma5 computes all rewards from delayed observations. iris_ma6 computes both task reward and safety penalty from ground truth. The centralized critic also receives GT positions as privileged input.

### 1.4 Architecture Overview

```
                        TRAINING                              DEPLOYMENT
                        
  Delayed Obs ──→ [MAPPO-RNN Policy] ──→ v_nom    Delayed Obs ──→ [MAPPO-RNN Policy] ──→ v_nom
                         │                                            │
              GT state   │                              Delayed pos   │
                 │       │                                    │       │
                 ▼       ▼                                    ▼       ▼
           ┌─────────────────────┐                    ┌──────────────────────┐
           │  CPA-CBF Penalty    │                    │  Distance-CBF Filter │
           │  (reward shaping,   │                    │  (hard constraint,   │
           │   GT positions,     │                    │   inflated D_deploy, │
           │   no delay)         │                    │   delay-robust)      │
           └─────────────────────┘                    └──────────────────────┘
                 │       │                                    │
          penalty│       │v_nom (unfiltered)                  │v_safe
                 ▼       ▼                                    ▼
           reward = task_reward                         PX4 Offboard
                   - λ·penalty                               │
                                                        PX4 Failsafes
```

Note: during training, the hard filter is **not applied** to actions. The policy acts unfiltered so that the soft penalty gradient is meaningful (the policy experiences the consequences of its actions). Episode termination on collision provides the backstop during training.

---

## 2. Mathematical Formulation

### 2.1 System Model

Each drone is modeled as a single integrator from the CBF's perspective:

$$\dot{\mathbf{p}}_i = \mathbf{v}_i, \quad \mathbf{v}_i \in \mathbb{R}^3$$

where $\mathbf{v}_i$ is the velocity command sent to PX4's offboard interface. PX4's velocity controller introduces tracking lag (time constant $\tau_{PX4} \approx 0.2$–$0.5$ s), accounted for differently at each layer.

### 2.2 Training-Time Barrier: Predictive Closest Point of Approach (CPA)

Computed from **ground-truth** positions and **commanded** velocities (both noise-free in the simulator). This barrier encodes collision geometry more accurately than pure distance, avoiding unnecessary conservatism during parallel flight and lateral repositioning.

**Step 1 — Time of closest approach:**

$$\tau^* = -\frac{\Delta\mathbf{p}_{ij}^\top \Delta\mathbf{v}_{ij}}{\|\Delta\mathbf{v}_{ij}\|^2 + \epsilon}, \quad \tau = \text{clamp}(\tau^*, 0, T)$$

where $\Delta\mathbf{p}_{ij} = \mathbf{p}_i - \mathbf{p}_j$, $\Delta\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j$, $T$ is the look-ahead horizon, and $\epsilon$ is a small constant for numerical stability when $\Delta\mathbf{v} \approx 0$.

**Step 2 — Predicted closest approach distance:**

$$d_{CPA}^2 = \|\Delta\mathbf{p}_{ij} + \tau \cdot \Delta\mathbf{v}_{ij}\|^2$$

**Step 3 — Barrier function:**

$$h_{ij}^{CPA} = d_{CPA}^2 - D_s^2$$

Properties:
- $h_{ij}^{CPA} > 0$: predicted miss distance exceeds $D_s$ (safe trajectory)
- $h_{ij}^{CPA} = 0$: predicted grazing collision
- $h_{ij}^{CPA} < 0$: predicted collision within horizon $T$

**Why CPA over distance-based:** The CPA barrier is velocity-aware. Two drones at 2.1 m separation ($D_s = 2.0$ m) flying in parallel at 15 m/s have $\Delta\mathbf{v} \approx 0$, so $d_{CPA} \approx 2.1$ m — no penalty. The distance-based barrier would restrict any approach velocity regardless of direction. This matters at high speeds where lateral repositioning during target tracking is common.

**Why CPA is safe for training-time use:** The CPA prediction assumes constant velocity, which is wrong beyond one timestep. But in the training context, this doesn't matter — the barrier is used for **reward shaping**, not for a hard safety guarantee. A slightly wrong prediction produces a slightly wrong penalty, which is still a much better gradient signal than a distance-only penalty or a sparse collision termination.

### 2.3 Training-Time Reward Signal

The safety penalty is a hinge loss on the CPA barrier margin. Using a discrete-time formulation that avoids differentiating any state signal:

$$\text{penalty}_{ij} = \max\!\Big(0,\; (1 - \gamma\,\Delta t)\;h_{ij}^{CPA,\text{current}} - h_{ij}^{CPA,\text{next}}\Big)$$

where $h_{ij}^{CPA,\text{next}}$ is evaluated at predicted next positions $\mathbf{p}_i + \mathbf{v}_i^{cmd}\Delta t$ using the commanded velocity. Alternatively, the continuous-time margin hinge can be used as a simpler approximation:

$$\text{penalty}_{ij}^{simple} = \max(0,\; -m_{ij})$$

where $m_{ij} = \dot{h}_{ij}^{CPA} + \gamma\,h_{ij}^{CPA}$ and $\dot{h}_{ij}^{CPA}$ is computed analytically from GT positions and commanded velocities (clean signals in the simulator, no noise).

The total safety reward:

$$r_{safety} = -\lambda_{cbf} \sum_{i < j} \text{penalty}_{ij}$$

This penalty is:
- **Zero** when the predicted miss distance is safe and not worsening — no intrusion on task reward.
- **Positive and proportional** to how fast the miss distance is degrading — dense, directional gradient.
- **Velocity-aware** — penalizes approach rate, not proximity. Parallel flight at close range incurs no penalty.

### 2.4 Deployment-Time Barrier: Simple Distance with Inflated Margin

At deployment, only delayed positions are available. No velocity differentiation, no prediction — just the simplest possible barrier with a margin that covers the worst-case position error from comm delay.

$$h_{ij}^{deploy} = \|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j\|^2 - D_{deploy}^2$$

where $\hat{\mathbf{p}}$ denotes the latest received (delayed) position and:

$$D_{deploy} = D_s + v_{max} \cdot \tau_{d,max} + \epsilon_{margin}$$

The inflation $v_{max} \cdot \tau_{d,max}$ covers the worst-case position drift during the maximum communication delay. The additional $\epsilon_{margin}$ accounts for PX4's velocity tracking lag.

The continuous-time CBF condition:

$$2\,\Delta\hat{\mathbf{p}}_{ij}^\top(\mathbf{v}_i - \hat{\mathbf{v}}_j) + \gamma_{deploy}\,h_{ij}^{deploy} \geq 0$$

where $\hat{\mathbf{v}}_j$ is the last received velocity of agent $j$ (from PX4's EKF estimate or communicated command). This uses the standard distance-based formulation — relative degree 1, linear constraint, closed-form halfspace projection.

**Why simple is correct here:** With delayed data, a sophisticated barrier (CPA, collision cone) would compute predictions from stale state, producing confident-but-wrong assessments. The simple distance barrier with inflated margin acknowledges the uncertainty honestly — "I don't know exactly where you are, so I'll keep a bigger buffer." The MAPPO-RNN policy handles the intelligent, delay-aware part of collision avoidance. The deployment filter just catches the edge cases.

### 2.5 Closed-Form Halfspace Projection (Deployment Filter)

With a single pairwise constraint, the CBF-QP reduces to:

$$\mathbf{v}_i^{safe} = \begin{cases} \mathbf{v}_i^{nom} & \text{if } \mathbf{a}_{ij}^\top\mathbf{v}_i^{nom} \geq b_{ij} \\[4pt] \mathbf{v}_i^{nom} + \dfrac{b_{ij} - \mathbf{a}_{ij}^\top\mathbf{v}_i^{nom}}{\|\mathbf{a}_{ij}\|^2}\;\mathbf{a}_{ij} & \text{otherwise} \end{cases}$$

where $\mathbf{a}_{ij} = 2\,\Delta\hat{\mathbf{p}}_{ij}$ and $b_{ij} = -2\,\Delta\hat{\mathbf{p}}_{ij}^\top\hat{\mathbf{v}}_j - \gamma_{deploy}\,h_{ij}^{deploy}$.

For 3 agents (2 pairwise constraints per drone), Gauss-Seidel iterative projection is used (see Section 4.2).

---

## 3. Integration Architecture

### 3.1 Training Loop

```python
def step(self, action):
    v_nom = action                                   # (E, N, 3)

    # Apply action UNFILTERED — policy experiences full consequences
    self._set_velocity_commands(v_nom)
    self.sim.step()

    # Rewards from GROUND TRUTH (Dec-POMDP standard)
    task_reward = self.triangulation_quality(
        self.true_positions, self.true_target_pos     # GT
    )
    cbf_penalty = self.cbf_reward_shaper.compute_penalty(
        v_nom=v_nom,
        positions=self.true_positions,                # GT, no delay
        velocities=v_nom                              # commanded, no noise
    )

    reward = task_reward - lambda_cbf * cbf_penalty

    # Episode termination on actual collision (GT)
    collided = self.check_collision(self.true_positions, D_s)

    # Observations to agents remain DELAYED
    obs = self.get_delayed_observations()

    return obs, reward, collided, info
```

**Key design decisions:**
- **No hard filter during training.** The policy acts unfiltered so that (a) the soft penalty gradient reflects the policy's actual intentions, and (b) the policy learns to avoid collisions itself rather than relying on the filter. Episode termination is the backstop.
- **GT for all rewards.** Task reward and safety penalty both use true positions. Credit assignment is clean — the advantage function immediately attributes safety violations to the actions that caused them.
- **Critic gets GT.** Under CTDE, the centralized critic observes true positions as privileged information, producing lower-variance value estimates.

### 3.2 Deployment Loop

```python
def deploy_step(self, obs):
    v_nom = self.policy(obs)                          # from delayed observations

    # Hard CBF filter on delayed positions
    v_safe = self.deploy_filter.filter(
        v_nom=v_nom,
        positions=self.delayed_positions,             # latest received
        neighbor_velocities=self.delayed_velocities   # latest received
    )

    self.send_to_px4(v_safe)
```

### 3.3 Data Flow

| Signal | Shape | Device | Source | Used By |
|---|---|---|---|---|
| `true_positions` | `(E, N, 3)` | GPU | Sim ground truth | CPA penalty, task reward, critic, collision check |
| `delayed_positions` | `(E, N, 3)` | GPU | Sim with delay model | Policy observations, deployment filter |
| `v_nom` | `(E, N, 3)` | GPU | Policy output | CPA penalty (as commanded vel), deployment filter |
| `cbf_penalty` | `(E,)` | GPU | CPA barrier computation | Reward composition |
| `v_safe` | `(E, N, 3)` | GPU | Deployment filter output | PX4 command (deployment only) |

### 3.4 Deployment Node Architecture (ROS)

```
MARL Node  →  /cmd_vel_nominal  →  [CBF Filter Node]  →  /mavros/setpoint_velocity/cmd_vel  →  PX4
                                          ↑
                                    /agent_positions (delayed)
                                    /agent_velocities (delayed or commanded)
```

---

## 4. Implementation

### 4.1 Training-Time: CPA Reward Shaper

```python
class CPARewardShaper:
    """Predictive CPA-based safety penalty for reward shaping.
    
    Operates on ground-truth state. Used ONLY during training.
    """
    def __init__(self, D_s: float, gamma: float, T: float, num_agents: int):
        self.D_s = D_s
        self.gamma = gamma
        self.T = T              # look-ahead horizon
        self.num_agents = num_agents

    def compute_penalty(self, v_nom, positions, velocities=None):
        """CPA barrier hinge loss.

        Args:
            v_nom:      (E, N, 3) commanded velocities (noise-free)
            positions:  (E, N, 3) ground-truth positions
            velocities: (E, N, 3) optional; defaults to v_nom

        Returns:
            penalty:    (E,) summed penalty per environment
        """
        if velocities is None:
            velocities = v_nom

        E = v_nom.shape[0]
        penalty = torch.zeros(E, device=v_nom.device)

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dp = positions[:, i] - positions[:, j]         # (E, 3)
                dv = velocities[:, i] - velocities[:, j]       # (E, 3)

                # Time of closest approach
                dv_sq = (dv * dv).sum(dim=-1) + 1e-8           # (E,)
                tau_star = -(dp * dv).sum(dim=-1) / dv_sq      # (E,)
                tau = tau_star.clamp(0.0, self.T)               # (E,)

                # Predicted CPA position and barrier
                dp_cpa = dp + tau.unsqueeze(-1) * dv            # (E, 3)
                d_cpa_sq = (dp_cpa * dp_cpa).sum(dim=-1)       # (E,)
                h_cpa = d_cpa_sq - self.D_s ** 2                # (E,)

                # Also compute h at next step (discrete-time condition)
                dp_next = dp + dv * self.dt                     # (E, 3) approximate
                # Recompute tau for next-step state
                dv_sq_next = dv_sq                              # same commands
                tau_next = (-(dp_next * dv).sum(dim=-1) / dv_sq_next).clamp(0, self.T)
                dp_cpa_next = dp_next + tau_next.unsqueeze(-1) * dv
                h_cpa_next = (dp_cpa_next * dp_cpa_next).sum(dim=-1) - self.D_s ** 2

                # Discrete-time barrier violation
                required = (1 - self.gamma * self.dt) * h_cpa
                violation = torch.relu(required - h_cpa_next)
                penalty += violation

        return penalty
```

### 4.2 Deployment-Time: Robust Distance-Based Filter

```python
class RobustDeploymentFilter:
    """Simple distance-based CBF with inflated margin.
    
    Designed for delayed, noisy observations. Used at deployment.
    """
    def __init__(self, D_s: float, v_max: float, tau_delay_max: float,
                 tau_px4: float, gamma: float, num_agents: int):
        self.D_s = D_s
        self.gamma = gamma
        self.num_agents = num_agents
        # Inflate safety distance for delay + tracking lag
        self.D_deploy = D_s + v_max * tau_delay_max + v_max * tau_px4
        
    def _project_halfspace(self, v, a, b):
        """Closed-form projection onto halfspace {v : a^T v >= b}."""
        a_dot_v = (a * v).sum(dim=-1)                  # (E,)
        a_norm_sq = (a * a).sum(dim=-1) + 1e-8         # (E,)
        violated = a_dot_v < b                          # (E,) bool
        scale = ((b - a_dot_v) / a_norm_sq).clamp(min=0.0)
        correction = scale.unsqueeze(-1) * a            # (E, 3)
        v_safe = torch.where(violated.unsqueeze(-1), v + correction, v)
        return v_safe, violated

    def filter(self, v_nom, positions, neighbor_velocities=None, num_iters=2):
        """Hard CBF-QP via iterative halfspace projection.

        Args:
            v_nom:                (E, N, 3) nominal velocities from policy
            positions:            (E, N, 3) latest received (delayed) positions
            neighbor_velocities:  (E, N, 3) latest received velocities;
                                  if None, assumes zero (most conservative)
            num_iters:            Gauss-Seidel passes

        Returns:
            v_safe:     (E, N, 3) filtered safe velocities
            info:       dict with diagnostics
        """
        v_safe = v_nom.clone()
        any_active = torch.zeros(v_nom.shape[0], v_nom.shape[1],
                                 dtype=torch.bool, device=v_nom.device)

        for _ in range(num_iters):
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i == j:
                        continue
                    dp = positions[:, i] - positions[:, j]      # (E, 3)
                    h = (dp * dp).sum(dim=-1) - self.D_deploy ** 2

                    a = 2.0 * dp                                 # (E, 3)
                    vj = (neighbor_velocities[:, j]
                          if neighbor_velocities is not None
                          else torch.zeros_like(v_safe[:, j]))
                    b = -(a * vj).sum(dim=-1) - self.gamma * h  # (E,)

                    v_safe_i, violated = self._project_halfspace(
                        v_safe[:, i], a, b
                    )
                    v_safe[:, i] = v_safe_i
                    any_active[:, i] |= violated

        info = {
            "deploy_cbf/filter_active": any_active.float().mean().item(),
        }
        return v_safe, info
```

### 4.3 Training Diagnostics

```python
class CBFDiagnostics:
    """Tracks safety metrics during training. All computed from GT."""

    @staticmethod
    def compute(v_nom, true_positions, D_s, num_agents):
        metrics = {}

        # Minimum pairwise separation (GT)
        min_dists = []
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dp = true_positions[:, i] - true_positions[:, j]
                dist = dp.norm(dim=-1)
                min_dists.append(dist)
        min_sep = torch.stack(min_dists).min(dim=0).values

        metrics["safety/min_separation_mean"] = min_sep.mean()
        metrics["safety/min_separation_min"] = min_sep.min()
        metrics["safety/collision_fraction"] = (min_sep < D_s).float().mean()

        return metrics
```

---

## 5. Parameters

### 5.1 Training-Time Parameters (CPA Reward Shaper)

| Parameter | Symbol | Default | Notes |
|---|---|---|---|
| Physical safety distance | $D_s$ | 2.0 m | True minimum separation. No inflation needed — using GT. |
| CBF decay rate | $\gamma$ | 2.0 | Controls how fast barrier is allowed to shrink per step. |
| Look-ahead horizon | $T$ | 1.0 s | CPA prediction horizon. Set $\approx D_s / v_{max}$. |
| Reward penalty weight | $\lambda_{cbf}$ | (tune) | See Section 5.3. |

### 5.2 Deployment-Time Parameters (Robust Filter)

| Parameter | Symbol | Default | Notes |
|---|---|---|---|
| Physical safety distance | $D_s$ | 2.0 m | Same as training. |
| Max agent speed | $v_{max}$ | 15.0 m/s | Maximum expected velocity magnitude. |
| Max comm delay | $\tau_{d,max}$ | 0.2 s | Worst-case observation staleness. |
| PX4 tracking lag | $\tau_{PX4}$ | 0.3 s | Velocity controller time constant. |
| **Deployed safety distance** | $D_{deploy}$ | **$\approx 9.5$ m** | $= D_s + v_{max}(\tau_{d,max} + \tau_{PX4}) = 2 + 15 \times 0.5$ |
| Deploy CBF decay rate | $\gamma_{deploy}$ | 1.0 | Conservative — slower allowed approach at boundary. |
| Projection iterations | `num_iters` | 2 | Gauss-Seidel passes. 1 usually sufficient for 3 agents. |

Note: $D_{deploy} = 9.5$ m is large but honest — it reflects the physical reality that at 15 m/s with 0.5 s of total uncertainty, the drone could be anywhere in a 7.5 m radius beyond the last known position. The MAPPO policy, trained with GT rewards, will learn to avoid triggering this filter by maintaining sufficient margins naturally. The filter activation rate is a key diagnostic — if it's high, either the policy needs more training or the delay model is too aggressive.

### 5.3 Tuning $\lambda_{cbf}$

Heuristic: at a "mildly concerning" configuration (drones at $1.5 \times D_s$ apart, approaching at 1 m/s), the CPA penalty should produce a reward hit roughly equal to one timestep of the main task reward.

Compute: at 3.0 m apart approaching at 1 m/s:
- $\tau^* = 3.0/1.0 = 3.0$ s, clamped to $T = 1.0$ s
- $d_{CPA} = \sqrt{(3.0 - 1.0)^2} = 2.0$ m (right at boundary)
- $h^{CPA} = 4.0 - 4.0 = 0.0$

This configuration should produce a moderate penalty. Adjust $\lambda_{cbf}$ so that $\lambda_{cbf} \cdot \text{penalty} \approx |r_{task}^{typical}|$.

---

## 6. Layered Safety Architecture

### 6.1 Overview

| Layer | Mechanism | Input data | Guarantee | When active |
|---|---|---|---|---|
| L0 | Soft CPA penalty in reward | GT positions + commanded velocities | Learned delay-aware avoidance | Training only |
| L1 | MAPPO-RNN policy | Delayed observations | Trained collision avoidance (no formal guarantee) | Training + Deployment |
| L2 | Distance-CBF hard filter | Delayed positions, inflated margin | Forward invariance (single-integrator, worst-case delay) | Deployment only |
| L3 | PX4 failsafes | Onboard sensors | Firmware-level safety envelope | Always |

Each layer is independent. If L1 (policy) fails, L2 (filter) catches it. If L2 has numerical issues, L3 (PX4) provides the final backstop. Layers degrade gracefully.

### 6.2 Why MAPPO-RNN Is the Primary Collision Avoidance

The MAPPO-RNN, trained with observation delays in the loop and GT-based safety rewards, learns collision avoidance that no analytical CBF can match under delay:

- **Memory:** The RNN hidden state maintains an implicit belief about agents' true current positions from delayed observations — effective state estimation.
- **Prediction:** The policy learns delay-compensating behavior: "if I see a drone at position X with 200 ms delay, it might actually be 3 m closer, so I should maintain a larger margin."
- **Multi-objective:** The policy jointly optimizes triangulation quality and collision avoidance, finding configurations that serve both — analytical CBF can only constrain, not optimize.

The CPA reward shaping (L0) accelerates this learning by providing dense, directional gradient information about the collision geometry, computed from clean GT signals.

### 6.3 Why the Deployment Filter Is Simple

At deployment, all state information is delayed and noisy. Sophisticated barriers (CPA, collision cone) would compute predictions from stale data, producing confident-but-wrong assessments. The simple distance barrier with inflated margin:

- **Makes no velocity assumptions** — only needs positions (the most reliable measurement).
- **Acknowledges uncertainty honestly** — the inflated $D_{deploy}$ says "I don't know exactly where you are."
- **Never makes things worse** — worst case is over-conservative, which the policy has learned to avoid triggering.
- **Is formally verifiable** — simple enough for static analysis and certification arguments.

### 6.4 Relationship to Episode Termination

The CPA reward shaping does **not** replace collision termination. Both remain active during training:

- **Episode termination** on collision (GT): strong signal that the overall trajectory was unacceptable.
- **CPA penalty** (GT): per-timestep gradient about which actions approached the unsafe regime.
- **Together:** the penalty teaches the boundary; termination teaches the consequence. As training progresses, collisions become rare and episodes grow longer, allowing more task learning.

### 6.5 Comparison to iris_ma5

| Aspect | iris_ma5 | iris_ma6 |
|---|---|---|
| Task reward source | Delayed observations | Ground truth |
| Safety reward source | Delayed observations | Ground truth |
| Safety reward type | (collision termination only) | CPA barrier hinge loss + termination |
| Critic input | Delayed observations | GT positions (privileged CTDE) |
| Deployment safety filter | None | Distance-CBF with inflated margin |
| Delay handling | Implicit (policy learns from delayed rewards) | Explicit (GT rewards + delayed obs → policy learns delay compensation) |

---

## 7. Future Extensions

### 7.1 Adaptive $D_{deploy}$ via Delay Estimation

Rather than using worst-case $\tau_{d,max}$, estimate actual delay per agent online and adjust $D_{deploy}$ dynamically:

$$D_{deploy}^{ij}(t) = D_s + v_{max} \cdot \hat{\tau}_d^j(t) + \epsilon$$

Reduces conservatism when comm conditions are good, maintains safety when they degrade.

### 7.2 Communicated Intent Protocol

For cooperative agents, broadcast commanded velocities alongside position updates. Even with comm delay, the commanded velocity represents the agent's *intent* at the time of broadcast — more informative than the position alone. The deployment filter can use this to make the CPA prediction even with delayed data:

$$d_{CPA}^{deploy} = \|\Delta\hat{\mathbf{p}} + \tau \cdot \Delta\hat{\mathbf{v}}^{cmd}\|$$

This is a middle ground between the conservative distance-only filter and the GT-based CPA. Requires analysis of how prediction error from delay affects the safety guarantee.

### 7.3 Simplex Runtime Assurance

The layered architecture maps naturally onto the Simplex framework:
- **Advanced controller:** MAPPO-RNN policy (unverified, high performance)
- **Baseline controller:** PX4 waypoint navigation (verified, safe)
- **Decision module:** CBF margin monitor — switches to baseline if margin drops below threshold

This provides a stronger system-level safety argument for certification: the learned policy is never trusted for safety — only for performance.

### 7.4 Learned CBF (GCBF) — When Needed

GCBF is recommended when scaling beyond the current 2–3 agent setup:
- **5+ cooperative agents:** analytical CBF constraints scale quadratically; GCBF with GNN handles this naturally.
- **Adversarial scenarios (드론공방전):** can't trust communicated intent from opponents; need a safety mechanism that reasons about worst-case neighbor behavior.
- **Mixed cooperative/adversarial:** e.g., tracker team (cooperative) vs. target drone (adversarial).

**Not recommended for current IRIS setup** because: (a) 2–3 agents don't need the scalability, (b) adding a second learned component (GCBF) complicates the safety argument for certification, (c) the comm delay problem is better addressed by the GT reward shaping + robust deployment filter architecture.

### 7.5 Higher-Order CBF for Double Integrator

If PX4's velocity tracking lag causes the single-integrator model to be insufficient (deployment filter activates too frequently due to overshoot), upgrade to a double-integrator model with Exponential CBFs (ECBFs). This requires knowledge of PX4's internal controller gains to model the acceleration response.

---

## 8. Implementation Checklist

- [ ] Implement `CPARewardShaper` with GT state access in Isaac Lab env
- [ ] Implement `RobustDeploymentFilter` with delayed state input
- [ ] Modify reward function: task reward from GT, safety penalty from GT
- [ ] Modify critic: add GT positions to privileged observation space (CTDE)
- [ ] Add `CBFDiagnostics` to training logger
- [ ] Tune $\lambda_{cbf}$: sweep with fixed task reward, target ~0 collision rate
- [ ] Validate deployment filter: test with simulated delay + noise, measure activation rate
- [ ] Compare vs iris_ma5 baseline: collision rate, triangulation quality, training speed
- [ ] Ablation: GT rewards vs delayed rewards, CPA penalty vs distance penalty vs none

---

## 9. References

- Ames, A.D. et al. "Control Barrier Functions: Theory and Applications." ECC 2019.
- Borrmann, U. et al. "Control Barrier Certificates for Safe Swarm Behavior." ADHS 2015.
- Brunke, L. et al. "Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning." Annual Review of Control, Robotics, and Autonomous Systems, 2022.
- Clark, A. "Control Barrier Functions for Stochastic Systems." Automatica, 2021.
- Mehmood, U. et al. "The Black-Box Simplex Architecture for Runtime Assurance of Autonomous CPS." NFM 2022.
- Mestres, J. et al. "Explicit Control Barrier Function-based Safety Filters and their Resource-Aware Computation." arXiv:2512.10118, 2025.
- Qin, Z. et al. "GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control." T-RO 2024.
- Tayal, M. et al. "A Collision Cone Approach for Control Barrier Functions." arXiv:2403.07043, 2024.
- Yang, L. et al. "CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions." arXiv:2510.14959, 2025.
- Zhang, Z. et al. "Safety Guaranteed Robust Multi-Agent Reinforcement Learning with Hierarchical Control for Connected and Automated Vehicles." arXiv:2309.11057, 2024.