# CBF Safety — Mathematical Reference

Module-level math for the two CBFs in [iris_ma6/cbf_safety/](../). Higher-level
motivation, the layered architecture, and the deployment ROS plumbing live in
[doc/safety_spec.md](../../doc/safety_spec.md); this file derives only the
equations the code actually implements, with pointers to the source.

---

## 1. Setup

### 1.1 Notation

For each environment, $N$ agents are indexed by $i, j \in \{1, \ldots, N\}$.
Pairs are indexed by $(i,j)$ with $i < j$ during training (symmetric pairs are
not double-counted) and $i \neq j$ during deployment (Gauss-Seidel iterates
over directed pairs).

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{p}_i \in \mathbb{R}^3$ | Position of agent $i$ (world frame) | $(E, 3)$ |
| $\mathbf{v}_i \in \mathbb{R}^3$ | Velocity command of agent $i$ to PX4 | $(E, 3)$ |
| $\Delta \mathbf{p}_{ij} \equiv \mathbf{p}_i - \mathbf{p}_j$ | Relative position | $(E, 3)$ |
| $\Delta \mathbf{v}_{ij} \equiv \mathbf{v}_i - \mathbf{v}_j$ | Relative velocity | $(E, 3)$ |
| $D_s$ | Physical safety distance | scalar |
| $\gamma$ | CBF decay rate (training) | scalar (1/s) |
| $\gamma_d$ | CBF decay rate (deployment) | scalar (1/s) |
| $T$ | CPA look-ahead horizon | scalar (s) |
| $\Delta t$ | Simulation step (env decimation) | scalar (s) |
| $E$ | Number of parallel envs | int |
| $\hat{\cdot}$ | Delayed measurement (deployment only) | — |

### 1.2 Kinematic model

From the CBF's perspective each drone is a single integrator on its commanded
velocity:

$$\dot{\mathbf{p}}_i = \mathbf{v}_i, \qquad \mathbf{v}_i \in \mathbb{R}^3 .$$

PX4's inner velocity-tracking lag $\tau_{PX4}$ does **not** appear in the
training-time barrier: training uses ground-truth simulator positions and the
*commanded* velocity, both clean. The lag is folded into the deployment-time
margin (§5).

### 1.3 Safe set

For each pair, the unsafe set is the open ball of radius $D_s$ around relative
position zero. The training barrier function

$$h_{ij}(\mathbf{p}_i, \mathbf{p}_j, \mathbf{v}_i, \mathbf{v}_j) \;=\; d_{CPA}^2(\mathbf{p}, \mathbf{v}) \,-\, D_s^{\,2}$$

is positive on the safe set, zero on the boundary, negative inside. The
super-level set $\{h \geq 0\}$ is what we want forward-invariant.

---

## 2. The CPA Barrier (Training)

### 2.1 Time of closest approach

Under the constant-velocity prediction $\mathbf{p}_i(t+\tau) = \mathbf{p}_i + \tau \mathbf{v}_i$,
the squared inter-agent distance evolves as

$$D_{ij}^{\,2}(\tau) \;=\; \| \Delta\mathbf{p}_{ij} + \tau \Delta\mathbf{v}_{ij} \|^2
= \|\Delta\mathbf{p}_{ij}\|^2 + 2\tau\,\Delta\mathbf{p}_{ij}^{\!\top}\!\Delta\mathbf{v}_{ij} + \tau^2 \|\Delta\mathbf{v}_{ij}\|^2 .$$

Setting $\frac{d}{d\tau} D_{ij}^{\,2} = 0$ gives the unconstrained closest-approach time

$$\tau_{ij}^{\,\star} \;=\; -\,\frac{\Delta\mathbf{p}_{ij}^{\!\top}\!\Delta\mathbf{v}_{ij}}{\|\Delta\mathbf{v}_{ij}\|^2 + \varepsilon} \tag{2.1}$$

with $\varepsilon = 10^{-8}$ for numerical stability when relative velocity is
near zero. The CPA-relevant time is then clamped to a finite, non-negative
horizon:

$$\tau_{ij} \;=\; \mathrm{clip}(\tau_{ij}^{\,\star},\; 0,\; T) . \tag{2.2}$$

The lower clamp at $0$ rejects configurations already separating ($\tau^\star < 0$
means CPA is in the past — drones moving apart). The upper clamp at $T$
rejects extrapolation: beyond $T$ seconds the constant-velocity prediction is
unreliable.

Implementation: [cpa_reward_shaper.py:99-103](../cpa_reward_shaper.py#L99-L103).

### 2.2 Predicted miss distance

Substituting $\tau_{ij}$ back into the position prediction gives the CPA
relative position and squared CPA distance:

$$\mathbf{r}_{ij}^{CPA} \;=\; \Delta\mathbf{p}_{ij} + \tau_{ij}\, \Delta\mathbf{v}_{ij},
\qquad d_{CPA,ij}^{\,2} \;=\; \mathbf{r}_{ij}^{CPA \, \!\top}\,\mathbf{r}_{ij}^{CPA} . \tag{2.3}$$

The CPA barrier value is the squared margin against $D_s$:

$$\boxed{\;h_{ij}^{CPA} \;=\; d_{CPA,ij}^{\,2} \,-\, D_s^{\,2}\;} \tag{2.4}$$

with the standard interpretation: $h > 0$ predicted-safe, $h = 0$ predicted
grazing collision, $h < 0$ predicted collision within $[0, T]$.

Implementation: [cpa_reward_shaper.py:105-115](../cpa_reward_shaper.py#L105-L115).

### 2.3 Why CPA, not distance

For two drones at $\|\Delta\mathbf{p}\| = D_s + \delta$ with $\delta$ small,
moving in parallel at high speed ($\Delta\mathbf{v} \approx 0$):

- Distance barrier $h = \|\Delta\mathbf{p}\|^2 - D_s^2 \approx 2 D_s \delta$,
  with $\dot h = 2\,\Delta\mathbf{p}^\top \Delta\mathbf{v} \approx 0$. Margin
  $m = \dot h + \gamma h \approx 2\gamma D_s \delta$ — small positive,
  near zero penalty even at full speed.
- CPA barrier: $\tau^\star \approx 0$, $d_{CPA} \approx \|\Delta\mathbf{p}\|$,
  so $h^{CPA} \approx h$ — same value, but the $\tau$-clamping behavior
  differs once $\Delta\mathbf{v}$ becomes nonzero.

The advantage shows up when $\Delta\mathbf{v} \neq 0$ but parallel: $\tau^\star$
is small (drones don't get appreciably closer), so $d_{CPA}$ stays large —
**the barrier doesn't penalize lateral repositioning at high speed**, which the
distance-only barrier would.

---

## 3. Discrete-Time CBF Penalty

### 3.1 Discrete-time CBF condition

The continuous-time CBF condition is

$$\dot h + \gamma h \geq 0 \quad \Longleftrightarrow \quad \frac{d}{dt}\,h \;\geq\; -\gamma h .$$

For a fixed step $\Delta t$, the standard discrete-time analogue is

$$h_{k+1} \;\geq\; (1 - \gamma\,\Delta t)\, h_k . \tag{3.1}$$

Rewriting as a violation magnitude (positive when the constraint is broken):

$$V_{ij,k} \;=\; \max\!\Big(0,\; (1 - \gamma\,\Delta t)\, h_{ij,k}^{CPA} \;-\; h_{ij,k+1}^{CPA}\Big) . \tag{3.2}$$

The "next" barrier value $h_{k+1}^{CPA}$ is computed by **predicting forward
one step** under constant velocity:

$$\Delta\mathbf{p}_{ij}^{\,(k+1)} \;=\; \Delta\mathbf{p}_{ij}^{\,(k)} + \Delta t \cdot \Delta\mathbf{v}_{ij}^{\,(k)},
\qquad \Delta\mathbf{v}_{ij}^{\,(k+1)} \;=\; \Delta\mathbf{v}_{ij}^{\,(k)} \tag{3.3}$$

(commanded velocity is held constant across the lookahead — the policy hasn't
issued the next command yet). Then $h_{ij,k+1}^{CPA}$ is computed by
re-running (2.1)–(2.4) with the predicted $\Delta\mathbf{p}^{(k+1)}$.

Implementation: [cpa_reward_shaper.py:158-171](../cpa_reward_shaper.py#L158-L171).

### 3.2 Per-environment penalty and reward

Sum over all unique pairs $(i, j)$ with $i < j$:

$$\mathrm{penalty}^{\,\text{env}} \;=\; \sum_{i < j} V_{ij}, \qquad
r_{\text{safety}} \;=\; -\,\lambda_{cbf}\,\mathrm{penalty}^{\,\text{env}} . \tag{3.4}$$

Implementation: [cpa_reward_shaper.py:173-176](../cpa_reward_shaper.py#L173-L176).
The composition into total reward happens in
[iris_ma_env6_test.py:1480-1610](../../iris_ma_env6_test.py#L1480-L1610).

### 3.3 Properties of the hinge penalty

- **Sparsity**: $V_{ij} = 0$ whenever the predicted miss distance is
  non-decreasing fast enough ($h_{k+1} \geq (1 - \gamma \Delta t) h_k$).
  Idle pairs incur no cost — does not bias the policy toward "stay far apart"
  globally.
- **Direction**: Non-zero only along approach trajectories that violate the
  decay budget. The gradient $\partial V_{ij} / \partial \mathbf{v}_i$ pushes
  back along the relative-velocity axis, not the relative-position axis —
  exactly the "slow your approach" signal.
- **Magnitude scaling**: $V_{ij}$ has units of squared distance $\times$ rate;
  $\lambda_{cbf}$ converts to reward units. See §7 for tuning.

---

## 4. Continuous-Time Alternative

A simpler approximation lives in
[cpa_reward_shaper.py:178-227](../cpa_reward_shaper.py#L178-L227) as
`compute_penalty_simple`. It evaluates the continuous-time margin at the
*current* state, with $\dot h$ computed analytically from the **distance
barrier** $h^{dist} = \|\Delta\mathbf{p}\|^2 - D_s^2$:

$$\dot h^{dist} \;=\; 2\, \Delta\mathbf{p}^\top \Delta\mathbf{v}, \qquad
m_{ij} \;=\; \dot h^{dist} + \gamma\, h_{ij}^{CPA},
\qquad V_{ij}^{simple} \;=\; \max(0,\, -m_{ij}) . \tag{4.1}$$

This mixes the distance barrier's derivative with the CPA barrier value. It is
exact when $\tau_{ij} = 0$ (drones at closest approach right now) and an
approximation otherwise. Cheaper than (3.2) by one CPA evaluation; used as a
sanity check during development. The default reward path uses (3.2).

---

## 5. Deployment-Time Distance CBF

### 5.1 Inflated safety distance

Deployment uses delayed positions $\hat{\mathbf{p}}_i$. The worst-case position
error from communication delay $\tau_{d,\max}$ at maximum speed $v_{\max}$ is
$v_{\max}\tau_{d,\max}$. Layered onto that, PX4's velocity-tracking time
constant $\tau_{PX4}$ contributes a further $v_{\max}\tau_{PX4}$ of drift
between command and actual motion. The deployment safety distance inflates by
both:

$$\boxed{\;D_{deploy} \;=\; D_s + v_{\max}(\tau_{d,\max} + \tau_{PX4})\;} \tag{5.1}$$

Implementation: [cbf_cfg.py:86-95](../cbf_cfg.py#L86-L95) (as a property on
`RobustDeploymentFilterCfg`). With defaults $D_s = 2$ m, $v_{\max} = 15$ m/s,
$\tau_{d,\max} = 0.2$ s, $\tau_{PX4} = 0.3$ s the result is $D_{deploy} = 9.5$
m.

### 5.2 Distance barrier and CBF constraint

$$h_{ij}^{deploy} \;=\; \|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j\|^2 \,-\, D_{deploy}^{\,2} . \tag{5.2}$$

$\dot h^{deploy}$ under single-integrator dynamics is

$$\dot h_{ij}^{deploy} \;=\; 2\,\Delta\hat{\mathbf{p}}_{ij}^{\,\!\top}(\mathbf{v}_i - \hat{\mathbf{v}}_j) , \tag{5.3}$$

where $\hat{\mathbf{v}}_j$ is the latest received velocity of agent $j$ (or
zero, the most conservative choice, when not available). The CBF condition
$\dot h + \gamma_d h \geq 0$ becomes a **linear inequality in the optimization
variable $\mathbf{v}_i$**:

$$2\,\Delta\hat{\mathbf{p}}_{ij}^{\,\!\top}\,\mathbf{v}_i
\;\geq\;
2\,\Delta\hat{\mathbf{p}}_{ij}^{\,\!\top}\,\hat{\mathbf{v}}_j
\;-\;
\gamma_d \, h_{ij}^{deploy} . \tag{5.4}$$

Writing $\mathbf{a}_{ij} = 2\,\Delta\hat{\mathbf{p}}_{ij}$ and
$b_{ij} = 2\,\Delta\hat{\mathbf{p}}_{ij}^{\,\!\top}\hat{\mathbf{v}}_j - \gamma_d\,h_{ij}^{deploy}$,
this is the halfspace constraint

$$\mathbf{a}_{ij}^{\,\!\top}\,\mathbf{v}_i \;\geq\; b_{ij} . \tag{5.5}$$

Implementation: [deploy_filter.py:149-160](../deploy_filter.py#L149-L160).

> **Sign convention note.** The implementation uses
> $b_{ij} = -\mathbf{a}_{ij}^{\,\!\top} \hat{\mathbf{v}}_j - \gamma_d h$ (with a
> minus on the velocity term). This corresponds to writing the constraint with
> $\Delta \mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j$ on one side and
> moving $-\mathbf{a}_{ij}^\top \mathbf{v}_j$ to the right. The two forms are
> algebraically identical; the equation above is written in the "everything on
> the LHS" form, the code in the "constant on the RHS" form.

### 5.3 Closed-form halfspace projection

The CBF-QP for a single pairwise constraint reduces to a one-line projection:

$$\mathbf{v}_i^{\,safe} \;=\;
\begin{cases}
\mathbf{v}_i^{\,nom} & \text{if } \mathbf{a}_{ij}^{\,\!\top}\mathbf{v}_i^{\,nom} \geq b_{ij} \\[6pt]
\mathbf{v}_i^{\,nom} + \dfrac{b_{ij} - \mathbf{a}_{ij}^{\,\!\top}\mathbf{v}_i^{\,nom}}{\|\mathbf{a}_{ij}\|^2 + \varepsilon}\,\mathbf{a}_{ij}
& \text{otherwise} .
\end{cases} \tag{5.6}$$

This is the orthogonal projection of $\mathbf{v}^{nom}$ onto the halfspace
$\{\mathbf{v} : \mathbf{a}^\top \mathbf{v} \geq b\}$ — minimum-norm correction.

Implementation: [deploy_filter.py:69-104](../deploy_filter.py#L69-L104). The
$\mathrm{ReLU}((b - \mathbf{a}^\top \mathbf{v}) / \|\mathbf{a}\|^2)$ in the
code is the unified branch — positive only when the constraint is violated.

### 5.4 Multiple constraints: Gauss-Seidel iteration

For $N$ agents each agent has $N - 1$ pairwise constraints. The full QP is

$$\mathbf{v}_i^{\,safe} = \arg\min_{\mathbf{v}_i}\; \tfrac{1}{2} \|\mathbf{v}_i - \mathbf{v}_i^{\,nom}\|^2
\quad \text{s.t.} \quad \mathbf{a}_{ij}^{\,\!\top}\mathbf{v}_i \geq b_{ij} \;\;\forall\, j \neq i. \tag{5.7}$$

The implementation does not solve (5.7) exactly — it sweeps the constraints
sequentially, projecting onto each in turn (Gauss-Seidel). For each iteration
$k = 1, \ldots, K$ and each $j \in \{1, \ldots, N\} \setminus \{i\}$:

$$\mathbf{v}_i^{\,(k, j)} \;=\; \mathrm{Proj}_{\,\{\mathbf{a}_{ij}^{\!\top}\mathbf{v} \,\geq\, b_{ij}\}}\bigl(\mathbf{v}_i^{\,(k, j-1)}\bigr) . \tag{5.8}$$

For two-constraint cases (3 agents) and convex feasible regions, $K = 1$–$2$
sweeps converge. The default `num_iters = 2` from
[cbf_cfg.py:79-80](../cbf_cfg.py#L79-L80) is enough for the 2–3 agent setup.

Implementation: [deploy_filter.py:142-168](../deploy_filter.py#L142-L168).

> **When Gauss-Seidel can fail**: if two constraints have nearly opposite
> normals ($\mathbf{a}_{ij} \approx -\mathbf{a}_{ik}$) and the nominal velocity
> is far from the feasible cone, sequential projection oscillates and may not
> reach feasibility in $K$ iterations. The deployment design avoids this by
> training with the soft penalty so that the policy almost never produces
> nominal velocities that need large corrections — the filter activates rarely
> and only for small adjustments.

---

## 6. Episode termination

A separate, sparse signal: the env terminates whenever any GT pair distance
falls below `cbf_safety.collision_distance` (defaults to `D_s = 2.0` m). This
is enforced outside the barrier — the barrier learns the gradient of where to
go; termination encodes "going there is unacceptable." Both signals coexist
during training. See [cbf_cfg.py:149-154](../cbf_cfg.py#L149-L154).

---

## 7. Parameters

### 7.1 CPA shaper

| Param | Symbol | Default | Range / unit | Effect |
|---|---|---|---|---|
| `D_s` | $D_s$ | 2.0 | m | Squared on the right-hand side of (2.4); shifts the barrier zero. Match physical safety distance. |
| `gamma` | $\gamma$ | 2.0 | 1/s | Allowed exponential decay rate of $h$. Larger ⇒ stricter (less tolerance for approach). At $\gamma \Delta t = 1$ the barrier may shrink to zero in one step. |
| `T` | $T$ | 3.0 | s | Lookahead horizon for the $\tau$-clamp. Set $T \approx D_s / v_{\max}$ — beyond that, the constant-velocity prediction is unreliable. |
| `lambda_cbf` | $\lambda_{cbf}$ | 2.0 | dimensionless | Reward weight; scales the hinge penalty into reward units. See §7.3. |
| `epsilon` | $\varepsilon$ | $10^{-8}$ | — | Stabilizes (2.1) when $\|\Delta\mathbf{v}\|$ → 0. |

### 7.2 Deployment filter

| Param | Symbol | Default | Range / unit | Effect |
|---|---|---|---|---|
| `D_s` | $D_s$ | 2.0 | m | As above; appears inside (5.1). |
| `v_max` | $v_{\max}$ | 15.0 | m/s | Used only for inflation (5.1). Higher ⇒ more conservative. |
| `tau_delay_max` | $\tau_{d,\max}$ | 0.2 | s | Worst-case comm staleness. |
| `tau_px4` | $\tau_{PX4}$ | 0.3 | s | PX4 inner velocity-loop time constant. |
| `gamma_deploy` | $\gamma_d$ | 1.0 | 1/s | Slower than $\gamma$: more conservative under uncertainty. |
| `num_iters` | $K$ | 2 | int | Gauss-Seidel sweeps. |

### 7.3 Tuning $\lambda_{cbf}$

The penalty (3.2) has units of squared distance times a per-step rate. A handy
heuristic: at a "mildly concerning" config (drones at $1.5\,D_s$ apart,
approaching each other at 1 m/s, $\gamma = 2$, $\Delta t = 0.04$ s), what
penalty should one timestep produce relative to one timestep of task reward?

For two drones at $\|\Delta\mathbf{p}\| = 3$ m (with $D_s = 2$ m) approaching
along the line at 1 m/s:

- $\Delta\mathbf{p} = (3, 0, 0)$, $\Delta\mathbf{v} = (-1, 0, 0)$.
- $\tau^\star = -(3)(-1) / (1 + \varepsilon) = 3$ s, clamped to $T = 3$ s ⇒ $\tau = 3$ s.
- $\mathbf{r}^{CPA} = (3, 0, 0) + 3 \cdot (-1, 0, 0) = (0, 0, 0)$,
  so $d_{CPA}^2 = 0$ and $h^{CPA} = -4$ m².
- $\Delta \mathbf{p}^{(k+1)} = (3 - 0.04, 0, 0) = (2.96, 0, 0)$,
  $\tau$ recomputed to $2.96$ s, $\mathbf{r}^{CPA, (k+1)} = (0, 0, 0)$,
  $h^{CPA}_{k+1} = -4$ m².
- Required: $(1 - 2 \cdot 0.04)\cdot(-4) = -3.68$. Violation:
  $\max(0, -3.68 - (-4)) = 0.32$ m².

So $V \approx 0.32$ per step at this config. Compare to one step of task
reward at the working point — the dense triangulation reward is $\sim 4$
per step at the peak. Setting $\lambda_{cbf} = 2$ gives $|\lambda_{cbf} V|
\approx 0.64$ per step, ~16% of typical task reward. That's enough to bend the
gradient without overwhelming the task signal.

Symptoms outside the sweet spot:
- $\lambda_{cbf}$ too low (e.g. 1.0, the previous default): the CBF penalty
  episode-summed reads at $\sim$ -0.04, while the per-collision penalty reads
  at $\sim$ -0.20 — barrier is shaped, but its gradient is dominated by the
  collision spike. Policy "doesn't know" the boundary until it hits it.
- $\lambda_{cbf}$ too high: episodes become long but timid; sigma collapses
  early and triangulation reward suffers because the policy refuses to
  approach close formations.

---

## 8. Caveats and known limitations

### 8.1 Constant-velocity assumption

The CPA prediction (2.1)–(2.3) assumes both drones hold their current
commanded velocities for the duration of $\tau \in [0, T]$. In reality the
policy issues a new $\mathbf{v}$ every $\Delta t$, so the prediction error
grows with $\tau$. The clamp $\tau \leq T$ bounds the worst case.

For training, this is fine: the penalty is reward shaping, not a guarantee.
For deployment we drop CPA entirely and use the distance barrier (5.2), which
makes no velocity assumption.

### 8.2 Two coupled $h$ evaluations per step

The discrete-time hinge (3.2) recomputes the full CPA pipeline at $k$ and at
$k+1$. For small $\Delta t$ and slow-moving drones, $h_{k+1} \approx h_k$ to
first order, and (3.2) is dominated by the $-\gamma \Delta t \cdot h_k$ term.
This means at low speed the discrete penalty effectively reduces to a
"barrier-value times decay rate" hinge — the predictive value comes from the
$\tau$-clamp catching trajectories that *will* cross the boundary even though
the current pair is fine.

### 8.3 Pairs are computed exhaustively

For $N$ agents, the implementation enumerates all $N(N-1)/2$ unordered pairs
in `_generate_pair_indices` and processes them as a single batched tensor of
shape $(E, P, 3)$ with $P = N(N-1)/2$. Memory and compute scale as $O(E P)$;
fine through $N \sim 8$ on an RTX 4090. Beyond that the recommended swap is
GCBF (graph CBF with neighbor sparsification), not a re-implementation of
this module.

### 8.4 The deployment filter is an approximation, not a proof

(5.1) bounds worst-case position drift but not worst-case acceleration error
(PX4's inner loop is modeled only by its time constant). For a formal
forward-invariance proof of the closed-loop system one needs an Exponential CBF
or a higher-relative-degree formulation; (5.6) is an *engineering* safety net
under the assumption that the trained policy almost never tries to violate it.
See [doc/safety_spec.md §7.5](../../doc/safety_spec.md) for the upgrade path.

---

## 9. Implementation index

| Equation | File | Lines |
|---|---|---|
| $\tau^\star,\, \tau$ — (2.1)–(2.2) | [cpa_reward_shaper.py](../cpa_reward_shaper.py) | 99–103 |
| $h^{CPA}$ — (2.4) | [cpa_reward_shaper.py](../cpa_reward_shaper.py) | 105–115 |
| Discrete violation — (3.2) | [cpa_reward_shaper.py](../cpa_reward_shaper.py) | 158–171 |
| Pair sum — (3.4) | [cpa_reward_shaper.py](../cpa_reward_shaper.py) | 173–176 |
| Continuous alternative — (4.1) | [cpa_reward_shaper.py](../cpa_reward_shaper.py) | 178–227 |
| $D_{deploy}$ — (5.1) | [cbf_cfg.py](../cbf_cfg.py) | 86–95 |
| $h^{deploy}$ + linearization — (5.2)–(5.5) | [deploy_filter.py](../deploy_filter.py) | 149–160 |
| Halfspace projection — (5.6) | [deploy_filter.py](../deploy_filter.py) | 69–104 |
| Gauss-Seidel sweep — (5.8) | [deploy_filter.py](../deploy_filter.py) | 142–168 |
| Episode termination | [cbf_cfg.py](../cbf_cfg.py) | 149–154 |
| Reward composition $r_\text{safety}$ | [iris_ma_env6_test.py](../../iris_ma_env6_test.py) | 1480–1610 |
