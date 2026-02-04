# iris_ma4 Technical Report (Environment V4)

This document is a technical report for the `iris_ma4` multi-agent environment implemented in `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/iris_ma_env4.py`.
It focuses on the mathematical formulation and the interactions between the environment’s submodules:

- Multi-agent environment and dataflow
- Action → command mapping, controllers, and gimbal stabilization
- Delay + noise modeling (DelaySystemV2-based)
- Detection (BBoxRayCaster) and camera projection model
- Triangulation and triangulation covariance
- Safety penalties (collision + TTC)
- Curriculum, randomization, and moving target dynamics
- Training interface assumptions (MAPPO-RNN config)

---

## 1) Problem Formulation

### 1.1 Multi-agent POMDP (with delays)

Let there be:
- `C` agents/cameras (drones), indexed by `i ∈ {1,…,C}`.
- `T` targets (here typically `T = 1`), indexed by `t`.
- `N` parallel vectorized environments, indexed by `n`.
- Discrete control steps of size `Δt = step_dt` (decimated from physics stepping).

The environment is a delayed partially observable multi-agent process. At each control step `k`:

- Each agent executes an action:
  \[
  a_{k}^{(i)} \in [-1, 1]^7
  \]

- Physics evolves at a smaller simulation step `dt = cfg.sim.dt` for `cfg.decimation` steps per control step.

- Each agent receives an observation (delayed and noisy):
  \[
  o_{k}^{(i)} = \mathcal{H}^{(i)}\!\left(\{x_{k-\delta_{ij}}^{(j)}\}_{j=1}^{C},\ x_{k-\delta_{iT}}^{(T)};\ \eta\right)
  \]
  where:
  - `δ` models perspective-specific delays (ego vs inter-agent),
  - `η` represents observation noise and detection/comm dropouts.

Training typically optimizes expected discounted return:
\[
\max_{\pi}\ \mathbb{E}\left[\sum_{k=0}^{K-1}\gamma^k \sum_{i=1}^C r_k^{(i)}\right]
\]

---

## 2) Environment Structure and Dataflow

### 2.1 Core loop anatomy (V4)

In `IrisMAEnvV4` (`iris_ma_env4.py`), the core loop is organized as:

1. `_pre_physics_step(actions)`: curriculum progress + action preprocessing (decimated rate)
2. `_apply_action()`: applies forces/torques + gimbal targets at physics stepping
3. `_compute_intermediate_values()`: updates delay system, detection, comms (called from rewards)
4. `_get_rewards()`: clean pipeline reward computation
5. `_get_observations()`: noisy pipeline observation construction

The key architectural idea is a **dual-path delay system**:
- **Clean** delayed states for *rewards*
- **Noisy** delayed states for *observations*

This separation reduces reward corruption while preserving realism in the learning signal.

---

## 3) Action Space and Command Mapping

### 3.1 Action definition

Per agent, the action has 7 dimensions:

\[
a = [a_{vx}, a_{vy}, a_{vz}, a_{\dot\psi}, a_{\dot\alpha}, a_{\dot\beta}, a_{\dot z}]
\]

corresponding to:
- commanded linear velocity in world: `(vx, vy, vz)`
- commanded yaw rate: `\dot\psi`
- commanded gimbal yaw rate: `\dot\alpha`
- commanded gimbal pitch rate: `\dot\beta`
- commanded zoom rate: `\dot z`

### 3.2 Normalized → physical commands

In `_pre_physics_step`:

\[
v^{(i)}_{\text{cmd},k} = a^{(i)}_{k,0:3}\ \cdot\ v_{\max}
\]
\[
\dot\psi^{(i)}_{\text{cmd},k} = a^{(i)}_{k,3}\ \cdot\ \dot\psi_{\max}
\]
\[
\dot\alpha^{(i)}_{\text{cmd},k} = a^{(i)}_{k,4}\ \cdot\ \dot\alpha_{\max},\qquad
\dot\beta^{(i)}_{\text{cmd},k} = a^{(i)}_{k,5}\ \cdot\ \dot\beta_{\max}
\]
\[
\dot z^{(i)}_{\text{cmd},k} = a^{(i)}_{k,6}\ \cdot\ \dot z_{\max}
\]

Gimbal angles are integrated (clamped) at control rate:
\[
\alpha_{k+1} = \mathrm{clip}\big(\alpha_k + \dot\alpha_{\text{cmd},k}\ \Delta t,\ \alpha_{\min},\alpha_{\max}\big)
\]
\[
\beta_{k+1} = \mathrm{clip}\big(\beta_k + \dot\beta_{\text{cmd},k}\ \Delta t,\ \beta_{\min},\beta_{\max}\big)
\]

Zoom is integrated at physics rate in `_apply_action`:
\[
z_{m+1} = \mathrm{clip}(z_m + \dot z_{\text{cmd}}\ dt,\ 1,\ z_{\max})
\]
where `m` indexes physics substeps.

Note: In `iris_ma_env4.py`, `cmd_vel[...,6]` is already scaled by `cfg.max_zoom_rate`, and `_apply_action` multiplies again by `cfg.max_zoom_rate`. This yields an *effective* zoom rate scale of approximately `max_zoom_rate^2`. Treat this as “implementation is source of truth” unless you refactor it.

---

## 4) Robot Root Controller: PointMass

The environment applies external force/torque to the robot body using:
- `PointMass.compute_control_quat_exact` (`controller/point_mass.py`)

This controller is best interpreted as:
- velocity tracking in translation,
- quaternion-based stabilization in rotation,
- applied directly as body wrench `(F, τ)` through `set_external_force_and_torque`.

### 4.1 Translational velocity tracking (PID in body frame)

Define:
- world-frame velocity error:
  \[
  e_v = v_{\text{cmd}}^{w} - v^{w}
  \]
- body-frame error:
  \[
  e_v^{b} = R(q)^\top e_v
  \]
  where `R(q)` is the rotation matrix of quaternion `q = curr_quat_w`.

Integral accumulator (anti-windup clamp):
\[
I_v \leftarrow \mathrm{clip}(I_v + e_v\ \Delta t,\ -I_{\max}, I_{\max})
\]
\[
I_v^{b} = R(q)^\top I_v
\]

Force command (body frame):
\[
F^{b} = K_p e_v^{b} + K_i I_v^{b} - K_d a^{b}
\]
where `a^b = curr_lin_acc_b` is measured body acceleration.

### 4.2 Attitude stabilization via exact quaternion log map

The controller builds a fixed desired orientation:
- roll = π (to match the IRIS USD body frame convention),
- pitch = 0,
- yaw = 0.

Let:
\[
q_d = q(\phi=\pi,\theta=0,\psi=0),\qquad q = q_{\text{current}}
\]

Quaternion error:
\[
q_e = q_d \otimes q^{-1}
\]

Shortest-path sign correction:
\[
q_e \leftarrow \mathrm{sign}(w_e)\ q_e
\]

Exact rotation vector (logarithm map on `SO(3)`):
For unit quaternion \(q_e=[w,\mathbf{v}]\) with \(\mathbf{v}\in\mathbb{R}^3\),
\[
\theta/2 = \arccos(\mathrm{clip}(w,-1,1)),\qquad \|\mathbf{v}\|=\sin(\theta/2)
\]
\[
\mathbf{e}_\omega^{w} = 2\ \mathbf{v}\ \frac{\theta/2}{\|\mathbf{v}\|}
\]
with a small-angle fallback as \(\|\mathbf{v}\|\to 0\).

Transform to body frame:
\[
\mathbf{e}_\omega^{b} = R(q)^\top\ \mathbf{e}_\omega^{w}
\]

Moment (PD):
\[
\tau^{b} = -K_{p,\text{att}}\ \mathbf{e}_\omega^{b} - K_{d,\text{att}}\ \omega^{b}
\]

Yaw-rate tracking overwrites the z-axis moment:
\[
\tau_z^{b} = K_{p,\text{yaw}}\ (\omega_{\text{cmd},z}^{b} - \omega_{z}^{b})
\]

---

## 5) Gimbal Stabilization and Roll Compensation

The environment uses `GimbalStabilizer.compute_stabilizing_roll` (`controller/gimbal_stabilizer.py`) to keep the camera horizon level.

### 5.1 Pointing geometry

Given a desired pointing direction in world frame `d_w` (unit vector), the stabilizer transforms to gimbal base (drone body) frame:
\[
d_b = R(q)^\top d_w
\]

Using FLU axes `(x=forward, y=left, z=up)`:
- yaw:
  \[
  \alpha = \mathrm{atan2}(d_{b,y}, d_{b,x})
  \]
- pitch (positive is “down” in their sign convention):
  \[
  \beta = \mathrm{atan2}(-d_{b,z}, \sqrt{d_{b,x}^2 + d_{b,y}^2})
  \]

### 5.2 Stabilizing roll (horizon leveling)

Let `u_w = [0,0,1]` be world up, and:
\[
u_b = R(q)^\top u_w
\]

After undoing yaw (rotate by `-α` around the z-axis), you obtain `(u_x', u_y', u_z')`. After a pitch rotation β, the camera “up” vector depends on roll `ρ`. The implementation derives a robust formula via `atan2`:
\[
\rho = \mathrm{atan2}\left(-u_y',\ \frac{u_z'}{\cos(\beta)+\epsilon}\right)
\]
then clamps to roll limits.

---

## 6) Delay System V2 (MultiAgentDelaySystemV2)

V4 uses `MultiAgentDelaySystemV2` (`delay_system_v2/multi_agent_delay_system_v2.py`) built on a per-field delay pipeline (`delay_system_v2/delay_pipeline.py`).

### 6.1 Dual pipeline: clean vs noisy

The system maintains two logically distinct signal paths:
- Clean path: delayed states for reward computation
- Noisy path: delayed + noise + dropout for observations

Noise is injected before the delay pipeline (for the noisy path), scaled by a curriculum factor:
\[
\tilde x = x + \sigma\ \epsilon,\qquad \epsilon\sim\mathcal{N}(0, I)
\]
\[
\sigma_{\text{eff}} = \sigma\cdot s_{\text{noise}},\quad s_{\text{noise}}\in[0,1]
\]

### 6.2 Perspective-aware delays: ego vs other

For each agent, every field is stored twice:
- `agent_i.<field>.ego` (fast “self view”)
- `agent_i.<field>.other` (slower “received from others”)

This implements realistic asymmetry: you know your own state quickly, others’ state through comms.

### 6.3 Per-field delay pipeline stages

For a generic field \(x\) sampled at simulation time \(t\), the pipeline applies:

#### Stage 1: first-order lag (dynamics / filtering)
Implementation:
\[
x_f^{k+1} = x_f^{k} + \alpha (x^{k} - x_f^{k}),\qquad
\alpha = \frac{dt}{dt+\tau}
\]

#### Stage 2: staleness (sample-and-hold)
Hold the last sampled value until a (random) sample period elapses:
\[
x_s(t) = x(t_k)\quad \text{for } t\in[t_k, t_{k+1})
\]
with \(t_{k+1}-t_k\) sampled from a configured FPS distribution.

#### Stage 3: transport latency (history buffer)
Return a buffered past value:
\[
x_\ell^{k} = x_s^{k-d}
\]
where \(d \approx \lfloor \text{latency}/dt \rfloor\), sampled from a latency distribution.

#### Stage 4: dropout (sample freeze)
With probability \(p\), keep the previous output:
\[
x_d^{k} =
\begin{cases}
x_d^{k-1}, & \text{with prob. } p\\
x_\ell^{k}, & \text{with prob. } 1-p
\end{cases}
\]

### 6.4 Derived fields (computed after delays)

Derived quantities are recomputed from delayed raw fields (rather than being delayed themselves), including:

- camera position:
  \[
  p_c^w = p_b^w + R(q_b^w)^\top p_{c/b}^b
  \]
  (`compute_camera_position` uses quaternion-based rotation of the camera offset).

- camera orientation from body + gimbal:
  \[
  q_c^w = q_b^w \otimes q_{\text{gimbal}}(\alpha,\beta) \otimes q_{c/b}
  \]
  with a specific yaw-then-pitch quaternion construction in `compute_camera_orientation_from_gimbal`.

- ray directions from bbox:
  (see Section 8)

---

## 7) Camera Model and Detection (BBoxRayCaster)

### 7.1 Pinhole intrinsics (with zoom)

In `set_camera_configs`:
\[
f_x = f\ \frac{W}{A_h},\qquad f_y = f\ \frac{H}{A_v}
\]
\[
c_x = \frac{W}{2},\qquad c_y = \frac{H}{2}
\]
with:
- focal length `f` (mm),
- aperture `A_h, A_v` (mm),
- image size `(W,H)` (pixels).

Zoom is modeled as scaling focal lengths:
\[
f_x^{(\text{zoom})} = z\ f_x,\qquad f_y^{(\text{zoom})} = z\ f_y
\]
while the principal point remains unchanged.

### 7.2 3D point projection

For a 3D point in camera frame \(X_c=[X,Y,Z]\),
\[
u = f_x \frac{X}{Z} + c_x,\qquad v = f_y \frac{Y}{Z} + c_y
\]

### 7.3 Bounding boxes from projected 3D corners

`BBoxRayCaster` projects the 8 corners of the target’s 3D bounding box into each camera view, then computes:
\[
u_{\min}=\min_j u_j,\ \ u_{\max}=\max_j u_j,\qquad
v_{\min}=\min_j v_j,\ \ v_{\max}=\max_j v_j
\]

Converted to `(center_x, center_y, width, height)` (pixel space):
\[
c_x=\frac{u_{\min}+u_{\max}}{2},\quad c_y=\frac{v_{\min}+v_{\max}}{2},\quad
w=u_{\max}-u_{\min},\quad h=v_{\max}-v_{\min}
\]

Then normalized bbox is:
\[
\hat b = \left[\frac{c_x}{W},\frac{c_y}{H},\frac{w}{W},\frac{h}{H}\right]
\]

### 7.4 Detection validity

The raycaster validates detections using:
- corner visibility constraints (optionally partial detection),
- size constraints:
  \[
  (w,h)\in [w_{\min},w_{\max}]\times[h_{\min},h_{\max}]
  \]
- minimum bbox area (pixels),
- optional occlusion checks (disabled in default config for V4).

In the environment, a simpler validity check `validate_bbox` is also used to guard observation/reward terms.

---

## 8) Rays from 2D Bounding Boxes

The delay system derives a 3D ray direction from a bbox center `(u,v)`:

### 8.1 Unprojection to normalized camera ray

With intrinsics:
\[
x_n = \frac{u-c_x}{f_x},\qquad y_n = \frac{v-c_y}{f_y}
\]

Camera-frame ray (homogeneous):
\[
d_c = \begin{bmatrix}x_n\\y_n\\1\end{bmatrix},\qquad
\hat d_c = \frac{d_c}{\|d_c\|}
\]

### 8.2 Rotate to world frame

Let \(R_c^w\) be the camera rotation matrix from `camera_orientation_w`:
\[
\hat d_w = R_c^w\ \hat d_c
\]

When a bbox is invalid, the derived ray direction is set to zero to avoid contaminating downstream math.

---

## 9) Triangulation (Midpoint Method) and Covariance

V4 uses two triangulation notions:
- **Position estimate** from multiple rays for observations (`midpoint_method_batched`)
- **Uncertainty estimate** (covariance) for both rewards and observations (`triangulation_covariance_multi_camera` via `_compute_triangulation_covariance`)

### 9.1 Multi-ray midpoint triangulation (least squares)

For camera `i`, a ray is:
\[
X = p_i + \lambda_i\ d_i
\]
with origin \(p_i\) and unit direction \(d_i\).

Define the perpendicular projection matrix:
\[
P_i = I - d_i d_i^\top
\]

The midpoint method solves:
\[
\min_X \sum_i \|P_i (X - p_i)\|^2
\]
leading to the normal equations:
\[
A X = b,\qquad
A = \sum_i P_i,\quad b = \sum_i P_i p_i
\]

The implementation is batched, masks invalid cameras, and uses `torch.linalg.lstsq` for robustness. Validity requires:
- at least 2 valid cameras,
- `rank(A) = 3` (non-degenerate geometry),
- the solution is not behind all valid cameras (dot test).

### 9.2 Triangulation covariance (weighted least squares)

The covariance code in `triangulation/triang_cov_reward_torch.py` models uncertainty from:
- pixel noise (bbox center/size → pixel measurement error),
- optional pose and gimbal uncertainties,
- optional intrinsics uncertainties.

For each camera, define a 2D measurement function \(u = \pi(K, R, t, X)\).
Linearizing around nominal values:
\[
\delta u \approx J_X\ \delta X + J_\theta\ \delta\theta
\]

The weighted normal matrix (stacking all cameras) is:
\[
A = J_X^\top W J_X,\qquad W=\Sigma_{\text{pix}}^{-1}
\]

The implementation constructs a residual covariance that includes nuisance parameters \(\theta\) (pose, orientation, gimbal, intrinsics) and produces:
\[
\Sigma_X \approx A^{-1}\ (J_X^\top W\ \Sigma_{\text{resid}}\ W J_X)\ A^{-\top}
\]

The scalar “quality” proxy used by the environment is:
\[
\mathrm{trace\_cov} = \mathrm{tr}(\Sigma_X)
\]
and per-axis standard deviations are:
\[
\sigma = \sqrt{\mathrm{diag}(\Sigma_X)}
\]

---

## 10) Safety Subsystem (Collision + TTC)

The environment uses `SafetyManager` (`safety/safety_manager.py`) which wraps:
- `CollisionDetector` (`safety/collision_detector.py`)
- `TTCComputer` (`safety/ttc_computer.py`)

### 10.1 Inter-agent collision detection

Given agent positions \(p_i \in \mathbb{R}^3\), compute pairwise distances:
\[
D_{ij} = \|p_i - p_j\|
\]
and collisions:
\[
\text{collision}_{ij} = \left(D_{ij} < d_{\min}\right)
\]

Per-agent collision penalty is proportional to number of collisions:
\[
\phi_{\text{coll}}^{(i)} = -\sum_{j\neq i} \mathbb{1}[D_{ij} < d_{\min}]
\]

### 10.2 Time-to-collision (zoom-invariant looming)

The TTC model is based on the time derivative of a zoom-invariant quantity:

- bbox geometric mean size (in the implementation, `w,h` are bbox width/height in **pixels** from `bboxes_2d`, not necessarily normalized; the formulation still works as long as `f_x,f_y` are in the same units):
  \[
  s = \sqrt{w\ h}
  \]
- effective focal length (pixels):
  \[
  f = \sqrt{f_x f_y}
  \]
- zoom-invariant log ratio:
  \[
  g = \log(s) - \log(f) = \log\left(\frac{s}{f}\right)
  \]

Derivative:
\[
\dot g \approx \frac{g_k - g_{k-1}}{\Delta t}
\]

If approaching (\(\dot g < 0\)), TTC:
\[
\tau = -\frac{1}{\dot g}
\]

Mapped to a horizon `H` into a penalty in `[0,1]`:
\[
\phi_{\text{ttc}} = \mathrm{clip}\left(\frac{H - \min(\tau,H)}{H},\ 0,\ 1\right)
\]

This penalty is gated by:
- staleness decay for missing detections,
- a zoom-motion gate to avoid false positives during zoom transients.

---

## 11) Reward Function (Per-Agent)

Rewards are computed from **clean delayed states** (`get_all_states_for_rewards`).

Let `Δt = step_dt` and define:

### 11.1 Action penalties

Weighted action magnitude:
\[
r_{\text{act}} = s_{\text{act}}\ \Delta t\ \sum_{j=1}^7 (w_j a_j)^2
\]

Smoothness penalty:
\[
r_{\Delta\text{act}} = s_{\Delta\text{act}}\ \Delta t\ \sum_{j=1}^7 (\tilde w_j (a_j - a_{j,\text{prev}}))^2
\]

### 11.2 Image-space tracking rewards

With normalized bbox center \(c\in[0,1]^2\) and validity `v ∈ {0,1}`:
\[
r_{\text{center}} = s_{\text{center}}\ \Delta t\ \exp(-10\|c - 0.5\|_2)\ v
\]

BBox area (normalized):
\[
A = w\ h
\]
size shaping:
\[
r_{\text{size}} = s_{\text{size}}\ \Delta t\ \exp(-|A - 0.2|)\ v
\]

### 11.3 Triangulation quality reward

With trace covariance `tr = trace_cov` and validity `v_tri`:
\[
q_{\text{tri}} =
\begin{cases}
\frac{1}{1 + \frac{tr}{7}}, & v_{\text{tri}}=1 \\
0, & \text{otherwise}
\end{cases}
\]
\[
r_{\text{tri}} = s_{\text{tri}}\ \Delta t\ q_{\text{tri}}\ \cdot \text{progress}_{\text{coord}}
\]

### 11.4 Safety penalties (curriculum gated)

\[
r_{\text{coll}} = s_{\text{coll}}\ \Delta t\ \phi_{\text{coll}}\ \cdot \text{progress}_{\text{safety}}
\]
\[
r_{\text{ttc}} = s_{\text{ttc}}\ \Delta t\ \phi_{\text{ttc}}\ \cdot \text{progress}_{\text{safety}}
\]

Total reward is the sum of terms above.

---

## 12) Observation Vector (Per-Agent, 47D in default cfg)

Observations are constructed from **noisy delayed states** (`get_all_states_for_observations`).

For a 2-agent setup (`C=2`) the observation concatenates:

### 12.1 Ego features
- position \(p^w\) (3)
- yaw \(\psi\) from quaternion (1)
- linear velocity \(v^w\) (3)
- yaw rate \(\dot\psi\) from angular velocity z (1)
- linear acceleration (3)
- gimbal pitch, yaw (2)
- combined angular velocity (3)
- bbox normalized `(cx, cy, w, h)` (4)
- bbox valid (1)
- detection age (1)
- zoom level (1)
- ray direction (3)

### 12.2 Other-agent features (delayed via “other” perspective)
- other position (3)
- other linear velocity (3)
- other combined angular velocity (3)
- other bbox validity (1)
- other ray direction (3)
- other detection age (1) and data age (1) (duplicated in the current implementation)

### 12.3 Triangulation estimate + uncertainty
- triangulated target position (3), masked to ego position if invalid
- per-axis std dev from covariance (3)

This yields:
\[
16\ (\text{ego motion/att}) + 10\ (\text{ego vision}) + 15\ (\text{other}) + 6\ (\text{tri}) = 47
\]

---

## 13) Curriculum Scheduling

The environment uses linear progress ramps:
\[
\mathrm{progress}(k; k_s,k_e) = \mathrm{clip}\left(\frac{k-k_s}{k_e-k_s},\ 0,\ 1\right)
\]

In `_pre_physics_step`, these control:
- delay/noise scale (`progress_delay`),
- formation and target sampling difficulty (`progress_tracking`),
- triangulation reward enable (`progress_coord`),
- safety penalties enable (`progress_safety`),
- moving target aggressiveness (`progress_move`),
- dynamics randomization (`progress_dynamics`),
- zoom limit (`curriculum.get_max_zoom_level(step)`).

---

## 14) Randomization and Reset Geometry

At reset (`_reset_idx`):

- A formation type is selected based on `progress_tracking`:
  - planar → grid → line (increasing 3D complexity)
- Agent poses are sampled with constraints:
  - min/max separations,
  - curriculum-scaled vertical variation,
  - gimbal feasibility enforced by computing initial stabilized angles to the target and asserting limits.

Target position is sampled conditioned on formation geometry, then offset by environment origins.

---

## 15) Moving Target Dynamics (TargetMovement)

`TargetMovement` (`target_movement/target_movement.py`) generates a desired target velocity, with two modes:
- linear (random direction + speed),
- circular/orbit (random radius + angular speed).

The target velocity is tracked via an acceleration-based controller:

Let desired velocity be \(v_d\) and current be \(v\):
\[
a = k_a (v_d - v)
\]
with acceleration magnitude clamped:
\[
a \leftarrow a\ \min\left(1,\ \frac{a_{\max}}{\|a\|+\epsilon}\right)
\]

Then velocity update with damping:
\[
v \leftarrow (v + a\ dt)\ \cdot \lambda
\]
and speed clamped to `max_speed`.

Geofencing scales with curriculum:
\[
g = g_{\min} + \mathrm{progress}\ (g_{\max} - g_{\min})
\]
and is enforced by reflecting/steering the velocity back into a bounded region around each env origin.

---

## 16) Training Configuration Notes (SKRL MAPPO-RNN)

The included config `agents/skrl_mappo_rnn_cfg.yaml` suggests training under partial observability using a recurrent policy with sequence length 32.

MAPPO/PPO-style updates optimize the clipped objective (conceptually):
\[
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r(\theta)\hat A,\ \mathrm{clip}(r(\theta),1-\epsilon,1+\epsilon)\hat A\right)\right]
\]
with:
\[
r(\theta) = \frac{\pi_\theta(a|o)}{\pi_{\theta_{\text{old}}}(a|o)}
\]
and advantage estimates \(\hat A\) from GAE(\(\lambda\)).

The config also includes `episode_start_mask_steps`, which is commonly used to reduce loss impact in the first few steps of an episode (useful when the delay system makes early observations less informative).

---

## 17) Implementation Notes / Pitfalls

- Zoom rate scaling: as noted in Section 3.2, zoom appears to be scaled twice in the current implementation.
- Camera position computation: `compute_camera_position` uses `quat_rotate_inverse` to rotate the offset vector; treat the implementation as authoritative for frame conventions.
- Timestamp modeling: `timestamp_detection` is delayed identically to detection data (no smoothing), enabling explicit “detection age” features in the observation.
- Triangulation for rewards uses ground-truth target position `X_w_gt` but uncertainty from camera geometry/detections; this makes the triangulation term a *geometry quality* reward rather than an “estimation accuracy” reward.

---

## 18) Pointers to Source-of-Truth Code

- Core env: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/iris_ma_env4.py`
- Config: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/iris_ma_env4_cfg.py`
- Delay system: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/delay_system_v2/multi_agent_delay_system_v2.py`
- Per-field delay pipeline: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/delay_system_v2/delay_pipeline.py`
- Derived camera + rays: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/delay_system_v2/derived_field_computers.py`
- Detection: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/bbox_raycaster/bbox_raycaster.py`
- Triangulation + covariance: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/triangulation/triang_cov_reward_torch.py`
- Safety manager: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/safety/safety_manager.py`
- TTC: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/safety/ttc_computer.py`
- Target motion: `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/target_movement/target_movement.py`
