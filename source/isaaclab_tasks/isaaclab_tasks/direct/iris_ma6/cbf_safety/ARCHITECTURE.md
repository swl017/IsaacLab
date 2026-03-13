# CBF Safety Filter Module Architecture

**Module:** `cbf_safety`
**Project:** iris_ma6 — Multi-Drone Active Triangulation
**Spec:** [safety_spec.md](../doc/safety_spec.md)

---

## 1. Overview

The CBF Safety Filter Module implements a two-layer safety architecture for multi-agent collision avoidance:

| Layer | Component | When Used | Data Source | Purpose |
|-------|-----------|-----------|-------------|---------|
| **Training** | CPARewardShaper | Training only | Ground truth | Soft penalty for reward shaping |
| **Deployment** | RobustDeploymentFilter | Deployment only | Delayed observations | Hard constraint on actions |

**Key Design Principle:** Training-time safety shaping and deployment-time safety filtering serve fundamentally different roles and use different mechanisms.

---

## 2. Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       CBFManager                            │
│                    (Facade Class)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌────────────────────┐               │
│  │ CPARewardShaper │  │ RobustDeployFilter │               │
│  │  (Training)     │  │   (Deployment)     │               │
│  │                 │  │                    │               │
│  │ • CPA barrier   │  │ • Distance barrier │               │
│  │ • GT positions  │  │ • Inflated margin  │               │
│  │ • Soft penalty  │  │ • Hard constraint  │               │
│  └─────────────────┘  └────────────────────┘               │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ CBFDiagnostics  │  Metrics: min_separation, collisions  │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

### 3.1 Training-Time: CPA Barrier

The Closest Point of Approach (CPA) barrier is velocity-aware:

**Time of closest approach:**
```
τ* = -Δp·Δv / (||Δv||² + ε)
τ = clamp(τ*, 0, T)
```

**CPA distance and barrier:**
```
d_CPA² = ||Δp + τ·Δv||²
h_CPA = d_CPA² - D_s²
```

**Discrete-time penalty:**
```
violation = ReLU((1 - γΔt)·h_current - h_next)
penalty = Σ violations across all pairs
```

**Why CPA over distance:** Two drones at 2.1m separation flying in parallel at 15 m/s have Δv ≈ 0, so d_CPA ≈ 2.1m — no penalty. A distance-based barrier would restrict any approach velocity regardless of direction.

### 3.2 Deployment-Time: Distance Barrier with Inflated Margin

**Inflated safety distance:**
```
D_deploy = D_s + v_max · (τ_delay + τ_PX4)
         ≈ 2.0 + 15.0 × 0.5
         ≈ 9.5 m
```

**Distance barrier:**
```
h_deploy = ||p_i - p_j||² - D_deploy²
```

**CBF constraint (halfspace):**
```
2·Δp^T·(v_i - v_j) + γ·h >= 0

Rearranged: a^T·v_i >= b
where:
  a = 2·Δp
  b = 2·Δp^T·v_j - γ·h
```

**Closed-form projection:**
```
if a^T·v < b:
    v_safe = v + (b - a^T·v) / ||a||² · a
```

---

## 4. Integration Points

### 4.1 Environment Integration

```python
# In __init__():
self.cbf_manager = CBFManager(
    cfg=self.cfg.cbf_safety,
    num_envs=self.num_envs,
    num_agents=len(cfg.possible_agents),
    device=self.device,
)

# In _get_rewards():
gt_positions = torch.stack([self._root_pos_w[id] for id in agents], dim=1)
cbf_penalty = self.cbf_manager.compute_training_penalty(gt_positions, cmd_vel, dt)
reward = task_reward - lambda_cbf * cbf_penalty

# In _get_dones():
collided = self.cbf_manager.check_collisions(gt_positions)
terminated = crashed | collided

# In _reset_idx():
self.cbf_manager.reset(env_ids)
```

### 4.2 Data Flow

```
Policy Output (v_nom)
       │
       ▼
[TRAINING MODE]                    [DEPLOYMENT MODE]
       │                                  │
       │                                  ▼
       │                     RobustDeploymentFilter
       │                     (delayed positions)
       │                                  │
       ▼                                  ▼
   Unfiltered                         Filtered
   v_nom → Sim                     v_safe → PX4
       │
       ▼
CPARewardShaper (GT)
       │
       ▼
   Penalty → Reward
```

---

## 5. Configuration

### 5.1 Training Parameters (CPARewardShaperCfg)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D_s` | 2.0 m | Physical safety distance |
| `gamma` | 2.0 | CBF decay rate |
| `T` | 1.0 s | CPA look-ahead horizon |
| `lambda_cbf` | 1.0 | Penalty weight |

### 5.2 Deployment Parameters (RobustDeploymentFilterCfg)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D_s` | 2.0 m | Physical safety distance |
| `v_max` | 15.0 m/s | Maximum agent velocity |
| `tau_delay_max` | 0.2 s | Max communication delay |
| `tau_px4` | 0.3 s | PX4 tracking lag |
| `gamma_deploy` | 1.0 | Conservative decay rate |
| `num_iters` | 2 | Gauss-Seidel iterations |

**Computed:** `D_deploy = D_s + v_max × (tau_delay + tau_px4) ≈ 9.5 m`

### 5.3 Manager Configuration (CBFManagerCfg)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_training_penalty` | True | Enable CPA penalty |
| `enable_deployment_filter` | False | Enable hard filter |
| `enable_collision_termination` | True | Terminate on collision |
| `collision_distance` | 2.0 m | Termination threshold |

---

## 6. File Structure

```
cbf_safety/
├── __init__.py              # Module exports
├── cbf_cfg.py               # Configuration dataclasses
├── cbf_diagnostics.py       # Training metrics
├── cbf_manager.py           # Facade class
├── cpa_reward_shaper.py     # Training-time CPA penalty
├── deploy_filter.py         # Deployment-time hard filter
├── ARCHITECTURE.md          # This file
└── tests/
    ├── __init__.py
    ├── run_tests.py         # AppLauncher-based test runner
    └── README.md
```

---

## 7. Safety Layers

The complete safety architecture has four layers:

| Layer | Mechanism | Data | Guarantee |
|-------|-----------|------|-----------|
| L0 | CPA penalty | GT | Learned avoidance (training) |
| L1 | MAPPO-RNN policy | Delayed obs | Trained behavior |
| L2 | Distance-CBF filter | Delayed obs | Forward invariance (deploy) |
| L3 | PX4 failsafes | Onboard | Firmware safety |

**Key insight:** The MAPPO-RNN, trained with GT-based safety rewards, is the primary collision avoidance mechanism. The deployment filter is a safety net, not the main controller.

---

## 8. Performance Considerations

### 8.1 GPU Efficiency

All computations are fully batched:
- Pairwise distances: O(E × N²) via broadcasting
- CPA computation: O(E × P) where P = N(N-1)/2 pairs
- No Python loops over environments

### 8.2 Memory

Pre-allocated buffers:
- Distance matrices: (E, N, N)
- Filter active flags: (E, N)
- Pair indices: Computed once at init

---

## 9. Testing

```bash
# Run all tests
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/run_tests.py

# Verbose output
./isaaclab.sh -p ... --test-verbose

# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p ...
```

Test categories:
1. **CPARewardShaper:** Stationary, approaching, parallel flight, numerical stability
2. **RobustDeploymentFilter:** No constraint, single/multi constraint, conservative
3. **CBFManager:** Training mode, deployment mode, collision detection, reset

---

## 10. References

- safety_spec.md — Full specification with mathematical derivation
- Ames et al. "Control Barrier Functions: Theory and Applications" (ECC 2019)
- Qin et al. "GCBF+: Neural Graph Control Barrier Functions" (T-RO 2024)
