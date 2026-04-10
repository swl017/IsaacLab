# PPO/MAPPO Deep-Dive: Teaching Plan

> **Audience**: Strong optimal control background, basic deep RL knowledge.
> **Goal**: Build rigorous understanding of PPO/MAPPO, grounded in your iris_ma6 training runs.

---

## Chapter 1: Bridging Optimal Control to RL
**File**: `ch1_optimal_control_to_rl.md`

- Value function as cost-to-go (flipped sign)
- Policy gradient theorem as stochastic sensitivity analysis
- Why we can't solve Bellman directly -> function approximation -> deep RL
- Temporal difference learning as bootstrapped Bellman residual
- GAE: bias-variance tradeoff in advantage estimation
- **Sutton & Barto**: Ch 3 (MDP), Ch 6.1-6.3 (TD), Ch 13.1-13.3 (policy gradient)
- **Quiz**: TD vs Monte Carlo, GAE lambda interpolation

## Chapter 2: PPO -- The Algorithm
**File**: `ch2_ppo_algorithm.md`

- TRPO motivation: unconstrained gradient = divergence (Newton without step control)
- The clipped surrogate objective -- a practical trust region
- Value function clipping, entropy bonus, combined loss
- The update loop: rollout -> GAE -> mini-batch SGD epochs -> repeat
- SKRL source code walkthrough (exact file/line references)
- **Sutton & Barto**: Ch 13.4-13.5 (REINFORCE, baseline)
- **Quizzes**: clip behavior scenarios, loss sign interpretation

## Chapter 3: MAPPO & Your Training Dynamics
**File**: `ch3_mappo_training_dynamics.md`

- CTDE: centralized value, decentralized policy
- Parameter sharing vs independent learning
- RNN sequence processing (GRU in your setup)
- Phase-by-phase walkthrough of your training logs (from ppo_mappo_algorithm.md)
- The causal feedback diagram: sigma <-> KL <-> LR <-> advantages <-> reward

## Chapter 4: The KL-Adaptive LR & Collapse Analysis
**File**: `ch4_kl_adaptive_lr_collapse.md`

- KL divergence for Gaussians (full derivation)
- The KL-adaptive scheduler as automatic trust region
- Why wider action spaces amplify KL (from lr_collapse_analysis.md)
- The collapse cascade: positive feedback loop
- The fixes (min_lr floor, fewer epochs) and why they work
- **Quiz**: design collapse vs oscillation vs healthy scenarios

---

## Delivery Format
- **Chapters 1-2**: Written text with Sutton readings + quizzes (foundational, read with pencil/paper)
- **Chapters 3-4**: Interactive walkthrough of your training data and SKRL code (applied)

## Chapter 5: Practical Debugging Guide
**File**: `ch5_practical_debugging_guide.md`

- Every Tensorboard curve explained (PPO + iris_ma6 task metrics)
- Reward decomposition: reading individual reward terms, not just total
- Failure mode catalog: LR collapse, entropy collapse, value divergence, reward hacking, agent symmetry, curriculum-too-fast
- Decision tree: "I see X in Tensorboard -> check Y -> try Z"
- Hyperparameter cheat sheet
- Curriculum transition survival guide (expected impact per phase)
- **Quizzes**: Diagnose scenarios from Tensorboard symptoms

## Prerequisites
- Sutton & Barto textbook: `suttonbarto.pdf` (same folder)
- Training analysis: `ppo_mappo_algorithm.md`, `lr_collapse_analysis.md`
- SKRL source: installed in Isaac Lab environment
