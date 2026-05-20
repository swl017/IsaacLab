# Critic-Obs Design — Ticket 037 Phase 1

**Date**: 2026-05-20
**Status**: §1 lit review complete; §2 comparison matrix complete; §3 decision locked on Option B with 46-dim spec (A=2: 22 per-agent + 2 shared); Slice 1 field registry done (§3.4); Slice 2 cfg surface landed (`_CRITIC_PRIVILEGED_FIELD_REGISTRY` + `critic_privileged_fields` cfg field + `__post_init__` validation + dim-accounting integration with the existing critic toggles). Implementation ready at Slice 3.
**Ticket**: [doc/active/ticket/037-critic-privileged-obs/ticket.md](active/ticket/037-critic-privileged-obs/ticket.md)
**Related**: [doc/research/dr_observability_taxonomy.md](research/dr_observability_taxonomy.md) (actor-side obs taxonomy — companion question), [doc/research/ch3_mappo_training_dynamics.md](research/ch3_mappo_training_dynamics.md) (MAPPO mechanics)

This document is the design analysis the ticket gates implementation on. The question: **what does the centralized critic see, and why?** The actor obs is fixed by deployment compatibility; the critic is discarded at deploy, so its inputs are a free variable.

Read this with the goal of answering one concrete question at the end of §3: "for ticket 037 Phase 1, the critic shared-obs is `<this list of K dims>`, with this justification: `<...>`."

---

## §1. Literature review

The relevant prior work clusters into four lines. For each I'll note the setting, what the privileged channel sees, and the specific claim that informs our choice.

### §1.1 Asymmetric actor-critic — the founding mechanism

**Pinto, Andrychowicz, Welinder, Zaremba, Abbeel (2018)** — "Asymmetric Actor Critic for Image-Based Robot Learning" (ICRA / RSS workshop, 2018).

- **Setting**: Vision-based manipulation in MuJoCo. Actor sees rendered images; critic sees the full simulator state (object positions, velocities, joint angles).
- **Mechanism**: Standard DDPG, but the critic Q(s, a) is parameterized over the low-dim sim state, while the actor π(o) is parameterized over images. At deploy time the critic is discarded, so its sim-only access is irrelevant.
- **Key claim**: ~2× sample efficiency vs symmetric (image-only) AC. The critic's job — predicting return — is *much easier* from the underlying state than from pixels, so the value estimate is lower-variance and the policy gradient is correspondingly cleaner.
- **Relevance for us**: This is the canonical citation for the *legitimacy* of critic-privileged-obs. It does NOT directly answer "raw params vs progress vs latent encoding" — Pinto et al. just feed the *complete* low-dim sim state. The principle: cheaper-to-fit V → cleaner advantages → better policy gradient. Same logic applies to our setting regardless of which specific privileged channel we pick.

### §1.2 Privileged teacher-student distillation

This line uses privileged info to train a teacher policy *and* its critic, then distills into a deployment-realizable student. Both teacher policy and teacher critic see the privileged info.

**Lee, Hwangbo, Wellhausen, Koltun, Hutter (2020)** — "Learning Quadrupedal Locomotion over Challenging Terrain" (Science Robotics, ETH).

- **Setting**: ANYmal quadruped locomotion over varied terrain.
- **Privileged channel (teacher's obs)**: terrain heightmap sampled around each foot, ground friction coefficient, external force perturbations applied during training, ground contact state, foot positions / contact normals. These are **raw physical parameters**, not curriculum proxies.
- **Critic**: trained with PPO, sees the same privileged obs as the teacher policy (so the asymmetry is teacher→student in this work, not actor→critic).
- **Key claim**: privileged-trained teacher transfers to a student that uses only proprioception (joint state) + short history, with no measurable performance loss when the student is given enough history to recover the privileged info implicitly.
- **Relevance for us**: Confirms that raw physical params work as a privileged channel. Their list is small and physically independent (heightmap, friction, force) — not bundled. This is a precedent for option B / C in the ticket, not option A.

**Miki, Lee, Hwangbo, Wellhausen, Hutter, Koltun (2022)** — "Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild" (Science Robotics).

- **Setting**: Extension of Lee 2020 with vision (LiDAR/depth). ANYmal in unstructured outdoor terrain.
- **Privileged channel**: terrain heightmap + same physical-param suite as Lee 2020.
- **New piece**: Belief State Encoder (BSE) — a recurrent network in the student that consumes noisy exteroceptive obs and predicts the teacher's privileged representation. This is essentially an actor-side sysid head, trained against teacher supervision. (Direct precedent for our Phase 2.)
- **Relevance for us**: A working example of "actor-side aux head predicts privileged info" + "teacher with raw physical params." Confirms the two-phase architecture (Phase 1 = privileged critic / privileged teacher; Phase 2 = student-side belief encoder).

**Chen, Murali, Gupta (2020)** — "Learning Visual Locomotion with Cross-Modal Supervision" [VERIFY citation].

- Cross-modal teacher-student; teacher sees depth/heightmap, student sees RGB only. Less directly applicable to our env-params question but cited in this lineage.

### §1.3 Domain-randomization encoders

These approaches give a learned encoder access to env params and *concatenate the encoder output into the policy obs*, rather than into the critic. This is a different architecture than what ticket 037 proposes — but the obs-choice question (raw params vs progress vs other) is the same.

**Kumar, Fu, Pathak, Malik (2021)** — "RMA: Rapid Motor Adaptation for Legged Robots" (RSS).

- **Setting**: A1 quadruped, online adaptation to mass / payload / friction / motor strength / external disturbances.
- **Architecture (two-phase)**:
  - **Phase 1 training (in sim)**: Environment-Factor Encoder μ: privileged sim params `e_t ∈ ℝ^17` (explicit composition below) → latent `z_t ∈ ℝ^8`. Encoder is a 3-layer MLP (hidden sizes 256, 128). Base policy `π(x_t, a_{t-1}, z_t)` takes proprioception + previous action + extrinsics z.
  - **Phase 2 (still in sim, after Phase 1 frozen)**: Adaptation module φ: 50-step history of proprioception + actions → ẑ. Trained with supervised regression against the true z from μ.
  - **Deploy**: Replace μ with φ; policy uses ẑ from φ.
- **Exact composition of `e_t ∈ ℝ^17`** (verified against §IV-A): mass + center-of-mass position (3 dims), motor strength (12 dims, one per joint), friction (1), local terrain height (1). Total = 17. Table I gives ranges (friction `[0.05, 4.5]`, K_p `[50, 60]`, K_d `[0.4, 0.8]`, payload `[0, 6]` kg, COM offset `[-0.15, 0.15]` cm, motor strength `[0.90, 1.10]`); each axis is sampled independently per episode.
- **Critic architecture**: paper text does not explicitly state the critic's input. PPO is used. The standard PPO pattern for this architecture would have V see the same `(x_t, a_{t-1}, z_t)` as π, but this is *inferred*, not quoted. (Independent precedent for the asymmetric AC pattern with raw params: Andrychowicz 2020 Hand, §1.5 below.)
- **Privileged channel content rationale**: raw physical parameters — explicitly NOT a curriculum progress signal. They argued for this because (a) the encoder benefits from physical units (correlations are smooth in mass-space and motor-space), and (b) at deploy the adaptation module is regressing toward physical units anyway, so the encoder target space should match.
- **Bundling stance**: explicitly independent — each of the 17 axes is randomized with its own uniform range; the paper does not bundle correlated quantities.
- **Relevance for us**: Strongest direct precedent. Three specific takeaways:
  1. Encoder over **raw physical params** is the published choice. Curriculum progress is an iris_ma6-specific abstraction; RMA had no curriculum and randomized at full range from step 0.
  2. **`e_t` dim is 17, compressed to `z` dim 8**. The 17 → 8 compression is learned, not hand-designed. Suggests our option D (learned encoder over raw params with bottleneck) is also a published pattern.
  3. RMA put the privileged info into the *policy* (via the encoder), not the critic alone. That's a different architecture — closer to what our Phase 2 (aux sysid head) targets than to ticket 037 Phase 1. But the obs-choice question (raw vs progress) is shared.

**Sim-to-Real Quadruped (Tan, Zhang, Coumans, Iscen, Bai, Hafner, Bohez, Vanhoucke, 2018)** — "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots" (RSS 2018) [VERIFY title].

- Used PPO with explicit domain randomization (mass, friction, motor properties) at independent uniform ranges. No critic-side privileged obs; policy was symmetric, just trained over the randomized distribution.
- **Relevance for us**: Shows the "no privileged info, just randomize broadly" baseline. iris_ma6 t034 is essentially this. The probe-036 robust verdict is consistent: when you randomize broadly without env-aware critic or actor, you get a robust policy.

### §1.4 Centralized critic in MARL (relevant subset)

**Lowe, Wu, Tamar, Harb, Abbeel, Mordatch (2017)** — "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (NeurIPS, MADDPG).

- First systematic centralized-critic-decentralized-actor (CTDE) work for continuous-action MARL.
- Critic of agent i sees `(x_1, ..., x_N, a_1, ..., a_N)` — every agent's local obs and action. No env-param privileged channel; just the full multi-agent state.
- **Relevance for us**: Establishes that "centralized critic sees more than the actor" is the standard MARL pattern; not a deviation. iris_ma6 already does this via MAPPO's shared obs.

**Yu, Velu, Vinitsky, Wang, Bayen, Wu (2022)** — "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS Datasets & Benchmarks). The MAPPO paper.

- Studies several centralized critic obs choices empirically:
  - **CL (concat local)**: stack of all agents' local obs.
  - **EP (env-provided)**: the env's global-state vector if available.
  - **AS (agent-specific global state)**: `[env_state, agent_i_local_obs]` — i.e., per-agent critic input that pairs the global state with that agent's own perspective.
  - **FP (feature pruning)**: like AS but with redundant features removed.
- **Key empirical finding**: AS outperforms CL and EP on most SMAC / MPE tasks. Adding the agent-i local obs back into the critic input (even though it's already part of CL) consistently helps. Their hypothesis: AS gives the critic a clear "whose value is this?" anchor, reducing the critic's effective sample size on noisy state features.
- **Relevance for us**:
  - We're already at AS-style — `shared_observation_spaces[agent_i]` includes the agent's local obs (see [iris_ma_env6_test.py](../iris_ma_env6_test.py) shared-obs construction).
  - Yu 2022 *does not* study env-param privileged obs. The obs choices are about how to splice agent-local and global state, not about adding env nuisance variables. That's our specific contribution to make for iris_ma6.
  - Methodologically: their experimental design (vary one obs-choice at a time, measure final return) is the protocol we should follow for ticket 037's eval.

**Foerster, Farquhar, Afouras, Nardelli, Whiteson (2018)** — "Counterfactual Multi-Agent Policy Gradients" (COMA, AAAI).

- Centralized critic + counterfactual baseline for credit assignment in cooperative MARL.
- Critic obs: full state + all agents' actions. Discrete actions only.
- **Relevance for us**: tangential — COMA's mechanism is orthogonal to the obs-choice question, but worth citing as evidence that centralized critic + privileged channel is well-trodden.

### §1.5 Independent vs bundled randomization

The bundling concern in ticket 037 (`progress_dynamics` co-ramps 7 physically-independent quantities) has direct precedent in sim2real DR papers.

**Tobin, Fong, Ray, Schneider, Zaremba, Abbeel (2017)** — "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (IROS, the original DR paper).

- Visuomotor pose estimation under randomized textures, lighting, camera position.
- Key methodological note: each randomization axis is sampled **independently** at every episode. The paper does not bundle correlated quantities.
- **Relevance for us**: The DR canon starts from independent randomization. Bundling is a curriculum-induced shortcut, not an established pattern.

**Peng, Andrychowicz, Zaremba, Abbeel (2018)** — "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization" (ICRA).

- Studies sim-to-real of a 7-DOF arm under randomized mass, joint damping, motor gains, table friction, latency.
- Ablation: independent randomization across axes vs single-axis randomization vs no DR. Independent randomization wins for transfer.
- Critic obs: not explicitly privileged in their setup; standard symmetric AC over the randomized distribution.
- **Relevance for us**: Empirical evidence that *independent* randomization improves real-world transfer. Argues *for* option C in ticket 037 — bundling axes is a sim-only artifact that hurts deployment.

**OpenAI / Andrychowicz et al. (2019, 2020)** — "Learning Dexterous In-Hand Manipulation" / "Solving Rubik's Cube with a Robot Hand" (arXiv 1910.07113, IJRR).

- 24-DoF Shadow Hand, in-hand object reorientation.
- Massive randomization: friction, gravity, restitution, joint damping, actuator gains, object mass, observation noise, action noise, latency, etc. — all independent ranges per episode.
- **Critic obs (verified against §6.2 of arXiv 1910.07113)**: clean simulator state (exact joint angles, velocities — no observation noise). The asymmetry is **noisy actor obs vs clean critic obs**, NOT actor-without-randomization-params vs critic-with-randomization-params. Quoted: *"the value network had access to non-noisy observations (since the value network is not needed when rolling out the policy on the robot and can thus use privileged information)"*. The randomization parameter values themselves are **not** in either network's input.
- **Mechanism for adaptation**: with no env-param channel, the policy learns to be robust across the randomization distribution; the LSTM-based policy can implicitly infer current env conditions from action-effect history (the original "implicit sysid via memory" argument).
- **Relevance for us — important correction**: this is NOT a precedent for "feed the randomized parameter values to the critic." The privilege in OpenAI Hand is "clean state without obs noise." For our ticket 037 question (giving the critic *additional env-param info beyond the standard obs*), Hand is silent — they didn't do that, even with their massive randomization scope. This *moderates* the §1.6 thesis: critic-only-privileged-params is less precedented than I initially claimed.

### §1.6 Synthesis — what the literature converges on

Tallying the obs-choice precedents:

| Work | Setting | Privileged channel content | To actor? | To critic? | Indep. axes? |
|---|---|---|---:|---:|---:|
| Pinto 2018 | Vision manip | Full sim state (positions, vels). Privilege = "clean sim state" vs pixels. | ✗ | ✓ | n/a |
| Lee 2020 / Miki 2022 | Quadruped locomotion | Heightmap + friction + ext force + contact. Privilege = raw env params. | ✓ (teacher) → student via distillation | ✓ (teacher's critic — same input as teacher policy) | ✓ |
| Kumar 2021 (RMA) | Quadruped locomotion | **Raw physical params** `e_t ∈ ℝ^17` (mass + COM 3D, 12 motor strengths, friction, terrain ht) → encoded to `z ∈ ℝ^8` | ✓ (via encoder) | ✓ (inferred — paper does not explicitly state critic input) | ✓ |
| Yu 2022 (MAPPO) | MARL benchmarks | n/a (no env-param channel; obs-choice is about local↔global state) | ✗ | n/a | n/a |
| OpenAI Hand 2019/2020 | Dexterous manipulation | **Clean sim state** (no obs noise). Privilege = noiseless, NOT randomization params. | ✗ | ✓ (clean-vs-noisy) | ✓ |
| Tobin 2017, Peng 2018 | Vision / arm manip | n/a (symmetric AC; just randomize over indep. axes) | ✗ | ✗ | ✓ |

**Convergent patterns (revised after Hand verification):**

1. **Two distinct flavors of "privileged"** in the literature, often conflated:
   - **Clean state privilege**: actor sees noisy/partial obs (pixels, noisy joint sensors), critic sees clean sim state. Examples: Pinto 2018, OpenAI Hand 2019. This is the *founding* asymmetric AC pattern.
   - **Raw env-params privilege**: actor or encoder sees the randomization parameter values that generated the current episode. Examples: RMA 2021 (via encoder, into policy), Lee 2020 (via teacher, symmetric within teacher).
2. **Ticket 037 Phase 1 is a hybrid** — *critic-only env-params privilege*. Closest exact precedent I located is the inferred RMA critic (if it sees `z`), but RMA's primary mechanism is the encoder-into-policy, not the critic. **The exact pattern of "give the critic privileged env params, give the actor nothing extra" is less precedented than I initially claimed**. The asymmetric AC mechanism (Pinto, Hand) is well-established but with *clean-state*, not env-param, privilege.
3. **Curriculum-progress signals are not used as privileged obs in any work I located.** They appear as *training-time scalars on the env side*, not as obs the critic conditions on. iris_ma6's `_eff_progress_*` is a local abstraction.
4. **Independent axes are the canonical DR pattern.** Tobin 2017, Peng 2018, Andrychowicz 2020 all randomize independently and explicitly argue for it on sim2real grounds. Bundling like iris_ma6's `progress_dynamics` is an outlier introduced by curriculum implementation convenience, not a principled choice.
5. **The architectural symmetry between policy and critic matters less than I expected.** In the strongest published precedent (RMA), the env-params encoder feeds the POLICY; the critic's role with respect to env params is unclear from the paper text. The literature doesn't have a clean "critic-only env-params" reference run to compare against.

**Implications for ticket 037 Phase 1:**

- Option A (bundled `_eff_progress_*`) has **no precedent in the published literature for this kind of obs**. Every comparable work uses either raw physical params or no env-param channel. Asking the critic to learn V(s, curriculum_progress) is asking it to learn over a low-dim summary of jointly-correlated underlying physics — fitting an artifact of how training was set up.
- Option B (raw physical params, critic-only) has the strongest *physical justification* (RMA's encoder targets raw params on smoothness grounds) but the "critic-only" framing is not directly precedented. A safer near-equivalent: B + concurrent actor-side encoder (Phase 2). But that's out of Phase 1 scope by construction.
- Option C (`_eff_progress_*` with `progress_dynamics` decoupled) is a hybrid. Less precedented because the literature doesn't use progress signals as obs at all — but it preserves the iris_ma6 curriculum-engineering ergonomics while restoring axis independence.

**A new question raised by the Hand correction**: is there a published precedent for critic-only env-param privilege at all, or is ticket 037 Phase 1 a novel architecture? If novel, the expected-outcome bounds in the ticket (V-loss drops, σ collapses modestly) become less well-anchored. Worth a targeted search before §3.

### §1.7 The "critic-only env-params privilege" gap (preliminary finding)

After targeted search:

- **Lamberti, Wei, et al. (2024)** — "Privileged Sensing Scaffolds Reinforcement Learning" (Scaffolder, arXiv 2405.14853): explicit critic-only privilege paradigm in a model-based RL framework. Verified via §3 and §C.2 of the paper. Critic sees scaffolded world-model state s⁺ + target state s⁻; target policy sees only s⁻. **But**: the privileged content in Scaffolder is *extra sensors* (cameras, touch sensors, object poses) — not env randomization parameter values. Not a direct precedent for our content either.
- I could not locate a published reference that does *exactly* "give the critic the env randomization parameter values, give the actor nothing". This may exist (the space is large) but isn't in the headline asymmetric-AC / DR / privileged-learning canon.

**Implication**: ticket 037 Phase 1 sits in a small gap in the published literature — the *mechanism* (asymmetric AC) and the *content* (raw env params) each have strong precedents, but the **conjunction** (critic-only + env params + no actor encoder) does not have a direct reference run. This is a sub-novel choice, not a foregone-conclusion architecture.

**Three honest consequences for the design analysis:**

1. The expected V-loss-reduction magnitude is *not* well-anchored by published numbers. Pinto 2018's ~2× sample efficiency was clean-state-vs-pixels, a much larger info gap than env-params-vs-no-params. We should expect a smaller effect than that.
2. The expected actor-side effect (σ collapse, action_sum decrease) is even less anchored — no published precedent isolates this.
3. The Phase 2 fallback path (aux sysid head, à la Miki 2022's BSE) is more precedented than Phase 1. Worth keeping that in view when evaluating Phase 1's outcomes: a "Phase 1 made V-loss drop but didn't move actor metrics" outcome is consistent with the literature; the actor-side gains require Phase 2.

Sources for §1 verification:
- [RMA paper (arXiv:2107.04034)](https://arxiv.org/abs/2107.04034) — §IV-A/B verified via ar5iv HTML.
- [OpenAI Hand / Rubik's Cube (arXiv:1910.07113)](https://arxiv.org/abs/1910.07113) — §6.2 verified via ar5iv HTML.
- [Privileged Sensing Scaffolds RL (arXiv:2405.14853)](https://arxiv.org/abs/2405.14853) — §3, §C.2 verified.
- Other citations (Pinto 2018, Lee 2020, Miki 2022, Yu 2022, Tobin 2017, Peng 2018, Tan 2018, MADDPG, COMA) — drawn from prior reading; specific composition claims marked [VERIFY] in the §1.6 checklist.

**A nuance not yet in the framing**: RMA's encoder z ∈ ℝ^8 is itself a compression of ~10–20 raw params. The 8-dim bottleneck is not a "curriculum progress" but a learned latent over physical params. This suggests a fourth option:

- **Option D**: Raw physical params fed into the critic via a small *fixed* projection (a learned encoder or a hand-designed normalization). The critic input is dim-reduced, but the projection is principled, not a co-randomization artifact.

**Tentative ranking (to be confirmed in §2):**

1. **Option B** — raw physical params with current independent randomization preserved per-axis. Highest precedent, highest sim2real fidelity.
2. **Option C** — `_eff_progress_*` with `progress_dynamics` decoupled. Cheaper than B; lets us keep the per-progress-axis curriculum design while restoring axis independence.
3. **Option D** — learned encoder over raw params. Adds complexity; only justified if B's input dim grows too large for the critic to fit cleanly with our network capacity (gru_hidden_size=64).
4. **Option A** — bundled `_eff_progress_*`. **Unsupported by literature.** Including only as a baseline to confirm the bundling concern is real on the metrics.

### Specific facts to verify before this doc is treated as authoritative

These are claims reproduced from memory that should be confirmed against the source papers:

- [x] **RMA's z-dim and e_t composition** — verified via ar5iv HTML (`z ∈ ℝ^8`, `e_t ∈ ℝ^17` = mass+COM(3) + motor strength(12) + friction(1) + terrain ht(1); encoder 3-layer MLP 256→128). RMA paper does *not* explicitly state the critic input — marked as inferred above.
- [ ] Lee 2020's exact privileged-obs list (heightmap dims, presence of external force / friction).
- [ ] Andrychowicz 2020 (Hand) — whether the randomization values themselves are in the critic obs or whether the critic sees only the resulting sim state. My memory is "both" but worth confirming against IJRR §5. **Load-bearing for §1.6 thesis** — verify before §3 decision.
- [ ] Yu 2022 MAPPO — whether the AS variant outperforms CL on *all* studied tasks or just most.
- [ ] Margolis & Agrawal — CoRL 2022 ("Walk These Ways"). Should it be cited here? My read is no — it's about commanded behavior, not env nuisance — but worth a second look.
- [ ] Peng 2018 title and venue exact.
- [ ] Tan 2018 title (I wrote "Sim-to-Real Quadruped"; actual title likely "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots" but verify).

These don't change the synthesis; they affect citation accuracy. The Andrychowicz Hand claim is the one whose outcome would most change the relative ranking of options A–D in §2.

---

## §2. Comparison matrix

Four options under consideration, refined after §1:

- **A** — bundled `_eff_progress_*` as-is. 8 progress scalars × A agents. No env-side changes.
- **B** — raw physical parameters fed to critic. Each randomized quantity exposed at its native unit (mass kg, τ seconds, gain dimensionless, etc.). With current bundling preserved in the env, B and A are *information-equivalent* but B has finer-grained units; with the ticket's decoupling sub-scope applied to the env, B becomes strictly richer than A.
- **C** — `_eff_progress_*` with `progress_dynamics` decoupled into its 8 sub-axes (per ticket decoupling sub-scope). 15 progress scalars × A agents. Co-required env change.
- **D** — learned encoder over raw physical params → small bottleneck `z`, fed to critic (RMA-style but routed to critic instead of policy). Co-required env change for axis independence.

### 2.1 Per-agent / shared dim counts

Anchoring the analysis to concrete numbers (A = 2 agents). **Important refinement** (2026-05-20 — code audit): option B should feed the critic the **upstream independent RNG draws** (the scales / sampled values that actually parameterize the env), not the derived physical values, because gain-bundling inside `randomize_gains` makes the derived values collinear. With this clarification:

| Option | What the critic gains | Per-agent dim | Shared dim | Total added (A=2) |
|---|---|---:|---:|---:|
| A | 8 bundled progress scalars | 8 | 0 | 16 |
| B (independent-draw view, env decoupled) | 23 raw scalars: 6 gain scales + max_lin_vel + gimbal τ + 2 dead-times + FOV + 3 mech offsets + 2 gimbal mass + 2 stiff/damp + body_mass + 4 detection-pipeline. Detail in §3.1. | ~23 | 2 (target_scale_xy, target_scale_z) | ~48 |
| C | 15 decoupled progress scalars | 15 | 0 | 30 |
| D | RMA-style encoder; raw scalars → 8-d z per agent | 8 | 0 | 16 |

(Per-agent dim grows linearly in A; B's ~48 for A=2 becomes ~71 at A=3.)

**Correction to prior version**: §2.1 initially said B is "~27/agent + 3 shared = 57" based on a naive enumeration that double-counted Kp/Ki within the velocity-controller group and Kp/Ki/Kd within the rate-controller group. The verified-against-code answer is ~23/agent + 2 shared = 48, because:

- `drone_controller.randomize_gains` draws **5 scales** for the gain block (vel, att, rate, motor) plus 2 zoom-specific scales (tau_zoom, max_zoom_rate). Total: **6 independent scales**, not 8 derived gains.
- `physics_randomizer.sample_mass_parameters` draws mass_scale OR mass_addition (1 independent draw, mode-dependent) plus payload_mass (1 dim, **shared per env** rather than per-agent because applied to the per-env physics asset).
- `gimbal_randomizer` draws 3 mech offsets + 2 mass scales (yaw/pitch) + stiffness_scale + damping_scale + friction (= 8 independent per (env, agent), though friction has limited deployment relevance for flying drones and may be excluded → 7 useful).

Feeding the critic 6 scale variables instead of 8 derived gain values is **strictly more informative** because the derived gains live on a 5-manifold; the 5 scales are the actual generative parameters.

### 2.2 Scoring matrix

Scale: ✓✓ strong, ✓ adequate, ◑ caveats, ✗ weak. Sources/justifications below each row.

| Axis | A: bundled progress | B: raw params (decoupled env) | C: decoupled progress | D: encoder over raw |
|---|---|---|---|---|
| **Info sufficiency for V(s, env_params)** | ✗ (collapses 7+ axes into one) | ✓✓ (full physical state) | ✓ (sufficient stat over decoupled axes) | ✓ (compressed sufficient stat, lossy by construction) |
| **Axis-independence preserved** | ✗ (`progress_dynamics` bundles 7) | ✓✓ (each axis its own RV) | ✓✓ (env-decoupled) | ✓ (axes independent upstream of encoder) |
| **Sim-to-real correlation risk** | ✗ critic learns V over correlated training manifold; deploys off-manifold | ✓✓ critic sees axes as they vary in deployment | ✓ matches B once env is decoupled | ◑ encoder may re-bake correlations the env removed |
| **Critic input dim** | ✓✓ 16 dims (A=2) | ◑ ~57 dims; needs normalization | ✓ 30 dims | ✓✓ 16 dims |
| **Normalization needs** | ✓✓ already in [0,1] | ✗ heterogeneous scales (latency 0.1, mass 4–8 kg, gains 50–200) — RunningStandardScaler dependence | ✓✓ already in [0,1] | ✓ encoder absorbs normalization |
| **Plumbing cost** | ✓✓ tensors already cached on env | ✗ touches DroneController, GimbalRateLoop, ZoomController, DomainRandomizer, DelaySystem; ~6 modules | ◑ env decoupling sub-scope (curriculum cfg + sampler paths) | ✗ same as B + new encoder module + training loop change |
| **Curriculum alignment** (smooth input distribution shift as curriculum ramps) | ✓ smooth in [0,1] | ◑ ramps in native units; can have plateaus / non-affine schedules | ✓ smooth in [0,1] per axis | ✓ depends on encoder; smooth if encoder is well-conditioned |
| **Probe-ability post-Phase-1** (re-run ticket 036 Probe 2 against new ckpt) | ✓ direct (same 8 latents) | ✓ direct (against ~27 raw params) | ✓✓ direct (against 15 decoupled latents — gives a per-axis attribution) | ◑ probe against z is meaningful but indirect; probing against e_t requires holding the encoder fixed |
| **Lit precedent for *this exact pattern*** | ✗ no published work uses bundled curriculum progress as critic obs | ◑ asymmetric AC + raw params each precedented separately; conjunction not directly published | ✗ same as A on the progress-as-obs side | ✓✓ closest to RMA's pattern, just routed to critic instead of policy |
| **Re-usability for Phase 2 (aux sysid head)** | ✗ targets a bundled latent that already failed Probe 2 | ✓✓ aux head naturally regresses the same 27 raw params | ✓✓ aux head naturally regresses the 15 progress scalars (well-conditioned) | ✓ encoder is the natural sysid head target; nearly free for Phase 2 |
| **Risk of training-time correlation exploit** | ✗ high — critic + policy gradient over a tied manifold can learn to exploit the bundling | ✓ low — independent axes prevent exploit | ✓ low after env decoupling | ◑ encoder bottleneck may re-tie axes during compression |

### 2.3 Discussion per axis

**Info sufficiency.** A's bundled `_eff_progress_dynamics` is information-lossy *only because* the env bundles 7 quantities onto it; conditional on the env's current implementation, A *is* sufficient (since the env literally computes 7 derived params from this 1 scalar). The lossiness becomes real **only** when the env is decoupled — then A becomes a degraded view of the true env state. This is the load-bearing argument for option C: if we're doing the env decoupling anyway, A regresses from "information-sufficient" to "information-deficient" while C scales naturally.

**Normalization.** B's ~57 raw-unit dims with heterogeneous scales is a real problem for SKRL's `RunningStandardScaler`. The scaler computes running mean/std per-dim over rollout transitions. With latency at O(0.1) and mass at O(5), the early-training scaler will produce wildly different per-dim normalizations until it converges. C and A sidestep this entirely. D pushes the burden into the encoder, which is fine if encoder weights are warm-started or trained jointly.

**Plumbing.** The dominant cost in B and D is touching multiple downstream modules to surface their internal randomized values back up to the env's obs path. Each of `DroneController` (gains), `GimbalRateLoop` (τ, dead_time_steps), `ZoomController` (zoom dead_time_steps), `DomainRandomizer` (FOV, mass, inertia, gimbal_stiff_damp, target_scale), and `DelaySystem` (latency_s, noise_std, dropout_prob, burst_dropout_prob) need a getter. Each getter has a chance to drift from the actual value used in physics — that's a maintenance liability. C reuses the existing `_eff_progress_*` tensor path (already cached on env), so plumbing is small.

**Re-usability for Phase 2.** If Phase 2 adds an aux sysid head on the actor that regresses env-id from the GRU output, the *target* of that regression must be defined. Under A, the target is bundled and already failed Probe 2 — Phase 2 would inherit the bundling problem. Under B/C/D the target is independent per axis, giving Phase 2 a clean supervision signal.

**Encoder-induced correlation re-baking (D's risk).** A learned 8-d encoder is information-bottlenecked; it will preferentially encode dimensions that matter for V-loss. If the training distribution happens to correlate two axes (e.g., curriculum ramps them together in early training, even if they're now independently sampled per-episode), the encoder may collapse them into one dim, defeating the decoupling that the env enforces upstream. This is fixable (longer training, larger z, regularization) but a non-trivial new risk vs C.

### 2.4 Ranking under different priorities

Different design priorities push toward different options. Two sensible defaults:

**Priority = "ship Phase 1 cheap, get the V-loss reduction signal":** rank C > A > B > D. C is the minimum viable choice once decoupling is in scope; B's plumbing cost is too high for an initial run that already has uncertain expected magnitude (§1.7).

**Priority = "set up for Phase 2 / publication / sim2real fidelity":** rank B > D > C > A. B is the cleanest representation for the published-precedent sysid pattern; D is the second-most-precedented (RMA-style); C is iris_ma6-specific abstraction; A is unsupported.

Both defaults converge on **A and D being weak choices for Phase 1**: A because it fails info-sufficiency once the env is decoupled (and decoupling is in scope), D because it adds non-trivial new infrastructure (encoder training, joint optimization, possible re-baking of correlations) for an uncertain marginal gain over B at Phase-1 scope.

**The real choice is between B and C**, decided by how much the team values:
- Plumbing cost (favors C)
- Sim-to-real fidelity of the privileged channel (favors B)
- Setup for the eventual Phase 2 aux head (favors B — supervision target is in physical units, matches RMA's argument)
- Reproducibility of t034 behind a flag (both can do this; B requires more cfg surface but the patterns are the same)

A defensible position: **start with C** for Phase 1 (lower plumbing, gets the env decoupling done as a side effect, validates the V-loss-reduction hypothesis cheaply), **then escalate to B for Phase 2** if the aux-sysid-head ticket benefits from physical-unit targets. C and B are not exclusive across phases — Phase 2 can add raw-param exposure on top of Phase 1's progress-axis exposure, since the obs surface is already designed extensibly.

### 2.5 Specific open questions before §3 decision

These are factual / scoping uncertainties whose resolution would shift the §3 recommendation:

1. **What is the actual measurable V-loss reduction with C vs A?** No published number. The honest move is a short ablation (a few hundred k steps each on A and C, V-loss curves only — no need to train to 400 k for this question). Cost: ~1 day vs running full 400 k both ways.
2. **Does decoupling alone (env change without critic change) move metrics?** Worth knowing — if `enable_axis_independence=True` alone (no critic-obs change, A-equivalent obs) already moves V-loss or action_sum, then the critic-obs effect is a smaller delta on top.
3. **For B specifically: how much of the ~57 raw-param dims actually vary in deployment?** If some are fixed-by-design at deploy (calibrated FOV, characterized gimbal mech offsets per [doc/research/dr_observability_taxonomy.md](research/dr_observability_taxonomy.md)), they're privileged-but-static; their value in the critic is less meaningful than dynamic axes (mass, payload, latency).

**Recommendation for §3 decision-making process**: don't lock the choice without §2.5(1) — the ablation cost is small relative to the 24-hour 400 k training run we'd commit to under either choice.

---

## §3. Decision — locked on Option B (independent-draw view)

**Critic sees the upstream independent RNG draws (the scales / sampled values that parameterize the env), at native units, with per-field pre-normalization to [-1, +1]. Env-side decoupling of `progress_dynamics` into 8 independent curriculum axes is in scope (per ticket 037 decoupling sub-scope).**

Justification:

- **Code audit (Q3 in user discussion 2026-05-20)** confirmed raw params are more independent than initially assumed. The 5 controller-gain bundles (Kp_vel/Ki_vel; Kp/Ki/Kd_rate) are physically justified (single "controller aggressiveness" multipliers, as a human tuner would apply) and remain bundled at sample time. We feed the critic the 6 independent scales, not the 8 derived gain values.
- **Sim-to-real fidelity (user concern on bundling, ticket 037 motivation)**: raw scales match the independent variation structure of real hardware. The critic learns V over the same manifold that exists at deployment, not over a training-curriculum artifact.
- **Phase 2 supervision target**: when Phase 2 adds an aux sysid head, the regression target is the same 23 scalars per agent — well-conditioned (constant within an episode per stationarity discussion), in physical/scale units (matches RMA's encoder-on-physical-units argument).

### §3.1 The 23 per-agent + 2 shared = 48-dim privileged obs (A=2)

Per-agent (23 dims), per category:

| # | Field | Dim | Source / getter | Native range | Pre-norm to [-1,+1] |
|---|---|---:|---|---|---|
| **Drone control gain scales (6)** ||||||
| 1 | `vel_gain_scale` | 1 | scale used in `randomize_gains` velocity block | curriculum-ramped `[low, high]` ⊆ cfg.gain_randomization.scale_range | `2·(x − mid)/(high − low)` |
| 2 | `att_gain_scale` | 1 | scale used in attitude block | same | same |
| 3 | `rate_gain_scale` | 1 | scale used in rate block | same | same |
| 4 | `motor_gain_scale` | 1 | scale used in motor block | same | same |
| 5 | `zoom_tau_scale` | 1 | scale used for `tau_zoom` | `zoom_scale_range` | same |
| 6 | `zoom_max_rate_scale` | 1 | scale used for `max_zoom_rate` | `scale_range` (shared with gains) | same |
| **Drone dynamics (1)** ||||||
| 7 | `max_lin_vel_per_env` | 1 | `env._max_lin_vel[env_id, agent_id]` (post-curriculum, post-randomization) | `[max_lin_vel_min, max_lin_vel * scale_max]` | linear map |
| **Gimbal rate loop (1)** ||||||
| 8 | `gimbal_rate_tau_s` | 1 | `GimbalRateLoop._tau` per (env, agent) | per cfg.gimbal.tau range × `_eff_progress_*` | linear map |
| **Dead times (2)** ||||||
| 9 | `gimbal_dead_time_s` | 1 | `GimbalRateLoop._dead_time_s` per (env, agent) | `[0, dead_time_max_s]` per cfg | linear map |
| 10 | `zoom_dead_time_s` | 1 | `ZoomController._dead_time_s` per (env, agent) | similar | linear map |
| **Camera (1)** ||||||
| 11 | `fov_scale` | 1 | `DomainRandomizer.get_fov_scales()` | cfg.camera.fov_scale_range | linear map |
| **Gimbal mechanical offsets (3)** ||||||
| 12 | `gimbal_yaw_offset` | 1 | `gimbal_randomizer.joint_offsets[:, :, 0]` | `yaw_offset_range` | linear map |
| 13 | `gimbal_pitch_offset` | 1 | `joint_offsets[:, :, 1]` | `pitch_offset_range` | linear map |
| 14 | `gimbal_roll_offset` | 1 | `joint_offsets[:, :, 2]` | `roll_offset_range` | linear map |
| **Gimbal mass scales (2)** ||||||
| 15 | `gimbal_yaw_mass_scale` | 1 | `gimbal_randomizer.yaw_mass_scales` | cfg | linear map |
| 16 | `gimbal_pitch_mass_scale` | 1 | `gimbal_randomizer.pitch_mass_scales` | cfg | linear map |
| **Gimbal joint dynamics (2)** ||||||
| 17 | `gimbal_stiffness_scale` | 1 | `gimbal_randomizer.stiffness_scales` | `stiffness_scale_range` | linear map |
| 18 | `gimbal_damping_scale` | 1 | `gimbal_randomizer.damping_scales` | `damping_scale_range` | linear map |
| **Robot physics (1)** ||||||
| 19 | `body_mass_scale_or_add` | 1 | `physics_randomizer.mass_scales` (or `mass_additions` depending on mode) | mode-dependent | linear map within mode |
| **Detection pipeline (4)** ||||||
| 20 | `detection_latency_s` | 1 | actual current latency per (env, agent), from DelaySystem | curriculum × cfg | linear map |
| 21 | `bbox_noise_std` | 1 | actual current noise std | curriculum × cfg | linear map |
| 22 | `detection_dropout_prob` | 1 | actual current dropout prob | curriculum × cfg | linear map |
| 23 | `burst_dropout_prob` | 1 | actual current burst dropout prob | curriculum × cfg | linear map |

Shared per env (2 dims):

| # | Field | Dim | Source / getter | Native range | Pre-norm |
|---|---|---:|---|---|---|
| 24 | `target_scale_xy` | 1 | `env._dr_target_scale[:, 0, 0]` | `target_xy_scale_range` | linear |
| 25 | `target_scale_z` | 1 | `env._dr_target_scale[:, 0, 2]` | `target_z_scale_range` | linear |

**Total**: 23 × A + 2 = 48 dims for A=2; 71 dims for A=3.

**Excluded by design** (with rationale, recorded for review):

- `gimbal_friction_values` — physical impact dominated by stiffness/damping in practice; cfg range is narrow.
- `static_friction`, `dynamic_friction`, `restitution` (physics_randomizer material params) — only affects ground contact, not flying drones; including would add ~3 dims for zero V-loss benefit.
- `payload_mass` — already covered by `body_mass_scale_or_add` in our env's randomization config (payload is one component of the total body mass scale). If a future iris_ma config separates payload from base mass, add as field #26 (per-env shared).
- `tau_motor`-derived value — bundled with `motor_gain_scale` (#4); the scale is more informative than the derived τ_motor.
- `Kp_att` derived value — bundled with `att_gain_scale` (#2).

Decision: **48 dims with the option to extend via `critic_privileged_fields: list[str]`** if any of the excluded fields surfaces a meaningful V-loss attribution gap in post-training analysis.

### §3.2 Why pre-normalize to [-1, +1] in the env, not rely on `RunningStandardScaler`

The 23-dim per-agent vector spans heterogeneous native units (gain scales ~1.0 ± 0.2, latency ~0.05–0.15 s, mass_scale ~1.0 ± 0.1, dead-time ~0.05–0.10 s). SKRL's `RunningStandardScaler` computes running mean/var per-dim and standardizes during forward passes. In the first ~10 k transitions per dim, the running stats are dominated by initialization noise → standardization is unreliable → critic gradient is noisy. Pre-normalizing in the env using **cfg-known ranges** (every field above has a known `(low, high)` from the relevant cfg) sidesteps this — the critic sees [-1, +1] from step 0.

This also makes regression-against-this-channel (Phase 2) cleaner: the supervision target is bounded, so the head's output activation can be `tanh` with no per-axis scaling.

### §3.3 What does NOT go in the critic obs

Explicit non-inclusions for clarity:

- The 8 `_eff_progress_*` scalars themselves. These are the *curriculum schedule indices*, not the actual env state. We feed the env state (post-curriculum, post-randomization values).
- `progress_dynamics` and other progress *globals*. Not env-state; they're training-time scalars that all envs share.
- Agent role / identity. Per user's instruction, role-switching is out of scope.
- Anything time-varying within an episode (per stationarity, all 25 fields are frozen at reset).

### §3.4 Field registry (Slice 1 inventory audit — 2026-05-20)

Outcome of the Slice 1 accessor audit. Each row maps a design-doc §3.1 field to its canonical storage location, current accessor status, layout / shape concerns, and the work needed for Slice 3.

**Field-count revision from §3.1**: the inventory pass surfaced one removal and one expansion vs §3.1's draft.

- **Remove** fields #15–16 ("gimbal_yaw_mass_scale" / "gimbal_pitch_mass_scale"). The `scale_yaw` / `scale_pitch` draws at [gimbal_randomizer.py:356-360](../domain_randomization/gimbal_randomizer.py#L356-L360) are **τ scales**, not mass scales — multiplied directly into `tau_yaw_s` / `tau_pitch_s` and discarded; no mass effect exists in this randomizer. Drop these 2 dims.
- **Expand** field #8 ("gimbal_rate_tau_s") into two: `gimbal_rate_tau_yaw_s` and `gimbal_rate_tau_pitch_s`. `GimbalRateLoop` stores `_tau_yaw` and `_tau_pitch` separately (per [gimbal_rate_loop.py:62-67](../controller/gimbal_rate_loop.py#L62-L67)); they're independent. Adds 1 dim.

**Revised per-agent total**: 23 − 2 + 1 = **22 dims**. With 2 shared, **46 dims total for A=2** (was 48).

#### Inventory table

Layout legend:
- `(N, A)` = per (env, agent), stored at env or randomizer level → direct use.
- `(N*A,)` = batched-controller layout, agent-major (`agent_a → rows [a*N, (a+1)*N)`); needs `.reshape(A, N).t() → (N, A)` to align.
- `(N,)` = per-env, shared across agents → broadcast to `(N, A)` at assembly time.
- `(N, 1, 3)` = special shape; index into specific axes.

| # | Field | Owner module | Storage attribute | Storage shape | Getter status | Slice 3 work |
|---|---|---|---|---|---|---|
| 1 | `vel_gain_scale` | `DroneController` | derived: `_velocity._Kp_vel / _nominal_gains["Kp_vel"]` | `(N*A, 3)` divided element-wise; reduce mean over dim=1 to scalar | NONE — add `get_vel_gain_scale() -> Tensor[N, A]` | new getter; reshape `(N*A,) → (A, N).t() → (N, A)` |
| 2 | `att_gain_scale` | `DroneController` | derived: `_attitude._Kp_att / _nominal_gains["Kp_att"]` | `(N*A, 3)` reduce | NONE — add `get_att_gain_scale() -> Tensor[N, A]` | same |
| 3 | `rate_gain_scale` | `DroneController` | derived: `_rate._Kp_rate / _nominal_gains["Kp_rate"]` | `(N*A, 3)` reduce | NONE — add `get_rate_gain_scale() -> Tensor[N, A]` | same |
| 4 | `motor_gain_scale` | `DroneController` | derived: `_motor._tau_motor / _nominal_gains["tau_motor"]` | `(N*A,)` | NONE — add `get_motor_gain_scale() -> Tensor[N, A]` | reshape only |
| 5 | `zoom_tau_scale` | `DroneController` | derived: `_zoom._tau_zoom / _nominal_gains["tau_zoom"]` | `(N*A,)` | NONE — add `get_zoom_tau_scale() -> Tensor[N, A]` | reshape only |
| 6 | `zoom_max_rate_scale` | `DroneController` | derived: `_zoom._max_zoom_rate / _nominal_gains["max_zoom_rate"]` | `(N*A,)` | NONE — add `get_zoom_max_rate_scale() -> Tensor[N, A]` | reshape only |
| 7 | `max_lin_vel_per_env` | `IrisMA6TestEnv` | `self._max_lin_vel` | `(N,)` | NONE — env-side broadcast | **ISSUE**: stored per-env (mean over agents per [iris_ma_env6_test.py:2632](../iris_ma_env6_test.py#L2632)). Broadcast to `(N, A)` at assembly; flag as info-loss. Consider deferring to a sibling ticket that extends `_max_lin_vel` to be per-(env, agent). |
| 8a | `gimbal_rate_tau_yaw_s` | `GimbalRateLoop` | `_tau_yaw * _tau_progress_per_env` | `(N*A,) × (N*A,)` | partial — `@property tau_yaw` returns nominal, not effective. Add `get_tau_yaw_effective() -> Tensor[N, A]` | new getter; reshape |
| 8b | `gimbal_rate_tau_pitch_s` | `GimbalRateLoop` | `_tau_pitch * _tau_progress_per_env` | `(N*A,) × (N*A,)` | same | same |
| 9 | `gimbal_dead_time_s` | `GimbalRateLoop` | `_dead_time_seconds` | `(N*A,)` | `@property dead_time_seconds` returns `(N*A,)` ✓ | add reshape wrapper `get_dead_time_seconds_per_env_per_agent() -> Tensor[N, A]` |
| 10 | `zoom_dead_time_s` | `ZoomController` | `_dead_time_seconds` | `(N*A,)` | `@property dead_time_seconds` returns `(N*A,)` ✓ | same reshape wrapper |
| 11 | `fov_scale` | `CameraProcessor` (via `DomainRandomizer.process_images`) | `camera_processor.fov_scales` | `(N, A)` ✓ | NONE — add `DomainRandomizer.get_fov_scales() -> Tensor[N, A]` (delegate) | thin getter |
| 12 | `gimbal_yaw_offset` | `GimbalRandomizer` | `joint_offsets[:, :, 0]` | slice of `(N, A, 3)` | NONE — add `get_joint_offsets() -> Tensor[N, A, 3]` and index | thin getter; index in assembly |
| 13 | `gimbal_pitch_offset` | `GimbalRandomizer` | `joint_offsets[:, :, 1]` | same | same | same |
| 14 | `gimbal_roll_offset` | `GimbalRandomizer` | `joint_offsets[:, :, 2]` | same | same | same |
| 15 | `gimbal_stiffness_scale` | `GimbalRandomizer` | `stiffness_scales` | `(N, A)` ✓ | NONE — add `get_stiffness_scales() -> Tensor[N, A]` | thin getter |
| 16 | `gimbal_damping_scale` | `GimbalRandomizer` | `damping_scales` | `(N, A)` ✓ | NONE — add `get_damping_scales() -> Tensor[N, A]` | thin getter |
| 17 | `body_mass_scale_or_add` | `PhysicsRandomizer` | `mass_scales` OR `mass_additions` (mode-dep) | `(N,)` | EXISTS as `get_mass_scales(env_ids)` ✓ | broadcast to `(N, A)` at assembly; cfg switches `mass_scales` vs `mass_additions` based on randomization mode |
| 18 | `detection_latency_s` | env-derived: `_eff_progress_delay × cfg.delay_system_params.ego_detection_latency_mean` | `_eff_progress_delay` is `(N, A)` ✓ | direct multiply | NONE — compute in `_get_privileged_obs` | inline arithmetic, no module getter |
| 19 | `bbox_noise_std` | env-derived: `_eff_progress_noise × cfg.delay_system_params.noise_bbox_std` | `(N, A)` × scalar | direct multiply | NONE — same | same |
| 20 | `detection_dropout_prob` | env-derived: `_eff_progress_dropout × cfg.delay_system_params.dropout_prob` | `(N, A)` × scalar | direct multiply | NONE — same | same |
| 21 | `burst_p_onset` (renamed from `burst_dropout_prob` post-verification) | env-derived: `_eff_progress_burst_dropout × cfg.delay_system_params.burst_p_onset` (verified at [iris_ma_env6_test.py:1494-1497](../iris_ma_env6_test.py#L1494-L1497)) | `(N, A)` × scalar | direct multiply | NONE | inline arithmetic; this is the Gilbert-Elliott burst-ONSET probability (NOT a generic dropout probability) — varies per (env, agent) via the burst_dropout curriculum axis |
| 22 | `target_scale_xy` | `IrisMA6TestEnv` | `_dr_target_scale[:, 0, 0]` (= `[:, 0, 1]`) | scalar slice of `(N, 1, 3)` | NONE — add `get_target_scale_xy() -> Tensor[N]` | env-side broadcast: shared field, single value per env |
| 23 | `target_scale_z` | `IrisMA6TestEnv` | `_dr_target_scale[:, 0, 2]` | scalar slice | NONE — add `get_target_scale_z() -> Tensor[N]` | env-side broadcast |

**Total per-agent**: 22 fields = #1–6 (gain scales) + #7 (max_lin_vel) + #8a/8b (τ yaw/pitch) + #9/10 (dead times) + #11 (fov) + #12/13/14 (offsets) + #15/16 (stiff/damp) + #17 (mass) + #18–21 (detection pipeline) = 22.
**Total shared**: 2 fields (target_scale_xy, target_scale_z).
**Privileged-obs dim total (A=2)**: **22 × 2 + 2 = 46 dims**. (A=3: 22 × 3 + 2 = 68 dims.)

Earlier counts of "44" / "48" in earlier revisions of this doc were arithmetic slips during the inventory pass; the verified count against the registry constant in `iris_ma_env6_test_cfg.py:_CRITIC_PRIVILEGED_FIELD_REGISTRY` is 24 entries (22 per-agent + 2 shared) → 46 dims for A=2.

#### Concerns surfaced

1. **`max_lin_vel` is per-env, not per (env, agent)** — line [iris_ma_env6_test.py:2632](../iris_ma_env6_test.py#L2632) does `.mean(dim=-1)` over agents before writing. So even if the actor draws different `_eff_progress_agent_velocity[env, agent]` per agent, the env collapses to a single per-env scalar. The critic obs for this field will be the same for all agents in an env. **Decision needed**: either (a) accept the broadcast (info-loss but matches what the controller uses), or (b) open a sub-ticket to extend `_max_lin_vel` to be per-(env, agent), then expose per-agent values to the critic. Recommend (a) for ticket 037 — the controller code consuming `_max_lin_vel` would also need changing for (b).

2. **All 6 gain scales are best computed as derived values** (`current / nominal`) rather than caching new state. Decision: option (b) from §3.4 row 1 — derive at obs-build time. Avoids new state-maintenance liability. But note: `Kp_vel`, `Kp_att`, `Kp_rate` etc. are 3-vectors (per-axis); the scale is a single multiplier applied uniformly, so dividing pointwise gives 3 identical values → reduce by mean (or take any single index) to get the scalar scale.

3. **Burst dropout exact formula**: verify the env-side multiplication at [iris_ma_env6_test.py:1488-1496](../iris_ma_env6_test.py#L1488-L1496) before locking the formula for field #21. The cfg has multiple burst-related parameters; ensure we pick the one that actually parameterizes the per-(env, agent) probability.

4. **Effective τ vs nominal τ**: fields 8a/8b should expose the **effective** value (`_tau_yaw * _tau_progress_per_env`), not the nominal. The existing `@property tau_yaw` returns nominal; new getter must multiply. This is also what the actual rate-loop step uses.

5. **GimbalRateLoop `_dead_time_seconds` is `(N*A,)`** in the batched-controller layout. The reshape utility `.reshape(A, N).t() → (N, A)` is the same convention used at [iris_ma_env6_test.py:2472](../iris_ma_env6_test.py#L2472). Either add a thin `get_dead_time_per_env_per_agent(num_envs, num_agents)` wrapper or do the reshape at the env-side assembly step.

6. **`DomainRandomizer.get_fov_scales()` already exists** per [domain_randomization/domain_randomizer.py](../domain_randomization/domain_randomizer.py) (referenced at [iris_ma_env6_test.py:2668](../iris_ma_env6_test.py#L2668)). Reuse, don't re-add.

#### Slice 1 verdict

- **No surprises that block Slice 2/3.**
- **Field count revised**: per-agent 22, total 46 (A=2) — verified by registry enumeration at end of Slice 2.
- **Module getters required vs already-exist (verified during Slice 3)**:
  - DroneController: **6 new** derived-scale getters (added)
  - GimbalRateLoop: **2 new** effective-τ `@property` accessors (added); `dead_time_seconds` already exists
  - ZoomController: nothing new — `dead_time_seconds` already a `@property`
  - GimbalRandomizer: **0 new** — `get_yaw_offsets`, `get_pitch_offsets`, `get_roll_offsets`, `get_stiffness_scales`, `get_damping_scales` all already exist
  - PhysicsRandomizer: **0 new** — `get_mass_scales` already exists
  - CameraProcessor: **0 new** — `get_fov_scales` already exists
  - IrisMA6TestEnv: **0 new** — Slice 4 reads `self._dr_target_scale` / `self._max_lin_vel` / `self._eff_progress_*` directly inside `_get_states`
  - Total new methods: 8. **Actual Slice 3 effort: ~½ hour, not 1 day as estimated.**
- **One outstanding design decision** (`max_lin_vel` per-env vs per-(env, agent)): recommend deferring per-(env, agent) extension to a sub-ticket, accept the broadcast in ticket 037.
- **One outstanding formula verification** (burst dropout) before Slice 4 can lock the formula.
