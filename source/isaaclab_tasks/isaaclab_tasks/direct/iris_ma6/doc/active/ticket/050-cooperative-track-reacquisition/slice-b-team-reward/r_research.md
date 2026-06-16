# Stage R — Codebase Research Report (Slice B)

Unprefixed file refs are `iris_ma_env6_test.py`.

## Reward stack (_get_rewards, :1685–2115)
- Per-agent loop (:1952) → `rewards` dict summed (minus diagnostics) into `rewards_dict[agent]` (:2103).
- Terms ×step_dt with cfg scales: bbox_center=90, bbox_size=30, triangulation=8 (×progress_coord),
  action_sum/action_delta(Δ=−24), cbf_penalty/collision/altitude/target_proximity (×progress_safety),
  est_error_gt/e2e (diagnostic, excluded from total).
- **triangulation term is SHARED across agents**: `_compute_fim_reward()` (:2135) computed ONCE, added
  identically to every agent. No per-agent credit decomposition exists today.
- `_compute_fim_reward` = clamp(sqrt(10/trace(Σ)), max=100), 0 where invalid, NaN→1e6 (:2146–2158).
  "FIM" is a misnomer = trace of LS covariance.
- All terms NaN/Inf-sanitized to 0 (:2087). _step_rewards/_episode_sums per term.
- Defaults: task_reward_level=1 (trace), curriculum_task_levels=False (cfg:938/961).
- progress_coord (:626) = step-at gate, held constant after coordination_start_step; shape [N].

## Triangulation / covariance
- Linear LS midpoint; needs ≥2 detections (min_cameras_required=2); pos/cov/quality NaN below 2
  (triangulation.py:374,414,456,917). Covariance [N,T,3,3] = weighted-LS propagation, NOT info matrix.
  quality default = trace. NO FIM/information/prior/HᵀR⁻¹H machinery (grep-confirmed).
- Per-agent bearings available at reward time: reward_states[a].data.camera_ray_directions_w[:,0,:] [N,3]
  world unit ray. Camera pos = _root_pos_w[a] [N,3]. GT target = _target_pos_w [N,3]. Per-agent
  validity = _per_agent_bbox_nonempty [N,A].

## CTDE critic (_get_states, :2487)
- Asymmetric critic activates if enable_critic_continuous_zoom OR enable_critic_gt_target (:263).
  state_space auto-sized in __post_init__; state_spaces mirrored per agent.
- State = concat(actor obs) ++ [zoom_internal,zoom_target] ++ [GT target pos(3) if enable_critic_gt_target]
  ++ [GT target vel(3) if velocity] (:2498–2505).
- enable_critic_gt_target DEFAULT FALSE (cfg:262); flipping True = privileged-critic stabilizer (Q#7),
  actor obs unchanged. enable_full_critic_priv_obs=True default (cfg:315).

## Gaps / inconsistencies (factual)
1. Cooperative reward is shared/identical across agents — no per-agent marginal term.
2. No information-matrix machinery; team quality is LS-cov trace, NaN→0 reward below 2 detections —
   cooperative signal is exactly ZERO during a single-agent deficit (the recovery moment).
3. Per-agent bearings + camera pos + GT target all available at reward time → can build per-bearing
   information sum with a prior.
4. enable_critic_gt_target exists, wired, defaults off.
5. enable_*_critic_* flags auto-size state_space in __post_init__; Hydra from_dict (Slice-A caveat)
   would mis-size the critic if set late — verify launch path sets it pre-construction.

## Design-shaping conclusions (→ Stage I)
- Build FIM-with-prior from per-agent bearings (absent) → defined at 0/1/N → read trace(Σ=FIM⁻¹) to
  match current form; per-agent difference = drop each bearing's rank term.
- shared→difference change localized to the triangulation term; rebalance rides existing
  progress_*/task_level machinery.
- enable_critic_gt_target=True = one-flag privileged-critic stabilizer (watch gap #5).
