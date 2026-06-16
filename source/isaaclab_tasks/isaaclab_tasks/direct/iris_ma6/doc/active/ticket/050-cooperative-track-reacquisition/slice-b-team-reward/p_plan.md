# Stage P — Implementation Plan (Slice B)

4 vertical slices, each end-to-end testable; default-off until the A/B confirms. Builds on the
merged Slice A (b4edcfa5af).

## Slice 1: information_reward/ standalone module (pure, unit-testable)
- 1.1 information_reward_cfg.py → InformationRewardCfg (sigma_theta, sigma_prior, quality_const_c,
  info_scale_max, in_plane_only, z_prior_scale, enabled).
- 1.2 information_reward.py → __init__ + _prior() (Lambda_prior = diag(1/sigma_prior^2, .., z-scaled)).
- 1.3 _build_fim(bearings_w, cam_pos_w, target_pos_w, valid) → FIM[N,3,3] = prior + Σ_valid w_i (I-d d^T),
  w_i = 1/(r_i^2 sigma_theta^2); valid-masked (invalid agents contribute nothing).
- 1.4 _quality(Sigma) → sqrt(c/trace), in-plane-aware (2x2 xy if in_plane_only).
- 1.5 compute(...) → {team_quality[N], r_diff[N,A]}; r_diff_i = J(Sigma) - J(Sigma_{-i}) via
  drop-one-rank (FIM - w_i(I-d d^T)); invalid agents r_diff=0.
- 1.6 __init__.py + CONTEXT.md (§4.4: compute READ-only, once/step).
- 1.7 tests/run_tests.py: 0/1/2 bearings all defined (no NaN); 2 well-separated bearings collapse
  Sigma (high quality); 2 near-collinear bearings stay poor (GDOP); r_diff largest for the bearing
  whose removal most raises trace; redundant bearing small; 2-step (unit vectors then realistic).
- 1.8 iterate to green (test_result.txt / error_log.txt).
- Test checkpoint: run_tests.py all pass.

## Slice 2: Env integration at full strength (default-off) + privileged critic
- 2.1 iris_ma_env6_test_cfg.py: import + `information_reward: InformationRewardCfg`;
  `bbox_center_reward_scale_team=30`, `bbox_size_reward_scale_team=20`.
- 2.2 iris_ma_env6_test.py __init__: instantiate `self._info_reward` (guard enabled); init
  `self.progress_rebalance=1.0` (full strength until Slice 3 adds the ramp).
- 2.3 `_gather_info_signals(reward_states)` → (bearings_w[N,A,3], cam_pos_w[N,A,3], valid[N,A]);
  refactor _gather_reacq_signals to reuse it.
- 2.4 _get_rewards: when _info_reward not None, compute() once; set triangulation slot to
  r_diff[:,i]*info_scale_max (×triangulation_reward_scale×step_dt×progress_coord). Compute
  sigma_theta default from live intrinsics; log once.
- 2.5 Wire enable_critic_gt_target for the Slice-B path (set pre-__post_init__; verify state_space sizes).
- Test checkpoint: short run information_reward.enabled=True → triangulation reward reflects per-agent
  r_diff, no crash, critic state_space sized; enabled=False bit-exact.

## Slice 3: Rebalance curriculum (bbox -> team ramp)
- 3.1 curriculum/curriculum_cfg.py: `reward_rebalance_start_step=80000`, `_end_step=200000`,
  `get_reward_rebalance_progress(step)`.
- 3.2 env progress block (by progress_coord): set `self.progress_rebalance = ...get_reward_rebalance_progress`.
- 3.3 _get_rewards: bbox_center scale = 90 + p_rebal*(30-90); bbox_size = 30 + p_rebal*(20-30);
  triangulation = (1-p_rebal)*trace_quality + p_rebal*r_diff[:,i]*info_scale_max.
- 3.4 config/curriculum smoke: p_rebal at steps 0 / 140k / 300k → expected blended scales.
- Test checkpoint: smoke shows bbox_center 90→30 and info weight 0→full across 80k-200k; flag-off unchanged.

## Slice 4: Experiment entries + A/B readiness
- 4.1 experiments/experiment_registry.py: `b050_team_reward` (information_reward.enabled,
  enable_track_loss_scenario, cooperation_metrics.enable, enable_critic_gt_target, rebalance window,
  warm-start ckpt) + `b050_baseline_no_info` (info off).
- 4.2 Verify registry loads + cfg overrides resolve (enable_critic_gt_target pre-construction).
- 4.3 Short warm-start training smoke (few-k steps) from t048 wide: info reward + Coop/* log, no collapse.
- Test checkpoint: both experiments list; short run launches and logs; full A/B + multi-seed = engineer's run.

## Gate (acceptance, engineer's run)
reacq_success_rate & time-to-reacquire improve vs C2 baseline (0.385 / 0.46s) under the same scenario,
without collapsing base tracking — measured via evaluate.py --track_loss_scenario --coop_metrics.
