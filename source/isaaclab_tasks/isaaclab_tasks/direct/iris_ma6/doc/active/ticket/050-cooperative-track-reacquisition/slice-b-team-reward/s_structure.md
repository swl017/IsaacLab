# Stage S — Structure Outline (Slice B)

## New files

### information_reward/ (standalone module; torch + configclass only)
- `__init__.py` — exports InformationRewardCfg, InformationReward.
- `CONTEXT.md` — ICM L2 contract (+ §4.4: compute() is READ-only/pure, call once per step in _get_rewards).
- `information_reward_cfg.py`
  ```
  @configclass InformationRewardCfg:
      enabled: bool = False
      sigma_theta: float = 0.005      # bearing angular noise [rad]; derived in S from intrinsics
      sigma_prior: float = 40.0       # prior position std [m] ~ scenario scale
      quality_const_c: float = 10.0   # matches current sqrt(10/trace) readout
      info_scale_max: float = 8.0     # difference-reward scale at full ramp (≈ triangulation_reward_scale)
      in_plane_only: bool = False     # readout 2x2 (xy) vs full 3x3 (risk #3)
      z_prior_scale: float = 4.0      # extra z-prior strength when 3x3 (weak vertical observability)
  ```
- `information_reward.py`
  ```
  class InformationReward:
      __init__(self, cfg, num_envs, num_agents, device)
      # READ-only pure function of current geometry; once/step
      compute(self, bearings_w[N,A,3], cam_pos_w[N,A,3], target_pos_w[N,3], valid[N,A] bool)
          -> dict{ "team_quality": [N], "r_diff": [N,A] }
      # internals (signatures only)
      _build_fim(self, bearings_w, cam_pos_w, target_pos_w, valid) -> (FIM[N,3,3], info_terms[N,A,3,3])
      _quality(self, Sigma[N,3,3]) -> [N]          # sqrt(c/trace), in-plane-aware
      _prior(self) -> [3,3]                         # Lambda_prior = diag(1/sigma_prior^2, .., z-scaled)
  ```
- `tests/` — __init__.py, run_tests.py (AppLauncher), README.md, test_result.txt, error_log.txt.

## Modified files

### iris_ma_env6_test_cfg.py
- `[add import]` InformationRewardCfg.
- `[add]` `information_reward: InformationRewardCfg = InformationRewardCfg()`.
- `[add]` rebalance ramp targets: `bbox_center_reward_scale_team: float = 30.0`,
  `bbox_size_reward_scale_team: float = 20.0`.
- `[existing]` `enable_critic_gt_target` — set True for Slice-B runs via the experiment entry
  (pre-__post_init__), NOT a global default flip.

### curriculum/curriculum_cfg.py
- `[add]` `reward_rebalance_start_step: int = 80000`, `reward_rebalance_end_step: int = 200000`.
- `[add]` `get_reward_rebalance_progress(self, current_step) -> float`.

### iris_ma_env6_test.py
- `[modify] __init__` — instantiate `self._info_reward = InformationReward(...)` when
  `cfg.information_reward.enabled` else None; init `self.progress_rebalance = 0.0`.
- `[modify]` progress-update block (where progress_coord is set) — set
  `self.progress_rebalance = curriculum.get_reward_rebalance_progress(current_step)`.
- `[add]` `_gather_info_signals(self, reward_states) -> (bearings_w[N,A,3], cam_pos_w[N,A,3], valid[N,A])`
  (factored from the bearing/pos gather; reused by _gather_reacq_signals).
- `[modify] _get_rewards` — when `_info_reward` is not None: call `compute()` ONCE before the agent
  loop → `team_quality[N]`, `r_diff[N,A]`. In the loop:
  - `info_term_i = (1-p_rebal)*triangulation_quality + p_rebal*(r_diff[:,i]*info_scale_max)`
    into the existing `"triangulation"` slot (×triangulation_reward_scale×step_dt×progress_coord).
  - `bbox_center` scale → `bbox_center_reward_scale + p_rebal*(bbox_center_reward_scale_team - bbox_center_reward_scale)`;
    same pattern for `bbox_size`.
  When `_info_reward` is None → unchanged (bit-exact baseline).

### experiments/experiment_registry.py
- `[add]` Slice-B experiment(s): `b050_team_reward` (information_reward.enabled=True,
  enable_track_loss_scenario=True, cooperation_metrics.enable=True, enable_critic_gt_target=True,
  curriculum.reward_rebalance_* window, warm-start checkpoint) + `b050_baseline_no_info` (info off)
  for the A/B vs the C2 baseline.

### ARCHITECTURE.md
- `[add]` information_reward node (standalone; consumed by env _get_rewards).

## Notes
- No obs change. GT target/range used reward-side only (consistent with use_gt_target=True).
- compute() is pure/stateless — unit-testable in isolation (the QRISPY-correct first slice).
- σ_theta default computed in implementation from live intrinsics (img_w / 2tan(hfov/2), σ_pix≈7).
