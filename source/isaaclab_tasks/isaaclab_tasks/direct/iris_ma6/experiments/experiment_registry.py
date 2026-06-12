# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central registry of all iris_ma6 experiments.

Experiments are organized by ablation study:
    A1 - Delay modeling (with/without delay, AoI)
    A3 - Reward path (delay x noise: 4 combinations)
    A4 - Covariance reward mode
    A5 - CBF safety (penalty lambda, safety radius)
    A6 - Controller robustness (gain randomization, velocity limits)
    A7 - Task reward levels (FIM, GT-anchored, E2E composite)
    Baselines - Greedy orbit, no-triangulation
    Sweeps - Agent count, delay, noise, target speed
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    from .experiment_cfg import ExperimentCfg, ExperimentSuiteCfg
except ImportError:
    from experiment_cfg import ExperimentCfg, ExperimentSuiteCfg

# ===========================================================================
# Global registries
# ===========================================================================

_EXPERIMENTS: Dict[str, ExperimentCfg] = {}
_SUITES: Dict[str, ExperimentSuiteCfg] = {}


def register_experiment(cfg: ExperimentCfg) -> None:
    """Register a named experiment."""
    if cfg.name in _EXPERIMENTS:
        raise ValueError(f"Experiment '{cfg.name}' already registered")
    _EXPERIMENTS[cfg.name] = cfg


def get_experiment(name: str) -> ExperimentCfg:
    """Get a registered experiment by name."""
    if name not in _EXPERIMENTS:
        available = sorted(_EXPERIMENTS.keys())
        raise KeyError(f"Experiment '{name}' not found. Available: {available}")
    return _EXPERIMENTS[name]


def list_experiments(group: Optional[str] = None) -> list[str]:
    """List registered experiment names, optionally filtered by group."""
    if group:
        return sorted(k for k, v in _EXPERIMENTS.items() if v.group == group)
    return sorted(_EXPERIMENTS.keys())


def register_suite(cfg: ExperimentSuiteCfg) -> None:
    """Register a named experiment suite."""
    if cfg.name in _SUITES:
        raise ValueError(f"Suite '{cfg.name}' already registered")
    _SUITES[cfg.name] = cfg


def get_suite(name: str) -> ExperimentSuiteCfg:
    """Get a registered suite by name."""
    if name not in _SUITES:
        raise KeyError(f"Suite '{name}' not found. Available: {sorted(_SUITES.keys())}")
    return _SUITES[name]


def list_suites() -> list[str]:
    """List all registered suite names."""
    return sorted(_SUITES.keys())


# ===========================================================================
# Helper
# ===========================================================================

def _compute_obs_dim(enable_triangulation: bool = True) -> int:
    """Compute observation dimension for iris_ma6.

    Base: 24D (pos, vel, quat, ang_vel_b, lin_acc_b, gimbal_yaw, gimbal_pitch, zoom, bbox, bbox_empty)
    With triangulation: +6D (triangulated position + std_dev)
    """
    return 24 + (6 if enable_triangulation else 0)


# ===========================================================================
# A1: Delay Modeling Ablation
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a1_no_delay",
    description="No delay, no noise, no dropout (clean baseline)",
    group="A1",
    env_overrides={
        "enable_delay_system": False,
        # Push all delay/noise curriculum steps beyond training length
        "curriculum.noise_start_step": 999999,
        "curriculum.noise_end_step": 999999,
        "curriculum.fixed_delay_start_step": 999999,
        "curriculum.fixed_delay_end_step": 999999,
        "curriculum.random_delay_start_step": 999999,
        "curriculum.random_delay_end_step": 999999,
        "curriculum.dropout_start_step": 999999,
        "curriculum.dropout_end_step": 999999,
    },
))

register_experiment(ExperimentCfg(
    name="a1_stochastic_delay",
    description="Full stochastic delay pipeline (default params with curriculum)",
    group="A1",
    env_overrides={},  # Default config has full delay pipeline
))

register_experiment(ExperimentCfg(
    name="a1_with_aoi",
    description="Full delay system with AoI in observations (default)",
    group="A1",
    env_overrides={},
))

register_experiment(ExperimentCfg(
    name="a1_without_aoi",
    description="Full delay system with AoI fields zeroed out in observations",
    group="A1",
    env_overrides={},
    # NOTE: AoI masking in obs needs implementation in env (future work)
))


# ===========================================================================
# A3: Reward Path (Delay x Noise — 4 combinations)
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a3_clean",
    description="Reward path: no delay, no noise (clean GT rewards)",
    group="A3",
    env_overrides={
        "delay_system_params.reward_use_delay": False,
        "delay_system_params.reward_use_noise": False,
    },
))

register_experiment(ExperimentCfg(
    name="a3_delay_only",
    description="Reward path: delay but no noise (default paper system)",
    group="A3",
    env_overrides={
        "delay_system_params.reward_use_delay": True,
        "delay_system_params.reward_use_noise": False,
    },
))

register_experiment(ExperimentCfg(
    name="a3_noise_only",
    description="Reward path: noise but no delay",
    group="A3",
    env_overrides={
        "delay_system_params.reward_use_delay": False,
        "delay_system_params.reward_use_noise": True,
    },
))

register_experiment(ExperimentCfg(
    name="a3_noisy_delay",
    description="Reward path: delay + noise (full observation pipeline for rewards)",
    group="A3",
    env_overrides={
        "use_noisy_rewards": True,
    },
))


# ===========================================================================
# A4: Covariance Reward Mode
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a4_analytical",
    description="Multi-source FIM covariance reward (paper default, Level 1)",
    group="A4",
    env_overrides={
        "task_reward_level": 1,
    },
))

register_experiment(ExperimentCfg(
    name="a4_analytical_delay",
    description="TODO: Improved covariance model accounting for delay-induced uncertainty",
    group="A4",
    env_overrides={
        "task_reward_level": 1,
        # TBD: covariance model overrides for delay-aware FIM
    },
))


# ===========================================================================
# A5: CBF Safety (NEW for iris_ma6)
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a5_no_cbf",
    description="No safety penalty, no collision termination",
    group="A5",
    env_overrides={
        "cbf_safety.enable_training_penalty": False,
        "cbf_safety.enable_collision_termination": False,
    },
))

register_experiment(ExperimentCfg(
    name="a5_cbf_default",
    description="Default CBF (lambda=1.0, D_s=2.0m)",
    group="A5",
    env_overrides={},
))

register_experiment(ExperimentCfg(
    name="a5_cbf_lambda_0.5",
    description="Reduced CBF penalty weight (lambda=0.5)",
    group="A5",
    env_overrides={
        "cbf_safety.cpa_cfg.lambda_cbf": 0.5,
    },
))

register_experiment(ExperimentCfg(
    name="a5_cbf_lambda_2.0",
    description="Increased CBF penalty weight (lambda=2.0)",
    group="A5",
    env_overrides={
        "cbf_safety.cpa_cfg.lambda_cbf": 2.0,
    },
))

register_experiment(ExperimentCfg(
    name="a5_cbf_lambda_5.0",
    description="Strong CBF penalty weight (lambda=5.0)",
    group="A5",
    env_overrides={
        "cbf_safety.cpa_cfg.lambda_cbf": 5.0,
    },
))

register_experiment(ExperimentCfg(
    name="a5_cbf_radius_1.5m",
    description="Tighter safety margin (D_s=1.5m)",
    group="A5",
    env_overrides={
        "cbf_safety.cpa_cfg.D_s": 1.5,
        "cbf_safety.collision_distance": 1.5,
    },
))

register_experiment(ExperimentCfg(
    name="a5_cbf_radius_3.0m",
    description="Wider safety margin (D_s=3.0m)",
    group="A5",
    env_overrides={
        "cbf_safety.cpa_cfg.D_s": 3.0,
        "cbf_safety.collision_distance": 3.0,
    },
))

register_experiment(ExperimentCfg(
    name="a5_termination_only",
    description="Collision terminates but no reward shaping",
    group="A5",
    env_overrides={
        "cbf_safety.enable_training_penalty": False,
        "cbf_safety.enable_collision_termination": True,
    },
))


# ===========================================================================
# A6: Controller Robustness (NEW for iris_ma6)
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a6_no_gain_rand",
    description="No controller gain randomization",
    group="A6",
    env_overrides={
        "gain_randomization.enabled": False,
    },
))

register_experiment(ExperimentCfg(
    name="a6_gain_rand_default",
    description="Default +-20% gain randomization",
    group="A6",
    env_overrides={},
))

register_experiment(ExperimentCfg(
    name="a6_gain_rand_40pct",
    description="+-40% gain randomization",
    group="A6",
    env_overrides={
        "gain_randomization.scale_range": (0.6, 1.4),
    },
))

register_experiment(ExperimentCfg(
    name="a6_max_vel_5",
    description="Conservative velocity limit (5 m/s)",
    group="A6",
    env_overrides={
        "max_lin_vel": 5.0,
        "max_lin_vel_min": 3.0,
    },
))

register_experiment(ExperimentCfg(
    name="a6_max_vel_15",
    description="Aggressive velocity limit (15 m/s)",
    group="A6",
    env_overrides={
        "max_lin_vel": 15.0,
        "max_lin_vel_min": 8.0,
    },
))


# ===========================================================================
# A7: Task Reward Levels (NEW for iris_ma6)
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a7_level1_fim",
    description="Level 1: FIM proxy only (sqrt(10/Tr(Sigma)))",
    group="A7",
    env_overrides={
        "task_reward_level": 1,
        "curriculum_task_levels": False,
    },
))

register_experiment(ExperimentCfg(
    name="a7_level2_gt_anchored",
    description="Level 2: GT-anchored estimation error",
    group="A7",
    env_overrides={
        "task_reward_level": 2,
        "curriculum_task_levels": False,
    },
))

register_experiment(ExperimentCfg(
    name="a7_level3_composite",
    description="Level 3: Composite (GT + E2E blend, e2e_weight=0.5)",
    group="A7",
    env_overrides={
        "task_reward_level": 3,
        "curriculum_task_levels": False,
        "e2e_weight": 0.5,
    },
))

register_experiment(ExperimentCfg(
    name="a7_curriculum_blend",
    description="Progressive task level transitions via curriculum",
    group="A7",
    env_overrides={
        "curriculum_task_levels": True,
    },
))

register_experiment(ExperimentCfg(
    name="a7_e2e_weight_0.0",
    description="Level 3 with pure GT-anchored (e2e_weight=0)",
    group="A7",
    env_overrides={
        "task_reward_level": 3,
        "curriculum_task_levels": False,
        "e2e_weight": 0.0,
    },
))

register_experiment(ExperimentCfg(
    name="a7_e2e_weight_0.3",
    description="Level 3 with mostly GT-anchored (e2e_weight=0.3)",
    group="A7",
    env_overrides={
        "task_reward_level": 3,
        "curriculum_task_levels": False,
        "e2e_weight": 0.3,
    },
))

register_experiment(ExperimentCfg(
    name="a7_e2e_weight_0.7",
    description="Level 3 with mostly E2E (e2e_weight=0.7)",
    group="A7",
    env_overrides={
        "task_reward_level": 3,
        "curriculum_task_levels": False,
        "e2e_weight": 0.7,
    },
))

register_experiment(ExperimentCfg(
    name="a7_e2e_weight_1.0",
    description="Level 3 with pure E2E (e2e_weight=1.0)",
    group="A7",
    env_overrides={
        "task_reward_level": 3,
        "curriculum_task_levels": False,
        "e2e_weight": 1.0,
    },
))

register_experiment(ExperimentCfg(
    name="a7_high_temp",
    description="Level 2 with sharper reward falloff (temp=3.0)",
    group="A7",
    env_overrides={
        "task_reward_level": 2,
        "curriculum_task_levels": False,
        "estimation_error_temp": 3.0,
    },
))

register_experiment(ExperimentCfg(
    name="a7_low_temp",
    description="Level 2 with gentler reward falloff (temp=0.3)",
    group="A7",
    env_overrides={
        "task_reward_level": 2,
        "curriculum_task_levels": False,
        "estimation_error_temp": 0.3,
    },
))


# ===========================================================================
# Baselines
# ===========================================================================

# baseline_greedy is eval-only — see baselines/greedy_policy.py
register_experiment(ExperimentCfg(
    name="baseline_greedy",
    description="Greedy equiangular orbit (eval-only, scripted policy)",
    group="baseline",
    env_overrides={},
))

register_experiment(ExperimentCfg(
    name="baseline_no_triangulation",
    description="BBox tracking only (no triangulation reward)",
    group="baseline",
    env_overrides={
        "enable_triangulation": False,
        "triangulation_reward_scale": 0.0,
    },
))

register_experiment(ExperimentCfg(
    name="validation_obs_v1_short",
    description="Short A/B validation run for heading-frame v1 observations",
    group="validation",
    task="Isaac-Iris-MA6-Direct-V1-v0",
    total_timesteps=20000,
    env_overrides={},
))


# ===========================================================================
# Ticket 042: Per-channel ego-motion EKF latency A/B
# ===========================================================================
# Validates the per-channel EKF lag patch from ticket 041's measurement.
# Same task, same RNG, same curriculum. Two configs only differ in whether
# ego motion uses the legacy bulk 5±2 ms lag or the per-channel values from
# ekf_state_lag.json (pos/vel 0 ms, orient 18 ms, ang_vel 15 ms, lin_acc 35 ms).
# Decision rule: treatment within ±5% of baseline on every tracked metric.

register_experiment(ExperimentCfg(
    name="validation_obs_v2_short_baseline_bulk",
    description="Ticket 042 A/B (control): legacy bulk 5±2 ms ego motion lag (post-patch code, pre-patch behavior)",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "delay_system_params.use_bulk_ego_motion_latency": True,
    },
))

register_experiment(ExperimentCfg(
    name="validation_obs_v2_short_treatment_per_channel",
    description="Ticket 042 A/B (treatment): per-channel ego motion lag from ekf_state_lag.json (post-patch default)",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "delay_system_params.use_bulk_ego_motion_latency": False,
    },
))


# ===========================================================================
# Ticket 043: prev-action obs + first-order LP on cmd_vel A/B
# ===========================================================================
# Three-way A/B comparing the architectural smoothness intervention against
# the pre-patch baseline. Same task, same RNG, same curriculum. Configs only
# differ in the two feature flags (enable_prev_action_obs / enable_action_lowpass).
# Acceptance: full ≥ 30% cmd_vel_delta reduction vs baseline at 200k, task
# quality within ±5%.

register_experiment(ExperimentCfg(
    name="validation_action_smoothness_short_baseline",
    description="Ticket 043 A/B (control): pre-patch behavior (no prev-action obs, no cmd_vel LP)",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_prev_action_obs": False,
        "enable_action_lowpass": False,
    },
))

register_experiment(ExperimentCfg(
    name="validation_action_smoothness_short_prev_action_only",
    description="Ticket 043 A/B (diagnostic): prev-action obs ON, cmd_vel LP OFF — isolates the obs-channel contribution",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_prev_action_obs": True,
        "enable_action_lowpass": False,
    },
))

register_experiment(ExperimentCfg(
    name="validation_action_smoothness_short_full",
    description="Ticket 043 A/B (treatment): prev-action obs + cmd_vel LP both ON — the shipping candidate",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_prev_action_obs": True,
        "enable_action_lowpass": True,
    },
))


# ===========================================================================
# Ticket 044: per-channel slew-rate clip on raw actions (PX4-aligned) A/B
# ===========================================================================
# Two-way A/B against the post-043 default (prev_action_only). Same task, same
# RNG, same curriculum. PX4-strict δ defaults (vel_xy=0.020, vel_z=0.053, ...)
# baked into IrisMA6TestEnvCfg per Slice-0 decision. Treatment is expected to
# fail the task-quality bar (Slice-0 evidence shows trained policy at 9-15×
# the PX4 envelope on velocity); the regression triggers the Slice-4
# task-difficulty calibration follow-up.

register_experiment(ExperimentCfg(
    name="validation_action_slew_short_baseline",
    description="Ticket 044 A/B (control): slew clip OFF, prev-action obs ON (post-043 default)",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_action_slew_clip": False,
        "enable_prev_action_obs": True,
        "enable_action_lowpass": False,
    },
))

register_experiment(ExperimentCfg(
    name="validation_action_slew_short_treatment",
    description="Ticket 044 A/B (treatment): slew clip ON (PX4-strict δ), prev-action obs ON",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_action_slew_clip": True,
        "enable_prev_action_obs": True,
        "enable_action_lowpass": False,
    },
))


# ===========================================================================
# Ticket 045: task-difficulty calibration to match PX4-strict slew envelope
# ===========================================================================
# Coordinated downshift of the velocity envelope so the PX4-strict slew clip
# from ticket 044 becomes achievable for the multi-agent triangulation task.
# Three knobs scale together:
#   - max_lin_vel:                10 → 5 m/s  (×0.5)
#   - action_slew_vel_xy:         0.020 → 0.040  (×2.0; holds physical accel
#                                              constant at MPC_ACC_HOR_MAX = 5 m/s²)
#   - target_controller.max_speed_end: 5.0 → 2.5 m/s  (×0.5; preserves 2:1
#                                              agent-vs-target speed ratio)
# t044 result anchor: at the original envelope, the slew clip won the
# smoothness battle (-69% action_delta) but cratered triangulation (-49%),
# tracking_lost (+253%), and collision_fraction (+122%). All 7 channels
# saturated at 92-99% — cross-channel adaptation under velocity starvation.
# Hypothesis: a coordinated downshift gives the policy headroom for
# bearing-change maneuvers within the slew bandwidth budget.

register_experiment(ExperimentCfg(
    name="validation_task_difficulty_baseline",
    description="Ticket 045 A/B (control): t044 cfg unchanged (max_lin_vel=10, slew δ_xy=0.020). The t044 in-flight run's TB IS this data through 204k; this entry exists for reproducibility, not as a launch target",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_action_slew_clip": True,
        "enable_prev_action_obs": True,
        "enable_action_lowpass": False,
        # max_lin_vel + action_slew_vel_xy + target_controller.max_speed_end
        # all at cfg defaults (10 / 0.020 / 5.0).
    },
))

register_experiment(ExperimentCfg(
    name="validation_task_difficulty_treatment",
    description="Ticket 045 A/B (treatment): coordinated envelope downshift — max_lin_vel=5, slew δ_xy=0.040 (physical PX4-strict preserved), target max_speed_end=2.5 (preserves 2:1 ratio)",
    group="validation",
    total_timesteps=400000,
    seeds=[42],
    env_overrides={
        "enable_action_slew_clip": True,
        "enable_prev_action_obs": True,
        "enable_action_lowpass": False,
        "max_lin_vel": 5.0,
        "action_slew_vel_xy": 0.040,
        "target_controller.max_speed_end": 2.5,
    },
))


# ===========================================================================
# Ticket 046: closer spawn + 2D target motion (task-difficulty reduction P1)
# ===========================================================================
# Two-way A/B on the t045 envelope (max_lin_vel=5 + slew δ_xy=0.040). Both
# share the post-t044/045 slew clip + prev-action obs posture. Baseline
# restores the pre-046 spawn geometry (cylinder_diameter_max=60,
# target_distance_max=25, target_height_offset_max=4) and 3D target motion;
# treatment uses the new lower cfg defaults (30/15/2) and clamps the target
# velocity to the horizontal plane via enable_z_motion=False. Acceptance:
# pair_valid_rate ≥ 0.95, triangulation ≥ 32, ≥2 channels with slew_sat ≤ 0.70,
# reward ≥ t045 reward at 200k.

register_experiment(ExperimentCfg(
    name="validation_task_geom_baseline",
    description="Ticket 046 A/B (control): pre-046 spawn geometry (diam=60, dist=25, h_off=4) + 3D target motion",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_action_slew_clip": True,
        "enable_prev_action_obs": True,
        "enable_action_lowpass": False,
        "max_lin_vel": 5.0,
        "action_slew_vel_xy": 0.040,
        "target_controller.max_speed_end": 2.5,
        # Restore pre-046 spawn geometry (env-cfg state before this ticket).
        "initial_states.cylinder_diameter_max": 60.0,
        "initial_states.target_distance_max": 25.0,
        "initial_states.target_height_offset_max": 4.0,
        # 3D target motion (default).
        "target_controller.enable_z_motion": True,
    },
))

register_experiment(ExperimentCfg(
    name="validation_task_geom_treatment",
    description="Ticket 046 A/B (treatment): closer spawn (diam=30, dist=15, h_off=2) + 2D target motion (enable_z_motion=False)",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    env_overrides={
        "enable_action_slew_clip": True,
        "enable_prev_action_obs": True,
        "enable_action_lowpass": False,
        "max_lin_vel": 5.0,
        "action_slew_vel_xy": 0.040,
        "target_controller.max_speed_end": 2.5,
        # Closer spawn (already the new cfg defaults; restated for explicitness).
        "initial_states.cylinder_diameter_max": 30.0,
        "initial_states.target_distance_max": 15.0,
        "initial_states.target_height_offset_max": 2.0,
        # 2D target motion (the load-bearing flag for the treatment).
        "target_controller.enable_z_motion": False,
    },
))


# ===========================================================================
# Ticket 048: network width sweep (MLP encoder + GRU hidden, MAPPO-RNN)
# ===========================================================================
# Agent-cfg-only sweep. Varies hidden_size and gru_hidden_size jointly at a
# 1:1 ratio across three scales; gru_num_layers held at 1. No env-side change:
# slew clip, prev-action obs, spawn geometry, reward weights, EKF lag all stay
# at the current shipping cfg defaults (post-045/046/047). The only variable is
# policy/value capacity. Mirrors policy overrides onto value so the encoder and
# GRU scale together and neither becomes a bottleneck.
#
# Hydra override paths follow the existing sweep_agents_n* convention
# (models.policy.hidden_size / models.policy.gru_hidden_size, mirrored to value).
# Sequential A/B at 200k x seed=42 per config (~24 wall-h each, ~72h total),
# single GPU. Acceptance: at least one widened config reaches triangulation >= 32
# at 200k, smoothness (total_rms) within +-10% of baseline width, collision_fraction
# <= 2x t043 baseline, wall-time/step on the Pareto winner <= 2.5x baseline.

register_experiment(ExperimentCfg(
    name="validation_net_width_baseline",
    description="Ticket 048 (control): current shipping width — hidden_size=64, gru_hidden_size=64, gru_num_layers=1 (policy + value)",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    agent_overrides={
        "models.policy.hidden_size": 64,
        "models.policy.gru_hidden_size": 64,
        "models.value.hidden_size": 64,
        "models.value.gru_hidden_size": 64,
    },
))

register_experiment(ExperimentCfg(
    name="validation_net_width_mid",
    description="Ticket 048 (treatment): mid width — hidden_size=128, gru_hidden_size=128, gru_num_layers=1 (policy + value), ~3.7x params",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    agent_overrides={
        "models.policy.hidden_size": 128,
        "models.policy.gru_hidden_size": 128,
        "models.value.hidden_size": 128,
        "models.value.gru_hidden_size": 128,
    },
))

register_experiment(ExperimentCfg(
    name="validation_net_width_wide",
    description="Ticket 048 (treatment): wide width — hidden_size=256, gru_hidden_size=256, gru_num_layers=1 (policy + value), ~14.5x params",
    group="validation",
    total_timesteps=200000,
    seeds=[42],
    agent_overrides={
        "models.policy.hidden_size": 256,
        "models.policy.gru_hidden_size": 256,
        "models.value.hidden_size": 256,
        "models.value.gru_hidden_size": 256,
    },
))


# ===========================================================================
# Sweeps: Agent Count
# ===========================================================================

for _n_agents in [2, 3, 4]:
    _obs_dim = _compute_obs_dim(enable_triangulation=True)
    _agent_ovr = (
        {
            "models.policy.hidden_size": 64,
            "models.policy.gru_hidden_size": 64,
            "models.value.hidden_size": 64,
            "models.value.gru_hidden_size": 64,
        }
        if _n_agents >= 3
        else {}
    )
    register_experiment(ExperimentCfg(
        name=f"sweep_agents_n{_n_agents}",
        description=f"Agent count sweep: N={_n_agents}",
        group="agent_sweep",
        env_overrides={
            "num_agents": _n_agents,
        },
        agent_overrides=_agent_ovr,
    ))


# ===========================================================================
# Sweeps: Communication Delay
# ===========================================================================

for _delay_ms in [200, 500, 800]:
    _delay_s = _delay_ms / 1000.0
    register_experiment(ExperimentCfg(
        name=f"sweep_delay_{_delay_ms}ms",
        description=f"Comm delay sweep: {_delay_ms}ms mean",
        group="delay_sweep",
        env_overrides={
            "delay_system_params.other_latency_mean": _delay_s,
        },
    ))


# ===========================================================================
# Sweeps: Detection Noise
# ===========================================================================

for _sigma_px in [0, 3, 7, 14]:
    register_experiment(ExperimentCfg(
        name=f"sweep_noise_{_sigma_px}px",
        description=f"BBox noise sweep: {_sigma_px}px std",
        group="noise_sweep",
        env_overrides={
            "delay_system_params.noise_bbox_std": float(_sigma_px),
        },
    ))


# ===========================================================================
# Sweeps: Target Speed
# ===========================================================================

for _speed, _min_speed in [(3, 2), (5, 3), (10, 5)]:
    register_experiment(ExperimentCfg(
        name=f"sweep_target_speed_{_speed}",
        description=f"Target speed sweep: max {_speed} m/s",
        group="speed_sweep",
        env_overrides={
            "max_lin_vel": float(_speed),
            "max_lin_vel_min": float(_min_speed),
        },
    ))


# ===========================================================================
# A9: Bbox Size Reward Ablation (ticket-014)
# ===========================================================================
# Does the policy discover optimal zoom from miss rate + triangulation signals
# alone, without the heuristic bbox_size_reward (target area = 20% of image)?

register_experiment(ExperimentCfg(
    name="a9_bbox_size_baseline",
    description="Bbox size reward enabled (default scale=60, baseline)",
    group="A9",
    env_overrides={},  # Default: bbox_size_reward_scale=60.0
))

register_experiment(ExperimentCfg(
    name="a9_no_bbox_size",
    description="Bbox size reward disabled — policy learns zoom from miss rate + triangulation",
    group="A9",
    env_overrides={
        "bbox_size_reward_scale": 0.0,
    },
))


# ===========================================================================
# A8: Action Smoothness (Weight Tuning)
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a8_weight_config_a",
    description="Conservative 5x action penalty increase, zoom weight 1.0 (new defaults)",
    group="A8",
    total_timesteps=150000,
    env_overrides={
        "action_sum_penalty_scale": -10.0,
        "action_delta_penalty_scale": -5.0,
        "action_weight": [1, 1, 5, 1, 0.5, 0.5, 1.0],
        "action_delta_weight": [1, 1, 1, 1, 0.5, 0.5, 1.0],
    },
))

register_experiment(ExperimentCfg(
    name="a8_weight_config_b",
    description="Aggressive 15x action penalty increase, zoom weight 1.0",
    group="A8",
    total_timesteps=150000,
    env_overrides={
        "action_sum_penalty_scale": -30.0,
        "action_delta_penalty_scale": -15.0,
        "action_weight": [1, 1, 5, 1, 0.5, 0.5, 1.0],
        "action_delta_weight": [1, 1, 1, 1, 0.5, 0.5, 1.0],
    },
))


# ===========================================================================
# Experiment Suites
# ===========================================================================

register_suite(ExperimentSuiteCfg(
    name="iros2026_must",
    experiments=[
        "a1_no_delay", "a1_stochastic_delay",
        "a3_clean", "a3_delay_only", "a3_noise_only", "a3_noisy_delay",
        "a5_no_cbf", "a5_cbf_default",
        "a7_level1_fim", "a7_level2_gt_anchored", "a7_curriculum_blend",
        "baseline_greedy",
    ],
))

register_suite(ExperimentSuiteCfg(
    name="iros2026_should",
    experiments=[
        "a4_analytical", "a4_analytical_delay",
        "a5_cbf_lambda_0.5", "a5_cbf_lambda_2.0", "a5_cbf_lambda_5.0",
        "a6_no_gain_rand", "a6_gain_rand_default",
        "a7_e2e_weight_0.0", "a7_e2e_weight_0.3", "a7_e2e_weight_0.7", "a7_e2e_weight_1.0",
    ],
))

# ===========================================================================
# Phase A: Minimum Viable Task (MVT)
# Narrowed initial-state distribution, non-obs curriculum ramps frozen.
# Isolates observation-corruption (noise/delay/dropout/burst) as the sole
# study axis en route to sim-to-sim (IsaacSim + PX4 + ROS2) transfer.
# ===========================================================================

register_experiment(ExperimentCfg(
    name="phase_a_mvt_baseline",
    description=(
        "Phase-A MVT baseline: narrowed init-state distribution, non-obs "
        "curriculum frozen, obs-corruption ramps (100k-220k) live."
    ),
    group="phase_a",
    task="Isaac-Iris-MA6-Direct-MVT-v0",
    env_overrides={},
    agent_overrides={},
    seeds=[42],
    total_timesteps=400_000,
))


register_suite(ExperimentSuiteCfg(
    name="iros2026_sweeps",
    experiments=(
        list_experiments("agent_sweep")
        + list_experiments("delay_sweep")
        + list_experiments("noise_sweep")
        + list_experiments("speed_sweep")
    ),
))

register_suite(ExperimentSuiteCfg(
    name="iros2026_full",
    experiments=(
        list_experiments("A1") + list_experiments("A3") + list_experiments("A4")
        + list_experiments("A5") + list_experiments("A6") + list_experiments("A7")
        + list_experiments("baseline")
        + list_experiments("agent_sweep") + list_experiments("delay_sweep")
        + list_experiments("noise_sweep") + list_experiments("speed_sweep")
    ),
))
