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
