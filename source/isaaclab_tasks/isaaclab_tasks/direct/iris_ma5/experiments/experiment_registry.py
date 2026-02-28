# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central registry of all IROS 2026 experiments.

Experiments are organized by ablation study:
    A1 - Delay modeling (with/without delay, AoI, delay sweep)
    A2 - RNN vs MLP architecture
    A3 - Dual-path (clean vs noisy reward pipeline)
    A4 - Covariance reward (analytical vs angular-only vs heuristic vs image-only)
    Baselines - Gavin2024-style, greedy orbit
    Sweeps - Agent count, detection noise
"""

from __future__ import annotations

from typing import Dict, Optional

from .experiment_cfg import ExperimentCfg, ExperimentSuiteCfg

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

def _compute_obs_dim(n_agents: int) -> int:
    """Compute observation dimension for N agents.

    Layout: ego(26) + other_fields(15*(N-1)) + tri_tail(6)
    """
    return 26 + 15 * (n_agents - 1) + 6


# ===========================================================================
# A1: Delay Modeling Ablation
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a1_no_delay",
    description="No delay, no noise, no dropout (clean baseline)",
    group="A1",
    env_overrides={
        # Push all delay/noise curriculum steps beyond training length
        "curriculum.noise_start_step": 999999,
        "curriculum.noise_end_step": 999999,
        "curriculum.fixed_delay_start_step": 999999,
        "curriculum.fixed_delay_end_step": 999999,
        "curriculum.random_delay_start_step": 999999,
        "curriculum.random_delay_end_step": 999999,
        "curriculum.dropout_start_step": 999999,
        "curriculum.dropout_end_step": 999999,
        "enable_noise_in_observations": False,
    },
))

register_experiment(ExperimentCfg(
    name="a1_stochastic_delay",
    description="Full stochastic delay (80-150ms) with curriculum",
    group="A1",
    env_overrides={
        # Default curriculum with explicit 80-150ms latency range
        "detection_mean_latency": 0.115,
        "detection_std_latency": 0.035,
        "comm_mean_latency": 0.115,
        "comm_std_latency": 0.035,
    },
))

register_experiment(ExperimentCfg(
    name="a1_with_aoi",
    description="Full system with AoI in observations (default)",
    group="A1",
    env_overrides={
        "mask_aoi_in_obs": False,
    },
))

register_experiment(ExperimentCfg(
    name="a1_without_aoi",
    description="Full system with AoI fields zeroed out in observations",
    group="A1",
    env_overrides={
        "mask_aoi_in_obs": True,
    },
))

register_experiment(ExperimentCfg(
    name="a1_curriculum_no_delay",
    description="Full curriculum (tracking/safety/coordination/moving-target) but zero delay",
    group="A1",
    env_overrides={
        # Disable delay/noise/dropout curriculum phases
        # "curriculum.noise_start_step": 999999,
        # "curriculum.noise_end_step": 999999,
        # "curriculum.fixed_delay_start_step": 999999,
        # "curriculum.fixed_delay_end_step": 999999,
        # "curriculum.random_delay_start_step": 999999,
        # "curriculum.random_delay_end_step": 999999,
        # "curriculum.dropout_start_step": 999999,
        # "curriculum.dropout_end_step": 999999,
        # Explicitly zero latencies
        "detection_fps_mean": 100.0,
        "detection_fps_std": 0.0,
        "detection_mean_latency": 0.0,
        "detection_std_latency": 0.0,
        "comm_mean_latency": 0.0,
        "comm_std_latency": 0.0,
        "enable_noise_in_observations": True,
        # tracking/safety/coordination/moving_target curriculum stays at defaults
    },
))

# A1 delay sweep: 0, 50, 100, 150, 200ms
for _delay_ms in [0, 50, 100, 150, 200]:
    _delay_s = _delay_ms / 1000.0
    register_experiment(ExperimentCfg(
        name=f"a1_delay_sweep_{_delay_ms}ms",
        description=f"Fixed delay sweep at {_delay_ms}ms",
        group="A1_sweep",
        env_overrides={
            # Disable curriculum - use fixed delay from start
            "curriculum.noise_start_step": 0,
            "curriculum.noise_end_step": 0,
            "curriculum.fixed_delay_start_step": 0,
            "curriculum.fixed_delay_end_step": 0,
            # Keep random delay disabled for sweep (deterministic delay)
            "curriculum.random_delay_start_step": 999999,
            "curriculum.random_delay_end_step": 999999,
            "curriculum.dropout_start_step": 999999,
            "curriculum.dropout_end_step": 999999,
            "detection_mean_latency": _delay_s,
            "detection_std_latency": 0.001,  # ~deterministic
            "comm_mean_latency": _delay_s,
            "comm_std_latency": 0.001,
        },
    ))


# ===========================================================================
# A2: RNN vs MLP
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a2_rnn",
    description="MAPPO-RNN (default architecture)",
    group="A2",
    use_mlp_model=False,
))

register_experiment(ExperimentCfg(
    name="a2_mlp",
    description="MAPPO-MLP with frame-skip stacking (K=4, skip=10, ~1.2s horizon)",
    group="A2",
    use_mlp_model=True,
    frame_stack=4,
    frame_skip=10,
    agent_overrides={
        "models.policy.hidden_size": 256,
        "models.policy.num_hidden_layers": 3,
        "agent.sequence_length": 1,
        "agent.learning_rate": 1e-4,
        "agent.grad_norm_clip": 1.0,
        "use_noisy_rewards": False,
    },
))


# ===========================================================================
# A3: Dual-Path Architecture
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a3_clean_reward",
    description="Clean-path reward (default dual-path, paper system)",
    group="A3",
    env_overrides={
        "use_noisy_rewards": False,
    },
))

register_experiment(ExperimentCfg(
    name="a3_noisy_reward",
    description="Noisy-path reward (ablation: rewards from delayed+noisy states)",
    group="A3",
    env_overrides={
        "use_noisy_rewards": True,
    },
))


# ===========================================================================
# A4: Covariance Reward
# ===========================================================================

register_experiment(ExperimentCfg(
    name="a4_analytical",
    description="Multi-source covariance reward (pixel+pose+gimbal+intrinsics, paper system)",
    group="A4",
    env_overrides={
        "reward_mode": "analytical",
    },
))

register_experiment(ExperimentCfg(
    name="a4_angular_only",
    description="Angular-noise-only covariance (Gavin2024-style, pixel noise only, no pose/gimbal/intrinsics)",
    group="A4",
    env_overrides={
        "reward_mode": "angular_only",
        "pos_std": 0.001,
        "intrinsics_std": 0.001,
        "triangulation_reward_scale": 5.0/7.0,
    },
))

register_experiment(ExperimentCfg(
    name="a4_heuristic",
    description="Heuristic reward (distance + viewing angle diversity)",
    group="A4",
    env_overrides={
        "reward_mode": "heuristic",
    },
))

register_experiment(ExperimentCfg(
    name="a4_image_only",
    description="Image-only reward (bbox center + size, no triangulation)",
    group="A4",
    env_overrides={
        "reward_mode": "image_only",
    },
))


# ===========================================================================
# Baselines
# ===========================================================================

register_experiment(ExperimentCfg(
    name="baseline_gavin2024",
    description="Gavin2024-style: MAPPO-MLP, no delay, no AoI, angular-only covariance, omnidirectional, no curriculum",
    group="baseline",
    use_mlp_model=True,
    env_overrides={
        # No delay modeling
        "curriculum.noise_start_step": 999999,
        "curriculum.noise_end_step": 999999,
        "curriculum.fixed_delay_start_step": 999999,
        "curriculum.fixed_delay_end_step": 999999,
        "curriculum.random_delay_start_step": 999999,
        "curriculum.random_delay_end_step": 999999,
        "curriculum.dropout_start_step": 999999,
        "curriculum.dropout_end_step": 999999,
        "enable_noise_in_observations": False,
        # No curriculum (all phases at full difficulty from start)
        "curriculum.tracking_start_step": 0,
        "curriculum.tracking_end_step": 0,
        "curriculum.safety_start_step": 0,
        "curriculum.safety_end_step": 0,
        "curriculum.moving_target_start_step": 0,
        "curriculum.moving_target_end_step": 0,
        "curriculum.coordination_start_step": 0,
        "curriculum.coordination_end_step": 0,
        "curriculum.dynamics_start_step": 999999,
        "curriculum.dynamics_end_step": 999999,
        # No AoI
        "mask_aoi_in_obs": True,
        # Angular-noise-only covariance (Gavin2024 uses Tr(Sigma_X) with angular noise only)
        "reward_mode": "angular_only",
        # Omnidirectional cameras — no FoV/bbox detection layer
        "use_omnidirectional_cameras": True,
        # Gavin2024 Eq. 15: r = 1/sqrt(Tr(Sigma_X)) if safe, else -r_penalty
        "use_gavin2024_reward": True,
        "gavin2024_d_threshold": 2.0,
        "gavin2024_r_penalty": 1.0,
    },
))

# baseline_greedy is eval-only — see baselines/greedy_policy.py
register_experiment(ExperimentCfg(
    name="baseline_greedy",
    description="Greedy equiangular orbit (eval-only, scripted policy)",
    group="baseline",
    env_overrides={},
))


# ===========================================================================
# SHOULD: Agent Count Sweep
# ===========================================================================

for _n_agents in [2, 3, 4]:
    _agent_ids = [f"drone_{i}" for i in range(_n_agents)]
    _obs_dim = _compute_obs_dim(_n_agents)
    # Scale network for larger observation spaces (n>=3: obs 62+, shared_obs 186+)
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
        description=f"Agent count sweep: N={_n_agents}, full system (AoI + delay + analytical covariance)",
        group="agent_sweep",
        env_overrides={
            # Agent geometry (must be consistent with num_cameras_per_env)
            "possible_agents": _agent_ids,
            "action_spaces": {aid: 7 for aid in _agent_ids},
            "observation_spaces": {aid: _obs_dim for aid in _agent_ids},
            # bbox_raycaster must match number of agents
            "bbox_raycaster.num_cameras_per_env": _n_agents,
            "use_noisy_rewards": True,
        },
        agent_overrides=_agent_ovr,
    ))


# ===========================================================================
# SHOULD: Delay Sweep (training-friendly, full curriculum)
# ===========================================================================

for _delay_ms in [200, 400, 600, 800]:
    _delay_s = _delay_ms / 1000.0
    register_experiment(ExperimentCfg(
        name=f"sweep_delay_{_delay_ms}ms",
        description=f"Delay sweep: {_delay_ms}ms mean latency, full curriculum",
        group="delay_sweep",
        env_overrides={
            # "detection_mean_latency": _delay_s,
            # "detection_std_latency": max(0.001, _delay_s * 0.1),
            "comm_mean_latency": _delay_s,
            "comm_std_latency": max(0.001, _delay_s * 0.1),
        },
    ))


# ===========================================================================
# SHOULD: Detection Noise Sweep
# ===========================================================================

for _sigma_px in [0, 1, 2, 4]:
    register_experiment(ExperimentCfg(
        name=f"sweep_noise_sigma{_sigma_px}px",
        description=f"Detection noise sweep: sigma_pixel={_sigma_px}",
        group="noise_sweep",
        env_overrides={
            "pix_std": float(_sigma_px),
        },
    ))


# ===========================================================================
# Experiment Suites
# ===========================================================================

register_suite(ExperimentSuiteCfg(
    name="iros2026_must",
    experiments=[
        "a1_no_delay", "a1_stochastic_delay", "a1_with_aoi", "a1_without_aoi",
        "a1_curriculum_no_delay",
        "a2_rnn", "a2_mlp",
        "a3_clean_reward", "a3_noisy_reward",
        "a4_analytical", "a4_angular_only", "a4_heuristic", "a4_image_only",
        "baseline_gavin2024",
    ],
))

register_suite(ExperimentSuiteCfg(
    name="iros2026_should",
    experiments=[
        "sweep_agents_n2", "sweep_agents_n3", "sweep_agents_n4",
        "sweep_delay_200ms", "sweep_delay_400ms", "sweep_delay_600ms", "sweep_delay_800ms",
        "sweep_noise_sigma0px", "sweep_noise_sigma1px", "sweep_noise_sigma2px",
        "sweep_noise_sigma4px",
    ],
))

register_suite(ExperimentSuiteCfg(
    name="delay_sweep",
    experiments=[f"sweep_delay_{d}ms" for d in [200, 400, 600, 800]],
))

register_suite(ExperimentSuiteCfg(
    name="a1_delay_sweep",
    experiments=[f"a1_delay_sweep_{d}ms" for d in [200, 400, 600, 800]],
))

register_suite(ExperimentSuiteCfg(
    name="iros2026_full",
    experiments=(
        list_experiments("A1") + list_experiments("A1_sweep")
        + list_experiments("A2") + list_experiments("A3") + list_experiments("A4")
        + list_experiments("baseline") + list_experiments("agent_sweep")
        + list_experiments("delay_sweep") + list_experiments("noise_sweep")
    ),
))
