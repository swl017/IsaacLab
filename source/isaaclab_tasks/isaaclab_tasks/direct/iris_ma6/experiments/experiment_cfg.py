# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment configuration dataclasses for iris_ma6 ablation studies."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List

from isaaclab.utils import configclass


@configclass
class ExperimentCfg:
    """Configuration for a single experiment run.

    Each experiment is defined as a set of overrides on top of the base
    IrisMA6TestEnvCfg and agent config. This enables full reproducibility --
    the experiment is completely determined by (base_config + overrides + seed).
    """

    # Identification
    name: str = ""
    """Unique experiment name (e.g., 'a1_no_delay', 'a5_cbf_default')."""

    description: str = ""
    """Human-readable description for logging."""

    group: str = ""
    """Experiment group (e.g., 'A1', 'A3', 'A5', 'baseline', 'sweep')."""

    # Environment overrides (applied to IrisMA6TestEnvCfg)
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    """Dictionary of dotted-path overrides for IrisMA6TestEnvCfg.

    Example: {'curriculum.fixed_delay_start_step': 0, 'task_reward_level': 2}
    """

    # Agent config overrides (applied to agent YAML dict)
    agent_overrides: Dict[str, Any] = field(default_factory=dict)
    """Dictionary of dotted-path overrides for agent config dict.

    Example: {'models.policy.gru_hidden_size': 128, 'agent.learning_rate': 0.0003}
    """

    # Training params
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    """Random seeds for multi-seed runs."""

    num_envs: int = 4096
    """Number of parallel environments."""

    total_timesteps: int = 400000
    """Total training timesteps."""

    # Evaluation params
    eval_num_envs: int = 256
    """Number of envs for evaluation rollouts."""

    eval_episodes: int = 100
    """Number of episodes for evaluation."""


@configclass
class ExperimentSuiteCfg:
    """Configuration for a batch of experiments."""

    name: str = ""
    """Suite name (e.g., 'iros2026_full', 'a5_sweep')."""

    experiments: List[str] = field(default_factory=list)
    """List of experiment names to run (references registry keys)."""

    parallel_seeds: bool = True
    """If True, seeds can be run in parallel (different GPU). If False, sequential."""

    base_log_dir: str = "logs/experiments"
    """Base directory for experiment logs."""
