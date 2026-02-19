# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment configuration dataclasses for IROS 2026 ablation studies."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List

from isaaclab.utils import configclass


@configclass
class ExperimentCfg:
    """Configuration for a single experiment run.

    Each experiment is defined as a set of overrides on top of the base
    IrisMAEnvCfg and agent config. This enables full reproducibility --
    the experiment is completely determined by (base_config + overrides + seed).
    """

    # Identification
    name: str = ""
    """Unique experiment name (e.g., 'a1_no_delay', 'a2_mlp')."""

    description: str = ""
    """Human-readable description for logging."""

    group: str = ""
    """Experiment group (e.g., 'A1', 'A2', 'baseline', 'sweep')."""

    # Environment overrides (applied to IrisMAEnvCfg)
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    """Dictionary of dotted-path overrides for IrisMAEnvCfg.

    Example: {'curriculum.fixed_delay_start_step': 0, 'reward_mode': 'heuristic'}
    """

    # Agent config overrides (applied to agent YAML dict)
    agent_overrides: Dict[str, Any] = field(default_factory=dict)
    """Dictionary of dotted-path overrides for agent config dict.

    Example: {'models.policy.gru_num_layers': 0, 'agent.learning_rate': 0.0003}
    """

    # Training params
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    """Random seeds for multi-seed runs."""

    num_envs: int = 4096
    """Number of parallel environments."""

    total_timesteps: int = 200000
    """Total training timesteps."""

    # Architecture selection
    use_mlp_model: bool = False
    """If True, use MLP-only policy/value models instead of RNN."""

    # Evaluation params
    eval_num_envs: int = 256
    """Number of envs for evaluation rollouts."""

    eval_episodes: int = 100
    """Number of episodes for evaluation."""


@configclass
class ExperimentSuiteCfg:
    """Configuration for a batch of experiments."""

    name: str = ""
    """Suite name (e.g., 'iros2026_full', 'a1_sweep')."""

    experiments: List[str] = field(default_factory=list)
    """List of experiment names to run (references registry keys)."""

    parallel_seeds: bool = True
    """If True, seeds can be run in parallel (different GPU). If False, sequential."""

    base_log_dir: str = "logs/experiments"
    """Base directory for experiment logs."""
