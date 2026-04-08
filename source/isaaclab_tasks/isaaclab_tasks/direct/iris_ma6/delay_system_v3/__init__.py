# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delay System V3: Simplified multi-agent delay simulation.

This module provides a unified delay system with:
- Per-step, per-env, per-episode configurable latency, staleness, and dropout
- Timestamp guarantees: data and timestamp always travel together
- Curriculum learning support with mode/progress control
- Clean API for multi-agent state management

Key classes:
- MultiAgentDelaySystemV3: Main entry point for multi-agent delay simulation
- UnifiedDelaySystem: Lower-level unified system for single/multi field delay
- DelayPipelineV3: Core pipeline with timestamp guarantees

Configuration:
- MultiAgentDelayCfgV3: Full configuration for multi-agent system
- UnifiedDelayCfgV3: Configuration for unified delay system
- DelayPipelineCfgV3: Pipeline-level configuration

Data structures:
- AgentStates: Container for agent state data
- AgentStatesData: Raw data container

Usage:
    from isaaclab_tasks.direct.iris_ma6.delay_system_v3 import (
        MultiAgentDelaySystemV3,
        MultiAgentDelayCfgV3,
    )

    # Create configuration
    cfg = MultiAgentDelayCfgV3()

    # Create delay system
    delay_system = MultiAgentDelaySystemV3(
        cfg=cfg,
        possible_agents=["agent_0", "agent_1"],
        num_envs=64,
        num_joints=3,
        num_targets=2,
        device=device,
    )

    # Each step:
    delay_system.set_time(t_current)
    delay_system.update_ground_truth(agent_id, states)

    # Get delayed states
    reward_states = delay_system.get_all_states_for_rewards(ego_agent_id)
    obs_states = delay_system.get_all_states_for_observations(ego_agent_id)

    # Curriculum control
    delay_system.set_delay_mode("random", progress=0.5)
    delay_system.set_dropout_rate(0.05)
"""

# Configuration
from .delay_cfg_v3 import (
    DistributionCfg,
    SamplingCfg,
    LatencyCfg,
    StalenessCfg,
    DropoutCfg,
    NoiseCfg,
    FirstOrderLagCfg,
    PerAgentDelayCfg,
    RewardStateCfg,
    DelayPipelineCfgV3,
    PerspectiveCfg,
    UnifiedDelayCfgV3,
    MultiAgentDelayCfgV3,
    # Key parameters (simplified interface)
    DelaySystemKeyParams,
    create_delay_cfg_from_params,
    # Preset configurations
    create_no_delay_cfg,
    create_fixed_delay_cfg,
    create_random_delay_cfg,
)

# Sampling strategies
from .sampling_strategies import (
    DistributionSampler,
    ParameterSampler,
    LatencySampler,
    StalenessSampler,
    DropoutSampler,
    # Per-agent variants
    PerAgentDistributionSampler,
    PerAgentParameterSampler,
    PerAgentLatencySampler,
    PerAgentStalenessSampler,
)

# Storage
from .field_storage import FieldStorage, MultiFieldStorage

# Core pipeline
from .delay_pipeline_v3 import DelayPipelineV3

# Unified system
from .delay_system_v3 import UnifiedDelaySystem, DelayedFieldAccessor

# Burst dropout
from .burst_dropout import BurstDropoutCfg, BurstDropoutSampler

# Multi-agent wrapper
from .multi_agent_wrapper import MultiAgentDelaySystemV3

# Data structures
from .agent_states import AgentStates, AgentStatesData, MultiAgentStates

# Derived field computation
from .derived_field_computers import (
    compute_camera_orientation_from_gimbal,
    compute_camera_position,
)

__all__ = [
    # Configuration
    "DistributionCfg",
    "SamplingCfg",
    "LatencyCfg",
    "StalenessCfg",
    "DropoutCfg",
    "NoiseCfg",
    "FirstOrderLagCfg",
    "PerAgentDelayCfg",
    "RewardStateCfg",
    "DelayPipelineCfgV3",
    "PerspectiveCfg",
    "UnifiedDelayCfgV3",
    "MultiAgentDelayCfgV3",
    # Key parameters (simplified interface)
    "DelaySystemKeyParams",
    "create_delay_cfg_from_params",
    # Preset configurations
    "create_no_delay_cfg",
    "create_fixed_delay_cfg",
    "create_random_delay_cfg",
    # Sampling
    "DistributionSampler",
    "ParameterSampler",
    "LatencySampler",
    "StalenessSampler",
    "DropoutSampler",
    # Per-agent sampling
    "PerAgentDistributionSampler",
    "PerAgentParameterSampler",
    "PerAgentLatencySampler",
    "PerAgentStalenessSampler",
    # Storage
    "FieldStorage",
    "MultiFieldStorage",
    # Burst dropout
    "BurstDropoutCfg",
    "BurstDropoutSampler",
    # Core
    "DelayPipelineV3",
    "UnifiedDelaySystem",
    "DelayedFieldAccessor",
    "MultiAgentDelaySystemV3",
    # Data structures
    "AgentStates",
    "AgentStatesData",
    "MultiAgentStates",
    # Utilities
    "compute_camera_orientation_from_gimbal",
    "compute_camera_position",
]
