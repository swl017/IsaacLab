# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Delay System V2 for Multi-Agent Environments.

This package provides a **self-contained** multi-agent delay system with
field-based DelaySystemV2 architecture. It supports:

- Arbitrary number of agents
- Dual pipeline (clean for rewards, noisy for observations)
- Perspective-aware delays (fast ego, slow inter-agent)
- First-order lag filtering for motion/orientation/joints
- Stochastic delays for detection (staleness, latency, dropout)
- Derived field computation (camera geometry, ray directions)

This package is self-contained and does not depend on quadcopter/delay_system.
"""

# Core delay system components (self-contained)
from .delay_cfg import (
    DelayCfg,
    DelaySystemCfg,
    DelaySystemCfgV2,
    DistributionCfg,
    FieldDelayCfg,
    NoiseCfg,
)
from .delay_pipeline import DelayPipeline
from .delay_system_v2 import DelaySystemV2
from .data_bus import DataBus
from .derived_fields import DerivedFieldComputer, DerivedFieldDef

# Multi-agent components
from .agent_states import AgentStates, AgentStatesData, MultiAgentStates
from .multi_agent_delay_system_v2 import MultiAgentDelaySystemV2
from .multi_agent_delay_system_v2_cfg import MultiAgentDelaySystemV2Cfg

# Derived field computers
from .derived_field_computers import (
    compute_camera_orientation_from_gimbal,
    compute_camera_position,
    compute_combined_angular_velocity,
    compute_ray_directions_from_bbox,
    compute_ray_origins,
)

__all__ = [
    # Core delay system (self-contained)
    "DelayCfg",
    "DelaySystemCfg",
    "DelaySystemCfgV2",
    "DistributionCfg",
    "FieldDelayCfg",
    "NoiseCfg",
    "DelayPipeline",
    "DelaySystemV2",
    "DataBus",
    "DerivedFieldComputer",
    "DerivedFieldDef",
    # Multi-agent classes
    "MultiAgentDelaySystemV2",
    "MultiAgentDelaySystemV2Cfg",
    # Data structures
    "AgentStates",
    "AgentStatesData",
    "MultiAgentStates",
    # Derived field computers
    "compute_camera_orientation_from_gimbal",
    "compute_camera_position",
    "compute_combined_angular_velocity",
    "compute_ray_directions_from_bbox",
    "compute_ray_origins",
]
