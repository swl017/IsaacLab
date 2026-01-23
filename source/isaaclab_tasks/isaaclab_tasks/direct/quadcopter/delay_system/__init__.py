# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delay system module for simulating action and observation latency in RL environments.

This module provides two versions of the delay system:

1. **Legacy (v1)**: Simple step-based delays using Isaac Lab's DelayBuffer.
   - Classes: `DelayCfg`, `DelaySystemCfg`, `DelaySystem`

2. **Enhanced (v2)**: Full data bus architecture with per-field configuration.
   - Classes: `DistributionCfg`, `NoiseCfg`, `FieldDelayCfg`, `DelaySystemCfgV2`, `DelaySystemV2`
   - Additional: `DataBus`, `DelayPipeline`, `DerivedFieldComputer`, `DerivedFieldDef`

The enhanced version supports:
- First-order lag filtering (dynamics delay)
- Staleness (sample-and-hold at lower rate)
- Random latency (transport delay)
- Dropout (packet loss)
- Per-step Gaussian noise injection
- Clean/noisy data separation for reward vs observation computation
- Derived field computation from delayed raw fields

Example (Legacy):
    from isaaclab_tasks.direct.quadcopter.delay_system import DelaySystem, DelaySystemCfg, DelayCfg

    cfg = DelaySystemCfg(
        action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
    )
    system = DelaySystem(cfg, num_envs=4096, action_dim=4, observation_dim=12, device="cuda:0")

Example (Enhanced):
    from isaaclab_tasks.direct.quadcopter.delay_system import (
        DelaySystemV2, DelaySystemCfgV2, FieldDelayCfg, DistributionCfg, NoiseCfg
    )

    cfg = DelaySystemCfgV2(
        dt=0.01,
        use_enhanced_mode=True,
        field_configs={
            "imu_accel": FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=0.005,
                noise=NoiseCfg(enabled=True, std=0.1),
            ),
        },
    )
    system = DelaySystemV2(cfg, num_envs=4096, device="cuda:0", field_dims={"imu_accel": 3})
"""

# Legacy (v1) exports
from .delay_cfg import DelayCfg, DelaySystemCfg
from .delay_system import DelaySystem

# Enhanced (v2) configuration exports
from .delay_cfg import DistributionCfg, FieldDelayCfg, NoiseCfg, DelaySystemCfgV2

# Enhanced (v2) core components
from .data_bus import DataBus
from .delay_pipeline import DelayPipeline
from .delay_system_v2 import DelaySystemV2
from .derived_fields import DerivedFieldComputer, DerivedFieldDef, QUADCOPTER_DERIVED_FIELDS

__all__ = [
    # Legacy (v1)
    "DelayCfg",
    "DelaySystemCfg",
    "DelaySystem",
    # Enhanced (v2) configuration
    "DistributionCfg",
    "NoiseCfg",
    "FieldDelayCfg",
    "DelaySystemCfgV2",
    # Enhanced (v2) core components
    "DataBus",
    "DelayPipeline",
    "DelaySystemV2",
    "DerivedFieldComputer",
    "DerivedFieldDef",
    "QUADCOPTER_DERIVED_FIELDS",
]
