# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Initial states module for iris_ma6 environment.

This module provides curriculum-driven randomization of agent and target
initial states with cylinder-based placement and designated observer support.

Example:
    ```python
    from isaaclab_tasks.direct.iris_ma6.initial_states import (
        InitialStatesCfg,
        InitialStates,
    )

    # Create configuration
    cfg = InitialStatesCfg(
        cylinder_diameter_min=30.0,
        cylinder_diameter_max=100.0,
        target_distance_min=30.0,
        target_distance_max=200.0,
    )

    # Create generator
    initial_states = InitialStates(
        cfg=cfg,
        num_envs=256,
        num_agents=3,
        device=torch.device("cuda"),
    )

    # Generate at different curriculum stages
    result_easy = initial_states.generate(curriculum_progress=0.0)
    result_hard = initial_states.generate(curriculum_progress=1.0)
    ```
"""

from .initial_states_cfg import InitialStatesCfg, InitialStatesResult
from .initial_states_generator import InitialStatesGenerator
from .initial_states import InitialStates

__all__ = [
    "InitialStatesCfg",
    "InitialStatesResult",
    "InitialStatesGenerator",
    "InitialStates",
]
