# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Randomization module for iris_ma4 multi-agent environment.

This module provides distance-based formation generation where the primary
difficulty factor is the distance between agents and target.

Usage:
    ```python
    from isaaclab_tasks.direct.iris_ma4.randomization import (
        Randomizer,
        DistanceBasedFormationCfg,
    )

    cfg = DistanceBasedFormationCfg(distance_min=10.0, distance_max=80.0)
    randomizer = Randomizer(num_envs=100, num_agents=2, device=device, cfg=cfg)

    result = randomizer.generate_formation_and_target(scale_factor=0.5)
    ```

For legacy gimbal-constraint-based randomization, import from the legacy submodule:
    ```python
    from isaaclab_tasks.direct.iris_ma4.randomization.legacy import (
        LegacyRandomizer,
        InitialStatesRandomizer,
        TargetSampler,
    )
    ```
"""

# Primary exports (new distance-based system)
from .formation_cfg import DistanceBasedFormationCfg, FormationResult
from .distance_based_generator import DistanceBasedFormationGenerator
from .randomizer import Randomizer

__all__ = [
    # New distance-based system
    "Randomizer",
    "DistanceBasedFormationCfg",
    "DistanceBasedFormationGenerator",
    "FormationResult",
]
