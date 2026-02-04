# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Legacy randomization module.

This module contains the original gimbal-constraint-focused randomization system.
It has been superseded by the distance-based formation generator.

Kept for backward compatibility and reference.
"""

from .randomizer import Randomizer as LegacyRandomizer
from .initial_states import InitialStatesRandomizer, InitialStatesRandomizerCfg
from .target_sampling import TargetSampler, TargetSamplerCfg, create_target_sampler

__all__ = [
    "LegacyRandomizer",
    "InitialStatesRandomizer",
    "InitialStatesRandomizerCfg",
    "TargetSampler",
    "TargetSamplerCfg",
    "create_target_sampler",
]
