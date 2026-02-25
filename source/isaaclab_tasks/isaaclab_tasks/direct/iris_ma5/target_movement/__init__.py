# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target movement module for multi-agent drone environments."""

from .target_movement_cfg import TargetMovementCfg
from .target_movement import TargetMovement

__all__ = ["TargetMovementCfg", "TargetMovement"]
