# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for gimbal stabilizer."""

from __future__ import annotations

import math

from isaaclab.utils import configclass


@configclass
class GimbalStabilizerCfg:
    """Configuration for the GimbalStabilizer.

    Defines gimbal angle limits and rate constraints. This config is the
    single source of truth for gimbal limits and is shared with other
    components like TargetSampler.

    All angles are in radians.
    """

    # ==========================================================================
    # Angle Limits (radians)
    # ==========================================================================

    yaw_limits: tuple[float, float] = (math.radians(-200.0), math.radians(200.0))
    """Yaw angle limits [min, max] in radians. CCW positive when viewed from above."""

    pitch_limits: tuple[float, float] = (math.radians(-45.0), math.radians(10.0))
    """Pitch angle limits [min, max] in radians. Down is negative."""

    roll_limits: tuple[float, float] = (math.radians(-45.0), math.radians(45.0))
    """Roll angle limits [min, max] in radians."""

    # ==========================================================================
    # Rate Limits
    # ==========================================================================

    max_rate: float = math.radians(360.0)
    """Maximum angular rate for all gimbal axes in rad/s."""
