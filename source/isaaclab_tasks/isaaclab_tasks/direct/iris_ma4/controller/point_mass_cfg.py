# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for point mass controller."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class PointMassCfg:
    """Configuration for the PointMass velocity tracking controller.

    This controller uses PID control for velocity tracking and PD control
    for attitude stabilization. All gains can be tuned via this config.
    """

    # ==========================================================================
    # Position/Velocity Control Gains
    # ==========================================================================

    kp_x: float = 3.0
    """Proportional gain for X velocity tracking."""

    ki_x: float = 0.5
    """Integral gain for velocity tracking (anti-windup enabled)."""

    kd_x: float = 0.1
    """Derivative gain for velocity tracking (acceleration feedback)."""

    kp_y: float = 3.0
    """Proportional gain for Y velocity tracking."""

    kd_y: float = 0.1
    """Derivative gain for Y velocity tracking."""

    kp_z: float = 2.0
    """Proportional gain for Z velocity tracking."""

    # ==========================================================================
    # Attitude Control Gains
    # ==========================================================================

    kp_att: float = 2.0
    """Proportional gain for roll/pitch attitude stabilization."""

    kd_att: float = 2.2
    """Derivative gain for roll/pitch (angular velocity damping)."""

    kp_yaw: float = 1.2
    """Proportional gain for yaw rate tracking."""

    kd_yaw: float = 0.5
    """Derivative gain for yaw (angular velocity damping)."""

    # ==========================================================================
    # Limits
    # ==========================================================================

    max_moment: tuple[float, float, float] = (5.0, 5.0, 2.0)
    """Maximum control moment (roll, pitch, yaw) in Nm."""

    integral_max: float = 5.0
    """Maximum integral accumulation for anti-windup (N*s)."""

    thrust_scale_min: float = 1.0
    """Minimum thrust scaling factor (multiple of weight)."""

    thrust_scale_max: float = 3.0
    """Maximum thrust scaling factor (multiple of weight)."""
