# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for zoom controller."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ZoomControllerCfg:
    """Configuration for zoom controller with first-order dynamics.

    The zoom controller simulates mechanical/optical zoom with transient dynamics.
    """

    tau_zoom: float = 0.1
    """Zoom time constant [s]. Default 100ms for mechanical zoom mechanism."""

    zoom_min: float = 1.0
    """Minimum zoom level (1x = no zoom)."""

    zoom_max: float = 10.0
    """Maximum zoom level (10x)."""

    max_zoom_rate: float = 2.0
    """Maximum zoom change rate [1/s]. How fast zoom level can change."""
