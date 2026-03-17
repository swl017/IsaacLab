# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Triangulation module for iris_ma6.

This module provides triangulation and uncertainty estimation for multi-camera
multi-target observation systems. Key improvements over iris_ma5:

- Validity-based returns (NaN + is_valid mask) instead of sentinel values
- Per-camera-target valid mask support [N, C, T]
- Condition number checking with configurable threshold
- Observability checks (minimum cameras, well-conditioned geometry)
"""

from .triangulation_cfg import TriangulationCfg
from .triangulation import (
    triangulate_targets,
    compute_triangulation_covariance,
    compute_full_triangulation,
    get_ray_directions_from_bbox,
    TriangulationResult,
)
