# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gimbal controller factory — dispatches based on cfg.mode.

Two implementations:
- "analytical": Direct atan2 IK (gimbal_controller_analytical.py)
- "jacobian":   J^{-1} velocity tracking (gimbal_controller_jacobian.py)

Both share the same interface (compute_control, reset, etc.) and
GimbalControllerCfg. YAW_JOINT_OFFSET is re-exported for convenience.
"""

from __future__ import annotations

import math

from .gimbal_controller_cfg import GimbalControllerCfg

# Re-export the constant so existing `from .gimbal_controller import YAW_JOINT_OFFSET` keeps working
YAW_JOINT_OFFSET = -math.pi / 2


def GimbalController(cfg: GimbalControllerCfg, **kwargs):
    """Factory that returns the appropriate gimbal controller based on cfg.mode."""
    if cfg.mode == "analytical":
        from .gimbal_controller_analytical import GimbalController as _Cls
    elif cfg.mode == "jacobian":
        from .gimbal_controller_jacobian import GimbalController as _Cls
    else:
        raise ValueError(f"Unknown gimbal mode '{cfg.mode}'. Use 'analytical' or 'jacobian'.")
    return _Cls(cfg=cfg, **kwargs)
