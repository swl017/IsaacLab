# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions to apply experiment overrides to configs."""

from __future__ import annotations

from typing import Any, Dict


def apply_env_overrides(cfg: object, overrides: Dict[str, Any]) -> object:
    """Apply dotted-path overrides to an IrisMAEnvCfg instance.

    Supports nested access via dots, e.g.:
        'curriculum.noise_start_step' -> cfg.curriculum.noise_start_step

    Args:
        cfg: Base environment config to modify (modified in-place).
        overrides: Dictionary of {dotted_path: value} overrides.

    Returns:
        The modified config.
    """
    for path, value in overrides.items():
        parts = path.split(".")
        obj = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise AttributeError(
                    f"Config object {type(obj).__name__} has no attribute '{part}' "
                    f"(from path '{path}')"
                )
            obj = getattr(obj, part)
        final_attr = parts[-1]
        if not hasattr(obj, final_attr):
            raise AttributeError(
                f"Config object {type(obj).__name__} has no attribute '{final_attr}' "
                f"(from path '{path}')"
            )
        setattr(obj, final_attr, value)
    return cfg


def apply_agent_overrides(agent_cfg: dict, overrides: Dict[str, Any]) -> dict:
    """Apply dotted-path overrides to agent config dict.

    Args:
        agent_cfg: Base agent config dictionary.
        overrides: Dictionary of {dotted_path: value} overrides.

    Returns:
        The modified agent config dict.
    """
    for path, value in overrides.items():
        parts = path.split(".")
        obj = agent_cfg
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        obj[parts[-1]] = value
    return agent_cfg
