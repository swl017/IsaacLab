"""Configuration for {{ModuleName}}.

Rename this file to {{MODULE_NAME}}_cfg.py after copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from isaaclab.utils import configclass


@configclass
class {{ModuleNameCfg}}:
    """{{MODULE_DESCRIPTION}}.

    Extended description of the module's purpose and typical usage.
    """

    # ===== Core Parameters =====

    enabled: bool = True
    """Whether this module is active."""

    # param_name: float = 1.0
    # """Description. Units: [m/s]. Typical range: [0.5, 2.0]."""

    # ===== Tuning Parameters =====

    # param_tuple: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # """Description [x, y, z] [units]."""

    # ===== Sub-Components [OPTIONAL] =====

    # sub_component: SubComponentCfg = SubComponentCfg()
    # """Sub-component configuration."""
