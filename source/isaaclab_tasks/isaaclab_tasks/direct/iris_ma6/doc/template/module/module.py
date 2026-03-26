"""{{ModuleName}} — {{MODULE_DESCRIPTION}}.

Rename this file to {{MODULE_NAME}}.py after copying.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .{{MODULE_NAME}}_cfg import {{ModuleNameCfg}}


class {{ModuleName}}:
    """{{MODULE_DESCRIPTION}}.

    Extended description of the module's behavior, inputs, and outputs.
    """

    def __init__(
        self,
        cfg: {{ModuleNameCfg}},
        num_envs: int,
        device: torch.device,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Internal state
        # self._last_update_time = torch.full((num_envs,), -1.0, device=device)

    # --- WRITE methods (call once per step) ---

    # def update(self, data: torch.Tensor, t_current: torch.Tensor) -> None:
    #     """Advance internal state. Call once per step from _post_physics_step().
    #
    #     Idempotent within a single sim step via _last_update_time guard.
    #     """
    #     already_updated = (t_current - self._last_update_time).abs().max().item() < 1e-6
    #     if already_updated:
    #         return
    #     self._last_update_time = t_current.clone()
    #     # ... state mutation logic ...

    # --- READ methods (safe to call multiple times) ---

    # def compute(self, inputs: torch.Tensor) -> torch.Tensor:
    #     """Compute output from current state. Safe to call multiple times per step."""
    #     pass

    # --- CONFIG methods ---

    # def set_param(self, value: float) -> None:
    #     """Update a runtime parameter (e.g., from curriculum)."""
    #     pass

    # --- Reset ---

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset internal state for specified environments."""
        # self._last_update_time[env_ids] = -1.0
        pass
