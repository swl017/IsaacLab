# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Field storage for delay system V3.

This module provides a simplified data storage system that:
- Stores raw (ground truth) and noisy data separately
- Maintains timestamps decoupled from data
- Supports noise injection at storage time
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple, Optional


class FieldStorage:
    """Storage for raw and noisy field data with timestamps.

    Key design principle: timestamps are stored SEPARATELY from data,
    not concatenated as in V2. This simplifies timestamp handling
    throughout the pipeline.
    """

    def __init__(self, num_envs: int, device: torch.device):
        """Initialize field storage.

        Args:
            num_envs: Number of parallel environments.
            device: Torch device for tensor operations.
        """
        self._num_envs = num_envs
        self._device = device

        # Raw (ground truth) data storage
        self._raw: Dict[str, torch.Tensor] = {}

        # Noisy data storage (with observation noise applied)
        self._noisy: Dict[str, torch.Tensor] = {}

        # Timestamps for each field (capture time)
        self._timestamps: Dict[str, torch.Tensor] = {}

        # Current simulation time
        self._t_current = torch.zeros(num_envs, device=device)

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    @property
    def t_current(self) -> torch.Tensor:
        """Current simulation time for each environment."""
        return self._t_current

    @property
    def field_names(self) -> list[str]:
        """List of stored field names."""
        return list(self._raw.keys())

    def set_time(self, t: torch.Tensor):
        """Set current simulation time.

        Args:
            t: Current time tensor of shape (num_envs,).
        """
        self._t_current = t.to(self._device)

    def store(
        self,
        field_name: str,
        data: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
        noisy_data: Optional[torch.Tensor] = None,
    ):
        """Store field data with optional noise injection.

        The noisy payload is only populated when the caller actually provides
        one (via ``noisy_data``) or requests Gaussian injection (``noise_std > 0``).
        Fields stored without a noisy payload return ``False`` from :meth:`has_noisy`,
        signalling to the delay pipeline that there is no separate noisy stream
        to cache — the observation path will read the raw cache instead.

        Args:
            field_name: Name of the field (e.g., "agent_0.body_position_w").
            data: Data tensor of shape (num_envs, ...). Stored as ground truth.
            timestamp: Optional capture timestamp. If None, uses current time.
            noise_std: Standard deviation for Gaussian noise. 0 = no noise.
                Ignored when noisy_data is provided.
            noisy_data: Pre-noised data tensor (same shape as data). When provided,
                stored directly as the noisy version instead of generating Gaussian
                noise. Used by the detector replicator to pass calibrated noise.
        """
        # Ensure data is on correct device
        data = data.to(self._device)

        # Store raw (ground truth)
        self._raw[field_name] = data.clone()

        # Store noisy version only when caller provides noise (explicit payload
        # or positive std). Clean-only fields leave _noisy unpopulated so the
        # pipeline can skip the noisy cache path entirely.
        # Ticket 034: noise_std may be a scalar (legacy) or a Tensor[N]
        # (per-env, from MultiAgentDelaySystemWrapper). Tensor std is
        # broadcast over data's trailing dims at noise application.
        if noisy_data is not None:
            self._noisy[field_name] = noisy_data.to(self._device).clone()
        elif isinstance(noise_std, torch.Tensor):
            if (noise_std > 0).any():
                # Reshape (N,) to (N, 1, 1, ...) for broadcast.
                broadcast_shape = (data.shape[0],) + (1,) * (data.dim() - 1)
                std_b = noise_std.to(self._device).view(broadcast_shape)
                noise = torch.randn_like(data) * std_b
                self._noisy[field_name] = data + noise
            else:
                self._noisy.pop(field_name, None)
        elif noise_std > 0:
            noise = torch.randn_like(data) * noise_std
            self._noisy[field_name] = data + noise
        else:
            self._noisy.pop(field_name, None)

        # Store timestamp
        if timestamp is None:
            self._timestamps[field_name] = self._t_current.clone()
        else:
            self._timestamps[field_name] = timestamp.to(self._device)

    def get_raw(self, field_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get raw (ground truth) data and timestamp.

        Args:
            field_name: Name of the field.

        Returns:
            Tuple of (data, timestamp).

        Raises:
            KeyError: If field not found.
        """
        if field_name not in self._raw:
            raise KeyError(f"Field '{field_name}' not found in storage")
        return self._raw[field_name], self._timestamps[field_name]

    def get_noisy(self, field_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get noisy data and timestamp.

        Args:
            field_name: Name of the field.

        Returns:
            Tuple of (data, timestamp).

        Raises:
            KeyError: If field not found.
        """
        if field_name not in self._noisy:
            raise KeyError(f"Field '{field_name}' not found in storage")
        return self._noisy[field_name], self._timestamps[field_name]

    def get(
        self, field_name: str, use_noise: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get data and timestamp with optional noise selection.

        Args:
            field_name: Name of the field.
            use_noise: If True, return noisy data. If False, return raw.

        Returns:
            Tuple of (data, timestamp).
        """
        if use_noise:
            return self.get_noisy(field_name)
        return self.get_raw(field_name)

    def has_field(self, field_name: str) -> bool:
        """Check if a field exists in storage.

        Args:
            field_name: Name of the field.

        Returns:
            True if field exists.
        """
        return field_name in self._raw

    def has_noisy(self, field_name: str) -> bool:
        """Check if a field has a separate noisy payload.

        Returns False for clean-only fields (stored without ``noise_std`` or
        ``noisy_data``). The delay pipeline uses this to decide whether the
        observation path needs its own cache slot.

        Args:
            field_name: Name of the field.

        Returns:
            True if a distinct noisy payload exists for this field.
        """
        return field_name in self._noisy

    def clear_field(self, field_name: str):
        """Remove a field from storage.

        Args:
            field_name: Name of the field to remove.
        """
        self._raw.pop(field_name, None)
        self._noisy.pop(field_name, None)
        self._timestamps.pop(field_name, None)

    def clear_all(self):
        """Clear all stored data."""
        self._raw.clear()
        self._noisy.clear()
        self._timestamps.clear()

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset storage for specified environments.

        This clears the data for specified environments by setting them to zeros.
        Note: For proper reset, the environment should re-store data on the
        first step after reset.

        Args:
            env_ids: Environment indices to reset. If None, clears all.
        """
        if env_ids is None:
            # Reset time to zero for all environments
            self._t_current.zero_()
            # Note: We don't clear the data - it will be overwritten on next store
        else:
            # Reset time for specific environments
            self._t_current[env_ids] = 0.0


class MultiFieldStorage:
    """Storage manager for multiple agents' field data.

    Provides a convenient interface for storing and retrieving
    data for multiple agents with field name prefixing.
    """

    def __init__(self, num_envs: int, device: torch.device):
        """Initialize multi-field storage.

        Args:
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self._storage = FieldStorage(num_envs, device)
        self._registered_agents: set[str] = set()
        self._registered_fields: set[str] = set()

    @property
    def storage(self) -> FieldStorage:
        """Underlying field storage."""
        return self._storage

    def register_agent(self, agent_id: str):
        """Register an agent for field storage.

        Args:
            agent_id: Unique agent identifier.
        """
        self._registered_agents.add(agent_id)

    def register_field(self, field_name: str):
        """Register a field type for storage.

        Args:
            field_name: Base field name (e.g., "body_position_w").
        """
        self._registered_fields.add(field_name)

    def _make_key(self, agent_id: str, field_name: str) -> str:
        """Create storage key from agent and field name.

        Args:
            agent_id: Agent identifier.
            field_name: Field name.

        Returns:
            Combined key: "{agent_id}.{field_name}".
        """
        return f"{agent_id}.{field_name}"

    def set_time(self, t: torch.Tensor):
        """Set current simulation time.

        Args:
            t: Current time tensor.
        """
        self._storage.set_time(t)

    def store_agent_field(
        self,
        agent_id: str,
        field_name: str,
        data: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ):
        """Store field data for an agent.

        Args:
            agent_id: Agent identifier.
            field_name: Field name.
            data: Data tensor.
            timestamp: Optional timestamp.
            noise_std: Noise standard deviation.
        """
        key = self._make_key(agent_id, field_name)
        self._storage.store(key, data, timestamp, noise_std)

    def get_agent_field(
        self,
        agent_id: str,
        field_name: str,
        use_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get field data for an agent.

        Args:
            agent_id: Agent identifier.
            field_name: Field name.
            use_noise: Whether to return noisy data.

        Returns:
            Tuple of (data, timestamp).
        """
        key = self._make_key(agent_id, field_name)
        return self._storage.get(key, use_noise)

    def has_agent_field(self, agent_id: str, field_name: str) -> bool:
        """Check if an agent field exists.

        Args:
            agent_id: Agent identifier.
            field_name: Field name.

        Returns:
            True if field exists.
        """
        key = self._make_key(agent_id, field_name)
        return self._storage.has_field(key)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset storage for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._storage.reset(env_ids)
