"""
Timestamp management for delayed agent states.

This module provides classes for tracking timing metadata throughout
the delay system, enabling staleness calculation and latency analysis.
"""

from __future__ import annotations
import torch
from typing import Dict, Optional
from dataclasses import dataclass


class FieldTimestamp:
    """
    Timestamp metadata for a field or field group.

    Tracks when data was captured, sampled, and became available,
    enabling staleness and latency calculations.
    """

    def __init__(self, num_envs: int, device: torch.device):
        """
        Args:
            num_envs: Number of parallel environments
            device: Device to allocate tensors on
        """
        self.num_envs = num_envs
        self.device = device

        # Timestamp tensors (all in seconds)
        self.t_captured = torch.zeros(num_envs, device=device)
        self.t_sampled = torch.zeros(num_envs, device=device)
        self.t_available = torch.zeros(num_envs, device=device)
        self.t_current = torch.zeros(num_envs, device=device)

    def update(
        self,
        t_captured: torch.Tensor,
        t_sampled: torch.Tensor,
        t_available: torch.Tensor,
        t_current: torch.Tensor,
    ):
        """
        Update timestamp values.

        Args:
            t_captured: Time when data was captured in simulation
            t_sampled: Time when data was last sampled (period-based)
            t_available: Time when data became available (after latency)
            t_current: Current simulation time
        """
        self.t_captured = t_captured.clone()
        self.t_sampled = t_sampled.clone()
        self.t_available = t_available.clone()
        self.t_current = t_current.clone()

    def update_from_info(self, info: dict, t_current: torch.Tensor):
        """
        Update timestamp from sampler chain info dictionary.

        Args:
            info: Info dict from SamplerChain.update()
            t_current: Current simulation time
        """
        self.t_captured = info['t_captured'].clone()
        self.t_available = info['t_available'].clone()
        self.t_current = t_current.clone()

        # t_sampled is typically the time before the final latency stage
        # For simplicity, approximate as t_available - total_latency
        self.t_sampled = self.t_available.clone()

    @property
    def staleness(self) -> torch.Tensor:
        """
        Time since data was captured (seconds).

        This represents how "stale" the data is - the total time elapsed
        from when the data was originally captured to now.

        Returns:
            Staleness tensor (num_envs,)
        """
        return self.t_current - self.t_captured

    @property
    def latency(self) -> torch.Tensor:
        """
        Total latency from capture to availability (seconds).

        This represents the processing delay - how long it took from
        capturing the data to having it available for use.

        Returns:
            Latency tensor (num_envs,)
        """
        return self.t_available - self.t_captured

    @property
    def age(self) -> torch.Tensor:
        """
        Time since data became available (seconds).

        This represents how long the data has been sitting available
        but potentially unused.

        Returns:
            Age tensor (num_envs,)
        """
        return self.t_current - self.t_available

    @property
    def sample_period(self) -> torch.Tensor:
        """
        Time since last sample (seconds).

        Returns:
            Sample period tensor (num_envs,)
        """
        return self.t_current - self.t_sampled

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """
        Reset timestamps for specified environments.

        Args:
            env_ids: Indices of environments to reset. None means reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.t_captured[env_ids] = 0.0
        self.t_sampled[env_ids] = 0.0
        self.t_available[env_ids] = 0.0
        self.t_current[env_ids] = 0.0

    def clone(self) -> FieldTimestamp:
        """
        Create a deep copy of this timestamp.

        Returns:
            New FieldTimestamp with cloned values
        """
        new_ts = FieldTimestamp(self.num_envs, self.device)
        new_ts.t_captured = self.t_captured.clone()
        new_ts.t_sampled = self.t_sampled.clone()
        new_ts.t_available = self.t_available.clone()
        new_ts.t_current = self.t_current.clone()
        return new_ts


class AgentStatesTimestamps:
    """
    Timestamp tracking for all AgentStates field groups.

    Organizes timestamps by field group (motion, detection, etc.)
    to provide group-level timing metadata.
    """

    def __init__(self, field_groups: list[str], num_envs: int, device: torch.device):
        """
        Args:
            field_groups: List of field group names (e.g., ['motion', 'detection'])
            num_envs: Number of parallel environments
            device: Device to allocate tensors on
        """
        self.field_groups = field_groups
        self.num_envs = num_envs
        self.device = device

        # Create timestamp tracker for each group
        self.timestamps: Dict[str, FieldTimestamp] = {}
        for group_name in field_groups:
            self.timestamps[group_name] = FieldTimestamp(num_envs, device)

    def update_group(self, group_name: str, info: dict, t_current: torch.Tensor):
        """
        Update timestamps for a field group.

        Args:
            group_name: Name of the field group
            info: Info dict from SamplerChain.update()
            t_current: Current simulation time
        """
        if group_name not in self.timestamps:
            raise ValueError(f"Unknown field group: {group_name}")

        self.timestamps[group_name].update_from_info(info, t_current)

    def get_group_staleness(self, group_name: str) -> torch.Tensor:
        """
        Get staleness for a field group.

        Args:
            group_name: Name of the field group

        Returns:
            Staleness tensor (num_envs,)
        """
        return self.timestamps[group_name].staleness

    def get_group_latency(self, group_name: str) -> torch.Tensor:
        """
        Get latency for a field group.

        Args:
            group_name: Name of the field group

        Returns:
            Latency tensor (num_envs,)
        """
        return self.timestamps[group_name].latency

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """
        Reset all timestamps for specified environments.

        Args:
            env_ids: Indices of environments to reset. None means reset all.
        """
        for timestamp in self.timestamps.values():
            timestamp.reset(env_ids)

    def clone(self) -> AgentStatesTimestamps:
        """
        Create a deep copy of all timestamps.

        Returns:
            New AgentStatesTimestamps with cloned values
        """
        new_timestamps = AgentStatesTimestamps(self.field_groups, self.num_envs, self.device)
        for group_name, timestamp in self.timestamps.items():
            new_timestamps.timestamps[group_name] = timestamp.clone()
        return new_timestamps


class TimestampManager:
    """
    Central manager for timestamps across all agents and field groups.

    Provides convenience methods for updating and querying timestamps
    in a multi-agent environment.
    """

    def __init__(
        self,
        agent_ids: list[str],
        field_groups: list[str],
        num_envs: int,
        device: torch.device,
    ):
        """
        Args:
            agent_ids: List of agent identifiers
            field_groups: List of field group names
            num_envs: Number of parallel environments
            device: Device to allocate tensors on
        """
        self.agent_ids = agent_ids
        self.field_groups = field_groups
        self.num_envs = num_envs
        self.device = device

        # Create timestamp tracker for each agent
        self.agent_timestamps: Dict[str, AgentStatesTimestamps] = {}
        for agent_id in agent_ids:
            self.agent_timestamps[agent_id] = AgentStatesTimestamps(field_groups, num_envs, device)

    def update_agent_group(
        self,
        agent_id: str,
        group_name: str,
        info: dict,
        t_current: torch.Tensor,
    ):
        """
        Update timestamps for a specific agent's field group.

        Args:
            agent_id: Agent identifier
            group_name: Field group name
            info: Info dict from SamplerChain.update()
            t_current: Current simulation time
        """
        if agent_id not in self.agent_timestamps:
            raise ValueError(f"Unknown agent ID: {agent_id}")

        self.agent_timestamps[agent_id].update_group(group_name, info, t_current)

    def get_agent_timestamps(self, agent_id: str) -> AgentStatesTimestamps:
        """
        Get all timestamps for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentStatesTimestamps for the agent
        """
        if agent_id not in self.agent_timestamps:
            raise ValueError(f"Unknown agent ID: {agent_id}")

        return self.agent_timestamps[agent_id]

    def reset(self, agent_id: Optional[str] = None, env_ids: Optional[torch.Tensor] = None):
        """
        Reset timestamps.

        Args:
            agent_id: Agent to reset. None means reset all agents.
            env_ids: Environments to reset. None means reset all environments.
        """
        if agent_id is not None:
            self.agent_timestamps[agent_id].reset(env_ids)
        else:
            for timestamps in self.agent_timestamps.values():
                timestamps.reset(env_ids)
