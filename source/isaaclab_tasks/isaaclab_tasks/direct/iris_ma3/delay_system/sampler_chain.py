"""
Sampler chain composition for complex delay pipelines.

This module allows chaining multiple samplers together to model
multi-stage processing with compounding delays.

Example:
    Motion → Detector → Communication
    (filter) (20Hz, 0.3s) (30Hz, 0.1s)
"""

from __future__ import annotations
import torch
from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .stochastic_sampler import StochasticSampler
    from .specialized_samplers import FirstOrderLagSampler, QuaternionFirstOrderLagSampler, PassthroughSampler

# Type alias for any sampler type
SamplerType = Union['StochasticSampler', 'FirstOrderLagSampler', 'QuaternionFirstOrderLagSampler', 'PassthroughSampler']


class SamplerChain:
    """
    Chain of samplers for a field, with timestamp tracking through stages.

    Each sampler in the chain processes data sequentially, with latencies
    and delays compounding across stages.
    """

    def __init__(
        self,
        field_name: str,
        samplers: List[SamplerType],
        num_envs: int,
        device: torch.device,
    ):
        """
        Args:
            field_name: Name of the field this chain processes
            samplers: List of samplers to chain (applied in order)
            num_envs: Number of environments
            device: Device to allocate tensors on
        """
        self.field_name = field_name
        self.samplers = samplers
        self.num_envs = num_envs
        self.device = device

        # Timestamp tracking through chain
        self.t_captured = torch.zeros(num_envs, device=device)
        self.t_available = torch.zeros(num_envs, device=device)
        self.stage_timestamps = []

        # Metadata aggregation
        self.total_accumulated_latency = torch.zeros(num_envs, device=device)

    def update(
        self,
        data: torch.Tensor,
        t_current: torch.Tensor,
        force_update: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Pass data through sampler chain.

        Args:
            data: Input data tensor
            t_current: Current simulation time (num_envs,)
            force_update: Boolean mask to force update for specific envs

        Returns:
            Tuple of (output_data, info_dict):
                - output_data: Data after passing through all samplers
                - info_dict: Dictionary with aggregated metadata:
                    - 't_captured': When data was captured
                    - 't_available': When data became available (after all stages)
                    - 't_current': Current simulation time
                    - 'total_latency': Accumulated latency across all stages
                    - 'stage_info': List of info dicts from each sampler
        """
        current_data = data
        self.t_captured = t_current.clone()
        self.stage_timestamps = [t_current.clone()]
        stage_info_list = []

        # Pass through each sampler stage
        for sampler in self.samplers:
            current_data, stage_info = sampler.update(current_data, t_current, force_update)
            stage_info_list.append(stage_info)

            # Accumulate timestamp based on stage latency
            stage_latency = stage_info.get('accumulated_latency', torch.zeros(self.num_envs, device=self.device))
            current_timestamp = self.stage_timestamps[-1] + stage_latency
            self.stage_timestamps.append(current_timestamp)

        # Compute total accumulated latency
        self.total_accumulated_latency = self.stage_timestamps[-1] - self.t_captured
        self.t_available = self.stage_timestamps[-1]

        # Build aggregated info
        info = {
            't_captured': self.t_captured.clone(),
            't_available': self.t_available.clone(),
            't_current': t_current.clone(),
            'total_latency': self.total_accumulated_latency.clone(),
            'stage_info': stage_info_list,
        }

        return current_data, info

    def reset(self, env_ids: Optional[torch.Tensor] = None, initial_data: Optional[torch.Tensor] = None):
        """
        Reset all samplers in the chain for specified environments.

        Args:
            env_ids: Indices of environments to reset. None means reset all.
            initial_data: Initial data for reset environments.
        """
        for sampler in self.samplers:
            sampler.reset(env_ids, initial_data)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.t_captured[env_ids] = 0.0
        self.t_available[env_ids] = 0.0
        self.total_accumulated_latency[env_ids] = 0.0


class ComposableSampler:
    """
    Compose multiple sampler chains for different perspectives.

    This allows having both ego and other-agent perspectives with
    different final communication stages.

    Example:
        base_chain = [motion_filter, detector]
        ego_chain = ComposableSampler(base_chain + [local_comm])
        other_chain = ComposableSampler(base_chain + [inter_agent_comm])
    """

    def __init__(self, samplers: List[SamplerType]):
        """
        Args:
            samplers: List of samplers to compose (applied in order)
        """
        self.samplers = samplers

    def update(
        self,
        data: torch.Tensor,
        t_current: torch.Tensor,
        force_update: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Pass data through all samplers.

        Args:
            data: Input data
            t_current: Current simulation time
            force_update: Forced update mask

        Returns:
            Tuple of (final_data, aggregated_info)
        """
        current_data = data
        all_stage_info = []

        for sampler in self.samplers:
            current_data, stage_info = sampler.update(current_data, t_current, force_update)
            all_stage_info.append(stage_info)

        # Aggregate info from all stages
        info = {
            'all_stages': all_stage_info,
        }

        return current_data, info

    def reset(self, env_ids: Optional[torch.Tensor] = None, initial_data: Optional[torch.Tensor] = None):
        """Reset all samplers."""
        for sampler in self.samplers:
            sampler.reset(env_ids, initial_data)
