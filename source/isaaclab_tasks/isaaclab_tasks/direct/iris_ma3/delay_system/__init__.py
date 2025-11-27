"""
Delay System for Multi-Agent RL Environments

This package provides realistic time-shifting phenomena simulation for sim-to-real transfer:
- Stochastic sample-and-hold with dropout
- Multi-stage delay pipelines
- Per-field configuration
- Multi-agent perspective handling
"""

from .delay_system import DelaySystem
from .delay_system_cfg import DelaySystemCfg
from .agent_states import AgentStates, AgentStatesData, MultiAgentStates
from .timestamp_manager import FieldTimestamp, AgentStatesTimestamps, TimestampManager
from .stochastic_sampler import (
    StochasticSampler,
    SamplerConfig,
    DistributionConfig,
    create_detector_sampler,
    create_communication_sampler,
    create_sensor_sampler,
)
from .specialized_samplers import (
    FirstOrderLagSampler,
    QuaternionFirstOrderLagSampler,
    PassthroughSampler,
)

# Backward compatibility alias for FirstOrderLag (used in iris_ma_env3.py for _robot_dynamics)
FirstOrderLag = FirstOrderLagSampler
from .sampler_chain import SamplerChain, ComposableSampler
from .field_configs import (
    FIELD_GROUPS,
    FIELD_TO_GROUP,
    FIELD_DIMENSIONS,
    get_field_group,
    get_group_fields,
    get_all_groups,
)
from .multi_agent_wrapper import MultiAgentDelaySystem

__all__ = [
    # Main system
    "DelaySystem",
    "DelaySystemCfg",
    # Agent states
    "AgentStates",
    "AgentStatesData",
    "MultiAgentStates",
    # Timestamp management
    "FieldTimestamp",
    "AgentStatesTimestamps",
    "TimestampManager",
    # Samplers
    "StochasticSampler",
    "FirstOrderLagSampler",
    "FirstOrderLag",  # Backward compatibility alias
    "QuaternionFirstOrderLagSampler",
    "PassthroughSampler",
    # Sampler configuration
    "SamplerConfig",
    "DistributionConfig",
    # Factory functions
    "create_detector_sampler",
    "create_communication_sampler",
    "create_sensor_sampler",
    # Sampler chains
    "SamplerChain",
    "ComposableSampler",
    # Field configuration
    "FIELD_GROUPS",
    "FIELD_TO_GROUP",
    "FIELD_DIMENSIONS",
    "get_field_group",
    "get_group_fields",
    "get_all_groups",
    # Multi-agent wrapper
    "MultiAgentDelaySystem",
]
