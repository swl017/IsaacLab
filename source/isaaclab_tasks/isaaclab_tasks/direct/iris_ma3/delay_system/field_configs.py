"""
Field grouping and sampler configuration for AgentStates.

This module defines which fields belong to which groups and provides
factory functions for creating appropriately configured samplers.
"""

from __future__ import annotations
import torch
from typing import Dict, List
from .stochastic_sampler import StochasticSampler, SamplerConfig, DistributionConfig
from .specialized_samplers import FirstOrderLagSampler, QuaternionFirstOrderLagSampler, PassthroughSampler
from .sampler_chain import SamplerChain


# ==================== Field Group Definitions ====================

# Fields grouped by timing characteristics
FIELD_GROUPS = {
    # Motion states: First-order lag (100 Hz), same filter
    'motion': [
        'body_position_w',
        'body_linear_velocity_w',
        'body_linear_velocity_b',
        'body_angular_velocity_w',
        'body_angular_velocity_b',
        'body_combined_angular_velocity_w', # robot body + gimbal orientation
        'body_combined_angular_velocity_b',
        'body_linear_acceleration_w',
        'body_linear_acceleration_b',
        'body_angular_acceleration_b',
    ],

    # Orientation: Quaternion SLERP filtering (100 Hz)
    'orientation': [
        'body_orientation_w',
        'camera_orientation_w',
    ],

    # Joint states: First-order lag (100 Hz)
    'joints': [
        'joint_positions_b',
        'joint_velocities_b',
        'joint_accelerations_b',
    ],

    # Camera zoom: First-order lag (optical zoom = mechanical lens movement)
    # Separate group because dimension differs from joints [N] vs [N, J]
    'zoom': [
        'camera_zoom_level',
    ],

    # Detection states: Detector sampling (20 Hz, 300ms latency, dropout)
    # Split by dimension compatibility: bboxes [N,T,4] vs rays [N,T,3]
    'detection_bbox': [
        'bboxes_2d',  # [N, T, 4]
    ],

    'detection_rays': [
        'camera_ray_directions_w',  # [N, T, 3]
        'camera_ray_origins_w',     # [N, T, 3]
    ],

    # Camera intrinsics: Static, no delay
    # Note: camera_zoom_level moved to 'zoom' group (mechanical actuator)
    'camera_intrinsics': [
        'camera_width',
        'camera_height',
        'camera_default_focal_length',  # Base focal length (unzoomed)
        'camera_horizontal_aperture',
        'camera_vertical_aperture',
        'camera_offset_position_b',
        'camera_offset_rotation_b',
        'camera_intrinsics',
        'camera_position_w',
    ],

    # Timestamps (special handling, not sampled)
    'timestamps': [
        'timestamp_sim_walltime',
        'timestamp_motion',
        'timestamp_detection',
    ],
}

# Reverse mapping: field_name -> group_name
FIELD_TO_GROUP: Dict[str, str] = {}
for group_name, fields in FIELD_GROUPS.items():
    for field in fields:
        FIELD_TO_GROUP[field] = group_name


# ==================== Field Dimension Mapping ====================

# Dimensions for each field (used to initialize samplers)
FIELD_DIMENSIONS = {
    # Motion states (3D vectors)
    'body_position_w': 3,
    'body_linear_velocity_w': 3,
    'body_linear_velocity_b': 3,
    'body_angular_velocity_w': 3,
    'body_angular_velocity_b': 3,
    'body_combined_angular_velocity_w': 3,
    'body_combined_angular_velocity_b': 3,
    'body_linear_acceleration_w': 3,
    'body_linear_acceleration_b': 3,
    'body_angular_acceleration_b': 3,

    # Orientations (quaternions)
    'body_orientation_w': 4,
    'camera_orientation_w': 4,

    # Camera intrinsics (scalars)
    'camera_width': 1,
    'camera_height': 1,
    'camera_default_focal_length': 1,
    'camera_horizontal_aperture': 1,
    'camera_vertical_aperture': 1,
    'camera_zoom_level': 1,  # In 'zoom' group (mechanical zoom, first-order lag)

    # Camera intrinsics (vectors/matrices)
    'camera_offset_position_b': 3,
    'camera_offset_rotation_b': 4,
    'camera_position_w': 3,

    # Timestamps (scalars)
    'timestamp_sim_walltime': 1,
    'timestamp_motion': 1,
    'timestamp_detection': 1,
}

# Note: Some fields have dynamic dimensions (e.g., joints, bboxes, camera_intrinsics matrix)
# These are handled specially during sampler creation


# ==================== Sampler Factory Functions ====================

def create_motion_sampler(
    field_name: str,
    num_envs: int,
    time_constant: float,
    dt: float,
    device: torch.device,
) -> FirstOrderLagSampler:
    """
    Create a first-order lag sampler for motion fields.

    Args:
        field_name: Name of the motion field
        num_envs: Number of environments
        time_constant: Filter time constant (seconds)
        dt: Simulation timestep (seconds)
        device: Device

    Returns:
        FirstOrderLagSampler configured for the field
    """
    state_dim = FIELD_DIMENSIONS.get(field_name, 3)  # Default to 3D
    return FirstOrderLagSampler(
        num_envs=num_envs,
        state_dim=state_dim,
        time_constant=time_constant,
        dt=dt,
        device=device,
    )


def create_orientation_sampler(
    num_envs: int,
    time_constant: float,
    dt: float,
    device: torch.device,
) -> QuaternionFirstOrderLagSampler:
    """
    Create a quaternion SLERP sampler for orientation fields.

    Args:
        num_envs: Number of environments
        time_constant: Filter time constant (seconds)
        dt: Simulation timestep (seconds)
        device: Device

    Returns:
        QuaternionFirstOrderLagSampler
    """
    return QuaternionFirstOrderLagSampler(
        num_envs=num_envs,
        time_constant=time_constant,
        dt=dt,
        device=device,
    )


def create_joint_sampler(
    num_joints: int,
    num_envs: int,
    time_constant: float,
    dt: float,
    device: torch.device,
) -> FirstOrderLagSampler:
    """
    Create a first-order lag sampler for joint fields.

    Args:
        num_joints: Number of joints
        num_envs: Number of environments
        time_constant: Filter time constant (seconds)
        dt: Simulation timestep (seconds)
        device: Device

    Returns:
        FirstOrderLagSampler configured for joints
    """
    return FirstOrderLagSampler(
        num_envs=num_envs,
        state_dim=num_joints,
        time_constant=time_constant,
        dt=dt,
        device=device,
    )


def create_detection_sampler(
    fps_mean: float,
    fps_std: float,
    latency_mean: float,
    latency_std: float,
    dropout_rate: float,
    num_envs: int,
    dt: float,
    device: torch.device,
) -> StochasticSampler:
    """
    Create a stochastic sampler for detection fields.

    Args:
        fps_mean: Mean detection rate (Hz)
        fps_std: Std deviation of detection rate (Hz)
        latency_mean: Mean processing latency (seconds)
        latency_std: Std deviation of latency (seconds)
        dropout_rate: Frame dropout rate (0-1)
        num_envs: Number of environments
        dt: Simulation timestep
        device: Device

    Returns:
        StochasticSampler configured for detector
    """
    config = SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="normal",
            mean=1.0 / fps_mean,
            std=fps_std / (fps_mean ** 2),
        ),
        latency_dist=DistributionConfig(
            distribution_type="normal",
            mean=latency_mean,
            std=latency_std,
        ),
        dropout_dist=DistributionConfig(
            distribution_type="constant",
            value=dropout_rate,
        ),
    )

    return StochasticSampler(config, num_envs, device, dt)


def create_communication_sampler(
    rate_min: float,
    rate_max: float,
    latency_mean: float,
    latency_std: float,
    dropout_rate: float,
    num_envs: int,
    dt: float,
    device: torch.device,
) -> StochasticSampler:
    """
    Create a stochastic sampler for communication.

    Args:
        rate_min: Minimum communication rate (Hz)
        rate_max: Maximum communication rate (Hz)
        latency_mean: Mean network latency (seconds)
        latency_std: Std deviation of latency (seconds)
        dropout_rate: Packet dropout rate (0-1)
        num_envs: Number of environments
        dt: Simulation timestep
        device: Device

    Returns:
        StochasticSampler configured for communication
    """
    config = SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="uniform",
            min_value=1.0 / rate_max,
            max_value=1.0 / rate_min,
        ),
        latency_dist=DistributionConfig(
            distribution_type="normal",
            mean=latency_mean,
            std=latency_std,
        ),
        dropout_dist=DistributionConfig(
            distribution_type="constant",
            value=dropout_rate,
        ),
    )

    return StochasticSampler(config, num_envs, device, dt)


def create_passthrough_sampler(
    num_envs: int,
    device: torch.device,
) -> PassthroughSampler:
    """
    Create a passthrough sampler (no delay).

    Args:
        num_envs: Number of environments
        device: Device

    Returns:
        PassthroughSampler
    """
    return PassthroughSampler(num_envs, device)


# ==================== Helper Functions ====================

def get_field_group(field_name: str) -> str:
    """
    Get the group name for a field.

    Args:
        field_name: Name of the field

    Returns:
        Group name

    Raises:
        ValueError: If field is not in any group
    """
    if field_name not in FIELD_TO_GROUP:
        raise ValueError(f"Unknown field: {field_name}. Not in any defined group.")
    return FIELD_TO_GROUP[field_name]


def get_group_fields(group_name: str) -> List[str]:
    """
    Get all fields in a group.

    Args:
        group_name: Name of the group

    Returns:
        List of field names

    Raises:
        ValueError: If group doesn't exist
    """
    if group_name not in FIELD_GROUPS:
        raise ValueError(f"Unknown group: {group_name}")
    return FIELD_GROUPS[group_name]


def get_all_groups() -> List[str]:
    """
    Get list of all field group names.

    Returns:
        List of group names
    """
    return list(FIELD_GROUPS.keys())
