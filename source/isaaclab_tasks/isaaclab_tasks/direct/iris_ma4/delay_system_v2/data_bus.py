# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central data bus for storing clean and noisy sensor data."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .delay_cfg import NoiseCfg


class DataBus:
    """Central storage for clean and noisy sensor data.

    The data bus serves as a central repository for all sensor data in the delay system.
    It maintains two parallel buffers:
    - Clean buffer: Raw sensor data without noise (used for reward computation)
    - Noisy buffer: Sensor data with Gaussian noise injected (used for observations)

    This separation allows the RL agent to receive realistic noisy observations while
    rewards are computed from ground truth values.

    Example usage:
        data_bus = DataBus(num_envs=4096, device="cuda:0")

        # Store sensor data
        data_bus.store("imu_accel", raw_accel, noise_std=0.1)
        data_bus.store("gps_position", raw_gps, noise_std=0.5)

        # Retrieve for reward computation (clean)
        clean_pos = data_bus.get_clean("gps_position")

        # Retrieve for observation (noisy)
        noisy_accel = data_bus.get_noisy("imu_accel")

        # Advance time
        data_bus.step(dt=0.01)
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        """Initialize the data bus.

        Args:
            num_envs: Number of parallel environments.
            device: Device to allocate tensors on (e.g., "cuda:0" or "cpu").
        """
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device

        # Clean data buffer (for rewards) - no noise
        self.clean: dict[str, torch.Tensor] = {}

        # Noisy data buffer (for observations) - with noise injected
        self.noisy: dict[str, torch.Tensor] = {}

        # Current simulation time per environment
        self.t_current = torch.zeros(num_envs, device=self.device)

    def store(self, field_name: str, data: torch.Tensor, noise_std: float = 0.0):
        """Store sensor data to both clean and noisy buffers.

        The clean buffer receives the raw data unchanged.
        The noisy buffer receives data with additive Gaussian noise.

        Args:
            field_name: Name of the field (e.g., "imu_accel", "gps_position").
            data: Raw sensor data tensor of shape (num_envs, ...).
            noise_std: Standard deviation of Gaussian noise to add. If 0, no noise is added.
        """
        # Store raw data to clean buffer
        self.clean[field_name] = data.clone()

        # Store data with noise to noisy buffer
        if noise_std > 0:
            noise = torch.randn_like(data) * noise_std
            self.noisy[field_name] = data + noise
        else:
            self.noisy[field_name] = data.clone()

    def store_raw(self, field_name: str, data: torch.Tensor):
        """Store raw sensor data to clean buffer only.

        Args:
            field_name: Name of the field.
            data: Raw sensor data tensor of shape (num_envs, ...).
        """
        self.clean[field_name] = data.clone()

    def store_noisy(self, field_name: str, data: torch.Tensor, noise_std: float = 0.0):
        """Store data with noise to noisy buffer.

        Args:
            field_name: Name of the field.
            data: Sensor data tensor of shape (num_envs, ...).
            noise_std: Standard deviation of Gaussian noise. If 0, no noise is added.
        """
        if noise_std > 0:
            noise = torch.randn_like(data) * noise_std
            self.noisy[field_name] = data + noise
        else:
            self.noisy[field_name] = data.clone()

    def get_clean(self, field_name: str) -> torch.Tensor:
        """Get clean data for reward computation.

        Args:
            field_name: Name of the field to retrieve.

        Returns:
            Clean sensor data tensor.

        Raises:
            KeyError: If field_name is not found in clean buffer.
        """
        return self.clean[field_name]

    def get_noisy(self, field_name: str) -> torch.Tensor:
        """Get noisy data for observation computation.

        Args:
            field_name: Name of the field to retrieve.

        Returns:
            Noisy sensor data tensor.

        Raises:
            KeyError: If field_name is not found in noisy buffer.
        """
        return self.noisy[field_name]

    def has_field(self, field_name: str) -> bool:
        """Check if a field exists in the data bus.

        Args:
            field_name: Name of the field to check.

        Returns:
            True if field exists in both clean and noisy buffers.
        """
        return field_name in self.clean and field_name in self.noisy

    def get_field_names(self) -> list[str]:
        """Get list of all field names stored in the data bus.

        Returns:
            List of field names.
        """
        return list(self.clean.keys())

    def step(self, dt: float):
        """Advance simulation time.

        Args:
            dt: Time step in seconds.
        """
        self.t_current += dt

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset buffers for specified environments.

        Clears the stored data for the specified environments and resets time to zero.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            # Reset all
            self.t_current.zero_()
            # Clear all stored data
            for field_name in list(self.clean.keys()):
                self.clean[field_name].zero_()
            for field_name in list(self.noisy.keys()):
                self.noisy[field_name].zero_()
        else:
            # Reset specific environments
            self.t_current[env_ids] = 0.0
            for field_name in self.clean.keys():
                self.clean[field_name][env_ids] = 0.0
            for field_name in self.noisy.keys():
                self.noisy[field_name][env_ids] = 0.0

    def clear(self):
        """Clear all stored data from the data bus."""
        self.clean.clear()
        self.noisy.clear()
        self.t_current.zero_()
