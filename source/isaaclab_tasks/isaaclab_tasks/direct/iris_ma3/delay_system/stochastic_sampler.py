"""
Stochastic Sample-and-Hold with Dropout Mechanism

This module implements a unified pattern for modeling time-shifting phenomena in sim-to-real RL:
- Latency (processing delays)
- Sampling rate discrepancy (FPS throttling)
- Dropout (packet loss, frame drops)

Key insight: All these phenomena share a common "stochastic sample-and-hold with dropout" algorithm.
"""

from __future__ import annotations
import torch
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class DistributionConfig:
    """Configuration for stochastic sampling distributions."""

    distribution_type: Literal["uniform", "normal", "constant"]
    """Type of distribution: uniform, normal, or constant"""

    # For uniform distribution
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # For normal distribution
    mean: Optional[float] = None
    std: Optional[float] = None

    # For constant
    value: Optional[float] = None

    def validate(self):
        """Validate that required parameters are set for the distribution type."""
        if self.distribution_type == "uniform":
            assert self.min_value is not None and self.max_value is not None, \
                "Uniform distribution requires min_value and max_value"
            assert self.min_value <= self.max_value, \
                f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
        elif self.distribution_type == "normal":
            assert self.mean is not None and self.std is not None, \
                "Normal distribution requires mean and std"
            assert self.std >= 0, f"std ({self.std}) must be >= 0"
        elif self.distribution_type == "constant":
            assert self.value is not None, \
                "Constant distribution requires value"
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")


@dataclass
class SamplerConfig:
    """Configuration for a stochastic sampler."""

    # Sampling period configuration
    period_dist: DistributionConfig
    """Distribution for sampling period (time between samples)"""

    # Dropout configuration
    dropout_dist: Optional[DistributionConfig] = None
    """Distribution for dropout probability (0-1 range). None means no dropout."""

    # Optional latency/delay
    latency_dist: Optional[DistributionConfig] = None
    """Distribution for processing latency. None means no latency."""

    def __post_init__(self):
        self.period_dist.validate()
        if self.dropout_dist is not None:
            self.dropout_dist.validate()
        if self.latency_dist is not None:
            self.latency_dist.validate()


class StochasticSampler:
    """
    Generic stochastic sample-and-hold mechanism with dropout.

    This class implements the core algorithm:
    1. Query at high frequency (simulation rate)
    2. Sample at lower, stochastic frequency (period_dist)
    3. Randomly dropout samples (dropout_dist)
    4. Hold last valid sample between updates
    5. Optionally delay samples (latency_dist)

    Example use cases:
    - Object detector: 20Hz ± 5Hz, 30% dropout, 300ms ± 50ms latency
    - Communication: 30Hz ± 10Hz, 5% dropout, 100ms ± 20ms latency
    - Sensor: 100Hz constant, 0% dropout, 10ms ± 2ms latency
    """

    def __init__(
        self,
        config: SamplerConfig,
        num_envs: int,
        device: torch.device,
        dt: float,
    ):
        """
        Args:
            config: Sampler configuration
            num_envs: Number of parallel environments
            device: Device to allocate tensors on
            dt: Simulation timestep (seconds)
        """
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.dt = dt

        # State tracking
        self.current_time = torch.zeros(num_envs, device=device)
        self.last_sample_time = torch.zeros(num_envs, device=device)
        self.next_sample_threshold = self._sample_period()

        # Held data (initialized to None, set on first update)
        self.held_data: Optional[torch.Tensor] = None

        # For latency: store samples with their release times
        if config.latency_dist is not None:
            self.pending_samples = []  # List of dicts with env_idx, release_time, data
            self.has_latency = True
        else:
            self.has_latency = False

        # Track accumulated latency for timestamp calculation
        self.accumulated_latency = torch.zeros(num_envs, device=device)

    def _sample_from_dist(self, dist_config: DistributionConfig) -> torch.Tensor:
        """Sample values from a distribution configuration."""
        if dist_config.distribution_type == "uniform":
            return torch.rand(self.num_envs, device=self.device) * \
                   (dist_config.max_value - dist_config.min_value) + dist_config.min_value
        elif dist_config.distribution_type == "normal":
            return torch.randn(self.num_envs, device=self.device) * \
                   dist_config.std + dist_config.mean
        elif dist_config.distribution_type == "constant":
            return torch.full((self.num_envs,), dist_config.value, device=self.device)

    def _sample_period(self) -> torch.Tensor:
        """Sample next sampling period for each environment."""
        period = self._sample_from_dist(self.config.period_dist)
        return torch.clamp(period, min=self.dt)  # Ensure period >= dt

    def _sample_dropout(self) -> torch.Tensor:
        """Sample dropout mask (True = dropout, False = keep)."""
        if self.config.dropout_dist is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        dropout_prob = self._sample_from_dist(self.config.dropout_dist)
        dropout_prob = torch.clamp(dropout_prob, min=0.0, max=1.0)
        random_vals = torch.rand(self.num_envs, device=self.device)
        return random_vals < dropout_prob

    def _sample_latency(self) -> torch.Tensor:
        """Sample processing latency for each environment."""
        if self.config.latency_dist is None:
            return torch.zeros(self.num_envs, device=self.device)

        latency = self._sample_from_dist(self.config.latency_dist)
        return torch.clamp(latency, min=0.0)

    def update(
        self,
        data: torch.Tensor,
        t_current: Optional[torch.Tensor] = None,
        force_update: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Update sampler with new data and return current held values.

        Args:
            data: Input data tensor of shape (num_envs, ...)
            t_current: Current simulation time (num_envs,). If None, uses internal counter.
            force_update: Boolean mask to force immediate update for specific envs (num_envs,).
                         Useful for resets. None means no forced updates.

        Returns:
            Tuple of (held_data, info_dict):
                - held_data: Current held data (stale or fresh)
                - info_dict: Dictionary with metadata:
                    - 'sampled': Boolean mask of environments that sampled this step
                    - 'dropped_out': Boolean mask of environments that dropped out
                    - 'accumulated_latency': Current accumulated latency per env
        """
        # Initialize held data on first call
        if self.held_data is None:
            self.held_data = data.clone()

        if force_update is None:
            force_update = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Advance time
        if t_current is not None:
            self.current_time = t_current.clone()
        else:
            self.current_time += self.dt

        # Check if it's time to sample
        is_sample_period = (self.current_time - self.last_sample_time) >= self.next_sample_threshold

        # Check for dropout
        is_dropout = self._sample_dropout()

        # Determine which environments should update: (sample period AND not dropout) OR forced
        should_sample = (is_sample_period & ~is_dropout) | force_update

        # Metadata for output
        info = {
            'sampled': should_sample.clone(),
            'dropped_out': is_dropout.clone(),
            'accumulated_latency': self.accumulated_latency.clone(),
        }

        # For environments that should sample
        if should_sample.any():
            if self.has_latency:
                # Add to pending samples with release time
                latency = self._sample_latency()
                release_time = self.current_time + latency

                # Store per-environment pending samples
                for env_idx in torch.where(should_sample)[0]:
                    self.pending_samples.append({
                        'env_idx': env_idx.item(),
                        'release_time': release_time[env_idx].item(),
                        'data': data[env_idx].clone(),
                    })
                    # Update accumulated latency
                    self.accumulated_latency[env_idx] = latency[env_idx]

                # Process pending samples that are ready
                self._process_pending_samples()
            else:
                # Immediate update (no latency)
                self.held_data[should_sample] = data[should_sample]
                self.accumulated_latency[should_sample] = 0.0

            # Update state for environments that sampled
            self.last_sample_time[should_sample] = self.current_time[should_sample]
            self.next_sample_threshold[should_sample] = self._sample_period()[should_sample]

        return self.held_data.clone(), info

    def _process_pending_samples(self):
        """Process pending samples that have passed their latency period."""
        if not self.pending_samples:
            return

        # Find samples ready for release
        ready_indices = []
        for idx, sample in enumerate(self.pending_samples):
            env_idx = sample['env_idx']
            if self.current_time[env_idx] >= sample['release_time']:
                ready_indices.append(idx)

                # Update held data
                self.held_data[env_idx] = sample['data']

        # Remove processed samples (in reverse to maintain indices)
        for idx in reversed(ready_indices):
            self.pending_samples.pop(idx)

    def reset(self, env_ids: Optional[torch.Tensor] = None, initial_data: Optional[torch.Tensor] = None):
        """
        Reset sampler state for specified environments.

        Args:
            env_ids: Indices of environments to reset. None means reset all.
            initial_data: Initial data for reset environments. None means keep current.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.current_time[env_ids] = 0.0
        self.last_sample_time[env_ids] = 0.0
        self.next_sample_threshold[env_ids] = self._sample_period()[env_ids]
        self.accumulated_latency[env_ids] = 0.0

        if initial_data is not None:
            self.held_data[env_ids] = initial_data[env_ids]

        # Clear pending samples for reset environments
        if self.has_latency:
            env_ids_set = set(env_ids.cpu().tolist())
            self.pending_samples = [
                s for s in self.pending_samples
                if s['env_idx'] not in env_ids_set
            ]

    def update_config(
        self,
        period_mean: Optional[float] = None,
        period_std: Optional[float] = None,
        period_min: Optional[float] = None,
        period_max: Optional[float] = None,
        latency_mean: Optional[float] = None,
        latency_std: Optional[float] = None,
        dropout_prob: Optional[float] = None,
    ):
        """
        Dynamically update sampler configuration parameters without recreation.

        This method enables curriculum learning by allowing external modules to
        adjust delay parameters during training.

        Args:
            period_mean: New mean for period distribution (normal only)
            period_std: New std for period distribution (normal only)
            period_min: New min for period distribution (uniform only)
            period_max: New max for period distribution (uniform only)
            latency_mean: New mean for latency distribution (normal only)
            latency_std: New std for latency distribution (normal only)
            dropout_prob: New dropout probability (constant value)

        Note:
            - Only updates parameters compatible with current distribution types
            - Validates parameter ranges before applying
            - Ignores parameters not compatible with current config
        """
        # Update period distribution
        if self.config.period_dist.distribution_type == "normal":
            if period_mean is not None:
                self.config.period_dist.mean = period_mean
            if period_std is not None:
                assert period_std >= 0, f"period_std must be >= 0, got {period_std}"
                self.config.period_dist.std = period_std
        elif self.config.period_dist.distribution_type == "uniform":
            if period_min is not None:
                self.config.period_dist.min_value = period_min
            if period_max is not None:
                self.config.period_dist.max_value = period_max
            # Validate range if both are set
            if self.config.period_dist.min_value is not None and \
               self.config.period_dist.max_value is not None:
                assert self.config.period_dist.min_value <= self.config.period_dist.max_value, \
                    f"period_min ({self.config.period_dist.min_value}) must be <= " \
                    f"period_max ({self.config.period_dist.max_value})"

        # Update latency distribution
        if self.config.latency_dist is not None:
            if self.config.latency_dist.distribution_type == "normal":
                if latency_mean is not None:
                    self.config.latency_dist.mean = latency_mean
                if latency_std is not None:
                    assert latency_std >= 0, f"latency_std must be >= 0, got {latency_std}"
                    self.config.latency_dist.std = latency_std
            elif self.config.latency_dist.distribution_type == "constant":
                if latency_mean is not None:
                    self.config.latency_dist.value = latency_mean

        # Update dropout distribution
        if self.config.dropout_dist is not None:
            if dropout_prob is not None:
                assert 0.0 <= dropout_prob <= 1.0, \
                    f"dropout_prob must be in [0, 1], got {dropout_prob}"
                if self.config.dropout_dist.distribution_type == "constant":
                    self.config.dropout_dist.value = dropout_prob
                elif self.config.dropout_dist.distribution_type == "normal":
                    self.config.dropout_dist.mean = dropout_prob
                elif self.config.dropout_dist.distribution_type == "uniform":
                    # For uniform dropout, adjust midpoint while keeping range
                    range_half = (self.config.dropout_dist.max_value -
                                  self.config.dropout_dist.min_value) / 2.0
                    self.config.dropout_dist.min_value = max(0.0, dropout_prob - range_half)
                    self.config.dropout_dist.max_value = min(1.0, dropout_prob + range_half)


# ==================== Convenience Factory Functions ====================

def create_detector_sampler(
    fps_mean: float,
    fps_std: float,
    latency_mean: float,
    latency_std: float,
    dropout_rate: float,
    num_envs: int,
    device: torch.device,
    dt: float,
) -> StochasticSampler:
    """
    Create a sampler configured for an object detector.

    Args:
        fps_mean: Mean detection rate (Hz)
        fps_std: Std deviation of detection rate (Hz)
        latency_mean: Mean processing latency (seconds)
        latency_std: Std deviation of latency (seconds)
        dropout_rate: Frame dropout rate (0-1)
        num_envs: Number of environments
        device: Device
        dt: Simulation timestep

    Returns:
        Configured StochasticSampler for detector
    """
    config = SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="normal",
            mean=1.0 / fps_mean,
            std=fps_std / (fps_mean ** 2),  # Convert FPS std to period std
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
    comm_rate_min: float,
    comm_rate_max: float,
    latency_mean: float,
    latency_std: float,
    dropout_rate: float,
    num_envs: int,
    device: torch.device,
    dt: float,
) -> StochasticSampler:
    """
    Create a sampler configured for network communication.

    Args:
        comm_rate_min: Minimum communication rate (Hz)
        comm_rate_max: Maximum communication rate (Hz)
        latency_mean: Mean network latency (seconds)
        latency_std: Std deviation of latency (seconds)
        dropout_rate: Packet dropout rate (0-1)
        num_envs: Number of environments
        device: Device
        dt: Simulation timestep

    Returns:
        Configured StochasticSampler for communication
    """
    config = SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="uniform",
            min_value=1.0 / comm_rate_max,
            max_value=1.0 / comm_rate_min,
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


def create_sensor_sampler(
    fps: float,
    latency: float,
    num_envs: int,
    device: torch.device,
    dt: float,
) -> StochasticSampler:
    """
    Create a sampler for a deterministic sensor (e.g., IMU, encoders).

    Args:
        fps: Sensor sampling rate (Hz)
        latency: Fixed sensor latency (seconds)
        num_envs: Number of environments
        device: Device
        dt: Simulation timestep

    Returns:
        Configured StochasticSampler for sensor
    """
    config = SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="constant",
            value=1.0 / fps,
        ),
        latency_dist=DistributionConfig(
            distribution_type="constant",
            value=latency,
        ),
        dropout_dist=None,  # No dropout for deterministic sensors
    )

    return StochasticSampler(config, num_envs, device, dt)
