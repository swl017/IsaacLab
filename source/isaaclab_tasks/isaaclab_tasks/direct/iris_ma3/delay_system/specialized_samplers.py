"""
Specialized samplers for domain-specific data types and filtering.

This module provides samplers with specialized behavior:
- FirstOrderLagSampler: Continuous first-order filtering
- QuaternionFirstOrderLagSampler: SLERP-based quaternion filtering
- PassthroughSampler: No delay, instant updates (for static data)
"""

from __future__ import annotations
import torch
from typing import Optional


class FirstOrderLagSampler:
    """
    First-order lag filter for continuous state smoothing.

    Implements exponential filtering:
        x_filtered = x_filtered + alpha * (x_measured - x_filtered)
    where alpha = dt / (dt + tau)

    This is NOT a sample-and-hold sampler - it updates every step with smooth filtering.
    """

    def __init__(
        self,
        num_envs: int,
        state_dim: int,
        time_constant: torch.Tensor | float,
        dt: float,
        device: torch.device | str,
        initial_state: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            num_envs: Number of environments.
            state_dim: Dimension of the state.
            time_constant: Time constant for the filter (tau). Can be:
                - float: Same tau for all envs and dimensions
                - Tensor (N, 1): Per-env tau, same for all dimensions
                - Tensor (N, state_dim): Per-env, per-dimension tau
            dt: Time step.
            device: Device to allocate tensors on.
            initial_state: Initial state of the filter (N, state_dim).
        """
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.initial_state = initial_state
        self.dt = dt
        self.device = device

        # Handle time constant
        if isinstance(time_constant, float):
            self.tau = torch.full((num_envs, state_dim), time_constant, device=device)
        else:
            # Ensure correct shape
            if time_constant.dim() == 1:
                self.tau = time_constant.unsqueeze(-1).expand(num_envs, state_dim)
            else:
                self.tau = time_constant.to(device)

        # Compute filter coefficient
        self.alpha = dt / (dt + self.tau)  # (N, state_dim)

        # Initialize filtered state
        self.filtered_state = initial_state.clone() if initial_state is not None else \
                              torch.zeros(num_envs, state_dim, device=device)

        # No explicit latency (continuous filtering)
        self.accumulated_latency = torch.zeros(num_envs, device=device)

        self.post_init()

    def post_init(self):
        """Hook for subclass initialization."""
        pass

    def update_time_constants(self, time_constant: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        Update time constants for specific environments.

        Args:
            time_constant: New time constants (len(env_ids), state_dim) or (len(env_ids), 1)
            env_ids: Environment indices to update. None means all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.tau[env_ids] = time_constant
        self.alpha[env_ids] = self.dt / (self.dt + time_constant)

    def update_config(self, time_constant: Optional[float] = None):
        """
        Dynamically update sampler configuration parameters without recreation.

        This method enables curriculum learning by allowing external modules to
        adjust time constants during training.

        Args:
            time_constant: New time constant value (tau) for all environments.
                          If None, no update is performed.

        Note:
            - Updates time constant uniformly across all environments
            - Recalculates filter coefficients (alpha) automatically
            - For per-environment updates, use update_time_constants() instead
        """
        if time_constant is not None:
            assert time_constant > 0, f"time_constant must be > 0, got {time_constant}"
            self.tau[:] = time_constant
            self.alpha = self.dt / (self.dt + self.tau)

    def update(
        self,
        data: torch.Tensor,
        t_current: Optional[torch.Tensor] = None,
        force_update: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Update filter with new measurements.

        Args:
            data: Measured state (num_envs, state_dim)
            t_current: Current time (unused, for interface compatibility)
            force_update: Boolean mask for forced updates (unused, for interface compatibility)

        Returns:
            Tuple of (filtered_state, info_dict)
        """
        self.filtered_state = self.filtered_state + self.alpha * (data - self.filtered_state)

        info = {
            'sampled': torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            'dropped_out': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            'accumulated_latency': self.accumulated_latency.clone(),
        }

        return self.filtered_state.clone(), info

    def reset(self, env_ids: Optional[torch.Tensor] = None, initial_data: Optional[torch.Tensor] = None):
        """
        Reset filter state for specified environments.

        Args:
            env_ids: Indices of environments to reset. None means reset all.
            initial_data: Initial state for reset environments (len(env_ids), state_dim).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if initial_data is not None:
            self.filtered_state[env_ids] = initial_data
        else:
            self.filtered_state[env_ids] = 0.0


class QuaternionFirstOrderLagSampler(FirstOrderLagSampler):
    """
    First-order lag filter for quaternions using SLERP interpolation.

    Implements spherical linear interpolation (SLERP) instead of linear interpolation
    to maintain quaternion properties (unit norm, geodesic path).
    """

    def __init__(
        self,
        num_envs: int,
        time_constant: torch.Tensor | float,
        dt: float,
        device: torch.device | str,
        initial_state: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            num_envs: Number of environments.
            time_constant: Time constant for the filter (tau).
            dt: Time step.
            device: Device to allocate tensors on.
            initial_state: Initial quaternion state (N, 4) in (w, x, y, z) format.
        """
        state_dim = 4  # Quaternion has 4 components
        super().__init__(num_envs, state_dim, time_constant, dt, device, initial_state)

    def post_init(self):
        """Initialize to unit quaternions."""
        if self.initial_state is not None:
            # Normalize provided quaternions
            self.filtered_state = self.initial_state / (
                torch.norm(self.initial_state, dim=-1, keepdim=True) + 1e-8
            )
        else:
            # Default to identity quaternion (1, 0, 0, 0)
            self.filtered_state = torch.zeros(self.num_envs, 4, device=self.device)
            self.filtered_state[:, 0] = 1.0

    def update(
        self,
        data: torch.Tensor,
        t_current: Optional[torch.Tensor] = None,
        force_update: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Update filter using SLERP.

        Args:
            data: Measured quaternions (num_envs, 4) in (w, x, y, z) format
            t_current: Current time (unused)
            force_update: Forced update mask (unused)

        Returns:
            Tuple of (filtered_quaternions, info_dict)
        """
        # Use SLERP with alpha as interpolation factor
        # Note: alpha is (N, 4) but SLERP uses per-env scalar, take mean or first component
        alpha_scalar = self.alpha[:, 0]  # Use first component as scalar (all should be same for quat)

        self.filtered_state = self._slerp(self.filtered_state, data, alpha_scalar)

        info = {
            'sampled': torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            'dropped_out': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            'accumulated_latency': self.accumulated_latency.clone(),
        }

        return self.filtered_state.clone(), info

    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Spherical linear interpolation between quaternions.

        Args:
            q0: Starting quaternion (N, 4).
            q1: Ending quaternion (N, 4).
            t: Interpolation factor (N,) in [0, 1].

        Returns:
            Interpolated quaternion (N, 4).
        """
        # Normalize quaternions
        q0 = q0 / (torch.norm(q0, dim=-1, keepdim=True) + 1e-8)
        q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)

        # Compute dot product
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

        # If dot < 0, negate q1 to take shorter path
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.abs(dot)

        # For very similar quaternions, use linear interpolation (LERP)
        mask = (dot > 0.9995).squeeze(-1)

        # SLERP calculation
        theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_theta = torch.sin(theta)
        sin_theta = torch.where(sin_theta < 1e-8, torch.ones_like(sin_theta), sin_theta)

        t_exp = t.unsqueeze(-1)
        w0 = torch.sin((1 - t_exp) * theta) / sin_theta
        w1 = torch.sin(t_exp * theta) / sin_theta
        result_slerp = w0 * q0 + w1 * q1

        # LERP calculation (fallback for similar quaternions)
        result_lerp = (1 - t_exp) * q0 + t_exp * q1
        result_lerp = result_lerp / (torch.norm(result_lerp, dim=-1, keepdim=True) + 1e-8)

        # Choose based on similarity
        return torch.where(mask.unsqueeze(-1), result_lerp, result_slerp)


class PassthroughSampler:
    """
    Passthrough sampler with no delay or filtering.

    Used for static or nearly-instant data (e.g., camera intrinsics, configuration).
    Simply returns input data immediately.
    """

    def __init__(self, num_envs: int, device: torch.device):
        """
        Args:
            num_envs: Number of environments.
            device: Device to allocate tensors on.
        """
        self.num_envs = num_envs
        self.device = device
        self.accumulated_latency = torch.zeros(num_envs, device=device)

    def update(
        self,
        data: torch.Tensor,
        t_current: Optional[torch.Tensor] = None,
        force_update: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Pass through data unchanged.

        Args:
            data: Input data
            t_current: Current time (unused)
            force_update: Forced update mask (unused)

        Returns:
            Tuple of (data, info_dict)
        """
        info = {
            'sampled': torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            'dropped_out': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            'accumulated_latency': self.accumulated_latency.clone(),
        }

        return data.clone(), info

    def reset(self, env_ids: Optional[torch.Tensor] = None, initial_data: Optional[torch.Tensor] = None):
        """
        Reset (no-op for passthrough).

        Args:
            env_ids: Environment indices (unused).
            initial_data: Initial data (unused).
        """
        pass
