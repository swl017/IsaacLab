# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DelaySystem class for managing action and observation delays."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from .delay_cfg import DelayCfg, DelaySystemCfg


class DelaySystem:
    """Manages delay buffers for actions and observations.

    This class creates and manages DelayBuffer instances based on the provided configuration.
    It handles initialization, reset, delay randomization, and computation of delayed data.

    The delay system can be used to simulate:
    - Actuator latency (action delay): Time between control command and physical execution
    - Sensor latency (observation delay): Time between physical state and observation availability

    Example usage:
        from isaaclab_tasks.direct.quadcopter.delay_system import DelaySystem, DelaySystemCfg, DelayCfg

        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
        )

        delay_system = DelaySystem(
            cfg=cfg,
            num_envs=4096,
            action_dim=4,
            observation_dim=12,
            device="cuda:0",
        )

        # In pre_physics_step
        delayed_actions = delay_system.compute_delayed_action(actions)

        # In get_observations
        delayed_obs = delay_system.compute_delayed_observation(observations)

        # On reset
        delay_system.reset(env_ids)
    """

    def __init__(
        self,
        cfg: DelaySystemCfg,
        num_envs: int,
        action_dim: int,
        observation_dim: int,
        device: str,
    ):
        """Initialize the delay system.

        Args:
            cfg: The delay system configuration.
            num_envs: Number of parallel environments.
            action_dim: Dimension of the action space.
            observation_dim: Dimension of the observation space.
            device: The device to use for tensors (e.g., "cuda:0" or "cpu").
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.device = device

        # Initialize action delay buffer
        self._action_buffer: DelayBuffer | None = None
        if cfg.action_delay.enabled:
            history_length = max(cfg.action_delay.max_delay, 1)
            self._action_buffer = DelayBuffer(
                history_length=history_length,
                batch_size=num_envs,
                device=device,
            )
            # Initialize with zeros
            self._action_buffer.compute(torch.zeros(num_envs, action_dim, device=device))

        # Initialize observation delay buffer
        self._observation_buffer: DelayBuffer | None = None
        if cfg.observation_delay.enabled:
            history_length = max(cfg.observation_delay.max_delay, 1)
            self._observation_buffer = DelayBuffer(
                history_length=history_length,
                batch_size=num_envs,
                device=device,
            )
            # Initialize with zeros
            self._observation_buffer.compute(torch.zeros(num_envs, observation_dim, device=device))

    @property
    def action_delay_enabled(self) -> bool:
        """Whether action delay is enabled."""
        return self._action_buffer is not None

    @property
    def observation_delay_enabled(self) -> bool:
        """Whether observation delay is enabled."""
        return self._observation_buffer is not None

    @property
    def action_buffer(self) -> DelayBuffer | None:
        """The action delay buffer. None if action delay is disabled."""
        return self._action_buffer

    @property
    def observation_buffer(self) -> DelayBuffer | None:
        """The observation delay buffer. None if observation delay is disabled."""
        return self._observation_buffer

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset delay buffers and randomize delays for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            batch_ids = None
        else:
            batch_ids = env_ids.tolist()

        # Reset and randomize action delays
        if self._action_buffer is not None:
            self._action_buffer.reset(batch_ids)
            self._randomize_delays(
                self._action_buffer,
                self.cfg.action_delay,
                batch_ids,
            )

        # Reset and randomize observation delays
        if self._observation_buffer is not None:
            self._observation_buffer.reset(batch_ids)
            self._randomize_delays(
                self._observation_buffer,
                self.cfg.observation_delay,
                batch_ids,
            )

    def _randomize_delays(
        self,
        buffer: DelayBuffer,
        delay_cfg: DelayCfg,
        batch_ids: list[int] | None = None,
    ):
        """Randomize delays for the given buffer.

        Args:
            buffer: The delay buffer to configure.
            delay_cfg: The delay configuration.
            batch_ids: Batch indices to randomize. If None, randomizes all.
        """
        if delay_cfg.randomize_on_reset and delay_cfg.min_delay < delay_cfg.max_delay:
            # Randomize delays uniformly within [min_delay, max_delay]
            if batch_ids is None:
                num_to_sample = self.num_envs
            else:
                num_to_sample = len(batch_ids)

            random_delays = torch.randint(
                low=delay_cfg.min_delay,
                high=delay_cfg.max_delay + 1,
                size=(num_to_sample,),
                device=self.device,
                dtype=torch.int,
            )
            buffer.set_time_lag(random_delays, batch_ids)
        else:
            # Use max_delay for all environments
            buffer.set_time_lag(delay_cfg.max_delay, batch_ids)

    def compute_delayed_action(self, action: torch.Tensor) -> torch.Tensor:
        """Compute delayed action.

        Args:
            action: The current action tensor of shape (num_envs, action_dim).

        Returns:
            The delayed action tensor. If delay is disabled, returns the input action.
        """
        if self._action_buffer is None:
            return action
        return self._action_buffer.compute(action)

    def compute_delayed_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute delayed observation.

        Args:
            observation: The current observation tensor of shape (num_envs, obs_dim).

        Returns:
            The delayed observation tensor. If delay is disabled, returns the input observation.
        """
        if self._observation_buffer is None:
            return observation
        return self._observation_buffer.compute(observation)

    def get_delay_info(self) -> dict:
        """Get current delay information for logging.

        Returns:
            Dictionary containing delay statistics including min, max, and mean delays.
        """
        info = {}
        if self._action_buffer is not None:
            info["action_delay_min"] = self._action_buffer.min_time_lag
            info["action_delay_max"] = self._action_buffer.max_time_lag
            info["action_delay_mean"] = self._action_buffer.time_lags.float().mean().item()
        if self._observation_buffer is not None:
            info["observation_delay_min"] = self._observation_buffer.min_time_lag
            info["observation_delay_max"] = self._observation_buffer.max_time_lag
            info["observation_delay_mean"] = self._observation_buffer.time_lags.float().mean().item()
        return info

    def get_action_delays(self) -> torch.Tensor | None:
        """Get the current action delay for each environment.

        Returns:
            Tensor of shape (num_envs,) with delay values, or None if action delay is disabled.
        """
        if self._action_buffer is None:
            return None
        return self._action_buffer.time_lags.clone()

    def get_observation_delays(self) -> torch.Tensor | None:
        """Get the current observation delay for each environment.

        Returns:
            Tensor of shape (num_envs,) with delay values, or None if observation delay is disabled.
        """
        if self._observation_buffer is None:
            return None
        return self._observation_buffer.time_lags.clone()

    def set_action_delays(self, delays: int | torch.Tensor, env_ids: list[int] | None = None):
        """Manually set action delays.

        Args:
            delays: Delay value(s) to set. Can be a single int or tensor of shape (len(env_ids),).
            env_ids: Environment indices to set delays for. If None, sets for all environments.
        """
        if self._action_buffer is not None:
            self._action_buffer.set_time_lag(delays, env_ids)

    def set_observation_delays(self, delays: int | torch.Tensor, env_ids: list[int] | None = None):
        """Manually set observation delays.

        Args:
            delays: Delay value(s) to set. Can be a single int or tensor of shape (len(env_ids),).
            env_ids: Environment indices to set delays for. If None, sets for all environments.
        """
        if self._observation_buffer is not None:
            self._observation_buffer.set_time_lag(delays, env_ids)
