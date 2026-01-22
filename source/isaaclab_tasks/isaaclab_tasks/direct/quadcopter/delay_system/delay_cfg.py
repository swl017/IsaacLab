# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for the delay system."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class DelayCfg:
    """Configuration for a single delay buffer.

    This class defines the delay parameters for either actions or observations.
    The delay is specified in terms of simulation time steps (not decimated steps).

    Example usage:
        # Fixed 10-step delay
        delay_cfg = DelayCfg(enabled=True, min_delay=10, max_delay=10)

        # Randomized delay between 5 and 15 steps
        delay_cfg = DelayCfg(enabled=True, min_delay=5, max_delay=15, randomize_on_reset=True)
    """

    enabled: bool = False
    """Whether to enable delay for this buffer. Default is False."""

    min_delay: int = 0
    """Minimum delay in simulation steps. Default is 0 (no delay).

    This value must be non-negative and less than or equal to max_delay.
    """

    max_delay: int = 0
    """Maximum delay in simulation steps. Default is 0 (no delay).

    If max_delay > min_delay, delays are randomized per-environment within [min_delay, max_delay].
    This value determines the history buffer size.
    """

    randomize_on_reset: bool = True
    """Whether to randomize delays for each environment on reset. Default is True.

    If True, each environment gets a random delay uniformly sampled from [min_delay, max_delay].
    If False, all environments use max_delay.
    """

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_delay < 0:
            raise ValueError(f"min_delay must be non-negative, got {self.min_delay}")
        if self.max_delay < self.min_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= min_delay ({self.min_delay})"
            )


@configclass
class DelaySystemCfg:
    """Configuration for the complete delay system.

    This class holds configurations for both action and observation delays.
    Action and observation delays can be configured independently.

    Example usage:
        # Action delay only
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
        )

        # Both action and observation delays
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
        )
    """

    action_delay: DelayCfg = DelayCfg()
    """Configuration for action delay.

    Actions are delayed before being applied to the robot.
    This simulates actuator latency and control loop delays.
    """

    observation_delay: DelayCfg = DelayCfg()
    """Configuration for observation delay.

    Observations are delayed before being returned to the policy.
    This simulates sensor latency and communication delays.
    """
