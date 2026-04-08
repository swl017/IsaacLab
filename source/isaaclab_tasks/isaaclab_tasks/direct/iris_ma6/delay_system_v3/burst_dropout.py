# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gilbert-Elliott burst dropout model for correlated packet loss.

Real wireless links drop packets in bursts (5-20 consecutive frames), not
independently per frame. This module implements a two-state Markov chain
(Good/Bad) per directional communication channel to simulate bursty loss.

Channel j->i represents data flowing from agent j to agent i. Each direction
has an independent Markov state, enabling asymmetric dropout (agent B can be
invisible to A while still seeing A).

Calling contract:
- advance(): WRITE -- transitions Markov chain and caches dropout masks. Once per step.
- sample_mask(i, j): READ -- returns cached mask for channel j->i. Safe to call multiple times.
- set_burst_params(): CONFIG -- curriculum control.
- reset(): resets Markov state to Good.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional

from .delay_cfg_v3 import DistributionCfg
from .sampling_strategies import DistributionSampler


# Markov states
_GOOD = 0
_BAD = 1


@dataclass
class BurstDropoutCfg:
    """Configuration for Gilbert-Elliott burst dropout model.

    The model uses a two-state Markov chain per directional channel:
    - Good state: low dropout probability (normal operation)
    - Bad state: high dropout probability (burst/outage)

    Mean burst length = 1 / p_recovery (in steps at observation rate).
    Mean good length = 1 / p_onset (in steps).
    """

    enabled: bool = False
    """Whether burst dropout is active. When False, i.i.d. Bernoulli dropout is used."""

    # ===== Markov Transition Probabilities =====

    p_onset: float = 0.01
    """Probability of transitioning from Good to Bad state per step.
    Controls burst frequency. Mean good-run length = 1/p_onset.
    Default 0.01 -> mean 100 steps between bursts."""

    p_recovery: float = 0.1
    """Probability of transitioning from Bad to Good state per step.
    Controls burst duration. Mean burst length = 1/p_recovery.
    Default 0.1 -> mean 10-step bursts."""

    # ===== State-Dependent Dropout Probabilities =====

    good_dropout_prob: float = 0.01
    """Dropout probability in Good state. Low baseline loss."""

    bad_dropout_prob: float = 0.9
    """Dropout probability in Bad state. Near-total loss during burst."""

    # ===== Per-Episode Randomization =====

    p_onset_distribution: Optional[DistributionCfg] = None
    """Optional distribution for per-episode p_onset randomization.
    When set, p_onset is resampled per environment at episode reset,
    creating varied burst frequencies across episodes."""

    p_recovery_distribution: Optional[DistributionCfg] = None
    """Optional distribution for per-episode p_recovery randomization.
    When set, p_recovery is resampled per environment at episode reset,
    creating varied burst lengths across episodes."""


class BurstDropoutSampler:
    """Gilbert-Elliott two-state Markov chain for burst dropout.

    Manages a Markov state tensor of shape (num_envs, num_agents, num_agents)
    where entry [e, i, j] is the state of channel j->i in environment e.
    Diagonal entries (i==i) are unused.

    Usage:
        sampler = BurstDropoutSampler(cfg, num_envs=64, num_agents=3, device=device)

        # Each sim step (WRITE -- once):
        sampler.advance()

        # Query dropout mask for channel j->i (READ -- multiple times ok):
        mask = sampler.sample_mask(i=0, j=1)  # shape (num_envs,), True = dropped
    """

    def __init__(
        self,
        cfg: BurstDropoutCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        self._cfg = cfg
        self._num_envs = num_envs
        self._num_agents = num_agents
        self._device = device

        # Markov state: 0=Good, 1=Bad. Shape (num_envs, num_agents, num_agents)
        self._state = torch.zeros(
            num_envs, num_agents, num_agents, dtype=torch.long, device=device
        )

        # Per-env transition probabilities. Shape (num_envs,) -- shared across all
        # channels within an env, but randomized per episode.
        self._p_onset = torch.full((num_envs,), cfg.p_onset, device=device)
        self._p_recovery = torch.full((num_envs,), cfg.p_recovery, device=device)

        # Cached dropout mask after advance(). Shape (num_envs, num_agents, num_agents), bool
        self._dropout_mask = torch.zeros(
            num_envs, num_agents, num_agents, dtype=torch.bool, device=device
        )

        # Base values for curriculum scaling
        self._base_p_onset = cfg.p_onset
        self._base_p_recovery = cfg.p_recovery

        # Optional per-episode distribution samplers
        self._p_onset_sampler: Optional[DistributionSampler] = None
        self._p_recovery_sampler: Optional[DistributionSampler] = None

        if cfg.p_onset_distribution is not None:
            self._p_onset_sampler = DistributionSampler(
                cfg.p_onset_distribution, num_envs, device
            )
        if cfg.p_recovery_distribution is not None:
            self._p_recovery_sampler = DistributionSampler(
                cfg.p_recovery_distribution, num_envs, device
            )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def state(self) -> torch.Tensor:
        """Current Markov state. Shape (num_envs, num_agents, num_agents). 0=Good, 1=Bad."""
        return self._state

    @property
    def dropout_mask(self) -> torch.Tensor:
        """Cached dropout mask. Shape (num_envs, num_agents, num_agents). True=dropped."""
        return self._dropout_mask

    def advance(self) -> None:
        """Transition Markov chain and cache dropout masks. WRITE -- call once per step.

        Two-step process:
        1. Transition: each channel independently transitions based on current state.
        2. Dropout: sample dropout from state-dependent probability.
        """
        E, A = self._num_envs, self._num_agents

        # --- Step 1: Markov transition ---
        # Random values for transition decision: (E, A, A)
        rand_transition = torch.rand(E, A, A, device=self._device)

        # Expand p_onset and p_recovery to (E, 1, 1) for broadcasting
        p_onset = self._p_onset.view(E, 1, 1)
        p_recovery = self._p_recovery.view(E, 1, 1)

        # Transition from Good (0): if rand < p_onset -> go to Bad (1)
        good_to_bad = (self._state == _GOOD) & (rand_transition < p_onset)

        # Transition from Bad (1): if rand < p_recovery -> go to Good (0)
        bad_to_good = (self._state == _BAD) & (rand_transition < p_recovery)

        # Apply transitions
        self._state = torch.where(good_to_bad, torch.ones_like(self._state), self._state)
        self._state = torch.where(bad_to_good, torch.zeros_like(self._state), self._state)

        # --- Step 2: Sample dropout mask from state-dependent probabilities ---
        rand_dropout = torch.rand(E, A, A, device=self._device)

        # Dropout probability depends on current state
        dropout_prob = torch.where(
            self._state == _GOOD,
            torch.tensor(self._cfg.good_dropout_prob, device=self._device),
            torch.tensor(self._cfg.bad_dropout_prob, device=self._device),
        )

        self._dropout_mask = rand_dropout < dropout_prob

    def sample_mask(self, i: int, j: int) -> torch.Tensor:
        """Return cached dropout mask for channel j->i. READ -- safe to call multiple times.

        Args:
            i: Receiving agent index.
            j: Sending agent index.

        Returns:
            Boolean tensor of shape (num_envs,). True = packet dropped.
        """
        return self._dropout_mask[:, i, j]

    def set_burst_params(self, p_onset: float, p_recovery: float) -> None:
        """Set burst parameters for curriculum control.

        Args:
            p_onset: Good->Bad transition probability. 0 disables bursts.
            p_recovery: Bad->Good transition probability.
        """
        self._base_p_onset = max(0.0, min(1.0, p_onset))
        self._base_p_recovery = max(0.0, min(1.0, p_recovery))

        # Update per-env values (unless overridden by distribution sampler)
        if self._p_onset_sampler is None:
            self._p_onset.fill_(self._base_p_onset)
        if self._p_recovery_sampler is None:
            self._p_recovery.fill_(self._base_p_recovery)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset Markov state to Good for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self._state.zero_()
            self._dropout_mask.zero_()
        else:
            self._state[env_ids] = _GOOD
            self._dropout_mask[env_ids] = False

    def randomize_params(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Resample p_onset and p_recovery per environment at episode reset.

        Only active when distribution configs are provided. Otherwise no-op.

        Args:
            env_ids: Environment indices being reset. If None, resamples all.
        """
        if self._p_onset_sampler is not None:
            samples = self._p_onset_sampler.sample(env_ids)
            samples = torch.clamp(samples, min=0.0, max=1.0)
            if env_ids is None:
                self._p_onset = samples
            else:
                self._p_onset[env_ids] = samples

        if self._p_recovery_sampler is not None:
            samples = self._p_recovery_sampler.sample(env_ids)
            samples = torch.clamp(samples, min=0.0, max=1.0)
            if env_ids is None:
                self._p_recovery = samples
            else:
                self._p_recovery[env_ids] = samples
