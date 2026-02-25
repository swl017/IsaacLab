# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-skip stacking wrapper for multi-agent vectorized environments.

Stacks K observation frames spread out by a skip interval, giving MLP policies
temporal context without the full dimensionality of consecutive frame stacking.

Example with num_stack=4, frame_skip=10 at 25Hz control:
    Output = [obs_{t-30}, obs_{t-20}, obs_{t-10}, obs_t]  (oldest first)
    Temporal span: 1.2 seconds, observation dim: 4 * obs_dim
"""

from __future__ import annotations

import gymnasium
import numpy as np
import torch
from typing import Any, Dict, Tuple


class MultiAgentFrameStackWrapper:
    """Frame-skip stacking wrapper for multi-agent vectorized environments.

    Wraps a skrl MultiAgentEnvWrapper to stack K observation frames separated
    by ``frame_skip`` timesteps. Uses a GPU ring buffer for efficiency.

    Args:
        env: The wrapped skrl multi-agent environment.
        num_stack: Number of frames to stack (K). Default: 4.
        frame_skip: Gap between stacked frames in timesteps. Default: 10.
            At 25Hz control, skip=10 means 400ms between frames.
    """

    def __init__(self, env: Any, num_stack: int = 4, frame_skip: int = 10) -> None:
        self._env = env
        self._num_stack = num_stack
        self._frame_skip = frame_skip
        self._buffer_size = frame_skip * (num_stack - 1) + 1

        # Build stacked observation spaces
        self._stacked_obs_spaces: Dict[str, gymnasium.spaces.Box] = {}
        for agent_id, space in env.observation_spaces.items():
            low = np.tile(space.low, num_stack)
            high = np.tile(space.high, num_stack)
            self._stacked_obs_spaces[agent_id] = gymnasium.spaces.Box(
                low=low, high=high, dtype=space.dtype
            )

        # History buffers — initialized lazily on first reset/step
        self._buffers: Dict[str, torch.Tensor] = {}  # {agent_id: [N, buffer_size, obs_dim]}
        self._head: int = 0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Properties — delegate to wrapped env
    # ------------------------------------------------------------------

    @property
    def observation_spaces(self) -> Dict[str, gymnasium.spaces.Box]:
        return self._stacked_obs_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def agents(self):
        return self._env.agents

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def num_agents(self):
        return self._env.num_agents

    @property
    def device(self):
        return self._env.device

    @property
    def unwrapped(self):
        return self._env.unwrapped

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _init_buffers(self, obs_dict: Dict[str, torch.Tensor]) -> None:
        """Initialize ring buffers from first observation."""
        for agent_id, obs in obs_dict.items():
            N, obs_dim = obs.shape
            # Fill entire buffer with the initial observation
            buf = obs.unsqueeze(1).expand(N, self._buffer_size, obs_dim).clone()
            self._buffers[agent_id] = buf
        self._head = 0
        self._initialized = True

    def _push(self, obs_dict: Dict[str, torch.Tensor]) -> None:
        """Push new observations into the ring buffer."""
        self._head = (self._head + 1) % self._buffer_size
        for agent_id, obs in obs_dict.items():
            self._buffers[agent_id][:, self._head, :] = obs

    def _reset_envs(
        self, obs_dict: Dict[str, torch.Tensor], reset_mask: torch.Tensor
    ) -> None:
        """Fill entire buffer with new obs for reset environments.

        Args:
            obs_dict: New observations after reset.
            reset_mask: Boolean mask [N] indicating which envs reset.
        """
        if not reset_mask.any():
            return
        reset_idx = reset_mask.nonzero(as_tuple=True)[0]
        for agent_id, obs in obs_dict.items():
            expanded = obs[reset_idx].unsqueeze(1).expand(
                -1, self._buffer_size, obs.shape[-1]
            )
            self._buffers[agent_id][reset_idx] = expanded

    def _read_stacked(self) -> Dict[str, torch.Tensor]:
        """Read stacked observations from the ring buffer.

        Returns oldest-first layout: [obs_{t-K*skip}, ..., obs_{t-skip}, obs_t].
        """
        # Indices from newest (head) backwards by frame_skip
        indices = [
            (self._head - i * self._frame_skip) % self._buffer_size
            for i in range(self._num_stack)
        ]
        # Reverse so oldest is first
        indices = indices[::-1]

        result = {}
        for agent_id, buf in self._buffers.items():
            frames = buf[:, indices, :]  # [N, num_stack, obs_dim]
            result[agent_id] = frames.reshape(frames.shape[0], -1)  # [N, num_stack * obs_dim]
        return result

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict]:
        obs_dict, info = self._env.reset()
        self._init_buffers(obs_dict)
        return self._read_stacked(), info

    def step(
        self, actions: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict, Dict, Dict, Dict]:
        obs_dict, rewards, terminated, truncated, infos = self._env.step(actions)

        if not self._initialized:
            self._init_buffers(obs_dict)
            return self._read_stacked(), rewards, terminated, truncated, infos

        # Push new observations
        self._push(obs_dict)

        # Handle per-env resets: in Isaac Lab auto-resetting envs, the obs
        # returned on termination is already the first obs of the new episode.
        # We fill the entire buffer with this obs to avoid temporal leakage.
        first_agent = self.possible_agents[0]
        term = terminated[first_agent]
        trunc = truncated[first_agent]
        reset_mask = (term | trunc).squeeze(-1)  # [N]
        self._reset_envs(obs_dict, reset_mask)

        return self._read_stacked(), rewards, terminated, truncated, infos

    def state(self) -> torch.Tensor:
        """Return stacked shared state (concatenation of all agents' stacked obs)."""
        if not self._initialized:
            return self._env.state()
        stacked = self._read_stacked()
        return torch.cat(
            [stacked[uid] for uid in self.possible_agents], dim=-1
        )  # [N, C * obs_dim * num_stack]

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

    def __getattr__(self, key: str) -> Any:
        """Delegate attribute access to wrapped environment."""
        return getattr(self._env, key)
