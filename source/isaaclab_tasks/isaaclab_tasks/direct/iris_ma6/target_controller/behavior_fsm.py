# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finite State Machine for attacker behavior management.

Implements the FSM states and transitions from iris_ma6_env_spec.md Section 4.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .target_controller_cfg import TargetControllerCfg


class FSMState(IntEnum):
    """Attacker finite state machine states.

    State Transitions:
        APPROACH -> EVADE: When interceptor in INTERCEPT mode is within evade_trigger_distance
        EVADE -> APPROACH: When evasion_timer expires
        APPROACH/EVADE -> DEAD: When intercepted (distance < capture_distance)
        APPROACH -> BREACH: When target reaches facility (distance < facility_radius)
    """

    APPROACH = 0  # Moving toward facility
    EVADE = 1  # Evading interceptor
    DEAD = 2  # Intercepted (deactivated)
    BREACH = 3  # Reached facility (episode failure)


class VelocityMode(IntEnum):
    """Velocity generation mode enumeration.

    Used for non-attacker modes (iris_ma5 compatibility).
    """

    LINEAR = 0  # Random direction changes
    CIRCULAR = 1  # Orbital flight
    APPROACH = 2  # Goal-directed approach (attacker)


class BehaviorFSM:
    """Finite State Machine for attacker behavior management.

    Manages FSM states, transitions, and velocity mode selection
    for target movement.

    Features:
    - State tracking for each target
    - Transition condition checking
    - Velocity mode mapping based on state
    - Alive/dead status management
    """

    def __init__(
        self,
        cfg: TargetControllerCfg,
        num_envs: int,
        num_targets: int,
        device: torch.device,
    ):
        """Initialize the behavior FSM.

        Args:
            cfg: Target controller configuration.
            num_envs: Number of parallel environments.
            num_targets: Maximum number of targets per environment.
            device: Torch device (cuda or cpu).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_targets = num_targets
        self.device = device
        self.total_targets = num_envs * num_targets

        # FSM state for each target [N_env, N_target]
        self.fsm_state = torch.zeros(
            num_envs, num_targets, dtype=torch.long, device=device
        )

        # Alive status [N_env, N_target]
        self.alive = torch.ones(num_envs, num_targets, dtype=torch.bool, device=device)

        # Velocity mode for each target [N_env, N_target]
        # Used for non-attacker modes (linear/circular)
        self.velocity_mode = torch.zeros(
            num_envs, num_targets, dtype=torch.long, device=device
        )

        # Behavior profile index [N_env, N_target]
        self.behavior_profile = torch.zeros(
            num_envs, num_targets, dtype=torch.long, device=device
        )

    def update(
        self,
        current_position: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
        facility_position: torch.Tensor,
        facility_radius: float,
        evasion_agility: torch.Tensor,
        evasion_timer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update FSM states based on current conditions.

        Args:
            current_position: [N_env, N_target, 3] target positions.
            interceptor_positions: [N_env, num_defenders, 3] defender positions.
            interceptor_roles: [N_env, num_defenders] defender roles (0=OBSERVE, 1=INTERCEPT).
            facility_position: [N_env, 3] facility positions.
            facility_radius: Facility radius for breach detection [m].
            evasion_agility: [N_env * N_target] evasion agility values.
            evasion_timer: [N_env * N_target] remaining evasion time.

        Returns:
            transition_to_evade: Flat indices of targets transitioning to EVADE.
            transition_to_approach: Flat indices of targets transitioning to APPROACH.
            breach_indices: Flat indices of targets that breached the facility.
        """
        # Flatten for easier processing
        pos_flat = current_position.view(-1, 3)
        alive_flat = self.alive.view(-1)
        fsm_flat = self.fsm_state.view(-1)

        # Expand facility position to per-target
        facility_expanded = (
            facility_position.unsqueeze(1)
            .expand(-1, self.num_targets, -1)
            .reshape(-1, 3)
        )

        # Get alive indices
        alive_indices = torch.where(alive_flat)[0]

        if len(alive_indices) == 0:
            return (
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.tensor([], dtype=torch.long, device=self.device),
            )

        # -----------------------------------------------------------------
        # Check APPROACH -> EVADE transitions (only in attacker mode)
        # -----------------------------------------------------------------
        approach_mask = fsm_flat[alive_indices] == FSMState.APPROACH
        approach_indices = alive_indices[approach_mask]

        transition_to_evade = torch.tensor([], dtype=torch.long, device=self.device)

        # Skip evasion transitions when attacker mode is disabled
        if self.cfg.use_attacker_mode and len(approach_indices) > 0:
            # Check interceptor proximity
            dist_to_nearest = self._compute_nearest_interceptor_distance(
                approach_indices, pos_flat, interceptor_positions, interceptor_roles
            )

            # Check if evasion should trigger (agility > 0)
            agility = evasion_agility[approach_indices]
            should_evade = (
                (dist_to_nearest < self.cfg.evade_trigger_distance) & (agility > 0)
            )

            if should_evade.any():
                transition_to_evade = approach_indices[should_evade]
                self._set_state(transition_to_evade, FSMState.EVADE)

        # -----------------------------------------------------------------
        # Check EVADE -> APPROACH transitions (evasion complete)
        # -----------------------------------------------------------------
        evade_mask = fsm_flat[alive_indices] == FSMState.EVADE
        evade_indices = alive_indices[evade_mask]

        transition_to_approach = torch.tensor([], dtype=torch.long, device=self.device)

        if len(evade_indices) > 0:
            evasion_complete = evasion_timer[evade_indices] <= 0

            if evasion_complete.any():
                transition_to_approach = evade_indices[evasion_complete]
                self._set_state(transition_to_approach, FSMState.APPROACH)

        # -----------------------------------------------------------------
        # Check APPROACH/EVADE -> BREACH transitions
        # -----------------------------------------------------------------
        active_mask = (fsm_flat[alive_indices] == FSMState.APPROACH) | (
            fsm_flat[alive_indices] == FSMState.EVADE
        )
        active_indices = alive_indices[active_mask]

        breach_indices = torch.tensor([], dtype=torch.long, device=self.device)

        if len(active_indices) > 0:
            dist_to_facility = torch.norm(
                pos_flat[active_indices] - facility_expanded[active_indices], dim=1
            )
            breached = dist_to_facility < facility_radius

            if breached.any():
                breach_indices = active_indices[breached]
                self._set_state(breach_indices, FSMState.BREACH)
                self._set_dead(breach_indices)

        return transition_to_evade, transition_to_approach, breach_indices

    def check_intercept(
        self,
        current_position: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
        capture_distance: float,
    ) -> torch.Tensor:
        """Check for intercept events (target captured by interceptor).

        Args:
            current_position: [N_env, N_target, 3] target positions.
            interceptor_positions: [N_env, num_defenders, 3] defender positions.
            interceptor_roles: [N_env, num_defenders] defender roles.
            capture_distance: Distance threshold for capture [m].

        Returns:
            intercepted_indices: Flat indices of intercepted targets.
        """
        pos_flat = current_position.view(-1, 3)
        alive_flat = self.alive.view(-1)
        fsm_flat = self.fsm_state.view(-1)

        # Only check alive, non-DEAD, non-BREACH targets
        active_mask = alive_flat & (fsm_flat != FSMState.DEAD) & (fsm_flat != FSMState.BREACH)
        active_indices = torch.where(active_mask)[0]

        if len(active_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        # Check distance to nearest INTERCEPT-mode defender
        dist_to_nearest = self._compute_nearest_interceptor_distance(
            active_indices, pos_flat, interceptor_positions, interceptor_roles
        )

        intercepted = dist_to_nearest < capture_distance

        if intercepted.any():
            intercepted_indices = active_indices[intercepted]
            self._set_state(intercepted_indices, FSMState.DEAD)
            self._set_dead(intercepted_indices)
            return intercepted_indices

        return torch.tensor([], dtype=torch.long, device=self.device)

    def _compute_nearest_interceptor_distance(
        self,
        target_indices: torch.Tensor,
        pos_flat: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance to nearest INTERCEPT-mode defender.

        Args:
            target_indices: Flat indices for targets.
            pos_flat: [N_total, 3] all target positions.
            interceptor_positions: [N_env, num_defenders, 3] defender positions.
            interceptor_roles: [N_env, num_defenders] defender roles.

        Returns:
            [N] distances to nearest interceptor.
        """
        n = len(target_indices)

        # Convert flat indices to env_ids
        env_ids = target_indices // self.num_targets

        # Get interceptor info for each target's environment
        target_interceptor_pos = interceptor_positions[env_ids]  # [N, num_defenders, 3]
        target_interceptor_roles = interceptor_roles[env_ids]  # [N, num_defenders]

        # Compute distance to each interceptor
        target_pos = pos_flat[target_indices].unsqueeze(1)  # [N, 1, 3]
        dist_to_interceptors = torch.norm(
            target_pos - target_interceptor_pos, dim=-1
        )  # [N, num_defenders]

        # Mask out non-INTERCEPT defenders with large distance
        intercept_mask = target_interceptor_roles == 1
        dist_to_interceptors[~intercept_mask] = 1e6

        # Find nearest
        nearest_dist, _ = dist_to_interceptors.min(dim=1)

        return nearest_dist

    def _set_state(self, flat_indices: torch.Tensor, state: FSMState):
        """Set FSM state for specified flat indices.

        Args:
            flat_indices: Flat indices to update.
            state: New FSM state.
        """
        env_ids = flat_indices // self.num_targets
        target_ids = flat_indices % self.num_targets
        self.fsm_state[env_ids, target_ids] = state

    def _set_dead(self, flat_indices: torch.Tensor):
        """Mark targets as dead (not alive).

        Args:
            flat_indices: Flat indices to mark as dead.
        """
        env_ids = flat_indices // self.num_targets
        target_ids = flat_indices % self.num_targets
        self.alive[env_ids, target_ids] = False

    def reset(self, env_ids: torch.Tensor):
        """Reset FSM state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if len(env_ids) == 0:
            return

        # Reset FSM state to APPROACH
        self.fsm_state[env_ids] = FSMState.APPROACH

        # Reset alive status
        self.alive[env_ids] = True

        # Reset velocity mode based on configuration
        if self.cfg.use_attacker_mode:
            self.velocity_mode[env_ids] = VelocityMode.APPROACH
        else:
            # Randomly assign linear or circular mode (iris_ma5 compatibility)
            mode_probs = torch.rand(len(env_ids), self.num_targets, device=self.device)
            self.velocity_mode[env_ids] = (
                (mode_probs > self.cfg.linear_weight).long()
            )  # 0=LINEAR, 1=CIRCULAR

    def get_alive_indices(self) -> torch.Tensor:
        """Get flat indices of all alive targets.

        Returns:
            Flat indices of alive targets.
        """
        return torch.where(self.alive.view(-1))[0]

    def get_state_indices(self, state: FSMState) -> torch.Tensor:
        """Get flat indices of targets in a specific state.

        Args:
            state: FSM state to filter by.

        Returns:
            Flat indices of targets in the specified state.
        """
        alive_flat = self.alive.view(-1)
        fsm_flat = self.fsm_state.view(-1)
        return torch.where(alive_flat & (fsm_flat == state))[0]

    def get_num_alive(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Get number of alive targets per environment.

        Args:
            env_ids: Environment indices (optional, defaults to all).

        Returns:
            [N_env] tensor of alive counts.
        """
        if env_ids is None:
            return self.alive.sum(dim=1)
        return self.alive[env_ids].sum(dim=1)

    def get_num_in_state(self, state: FSMState, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Get number of targets in a specific state per environment.

        Args:
            state: FSM state to count.
            env_ids: Environment indices (optional, defaults to all).

        Returns:
            [N_env] tensor of state counts.
        """
        state_mask = (self.fsm_state == state) & self.alive
        if env_ids is None:
            return state_mask.sum(dim=1)
        return state_mask[env_ids].sum(dim=1)
