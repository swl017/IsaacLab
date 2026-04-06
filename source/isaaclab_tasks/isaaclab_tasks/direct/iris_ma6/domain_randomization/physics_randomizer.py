# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics property randomization for domain randomization.

This module provides randomization of physics properties including:
- Rigid body mass and inertia
- Physics materials (friction, restitution)

The randomization wraps Isaac Lab's existing physics manipulation APIs
and stores sampled values for logging and debugging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .domain_randomization_cfg import PhysicsRandomizationCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


class PhysicsRandomizer:
    """Physics property randomizer for mass and material properties.

    This class samples and stores physics randomization parameters, then
    applies them to assets using Isaac Lab's physics APIs.

    Example:
        ```python
        cfg = PhysicsRandomizationCfg()
        randomizer = PhysicsRandomizer(cfg, num_envs=256, device="cuda")

        # Sample new parameters for some environments
        randomizer.sample_mass_scales(env_ids=torch.arange(128))

        # Apply to asset (requires asset reference)
        randomizer.apply_mass_randomization(robot_asset, env_ids=torch.arange(128))
        ```
    """

    def __init__(
        self,
        cfg: PhysicsRandomizationCfg,
        num_envs: int,
        device: torch.device | str,
    ):
        """Initialize the physics randomizer.

        Args:
            cfg: Physics randomization configuration.
            num_envs: Number of parallel environments.
            device: PyTorch device for tensor allocation.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-allocate storage for sampled values
        # Mass randomization
        self.mass_scales = torch.ones(num_envs, device=self.device)
        self.mass_additions = torch.zeros(num_envs, device=self.device)
        self.payload_masses = torch.zeros(num_envs, device=self.device)

        # Material randomization
        self.static_friction = torch.ones(num_envs, device=self.device)
        self.dynamic_friction = torch.ones(num_envs, device=self.device)
        self.restitution = torch.zeros(num_envs, device=self.device)

        # Initialize with default (non-randomized) values
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize all parameters with default (non-randomized) values."""
        # Mass: no scaling, no additions
        self.mass_scales.fill_(1.0)
        self.mass_additions.zero_()
        self.payload_masses.zero_()

        # Materials: middle of ranges
        mass_cfg = self.cfg.mass
        mat_cfg = self.cfg.material

        if mat_cfg.enabled:
            self.static_friction.fill_(
                (mat_cfg.static_friction_range[0] + mat_cfg.static_friction_range[1]) / 2
            )
            self.dynamic_friction.fill_(
                (mat_cfg.dynamic_friction_range[0] + mat_cfg.dynamic_friction_range[1]) / 2
            )
            self.restitution.fill_(
                (mat_cfg.restitution_range[0] + mat_cfg.restitution_range[1]) / 2
            )

    def sample_mass_parameters(self, env_ids: torch.Tensor | None = None):
        """Sample mass randomization parameters for specified environments.

        Samples mass scales, additive masses, and payload masses according
        to the configuration. Does NOT apply to simulation - call
        apply_mass_randomization() separately.

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.mass.enabled:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        num_samples = len(env_ids)
        mass_cfg = self.cfg.mass

        # Sample based on distribution type
        if mass_cfg.distribution == "uniform":
            # Mass scale
            self.mass_scales[env_ids] = torch.empty(
                num_samples, device=self.device
            ).uniform_(mass_cfg.body_mass_scale_range[0], mass_cfg.body_mass_scale_range[1])

            # Mass addition
            self.mass_additions[env_ids] = torch.empty(
                num_samples, device=self.device
            ).uniform_(mass_cfg.body_mass_add_range[0], mass_cfg.body_mass_add_range[1])

            # Payload mass
            self.payload_masses[env_ids] = torch.empty(
                num_samples, device=self.device
            ).uniform_(mass_cfg.payload_mass_range[0], mass_cfg.payload_mass_range[1])

        elif mass_cfg.distribution == "log_uniform":
            # Log-uniform sampling for scale (good for ratios)
            log_min = torch.log(torch.tensor(mass_cfg.body_mass_scale_range[0]))
            log_max = torch.log(torch.tensor(mass_cfg.body_mass_scale_range[1]))
            log_samples = torch.empty(num_samples, device=self.device).uniform_(
                log_min.item(), log_max.item()
            )
            self.mass_scales[env_ids] = torch.exp(log_samples)

            # Uniform for additions and payload
            self.mass_additions[env_ids] = torch.empty(
                num_samples, device=self.device
            ).uniform_(mass_cfg.body_mass_add_range[0], mass_cfg.body_mass_add_range[1])

            self.payload_masses[env_ids] = torch.empty(
                num_samples, device=self.device
            ).uniform_(mass_cfg.payload_mass_range[0], mass_cfg.payload_mass_range[1])

        elif mass_cfg.distribution == "gaussian":
            # Gaussian sampling (mean at center, std covers range)
            mean_scale = (
                mass_cfg.body_mass_scale_range[0] + mass_cfg.body_mass_scale_range[1]
            ) / 2
            std_scale = (
                mass_cfg.body_mass_scale_range[1] - mass_cfg.body_mass_scale_range[0]
            ) / 4  # ~95% within range

            self.mass_scales[env_ids] = torch.empty(
                num_samples, device=self.device
            ).normal_(mean_scale, std_scale).clamp(
                mass_cfg.body_mass_scale_range[0], mass_cfg.body_mass_scale_range[1]
            )

            mean_add = (
                mass_cfg.body_mass_add_range[0] + mass_cfg.body_mass_add_range[1]
            ) / 2
            std_add = (
                mass_cfg.body_mass_add_range[1] - mass_cfg.body_mass_add_range[0]
            ) / 4

            self.mass_additions[env_ids] = torch.empty(
                num_samples, device=self.device
            ).normal_(mean_add, std_add).clamp(
                mass_cfg.body_mass_add_range[0], mass_cfg.body_mass_add_range[1]
            )

            mean_payload = (
                mass_cfg.payload_mass_range[0] + mass_cfg.payload_mass_range[1]
            ) / 2
            std_payload = (
                mass_cfg.payload_mass_range[1] - mass_cfg.payload_mass_range[0]
            ) / 4

            self.payload_masses[env_ids] = torch.empty(
                num_samples, device=self.device
            ).normal_(mean_payload, std_payload).clamp(
                mass_cfg.payload_mass_range[0], mass_cfg.payload_mass_range[1]
            )

    def sample_material_parameters(self, env_ids: torch.Tensor | None = None):
        """Sample material randomization parameters for specified environments.

        Samples friction and restitution values according to configuration.
        Does NOT apply to simulation - call apply_material_randomization() separately.

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.material.enabled:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        num_samples = len(env_ids)
        mat_cfg = self.cfg.material

        # Sample static friction
        self.static_friction[env_ids] = torch.empty(
            num_samples, device=self.device
        ).uniform_(mat_cfg.static_friction_range[0], mat_cfg.static_friction_range[1])

        # Sample dynamic friction
        self.dynamic_friction[env_ids] = torch.empty(
            num_samples, device=self.device
        ).uniform_(mat_cfg.dynamic_friction_range[0], mat_cfg.dynamic_friction_range[1])

        # Sample restitution
        self.restitution[env_ids] = torch.empty(
            num_samples, device=self.device
        ).uniform_(mat_cfg.restitution_range[0], mat_cfg.restitution_range[1])

        # Ensure physical consistency: dynamic <= static
        if mat_cfg.make_consistent:
            self.dynamic_friction[env_ids] = torch.minimum(
                self.dynamic_friction[env_ids], self.static_friction[env_ids]
            )

    def apply_mass_randomization(
        self,
        asset: Articulation | RigidObject,
        env_ids: torch.Tensor | None = None,
        body_ids: list[int] | None = None,
    ):
        """Apply sampled mass randomization to an asset.

        Uses the previously sampled mass_scales, mass_additions, and payload_masses
        to modify the asset's body masses.

        Args:
            asset: The Isaac Lab asset to modify.
            env_ids: Environment indices to apply. If None, applies to all.
            body_ids: Specific body indices to modify. If None, modifies all bodies.

        Note:
            This requires an asset reference and modifies simulation state.
            Call sample_mass_parameters() before this to set values.
        """
        if not self.cfg.mass.enabled:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        # Resolve body indices
        if body_ids is None:
            body_ids = slice(None)

        # env_ids must be on CPU for PhysX view indexing
        env_ids_cpu = env_ids.cpu()

        # Get full mass tensor and reset to defaults for target envs
        masses = asset.root_physx_view.get_masses()  # (num_envs, num_bodies)
        default_masses = asset.data.default_mass[env_ids_cpu, body_ids].clone()
        masses[env_ids_cpu, body_ids] = default_masses

        # Apply scale
        scales = self.mass_scales[env_ids].unsqueeze(-1).cpu()  # (M, 1)
        new_masses = default_masses * scales

        # Apply additive mass (distributed across bodies)
        additions = self.mass_additions[env_ids].unsqueeze(-1).cpu()
        num_bodies = new_masses.shape[-1] if new_masses.dim() > 1 else 1
        new_masses = new_masses + additions / num_bodies

        # Apply payload mass (add to base link, index 0)
        payloads = self.payload_masses[env_ids].cpu()
        if new_masses.dim() > 1:
            new_masses[:, 0] = new_masses[:, 0] + payloads
        else:
            new_masses = new_masses + payloads

        # Ensure positive masses
        new_masses = new_masses.clamp(min=0.001)

        # Write to simulation via PhysX view API
        masses[env_ids_cpu, body_ids] = new_masses
        asset.root_physx_view.set_masses(masses, env_ids_cpu)

        # Optionally recompute inertia
        if self.cfg.mass.recompute_inertia:
            inertias = asset.root_physx_view.get_inertias()  # (num_envs, num_bodies, 9)
            default_inertias = asset.data.default_inertia[env_ids_cpu, body_ids].clone()
            mass_ratios = new_masses / default_masses.clamp(min=0.001)
            if default_inertias.dim() == 3:
                mass_ratios = mass_ratios.unsqueeze(-1)  # (M, num_bodies, 1) for broadcasting
            new_inertias = default_inertias * mass_ratios
            inertias[env_ids_cpu, body_ids] = new_inertias
            asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    def apply_material_randomization(
        self,
        asset: Articulation | RigidObject,
        env_ids: torch.Tensor | None = None,
    ):
        """Apply sampled material randomization to an asset.

        Uses the previously sampled friction and restitution values to
        modify the asset's physics materials.

        Args:
            asset: The Isaac Lab asset to modify.
            env_ids: Environment indices to apply. If None, applies to all.

        Note:
            Material randomization in Isaac Sim is more complex and may require
            using the Replicator API or direct USD manipulation. This method
            provides a simplified interface.
        """
        if not self.cfg.material.enabled:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        # Note: Direct material modification is complex in Isaac Sim
        # For full implementation, use isaaclab.envs.mdp.events.randomize_rigid_body_material
        # which handles material bucket creation and assignment

        # For now, store values - actual application handled by environment
        pass

    def get_mass_scales(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get current mass scales for specified environments.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Mass scales tensor.
        """
        if env_ids is None:
            return self.mass_scales.clone()
        return self.mass_scales[env_ids].clone()

    def get_payload_masses(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get current payload masses for specified environments.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Payload masses tensor in kg.
        """
        if env_ids is None:
            return self.payload_masses.clone()
        return self.payload_masses[env_ids].clone()

    def get_friction_coefficients(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current friction coefficients for specified environments.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Tuple of (static_friction, dynamic_friction) tensors.
        """
        if env_ids is None:
            return self.static_friction.clone(), self.dynamic_friction.clone()
        return (
            self.static_friction[env_ids].clone(),
            self.dynamic_friction[env_ids].clone(),
        )

    def get_restitution(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get current restitution coefficients for specified environments.

        Args:
            env_ids: Environment indices. If None, returns all.

        Returns:
            Restitution coefficients tensor.
        """
        if env_ids is None:
            return self.restitution.clone()
        return self.restitution[env_ids].clone()
