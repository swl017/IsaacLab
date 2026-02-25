# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derived field computation for computing fields from delayed raw sensor data."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Callable


@dataclass
class DerivedFieldDef:
    """Definition for a derived field.

    A derived field is computed from one or more source fields. Derived fields
    are computed from delayed raw fields (not delayed separately), ensuring
    consistency between related fields.

    Example:
        # Define a derived field for camera position
        camera_pos_def = DerivedFieldDef(
            name="camera_position_w",
            sources=["body_position_w", "body_orientation_w", "camera_offset_b"],
            compute_fn=compute_camera_position,
            dependency_level=1,
        )
    """

    name: str
    """Name of the derived field."""

    sources: list[str]
    """List of source field names this depends on."""

    compute_fn: Callable[..., torch.Tensor]
    """Function to compute the derived field.

    The function receives source fields as keyword arguments matching the source names.
    It should return a tensor of shape (num_envs, ...).

    Example:
        def compute_velocity_magnitude(linear_velocity_b: torch.Tensor) -> torch.Tensor:
            return torch.norm(linear_velocity_b, dim=-1, keepdim=True)
    """

    dependency_level: int = 1
    """Dependency level for ordering computation.

    - Level 0: Raw sensor fields (not derived)
    - Level 1: Derived from raw fields only
    - Level 2: Derived from level 1 fields
    - etc.

    Fields are computed in order of increasing dependency level.
    """


class DerivedFieldComputer:
    """Computes derived fields from delayed raw sensor data.

    This class manages a collection of derived field definitions and computes
    them in the correct dependency order. Derived fields are computed from
    delayed raw fields, ensuring that computed values are consistent with
    the delayed sensor state.

    Key insight: Derived fields should be computed FROM delayed raw fields,
    not delayed separately. For example, camera position should be computed
    from the delayed body position and orientation, not by delaying the
    camera position independently.

    Example usage:
        # Define derived fields
        velocity_mag_def = DerivedFieldDef(
            name="velocity_magnitude",
            sources=["linear_velocity_b"],
            compute_fn=lambda linear_velocity_b: torch.norm(linear_velocity_b, dim=-1, keepdim=True),
            dependency_level=1,
        )

        # Create computer
        computer = DerivedFieldComputer([velocity_mag_def])

        # Compute derived fields from delayed raw fields
        delayed_fields = {"linear_velocity_b": delayed_vel}
        all_fields = computer.compute_all(delayed_fields)
        velocity_mag = all_fields["velocity_magnitude"]
    """

    def __init__(self, field_definitions: list[DerivedFieldDef] | None = None):
        """Initialize the derived field computer.

        Args:
            field_definitions: List of derived field definitions. Can be None
                if fields will be registered later.
        """
        self.definitions: dict[str, DerivedFieldDef] = {}
        if field_definitions:
            for field_def in field_definitions:
                self.register(field_def)

    def register(self, field_def: DerivedFieldDef):
        """Register a derived field definition.

        Args:
            field_def: The derived field definition to register.
        """
        self.definitions[field_def.name] = field_def

    def unregister(self, name: str):
        """Unregister a derived field.

        Args:
            name: Name of the field to unregister.
        """
        if name in self.definitions:
            del self.definitions[name]

    def compute_all(self, raw_fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute all derived fields from raw (delayed) fields.

        Fields are computed in order of increasing dependency level to ensure
        that dependencies are available when needed.

        Args:
            raw_fields: Dictionary of delayed raw sensor fields.
                Keys are field names, values are tensors.

        Returns:
            Dictionary containing both raw fields and computed derived fields.
        """
        # Start with a copy of raw fields
        result = dict(raw_fields)

        # Sort definitions by dependency level
        sorted_defs = sorted(self.definitions.values(), key=lambda x: x.dependency_level)

        # Compute each derived field in order
        for field_def in sorted_defs:
            # Gather source fields
            sources = {}
            all_sources_available = True
            for source_name in field_def.sources:
                if source_name in result:
                    sources[source_name] = result[source_name]
                else:
                    all_sources_available = False
                    break

            # Compute derived field if all sources are available
            if all_sources_available:
                result[field_def.name] = field_def.compute_fn(**sources)

        return result

    def compute_single(
        self,
        field_name: str,
        raw_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Compute a single derived field.

        Note: This does not compute intermediate dependencies. Use compute_all()
        if the field depends on other derived fields.

        Args:
            field_name: Name of the derived field to compute.
            raw_fields: Dictionary of available fields (raw and derived).

        Returns:
            Computed tensor, or None if sources are not available.
        """
        if field_name not in self.definitions:
            return None

        field_def = self.definitions[field_name]

        # Gather source fields
        sources = {}
        for source_name in field_def.sources:
            if source_name not in raw_fields:
                return None
            sources[source_name] = raw_fields[source_name]

        return field_def.compute_fn(**sources)

    def get_dependency_order(self) -> list[str]:
        """Get the order in which fields should be computed.

        Returns:
            List of field names in computation order.
        """
        sorted_defs = sorted(self.definitions.values(), key=lambda x: x.dependency_level)
        return [d.name for d in sorted_defs]

    def get_sources_for_field(self, field_name: str) -> list[str] | None:
        """Get the source field names for a derived field.

        Args:
            field_name: Name of the derived field.

        Returns:
            List of source field names, or None if field is not registered.
        """
        if field_name not in self.definitions:
            return None
        return self.definitions[field_name].sources.copy()


# ==============================================================================
# Example derived field computation functions
# ==============================================================================


def compute_velocity_magnitude(linear_velocity_b: torch.Tensor) -> torch.Tensor:
    """Compute velocity magnitude from velocity vector.

    Args:
        linear_velocity_b: Linear velocity in body frame, shape (num_envs, 3).

    Returns:
        Velocity magnitude, shape (num_envs, 1).
    """
    return torch.norm(linear_velocity_b, dim=-1, keepdim=True)


def compute_speed_horizontal(linear_velocity_b: torch.Tensor) -> torch.Tensor:
    """Compute horizontal speed from velocity vector.

    Args:
        linear_velocity_b: Linear velocity in body frame, shape (num_envs, 3).
            Assumes XY are horizontal, Z is vertical.

    Returns:
        Horizontal speed, shape (num_envs, 1).
    """
    return torch.norm(linear_velocity_b[..., :2], dim=-1, keepdim=True)


def compute_altitude_rate(linear_velocity_b: torch.Tensor) -> torch.Tensor:
    """Compute altitude rate (vertical velocity) from velocity vector.

    Args:
        linear_velocity_b: Linear velocity in body frame, shape (num_envs, 3).
            Assumes Z is vertical.

    Returns:
        Altitude rate, shape (num_envs, 1).
    """
    return linear_velocity_b[..., 2:3]


# ==============================================================================
# Predefined field definitions
# ==============================================================================


# Common derived fields for quadcopter
QUADCOPTER_DERIVED_FIELDS = [
    DerivedFieldDef(
        name="velocity_magnitude",
        sources=["linear_velocity_b"],
        compute_fn=compute_velocity_magnitude,
        dependency_level=1,
    ),
    DerivedFieldDef(
        name="speed_horizontal",
        sources=["linear_velocity_b"],
        compute_fn=compute_speed_horizontal,
        dependency_level=1,
    ),
    DerivedFieldDef(
        name="altitude_rate",
        sources=["linear_velocity_b"],
        compute_fn=compute_altitude_rate,
        dependency_level=1,
    ),
]
