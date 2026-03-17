# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization module for iris_ma6.

This module provides comprehensive domain randomization for sim-to-real transfer:

- **Physics Randomization**: Mass, friction, restitution variations
- **Camera Randomization**: Computational FOV/resolution simulation
- **Gimbal Randomization**: Joint offset and dynamics variations

Scope Distinction:
    - Domain Randomization (this module): *What* the system is
      (mass, friction, camera FOV, sensor characteristics)
    - Initial States (separate module): *Where* the system starts
      (position, velocity, orientation)

Example:
    ```python
    from isaaclab_tasks.direct.iris_ma6.domain_randomization import (
        DomainRandomizationCfg,
        DomainRandomizer,
    )

    # Create configuration
    cfg = DomainRandomizationCfg(
        enabled=True,
        camera=CameraRandomizationCfg(
            fov_scale_range=(0.6, 1.0),
            focal_length_range=(800.0, 1200.0),
        ),
        physics=PhysicsRandomizationCfg(
            mass=MassRandomizationCfg(
                body_mass_scale_range=(0.9, 1.1),
            ),
        ),
    )

    # Create randomizer
    randomizer = DomainRandomizer(
        cfg=cfg,
        num_envs=256,
        num_agents=3,
        device=torch.device("cuda"),
    )

    # Randomize at episode reset
    randomizer.randomize_all(env_ids=torch.arange(128))

    # Process camera images
    processed, intrinsics = randomizer.process_images(images, output_size=(360, 640))

    # Get gimbal offsets for control
    offsets = randomizer.get_gimbal_offsets()
    ```
"""

# Configuration classes
from .domain_randomization_cfg import (
    CameraRandomizationCfg,
    DomainRandomizationCfg,
    GimbalRandomizationCfg,
    MassRandomizationCfg,
    MaterialRandomizationCfg,
    MountOffsetRandomizationCfg,
    PhysicsRandomizationCfg,
    ScaleRandomizationCfg,
)

# Component classes
from .camera_processor import CameraProcessor
from .gimbal_randomizer import GimbalRandomizer
from .physics_randomizer import PhysicsRandomizer

# Main orchestrator
from .domain_randomizer import DomainRandomizer

__all__ = [
    # Configurations
    "DomainRandomizationCfg",
    "PhysicsRandomizationCfg",
    "MassRandomizationCfg",
    "ScaleRandomizationCfg",
    "MaterialRandomizationCfg",
    "CameraRandomizationCfg",
    "GimbalRandomizationCfg",
    "MountOffsetRandomizationCfg",
    # Components
    "CameraProcessor",
    "PhysicsRandomizer",
    "GimbalRandomizer",
    # Main class
    "DomainRandomizer",
]
