# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Iris Multi-Agent environment (v4)."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import IRIS_GIMBAL2_CFG

from isaaclab_tasks.direct.iris_ma3 import bbox_raycaster

from .controller import PointMassCfg, GimbalStabilizerCfg
from .curriculum import CurriculumCfg
from .target_movement import TargetMovementCfg


@configclass
class IrisMAEnvCfg(DirectMARLEnvCfg):
    """Configuration for the Iris Multi-Agent environment.

    This environment supports multi-agent drone coordination for target tracking
    with gimbal-stabilized cameras and triangulation-based localization.
    """

    # ==========================================================================
    # Environment Meta
    # ==========================================================================

    episode_length_s: float = 20.0
    """Episode length in seconds."""

    decimation: int = 4
    """Physics steps per control step."""

    possible_agents: list[str] = ["drone_0", "drone_1"]
    """List of agent identifiers."""

    action_spaces: dict = {
        "drone_0": 7,  # [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        "drone_1": 7,
    }
    """Action space dimensions per agent."""

    observation_spaces: dict = {
        "drone_0": 47,
        "drone_1": 47,
    }
    """Observation space dimensions per agent."""

    state_space: int = -1
    """State space dimension. -1 means concatenate all observations."""

    debug_vis: bool = True
    """Enable debug visualization."""

    # ==========================================================================
    # Simulation
    # ==========================================================================

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ==========================================================================
    # Assets
    # ==========================================================================

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/{robot_name}/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={
            "class:target": (255, 36, 66, 255),
            "class:robot": (255, 36, 255, 255),
        },
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.7071068, 0, 0, -0.7071068),
            convention="world",
        ),
    )
    """Camera configuration template (applied to each agent)."""

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_body.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                rigid_body_enabled=True,
            ),
            copy_from_source=False,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 3.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    """Target rigid body configuration."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(50.0, 50.0, 50.0),
        lookat=(0.0, 0.0, 0.0),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=25.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/{robot_name}")
    """Robot articulation configuration template."""

    # ==========================================================================
    # Submodule Configs
    # ==========================================================================

    point_mass: PointMassCfg = PointMassCfg()
    """Point mass controller configuration."""

    gimbal: GimbalStabilizerCfg = GimbalStabilizerCfg()
    """Gimbal stabilizer configuration (single source of truth for gimbal limits)."""

    curriculum: CurriculumCfg = CurriculumCfg()
    """Training curriculum configuration."""

    target_movement: TargetMovementCfg = TargetMovementCfg()
    """Target movement configuration for randomized target motion."""

    bbox_raycaster: bbox_raycaster.BBoxRayCasterCfg = bbox_raycaster.BBoxRayCasterCfg(
        target_prim_paths=[],
        mesh_prim_paths=["/World/ground"],
        num_cameras_per_env=2,
        num_cameras_per_agent=1,
        load_agent_meshes=False,
        agent_mesh_simplification=1.1,
        use_collision_proxy=False,
        min_bbox_size=(0.01, 0.01),
        max_bbox_size=(0.90, 0.90),
        partial_detection_allowed=False,
        min_bbox_area_pixels=400.0,
        enable_occlusion_check=False,
        occlusion_ray_pattern="9point",
        occlusion_visibility_threshold=0.5,
        occlusion_ray_tolerance=1.1,
        max_distance=100.0,
        debug_vis=False,
        debug_memory=False,
    )
    """Bounding box raycaster configuration."""

    # ==========================================================================
    # Motion Limits
    # ==========================================================================

    max_lin_vel: float = 10.0
    """Maximum linear velocity (m/s). Reduced from 20.0 for smoother actions."""

    max_yaw_rate: float = math.radians(90.0)
    """Maximum yaw rate (rad/s). Reduced from 180° for smoother rotations."""

    max_zoom_rate: float = 2.0
    """Maximum zoom rate (zoom levels per second)."""

    max_zoom_level: float = 10.0
    """Maximum zoom level."""

    # ==========================================================================
    # Reward Scales
    # ==========================================================================

    lin_vel_penalty_scale: float = -0.1
    """Penalty scale for linear velocity."""

    action_sum_penalty_scale: float = -2.0
    """Penalty scale for total action magnitude. Increased from -1.0 for smoother actions."""

    # [0 vx, 1 vy, 2 vz, 3 yaw_rate, 4 gimbal_yaw_rate, 5 gimbal_pitch_rate, 6 zoom_rate]
    action_weight: list = [1, 1, 5, 1, 0.5, 0.5, 0.3]
    """Weights for each action dimension in penalty computation."""

    action_delta_weight: list = [1, 1, 1, 1, 0.5, 0.5, 0.3]
    """Weights for action delta (smoothness) penalty."""

    action_delta_penalty_scale: float = -0.05
    """Penalty scale for action changes (smoothness). Increased from -0.05 for less jerky actions."""

    # Single-agent tracking rewards
    bbox_center_reward_scale: float = 60.0
    """Reward scale for centering target in image."""

    bbox_size_reward_scale: float = 60.0
    """Reward scale for appropriate bbox size."""

    # Multi-agent coordination rewards
    triangulation_reward_scale: float = 5.0
    """Reward scale for triangulation quality."""

    collision_penalty_scale: float = -100.0
    """Penalty scale for inter-agent collisions."""

    collision_min_safe_distance: float = 2.0
    """Minimum safe distance between agents (m)."""

    ttc_penalty_scale: float = -10.0
    """Penalty scale for time-to-collision risk."""

    ttc_horizon: float = 5.0
    """Time horizon for TTC computation (s)."""

    # ==========================================================================
    # Noise and Uncertainty Parameters
    # ==========================================================================

    pix_std: float = 7.0
    """Pixel noise standard deviation for bbox detection."""

    pos_std: float = 0.1
    """Position noise standard deviation (m)."""

    ori_std: float = 0.001
    """Orientation noise standard deviation (rad)."""

    gimbal_std: float = 0.001
    """Gimbal angle noise standard deviation (rad)."""

    intrinsics_std: float = 10.0
    """Camera intrinsics noise standard deviation."""

    # ==========================================================================
    # Delay and Communication System
    # ==========================================================================

    enable_noise_in_observations: bool = True
    """Whether to add noise to observations."""

    # Dynamics lag
    motion_time_constant: float = 0.001
    """Motion filtering time constant (s)."""

    gimbal_time_constant: float = 0.001
    """Gimbal filtering time constant (s)."""

    # Detection
    detection_fps_mean: float = 20.0
    """Detection frame rate (Hz)."""

    detection_fps_std: float = 1.0
    """Standard deviation of detection frame rate (Hz)."""

    detection_mean_latency: float = 0.1
    """Mean detection latency (s)."""

    detection_std_latency: float = 0.08
    """Standard deviation of detection latency (s)."""

    detection_failure_rate: float = 0.05
    """Detection dropout/failure rate."""

    # Communication
    comm_mean_latency: float = 0.1
    """Mean inter-agent communication latency (s)."""

    comm_std_latency: float = 0.08
    """Standard deviation of communication latency (s)."""

    comm_dropout_rate: float = 0.05
    """Communication dropout rate."""

    # Reward decay
    detection_decay_time_constant: float = 1.0
    """Time constant for detection confidence decay (s)."""

    # ==========================================================================
    # Debug
    # ==========================================================================

    play_sim_at_step: int = 100000#200000
    """Training step at which to enable rendering for debugging."""
