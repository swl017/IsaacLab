# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Iris MA6 Test environment."""

from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import IRIS_GIMBAL3_CFG

from .bbox_raycaster_v2 import BBoxRayCasterV2Cfg
from .cbf_safety import CBFManagerCfg
from .controller import DroneControllerCfg
from .controller.gain_randomization_cfg import GainRandomizationCfg
from .controller.tuning import TUNED_CONTROLLER_CFG
from .curriculum import CurriculumCfg
from .delay_system_v3 import (
    MultiAgentDelayCfgV3,
    DelaySystemKeyParams,
    create_delay_cfg_from_params,
)
from .initial_states import InitialStatesCfg
from .target_controller import TargetControllerCfg
from .triangulation import TriangulationCfg


def _create_robot_cfg() -> ArticulationCfg:
    """Create robot config with gravity and gyroscopic forces enabled."""
    cfg = copy.deepcopy(IRIS_GIMBAL3_CFG)
    cfg.prim_path = "/World/envs/env_.*/{robot_name}"
    # Override rigid body properties to enable gravity and gyroscopic forces
    # Type ignore: spawn is UsdFileCfg at runtime which has rigid_props
    cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(  # type: ignore[union-attr]
        disable_gravity=False,  # Enable gravity for realistic testing
        max_depenetration_velocity=10.0,
        enable_gyroscopic_forces=False,  # Disabled - no gyroscopic coupling needed
    )
    return cfg


def _create_target_cfg() -> RigidObjectCfg:
    """Create target config as RigidObject with gravity enabled.

    Uses iris_body.usda (drone body without propellers) as a simple rigid body.
    RigidObjectCfg is used instead of ArticulationCfg because the USD has no joints.
    """
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_body.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # Enable gravity for physics-based movement
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Disable articulation — treat as pure rigid body
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(5.0, 0.0, 3.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# Pre-create robot config with overridden physics properties
IRIS_GIMBAL2_TEST_CFG = _create_robot_cfg()

# Pre-create target config with gravity enabled
IRIS_TARGET_CFG = _create_target_cfg()


@configclass
class IrisMA6TestEnvCfg(DirectMARLEnvCfg):
    """Configuration for the Iris MA6 Test environment.

    This is a simplified test environment to validate the DroneController
    integration with 3 agents.

    Observation space per agent: 30D ego (pos, vel, rpy, ang_vel_b, lin_acc_b, gimbal, ray, sweep, aoi, zoom, bbox, bbox_empty)
    Action space per agent: 7D (vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate)
    """

    # ==========================================================================
    # Environment Meta
    # ==========================================================================

    num_agents: int = 2
    """Number of agents."""

    episode_length_s: float = 20.0
    """Episode length in seconds."""

    decimation: int = 4
    """Physics steps per control step (25 Hz policy at 100 Hz sim)."""

    # These are populated dynamically in __post_init__ based on num_agents.
    possible_agents: list[str] = ["drone_0", "drone_1", "drone_2"]
    """List of agent identifiers (auto-populated from num_agents)."""

    action_spaces: dict = {"drone_0": 7, "drone_1": 7, "drone_2": 7}
    """Action space dimensions per agent (auto-populated from num_agents)."""

    observation_spaces: dict = {"drone_0": 62, "drone_1": 62, "drone_2": 62}
    """Observation space dimensions per agent. 30D ego + 16D*(num_agents-1) inter-agent [+6D triangulation]."""

    state_space: int = -1
    """State space dimension. -1 means concatenate all observations."""

    # ==========================================================================
    # Simulation
    # ==========================================================================

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_rigid_contact_count=2**23,
            gpu_heap_capacity=2**27,
            gpu_temp_buffer_capacity=2**25,
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
    # Aesthetic Scene
    # ==========================================================================

    debug_vis: bool = True
    """Enable debug visualization."""

    debug_frame_vis: bool = False
    """Enable visualization of coordinate frames for debugging."""

    enable_tiled_cameras: bool = False
    """Enable TiledCamera sensors in the scene. Disable to skip camera creation for faster headless training."""

    use_flight_scene: bool = True # Enable for visualization/testing, disable for faster headless training.
    """Toggle to replace the flat ground plane with the Flight aesthetic scene (Y-up USD, auto-rotated to Z-up)."""

    flight_scene_usd: str = "/home/usrg/IsaacPX4/world/Flight/Flight_original.usd"
    """Path to the Flight scene USD file."""

    flight_scene_scale: float = 0.001
    """Uniform scale applied to the Flight scene.
    The USD is in centimeters (metersPerUnit=0.01) with coordinates in the hundreds of thousands. 
    (~600k, ~1.4M in native units), 0.01 converts cm->m 
    adjust further if needed to fit the environment."""

    flight_scene_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation offset (x, y, z) in meters applied after scaling, to center the scene on the environment."""

    # ==========================================================================
    # Assets
    # ==========================================================================

    target_cfg: RigidObjectCfg = IRIS_TARGET_CFG
    """Target rigid object configuration with gravity enabled."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(-10.0, 0.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_body",
        env_index=0,
        asset_name="Robot_0",
        body_name="body",
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=50.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_GIMBAL2_TEST_CFG
    """Robot articulation configuration template with gravity and gyroscopic forces enabled."""

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/{robot_name}/pitch_link/camera",
        update_period=0.04,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            # ROS convention: forward=+Z, up=-Y
            # Maps camera forward → body +X (forward), camera up → body +Z (up)
            # Same quaternion value as frustum offset for consistency.
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    """Reference camera parameters used to build zoom-aware intrinsics for bbox_raycaster_v2."""

    bbox_raycaster_v2: BBoxRayCasterV2Cfg = BBoxRayCasterV2Cfg(
        target_prim_paths=["/World/envs/env_.*/target"],
        mesh_prim_paths=["/World/ground"],
        num_cameras_per_env=3,
        num_cameras_per_agent=1,
        load_agent_meshes=True,  # Enabled - loads agent body meshes for occlusion
        min_bbox_size=(0.01, 0.01),
        max_bbox_size=(0.95, 0.95),
        partial_detection_allowed=False,
        min_bbox_area_pixels=4.0,
        enable_occlusion_check=True,  # Enabled - raycasting occlusion detection
        enable_self_occlusion=True,  # Enabled - camera's own body can block view
        enable_inter_target_occlusion=True,  # Enabled - image-space inter-target occlusion
        occlusion_ray_pattern="9point",  # Use 9 test points for more robust detection
        occlusion_visibility_threshold=0.3,  # 30% of points must be visible (lower = more sensitive to occlusion)
        occlusion_ray_tolerance=1.2,  # Slightly larger tolerance to reduce edge oscillation
        self_occlusion_min_hit_distance_m=0.05,  # Avoids camera mount false positives
        max_distance=100.0,
        debug_vis=False,
        debug_memory=False,
    )
    """BBox raycaster V2 configuration used for camera/zoom/detection validation."""

    # ==========================================================================
    # Controller Config
    # ==========================================================================

    drone_controller: DroneControllerCfg = TUNED_CONTROLLER_CFG
    """Drone controller configuration. Loaded from auto-tuning results."""

    # ==========================================================================
    # Motion Limits
    # ==========================================================================

    max_lin_vel: float = 10.0
    """Maximum linear velocity (m/s) at full curriculum."""

    max_lin_vel_min: float = 3.0
    """Minimum linear velocity (m/s) at curriculum progress=0. Ramps to max_lin_vel with agent velocity curriculum."""

    max_yaw_rate: float = math.radians(45.0)
    """Maximum yaw rate (rad/s)."""

    max_gimbal_rate: float = math.radians(360.0)
    """Maximum gimbal rate (rad/s)."""

    max_zoom_rate: float = 1.0
    """Maximum zoom rate (zoom levels per second)."""

    # ==========================================================================
    # CBF Safety Configuration
    # ==========================================================================

    cbf_safety: CBFManagerCfg = CBFManagerCfg()
    """CBF safety filter configuration for collision avoidance.

    Training mode (default):
    - enable_training_penalty=True: CPA reward shaping using GT positions
    - enable_deployment_filter=False: Actions unfiltered during training
    - enable_collision_termination=True: Episodes terminate on GT collision

    Deployment mode:
    - enable_training_penalty=False
    - enable_deployment_filter=True: Hard CBF constraint on actions
    """

    # ==========================================================================
    # Delay System Configuration
    # ==========================================================================

    delay_system_params: DelaySystemKeyParams = DelaySystemKeyParams(
        # === Ego Motion Latency (proprioceptive sensing) ===
        ego_motion_latency_enabled=True,  # First-order lag only (fast IMU/GPS)
        ego_motion_fol_tau=0.005,          # 5ms time constant for smoothing
        # === Ego Detection Latency (NN inference) ===
        ego_detection_latency_mean=0.1,   # 100ms mean (GPU inference time)
        ego_detection_latency_std=0.015,   # 15ms std
        # === Other Agent Latency (communication) ===
        other_latency_mean=0.5,            # 500ms mean for other agents (network delay)
        other_latency_std=0.08,            # 80ms std
        # === Staleness (detection FPS) ===
        staleness_fps_mean=25.0,           # 25 FPS mean detection rate
        staleness_fps_range=5.0,           # ±5 FPS range -> [20, 30] FPS
        # === Dropout (missed detections) ===
        dropout_prob=0.05,                 # 5% dropout probability per step
        # === Noise ===
        noise_enabled=True,
        noise_position_std=0.1,            # 10cm position noise
        noise_velocity_std=0.05,           # 5cm/s velocity noise
        noise_orientation_std=0.01,        # ~0.6° orientation noise
        noise_bbox_std=7.0,                # 7 pixels bbox noise (center x,y and w,h)
        # === Reward computation ===
        reward_use_delay=True,             # Use delayed states for rewards
        reward_use_noise=False,            # But without noise (clean delayed)
    )
    """Key parameters for delay system - tune these for sim-to-real transfer.

    Ego Latency Architecture (matches delay_system_v2):
    - Ego MOTION fields (pos, vel, orientation): First-order lag only (fast proprioceptive)
    - Ego DETECTION fields (bboxes): Latency + staleness + dropout (NN inference time)
    - Other agents: Full delay pipeline (communication delay)

    These are the most important parameters affecting observation realism:
    - Ego motion: Nearly instant via first-order lag (proprioceptive sensing)
    - Ego detection: 100ms latency for NN processing on own camera
    - Other agents: 500ms communication delay + staleness + dropout
    - Noise: Sensor measurement noise for all fields
    """

    delay_system: MultiAgentDelayCfgV3 = None  # type: ignore[assignment]
    """Full delay system configuration (built from delay_system_params in __post_init__)."""

    enable_delay_system: bool = True
    """Enable delay system for observations. Set False for ground-truth testing."""

    # ==========================================================================
    # Triangulation Configuration
    # ==========================================================================

    triangulation: TriangulationCfg = TriangulationCfg()
    """Triangulation module configuration for multi-camera target localization."""

    enable_triangulation: bool = True
    """Enable triangulation-based rewards and observations. Default False for backward compatibility."""

    triangulation_reward_scale: float = 5.0
    """Scale factor for triangulation quality reward (analytical mode: 1/sqrt(trace))."""

    # ==========================================================================
    # Reward Scales (migrated from iris_ma5)
    # ==========================================================================

    action_sum_penalty_scale: float = -2.0
    """Penalty scale for total action magnitude."""

    # [0 vx, 1 vy, 2 vz, 3 yaw_rate, 4 gimbal_yaw_rate, 5 gimbal_pitch_rate, 6 zoom_rate]
    action_weight: list = [1, 1, 5, 1, 0.5, 0.5, 0.3]
    """Weights for each action dimension in penalty computation."""

    action_delta_weight: list = [1, 1, 1, 1, 0.5, 0.5, 0.3]
    """Weights for action delta (smoothness) penalty."""

    action_delta_penalty_scale: float = -1.0
    """Penalty scale for action changes (smoothness)."""

    bbox_center_reward_scale: float = 60.0
    """Reward scale for centering target in image."""

    bbox_size_reward_scale: float = 60.0
    """Reward scale for appropriate bbox size (~20% of image area)."""

    collision_penalty_scale: float = -100.0
    """Penalty applied at the moment of collision (sharp spike).
    Provides immediate, localized gradient signal complementing the
    continuous CPA barrier and episode termination."""

    altitude_penalty_scale: float = -100.0
    """Penalty per metre of altitude deficit below altitude_min_threshold.
    Continuous: reward = clamp(threshold - z, 0) * scale * dt."""

    altitude_min_threshold: float = 2.0
    """Minimum altitude (m). Below this, a continuous penalty ramps linearly with deficit."""

    # ==========================================================================
    # Tracking-Lost Truncation
    # ==========================================================================

    enable_tracking_truncation: bool = True
    """Truncate episode when all agents lose detection for tracking_lost_timeout_s."""

    tracking_lost_timeout_s: float = 3.0
    """Seconds of all-agents-blind before truncation."""

    tracking_truncation_grace_steps: int = 50
    """Steps after reset before the tracking-lost counter starts.
    Gives the delay system time to propagate initial detections."""

    # ==========================================================================
    # Curriculum Configuration
    # ==========================================================================

    curriculum: CurriculumCfg = CurriculumCfg()
    """Curriculum configuration for progressive reward scaling."""

    # ==========================================================================
    # Experiment Ablation Flags
    # ==========================================================================

    task_reward_level: int = 1
    """Task reward level for the triangulation reward slot.

    Level 1: FIM proxy - sqrt(10/Tr(Sigma)). Smooth, well-behaved geometric proxy.
    Level 2: GT-anchored estimation error - exp(-temp * ||p_hat_GT - p_true||).
             Triangulates with GT drone positions, measures error vs GT target.
    Level 3: Composite - (1-w)*Level2 + w*E2E estimation error.
             Blends GT-anchored and end-to-end (delayed drone positions).
    """

    estimation_error_scale: float = 1.0
    """Scale for estimation error reward (Levels 2/3). Reward mapped to [0, scale]."""

    estimation_error_temp: float = 1.0
    """Temperature for exponential mapping of estimation error.
    Higher values = sharper reward falloff with distance.
    r = scale * exp(-temp * ||error||)."""

    e2e_weight: float = 0.5
    """End-to-end weight in Level 3 composite reward.
    0.0 = pure GT-anchored, 1.0 = pure E2E.
    Level 3 reward = (1 - e2e_weight) * r_est_GT + e2e_weight * r_est_E2E."""

    curriculum_task_levels: bool = False
    """If True, progressively switch task reward levels using curriculum schedule.
    If False, use task_reward_level as a fixed setting throughout training."""

    use_noisy_rewards: bool = False
    """If True, compute rewards using the noisy delayed pipeline instead of
    the clean pipeline. Used for dual-path ablation."""

    use_omnidirectional_cameras: bool = False
    """If True, simulate omnidirectional cameras.
    Forces all bbox detections to be valid in the reward path."""

    # ==========================================================================
    # Initial States Configuration
    # ==========================================================================

    initial_states: InitialStatesCfg = InitialStatesCfg()
    """Initial states configuration for reset randomization.

    Controls curriculum-driven randomization of:
    - Agent positions (cylinder-based placement)
    - Target position and velocity
    - Gimbal joint angles (designated observer points at target)
    - Zoom levels

    Curriculum sampling prevents forgetting:
    - value ~ Uniform(min, min + progress * (max - min))
    """

    enable_initial_states_randomization: bool = True
    """Enable randomized initial states.

    If True, uses InitialStates module for curriculum-driven randomization.
    If False, uses hardcoded triangle formation (for simple tests/debugging).

    Default is False for simple testing. Set to True for training with curriculum.
    """

    # ==========================================================================
    # Target Controller Configuration
    # ==========================================================================

    target_controller: TargetControllerCfg = TargetControllerCfg()
    """Target controller configuration for physics-based target movement.

    Uses DroneController architecture to generate forces/torques for targets
    instead of direct velocity writes. Supports:
    - Linear/circular movement modes (iris_ma5 compatible)
    - Approach/evade modes (attacker behavior)
    - Curriculum-scaled difficulty
    - Behavior profiles (kamikaze, standard, evasive, stealth)
    """

    enable_target_controller: bool = True
    """Enable physics-based target controller.

    If True, uses TargetController to apply forces/torques to target.
    If False, target remains stationary (for simple tests/debugging).

    Default is False for backward compatibility. Set to True to enable
    physics-based target movement with realistic dynamics.
    """

    # ==========================================================================
    # Controller Gain Randomization
    # ==========================================================================

    gain_randomization: GainRandomizationCfg = GainRandomizationCfg()
    """Configuration for per-env controller gain randomization.

    Applies +-20% uniform scaling on controller gains (velocity PID,
    attitude P, rate PID, motor time constant). Curriculum-gated to
    dynamics phase (dynamics_start_step to dynamics_end_step).
    """

    # ==========================================================================
    # Debugging and Testing Flags
    # ==========================================================================
    debug_initial_step: int = 0 #200000
    """If > 0, initializes the environment at the specified training step for debugging."""


    def __post_init__(self):
        """Populate agent-specific fields from num_agents."""
        if self.num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {self.num_agents}")

        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        self.action_spaces = {a: 7 for a in self.possible_agents}
        self.bbox_raycaster_v2.num_cameras_per_env = self.num_agents

        # Build delay system from key parameters
        self.delay_system = create_delay_cfg_from_params(self.delay_system_params)

        # Update observation space:
        # Ego: 30D (pos, vel, rpy, ang_vel_b, lin_acc_b, gimbal_yaw_body, gimbal_pitch_body,
        #           ray_direction_w, combined_ang_vel_w, bbox_aoi, zoom, bbox, bbox_empty)
        # Inter-agent: 16D per other agent (pos, vel, ray_direction_w, combined_ang_vel_w,
        #              zoom, bbox_empty, data_age, bbox_age)
        # Optional: +6D triangulation (tri_pos + tri_std)
        obs_dim = 30 + 16 * (self.num_agents - 1)
        if self.enable_triangulation:
            obs_dim += 6  # triangulated position (3) + std_dev (3)
        self.observation_spaces = {a: obs_dim for a in self.possible_agents}
