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

from .bbox_raycaster_v2 import BBoxRayCasterV2Cfg, DetectorReplicatorCfg
from .cbf_safety import CBFManagerCfg
from .cbf_safety.cbf_cfg import CPARewardShaperCfg
from .controller import DroneControllerCfg
from .controller.gain_randomization_cfg import GainRandomizationCfg
from .controller.tuning.tuning_results.px4_matched import PX4_MATCHED_CONTROLLER_CFG
from .curriculum import CurriculumCfg
from .domain_randomization import DomainRandomizationCfg, MountOffsetRandomizationCfg
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
            # usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/asset/iris_body.usda",
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


# ============================================================================
# Critic-only privileged env-param obs (ticket 037)
# ============================================================================
# Registry of fields the centralized MAPPO critic can receive as privileged
# observation tails, beyond the actor obs and the existing zoom / GT-target
# tails. Each field is one or more scalars derived from the env's current
# (per-episode, per-(env, agent)) randomization state.
#
# - scope="per_agent": dim per agent; tail size = dim × num_agents.
# - scope="shared":    single global value per env; broadcast / appended once.
#
# Field details (storage owner, shape, getter status, pre-norm range) are in
# doc/critic_obs_design.md §3.4. Implementation:
#   - Slice 2 (this file): cfg surface + dim accounting in __post_init__
#   - Slice 3: getter methods on each owner module
#   - Slice 4: privileged-obs assembly in _get_states()
# Setting `critic_privileged_fields = []` (default) reproduces t034 behavior.
# ============================================================================

# Subset of registry fields whose accessor reads from `self._domain_randomizer`
# (instantiated only when `cfg.domain_randomization.enabled = True`). Requesting
# any of these via `critic_privileged_fields` while DR is disabled would crash
# with AttributeError at obs-build time; `__post_init__` validates and raises
# a clearer error at cfg-time instead.
_CRITIC_PRIVILEGED_FIELDS_REQUIRING_DR: frozenset = frozenset({
    "fov_scale",
    "gimbal_yaw_offset",
    "gimbal_pitch_offset",
    "gimbal_roll_offset",
    "gimbal_stiffness_scale",
    "gimbal_damping_scale",
    "body_mass_scale_or_add",
})


_CRITIC_PRIVILEGED_FIELD_REGISTRY: dict[str, dict] = {
    # --- drone controller gain scales (derived: current / nominal) ---
    "vel_gain_scale":            {"dim": 1, "scope": "per_agent"},
    "att_gain_scale":            {"dim": 1, "scope": "per_agent"},
    "rate_gain_scale":           {"dim": 1, "scope": "per_agent"},
    "motor_gain_scale":          {"dim": 1, "scope": "per_agent"},
    "zoom_tau_scale":            {"dim": 1, "scope": "per_agent"},
    "zoom_max_rate_scale":       {"dim": 1, "scope": "per_agent"},
    # --- drone dynamics (broadcast: stored per-env, applied per-agent) ---
    "max_lin_vel_per_env":       {"dim": 1, "scope": "per_agent"},
    # --- gimbal rate loop (effective τ = nominal × _tau_progress_per_env) ---
    "gimbal_rate_tau_yaw_s":     {"dim": 1, "scope": "per_agent"},
    "gimbal_rate_tau_pitch_s":   {"dim": 1, "scope": "per_agent"},
    # --- dead times ---
    "gimbal_dead_time_s":        {"dim": 1, "scope": "per_agent"},
    "zoom_dead_time_s":          {"dim": 1, "scope": "per_agent"},
    # --- camera intrinsics ---
    "fov_scale":                 {"dim": 1, "scope": "per_agent"},
    # --- gimbal mechanical offsets ---
    "gimbal_yaw_offset":         {"dim": 1, "scope": "per_agent"},
    "gimbal_pitch_offset":       {"dim": 1, "scope": "per_agent"},
    "gimbal_roll_offset":        {"dim": 1, "scope": "per_agent"},
    # --- gimbal joint dynamics ---
    "gimbal_stiffness_scale":    {"dim": 1, "scope": "per_agent"},
    "gimbal_damping_scale":      {"dim": 1, "scope": "per_agent"},
    # --- robot physics (mass scale OR addition, mode-dependent) ---
    "body_mass_scale_or_add":    {"dim": 1, "scope": "per_agent"},
    # --- detection pipeline (env-derived: _eff_progress_* × cfg base) ---
    # `detection_latency_s` is the curriculum-scaled expected latency
    # (_eff_progress_delay × cfg.latency_distribution.mean); the
    # per-episode actual sampled value lives in LatencySampler.step_values
    # — exposing the expected approximates the actual under "fixed" mode
    # and matches the latency distribution mean under "random" mode.
    # `burst_p_onset` is the Gilbert-Elliott burst-onset probability,
    # NOT a generic dropout prob; see _reset_idx ~L1493-1500.
    "detection_latency_s":       {"dim": 1, "scope": "per_agent"},
    "bbox_noise_std":            {"dim": 1, "scope": "per_agent"},
    "detection_dropout_prob":    {"dim": 1, "scope": "per_agent"},
    "burst_p_onset":             {"dim": 1, "scope": "per_agent"},
    # --- target physics (shared per env) ---
    "target_scale_xy":           {"dim": 1, "scope": "shared"},
    "target_scale_z":            {"dim": 1, "scope": "shared"},
}
"""Field registry for ticket 037 critic-only privileged obs. 22 per-agent +
2 shared = 44 dims for A=2. See doc/critic_obs_design.md §3.4 for storage
owners, accessor status, and pre-norm ranges."""


def _critic_privileged_dim_split(field_names: list[str]) -> tuple[int, int]:
    """Sum per-agent and shared dim contributions for the given field list.

    Returns:
        (per_agent_dim, shared_dim) — used by __post_init__ to size state_space.
    """
    per_agent = 0
    shared = 0
    for fname in field_names:
        spec = _CRITIC_PRIVILEGED_FIELD_REGISTRY[fname]
        if spec["scope"] == "per_agent":
            per_agent += spec["dim"]
        elif spec["scope"] == "shared":
            shared += spec["dim"]
        else:
            raise ValueError(f"unknown scope {spec['scope']!r} for field {fname!r}")
    return per_agent, shared


@configclass
class IrisMA6TestEnvCfg(DirectMARLEnvCfg):
    """Configuration for the Iris MA6 Test environment.

    This is a simplified test environment to validate the DroneController
    integration with 3 agents.

    Observation space per agent: 31D ego (pos, vel, rpy, ang_vel_b, lin_acc_b, gimbal, ray, sweep, aoi, zoom, effective_hfov, bbox, bbox_empty)
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
    """Physics steps per control step (e.g. 25 Hz policy at 100 Hz sim)."""

    # These are populated dynamically in __post_init__ based on num_agents.
    possible_agents: list[str] = ["drone_0", "drone_1", "drone_2"]
    """List of agent identifiers (auto-populated from num_agents)."""

    action_spaces: dict = {"drone_0": 7, "drone_1": 7, "drone_2": 7}
    """Action space dimensions per agent (auto-populated from num_agents)."""

    observation_spaces: dict = {"drone_0": 62, "drone_1": 62, "drone_2": 62}
    """Observation space dimensions per agent. 31D ego + 16D*(num_agents-1) inter-agent [+6D triangulation]."""

    state_space: int = -1
    """State space dimension. -1 means concatenate all observations.

    When any of the following toggles is set, the env auto-replaces this with
    a positive int matching the actor-obs concat plus the privileged tails
    enabled:

    - ``enable_critic_continuous_zoom``: 2 scalars per agent
    - ``enable_critic_gt_target``: 3 globals (target world position)
    - ``enable_critic_gt_target_velocity``: 3 globals (target world linear velocity)
    - ``critic_privileged_fields`` (ticket 037): per-field dim from the registry
      (22 per-agent + 2 shared = 44 dims for A=2 if all fields enabled)

    The env also exposes ``shared_observation_spaces`` so the centralized
    MAPPO critic picks up the new size.
    """

    enable_critic_continuous_zoom: bool = True
    """Asymmetric actor-critic: feed the centralized MAPPO critic privileged
    zoom-state info (continuous ``zoom_internal`` and integrator
    ``zoom_target``) appended after the standard concat-of-actor-obs. The
    actor's observation is unchanged (still sees the quantized published
    zoom that hardware will expose at deployment).

    Rationale (siyi_a8 mode): with quantum=0.1 and max Δzoom/policy_step≈0.08,
    the actor's zoom obs is staircased and the per-step gradient signal on the
    zoom dim drops to zero between quanta. A symmetric critic (sees same
    quantized obs as the actor) is forced to predict the *average* return
    across all internal states sharing one quantum bin, blurring V predictions
    and inflating advantage variance. An asymmetric critic seeing the
    continuous internal state can resolve those internal states, producing
    sharper V estimates and crisper temporal credit assignment for the
    integrator chain that ultimately produces a quantum crossing several
    steps later. No deployment-side change — the critic is training-only."""

    enable_critic_gt_target: bool = False
    """Asymmetric actor-critic: feed the centralized MAPPO critic the
    GT target world position (3 dims) appended after the actor-obs concat
    (and after the zoom tail, if enabled). The actor's observation is
    unchanged — actor sees only delayed/noisy bbox-derived target signals,
    matching deployment.

    Rationale: every reward term in this env is a function of the drone's
    geometric relationship to the target (bbox_center, bbox_size,
    triangulation FIM, target_proximity, tracking_lost). Giving the critic
    the GT target lets V regress against the causal observable directly
    instead of inferring it through delayed/noisy bbox observations,
    cutting advantage variance and stabilizing the curriculum-stress
    regime. Indirectly stabilizes the policy tri head's training
    distribution by keeping ``effective_sample_fraction`` high (the head
    relies on per-agent bbox+scene-valid masks that collapse when policy
    training thrashes).

    Source: ``self._target_pos_w`` (snapshot of ``self.target.data.root_pos_w``,
    updated each step in ``_update_state_cache``). Same source as the aux-head
    supervision target, so the critic and the head are aligned.

    No deployment-side change — the critic is training-only."""

    enable_critic_gt_target_velocity: bool = False
    """Asymmetric actor-critic: also append the GT target world-frame linear
    velocity (3 dims) to the critic state. Off by default — pure-position
    privileged info usually captures the dominant signal. Turn on if you
    want to test whether velocity helps the critic anticipate next-step
    rewards (closing speed, occlusion onset). Requires
    ``enable_critic_gt_target=True``."""

    enable_axis_independence: bool = True
    """Ticket 037 Slice 7 — when True, the 8 sub-axes of ``progress_dynamics``
    are sampled independently per (env, agent) using their own curriculum
    schedules from ``curriculum.{axis}_(start|end)_step``. When False
    (default), all 8 axes share the ``dynamics_(start|end)_step`` schedule
    — bit-exact t034 behavior.

    The 8 axes (each per-(env, agent)):
    - gimbal_rate_tau (replaces _eff_progress_dynamics for the gimbal rate-loop τ)
    - drone_gains (controller gain randomization)
    - max_lin_vel_scale (±20 % max_lin_vel multiplier)
    - camera_fov (FOV scale)
    - gimbal_mech_offsets (yaw/pitch/roll mechanical misalignment)
    - mass_inertia (body mass + inertia)
    - gimbal_stiff_damp (gimbal joint stiffness/damping)
    - target_scale (target object xy/z scale)

    Real hardware has no correlation between these axes; bundling masks
    sim-to-real failure modes. See doc/critic_obs_design.md §2.3 and
    ticket 037 for motivation."""

    enable_full_critic_priv_obs: bool = True
    """Convenience toggle for ticket 037 — when True and
    ``critic_privileged_fields`` is empty, ``__post_init__`` populates the
    list with all ~24 entries in ``_CRITIC_PRIVILEGED_FIELD_REGISTRY``.

    Useful for CLI overrides (Hydra's struct mode treats an empty
    ``list[str]`` default as length-locked, so ``critic_privileged_fields``
    cannot be overridden from an empty default via ``++env.critic_priv...``).
    For ablation, set ``critic_privileged_fields`` directly to a subset
    (programmatically, in cfg constructor / experiment_registry)."""

    critic_privileged_fields: list[str] = []
    """Ticket 037 — asymmetric actor-critic env-param privileged tail.

    List of field names from ``_CRITIC_PRIVILEGED_FIELD_REGISTRY`` (defined
    at module scope above the cfg class) to append to the critic state.
    Each field contributes its registered ``dim`` × ``scope``:

    - ``scope="per_agent"``: 1 scalar per agent appended at this slot.
    - ``scope="shared"``: 1 scalar per env appended once at this slot.

    The actor observation is unchanged. Setting this list to ``[]``
    (default) reproduces t034 behavior bit-exactly.

    Typical "full Phase 1" configuration is all 24 fields in the registry
    (22 per-agent + 2 shared = 44 dims for A=2). Smaller subsets can be
    used for ablation: e.g. ``["max_lin_vel_per_env"]`` to isolate the
    velocity-axis V-loss contribution.

    Field details (storage owner, accessor, pre-norm range) are in
    doc/critic_obs_design.md §3.4. Implementation slices:

    - Slice 2 (this file): cfg surface + dim accounting in __post_init__.
    - Slice 3: getter methods on each owner module.
    - Slice 4: privileged-obs assembly in _get_states().

    Validated against the registry in __post_init__; unknown field names
    raise ValueError at cfg-time (before env init), so config errors
    surface early."""

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
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 0.0),
            opacity=0.0,
        ),
        debug_vis=False,
    )

    # ==========================================================================
    # Aesthetic Scene
    # ==========================================================================

    debug_vis: bool = True
    """Enable debug visualization."""

    debug_lock_gimbal_to_target: bool = False
    """Bypass policy gimbal commands and always point gimbal at GT target position.
    Useful for calibration scripts and testing bbox pipeline without gimbal noise."""

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

    flight_scene_offset: tuple[float, float, float] = (0.0, 0.0, -20.0)
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
    # viewer: ViewerCfg = ViewerCfg(
    #     eye=(120.0, 120.0, 95.0),
    #     lookat=(0.0, 0.0, 32.0),
    # )
    # viewer: ViewerCfg = ViewerCfg(
    #     eye=(-2.0, 2.0, 2.0),
    #     lookat=(0.0, 0.0, 0.0),
    #     origin_type="asset_body",
    #     env_index=0,
    #     asset_name="Robot_0",
    #     body_name="body",
    # )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=200.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_GIMBAL2_TEST_CFG
    """Robot articulation configuration template with gravity and gyroscopic forces enabled."""

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/{robot_name}/pitch_link/camera",
        update_period=0.04,
        # Resolution + intrinsics aligned to mrcal 1x calibration
        # (see /home/usrg/mas/datasets/camera_calibration/2026-04-17/1x/intrinsics_summary.json
        # and src/scripts/sim2real_model_fitting/output/intrinsics_for_sim.json:trustworthy_zooms.1x).
        # Real camera: 1920x1080 with fx ~ 1053 px (HFOV ~ 85 deg).
        # focal_length / horizontal_aperture * width = fx_px:
        #   (11.493 / 20.955) * 1920 = 1053.0446 px (target 1053.044591).
        # horizontal_aperture kept at the prior 20.955 mm "sensor-size" convention;
        # focal_length solved to satisfy the fx target. Principal point stays
        # centered (W/2, H/2) per domain_randomization/doc/README.md design choice.
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.493,
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

    drone_controller: DroneControllerCfg = PX4_MATCHED_CONTROLLER_CFG
    """Drone controller configuration. PX4 SITL-matched gains from sysid replicator (ticket-008)."""

    # ==========================================================================
    # Motion Limits
    # ==========================================================================

    max_lin_vel: float = 10.0
    """Maximum linear velocity (m/s) at full curriculum."""

    max_lin_vel_min: float = 3.0
    """Minimum linear velocity (m/s) at curriculum progress=0. Ramps to max_lin_vel with agent velocity curriculum."""

    max_yaw_rate: float = math.radians(45.0)
    """Maximum yaw rate (rad/s)."""

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
        ego_motion_latency_enabled=True,  # Transport latency for IMU/GPS
        ego_motion_latency_mean=0.005,    # 5ms mean transport latency
        ego_motion_latency_std=0.002,     # 2ms std
        ego_motion_fol_tau=0.005,          # 5ms time constant for smoothing
        # === Ego Detection Latency (glass -> detection output) ===
        # LOW:  0.296 / 0.022
        # MID: 	0.310 / 0.021
        # HIGH: 0.386 / 0.021
        ego_detection_latency_mean=0.31,   # 310 ms — glass-to-topic p50
        ego_detection_latency_std=0.021,    # 30 ms — std of QR-bench distribution
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

    continuous_aoi_jitter: bool = True
    """Add uniform sub-step jitter to capture timestamps for continuous AoI.

    When True, timestamp_motion and timestamp_detection get independent
    U(-step_dt, 0) offsets so the policy sees continuous
    age-of-information instead of staircase multiples of step_dt."""

    # ==========================================================================
    # Detector Replicator Configuration
    # ==========================================================================

    calibrated_bbox_noise: DetectorReplicatorCfg = DetectorReplicatorCfg()
    """Calibrated bbox noise from offline YOLO calibration.

    When enabled, applies Student's t noise with size-dependent scale to raycaster
    bboxes, replicating real detector error profile. Replaces delay system bbox_std.
    See experiments/calibrate_bbox_noise.py for calibration workflow.
    """

    # ==========================================================================
    # Triangulation Configuration
    # ==========================================================================

    triangulation: TriangulationCfg = TriangulationCfg()
    """Triangulation module configuration for multi-camera target localization."""

    enable_triangulation: bool = False
    """Append triangulation tail to the actor observation (and draw the observed-triangulation
    ellipsoid in viz). Reward-side triangulation (_triangulation_result_gt / _tri_result_l2 /
    _tri_result_l3) is always computed, independent of this flag."""

    triangulation_reward_scale: float = 5.0
    """Scale factor for triangulation quality reward (analytical mode: 1/sqrt(trace))."""

    # ==========================================================================
    # Reward Scales (migrated from iris_ma5)
    # ==========================================================================

    action_sum_penalty_scale: float = -8.0
    """Penalty scale for total action magnitude."""

    # [0 vx, 1 vy, 2 vz, 3 yaw_rate, 4 gimbal_yaw_rate, 5 gimbal_pitch_rate, 6 zoom_rate]
    action_weight: list = [1, 1, 5, 1, 1, 1, 1]
    """Weights for each action dimension in penalty computation."""

    action_delta_weight: list = [1, 1, 1, 1, 1, 1, 1]
    """Weights for action delta (smoothness) penalty."""

    action_delta_penalty_scale: float = -12.0
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

    target_proximity_penalty_scale: float = -50.0
    """Penalty per metre of proximity deficit below target_proximity_threshold.
    Continuous: reward = clamp(threshold - dist_to_target, 0) * scale * dt.
    Gated by progress_safety curriculum."""

    target_proximity_threshold: float = 7.0
    """Minimum distance (m) from target. Below this, a continuous penalty ramps linearly."""

    # ==========================================================================
    # Tracking-Lost Truncation
    # ==========================================================================

    enable_tracking_truncation: bool = True
    """Truncate episode when all agents lose detection for tracking_lost_timeout_s."""

    tracking_lost_timeout_s: float = 2.0
    """Seconds of all-agents-blind before truncation.

    Increased from 2.0 → 3.0 to give the policy more time to recover from
    temporary bbox blackouts during the FP/FN curriculum phase. Each
    truncation cuts off the learning signal, so a longer grace window
    lets the policy experience recovery gradients.
    """

    tracking_truncation_grace_steps: int = 50
    """Steps after reset before the tracking-lost counter starts.
    Gives the delay system time to propagate initial detections."""

    tracking_reacquire_steps: int = 50
    """Consecutive valid-detection frames required to reset the lost counter.
    Prevents noise-induced single valid frames from resetting the timer.
    At decimation=4, dt=0.01: 50 steps ~ 2.0s of sustained reacquisition."""

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

    initial_states: InitialStatesCfg = InitialStatesCfg(
        # agent_velocity_scale_max=0.0,
        # target_velocity_scale_max=0.0,
        # max_yaw_rate=0.0,
        cylinder_diameter_max=60.0,
        target_distance_max=25.0,
    )
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

    Applies +-5% uniform scaling on controller gains (velocity PID,
    attitude P, rate PID, motor time constant). Curriculum-gated to
    dynamics phase (dynamics_start_step to dynamics_end_step).
    """

    # ==========================================================================
    # Domain Randomization
    # ==========================================================================

    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enabled=False,
        mount_offset=MountOffsetRandomizationCfg(enabled=False),
    )
    """Domain randomization for sim-to-real transfer.

    Randomizes camera intrinsics, physics mass/material, and gimbal
    dynamics at each episode reset. Curriculum-gated to the dynamics
    phase (progress_dynamics, same as gain_randomization).

    Camera randomization affects the intrinsic matrix used by
    bbox_raycaster and delay system (no actual RGB processing).
    Physics mass is applied to simulation assets; the controller
    retains nominal mass (intentional model mismatch for robustness).
    Gimbal offsets are added to joint position targets.
    """

    target_xy_scale_range: tuple[float, float] = (0.5, 2.5)
    """XY-axis scale range for target visual size randomization (x=y, uniform).

    Enriches bbox size variation by scaling the target's apparent width/depth.
    Applied to the bbox raycaster via target_scale parameter (no USD/physics change).
    Curriculum-gated by progress_dynamics like other DR parameters.
    """

    target_z_scale_range: tuple[float, float] = (1.0, 3.0)
    """Z-axis scale range for target visual size randomization.

    Z-scale is clamped to be >= xy-scale so the target never appears
    squashed vertically. Applied via bbox raycaster target_scale.
    Curriculum-gated by progress_dynamics like other DR parameters.
    """

    # ==========================================================================
    # Debugging and Testing Flags
    # ==========================================================================
    use_debug_initial_step: bool = False
    debug_initial_step: int = 400000
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

        # When detector replicator is enabled, zero out delay system bbox noise
        # (calibrated noise is applied upstream by the replicator)
        if self.calibrated_bbox_noise.enabled:
            from .delay_system_v3 import DistributionCfg
            self.delay_system.noise.bbox_std = DistributionCfg(
                type="constant", value=0.0, min_value=0.0
            )

        # Update observation space:
        # Ego: 31D (pos, vel, rpy, ang_vel_b, lin_acc_b, gimbal_yaw_body, gimbal_pitch_body,
        #           ray_direction_w, combined_ang_vel_w, bbox_aoi, zoom, effective_hfov,
        #           bbox, bbox_empty)
        # Inter-agent: 16D per other agent (pos, vel, ray_direction_w, combined_ang_vel_w,
        #              zoom, bbox_empty, data_age, bbox_age)
        # Optional: +6D triangulation (tri_pos + tri_std)
        obs_dim = 31 + 16 * (self.num_agents - 1)
        if self.enable_triangulation:
            obs_dim += 6  # triangulated position (3) + std_dev (3)
        self.observation_spaces = {a: obs_dim for a in self.possible_agents}

        # ------------------------------------------------------------------
        # Asymmetric actor-critic: when any enable_critic_* toggle is True,
        # set state_space to a positive int matching the critic-side state
        # size: concat-of-actor-obs plus the enabled privileged tails.
        # The env's _get_states() materializes this state; DirectMARLEnv.state()
        # routes to it; skrl's MAPPO trainer reads it via env.state() and
        # injects into infos before record_transition. Untouched when all
        # toggles are False (state_space stays -1, auto-concat-of-obs path).
        #
        # Tails:
        #   - zoom (2 dims/agent)        if enable_critic_continuous_zoom
        #   - GT target position (3)     if enable_critic_gt_target
        #   - GT target velocity (3)     if enable_critic_gt_target_velocity
        #     (requires enable_critic_gt_target)
        # ------------------------------------------------------------------
        critic_extra_per_agent = 0
        if getattr(self, "enable_critic_continuous_zoom", False):
            critic_extra_per_agent += 2  # zoom_internal + zoom_target

        critic_extra_global = 0  # single-target globals (one target in v0)
        if getattr(self, "enable_critic_gt_target", False):
            critic_extra_global += 3  # GT target world position
            if getattr(self, "enable_critic_gt_target_velocity", False):
                critic_extra_global += 3  # GT target world linear velocity

        # Ticket 037 — env-param privileged tail. The convenience bool
        # `enable_full_critic_priv_obs` expands to the registry when the
        # explicit list is empty. When `domain_randomization.enabled=False`,
        # DR-dependent fields are EXCLUDED (with a log line) since their
        # accessors would crash; this keeps the toggle ergonomic across
        # both DR-on and DR-off configs.
        if self.enable_full_critic_priv_obs and not self.critic_privileged_fields:
            all_fields = list(_CRITIC_PRIVILEGED_FIELD_REGISTRY.keys())
            if self.domain_randomization.enabled:
                self.critic_privileged_fields = all_fields
            else:
                self.critic_privileged_fields = [
                    f for f in all_fields if f not in _CRITIC_PRIVILEGED_FIELDS_REQUIRING_DR
                ]
                dropped = sorted(set(all_fields) - set(self.critic_privileged_fields))
                print(
                    f"[IrisMA6TestEnvCfg] enable_full_critic_priv_obs=True with "
                    f"domain_randomization.enabled=False — dropped {len(dropped)} "
                    f"DR-dependent fields: {dropped}. Enabled fields: "
                    f"{len(self.critic_privileged_fields)} non-DR.",
                    flush=True,
                )
        # cfg-driven by critic_privileged_fields list, validated against the registry.
        if self.critic_privileged_fields:
            unknown = (
                set(self.critic_privileged_fields)
                - set(_CRITIC_PRIVILEGED_FIELD_REGISTRY.keys())
            )
            if unknown:
                raise ValueError(
                    f"unknown critic_privileged_fields: {sorted(unknown)}. "
                    f"valid keys: {sorted(_CRITIC_PRIVILEGED_FIELD_REGISTRY.keys())}"
                )
            duplicates = [
                f for f in set(self.critic_privileged_fields)
                if self.critic_privileged_fields.count(f) > 1
            ]
            if duplicates:
                raise ValueError(
                    f"duplicate critic_privileged_fields: {sorted(duplicates)}"
                )
            # DR-dependent fields require domain_randomization.enabled=True;
            # accessor would crash with AttributeError on disabled DR.
            if not self.domain_randomization.enabled:
                needs_dr = (
                    set(self.critic_privileged_fields)
                    & _CRITIC_PRIVILEGED_FIELDS_REQUIRING_DR
                )
                if needs_dr:
                    raise ValueError(
                        f"critic_privileged_fields includes DR-dependent fields "
                        f"{sorted(needs_dr)} but domain_randomization.enabled=False. "
                        f"Enable DR or remove these fields."
                    )
            pa, sh = _critic_privileged_dim_split(self.critic_privileged_fields)
            critic_extra_per_agent += pa
            critic_extra_global += sh

        if critic_extra_per_agent > 0 or critic_extra_global > 0:
            actor_concat_dim = obs_dim * self.num_agents
            self.state_space = (
                actor_concat_dim
                + critic_extra_per_agent * self.num_agents
                + critic_extra_global
            )
