#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained checkpoint and compute paper metrics for iris_ma6.

Modes of operation:
    1. Standard evaluation      -- compute metrics + timeseries
    2. Parameter sweep          -- override runtime params (delay, speed, noise, etc.)
    3. Trajectory recording     -- record agent/target XYZ paths for visualization

Usage examples:

    # Standard evaluation of a trained policy
    ./isaaclab.sh -p .../evaluate.py \\
        --experiment a3_delay_only \\
        --checkpoint /path/to/best_agent.pt \\
        --num_episodes 1 --num_envs 4096 \\
        --output results.json

    # Evaluate greedy baseline (no checkpoint needed)
    ./isaaclab.sh -p .../evaluate.py \\
        --experiment baseline_greedy \\
        --num_episodes 1 --num_envs 4096

    # Parameter sweep: override detection delay
    ./isaaclab.sh -p .../evaluate.py \\
        --experiment a3_delay_only \\
        --checkpoint /path/to/best_agent.pt \\
        --delay-override 0.1 --output results_100ms.json

    # Record trajectories for visualization
    ./isaaclab.sh -p .../evaluate.py \\
        --experiment a3_delay_only \\
        --checkpoint /path/to/best_agent.pt \\
        --record-trajectory --trajectory-envs 8 \\
        --no-timeseries --output traj.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from experiment_registry import get_experiment

parser = argparse.ArgumentParser(description="Evaluate trained policy with paper metrics (iris_ma6).")
parser.add_argument("--experiment", type=str, default="a1_with_aoi", help="Experiment name from registry")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
parser.add_argument("--num_episodes", type=int, default=1, help="Episodes per env")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel eval envs")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA6-Direct-Test-v0",
                    help="Task name (overrides experiment task)")
parser.add_argument("--output", type=str, default=None, help="Output JSON path")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--enable_cameras", action="store_true", default=False)
# Runtime parameter overrides
parser.add_argument("--delay-override", type=float, default=None,
                    help="Override detection latency in seconds")
parser.add_argument("--comm-delay-override", type=float, default=None,
                    help="Override communication latency in seconds")
parser.add_argument("--target-speed", type=float, default=None,
                    help="Override target speed in m/s")
parser.add_argument("--noise-override", type=float, default=None,
                    help="Override bbox pixel noise std")
parser.add_argument("--detection-dropout-override", type=float, default=None,
                    help="Override detection failure rate (0-1)")
parser.add_argument("--no-timeseries", action="store_true", default=False,
                    help="Disable per-timestep timeseries collection")
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--step", type=int, default=None,
                    help="Training step to pin curriculum to via debug_initial_step. "
                         "When set, all curriculum-driven parameters evaluate as if at this "
                         "training step. When unset (default), all curriculum start/end steps "
                         "are forced to 0 (full difficulty everywhere).")
parser.add_argument("--target-trajectory-mode", type=str, default=None,
                    choices=["linear", "circular"],
                    help="Force target trajectory mode (overrides linear_weight)")
parser.add_argument("--policy-combo", type=str, default="default",
                    choices=["default", "a0a0", "a0a1", "a1a1"],
                    help="Policy weight assignment: which drone's weights each agent uses")
# Trajectory recording
parser.add_argument("--record-trajectory", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--trajectory-envs", type=int, default=8)
# Action diagnostics (per-axis RMS / FFT; trace serialization gated)
parser.add_argument("--record-action-trace", action=argparse.BooleanOptionalAction, default=True,
                    help="Include per-step action trace arrays in action_diagnostics output")
# Video recording
parser.add_argument("--record-video", type=str, default=None,
                    help="Output path for video (e.g., demo.mp4)")
parser.add_argument("--record-video-scene", type=str, default=None,
                    help="Output .mp4 path for a wide-shot RTX SCENE video. Renders "
                         "env_cfg.viewer (isometric view of env 0) to rgb_array and "
                         "writes every frame manually in the rollout loop (not "
                         "gymnasium RecordVideo, which stops on the vec-env done). "
                         "Shows the drone USD models + scene visuals. Needs the warp "
                         "'owner' shim (auto-applied) for replicator RGB capture.")
parser.add_argument("--camera-mode", type=str, default="overhead",
                    choices=["overhead", "chase", "side", "orbit",
                             "formation", "closeup", "wide", "isometric"],
                    help="Camera preset mode")
parser.add_argument("--camera-smoothing", type=float, default=0.08,
                    help="Camera smoothing factor 0-1 (lower=smoother)")
parser.add_argument("--video-fps", type=int, default=30, help="Video frame rate")
parser.add_argument("--video-resolution", type=int, nargs=2, default=[1920, 1080],
                    metavar=("WIDTH", "HEIGHT"), help="Video resolution")
args_cli, hydra_args = parser.parse_known_args()

_selected_experiment = get_experiment(args_cli.experiment)
_resolved_task = (
    args_cli.task
    if args_cli.task is not None
    else _selected_experiment.task
)
args_cli.task = _resolved_task

sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher

# Video recording: headless mode with offscreen camera
_headless_video = args_cli.record_video is not None and args_cli.headless
_enable_cameras = args_cli.enable_cameras
if _headless_video and not _enable_cameras:
    _enable_cameras = True
    print("[EVAL] Auto-enabling cameras for headless video recording")
if args_cli.record_video_scene is not None and not _enable_cameras:
    _enable_cameras = True
    print("[EVAL] Auto-enabling cameras for scene video (RecordVideo) recording")
if args_cli.record_video is not None:
    if _headless_video:
        print("[EVAL] Video recording: using offscreen camera (headless mode)")
    else:
        print("[EVAL] Video recording: using viewport capture (GUI mode)")

app_args = argparse.Namespace(
    headless=args_cli.headless,
    device="cuda:0",
    experience="",
    enable_cameras=_enable_cameras,
)
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

# --- warp <-> replicator compatibility shim (for RTX rgb_array capture) ---
# Isaac Sim's bundled omni.replicator (annotator_utils._reshape_output_ptr) builds
# the RGB buffer with wp.types.array(..., owner=False, ...), but warp>=1.x removed
# the `owner` kwarg (replaced by `deleter`). owner=False == a non-owning view, which
# is exactly the default when no deleter is supplied — so we just drop the kwarg.
# Without this, EVERY rgb_array render raises:
#   TypeError: array.__init__() got an unexpected keyword argument 'owner'
# Applied only when a video is requested, so non-video eval is byte-for-byte unchanged.
if args_cli.record_video is not None or args_cli.record_video_scene is not None:
    try:
        import inspect as _inspect
        import warp as _wp
        if "owner" not in _inspect.signature(_wp.types.array.__init__).parameters:
            _orig_wp_array_init = _wp.types.array.__init__
            def _wp_array_init_compat(self, *a, **kw):
                kw.pop("owner", None)
                return _orig_wp_array_init(self, *a, **kw)
            _wp.types.array.__init__ = _wp_array_init_compat
            print("[EVAL] Applied warp 'owner'-kwarg compat shim for replicator RGB capture")
    except Exception as _e:
        print(f"[EVAL] WARNING: warp compat shim not applied ({_e}); RTX capture may fail")

"""Rest follows after Isaac Sim is initialized."""

import torch
import gymnasium as gym
import numpy as np
import copy

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.iris_ma6.experiments import (
    get_experiment,
    apply_env_overrides,
    apply_agent_overrides,
)
from isaaclab_tasks.direct.iris_ma6.experiments.metrics import (
    ActionTraceRecorder,
    MetricTracker,
    TimeseriesTracker,
)

if args_cli.record_video is not None:
    from isaaclab_tasks.direct.iris_ma5.video_recording import (
        SmoothCameraController, VideoRecorder, VideoRecorderCfg, get_preset,
    )

# Add skrl scripts dir to path
_skrl_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
                                  "scripts", "reinforcement_learning", "skrl")
_skrl_scripts_dir = os.path.normpath(_skrl_scripts_dir)
sys.path.insert(0, _skrl_scripts_dir)


def _create_video_camera(prim_path):
    """Create a UsdGeom.Camera stage prim we can drive per-frame (wide FOV)."""
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            prim = stage.DefinePrim(prim_path, "Camera")
        cam = UsdGeom.Camera(prim)
        cam.GetFocalLengthAttr().Set(16.0)              # wide-ish FOV to fit 3 agents
        cam.GetHorizontalApertureAttr().Set(20.955)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
    except Exception as _e:
        print(f"[EVAL] WARNING: could not create video camera: {_e}")


def _aim_camera_prim(prim_path, eye, lookat):
    """Set a USD camera prim's transform from eye/lookat (world coords).

    Bypasses sim.set_camera_view, which is a no-op in headless configs. Mirrors
    the look-at math in iris_ma5 video_recording/video_recorder.py.
    """
    try:
        import numpy as _np
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        cam = stage.GetPrimAtPath(prim_path)
        if not cam.IsValid():
            return
        eye = _np.asarray(eye, dtype=float)
        fwd = _np.asarray(lookat, dtype=float) - eye
        fwd = fwd / (_np.linalg.norm(fwd) + 1e-8)
        up_w = _np.array([0.0, 0.0, 1.0])
        right = _np.cross(fwd, up_w)
        if _np.linalg.norm(right) < 1e-6:
            up_w = _np.array([0.0, 1.0, 0.0])
            right = _np.cross(fwd, up_w)
        right = right / (_np.linalg.norm(right) + 1e-8)
        up = _np.cross(right, fwd)
        up = up / (_np.linalg.norm(up) + 1e-8)
        R = _np.eye(3)
        R[:, 0] = right
        R[:, 1] = up
        R[:, 2] = -fwd  # camera looks along -Z
        gf_rot = Gf.Matrix3d(
            Gf.Vec3d(float(R[0, 0]), float(R[1, 0]), float(R[2, 0])),
            Gf.Vec3d(float(R[0, 1]), float(R[1, 1]), float(R[2, 1])),
            Gf.Vec3d(float(R[0, 2]), float(R[1, 2]), float(R[2, 2])),
        )
        T = Gf.Matrix4d()
        T.SetRotate(gf_rot)
        T.SetTranslateOnly(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
        xf = UsdGeom.Xformable(cam)
        ops = xf.GetOrderedXformOps()
        if ops:
            ops[0].Set(T)
        else:
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(T)
    except Exception:
        pass


def _collect_step_metrics(
    tracker: "MetricTracker | None",
    env,
    ts_tracker: "TimeseriesTracker | None" = None,
    done_mask: "torch.Tensor | None" = None,
) -> None:
    """Extract per-step metrics from iris_ma6 environment internals.

    Uses the observation-path triangulation for position estimate (midpoint method)
    and the GT-path triangulation for covariance trace.
    """
    # Triangulation results from GT path (for trace/covariance)
    tri_gt = env._triangulation_result_gt
    # Triangulation results from obs path (for RMSE — midpoint estimate)
    tri_obs = env._triangulation_result_obs
    # Policies trained without the triangulation obs-tail (cfg.enable_triangulation=False,
    # e.g. the t047 checkpoints) never populate the obs-path result during _get_observations.
    # Compute it here for metrics ONLY — this mirrors the env's own obs-path call
    # (iris_ma_env6_test.py:2319-2332) and does not touch the observation vector, so it is
    # safe for checkpoints whose obs_dim excludes the triangulation tail.
    if tri_obs is None:
        if env._delay_system is not None:
            _tri_states = env._delay_system.get_all_states_for_observations(
                ego_agent_id=env.cfg.possible_agents[0]
            )
        else:
            _tri_states = env._build_gt_states()
        tri_obs = env._compute_triangulation(states=_tri_states, use_gt_target=False)

    if tri_gt is not None:
        cov = tri_gt.covariance[:, 0, :, :]  # (N, 3, 3)
        trace_sigma = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)  # (N,)
        # NaN/Inf from ill-conditioned FIM — mark as invalid
        bad_trace = torch.isnan(trace_sigma) | torch.isinf(trace_sigma)
        trace_sigma = torch.where(bad_trace, torch.zeros_like(trace_sigma), trace_sigma)
        tri_valid_gt = tri_gt.is_valid[:, 0] & ~bad_trace  # (N,)
    else:
        trace_sigma = torch.ones(env.num_envs, device=env.device) * 999.0
        tri_valid_gt = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if tri_obs is not None:
        tri_pos = tri_obs.position[:, 0, :]  # (N, 3)
        tri_valid_obs = tri_obs.is_valid[:, 0]  # (N,)
    else:
        tri_pos = torch.zeros(env.num_envs, 3, device=env.device)
        tri_valid_obs = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # Use obs path validity for metric gating (what the agent actually sees)
    # Also require GT path validity for covariance metrics
    tri_valid = tri_valid_obs
    tri_valid_with_cov = tri_valid_obs & tri_valid_gt

    # Ground truth target position
    gt_target_pos = env._target_pos_w  # (N, 3)

    # Bbox validity per agent
    bbox_valid_list = []
    for agent_id in env.cfg.possible_agents:
        if env._delay_system is not None:
            states = env._delay_system.get_all_states_for_observations(ego_agent_id=agent_id)
            bbox = states[agent_id].data.bboxes_2d[:, 0, :]
        else:
            idx = env.cfg.possible_agents.index(agent_id)
            bbox = env.bbox_raycaster_v2.data.bboxes_normalized[:, idx, 0, :]
        valid = bbox.abs().sum(dim=-1) > 1e-6
        bbox_valid_list.append(valid)
    bbox_valid_mask = torch.stack(bbox_valid_list, dim=1)  # (N, A)

    # Collision detection via CBF manager
    gt_positions = torch.stack(
        [env._root_pos_w[a] for a in env.cfg.possible_agents], dim=1
    )  # (N, A, 3)
    collision_flags = env.cbf_manager.check_collisions(gt_positions)  # (N,)

    # CBF penalty
    cmd_vel = env.cmd_vel[:, :, 0:3]
    dt = env.cfg.sim.dt * env.cfg.decimation
    cbf_penalty = env.cbf_manager.compute_training_penalty(
        gt_positions=gt_positions,
        commanded_velocities=cmd_vel,
        dt=dt,
    )  # (N,)

    # Min pairwise distance
    num_agents = len(env.cfg.possible_agents)
    if num_agents >= 2:
        dists = []
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                d = (gt_positions[:, i] - gt_positions[:, j]).norm(dim=-1)
                dists.append(d)
        min_dist = torch.stack(dists, dim=-1).min(dim=-1).values  # (N,)
    else:
        min_dist = torch.full((env.num_envs,), float("inf"), device=env.device)

    if tracker is not None:
        tracker.step(
            trace_sigma=trace_sigma,
            triangulated_pos=tri_pos,
            gt_target_pos=gt_target_pos,
            bbox_valid_mask=bbox_valid_mask,
            collision_flags=collision_flags,
            tri_valid=tri_valid,
            cbf_penalty=cbf_penalty,
            min_pairwise_dist=min_dist,
        )

    # Feed timeseries tracker
    if ts_tracker is not None:
        rmse = torch.norm(tri_pos - gt_target_pos, dim=-1)  # (N,)
        sqrt_trace = torch.sqrt(trace_sigma.clamp(min=0))  # (N,)
        # Zero out sqrt_trace where GT covariance is invalid to prevent NaN poisoning
        sqrt_trace = torch.where(
            tri_valid_with_cov, sqrt_trace, torch.zeros_like(sqrt_trace)
        )
        visibility = bbox_valid_mask.float().mean(dim=-1)  # (N,)
        tri_valid_float = tri_valid.float()  # (N,)

        # Mean agent-to-target distance
        agent_dists = []
        for aid in env.cfg.possible_agents:
            dist = torch.norm(env._root_pos_w[aid] - gt_target_pos, dim=-1)
            agent_dists.append(dist)
        distance = torch.stack(agent_dists, dim=-1).mean(dim=-1)

        # Target speed
        target_vel = env.target.data.root_lin_vel_w[:, :3]
        tgt_speed = torch.norm(target_vel, dim=-1)

        # Mean pairwise viewing angle
        ray_dirs = []
        for aid in env.cfg.possible_agents:
            to_target = gt_target_pos - env._root_pos_w[aid]
            ray_dirs.append(to_target / (to_target.norm(dim=-1, keepdim=True) + 1e-6))
        cos_angles = []
        for k in range(len(ray_dirs)):
            for j in range(k + 1, len(ray_dirs)):
                cos_angles.append((ray_dirs[k] * ray_dirs[j]).sum(dim=-1))
        cos_mean = torch.stack(cos_angles, dim=1).mean(dim=1)
        viewing_angle = torch.acos(cos_mean.clamp(-1, 1)) * (180.0 / torch.pi)

        active_mask = ~done_mask.squeeze(-1) if done_mask is not None else None
        ts_tracker.step(
            rmse=rmse,
            sqrt_trace=sqrt_trace,
            visibility=visibility,
            tri_valid=tri_valid_float,
            distance=distance,
            target_speed=tgt_speed,
            viewing_angle=viewing_angle,
            active_mask=active_mask,
            tri_valid_mask=tri_valid_with_cov,
            cbf_penalty=cbf_penalty,
        )


def _load_rnn_policy(checkpoint_path, env, agent_cfg, possible_agents, policy_combo="default"):
    """Load a trained MAPPO-RNN policy from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        env: Wrapped environment.
        agent_cfg: Agent configuration dict.
        possible_agents: List of agent IDs.
        policy_combo: Policy weight assignment. "default" uses the checkpoint as-is.
            "a0a0" loads drone_0 weights for both agents, "a0a1" loads drone_0/drone_1
            respectively, "a1a1" loads drone_1 for both.
    """
    from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    device = env.device
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    value_cfg = model_cfg.get("value", {})
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 32)

    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

    def _make_policy():
        return MAPPORNNPolicy(
            observation_space=env.observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=policy_cfg.get("hidden_size", 64),
            gru_num_layers=policy_cfg.get("gru_num_layers", 1),
            gru_hidden_size=policy_cfg.get("gru_hidden_size", 64),
            num_envs=env.num_envs,
            initial_log_std=policy_cfg.get("initial_log_std", -0.5),
            min_log_std=policy_cfg.get("min_log_std", -5.0),
            max_log_std=policy_cfg.get("max_log_std", 0.7),
            sequence_length=sequence_length,
        )

    def _make_value():
        return MAPPORNNValue(
            observation_space=shared_observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=value_cfg.get("hidden_size", 64),
            gru_num_layers=value_cfg.get("gru_num_layers", 1),
            gru_hidden_size=value_cfg.get("gru_hidden_size", 64),
            num_envs=env.num_envs,
            sequence_length=sequence_length,
        )

    # Build policy combo mapping: agent_id -> checkpoint drone key
    _COMBO_MAP = {
        "a0a0": {"drone_0": "drone_0", "drone_1": "drone_0"},
        "a0a1": {"drone_0": "drone_0", "drone_1": "drone_1"},
        "a1a1": {"drone_0": "drone_1", "drone_1": "drone_1"},
    }
    use_separate = policy_combo != "default" and policy_combo in _COMBO_MAP
    combo_map = _COMBO_MAP.get(policy_combo, {})

    if use_separate:
        # Create separate policy/value instances per agent
        models = {}
        for agent_id in possible_agents:
            models[agent_id] = {"policy": _make_policy(), "value": _make_value()}
    else:
        # Shared policy (default behavior)
        shared_policy = _make_policy()
        shared_value = _make_value()
        models = {aid: {"policy": shared_policy, "value": shared_value} for aid in possible_agents}

    memories = {}
    for agent_id in possible_agents:
        memories[agent_id] = RandomMemory(
            memory_size=agent_cfg.get("agent", {}).get("rollouts", 32),
            num_envs=env.num_envs,
            device=device,
        )

    # Setup preprocessors
    agent_cfg_copy = copy.deepcopy(agent_cfg)
    if "state_preprocessor" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["state_preprocessor"] == "RunningStandardScaler":
            agent_cfg_copy["agent"]["state_preprocessor"] = RunningStandardScaler
        if agent_cfg_copy["agent"].get("state_preprocessor_kwargs") is None:
            agent_cfg_copy["agent"]["state_preprocessor_kwargs"] = {}
        agent_cfg_copy["agent"]["state_preprocessor_kwargs"]["size"] = env.observation_spaces[possible_agents[0]]
        agent_cfg_copy["agent"]["state_preprocessor_kwargs"]["device"] = device
    if "shared_state_preprocessor" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["shared_state_preprocessor"] == "RunningStandardScaler":
            agent_cfg_copy["agent"]["shared_state_preprocessor"] = RunningStandardScaler
        if agent_cfg_copy["agent"].get("shared_state_preprocessor_kwargs") is None:
            agent_cfg_copy["agent"]["shared_state_preprocessor_kwargs"] = {}
        agent_cfg_copy["agent"]["shared_state_preprocessor_kwargs"]["size"] = shared_observation_spaces[possible_agents[0]]
        agent_cfg_copy["agent"]["shared_state_preprocessor_kwargs"]["device"] = device
    if "value_preprocessor" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["value_preprocessor"] == "RunningStandardScaler":
            agent_cfg_copy["agent"]["value_preprocessor"] = RunningStandardScaler
        if agent_cfg_copy["agent"].get("value_preprocessor_kwargs") is None:
            agent_cfg_copy["agent"]["value_preprocessor_kwargs"] = {}
        agent_cfg_copy["agent"]["value_preprocessor_kwargs"]["device"] = device
    if "learning_rate_scheduler" in agent_cfg_copy.get("agent", {}):
        if agent_cfg_copy["agent"]["learning_rate_scheduler"] == "KLAdaptiveLR":
            agent_cfg_copy["agent"]["learning_rate_scheduler"] = KLAdaptiveLR

    mappo_cfg = copy.deepcopy(MAPPO_RNN_DEFAULT_CONFIG)
    mappo_cfg.update(agent_cfg_copy.get("agent", {}))

    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if use_separate and "drone_0" in ckpt:
        # Policy combo: load per-agent weights from mapped drone keys
        for agent_id in possible_agents:
            src_key = combo_map[agent_id]
            src_data = ckpt[src_key]
            models[agent_id]["policy"].load_state_dict(src_data["policy"])
            models[agent_id]["value"].load_state_dict(src_data["value"])
        # Load preprocessor states via agent.load first, then overwrite model weights
        agent.load(checkpoint_path)
        for agent_id in possible_agents:
            src_key = combo_map[agent_id]
            src_data = ckpt[src_key]
            models[agent_id]["policy"].load_state_dict(src_data["policy"])
            models[agent_id]["value"].load_state_dict(src_data["value"])
            # Load preprocessor states from mapped drone
            for pp_key in ("state_preprocessor", "shared_state_preprocessor", "value_preprocessor"):
                pp_dict = getattr(agent, f"_{pp_key}", None)
                if isinstance(pp_dict, dict) and agent_id in pp_dict and pp_key in src_data:
                    pp_dict[agent_id].load_state_dict(src_data[pp_key])
        print(f"[EVAL] Loaded checkpoint with policy combo '{policy_combo}': "
              f"{{{', '.join(f'{aid}<-{combo_map[aid]}' for aid in possible_agents)}}}")
    elif "drone_0" in ckpt:
        agent.load(checkpoint_path)
        print(f"[EVAL] Loaded checkpoint (best_agent.pt SKRL): {checkpoint_path}")
    elif "policy_state_dict" in ckpt:
        shared_policy = models[possible_agents[0]]["policy"]
        shared_value = models[possible_agents[0]]["value"]
        shared_policy.load_state_dict(ckpt["policy_state_dict"])
        shared_value.load_state_dict(ckpt["value_state_dict"])
        print(f"[EVAL] Loaded checkpoint (final.pt): {checkpoint_path}")

        companion = os.path.join(os.path.dirname(checkpoint_path), "checkpoints", "best_agent.pt")
        if not os.path.exists(companion):
            companion = os.path.join(os.path.dirname(checkpoint_path), "best_agent.pt")
        if os.path.exists(companion):
            agent.load(companion)
            shared_policy.load_state_dict(ckpt["policy_state_dict"])
            shared_value.load_state_dict(ckpt["value_state_dict"])
            print(f"[EVAL] Loaded preprocessor state from: {companion}")
        else:
            print(f"[EVAL] WARNING: No preprocessor state found")
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

    agent.set_mode("eval")

    # Freeze preprocessors
    for uid in possible_agents:
        for key in ("_state_preprocessor", "_shared_state_preprocessor", "_value_preprocessor"):
            pp = getattr(agent, key, None)
            if isinstance(pp, dict) and uid in pp and hasattr(pp[uid], "eval"):
                pp[uid].eval()

    return agent


@hydra_task_config(args_cli.task, "skrl_mappo_rnn_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Evaluate a trained policy."""
    exp_cfg = _selected_experiment
    print(f"[EVAL] Experiment: {exp_cfg.name} — {exp_cfg.description}")
    print(f"[EVAL] Task: {args_cli.task}")

    if args_cli.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args_cli.output = f"./eval_{args_cli.experiment}_{timestamp}.json"

    # Apply experiment overrides
    if exp_cfg.env_overrides:
        apply_env_overrides(env_cfg, exp_cfg.env_overrides)
    if exp_cfg.agent_overrides:
        apply_agent_overrides(agent_cfg, exp_cfg.agent_overrides)

    # Apply runtime parameter overrides
    if args_cli.delay_override is not None:
        env_cfg.delay_system_params.ego_detection_latency_mean = args_cli.delay_override
        env_cfg.delay_system_params.ego_detection_latency_std = 0.001
        print(f"[EVAL] Detection delay override: {args_cli.delay_override*1000:.0f}ms")
    if args_cli.comm_delay_override is not None:
        env_cfg.delay_system_params.other_latency_mean = args_cli.comm_delay_override
        env_cfg.delay_system_params.other_latency_std = 0.001
        print(f"[EVAL] Comm delay override: {args_cli.comm_delay_override*1000:.0f}ms")
    if args_cli.target_speed is not None:
        env_cfg.max_lin_vel = args_cli.target_speed
        print(f"[EVAL] Target speed override: {args_cli.target_speed:.1f} m/s")
    if args_cli.noise_override is not None:
        env_cfg.delay_system_params.noise_bbox_std = args_cli.noise_override
        print(f"[EVAL] Pixel noise override: {args_cli.noise_override:.1f} px")
    if args_cli.detection_dropout_override is not None:
        env_cfg.delay_system_params.dropout_prob = args_cli.detection_dropout_override
        print(f"[EVAL] Detection dropout override: {args_cli.detection_dropout_override:.2f}")

    # Apply seed
    torch.manual_seed(args_cli.seed)
    env_cfg.seed = args_cli.seed
    print(f"[EVAL] Seed: {args_cli.seed}")

    # Apply target trajectory mode override
    if args_cli.target_trajectory_mode is not None:
        if args_cli.target_trajectory_mode == "linear":
            env_cfg.target_controller.linear_weight = 1.0
        else:  # circular
            env_cfg.target_controller.linear_weight = 0.0
        print(f"[EVAL] Target trajectory mode: {args_cli.target_trajectory_mode} "
              f"(linear_weight={env_cfg.target_controller.linear_weight})")

    # Set eval params
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 20.0

    if args_cli.step is not None:
        # Pin curriculum to a specific training step (more intuitive than zeroing
        # start/end steps). The env reads debug_initial_step in place of
        # common_step_counter for all curriculum lookups when use_debug_initial_step
        # is True.
        env_cfg.use_debug_initial_step = True
        env_cfg.debug_initial_step = int(args_cli.step)
        print(f"[EVAL] Curriculum pinned to step {args_cli.step} via debug_initial_step")
    else:
        # Force all curriculum to full difficulty (legacy behavior when --step is unset)
        env_cfg.curriculum.tracking_start_step = 0
        env_cfg.curriculum.tracking_end_step = 0
        env_cfg.curriculum.safety_start_step = 0
        env_cfg.curriculum.safety_end_step = 0
        env_cfg.curriculum.moving_target_start_step = 0
        env_cfg.curriculum.moving_target_end_step = 0
        env_cfg.curriculum.coordination_start_step = 0
        env_cfg.curriculum.coordination_end_step = 0
        env_cfg.curriculum.noise_start_step = 0
        env_cfg.curriculum.noise_end_step = 0
        env_cfg.curriculum.fixed_delay_start_step = 0
        env_cfg.curriculum.fixed_delay_end_step = 0
        env_cfg.curriculum.random_delay_start_step = 0
        env_cfg.curriculum.random_delay_end_step = 0
        env_cfg.curriculum.dropout_start_step = 0
        env_cfg.curriculum.dropout_end_step = 0
        env_cfg.curriculum.dynamics_start_step = 0
        env_cfg.curriculum.dynamics_end_step = 0
        env_cfg.curriculum.agent_velocity_start_step = 0
        env_cfg.curriculum.agent_velocity_end_step = 0

    # Create environment
    # Scene video: render the env's own viewport (env_cfg.viewer) to rgb_array and
    # write frames MANUALLY in the rollout loop. We do NOT use gymnasium RecordVideo
    # here — it is a single-env wrapper that stops the moment the vectorized env
    # signals done (it captured only 2 warmup-black frames). Manual capture of
    # unwrapped.render() uses the env's own render product (which works once the
    # warp 'owner' shim is applied) and grabs every frame regardless of episode.
    _scene_video = args_cli.record_video_scene is not None
    if _scene_video:
        # Render from a camera WE create (/World/VideoCam). The default
        # /OmniverseKit_Persp is a Kit-managed viewport camera whose transform we
        # cannot move headless (USD writes ignored, set_camera_view guarded off), so
        # we point env.render()'s render product at our own stage-prim camera, which
        # we then drive per-frame. Force the camera-frustum debug visuals on.
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.cam_prim_path = "/World/VideoCam"
        if hasattr(env_cfg, "debug_vis"):
            env_cfg.debug_vis = True
        env_cfg.viewer.resolution = tuple(args_cli.video_resolution)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        print(f"[EVAL] Scene video (manual env.render capture) -> "
              f"{args_cli.record_video_scene} (isometric view of env 0)")
    else:
        env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env)
    unwrapped = env.unwrapped

    possible_agents = env_wrapped.possible_agents
    num_agents = len(possible_agents)

    tracker = MetricTracker(
        num_envs=args_cli.num_envs,
        num_agents=num_agents,
        device=unwrapped.device,
    )

    if not args_cli.no_timeseries:
        ts_tracker = TimeseriesTracker(
            max_episode_steps=unwrapped.max_episode_length,
            num_envs=args_cli.num_envs,
            device=unwrapped.device,
        )
    else:
        ts_tracker = None

    # Trajectory recorder
    if args_cli.record_trajectory:
        from isaaclab_tasks.direct.iris_ma6.experiments.metrics import TrajectoryRecorder
        traj_recorder = TrajectoryRecorder(
            num_sample_envs=args_cli.trajectory_envs,
            num_total_envs=args_cli.num_envs,
            agent_ids=list(possible_agents),
            max_steps=unwrapped.max_episode_length,
        )
    else:
        traj_recorder = None

    # Action diagnostics recorder (per-axis RMS, ΔRMS, peak FFT freq; cmd_vel
    # dims 4-6 are normalized in env, scaled to physical units here).
    action_trace_recorder = ActionTraceRecorder(
        num_sample_envs=args_cli.trajectory_envs,
        num_total_envs=args_cli.num_envs,
        agent_ids=list(possible_agents),
        max_steps=unwrapped.max_episode_length,
        policy_rate_hz=1.0 / float(unwrapped.step_dt),
        gimbal_rate_scale=unwrapped._controller.cfg.gimbal.max_gimbal_rate,
        zoom_rate_scale=unwrapped._controller._zoom._max_zoom_rate.clone(),
        record_trace=args_cli.record_action_trace,
    )

    # Video recorder
    if args_cli.record_video is not None:
        camera_cfg = get_preset(args_cli.camera_mode)
        camera_cfg.smoothing_factor = args_cli.camera_smoothing

        smooth_camera = SmoothCameraController(
            viewport_controller=None if _headless_video else getattr(unwrapped, "viewport_camera_controller", None),
            cfg=camera_cfg,
            env_index=0,
        )
        video_recorder = VideoRecorder(
            cfg=VideoRecorderCfg(
                output_path=args_cli.record_video,
                fps=args_cli.video_fps,
                resolution=tuple(args_cli.video_resolution),
                headless=_headless_video,
            )
        )
        video_recorder.start()
        if _headless_video:
            video_recorder.setup_offscreen_camera(unwrapped)
    else:
        smooth_camera = None
        video_recorder = None

    # Load policy
    is_greedy = (args_cli.experiment == "baseline_greedy")
    if is_greedy:
        from isaaclab_tasks.direct.iris_ma6.experiments.baselines.greedy_policy import (
            GreedyEquiangularPolicy,
        )
        policy = GreedyEquiangularPolicy(
            num_agents=num_agents, device=str(unwrapped.device)
        )
        print(f"[EVAL] Using greedy equiangular policy")
    else:
        if args_cli.checkpoint is None:
            raise ValueError("--checkpoint required for trained policy evaluation")
        policy = _load_rnn_policy(
            args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents,
            policy_combo=args_cli.policy_combo,
        )

    # Run evaluation
    completed = 0
    total_target = args_cli.num_episodes * args_cli.num_envs
    obs, info = env_wrapped.reset()

    # Scene-video writer (manual capture of env.render()). record_video_scene is a
    # file path (.mp4). Warm up the RTX renderer first — the first frames are black
    # until the path tracer converges.
    _scene_writer = None
    if _scene_video:
        import cv2 as _cv2
        _scene_w, _scene_h = tuple(args_cli.video_resolution)
        _sv_dir = os.path.dirname(os.path.abspath(args_cli.record_video_scene))
        if _sv_dir:
            os.makedirs(_sv_dir, exist_ok=True)
        _scene_writer = _cv2.VideoWriter(
            args_cli.record_video_scene, _cv2.VideoWriter_fourcc(*"mp4v"),
            float(args_cli.video_fps), (_scene_w, _scene_h))
        # Create our own camera prim (env.render() will render from it) and aim it
        # at env 0 before the render product is created on the first render() call.
        _cam_path = "/World/VideoCam"
        _scene_origin = unwrapped.scene.env_origins[0].detach().cpu().numpy()
        _create_video_camera(_cam_path)
        _aim_camera_prim(_cam_path, _scene_origin + np.array([16.0, -28.0, 44.0]),
                         _scene_origin + np.array([-2.0, 2.0, 21.0]))
        print("[EVAL] Warming up RTX renderer for scene video ...")
        for _ in range(40):
            unwrapped.sim.render()
            unwrapped.render()
        print(f"[EVAL] Scene writer ready ({_scene_w}x{_scene_h} @ {args_cli.video_fps}fps)")

    print(f"[EVAL] Starting: {args_cli.num_episodes} ep/env x {args_cli.num_envs} envs = {total_target} total")

    step_count = 0
    episodes_done = False

    needs_full_episode = ts_tracker is not None or traj_recorder is not None
    max_steps_for_ts = unwrapped.max_episode_length if needs_full_episode else 0

    while completed < total_target or step_count < max_steps_for_ts:
        # Compute actions
        if is_greedy:
            agent_positions = {
                aid: unwrapped._root_pos_w[aid]
                for aid in possible_agents
            }
            target_pos = unwrapped._target_pos_w
            agent_orientations = {
                aid: unwrapped._root_quat_w[aid]
                for aid in possible_agents
            }
            actions = policy.compute_actions(agent_positions, target_pos, agent_orientations)
        else:
            with torch.no_grad():
                actions, _, outputs = policy.act(obs, 0, 0)
                # Use mean actions for deterministic evaluation
                for agent_id in possible_agents:
                    actions[agent_id] = outputs[agent_id]["mean_actions"]

        step_count += 1

        obs, rewards, terminated, truncated, info = env_wrapped.step(actions)

        # Record per-step action commands (physical units; dims 4-6 scaled inside recorder)
        action_trace_recorder.step(unwrapped.cmd_vel)

        # Record video frame
        if video_recorder is not None:
            _vid_agents = torch.stack(
                [unwrapped._root_pos_w[aid][0, :3] for aid in possible_agents], dim=0
            )
            _vid_target = unwrapped.target.data.root_pos_w[0, :3]
            smooth_camera.update(_vid_agents, _vid_target, dt=unwrapped.step_dt)
            if _headless_video:
                _eye, _lookat = smooth_camera.get_current_pose()
                video_recorder.set_camera_pose(_eye, _lookat)
            video_recorder.capture_frame(unwrapped)

        # Scene video: follow the agents+target centroid (zoom-to-fit) by driving our
        # own /World/VideoCam prim, then capture — drone USD models + frustum visuals.
        if _scene_writer is not None:
            _pts = torch.stack(
                [unwrapped._root_pos_w[aid][0, :3] for aid in possible_agents]
                + [unwrapped.target.data.root_pos_w[0, :3]], dim=0)
            _origin = unwrapped.scene.env_origins[0].detach().cpu().numpy()
            _loc = (_pts - unwrapped.scene.env_origins[0]).detach().cpu().numpy()
            _c = _loc.mean(axis=0)
            _radius = float(np.linalg.norm(_loc - _c, axis=1).max())
            _dist = max(16.0, _radius * 1.8 + 8.0)
            _dir = np.array([0.78, -0.6, 0.18]); _dir = _dir / np.linalg.norm(_dir)
            _aim_camera_prim("/World/VideoCam", _origin + _c + _dir * _dist, _origin + _c)
            # iris_ma6 has RTX sensors, so unwrapped.render() does NOT call sim.render()
            # (it assumes the step already rendered) — our camera move would be ignored.
            # Force a re-render at the new pose before reading the annotator.
            unwrapped.sim.render()
            _frame = unwrapped.render()
            if _frame is not None:
                _frame = np.asarray(_frame)
                if _frame.ndim == 3 and _frame.shape[2] >= 3:
                    _bgr = _cv2.cvtColor(_frame[..., :3].astype(np.uint8), _cv2.COLOR_RGB2BGR)
                    if (_bgr.shape[1], _bgr.shape[0]) != (_scene_w, _scene_h):
                        _bgr = _cv2.resize(_bgr, (_scene_w, _scene_h))
                    _scene_writer.write(_bgr)

        # Detect done envs
        first_agent = possible_agents[0]
        if isinstance(terminated, dict):
            done_mask = terminated[first_agent] | truncated[first_agent]
        else:
            done_mask = terminated | truncated

        # Collect metrics
        if not episodes_done:
            _collect_step_metrics(tracker, unwrapped, ts_tracker, done_mask)
        elif ts_tracker is not None:
            _collect_step_metrics(tracker=None, env=unwrapped, ts_tracker=ts_tracker, done_mask=done_mask)

        # Record trajectories
        if traj_recorder is not None and not traj_recorder.is_done():
            _traj_agent_pos = {
                aid: unwrapped._root_pos_w[aid]
                for aid in possible_agents
            }
            _traj_target_pos = unwrapped._target_pos_w

            if unwrapped._triangulation_result_obs is not None:
                _tri = unwrapped._triangulation_result_obs
                _traj_tri_pos = _tri.position[:, 0, :]
                _traj_tri_valid = _tri.is_valid[:, 0]
            else:
                _traj_tri_pos = torch.zeros(args_cli.num_envs, 3, device=unwrapped.device)
                _traj_tri_valid = torch.zeros(args_cli.num_envs, device=unwrapped.device, dtype=torch.bool)

            # Viewing angle
            _ray_dirs = []
            for aid in possible_agents:
                _to_tgt = _traj_target_pos - unwrapped._root_pos_w[aid]
                _ray_dirs.append(_to_tgt / (_to_tgt.norm(dim=-1, keepdim=True) + 1e-6))
            _cos_angles = []
            for _k in range(len(_ray_dirs)):
                for _j in range(_k + 1, len(_ray_dirs)):
                    _cos_angles.append((_ray_dirs[_k] * _ray_dirs[_j]).sum(dim=-1))
            _cos_mean = torch.stack(_cos_angles, dim=1).mean(dim=1)
            _traj_viewing_angle = torch.acos(_cos_mean.clamp(-1, 1)) * (180.0 / torch.pi)

            _traj_rmse = torch.norm(_traj_tri_pos - _traj_target_pos, dim=-1)

            # Extract gimbal angles and zoom levels
            _traj_gimbal_angles = {}
            _traj_zoom_levels = {}
            for _i, aid in enumerate(possible_agents):
                # Batched controller: agent _i occupies rows [_i*N, (_i+1)*N)
                _s = _i * unwrapped.num_envs
                _e = _s + unwrapped.num_envs
                gimbal = unwrapped._controller._gimbal
                _traj_gimbal_angles[aid] = torch.stack(
                    [gimbal._yaw[_s:_e], gimbal._pitch[_s:_e]], dim=-1
                )  # (N, 2)
                _traj_zoom_levels[aid] = unwrapped.zoom_level[:, _i]  # (N,)

            # Extract velocities
            _traj_agent_vel = {
                aid: unwrapped._root_lin_vel_w[aid]
                for aid in possible_agents
            }
            _traj_target_vel = unwrapped.target.data.root_lin_vel_w[:, :3]

            # Target position in each camera's image: normalized (u,v) center in
            # [0,1] (0.5,0.5 = centered, matches the bbox_center reward) + validity.
            _traj_target_bbox = {}
            for _i, aid in enumerate(possible_agents):
                _bb_px = unwrapped.bbox_raycaster_v2.data.bboxes[:, _i, 0, :]  # (N,4) cx,cy,w,h px
                _uv = _bb_px[:, 0:2] / unwrapped._img_dims                      # (N,2) normalized
                _bvalid = (_bb_px.abs().sum(dim=-1) > 1e-6).float().unsqueeze(-1)
                _traj_target_bbox[aid] = torch.cat([_uv, _bvalid], dim=-1)      # (N,3)

            traj_recorder.step(
                agent_positions=_traj_agent_pos,
                target_pos=_traj_target_pos,
                tri_pos=_traj_tri_pos,
                tri_valid=_traj_tri_valid,
                rmse=_traj_rmse,
                viewing_angle=_traj_viewing_angle,
                env_origins=unwrapped._terrain.env_origins,
                gimbal_angles=_traj_gimbal_angles,
                zoom_levels=_traj_zoom_levels,
                agent_velocities=_traj_agent_vel,
                target_vel=_traj_target_vel,
                target_bbox=_traj_target_bbox,
            )

        # Handle episode completions
        done_envs = torch.where(done_mask.squeeze(-1) if done_mask.dim() > 1 else done_mask)[0]
        if done_envs.numel() > 0:
            if not episodes_done:
                tracker.record_episode_end(done_envs)
                completed += done_envs.numel()
                if completed >= total_target:
                    episodes_done = True
                    print(f"[EVAL] Completed {completed}/{total_target} episodes at step {step_count}")
                elif completed % max(total_target // 10, 1) == 0:
                    print(f"[EVAL] Completed {completed}/{total_target} episodes")
            if ts_tracker is not None:
                ts_tracker.record_episode_end(done_envs)
            if traj_recorder is not None:
                traj_recorder.record_episode_end(done_envs)
            action_trace_recorder.record_episode_end(done_envs)

    # Compute final results
    results = tracker.compute_final_metrics()
    if ts_tracker is not None:
        ts_data = ts_tracker.compute_timeseries()
        ts_data["step_dt_seconds"] = float(unwrapped.step_dt)
        results["timeseries"] = ts_data
    if traj_recorder is not None:
        traj_data = traj_recorder.to_dict()
        traj_data["step_dt_seconds"] = float(unwrapped.step_dt)
        results["trajectories"] = traj_data
    results["action_diagnostics"] = action_trace_recorder.to_dict()
    results["experiment"] = exp_cfg.name
    results["checkpoint"] = args_cli.checkpoint or "greedy"
    results["eval_params"] = {
        "num_episodes": args_cli.num_episodes,
        "num_envs": args_cli.num_envs,
        "delay_override": args_cli.delay_override,
        "comm_delay_override": args_cli.comm_delay_override,
        "target_speed": args_cli.target_speed,
        "noise_override": args_cli.noise_override,
        "detection_dropout_override": args_cli.detection_dropout_override,
    }

    # Save results
    if args_cli.output:
        output_path = os.path.abspath(args_cli.output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    if video_recorder is not None:
        video_recorder.stop()
        print(f"Video saved to: {args_cli.record_video}")

    if _scene_writer is not None:
        _scene_writer.release()
        print(f"Scene video saved to: {args_cli.record_video_scene}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
