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

parser = argparse.ArgumentParser(description="Evaluate trained policy with paper metrics (iris_ma6).")
parser.add_argument("--experiment", type=str, required=True, help="Experiment name from registry")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
parser.add_argument("--num_episodes", type=int, default=1, help="Episodes per env")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel eval envs")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA6-Direct-Test-v0")
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
parser.add_argument("--target-trajectory-mode", type=str, default=None,
                    choices=["linear", "circular"],
                    help="Force target trajectory mode (overrides linear_weight)")
parser.add_argument("--policy-combo", type=str, default="default",
                    choices=["default", "a0a0", "a0a1", "a1a1"],
                    help="Policy weight assignment: which drone's weights each agent uses")
# Trajectory recording
parser.add_argument("--record-trajectory", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--trajectory-envs", type=int, default=8)
# Video recording
parser.add_argument("--record-video", type=str, default=None,
                    help="Output path for video (e.g., demo.mp4)")
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

sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher

# Video recording: headless mode with offscreen camera
_headless_video = args_cli.record_video is not None and args_cli.headless
_enable_cameras = args_cli.enable_cameras
if _headless_video and not _enable_cameras:
    _enable_cameras = True
    print("[EVAL] Auto-enabling cameras for headless video recording")
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

"""Rest follows after Isaac Sim is initialized."""

import os
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
from isaaclab_tasks.direct.iris_ma6.experiments.metrics import MetricTracker, TimeseriesTracker

if args_cli.record_video is not None:
    from isaaclab_tasks.direct.iris_ma5.video_recording import (
        SmoothCameraController, VideoRecorder, VideoRecorderCfg, get_preset,
    )

# Add skrl scripts dir to path
_skrl_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
                                  "scripts", "reinforcement_learning", "skrl")
_skrl_scripts_dir = os.path.normpath(_skrl_scripts_dir)
sys.path.insert(0, _skrl_scripts_dir)


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
    exp_cfg = get_experiment(args_cli.experiment)
    print(f"[EVAL] Experiment: {exp_cfg.name} — {exp_cfg.description}")

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

    # Force all curriculum to full difficulty
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

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
