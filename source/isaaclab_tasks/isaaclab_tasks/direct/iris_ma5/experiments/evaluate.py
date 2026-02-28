#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained checkpoint and compute the 8 IROS 2026 paper metrics.

Usage:
    # Evaluate a trained policy
    ./isaaclab.sh -p .../experiments/evaluate.py \\
        --experiment a1_stochastic_delay \\
        --checkpoint /path/to/agent_drone_0_final.pt \\
        --num_episodes 100 --output results.json

    # Evaluate greedy baseline (no checkpoint needed)
    ./isaaclab.sh -p .../experiments/evaluate.py \\
        --experiment baseline_greedy --num_episodes 100

    # Evaluation with runtime parameter sweep (delay override)
    ./isaaclab.sh -p .../experiments/evaluate.py \\
        --experiment a1_with_aoi \\
        --checkpoint /path/to/best_agent.pt \\
        --delay-override 0.1 --output results_100ms.json

Supports two checkpoint formats:
    - agent_drone_0_final.pt: {policy_state_dict, value_state_dict}
    - best_agent.pt (SKRL): {drone_0: {policy, value, ...}, ...}
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Evaluate trained policy with 8 paper metrics.")
parser.add_argument("--experiment", type=str, required=True, help="Experiment name from registry")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
parser.add_argument("--num_episodes", type=int, default=1, help="Episodes per env (total = num_episodes * num_envs)")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel eval envs")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA5-Direct-v0")
parser.add_argument("--output", type=str, default=None, help="Output JSON path for results")
parser.add_argument("--headless", action="store_true", default=True)
# Runtime parameter overrides for evaluation sweeps
parser.add_argument("--delay-override", type=float, default=None,
                    help="Override detection latency in seconds (e.g., 0.1 for 100ms)")
parser.add_argument("--comm-delay-override", type=float, default=None,
                    help="Override communication latency in seconds")
parser.add_argument("--target-speed", type=float, default=None,
                    help="Override target speed in m/s")
parser.add_argument("--noise-override", type=float, default=None,
                    help="Override pixel noise std (pix_std) in pixels")
parser.add_argument("--accel-override", type=float, default=None,
                    help="Override target max acceleration in m/s^2")
parser.add_argument("--detection-dropout-override", type=float, default=None,
                    help="Override detection failure rate (0-1)")
parser.add_argument("--comm-dropout-override", type=float, default=None,
                    help="Override communication dropout rate (0-1)")
parser.add_argument("--no-timeseries", action="store_true", default=False,
                    help="Disable per-timestep timeseries collection")
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv so Hydra only sees its own arguments
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher

app_args = argparse.Namespace(headless=args_cli.headless, device="cuda:0", experience="")
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is initialized."""

import os
import torch
import gymnasium as gym
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.iris_ma5.experiments import (
    get_experiment,
    apply_env_overrides,
    apply_agent_overrides,
)
from isaaclab_tasks.direct.iris_ma5.experiments.metrics import MetricTracker, TimeseriesTracker

# Add skrl scripts dir to path for mappo_rnn imports
_skrl_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
                                  "scripts", "reinforcement_learning", "skrl")
_skrl_scripts_dir = os.path.normpath(_skrl_scripts_dir)
sys.path.insert(0, _skrl_scripts_dir)


def _collect_step_metrics(
    tracker: MetricTracker,
    env,
    ts_tracker: "TimeseriesTracker | None" = None,
    done_mask: "torch.Tensor | None" = None,
) -> None:
    """Extract per-step metrics from environment internals.

    Uses the noisy-path triangulation for position estimate and trace, since
    that reflects actual system performance. RMSE compares the midpoint-method
    triangulation estimate against ground truth target position.

    The rewards-path ``triangulation_results`` stores ``X_w_gt`` (ground truth)
    at index 0, not an estimate, so it cannot be used for RMSE.

    Args:
        tracker: Episode-level metric tracker.
        env: Unwrapped environment instance.
        ts_tracker: Optional timeseries tracker for per-timestep stats.
        done_mask: Optional [N, 1] bool tensor — True for envs that just
            finished (post-reset). Used to exclude them from timeseries.
    """
    first_agent = env.cfg.possible_agents[0]

    # Get triangulation results from the noisy (observation) path.
    # Index 0 = X_w_triangulated (midpoint-method estimate), not GT.
    if hasattr(env, "triangulation_results_noisy") and first_agent in env.triangulation_results_noisy:
        X_w_tri, Sigma_X, std_dev, trace_cov, is_valid = env.triangulation_results_noisy[first_agent]
    else:
        trace_cov = torch.ones(env.num_envs, 1, device=env.device) * 999.0
        X_w_tri = torch.zeros(env.num_envs, 1, 3, device=env.device)
        is_valid = torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.bool)

    # Ground truth target position
    gt_target_pos = env.target.data.root_pos_w

    # Triangulated position estimate (from noisy path midpoint method)
    tri_pos = X_w_tri.squeeze(1) if X_w_tri.dim() == 3 else X_w_tri

    # Bbox validity for all agents (from current observation states)
    bbox_valid_list = []
    for aid in env.cfg.possible_agents:
        states = env.delay_system.get_all_states_for_observations(aid)
        bbox = states[aid].data.bboxes_2d[:, 0, :]
        valid = env.bbox_raycaster.validate_bbox(bbox).squeeze(-1)
        bbox_valid_list.append(valid)
    bbox_valid_mask = torch.stack(bbox_valid_list, dim=1)

    # Collision detection — any pair in collision → per-env flag [N]
    coll_mat = env.safety_manager.get_collision_matrix()
    if coll_mat is not None:
        collision_flags = coll_mat.any(dim=-1).any(dim=-1)  # [N, A, A] → [N]
    else:
        collision_flags = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    tracker.step(
        trace_sigma=trace_cov,
        triangulated_pos=tri_pos,
        gt_target_pos=gt_target_pos,
        bbox_valid_mask=bbox_valid_mask,
        collision_flags=collision_flags,
        tri_valid=is_valid,
    )

    # Feed timeseries tracker with derived per-env metrics
    if ts_tracker is not None:
        rmse = torch.norm(tri_pos[:, :3] - gt_target_pos[:, :3], dim=-1)  # [N]
        sqrt_trace = torch.sqrt(trace_cov.squeeze(-1).clamp(min=0))  # [N]
        visibility = bbox_valid_mask.float().mean(dim=-1)  # [N]
        tri_valid_float = is_valid.squeeze(-1).float()  # [N]

        # Mean agent-to-target distance
        agent_dists = []
        for aid in env.cfg.possible_agents:
            agent_pos = env._robots[aid].data.root_pos_w[:, :3]
            dist = torch.norm(agent_pos - gt_target_pos[:, :3], dim=-1)
            agent_dists.append(dist)
        distance = torch.stack(agent_dists, dim=-1).mean(dim=-1)  # [N]

        # Target ground-truth speed
        target_vel = env.target.data.root_lin_vel_w[:, :3]  # [N, 3]
        tgt_speed = torch.norm(target_vel, dim=-1)  # [N]

        active_mask = ~done_mask.squeeze(-1) if done_mask is not None else None
        ts_tracker.step(
            rmse=rmse,
            sqrt_trace=sqrt_trace,
            visibility=visibility,
            tri_valid=tri_valid_float,
            distance=distance,
            target_speed=tgt_speed,
            active_mask=active_mask,
            tri_valid_mask=is_valid.squeeze(-1),
        )


def _load_rnn_policy(checkpoint_path, env, agent_cfg, possible_agents):
    """Load a trained MAPPO-RNN policy from checkpoint."""
    from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    import copy

    device = env.device
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    value_cfg = model_cfg.get("value", {})
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 32)

    # Shared obs space
    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

    shared_policy = MAPPORNNPolicy(
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
    shared_value = MAPPORNNValue(
        observation_space=shared_observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=value_cfg.get("hidden_size", 64),
        gru_num_layers=value_cfg.get("gru_num_layers", 1),
        gru_hidden_size=value_cfg.get("gru_hidden_size", 64),
        num_envs=env.num_envs,
        sequence_length=sequence_length,
    )

    models = {}
    memories = {}
    for agent_id in possible_agents:
        models[agent_id] = {"policy": shared_policy, "value": shared_value}
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

    # Load checkpoint — use agent.load() for best_agent.pt (SKRL format)
    # which includes preprocessor state. For final.pt, load policy/value manually
    # and also load preprocessor from companion best_agent.pt if available.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "drone_0" in ckpt:
        # best_agent.pt (SKRL multi-agent format) — use agent.load() for full state
        agent.load(checkpoint_path)
        print(f"[EVAL] Loaded checkpoint (best_agent.pt SKRL, with preprocessors): {checkpoint_path}")
    elif "policy_state_dict" in ckpt:
        # agent_drone_0_final.pt format — manual policy/value load
        shared_policy.load_state_dict(ckpt["policy_state_dict"])
        shared_value.load_state_dict(ckpt["value_state_dict"])
        print(f"[EVAL] Loaded checkpoint (final.pt): {checkpoint_path}")

        # Try to load preprocessor state from companion best_agent.pt
        companion = os.path.join(os.path.dirname(checkpoint_path), "checkpoints", "best_agent.pt")
        if not os.path.exists(companion):
            companion = os.path.join(os.path.dirname(checkpoint_path), "best_agent.pt")
        if os.path.exists(companion):
            agent.load(companion)
            # Re-load policy/value from final.pt (agent.load overwrote with best_agent weights)
            shared_policy.load_state_dict(ckpt["policy_state_dict"])
            shared_value.load_state_dict(ckpt["value_state_dict"])
            print(f"[EVAL] Loaded preprocessor state from: {companion}")
        else:
            print(f"[EVAL] WARNING: No preprocessor state found — observations may not be normalized correctly")
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

    agent.set_mode("eval")
    return agent


def _load_mlp_policy(checkpoint_path, env, agent_cfg, possible_agents):
    """Load a trained MAPPO-MLP policy from checkpoint."""
    from isaaclab_tasks.direct.iris_ma5.experiments.models.mappo_mlp import (
        MAPPOMLPPolicy,
        MAPPOMLPValue,
    )
    from isaaclab_tasks.direct.iris_ma5.experiments.models.mappo_mlp_agent import (
        MAPPO_MLP,
        MAPPO_MLP_DEFAULT_CONFIG,
    )
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    import copy

    device = env.device
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    value_cfg = model_cfg.get("value", {})

    # Shared obs space
    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

    shared_policy = MAPPOMLPPolicy(
        observation_space=env.observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=policy_cfg.get("hidden_size", 256),
        num_hidden_layers=policy_cfg.get("num_hidden_layers", 3),
        num_envs=env.num_envs,
        initial_log_std=policy_cfg.get("initial_log_std", -0.5),
        min_log_std=policy_cfg.get("min_log_std", -5.0),
        max_log_std=policy_cfg.get("max_log_std", 0.7),
    )
    shared_value = MAPPOMLPValue(
        observation_space=shared_observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=value_cfg.get("hidden_size", 256),
        num_hidden_layers=value_cfg.get("num_hidden_layers", 3),
        num_envs=env.num_envs,
    )

    models = {}
    memories = {}
    for agent_id in possible_agents:
        models[agent_id] = {"policy": shared_policy, "value": shared_value}
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

    mappo_cfg = copy.deepcopy(MAPPO_MLP_DEFAULT_CONFIG)
    mappo_cfg.update(agent_cfg_copy.get("agent", {}))

    agent = MAPPO_MLP(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )

    # Load checkpoint — same logic as RNN loader
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "drone_0" in ckpt:
        agent.load(checkpoint_path)
        print(f"[EVAL] Loaded MLP checkpoint (best_agent.pt SKRL, with preprocessors): {checkpoint_path}")
    elif "policy_state_dict" in ckpt:
        shared_policy.load_state_dict(ckpt["policy_state_dict"])
        shared_value.load_state_dict(ckpt["value_state_dict"])
        print(f"[EVAL] Loaded MLP checkpoint (final.pt): {checkpoint_path}")

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
    return agent


@hydra_task_config(args_cli.task, "skrl_mappo_rnn_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Evaluate a trained policy."""
    exp_cfg = get_experiment(args_cli.experiment)
    print(f"[EVAL] Experiment: {exp_cfg.name} — {exp_cfg.description}")

    # Apply experiment overrides
    if exp_cfg.env_overrides:
        apply_env_overrides(env_cfg, exp_cfg.env_overrides)
    if exp_cfg.agent_overrides:
        apply_agent_overrides(agent_cfg, exp_cfg.agent_overrides)

    # Apply runtime parameter overrides for evaluation sweeps
    if args_cli.delay_override is not None:
        env_cfg.detection_mean_latency = args_cli.delay_override
        env_cfg.detection_std_latency = 0.001  # near-deterministic
        print(f"[EVAL] Detection delay override: {args_cli.delay_override*1000:.0f}ms")
    if args_cli.comm_delay_override is not None:
        env_cfg.comm_mean_latency = args_cli.comm_delay_override
        env_cfg.comm_std_latency = 0.001
        print(f"[EVAL] Comm delay override: {args_cli.comm_delay_override*1000:.0f}ms")
    if args_cli.target_speed is not None:
        env_cfg.target_movement.max_speed = args_cli.target_speed
        print(f"[EVAL] Target speed override: {args_cli.target_speed:.1f} m/s")
    if args_cli.noise_override is not None:
        env_cfg.pix_std = args_cli.noise_override
        print(f"[EVAL] Pixel noise override: {args_cli.noise_override:.1f} px")
    if args_cli.accel_override is not None:
        env_cfg.target_movement.max_acceleration = args_cli.accel_override
        print(f"[EVAL] Target acceleration override: {args_cli.accel_override:.1f} m/s^2")
    if args_cli.detection_dropout_override is not None:
        env_cfg.detection_failure_rate = args_cli.detection_dropout_override
        print(f"[EVAL] Detection dropout override: {args_cli.detection_dropout_override:.2f}")
    if args_cli.comm_dropout_override is not None:
        env_cfg.comm_dropout_rate = args_cli.comm_dropout_override
        print(f"[EVAL] Comm dropout override: {args_cli.comm_dropout_override:.2f}")

    # Set eval params
    env_cfg.scene.num_envs = args_cli.num_envs

    # For evaluation, set all curriculum to full difficulty
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
    env_cfg.eval_mode = True

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env)
    unwrapped = env.unwrapped

    # Apply frame-skip stacking for MLP evaluation
    if exp_cfg.frame_stack > 1:
        from isaaclab_tasks.direct.iris_ma5.experiments.frame_stack_wrapper import MultiAgentFrameStackWrapper
        env_wrapped = MultiAgentFrameStackWrapper(env_wrapped, num_stack=exp_cfg.frame_stack, frame_skip=exp_cfg.frame_skip)
        print(f"[EVAL] Frame stacking: K={exp_cfg.frame_stack}, skip={exp_cfg.frame_skip}")

    is_multi_agent = isinstance(unwrapped, DirectMARLEnv)
    if is_multi_agent:
        possible_agents = env_wrapped.possible_agents
    else:
        raise ValueError("Evaluation currently only supports multi-agent environments")

    num_agents = len(possible_agents)
    tracker = MetricTracker(
        num_envs=args_cli.num_envs,
        num_agents=num_agents,
        device=unwrapped.device,
    )

    # Timeseries tracker for per-timestep statistics
    if not args_cli.no_timeseries:
        ts_tracker = TimeseriesTracker(
            max_episode_steps=unwrapped.max_episode_length,
            num_envs=args_cli.num_envs,
            device=unwrapped.device,
        )
    else:
        ts_tracker = None

    # Load policy — route based on experiment type
    is_greedy = (args_cli.experiment == "baseline_greedy")
    if is_greedy:
        from isaaclab_tasks.direct.iris_ma5.experiments.baselines.greedy_policy import (
            GreedyEquiangularPolicy,
        )
        policy = GreedyEquiangularPolicy(
            num_agents=num_agents, device=str(unwrapped.device)
        )
        print(f"[EVAL] Using greedy equiangular policy")
    else:
        if args_cli.checkpoint is None:
            raise ValueError("--checkpoint required for trained policy evaluation")
        if exp_cfg.use_mlp_model:
            policy = _load_mlp_policy(args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents)
        else:
            policy = _load_rnn_policy(args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents)

    # Run evaluation rollouts
    completed = 0
    total_target = args_cli.num_episodes * args_cli.num_envs
    obs, info = env_wrapped.reset()

    print(f"[EVAL] Starting evaluation: {args_cli.num_episodes} ep/env × {args_cli.num_envs} envs = {total_target} total episodes")

    while completed < total_target:
        # Compute actions
        if is_greedy:
            agent_positions = {
                aid: unwrapped._robots[aid].data.root_pos_w
                for aid in possible_agents
            }
            target_pos = unwrapped.target.data.root_pos_w
            agent_orientations = {
                aid: unwrapped._robots[aid].data.root_quat_w
                for aid in possible_agents
            }
            actions = policy.compute_actions(agent_positions, target_pos, agent_orientations)
        else:
            with torch.no_grad():
                actions, _, _ = policy.act(obs, 0, 0)

        obs, rewards, terminated, truncated, info = env_wrapped.step(actions)

        # Detect done envs before collecting metrics (so timeseries can
        # exclude post-reset environments from the completed episode's data)
        first_agent = possible_agents[0]
        if isinstance(terminated, dict):
            done_mask = terminated[first_agent] | truncated[first_agent]
        else:
            done_mask = terminated | truncated

        # Collect metrics from env internals
        _collect_step_metrics(tracker, unwrapped, ts_tracker, done_mask)

        # Handle episode completions
        done_envs = torch.where(done_mask.squeeze(-1) if done_mask.dim() > 1 else done_mask)[0]
        if done_envs.numel() > 0:
            tracker.record_episode_end(done_envs)
            if ts_tracker is not None:
                ts_tracker.record_episode_end(done_envs)
            completed += done_envs.numel()
            if completed % max(total_target // 10, 1) == 0 or completed >= total_target:
                print(f"[EVAL] Completed {completed}/{total_target} episodes")

    # Compute final metrics
    results = tracker.compute_final_metrics()
    if ts_tracker is not None:
        ts_data = ts_tracker.compute_timeseries()
        ts_data["step_dt_seconds"] = float(unwrapped.step_dt)
        results["timeseries"] = ts_data
    results["experiment"] = exp_cfg.name
    results["checkpoint"] = args_cli.checkpoint or "greedy"
    results["eval_params"] = {
        "num_episodes": args_cli.num_episodes,
        "num_envs": args_cli.num_envs,
        "delay_override": args_cli.delay_override,
        "comm_delay_override": args_cli.comm_delay_override,
        "target_speed": args_cli.target_speed,
        "noise_override": args_cli.noise_override,
        "accel_override": args_cli.accel_override,
        "detection_dropout_override": args_cli.detection_dropout_override,
        "comm_dropout_override": args_cli.comm_dropout_override,
    }

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")
    print(f"{'='*80}")

    # Save results
    if args_cli.output:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        with open(args_cli.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
