#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained checkpoint and compute the 7 IROS 2026 paper metrics.

Usage:
    # Evaluate a trained policy
    ./isaaclab.sh -p .../experiments/evaluate.py \\
        --experiment a1_stochastic_delay \\
        --checkpoint /path/to/agent_drone_0_final.pt \\
        --num_episodes 100 --output results.json

    # Evaluate greedy baseline (no checkpoint needed)
    ./isaaclab.sh -p .../experiments/evaluate.py \\
        --experiment baseline_greedy --num_episodes 100
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Evaluate trained policy with 7 paper metrics.")
parser.add_argument("--experiment", type=str, required=True, help="Experiment name from registry")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of evaluation episodes")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel eval envs")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA4-Direct-v0")
parser.add_argument("--output", type=str, default=None, help="Output JSON path for results")
parser.add_argument("--headless", action="store_true", default=True)
args_cli = parser.parse_args()

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

from isaaclab_tasks.direct.iris_ma4.experiments import (
    get_experiment,
    apply_env_overrides,
)
from isaaclab_tasks.direct.iris_ma4.experiments.metrics import MetricTracker


def _collect_step_metrics(tracker: MetricTracker, env) -> None:
    """Extract per-step metrics from environment internals."""
    first_agent = env.cfg.possible_agents[0]

    # Get triangulation results (noisy path, what the agent observes)
    if hasattr(env, "triangulation_results_noisy") and first_agent in env.triangulation_results_noisy:
        X_w_gt, Sigma_X, std_dev, trace_cov, is_valid = env.triangulation_results_noisy[first_agent]
    else:
        trace_cov = torch.ones(env.num_envs, 1, device=env.device) * 999.0
        X_w_gt = torch.zeros(env.num_envs, 1, 3, device=env.device)
        is_valid = torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.bool)

    # Also get clean triangulation for RMSE (use rewards path for ground truth comparison)
    if hasattr(env, "triangulation_results") and first_agent in env.triangulation_results:
        X_w_gt_clean, Sigma_X_clean, std_clean, trace_clean, is_valid_clean = env.triangulation_results[first_agent]
    else:
        X_w_gt_clean = X_w_gt
        is_valid_clean = is_valid

    # Ground truth target position
    gt_target_pos = env.target.data.root_pos_w

    # Triangulated position from clean path (for RMSE)
    # Note: X_w_gt from triangulation_results is actually the GT position used for covariance
    # The actual triangulated position is computed differently. For RMSE, we use the
    # noisy triangulation estimate if available.
    tri_pos = X_w_gt_clean.squeeze(1) if X_w_gt_clean.dim() == 3 else X_w_gt_clean

    # Bbox validity for all agents
    bbox_valid_list = []
    for aid in env.cfg.possible_agents:
        states = env.delay_system.get_all_states_for_observations(aid)
        bbox = states[aid].data.bboxes_2d[:, 0, :]
        valid = env.bbox_raycaster.validate_bbox(bbox).squeeze(-1)
        bbox_valid_list.append(valid)
    bbox_valid_mask = torch.stack(bbox_valid_list, dim=1)

    # Collision detection
    collision_flags = env.safety_manager.get_collision_flags()

    tracker.step(
        trace_sigma=trace_cov,
        triangulated_pos=tri_pos,
        gt_target_pos=gt_target_pos,
        bbox_valid_mask=bbox_valid_mask,
        collision_flags=collision_flags,
        tri_valid=is_valid_clean,
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

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    shared_policy.load_state_dict(ckpt["policy_state_dict"])
    shared_value.load_state_dict(ckpt["value_state_dict"])
    print(f"[EVAL] Loaded checkpoint: {checkpoint_path}")

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

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env)
    unwrapped = env.unwrapped

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

    # Load policy
    is_greedy = (args_cli.experiment == "baseline_greedy")
    if is_greedy:
        from isaaclab_tasks.direct.iris_ma4.experiments.baselines.greedy_policy import (
            GreedyEquiangularPolicy,
        )
        policy = GreedyEquiangularPolicy(
            num_agents=num_agents, device=str(unwrapped.device)
        )
        print(f"[EVAL] Using greedy equiangular policy")
    else:
        if args_cli.checkpoint is None:
            raise ValueError("--checkpoint required for trained policy evaluation")
        policy = _load_rnn_policy(args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents)

    # Run evaluation rollouts
    completed = 0
    obs, info = env_wrapped.reset()

    print(f"[EVAL] Starting evaluation: {args_cli.num_episodes} episodes, {args_cli.num_envs} envs")

    while completed < args_cli.num_episodes:
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

        # Collect metrics from env internals
        _collect_step_metrics(tracker, unwrapped)

        # Check for episode completions
        first_agent = possible_agents[0]
        if isinstance(terminated, dict):
            done_mask = terminated[first_agent] | truncated[first_agent]
        else:
            done_mask = terminated | truncated
        done_envs = torch.where(done_mask.squeeze(-1) if done_mask.dim() > 1 else done_mask)[0]
        if done_envs.numel() > 0:
            tracker.record_episode_end(done_envs)
            completed += done_envs.numel()
            if completed % 20 == 0 or completed >= args_cli.num_episodes:
                print(f"[EVAL] Completed {completed}/{args_cli.num_episodes} episodes")

    # Compute final metrics
    results = tracker.compute_final_metrics()
    results["experiment"] = exp_cfg.name
    results["checkpoint"] = args_cli.checkpoint or "greedy"

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
