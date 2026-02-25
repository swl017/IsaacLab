#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a single named experiment from the registry.

Usage:
    # Run with default seed
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/experiments/run_experiment.py \\
        --experiment a1_no_delay

    # Override seed and num_envs
    ./isaaclab.sh -p .../experiments/run_experiment.py \\
        --experiment a2_mlp --seed 42 --num_envs 2048

    # List available experiments
    ./isaaclab.sh -p .../experiments/run_experiment.py --list
"""

import argparse
import re
import sys
import os

# Parse experiment-specific args before AppLauncher
parser = argparse.ArgumentParser(description="Run a named experiment from the registry.")
parser.add_argument("--experiment", type=str, default=None, help="Experiment name from registry")
parser.add_argument("--seed", type=int, default=None, help="Override seed (else uses experiment default)")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA5-Direct-v0", help="Task name")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint file to resume training from (e.g., checkpoints/agent_140000.pt)")
parser.add_argument("--list", action="store_true", help="List all registered experiments and exit")
args_cli, hydra_args = parser.parse_known_args()

# Handle --list before AppLauncher
if args_cli.list:
    # Import just the registry (no sim needed)
    sys.path.insert(0, os.path.dirname(__file__))
    from experiment_registry import list_experiments, list_suites, get_experiment
    print("\n=== Registered Experiments ===")
    for name in list_experiments():
        exp = get_experiment(name)
        print(f"  {name:40s} [{exp.group:12s}] {exp.description}")
    print(f"\n=== Registered Suites ===")
    for name in list_suites():
        print(f"  {name}")
    sys.exit(0)

if args_cli.experiment is None:
    parser.error("--experiment is required (or use --list)")

sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
from isaaclab.app import AppLauncher

app_args = argparse.Namespace(headless=args_cli.headless, device="cuda:0", experience="")
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is initialized."""

import torch
import copy
from datetime import datetime
import gymnasium as gym
import numpy as np

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Import experiment infrastructure
from isaaclab_tasks.direct.iris_ma5.experiments import (
    get_experiment,
    apply_env_overrides,
    apply_agent_overrides,
)

# Import training components (mappo_rnn.py lives in scripts/reinforcement_learning/skrl/)
_skrl_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
                                  "scripts", "reinforcement_learning", "skrl")
_skrl_scripts_dir = os.path.normpath(_skrl_scripts_dir)
sys.path.insert(0, _skrl_scripts_dir)
from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue


class SingleAgentWrapper:
    """Wrapper to convert single-agent env to multi-agent format with agent_0."""

    def __init__(self, env):
        self.env = env
        self.possible_agents = ["agent_0"]
        self.num_envs = env.num_envs
        self.device = env.device
        self.observation_space = {"agent_0": env.observation_space}
        self.action_space = {"agent_0": env.action_space}
        self.observation_spaces = self.observation_space
        self.action_spaces = self.action_space
        self.shared_observation_spaces = self.observation_space

    def reset(self):
        obs, info = self.env.reset()
        info_wrapped = {"agent_0": info if info is not None else {}}
        if info is not None and "log" in info:
            info_wrapped["log"] = info["log"]
        return {"agent_0": obs}, info_wrapped

    def step(self, actions):
        action = actions["agent_0"]
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_wrapped = {"agent_0": info if info is not None else {}}
        if info is not None and "log" in info:
            info_wrapped["log"] = info["log"]
        return (
            {"agent_0": obs},
            {"agent_0": reward},
            {"agent_0": terminated},
            {"agent_0": truncated},
            info_wrapped,
        )

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def _setup_preprocessors(agent_cfg, observation_spaces, shared_observation_spaces, possible_agents, device):
    """Configure preprocessors in agent config."""
    if "state_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["state_preprocessor"] == "RunningStandardScaler":
            agent_cfg["agent"]["state_preprocessor"] = RunningStandardScaler
        if agent_cfg["agent"].get("state_preprocessor_kwargs") is None:
            agent_cfg["agent"]["state_preprocessor_kwargs"] = {}
        agent_cfg["agent"]["state_preprocessor_kwargs"]["size"] = observation_spaces[possible_agents[0]]
        agent_cfg["agent"]["state_preprocessor_kwargs"]["device"] = device

    if "shared_state_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["shared_state_preprocessor"] == "RunningStandardScaler":
            agent_cfg["agent"]["shared_state_preprocessor"] = RunningStandardScaler
        if agent_cfg["agent"].get("shared_state_preprocessor_kwargs") is None:
            agent_cfg["agent"]["shared_state_preprocessor_kwargs"] = {}
        agent_cfg["agent"]["shared_state_preprocessor_kwargs"]["size"] = shared_observation_spaces[possible_agents[0]]
        agent_cfg["agent"]["shared_state_preprocessor_kwargs"]["device"] = device

    if "value_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["value_preprocessor"] == "RunningStandardScaler":
            agent_cfg["agent"]["value_preprocessor"] = RunningStandardScaler
        if agent_cfg["agent"].get("value_preprocessor_kwargs") is None:
            agent_cfg["agent"]["value_preprocessor_kwargs"] = {}
        agent_cfg["agent"]["value_preprocessor_kwargs"]["device"] = device

    if "learning_rate_scheduler" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["learning_rate_scheduler"] == "KLAdaptiveLR":
            agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveLR

    if "learning_rate" in agent_cfg.get("agent", {}):
        lr = agent_cfg["agent"]["learning_rate"]
        if isinstance(lr, str):
            agent_cfg["agent"]["learning_rate"] = float(lr)


def _create_env(task, env_cfg):
    """Create and wrap the environment."""
    env = gym.make(task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)
    is_multi_agent = isinstance(env.unwrapped.unwrapped, DirectMARLEnv)

    if is_multi_agent:
        possible_agents = env.possible_agents
    else:
        env = SingleAgentWrapper(env)
        possible_agents = env.possible_agents
        is_multi_agent = True

    return env, possible_agents, is_multi_agent


def _resume_from_checkpoint(checkpoint_path, trainer, agent):
    """Load checkpoint into agent and set trainer to resume from the correct timestep."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    # Auto-detect timestep from filename (e.g., agent_140000.pt -> 140000)
    match = re.search(r"agent_(\d+)\.pt$", os.path.basename(checkpoint_path))
    if match:
        resume_timestep = int(match.group(1))
    else:
        # For best_agent.pt, find highest numbered checkpoint in same dir
        ckpt_dir = os.path.dirname(checkpoint_path)
        numbered = [
            int(re.search(r"agent_(\d+)\.pt$", f).group(1))
            for f in os.listdir(ckpt_dir)
            if re.search(r"agent_(\d+)\.pt$", f)
        ]
        if numbered:
            resume_timestep = max(numbered)
            print(f"[INFO] Inferred resume timestep {resume_timestep} from highest numbered checkpoint")
        else:
            raise ValueError(
                f"Cannot determine resume timestep from '{os.path.basename(checkpoint_path)}'. "
                "Use a numbered checkpoint (e.g., agent_140000.pt) instead."
            )

    trainer.initial_timestep = resume_timestep
    print(f"[INFO] Resuming training from timestep {resume_timestep}")


def _log_dir_from_checkpoint(checkpoint_path):
    """Derive the original log directory from a checkpoint path.

    E.g. .../2026-02-14_15-58-58_a1_with_aoi_seed42/checkpoints/agent_140000.pt
      -> .../2026-02-14_15-58-58_a1_with_aoi_seed42/
    """
    ckpt_abs = os.path.abspath(checkpoint_path)
    return os.path.dirname(os.path.dirname(ckpt_abs))


def _setup_logging(agent_cfg, exp_name, seed):
    """Configure logging directories."""
    log_root_path = os.path.join(
        "logs", "skrl",
        agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "iris_ma5_experiments"),
    )
    log_root_path = os.path.abspath(log_root_path)

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{exp_name}_seed{seed}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    full_log_dir = os.path.join(log_root_path, log_dir)
    return full_log_dir


def _train_rnn(env, possible_agents, agent_cfg, exp_cfg, seed, log_dir, env_cfg, checkpoint_path=None):
    """Train with MAPPO-RNN (default path)."""
    device = env.device
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 32)

    # Shared observation spaces
    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

    # Create models
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    value_cfg = model_cfg.get("value", {})

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

    _setup_preprocessors(agent_cfg, env.observation_spaces, shared_observation_spaces, possible_agents, device)

    # Dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Create agent
    mappo_cfg = copy.deepcopy(MAPPO_RNN_DEFAULT_CONFIG)
    mappo_cfg.update(agent_cfg.get("agent", {}))

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

    # Train
    trainer_cfg = {
        "timesteps": exp_cfg.total_timesteps,
        "headless": True,
        "environment_info": "log",
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    if checkpoint_path:
        _resume_from_checkpoint(checkpoint_path, trainer, agent)

    print("=" * 80)
    print(f"MAPPO-RNN Training: {exp_cfg.name} (seed={seed})")
    print("=" * 80)
    trainer.train()

    # Save final models
    for agent_id in possible_agents:
        agent_path = os.path.join(log_dir, f"agent_{agent_id}_final.pt")
        torch.save({
            "policy_state_dict": models[agent_id]["policy"].state_dict(),
            "value_state_dict": models[agent_id]["value"].state_dict(),
        }, agent_path)
    print(f"[INFO] Models saved to: {log_dir}")


def _train_mlp(env, possible_agents, agent_cfg, exp_cfg, seed, log_dir, env_cfg, checkpoint_path=None):
    """Train with MAPPO-MLP (no recurrence)."""
    from isaaclab_tasks.direct.iris_ma5.experiments.models.mappo_mlp import (
        MAPPOMLPPolicy,
        MAPPOMLPValue,
    )
    from isaaclab_tasks.direct.iris_ma5.experiments.models.mappo_mlp_agent import (
        MAPPO_MLP,
        MAPPO_MLP_DEFAULT_CONFIG,
    )

    device = env.device

    # Shared observation spaces
    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

    # Create MLP models
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    value_cfg = model_cfg.get("value", {})

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

    _setup_preprocessors(agent_cfg, env.observation_spaces, shared_observation_spaces, possible_agents, device)

    # Dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Create MAPPO_MLP agent
    mappo_cfg = copy.deepcopy(MAPPO_MLP_DEFAULT_CONFIG)
    mappo_cfg.update(agent_cfg.get("agent", {}))

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

    # Train
    trainer_cfg = {
        "timesteps": exp_cfg.total_timesteps,
        "headless": True,
        "environment_info": "log",
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    if checkpoint_path:
        _resume_from_checkpoint(checkpoint_path, trainer, agent)

    print("=" * 80)
    print(f"MAPPO-MLP Training: {exp_cfg.name} (seed={seed})")
    print("=" * 80)
    trainer.train()

    # Save final models
    for agent_id in possible_agents:
        agent_path = os.path.join(log_dir, f"agent_{agent_id}_final.pt")
        torch.save({
            "policy_state_dict": models[agent_id]["policy"].state_dict(),
            "value_state_dict": models[agent_id]["value"].state_dict(),
        }, agent_path)
    print(f"[INFO] Models saved to: {log_dir}")


@hydra_task_config(args_cli.task, "skrl_mappo_rnn_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Run a named experiment."""
    # Load experiment config
    exp_cfg = get_experiment(args_cli.experiment)
    print(f"\n{'='*80}")
    print(f"[EXPERIMENT] Name:        {exp_cfg.name}")
    print(f"[EXPERIMENT] Group:       {exp_cfg.group}")
    print(f"[EXPERIMENT] Description: {exp_cfg.description}")
    print(f"[EXPERIMENT] MLP model:   {exp_cfg.use_mlp_model}")
    print(f"{'='*80}\n")

    # Apply experiment env overrides
    if exp_cfg.env_overrides:
        apply_env_overrides(env_cfg, exp_cfg.env_overrides)
        print(f"[EXPERIMENT] Applied {len(exp_cfg.env_overrides)} env overrides")

    # Apply experiment agent overrides
    if exp_cfg.agent_overrides:
        apply_agent_overrides(agent_cfg, exp_cfg.agent_overrides)
        print(f"[EXPERIMENT] Applied {len(exp_cfg.agent_overrides)} agent overrides")

    # CLI overrides
    seed = args_cli.seed if args_cli.seed is not None else exp_cfg.seeds[0]
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else exp_cfg.num_envs
    env_cfg.scene.num_envs = num_envs
    agent_cfg["seed"] = seed
    set_seed(seed)

    # Setup logging
    if args_cli.checkpoint:
        # Resume into original experiment directory
        log_dir = _log_dir_from_checkpoint(args_cli.checkpoint)
        log_root = os.path.dirname(log_dir)
        agent_cfg["agent"]["experiment"]["directory"] = log_root
        agent_cfg["agent"]["experiment"]["experiment_name"] = os.path.basename(log_dir)
        print(f"[EXPERIMENT] Resuming into existing log dir: {log_dir}")
    else:
        log_dir = _setup_logging(agent_cfg, exp_cfg.name, seed)
        print(f"[EXPERIMENT] Log dir: {log_dir}")

    # Create environment
    env, possible_agents, _ = _create_env(args_cli.task, env_cfg)
    print(f"[EXPERIMENT] Agents: {possible_agents}, Envs: {num_envs}")

    # Apply frame-skip stacking if configured (for MLP temporal context)
    if exp_cfg.frame_stack > 1:
        from isaaclab_tasks.direct.iris_ma5.experiments.frame_stack_wrapper import MultiAgentFrameStackWrapper
        env = MultiAgentFrameStackWrapper(env, num_stack=exp_cfg.frame_stack, frame_skip=exp_cfg.frame_skip)
        print(f"[EXPERIMENT] Frame stacking: K={exp_cfg.frame_stack}, skip={exp_cfg.frame_skip}")

    # Route to appropriate training function
    if exp_cfg.use_mlp_model:
        _train_mlp(env, possible_agents, agent_cfg, exp_cfg, seed, log_dir, env_cfg, args_cli.checkpoint)
    else:
        _train_rnn(env, possible_agents, agent_cfg, exp_cfg, seed, log_dir, env_cfg, args_cli.checkpoint)

    print(f"\n{'='*80}")
    print(f"[EXPERIMENT] Completed: {exp_cfg.name} (seed={seed})")
    print(f"{'='*80}")

    env.close()


if __name__ == "__main__":
    main()
