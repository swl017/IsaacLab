#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train RL agent with MAPPO-RNN using Hydra configuration system.

This script uses Hydra for configuration management, making it fully compatible
with Ray Tune hyperparameter optimization via hydra_args overrides.

Usage:
    # Standard training
    python train_mappo_rnn_hydra.py --task Isaac-Quadcopter-Direct-v0 --num_envs 1024

    # With Hydra overrides (for hyperparameter tuning)
    python train_mappo_rnn_hydra.py --task Isaac-Quadcopter-Direct-v0 --num_envs 1024 \
        agent.learning_rate=0.001 models.policy.hidden_size=256 agent.sequence_length=50

    # Ray Tune will use this format automatically via hydra_args
"""

import argparse
import sys

# Parse arguments BEFORE launching Isaac Sim
parser = argparse.ArgumentParser(description="Train MAPPO-RNN agent with Hydra config system.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# parser.add_argument("--experiment_name", type=str, default="", help="Custom experiment name")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode (no GUI).")

# Parse known args (separates our args from Hydra args)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra (Hydra will parse hydra_args)
sys.argv = [sys.argv[0]] + hydra_args

# NOW we can import Isaac Lab and launch the simulator
from isaaclab.app import AppLauncher

# Create minimal AppLauncher args
app_args = argparse.Namespace(headless=args_cli.headless, device="cuda:0", experience="")
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is initialized."""

import torch
import os
from datetime import datetime
import gymnasium as gym
import numpy as np

from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue

# Import SKRL components
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Import Isaac Lab components
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper


class SingleAgentWrapper:
    """Wrapper to convert single-agent env to multi-agent format with agent_0."""

    def __init__(self, env):
        self.env = env
        self.possible_agents = ["agent_0"]
        self.num_envs = env.num_envs
        self.device = env.device

        # Wrap spaces in dictionaries
        self.observation_space = {"agent_0": env.observation_space}
        self.action_space = {"agent_0": env.action_space}
        self.observation_spaces = self.observation_space
        self.action_spaces = self.action_space
        self.shared_observation_spaces = self.observation_space

    def reset(self):
        obs, info = self.env.reset()
        # Wrap per-agent data but preserve "log" key at top level for trainer logging
        info_wrapped = {"agent_0": info if info is not None else {}}
        if info is not None and "log" in info:
            info_wrapped["log"] = info["log"]
        return {"agent_0": obs}, info_wrapped

    def step(self, actions):
        action = actions["agent_0"]
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Wrap per-agent data but preserve "log" key at top level for trainer logging
        info_wrapped = {"agent_0": info if info is not None else {}}
        if info is not None and "log" in info:
            info_wrapped["log"] = info["log"]
        return (
            {"agent_0": obs},
            {"agent_0": reward},
            {"agent_0": terminated},
            {"agent_0": truncated},
            info_wrapped
        )

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def create_models(agent_cfg: dict, env, device, is_multi_agent: bool):
    """Create policy and value models based on config."""

    models = {}
    memories = {}

    # Get config sections
    model_cfg = agent_cfg.get("models", {})
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 50)

    if is_multi_agent:
        possible_agents = env.possible_agents

        # Check for shared observation space
        try:
            shared_observation_spaces = env.shared_observation_spaces
        except AttributeError:
            obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
            shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
            shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

        # Create shared networks
        policy_cfg = model_cfg.get("policy", {})
        value_cfg = model_cfg.get("value", {})

        shared_policy = MAPPORNNPolicy(
            observation_space=env.observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=policy_cfg.get("hidden_size", 256),
            gru_num_layers=policy_cfg.get("gru_num_layers", 2),
            gru_hidden_size=policy_cfg.get("gru_hidden_size", 256),
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
            hidden_size=value_cfg.get("hidden_size", 256),
            gru_num_layers=value_cfg.get("gru_num_layers", 2),
            gru_hidden_size=value_cfg.get("gru_hidden_size", 256),
            num_envs=env.num_envs,
            sequence_length=sequence_length,
        )

        for agent_id in possible_agents:
            models[agent_id] = {"policy": shared_policy, "value": shared_value}
            memories[agent_id] = RandomMemory(
                memory_size=agent_cfg.get("agent", {}).get("rollouts", 800),
                num_envs=env.num_envs,
                device=device,
            )

        return models, memories, shared_observation_spaces, env.observation_spaces, env.action_spaces

    else:
        # Single-agent
        possible_agents = ["agent_0"]
        observation_spaces = {"agent_0": env.observation_space}
        action_spaces = {"agent_0": env.action_space}
        shared_observation_spaces = {"agent_0": env.observation_space}

        policy_cfg = model_cfg.get("policy", {})
        value_cfg = model_cfg.get("value", {})

        policy = MAPPORNNPolicy(
            observation_space=observation_spaces["agent_0"],
            action_space=action_spaces["agent_0"],
            device=device,
            hidden_size=policy_cfg.get("hidden_size", 256),
            gru_num_layers=policy_cfg.get("gru_num_layers", 2),
            gru_hidden_size=policy_cfg.get("gru_hidden_size", 256),
            num_envs=env.num_envs,
            initial_log_std=policy_cfg.get("initial_log_std", -0.5),
            min_log_std=policy_cfg.get("min_log_std", -5.0),
            max_log_std=policy_cfg.get("max_log_std", 0.7),
            sequence_length=sequence_length,
        )

        value = MAPPORNNValue(
            observation_space=shared_observation_spaces["agent_0"],
            action_space=action_spaces["agent_0"],
            device=device,
            hidden_size=value_cfg.get("hidden_size", 256),
            gru_num_layers=value_cfg.get("gru_num_layers", 2),
            gru_hidden_size=value_cfg.get("gru_hidden_size", 256),
            num_envs=env.num_envs,
            sequence_length=sequence_length,
        )

        models["agent_0"] = {"policy": policy, "value": value}
        memories["agent_0"] = RandomMemory(
            memory_size=agent_cfg.get("agent", {}).get("rollouts", 800),
            num_envs=env.num_envs,
            device=device,
        )

        return models, memories, shared_observation_spaces, observation_spaces, action_spaces


@hydra_task_config(args_cli.task, "skrl_mappo_rnn_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train MAPPO-RNN agent with Hydra-managed configuration."""

    # Override with CLI args
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    # Set seed
    seed = agent_cfg.get("seed", 42)
    set_seed(seed)

    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)
    device = env.device

    # Detect multi-agent
    is_multi_agent = isinstance(env.unwrapped.unwrapped, DirectMARLEnv)

    if is_multi_agent:
        possible_agents = env.possible_agents
        print(f"[INFO] Multi-agent environment: {len(possible_agents)} agents")
    else:
        print(f"[INFO] Single-agent environment")
        env = SingleAgentWrapper(env)
        possible_agents = env.possible_agents
        is_multi_agent = True

    # Validate burn-in
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 50)
    burn_in_steps = agent_cfg.get("agent", {}).get("burn_in_steps", 0)

    if burn_in_steps >= sequence_length:
        raise ValueError(f"burn_in_steps ({burn_in_steps}) must be < sequence_length ({sequence_length})")

    # Create models
    models, memories, shared_observation_spaces, observation_spaces, action_spaces = create_models(
        agent_cfg, env, device, is_multi_agent
    )

    # Process preprocessor configs
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

    # Convert learning_rate to float if it's a string (YAML may parse "1e-3" as string)
    if "learning_rate" in agent_cfg.get("agent", {}):
        lr = agent_cfg["agent"]["learning_rate"]
        if isinstance(lr, str):
            agent_cfg["agent"]["learning_rate"] = float(lr)
            print(f"[INFO] Converted learning_rate from string '{lr}' to float {agent_cfg['agent']['learning_rate']}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "mappo_rnn_logs"))
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_mappo_rnn_torch"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Create agent
    mappo_cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
    mappo_cfg.update(agent_cfg.get("agent", {}))

    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        device=device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )

    # Create trainer
    trainer_cfg = {
        "timesteps": agent_cfg.get("trainer", {}).get("timesteps", 400000),
        "headless": args_cli.headless,
        "environment_info": "log",
    }

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # Print training info
    print("=" * 80)
    print("Starting MAPPO-RNN Training")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Device: {device}")
    print(f"Num envs: {env_cfg.scene.num_envs}")
    print(f"Num agents: {len(possible_agents)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Burn-in steps: {burn_in_steps}")
    print(f"Learning rate: {agent_cfg.get('agent', {}).get('learning_rate', 'default')}")
    print(f"Total timesteps: {trainer_cfg['timesteps']}")
    print("=" * 80)

    # Train
    trainer.train()

    # Save models
    for agent_id in possible_agents:
        agent_path = os.path.join(log_dir, f"agent_{agent_id}_final.pt")
        torch.save({
            'policy_state_dict': models[agent_id]["policy"].state_dict(),
            'value_state_dict': models[agent_id]["value"].state_dict(),
        }, agent_path)
        print(f"[INFO] Agent {agent_id} saved to: {agent_path}")

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
