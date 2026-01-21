#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train RL agent with MAPPO-RNN using configurable YAML files.

Usage:
    # Single-agent environment
    python train_mappo_rnn.py --task Isaac-Quadcopter-Direct-v0 --num_envs 1024 --headless

    # Multi-agent environment
    python train_mappo_rnn.py --task Isaac-Iris-MA2-Direct-Comm-v0 --num_envs 400 --headless

    # With custom hyperparameters
    python train_mappo_rnn.py --task Isaac-Quadcopter-Direct-v0 --num_envs 1024 \
        --burn_in_steps 10 --sequence_length 50 --headless
"""

import argparse
import torch
import os
from datetime import datetime
import gymnasium as gym
import numpy as np
import yaml
from pathlib import Path
from typing import Any

from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue

# Import SKRL components
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# Parse arguments
parser = argparse.ArgumentParser(description="Train MAPPO-RNN agent with configurable YAML files.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# MAPPO-RNN specific arguments
parser.add_argument("--sequence_length", type=int, default=None, help="RNN sequence length for training")
parser.add_argument("--burn_in_steps", type=int, default=None, help="Burn-in steps for RNN hidden state warm-up")
parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
parser.add_argument("--rollouts", type=int, default=None, help="Number of rollouts before updating")

# Experiment tracking
parser.add_argument("--experiment_name", type=str, default="", help="Custom experiment name")
parser.add_argument("--headless", action="store_true", default=None, help="Run in headless mode")

args_cli = parser.parse_args()


def load_config_yaml(task_name: str) -> dict[str, Any]:
    """Load MAPPO-RNN config from task's YAML file."""
    try:
        # Get the environment spec
        env_spec = gym.spec(task_name)

        # Get the config entry point
        cfg_entry_point = env_spec.kwargs.get("skrl_mappo_rnn_cfg_entry_point", None)

        if cfg_entry_point is None:
            raise ValueError(
                f"Task '{task_name}' does not have 'skrl_mappo_rnn_cfg_entry_point' defined in __init__.py.\n"
                f"Please add it to the gym.register() call."
            )

        # Parse the entry point (format: "module.path:config.yaml")
        module_path, file_name = cfg_entry_point.split(":")
        module = __import__(module_path, fromlist=[""])
        cfg_file_path = Path(module.__file__).parent / file_name

        # Load YAML
        with open(cfg_file_path, "r") as f:
            cfg_raw = yaml.safe_load(f)

        print(f"[INFO] Loaded MAPPO-RNN config from: {cfg_file_path}")

        # Ensure it's a dict
        if not isinstance(cfg_raw, dict):
            raise ValueError(f"Config file must be a YAML dictionary, got {type(cfg_raw)}")

        return cfg_raw

    except Exception as e:
        print(f"[ERROR] Failed to load config for task '{task_name}': {e}")
        print(f"[INFO] Using default MAPPO-RNN config")
        return {"seed": 42, "agent": {}, "trainer": {}, "models": {}}


def process_config(cfg: dict[str, Any], device: Any, args_cli: Any) -> dict[str, Any]:
    """Process config to convert string references to actual classes and set runtime values."""

    # Convert string class names to actual classes
    if "learning_rate_scheduler" in cfg.get("agent", {}):
        scheduler_name = cfg["agent"]["learning_rate_scheduler"]
        if scheduler_name == "KLAdaptiveLR":
            cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveLR

    if "state_preprocessor" in cfg.get("agent", {}):
        preprocessor_name = cfg["agent"]["state_preprocessor"]
        if preprocessor_name == "RunningStandardScaler":
            cfg["agent"]["state_preprocessor"] = RunningStandardScaler

    if "shared_state_preprocessor" in cfg.get("agent", {}):
        preprocessor_name = cfg["agent"]["shared_state_preprocessor"]
        if preprocessor_name == "RunningStandardScaler":
            cfg["agent"]["shared_state_preprocessor"] = RunningStandardScaler

    if "value_preprocessor" in cfg.get("agent", {}):
        preprocessor_name = cfg["agent"]["value_preprocessor"]
        if preprocessor_name == "RunningStandardScaler":
            cfg["agent"]["value_preprocessor"] = RunningStandardScaler

    # Set device in preprocessor kwargs
    if "state_preprocessor_kwargs" in cfg.get("agent", {}):
        if cfg["agent"]["state_preprocessor_kwargs"] is None:
            cfg["agent"]["state_preprocessor_kwargs"] = {}
        cfg["agent"]["state_preprocessor_kwargs"]["device"] = device

    if "shared_state_preprocessor_kwargs" in cfg.get("agent", {}):
        if cfg["agent"]["shared_state_preprocessor_kwargs"] is None:
            cfg["agent"]["shared_state_preprocessor_kwargs"] = {}
        cfg["agent"]["shared_state_preprocessor_kwargs"]["device"] = device

    if "value_preprocessor_kwargs" in cfg.get("agent", {}):
        if cfg["agent"]["value_preprocessor_kwargs"] is None:
            cfg["agent"]["value_preprocessor_kwargs"] = {}
        if "device" in cfg["agent"]["value_preprocessor_kwargs"]:
            cfg["agent"]["value_preprocessor_kwargs"]["device"] = device

    # Override with CLI arguments
    if args_cli.sequence_length is not None:
        cfg["agent"]["sequence_length"] = args_cli.sequence_length

    if args_cli.burn_in_steps is not None:
        cfg["agent"]["burn_in_steps"] = args_cli.burn_in_steps

    if args_cli.learning_rate is not None:
        cfg["agent"]["learning_rate"] = args_cli.learning_rate

    if args_cli.rollouts is not None:
        cfg["agent"]["rollouts"] = args_cli.rollouts

    if args_cli.max_iterations is not None:
        cfg["trainer"]["timesteps"] = args_cli.max_iterations * cfg["agent"]["rollouts"]

    # Set seed
    if args_cli.seed is not None:
        cfg["seed"] = args_cli.seed

    return cfg


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

        # For MAPPO, shared observation space is same as observation space for single agent
        self.shared_observation_spaces = self.observation_space

    def reset(self):
        """Reset environment and wrap observations."""
        obs, info = self.env.reset()
        # Handle info being None or empty dict
        info_wrapped = {"agent_0": info if info is not None else {}}
        return {"agent_0": obs}, info_wrapped

    def step(self, actions):
        """Step environment and wrap observations/rewards/dones."""
        # Extract action from dictionary
        action = actions["agent_0"]

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Wrap everything in dictionaries
        # Handle info being None or empty dict
        info_wrapped = {"agent_0": info if info is not None else {}}
        return (
            {"agent_0": obs},
            {"agent_0": reward},
            {"agent_0": terminated},
            {"agent_0": truncated},
            info_wrapped
        )

    def close(self):
        """Close environment."""
        self.env.close()

    def __getattr__(self, name):
        """Forward other attributes to wrapped env."""
        return getattr(self.env, name)


def create_models(cfg: dict[str, Any], env: Any, device: Any, is_multi_agent: bool) -> tuple[Any, Any, Any, Any, Any]:
    """Create policy and value models based on config.

    Returns:
        models: Dictionary of models per agent
        memories: Dictionary of memories per agent
        shared_observation_spaces: Dictionary of shared observation spaces
        observation_spaces: Dictionary of observation spaces (for single-agent)
        action_spaces: Dictionary of action spaces (for single-agent)
    """

    models: dict[str, Any] = {}
    memories: dict[str, Any] = {}

    # Get model config
    model_cfg = cfg.get("models", {})
    agent_cfg = cfg.get("agent", {})

    sequence_length = agent_cfg.get("sequence_length", 50)

    if is_multi_agent:
        # Multi-agent: Create models for each agent
        possible_agents = env.possible_agents

        # Check if shared observation space exists
        try:
            shared_observation_spaces = env.shared_observation_spaces
        except AttributeError:
            print("[INFO] No shared_observation_spaces found. Creating by concatenating agent observations.")
            obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
            shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
            shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

        # Check if all agents have the same spaces
        obs_spaces_match = all(
            env.observation_spaces[agent_id] == env.observation_spaces[possible_agents[0]]
            for agent_id in possible_agents
        )
        action_spaces_match = all(
            env.action_spaces[agent_id] == env.action_spaces[possible_agents[0]]
            for agent_id in possible_agents
        )

        if not obs_spaces_match or not action_spaces_match:
            raise NotImplementedError(
                "Heterogeneous agent spaces not yet supported. "
                "All agents must have the same observation and action spaces."
            )

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

        # Assign to all agents
        for agent_id in possible_agents:
            models[agent_id] = {
                "policy": shared_policy,
                "value": shared_value,
            }

            memories[agent_id] = RandomMemory(
                memory_size=agent_cfg.get("rollouts", 800),
                num_envs=env.num_envs,
                device=device,
            )

        # For multi-agent, return env's observation/action spaces
        return models, memories, shared_observation_spaces, env.observation_spaces, env.action_spaces

    else:
        # Single-agent: Wrap as dictionary with single agent
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
            memory_size=agent_cfg.get("rollouts", 800),
            num_envs=env.num_envs,
            device=device,
        )

        return models, memories, shared_observation_spaces, observation_spaces, action_spaces


def main():
    """Train MAPPO-RNN agent."""

    # Set seed for reproducibility
    seed = args_cli.seed if args_cli.seed is not None else 42
    set_seed(seed)

    # Load environment
    task_name = args_cli.task
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 400

    print(f"[INFO] Loading environment: {task_name}")
    print(f"[INFO] Number of environments: {num_envs}")

    # load_isaaclab_env handles AppLauncher initialization internally
    env = load_isaaclab_env(task_name=task_name, num_envs=num_envs, headless=True)
    env = wrap_env(env)
    device = env.device

    # Import isaaclab modules AFTER env is loaded (Isaac Sim is already initialized)
    from isaaclab.envs import DirectMARLEnv
    from isaaclab.utils.io import dump_pickle, dump_yaml
    import isaaclab_tasks  # noqa: F401

    # Detect if multi-agent
    is_multi_agent = isinstance(env.unwrapped, DirectMARLEnv)

    if is_multi_agent:
        possible_agents = env.possible_agents
        num_agents = len(possible_agents)
        print(f"[INFO] Multi-agent environment detected: {num_agents} agents")
        print(f"[INFO] Agents: {possible_agents}")
    else:
        print(f"[INFO] Single-agent environment detected")
        # Wrap single-agent env to look like multi-agent with agent_0
        env = SingleAgentWrapper(env)
        possible_agents = env.possible_agents
        # After wrapping, treat as multi-agent for model creation
        is_multi_agent = True

    # Load config from YAML
    cfg = load_config_yaml(task_name)

    # Process config
    cfg = process_config(cfg, device, args_cli)

    # Validate burn_in_steps
    sequence_length = cfg["agent"].get("sequence_length", 50)
    burn_in_steps = cfg["agent"].get("burn_in_steps", 0)

    if burn_in_steps >= sequence_length:
        raise ValueError(f"burn_in_steps ({burn_in_steps}) must be less than sequence_length ({sequence_length})")

    if burn_in_steps > 0:
        print(f"[INFO] RNN burn-in enabled: {burn_in_steps} steps (out of {sequence_length} sequence length)")

    # Create models (always returns 5 values now)
    models, memories, shared_observation_spaces, observation_spaces, action_spaces = create_models(
        cfg, env, device, is_multi_agent=is_multi_agent
    )

    # Set preprocessor sizes
    if "state_preprocessor_kwargs" in cfg["agent"]:
        if cfg["agent"]["state_preprocessor_kwargs"] is None:
            cfg["agent"]["state_preprocessor_kwargs"] = {}
        cfg["agent"]["state_preprocessor_kwargs"]["size"] = observation_spaces[possible_agents[0]]

    if "shared_state_preprocessor_kwargs" in cfg["agent"]:
        if cfg["agent"]["shared_state_preprocessor_kwargs"] is None:
            cfg["agent"]["shared_state_preprocessor_kwargs"] = {}
        cfg["agent"]["shared_state_preprocessor_kwargs"]["size"] = shared_observation_spaces[possible_agents[0]]

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args_cli.experiment_name if args_cli.experiment_name else "mappo_rnn"
    experiment_dir = cfg["agent"].get("experiment", {}).get("directory", "mappo_rnn_logs")
    log_dir = f"logs/skrl/{experiment_dir}/{timestamp}_{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    cfg["agent"]["experiment"]["directory"] = log_dir
    cfg["agent"]["experiment"]["experiment_name"] = experiment_name

    print(f"[INFO] Experiment directory: {log_dir}")

    # Save configs
    dump_yaml(os.path.join(log_dir, "params", "config.yaml"), cfg)
    dump_pickle(os.path.join(log_dir, "params", "config.pkl"), cfg)

    # Create MAPPO-RNN agent config
    mappo_cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
    mappo_cfg.update(cfg.get("agent", {}))

    # Create agent
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
    trainer_cfg = cfg.get("trainer", {})
    trainer_cfg.update({
        "timesteps": trainer_cfg.get("timesteps", 400000),
        "headless": True,
        "environment_info": "log",
    })

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)  # type: ignore

    # Print training info
    print("=" * 80)
    print(f"Starting MAPPO-RNN Training")
    print("=" * 80)
    print(f"Task: {task_name}")
    print(f"Device: {device}")
    print(f"Num envs: {num_envs}")
    print(f"Num agents: {len(possible_agents)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Burn-in steps: {burn_in_steps}")
    print(f"Total timesteps: {trainer_cfg['timesteps']}")
    print(f"Log directory: {log_dir}")
    print("=" * 80)

    # Train
    trainer.train()

    # Save trained agents
    for agent_id in possible_agents:
        agent_path = os.path.join(log_dir, f"agent_{agent_id}_final.pt")
        torch.save({
            'policy_state_dict': models[agent_id]["policy"].state_dict(),
            'value_state_dict': models[agent_id]["value"].state_dict(),
        }, agent_path)
        print(f"[INFO] Agent {agent_id} saved to: {agent_path}")

    # Save training config
    config_path = os.path.join(log_dir, "training_config.pt")
    config_to_save = {
        "agent_config": mappo_cfg,
        "trainer_config": trainer_cfg,
        "task_name": task_name,
        "num_agents": len(possible_agents),
        "possible_agents": possible_agents,
        "sequence_length": sequence_length,
        "burn_in_steps": burn_in_steps,
        "timestamp": timestamp,
        "device": str(device),
    }
    torch.save(config_to_save, config_path)
    print(f"[INFO] Training configuration saved to: {config_path}")

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
