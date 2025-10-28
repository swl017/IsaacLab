# file: train_iris_mappo_rnn.py
import argparse
import torch
import torch.nn as nn
import os
import shutil
import sys
from datetime import datetime
import gymnasium as gym
import numpy as np

from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue

# Import SKRL components
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import yaml


# Parse arguments
parser = argparse.ArgumentParser(description="Train MAPPO RNN agents for Iris Multi-Agent task")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA2-Direct-Delay-v0", help="Name of the task.")
parser.add_argument("--tuning-param", type=str, help="Custom tuning parameter")
parser.add_argument("--experiment-name", type=str, default="sweep14_noise_o_delay_o_rnn_obsdim29", help="Name of the experiment")
args_cli = parser.parse_args()

# Set seed for reproducibility
set_seed(42)

# Load and wrap the Isaac Lab multi-agent environment
task_name = args_cli.task
experiment_name = args_cli.experiment_name
env = load_isaaclab_env(task_name=task_name, num_envs=1024, headless=True)
env = wrap_env(env)
device = env.device

possible_agents = env.possible_agents
num_agents = len(possible_agents)

print(f"Training {num_agents} agents using MAPPO with RNN")
print(f"Agents: {possible_agents}")

# Create models for each agent
models = {}
memories = {}

# The shared space is the concatenation of all individual agent observations.
try:
    # This will fail, as per the error message
    shared_observation_spaces = env.shared_observation_spaces
except AttributeError:
    print("env.shared_observation_spaces not found. Creating a shared observation space by concatenating all agent observations.")
    # Assuming all observation spaces are Box and 1D
    obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
    shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
    # Each agent's value function will receive the same shared observation space
    shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}
except Exception as e:
    print(f"Error concatenating observation spaces: {e}")
    sys.exit(1)

# Configure MAPPO_RNN
cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 64
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 8
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-5
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.004}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_spaces[possible_agents[0]], "device": device}
cfg["shared_state_preprocessor"] = RunningStandardScaler
cfg["shared_state_preprocessor_kwargs"] = {"size": shared_observation_spaces[possible_agents[0]], "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}


# Create a single shared policy network
# First, check if all agents have the same observation and action spaces
obs_spaces_match = all(
    env.observation_spaces[agent_id] == env.observation_spaces[possible_agents[0]] 
    for agent_id in possible_agents
)
action_spaces_match = all(
    env.action_spaces[agent_id] == env.action_spaces[possible_agents[0]] 
    for agent_id in possible_agents
)

if not obs_spaces_match or not action_spaces_match:
    # Separate policy, shared value
    print("Agents have different observation or action spaces. Creating separate policy networks for each agent, but sharing the value network.")
    sys.exit(1)
    for agent_id in possible_agents:
        # Each agent gets its own policy and value function
        models[agent_id] = {}
        
        # Policy network (uses local observations)
        models[agent_id]["policy"] = MAPPORNNPolicy(
            observation_space=env.observation_spaces[agent_id],
            action_space=env.action_spaces[agent_id],
            device=device,
            hidden_size=256,
            gru_num_layers=2,
            gru_hidden_size=256,
            num_envs=env.num_envs
        )
        
        if agent_id != possible_agents[0]:
            models[agent_id]["value"] = models[possible_agents[0]]["value"]
        else:
            models[agent_id]["value"] = MAPPORNNValue(
                observation_space=shared_observation_spaces[agent_id],
                action_space=env.action_spaces[agent_id],
                device=device,
                hidden_size=256,
                gru_num_layers=2,
                gru_hidden_size=256,
                num_envs=env.num_envs
            )

        memory_cfg = {
            "memory_size": cfg["rollouts"],
            "num_envs": env.num_envs,
            "device": device,
        }
        memories[agent_id] = RandomMemory(**memory_cfg)
else:
    print("All agents have the same observation and action spaces. Creating shared policy and value networks.")
    # Create the shared policy network once
    shared_policy = MAPPORNNPolicy(
        observation_space=env.observation_spaces[possible_agents[0]],  # Use first agent's space as reference
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=256,
        gru_num_layers=2,
        gru_hidden_size=256,
        num_envs=env.num_envs
    )

    # Create the shared value network once
    shared_value = MAPPORNNValue(
        observation_space=shared_observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=256,
        gru_num_layers=2,
        gru_hidden_size=256,
        num_envs=env.num_envs
    )

    # Assign the same networks to all agents
    for agent_id in possible_agents:
        models[agent_id] = {}
        models[agent_id]["policy"] = shared_policy  # All agents share the same policy
        models[agent_id]["value"] = shared_value    # All agents share the same value function
        
        # Create memory for each agent (still separate memories)
        memory_cfg = {
            "memory_size": cfg["rollouts"],
            "num_envs": env.num_envs,
            "device": device,
        }
        memories[agent_id] = RandomMemory(**memory_cfg)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"logs/skrl/iris_ma2/{timestamp}_{experiment_name}"
os.makedirs(experiment_dir, exist_ok=True)

# Copy training script for reproducibility
current_script = __file__
script_backup_path = os.path.join(experiment_dir, "train_script.py")
shutil.copy2(current_script, script_backup_path)
print(f"Training script copied to: {script_backup_path}")

# Copy MAPPO_RNN implementation
# mappo_rnn_backup_path = os.path.join(experiment_dir, "mappo_rnn.py")
# shutil.copy2("mappo_rnn.py", mappo_rnn_backup_path)
# print(f"MAPPO_RNN implementation copied to: {mappo_rnn_backup_path}")

env_folder = "/home/usrg/IsaacPX4/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma2"
shutil.copytree(env_folder, os.path.join(experiment_dir, "iris_ma2"))
print(f"Env script copied to: {experiment_dir}")

cfg["experiment"]["directory"] = experiment_dir
cfg["experiment"]["experiment_name"] = f"{experiment_name}"
cfg["experiment"]["wandb"] = False

# Instantiate MAPPO_RNN agent
agent = MAPPO_RNN(
    possible_agents=possible_agents,
    models=models,
    memories=memories,
    observation_spaces=env.observation_spaces,
    action_spaces=env.action_spaces,
    device=device,
    cfg=cfg,
    shared_observation_spaces=shared_observation_spaces
)

# Configure trainer
cfg_trainer = {
    "timesteps": 200000,
    "headless": True,
    "environment_info": "log"
}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Start training
print(f"Starting decentralized MAPPO training with RNN...")
print(f"Experiment directory: {experiment_dir}")
print(f"Number of agents: {num_agents}")
print(f"Each agent has its own policy: {'Yes' if len(set(id(models[a]['policy']) for a in possible_agents)) == num_agents else 'No (shared)'}")
print(f"Each agent has its own value function: {'Yes' if len(set(id(models[a]['value']) for a in possible_agents)) == num_agents else 'No (shared)'}")

trainer.train()

# Save trained agents
for agent_id in possible_agents:
    agent_path = os.path.join(experiment_dir, f"agent_{agent_id}_final.pt")
    torch.save({
        'policy_state_dict': models[agent_id]["policy"].state_dict(),
        'value_state_dict': models[agent_id]["value"].state_dict(),
    }, agent_path)
    print(f"Agent {agent_id} saved to: {agent_path}")

# Save training configuration
config_path = os.path.join(experiment_dir, "training_config.yaml")
config_to_save = {
    "agent_config": cfg,
    "trainer_config": cfg_trainer,
    "task_name": task_name,
    "num_agents": num_agents,
    "possible_agents": possible_agents,
    "timestamp": timestamp,
    "device": str(device)
}

with open(config_path, 'w') as f:
    yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
print(f"Training configuration saved to: {config_path}")

print(f"Decentralized MAPPO training completed!")