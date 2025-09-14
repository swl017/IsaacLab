# file: play_iris_mappo_rnn.py
"""
Script to play trained MAPPO RNN agents for Iris Multi-Agent task.

This script loads models trained with the train_iris_mappo_rnn.py script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play trained MAPPO RNN agents for Iris Multi-Agent task")
parser.add_argument("--model_path", type=str, help="Path to model checkpoint directory")
parser.add_argument("--experiment_dir", type=str, help="Path to experiment directory")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA-Direct-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=16, help="Number of play environments")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps)")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import torch.nn as nn
import os
import glob
import time
import gymnasium as gym
import numpy as np
import sys
from datetime import datetime

# Import MAPPO_RNN components
from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue

# Import SKRL components
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed

# Import IsaacLab components
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.dict import print_dict


def find_latest_experiment(task_name="Isaac-Iris-MA-Direct-v0"):
    """Find the latest experiment directory for the given task."""
    logs_dir = f"logs/skrl/{task_name}"
    if not os.path.exists(logs_dir):
        raise ValueError(f"No experiments found for task {task_name} in {logs_dir}")
    
    # Get all experiment directories (timestamps)
    experiment_dirs = glob.glob(os.path.join(logs_dir, "*"))
    experiment_dirs = [d for d in experiment_dirs if os.path.isdir(d)]
    
    if not experiment_dirs:
        raise ValueError(f"No experiment directories found in {logs_dir}")
    
    # Sort by modification time to get the latest
    latest_dir = max(experiment_dirs, key=os.path.getmtime)
    return latest_dir


def find_model_checkpoints(experiment_dir, possible_agents):
    """Find model checkpoints for all agents in the experiment directory."""
    agent_models = {}
    
    # Look for final agent models first
    for agent_id in possible_agents:
        final_agent_path = os.path.join(experiment_dir, f"agent_{agent_id}_final.pt")
        if os.path.exists(final_agent_path):
            agent_models[agent_id] = final_agent_path
            continue
        
        # Look for checkpoint files in subdirectories
        checkpoint_pattern = os.path.join(experiment_dir, "*", "checkpoints", f"agent_{agent_id}_*.pt")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if checkpoints:
            # Return the latest checkpoint
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            agent_models[agent_id] = latest_checkpoint
    
    if not agent_models:
        # Try loading a single shared checkpoint if individual agent files don't exist
        checkpoint_pattern = os.path.join(experiment_dir, "*", "checkpoints", "agent_*.pt")
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            # Use the same checkpoint for all agents (shared policy)
            for agent_id in possible_agents:
                agent_models[agent_id] = latest_checkpoint
    
    if not agent_models:
        raise ValueError(f"No model checkpoints found in {experiment_dir}")
    
    return agent_models


def load_training_config(experiment_dir):
    """Load the training configuration if available."""
    config_path = os.path.join(experiment_dir, "training_config.pt")
    if os.path.exists(config_path):
        return torch.load(config_path, weights_only=False)
    return None

from packaging import version
def load_checkpoint(self, path: str) -> None:
    """Load the model from the specified path

    The final storage device is determined by the constructor of the model

    :param path: Path to load the model from
    :type path: str
    """
    if version.parse(torch.__version__) >= version.parse("1.13"):
        modules = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
    else:
        modules = torch.load(path, map_location=self.device)
    if type(modules) is dict:
        for name, data in modules.items():
            module = self.checkpoint_modules.get(name, None)
            if module is not None:
                if hasattr(module, "load_state_dict"):
                    module.load_state_dict(data)
                    if hasattr(module, "eval"):
                        module.eval()
                else:
                    raise NotImplementedError
            else:
                print(f"Cannot load the {name} module. The agent doesn't have such an instance")


def main():
    """Play with trained MAPPO RNN agents."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create Isaac Lab environment
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Verify this is a multi-agent environment
    if not isinstance(env.unwrapped, DirectMARLEnv):
        raise ValueError(f"Task {args_cli.task} is not a multi-agent environment")
    
    # Get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt
    
    # Wrap for video recording if requested
    if args_cli.video:
        # Create video directory
        if args_cli.experiment_dir:
            experiment_dir = args_cli.experiment_dir
        elif args_cli.model_path:
            experiment_dir = args_cli.model_path
        else:
            experiment_dir = find_latest_experiment(args_cli.task)
        
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap environment for SKRL
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    device = env.device
    
    possible_agents = env.possible_agents
    num_agents = len(possible_agents)
    
    print(f"Playing {num_agents} agents using MAPPO with RNN")
    print(f"Agents: {possible_agents}")
    print(f"Using device: {device}")
    
    # Determine experiment directory and model paths
    if args_cli.experiment_dir:
        experiment_dir = args_cli.experiment_dir
    else:
        # Find latest experiment
        experiment_dir = find_latest_experiment(args_cli.task)
    
    print(f"Experiment directory: {experiment_dir}")
    
    # Load training configuration if available
    training_config = load_training_config(experiment_dir)
    if training_config:
        print(f"Training config found:")
        print(f"  - Original task: {training_config.get('task_name', 'Unknown')}")
        print(f"  - Training timestamp: {training_config.get('timestamp', 'Unknown')}")
        print(f"  - Number of agents: {training_config.get('num_agents', 'Unknown')}")
    
    # Find model checkpoints
    if args_cli.model_path:
        model_path = args_cli.model_path
        print(f"Using provided model path: {model_path}")
        # Assume same model for all agents if only one path is provided
        agent_model_paths = {agent_id: model_path for agent_id in possible_agents}
    else:
        print("Searching for model checkpoints...")
        agent_model_paths = find_model_checkpoints(experiment_dir, possible_agents)
        print(f"Found model checkpoints for agents: {list(agent_model_paths.keys())}")
    
    # Create shared observation spaces (same as in training)
    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        print("env.shared_observation_spaces not found. Creating shared observation space by concatenating all agent observations.")
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}
    
    # Create models and memories for each agent
    models = {}
    memories = {}
    
    # Check if all agents have the same observation and action spaces (for shared policy)
    obs_spaces_match = all(
        env.observation_spaces[agent_id] == env.observation_spaces[possible_agents[0]] 
        for agent_id in possible_agents
    )
    action_spaces_match = all(
        env.action_spaces[agent_id] == env.action_spaces[possible_agents[0]] 
        for agent_id in possible_agents
    )
    
    # Determine if we're using shared networks
    using_shared_policy = obs_spaces_match and action_spaces_match
    unique_model_paths = set(agent_model_paths.values())
    actually_shared = len(unique_model_paths) == 1
    
    if using_shared_policy:
        print("Creating shared policy and value networks (same as training).")
        
        # Create the shared networks once
        shared_policy = MAPPORNNPolicy(
            observation_space=env.observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=256,
            gru_num_layers=2,
            gru_hidden_size=256,
            num_envs=env.num_envs
        )
        
        shared_value = MAPPORNNValue(
            observation_space=shared_observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=256,
            gru_num_layers=2,
            gru_hidden_size=256,
            num_envs=env.num_envs
        )
        
        # Load the shared model weights (use the first agent's checkpoint)
        checkpoint = torch.load(list(agent_model_paths.values())[0], weights_only=False)
        shared_policy.load_state_dict(checkpoint[possible_agents[0]]["policy"])
        shared_value.load_state_dict(checkpoint[possible_agents[0]]["value"])
        print(f"Loaded shared model from: {list(agent_model_paths.values())[0]}")
        
        # Assign the same networks to all agents
        for agent_id in possible_agents:
            models[agent_id] = {}
            models[agent_id]["policy"] = shared_policy
            models[agent_id]["value"] = shared_value
    else:
        print("Creating separate policy networks for each agent.")
        # Load individual models for each agent
        for agent_id in possible_agents:
            models[agent_id] = {}
            
            # Create policy network
            models[agent_id]["policy"] = MAPPORNNPolicy(
                observation_space=env.observation_spaces[agent_id],
                action_space=env.action_spaces[agent_id],
                device=device,
                hidden_size=256,
                gru_num_layers=2,
                gru_hidden_size=256,
                num_envs=env.num_envs
            )
            
            # Create value network
            models[agent_id]["value"] = MAPPORNNValue(
                observation_space=shared_observation_spaces[agent_id],
                action_space=env.action_spaces[agent_id],
                device=device,
                hidden_size=256,
                gru_num_layers=2,
                gru_hidden_size=256,
                num_envs=env.num_envs
            )
            
            # Load model weights
            checkpoint = torch.load(agent_model_paths[agent_id], weights_only=False)
            models[agent_id]["policy"].load_state_dict(checkpoint[possible_agents[0]]["policy"])
            models[agent_id]["value"].load_state_dict(checkpoint[possible_agents[0]]["value"])
            print(f"Loaded model for agent {agent_id} from: {agent_model_paths[agent_id]}")
    
    # Create memories for each agent
    memory_cfg = {
        "memory_size": 1,  # Minimal size for evaluation
        "num_envs": env.num_envs,
        "device": device,
    }
    for agent_id in possible_agents:
        memories[agent_id] = RandomMemory(**memory_cfg)
    
    # Configure MAPPO_RNN agent (minimal config for evaluation)
    cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 1
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_spaces[possible_agents[0]], "device": device}
    cfg["shared_state_preprocessor"] = RunningStandardScaler
    cfg["shared_state_preprocessor_kwargs"] = {"size": shared_observation_spaces[possible_agents[0]], "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    # Disable all training-related features
    cfg["experiment"] = {
        "write_interval": 0,
        "checkpoint_interval": 0,
        "wandb": False
    }
    
    # Create MAPPO_RNN agent
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
    
    # Initialize agent
    agent.init()
    agent.set_mode("eval")
    
    # Reset environment
    print("Resetting the environment...")
    states, infos = env.reset()
    
    # Initialize variables for the play loop
    timestep = 0
    episode_rewards = {agent_id: 0.0 for agent_id in possible_agents}
    episode_count = 0
    
    print("Starting evaluation...")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Real-time mode: {args_cli.real_time}")
    print(f"Recording video: {args_cli.video}")
    print(f"Shared policy: {using_shared_policy and actually_shared}")
    
    # Main play loop
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run everything in inference mode
        with torch.inference_mode():
            # Get actions from agent
            actions, log_probs, outputs = agent.act(states, timestep=timestep, timesteps=10000)
            
            # Step the environment
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            
            # Accumulate rewards
            for agent_id in possible_agents:
                episode_rewards[agent_id] += rewards[agent_id].mean().item()
            
            # Check for episode end
            any_done = False
            for agent_id in possible_agents:
                if (terminated[agent_id] | truncated[agent_id]).any():
                    any_done = True
                    break
            
            if any_done:
                episode_count += 1
                avg_rewards = {agent_id: episode_rewards[agent_id] / (timestep + 1) for agent_id in possible_agents}
                print(f"Episode {episode_count} completed at timestep {timestep}")
                print(f"  Average rewards: {avg_rewards}")
                
                # Reset episode rewards
                episode_rewards = {agent_id: 0.0 for agent_id in possible_agents}
            
            # Update states for next iteration
            states = next_states
            
            # Handle RNN state resets for finished environments
            for agent_id in possible_agents:
                finished = (terminated[agent_id] | truncated[agent_id]).nonzero(as_tuple=False)
                if finished.numel():
                    # Reset RNN states for finished environments
                    for rnn_state in agent._rnn_states[agent_id]["policy"]:
                        rnn_state[:, finished[:, 0]] = 0
                    if agent.policies[agent_id] is not agent.values[agent_id]:
                        for rnn_state in agent._rnn_states[agent_id]["value"]:
                            rnn_state[:, finished[:, 0]] = 0
        
        timestep += 1
        
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                print(f"Video recording completed ({timestep} steps)")
                break
        
        # Time delay for real-time evaluation
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Optional: Exit after a certain number of episodes
        if not args_cli.video and episode_count >= 10:
            print(f"Completed {episode_count} episodes")
            break
    
    # Close the environment
    env.close()
    print("Play session completed!")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()