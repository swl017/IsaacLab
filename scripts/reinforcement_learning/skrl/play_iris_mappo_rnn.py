# file: play_iris_mappo_rnn.py
"""
Script to play trained MAPPO RNN agents for Iris Multi-Agent task.

This script loads models trained with the train_iris_mappo_rnn.py script.
"""

import debugpy
import os

# Only start debugger if an environment variable is set
if os.getenv("IDE_DEBUG_MODE") == "True":
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client() # The script will freeze here until you hit F5

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play trained MAPPO RNN agents for Iris Multi-Agent task")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint file (e.g., .../checkpoints/agent_200000.pt)")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA4-Direct-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=16, help="Number of play environments")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps)")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time")
parser.add_argument("--deterministic", action="store_true", default=True,
                    help="Use deterministic (mean) actions instead of sampling (default: True)")
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
parser.add_argument(
    "--step",
    type=str,
    default=None,
    help="The environment step at which to evaluate the agent. If not specified, evaluates from step 0",
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
import os
import time
import pickle
import gymnasium as gym
import numpy as np
import yaml
from typing import Optional, Sequence

# Import MAPPO_RNN components
from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue

# Import SKRL components
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

# Import IsaacLab components
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.dict import print_dict


def get_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    """Derive run directory from checkpoint path.

    .../run_dir/checkpoints/agent_*.pt -> .../run_dir/
    .../run_dir/agent_*_final.pt -> .../run_dir/
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    parent = os.path.dirname(checkpoint_path)
    if os.path.basename(parent) == "checkpoints":
        return os.path.dirname(parent)
    return parent


def load_agent_config(run_dir: str) -> Optional[dict]:
    """Load agent config from run_dir/params/agent.pkl (or agent.yaml as fallback).

    Prefers pickle format since YAML may contain Python objects that safe_load cannot handle.
    """
    # Try pickle first (handles Python objects correctly)
    pkl_path = os.path.join(run_dir, "params", "agent.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            config = pickle.load(f)
            return config if isinstance(config, dict) else None

    # Fallback to YAML (may fail if it contains Python objects)
    yaml_path = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if isinstance(config, dict) else None
        except Exception:
            print(f"[WARN] Could not load {yaml_path}. Using defaults.")
            return None

    return None


def get_model_hyperparams(agent_cfg: Optional[dict]) -> dict:
    """Extract model hyperparameters from agent config."""
    defaults = {
        "hidden_size": 256,
        "gru_num_layers": 2,
        "gru_hidden_size": 256,
        "initial_log_std": -0.5,
        "min_log_std": -5.0,
        "max_log_std": 0.7,
    }
    if agent_cfg is None:
        return defaults

    policy_cfg = agent_cfg.get("models", {}).get("policy", {})
    return {
        "hidden_size": policy_cfg.get("hidden_size", defaults["hidden_size"]),
        "gru_num_layers": policy_cfg.get("gru_num_layers", defaults["gru_num_layers"]),
        "gru_hidden_size": policy_cfg.get("gru_hidden_size", defaults["gru_hidden_size"]),
        "initial_log_std": policy_cfg.get("initial_log_std", defaults["initial_log_std"]),
        "min_log_std": policy_cfg.get("min_log_std", defaults["min_log_std"]),
        "max_log_std": policy_cfg.get("max_log_std", defaults["max_log_std"]),
    }


def get_sequence_length(agent_cfg: Optional[dict]) -> int:
    """Extract sequence length from agent config."""
    if agent_cfg is None:
        return 128  # default
    return agent_cfg.get("agent", {}).get("sequence_length", 128)


def load_checkpoint_weights(checkpoint_path: str, possible_agents: Sequence[str], device) -> dict:
    """Load checkpoint and return state dicts for each agent.

    Handles both SKRL format and custom final format.
    Returns: Dict[agent_id, {"policy": state_dict, "value": state_dict,
                             "state_preprocessor": state_dict, ...}]
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect format
    if "policy_state_dict" in checkpoint:
        # Custom final format - same weights for all agents (shared policy)
        print(f"[INFO] Detected custom final checkpoint format")
        return {agent_id: {
            "policy": checkpoint["policy_state_dict"],
            "value": checkpoint["value_state_dict"],
            "state_preprocessor": checkpoint.get("state_preprocessor"),
            "shared_state_preprocessor": checkpoint.get("shared_state_preprocessor"),
            "value_preprocessor": checkpoint.get("value_preprocessor"),
        } for agent_id in possible_agents}
    else:
        # SKRL format - per-agent weights (includes preprocessor states)
        print(f"[INFO] Detected SKRL checkpoint format with agents: {list(checkpoint.keys())}")
        result = {}
        for agent_id in possible_agents:
            if agent_id in checkpoint:
                agent_data = checkpoint[agent_id]
                result[agent_id] = {
                    "policy": agent_data["policy"],
                    "value": agent_data["value"],
                    "state_preprocessor": agent_data.get("state_preprocessor"),
                    "shared_state_preprocessor": agent_data.get("shared_state_preprocessor"),
                    "value_preprocessor": agent_data.get("value_preprocessor"),
                }
        return result


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
    env_cfg.enable_tiled_cameras = True
    env_cfg.use_debug_initial_step = True
    env_cfg.debug_initial_step = int(args_cli.step)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Verify this is a multi-agent environment
    if not isinstance(env.unwrapped, DirectMARLEnv):
        raise ValueError(f"Task {args_cli.task} is not a multi-agent environment")
    
    # Get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt
    
    # Get run directory and load config from checkpoint path
    run_dir = get_run_dir_from_checkpoint(args_cli.checkpoint)
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Run directory: {run_dir}")

    # Load agent configuration from saved params
    agent_cfg = load_agent_config(run_dir)
    if agent_cfg:
        print(f"[INFO] Loaded agent config from {run_dir}/params/agent.yaml")
    else:
        print(f"[WARN] No agent config found, using defaults")

    # Extract hyperparameters from config
    model_params = get_model_hyperparams(agent_cfg)
    sequence_length = get_sequence_length(agent_cfg)

    print(f"[INFO] Model hyperparameters:")
    print(f"  - hidden_size: {model_params['hidden_size']}")
    print(f"  - gru_num_layers: {model_params['gru_num_layers']}")
    print(f"  - gru_hidden_size: {model_params['gru_hidden_size']}")
    print(f"  - sequence_length: {sequence_length}")

    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(run_dir, "videos", "play"),
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

    print(f"[INFO] Playing {num_agents} agents using MAPPO with RNN")
    print(f"[INFO] Agents: {possible_agents}")
    print(f"[INFO] Using device: {device}")
    
    # Create shared observation spaces (same as in training)
    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        print("[INFO] env.shared_observation_spaces not found. Creating by concatenating all agent observations.")
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

    # Load checkpoint weights
    agent_weights = load_checkpoint_weights(args_cli.checkpoint, possible_agents, device)

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

    # Create models and memories for each agent
    models = {}
    memories = {}

    if using_shared_policy:
        print("[INFO] Creating shared policy and value networks.")

        # Create the shared networks once
        shared_policy = MAPPORNNPolicy(
            observation_space=env.observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=model_params["hidden_size"],
            gru_num_layers=model_params["gru_num_layers"],
            gru_hidden_size=model_params["gru_hidden_size"],
            num_envs=env.num_envs,
            sequence_length=sequence_length,
            initial_log_std=model_params["initial_log_std"],
            min_log_std=model_params["min_log_std"],
            max_log_std=model_params["max_log_std"],
        )

        shared_value = MAPPORNNValue(
            observation_space=shared_observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=model_params["hidden_size"],
            gru_num_layers=model_params["gru_num_layers"],
            gru_hidden_size=model_params["gru_hidden_size"],
            num_envs=env.num_envs,
            sequence_length=sequence_length,
        )

        # Load the shared model weights (use the first agent's weights)
        first_agent = possible_agents[0]
        shared_policy.load_state_dict(agent_weights[first_agent]["policy"])
        shared_value.load_state_dict(agent_weights[first_agent]["value"])
        print(f"[INFO] Loaded shared model weights from checkpoint")

        # Assign the same networks to all agents
        for agent_id in possible_agents:
            models[agent_id] = {"policy": shared_policy, "value": shared_value}
    else:
        print("[INFO] Creating separate policy networks for each agent.")
        # Load individual models for each agent
        for agent_id in possible_agents:
            # Create policy network
            policy = MAPPORNNPolicy(
                observation_space=env.observation_spaces[agent_id],
                action_space=env.action_spaces[agent_id],
                device=device,
                hidden_size=model_params["hidden_size"],
                gru_num_layers=model_params["gru_num_layers"],
                gru_hidden_size=model_params["gru_hidden_size"],
                num_envs=env.num_envs,
                sequence_length=sequence_length,
                initial_log_std=model_params["initial_log_std"],
                min_log_std=model_params["min_log_std"],
                max_log_std=model_params["max_log_std"],
            )

            # Create value network
            value = MAPPORNNValue(
                observation_space=shared_observation_spaces[agent_id],
                action_space=env.action_spaces[agent_id],
                device=device,
                hidden_size=model_params["hidden_size"],
                gru_num_layers=model_params["gru_num_layers"],
                gru_hidden_size=model_params["gru_hidden_size"],
                num_envs=env.num_envs,
                sequence_length=sequence_length,
            )

            # Load model weights
            policy.load_state_dict(agent_weights[agent_id]["policy"])
            value.load_state_dict(agent_weights[agent_id]["value"])
            models[agent_id] = {"policy": policy, "value": value}
            print(f"[INFO] Loaded model for agent {agent_id}")
    
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

    # Load preprocessor states from checkpoint (CRITICAL for correct evaluation)
    print("[INFO] Loading preprocessor states from checkpoint...")
    preprocessor_loaded = False
    for agent_id in possible_agents:
        if agent_id in agent_weights:
            # Load state preprocessor
            if agent_weights[agent_id].get("state_preprocessor") is not None:
                agent._state_preprocessor[agent_id].load_state_dict(
                    agent_weights[agent_id]["state_preprocessor"]
                )
                preprocessor_loaded = True
            # Load shared state preprocessor
            if agent_weights[agent_id].get("shared_state_preprocessor") is not None:
                agent._shared_state_preprocessor[agent_id].load_state_dict(
                    agent_weights[agent_id]["shared_state_preprocessor"]
                )
            # Load value preprocessor
            if agent_weights[agent_id].get("value_preprocessor") is not None:
                agent._value_preprocessor[agent_id].load_state_dict(
                    agent_weights[agent_id]["value_preprocessor"]
                )

    if preprocessor_loaded:
        # Verify preprocessor loaded correctly
        first_agent = possible_agents[0]
        if hasattr(agent._state_preprocessor[first_agent], 'running_mean'):
            running_mean = agent._state_preprocessor[first_agent].running_mean
            print(f"[INFO] Preprocessor loaded successfully!")
            print(f"[INFO]   running_mean (first 5): {running_mean[:5].tolist()}")
        else:
            print("[INFO] Preprocessor loaded (no running_mean attribute)")
    else:
        print("[WARN] No preprocessor states found in checkpoint - using fresh scalers")

    # Reset environment
    print("Resetting the environment...")
    states, infos = env.reset()
    
    # Initialize variables for the play loop
    timestep = 0
    episode_rewards = {agent_id: 0.0 for agent_id in possible_agents}
    episode_count = 0
    
    # Resolve deterministic mode (--deterministic)
    use_deterministic = args_cli.deterministic

    print("[INFO] Starting evaluation...")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Real-time mode: {args_cli.real_time}")
    print(f"[INFO] Recording video: {args_cli.video}")
    print(f"[INFO] Shared policy: {using_shared_policy}")
    print(f"[INFO] Deterministic actions: {use_deterministic}")
    print(f"[INFO] RNN sequence length: {sequence_length}")
    
    # Main play loop
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run everything in inference mode
        with torch.inference_mode():
            # Get actions from agent
            actions, log_probs, outputs = agent.act(states, timestep=timestep, timesteps=10000)

            # Use mean actions for deterministic evaluation
            if use_deterministic:
                for agent_id in possible_agents:
                    actions[agent_id] = outputs[agent_id]["mean_actions"]

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
        
        # # Optional: Exit after a certain number of episodes
        # if not args_cli.video and episode_count >= 10:
        #     print(f"Completed {episode_count} episodes")
        #     break
    
    # Close the environment
    env.close()
    print("Play session completed!")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()