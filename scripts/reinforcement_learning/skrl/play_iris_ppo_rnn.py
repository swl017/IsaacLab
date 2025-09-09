"""
Script to play a trained PPO RNN agent for Iris Gimbal task.

This script loads models trained with the custom train_iris_ppo_rnn.py script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained PPO RNN agent for Iris Gimbal task")
parser.add_argument("--model-path", type=str, help="Path to model checkpoint (.pt file)")
parser.add_argument("--experiment-dir", type=str, help="Path to experiment directory")
parser.add_argument("--task", type=str, default="Isaac-Iris-Gimbal2-Zoom-Direct-v0", help="Task name")
parser.add_argument("--num-envs", type=int, default=16, help="Number of play environments")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps)")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
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
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
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

# Import all SKRL components after SimulationApp is initialized
from skrl.agents.torch.ppo import PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab.utils.dict import print_dict
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

def create_policy_class():
    """Create the Policy class after importing SKRL components."""
    from skrl.models.torch import GaussianMixin, Model
    
    # define RNN-based policy model (same as training script)
    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, hidden_size=256, gru_num_layers=1, gru_hidden_size=256, num_envs=1):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True)

            self.net = nn.Sequential(nn.Linear(self.num_observations, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU())
            self.gru = nn.GRU(hidden_size, gru_hidden_size, num_layers=gru_num_layers, batch_first=True, device=device, dtype=torch.float32)
            self.policy_layer = nn.Linear(gru_hidden_size, self.num_actions, device=device, dtype=torch.float32)
            self.value_layer = nn.Linear(gru_hidden_size, 1, device=device, dtype=torch.float32)
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))
            self.num_envs = num_envs

        def act(self, inputs, role):
            if role == "policy":
                return GaussianMixin.act(self, inputs, role)
            elif role == "value":
                values, _, outputs = self.compute(inputs, role)
                return values, None, outputs
              
        def get_specification(self):
            # The agent will look for the "rnn" key to setup the recurrent states
            # and "sequence_length" for the sampler
            return {"rnn": {"sequence_length": 10,
                            "sizes": [(self.gru.num_layers, self.num_envs, self.gru.hidden_size)]}}

        def compute(self, inputs, role):
            x = self.net(inputs["states"])
            hidden_states = inputs["rnn"][0]
            # view shape: (batch_size, sequence_length, input_size)
            gru_input = x.view(-1, 1, x.shape[-1])
            gru_output, hidden_states = self.gru(gru_input, hidden_states)
            # view shape: (batch_size, hidden_size)
            output = gru_output.view(-1, self.gru.hidden_size)
            if role == "policy":
                return self.policy_layer(output), self.log_std_parameter, {"rnn": [hidden_states]}
            elif role == "value":
                return self.value_layer(output), None, {"rnn": [hidden_states]}
    
    return Policy


def find_latest_experiment(task_name="Isaac-Iris-Gimbal2-Direct-v0"):
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


def find_model_checkpoint(experiment_dir):
    """Find the model checkpoint in the experiment directory."""
    # Look for final agent first, then checkpoints
    final_agent_path = os.path.join(experiment_dir, "final_agent.pt")
    if os.path.exists(final_agent_path):
        return final_agent_path
    
    # Look for checkpoint files
    checkpoint_pattern = os.path.join(experiment_dir, "*", "checkpoints", "agent_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if checkpoints:
        # Return the latest checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        return latest_checkpoint
    
    raise ValueError(f"No model checkpoints found in {experiment_dir}")


def load_training_config(experiment_dir):
    """Load the training configuration if available."""
    config_path = os.path.join(experiment_dir, "training_config.pt")
    if os.path.exists(config_path):
        return torch.load(config_path, weights_only=False)
    return None


def main():
    """Play with trained PPO RNN agent."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create Policy class
    Policy = create_policy_class()

    algorithm = args_cli.algorithm.lower()
    
    # Determine model path
    if args_cli.model_path:
        model_path = args_cli.model_path
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        experiment_dir = os.path.dirname(model_path)
    elif args_cli.experiment_dir:
        experiment_dir = args_cli.experiment_dir
        model_path = find_model_checkpoint(experiment_dir)
    else:
        # Find latest experiment
        experiment_dir = find_latest_experiment(args_cli.task)
        model_path = find_model_checkpoint(experiment_dir)
    
    print(f"Loading model from: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    
    # Load training configuration if available
    training_config = load_training_config(experiment_dir)
    if training_config:
        print(f"Training config found - Original task: {training_config.get('task_name', 'Unknown')}")
        print(f"Training timestamp: {training_config.get('timestamp', 'Unknown')}")
        # create isaac environment

    env_cfg = parse_env_cfg(
        # training_config.get('task_name', 'Unknown'), device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # Load and wrap the Isaac Lab environment
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`
    # env = load_isaaclab_env(task_name="Isaac-Iris-Gimbal2-Direct-v0", num_envs=args_cli.num_envs, headless=False)
    # env = wrap_env(env)
    
    device = env.device
    print(f"Using device: {device}")
    
    # Get environment step dt for real-time evaluation
    dt = 1.0 / 60.0  # Default to 60 FPS if not available
    try:
        dt = env.step_dt if hasattr(env, 'step_dt') else env.unwrapped.step_dt
    except AttributeError:
        pass
    
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("Recording videos during play.")
        env = gym.wrappers.RecordVideo(env.unwrapped, **video_kwargs)
        env = wrap_env(env)
    
    # instantiate a memory as rollout buffer
    memory_cfg = {
        "memory_size": 100,  # rollouts (same as training)
        "num_envs": env.num_envs,
        "device": device,
    }
    memory = RandomMemory(**memory_cfg)
    
    # instantiate the agent's models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, num_envs=env.num_envs)
    models["value"] = models["policy"]
    
    # configure the agent (use same config as training)
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 100
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 1
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 5e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = None
    cfg["time_limit_bootstrap"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    # Disable logging for play
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["experiment"]["wandb"] = False
    
    # Create agent
    agent = PPO_RNN(models=models,
                    memory=memory,
                    cfg=cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    
    # Load the trained model
    print("Loading trained model...")
    agent.load(model_path)
    print("Model loaded successfully!")
    agent.init(cfg)
    # Set agent to evaluation mode
    agent.set_running_mode("eval")
    
    # reset the underlying environment to get the initial observation dictionary
    print("Resetting the environment...")
    obs_dict, _ = env.unwrapped.reset()
    # extract the policy tensor, ensure it's on the correct device, and add a batch dimension if needed
    obs = obs_dict["policy"].to(device)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    # initialize variables for the play loop
    done = False
    timestep = 0

    print("Starting evaluation...")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Real-time mode: {args_cli.real_time}")
    print(f"Recording video: {args_cli.video}")

    # Simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run everything in inference mode
        with torch.inference_mode():
            # Agent stepping: get actions and the new RNN state from the agent
            # The agent uses its internal state (_rnn_initial_states) and returns the new state in 'outputs'
            actions, _, outputs = agent.act(obs, timestep=0, timesteps=0)
            
            # Extract deterministic actions for evaluation (mean of the distribution)
            # The 'actions' tensor is now used as a fallback if 'mean_actions' isn't available
            deterministic_actions = outputs.get("mean_actions", actions)
                
            # Step the underlying environment with the deterministic actions
            obs_dict, rewards, terminated, truncated, info = env.unwrapped.step(deterministic_actions)
            
            # Extract the policy tensor for the next iteration
            obs = obs_dict["policy"].to(device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            # MANUALLY CYCLE THE RNN STATE FOR THE NEXT STEP
            # This is the key part you correctly identified was missing.
            agent._rnn_initial_states["policy"] = outputs.get("rnn", [])

            # Reset the RNN state for environments that have finished
            dones = (terminated | truncated).to(device=device, dtype=torch.int)
            if dones.any():
                # iterate over the RNN states (one for each layer, though typically 1)
                for i in range(len(agent._rnn_initial_states["policy"])):
                    # hidden states have shape (num_layers, num_envs, hidden_size)
                    hidden_state = agent._rnn_initial_states["policy"][i]
                    # set the state to zero for the finished environments
                    hidden_state[:, dones.nonzero(as_tuple=True)[0], :] = 0.0
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        # Time delay for real-time evaluation
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Close the environment
    env.close()
    print("Play session completed!")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()