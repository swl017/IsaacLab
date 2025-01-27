"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import carb

from robust_tracking.simulator.launcher import simulation_app, args_cli
from robust_tracking.simulator.pegasus import PegasusApp
simulation_app = simulation_app
args_cli = args_cli

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
import omni.timeline
from omni.isaac.core.world import World

# Isaac Lab
from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg


# Custom code
import os
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import wandb
from wandb.integration.sb3 import WandbCallback

import gym
from gymnasium import spaces, Env
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import robust_tracking.agents as agents
import robust_tracking.ros2.ros2_node as ros2_node
from robust_tracking.ros2.ros2_node import ROS2SpinCallback, ISAACStepCallback

env = Env()
register(
    id="DroneEnv-v0",
    entry_point="robust_tracking.environment.drone_env:DroneEnv",
    max_episode_steps=3000,
    # kwargs={"num_drones": 2},
)
num_ego = 2
num_target = 1

pg_app = PegasusApp()
rosnode = ros2_node.ROS2Node(num_ego, num_target)

def make_env(rank, log_dir, video_record=False):
   def _init():
       env = agents.DroneEnv(num_agents=num_ego, ros2_node=rosnode)
       env = Monitor(env, os.path.join(log_dir, f'drone_env_{rank}'))
       if video_record and rank == 0:
            if args_cli.video:
                video_kwargs = {
                    "video_folder": os.path.join(log_dir, "videos", "train"),
                    "step_trigger": lambda step: step % args_cli.video_interval == 0,
                    "video_length": args_cli.video_length,
                    "disable_logger": True,
                }
                print("[INFO] Recording videos during training.")
                print_dict(video_kwargs, nesting=4)
                env = gym.wrappers.RecordVideo(env, **video_kwargs)
   return _init

def train(env_cfg, agent_cfg, args):
   # Setup logging directory
   timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
   log_dir = os.path.join("logs", "sb3", args.task, timestamp)
   os.makedirs(log_dir, exist_ok=True)
   
   # Save configs
   os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
   dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
   dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
   
   # Initialize W&B
   run = wandb.init(
       project="drone_tracking",
       config={**env_cfg, **agent_cfg},
       sync_tensorboard=True
   )
   
   # Create vectorized environment
   env = SubprocVecEnv([make_env(i, log_dir, i == 0 and args.video) for i in range(args.num_envs)])
   
   # Normalize observations and rewards
   env = VecNormalize(
       env,
       norm_obs=agent_cfg.get("normalize_input", True),
       norm_reward=agent_cfg.get("normalize_value", True),
       clip_obs=agent_cfg.get("clip_obs", 10.),
       gamma=agent_cfg["gamma"]
   )
   
   # Configure logger
   logger = configure(log_dir, ["stdout", "tensorboard", "csv"])
   
   # Initialize PPO
   model = PPO(
       policy=agent_cfg["policy"],
       env=env,
       verbose=1,
       **{k: v for k, v in agent_cfg.items() 
          if k not in ["policy", "n_timesteps", "normalize_input", "normalize_value"]}
   )
   model.set_logger(logger)
   
   # Setup callbacks
   callbacks = [
       CheckpointCallback(
           save_freq=1000,
           save_path=os.path.join(log_dir, "checkpoints"),
           name_prefix="model"
       ),
       WandbCallback(
           gradient_save_freq=100,
           model_save_path=os.path.join(log_dir, "wandb_models"),
           verbose=2
       ),
       ROS2SpinCallback(rosnode),
       ISAACStepCallback(pg_app.world)
   ]
   
   # Train
   model.learn(
       total_timesteps=agent_cfg["n_timesteps"],
       callback=callbacks
   )
   
   # Save final model and normalization stats
   model.save(os.path.join(log_dir, "model_final"))
   env.save(os.path.join(log_dir, "vec_normalize.pkl"))
   
   wandb.finish()

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    train(env_cfg, agent_cfg, args_cli)

if __name__ == "__main__":
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    rosnode.is_node_ready(pg_app.world)
    carb.log_warn("Ready to train RL agent.")
    main()
    carb.log_warn("PegasusApp Simulation App is closing.")
    rosnode.shutdown()
    timeline.stop()
    simulation_app.close()