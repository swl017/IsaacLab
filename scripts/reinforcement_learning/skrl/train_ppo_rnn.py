# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
    choices=["AMP", "PPO", "IPPO", "MAPPO", "PPO_RNN"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime
import copy

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner
# if args_cli.ml_framework.startswith("torch"):
#     from skrl.utils.runner.torch import Runner
# elif args_cli.ml_framework.startswith("jax"):
#     from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

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
        return self.policy_layer(output), self.log_std_parameter, self.value_layer(output), [hidden_states]
    
# def _process_cfg(cfg: dict) -> dict:
#         """Convert simple types to skrl classes/components

#         :param cfg: A configuration dictionary

#         :return: Updated dictionary
#         """
#         _direct_eval = [
#             "learning_rate_scheduler",
#             "shared_state_preprocessor",
#             "state_preprocessor",
#             "value_preprocessor",
#             "amp_state_preprocessor",
#             "noise",
#             "smooth_regularization_noise",
#         ]

#         def reward_shaper_function(scale):
#             def reward_shaper(rewards, *args, **kwargs):
#                 return rewards * scale

#             return reward_shaper

#         def update_dict(d):
#             for key, value in d.items():
#                 if isinstance(value, dict):
#                     update_dict(value)
#                 else:
#                     if key in _direct_eval:
#                         if isinstance(value, str):
#                             d[key] = eval(value)
#                     elif key.endswith("_kwargs"):
#                         d[key] = value if value is not None else {}
#                     elif key in ["rewards_shaper_scale"]:
#                         d["rewards_shaper"] = reward_shaper_function(value)
#             return d

#         return update_dict(copy.deepcopy(cfg))

def _process_agent_cfg(cfg: dict) -> dict:
    """Process agent config to convert string class paths to class objects."""
    config = copy.deepcopy(cfg)

    # learning rate scheduler
    scheduler_class = config.get("learning_rate_scheduler", None)
    if isinstance(scheduler_class, str):
        config["learning_rate_scheduler"] = eval(scheduler_class)

    # state preprocessor
    state_preprocessor_class = config.get("state_preprocessor", None)
    if isinstance(state_preprocessor_class, str):
        config["state_preprocessor"] = eval(state_preprocessor_class)

    # value preprocessor
    value_preprocessor_class = config.get("value_preprocessor", None)
    if isinstance(value_preprocessor_class, str):
        config["value_preprocessor"] = eval(value_preprocessor_class)

    return config

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # 1. Instantiate the RNN model
    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device, num_envs=args_cli.num_envs),
        "value": Policy(env.observation_space, env.action_space, env.device, num_envs=args_cli.num_envs),
    }

    # 2. Configure the memory
    #    RNNs require a memory that can handle sequences
    memory_cfg = {
        "memory_size": agent_cfg["agent"]["rollouts"],  # Use rollouts from your config
        "num_envs": env.num_envs,
        "device": env.device,
    }
    memory = RandomMemory(**memory_cfg)

    # 3. Instantiate the PPO_RNN agent
    #    The agent_cfg dictionary from hydra is still used for hyperparameters
    # processed_agent_cfg = _process_agent_cfg(agent_cfg["agent"])
    agent = PPO_RNN(models=models,
                  memory=memory,
                  cfg=agent_cfg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=env.device)


    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner_cfg = {"trainer": agent_cfg["trainer"], "seed": agent_cfg["seed"]}
    runner = Runner(env=env, cfg=agent_cfg, agent=agent, models=models)
    # runner = Runner(env, cfg=runner_cfg, agent=agent)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
