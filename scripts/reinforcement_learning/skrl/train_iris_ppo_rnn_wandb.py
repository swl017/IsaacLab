import argparse

parser = argparse.ArgumentParser(description="Train PPO RNN agent for Iris Gimbal task")
parser.add_argument("--tuning-param", type=str, help="Custom tuning parameter")
parser.add_argument("--wandb-project", type=str, default="iris-gimbal-ppo-rnn", help="Wandb project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity/team name")
parser.add_argument("--wandb-run-name", type=str, default=None, help="Custom wandb run name")
parser.add_argument("--wandb-tags", nargs="*", default=[], help="Tags for wandb run")
parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")
args_cli = parser.parse_args()

import torch
import torch.nn as nn
import os
import shutil
from datetime import datetime
import wandb

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed(42)


# define RNN-based policy model
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
        return {"rnn": {"sequence_length": 20,
                        "sizes": [(self.gru.num_layers, self.num_envs, self.gru.hidden_size)]}}

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        hidden_states = inputs["rnn"][0]
        # view shape: (batch_size, sequence_length, input_size)
        gru_input = x.view(-1, 1, x.shape[-1])
        gru_output, hidden_states = self.gru(gru_input, hidden_states)
        # view shape: (batch_size, hidden_size)
        output = gru_output.view(-1, self.gru.hidden_size)
        # return self.policy_layer(output), self.log_std_parameter, self.value_layer(output), [hidden_states]
        if role == "policy":
            return self.policy_layer(output), self.log_std_parameter, {"rnn": [hidden_states]}
        elif role == "value":
            return self.value_layer(output), None, {"rnn": [hidden_states]}



# load and wrap the Isaac Lab environment
task_name = "Isaac-Iris-Gimbal2-Zoom-Direct-v0"
experiment_name = "IrisGimbal2Zoom_wandb_test"
env = load_isaaclab_env(task_name=task_name, num_envs=4096, headless=True)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory_cfg = {
    "memory_size": 32,  # rollouts
    "num_envs": env.num_envs,
    "device": device,
}
memory = RandomMemory(**memory_cfg)


# instantiate the agent's models (function approximators).
# PPO_RNN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["value"] = models["policy"]

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 32  # memory_size
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
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
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"logs/skrl/{task_name}/{timestamp}"
os.makedirs(experiment_dir, exist_ok=True)

# Copy the current training script to the experiment directory for reproducibility
current_script = __file__
script_backup_path = os.path.join(experiment_dir, "train_script.py")
shutil.copy2(current_script, script_backup_path)
print(f"Training script copied to: {script_backup_path}")
env_file = "/home/usrg/IsaacPX4/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_gimbal2_zoom/iris_gimbal2_zoom_env.py"
env_backup_path = os.path.join(experiment_dir, "env_script.py")
shutil.copy2(env_file, env_backup_path)
print(f"Env script copied to: {env_backup_path}")

# Configure experiment directory and wandb
cfg["experiment"]["directory"] = experiment_dir
cfg["experiment"]["experiment_name"] = f"{experiment_name}"
cfg["experiment"]["wandb"] = not args_cli.disable_wandb  # Enable wandb unless explicitly disabled

# Initialize wandb if enabled
if cfg["experiment"]["wandb"]:
    # Prepare wandb configuration
    wandb_config = {
        "task_name": task_name,
        "algorithm": "PPO_RNN",
        "num_envs": env.num_envs,
        "rollouts": cfg["rollouts"],
        "learning_epochs": cfg["learning_epochs"],
        "mini_batches": cfg["mini_batches"],
        "discount_factor": cfg["discount_factor"],
        "lambda": cfg["lambda"],
        "learning_rate": cfg["learning_rate"],
        "grad_norm_clip": cfg["grad_norm_clip"],
        "ratio_clip": cfg["ratio_clip"],
        "value_clip": cfg["value_clip"],
        "entropy_loss_scale": cfg["entropy_loss_scale"],
        "value_loss_scale": cfg["value_loss_scale"],
        "kl_threshold": cfg["learning_rate_scheduler_kwargs"]["kl_threshold"],
        "hidden_size": 256,
        "gru_num_layers": 1,
        "gru_hidden_size": 256,
        "sequence_length": 20,
        "timestamp": timestamp,
        "device": str(device),
    }
    
    # Add custom tuning parameter if provided
    if args_cli.tuning_param:
        wandb_config["tuning_param"] = args_cli.tuning_param
    
    # Determine run name
    run_name = args_cli.wandb_run_name or f"{task_name}_{timestamp}"
    
    # Prepare tags
    tags = ["PPO_RNN", task_name, "IsaacLab"] + args_cli.wandb_tags
    
    # Initialize wandb
    wandb.init(
        project=args_cli.wandb_project,
        entity=args_cli.wandb_entity,
        name=run_name,
        config=wandb_config,
        tags=tags,
        dir=experiment_dir,
        sync_tensorboard=True,  # Automatically sync tensorboard logs
        monitor_gym=False,  # Don't monitor gym metrics (handled by skrl)
        save_code=True,  # Save the code
    )
    
    # Log the experiment directory as artifact
    artifact = wandb.Artifact(
        name=f"experiment_scripts_{timestamp}",
        type="code",
        description="Training and environment scripts"
    )
    artifact.add_file(script_backup_path)
    artifact.add_file(env_backup_path)
    wandb.log_artifact(artifact)
    
    print(f"Wandb initialized with project: {args_cli.wandb_project}")
    print(f"Run name: {run_name}")
    print(f"Run URL: {wandb.run.url}")
else:
    print("Wandb logging disabled")

# Configure additional wandb settings for skrl
if cfg["experiment"]["wandb"]:
    cfg["experiment"]["wandb_kwargs"] = {
        "project": args_cli.wandb_project,
        "entity": args_cli.wandb_entity,
        "reinit": False,  # Don't reinitialize since we already did
    }

agent = PPO_RNN(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# configure and instantiate the RL trainer
cfg_trainer = {
    "timesteps": 80800, 
    "headless": True,
    "environment_info": "log"  # Enable logging of environment extras
}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
print(f"Starting training...")
print(f"Experiment directory: {experiment_dir}")
print(f"Logging interval: {cfg['experiment']['write_interval']} timesteps")
print(f"Checkpoint interval: {cfg['experiment']['checkpoint_interval']} timesteps")

try:
    trainer.train()
    
    # Log final metrics to wandb if enabled
    if cfg["experiment"]["wandb"]:
        wandb.log({
            "training_completed": True,
            "final_timesteps": cfg_trainer["timesteps"]
        })
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    if cfg["experiment"]["wandb"]:
        wandb.log({"training_interrupted": True})
except Exception as e:
    print(f"\nTraining failed with error: {e}")
    if cfg["experiment"]["wandb"]:
        wandb.log({"training_failed": True, "error": str(e)})
    raise
finally:
    # save the trained agent
    agent_path = os.path.join(experiment_dir, "final_agent.pt")
    agent.save(agent_path)
    print(f"Agent saved to: {agent_path}")
    
    # save training configuration for reproducibility
    config_path = os.path.join(experiment_dir, "training_config.pt")
    torch.save({
        "agent_config": cfg,
        "trainer_config": cfg_trainer,
        "task_name": task_name,
        "timestamp": timestamp,
        "device": str(device),
        "wandb_project": args_cli.wandb_project if cfg["experiment"]["wandb"] else None,
        "wandb_run_id": wandb.run.id if cfg["experiment"]["wandb"] else None,
        "wandb_run_name": wandb.run.name if cfg["experiment"]["wandb"] else None,
    }, config_path)
    print(f"Training configuration saved to: {config_path}")
    
    # Log final model as artifact if wandb is enabled
    if cfg["experiment"]["wandb"]:
        model_artifact = wandb.Artifact(
            name=f"trained_model_{timestamp}",
            type="model",
            description="Final trained PPO RNN model"
        )
        model_artifact.add_file(agent_path)
        model_artifact.add_file(config_path)
        wandb.log_artifact(model_artifact)
        
        # Finish wandb run
        wandb.finish()
        print("Wandb run finished successfully")

print("Training completed.")