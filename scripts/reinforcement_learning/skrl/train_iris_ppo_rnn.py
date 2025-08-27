import torch
import torch.nn as nn

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
        # return self.policy_layer(output), self.log_std_parameter, self.value_layer(output), [hidden_states]
        if role == "policy":
            return self.policy_layer(output), self.log_std_parameter, {"rnn": [hidden_states]}
        elif role == "value":
            return self.value_layer(output), None, {"rnn": [hidden_states]}



# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Iris-Gimbal2-Direct-v0", num_envs=1024, headless=True)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory_cfg = {
    "memory_size": 16,  # rollouts
    "num_envs": env.num_envs,
    "device": device,
}
memory = RandomMemory(**memory_cfg)


# instantiate the agent's models (function approximators).
# PPO_RNN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
# models = {
#     "policy": Policy(env.observation_space, env.action_space, device, num_envs=env.num_envs),
#     "value": Policy(env.observation_space, env.action_space, device, num_envs=env.num_envs),
# }
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["value"] = models["policy"]

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 1  # 16 * 512 / 8192
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 16
cfg["experiment"]["checkpoint_interval"] = 80
cfg["experiment"]["directory"] = "runs/torch/Isaac-Cartpole-v0"

agent = PPO_RNN(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1600, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/IsaacOrbit-Isaac-Cartpole-v0-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
# trainer.eval()


