import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Mapping, Optional, Sequence, Union
import copy
import itertools
from packaging import version
import gymnasium

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG


# MAPPO_RNN Configuration
MAPPO_RNN_DEFAULT_CONFIG = copy.deepcopy(PPO_DEFAULT_CONFIG)
MAPPO_RNN_DEFAULT_CONFIG.update({
    "shared_state_preprocessor": None,
    "shared_state_preprocessor_kwargs": {},
})


class MAPPO_RNN(MultiAgent):
    """Multi-Agent Proximal Policy Optimization with RNN Support"""
    
    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        shared_observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
    ) -> None:
        _cfg = copy.deepcopy(MAPPO_RNN_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
        )
        
        self.shared_observation_spaces = shared_observation_spaces
        
        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}
        
        # RNN states for each agent
        self._rnn_states = {
            uid: {"policy": [], "value": []} for uid in self.possible_agents
        }
        self._rnn_initial_states = {
            uid: {"policy": [], "value": []} for uid in self.possible_agents
        }
        self._rnn_sequence_lengths = {uid: 1 for uid in self.possible_agents}
        
        # configuration
        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0
        
        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])
        self._ratio_clip = self._as_dict(self.cfg["ratio_clip"])
        self._value_clip = self._as_dict(self.cfg["value_clip"])
        self._clip_predicted_values = self._as_dict(self.cfg["clip_predicted_values"])
        
        self._value_loss_scale = self._as_dict(self.cfg["value_loss_scale"])
        self._entropy_loss_scale = self._as_dict(self.cfg["entropy_loss_scale"])
        
        self._kl_threshold = self._as_dict(self.cfg["kl_threshold"])
        
        self._learning_rate = self._as_dict(self.cfg["learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])
        
        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])
        self._shared_state_preprocessor = self._as_dict(self.cfg["shared_state_preprocessor"])
        self._shared_state_preprocessor_kwargs = self._as_dict(self.cfg["shared_state_preprocessor_kwargs"])
        self._value_preprocessor = self._as_dict(self.cfg["value_preprocessor"])
        self._value_preprocessor_kwargs = self._as_dict(self.cfg["value_preprocessor_kwargs"])
        
        self._discount_factor = self._as_dict(self.cfg["discount_factor"])
        self._lambda = self._as_dict(self.cfg["lambda"])
        
        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]
        
        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self._as_dict(self.cfg["time_limit_bootstrap"])
        
        self._mixed_precision = self.cfg["mixed_precision"]
        
        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
        
        # set up optimizers
        self.optimizers = {}
        self.schedulers = {}
        
        for uid in self.possible_agents:
            policy = self.policies[uid]
            value = self.values[uid]
            if policy is not None and value is not None:
                if policy is value:
                    optimizer = torch.optim.Adam(policy.parameters(), lr=self._learning_rate[uid])
                else:
                    optimizer = torch.optim.Adam(
                        itertools.chain(policy.parameters(), value.parameters()), lr=self._learning_rate[uid]
                    )
                self.optimizers[uid] = optimizer
                if self._learning_rate_scheduler[uid] is not None:
                    self.schedulers[uid] = self._learning_rate_scheduler[uid](
                        optimizer, **self._learning_rate_scheduler_kwargs[uid]
                    )
            
            self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]
            
            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor
            
            if self._shared_state_preprocessor[uid] is not None:
                self._shared_state_preprocessor[uid] = self._shared_state_preprocessor[uid](
                    **self._shared_state_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["shared_state_preprocessor"] = self._shared_state_preprocessor[uid]
            else:
                self._shared_state_preprocessor[uid] = self._empty_preprocessor
            
            if self._value_preprocessor[uid] is not None:
                self._value_preprocessor[uid] = self._value_preprocessor[uid](**self._value_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor
    
    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        
        # create tensors in memories
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(
                    name="shared_states", size=self.shared_observation_spaces[uid], dtype=torch.float32
                )
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)
                
                # RNN specifications for each agent
                self._rnn_tensors_names = {}
                
                # Initialize RNN states for each agent's policy
                policy_spec = self.policies[uid].get_specification()
                policy_rnn_spec = policy_spec.get("rnn", {})
                if policy_rnn_spec:
                    self._rnn_sequence_lengths[uid] = policy_rnn_spec.get("sequence_length", 1)
                    self._rnn_tensors_names[uid] = []
                    
                    for i, size in enumerate(policy_rnn_spec.get("sizes", [])):
                        # Create tensors in memory for RNN states
                        self.memories[uid].create_tensor(
                            name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                        )
                        self._rnn_tensors_names[uid].append(f"rnn_policy_{i}")
                        # Initialize RNN states
                        self._rnn_initial_states[uid]["policy"].append(
                            torch.zeros(size, dtype=torch.float32, device=self.device)
                        )
                
                # Initialize RNN states for each agent's value function
                if self.policies[uid] is self.values[uid]:
                    self._rnn_initial_states[uid]["value"] = self._rnn_initial_states[uid]["policy"]
                else:
                    value_spec = self.values[uid].get_specification()
                    value_rnn_spec = value_spec.get("rnn", {})
                    if value_rnn_spec:
                        for i, size in enumerate(value_rnn_spec.get("sizes", [])):
                            self.memories[uid].create_tensor(
                                name=f"rnn_value_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                            )
                            self._rnn_tensors_names[uid].append(f"rnn_value_{i}")
                            self._rnn_initial_states[uid]["value"].append(
                                torch.zeros(size, dtype=torch.float32, device=self.device)
                            )
        
        # tensors sampled during training
        self._tensors_names = [
            "states",
            "shared_states",
            "actions",
            "terminated",
            "truncated",
            "log_prob",
            "values",
            "returns",
            "advantages",
        ]
        
        # create temporary variables needed for storage and computation
        self._current_log_prob = {}
        self._current_shared_next_states = None
        self._rnn_states = copy.deepcopy(self._rnn_initial_states)
    
    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policies"""
        
        actions = {}
        log_probs = {}
        outputs = {}
        
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            for uid in self.possible_agents:
                # Prepare RNN states if available
                rnn_input = {}
                if self._rnn_states[uid]["policy"]:
                    rnn_input = {"rnn": self._rnn_states[uid]["policy"]}
                
                # Get action from policy
                action, log_prob, output = self.policies[uid].act(
                    {"states": self._state_preprocessor[uid](states[uid]), **rnn_input}, 
                    role="policy"
                )
                
                actions[uid] = action
                log_probs[uid] = log_prob
                outputs[uid] = output
                
                # Update RNN states
                if "rnn" in output:
                    self._rnn_states[uid]["policy"] = output["rnn"]
        
        self._current_log_prob = log_probs
        
        return actions, log_probs, outputs
    
    def record_transition(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory"""
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )
        
        if self.memories:
            # Obtain shared states for centralized training.
            # If the environment doesn't provide them, concatenate all agent observations.
            if "shared_states" in infos and "shared_next_states" in infos:
                shared_states_payload = infos["shared_states"]
                self._current_shared_next_states = infos["shared_next_states"]
            else:
                # Concatenate states and next_states along the last dimension
                sorted_states_list = [states[uid] for uid in self.possible_agents]
                concatenated_states = torch.cat(sorted_states_list, dim=-1)

                sorted_next_states_list = [next_states[uid] for uid in self.possible_agents]
                concatenated_next_states = torch.cat(sorted_next_states_list, dim=-1)

                # The agent expects a dictionary where each agent maps to the shared state
                shared_states_payload = {uid: concatenated_states for uid in self.possible_agents}
                self._current_shared_next_states = {uid: concatenated_next_states for uid in self.possible_agents}
            
            for uid in self.possible_agents:
                # reward shaping
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)
                
                # compute values using shared states for centralized training
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # Prepare RNN states for value function
                    rnn_input = {}
                    if self._rnn_states[uid]["value"]:
                        rnn_input = {"rnn": self._rnn_states[uid]["value"]}
                    
                    values, _, value_outputs = self.values[uid].act(
                        {"states": self._shared_state_preprocessor[uid](shared_states_payload), **rnn_input}, 
                        role="value"
                    )
                    values = self._value_preprocessor[uid](values, inverse=True)
                    
                    # Update value RNN states
                    if "rnn" in value_outputs:
                        self._rnn_states[uid]["value"] = value_outputs["rnn"]
                
                # time-limit (truncation) bootstrapping
                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * values * truncated[uid]
                
                # Package RNN states for storage
                rnn_states_storage = {}
                if self._rnn_initial_states[uid]["policy"]:
                    for i, s in enumerate(self._rnn_initial_states[uid]["policy"]):
                        rnn_states_storage[f"rnn_policy_{i}"] = s.transpose(0, 1)
                if self.policies[uid] is not self.values[uid] and self._rnn_initial_states[uid]["value"]:
                    for i, s in enumerate(self._rnn_initial_states[uid]["value"]):
                        rnn_states_storage[f"rnn_value_{i}"] = s.transpose(0, 1)
                
                # storage transition in memory
                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=values,
                    shared_states=shared_states_payload,
                    **rnn_states_storage,
                )
                
                # Reset RNN states for terminated episodes
                finished = (terminated[uid] | truncated[uid]).nonzero(as_tuple=False)
                if finished.numel():
                    for rnn_state in self._rnn_states[uid]["policy"]:
                        rnn_state[:, finished[:, 0]] = 0
                    if self.policies[uid] is not self.values[uid]:
                        for rnn_state in self._rnn_states[uid]["value"]:
                            rnn_state[:, finished[:, 0]] = 0
                
                # Update initial states for next step
                self._rnn_initial_states[uid] = copy.deepcopy(self._rnn_states[uid])
    
    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment"""
        pass
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment"""
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")
        
        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)
    
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step"""
        
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)"""
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            
            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return returns, advantages
        
        for uid in self.possible_agents:
            policy = self.policies[uid]
            value = self.values[uid]
            memory = self.memories[uid]
            
            # compute returns and advantages
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                # Prepare RNN states for last value computation
                rnn_input = {}
                if self._rnn_states[uid]["value"]:
                    rnn_input = {"rnn": self._rnn_states[uid]["value"]}
                
                # Get shared state for last value computation
                if isinstance(self._current_shared_next_states, dict):
                    last_shared_state = self._current_shared_next_states[uid]
                else:
                    last_shared_state = self._current_shared_next_states
                
                value.train(False)
                last_values, _, _ = value.act(
                    {"states": self._shared_state_preprocessor[uid](last_shared_state.float()), **rnn_input},
                    role="value",
                )
                value.train(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)
            
            values = memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=memory.get_tensor_by_name("rewards"),
                dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                values=values,
                next_values=last_values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )
            
            memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
            memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
            memory.set_tensor_by_name("advantages", advantages)
            
            # sample mini-batches from memory
            sampled_batches = memory.sample_all(
                names=self._tensors_names, 
                mini_batches=self._mini_batches[uid],
                sequence_length=self._rnn_sequence_lengths[uid]
            )
            
            # Sample RNN states if available
            sampled_rnn_batches = None
            if uid in self._rnn_tensors_names and self._rnn_tensors_names[uid]:
                sampled_rnn_batches = memory.sample_all(
                    names=self._rnn_tensors_names[uid],
                    mini_batches=self._mini_batches[uid],
                    sequence_length=self._rnn_sequence_lengths[uid],
                )
            
            cumulative_policy_loss = 0
            cumulative_entropy_loss = 0
            cumulative_value_loss = 0
            
            # learning epochs
            for epoch in range(self._learning_epochs[uid]):
                kl_divergences = []
                
                # mini-batches loop
                for i, (
                    sampled_states,
                    sampled_shared_states,
                    sampled_actions,
                    sampled_terminated,
                    sampled_truncated,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) in enumerate(sampled_batches):
                    
                    # Prepare RNN inputs if available
                    rnn_policy = {}
                    rnn_value = {}
                    if sampled_rnn_batches:
                        if policy is value:
                            rnn_policy = {
                                "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                                "terminated": sampled_terminated | sampled_truncated,
                            }
                            rnn_value = rnn_policy
                        else:
                            rnn_policy = {
                                "rnn": [
                                    s.transpose(0, 1)
                                    for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names[uid])
                                    if "policy" in n
                                ],
                                "terminated": sampled_terminated | sampled_truncated,
                            }
                            rnn_value = {
                                "rnn": [
                                    s.transpose(0, 1)
                                    for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names[uid])
                                    if "value" in n
                                ],
                                "terminated": sampled_terminated | sampled_truncated,
                            }
                    
                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        
                        sampled_states = self._state_preprocessor[uid](sampled_states, train=not epoch)
                        sampled_shared_states = self._shared_state_preprocessor[uid](
                            sampled_shared_states, train=not epoch
                        )
                        
                        _, next_log_prob, _ = policy.act(
                            {"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
                        )
                        
                        # compute approximate KL divergence
                        with torch.no_grad():
                            ratio = next_log_prob - sampled_log_prob
                            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                            kl_divergences.append(kl_divergence)
                        
                        # early stopping with KL divergence
                        if self._kl_threshold[uid] and kl_divergence > self._kl_threshold[uid]:
                            break
                        
                        # compute entropy loss
                        if self._entropy_loss_scale[uid]:
                            entropy_loss = -self._entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                        else:
                            entropy_loss = 0
                        
                        # compute policy loss
                        ratio = torch.exp(next_log_prob - sampled_log_prob)
                        surrogate = sampled_advantages * ratio
                        surrogate_clipped = sampled_advantages * torch.clip(
                            ratio, 1.0 - self._ratio_clip[uid], 1.0 + self._ratio_clip[uid]
                        )
                        
                        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
                        
                        # compute value loss
                        predicted_values, _, _ = value.act({"states": sampled_shared_states, **rnn_value}, role="value")
                        
                        if self._clip_predicted_values:
                            predicted_values = sampled_values + torch.clip(
                                predicted_values - sampled_values, min=-self._value_clip[uid], max=self._value_clip[uid]
                            )
                        value_loss = self._value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)
                    
                    # optimization step
                    self.optimizers[uid].zero_grad()
                    self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()
                    
                    if config.torch.is_distributed:
                        policy.reduce_parameters()
                        if policy is not value:
                            value.reduce_parameters()
                    
                    if self._grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.optimizers[uid])
                        if policy is value:
                            nn.utils.clip_grad_norm_(policy.parameters(), self._grad_norm_clip[uid])
                        else:
                            nn.utils.clip_grad_norm_(
                                itertools.chain(policy.parameters(), value.parameters()), self._grad_norm_clip[uid]
                            )
                    
                    self.scaler.step(self.optimizers[uid])
                    self.scaler.update()
                    
                    # update cumulative losses
                    cumulative_policy_loss += policy_loss.item()
                    cumulative_value_loss += value_loss.item()
                    if self._entropy_loss_scale[uid]:
                        cumulative_entropy_loss += entropy_loss.item()
                
                # update learning rate
                if self._learning_rate_scheduler[uid]:
                    if isinstance(self.schedulers[uid], KLAdaptiveLR):
                        kl = torch.tensor(kl_divergences, device=self.device).mean()
                        # reduce (collect from all workers/processes) KL in distributed runs
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                        self.schedulers[uid].step(kl.item())
                    else:
                        self.schedulers[uid].step()
            
            # record data
            self.track_data(
                f"Loss / Policy loss ({uid})",
                cumulative_policy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            self.track_data(
                f"Loss / Value loss ({uid})",
                cumulative_value_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            if self._entropy_loss_scale:
                self.track_data(
                    f"Loss / Entropy loss ({uid})",
                    cumulative_entropy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
                )
            
            self.track_data(
                f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item()
            )
            
            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])


class MAPPORNNBaseModel(Model):
    """Base class for MAPPO RNN models with shared logic."""
    def __init__(self, observation_space, action_space, device,
                 hidden_size=256, gru_num_layers=2, gru_hidden_size=256, num_envs=1):
        super().__init__(observation_space, action_space, device)
        
        self.num_envs = num_envs
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_size, gru_hidden_size, num_layers=gru_num_layers, batch_first=True)
        
        # Store GRU specs for easy access
        self.gru_num_layers = self.gru.num_layers
        self.gru_hidden_size = self.gru.hidden_size

    def get_specification(self):
        # The batch size in the RNN state should correspond to the number of parallel environments
        return {"rnn": {"sequence_length": 32,
                       "sizes": [(self.gru_num_layers, self.num_envs, self.gru_hidden_size)]}}

    def compute_base(self, inputs):
        """Shared forward pass logic for both policy and value networks."""
        x = self.net(inputs["states"])
        hidden_states = inputs.get("rnn", [None])[0]

        # Check if we are processing a single time-step (e.g., during interaction)
        is_single_step = x.dim() == 2
        if is_single_step:
            # Add a sequence dimension: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)

        # GRU forward pass
        gru_output, hidden_states = self.gru(x, hidden_states.contiguous() if hidden_states is not None else None)

        # If it was a single step, remove the sequence dimension
        if is_single_step:
            # (batch, 1, hidden_size) -> (batch, hidden_size)
            output = gru_output.squeeze(1)
        else:
            # Otherwise, flatten batch and sequence dimensions for the final layer
            # (batch, sequence, hidden_size) -> (batch * sequence, hidden_size)
            output = torch.flatten(gru_output, start_dim=0, end_dim=1)
            
        return output, hidden_states


class MAPPORNNPolicy(GaussianMixin, MAPPORNNBaseModel):
    """Policy network with RNN for MAPPO."""
    def __init__(self, observation_space, action_space, device,
                 hidden_size=256, gru_num_layers=2, gru_hidden_size=256, num_envs=1):
        MAPPORNNBaseModel.__init__(self, observation_space, action_space, device, hidden_size,
                                   gru_num_layers, gru_hidden_size, num_envs)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True)

        self.policy_layer = nn.Linear(self.gru_hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        output, hidden_states = self.compute_base(inputs)
        return self.policy_layer(output), self.log_std_parameter, {"rnn": [hidden_states]}


class MAPPORNNValue(DeterministicMixin, MAPPORNNBaseModel):
    """Value network with RNN for MAPPO (uses shared/global observations)."""
    def __init__(self, observation_space, action_space, device,
                 hidden_size=256, gru_num_layers=2, gru_hidden_size=256, num_envs=1):
        MAPPORNNBaseModel.__init__(self, observation_space, action_space, device, hidden_size,
                                   gru_num_layers, gru_hidden_size, num_envs)
        DeterministicMixin.__init__(self)

        self.value_layer = nn.Linear(self.gru_hidden_size, 1)

    def compute(self, inputs, role):
        output, hidden_states = self.compute_base(inputs)
        return self.value_layer(output), {"rnn": [hidden_states]}