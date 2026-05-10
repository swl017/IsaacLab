# file: mappo_rnn.py
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

from mappo_rnn_groups import (
    OwnershipGroup,
    assert_groups_partition_params,
    build_ownership_groups,
    homogeneous_config_check,
)


# MAPPO_RNN Configuration
MAPPO_RNN_DEFAULT_CONFIG = copy.deepcopy(PPO_DEFAULT_CONFIG)
MAPPO_RNN_DEFAULT_CONFIG.update({
    "shared_state_preprocessor": None,
    "shared_state_preprocessor_kwargs": {},
    "burn_in_steps": 0,  # (0 = disabled, >0 = enabled)
    "episode_start_mask_steps": 0,  # D (0 = disabled) - mask out first D steps of each episode from loss
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
        self._rnn_sequence_lengths = {}
        
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
        
        self._burn_in_steps = self.cfg.get("burn_in_steps", 0)
        if self._burn_in_steps > 0:
            logger.info(f"RNN burn-in enabled: {self._burn_in_steps} steps will be used for hidden state warm-up")

        self._episode_start_mask_steps = self.cfg.get("episode_start_mask_steps", 0)
        if self._episode_start_mask_steps > 0:
            logger.info(f"Episode-start loss masking enabled: first {self._episode_start_mask_steps} steps are masked out")
        
        self._mixed_precision = self.cfg["mixed_precision"]
        
        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
        
        # set up optimizers and schedulers, keyed by parameter-ownership group
        # rather than by agent uid. When two uids share the same Python model
        # object (homogeneous shared-policy/value setup), they fall into the
        # same group and back exactly one Adam optimizer + one KL scheduler;
        # when uids own disjoint parameter sets, the grouping degenerates to
        # one singleton group per uid, matching the stock skrl per-uid layout.
        self._opt_groups: list[OwnershipGroup] = build_ownership_groups(
            self.possible_agents, self.policies, self.values
        )
        assert_groups_partition_params(self._opt_groups)
        self._uid_to_group: dict[str, OwnershipGroup] = {}
        for group in self._opt_groups:
            for uid in group.uids:
                self._uid_to_group[uid] = group

        # Per-uid optimizer/scheduler maps preserved for backward compatibility
        # with the rest of this class and with skrl's checkpoint_modules layout.
        # For uids in the same group, all entries alias the same Python object,
        # so optimizer.step() / scheduler.step() called once per group steps the
        # shared parameters exactly once.
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict = {}
        # Group-level state used by _update():
        self._group_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._group_schedulers: dict = {}

        for group in self._opt_groups:
            if not group.params:
                # Degenerate group whose uids have neither trainable policy nor
                # value parameters. Skip optimizer construction (matches stock
                # skrl behavior of "no optimizer for empty agent").
                continue

            # Shared per-uid hyperparameters must agree across the group, since
            # one optimizer step drives all contributors at once. Singleton
            # groups (heterogeneous case) just pass the lone value through.
            group_lr = homogeneous_config_check(group, self._learning_rate, "learning_rate")
            group_scheduler_cls = homogeneous_config_check(
                group, self._learning_rate_scheduler, "learning_rate_scheduler"
            )
            group_scheduler_kwargs = homogeneous_config_check(
                group, self._learning_rate_scheduler_kwargs, "learning_rate_scheduler_kwargs"
            )

            optimizer = torch.optim.Adam(group.params, lr=group_lr)
            self._group_optimizers[group.name] = optimizer

            scheduler = None
            if group_scheduler_cls is not None:
                scheduler = group_scheduler_cls(optimizer, **group_scheduler_kwargs)
                self._group_schedulers[group.name] = scheduler

            for uid in group.uids:
                self.optimizers[uid] = optimizer
                if scheduler is not None:
                    self.schedulers[uid] = scheduler

        for uid in self.possible_agents:
            if uid in self.optimizers:
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

        # Log the resolved ownership layout once, so a future reader can tell
        # at a glance whether the optimizer is shared or per-uid.
        for group in self._opt_groups:
            n_params = sum(p.numel() for p in group.params)
            logger.info(
                f"MAPPO_RNN ownership group '{group.name}': uids={group.uids}, "
                f"unique tensors={len(group.params)}, total scalar params={n_params}"
            )
    
    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        
        # create tensors in memories
        if self.memories:
            self._rnn_tensors_names = {}
            
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
                self.memories[uid].create_tensor(name="episode_step", size=1, dtype=torch.int32)

                # Initialize RNN states for each agent's policy
                policy_spec = self.policies[uid].get_specification()
                policy_rnn_spec = policy_spec.get("rnn", {})
                if policy_rnn_spec:
                    seq_len = policy_rnn_spec.get("sequence_length", 32)
                    self._rnn_sequence_lengths[uid] = seq_len
                    logger.info(f"Agent {uid}: RNN sequence_length set to {seq_len}")
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
                else:
                    # No RNN, default to sequence length of 1
                    self._rnn_sequence_lengths[uid] = 1
                    self._rnn_tensors_names[uid] = []
                
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

        # per-agent per-env episode step counters (int32)
        self._episode_steps = {}
        for uid in self.possible_agents:
            num_envs = self.memories[uid].num_envs
            self._episode_steps[uid] = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
    
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
                shared_states_raw = infos["shared_states"]
                shared_next_states_raw = infos["shared_next_states"]

                # Handle both tensor and dict formats from environment
                if isinstance(shared_states_raw, torch.Tensor):
                    shared_states_payload = {uid: shared_states_raw for uid in self.possible_agents}
                else:
                    shared_states_payload = shared_states_raw

                if isinstance(shared_next_states_raw, torch.Tensor):
                    self._current_shared_next_states = {uid: shared_next_states_raw for uid in self.possible_agents}
                else:
                    self._current_shared_next_states = shared_next_states_raw
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
                        {"states": self._shared_state_preprocessor[uid](shared_states_payload[uid]), **rnn_input},
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
                # NOTE: next_states removed - tensor was never created in memory
                # PPO doesn't need next_states since it uses GAE with stored values
                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=values,
                    shared_states=shared_states_payload[uid],
                    episode_step=self._episode_steps[uid].view(-1, 1),  # (N, 1) for memory storage
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

                # Update episode step counters: step++ for all, reset finished envs to 0
                self._episode_steps[uid] += 1
                if finished.numel():
                    self._episode_steps[uid][finished[:, 0]] = 0

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
    
    def _per_uid_minibatch_components(
        self,
        uid: str,
        state: Mapping[str, Any],
        mb_i: int,
        epoch: int,
    ) -> Mapping[str, torch.Tensor]:
        """Compute one uid's PPO loss components for a single minibatch.

        Pulled out of `_update` so the per-group loop can call this once per
        contributing uid and sum the losses before backward+step. The returned
        `policy_loss`, `value_loss`, and `entropy_loss` carry autograd
        history; `kl_divergence` is detached and used only for diagnostics
        and KL-adaptive scheduling.
        """
        policy = state["policy"]
        value = state["value"]
        sampled_batches = state["sampled_batches"]
        sampled_rnn_batches = state["sampled_rnn_batches"]

        (
            sampled_states,
            sampled_shared_states,
            sampled_actions,
            sampled_terminated,
            sampled_truncated,
            sampled_log_prob,
            sampled_values,
            sampled_returns,
            sampled_advantages,
            sampled_episode_step,
        ) = sampled_batches[mb_i]

        rnn_policy: dict = {}
        rnn_value: dict = {}
        if sampled_rnn_batches:
            combined_terminated = sampled_terminated | sampled_truncated
            if combined_terminated.dim() == 3 and combined_terminated.shape[-1] == 1:
                combined_terminated = combined_terminated.squeeze(-1)

            if policy is value:
                rnn_policy = {
                    "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[mb_i]],
                    "terminated": combined_terminated,
                }
                rnn_value = rnn_policy
            else:
                rnn_policy = {
                    "rnn": [
                        s.transpose(0, 1)
                        for s, n in zip(sampled_rnn_batches[mb_i], self._rnn_tensors_names[uid])
                        if "policy" in n
                    ],
                    "terminated": combined_terminated,
                }
                rnn_value = {
                    "rnn": [
                        s.transpose(0, 1)
                        for s, n in zip(sampled_rnn_batches[mb_i], self._rnn_tensors_names[uid])
                        if "value" in n
                    ],
                    "terminated": combined_terminated,
                }

        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            is_sequence = sampled_states.dim() == 3
            if is_sequence:
                batch_size, seq_length = sampled_states.shape[0], sampled_states.shape[1]
            else:
                batch_size, seq_length = sampled_states.shape[0], 1
            episode_mask = None
            effective_seq_length = seq_length

            def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                x_flat = x.flatten()
                x_reshaped = x_flat.view(batch_size, effective_seq_length)
                m_f = m.float()
                denom = m_f.sum().clamp_min(1.0)
                return (x_reshaped * m_f).sum() / denom

            if is_sequence:
                flat_actions = sampled_actions.flatten(0, 1)
            else:
                flat_actions = sampled_actions

            proc_states = self._state_preprocessor[uid](sampled_states, train=not epoch)
            proc_shared_states = self._shared_state_preprocessor[uid](sampled_shared_states, train=not epoch)

            _, next_log_prob, _ = policy.act(
                {"states": proc_states, "taken_actions": flat_actions, **rnn_policy}, role="policy"
            )
            predicted_values, _, _ = value.act(
                {"states": proc_shared_states, **rnn_value}, role="value"
            )

            if is_sequence and self._burn_in_steps > 0:
                burn_in = self._burn_in_steps
                burn_in = min(burn_in, seq_length - 1)
                if burn_in < 0:
                    burn_in = 0

                next_log_prob = next_log_prob.view(batch_size, seq_length, -1)
                predicted_values = predicted_values.view(batch_size, seq_length, -1)

                next_log_prob = next_log_prob[:, burn_in:]
                predicted_values = predicted_values[:, burn_in:]
                sampled_log_prob = sampled_log_prob[:, burn_in:]
                sampled_values = sampled_values[:, burn_in:]
                sampled_returns = sampled_returns[:, burn_in:]
                sampled_advantages = sampled_advantages[:, burn_in:]
                sampled_episode_step = sampled_episode_step[:, burn_in:]

                effective_seq_length = seq_length - burn_in

                if self._episode_start_mask_steps > 0:
                    D = self._episode_start_mask_steps
                    episode_mask = (sampled_episode_step.squeeze(-1) >= D)

                next_log_prob = next_log_prob.flatten()
                predicted_values = predicted_values.flatten()
                sampled_log_prob = sampled_log_prob.flatten()
                sampled_values = sampled_values.flatten()
                sampled_returns = sampled_returns.flatten()
                sampled_advantages = sampled_advantages.flatten()
            elif is_sequence:
                next_log_prob = next_log_prob.flatten()
                predicted_values = predicted_values.flatten()
                sampled_log_prob = sampled_log_prob.flatten()
                sampled_values = sampled_values.flatten()
                sampled_returns = sampled_returns.flatten()
                sampled_advantages = sampled_advantages.flatten()

                if self._episode_start_mask_steps > 0:
                    D = self._episode_start_mask_steps
                    episode_mask = (sampled_episode_step.squeeze(-1) >= D)

            with torch.no_grad():
                ratio_kl = next_log_prob - sampled_log_prob
                kl_terms = (torch.exp(ratio_kl) - 1) - ratio_kl
                if episode_mask is not None:
                    kl_divergence = masked_mean(kl_terms, episode_mask)
                else:
                    kl_divergence = kl_terms.mean()

            if self._entropy_loss_scale[uid]:
                entropy = policy.get_entropy(role="policy")
                if entropy.dim() > 1:
                    entropy = entropy.sum(dim=-1)
                if episode_mask is not None:
                    entropy_loss = -self._entropy_loss_scale[uid] * masked_mean(entropy, episode_mask)
                else:
                    entropy_loss = -self._entropy_loss_scale[uid] * entropy.mean()
            else:
                entropy_loss = torch.zeros((), device=self.device)

            ratio = torch.exp(next_log_prob - sampled_log_prob)
            surrogate = sampled_advantages * ratio
            surrogate_clipped = sampled_advantages * torch.clip(
                ratio, 1.0 - self._ratio_clip[uid], 1.0 + self._ratio_clip[uid]
            )
            policy_terms = torch.min(surrogate, surrogate_clipped)
            if episode_mask is not None:
                policy_loss = -masked_mean(policy_terms, episode_mask)
            else:
                policy_loss = -policy_terms.mean()

            if self._clip_predicted_values[uid]:
                predicted_values = sampled_values + torch.clip(
                    predicted_values - sampled_values,
                    min=-self._value_clip[uid],
                    max=self._value_clip[uid],
                )
            mse = (sampled_returns - predicted_values).pow(2)
            if episode_mask is not None:
                value_loss = self._value_loss_scale[uid] * masked_mean(mse.flatten(), episode_mask)
            else:
                value_loss = self._value_loss_scale[uid] * mse.mean()

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "kl_divergence": kl_divergence,
        }

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
        
        # ============================================================
        # PHASE A — per-uid rollout preparation
        # Compute returns/advantages, build sequence mini-batches, stash the
        # results into per_uid_state. This phase does NOT touch optimizers;
        # all parameter updates happen in Phase B, where contributing uids
        # within a single ownership group are aggregated into one step.
        # ============================================================
        per_uid_state: dict = {}
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

            # ============================================================
            # CUSTOM RECURRENT MINIBATCH GENERATOR
            # skrl's sample_all returns flattened data - we need sequences
            # ============================================================
            seq_len = self._rnn_sequence_lengths[uid]
            num_envs = memory.num_envs
            memory_size = memory.memory_size  # This is T (timesteps per rollout)

            # Get raw tensors from memory - shape: (T, N, ...)
            raw_states = memory.get_tensor_by_name("states")  # (T, N, obs_dim)
            raw_shared_states = memory.get_tensor_by_name("shared_states")
            raw_actions = memory.get_tensor_by_name("actions")
            raw_terminated = memory.get_tensor_by_name("terminated")
            raw_truncated = memory.get_tensor_by_name("truncated")
            raw_log_prob = memory.get_tensor_by_name("log_prob")
            raw_values = memory.get_tensor_by_name("values")
            raw_returns = memory.get_tensor_by_name("returns")
            raw_advantages = memory.get_tensor_by_name("advantages")
            raw_episode_step = memory.get_tensor_by_name("episode_step")  # (T, N, 1) int32

            # Get RNN initial states if available
            raw_rnn_states = []
            if uid in self._rnn_tensors_names and self._rnn_tensors_names[uid]:
                for rnn_name in self._rnn_tensors_names[uid]:
                    raw_rnn_states.append(memory.get_tensor_by_name(rnn_name))  # (T, N, layers, hidden)

            # Transpose to (N, T, ...) for easier sequence slicing
            states_NT = raw_states.transpose(0, 1).contiguous()  # (N, T, obs_dim)
            shared_states_NT = raw_shared_states.transpose(0, 1).contiguous()
            actions_NT = raw_actions.transpose(0, 1).contiguous()
            terminated_NT = raw_terminated.transpose(0, 1).contiguous()
            truncated_NT = raw_truncated.transpose(0, 1).contiguous()
            log_prob_NT = raw_log_prob.transpose(0, 1).contiguous()
            values_NT = raw_values.transpose(0, 1).contiguous()
            returns_NT = raw_returns.transpose(0, 1).contiguous()
            advantages_NT = raw_advantages.transpose(0, 1).contiguous()
            episode_step_NT = raw_episode_step.transpose(0, 1).contiguous()  # (N, T, 1)

            rnn_states_NT = [r.transpose(0, 1).contiguous() for r in raw_rnn_states]  # List of (N, T, layers, hidden)

            # Create non-overlapping sequences
            # T must be divisible by seq_len, or we truncate
            num_sequences_per_env = memory_size // seq_len
            effective_T = num_sequences_per_env * seq_len

            if effective_T < memory_size:
                # Truncate to fit exact sequences
                states_NT = states_NT[:, :effective_T]
                shared_states_NT = shared_states_NT[:, :effective_T]
                actions_NT = actions_NT[:, :effective_T]
                terminated_NT = terminated_NT[:, :effective_T]
                truncated_NT = truncated_NT[:, :effective_T]
                log_prob_NT = log_prob_NT[:, :effective_T]
                values_NT = values_NT[:, :effective_T]
                returns_NT = returns_NT[:, :effective_T]
                advantages_NT = advantages_NT[:, :effective_T]
                episode_step_NT = episode_step_NT[:, :effective_T]
                rnn_states_NT = [r[:, :effective_T] for r in rnn_states_NT]

            # Reshape to (N * num_seq, seq_len, ...)
            total_sequences = num_envs * num_sequences_per_env

            def reshape_to_sequences(tensor_NT, seq_len):
                # (N, T, ...) -> (N, num_seq, seq_len, ...) -> (N*num_seq, seq_len, ...)
                shape = tensor_NT.shape
                N, T = shape[0], shape[1]
                rest = shape[2:] if len(shape) > 2 else ()
                num_seq = T // seq_len
                reshaped = tensor_NT.view(N, num_seq, seq_len, *rest)
                return reshaped.view(N * num_seq, seq_len, *rest)

            seq_states = reshape_to_sequences(states_NT, seq_len)  # (total_seq, seq_len, obs_dim)
            seq_shared_states = reshape_to_sequences(shared_states_NT, seq_len)
            seq_actions = reshape_to_sequences(actions_NT, seq_len)
            seq_terminated = reshape_to_sequences(terminated_NT, seq_len)
            seq_truncated = reshape_to_sequences(truncated_NT, seq_len)
            seq_log_prob = reshape_to_sequences(log_prob_NT, seq_len)
            seq_values = reshape_to_sequences(values_NT, seq_len)
            seq_returns = reshape_to_sequences(returns_NT, seq_len)
            seq_advantages = reshape_to_sequences(advantages_NT, seq_len)
            seq_episode_step = reshape_to_sequences(episode_step_NT, seq_len)  # (B, S, 1)

            # For RNN initial states, we need the state at the START of each sequence
            # rnn_states_NT is (N, T, layers, hidden)
            # We need states at t=0, seq_len, 2*seq_len, ... for each env
            seq_rnn_initial = []
            for rnn_NT in rnn_states_NT:
                # (N, T, layers, hidden) -> select every seq_len timesteps
                N, T, layers, hidden = rnn_NT.shape
                num_seq = T // seq_len
                # Indices: 0, seq_len, 2*seq_len, ...
                indices = torch.arange(0, effective_T, seq_len, device=rnn_NT.device)
                initial_states = rnn_NT[:, indices, :, :]  # (N, num_seq, layers, hidden)
                initial_states = initial_states.view(N * num_seq, layers, hidden)  # (total_seq, layers, hidden)
                seq_rnn_initial.append(initial_states)

            # Create mini-batches by shuffling sequence indices
            mini_batch_size = total_sequences // self._mini_batches[uid]
            indices = torch.randperm(total_sequences, device=self.device)

            # Build sampled_batches list (similar format to original but with sequences)
            sampled_batches = []
            sampled_rnn_batches = []
            for mb in range(self._mini_batches[uid]):
                start = mb * mini_batch_size
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                sampled_batches.append((
                    seq_states[mb_indices],           # (B, S, obs_dim)
                    seq_shared_states[mb_indices],
                    seq_actions[mb_indices],
                    seq_terminated[mb_indices],
                    seq_truncated[mb_indices],
                    seq_log_prob[mb_indices],
                    seq_values[mb_indices],
                    seq_returns[mb_indices],
                    seq_advantages[mb_indices],
                    seq_episode_step[mb_indices],     # (B, S, 1) int32
                ))

                if seq_rnn_initial:
                    sampled_rnn_batches.append([rnn[mb_indices] for rnn in seq_rnn_initial])

            # ============================================================
            # SANITY CHECKS - Verify RNN training is working correctly
            # ============================================================
            if sampled_batches and self._rollout <= self._rollouts:  # Only on first update
                first_batch = sampled_batches[0]
                sampled_states_check = first_batch[0]  # states
                sampled_terminated_check = first_batch[3]  # terminated
                sampled_truncated_check = first_batch[4]  # truncated

                logger.info(f"\n{'='*60}")
                logger.info(f"MAPPO_RNN SANITY CHECK (Agent: {uid})")
                logger.info(f"{'='*60}")
                logger.info(f"Memory: T={memory_size}, N={num_envs}")
                logger.info(f"Sequence length: {seq_len}")
                logger.info(f"Sequences per env: {num_sequences_per_env}")
                logger.info(f"Total sequences: {total_sequences}")
                logger.info(f"Mini-batch size: {mini_batch_size}")
                logger.info(f"Number of mini-batches: {len(sampled_batches)}")

                # Check 1: sampled_states shape
                logger.info(f"\n[CHECK 1] sampled_states.shape: {sampled_states_check.shape}")
                if sampled_states_check.dim() == 2:
                    logger.warning("  WARNING: sampled_states is 2D (batch, obs_dim)")
                    logger.warning("      This means you are NOT doing BPTT!")
                    logger.warning("      Expected 3D: (batch, sequence, obs_dim)")
                elif sampled_states_check.dim() == 3:
                    logger.info(f"  Good: sampled_states is 3D (batch={sampled_states_check.shape[0]}, "
                               f"seq={sampled_states_check.shape[1]}, obs={sampled_states_check.shape[2]})")

                # Check 2: RNN hidden states shape
                if sampled_rnn_batches:
                    rnn_batch_0 = sampled_rnn_batches[0]
                    logger.info(f"\n[CHECK 2] sampled_rnn_batches[0] (initial hidden states):")
                    for idx, rnn_tensor in enumerate(rnn_batch_0):
                        logger.info(f"  rnn_tensor[{idx}].shape: {rnn_tensor.shape}")
                        # New format: (batch, layers, hidden) - 3D is correct
                        if rnn_tensor.dim() == 3:
                            logger.info(f"    Good: (batch={rnn_tensor.shape[0]}, layers={rnn_tensor.shape[1]}, hidden={rnn_tensor.shape[2]})")
                else:
                    logger.warning("\n[CHECK 2] No RNN batches sampled!")

                # Check 3: terminated/truncated shape
                logger.info(f"\n[CHECK 3] terminated/truncated shapes:")
                logger.info(f"  sampled_terminated.shape: {sampled_terminated_check.shape}")
                logger.info(f"  sampled_truncated.shape: {sampled_truncated_check.shape}")
                combined_dones = sampled_terminated_check | sampled_truncated_check
                logger.info(f"  combined dones shape: {combined_dones.shape}")
                if combined_dones.dim() == 3 and combined_dones.shape[-1] == 1:
                    logger.info("  Note: dones has trailing dim 1, will be squeezed")

                # Check 4: Episode-start masking configuration
                if self._episode_start_mask_steps > 0:
                    sampled_episode_step_check = first_batch[9]  # episode_step
                    logger.info(f"\n[CHECK 4] Episode-start masking:")
                    logger.info(f"  D (mask steps): {self._episode_start_mask_steps}")
                    logger.info(f"  sampled_episode_step.shape: {sampled_episode_step_check.shape}")
                    logger.info(f"  episode_step range: [{sampled_episode_step_check.min().item()}, {sampled_episode_step_check.max().item()}]")
                    # Preview mask coverage
                    preview_mask = (sampled_episode_step_check.squeeze(-1) >= self._episode_start_mask_steps)
                    mask_coverage = preview_mask.float().mean().item()
                    logger.info(f"  Mask coverage (True ratio): {mask_coverage:.3f}")
                    if mask_coverage < 0.5:
                        logger.warning(f"  WARNING: Low mask coverage ({mask_coverage:.1%}). "
                                      f"Consider reducing D or increasing sequence length.")

                # Summary
                is_seq = sampled_states_check.dim() == 3
                logger.info(f"\n{'='*60}")
                logger.info(f"VERDICT: {'SEQUENCE TRAINING ACTIVE - BPTT enabled!' if is_seq else 'NOT doing BPTT'}")
                logger.info(f"{'='*60}\n")

            # ----------------------------------------------------------------
            # Stash this uid's prepared rollout state for Phase B.
            # Sequence/RNN batching has produced sampled_batches and
            # sampled_rnn_batches above; everything Phase B needs to compute
            # one minibatch's losses for this uid lives in this dict.
            # ----------------------------------------------------------------
            per_uid_state[uid] = {
                "policy": policy,
                "value": value,
                "sampled_batches": sampled_batches,
                "sampled_rnn_batches": sampled_rnn_batches,
                "cumulative_policy_loss": 0.0,
                "cumulative_value_loss": 0.0,
                "cumulative_entropy_loss": 0.0,
                "n_loss_samples": 0,
                "kl_history": [],
            }

        # ============================================================
        # PHASE B — per-ownership-group optimization
        # For every group of uids that share parameter tensors we sum the
        # per-uid losses at each minibatch and call backward + optimizer.step
        # exactly once. For singleton (heterogeneous) groups this collapses
        # to the original per-uid behavior; for shared-model groups (the
        # iris_ma6 case) it removes the double-stepping that previously made
        # two independent Adam optimizers and two KLAdaptiveLR controllers
        # fight over the same weights.
        # ============================================================
        for group in self._opt_groups:
            if not group.params:
                continue
            optimizer = self._group_optimizers[group.name]
            scheduler = self._group_schedulers.get(group.name)

            # Per-uid hyperparameters that drive the optimization loop must
            # agree across the group (asserted, since one optimizer step
            # serves all contributors). Singleton groups pass through.
            n_epochs = homogeneous_config_check(group, self._learning_epochs, "learning_epochs")
            n_minibatches = homogeneous_config_check(group, self._mini_batches, "mini_batches")
            kl_threshold = homogeneous_config_check(group, self._kl_threshold, "kl_threshold")
            grad_norm_clip = homogeneous_config_check(group, self._grad_norm_clip, "grad_norm_clip")

            for epoch in range(n_epochs):
                aggregate_kl_per_minibatch = []

                for mb_i in range(n_minibatches):
                    optimizer.zero_grad()

                    # Compute each contributing uid's loss components on the
                    # current (un-stepped) weights, accumulate them into a
                    # combined loss, then backward+step once.
                    components_per_uid: dict = {}
                    for uid in group.uids:
                        components_per_uid[uid] = self._per_uid_minibatch_components(
                            uid, per_uid_state[uid], mb_i, epoch
                        )

                    # Aggregate KL across uids at this minibatch. KL is
                    # detached so this does not enter the autograd graph.
                    kl_terms = [c["kl_divergence"] for c in components_per_uid.values()]
                    aggregate_kl = torch.stack(kl_terms).mean()
                    aggregate_kl_per_minibatch.append(aggregate_kl)

                    # KL early stopping: applied to the group-level signal so
                    # the decision is shared across all contributing uids.
                    if kl_threshold and aggregate_kl.item() > kl_threshold:
                        break

                    # Aggregate per-uid losses by MEAN (not SUM) so that the
                    # effective magnitude of `policy_loss`, `value_loss`, and
                    # `entropy_loss` — and hence the effective values of
                    # `value_loss_scale`, `entropy_loss_scale`, and the per-MB
                    # KL that KLAdaptiveLR observes — are independent of how
                    # many uids share the group. This matches the
                    # canonical MAPPO formulation L = (1/N) Σ_i L_PPO_i.
                    total_loss = sum(
                        c["policy_loss"] + c["value_loss"] + c["entropy_loss"]
                        for c in components_per_uid.values()
                    ) / len(group.uids)

                    self.scaler.scale(total_loss).backward()

                    if config.torch.is_distributed:
                        for uid in group.uids:
                            policy_u = self.policies[uid]
                            value_u = self.values[uid]
                            if policy_u is not None:
                                policy_u.reduce_parameters()
                            if value_u is not None and value_u is not policy_u:
                                value_u.reduce_parameters()

                    if grad_norm_clip > 0:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(group.params, grad_norm_clip)

                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # Per-uid diagnostics: cumulative losses for the
                    # interpretable per-agent loss curves.
                    for uid, c in components_per_uid.items():
                        per_uid_state[uid]["cumulative_policy_loss"] += c["policy_loss"].item()
                        per_uid_state[uid]["cumulative_value_loss"] += c["value_loss"].item()
                        if self._entropy_loss_scale[uid]:
                            per_uid_state[uid]["cumulative_entropy_loss"] += c["entropy_loss"].item()
                        per_uid_state[uid]["n_loss_samples"] += 1
                        per_uid_state[uid]["kl_history"].append(c["kl_divergence"].item())

                # Scheduler step at the end of each epoch using the aggregate
                # KL across the group's contributors and minibatches.
                if scheduler is not None:
                    if isinstance(scheduler, KLAdaptiveLR):
                        kl = torch.stack(aggregate_kl_per_minibatch).mean()
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                        scheduler.step(kl.item())
                    else:
                        scheduler.step()

            # Per-group diagnostics: one LR per ownership group, since per-uid
            # LR is now meaningless for shared groups (they all share one
            # optimizer/scheduler).
            self.track_data(
                f"Learning / Learning rate (group: {group.name})",
                optimizer.param_groups[0]["lr"],
            )
            if group.uids:
                # Group-level KL diagnostic, averaged across the most recent
                # epoch's minibatches.
                if aggregate_kl_per_minibatch:
                    self.track_data(
                        f"Learning / KL (group: {group.name})",
                        torch.stack(aggregate_kl_per_minibatch).mean().item(),
                    )

        # ============================================================
        # PHASE C — per-uid diagnostics
        # Per-agent loss/std curves remain available so each agent's local
        # contribution is still interpretable, even though they now share an
        # optimizer for shared groups.
        # ============================================================
        for uid in self.possible_agents:
            state = per_uid_state[uid]
            n = max(state["n_loss_samples"], 1)
            self.track_data(f"Loss / Policy loss ({uid})", state["cumulative_policy_loss"] / n)
            self.track_data(f"Loss / Value loss ({uid})", state["cumulative_value_loss"] / n)
            if self._entropy_loss_scale[uid]:
                self.track_data(
                    f"Loss / Entropy loss ({uid})", state["cumulative_entropy_loss"] / n
                )

            policy_u = state["policy"]
            if policy_u is not None:
                self.track_data(
                    f"Policy / Standard deviation ({uid})",
                    policy_u.distribution(role="policy").stddev.mean().item(),
                )


class MAPPORNNBaseModel(Model):
    """Base class for MAPPO RNN models with shared logic."""
    def __init__(self, observation_space, action_space, device,
                 hidden_size=256, gru_num_layers=2, gru_hidden_size=256, num_envs=1, sequence_length=128):
        super().__init__(observation_space, action_space, device)
        
        self.num_envs = num_envs
        self.sequence_length = sequence_length
        
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
        return {"rnn": {"sequence_length": self.sequence_length,
                       "sizes": [(self.gru_num_layers, self.num_envs, self.gru_hidden_size)]}}

    def compute_base(self, inputs):
        """Shared forward pass logic with proper episode masking.

        This method processes sequences step-by-step, zeroing the hidden state
        whenever an episode terminates within the sequence. This prevents
        gradient contamination across episode boundaries.
        """
        x = self.net(inputs["states"])
        h = inputs.get("rnn", [None])[0]  # (layers, batch, hidden)
        dones = inputs.get("terminated", None)  # (batch, seq) or None

        # Fix dones shape: ensure it's (B, S) not (B, S, 1)
        if dones is not None and dones.dim() == 3 and dones.shape[-1] == 1:
            dones = dones.squeeze(-1)

        # Check if we are processing a single time-step (e.g., during interaction)
        is_single_step = x.dim() == 2
        if is_single_step:
            # Add a sequence dimension: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
            if dones is not None:
                dones = dones.unsqueeze(1)  # (batch, 1)

        B, S, F = x.shape
        
        # Initialize hidden state if None
        if h is None:
            h = torch.zeros(self.gru_num_layers, B, self.gru_hidden_size, 
                           device=x.device, dtype=x.dtype)
        
        need_reset = dones is not None and dones.any()
        if need_reset:
            h = h.clone()  # Clone ONCE for the entire sequence
        
        # Process sequence step-by-step to handle episode boundaries
        outputs = []
        for t in range(S):
            # Zero hidden state for terminated episodes at this timestep
            if need_reset:
                done_mask = dones[:, t]  # (B,) boolean tensor
                if done_mask.any():
                    # Convert boolean mask to indices
                    done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)  # (num_done,)
                    # Zero out hidden state for finished environments
                    # Clone to avoid in-place modification issues with autograd
                    h[:, done_indices, :] = 0.0
            # Ensure h is contiguous after indexing
            h = h.contiguous()
            
            # Forward pass for this timestep
            out_t, h = self.gru(x[:, t:t+1, :], h)  # (B, 1, H), (L, B, H)
            outputs.append(out_t)
        
        gru_output = torch.cat(outputs, dim=1)  # (B, S, H)
        
        # Format output based on input shape
        if is_single_step:
            # (batch, 1, hidden_size) -> (batch, hidden_size)
            output = gru_output.squeeze(1)
        else:
            # Otherwise, flatten batch and sequence dimensions for the final layer
            # (batch, sequence, hidden_size) -> (batch * sequence, hidden_size)
            output = gru_output.flatten(0, 1)
            
        return output, h


class MAPPORNNPolicy(GaussianMixin, MAPPORNNBaseModel):
    """Policy network with RNN for MAPPO."""
    def __init__(self, observation_space, action_space, device,
                 hidden_size=256, gru_num_layers=2, gru_hidden_size=256, num_envs=1,
                 initial_log_std=-0.5, min_log_std=-5.0, max_log_std=0.7, sequence_length=128):
        """
        Args:
            observation_space: Observation space
            action_space: Action space
            device: Device to use
            hidden_size: Hidden layer size for preprocessing network
            gru_num_layers: Number of GRU layers
            gru_hidden_size: Hidden size of GRU
            num_envs: Number of parallel environments
            initial_log_std: Initial log standard deviation (default: -0.5, σ≈0.6)
            min_log_std: Minimum log standard deviation (default: -5.0, σ≈0.007)
            max_log_std: Maximum log standard deviation (default: 0.7, σ≈2.0)
            sequence_length: RNN sequence length for training (default: 128)
        """
        MAPPORNNBaseModel.__init__(self, observation_space, action_space, device, hidden_size,
                                   gru_num_layers, gru_hidden_size, num_envs, sequence_length)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        self.policy_layer = nn.Linear(self.gru_hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        output, hidden_states = self.compute_base(inputs)
        return self.policy_layer(output), self.log_std_parameter, {"rnn": [hidden_states]}


class MAPPORNNValue(DeterministicMixin, MAPPORNNBaseModel):
    """Value network with RNN for MAPPO (uses shared/global observations)."""
    def __init__(self, observation_space, action_space, device,
                 hidden_size=256, gru_num_layers=2, gru_hidden_size=256, num_envs=1, sequence_length=128):
        MAPPORNNBaseModel.__init__(self, observation_space, action_space, device, hidden_size,
                                   gru_num_layers, gru_hidden_size, num_envs, sequence_length)
        DeterministicMixin.__init__(self)

        self.value_layer = nn.Linear(self.gru_hidden_size, 1)

    def compute(self, inputs, role):
        output, hidden_states = self.compute_base(inputs)
        return self.value_layer(output), {"rnn": [hidden_states]}