# file: mappo_with_aux.py
"""MAPPO_RNN extended with an auxiliary policy head that predicts target
position + diagonal log-variance from each agent's actor observation.

See iris_ma6/doc/policy_triangulation_head_spec.md and
iris_ma6/doc/active/ticket/031-policy-triangulation-head/ticket.md.

Slice 1 deliverables:
- MAPPO_WITH_AUX_DEFAULT_CONFIG
- MAPPOWithAuxPolicy   (subclass of MAPPORNNPolicy, adds tri_head)
- MAPPOWithAuxValue    (re-export, identical to MAPPORNNValue)
- compute_aux_nll_loss (standalone Gaussian-NLL helper, masked)

Slice 2 will add MAPPOWithAux (the trainer subclass).
Slice 3 will override _update to inject the aux loss.
"""
import copy
import itertools
from typing import Any, Mapping, Optional, Sequence, Union

import gymnasium
import torch
import torch.nn as nn

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

from mappo_rnn import (
    MAPPO_RNN,
    MAPPO_RNN_DEFAULT_CONFIG,
    MAPPORNNBaseModel,
)


# =========================================================================
# Default config
# =========================================================================

MAPPO_WITH_AUX_DEFAULT_CONFIG = copy.deepcopy(MAPPO_RNN_DEFAULT_CONFIG)
MAPPO_WITH_AUX_DEFAULT_CONFIG.update({
    "aux_loss_scale": 0.0,                    # gradient OFF by default; YAML flips to 0.1
    "aux_log_var_clamp": (-10.0, 4.0),        # σ ∈ [≈7e-3, ≈7] meters per axis
    "aux_nll_clamp": 1000.0,                  # per-dim NLL ceiling, init-time stability
    "tri_head_enabled": True,
    "tri_head_hidden": (128,),
})


# =========================================================================
# Models
# =========================================================================

class MAPPOWithAuxPolicy(GaussianMixin, MAPPORNNBaseModel):
    """Policy network with RNN + auxiliary triangulation head.

    Architecture mirrors MAPPORNNPolicy with one addition: a feed-forward
    `tri_head` consuming the post-GRU hidden state. The action head is
    unchanged — it does NOT consume tri_head outputs (no skip connection).

    Auxiliary outputs flow through the standard skrl `extra_dict` channel of
    `compute()`. Do NOT override `act()` — `GaussianMixin.act` wraps `compute`
    and produces `log_prob` correctly; auxiliary outputs are passed through
    without affecting the action distribution.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_size: int = 256,
        gru_num_layers: int = 2,
        gru_hidden_size: int = 256,
        num_envs: int = 1,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 0.7,
        sequence_length: int = 128,
        # Auxiliary head config
        tri_head_enabled: bool = True,
        tri_head_hidden: Sequence[int] = (128,),
        tri_log_var_clamp: Sequence[float] = (-10.0, 4.0),
    ):
        MAPPORNNBaseModel.__init__(
            self, observation_space, action_space, device, hidden_size,
            gru_num_layers, gru_hidden_size, num_envs, sequence_length,
        )
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True,
            min_log_std=min_log_std, max_log_std=max_log_std,
        )

        self.policy_layer = nn.Linear(self.gru_hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

        self.tri_head_enabled = bool(tri_head_enabled)
        self.tri_log_var_min = float(tri_log_var_clamp[0])
        self.tri_log_var_max = float(tri_log_var_clamp[1])

        if self.tri_head_enabled:
            layers = []
            in_dim = self.gru_hidden_size
            for h in tri_head_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ELU())
                in_dim = h
            layers.append(nn.Linear(in_dim, 6))   # 3 mean + 3 log-variance
            self.tri_head = nn.Sequential(*layers)
        else:
            self.tri_head = None

    def compute(self, inputs, role):
        output, hidden_states = self.compute_base(inputs)

        extra = {"rnn": [hidden_states]}
        if self.tri_head is not None:
            tri_out = self.tri_head(output)
            extra["tri_mu"] = tri_out[..., :3]
            extra["tri_log_var"] = tri_out[..., 3:].clamp(
                self.tri_log_var_min, self.tri_log_var_max,
            )

        return self.policy_layer(output), self.log_std_parameter, extra


class MAPPOWithAuxValue(DeterministicMixin, MAPPORNNBaseModel):
    """Value network — identical to MAPPORNNValue. No tri head.

    Re-exported here so callers can import the policy and value classes from
    the same module.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_size: int = 256,
        gru_num_layers: int = 2,
        gru_hidden_size: int = 256,
        num_envs: int = 1,
        sequence_length: int = 128,
    ):
        MAPPORNNBaseModel.__init__(
            self, observation_space, action_space, device, hidden_size,
            gru_num_layers, gru_hidden_size, num_envs, sequence_length,
        )
        DeterministicMixin.__init__(self)
        self.value_layer = nn.Linear(self.gru_hidden_size, 1)

    def compute(self, inputs, role):
        output, hidden_states = self.compute_base(inputs)
        return self.value_layer(output), {"rnn": [hidden_states]}


# =========================================================================
# Auxiliary loss helper (slice 1)
# =========================================================================

def compute_aux_nll_loss(
    tri_mu: torch.Tensor,
    tri_log_var: torch.Tensor,
    tri_target: torch.Tensor,
    tri_valid: torch.Tensor,
    episode_step: torch.Tensor,
    episode_start_mask_steps: int,
    nll_clamp: float = 1000.0,
):
    """Masked Gaussian NLL for the policy tri head.

    Shapes (any leading dims OK; everything is elementwise/broadcasting):
        tri_mu, tri_log_var, tri_target : (..., 3)
        tri_valid                        : (...,) bool or float (will be cast)
        episode_step                     : (...,) int/float (will be cast)

    The mask combines per-sample validity with the same episode-start mask
    used by MAPPO_RNN's policy/value losses, so warm-up steps are excluded
    consistently across all loss terms.

    Returns:
        (aux_loss, diag_dict)
            aux_loss : 0-dim tensor
            diag_dict: predicted_rmse, calibration_2sigma_coverage,
                       effective_sample_fraction (all 0-dim tensors)

    When the combined mask sums to zero, aux_loss == 0 and diag values
    default to 0 (no NaNs).
    """
    valid = tri_valid.to(dtype=torch.bool) if tri_valid.dtype != torch.bool else tri_valid
    # Squeeze a trailing singleton if present (memory stores valid as size-1 last dim)
    if valid.dim() > episode_step.dim():
        valid = valid.squeeze(-1)
    ep = episode_step
    if ep.dim() > 0 and ep.shape[-1] == 1 and ep.dim() > tri_mu.dim() - 1:
        ep = ep.squeeze(-1)

    episode_mask = ep >= int(episode_start_mask_steps)
    combined = valid & episode_mask                      # (...,) bool

    # Per-dim NLL = 0.5 * ((target - mu)^2 / σ^2 + log σ^2)
    inv_var = (-tri_log_var).exp()                       # (..., 3)
    sq_err = (tri_target - tri_mu).pow(2)                # (..., 3)
    per_dim_nll = 0.5 * (sq_err * inv_var + tri_log_var) # (..., 3)
    per_dim_nll = per_dim_nll.clamp(max=float(nll_clamp))

    nll = per_dim_nll.sum(dim=-1)                        # (...,)
    mask_f = combined.to(dtype=nll.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    aux_loss = (nll * mask_f).sum() / denom

    # Diagnostics — guard against denom=1 fallback giving meaningless numbers
    has_any = combined.any()
    if has_any:
        masked_sq = (sq_err.sum(dim=-1) * mask_f).sum()
        rmse = torch.sqrt(masked_sq / (3.0 * denom))
        # ±2σ-per-axis coverage on valid samples
        std = (0.5 * tri_log_var).exp()                  # σ_i
        within = (sq_err <= (4.0 * std.pow(2))).all(dim=-1).to(dtype=mask_f.dtype)
        coverage = (within * mask_f).sum() / denom
    else:
        rmse = torch.zeros((), dtype=nll.dtype, device=nll.device)
        coverage = torch.zeros((), dtype=nll.dtype, device=nll.device)

    sample_fraction = mask_f.mean()

    diag = {
        "predicted_rmse": rmse.detach(),
        "calibration_2sigma_coverage": coverage.detach(),
        "effective_sample_fraction": sample_fraction.detach(),
    }
    return aux_loss, diag


# =========================================================================
# Trainer subclass (slice 2 — init + record_transition only; _update added slice 3)
# =========================================================================

class MAPPOWithAux(MAPPO_RNN):
    """MAPPO_RNN + auxiliary policy-head supervision plumbing.

    Slice 2 deliverable: registers per-agent ``tri_target_position_w`` and
    ``tri_target_valid`` memory tensors and routes per-agent supervision from
    the trainer-injected ``infos`` dict into them. ``_update`` is NOT yet
    overridden — it falls through to ``MAPPO_RNN._update`` and the auxiliary
    tensors are unused for now.

    Slice 3 will override ``_update`` to consume those tensors and add the
    Gaussian-NLL term to the total loss.
    """

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
        # Merge the aux defaults under the user-supplied cfg so YAML overrides win.
        _cfg = copy.deepcopy(MAPPO_WITH_AUX_DEFAULT_CONFIG)
        if cfg is not None:
            _cfg.update(cfg)
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
            shared_observation_spaces=shared_observation_spaces,
        )

        # Per-agent aux config dicts (mirror of MAPPO_RNN's pattern, see L71-113).
        self._aux_loss_scale = self._as_dict(self.cfg["aux_loss_scale"])
        self._aux_nll_clamp = self._as_dict(self.cfg["aux_nll_clamp"])
        # log_var clamp lives on the model; kept here only for diagnostics.
        self._aux_log_var_clamp = self.cfg["aux_log_var_clamp"]

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Register the auxiliary supervision tensors after standard memory init."""
        super().init(trainer_cfg=trainer_cfg)

        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(
                    name="tri_target_position_w", size=3, dtype=torch.float32
                )
                self.memories[uid].create_tensor(
                    name="tri_target_valid", size=1, dtype=torch.float32
                )

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
        """Pre-write per-agent supervision tensors at each memory's current
        write index, then defer to ``MAPPO_RNN.record_transition`` to write the
        standard tensors at the SAME index and advance the index. This keeps
        ``tri_target_*`` aligned with ``states``/``actions``/etc. without
        replicating the parent's per-agent loop.
        """
        # Skip silently if the trainer wrapper didn't inject supervision (e.g.
        # in unit tests or when running an unrelated env). The aux loss in
        # _update will mask everything out via tri_target_valid==0.
        if (
            self.memories
            and "tri_target_position_w" in infos
            and "tri_target_valid" in infos
            and infos["tri_target_position_w"] is not None
            and infos["tri_target_valid"] is not None
        ):
            tri_pos = infos["tri_target_position_w"]      # (N, A, 3)
            tri_valid = infos["tri_target_valid"]         # (N, A) bool or float
            for agent_idx, uid in enumerate(self.possible_agents):
                mem = self.memories[uid]
                if "tri_target_position_w" not in mem.tensors:
                    continue
                mi = mem.memory_index
                mem.tensors["tri_target_position_w"][mi].copy_(tri_pos[:, agent_idx, :])
                valid_slice = tri_valid[:, agent_idx:agent_idx + 1].to(dtype=torch.float32)
                mem.tensors["tri_target_valid"][mi].copy_(valid_slice)

        super().record_transition(
            states, actions, rewards, next_states,
            terminated, truncated, infos, timestep, timesteps,
        )

    def _update(self, timestep: int, timesteps: int) -> None:
        """MAPPO_RNN._update with the auxiliary policy-tri-head NLL term.

        Body is a faithful copy of MAPPO_RNN._update with three additions
        (clearly marked with ``# [AUX]`` comments):
          1. Sample tri_target_position_w / tri_target_valid alongside the
             standard tensors using the same reshape_to_sequences helper.
          2. Capture ``policy_outputs`` from ``policy.act`` (parent throws
             it away with ``_``) so we can read tri_mu / tri_log_var.
          3. Compute aux NLL via compute_aux_nll_loss(...) and add
             ``aux_loss_scale * aux_loss`` to the total loss before backward.

        The bit-exact regression test (test_loss_scale_zero_regression.py)
        verifies that with aux_loss_scale=0.0 the gradients on the shared
        parameters match a baseline run with tri_head_enabled=False. Any drift
        from MAPPO_RNN's loss math (introduced by an accidental refactor here)
        would surface as a failure of that test.
        """

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
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        for uid in self.possible_agents:
            policy = self.policies[uid]
            value = self.values[uid]
            memory = self.memories[uid]

            # compute returns and advantages
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                rnn_input = {}
                if self._rnn_states[uid]["value"]:
                    rnn_input = {"rnn": self._rnn_states[uid]["value"]}
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

            # Custom recurrent minibatch generator (mirrors MAPPO_RNN exactly)
            seq_len = self._rnn_sequence_lengths[uid]
            num_envs = memory.num_envs
            memory_size = memory.memory_size

            raw_states = memory.get_tensor_by_name("states")
            raw_shared_states = memory.get_tensor_by_name("shared_states")
            raw_actions = memory.get_tensor_by_name("actions")
            raw_terminated = memory.get_tensor_by_name("terminated")
            raw_truncated = memory.get_tensor_by_name("truncated")
            raw_log_prob = memory.get_tensor_by_name("log_prob")
            raw_values = memory.get_tensor_by_name("values")
            raw_returns = memory.get_tensor_by_name("returns")
            raw_advantages = memory.get_tensor_by_name("advantages")
            raw_episode_step = memory.get_tensor_by_name("episode_step")
            # [AUX] additional tensors for the policy tri head supervision.
            raw_tri_target_pos = memory.get_tensor_by_name("tri_target_position_w")  # (T, N, 3)
            raw_tri_target_valid = memory.get_tensor_by_name("tri_target_valid")     # (T, N, 1)

            raw_rnn_states = []
            if uid in self._rnn_tensors_names and self._rnn_tensors_names[uid]:
                for rnn_name in self._rnn_tensors_names[uid]:
                    raw_rnn_states.append(memory.get_tensor_by_name(rnn_name))

            states_NT = raw_states.transpose(0, 1).contiguous()
            shared_states_NT = raw_shared_states.transpose(0, 1).contiguous()
            actions_NT = raw_actions.transpose(0, 1).contiguous()
            terminated_NT = raw_terminated.transpose(0, 1).contiguous()
            truncated_NT = raw_truncated.transpose(0, 1).contiguous()
            log_prob_NT = raw_log_prob.transpose(0, 1).contiguous()
            values_NT = raw_values.transpose(0, 1).contiguous()
            returns_NT = raw_returns.transpose(0, 1).contiguous()
            advantages_NT = raw_advantages.transpose(0, 1).contiguous()
            episode_step_NT = raw_episode_step.transpose(0, 1).contiguous()
            # [AUX] same transpose for the tri tensors.
            tri_target_pos_NT = raw_tri_target_pos.transpose(0, 1).contiguous()      # (N, T, 3)
            tri_target_valid_NT = raw_tri_target_valid.transpose(0, 1).contiguous()  # (N, T, 1)

            rnn_states_NT = [r.transpose(0, 1).contiguous() for r in raw_rnn_states]

            num_sequences_per_env = memory_size // seq_len
            effective_T = num_sequences_per_env * seq_len

            if effective_T < memory_size:
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
                # [AUX] truncate the tri tensors identically.
                tri_target_pos_NT = tri_target_pos_NT[:, :effective_T]
                tri_target_valid_NT = tri_target_valid_NT[:, :effective_T]

            total_sequences = num_envs * num_sequences_per_env

            def reshape_to_sequences(tensor_NT, seq_len):
                shape = tensor_NT.shape
                N, T = shape[0], shape[1]
                rest = shape[2:] if len(shape) > 2 else ()
                num_seq = T // seq_len
                reshaped = tensor_NT.view(N, num_seq, seq_len, *rest)
                return reshaped.view(N * num_seq, seq_len, *rest)

            seq_states = reshape_to_sequences(states_NT, seq_len)
            seq_shared_states = reshape_to_sequences(shared_states_NT, seq_len)
            seq_actions = reshape_to_sequences(actions_NT, seq_len)
            seq_terminated = reshape_to_sequences(terminated_NT, seq_len)
            seq_truncated = reshape_to_sequences(truncated_NT, seq_len)
            seq_log_prob = reshape_to_sequences(log_prob_NT, seq_len)
            seq_values = reshape_to_sequences(values_NT, seq_len)
            seq_returns = reshape_to_sequences(returns_NT, seq_len)
            seq_advantages = reshape_to_sequences(advantages_NT, seq_len)
            seq_episode_step = reshape_to_sequences(episode_step_NT, seq_len)
            # [AUX] reshape tri tensors with the same helper for index alignment.
            seq_tri_target_pos = reshape_to_sequences(tri_target_pos_NT, seq_len)        # (B*, S, 3)
            seq_tri_target_valid = reshape_to_sequences(tri_target_valid_NT, seq_len)    # (B*, S, 1)

            seq_rnn_initial = []
            for rnn_NT in rnn_states_NT:
                N, T, layers, hidden = rnn_NT.shape
                num_seq = T // seq_len
                indices = torch.arange(0, effective_T, seq_len, device=rnn_NT.device)
                initial_states = rnn_NT[:, indices, :, :]
                initial_states = initial_states.view(N * num_seq, layers, hidden)
                seq_rnn_initial.append(initial_states)

            mini_batch_size = total_sequences // self._mini_batches[uid]
            indices = torch.randperm(total_sequences, device=self.device)

            sampled_batches = []
            sampled_rnn_batches = []
            for mb in range(self._mini_batches[uid]):
                start = mb * mini_batch_size
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                sampled_batches.append((
                    seq_states[mb_indices],
                    seq_shared_states[mb_indices],
                    seq_actions[mb_indices],
                    seq_terminated[mb_indices],
                    seq_truncated[mb_indices],
                    seq_log_prob[mb_indices],
                    seq_values[mb_indices],
                    seq_returns[mb_indices],
                    seq_advantages[mb_indices],
                    seq_episode_step[mb_indices],
                    # [AUX] tri sampled slices (positions 10, 11 in the tuple)
                    seq_tri_target_pos[mb_indices],
                    seq_tri_target_valid[mb_indices],
                ))

                if seq_rnn_initial:
                    sampled_rnn_batches.append([rnn[mb_indices] for rnn in seq_rnn_initial])

            cumulative_policy_loss = 0
            cumulative_entropy_loss = 0
            cumulative_value_loss = 0
            # [AUX] cumulative loss + diagnostic accumulators
            cumulative_aux_loss = 0.0
            cumulative_aux_rmse = 0.0
            cumulative_aux_coverage = 0.0
            cumulative_aux_sample_frac = 0.0

            for epoch in range(self._learning_epochs[uid]):
                kl_divergences = []

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
                    sampled_episode_step,
                    sampled_tri_target_pos,    # [AUX]
                    sampled_tri_target_valid,  # [AUX]
                ) in enumerate(sampled_batches):

                    rnn_policy = {}
                    rnn_value = {}
                    if sampled_rnn_batches:
                        combined_terminated = sampled_terminated | sampled_truncated
                        if combined_terminated.dim() == 3 and combined_terminated.shape[-1] == 1:
                            combined_terminated = combined_terminated.squeeze(-1)

                        if policy is value:
                            rnn_policy = {
                                "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                                "terminated": combined_terminated,
                            }
                            rnn_value = rnn_policy
                        else:
                            rnn_policy = {
                                "rnn": [
                                    s.transpose(0, 1)
                                    for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names[uid])
                                    if "policy" in n
                                ],
                                "terminated": combined_terminated,
                            }
                            rnn_value = {
                                "rnn": [
                                    s.transpose(0, 1)
                                    for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names[uid])
                                    if "value" in n
                                ],
                                "terminated": combined_terminated,
                            }

                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                        is_sequence = sampled_states.dim() == 3
                        batch_size, seq_length = (
                            (sampled_states.shape[0], sampled_states.shape[1])
                            if is_sequence else (sampled_states.shape[0], 1)
                        )

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

                        # [AUX] capture policy_outputs (parent uses `_`, dropping aux head outputs).
                        _, next_log_prob, policy_outputs = policy.act(
                            {"states": proc_states, "taken_actions": flat_actions, **rnn_policy}, role="policy"
                        )

                        predicted_values, _, _ = value.act(
                            {"states": proc_shared_states, **rnn_value}, role="value"
                        )

                        # [AUX] aux head outputs (None when tri_head_enabled=False on the model)
                        tri_mu = policy_outputs.get("tri_mu") if isinstance(policy_outputs, dict) else None
                        tri_log_var = policy_outputs.get("tri_log_var") if isinstance(policy_outputs, dict) else None

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

                            # [AUX] mirror burn-in slicing for aux supervision + outputs.
                            sampled_tri_target_pos = sampled_tri_target_pos[:, burn_in:]
                            sampled_tri_target_valid = sampled_tri_target_valid[:, burn_in:]
                            if tri_mu is not None:
                                tri_mu = tri_mu.view(batch_size, seq_length, -1)[:, burn_in:]
                                tri_log_var = tri_log_var.view(batch_size, seq_length, -1)[:, burn_in:]

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
                            ratio = next_log_prob - sampled_log_prob
                            kl_terms = (torch.exp(ratio) - 1) - ratio
                            if episode_mask is not None:
                                kl_divergence = masked_mean(kl_terms, episode_mask)
                            else:
                                kl_divergence = kl_terms.mean()
                            kl_divergences.append(kl_divergence)

                        if self._kl_threshold[uid] and kl_divergence > self._kl_threshold[uid]:
                            break

                        if self._entropy_loss_scale[uid]:
                            entropy = policy.get_entropy(role="policy")
                            if entropy.dim() > 1:
                                entropy = entropy.sum(dim=-1)
                            if episode_mask is not None:
                                entropy_loss = -self._entropy_loss_scale[uid] * masked_mean(entropy, episode_mask)
                            else:
                                entropy_loss = -self._entropy_loss_scale[uid] * entropy.mean()
                        else:
                            entropy_loss = 0

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
                                min=-self._value_clip[uid], max=self._value_clip[uid],
                            )

                        mse = (sampled_returns - predicted_values).pow(2)
                        if episode_mask is not None:
                            value_loss = self._value_loss_scale[uid] * masked_mean(mse.flatten(), episode_mask)
                        else:
                            value_loss = self._value_loss_scale[uid] * mse.mean()

                        # [AUX] auxiliary policy-head NLL loss
                        aux_loss = torch.zeros((), device=self.device)
                        aux_diag = None
                        if (
                            tri_mu is not None
                            and self._aux_loss_scale[uid] > 0.0  # short-circuit at zero scale
                        ):
                            # Flatten supervision to match the (B*S, ...) shape of policy outputs.
                            tri_target = sampled_tri_target_pos.flatten(0, 1).to(dtype=tri_mu.dtype)
                            tri_valid = sampled_tri_target_valid.flatten(0, 1)  # (B*S, 1)
                            ep_step = sampled_episode_step.flatten(0, 1)        # (B*S, 1) int32
                            mu_flat = tri_mu.reshape(-1, 3) if tri_mu.dim() > 2 else tri_mu
                            log_var_flat = (
                                tri_log_var.reshape(-1, 3) if tri_log_var.dim() > 2 else tri_log_var
                            )

                            aux_loss, aux_diag = compute_aux_nll_loss(
                                tri_mu=mu_flat,
                                tri_log_var=log_var_flat,
                                tri_target=tri_target,
                                tri_valid=tri_valid,
                                episode_step=ep_step,
                                episode_start_mask_steps=self._episode_start_mask_steps,
                                nll_clamp=self._aux_nll_clamp[uid],
                            )

                    # optimization step — [AUX] aux_loss added to the sum
                    self.optimizers[uid].zero_grad()
                    self.scaler.scale(
                        policy_loss + entropy_loss + value_loss
                        + self._aux_loss_scale[uid] * aux_loss
                    ).backward()

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
                                itertools.chain(policy.parameters(), value.parameters()),
                                self._grad_norm_clip[uid],
                            )

                    self.scaler.step(self.optimizers[uid])
                    self.scaler.update()

                    cumulative_policy_loss += policy_loss.item()
                    cumulative_value_loss += value_loss.item()
                    if self._entropy_loss_scale[uid]:
                        cumulative_entropy_loss += entropy_loss.item()
                    # [AUX] accumulate diagnostics for tensorboard logging
                    if aux_diag is not None:
                        cumulative_aux_loss += aux_loss.item()
                        cumulative_aux_rmse += aux_diag["predicted_rmse"].item()
                        cumulative_aux_coverage += aux_diag["calibration_2sigma_coverage"].item()
                        cumulative_aux_sample_frac += aux_diag["effective_sample_fraction"].item()

                if self._learning_rate_scheduler[uid]:
                    if isinstance(self.schedulers[uid], KLAdaptiveLR):
                        kl = torch.tensor(kl_divergences, device=self.device).mean()
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                        self.schedulers[uid].step(kl.item())
                    else:
                        self.schedulers[uid].step()

            # record data
            denom_runs = max(1, self._learning_epochs[uid] * self._mini_batches[uid])
            self.track_data(
                f"Loss / Policy loss ({uid})", cumulative_policy_loss / denom_runs,
            )
            self.track_data(
                f"Loss / Value loss ({uid})", cumulative_value_loss / denom_runs,
            )
            if self._entropy_loss_scale:
                self.track_data(
                    f"Loss / Entropy loss ({uid})", cumulative_entropy_loss / denom_runs,
                )
            self.track_data(
                f"Policy / Standard deviation ({uid})",
                policy.distribution(role="policy").stddev.mean().item(),
            )
            if self._learning_rate_scheduler[uid]:
                self.track_data(
                    f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0],
                )

            # [AUX] aux-head metrics
            if self._aux_loss_scale[uid] > 0.0:
                self.track_data(f"aux / nll ({uid})", cumulative_aux_loss / denom_runs)
                self.track_data(f"aux / predicted_rmse ({uid})", cumulative_aux_rmse / denom_runs)
                self.track_data(
                    f"aux / calibration_2sigma_coverage ({uid})",
                    cumulative_aux_coverage / denom_runs,
                )
                self.track_data(
                    f"aux / effective_sample_fraction ({uid})",
                    cumulative_aux_sample_frac / denom_runs,
                )


__all__ = [
    "MAPPO_WITH_AUX_DEFAULT_CONFIG",
    "MAPPOWithAuxPolicy",
    "MAPPOWithAuxValue",
    "MAPPOWithAux",
    "compute_aux_nll_loss",
]
