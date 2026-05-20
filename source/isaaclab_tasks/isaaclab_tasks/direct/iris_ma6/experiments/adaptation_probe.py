#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy-adaptation probes for iris_ma6 MAPPO-RNN (ticket 036).

Three read-only diagnostic experiments against a trained checkpoint:

  - Probe 1: action energy vs episode step (adaptive policies "probe" early then
             commit; robust policies are stationary).
  - Probe 2: linear / MLP decoding of 8 per-(env, agent) jitter latents from
             pre-GRU MLP features and post-GRU hidden states across steps T ∈
             {0, 4, 8, 16, 32, 64, 128, 256, 499}.
  - Probe 3: GRU hidden-state swap counterfactual at T=100; measures
             post-swap action divergence. Includes identity-swap control.

All three use a single checkpoint (default: agent_400000.pt from ticket-034).
No env mutations beyond normal env.step(); no checkpoint mutations.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# CLI parse must precede AppLauncher.
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
from experiment_registry import get_experiment  # noqa: E402

parser = argparse.ArgumentParser(description="iris_ma6 policy adaptation probe (ticket 036).")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint (.pt)")
parser.add_argument("--probe", type=str, default="all",
                    choices=["1", "2", "3", "all"],
                    help="Which probe(s) to run")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA6-Direct-Test-v0")
parser.add_argument("--experiment", type=str, default="a1_with_aoi",
                    help="Experiment name from registry (env-config base)")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="Number of parallel envs (1024 used for probes 2/3; "
                         "probe 1 will internally use min(num_envs, 512))")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory; defaults to "
                         "./experiments/outputs/<YYYY-MM-DD>/probe036")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--enable_cameras", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--step", type=int, default=199000,
                    help="Training step to pin curriculum to via debug_initial_step")
parser.add_argument("--swap-step", type=int, default=100,
                    help="Step at which Probe 3 swaps hidden states")
parser.add_argument("--probe2-steps", type=int, nargs="+",
                    default=[0, 4, 8, 16, 32, 64, 128, 256, 499],
                    help="Steps at which to snapshot features for Probe 2")
args_cli, hydra_args = parser.parse_known_args()

_selected_experiment = get_experiment(args_cli.experiment)
args_cli.task = args_cli.task or _selected_experiment.task

sys.argv = [sys.argv[0]] + hydra_args

# ---------------------------------------------------------------------------
# AppLauncher initialization.
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher  # noqa: E402

app_args = argparse.Namespace(
    headless=args_cli.headless,
    device="cuda:0",
    experience="",
    enable_cameras=args_cli.enable_cameras,
)
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-AppLauncher imports.
# ---------------------------------------------------------------------------

import copy  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from isaaclab_tasks.direct.iris_ma6.experiments import (  # noqa: E402
    apply_agent_overrides,
    apply_env_overrides,
)

# Re-use the policy-load helper from evaluate.py — no need to duplicate.
_skrl_scripts_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
                 "scripts", "reinforcement_learning", "skrl")
)
sys.path.insert(0, _skrl_scripts_dir)

# evaluate.py parses argv on import — avoid that. Instead inline a slim copy
# of its _load_rnn_policy helper. Keeps both files independently runnable.
from mappo_rnn import (  # noqa: E402
    MAPPO_RNN,
    MAPPO_RNN_DEFAULT_CONFIG,
    MAPPORNNPolicy,
    MAPPORNNValue,
)
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: E402
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: E402

# ---------------------------------------------------------------------------
# Optional plotting — fail soft if matplotlib missing in some envs.
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    _MPL_OK = True
except Exception as _e:  # pragma: no cover
    print(f"[PROBE] matplotlib unavailable ({_e}); PNG outputs will be skipped.")
    _MPL_OK = False


# ---------------------------------------------------------------------------
# Names of the 8 _eff_progress_* latent tensors stored on the env. Each is
# shape [N, A]; they are per-(env, agent) draws frozen for one episode.
# ---------------------------------------------------------------------------
EFF_PROGRESS_KEYS = [
    "_eff_progress_agent_velocity",
    "_eff_progress_dynamics",
    "_eff_progress_gimbal_dead_time",
    "_eff_progress_zoom_dead_time",
    "_eff_progress_noise",
    "_eff_progress_dropout",
    "_eff_progress_delay",
    "_eff_progress_burst_dropout",
]


# ---------------------------------------------------------------------------
# Policy-load helper (slim copy of evaluate.py:_load_rnn_policy).
# ---------------------------------------------------------------------------

def _load_rnn_policy(checkpoint_path: str, env, agent_cfg: dict,
                    possible_agents: List[str]):
    device = env.device
    model_cfg = agent_cfg.get("models", {})
    policy_cfg = model_cfg.get("policy", {})
    value_cfg = model_cfg.get("value", {})
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 32)

    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,),
                                      dtype=np.float32)
        shared_observation_spaces = {a: shared_space for a in possible_agents}

    shared_policy = MAPPORNNPolicy(
        observation_space=env.observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=policy_cfg.get("hidden_size", 64),
        gru_num_layers=policy_cfg.get("gru_num_layers", 1),
        gru_hidden_size=policy_cfg.get("gru_hidden_size", 64),
        num_envs=env.num_envs,
        initial_log_std=policy_cfg.get("initial_log_std", -0.5),
        min_log_std=policy_cfg.get("min_log_std", -5.0),
        max_log_std=policy_cfg.get("max_log_std", 0.7),
        sequence_length=sequence_length,
    )
    shared_value = MAPPORNNValue(
        observation_space=shared_observation_spaces[possible_agents[0]],
        action_space=env.action_spaces[possible_agents[0]],
        device=device,
        hidden_size=value_cfg.get("hidden_size", 64),
        gru_num_layers=value_cfg.get("gru_num_layers", 1),
        gru_hidden_size=value_cfg.get("gru_hidden_size", 64),
        num_envs=env.num_envs,
        sequence_length=sequence_length,
    )
    models = {a: {"policy": shared_policy, "value": shared_value}
              for a in possible_agents}

    memories = {a: RandomMemory(memory_size=agent_cfg.get("agent", {}).get("rollouts", 32),
                                num_envs=env.num_envs, device=device)
                for a in possible_agents}

    agent_cfg_copy = copy.deepcopy(agent_cfg)
    a_cfg = agent_cfg_copy.get("agent", {})
    if a_cfg.get("state_preprocessor") == "RunningStandardScaler":
        a_cfg["state_preprocessor"] = RunningStandardScaler
        if a_cfg.get("state_preprocessor_kwargs") is None:
            a_cfg["state_preprocessor_kwargs"] = {}
        a_cfg["state_preprocessor_kwargs"]["size"] = env.observation_spaces[possible_agents[0]]
        a_cfg["state_preprocessor_kwargs"]["device"] = device
    if a_cfg.get("shared_state_preprocessor") == "RunningStandardScaler":
        a_cfg["shared_state_preprocessor"] = RunningStandardScaler
        if a_cfg.get("shared_state_preprocessor_kwargs") is None:
            a_cfg["shared_state_preprocessor_kwargs"] = {}
        a_cfg["shared_state_preprocessor_kwargs"]["size"] = shared_observation_spaces[possible_agents[0]]
        a_cfg["shared_state_preprocessor_kwargs"]["device"] = device
    if a_cfg.get("value_preprocessor") == "RunningStandardScaler":
        a_cfg["value_preprocessor"] = RunningStandardScaler
        if a_cfg.get("value_preprocessor_kwargs") is None:
            a_cfg["value_preprocessor_kwargs"] = {}
        a_cfg["value_preprocessor_kwargs"]["device"] = device
    if a_cfg.get("learning_rate_scheduler") == "KLAdaptiveLR":
        a_cfg["learning_rate_scheduler"] = KLAdaptiveLR

    mappo_cfg = copy.deepcopy(MAPPO_RNN_DEFAULT_CONFIG)
    mappo_cfg.update(a_cfg)

    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )
    print(f"[PROBE] agent constructed; loading checkpoint…", flush=True)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[PROBE] ckpt keys = {list(ckpt.keys())}", flush=True)
    if "drone_0" in ckpt:
        agent.load(checkpoint_path)
        print(f"[PROBE] Loaded checkpoint: {checkpoint_path}")
    elif "policy_state_dict" in ckpt:
        shared_policy.load_state_dict(ckpt["policy_state_dict"])
        shared_value.load_state_dict(ckpt["value_state_dict"])
        companion = os.path.join(os.path.dirname(checkpoint_path), "checkpoints", "best_agent.pt")
        if not os.path.exists(companion):
            companion = os.path.join(os.path.dirname(checkpoint_path), "best_agent.pt")
        if os.path.exists(companion):
            agent.load(companion)
            shared_policy.load_state_dict(ckpt["policy_state_dict"])
            shared_value.load_state_dict(ckpt["value_state_dict"])
            print(f"[PROBE] Loaded final.pt + preprocessor from {companion}")
        else:
            print(f"[PROBE] WARNING: no preprocessor companion found for {checkpoint_path}")
    else:
        raise ValueError(f"Unknown checkpoint format: keys={list(ckpt.keys())}")

    print(f"[PROBE] checkpoint load complete; setting eval mode…", flush=True)
    agent.set_mode("eval")
    print(f"[PROBE] eval mode set; freezing preprocessors…", flush=True)
    for uid in possible_agents:
        for key in ("_state_preprocessor", "_shared_state_preprocessor", "_value_preprocessor"):
            pp = getattr(agent, key, None)
            if isinstance(pp, dict) and uid in pp and hasattr(pp[uid], "eval"):
                pp[uid].eval()
    print(f"[PROBE] preprocessors frozen; returning agent", flush=True)
    return agent


# ---------------------------------------------------------------------------
# Probe utilities.
# ---------------------------------------------------------------------------

def _read_eff_progress(unwrapped) -> Dict[str, torch.Tensor]:
    """Return the 8 per-(env, agent) latent tensors, each shape [N, A]."""
    return {k.lstrip("_"): getattr(unwrapped, k).clone() for k in EFF_PROGRESS_KEYS}


def _stack_latents(latents_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str]]:
    """Stack the 8 latents into a single tensor of shape [N, A, 8]."""
    names = [k.lstrip("_") for k in EFF_PROGRESS_KEYS]
    return torch.stack([latents_dict[n] for n in names], dim=-1), names


class _PreGRUCapture:
    """Forward hook on `policy.net` capturing the pre-GRU MLP output.

    The MAPPORNN policy is structured as: net (MLP) → gru → policy_layer.
    A forward hook on `net` records the tensor entering the GRU.

    Captures form a list — one entry per .net forward pass. In MAPPO_RNN.act()
    each agent's policy.act() is invoked in order over possible_agents, so a
    shared policy fires the hook A times in that order. The caller clears
    `captures` before agent.act() and reads them after.
    """

    def __init__(self, policy_modules: List):
        self.captures: List[torch.Tensor] = []
        self._handles = []
        seen = set()
        for m in policy_modules:
            if id(m) in seen:
                continue
            seen.add(id(m))
            self._handles.append(m.net.register_forward_hook(self._hook))

    def _hook(self, _module, _inputs, output):
        self.captures.append(output.detach().clone())

    def reset(self):
        self.captures = []

    def detach(self):
        for h in self._handles:
            h.remove()


# ---------------------------------------------------------------------------
# Probe 1: action energy vs episode step.
# ---------------------------------------------------------------------------

def run_probe1(agent, env_wrapped, unwrapped, possible_agents: List[str],
               num_envs: int, output_dir: str) -> dict:
    print("\n" + "=" * 80, flush=True)
    print("PROBE 1 — Action energy vs episode step", flush=True)
    print("=" * 80, flush=True)
    device = unwrapped.device
    A = len(possible_agents)
    T_max = int(unwrapped.max_episode_length)

    print(f"  num_envs={num_envs}, num_agents={A}, max_episode_length={T_max}", flush=True)
    print(f"  resetting env…", flush=True)

    # Two-pass accumulators indexed by step-in-episode.
    sum_action_sum = torch.zeros(T_max, device=device)
    sum_action_delta = torch.zeros(T_max, device=device)
    sumsq_action_sum = torch.zeros(T_max, device=device)
    sumsq_action_delta = torch.zeros(T_max, device=device)
    count = torch.zeros(T_max, device=device, dtype=torch.long)

    prev_actions: Dict[str, torch.Tensor] | None = None
    obs, _ = env_wrapped.reset()

    # Run for ~2 episodes worth to cover every env crossing every step.
    n_steps = 2 * T_max
    for step in range(n_steps):
        with torch.no_grad():
            actions, _, outputs = agent.act(obs, 0, 0)
            for uid in possible_agents:
                actions[uid] = outputs[uid]["mean_actions"]

        # Aggregate |a|_1 and |Δa|_1 across agents → per-env scalar.
        a_stack = torch.stack([actions[uid] for uid in possible_agents], dim=1)  # [N, A, act_dim]
        action_sum_per_env = a_stack.abs().sum(dim=(1, 2)) / A  # [N]
        if prev_actions is None:
            action_delta_per_env = torch.zeros_like(action_sum_per_env)
        else:
            d_stack = torch.stack(
                [actions[uid] - prev_actions[uid] for uid in possible_agents], dim=1
            )
            action_delta_per_env = d_stack.abs().sum(dim=(1, 2)) / A

        step_in_ep = unwrapped.episode_length_buf.clamp(max=T_max - 1).long()  # [N]
        # Scatter-add into bins.
        sum_action_sum.scatter_add_(0, step_in_ep, action_sum_per_env)
        sum_action_delta.scatter_add_(0, step_in_ep, action_delta_per_env)
        sumsq_action_sum.scatter_add_(0, step_in_ep, action_sum_per_env ** 2)
        sumsq_action_delta.scatter_add_(0, step_in_ep, action_delta_per_env ** 2)
        count.scatter_add_(0, step_in_ep,
                           torch.ones_like(step_in_ep, dtype=torch.long))

        prev_actions = {uid: actions[uid].clone() for uid in possible_agents}
        obs, _, _, _, _ = env_wrapped.step(actions)

        if (step + 1) % max(n_steps // 10, 1) == 0:
            pct = 100.0 * (step + 1) / n_steps
            print(f"    step {step + 1}/{n_steps} ({pct:.0f}%)", flush=True)

    cnt_f = count.clamp(min=1).float()
    mean_sum = (sum_action_sum / cnt_f).cpu().numpy()
    mean_delta = (sum_action_delta / cnt_f).cpu().numpy()
    var_sum = (sumsq_action_sum / cnt_f - (sum_action_sum / cnt_f) ** 2).clamp(min=0)
    var_delta = (sumsq_action_delta / cnt_f - (sum_action_delta / cnt_f) ** 2).clamp(min=0)
    std_sum = var_sum.sqrt().cpu().numpy()
    std_delta = var_delta.sqrt().cpu().numpy()
    counts = count.cpu().numpy().tolist()

    sample_steps = [s for s in [0, 4, 8, 16, 32, 64, 128, 256, T_max - 1] if s < T_max]
    table = {
        int(s): {
            "n_samples":   counts[s],
            "action_sum":  {"mean": float(mean_sum[s]), "std": float(std_sum[s])},
            "action_delta": {"mean": float(mean_delta[s]), "std": float(std_delta[s])},
        }
        for s in sample_steps
    }

    # Verdict rule: early-step delta vs steady-state delta. Steady is steps
    # max(30, T/4)..T/2 (skip transient + skip late truncation effects).
    early = mean_delta[5:30].mean() if T_max > 30 else mean_delta[5:T_max // 2].mean()
    steady = mean_delta[max(30, T_max // 4): T_max // 2].mean()
    ratio = float(early / max(steady, 1e-8))
    if ratio >= 1.3:
        verdict = "adaptive"
    elif ratio <= 1.1:
        verdict = "robust"
    else:
        verdict = "partially-adaptive"

    print(f"\n  early/steady action_delta ratio = {ratio:.3f}  →  {verdict}")

    result = {
        "n_envs": num_envs,
        "max_episode_length": T_max,
        "samples_per_step": counts,
        "mean_action_sum_by_step": mean_sum.tolist(),
        "std_action_sum_by_step": std_sum.tolist(),
        "mean_action_delta_by_step": mean_delta.tolist(),
        "std_action_delta_by_step": std_delta.tolist(),
        "sample_table": table,
        "early_steady_ratio": ratio,
        "verdict": verdict,
    }

    # Long-format CSV.
    csv_path = os.path.join(output_dir, "probe1_action_vs_step.csv")
    with open(csv_path, "w") as f:
        f.write("step,n_samples,mean_action_sum,std_action_sum,mean_action_delta,std_action_delta\n")
        for s in range(T_max):
            f.write(f"{s},{counts[s]},{mean_sum[s]:.6f},{std_sum[s]:.6f},"
                    f"{mean_delta[s]:.6f},{std_delta[s]:.6f}\n")
    print(f"  wrote {csv_path}")

    json_path = os.path.join(output_dir, "probe1_action_vs_step.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {json_path}")

    if _MPL_OK:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(T_max)
        ax1.plot(x, mean_sum, color="C0", label="mean")
        ax1.fill_between(x, mean_sum - std_sum, mean_sum + std_sum, alpha=0.2,
                         color="C0", label="±1σ")
        ax1.set_xlabel("step in episode")
        ax1.set_ylabel("|action|_1 (mean over agents)")
        ax1.set_title("Probe 1 — action_sum")
        ax1.legend()
        ax2.plot(x, mean_delta, color="C1", label="mean")
        ax2.fill_between(x, mean_delta - std_delta, mean_delta + std_delta,
                         alpha=0.2, color="C1", label="±1σ")
        ax2.set_xlabel("step in episode")
        ax2.set_ylabel("|Δaction|_1 (mean over agents)")
        ax2.set_title(f"Probe 1 — action_delta (early/steady={ratio:.2f}, verdict={verdict})")
        ax2.legend()
        fig.tight_layout()
        png_path = os.path.join(output_dir, "probe1_action_vs_step.png")
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  wrote {png_path}")

    return result


# ---------------------------------------------------------------------------
# Probe 2: feature decoding (pre-GRU MLP + post-GRU h).
# ---------------------------------------------------------------------------

def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0
              ) -> Tuple[float, float]:
    """Closed-form ridge regression. Returns (test_R2, pearson_r)."""
    if y.std() < 1e-6:
        # Latent is degenerate (effectively constant) — decoding undefined.
        return float("nan"), float("nan")
    n = X.shape[0]
    split = int(0.8 * n)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    y_mu = y_tr.mean()
    y_tr_c = y_tr - y_mu
    d = X_tr.shape[1]
    A = X_tr.T @ X_tr + alpha * np.eye(d)
    w = np.linalg.solve(A, X_tr.T @ y_tr_c)
    pred = X_te @ w + y_mu
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    if ss_tot < 1e-10:
        return float("nan"), float("nan")
    ss_res = ((y_te - pred) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot)
    pcc = float(np.corrcoef(pred, y_te)[0, 1]) if y_te.std() > 1e-8 else float("nan")
    return r2, pcc


def _fit_mlp(X: np.ndarray, y: np.ndarray, hidden: int = 32,
             epochs: int = 300, lr: float = 1e-2) -> float:
    """Small MLP fit on standardized target. Returns test R²."""
    if y.std() < 1e-6:
        return float("nan")
    n = X.shape[0]
    split = int(0.8 * n)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-8
    Xn = (X - mu) / sd
    y_mu, y_sd = y[train_idx].mean(), y[train_idx].std() + 1e-8
    yn = (y - y_mu) / y_sd  # standardize target → loss stays well-scaled
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(yn, dtype=torch.float32, device=device)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        pred_tr = net(Xt[train_idx]).squeeze(-1)
        loss = ((pred_tr - yt[train_idx]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_te_n = net(Xt[test_idx]).squeeze(-1).cpu().numpy()
    pred_te = pred_te_n * y_sd + y_mu  # back to original scale
    y_te = y[test_idx]
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    if ss_tot < 1e-10:
        return float("nan")
    ss_res = ((y_te - pred_te) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def run_probe2(agent, env_wrapped, unwrapped, possible_agents: List[str],
               num_envs: int, sample_steps: List[int], output_dir: str) -> dict:
    print("\n" + "=" * 80)
    print("PROBE 2 — Linear / MLP decoding of latents from hidden features")
    print("=" * 80)
    device = unwrapped.device
    A = len(possible_agents)
    T_max = int(unwrapped.max_episode_length)
    sample_steps = sorted(set(s for s in sample_steps if 0 <= s < T_max))
    print(f"  num_envs={num_envs}, num_agents={A}, sample_steps={sample_steps}")

    # Forward hook(s) on policy.net — captures pre-GRU MLP output per agent.act().
    pre_capture = _PreGRUCapture([agent.policies[uid] for uid in possible_agents])

    # Storage: per T, per agent, list of (h_post, h_pre, latents) tensors. We
    # collect across the 2-episode rollout: each env contributes once per T,
    # but with 2 episodes each env may pass T multiple times — we keep the
    # first occurrence per (env, episode-1).
    per_T_post: Dict[int, List[torch.Tensor]] = {s: [] for s in sample_steps}
    per_T_pre: Dict[int, List[torch.Tensor]] = {s: [] for s in sample_steps}
    per_T_lat: Dict[int, List[torch.Tensor]] = {s: [] for s in sample_steps}

    # Also track the last episode-id each env has crossed each T at so we
    # gather one sample per (env, episode).
    seen_at_T: Dict[int, torch.Tensor] = {
        s: torch.zeros(num_envs, dtype=torch.long, device=device) for s in sample_steps
    }
    episode_id = torch.zeros(num_envs, dtype=torch.long, device=device)

    obs, _ = env_wrapped.reset()
    n_steps = 2 * T_max
    for step in range(n_steps):
        pre_capture.reset()
        with torch.no_grad():
            actions, _, outputs = agent.act(obs, 0, 0)
            for uid in possible_agents:
                actions[uid] = outputs[uid]["mean_actions"]

        # post-GRU hidden state — outputs[uid]["rnn"][0] is shape [layers, B, H].
        post_per_agent = []
        for uid in possible_agents:
            h = outputs[uid]["rnn"][0].detach()
            post_per_agent.append(h.transpose(0, 1).reshape(num_envs, -1))  # [N, layers*H]
        post = torch.stack(post_per_agent, dim=1)  # [N, A, layers*H]

        # pre-GRU MLP features captured by the forward hook during agent.act().
        # The hook fires once per agent in possible_agents order.
        if len(pre_capture.captures) != A:
            raise RuntimeError(
                f"expected {A} pre-GRU captures, got {len(pre_capture.captures)} "
                "(shared policy + multiple agents.act() calls expected)"
            )
        pre = torch.stack(pre_capture.captures, dim=1)  # [N, A, hidden]

        lats, _names = _stack_latents(_read_eff_progress(unwrapped))  # [N, A, 8]

        step_in_ep = unwrapped.episode_length_buf.long()
        for s in sample_steps:
            mask = (step_in_ep == s) & (seen_at_T[s] == episode_id)
            if mask.any():
                idx = mask.nonzero(as_tuple=False).squeeze(-1)
                per_T_post[s].append(post[idx].cpu())
                per_T_pre[s].append(pre[idx].cpu())
                per_T_lat[s].append(lats[idx].cpu())
                seen_at_T[s][idx] += 1

        # advance env, update episode tracker
        prev_step = step_in_ep
        obs, _, terminated, truncated, _ = env_wrapped.step(actions)
        # detect resets
        first_uid = possible_agents[0]
        done = (terminated[first_uid] | truncated[first_uid]) if isinstance(terminated, dict) \
               else (terminated | truncated)
        if done.dim() > 1:
            done = done.squeeze(-1)
        if done.any():
            episode_id[done] += 1

        if (step + 1) % max(n_steps // 10, 1) == 0:
            pct = 100.0 * (step + 1) / n_steps
            print(f"    step {step + 1}/{n_steps} ({pct:.0f}%)", flush=True)

    pre_capture.detach()

    # Fit per (T, latent-axis, feature-type).
    latent_names = [k.lstrip("_") for k in EFF_PROGRESS_KEYS]
    results: Dict[str, dict] = {}
    for s in sample_steps:
        if not per_T_post[s]:
            print(f"  step T={s}: no samples — skipping")
            continue
        dim_post = per_T_post[s][0].shape[-1]
        dim_pre = per_T_pre[s][0].shape[-1]
        dim_lat = per_T_lat[s][0].shape[-1]
        post_X = torch.cat(per_T_post[s], dim=0).reshape(-1, dim_post).numpy()  # [N*A, dim_post]
        pre_X = torch.cat(per_T_pre[s], dim=0).reshape(-1, dim_pre).numpy()    # [N*A, dim_pre]
        lat_Y = torch.cat(per_T_lat[s], dim=0).reshape(-1, dim_lat).numpy()    # [N*A, 8]
        results[str(s)] = {}
        print(f"\n  T={s}: n={post_X.shape[0]} (env,agent) samples")
        for k, name in enumerate(latent_names):
            y = lat_Y[:, k]
            r2_post, r_post = _fit_ridge(post_X, y)
            r2_pre, r_pre = _fit_ridge(pre_X, y)
            r2_post_mlp = _fit_mlp(post_X, y)
            r2_pre_mlp = _fit_mlp(pre_X, y)
            results[str(s)][name] = {
                "post_R2_ridge": r2_post,
                "post_R2_mlp": r2_post_mlp,
                "post_pearson_r": r_post,
                "pre_R2_ridge": r2_pre,
                "pre_R2_mlp": r2_pre_mlp,
                "pre_pearson_r": r_pre,
            }
            print(f"    {name:32s}  post R² ridge={r2_post:+.3f} mlp={r2_post_mlp:+.3f}  "
                  f"pre R² ridge={r2_pre:+.3f} mlp={r2_pre_mlp:+.3f}")

    # Verdict: per-axis adaptive iff max_T R²_mlp > 0.5 on either feature.
    per_axis_verdict = {}
    for name in latent_names:
        post_vals = [results[str(s)][name]["post_R2_mlp"]
                     for s in sample_steps if str(s) in results and name in results[str(s)]
                     and np.isfinite(results[str(s)][name]["post_R2_mlp"])]
        pre_vals = [results[str(s)][name]["pre_R2_mlp"]
                    for s in sample_steps if str(s) in results and name in results[str(s)]
                    and np.isfinite(results[str(s)][name]["pre_R2_mlp"])]
        max_r2 = max(post_vals) if post_vals else float("nan")
        max_r2_pre = max(pre_vals) if pre_vals else float("nan")
        candidates = [v for v in (max_r2, max_r2_pre) if np.isfinite(v)]
        if not candidates:
            v = "no-data (degenerate latent)"
            best = float("nan")
        else:
            best = max(candidates)
            if best > 0.5:
                v = "adaptive"
            elif best < 0.1:
                v = "robust"
            else:
                v = "partial"
        per_axis_verdict[name] = {"max_R2_post": float(max_r2) if np.isfinite(max_r2) else None,
                                  "max_R2_pre": float(max_r2_pre) if np.isfinite(max_r2_pre) else None,
                                  "verdict": v}

    result = {
        "sample_steps": sample_steps,
        "n_envs": num_envs,
        "per_T": results,
        "per_axis_verdict": per_axis_verdict,
    }
    print("\n  Per-axis verdict:")
    for name, v in per_axis_verdict.items():
        print(f"    {name:32s}  {v['verdict']:10s}  "
              f"(post max R²={v['max_R2_post']}, pre max R²={v['max_R2_pre']})")

    json_path = os.path.join(output_dir, "probe2_hidden_decoding.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  wrote {json_path}")

    if _MPL_OK:
        # Heatmap of post-GRU R²_mlp.
        mat_post = np.full((len(latent_names), len(sample_steps)), np.nan)
        mat_pre = np.full((len(latent_names), len(sample_steps)), np.nan)
        for j, s in enumerate(sample_steps):
            if str(s) not in results:
                continue
            for i, name in enumerate(latent_names):
                if name in results[str(s)]:
                    mat_post[i, j] = results[str(s)][name]["post_R2_mlp"]
                    mat_pre[i, j] = results[str(s)][name]["pre_R2_mlp"]
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))
        for ax, mat, title in ((axa, mat_post, "post-GRU"), (axb, mat_pre, "pre-GRU")):
            im = ax.imshow(mat, vmin=-0.2, vmax=1.0, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(sample_steps)))
            ax.set_xticklabels(sample_steps)
            ax.set_yticks(range(len(latent_names)))
            ax.set_yticklabels(latent_names)
            ax.set_xlabel("step T")
            ax.set_title(f"Probe 2 — R²_mlp ({title})")
            for i in range(len(latent_names)):
                for j in range(len(sample_steps)):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                                color="white" if mat[i, j] < 0.5 else "black", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        png_path = os.path.join(output_dir, "probe2_hidden_decoding.png")
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  wrote {png_path}")

    return result


# ---------------------------------------------------------------------------
# Probe 3: hidden-state swap counterfactual.
# ---------------------------------------------------------------------------

def _zero_rnn_states(agent, possible_agents: List[str]):
    """Zero the agent's GRU hidden state across all envs (for clean rollout starts)."""
    for uid in possible_agents:
        states = agent._rnn_states.get(uid, {}).get("policy", None)
        if states is None:
            continue
        if isinstance(states, list):
            for t in states:
                t.zero_()
        else:
            states.zero_()


def _rollout_with_swap(agent, env_wrapped, unwrapped, possible_agents: List[str],
                       num_envs: int, swap_step: int, swap_mode: str, seed: int,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single rollout to max_episode_length. At swap_step (per env), swap
    bank-A (envs 0..N/2) with bank-B (N/2..N) GRU hidden states, then record
    the L2 ratio ‖μ_swap − μ_ref‖² / σ² each step after.

    swap_mode: 'cross' (A↔B) or 'identity' (no-op control — should yield zero).

    Returns (mu_actions, log_std):
        mu_actions: [T_max, num_envs, A, act_dim] tensor of post-swap mean actions
        log_std:    [A, act_dim]  policy log-std (state-independent in this model)
    """
    device = unwrapped.device
    A = len(possible_agents)
    T_max = int(unwrapped.max_episode_length)
    half = num_envs // 2

    _zero_rnn_states(agent, possible_agents)
    # Force reproducible env init across rollouts: pass seed to gym.reset().
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        obs, _ = env_wrapped.reset(seed=seed)
    except TypeError:
        obs, _ = env_wrapped.reset()
    mu_traj = []
    swapped = torch.zeros(num_envs, dtype=torch.bool, device=device)

    log_std = None

    for step in range(T_max):
        with torch.no_grad():
            actions, _, outputs = agent.act(obs, 0, 0)
            for uid in possible_agents:
                actions[uid] = outputs[uid]["mean_actions"]

        # Stack mean actions: [num_envs, A, act_dim]
        mu = torch.stack([outputs[uid]["mean_actions"] for uid in possible_agents], dim=1)
        mu_traj.append(mu.detach().cpu().clone())

        if log_std is None:
            # log_std_parameter is per-action, broadcast across batch.
            # Shape per-policy: [act_dim]; we expand to [A, act_dim].
            ls = torch.stack(
                [agent.policies[uid].log_std_parameter.detach() for uid in possible_agents],
                dim=0,
            )
            log_std = ls.cpu()

        step_in_ep = unwrapped.episode_length_buf.long()
        to_swap = (step_in_ep == swap_step) & (~swapped)
        if to_swap.any() and swap_mode != "none":
            # Swap GRU hidden state of paired envs (i, i+half) for to-swap envs in bank A.
            for uid in possible_agents:
                h = agent._rnn_states[uid]["policy"][0]  # [layers, num_envs, hidden]
                idx_A = to_swap[:half].nonzero(as_tuple=False).squeeze(-1)
                if idx_A.numel() == 0:
                    continue
                idx_B = idx_A + half
                if swap_mode == "cross":
                    tmp = h[:, idx_A, :].clone()
                    h[:, idx_A, :] = h[:, idx_B, :].clone()
                    h[:, idx_B, :] = tmp
                # identity mode: do nothing — control case
            swapped[to_swap] = True

        obs, _, _, _, _ = env_wrapped.step(actions)

    mu_arr = torch.stack(mu_traj, dim=0).numpy()  # [T_max, N, A, act_dim]
    return mu_arr, log_std.numpy() if log_std is not None else np.zeros((A, mu_arr.shape[-1]))


def run_probe3(agent, env_wrapped, unwrapped, possible_agents: List[str],
               num_envs: int, swap_step: int, output_dir: str) -> dict:
    print("\n" + "=" * 80)
    print("PROBE 3 — Hidden-state swap counterfactual")
    print("=" * 80)
    if num_envs % 2 != 0:
        raise ValueError("Probe 3 requires even num_envs (pairs A/B).")
    half = num_envs // 2
    T_max = int(unwrapped.max_episode_length)
    print(f"  num_envs={num_envs}, swap_step={swap_step}, T_max={T_max}")

    # Reference rollout: no swap.
    print("  [1/3] reference rollout (no swap)…", flush=True)
    mu_ref, log_std = _rollout_with_swap(
        agent, env_wrapped, unwrapped, possible_agents, num_envs, swap_step,
        swap_mode="none", seed=args_cli.seed,
    )

    # Cross-swap rollout: same seed → should match ref exactly up to T=swap_step.
    print("  [2/3] cross-swap rollout…", flush=True)
    mu_swap, _ = _rollout_with_swap(
        agent, env_wrapped, unwrapped, possible_agents, num_envs, swap_step,
        swap_mode="cross", seed=args_cli.seed,
    )

    # Identity-swap control: same seed, no-op at swap step → must equal ref.
    print("  [3/3] identity-swap control rollout (reproducibility check)…", flush=True)
    mu_ctrl, _ = _rollout_with_swap(
        agent, env_wrapped, unwrapped, possible_agents, num_envs, swap_step,
        swap_mode="none", seed=args_cli.seed,
    )

    # KL approximation under fixed log-std Gaussian: 0.5 * ‖μ1 − μ2‖² / σ²
    sigma2 = np.exp(2 * log_std)  # [A, act_dim]
    # Use bank A (first half) for divergence — that's whose h was overwritten.
    def kl_curve(mu_a: np.ndarray, mu_b: np.ndarray) -> np.ndarray:
        diff2 = (mu_a[:, :half] - mu_b[:, :half]) ** 2  # [T, half, A, act_dim]
        kl = 0.5 * (diff2 / sigma2[None, None, :, :]).sum(axis=(-1, -2))  # [T, half]
        return kl.mean(axis=-1)  # [T]

    kl_swap = kl_curve(mu_swap, mu_ref)
    kl_ctrl = kl_curve(mu_ctrl, mu_ref)

    # Verdict — detrended: post-swap excess KL minus pre-swap noise floor.
    pre_window = slice(0, swap_step)
    post_window = slice(swap_step, min(swap_step + 100, T_max))
    swap_pre = float(kl_swap[pre_window].mean())
    swap_post = float(kl_swap[post_window].mean())
    ctrl_pre = float(kl_ctrl[pre_window].mean())
    ctrl_post = float(kl_ctrl[post_window].mean())
    # Pre-swap divergence between rollouts is the Isaac Sim non-determinism floor.
    swap_excess = swap_post - swap_pre
    ctrl_excess = ctrl_post - ctrl_pre
    detrended_signal = swap_excess - ctrl_excess
    print(f"\n  pre-swap KL  swap={swap_pre:.3f}  ctrl={ctrl_pre:.3f}", flush=True)
    print(f"  post-swap KL swap={swap_post:.3f}  ctrl={ctrl_post:.3f}", flush=True)
    print(f"  excess (post-pre)   swap={swap_excess:+.3f}  ctrl={ctrl_excess:+.3f}", flush=True)
    print(f"  detrended signal (swap_excess − ctrl_excess) = {detrended_signal:+.3f}", flush=True)

    # Detrended-signal-based verdict.
    if abs(detrended_signal) < 0.5:
        verdict = "robust"
    elif detrended_signal > 2.0:
        verdict = "adaptive"
    else:
        verdict = "partial"
    if ctrl_excess > 1.0:
        # The "control" should not show post-swap excess; if it does, the noise
        # floor itself is drifting — interpret detrended_signal cautiously.
        verdict += " (caution: control excess >1 nat; non-determinism floor drifts)"
    print(f"  verdict: {verdict}", flush=True)

    result = {
        "n_envs": num_envs,
        "swap_step": swap_step,
        "T_max": T_max,
        "kl_swap": kl_swap.tolist(),
        "kl_ctrl": kl_ctrl.tolist(),
        "swap_pre_kl": swap_pre,
        "swap_post_kl": swap_post,
        "ctrl_pre_kl": ctrl_pre,
        "ctrl_post_kl": ctrl_post,
        "swap_excess": swap_excess,
        "ctrl_excess": ctrl_excess,
        "detrended_signal": detrended_signal,
        "verdict": verdict,
    }
    csv_path = os.path.join(output_dir, "probe3_swap_kl.csv")
    with open(csv_path, "w") as f:
        f.write("step,kl_swap,kl_ctrl\n")
        for s in range(T_max):
            f.write(f"{s},{kl_swap[s]:.6e},{kl_ctrl[s]:.6e}\n")
    print(f"  wrote {csv_path}")

    json_path = os.path.join(output_dir, "probe3_swap_kl.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {json_path}")

    if _MPL_OK:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(T_max)
        ax.plot(x, kl_swap, color="C3", label="cross-swap KL")
        ax.plot(x, kl_ctrl, color="C7", alpha=0.7, label="identity-control KL")
        ax.axvline(swap_step, color="k", linestyle="--", alpha=0.5,
                   label=f"swap @ T={swap_step}")
        ax.set_xlabel("step in episode")
        ax.set_ylabel("approximate KL (per env, mean over bank A)")
        ax.set_title(f"Probe 3 — hidden-state swap (verdict={verdict})")
        ax.legend()
        fig.tight_layout()
        png_path = os.path.join(output_dir, "probe3_swap_kl.png")
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  wrote {png_path}")

    return result


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, "skrl_mappo_rnn_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    exp_cfg = _selected_experiment
    print(f"[PROBE] Task: {args_cli.task} (experiment base: {exp_cfg.name})")

    if exp_cfg.env_overrides:
        apply_env_overrides(env_cfg, exp_cfg.env_overrides)
    if exp_cfg.agent_overrides:
        apply_agent_overrides(agent_cfg, exp_cfg.agent_overrides)

    torch.manual_seed(args_cli.seed)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 20.0

    # Pin curriculum to a fully-ramped step.
    env_cfg.use_debug_initial_step = True
    env_cfg.debug_initial_step = int(args_cli.step)
    print(f"[PROBE] Curriculum pinned to step {args_cli.step}")
    print(f"[PROBE] num_envs={args_cli.num_envs}, seed={args_cli.seed}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env)
    unwrapped = env.unwrapped
    possible_agents = env_wrapped.possible_agents

    agent = _load_rnn_policy(args_cli.checkpoint, env_wrapped, agent_cfg, possible_agents)
    print(f"[PROBE] agent ready (args_cli.output_dir={args_cli.output_dir!r})", flush=True)

    # Output dir.
    if args_cli.output_dir is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_dir = os.path.join(os.path.dirname(__file__), "outputs", date_str, "probe036")
    else:
        out_dir = args_cli.output_dir
    print(f"[PROBE] mkdir({out_dir!r})", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[PROBE] Output dir ready: {out_dir}", flush=True)

    probes = ["1", "2", "3"] if args_cli.probe == "all" else [args_cli.probe]
    summary: dict = {
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "curriculum_step": args_cli.step,
        "seed": args_cli.seed,
        "results": {},
    }

    if "1" in probes:
        print(f"[PROBE] starting Probe 1…", flush=True)
        n_p1 = min(args_cli.num_envs, 512)
        if n_p1 != args_cli.num_envs:
            print(f"[PROBE 1] using num_envs={n_p1} (capped at 512)", flush=True)
        summary["results"]["probe1"] = run_probe1(
            agent, env_wrapped, unwrapped, possible_agents,
            num_envs=args_cli.num_envs, output_dir=out_dir,
        )
    if "2" in probes:
        summary["results"]["probe2"] = run_probe2(
            agent, env_wrapped, unwrapped, possible_agents,
            num_envs=args_cli.num_envs, sample_steps=args_cli.probe2_steps,
            output_dir=out_dir,
        )
    if "3" in probes:
        summary["results"]["probe3"] = run_probe3(
            agent, env_wrapped, unwrapped, possible_agents,
            num_envs=args_cli.num_envs, swap_step=args_cli.swap_step,
            output_dir=out_dir,
        )

    with open(os.path.join(out_dir, "probe036_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[PROBE] All done. Summary: {os.path.join(out_dir, 'probe036_summary.json')}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
