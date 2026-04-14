#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the unified delay pipeline (ticket 029 design).

Plots 5 independent cases (envs) over a 2-second window. Each case shows:
    - black: ground-truth signal (no noise, no delay)
    - blue:  reward-path output (clean + delayed; `use_noise=False`)
    - red:   observation-path output (noisy + delayed + dropout;
             `use_noise=True`)

After ticket 029 the pipeline advances raw + noisy payloads together under
a single delay realization (shared staleness mask, shared latency draw,
shared dropout mask). Expected behavior in every case:
    - red tracks blue in timing (same delay, same dropout holds)
    - red deviates from blue in amplitude (by the injected noise)

Self-contained — does not require Isaac Sim.

Usage:
    conda run -n env_isaaclab python plot_delay_pipeline.py
    conda run -n env_isaaclab python plot_delay_pipeline.py --output /tmp/delay.png
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load delay_system_v3 submodules directly to avoid pulling in Isaac Sim
# through `delay_system_v3/__init__.py` → `multi_agent_wrapper.py`.
_MODULE_DIR = Path(__file__).resolve().parent.parent


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pkg_name = "delay_system_v3_plot"
_pkg = types.ModuleType(_pkg_name)
_pkg.__path__ = [str(_MODULE_DIR)]
sys.modules[_pkg_name] = _pkg

_cfg_mod = _load(f"{_pkg_name}.delay_cfg_v3", _MODULE_DIR / "delay_cfg_v3.py")
_storage_mod = _load(f"{_pkg_name}.field_storage", _MODULE_DIR / "field_storage.py")
_sampling_mod = _load(f"{_pkg_name}.sampling_strategies", _MODULE_DIR / "sampling_strategies.py")
_pipeline_mod = _load(f"{_pkg_name}.delay_pipeline_v3", _MODULE_DIR / "delay_pipeline_v3.py")
_system_mod = _load(f"{_pkg_name}.delay_system_v3", _MODULE_DIR / "delay_system_v3.py")

DelayPipelineCfgV3 = _cfg_mod.DelayPipelineCfgV3
DistributionCfg = _cfg_mod.DistributionCfg
DropoutCfg = _cfg_mod.DropoutCfg
FirstOrderLagCfg = _cfg_mod.FirstOrderLagCfg
LatencyCfg = _cfg_mod.LatencyCfg
PerspectiveCfg = _cfg_mod.PerspectiveCfg
StalenessCfg = _cfg_mod.StalenessCfg
UnifiedDelayCfgV3 = _cfg_mod.UnifiedDelayCfgV3
UnifiedDelaySystem = _system_mod.UnifiedDelaySystem


FIELD = "drone_0.bbox_detection"
PERSPECTIVE = "other"  # "other" exercises the longer (communication) delay path


def build_cfg(dt: float) -> UnifiedDelayCfgV3:
    pipeline = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=True,
            distribution=DistributionCfg(
                type="normal", mean=0.20, std=0.06, min_value=0.04
            ),
        ),
        staleness=StalenessCfg(
            enabled=True,
            fps_distribution=DistributionCfg(
                type="uniform", mean=15.0, half_range=5.0, min_value=5.0, max_value=60.0
            ),
        ),
        dropout=DropoutCfg(enabled=True, probability=0.08),
        first_order_lag=FirstOrderLagCfg(enabled=False),
    )
    cfg = UnifiedDelayCfgV3(dt=dt)
    cfg.ego = PerspectiveCfg(pipeline=pipeline)
    cfg.other = PerspectiveCfg(pipeline=pipeline)
    return cfg


def simulate(num_envs: int, num_steps: int, dt: float, noise_std: float, seed: int) -> dict:
    """Run the delay system and collect (gt, reward-path, obs-path) traces.

    Reproduces the real env-step ordering: reward call (use_noise=False) runs
    BEFORE observation call (use_noise=True) within each step. After ticket
    029 the first call advances once, filling both caches; the second call is
    a cache read.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")

    cfg = build_cfg(dt=dt)
    system = UnifiedDelaySystem(cfg=cfg, num_envs=num_envs, device=device)
    system.register_field(FIELD, data_shape=(4,))
    system.set_delay_mode("random", progress=1.0)

    gt_hist = np.zeros((num_steps, num_envs))
    reward_hist = np.zeros((num_steps, num_envs))
    obs_hist = np.zeros((num_steps, num_envs))

    for step in range(num_steps):
        t = torch.full((num_envs,), step * dt, device=device)
        system.set_time(t)
        system.step(reset_env_ids=None)

        # Ground-truth bbox cx: sinusoidal sweep. Other components fixed.
        cx_gt = 320.0 + 200.0 * np.sin(2.0 * np.pi * 0.8 * (step * dt))
        data = torch.zeros(num_envs, 4, device=device)
        data[:, 0] = cx_gt
        data[:, 1] = 240.0
        data[:, 2] = 40.0
        data[:, 3] = 40.0

        # Per-env independent noise on cx only (simulates detector center noise).
        noisy = data.clone()
        noisy[:, 0] = data[:, 0] + torch.randn(num_envs, device=device) * noise_std

        system.store(FIELD, data, noisy_data=noisy)

        # Reward-path query first (matches _get_rewards before _get_observations).
        reward_data, _ = system.get_delayed(
            FIELD, PERSPECTIVE, use_noise=False, allow_dropout=False
        )
        # Observation-path query second — must get the SAME delay but noisy payload.
        obs_data, _ = system.get_delayed(
            FIELD, PERSPECTIVE, use_noise=True, allow_dropout=True
        )

        gt_hist[step] = cx_gt
        reward_hist[step] = reward_data[:, 0].cpu().numpy()
        obs_hist[step] = obs_data[:, 0].cpu().numpy()

    return {
        "t": np.arange(num_steps) * dt,
        "gt": gt_hist,
        "reward": reward_hist,
        "obs": obs_hist,
    }


def plot(data: dict, output: Path, num_envs: int):
    fig, axes = plt.subplots(num_envs, 1, figsize=(10, 2.2 * num_envs), sharex=True)
    if num_envs == 1:
        axes = [axes]
    fig.suptitle(
        "Unified delay pipeline: reward and observation share one delay realization "
        f"(perspective='{PERSPECTIVE}', bbox center x)",
        fontsize=12,
    )

    t = data["t"]
    for env_i in range(num_envs):
        ax = axes[env_i]
        ax.plot(t, data["gt"][:, env_i], color="black", linewidth=1.0,
                label="GT (clean, no delay)", alpha=0.7)
        ax.plot(t, data["reward"][:, env_i], color="tab:blue", linewidth=1.6,
                label="reward path (clean + delayed)")
        ax.plot(t, data["obs"][:, env_i], color="tab:red", linewidth=1.0,
                label="obs path (noisy + delayed + dropout)", alpha=0.9)
        ax.set_ylabel(f"case {env_i}\ncx [px]", fontsize=9)
        ax.grid(True, alpha=0.3)
        if env_i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output, dpi=140)
    print(f"Saved plot → {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=5)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=25.0,
                        help="Gaussian std applied to cx on the noisy payload (pixels)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "delay_pipeline.png")
    args = parser.parse_args()

    num_steps = int(round(args.duration / args.dt))
    print(f"Simulating {args.duration}s ({num_steps} steps) × {args.envs} cases, "
          f"dt={args.dt}s, noise_std={args.noise_std}px")

    result = simulate(
        num_envs=args.envs, num_steps=num_steps, dt=args.dt,
        noise_std=args.noise_std, seed=args.seed,
    )

    # Contract checks: reward and obs share the same delay (same staleness holds,
    # same latency draw, same dropout mask) but differ by the injected noise.
    # Use allow_dropout=False on both so dropout-hold differences don't confound.
    diff = result["obs"] - result["reward"]
    print(f"RMS(obs - reward) on cx:    {float(np.sqrt(np.mean(diff**2))):.3f} px  "
          f"(≈ noise_std ⇒ noise passes through)")
    print(f"mean|obs - reward| on cx:   {float(np.mean(np.abs(diff))):.3f} px")

    plot(result, args.output, num_envs=args.envs)


if __name__ == "__main__":
    main()