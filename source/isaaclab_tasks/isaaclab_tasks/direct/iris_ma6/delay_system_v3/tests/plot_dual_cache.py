#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the dual-cache delay pipeline (ticket 029).

Produces a 4-panel figure illustrating the invariants verified by
``test_dual_cache.py``:

1. **Shared delay realization** — a state field with a Gaussian-noisy
   sibling flows through ONE DelayPipelineV3. Both payloads exit with
   the same delay and timestamp; the difference between them is the
   injected noise, preserved through the delay. Plots both over time for
   several independent envs (sampled latency varies per env).

2. **Reward vs observation at the same sim step** — the regression for
   the shared-pipeline idempotency bug. On every step we advance ONCE
   with ``(raw, noisy)``, then query both ``use_noise=False`` (reward)
   and ``use_noise=True`` (obs). Both queries return distinct payloads;
   their timestamps match. This is what the old code silently conflated.

3. **Clean-only field** — when a field has no separate noisy payload
   (``advance(raw, None, ...)``), the noisy cache mirrors raw. Obs and
   reward queries return identical data; the gap between them is always
   zero. This is the state-field-with-no-curriculum-noise case.

4. **Shared dropout mask holds raw and noisy jointly** — under high
   dropout probability, the pipeline's dropout stage samples ONE mask
   per step and applies it to both payloads. On a dropped step, the
   held raw and held noisy buffers both retain their previous values
   together; timestamps freeze alongside. This is the property that
   makes raycaster + replicator usable as a pair: a dropped detection
   drops BOTH views simultaneously.

Self-contained — does not require Isaac Sim. Imports
``delay_system_v3`` submodules directly via importlib.

Usage:
    conda run -n env_isaaclab python plot_dual_cache.py
    conda run -n env_isaaclab python plot_dual_cache.py --output /tmp/dual_cache.png
    conda run -n env_isaaclab python plot_dual_cache.py --envs 6 --duration 2.5
"""

from __future__ import annotations

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


# -----------------------------------------------------------------------------
# Import delay_system_v3 submodules without triggering Isaac Sim deps.
# -----------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent.parent


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pkg_name = "delay_system_v3_plot_dual"
_pkg = types.ModuleType(_pkg_name)
_pkg.__path__ = [str(_MODULE_DIR)]
sys.modules[_pkg_name] = _pkg

_cfg_mod = _load(f"{_pkg_name}.delay_cfg_v3", _MODULE_DIR / "delay_cfg_v3.py")
_sampling_mod = _load(f"{_pkg_name}.sampling_strategies", _MODULE_DIR / "sampling_strategies.py")
_pipeline_mod = _load(f"{_pkg_name}.delay_pipeline_v3", _MODULE_DIR / "delay_pipeline_v3.py")

DelayPipelineCfgV3 = _cfg_mod.DelayPipelineCfgV3
DistributionCfg = _cfg_mod.DistributionCfg
DropoutCfg = _cfg_mod.DropoutCfg
FirstOrderLagCfg = _cfg_mod.FirstOrderLagCfg
LatencyCfg = _cfg_mod.LatencyCfg
SamplingCfg = _cfg_mod.SamplingCfg
StalenessCfg = _cfg_mod.StalenessCfg
DelayPipelineV3 = _pipeline_mod.DelayPipelineV3


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_pipeline(
    num_envs: int,
    device: torch.device,
    *,
    dt: float,
    latency_mean: float = 0.12,
    latency_std: float = 0.04,
    dropout_prob: float = 0.0,
    staleness_fps_mean: float | None = None,
) -> DelayPipelineV3:
    """Construct a pipeline with a random latency distribution."""
    cfg = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=True,
            distribution=DistributionCfg(
                type="normal",
                mean=latency_mean,
                std=latency_std,
                min_value=dt,
            ),
            sampling=SamplingCfg(frequency="per_episode"),
        ),
        staleness=StalenessCfg(
            enabled=staleness_fps_mean is not None,
            fps_distribution=(
                DistributionCfg(
                    type="uniform",
                    mean=staleness_fps_mean or 15.0,
                    half_range=3.0,
                    min_value=5.0,
                    max_value=60.0,
                )
                if staleness_fps_mean is not None
                else DistributionCfg(type="constant", value=30.0)
            ),
        ),
        dropout=DropoutCfg(
            enabled=dropout_prob > 0.0,
            probability=dropout_prob,
            sampling=SamplingCfg(frequency="per_step"),
        ),
        first_order_lag=FirstOrderLagCfg(enabled=False),
    )
    pipe = DelayPipelineV3(cfg, num_envs, device, dt, data_shape=(1,))
    pipe.set_mode("random" if staleness_fps_mean is not None else "fixed", progress=1.0)
    return pipe


def _gt_signal(step: int, dt: float) -> float:
    """Clean ground-truth signal (sinusoid) at time = step*dt."""
    t = step * dt
    return 100.0 + 80.0 * np.sin(2.0 * np.pi * 0.6 * t)


# -----------------------------------------------------------------------------
# Panel 1 — shared delay realization, state field with Gaussian sibling
# -----------------------------------------------------------------------------
def panel1_shared_delay(ax_raw, ax_noisy, num_envs: int, num_steps: int,
                        dt: float, noise_std: float, seed: int):
    """Verifies: advance(raw, noisy) → raw and noisy come out with identical
    delay; the noise is preserved as the difference.

    Plots raw (solid) and noisy (dashed) per case.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    pipe = _make_pipeline(num_envs, device, dt=dt)

    gt = np.zeros(num_steps)
    raw_hist = np.zeros((num_steps, num_envs))
    noisy_hist = np.zeros((num_steps, num_envs))

    for step in range(num_steps):
        val = _gt_signal(step, dt)
        gt[step] = val
        raw = torch.full((num_envs, 1), val, device=device)
        noise = torch.randn(num_envs, 1, device=device) * noise_std
        noisy = raw + noise

        t = torch.full((num_envs,), step * dt, device=device)
        raw_out, raw_ts = pipe.process(raw, noisy, t.clone(), t.clone(),
                                       use_noise=False, allow_dropout=False)
        noisy_out, noisy_ts = pipe.query(use_noise=True, allow_dropout=False)

        raw_hist[step] = raw_out[:, 0].numpy()
        noisy_hist[step] = noisy_out[:, 0].numpy()
        # Sanity: timestamps must match at every step
        assert torch.allclose(raw_ts, noisy_ts), "shared-delay invariant broken"

    t_arr = np.arange(num_steps) * dt
    colors = plt.get_cmap("tab10").colors

    ax_raw.plot(t_arr, gt, color="black", linewidth=1.4, label="GT (no delay)", alpha=0.7)
    for env_i in range(num_envs):
        ax_raw.plot(t_arr, raw_hist[:, env_i], color=colors[env_i % 10],
                    linewidth=1.0, alpha=0.85)
    ax_raw.set_ylabel("raw (reward path)", fontsize=9)
    ax_raw.set_title("Panel 1 — Shared delay realization\n"
                     "raw and noisy advance together through one pipeline",
                     fontsize=10)
    ax_raw.legend(loc="upper right", fontsize=8)
    ax_raw.grid(True, alpha=0.3)

    ax_noisy.plot(t_arr, gt, color="black", linewidth=1.4, label="GT (no delay)", alpha=0.7)
    for env_i in range(num_envs):
        ax_noisy.plot(t_arr, noisy_hist[:, env_i], color=colors[env_i % 10],
                      linewidth=1.0, alpha=0.85, linestyle="--")
    ax_noisy.set_ylabel("noisy (obs path)", fontsize=9)
    ax_noisy.set_xlabel("time [s]", fontsize=9)
    ax_noisy.legend(loc="upper right", fontsize=8)
    ax_noisy.grid(True, alpha=0.3)


# -----------------------------------------------------------------------------
# Panel 2 — reward vs obs at the same sim step (the regression)
# -----------------------------------------------------------------------------
def panel2_reward_vs_obs(ax, num_envs: int, num_steps: int, dt: float,
                         noise_std: float, seed: int):
    """Demonstrates the bug fix. Each step: advance ONCE with (raw, noisy),
    then query reward (use_noise=False) and obs (use_noise=True). Both
    come back as DISTINCT payloads. Overplot both for one env, plus the
    running RMS of (obs - reward) to show the noise is preserved.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    pipe = _make_pipeline(num_envs, device, dt=dt, latency_mean=0.10, latency_std=0.02)

    gt = np.zeros(num_steps)
    reward_series = np.zeros(num_steps)
    obs_series = np.zeros(num_steps)
    diff_rms = np.zeros(num_steps)

    env_i = 0  # show one env for clarity
    for step in range(num_steps):
        val = _gt_signal(step, dt)
        gt[step] = val
        raw = torch.full((num_envs, 1), val, device=device)
        noisy = raw + torch.randn(num_envs, 1, device=device) * noise_std

        t = torch.full((num_envs,), step * dt, device=device)
        pipe.advance(raw, noisy, t.clone(), t.clone())
        reward, reward_ts = pipe.query(use_noise=False, allow_dropout=False)
        obs, obs_ts = pipe.query(use_noise=True, allow_dropout=False)

        reward_series[step] = reward[env_i, 0].item()
        obs_series[step] = obs[env_i, 0].item()
        diff_rms[step] = (obs - reward).pow(2).mean().sqrt().item()
        assert torch.allclose(reward_ts, obs_ts), "timestamp mismatch (bug regression)"

    t_arr = np.arange(num_steps) * dt

    ax.plot(t_arr, gt, color="black", linewidth=1.2, label="GT", alpha=0.6)
    ax.plot(t_arr, reward_series, color="tab:blue", linewidth=1.6,
            label="reward path (use_noise=False)")
    ax.plot(t_arr, obs_series, color="tab:red", linewidth=1.2,
            label="obs path (use_noise=True)", alpha=0.9)

    # Shade the noise envelope (RMS across envs)
    ax2 = ax.twinx()
    ax2.plot(t_arr, diff_rms, color="tab:purple", linewidth=1.0,
             linestyle=":", alpha=0.7, label="RMS(obs − reward)")
    ax2.set_ylabel("RMS noise reaching obs path", fontsize=8, color="tab:purple")
    ax2.tick_params(axis='y', labelcolor="tab:purple")
    ax2.set_ylim(bottom=0)

    ax.set_title(
        "Panel 2 — Reward-then-obs queries at the same sim step\n"
        "single advance, dual query. Both payloads distinct; timestamps identical.",
        fontsize=10,
    )
    ax.set_xlabel("time [s]", fontsize=9)
    ax.set_ylabel("signal value (env 0)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)


# -----------------------------------------------------------------------------
# Panel 3 — clean-only field: obs cache mirrors raw cache
# -----------------------------------------------------------------------------
def panel3_clean_only(ax, num_envs: int, num_steps: int, dt: float, seed: int):
    """advance(raw, None, ...) mirrors raw into the noisy cache.
    reward and obs queries return identical data — the gap is always zero.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    pipe = _make_pipeline(num_envs, device, dt=dt, latency_mean=0.08, latency_std=0.015)

    gt = np.zeros(num_steps)
    reward_series = np.zeros(num_steps)
    obs_series = np.zeros(num_steps)

    env_i = 0
    for step in range(num_steps):
        val = _gt_signal(step, dt)
        gt[step] = val
        raw = torch.full((num_envs, 1), val, device=device)

        t = torch.full((num_envs,), step * dt, device=device)
        # noisy_data=None → obs cache mirrors raw cache.
        pipe.advance(raw, None, t.clone(), t.clone())
        reward, _ = pipe.query(use_noise=False, allow_dropout=False)
        obs, _ = pipe.query(use_noise=True, allow_dropout=False)
        reward_series[step] = reward[env_i, 0].item()
        obs_series[step] = obs[env_i, 0].item()

    t_arr = np.arange(num_steps) * dt
    ax.plot(t_arr, gt, color="black", linewidth=1.2, label="GT", alpha=0.6)
    ax.plot(t_arr, reward_series, color="tab:blue", linewidth=2.0,
            label="reward path (use_noise=False)")
    ax.plot(t_arr, obs_series, color="tab:orange", linewidth=1.0, linestyle="--",
            label="obs path on clean-only field (identical)")
    ax.set_title(
        "Panel 3 — Clean-only field (noisy_data=None)\n"
        "obs cache mirrors raw cache. Reward and obs queries return identical data.",
        fontsize=10,
    )
    ax.set_xlabel("time [s]", fontsize=9)
    ax.set_ylabel("signal value", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


# -----------------------------------------------------------------------------
# Panel 4 — shared dropout mask holds raw and noisy jointly
# -----------------------------------------------------------------------------
def panel4_shared_dropout(ax, num_envs: int, num_steps: int, dt: float,
                          noise_std: float, seed: int):
    """Under heavy i.i.d. dropout, the pipeline samples ONE mask per step
    and applies it to BOTH payloads plus the shared timestamp. Dropped
    steps freeze raw, noisy, and timestamp together. We plot the raw and
    noisy post-dropout outputs for one env and mark the drop events.

    This is the property that makes raycaster + replicator usable as a
    pair for bbox detection: a dropped packet drops both views and their
    shared timestamp together, never decoupling the reward path from the
    observation path.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    # Heavy dropout (60%) with minimal latency to keep the storyboard clean.
    pipe = _make_pipeline(
        num_envs, device, dt=dt,
        latency_mean=0.02, latency_std=0.005,  # ~1-step latency
        dropout_prob=0.6,
    )

    gt = np.zeros(num_steps)
    raw_series = np.zeros(num_steps)
    noisy_series = np.zeros(num_steps)
    ts_series = np.zeros(num_steps)

    env_i = 0
    for step in range(num_steps):
        val = _gt_signal(step, dt)
        gt[step] = val
        raw = torch.full((num_envs, 1), val, device=device)
        noisy = raw + torch.randn(num_envs, 1, device=device) * noise_std

        t = torch.full((num_envs,), step * dt, device=device)
        pipe.advance(raw, noisy, t.clone(), t.clone())
        raw_post, ts_post = pipe.query(use_noise=False, allow_dropout=True)
        noisy_post, _ = pipe.query(use_noise=True, allow_dropout=True)

        raw_series[step] = raw_post[env_i, 0].item()
        noisy_series[step] = noisy_post[env_i, 0].item()
        ts_series[step] = ts_post[env_i].item()

    t_arr = np.arange(num_steps) * dt
    # A "drop" for env_i is when the post-dropout timestamp did not advance.
    ts_diff = np.diff(np.concatenate([[ts_series[0] - dt], ts_series]))
    drop_steps = np.where(ts_diff < 1e-6)[0]

    ax.plot(t_arr, gt, color="black", linewidth=1.0, label="GT", alpha=0.5)
    ax.plot(t_arr, raw_series, color="tab:blue", linewidth=1.4,
            label="raw post-dropout (reward path)")
    ax.plot(t_arr, noisy_series, color="tab:red", linewidth=1.2, alpha=0.9,
            label="noisy post-dropout (obs path)")

    if len(drop_steps):
        ymin, ymax = ax.get_ylim() if ax.get_ylim()[0] != ax.get_ylim()[1] else (0.0, 1.0)
        ax.vlines(
            t_arr[drop_steps], ymin=min(raw_series.min(), noisy_series.min()) - 10,
            ymax=max(raw_series.max(), noisy_series.max()) + 10,
            color="tab:gray", alpha=0.25, linewidth=1.0,
            label="drop (held jointly)",
        )

    ax.set_title(
        "Panel 4 — Shared dropout mask\n"
        "one mask holds raw and noisy jointly; vertical lines = dropped steps",
        fontsize=10,
    )
    ax.set_xlabel("time [s]", fontsize=9)
    ax.set_ylabel("signal value (env 0)", fontsize=9)
    ax.grid(True, alpha=0.3)
    # Deduplicate legend (vlines may repeat the label per line)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in uniq], [l for _, l in uniq], loc="upper right", fontsize=8)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=5, help="Number of envs for Panel 1")
    parser.add_argument("--duration", type=float, default=2.5, help="Sim duration in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Sim step in seconds")
    parser.add_argument("--noise-std", type=float, default=15.0,
                        help="Gaussian noise std added to the obs payload")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "dual_cache_invariants.png",
    )
    args = parser.parse_args()

    num_steps = int(round(args.duration / args.dt))
    print(f"Simulating {args.duration}s ({num_steps} steps), dt={args.dt}s, "
          f"noise_std={args.noise_std}, seed={args.seed}")

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.0, 1.2],
        hspace=0.45, wspace=0.25,
    )
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[1, 0], sharex=ax1a)
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    panel1_shared_delay(ax1a, ax1b, args.envs, num_steps, args.dt,
                        args.noise_std, args.seed)
    panel2_reward_vs_obs(ax2, args.envs, num_steps, args.dt,
                         args.noise_std, args.seed + 1)
    panel3_clean_only(ax3, args.envs, num_steps, args.dt, args.seed + 2)
    panel4_shared_dropout(ax4, args.envs, num_steps, args.dt,
                          args.noise_std, args.seed + 3)

    fig.suptitle(
        "DelayPipelineV3 dual-cache invariants (ticket 029) — "
        f"{args.envs} envs, {args.duration:.1f}s @ {args.dt*1000:.0f}ms",
        fontsize=13,
    )

    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
