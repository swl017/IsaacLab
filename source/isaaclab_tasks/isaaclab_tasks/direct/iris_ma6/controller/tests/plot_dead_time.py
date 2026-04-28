#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate visualisation plots for mas/036 dead-time buffer.

Three figures:
  1. step_response.png  — cmd / dead-time-buffer output / lag output for a
     single env at u=0.5 with mean dead time = 70 ms. Annotated with the
     analytical step response of the first-order model the rate loop is
     reproducing, dead time included.
  2. distribution.png   — 10k empirical dead-time samples at scale=1.0 vs
     the per-axis bench histograms (yaw + pitch from rate_step_summary.csv).
     Configured Gaussian PDF overlaid.
  3. curriculum.png     — empirical sample mean / std vs curriculum scale
     in [0, 1] (linear scaling property mas/036 promises).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="mas/036 dead-time plots")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop import GimbalRateLoop
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop_cfg import GimbalRateLoopCfg


PLOTS_DIR = Path(__file__).parent / "plots_mas036"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RATE_STEP_CSV = Path(
    "/home/usrg/mas/src/gimbal_controller/scripts/"
    "gimbal_rate_step_followspeed_tune/rate_step_summary.csv"
)


def load_measured_latencies():
    yaw, pitch = [], []
    with RATE_STEP_CSV.open() as f:
        for row in csv.DictReader(f):
            t = float(row["latency_s"])
            (yaw if row["axis"] == "yaw" else pitch).append(t)
    return yaw, pitch


# ---------------------------------------------------------------- plot 1


def plot_step_response(device: torch.device):
    """A single-env trace at u=0.5 (cmd ≈ 0.64 rad/s), with mean dead time
    = 70 ms (std=0 → deterministic). Three traces:
      - command (instant step at t=0)
      - rate-loop input post-buffer (zero for the first d steps, then jumps)
      - rate-loop output (post first-order lag)
      - analytical reference: ω(t) = ω_ss · (1 − exp(−(t − Δ)/τ)) for t ≥ Δ
    """
    dt = 0.005   # finer than physics dt to make the curve smoother
    duration = 0.5
    cmd_radps = 0.5  # well inside ±1.28 saturation
    mean_s = 0.07
    tau = 0.0995  # yaw default

    cfg = GimbalRateLoopCfg(
        dead_time_mean_s=mean_s,
        dead_time_std_s=0.0,
        dead_time_max_s=0.15,
    )
    rl = GimbalRateLoop(cfg, num_envs=1, device=device)
    rl.set_progress(1.0)
    rl.set_dead_time_curriculum_scale(1.0)
    rl.reset()

    n = int(round(duration / dt))
    t = np.arange(n) * dt
    cmd_trace = np.full(n, cmd_radps)
    out_trace = np.zeros(n)
    buf_trace = np.zeros(n)  # pre-lag, post-buffer trace

    cmd_t = torch.tensor([[cmd_radps, 0.0]], device=device)
    for i in range(n):
        # peek the dead-time buffer's *output* by stepping the loop and
        # also reconstructing the post-buffer signal: omega cycles through
        # zero (cold/dead) → step at the dead-time depth.
        d_steps = int(rl.dead_time_steps[0].item())
        if (i + 1) > d_steps:
            buf_trace[i] = cmd_radps
        else:
            buf_trace[i] = 0.0
        out = rl.step(cmd_t, dt)
        out_trace[i] = out[0, 0].item()

    # Analytical step-response reference: a single-stage first-order lag
    # delayed by the dead time, asymptote = cmd.
    delta = mean_s
    analytical = np.where(
        t < delta, 0.0, cmd_radps * (1.0 - np.exp(-(t - delta) / tau))
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="grey", lw=0.5, alpha=0.4)
    ax.plot(t * 1000, cmd_trace, color="black", lw=1.2, alpha=0.6,
            label=f"command (step to {cmd_radps:.2f} rad/s)")
    ax.plot(t * 1000, buf_trace, color="tab:orange", lw=1.5,
            label=f"post-dead-time-buffer (Δ = {mean_s*1000:.0f} ms)")
    ax.plot(t * 1000, out_trace, color="tab:blue", lw=2.0,
            label=f"rate-loop output (sim, τ_yaw = {tau*1000:.1f} ms)")
    ax.plot(t * 1000, analytical, color="tab:red", lw=1.4, ls="--",
            label="analytical: ω_ss · (1 − e^(−(t−Δ)/τ))")
    ax.axvline(mean_s * 1000, color="grey", ls=":", lw=0.8)
    ax.text(mean_s * 1000 + 2, cmd_radps * 0.05,
            f"Δ = {mean_s*1000:.0f} ms", fontsize=9, color="grey")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("yaw rate (rad/s)")
    ax.set_title(
        f"mas/036 step response: dead time + first-order lag\n"
        f"u = 0.5  →  ω_cmd = {cmd_radps:.2f} rad/s,  Δ = {mean_s*1000:.0f} ms,  τ = {tau*1000:.1f} ms"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    out = PLOTS_DIR / "step_response.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- plot 2


def plot_distribution(device: torch.device):
    """Empirical 10k-sample distribution at scale=1 vs measured bench
    histograms vs configured Gaussian PDF."""
    cfg = GimbalRateLoopCfg()  # uses fit defaults
    n_samples = 10_000
    rl = GimbalRateLoop(cfg, num_envs=n_samples, device=device)
    rl.set_dead_time_curriculum_scale(1.0)
    rl.reset()
    sim_samples = rl.dead_time_seconds.cpu().numpy() * 1000  # ms

    yaw_meas, pitch_meas = load_measured_latencies()
    yaw_meas_ms = np.array(yaw_meas) * 1000
    pitch_meas_ms = np.array(pitch_meas) * 1000

    fig, ax = plt.subplots(figsize=(9, 5))

    # Empirical sim samples
    ax.hist(
        sim_samples, bins=60, range=(0, 130), density=True,
        color="tab:blue", alpha=0.55,
        label=f"sim (10k env-resets, scale=1.0)  mean={sim_samples.mean():.1f} ms",
    )

    # Bench measurements (rugplot style at the bottom)
    ax.scatter(
        yaw_meas_ms, np.full_like(yaw_meas_ms, -0.0015),
        color="tab:orange", s=40, marker="|",
        label=f"bench yaw (n={len(yaw_meas_ms)})  mean={yaw_meas_ms.mean():.1f} ms",
    )
    ax.scatter(
        pitch_meas_ms, np.full_like(pitch_meas_ms, -0.003),
        color="tab:green", s=40, marker="|",
        label=f"bench pitch (n={len(pitch_meas_ms)})  mean={pitch_meas_ms.mean():.1f} ms",
    )

    # Configured Gaussian PDF (clipped at 0 and dead_time_max_s)
    x = np.linspace(0, 130, 400)
    mean_ms = cfg.dead_time_mean_s * 1000
    std_ms = cfg.dead_time_std_s * 1000
    pdf = np.exp(-0.5 * ((x - mean_ms) / std_ms) ** 2) / (std_ms * math.sqrt(2 * math.pi))
    pdf[x > cfg.dead_time_max_s * 1000] = 0.0
    ax.plot(x, pdf, color="tab:red", lw=1.6, ls="--",
            label=f"configured N({mean_ms:.0f}, {std_ms:.0f}²) ms")

    ax.set_xlabel("dead time (ms)")
    ax.set_ylabel("density")
    ax.set_title(
        "mas/036 dead-time distribution: 10k env-samples vs bench\n"
        f"cfg defaults: mean={mean_ms:.0f} ms, std={std_ms:.0f} ms, max={cfg.dead_time_max_s*1000:.0f} ms"
    )
    ax.set_xlim(0, 130)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    out = PLOTS_DIR / "distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- plot 3


def plot_curriculum_scaling(device: torch.device):
    """Empirical mean and std vs curriculum scale ∈ [0, 1]. Both should
    scale linearly (mean scales, std scales)."""
    cfg = GimbalRateLoopCfg()
    n = 5_000
    rl = GimbalRateLoop(cfg, num_envs=n, device=device)

    scales = np.linspace(0.0, 1.0, 11)
    means = np.zeros_like(scales)
    stds = np.zeros_like(scales)
    for i, s in enumerate(scales):
        rl.set_dead_time_curriculum_scale(float(s))
        rl.reset()
        means[i] = rl.dead_time_seconds.mean().item() * 1000
        stds[i] = rl.dead_time_seconds.std(unbiased=False).item() * 1000

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        scales, means, yerr=stds, fmt="o-", color="tab:blue",
        capsize=4, ms=7, lw=1.8,
        label="empirical mean ± std (n=5k samples per scale)",
    )
    # Linear-prediction reference
    pred_mean = scales * cfg.dead_time_mean_s * 1000
    pred_std = scales * cfg.dead_time_std_s * 1000
    ax.plot(scales, pred_mean, color="tab:red", ls="--", lw=1.4,
            label=f"prediction: scale × {cfg.dead_time_mean_s*1000:.0f} ms")
    ax.fill_between(
        scales, pred_mean - pred_std, pred_mean + pred_std,
        color="tab:red", alpha=0.12, label="prediction: ± scale × std",
    )

    ax.set_xlabel("curriculum scale")
    ax.set_ylabel("dead time (ms)")
    ax.set_title(
        "mas/036 curriculum scaling: mean and std are linear in `scale`\n"
        "(empirical at 11 points, n=5k each, vs scale × cfg)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    out = PLOTS_DIR / "curriculum.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    print("=" * 80)
    print("mas/036 dead-time plotting")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(0)
    plot_step_response(device)
    plot_distribution(device)
    plot_curriculum_scaling(device)
    print(f"\nAll plots written to {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    main()
