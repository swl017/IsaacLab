#!/usr/bin/env python3
"""Animate one t047 episode from recorded trajectories (no Isaac Sim / no RTX).

RTX video capture is broken on this box (warp 1.8.0 vs the Isaac Sim bundled
replicator: `wp.types.array() got an unexpected keyword argument 'owner'`), so
both the custom VideoRecorder and the standard gymnasium RecordVideo path fail.
This renders the episode purely from the trajectory JSON that evaluate.py writes
with --record-trajectory — the same data-driven approach used for final.mp4.

Shows all three aircraft (drone_0, drone_1, target) in a 3D scene with fading
trails and current velocity arrows, plus a speed-vs-time inset with a moving
cursor so jerky velocity commands are visible.

Usage:
    python animate_t047_episode.py [traj.json] [env_key] [out.mp4]
      traj.json : default outputs/t047_dualstep/t047_traj_400k.json
      env_key   : which sampled env to animate (default: first)
      out.mp4   : default outputs/t047_dualstep/t047_episode_3d.mp4
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "t047_dualstep")

TRAJ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "t047_traj_400k.json")
ENV_KEY = sys.argv[2] if len(sys.argv) > 2 else None
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(RESULTS_DIR, "t047_episode_3d.mp4")

AGENT_COLORS = {"drone_0": "#1f77b4", "drone_1": "#2ca02c"}
TARGET_COLOR = "#d62728"
TRAIL = 50          # trailing points
VEC_SCALE = 0.6     # velocity arrow length scale (m per (m/s))
FPS = 25            # 1/0.04 -> real time


def load():
    with open(TRAJ) as f:
        tr = json.load(f)["trajectories"]
    envs = tr["envs"]
    key = ENV_KEY if ENV_KEY in envs else sorted(envs, key=int)[0]
    e = envs[key]
    aids = tr["agent_ids"]
    dt = tr.get("step_dt_seconds", 0.04)

    def xyz(d):
        return np.array([d["x"], d["y"], d["z"]]).T  # (T,3)

    def vel(d):
        return np.array([d["vx"], d["vy"], d["vz"]]).T

    agents = {a: {"p": xyz(e["agents"][a]), "v": vel(e["agents"][a])} for a in aids}
    target = {"p": xyz(e["target"]), "v": vel(e["target"])}
    T = min([agents[a]["p"].shape[0] for a in aids] + [target["p"].shape[0]])
    return key, aids, dt, agents, target, T


def main():
    key, aids, dt, agents, target, T = load()
    speed = {a: np.linalg.norm(agents[a]["v"], axis=1) for a in aids}
    tgt_speed = np.linalg.norm(target["v"], axis=1)
    t_s = np.arange(T) * dt

    # Axis bounds across all entities + margin, with equal aspect.
    allp = np.concatenate([agents[a]["p"][:T] for a in aids] + [target["p"][:T]], axis=0)
    ctr = allp.mean(axis=0)
    rng = (allp.max(axis=0) - allp.min(axis=0)).max() * 0.6 + 2.0

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.0, 0.30, 0.72, 0.68], projection="3d")
    axs = fig.add_axes([0.78, 0.12, 0.20, 0.78])   # speed inset
    axt = fig.add_axes([0.06, 0.06, 0.62, 0.18])   # speed-vs-time strip

    def set_scene():
        ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
        ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
        ax.set_zlim(max(0, ctr[2] - rng), ctr[2] + rng)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=-58)

    # speed-vs-time strip (static curves + moving cursor)
    for a in aids:
        axt.plot(t_s, speed[a], color=AGENT_COLORS[a], lw=0.9, label=f"{a} |v|")
    axt.plot(t_s, tgt_speed, color=TARGET_COLOR, lw=0.9, ls="--", label="target |v|")
    axt.set_xlim(0, t_s[-1]); axt.set_ylim(bottom=0)
    axt.set_xlabel("time (s)"); axt.set_ylabel("speed (m/s)")
    axt.legend(fontsize=6, ncol=3, loc="upper right")
    axt.grid(alpha=0.3)
    cursor = axt.axvline(0, color="k", lw=1.0)

    fig.suptitle("t047 deployment policy (agent_400000 @ full difficulty) — one episode",
                 fontsize=11, y=0.99)

    def draw(i):
        ax.cla(); set_scene()
        lo = max(0, i - TRAIL)
        for a in aids:
            p = agents[a]["p"]; v = agents[a]["v"]
            ax.plot(p[lo:i+1, 0], p[lo:i+1, 1], p[lo:i+1, 2], color=AGENT_COLORS[a], lw=1.4, alpha=0.8)
            ax.scatter(p[i, 0], p[i, 1], p[i, 2], color=AGENT_COLORS[a], s=55, depthshade=False)
            ax.quiver(p[i, 0], p[i, 1], p[i, 2], v[i, 0]*VEC_SCALE, v[i, 1]*VEC_SCALE, v[i, 2]*VEC_SCALE,
                      color=AGENT_COLORS[a], lw=1.6, arrow_length_ratio=0.3)
            # line of sight to target
            ax.plot([p[i, 0], target["p"][i, 0]], [p[i, 1], target["p"][i, 1]],
                    [p[i, 2], target["p"][i, 2]], color=AGENT_COLORS[a], lw=0.5, ls=":", alpha=0.5)
        tp = target["p"]
        ax.plot(tp[lo:i+1, 0], tp[lo:i+1, 1], tp[lo:i+1, 2], color=TARGET_COLOR, lw=1.4, alpha=0.8)
        ax.scatter(tp[i, 0], tp[i, 1], tp[i, 2], color=TARGET_COLOR, s=80, marker="*", depthshade=False)
        ax.set_title(f"env {key}   t = {i*dt:5.2f} s   step {i}/{T-1}", fontsize=9)

        # instantaneous speed bars
        axs.cla()
        names = aids + ["target"]
        vals = [speed[a][i] for a in aids] + [tgt_speed[i]]
        cols = [AGENT_COLORS[a] for a in aids] + [TARGET_COLOR]
        axs.bar(range(len(names)), vals, color=cols)
        axs.set_xticks(range(len(names))); axs.set_xticklabels(names, rotation=20, fontsize=7)
        axs.set_ylim(0, max(1.0, max(np.max(speed[a]) for a in aids)) * 1.1)
        axs.set_ylabel("|v| (m/s)", fontsize=8); axs.set_title("instant speed", fontsize=8)
        axs.grid(alpha=0.3, axis="y")

        cursor.set_xdata([i*dt, i*dt])
        return []

    print(f"Rendering {T} frames -> {OUT} ...")
    anim = FuncAnimation(fig, draw, frames=T, interval=1000/FPS, blit=False)
    writer = FFMpegWriter(fps=FPS, bitrate=4000, metadata={"title": "t047 episode"})
    anim.save(OUT, writer=writer)
    plt.close()
    print(f"Saved: {OUT}  ({T} frames, {T*dt:.1f}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
