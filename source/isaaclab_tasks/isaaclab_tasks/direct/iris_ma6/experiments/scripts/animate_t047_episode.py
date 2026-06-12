#!/usr/bin/env python3
"""Animate one t047 episode from recorded trajectories (no Isaac Sim / no RTX).

RTX video capture cannot aim the camera in this headless Isaac Sim install, so we
render the episode purely from the trajectory JSON that evaluate.py writes with
--record-trajectory. Shows all three aircraft (drone_0, drone_1, target) in a 3D
scene with:
  - a quadrotor-style glyph per drone (body + 4 arms),
  - each drone's CAMERA FRUSTUM cone (apex at the drone, aimed along the line of
    sight to the tracked target, FOV width from the recorded zoom: base HFOV ~85deg
    scaled by 1/zoom — so zooming in narrows the cone),
  - fading position trails and velocity arrows,
  - a speed-vs-time strip with a moving cursor (makes jerky commands visible).

Usage:
    python animate_t047_episode.py [traj.json] [env_key] [out.mp4]
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "t047_dualstep")

TRAJ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "t047_traj_400k.json")
ENV_KEY = sys.argv[2] if len(sys.argv) > 2 else None
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(RESULTS_DIR, "t047_episode_3d.mp4")

AGENT_COLORS = {"drone_0": "#1f77b4", "drone_1": "#2ca02c"}
TARGET_COLOR = "#d62728"
TRAIL = 50
VEC_SCALE = 0.6
FPS = 25
BASE_HFOV_DEG = 85.0      # real-camera HFOV at zoom 1 (fx~1053 @ 1920px)
ASPECT = 16.0 / 9.0       # frustum width:height
ARM = 0.9                 # drone glyph arm half-length (m, visual only)


def load():
    with open(TRAJ) as f:
        tr = json.load(f)["trajectories"]
    envs = tr["envs"]
    key = ENV_KEY if (ENV_KEY in envs) else sorted(envs, key=int)[0]
    e = envs[key]
    aids = tr["agent_ids"]
    dt = tr.get("step_dt_seconds", 0.04)

    def arr(d, *k):
        return np.array([d[ki] for ki in k]).T

    def _bbox(d):
        n = len(d["x"])
        return {
            "u": np.array(d.get("bbox_u", [0.5] * n)),
            "v": np.array(d.get("bbox_v", [0.5] * n)),
            "valid": np.array(d.get("bbox_valid", [0.0] * n)),
        }

    agents = {a: {
        "p": arr(e["agents"][a], "x", "y", "z"),
        "v": arr(e["agents"][a], "vx", "vy", "vz"),
        "zoom": np.array(e["agents"][a].get("zoom", [1.0] * len(e["agents"][a]["x"]))),
        "bbox": _bbox(e["agents"][a]),
    } for a in aids}
    target = {"p": arr(e["target"], "x", "y", "z"), "v": arr(e["target"], "vx", "vy", "vz")}
    T = min([agents[a]["p"].shape[0] for a in aids] + [target["p"].shape[0]])
    return key, aids, dt, agents, target, T


def _basis(forward):
    f = forward / (np.linalg.norm(forward) + 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(f, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up); r /= (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f); u /= (np.linalg.norm(u) + 1e-9)
    return f, r, u


def draw_frustum(ax, apex, target_pt, zoom, color):
    """Frustum aimed apex->target, FOV from zoom. Returns nothing (draws lines)."""
    d = target_pt - apex
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        return
    f, r, u = _basis(d)
    hfov = np.deg2rad(BASE_HFOV_DEG) / max(zoom, 1e-3)
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) / ASPECT)
    tw, th = np.tan(hfov / 2.0) * L, np.tan(vfov / 2.0) * L
    center = apex + f * L
    corners = [center + sx * tw * r + sy * th * u
               for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    # apex->corner edges
    for c in corners:
        ax.plot(*zip(apex, c), color=color, lw=0.7, alpha=0.55)
    # far rectangle
    for i in range(4):
        ax.plot(*zip(corners[i], corners[(i + 1) % 4]), color=color, lw=0.7, alpha=0.55)


def draw_drone(ax, p, color):
    ax.scatter(*p, color=color, s=45, depthshade=False)
    for dx, dy in [(ARM, ARM), (ARM, -ARM), (-ARM, ARM), (-ARM, -ARM)]:
        ax.plot([p[0], p[0] + dx], [p[1], p[1] + dy], [p[2], p[2]], color=color, lw=1.3)


def draw_cam_panel(axc, aid, u, v, valid, color):
    """Draw one camera's image frame with the target's position in it (v down)."""
    axc.cla()
    axc.set_xlim(0, 1); axc.set_ylim(1, 0)  # image convention: v=0 at top
    axc.set_xticks([]); axc.set_yticks([])
    axc.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="0.35", lw=1.3))
    axc.axhline(0.5, color="0.8", lw=0.5); axc.axvline(0.5, color="0.8", lw=0.5)
    axc.add_patch(Rectangle((0.4, 0.4), 0.2, 0.2, fill=False, ec="0.8", lw=0.6, ls="--"))
    if valid > 0.5:
        axc.scatter([u], [v], s=90, color=color, edgecolor="white", linewidth=1.0, zorder=5)
        off = float(np.hypot(u - 0.5, v - 0.5))
        axc.set_title(f"{aid} camera  (off-center {off:.2f})", fontsize=8, color=color)
    else:
        axc.scatter([0.5], [0.5], s=70, facecolor="none", edgecolor="0.6", marker="x")
        axc.text(0.5, 0.12, "no detection", ha="center", fontsize=7, color="0.5")
        axc.set_title(f"{aid} camera", fontsize=8, color=color)


def main():
    key, aids, dt, agents, target, T = load()
    speed = {a: np.linalg.norm(agents[a]["v"], axis=1) for a in aids}
    tgt_speed = np.linalg.norm(target["v"], axis=1)
    t_s = np.arange(T) * dt

    allp = np.concatenate([agents[a]["p"][:T] for a in aids] + [target["p"][:T]], axis=0)
    ctr = allp.mean(axis=0)
    rng = (allp.max(axis=0) - allp.min(axis=0)).max() * 0.6 + 3.0

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.0, 0.28, 0.64, 0.70], projection="3d")
    # Two camera-image panels (right column): where the target sits in each image.
    cam_axes = {aids[0]: fig.add_axes([0.70, 0.60, 0.27, 0.27]),
                aids[1]: fig.add_axes([0.70, 0.30, 0.27, 0.27])} if len(aids) >= 2 \
        else {aids[0]: fig.add_axes([0.70, 0.45, 0.27, 0.27])}
    axt = fig.add_axes([0.06, 0.05, 0.58, 0.18])

    def set_scene():
        ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
        ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
        ax.set_zlim(max(0, ctr[2] - rng), ctr[2] + rng)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=20, azim=-58)

    for a in aids:
        axt.plot(t_s, speed[a], color=AGENT_COLORS[a], lw=0.9, label=f"{a} |v|")
    axt.plot(t_s, tgt_speed, color=TARGET_COLOR, lw=0.9, ls="--", label="target |v|")
    axt.set_xlim(0, t_s[-1]); axt.set_ylim(bottom=0)
    axt.set_xlabel("time (s)"); axt.set_ylabel("speed (m/s)")
    axt.legend(fontsize=6, ncol=3, loc="upper right"); axt.grid(alpha=0.3)
    cursor = axt.axvline(0, color="k", lw=1.0)

    fig.suptitle("t047 deployment policy (agent_400000 @ full difficulty) — one episode\n"
                 "drone glyph + camera frustum (FOV from zoom, aimed at tracked target)",
                 fontsize=9.5, y=0.99)

    def draw(i):
        ax.cla(); set_scene()
        lo = max(0, i - TRAIL)
        tp = target["p"]
        for a in aids:
            p = agents[a]["p"]; v = agents[a]["v"]
            ax.plot(p[lo:i+1, 0], p[lo:i+1, 1], p[lo:i+1, 2], color=AGENT_COLORS[a], lw=1.3, alpha=0.8)
            draw_drone(ax, p[i], AGENT_COLORS[a])
            draw_frustum(ax, p[i], tp[i], float(agents[a]["zoom"][i]), AGENT_COLORS[a])
            ax.quiver(p[i, 0], p[i, 1], p[i, 2], v[i, 0]*VEC_SCALE, v[i, 1]*VEC_SCALE, v[i, 2]*VEC_SCALE,
                      color=AGENT_COLORS[a], lw=1.5, arrow_length_ratio=0.3)
        ax.plot(tp[lo:i+1, 0], tp[lo:i+1, 1], tp[lo:i+1, 2], color=TARGET_COLOR, lw=1.3, alpha=0.8)
        ax.scatter(tp[i, 0], tp[i, 1], tp[i, 2], color=TARGET_COLOR, s=90, marker="*", depthshade=False)
        ax.text2D(0.02, 0.93, f"env {key}   t = {i*dt:5.2f} s   step {i}/{T-1}",
                  transform=ax.transAxes, fontsize=9,
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))

        for a in aids:
            bb = agents[a]["bbox"]
            draw_cam_panel(cam_axes[a], a, float(bb["u"][i]), float(bb["v"][i]),
                           float(bb["valid"][i]), AGENT_COLORS[a])
        cursor.set_xdata([i*dt, i*dt])
        return []

    print(f"Rendering {T} frames -> {OUT} ...")
    anim = FuncAnimation(fig, draw, frames=T, interval=1000/FPS, blit=False)
    anim.save(OUT, writer=FFMpegWriter(fps=FPS, bitrate=4000, metadata={"title": "t047 episode"}))
    plt.close()
    print(f"Saved: {OUT}  ({T} frames, {T*dt:.1f}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
