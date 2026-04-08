#!/usr/bin/env python3
"""Plot the detector replicator miss rate and FP model from current NoiseModelParams defaults.

Usage:
    python plot_detector_model.py [--save detector_model.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

# ── Current NoiseModelParams defaults (from detector_replicator.py) ──────────

# Localization noise: scale = a / target_size + b
center_noise_a = 3.32094098449108e-08
center_noise_b = 0.9903605416833782
center_noise_df = 3.14485116541261

size_noise_a = 52.57566163317716
size_noise_b = 1.0712118528882864
size_noise_df = 7.728841290635469

center_bias_x = -0.236480712890625
center_bias_y = -0.71551513671875

# Miss rate: dual sigmoids
miss_sigmoid_a_sky = 0.024751038174665885
miss_size_threshold_sky = 1.2214178344000452e-12
miss_size_cutoff_sky = 33.0

miss_sigmoid_a_gnd = 0.013808565449099726
miss_size_threshold_gnd = 45.56221154006872
miss_size_cutoff_gnd = 60.0

# Tuned multipliers (matching _apply_miss code)
sky_multiplier = 1.0
gnd_multiplier = 2.0

# FP
fp_rate = 0.03469779341221618
fp_size_range = (10.0, 60.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description="Plot detector replicator model.")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path.")
    args = parser.parse_args()

    s = np.linspace(1, 200, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Detector Replicator Model (NoiseModelParams defaults)", fontsize=13, fontweight="bold")

    # ── Panel 1: Miss rate vs target size ─────────────────────────────────
    ax = axes[0, 0]

    # Sky sigmoid (with multiplier and cutoff)
    p_miss_sky_raw = sky_multiplier * sigmoid(miss_sigmoid_a_sky * (miss_size_threshold_sky - s))
    p_miss_sky = np.where(s < miss_size_cutoff_sky, 1.0, np.minimum(p_miss_sky_raw, 1.0))

    # Ground sigmoid (with multiplier and cutoff)
    p_miss_gnd_raw = gnd_multiplier * sigmoid(miss_sigmoid_a_gnd * (miss_size_threshold_gnd - s))
    p_miss_gnd = np.where(s < miss_size_cutoff_gnd, 1.0, np.minimum(p_miss_gnd_raw, 1.0))

    ax.plot(s, p_miss_sky, color="deepskyblue", lw=2.5, label=f"Sky ({sky_multiplier}x sigmoid, cutoff={miss_size_cutoff_sky}px)")
    ax.plot(s, p_miss_gnd, color="saddlebrown", lw=2.5, label=f"Ground ({gnd_multiplier}x sigmoid, cutoff={miss_size_cutoff_gnd}px)")

    # Mark cutoffs
    ax.axvline(miss_size_cutoff_sky, color="deepskyblue", ls=":", alpha=0.5)
    ax.axvline(miss_size_cutoff_gnd, color="saddlebrown", ls=":", alpha=0.5)

    # Mark sigmoid thresholds
    if miss_size_threshold_gnd > 1:
        ax.axvline(miss_size_threshold_gnd, color="saddlebrown", ls="--", alpha=0.3,
                   label=f"Gnd threshold={miss_size_threshold_gnd:.1f}px")

    ax.set_xlabel("Target size (px)")
    ax.set_ylabel("Miss probability (before curriculum)")
    ax.set_title("Miss Rate Model")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Miss rate at different curriculum scales ──────────────────
    ax = axes[0, 1]
    fp_fn_scales = [0.01, 0.1, 0.25, 0.5, 1.0]
    colors_curr = plt.cm.viridis(np.linspace(0.2, 0.9, len(fp_fn_scales)))

    for scale, color in zip(fp_fn_scales, colors_curr):
        p_gnd_curr = p_miss_gnd * scale
        ax.plot(s, p_gnd_curr, color=color, lw=1.5, label=f"fp_fn_scale={scale}")

    ax.set_xlabel("Target size (px)")
    ax.set_ylabel("Effective miss probability (ground)")
    ax.set_title("Ground Miss Rate vs Curriculum Scale")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Noise scale vs target size ───────────────────────────────
    ax = axes[1, 0]
    center_scale = center_noise_a / np.maximum(s, 5.0) + center_noise_b
    size_scale = size_noise_a / np.maximum(s, 5.0) + size_noise_b

    ax.plot(s, center_scale, color="royalblue", lw=2, label=f"Center noise (a={center_noise_a:.2e}, b={center_noise_b:.2f}, df={center_noise_df:.1f})")
    ax.plot(s, size_scale, color="darkorange", lw=2, label=f"Size noise (a={size_noise_a:.1f}, b={size_noise_b:.2f}, df={size_noise_df:.1f})")

    ax.set_xlabel("Target size (px)")
    ax.set_ylabel("Noise scale (px)")
    ax.set_title("Localization Noise Scale (Student's t)")
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add Student's t PDF inset
    inset = ax.inset_axes([0.55, 0.45, 0.4, 0.45])
    x_pdf = np.linspace(-5, 5, 200)
    inset.plot(x_pdf, student_t.pdf(x_pdf, center_noise_df), color="royalblue", lw=1.5,
               label=f"Center (df={center_noise_df:.1f})")
    inset.plot(x_pdf, student_t.pdf(x_pdf, size_noise_df), color="darkorange", lw=1.5,
               label=f"Size (df={size_noise_df:.1f})")
    from scipy.stats import norm
    inset.plot(x_pdf, norm.pdf(x_pdf), color="gray", lw=1, ls="--", alpha=0.5, label="Gaussian")
    inset.set_title("Student's t PDF (unit scale)", fontsize=7)
    inset.legend(fontsize=6)
    inset.set_yticks([])

    # ── Panel 4: FP model summary ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    fp_size_min, fp_size_max = fp_size_range
    summary = (
        f"False Positive Model\n"
        f"{'─' * 40}\n"
        f"FP rate:     {fp_rate:.4f} per camera per step\n"
        f"FP size:     U({fp_size_min:.0f}, {fp_size_max:.0f}) px  (w and h independent)\n"
        f"FP position: U(0, img_w) x U(0, img_h)\n"
        f"Curriculum:  p_fp = fp_rate × fp_fn_scale\n"
        f"\n"
        f"At fp_fn_scale=1.0:\n"
        f"  ~{fp_rate * 100:.1f}% chance per camera per step\n"
        f"  ~{fp_rate * 25:.1f} FPs per second (at 25 Hz)\n"
        f"\n"
        f"{'─' * 40}\n"
        f"Systematic Bias\n"
        f"{'─' * 40}\n"
        f"Center bias X: {center_bias_x:.3f} px\n"
        f"Center bias Y: {center_bias_y:.3f} px\n"
        f"\n"
        f"{'─' * 40}\n"
        f"Noise Correlation\n"
        f"{'─' * 40}\n"
        f"Width/height noise: correlated\n"
        f"  (single Student's t draw for both)\n"
        f"Center X/Y noise: independent"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
