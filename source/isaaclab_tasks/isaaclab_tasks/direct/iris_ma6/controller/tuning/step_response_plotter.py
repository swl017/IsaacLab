"""Step response plotting for controller auto-tuning.

Generates multi-panel PDF figures showing velocity, attitude error, and rate error
time series for each tuning trial. Used to visually inspect oscillation, overshoot,
and settling behavior across different parameter sets.

Usage:
    Called from auto_tune.py after tuning completes. Not intended to be run standalone.
"""

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import numpy as np


class StepResponsePlotter:
    """Generates step response PDF plots for tuning trials."""

    def __init__(self, output_dir: Path):
        """Initialize plotter.

        Args:
            output_dir: Directory to save PDF files (e.g. tuning_results/step_responses/).
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_trial(
        self,
        trial_idx: int,
        params,
        metrics,
        histories: dict,
        dt: float,
    ) -> Path:
        """Generate a multi-panel step response PDF for one trial.

        Args:
            trial_idx: Trial index (for filename).
            params: ParameterSet for this trial.
            metrics: TuningMetrics for this trial.
            histories: Dict with keys:
                - vel_5: {vel_history (T,), att_error_history (T, 3), rate_error_history (T, 3)}
                - vel_10: same structure
                - att: {att_errors (T,), rate_error_history (T, 3)}
            dt: Policy timestep [s].

        Returns:
            Path to the saved PDF.
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(
            f"Trial {trial_idx}  |  Score: {metrics.score:.2f}  |  "
            f"Kp_vel_xy={params.Kp_vel_xy:.2f}  Kp_att_rp={params.Kp_att_rp:.2f}  "
            f"Kp_rate_rp={params.Kp_rate_rp:.3f}",
            fontsize=11,
        )

        # --- Row 0: Velocity step responses ---
        for col, (key, target, label) in enumerate([
            ("vel_5", 5.0, "5 m/s"),
            ("vel_10", 10.0, "10 m/s"),
        ]):
            ax = axes[0, col]
            data = histories[key]
            vel = data["vel_history"].cpu().numpy()
            t = np.arange(len(vel)) * dt

            ax.plot(t, vel, "b-", linewidth=1.0, label="Actual")
            ax.axhline(y=target, color="r", linestyle="--", linewidth=0.8, label="Target")
            ax.axhline(y=0.95 * target, color="gray", linestyle=":", linewidth=0.5, label="95%")

            # Mark settling time
            osc = getattr(metrics, f"vel_osc_{int(target)}")
            ax.set_title(
                f"Velocity {label}  |  ζ={osc.damping_ratio:.3f}  ZC={osc.zero_crossings}  "
                f"amp={osc.ss_amplitude:.3f}  f={osc.frequency:.1f}Hz",
                fontsize=9,
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Velocity [m/s]")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)

        # --- Row 1: Attitude error ---
        for col, (key, label) in enumerate([
            ("vel_5", "During 5 m/s step"),
            ("att", "Attitude recovery"),
        ]):
            ax = axes[1, col]
            data = histories[key]

            if key == "att":
                # Scalar attitude error in degrees
                err = data["att_errors"].cpu().numpy()
                t = np.arange(len(err)) * dt
                ax.plot(t, err, "b-", linewidth=1.0)
                ax.set_ylabel("Attitude error [deg]")
                osc = metrics.att_osc
            else:
                # 3D attitude error in radians → degrees
                err_3d = data["att_error_history"].cpu().numpy()
                t = np.arange(len(err_3d)) * dt
                labels = ["Roll", "Pitch", "Yaw"]
                colors = ["r", "g", "b"]
                for i, (lbl, clr) in enumerate(zip(labels, colors)):
                    ax.plot(t, np.rad2deg(err_3d[:, i]), color=clr, linewidth=0.8, label=lbl)
                ax.set_ylabel("Attitude error [deg]")
                ax.legend(fontsize=7)
                osc = metrics.vel_osc_5

            ax.set_title(
                f"Attitude error — {label}  |  ζ={osc.damping_ratio:.3f}  ZC={osc.zero_crossings}",
                fontsize=9,
            )
            ax.set_xlabel("Time [s]")
            ax.grid(True, alpha=0.3)

        # --- Row 2: Rate error ---
        for col, (key, rate_key, label) in enumerate([
            ("vel_5", "rate_osc_vel5", "During 5 m/s step"),
            ("att", "rate_osc_att", "Attitude recovery"),
        ]):
            ax = axes[2, col]
            data = histories[key]
            rate_err = data["rate_error_history"].cpu().numpy()
            t = np.arange(len(rate_err)) * dt

            labels = ["Roll", "Pitch", "Yaw"]
            colors = ["r", "g", "b"]
            for i, (lbl, clr) in enumerate(zip(labels, colors)):
                ax.plot(t, np.rad2deg(rate_err[:, i]), color=clr, linewidth=0.8, label=lbl)

            osc = getattr(metrics, rate_key)
            ax.set_title(
                f"Rate error — {label}  |  ζ={osc.damping_ratio:.3f}  ZC={osc.zero_crossings}  "
                f"amp={osc.ss_amplitude:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Rate error [deg/s]")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = self.output_dir / f"trial_{trial_idx:04d}.pdf"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def plot_top_k(
        self,
        ranked_results: list,
        k: int,
        dt: float,
    ) -> list:
        """Plot step responses for the top-K results.

        Args:
            ranked_results: List of dicts with keys "trial_idx", "params", "metrics",
                "histories". Must already be sorted by score (best first).
            k: Number of top results to plot.
            dt: Policy timestep [s].

        Returns:
            List of Paths to saved PDFs.
        """
        paths = []
        n = min(k, len(ranked_results))
        print(f"\nGenerating step response plots for top {n} results...")

        for rank, entry in enumerate(ranked_results[:n]):
            path = self.plot_trial(
                trial_idx=entry["trial_idx"],
                params=entry["params"],
                metrics=entry["metrics"],
                histories=entry["histories"],
                dt=dt,
            )
            paths.append(path)
            print(f"  [{rank+1}/{n}] Trial {entry['trial_idx']} "
                  f"(score={entry['metrics'].score:.3f}) → {path.name}")

        print(f"Step response plots saved to: {self.output_dir}")
        return paths
