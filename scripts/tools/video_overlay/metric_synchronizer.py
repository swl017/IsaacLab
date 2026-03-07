# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synchronize video frames with simulation timesteps."""

from __future__ import annotations

import math
from typing import Any


class MetricSynchronizer:
    """Synchronize video frames with simulation timesteps.

    Handles the mapping between video frame indices and simulation timesteps,
    accounting for different frame rates between the video and simulation.

    Example:
        # Video at 30 FPS, simulation at 60 FPS (step_dt = 0.01667)
        sync = MetricSynchronizer(
            video_fps=30.0,
            step_dt_seconds=0.01667,
            num_timesteps=2500,
        )

        # Get metrics for video frame 100
        metrics = sync.get_metrics_for_frame_timeseries(100, timeseries_data)
    """

    def __init__(
        self,
        video_fps: float,
        step_dt_seconds: float,
        num_timesteps: int,
    ):
        """Initialize the metric synchronizer.

        Args:
            video_fps: Video frame rate in frames per second.
            step_dt_seconds: Simulation time step in seconds.
            num_timesteps: Total number of timesteps in the data.
        """
        self.video_fps = video_fps
        self.step_dt_seconds = step_dt_seconds
        self.sim_fps = 1.0 / step_dt_seconds
        self.num_timesteps = num_timesteps

        # Ratio of simulation steps per video frame
        self.frame_to_step_ratio = self.sim_fps / self.video_fps

    def get_timestep_for_frame(self, frame_idx: int) -> int:
        """Map video frame index to simulation timestep.

        Args:
            frame_idx: Video frame index (0-indexed).

        Returns:
            Corresponding simulation timestep index.
        """
        timestep = int(frame_idx * self.frame_to_step_ratio)
        return min(timestep, self.num_timesteps - 1)

    def get_sim_time_for_frame(self, frame_idx: int) -> float:
        """Get simulation time for a video frame.

        Args:
            frame_idx: Video frame index (0-indexed).

        Returns:
            Simulation time in seconds.
        """
        timestep = self.get_timestep_for_frame(frame_idx)
        return timestep * self.step_dt_seconds

    def get_metrics_for_frame_timeseries(
        self,
        frame_idx: int,
        timeseries: dict[str, Any],
    ) -> dict[str, float]:
        """Extract metrics for a given video frame from timeseries data.

        Args:
            frame_idx: Video frame index.
            timeseries: Timeseries data dict with structure:
                {
                    "viewing_angle": {"mean": [...], "std": [...]},
                    "triangulation_rmse": {"mean": [...], "std": [...]},
                    ...
                }

        Returns:
            Dictionary of metric values for this frame.
        """
        ts = self.get_timestep_for_frame(frame_idx)

        metrics = {
            "timestep": ts,
            "sim_time": ts * self.step_dt_seconds,
        }

        # Extract each metric's mean value at this timestep
        metric_keys = [
            "viewing_angle",
            "triangulation_rmse",
            "visibility",
            "tri_valid",
            "sqrt_trace_sigma",
            "distance_to_target",
            "target_speed",
        ]

        for key in metric_keys:
            if key in timeseries:
                data = timeseries[key]
                if isinstance(data, dict) and "mean" in data:
                    means = data["mean"]
                    if ts < len(means):
                        metrics[key] = means[ts]
                    else:
                        metrics[key] = None
                elif isinstance(data, list):
                    if ts < len(data):
                        metrics[key] = data[ts]
                    else:
                        metrics[key] = None
            else:
                metrics[key] = None

        return metrics

    def get_metrics_for_frame_trajectory(
        self,
        frame_idx: int,
        traj_data: dict[str, Any],
    ) -> dict[str, float]:
        """Extract metrics for a given video frame from trajectory data.

        Args:
            frame_idx: Video frame index.
            traj_data: Trajectory data dict for a single environment with structure:
                {
                    "viewing_angle": [...],
                    "rmse": [...],
                    "tri_valid": [...],
                    ...
                }

        Returns:
            Dictionary of metric values for this frame.
        """
        ts = self.get_timestep_for_frame(frame_idx)
        num_steps = traj_data.get("num_steps", len(traj_data.get("viewing_angle", [])))

        if ts >= num_steps:
            ts = num_steps - 1

        metrics = {
            "timestep": ts,
            "sim_time": ts * self.step_dt_seconds,
        }

        # Extract viewing angle
        if "viewing_angle" in traj_data and ts < len(traj_data["viewing_angle"]):
            metrics["viewing_angle"] = traj_data["viewing_angle"][ts]
        else:
            metrics["viewing_angle"] = None

        # Extract RMSE (may be null for invalid triangulation)
        if "rmse" in traj_data and ts < len(traj_data["rmse"]):
            metrics["triangulation_rmse"] = traj_data["rmse"][ts]
        else:
            metrics["triangulation_rmse"] = None

        # Extract tri_valid
        if "tri_valid" in traj_data and ts < len(traj_data["tri_valid"]):
            val = traj_data["tri_valid"][ts]
            metrics["tri_valid"] = 1.0 if val else 0.0
        else:
            metrics["tri_valid"] = None

        # Compute visibility from agent detection status if available
        # (Not directly stored in trajectory data, default to 1.0 if tri_valid)
        if metrics["tri_valid"] is not None:
            metrics["visibility"] = metrics["tri_valid"]
        else:
            metrics["visibility"] = None

        return metrics

    def get_graph_window(
        self,
        frame_idx: int,
        timeseries: dict[str, Any],
        metric_keys: list[str],
        window_size: int = 100,
    ) -> dict[str, list[float]]:
        """Get a window of metric history for animated graph.

        Args:
            frame_idx: Current video frame index.
            timeseries: Timeseries data dict.
            metric_keys: List of metric keys to extract.
            window_size: Number of timesteps to include in window.

        Returns:
            Dictionary mapping metric keys to lists of values.
        """
        ts = self.get_timestep_for_frame(frame_idx)

        # Calculate window bounds
        end_ts = ts + 1
        start_ts = max(0, end_ts - window_size)

        result = {}
        for key in metric_keys:
            if key in timeseries:
                data = timeseries[key]
                if isinstance(data, dict) and "mean" in data:
                    means = data["mean"]
                    result[key] = means[start_ts:end_ts]
                elif isinstance(data, list):
                    result[key] = data[start_ts:end_ts]
                else:
                    result[key] = []
            else:
                result[key] = []

        return result

    def get_total_frames_for_episode(self) -> int:
        """Calculate expected number of video frames for the full episode.

        Returns:
            Estimated number of video frames.
        """
        episode_duration_s = self.num_timesteps * self.step_dt_seconds
        return int(math.ceil(episode_duration_s * self.video_fps))
