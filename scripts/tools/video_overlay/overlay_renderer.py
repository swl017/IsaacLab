# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core overlay rendering with OpenCV."""

from __future__ import annotations

import cv2
import math
import numpy as np
from typing import Any

from overlay_config import OverlayConfig, MetricDisplayConfig, PanelLayoutConfig, ColorTheme


class OverlayRenderer:
    """Render metric overlays on video frames.

    This class handles drawing all visual overlays including:
    - Semi-transparent info panel with metric values
    - Color-coded status indicators
    - Animated metric graphs (optional)
    - Timestamp display

    Example:
        config = OverlayConfig(
            panel=PanelLayoutConfig(position="top_right"),
            metrics=[...],
        )
        renderer = OverlayRenderer(config)

        # In frame processing loop
        output_frame = renderer.render_frame(frame, metrics)
    """

    def __init__(self, config: OverlayConfig):
        """Initialize the overlay renderer.

        Args:
            config: Configuration for overlay rendering.
        """
        self.config = config
        self.theme = config.theme
        self.panel_cfg = config.panel

        # Pre-compute panel height based on number of metrics
        # Each metric row now has: name + value on one line, then a proper graph below
        visible_metrics = [m for m in config.metrics if m.show_in_panel]
        title_height = 35
        self.graph_height = 50  # Height of each metric graph (larger for readability)
        self.time_label_height = 15  # Height for time axis labels
        row_height = self.panel_cfg.line_height + self.graph_height + self.time_label_height + 8
        metrics_height = len(visible_metrics) * row_height
        self.panel_height = title_height + metrics_height + 2 * self.panel_cfg.padding

        # History buffer for metric graphs (stores last N values per metric)
        self.graph_window = 120  # ~4 seconds at 30fps for better visualization
        self.metric_history: dict[str, list[float]] = {
            m.key: [] for m in config.metrics
        }
        self.time_history: list[float] = []  # Store simulation times

    def render_frame(
        self,
        frame: np.ndarray,
        metrics: dict[str, float],
        graph_data: dict[str, list[float]] | None = None,
    ) -> np.ndarray:
        """Render all overlays onto a single frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format.
            metrics: Dictionary of metric values for this frame.
            graph_data: Optional graph data for animated graph.

        Returns:
            Frame with overlays as numpy array.
        """
        out = frame.copy()

        # Update metric history and time history for graphs
        sim_time = metrics.get("sim_time", 0.0)
        self.time_history.append(sim_time)
        if len(self.time_history) > self.graph_window:
            self.time_history = self.time_history[-self.graph_window:]

        for metric_cfg in self.config.metrics:
            key = metric_cfg.key
            value = metrics.get(key)
            if key in self.metric_history:
                if value is not None:
                    self.metric_history[key].append(value)
                # Keep only last N values
                if len(self.metric_history[key]) > self.graph_window:
                    self.metric_history[key] = self.metric_history[key][-self.graph_window:]

        # Draw info panel with metric values and graphs
        out = self._draw_info_panel(out, metrics)

        # Draw animated graph if enabled and data provided
        if self.config.graph.enabled and graph_data is not None:
            out = self._draw_graph(out, graph_data, metrics.get("timestep", 0))

        # Draw timestamp if enabled
        if self.config.show_timestamp:
            out = self._draw_timestamp(out, metrics.get("sim_time", 0.0))

        return out

    def _draw_info_panel(self, frame: np.ndarray, metrics: dict[str, float]) -> np.ndarray:
        """Draw semi-transparent info panel with metric values.

        Args:
            frame: Input frame.
            metrics: Metric values dictionary.

        Returns:
            Frame with info panel drawn.
        """
        h, w = frame.shape[:2]
        cfg = self.panel_cfg

        # Calculate panel position
        panel_w = cfg.width
        panel_h = self.panel_height

        if cfg.position == "top_right":
            x1 = w - panel_w - cfg.margin
            y1 = cfg.margin
        elif cfg.position == "top_left":
            x1 = cfg.margin
            y1 = cfg.margin
        elif cfg.position == "bottom_right":
            x1 = w - panel_w - cfg.margin
            y1 = h - panel_h - cfg.margin
        else:  # bottom_left
            x1 = cfg.margin
            y1 = h - panel_h - cfg.margin

        x2 = x1 + panel_w
        y2 = y1 + panel_h

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.theme.background, -1)
        cv2.addWeighted(
            overlay, cfg.background_alpha,
            frame, 1 - cfg.background_alpha,
            0, frame
        )

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.theme.accent, cfg.border_width)

        # Draw title
        title = "Tracking Metrics"
        title_y = y1 + cfg.padding + 20
        cv2.putText(
            frame, title,
            (x1 + cfg.padding, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.title_font_scale,
            self.theme.accent,
            2
        )

        # Draw horizontal line under title
        line_y = title_y + 8
        cv2.line(frame, (x1 + cfg.padding, line_y), (x2 - cfg.padding, line_y), self.theme.accent, 1)

        # Draw metric values with time-series graphs
        y_offset = line_y + cfg.line_height
        row_height = cfg.line_height + self.graph_height + self.time_label_height + 8

        for metric_cfg in self.config.metrics:
            if not metric_cfg.show_in_panel:
                continue

            value = metrics.get(metric_cfg.key)
            color = self._get_metric_color(value, metric_cfg)

            # Format value string
            if value is None:
                value_str = "N/A"
                color = self.theme.na_color
            elif metric_cfg.key == "visibility":
                # Show as percentage
                value_str = f"{value * 100:.{metric_cfg.decimals}f}"
            else:
                value_str = f"{value:.{metric_cfg.decimals}f}"

            # Draw metric name
            name_text = f"{metric_cfg.name}:"
            cv2.putText(
                frame, name_text,
                (x1 + cfg.padding, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                self.theme.text_secondary,
                1
            )

            # Draw value with unit (right-aligned)
            value_text = f"{value_str} {metric_cfg.unit}".strip()
            (text_w, _), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1)
            value_x = x2 - cfg.padding - text_w
            cv2.putText(
                frame, value_text,
                (value_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                color,
                1
            )

            # Draw time-series graph below the metric value
            graph_y = y_offset + 5
            graph_x1 = x1 + cfg.padding
            graph_x2 = x2 - cfg.padding
            graph_w = graph_x2 - graph_x1

            history = self.metric_history.get(metric_cfg.key, [])
            if len(history) >= 2:
                self._draw_metric_graph(
                    frame,
                    history,
                    self.time_history,
                    graph_x1, graph_y,
                    graph_w, self.graph_height,
                    color, metric_cfg
                )
            else:
                # Draw empty graph area (just a border)
                cv2.rectangle(
                    frame,
                    (graph_x1, graph_y),
                    (graph_x2, graph_y + self.graph_height),
                    (60, 60, 60), 1
                )

            y_offset += row_height

        return frame

    def _draw_metric_graph(
        self,
        frame: np.ndarray,
        values: list[float],
        times: list[float],
        x: int, y: int,
        width: int, height: int,
        color: tuple[int, int, int],
        metric_cfg: MetricDisplayConfig,
    ):
        """Draw a time-series graph with time axis labels.

        Args:
            frame: Frame to draw on.
            values: List of metric values.
            times: List of simulation times corresponding to values.
            x, y: Top-left position.
            width, height: Graph dimensions (excluding time labels).
            color: Line color.
            metric_cfg: Metric configuration for thresholds.
        """
        if len(values) < 2:
            return

        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)

        # Calculate value range
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val
        if val_range < 0.001:
            val_range = 1.0
            min_val = min_val - 0.5

        # Draw threshold lines if meaningful
        good_th = metric_cfg.good_threshold
        bad_th = metric_cfg.bad_threshold
        if metric_cfg.invert_thresholds:
            # Higher is better - draw good threshold
            if min_val < good_th < max_val:
                th_y = y + height - int((good_th - min_val) * (height - 4) / val_range) - 2
                cv2.line(frame, (x + 2, th_y), (x + width - 2, th_y), self.theme.good, 1)
        else:
            # Lower is better - draw bad threshold
            if min_val < bad_th < max_val:
                th_y = y + height - int((bad_th - min_val) * (height - 4) / val_range) - 2
                cv2.line(frame, (x + 2, th_y), (x + width - 2, th_y), self.theme.bad, 1)

        # Create points for the graph line
        points = []
        padding = 2
        plot_width = width - 2 * padding
        plot_height = height - 2 * padding

        for i, v in enumerate(values):
            px = x + padding + int(i * plot_width / max(len(values) - 1, 1))
            py = y + height - padding - int((v - min_val) * plot_height / val_range)
            py = max(y + padding, min(y + height - padding, py))
            points.append((px, py))

        # Draw the graph line
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

            # Draw current value dot at the end
            last_pt = points[-1]
            cv2.circle(frame, last_pt, 4, color, -1)

        # Draw time axis labels below the graph
        label_y = y + height + 12
        font_scale = 0.35
        font_color = (150, 150, 150)

        if len(times) >= 2:
            # Start time
            start_time = times[0]
            start_label = f"{start_time:.1f}s"
            cv2.putText(frame, start_label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)

            # End time (current)
            end_time = times[-1]
            end_label = f"{end_time:.1f}s"
            (tw, _), _ = cv2.getTextSize(end_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.putText(frame, end_label, (x + width - tw, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)

            # Middle time marker (optional, if space allows)
            if width > 200:
                mid_time = (start_time + end_time) / 2
                mid_label = f"{mid_time:.1f}s"
                (mw, _), _ = cv2.getTextSize(mid_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.putText(frame, mid_label, (x + width // 2 - mw // 2, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)

    def _get_metric_color(
        self,
        value: float | None,
        metric_cfg: MetricDisplayConfig
    ) -> tuple[int, int, int]:
        """Get color based on metric value and thresholds.

        Args:
            value: Metric value (or None).
            metric_cfg: Metric configuration.

        Returns:
            BGR color tuple.
        """
        if value is None:
            return self.theme.na_color

        good_th = metric_cfg.good_threshold
        bad_th = metric_cfg.bad_threshold

        if metric_cfg.invert_thresholds:
            # Higher is better (e.g., viewing angle, visibility)
            if value >= good_th:
                return self.theme.good
            elif value <= bad_th:
                return self.theme.bad
            else:
                return self.theme.warning
        else:
            # Lower is better (e.g., RMSE)
            if value <= good_th:
                return self.theme.good
            elif value >= bad_th:
                return self.theme.bad
            else:
                return self.theme.warning

    def _draw_graph(
        self,
        frame: np.ndarray,
        graph_data: dict[str, list[float]],
        current_timestep: int,
    ) -> np.ndarray:
        """Draw animated time-series graph.

        Args:
            frame: Input frame.
            graph_data: Dictionary mapping metric keys to value lists.
            current_timestep: Current simulation timestep.

        Returns:
            Frame with graph drawn.
        """
        h, w = frame.shape[:2]
        graph_cfg = self.config.graph

        # Calculate graph position
        graph_w = graph_cfg.width
        graph_h = graph_cfg.height
        margin = 20

        if graph_cfg.position == "bottom_left":
            x1, y1 = margin, h - graph_h - margin
        elif graph_cfg.position == "bottom_right":
            x1, y1 = w - graph_w - margin, h - graph_h - margin
        elif graph_cfg.position == "top_left":
            x1, y1 = margin, margin
        else:  # top_right
            x1, y1 = w - graph_w - margin, margin

        x2, y2 = x1 + graph_w, y1 + graph_h

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.theme.background, -1)
        cv2.addWeighted(
            overlay, graph_cfg.background_alpha,
            frame, 1 - graph_cfg.background_alpha,
            0, frame
        )

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.theme.accent, 1)

        # Draw each metric as a line
        colors = [
            (0, 255, 255),  # Yellow
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (255, 100, 255),  # Pink
        ]

        padding = 10
        plot_x1 = x1 + padding
        plot_y1 = y1 + padding
        plot_w = graph_w - 2 * padding
        plot_h = graph_h - 2 * padding

        for i, metric_key in enumerate(graph_cfg.metrics):
            if metric_key not in graph_data:
                continue

            values = graph_data[metric_key]
            if len(values) < 2:
                continue

            # Filter out None values
            valid_values = [v for v in values if v is not None]
            if len(valid_values) < 2:
                continue

            # Normalize values to plot area
            min_val = min(valid_values)
            max_val = max(valid_values)
            val_range = max_val - min_val
            if val_range < 0.001:
                val_range = 1.0

            # Create points for polyline
            points = []
            for j, v in enumerate(values):
                if v is None:
                    continue
                px = plot_x1 + int(j * plot_w / max(len(values) - 1, 1))
                py = plot_y1 + plot_h - int((v - min_val) * plot_h / val_range)
                points.append((px, py))

            # Draw line
            if len(points) >= 2:
                pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, colors[i % len(colors)], 2)

        return frame

    def _draw_timestamp(self, frame: np.ndarray, sim_time: float) -> np.ndarray:
        """Draw simulation timestamp.

        Args:
            frame: Input frame.
            sim_time: Simulation time in seconds.

        Returns:
            Frame with timestamp drawn.
        """
        h, w = frame.shape[:2]
        text = f"t = {sim_time:.2f}s"

        # Calculate position
        margin = 20
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        if self.config.timestamp_position == "bottom_left":
            x, y = margin, h - margin
        elif self.config.timestamp_position == "bottom_right":
            x, y = w - text_w - margin, h - margin
        elif self.config.timestamp_position == "top_left":
            x, y = margin, text_h + margin
        else:  # top_right
            x, y = w - text_w - margin, text_h + margin

        # Draw background for readability
        pad = 5
        cv2.rectangle(
            frame,
            (x - pad, y - text_h - pad),
            (x + text_w + pad, y + pad),
            self.theme.background,
            -1
        )

        # Draw text
        cv2.putText(
            frame, text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.theme.text_secondary,
            1
        )

        return frame
