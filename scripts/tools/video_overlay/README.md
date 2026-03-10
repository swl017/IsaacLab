# Video Tools for IROS Paper Demonstrations

A suite of tools for creating professional demonstration videos with metrics visualization.

## Tools Overview

| Tool | Description |
|------|-------------|
| `video_overlay.py` | Overlay metrics panel on simulation video |
| `plot_video.py` | Generate animated time-series plot video |
| `combine_videos.py` | Combine simulation + plot videos |
| `generate_demo_video.sh` | One-command pipeline for all layouts |

## Features

- **Metrics Panel**: Semi-transparent panel showing viewing angle, RMSE, visibility, etc.
- **Animated Plots**: Full-width time-series graphs with playhead animation
- **Color Coding**: Green/yellow/red thresholds for quick visual assessment
- **Multiple Layouts**: Overlay, side-by-side, stacked, picture-in-picture
- **Professional Styling**: Dark/light/paper color schemes
- **Flexible Configuration**: Customizable panel positions, sizes, and metrics

## Usage

### Basic Overlay

```bash
python video_overlay.py \
    --input-video demo_raw.mp4 \
    --input-json results.json \
    --output-video demo_overlay.mp4
```

### Select Specific Metrics

```bash
python video_overlay.py \
    --input-video demo_raw.mp4 \
    --input-json results.json \
    --output-video demo_overlay.mp4 \
    --metrics viewing_angle rmse visibility
```

### With Animated Graph

```bash
python video_overlay.py \
    --input-video demo_raw.mp4 \
    --input-json results.json \
    --output-video demo_overlay.mp4 \
    --enable-graph \
    --graph-metrics viewing_angle rmse
```

### Using Trajectory Data

```bash
python video_overlay.py \
    --input-video demo_raw.mp4 \
    --input-json trajectory.json \
    --output-video demo_overlay.mp4 \
    --data-type trajectory \
    --env-id 0
```

### Custom Thresholds

```bash
python video_overlay.py \
    --input-video demo_raw.mp4 \
    --input-json results.json \
    --output-video demo_overlay.mp4 \
    --rmse-thresholds 2.0 5.0 \
    --angle-thresholds 30.0 20.0 \
    --visibility-thresholds 0.8 0.5
```

## Command-Line Arguments

### Required
- `--input-video`: Path to raw video file from Isaac Sim
- `--input-json`: Path to JSON metrics from evaluate.py
- `--output-video`: Path for output video with overlays

### Data Options
- `--data-type`: `timeseries` (aggregate) or `trajectory` (per-env)
- `--env-id`: Environment ID for trajectory data (default: 0)

### Panel Configuration
- `--panel-position`: `top_left`, `top_right`, `bottom_left`, `bottom_right`
- `--panel-width`: Width in pixels (default: 300)
- `--panel-alpha`: Background transparency 0-1 (default: 0.7)

### Metrics
- `--metrics`: Space-separated list of metrics to display
  - Available: `viewing_angle`, `rmse`, `visibility`, `tri_valid`, `sqrt_trace_sigma`

### Graph Options
- `--enable-graph`: Enable animated metric graph
- `--graph-metrics`: Metrics to show in graph
- `--graph-position`: Graph panel position
- `--graph-window`: Number of timesteps in window (default: 100)

### Thresholds (for color coding)
- `--rmse-thresholds GOOD BAD`: RMSE thresholds in meters
- `--angle-thresholds GOOD BAD`: Viewing angle thresholds in degrees
- `--visibility-thresholds GOOD BAD`: Visibility thresholds as ratio

### Output
- `--fps`: Output video FPS (default: match input)
- `--codec`: OpenCV fourcc codec (default: mp4v)
- `--no-timestamp`: Disable timestamp display

## Input JSON Format

### Timeseries Data (from evaluate.py)

```json
{
  "timeseries": {
    "step_dt_seconds": 0.01667,
    "viewing_angle": {"mean": [...], "std": [...]},
    "triangulation_rmse": {"mean": [...], "std": [...]},
    "visibility": {"mean": [...], "std": [...]},
    "tri_valid": {"mean": [...], "std": [...]}
  }
}
```

### Trajectory Data (from evaluate.py --record-trajectory)

```json
{
  "trajectories": {
    "step_dt_seconds": 0.01667,
    "envs": {
      "0": {
        "num_steps": 2500,
        "viewing_angle": [...],
        "rmse": [...],
        "tri_valid": [true, false, ...]
      }
    }
  }
}
```

## End-to-End Workflow

This workflow produces professional demonstration videos for IROS paper submissions.

### Step 1: Record Video with Smooth Camera

Run evaluation with video recording enabled. The smooth camera controller provides professional-looking camera motion.

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/experiments/evaluate.py \
    --experiment a3_noisy_reward \
    --checkpoint /path/to/best_agent.pt \
    --num_episodes 1 \
    --num_envs 16 \
    --record-trajectory \
    --record-video raw_demo.mp4 \
    --camera-mode chase \
    --camera-smoothing 0.08 \
    --video-fps 30 \
    --video-resolution 1920 1080 \
    --output metrics.json
```

**Camera Mode Options:**

| Mode | Description | Best For |
|------|-------------|----------|
| `overhead` | Bird's eye view looking down | Formation overview, tactical view |
| `chase` | Follows target from behind | Action shots, dynamic tracking |
| `side` | Fixed lateral view | Consistent framing |
| `orbit` | Circular motion around scene | Cinematic establishing shots |
| `formation` | Follows agent centroid | Multi-agent coordination |
| `closeup` | Tight follow on target | Detail shots |
| `wide` | Distant overview | Full scene context |
| `isometric` | 45-degree tactical view | Classic game-style perspective |

**Smoothing Factor:**
- `0.03` - Very smooth (slow response, cinematic)
- `0.08` - Default (balanced)
- `0.15` - Responsive (faster tracking)

### Step 2: Add Metric Overlays

Post-process the recorded video to add real-time metric displays.

```bash
python scripts/tools/video_overlay/video_overlay.py \
    --input-video raw_demo.mp4 \
    --input-json metrics.json \
    --output-video final_demo.mp4 \
    --metrics viewing_angle rmse visibility \
    --panel-position top_right \
    --enable-graph \
    --graph-metrics viewing_angle rmse
```

### Complete Example: Paper Figure Video

```bash
# Step 1: Record with overhead camera for formation visualization
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/experiments/evaluate.py \
    --experiment a3_noisy_reward \
    --checkpoint logs/iris_ma5/a3_noisy/best_agent.pt \
    --num_episodes 1 \
    --num_envs 4 \
    --record-trajectory \
    --trajectory-envs 1 \
    --record-video formation_raw.mp4 \
    --camera-mode overhead \
    --camera-smoothing 0.05 \
    --target-trajectory-mode linear \
    --output formation_metrics.json

# Step 2: Add overlays with all key metrics
python scripts/tools/video_overlay/video_overlay.py \
    --input-video formation_raw.mp4 \
    --input-json formation_metrics.json \
    --output-video formation_final.mp4 \
    --data-type trajectory \
    --env-id 0 \
    --metrics viewing_angle rmse visibility tri_valid \
    --panel-position top_right \
    --panel-width 320 \
    --rmse-thresholds 2.0 5.0 \
    --angle-thresholds 30.0 20.0
```

### Output Files

After running the workflow, you'll have:
- `raw_demo.mp4` - Raw video with smooth camera motion
- `metrics.json` - JSON file with timeseries and trajectory data
- `final_demo.mp4` - Final video with metric overlays

### Tips for Best Results

1. **Use fewer environments** (`--num-envs 4-16`) for video recording to reduce GPU load
2. **Linear target mode** (`--target-trajectory-mode linear`) produces cleaner demo trajectories
3. **Lower smoothing** (`0.03-0.05`) for cinematic shots, higher (`0.10-0.15`) for action
4. **Record at native resolution** and downscale in post if needed for file size

## Metric Thresholds (Defaults)

| Metric | Good (Green) | Warning (Yellow) | Bad (Red) |
|--------|-------------|------------------|-----------|
| Viewing Angle | > 30 deg | 20-30 deg | < 20 deg |
| RMSE | < 2.0 m | 2.0-5.0 m | > 5.0 m |
| Visibility | > 80% | 50-80% | < 50% |
| Tri Valid | = 1.0 | - | = 0.0 |

---

## Animated Plot Video (plot_video.py)

Generate a standalone video with professional animated time-series graphs.

### Basic Usage

```bash
python plot_video.py \
    --input-json metrics.json \
    --output-video metrics_plot.mp4
```

### With Custom Styling

```bash
python plot_video.py \
    --input-json metrics.json \
    --output-video metrics_plot.mp4 \
    --metrics viewing_angle triangulation_rmse visibility \
    --style dark \
    --fps 30 \
    --resolution 1920 1080
```

### Style Options

| Style | Description | Best For |
|-------|-------------|----------|
| `dark` | Dark background with bright accents | Presentations, demos |
| `light` | Light background | Print-friendly |
| `paper` | Clean white, publication colors | Paper figures |

### Overlay Multiple Metrics on Same Axes

Combine related metrics on the same plot axes using `--overlay`:

```bash
# Overlay ego and communication delay on the same axes
python plot_video.py \
    --input-json metrics.json \
    --output-video delay_plot.mp4 \
    --metrics viewing_angle triangulation_rmse ego_aoi other_aoi \
    --overlay ego_aoi other_aoi
```

This creates a plot with:
- Viewing angle (separate axes)
- Triangulation RMSE (separate axes)
- Ego + Comm delay (shared axes with legend)

### Arguments

- `--input-json`: Path to metrics JSON file
- `--output-video`: Output video path
- `--data-type`: `timeseries` or `trajectory`
- `--env-id`: Environment ID for trajectory data
- `--metrics`: List of metrics to plot
- `--style`: Color scheme (`dark`, `light`, `paper`)
- `--fps`: Output FPS (default: 30)
- `--resolution WIDTH HEIGHT`: Video resolution
- `--no-std`: Hide standard deviation bands
- `--no-thresholds`: Hide threshold regions
- `--title`: Custom video title
- `--sync-video`: Match duration to another video
- `--overlay M1 M2`: Overlay metrics on same axes (can be used multiple times)

---

## Combine Videos (combine_videos.py)

Combine simulation and plot videos into various layouts.

### Side-by-Side Layout

```bash
python combine_videos.py \
    --simulation-video demo.mp4 \
    --plot-video metrics_plot.mp4 \
    --output-video combined.mp4 \
    --layout side_by_side \
    --sim-weight 0.6
```

### Stacked Layout

```bash
python combine_videos.py \
    --simulation-video demo.mp4 \
    --plot-video metrics_plot.mp4 \
    --output-video combined.mp4 \
    --layout stacked
```

### Picture-in-Picture

```bash
python combine_videos.py \
    --simulation-video demo.mp4 \
    --plot-video metrics_plot.mp4 \
    --output-video combined.mp4 \
    --layout pip \
    --pip-position bottom_right \
    --pip-scale 0.35
```

### Layout Options

| Layout | Description |
|--------|-------------|
| `side_by_side` | Simulation left, plots right |
| `stacked` | Simulation top, plots bottom |
| `pip` | Plots overlaid on simulation corner |

### Reuse a Saved Plot Video

Skip plot generation by combining directly with a previously saved plot video:

```bash
python combine_videos.py \
    --simulation-video raw_demo.mp4 \
    --plot-video delay_analysis_plot.mp4 \
    --output-video final_combined.mp4 \
    --layout side_by_side \
    --sim-weight 0.55
```

This is useful when:
- Re-combining with a different simulation video
- Testing different layout options without regenerating plots
- The plot video was saved using `--save-plot` in generate_demo_video.sh

---

## One-Command Pipeline (generate_demo_video.sh)

Generate complete demonstration videos with a single command.

### Side-by-Side (Recommended)

```bash
./generate_demo_video.sh \
    --sim-video raw_demo.mp4 \
    --json metrics.json \
    --output final.mp4 \
    --layout side_by_side
```

### Plot Only (No Simulation Video)

```bash
./generate_demo_video.sh \
    --json metrics.json \
    --output metrics_plot.mp4 \
    --layout plot_only \
    --style dark
```

### All Layout Options

```bash
# Simple overlay on simulation
./generate_demo_video.sh --sim-video demo.mp4 --json metrics.json --output out.mp4 --layout overlay

# Side-by-side (simulation + plots)
./generate_demo_video.sh --sim-video demo.mp4 --json metrics.json --output out.mp4 --layout side_by_side

# Vertically stacked
./generate_demo_video.sh --sim-video demo.mp4 --json metrics.json --output out.mp4 --layout stacked

# Picture-in-picture
./generate_demo_video.sh --sim-video demo.mp4 --json metrics.json --output out.mp4 --layout pip

# Plot video only
./generate_demo_video.sh --json metrics.json --output out.mp4 --layout plot_only
```

### Using Presets

Use built-in presets for common configurations:

```bash
# Default preset: viewing_angle, triangulation_rmse, visibility
./generate_demo_video.sh \
    --sim-video raw_demo.mp4 \
    --json metrics.json \
    --output final.mp4 \
    --preset default

# Delay preset: viewing_angle, rmse, and overlaid ego/other delays
./generate_demo_video.sh \
    --sim-video raw_demo.mp4 \
    --json metrics.json \
    --output delay_analysis.mp4 \
    --preset delay
```

### Using Overlay (Manual)

Overlay specific metrics on the same axes:

```bash
./generate_demo_video.sh \
    --sim-video raw_demo.mp4 \
    --json metrics.json \
    --output custom.mp4 \
    --metrics viewing_angle rmse ego_aoi other_aoi \
    --overlay ego_aoi other_aoi
```

### Save Plot Video for Reuse

Save the intermediate plot video for later reuse:

```bash
./generate_demo_video.sh \
    --sim-video raw_demo.mp4 \
    --json metrics.json \
    --output delay_analysis.mp4 \
    --preset delay \
    --layout side_by_side \
    --save-plot
```

This creates:
- `delay_analysis.mp4` - Combined final video
- `delay_analysis_plot.mp4` - Reusable plot video

### Arguments

- `--sim-video`: Input simulation video
- `--json`: Input metrics JSON file
- `--output`: Output video file
- `--style`: Color style (`dark`, `light`, `paper`)
- `--layout`: Layout mode (`overlay`, `plot_only`, `side_by_side`, `stacked`, `pip`)
- `--metrics`: List of metrics to show
- `--overlay M1 M2`: Overlay metrics on same axes (can be used multiple times)
- `--preset`: Use preset configuration (`default`, `delay`)
- `--fps`: Output FPS
- `--save-plot`: Save intermediate plot video as `{output}_plot.mp4`

### Available Presets

| Preset | Metrics | Overlays |
|--------|---------|----------|
| `default` | viewing_angle, triangulation_rmse, visibility | none |
| `delay` | viewing_angle, triangulation_rmse, ego_aoi, other_aoi | ego_aoi + other_aoi |

---

## Complete Workflow Example

### Step 1: Run Evaluation with Video Recording

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/experiments/evaluate.py \
    --experiment a3_noisy_reward \
    --checkpoint logs/iris_ma5/a3_noisy/best_agent.pt \
    --num_episodes 1 \
    --num_envs 16 \
    --record-trajectory \
    --record-video raw_demo.mp4 \
    --camera-mode overhead \
    --camera-smoothing 0.08 \
    --video-fps 25 \
    --output metrics.json
```

**Note:** Use `--video-fps 25` to match the simulation step rate (25 Hz) for real-time playback.

### Step 2: Generate Demo Video with Saved Plot

```bash
# Generate combined video and save the plot video for reuse
./scripts/tools/video_overlay/generate_demo_video.sh \
    --sim-video raw_demo.mp4 \
    --json metrics.json \
    --output delay_analysis.mp4 \
    --preset delay \
    --layout side_by_side \
    --style dark \
    --save-plot
```

### Step 3 (Optional): Reuse Plot Video with Different Layout

```bash
# Combine saved plot video with different settings (no regeneration)
python scripts/tools/video_overlay/combine_videos.py \
    --simulation-video raw_demo.mp4 \
    --plot-video delay_analysis_plot.mp4 \
    --output-video delay_analysis_pip.mp4 \
    --layout pip \
    --pip-position bottom_right \
    --pip-scale 0.4
```

### Output Files

| File | Description |
|------|-------------|
| `raw_demo.mp4` | Raw simulation video with smooth camera |
| `metrics.json` | Metrics data (timeseries + trajectory) |
| `delay_analysis.mp4` | Final combined video |
| `delay_analysis_plot.mp4` | Reusable plot video (if `--save-plot` used) |

### Quick Reference: Common Pipelines

```bash
# Full pipeline with evaluate.sh helper script
bash scripts/tools/video_overlay/evaluate.sh && \
./scripts/tools/video_overlay/generate_demo_video.sh \
    --sim-video raw_demo.mp4 \
    --json metrics.json \
    --output delay_analysis.mp4 \
    --preset delay \
    --layout side_by_side \
    --save-plot

# Reuse saved plot with new simulation video
python scripts/tools/video_overlay/combine_videos.py \
    --simulation-video new_demo.mp4 \
    --plot-video delay_analysis_plot.mp4 \
    --output-video new_combined.mp4 \
    --layout side_by_side \
    --sim-weight 0.55
```

---

## Dependencies

- `opencv-python (cv2)`: Video I/O, drawing
- `numpy`: Array operations
- `matplotlib`: Animated plot generation (for plot_video.py)
- Standard library: `json`, `argparse`, `pathlib`
