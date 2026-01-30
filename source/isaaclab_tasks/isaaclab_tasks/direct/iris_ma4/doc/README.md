# iris_ma4: Multi-Agent Drone Environment

A GPU-accelerated multi-agent reinforcement learning environment for cooperative drone target tracking with realistic sensor delays and communication imperfections.

## Overview

**iris_ma4** simulates multiple drones equipped with gimbal-stabilized cameras that must cooperatively track and triangulate a target. Each drone observes the target from its own perspective, and the agents share information (with realistic delays) to improve localization accuracy through multi-view triangulation.

This environment is designed for research in:
- Multi-agent reinforcement learning (MARL)
- Cooperative perception and sensor fusion
- Communication-aware decision making
- Sim-to-real transfer with realistic sensor models

## Key Features

- **Multi-Agent Coordination**: 2+ drones with individual action/observation spaces
- **Gimbal-Stabilized Cameras**: Independent camera control for target tracking
- **Realistic Delay System**: Ego states are fast, inter-agent communication is slow with dropout
- **Triangulation Rewards**: Encourages agents to maintain good viewing geometry
- **Safety Constraints**: Collision avoidance and time-to-collision penalties
- **Curriculum Learning**: 5-phase progressive difficulty training
- **Domain Randomization**: Mass, dynamics, and trajectory randomization

## Architecture

```
                              iris_ma4 Environment
    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
    │  │   drone_0    │     │   drone_1    │     │   drone_N    │          │
    │  │              │     │              │     │     ...      │          │
    │  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │          │
    │  │ │PointMass │ │     │ │PointMass │ │     │ │PointMass │ │          │
    │  │ │Controller│ │     │ │Controller│ │     │ │Controller│ │          │
    │  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │          │
    │  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │          │
    │  │ │  Gimbal  │ │     │ │  Gimbal  │ │     │ │  Gimbal  │ │          │
    │  │ │Stabilizer│ │     │ │Stabilizer│ │     │ │Stabilizer│ │          │
    │  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │          │
    │  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │          │
    │  │ │  BBox    │ │     │ │  BBox    │ │     │ │  BBox    │ │          │
    │  │ │Raycaster │ │     │ │Raycaster │ │     │ │Raycaster │ │          │
    │  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │          │
    │  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘          │
    │         │                    │                    │                   │
    │         └────────────────────┼────────────────────┘                   │
    │                              ▼                                        │
    │                  ┌───────────────────────┐                            │
    │                  │  MultiAgentDelaySystem │                           │
    │                  │          V2            │                           │
    │                  ├───────────────────────┤                            │
    │                  │ ┌───────┐ ┌─────────┐ │                            │
    │                  │ │ Clean │ │  Noisy  │ │                            │
    │                  │ │Pipeline│ │Pipeline │ │                            │
    │                  │ │(rewards)│(observations)│                          │
    │                  │ └───────┘ └─────────┘ │                            │
    │                  └───────────┬───────────┘                            │
    │                              │                                        │
    │         ┌────────────────────┼────────────────────┐                   │
    │         ▼                    ▼                    ▼                   │
    │  ┌─────────────┐    ┌───────────────┐    ┌─────────────┐             │
    │  │Triangulation│    │SafetyManager  │    │ Curriculum  │             │
    │  │  Reward     │    │(Collision/TTC)│    │  Manager    │             │
    │  └─────────────┘    └───────────────┘    └─────────────┘             │
    │                                                                        │
    └───────────────────────────────────────────────────────────────────────┘
```

## Components

### Core Environment
| Component | Description |
|-----------|-------------|
| [iris_ma_env4.py](../iris_ma_env4.py) | Main environment implementation |
| [iris_ma_env4_cfg.py](../iris_ma_env4_cfg.py) | Configuration dataclass |

### Delay System V2
Realistic sensor delays with perspective-aware communication. See [delay_system_v2/doc/README.md](../delay_system_v2/doc/README.md).

| Perspective | Latency | Features |
|-------------|---------|----------|
| Ego (own state) | ~5ms | First-order lag only |
| Inter-agent | ~100ms | Lag + staleness + latency + dropout |

### Controllers
| Component | Purpose |
|-----------|---------|
| [PointMass](../controller/point_mass.py) | Velocity commands → thrust/torque |
| [GimbalStabilizer](../controller/gimbal_stabilizer.py) | Camera gimbal control |

### Safety
| Component | Purpose |
|-----------|---------|
| [SafetyManager](../safety/safety_manager.py) | Collision detection and TTC computation |

### Detection & Triangulation
| Component | Purpose |
|-----------|---------|
| [BBoxRaycaster](../bbox_raycaster/bbox_raycaster.py) | GPU-accelerated target detection |
| [Triangulation](../triangulation/triang_cov_reward_torch.py) | Multi-view 3D localization |

## Action Space

Per agent: **7 dimensions**
```
[vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
```

## Observation Space

Per agent: **47 dimensions** including:
- Ego state (position, orientation, velocities, gimbal angles, zoom)
- Detection info (bounding box, validity, age)
- Other agents' states (delayed/noisy)

## Reward Structure

| Reward Type | Scale | Description |
|-------------|-------|-------------|
| BBox Center | 60.0 | Target centering in image |
| BBox Size | 60.0 | Appropriate zoom level |
| Triangulation | 5.0 | Multi-agent localization quality |
| Collision | -100.0 | Penalty for inter-agent collision |
| TTC | -10.0 | Time-to-collision risk penalty |
| Action | -1.0 | Action magnitude penalty |

## Training

Supported RL frameworks:
- SKRL (MAPPO, MAPPO-RNN, PPO)
- RL-Games
- RSL-RL
- Stable-Baselines3

Example training command:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-IrisMA-v4 \
    --num_envs 512
```

## Curriculum Phases

| Phase | Focus | Steps |
|-------|-------|-------|
| 1 | Single-agent tracking | 10k-80k |
| 2 | Delay system introduction | 60k-150k |
| 3 | Multi-agent coordination | 120k-200k |
| 4 | Safety constraints | 180k-230k |
| 5 | Dynamics randomization | 40k-120k |

## File Structure

```
iris_ma4/
├── iris_ma_env4.py              # Main environment
├── iris_ma_env4_cfg.py          # Configuration
├── delay_system_v2/             # Realistic delays
├── controller/                  # Drone & gimbal control
├── safety/                      # Collision & TTC
├── bbox_raycaster/              # Target detection
├── triangulation/               # Multi-view localization
├── randomization/               # Domain randomization
├── curriculum/                  # Training curriculum
├── agents/                      # RL framework configs
├── visualization/               # Debug visualization
└── doc/                         # Documentation
```

## Related Documentation

- [Delay System V2](../delay_system_v2/doc/README.md) - Detailed delay pipeline documentation
- [Safety Module](../safety/README.md) - Collision detection and TTC
- [Controller](../controller/README.md) - Point mass and gimbal control
