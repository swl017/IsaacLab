# Initial States Module Documentation

## Overview

This module provides curriculum-driven randomization of agent and target initial states for the iris_ma6 multi-agent drone environment. It replaces hardcoded triangle formations with flexible, curriculum-aware placement that prevents catastrophic forgetting during RL training.

## Documents

| Document | Description |
|----------|-------------|
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Complete implementation details, algorithm, and integration guide |
| [../../doc/initial_states_spec.md](../../doc/initial_states_spec.md) | Original specification document (in parent doc folder) |

## Quick Reference

### Module Structure
```
initial_states/
├── __init__.py                    # Public exports
├── initial_states_cfg.py          # Configuration dataclasses
├── initial_states_generator.py    # Core generation logic (10-step algorithm)
├── initial_states.py              # Wrapper class for environment integration
├── doc/
│   ├── DOCUMENT_INDEX.md          # This file
│   └── IMPLEMENTATION_SUMMARY.md  # Implementation details
└── tests/
    ├── __init__.py
    ├── run_tests.py               # 24 unit tests
    └── README.md                  # Test documentation
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `InitialStatesCfg` | initial_states_cfg.py | Configuration parameters |
| `InitialStatesResult` | initial_states_cfg.py | Output dataclass with all state tensors |
| `InitialStatesGenerator` | initial_states_generator.py | Core generation algorithm |
| `InitialStates` | initial_states.py | Wrapper for environment integration |

### Usage Example

```python
from isaaclab_tasks.direct.iris_ma6.initial_states import (
    InitialStatesCfg,
    InitialStates,
)

# Create configuration
cfg = InitialStatesCfg(
    cylinder_diameter_min=30.0,
    cylinder_diameter_max=100.0,
    target_distance_min=30.0,
    target_distance_max=200.0,
)

# Create generator
initial_states = InitialStates(
    cfg=cfg,
    num_envs=256,
    num_agents=3,
    device=torch.device("cuda"),
)

# Generate at different curriculum stages
result_easy = initial_states.generate(curriculum_progress=0.0)
result_hard = initial_states.generate(curriculum_progress=1.0)
```

### Curriculum Sampling Strategy

**Key insight**: To prevent catastrophic forgetting, curriculum-controlled parameters use:

```
value ~ Uniform(min, min + progress * (max - min))
```

This ensures all previous difficulty levels remain in the sampling distribution as training progresses.

## Running Tests

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/initial_states/tests/run_tests.py
```

All 24 tests should pass:
- Configuration tests (2)
- Generator tests (7)
- Designated observer tests (3)
- Gimbal curriculum tests (2)
- Velocity tests (4)
- Zoom tests (2)
- Integration tests (4)
