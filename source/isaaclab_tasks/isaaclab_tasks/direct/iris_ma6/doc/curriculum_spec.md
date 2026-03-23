# Curriculum Spec for iris_ma6

## Goal

Train a multi-agent drone tracking policy by increasing difficulty along one axis at a time, while **preventing catastrophic forgetting** of skills learned in earlier phases.

## Phase Timeline

```
Step:    0k   20k   40k   60k   80k   100k  130k  160k  200k
         |     |     |     |     |     |     |     |     |
Tracking:[────────────ramp────────────]full───────────────────
Target:        [────────────ramp────────────]full─────────────
Safety:                      [────ramp────]full───────────────
Coord:                       [────ramp────]full───────────────
Noise:                              [─ramp─]full──────────────
FixDelay:                                   [───ramp───]full──
RndDelay:                                         [──ramp──]f
Dropout:                                                [ramp]
Dynamics:                                                [ramp]
```

## Anti-Forgetting Principle

Parameters that ramp from easy→hard must **still sample easy values at full progress**. Two patterns:

1. **Expand-range sampling**: `uniform(min, min + progress * (max - min))` — the lower bound never moves. Used by initial states (geometry, zoom, velocities).

2. **Wide scale-range randomization**: At reset, multiply nominal by `uniform(low, high)` where `low` can reach near-zero. Used by controller gains (tau_zoom: 0.01–1.0×, max_lin_vel: 0.5–1.2×).

Parameters that use a **single scalar** (not per-env/per-agent sampled) cannot preserve easy values. These need API changes to support per-agent sampling.

## Current Status by Parameter

### Safe (easy values preserved at full progress)
| Parameter | Mechanism | Range at p=1 |
|-----------|-----------|-------------|
| Formation geometry | expand-range | [20m, 100m] diameter |
| Target distance | expand-range | [10m, 40m] |
| Initial zoom | expand-range | [1.0, 6.0]× |
| Agent/target velocity | expand-range | [0, max] |
| tau_zoom | wide scale | [0.01, 1.0]× nominal |
| max_lin_vel | wide scale | [0.5, 1.2]× nominal |
| Controller gains (Kp, Ki, Kd) | per-agent randomization | [0.8, 1.2]× nominal |
| Gimbal init orientation | probabilistic ("gradual") | mix of pointing + random |

### Monotonic ramps (intentional, not fixing)
| Parameter | Rationale |
|-----------|-----------|
| CBF penalty weight | Safety must be monotonically enforced |
| Task reward levels (FIM→L2→L3) | Reward shaping is intentionally irreversible |
| Coordination reward scale | Additive reward, not a difficulty axis |

### Needs per-agent API (deferred)
| Parameter | Current | Needed |
|-----------|---------|--------|
| Noise scale | single float for all agents | per-agent tensor |
| Dropout rate | single float for all agents | per-agent tensor |
| max_lin_vel | per-env, shared across agents | per-agent `(N, num_agents)` |

See `doc/active/todo.md` for tracking.

## Implementation Files
- Phase definitions: `curriculum/curriculum_cfg.py`
- Gain randomization: `controller/gain_randomization_cfg.py`, `controller/drone_controller.py`
- Initial states: `initial_states/initial_states_generator.py`
- Delay/noise/dropout: `delay_system_v3/multi_agent_wrapper.py`
- Integration: `iris_ma_env6_test.py` (`_reset_idx`, `_update_curriculum`)
