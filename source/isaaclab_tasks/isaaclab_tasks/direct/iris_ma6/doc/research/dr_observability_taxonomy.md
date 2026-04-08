# Domain Randomization Observability Taxonomy

**Date**: 2026-04-08
**Context**: Ticket-006 (DR integration) introduced per-episode randomization of camera intrinsics, physics mass, gimbal offsets, gimbal dynamics, and target scale. This document analyzes which randomized parameters the RL policy should observe explicitly vs. learn to handle implicitly.

## Decision Framework

For each DR parameter, ask: **at deployment, can the agent know this value before acting?**

Three categories emerge:

1. **Explicit** — parameter is known, static, and trustworthy at deployment. Withholding it from the policy caps performance unnecessarily.
2. **Implicit (feedback-observable)** — parameter is unknown but its effect is visible in the observation stream. The policy learns to compensate from downstream effects via closed-loop feedback.
3. **Implicit (unobservable)** — parameter is unknown and its effect cannot be distinguished from other factors. The policy must learn a robust strategy that works across the full range.

## Per-Parameter Analysis

| DR Parameter | Knowable at deployment? | Observable indirectly via feedback? | Category | Obs? |
|---|---|---|---|---|
| Camera intrinsics (FOV / focal length) | **Yes** — factory calibrated, static per camera | No — single-view bbox is ambiguous without target size/distance | **Explicit** | **Yes (1D)** |
| Zoom level | **Yes** — commanded by policy | N/A | **Explicit** | Already in obs |
| Gimbal joint offsets | **Partially** — initial calibration known, drifts with temperature/vibration | **Yes** — bbox consistently off-center when gimbal thinks it's on-target | **Implicit (feedback)** | No |
| Mass / payload | **No** — varies per flight (battery, payload) | **Yes** — hover throttle percentage, acceleration response to commands | **Implicit (feedback)** | No |
| Gimbal dynamics (stiffness, damping) | **No** — temperature and wear dependent | **Yes** — gimbal tracking lag visible in bbox motion | **Implicit (feedback)** | No |
| Material properties (friction, restitution) | **No** — irrelevant in flight | No — only affects ground contact | **Implicit (unobservable)** | No |
| Target 3D size (scale) | **No** — unknown target, genuinely uncertain | **No** — single-view size-distance ambiguity is fundamental | **Implicit (unobservable)** | No |

## Detailed Reasoning

### Camera Intrinsics: Explicit (the exception)

Camera intrinsics are the **only** DR parameter that satisfies all three conditions for explicit observation:

1. **Known at deployment**: Every camera ships with a calibration (focal length, principal point). This is measured once and stored.
2. **Static within episode**: Unlike mass (battery drains) or gimbal offsets (thermal drift), intrinsics don't change mid-flight.
3. **Not inferable from feedback**: A 50x30px bbox at zoom=1.0 could mean a normal target at 20m with nominal intrinsics, a 2x-scaled target at 40m, or a normal target at 10m with narrow FOV. Without knowing intrinsics AND target size, distance is unrecoverable from a single bbox. Multi-view triangulation helps, but knowing intrinsics strictly improves the estimate.

Withholding intrinsics forces the policy to learn a "one-size-fits-all" mapping from bbox pixels to geometry, averaged over the FOV distribution. This caps asymptotic performance because the agent can't exploit its known camera geometry for zoom decisions, formation spacing, or triangulation baseline optimization.

**Representation**: 1D scalar `effective_focal_length / nominal_focal_length` (i.e., `dr_intrinsic_scale`). The agent already observes `zoom` — the combined effective scaling is `zoom * dr_intrinsic_scale`, but keeping them separate lets the agent reason about "my camera" vs. "my zoom command" independently.

### Gimbal Offsets: Implicit (feedback-observable)

Gimbal offsets simulate mechanical misalignment (yaw ±0.1 rad, pitch ±0.05 rad, roll ±0.02 rad).

**Why NOT explicit**: In the real world, the initial calibration value drifts with temperature and vibration. Training the policy to rely on a stored offset value creates brittleness when it inevitably drifts. Better to never trust the stored value.

**Why implicit works**: The policy has a closed-loop feedback signal — the bbox position in the image. If there's a +0.05 rad yaw offset, the target appears consistently shifted from where the gimbal "thinks" it's pointing. The policy sees this shift every step and adjusts gimbal commands to center it. This is standard output-feedback robust control: you don't need to know the disturbance if you can observe its effect.

The RNN memory architecture helps here — it can estimate the persistent bias over a few steps and compensate it as a learned integrator, similar to how an I-term in PID compensates for steady-state error without knowing its source.

### Mass / Payload: Implicit (feedback-observable)

Mass affects thrust-to-weight ratio. A heavier drone needs more throttle to hover and accelerates more slowly.

**Why NOT explicit**: Mass changes per flight (different batteries, payloads). No reliable pre-flight measurement in the field.

**Why implicit works**: The policy observes body linear acceleration (`lin_acc_b`) and velocity. If the drone is 10% heavier, altitude commands produce 10% less acceleration. The RNN can identify this relationship within a few steps and adjust command aggressiveness accordingly. This is analogous to adaptive control — the policy implicitly estimates the plant gain.

### Gimbal Dynamics: Implicit (feedback-observable)

Stiffness and damping scaling affects how quickly the gimbal tracks commanded positions.

**Why NOT explicit**: These properties depend on temperature, wear, and lubrication state. Not measured pre-flight.

**Why implicit works**: The policy observes gimbal joint positions and the resulting bbox motion. If damping is 1.2x nominal, the gimbal responds 20% more sluggishly. The RNN can detect this from the lag between gimbal commands and bbox response, and learn to command more aggressively or more smoothly accordingly.

### Target 3D Size: Implicit (unobservable)

Target scale (xy: 0.5-2.5x, z: 1.0-3.0x) varies the apparent 3D size.

**Why NOT explicit**: The target's true size is genuinely unknown at deployment. Different targets (cars, people, drones) have different sizes.

**Why implicit doesn't fully work either**: Unlike gimbal offsets or mass, target size creates a **fundamental ambiguity** with distance. A large target far away looks identical to a small target nearby from a single camera. This ambiguity is resolved by multi-view triangulation (two cameras at different angles can recover distance regardless of target size), which is exactly what the multi-agent formation is designed to do.

**This is the core benefit of target scale DR**: it breaks the brittle monocular size-distance coupling and forces the policy to rely on multi-view triangulation geometry rather than single-view size heuristics. The policy learns that bbox size alone is unreliable and must coordinate with other agents for distance estimation.

## Summary

```
                    Known at deployment?
                    /                \
                  Yes                 No
                   |                   |
          Observable via feedback?  Observable via feedback?
              /          \             /          \
         (N/A - known)  (N/A)       Yes            No
              |                      |              |
          EXPLICIT              IMPLICIT        IMPLICIT
          (in obs)            (feedback)      (unobservable)
              |                      |              |
        Camera intrinsics    Gimbal offsets    Target size
        (1D: focal scale)    Mass/payload      Materials
                             Gimbal dynamics
```

**Policy observation change**: +1D camera intrinsic scale in ego obs (ticket-012).
Everything else stays implicit — the policy learns robustness through closed-loop feedback or multi-view geometry.

## Connection to Control Theory

This taxonomy maps to classical robust control concepts:

- **Explicit params** ↔ **Measured disturbance feedforward**: Known plant parameters go into the controller's model. Withholding them means operating in "nominal-only" mode.
- **Implicit (feedback)** ↔ **Output-feedback robust control**: The controller doesn't know the disturbance but can observe its effect on the output. Integral action (RNN memory) compensates persistent disturbances.
- **Implicit (unobservable)** ↔ **H∞ / min-max robustness**: The controller must guarantee performance across a bounded uncertainty set without any feedback about the specific realization.

The RL policy's RNN memory serves the role of both the integral term (for feedback-observable disturbances) and the adaptive estimator (for slowly-varying parameters like gimbal dynamics). Domain randomization during training serves the role of the uncertainty set definition in robust control.

## Open Questions

1. **Inter-agent intrinsic sharing**: Should agent A's observation include agent B's camera intrinsic scale? At deployment, agents could share calibration over the comm link. This would let agent A reason about agent B's camera geometry for cooperative triangulation planning. Cost: +1D per other agent in inter-agent obs. Benefit: better formation geometry optimization. See ticket-012.

2. **Intrinsic scale noise**: Should the observed intrinsic scale have noise to simulate calibration uncertainty? Small noise (1-2%) would prevent over-reliance on exact intrinsics. But real calibration is quite precise (<0.5% error), so this may be unnecessary.

3. **Target size estimation**: Could the policy benefit from a running estimate of target size (derived from triangulation distance + bbox size)? This is NOT adding target size to obs (that's cheating) — it's whether the policy architecture should have capacity to maintain such an estimate internally. The RNN may already do this implicitly.