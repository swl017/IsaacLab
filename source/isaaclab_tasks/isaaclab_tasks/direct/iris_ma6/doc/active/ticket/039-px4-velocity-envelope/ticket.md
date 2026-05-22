## Ticket 039 — Asymmetric PX4 z-velocity envelope (sim-to-real gap)

**Status**: Implemented (Slices 1–3 done, training/sim-to-sim validation pending)
**Created**: 2026-05-22
**Implemented**: 2026-05-22
**Discovered in**: Ticket 037 / Phase 1 training-action diagnostics (user observation of jerky vertical motion + attitude destabilization).
**Affected modules**: `iris_ma_env6_test.py` (`_pre_physics_step` action scaling), `iris_ma_env6_test_cfg.py` (velocity-envelope cfg).
**Blocks**: any sim-to-sim or sim-to-real testing on the PX4 stack.

---

### Implementation notes (2026-05-22)

- Cfg fields added (`iris_ma_env6_test_cfg.py`, just after `max_yaw_rate`):
  `enable_asymmetric_z_envelope: bool = False`, `max_vel_z_up: float = 3.0`,
  `max_vel_z_dn: float = 1.5`. Defaults preserve t034 behavior.
- Action scaling branch added in `_pre_physics_step`
  (`iris_ma_env6_test.py:811-826`). xy keeps `_max_lin_vel`; z uses
  `torch.where(z>=0, up, dn)` for sign-dependent scaling. Composes with
  ticket 038 because xy scaling is untouched.
- Tests: `tests/test_asymmetric_z_envelope.py` — 3 tests, all PASS.
  Single SimulationApp; flag toggled on cfg at runtime between assertions
  (no rebuild). Run with
  `./isaaclab.sh -p source/.../iris_ma6/tests/test_asymmetric_z_envelope.py`.
- Slice 4 (training A/B) deferred per ticket scope-boundary note.

---

### What

The env currently applies a single per-env scalar `_max_lin_vel` symmetrically to all three velocity action dims at [iris_ma_env6_test.py:740](../../../../iris_ma_env6_test.py#L740):

```python
self.cmd_vel[:, idx, 0:3] = action[:, 0:3] * self._max_lin_vel.unsqueeze(-1)
```

PX4 enforces an **asymmetric** vertical envelope per `MPC_Z_VEL_MAX_DN = 1.5 m/s` (descend) and `MPC_Z_VEL_MAX_UP = 3.0 m/s` (climb), independent of any horizontal velocity cap. At deployment, the policy's z action gets clipped from ±10 m/s (sim) down to [−1.5, +3.0] (real), producing trajectory undershoot and behavior that wasn't trained.

### Why

Two failure modes when this is left unfixed at sim-to-real handoff:

1. **Vertical trajectory undershoot**. Policy commands `vz = -1.0` (full descend, 10 m/s in sim); PX4 delivers −1.5 m/s. Anything timed off the sim's vertical motion budget (intercept geometry, evasive dives, formation altitude transitions) arrives late or short.
2. **Unlearned asymmetry**. The policy can use "rapid descent" as a sim-only tactic — e.g., diving from orbit altitude to intercept altitude in a fraction of a second. The real system can't do that. Closely related: target tracking under rapidly-descending target requires the sim policy to match descent rates that don't exist in deploy.

Additional motivation from the action-jerk analysis: the asymmetric envelope at deploy forces the policy to spend longer in transient regimes (max-rate hold) than it does in sim, where the bounds are looser. The jerk-attitude-coupling problem the user observed may be magnified at deploy because the policy reaches commanded-but-clipped vertical states more often.

### Scope

- Replace `_max_lin_vel: Tensor[N]` (single scalar per env) with:
  - `_max_vel_xy: Tensor[N]` — horizontal velocity cap (curriculum + DR-jitter, current behavior preserved).
  - `_max_vel_z_up: float` (or `Tensor[N]` if per-vehicle jitter wanted) — climb rate cap.
  - `_max_vel_z_dn: float` — descend rate cap.
- Update action scaling: scale `action[:, 0:2]` by `_max_vel_xy`, scale `action[:, 2]` by `_max_vel_z_up` if action[:, 2] > 0 else `_max_vel_z_dn`.
- Defaults from PX4: `MPC_Z_VEL_MAX_UP = 3.0`, `MPC_Z_VEL_MAX_DN = 1.5`. These should NOT randomize (real-world vehicles are tuned to these values; they don't drift).
- Cfg flag `enable_asymmetric_z_envelope: bool = False` for bit-exact t034 reproduction. Default False; new runs opt in.

### Scope boundary

- DO: Apply asymmetric clipping at the env-physics boundary so the policy is trained against the deployment-realistic envelope.
- DO: Keep horizontal cap under curriculum / DR randomization — that's the existing behavior, not the bug.
- DO: Compose with ticket 038 (per-(env, agent) `_max_lin_vel`) — when 038 lands, the horizontal cap becomes `Tensor[N, A]`; the z bounds remain scalar per design.
- DO NOT: Randomize z bounds. They're calibration-set per PX4 vehicle.
- DO NOT: Add z asymmetry to the curriculum (would change difficulty mid-training; not the goal).

### Acceptance criteria

- `enable_asymmetric_z_envelope=False` reproduces t034 dynamics bit-exactly (regression: 1k-step fixed-seed rollout matches existing behavior).
- `enable_asymmetric_z_envelope=True`: `cmd_vel[:, :, 2]` is in `[-1.5, +3.0]` m/s regardless of action / horizontal-cap settings. Verify via teleop or a probe-style scripted command sweep.
- Sim-to-sim validation against PegasusSimulator with the same PX4 params: trained policy's commanded vz trajectory falls within the PX4 envelope without persistent saturation.

### Relation to other tickets

- **Ticket 037 (critic priv obs Phase 1)**: independent. Once 037 ships, this becomes a sim-to-real prerequisite. Phase 1 training in progress today still uses symmetric envelope — that's a known limitation we accept for the A/B vs t034.
- **Ticket 038 (per-(env, agent) max_lin_vel)**: composes. After 038, `_max_vel_xy` is shape `(N, A)` and z bounds remain scalar. Action scaling combines both.
- **Jerk-attitude reanalysis**: the asymmetric envelope may itself contribute to the attitude-destabilization-on-fast-velocity-change problem. Once 039 lands, re-evaluate whether the policy is more or less jerky given the more realistic bounds.

### Implementation effort

~½ day:
- Slice 1: cfg + env attribute additions; flag-gated. Bit-exact when flag=False.
- Slice 2: action scaling update.
- Slice 3: unit test that flag=True produces clipped cmd_vel; regression test that flag=False matches current.
- Slice 4: short training run (or skip — pending Phase 1 + other ticket sequencing).

### References

- PX4 params: `MPC_Z_VEL_MAX_UP`, `MPC_Z_VEL_MAX_DN` ([PX4 User Guide — Multicopter Position Control](https://docs.px4.io/main/en/advanced_config/parameter_reference.html)). User-confirmed values: 3.0 m/s up, 1.5 m/s down for the iris_ma6 deployment vehicle.
- Ticket 037 — action-smoothness diagnostic discussion (where this sim-to-real gap was surfaced).
