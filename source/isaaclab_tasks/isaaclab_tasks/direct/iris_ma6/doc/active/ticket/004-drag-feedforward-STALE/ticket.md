## Ticket: Velocity-dependent drag feedforward in velocity controller

**What**: Add a drag feedforward term to the velocity controller so the PI integral only handles model error, not the entire aerodynamic drag force. At high speeds (10 m/s), the integral term alone cannot compensate drag — causing permanent steady-state velocity error even with high Ki and raised integral limits.

**Why**: Ticket-003 (controller tuning) revealed that with aerodynamic drag enabled (level 1+), the velocity controller has ~0.5-1.0 m/s steady-state error at 10 m/s. The root cause is structural: `a_integral_max = Ki × integral_limit`. Even with Ki=2.0 and limit=8.0, the integral saturates before fully compensating drag at high speed. PX4 solves this with `MPC_XY_VEL_D_ACC` — a velocity-dependent feedforward that estimates `a_ff = Cd_est × v²/m` and adds it directly to the desired acceleration, so the integral only corrects for model mismatch.

**Evidence**:
- Ticket-003 tuning runs with `--aero-level 1`: best trials show 0.5-1.0 m/s SS error at 10 m/s
- Drag force at 10 m/s: `F = 0.5 × 1.225 × 1.0 × 0.1 × 100 = 6.13 N`, requiring `a = 3.78 m/s²` integral compensation
- Ki_vel range [0.3, 2.0] with integral_limit=8.0 can theoretically cover this, but in practice the integral winds up slowly and interacts with the attitude loop, causing oscillation before reaching steady-state
- PX4 v1.14+ uses `MPC_XY_VEL_D_ACC` feedforward to avoid this problem entirely

**Scope**:
- Add configurable drag feedforward to `VelocityController.compute_control()`:
  `a_ff = drag_ff_gain × v_current² / mass` (applied in the velocity direction)
- Add `drag_ff_gain` field to `VelocityControllerCfg` (default: estimated from `AerodynamicsCfg.C_d × A × rho / 2`)
- Add `drag_ff_gain` to the tuner's `ParameterSet` and search range
- Validate: 10 m/s SS error should drop below 0.2 m/s with correct feedforward gain

**Scope boundary**:
- Do NOT change the cascade architecture (keep 4-loop)
- Do NOT add model-predictive control or other advanced schemes
- Do NOT modify the aerodynamics model itself
- Do NOT change the RL policy or reward function

**Affected modules**:
- `controller/velocity_controller.py` — add feedforward term in `compute_control()`
- `controller/velocity_controller_cfg.py` — add `drag_ff_gain` field
- `controller/tuning/auto_tune.py` — add `drag_ff_gain` to `ParameterSet` and search

**Key references**:
- Velocity controller: `controller/velocity_controller.py:79-155` (compute_control)
- Aerodynamics config: `controller/aerodynamics_cfg.py` (C_d=1.0, A=0.1, rho=1.225)
- PX4 reference: `MPC_XY_VEL_D_ACC` parameter in mc_pos_control
- Ticket-003 tuning results: `controller/tuning/step_result_aero1/`, `step_result_aero3-2/`

**Acceptance criteria**:
1. 10 m/s velocity SS error < 0.5 m/s with drag feedforward enabled
2. No degradation in 5 m/s tracking or hover performance
3. Tuner can search over `drag_ff_gain` alongside existing PID gains

**Flow**: Light
