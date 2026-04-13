## Ticket: Model fitting and sim-real residual analysis (A6)

**What**: Implement an offline analysis pipeline that compares Isaac Sim controller/physics response against real flight data, tunes nominal sim parameters, and quantifies the sim-real residual to set principled DR ranges.

**Why**: The sim-to-real priority doc (ticket-021) identifies system identification as the highest-impact item after wind modeling. Zhao 2024 showed SysID-only (0.028m error) outperforms SysID+30%DR (0.066m). The residual after model fitting determines what DR must cover — instead of guessing ±20% on everything, DR ranges should reflect actual model uncertainty.

**Depends on**: Flight data from ticket-023 (A2 velocity steps) and ticket-024 (A3 attitude steps, A4 yaw rate). Cannot produce meaningful results until Phase 2-3 flight data is collected.

**Pipeline**:

### Step 1: Data ingestion
- Parse PX4 `.ulg` logs from A1-A4 flights (using `pyulog`)
- Extract time-synchronized: position, velocity, attitude (quat), angular velocity, motor commands/RPM, velocity setpoints, attitude setpoints, rate setpoints
- Parse ROS2 bags for gimbal joint data (if needed)
- Align timestamps and resample to common rate

### Step 2: Sim replay
- Replay the same velocity commands through the iris_ma6 controller + Isaac Sim physics
- Use the existing DroneController with nominal parameters
- Match initial conditions from flight data
- Record sim state trajectory at same rate as real data

### Step 3: Residual computation
- Per-channel residual: `r(t) = state_real(t) - state_sim(t)`
- Metrics: RMSE, max error, correlation, per-channel statistics
- Frequency-domain analysis: where does the residual have power? (low-freq = model bias, high-freq = noise/unmodeled dynamics)
- Visualization: overlay plots of real vs. sim per channel

### Step 4: Parameter tuning
- Optimize nominal parameters to minimize residual
- Parameters to tune (mapped from interception project analysis):
  - **Directly tunable**: Thrust coefficient (`k_f` ↔ interception `k_w`), drag (`C_d`, `A`), motor time constant (`tau_motor`)
  - **Controller gains**: velocity PID, attitude P, rate PID — compare PX4 actual (F1) vs. sim nominal
  - **Aerodynamics**: fit `C_d * A` from steady-state drag data (B1, extracted from A2)
- Method: Scipy `minimize` or `least_squares` (same approach as interception project sysid.py)
- Report: nominal value, fitted value, confidence interval, residual reduction

### Step 5: DR range recommendation
- For each parameter: DR range = fitted_value ± k * residual_std (k=2 for 95% coverage)
- Compare against current DR ranges (mass ±10%, gains ±20%, etc.)
- Flag parameters where current DR is too narrow (doesn't cover observed variation) or too wide (wastes training on unrealistic configs)

**Relationship to interception project sysid**:
- Their model is at a different abstraction level (direct motor RPM → forces via 23 empirical coefficients, no cascade controller). Not directly reusable.
- Their methodology IS reusable: OLS regression for linear parameters, nonlinear optimization for motor dynamics, 95% confidence intervals via Student-t distribution.
- Parameters that map between models:
  - `k_w` ↔ `k_f` (thrust coefficient)
  - `k_x, k_y` ↔ `AerodynamicsCfg.C_d, A` (drag)
  - `tau_motor` ↔ `MotorDynamicsCfg.tau`
  - `w_min, w_max` ↔ `MotorDynamicsCfg.omega_min/max`
- Their 16 moment coefficients (k_p1-k_r8) are NOT portable — iris_ma6 uses inertia matrix + mixer geometry instead

**Scope boundary**:
- DO: Implement data ingestion (`.ulg` parsing + resampling)
- DO: Implement sim replay (run iris_ma6 controller with recorded commands)
- DO: Implement residual computation and visualization
- DO: Implement parameter fitting for thrust, drag, motor tau
- DO: Output DR range recommendations
- DO NOT: Implement the flight test scripts (tickets 023, 024)
- DO NOT: Modify iris_ma6 environment or controller code (this is analysis only)
- DO NOT: Implement residual dynamics learning (R1 in sim2real_priority.md — that's post-deployment)

**Affected files**:
- NEW: Analysis script(s) in `offboard_py/offboard_py/` or a new `analysis/` directory
- NEW: Visualization/report output directory
- Reference: `/home/usrg/source/Aerial_To_Aerial_Interception/scripts/system_identification/sysid.py` for methodology

**Acceptance criteria**:
- Parses `.ulg` flight logs and extracts all required channels
- Produces overlay plots (real vs. sim) for position, velocity, attitude
- Reports per-channel RMSE before and after parameter tuning
- Outputs recommended DR ranges with justification
- Confidence intervals on fitted parameters

**Reference**: [checklist.md](../021-sim2real-measurement-checklist/checklist.md) item A6; [sim2real_priority.md](../021-sim2real-measurement-checklist/sim2real_priority.md) item H2

**Flow**: Full QRISPY (analysis methodology needs validation)
