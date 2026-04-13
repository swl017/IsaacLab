## Ticket: Attitude and yaw rate test scripts (A3, A4)

**What**: Implement `offboard_attitude_step.py` (A3) and `offboard_yaw_rate.py` (A4) using the flight test harness from ticket-023.

**Why**: Attitude step response (A3) validates attitude controller time constants and rate limits in sim. Yaw rate response (A4) validates yaw dynamics which are typically slower and less damped than roll/pitch. Both are needed for principled controller gain DR ranges.

**Depends on**: Ticket-023 (flight test harness must exist)

**Prerequisites for A3**: A1 hover thrust data — the attitude step test must command a fixed thrust (hover throttle) while commanding attitude setpoints. The hover throttle value comes from A1 data analysis. This should be a configurable parameter.

**Scripts**:

### offboard_attitude_step.py (A3)
- Commands attitude setpoints (roll or pitch) at configurable angles (10, 20, 30 deg) while maintaining hover thrust
- Uses `mavros/setpoint_raw/attitude` (AttitudeTarget) with type mask for attitude + thrust
- Two modes:
  - **Scripted (preferred)**: Offboard attitude commands with hover thrust from A1
  - **Manual fallback**: Print instructions for pilot to use stabilized mode with stick inputs (no script needed, just a procedure doc)
- Sequence: 0° → +angle → 0° → -angle → 0° (returns to level)
- 5 repetitions per angle per axis

### offboard_yaw_rate.py (A4)
- Commands yaw rate setpoints at configurable rates (30, 60, 90 deg/s)
- Uses velocity setpoint with yaw_rate field (simpler than attitude mode)
- Sequence: 0 → +rate (hold 3s) → 0 (hold 3s) → -rate (hold 3s) → 0
- 5 repetitions per rate

**Scope boundary**:
- DO: Implement both scripts using ticket-023 harness
- DO: Add entry points to `setup.py`
- DO: Make hover thrust a configurable parameter (from A1 data)
- DO NOT: Implement the harness (ticket-023)
- DO NOT: Analyze the collected data (ticket-027)

**Affected files**:
- NEW: `offboard_py/offboard_py/offboard_attitude_step.py`
- NEW: `offboard_py/offboard_py/offboard_yaw_rate.py`
- MODIFY: `offboard_py/setup.py` — add entry points

**Acceptance criteria**:
- A3: Attitude setpoints are commanded at correct angles with configurable hover thrust
- A4: Yaw rate setpoints are commanded at correct rates
- Both: Use harness for mode detection, bag recording, safety abort
- Both: CSV output includes attitude/rate setpoints alongside state
- Both: Tested in PX4 SITL before real hardware

**Reference**: [checklist.md](../021-sim2real-measurement-checklist/checklist.md) items A3, A4

**Flow**: Light (scope is clear, harness is done)
