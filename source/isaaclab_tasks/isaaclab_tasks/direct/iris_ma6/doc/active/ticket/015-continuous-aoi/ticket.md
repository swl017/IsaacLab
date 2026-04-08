## Ticket: Continuous AoI — remove step-quantized timestamps

**What**: Make Age-of-Information (AoI) continuous rather than quantized by simulation step length. Currently timestamps snap to discrete step boundaries, creating staircase AoI values. The policy should see smooth, continuous AoI that reflects actual elapsed time.

**Why**: Quantized AoI loses sub-step timing information and creates artificial discontinuities. Continuous AoI is more physically accurate (real sensors report at arbitrary times, not aligned to a global clock) and provides smoother gradients for policy learning.

**Scope boundary**:
- Do NOT change the delay pipeline stage order or ringbuffer architecture
- Do NOT change the dropout or staleness mechanisms
- Focus on timestamp representation and AoI computation only

**Affected modules**:
- `delay_system_v3/` — timestamp handling in pipeline stages
- Possibly `iris_ma_env6_test.py` — AoI observation computation

**Flow**: TBD (needs scoping — likely Light or Direct)
