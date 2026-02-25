# TODOs for iris_ma5

## Goal for iris_ma5
- [X] Generalize to 3 or more agents
- [ ] Investigate and add more randomization(control gain, mass, target size, etc.)
- [X] Add stacked observations for MLP case
- [ ] Introduce false negative and false positive to the detection pipeline
  - **Root cause analysis (2026-02-25):** Current hold-last-value dropout semantics model packet
    jitter, not communication loss. When comm drops, `dropout_held_data` (initialized to the
    real step-1 sensor snapshot) is returned forever — the frozen bbox still passes
    `validate_bbox()` and the frozen (position, orientation, bbox) triplet is self-consistent,
    so triangulation RMSE is unaffected even at 100% dropout.
  - **Fix:** On dropout, set bbox to `[0, 0, 0, 0]` (or any value outside `[0.01, 0.99]` image
    bounds) instead of holding the last value. This makes `validate_bbox` return False, signals
    "no measurement" to the triangulation pipeline, and correctly degrades TriValid at high
    dropout rates.
  - **Why RMSE doesn't degrade currently:**
    1. `dropout_held_data = None` at construction → first `process()` call (at sim step 1,
       after real sensor data is available) returns FRESH data and initializes held state to
       real step-1 values — NOT zeros.
    2. Frozen step-1 snapshot has a valid bbox and self-consistent ray origin/direction.
    3. Tracking task geometry: drones orbit the target, so the step-1 ray approximately points
       at the target throughout the episode → triangulation error growth is gradual and stays
       within the baseline noise floor.
    4. `NUM_EPISODES=1` in the performance envelope sweep → post-reset "zeros" bug (where
       `dropout_held_data[env_ids]=0.0` after reset would make the other agent invisible)
       never triggers, since most envs complete exactly one episode without mid-episode reset.