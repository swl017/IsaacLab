## Ticket: False positive/negative detection model — detector replicator in bbox pipeline

**What**: Add a `detector_replicator` submodule inside `bbox_raycaster_v2` that applies probabilistic false negatives (miss rate conditioned on target size and background type) and false positives (spurious detections with random location and size). The replicator replaces the current deterministic, all-or-nothing detection with a stochastic model calibrated from real YOLO data. No observation dimension change.

**Why**: Real object detectors (YOLO) have size-dependent miss rates (small/distant targets are missed more often) and background-dependent miss rates (ground clutter +17.5% miss penalty vs sky). They also produce occasional false positives from scene clutter. iris_ma6's current binary `bbox_empty` flag is deterministic — a target either is or isn't detected with no stochasticity. Adding probabilistic FN/FP lets the policy learn to handle detector unreliability, improving sim-to-real transfer.

**Evidence**:
- Ticket-005 §1.4: false positives, false negatives (miss rate) listed as gaps
- Ticket-010 findings.md: calibrated miss rates from YOLO — sky 64.3%, ground 81.8%, +17.5% background penalty
- Ticket-010 findings.md §3: miss rate vs target size curves (sky vs ground), critical zone 10-60px
- Ticket-010 findings.md §4: FP rate ~0.49/frame (inflated by other-agent detections, needs re-measurement)
- Current `bbox_empty` is binary: set for occluded/OOF/too-small targets, deterministic

**Scope**:
1. **Miss rate model**: Probability of missing a valid target, conditioned on:
   - Target bbox size (pixels) — primary variable (accounts for zoom)
   - Background type (sky vs ground) — determined by `raycast_mesh` query through target past the target mesh
   - Parameterized as dual sigmoids: `p_miss = sigmoid(a * (threshold - bbox_size))` with separate (a, threshold) for sky and ground backgrounds
   - When missed: zero out bbox, set `bbox_empty=True`
   - Skip miss rate when `bbox_empty` is already True (target already not detected)
2. **False positive model**: Per-step probability of generating a spurious bbox with random location and random size. FPs are treated as true positive detections (ingested into delay system as normal). Parameterized by `p_fp`
3. **Bug fix**: Zero out bbox values when `bbox_empty=True` (currently noisy bboxes leak through on empty frames)
4. **Dual output**: `bbox_raycaster_v2` outputs both pure GT bboxes and replicated bboxes. The GT bboxes are used for rewards (privileged and perception-aligned — rewards must never be corrupted by detector stochasticity). The replicated bboxes are used for observations only. This ensures the policy is *guided* by physical truth but learns to *cope* with FN/FP through its observations.
5. **Integration**: `detector_replicator` lives inside `bbox_raycaster_v2` as a configurable/toggleable post-processing step. Applied after raycasting, before delay system ingestion. Curriculum-gated (disabled at early training, ramped in)

**Scope boundary**:
- Do NOT modify the raycaster core (mesh intersection, occlusion) — only post-processing
- Do NOT plug YOLO into the training loop
- Do NOT add detection confidence to observations (deferred — requires principled model from ticket-009 calibration data)
- Do NOT change observation dimensions or SKRL configs
- Do NOT change the delay system architecture
- Calibration data comes from ticket-009/010 findings; this ticket implements the model
- Partial detection (non-axis-aligned boxes, truncated boxes) is deferred

**Affected modules**:
- `bbox_raycaster_v2/` — new `detector_replicator` submodule (post-processing, uses existing mesh handle for background queries)
- `iris_ma_env6_test_cfg.py` — detector replicator config parameters
- `curriculum_cfg.py` — FP/FN curriculum ramp parameters

**Dependencies**:
- Ticket-009 (bbox noise model): provides the calibrated noise parameters and establishes the `detector_replicator` as the architectural home for all detector modeling (noise + FN + FP). Ticket-009's design should be revised to place noise inside the replicator rather than post-delay in `_get_observations()`.
- Ticket-010 findings.md: provides calibrated miss rate curves and FP statistics

**Acceptance criteria**:
1. Miss rate applied as function of bbox size with sky/ground conditioning (configurable parameters)
2. Background type determined by `raycast_mesh` query (sky = ray misses all geometry)
3. False positive detections generated at configurable rate with random location/size
4. FPs treated as real detections through the delay pipeline
5. `bbox_empty=True` frames have zeroed bbox values (bug fix)
6. Curriculum-gated: no FP/FN at early training, full model at late training
7. Detector replicator is toggleable (disabled by default, enabled via config)

**Flow**: Full QRISPY
