## Ticket: False positive/negative detection model — probabilistic detection in bbox pipeline

**What**: Add a probabilistic detection model to the bbox pipeline: false negatives (miss rate as a function of target size/distance/occlusion), false positives (spurious detections), and detection confidence as an observation signal. Currently the raycaster produces deterministic, all-or-nothing detections.

**Why**: Real object detectors (YOLO) have size-dependent miss rates (small/distant targets are missed more often), occasional false positives (clutter, other objects), and confidence scores that the policy could use to modulate its behavior. iris_ma6's binary `bbox_empty` flag doesn't capture this — a target at the edge of detection range flickers between detected/not-detected with no intermediate signal. Adding confidence and probabilistic miss/FP lets the policy learn to handle detector uncertainty.

**Evidence**:
- Ticket-005 §1.4: false positives, false negatives (miss rate), detection confidence, and partial detection are all listed as gaps
- Current `bbox_empty` is binary: set for occluded/OOF/too-small targets, deterministic
- `partial_detection_allowed=False` in bbox raycaster — real detectors have partial boxes

**Scope**:
1. **Miss rate model**: Probability of missing a valid target, as a function of bbox size (pixels). Small targets → higher miss probability. Parameterized: `p_miss = sigmoid(a * (size_threshold - bbox_size))` or lookup table from ticket-009 calibration data
2. **False positive model**: Per-step probability of generating a spurious bbox at a random location. Low rate (e.g., 0.1-1% per step). Parameterized by `p_fp`
3. **Detection confidence**: Add a continuous `detection_confidence` field (0-1) to observation, replacing or augmenting the binary `bbox_empty`. Confidence = f(bbox_size, occlusion_fraction, noise). Requires observation dimension change
4. **Integration**: Apply miss/FP/confidence in the bbox post-processing path (after raycaster, before delay system ingestion). Curriculum-gated (disabled at early training, ramped in with observability phases)

**Scope boundary**:
- Do NOT modify the raycaster core (mesh intersection, occlusion) — only post-processing
- Do NOT plug YOLO into the training loop
- Do NOT change the delay system architecture — confidence is an additional field
- Calibration data comes from ticket-009; this ticket implements the model, not the data collection
- Partial detection (non-axis-aligned boxes, truncated boxes) is deferred — binary miss is sufficient for V1

**Affected modules**:
- `bbox_raycaster_v2/` — post-processing: apply miss/FP after raycasting
- `delay_system_v3/` — add `detection_confidence` field to AgentStates
- `iris_ma_env6_test.py` — observation dimension change (add confidence), curriculum gating
- `iris_ma_env6_test_cfg.py` — FP/FN model parameters

**Dependencies**:
- Ticket-009 (bbox noise model) provides the calibrated miss rate and FP rate parameters. Can be implemented with placeholder parameters before ticket-009 completes.

**Acceptance criteria**:
1. Miss rate applied as function of bbox size (configurable parameters)
2. False positive detections generated at configurable rate
3. `detection_confidence` in observation vector (continuous 0-1)
4. Curriculum-gated: no FP/FN at early training, full model at late training
5. Observation dimension documented and SKRL config updated

**Flow**: Full QRISPY
