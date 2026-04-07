## Ticket: Bbox noise model — calibrate raycaster noise from real detector statistics

**What**: Calibrate the bbox raycaster's noise model (`bbox_std`, miss rate, localization bias) from real YOLO detector statistics, so that the geometric raycaster's output distribution during training matches what the policy will see at deployment. Apply calibrated noise inside a `detector_replicator` submodule in `bbox_raycaster_v2`.

**Why**: iris_ma6 uses a geometric raycaster for bbox extraction (fast, deterministic, GPU-accelerated). The sim-to-real gap is then "how different is the real detector's output distribution from the geometric model?" Currently `bbox_std=7.0` pixels is a hand-tuned constant. Real detectors have size-dependent noise (small targets → noisier), confidence-dependent localization error, and systematic biases. Without calibration, the policy may overfit to the raycaster's noise profile.

**Evidence**:
- Ticket-005 §1.4: bbox noise is covered (`bbox_std=7.0`), but miss rate, FP rate, confidence, and partial detection are gaps
- Ticket-005 §1.4 assessment: "gaps need to be bridged by a bbox noise model calibrated from real detector statistics"
- Current NoiseCfg in delay system: `bbox_center_std`, `bbox_size_std` — both fixed scalars, not dependent on target size or distance

**Scope**:
1. **Data collection**: Run YOLO detector on Isaac Sim camera feed (calibration run, 64 envs). Record: GT bbox (from raycaster), detected bbox (from YOLO), confidence, miss/hit per frame. **Status: DONE** — calibration scripts completed, findings in `findings.md`
2. **Statistics extraction**: Compute per-frame localization error as function of target size. Fit Student's t noise model. Compute miss rate, FP rate. **Status: DONE** — parameters fitted
3. **Detector replicator**: Apply calibrated Student's t noise inside a `detector_replicator` submodule in `bbox_raycaster_v2`. The replicator is the single architectural home for all detector modeling (noise + FN + FP from ticket-010). `bbox_raycaster_v2` outputs dual bboxes: pure GT for rewards, replicated (noisy) for observations. Delay system `bbox_std` set to 0 when replicator is enabled.
4. **Validation**: Compare raycaster + calibrated noise vs YOLO output distribution (histogram overlay)

**Scope boundary**:
- Do NOT plug YOLO into the training loop (too slow — see ticket-005 §1.4 analysis)
- Do NOT modify the raycaster core (mesh intersection, occlusion) — only post-processing
- Do NOT add detection confidence to observations (deferred)
- Do NOT apply noise in the delay system — noise moves to the detector replicator (pre-delay)
- FN injection and FP injection are ticket-010 scope — this ticket only implements localization noise
- Data collection uses Isaac Sim (sim-to-sim), not real drone footage (sim-to-real calibration is a separate step)

**Affected modules**:
- `bbox_raycaster_v2/` — new `detector_replicator` submodule for calibrated noise application; dual output (GT + replicated bboxes)
- `iris_ma_env6_test_cfg.py` — `CalibratedBBoxNoiseCfg` with JSON path and toggles
- `iris_ma_env6_test.py` — wire dual bboxes: GT to reward path, replicated to observation path; set delay system `bbox_std=0` when replicator enabled

**Dependencies**:
- Ticket-010 (FP/FN model) will extend the `detector_replicator` with miss rate and false positive injection. This ticket establishes the replicator architecture; ticket-010 adds to it.

**Acceptance criteria**:
1. YOLO vs raycaster comparison dataset collected (>1000 frames) — **DONE**
2. Noise model parameters fitted with documented methodology — **DONE**
3. `detector_replicator` in `bbox_raycaster_v2` applies Student's t noise with calibrated parameters
4. `bbox_raycaster_v2` outputs dual bboxes: GT (clean) for rewards, replicated (noisy) for observations
5. Delay system `bbox_std=0` when replicator enabled
6. Histogram comparison shows raycaster+noise matches YOLO output distribution

**Flow**: Full QRISPY
