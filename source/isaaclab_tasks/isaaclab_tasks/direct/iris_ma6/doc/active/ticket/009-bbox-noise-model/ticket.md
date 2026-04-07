## Ticket: Bbox noise model — calibrate raycaster noise from real detector statistics

**What**: Calibrate the bbox raycaster's noise model (`bbox_std`, miss rate, localization bias) from real YOLO detector statistics, so that the geometric raycaster's output distribution during training matches what the policy will see at deployment.

**Why**: iris_ma6 uses a geometric raycaster for bbox extraction (fast, deterministic, GPU-accelerated). The sim-to-real gap is then "how different is the real detector's output distribution from the geometric model?" Currently `bbox_std=7.0` pixels is a hand-tuned constant. Real detectors have size-dependent noise (small targets → noisier), confidence-dependent localization error, and systematic biases. Without calibration, the policy may overfit to the raycaster's noise profile.

**Evidence**:
- Ticket-005 §1.4: bbox noise is covered (`bbox_std=7.0`), but miss rate, FP rate, confidence, and partial detection are gaps
- Ticket-005 §1.4 assessment: "gaps need to be bridged by a bbox noise model calibrated from real detector statistics"
- Current NoiseCfg in delay system: `bbox_center_std`, `bbox_size_std` — both fixed scalars, not dependent on target size or distance

**Scope**:
1. **Data collection**: Run YOLO detector on PegasusSimulator camera feed (sim-to-sim validation, 1-3 envs). Record: GT bbox (from raycaster), detected bbox (from YOLO), confidence, miss/hit per frame
2. **Statistics extraction**: Compute per-frame: localization error (center, size) as function of target size, distance, and confidence. Compute miss rate, FP rate. Fit parametric noise model (e.g., noise_std = a / target_size_px + b)
3. **NoiseCfg update**: Update delay system noise parameters with calibrated values. Add size-dependent noise scaling if the data warrants it
4. **Validation**: Compare raycaster + calibrated noise vs YOLO output distribution (histogram overlay)

**Scope boundary**:
- Do NOT plug YOLO into the training loop (too slow — see ticket-005 §1.4 analysis)
- Do NOT modify the raycaster itself — only its noise post-processing
- Do NOT add detection confidence to observations (that's ticket-010)
- Data collection uses PegasusSimulator (sim-to-sim), not real drone footage (sim-to-real calibration is a separate step)

**Affected modules**:
- `delay_system_v3/` — NoiseCfg, noise application in delay pipeline
- `bbox_raycaster_v2/` — noise parameters
- PegasusSimulator camera feed + YOLO inference (data collection only)

**Acceptance criteria**:
1. YOLO vs raycaster comparison dataset collected (>1000 frames)
2. Noise model parameters fitted with documented methodology
3. NoiseCfg updated with calibrated values
4. Histogram comparison shows raycaster+noise matches YOLO output distribution

**Flow**: Full QRISPY
