## Open Questions — Answered

### Assumptions requiring confirmation

1. **Observation dimension change**: ~~Add detection_confidence?~~ → **No. Do NOT add observation dimensions. Ticket revised.**

2. **Confidence replaces vs. augments bbox_empty**: → **Neither. No confidence field in this ticket. Deferred pending principled model from ticket-009 calibration data.**

3. **Granularity of miss-rate application**: → **Per target per camera per step. Additionally, miss rate is conditioned on background type (sky vs ground) determined by `raycast_mesh` query through the target.**

4. **False positive bbox geometry**: → **Random location and random size.**

5. **False positive interaction with delay system**: → **FPs are treated as true positives — ingested into delay system as normal detections.**

6. **Curriculum gating mechanism**: → **Deferred to ticket-010 design stage. Recommendation: use existing curriculum progress schedule with `fp_fn_start_step`/`fp_fn_end_step` analogous to noise ramp.**

7. **Miss rate during bbox_empty=True**: → **Skip miss rate model. `bbox_empty=True` means no detection was produced (target occluded/OOF/too-small). The miss rate model only applies when the raycaster produces a valid detection.**

8. **Confidence computation inputs**: → **Deferred. Research found: detector confidence is not calibrated probability (it's a ranking metric), no strong RL ablation evidence for confidence-as-observation. Principled model requires ticket-009 calibration data. Not in scope for this ticket.**

9. **Ticket-009 dependency status**: → **Wait for ticket-009. Use placeholder parameters for development. Ticket-009's design should be revised to place noise inside the `detector_replicator` (architectural alignment).**

10. **Per-agent or per-target confidence**: → **Deferred with confidence. Further decisions blocked on "how to model detection confidence" question, which is itself deferred.**

### Architectural decisions requiring human input

11. **Where to apply FP/FN**: → **Single `detector_replicator` submodule inside `bbox_raycaster_v2`. Shares existing mesh handle for background queries. Configurable/toggleable — disabled by default.**

12. **Detection confidence as delay field vs env-level**: → **Deferred. No confidence in this ticket.**

13. **State management for FP/FN sampling**: → **`detector_replicator` is a method/submodule of `bbox_raycaster_v2`, follows the existing `_last_update_time` idempotency guard pattern.**

14. **SKRL config update scope**: → **No SKRL config changes needed. No observation dimension change.**

### Additional decisions from design review (2026-04-08)

15. **Student's t df parameter**: → **df~12-16 is correct (from report-5 with more matches). Findings.md §2 value of df≈4.9 is from an earlier run with fewer matches. Design should use df~12-16.**

16. **Noise scale**: → **~13.4px from findings is authoritative (supersedes design's ~10.5px).**

17. **Noise application location**: → **Inside `detector_replicator` in `bbox_raycaster_v2` (pre-delay). NOT post-delay in `_get_observations()`. The "simpler alternative" from ticket-009 design is revoked — noise and miss rate must be co-located because missed detections should not have noise applied.**

18. **bbox_empty zero-out bug**: → **Existing bug: noise leaks through on `bbox_empty=True` frames. Fix as part of this ticket — zero out bbox values when `bbox_empty=True`.**

19. **Ticket-009 design revision needed**: → **Ticket-009's i_design.md "simpler alternative" (post-delay noise) is incompatible with the replicator architecture. Ticket-009 should revise to place noise inside the replicator, co-located with FN/FP.**

20. **Reward path must use GT bboxes, not replicated bboxes**: → **`bbox_raycaster_v2` outputs both pure GT bboxes and replicated bboxes. GT bboxes go to the reward path (both privileged and perception-aligned). Replicated bboxes go to the observation path only. Rationale: (a) FN frames would zero the reward for a coin flip the policy can't control — adds variance without useful gradient. (b) FP frames would reward centering a phantom target — creates perverse incentives. The replicator is purely an observation-domain transformation; rewards are always grounded in physical truth.**
