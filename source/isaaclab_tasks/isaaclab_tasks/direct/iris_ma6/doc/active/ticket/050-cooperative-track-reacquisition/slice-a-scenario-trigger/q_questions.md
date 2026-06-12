# Stage Q — Open Questions (Slice A) + engineer answers

Slice A scope: raise difficulty so the target leaves individual FOVs (raise ceiling, keep floor
easy); exploit independent per-agent dropout/occlusion to manufacture single-agent loss while a
peer retains the target; instrument track-loss / re-acquisition. Gate: track-loss events occur at
a measurable, non-trivial rate.

## Assumptions requiring confirmation
1. Slice A is the right starting point (not jumping to reward/Slice B). — **CONFIRM**
2. Agent count for the Slice-A dev loop is 2 (minimal hand-off). — **CONFIRM**
3. Baseline = t047 `agent_400000.pt` @ width 256, t047 reward defaults; Slice A adds no reward
   changes. — **CONFIRM** ("flipped weights" = the 6 cfg defaults t047 changed: bbox_center 60→90,
   bbox_size 60→30, triangulation 5→8, action_delta −12→−24, max_log_std 0.7→0.4, entropy schedule).
4. Ships behind default-off feature flags, backward-compatible. — **CONFIRM; flip on per experiment.**
5. Agent proposes the concrete "non-trivial rate" gate threshold. — **CONFIRM.**

## Architectural decisions requiring human input
6. Loss mechanism: (a) scripted FOV-exit, (b) detector-dropout, (c) occlusion, or combo.
   — **(a)+(b). (a) splits: far-target→zoom-in vs edge-of-frustum→gimbal slew; distinguish them.**
7. Guarantee single-agent (not team-wide) loss? — **No hard guarantee; independence + measure.**
8. Coordinate/merge with t045/t046 difficulty-revert? — **(a) Slice A owns it; no merge.**
9. Curriculum ceiling-vs-floor / is detector-dropout alone enough?
   — **Decide empirically after inspecting current track-loss impl (resolved in R).**
10. Where instrumentation lives. — **Both TensorBoard (training) + eval metrics. Define metric carefully.**
11. Re-acquisition metric definition (filter flicker; capture mid-episode regain, not just cold).
    — **Unified peer-assisted-deficit metric (see ticket.md Q10/11).**
13. How limiting is the NaN; fallback options. — **No NaN (verified R); detect deficit; privileged
    critic optional-early; EKF→Slice C. Emergent behavior: naive "re-point along peer bearing"
    likely; "close-in-to-peer" uncertain but is what the Slice-B covariance reward pays for —
    instrument inter-agent distance + bearing-alignment during recovery to disambiguate.**
