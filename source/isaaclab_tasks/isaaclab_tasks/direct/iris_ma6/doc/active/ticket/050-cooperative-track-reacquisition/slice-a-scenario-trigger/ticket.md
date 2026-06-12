# Ticket 050 / Slice A — Make the scenario demand cooperation (scenario + instrumentation)

**Parent:** [../ticket.md](../ticket.md) (050 — Cooperative track re-acquisition, research arc)
**Status:** In QRISPY (Q done, R done, I in progress)
**Created:** 2026-06-12
**Flow:** Full QRISPY

## What
The current policy never experiences single-agent track-loss events in training (cause C1), so
there is near-zero gradient for "recover a lost track using peer information." Slice A makes the
scenario *manufacture* single-agent loss (one agent loses the target while a peer retains it) and
instruments track-loss / re-acquisition as first-class metrics — the prerequisite that B/C/D build on.

## Why
C1 is load-bearing: until track-loss events occur at a measurable, non-trivial rate, no reward
(Slice B) or architecture (Slice D) can teach recovery. This is the prerequisite slice of the
journal-contribution arc.

## Scope boundary (confirmed with engineer 2026-06-12)
- IN: harder target behaviour + initial conditions only (raise curriculum *ceiling* via `*_max`,
  keep `*_min` floor easy); reuse independent per-agent detector-miss / dropout to manufacture
  single-agent loss; track-loss / re-acquisition instrumentation (TensorBoard during training +
  eval-time metrics).
- OUT: `max_lin_vel` / slew envelope (sysid lock). Reward changes (Slice B). Explicit peer-bearing
  channel / learned observer (Slice C). Architecture (Slice D).
- Slice A owns the t045/t046 difficulty-revert entirely (no sibling coordination) — engineer #8(a).
- Ships behind default-OFF feature flags (bit-exact backward-compat); flip defaults ON only after
  each experiment confirms.

## Stage-Q answers (engineer, 2026-06-12)
1. Slice A is the start; B/C/D future children. CONFIRM.
2. Develop with 2 agents (minimal hand-off). CONFIRM.
3. Baseline = t047 `agent_400000.pt`, width 256, t047 reward defaults; Slice A adds no reward changes. CONFIRM.
4. Default-off flags; flip on per confirmed experiment. CONFIRM.
5. Agent proposes the "non-trivial rate" gate threshold. CONFIRM.
6. Loss mechanism = (a) FOV-exit + (b) detector-dropout. (a) splits into far-target (needs zoom-in)
   and edge-of-frustum (needs gimbal slew); tag both.
7. No hard guarantee of single-agent-only loss needed — independent per-agent dropout yields
   P(exactly one lost)=2p(1-p); measure rather than force. Keep dropout independent (not shared).
8. (a) Slice A owns the difficulty-revert; no merge with a t045/t046 sibling.
9. Decide detector-dropout-alone sufficiency empirically after instrumenting (resolved in R).
10/11. Re-acquisition metric = "peer-assisted deficit interval" (agent i d=0 while a peer d=1),
   onset tagged cold (t=0) | mid-loss (1→0, sub-tagged far/edge/dropout); success = i recovers and
   holds ≥ τ_hold; deficit counts only if length ≥ τ_min (flicker filter). Captures both cold
   acquisition and mid-episode re-acquisition.
13. Single-agent loss does NOT NaN (obs tri tail is zeroed + −1.0 std sentinel — verified in R).
   Slice A minimal: ensure policy can detect the deficit. Privileged-critic GT-target (cheap,
   CTDE plumbing exists) flagged as optional early stabilizer; bearing-only EKF deferred to C.
