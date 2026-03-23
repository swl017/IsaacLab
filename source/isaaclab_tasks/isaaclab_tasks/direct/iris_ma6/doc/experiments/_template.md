# Experiment: <experiment_name>

**Commit**: `<full_commit_hash>`
**Date**: YYYY-MM-DD
**Experiment ID**: `<registry_id>` (from experiment_registry.py)
**Base**: <what this builds on — prior experiment hash or "baseline">

---

## 1. Hypothesis
What change was made and what was expected to happen.

## 2. Configuration Delta
Key overrides from the base config (not the full config — only what changed).

```python
# Example:
# curriculum.velocity_ramp.end_step: 5e6 → 8e6
# reward.triangulation_weight: 0.3 → 0.5
```

## 3. Results

### 3.1 Training Curves
Summary of key metrics at convergence (or at failure point):
- Episode reward (mean, std)
- Episode length
- Policy loss / value loss
- Key reward components

### 3.2 Behavior Observations
What the agents actually did — qualitative notes from video/teleoperation review.

## 4. Analysis / Open question
Why did it work / not work? What does this tell us about the system?

## 5. Decision / Todo
What to do next based on these results. Reference the next experiment if one was created.
