# Experiment Analyses

## Purpose
Post-training analysis of individual experiment runs. Each file documents what was learned from a training run: metrics, failure modes, hyperparameter insights, and decisions for the next iteration.

## Inputs
- Training logs (TensorBoard, checkpoints) from `logs/skrl/`
- Experiment definition from `experiments/experiment_registry.py`
- Curriculum and reward specs from `doc/*_spec.md`

## Outputs
- Per-run analysis files (markdown) with metrics, plots, and conclusions

## Naming Convention
Files are named by commit hash prefix + experiment name, matching the git commit that defined the run:
```
<commit_hash_prefix>_<experiment_name>.md
```
Example: `e36e8ca123_curriculum_update.md`

This mirrors the convention used in commit messages (`Running <hash>_<name> experiment`).

## Template
Use `_template.md` as the starting point for each new analysis.

## Key Distinction
- `doc/*_spec.md` → defines *what to build*
- `doc/active/progress.txt` → records *what was done* per session
- `doc/experiments/*.md` → records *what was learned* per training run
