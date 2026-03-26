# Workflow Templates

Reusable scaffolds for bootstrapping the harness engineering workflow described in `doc/ai_workflow.md`.

## Quick Start

1. Copy the relevant template(s) to your target directory
2. Replace all `{{PLACEHOLDER}}` tokens with project-specific values
3. Delete any sections marked `[OPTIONAL]` that don't apply

## Template Index

| Template | Purpose | When to Use |
|----------|---------|-------------|
| `PROJECT_CLAUDE.md` | Session workflow protocol | Starting a new project or iteration (e.g., iris_ma7) |
| `ARCHITECTURE.md` | Module dependency graph | Starting a new project with multiple modules |
| `MODULE_CONTEXT.md` | Per-module routing contract (ICM Layer 2) | Adding a new module to an existing project |
| `SPEC.md` | Specification document | Defining what a module should do before/during implementation |
| `module/` | Complete module scaffold | Creating a new sub-module from scratch |
| `doc/active/` | Feature tracking + session log | Starting multi-session tracking for a project |
| `commands/update-progress.md` | `/update-progress` slash command | Setting up end-of-session automation |

## Pattern Selection Guide

| Task | Templates to Use |
|------|-----------------|
| Start a new project iteration | `PROJECT_CLAUDE.md` + `ARCHITECTURE.md` + `doc/active/` + `commands/` |
| Add a new module | `module/` scaffold + `MODULE_CONTEXT.md` + `SPEC.md` |
| Add tests to existing module | `module/tests/` only |
| Define a feature before coding | `SPEC.md` first, then `module/` when ready |
| Wire an existing module into env | None — follow the integration checklist in `ai_workflow.md` Section 5.5 |
| Add parameters to existing module | None — update existing `*_cfg.py` |

## Placeholder Tokens

All templates use `{{TOKEN}}` syntax for project-specific values:

| Token | Description | Example |
|-------|-------------|---------|
| `{{PROJECT_NAME}}` | Project identifier | `iris_ma7` |
| `{{PROJECT_DESCRIPTION}}` | One-line project description | `Multi-agent drone observation environment` |
| `{{ENV_FILE}}` | Main environment filename | `iris_ma_env7.py` |
| `{{ENV_CFG_FILE}}` | Environment config filename | `iris_ma_env7_cfg.py` |
| `{{PROJECT_PATH}}` | Relative path to project root | `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma7` |
| `{{MODULE_NAME}}` | Module identifier (snake_case) | `cbf_safety` |
| `{{ModuleName}}` | Module class name (PascalCase) | `CbfSafety` |
| `{{ModuleNameCfg}}` | Config class name (PascalCase) | `CbfSafetyCfg` |
| `{{MODULE_DESCRIPTION}}` | One-line module description | `Collision avoidance using Control Barrier Functions` |

## After Creating from Template

- **New project**: Update the parent `CLAUDE.md` to point to the new project directory
- **New module**: Update `ARCHITECTURE.md` with the new module and its dependencies
- **New spec**: Add the spec path to the module's `CONTEXT.md` under "Spec"

## Reference

For the rationale behind this workflow, see `doc/ai_workflow.md`.
