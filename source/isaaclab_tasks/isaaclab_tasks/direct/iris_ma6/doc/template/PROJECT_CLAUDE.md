# {{PROJECT_NAME}} — {{PROJECT_DESCRIPTION}}

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for module dependency graph and data flow.

## Session Workflow Protocol

At the **START** of each session:
- Read `doc/active/feature_list.json` and `doc/active/progress.txt`
- Read `ARCHITECTURE.md` for module boundaries

At the **END** of each session:
- Append to `doc/active/progress.txt`: what was done, what's next
- Update `doc/active/feature_list.json` if any feature status changed
- Update the relevant module's `CONTEXT.md` if its interface (inputs, outputs, dependencies) changed
- Update `ARCHITECTURE.md` if module dependencies changed

## Module Navigation
Each sub-module has a `CONTEXT.md` describing its purpose, inputs, outputs, dependencies, and key files. Read the relevant `CONTEXT.md` before working on a module.

## Specs
Authoritative specifications live in `doc/*_spec.md`. These define *what* to build. Do not duplicate spec content elsewhere.

## Folder Semantics
```
{{PROJECT_NAME}}/
├── ARCHITECTURE.md       # Module dependency graph
├── CLAUDE.md             # This file (session workflow)
├── doc/
│   ├── active/           # Multi-session tracking (feature_list.json, progress.txt)
│   ├── backlog/          # Low-urgency future tasks and research
│   └── *_spec.md         # Authoritative specs
├── <module>/
│   ├── CONTEXT.md        # Module routing contract
│   └── tests/            # Per-module test suite
└── {{ENV_FILE}}          # Main environment (integrates all modules)
```

## Rationale
For background on why this workflow exists, see `doc/harness_rationale.md`. For the stage-gated development workflow (QRISPY), see `doc/qrispy_workflow.md`.
