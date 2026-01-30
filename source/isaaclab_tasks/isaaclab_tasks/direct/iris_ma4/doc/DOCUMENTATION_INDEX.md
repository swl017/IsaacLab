# IRIS MA3 Environment Documentation Index

This directory contains documentation for the `iris_ma3` multi-agent drone environment, including integration plans, architecture decisions, and implementation guides.

## Document Catalog

### Integration Plans

| Document | Status | Description |
|----------|--------|-------------|
| [DELAY_SYSTEM_INTEGRATION_PLAN.md](DELAY_SYSTEM_INTEGRATION_PLAN.md) | **In Progress** | Plan for replacing `delayed_states` module with `delay_system` v2.2 in `iris_ma_env3.py` |

### Architecture Documents

| Document | Location | Description |
|----------|----------|-------------|
| delay_system README | [delay_system/README.md](../delay_system/README.md) | Overview of delay system v2.2, quick start, API usage |
| delay_system API Reference | [delay_system/API_REFERENCE.md](../delay_system/API_REFERENCE.md) | Detailed API documentation with all method signatures |
| delay_system Design Pattern | [delay_system/DESIGN_PATTERN.md](../delay_system/DESIGN_PATTERN.md) | Architecture decisions, migration guides, version history |

### Module Reference

| Module | Location | Description |
|--------|----------|-------------|
| `delay_system` | [delay_system/](../delay_system/) | v2.2 dual-pipeline delay system (current) |
| `delayed_states` | [delayed_states/](../delayed_states/) | Legacy state manager (to be replaced) |
| `iris_ma_env3.py` | [iris_ma_env3.py](../iris_ma_env3.py) | Main environment implementation |
| `iris_ma_env3_cfg.py` | [iris_ma_env3_cfg.py](../iris_ma_env3_cfg.py) | Environment configuration |

## Current Integration Status

### Phase 1: delay_system Cleanup (COMPLETED)
- Removed deprecated code from `delay_system` module
- Updated documentation to clearly mark removed methods
- Dual pipeline architecture (v2.2) is fully functional

### Phase 2: iris_ma_env3 Integration (PLANNED)
- Replace `MultiAgentStateManager` with `MultiAgentDelaySystem`
- Migrate from deprecated API to v2.2 view scheme API
- See [DELAY_SYSTEM_INTEGRATION_PLAN.md](DELAY_SYSTEM_INTEGRATION_PLAN.md) for details

## Quick Links

- **To understand delay system architecture**: Start with [delay_system/DESIGN_PATTERN.md](../delay_system/DESIGN_PATTERN.md)
- **To see API examples**: See [delay_system/README.md](../delay_system/README.md)
- **To track integration progress**: See [DELAY_SYSTEM_INTEGRATION_PLAN.md](DELAY_SYSTEM_INTEGRATION_PLAN.md)

## Changelog

| Date | Change |
|------|--------|
| 2025-11-27 | Resolved open questions in integration plan |
| 2025-11-27 | Created documentation index and integration plan |
| 2025-11-27 | Completed delay_system v2.2 cleanup |
