# IRIS MA5 Environment Documentation Index

This directory contains documentation for the `iris_ma5` multi-agent drone environment (N-agent support).

## Document Catalog

### Architecture Documents

| Document | Location | Description |
|----------|----------|-------------|
| README | [README.md](README.md) | Overview of iris_ma5: architecture, action/observation spaces, conventions |
| Technical Report | [IRIS_MA5_TECHNICAL_REPORT.md](IRIS_MA5_TECHNICAL_REPORT.md) | Full mathematical formulations, reward structure, observation layout |
| delay_system_v2 README | [delay_system_v2/doc/README.md](../delay_system_v2/doc/README.md) | Overview of field-based delay system v2 |

### Module Reference

| Module | Location | Description |
|--------|----------|-------------|
| `iris_ma_env5.py` | [iris_ma_env5.py](../iris_ma_env5.py) | Main environment implementation (N-agent) |
| `iris_ma_env5_cfg.py` | [iris_ma_env5_cfg.py](../iris_ma_env5_cfg.py) | Environment configuration (num_agents parameter) |
| `delay_system_v2/` | [delay_system_v2/](../delay_system_v2/) | Field-based delay pipeline (current) |
| `triangulation/` | [triangulation/](../triangulation/) | Multi-view triangulation covariance |
| `safety/` | [safety/](../safety/) | Collision detection and TTC |
| `randomization/` | [randomization/](../randomization/) | Distance-based formation generation |
| `curriculum/` | [curriculum/](../curriculum/) | 5-phase training curriculum |

## N-Agent Support (iris_ma5)

iris_ma5 generalizes iris_ma4 (2-agent) to N agents (N >= 2):

- **Observation space**: `26 + 15*(N-1) + 6` per agent
- **Valid triangulation**: requires at least 2 of N agents to detect the target
- **num_agents**: configured in `IrisMAEnvCfg(num_agents=N)`, default 3

## Changelog

| Date | Change |
|------|--------|
| 2026-02-24 | Upgraded from iris_ma4 to iris_ma5 with N-agent support |
| 2026-02-24 | Removed legacy delay_system/, keeping only delay_system_v2/ |
| 2026-02-24 | Added __post_init__ to IrisMAEnvCfg for dynamic agent config |
