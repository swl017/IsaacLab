# Delay System V2 Documentation Index

This directory contains documentation for the `delay_system_v2` module.

## Documents

| Document | Description | Last Updated |
|----------|-------------|--------------|
| [README.md](README.md) | Main documentation - architecture, API reference, configuration, usage examples | 2026-01-30 |

## Quick Links

- **Architecture Overview**: [README.md#architecture](README.md#architecture)
- **Quick Start**: [README.md#quick-start](README.md#quick-start)
- **Configuration Reference**: [README.md#configuration-reference](README.md#configuration-reference)
- **API Reference**: [README.md#api-reference](README.md#api-reference)
- **Testing**: [README.md#testing](README.md#testing)
- **Migration Guide**: [README.md#migration-from-iris_ma3](README.md#migration-from-iris_ma3)

## Related Documentation

- **Test Suite**: [../tests/README.md](../tests/README.md) - Test documentation and instructions
- **Quadcopter DelaySystemV2**: `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/` - Base implementation

## Summary

The `MultiAgentDelaySystemV2` provides a field-based delay system for multi-agent drone simulations with:

1. **Dual Pipeline**: Clean (rewards) and noisy (observations) data paths
2. **Perspective-Aware Delays**: Fast ego (~5ms) vs slow inter-agent (~100ms) communication
3. **Per-Field Processing**: First-order lag, staleness, latency, dropout, and noise injection
4. **Derived Field Computation**: Camera geometry computed from delayed raw fields
5. **API Compatibility**: Drop-in replacement for iris_ma3 delay system
