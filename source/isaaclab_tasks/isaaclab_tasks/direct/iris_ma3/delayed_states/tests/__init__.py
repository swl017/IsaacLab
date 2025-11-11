"""
Test suite for delayed_states.py components.

This package contains comprehensive tests for:
- DelayBufferCustom: Buffer corruption prevention
- ChannelBuffer: Throttle, dropout, latency impairments
- MultiAgentCommChannel: Multi-agent communication
- FirstOrderLag / QuaternionFirstOrderLag: Filtering
- MultiAgentObservationPipeline: End-to-end integration
- Validity mask handling
- Memory leaks and performance
"""

__version__ = "1.0.0"
