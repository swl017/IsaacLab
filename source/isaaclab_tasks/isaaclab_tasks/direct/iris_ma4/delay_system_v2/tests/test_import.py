#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that all delay_system_v2 modules can be imported correctly."""

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Test delay_system_v2 imports")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import modules
import torch


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 80)
    print("Testing delay_system_v2 imports")
    print("=" * 80)

    try:
        from isaaclab_tasks.direct.iris_ma4.delay_system_v2 import (
            DelayCfg,
            DelaySystemCfg,
            DelaySystemCfgV2,
            DistributionCfg,
            FieldDelayCfg,
            NoiseCfg,
            DelayPipeline,
            DelaySystemV2,
            DataBus,
            DerivedFieldComputer,
            DerivedFieldDef,
            MultiAgentDelaySystemV2,
            MultiAgentDelaySystemV2Cfg,
            AgentStates,
        )
        print("  ✓ All imports successful!")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

    # Test DistributionCfg with half_range
    try:
        cfg = DistributionCfg(type="uniform", mean=30.0, half_range=10.0)
        assert cfg.type == "uniform"
        assert cfg.mean == 30.0
        assert cfg.half_range == 10.0
        print(f"  ✓ DistributionCfg: type={cfg.type}, mean={cfg.mean}, half_range={cfg.half_range}")
    except Exception as e:
        print(f"  ✗ DistributionCfg test failed: {e}")
        return False

    # Test FieldDelayCfg
    try:
        field_cfg = FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.1,
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="normal", mean=30.0, std=5.0),
        )
        assert field_cfg.first_order_lag_enabled
        assert field_cfg.time_constant == 0.1
        print(f"  ✓ FieldDelayCfg: lag={field_cfg.first_order_lag_enabled}, tau={field_cfg.time_constant}")
    except Exception as e:
        print(f"  ✗ FieldDelayCfg test failed: {e}")
        return False

    # Test DataBus
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_bus = DataBus(num_envs=16, device=device)
        test_data = torch.randn(16, 3, device=device)
        data_bus.store("test_field", test_data, noise_std=0.1)
        clean = data_bus.get_clean("test_field")
        noisy = data_bus.get_noisy("test_field")
        assert clean.shape == (16, 3)
        assert noisy.shape == (16, 3)
        print(f"  ✓ DataBus: clean.shape={clean.shape}, noisy.shape={noisy.shape}")
    except Exception as e:
        print(f"  ✗ DataBus test failed: {e}")
        return False

    # Test DelayPipeline
    try:
        pipeline_cfg = FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.05,
        )
        pipeline = DelayPipeline(
            cfg=pipeline_cfg,
            num_envs=16,
            field_dim=3,
            dt=0.01,
            device=device,
        )
        t_current = torch.zeros(16, device=device)
        result = pipeline.process(test_data, t_current)
        assert result.shape == (16, 3)
        print(f"  ✓ DelayPipeline: result.shape={result.shape}")
    except Exception as e:
        print(f"  ✗ DelayPipeline test failed: {e}")
        return False

    # Test DelaySystemV2
    try:
        v2_cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "test_field": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.05,
                ),
            },
        )
        delay_system = DelaySystemV2(
            cfg=v2_cfg,
            num_envs=16,
            device=str(device),
            field_dims={"test_field": 3},
        )
        delay_system.store("test_field", test_data)
        delayed_clean = delay_system.get_delayed_clean("test_field")
        delayed_noisy = delay_system.get_delayed_noisy("test_field")
        assert delayed_clean.shape == (16, 3)
        assert delayed_noisy.shape == (16, 3)
        print(f"  ✓ DelaySystemV2: delayed_clean.shape={delayed_clean.shape}")
    except Exception as e:
        print(f"  ✗ DelaySystemV2 test failed: {e}")
        return False

    # Test MultiAgentDelaySystemV2Cfg
    try:
        ma_cfg = MultiAgentDelaySystemV2Cfg(
            dt=0.01,
            motion_time_constant=0.1,
            detection_fps_mean=30.0,
            enable_noise=True,
        )
        assert ma_cfg.dt == 0.01
        assert ma_cfg.motion_time_constant == 0.1
        print(f"  ✓ MultiAgentDelaySystemV2Cfg: dt={ma_cfg.dt}, motion_tau={ma_cfg.motion_time_constant}")
    except Exception as e:
        print(f"  ✗ MultiAgentDelaySystemV2Cfg test failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_imports()
    simulation_app.close()
    exit(0 if success else 1)
