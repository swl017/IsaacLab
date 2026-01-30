#!/usr/bin/env python3
"""Test behind-camera filtering in midpoint_method_batched."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test behind-camera filtering")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.direct.iris_ma3.triangulation.triang_cov_reward_torch import midpoint_method_batched


def test_behind_camera_filtering():
    """Test that behind-camera triangulations are detected and flagged."""
    print("=" * 80)
    print("Testing midpoint_method_batched with behind-camera filtering")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, C, T = 4, 3, 1  # 4 envs, 3 cameras, 1 target

    # Camera positions forming a triangle
    pts = torch.tensor([
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]],  # env 0: valid triangulation
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]],  # env 1: valid triangulation
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]],  # env 2: behind-camera case
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]],  # env 3: insufficient cameras
    ], device=device)  # [4, 3, 3]

    # Ray directions - all cameras looking at a point in front
    target_point = torch.tensor([5.0, 3.0, 5.0], device=device)
    dirs_forward = target_point.unsqueeze(0).unsqueeze(0) - pts.unsqueeze(2)  # [N, C, 1, 3]
    dirs_forward = dirs_forward / torch.norm(dirs_forward, dim=-1, keepdim=True)

    # env 2: rays pointing AWAY from center (behind camera)
    dirs_behind = -dirs_forward.clone()

    # Create combined dirs tensor
    dirs = dirs_forward.clone()
    dirs[2] = dirs_behind[2]  # env 2 has behind-camera rays

    # Valid mask - env 3 has only 1 valid camera
    valid_mask = torch.ones(N, C, dtype=torch.bool, device=device)
    valid_mask[3, 1:] = False  # env 3 only has camera 0 valid

    print(f"\nTest setup:")
    print(f"  Cameras per env: {C}")
    print(f"  Target point: {target_point.tolist()}")
    print(f"  Valid mask: env0={valid_mask[0].tolist()}, env1={valid_mask[1].tolist()}, env2={valid_mask[2].tolist()}, env3={valid_mask[3].tolist()}")

    # Run triangulation
    X_mid, is_valid = midpoint_method_batched(pts, dirs, valid_mask)

    print(f"\nResults:")
    print(f"  X_mid shape: {X_mid.shape}")
    print(f"  is_valid shape: {is_valid.shape}")

    for i in range(N):
        status = "VALID" if is_valid[i, 0].item() else "INVALID"
        print(f"  Env {i}: X_mid={[f'{x:.2f}' for x in X_mid[i, 0].tolist()]}, is_valid={status}")

    print(f"\nExpected results:")
    print(f"  Env 0: Valid, near {target_point.tolist()}")
    print(f"  Env 1: Valid, near {target_point.tolist()}")
    print(f"  Env 2: Invalid (behind camera), fallback to mean")
    print(f"  Env 3: Invalid (insufficient cameras), fallback to camera 0")

    # Verification
    passed = True

    if is_valid[0, 0].item() != True:
        print("✗ FAIL: Env 0 should be valid")
        passed = False
    else:
        print("✓ Env 0 valid as expected")

    if is_valid[1, 0].item() != True:
        print("✗ FAIL: Env 1 should be valid")
        passed = False
    else:
        print("✓ Env 1 valid as expected")

    if is_valid[2, 0].item() != False:
        print("✗ FAIL: Env 2 should be invalid (behind camera)")
        passed = False
    else:
        print("✓ Env 2 invalid (behind camera) as expected")

    if is_valid[3, 0].item() != False:
        print("✗ FAIL: Env 3 should be invalid (insufficient cameras)")
        passed = False
    else:
        print("✓ Env 3 invalid (insufficient cameras) as expected")

    # Check triangulation accuracy for valid envs
    dist_0 = torch.norm(X_mid[0, 0] - target_point).item()
    dist_1 = torch.norm(X_mid[1, 0] - target_point).item()

    print(f"\nTriangulation accuracy:")
    print(f"  Env 0 distance from target: {dist_0:.4f}")
    print(f"  Env 1 distance from target: {dist_1:.4f}")

    if dist_0 > 0.1:
        print(f"✗ FAIL: Env 0 triangulation error too large (>{dist_0:.4f})")
        passed = False
    else:
        print("✓ Env 0 triangulation accurate")

    if dist_1 > 0.1:
        print(f"✗ FAIL: Env 1 triangulation error too large (>{dist_1:.4f})")
        passed = False
    else:
        print("✓ Env 1 triangulation accurate")

    print("\n" + "=" * 80)
    if passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return passed


def main():
    """Main entry point."""
    success = test_behind_camera_filtering()

    # Cleanup
    simulation_app.close()

    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
