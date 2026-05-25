# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-channel ego motion latency regression tests (ticket 042).

Locks in three invariants for the per-channel ego-motion-latency patch:

1. Bit-exact legacy: ``use_bulk_ego_motion_latency=True`` produces a
   ``cfg.delay_cfg.ego.field_overrides`` dict that contains only the
   ``bboxes_2d`` override — no per-channel motion entries leak in. This
   is the rollback path; any divergence here changes the behavior of
   every pre-ticket-042 experiment that explicitly opts back into bulk.

2. Per-channel defaults from ticket 041: when ``use_bulk_ego_motion_latency``
   is False (the post-patch default), every motion field listed in
   ``multi_agent_wrapper.BODY_FIELDS`` is mapped to a channel-specific
   ``DelayPipelineCfgV3`` whose latency mean matches the value measured
   in ``controller/sysid_output/ekf_state_lag/ekf_state_lag.json``. The
   body-frame angular-velocity and combined-angular-velocity siblings
   share their world-frame channel's latency (same IMU source).

3. Risk #3 mitigation: if a per-channel override key drifts away from
   the registered AgentStates field set, ``create_delay_cfg_from_params``
   raises ``ValueError`` instead of silently falling through to the
   legacy bulk pipeline. This locks in the assertion that catches the
   class of bug that the original draft patch shipped with.
"""

from __future__ import annotations

import traceback

from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
    DelaySystemKeyParams,
    create_delay_cfg_from_params,
)
from isaaclab_tasks.direct.iris_ma6.delay_system_v3.multi_agent_wrapper import (
    BODY_FIELDS,
)


# Channels documented in ticket 041 (ekf_state_lag.json), mapped to the
# registered AgentStates field names that share the underlying source.
# This mirrors the channel_to_fields dict inside create_delay_cfg_from_params;
# keeping it duplicated here is intentional — the test exists to catch
# accidental drift between the two.
_EXPECTED_PER_CHANNEL_FIELDS: dict[str, tuple[str, ...]] = {
    "body_position_w":                    ("position",),
    "body_linear_velocity_w":             ("velocity",),
    "body_orientation_w":                 ("orientation",),
    "body_angular_velocity_w":            ("angular_velocity",),
    "body_angular_velocity_b":            ("angular_velocity",),
    "body_combined_angular_velocity_w":   ("angular_velocity",),
    "body_linear_acceleration_b":         ("linear_acceleration",),
}

# Ticket 041 measurement values that the post-patch defaults must match.
_EXPECTED_LATENCY_MEAN_S: dict[str, float] = {
    "body_position_w":                    0.000,
    "body_linear_velocity_w":             0.000,
    "body_orientation_w":                 0.018,
    "body_angular_velocity_w":            0.015,
    "body_angular_velocity_b":            0.015,
    "body_combined_angular_velocity_w":   0.015,
    "body_linear_acceleration_b":         0.035,
}

# Pre-patch defaults — only used to construct the bulk-mode params so the
# bit-exact rollback test mirrors what a downstream experiment that explicitly
# requested the legacy path would receive.
_PRE_PATCH_BULK_PARAMS = DelaySystemKeyParams(
    use_bulk_ego_motion_latency=True,
    ego_motion_latency_enabled=True,
    ego_motion_latency_mean=0.005,
    ego_motion_latency_std=0.002,
    ego_motion_fol_tau=0.005,
    burst_dropout_enabled=False,
)


def _ego_motion_pipeline_repr(cfg) -> str:
    """Stable repr of the ego motion default pipeline (the non-bbox fallback)."""
    return repr(cfg.delay_cfg.ego.pipeline)


def run_per_channel_latency_tests(results, device=None, verbose: bool = False):
    """Run per-channel ego motion latency regression tests.

    ``device`` is accepted for runner-API parity but unused — these are
    pure config-shape tests, no tensors.
    """
    results.set_suite("Per-Channel Ego Motion Latency (ticket 042)")

    # -------------------------------------------------------------------
    # Test 1: bulk mode produces exactly the pre-patch field_overrides set
    # -------------------------------------------------------------------
    # Pre-patch, the ONLY entry in cfg.delay_cfg.ego.field_overrides was
    # the bboxes_2d detection pipeline. Per-channel motion overrides must
    # NOT appear when use_bulk_ego_motion_latency=True.
    try:
        cfg = create_delay_cfg_from_params(_PRE_PATCH_BULK_PARAMS)
        ego_overrides = cfg.delay_cfg.ego.field_overrides

        assert set(ego_overrides.keys()) == {"bboxes_2d"}, (
            f"bulk mode must only override bboxes_2d; got {sorted(ego_overrides.keys())!r}"
        )
        # The bboxes_2d override is the detection pipeline (NN-inference latency).
        # The motion default — the per-channel patch's intended-NoOp surface in bulk mode —
        # falls back to cfg.delay_cfg.ego.pipeline (the legacy bulk motion pipeline).
        assert cfg.delay_cfg.ego.pipeline.latency.enabled is True, (
            "bulk mode default pipeline must have latency enabled (matches pre-patch)"
        )
        assert cfg.delay_cfg.ego.pipeline.latency.distribution.mean == 0.005, (
            "bulk mode latency mean must remain 5 ms (pre-patch value)"
        )
        results.add_pass("bulk mode produces pre-patch field_overrides only")
    except Exception:
        results.add_fail(
            "bulk mode produces pre-patch field_overrides only",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 2: per-channel mode overrides every IMU-source motion field
    # -------------------------------------------------------------------
    # Post-patch default uses use_bulk_ego_motion_latency=False with the
    # ticket 041 values baked in. Every motion field that shares an EKF
    # or IMU source must get its own override entry.
    try:
        params = DelaySystemKeyParams(burst_dropout_enabled=False)
        # Defensive: the dataclass default is documented as False; assert
        # so we fail fast if a future edit flips it.
        assert params.use_bulk_ego_motion_latency is False, (
            "DelaySystemKeyParams must default to per-channel mode "
            "(use_bulk_ego_motion_latency=False) post-ticket-042"
        )

        cfg = create_delay_cfg_from_params(params)
        ego_overrides = cfg.delay_cfg.ego.field_overrides

        expected_keys = set(_EXPECTED_PER_CHANNEL_FIELDS.keys()) | {"bboxes_2d"}
        assert set(ego_overrides.keys()) == expected_keys, (
            "per-channel mode must override every IMU/EKF motion field plus "
            f"bboxes_2d; got {sorted(ego_overrides.keys())!r}, "
            f"expected {sorted(expected_keys)!r}"
        )
        results.add_pass("per-channel mode covers every IMU/EKF motion field")
    except Exception:
        results.add_fail(
            "per-channel mode covers every IMU/EKF motion field",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 3: every override key resolves to a registered AgentStates field
    # -------------------------------------------------------------------
    # The silent-failure mode is an override key (e.g. "body_velocity_w")
    # that doesn't match any field name in multi_agent_wrapper.BODY_FIELDS.
    # The runtime fall-through is invisible (just silently uses the bulk
    # default), so we lock it down here.
    try:
        params = DelaySystemKeyParams(burst_dropout_enabled=False)
        cfg = create_delay_cfg_from_params(params)

        valid_body_fields = {name for name, _ in BODY_FIELDS}
        motion_keys = set(cfg.delay_cfg.ego.field_overrides.keys()) - {"bboxes_2d"}
        unknown = motion_keys - valid_body_fields

        assert not unknown, (
            f"per-channel overrides include keys not registered as ego "
            f"motion fields in multi_agent_wrapper.BODY_FIELDS: {sorted(unknown)!r}. "
            "These would silently fall through to the bulk default pipeline."
        )
        results.add_pass("every per-channel override key matches a registered ego field")
    except Exception:
        results.add_fail(
            "every per-channel override key matches a registered ego field",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 4: per-channel latency means match ticket 041 measurements
    # -------------------------------------------------------------------
    # The numeric defaults are the load-bearing claim of the patch — if a
    # downstream edit accidentally bumps one (e.g. flips position from 0 to
    # 5 ms "for consistency"), we lose the load-bearing reference to
    # ekf_state_lag.json. This test guards against that drift.
    try:
        params = DelaySystemKeyParams(burst_dropout_enabled=False)
        cfg = create_delay_cfg_from_params(params)
        ego_overrides = cfg.delay_cfg.ego.field_overrides

        for field_name, expected_mean_s in _EXPECTED_LATENCY_MEAN_S.items():
            assert field_name in ego_overrides, (
                f"missing per-channel override for {field_name!r}"
            )
            actual_mean = ego_overrides[field_name].latency.distribution.mean
            assert abs(actual_mean - expected_mean_s) < 1e-9, (
                f"{field_name}: expected latency mean {expected_mean_s} s "
                f"(from ekf_state_lag.json), got {actual_mean} s"
            )
            # Std is the same across all channels per the patch contract.
            actual_std = ego_overrides[field_name].latency.distribution.std
            assert abs(actual_std - params.ego_per_channel_latency_std_s) < 1e-9, (
                f"{field_name}: latency std should be ego_per_channel_latency_std_s "
                f"({params.ego_per_channel_latency_std_s} s), got {actual_std} s"
            )
        results.add_pass("per-channel latency means match ticket 041 defaults")
    except Exception:
        results.add_fail(
            "per-channel latency means match ticket 041 defaults",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 5: siblings sharing an IMU source receive identical pipelines
    # -------------------------------------------------------------------
    # body_angular_velocity_{w,b} and body_combined_angular_velocity_w share
    # the gyro source, so they must carry the same (latency_mean, fol_tau)
    # — otherwise the policy sees a mix of differently-aged copies of the
    # same physical signal.
    try:
        params = DelaySystemKeyParams(burst_dropout_enabled=False)
        cfg = create_delay_cfg_from_params(params)
        ego_overrides = cfg.delay_cfg.ego.field_overrides

        siblings = (
            "body_angular_velocity_w",
            "body_angular_velocity_b",
            "body_combined_angular_velocity_w",
        )
        means = {
            name: ego_overrides[name].latency.distribution.mean for name in siblings
        }
        taus = {
            name: ego_overrides[name].first_order_lag.tau for name in siblings
        }
        unique_means = set(means.values())
        unique_taus = set(taus.values())
        assert len(unique_means) == 1, (
            f"angular-velocity siblings must share latency mean; got {means!r}"
        )
        assert len(unique_taus) == 1, (
            f"angular-velocity siblings must share first-order-lag tau; got {taus!r}"
        )
        results.add_pass("angular-velocity siblings share one pipeline realization")
    except Exception:
        results.add_fail(
            "angular-velocity siblings share one pipeline realization",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 6: assertion fires on a drifted field name (Risk #3)
    # -------------------------------------------------------------------
    # Monkey-patch BODY_FIELDS to remove "body_position_w" — this simulates
    # the inverse of a real-world rename in agent_states.py. The factory
    # must raise ValueError instead of silently falling back to bulk.
    try:
        from isaaclab_tasks.direct.iris_ma6.delay_system_v3 import multi_agent_wrapper

        original_body_fields = multi_agent_wrapper.BODY_FIELDS
        # Strip one field to force a mismatch.
        multi_agent_wrapper.BODY_FIELDS = [
            entry for entry in original_body_fields if entry[0] != "body_position_w"
        ]
        try:
            params = DelaySystemKeyParams(burst_dropout_enabled=False)
            raised = False
            try:
                create_delay_cfg_from_params(params)
            except ValueError as exc:
                raised = True
                msg = str(exc)
                assert "body_position_w" in msg, (
                    f"error message should mention the unknown field name; got {msg!r}"
                )
                assert "fall-through" in msg or "fall through" in msg.lower(), (
                    "error message should explain the silent-fall-through risk; "
                    f"got {msg!r}"
                )
            assert raised, (
                "create_delay_cfg_from_params must raise ValueError when an "
                "override key is not in multi_agent_wrapper.BODY_FIELDS"
            )
            results.add_pass("ValueError raised on drifted override field name")
        finally:
            multi_agent_wrapper.BODY_FIELDS = original_body_fields
    except Exception:
        results.add_fail(
            "ValueError raised on drifted override field name",
            traceback.format_exc(),
        )
