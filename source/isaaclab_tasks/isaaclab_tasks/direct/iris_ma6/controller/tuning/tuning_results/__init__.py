# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tuning results — DroneController gain sets, one per (plant, source) pair.

Three configurations are exported, all sharing the same PX4-style cascaded
architecture: Velocity (PI) → Attitude (P) → Rate (PID) → Mixer → Motor.

    ``TUNED_CONTROLLER_CFG``
        Hand-tuned baseline from ``auto_tune.py`` (pre-ticket-008). Used as the
        "current gains" baseline in ``sysid_replicator.py``'s 3-way comparison
        plots and by the controller unit tests
        (``controller/tests/run_controller_test.py``,
        ``controller/tests/test_gimbal_sim.py``). NOT the env-default —
        ``iris_ma_env6_test_cfg.py`` selects one of the ``PX4_MATCHED_*``
        sets below based on ``physics_mode``.

    ``PX4_MATCHED_CONTROLLER_CFG`` (from ``px4_matched.py``)
        Gains fit to PX4 SITL CSVs against the **default-mode** plant
        (racing-class numerics: k_f=1.2e-5, omega_max=5000, τ_motor=10 ms,
        quadratic body drag). Produced by ``sysid_replicator.py --plant default``
        (ticket 008).

    ``PX4_MATCHED_PEGASUS_CONTROLLER_CFG`` (from ``px4_matched_pegasus.py``)
        Gains fit to PX4 SITL CSVs against the **pegasus-mode** plant
        (PegasusSimulator IrisConfig parity: k_f=8.55e-6, omega_max=1100,
        τ_motor≈0, linear-diagonal body drag) WITH ticket-041 EKF state lag
        applied to the feedback path (18 ms attitude, 15 ms angular velocity).
        Produced by ``sysid_replicator.py --plant pegasus --ekf-lag`` (ticket
        040 + 041). Yields vel_5_settling gap 0.4%, yaw_settling gap 40%,
        score 0.0168 — best of the four ablation runs.

Auto-pairing in ``IrisMA6TestEnvCfg.__post_init__``:
    physics_mode = "pegasus" (default) → PX4_MATCHED_PEGASUS_CONTROLLER_CFG
    physics_mode = "default"           → PX4_MATCHED_CONTROLLER_CFG
"""

from isaaclab_tasks.direct.iris_ma6.controller.controller_cfg import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg

from .px4_matched import PX4_MATCHED_CONTROLLER_CFG
from .px4_matched_pegasus import PX4_MATCHED_PEGASUS_CONTROLLER_CFG

# Hand-tuned baseline from auto_tune.py. Retained as the "current-gains"
# reference for the sysid_replicator's 3-way comparison and as the default
# in controller-level unit tests. NOT used by the env at runtime.
TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(2.0416, 2.0416, 1.5355),
        Ki_vel=(1.3005, 1.3005, 0.7767),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(5.2060, 5.2060, 1.9334),
    ),
    rate=RateControllerCfg(
        Kp_rate=(0.3932, 0.3932, 0.3956),
        Ki_rate=(0.1805, 0.1805, 0.0900),
        Kd_rate=(0.01451, 0.01451, 0.00000),
    ),
)

__all__ = [
    "TUNED_CONTROLLER_CFG",
    "PX4_MATCHED_CONTROLLER_CFG",
    "PX4_MATCHED_PEGASUS_CONTROLLER_CFG",
]
