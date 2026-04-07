#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sysid replicator — tune iris_ma6 controller gains to match PX4 SITL response.

Sweeps DroneController gain sets, collects step responses, and scores each
against PX4 SITL recordings. The output is the gain set whose iris_ma6
response best matches PX4 SITL.

Usage:
    # Full sweep with 1024 candidates
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/sysid_replicator.py \\
        --headless --sysid-dir source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/sysid_output

    # Dry run — load targets only, no Isaac Sim
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/sysid_replicator.py \\
        --sysid-dir source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/sysid_output --dry-run
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sysid replicator: match iris_ma6 to PX4 SITL")
AppLauncher.add_app_launcher_args(parser)
import os as _os
_DEFAULT_SYSID_DIR = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "sysid_output"
)
parser.add_argument("--sysid-dir", type=str, default=_DEFAULT_SYSID_DIR,
                    help="Directory with PX4 SITL CSVs (default: controller/sysid_output)")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: sysid-dir/analysis)")
parser.add_argument("--num-trials", type=int, default=1024, help="Number of random gain candidates")
parser.add_argument("--aero-level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Aerodynamic fidelity level")
parser.add_argument("--top-k", type=int, default=5, help="Number of top results to plot")
parser.add_argument("--dry-run", action="store_true", help="Load targets only, skip Isaac Sim")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-AppLauncher imports ---
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class TargetTimeseries:
    """Resampled PX4 SITL target for one test."""

    t: np.ndarray
    """(T,) time at 100 Hz [s], starting from 0."""

    vel_x: np.ndarray
    """(T,) forward velocity [m/s]."""

    yaw_rate: np.ndarray
    """(T,) yaw rate [rad/s]."""

    att_error: np.ndarray
    """(T, 3) attitude error [rad] (actual vs att_sp, roll/pitch/yaw)."""

    rate_error: np.ndarray
    """(T, 3) rate error [rad/s] (actual vs rate_sp, body frame)."""


@dataclass
class ReplicatorScoreWeights:
    """MSE weights per signal group."""

    velocity: float = 0.60
    attitude: float = 0.25
    rate: float = 0.15


@dataclass
class ParameterSet:
    """Controller parameter set (mirrored from auto_tune.py).

    4-Loop Architecture (PX4-style):
    - Velocity Controller (PI): v_cmd -> attitude_cmd + thrust
    - Attitude Controller (P-only): attitude_error -> rate_setpoint
    - Rate Controller (PID): rate_error -> torque
    - Mixer: torque -> motor commands
    """

    # Velocity controller (PI)
    Kp_vel_xy: float = 3.0
    Kp_vel_z: float = 2.0
    Ki_vel_xy: float = 0.5
    Ki_vel_z: float = 0.3

    # Attitude controller (P-only)
    Kp_att_rp: float = 6.5
    Kp_att_y: float = 2.8

    # Rate controller (PID)
    Kp_rate_rp: float = 0.15
    Kp_rate_y: float = 0.2
    Ki_rate_rp: float = 0.2
    Ki_rate_y: float = 0.1
    Kd_rate_rp: float = 0.003
    Kd_rate_y: float = 0.0


def generate_random_params(num_trials: int) -> list[ParameterSet]:
    """Generate random parameter sets (mirrored from auto_tune.py).

    Search ranges and coupling constraints are identical to auto_tune.py.
    """
    params_list = []
    for _ in range(num_trials):
        Kp_rate_rp = np.random.uniform(0.10, 0.40)
        Ki_rate_rp = np.random.uniform(0.05, 0.30)
        Kd_rate_rp = np.random.uniform(0.002, 0.015)
        Kp_att_rp = np.random.uniform(4.0, 10.0)
        Kp_vel_max = min(5.0, 2.0 + 10.0 * Kp_rate_rp)
        Kp_vel_xy = np.random.uniform(1.5, Kp_vel_max)
        Ki_vel_xy = np.random.uniform(0.3, 2.0)

        params = ParameterSet(
            Kp_vel_xy=Kp_vel_xy,
            Kp_vel_z=Kp_vel_xy * np.random.uniform(0.6, 0.8),
            Ki_vel_xy=Ki_vel_xy,
            Ki_vel_z=Ki_vel_xy * np.random.uniform(0.5, 0.7),
            Kp_att_rp=Kp_att_rp,
            Kp_att_y=Kp_att_rp * np.random.uniform(0.35, 0.5),
            Kp_rate_rp=Kp_rate_rp,
            Kp_rate_y=Kp_rate_rp * np.random.uniform(1.0, 1.5),
            Ki_rate_rp=Ki_rate_rp,
            Ki_rate_y=Ki_rate_rp * np.random.uniform(0.4, 0.6),
            Kd_rate_rp=Kd_rate_rp,
            Kd_rate_y=0.0,
        )
        params_list.append(params)
    return params_list


# =============================================================================
# Test command definitions (matching sysid_node.py)
# =============================================================================

# Maps test name → (description, nominal_duration from sysid_node)
# Durations here are fallbacks; actual duration comes from CSV data length.
# vel_step_10 excluded: iris_ma6 lacks drag feedforward, so the 10 m/s dynamics
# gap is plant-level (not closable by gain tuning alone).
TEST_COMMANDS = {
    "hover_drift": {
        "description": "Zero velocity hover",
    },
    "vel_step_5": {
        "description": "5 m/s forward step",
        "target_speed": 5.0,
    },
    "vel_impulse_recovery": {
        "description": "5 m/s for 1s then zero",
        "impulse_speed": 5.0,
        "impulse_duration": 1.0,
    },
    "yaw_step": {
        "description": "0.5 rad/s yaw for 2s then zero",
        "yaw_rate": 0.5,
        "command_duration": 2.0,
    },
}


# =============================================================================
# PX4 Target Loader
# =============================================================================


class PX4TargetLoader:
    """Loads PX4 SITL CSVs and resamples to 100 Hz."""

    IRIS_MA6_DT = 0.01  # 100 Hz target rate

    def __init__(self, sysid_dir: str | Path):
        self._sysid_dir = Path(sysid_dir)
        self._targets: dict[str, TargetTimeseries] = {}

        for test_name in TEST_COMMANDS:
            csv_path = self._sysid_dir / f"{test_name}.csv"
            if not csv_path.exists():
                print(f"  WARNING: {csv_path} not found, skipping {test_name}")
                continue
            self._targets[test_name] = self._load_and_process(test_name, csv_path)

    def _load_and_process(self, test_name: str, csv_path: Path) -> TargetTimeseries:
        """Load CSV, resample to 100 Hz, compute cascade errors."""
        df = self._load_csv(csv_path)
        df = self._resample_to_100hz(df)

        t = df["t"].values
        vel_x = df["vel_x"].values
        yaw_rate = df["ang_vel_z"].values

        att_error = self._compute_att_error(df)
        rate_error = self._compute_rate_error(df)

        return TargetTimeseries(
            t=t,
            vel_x=vel_x,
            yaw_rate=yaw_rate,
            att_error=att_error,
            rate_error=rate_error,
        )

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Read CSV, dedup timestamps, add relative time, reorder quaternions."""
        df = pd.read_csv(path)

        # Deduplicate timestamps (keep last — most recent state)
        df = df.drop_duplicates(subset=["timestamp_s"], keep="last").reset_index(drop=True)

        # Relative time starting at 0
        df["t"] = df["timestamp_s"] - df["timestamp_s"].min()

        # Reorder actual quaternion: CSV is xyzw → we keep as-is for pandas,
        # but compute errors using wxyz convention internally.
        # Store wxyz columns for internal use.
        df["quat_w_wxyz"] = df["quat_w"]
        df["quat_x_wxyz"] = df["quat_x"]
        df["quat_y_wxyz"] = df["quat_y"]
        df["quat_z_wxyz"] = df["quat_z"]

        # Setpoint quaternion: also xyzw in CSV
        if "att_sp_qw" in df.columns:
            df["att_sp_w_wxyz"] = df["att_sp_qw"]
            df["att_sp_x_wxyz"] = df["att_sp_qx"]
            df["att_sp_y_wxyz"] = df["att_sp_qy"]
            df["att_sp_z_wxyz"] = df["att_sp_qz"]

        return df

    def _resample_to_100hz(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample from ~125 Hz to 100 Hz via linear interpolation."""
        t_orig = df["t"].values
        t_max = t_orig[-1]
        # Create 100 Hz grid
        t_new = np.arange(0, t_max, self.IRIS_MA6_DT)

        # Interpolate all numeric columns
        new_data = {"t": t_new}
        for col in df.columns:
            if col == "t" or col == "timestamp_s":
                continue
            vals = df[col].values
            if np.issubdtype(vals.dtype, np.number):
                new_data[col] = np.interp(t_new, t_orig, vals)

        return pd.DataFrame(new_data)

    def _compute_att_error(self, df: pd.DataFrame) -> np.ndarray:
        """Compute attitude error: actual vs setpoint quaternion.

        Uses same formula as AttitudeController.compute_attitude_error:
            q_err = q_des^{-1} * q_actual
            att_error = 2 * sign(q_err.w) * q_err.xyz

        Args:
            df: Resampled DataFrame with wxyz quaternion columns.

        Returns:
            (T, 3) attitude error [rad] (roll, pitch, yaw).
        """
        if "att_sp_w_wxyz" not in df.columns:
            return np.zeros((len(df), 3))

        # Actual quaternion (wxyz)
        q_w = df["quat_w_wxyz"].values
        q_x = df["quat_x_wxyz"].values
        q_y = df["quat_y_wxyz"].values
        q_z = df["quat_z_wxyz"].values

        # Setpoint quaternion (wxyz)
        sp_w = df["att_sp_w_wxyz"].values
        sp_x = df["att_sp_x_wxyz"].values
        sp_y = df["att_sp_y_wxyz"].values
        sp_z = df["att_sp_z_wxyz"].values

        # q_err = q_des^{-1} * q_actual
        # For unit quaternion, inverse = conjugate: q^{-1} = (w, -x, -y, -z)
        # Hamilton product: (a1,b1,c1,d1) * (a2,b2,c2,d2)
        # w = a1*a2 - b1*b2 - c1*c2 - d1*d2
        # x = a1*b2 + b1*a2 + c1*d2 - d1*c2
        # y = a1*c2 - b1*d2 + c1*a2 + d1*b2
        # z = a1*d2 + b1*c2 - c1*b2 + d1*a2
        #
        # With q_des_inv = (sp_w, -sp_x, -sp_y, -sp_z):
        err_w = sp_w * q_w + sp_x * q_x + sp_y * q_y + sp_z * q_z
        err_x = sp_w * q_x - sp_x * q_w - sp_y * q_z + sp_z * q_y
        err_y = sp_w * q_y + sp_x * q_z - sp_y * q_w - sp_z * q_x
        err_z = sp_w * q_z - sp_x * q_y + sp_y * q_x - sp_z * q_w

        # Shortest path: sign(w)
        sign_w = np.sign(err_w)
        sign_w[sign_w == 0] = 1.0

        # att_error = 2 * sign(w) * [x, y, z]
        att_error = np.stack([
            2.0 * sign_w * err_x,
            2.0 * sign_w * err_y,
            2.0 * sign_w * err_z,
        ], axis=-1)

        return att_error

    def _compute_rate_error(self, df: pd.DataFrame) -> np.ndarray:
        """Compute rate error: rate_setpoint - actual angular velocity.

        Both are in body frame [rad/s].

        Returns:
            (T, 3) rate error [rad/s] (x, y, z body).
        """
        if "rate_sp_x" not in df.columns:
            return np.zeros((len(df), 3))

        rate_sp = np.stack([
            df["rate_sp_x"].values,
            df["rate_sp_y"].values,
            df["rate_sp_z"].values,
        ], axis=-1)

        ang_vel = np.stack([
            df["ang_vel_x"].values,
            df["ang_vel_y"].values,
            df["ang_vel_z"].values,
        ], axis=-1)

        return rate_sp - ang_vel

    def get_target(self, test_name: str) -> TargetTimeseries:
        """Get resampled target timeseries for a test."""
        return self._targets[test_name]

    @property
    def test_names(self) -> list[str]:
        """Available test names (loaded successfully)."""
        return list(self._targets.keys())

    @property
    def test_durations(self) -> dict[str, float]:
        """Duration in seconds for each loaded test."""
        return {name: target.t[-1] for name, target in self._targets.items()}

    def print_summary(self):
        """Print summary of loaded targets."""
        print(f"\nPX4 SITL Targets ({len(self._targets)} tests loaded)")
        print("-" * 70)
        print(f"  {'Test':<25} {'Duration':>8} {'Samples':>8} {'vel_x range':>18} {'att_err range':>18}")
        print("-" * 70)
        for name, target in self._targets.items():
            dur = target.t[-1]
            n = len(target.t)
            vx_range = f"[{target.vel_x.min():.2f}, {target.vel_x.max():.2f}]"
            ae_norm = np.linalg.norm(target.att_error, axis=-1)
            ae_range = f"[{ae_norm.min():.4f}, {ae_norm.max():.4f}]"
            print(f"  {name:<25} {dur:>7.2f}s {n:>8} {vx_range:>18} {ae_range:>18}")

        print("-" * 70)
        print(f"  Resample rate: {1.0 / self.IRIS_MA6_DT:.0f} Hz (dt={self.IRIS_MA6_DT}s)")

        # Sanity checks
        for name, target in self._targets.items():
            if np.isnan(target.vel_x).any():
                print(f"  WARNING: {name} vel_x contains NaN!")
            if np.isnan(target.att_error).any():
                print(f"  WARNING: {name} att_error contains NaN!")
            if np.isnan(target.rate_error).any():
                print(f"  WARNING: {name} rate_error contains NaN!")


# =============================================================================
# SysidReplicator — Gain sweep scored against PX4 SITL timeseries
# =============================================================================


class SysidReplicator:
    """Gain sweep scored against PX4 SITL timeseries.

    Composes ParallelTuner for env/controller infrastructure. Runs the same
    test commands as sysid_node.py at iris_ma6's 100 Hz physics rate, collecting
    timeseries for comparison against PX4 SITL targets.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        aero_level: int,
        targets: dict[str, TargetTimeseries],
        weights: ReplicatorScoreWeights | None = None,
    ):
        import torch

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self.targets = targets
        self.weights = weights or ReplicatorScoreWeights()
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Create environment (same as ParallelTuner.__init__)
        print("Creating Isaac Sim environment...", flush=True)
        task_name = "Isaac-Iris-MA6-Direct-Test-v0"
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        self.env = gym.make(task_name, cfg=env_cfg)
        self.unwrapped = self.env.unwrapped

        agent_id = self.unwrapped.cfg.possible_agents[0]
        self.robot = self.unwrapped._robots[agent_id]
        self.sim = self.unwrapped.sim
        self.physics_dt = self.unwrapped.cfg.sim.dt
        self.dt = self.physics_dt

        body_ids, _ = self.robot.find_bodies("body")
        self.body_ids = body_ids

        self.env.reset()
        all_masses = self.robot.root_physx_view.get_masses()
        per_env_mass = all_masses[0].sum().item()
        self.mass = per_env_mass if per_env_mass > 0.1 else 1.5
        print(f"Environment: {num_envs} envs, dt={self.dt:.3f}s, mass={self.mass:.2f}kg", flush=True)

        # Create DroneController (same as ParallelTuner)
        from isaaclab_tasks.direct.iris_ma6.controller import DroneController, DroneControllerCfg
        from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg

        cfg = DroneControllerCfg(
            velocity=VelocityControllerCfg(
                integral_limit=(8.0, 8.0, 4.0),
            ),
        )
        self.controller = DroneController(
            cfg=cfg,
            mass=self.mass,
            gravity=9.81,
            num_envs=num_envs,
            device=device,
        )
        self.controller.set_aerodynamic_level(aero_level)
        if aero_level > 0:
            print(f"Aerodynamics enabled: level {aero_level}", flush=True)

        self._num_envs = num_envs
        self._device = self.device
        self._dt = self.dt

    def _reset_controller_state(self):
        """Reset DroneController for all environments."""
        import torch
        all_ids = torch.arange(self.num_envs, device=self._device)
        self.controller.reset(all_ids)

    def _reset_to_hover(self):
        """Reset all envs to clean hover state (level attitude, 2m height, zero velocity)."""
        self.env.reset()
        self._reset_controller_state()

        full_root_state = self.robot.data.default_root_state.clone()
        full_root_state[:, 2] = 2.0
        full_root_state[:, 3] = 1.0
        full_root_state[:, 4:7] = 0.0
        full_root_state[:, 7:10] = 0.0
        full_root_state[:, 10:13] = 0.0

        self.robot.write_root_state_to_sim(full_root_state)
        self.robot.update(self.physics_dt)

    def _setup_batch(self, params_list: list):
        """Set per-environment gains on the DroneController."""
        import torch

        N = self.num_envs
        assert len(params_list) == N

        Kp_vel = torch.zeros(N, 3, device=self._device)
        Ki_vel = torch.zeros(N, 3, device=self._device)
        Kp_att = torch.zeros(N, 3, device=self._device)
        Kp_rate = torch.zeros(N, 3, device=self._device)
        Ki_rate = torch.zeros(N, 3, device=self._device)
        Kd_rate = torch.zeros(N, 3, device=self._device)

        for i, p in enumerate(params_list):
            Kp_vel[i] = torch.tensor([p.Kp_vel_xy, p.Kp_vel_xy, p.Kp_vel_z])
            Ki_vel[i] = torch.tensor([p.Ki_vel_xy, p.Ki_vel_xy, p.Ki_vel_z])
            Kp_att[i] = torch.tensor([p.Kp_att_rp, p.Kp_att_rp, p.Kp_att_y])
            Kp_rate[i] = torch.tensor([p.Kp_rate_rp, p.Kp_rate_rp, p.Kp_rate_y])
            Ki_rate[i] = torch.tensor([p.Ki_rate_rp, p.Ki_rate_rp, p.Ki_rate_y])
            Kd_rate[i] = torch.tensor([p.Kd_rate_rp, p.Kd_rate_rp, p.Kd_rate_y])

        self.controller._velocity.set_gains(Kp_vel=Kp_vel, Ki_vel=Ki_vel)
        self.controller._attitude.set_gains(Kp_att=Kp_att)
        self.controller._rate.set_gains(Kp_rate=Kp_rate, Ki_rate=Ki_rate, Kd_rate=Kd_rate)

    def _step_with_controller(self, v_cmd, yaw_rate_cmd=None):
        """Run one physics step, returning att_error, rate_error.

        Wraps ParallelTuner._step_with_controller but supports yaw_rate_cmd.

        Args:
            v_cmd: (N, 3) velocity command [m/s].
            yaw_rate_cmd: (N,) yaw rate command [rad/s], or None for zero.

        Returns:
            att_error: (N, 3) attitude error [rad].
            rate_error: (N, 3) rate error [rad/s].
        """
        import torch

        N = self._num_envs
        q_body = self.robot.data.root_quat_w[:N]
        v_body = self.robot.data.root_lin_vel_w[:N]
        omega_body = self.robot.data.root_ang_vel_b[:N]

        gimbal_pos = torch.zeros(N, 3, device=self._device)
        zeros_n = torch.zeros(N, device=self._device)

        yaw_cmd = yaw_rate_cmd if yaw_rate_cmd is not None else zeros_n

        F_body, tau_body, _, _, _ = self.controller.step_policy(
            v_cmd=v_cmd,
            yaw_rate_cmd=yaw_cmd,
            gimbal_yaw_rate_cmd=zeros_n,
            gimbal_pitch_rate_cmd=zeros_n,
            zoom_rate_cmd=zeros_n,
            q_body=q_body,
            v_body=v_body,
            omega_body=omega_body,
            sim_dt=self.physics_dt,
            gimbal_joint_positions=gimbal_pos,
            physics_dt=self.physics_dt,
        )

        # Extract intermediate cascade signals
        q_des = self.controller._last_q_des if hasattr(self.controller, '_last_q_des') else q_body
        att_error = self.controller._attitude.compute_attitude_error(q_des, q_body)

        rate_setpoint = -self.controller._attitude._Kp_att * att_error
        rate_error = rate_setpoint - omega_body

        # Apply forces and step physics
        forces = torch.zeros(N, 1, 3, device=self._device)
        torques = torch.zeros(N, 1, 3, device=self._device)
        forces[:, 0] = F_body
        torques[:, 0] = tau_body

        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.body_ids)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(self.physics_dt)

        return att_error, rate_error

    def _run_hover_test(self, duration: float) -> dict | None:
        """Run hover test for specified duration.

        Args:
            duration: Test duration [s].

        Returns:
            Dict with vel_history (T, N, 3), att_error_history (T, N, 3),
            rate_error_history (T, N, 3), or None if NaN detected.
        """
        import torch

        N = self._num_envs
        self._reset_to_hover()

        v_cmd = torch.zeros(N, 3, device=self._device)

        vel_history = []
        att_error_history = []
        rate_error_history = []

        num_steps = int(duration / self._dt)
        for _ in range(num_steps):
            vel = self.robot.data.root_lin_vel_w[:N]
            if torch.isnan(vel).any():
                return None

            att_error, rate_error = self._step_with_controller(v_cmd)

            vel_history.append(self.robot.data.root_lin_vel_w[:N].clone())
            att_error_history.append(att_error[:N].clone())
            rate_error_history.append(rate_error[:N].clone())

        return {
            "vel_history": torch.stack(vel_history),            # (T, N, 3)
            "att_error_history": torch.stack(att_error_history),  # (T, N, 3)
            "rate_error_history": torch.stack(rate_error_history),  # (T, N, 3)
        }

    def _run_velocity_test(self, duration: float, target_speed: float) -> dict | None:
        """Run velocity step response test.

        Args:
            duration: Test duration [s].
            target_speed: Target forward velocity [m/s].

        Returns:
            Dict with vel_x_history (T, N), att_error_history (T, N, 3),
            rate_error_history (T, N, 3), or None if NaN detected.
        """
        import torch

        N = self._num_envs
        self._reset_to_hover()

        v_cmd = torch.zeros(N, 3, device=self._device)
        v_cmd[:, 0] = target_speed

        vel_x_history = []
        att_error_history = []
        rate_error_history = []

        num_steps = int(duration / self._dt)
        for _ in range(num_steps):
            vel = self.robot.data.root_lin_vel_w[:N]
            if torch.isnan(vel).any():
                return None

            att_error, rate_error = self._step_with_controller(v_cmd)

            vel_x_history.append(self.robot.data.root_lin_vel_w[:N, 0].clone())
            att_error_history.append(att_error[:N].clone())
            rate_error_history.append(rate_error[:N].clone())

        return {
            "vel_x_history": torch.stack(vel_x_history),         # (T, N)
            "att_error_history": torch.stack(att_error_history),    # (T, N, 3)
            "rate_error_history": torch.stack(rate_error_history),  # (T, N, 3)
        }

    def _run_yaw_step_test(self, duration: float) -> dict | None:
        """Run yaw step response test.

        Commands 0.5 rad/s yaw rate for 2s, then zero for the remainder.

        Args:
            duration: Total test duration [s].

        Returns:
            Dict with yaw_rate_history (T, N), att_error_history (T, N, 3),
            rate_error_history (T, N, 3), or None if NaN detected.
        """
        import torch

        N = self._num_envs
        self._reset_to_hover()

        v_cmd = torch.zeros(N, 3, device=self._device)
        yaw_rate_on = torch.full((N,), 0.5, device=self._device)
        yaw_rate_off = torch.zeros(N, device=self._device)
        command_duration = 2.0  # seconds

        yaw_rate_history = []
        att_error_history = []
        rate_error_history = []

        num_steps = int(duration / self._dt)
        cmd_steps = int(command_duration / self._dt)

        for step in range(num_steps):
            vel = self.robot.data.root_lin_vel_w[:N]
            if torch.isnan(vel).any():
                return None

            yaw_cmd = yaw_rate_on if step < cmd_steps else yaw_rate_off
            att_error, rate_error = self._step_with_controller(v_cmd, yaw_rate_cmd=yaw_cmd)

            omega = self.robot.data.root_ang_vel_b[:N]
            yaw_rate_history.append(omega[:, 2].clone())  # yaw = z-axis body
            att_error_history.append(att_error[:N].clone())
            rate_error_history.append(rate_error[:N].clone())

        return {
            "yaw_rate_history": torch.stack(yaw_rate_history),     # (T, N)
            "att_error_history": torch.stack(att_error_history),    # (T, N, 3)
            "rate_error_history": torch.stack(rate_error_history),  # (T, N, 3)
        }

    def _run_impulse_recovery_test(self, duration: float) -> dict | None:
        """Run impulse recovery test.

        Commands 5.0 m/s forward for 1s, then zero for the remainder.

        Args:
            duration: Total test duration [s].

        Returns:
            Dict with vel_x_history (T, N), att_error_history (T, N, 3),
            rate_error_history (T, N, 3), or None if NaN detected.
        """
        import torch

        N = self._num_envs
        self._reset_to_hover()

        v_cmd_on = torch.zeros(N, 3, device=self._device)
        v_cmd_on[:, 0] = 5.0
        v_cmd_off = torch.zeros(N, 3, device=self._device)
        impulse_duration = 1.0  # seconds

        vel_x_history = []
        att_error_history = []
        rate_error_history = []

        num_steps = int(duration / self._dt)
        impulse_steps = int(impulse_duration / self._dt)

        for step in range(num_steps):
            vel = self.robot.data.root_lin_vel_w[:N]
            if torch.isnan(vel).any():
                return None

            v_cmd = v_cmd_on if step < impulse_steps else v_cmd_off
            att_error, rate_error = self._step_with_controller(v_cmd)

            vel_x_history.append(self.robot.data.root_lin_vel_w[:N, 0].clone())
            att_error_history.append(att_error[:N].clone())
            rate_error_history.append(rate_error[:N].clone())

        return {
            "vel_x_history": torch.stack(vel_x_history),         # (T, N)
            "att_error_history": torch.stack(att_error_history),    # (T, N, 3)
            "rate_error_history": torch.stack(rate_error_history),  # (T, N, 3)
        }

    def evaluate_batch(self, params_list: list) -> tuple[list[dict], list[dict]]:
        """Run all 5 tests for each gain set and collect histories.

        Args:
            params_list: List of ParameterSet (length == num_envs).

        Returns:
            per_env_histories: List of dicts, one per env, each containing
                test_name → history dict.
            failed_envs: List of env indices that produced NaN.
        """
        import torch

        N = self._num_envs
        assert len(params_list) == N, f"Expected {N} params, got {len(params_list)}"

        # Set per-env gains
        self._setup_batch(params_list)

        # Run each test at PX4-matched durations
        test_results = {}
        for test_name in self.targets:
            duration = self.targets[test_name].t[-1]
            print(f"  Running {test_name} ({duration:.1f}s)...", end="", flush=True)

            if test_name == "hover_drift":
                data = self._run_hover_test(duration)
            elif test_name == "vel_step_5":
                target_speed = TEST_COMMANDS[test_name]["target_speed"]
                data = self._run_velocity_test(duration, target_speed)
            elif test_name == "vel_impulse_recovery":
                data = self._run_impulse_recovery_test(duration)
            elif test_name == "yaw_step":
                data = self._run_yaw_step_test(duration)
            else:
                print(f" SKIP (unknown test)", flush=True)
                continue

            if data is None:
                print(f" FAILED (NaN)", flush=True)
            else:
                print(f" done", flush=True)
            test_results[test_name] = data

        # Reorganize per-env
        per_env_histories = []
        failed_envs = []
        for i in range(N):
            env_hist = {}
            env_failed = False
            for test_name, data in test_results.items():
                if data is None:
                    env_failed = True
                    continue
                # Extract env i from each tensor
                env_data = {}
                for key, tensor in data.items():
                    if tensor.dim() == 2:  # (T, N)
                        env_data[key] = tensor[:, i].cpu().numpy()
                    elif tensor.dim() == 3:  # (T, N, 3)
                        env_data[key] = tensor[:, i, :].cpu().numpy()
                env_hist[test_name] = env_data
            per_env_histories.append(env_hist)
            if env_failed:
                failed_envs.append(i)

        return per_env_histories, failed_envs

    def score_env(self, env_histories: dict[str, dict]) -> tuple[float, dict]:
        """Score a single environment's histories against PX4 targets.

        Args:
            env_histories: {test_name: {signal_name: np.ndarray}} for one env.

        Returns:
            total_score: Weighted MSE across all tests.
            per_test_scores: {test_name: {vel_mse, att_mse, rate_mse, weighted}}.
        """
        per_test_scores = {}
        total_weighted = 0.0
        num_tests = 0

        for test_name, data in env_histories.items():
            if test_name not in self.targets:
                continue
            target = self.targets[test_name]
            scores = self._compute_timeseries_mse(data, target, test_name)
            per_test_scores[test_name] = scores
            total_weighted += scores["weighted"]
            num_tests += 1

        total_score = total_weighted / max(num_tests, 1)
        return total_score, per_test_scores

    def _compute_timeseries_mse(
        self, iris_data: dict, target: TargetTimeseries, test_name: str
    ) -> dict:
        """Compute weighted MSE between iris_ma6 and PX4 SITL timeseries.

        Trims both to the shorter length, then computes per-signal MSE.

        Args:
            iris_data: Dict of numpy arrays from one env's test.
            target: PX4 SITL target timeseries.
            test_name: Name of the test (for signal selection).

        Returns:
            Dict with vel_mse, att_mse, rate_mse, weighted.
        """
        # Determine common length
        target_len = len(target.t)

        # Velocity MSE (X-axis for velocity tests, yaw for yaw test)
        if test_name == "yaw_step":
            iris_yaw = iris_data.get("yaw_rate_history", np.zeros(1))
            T = min(len(iris_yaw), target_len)
            vel_mse = float(np.mean((iris_yaw[:T] - target.yaw_rate[:T]) ** 2))
        elif test_name == "hover_drift":
            iris_vel = iris_data.get("vel_history", np.zeros((1, 3)))
            T = min(len(iris_vel), target_len)
            # For hover, match vel_x only
            vel_mse = float(np.mean((iris_vel[:T, 0] - target.vel_x[:T]) ** 2))
        else:
            # vel_step_5, vel_step_10, vel_impulse_recovery
            iris_vx = iris_data.get("vel_x_history", np.zeros(1))
            T = min(len(iris_vx), target_len)
            vel_mse = float(np.mean((iris_vx[:T] - target.vel_x[:T]) ** 2))

        # Attitude error MSE (3-axis)
        iris_att = iris_data.get("att_error_history", np.zeros((1, 3)))
        T_att = min(len(iris_att), target_len)
        att_mse = float(np.mean((iris_att[:T_att] - target.att_error[:T_att]) ** 2))

        # Rate error MSE (3-axis)
        iris_rate = iris_data.get("rate_error_history", np.zeros((1, 3)))
        T_rate = min(len(iris_rate), target_len)
        rate_mse = float(np.mean((iris_rate[:T_rate] - target.rate_error[:T_rate]) ** 2))

        # Weighted combination
        w = self.weights
        weighted = w.velocity * vel_mse + w.attitude * att_mse + w.rate * rate_mse

        return {
            "vel_mse": vel_mse,
            "att_mse": att_mse,
            "rate_mse": rate_mse,
            "weighted": weighted,
        }

    def _compute_metric_comparison(
        self, iris_data: dict, target: TargetTimeseries, test_name: str
    ) -> dict:
        """Compute metric-based comparison (settling time, SS error, etc.).

        Secondary scoring — for the mismatch improvement table.

        Args:
            iris_data: Dict of numpy arrays from one env's test.
            target: PX4 SITL target timeseries.
            test_name: Name of the test.

        Returns:
            Dict of {metric_name: {iris_ma6: val, px4_sitl: val, diff_pct: val}}.
        """
        comparison = {}

        if test_name in ("vel_step_5", "vel_step_10"):
            target_speed = TEST_COMMANDS[test_name]["target_speed"]
            dt = 0.01

            # iris_ma6 metrics
            iris_vx = iris_data.get("vel_x_history", np.zeros(1))
            iris_settling = self._find_settling_time(iris_vx, target_speed, dt)
            iris_ss_err = self._compute_ss_error(iris_vx, target_speed)

            # PX4 metrics
            px4_vx = target.vel_x
            px4_settling = self._find_settling_time(px4_vx, target_speed, dt)
            px4_ss_err = self._compute_ss_error(px4_vx, target_speed)

            speed_label = test_name.replace("vel_step_", "")
            comparison[f"vel_{speed_label}_settling_time"] = self._metric_entry(
                iris_settling, px4_settling, "s"
            )
            comparison[f"vel_{speed_label}_ss_error"] = self._metric_entry(
                iris_ss_err, px4_ss_err, "m/s"
            )

        elif test_name == "hover_drift":
            iris_vel = iris_data.get("vel_history", np.zeros((1, 3)))
            # Approximate drift from velocity integral (simple)
            iris_drift = float(np.mean(np.sqrt(iris_vel[:, 0] ** 2 + iris_vel[:, 1] ** 2)))
            px4_drift = float(np.mean(np.sqrt(target.vel_x ** 2)))  # Approximate
            comparison["hover_vel_magnitude"] = self._metric_entry(
                iris_drift, px4_drift, "m/s"
            )

        elif test_name == "yaw_step":
            iris_yr = iris_data.get("yaw_rate_history", np.zeros(1))
            px4_yr = target.yaw_rate
            dt = 0.01

            iris_settling = self._find_settling_time(iris_yr, 0.5, dt)
            px4_settling = self._find_settling_time(px4_yr, 0.5, dt)
            comparison["yaw_settling_time"] = self._metric_entry(
                iris_settling, px4_settling, "s"
            )

        return comparison

    @staticmethod
    def _find_settling_time(signal: np.ndarray, target: float, dt: float) -> float:
        """Find time when signal first reaches 95% of target."""
        threshold = 0.95 * target
        indices = np.where(signal >= threshold)[0]
        if len(indices) == 0:
            return len(signal) * dt  # Never settled
        return float(indices[0] * dt)

    @staticmethod
    def _compute_ss_error(signal: np.ndarray, target: float) -> float:
        """Compute steady-state error (mean of last 0.5s at 100 Hz = 50 samples)."""
        last_n = min(50, len(signal))
        return float(abs(target - np.mean(signal[-last_n:])))

    @staticmethod
    def _metric_entry(iris_val: float, px4_val: float, unit: str) -> dict:
        """Create a comparison entry with diff_pct."""
        if abs(px4_val) > 1e-9:
            diff_pct = abs(iris_val - px4_val) / abs(px4_val) * 100.0
        else:
            diff_pct = 0.0 if abs(iris_val) < 1e-9 else float("inf")
        return {
            "iris_ma6": iris_val,
            "px4_sitl": px4_val,
            "diff_pct": diff_pct,
            "unit": unit,
        }

    def close(self):
        """Close environment."""
        self.env.close()


# =============================================================================
# Comparison plots
# =============================================================================


def generate_comparison_plots(
    best_histories: dict[str, dict],
    targets: dict[str, TargetTimeseries],
    output_dir: Path,
    dt: float,
    baseline_histories: dict[str, dict] | None = None,
):
    """Generate per-test PDF plots overlaying iris_ma6 best vs PX4 SITL.

    Each plot has 3 rows: velocity/yaw_rate, attitude error, rate error.
    If baseline_histories is provided, adds a third trace for current gains.

    Args:
        best_histories: Best candidate's per-test histories.
        targets: PX4 SITL target timeseries.
        output_dir: Directory to save PDFs.
        dt: Physics timestep [s].
        baseline_histories: Optional current-gains histories for 3-way comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for test_name, iris_data in best_histories.items():
        if test_name not in targets:
            continue
        target = targets[test_name]
        base_data = baseline_histories.get(test_name) if baseline_histories else None

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Time axes
        iris_len = _get_iris_length(iris_data)
        t_iris = np.arange(iris_len) * dt
        t_px4 = target.t[:len(target.vel_x)]
        t_base = np.arange(_get_iris_length(base_data)) * dt if base_data else None

        # Row 0: Velocity or yaw rate
        ax = axes[0]
        if test_name == "yaw_step":
            iris_sig = iris_data.get("yaw_rate_history", np.zeros(1))
            ax.plot(t_iris[:len(iris_sig)], iris_sig, "b-", label="PX4-matched", linewidth=1.0)
            ax.plot(t_px4, target.yaw_rate, "r--", label="PX4 SITL", linewidth=1.0, alpha=0.8)
            if base_data is not None:
                base_sig = base_data.get("yaw_rate_history", np.zeros(1))
                ax.plot(t_base[:len(base_sig)], base_sig, "g-.", label="current gains", linewidth=1.0, alpha=0.7)
            ax.set_ylabel("Yaw rate [rad/s]")
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, label="target 0.5 rad/s")
        elif test_name == "hover_drift":
            iris_vel = iris_data.get("vel_history", np.zeros((1, 3)))
            ax.plot(t_iris[:len(iris_vel)], iris_vel[:, 0], "b-", label="PX4-matched vel_x", linewidth=1.0)
            ax.plot(t_px4, target.vel_x, "r--", label="PX4 SITL vel_x", linewidth=1.0, alpha=0.8)
            if base_data is not None:
                base_vel = base_data.get("vel_history", np.zeros((1, 3)))
                ax.plot(t_base[:len(base_vel)], base_vel[:, 0], "g-.", label="current gains vel_x", linewidth=1.0, alpha=0.7)
            ax.set_ylabel("Velocity [m/s]")
        else:
            iris_vx = iris_data.get("vel_x_history", np.zeros(1))
            ax.plot(t_iris[:len(iris_vx)], iris_vx, "b-", label="PX4-matched vel_x", linewidth=1.0)
            ax.plot(t_px4, target.vel_x, "r--", label="PX4 SITL vel_x", linewidth=1.0, alpha=0.8)
            if base_data is not None:
                base_vx = base_data.get("vel_x_history", np.zeros(1))
                ax.plot(t_base[:len(base_vx)], base_vx, "g-.", label="current gains vel_x", linewidth=1.0, alpha=0.7)
            ax.set_ylabel("Velocity [m/s]")
            if "target_speed" in TEST_COMMANDS.get(test_name, {}):
                tgt = TEST_COMMANDS[test_name]["target_speed"]
                ax.axhline(tgt, color="gray", linestyle=":", alpha=0.4, label=f"target {tgt} m/s")
        ax.legend(fontsize=8, loc="best")
        ax.set_title(f"Replicator: {test_name} — PX4-matched (blue) vs PX4 SITL (red) vs current (green)")
        ax.grid(True, alpha=0.3)

        # Row 1: Attitude error (norm only for clarity with 3 traces)
        ax = axes[1]
        iris_att = iris_data.get("att_error_history", np.zeros((1, 3)))
        T_att = min(len(iris_att), len(target.att_error))
        iris_att_norm = np.degrees(np.linalg.norm(iris_att[:T_att], axis=-1))
        px4_att_norm = np.degrees(np.linalg.norm(target.att_error[:T_att], axis=-1))
        ax.plot(t_iris[:T_att], iris_att_norm, "b-", label="PX4-matched", linewidth=0.8)
        ax.plot(t_px4[:T_att], px4_att_norm, "r--", label="PX4 SITL", linewidth=0.8, alpha=0.7)
        if base_data is not None:
            base_att = base_data.get("att_error_history", np.zeros((1, 3)))
            T_base_att = min(len(base_att), T_att)
            base_att_norm = np.degrees(np.linalg.norm(base_att[:T_base_att], axis=-1))
            ax.plot(t_base[:T_base_att], base_att_norm, "g-.", label="current gains", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Attitude error norm [deg]")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # Row 2: Rate error (norm only for clarity with 3 traces)
        ax = axes[2]
        iris_rate = iris_data.get("rate_error_history", np.zeros((1, 3)))
        T_rate = min(len(iris_rate), len(target.rate_error))
        iris_rate_norm = np.degrees(np.linalg.norm(iris_rate[:T_rate], axis=-1))
        px4_rate_norm = np.degrees(np.linalg.norm(target.rate_error[:T_rate], axis=-1))
        ax.plot(t_iris[:T_rate], iris_rate_norm, "b-", label="PX4-matched", linewidth=0.8)
        ax.plot(t_px4[:T_rate], px4_rate_norm, "r--", label="PX4 SITL", linewidth=0.8, alpha=0.7)
        if base_data is not None:
            base_rate = base_data.get("rate_error_history", np.zeros((1, 3)))
            T_base_rate = min(len(base_rate), T_rate)
            base_rate_norm = np.degrees(np.linalg.norm(base_rate[:T_base_rate], axis=-1))
            ax.plot(t_base[:T_base_rate], base_rate_norm, "g-.", label="current gains", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Rate error norm [deg/s]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf_path = output_dir / f"replicator_{test_name}.pdf"
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {pdf_path}", flush=True)


def _get_iris_length(iris_data: dict) -> int:
    """Get timeseries length from first available signal."""
    for val in iris_data.values():
        if isinstance(val, np.ndarray):
            return val.shape[0]
    return 0


# =============================================================================
# Config export
# =============================================================================


def export_px4_matched_config(
    best_params: ParameterSet,
    score: float,
    output_path: Path,
):
    """Write PX4_MATCHED_CONTROLLER_CFG to a Python file.

    Args:
        best_params: Best-matching gain set.
        score: MSE score of the best match.
        output_path: Path to write the config file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_path, "w") as f:
        f.write(f"# PX4 SITL-matched DroneController configuration\n")
        f.write(f"# Generated by sysid_replicator.py: {timestamp}\n")
        f.write(f"# MSE score: {score:.6f}\n")
        f.write(f"#\n")
        f.write(f"# Architecture: Velocity (PI) -> Attitude (P) -> Rate (PID) -> Motor\n")
        f.write(f"# Target: PX4 SITL response (hover, vel_step_5, impulse_recovery, yaw_step)\n\n")
        f.write("from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg\n")
        f.write("from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg\n")
        f.write("from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg\n")
        f.write("from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg\n\n")
        f.write("PX4_MATCHED_CONTROLLER_CFG = DroneControllerCfg(\n")
        f.write("    velocity=VelocityControllerCfg(\n")
        f.write(f"        Kp_vel=({best_params.Kp_vel_xy:.4f}, {best_params.Kp_vel_xy:.4f}, {best_params.Kp_vel_z:.4f}),\n")
        f.write(f"        Ki_vel=({best_params.Ki_vel_xy:.4f}, {best_params.Ki_vel_xy:.4f}, {best_params.Ki_vel_z:.4f}),\n")
        f.write("    ),\n")
        f.write("    attitude=AttitudeControllerCfg(\n")
        f.write(f"        Kp_att=({best_params.Kp_att_rp:.4f}, {best_params.Kp_att_rp:.4f}, {best_params.Kp_att_y:.4f}),\n")
        f.write("    ),\n")
        f.write("    rate=RateControllerCfg(\n")
        f.write(f"        Kp_rate=({best_params.Kp_rate_rp:.4f}, {best_params.Kp_rate_rp:.4f}, {best_params.Kp_rate_y:.4f}),\n")
        f.write(f"        Ki_rate=({best_params.Ki_rate_rp:.4f}, {best_params.Ki_rate_rp:.4f}, {best_params.Ki_rate_y:.4f}),\n")
        f.write(f"        Kd_rate=({best_params.Kd_rate_rp:.5f}, {best_params.Kd_rate_rp:.5f}, {best_params.Kd_rate_y:.5f}),\n")
        f.write("    ),\n")
        f.write(")\n")
    print(f"  Config saved: {output_path}", flush=True)


# =============================================================================
# Main
# =============================================================================


def main():
    """Load PX4 SITL targets and optionally run gain sweep."""
    import sys

    import torch

    print("=" * 80, flush=True)
    print("SYSID REPLICATOR — Match iris_ma6 to PX4 SITL", flush=True)
    print("=" * 80, flush=True)

    sysid_dir = Path(args_cli.sysid_dir)
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else sysid_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load PX4 SITL targets
    print(f"\nLoading PX4 SITL targets from: {sysid_dir}", flush=True)
    loader = PX4TargetLoader(sysid_dir)
    loader.print_summary()
    sys.stdout.flush()

    if not loader.test_names:
        print("ERROR: No test data loaded. Check --sysid-dir path.", flush=True)
        simulation_app.close()
        return

    if args_cli.dry_run:
        print("\n[DRY RUN] Target loading complete. Exiting.", flush=True)
        simulation_app.close()
        return

    # Build targets dict
    targets = {name: loader.get_target(name) for name in loader.test_names}

    # Step 2: Create replicator and run evaluation
    num_trials = args_cli.num_trials
    print(f"\nGenerating {num_trials} random gain candidates...", flush=True)
    params_list = generate_random_params(num_trials)

    print(f"Creating SysidReplicator ({num_trials} parallel envs)...", flush=True)
    replicator = SysidReplicator(
        num_envs=num_trials,
        device="cuda:0",
        aero_level=args_cli.aero_level,
        targets=targets,
    )

    print(f"\nEvaluating {num_trials} candidates...", flush=True)
    start_time = time.time()
    per_env_histories, failed_envs = replicator.evaluate_batch(params_list)
    elapsed = time.time() - start_time

    print(f"\nDone in {elapsed:.1f}s. Failed envs: {len(failed_envs)}/{num_trials}", flush=True)

    # Step 3: Score each candidate against PX4 targets
    print("\nScoring candidates...", flush=True)
    scores = []
    all_comparisons = []
    for i, env_hist in enumerate(per_env_histories):
        if i in failed_envs or not env_hist:
            scores.append(float("inf"))
            all_comparisons.append({})
            continue
        score, per_test = replicator.score_env(env_hist)
        scores.append(score)
        # Compute metric comparison for this env
        comparison = {}
        for test_name, data in env_hist.items():
            if test_name in targets:
                comp = replicator._compute_metric_comparison(data, targets[test_name], test_name)
                comparison.update(comp)
        all_comparisons.append(comparison)

    # Find best
    best_idx = int(np.argmin(scores))
    best_score = scores[best_idx]
    best_params = params_list[best_idx]
    best_comparison = all_comparisons[best_idx]
    best_histories = per_env_histories[best_idx]

    # Count stable (finite score)
    stable_count = sum(1 for s in scores if np.isfinite(s))

    print(f"\n{'=' * 80}", flush=True)
    print("REPLICATOR RESULTS", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"Total time: {elapsed:.1f}s", flush=True)
    print(f"Stable: {stable_count}/{num_trials}", flush=True)
    print(f"Best candidate: env {best_idx}, score: {best_score:.6f}", flush=True)

    # Print best gains
    print(f"\nBest gains:", flush=True)
    print(f"  [Velocity PI]", flush=True)
    print(f"    Kp_vel: ({best_params.Kp_vel_xy:.4f}, {best_params.Kp_vel_xy:.4f}, {best_params.Kp_vel_z:.4f})", flush=True)
    print(f"    Ki_vel: ({best_params.Ki_vel_xy:.4f}, {best_params.Ki_vel_xy:.4f}, {best_params.Ki_vel_z:.4f})", flush=True)
    print(f"  [Attitude P]", flush=True)
    print(f"    Kp_att: ({best_params.Kp_att_rp:.4f}, {best_params.Kp_att_rp:.4f}, {best_params.Kp_att_y:.4f})", flush=True)
    print(f"  [Rate PID]", flush=True)
    print(f"    Kp_rate: ({best_params.Kp_rate_rp:.4f}, {best_params.Kp_rate_rp:.4f}, {best_params.Kp_rate_y:.4f})", flush=True)
    print(f"    Ki_rate: ({best_params.Ki_rate_rp:.4f}, {best_params.Ki_rate_rp:.4f}, {best_params.Ki_rate_y:.4f})", flush=True)
    print(f"    Kd_rate: ({best_params.Kd_rate_rp:.5f}, {best_params.Kd_rate_rp:.5f}, {best_params.Kd_rate_y:.5f})", flush=True)

    # Print per-test MSE breakdown
    _, per_test_scores = replicator.score_env(best_histories)
    print(f"\nPer-test MSE (best candidate):", flush=True)
    print(f"  {'Test':<25} {'vel_mse':>10} {'att_mse':>10} {'rate_mse':>10} {'weighted':>10}", flush=True)
    print(f"  {'-' * 65}", flush=True)
    for test_name, s in per_test_scores.items():
        print(f"  {test_name:<25} {s['vel_mse']:>10.6f} {s['att_mse']:>10.6f} {s['rate_mse']:>10.6f} {s['weighted']:>10.6f}", flush=True)

    # Print mismatch table
    if best_comparison:
        print(f"\nMismatch table (best candidate vs PX4 SITL):", flush=True)
        print(f"  {'Metric':<30} {'iris_ma6':>10} {'px4_sitl':>10} {'diff%':>8} {'status':>8}", flush=True)
        print(f"  {'-' * 70}", flush=True)
        for metric_name, entry in best_comparison.items():
            status = "PASS" if entry["diff_pct"] <= 20.0 else "FAIL"
            print(f"  {metric_name:<30} {entry['iris_ma6']:>10.4f} {entry['px4_sitl']:>10.4f} "
                  f"{entry['diff_pct']:>7.1f}% {status:>8}", flush=True)

    # Save results JSON
    results_data = {
        "config": {
            "num_trials": num_trials,
            "aero_level": args_cli.aero_level,
            "elapsed_time": elapsed,
            "weights": asdict(replicator.weights),
        },
        "best": {
            "env_idx": best_idx,
            "score": best_score,
            "params": asdict(best_params),
            "per_test_mse": per_test_scores,
            "comparison": best_comparison,
        },
        "summary": {
            "stable_count": stable_count,
            "total_count": num_trials,
            "score_min": float(np.nanmin([s for s in scores if np.isfinite(s)])) if stable_count > 0 else None,
            "score_median": float(np.nanmedian([s for s in scores if np.isfinite(s)])) if stable_count > 0 else None,
            "score_max": float(np.nanmax([s for s in scores if np.isfinite(s)])) if stable_count > 0 else None,
        },
    }

    results_file = output_dir / "replicator_metrics.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}", flush=True)

    # Step 4: Run baseline (current gains) for 3-way comparison
    # Reuse existing replicator — fill all envs with the same baseline params
    print("\nRunning baseline with current TUNED_CONTROLLER_CFG...", flush=True)
    from isaaclab_tasks.direct.iris_ma6.controller.tuning.tuning_results import TUNED_CONTROLLER_CFG

    baseline_params = ParameterSet(
        Kp_vel_xy=TUNED_CONTROLLER_CFG.velocity.Kp_vel[0],
        Kp_vel_z=TUNED_CONTROLLER_CFG.velocity.Kp_vel[2],
        Ki_vel_xy=TUNED_CONTROLLER_CFG.velocity.Ki_vel[0],
        Ki_vel_z=TUNED_CONTROLLER_CFG.velocity.Ki_vel[2],
        Kp_att_rp=TUNED_CONTROLLER_CFG.attitude.Kp_att[0],
        Kp_att_y=TUNED_CONTROLLER_CFG.attitude.Kp_att[2],
        Kp_rate_rp=TUNED_CONTROLLER_CFG.rate.Kp_rate[0],
        Kp_rate_y=TUNED_CONTROLLER_CFG.rate.Kp_rate[2],
        Ki_rate_rp=TUNED_CONTROLLER_CFG.rate.Ki_rate[0],
        Ki_rate_y=TUNED_CONTROLLER_CFG.rate.Ki_rate[2],
        Kd_rate_rp=TUNED_CONTROLLER_CFG.rate.Kd_rate[0],
        Kd_rate_y=TUNED_CONTROLLER_CFG.rate.Kd_rate[2],
    )
    baseline_batch = [baseline_params] * num_trials  # Same gains in all envs
    baseline_histories_list, _ = replicator.evaluate_batch(baseline_batch)
    baseline_histories = baseline_histories_list[0] if baseline_histories_list else None

    if baseline_histories:
        baseline_score, _ = replicator.score_env(baseline_histories)
        print(f"  Baseline score: {baseline_score:.6f} (vs best: {best_score:.6f})", flush=True)

    # Step 5: Generate comparison plots (3-way overlay)
    print("\nGenerating comparison plots...", flush=True)
    generate_comparison_plots(best_histories, targets, output_dir, replicator.dt,
                              baseline_histories=baseline_histories)

    # Step 5: Export PX4-matched config
    print("\nExporting PX4-matched config...", flush=True)
    config_path = Path(__file__).parent / "tuning_results" / "px4_matched.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    export_px4_matched_config(best_params, best_score, config_path)

    print(f"\n{'=' * 80}", flush=True)
    print("REPLICATOR COMPLETE", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"  Results JSON:  {results_file}", flush=True)
    print(f"  Comparison PDFs: {output_dir}/replicator_*.pdf", flush=True)
    print(f"  Matched config:  {config_path}", flush=True)

    replicator.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
