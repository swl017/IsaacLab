#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel auto-tuning for DroneController parameters.

Each environment tests a different parameter set simultaneously, achieving ~Nx speedup
where N is the number of parallel environments.

Usage:
    # Grid search with 64 parallel envs
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless

    # Random search with 100 trials
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless --search-mode random --num-trials 100
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto-tune DroneController parameters")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--search-mode", type=str, default="random", choices=["grid", "random"])
parser.add_argument("--num-trials", type=int, default=100, help="Number of trials (random mode). Each trial gets its own parallel env.")
parser.add_argument("--output-dir", type=str, default="tuning_results", help="Output directory")
parser.add_argument("--top-k", type=int, default=10, help="Number of top results to plot step responses for")
parser.add_argument("--aero-level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Aerodynamic fidelity level (0=off, 1=drag, 2=drag+wind, 3=drag+wind+rotor)")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg


@dataclass
class ParameterSet:
    """Controller parameter set.

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

    # Attitude controller (P-only, outputs rate setpoint in rad/s per rad error)
    Kp_att_rp: float = 6.5  # Roll/pitch P-gain (PX4: MC_ROLL_P = 6.5)
    Kp_att_y: float = 2.8   # Yaw P-gain (PX4: MC_YAW_P = 2.8)

    # Rate controller (PID, outputs torque)
    Kp_rate_rp: float = 0.15  # Roll/pitch P-gain (PX4: MC_ROLLRATE_P = 0.15)
    Kp_rate_y: float = 0.2    # Yaw P-gain (PX4: MC_YAWRATE_P = 0.2)
    Ki_rate_rp: float = 0.2   # Roll/pitch I-gain (PX4: MC_ROLLRATE_I = 0.2)
    Ki_rate_y: float = 0.1    # Yaw I-gain (PX4: MC_YAWRATE_I = 0.1)
    Kd_rate_rp: float = 0.003 # Roll/pitch D-gain (PX4: MC_ROLLRATE_D = 0.003)
    Kd_rate_y: float = 0.0    # Yaw D-gain (typically 0)


@dataclass
class OscillationMetrics:
    """Oscillation characterization for a single signal.

    Computed from the error signal after 95% settling time.
    """
    damping_ratio: float = 1.0
    """Damping ratio from logarithmic decrement. 0=undamped, 1=critically damped."""

    zero_crossings: int = 0
    """Number of sign changes in the error signal after settling."""

    ss_amplitude: float = 0.0
    """Peak-to-peak amplitude in the steady-state window."""

    frequency: float = 0.0
    """Dominant oscillation frequency estimated from zero-crossing intervals [Hz]."""


@dataclass
class TuningScoreWeights:
    """Configurable weights for the combined tuning score (lower is better).

    Design: velocity tracking quality and oscillation are the primary
    discriminators. Attitude recovery is secondary — a 20° perturbation
    with drag doesn't fully recover in 3s for any gain set, so it mostly
    adds a constant offset. Hover is easy for all controllers.

    Approximate score breakdown for a "good" trial (score ~15-25):
        hover:       ~0.1  (negligible — all controllers hover well)
        velocity:    ~5-10 (settling + ss_error + overshoot — main differentiator)
        attitude:    ~3-6  (recovery time + final error — secondary)
        oscillation: ~5-10 (damping + amplitude — main differentiator)
    """
    # Hover (small contribution — all controllers hover well)
    hover_drift_mean: float = 1.0
    hover_drift_max: float = 0.5
    # Velocity tracking (primary: this is what we're tuning for)
    vel_settling_time: float = 3.0
    """Penalty per second of settling time. Fast response matters."""
    vel_overshoot: float = 0.5
    """Penalty per % overshoot. Increased from 0.1 — overshoot indicates underdamping."""
    vel_ss_error: float = 8.0
    """Penalty per m/s of steady-state error. Tracking accuracy is critical."""
    # Attitude recovery (secondary: harsh test conditions make full recovery unlikely)
    att_recovery_time: float = 1.0
    """Reduced from 3.0 — recovery time has less variance across gain sets."""
    att_max_error: float = 0.02
    """Reduced from 0.1 — initial transient is similar across all trials."""
    att_final_error: float = 0.3
    """Reduced from 2.0 — partial recovery (14° from 30°) is expected with drag."""
    # Oscillation penalties (primary: this is the ticket's core concern)
    oscillation_damping: float = 10.0
    """Penalty per (1 - damping_ratio). Doubled from 5.0 — underdamping is the main issue."""
    oscillation_zero_crossings: float = 0.5
    """Penalty per zero-crossing count. Increased — more crossings = more oscillatory."""
    oscillation_ss_amplitude: float = 5.0
    """Penalty per peak-to-peak steady-state amplitude [m/s or rad/s]. Increased from 3.0."""


def _default_osc() -> OscillationMetrics:
    return OscillationMetrics()


@dataclass
class TuningMetrics:
    """Evaluation metrics."""
    hover_drift_mean: float = 0.0
    hover_drift_max: float = 0.0
    vel_settling_time: float = 0.0
    vel_overshoot: float = 0.0
    vel_ss_error: float = 0.0
    # Attitude recovery metrics
    att_recovery_time: float = 0.0
    att_max_error: float = 0.0
    att_final_error: float = 0.0
    att_recovered: bool = True
    is_stable: bool = True
    score: float = 0.0
    # Oscillation metrics
    vel_osc_5: OscillationMetrics = field(default_factory=_default_osc)
    """Velocity error oscillation at 5 m/s target."""
    vel_osc_10: OscillationMetrics = field(default_factory=_default_osc)
    """Velocity error oscillation at 10 m/s target."""
    att_osc: OscillationMetrics = field(default_factory=_default_osc)
    """Attitude error oscillation during attitude recovery test."""
    rate_osc_vel5: OscillationMetrics = field(default_factory=_default_osc)
    """Rate error oscillation during 5 m/s velocity test."""
    rate_osc_vel10: OscillationMetrics = field(default_factory=_default_osc)
    """Rate error oscillation during 10 m/s velocity test."""
    rate_osc_att: OscillationMetrics = field(default_factory=_default_osc)
    """Rate error oscillation during attitude recovery test."""


def compute_oscillation_metrics(
    error_signal: torch.Tensor,
    settling_idx: int,
    dt: float,
) -> OscillationMetrics:
    """Compute oscillation metrics from an error signal after settling.

    Analyzes the portion of the error signal after the 95% settling index
    to characterize residual oscillation.

    Args:
        error_signal: (T,) 1D error signal (e.g. velocity error along one axis).
        settling_idx: Index at which the signal first reaches 95% of target.
            Steady-state analysis starts from this index.
        dt: Timestep between samples [s].

    Returns:
        OscillationMetrics with damping ratio, zero-crossing count,
        steady-state amplitude, and dominant frequency.
    """
    T = error_signal.shape[0]

    # Ensure settling_idx is within bounds, leave at least 4 samples for analysis
    settling_idx = max(0, min(settling_idx, T - 4))
    ss_signal = error_signal[settling_idx:]

    if ss_signal.shape[0] < 4:
        return OscillationMetrics()

    # Move to CPU for scalar analysis
    ss = ss_signal.detach().cpu().float()

    # --- Zero crossings ---
    signs = torch.sign(ss)
    # Remove exact zeros (treat as previous sign)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    sign_changes = (signs[1:] * signs[:-1]) < 0
    zero_crossings = int(sign_changes.sum().item())

    # --- Peak-to-peak steady-state amplitude ---
    ss_amplitude = float((ss.max() - ss.min()).item())

    # --- Damping ratio via logarithmic decrement ---
    damping_ratio = 1.0  # Default: critically damped (no oscillation detected)

    # Find peaks (local maxima) in absolute error
    abs_ss = ss.abs()
    peaks = []
    for i in range(1, len(abs_ss) - 1):
        if abs_ss[i] > abs_ss[i - 1] and abs_ss[i] > abs_ss[i + 1]:
            peaks.append((i, abs_ss[i].item()))

    if len(peaks) >= 2:
        # Use first two peaks for logarithmic decrement
        peak1_val = peaks[0][1]
        peak2_val = peaks[1][1]

        if peak1_val > 1e-8 and peak2_val > 1e-8 and peak1_val > peak2_val:
            log_dec = math.log(peak1_val / peak2_val)
            damping_ratio = log_dec / math.sqrt(4.0 * math.pi**2 + log_dec**2)
            damping_ratio = min(max(damping_ratio, 0.0), 1.0)
        elif peak1_val > 1e-8 and peak2_val >= peak1_val:
            # Growing or constant oscillation — undamped
            damping_ratio = 0.0

    # --- Dominant frequency from zero-crossing intervals ---
    frequency = 0.0
    if zero_crossings >= 2:
        crossing_indices = sign_changes.nonzero(as_tuple=False).squeeze(-1)
        if crossing_indices.dim() == 0:
            crossing_indices = crossing_indices.unsqueeze(0)
        if len(crossing_indices) >= 2:
            intervals = (crossing_indices[1:] - crossing_indices[:-1]).float() * dt
            mean_half_period = intervals.mean().item()
            if mean_half_period > 1e-8:
                frequency = 1.0 / (2.0 * mean_half_period)

    return OscillationMetrics(
        damping_ratio=damping_ratio,
        zero_crossings=zero_crossings,
        ss_amplitude=ss_amplitude,
        frequency=frequency,
    )


@dataclass
class AttitudeTestConfig:
    """Configuration for deterministic attitude step response test.

    Uses fixed extreme attitudes for consistent, reproducible characterization
    of step response across different parameter sets.
    """
    # Fixed initial attitudes for step response (deterministic).
    # Must be within the real controller's recovery envelope (max_tilt=45°
    # leaves zero margin at 45°; 20° gives a realistic step with headroom).
    roll_deg: float = 20.0   # Initial roll perturbation
    pitch_deg: float = 20.0  # Initial pitch perturbation
    yaw_deg: float = 15.0    # Initial yaw perturbation
    # Initial angular rates (zero for clean step response)
    roll_rate: float = 0.0   # rad/s
    pitch_rate: float = 0.0  # rad/s
    yaw_rate: float = 0.0    # rad/s
    test_duration: float = 3.0  # seconds
    recovery_threshold_deg: float = 5.0  # Attitude considered recovered when error < this


class ParallelTuner:
    """Parallel parameter tuner using Isaac Sim and the real DroneController.

    Tests N different parameter sets simultaneously in N parallel environments.
    Each env gets its own per-env gains set on the shared DroneController.
    """

    def __init__(self, num_envs: int, device: str = "cuda:0", aero_level: int = 0):
        """Initialize tuner with Isaac Sim environment and real DroneController."""
        self.num_envs = num_envs
        self.device = device

        # Create environment
        print("Creating Isaac Sim environment...", flush=True)
        task_name = "Isaac-Iris-MA6-Direct-Test-v0"
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        self.env = gym.make(task_name, cfg=env_cfg)
        self.unwrapped = self.env.unwrapped

        # Get references
        agent_id = self.unwrapped.cfg.possible_agents[0]
        self.robot = self.unwrapped._robots[agent_id]
        self.sim = self.unwrapped.sim
        self.physics_dt = self.unwrapped.cfg.sim.dt  # Single physics step dt
        self.dt = self.physics_dt  # Step at physics rate for maximum resolution

        # Find the correct body ID for force application
        body_ids, _ = self.robot.find_bodies("body")
        self.body_ids = body_ids

        # Get mass from physics view
        self.env.reset()
        all_masses = self.robot.root_physx_view.get_masses()
        per_env_mass = all_masses[0].sum().item()
        self.mass = per_env_mass if per_env_mass > 0.1 else 1.5
        print(f"Environment: {num_envs} envs, dt={self.dt:.3f}s, mass={self.mass:.2f}kg", flush=True)

        # Create the real DroneController (batched over all envs)
        from isaaclab_tasks.direct.iris_ma6.controller import DroneController, DroneControllerCfg
        from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
        cfg = DroneControllerCfg(
            velocity=VelocityControllerCfg(
                # Raise integral limit so Ki up to 2.0 can fully compensate drag at 10 m/s
                # Required: Ki * limit > F_drag/m = 3.78 m/s². With Ki=2.0, limit=4 gives 8.0.
                # But lower Ki (e.g. 0.5) with limit=4 gives only 2.0 — not enough.
                # Raise limit to 8.0 so Ki=0.5 can reach 4.0 m/s².
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

    def _setup_batch(self, params_list: list):
        """Set per-environment gains on the real DroneController.

        Builds (N, 3) gain tensors from the params_list and calls the
        sub-controller set_gains() APIs. len(params_list) must equal num_envs.
        """
        assert len(params_list) == self.num_envs, (
            f"params_list length ({len(params_list)}) must equal num_envs ({self.num_envs})"
        )
        # All envs are active — one param set per env

        # Build per-env gain tensors — (N, 3) with one row per env
        Kp_vel = torch.tensor(
            [[p.Kp_vel_xy, p.Kp_vel_xy, p.Kp_vel_z] for p in params_list],
            dtype=torch.float32, device=self.device,
        )
        Ki_vel = torch.tensor(
            [[p.Ki_vel_xy, p.Ki_vel_xy, p.Ki_vel_z] for p in params_list],
            dtype=torch.float32, device=self.device,
        )
        Kp_att = torch.tensor(
            [[p.Kp_att_rp, p.Kp_att_rp, p.Kp_att_y] for p in params_list],
            dtype=torch.float32, device=self.device,
        )
        Kp_rate = torch.tensor(
            [[p.Kp_rate_rp, p.Kp_rate_rp, p.Kp_rate_y] for p in params_list],
            dtype=torch.float32, device=self.device,
        )
        Ki_rate = torch.tensor(
            [[p.Ki_rate_rp, p.Ki_rate_rp, p.Ki_rate_y] for p in params_list],
            dtype=torch.float32, device=self.device,
        )
        Kd_rate = torch.tensor(
            [[p.Kd_rate_rp, p.Kd_rate_rp, p.Kd_rate_y] for p in params_list],
            dtype=torch.float32, device=self.device,
        )

        # Apply to real controller sub-modules
        self.controller._velocity.set_gains(Kp_vel=Kp_vel, Ki_vel=Ki_vel)
        self.controller._attitude.set_gains(Kp_att=Kp_att)
        self.controller._rate.set_gains(Kp_rate=Kp_rate, Ki_rate=Ki_rate, Kd_rate=Kd_rate)

    def _step_with_controller(self, v_cmd: torch.Tensor) -> tuple:
        """Run one physics step through the real DroneController.

        Steps at physics_dt (0.01s) — not policy_dt — for maximum time resolution.
        The controller is called every physics step so all 4 cascade loops run
        at full rate, and oscillation signals are sampled at 100 Hz.

        Args:
            v_cmd: (num_envs, 3) desired velocity in world frame [m/s].

        Returns:
            att_error: (num_envs, 3) attitude error from the attitude controller [rad].
            rate_error: (num_envs, 3) rate error (rate_setpoint - omega) [rad/s].
        """
        N = self.num_envs
        q_body = self.robot.data.root_quat_w[:N]
        v_body = self.robot.data.root_lin_vel_w[:N]
        omega_body = self.robot.data.root_ang_vel_b[:N]

        # Read gimbal joint positions (pitch, yaw, roll order in sim)
        gimbal_pos = torch.zeros(N, 3, device=self.device)

        # Zero gimbal/yaw/zoom commands — tuning focuses on flight dynamics
        zeros_n = torch.zeros(N, device=self.device)

        # Call controller at physics_dt — all cascade loops get one substep
        F_body, tau_body, gimbal_targets, gimbal_vels, zoom = self.controller.step_policy(
            v_cmd=v_cmd,
            yaw_rate_cmd=zeros_n,
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

        # Extract intermediate signals for oscillation analysis
        q_des = self.controller._last_q_des if hasattr(self.controller, '_last_q_des') else q_body
        att_error = self.controller._attitude.compute_attitude_error(q_des, q_body)  # (N, 3) rad

        rate_setpoint = -self.controller._attitude._Kp_att * att_error
        rate_error = rate_setpoint - omega_body  # (N, 3) rad/s

        # Apply forces and step physics once
        forces = torch.zeros(N, 1, 3, device=self.device)
        torques = torch.zeros(N, 1, 3, device=self.device)
        forces[:, 0] = F_body
        torques[:, 0] = tau_body

        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.body_ids)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(self.physics_dt)

        return att_error, rate_error

    def _reset_controller_state(self):
        """Reset the real DroneController for all environments."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.controller.reset(all_ids)

    def _generate_step_response_initial_conditions(self, att_cfg: AttitudeTestConfig) -> tuple:
        """Generate deterministic initial conditions for step response characterization.

        All environments receive the SAME extreme initial attitude. This ensures
        consistent, reproducible step response metrics across different parameter sets.

        Args:
            att_cfg: Attitude test configuration with fixed initial values

        Returns:
            quat: Quaternion (N, 4) in wxyz format - same for all envs
            ang_vel: Angular velocity (N, 3) in body frame (rad/s) - same for all envs
        """
        N = self.num_envs

        # Fixed Euler angles (same extreme attitude for all environments)
        roll = torch.full((N,), att_cfg.roll_deg, device=self.device)
        pitch = torch.full((N,), att_cfg.pitch_deg, device=self.device)
        yaw = torch.full((N,), att_cfg.yaw_deg, device=self.device)

        # Convert to radians
        roll = torch.deg2rad(roll)
        pitch = torch.deg2rad(pitch)
        yaw = torch.deg2rad(yaw)

        quat = quat_from_euler_xyz(roll, pitch, yaw)

        # Fixed angular velocity (typically zero for clean step response)
        ang_vel = torch.zeros(N, 3, device=self.device)
        ang_vel[:, 0] = att_cfg.roll_rate
        ang_vel[:, 1] = att_cfg.pitch_rate
        ang_vel[:, 2] = att_cfg.yaw_rate

        return quat, ang_vel

    def _set_initial_conditions(self, quat: torch.Tensor, ang_vel: torch.Tensor):
        """Set initial attitude and angular velocity for the robot.

        Args:
            quat: Quaternion (N, 4) in wxyz format
            ang_vel: Angular velocity (N, 3) in body frame
        """
        N = quat.shape[0]

        # Get current root state
        root_state = self.robot.data.default_root_state[:N].clone()

        # Set orientation (quaternion in wxyz)
        root_state[:, 3:7] = quat

        # Set angular velocity (body frame)
        root_state[:, 10:13] = ang_vel

        # Set initial height to give room for recovery
        root_state[:, 2] = 2.0  # 2m height

        # Apply to simulation (need to expand to full env size)
        full_root_state = self.robot.data.default_root_state.clone()
        full_root_state[:N] = root_state

        # Write root state
        self.robot.write_root_state_to_sim(full_root_state)
        self.robot.update(self.physics_dt)

    def _reset_to_hover(self):
        """Reset all envs to identical clean hover state.

        Preserves per-env grid positions (from default_root_state) but sets
        level attitude, 2m height, zero velocity. This ensures the only
        variable across envs is the controller gains.
        """
        self.env.reset()
        self._reset_controller_state()

        # Start from default state (preserves per-env grid x,y offsets)
        full_root_state = self.robot.data.default_root_state.clone()
        full_root_state[:, 2] = 2.0          # Height = 2m (override default)
        full_root_state[:, 3] = 1.0          # quat w = 1 (identity)
        full_root_state[:, 4:7] = 0.0        # quat xyz = 0 (level)
        full_root_state[:, 7:10] = 0.0       # Linear velocity = 0
        full_root_state[:, 10:13] = 0.0      # Angular velocity = 0

        self.robot.write_root_state_to_sim(full_root_state)
        self.robot.update(self.physics_dt)

    def _compute_attitude_error_deg(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute attitude error from level (degrees) using the real attitude controller.

        Args:
            quat: Current quaternion (N, 4) in wxyz format.

        Returns:
            Attitude error in degrees (N,) — angle from level attitude.
        """
        q_level = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(quat.shape[0], 4)
        att_error_rad = self.controller._attitude.compute_attitude_error(q_level, quat)  # (N, 3)
        angle_rad = att_error_rad.norm(dim=-1)  # Scalar error magnitude
        return torch.rad2deg(angle_rad)

    def evaluate_batch(self, params_list: list) -> tuple:
        """Evaluate multiple parameter sets in parallel.

        Args:
            params_list: List of ParameterSet to evaluate.

        Returns:
            Tuple of (metrics_list, per_env_histories) where:
                metrics_list: List of TuningMetrics for each parameter set.
                per_env_histories: List of dicts per env, each with keys
                    "vel_5", "vel_10", "att" containing per-env time-series.
                    Returns None if NaN detected (all envs failed).
        """
        self._setup_batch(params_list)
        N = self.num_envs

        # === Hover Test (3 seconds) ===
        self._reset_to_hover()

        initial_pos = self.robot.data.root_pos_w[:N].clone()
        hover_drifts = []

        v_cmd_hover = torch.zeros(N, 3, device=self.device)
        num_hover_steps = int(3.0 / self.dt)
        for step in range(num_hover_steps):
            pos = self.robot.data.root_pos_w[:N]
            if torch.isnan(pos).any():
                print(f"  NaN detected at hover step {step}", flush=True)
                return [TuningMetrics(is_stable=False, score=9999.0)] * N, None

            self._step_with_controller(v_cmd_hover)
            drift = (self.robot.data.root_pos_w[:N] - initial_pos).norm(dim=-1)
            hover_drifts.append(drift.clone())

        hover_drifts = torch.stack(hover_drifts)  # (T, N)

        # === Velocity Test at 5 m/s (5 seconds) ===
        vel_5_data = self._run_velocity_test(duration=5.0, target_speed=5.0)
        if vel_5_data is None:
            return [TuningMetrics(is_stable=False, score=9999.0)] * N, None

        # === Velocity Test at 10 m/s (5 seconds) ===
        vel_10_data = self._run_velocity_test(duration=5.0, target_speed=10.0)
        if vel_10_data is None:
            return [TuningMetrics(is_stable=False, score=9999.0)] * N, None

        # === Attitude Step Response Test (3 seconds) ===
        # Per-env failure tracking — never returns None
        att_data = self._run_attitude_test(AttitudeTestConfig())
        att_env_failed = att_data["failed_envs"]  # (N,) bool

        # === Compute Metrics Per Environment ===
        weights = TuningScoreWeights()
        results = []
        for i in range(N):
            m = TuningMetrics()

            # Hover metrics
            m.hover_drift_mean = hover_drifts[:, i].mean().item()
            m.hover_drift_max = hover_drifts[:, i].max().item()

            # Velocity metrics (use 5 m/s test for settling/overshoot/ss_error)
            vt = vel_5_data["vel_history"][:, i]
            target = 5.0

            threshold = 0.95 * target
            settled = vt >= threshold
            if settled.any():
                m.vel_settling_time = settled.nonzero()[0].item() * self.dt
            else:
                m.vel_settling_time = 5.0

            max_vel = vt.max().item()
            m.vel_overshoot = max(0, (max_vel - target) / target * 100)

            last_steps = max(1, int(0.5 / self.dt))
            m.vel_ss_error = abs(target - vt[-last_steps:].mean().item())

            # Attitude recovery metrics (per-env: failed envs get penalty values)
            this_env_att_failed = bool(att_env_failed[i].item())
            m.att_recovery_time = att_data["recovery_times"][i].item()
            m.att_max_error = att_data["att_errors"][:, i].max().item()
            m.att_final_error = att_data["att_errors"][-1, i].item()
            m.att_recovered = bool(att_data["recovered_flags"][i].item())

            # Oscillation metrics — velocity error at 5 m/s (X axis)
            vel_error_5 = vel_5_data["vel_history"][:, i] - 5.0
            settling_idx_5 = int(m.vel_settling_time / self.dt) if m.vel_settling_time < 5.0 else 0
            m.vel_osc_5 = compute_oscillation_metrics(vel_error_5, settling_idx_5, self.dt)

            # Oscillation metrics — velocity error at 10 m/s (X axis)
            vt_10 = vel_10_data["vel_history"][:, i]
            vel_error_10 = vt_10 - 10.0
            threshold_10 = 0.95 * 10.0
            settled_10 = vt_10 >= threshold_10
            settling_idx_10 = int(settled_10.nonzero()[0].item()) if settled_10.any() else 0
            m.vel_osc_10 = compute_oscillation_metrics(vel_error_10, settling_idx_10, self.dt)

            # Oscillation metrics — attitude (only for non-failed envs)
            if not this_env_att_failed:
                att_err_norm = att_data["att_errors"][:, i]
                att_settling_idx = int(m.att_recovery_time / self.dt) if m.att_recovered else 0
                m.att_osc = compute_oscillation_metrics(att_err_norm, att_settling_idx, self.dt)

                rate_err_att_norm = att_data["rate_error_history"][:, i, :].norm(dim=-1)
                m.rate_osc_att = compute_oscillation_metrics(rate_err_att_norm, att_settling_idx, self.dt)

            # Oscillation metrics — rate error during velocity tests
            rate_err_5_norm = vel_5_data["rate_error_history"][:, i, :].norm(dim=-1)
            m.rate_osc_vel5 = compute_oscillation_metrics(rate_err_5_norm, settling_idx_5, self.dt)

            rate_err_10_norm = vel_10_data["rate_error_history"][:, i, :].norm(dim=-1)
            m.rate_osc_vel10 = compute_oscillation_metrics(rate_err_10_norm, settling_idx_10, self.dt)

            # Stability check — att_final_error is the gate, not att_recovered
            # (att_recovered uses a tight 5° threshold; final error < 15° is sufficient)
            m.is_stable = (
                m.hover_drift_max < 2.0
                and not np.isnan(m.hover_drift_mean)
                and not this_env_att_failed
                and m.att_final_error < 15.0
            )

            # Combined score with oscillation penalties
            if m.is_stable:
                # Base metrics
                base_score = (
                    weights.hover_drift_mean * m.hover_drift_mean
                    + weights.hover_drift_max * m.hover_drift_max
                    + weights.vel_settling_time * m.vel_settling_time
                    + weights.vel_overshoot * m.vel_overshoot
                    + weights.vel_ss_error * m.vel_ss_error
                    + weights.att_recovery_time * m.att_recovery_time
                    + weights.att_max_error * m.att_max_error
                    + weights.att_final_error * m.att_final_error
                )

                # Oscillation penalty — averaged across all 6 signals
                osc_list = [
                    m.vel_osc_5, m.vel_osc_10, m.att_osc,
                    m.rate_osc_vel5, m.rate_osc_vel10, m.rate_osc_att,
                ]
                osc_penalty = 0.0
                for osc in osc_list:
                    osc_penalty += (
                        weights.oscillation_damping * (1.0 - osc.damping_ratio)
                        + weights.oscillation_zero_crossings * osc.zero_crossings
                        + weights.oscillation_ss_amplitude * osc.ss_amplitude
                    )
                osc_penalty /= len(osc_list)

                m.score = base_score + osc_penalty
            else:
                m.score = 9999.0

            results.append(m)

        # Build per-env histories for plotting
        per_env_histories = []
        for i in range(N):
            per_env_histories.append({
                "vel_5": {
                    "vel_history": vel_5_data["vel_history"][:, i],
                    "att_error_history": vel_5_data["att_error_history"][:, i],
                    "rate_error_history": vel_5_data["rate_error_history"][:, i],
                },
                "vel_10": {
                    "vel_history": vel_10_data["vel_history"][:, i],
                    "att_error_history": vel_10_data["att_error_history"][:, i],
                    "rate_error_history": vel_10_data["rate_error_history"][:, i],
                },
                "att": {
                    "att_errors": att_data["att_errors"][:, i],
                    "rate_error_history": att_data["rate_error_history"][:, i],
                },
            })

        return results, per_env_histories

    def _run_velocity_test(self, duration: float, target_speed: float) -> dict | None:
        """Run a velocity step response test.

        Args:
            duration: Test duration [s].
            target_speed: Target forward velocity [m/s].

        Returns:
            Dict with vel_history (T, N), att_error_history (T, N, 3),
            rate_error_history (T, N, 3), or None if NaN detected.
        """
        N = self.num_envs
        self._reset_to_hover()

        v_cmd = torch.zeros(N, 3, device=self.device)
        v_cmd[:, 0] = target_speed

        vel_history = []
        att_error_history = []
        rate_error_history = []

        num_steps = int(duration / self.dt)
        for _ in range(num_steps):
            vel = self.robot.data.root_lin_vel_w[:N]
            if torch.isnan(vel).any():
                return None

            att_error, rate_error = self._step_with_controller(v_cmd)

            vel_history.append(self.robot.data.root_lin_vel_w[:N, 0].clone())
            att_error_history.append(att_error[:N].clone())
            rate_error_history.append(rate_error[:N].clone())

        return {
            "vel_history": torch.stack(vel_history),          # (T, N)
            "att_error_history": torch.stack(att_error_history),  # (T, N, 3)
            "rate_error_history": torch.stack(rate_error_history),  # (T, N, 3)
        }

    def _run_attitude_test(self, att_cfg: AttitudeTestConfig) -> dict | None:
        """Run an attitude step response test.

        Args:
            att_cfg: Attitude test configuration.

        Returns:
            Dict with att_errors (T, N), att_error_3d (T, N, 3),
            rate_error_history (T, N, 3), recovery_times (N,),
            recovered_flags (N,), or None if NaN/crash detected.
        """
        N = self.num_envs
        self._reset_to_hover()

        # Override with tilted attitude for step response test
        init_quat, init_ang_vel = self._generate_step_response_initial_conditions(att_cfg)
        self._set_initial_conditions(init_quat, init_ang_vel)

        att_errors = []  # (T, N) scalar attitude error in degrees
        rate_error_history = []
        recovery_times = torch.full((N,), att_cfg.test_duration, device=self.device)
        recovered_flags = torch.zeros(N, dtype=torch.bool, device=self.device)
        failed_envs = torch.zeros(N, dtype=torch.bool, device=self.device)

        v_cmd = torch.zeros(N, 3, device=self.device)
        num_steps = int(att_cfg.test_duration / self.dt)
        for step in range(num_steps):
            quat = self.robot.data.root_quat_w[:N]
            pos = self.robot.data.root_pos_w[:N]

            # Track per-env failures (NaN or ground crash)
            env_nan = torch.isnan(quat).any(dim=-1)
            env_crash = pos[:, 2] < 0.1
            newly_failed = (env_nan | env_crash) & ~failed_envs
            if newly_failed.any():
                n_new = newly_failed.sum().item()
                n_total = (failed_envs | newly_failed).sum().item()
                print(f"  Attitude test: {n_new} envs failed at step {step}/{num_steps} "
                      f"({n_total}/{N} total failed)", flush=True)
            failed_envs = failed_envs | newly_failed

            # For failed envs, clamp quat to identity to avoid NaN propagation
            quat = torch.where(failed_envs.unsqueeze(-1),
                               torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
                               quat)

            att_error_deg = self._compute_attitude_error_deg(quat)
            # Failed envs get max error
            att_error_deg = torch.where(failed_envs, torch.tensor(999.0, device=self.device), att_error_deg)
            att_errors.append(att_error_deg.clone())

            just_recovered = (att_error_deg < att_cfg.recovery_threshold_deg) & ~recovered_flags & ~failed_envs
            recovery_times[just_recovered] = step * self.dt
            recovered_flags = recovered_flags | just_recovered

            _, rate_error = self._step_with_controller(v_cmd)
            # Zero out rate error for failed envs
            rate_error = torch.where(failed_envs.unsqueeze(-1), torch.zeros_like(rate_error), rate_error)
            rate_error_history.append(rate_error[:N].clone())

        n_failed = failed_envs.sum().item()
        if n_failed > 0:
            print(f"  Attitude test complete: {n_failed}/{N} envs failed", flush=True)

        return {
            "att_errors": torch.stack(att_errors),                  # (T, N)
            "rate_error_history": torch.stack(rate_error_history),  # (T, N, 3)
            "recovery_times": recovery_times,                       # (N,)
            "recovered_flags": recovered_flags,                     # (N,)
            "failed_envs": failed_envs,                             # (N,)
        }

    def close(self):
        """Close environment."""
        self.env.close()


def generate_grid_params() -> list:
    """Generate parameter sets for grid search (4-loop architecture).

    Search space based on PX4 default ranges:
    - Velocity: Kp=1.8-5.0, Ki=0.2-0.8
    - Attitude: Kp=4.0-10.0 (P-only, outputs rate setpoint)
    - Rate: Kp=0.1-0.3, Ki=0.1-0.4, Kd=0.001-0.01
    """
    # Velocity controller (PI)
    Kp_vel_xy_range = [2.0, 3.0, 4.0]
    Ki_vel_xy_range = [0.3, 0.5]

    # Attitude controller (P-only, outputs rate setpoint)
    Kp_att_rp_range = [5.0, 6.5, 8.0]

    # Rate controller (PID)
    Kp_rate_rp_range = [0.10, 0.15, 0.20]
    Ki_rate_rp_range = [0.15, 0.25]

    params_list = []
    for Kp_vel_xy, Ki_vel_xy, Kp_att_rp, Kp_rate_rp, Ki_rate_rp in itertools.product(
        Kp_vel_xy_range, Ki_vel_xy_range, Kp_att_rp_range, Kp_rate_rp_range, Ki_rate_rp_range
    ):
        params = ParameterSet(
            # Velocity controller
            Kp_vel_xy=Kp_vel_xy,
            Kp_vel_z=Kp_vel_xy * 0.7,  # Z typically lower
            Ki_vel_xy=Ki_vel_xy,
            Ki_vel_z=Ki_vel_xy * 0.6,
            # Attitude controller (P-only)
            Kp_att_rp=Kp_att_rp,
            Kp_att_y=Kp_att_rp * 0.43,  # PX4: 2.8/6.5 ≈ 0.43
            # Rate controller (PID)
            Kp_rate_rp=Kp_rate_rp,
            Kp_rate_y=Kp_rate_rp * 1.33,  # PX4: 0.2/0.15 ≈ 1.33
            Ki_rate_rp=Ki_rate_rp,
            Ki_rate_y=Ki_rate_rp * 0.5,  # PX4: 0.1/0.2 = 0.5
            Kd_rate_rp=0.003,  # PX4 default
            Kd_rate_y=0.0,  # Typically 0 for yaw
        )
        params_list.append(params)

    return params_list


def generate_random_params(num_trials: int) -> list:
    """Generate parameter sets for random search (4-loop architecture).

    Search ranges are informed by cascade stability constraints:
    - Inner loops (rate) must be faster than outer loops (velocity)
    - Motor dynamics (tau=10ms) require higher rate gains to compensate lag
    - Low velocity P-gain avoids overdriving the attitude loop
    - D-gain on rate controller damps motor lag oscillation

    Gain coupling: Kp_vel and Kp_rate are sampled jointly to ensure the
    rate controller bandwidth exceeds the velocity controller bandwidth.
    """
    params_list = []
    for _ in range(num_trials):
        # --- Rate controller (innermost — sample first) ---
        # Expanded upper range: motor dynamics need higher gains
        Kp_rate_rp = np.random.uniform(0.10, 0.40)
        Ki_rate_rp = np.random.uniform(0.05, 0.30)
        Kd_rate_rp = np.random.uniform(0.002, 0.015)  # D-gain matters for motor lag

        # --- Attitude controller (middle) ---
        Kp_att_rp = np.random.uniform(4.0, 10.0)

        # --- Velocity controller (outermost — coupled to rate gains) ---
        # Bandwidth constraint: Kp_vel should not exceed what the rate loop
        # can track. With Kp_rate ~ 0.2, Kp_vel > 3 causes oscillation.
        # Scale max Kp_vel with Kp_rate to maintain stability margin.
        Kp_vel_max = min(5.0, 2.0 + 10.0 * Kp_rate_rp)  # Higher rate → allows higher vel
        Kp_vel_xy = np.random.uniform(1.5, Kp_vel_max)
        Ki_vel_xy = np.random.uniform(0.3, 2.0)  # Must be > 0.95 for 10 m/s drag compensation

        params = ParameterSet(
            # Velocity controller (PI)
            Kp_vel_xy=Kp_vel_xy,
            Kp_vel_z=Kp_vel_xy * np.random.uniform(0.6, 0.8),
            Ki_vel_xy=Ki_vel_xy,
            Ki_vel_z=Ki_vel_xy * np.random.uniform(0.5, 0.7),
            # Attitude controller (P-only)
            Kp_att_rp=Kp_att_rp,
            Kp_att_y=Kp_att_rp * np.random.uniform(0.35, 0.5),
            # Rate controller (PID)
            Kp_rate_rp=Kp_rate_rp,
            Kp_rate_y=Kp_rate_rp * np.random.uniform(1.0, 1.5),
            Ki_rate_rp=Ki_rate_rp,
            Ki_rate_y=Ki_rate_rp * np.random.uniform(0.4, 0.6),
            Kd_rate_rp=Kd_rate_rp,
            Kd_rate_y=0.0,  # Typically 0 for yaw
        )
        params_list.append(params)
    return params_list


def main():
    """Run parallel auto-tuning."""
    print("=" * 80)
    print("DRONE CONTROLLER AUTO-TUNING (PARALLEL)")
    print("=" * 80)

    # Output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate parameters — num_envs == num_trials (one param set per env)
    if args_cli.search_mode == "grid":
        params_list = generate_grid_params()
        print(f"Grid search: {len(params_list)} combinations")
    else:
        params_list = generate_random_params(args_cli.num_trials)
        print(f"Random search: {len(params_list)} trials")

    num_envs = len(params_list)
    print(f"Creating {num_envs} parallel environments (one per param set)")
    print("-" * 80)

    # Create tuner sized to match param count
    tuner = ParallelTuner(num_envs=num_envs, device="cuda:0", aero_level=args_cli.aero_level)

    # Single-shot evaluation — all params tested in parallel
    start_time = time.time()
    print(f"Evaluating {num_envs} configs in parallel...", flush=True)

    metrics_list, per_env_histories = tuner.evaluate_batch(params_list)

    # Process results
    results = []
    plot_candidates = []
    best_score = float("inf")
    best_params = None
    best_metrics = None

    for i, (params, metrics) in enumerate(zip(params_list, metrics_list)):
        results.append({"params": asdict(params), "metrics": asdict(metrics)})

        if metrics.score < best_score:
            best_score = metrics.score
            best_params = params
            best_metrics = metrics

        # Store candidate for top-K plotting (only stable results with histories)
        if metrics.is_stable and per_env_histories is not None:
            plot_candidates.append({
                "trial_idx": i,
                "params": params,
                "metrics": metrics,
                "histories": per_env_histories[i],
            })

    stable_count = sum(1 for m in metrics_list if m.is_stable)
    print(f"Done. Stable: {stable_count}/{num_envs}, Best score: {best_score:.3f}")

    # Keep only top-K candidates by score
    plot_candidates.sort(key=lambda x: x["metrics"].score)
    plot_candidates = plot_candidates[:args_cli.top_k]

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Stable: {sum(1 for r in results if r['metrics']['is_stable'])}/{len(results)}")

    if best_params:
        print(f"\nBest configuration (score: {best_score:.4f}):")
        print(f"  [Velocity PI]")
        print(f"    Kp_vel: ({best_params.Kp_vel_xy:.2f}, {best_params.Kp_vel_xy:.2f}, {best_params.Kp_vel_z:.2f})")
        print(f"    Ki_vel: ({best_params.Ki_vel_xy:.2f}, {best_params.Ki_vel_xy:.2f}, {best_params.Ki_vel_z:.2f})")
        print(f"  [Attitude P]")
        print(f"    Kp_att: ({best_params.Kp_att_rp:.2f}, {best_params.Kp_att_rp:.2f}, {best_params.Kp_att_y:.2f})")
        print(f"  [Rate PID]")
        print(f"    Kp_rate: ({best_params.Kp_rate_rp:.3f}, {best_params.Kp_rate_rp:.3f}, {best_params.Kp_rate_y:.3f})")
        print(f"    Ki_rate: ({best_params.Ki_rate_rp:.3f}, {best_params.Ki_rate_rp:.3f}, {best_params.Ki_rate_y:.3f})")
        print(f"    Kd_rate: ({best_params.Kd_rate_rp:.4f}, {best_params.Kd_rate_rp:.4f}, {best_params.Kd_rate_y:.4f})")
        print(f"\n  Hover drift: {best_metrics.hover_drift_mean:.4f}m (max: {best_metrics.hover_drift_max:.4f}m)")
        print(f"  Velocity settling: {best_metrics.vel_settling_time:.3f}s")
        print(f"  Velocity overshoot: {best_metrics.vel_overshoot:.1f}%")
        print(f"  Steady-state error: {best_metrics.vel_ss_error:.4f}m/s")
        print(f"\n  Attitude recovery: {best_metrics.att_recovery_time:.3f}s (recovered: {best_metrics.att_recovered})")
        print(f"  Attitude max error: {best_metrics.att_max_error:.1f}deg")
        print(f"  Attitude final error: {best_metrics.att_final_error:.1f}deg")
        print(f"\n  Oscillation (vel 5m/s):  ζ={best_metrics.vel_osc_5.damping_ratio:.3f}, "
              f"ZC={best_metrics.vel_osc_5.zero_crossings}, "
              f"amp={best_metrics.vel_osc_5.ss_amplitude:.3f}, "
              f"freq={best_metrics.vel_osc_5.frequency:.1f}Hz")
        print(f"  Oscillation (vel 10m/s): ζ={best_metrics.vel_osc_10.damping_ratio:.3f}, "
              f"ZC={best_metrics.vel_osc_10.zero_crossings}, "
              f"amp={best_metrics.vel_osc_10.ss_amplitude:.3f}, "
              f"freq={best_metrics.vel_osc_10.frequency:.1f}Hz")
        print(f"  Oscillation (attitude):  ζ={best_metrics.att_osc.damping_ratio:.3f}, "
              f"ZC={best_metrics.att_osc.zero_crossings}, "
              f"amp={best_metrics.att_osc.ss_amplitude:.3f}, "
              f"freq={best_metrics.att_osc.frequency:.1f}Hz")

    # Generate step response plots for top-K results
    print(f"\nPlot candidates: {len(plot_candidates)} stable results available for plotting")
    if plot_candidates:
        from isaaclab_tasks.direct.iris_ma6.controller.tuning.step_response_plotter import StepResponsePlotter
        plotter = StepResponsePlotter(output_dir / "step_responses")
        plotter.plot_top_k(plot_candidates, k=args_cli.top_k, dt=tuner.dt)
    else:
        print("WARNING: No stable results to plot. All trials may have failed (NaN/crash).")

    # Save results
    output_file = output_dir / f"tuning_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "search_mode": args_cli.search_mode,
                "num_envs": num_envs,
                "num_trials": len(params_list),
                "aero_level": args_cli.aero_level,
                "elapsed_time": elapsed,
            },
            "best": {
                "params": asdict(best_params) if best_params else None,
                "metrics": asdict(best_metrics) if best_metrics else None,
                "score": best_score,
            },
            "all_results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Save best config as Python file
    if best_params:
        config_file = output_dir / f"best_config_{timestamp}.py"
        with open(config_file, "w") as f:
            f.write(f"# Auto-tuned DroneController configuration (4-loop architecture)\n")
            f.write(f"# Generated: {timestamp}\n")
            f.write(f"# Score: {best_score:.4f}\n")
            f.write(f"#\n")
            f.write(f"# Architecture: Velocity (PI) -> Attitude (P) -> Rate (PID) -> Motor\n\n")
            f.write("from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg\n")
            f.write("from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg\n")
            f.write("from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg\n")
            f.write("from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg\n\n")
            f.write("TUNED_CONTROLLER_CFG = DroneControllerCfg(\n")
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
        print(f"Best config saved to: {config_file}")

    print("=" * 80)

    tuner.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
