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
parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
parser.add_argument("--search-mode", type=str, default="random", choices=["grid", "random"])
parser.add_argument("--num-trials", type=int, default=100, help="Number of trials (random mode)")
parser.add_argument("--output-dir", type=str, default="tuning_results", help="Output directory")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import itertools
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


@dataclass
class ParameterSet:
    """Controller parameter set."""
    Kp_vel_xy: float = 3.0
    Kp_vel_z: float = 2.0
    Ki_vel_xy: float = 0.5
    Ki_vel_z: float = 0.3
    Kp_att_rp: float = 8.0
    Kp_att_y: float = 4.0
    Kd_att_rp: float = 2.5
    Kd_att_y: float = 1.0


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


@dataclass
class AttitudeTestConfig:
    """Configuration for attitude perturbation test."""
    max_roll_deg: float = 45.0  # Max initial roll perturbation
    max_pitch_deg: float = 45.0  # Max initial pitch perturbation
    max_yaw_deg: float = 30.0  # Max initial yaw perturbation
    max_roll_rate: float = 2.0  # rad/s
    max_pitch_rate: float = 2.0  # rad/s
    max_yaw_rate: float = 1.0  # rad/s
    test_duration: float = 3.0  # seconds
    recovery_threshold_deg: float = 5.0  # Attitude considered recovered when error < this


class ParallelTuner:
    """Parallel parameter tuner using Isaac Sim.

    Tests N different parameter sets simultaneously in N parallel environments.
    """

    def __init__(self, num_envs: int, device: str = "cuda:0"):
        """Initialize tuner with Isaac Sim environment."""
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
        self.decimation = self.unwrapped.cfg.decimation
        self.physics_dt = self.unwrapped.cfg.sim.dt  # Single physics step dt
        self.dt = self.physics_dt * self.decimation  # Policy dt

        # Find the correct body ID for force application
        body_ids, _ = self.robot.find_bodies("body")
        self.body_ids = body_ids

        # Get mass from physics view
        self.env.reset()
        all_masses = self.robot.root_physx_view.get_masses()
        per_env_mass = all_masses[0].sum().item()
        self.mass = per_env_mass if per_env_mass > 0.1 else 1.5
        print(f"Environment: {num_envs} envs, dt={self.dt:.3f}s, mass={self.mass:.2f}kg", flush=True)

        # Per-env parameters (set by _setup_batch)
        self.Kp_vel = None  # (N, 3)
        self.Ki_vel = None  # (N, 3)
        self.Kp_att = None  # (N, 3)
        self.Kd_att = None  # (N, 3)
        self.integral = None  # (N, 3)
        self.batch_size = 0

    def _setup_batch(self, params_list: list):
        """Setup per-environment parameter tensors."""
        self.batch_size = len(params_list)

        self.Kp_vel = torch.stack([
            torch.tensor([p.Kp_vel_xy, p.Kp_vel_xy, p.Kp_vel_z], dtype=torch.float32)
            for p in params_list
        ]).to(self.device)

        self.Ki_vel = torch.stack([
            torch.tensor([p.Ki_vel_xy, p.Ki_vel_xy, p.Ki_vel_z], dtype=torch.float32)
            for p in params_list
        ]).to(self.device)

        self.Kp_att = torch.stack([
            torch.tensor([p.Kp_att_rp, p.Kp_att_rp, p.Kp_att_y], dtype=torch.float32)
            for p in params_list
        ]).to(self.device)

        self.Kd_att = torch.stack([
            torch.tensor([p.Kd_att_rp, p.Kd_att_rp, p.Kd_att_y], dtype=torch.float32)
            for p in params_list
        ]).to(self.device)

        self.integral = torch.zeros(self.batch_size, 3, device=self.device)

    def _compute_control(self, v_des: torch.Tensor) -> tuple:
        """Compute control with per-environment parameters.

        Args:
            v_des: Desired velocity (N, 3)

        Returns:
            F_body: Force in body frame (N, 3)
            tau_body: Torque in body frame (N, 3)
        """
        N = self.batch_size

        # Get state
        vel = self.robot.data.root_lin_vel_w[:N]
        quat = self.robot.data.root_quat_w[:N]
        omega = self.robot.data.root_ang_vel_b[:N]

        # === Velocity Control (PI) ===
        vel_error = v_des - vel
        self.integral = torch.clamp(self.integral + vel_error * self.dt, -2.0, 2.0)

        # Per-env gains: (N,3) * (N,3) = (N,3)
        a_des = self.Kp_vel * vel_error + self.Ki_vel * self.integral
        a_des[:, 2] += 9.81  # Gravity compensation

        # Clamp desired acceleration magnitude to prevent runaway
        a_des_norm = torch.norm(a_des, dim=-1, keepdim=True)
        max_accel = 20.0  # Max 2g acceleration
        a_des = a_des * torch.clamp(max_accel / (a_des_norm + 1e-6), max=1.0)

        # Thrust magnitude (clamped to reasonable range)
        thrust = self.mass * torch.norm(a_des, dim=-1)
        thrust = torch.clamp(thrust, 0, self.mass * 30.0)  # Max 3g thrust

        # Ensure a_des points somewhat upward for attitude computation
        # If pointing downward, use hover attitude
        a_des_safe = a_des.clone()
        a_des_safe[:, 2] = torch.clamp(a_des[:, 2], min=1.0)  # At least 1 m/s^2 upward

        # Desired attitude from safe acceleration direction
        q_des = self._accel_to_quat(a_des_safe)

        # === Attitude Control (PD) ===
        att_error = self._quat_error(q_des, quat)
        tau = -self.Kp_att * att_error - self.Kd_att * omega
        tau = torch.clamp(tau, -5.0, 5.0)

        # Force in body frame
        F_body = torch.zeros(N, 3, device=self.device)
        F_body[:, 2] = thrust

        return F_body, tau

    def _accel_to_quat(self, a_des: torch.Tensor) -> torch.Tensor:
        """Convert desired acceleration to quaternion (wxyz)."""
        N = a_des.shape[0]

        # Normalize to get body z-axis direction
        z_body = a_des / (torch.norm(a_des, dim=-1, keepdim=True) + 1e-6)

        # Choose x-axis perpendicular to world y and body z
        y_world = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(N, 3)
        x_body = torch.cross(y_world, z_body, dim=-1)
        x_body = x_body / (torch.norm(x_body, dim=-1, keepdim=True) + 1e-6)
        y_body = torch.cross(z_body, x_body, dim=-1)

        # Rotation matrix to quaternion
        R = torch.stack([x_body, y_body, z_body], dim=-1)  # (N, 3, 3)
        return self._rotmat_to_quat(R)

    def _rotmat_to_quat(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion (wxyz)."""
        N = R.shape[0]
        q = torch.zeros(N, 4, device=self.device)

        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        mask = trace > 0

        # Case 1: trace > 0
        if mask.any():
            s = torch.sqrt(trace[mask] + 1.0) * 2
            q[mask, 0] = 0.25 * s
            q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
            q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
            q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

        # Case 2: trace <= 0, use largest diagonal element
        mask2 = ~mask
        if mask2.any():
            # Simplified: assume z is largest (typical for hover)
            s = torch.sqrt(1.0 + R[mask2, 2, 2] - R[mask2, 0, 0] - R[mask2, 1, 1]) * 2
            q[mask2, 0] = (R[mask2, 1, 0] - R[mask2, 0, 1]) / s
            q[mask2, 1] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
            q[mask2, 2] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s
            q[mask2, 3] = 0.25 * s

        # Normalize
        return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    def _quat_error(self, q_des: torch.Tensor, q_curr: torch.Tensor) -> torch.Tensor:
        """Compute attitude error from quaternion error (wxyz convention)."""
        # q_err = q_des^-1 * q_curr
        q_des_inv = q_des * torch.tensor([1, -1, -1, -1], device=self.device)
        q_err = self._quat_multiply(q_des_inv, q_curr)

        # Ensure shortest path
        sign = torch.sign(q_err[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        q_err = q_err * sign

        # Attitude error: 2 * q_err.xyz
        return 2.0 * q_err[:, 1:4]

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication (wxyz convention)."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def _euler_to_quat(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Convert Euler angles (radians) to quaternion (wxyz convention).

        Args:
            roll, pitch, yaw: (N,) tensors of angles in radians

        Returns:
            Quaternion (N, 4) in wxyz format
        """
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=-1)

    def _quat_to_euler(self, q: torch.Tensor) -> tuple:
        """Convert quaternion (wxyz) to Euler angles (roll, pitch, yaw) in radians.

        Args:
            q: Quaternion (N, 4) in wxyz format

        Returns:
            roll, pitch, yaw: (N,) tensors in radians
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _compute_attitude_error_deg(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute attitude error from level (degrees).

        Args:
            quat: Current quaternion (N, 4) in wxyz format

        Returns:
            Attitude error in degrees (N,) - angle from level attitude
        """
        # Level attitude quaternion (identity rotation)
        q_level = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(quat.shape[0], 4)

        # Compute error quaternion
        q_level_inv = q_level * torch.tensor([1, -1, -1, -1], device=self.device)
        q_err = self._quat_multiply(q_level_inv, quat)

        # Ensure shortest path
        sign = torch.sign(q_err[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        q_err = q_err * sign

        # Angle from quaternion: 2 * acos(w)
        angle_rad = 2.0 * torch.acos(torch.clamp(q_err[:, 0], -1.0, 1.0))
        return torch.rad2deg(angle_rad)

    def _generate_random_initial_conditions(self, att_cfg: AttitudeTestConfig) -> tuple:
        """Generate random initial attitude and angular rate.

        Args:
            att_cfg: Attitude test configuration

        Returns:
            quat: Random quaternion (N, 4) in wxyz format
            ang_vel: Random angular velocity (N, 3) in body frame (rad/s)
        """
        N = self.batch_size

        # Random Euler angles (uniform in range)
        roll = torch.deg2rad(torch.empty(N, device=self.device).uniform_(
            -att_cfg.max_roll_deg, att_cfg.max_roll_deg))
        pitch = torch.deg2rad(torch.empty(N, device=self.device).uniform_(
            -att_cfg.max_pitch_deg, att_cfg.max_pitch_deg))
        yaw = torch.deg2rad(torch.empty(N, device=self.device).uniform_(
            -att_cfg.max_yaw_deg, att_cfg.max_yaw_deg))

        quat = self._euler_to_quat(roll, pitch, yaw)

        # Random angular velocity (uniform in range)
        ang_vel = torch.zeros(N, 3, device=self.device)
        ang_vel[:, 0] = torch.empty(N, device=self.device).uniform_(
            -att_cfg.max_roll_rate, att_cfg.max_roll_rate)
        ang_vel[:, 1] = torch.empty(N, device=self.device).uniform_(
            -att_cfg.max_pitch_rate, att_cfg.max_pitch_rate)
        ang_vel[:, 2] = torch.empty(N, device=self.device).uniform_(
            -att_cfg.max_yaw_rate, att_cfg.max_yaw_rate)

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

    def _step_physics(self, F_body: torch.Tensor, tau_body: torch.Tensor):
        """Apply forces and step simulation for one policy timestep.

        This steps the physics by self.decimation steps (e.g., 4x at 0.01s each = 0.04s total).
        """
        # Expand to full environment size
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:self.batch_size, 0] = F_body
        torques[:self.batch_size, 0] = tau_body

        # Step physics multiple times (decimation) - matching direct_marl_env.py pattern
        for _ in range(self.decimation):
            # Apply forces (body frame)
            self.robot.set_external_force_and_torque(forces, torques, body_ids=self.body_ids)

            # Write to sim and step (matching scene.write_data_to_sim pattern)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)

            # Update robot state at physics dt (CRITICAL - must be inside loop!)
            self.robot.update(self.physics_dt)

        # Render once per policy step
        self.sim.render()

    def evaluate_batch(self, params_list: list) -> list:
        """Evaluate multiple parameter sets in parallel.

        Args:
            params_list: List of ParameterSet to evaluate

        Returns:
            List of TuningMetrics for each parameter set
        """
        self._setup_batch(params_list)
        N = self.batch_size

        # === Hover Test (3 seconds) ===
        self.env.reset()
        self.integral.zero_()
        self.robot.update(self.physics_dt)

        initial_pos = self.robot.data.root_pos_w[:N].clone()
        hover_drifts = []

        num_hover_steps = int(3.0 / self.dt)
        for step in range(num_hover_steps):
            # Check for NaN
            pos = self.robot.data.root_pos_w[:N]
            if torch.isnan(pos).any():
                print(f"  NaN detected at step {step}", flush=True)
                return [TuningMetrics(is_stable=False, score=9999.0)] * N

            v_des = torch.zeros(N, 3, device=self.device)
            F, tau = self._compute_control(v_des)
            self._step_physics(F, tau)

            drift = (self.robot.data.root_pos_w[:N] - initial_pos).norm(dim=-1)
            hover_drifts.append(drift.clone())


        hover_drifts = torch.stack(hover_drifts)  # (T, N)

        # === Velocity Test (2 seconds) ===
        self.env.reset()
        self.integral.zero_()
        self.robot.update(self.physics_dt)
        vel_history = []

        v_des = torch.zeros(N, 3, device=self.device)
        v_des[:, 0] = 3.0  # Target: 3 m/s forward

        num_vel_steps = int(2.0 / self.dt)
        for _ in range(num_vel_steps):
            vel = self.robot.data.root_lin_vel_w[:N]
            if torch.isnan(vel).any():
                return [TuningMetrics(is_stable=False, score=9999.0)] * N

            F, tau = self._compute_control(v_des)
            self._step_physics(F, tau)

            vel_history.append(self.robot.data.root_lin_vel_w[:N, 0].clone())

        vel_history = torch.stack(vel_history)  # (T, N)

        # === Attitude Recovery Test (3 seconds) ===
        att_cfg = AttitudeTestConfig()
        self.env.reset()
        self.integral.zero_()

        # Generate and set random initial conditions
        init_quat, init_ang_vel = self._generate_random_initial_conditions(att_cfg)
        self._set_initial_conditions(init_quat, init_ang_vel)

        att_errors = []  # (T, N) attitude error in degrees
        recovery_times = torch.full((N,), att_cfg.test_duration, device=self.device)  # Default to max time
        recovered_flags = torch.zeros(N, dtype=torch.bool, device=self.device)

        num_att_steps = int(att_cfg.test_duration / self.dt)
        for step in range(num_att_steps):
            quat = self.robot.data.root_quat_w[:N]
            pos = self.robot.data.root_pos_w[:N]

            # Check for NaN or crashed (hit ground)
            if torch.isnan(quat).any() or (pos[:, 2] < 0.1).any():
                print(f"  NaN/crash detected at attitude test step {step}", flush=True)
                return [TuningMetrics(is_stable=False, att_recovered=False, score=9999.0)] * N

            # Compute attitude error from level
            att_error_deg = self._compute_attitude_error_deg(quat)
            att_errors.append(att_error_deg.clone())

            # Check for recovery (first time error < threshold)
            just_recovered = (att_error_deg < att_cfg.recovery_threshold_deg) & ~recovered_flags
            recovery_times[just_recovered] = step * self.dt
            recovered_flags = recovered_flags | just_recovered

            # Control: try to stabilize to hover
            v_des = torch.zeros(N, 3, device=self.device)
            F, tau = self._compute_control(v_des)
            self._step_physics(F, tau)

        att_errors = torch.stack(att_errors)  # (T, N)

        # === Compute Metrics Per Environment ===
        results = []
        for i in range(N):
            m = TuningMetrics()

            # Hover metrics
            m.hover_drift_mean = hover_drifts[:, i].mean().item()
            m.hover_drift_max = hover_drifts[:, i].max().item()

            # Velocity metrics
            vt = vel_history[:, i]
            target = 3.0

            # Settling time (95% of target)
            threshold = 0.95 * target
            settled = vt >= threshold
            if settled.any():
                m.vel_settling_time = settled.nonzero()[0].item() * self.dt
            else:
                m.vel_settling_time = 2.0

            # Overshoot
            max_vel = vt.max().item()
            m.vel_overshoot = max(0, (max_vel - target) / target * 100)

            # Steady-state error (last 0.5s)
            last_steps = max(1, int(0.5 / self.dt))
            m.vel_ss_error = abs(target - vt[-last_steps:].mean().item())

            # Attitude recovery metrics
            m.att_recovery_time = recovery_times[i].item()
            m.att_max_error = att_errors[:, i].max().item()
            m.att_final_error = att_errors[-1, i].item()
            m.att_recovered = bool(recovered_flags[i].item())

            # Stability check - must pass all tests
            m.is_stable = (
                m.hover_drift_max < 2.0 and
                not np.isnan(m.hover_drift_mean) and
                m.att_recovered and
                m.att_final_error < 10.0  # Must stabilize to within 10 degrees
            )

            # Combined score (lower is better)
            if m.is_stable:
                m.score = (
                    # Hover performance
                    1.0 * m.hover_drift_mean +
                    0.5 * m.hover_drift_max +
                    # Velocity tracking
                    2.0 * m.vel_settling_time +
                    0.1 * m.vel_overshoot +
                    5.0 * m.vel_ss_error +
                    # Attitude recovery (heavily weighted)
                    3.0 * m.att_recovery_time +
                    0.1 * m.att_max_error +
                    2.0 * m.att_final_error
                )
            else:
                m.score = 9999.0

            results.append(m)

        return results

    def close(self):
        """Close environment."""
        self.env.close()


def generate_grid_params() -> list:
    """Generate parameter sets for grid search."""
    Kp_vel_xy_range = [2.0, 3.0, 4.0, 5.0]
    Kp_vel_z_range = [1.5, 2.0, 3.0]
    Ki_vel_xy_range = [0.3, 0.5, 0.8]
    Kp_att_rp_range = [6.0, 8.0, 10.0, 12.0]
    Kd_att_rp_range = [1.5, 2.5, 3.5]

    params_list = []
    for Kp_vel_xy, Kp_vel_z, Ki_vel_xy, Kp_att_rp, Kd_att_rp in itertools.product(
        Kp_vel_xy_range, Kp_vel_z_range, Ki_vel_xy_range, Kp_att_rp_range, Kd_att_rp_range
    ):
        params = ParameterSet(
            Kp_vel_xy=Kp_vel_xy,
            Kp_vel_z=Kp_vel_z,
            Ki_vel_xy=Ki_vel_xy,
            Ki_vel_z=Ki_vel_xy * 0.6,
            Kp_att_rp=Kp_att_rp,
            Kp_att_y=Kp_att_rp * 0.5,
            Kd_att_rp=Kd_att_rp,
            Kd_att_y=Kd_att_rp * 0.4,
        )
        params_list.append(params)

    return params_list


def generate_random_params(num_trials: int) -> list:
    """Generate parameter sets for random search."""
    params_list = []
    for _ in range(num_trials):
        params = ParameterSet(
            Kp_vel_xy=np.random.uniform(1.5, 6.0),
            Kp_vel_z=np.random.uniform(1.0, 4.0),
            Ki_vel_xy=np.random.uniform(0.1, 1.0),
            Ki_vel_z=np.random.uniform(0.1, 0.6),
            Kp_att_rp=np.random.uniform(4.0, 15.0),
            Kp_att_y=np.random.uniform(2.0, 8.0),
            Kd_att_rp=np.random.uniform(1.0, 5.0),
            Kd_att_y=np.random.uniform(0.5, 2.5),
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

    # Create tuner
    tuner = ParallelTuner(num_envs=args_cli.num_envs, device="cuda:0")

    # Generate parameters
    if args_cli.search_mode == "grid":
        params_list = generate_grid_params()
        print(f"Grid search: {len(params_list)} combinations")
    else:
        params_list = generate_random_params(args_cli.num_trials)
        print(f"Random search: {len(params_list)} trials")

    # Batch info
    batch_size = args_cli.num_envs
    num_batches = (len(params_list) + batch_size - 1) // batch_size
    print(f"Testing in {num_batches} batches of {batch_size}")
    print("-" * 80)

    # Run tuning
    results = []
    best_score = float("inf")
    best_params = None
    best_metrics = None

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(params_list))
        batch = params_list[batch_start:batch_end]

        print(f"Batch [{batch_idx+1}/{num_batches}] Testing {len(batch)} configs...", end=" ", flush=True)

        metrics_list = tuner.evaluate_batch(batch)

        # Process results
        stable_count = sum(1 for m in metrics_list if m.is_stable)
        batch_best = min(m.score for m in metrics_list)

        for params, metrics in zip(batch, metrics_list):
            results.append({"params": asdict(params), "metrics": asdict(metrics)})
            if metrics.score < best_score:
                best_score = metrics.score
                best_params = params
                best_metrics = metrics

        print(f"Stable: {stable_count}/{len(batch)}, Best: {batch_best:.3f}, Overall: {best_score:.3f}")

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(params_list):.2f}s per trial)")
    print(f"Stable: {sum(1 for r in results if r['metrics']['is_stable'])}/{len(results)}")

    if best_params:
        print(f"\nBest configuration (score: {best_score:.4f}):")
        print(f"  Kp_vel: ({best_params.Kp_vel_xy:.2f}, {best_params.Kp_vel_xy:.2f}, {best_params.Kp_vel_z:.2f})")
        print(f"  Ki_vel: ({best_params.Ki_vel_xy:.2f}, {best_params.Ki_vel_xy:.2f}, {best_params.Ki_vel_z:.2f})")
        print(f"  Kp_att: ({best_params.Kp_att_rp:.2f}, {best_params.Kp_att_rp:.2f}, {best_params.Kp_att_y:.2f})")
        print(f"  Kd_att: ({best_params.Kd_att_rp:.2f}, {best_params.Kd_att_rp:.2f}, {best_params.Kd_att_y:.2f})")
        print(f"\n  Hover drift: {best_metrics.hover_drift_mean:.4f}m (max: {best_metrics.hover_drift_max:.4f}m)")
        print(f"  Velocity settling: {best_metrics.vel_settling_time:.3f}s")
        print(f"  Velocity overshoot: {best_metrics.vel_overshoot:.1f}%")
        print(f"  Steady-state error: {best_metrics.vel_ss_error:.4f}m/s")
        print(f"\n  Attitude recovery: {best_metrics.att_recovery_time:.3f}s (recovered: {best_metrics.att_recovered})")
        print(f"  Attitude max error: {best_metrics.att_max_error:.1f}deg")
        print(f"  Attitude final error: {best_metrics.att_final_error:.1f}deg")

    # Save results
    output_file = output_dir / f"tuning_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "search_mode": args_cli.search_mode,
                "num_envs": args_cli.num_envs,
                "num_trials": len(params_list),
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
            f.write(f"# Auto-tuned DroneController configuration\n")
            f.write(f"# Generated: {timestamp}\n")
            f.write(f"# Score: {best_score:.4f}\n\n")
            f.write("from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg\n")
            f.write("from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg\n")
            f.write("from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg\n\n")
            f.write("TUNED_CONTROLLER_CFG = DroneControllerCfg(\n")
            f.write("    velocity=VelocityControllerCfg(\n")
            f.write(f"        Kp_vel=({best_params.Kp_vel_xy:.4f}, {best_params.Kp_vel_xy:.4f}, {best_params.Kp_vel_z:.4f}),\n")
            f.write(f"        Ki_vel=({best_params.Ki_vel_xy:.4f}, {best_params.Ki_vel_xy:.4f}, {best_params.Ki_vel_z:.4f}),\n")
            f.write("    ),\n")
            f.write("    attitude=AttitudeControllerCfg(\n")
            f.write(f"        Kp_att=({best_params.Kp_att_rp:.4f}, {best_params.Kp_att_rp:.4f}, {best_params.Kp_att_y:.4f}),\n")
            f.write(f"        Kd_att=({best_params.Kd_att_rp:.4f}, {best_params.Kd_att_rp:.4f}, {best_params.Kd_att_y:.4f}),\n")
            f.write("    ),\n")
            f.write(")\n")
        print(f"Best config saved to: {config_file}")

    print("=" * 80)

    tuner.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
