# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone controller orchestrator that combines all control subsystems.

Control Architecture:
    Policy (25Hz) -> Velocity Controller -> Attitude Controller -> Motor Dynamics -> Physics
                  -> Gimbal Controller -> Joint Targets
                  -> Zoom Controller -> Zoom Level

Frame Convention:
- World: ENU (East-North-Up)
- Body: FLU (Forward-Left-Up)
- Quaternion: wxyz (scalar first)
"""

from __future__ import annotations

import torch

from .aerodynamics import AerodynamicEffects
from .attitude_controller import AttitudeController
from .controller_cfg import DroneControllerCfg
from .gimbal_controller import GimbalController
from .mixer import MixerMatrix
from .motor_dynamics import MotorDynamics
from .velocity_controller import VelocityController
from .zoom_controller import ZoomController


class DroneController:
    """Orchestrator for complete drone control system.

    Manages the cascaded control loop:
    1. Velocity controller: v_cmd -> attitude_cmd, thrust_cmd
    2. Attitude controller: attitude_cmd -> omega_cmd (rotor speeds)
    3. Motor dynamics: omega_cmd -> F_body, tau_body
    4. Aerodynamics: Additional forces/moments (optional)

    Plus parallel subsystems:
    - Gimbal controller: gimbal rate commands -> joint targets
    - Zoom controller: zoom rate commands -> zoom level
    """

    def __init__(
        self,
        cfg: DroneControllerCfg,
        mass: float,
        gravity: float,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize drone controller.

        Args:
            cfg: Drone controller configuration.
            mass: Drone mass [kg].
            gravity: Gravity magnitude [m/s^2].
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.mass = mass
        self.gravity = gravity
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Create mixer (shared between components)
        self._mixer = MixerMatrix(
            arm_length=cfg.motor.arm_length,
            k_f=cfg.motor.k_f,
            k_m=cfg.motor.k_m,
            device=self.device,
        )

        # Create sub-controllers
        self._motor = MotorDynamics(
            cfg=cfg.motor,
            num_envs=num_envs,
            device=self.device,
        )

        self._attitude = AttitudeController(
            cfg=cfg.attitude,
            mixer=self._mixer,
            num_envs=num_envs,
            device=self.device,
        )

        self._velocity = VelocityController(
            cfg=cfg.velocity,
            mass=mass,
            gravity=gravity,
            num_envs=num_envs,
            device=self.device,
        )

        self._gimbal = GimbalController(
            cfg=cfg.gimbal,
            num_envs=num_envs,
            device=self.device,
        )

        self._zoom = ZoomController(
            cfg=cfg.zoom,
            num_envs=num_envs,
            device=self.device,
        )

        self._aerodynamics = AerodynamicEffects(
            cfg=cfg.aerodynamics,
            num_envs=num_envs,
            device=self.device,
        )

    @property
    def motor_dynamics(self) -> MotorDynamics:
        """Get motor dynamics component."""
        return self._motor

    @property
    def attitude_controller(self) -> AttitudeController:
        """Get attitude controller component."""
        return self._attitude

    @property
    def velocity_controller(self) -> VelocityController:
        """Get velocity controller component."""
        return self._velocity

    @property
    def gimbal_controller(self) -> GimbalController:
        """Get gimbal controller component."""
        return self._gimbal

    @property
    def zoom_controller(self) -> ZoomController:
        """Get zoom controller component."""
        return self._zoom

    @property
    def aerodynamics(self) -> AerodynamicEffects:
        """Get aerodynamics component."""
        return self._aerodynamics

    def step_policy(
        self,
        v_cmd: torch.Tensor,
        yaw_rate_cmd: torch.Tensor,
        gimbal_yaw_rate_cmd: torch.Tensor,
        gimbal_pitch_rate_cmd: torch.Tensor,
        zoom_rate_cmd: torch.Tensor,
        q_body: torch.Tensor,
        v_body: torch.Tensor,
        omega_body: torch.Tensor,
        sim_dt: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Execute full control cascade for one policy step.

        Args:
            v_cmd: (N, 3) velocity command in world frame [m/s].
            yaw_rate_cmd: (N,) yaw rate command [rad/s].
            gimbal_yaw_rate_cmd: (N,) gimbal yaw rate command [-1, 1].
            gimbal_pitch_rate_cmd: (N,) gimbal pitch rate command [-1, 1].
            zoom_rate_cmd: (N,) zoom rate command [-1, 1].
            q_body: (N, 4) current body quaternion (wxyz).
            v_body: (N, 3) current body velocity in world frame [m/s].
            omega_body: (N, 3) current body angular velocity [rad/s].
            sim_dt: Simulation timestep [s].

        Returns:
            F_body: (N, 3) force to apply in body frame [N].
            tau_body: (N, 3) torque to apply in body frame [Nm].
            gimbal_targets: (yaw, roll, pitch) joint position targets [rad].
            zoom_level: (N,) current zoom level.
        """
        # Compute number of inner loop steps
        num_substeps = max(1, int(sim_dt / self.cfg.control_dt))
        inner_dt = sim_dt / num_substeps

        # Run velocity controller (outer loop, runs at policy rate)
        q_des, thrust_cmd, yaw_rate = self._velocity.compute_control(
            v_des=v_cmd,
            yaw_rate_des=yaw_rate_cmd,
            v_current=v_body,
            q_current=q_body,
            dt=sim_dt,
        )

        # Run inner loop at higher rate
        for _ in range(num_substeps):
            # Attitude controller
            omega_cmd, tau_cmd = self._attitude.compute_control(
                q_des=q_des,
                yaw_rate_des=yaw_rate,
                q_current=q_body,
                omega_current=omega_body,
                thrust_cmd=thrust_cmd,
            )

            # Motor dynamics
            self._motor.step(omega_cmd, inner_dt)

        # Get body wrench from motors
        F_motor, tau_motor = self._motor.compute_body_wrench()

        # Add aerodynamic effects
        F_aero, tau_aero = self._aerodynamics.compute_forces(
            v_body_world=v_body,
            omega_rotors=self._motor.omega,
            q_body=q_body,
            dt=sim_dt,
        )

        # Total wrench in body frame
        F_body = F_motor + F_aero
        tau_body = tau_motor + tau_aero

        # Gimbal controller
        gimbal_yaw, gimbal_roll, gimbal_pitch = self._gimbal.compute_control(
            gimbal_yaw_rate_cmd=gimbal_yaw_rate_cmd,
            gimbal_pitch_rate_cmd=gimbal_pitch_rate_cmd,
            q_body=q_body,
            dt=sim_dt,
        )

        # Zoom controller
        zoom_level = self._zoom.compute_control(
            zoom_rate_cmd=zoom_rate_cmd,
            dt=sim_dt,
        )

        # Return forces in body frame (Isaac Lab expects local frame)
        return F_body, tau_body, (gimbal_yaw, gimbal_roll, gimbal_pitch), zoom_level

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset all controller states.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        self._motor.reset(env_ids)
        self._attitude.reset(env_ids)
        self._velocity.reset(env_ids)
        self._gimbal.reset(env_ids)
        self._zoom.reset(env_ids)
        self._aerodynamics.reset(env_ids)

    def get_camera_direction_world(
        self,
        q_body: torch.Tensor,
    ) -> torch.Tensor:
        """Get camera pointing direction in world frame.

        Args:
            q_body: (N, 4) body quaternion (wxyz).

        Returns:
            direction: (N, 3) camera pointing direction in world frame.
        """
        return self._gimbal.get_camera_direction_world(q_body)

    def get_fov(self, base_fov: float = 90.0) -> torch.Tensor:
        """Get current field of view based on zoom level.

        Args:
            base_fov: Base FOV at 1x zoom [degrees].

        Returns:
            fov: (N,) current field of view [degrees].
        """
        return self._zoom.get_fov(base_fov)

    def set_aerodynamic_level(self, level: int):
        """Set aerodynamic fidelity level.

        Args:
            level: Fidelity level (0-3).
        """
        self._aerodynamics.set_fidelity_level(level)
