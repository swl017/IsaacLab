# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone controller orchestrator that combines all control subsystems.

Control Architecture (PX4-style cascaded loops):
    Policy (25Hz) -> Velocity Controller -> Attitude Controller -> Rate Controller -> Motor Dynamics -> Physics
                  -> Gimbal Controller -> Joint Targets
                  -> Zoom Controller -> Zoom Level

Cascaded Loop Hierarchy:
    1. Velocity Controller (outer): v_cmd -> q_des, thrust_cmd
    2. Attitude Controller (middle): q_des -> rate_setpoint (P-only)
    3. Rate Controller (inner): rate_setpoint -> tau_cmd -> motor_cmd (PID)

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
from .gain_randomization_cfg import GainRandomizationCfg
from .gimbal_controller import GimbalController
from .mixer import MixerMatrix
from .motor_dynamics import MotorDynamics
from .rate_controller import RateController
from .velocity_controller import VelocityController
from .zoom_controller import ZoomController


class DroneController:
    """Orchestrator for complete drone control system.

    Manages the cascaded control loop following PX4 architecture:
    1. Velocity controller: v_cmd -> attitude_cmd, thrust_cmd
    2. Attitude controller: attitude_cmd -> rate_setpoint (P-only)
    3. Rate controller: rate_setpoint -> omega_cmd (PID with mixer)
    4. Motor dynamics: omega_cmd -> F_body, tau_body
    5. Aerodynamics: Additional forces/moments (optional)

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

        # Create mixer (shared between rate controller and motor dynamics)
        self._mixer = MixerMatrix(
            arm_length=cfg.motor.arm_length,
            k_f=cfg.motor.k_f,
            k_m=cfg.motor.k_m,
            device=self.device,
        )

        # Create sub-controllers in order from inner to outer
        self._motor = MotorDynamics(
            cfg=cfg.motor,
            num_envs=num_envs,
            device=self.device,
        )

        # Rate controller (innermost loop - PID)
        self._rate = RateController(
            cfg=cfg.rate,
            mixer=self._mixer,
            num_envs=num_envs,
            device=self.device,
        )

        # Attitude controller (middle loop - P-only, outputs rate setpoint)
        self._attitude = AttitudeController(
            cfg=cfg.attitude,
            num_envs=num_envs,
            device=self.device,
        )

        # Velocity controller (outer loop)
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
    def rate_controller(self) -> RateController:
        """Get rate controller component."""
        return self._rate

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
        gimbal_joint_positions: torch.Tensor,
        physics_dt: float | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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
            sim_dt: Decimated policy timestep [s] (sim.dt * decimation).
            gimbal_joint_positions: (N, 3) actual gimbal joint positions [pitch, yaw, roll].
            physics_dt: Actual physics step [s] (sim.dt). If None, uses sim_dt.
                Used by gimbal/zoom for correct rate integration when called every physics step.

        Returns:
            F_body: (N, 3) force to apply in body frame [N].
            tau_body: (N, 3) torque to apply in body frame [Nm].
            gimbal_pos_targets: (yaw, roll, pitch) joint position targets [rad].
            gimbal_vel_targets: (yaw, roll, pitch) joint velocity feedforward [rad/s].
            zoom_level: (N,) current zoom level.
        """
        # Resolve physics dt (actual time between calls)
        _physics_dt = physics_dt if physics_dt is not None else sim_dt

        # Compute number of inner loop steps
        num_substeps = max(1, int(sim_dt / self.cfg.control_dt))
        inner_dt = sim_dt / num_substeps

        # =====================================================================
        # OUTER LOOP: Velocity Controller (runs at policy rate)
        # Converts velocity command to desired attitude + thrust
        # =====================================================================
        q_des, thrust_cmd, yaw_rate = self._velocity.compute_control(
            v_des=v_cmd,
            yaw_rate_des=yaw_rate_cmd,
            v_current=v_body,
            q_current=q_body,
            dt=sim_dt,
        )

        # =====================================================================
        # INNER LOOPS: Run at higher rate for stability
        # =====================================================================
        for _ in range(num_substeps):
            # -----------------------------------------------------------------
            # MIDDLE LOOP: Attitude Controller (P-only)
            # Converts attitude error to rate setpoint
            # -----------------------------------------------------------------
            rate_setpoint = self._attitude.compute_control(
                q_des=q_des,
                yaw_rate_des=yaw_rate,
                q_current=q_body,
            )

            # -----------------------------------------------------------------
            # INNER LOOP: Rate Controller (PID with mixer)
            # Converts rate error to motor commands
            # -----------------------------------------------------------------
            omega_cmd, tau_cmd, _ = self._rate.compute_control(
                rate_setpoint=rate_setpoint,
                omega_current=omega_body,
                thrust_cmd=thrust_cmd,
                dt=inner_dt,
            )
            # Note: saturation status returned but not yet used for velocity anti-windup

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

        # Gimbal controller — uses physics_dt for correct rate integration
        # since it's called every physics step, not every policy step.
        gimbal_pos, gimbal_vel = self._gimbal.compute_control(
            gimbal_yaw_rate_cmd=gimbal_yaw_rate_cmd,
            gimbal_pitch_rate_cmd=gimbal_pitch_rate_cmd,
            q_body=q_body,
            dt=_physics_dt,
            omega_body=omega_body,
            joint_positions_actual=gimbal_joint_positions,
        )
        gimbal_yaw, gimbal_roll, gimbal_pitch = gimbal_pos

        # Zoom controller — also uses physics_dt
        zoom_level = self._zoom.compute_control(
            zoom_rate_cmd=zoom_rate_cmd,
            dt=_physics_dt,
        )

        # Return forces in body frame (Isaac Lab expects local frame)
        return F_body, tau_body, (gimbal_yaw, gimbal_roll, gimbal_pitch), gimbal_vel, zoom_level

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset all controller states.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        self._motor.reset(env_ids)
        self._rate.reset(env_ids)
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

    def randomize_gains(
        self,
        env_ids: torch.Tensor,
        progress: float,
        cfg: GainRandomizationCfg,
        tau_zoom_nominal: float | None = None,
    ):
        """Randomize controller gains for specified environments.

        Applies per-env uniform scaling to nominal gains. The scaling range
        is curriculum-ramped: at progress=0 all gains are nominal, at
        progress=1 gains are sampled from cfg.scale_range.

        Args:
            env_ids: (M,) environment indices to randomize.
            progress: Curriculum progress [0, 1].
            cfg: Gain randomization configuration.
        """
        if not cfg.enabled or progress <= 0.0:
            return

        # Store nominal gains on first call
        if not hasattr(self, "_nominal_gains"):
            self._nominal_gains = {
                "Kp_vel": self._velocity._Kp_vel.clone(),
                "Ki_vel": self._velocity._Ki_vel.clone(),
                "Kp_att": self._attitude._Kp_att.clone(),
                "Kp_rate": self._rate._Kp_rate.clone(),
                "Ki_rate": self._rate._Ki_rate.clone(),
                "Kd_rate": self._rate._Kd_rate.clone(),
                "tau_motor": (
                    self._motor._tau_motor.clone()
                    if isinstance(self._motor._tau_motor, torch.Tensor)
                    else torch.tensor(self._motor._tau_motor, device=self.device)
                ),
                "tau_zoom": (
                    self._zoom._tau_zoom.clone()
                    if isinstance(self._zoom._tau_zoom, torch.Tensor)
                    else torch.tensor(self._zoom._tau_zoom, device=self.device)
                ),
                "max_zoom_rate": (
                    self._zoom._max_zoom_rate.clone()
                    if isinstance(self._zoom._max_zoom_rate, torch.Tensor)
                    else torch.tensor(self._zoom._max_zoom_rate, device=self.device)
                ),
            }
            # Expand all gains to (N, 3) for per-env storage
            for key in ["Kp_vel", "Ki_vel", "Kp_att", "Kp_rate", "Ki_rate", "Kd_rate"]:
                g = self._nominal_gains[key]
                if g.dim() == 1:
                    self._nominal_gains[key] = g.unsqueeze(0).expand(self.num_envs, -1).clone()
                    # Also set the controller to use (N, 3)
            # Expand scalar gains to (N,)
            tau = self._nominal_gains["tau_motor"]
            if tau.dim() == 0:
                self._nominal_gains["tau_motor"] = tau.expand(self.num_envs).clone()
            tau_z = self._nominal_gains["tau_zoom"]
            if tau_z.dim() == 0:
                self._nominal_gains["tau_zoom"] = tau_z.expand(self.num_envs).clone()
            mzr = self._nominal_gains["max_zoom_rate"]
            if mzr.dim() == 0:
                self._nominal_gains["max_zoom_rate"] = mzr.expand(self.num_envs).clone()

            # Initialize per-env gains from nominal
            self._velocity.set_gains(
                Kp_vel=self._nominal_gains["Kp_vel"],
                Ki_vel=self._nominal_gains["Ki_vel"],
            )
            self._attitude.set_gains(Kp_att=self._nominal_gains["Kp_att"])
            self._rate.set_gains(
                Kp_rate=self._nominal_gains["Kp_rate"],
                Ki_rate=self._nominal_gains["Ki_rate"],
                Kd_rate=self._nominal_gains["Kd_rate"],
            )
            self._motor.set_tau_motor(self._nominal_gains["tau_motor"])
            self._zoom.set_tau_zoom(self._nominal_gains["tau_zoom"])
            self._zoom.set_max_zoom_rate(self._nominal_gains["max_zoom_rate"])

        # Curriculum-ramped range: at progress=0 -> (1,1), at progress=1 -> scale_range
        low = 1.0 - progress * (1.0 - cfg.scale_range[0])
        high = 1.0 + progress * (cfg.scale_range[1] - 1.0)
        M = len(env_ids)

        if cfg.randomize_velocity:
            scale = torch.empty(M, 1, device=self.device).uniform_(low, high)
            new_Kp = self._nominal_gains["Kp_vel"].clone()
            new_Ki = self._nominal_gains["Ki_vel"].clone()
            new_Kp[env_ids] = self._nominal_gains["Kp_vel"][env_ids] * scale
            new_Ki[env_ids] = self._nominal_gains["Ki_vel"][env_ids] * scale
            self._velocity.set_gains(Kp_vel=new_Kp, Ki_vel=new_Ki)

        if cfg.randomize_attitude:
            scale = torch.empty(M, 1, device=self.device).uniform_(low, high)
            new_Kp = self._nominal_gains["Kp_att"].clone()
            new_Kp[env_ids] = self._nominal_gains["Kp_att"][env_ids] * scale
            self._attitude.set_gains(Kp_att=new_Kp)

        if cfg.randomize_rate:
            scale = torch.empty(M, 1, device=self.device).uniform_(low, high)
            new_Kp = self._nominal_gains["Kp_rate"].clone()
            new_Ki = self._nominal_gains["Ki_rate"].clone()
            new_Kd = self._nominal_gains["Kd_rate"].clone()
            new_Kp[env_ids] = self._nominal_gains["Kp_rate"][env_ids] * scale
            new_Ki[env_ids] = self._nominal_gains["Ki_rate"][env_ids] * scale
            new_Kd[env_ids] = self._nominal_gains["Kd_rate"][env_ids] * scale
            self._rate.set_gains(Kp_rate=new_Kp, Ki_rate=new_Ki, Kd_rate=new_Kd)

        if cfg.randomize_motor:
            scale = torch.empty(M, device=self.device).uniform_(low, high)
            new_tau = self._nominal_gains["tau_motor"].clone()
            new_tau[env_ids] = self._nominal_gains["tau_motor"][env_ids] * scale
            self._motor.set_tau_motor(new_tau)

        if cfg.randomize_zoom:
            # Update nominal if curriculum override provided
            if tau_zoom_nominal is not None:
                self._nominal_gains["tau_zoom"][env_ids] = tau_zoom_nominal

            # Zoom uses its own wider scale range to prevent catastrophic forgetting
            z_low = 1.0 - progress * (1.0 - cfg.zoom_scale_range[0])
            z_high = 1.0 + progress * (cfg.zoom_scale_range[1] - 1.0)

            scale = torch.empty(M, device=self.device).uniform_(z_low, z_high)
            new_tau_z = self._nominal_gains["tau_zoom"].clone()
            new_tau_z[env_ids] = self._nominal_gains["tau_zoom"][env_ids] * scale
            self._zoom.set_tau_zoom(new_tau_z)

            scale2 = torch.empty(M, device=self.device).uniform_(low, high)
            new_mzr = self._nominal_gains["max_zoom_rate"].clone()
            new_mzr[env_ids] = self._nominal_gains["max_zoom_rate"][env_ids] * scale2
            self._zoom.set_max_zoom_rate(new_mzr)

    def set_aerodynamic_level(self, level: int):
        """Set aerodynamic fidelity level.

        Args:
            level: Fidelity level (0-3).
        """
        self._aerodynamics.set_fidelity_level(level)
