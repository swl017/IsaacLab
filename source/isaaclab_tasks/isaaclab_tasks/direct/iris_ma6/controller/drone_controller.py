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
from .gimbal_rate_loop import GimbalRateLoop
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

        # Create mixer (shared between rate controller and motor dynamics).
        # Ticket 040: reads effective k_f/k_m so motor mode swap propagates here too.
        self._mixer = MixerMatrix(
            arm_length=cfg.motor.arm_length,
            k_f=cfg.motor.get_effective_k_f(),
            k_m=cfg.motor.get_effective_k_m(),
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

        # Rate loop (mas/035): models the SIYI hardware user-command path —
        # first-order lag and saturation on the policy's (yaw_rate, pitch_rate)
        # command BEFORE it enters the gimbal controller. Body-motion
        # compensation downstream remains instantaneous (LOS-stabilization
        # bypass invariant).
        self._gimbal_rate_loop = GimbalRateLoop(
            cfg=cfg.gimbal_rate_loop,
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

        # Snapshot nominal gains at construction time so per-env randomization
        # always references the configured values, not whatever the env may
        # have written into the per-env tensors via curriculum gating later.
        # See iris_ma6 dynamics-curriculum bug 4.
        self._nominal_gains = self._snapshot_nominal_gains()
        # Promote per-env gain tensors to (num_envs, ...) shape so randomize_gains
        # can write to per-env rows without reshaping the controllers each call.
        # Clone on hand-off: the sub-controllers' setters store references, and
        # in-place randomization on the live tensors must not alias the
        # never-mutated _nominal_gains snapshot.
        self._velocity.set_gains(
            Kp_vel=self._nominal_gains["Kp_vel"].clone(),
            Ki_vel=self._nominal_gains["Ki_vel"].clone(),
        )
        self._attitude.set_gains(Kp_att=self._nominal_gains["Kp_att"].clone())
        self._rate.set_gains(
            Kp_rate=self._nominal_gains["Kp_rate"].clone(),
            Ki_rate=self._nominal_gains["Ki_rate"].clone(),
            Kd_rate=self._nominal_gains["Kd_rate"].clone(),
        )
        self._motor.set_tau_motor(self._nominal_gains["tau_motor"].clone())
        self._zoom.set_tau_zoom(self._nominal_gains["tau_zoom"].clone())
        self._zoom.set_max_zoom_rate(self._nominal_gains["max_zoom_rate"].clone())

    def _snapshot_nominal_gains(self) -> dict[str, torch.Tensor]:
        """Capture per-env nominal gains from the freshly-constructed sub-controllers.

        All vector gains are expanded to (num_envs, 3) and all scalar/per-env
        gains are expanded to (num_envs,). The resulting tensors are owned by
        ``self._nominal_gains`` and never mutated; randomize_gains samples
        scaled copies into the live controller buffers.
        """
        def _expand_vec(g: torch.Tensor) -> torch.Tensor:
            t = g.detach().clone() if isinstance(g, torch.Tensor) else torch.tensor(g, device=self.device)
            t = t.to(self.device)
            if t.dim() == 1:
                t = t.unsqueeze(0).expand(self.num_envs, -1).contiguous()
            return t

        def _expand_scalar(g) -> torch.Tensor:
            if isinstance(g, torch.Tensor):
                t = g.detach().clone().to(self.device)
            else:
                t = torch.tensor(g, dtype=torch.float32, device=self.device)
            if t.dim() == 0:
                t = t.expand(self.num_envs).contiguous()
            return t

        return {
            "Kp_vel": _expand_vec(self._velocity._Kp_vel),
            "Ki_vel": _expand_vec(self._velocity._Ki_vel),
            "Kp_att": _expand_vec(self._attitude._Kp_att),
            "Kp_rate": _expand_vec(self._rate._Kp_rate),
            "Ki_rate": _expand_vec(self._rate._Ki_rate),
            "Kd_rate": _expand_vec(self._rate._Kd_rate),
            "tau_motor": _expand_scalar(self._motor._tau_motor),
            "tau_zoom": _expand_scalar(self._zoom._tau_zoom),
            "max_zoom_rate": _expand_scalar(self._zoom._max_zoom_rate),
            # mas/037: lens slew clip nominal (siyi_a8 mode). Snapshotted for
            # forward-compat; v_max DR is intentionally disabled (design 6,
            # Tip 6) — see ZoomControllerCfg.randomize_v_max docstring.
            "v_max_levels_per_s": _expand_scalar(self._zoom._v_max),
        }

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
    def gimbal_rate_loop(self) -> GimbalRateLoop:
        """Get gimbal rate-loop component (mas/035)."""
        return self._gimbal_rate_loop

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
        self._last_q_des = q_des  # Stored for external oscillation analysis

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

        # mas/035: route the policy's normalized rate commands through the
        # SIYI rate loop (saturation + first-order lag) before they reach the
        # gimbal controller. The controller's signature is unchanged — the
        # rate loop preserves the normalized-[-1, 1] convention by
        # converting to rad/s, applying dynamics, and converting back.
        max_rate = self.cfg.gimbal.max_gimbal_rate
        omega_cmd_radps = torch.stack(
            [gimbal_yaw_rate_cmd * max_rate, gimbal_pitch_rate_cmd * max_rate],
            dim=-1,
        )
        omega_actual_radps = self._gimbal_rate_loop.step(omega_cmd_radps, _physics_dt)
        gimbal_yaw_rate_cmd_filtered = omega_actual_radps[:, 0] / max_rate
        gimbal_pitch_rate_cmd_filtered = omega_actual_radps[:, 1] / max_rate

        # Gimbal controller — uses physics_dt for correct rate integration
        # since it's called every physics step, not every policy step.
        gimbal_pos, gimbal_vel = self._gimbal.compute_control(
            gimbal_yaw_rate_cmd=gimbal_yaw_rate_cmd_filtered,
            gimbal_pitch_rate_cmd=gimbal_pitch_rate_cmd_filtered,
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

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        gimbal_az_initial: torch.Tensor | None = None,
        gimbal_el_initial: torch.Tensor | None = None,
    ):
        """Reset all controller states.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
            gimbal_az_initial: Optional world-frame azimuth to seed the
                gimbal controller from at reset (mas/035 smooth-reset
                semantics). When None, the controller falls back to its
                configured `initial_yaw`.
            gimbal_el_initial: Optional world-frame elevation to seed the
                gimbal controller from. Pairs with `gimbal_az_initial`.
        """
        self._motor.reset_to_hover(self.mass, self.gravity, env_ids)
        self._rate.reset(env_ids)
        self._attitude.reset(env_ids)
        self._velocity.reset(env_ids)
        self._gimbal.reset(env_ids, az_initial=gimbal_az_initial, el_initial=gimbal_el_initial)
        self._gimbal_rate_loop.reset(env_ids)
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
    ):
        """Randomize controller gains for specified environments.

        Applies per-env uniform scaling to nominal gains. The scaling range
        is curriculum-ramped: at progress=0 all gains are nominal, at
        progress=1 gains are sampled from cfg.scale_range.

        ``self._nominal_gains`` is captured at controller construction time
        and never mutated, so this call always references the configured
        values regardless of what env-side curriculum logic has written into
        the per-env gain tensors.

        Args:
            env_ids: (M,) environment indices to randomize.
            progress: Curriculum progress [0, 1].
            cfg: Gain randomization configuration.
        """
        if not cfg.enabled or progress <= 0.0:
            return

        # Curriculum-ramped range: at progress=0 -> (1,1), at progress=1 -> scale_range
        low = 1.0 - progress * (1.0 - cfg.scale_range[0])
        high = 1.0 + progress * (cfg.scale_range[1] - 1.0)
        M = len(env_ids)

        # In-place updates: only modify the specified env_ids rows.
        # This is safe for batched controllers where different agent slices
        # are randomized in separate calls.
        if cfg.randomize_velocity:
            scale = torch.empty(M, 1, device=self.device).uniform_(low, high)
            self._velocity._Kp_vel[env_ids] = self._nominal_gains["Kp_vel"][env_ids] * scale
            self._velocity._Ki_vel[env_ids] = self._nominal_gains["Ki_vel"][env_ids] * scale

        if cfg.randomize_attitude:
            scale = torch.empty(M, 1, device=self.device).uniform_(low, high)
            self._attitude._Kp_att[env_ids] = self._nominal_gains["Kp_att"][env_ids] * scale

        if cfg.randomize_rate:
            scale = torch.empty(M, 1, device=self.device).uniform_(low, high)
            self._rate._Kp_rate[env_ids] = self._nominal_gains["Kp_rate"][env_ids] * scale
            self._rate._Ki_rate[env_ids] = self._nominal_gains["Ki_rate"][env_ids] * scale
            self._rate._Kd_rate[env_ids] = self._nominal_gains["Kd_rate"][env_ids] * scale

        if cfg.randomize_motor:
            scale = torch.empty(M, device=self.device).uniform_(low, high)
            self._motor._tau_motor[env_ids] = self._nominal_gains["tau_motor"][env_ids] * scale

        if cfg.randomize_zoom:
            # Zoom uses its own scale range. The base is the *live* per-env
            # ``_zoom._tau_zoom`` which the env writes at every reset as
            #     max(cfg.zoom.tau_zoom * progress_dynamics, 1e-4)
            # so bootstrap retains near-instant zoom and the random scaling
            # composes on top of the curriculum value (in-place).
            # ``_nominal_gains["tau_zoom"]`` is intentionally NOT used here:
            # it equals cfg.zoom.tau_zoom from construction (Bug 4 fix) and
            # ignoring it for zoom is what lets bootstrap stay instant.
            z_low = 1.0 - progress * (1.0 - cfg.zoom_scale_range[0])
            z_high = 1.0 + progress * (cfg.zoom_scale_range[1] - 1.0)

            scale = torch.empty(M, device=self.device).uniform_(z_low, z_high)
            self._zoom._tau_zoom[env_ids] = self._zoom._tau_zoom[env_ids] * scale

            scale2 = torch.empty(M, device=self.device).uniform_(low, high)
            self._zoom._max_zoom_rate[env_ids] = self._nominal_gains["max_zoom_rate"][env_ids] * scale2

        # ------------------------------------------------------------------
        # Ticket 040 — Pegasus-mode physics DR (k_f, k_m, drag_coefs).
        # No-ops when motor.model == "default" OR randomize_physics_pegasus is
        # False; each knob writes per-env multiplicative scales that the
        # MotorDynamics / AerodynamicEffects compute paths apply at force time.
        # All three scale ranges are curriculum-ramped through the same
        # progress factor as the rest of randomize_gains.
        # ------------------------------------------------------------------
        is_pegasus_mode = self.cfg.motor.model == "pegasus"
        if is_pegasus_mode and getattr(cfg, "randomize_physics_pegasus", False):
            # k_f scale — per-env uniform.
            kf_low = 1.0 - progress * (1.0 - cfg.k_f_scale_range[0])
            kf_high = 1.0 + progress * (cfg.k_f_scale_range[1] - 1.0)
            kf_scale = torch.empty(M, device=self.device).uniform_(kf_low, kf_high)
            self._motor._k_f_scale[env_ids] = kf_scale

            # k_m scale — per-env uniform, INDEPENDENT of k_f scale (yaw
            # authority drifts independently of thrust authority across builds).
            km_low = 1.0 - progress * (1.0 - cfg.k_m_scale_range[0])
            km_high = 1.0 + progress * (cfg.k_m_scale_range[1] - 1.0)
            km_scale = torch.empty(M, device=self.device).uniform_(km_low, km_high)
            self._motor._k_m_scale[env_ids] = km_scale

            # drag_coefs scale — one scalar per env, applied uniformly to all
            # 3 body axes. Multiplies the constructed cfg coefs in-place.
            d_low = 1.0 - progress * (1.0 - cfg.drag_coefs_scale_range[0])
            d_high = 1.0 + progress * (cfg.drag_coefs_scale_range[1] - 1.0)
            d_scale = torch.empty(M, 1, device=self.device).uniform_(d_low, d_high)
            base_coefs = torch.tensor(
                self.cfg.aerodynamics.drag_coefs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, 3)
            self._aerodynamics._drag_coefs[env_ids] = base_coefs * d_scale

    def set_aerodynamic_level(self, level: int):
        """Set aerodynamic fidelity level.

        Args:
            level: Fidelity level (0-3).
        """
        self._aerodynamics.set_fidelity_level(level)

    # ------------------------------------------------------------------
    # Ticket 037 — Privileged-obs accessors (read-only).
    # Each returns the per-row gain SCALE (current / nominal) at native
    # batched-controller layout shape (N*A,) where the env's agent-major
    # convention is `agent_a → rows [a*N, (a+1)*N)`. The env-side assembly
    # reshapes to (N, A) via `.reshape(num_agents, num_envs).t()`.
    #
    # The 5 controller-block scales (vel, att, rate, motor, zoom_tau,
    # zoom_max_rate — 6 total) are the upstream independent RNG draws of
    # `randomize_gains`. We derive them at read time (current / nominal)
    # rather than caching the scales as new state, since `_nominal_gains`
    # is captured at construction and never mutated. For multi-axis gains
    # (Kp_vel, Kp_att, Kp_rate are 3-vectors), the scale is uniform across
    # axes per the broadcasting at randomize_gains() — we reduce by mean
    # to recover the scalar.
    #
    # See doc/critic_obs_design.md §3.4 for field semantics.
    # ------------------------------------------------------------------

    def get_vel_gain_scale(self) -> torch.Tensor:
        """Current velocity-controller gain scale per row. Shape (N*A,)."""
        return (self._velocity._Kp_vel / self._nominal_gains["Kp_vel"]).mean(dim=-1)

    def get_att_gain_scale(self) -> torch.Tensor:
        """Current attitude-controller gain scale per row. Shape (N*A,)."""
        return (self._attitude._Kp_att / self._nominal_gains["Kp_att"]).mean(dim=-1)

    def get_rate_gain_scale(self) -> torch.Tensor:
        """Current rate-controller gain scale per row. Shape (N*A,)."""
        return (self._rate._Kp_rate / self._nominal_gains["Kp_rate"]).mean(dim=-1)

    def get_motor_gain_scale(self) -> torch.Tensor:
        """Current motor τ scale per row. Shape (N*A,)."""
        return self._motor._tau_motor / self._nominal_gains["tau_motor"]

    def get_zoom_tau_scale(self) -> torch.Tensor:
        """Current zoom τ scale per row. Shape (N*A,)."""
        return self._zoom._tau_zoom / self._nominal_gains["tau_zoom"]

    def get_zoom_max_rate_scale(self) -> torch.Tensor:
        """Current zoom max-rate scale per row. Shape (N*A,)."""
        return self._zoom._max_zoom_rate / self._nominal_gains["max_zoom_rate"]
