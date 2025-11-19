import torch
from isaaclab.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    quat_inv,
    axis_angle_from_quat,
    )

class PointMass:
    def __init__(self, mass: float = 1.0, weight: float | None = None, num_envs: int = 1, disable_gravity: bool = True, device: str = 'cuda:0'):
        self.mass = mass
        self.weight = weight if weight is not None else mass * 9.81
        self.weight_tensor = torch.tensor([0, 0, -self.weight], dtype=torch.float32, device=device).view(1, 3).expand(num_envs, 3) if num_envs > 1 else torch.tensor([0, 0, -self.weight], dtype=torch.float32, device=device).view(1, 3)
        # self.weight_tensor = torch.tensor(self.weight, dtype=torch.float32).view(1, 1).expand(num_envs, 1) if num_envs > 1 else torch.tensor(self.weight, dtype=torch.float32).view(1, 1)
        self.num_envs = num_envs
        self.disable_gravity = disable_gravity

        self.device = device

        # Attitude control gains (proportional and derivative)
        self.default_gains = {
            # Position control
            "kp_x": torch.tensor([3.0], device=device).expand(num_envs),
            "ki_x": torch.tensor([0.5], device=device).expand(num_envs),  # Integral gain
            "kd_x": torch.tensor([0.1], device=device).expand(num_envs),
            "kp_y": torch.tensor([3.0], device=device).expand(num_envs),
            "kd_y": torch.tensor([0.1], device=device).expand(num_envs),
            "kp_z": torch.tensor([2.0], device=device).expand(num_envs),
            # Attitude control - increased for better stability
            "kp_att": torch.tensor([2.0], device=device).expand(num_envs),  # Proportional gain for roll/pitch
            "kd_att": torch.tensor([2.2], device=device).expand(num_envs),  # Derivative gain for roll/pitch
            "kp_yaw": torch.tensor([1.2], device=device).expand(num_envs),  # Yaw proportional gain (for rate tracking)
            "kd_yaw": torch.tensor([0.5], device=device).expand(num_envs),  # Yaw derivative gain (deprecated for rate tracking)
        }

        # Maximum moment limits (Nm) - prevent excessive torques and numerical explosion
        # Balanced limits: strong enough for control, prevent simulation instability
        self.max_moment = torch.tensor([5.0, 5.0, 2.0], device=device).view(1, 3).expand(num_envs, 3)

        # Integral state for velocity tracking (with anti-windup)
        self._vel_error_integral = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self._integral_max = 5.0  # Maximum integral accumulation (N·s)

    def wrap_to_pi(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Wraps the angle to the range [-pi, pi].
        """
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def compute_control(
        self,
        cmd_lin_vel_w: torch.Tensor,  # Command linear velocity in world frame
        cmd_yaw_vel: torch.Tensor,    # Command yaw velocity (scalar)
        curr_quat_w: torch.Tensor,    # Current orientation in world frame (w,x,y,z)
        curr_lin_vel_w: torch.Tensor, # Current linear velocity in world frame
        curr_ang_vel_b: torch.Tensor, # Current angular velocity in body frame
        curr_lin_acc_b: torch.Tensor, # Current linear acceleration in body frame
        dt: float = 0.02,             # Time step
        gains: dict = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the control commands for the point mass using quaternion-based attitude control.

        This implementation avoids gimbal lock by using quaternion error representation
        and applies proper PD control in the body frame.

        Args:
            cmd_lin_vel_w: Commanded linear velocity in world frame (N, 3)
            cmd_yaw_vel: Commanded yaw angular velocity (N,) or (N, 1)
            curr_quat_w: Current orientation quaternion (w, x, y, z) (N, 4)
            curr_lin_vel_w: Current linear velocity in world frame (N, 3)
            curr_ang_vel_b: Current angular velocity in body frame (N, 3)
            dt: Time step for integration
            gains: Optional dictionary of control gains

        Returns:
            force: Control force in body frame (N, 3)
            moment: Control moment in body frame (N, 3)
        """
        if gains is None:
            gains = self.default_gains

        # ========== ATTITUDE CONTROL (Quaternion-based) ==========
        # Desired orientation: level attitude (roll=0, pitch=0) with current yaw
        # This keeps the drone stable and prevents flipping

        # Extract current yaw from quaternion (avoid full Euler conversion)
        curr_yaw = self._extract_yaw_from_quat(curr_quat_w)

        # Desired quaternion: zero roll and pitch, current yaw
        # This is the key - we want the drone to stay level
        desired_quat_w = quat_from_euler_xyz(
            torch.zeros_like(curr_yaw),  # roll = 0
            torch.zeros_like(curr_yaw),  # pitch = 0
            curr_yaw                      # maintain current yaw
        )

        # Compute quaternion error: q_error = q_desired * q_current^{-1}
        # This gives us the rotation needed to go from current to desired
        quat_error = quat_mul(desired_quat_w, quat_inv(curr_quat_w))

        # Convert quaternion error to axis-angle representation
        # This gives us a 3D error vector in the body frame
        axis_angle_error = axis_angle_from_quat(quat_error)

        # For small angles, axis-angle ≈ attitude error
        # Extract roll and pitch errors (in body frame)
        att_error_b = self.wrap_to_pi(axis_angle_error[:, :3])  # (N, 3) - [roll_err, pitch_err, yaw_err]

        # Yaw control: track commanded yaw rate
        # We only control yaw velocity, not position
        yaw_error = cmd_yaw_vel.view(-1) - curr_ang_vel_b[:, 2]

        # PD Control for attitude stabilization
        # P term: proportional to attitude error
        # D term: damping based on angular velocity
        moment_roll_pitch = (
            -gains["kp_att"].view(-1, 1) * att_error_b[:, :2]  # Proportional term
            - gains["kd_att"].view(-1, 1) * curr_ang_vel_b[:, :2]  # Derivative term (damping)
        )

        # PD Control for yaw
        moment_yaw = (
            gains["kp_yaw"].view(-1) * yaw_error  # Proportional term
            - gains["kd_yaw"].view(-1) * curr_ang_vel_b[:, 2]  # Derivative term (damping)
        )

        # Combine moments
        moment = torch.cat([
            moment_roll_pitch,
            moment_yaw.view(-1, 1)
        ], dim=1)

        # Saturate moments to prevent excessive control
        # moment = torch.clamp(moment, -self.max_moment, self.max_moment)

        # ========== TRANSLATIONAL CONTROL ==========
        # Velocity tracking control with gravity compensation

        # Velocity error in world frame
        vel_error_w = cmd_lin_vel_w - curr_lin_vel_w

        # Transform to body frame for control
        vel_error_b = quat_rotate_inverse(curr_quat_w, vel_error_w)

        # Proportional control on velocity error
        force = vel_error_b * gains["kp_x"].view(-1, 1) - curr_lin_acc_b * gains["kd_x"].view(-1, 1)  # Using kp_x for all axes for simplicity

        # Add gravity compensation in body frame
        # Transform gravity vector to body frame
        if not self.disable_gravity:
            gravity_b = quat_rotate_inverse(curr_quat_w, self.weight_tensor)
            force = force - gravity_b

        # Optional: clamp upward thrust to prevent excessive acceleration
        force[:, 2] = torch.clamp(force[:, 2], min=-self.weight * 1.0, max=self.weight * 3.0)

        return force, moment

    def compute_control_quat(
        self,
        cmd_lin_vel_w: torch.Tensor,  # Command linear velocity in world frame
        cmd_yaw_vel: torch.Tensor,    # Command yaw velocity (scalar)
        curr_quat_w: torch.Tensor,    # Current orientation in world frame (w,x,y,z)
        curr_lin_vel_w: torch.Tensor, # Current linear velocity in world frame
        curr_ang_vel_b: torch.Tensor, # Current angular velocity in body frame
        curr_lin_acc_b: torch.Tensor, # Current linear acceleration in body frame
        dt: float = 0.02,             # Time step
        gains: dict = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute control commands using fully quaternion-based attitude control.

        This implementation uses the vector part of the quaternion error directly for control,
        avoiding conversions to Euler angles or axis-angle representation. This is more
        computationally efficient and mathematically pure.

        Mathematical basis:
        - For a unit quaternion q_error = [w, x, y, z] = [cos(θ/2), sin(θ/2) * axis]
        - For small angles θ: [x, y, z] ≈ (θ/2) * axis
        - Therefore, 2 * [x, y, z] represents the rotation error vector
        - This gives us a direct mapping from quaternion error to control torque

        Args:
            cmd_lin_vel_w: Commanded linear velocity in world frame (N, 3)
            cmd_yaw_vel: Commanded yaw angular velocity (N,) or (N, 1)
            curr_quat_w: Current orientation quaternion (w, x, y, z) (N, 4)
            curr_lin_vel_w: Current linear velocity in world frame (N, 3)
            curr_ang_vel_b: Current angular velocity in body frame (N, 3)
            curr_lin_acc_b: Current linear acceleration in body frame (N, 3)
            dt: Time step for integration
            gains: Optional dictionary of control gains

        Returns:
            force: Control force in body frame (N, 3)
            moment: Control moment in body frame (N, 3)
        """
        if gains is None:
            gains = self.default_gains

        # ========== ATTITUDE CONTROL (Pure Quaternion-based) ==========
        # Desired orientation: level attitude (roll=0, pitch=0) with current yaw

        # Extract current yaw to maintain it
        curr_yaw = self._extract_yaw_from_quat(curr_quat_w)

        # Desired quaternion: zero roll and pitch, current yaw
        desired_quat_w = quat_from_euler_xyz(
            torch.zeros_like(curr_yaw),  # roll = 0
            torch.zeros_like(curr_yaw),  # pitch = 0
            curr_yaw                      # maintain current yaw
        )

        # Compute quaternion error: q_error = q_desired * q_current^{-1}
        quat_error = quat_mul(desired_quat_w, quat_inv(curr_quat_w))

        # Ensure shortest path rotation (quaternion double cover)
        # If w < 0, negate the quaternion (q and -q represent the same rotation,
        # but we want the one with positive w for the shortest path)
        sign = torch.sign(quat_error[:, 0:1])  # (N, 1)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # Handle w=0 case
        quat_error = quat_error * sign

        # Extract attitude error directly from quaternion vector part
        # For small angles: error_vector ≈ 2 * [x, y, z]
        # This is the key advantage - no conversion needed!
        att_error_b = 2.0 * quat_error[:, 1:4]  # (N, 3) - [x, y, z] components

        # Separate roll/pitch control from yaw control
        roll_pitch_error = att_error_b[:, :2]

        # Yaw control: track commanded yaw rate
        yaw_error = cmd_yaw_vel.view(-1) - curr_ang_vel_b[:, 2]

        # PD Control for roll and pitch stabilization
        # P term: proportional to attitude error
        # D term: damping based on angular velocity
        moment_roll_pitch = (
            -gains["kp_att"].view(-1, 1) * roll_pitch_error  # Proportional term
            - gains["kd_att"].view(-1, 1) * curr_ang_vel_b[:, :2]  # Derivative term (damping)
        )

        # PD Control for yaw
        moment_yaw = (
            gains["kp_yaw"].view(-1) * yaw_error  # Proportional term
            - gains["kd_yaw"].view(-1) * curr_ang_vel_b[:, 2]  # Derivative term (damping)
        )

        # Combine moments
        moment = torch.cat([
            moment_roll_pitch,
            moment_yaw.view(-1, 1)
        ], dim=1)

        # Saturate moments to prevent excessive control
        # moment = torch.clamp(moment, -self.max_moment, self.max_moment)

        # ========== TRANSLATIONAL CONTROL ==========
        # Velocity tracking control with gravity compensation
        # (Same as original implementation)

        # Velocity error in world frame
        vel_error_w = cmd_lin_vel_w - curr_lin_vel_w

        # Transform to body frame for control
        vel_error_b = quat_rotate_inverse(curr_quat_w, vel_error_w)

        # PD control on velocity error
        force = vel_error_b * gains["kp_x"].view(-1, 1) - curr_lin_acc_b * gains["kd_x"].view(-1, 1)

        # Add gravity compensation in body frame
        if not self.disable_gravity:
            gravity_b = quat_rotate_inverse(curr_quat_w, self.weight_tensor)
            force = force - gravity_b

        # Clamp upward thrust to prevent excessive acceleration
        force[:, 2] = torch.clamp(force[:, 2], min=-self.weight * 1.0, max=self.weight * 3.0)

        return force, moment

    def compute_control_quat_exact(
        self,
        cmd_lin_vel_w: torch.Tensor,  # Command linear velocity in world frame
        cmd_yaw_vel_w: torch.Tensor,    # Command yaw velocity (scalar)
        curr_quat_w: torch.Tensor,    # Current orientation in world frame (w,x,y,z)
        curr_lin_vel_w: torch.Tensor, # Current linear velocity in world frame
        curr_ang_vel_b: torch.Tensor, # Current angular velocity in body frame
        curr_lin_acc_b: torch.Tensor, # Current linear acceleration in body frame
        dt: float = 0.02,             # Time step
        gains: dict = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute control commands using exact quaternion-based attitude control.

        This implementation uses the exact quaternion logarithm map without small angle
        approximations. It is valid for arbitrary rotation angles and provides accurate
        attitude error representation across the full SO(3) space.

        Mathematical basis:
        - For a unit quaternion q = [w, x, y, z] = [cos(θ/2), sin(θ/2) * axis]
        - The exact rotation vector (axis * angle) is given by the quaternion logarithm:
          log(q) = [0, (θ/2) * axis / sin(θ/2) * 2] = [0, θ * axis / sin(θ/2)]
        - Where θ = 2 * acos(w) is the rotation angle
        - For θ → 0: lim (θ / sin(θ/2)) = 2 (small angle case handled via Taylor series)

        This formulation:
        - Works for any rotation angle (0 to π)
        - Avoids gimbal lock
        - Provides exact error representation
        - Handles singularities numerically

        Args:
            cmd_lin_vel_w: Commanded linear velocity in world frame (N, 3)
            cmd_yaw_vel_w: Commanded yaw angular velocity in world frame (N,) or (N, 1)
            curr_quat_w: Current orientation quaternion (w, x, y, z) (N, 4)
            curr_lin_vel_w: Current linear velocity in world frame (N, 3)
            curr_ang_vel_b: Current angular velocity in body frame (N, 3)
            curr_lin_acc_b: Current linear acceleration in body frame (N, 3)
            dt: Time step for integration
            gains: Optional dictionary of control gains

        Returns:
            force: Control force in body frame (N, 3)
            moment: Control moment in body frame (N, 3)
        """
        if gains is None:
            gains = self.default_gains

        # ========== ATTITUDE CONTROL (Exact Quaternion-based) ==========
        # HYBRID APPROACH:
        # - Roll/Pitch: Position control to maintain level (zero roll/pitch)
        # - Yaw: Rate tracking control (track commanded yaw rate directly)
        #
        # This avoids:
        # 1. Roll-yaw coupling from Euler angle extraction
        # 2. Yaw integration instability from position control
        #
        # Strategy: Use ZERO yaw reference to avoid coupling, control yaw independently via rate

        # Desired quaternion: 180° roll to match USD body frame orientation
        # The IRIS USD has a flipped body frame, so "upright" = 180° roll, not 0°
        import math
        desired_quat_w = quat_from_euler_xyz(
            torch.full((curr_quat_w.shape[0],), math.pi, device=curr_quat_w.device),  # roll = 180°
            torch.zeros(curr_quat_w.shape[0], device=curr_quat_w.device),  # pitch = 0
            torch.zeros(curr_quat_w.shape[0], device=curr_quat_w.device)   # yaw = 0 (world aligned)
        )

        # Compute quaternion error: q_error = q_desired * q_current^{-1}
        quat_error = quat_mul(desired_quat_w, quat_inv(curr_quat_w))

        # Ensure shortest path rotation (quaternion double cover)
        # If w < 0, negate the quaternion for shortest path
        sign = torch.sign(quat_error[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        quat_error = quat_error * sign

        # Extract quaternion components
        w_part = quat_error[:, 0]      # Scalar part (N,)
        vec_part = quat_error[:, 1:4]  # Vector part [x, y, z] (N, 3)

        # Compute exact attitude error using quaternion logarithm
        # error_vector = 2 * θ * axis where θ = acos(w), axis = vec / ||vec||
        # Combining: error_vector = 2 * acos(w) * vec / ||vec||

        # Compute the norm of the vector part
        vec_norm = torch.norm(vec_part, dim=1, keepdim=True)  # (N, 1)

        # Compute half angle: θ/2 = acos(w), clamped for numerical stability
        half_theta = torch.acos(torch.clamp(w_part, -1.0, 1.0))  # (N,)

        # Compute the scaling factor: θ / sin(θ/2)
        # For small angles (θ < threshold), use Taylor series: θ/sin(θ/2) ≈ 2
        # For larger angles, compute exact value
        threshold = 1e-4
        vec_norm_squeezed = vec_norm.squeeze(-1)  # (N,)

        # sin(θ/2) = ||vec|| for unit quaternions
        # Factor = 2 * θ/2 / sin(θ/2) = θ / ||vec||
        # Safe division with threshold for small angles
        safe_vec_norm = torch.where(
            vec_norm_squeezed < threshold,
            torch.ones_like(vec_norm_squeezed),
            vec_norm_squeezed
        )

        factor = half_theta / safe_vec_norm

        # For very small angles, use the limit: lim_{θ→0} θ/sin(θ/2) = 2
        factor = torch.where(
            vec_norm_squeezed < threshold,
            torch.ones_like(factor),  # Limit value is 1 (since we already have 2* below)
            factor
        )

        # Compute exact attitude error vector
        att_error_w = 2.0 * vec_part * factor.unsqueeze(-1)  # (N, 3)

        att_error_b = quat_rotate_inverse(curr_quat_w, att_error_w)  # Transform to body frame

        # ROLL/PITCH PD CONTROL
        # moment = -kp * error - kd * velocity
        moment = \
            -gains["kp_att"].view(-1, 1) * att_error_b \
            - gains["kd_att"].view(-1, 1) * curr_ang_vel_b

        # YAW RATE TRACKING (overwrite z-axis moment)
        # Independent yaw rate control to avoid integration instability
        cmd_yaw_vel_b = quat_rotate_inverse(curr_quat_w, torch.stack([torch.zeros_like(cmd_yaw_vel_w), torch.zeros_like(cmd_yaw_vel_w), cmd_yaw_vel_w], dim=1))  # (N, 3)
        yaw_rate_error = cmd_yaw_vel_b[:, 2] - curr_ang_vel_b[:, 2]
        moment[:, 2] = gains["kp_yaw"].view(-1) * yaw_rate_error

        # Saturate moments to prevent excessive control and numerical explosion
        moment = torch.clamp(moment, -self.max_moment, self.max_moment)

        # ========== TRANSLATIONAL CONTROL ==========
        # PID velocity tracking control with anti-windup and gravity compensation

        # Velocity error in world frame
        vel_error_w = cmd_lin_vel_w - curr_lin_vel_w

        # Transform to body frame for control
        vel_error_b = quat_rotate_inverse(curr_quat_w, vel_error_w)

        # Update integral with anti-windup
        # Only integrate in world frame to avoid frame rotation issues
        self._vel_error_integral += vel_error_w * dt

        # Anti-windup: Clamp integral to prevent excessive buildup
        integral_clamped = torch.clamp(
            self._vel_error_integral,
            min=-self._integral_max,
            max=self._integral_max
        )
        self._vel_error_integral = integral_clamped

        # Transform integral to body frame
        integral_b = quat_rotate_inverse(curr_quat_w, integral_clamped)

        # PID control on velocity error
        # Force = Kp * error + Ki * integral + Kd * derivative
        # Note: derivative of velocity is acceleration (negative feedback on acceleration)
        force = (
            gains["kp_x"].view(-1, 1) * vel_error_b +
            gains["ki_x"].view(-1, 1) * integral_b -
            gains["kd_x"].view(-1, 1) * curr_lin_acc_b
        )

        # Add gravity compensation in body frame
        if not self.disable_gravity:
            gravity_b = quat_rotate_inverse(curr_quat_w, self.weight_tensor)
            force = force - gravity_b

        # Clamp upward thrust to prevent excessive acceleration
        # Also implement back-calculation anti-windup: reset integral if saturated
        force_before_clamp = force.clone()
        force[:, 2] = torch.clamp(force[:, 2], min=-self.weight * 1.0, max=self.weight * 3.0)

        # Back-calculation anti-windup: if force saturated, reduce integral
        saturated = (force_before_clamp[:, 2] != force[:, 2])
        if saturated.any():
            # Reduce integral for saturated environments
            self._vel_error_integral[saturated] *= 0.9

        return force, moment

    def reset_integral(self, env_ids: torch.Tensor = None):
        """
        Reset integral state for specified environments (or all if None).
        Call this when environments reset to prevent integral windup carryover.

        Args:
            env_ids: Environment indices to reset (None = reset all)
        """
        if env_ids is None:
            self._vel_error_integral.zero_()
        else:
            self._vel_error_integral[env_ids] = 0.0

    def _extract_yaw_from_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Extract yaw angle from quaternion without full Euler conversion.
        This is more numerically stable than full Euler angle extraction.

        Args:
            quat: Quaternion (w, x, y, z) (N, 4)

        Returns:
            yaw: Yaw angle in radians (N,)
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        return yaw
        
    def reset(self, env_ids=None):
        """Reset controller state for specified environments."""
        if hasattr(self, '_desired_yaw'):
            if env_ids is None:
                # Reset all
                delattr(self, '_desired_yaw')
            else:
                # Reset specific environments - will be reinitialized on next control call
                pass  # The yaw will be reinitialized from current quaternion automatically