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
            "kd_x": torch.tensor([0.1], device=device).expand(num_envs),
            "kp_y": torch.tensor([3.0], device=device).expand(num_envs),
            "kd_y": torch.tensor([0.1], device=device).expand(num_envs),
            "kp_z": torch.tensor([2.0], device=device).expand(num_envs),
            # Attitude control - increased for better stability
            "kp_att": torch.tensor([2.0], device=device).expand(num_envs),  # Proportional gain for roll/pitch
            "kd_att": torch.tensor([2.2], device=device).expand(num_envs),  # Derivative gain for roll/pitch
            "kp_yaw": torch.tensor([1.2], device=device).expand(num_envs),  # Yaw proportional gain
            "kd_yaw": torch.tensor([0.5], device=device).expand(num_envs),  # Yaw derivative gain
        }

        # Maximum moment limits (Nm) - prevent excessive torques
        self.max_moment = torch.tensor([2.5, 2.5, 1.5], device=device).view(1, 3).expand(num_envs, 3)

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
        
    def reset(self, env):
        pass