import torch
from isaaclab.utils.math import (
    quat_rotate, 
    quat_rotate_inverse, 
    euler_xyz_from_quat, 
    quat_from_euler_xyz,
    )

class DroneStabilizingController:
    """Controller for drone stabilization with velocity commands in world frame.
    
    The controller takes linear velocity commands in world frame and yaw velocity command,
    while automatically managing roll and pitch for stability. It generates appropriate 
    accelerations in the body frame.
    """

    def __init__(
        self,
        linear_p_gain: float = 2.8,
        linear_d_gain: float = 0.2,
        linear_i_gain: float = 1.7,
        yaw_p_gain: float = 2.8,
        yaw_d_gain: float = 0.0,
        attitude_p_gain: float = 5.5,
        attitude_d_gain: float = 1.0,
        device: str = "cuda"
    ):
        """Initialize the drone stabilizing controller.
        
        Args:
            linear_p_gain: Proportional gain for linear velocity control. Defaults to 2.0.
            linear_d_gain: Derivative gain for linear velocity control. Defaults to 1.0.
            yaw_p_gain: Proportional gain for yaw control. Defaults to 2.0.
            yaw_d_gain: Derivative gain for yaw control. Defaults to 1.0.
            attitude_p_gain: Proportional gain for roll/pitch control. Defaults to 3.0.
            attitude_d_gain: Derivative gain for roll/pitch control. Defaults to 1.0.
            device: Device to create tensors on. Defaults to "cpu".
        """
        self.device = device
        # Store gains as tensors
        self.kp_lin = torch.tensor(linear_p_gain, device=device)
        self.kd_lin = torch.tensor(linear_d_gain, device=device)
        self.ki_lin = torch.tensor(linear_i_gain, device=device)
        self.kp_yaw = torch.tensor(yaw_p_gain, device=device)
        self.kd_yaw = torch.tensor(yaw_d_gain, device=device)
        self.kp_att = torch.tensor(attitude_p_gain, device=device)
        self.kd_att = torch.tensor(attitude_d_gain, device=device)
        self.cumul_lin_vel_error = None
        self.last_lin_vel_error = None
        self.last_ang_vel_error_b = None
        
    def compute_control(
        self,
        cmd_lin_vel_w: torch.Tensor,  # Command linear velocity in world frame
        cmd_yaw_vel: torch.Tensor,    # Command yaw velocity (scalar)
        curr_quat_w: torch.Tensor,    # Current orientation in world frame (w,x,y,z)
        curr_lin_vel_w: torch.Tensor, # Current linear velocity in world frame
        curr_ang_vel_b: torch.Tensor, # Current angular velocity in body frame
        dt: float = 0.02             # Time step
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # batch_size = cmd_lin_vel_w.shape[0]
        
        # Extract current roll, pitch, yaw
        roll, pitch, yaw = euler_xyz_from_quat(curr_quat_w)
        
        # Compute linear velocity error in world frame
        lin_vel_error_w = cmd_lin_vel_w - curr_lin_vel_w
        
        # Convert linear velocity error to body frame for control
        curr_yaw_quat_w = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)
        lin_vel_error_b_w = quat_rotate_inverse(curr_yaw_quat_w, lin_vel_error_w)
        
        # Compute control outputs with proper broadcasting
        # Linear acceleration in body frame
        if self.cumul_lin_vel_error is None:
            self.cumul_lin_vel_error = torch.zeros_like(lin_vel_error_w)
        self.cumul_lin_vel_error += lin_vel_error_w
        # self.cumul_lin_vel_error.clip(-10.0 * 0.1 / dt, 10.0 * 0.1 / dt)
        if self.last_lin_vel_error is None:
            self.last_lin_vel_error = lin_vel_error_w
        lin_acc_cmd_w = (
            self.kp_lin * lin_vel_error_w +
            self.kd_lin * (-self.last_lin_vel_error) +
            self.ki_lin * self.cumul_lin_vel_error
        )
        # lin_acc_cmd_w = quat_rotate(curr_yaw_quat_w, lin_acc_cmd_w)
        lin_acc_cmd_w[:, :2] = lin_acc_cmd_w[:, :2].clip(-5.0, 5.0)
        lin_acc_cmd_w[:, 2] = lin_acc_cmd_w[:, 2].clip(-1.0, 5.0)

        # Get desired tilt direction from velocity command
        # Compute xy velocity command magnitude
        xy_vel_cmd = cmd_lin_vel_w[:2]
        xy_vel_norm = torch.linalg.norm(xy_vel_cmd)
        


        # desired_thrust_per_weight = (9.82 + lin_acc_cmd_b_w[2]) / (
        #     z_w * quat_rotate_inverse(curr_quat_w, z_w))

        desired_thrust_per_weight = ((9.81 + lin_acc_cmd_w[:, 2]) / torch.cos(roll)) / torch.cos(pitch)
        # desired_thrust_per_weight = desired_thrust_per_weight.clip(0.0, 5.0)

        # desired_thrust_per_weight = torch.sqrt(
        #     (lin_acc_cmd_b_w[2] + 9.81)**2 * (1.0 + torch.tan(-roll)**2 + torch.tan(pitch)**2))

        # desired_thrust_per_weight = torch.sqrt(lin_acc_cmd_b_w[0]**2 + lin_acc_cmd_b_w[1]**2 + (lin_acc_cmd_b_w[2]-9.81)**2)

        # Initialize desired roll and pitch tensors
        desired_roll = torch.zeros_like(roll)
        desired_pitch = torch.zeros_like(pitch)

        # Compute desired roll and pitch only where velocity is significant
        # tilt_threshold = 0.0
        # mask = xy_vel_norm > tilt_threshold
        # angle_limit = torch.pi / 180.0 * 40.0
        # desired_roll[mask] -= torch.arctan(lin_acc_cmd_b_w[mask, 1])
        # desired_roll.clip(-angle_limit, angle_limit)
        # desired_pitch[mask] += torch.arctan(lin_acc_cmd_b_w[mask, 0])
        # desired_pitch.clip(-angle_limit, angle_limit)

        desired_roll_pitch_sin = (1/desired_thrust_per_weight).view(-1, 1, 1) * (torch.linalg.inv(
            torch.stack([torch.stack([torch.cos(roll)*torch.cos(yaw), torch.sin(yaw)], dim=1),
                        torch.stack([torch.cos(roll)*torch.sin(yaw), -torch.cos(yaw)], dim=1)], dim=1)
            ) @ torch.stack([lin_acc_cmd_w[:, 0], lin_acc_cmd_w[:, 1]], dim=1).unsqueeze(-1))


        desired_roll = torch.arcsin(desired_roll_pitch_sin[:,1])
        desired_pitch = torch.arcsin(desired_roll_pitch_sin[:,0])
        tilt_limit = torch.pi / 180.0 * 35.0
        desired_roll = desired_roll.clip(-tilt_limit, tilt_limit)
        desired_pitch = desired_pitch.clip(-tilt_limit, tilt_limit)
        self.desired_roll = desired_roll
        self.desired_pitch = desired_pitch

        
        # lin_acc_cmd_b = quat_rotate_inverse(
        #     quat_from_euler_xyz(desired_roll, desired_pitch, yaw), 
        #     lin_acc_cmd_w)
        
        # (lin_acc_cmd_b_w[1] / torch.sin(-desired_roll) +
        #                             lin_acc_cmd_b_w[0] / torch.sin(desired_pitch) +
        #                             lin_acc_cmd_b[2])

        # Compute attitude error
        roll_error = (desired_roll - roll.view(-1,1))
        pitch_error = (desired_pitch - pitch.view(-1,1))
        yaw_vel_error = (cmd_yaw_vel[:,] - curr_ang_vel_b[:,2])
        
        # Compute desired body rates with proper broadcasting
        desired_ang_vel_b = torch.zeros_like(curr_ang_vel_b)
        desired_ang_vel_b[:,0] = self.kp_att * roll_error[:,0]
        desired_ang_vel_b[:,1] = self.kp_att * pitch_error[:,0]
        desired_ang_vel_b[:,2] = cmd_yaw_vel
        
        # Angular velocity error in body frame
        ang_vel_error_b = desired_ang_vel_b - curr_ang_vel_b
        
        # Angular acceleration in body frame
        ang_acc_cmd_b = torch.zeros_like(curr_ang_vel_b)
        
        if self.last_ang_vel_error_b is None:
            self.last_ang_vel_error_b = torch.zeros_like(curr_ang_vel_b)

        # Roll and pitch control
        ang_acc_cmd_b[:,:2] = (
            self.kp_att * ang_vel_error_b[:,:2] +
            self.kd_att * (-self.last_ang_vel_error_b[:,:2])
        )
        self.last_ang_vel_error_b = ang_vel_error_b
        
        # Yaw control
        ang_acc_cmd_b[:,2] = (
            self.kp_yaw * yaw_vel_error +
            self.kd_yaw * (-curr_ang_vel_b[:,2])
        )
        
        return desired_thrust_per_weight, ang_acc_cmd_b
    
    def compute_desired_roll_pitch(self, desired_thrust_per_weight, roll, yaw, lin_acc_cmd_w):
        """
        Compute desired roll and pitch using PyTorch tensors.
        
        Args:
            desired_thrust_per_weight: Tensor of shape (num_dim,)
            roll: Tensor of shape (num_dim,)
            yaw: Tensor of shape (num_dim,)
            lin_acc_cmd_w: Tensor of shape (num_dim, 3)
        
        Returns:
            desired_roll_pitch_sin: Tensor of shape (num_dim, 2)
        """
        # Create transformation matrix elements
        cos_roll = torch.cos(roll)  # (num_dim,)
        sin_yaw = torch.sin(yaw)    # (num_dim,)
        cos_yaw = torch.cos(yaw)    # (num_dim,)
        
        # Create the 2x2 matrix for each dimension
        # We need to reshape to add the matrix dimensions
        batch_size = roll.shape[0]
        
        # First row: [cos(roll)*cos(yaw), sin(yaw)]
        row1_col1 = cos_roll * cos_yaw  # (num_dim,)
        row1_col2 = sin_yaw             # (num_dim,)
        
        # Second row: [cos(roll)*sin(yaw), -cos(yaw)]
        row2_col1 = cos_roll * sin_yaw  # (num_dim,)
        row2_col2 = -cos_yaw            # (num_dim,)
        
        # Stack the matrix elements
        matrices = torch.stack([
            torch.stack([row1_col1, row1_col2], dim=1),
            torch.stack([row2_col1, row2_col2], dim=1)
        ], dim=1)  # Shape: (num_dim, 2, 2)
        
        # Compute inverse for each 2x2 matrix
        matrices_inv = torch.inverse(matrices)  # Shape: (num_dim, 2, 2)
        
        # Extract the first two components of lin_acc_cmd_w
        acc_cmd = lin_acc_cmd_w[:, :2]  # Shape: (num_dim, 2)
        
        # Reshape desired_thrust_per_weight for broadcasting
        thrust_inv = (1.0 / desired_thrust_per_weight).unsqueeze(1)  # Shape: (num_dim, 1)
        
        # Compute final result
        # matmul will handle the batch dimension automatically
        desired_roll_pitch_sin = thrust_inv * torch.bmm(
            matrices_inv,
            acc_cmd.unsqueeze(2)  # Add dimension for matrix multiplication
        ).squeeze(2)  # Remove extra dimension after multiplication
        
        return desired_roll_pitch_sin
    
    def reset(self):
        """Reset the controller state."""
        pass  # No internal state to reset in this implementation

if __name__ == "__main__":
    # Test the controller
    controller = DroneStabilizingController()
    cmd_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    cmd_yaw_vel = torch.tensor([0.0], device="cuda")
    curr_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
    curr_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    curr_ang_vel_b = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    thrust, ang_acc = controller.compute_control(
        cmd_lin_vel_w, cmd_yaw_vel, curr_quat_w, curr_lin_vel_w, curr_ang_vel_b)
    print("Thrust:", thrust)
    print("Angular acceleration:", ang_acc)
    controller.reset()