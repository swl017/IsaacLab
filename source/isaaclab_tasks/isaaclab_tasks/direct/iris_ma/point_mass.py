import torch
from isaaclab.utils.math import (
    quat_rotate, 
    quat_rotate_inverse, 
    euler_xyz_from_quat, 
    quat_from_euler_xyz,
    quat_mul,
    quat_inv
    )

class PointMass:
    def __init__(self, mass: float = 1.0, weight: float | None = None, num_envs: int = 1, device: str = 'cuda:0'):
        self.mass = mass
        self.weight = weight if weight is not None else mass * 9.81
        self.weight_tensor = torch.tensor([0, 0, -self.weight], dtype=torch.float32, device=device).view(1, 3).expand(num_envs, 3) if num_envs > 1 else torch.tensor([0, 0, -self.weight], dtype=torch.float32, device=device).view(1, 3)
        # self.weight_tensor = torch.tensor(self.weight, dtype=torch.float32).view(1, 1).expand(num_envs, 1) if num_envs > 1 else torch.tensor(self.weight, dtype=torch.float32).view(1, 1)
        self.num_envs = num_envs

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
        dt: float = 0.02             # Time step
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the control commands for the point mass.
        Apply torques to stabilize.
        """
        # Convert current orientation to Euler angles
        roll, pitch, yaw = euler_xyz_from_quat(curr_quat_w)
        curr_ang_vel_w = quat_rotate(curr_quat_w, curr_ang_vel_b)
        moment_w = torch.stack([
            -self.wrap_to_pi(roll) * 2 - curr_ang_vel_w[:,0]*1.3,
            -self.wrap_to_pi(pitch) * 2 - curr_ang_vel_w[:,1]*1.3,
            (cmd_yaw_vel - curr_ang_vel_w[:,2])
        ], dim=1)
        moment = quat_rotate_inverse(curr_quat_w, moment_w)

        feedback = cmd_lin_vel_w - curr_lin_vel_w
        feedforward = -self.weight_tensor
        force = quat_rotate_inverse(curr_quat_w, feedback * 3.0)
        # force[:, 2] *= 3.0
        # force[:, 2] = torch.clamp(force[:, 2], min=0.0)
        
        
        return force, moment
        
    def reset(self, env):
        pass