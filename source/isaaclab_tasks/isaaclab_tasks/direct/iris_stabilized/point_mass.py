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
    def __init__(self):
        pass

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
        curr_quat_w_yaw = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)
        roll_b, pitch_b, _ = euler_xyz_from_quat(quat_mul(curr_quat_w, quat_inv(curr_quat_w_yaw)))

        moment = torch.concat([
            -roll_b * 0.01 - curr_ang_vel_b[:,0],
            -pitch_b * 0.01 - curr_ang_vel_b[:,1],
            cmd_yaw_vel - curr_ang_vel_b[:,2]
        ])
        force = quat_rotate_inverse(curr_quat_w, cmd_lin_vel_w - curr_lin_vel_w)
        force[:, 2] *= 10
        
        return force, moment
        
    def reset(self, env):
        pass