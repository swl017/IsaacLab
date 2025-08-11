import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
import torch
import math
from isaaclab.utils.math import (
    subtract_frame_transforms, 
    euler_xyz_from_quat, 
    quat_from_euler_xyz,
    quat_mul,
    quat_inv,
    quat_rotate
)

class CameraFrustrum:
    def __init__(self):
        self.camera_name = None
        self.camera_position = None
        self.camera_orientation = None
        self.camera_intrinsics = None

        self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    def draw_frustrum(self, camera_position, camera_orientation, camera_intrinsics: torch.tensor, device='cpu'):
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        self.camera_intrinsics = camera_intrinsics

        frustum_b = self.compute_camera_frustum_batched(camera_intrinsics, device=device)['far_corners']
        frustum_w = frustum_b.clone()
        for i in range(frustum_b.shape[1]):
            frustum_w[:,i] = self.camera_position + quat_rotate(self.camera_orientation, frustum_b[:,i])

        line_colors = [[1.0, 1.0, 1.0, 0.3]] * self.camera_position.shape[0]
        line_thicknesses = [1.0] * self.camera_position.shape[0]
        for i in range(frustum_b.shape[1]):
            self.draw_interface.draw_lines(self.camera_position.tolist(), frustum_w[:,i,:].tolist(), line_colors, line_thicknesses)
            if i < frustum_b.shape[1] - 1:
                self.draw_interface.draw_lines(frustum_w[:,i,:].tolist(), frustum_w[:,i+1,:].tolist(), line_colors, line_thicknesses)
            else:
                self.draw_interface.draw_lines(frustum_w[:,i,:].tolist(), frustum_w[:,0,:].tolist(), line_colors, line_thicknesses)
                

    def compute_camera_frustum_batched(self, camera_cfg_tensor: torch.tensor, device='cpu', dtype=torch.float32):
        """
        Compute camera frustum parameters for n environments using batched PyTorch tensors.
        
        Args:
            camera_cfg_tensor: Batched tensor containing camera parameters with shape [n, num_params]
                            Expected parameter order:
                            [width, height, focal_length, horizontal_aperture, near_plane, far_plane]
            device: Device to create tensors on ('cpu', 'cuda', etc.)
            dtype: Data type for tensors (default: torch.float32)
        
        Returns:
            dict: Dictionary containing batched frustum parameters as PyTorch tensors
        """
        # Ensure input is on correct device and dtype
        camera_cfg_tensor = camera_cfg_tensor.to(device=device, dtype=dtype)
        n = camera_cfg_tensor.shape[0]
        
        # Extract camera parameters for all environments
        width = camera_cfg_tensor[:, 0]          # [n]
        height = camera_cfg_tensor[:, 1]         # [n]
        focal_length = camera_cfg_tensor[:, 2]   # [n] in cm
        horizontal_aperture = camera_cfg_tensor[:, 3]  # [n] in cm
        near_plane_max = torch.ones_like(camera_cfg_tensor[:, 4]) * 0.5
        near_plane = torch.where(camera_cfg_tensor[:, 4] > near_plane_max, camera_cfg_tensor[:, 4], near_plane_max)  # [n]
        far_plane_min = torch.ones_like(camera_cfg_tensor[:, 5]) * 1.0
        far_plane_max = torch.ones_like(camera_cfg_tensor[:, 5]) * 2.0
        far_plane = torch.where(camera_cfg_tensor[:, 5] < far_plane_min, far_plane_min, torch.where(camera_cfg_tensor[:, 5] > far_plane_max, far_plane_max, camera_cfg_tensor[:, 5]))  # [n]

        # Calculate field of view for all environments
        # Horizontal FOV from focal length and sensor width (horizontal aperture)
        horizontal_fov_rad = 2 * torch.atan(horizontal_aperture / (2 * focal_length))  # [n]
        horizontal_fov_deg = torch.rad2deg(horizontal_fov_rad)  # [n]
        
        # Calculate vertical aperture based on aspect ratio
        aspect_ratio = width / height  # [n]
        vertical_aperture = horizontal_aperture / aspect_ratio  # [n]
        
        # Vertical FOV
        vertical_fov_rad = 2 * torch.atan(vertical_aperture / (2 * focal_length))  # [n]
        vertical_fov_deg = torch.rad2deg(vertical_fov_rad)  # [n]
        
        # Calculate frustum dimensions at near and far planes
        # At near plane
        near_height = 2 * near_plane * torch.tan(vertical_fov_rad / 2)    # [n]
        near_width = 2 * near_plane * torch.tan(horizontal_fov_rad / 2)   # [n]
        
        # At far plane
        far_height = 2 * far_plane * torch.tan(vertical_fov_rad / 2)      # [n]
        far_width = 2 * far_plane * torch.tan(horizontal_fov_rad / 2)     # [n]
        
        # Frustum corners at near plane (in camera coordinates)
        # Shape: [n, 4, 3] - n environments, 4 corners, 3 coordinates (x,y,z)
        near_corners = torch.stack([
            torch.stack([-near_width/2, -near_height/2, near_plane], dim=1),  # bottom-left
            torch.stack([ near_width/2, -near_height/2, near_plane], dim=1),  # bottom-right
            torch.stack([ near_width/2,  near_height/2, near_plane], dim=1),  # top-right
            torch.stack([-near_width/2,  near_height/2, near_plane], dim=1),  # top-left
        ], dim=1)  # [n, 4, 3]
        
        # Frustum corners at far plane (in camera coordinates)
        # Shape: [n, 4, 3] - n environments, 4 corners, 3 coordinates (x,y,z)
        far_corners = torch.stack([
            torch.stack([-far_width/2, -far_height/2, far_plane], dim=1),  # bottom-left
            torch.stack([ far_width/2, -far_height/2, far_plane], dim=1),  # bottom-right
            torch.stack([ far_width/2,  far_height/2, far_plane], dim=1),  # top-right
            torch.stack([-far_width/2,  far_height/2, far_plane], dim=1),  # top-left
        ], dim=1)  # [n, 4, 3]
        
        # Create projection matrix (perspective)
        f_x = focal_length / horizontal_aperture * width  # focal length in pixels [n]
        f_y = focal_length / vertical_aperture * height   # focal length in pixels [n]
        c_x = width / 2   # principal point x [n]
        c_y = height / 2  # principal point y [n]
        
        # Camera intrinsic matrix - batched [n, 3, 3]
        zeros = torch.zeros_like(f_x)
        ones = torch.ones_like(f_x)
        
        K = torch.stack([
            torch.stack([f_x, zeros, c_x], dim=1),
            torch.stack([zeros, f_y, c_y], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=1)  # [n, 3, 3]
        
        # Perspective projection matrix - batched [n, 4, 4]
        P_00 = 2 * near_plane / near_width
        P_11 = 2 * near_plane / near_height
        P_22 = -(far_plane + near_plane) / (far_plane - near_plane)
        P_23 = -2 * far_plane * near_plane / (far_plane - near_plane)
        neg_ones = -torch.ones_like(f_x)
        
        P = torch.stack([
            torch.stack([P_00, zeros, zeros, zeros], dim=1),
            torch.stack([zeros, P_11, zeros, zeros], dim=1),
            torch.stack([zeros, zeros, P_22, P_23], dim=1),
            torch.stack([zeros, zeros, neg_ones, zeros], dim=1)
        ], dim=1)  # [n, 4, 4]
        
        return {
            'n_environments': n,
            'width': width,                          # [n]
            'height': height,                        # [n]
            'aspect_ratio': aspect_ratio,            # [n]
            'focal_length_mm': focal_length,         # [n]
            'horizontal_aperture_mm': horizontal_aperture,  # [n]
            'vertical_aperture_mm': vertical_aperture,      # [n]
            'horizontal_fov_deg': horizontal_fov_deg,       # [n]
            'vertical_fov_deg': vertical_fov_deg,           # [n]
            'horizontal_fov_rad': horizontal_fov_rad,       # [n]
            'vertical_fov_rad': vertical_fov_rad,           # [n]
            'near_plane': near_plane,                # [n]
            'far_plane': far_plane,                  # [n]
            'near_width': near_width,                # [n]
            'near_height': near_height,              # [n]
            'far_width': far_width,                  # [n]
            'far_height': far_height,                # [n]
            'near_corners': near_corners,            # [n, 4, 3]
            'far_corners': far_corners,              # [n, 4, 3]
            'intrinsic_matrix': K,                   # [n, 3, 3]
            'projection_matrix': P,                  # [n, 4, 4]
            'focal_length_pixels': torch.stack([f_x, f_y], dim=1),  # [n, 2]
            'principal_point': torch.stack([c_x, c_y], dim=1),      # [n, 2]
        }

    def create_camera_cfg_tensor(self, camera_cfg: CameraCfg, num_envs: int, device='cpu', dtype=torch.float32):
        """
        Helper function to create batched tensor from list of camera configurations.
        
        Args:
            camera_cfg: A TiledCameraCfg object
            device: Device to create tensors on
            dtype: Data type for tensors
        
        Returns:
            torch.Tensor: Batched tensor of shape [n, 6] containing camera parameters
        """
        camera_params = [
            (camera_cfg.width, camera_cfg.height, camera_cfg.spawn.focal_length,
            camera_cfg.spawn.horizontal_aperture, camera_cfg.spawn.clipping_range[0],
            camera_cfg.spawn.clipping_range[1])
        ] * num_envs  # Repeat for each environment

        camera_cfg_batch = torch.tensor(camera_params, device=device, dtype=dtype)

        return camera_cfg_batch