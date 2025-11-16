"""
Gimbal Stabilization System - Gimbal Lock Free Implementation

This module provides gimbal stabilization by computing angles directly in the gimbal base frame.

Frame Convention:
- Drone Base: ENU (East-North-Up), FLU (Forward-Left-Up)
- Gimbal Base: ENU (East-North-Up), FLU (Forward-Left-Up)
- Gimbal Yaw: FLU, rotation around Up axis (right-hand rule, CCW positive when viewed from above)
- Gimbal Roll: FLU, rotation around Forward axis (right-hand rule, applied after yaw)
- Gimbal Pitch: FLU, rotation around Left axis (right-hand rule, left/down positive, applied after yaw and roll)

Author: Claude Code
"""

import torch
import math


class GimbalStabilizer:
    """
    Gimbal stabilizer that computes target angles directly in gimbal base frame.

    This approach avoids gimbal lock by:
    1. Working directly with the desired pointing vector in gimbal base frame
    2. Computing yaw and pitch angles from this vector
    3. Never converting through Euler angles from quaternions
    """

    def __init__(
        self,
        device: str,
        yaw_limits: list[float] = [-math.pi, math.pi],
        roll_limits: list[float] = [-math.pi, math.pi],
        pitch_limits: list[float] = [-math.pi/2, math.pi/2],
    ):
        """
        Initialize gimbal stabilizer.

        Args:
            device: PyTorch device (cpu or cuda)
            yaw_limits: [min, max] yaw angle limits in radians
            roll_limits: [min, max] roll angle limits in radians
            pitch_limits: [min, max] pitch angle limits in radians
        """
        self.device = device
        self.yaw_limits = yaw_limits
        self.roll_limits = roll_limits
        self.pitch_limits = pitch_limits

    def compute_stabilized_angles(
        self,
        target_direction_world: torch.Tensor,
        drone_quat_world: torch.Tensor,
        desired_roll: torch.Tensor | None = None,
        auto_stabilize_roll: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gimbal yaw, roll, and pitch angles to point at target direction in world frame.

        Args:
            target_direction_world: (N, 3) target pointing direction in world frame (ENU)
            drone_quat_world: (N, 4) drone orientation quaternion in world frame (w, x, y, z)
            desired_roll: (N,) desired roll angle in radians. If None and auto_stabilize_roll=True,
                         computes stabilizing roll automatically. If None and auto_stabilize_roll=False,
                         roll = 0.
            auto_stabilize_roll: If True and desired_roll is None, automatically compute stabilizing
                                roll to keep horizon level. Default: True.

        Returns:
            gimbal_yaw: (N,) yaw angle in radians (around up axis, CCW positive)
            gimbal_roll: (N,) roll angle in radians (around forward axis after yaw)
            gimbal_pitch: (N,) pitch angle in radians (around left axis, left/down positive)
        """
        # Normalize target direction
        target_direction_world = target_direction_world / (
            torch.norm(target_direction_world, dim=-1, keepdim=True) + 1e-8
        )

        # Transform target direction from world frame to gimbal base frame (drone body frame)
        # Gimbal base is fixed to drone body, so we use drone's inverse rotation
        target_direction_gimbal = self._rotate_vector_by_quat_inverse(
            target_direction_world, drone_quat_world
        )

        # Extract components in gimbal base frame (FLU convention)
        # x: forward, y: left, z: up
        x = target_direction_gimbal[:, 0]  # forward
        y = target_direction_gimbal[:, 1]  # left
        z = target_direction_gimbal[:, 2]  # up

        # Compute yaw angle (rotation around up/z axis)
        # atan2(left, forward) gives yaw in FLU frame
        # Positive yaw rotates forward vector toward left (CCW when viewed from above)
        gimbal_yaw = torch.atan2(y, x)

        # Compute pitch angle (rotation around left/y axis after yaw and roll)
        # After yaw rotation, we're in the yaw-rotated frame
        # Pitch is the angle from horizontal plane to target
        # atan2(-up, forward_horizontal) where forward_horizontal = sqrt(x^2 + y^2)
        horizontal_dist = torch.sqrt(x**2 + y**2 + 1e-8)
        gimbal_pitch = torch.atan2(-z, horizontal_dist)  # Negative because down is positive pitch

        # Determine roll angle
        if desired_roll is not None:
            gimbal_roll = desired_roll
        elif auto_stabilize_roll:
            # Compute stabilizing roll to keep horizon level
            gimbal_roll = self.compute_stabilizing_roll(gimbal_yaw, gimbal_pitch, drone_quat_world)
        else:
            gimbal_roll = torch.zeros_like(gimbal_yaw)

        # Clamp angles to limits
        gimbal_yaw = torch.clamp(gimbal_yaw, self.yaw_limits[0], self.yaw_limits[1])
        gimbal_roll = torch.clamp(gimbal_roll, self.roll_limits[0], self.roll_limits[1])
        gimbal_pitch = torch.clamp(gimbal_pitch, self.pitch_limits[0], self.pitch_limits[1])

        return gimbal_yaw, gimbal_roll, gimbal_pitch

    def compute_stabilized_angles_from_target_point(
        self,
        target_point_world: torch.Tensor,
        drone_position_world: torch.Tensor,
        drone_quat_world: torch.Tensor,
        desired_roll: torch.Tensor | None = None,
        auto_stabilize_roll: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gimbal angles to point camera at a target point in world frame.

        Args:
            target_point_world: (N, 3) target point position in world frame
            drone_position_world: (N, 3) drone position in world frame
            drone_quat_world: (N, 4) drone orientation quaternion (w, x, y, z)
            desired_roll: (N,) desired roll angle in radians. If None and auto_stabilize_roll=True,
                         computes stabilizing roll automatically.
            auto_stabilize_roll: If True and desired_roll is None, automatically compute stabilizing
                                roll to keep horizon level. Default: True.

        Returns:
            gimbal_yaw: (N,) yaw angle in radians
            gimbal_roll: (N,) roll angle in radians
            gimbal_pitch: (N,) pitch angle in radians
        """
        # Compute direction vector from drone to target
        target_direction_world = target_point_world - drone_position_world

        return self.compute_stabilized_angles(
            target_direction_world, drone_quat_world, desired_roll, auto_stabilize_roll
        )

    def integrate_gimbal_rates(
        self,
        current_yaw: torch.Tensor,
        current_roll: torch.Tensor,
        current_pitch: torch.Tensor,
        yaw_rate: torch.Tensor,
        roll_rate: torch.Tensor,
        pitch_rate: torch.Tensor,
        dt: float,
        max_rate: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate gimbal rates to get new angles with rate limiting.

        Args:
            current_yaw: (N,) current yaw angles in radians
            current_roll: (N,) current roll angles in radians
            current_pitch: (N,) current pitch angles in radians
            yaw_rate: (N,) commanded yaw rate (normalized -1 to 1)
            roll_rate: (N,) commanded roll rate (normalized -1 to 1)
            pitch_rate: (N,) commanded pitch rate (normalized -1 to 1)
            dt: time step in seconds
            max_rate: maximum angular rate in rad/s

        Returns:
            new_yaw: (N,) new yaw angles in radians
            new_roll: (N,) new roll angles in radians
            new_pitch: (N,) new pitch angles in radians
        """
        # Apply rate limiting
        yaw_rate = torch.clamp(yaw_rate, -1.0, 1.0)
        roll_rate = torch.clamp(roll_rate, -1.0, 1.0)
        pitch_rate = torch.clamp(pitch_rate, -1.0, 1.0)

        # Integrate
        new_yaw = current_yaw + yaw_rate * max_rate * dt
        new_roll = current_roll + roll_rate * max_rate * dt
        new_pitch = current_pitch + pitch_rate * max_rate * dt

        # Clamp to limits
        new_yaw = torch.clamp(new_yaw, self.yaw_limits[0], self.yaw_limits[1])
        new_roll = torch.clamp(new_roll, self.roll_limits[0], self.roll_limits[1])
        new_pitch = torch.clamp(new_pitch, self.pitch_limits[0], self.pitch_limits[1])

        return new_yaw, new_roll, new_pitch

    def compute_stabilizing_roll(
        self,
        gimbal_yaw: torch.Tensor,
        gimbal_pitch: torch.Tensor,
        drone_quat_world: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the roll angle needed to keep the horizon level (stabilized) given yaw and pitch.

        This method finds the roll angle that aligns the gimbal's "up" vector with the world's
        up vector (gravity direction), keeping the horizon level in the camera view.

        Args:
            gimbal_yaw: (N,) yaw angles in radians
            gimbal_pitch: (N,) pitch angles in radians
            drone_quat_world: (N, 4) drone orientation quaternion in world frame (w, x, y, z)

        Returns:
            stabilizing_roll: (N,) roll angle in radians that keeps horizon level
        """
        # World up vector (gravity direction in ENU is [0, 0, 1])
        world_up = torch.zeros((gimbal_yaw.shape[0], 3), device=self.device)
        world_up[:, 2] = 1.0

        # Transform world up to gimbal base frame (drone body frame)
        up_in_gimbal_base = self._rotate_vector_by_quat_inverse(world_up, drone_quat_world)

        # After yaw rotation, we're in the yawed frame
        # We need to find the roll that aligns the camera's up with the projected world up

        # Rotate the up vector by inverse yaw to get it into the yawed frame
        cos_yaw = torch.cos(gimbal_yaw)
        sin_yaw = torch.sin(gimbal_yaw)

        # Inverse yaw rotation around z-axis
        up_x_yawed = up_in_gimbal_base[:, 0] * cos_yaw + up_in_gimbal_base[:, 1] * sin_yaw
        up_y_yawed = -up_in_gimbal_base[:, 0] * sin_yaw + up_in_gimbal_base[:, 1] * cos_yaw
        up_z_yawed = up_in_gimbal_base[:, 2]

        # After yaw, before roll and pitch, the gimbal's up direction is [0, 0, 1] in yawed frame
        # After pitch rotation by angle p around y-axis, the up direction becomes:
        # up_after_pitch = [sin(p), 0, cos(p)]

        # We want: R_roll * up_after_pitch ≈ [up_x_yawed, up_y_yawed, up_z_yawed]
        # where R_roll is rotation around x-axis (forward in yawed frame)

        # The pitch rotation moves the up vector to [sin(p), 0, cos(p)]
        cos_pitch = torch.cos(gimbal_pitch)
        sin_pitch = torch.sin(gimbal_pitch)

        # Up vector after pitch (in yawed frame, before roll)
        up_after_pitch_x = sin_pitch
        up_after_pitch_y = torch.zeros_like(sin_pitch)
        up_after_pitch_z = cos_pitch

        # We need to find roll such that rotating up_after_pitch by roll gives us the target up
        # R_roll(r) * [sin(p), 0, cos(p)] should align with [up_x_yawed, up_y_yawed, up_z_yawed]

        # After roll around x-axis:
        # x' = sin(p)  (unchanged by roll around x)
        # y' = -cos(p) * sin(r)
        # z' = cos(p) * cos(r)

        # We want y' ≈ up_y_yawed, so:
        # -cos(p) * sin(r) = up_y_yawed
        # sin(r) = -up_y_yawed / cos(p)

        # And z' ≈ up_z_yawed, so:
        # cos(p) * cos(r) = up_z_yawed
        # cos(r) = up_z_yawed / cos(p)

        # Use atan2 for robust computation
        stabilizing_roll = torch.atan2(-up_y_yawed, up_z_yawed / (cos_pitch + 1e-8))

        # Clamp to limits
        stabilizing_roll = torch.clamp(stabilizing_roll, self.roll_limits[0], self.roll_limits[1])

        return stabilizing_roll

    def get_camera_pointing_direction(
        self,
        gimbal_yaw: torch.Tensor,
        gimbal_roll: torch.Tensor,
        gimbal_pitch: torch.Tensor,
        drone_quat_world: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the camera pointing direction in world frame given gimbal angles.

        Args:
            gimbal_yaw: (N,) yaw angles in radians
            gimbal_roll: (N,) roll angles in radians
            gimbal_pitch: (N,) pitch angles in radians
            drone_quat_world: (N, 4) drone orientation quaternion (w, x, y, z)

        Returns:
            pointing_direction_world: (N, 3) camera pointing direction in world frame
        """
        # Compute pointing direction in gimbal base frame
        # Apply rotations in order: yaw -> roll -> pitch
        # Starting with forward (1, 0, 0) in gimbal base frame

        # Create rotation matrices for each axis
        cos_yaw = torch.cos(gimbal_yaw)
        sin_yaw = torch.sin(gimbal_yaw)
        cos_roll = torch.cos(gimbal_roll)
        sin_roll = torch.sin(gimbal_roll)
        cos_pitch = torch.cos(gimbal_pitch)
        sin_pitch = torch.sin(gimbal_pitch)

        # Combined rotation: R_pitch * R_roll * R_yaw * [1, 0, 0]
        # After working through the matrix multiplications for yaw->roll->pitch on [1,0,0]:
        x = cos_yaw * cos_pitch - sin_yaw * sin_roll * sin_pitch
        y = sin_yaw * cos_pitch + cos_yaw * sin_roll * sin_pitch
        z = -cos_roll * sin_pitch

        direction_gimbal = torch.stack([x, y, z], dim=-1)

        # Transform to world frame using drone orientation
        pointing_direction_world = self._rotate_vector_by_quat(
            direction_gimbal, drone_quat_world
        )

        return pointing_direction_world

    def _rotate_vector_by_quat(
        self,
        vec: torch.Tensor,
        quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Rotate vector by quaternion: q * v * q^-1

        Args:
            vec: (N, 3) vectors to rotate
            quat: (N, 4) quaternions (w, x, y, z)

        Returns:
            rotated_vec: (N, 3) rotated vectors
        """
        # Extract quaternion components
        w = quat[:, 0:1]
        x = quat[:, 1:2]
        y = quat[:, 2:3]
        z = quat[:, 3:4]

        # Quaternion rotation formula: v' = v + 2*r x (s*v + r x v)
        # where r = (x, y, z), s = w
        r = quat[:, 1:4]  # (N, 3)
        s = quat[:, 0:1]  # (N, 1)

        # Cross product: r x v
        r_cross_v = torch.cross(r, vec, dim=-1)

        # s*v + r x v
        sv_plus_rcv = s * vec + r_cross_v

        # r x (s*v + r x v)
        r_cross_sv = torch.cross(r, sv_plus_rcv, dim=-1)

        # v' = v + 2 * r x (s*v + r x v)
        rotated_vec = vec + 2.0 * r_cross_sv

        return rotated_vec

    def _rotate_vector_by_quat_inverse(
        self,
        vec: torch.Tensor,
        quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Rotate vector by inverse quaternion: q^-1 * v * q

        Args:
            vec: (N, 3) vectors to rotate
            quat: (N, 4) quaternions (w, x, y, z)

        Returns:
            rotated_vec: (N, 3) rotated vectors
        """
        # Conjugate quaternion (inverse for unit quaternions)
        quat_inv = quat.clone()
        quat_inv[:, 1:4] = -quat_inv[:, 1:4]  # Negate x, y, z components

        return self._rotate_vector_by_quat(vec, quat_inv)


def create_gimbal_stabilizer(
    device: str,
    yaw_limits_deg: tuple[float, float] = (-180.0, 180.0),
    roll_limits_deg: tuple[float, float] = (-180.0, 180.0),
    pitch_limits_deg: tuple[float, float] = (-90.0, 90.0),
) -> GimbalStabilizer:
    """
    Factory function to create a gimbal stabilizer with degree inputs.

    Args:
        device: PyTorch device
        yaw_limits_deg: Yaw limits in degrees
        roll_limits_deg: Roll limits in degrees
        pitch_limits_deg: Pitch limits in degrees

    Returns:
        GimbalStabilizer instance
    """
    yaw_limits_rad = (math.radians(yaw_limits_deg[0]), math.radians(yaw_limits_deg[1]))
    roll_limits_rad = (math.radians(roll_limits_deg[0]), math.radians(roll_limits_deg[1]))
    pitch_limits_rad = (math.radians(pitch_limits_deg[0]), math.radians(pitch_limits_deg[1]))

    return GimbalStabilizer(
        device=device,
        yaw_limits=yaw_limits_rad,
        roll_limits=roll_limits_rad,
        pitch_limits=pitch_limits_rad,
    )
