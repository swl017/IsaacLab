# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder for capturing simulation viewport frames."""

from __future__ import annotations

import cv2
import numpy as np
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


@dataclass
class VideoRecorderCfg:
    """Configuration for video recorder."""

    output_path: str = "output.mp4"
    """Output video file path."""

    fps: int = 30
    """Frames per second for output video."""

    resolution: tuple[int, int] = (1920, 1080)
    """Video resolution (width, height)."""

    codec: str = "mp4v"
    """OpenCV fourcc codec string."""

    camera_name: str | None = None
    """Name of camera sensor to use. If None, uses viewport capture."""

    env_index: int = 0
    """Environment index to record (for camera sensor mode)."""

    resize_mode: str = "fit"
    """How to handle resolution mismatch: 'fit', 'crop', 'stretch'."""

    headless: bool = False
    """Enable headless recording using offscreen camera (no GUI required)."""

    headless_prim_path: str = "/World/VideoCamera"
    """Prim path for headless offscreen camera."""

    use_replicator: bool = True
    """Use omni.replicator for headless rendering (more reliable)."""


class VideoRecorder:
    """Record simulation frames to video file.

    This recorder can capture frames from either:
    1. A camera sensor in the scene (preferred for consistent resolution)
    2. The viewport directly (for viewport camera views)
    3. An offscreen camera (for headless mode)

    Usage:
        recorder = VideoRecorder(cfg=VideoRecorderCfg(output_path="demo.mp4"))
        recorder.start()

        for step in range(num_steps):
            # ... simulation step ...
            recorder.capture_frame(env)

        recorder.stop()

    For headless mode:
        recorder = VideoRecorder(cfg=VideoRecorderCfg(
            output_path="demo.mp4",
            headless=True,
        ))
        recorder.start()
        recorder.setup_offscreen_camera(env)  # Call after env is created

        for step in range(num_steps):
            recorder.set_camera_pose(eye, lookat)  # From smooth camera
            recorder.capture_frame(env)

        recorder.stop()
    """

    def __init__(self, cfg: VideoRecorderCfg | None = None):
        """Initialize the video recorder.

        Args:
            cfg: Configuration for the recorder. If None, uses defaults.
        """
        self.cfg = cfg if cfg is not None else VideoRecorderCfg()

        self._writer: cv2.VideoWriter | None = None
        self._recording: bool = False
        self._frame_count: int = 0
        self._initialized: bool = False

        # Headless mode state
        self._offscreen_camera = None
        self._camera_eye: np.ndarray | None = None
        self._camera_lookat: np.ndarray | None = None

        # Replicator-based rendering state
        self._render_product = None
        self._rgb_annotator = None
        self._camera_prim = None

    @property
    def is_recording(self) -> bool:
        """Whether recording is active."""
        return self._recording

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return self._frame_count

    def start(self):
        """Start recording video.

        Creates the output directory and initializes the video writer.
        """
        if self._recording:
            print("[VideoRecorder] Already recording, ignoring start()")
            return

        # Ensure output directory exists
        output_path = Path(self.cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine best codec based on output format
        output_ext = output_path.suffix.lower()
        if output_ext == ".mp4":
            # Try H264 first, fallback to mp4v
            codecs_to_try = ["avc1", "H264", "X264", "mp4v"]
        elif output_ext == ".avi":
            codecs_to_try = ["XVID", "MJPG", "DIVX"]
        else:
            codecs_to_try = [self.cfg.codec]

        # Try codecs in order until one works
        self._writer = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self._writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.cfg.fps,
                    self.cfg.resolution,
                )
                if self._writer.isOpened():
                    print(f"[VideoRecorder] Using codec: {codec}")
                    break
                else:
                    self._writer.release()
                    self._writer = None
            except Exception:
                continue

        if self._writer is None or not self._writer.isOpened():
            print(f"[VideoRecorder] WARNING: Could not initialize video writer with any codec")
            print(f"[VideoRecorder] Tried codecs: {codecs_to_try}")
            # Create a fallback writer anyway
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.cfg.fps,
                self.cfg.resolution,
            )

        self._recording = True
        self._frame_count = 0
        self._initialized = False

        print(f"[VideoRecorder] Started recording to: {output_path}")

    def stop(self):
        """Stop recording and finalize video file."""
        if not self._recording:
            return

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        # Clean up replicator resources
        if self._rgb_annotator is not None:
            try:
                self._rgb_annotator.detach([self._render_product])
            except Exception:
                pass
            self._rgb_annotator = None

        if self._render_product is not None:
            try:
                import omni.replicator.core as rep
                # rep.orchestrator.stop()  # Don't stop orchestrator, might affect sim
            except Exception:
                pass
            self._render_product = None

        self._recording = False
        print(f"[VideoRecorder] Stopped recording. Total frames: {self._frame_count}")

    def setup_offscreen_camera(self, env: DirectRLEnv):
        """Set up an offscreen camera for headless video recording.

        Creates either an Isaac Lab Camera sensor or uses omni.replicator
        for headless rendering.

        Args:
            env: The environment to add the camera to.
        """
        if not self.cfg.headless:
            return

        width, height = self.cfg.resolution

        if self.cfg.use_replicator:
            # Method 1: Use omni.replicator for rendering (more reliable)
            self._setup_replicator_camera(width, height)
        else:
            # Method 2: Use Isaac Lab Camera sensor
            self._setup_isaaclab_camera(width, height)

    def _setup_replicator_camera(self, width: int, height: int):
        """Set up camera using omni.replicator for headless rendering."""
        try:
            import omni.replicator.core as rep
            from pxr import UsdGeom, Sdf, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()

            # Create camera prim if it doesn't exist
            camera_path = self.cfg.headless_prim_path
            camera_prim = stage.GetPrimAtPath(camera_path)

            if not camera_prim.IsValid():
                # Create camera using USD API
                camera_prim = stage.DefinePrim(camera_path, "Camera")
                camera = UsdGeom.Camera(camera_prim)

                # Set camera properties
                camera.GetFocalLengthAttr().Set(24.0)
                camera.GetHorizontalApertureAttr().Set(20.955)
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

                # Set initial position (will be updated by smooth camera)
                xformable = UsdGeom.Xformable(camera_prim)
                xform_op = xformable.AddTransformOp()
                transform = Gf.Matrix4d()
                transform.SetTranslateOnly(Gf.Vec3d(0, 0, 50))
                xform_op.Set(transform)

            self._camera_prim = camera_prim

            # Create render product for the camera
            self._render_product = rep.create.render_product(camera_path, (width, height))

            # Create RGB annotator and attach to render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach([self._render_product])

            print(f"[VideoRecorder] Created replicator camera at {camera_path}")
            print(f"[VideoRecorder] Resolution: {width}x{height}")

        except Exception as e:
            print(f"[VideoRecorder] Error creating replicator camera: {e}")
            import traceback
            traceback.print_exc()
            self._render_product = None
            self._rgb_annotator = None

    def _setup_isaaclab_camera(self, width: int, height: int):
        """Set up camera using Isaac Lab Camera sensor."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import Camera, CameraCfg

            camera_cfg = CameraCfg(
                prim_path=self.cfg.headless_prim_path,
                update_period=0,  # Update every step
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1000.0),
                ),
            )

            self._offscreen_camera = Camera(cfg=camera_cfg)
            self._offscreen_camera.reset()

            print(f"[VideoRecorder] Created Isaac Lab camera at {self.cfg.headless_prim_path}")
            print(f"[VideoRecorder] Resolution: {width}x{height}")

        except Exception as e:
            print(f"[VideoRecorder] Error creating Isaac Lab camera: {e}")
            import traceback
            traceback.print_exc()
            self._offscreen_camera = None

    def set_camera_pose(self, eye: np.ndarray | torch.Tensor, lookat: np.ndarray | torch.Tensor):
        """Set the camera position and look-at target for headless recording.

        This is called from the smooth camera controller to update camera position.

        Args:
            eye: Camera position [x, y, z].
            lookat: Look-at target position [x, y, z].
        """
        # Convert to numpy if needed
        if isinstance(eye, torch.Tensor):
            eye = eye.detach().cpu().numpy()
        if isinstance(lookat, torch.Tensor):
            lookat = lookat.detach().cpu().numpy()

        self._camera_eye = np.asarray(eye, dtype=np.float32)
        self._camera_lookat = np.asarray(lookat, dtype=np.float32)

        # Debug output on first pose update and periodically
        if self._frame_count == 0 and not self._initialized:
            print(f"[VideoRecorder] First camera pose: eye={eye}, lookat={lookat}")
        elif self._frame_count == 50:
            # Log at frame 50 to verify camera is moving
            print(f"[VideoRecorder] Frame 50 camera pose: eye={eye}, lookat={lookat}")

        # Update the camera position (works for both replicator and Isaac Lab cameras)
        if self._camera_prim is not None or self._offscreen_camera is not None:
            self._update_offscreen_camera_pose()

            # Trigger replicator render to apply the new camera transform
            # This is crucial for headless mode - the replicator needs to render
            # a new frame with the updated camera position
            try:
                import omni.replicator.core as rep
                # Step the replicator to render a frame with new camera pose
                rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
            except Exception:
                # Fallback to app.update() if replicator step fails
                try:
                    import omni.kit.app
                    omni.kit.app.get_app().update()
                except Exception:
                    pass

    def _update_offscreen_camera_pose(self):
        """Update the offscreen camera's pose in the scene."""
        if self._camera_eye is None or self._camera_lookat is None:
            return

        # Need either replicator camera or Isaac Lab camera
        if self._camera_prim is None and self._offscreen_camera is None:
            return

        try:
            from pxr import UsdGeom, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(self.cfg.headless_prim_path)

            if not camera_prim.IsValid():
                return

            # Calculate camera orientation from eye and lookat
            eye = self._camera_eye
            lookat = self._camera_lookat

            # Calculate forward direction (camera looks along -Z in camera space)
            forward = lookat - eye
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            # Calculate right vector (cross product of forward and world up)
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)

            if right_norm < 1e-6:
                # Looking straight up or down, use different up vector
                world_up = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, world_up)
                right_norm = np.linalg.norm(right)

            right = right / (right_norm + 1e-8)

            # Calculate actual up vector
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)

            # Build rotation matrix (camera convention: -Z forward, Y up, X right)
            rot_matrix = np.eye(3)
            rot_matrix[:, 0] = right      # X axis
            rot_matrix[:, 1] = up         # Y axis
            rot_matrix[:, 2] = -forward   # Z axis (negative because camera looks along -Z)

            # Convert to Gf matrix
            gf_rot = Gf.Matrix3d(
                Gf.Vec3d(float(rot_matrix[0, 0]), float(rot_matrix[1, 0]), float(rot_matrix[2, 0])),
                Gf.Vec3d(float(rot_matrix[0, 1]), float(rot_matrix[1, 1]), float(rot_matrix[2, 1])),
                Gf.Vec3d(float(rot_matrix[0, 2]), float(rot_matrix[1, 2]), float(rot_matrix[2, 2])),
            )

            # Create transform
            gf_transform = Gf.Matrix4d()
            gf_transform.SetRotate(gf_rot)
            gf_transform.SetTranslateOnly(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))

            # Apply to camera prim - reuse existing xform op if possible
            xformable = UsdGeom.Xformable(camera_prim)
            xform_ops = xformable.GetOrderedXformOps()

            if xform_ops:
                # Reuse existing transform op
                xform_ops[0].Set(gf_transform)
            else:
                # Create new transform op only on first call
                xformable.ClearXformOpOrder()
                xform_op = xformable.AddTransformOp()
                xform_op.Set(gf_transform)

        except Exception as e:
            # Silently ignore transform errors during recording
            pass

    def capture_frame(self, env: DirectRLEnv):
        """Capture a single frame from the environment.

        Args:
            env: The environment to capture from.
        """
        if not self._recording:
            return

        # Get frame from camera sensor or viewport
        frame = self._get_frame(env)

        if frame is None:
            if self._frame_count == 0:
                print("[VideoRecorder] WARNING: First frame capture returned None")
            return

        # Re-initialize writer with actual frame size on first frame
        if not self._initialized:
            print(f"[VideoRecorder] First frame captured: shape={frame.shape}, dtype={frame.dtype}")
            self._reinitialize_writer(frame.shape[1], frame.shape[0])
            self._initialized = True

        # Resize frame if needed
        frame = self._resize_frame(frame)

        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            frame_bgr = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Write frame
        self._writer.write(frame_bgr)
        self._frame_count += 1

        # Log progress periodically
        if self._frame_count == 1:
            print(f"[VideoRecorder] Recording started successfully")
        elif self._frame_count % 300 == 0:
            print(f"[VideoRecorder] Recorded {self._frame_count} frames...")

    def _get_frame(self, env: DirectRLEnv) -> np.ndarray | None:
        """Get a frame from camera sensor or viewport.

        Args:
            env: The environment.

        Returns:
            Frame as numpy array (H, W, C) in RGB format, or None if unavailable.
        """
        # Try replicator-based capture first (most reliable for headless)
        if self.cfg.headless and self._rgb_annotator is not None:
            frame = self._get_replicator_frame(env)
            if frame is not None:
                return frame

        # Try Isaac Lab offscreen camera
        if self.cfg.headless and self._offscreen_camera is not None:
            frame = self._get_offscreen_frame(env)
            if frame is not None:
                return frame

        # Try camera sensor if specified
        if self.cfg.camera_name is not None:
            frame = self._get_camera_frame(env)
            if frame is not None:
                return frame

        # Fall back to viewport capture
        return self._get_viewport_frame(env)

    def _get_replicator_frame(self, env: DirectRLEnv) -> np.ndarray | None:
        """Get frame from the replicator annotator.

        Args:
            env: The environment.

        Returns:
            Frame as numpy array or None.
        """
        try:
            if self._rgb_annotator is None:
                return None

            # Get RGB data from annotator
            # Note: app.update() was already called in set_camera_pose() to propagate transforms
            rgb_data = self._rgb_annotator.get_data()

            if rgb_data is None:
                return None

            # Convert to numpy array
            frame = np.array(rgb_data)

            # Handle different formats
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            elif frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Handle RGBA by taking only RGB channels
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[..., :3]

            return frame

        except Exception as e:
            # Only print error once per 100 frames to avoid spam
            if self._frame_count % 100 == 0:
                print(f"[VideoRecorder] Error getting replicator frame: {e}")
            return None

    def _get_offscreen_frame(self, env: DirectRLEnv) -> np.ndarray | None:
        """Get frame from the offscreen camera.

        Args:
            env: The environment.

        Returns:
            Frame as numpy array or None.
        """
        try:
            if self._offscreen_camera is None:
                return None

            # Update the camera sensor with environment index
            self._offscreen_camera.update(dt=env.step_dt)

            # Get RGB data from camera
            if not hasattr(self._offscreen_camera, "data"):
                return None

            # Access RGB output - handle different Isaac Lab versions
            rgb_data = None
            if hasattr(self._offscreen_camera.data, "output"):
                rgb_data = self._offscreen_camera.data.output.get("rgb")
            elif hasattr(self._offscreen_camera.data, "rgb"):
                rgb_data = self._offscreen_camera.data.rgb

            if rgb_data is None:
                return None

            # Use first environment's data (offscreen camera is single instance)
            if len(rgb_data.shape) == 4:
                rgb_frame = rgb_data[0]  # [H, W, C]
            else:
                rgb_frame = rgb_data

            # Convert to numpy
            if isinstance(rgb_frame, torch.Tensor):
                rgb_frame = rgb_frame.detach().cpu().numpy()

            # Handle different formats
            if rgb_frame.dtype == np.float32 or rgb_frame.dtype == np.float64:
                frame = (rgb_frame * 255).astype(np.uint8)
            else:
                frame = rgb_frame.astype(np.uint8)

            # Handle RGBA by taking only RGB channels
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[..., :3]

            return frame

        except Exception as e:
            # Only print error once per 100 frames to avoid spam
            if self._frame_count % 100 == 0:
                print(f"[VideoRecorder] Error getting offscreen frame: {e}")
            return None

    def _get_camera_frame(self, env: DirectRLEnv) -> np.ndarray | None:
        """Get frame from a camera sensor.

        Args:
            env: The environment.

        Returns:
            Frame as numpy array or None.
        """
        try:
            if not hasattr(env, "scene"):
                return None

            camera = getattr(env.scene, self.cfg.camera_name, None)
            if camera is None:
                return None

            if not hasattr(camera, "data") or not hasattr(camera.data, "output"):
                return None

            camera_output = camera.data.output
            if "rgb" not in camera_output:
                return None

            # Get RGB data for specified environment
            rgb_data = camera_output["rgb"][self.cfg.env_index]

            # Convert to numpy
            if isinstance(rgb_data, torch.Tensor):
                rgb_data = rgb_data.detach().cpu().numpy()

            # Handle different formats
            if rgb_data.dtype == np.float32 or rgb_data.dtype == np.float64:
                # Normalize from [0, 1] to [0, 255]
                frame = (rgb_data * 255).astype(np.uint8)
            else:
                frame = rgb_data.astype(np.uint8)

            # Handle RGBA by taking only RGB channels
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[..., :3]

            return frame

        except Exception as e:
            print(f"[VideoRecorder] Error getting camera frame: {e}")
            return None

    def _get_viewport_frame(self, env: DirectRLEnv) -> np.ndarray | None:
        """Get frame from the viewport.

        This uses Isaac Sim's viewport capture API.

        Args:
            env: The environment.

        Returns:
            Frame as numpy array or None.
        """
        try:
            # Import viewport utilities from omniverse
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                return None

            # Try different capture methods based on Isaac Sim version
            frame = None

            # Method 1: Try capture_viewport_to_file and read back (most reliable)
            try:
                import tempfile
                import os
                from omni.kit.viewport.utility import capture_viewport_to_file

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name

                # Capture to temp file
                capture_viewport_to_file(viewport, tmp_path)

                # Read back
                if os.path.exists(tmp_path):
                    frame = cv2.imread(tmp_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    os.unlink(tmp_path)

                if frame is not None:
                    return frame
            except Exception:
                pass

            # Method 2: Use viewport's render product
            try:
                render_product_path = viewport.get_render_product_path()
                if render_product_path:
                    import omni.replicator.core as rep

                    # Create annotator for RGB capture
                    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                    rgb_annot.attach([render_product_path])

                    # Get data
                    rgb_data = rgb_annot.get_data()
                    if rgb_data is not None:
                        frame = np.array(rgb_data)
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        return frame
            except Exception:
                pass

            return None

        except ImportError:
            # Viewport utilities not available (headless mode)
            return None
        except Exception as e:
            # Only print error once per 100 frames
            if self._frame_count % 100 == 0:
                print(f"[VideoRecorder] Error capturing viewport: {e}")
            return None

    def _reinitialize_writer(self, width: int, height: int):
        """Re-initialize video writer with actual frame dimensions.

        Args:
            width: Frame width.
            height: Frame height.
        """
        if self._writer is not None:
            self._writer.release()

        output_w, output_h = self.cfg.resolution
        output_path = Path(self.cfg.output_path)
        output_ext = output_path.suffix.lower()

        # Use same codec selection as start()
        if output_ext == ".mp4":
            codecs_to_try = ["avc1", "H264", "X264", "mp4v"]
        elif output_ext == ".avi":
            codecs_to_try = ["XVID", "MJPG", "DIVX"]
        else:
            codecs_to_try = [self.cfg.codec]

        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self._writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.cfg.fps,
                    (output_w, output_h),
                )
                if self._writer.isOpened():
                    break
                else:
                    self._writer.release()
                    self._writer = None
            except Exception:
                continue

        if self._writer is None:
            # Fallback
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.cfg.fps,
                (output_w, output_h),
            )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target resolution.

        Args:
            frame: Input frame (H, W, C).

        Returns:
            Resized frame (H, W, 3) - always RGB.
        """
        # Ensure RGB (3 channels) - convert RGBA if needed
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]

        target_w, target_h = self.cfg.resolution
        h, w = frame.shape[:2]

        if w == target_w and h == target_h:
            return frame

        mode = self.cfg.resize_mode

        if mode == "stretch":
            # Simple resize
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        elif mode == "fit":
            # Fit frame into target while maintaining aspect ratio, pad with black
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create black canvas and center the frame (match channels of resized frame)
            num_channels = resized.shape[2] if len(resized.shape) == 3 else 1
            canvas = np.zeros((target_h, target_w, num_channels), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

            return canvas

        elif mode == "crop":
            # Center crop to target aspect ratio, then resize
            target_aspect = target_w / target_h
            current_aspect = w / h

            if current_aspect > target_aspect:
                # Too wide, crop width
                new_w = int(h * target_aspect)
                x_start = (w - new_w) // 2
                frame = frame[:, x_start : x_start + new_w]
            else:
                # Too tall, crop height
                new_h = int(w / target_aspect)
                y_start = (h - new_h) // 2
                frame = frame[y_start : y_start + new_h, :]

            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        else:
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def __del__(self):
        """Destructor to ensure video is finalized."""
        if self._recording:
            self.stop()
