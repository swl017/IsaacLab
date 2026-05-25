#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""EKF2 state-lag recorder — ticket 041.

Records Pegasus ground-truth state topics and PX4 EKF2 estimates (via MAVROS)
into per-topic CSVs for a single maneuver. The recorder also drives the
vehicle through the maneuver by publishing velocity setpoints on
``/<ns>/mavros/setpoint_velocity/cmd_vel`` (ENU local frame) and managing
the OFFBOARD ↔ AUTO.LOITER mode switch via ``/<ns>/mavros/set_mode``.

Per-maneuver invocation (single trial). Loop over trials externally in
``run_measurement.sh``.

Usage::

    python3 ekf_lag_recorder.py \\
        --maneuver vel_step_x \\
        --trial 0 \\
        --duration 30 \\
        --namespace px4_1 \\
        --out-dir <abs path>/raw

Preconditions (see ekf_state_lag_spec.md §3):
  * Pegasus SITL with PX4 lockstep enabled is already running.
  * MAVROS bridge is up and publishing /<ns>/mavros/local_position/pose,
    /<ns>/mavros/local_position/velocity_local, /<ns>/mavros/local_position/velocity_body,
    /<ns>/mavros/imu/data.
  * The vehicle is armed and in AUTO.LOITER (in-air, hovering). The recorder
    transitions to OFFBOARD for the maneuver and back to AUTO.LOITER on exit.
  * No other node is publishing setpoints to <ns> (would conflict with the
    recorder's velocity stream).
  * /clock is being published (sim-time on every node).

Maneuver commands are interpreted by PX4 as ENU local-frame velocities (the
standard mavros/setpoint_velocity/cmd_vel convention), not body-frame. Lag
measurement is frame-independent so this does not affect channel fits.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time as _wall
from dataclasses import dataclass
from typing import Callable, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
from sensor_msgs.msg import Imu

try:
    from mavros_msgs.srv import SetMode
    _HAS_MAVROS_MSGS = True
except ImportError:
    _HAS_MAVROS_MSGS = False


# ----------------------------------------------------------------------
# Maneuver command profiles — each returns (linear_xyz, angular_xyz) at time t
# ----------------------------------------------------------------------

def _vel_step_x(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    block = int(t // 5.0) % 4
    vx = {0: 1.0, 1: 0.0, 2: -1.0, 3: 0.0}[block]
    return (vx, 0.0, 0.0), (0.0, 0.0, 0.0)


def _vel_step_z(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    block = int(t // 5.0) % 4
    vz = {0: 0.5, 1: 0.0, 2: -0.5, 3: 0.0}[block]
    return (0.0, 0.0, vz), (0.0, 0.0, 0.0)


def _yaw_step(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # +1 rad/s for 0.5s, hold 5s, -1 rad/s for 0.5s, hold 5s, repeat.
    period = 11.0
    tau = t % period
    if 0.0 <= tau < 0.5:
        wz = +1.0
    elif 5.5 <= tau < 6.0:
        wz = -1.0
    else:
        wz = 0.0
    return (0.0, 0.0, 0.0), (0.0, 0.0, wz)


def _chirp(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # Linear chirp 0.1 -> 3.0 Hz over 30 s on linear.x amplitude 0.5 m/s.
    f0, f1, T = 0.1, 3.0, 30.0
    f_t = f0 + (f1 - f0) * min(t / T, 1.0)
    vx = 0.5 * math.sin(2.0 * math.pi * f_t * t)
    return (vx, 0.0, 0.0), (0.0, 0.0, 0.0)


def _vel_impulse_recovery(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # 2 Hz square wave on linear.x for first 5 s, then 25 s hover.
    if t < 5.0:
        cycle = int(t * 4) % 2  # 2 Hz square: 250 ms high, 250 ms low
        vx = 0.5 if cycle == 0 else -0.5
    else:
        vx = 0.0
    return (vx, 0.0, 0.0), (0.0, 0.0, 0.0)


def _hover_drift(t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)


MANEUVERS: dict[str, Callable[[float], tuple]] = {
    "hover_drift": _hover_drift,
    "vel_step_x": _vel_step_x,
    "vel_step_z": _vel_step_z,
    "yaw_step": _yaw_step,
    "chirp": _chirp,
    "vel_impulse_recovery": _vel_impulse_recovery,
}


# ----------------------------------------------------------------------
# CSV writer table — keyed by topic slug
# ----------------------------------------------------------------------

@dataclass
class TopicSpec:
    slug: str
    topic_suffix: str
    msg_type: type
    columns: List[str]
    extract: Callable[[object], List[float]]


def _ext_pose(msg: PoseStamped) -> List[float]:
    p = msg.pose.position
    q = msg.pose.orientation
    return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]


def _ext_twist_full(msg: TwistStamped) -> List[float]:
    l = msg.twist.linear
    a = msg.twist.angular
    return [l.x, l.y, l.z, a.x, a.y, a.z]


def _ext_twist_linear_only(msg: TwistStamped) -> List[float]:
    l = msg.twist.linear
    return [l.x, l.y, l.z]


def _ext_accel(msg: AccelStamped) -> List[float]:
    a = msg.accel.linear
    return [a.x, a.y, a.z]


def _ext_imu(msg: Imu) -> List[float]:
    q = msg.orientation
    w = msg.angular_velocity
    a = msg.linear_acceleration
    return [q.x, q.y, q.z, q.w, w.x, w.y, w.z, a.x, a.y, a.z]


def build_topic_table() -> List[TopicSpec]:
    return [
        TopicSpec("state_pose", "state/pose", PoseStamped,
                  ["px", "py", "pz", "qx", "qy", "qz", "qw"], _ext_pose),
        TopicSpec("state_twist", "state/twist", TwistStamped,
                  ["vx_body", "vy_body", "vz_body", "wx_body", "wy_body", "wz_body"], _ext_twist_full),
        TopicSpec("state_twist_inertial", "state/twist_inertial", TwistStamped,
                  ["vx_world", "vy_world", "vz_world"], _ext_twist_linear_only),
        TopicSpec("state_accel", "state/accel", AccelStamped,
                  ["ax_world", "ay_world", "az_world"], _ext_accel),
        TopicSpec("mavros_local_position_pose", "mavros/local_position/pose", PoseStamped,
                  ["px", "py", "pz", "qx", "qy", "qz", "qw"], _ext_pose),
        TopicSpec("mavros_local_position_velocity_local", "mavros/local_position/velocity_local", TwistStamped,
                  ["vx_world", "vy_world", "vz_world", "wx_world", "wy_world", "wz_world"], _ext_twist_full),
        TopicSpec("mavros_local_position_velocity_body", "mavros/local_position/velocity_body", TwistStamped,
                  ["vx_body", "vy_body", "vz_body", "wx_body", "wy_body", "wz_body"], _ext_twist_full),
        TopicSpec("mavros_imu_data", "mavros/imu/data", Imu,
                  ["qx", "qy", "qz", "qw", "wx_body", "wy_body", "wz_body", "ax_body", "ay_body", "az_body"], _ext_imu),
    ]


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

class EkfLagRecorder(Node):
    """Records all configured truth + EKF topics; drives cmd_vel for the maneuver.

    One instance per (maneuver, trial). The node tears itself down after
    ``duration_s`` of wall-time elapsed since ``start_t0`` (sim-time).
    """

    def __init__(self, maneuver: str, trial: int, duration_s: float,
                 namespace: str, out_dir: str, cmd_rate_hz: float = 100.0,
                 pre_stream_s: float = 2.0, manage_mode: bool = True,
                 idle_mode: str = "AUTO.LOITER"):
        super().__init__(f"ekf_lag_recorder_{maneuver}_t{trial}")

        # Force use_sim_time on. The launching shell should also pass
        # --ros-args -p use_sim_time:=true, but we set it explicitly here so
        # a forgotten flag doesn't silently produce wall-clock recordings.
        from rclpy.parameter import Parameter
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        if maneuver not in MANEUVERS:
            raise ValueError(f"Unknown maneuver {maneuver!r}; valid: {list(MANEUVERS)}")
        self._maneuver_fn = MANEUVERS[maneuver]
        self._maneuver = maneuver
        self._trial = trial
        self._duration_s = float(duration_s)
        self._ns = namespace.strip("/")
        self._out_dir = out_dir
        os.makedirs(self._out_dir, exist_ok=True)

        # Sim-time t0 captured on first /clock tick (or first message); marker for cmd profile timing.
        self._t0: Optional[float] = None
        self._t0_lock = threading.Lock()

        # CSV writers — one file per (maneuver, trial, topic).
        self._csv_files = {}
        self._csv_writers = {}
        self._csv_lock = threading.Lock()
        topic_table = build_topic_table()
        for spec in topic_table:
            path = os.path.join(out_dir, f"{maneuver}_trial{trial}_{spec.slug}.csv")
            fp = open(path, "w", newline="")
            wr = csv.writer(fp)
            wr.writerow(["stamp_s", "recv_stamp_s"] + spec.columns)
            self._csv_files[spec.slug] = fp
            self._csv_writers[spec.slug] = wr

        # Subscribers — bind a closure per topic for the callback.
        self._subs = []
        for spec in topic_table:
            topic = f"/{self._ns}/{spec.topic_suffix}"
            cb = self._make_callback(spec)
            sub = self.create_subscription(spec.msg_type, topic, cb, qos_profile_sensor_data)
            self._subs.append(sub)
            self.get_logger().info(f"subscribing: {topic}")

        # Velocity setpoint publisher — standard MAVROS path. ENU local frame.
        self._cmd_pub = self.create_publisher(
            TwistStamped, f"/{self._ns}/mavros/setpoint_velocity/cmd_vel", 10
        )
        self._cmd_period_s = 1.0 / float(cmd_rate_hz)
        self._cmd_timer = self.create_timer(self._cmd_period_s, self._publish_cmd)

        # OFFBOARD-mode handshake state.
        self._pre_stream_s = float(pre_stream_s)
        self._manage_mode = bool(manage_mode) and _HAS_MAVROS_MSGS
        self._idle_mode = str(idle_mode)
        self._mode_t0_sim = None    # sim-time at which we started pre-streaming zeros
        self._maneuver_t0_sim = None  # sim-time at which the maneuver clock starts
        self._offboard_engaged = False
        self._idle_future = None    # set when LOITER is requested at teardown
        self._mode_client = None
        if self._manage_mode:
            self._mode_client = self.create_client(SetMode, f"/{self._ns}/mavros/set_mode")
        elif not _HAS_MAVROS_MSGS:
            self.get_logger().warn("mavros_msgs not importable — skipping mode-switch management")

        # Shutdown timer — polls sim-time at 10 Hz; tears down when t - t0 >= duration_s.
        self._shutdown_timer = self.create_timer(0.1, self._maybe_shutdown)
        self._done = False
        self._shutdown_in_progress = False

        self.get_logger().info(
            f"EkfLagRecorder ready: maneuver={maneuver} trial={trial} duration={duration_s}s "
            f"namespace={self._ns} pre_stream={self._pre_stream_s}s manage_mode={self._manage_mode} "
            f"out={out_dir}"
        )

    # --- helpers -------------------------------------------------------

    def _now_s(self) -> float:
        t = self.get_clock().now().nanoseconds * 1e-9
        return t

    def _make_callback(self, spec: TopicSpec):
        slug = spec.slug
        extract = spec.extract

        def _cb(msg):
            stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            recv_s = self._now_s()
            # Latch t0 on first incoming message (any topic) — sim-time clock
            # may take a moment to warm up after launch.
            with self._t0_lock:
                if self._t0 is None:
                    self._t0 = recv_s
            row = [stamp_s, recv_s, *extract(msg)]
            with self._csv_lock:
                self._csv_writers[slug].writerow(row)

        return _cb

    def _publish_cmd(self):
        # Pre-stream phase: publish zeros (needed before PX4 will accept OFFBOARD).
        # Maneuver phase: publish the maneuver profile.
        # Post-maneuver: publish zeros (safe parking before mode switch back to LOITER).
        with self._t0_lock:
            t0 = self._t0
        if t0 is None:
            # Time-base not yet live (no /clock tick observed). Don't publish anything.
            return

        # Latch pre-stream start on the first publish tick after t0 is set.
        if self._mode_t0_sim is None:
            self._mode_t0_sim = self._now_s()

        sim_now = self._now_s()
        in_pre_stream = (sim_now - self._mode_t0_sim) < self._pre_stream_s
        maneuver_started = self._maneuver_t0_sim is not None

        if not maneuver_started and not in_pre_stream:
            # Pre-stream complete: engage OFFBOARD (if managing mode) and start maneuver clock.
            if self._manage_mode and not self._offboard_engaged:
                self._engage_mode("OFFBOARD")
                self._offboard_engaged = True
                self.get_logger().info("OFFBOARD requested")
            self._maneuver_t0_sim = sim_now
            self.get_logger().info(f"maneuver clock t0 = {sim_now:.3f} sim-s")

        if maneuver_started:
            t = sim_now - self._maneuver_t0_sim
            if 0.0 <= t < self._duration_s:
                lin, ang = self._maneuver_fn(t)
            else:
                lin, ang = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        else:
            lin, ang = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # ENU local frame (mavros convention)
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = lin
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = ang
        self._cmd_pub.publish(msg)

    def _maybe_shutdown(self):
        # Runs inside spin_once — set the done flag and let main() do the actual
        # teardown after the executor returns. Calling shutdown() / destroy_node()
        # from within a callback races the executor's wait set.
        if self._done or self._shutdown_in_progress:
            return
        if self._maneuver_t0_sim is None:
            return
        elapsed = self._now_s() - self._maneuver_t0_sim
        if elapsed >= self._duration_s:
            self._shutdown_in_progress = True
            self.get_logger().info(
                f"maneuver {self._maneuver} trial {self._trial}: elapsed {elapsed:.2f}s; "
                f"switching back to {self._idle_mode}"
            )
            # Stop publishing OFFBOARD setpoints first — PX4 prioritises an
            # active setpoint stream over a mode-switch request, so leaving
            # the timer alive would keep the vehicle in OFFBOARD even after
            # set_mode(LOITER) succeeds.
            try:
                self._cmd_timer.cancel()
            except Exception:
                pass
            # Then request the mode switch and stash the future for main() to
            # block on before tearing down rclpy.
            self._idle_future = self._engage_mode(self._idle_mode) if self._manage_mode else None
            self._done = True

    def _engage_mode(self, custom_mode: str):
        """Returns the in-flight future, or None on failure."""
        if self._mode_client is None:
            return None
        if not self._mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"set_mode service not available — leaving mode as-is (wanted {custom_mode})"
            )
            return None
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = custom_mode
        return self._mode_client.call_async(req)

    def shutdown(self):
        # Cancel timers, close CSV files, drop subscribers. The launcher will
        # then rclpy.shutdown() and exit.
        try:
            self._cmd_timer.cancel()
        except Exception:
            pass
        try:
            self._shutdown_timer.cancel()
        except Exception:
            pass
        with self._csv_lock:
            for fp in self._csv_files.values():
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    # Strip ROS args (e.g. `--ros-args -p use_sim_time:=true`) before argparse so
    # the orchestrator can pass them through without confusing argparse.
    from rclpy.utilities import remove_ros_args
    raw_args = sys.argv[1:] if argv is None else argv
    argv_clean = remove_ros_args(args=raw_args) if raw_args else []
    parser = argparse.ArgumentParser(description="EKF2 state-lag recorder")
    parser.add_argument("--maneuver", required=True, choices=list(MANEUVERS),
                        help="Maneuver name (see ekf_state_lag_spec.md §5).")
    parser.add_argument("--trial", type=int, default=0, help="Trial index (stochastic averaging).")
    parser.add_argument("--duration", type=float, default=30.0, help="Maneuver duration in seconds.")
    parser.add_argument("--namespace", type=str, default="px4_1",
                        help="Vehicle namespace (matches offboard_py / pegasus).")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Absolute path to raw/ output directory.")
    parser.add_argument("--cmd-rate-hz", type=float, default=100.0,
                        help="cmd_vel publish rate.")
    parser.add_argument("--startup-wait-s", type=float, default=1.0,
                        help="Extra wait after node init before starting the maneuver clock.")
    parser.add_argument("--pre-stream-s", type=float, default=2.0,
                        help="Seconds of zero-velocity setpoint streaming before OFFBOARD engage.")
    parser.add_argument("--no-mode-switch", action="store_true",
                        help="Skip OFFBOARD/LOITER mode-switch service calls (use when an external"
                             " supervisor is handling mode).")
    parser.add_argument("--idle-mode", type=str, default="AUTO.LOITER",
                        help="Mode to return to after the maneuver (PX4 custom_mode string).")
    args = parser.parse_args(argv_clean)

    rclpy.init()
    node = EkfLagRecorder(
        maneuver=args.maneuver,
        trial=args.trial,
        duration_s=args.duration,
        namespace=args.namespace,
        out_dir=os.path.abspath(args.out_dir),
        cmd_rate_hz=args.cmd_rate_hz,
        pre_stream_s=args.pre_stream_s,
        manage_mode=not args.no_mode_switch,
        idle_mode=args.idle_mode,
    )

    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)

        # Spin until the LOITER future completes (PX4 needs the response to
        # confirm the mode switch took effect). Keep the cmd_vel timer alive
        # so the OFFBOARD setpoint stream doesn't go stale during the switch.
        if node._idle_future is not None:
            t_deadline = _wall.time() + 5.0
            while rclpy.ok() and not node._idle_future.done() and _wall.time() < t_deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node._idle_future.done():
                resp = node._idle_future.result()
                node.get_logger().info(
                    f"set_mode({node._idle_mode}) -> mode_sent={getattr(resp, 'mode_sent', '?')}"
                )
            else:
                node.get_logger().warn("set_mode response timed out — vehicle may not have switched")
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt — flushing CSVs and exiting")
    finally:
        node.shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
