#!/usr/bin/env python3
"""Reactive perception/VSLAM demo navigator for factory_bot.

Inputs:
- /camera/depth/image_raw for obstacle sectors
- /semantic_spatial/visual_odometry if available, otherwise /odom for pose
- /semantic_spatial/diagnostics and /semantic_spatial/objects for stack logs

Output:
- ROS velocity commands on /factory_bot/cmd_vel, bridged to Gazebo Sim

This is a live demo navigator: it knows the target coordinate, but it does not
know obstacle locations ahead of time. It uses depth perception to slow down,
turn away from obstacles, and speed up when the path is clear.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

from rclpy.executors import ExternalShutdownException

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

ROS_CMD_TOPIC = "/factory_bot/cmd_vel"
TARGET_X = 14.0
TARGET_Y = 10.2


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    source: str
    stamp: float


class FactoryBotStackNavigator(Node):
    def __init__(self):
        super().__init__("factory_bot_stack_navigator")
        self.depth = None
        self.depth_encoding = ""
        self.rgb = None
        self.rgb_encoding = ""
        self.target_marker_seen = False
        self.pose: Optional[Pose2D] = None
        self.last_diag = ""
        self.last_objects = ""
        self.last_log = 0.0
        self.last_depth_stamp = 0.0
        self.last_pose_stamp = 0.0
        self.integral_heading = 0.0
        self.prev_heading_error = 0.0
        self.detour_sign = 1.0
        self.arrived = False

        self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_rgb, 10)
        self.create_subscription(Odometry, "/semantic_spatial/visual_odometry", self.on_vo, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/semantic_spatial/diagnostics", self.on_diag, 10)
        self.create_subscription(String, "/semantic_spatial/objects", self.on_objects, 10)
        self.cmd_pub = self.create_publisher(Twist, ROS_CMD_TOPIC, 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info("Factory bot stack navigator waiting for depth + pose/VSLAM")

    def on_depth(self, msg: Image) -> None:
        self.depth_encoding = msg.encoding
        self.depth = image_to_depth_m(msg)
        self.last_depth_stamp = self.get_clock().now().nanoseconds * 1e-9

    def on_rgb(self, msg: Image) -> None:
        self.rgb_encoding = msg.encoding
        self.rgb = image_to_rgb(msg)
        self.target_marker_seen = green_target_visible(self.rgb)

    def on_vo(self, msg: Odometry) -> None:
        self.pose = odom_to_pose2d(msg, "vslam")
        self.last_pose_stamp = self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg: Odometry) -> None:
        if self.pose is None or self.pose.source != "vslam":
            self.pose = odom_to_pose2d(msg, "odom")
            self.last_pose_stamp = self.get_clock().now().nanoseconds * 1e-9

    def on_diag(self, msg: String) -> None:
        self.last_diag = msg.data

    def on_objects(self, msg: String) -> None:
        self.last_objects = msg.data

    def tick(self) -> None:
        if self.arrived:
            self.publish_cmd(0.0, 0.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.depth is None:
            self.log_waiting("depth")
            self.publish_cmd(0.0, 0.0)
            return
        if now - self.last_depth_stamp > 1.0:
            self.log_waiting("fresh depth")
            self.publish_cmd(0.0, 0.0)
            return
        if self.pose is None:
            self.log_waiting("pose/VSLAM")
            self.publish_cmd(0.0, 0.0)
            return
        if now - self.last_pose_stamp > 1.0:
            self.log_waiting("fresh pose/VSLAM")
            self.publish_cmd(0.0, 0.0)
            return

        sectors = depth_sectors(self.depth)
        dx = TARGET_X - self.pose.x
        dy = TARGET_Y - self.pose.y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        heading_error = wrap_angle(target_yaw - self.pose.yaw)

        if distance < 0.35 or (distance < 0.85 and self.target_marker_seen):
            self.arrived = True
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info(
                f"ARRIVED target=({TARGET_X:.1f},{TARGET_Y:.1f}) pose=({self.pose.x:.2f},{self.pose.y:.2f}) "
                f"target_marker_seen={self.target_marker_seen}"
            )
            return

        front = sectors["front"]
        left = sectors["left"]
        right = sectors["right"]
        near = min(front, left, right)

        mode = "clear_path"
        if front < 0.75:
            if abs(left - right) > 0.15:
                self.detour_sign = 1.0 if left > right else -1.0
            elif abs(heading_error) > 0.12:
                self.detour_sign = 1.0 if heading_error > 0.0 else -1.0
            linear = 0.05
            obstacle_bias = 0.90 * self.detour_sign
            mode = "blocked_turning"
        elif front < 1.25:
            if abs(left - right) > 0.12:
                self.detour_sign = 1.0 if left > right else -1.0
            linear = 0.12
            obstacle_bias = 0.48 * self.detour_sign
            mode = "slow_near_obstacle"
        elif near < 1.0:
            linear = 0.14
            obstacle_bias = -0.28 if left < right else 0.28
            mode = "side_obstacle"
        elif front > 3.0 and abs(heading_error) < 0.45:
            linear = 0.38 if not self.target_marker_seen else 0.30
            obstacle_bias = 0.0
            mode = "open_path_speedup"
        else:
            linear = 0.22
            obstacle_bias = 0.0

        if abs(heading_error) > 1.2:
            linear = min(linear, 0.08)

        self.integral_heading = float(np.clip(self.integral_heading + heading_error * 0.2, -1.0, 1.0))
        derivative = (heading_error - self.prev_heading_error) / 0.2
        self.prev_heading_error = heading_error
        angular = 0.85 * heading_error + 0.02 * self.integral_heading + 0.035 * derivative + obstacle_bias
        angular = float(np.clip(angular, -0.95, 0.95))
        if abs(angular) > 0.80:
            linear = min(linear, 0.12)
        elif abs(angular) > 0.55:
            linear = min(linear, 0.22)
        linear = float(np.clip(linear, 0.0, 0.38))

        self.publish_cmd(linear, angular)
        if now - self.last_log > 0.8:
            self.last_log = now
            diag_summary = summarize_json(self.last_diag)
            object_summary = summarize_objects(self.last_objects)
            self.get_logger().info(
                "perception "
                f"mode={mode} depth_encoding={self.depth_encoding} "
                f"front={front:.2f}m left={left:.2f}m right={right:.2f}m "
                f"goal_dist={distance:.2f}m heading_err={heading_error:.2f}rad "
                f"cmd_v={linear:.2f} cmd_w={angular:.2f} pose_source={self.pose.source} "
                f"target_marker_seen={self.target_marker_seen} rgb_encoding={self.rgb_encoding or 'none'} "
                f"diag={diag_summary} objects={object_summary}"
            )


    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def log_waiting(self, missing: str) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log > 1.0:
            self.last_log = now
            self.get_logger().info(f"waiting_for={missing}; navigator will not move without live perception + pose")


def image_to_depth_m(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    if enc == "32fc1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    elif enc in {"16uc1", "mono16"}:
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 0.001
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")
    arr[~np.isfinite(arr)] = np.nan
    return arr


def image_to_rgb(msg: Image) -> Optional[np.ndarray]:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    if enc in {"rgb8", "bgr8"}:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        return arr
    if enc in {"rgba8", "bgra8"}:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4).copy()[:, :, :3]
        if enc == "bgra8":
            arr = arr[:, :, ::-1]
        return arr
    return None


def green_target_visible(rgb: Optional[np.ndarray]) -> bool:
    if rgb is None:
        return False
    h, w = rgb.shape[:2]
    roi = rgb[int(h * 0.18):int(h * 0.75), int(w * 0.25):int(w * 0.75), :].astype(np.int16)
    if roi.size == 0:
        return False
    r = roi[:, :, 0]
    g = roi[:, :, 1]
    b = roi[:, :, 2]
    green = (g > 95) & (g > r * 1.35) & (g > b * 1.25)
    return float(np.count_nonzero(green)) / float(green.size) > 0.012


def depth_sectors(depth: np.ndarray) -> dict[str, float]:
    h, w = depth.shape[:2]
    # Use the upper-middle image band so the floor and robot chassis do not dominate near-range readings.
    y0, y1 = int(h * 0.18), int(h * 0.52)
    bands = {
        "left": depth[y0:y1, int(w * 0.05):int(w * 0.35)],
        "front": depth[y0:y1, int(w * 0.35):int(w * 0.65)],
        "right": depth[y0:y1, int(w * 0.65):int(w * 0.95)],
    }
    out = {}
    for name, band in bands.items():
        valid = band[np.isfinite(band) & (band > 0.12)]
        out[name] = float(np.percentile(valid, 35.0)) if valid.size else 8.0
    return out


def odom_to_pose2d(msg: Odometry, source: str) -> Pose2D:
    q = msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return Pose2D(
        x=float(msg.pose.pose.position.x),
        y=float(msg.pose.pose.position.y),
        yaw=yaw,
        source=source,
        stamp=float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
    )


def summarize_json(raw: str) -> str:
    if not raw:
        return "none"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "unparsed"
    parts = []
    for key in ["health", "tracking_ok", "camera_info_ready", "map_points", "semantic_objects", "last_error"]:
        if key in data:
            parts.append(f"{key}={data[key]}")
    return ",".join(parts) if parts else "ok"


def summarize_objects(raw: str) -> str:
    if not raw:
        return "none"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "unparsed"
    if not isinstance(data, list):
        return "none"
    return f"count={len(data)}"


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def main() -> None:
    rclpy.init()
    node = FactoryBotStackNavigator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_cmd(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
