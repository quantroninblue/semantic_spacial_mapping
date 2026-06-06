from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from runtime.core.config import RuntimeConfig, load_runtime_config
from runtime.core.runtime_logger import RuntimeSessionLogger, default_rosbag_topics
from runtime.core.segmentation import build_segmentation_provider
from runtime.core.semantic_runtime import DeploymentSemanticRuntime
from runtime.core.types import FramePacket

from .converters import image_msg_to_numpy, make_calibration, stamp_to_sec
from .image_sync import ApproximateRgbdSync
from .pose_sources import (
    RosOdometryPoseProvider,
    RosPoseStampedPoseProvider,
    build_pose_provider,
)
from .publishers import RuntimePublishers


class SemanticSpatialNode(Node):
    def __init__(self):
        super().__init__("semantic_spatial_node")
        self.declare_parameter("config_path", "")
        self.config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self.config = self._load_config()
        self.session_logger = RuntimeSessionLogger(
            config=self.config,
            config_path=self.config_path,
        )
        self.frame_id = 0
        self.busy = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_provider = build_pose_provider(self.config, self.tf_buffer)
        self.runtime = DeploymentSemanticRuntime(
            config=self.config,
            pose_provider=self.pose_provider,
            segmentation_module=self._build_segmentation(),
        )
        self.runtime_publishers = RuntimePublishers(self, self.config)
        self.sync = ApproximateRgbdSync(
            queue_size=self.config.sync.queue_size,
            slop_sec=self.config.sync.slop_sec,
            max_fps=self.config.sync.max_fps,
            callback=self._process_synced,
        )
        self._create_subscriptions()
        self.session_logger.log_subscriptions(self._topic_report())
        self.session_logger.start_embedded_rosbag(default_rosbag_topics(self.config))
        self.get_logger().info(
            f"semantic spatial runtime started profile={self.config.profile} "
            f"pose_source={self.config.pose.source} "
            f"log={self.session_logger.report_path}"
        )
        self.session_logger.log_event(
            "node_started",
            profile=self.config.profile,
            pose_source=self.config.pose.source,
        )

    def _load_config(self) -> RuntimeConfig:
        path = self.config_path
        if path:
            return load_runtime_config(Path(path))
        return RuntimeConfig()

    def _build_segmentation(self):
        return build_segmentation_provider(self.config.segmentation)

    def _create_subscriptions(self) -> None:
        qos = 10
        self.create_subscription(
            Image,
            self.config.topics.rgb,
            self.sync.add_rgb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.config.topics.depth,
            self.sync.add_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            self.config.topics.rgb_camera_info,
            self.sync.set_rgb_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            self.config.topics.depth_camera_info,
            self.sync.set_depth_info,
            qos_profile_sensor_data,
        )

        if isinstance(self.pose_provider, RosOdometryPoseProvider):
            self.create_subscription(
                Odometry,
                self.config.topics.odom,
                self.pose_provider.callback,
                qos,
            )
        elif isinstance(self.pose_provider, RosPoseStampedPoseProvider):
            self.create_subscription(
                PoseStamped,
                self.config.topics.pose,
                self.pose_provider.callback,
                qos,
            )

    def _process_synced(self, rgb_msg, depth_msg, rgb_info, depth_info) -> None:
        if self.busy and self.config.sync.drop_when_busy:
            self.runtime.diagnostics.dropped_frames += 1
            return

        self.busy = True
        try:
            rgb = image_msg_to_numpy(rgb_msg)
            depth = image_msg_to_numpy(depth_msg) if depth_msg is not None else None
            packet_time = stamp_to_sec(rgb_msg.header.stamp)
            rgb_info = self._fresh_camera_info(rgb_info, packet_time)
            depth_info = self._fresh_camera_info(depth_info, packet_time)
            self.runtime.diagnostics.camera_info_stale = (
                self.sync.rgb_info is not None
                and rgb_info is None
                or self.sync.depth_info is not None
                and depth_info is None
            )
            calibration = make_calibration(
                rgb_info,
                depth_info,
                default_rgb_frame=self.config.frames.rgb_optical_frame,
                default_depth_frame=self.config.frames.depth_optical_frame,
                T_depth_rgb=self._configured_depth_to_rgb(),
            )
            self.runtime.diagnostics.rgb_camera_info_received = rgb_info is not None
            self.runtime.diagnostics.depth_camera_info_received = depth_info is not None
            self.runtime.diagnostics.sync_failures = self.sync.sync_failures
            self.runtime.diagnostics.dropped_frames += self.sync.dropped_frames
            self.sync.dropped_frames = 0

            packet = FramePacket(
                frame_id=self.frame_id,
                timestamp=packet_time,
                rgb_frame=rgb,
                depth_frame=depth,
                calibration=calibration,
                source=self.config.profile,
                metadata={"rgb_frame_id": rgb_msg.header.frame_id},
            )
            output = self.runtime.update(packet)
            self.runtime_publishers.publish(
                output=output,
                latest_map=self.runtime.latest_map(),
                rgb_frame_id=rgb_msg.header.frame_id or self.config.frames.rgb_optical_frame,
            )
            self.session_logger.log_frame(
                frame_packet=packet,
                output=output,
                rgb_msg=rgb_msg,
                depth_msg=depth_msg,
                rgb_info=rgb_info,
                depth_info=depth_info,
            )
            self.frame_id += 1
        except Exception as exc:
            self.runtime.diagnostics.last_error = str(exc)
            self.runtime.diagnostics.messages["last_exception"] = str(exc)
            self.session_logger.log_exception(
                where="_process_synced",
                exc=exc,
                traceback_text=traceback.format_exc(),
            )
            self.get_logger().error(f"runtime frame processing failed: {exc}")
            self.get_logger().debug(traceback.format_exc())
        finally:
            self.busy = False

    def _configured_depth_to_rgb(self):
        transform = self.config.extrinsics.depth_to_rgb
        if transform is None:
            return None
        return np.asarray(transform, dtype=np.float32).reshape(4, 4)

    def _fresh_camera_info(self, msg, packet_time: float):
        if msg is None:
            return None
        info_time = stamp_to_sec(msg.header.stamp)
        if info_time <= 0.0:
            return msg
        max_age = self.config.sync.camera_info_max_age_sec
        if max_age > 0.0 and packet_time - info_time > max_age:
            return None
        return msg

    def _topic_report(self) -> dict[str, str]:
        return {
            "rgb": self.config.topics.rgb,
            "depth": self.config.topics.depth,
            "rgb_camera_info": self.config.topics.rgb_camera_info,
            "depth_camera_info": self.config.topics.depth_camera_info,
            "odom": self.config.topics.odom,
            "pose": self.config.topics.pose,
            "semantic_points": self.config.topics.semantic_points,
            "semantic_objects": self.config.topics.semantic_objects,
            "map_points": self.config.topics.map_points,
            "debug_overlay": self.config.topics.debug_overlay,
            "diagnostics": self.config.topics.diagnostics,
            "vo_odom": self.config.topics.vo_odom,
        }


def main(args=None):
    rclpy.init(args=args)
    node = SemanticSpatialNode()
    try:
        rclpy.spin(node)
    finally:
        node.session_logger.log_event("node_shutdown")
        node.runtime.close()
        node.session_logger.close()
        node.destroy_node()
        rclpy.shutdown()
