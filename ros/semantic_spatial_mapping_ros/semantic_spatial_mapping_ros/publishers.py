from __future__ import annotations

import json

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from runtime.core.config import RuntimeConfig
from runtime.core.types import RuntimeOutput

from .converters import matrix_to_pose_msg, numpy_to_image_msg, points_to_pointcloud2, sec_to_stamp


class RuntimePublishers:
    def __init__(self, node, config: RuntimeConfig):
        self.node = node
        self.config = config
        qos = 10
        self.semantic_points = node.create_publisher(
            PointCloud2,
            config.topics.semantic_points,
            qos,
        )
        self.semantic_objects = node.create_publisher(
            String,
            config.topics.semantic_objects,
            qos,
        )
        self.map_points = node.create_publisher(PointCloud2, config.topics.map_points, qos)
        self.diagnostics = node.create_publisher(String, config.topics.diagnostics, qos)
        self.vo_odom = node.create_publisher(Odometry, config.topics.vo_odom, qos)
        self.overlay = None
        if config.segmentation.publish_overlay:
            self.overlay = node.create_publisher(Image, config.topics.debug_overlay, qos)

    def publish(self, output: RuntimeOutput, latest_map: np.ndarray, rgb_frame_id: str) -> None:
        stamp = sec_to_stamp(self.node, output.timestamp)
        frame_id = self.config.frames.map_frame

        self.semantic_points.publish(
            points_to_pointcloud2(output.world_points, frame_id=frame_id, stamp=stamp)
        )

        if self._should_publish_map(output.frame_id):
            self.map_points.publish(
                points_to_pointcloud2(latest_map, frame_id=frame_id, stamp=stamp)
            )

        if output.pose is not None and output.pose.success:
            self.vo_odom.publish(self._pose_to_odom(output, stamp))

        if self.overlay is not None and output.segmentation is not None:
            overlay = output.segmentation.get("overlay")
            if overlay is not None:
                self.overlay.publish(
                    numpy_to_image_msg(
                        overlay,
                        frame_id=rgb_frame_id,
                        stamp=stamp,
                        encoding="rgb8",
                    )
                )

        self.diagnostics.publish(String(data=json.dumps(output.diagnostics or {})))
        self.semantic_objects.publish(
            String(data=json.dumps(output.semantic_objects or []))
        )

    def _pose_to_odom(self, output: RuntimeOutput, stamp) -> Odometry:
        pose = output.pose
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = pose.frame_id or self.config.frames.map_frame
        msg.child_frame_id = pose.child_frame_id or self.config.frames.camera_frame
        msg.pose.pose = matrix_to_pose_msg(pose.T_world_cam)
        if pose.covariance is not None:
            cov = np.asarray(pose.covariance, dtype=float).reshape(-1)
            msg.pose.covariance = list(cov[:36]) if len(cov) >= 36 else list(msg.pose.covariance)
        return msg

    def _should_publish_map(self, frame_id: int) -> bool:
        n = max(int(self.config.mapping.publish_every_n_frames), 1)
        return frame_id % n == 0
