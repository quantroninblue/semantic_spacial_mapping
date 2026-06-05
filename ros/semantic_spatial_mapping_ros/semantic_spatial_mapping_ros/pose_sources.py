from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformException

from runtime.core.config import RuntimeConfig, matrix4
from runtime.core.pose import PoseEstimate, PoseProvider
from runtime.core.pose_providers import (
    ExternalPoseProvider,
    StaticIdentityPoseProvider,
    VisualOdometryPoseProvider,
)

from .converters import pose_msg_to_matrix, stamp_to_sec


class RosOdometryPoseProvider(ExternalPoseProvider):
    def __init__(self, config: RuntimeConfig):
        super().__init__("odom", max_age_sec=config.pose.max_age_sec)
        self.config = config

    def callback(self, msg: Odometry) -> None:
        T_world_cam = ros_pose_to_runtime_camera_pose(
            pose_matrix=pose_msg_to_matrix(msg.pose.pose),
            config=self.config,
        )
        self.set_latest(
            PoseEstimate(
                success=True,
                timestamp=stamp_to_sec(msg.header.stamp),
                T_world_cam=T_world_cam,
                source=self.source_name,
                contract="T_world_cam",
                frame_id=msg.header.frame_id or self.config.frames.odom_frame,
                child_frame_id=self.config.frames.camera_frame,
                status="OK",
            )
        )


class RosPoseStampedPoseProvider(ExternalPoseProvider):
    def __init__(self, config: RuntimeConfig):
        super().__init__("pose", max_age_sec=config.pose.max_age_sec)
        self.config = config

    def callback(self, msg: PoseStamped) -> None:
        T_world_cam = ros_pose_to_runtime_camera_pose(
            pose_matrix=pose_msg_to_matrix(msg.pose),
            config=self.config,
        )
        self.set_latest(
            PoseEstimate(
                success=True,
                timestamp=stamp_to_sec(msg.header.stamp),
                T_world_cam=T_world_cam,
                source=self.source_name,
                contract="T_world_cam",
                frame_id=msg.header.frame_id or self.config.frames.map_frame,
                child_frame_id=self.config.frames.camera_frame,
                status="OK",
            )
        )


class TfPoseProvider(PoseProvider):
    source_name = "tf"

    def __init__(self, config: RuntimeConfig, tf_buffer: Buffer):
        self.config = config
        self.tf_buffer = tf_buffer
        self._latest = None

    def update(self, frame_packet) -> PoseEstimate:
        source_frame = (
            self.config.frames.base_frame
            if self.config.pose.input_pose.lower() == "world_base"
            else self.config.frames.camera_frame
        )
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config.frames.map_frame,
                source_frame,
                rclpy_time_from_sec(frame_packet.timestamp),
            )
        except TransformException as exc:
            return PoseEstimate(
                success=False,
                timestamp=frame_packet.timestamp,
                source=self.source_name,
                status=f"TF_LOOKUP_FAILED: {exc}",
            )

        T = transform_msg_to_matrix(transform.transform)
        T_world_cam = ros_pose_to_runtime_camera_pose(
            pose_matrix=T,
            config=self.config,
        )
        self._latest = PoseEstimate(
            success=True,
            timestamp=stamp_to_sec(transform.header.stamp),
            T_world_cam=T_world_cam,
            source=self.source_name,
            contract="T_world_cam",
            frame_id=transform.header.frame_id,
            child_frame_id=self.config.frames.camera_frame,
            status="OK",
        )
        return self._latest

    def latest(self):
        return self._latest


def rclpy_time_from_sec(timestamp: float):
    from rclpy.time import Time

    if timestamp <= 0.0:
        return Time()
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    return Time(seconds=sec, nanoseconds=nanosec)


def transform_msg_to_matrix(transform) -> np.ndarray:
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x = transform.translation.x
    pose.position.y = transform.translation.y
    pose.position.z = transform.translation.z
    pose.orientation = transform.rotation
    return pose_msg_to_matrix(pose)


def ros_pose_to_runtime_camera_pose(pose_matrix: np.ndarray, config: RuntimeConfig) -> np.ndarray:
    input_pose = config.pose.input_pose.lower()
    if input_pose == "world_camera":
        return pose_matrix
    if input_pose == "world_base":
        T_base_cam = matrix4(
            config.extrinsics.base_to_camera,
            "extrinsics.base_to_camera",
        )
        if T_base_cam is None:
            raise ValueError(
                "pose.input_pose=world_base requires extrinsics.base_to_camera"
            )
        return pose_matrix @ T_base_cam
    raise ValueError(f"Unsupported pose.input_pose: {config.pose.input_pose}")


def build_pose_provider(config: RuntimeConfig, tf_buffer: Buffer | None = None) -> PoseProvider:
    source = (config.pose.source or "identity").lower()
    if source in ("odom", "odometry"):
        return RosOdometryPoseProvider(config)
    if source in ("pose", "pose_stamped", "posestamped"):
        return RosPoseStampedPoseProvider(config)
    if source == "tf":
        if tf_buffer is None:
            raise ValueError("TF pose source requires a tf2_ros.Buffer")
        return TfPoseProvider(config, tf_buffer)
    if source in ("internal_vslam", "vslam", "vo"):
        return VisualOdometryPoseProvider(config)
    if source in ("identity", "none", "degraded"):
        return StaticIdentityPoseProvider()
    raise ValueError(f"Unsupported pose source: {config.pose.source}")
