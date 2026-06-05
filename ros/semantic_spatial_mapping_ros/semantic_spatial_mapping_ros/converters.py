from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

from geometry.transforms.camera_models import CameraIntrinsics
from runtime.core.types import CameraCalibration


_POINT_STEP = 12


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def sec_to_stamp(node, timestamp: float):
    msg = node.get_clock().now().to_msg()
    if timestamp > 0.0:
        msg.sec = int(timestamp)
        msg.nanosec = int((timestamp - msg.sec) * 1e9)
    return msg


def camera_info_to_intrinsics(msg: CameraInfo) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(msg.k[0]),
        fy=float(msg.k[4]),
        cx=float(msg.k[2]),
        cy=float(msg.k[5]),
        width=int(msg.width),
        height=int(msg.height),
    )


def make_calibration(
    rgb_info: Optional[CameraInfo],
    depth_info: Optional[CameraInfo],
    default_rgb_frame: str,
    default_depth_frame: str,
    T_depth_rgb: Optional[np.ndarray] = None,
) -> CameraCalibration:
    rgb_intrinsics = camera_info_to_intrinsics(rgb_info) if rgb_info is not None else None
    depth_intrinsics = (
        camera_info_to_intrinsics(depth_info) if depth_info is not None else None
    )
    return CameraCalibration(
        rgb_intrinsics=rgb_intrinsics,
        depth_intrinsics=depth_intrinsics,
        rgb_frame_id=rgb_info.header.frame_id if rgb_info is not None else default_rgb_frame,
        depth_frame_id=(
            depth_info.header.frame_id if depth_info is not None else default_depth_frame
        ),
        T_depth_rgb=T_depth_rgb,
    )


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)

    if encoding in ("rgb8", "bgr8"):
        dtype = np.uint8
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        dtype = np.uint8
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        dtype = np.uint8
        channels = 1
    elif encoding in ("mono16", "16uc1"):
        dtype = np.uint16
        channels = 1
    elif encoding == "32fc1":
        dtype = np.float32
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    row_stride = int(msg.step)
    itemsize = np.dtype(dtype).itemsize
    expected_row = width * channels * itemsize
    raw = np.frombuffer(msg.data, dtype=dtype)

    if row_stride == expected_row:
        array = raw.reshape(height, width, channels) if channels > 1 else raw.reshape(height, width)
    else:
        row_items = row_stride // itemsize
        padded = raw.reshape(height, row_items)
        array = padded[:, : width * channels]
        array = array.reshape(height, width, channels) if channels > 1 else array.reshape(height, width)

    array = array.copy()
    if encoding == "bgr8":
        array = array[:, :, ::-1].copy()
    elif encoding == "bgra8":
        array = array[:, :, [2, 1, 0, 3]].copy()
    elif encoding in ("rgba8", "bgra8"):
        array = array[:, :, :3].copy()
    return array


def numpy_to_image_msg(
    image: np.ndarray,
    frame_id: str,
    stamp,
    encoding: str = "rgb8",
) -> Image:
    arr = np.asarray(image)
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(arr.strides[0])
    msg.data = arr.tobytes()
    return msg


def quaternion_to_matrix(q: Quaternion) -> np.ndarray:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(R: np.ndarray) -> Quaternion:
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def pose_msg_to_matrix(pose: Pose) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(pose.orientation)
    T[:3, 3] = [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    ]
    return T


def matrix_to_pose_msg(T: np.ndarray) -> Pose:
    T = np.asarray(T, dtype=np.float64)
    msg = Pose()
    msg.position = Point(
        x=float(T[0, 3]),
        y=float(T[1, 3]),
        z=float(T[2, 3]),
    )
    msg.orientation = matrix_to_quaternion(T[:3, :3])
    return msg


def points_to_pointcloud2(
    points: Iterable[np.ndarray],
    frame_id: str,
    stamp,
) -> PointCloud2:
    if isinstance(points, np.ndarray):
        array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    else:
        chunks = [np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in points if len(p)]
        array = np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float32)

    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = int(len(array))
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = _POINT_STEP
    msg.row_step = _POINT_STEP * int(len(array))
    msg.is_dense = bool(len(array) > 0 and np.isfinite(array).all())
    msg.data = array.astype(np.float32, copy=False).tobytes()
    return msg
