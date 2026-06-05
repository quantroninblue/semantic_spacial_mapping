from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROS_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros"
    / "semantic_spatial_mapping_ros"
)
sys.path.insert(0, str(ROS_PACKAGE_ROOT))

try:
    from geometry_msgs.msg import Pose
    from sensor_msgs.msg import CameraInfo, Image

    from semantic_spatial_mapping_ros.converters import (
        camera_info_to_intrinsics,
        image_msg_to_numpy,
        matrix_to_pose_msg,
        points_to_pointcloud2,
        pose_msg_to_matrix,
    )

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, "ROS2 Python message packages are not available")
class RosConverterTests(unittest.TestCase):
    def test_rgb8_image_conversion(self):
        image = Image()
        image.height = 2
        image.width = 2
        image.encoding = "rgb8"
        image.step = 6
        image.data = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        array = image_msg_to_numpy(image)
        self.assertEqual(array.shape, (2, 2, 3))
        self.assertEqual(int(array[0, 0, 0]), 1)

    def test_camera_info_conversion(self):
        info = CameraInfo()
        info.width = 640
        info.height = 480
        info.k = [500.0, 0.0, 320.0, 0.0, 501.0, 240.0, 0.0, 0.0, 1.0]

        intr = camera_info_to_intrinsics(info)
        self.assertEqual(intr.width, 640)
        self.assertEqual(intr.height, 480)
        self.assertEqual(intr.fx, 500.0)
        self.assertEqual(intr.fy, 501.0)

    def test_pose_round_trip(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        pose = matrix_to_pose_msg(T)
        recovered = pose_msg_to_matrix(pose)
        np.testing.assert_allclose(recovered, T, atol=1e-6)

    def test_pointcloud2_packing(self):
        msg = points_to_pointcloud2(
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            frame_id="map",
            stamp=None,
        )
        self.assertEqual(msg.width, 1)
        self.assertEqual(msg.point_step, 12)
        self.assertEqual(len(msg.data), 12)


if __name__ == "__main__":
    unittest.main()
