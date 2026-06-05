from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from geometry.transforms.camera_models import CameraIntrinsics
from runtime.core.config import RuntimeConfig, load_runtime_config, validate_runtime_config
from runtime.core.pose import PoseEstimate
from runtime.core.pose_providers import ExternalPoseProvider, StaticIdentityPoseProvider
from runtime.core.runtime_logger import RuntimeSessionLogger, default_rosbag_topics
from runtime.core.segmentation import (
    DisabledSegmentationProvider,
    MockSegmentationProvider,
    build_segmentation_provider,
)
from runtime.core.semantic_runtime import DeploymentSemanticRuntime
from runtime.core.types import CameraCalibration, FramePacket


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FixedPoseProvider:
    def __init__(self, poses):
        self.poses = list(poses)
        self.index = 0

    def update(self, frame_packet):
        pose = self.poses[min(self.index, len(self.poses) - 1)]
        self.index += 1
        return pose


class _EmptySegmentation:
    def segment(self, rgb_frame):
        return {
            "overlay": rgb_frame,
            "masks": [],
            "obbs": [],
            "elapsed_ms": 0.0,
        }


class RuntimeCoreTests(unittest.TestCase):
    def test_shipped_configs_validate(self):
        for name in ("gazebo.yaml", "embedded_oakd.yaml"):
            config = load_runtime_config(
                REPO_ROOT / "ros" / "semantic_spatial_mapping_ros" / "config" / name
            )
            validate_runtime_config(config)

    def test_invalid_config_rejected(self):
        config = RuntimeConfig()
        config.depth.depth_unit = 0.0
        with self.assertRaises(ValueError):
            validate_runtime_config(config)

    def test_bad_extrinsics_rejected(self):
        config = RuntimeConfig()
        config.extrinsics.depth_to_rgb = [[1.0, 0.0], [0.0, 1.0]]
        with self.assertRaises(ValueError):
            validate_runtime_config(config)

    def test_external_pose_provider_rejects_stale_pose(self):
        provider = ExternalPoseProvider("odom", max_age_sec=0.1)
        provider.set_latest(
            PoseEstimate(
                success=True,
                timestamp=1.0,
                T_world_cam=np.eye(4),
                source="odom",
            )
        )
        packet = FramePacket(
            frame_id=0,
            timestamp=1.2,
            rgb_frame=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        result = provider.update(packet)
        self.assertFalse(result.success)
        self.assertIn("STALE_EXTERNAL_POSE", result.status)

    def test_segmentation_provider_factory(self):
        config = RuntimeConfig().segmentation
        config.enabled = False
        self.assertIsInstance(build_segmentation_provider(config), DisabledSegmentationProvider)

        config.enabled = True
        config.backend = "mock"
        provider = build_segmentation_provider(config)
        self.assertIsInstance(provider, MockSegmentationProvider)
        result = provider.segment(np.zeros((16, 16, 3), dtype=np.uint8))
        self.assertEqual(len(result["masks"]), 1)

    def test_runtime_maps_synthetic_rgbd(self):
        config = RuntimeConfig()
        config.segmentation.enabled = True
        config.segmentation.backend = "mock"
        config.segmentation.minimum_mask_area = 10
        config.depth.stride = 2
        config.mapping.max_points = 200

        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=StaticIdentityPoseProvider(),
            segmentation_module=MockSegmentationProvider(),
        )
        packet = FramePacket(
            frame_id=0,
            timestamp=1.0,
            rgb_frame=np.zeros((32, 32, 3), dtype=np.uint8),
            depth_frame=np.full((32, 32), 1000, dtype=np.uint16),
            calibration=CameraCalibration(
                rgb_intrinsics=CameraIntrinsics(40.0, 40.0, 16.0, 16.0, 32, 32),
                depth_intrinsics=CameraIntrinsics(40.0, 40.0, 16.0, 16.0, 32, 32),
                T_depth_rgb=np.eye(4, dtype=np.float32),
            ),
        )
        output = runtime.update(packet)
        self.assertTrue(output.tracking_ok)
        self.assertGreater(len(runtime.latest_map()), 0)
        self.assertLessEqual(len(runtime.latest_map()), config.mapping.max_points)
        self.assertEqual(output.diagnostics["health"], "OK")
        self.assertTrue(output.diagnostics["camera_info_ready"])
        self.assertGreaterEqual(output.diagnostics["semantic_objects"], 1)
        self.assertGreaterEqual(len(output.semantic_objects), 1)

    def test_missing_depth_does_not_map(self):
        config = RuntimeConfig()
        config.segmentation.enabled = True
        config.segmentation.backend = "mock"
        config.segmentation.minimum_mask_area = 10
        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=StaticIdentityPoseProvider(),
            segmentation_module=MockSegmentationProvider(),
        )
        output = runtime.update(
            FramePacket(
                frame_id=0,
                timestamp=1.0,
                rgb_frame=np.zeros((32, 32, 3), dtype=np.uint8),
            )
        )
        self.assertTrue(output.tracking_ok)
        self.assertEqual(len(runtime.latest_map()), 0)
        self.assertFalse(output.diagnostics["depth_received"])

    def test_missing_camera_info_does_not_map(self):
        config = RuntimeConfig()
        config.segmentation.enabled = True
        config.segmentation.backend = "mock"
        config.segmentation.minimum_mask_area = 10
        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=StaticIdentityPoseProvider(),
            segmentation_module=MockSegmentationProvider(),
        )
        output = runtime.update(
            FramePacket(
                frame_id=0,
                timestamp=1.0,
                rgb_frame=np.zeros((32, 32, 3), dtype=np.uint8),
                depth_frame=np.full((32, 32), 1000, dtype=np.uint16),
            )
        )
        self.assertTrue(output.tracking_ok)
        self.assertEqual(len(runtime.latest_map()), 0)
        self.assertFalse(output.diagnostics["camera_info_ready"])

    def test_empty_segmentation_does_not_map(self):
        config = RuntimeConfig()
        config.segmentation.enabled = True
        config.segmentation.backend = "mock"
        config.segmentation.minimum_mask_area = 10
        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=StaticIdentityPoseProvider(),
            segmentation_module=_EmptySegmentation(),
        )
        output = runtime.update(self._packet())
        self.assertTrue(output.tracking_ok)
        self.assertEqual(len(runtime.latest_map()), 0)
        self.assertEqual(output.diagnostics["masks_processed"], 0)

    def test_object_observations_fuse_across_frames(self):
        config = RuntimeConfig()
        config.segmentation.enabled = True
        config.segmentation.backend = "mock"
        config.segmentation.minimum_mask_area = 10
        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=StaticIdentityPoseProvider(),
            segmentation_module=MockSegmentationProvider(),
        )
        runtime.update(self._packet(frame_id=0, timestamp=1.0))
        runtime.update(self._packet(frame_id=1, timestamp=2.0))
        objects = runtime.latest_objects()
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["observations"], 2)

    def test_nonfinite_pose_is_rejected(self):
        config = RuntimeConfig()
        config.segmentation.minimum_mask_area = 10
        bad_pose = np.eye(4)
        bad_pose[0, 3] = np.nan
        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=_FixedPoseProvider(
                [
                    PoseEstimate(
                        success=True,
                        timestamp=1.0,
                        T_world_cam=bad_pose,
                        source="test",
                    )
                ]
            ),
            segmentation_module=MockSegmentationProvider(),
        )
        output = runtime.update(self._packet())
        self.assertFalse(output.tracking_ok)
        self.assertIn("INVALID_POSE_NONFINITE", output.pose.status)

    def test_pose_jump_is_rejected(self):
        config = RuntimeConfig()
        config.segmentation.minimum_mask_area = 10
        config.pose.max_jump_m = 1.0
        jumped = np.eye(4)
        jumped[0, 3] = 2.0
        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=_FixedPoseProvider(
                [
                    PoseEstimate(True, 1.0, np.eye(4), source="test"),
                    PoseEstimate(True, 2.0, jumped, source="test"),
                ]
            ),
            segmentation_module=MockSegmentationProvider(),
        )
        first = runtime.update(self._packet(frame_id=0, timestamp=1.0))
        second = runtime.update(self._packet(frame_id=1, timestamp=2.0))
        self.assertTrue(first.tracking_ok)
        self.assertFalse(second.tracking_ok)
        self.assertIn("INVALID_POSE_JUMP", second.pose.status)

    def test_point_caps_are_applied(self):
        config = RuntimeConfig()
        config.segmentation.enabled = True
        config.segmentation.backend = "mock"
        config.segmentation.minimum_mask_area = 10
        config.depth.stride = 1
        config.mapping.max_points_per_frame = 10
        config.mapping.max_points_per_object = 10
        runtime = DeploymentSemanticRuntime(
            config=config,
            pose_provider=StaticIdentityPoseProvider(),
            segmentation_module=MockSegmentationProvider(),
        )
        output = runtime.update(self._packet())
        self.assertLessEqual(output.diagnostics["points_generated"], 10)
        self.assertLessEqual(len(runtime.latest_map()), 10)

    def test_runtime_logger_writes_text_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeConfig()
            config.profile = "gazebo"
            config.logging.directory = tmpdir
            logger = RuntimeSessionLogger(config=config, config_path="test.yaml")
            runtime = DeploymentSemanticRuntime(
                config=config,
                pose_provider=StaticIdentityPoseProvider(),
                segmentation_module=MockSegmentationProvider(),
            )
            output = runtime.update(self._packet())
            logger.log_subscriptions({"rgb": config.topics.rgb})
            logger.log_frame(self._packet(), output)
            logger.close()

            self.assertTrue(logger.report_path.exists())
            text = logger.report_path.read_text(encoding="utf-8")
            self.assertIn("RUNTIME_SESSION", text)
            self.assertIn("\"kind\": \"FRAME\"", text)
            self.assertIn("RUNTIME_CONFIG_JSON", text)

    def test_default_rosbag_topics_include_embedded_inputs_and_outputs(self):
        config = RuntimeConfig()
        config.profile = "embedded_oakd"
        topics = default_rosbag_topics(config)
        self.assertIn(config.topics.rgb, topics)
        self.assertIn(config.topics.depth, topics)
        self.assertIn(config.topics.semantic_objects, topics)
        self.assertIn(config.topics.diagnostics, topics)

    def _packet(self, frame_id=0, timestamp=1.0):
        return FramePacket(
            frame_id=frame_id,
            timestamp=timestamp,
            rgb_frame=np.zeros((32, 32, 3), dtype=np.uint8),
            depth_frame=np.full((32, 32), 1000, dtype=np.uint16),
            calibration=CameraCalibration(
                rgb_intrinsics=CameraIntrinsics(40.0, 40.0, 16.0, 16.0, 32, 32),
                depth_intrinsics=CameraIntrinsics(40.0, 40.0, 16.0, 16.0, 32, 32),
                T_depth_rgb=np.eye(4, dtype=np.float32),
            ),
        )


if __name__ == "__main__":
    unittest.main()
