from __future__ import annotations

import time
from typing import Optional

import numpy as np

from geometry.pointclouds.object_pointclouds import ObjectPointCloudExtractor
from geometry.pointclouds.pointcloud_generation import PointCloudGenerator
from geometry.transforms.depth_to_rgb_projection import DepthToRGBProjector
from geometry.transforms.extrinsics import ExtrinsicTransform
from mapping.global_map.semantic_object_map import SemanticObjectMap
from mapping.global_map.world_map import WorldMap
from world.transforms.world_frame_projections import WorldFrameProjector

from .config import RuntimeConfig, validate_runtime_config
from .diagnostics import RuntimeDiagnostics
from .perception import (
    compute_object_geometry,
    filter_point_outliers,
    semantic_frame_from_result,
)
from .pose import PoseEstimate, PoseProvider
from .types import FramePacket, RuntimeOutput


class DeploymentSemanticRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        pose_provider: PoseProvider,
        segmentation_module=None,
    ):
        validate_runtime_config(config)
        self.config = config
        self.pose_provider = pose_provider
        self.segmentation = segmentation_module
        self.world_projector = WorldFrameProjector()
        self.world_map = WorldMap(
            voxel_size_m=config.mapping.voxel_size_m,
            max_points=config.mapping.max_points,
        )
        self.semantic_object_map = SemanticObjectMap(
            association_distance_m=config.mapping.object_association_distance_m,
            max_objects=config.mapping.max_objects,
            max_points_per_object=config.mapping.max_points_per_object,
            voxel_size_m=config.mapping.voxel_size_m,
        )
        self.diagnostics = RuntimeDiagnostics(
            segmentation_active=(
                config.segmentation.enabled
                and config.segmentation.backend.lower() != "disabled"
            )
        )
        self._extractor: Optional[ObjectPointCloudExtractor] = None
        self._last_valid_T_world_cam: Optional[np.ndarray] = None

    def update(self, packet: FramePacket) -> RuntimeOutput:
        t0 = time.perf_counter()
        self.diagnostics.rgb_received = packet.rgb_frame is not None
        self.diagnostics.depth_received = packet.depth_frame is not None
        self.diagnostics.camera_info_ready = (
            packet.calibration is not None and packet.calibration.ready_for_rgbd
        )
        self.diagnostics.last_error = ""

        pose = self.pose_provider.update(packet)
        pose = self._validate_pose(pose, packet.timestamp)
        self.diagnostics.pose_source_active = pose.success
        self.diagnostics.tracking_ok = pose.success
        self.diagnostics.pose_age_sec = max(0.0, packet.timestamp - pose.timestamp)

        seg_t0 = time.perf_counter()
        raw_segmentation = self._segment(packet)
        self.diagnostics.segmentation_ms = (time.perf_counter() - seg_t0) * 1000.0
        if raw_segmentation is not None:
            self.diagnostics.segmentation_ms = float(
                raw_segmentation.get("elapsed_ms", self.diagnostics.segmentation_ms)
            )
        semantic_frame = semantic_frame_from_result(
            result=raw_segmentation,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            config=self.config,
            depth_frame=packet.depth_frame,
        )
        segmentation_result = semantic_frame.to_legacy_dict()

        world_points = []

        map_t0 = time.perf_counter()
        masks_processed = 0
        points_generated = 0
        if (
            pose.success
            and packet.depth_frame is not None
            and semantic_frame is not None
            and packet.calibration is not None
            and packet.calibration.ready_for_rgbd
        ):
            extractor = self._get_extractor(packet)
            remaining_points = self.config.mapping.max_points_per_frame
            for instance in semantic_frame.instances:
                if remaining_points <= 0:
                    break
                masks_processed += 1
                points_camera = extractor.extract_object_pointcloud(
                    depth_frame=packet.depth_frame,
                    segmentation_mask=instance.mask,
                    depth_min_m=self.config.depth.min_m,
                    depth_max_m=self.config.depth.max_m,
                    stride=self.config.depth.stride,
                    depth_unit=self.config.depth.depth_unit,
                )
                if len(points_camera) == 0:
                    continue
                points_camera = filter_point_outliers(points_camera)
                points_camera = self._limit_points(
                    points_camera,
                    min(self.config.mapping.max_points_per_object, remaining_points),
                )
                remaining_points -= len(points_camera)
                points_generated += len(points_camera)
                points_world = self.world_projector.project_points_to_world(
                    points_camera=points_camera,
                    T_world_cam=pose.T_world_cam,
                )
                self.world_map.add_points(points_world)
                world_points.append(points_world)
                geometry = compute_object_geometry(
                    points_world,
                    valid_depth_ratio=instance.depth_support_ratio,
                )
                if geometry is not None:
                    obj = self.semantic_object_map.update(
                        class_id=instance.class_id,
                        label=instance.label,
                        confidence=instance.confidence,
                        centroid=geometry.centroid,
                        extent=geometry.extent,
                        covariance=geometry.covariance,
                        points=points_world,
                        timestamp=packet.timestamp,
                    )
        self.diagnostics.mapping_ms = (time.perf_counter() - map_t0) * 1000.0
        self.diagnostics.masks_processed = masks_processed
        self.diagnostics.points_generated = points_generated
        self.diagnostics.semantic_objects = len(self.semantic_object_map.objects())

        if pose.success:
            self._last_valid_T_world_cam = pose.T_world_cam.copy()

        self.diagnostics.processed_frames += 1
        self.diagnostics.map_points = len(self.world_map.get_all_points())
        self.diagnostics.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.diagnostics.recompute_health()

        return RuntimeOutput(
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            tracking_ok=pose.success,
            pose=pose,
            segmentation=segmentation_result,
            world_points=world_points,
            semantic_objects=self.semantic_object_map.as_dicts(),
            diagnostics=self.diagnostics.as_dict(),
        )

    def latest_map(self) -> np.ndarray:
        return self.world_map.get_all_points()

    def latest_objects(self) -> list[dict]:
        return self.semantic_object_map.as_dicts()

    def close(self) -> None:
        if hasattr(self.pose_provider, "close"):
            self.pose_provider.close()
        if self.segmentation is not None and hasattr(self.segmentation, "close"):
            self.segmentation.close()

    def _segment(self, packet: FramePacket) -> Optional[dict]:
        if self.segmentation is None or not self.config.segmentation.enabled:
            return {"overlay": packet.rgb_frame, "masks": [], "obbs": [], "elapsed_ms": 0.0}
        return self.segmentation.segment(packet.rgb_frame)

    def _validate_pose(self, pose: PoseEstimate, timestamp: float) -> PoseEstimate:
        if not pose.success:
            return pose

        T = np.asarray(pose.T_world_cam)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            return PoseEstimate(
                success=False,
                timestamp=timestamp,
                source=pose.source,
                status="INVALID_POSE_NONFINITE",
            )

        translation_norm = float(np.linalg.norm(T[:3, 3]))
        if translation_norm > self.config.pose.max_translation_norm_m:
            return PoseEstimate(
                success=False,
                timestamp=timestamp,
                source=pose.source,
                status=(
                    "INVALID_POSE_TRANSLATION_NORM "
                    f"norm_m={translation_norm:.3f}"
                ),
            )

        if self._last_valid_T_world_cam is not None and self.config.pose.max_jump_m > 0.0:
            jump = float(
                np.linalg.norm(T[:3, 3] - self._last_valid_T_world_cam[:3, 3])
            )
            if jump > self.config.pose.max_jump_m:
                return PoseEstimate(
                    success=False,
                    timestamp=timestamp,
                    source=pose.source,
                    status=f"INVALID_POSE_JUMP jump_m={jump:.3f}",
                )

        return pose

    @staticmethod
    def _limit_points(points: np.ndarray, max_points: int) -> np.ndarray:
        if len(points) <= max_points:
            return points
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        return points[indices]

    def _get_extractor(self, packet: FramePacket) -> ObjectPointCloudExtractor:
        calib = packet.calibration
        if self._extractor is None:
            generator = PointCloudGenerator(calib.depth_intrinsics)
            extrinsic = None
            if calib.T_depth_rgb is not None:
                T_depth_rgb = np.asarray(calib.T_depth_rgb, dtype=np.float32).reshape(4, 4)
                extrinsic = ExtrinsicTransform(
                    rotation_matrix=T_depth_rgb[:3, :3],
                    translation_vector=T_depth_rgb[:3, 3],
                )
            projector = DepthToRGBProjector(
                depth_intrinsics=calib.depth_intrinsics,
                rgb_intrinsics=calib.rgb_intrinsics,
                **({"extrinsic_transform": extrinsic} if extrinsic is not None else {}),
            )
            self._extractor = ObjectPointCloudExtractor(generator, projector)
        return self._extractor
