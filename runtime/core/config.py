from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


@dataclass
class TopicConfig:
    rgb: str = "/camera/color/image_raw"
    depth: str = "/camera/depth/image_raw"
    rgb_camera_info: str = "/camera/color/camera_info"
    depth_camera_info: str = "/camera/depth/camera_info"
    odom: str = "/odom"
    pose: str = "/pose"
    semantic_points: str = "/semantic_spatial/points"
    semantic_objects: str = "/semantic_spatial/objects"
    map_points: str = "/semantic_spatial/map"
    debug_overlay: str = "/semantic_spatial/debug_overlay"
    diagnostics: str = "/semantic_spatial/diagnostics"
    vo_odom: str = "/semantic_spatial/visual_odometry"


@dataclass
class FrameConfig:
    map_frame: str = "map"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    camera_frame: str = "camera_link"
    rgb_optical_frame: str = "camera_rgb_optical_frame"
    depth_optical_frame: str = "camera_depth_optical_frame"


@dataclass
class SyncConfig:
    queue_size: int = 8
    slop_sec: float = 0.05
    max_fps: float = 15.0
    camera_info_max_age_sec: float = 2.0
    drop_when_busy: bool = True


@dataclass
class DepthConfig:
    depth_unit: float = 0.001
    min_m: float = 0.1
    max_m: float = 5.0
    stride: int = 4


@dataclass
class PoseConfig:
    source: str = "odom"
    input_pose: str = "world_camera"
    max_age_sec: float = 0.25
    max_jump_m: float = 5.0
    max_translation_norm_m: float = 10000.0
    publish_internal_vslam: bool = False
    tf_timeout_sec: float = 0.05


@dataclass
class VSLAMConfig:
    enabled: bool = False
    local_mapping: bool = False
    loop_detection: bool = False
    use_bow_vocab: bool = False
    vocab_path: Optional[str] = None
    scale_mode: str = "rgbd"


@dataclass
class SegmentationConfig:
    enabled: bool = True
    backend: str = "yolo"
    model_path: str = "yolov8n-seg.pt"
    confidence_threshold: float = 0.45
    minimum_mask_area: int = 1500
    max_masks_per_frame: int = 8
    min_depth_support_ratio: float = 0.05
    max_border_touch_ratio: float = 0.35
    publish_overlay: bool = False


@dataclass
class MapConfig:
    voxel_size_m: float = 0.05
    max_points: int = 250000
    max_points_per_frame: int = 20000
    max_points_per_object: int = 5000
    max_objects: int = 512
    object_association_distance_m: float = 0.75
    publish_every_n_frames: int = 5


@dataclass
class ExtrinsicsConfig:
    depth_to_rgb: Optional[list[list[float]]] = None
    base_to_camera: Optional[list[list[float]]] = None


@dataclass
class RuntimeLoggingConfig:
    enabled: bool = True
    directory: str = "runtime_logs"
    frame_log_period: int = 1
    record_embedded_rosbag: bool = True
    rosbag_topics: list[str] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    profile: str = "generic"
    topics: TopicConfig = field(default_factory=TopicConfig)
    frames: FrameConfig = field(default_factory=FrameConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    vslam: VSLAMConfig = field(default_factory=VSLAMConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    mapping: MapConfig = field(default_factory=MapConfig)
    extrinsics: ExtrinsicsConfig = field(default_factory=ExtrinsicsConfig)
    logging: RuntimeLoggingConfig = field(default_factory=RuntimeLoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeConfig":
        data = data or {}
        return cls(
            profile=data.get("profile", "generic"),
            topics=_section(TopicConfig, data.get("topics")),
            frames=_section(FrameConfig, data.get("frames")),
            sync=_section(SyncConfig, data.get("sync")),
            depth=_section(DepthConfig, data.get("depth")),
            pose=_section(PoseConfig, data.get("pose")),
            vslam=_section(VSLAMConfig, data.get("vslam")),
            segmentation=_section(SegmentationConfig, data.get("segmentation")),
            mapping=_section(MapConfig, data.get("mapping")),
            extrinsics=_section(ExtrinsicsConfig, data.get("extrinsics")),
            logging=_section(RuntimeLoggingConfig, data.get("logging")),
        )


def _section(section_type, values):
    values = values or {}
    allowed = section_type.__dataclass_fields__.keys()
    return section_type(**{k: v for k, v in values.items() if k in allowed})


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    config = RuntimeConfig.from_dict(data)
    validate_runtime_config(config)
    return config


def validate_runtime_config(config: RuntimeConfig) -> None:
    errors = []

    valid_pose_sources = {"odom", "odometry", "pose", "pose_stamped", "posestamped", "tf", "internal_vslam", "vslam", "vo", "identity", "none", "degraded"}
    if config.pose.source.lower() not in valid_pose_sources:
        errors.append(f"pose.source must be one of {sorted(valid_pose_sources)}")

    valid_pose_inputs = {"world_camera", "world_base"}
    if config.pose.input_pose.lower() not in valid_pose_inputs:
        errors.append(f"pose.input_pose must be one of {sorted(valid_pose_inputs)}")

    if config.pose.max_age_sec < 0.0:
        errors.append("pose.max_age_sec must be non-negative")
    if config.pose.max_jump_m < 0.0:
        errors.append("pose.max_jump_m must be non-negative")
    if config.pose.max_translation_norm_m <= 0.0:
        errors.append("pose.max_translation_norm_m must be positive")

    if config.sync.queue_size <= 0:
        errors.append("sync.queue_size must be positive")
    if config.sync.slop_sec < 0.0:
        errors.append("sync.slop_sec must be non-negative")
    if config.sync.max_fps < 0.0:
        errors.append("sync.max_fps must be non-negative")
    if config.sync.camera_info_max_age_sec < 0.0:
        errors.append("sync.camera_info_max_age_sec must be non-negative")

    if config.depth.depth_unit <= 0.0:
        errors.append("depth.depth_unit must be positive")
    if config.depth.min_m < 0.0:
        errors.append("depth.min_m must be non-negative")
    if config.depth.max_m <= config.depth.min_m:
        errors.append("depth.max_m must be greater than depth.min_m")
    if config.depth.stride <= 0:
        errors.append("depth.stride must be positive")

    if config.mapping.voxel_size_m < 0.0:
        errors.append("mapping.voxel_size_m must be non-negative")
    if config.mapping.max_points <= 0:
        errors.append("mapping.max_points must be positive")
    if config.mapping.max_points_per_frame <= 0:
        errors.append("mapping.max_points_per_frame must be positive")
    if config.mapping.max_points_per_object <= 0:
        errors.append("mapping.max_points_per_object must be positive")
    if config.mapping.max_objects <= 0:
        errors.append("mapping.max_objects must be positive")
    if config.mapping.object_association_distance_m <= 0.0:
        errors.append("mapping.object_association_distance_m must be positive")
    if config.mapping.publish_every_n_frames <= 0:
        errors.append("mapping.publish_every_n_frames must be positive")

    valid_segmentation_backends = {"disabled", "mock", "yolo"}
    if config.segmentation.backend.lower() not in valid_segmentation_backends:
        errors.append(
            f"segmentation.backend must be one of {sorted(valid_segmentation_backends)}"
        )
    if config.segmentation.confidence_threshold < 0.0 or config.segmentation.confidence_threshold > 1.0:
        errors.append("segmentation.confidence_threshold must be in [0, 1]")
    if config.segmentation.minimum_mask_area < 0:
        errors.append("segmentation.minimum_mask_area must be non-negative")
    if config.segmentation.max_masks_per_frame <= 0:
        errors.append("segmentation.max_masks_per_frame must be positive")
    if (
        config.segmentation.min_depth_support_ratio < 0.0
        or config.segmentation.min_depth_support_ratio > 1.0
    ):
        errors.append("segmentation.min_depth_support_ratio must be in [0, 1]")
    if (
        config.segmentation.max_border_touch_ratio < 0.0
        or config.segmentation.max_border_touch_ratio > 1.0
    ):
        errors.append("segmentation.max_border_touch_ratio must be in [0, 1]")

    _validate_matrix(config.extrinsics.depth_to_rgb, "extrinsics.depth_to_rgb", errors)
    _validate_matrix(config.extrinsics.base_to_camera, "extrinsics.base_to_camera", errors)

    if not isinstance(config.logging.directory, str) or not config.logging.directory:
        errors.append("logging.directory must be a non-empty string")
    if config.logging.frame_log_period <= 0:
        errors.append("logging.frame_log_period must be positive")
    if not isinstance(config.logging.rosbag_topics, list):
        errors.append("logging.rosbag_topics must be a list")
    else:
        for idx, topic in enumerate(config.logging.rosbag_topics):
            if not isinstance(topic, str) or not topic:
                errors.append(f"logging.rosbag_topics[{idx}] must be a non-empty string")

    for section_name, section in (
        ("topics", config.topics),
        ("frames", config.frames),
    ):
        for key, value in section.__dict__.items():
            if not isinstance(value, str) or not value:
                errors.append(f"{section_name}.{key} must be a non-empty string")

    if errors:
        joined = "\n- ".join(errors)
        raise ValueError(f"Invalid runtime config:\n- {joined}")


def matrix4(values, name: str) -> Optional[np.ndarray]:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    return array


def _validate_matrix(values, name: str, errors: list[str]) -> None:
    if values is None:
        return
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        errors.append(f"{name} must be numeric")
        return
    if array.shape != (4, 4):
        errors.append(f"{name} must be a 4x4 matrix")
        return
    if not np.isfinite(array).all():
        errors.append(f"{name} must contain finite values")
