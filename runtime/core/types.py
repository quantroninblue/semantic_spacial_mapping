from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from geometry.transforms.camera_models import CameraIntrinsics


@dataclass
class CameraCalibration:
    rgb_intrinsics: Optional[CameraIntrinsics] = None
    depth_intrinsics: Optional[CameraIntrinsics] = None
    T_depth_rgb: Optional[np.ndarray] = None
    rgb_frame_id: str = "camera_rgb_optical_frame"
    depth_frame_id: str = "camera_depth_optical_frame"

    @property
    def ready_for_rgbd(self) -> bool:
        return (
            self.rgb_intrinsics is not None
            and self.depth_intrinsics is not None
        )


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    rgb_frame: np.ndarray
    depth_frame: Optional[np.ndarray] = None
    calibration: Optional[CameraCalibration] = None
    source: str = "runtime"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeOutput:
    frame_id: int
    timestamp: float
    tracking_ok: bool
    pose: Any = None
    segmentation: Optional[dict] = None
    world_points: list[np.ndarray] = field(default_factory=list)
    semantic_objects: list[dict] = field(default_factory=list)
    diagnostics: Optional[dict] = None
