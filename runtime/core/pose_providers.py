from __future__ import annotations

from typing import Optional

import numpy as np

from .config import RuntimeConfig
from .pose import PoseEstimate, PoseProvider


class StaticIdentityPoseProvider(PoseProvider):
    source_name = "identity"

    def update(self, frame_packet) -> PoseEstimate:
        return PoseEstimate(
            success=True,
            timestamp=frame_packet.timestamp,
            T_world_cam=np.eye(4),
            source=self.source_name,
            status="DEGRADED_IDENTITY_POSE",
        )


class ExternalPoseProvider(PoseProvider):
    def __init__(self, source_name: str, max_age_sec: float = 0.25):
        self.source_name = source_name
        self.max_age_sec = float(max_age_sec)
        self._latest: Optional[PoseEstimate] = None

    def set_latest(self, pose: PoseEstimate) -> None:
        self._latest = pose

    def update(self, frame_packet) -> PoseEstimate:
        if self._latest is None:
            return PoseEstimate(
                success=False,
                timestamp=frame_packet.timestamp,
                source=self.source_name,
                status="NO_EXTERNAL_POSE",
            )
        age = frame_packet.timestamp - self._latest.timestamp
        if self.max_age_sec > 0.0 and age > self.max_age_sec:
            return PoseEstimate(
                success=False,
                timestamp=frame_packet.timestamp,
                source=self.source_name,
                status=f"STALE_EXTERNAL_POSE age_sec={age:.3f}",
            )
        return self._latest

    def latest(self) -> Optional[PoseEstimate]:
        return self._latest

    def reset(self) -> None:
        self._latest = None


class VisualOdometryPoseProvider(PoseProvider):
    source_name = "internal_vslam"

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.vo: Optional[VisualOdometry] = None
        self._latest: Optional[PoseEstimate] = None

    def update(self, frame_packet) -> PoseEstimate:
        if self.vo is None:
            if frame_packet.calibration is None or frame_packet.calibration.rgb_intrinsics is None:
                return PoseEstimate(
                    success=False,
                    timestamp=frame_packet.timestamp,
                    source=self.source_name,
                    status="WAITING_FOR_CAMERA_INFO",
                )
            from motion.vo.camera import CameraModel
            from motion.vo.pipeline import VOConfig, VisualOdometry

            intr = frame_packet.calibration.rgb_intrinsics
            camera = CameraModel(
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.cx,
                cy=intr.cy,
                width=intr.width,
                height=intr.height,
            )
            vo_config = VOConfig(
                scale_mode=self.config.vslam.scale_mode,
            )
            self.vo = VisualOdometry(camera=camera, config=vo_config)

        update = self.vo.update(
            img=frame_packet.rgb_frame,
            timestamp=frame_packet.timestamp,
            depth_frame=frame_packet.depth_frame,
        )
        pose = PoseEstimate(
            success=update.success,
            timestamp=update.timestamp,
            T_world_cam=update.T_world_cam,
            source=self.source_name,
            status=update.tracking_state,
        )
        self._latest = pose
        return pose

    def latest(self) -> Optional[PoseEstimate]:
        return self._latest

    def reset(self) -> None:
        if self.vo is not None:
            self.vo.reset()
        self._latest = None
