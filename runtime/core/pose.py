from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class PoseProviderError(RuntimeError):
    pass


@dataclass
class PoseEstimate:
    success: bool
    timestamp: float
    T_world_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    source: str = "unknown"
    contract: str = "T_world_cam"
    frame_id: str = "map"
    child_frame_id: str = "camera_link"
    covariance: Optional[np.ndarray] = None
    status: str = "OK"

    @property
    def T_cam_world(self) -> np.ndarray:
        R = self.T_world_cam[:3, :3]
        t = self.T_world_cam[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T


class PoseProvider:
    source_name = "base"

    def update(self, frame_packet) -> PoseEstimate:
        raise NotImplementedError

    def latest(self) -> Optional[PoseEstimate]:
        return None

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.reset()
