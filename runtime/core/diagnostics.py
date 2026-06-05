from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict


class RuntimeHealth(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


@dataclass
class RuntimeDiagnostics:
    health: RuntimeHealth = RuntimeHealth.DEGRADED
    rgb_received: bool = False
    depth_received: bool = False
    rgb_camera_info_received: bool = False
    depth_camera_info_received: bool = False
    pose_source_active: bool = False
    segmentation_active: bool = False
    tracking_ok: bool = False
    camera_info_ready: bool = False
    camera_info_stale: bool = False
    dropped_frames: int = 0
    sync_failures: int = 0
    processed_frames: int = 0
    map_points: int = 0
    last_latency_ms: float = 0.0
    pose_age_sec: float = 0.0
    rgb_age_sec: float = 0.0
    depth_age_sec: float = 0.0
    segmentation_ms: float = 0.0
    mapping_ms: float = 0.0
    masks_processed: int = 0
    points_generated: int = 0
    semantic_objects: int = 0
    last_error: str = ""
    messages: Dict[str, str] = field(default_factory=dict)

    def recompute_health(self) -> None:
        required = [
            self.rgb_received,
            self.pose_source_active,
            self.tracking_ok,
        ]
        if all(required):
            self.health = RuntimeHealth.OK
        elif self.rgb_received:
            self.health = RuntimeHealth.DEGRADED
        else:
            self.health = RuntimeHealth.ERROR

    def as_dict(self) -> dict:
        data = asdict(self)
        data["health"] = self.health.value
        return data
