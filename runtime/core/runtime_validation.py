from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry.transforms.camera_models import CameraIntrinsics

from .config import load_runtime_config
from .pose_providers import StaticIdentityPoseProvider
from .segmentation import MockSegmentationProvider
from .semantic_runtime import DeploymentSemanticRuntime
from .types import CameraCalibration, FramePacket


def run_validation() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for config_name in ("gazebo.yaml", "embedded_oakd.yaml"):
        config = load_runtime_config(
            repo_root / "ros" / "semantic_spatial_mapping_ros" / "config" / config_name
        )
        assert config.profile

    config = load_runtime_config(
        repo_root / "ros" / "semantic_spatial_mapping_ros" / "config" / "embedded_oakd.yaml"
    )
    config.segmentation.enabled = True
    config.segmentation.minimum_mask_area = 10
    config.depth.stride = 2
    config.mapping.max_points = 500

    runtime = DeploymentSemanticRuntime(
        config=config,
        pose_provider=StaticIdentityPoseProvider(),
        segmentation_module=MockSegmentationProvider(),
    )

    intr = CameraIntrinsics(
        fx=60.0,
        fy=60.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
    )
    calibration = CameraCalibration(
        rgb_intrinsics=intr,
        depth_intrinsics=intr,
        T_depth_rgb=np.eye(4, dtype=np.float32),
    )
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 1000, dtype=np.uint16)

    output = runtime.update(
        FramePacket(
            frame_id=0,
            timestamp=1.0,
            rgb_frame=rgb,
            depth_frame=depth,
            calibration=calibration,
            source="validation",
        )
    )

    assert output.tracking_ok
    assert output.world_points
    assert len(runtime.latest_map()) > 0
    assert len(runtime.latest_map()) <= config.mapping.max_points
    assert output.diagnostics["health"] == "OK"


if __name__ == "__main__":
    run_validation()
    print("runtime validation passed")
