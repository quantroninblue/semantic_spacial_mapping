from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import SegmentationConfig


class SegmentationProvider(Protocol):
    def segment(self, rgb_frame) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DisabledSegmentationProvider:
    def segment(self, rgb_frame) -> dict:
        return {
            "overlay": rgb_frame,
            "masks": [],
            "obbs": [],
            "elapsed_ms": 0.0,
        }


class MockSegmentationProvider:
    def segment(self, rgb_frame) -> dict:
        mask = np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
        h, w = mask.shape
        y0, y1 = h // 4, max(h // 4 + 1, 3 * h // 4)
        x0, x1 = w // 4, max(w // 4 + 1, 3 * w // 4)
        mask[y0:y1, x0:x1] = 255
        return {
            "overlay": rgb_frame.copy(),
            "masks": [mask],
            "obbs": [],
            "elapsed_ms": 0.0,
        }


class YoloSegmentationProvider:
    def __init__(self, config: SegmentationConfig):
        from segmentation.segmentation_reference import SegmentationModule

        self.module = SegmentationModule(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            minimum_mask_area=config.minimum_mask_area,
        )

    def segment(self, rgb_frame) -> dict:
        return self.module.segment(rgb_frame)


def build_segmentation_provider(config: SegmentationConfig) -> SegmentationProvider:
    if not config.enabled:
        return DisabledSegmentationProvider()

    backend = config.backend.lower()
    if backend == "disabled":
        return DisabledSegmentationProvider()
    if backend == "mock":
        return MockSegmentationProvider()
    if backend == "yolo":
        return YoloSegmentationProvider(config)
    raise ValueError(f"Unsupported segmentation backend: {config.backend}")
