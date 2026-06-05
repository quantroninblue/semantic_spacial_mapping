from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .config import RuntimeConfig


@dataclass
class InstanceMask:
    mask: np.ndarray
    class_id: int = -1
    label: str = "object"
    confidence: float = 1.0
    bbox_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    area_px: int = 0
    depth_support_ratio: float = 0.0
    border_touch_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectGeometry:
    centroid: np.ndarray
    extent: np.ndarray
    covariance: np.ndarray
    num_points: int
    valid_depth_ratio: float


@dataclass
class SemanticFrame:
    timestamp: float
    frame_id: int
    instances: List[InstanceMask] = field(default_factory=list)
    overlay: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict:
        return {
            "overlay": self.overlay,
            "masks": [instance.mask for instance in self.instances],
            "class_ids": [instance.class_id for instance in self.instances],
            "labels": [instance.label for instance in self.instances],
            "confidences": [instance.confidence for instance in self.instances],
            "boxes": [instance.bbox_xyxy for instance in self.instances],
            "elapsed_ms": self.elapsed_ms,
        }


def semantic_frame_from_result(
    result: Optional[dict],
    frame_id: int,
    timestamp: float,
    config: RuntimeConfig,
    depth_frame: Optional[np.ndarray] = None,
) -> SemanticFrame:
    if result is None:
        result = {}

    masks = result.get("masks", [])
    class_ids = _sequence_or_default(result.get("class_ids"), len(masks), -1)
    labels = _sequence_or_default(result.get("labels"), len(masks), "object")
    confidences = _sequence_or_default(result.get("confidences"), len(masks), 1.0)
    boxes = _sequence_or_default(result.get("boxes"), len(masks), None)

    instances: list[InstanceMask] = []
    for idx, mask in enumerate(masks):
        instance = build_instance_mask(
            mask=mask,
            class_id=int(class_ids[idx]),
            label=str(labels[idx]),
            confidence=float(confidences[idx]),
            bbox=boxes[idx],
            depth_frame=depth_frame,
            config=config,
        )
        if instance is not None:
            instances.append(instance)
        if len(instances) >= config.segmentation.max_masks_per_frame:
            break

    return SemanticFrame(
        timestamp=timestamp,
        frame_id=frame_id,
        instances=instances,
        overlay=result.get("overlay"),
        elapsed_ms=float(result.get("elapsed_ms", 0.0)),
        metadata={k: v for k, v in result.items() if k not in {"overlay", "masks"}},
    )


def build_instance_mask(
    mask: np.ndarray,
    class_id: int,
    label: str,
    confidence: float,
    bbox,
    depth_frame: Optional[np.ndarray],
    config: RuntimeConfig,
) -> Optional[InstanceMask]:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    area_px = int(np.count_nonzero(mask_u8))
    if area_px < config.segmentation.minimum_mask_area:
        return None

    bbox_xyxy = _bbox_from_mask(mask_u8) if bbox is None else tuple(int(v) for v in bbox)
    border_touch_ratio = _border_touch_ratio(mask_u8)
    if border_touch_ratio > config.segmentation.max_border_touch_ratio:
        return None

    depth_support_ratio = _depth_support_ratio(mask_u8, depth_frame, config)
    if (
        depth_frame is not None
        and depth_support_ratio < config.segmentation.min_depth_support_ratio
    ):
        return None

    return InstanceMask(
        mask=mask_u8,
        class_id=class_id,
        label=label,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        area_px=area_px,
        depth_support_ratio=depth_support_ratio,
        border_touch_ratio=border_touch_ratio,
    )


def compute_object_geometry(
    points: np.ndarray,
    valid_depth_ratio: float,
) -> Optional[ObjectGeometry]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None

    lower = np.percentile(points, 2.0, axis=0)
    upper = np.percentile(points, 98.0, axis=0)
    clipped = points[np.all((points >= lower) & (points <= upper), axis=1)]
    if len(clipped) == 0:
        clipped = points

    centroid = np.median(clipped, axis=0).astype(np.float32)
    extent = (np.max(clipped, axis=0) - np.min(clipped, axis=0)).astype(np.float32)
    covariance = (
        np.cov(clipped.T).astype(np.float32)
        if len(clipped) >= 3
        else np.eye(3, dtype=np.float32) * 1e-6
    )
    return ObjectGeometry(
        centroid=centroid,
        extent=extent,
        covariance=covariance,
        num_points=int(len(clipped)),
        valid_depth_ratio=float(valid_depth_ratio),
    )


def filter_point_outliers(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 8:
        return points
    centroid = np.median(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    cutoff = np.percentile(distances, 95.0)
    return points[distances <= cutoff]


def _sequence_or_default(values, length: int, default):
    if values is None:
        return [default for _ in range(length)]
    values = list(values)
    if len(values) < length:
        values.extend(default for _ in range(length - len(values)))
    return values


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _border_touch_ratio(mask: np.ndarray) -> float:
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    border = (
        np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
    )
    return float(border) / float(area)


def _depth_support_ratio(
    mask: np.ndarray,
    depth_frame: Optional[np.ndarray],
    config: RuntimeConfig,
) -> float:
    if depth_frame is None:
        return 0.0
    depth = np.asarray(depth_frame)
    if depth.shape[:2] != mask.shape[:2]:
        return 0.0
    depth_m = depth.astype(np.float32) * float(config.depth.depth_unit)
    occupied = mask > 0
    area = int(np.count_nonzero(occupied))
    if area == 0:
        return 0.0
    valid = (
        occupied
        & np.isfinite(depth_m)
        & (depth_m >= config.depth.min_m)
        & (depth_m <= config.depth.max_m)
        & (depth_m > 0.0)
    )
    return float(np.count_nonzero(valid)) / float(area)
