from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SemanticObject:
    object_id: int
    class_id: int
    label: str
    confidence: float
    centroid: np.ndarray
    extent: np.ndarray
    covariance: np.ndarray
    points: np.ndarray
    first_seen: float
    last_seen: float
    observations: int = 1

    def as_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "class_id": self.class_id,
            "label": self.label,
            "confidence": float(self.confidence),
            "centroid": self.centroid.tolist(),
            "extent": self.extent.tolist(),
            "first_seen": float(self.first_seen),
            "last_seen": float(self.last_seen),
            "observations": int(self.observations),
            "num_points": int(len(self.points)),
        }


class SemanticObjectMap:
    def __init__(
        self,
        association_distance_m: float = 0.75,
        max_objects: int = 512,
        max_points_per_object: int = 5000,
        voxel_size_m: float = 0.05,
    ):
        self.association_distance_m = float(association_distance_m)
        self.max_objects = int(max_objects)
        self.max_points_per_object = int(max_points_per_object)
        self.voxel_size_m = float(voxel_size_m)
        self._objects: list[SemanticObject] = []
        self._next_id = 1

    def update(
        self,
        class_id: int,
        label: str,
        confidence: float,
        centroid: np.ndarray,
        extent: np.ndarray,
        covariance: np.ndarray,
        points: np.ndarray,
        timestamp: float,
    ) -> SemanticObject:
        points = self._downsample(points)
        match = self._find_match(class_id, label, centroid)
        if match is None:
            match = SemanticObject(
                object_id=self._next_id,
                class_id=class_id,
                label=label,
                confidence=float(confidence),
                centroid=np.asarray(centroid, dtype=np.float32),
                extent=np.asarray(extent, dtype=np.float32),
                covariance=np.asarray(covariance, dtype=np.float32),
                points=points,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            self._next_id += 1
            self._objects.append(match)
            self._enforce_object_limit()
            return match

        n = float(match.observations)
        alpha = 1.0 / (n + 1.0)
        match.centroid = ((1.0 - alpha) * match.centroid + alpha * centroid).astype(np.float32)
        match.extent = np.maximum(match.extent, extent).astype(np.float32)
        match.covariance = ((1.0 - alpha) * match.covariance + alpha * covariance).astype(np.float32)
        match.confidence = max(float(match.confidence), float(confidence))
        match.points = self._downsample(np.vstack([match.points, points]))
        if len(match.points) > self.max_points_per_object:
            match.points = match.points[-self.max_points_per_object :]
        match.last_seen = timestamp
        match.observations += 1
        return match

    def objects(self) -> list[SemanticObject]:
        return list(self._objects)

    def as_dicts(self) -> list[dict]:
        return [obj.as_dict() for obj in self._objects]

    def clear(self) -> None:
        self._objects = []
        self._next_id = 1

    def _find_match(
        self,
        class_id: int,
        label: str,
        centroid: np.ndarray,
    ) -> Optional[SemanticObject]:
        candidates = [
            obj
            for obj in self._objects
            if obj.class_id == class_id or (class_id < 0 and obj.label == label)
        ]
        if not candidates:
            return None
        distances = [float(np.linalg.norm(obj.centroid - centroid)) for obj in candidates]
        best_idx = int(np.argmin(distances))
        if distances[best_idx] > self.association_distance_m:
            return None
        return candidates[best_idx]

    def _downsample(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0 or self.voxel_size_m <= 0.0:
            return points
        voxels = np.floor(points / self.voxel_size_m).astype(np.int64)
        _, unique_idx = np.unique(voxels, axis=0, return_index=True)
        unique_idx.sort()
        return points[unique_idx].astype(np.float32, copy=False)

    def _enforce_object_limit(self) -> None:
        if len(self._objects) <= self.max_objects:
            return
        self._objects.sort(key=lambda obj: obj.last_seen)
        self._objects = self._objects[-self.max_objects :]
