
import numpy as np


class WorldMap:
    """
    Bounded world-frame semantic point map.

    The original map appended every point forever. That is acceptable for
    reference validation, but robot runtime needs a fixed memory policy.
    """

    def __init__(
        self,
        voxel_size_m: float = 0.05,
        max_points: int = 250000,
    ):
        self.voxel_size_m = float(voxel_size_m)
        self.max_points = int(max_points)
        self._points = np.empty((0, 3), dtype=np.float32)

    def add_points(self, points_world: np.ndarray):
        if points_world is None or len(points_world) == 0:
            return

        points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return

        if len(self._points) == 0:
            merged = points
        else:
            merged = np.vstack([self._points, points])

        self._points = self._voxel_downsample(merged)
        if len(self._points) > self.max_points:
            self._points = self._points[-self.max_points :]

    def get_all_points(self):
        return self._points.copy()

    def clear(self):
        self._points = np.empty((0, 3), dtype=np.float32)

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        if self.voxel_size_m <= 0.0 or len(points) == 0:
            return points.astype(np.float32, copy=False)

        voxels = np.floor(points / self.voxel_size_m).astype(np.int64)
        _, unique_idx = np.unique(voxels, axis=0, return_index=True)
        unique_idx.sort()
        return points[unique_idx].astype(np.float32, copy=False)
