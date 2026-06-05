import numpy as np

from geometry.transforms.camera_models import (
    CameraIntrinsics
)


class PointCloudGenerator:

    def __init__(

        self,

        intrinsics: CameraIntrinsics
    ):

        self.intrinsics = intrinsics

    # --------------------------------------------------------
    # Generate point cloud
    # --------------------------------------------------------

    def generate_pointcloud(

        self,

        depth_frame,

        depth_min_m=0.1,
        depth_max_m=5.0,

        stride=4,
        depth_unit=0.001,
    ):

        depth = np.asarray(depth_frame)
        if depth.ndim != 2:
            raise ValueError("depth_frame must be a single-channel image")

        stride = max(int(stride), 1)
        sampled = depth[::stride, ::stride].astype(np.float32) * float(depth_unit)

        v_coords, u_coords = np.indices(sampled.shape, dtype=np.float32)
        u = u_coords * stride
        v = v_coords * stride

        valid = (
            np.isfinite(sampled)
            & (sampled >= depth_min_m)
            & (sampled <= depth_max_m)
            & (sampled > 0.0)
        )

        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        z = sampled[valid]
        x = (u[valid] - self.intrinsics.cx) * z / self.intrinsics.fx
        y = (v[valid] - self.intrinsics.cy) * z / self.intrinsics.fy
        return np.column_stack([x, y, z]).astype(np.float32)
