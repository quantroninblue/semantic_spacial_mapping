import numpy as np

from geometry.transforms.depth_to_rgb_projection import (
    DepthToRGBProjector
)


class ObjectPointCloudExtractor:

    def __init__(

        self,

        pointcloud_generator,

        projector: DepthToRGBProjector
    ):

        self.generator = (
            pointcloud_generator
        )

        self.projector = (
            projector
        )

    # ========================================================
    # Reprojection-aware semantic point cloud extraction
    # ========================================================

    def extract_object_pointcloud(

        self,

        depth_frame,

        segmentation_mask,

        depth_min_m=0.1,
        depth_max_m=5.0,

        stride=4,
        depth_unit=0.001,
    ):

        intr = (
            self.generator.intrinsics
        )

        mask_h, mask_w = (
            segmentation_mask.shape
        )

        depth = np.asarray(depth_frame)
        if depth.ndim != 2:
            raise ValueError("depth_frame must be a single-channel image")

        stride = max(int(stride), 1)
        sampled = depth[::stride, ::stride].astype(np.float32) * float(depth_unit)
        v_grid, u_grid = np.indices(sampled.shape, dtype=np.float32)
        u_depth = u_grid * stride
        v_depth = v_grid * stride

        valid = (
            np.isfinite(sampled)
            & (sampled >= depth_min_m)
            & (sampled <= depth_max_m)
            & (sampled > 0.0)
        )
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        z = sampled[valid]
        u = u_depth[valid]
        v = v_depth[valid]
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        points_depth = np.column_stack([x, y, z]).astype(np.float32)

        points_rgb = self.projector.transform_depth_to_rgb_points(points_depth)
        pixels = self.projector.project_points_to_rgb(points_rgb)
        in_bounds = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < mask_w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < mask_h)
        )
        if not np.any(in_bounds):
            return np.empty((0, 3), dtype=np.float32)

        pixels_in = pixels[in_bounds]
        points_in = points_rgb[in_bounds]
        occupied = segmentation_mask[pixels_in[:, 1], pixels_in[:, 0]] > 0
        return points_in[occupied].astype(np.float32)

    # ========================================================
    # Geometry statistics
    # ========================================================

    def compute_geometry_stats(

        self,

        pointcloud
    ):

        if len(pointcloud) == 0:

            return None

        centroid = np.mean(
            pointcloud,
            axis=0
        )

        min_xyz = np.min(
            pointcloud,
            axis=0
        )

        max_xyz = np.max(
            pointcloud,
            axis=0
        )

        dimensions = (
            max_xyz - min_xyz
        )

        stats = {

            "point_count": len(pointcloud),

            "centroid": centroid,

            "dimensions": dimensions,

            "min_xyz": min_xyz,

            "max_xyz": max_xyz
        }

        return stats
