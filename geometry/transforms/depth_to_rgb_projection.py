"""
depth_to_rgb_projection.py

Canonical RGBD reprojection subsystem.

Pipeline
--------
Depth Pixel
    ->
Depth Camera 3D Point
    ->
Extrinsic Transform
    ->
RGB Camera 3D Point
    ->
RGB Image Projection

This module becomes the canonical runtime
projection layer for:
- replay validation
- semantic fusion
- point cloud overlays
- grasp estimation
- future SLAM integration
"""

import numpy as np

from geometry.transforms.camera_models import (
    CameraIntrinsics
)

from geometry.transforms.extrinsics import (
    DEPTH_TO_RGB_EXTRINSIC
)


class DepthToRGBProjector:

    def __init__(

        self,

        depth_intrinsics: CameraIntrinsics,

        rgb_intrinsics: CameraIntrinsics,

        extrinsic_transform=(
            DEPTH_TO_RGB_EXTRINSIC
        )
    ):

        self.depth_intrinsics = (
            depth_intrinsics
        )

        self.rgb_intrinsics = (
            rgb_intrinsics
        )

        self.extrinsic_transform = (
            extrinsic_transform
        )

    # ========================================================
    # Depth Pixel -> Depth Camera 3D
    # ========================================================

    def depth_pixel_to_3d(

        self,

        u,
        v,

        depth_m
    ):

        x = (

            (u - self.depth_intrinsics.cx) *

            depth_m /

            self.depth_intrinsics.fx
        )

        y = (

            (v - self.depth_intrinsics.cy) *

            depth_m /

            self.depth_intrinsics.fy
        )

        z = depth_m

        return np.array(

            [x, y, z],

            dtype=np.float32
        )

    # ========================================================
    # Depth Camera Frame -> RGB Camera Frame
    # ========================================================

    def transform_depth_to_rgb_frame(

        self,

        point_3d
    ):

        transformed = (

            self.extrinsic_transform
            .transform_point(
                point_3d
            )
        )

        return transformed

    def transform_depth_to_rgb_points(

        self,

        points_3d
    ):

        points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return points
        return self.extrinsic_transform.transform_pointcloud(points)

    # ========================================================
    # RGB Camera 3D -> RGB Image Plane
    # ========================================================

    def project_3d_to_rgb(

        self,

        point_3d
    ):

        x, y, z = point_3d

        # ----------------------------------------------------
        # Invalid depth
        # ----------------------------------------------------

        if z <= 0.0:

            return None

        # ----------------------------------------------------
        # RGB reprojection
        # ----------------------------------------------------

        u_rgb = (

            x *

            self.rgb_intrinsics.fx /

            z +

            self.rgb_intrinsics.cx
        )

        v_rgb = (

            y *

            self.rgb_intrinsics.fy /

            z +

            self.rgb_intrinsics.cy
        )

        return (

            int(round(u_rgb)),

            int(round(v_rgb))
        )

    def project_points_to_rgb(

        self,

        points_3d
    ):

        points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)

        z = points[:, 2]
        valid_z = z > 0.0
        u = np.full(len(points), -1, dtype=np.int32)
        v = np.full(len(points), -1, dtype=np.int32)
        u[valid_z] = np.rint(
            points[valid_z, 0] * self.rgb_intrinsics.fx / z[valid_z]
            + self.rgb_intrinsics.cx
        ).astype(np.int32)
        v[valid_z] = np.rint(
            points[valid_z, 1] * self.rgb_intrinsics.fy / z[valid_z]
            + self.rgb_intrinsics.cy
        ).astype(np.int32)
        return np.column_stack([u, v])

    # ========================================================
    # Full RGBD reprojection
    # ========================================================

    def depth_pixel_to_rgb_pixel(

        self,

        u_depth,
        v_depth,

        depth_m
    ):

        # ----------------------------------------------------
        # Depth pixel -> depth camera 3D
        # ----------------------------------------------------

        point_depth_frame = (

            self.depth_pixel_to_3d(

                u=u_depth,
                v=v_depth,

                depth_m=depth_m
            )
        )

        # ----------------------------------------------------
        # Transform into RGB frame
        # ----------------------------------------------------

        point_rgb_frame = (

            self.transform_depth_to_rgb_frame(
                point_depth_frame
            )
        )

        # ----------------------------------------------------
        # Project into RGB image
        # ----------------------------------------------------

        rgb_pixel = (

            self.project_3d_to_rgb(
                point_rgb_frame
            )
        )

        return rgb_pixel

    # ========================================================
    # Batch reprojection
    # ========================================================

    def pointcloud_to_rgb_pixels(

        self,

        pointcloud
    ):

        projected_pixels = []

        if len(pointcloud) == 0:

            return projected_pixels

        for point in pointcloud:

            rgb_pixel = (

                self.project_3d_to_rgb(
                    point
                )
            )

            if rgb_pixel is None:
                continue

            projected_pixels.append(
                rgb_pixel
            )

        return projected_pixels
