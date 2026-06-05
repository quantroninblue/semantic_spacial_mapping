from __future__ import annotations

import unittest

import numpy as np

from motion.vo.camera import CameraModel
from motion.vo.features import MatchResult
from motion.vo.pipeline import VOConfig, VisualOdometry


class RgbdVoTests(unittest.TestCase):
    def test_rgbd_pnp_recovers_metric_relative_motion(self):
        camera = CameraModel(
            fx=120.0,
            fy=120.0,
            cx=64.0,
            cy=48.0,
            width=128,
            height=96,
        )
        config = VOConfig(
            min_inliers=8,
            rgbd_min_depth_features=8,
            pnp_reproj_thresh=2.0,
            scale_mode="rgbd",
        )
        vo = VisualOdometry(camera=camera, config=config)

        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.25, 0.25, 4)
        points_ref = []
        for x in xs:
            for y in ys:
                points_ref.append([x, y, 2.0 + 0.2 * x])
        points_ref = np.asarray(points_ref, dtype=np.float64)

        pts_ref = camera.project(points_ref).astype(np.float32)
        T_cur_ref = np.eye(4)
        T_cur_ref[:3, 3] = [0.12, -0.02, 0.05]
        points_cur = (T_cur_ref[:3, :3] @ points_ref.T).T + T_cur_ref[:3, 3]
        pts_cur = camera.project(points_cur).astype(np.float32)

        depth = np.zeros((camera.height, camera.width), dtype=np.uint16)
        for pixel, point in zip(pts_ref, points_ref):
            u, v = np.rint(pixel).astype(int)
            if 0 <= u < camera.width and 0 <= v < camera.height:
                depth[v, u] = int(point[2] * 1000.0)

        vo._last_kf_depth = depth
        match_result = MatchResult(
            idx_ref=np.arange(len(pts_ref), dtype=np.int32),
            idx_cur=np.arange(len(pts_ref), dtype=np.int32),
            pts_ref=pts_ref,
            pts_cur=pts_cur,
            distances=np.zeros(len(pts_ref), dtype=np.float32),
        )

        motion = vo._estimate_rgbd_motion(match_result)
        self.assertIsNotNone(motion)
        self.assertEqual(int(np.count_nonzero(motion["inlier_mask"])), len(pts_ref))
        np.testing.assert_allclose(
            motion["T_cur_ref"][:3, 3],
            T_cur_ref[:3, 3],
            atol=0.03,
        )
        self.assertLess(motion["reproj_error_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
