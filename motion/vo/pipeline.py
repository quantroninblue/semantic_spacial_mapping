
"""
pipeline.py
-----------
Subsystem-oriented monocular visual odometry frontend.

This module is intentionally designed as a reusable motion-estimation
subsystem inside a larger semantic-spatial runtime.

Core Responsibilities
---------------------
1. Feature detection + matching
2. Relative motion estimation
3. Monocular pose accumulation
4. Sparse triangulation
5. Keyframe management
6. Pose-state exposure to external runtimes

This module DOES NOT own:
- application runtime orchestration
- visualization ownership
- semantic mapping
- tracking pipelines
- replay pipelines
- world-state accumulation

Those are handled externally by the parent runtime.

Architecture
------------
RGB Frame
    ↓
VO.update()
    ↓
PoseUpdate
    ↓
External Runtime
    ↓
World-frame semantic accumulation
"""

from __future__ import annotations

import cv2
import numpy as np
import time

from dataclasses import dataclass, field
from enum import Enum, auto

from typing import (
    Callable,
    List,
    Optional,
)

from .camera import CameraModel

from .features import (
    DetectorType,
    MatcherType,
    FeatureDetector,
    FeatureMatcher,
    FrameFeatures,
)

from .motion import (
    MotionEstimator,
    PoseEstimate,
    compose_pose,
    invert_pose,
)

from .triangulation import (
    Triangulator,
    MapPoint,
)

from .keyframe import (
    Keyframe,
    KeyframeSelector,
)

from .covisibility import CovisibilityGraph


# ═══════════════════════════════════════════════════════════════════════ #
#  Enums                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

class TrackingMode(Enum):

    DESCRIPTOR = auto()

    OPTICAL_FLOW = auto()


class VOState(Enum):

    NOT_INIT = auto()

    OK = auto()

    LOST = auto()


# ═══════════════════════════════════════════════════════════════════════ #
#  Configuration                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class VOConfig:

    # Feature detection
    detector_type: DetectorType = DetectorType.ORB

    max_features: int = 2000

    grid_rows: int = 4

    grid_cols: int = 4

    # Feature matching
    matcher_type: MatcherType = MatcherType.BF_HAMMING

    ratio_thresh: float = 0.75

    tracking_mode: TrackingMode = TrackingMode.DESCRIPTOR

    # Motion estimation
    ransac_thresh: float = 1.0

    ransac_prob: float = 0.999

    min_inliers: int = 20

    rgbd_min_depth_features: int = 12

    pnp_reproj_thresh: float = 3.0

    pnp_refine: bool = True

    max_pose_step_m: float = 50.0

    # Monocular scale
    scale_mode: str = "median_depth"

    fixed_scale: float = 1.0

    # Triangulation
    max_reproj_err: float = 2.0

    min_parallax_deg: float = 1.0

    min_depth: float = 0.1

    max_depth: float = 200.0

    # Keyframe insertion
    kf_min_parallax: float = 2.0

    kf_max_feat_ratio: float = 0.75

    kf_max_rot_deg: float = 15.0

    kf_min_frames: int = 3

    kf_max_frames: int = 20

    # Storage
    store_images: bool = False


# ═══════════════════════════════════════════════════════════════════════ #
#  Diagnostics                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class FrameStats:

    frame_id: int = 0

    num_detected: int = 0

    num_matched: int = 0

    num_inliers: int = 0

    num_map_pts: int = 0

    is_keyframe: bool = False

    kf_reason: str = ""

    h_score: float = 0.0

    process_ms: float = 0.0

    state: str = "OK"

    tracking_method: str = ""

    depth_support_ratio: float = 0.0

    reproj_error_px: float = 0.0


# ═══════════════════════════════════════════════════════════════════════ #
#  Pose Update                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class PoseUpdate:

    success: bool = False

    frame_id: int = 0

    timestamp: float = 0.0

    T_world_cam: np.ndarray = field(
        default_factory=lambda: np.eye(4)
    )

    T_cam_world: np.ndarray = field(
        default_factory=lambda: np.eye(4)
    )

    translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )

    rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )

    num_inliers: int = 0

    tracking_state: str = "NOT_INIT"

    is_keyframe: bool = False

    process_ms: float = 0.0

    tracking_method: str = ""

    depth_support_ratio: float = 0.0

    reproj_error_px: float = 0.0


# ═══════════════════════════════════════════════════════════════════════ #
#  Pose Graph                                                            #
# ═══════════════════════════════════════════════════════════════════════ #

class PoseGraph:

    """
    Lightweight absolute pose accumulator.

    Stores:
        T_world_cam

    for each processed frame.

    Future extensions:
    - loop closure
    - graph optimization
    - backend correction
    """

    def __init__(self):

        self._poses: List[np.ndarray] = []

    def add(
        self,
        T_world_cam: np.ndarray
    ):

        self._poses.append(
            T_world_cam.copy()
        )

    def update(
        self,
        frame_id: int,
        T_world_cam: np.ndarray
    ):

        if frame_id < len(self._poses):

            self._poses[frame_id] = (
                T_world_cam.copy()
            )

    @property
    def poses(self) -> List[np.ndarray]:

        return self._poses

    @property
    def positions(self) -> np.ndarray:

        if not self._poses:

            return np.empty((0, 3))

        return np.array([
            T[:3, 3]
            for T in self._poses
        ])

    def __len__(self):

        return len(self._poses)


# ═══════════════════════════════════════════════════════════════════════ #
#  Visual Odometry Frontend                                              #
# ═══════════════════════════════════════════════════════════════════════ #

class VisualOdometry:

    """
    Reusable monocular VO frontend subsystem.

    This class exposes:
    - persistent camera pose
    - motion estimation
    - sparse map generation
    - keyframe management

    It intentionally does NOT own:
    - visualization
    - replay orchestration
    - semantic mapping
    - runtime scheduling
    """

    def __init__(
        self,
        camera: CameraModel,
        config: Optional[VOConfig] = None,
    ):

        self.camera = camera

        self.cfg = config or VOConfig()

        # ───────────────────────────────────────────────────────────── #
        #  Subsystems
        # ───────────────────────────────────────────────────────────── #

        self.detector = FeatureDetector(

            detector_type=self.cfg.detector_type,

            max_features=self.cfg.max_features,

            grid_rows=self.cfg.grid_rows,

            grid_cols=self.cfg.grid_cols,
        )

        self.matcher = FeatureMatcher(

            matcher_type=self.cfg.matcher_type,

            ratio_thresh=self.cfg.ratio_thresh,
        )

        self.estimator = MotionEstimator(

            camera=camera,

            ransac_thresh=self.cfg.ransac_thresh,

            ransac_prob=self.cfg.ransac_prob,

            min_inliers=self.cfg.min_inliers,
        )

        self.triangulator = Triangulator(

            camera=camera,

            max_reproj_err=self.cfg.max_reproj_err,

            min_depth=self.cfg.min_depth,

            max_depth=self.cfg.max_depth,

            min_parallax=self.cfg.min_parallax_deg,
        )

        self.kf_selector = KeyframeSelector(

            min_parallax_deg=self.cfg.kf_min_parallax,

            max_feature_ratio=self.cfg.kf_max_feat_ratio,

            max_rotation_deg=self.cfg.kf_max_rot_deg,

            min_frames=self.cfg.kf_min_frames,

            max_frames=self.cfg.kf_max_frames,
        )

        self.pose_graph = PoseGraph()

        self.covis_graph = CovisibilityGraph(
            min_shared=15
        )

        # ───────────────────────────────────────────────────────────── #
        #  Runtime State
        # ───────────────────────────────────────────────────────────── #

        self.state: VOState = VOState.NOT_INIT

        self.frame_id: int = 0

        self.kf_id: int = 0

        self.keyframes: List[Keyframe] = []

        self.map_points: List[MapPoint] = []

        self.T_world_cam: np.ndarray = np.eye(4)

        self._last_kf: Optional[Keyframe] = None

        self._last_gray: Optional[np.ndarray] = None

        self._last_features: Optional[
            FrameFeatures
        ] = None

        self._last_depth: Optional[np.ndarray] = None

        self._last_kf_depth: Optional[np.ndarray] = None

        # ───────────────────────────────────────────────────────────── #
        #  External Runtime Hooks
        # ───────────────────────────────────────────────────────────── #

        self.on_new_keyframe: Optional[
            Callable[[Keyframe], None]
        ] = None

        self.on_pose_update: Optional[
            Callable[[PoseUpdate], None]
        ] = None

        # ───────────────────────────────────────────────────────────── #
        #  Diagnostics
        # ───────────────────────────────────────────────────────────── #

        self.stats_history: List[
            FrameStats
        ] = []

    # ═══════════════════════════════════════════════════════════════ #
    #  Public API                                                     #
    # ═══════════════════════════════════════════════════════════════ #

    def update(
        self,
        img: np.ndarray,
        timestamp: float = 0.0,
        depth_frame: Optional[np.ndarray] = None,
    ) -> PoseUpdate:

        """
        Update VO state using a new frame.

        Returns
        -------
        PoseUpdate
            Current world-frame camera pose and
            motion-estimation metadata.
        """

        t0 = time.perf_counter()

        gray = self._to_gray(img)

        stats = FrameStats(
            frame_id=self.frame_id
        )

        if self.state == VOState.NOT_INIT:

            stats = self._initialize(

                gray=gray,

                img=img,

                timestamp=timestamp,

                stats=stats,

                depth_frame=depth_frame,
            )

            self._last_depth = (
                None
                if depth_frame is None
                else depth_frame.copy()
            )

        else:

            stats = self._track(

                gray=gray,

                img=img,

                timestamp=timestamp,

                stats=stats,

                depth_frame=depth_frame,
            )

        stats.process_ms = (
            time.perf_counter() - t0
        ) * 1000

        stats.state = self.state.name

        self.stats_history.append(stats)

        pose_update = PoseUpdate(

            success=(
                self.state == VOState.OK
            ),

            frame_id=self.frame_id,

            timestamp=timestamp,

            T_world_cam=self.T_world_cam.copy(),

            T_cam_world=invert_pose(
                self.T_world_cam
            ),

            translation=self.T_world_cam[
                :3, 3
            ].copy(),

            rotation=self.T_world_cam[
                :3, :3
            ].copy(),

            num_inliers=stats.num_inliers,

            tracking_state=self.state.name,

            is_keyframe=stats.is_keyframe,

            process_ms=stats.process_ms,

            tracking_method=stats.tracking_method,

            depth_support_ratio=stats.depth_support_ratio,

            reproj_error_px=stats.reproj_error_px,
        )

        if self.on_pose_update:

            self.on_pose_update(
                pose_update
            )

        self.frame_id += 1

        return pose_update

    # ═══════════════════════════════════════════════════════════════ #
    #  Reset                                                          #
    # ═══════════════════════════════════════════════════════════════ #

    def reset(self):

        self.state = VOState.NOT_INIT

        self.frame_id = 0

        self.kf_id = 0

        self.keyframes = []

        self.map_points = []

        self.T_world_cam = np.eye(4)

        self._last_kf = None

        self._last_gray = None

        self._last_features = None

        self._last_depth = None

        self._last_kf_depth = None

        self.pose_graph = PoseGraph()

        self.covis_graph = CovisibilityGraph(
            min_shared=15
        )

        self.stats_history = []

        self.kf_selector.reset()

    # ═══════════════════════════════════════════════════════════════ #
    #  Properties                                                     #
    # ═══════════════════════════════════════════════════════════════ #

    @property
    def trajectory(self) -> np.ndarray:

        return self.pose_graph.positions

    @property
    def current_pose(self) -> np.ndarray:

        return self.T_world_cam.copy()

    # ═══════════════════════════════════════════════════════════════ #
    #  Initialization                                                 #
    # ═══════════════════════════════════════════════════════════════ #

    def _initialize(
        self,
        gray: np.ndarray,
        img: np.ndarray,
        timestamp: float,
        stats: FrameStats,
        depth_frame: Optional[np.ndarray] = None,
    ) -> FrameStats:

        feats = self.detector.detect_and_compute(
            gray
        )

        stats.num_detected = len(feats)

        if len(feats) < 10:

            return stats

        self.T_world_cam = np.eye(4)

        self.pose_graph.add(
            self.T_world_cam
        )

        kf = Keyframe(

            frame_id=self.frame_id,

            kf_id=self.kf_id,

            T_world_cam=self.T_world_cam.copy(),

            features=feats,

            timestamp=timestamp,

            image=(
                gray.copy()
                if self.cfg.store_images
                else None
            ),
        )

        self.keyframes.append(kf)

        self.covis_graph.add_keyframe(kf)

        self._last_kf = kf

        self._last_gray = gray.copy()

        self._last_features = feats

        self._last_kf_depth = (
            None
            if depth_frame is None
            else depth_frame.copy()
        )

        self.state = VOState.OK

        self.kf_id += 1

        stats.is_keyframe = True

        stats.kf_reason = "init"

        return stats

    # ═══════════════════════════════════════════════════════════════ #
    #  Tracking                                                       #
    # ═══════════════════════════════════════════════════════════════ #

    def _track(
        self,
        gray: np.ndarray,
        img: np.ndarray,
        timestamp: float,
        stats: FrameStats,
        depth_frame: Optional[np.ndarray] = None,
    ) -> FrameStats:

        kf = self._last_kf

        cur_feats = (
            self.detector.detect_and_compute(
                gray
            )
        )

        stats.num_detected = len(cur_feats)

        match_result = self.matcher.match(

            kf.features,

            cur_feats
        )

        stats.num_matched = len(match_result)

        if len(match_result) < self.cfg.min_inliers:

            self.state = VOState.LOST

            return stats

        rgbd_motion = self._estimate_rgbd_motion(
            match_result=match_result,
        )

        if rgbd_motion is not None:

            pose = None

            inlier_mask = rgbd_motion["inlier_mask"]

            T_rel = rgbd_motion["T_cur_ref"]

            R_for_kf = T_rel[:3, :3]

            stats.num_inliers = int(np.count_nonzero(inlier_mask))

            stats.tracking_method = "rgbd_pnp"

            stats.depth_support_ratio = rgbd_motion["depth_support_ratio"]

            stats.reproj_error_px = rgbd_motion["reproj_error_px"]

        else:

            pose: PoseEstimate = (
                self.estimator.estimate(

                    match_result.pts_ref,

                    match_result.pts_cur,
                )
            )

            stats.num_inliers = pose.num_inliers

            stats.h_score = pose.H_score

            stats.tracking_method = "essential"

            if not pose.success:

                self.state = VOState.LOST

                return stats

            scale = self._recover_scale(
                pose,
                match_result=match_result,
                depth_frame=depth_frame,
            )

            R_cur_ref = pose.R.T

            t_cur_ref = -(
                pose.R.T @ (
                    pose.t.ravel() * scale
                )
            )

            T_rel = np.eye(4)

            T_rel[:3, :3] = R_cur_ref

            T_rel[:3, 3] = t_cur_ref

            R_for_kf = pose.R

            inlier_mask = pose.inlier_mask

        T_candidate = compose_pose(

            self.T_world_cam,

            T_rel
        )

        step = np.linalg.norm(
            T_candidate[:3, 3] -
            self.T_world_cam[:3, 3]
        )

        if (
            not np.isfinite(T_candidate).all()
            or step > self.cfg.max_pose_step_m
        ):

            self.state = VOState.LOST

            return stats

        self.T_world_cam = T_candidate

        self.pose_graph.add(
            self.T_world_cam
        )

        # ───────────────────────────────────────────────────────── #
        #  Triangulation
        # ───────────────────────────────────────────────────────── #

        inlier_ref = (
            match_result.pts_ref[
                inlier_mask
            ]
        )

        inlier_cur = (
            match_result.pts_cur[
                inlier_mask
            ]
        )

        inlier_idx_ref = (
            match_result.idx_ref[
                inlier_mask
            ]
        )

        inlier_idx_cur = (
            match_result.idx_cur[
                inlier_mask
            ]
        )

        T_kf_world = kf.T_cam_world

        T_cur_world = invert_pose(
            self.T_world_cam
        )

        if rgbd_motion is not None:

            new_mps = self._create_rgbd_map_points(
                kf=kf,
                match_result=match_result,
                inlier_mask=inlier_mask,
            )

        else:

            new_mps, _ = (
                self.triangulator.triangulate(

                    T_ref_world=T_kf_world,

                    T_cur_world=T_cur_world,

                    pts_ref=inlier_ref,

                    pts_cur=inlier_cur,

                    idx_ref=inlier_idx_ref,

                    idx_cur=inlier_idx_cur,

                    descriptors=(
                        kf.features.descriptors
                    ),
                )
            )

        self.map_points.extend(
            new_mps
        )

        stats.num_map_pts = len(
            self.map_points
        )

        # ───────────────────────────────────────────────────────── #
        #  Keyframe Selection
        # ───────────────────────────────────────────────────────── #

        do_kf, kf_reason = (
            self.kf_selector.should_insert(

                last_kf=kf,

                R_rel=R_for_kf,

                pts_ref=inlier_ref,

                pts_cur=inlier_cur,

                num_tracked=stats.num_inliers,
            )
        )

        stats.is_keyframe = do_kf

        stats.kf_reason = kf_reason

        if do_kf:

            new_kf = Keyframe(

                frame_id=self.frame_id,

                kf_id=self.kf_id,

                T_world_cam=(
                    self.T_world_cam.copy()
                ),

                features=cur_feats,

                timestamp=timestamp,

                map_points=new_mps,

                image=(
                    gray.copy()
                    if self.cfg.store_images
                    else None
                ),
            )

            self.keyframes.append(
                new_kf
            )

            self._last_kf = new_kf

            self._last_kf_depth = (
                None
                if depth_frame is None
                else depth_frame.copy()
            )

            self.kf_id += 1

            self.covis_graph.add_keyframe(
                new_kf
            )

            if self.on_new_keyframe:

                self.on_new_keyframe(
                    new_kf
                )

        self._last_gray = gray.copy()

        self._last_features = cur_feats

        self._last_depth = (
            None
            if depth_frame is None
            else depth_frame.copy()
        )

        self.state = VOState.OK

        return stats

    def _estimate_rgbd_motion(
        self,
        match_result,
    ) -> Optional[dict]:

        if self._last_kf_depth is None:

            return None

        depths_ref = self._sample_depths(
            self._last_kf_depth,
            match_result.pts_ref,
        )

        valid_depth = (
            np.isfinite(depths_ref)
            & (depths_ref >= self.cfg.min_depth)
            & (depths_ref <= self.cfg.max_depth)
        )

        depth_support_ratio = (
            float(np.count_nonzero(valid_depth)) /
            float(len(match_result))
            if len(match_result) > 0
            else 0.0
        )

        if np.count_nonzero(valid_depth) < self.cfg.rgbd_min_depth_features:

            return None

        object_points = self._backproject_pixels_z(
            match_result.pts_ref[valid_depth],
            depths_ref[valid_depth],
        )

        image_points = match_result.pts_cur[valid_depth].astype(np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points.astype(np.float32),
            imagePoints=image_points,
            cameraMatrix=self.camera.K,
            distCoeffs=self.camera.dist_coeffs,
            iterationsCount=100,
            reprojectionError=self.cfg.pnp_reproj_thresh,
            confidence=self.cfg.ransac_prob,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok or inliers is None or len(inliers) < self.cfg.min_inliers:

            return None

        inliers = inliers.ravel().astype(int)

        if self.cfg.pnp_refine and len(inliers) >= 6:

            try:

                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points[inliers].astype(np.float32),
                    image_points[inliers].astype(np.float32),
                    self.camera.K,
                    self.camera.dist_coeffs,
                    rvec,
                    tvec,
                )

            except cv2.error:

                pass

        R_cur_ref, _ = cv2.Rodrigues(rvec)

        T_cur_ref = np.eye(4)

        T_cur_ref[:3, :3] = R_cur_ref

        T_cur_ref[:3, 3] = tvec.reshape(3)

        if not np.isfinite(T_cur_ref).all():

            return None

        projected, _ = cv2.projectPoints(
            object_points[inliers].astype(np.float32),
            rvec,
            tvec,
            self.camera.K,
            self.camera.dist_coeffs,
        )

        projected = projected.reshape(-1, 2)

        reproj_error = float(
            np.median(
                np.linalg.norm(
                    projected - image_points[inliers],
                    axis=1,
                )
            )
        )

        if not np.isfinite(reproj_error):

            return None

        full_inlier_mask = np.zeros(
            len(match_result),
            dtype=bool,
        )

        valid_indices = np.where(valid_depth)[0]

        full_inlier_mask[
            valid_indices[inliers]
        ] = True

        return {
            "T_cur_ref": T_cur_ref,
            "inlier_mask": full_inlier_mask,
            "depth_support_ratio": depth_support_ratio,
            "reproj_error_px": reproj_error,
        }

    def _create_rgbd_map_points(
        self,
        kf: Keyframe,
        match_result,
        inlier_mask: np.ndarray,
    ) -> List[MapPoint]:

        if self._last_kf_depth is None:

            return []

        pts_ref = match_result.pts_ref[inlier_mask]

        idx_ref = match_result.idx_ref[inlier_mask]

        idx_cur = match_result.idx_cur[inlier_mask]

        depths = self._sample_depths(
            self._last_kf_depth,
            pts_ref,
        )

        valid = (
            np.isfinite(depths)
            & (depths >= self.cfg.min_depth)
            & (depths <= self.cfg.max_depth)
        )

        if not np.any(valid):

            return []

        points_ref = self._backproject_pixels_z(
            pts_ref[valid],
            depths[valid],
        )

        ones = np.ones(
            (len(points_ref), 1),
            dtype=np.float64,
        )

        points_world = (
            kf.T_world_cam @
            np.hstack([points_ref, ones]).T
        ).T[:, :3]

        map_points: List[MapPoint] = []

        valid_idx_ref = idx_ref[valid]

        valid_idx_cur = idx_cur[valid]

        for i, xyz in enumerate(points_world):

            descriptor = (
                kf.features.descriptors[valid_idx_ref[i]]
                if kf.features.descriptors is not None
                and len(kf.features.descriptors) > valid_idx_ref[i]
                else None
            )

            map_points.append(
                MapPoint(
                    xyz=xyz.astype(np.float64),
                    ref_idx=int(valid_idx_ref[i]),
                    cur_idx=int(valid_idx_cur[i]),
                    reproj_err=0.0,
                    observations=2,
                    descriptor=descriptor,
                )
            )

        return map_points

    # ═══════════════════════════════════════════════════════════════ #
    #  Scale Recovery                                                #
    # ═══════════════════════════════════════════════════════════════ #

    def _recover_scale(
        self,
        pose: PoseEstimate,
        match_result=None,
        depth_frame: Optional[np.ndarray] = None,
    ) -> float:

        """
        Temporary monocular scale heuristic.

        This will eventually be replaced by:
        - RGBD metric grounding
        - semantic geometry constraints
        - sensor-fusion scale estimation
        """

        mode = self.cfg.scale_mode

        if mode == "fixed":

            return self.cfg.fixed_scale

        if mode == "none":

            return 1.0

        if (
            mode == "rgbd"
            and depth_frame is not None
            and match_result is not None
        ):

            rgbd_scale = self._recover_rgbd_scale(
                pose=pose,
                match_result=match_result,
                depth_frame=depth_frame,
            )

            if rgbd_scale is not None:

                return rgbd_scale

        if (
            self.map_points
            and mode == "median_depth"
        ):

            T_cur_world = invert_pose(
                self.T_world_cam
            )

            depth = (
                self.triangulator.compute_median_depth(

                    self.map_points[
                        -min(
                            200,
                            len(self.map_points)
                        ):
                    ],

                    T_cur_world,
                )
            )

            if depth > 0:

                return depth

        return 1.0

    def _recover_rgbd_scale(
        self,
        pose: PoseEstimate,
        match_result,
        depth_frame: np.ndarray,
    ) -> Optional[float]:

        if pose.t is None or np.linalg.norm(pose.t) < 1e-9:

            return None

        inlier_mask = pose.inlier_mask

        if inlier_mask is None or len(inlier_mask) == 0:

            return None

        pts_ref = match_result.pts_ref[inlier_mask]
        pts_cur = match_result.pts_cur[inlier_mask]

        if len(pts_ref) < 8:

            return None

        depths_ref = self._sample_depths(
            self._last_depth,
            pts_ref,
        )

        depths_cur = self._sample_depths(
            depth_frame,
            pts_cur,
        )

        valid = (
            np.isfinite(depths_ref)
            & np.isfinite(depths_cur)
            & (depths_ref > self.cfg.min_depth)
            & (depths_cur > self.cfg.min_depth)
            & (depths_ref < self.cfg.max_depth)
            & (depths_cur < self.cfg.max_depth)
        )

        if np.count_nonzero(valid) < 8:

            return None

        rays_ref = self.camera.backproject(
            pts_ref[valid],
            depth=1.0,
        )

        rays_cur = self.camera.backproject(
            pts_cur[valid],
            depth=1.0,
        )

        pts3_ref = rays_ref * depths_ref[valid, None]
        pts3_cur = rays_cur * depths_cur[valid, None]
        displacements = np.linalg.norm(
            pts3_cur - pts3_ref,
            axis=1,
        )
        displacement = float(np.median(displacements))

        if not np.isfinite(displacement) or displacement <= 0.0:

            return None

        return float(
            np.clip(
                displacement,
                0.02,
                5.0,
            )
        )

    @staticmethod
    def _sample_depths(
        depth_frame: Optional[np.ndarray],
        pts: np.ndarray,
    ) -> np.ndarray:

        if depth_frame is None or len(pts) == 0:

            return np.full(len(pts), np.nan)

        depth = np.asarray(depth_frame)
        h, w = depth.shape[:2]
        u = np.rint(pts[:, 0]).astype(int)
        v = np.rint(pts[:, 1]).astype(int)
        valid = (
            (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
        )
        values = np.full(len(pts), np.nan, dtype=np.float64)
        raw = depth[v[valid], u[valid]].astype(np.float64)
        if raw.size:
            if np.nanmedian(raw) > 20.0:
                raw = raw / 1000.0
            values[valid] = raw
        return values

    def _backproject_pixels_z(
        self,
        pts: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:

        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)

        depths = np.asarray(depths, dtype=np.float64).reshape(-1)

        x = (
            (pts[:, 0] - self.camera.cx) *
            depths /
            self.camera.fx
        )

        y = (
            (pts[:, 1] - self.camera.cy) *
            depths /
            self.camera.fy
        )

        return np.column_stack(
            [x, y, depths]
        )


    # ═══════════════════════════════════════════════════════════════ #
    #  Helpers                                                        #
    # ═══════════════════════════════════════════════════════════════ #

    @staticmethod
    def _to_gray(
        img: np.ndarray
    ) -> np.ndarray:

        if img.ndim == 3:

            return cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

        return img

    def summary(self) -> str:

        lines = [

            "=== Visual Odometry Summary ===",

            f"Frames processed : {self.frame_id}",

            f"Keyframes        : {len(self.keyframes)}",

            f"Map points       : {len(self.map_points)}",

            f"State            : {self.state.name}",
        ]

        if self.stats_history:

            proc_times = [

                s.process_ms
                for s in self.stats_history[1:]
            ]

            if proc_times:

                lines.append(

                    f"Avg process time : "
                    f"{np.mean(proc_times):.1f} ms "
                    f"({1000/np.mean(proc_times):.1f} fps)"
                )

        return "\n".join(lines)
