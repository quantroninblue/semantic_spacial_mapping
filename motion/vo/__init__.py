from .camera import CameraModel
from .pipeline import VOConfig, VisualOdometry
from .features import DetectorType
from .vocabulary import VisualVocabulary, BowVector
from .bow_database import BowDatabase, QueryResult
from .covisibility import CovisibilityGraph
from .local_mapping import LocalMapper
from .loop_detector import LoopDetector, LoopEvent, build_vocabulary_from_keyframes
from .place_recognition import PlaceRecognizer, LoopCandidate


def __getattr__(name):
    if name in {"FeatureOverlay", "TrajectoryPlot", "plot_trajectory_static"}:
        from .visualization import FeatureOverlay, TrajectoryPlot, plot_trajectory_static

        return {
            "FeatureOverlay": FeatureOverlay,
            "TrajectoryPlot": TrajectoryPlot,
            "plot_trajectory_static": plot_trajectory_static,
        }[name]
    raise AttributeError(name)
