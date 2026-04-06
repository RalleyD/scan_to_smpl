"""Detection module: image loading, person detection, keypoints, view classification."""

from scantosmpl.detection.image_loader import LoadedImage, load_directory, load_image
from scantosmpl.detection.keypoint_detector import KeypointDetector, KeypointResult
from scantosmpl.detection.person_detector import Detection, PersonDetector
from scantosmpl.detection.pipeline import DetectionPipeline
from scantosmpl.detection.view_classifier import ViewClassifier

__all__ = [
    "DetectionPipeline",
    "Detection",
    "KeypointDetector",
    "KeypointResult",
    "LoadedImage",
    "PersonDetector",
    "ViewClassifier",
    "load_directory",
    "load_image",
]
