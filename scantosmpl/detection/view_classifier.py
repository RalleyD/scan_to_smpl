"""View classification based on keypoint visibility.

Classifies views as FULL_BODY, PARTIAL, or SKIP based on which COCO-17
keypoints are detected with sufficient confidence.
"""

import logging

import numpy as np

from scantosmpl.detection.keypoint_detector import KeypointResult
from scantosmpl.types import ViewType

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


class ViewClassifier:
    """Classify views by keypoint visibility.

    FULL_BODY: At least one shoulder, one hip, one ankle, and one wrist visible.
        Subject spans head-to-toe in the image.
    PARTIAL: Person detected with >=5 visible keypoints but missing some extremities.
        Still useful for Phase 2 reprojection with visible joints.
    SKIP: Fewer than min_partial_keypoints visible, or no person detected.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        min_partial_keypoints: int = 5,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_partial_keypoints = min_partial_keypoints

    def _is_visible(self, confidences: np.ndarray, *indices: int) -> bool:
        """Check if at least one of the given keypoint indices is visible."""
        return any(confidences[i] > self.confidence_threshold for i in indices)

    def classify(self, keypoint_result: KeypointResult) -> ViewType:
        """Classify a view based on keypoint visibility.

        Args:
            keypoint_result: Keypoints and confidences from ViTPose.

        Returns:
            ViewType classification.
        """
        confs = keypoint_result.confidences
        num_visible = int((confs > self.confidence_threshold).sum())

        # SKIP: too few keypoints
        if num_visible < self.min_partial_keypoints:
            logger.debug("SKIP: only %d/%d keypoints visible", num_visible, len(confs))
            return ViewType.SKIP

        # FULL_BODY: need at least one of each extremity group
        has_shoulder = self._is_visible(confs, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
        has_hip = self._is_visible(confs, KP_LEFT_HIP, KP_RIGHT_HIP)
        has_ankle = self._is_visible(confs, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)
        has_wrist = self._is_visible(confs, KP_LEFT_WRIST, KP_RIGHT_WRIST)

        if has_shoulder and has_hip and has_ankle and has_wrist:
            logger.debug("FULL_BODY: %d keypoints visible, all groups present", num_visible)
            return ViewType.FULL_BODY

        logger.debug(
            "PARTIAL: %d keypoints visible (shoulder=%s, hip=%s, ankle=%s, wrist=%s)",
            num_visible, has_shoulder, has_hip, has_ankle, has_wrist,
        )
        return ViewType.PARTIAL
