"""2D keypoint detection using ViTPose++ via HuggingFace transformers.

Uses usyd-community/vitpose-plus-base for COCO-17 body keypoint estimation.
Takes a person bounding box crop and returns keypoints in original image coordinates.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class KeypointResult:
    """Keypoint detection result for a single person."""

    keypoints: np.ndarray  # (17, 2) x, y in image coordinates
    confidences: np.ndarray  # (17,) per-keypoint confidence
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 used for cropping

    @property
    def num_visible(self, threshold: float = 0.3) -> int:
        """Count keypoints above confidence threshold."""
        return int((self.confidences > threshold).sum())


# COCO-17 keypoint names for reference
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO-17 skeleton connections for visualisation
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


class KeypointDetector:
    """ViTPose++ keypoint detector via HuggingFace transformers."""

    # ViTPose++ MoE dataset indices (6 experts)
    # Index 0 = COCO (17 body keypoints) — the one we use
    DATASET_INDEX_COCO = 0

    def __init__(
        self,
        model_id: str = "usyd-community/vitpose-plus-base",
        device: str = "cpu",
        dataset_index: int = 0,
    ):
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

        logger.info("Loading ViTPose++ model: %s", model_id)
        self.processor = VitPoseImageProcessor.from_pretrained(model_id)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        self.dataset_index = dataset_index
        logger.info("ViTPose++ loaded on %s (dataset_index=%d)", device, dataset_index)

    @torch.no_grad()
    def detect(self, image: Image.Image, bbox: np.ndarray) -> KeypointResult:
        """Detect keypoints for a single person crop.

        Args:
            image: Full image (PIL).
            bbox: Person bounding box (4,) x1, y1, x2, y2 in pixels.

        Returns:
            KeypointResult with keypoints in original image coordinates.
        """
        # ViTPose processor expects boxes as list[list[list[float]]]
        # outer: batch, middle: persons per image, inner: bbox coords
        boxes = [[[float(c) for c in bbox]]]

        inputs = self.processor(images=image, boxes=boxes, return_tensors="pt").to(self.device)
        # ViTPose++ requires dataset_index to select the MoE expert head
        dataset_index = torch.tensor([self.dataset_index], device=self.device)
        outputs = self.model(**inputs, dataset_index=dataset_index)

        w, h = image.size
        results = self.processor.post_process_pose_estimation(
            outputs,
            boxes=boxes,
            target_sizes=[(h, w)],
        )

        # results is list[list[dict]] — one list per image, one dict per person
        if not results or not results[0]:
            logger.warning("No keypoints detected")
            return KeypointResult(
                keypoints=np.zeros((17, 2), dtype=np.float32),
                confidences=np.zeros(17, dtype=np.float32),
                bbox=bbox,
            )

        person = results[0][0]
        keypoints = person["keypoints"].cpu().numpy().astype(np.float32)  # (17, 2)
        scores = person["scores"].cpu().numpy().astype(np.float32)  # (17,)

        return KeypointResult(
            keypoints=keypoints,
            confidences=scores,
            bbox=bbox.copy(),
        )

    def detect_batch(
        self, images: list[Image.Image], bboxes: list[np.ndarray]
    ) -> list[KeypointResult]:
        """Detect keypoints for multiple images (sequential processing)."""
        return [self.detect(img, bbox) for img, bbox in zip(images, bboxes)]
