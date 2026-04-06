"""Person detection using RT-DETR via HuggingFace transformers.

Uses PekingU/rtdetr_r50vd_coco_o365 for robust person bounding box detection.
Returns the highest-confidence person detection per image.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

COCO_PERSON_LABEL = "person"


@dataclass
class Detection:
    """Single person detection result."""

    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in pixels
    confidence: float
    image_size: tuple[int, int]  # (width, height)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def bbox_fraction(self) -> float:
        """Fraction of image area covered by bbox."""
        w, h = self.image_size
        return self.area / (w * h) if w * h > 0 else 0.0


class PersonDetector:
    """RT-DETR person detector via HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = "PekingU/rtdetr_r50vd_coco_o365",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ):
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self.device = device
        self.confidence_threshold = confidence_threshold

        logger.info("Loading RT-DETR model: %s", model_id)
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        logger.info("RT-DETR loaded on %s", device)

    @torch.no_grad()
    def detect(self, image: Image.Image) -> Detection | None:
        """Detect the highest-confidence person in an image.

        Returns None if no person detected above threshold.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        w, h = image.size
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[h, w]], device=self.device),
            threshold=self.confidence_threshold,
        )[0]

        labels = results["labels"]
        scores = results["scores"]
        boxes = results["boxes"]

        # Find person detections
        id2label = self.model.config.id2label
        best_score = 0.0
        best_box = None

        for label_id, score, box in zip(labels, scores, boxes):
            label_name = id2label.get(label_id.item(), "")
            if label_name.lower() == COCO_PERSON_LABEL and score.item() > best_score:
                best_score = score.item()
                best_box = box.cpu().numpy()

        if best_box is None:
            logger.warning("No person detected (threshold=%.2f)", self.confidence_threshold)
            return None

        return Detection(
            bbox=best_box.astype(np.float32),
            confidence=best_score,
            image_size=(w, h),
        )

    def detect_batch(self, images: list[Image.Image]) -> list[Detection | None]:
        """Detect persons in a batch of images."""
        return [self.detect(img) for img in images]
