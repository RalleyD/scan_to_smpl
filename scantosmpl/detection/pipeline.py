"""Detection pipeline: image loading → person detection → keypoints → classification.

Orchestrates the full Phase 1 detection flow and produces debug outputs.
"""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scantosmpl.config import DetectionConfig
from scantosmpl.detection.image_loader import LoadedImage, load_directory
from scantosmpl.detection.keypoint_detector import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    KeypointDetector,
    KeypointResult,
)
from scantosmpl.detection.person_detector import Detection, PersonDetector
from scantosmpl.detection.view_classifier import ViewClassifier
from scantosmpl.types import CameraParams, ViewResult, ViewType

logger = logging.getLogger(__name__)

# Colours for keypoint visualisation (RGB)
_SKELETON_COLOR = (0, 255, 0)
_KP_COLOR_HIGH = (0, 255, 0)  # conf > 0.5
_KP_COLOR_MED = (255, 255, 0)  # conf 0.3-0.5
_KP_COLOR_LOW = (255, 0, 0)  # conf < 0.3
_BBOX_COLOR = (0, 200, 255)
_LABEL_COLORS = {
    ViewType.FULL_BODY: (0, 255, 0),
    ViewType.PARTIAL: (255, 200, 0),
    ViewType.SKIP: (255, 0, 0),
}


class DetectionPipeline:
    """End-to-end detection pipeline for Phase 1."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
        device: str = "cpu",
        orientation_overrides: dict[str, int] | None = None,
    ):
        self.config = config or DetectionConfig()
        self.device = device
        self.orientation_overrides = orientation_overrides

        self.person_detector = PersonDetector(
            model_id=self.config.person_detector,
            device=device,
            confidence_threshold=self.config.person_confidence,
        )
        self.keypoint_detector = KeypointDetector(
            model_id=self.config.keypoint_detector,
            device=device,
        )
        self.view_classifier = ViewClassifier(
            confidence_threshold=self.config.keypoint_confidence,
        )

    def process_image(
        self,
        loaded: LoadedImage,
    ) -> ViewResult:
        """Run detection pipeline on a single loaded image.

        Returns a populated ViewResult.
        """
        # Step 1: Person detection
        detection = self.person_detector.detect(loaded.image)

        if detection is None:
            return ViewResult(
                image_path=loaded.path,
                view_type=ViewType.SKIP,
                camera=loaded.camera,
            )

        # Step 2: Keypoint detection
        kp_result = self.keypoint_detector.detect(loaded.image, detection.bbox)

        # Step 3: View classification
        view_type = self.view_classifier.classify(kp_result)

        return ViewResult(
            image_path=loaded.path,
            view_type=view_type,
            bbox=detection.bbox,
            keypoints_2d=kp_result.keypoints,
            keypoint_confs=kp_result.confidences,
            camera=loaded.camera,
        )

    def process_directory(
        self,
        image_dir: Path,
        debug_dir: Path | None = None,
    ) -> list[ViewResult]:
        """Run detection pipeline on all images in a directory.

        Args:
            image_dir: Directory containing JPEG images.
            debug_dir: If set, save debug visualisations and detections.json here.

        Returns:
            List of ViewResult, one per image.
        """
        loaded_images = load_directory(
            image_dir, orientation_overrides=self.orientation_overrides
        )
        if not loaded_images:
            logger.warning("No images found in %s", image_dir)
            return []

        logger.info("Processing %d images from %s", len(loaded_images), image_dir)
        results = []
        for i, loaded in enumerate(loaded_images):
            logger.info(
                "[%d/%d] %s", i + 1, len(loaded_images), loaded.path.name
            )
            result = self.process_image(loaded)
            results.append(result)
            logger.info(
                "  -> %s (bbox_conf=%.2f, visible_kps=%d/17)",
                result.view_type.value,
                result.bbox[0] if result.bbox is not None else 0,  # just a placeholder
                int((result.keypoint_confs > 0.3).sum()) if result.keypoint_confs is not None else 0,
            )

        # Summary
        counts = {vt: 0 for vt in ViewType}
        for r in results:
            counts[r.view_type] += 1
        logger.info(
            "Detection complete: %d full-body, %d partial, %d skip",
            counts[ViewType.FULL_BODY], counts[ViewType.PARTIAL], counts[ViewType.SKIP],
        )

        # Debug output
        if debug_dir is not None:
            self._save_debug(loaded_images, results, debug_dir)

        return results

    def _save_debug(
        self,
        loaded_images: list[LoadedImage],
        results: list[ViewResult],
        debug_dir: Path,
    ):
        """Save debug visualisations and detection data."""
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Save detections.json
        detections_data = []
        for loaded, result in zip(loaded_images, results):
            entry = {
                "filename": loaded.path.name,
                "view_type": result.view_type.value,
                "image_size": list(loaded.image.size),
                "focal_length_px": loaded.camera.focal_length,
                "principal_point": list(loaded.camera.principal_point),
            }
            if result.bbox is not None:
                entry["bbox"] = result.bbox.tolist()
            if result.keypoints_2d is not None:
                entry["keypoints"] = result.keypoints_2d.tolist()
                entry["keypoint_confidences"] = result.keypoint_confs.tolist()
                entry["keypoint_names"] = COCO_KEYPOINT_NAMES
                visible = result.keypoint_confs > 0.3
                entry["visible_keypoints"] = [
                    COCO_KEYPOINT_NAMES[i] for i in range(17) if visible[i]
                ]
            detections_data.append(entry)

        with open(debug_dir / "detections.json", "w") as f:
            json.dump(detections_data, f, indent=2)

        # Save keypoint overlay images
        for loaded, result in zip(loaded_images, results):
            self._save_keypoint_overlay(loaded, result, debug_dir)

        # Save summary
        counts = {vt.value: 0 for vt in ViewType}
        for r in results:
            counts[r.view_type.value] += 1

        summary_lines = [
            f"Detection Summary ({len(results)} images)",
            "=" * 40,
            f"  FULL_BODY: {counts['full_body']}",
            f"  PARTIAL:   {counts['partial']}",
            f"  SKIP:      {counts['skip']}",
            "",
            "Per-image details:",
            "-" * 40,
        ]
        for loaded, result in zip(loaded_images, results):
            n_vis = (
                int((result.keypoint_confs > 0.3).sum())
                if result.keypoint_confs is not None
                else 0
            )
            summary_lines.append(
                f"  {loaded.path.name}: {result.view_type.value} ({n_vis}/17 keypoints)"
            )

        with open(debug_dir / "summary.txt", "w") as f:
            f.write("\n".join(summary_lines) + "\n")

        logger.info("Debug output saved to %s", debug_dir)

    def _save_keypoint_overlay(
        self,
        loaded: LoadedImage,
        result: ViewResult,
        debug_dir: Path,
    ):
        """Draw keypoints, skeleton, bbox, and classification on image."""
        img = loaded.image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Draw bbox
        if result.bbox is not None:
            x1, y1, x2, y2 = result.bbox
            draw.rectangle([x1, y1, x2, y2], outline=_BBOX_COLOR, width=3)

        # Draw skeleton
        if result.keypoints_2d is not None and result.keypoint_confs is not None:
            kps = result.keypoints_2d
            confs = result.keypoint_confs

            # Draw bones
            for i, j in COCO_SKELETON:
                if confs[i] > 0.3 and confs[j] > 0.3:
                    x1, y1 = kps[i]
                    x2, y2 = kps[j]
                    draw.line([(x1, y1), (x2, y2)], fill=_SKELETON_COLOR, width=2)

            # Draw keypoints
            radius = max(3, min(w, h) // 200)
            for i in range(17):
                x, y = kps[i]
                c = confs[i]
                if c > 0.5:
                    color = _KP_COLOR_HIGH
                elif c > 0.3:
                    color = _KP_COLOR_MED
                else:
                    color = _KP_COLOR_LOW
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=color,
                    outline=color,
                )

        # Draw classification label
        label = result.view_type.value.upper()
        label_color = _LABEL_COLORS.get(result.view_type, (255, 255, 255))
        # Draw text with background
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except (OSError, IOError):
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        margin = 10
        draw.rectangle([margin, margin, margin + tw + 20, margin + th + 20], fill=(0, 0, 0, 180))
        draw.text((margin + 10, margin + 10), label, fill=label_color, font=font)

        # Save
        stem = loaded.path.stem
        out_path = debug_dir / f"{stem}_keypoints.jpg"
        img.save(out_path, quality=85)
