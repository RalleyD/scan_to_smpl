"""Integration tests for Phase 1: Detection pipeline on real scanner images.

Requires:
- Scanner images in data/t-pose/jpg/
- HuggingFace model downloads (RT-DETR ~200MB, ViTPose++ ~200MB on first run)
- Ground truth CSV at data/t-pose/ground_truth.csv

Run with: pytest tests/integration/ -v
"""

import csv
from pathlib import Path

import pytest

from scantosmpl.types import ViewType

SCAN_DIR = Path("data/t-pose/jpg")
GROUND_TRUTH_CSV = Path("data/t-pose/ground_truth.csv")


def _scan_available() -> bool:
    return SCAN_DIR.exists() and any(SCAN_DIR.glob("*.JPG"))


requires_scan = pytest.mark.skipif(
    not _scan_available(),
    reason=f"Scanner images not found in {SCAN_DIR}",
)


@requires_scan
class TestPersonDetection:
    """1.3: RT-DETR detects exactly 1 person per scanner image."""

    @pytest.fixture(scope="class")
    def detector(self):
        import torch
        from scantosmpl.detection.person_detector import PersonDetector

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return PersonDetector(device=device)

    @pytest.fixture(scope="class")
    def loaded_images(self):
        from scantosmpl.detection.image_loader import load_directory

        return load_directory(SCAN_DIR)

    def test_all_images_detected(self, detector, loaded_images):
        """Every scanner image should have a person detection."""
        for loaded in loaded_images:
            det = detector.detect(loaded.image)
            assert det is not None, f"No person detected in {loaded.path.name}"
            assert det.confidence > 0.5, (
                f"Low confidence ({det.confidence:.2f}) for {loaded.path.name}"
            )

    def test_bbox_reasonable_size(self, detector, loaded_images):
        """Person bbox should cover a reasonable portion of each image.

        Some scanner images show only partial body (e.g., cam04_6 is rear-view
        missing head and arms), so we use a low 3% threshold.
        """
        for loaded in loaded_images:
            det = detector.detect(loaded.image)
            if det is None:
                continue
            assert det.bbox_fraction > 0.03, (
                f"Bbox too small ({det.bbox_fraction:.1%}) for {loaded.path.name}"
            )


@requires_scan
class TestKeypointDetection:
    """1.4: ViTPose++ detects reasonable keypoints on scanner images."""

    @pytest.fixture(scope="class")
    def detectors(self):
        import torch
        from scantosmpl.detection.keypoint_detector import KeypointDetector
        from scantosmpl.detection.person_detector import PersonDetector

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "person": PersonDetector(device=device),
            "keypoint": KeypointDetector(device=device),
        }

    @pytest.fixture(scope="class")
    def loaded_images(self):
        from scantosmpl.detection.image_loader import load_directory

        return load_directory(SCAN_DIR)

    def test_keypoints_detected(self, detectors, loaded_images):
        """Each image should have at least 5 keypoints with conf > 0.3."""
        for loaded in loaded_images:
            det = detectors["person"].detect(loaded.image)
            if det is None:
                continue
            kp = detectors["keypoint"].detect(loaded.image, det.bbox)
            n_visible = int((kp.confidences > 0.3).sum())
            assert n_visible >= 5, (
                f"Only {n_visible}/17 keypoints in {loaded.path.name}"
            )


@requires_scan
class TestClassificationGroundTruth:
    """1.5: Classification matches ground truth labels."""

    @pytest.fixture(scope="class")
    def pipeline_results(self):
        import torch
        from scantosmpl.detection.pipeline import DetectionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = DetectionPipeline(device=device)
        return {
            r.image_path.name: r
            for r in pipeline.process_directory(SCAN_DIR)
        }

    @pytest.fixture(scope="class")
    def ground_truth(self):
        if not GROUND_TRUTH_CSV.exists():
            pytest.skip("Ground truth CSV not found")
        gt = {}
        with open(GROUND_TRUTH_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vt = row["view_type"].strip().upper()
                gt[row["filename"]] = ViewType(vt.lower())
        return gt

    def test_classification_accuracy(self, pipeline_results, ground_truth):
        """At least 80% of views should match ground truth classification."""
        correct = 0
        total = 0
        mismatches = []
        for filename, expected_type in ground_truth.items():
            if filename not in pipeline_results:
                continue
            total += 1
            predicted = pipeline_results[filename].view_type
            if predicted == expected_type:
                correct += 1
            else:
                mismatches.append(
                    f"  {filename}: expected={expected_type.value}, got={predicted.value}"
                )

        accuracy = correct / total if total > 0 else 0
        msg = f"Classification accuracy: {correct}/{total} ({accuracy:.0%})"
        if mismatches:
            msg += "\nMismatches:\n" + "\n".join(mismatches)
        # 80% threshold — some edge cases are genuinely ambiguous
        assert accuracy >= 0.80, msg


@requires_scan
class TestDetectionPipeline:
    """End-to-end pipeline test."""

    def test_pipeline_runs(self, tmp_path):
        """Pipeline runs on scanner images and produces debug output."""
        import torch
        from scantosmpl.detection.pipeline import DetectionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = DetectionPipeline(device=device)
        debug_dir = tmp_path / "debug"

        results = pipeline.process_directory(SCAN_DIR, debug_dir=debug_dir)

        assert len(results) > 0
        assert (debug_dir / "detections.json").exists()
        assert (debug_dir / "summary.txt").exists()

        # Check all images produced results
        image_count = len(list(SCAN_DIR.glob("*.JPG")))
        assert len(results) == image_count

    def test_intrinsics_reasonable(self):
        """1.2: EXIF intrinsics produce reasonable focal lengths."""
        from scantosmpl.detection.image_loader import load_directory

        loaded = load_directory(SCAN_DIR)
        for img in loaded:
            f = img.camera.focal_length
            assert 500 < f < 15000, (
                f"Unreasonable focal length {f:.0f}px for {img.path.name}"
            )
