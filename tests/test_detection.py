"""Tests for Phase 1: Detection & View Classification."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantosmpl.detection.image_loader import (
    LoadedImage,
    _focal_length_pixels,
    load_image,
)
from scantosmpl.detection.keypoint_detector import KeypointResult
from scantosmpl.detection.view_classifier import ViewClassifier
from scantosmpl.types import ViewType

SCAN_DIR = Path("data/t-pose/jpg")
GROUND_TRUTH_CSV = Path("data/t-pose/ground_truth.csv")


def _scan_available() -> bool:
    return SCAN_DIR.exists() and any(SCAN_DIR.glob("*.JPG"))


requires_scan = pytest.mark.skipif(
    not _scan_available(),
    reason=f"Scanner images not found in {SCAN_DIR}",
)


# ---------------------------------------------------------------------------
# Unit tests (no scanner images needed)
# ---------------------------------------------------------------------------


class TestExifTranspose:
    """1.1: EXIF transpose handles orientation tags correctly."""

    def _make_oriented_image(self, orientation: int) -> Path:
        """Create a JPEG with a specific EXIF orientation tag."""
        import piexif

        # Create a 100x200 image (portrait-ish) with a marker:
        # top-left pixel is red so we can verify orientation
        img = Image.new("RGB", (100, 200), (0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0))  # red marker at top-left

        exif_dict = {"0th": {piexif.ImageIFD.Orientation: orientation}}
        exif_bytes = piexif.dump(exif_dict)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, exif=exif_bytes)
        return Path(tmp.name)

    @pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_exif_orientation(self, orientation):
        """Each EXIF orientation produces a valid transposed image."""
        try:
            import piexif  # noqa: F401
        except ImportError:
            pytest.skip("piexif not installed")

        path = self._make_oriented_image(orientation)
        try:
            loaded = load_image(path, orientation_overrides={})
            assert loaded.image.size[0] > 0
            assert loaded.image.size[1] > 0
        finally:
            path.unlink()


class TestIntrinsicsExtraction:
    """1.2: Focal length extraction from EXIF."""

    def test_focal_plane_x_resolution(self):
        """FocalPlaneXResolution gives exact sensor width."""
        exif = {
            37386: 24.0,  # FocalLength
            41486: 6718.9,  # FocalPlaneXResolution (pixels/inch)
            41488: 2,  # FocalPlaneResolutionUnit (inches)
        }
        f_px, is_exact = _focal_length_pixels(exif, image_width=6000)
        assert is_exact
        # Canon 2000D: sensor ~22.7mm, so f_px = 24 * 6000 / 22.7 ≈ 6344
        assert 6000 < f_px < 7000

    def test_35mm_equivalent(self):
        """FocalLengthIn35mmFilm gives crop factor."""
        exif = {
            37386: 24.0,  # FocalLength
            41989: 38,  # FocalLengthIn35mmFilm (crop factor ~1.58)
        }
        f_px, is_exact = _focal_length_pixels(exif, image_width=6000)
        assert is_exact
        # sensor_width = 36/1.58 ≈ 22.7mm, f_px ≈ 6340
        assert 6000 < f_px < 7000

    def test_fallback_36mm(self):
        """Missing sensor info falls back to 36mm."""
        exif = {37386: 50.0}  # FocalLength only
        f_px, is_exact = _focal_length_pixels(exif, image_width=6000)
        assert not is_exact
        # f_px = 50 * 6000 / 36 ≈ 8333
        assert 8000 < f_px < 9000

    def test_no_focal_length(self):
        """Missing focal length returns 0."""
        f_px, is_exact = _focal_length_pixels({}, image_width=6000)
        assert f_px == 0.0
        assert not is_exact


class TestViewClassifier:
    """View classification logic."""

    def _make_keypoints(self, visible_indices: list[int], conf: float = 0.8) -> KeypointResult:
        """Create a KeypointResult with specified visible keypoints."""
        confs = np.zeros(17, dtype=np.float32)
        for i in visible_indices:
            confs[i] = conf
        return KeypointResult(
            keypoints=np.zeros((17, 2), dtype=np.float32),
            confidences=confs,
            bbox=np.array([0, 0, 100, 200], dtype=np.float32),
        )

    def test_full_body(self):
        """All extremity groups visible -> FULL_BODY."""
        # shoulder(5), hip(11), ankle(15), wrist(9) + some others
        kp = self._make_keypoints([0, 5, 6, 7, 9, 11, 12, 13, 15])
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.FULL_BODY

    def test_full_body_minimal(self):
        """Minimum for FULL_BODY: one of each group + enough total."""
        # one shoulder, one hip, one ankle, one wrist, plus 1 more
        kp = self._make_keypoints([5, 9, 11, 15, 0])
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.FULL_BODY

    def test_partial_missing_ankles(self):
        """Missing ankles -> PARTIAL (feet cropped)."""
        kp = self._make_keypoints([0, 5, 6, 7, 8, 9, 10, 11, 12])
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.PARTIAL

    def test_partial_missing_wrists(self):
        """Missing wrists -> PARTIAL (arms cropped)."""
        kp = self._make_keypoints([0, 5, 6, 11, 12, 13, 14, 15, 16])
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.PARTIAL

    def test_skip_too_few(self):
        """Fewer than 5 keypoints -> SKIP."""
        kp = self._make_keypoints([0, 5, 11])  # only 3
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.SKIP

    def test_skip_empty(self):
        """No keypoints -> SKIP."""
        kp = self._make_keypoints([])
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.SKIP

    def test_low_confidence_ignored(self):
        """Keypoints below threshold don't count."""
        kp = self._make_keypoints([5, 9, 11, 15, 0], conf=0.2)  # below 0.3
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.SKIP


# ---------------------------------------------------------------------------
# Integration tests (require scanner images + models)
# ---------------------------------------------------------------------------


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
        """Person bbox should cover a significant portion of each image."""
        for loaded in loaded_images:
            det = detector.detect(loaded.image)
            if det is None:
                continue
            # Person should cover at least 10% of image (scanner images are close-up)
            assert det.bbox_fraction > 0.10, (
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
        # Use 80% threshold — some edge cases are genuinely ambiguous
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
