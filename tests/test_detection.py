"""Unit tests for Phase 1: Detection & View Classification.

These tests run without scanner images or model downloads.
Integration tests are in tests/integration/test_detection_integration.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantosmpl.detection.image_loader import (
    _focal_length_pixels,
    load_image,
)
from scantosmpl.detection.keypoint_detector import KeypointResult
from scantosmpl.detection.view_classifier import ViewClassifier
from scantosmpl.types import ViewType


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
        kp = self._make_keypoints([0, 5, 6, 7, 9, 11, 12, 13, 15])
        classifier = ViewClassifier()
        assert classifier.classify(kp) == ViewType.FULL_BODY

    def test_full_body_minimal(self):
        """Minimum for FULL_BODY: one of each group + enough total."""
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
