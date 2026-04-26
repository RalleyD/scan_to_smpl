"""
Unit tests for Phase 2 HMR helpers — no GPU, no checkpoints required.

Run with:
    pytest tests/test_hmr.py -v
"""

import numpy as np
import pytest
import torch
from PIL import Image

from scantosmpl.hmr.camera_hmr import CameraHMRInference, HMROutput
from scantosmpl.hmr.orientation import OrientationQuality, check_orientation_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_crop_M(cx: float, cy: float, box_size: float) -> np.ndarray:
    """Recreate the affine matrix used by _prepare_crop."""
    scale = 256.0 / box_size
    return np.array(
        [[scale, 0.0, 128.0 - cx * scale], [0.0, scale, 128.0 - cy * scale]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 1. Rotation matrix → axis-angle
# ---------------------------------------------------------------------------


class TestRotmatToAxisAngle:
    def test_identity_gives_zero(self):
        eye = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
        aa = CameraHMRInference._rotmat_to_aa(eye)
        assert aa.shape == (1, 3)
        assert np.allclose(aa, 0.0, atol=1e-5)

    def test_known_rotation_roundtrip(self):
        from scipy.spatial.transform import Rotation

        angles = [0.3, -0.8, 1.2]
        R_np = Rotation.from_euler("xyz", angles).as_matrix().astype(np.float32)
        R_t = torch.from_numpy(R_np).unsqueeze(0)  # (1, 3, 3)
        aa = CameraHMRInference._rotmat_to_aa(R_t)  # (1, 3)
        R_back = Rotation.from_rotvec(aa[0]).as_matrix()
        assert np.allclose(R_np, R_back, atol=1e-5)

    def test_batch_shape(self):
        R = torch.eye(3).unsqueeze(0).expand(23, -1, -1)  # (23, 3, 3)
        aa = CameraHMRInference._rotmat_to_aa(R)
        assert aa.shape == (23, 3)


# ---------------------------------------------------------------------------
# 2. Crop preprocessing
# ---------------------------------------------------------------------------


class TestCropPreprocessing:
    @pytest.fixture()
    def inference(self, monkeypatch):
        """Return a CameraHMRInference with models not loaded (we only test helpers)."""
        cfg = object.__new__(object)  # dummy config
        obj = object.__new__(CameraHMRInference)
        obj.config = cfg
        obj.device = "cpu"
        return obj

    def test_crop_is_256x256(self, inference):
        img = Image.fromarray(np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8))
        bbox = np.array([100, 200, 500, 900])
        tensor, cx, cy, box_size, M = inference._prepare_crop(img, bbox)
        assert tensor.shape == (3, 256, 256), "Crop should be 3×256×256"

    def test_center_and_scale_from_bbox(self, inference):
        img = Image.fromarray(np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8))
        x1, y1, x2, y2 = 100.0, 200.0, 400.0, 700.0
        bbox = np.array([x1, y1, x2, y2])
        _, cx, cy, box_size, _ = inference._prepare_crop(img, bbox)
        assert abs(cx - (x1 + x2) / 2.0) < 1e-4
        assert abs(cy - (y1 + y2) / 2.0) < 1e-4
        assert abs(box_size - max(x2 - x1, y2 - y1)) < 1e-4

    def test_imagenet_norm_range(self, inference):
        """Normalised crop values should cover a typical ImageNet range (roughly -2 to 2)."""
        img = Image.fromarray(np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8))
        bbox = np.array([50, 50, 550, 750])
        tensor, *_ = inference._prepare_crop(img, bbox)
        assert tensor.min().item() > -3.0
        assert tensor.max().item() < 3.0


# ---------------------------------------------------------------------------
# 3. Full-image preprocessing (FLNet)
# ---------------------------------------------------------------------------


class TestFullImagePreprocessing:
    @pytest.fixture()
    def inference(self):
        obj = object.__new__(CameraHMRInference)
        return obj

    def test_output_is_256x256(self, inference):
        img = Image.fromarray(np.random.randint(0, 255, (1200, 800, 3), dtype=np.uint8))
        tensor = inference._prepare_full_image(img)
        assert tensor.shape == (3, 256, 256)

    def test_landscape_and_portrait(self, inference):
        for h, w in [(600, 800), (1000, 600)]:
            img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
            tensor = inference._prepare_full_image(img)
            assert tensor.shape == (3, 256, 256)


# ---------------------------------------------------------------------------
# 4. FoV ↔ focal length conversion
# ---------------------------------------------------------------------------


class TestFoVConversion:
    def test_known_fov(self):
        """90° vFoV at 512px height → focal = 256px."""
        H = 512
        focal = 256.0
        fov_rad = 2.0 * np.arctan(H / (2.0 * focal))
        assert abs(np.degrees(fov_rad) - 90.0) < 0.01

    def test_roundtrip(self):
        """focal → fov → focal roundtrip."""
        H = 1000
        focal_in = 1500.0
        fov_rad = 2.0 * np.arctan(H / (2.0 * focal_in))
        focal_out = H / (2.0 * np.tan(fov_rad / 2.0))
        assert abs(focal_in - focal_out) < 0.01


# ---------------------------------------------------------------------------
# 5. Dense keypoint denormalisation
# ---------------------------------------------------------------------------


class TestDenseKPDenormalization:
    def test_center_maps_to_crop_center(self):
        """
        A keypoint at (0, 0) in [-0.5, 0.5] normalised space should map to
        the crop centre (128, 128) in crop pixels, then to the bbox centre in image.
        """
        cx, cy, box_size = 400.0, 300.0, 200.0
        M = _make_mock_crop_M(cx, cy, box_size)

        kps_norm = np.zeros((138, 3), dtype=np.float32)  # all at crop centre
        kps_px, confs = CameraHMRInference._denormalize_dense_kps(kps_norm, M)

        assert kps_px.shape == (138, 2)
        assert confs.shape == (138,)
        # Centre of normalised crop → image bbox centre
        assert np.allclose(kps_px[:, 0], cx, atol=1.0)
        assert np.allclose(kps_px[:, 1], cy, atol=1.0)

    def test_confidence_from_log_sigma(self):
        """Confidence = exp(-|log_sigma|): log_sigma=0 → conf=1, large → small."""
        kps_norm = np.zeros((138, 3), dtype=np.float32)
        kps_norm[:, 2] = 0.0  # log_sigma = 0 → confidence = 1
        M = _make_mock_crop_M(100.0, 100.0, 100.0)
        _, confs = CameraHMRInference._denormalize_dense_kps(kps_norm, M)
        assert np.allclose(confs, 1.0, atol=1e-5)

    def test_output_in_valid_range_for_bbox(self):
        """Keypoints from [-0.5, 0.5] crop should map near the bbox region."""
        cx, cy, box_size = 500.0, 500.0, 300.0
        M = _make_mock_crop_M(cx, cy, box_size)

        rng = np.random.default_rng(42)
        kps_norm = np.zeros((138, 3), dtype=np.float32)
        kps_norm[:, :2] = rng.uniform(-0.5, 0.5, (138, 2)).astype(np.float32)

        kps_px, _ = CameraHMRInference._denormalize_dense_kps(kps_norm, M)

        # All points should be within 2× box_size of the bbox centre
        dist = np.sqrt((kps_px[:, 0] - cx) ** 2 + (kps_px[:, 1] - cy) ** 2)
        assert (dist < box_size * 2).all(), "Dense kps out of expected range"


# ---------------------------------------------------------------------------
# 6. CLIFF camera conversion
# ---------------------------------------------------------------------------


class TestCLIFFCameraConversion:
    def test_known_values(self):
        """
        With s=1, tx=ty=0, centred bbox: tz = 2*fl/box_size, tx=ty=0.
        """
        pred_cam = np.array([1.0, 0.0, 0.0])  # s=1, no offset
        img_size = (1000, 800)
        H, W = img_size
        cx_bbox, cy_bbox = W / 2.0, H / 2.0  # bbox centred on principal point
        box_size = 400.0
        fl = 1000.0

        cam_t = CameraHMRInference._cliff_camera(
            pred_cam, (cx_bbox, cy_bbox), box_size, fl, img_size
        )

        expected_tz = 2.0 * fl / box_size  # 5.0
        assert abs(cam_t[2] - expected_tz) < 0.01, f"tz={cam_t[2]} expected {expected_tz}"
        assert abs(cam_t[0]) < 0.01, "tx should be ~0 for centred bbox"
        assert abs(cam_t[1]) < 0.01, "ty should be ~0 for centred bbox"

    def test_positive_tz(self):
        """Translation depth tz should always be positive (person in front of camera)."""
        for s in [0.5, 1.0, 2.0]:
            pred_cam = np.array([s, 0.1, -0.1])
            cam_t = CameraHMRInference._cliff_camera(
                pred_cam, (400.0, 300.0), 250.0, 1200.0, (600, 800)
            )
            assert cam_t[2] > 0, f"tz should be positive, got {cam_t[2]}"


# ---------------------------------------------------------------------------
# 7. Orientation quality checker
# ---------------------------------------------------------------------------


class TestOrientationQuality:
    def _make_keypoints(self, nose_y: float, hip_y: float) -> tuple[np.ndarray, np.ndarray]:
        kps = np.zeros((17, 2), dtype=np.float32)
        confs = np.zeros(17, dtype=np.float32)

        kps[0] = [400, nose_y]   # nose
        kps[11] = [380, hip_y]   # left hip
        kps[12] = [420, hip_y]   # right hip
        confs[0] = confs[11] = confs[12] = 0.9
        return kps, confs

    def test_upright_person_scores_high(self):
        kps, confs = self._make_keypoints(nose_y=200, hip_y=600)
        go = np.array([0.1, 0.0, 0.0])
        quality = check_orientation_quality(go, kps, confs, (1000, 800))
        assert quality.is_upright, "Nose above hips → should be upright"
        assert quality.score >= 0.67

    def test_inverted_person_flagged(self):
        kps, confs = self._make_keypoints(nose_y=700, hip_y=200)
        go = np.array([0.1, 0.0, 0.0])
        quality = check_orientation_quality(go, kps, confs, (1000, 800))
        assert not quality.is_upright, "Nose below hips → should be flagged inverted"
        assert any("inverted" in w.lower() for w in quality.warnings)

    def test_large_rotation_flagged(self):
        kps = np.zeros((17, 2), dtype=np.float32)
        confs = np.zeros(17, dtype=np.float32)
        go = np.array([7.5, 0.0, 0.0])  # > 2π
        quality = check_orientation_quality(go, kps, confs, (1000, 800))
        assert not quality.rotation_magnitude_ok

    def test_identity_rotation_ok(self):
        kps, confs = self._make_keypoints(nose_y=100, hip_y=500)
        go = np.array([0.0, 0.0, 0.0])
        quality = check_orientation_quality(go, kps, confs, (1000, 800))
        assert quality.rotation_magnitude_ok

    def test_t_pose_arms_check(self):
        kps = np.zeros((17, 2), dtype=np.float32)
        confs = np.zeros(17, dtype=np.float32)

        kps[0] = [400, 100]; confs[0] = 0.9   # nose
        kps[11] = [380, 600]; confs[11] = 0.9  # left hip
        kps[12] = [420, 600]; confs[12] = 0.9  # right hip

        # Arms roughly horizontal (T-pose): elbows near shoulder height
        kps[5] = [200, 300]; confs[5] = 0.9   # left shoulder
        kps[7] = [100, 310]; confs[7] = 0.9   # left elbow
        kps[6] = [600, 300]; confs[6] = 0.9   # right shoulder
        kps[8] = [700, 295]; confs[8] = 0.9   # right elbow

        go = np.array([0.0, 0.0, 0.0])
        quality = check_orientation_quality(go, kps, confs, (1000, 800))
        assert quality.score == 1.0  # all checks pass


# ---------------------------------------------------------------------------
# 8. HMROutput dataclass shape contracts
# ---------------------------------------------------------------------------


class TestHMROutputShapes:
    def test_shape_contracts(self):
        out = HMROutput(
            betas=np.zeros(10, dtype=np.float32),
            body_pose=np.zeros(69, dtype=np.float32),
            global_orient=np.zeros(3, dtype=np.float32),
            cam_translation=np.zeros(3, dtype=np.float32),
            dense_keypoints_2d=np.zeros((138, 2), dtype=np.float32),
            dense_keypoint_confs=np.zeros(138, dtype=np.float32),
            fov_flnet=35.0,
            fov_exif=34.5,
            vertices=np.zeros((6890, 3), dtype=np.float32),
        )
        assert out.betas.shape == (10,)
        assert out.body_pose.shape == (69,)
        assert out.global_orient.shape == (3,)
        assert out.cam_translation.shape == (3,)
        assert out.dense_keypoints_2d.shape == (138, 2)
        assert out.dense_keypoint_confs.shape == (138,)
        assert out.vertices is not None and out.vertices.shape == (6890, 3)

    def test_fov_flnet_can_be_none(self):
        out = HMROutput(
            betas=np.zeros(10),
            body_pose=np.zeros(69),
            global_orient=np.zeros(3),
            cam_translation=np.zeros(3),
            dense_keypoints_2d=np.zeros((138, 2)),
            dense_keypoint_confs=np.zeros(138),
            fov_flnet=None,
            fov_exif=34.5,
        )
        assert out.fov_flnet is None
        assert out.vertices is None
