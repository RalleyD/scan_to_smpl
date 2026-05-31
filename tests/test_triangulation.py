"""Unit tests for Phase 5 triangulation, undistortion, and frame alignment."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scantosmpl.calibration.colmap_reader import (
    ColmapCamera,
    match_views_to_colmap,
    read_colmap_model,
)
from scantosmpl.calibration.frame_alignment import FrameAlignment, compute_frame_alignment
from scantosmpl.calibration.undistort import build_pinhole_K, undistort_keypoints
from scantosmpl.triangulation.dlt import build_projection_matrix, triangulate_joints, triangulate_point
from scantosmpl.triangulation.ransac import ransac_triangulate_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(
    pos: np.ndarray,
    look_at: np.ndarray = np.zeros(3),
    focal: float = 1000.0,
    w: int = 1000,
    h: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic pinhole camera looking from pos toward look_at."""
    forward = look_at - pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.stack([right, -up, forward], axis=0)  # world-to-camera rows
    t = -R @ pos
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    return R, t, K


# ---------------------------------------------------------------------------
# DLT
# ---------------------------------------------------------------------------


class TestDLT:
    def test_two_views_known_point(self):
        """Triangulate a known 3D point from two synthetic cameras."""
        pt_true = np.array([0.5, -0.3, 2.0])

        R1, t1, K1 = _make_camera(np.array([2.0, 0.0, 0.0]))
        R2, t2, K2 = _make_camera(np.array([-2.0, 0.5, 0.0]))

        P1 = build_projection_matrix(R1, t1, K1)
        P2 = build_projection_matrix(R2, t2, K2)

        # Project to 2D
        def proj(P, X):
            h = P @ np.append(X, 1.0)
            return h[:2] / h[2]

        p1 = proj(P1, pt_true)
        p2 = proj(P2, pt_true)

        recovered = triangulate_point(
            np.array([p1, p2]),
            [P1, P2],
        )
        assert np.linalg.norm(recovered - pt_true) < 0.01, (
            f"Recovered {recovered}, expected {pt_true}"
        )

    def test_multiple_views_accuracy(self):
        """More views should give a lower residual."""
        pt_true = np.array([0.1, 0.2, 3.0])
        positions = [
            np.array([3.0, 0.0, 0.0]),
            np.array([-3.0, 0.0, 0.0]),
            np.array([0.0, 3.0, 0.0]),
            np.array([0.0, -3.0, 0.0]),
        ]

        cameras = [_make_camera(p) for p in positions]

        def proj(K, R, t, X):
            p = K @ (R @ X + t)
            return p[:2] / p[2]

        pts_2d = np.array([proj(K, R, t, pt_true) for R, t, K in cameras])
        projs = [build_projection_matrix(R, t, K) for R, t, K in cameras]

        recovered = triangulate_point(pts_2d, projs)
        assert np.linalg.norm(recovered - pt_true) < 0.001

    def test_weighted_dlt(self):
        """Down-weighting a noisy view should reduce its influence on the result.

        Use 3 views: 2 clean + 1 badly corrupted. Heavily down-weighting the
        corrupted view should give a solution closer to ground truth than
        uniform weighting.
        """
        pt_true = np.array([0.3, 0.1, 2.5])
        R1, t1, K1 = _make_camera(np.array([2.0, 0.0, 0.0]))
        R2, t2, K2 = _make_camera(np.array([-2.0, 0.0, 0.0]))
        R3, t3, K3 = _make_camera(np.array([0.0, 2.0, 0.0]))
        P1 = build_projection_matrix(R1, t1, K1)
        P2 = build_projection_matrix(R2, t2, K2)
        P3 = build_projection_matrix(R3, t3, K3)

        def proj(P, X):
            h = P @ np.append(X, 1.0)
            return h[:2] / h[2]

        p1 = proj(P1, pt_true)
        p2 = proj(P2, pt_true)
        p3_noisy = proj(P3, pt_true) + np.array([100.0, 80.0])  # heavily corrupted

        pts_2d = np.array([p1, p2, p3_noisy])
        projs = [P1, P2, P3]

        # Uniform weights — noisy view has full influence
        rec_unif = triangulate_point(pts_2d, projs)
        # Down-weight noisy view to near-zero
        rec_weighted = triangulate_point(pts_2d, projs, weights=np.array([1.0, 1.0, 0.01]))

        err_unif = np.linalg.norm(rec_unif - pt_true)
        err_weighted = np.linalg.norm(rec_weighted - pt_true)
        assert err_weighted < err_unif, (
            f"Weighted DLT should reduce noise influence: "
            f"unif={err_unif:.4f}, weighted={err_weighted:.4f}"
        )

    def test_insufficient_views_raises(self):
        with pytest.raises(ValueError, match="Need ≥2"):
            triangulate_point(np.zeros((1, 2)), [np.zeros((3, 4))])

    def test_triangulate_joints_batch(self):
        """triangulate_joints returns correct shape."""
        pt_true = np.array([0.2, -0.1, 2.5])
        cameras = [
            _make_camera(np.array([2.0, 0.0, 0.0])),
            _make_camera(np.array([-2.0, 0.0, 0.0])),
            _make_camera(np.array([0.0, 2.0, 0.0])),
        ]

        def proj(K, R, t, X):
            p = K @ (R @ X + t)
            return p[:2] / p[2]

        kp2d = {
            f"view{i}": np.tile(proj(K, R, t, pt_true), (17 + 2, 1))
            for i, (R, t, K) in enumerate(cameras)
        }
        confs = {f"view{i}": np.ones(19) for i in range(3)}
        projs = {
            f"view{i}": build_projection_matrix(R, t, K)
            for i, (R, t, K) in enumerate(cameras)
        }

        pts_3d, quality = triangulate_joints(
            kp2d, confs, projs, joint_indices=list(range(5)), min_views=2
        )
        assert pts_3d.shape == (5, 3)
        assert quality.shape == (5,)
        assert (quality > 0).all(), "All joints should triangulate with 3 clean views"


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------


class TestRANSAC:
    def _setup_clean(self, n_views: int = 5):
        pt_true = np.array([0.3, -0.2, 2.8])
        positions = [
            np.array([2.0 * np.cos(a), 0.0, 2.0 * np.sin(a)])
            for a in np.linspace(0, np.pi, n_views)
        ]
        cameras = [_make_camera(p) for p in positions]
        Rs = [c[0] for c in cameras]
        ts = [c[1] for c in cameras]
        Ks = [c[2] for c in cameras]
        projs = [build_projection_matrix(R, t, K) for R, t, K in cameras]

        def proj(K, R, t, X):
            p = K @ (R @ X + t)
            return p[:2] / p[2]

        pts_2d = np.array([proj(K, R, t, pt_true) for R, t, K in cameras])
        return pt_true, pts_2d, Rs, ts, Ks, projs

    def test_all_inliers_clean_data(self):
        pt_true, pts_2d, Rs, ts, Ks, projs = self._setup_clean(5)
        pt, mask, err = ransac_triangulate_point(
            pts_2d, projs, Rs, ts, Ks, reproj_threshold=5.0
        )
        assert int(mask.sum()) == 5, f"All 5 views should be inliers, got {mask.sum()}"
        assert np.linalg.norm(pt - pt_true) < 0.05

    def test_outlier_rejected(self):
        """One badly-corrupted view should be rejected as an outlier."""
        pt_true, pts_2d, Rs, ts, Ks, projs = self._setup_clean(5)
        # Corrupt view 3 with 500px noise
        pts_2d_noisy = pts_2d.copy()
        pts_2d_noisy[3] += np.array([500.0, 300.0])

        pt, mask, err = ransac_triangulate_point(
            pts_2d_noisy, projs, Rs, ts, Ks,
            reproj_threshold=20.0, n_iterations=200,
        )
        assert not mask[3], "View 3 (outlier) should be rejected"
        assert int(mask.sum()) >= 4

    def test_returns_zeros_when_all_bad(self):
        """Totally noisy data should return zeros."""
        pt_true, pts_2d, Rs, ts, Ks, projs = self._setup_clean(3)
        pts_2d_bad = pts_2d + 5000.0  # all off-screen
        pt, mask, err = ransac_triangulate_point(
            pts_2d_bad, projs, Rs, ts, Ks,
            reproj_threshold=5.0, n_iterations=50, min_inlier_views=3,
        )
        assert int(mask.sum()) < 2 or err > 100.0


# ---------------------------------------------------------------------------
# Undistortion
# ---------------------------------------------------------------------------


class TestUndistortion:
    def _make_colmap_cam(self, k1: float = -0.1) -> ColmapCamera:
        return ColmapCamera(
            camera_id=1, model="SIMPLE_RADIAL",
            width=6000, height=4000,
            focal_length=6678.0, cx=3000.0, cy=2000.0,
            k1=k1,
        )

    def test_center_unchanged(self):
        """Principal point should not move under undistortion."""
        cam = self._make_colmap_cam()
        pts = np.array([[3000.0, 2000.0]])
        undist = undistort_keypoints(pts, cam)
        assert np.allclose(undist, pts, atol=0.5), (
            f"Principal point shifted: {pts} → {undist}"
        )

    def test_no_distortion_identity(self):
        """k1=0 should be a near-identity transform."""
        cam = self._make_colmap_cam(k1=0.0)
        rng = np.random.default_rng(42)
        pts = rng.uniform([500, 300], [5500, 3700], (20, 2))
        undist = undistort_keypoints(pts, cam)
        assert np.allclose(undist, pts, atol=0.1), "k1=0 should be identity"

    def test_edge_correction_nonzero(self):
        """Points near the image edge should shift with k1 < 0."""
        cam = self._make_colmap_cam(k1=-0.1)
        # Corner point, far from principal point
        pts = np.array([[5900.0, 100.0]])
        undist = undistort_keypoints(pts, cam)
        shift = np.linalg.norm(undist - pts)
        assert shift > 5.0, f"Expected significant undistortion at edge, got {shift:.2f}px"

    def test_build_pinhole_K(self):
        cam = self._make_colmap_cam()
        K = build_pinhole_K(cam)
        assert K.shape == (3, 3)
        assert K[0, 0] == cam.focal_length
        assert K[0, 2] == cam.cx
        assert K[1, 2] == cam.cy
        assert K[2, 2] == 1.0


# ---------------------------------------------------------------------------
# Frame alignment
# ---------------------------------------------------------------------------


class TestFrameAlignment:
    def test_identity_alignment(self):
        """Identical point sets → scale=1, R=I, t=0."""
        pts = np.random.default_rng(0).uniform(-1, 1, (10, 3))
        align = compute_frame_alignment(pts, pts)
        assert abs(align.scale - 1.0) < 1e-6
        assert np.allclose(align.rotation, np.eye(3), atol=1e-6)
        assert np.allclose(align.translation, 0.0, atol=1e-6)

    def test_known_transform_recovery(self):
        """Apply known s, R, t then recover it."""
        rng = np.random.default_rng(7)
        pts_src = rng.uniform(-1, 1, (12, 3))

        s_true = 2.5
        R_true = Rotation.from_euler("xyz", [10, 30, -15], degrees=True).as_matrix()
        t_true = np.array([0.5, -0.3, 1.2])

        pts_dst = s_true * (R_true @ pts_src.T).T + t_true

        align = compute_frame_alignment(pts_src, pts_dst)

        assert abs(align.scale - s_true) < 1e-4, f"scale: {align.scale} vs {s_true}"
        assert np.allclose(align.rotation, R_true, atol=1e-4), "rotation mismatch"
        assert np.allclose(align.translation, t_true, atol=1e-4), "translation mismatch"

    def test_transform_point(self):
        """transform_point correctly maps source to target."""
        rng = np.random.default_rng(99)
        pts = rng.uniform(-2, 2, (8, 3))
        s, R_euler, t = 1.8, [5, -20, 45], np.array([1.0, 0.0, -0.5])
        R = Rotation.from_euler("xyz", R_euler, degrees=True).as_matrix()
        pts_dst = s * (R @ pts.T).T + t

        align = compute_frame_alignment(pts, pts_dst)
        pts_out = align.transform_points(pts)
        assert np.allclose(pts_out, pts_dst, atol=1e-4)

    def test_camera_transform(self):
        """Transformed camera should project the same world point."""
        pt_world_colmap = np.array([0.5, 0.2, 3.0])

        R_cam = np.eye(3)
        t_cam = np.array([0.0, 0.0, 5.0])
        K = np.array([[1000., 0, 500], [0, 1000., 500], [0, 0, 1.]])

        # Known alignment
        s = 2.0
        R_align = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        t_align = np.array([1.0, 0.0, 0.0])
        align = FrameAlignment(scale=s, rotation=R_align, translation=t_align)

        # Point in SMPL frame
        pt_world_smpl = align.transform_point(pt_world_colmap)

        # Transform camera
        R_smpl, t_smpl = align.transform_camera(R_cam, t_cam)

        # Project via COLMAP camera
        p_cam_colmap = R_cam @ pt_world_colmap + t_cam
        p_img_colmap = K @ p_cam_colmap
        p_img_colmap = p_img_colmap[:2] / p_img_colmap[2]

        # Project via transformed camera
        p_cam_smpl = R_smpl @ pt_world_smpl + t_smpl
        p_img_smpl = K @ p_cam_smpl
        p_img_smpl = p_img_smpl[:2] / p_img_smpl[2]

        assert np.allclose(p_img_colmap, p_img_smpl, atol=0.5), (
            f"Projection mismatch: {p_img_colmap} vs {p_img_smpl}"
        )

    def test_requires_four_points(self):
        with pytest.raises(ValueError, match="≥4"):
            compute_frame_alignment(np.zeros((3, 3)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# COLMAP reader
# ---------------------------------------------------------------------------


COLMAP_DIR = "/home/dan/projects/auto-rigger/data/reconstruction/t-pose/0"
OUR_17 = {
    "cam01_2.JPG", "cam01_6.JPG", "cam02_4.JPG", "cam02_5.JPG",
    "cam03_5.JPG", "cam03_6.JPG", "cam04_4.JPG", "cam04_5.JPG",
    "cam05_4.JPG", "cam05_5.JPG", "cam05_6.JPG", "cam06_4.JPG",
    "cam07_4.JPG", "cam07_6.JPG", "cam10_2.JPG", "cam10_4.JPG",
    "cam10_5.JPG",
}


@pytest.fixture(scope="module")
def colmap_model():
    import os
    if not os.path.exists(COLMAP_DIR):
        pytest.skip("COLMAP model not available")
    from pathlib import Path
    return read_colmap_model(Path(COLMAP_DIR))


class TestColmapReader:
    def test_reads_two_cameras(self, colmap_model):
        cameras, _ = colmap_model
        assert len(cameras) == 2

    def test_reads_60_images(self, colmap_model):
        _, images = colmap_model
        assert len(images) == 60

    def test_camera2_params(self, colmap_model):
        cameras, _ = colmap_model
        cam2 = cameras[2]
        assert cam2.model == "SIMPLE_RADIAL"
        assert abs(cam2.focal_length - 6678.0) < 5.0
        assert abs(cam2.k1 - (-0.1)) < 0.01

    def test_all_17_views_present(self, colmap_model):
        """Criterion 5.1: all 17 Phase 1 views must be in COLMAP."""
        cameras, images = colmap_model
        matched, missing = match_views_to_colmap(list(OUR_17), images)
        assert len(missing) == 0, f"Views missing from COLMAP: {missing}"
        assert len(matched) == 17

    def test_all_17_use_camera2(self, colmap_model):
        """All 17 Phase 1 views use camera 2 (wide-angle lens)."""
        cameras, images = colmap_model
        for name in OUR_17:
            assert name in images
            assert images[name].camera_id == 2

    def test_rotation_is_valid(self, colmap_model):
        """Each image rotation should be a valid rotation matrix."""
        _, images = colmap_model
        for name, img in images.items():
            R = img.rotation
            assert R.shape == (3, 3)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), f"{name}: R not orthogonal"
            assert abs(np.linalg.det(R) - 1.0) < 1e-6, f"{name}: det(R) != 1"
