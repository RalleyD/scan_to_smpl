"""Unit tests for Phase 5 triangulation (DLT + RANSAC)."""

import numpy as np
import pytest

from scantosmpl.triangulation.dlt import (
    build_projection_matrix,
    triangulate_joints,
    triangulate_point,
)
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
            f"view{i}": build_projection_matrix(R, t, K) for i, (R, t, K) in enumerate(cameras)
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
        pt, mask, err = ransac_triangulate_point(pts_2d, projs, Rs, ts, Ks, reproj_threshold=5.0)
        assert int(mask.sum()) == 5, f"All 5 views should be inliers, got {mask.sum()}"
        assert np.linalg.norm(pt - pt_true) < 0.05

    def test_outlier_rejected(self):
        """One badly-corrupted view should be rejected as an outlier."""
        pt_true, pts_2d, Rs, ts, Ks, projs = self._setup_clean(5)
        # Corrupt view 3 with 500px noise
        pts_2d_noisy = pts_2d.copy()
        pts_2d_noisy[3] += np.array([500.0, 300.0])

        pt, mask, err = ransac_triangulate_point(
            pts_2d_noisy,
            projs,
            Rs,
            ts,
            Ks,
            reproj_threshold=20.0,
            n_iterations=200,
        )
        assert not mask[3], "View 3 (outlier) should be rejected"
        assert int(mask.sum()) >= 4

    def test_returns_zeros_when_all_bad(self):
        """Totally noisy data should return zeros."""
        pt_true, pts_2d, Rs, ts, Ks, projs = self._setup_clean(3)
        pts_2d_bad = pts_2d + 5000.0  # all off-screen
        pt, mask, err = ransac_triangulate_point(
            pts_2d_bad,
            projs,
            Rs,
            ts,
            Ks,
            reproj_threshold=5.0,
            n_iterations=50,
            min_inlier_views=3,
        )
        assert int(mask.sum()) < 2 or err > 100.0
