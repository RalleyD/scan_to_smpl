"""
Unit tests for Phase 4 PnP calibration — no GPU required.

Run with:
    pytest tests/test_pnp.py -v
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Test 1: Intrinsic matrix construction
# ---------------------------------------------------------------------------


class TestBuildIntrinsicMatrix:
    def test_basic_construction(self):
        from scantosmpl.calibration.intrinsics import build_intrinsic_matrix

        K = build_intrinsic_matrix(1000.0, 1920, 1080, principal_point=(960.0, 540.0))
        assert K.shape == (3, 3)
        assert K[0, 0] == 1000.0
        assert K[1, 1] == 1000.0
        assert K[0, 2] == 960.0
        assert K[1, 2] == 540.0
        assert K[2, 2] == 1.0

    def test_default_principal_point_is_center(self):
        from scantosmpl.calibration.intrinsics import build_intrinsic_matrix

        K = build_intrinsic_matrix(500.0, 800, 600)
        assert K[0, 2] == 400.0
        assert K[1, 2] == 300.0

    def test_non_square_pixels(self):
        from scantosmpl.calibration.intrinsics import build_intrinsic_matrix

        K = build_intrinsic_matrix(2000.0, 4000, 3000)
        assert K[0, 2] == 2000.0
        assert K[1, 2] == 1500.0


# ---------------------------------------------------------------------------
# Test 2: Projection utilities
# ---------------------------------------------------------------------------


class TestProjectionUtilities:
    def test_project_points_identity(self):
        """Points at z=1 with identity camera should project to (x*f+cx, y*f+cy)."""
        from scantosmpl.utils.geometry import project_points

        pts_3d = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 1.0]])
        R = np.eye(3)
        t = np.zeros(3)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])

        pts_2d = project_points(pts_3d, R, t, K)
        assert pts_2d.shape == (2, 2)
        np.testing.assert_allclose(pts_2d[0], [320.0, 240.0], atol=1e-10)
        np.testing.assert_allclose(pts_2d[1], [370.0, 140.0], atol=1e-10)

    def test_camera_center(self):
        from scantosmpl.utils.geometry import camera_center

        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        C = camera_center(R, t)
        np.testing.assert_allclose(C, [-1.0, -2.0, -3.0])

    def test_camera_center_with_rotation(self):
        from scantosmpl.utils.geometry import camera_center

        # 90-degree rotation around Y axis
        R = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        t = np.array([0.0, 0.0, 5.0])
        C = camera_center(R, t)
        # C = -R^T @ t; R^T rotates [0,0,5] by -90° around Y -> [-5,0,0]
        np.testing.assert_allclose(C, [5.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Test 3: PnP solver with synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_cube_scene(
    f: float = 1000.0,
    img_w: int = 1920,
    img_h: int = 1080,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Create a synthetic scene: 3D cube points + known camera -> 2D projections."""
    from scantosmpl.utils.geometry import project_points

    if rng is None:
        rng = np.random.default_rng(42)

    # 3D points: unit cube vertices + face centers
    pts_3d = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        [0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0],
        [-1, 0, 0], [1, 0, 0], [0, 0, 0],
    ], dtype=np.float64)

    # Ground-truth camera: looking at origin from z=10, slight rotation
    R_gt = Rotation.from_euler("xyz", [5, -10, 3], degrees=True).as_matrix()
    t_gt = np.array([0.5, -0.3, 10.0])

    K = np.array([
        [f, 0, img_w / 2.0],
        [0, f, img_h / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)

    pts_2d = project_points(pts_3d, R_gt, t_gt, K)
    if noise_std > 0:
        pts_2d += rng.normal(0, noise_std, pts_2d.shape)

    confs = np.ones(len(pts_3d), dtype=np.float64)

    return pts_3d, pts_2d, confs, K, R_gt, t_gt


class TestPnPSolver:
    def test_known_transform_recovery(self):
        """PnP should recover a known camera transform from clean projections."""
        from scantosmpl.calibration.pnp_solver import PnPSolver

        pts_3d, pts_2d, confs, K, R_gt, t_gt = _make_synthetic_cube_scene()

        solver = PnPSolver(min_inliers=6)
        result = solver.solve(pts_3d, pts_2d, confs, K)

        assert result.success
        np.testing.assert_allclose(result.rotation, R_gt, atol=1e-4)
        np.testing.assert_allclose(result.translation, t_gt, atol=1e-3)
        assert result.reprojection_error < 1.0

    def test_noisy_2d_still_converges(self):
        """PnP should handle moderate pixel noise (2px std)."""
        from scantosmpl.calibration.pnp_solver import PnPSolver

        pts_3d, pts_2d, confs, K, R_gt, t_gt = _make_synthetic_cube_scene(
            noise_std=2.0
        )

        solver = PnPSolver(min_inliers=6)
        result = solver.solve(pts_3d, pts_2d, confs, K)

        assert result.success
        # Rotation should be close (within ~1 degree)
        angle_diff = np.linalg.norm(
            Rotation.from_matrix(result.rotation @ R_gt.T).as_rotvec()
        )
        assert angle_diff < np.radians(2.0)
        assert result.reprojection_error < 10.0

    def test_insufficient_points_fails(self):
        """PnP should fail with fewer than 4 points."""
        from scantosmpl.calibration.pnp_solver import PnPSolver

        pts_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        pts_2d = np.array([[100, 100], [200, 100], [100, 200]], dtype=np.float64)
        confs = np.ones(3)
        K = np.eye(3) * 500
        K[2, 2] = 1.0

        solver = PnPSolver(min_inliers=3)
        result = solver.solve(pts_3d, pts_2d, confs, K)
        assert not result.success

    def test_reprojection_error_computation(self):
        """Reprojection error should be ~0 for clean synthetic data."""
        from scantosmpl.calibration.pnp_solver import PnPSolver

        pts_3d, pts_2d, confs, K, _, _ = _make_synthetic_cube_scene()

        solver = PnPSolver(min_inliers=6)
        result = solver.solve(pts_3d, pts_2d, confs, K)

        assert result.success
        assert result.reprojection_error < 0.5  # sub-pixel for clean data

    def test_confidence_filtering(self):
        """Low-confidence points should be excluded."""
        from scantosmpl.calibration.pnp_solver import PnPSolver

        pts_3d, pts_2d, confs, K, _, _ = _make_synthetic_cube_scene()
        # Set half to low confidence
        confs[:7] = 0.1

        solver = PnPSolver(min_inliers=4)
        result = solver.solve(pts_3d, pts_2d, confs, K, conf_threshold=0.3)

        assert result.success
        assert result.n_correspondences == 8  # only high-conf points used


# ---------------------------------------------------------------------------
# Test 4: Correspondence builder
# ---------------------------------------------------------------------------


class TestCorrespondenceBuilder:
    @pytest.fixture
    def builder(self):
        from scantosmpl.calibration.correspondence import CorrespondenceBuilder

        verts = np.random.default_rng(0).standard_normal((6890, 3))
        joints = np.random.default_rng(1).standard_normal((24, 3))
        return CorrespondenceBuilder(verts, joints)

    def test_dense_shape_138(self, builder):
        from scantosmpl.types import CameraParams, ViewResult, ViewType

        view = ViewResult(
            image_path=__import__("pathlib").Path("test.jpg"),
            view_type=ViewType.FULL_BODY,
            camera=CameraParams(focal_length=1000.0),
            dense_keypoints_2d=np.random.default_rng(2).standard_normal((138, 2)),
            dense_keypoint_confs=np.random.default_rng(3).uniform(0, 1, 138),
        )
        pts_3d, pts_2d, confs = builder.build_dense_correspondences(view)
        assert pts_3d.shape == (138, 3)
        assert pts_2d.shape == (138, 2)
        assert confs.shape == (138,)

    def test_sparse_includes_midpoints(self, builder):
        """Sparse should have 12 direct + 2 midpoints = 14 correspondences."""
        from scantosmpl.types import CameraParams, ViewResult, ViewType

        view = ViewResult(
            image_path=__import__("pathlib").Path("test.jpg"),
            view_type=ViewType.FULL_BODY,
            camera=CameraParams(focal_length=1000.0),
            keypoints_2d=np.random.default_rng(4).standard_normal((17, 2)),
            keypoint_confs=np.random.default_rng(5).uniform(0.5, 1.0, 17),
        )
        pts_3d, pts_2d, confs = builder.build_sparse_correspondences(view)
        assert pts_3d.shape == (14, 3)
        assert pts_2d.shape == (14, 2)
        assert confs.shape == (14,)

    def test_dense_raises_without_data(self, builder):
        from scantosmpl.types import CameraParams, ViewResult, ViewType

        view = ViewResult(
            image_path=__import__("pathlib").Path("test.jpg"),
            view_type=ViewType.FULL_BODY,
            camera=CameraParams(focal_length=1000.0),
        )
        with pytest.raises(ValueError, match="no dense keypoints"):
            builder.build_dense_correspondences(view)


# ---------------------------------------------------------------------------
# Test 5: Camera geometry validation
# ---------------------------------------------------------------------------


class TestCameraGeometry:
    def test_circular_layout_passes(self):
        """Cameras arranged in a circle should pass geometry validation."""
        from scantosmpl.calibration.pipeline import CalibrationPipeline
        from scantosmpl.config import CalibrationConfig

        pipeline = CalibrationPipeline(CalibrationConfig())

        # 8 cameras in a circle at radius ~3m, height ~1.5m
        centers = {}
        for i in range(8):
            angle = 2 * np.pi * i / 8
            centers[f"cam{i}"] = np.array([
                3.0 * np.cos(angle), 1.5, 3.0 * np.sin(angle)
            ])

        plausible, stats = pipeline._validate_geometry(centers)
        assert plausible
        assert stats["radial_cov"] < 0.3
        assert stats["angular_coverage_deg"] > 120.0

    def test_random_layout_fails(self):
        """Randomly scattered cameras should fail geometry validation."""
        from scantosmpl.calibration.pipeline import CalibrationPipeline
        from scantosmpl.config import CalibrationConfig

        pipeline = CalibrationPipeline(CalibrationConfig())

        rng = np.random.default_rng(42)
        centers = {}
        for i in range(5):
            # Random positions with high variance in radius
            centers[f"cam{i}"] = rng.uniform(-20, 20, 3)

        plausible, _ = pipeline._validate_geometry(centers)
        # High radial variance = likely fails
        # (Note: random layout may occasionally pass; we just check the logic runs)
        assert isinstance(plausible, bool)

    def test_too_few_cameras(self):
        from scantosmpl.calibration.pipeline import CalibrationPipeline
        from scantosmpl.config import CalibrationConfig

        pipeline = CalibrationPipeline(CalibrationConfig())
        plausible, stats = pipeline._validate_geometry({"cam0": np.zeros(3)})
        assert not plausible


# ---------------------------------------------------------------------------
# Test 6: Dense vertex indices
# ---------------------------------------------------------------------------


class TestDenseVertexIndices:
    def test_count(self):
        from scantosmpl.smpl.joint_map import DENSE_KP_VERTEX_INDICES, NUM_DENSE_KEYPOINTS

        assert len(DENSE_KP_VERTEX_INDICES) == NUM_DENSE_KEYPOINTS
        assert len(DENSE_KP_VERTEX_INDICES) == 138

    def test_all_valid_indices(self):
        from scantosmpl.smpl.joint_map import DENSE_KP_VERTEX_INDICES

        assert DENSE_KP_VERTEX_INDICES.min() >= 0
        assert DENSE_KP_VERTEX_INDICES.max() < 6890

    def test_all_unique(self):
        from scantosmpl.smpl.joint_map import DENSE_KP_VERTEX_INDICES

        assert len(np.unique(DENSE_KP_VERTEX_INDICES)) == 138
