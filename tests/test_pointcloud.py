"""Unit tests for `scantosmpl.pointcloud` (Phase 6 input path).

Everything here is pure geometry / IO — no SMPL model weights are required
except the two tests explicitly marked `requires_smpl`, which exercise
`smpl_part_labels` against the real (6890, 24) `lbs_weights` array for extra
confidence beyond the synthetic one-hot test.

Run with:
    pytest tests/test_pointcloud.py -v
    pytest tests/test_pointcloud.py -k preprocess -v
    pytest tests/test_pointcloud.py -k align -v
    pytest tests/test_pointcloud.py -k segment -v
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation

from scantosmpl.evaluation.surface_metrics import sample_surface
from scantosmpl.pointcloud import align as align_module
from scantosmpl.pointcloud.align import (
    N_PROPER_ROTATIONS,
    CloudAlignment,
    align_cloud_to_smpl,
    enumerate_proper_rotations,
    pca_triad,
)
from scantosmpl.pointcloud.io import PointCloud, load_pointcloud, save_pointcloud
from scantosmpl.pointcloud.preprocess import PreprocessStats, bbox_diagonal, preprocess_cloud
from scantosmpl.pointcloud.segment import (
    SMPL_NUM_JOINTS,
    SMPL_PART_GROUPS,
    smpl_part_labels,
    transfer_labels_to_cloud,
    vertex_part_weights,
)

# ---------------------------------------------------------------------------
# Shared config stand-ins (Tier3Config is owned by another brief — these
# Protocol-satisfying dataclasses duplicate only the fields each module reads,
# per master Section 5.2's locked field names/defaults).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _PreprocessCfg:
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    target_points: int = 50_000
    voxel_fraction_of_bbox: float = 0.002
    estimate_normals: bool = True
    normal_knn: int = 30


@dataclasses.dataclass
class _AlignCfg:
    icp_max_iterations: int = 100
    icp_threshold_frac: float = 0.05
    icp_min_fitness: float = 0.5
    max_scale_deviation_factor: float = 3.0


#: Master Section 5.2's `Tier3Config.body_part_weights` default keys, duplicated
#: here (not imported — `Tier3Config` is owned by another brief) as the
#: documented contract `SMPL_PART_GROUPS` must match exactly (D7).
_TIER3_BODY_PART_WEIGHT_KEYS = {"torso", "arms", "legs", "head", "hands", "feet"}

EMPTY_PLY_TEXT = (
    "ply\nformat ascii 1.0\nelement vertex 0\n"
    "property float x\nproperty float y\nproperty float z\nend_header\n"
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rotation_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(unit * np.deg2rad(angle_deg)).as_matrix()


def _geodesic_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    relative = r1.T @ r2
    cos_theta = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _merge_meshes(meshes: list[o3d.geometry.TriangleMesh]) -> tuple[np.ndarray, np.ndarray]:
    verts_list = []
    faces_list = []
    offset = 0
    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.triangles, dtype=np.int64) + offset
        verts_list.append(v)
        faces_list.append(f)
        offset += v.shape[0]
    return np.vstack(verts_list), np.vstack(faces_list)


def _ellipsoid(
    radii: tuple[float, float, float], center: tuple[float, float, float], resolution: int
) -> o3d.geometry.TriangleMesh:
    """A unit sphere non-uniformly scaled per axis (Open3D's own `.scale()` is
    uniform-only) then translated -- a smooth, continuously-curved body part."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    verts = np.asarray(mesh.vertices) * np.array(radii)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.translate(center)
    return mesh


def _composite_body_mesh(resolution: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic, roughly human-scale, asymmetric mesh — plays the role of
    the Tier 2 SMPL mesh in the alignment tests. Asymmetric (a single arm, a
    single leg, an off-centre head) so it has no residual proper-rotation
    symmetry: exactly one of the 24 candidates is the correct basin.

    Two fixes here, both needed once the source cloud is a genuine SURFACE SAMPLE
    (see `_sample_mesh_surface`) rather than an exact copy of this mesh's own
    vertices (P0/P1 fix, iteration 1; the earlier exact-vertex-copy construction
    didn't need either):

    1. **Resolution.** An earlier (box-based, `subdivide_midpoint(1)`) version of
       this fixture had only ~260 vertices; that's sparse enough to under-constrain
       ICP correspondence and the true candidate would converge 50-170deg off even
       from a near-correct start. Ellipsoids at this `resolution` give ~2000+
       vertices — comfortably past that failure point (measured: true candidate
       converges <0.1deg off, `_target_coverage` decisively above every other
       candidate's). The real SMPL mesh (6890 vertices over a similar overall size)
       is well past this density already; this fixture just needs enough of its own.
    2. **Curvature.** The earlier version's torso/arm/leg were AXIS-ALIGNED BOXES:
       large flat faces are the textbook ICP "aperture problem" — point-to-plane
       residuals barely change for a translation *along* a flat face, so even an
       otherwise-correct fit can sit several mm off in that direction with no
       gradient pulling it back (measured: ~8mm stuck translation error, insensitive
       to further ICP polishing of any kind, because it is a genuine shallow-valley
       geometric ambiguity of a box shape, not an optimiser/threshold problem).
       Ellipsoids have no flat regions, so every direction is well-constrained
       (measured: translation error <2mm at the same rotation/scale/translation this
       fixture used to leave ~8mm on the table with boxes).
    """
    torso = _ellipsoid((0.17, 0.10, 0.275), (0.0, 0.0, 1.125), resolution)
    head = _ellipsoid((0.11, 0.11, 0.11), (0.0, 0.0, 1.55), resolution)
    arm = _ellipsoid((0.05, 0.05, 0.225), (0.25, 0.03, 1.175), resolution)
    leg = _ellipsoid((0.06, 0.06, 0.375), (-0.08, 0.02, 0.425), resolution)
    return _merge_meshes([torso, head, arm, leg])


def _grid_cloud(n_per_axis: int = 21) -> np.ndarray:
    """A deterministic (no RNG) dense grid over the unit cube [0, 1]^3."""
    lin = np.linspace(0.0, 1.0, n_per_axis)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _sample_mesh_surface(
    vertices: np.ndarray, faces: np.ndarray, n_points: int, seed: int
) -> np.ndarray:
    """Deterministic-given-`seed` area-weighted sample of points ON the mesh surface.

    P0/P1 regression fix: a source cloud built as an exact copy of the target mesh's
    own VERTICES (as the old versions of these tests did) attains an inlier RMSE of
    ~0 for the correctly-aligned candidate, which trivially beats any collapsed
    candidate and can never exercise the failure mode this test file exists to catch.
    A real scan samples the surface *between* vertices, which is exactly what makes
    the SMPL mesh's own tessellation floor (measured ~8mm mean) large enough that a
    scale-collapsed candidate's (degenerate, vertex-tessellation-immune) RMSE could
    beat it under the old raw-`inlier_rmse` selection criterion. This is test-only
    seeded randomness (a fixture generator, master D12's own carve-out) -- nothing in
    `scantosmpl/pointcloud/align.py` itself uses an RNG.

    Thin wrapper, kept for this file's local `n_points=`/positional-`seed` call
    style: the actual area-weighted barycentric sampling is
    `scantosmpl.evaluation.surface_metrics.sample_surface`, already implemented
    and tested there — reused rather than duplicated.
    """
    return sample_surface(vertices, faces, n_samples=n_points, seed=seed)


def _height_slice_source(
    verts: np.ndarray,
    faces: np.ndarray,
    frac_lo: float,
    frac_hi: float,
    *,
    n_samples: int,
    seed: int,
    rotation_true: np.ndarray,
    scale_true: float,
    translation_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """A partial-body probe (P1 regression, iteration 2): surface-sample the mesh,
    keep only the vertical (z, in the mesh/SMPL frame) slice between
    ``frac_lo``/``frac_hi`` of its height, then map that slice back through the
    INVERSE of a known ground-truth similarity into a synthetic source-frame cloud —
    mirroring how a genuinely partial scan (only a head, only legs, only feet) would
    arrive at `align_cloud_to_smpl`.

    Returns:
        smpl_frame_points: the kept slice, still in the mesh/SMPL frame (metres) —
            the ground truth an accepted alignment should reproduce.
        source_points: the same slice, mapped into the synthetic source frame.
    """
    surface = sample_surface(verts, faces, n_samples=n_samples, seed=seed)
    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
    height = zmax - zmin
    lo, hi = zmin + frac_lo * height, zmin + frac_hi * height
    mask = (surface[:, 2] >= lo) & (surface[:, 2] <= hi)
    smpl_frame_points = surface[mask]
    source_points = (smpl_frame_points - translation_true) / scale_true @ rotation_true
    return smpl_frame_points, source_points


def _find_group(joint: int) -> str:
    for name, joints in SMPL_PART_GROUPS.items():
        if joint in joints:
            return name
    raise AssertionError(f"joint {joint} not in any SMPL_PART_GROUPS group")


def _smpl_model_available() -> bool:
    return Path("models/smpl/SMPL_NEUTRAL.pkl").exists()


requires_smpl = pytest.mark.skipif(
    not _smpl_model_available(),
    reason="SMPL model files not found in models/smpl/ — see models/README.md",
)


# ---------------------------------------------------------------------------
# io.py — PointCloud + load/save
# ---------------------------------------------------------------------------


class TestPointCloudDataclass:
    def test_valid_construction(self):
        pts = np.zeros((5, 3))
        cloud = PointCloud(points=pts, normals=None, colors=None, source_path=Path("x.ply"))
        assert cloud.n_points == 5
        assert cloud.frame == "source"
        assert cloud.units == "arbitrary"

    def test_bad_points_shape_raises(self):
        with pytest.raises(ValueError):
            PointCloud(points=np.zeros((5, 2)), normals=None, colors=None, source_path=Path("x"))

    def test_mismatched_normals_shape_raises(self):
        with pytest.raises(ValueError):
            PointCloud(
                points=np.zeros((5, 3)),
                normals=np.zeros((4, 3)),
                colors=None,
                source_path=Path("x"),
            )

    def test_mismatched_colors_shape_raises(self):
        with pytest.raises(ValueError):
            PointCloud(
                points=np.zeros((5, 3)),
                normals=None,
                colors=np.zeros((3, 3)),
                source_path=Path("x"),
            )

    def test_to_open3d_round_trip(self):
        pts = np.random.default_rng(0).normal(size=(20, 3))
        cloud = PointCloud(points=pts, normals=None, colors=None, source_path=Path("x.ply"))
        pcd = cloud.to_open3d()
        assert np.allclose(np.asarray(pcd.points), pts)


class TestLoadPointcloud:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pointcloud(tmp_path / "nope.ply")

    def test_unsupported_suffix_raises_value_error(self, tmp_path):
        bad = tmp_path / "cloud.xyz"
        bad.write_text("not a real point cloud")
        with pytest.raises(ValueError):
            load_pointcloud(bad)

    def test_empty_ply_raises_value_error(self, tmp_path):
        empty = tmp_path / "empty.ply"
        empty.write_text(EMPTY_PLY_TEXT)
        with pytest.raises(ValueError):
            load_pointcloud(empty)

    def test_save_then_load_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(200, 3))
        normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        colors = rng.uniform(0.0, 1.0, size=(200, 3)).astype(np.float32)
        cloud = PointCloud(points=pts, normals=normals, colors=colors, source_path=Path("orig.ply"))

        out_path = tmp_path / "roundtrip.ply"
        save_pointcloud(cloud, out_path)
        loaded = load_pointcloud(out_path)

        assert loaded.frame == "source"
        assert loaded.units == "arbitrary"
        assert np.allclose(pts, loaded.points, atol=1e-4)
        assert loaded.normals is not None
        assert np.allclose(normals, loaded.normals, atol=1e-4)
        assert loaded.colors is not None
        # PLY colours round-trip through 8-bit storage.
        assert np.allclose(colors, loaded.colors, atol=1.0 / 255.0 + 1e-6)

    def test_save_pointcloud_requires_ply_suffix(self, tmp_path):
        cloud = PointCloud(
            points=np.zeros((3, 3)), normals=None, colors=None, source_path=Path("x")
        )
        with pytest.raises(ValueError):
            save_pointcloud(cloud, tmp_path / "out.obj")

    def test_max_points_deterministic_stride_subsample(self, tmp_path):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(1000, 3))
        path = tmp_path / "big.ply"
        save_pointcloud(
            PointCloud(points=pts, normals=None, colors=None, source_path=Path("b.ply")), path
        )

        loaded_a = load_pointcloud(path, max_points=100)
        loaded_b = load_pointcloud(path, max_points=100)

        assert loaded_a.n_points <= 100
        # Deterministic stride, not random (D12): identical across runs.
        assert np.array_equal(loaded_a.points, loaded_b.points)

    def test_obj_mesh_contributes_vertices(self, tmp_path):
        mesh = o3d.geometry.TriangleMesh.create_box()
        obj_path = tmp_path / "box.obj"
        o3d.io.write_triangle_mesh(str(obj_path), mesh)

        loaded = load_pointcloud(obj_path)
        assert loaded.n_points == len(mesh.vertices)
        assert loaded.frame == "source"


# ---------------------------------------------------------------------------
# preprocess.py — unit-free cleaning
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_bbox_diagonal_empty_is_zero(self):
        assert bbox_diagonal(np.zeros((0, 3))) == 0.0

    def test_stats_populated(self):
        pts = _grid_cloud(15)
        cloud = PointCloud(points=pts, normals=None, colors=None, source_path=Path("g.ply"))
        cfg = _PreprocessCfg(outlier_nb_neighbors=8, target_points=200, voxel_fraction_of_bbox=0.01)
        out, stats = preprocess_cloud(cloud, cfg)

        assert isinstance(stats, PreprocessStats)
        assert stats.n_input == pts.shape[0]
        assert stats.n_after_outlier_removal <= stats.n_input
        assert stats.n_output == out.n_points
        assert stats.n_output <= 1.05 * cfg.target_points
        assert stats.voxel_size_source_units > 0.0
        assert stats.bbox_diagonal_source_units == pytest.approx(np.sqrt(3.0), abs=1e-6)
        assert stats.normals_estimated is True
        assert out.normals is not None
        assert out.frame == cloud.frame == "source"
        assert out.units == cloud.units == "arbitrary"

    def test_outlier_removal_targets_injected_outliers(self):
        rng = np.random.default_rng(0)
        inliers = rng.normal(loc=0.0, scale=1.0, size=(3000, 3))
        directions = rng.uniform(-1.0, 1.0, size=(60, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        outliers = directions * 30.0  # far outside the inlier blob
        pts = np.vstack([inliers, outliers])
        cloud = PointCloud(points=pts, normals=None, colors=None, source_path=Path("g.ply"))

        cfg = _PreprocessCfg(
            outlier_nb_neighbors=16,
            outlier_std_ratio=2.5,
            target_points=0,  # isolate the outlier-removal step
            estimate_normals=False,
        )
        _, stats = preprocess_cloud(cloud, cfg)

        injected_fraction = 60 / pts.shape[0]
        assert stats.outlier_fraction == pytest.approx(injected_fraction, abs=0.01)
        assert stats.n_after_outlier_removal == 3000

    def test_voxel_downsample_is_scale_invariant(self):
        """The explicit brief requirement: the same cloud scaled by 1000x
        produces the same output point count (+/- 5%) — master D8."""
        pts = _grid_cloud(21)
        cfg = _PreprocessCfg(outlier_nb_neighbors=8, target_points=500, voxel_fraction_of_bbox=0.01)

        cloud_a = PointCloud(points=pts, normals=None, colors=None, source_path=Path("a.ply"))
        cloud_b = PointCloud(
            points=pts * 1000.0, normals=None, colors=None, source_path=Path("b.ply")
        )

        _, stats_a = preprocess_cloud(cloud_a, cfg)
        _, stats_b = preprocess_cloud(cloud_b, cfg)

        assert stats_b.voxel_size_source_units == pytest.approx(
            1000.0 * stats_a.voxel_size_source_units, rel=1e-6
        )
        assert stats_a.n_output == pytest.approx(stats_b.n_output, rel=0.05)

    def test_target_points_zero_skips_downsample(self):
        pts = _grid_cloud(10)
        cloud = PointCloud(points=pts, normals=None, colors=None, source_path=Path("g.ply"))
        cfg = _PreprocessCfg(outlier_nb_neighbors=8, target_points=0, estimate_normals=False)
        out, stats = preprocess_cloud(cloud, cfg)
        assert stats.voxel_size_source_units == 0.0
        assert out.n_points == stats.n_after_outlier_removal

    def test_preprocess_is_deterministic(self):
        pts = _grid_cloud(13)
        cloud = PointCloud(points=pts, normals=None, colors=None, source_path=Path("g.ply"))
        cfg = _PreprocessCfg(outlier_nb_neighbors=8, target_points=100)

        out_a, stats_a = preprocess_cloud(cloud, cfg)
        out_b, stats_b = preprocess_cloud(cloud, cfg)

        assert stats_a == stats_b
        assert np.array_equal(out_a.points, out_b.points)


# ---------------------------------------------------------------------------
# align.py — PCA triad + 24-candidate ICP
# ---------------------------------------------------------------------------


class TestPcaTriad:
    def test_orthonormal_proper_and_ordered(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(4000, 3)) * np.array([3.0, 2.0, 1.0])

        centroid, axes, extents = pca_triad(pts)

        assert centroid.shape == (3,)
        assert axes.shape == (3, 3)
        assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-8)
        assert np.linalg.det(axes) == pytest.approx(1.0, abs=1e-8)
        assert extents[0] >= extents[1] >= extents[2]
        # No rotation applied: the dominant axis should point along global x.
        assert abs(axes[0, 0]) > 0.9

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            pca_triad(np.zeros((2, 3)))


class TestEnumerateProperRotations:
    def test_group_properties(self):
        rotations = enumerate_proper_rotations(np.eye(3), np.eye(3))

        assert len(rotations) == N_PROPER_ROTATIONS == 24
        for r in rotations:
            assert r.shape == (3, 3)
            assert np.allclose(r.T @ r, np.eye(3), atol=1e-9)
            assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)

        for i in range(len(rotations)):
            for j in range(i + 1, len(rotations)):
                assert not np.allclose(rotations[i], rotations[j], atol=1e-9), (i, j)

    def test_deterministic_order(self):
        a = enumerate_proper_rotations(np.eye(3), np.eye(3))
        b = enumerate_proper_rotations(np.eye(3), np.eye(3))
        for ra, rb in zip(a, b, strict=True):
            assert np.array_equal(ra, rb)


class TestCloudAlignment:
    def test_apply_matches_as_matrix(self):
        alignment = CloudAlignment(
            scale=2.0,
            rotation=_rotation_from_axis_angle(np.array([1.0, 1.0, 1.0]), 40.0),
            translation=np.array([0.1, -0.2, 0.3]),
            inlier_rmse_m=0.0,
            fitness=1.0,
            n_candidates=24,
            candidate_index=0,
            converged=True,
            scale_init=2.0,
        )
        points = np.random.default_rng(2).normal(size=(15, 3))
        applied = alignment.apply(points)

        homogeneous = np.hstack([points, np.ones((15, 1))])
        via_matrix = (alignment.as_matrix() @ homogeneous.T).T[:, :3]

        assert np.allclose(applied, via_matrix, atol=1e-9)

    def test_apply_bad_shape_raises(self):
        alignment = CloudAlignment(
            scale=1.0,
            rotation=np.eye(3),
            translation=np.zeros(3),
            inlier_rmse_m=0.0,
            fitness=1.0,
            n_candidates=24,
            candidate_index=0,
            converged=True,
            scale_init=1.0,
        )
        with pytest.raises(ValueError):
            alignment.apply(np.zeros((5, 2)))


class TestAlignCloudToSmpl:
    def test_requires_source_frame(self):
        verts, faces = _composite_body_mesh()
        cloud = PointCloud(
            points=np.zeros((5, 3)),
            normals=None,
            colors=None,
            source_path=Path("x.ply"),
            frame="smpl_world",
            units="metres",
        )
        with pytest.raises(ValueError):
            align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())

    def test_alignment_recovers_ground_truth(self):
        """AC5: known-similarity round-trip on a synthetic non-symmetric shape.

        The source cloud is a SURFACE SAMPLE of the mesh (not its vertices) so the
        correctly-aligned candidate has a genuine, non-zero tessellation-floor RMSE --
        see `_sample_mesh_surface`'s docstring for why an exact-vertex source cloud
        cannot exercise the P0 scale-collapse regression this test guards against.
        """
        verts, faces = _composite_body_mesh()
        surface = _sample_mesh_surface(verts, faces, n_points=6000, seed=0)

        axis = np.array([0.3, -0.5, 0.8])
        rotation_true = _rotation_from_axis_angle(axis, 137.0)
        scale_true = 0.371
        translation_true = np.array([0.05, -0.03, 0.02])

        # p_smpl = scale*(R @ p_source) + t  =>  p_source = R^T @ ((p_smpl - t)/scale)
        p_source = (surface - translation_true) / scale_true @ rotation_true

        cloud = PointCloud(
            points=p_source, normals=None, colors=None, source_path=Path("synthetic.ply")
        )
        aligned, alignment = align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())

        scale_rel_err = abs(alignment.scale / scale_true - 1.0)
        rot_err_deg = _geodesic_angle_deg(alignment.rotation, rotation_true)
        trans_err_m = float(np.linalg.norm(alignment.translation - translation_true))

        assert scale_rel_err < 0.01, f"scale rel err {scale_rel_err}"
        assert rot_err_deg < 1.0, f"rotation err {rot_err_deg} deg"
        assert trans_err_m < 0.005, f"translation err {trans_err_m} m"
        assert alignment.converged is True
        assert alignment.n_candidates == 24
        assert aligned.frame == "smpl_world"
        assert aligned.units == "metres"
        assert np.allclose(aligned.points, alignment.apply(p_source))

    def test_alignment_180_flip_picks_correct_basin(self):
        """R4 / D9: a body flipped 180 deg must not be mistaken for the naive
        (identity-candidate) alignment — the winner must come from actually
        searching the full 24-candidate enumeration. Surface-sampled source cloud —
        see `_sample_mesh_surface`."""
        verts, faces = _composite_body_mesh()
        surface = _sample_mesh_surface(verts, faces, n_points=6000, seed=1)
        rotation_true = np.diag([-1.0, -1.0, 1.0])  # 180 deg about z, in the 24-group

        p_source = surface @ rotation_true  # scale=1, t=0

        cloud = PointCloud(points=p_source, normals=None, colors=None, source_path=Path("flip.ply"))
        _, alignment = align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())

        rot_err_deg = _geodesic_angle_deg(alignment.rotation, rotation_true)
        assert rot_err_deg < 1.0, f"rotation err {rot_err_deg} deg"
        assert alignment.candidate_index != 0, "must not default to the naive identity candidate"
        assert alignment.converged is True

    def test_alignment_rejects_scale_collapse_under_noise_and_outliers(self):
        """P0 regression (align.py:213-224/284-308 finding): the demonstrated failure
        was a scaled point-to-point ICP candidate collapsing toward scale -> 0 (or
        exploding) while Open3D's own `inlier_rmse` reported it as the *best*
        candidate (fitness=1.0, converged=True) -- because raw inlier RMSE is
        scale-degenerate. A heavily noisy, 20%-outlier cloud is the harshest case: it
        must not recover a scale that is orders of magnitude off ground truth, and it
        must not claim `converged=True` for a scale that departs wildly from the
        PCA-derived initial estimate `align_cloud_to_smpl` itself computes.
        """
        verts, faces = _composite_body_mesh()
        surface = _sample_mesh_surface(verts, faces, n_points=4000, seed=7)

        rng = np.random.default_rng(7)
        noisy = surface + rng.normal(scale=0.010, size=surface.shape)  # 10mm noise
        n_outliers = int(0.2 * noisy.shape[0])
        mins, maxs = surface.min(axis=0), surface.max(axis=0)
        outliers = rng.uniform(mins, maxs, size=(n_outliers, 3))
        combined_smpl_frame = np.vstack([noisy, outliers])

        rotation_true = _rotation_from_axis_angle(np.array([0.2, 0.9, -0.3]), 71.0)
        scale_true = 2.7
        translation_true = np.array([0.3, -0.6, 1.1])
        # p_smpl = scale*(R @ p_source) + t  =>  p_source = R^T @ ((p_smpl - t)/scale)
        p_source = (combined_smpl_frame - translation_true) / scale_true @ rotation_true

        cloud = PointCloud(
            points=p_source, normals=None, colors=None, source_path=Path("noisy.ply")
        )
        _, alignment = align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())

        # The headline P0 failure mode: scale collapsing to ~0 (measured 1.24e-30 in
        # the reported finding) or exploding by an order of magnitude or more.
        assert 0.05 < alignment.scale / scale_true < 20.0, (
            f"scale={alignment.scale} is orders of magnitude off scale_true={scale_true} "
            "-- looks like a scale collapse/explosion regression"
        )

        if alignment.converged:
            ratio = alignment.scale / alignment.scale_init
            assert 1.0 / 3.0 <= ratio <= 3.0, (
                f"converged=True but recovered scale={alignment.scale} departs "
                f"{ratio:.3g}x from the PCA initial estimate "
                f"scale_init={alignment.scale_init} -- converged must not be "
                "reported for a wildly-off alignment"
            )

    def test_alignment_deterministic(self):
        """AC7 / D12: two runs on identical input give the same alignment —
        same winning candidate, and rotation/translation/scale identical to
        float64 machine precision.

        NOTE (measured, see BUILD_RESULT notes): master Section 2 D12 / AC7's
        wording is "bitwise-identical". Under Open3D's default multi-threaded
        build, repeated runs of `registration_icp` on IDENTICAL input differ
        at the ~1e-15 relative level (e.g. scale 1.7000000000000022 vs
        1.7000000000000028) — this is floating-point summation-order
        non-associativity across OpenMP threads inside Open3D's compiled ICP,
        not an RNG: `OMP_NUM_THREADS=1` restores exact bitwise equality
        (verified). `align_cloud_to_smpl` exposes no thread-count knob (that
        is a process-level environment setting, made before Open3D is
        imported), so this test asserts machine-precision equality — the
        strongest bound this module's own code can promise — rather than a
        literal bit-pattern comparison.
        """
        verts, faces = _composite_body_mesh()
        surface = _sample_mesh_surface(verts, faces, n_points=6000, seed=2)
        rotation_true = _rotation_from_axis_angle(np.array([1.0, 0.2, -0.4]), 63.0)
        scale_true = 1.7
        translation_true = np.array([-0.1, 0.2, 0.05])
        p_source = (surface - translation_true) / scale_true @ rotation_true

        cfg = _AlignCfg()

        def _run() -> CloudAlignment:
            cloud = PointCloud(
                points=p_source.copy(), normals=None, colors=None, source_path=Path("d.ply")
            )
            _, alignment = align_cloud_to_smpl(cloud, verts, faces, cfg)
            return alignment

        a = _run()
        b = _run()

        assert a.scale == pytest.approx(b.scale, rel=1e-9, abs=1e-12)
        assert np.allclose(a.rotation, b.rotation, rtol=1e-9, atol=1e-12)
        assert np.allclose(a.translation, b.translation, rtol=1e-9, atol=1e-12)
        assert a.candidate_index == b.candidate_index
        assert a.inlier_rmse_m == pytest.approx(b.inlier_rmse_m, abs=1e-9)
        assert a.fitness == b.fitness

    def test_random_structureless_cloud_does_not_converge(self):
        """P1 regression (align.py:456-474 finding, iteration 1 -> 2): `converged`
        used to be tautological (measured at whatever threshold the polish cascade
        happened to stop on), so it could never actually fire even on maximally
        adversarial input. 20000 uniformly-random points in a 2m box (no body
        structure at all) must not read as a converged alignment under the fixed
        measurement threshold."""
        verts, faces = _composite_body_mesh()
        rng = np.random.default_rng(11)
        random_cloud = rng.uniform(-1.0, 1.0, size=(20_000, 3))

        cloud = PointCloud(
            points=random_cloud, normals=None, colors=None, source_path=Path("random.ply")
        )
        _, alignment = align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())

        assert alignment.converged is False, (
            f"structureless random cloud reported converged=True "
            f"(fitness={alignment.fitness}) -- gate cannot fire even on "
            "maximally adversarial input"
        )

    @pytest.mark.parametrize(
        ("name", "frac_lo", "frac_hi", "seed"),
        [
            ("head_only", 0.88, 1.0, 20),
            ("legs_only", 0.0, 0.5, 21),
            ("feet_only", 0.0, 0.15, 22),
        ],
    )
    def test_partial_body_slice_never_confidently_wrong(self, name, frac_lo, frac_hi, seed):
        """P1 regression (align.py:411/:219 finding, iteration 2): a partial-body
        cloud (only a head, only legs, only feet) must not both (a) crash with a
        bare `numpy.linalg.LinAlgError` from a non-finite ICP candidate transform,
        nor (b) silently report `converged=True` for a similarity whose mapping
        error is wildly wrong (measured in the original finding: feet-only,
        fitness=0.68, converged=True, mean error 1042mm). Either the alignment
        correctly reports non-convergence, or — if it does converge — the mapping
        error must be small."""
        verts, faces = _composite_body_mesh()
        rotation_true = _rotation_from_axis_angle(np.array([0.3, -0.5, 0.8]), 137.0)
        scale_true = 0.371
        translation_true = np.array([0.05, -0.03, 0.02])

        smpl_frame_points, source_points = _height_slice_source(
            verts,
            faces,
            frac_lo,
            frac_hi,
            n_samples=8000,
            seed=seed,
            rotation_true=rotation_true,
            scale_true=scale_true,
            translation_true=translation_true,
        )
        assert smpl_frame_points.shape[0] > 0, "slice produced no points -- fixture bug"

        cloud = PointCloud(
            points=source_points, normals=None, colors=None, source_path=Path(f"{name}.ply")
        )
        # Must not raise a bare LinAlgError (or any exception at all, on this
        # fixture's own scale of degeneracy) -- if it can't align, it must say so
        # via `converged=False`, not crash.
        aligned, alignment = align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())

        mean_err_mm = float(
            np.mean(np.linalg.norm(aligned.points - smpl_frame_points, axis=1)) * 1000.0
        )
        if alignment.converged:
            assert mean_err_mm < 100.0, (
                f"{name}: converged=True but mean mapping error {mean_err_mm:.1f}mm "
                ">= 100mm -- silently confident and wrong"
            )

    def test_skips_single_non_finite_candidate_and_still_aligns(self, monkeypatch):
        """P1 regression (align.py:411 finding, iteration 2): a single candidate's
        ICP result carrying a non-finite (NaN) transformation -- as Open3D's own
        scaled point-to-point ICP is observed to return on some low-extent inputs --
        must be skipped, not fed into `_decompose_similarity`'s `np.linalg.svd`
        (which would crash with a bare, unhelpful `LinAlgError`). The real Open3D
        failure is data/version-specific to reproduce reliably in a unit test, so
        this injects it deterministically via `_pick_nonzero_fitness`, the one
        function every candidate's result passes through before decomposition."""
        verts, faces = _composite_body_mesh()
        surface = _sample_mesh_surface(verts, faces, n_points=3000, seed=12)
        cloud = PointCloud(points=surface, normals=None, colors=None, source_path=Path("x.ply"))

        class _NaNResult:
            transformation = np.full((4, 4), np.nan)
            fitness = 1.0
            inlier_rmse = 0.0

        real_pick = align_module._pick_nonzero_fitness
        calls = {"n": 0}

        def fake_pick(a, b):
            calls["n"] += 1
            if calls["n"] == 5:  # corrupt exactly one of the 24 candidates
                return _NaNResult()
            return real_pick(a, b)

        monkeypatch.setattr(align_module, "_pick_nonzero_fitness", fake_pick)

        _, alignment = align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())

        assert calls["n"] == N_PROPER_ROTATIONS
        assert np.all(np.isfinite(alignment.rotation))
        assert np.isfinite(alignment.scale)
        assert np.all(np.isfinite(alignment.translation))

    def test_raises_clear_error_when_all_candidates_non_finite(self, monkeypatch):
        """P1 regression (align.py:411 finding, iteration 2): if EVERY candidate's
        ICP result is non-finite, `align_cloud_to_smpl` must raise a clear,
        named `ValueError` (mentioning the cloud) rather than propagate a bare
        `numpy.linalg.LinAlgError` from `_decompose_similarity`."""
        verts, faces = _composite_body_mesh()
        surface = _sample_mesh_surface(verts, faces, n_points=3000, seed=13)
        cloud = PointCloud(points=surface, normals=None, colors=None, source_path=Path("x.ply"))

        class _NaNResult:
            transformation = np.full((4, 4), np.nan)
            fitness = 1.0
            inlier_rmse = 0.0

        monkeypatch.setattr(align_module, "_pick_nonzero_fitness", lambda a, b: _NaNResult())

        with pytest.raises(ValueError, match="non-finite"):
            align_cloud_to_smpl(cloud, verts, faces, _AlignCfg())


# ---------------------------------------------------------------------------
# segment.py — semantic part weights from lbs_weights
# ---------------------------------------------------------------------------


class TestSmplPartGroups:
    def test_partitions_all_24_joints(self):
        all_joints = sorted(j for joints in SMPL_PART_GROUPS.values() for j in joints)
        assert all_joints == list(range(SMPL_NUM_JOINTS))

    def test_no_overlap_between_groups(self):
        seen: set[int] = set()
        for joints in SMPL_PART_GROUPS.values():
            for j in joints:
                assert j not in seen, f"joint {j} appears in more than one group"
                seen.add(j)

    def test_group_names_match_tier3_config_keys(self):
        """`Tier3Config` isn't importable from this package (owned by another
        brief) — the expected key set is master Section 5.2's
        `body_part_weights` default, duplicated here as the documented
        contract this module must satisfy (D7)."""
        assert set(SMPL_PART_GROUPS) == _TIER3_BODY_PART_WEIGHT_KEYS


class TestSmplPartLabels:
    def test_one_hot_per_joint(self):
        lbs = np.eye(SMPL_NUM_JOINTS, dtype=np.float64)
        labels = smpl_part_labels(lbs)

        group_order = list(SMPL_PART_GROUPS)
        for joint in range(SMPL_NUM_JOINTS):
            expected_group_id = group_order.index(_find_group(joint))
            assert labels[joint] == expected_group_id

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            smpl_part_labels(np.zeros((10, 23)))

    @requires_smpl
    def test_real_model_covers_all_6890_vertices(self):
        import smplx

        model = smplx.SMPL(model_path="models/smpl", gender="neutral")
        lbs = model.lbs_weights.detach().cpu().numpy()

        labels = smpl_part_labels(lbs)

        assert labels.shape == (6890,)
        assert not np.any(labels == -1), "every vertex must get exactly one label"
        assert set(np.unique(labels)) <= set(range(len(SMPL_PART_GROUPS)))


class TestVertexPartWeights:
    def test_maps_group_weight_per_vertex(self):
        lbs = np.eye(SMPL_NUM_JOINTS, dtype=np.float64)
        weights = {"torso": 1.0, "arms": 0.7, "legs": 0.7, "head": 0.5, "hands": 0.3, "feet": 0.4}

        out = vertex_part_weights(lbs, weights)

        assert out.dtype == np.float32
        assert out.shape == (SMPL_NUM_JOINTS,)
        for joint in range(SMPL_NUM_JOINTS):
            expected = weights[_find_group(joint)]
            assert out[joint] == pytest.approx(expected)

    def test_unknown_group_raises(self):
        lbs = np.eye(SMPL_NUM_JOINTS, dtype=np.float64)
        with pytest.raises(ValueError):
            vertex_part_weights(lbs, {"torso": 1.0, "not_a_group": 0.5})

    def test_missing_group_defaults_to_zero(self):
        lbs = np.eye(SMPL_NUM_JOINTS, dtype=np.float64)
        out = vertex_part_weights(lbs, {"torso": 1.0})
        head_joint = SMPL_PART_GROUPS["head"][0]
        assert out[head_joint] == 0.0


class TestTransferLabelsToCloud:
    def test_nearest_vertex_wins(self):
        mesh_vertices = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64
        )
        vertex_labels = np.array([0, 1, 2])
        cloud_points = np.array(
            [[0.1, 0.1, 0.0], [9.9, 0.2, 0.0], [0.2, 9.8, 0.0]], dtype=np.float64
        )

        labels = transfer_labels_to_cloud(cloud_points, mesh_vertices, vertex_labels)
        assert np.array_equal(labels, [0, 1, 2])

    def test_shape_mismatch_raises(self):
        mesh_vertices = np.zeros((5, 3))
        vertex_labels = np.zeros(4)  # wrong length
        with pytest.raises(ValueError):
            transfer_labels_to_cloud(np.zeros((3, 3)), mesh_vertices, vertex_labels)
