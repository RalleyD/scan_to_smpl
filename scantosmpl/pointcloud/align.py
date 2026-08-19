"""Align a point cloud TO the SMPL mesh with a recovered 7-DoF similarity.

Direction matters: the SMPL mesh already carries the correct metric scale and
orientation (Tier 2 solved them), so the *cloud* is what moves. The recovered
transform is therefore ``source -> SMPL/world``:

    p_smpl = scale * (rotation @ p_source) + translation

PCA gives an initial triad, but the PCA axes of a body are sign- and (for
near-degenerate eigenvalues) order-ambiguous. Rather than guessing an up-axis,
this module enumerates **all 24 proper rotations** mapping the cloud's triad
onto the mesh's, runs ICP from each and keeps the best (master D9).

No RNG anywhere — no RANSAC-FPFH global registration, no random subsampling
(master D12). Two runs on identical input produce bitwise-identical output.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import open3d as o3d

from scantosmpl.pointcloud.io import PointCloud
from scantosmpl.pointcloud.preprocess import bbox_diagonal

logger = logging.getLogger(__name__)

N_PROPER_ROTATIONS = 24


class AlignConfigLike(Protocol):
    """The `Tier3Config` fields `align_cloud_to_smpl` reads (master §5.2).

    Declared structurally so `scantosmpl.pointcloud` never imports `config.py`
    — `Tier3Config` satisfies this protocol by construction.
    """

    icp_max_iterations: int
    icp_threshold_frac: float
    icp_min_fitness: float


@dataclass
class CloudAlignment:
    """Similarity transform: ``p_smpl = scale * (rotation @ p_source) + translation``.

    Attributes:
        scale: Source units -> metres (metres per source unit).
        rotation: (3, 3) float64 proper rotation (det = +1), source -> SMPL/world.
        translation: (3,) float64 metres, in the SMPL/world frame.
        inlier_rmse_m: Open3D ICP inlier RMSE, metres.
        fitness: Open3D ICP fitness (fraction of source points with a
            correspondence within the threshold), in [0, 1].
        n_candidates: Number of enumerated rotations tried (24).
        candidate_index: Index of the winning candidate, in [0, 24).
        converged: ``fitness >= cfg.icp_min_fitness``.
    """

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inlier_rmse_m: float
    fitness: float
    n_candidates: int
    candidate_index: int
    converged: bool

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 3) source-frame points into the SMPL/world frame (metres)."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {pts.shape}")
        return np.asarray(self.scale * (pts @ self.rotation.T) + self.translation, dtype=np.float64)

    def as_matrix(self) -> np.ndarray:
        """(4, 4) float64 homogeneous form of the same similarity."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m


def pca_triad(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal-axis frame of a point set.

    The returned triad is made **proper** (det = +1) and sign-canonical (each
    axis' largest-magnitude component is positive), so the result is a pure
    function of the input — the 24-rotation enumeration handles the remaining
    orientation ambiguity.

    Args:
        points: (N, 3) positions, N >= 3. Any frame, any units.

    Returns:
        centroid: (3,) mean position, same units as ``points``.
        axes: (3, 3) eigenvector **columns**, ordered by descending eigenvalue.
        extents: (3,) sqrt of the covariance eigenvalues (a standard deviation
            per axis), same units as ``points``.

    Raises:
        ValueError: If fewer than 3 points are supplied.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError(f"Need >=3 points for a PCA triad, got {pts.shape[0]}")

    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axes = eigvecs[:, order]

    # Canonical sign: largest-magnitude component of each axis is positive.
    for j in range(3):
        if axes[np.argmax(np.abs(axes[:, j])), j] < 0:
            axes[:, j] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0

    extents = np.sqrt(np.clip(eigvals, 0.0, None))
    return centroid, axes, extents


def _signed_permutation_matrices() -> list[np.ndarray]:
    """The 24 proper signed-permutation matrices, in a fixed deterministic order."""
    mats: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3), dtype=np.float64)
            for j in range(3):
                m[perm[j], j] = signs[j]
            if np.linalg.det(m) > 0:
                mats.append(m)
    return mats


def _make_proper(axes: np.ndarray) -> np.ndarray:
    """Return an orthonormal triad with det = +1 (flips the third axis if needed)."""
    out = np.array(axes, dtype=np.float64, copy=True)
    if out.shape != (3, 3):
        raise ValueError(f"axes must be (3, 3), got {out.shape}")
    if np.linalg.det(out) < 0:
        out[:, 2] *= -1.0
    return out


def enumerate_proper_rotations(src_axes: np.ndarray, dst_axes: np.ndarray) -> list[np.ndarray]:
    """All 24 proper rotations mapping the src PCA triad onto the dst triad.

    Candidate ``i`` is ``dst @ M_i @ src.T`` where ``M_i`` runs over the 24
    proper signed-permutation matrices (the axis-permutation x sign-flip group,
    i.e. the rotation group of the cube). Enumeration order is fixed: the outer
    loop is ``itertools.permutations(range(3))`` and the inner loop is
    ``itertools.product((+1, -1), repeat=3)`` (master D9).

    Args:
        src_axes: (3, 3) source triad, eigenvectors as columns.
        dst_axes: (3, 3) destination triad, eigenvectors as columns.
            Both are made proper (det = +1) first; an improper input triad
            would otherwise yield 24 *reflections*.

    Returns:
        24 (3, 3) float64 rotation matrices, all orthonormal with det = +1.
    """
    src = _make_proper(src_axes)
    dst = _make_proper(dst_axes)
    return [dst @ m @ src.T for m in _signed_permutation_matrices()]


def _decompose_similarity(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Split a (4, 4) similarity into (scale, proper rotation, translation).

    Uses the SVD of the linear block: for an exact similarity ``A = s R`` the
    singular values are all ``s`` and ``R = U V^T``. The SVD also re-projects a
    numerically drifted block back onto SO(3).
    """
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    u, sv, vt = np.linalg.svd(linear)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    scale = float(np.mean(sv))
    translation = np.asarray(matrix[:3, 3], dtype=np.float64).copy()
    return scale, rot, translation


def _mesh_target(mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> o3d.geometry.PointCloud:
    """SMPL vertices as an ICP target, with face-derived vertex normals.

    Point-to-plane ICP needs normals on the *target*; the mesh topology gives
    exact ones, which is why ``mesh_faces`` is a required argument.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_faces, dtype=np.int32))
    mesh.compute_vertex_normals()

    target = o3d.geometry.PointCloud()
    target.points = mesh.vertices
    target.normals = mesh.vertex_normals
    return target


def _better(
    a: o3d.pipelines.registration.RegistrationResult,
    b: o3d.pipelines.registration.RegistrationResult,
) -> o3d.pipelines.registration.RegistrationResult:
    """Pick the better of two ICP results: lowest inlier RMSE among those with
    a non-zero fitness (Open3D reports rmse = 0 when there are NO
    correspondences at all, which would otherwise win every comparison)."""
    if a.fitness <= 0.0:
        return b
    if b.fitness <= 0.0:
        return a
    return b if b.inlier_rmse < a.inlier_rmse else a


def align_cloud_to_smpl(
    cloud: PointCloud,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    cfg: AlignConfigLike,
) -> tuple[PointCloud, CloudAlignment]:
    """Align a source-frame cloud onto the Tier 2 SMPL mesh.

    Pipeline (master D9): PCA triads -> 24 candidate rotations -> per candidate
    a scaled point-to-point ICP (which is what solves the unknown source-unit
    scale) followed by a point-to-plane polish (rigid, so it preserves that
    scale) -> keep the lowest inlier RMSE.

    Args:
        cloud: Cloud in its source frame, normally after `preprocess_cloud`.
        mesh_vertices: (V, 3) float64 SMPL vertices, SMPL/world frame, metres.
        mesh_faces: (F, 3) int64 SMPL faces — used for target vertex normals.
        cfg: Tier 3 config (see :class:`AlignConfigLike`).

    Returns:
        aligned: The same cloud transformed into the SMPL/world frame, in metres
            (``frame="smpl_world"``, ``units="metres"``). Normals are rotated
            (a uniform scale leaves them unit length); colors pass through.
        alignment: The recovered similarity and its ICP diagnostics.

    Raises:
        ValueError: On an empty/degenerate cloud, a malformed mesh, or a cloud
            that has already been aligned (``frame != "source"``).
    """
    if cloud.frame != "source":
        raise ValueError(
            f"align_cloud_to_smpl expects a source-frame cloud, got frame={cloud.frame!r}"
        )
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    faces = np.asarray(mesh_faces)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"mesh_vertices must be (V, 3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mesh_faces must be (F, 3), got {faces.shape}")

    src_centroid, src_axes, src_extents = pca_triad(cloud.points)
    dst_centroid, dst_axes, dst_extents = pca_triad(verts)

    if src_extents[0] <= 0.0:
        raise ValueError("Cloud has zero extent along its dominant axis; cannot align")
    scale_init = float(dst_extents[0] / src_extents[0])

    mesh_diag = bbox_diagonal(verts)
    threshold = cfg.icp_threshold_frac * mesh_diag
    if threshold <= 0.0:
        raise ValueError("SMPL mesh has zero bounding-box diagonal; cannot align")

    source = cloud.to_open3d()
    target = _mesh_target(verts, faces)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=cfg.icp_max_iterations
    )
    point_to_point = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )
    point_to_plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    rotations = enumerate_proper_rotations(src_axes, dst_axes)
    best_result = None
    best_index = -1

    for index, rot in enumerate(rotations):
        init = np.eye(4, dtype=np.float64)
        init[:3, :3] = scale_init * rot
        init[:3, 3] = dst_centroid - scale_init * (rot @ src_centroid)

        # Stage 1 solves the unknown source->metres scale; stage 2 is a rigid
        # point-to-plane polish, so the scale from stage 1 survives composition.
        res = o3d.pipelines.registration.registration_icp(
            source, target, threshold, init, point_to_point, criteria
        )
        res_plane = o3d.pipelines.registration.registration_icp(
            source, target, threshold, res.transformation, point_to_plane, criteria
        )
        candidate = _better(res, res_plane)

        if best_result is None or _better(best_result, candidate) is candidate:
            best_result = candidate
            best_index = index

    assert best_result is not None  # 24 candidates are always enumerated

    scale, rotation, translation = _decompose_similarity(np.asarray(best_result.transformation))
    alignment = CloudAlignment(
        scale=scale,
        rotation=rotation,
        translation=translation,
        inlier_rmse_m=float(best_result.inlier_rmse),
        fitness=float(best_result.fitness),
        n_candidates=len(rotations),
        candidate_index=int(best_index),
        converged=bool(best_result.fitness >= cfg.icp_min_fitness),
    )

    aligned = PointCloud(
        points=alignment.apply(cloud.points),
        # A similarity rotates normals and leaves them unit length.
        normals=None if cloud.normals is None else cloud.normals @ alignment.rotation.T,
        colors=cloud.colors,
        source_path=cloud.source_path,
        frame="smpl_world",
        units="metres",
    )

    logger.info(
        "Aligned cloud: candidate %d/%d, scale=%.6g, fitness=%.3f, rmse=%.4f m, converged=%s",
        alignment.candidate_index,
        alignment.n_candidates,
        alignment.scale,
        alignment.fitness,
        alignment.inlier_rmse_m,
        alignment.converged,
    )
    return aligned, alignment
