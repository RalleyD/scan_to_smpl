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
from scipy.spatial import cKDTree

from scantosmpl.pointcloud.io import PointCloud
from scantosmpl.pointcloud.preprocess import bbox_diagonal

logger = logging.getLogger(__name__)

N_PROPER_ROTATIONS = 24

#: Final polish: `icp_threshold_frac * mesh_bbox_diagonal` is intentionally loose
#: during the 24-candidate search (candidates start far from the correct pose, so a
#: tight threshold would find no correspondences at all in most of them) but that same
#: looseness (~100+ mm on a real body) saturates `fitness` near 1.0 for every candidate
#: and leaves ~2-3 cm of avoidable translation error on the winner. These fractions of
#: the coarse threshold are applied, in order, as a cascade of RIGID (no re-estimated
#: scale -- master D6, scale is solved once) point-to-plane polishes on the WINNING
#: candidate only, tightening the correspondence radius each step to squeeze out that
#: avoidable error.
#:
#: P1 fix (iteration 2): this cascade's ONLY job is producing the best TRANSFORM. It
#: used to also gate itself on `cfg.icp_min_fitness` (stopping -- i.e. keeping a
#: LOOSER, unpolished transform -- the moment a tighter stage's fitness dipped below
#: the bar), which made `CloudAlignment.converged` tautological: whatever transform the
#: cascade stopped on was, by construction, the one at or above `icp_min_fitness`, so
#: `converged` could never actually distinguish a good alignment from a bad one, and
#: tightening `icp_min_fitness` perversely made the loop stop EARLIER at a LOOSER
#: threshold where `fitness` reads higher. The cascade below no longer looks at
#: `cfg.icp_min_fitness` at all: it always tightens through every fraction, only
#: stopping early if a stage finds literally zero correspondences (a genuinely
#: degenerate radius, not a quality judgement -- see `_pick_nonzero_fitness`). The
#: REPORTED `fitness`/`inlier_rmse_m`/`converged` are computed separately, in one
#: `evaluate_registration` call at the fixed `_FINAL_MEASUREMENT_THRESHOLD_FRAC`,
#: regardless of which cascade stage the winning transform came from -- see that
#: constant's docstring for the calibration.
_POLISH_THRESHOLD_FRACTIONS: tuple[float, ...] = (0.3, 0.1, 0.03)

#: P1 fix (iteration 2): `CloudAlignment.fitness`/`inlier_rmse_m`/`converged` are
#: measured at exactly ONE fixed correspondence radius -- `icp_threshold_frac *
#: mesh_bbox_diagonal * _FINAL_MEASUREMENT_THRESHOLD_FRAC` -- independent of wherever
#: the polish cascade above stopped. 0.12 was chosen by direct measurement on the
#: master §7.3-style fixture (`_composite_body_mesh`, a ~2000-vertex-per-part
#: asymmetric ellipsoid body, `icp_threshold_frac=0.05`): a genuinely well-aligned
#: full-body surface sample reads `fitness ≈ 0.56-0.57` at this radius (comfortably
#: above the default `icp_min_fitness=0.5`, with margin, and essentially independent of
#: source cloud density -- 6k/20k/50k points all land within 0.56-0.58), while
#: height-sliced partial-body clouds (head-only/legs-only/feet-only) read `fitness ≈
#: 0.06-0.12` at the same radius -- a >4x gap. The tighter `_POLISH_THRESHOLD_FRACTIONS`
#: level (0.03, ~2.5mm on this fixture) was tried first and rejected: even the
#: genuinely well-aligned full-body case only reaches `fitness ≈ 0.06` there, because
#: at that radius the measurement is dominated by the mesh's own vertex tessellation
#: spacing and the source cloud's sampling density, not by alignment quality -- it
#: would misreport a perfect fit as unconverged. 0.12 sits just above that floor
#: (comparable to the ~8-10mm vertex spacing the real SMPL mesh's own topology is
#: documented to have -- see `_raycasting_scene`'s docstring) while still being tight
#: enough to separate a real fit from a partial/garbage one.
_FINAL_MEASUREMENT_THRESHOLD_FRAC = 0.12

#: P1 fix (iteration 2): companion bound to `icp_min_fitness` in `converged`'s
#: computation -- catches the failure mode measured on the master §7.3 fixture's
#: "feet-only" partial-cloud probe, where a low-extent cloud's OWN `fitness` (a
#: SOURCE-point-centric measure: "what fraction of MY points found a correspondence")
#: read misleadingly high (0.62-0.68) because every one of its few points sat near
#: *something* on the mesh, while it actually covered under 3% of the mesh's own
#: surface -- exactly the asymmetry `_target_coverage` exists to catch (see its
#: docstring). Measured at `_FINAL_MEASUREMENT_THRESHOLD_FRAC`: a genuine full-body fit
#: covers ~75-80% of the mesh's vertices; head-only/legs-only/feet-only partial slices
#: cover 2-6%. 0.3 sits with wide margin on both sides of that gap.
_MIN_TARGET_COVERAGE = 0.3


class AlignConfigLike(Protocol):
    """The `Tier3Config` fields `align_cloud_to_smpl` reads (master §5.2).

    Declared structurally so `scantosmpl.pointcloud` never imports `config.py`
    — `Tier3Config` satisfies this protocol by construction.
    """

    icp_max_iterations: int
    icp_threshold_frac: float
    icp_min_fitness: float
    #: Candidate-selection guard (P0 fix, iteration 1): a scaled point-to-point
    #: ICP stage (`with_scaling=True`) can walk a wrong-rotation candidate's
    #: scale toward a collapsed/degenerate solution that still reports a
    #: deceptively LOW inlier RMSE and fitness=1.0 (raw inlier RMSE is
    #: scale-degenerate -- shrinking the cloud shrinks the RMSE). A candidate
    #: whose recovered scale departs from the PCA ratio-of-extents estimate
    #: (`scale_init`, itself measured within ~1% of ground truth on the master
    #: §7.3 fixture) by more than this factor is rejected outright, before its
    #: RMSE/coverage score is even allowed to win. The same field also backs
    #: `Tier3Pipeline`'s downstream write-time gate 2 on the identical
    #: quantity (`scantosmpl.fitting.surface_pipeline._check_scale_deviation`)
    #: -- reading it through this Protocol, rather than a locally-duplicated
    #: constant, keeps the two checks from silently drifting apart if this
    #: value is ever overridden.
    max_scale_deviation_factor: float


@dataclass
class CloudAlignment:
    """Similarity transform: ``p_smpl = scale * (rotation @ p_source) + translation``.

    Attributes:
        scale: Source units -> metres (metres per source unit).
        rotation: (3, 3) float64 proper rotation (det = +1), source -> SMPL/world.
        translation: (3,) float64 metres, in the SMPL/world frame.
        inlier_rmse_m: Open3D inlier RMSE, metres, measured at the fixed
            ``_FINAL_MEASUREMENT_THRESHOLD_FRAC`` radius (P1 fix, iteration 2:
            NOT the radius the polish cascade happened to stop at -- see that
            constant's docstring).
        fitness: Open3D fitness (fraction of source points with a
            correspondence within the fixed measurement radius), in [0, 1].
        n_candidates: Number of enumerated rotations tried (24).
        candidate_index: Index of the winning candidate, in [0, 24).
        converged: ``fitness >= cfg.icp_min_fitness`` AND the winning
            transform's target coverage (fraction of the MESH's own vertices
            with a nearby aligned point -- see ``_target_coverage``) clears
            ``_MIN_TARGET_COVERAGE``. The coverage term is a P1 fix (iteration
            2): ``fitness`` alone is source-point-centric and reads
            misleadingly high for a low-extent/partial cloud where every point
            happens to sit near *something* on the mesh.
        scale_init: The PCA ratio-of-extents scale estimate that seeded ICP
            and gated candidate selection — exposed so a downstream consumer
            (e.g. `Tier3Pipeline`'s write-time sanity gates) can compare
            against it without recomputing `pca_triad` a second time.
    """

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inlier_rmse_m: float
    fitness: float
    n_candidates: int
    candidate_index: int
    converged: bool
    scale_init: float

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


def _pick_nonzero_fitness(
    a: o3d.pipelines.registration.RegistrationResult,
    b: o3d.pipelines.registration.RegistrationResult,
) -> o3d.pipelines.registration.RegistrationResult:
    """Prefer ``b`` (the later refinement stage) unless it found NO correspondences at
    all (Open3D reports ``inlier_rmse = 0`` with zero correspondences, which must not
    be read as "perfect"). This does **not** compare RMSE across ``a``/``b`` (that
    comparison is exactly the scale-degenerate mistake this module used to make --
    see ``_MAX_SCALE_DEVIATION_FACTOR``'s docstring) -- it only guards a degenerate
    empty-correspondence edge case."""
    if b.fitness <= 0.0:
        return a
    return b


def _raycasting_scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    """A reusable BVH-accelerated scene for exact, unsigned point-to-**triangle**
    distance queries (master D2) -- built once outside the 24-candidate loop.

    Unlike a nearest-*vertex* distance, this is not limited by the SMPL mesh's own
    ~8-10mm vertex spacing: a point sitting exactly between two vertices, on the
    mesh's own surface, still reads ~0. That floor is precisely what let a
    scale-collapsed candidate look deceptively good under the old vertex-based
    ``inlier_rmse`` criterion.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)),
    )
    return scene


def _cloud_to_mesh_rms(scene: o3d.t.geometry.RaycastingScene, points: np.ndarray) -> float:
    """RMS unsigned point-to-triangle distance (metres) from ``points`` to the mesh
    ``scene`` was built from. Not vertex-tessellation-limited -- see
    :func:`_raycasting_scene`."""
    if points.shape[0] == 0:
        return float("inf")
    query = o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
    distances = scene.compute_distance(query).numpy().astype(np.float64)
    return float(np.sqrt(np.mean(np.square(distances))))


def _target_coverage(
    mesh_vertices: np.ndarray, aligned_points: np.ndarray, threshold: float
) -> float:
    """Fraction of the TARGET mesh's vertices that have an aligned SOURCE point within
    ``threshold`` of them.

    This is deliberately the reverse direction from Open3D's own ``fitness`` (which is
    the fraction of SOURCE points with a correspondence -- a cloud collapsed into a
    tiny region can score ``fitness = 1.0`` there, because every one of its points is
    near *something*). A collapsed source cloud occupies almost none of a large target
    mesh's volume, so it covers almost none of the *target*'s vertices -- this term
    catches the P0 failure mode directly, independent of scale.
    """
    if aligned_points.shape[0] == 0:
        return 0.0
    tree = cKDTree(aligned_points)
    nearest, _ = tree.query(mesh_vertices, k=1)
    return float(np.mean(np.asarray(nearest) <= threshold))


def align_cloud_to_smpl(
    cloud: PointCloud,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    cfg: AlignConfigLike,
) -> tuple[PointCloud, CloudAlignment]:
    """Align a source-frame cloud onto the Tier 2 SMPL mesh.

    Pipeline (master D9): PCA triads -> 24 candidate rotations -> per candidate a
    scaled point-to-point ICP (which is what solves the unknown source-unit scale)
    followed by a point-to-plane polish (rigid, so it preserves that scale) -> keep
    the best-scoring candidate -> a final rigid, shrinking-threshold polish on the
    winner alone.

    Candidate selection is **not** "lowest Open3D inlier RMSE" (that quantity is
    scale-degenerate: shrinking a candidate's recovered scale toward zero shrinks its
    own inlier RMSE too, so a collapsed candidate can and does win a raw-RMSE
    comparison -- see ``cfg.max_scale_deviation_factor``). Instead, each candidate is
    scored by, in priority order: (1) whether its recovered scale stays within
    ``cfg.max_scale_deviation_factor``x of the PCA-derived ``scale_init`` (a collapsed
    or exploded candidate is rejected outright); (2) ``_target_coverage`` -- what
    fraction of the mesh's own vertices land near an aligned source point (a
    collapsed candidate covers almost none of it); (3) ``_cloud_to_mesh_rms`` -- an
    exact point-to-**triangle** RMS (master D2), so the winner is not penalised by the
    SMPL mesh's own vertex tessellation the way a nearest-vertex measurement would be.

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
        ValueError: On an empty/degenerate cloud, a malformed mesh, a cloud
            that has already been aligned (``frame != "source"``), or a cloud
            so low-extent/degenerate that every one of the 24 candidate ICP
            registrations produced a non-finite transform (P1 fix, iteration 2
            -- measured on partial-body slices of the master §7.3 fixture;
            this replaces what used to be a bare ``numpy.linalg.LinAlgError``
            from feeding a NaN transform straight into ``_decompose_similarity``).
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
    scene = _raycasting_scene(verts, faces)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=cfg.icp_max_iterations
    )
    point_to_point = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )
    point_to_plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    rotations = enumerate_proper_rotations(src_axes, dst_axes)
    scale_lower = scale_init / cfg.max_scale_deviation_factor
    scale_upper = scale_init * cfg.max_scale_deviation_factor

    best_key: tuple[bool, float, float] | None = None
    best_result: o3d.pipelines.registration.RegistrationResult | None = None
    best_transform: np.ndarray | None = None
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
        candidate = _pick_nonzero_fitness(res, res_plane)
        candidate_transform = np.asarray(candidate.transformation)

        # P1 fix (iteration 2): a scaled point-to-point ICP stage on a low-extent
        # (e.g. partial-body) source cloud can return an ALL-NaN transformation --
        # measured on head-only/legs-only slices of the master §7.3 fixture, where
        # this is Open3D's own SVD failing to converge internally. Decomposing a
        # non-finite transform below would crash `_decompose_similarity`'s
        # `np.linalg.svd` with a bare `LinAlgError` naming nothing about point
        # clouds or alignment -- skip the candidate entirely instead, before it
        # ever reaches scoring.
        if not np.all(np.isfinite(candidate_transform)):
            logger.warning(
                "align_cloud_to_smpl: candidate %d/%d produced a non-finite ICP "
                "transform (likely SVD non-convergence on a low-extent/degenerate "
                "point set); skipping it.",
                index,
                len(rotations),
            )
            continue

        scale_c, rot_c, trans_c = _decompose_similarity(candidate_transform)
        valid_scale = scale_lower <= scale_c <= scale_upper

        aligned_points = scale_c * (cloud.points @ rot_c.T) + trans_c
        rms = _cloud_to_mesh_rms(scene, aligned_points)
        coverage = _target_coverage(verts, aligned_points, threshold)

        # Lexicographic: pass the scale gate first, then maximise target coverage,
        # then minimise the (tessellation-floor-free) surface RMS.
        key = (valid_scale, coverage, -rms)
        if best_key is None or key > best_key:
            best_key = key
            best_result = candidate
            best_transform = candidate_transform
            best_index = index

    if best_result is None or best_transform is None:
        # P1 fix (iteration 2): every one of the 24 candidates was non-finite --
        # rather than propagating a bare LinAlgError (or an unguarded `assert`
        # failure), name the cloud so the caller has something to act on.
        raise ValueError(
            f"align_cloud_to_smpl: all {len(rotations)} candidate ICP registrations "
            f"produced a non-finite transform for a cloud with {cloud.n_points} points "
            f"and bbox diagonal {bbox_diagonal(cloud.points):.6g} (source units). The "
            "cloud is likely too small, too low-extent, or otherwise degenerate "
            "(e.g. a partial-body scan) for ICP to register against the full SMPL "
            "mesh -- provide a more complete point cloud."
        )
    assert best_key is not None

    if not best_key[0]:
        logger.warning(
            "align_cloud_to_smpl: no candidate's recovered scale fell within "
            "%.1fx of the PCA estimate scale_init=%.6g; using the best-scoring "
            "candidate anyway (index=%d, scale=%.6g). Treat this alignment with "
            "suspicion.",
            cfg.max_scale_deviation_factor,
            scale_init,
            best_index,
            _decompose_similarity(best_transform)[0],
        )

    # Final polish (winner only): the coarse `threshold` used for the 24-candidate
    # search is loose by design (candidates start far from correct, so a tight
    # threshold would find no correspondences in most of them); that same looseness
    # leaves avoidable translation error on the table. Tighten in a shrinking-threshold
    # cascade of RIGID (no re-estimated scale, master D6) polishes.
    #
    # P1 fix (iteration 2): this loop's ONLY job is producing the best TRANSFORM -- it
    # no longer looks at `cfg.icp_min_fitness` (see `_POLISH_THRESHOLD_FRACTIONS`'s
    # docstring for why that used to make `converged` tautological). It always
    # tightens through every fraction; the only early-stop condition is a stage
    # finding literally zero correspondences (a degenerate radius for this transform,
    # not a quality judgement -- mirrors `_pick_nonzero_fitness`), in which case the
    # previous, looser-but-still-valid `polished_transform` is kept.
    polished_transform = best_transform
    for frac in _POLISH_THRESHOLD_FRACTIONS:
        attempt = o3d.pipelines.registration.registration_icp(
            source, target, threshold * frac, polished_transform, point_to_plane, criteria
        )
        attempt_transform = np.asarray(attempt.transformation)
        if attempt.fitness <= 0.0 or not np.all(np.isfinite(attempt_transform)):
            break  # this radius found nothing (or produced garbage) -- stop tightening
        polished_transform = attempt_transform

    # P1 fix (iteration 2): `fitness`/`inlier_rmse_m`/`converged` are measured
    # SEPARATELY from the cascade above, in one `evaluate_registration` call at the
    # fixed `_FINAL_MEASUREMENT_THRESHOLD_FRAC` radius -- regardless of which cascade
    # stage `polished_transform` ended up at. `evaluate_registration` only scores the
    # given transform; it does not move it, so this cannot itself change the winning
    # alignment.
    final_threshold = threshold * _FINAL_MEASUREMENT_THRESHOLD_FRAC
    final_eval = o3d.pipelines.registration.evaluate_registration(
        source, target, final_threshold, polished_transform
    )

    scale, rotation, translation = _decompose_similarity(polished_transform)
    final_aligned_points = scale * (cloud.points @ rotation.T) + translation

    # P1 fix (iteration 2): `fitness` alone is source-point-centric (what fraction of
    # MY points found a correspondence) and reads misleadingly high for a low-extent
    # cloud where every point sits near *something* -- e.g. the measured "feet-only"
    # partial-cloud probe: fitness 0.62-0.68 despite covering <3% of the mesh and a
    # mean per-point mapping error over 1m. `_target_coverage` (mesh-vertex-centric)
    # catches that asymmetry directly -- see `_MIN_TARGET_COVERAGE`'s docstring.
    coverage = _target_coverage(verts, final_aligned_points, final_threshold)

    alignment = CloudAlignment(
        scale=scale,
        rotation=rotation,
        translation=translation,
        inlier_rmse_m=float(final_eval.inlier_rmse),
        fitness=float(final_eval.fitness),
        n_candidates=len(rotations),
        candidate_index=int(best_index),
        converged=bool(
            final_eval.fitness >= cfg.icp_min_fitness and coverage >= _MIN_TARGET_COVERAGE
        ),
        scale_init=scale_init,
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
        "Aligned cloud: candidate %d/%d, scale=%.6g, fitness=%.3f, rmse=%.4f m, "
        "coverage=%.3f, converged=%s",
        alignment.candidate_index,
        alignment.n_candidates,
        alignment.scale,
        alignment.fitness,
        alignment.inlier_rmse_m,
        coverage,
        alignment.converged,
    )
    return aligned, alignment
