"""Staged Tier 3 SMPL+D surface fitting optimiser (master D5, D6).

Given a Tier 2 ``RefinementResult`` (β, θ, translation, scale, joint-fitted
mesh) and a point cloud already aligned into the SMPL/world frame, this module
runs two stages, in this order, on the SAME ``SMPLModel`` instance:

  S2 ``model_fit``   — optimise (β?, θ, global_orient, translation) against the
                        surface with the displacement field ``D`` held at zero
                        throughout (``apply_displacements=False``).
  S3 ``displacement`` — freeze every SMPL parameter and optimise ``D`` alone.

The staging is the whole point (master §2 D5): it is what keeps ``D``
interpretable as genuine off-manifold geometry (clothing, hair, soft-tissue
detail) rather than a dumping ground for pose/shape/alignment error, which is
what REVIEW.md 7.B5 demands of the PSD residual downstream. Global scale is
never touched here — it is solved once by
:func:`scantosmpl.pointcloud.align.align_cloud_to_smpl` and stays frozen for
both stages (master D6); ``"scale"`` is asserted absent from every stage's
``params`` list at both import time (the shipped default schedule) and at the
start of every :meth:`Tier3SurfaceFitter.fit` call (a caller-supplied
schedule).

Frame discipline: every tensor here lives in the SMPL/world posed frame,
metres — the frame :meth:`scantosmpl.smpl.model.SMPLModel.forward` returns and
the frame a :class:`~scantosmpl.pointcloud.io.PointCloud` carries once
:func:`~scantosmpl.pointcloud.align.align_cloud_to_smpl` has run
(``frame="smpl_world"``, ``units="metres"``). :meth:`Tier3SurfaceFitter.fit`
asserts this before doing anything else — a source-frame cloud is the single
most damaging silent failure in this tier: it converges, it produces a
plausible-looking ``D``, and it is entirely wrong.

This module never imports ``scantosmpl.evaluation.surface_metrics`` — the
7.M-compliant report is a *reporting* concern, deliberately kept separate from
the *training* loss in ``scantosmpl.fitting.surface_losses`` (REVIEW.md 7.M6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import torch
from scipy.spatial import cKDTree

from scantosmpl.fitting.losses import pose_prior_loss, shape_regularisation
from scantosmpl.fitting.optimiser import RefinementResult
from scantosmpl.fitting.surface_losses import (
    build_uniform_laplacian,
    chamfer_loss,
    displacement_regularisation,
    laplacian_smoothing_loss,
    normal_consistency_loss,
)
from scantosmpl.pointcloud.io import PointCloud
from scantosmpl.pointcloud.segment import (
    SMPL_PART_GROUPS,
    smpl_part_labels,
    transfer_labels_to_cloud,
    vertex_part_weights,
)
from scantosmpl.smpl.model import SMPLModel

logger = logging.getLogger(__name__)

#: The six SMPL parameter tensors a `SurfaceStage.params` entry may name.
_PARAM_NAMES: tuple[str, ...] = (
    "betas",
    "body_pose",
    "global_orient",
    "translation",
    "scale",
    "displacements",
)

_CONVERGENCE_TOL = 1e-7
_CONVERGENCE_MIN_ITERS = 10


class SurfaceFitConfigLike(Protocol):
    """The `Tier3Config` fields `Tier3SurfaceFitter` reads (master §5.2).

    Declared structurally, mirroring `pointcloud.align.AlignConfigLike` /
    `pointcloud.preprocess.PreprocessConfigLike` — this module never imports
    `scantosmpl.config` (owned by `tier3-pipeline-artefacts`, and `Tier3Config`
    does not exist there yet). The real `Tier3Config` satisfies this Protocol
    structurally once it lands; a hand-rolled stand-in (as in
    `tests/test_pointcloud.py`) works identically for tests today.
    """

    lock_betas: bool
    chamfer_chunk_size: int
    chamfer_huber_delta_m: float
    chamfer_trim_quantile: float
    body_part_weights: dict[str, float]
    use_semantic_weighting: bool


@dataclass
class SurfaceStage:
    """One optimisation stage — mirrors `fitting.optimiser.OptimisationStage`.

    Attributes:
        name: Stage label, used for `loss_history` keys and log lines.
        params: Subset of `betas|body_pose|global_orient|translation|scale|
            displacements` to optimise this stage. Every other parameter is
            frozen (`requires_grad_(False)`) for the stage's duration.
            `"scale"` must never appear here — master D6.
        n_iterations: Adam steps for this stage.
        w_chamfer: Bidirectional chamfer weight (`surface_losses.chamfer_loss`).
        w_normal: Normal-consistency weight (S3 only in the default schedule).
        w_laplacian: Laplacian-smoothing weight on `D` (S3 only).
        w_displacement_reg: `||D||^2` weight (S3 only).
        w_pose_prior: L2-to-neutral-pose weight (S2 only).
        w_shape_reg: L2-to-mean-shape weight (S2 only).
        learning_rate: Adam learning rate for this stage.
    """

    name: str
    params: list[str]
    n_iterations: int
    w_chamfer: float = 1.0
    w_normal: float = 0.0
    w_laplacian: float = 0.0
    w_displacement_reg: float = 0.0
    w_pose_prior: float = 0.0
    w_shape_reg: float = 0.0
    learning_rate: float = 1e-2


@dataclass
class SurfaceFitResult:
    """Output of `Tier3SurfaceFitter.fit` (master §5.1)."""

    betas: np.ndarray  # (10,)
    body_pose: np.ndarray  # (69,)
    global_orient: np.ndarray  # (3,)
    translation: np.ndarray  # (3,)
    scale: float  # carried through unchanged from Tier 2 (D6)
    displacements: np.ndarray  # (6890, 3) float32, POSED WORLD frame, metres (D4)
    vertices: np.ndarray  # (6890, 3) = base_vertices + displacements
    base_vertices: np.ndarray  # (6890, 3) = SMPL(β,θ,t,s) with D = 0
    betas_locked: bool
    loss_history: dict[str, list[float]]
    metrics: dict[str, float] = field(default_factory=dict)


def _assert_no_scale_in_stages(stages: list[SurfaceStage]) -> None:
    """Master D6: `"scale"` must appear in NO stage's `params`.

    The ICP alignment (`pointcloud.align.align_cloud_to_smpl`) owns metric
    scale; letting S2/S3 also move it would make CLAUDE.md's "SMPL has the
    correct metric scale" premise false and make the two solves redundant.
    """
    offenders = [stage.name for stage in stages if "scale" in stage.params]
    if offenders:
        raise AssertionError(
            f"'scale' appears in stage(s) {offenders} — forbidden by master D6. "
            "Global scale is solved once by pointcloud.align.align_cloud_to_smpl "
            "and must stay frozen through every Tier 3 surface-fitting stage."
        )


#: Master §5.3, verbatim. S2 fits (β?, θ, global_orient, translation) with
#: D == 0 throughout; S3 freezes every SMPL parameter and solves D alone.
DEFAULT_SURFACE_STAGES: list[SurfaceStage] = [
    SurfaceStage(
        name="model_fit",
        params=["betas", "body_pose", "global_orient", "translation"],
        n_iterations=300,
        w_chamfer=1.0,
        w_pose_prior=0.01,
        w_shape_reg=0.01,
        learning_rate=5e-3,
    ),
    SurfaceStage(
        name="displacement",
        params=["displacements"],
        n_iterations=250,
        w_chamfer=1.0,
        w_normal=0.1,
        w_laplacian=0.1,
        w_displacement_reg=0.01,
        learning_rate=1e-3,
    ),
]

_assert_no_scale_in_stages(DEFAULT_SURFACE_STAGES)


class Tier3SurfaceFitter:
    """Runs S2 (`model_fit`) then S3 (`displacement`) on one `SMPLModel`.

    Args:
        smpl_model: The SMPL layer to optimise. Its device is used for every
            tensor built inside `fit`.
        cfg: Tier 3 config — see `SurfaceFitConfigLike` for the fields read.
    """

    def __init__(self, smpl_model: SMPLModel, cfg: SurfaceFitConfigLike) -> None:
        self.smpl = smpl_model
        self.cfg = cfg
        self.device = smpl_model.device

    def fit(
        self,
        tier2: RefinementResult,
        cloud: PointCloud,
        *,
        stages: list[SurfaceStage] | None = None,
        locked_betas: np.ndarray | None = None,
    ) -> SurfaceFitResult:
        """Run S2 then S3 (master D5) starting from a Tier 2 fit.

        Args:
            tier2: Phase 5 `RefinementResult` — the initialisation for β, θ,
                translation and scale. `tier2.vertices` also supplies the mesh
                used to transfer semantic part labels onto `cloud`.
            cloud: Point cloud, MUST already be `frame="smpl_world"`,
                `units="metres"` (i.e. post `align_cloud_to_smpl`).
            stages: Override the default two-stage schedule. Copied
                internally (this call never mutates the caller's list or its
                stages' `params` lists).
            locked_betas: (10,) array, required when `cfg.lock_betas` — the
                shape frozen from a prior reference-pose fit (master D10 /
                7.B1). Carried through to `result.betas` UNCHANGED (bypassing
                any float32 GPU round-trip), so `np.array_equal` holds exactly
                regardless of the input dtype.

        Returns:
            `SurfaceFitResult` with the optimised parameters, `D`, and both
            `vertices` (with `D`) and `base_vertices` (`D = 0`) so the D4
            identity is directly checkable from the result.

        Raises:
            ValueError: `cloud` is not `frame="smpl_world"` / `units="metres"`,
                or `cfg.lock_betas` is set without `locked_betas`.
        """
        if cloud.frame != "smpl_world" or cloud.units != "metres":
            raise ValueError(
                "Tier3SurfaceFitter.fit requires an ALIGNED cloud "
                f"(frame='smpl_world', units='metres'); got frame={cloud.frame!r}, "
                f"units={cloud.units!r}. A source-frame cloud silently converges to a "
                "plausible-looking but entirely wrong fit — run "
                "pointcloud.align.align_cloud_to_smpl on it first."
            )

        stages = [
            replace(s, params=list(s.params))
            for s in (DEFAULT_SURFACE_STAGES if stages is None else stages)
        ]
        _assert_no_scale_in_stages(stages)

        betas_locked = bool(self.cfg.lock_betas)
        if betas_locked:
            if locked_betas is None:
                raise ValueError(
                    "cfg.lock_betas=True requires `locked_betas` (a (10,) array) to be "
                    "supplied — there is nothing to lock beta TO otherwise. "
                    "(CLI: --lock-betas implies --betas-from.)"
                )
            locked_betas = np.asarray(locked_betas, dtype=np.float64).reshape(-1)
            if locked_betas.shape[0] != SMPLModel.NUM_BETAS:
                raise ValueError(
                    f"locked_betas must be ({SMPLModel.NUM_BETAS},), got {locked_betas.shape}"
                )
            # "betas" is removed from EVERY stage's params — not zero-weighted — so
            # AC14's exact np.array_equal holds under Adam's epsilon (brief step 2).
            stages = [replace(s, params=[p for p in s.params if p != "betas"]) for s in stages]

        betas_init = locked_betas if betas_locked else tier2.betas

        self.smpl.set_params(
            betas=torch.as_tensor(betas_init, dtype=torch.float32, device=self.device).reshape(
                1, -1
            ),
            body_pose=torch.as_tensor(
                tier2.body_pose, dtype=torch.float32, device=self.device
            ).reshape(1, -1),
            global_orient=torch.as_tensor(
                tier2.global_orient, dtype=torch.float32, device=self.device
            ).reshape(1, -1),
            translation=torch.as_tensor(
                tier2.translation, dtype=torch.float32, device=self.device
            ).reshape(1, -1),
            scale=torch.tensor([float(tier2.scale)], dtype=torch.float32, device=self.device),
            displacements=torch.zeros(
                1, SMPLModel.NUM_VERTICES, 3, dtype=torch.float32, device=self.device
            ),
        )

        cloud_t = torch.as_tensor(cloud.points, dtype=torch.float32, device=self.device)
        cloud_normals_t = (
            None
            if cloud.normals is None
            else torch.as_tensor(cloud.normals, dtype=torch.float32, device=self.device)
        )
        vertex_weights_t, cloud_weights_t = self._semantic_weights(tier2, cloud)

        faces_np = self.smpl.body_model.faces.astype(np.int64)
        faces_t = torch.as_tensor(faces_np, dtype=torch.long, device=self.device)
        laplacian = build_uniform_laplacian(faces_np, SMPLModel.NUM_VERTICES).to(self.device)

        loss_history: dict[str, list[float]] = {}

        for stage in stages:
            apply_displacements = "displacements" in stage.params
            for name in _PARAM_NAMES:
                getattr(self.smpl, name).requires_grad_(name in stage.params)

            params_to_opt = [getattr(self.smpl, name) for name in stage.params]
            if not params_to_opt:
                logger.warning("Stage %r has an empty params list — skipping", stage.name)
                loss_history[stage.name] = []
                continue

            optimiser = torch.optim.Adam(params_to_opt, lr=stage.learning_rate)
            stage_history: list[float] = []
            prev_loss = float("inf")

            for it in range(stage.n_iterations):
                optimiser.zero_grad()
                output = self.smpl.forward(apply_displacements=apply_displacements)
                verts = output.vertices.squeeze(0)

                loss = verts.new_zeros(())

                if stage.w_chamfer > 0:
                    chamfer, _diag = chamfer_loss(
                        verts,
                        cloud_t,
                        vertex_weights=vertex_weights_t,
                        cloud_weights=cloud_weights_t,
                        chunk_size=self.cfg.chamfer_chunk_size,
                        huber_delta=self.cfg.chamfer_huber_delta_m,
                        trim_quantile=self.cfg.chamfer_trim_quantile,
                    )
                    loss = loss + stage.w_chamfer * chamfer

                if stage.w_pose_prior > 0:
                    loss = loss + stage.w_pose_prior * pose_prior_loss(self.smpl.body_pose)

                if stage.w_shape_reg > 0:
                    loss = loss + stage.w_shape_reg * shape_regularisation(self.smpl.betas)

                if stage.w_normal > 0:
                    if cloud_normals_t is None:
                        raise ValueError(
                            f"Stage {stage.name!r} has w_normal={stage.w_normal} but `cloud` "
                            "carries no normals — estimate normals in preprocess_cloud, or set "
                            "this stage's w_normal=0.0."
                        )
                    loss = loss + stage.w_normal * normal_consistency_loss(
                        verts,
                        faces_t,
                        cloud_t,
                        cloud_normals_t,
                        chunk_size=self.cfg.chamfer_chunk_size,
                    )

                if stage.w_laplacian > 0:
                    loss = loss + stage.w_laplacian * laplacian_smoothing_loss(
                        self.smpl.displacements, laplacian
                    )

                if stage.w_displacement_reg > 0:
                    loss = loss + stage.w_displacement_reg * displacement_regularisation(
                        self.smpl.displacements
                    )

                loss.backward()
                optimiser.step()

                loss_val = float(loss.item())
                stage_history.append(loss_val)
                if abs(prev_loss - loss_val) < _CONVERGENCE_TOL and it > _CONVERGENCE_MIN_ITERS:
                    logger.debug("Stage %r converged at iter %d", stage.name, it)
                    break
                prev_loss = loss_val

            loss_history[stage.name] = stage_history
            logger.info(
                "Stage %r done (%d iters): loss %.6f -> %.6f",
                stage.name,
                len(stage_history),
                stage_history[0] if stage_history else 0.0,
                stage_history[-1] if stage_history else 0.0,
            )

        # Courtesy: leave every parameter trainable again for any caller that
        # reuses `smpl_model` after this call.
        for name in _PARAM_NAMES:
            getattr(self.smpl, name).requires_grad_(True)

        with torch.no_grad():
            base_output = self.smpl.forward(apply_displacements=False)
            full_output = self.smpl.forward(apply_displacements=True)
            params = self.smpl.get_params_dict()

        betas_out = (
            np.asarray(locked_betas, dtype=np.float64)
            if betas_locked
            else params["betas"].squeeze(0).cpu().numpy()
        )

        return SurfaceFitResult(
            betas=betas_out,
            body_pose=params["body_pose"].squeeze(0).cpu().numpy(),
            global_orient=params["global_orient"].squeeze(0).cpu().numpy(),
            translation=params["translation"].squeeze(0).cpu().numpy(),
            scale=float(tier2.scale),
            displacements=params["displacements"].squeeze(0).cpu().numpy().astype(np.float32),
            vertices=full_output.vertices.squeeze(0).cpu().numpy(),
            base_vertices=base_output.vertices.squeeze(0).cpu().numpy(),
            betas_locked=betas_locked,
            loss_history=loss_history,
        )

    def _semantic_weights(
        self, tier2: RefinementResult, cloud: PointCloud
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Master D7: per-vertex/per-cloud-point weights from `lbs_weights`.

        `None, None` is a genuine bypass (`cfg.use_semantic_weighting=False`,
        the AC10 A/B switch) — `chamfer_loss` treats `None` weights as
        uniform, so this is not merely "weights of 1.0" wearing a different
        hat; no weight tensor is built at all.
        """
        if not self.cfg.use_semantic_weighting:
            return None, None

        lbs_weights = self.smpl.body_model.lbs_weights.detach().cpu().numpy()
        vertex_labels = smpl_part_labels(lbs_weights)
        vertex_weights = vertex_part_weights(lbs_weights, self.cfg.body_part_weights)

        cloud_labels = transfer_labels_to_cloud(cloud.points, tier2.vertices, vertex_labels)
        group_lookup = np.array(
            [self.cfg.body_part_weights.get(name, 0.0) for name in SMPL_PART_GROUPS],
            dtype=np.float32,
        )
        cloud_weights = group_lookup[cloud_labels]

        return (
            torch.as_tensor(vertex_weights, dtype=torch.float32, device=self.device),
            torch.as_tensor(cloud_weights, dtype=torch.float32, device=self.device),
        )


# ---------------------------------------------------------------------------
# AC12 (7.5) — self-intersection count, for `test_pose_plausible_no_new_intersections`
# ---------------------------------------------------------------------------

#: Candidate-pair search radius, as a multiple of the largest per-face
#: centroid-to-vertex distance. 2.0x is the exact bound for "these two faces'
#: bounding SPHERES (radius = farthest vertex from centroid) could touch";
#: the extra margin covers the (rare, for a near-equilateral SMPL topology)
#: case where a face's true AABB reaches slightly past that sphere.
_INTERSECTION_CANDIDATE_RADIUS_SCALE = 2.5


def _moller_tri_tri_intersect(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    eps: float = 1e-9,
) -> bool:
    """Möller (1997) "no divisions" triangle-triangle intersection test.

    A direct translation of the classic public-domain reference algorithm
    (Tomas Möller, "A Fast Triangle-Triangle Intersection Test", JGT 1997).
    Coplanar pairs are conservatively reported as non-intersecting (the
    `coplanar_tri_tri` branch of the reference is not implemented) — SMPL
    faces are never exactly coplanar except at a shared edge, which the
    caller already excludes via the shared-vertex filter.

    Args:
        v0, v1, v2: (3,) vertices of the first triangle.
        u0, u1, u2: (3,) vertices of the second triangle.
        eps: Zero-distance tolerance for the plane-side tests.

    Returns:
        True if the (non-degenerate, non-coplanar) triangles share interior
        points.
    """

    def _intervals(
        vv0: float, vv1: float, vv2: float, d0: float, d1: float, d2: float
    ) -> tuple[float, float, float, float, float] | None:
        d0d1 = d0 * d1
        d0d2 = d0 * d2
        if d0d1 > 0.0:
            a, b, c = vv2, (vv0 - vv2) * d2, (vv1 - vv2) * d2
            x0, x1 = d2 - d0, d2 - d1
        elif d0d2 > 0.0:
            a, b, c = vv1, (vv0 - vv1) * d1, (vv2 - vv1) * d1
            x0, x1 = d1 - d0, d1 - d2
        elif d1 * d2 > 0.0 or d0 != 0.0:
            a, b, c = vv0, (vv1 - vv0) * d0, (vv2 - vv0) * d0
            x0, x1 = d0 - d1, d0 - d2
        elif d1 != 0.0:
            a, b, c = vv1, (vv0 - vv1) * d1, (vv2 - vv1) * d1
            x0, x1 = d1 - d0, d1 - d2
        elif d2 != 0.0:
            a, b, c = vv2, (vv0 - vv2) * d2, (vv1 - vv2) * d2
            x0, x1 = d2 - d0, d2 - d1
        else:
            return None  # coplanar — not handled (see docstring)
        return a, b, c, x0, x1

    e1 = v1 - v0
    e2 = v2 - v0
    n1 = np.cross(e1, e2)
    d1_ = -float(np.dot(n1, v0))

    du0 = float(np.dot(n1, u0)) + d1_
    du1 = float(np.dot(n1, u1)) + d1_
    du2 = float(np.dot(n1, u2)) + d1_
    if abs(du0) < eps:
        du0 = 0.0
    if abs(du1) < eps:
        du1 = 0.0
    if abs(du2) < eps:
        du2 = 0.0
    if du0 * du1 > 0.0 and du0 * du2 > 0.0:
        return False

    e1 = u1 - u0
    e2 = u2 - u0
    n2 = np.cross(e1, e2)
    d2_ = -float(np.dot(n2, u0))

    dv0 = float(np.dot(n2, v0)) + d2_
    dv1 = float(np.dot(n2, v1)) + d2_
    dv2 = float(np.dot(n2, v2)) + d2_
    if abs(dv0) < eps:
        dv0 = 0.0
    if abs(dv1) < eps:
        dv1 = 0.0
    if abs(dv2) < eps:
        dv2 = 0.0
    if dv0 * dv1 > 0.0 and dv0 * dv2 > 0.0:
        return False

    d_line = np.cross(n1, n2)
    abs_d = np.abs(d_line)
    index = int(np.argmax(abs_d))
    if abs_d[index] < eps:
        return False  # (near-)parallel planes

    res1 = _intervals(float(v0[index]), float(v1[index]), float(v2[index]), du0, du1, du2)
    if res1 is None:
        return False
    a1, b1, c1, x0_1, x1_1 = res1

    res2 = _intervals(float(u0[index]), float(u1[index]), float(u2[index]), dv0, dv1, dv2)
    if res2 is None:
        return False
    a2, b2, c2, x0_2, x1_2 = res2

    xx = x0_1 * x1_1
    yy = x0_2 * x1_2
    xxyy = xx * yy

    tmp = a1 * xxyy
    isect1 = sorted([tmp + b1 * x1_2 * xx, tmp + c1 * x0_2 * xx])

    tmp = a2 * xxyy
    isect2 = sorted([tmp + b2 * x1_1 * yy, tmp + c2 * x0_1 * yy])

    return not (isect1[1] < isect2[0] or isect2[1] < isect1[0])


def count_self_intersecting_faces(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Count mesh faces involved in at least one non-adjacent self-intersection.

    Broad phase: a KD-tree over face centroids narrows ~190M possible pairs
    (13776 choose 2) down to a candidate set via a bounding-sphere radius,
    then an exact AABB-overlap test on that candidate set — the "AABB pass"
    the brief asks for, made tractable by not enumerating every pair by hand.
    Narrow phase: an exact Möller (1997) triangle-triangle intersection test
    on whatever survives.

    Face pairs sharing a vertex are adjacent by construction (they touch at
    the shared vertex/edge, which is not a self-intersection) and are
    excluded before the narrow phase.

    Args:
        vertices: (V, 3) mesh vertices, any consistent frame/units.
        faces: (F, 3) integer face indices.

    Returns:
        The number of DISTINCT faces that intersect at least one non-adjacent
        face. Runs in well under a second for the SMPL topology (13776 faces).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    fcs = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {verts.shape}")
    if fcs.ndim != 2 or fcs.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {fcs.shape}")

    tri = verts[fcs]  # (F, 3, 3)
    centroids = tri.mean(axis=1)  # (F, 3)
    face_radii = np.linalg.norm(tri - centroids[:, None, :], axis=2).max(axis=1)  # (F,)
    max_radius = float(face_radii.max()) if fcs.shape[0] > 0 else 0.0
    if max_radius <= 0.0:
        return 0

    tree = cKDTree(centroids)
    pairs = tree.query_pairs(
        r=_INTERSECTION_CANDIDATE_RADIUS_SCALE * max_radius, output_type="ndarray"
    )
    if pairs.shape[0] == 0:
        return 0

    fa = fcs[pairs[:, 0]]
    fb = fcs[pairs[:, 1]]
    shares_vertex = (
        (fa[:, 0:1] == fb).any(axis=1)
        | (fa[:, 1:2] == fb).any(axis=1)
        | (fa[:, 2:3] == fb).any(axis=1)
    )
    pairs = pairs[~shares_vertex]
    if pairs.shape[0] == 0:
        return 0

    # Exact AABB overlap filter (the brief's "broad-phase AABB pass").
    mins = tri.min(axis=1)
    maxs = tri.max(axis=1)
    i, j = pairs[:, 0], pairs[:, 1]
    aabb_overlap = np.all((mins[i] <= maxs[j]) & (mins[j] <= maxs[i]), axis=1)
    pairs = pairs[aabb_overlap]

    hit_faces: set[int] = set()
    for a, b in pairs:
        ta, tb = tri[a], tri[b]
        if _moller_tri_tri_intersect(ta[0], ta[1], ta[2], tb[0], tb[1], tb[2]):
            hit_faces.add(int(a))
            hit_faces.add(int(b))
    return len(hit_faces)
