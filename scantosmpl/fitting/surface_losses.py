"""Differentiable surface losses for Tier 3 point-cloud refinement.

All tensors here live in the **SMPL/world posed frame, metres** — the frame
`SMPLModel.forward()` returns and the frame `PointCloud` carries once it has been
aligned (`frame='smpl_world'`, `units='metres'`). Nothing in this module converts
units; millimetres appear only at the `ChamferReport` / `Tier3Quality` boundary.

Design notes (master spec §2):
  * **D2** — no Kaolin. The differentiable half of the surface term is a chunked
    `torch.cdist`; the binding *metric* (point-to-surface) lives in
    `scantosmpl/evaluation/surface_metrics.py` and is deliberately a different
    computation.
  * **D3** — the chamfer loss is **bidirectional**, and both directions come out
    of the *same* chunked distance matrix, so the second direction is free. A
    one-sided loss shrink-wraps the mesh into the densest region of the cloud.
  * Robustness — cloud outliers are the norm in photogrammetry, so per-term
    residuals are Huber-bounded and then quantile-trimmed before averaging.
  * **D12** — Tier 3 introduces no stochastic step. Vertex-normal accumulation
    (`_vertex_normals`) therefore uses a cached sparse incidence matrix and
    `torch.sparse.mm` rather than `index_add_`/`scatter_add_`, whose CUDA
    `atomicAdd`-based accumulation order is documented-nondeterministic and
    was measured to differ run-to-run at the 1e-7 (forward) / 1e-8 (gradient)
    level — small per-call, but amplified to millimetres of displacement over
    an Adam schedule.
  * Normal-consistency trust-region weighting — each cloud point's `1 - |cos|`
    term is weighted by its own (detached) distance to its nearest mesh
    vertex, relative to `_NORMAL_TRUST_RADIUS_FRACTION` (1%) of the mesh's own
    local edge length, fading linearly from full weight at distance 0 to zero
    at the radius. CORRECTED (Review iteration 1, altitude finding): at 1% of
    edge length this radius is far smaller than a Voronoi-cell half-width
    (~50% of edge length), so it is **not** a scoped filter that gates out
    only the narrow ambiguous band near the boundary between two vertices'
    correspondence regions — a point a mere 0.7% of one edge length from its
    nearest vertex (well inside that vertex's own neighbourhood, nowhere near
    a boundary) already carries weight ~0.29, i.e. has already lost 71% of
    its contribution. In practice this behaves much closer to a broad,
    largely uniform attenuation of the whole term's gradient influence than a
    surgical fix for the specific correspondence-flip effect (the discrete
    nearest-vertex assignment recomputed every iteration flipping between two
    near-equidistant vertices from one Adam step to the next) it was
    originally designed to target. It is nonetheless a genuine, measured
    improvement over the unweighted term (self-intersecting faces after the
    shipped 250-iter
    `w_normal=0.1` displacement stage on the AC12 regression scenario in
    `tests/test_surface_fitting.py`: 117 -> 5893 unweighted vs. 117 -> ~1100
    with this weighting, and the stage loss now *decreases* over the run
    instead of rising monotonically) — but **it does not fully discharge
    AC12's +5-face bound**. Sweeping the trust-radius fraction down to the
    point of near-total gating still floors at roughly the same ~1000-face
    figure, which means part of the divergence is not a correspondence-flip
    effect at all: independent, confidently-assigned per-vertex normal pulls
    with no smoothness coupling between them are enough on their own to
    wrinkle the mesh under a quarter-thousand Adam steps. Reported upstream
    (see this component's BUILD_RESULT notes) rather than tuned further
    in-module, since a full fix likely needs a master §5.3 value change
    (`DEFAULT_SURFACE_STAGES` is owned by `scantosmpl/fitting/surface.py`,
    outside this module's boundary).
  * Review iteration 2 — a genuinely different mechanism (per-FACE
    correspondence: each cloud point assigned to its nearest face centroid,
    each matched face's target normal the mean of its assigned points'
    normals, loss weighted by that face's own detached area) was implemented
    and MEASURED against the literal shipped `DEFAULT_SURFACE_STAGES`
    (`w_normal=0.1`, 250 iterations) on the exact scenario this finding
    reports (master synthetic fixture + a realistic perturbed Tier-2 input,
    literal `Tier3Config()`): 149 -> 5348 self-intersecting faces — WORSE
    than this (iteration 1) trust-region weighting's 149 -> 345, not better.
    A smaller-cloud variant (`target_points=6000`) and a uniform-weight
    (non-area) variant were also measured, at 149 -> 1913 and 149 -> 3393
    respectively — both likewise worse than this module's shipped
    trust-region weighting. Reverted; not shipped. Full numbers and the
    diagnosed likely cause (area-weighting a face's contribution interacts
    badly with mesh degeneracy: as chamfer-driven buckling shrinks a face's
    area, its own corrective normal-consistency weight *shrinks with it*,
    weakening exactly the correction that would resist further buckling —
    a self-reinforcing loop candidate (b) in the iteration-2 review finding
    did not anticipate) are in this component's BUILD_RESULT notes.
"""

import hashlib
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "chamfer_loss",
    "normal_consistency_loss",
    "build_uniform_laplacian",
    "laplacian_smoothing_loss",
    "displacement_regularisation",
]

_EPS = 1e-12

# Cache of built Laplacians — SMPL topology is fixed, so this is built once per
# (face array, vertex count). Keyed by a hash of the face bytes.
_LAPLACIAN_CACHE: dict[tuple[str, int], torch.Tensor] = {}

# Cache of vertex<->face incidence matrices used by `_vertex_normals`. Keyed
# by (face-hash, n_verts, device, dtype) — built once per topology/device/
# dtype combination and reused (SMPL topology never changes within a run).
_INCIDENCE_CACHE: dict[tuple[str, int, str, str], torch.Tensor] = {}

# Fraction of the mesh's own (current) median incident-edge length used as
# the normal-consistency trust-region radius: a cloud point closer to its
# nearest vertex than this fraction gets full weight; beyond it, weight fades
# linearly to zero at the full radius. Tied to the mesh's own resolution
# (rather than a fixed metric constant) because SMPL's local vertex spacing
# — and hence the distance at which a nearest-vertex assignment becomes
# ambiguous between neighbours — varies severalfold across the body (dense at
# hands/face, coarse at the torso).
_NORMAL_TRUST_RADIUS_FRACTION = 0.01


def _as_points(x: torch.Tensor, name: str) -> torch.Tensor:
    """Accept (P, 3) or (1, P, 3) and return (P, 3). Batches > 1 are rejected."""
    if x.dim() == 3:
        if x.shape[0] != 1:
            raise ValueError(f"{name} must be (P, 3) or (1, P, 3); got {tuple(x.shape)}")
        x = x.squeeze(0)
    if x.dim() != 2 or x.shape[-1] != 3:
        raise ValueError(f"{name} must be (P, 3) or (1, P, 3); got {tuple(x.shape)}")
    return x


def _robust_mean(
    residuals: torch.Tensor,
    weights: torch.Tensor | None,
    huber_delta: float,
    trim_quantile: float,
) -> torch.Tensor:
    """Huber-bound, quantile-trim and weight-average a vector of residuals.

    Args:
        residuals: (P,) non-negative distances, metres.
        weights: (P,) non-negative per-term weights, or None for uniform.
        huber_delta: Huber transition point, metres. Residuals below it stay
            quadratic; above it they contribute linearly.
        trim_quantile: keep residuals at or below this quantile of the
            (detached) residual distribution. >= 1.0 disables trimming — the
            right-skewed tail of a photogrammetry cloud otherwise steers the
            gradient (the Tier 2 lesson from the Phase 5 plan, W1).

    Returns:
        Scalar tensor: sum(w * huber(r)) / sum(w) over the retained terms.
    """
    huber = F.huber_loss(
        residuals, torch.zeros_like(residuals), delta=huber_delta, reduction="none"
    )

    if weights is None:
        w = torch.ones_like(residuals)
    else:
        w = weights.to(device=residuals.device, dtype=residuals.dtype)
        if w.shape != residuals.shape:
            raise ValueError(
                f"weights shape {tuple(w.shape)} != residual shape {tuple(residuals.shape)}"
            )

    if trim_quantile < 1.0 and residuals.numel() > 1:
        cutoff = torch.quantile(residuals.detach().float(), trim_quantile)
        keep = residuals.detach() <= cutoff
        if not bool(keep.any()):
            keep = torch.ones_like(residuals, dtype=torch.bool)
        w = w * keep.to(w.dtype)

    denom = w.sum()
    if float(denom) <= _EPS:
        return residuals.sum() * 0.0
    return (w * huber).sum() / denom


def chamfer_loss(
    vertices: torch.Tensor,
    cloud: torch.Tensor,
    *,
    vertex_weights: torch.Tensor | None = None,
    cloud_weights: torch.Tensor | None = None,
    chunk_size: int = 10_000,
    huber_delta: float = 0.02,
    trim_quantile: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional chamfer between mesh vertices and a point cloud (master D3).

    Both directions come out of **one** chunked `torch.cdist`: `min(dim=1)` is
    mesh→cloud (running minimum across chunks) and `min(dim=0)` is cloud→mesh
    (final within each chunk). The second direction therefore costs nothing
    extra, and omitting it would let the mesh shrink-wrap into the densest part
    of the cloud while uncovered regions drift unpenalised.

    Args:
        vertices: (V, 3) or (1, V, 3) SMPL/world posed vertices, metres.
            Gradients flow through this argument.
        cloud: (N, 3) or (1, N, 3) cloud points, **already aligned** to the
            SMPL/world frame, metres.
        vertex_weights: (V,) per-vertex semantic weights (master D7), or None.
        cloud_weights: (N,) per-point semantic weights transferred from the
            nearest mesh vertex, or None.
        chunk_size: cloud points per distance-matrix chunk. Peak memory scales
            as V x chunk_size.
        huber_delta: Huber transition, metres.
        trim_quantile: per-direction residual quantile to keep, in (0, 1].

    Returns:
        (loss, diagnostics) where `loss` is the mean of the two directional
        terms (so its magnitude is comparable to a one-sided chamfer), and
        `diagnostics` holds detached, **untrimmed, unweighted** mean distances
        in metres under keys `mesh_to_cloud_m` and `cloud_to_mesh_m` — directly
        comparable with the reported surface metric (master R3).
    """
    verts = _as_points(vertices, "vertices")
    pts = _as_points(cloud, "cloud").to(device=verts.device, dtype=verts.dtype)

    if verts.shape[0] == 0 or pts.shape[0] == 0:
        raise ValueError("chamfer_loss requires non-empty vertices and cloud")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    n_points = pts.shape[0]
    mesh_to_cloud: torch.Tensor | None = None  # (V,) running minimum
    cloud_to_mesh_chunks: list[torch.Tensor] = []  # each (chunk,)

    for start in range(0, n_points, chunk_size):
        chunk = pts[start : start + chunk_size]  # (C, 3)
        dist = torch.cdist(verts, chunk)  # (V, C) — the ONE matrix
        chunk_min_per_vertex = dist.min(dim=1).values  # (V,)  mesh -> cloud
        cloud_to_mesh_chunks.append(dist.min(dim=0).values)  # (C,)  cloud -> mesh
        mesh_to_cloud = (
            chunk_min_per_vertex
            if mesh_to_cloud is None
            else torch.minimum(mesh_to_cloud, chunk_min_per_vertex)
        )

    assert mesh_to_cloud is not None  # non-empty cloud guaranteed above
    cloud_to_mesh = torch.cat(cloud_to_mesh_chunks)  # (N,)

    loss_m2c = _robust_mean(mesh_to_cloud, vertex_weights, huber_delta, trim_quantile)
    loss_c2m = _robust_mean(cloud_to_mesh, cloud_weights, huber_delta, trim_quantile)
    loss = 0.5 * (loss_m2c + loss_c2m)

    diagnostics = {
        "mesh_to_cloud_m": float(mesh_to_cloud.detach().mean().item()),
        "cloud_to_mesh_m": float(cloud_to_mesh.detach().mean().item()),
    }
    return loss, diagnostics


def _face_vertex_incidence(
    faces_long: torch.Tensor, n_verts: int, dtype: torch.dtype
) -> torch.Tensor:
    """Sparse (V, F) incidence matrix; `M[v, f] = 1` iff face `f` touches vertex `v`.

    Cached per (face topology, vertex count, device, dtype) — SMPL topology is
    fixed within a run. `M @ face_normals` sums, for every vertex, the normals
    of the faces around it — the same quantity `index_add_` over the three
    per-face corners produced, but via a deterministic `torch.sparse.mm`
    rather than an `atomicAdd`-based scatter (master D12; see module
    docstring).
    """
    device = faces_long.device
    key = (
        hashlib.sha1(
            np.ascontiguousarray(faces_long.detach().cpu().numpy(), dtype=np.int64).tobytes()
        ).hexdigest(),
        n_verts,
        str(device),
        str(dtype),
    )
    cached = _INCIDENCE_CACHE.get(key)
    if cached is not None:
        return cached

    n_faces = faces_long.shape[0]
    face_ids = torch.arange(n_faces, device=device).repeat(3)
    vert_ids = torch.cat([faces_long[:, 0], faces_long[:, 1], faces_long[:, 2]])
    indices = torch.stack([vert_ids, face_ids])
    values = torch.ones(vert_ids.shape[0], device=device, dtype=dtype)
    incidence = torch.sparse_coo_tensor(
        indices, values, size=(n_verts, n_faces), dtype=dtype, device=device
    ).coalesce()

    _INCIDENCE_CACHE[key] = incidence
    return incidence


def _face_corners(
    vertices: torch.Tensor, faces_long: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather each face's three corner vertices: `(v0, v1, v2)`, each `(F, 3)`.

    Shared by `_vertex_normals` and `_median_incident_edge_length`, which
    otherwise each re-derive the identical per-face gather from `vertices`
    and `faces_long`.
    """
    return vertices[faces_long[:, 0]], vertices[faces_long[:, 1]], vertices[faces_long[:, 2]]


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, (V, 3) unit length, same frame as vertices.

    Differentiable in `vertices`; `faces` is an index tensor and carries no
    gradient.
    """
    faces_long = faces.to(device=vertices.device, dtype=torch.long)
    v0, v1, v2 = _face_corners(vertices, faces_long)
    # Cross-product magnitude is twice the triangle area, so accumulating the
    # un-normalised face normal weights each face by its area.
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)

    incidence = _face_vertex_incidence(faces_long, vertices.shape[0], vertices.dtype)
    normals = torch.sparse.mm(incidence, face_normals)
    # `.norm(...)` is untyped (Any) in the torch stubs; the result is always a Tensor.
    return cast(torch.Tensor, normals / normals.norm(dim=1, keepdim=True).clamp(min=_EPS))


def _median_incident_edge_length(vertices: torch.Tensor, faces_long: torch.Tensor) -> torch.Tensor:
    """Scalar (detached) median triangle-edge length of the current mesh.

    Cheap (`O(F)`, shares `_face_corners`'s gather with `_vertex_normals`) and
    recomputed per call rather than cached, since it must track the *current*
    vertex positions — though in practice Tier 3's displacements are
    metres-small perturbations of a fixed-topology mesh, so this barely moves
    across an optimisation run.
    """
    with torch.no_grad():
        v0, v1, v2 = _face_corners(vertices, faces_long)
        edges = torch.cat([v1 - v0, v2 - v1, v0 - v2], dim=0)
        return cast(torch.Tensor, edges.norm(dim=1).median())


def normal_consistency_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    cloud: torch.Tensor,
    cloud_normals: torch.Tensor,
    *,
    chunk_size: int = 10_000,
) -> torch.Tensor:
    """`1 - |cos|` between each cloud normal and its nearest mesh vertex normal.

    The **absolute value is deliberate**: photogrammetry normal orientation is
    unreliable (a Meshroom cloud routinely carries inward-facing normals on part
    of the surface), so a signed term would fight the fit rather than regularise
    it. Only the surface *orientation*, not its sign, is constrained.

    Each point's term is additionally weighted by its own (detached) distance
    to its nearest mesh vertex, relative to a **trust radius** of
    `_NORMAL_TRUST_RADIUS_FRACTION` times the mesh's current median edge
    length: full weight inside the radius, fading linearly to zero at it. At
    the shipped 1% fraction this radius is much smaller than a Voronoi-cell
    half-width, so — despite the name — this behaves more like a broad
    attenuation of the whole term than a filter scoped to the narrow
    ambiguous band where nearest-vertex correspondence flips between two
    near-equidistant vertices from one Adam step to the next (see module
    docstring for the measured numbers). It substantially improves — but, at
    the shipped `w_normal=0.1` / 250-iteration schedule, does not fully
    eliminate — a divergence under sustained Adam optimisation; see this
    component's BUILD_RESULT notes for why a full fix is judged to be out of
    this module's boundary.

    Args:
        vertices: (V, 3) or (1, V, 3) SMPL/world posed vertices, metres.
        faces: (F, 3) integer face indices (SMPL template ordering).
        cloud: (N, 3) or (1, N, 3) aligned cloud points, metres.
        cloud_normals: (N, 3) unit-length cloud normals, same frame as `cloud`.
        chunk_size: cloud points per nearest-neighbour chunk.

    Returns:
        Scalar tensor in [0, 1]; 0 when every cloud normal is parallel
        (up to sign) to its nearest vertex normal (or every weight is zero).
    """
    verts = _as_points(vertices, "vertices")
    pts = _as_points(cloud, "cloud").to(device=verts.device, dtype=verts.dtype)
    nrm = _as_points(cloud_normals, "cloud_normals").to(device=verts.device, dtype=verts.dtype)
    if nrm.shape != pts.shape:
        raise ValueError(
            f"cloud_normals shape {tuple(nrm.shape)} != cloud shape {tuple(pts.shape)}"
        )

    faces_long = faces.to(device=verts.device, dtype=torch.long)
    vert_normals = _vertex_normals(verts, faces_long)  # (V, 3); already the right dtype/device
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=_EPS)

    with torch.no_grad():
        trust_radius = _NORMAL_TRUST_RADIUS_FRACTION * _median_incident_edge_length(
            verts, faces_long
        )
        trust_radius = trust_radius.clamp(min=_EPS)

    weighted_total = verts.new_zeros(())
    weight_total = verts.new_zeros(())
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start : start + chunk_size]
        with torch.no_grad():
            # Correspondence (and the residual used to weight it) is a
            # discrete choice — no gradient flows through either.
            dist_to_mesh = torch.cdist(chunk, verts)
            nearest_dist, nearest = dist_to_mesh.min(dim=1)  # (C,), (C,)
            weight = (1.0 - nearest_dist / trust_radius).clamp(min=0.0, max=1.0)
        cos = (vert_normals[nearest] * nrm[start : start + chunk.shape[0]]).sum(dim=1)
        weighted_total = weighted_total + (weight * (1.0 - cos.abs())).sum()
        weight_total = weight_total + weight.sum()

    if float(weight_total) <= _EPS:
        return verts.new_zeros(())
    return weighted_total / weight_total


def build_uniform_laplacian(faces: np.ndarray, n_verts: int) -> torch.Tensor:
    """Sparse (V, V) uniform (graph) Laplacian `L = D - A` for a fixed topology.

    Symmetric with zero row sums by construction, so a constant (pure
    translation) displacement field is in its null space. Built once and cached
    per (face array, vertex count) — SMPL topology never changes within a run.

    Args:
        faces: (F, 3) integer face indices.
        n_verts: number of vertices V.

    Returns:
        A coalesced sparse COO float32 tensor on the CPU. Move it to the working
        device once (`.to(device)`) and reuse it across iterations.
    """
    faces_arr = np.asarray(faces)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3); got {faces_arr.shape}")

    key = (
        hashlib.sha1(np.ascontiguousarray(faces_arr, dtype=np.int64).tobytes()).hexdigest(),
        n_verts,
    )
    cached = _LAPLACIAN_CACHE.get(key)
    if cached is not None:
        return cached

    f = faces_arr.astype(np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    # Undirected: keep both orientations, then deduplicate.
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    rows = edges[:, 0]
    cols = edges[:, 1]
    degree = np.bincount(rows, minlength=n_verts).astype(np.float32)

    indices = np.concatenate(
        [np.stack([rows, cols]), np.stack([np.arange(n_verts), np.arange(n_verts)])], axis=1
    )
    values = np.concatenate([-np.ones(rows.shape[0], dtype=np.float32), degree])

    laplacian = torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        size=(n_verts, n_verts),
        dtype=torch.float32,
    ).coalesce()

    _LAPLACIAN_CACHE[key] = laplacian
    return laplacian


def laplacian_smoothing_loss(displacements: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Mean squared magnitude of `L @ D` — the *roughness* of the field, not its size.

    Penalising `L @ D` rather than `D` lets a smooth soft-tissue bulge survive
    while a per-vertex spike is suppressed. Magnitude is the job of
    `displacement_regularisation`.

    Args:
        displacements: (V, 3) or (1, V, 3) displacement field, posed world
            metres.
        laplacian: sparse (V, V) Laplacian from `build_uniform_laplacian`. Moved
            to the displacement's device if needed (a no-op when it already
            lives there).

    Returns:
        Scalar tensor: mean over vertices of ||(L @ D)_i||^2.
    """
    disp = _as_points(displacements, "displacements")
    lap = laplacian.to(device=disp.device, dtype=disp.dtype)
    if lap.shape[0] != disp.shape[0]:
        raise ValueError(
            f"laplacian is ({lap.shape[0]}, {lap.shape[1]}) but D has {disp.shape[0]} vertices"
        )
    smoothed = torch.sparse.mm(lap, disp)  # (V, 3)
    # `.mean()` on a 0-d reduction is untyped (Any) in the torch stubs.
    return cast(torch.Tensor, (smoothed**2).sum(dim=1).mean())


def displacement_regularisation(displacements: torch.Tensor) -> torch.Tensor:
    """Mean squared ||D|| over vertices — keeps `D` minimal (master R2).

    A large `D` can silently absorb pose/shape error, which would corrupt the
    PSD residual downstream, so magnitude is penalised independently of
    smoothness.

    Args:
        displacements: (V, 3) or (1, V, 3), posed world metres.

    Returns:
        Scalar tensor.
    """
    disp = _as_points(displacements, "displacements")
    return (disp**2).sum(dim=1).mean()
