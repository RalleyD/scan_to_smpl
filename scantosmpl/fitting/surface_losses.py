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
"""

import hashlib

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


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, (V, 3) unit length, same frame as vertices.

    Differentiable in `vertices`; `faces` is an index tensor and carries no
    gradient.
    """
    faces_long = faces.to(device=vertices.device, dtype=torch.long)
    v0 = vertices[faces_long[:, 0]]
    v1 = vertices[faces_long[:, 1]]
    v2 = vertices[faces_long[:, 2]]
    # Cross-product magnitude is twice the triangle area, so accumulating the
    # un-normalised face normal weights each face by its area.
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)

    normals = torch.zeros_like(vertices)
    for k in range(3):
        normals = normals.index_add(0, faces_long[:, k], face_normals)
    return normals / normals.norm(dim=1, keepdim=True).clamp(min=_EPS)


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

    Args:
        vertices: (V, 3) or (1, V, 3) SMPL/world posed vertices, metres.
        faces: (F, 3) integer face indices (SMPL template ordering).
        cloud: (N, 3) or (1, N, 3) aligned cloud points, metres.
        cloud_normals: (N, 3) unit-length cloud normals, same frame as `cloud`.
        chunk_size: cloud points per nearest-neighbour chunk.

    Returns:
        Scalar tensor in [0, 1]; 0 when every cloud normal is parallel
        (up to sign) to its nearest vertex normal.
    """
    verts = _as_points(vertices, "vertices")
    pts = _as_points(cloud, "cloud").to(device=verts.device, dtype=verts.dtype)
    nrm = _as_points(cloud_normals, "cloud_normals").to(device=verts.device, dtype=verts.dtype)
    if nrm.shape != pts.shape:
        raise ValueError(
            f"cloud_normals shape {tuple(nrm.shape)} != cloud shape {tuple(pts.shape)}"
        )

    vert_normals = _vertex_normals(verts, faces)  # (V, 3)
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=_EPS)

    total = verts.new_zeros(())
    n_terms = 0
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start : start + chunk_size]
        with torch.no_grad():
            # Correspondence is a discrete choice — no gradient through argmin.
            nearest = torch.cdist(chunk, verts).argmin(dim=1)  # (C,)
        cos = (vert_normals[nearest] * nrm[start : start + chunk.shape[0]]).sum(dim=1)
        total = total + (1.0 - cos.abs()).sum()
        n_terms += chunk.shape[0]

    if n_terms == 0:
        return verts.new_zeros(())
    return total / n_terms


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

    key = (hashlib.sha1(np.ascontiguousarray(faces_arr, dtype=np.int64)).hexdigest(), n_verts)
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
    return (smoothed**2).sum(dim=1).mean()


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
