"""Tier 3 surface metrics — the *binding* cloud/mesh quality report (REVIEW.md 7.M1-7.M6).

This module is **reporting only**. Nothing here is differentiable and nothing here is used
inside the optimiser — the loss lives in ``scantosmpl/fitting/surface_losses.py`` and 7.M6
explicitly permits the loss and the metric to differ.

What 7.M binds us to:

* **7.M1** — the cloud -> mesh direction is point-to-**surface**: the distance from each cloud
  point to the nearest point on the nearest *triangle*, not to the nearest vertex. On the SMPL
  template (~10 mm triangles) a vertex-based measurement carries an irreducible offset of up to
  ~5.8 mm, which would consume most of the 8 mm Tier 3 budget before the optimiser does anything.
  Implemented with :class:`open3d.t.geometry.RaycastingScene`, which is an exact point-to-triangle
  query (BVH accelerated), **not** an approximation.
* **7.M2** — the mesh -> cloud direction may use vertex-to-nearest-point; a dense photogrammetry
  cloud makes that direction effectively floor-free.
* **7.M3** — both directions are reported separately and are **never** fused into one number.
  :class:`ChamferReport` has no combined field, so a fused number is unrepresentable.
* **7.M4** — aggregation (mean / median / rms / p95 / max) and units (millimetres) are explicit.
* **7.M5** — the tessellation floor is measured and reported beside the result so the headline
  number is interpretable.

Coordinate frames + units (repo spec Section "Coordinate frames + units"):

* All *inputs* are ``float64`` arrays in **metres**, in a single shared frame (SMPL/world after
  Tier 3 alignment). This module applies **no** transform — it assumes cloud and mesh are already
  in the same frame, which ``Tier3SurfaceFitter`` guarantees via ``PointCloud.frame``.
* ``point_to_surface_distances`` / ``vertex_to_point_distances`` return ``float64`` **metres**.
* The single conversion to **millimetres** happens at the :class:`ChamferReport` boundary, and
  every millimetre-valued key ends in ``_mm``.

Open3D interop: ``RaycastingScene`` requires ``float32`` vertices and ``uint32`` triangle
indices. That conversion is confined to the call boundary below — ``float32``/``uint32`` never
leak into this module's public signatures.

Determinism: the only stochastic step is the surface sampling in :func:`tessellation_floor`,
which is seeded with ``numpy.random.default_rng(seed)`` (master D12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

__all__ = [
    "ChamferReport",
    "chamfer_report",
    "point_to_surface_distances",
    "tessellation_floor",
    "vertex_to_point_distances",
]

# Master Section 5.2 defaults, duplicated here so this module is usable without `Tier3Config`
# (which is owned by the `tier3-pipeline-artefacts` brief). Keep these in sync with it.
DEFAULT_TESSELLATION_FLOOR_SAMPLES = 100_000
DEFAULT_TESSELLATION_FLOOR_SEED = 0

_M_TO_MM = 1000.0

#: Aggregations reported per direction (7.M4).
STAT_KEYS: tuple[str, ...] = ("mean", "median", "rms", "p95", "max")

#: Method strings recorded in the artefact metadata (7.M1 "record which was used").
CLOUD_TO_MESH_METHOD = "point_to_surface_open3d_raycasting"
MESH_TO_CLOUD_METHOD = "vertex_to_nearest_point"


@runtime_checkable
class Tier3MetricConfig(Protocol):
    """The slice of ``Tier3Config`` (master Section 5.2) this module reads.

    ``scantosmpl.config.Tier3Config`` satisfies this structurally, so
    ``chamfer_report(cloud, verts, faces, cfg)`` works unchanged once that brief lands.
    Declared as a Protocol rather than imported so this module does not depend on a
    dataclass owned by another component.
    """

    tessellation_floor_samples: int
    tessellation_floor_seed: int


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _as_points(array: np.ndarray, name: str, *, min_rows: int = 1) -> np.ndarray:
    """Validate an ``(N, 3)`` point/vertex array and return it as contiguous float64."""
    points = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if points.shape[0] < min_rows:
        raise ValueError(f"{name} must contain at least {min_rows} point(s), got {points.shape[0]}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


def _as_faces(array: np.ndarray, n_vertices: int, name: str = "faces") -> np.ndarray:
    """Validate an ``(F, 3)`` triangle index array and return it as contiguous int64."""
    faces = np.ascontiguousarray(np.asarray(array))
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array, got dtype {faces.dtype}")
    faces = faces.astype(np.int64, copy=False)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{name} must have shape (F, 3), got {faces.shape}")
    if faces.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one triangle")
    if faces.min() < 0 or faces.max() >= n_vertices:
        raise ValueError(
            f"{name} indices out of range for {n_vertices} vertices "
            f"(min={faces.min()}, max={faces.max()})"
        )
    return faces


# ---------------------------------------------------------------------------
# 7.M1 — cloud -> mesh, point-to-SURFACE
# ---------------------------------------------------------------------------


def point_to_surface_distances(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Unsigned point-to-triangle distances from a point cloud to a mesh surface (7.M1).

    This is the **binding** cloud -> mesh measurement. It is point-to-surface, not
    point-to-vertex: a query point 5 mm above the centroid of a 1 m triangle returns
    0.005 m even though its nearest vertex is ~0.577 m away.

    Distances are **unsigned** by design. A signed (inside/outside) distance is meaningless
    for an open photogrammetry cloud, which has holes and never encloses the body.

    Args:
        points: ``(N, 3)`` float64 query points, metres, SMPL/world frame.
        vertices: ``(V, 3)`` float64 mesh vertices, metres, **same frame as** ``points``.
        faces: ``(F, 3)`` int64 triangle vertex indices.

    Returns:
        ``(N,)`` float64 unsigned distances in **metres**, one per query point.

    Raises:
        ValueError: on malformed shapes/dtypes, empty input, non-finite values, or
            out-of-range face indices.

    Note:
        Non-differentiable — reporting only. Open3D's tensor API is fed ``float32``
        vertices and ``uint32`` triangles internally; that never reaches the caller.
    """
    query = _as_points(points, "points")
    verts = _as_points(vertices, "vertices", min_rows=3)
    tris = _as_faces(faces, verts.shape[0])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(verts.astype(np.float32)),
        o3d.core.Tensor(tris.astype(np.uint32)),
    )
    distances = scene.compute_distance(o3d.core.Tensor(query.astype(np.float32))).numpy()
    return np.asarray(distances, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# 7.M2 — mesh -> cloud, vertex-to-nearest-point
# ---------------------------------------------------------------------------


def vertex_to_point_distances(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance from each mesh vertex to its nearest cloud point (7.M2).

    Vertex-based is acceptable in *this* direction: a dense Meshroom cloud has sub-millimetre
    point spacing, so there is no meaningful sampling floor on the cloud side.

    Args:
        vertices: ``(V, 3)`` float64 mesh vertices, metres, SMPL/world frame.
        points: ``(N, 3)`` float64 cloud points, metres, **same frame as** ``vertices``.

    Returns:
        ``(V,)`` float64 distances in **metres**, one per vertex.

    Raises:
        ValueError: on malformed shapes, empty input, or non-finite values.
    """
    verts = _as_points(vertices, "vertices")
    cloud = _as_points(points, "points")

    tree = cKDTree(cloud)
    distances, _ = tree.query(verts, k=1)
    return np.asarray(distances, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# 7.M5 — tessellation floor
# ---------------------------------------------------------------------------


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(F,)`` float64 triangle areas in square metres."""
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    return np.asarray(0.5 * np.linalg.norm(cross, axis=1), dtype=np.float64)


def sample_surface(
    vertices: np.ndarray, faces: np.ndarray, *, n_samples: int, seed: int
) -> np.ndarray:
    """Area-weighted uniform samples on a triangle mesh surface.

    Args:
        vertices: ``(V, 3)`` float64 mesh vertices, metres.
        faces: ``(F, 3)`` int64 triangle indices.
        n_samples: number of surface samples to draw.
        seed: RNG seed — the only stochastic step in this module (master D12).

    Returns:
        ``(n_samples, 3)`` float64 points on the surface, metres, same frame as ``vertices``.
    """
    verts = _as_points(vertices, "vertices", min_rows=3)
    tris = _as_faces(faces, verts.shape[0])
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    areas = _triangle_areas(verts, tris)
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("mesh has zero total surface area — cannot sample")

    rng = np.random.default_rng(seed)

    # Area-weighted face choice via inverse-CDF sampling (deterministic given `seed`).
    cumulative = np.cumsum(areas)
    picks = np.searchsorted(cumulative, rng.random(n_samples) * total_area, side="right")
    picks = np.clip(picks, 0, tris.shape[0] - 1)

    # Uniform barycentric coordinates: reflect the (u + v > 1) half back into the triangle.
    uv = rng.random((n_samples, 2))
    outside = uv.sum(axis=1) > 1.0
    uv[outside] = 1.0 - uv[outside]

    a = verts[tris[picks, 0]]
    b = verts[tris[picks, 1]]
    c = verts[tris[picks, 2]]
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    return np.asarray(a + u * (b - a) + v * (c - a), dtype=np.float64)


def tessellation_floor(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    n_samples: int = DEFAULT_TESSELLATION_FLOOR_SAMPLES,
    seed: int = DEFAULT_TESSELLATION_FLOOR_SEED,
) -> dict[str, float]:
    """Measure the irreducible offset of any *vertex-based* surface distance (7.M5).

    Samples ``n_samples`` points uniformly (area-weighted) on the mesh surface and reports how
    far each one is from its nearest **vertex**. That is exactly the error a vertex-based
    cloud -> mesh metric would incur on a perfectly fitted mesh, so it belongs beside the
    headline number: on a mesh of equilateral triangles of edge ``L`` the maximum approaches
    ``L / sqrt(3)`` (the circumcentre), i.e. **5.77 mm at L = 10 mm**.

    Args:
        vertices: ``(V, 3)`` float64 mesh vertices, metres.
        faces: ``(F, 3)`` int64 triangle indices.
        n_samples: surface samples to draw (master Section 5.2 default: 100 000).
        seed: RNG seed (master Section 5.2 default: 0). Same seed => identical result.

    Returns:
        ``{"mean": float, "max": float}`` in **millimetres**.

    Raises:
        ValueError: on malformed inputs, ``n_samples < 1``, or a zero-area mesh.
    """
    verts = _as_points(vertices, "vertices", min_rows=3)
    samples = sample_surface(verts, faces, n_samples=n_samples, seed=seed)

    tree = cKDTree(verts)
    distances_m, _ = tree.query(samples, k=1)
    distances_mm = np.asarray(distances_m, dtype=np.float64) * _M_TO_MM
    return {"mean": float(distances_mm.mean()), "max": float(distances_mm.max())}


# ---------------------------------------------------------------------------
# 7.M3/7.M4 — the report
# ---------------------------------------------------------------------------


def _summarise_mm(distances_m: np.ndarray) -> dict[str, float]:
    """Aggregate metre-valued distances into the 7.M4 statistics, in **millimetres**.

    This is the module's single metre -> millimetre conversion point.
    """
    d = np.asarray(distances_m, dtype=np.float64).reshape(-1) * _M_TO_MM
    if d.size == 0:
        raise ValueError("cannot summarise an empty distance array")
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "rms": float(np.sqrt(np.mean(np.square(d)))),
        "p95": float(np.percentile(d, 95.0)),
        "max": float(d.max()),
    }


@dataclass
class ChamferReport:
    """7.M-compliant surface report.

    Deliberately has **no** combined/fused field: 7.M3 forbids reporting a single number, so
    the type makes it unrepresentable. A fused number hides which side is failing, and a
    one-directional chamfer is gameable by a mesh that collapses into the densest region of
    the cloud. Do not add ``chamfer_mm`` / ``combined`` / ``total`` / ``mean_both``.

    All distance values are in **millimetres** (7.M4); the conversion happens once, in
    :func:`chamfer_report`.
    """

    cloud_to_mesh_mm: dict[str, float] = field(default_factory=dict)
    """7.M1 point-to-SURFACE, cloud -> mesh. Keys: mean, median, rms, p95, max."""

    mesh_to_cloud_mm: dict[str, float] = field(default_factory=dict)
    """7.M2 vertex-to-nearest-point, mesh -> cloud. Keys: mean, median, rms, p95, max."""

    tessellation_floor_mm: dict[str, float] = field(default_factory=dict)
    """7.M5 vertex-sampling floor of this mesh. Keys: mean, max."""

    n_cloud_points: int = 0
    n_mesh_vertices: int = 0
    cloud_to_mesh_method: str = CLOUD_TO_MESH_METHOD
    mesh_to_cloud_method: str = MESH_TO_CLOUD_METHOD
    units: str = "mm"  # 7.M4

    def to_dict(self) -> dict[str, float | int | str]:
        """Flat, JSON-serialisable keys for ``quality.json``.

        Every distance key ends in ``_mm`` and names its direction and aggregation, e.g.
        ``chamfer_cloud_to_mesh_mean_mm``. The two directions stay separate (7.M3).
        """
        out: dict[str, float | int | str] = {}
        for stat, value in self.cloud_to_mesh_mm.items():
            out[f"chamfer_cloud_to_mesh_{stat}_mm"] = float(value)
        for stat, value in self.mesh_to_cloud_mm.items():
            out[f"chamfer_mesh_to_cloud_{stat}_mm"] = float(value)
        for stat, value in self.tessellation_floor_mm.items():
            out[f"tessellation_floor_{stat}_mm"] = float(value)
        out["n_cloud_points"] = int(self.n_cloud_points)
        out["n_mesh_vertices"] = int(self.n_mesh_vertices)
        out["cloud_to_mesh_method"] = self.cloud_to_mesh_method
        out["mesh_to_cloud_method"] = self.mesh_to_cloud_method
        out["units"] = self.units
        return out


def chamfer_report(
    cloud_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cfg: Tier3MetricConfig | None = None,
) -> ChamferReport:
    """Assemble the full 7.M-compliant surface report. Both directions, never fused (7.M3).

    Args:
        cloud_points: ``(N, 3)`` float64 cloud points, metres, SMPL/world frame (i.e. the cloud
            **after** Tier 3 alignment — this function applies no transform).
        vertices: ``(V, 3)`` float64 posed mesh vertices, metres, same frame.
        faces: ``(F, 3)`` int64 triangle indices.
        cfg: optional ``Tier3Config`` (master Section 5.2). Only
            ``tessellation_floor_samples`` and ``tessellation_floor_seed`` are read; ``None``
            uses the master defaults (100 000 samples, seed 0).

    Returns:
        A :class:`ChamferReport` with every distance in **millimetres**.
    """
    if cfg is None:
        n_samples = DEFAULT_TESSELLATION_FLOOR_SAMPLES
        seed = DEFAULT_TESSELLATION_FLOOR_SEED
    else:
        n_samples = cfg.tessellation_floor_samples
        seed = cfg.tessellation_floor_seed

    cloud = _as_points(cloud_points, "cloud_points")
    verts = _as_points(vertices, "vertices", min_rows=3)

    cloud_to_mesh_m = point_to_surface_distances(cloud, verts, faces)
    mesh_to_cloud_m = vertex_to_point_distances(verts, cloud)
    floor_mm = tessellation_floor(verts, faces, n_samples=n_samples, seed=seed)

    return ChamferReport(
        cloud_to_mesh_mm=_summarise_mm(cloud_to_mesh_m),
        mesh_to_cloud_mm=_summarise_mm(mesh_to_cloud_m),
        tessellation_floor_mm=floor_mm,
        n_cloud_points=int(cloud.shape[0]),
        n_mesh_vertices=int(verts.shape[0]),
    )
