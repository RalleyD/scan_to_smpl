"""Unit tests for the Tier 3 surface metrics (REVIEW.md 7.M1-7.M6).

Everything here is pure geometry: **no SMPL model weights, no GPU**. These are the tests the
Tier 3 gate rests on, so a skip is not a pass — this file carries no `requires_smpl` / `gpu`
marker by design.

Run with:
    pytest tests/test_surface_metrics.py -v
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from scantosmpl.evaluation.surface_metrics import (
    CLOUD_TO_MESH_METHOD,
    MESH_TO_CLOUD_METHOD,
    STAT_KEYS,
    ChamferReport,
    chamfer_report,
    point_to_surface_distances,
    sample_surface,
    tessellation_floor,
    vertex_to_point_distances,
)

# ---------------------------------------------------------------------------
# Fixtures — analytic geometry, hand-checkable
# ---------------------------------------------------------------------------

#: The 7.M1 reference triangle: a 1 m right triangle in the z = 0 plane.
UNIT_TRIANGLE_V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
UNIT_TRIANGLE_F = np.array([[0, 1, 2]], dtype=np.int64)

EQUILATERAL_EDGE_M = 0.010  # 10 mm — the SMPL template's characteristic triangle size


def _equilateral(edge: float = EQUILATERAL_EDGE_M) -> tuple[np.ndarray, np.ndarray]:
    """A single equilateral triangle of the given edge length, in the z = 0 plane."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [edge, 0.0, 0.0], [edge / 2.0, edge * math.sqrt(3.0) / 2.0, 0.0]],
        dtype=np.float64,
    )
    return vertices, np.array([[0, 1, 2]], dtype=np.int64)


def _subdivided_equilateral(edge: float = EQUILATERAL_EDGE_M) -> tuple[np.ndarray, np.ndarray]:
    """The same equilateral triangle split into 4 sub-triangles of edge/2."""
    v, _ = _equilateral(edge)
    a, b, c = v[0], v[1], v[2]
    ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
    vertices = np.stack([a, b, c, ab, bc, ca]).astype(np.float64)
    faces = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]], dtype=np.int64)
    return vertices, faces


def _unit_square_mesh() -> tuple[np.ndarray, np.ndarray]:
    """The unit square in the z = 0 plane as two triangles."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


# ---------------------------------------------------------------------------
# AC1 (7.M1) — cloud -> mesh is point-to-SURFACE, not vertex-based
# ---------------------------------------------------------------------------


def _nearest_vertex_distance(query: np.ndarray, vertices: np.ndarray) -> float:
    """Closed-form nearest-VERTEX distance — the quantity 7.M1 forbids for cloud->mesh."""
    return float(np.linalg.norm(vertices - query.reshape(1, 3), axis=1).min())


def test_point_to_surface_analytic():
    """AC1. All three Voronoi regions of a triangle, checked against closed-form answers.

    Each case also asserts the nearest-*vertex* distance, so this test fails loudly if a
    vertex-based implementation is ever silently swapped in.
    """
    v, f = UNIT_TRIANGLE_V, UNIT_TRIANGLE_F

    # --- Face region: 5 mm above an interior point. The brief's headline case. ---
    query_face = np.array([[0.25, 0.25, 0.005]], dtype=np.float64)
    d_face = point_to_surface_distances(query_face, v, f)
    assert d_face.shape == (1,)
    assert d_face.dtype == np.float64
    assert d_face[0] == pytest.approx(0.005, abs=1e-6)

    # ...whereas the nearest VERTEX is ~0.354 m away — nearly two orders of magnitude out.
    nearest_vertex_face = _nearest_vertex_distance(query_face[0], v)
    assert nearest_vertex_face == pytest.approx(math.sqrt(0.25**2 + 0.25**2 + 0.005**2), abs=1e-9)
    assert nearest_vertex_face == pytest.approx(0.3536, abs=1e-3)
    assert nearest_vertex_face > 50 * d_face[0]

    # --- Edge region: closest feature is the (0,0,0)-(1,0,0) edge at (0.5, 0, 0). ---
    query_edge = np.array([[0.5, -0.3, 0.4]], dtype=np.float64)
    d_edge = point_to_surface_distances(query_edge, v, f)
    assert d_edge[0] == pytest.approx(0.5, abs=1e-6)  # sqrt(0.3^2 + 0.4^2)
    assert _nearest_vertex_distance(query_edge[0], v) == pytest.approx(math.sqrt(0.5), abs=1e-9)
    assert d_edge[0] < _nearest_vertex_distance(query_edge[0], v) - 0.2

    # --- Vertex region: closest feature is vertex (0, 1, 0); surface == vertex distance. ---
    query_vertex = np.array([[-0.3, 1.4, 0.0]], dtype=np.float64)
    d_vertex = point_to_surface_distances(query_vertex, v, f)
    assert d_vertex[0] == pytest.approx(0.5, abs=1e-6)  # sqrt(0.3^2 + 0.4^2)
    assert d_vertex[0] == pytest.approx(_nearest_vertex_distance(query_vertex[0], v), abs=1e-6)


def test_point_to_surface_centroid_of_equilateral_metre_triangle():
    """AC1, in the master spec's exact wording: 5 mm above the *centroid* of a 1 m triangle
    returns 0.005 m, whereas the nearest vertex is the circumradius 1/sqrt(3) = 0.577 m.

    (The brief's right-triangle query at (0.25, 0.25, 0.005) is not that triangle's centroid,
    so its nearest-vertex distance is 0.354 m; both readings are covered, and both fail if a
    vertex-based implementation is substituted.)
    """
    edge = 1.0
    vertices = np.array(
        [[0.0, 0.0, 0.0], [edge, 0.0, 0.0], [edge / 2.0, edge * math.sqrt(3.0) / 2.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    centroid = vertices.mean(axis=0)
    query = (centroid + np.array([0.0, 0.0, 0.005])).reshape(1, 3)

    d = point_to_surface_distances(query, vertices, faces)
    assert d[0] == pytest.approx(0.005, abs=1e-6)

    nearest_vertex = _nearest_vertex_distance(query[0], vertices)
    assert nearest_vertex == pytest.approx(0.5774, abs=1e-3)  # 1/sqrt(3), the spec's 0.577
    assert nearest_vertex > 100 * d[0]


def test_point_to_surface_is_unsigned():
    """Points on either side of the triangle return the same positive distance (no sign)."""
    above = np.array([[0.25, 0.25, 0.01]], dtype=np.float64)
    below = np.array([[0.25, 0.25, -0.01]], dtype=np.float64)
    d_above = point_to_surface_distances(above, UNIT_TRIANGLE_V, UNIT_TRIANGLE_F)
    d_below = point_to_surface_distances(below, UNIT_TRIANGLE_V, UNIT_TRIANGLE_F)
    assert d_above[0] > 0.0
    assert d_below[0] > 0.0
    assert d_above[0] == pytest.approx(d_below[0], abs=1e-9)


def test_point_to_surface_on_surface_is_zero():
    """A point lying exactly on the surface has zero distance."""
    on_surface = np.array([[0.25, 0.25, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64)
    d = point_to_surface_distances(on_surface, UNIT_TRIANGLE_V, UNIT_TRIANGLE_F)
    assert np.allclose(d, 0.0, atol=1e-7)


def test_point_to_surface_returns_float64_metres_and_does_not_mutate_inputs():
    """float32/uint32 is an Open3D implementation detail; it must not leak outward."""
    points = np.array([[0.25, 0.25, 0.005], [0.1, 0.1, -0.002]], dtype=np.float64)
    vertices = UNIT_TRIANGLE_V.copy()
    faces = UNIT_TRIANGLE_F.copy()
    d = point_to_surface_distances(points, vertices, faces)

    assert d.dtype == np.float64
    assert d.shape == (2,)
    assert np.array_equal(vertices, UNIT_TRIANGLE_V)
    assert vertices.dtype == np.float64
    assert faces.dtype == np.int64


def test_point_to_surface_rejects_bad_input():
    bad_shape = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        point_to_surface_distances(bad_shape, UNIT_TRIANGLE_V, UNIT_TRIANGLE_F)

    with pytest.raises(ValueError, match="at least 1 point"):
        point_to_surface_distances(np.zeros((0, 3)), UNIT_TRIANGLE_V, UNIT_TRIANGLE_F)

    with pytest.raises(ValueError, match="non-finite"):
        point_to_surface_distances(np.array([[np.nan, 0.0, 0.0]]), UNIT_TRIANGLE_V, UNIT_TRIANGLE_F)

    with pytest.raises(ValueError, match="out of range"):
        point_to_surface_distances(
            np.array([[0.0, 0.0, 0.1]]), UNIT_TRIANGLE_V, np.array([[0, 1, 7]], dtype=np.int64)
        )

    with pytest.raises(ValueError, match="integer array"):
        point_to_surface_distances(
            np.array([[0.0, 0.0, 0.1]]), UNIT_TRIANGLE_V, np.array([[0.0, 1.0, 2.0]])
        )


# ---------------------------------------------------------------------------
# 7.M2 — mesh -> cloud, vertex-to-nearest-point
# ---------------------------------------------------------------------------


def test_vertex_to_point_distances_hand_checked():
    """Two vertices against three cloud points, all distances checkable by eye."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    points = np.array(
        [
            [0.0, 0.3, 0.0],  # 0.3 from vertex 0, 1.044 from vertex 1
            [1.0, 0.0, 0.5],  # 1.118 from vertex 0, 0.5 from vertex 1
            [5.0, 5.0, 5.0],  # far from both — must never win
        ],
        dtype=np.float64,
    )
    d = vertex_to_point_distances(vertices, points)

    assert d.shape == (2,)
    assert d.dtype == np.float64
    assert d[0] == pytest.approx(0.3, abs=1e-12)
    assert d[1] == pytest.approx(0.5, abs=1e-12)


def test_vertex_to_point_distances_zero_when_coincident():
    vertices = np.array([[0.1, 0.2, 0.3], [-1.0, 0.0, 2.0]], dtype=np.float64)
    d = vertex_to_point_distances(vertices, vertices.copy())
    assert np.allclose(d, 0.0, atol=1e-12)


def test_vertex_to_point_rejects_empty_cloud():
    with pytest.raises(ValueError, match="at least 1 point"):
        vertex_to_point_distances(UNIT_TRIANGLE_V, np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# AC4 (7.M5) — tessellation floor
# ---------------------------------------------------------------------------


def test_tessellation_floor_bound():
    """AC4. On an equilateral triangle of edge L the worst case is the circumcentre, L/sqrt(3).

    At L = 10 mm (the SMPL template's characteristic edge) that bound is 5.77 mm — the number
    the master spec quotes as the reason a vertex-based cloud->mesh distance is unacceptable.
    """
    vertices, faces = _equilateral(EQUILATERAL_EDGE_M)
    bound_mm = (EQUILATERAL_EDGE_M / math.sqrt(3.0)) * 1000.0
    assert bound_mm == pytest.approx(5.7735, abs=1e-3)  # the spec's 5.77 mm

    floor = tessellation_floor(vertices, faces, n_samples=100_000, seed=0)

    assert set(floor) == {"mean", "max"}
    assert floor["max"] <= 1.05 * bound_mm  # must not exceed the analytic worst case
    assert floor["max"] >= 0.95 * bound_mm  # dense sampling must approach it
    assert 0.0 < floor["mean"] < floor["max"]


def test_tessellation_floor_shrinks_with_finer_tessellation():
    """Halving the edge length must roughly halve the floor — it is a sampling artefact."""
    coarse = tessellation_floor(*_equilateral(), n_samples=50_000, seed=0)
    fine = tessellation_floor(*_subdivided_equilateral(), n_samples=50_000, seed=0)
    assert fine["max"] < 0.6 * coarse["max"]
    assert fine["mean"] < 0.6 * coarse["mean"]


def test_tessellation_floor_is_deterministic():
    """D12: the one seeded RNG in this module. Same seed => bitwise-identical result."""
    vertices, faces = _equilateral()
    a = tessellation_floor(vertices, faces, n_samples=5_000, seed=0)
    b = tessellation_floor(vertices, faces, n_samples=5_000, seed=0)
    assert a == b

    c = tessellation_floor(vertices, faces, n_samples=5_000, seed=1)
    assert c["mean"] != a["mean"]  # a different seed really does draw different samples


def test_sample_surface_is_area_weighted_and_on_the_surface():
    """Samples must lie on the mesh, and a triangle 9x the area must get ~9x the samples."""
    # Two DISJOINT coplanar triangles (the second is translated well clear of the first),
    # so "which triangle did this sample come from" is unambiguous.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
            [10.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)  # areas 0.5 and 4.5 -> 1:9
    samples = sample_surface(vertices, faces, n_samples=20_000, seed=0)

    assert samples.shape == (20_000, 3)
    assert samples.dtype == np.float64
    assert np.allclose(samples[:, 2], 0.0, atol=1e-12)  # both triangles lie in z = 0

    # Distance to the mesh surface is zero for every sample (round-trip against 7.M1).
    assert np.allclose(point_to_surface_distances(samples, vertices, faces), 0.0, atol=1e-6)

    # ~10% of samples fall on the small triangle (area ratio 0.5 : 4.5).
    in_small = samples[:, 0] < 5.0
    assert in_small.mean() == pytest.approx(0.1, abs=0.02)
    # Each sample is inside its own triangle, not merely in the bounding box.
    small = samples[in_small]
    large = samples[~in_small]
    assert (small[:, 0] + small[:, 1] <= 1.0 + 1e-9).all()
    assert ((large[:, 0] - 10.0) + large[:, 1] <= 3.0 + 1e-9).all()
    assert (small >= -1e-12)[:, :2].all()
    assert (large[:, 0] >= 10.0 - 1e-12).all()
    assert (large[:, 1] >= -1e-12).all()


def test_tessellation_floor_rejects_degenerate_mesh():
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="zero total surface area"):
        tessellation_floor(collinear, np.array([[0, 1, 2]], dtype=np.int64), n_samples=10)

    with pytest.raises(ValueError, match="n_samples must be"):
        tessellation_floor(*_equilateral(), n_samples=0)


# ---------------------------------------------------------------------------
# AC2 (7.M3) — the two directions are never fused
# ---------------------------------------------------------------------------


def test_chamfer_report_has_no_fused_field():
    """AC2. 7.M3 makes a single combined number a spec violation; the type forbids it."""
    names = [f.name for f in dataclasses.fields(ChamferReport)]

    assert "cloud_to_mesh_mm" in names
    assert "mesh_to_cloud_mm" in names

    forbidden = ("chamfer_mm", "combined", "total", "mean_both")
    for name in names:
        for token in forbidden:
            assert token not in name, f"ChamferReport.{name} fuses the two directions (7.M3)"

    # Nor may a fused number sneak in as a property/method on the class.
    for token in forbidden:
        assert not hasattr(ChamferReport, token)

    # ...nor into the serialised artefact.
    report = _report_on_offset_cloud()
    for key in report.to_dict():
        for token in forbidden:
            assert token not in key, f"quality.json key {key!r} fuses the two directions (7.M3)"


# ---------------------------------------------------------------------------
# AC3 (7.M4) — aggregation + units are explicit
# ---------------------------------------------------------------------------


def _report_on_offset_cloud(offset_m: float = 0.002, cfg=None) -> ChamferReport:
    """A cloud floating a known distance above the unit square: every cloud->mesh
    distance is exactly ``offset_m``, so the mm conversion is checkable."""
    vertices, faces = _unit_square_mesh()
    rng = np.random.default_rng(0)
    xy = rng.random((500, 2)) * 0.8 + 0.1  # strictly inside the square
    cloud = np.column_stack([xy, np.full(500, offset_m)])
    return chamfer_report(cloud, vertices, faces, cfg)


def test_chamfer_report_units_and_aggregations():
    """AC3. Both directions carry mean/median/rms/p95/max; units are declared as mm."""
    report = _report_on_offset_cloud(offset_m=0.002)

    assert set(report.cloud_to_mesh_mm) == set(STAT_KEYS)
    assert set(report.mesh_to_cloud_mm) == set(STAT_KEYS)
    assert set(report.tessellation_floor_mm) == {"mean", "max"}
    assert report.units == "mm"

    # 2 mm offset -> 2.0 in every cloud->mesh aggregation (conversion happens exactly once).
    for stat in STAT_KEYS:
        assert report.cloud_to_mesh_mm[stat] == pytest.approx(2.0, abs=1e-3)

    assert report.n_cloud_points == 500
    assert report.n_mesh_vertices == 4
    assert report.cloud_to_mesh_method == CLOUD_TO_MESH_METHOD
    assert CLOUD_TO_MESH_METHOD == "point_to_surface_open3d_raycasting"  # AC1 evidence
    assert report.mesh_to_cloud_method == MESH_TO_CLOUD_METHOD == "vertex_to_nearest_point"


def test_chamfer_report_directions_differ():
    """The two directions measure different things — the report must not collapse them."""
    report = _report_on_offset_cloud(offset_m=0.002)
    # Cloud sits above the square's interior; the corners are far from any cloud point,
    # so mesh->cloud is much larger than cloud->mesh. This asymmetry is exactly why 7.M3
    # forbids a single fused number.
    assert report.mesh_to_cloud_mm["mean"] > 5.0 * report.cloud_to_mesh_mm["mean"]


def test_chamfer_report_to_dict_is_flat_and_json_serialisable():
    import json

    report = _report_on_offset_cloud()
    d = report.to_dict()

    assert d["chamfer_cloud_to_mesh_mean_mm"] == pytest.approx(2.0, abs=1e-3)
    for stat in STAT_KEYS:
        assert f"chamfer_cloud_to_mesh_{stat}_mm" in d
        assert f"chamfer_mesh_to_cloud_{stat}_mm" in d
    assert "tessellation_floor_mean_mm" in d
    assert "tessellation_floor_max_mm" in d

    # Every distance-valued key names its units (7.M4); no nested containers.
    for key, value in d.items():
        assert isinstance(value, (int, float, str))
        if isinstance(value, float):
            assert key.endswith("_mm"), f"float key {key!r} does not declare its units"

    assert json.loads(json.dumps(d)) == d


def test_chamfer_report_respects_config_protocol():
    """`Tier3Config` is consumed structurally — only the two 7.M5 fields are read."""

    @dataclasses.dataclass
    class _Cfg:
        tessellation_floor_samples: int = 1_000
        tessellation_floor_seed: int = 7

    cfg = _Cfg()
    report = _report_on_offset_cloud(cfg=cfg)
    expected = tessellation_floor(*_unit_square_mesh(), n_samples=1_000, seed=7)
    assert report.tessellation_floor_mm == expected


def test_chamfer_report_is_deterministic():
    a = _report_on_offset_cloud()
    b = _report_on_offset_cloud()
    assert a == b
