---
component: surface-metrics
agent: python-engineer
worktree: true   # parallel-safe with pointcloud-package and smpld-and-losses — no shared files
---

# Component Brief — surface-metrics

## Goal

Implement the **binding** Tier 3 metric defined by REVIEW.md 7.M1–7.M6. This is the number the Tier 3
gate is judged on, so it must be point-to-*surface* in the cloud→mesh direction (not vertex-based),
must report both directions separately, must state its aggregation and units, and must report the
tessellation floor beside the result so the headline number is interpretable. Getting this module
right is what stops the project from spending half the 8 mm budget on SMPL's own ~1 cm triangle
size before the optimiser does anything.

This module is **reporting only** — nothing here needs to be differentiable, and nothing here is
used inside the optimiser. The loss lives in `surface_losses.py` (a sibling brief); 7.M6 explicitly
permits them to differ.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/evaluation/surface_metrics.py`
- `scantosmpl/evaluation/__init__.py`  (currently empty)
- `tests/test_surface_metrics.py`

**Does NOT touch**:
- `scantosmpl/evaluation/{ab_refit,leave_one_view_out,visualise}.py` — existing Tier 2 diagnostics,
  unrelated and unmodified.
- `scantosmpl/config.py` — accept `Tier3Config` as a typed parameter (import under `TYPE_CHECKING`)
  or take `n_samples`/`seed` as keyword arguments with the master §5.2 defaults. Do not define it.
- `scantosmpl/pointcloud/*`, `scantosmpl/fitting/*`, `scantosmpl/smpl/*`, `scantosmpl/types.py`
- Other components' `Owns` paths; `external/`, `models/`, `pyproject.toml`, `output/`

**Consumes**:
- `open3d` (0.19.0) and `numpy` only. **Not** `kaolin` (no wheel for torch 2.11 — master D2), **not**
  `trimesh.proximity` (needs `rtree`, absent). If you reach for either, you have taken a wrong turn.

**Produces**:
- `ChamferReport` — master §5.1
- `point_to_surface_distances`, `vertex_to_point_distances`, `tessellation_floor`, `chamfer_report`
  — master §5.3

## Steps

1. **`point_to_surface_distances`** — build an `o3d.t.geometry.RaycastingScene`, `add_triangles` the
   mesh, and call `compute_distance` on the query points. This is 7.M1's binding measurement.
   - Open3D's tensor API needs `float32` vertices and `uint32` triangles; convert at the call
     boundary and keep the **public signature `float64` in / `float64` out, in metres** (repo spec
     §Coordinate frames). Do not let `float32`/`uint32` leak outward.
   - Returns unsigned distances. Signed distance is not wanted — inside/outside is meaningless for
     an open photogrammetry cloud.
   **Verify**: `py-test` — `pytest tests/test_surface_metrics.py::test_point_to_surface_analytic -v`.
   This test is AC1 and must be **analytic**, not a snapshot: a single triangle
   `[(0,0,0),(1,0,0),(0,1,0)]` and a query at `(0.25, 0.25, 0.005)` must return `0.005 ± 1e-6`,
   while the nearest *vertex* is ≈ 0.577 away — assert both, so the test would fail if someone
   silently swapped in a vertex-based implementation. (This exact case is verified working in this
   env.) Add at least two more analytic cases: a point whose closest feature is an **edge**, and one
   whose closest feature is a **vertex**, so all three Voronoi regions of the triangle are covered.

2. **`vertex_to_point_distances`** — 7.M2's mesh→cloud direction, vertex-to-nearest-cloud-point via
   `scipy.spatial.cKDTree` (scipy is already a dependency) or an Open3D KD-tree. Metres.
   **Verify**: `py-test` — a hand-checkable case with 3 points and 2 vertices.

3. **`tessellation_floor`** — 7.M5. Area-weighted uniform sampling of `n_samples` points on the mesh
   surface, then each sample's distance to its **nearest vertex**; return `{"mean": mm, "max": mm}`.
   Seed with `numpy.random.default_rng(seed)` — this is the one seeded RNG in the module.
   **Verify**: `py-test` — `pytest tests/test_surface_metrics.py::test_tessellation_floor_bound -v`
   (AC4). On a single equilateral triangle of edge `L`, `max` must approach `L/sqrt(3)` (the
   centroid), and must not exceed `1.05 * L/sqrt(3)`. For reference, at `L = 10 mm` that bound is
   **5.77 mm** — the number the master spec quotes as the reason vertex-based cloud→mesh distance is
   unacceptable.

4. **`ChamferReport` + `chamfer_report`** — assemble both direction dicts with keys
   `mean, median, rms, p95, max`, all in **millimetres** (convert once, here — repo spec's unit
   boundary), plus the floor, counts, and the two method strings.
   - **7.M3 is enforced structurally**: `ChamferReport` must have **no** field that fuses the two
     directions. Do not add a `chamfer_mm`, `combined`, `total` or `mean_both` property, however
     convenient it looks — AC2 greps the dataclass fields for exactly this.
   - Provide `to_dict()` returning flat JSON-serialisable keys for `quality.json`.
   **Verify**: `py-test` — `pytest tests/test_surface_metrics.py -v`, plus AC2's field-name check
   asserted as a unit test in this file.

5. **`__init__.py` exports + lint.** Export `ChamferReport` and the four functions. Follow the
   `scantosmpl/calibration/__init__.py` pattern (docstring, explicit imports, `__all__`).
   **Verify**: `py-lint`, `py-typecheck`, then `pytest tests/ -x --tb=short`.

## Definition of done

- Every step's verification skill is green.
- AC1, AC2, AC3 and AC4 are each discharged by a named test in `tests/test_surface_metrics.py`.
- The `Produces` contract exactly matches master §5.1/§5.3.
- No `kaolin`, no `rtree`, no `pyproject.toml` change (AC22).
- All tests in this file run **without SMPL model weights and without a GPU** — the analytic cases
  are pure geometry, so this file must not carry `requires_smpl` or `gpu` markers. A skipped test is
  not a pass, and these are the tests the Tier 3 gate rests on.
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new
  skill — in particular, report the **measured** tessellation floor (mean and max) for the real SMPL
  template mesh if model weights happened to be available, since that number contextualises AC9's
  8 mm and belongs in the eventual report. State explicitly that `pipeline-smoke` was not run and why.
