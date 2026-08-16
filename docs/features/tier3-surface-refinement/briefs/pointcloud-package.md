---
component: pointcloud-package
agent: python-engineer
worktree: true   # parallel-safe with surface-metrics and smpld-and-losses — no shared files
---

# Component Brief — pointcloud-package

## Goal

Build out `scantosmpl/pointcloud/`, today an empty package, into the complete Phase 6 input path:
load a PLY/OBJ point cloud in its arbitrary source frame, clean it without assuming any particular
scale, align it **to** the Tier 2 SMPL mesh with a recovered 7-DoF similarity, and produce the
per-vertex semantic part weights AC 7.3 needs. When done, a Meshroom cloud with arbitrary scale,
orientation and origin becomes a metric, SMPL-frame `PointCloud` that the surface fitter can
consume, plus a `CloudAlignment` recording exactly what transform was applied.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/pointcloud/__init__.py`
- `scantosmpl/pointcloud/io.py`
- `scantosmpl/pointcloud/preprocess.py`
- `scantosmpl/pointcloud/align.py`
- `scantosmpl/pointcloud/segment.py`
- `tests/test_pointcloud.py`

**Does NOT touch**:
- `scantosmpl/config.py` — `Tier3Config` is `tier3-pipeline-artefacts`' territory. **Accept it as a
  typed parameter** and import it under `TYPE_CHECKING` only, or accept the individual values as
  keyword arguments with the master §5.2 defaults. Do NOT add the dataclass yourself, and do NOT
  block on it — the field names and defaults in master §5.2 are already locked, so write against
  them.
- `scantosmpl/types.py`, `scantosmpl/smpl/model.py`, `scantosmpl/fitting/*`, `scantosmpl/evaluation/*`
- Other components' `Owns` paths; `external/`, `models/`, `pyproject.toml`, `output/`

**Consumes**:
- `open3d` (0.19.0) and `numpy` — already dependencies. Nothing else. Do not `pip install`.
- SMPL `lbs_weights` `(6890, 24)` and `faces` `(13776, 3)` are passed in as arrays by the caller —
  this module must **not** import `SMPLModel` (keeps it testable without model weights).

**Produces** (available to other components):
- `PointCloud`, `load_pointcloud`, `save_pointcloud` — master §5.1 / §5.3
- `PreprocessStats`, `preprocess_cloud` — master §5.1 / §5.3
- `CloudAlignment`, `pca_triad`, `enumerate_proper_rotations`, `align_cloud_to_smpl` — master §5.1 / §5.3
- `SMPL_PART_GROUPS`, `smpl_part_labels`, `vertex_part_weights`, `transfer_labels_to_cloud` — master §5.3

## Steps

1. **`io.py` — `PointCloud` + load/save.** Implement the dataclass exactly per master §5.1,
   including the `frame` and `units` guard fields (defaults `"source"` / `"arbitrary"`). Load `.ply`
   and `.obj` via `open3d.io`; for an OBJ that parses as a mesh, take its vertices. Raise
   `ValueError` on an unsupported suffix or an empty cloud, `FileNotFoundError` on a missing path.
   `max_points` applies a deterministic stride subsample (`points[::step]`), **not** a random one.
   **Verify**: `py-typecheck` on `scantosmpl/pointcloud/io.py`.

2. **`preprocess.py` — unit-free cleaning.** `preprocess_cloud` runs, in order: statistical outlier
   removal (`remove_statistical_outlier`, `nb_neighbors` / `std_ratio`), then a voxel downsample
   whose voxel size is `voxel_fraction_of_bbox * bbox_diagonal` **in source units** (master D8 — a
   metric voxel size is meaningless before alignment and is the bug this step exists to avoid),
   iterating the fraction upward if the result still exceeds `target_points`. Then optional normal
   estimation (`estimate_normals` with `KDTreeSearchParamKNN(normal_knn)`). Populate every
   `PreprocessStats` field. `frame`/`units` pass through unchanged.
   **Verify**: `py-test` — `pytest tests/test_pointcloud.py -k preprocess -v`, including a test that
   the same cloud scaled by 1000× produces the same output point count (±5%).

3. **`align.py` — PCA triad + 24-candidate ICP.** `pca_triad` returns centroid, eigenvector columns
   ordered by descending eigenvalue, and `sqrt(eigenvalue)` extents. `enumerate_proper_rotations`
   returns all 24 rotations of the axis-permutation × sign-flip group with `det == +1`, in a fixed
   deterministic order — assert `len == 24`, all orthonormal, all `det ≈ +1`, all distinct.
   `align_cloud_to_smpl` then: builds both triads, derives an initial scale from the ratio of
   dominant extents, and for each of the 24 candidates runs Open3D point-to-plane ICP (correspondence
   distance = `icp_threshold_frac * smpl_bbox_diagonal`, `icp_max_iterations`) with scale estimation
   enabled; keeps the candidate with the lowest `inlier_rmse`. Set `converged = fitness >= icp_min_fitness`.
   Return the transformed cloud with `frame="smpl_world"`, `units="metres"`.
   - `CloudAlignment.apply` implements `scale * (R @ p) + t` and `as_matrix` the equivalent `(4,4)`.
   - **No RNG anywhere** (master D12). Do not use `registration_ransac_based_on_feature_matching`.
   **Verify**: `py-test` — `pytest tests/test_pointcloud.py -k align -v`. Must include: a
   known-similarity round-trip on a synthetic non-symmetric shape (recover scale <1%, rotation <1°,
   translation <5mm), a determinism test (two runs bitwise-identical, AC7), and a 180°-flip test
   proving the enumeration picks the correct basin.

4. **`segment.py` — semantic part weights from `lbs_weights`.** Define `SMPL_PART_GROUPS` exactly as
   master §5.3 (the six groups partition all 24 SMPL joints — assert this at import or in a test).
   `smpl_part_labels` maps `lbs_weights.argmax(axis=1)` → joint → group id. `vertex_part_weights`
   turns those into a `(6890,)` float32 weight array from a `{group: weight}` dict, raising on an
   unknown group name. `transfer_labels_to_cloud` assigns each cloud point its nearest mesh vertex's
   label (use `scipy.spatial.cKDTree` — scipy is already a dependency).
   **Verify**: `py-test` — `pytest tests/test_pointcloud.py -k segment -v`. Assert the six groups
   cover `set(range(24))` with no overlap, that every one of the 6890 vertices gets exactly one
   label, and that the group names match `Tier3Config.body_part_weights`' keys exactly.

5. **`__init__.py` exports + lint.** Export the public names above, following the
   `scantosmpl/triangulation/__init__.py` pattern (module docstring, explicit imports, `__all__`).
   **Verify**: `py-lint`, `py-typecheck` on all five files, then `pytest tests/ -x --tb=short`.

## Definition of done

- Every step's verification skill is green.
- The `Produces` contract exactly matches master §5.1/§5.3 — same names, same argument order, same
  return types.
- `align_cloud_to_smpl` is deterministic (AC7) and never imports `SMPLModel`.
- No new dependency (AC22): only `open3d`, `numpy`, `scipy`.
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new
  skill — in particular, report the **measured** rotation/scale/translation recovery error from
  step 3's round-trip test, since AC5's thresholds on the real fixture were set from estimates and
  the integration brief will inherit them. Also state explicitly that `pipeline-smoke` was **not**
  run and why (no pipeline exists yet).
