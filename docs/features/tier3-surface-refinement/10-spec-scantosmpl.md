# Repo Spec — Tier 3 Surface Refinement (ScanToSMPL)

Applies the master spec to this repo's conventions and lists the exact skills the loop runs.

## Subpackages touched

| Path | Change | Brief |
|---|---|---|
| `scantosmpl/pointcloud/__init__.py` | currently empty — gains exports | `pointcloud-package` |
| `scantosmpl/pointcloud/io.py` | **new** — `PointCloud`, `load_pointcloud`, `save_pointcloud` | `pointcloud-package` |
| `scantosmpl/pointcloud/preprocess.py` | **new** — `PreprocessStats`, `preprocess_cloud` | `pointcloud-package` |
| `scantosmpl/pointcloud/align.py` | **new** — `CloudAlignment`, `pca_triad`, `enumerate_proper_rotations`, `align_cloud_to_smpl` | `pointcloud-package` |
| `scantosmpl/pointcloud/segment.py` | **new** — `SMPL_PART_GROUPS`, `smpl_part_labels`, `vertex_part_weights`, `transfer_labels_to_cloud` | `pointcloud-package` |
| `scantosmpl/evaluation/surface_metrics.py` | **new** — `ChamferReport` + the binding 7.M metric | `surface-metrics` |
| `scantosmpl/evaluation/__init__.py` | currently empty — gains exports | `surface-metrics` |
| `scantosmpl/smpl/model.py` | **additive** — `displacements` parameter + `forward` kwargs | `smpld-and-losses` |
| `scantosmpl/fitting/surface_losses.py` | **new** — chamfer, normal, Laplacian, `D` regularisation | `smpld-and-losses` |
| `scantosmpl/fitting/surface.py` | **new** — `SurfaceStage`, `SurfaceFitResult`, `Tier3SurfaceFitter` | `surface-fitting` |
| `scantosmpl/fitting/surface_pipeline.py` | **new** — `Tier3Result`, `Tier3Pipeline` | `tier3-pipeline-artefacts` |
| `scantosmpl/fitting/artefacts.py` | **new** — the 7.B artefact + manifest writer | `tier3-pipeline-artefacts` |
| `scantosmpl/fitting/__init__.py` | export wiring for **all** new fitting symbols | `tier3-pipeline-artefacts` |
| `scantosmpl/types.py` | `Tier3Quality`, `PoseArtefact`, 3 constants, 1 field on `FittingResult` | `tier3-pipeline-artefacts` |
| `scantosmpl/config.py` | `Tier3Config` + `PipelineConfig.tier3` | `tier3-pipeline-artefacts` |
| `scantosmpl/cli.py` | `fit-surface` command | `tier3-pipeline-artefacts` |
| `tests/test_pointcloud.py` | **new** | `pointcloud-package` |
| `tests/test_surface_metrics.py` | **new** | `surface-metrics` |
| `tests/test_surface_losses.py` | **new** | `smpld-and-losses` |
| `tests/test_surface_fitting.py` | **new** | `surface-fitting` |
| `tests/test_artefacts.py` | **new** | `tier3-pipeline-artefacts` |
| `tests/integration/test_tier3_integration.py` | **new** | `tier3-pipeline-artefacts` |
| `tests/integration/fixtures/synthetic_cloud/` | **new** — `make_fixture.py`, generated `cloud.ply` + `ground_truth.json` | `tier3-pipeline-artefacts` |

**Deliberately NOT touched**: `scantosmpl/fitting/{losses,optimiser,pipeline,rear_views}.py`,
`scantosmpl/{detection,hmr,calibration,triangulation}/`, `scantosmpl/smpl/joint_map.py`,
`scantosmpl/utils/`, `pyproject.toml`, `external/`, `models/`.

`scantosmpl/smpl/__init__.py` stays empty (it is today) — no brief touches it.

## Coordinate frames + units

Silent frame mismatches are the P0 bug class here: a cloud left in its source frame will still
optimise, still converge, and still write a plausible-looking `D`. Hence `PointCloud.frame` exists
as a runtime guard and `Tier3SurfaceFitter.fit` asserts on it.

| Tensor / array | Shape | Dtype | Frame | Units |
|---|---|---|---|---|
| `PointCloud.points` (loaded) | `(N, 3)` | `float64` | source (Meshroom) | **arbitrary** |
| `PointCloud.points` (aligned) | `(N, 3)` | `float64` | SMPL/world | metres |
| `PointCloud.normals` | `(N, 3)` | `float64` | same as points, unit length | — |
| `CloudAlignment.rotation` / `.translation` | `(3,3)` / `(3,)` | `float64` | source → SMPL/world | — / metres |
| `CloudAlignment.scale` | scalar | `float` | source → SMPL/world | metres per source unit |
| `RefinementResult.vertices` (Tier 2 in) | `(6890, 3)` | `float64` | SMPL/world | metres |
| `SMPLOutput.vertices` | `(1, 6890, 3)` | `float32` | SMPL/world, posed | metres |
| `displacements` `D` | `(1, 6890, 3)` tensor / `(6890, 3)` npz | `float32` | **SMPL/world, posed** (`"posed_world"`) | metres |
| `SurfaceFitResult.base_vertices` | `(6890, 3)` | `float32` | SMPL/world, posed, `D = 0` | metres |
| `point_to_surface_distances` out | `(N,)` | `float64` | — | **metres** |
| `ChamferReport.*_mm` values | scalars | `float` | — | **millimetres** |
| `Tier3Quality.*_mm` values | scalars | `float` | — | **millimetres** |
| `mesh_faces` | `(13776, 3)` | `int64` | — | — |

**Unit boundary:** everything inside `pointcloud/`, `surface_losses.py` and `surface.py` works in
**metres**. The conversion to **millimetres** happens exactly once, at the `ChamferReport` /
`Tier3Quality` boundary, and every millimetre-valued key ends in `_mm`. No function returns a
mixed-unit dict.

**The defining identity (D4 / 7.B3)** — assert it, don't assume it:

```
D  ==  forward(β,θ,t,s).vertices  −  forward(β,θ,t,s, apply_displacements=False).vertices
```

Open3D interop note: `o3d.t.geometry.RaycastingScene` requires `float32` vertices and `uint32`
triangle indices. Convert at the call boundary inside `surface_metrics.py`; do **not** let
`float32`/`uint32` leak into the module's public signatures, which are `float64`/`int64`.

## Determinism

Tier 3's production path introduces **no stochastic step** (master D12). Each item below is either
deterministic by construction or seeded:

| Step | Stochastic? | Seed source |
|---|---|---|
| Statistical outlier removal (Open3D) | no — k-NN statistics | — |
| Voxel downsample (Open3D) | no — grid is a pure function of extents | — |
| Normal estimation (Open3D, k-NN PCA) | no | — |
| PCA triad + 24-rotation enumeration | no — fixed enumeration order | — |
| Point-to-plane ICP (Open3D) | no — deterministic given init | — |
| RANSAC-FPFH global registration | **not used** (D12) | — |
| `torch.cdist` chamfer + Adam | no | — |
| `tessellation_floor` surface sampling | yes | `Tier3Config.tessellation_floor_seed` (default 0) |
| Synthetic fixture generation | yes | hardcoded seed 0 in `make_fixture.py` |

`test_alignment_deterministic` (AC7) asserts bitwise-identical alignment across two runs. If Open3D
turns out to introduce nondeterminism in a threaded kernel, that is a **finding to report**, not
something to paper over with a tolerance — the whole point of avoiding RANSAC-FPFH was to keep this
tier reproducible.

## Verification

Skills the specialists run per step, in order:

- **`py-lint`** — after any code change. Line length 100, ruff `E,F,I,W`.
- **`py-typecheck`** — after any change to `scantosmpl/types.py`, `scantosmpl/config.py`, or any
  module defining a public dataclass (that is: all five briefs).
- **`py-test`** — after each behaviour change:
  - `pytest tests/test_pointcloud.py -v` — `pointcloud-package`
  - `pytest tests/test_surface_metrics.py -v` — `surface-metrics`
  - `pytest tests/test_surface_losses.py -v` — `smpld-and-losses`
  - `pytest tests/test_surface_fitting.py -v` — `surface-fitting`
  - `pytest tests/test_artefacts.py -v` — `tier3-pipeline-artefacts`
  - `pytest tests/ -x --tb=short` — full suite, at the end of every brief (guards R5)
  - `pytest tests/integration/test_tier3_integration.py -v --timeout=900` — `tier3-pipeline-artefacts`
  - `pytest tests/integration/test_phase5_integration.py -v -m gpu --timeout=600` — **required** at
    the end of `smpld-and-losses`, because it is the only brief touching `smpl/model.py` (R5).
- **`pipeline-smoke`** — at the end of `tier3-pipeline-artefacts` only. The existing `mini/` and
  `with-cloud/` fixtures the skill describes do **not** exist; the Tier 3 equivalent is:

  ```bash
  /home/dan/.pyenv/versions/smpl_psd_venv/bin/scantosmpl fit-surface \
      --tier2-dir output/debug/refinement/ \
      --pointcloud tests/integration/fixtures/synthetic_cloud/cloud.ply \
      --subject smoke --pose-name t-pose \
      --output output/_smoke/tier3/
  ```

  Assert: exit 0; `output/_smoke/tier3/t-pose/{smpl_params.npz,displacements.npz,registered.obj,alignment.json,quality.json}`
  all present; `output/_smoke/tier3/manifest.json` present; `quality.json`'s
  `chamfer_cloud_to_mesh_mean_mm` recorded in the verification note alongside its threshold.
  The other briefs do **not** run `pipeline-smoke` — they cannot affect end-to-end behaviour until
  the pipeline exists, and should say so explicitly in their `BUILD_RESULT` notes.

### Environment notes for the specialists

- **The authoritative interpreter is `/home/dan/.pyenv/versions/smpl_psd_venv/bin/python`.**
  Python 3.11.14, torch 2.12.1+cu130, open3d 0.19.0, trimesh 4.11.0, numpy 2.2.6, scipy 1.16.3,
  smplx 0.1.28, pytest 9.1.1, mypy 2.1.0, ruff 0.15.20. `kaolin` and `rtree` are confirmed absent.
  - There is **also** a `.venv/` at the repo root on **torch 2.11.0+cu130** — a *different*
    environment, not a symlink. An earlier draft of this note assumed they were the same
    interpreter; they are not. Use only the pyenv path above, matching
    `.claude/loop-engineering/agents/python-engineer.md` and the `/feature-loop` precondition.
    Do not run, test, lint or typecheck against `.venv/`, and do not report metrics measured there.
  - **Never create a virtualenv and never `pip install`.** No `python -m venv`, `uv venv`,
    `virtualenv`, `conda create`, and no `pip install -e .` (that last one rebinds the shared
    editable-install pointer and breaks every concurrent worktree). A package that is genuinely
    missing from the pyenv venv is a `status: "blocked"` BUILD_RESULT, not an install.
  - Invoke Python tooling as modules so a worktree's own source wins on `sys.path`:
    `cd <worktree-root> && /home/dan/.pyenv/versions/smpl_psd_venv/bin/python -m pytest ...`,
    `... -m mypy scantosmpl`.
- **Do not `pip install` anything.** AC22 forbids new dependencies. In particular: Kaolin has no
  wheel for this torch (master D2), and `trimesh.proximity` needs `rtree`, which is absent — if you
  find yourself reaching for either, you have taken a wrong turn.
- GPU is an RTX 3080 Ti Laptop, 16 GiB. The chamfer loss at 6890 × 50 000 chunked at 10 000 peaks
  at ~1.9 GiB — if you see materially more, chunking is broken.
- SMPL model files live in `models/smpl/`; tests needing them use the existing
  `@pytest.mark.requires_smpl` marker. GPU tests use `@pytest.mark.gpu`.

## Definition of done

- All 24 acceptance criteria in master §10 pass on the merged tree, **except AC9's real-cloud arm**,
  which is a documented deferred gate (skips with a reason naming the missing
  `data/t-pose/pointcloud.ply`; its synthetic bound must still pass).
- `py-lint`, `py-typecheck`, `py-test` all green; `pytest tests/ -x --tb=short` exits 0.
- `pytest tests/integration/test_phase5_integration.py -v -m gpu` still passes — Tier 2 is
  unregressed (R5).
- `pipeline-smoke` on `tests/integration/fixtures/synthetic_cloud/` exits 0 with all seven expected
  artefacts.
- `git diff pyproject.toml` is empty (AC22).
- The Tier 3 gate is reported as **PASS** only on real scanner data; on the synthetic fixture it is
  reported as **DEFERRED**. A loop that reports PASS from synthetic data alone has failed its
  definition of done.
- No P0 or P1 findings from the loop's `reviewer` remain open.
