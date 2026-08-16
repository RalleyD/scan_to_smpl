---
component: surface-fitting
agent: python-engineer
worktree: false   # serialises AFTER pointcloud-package, surface-metrics and smpld-and-losses
---

# Component Brief — surface-fitting

## Goal

Implement the staged Tier 3 optimiser: given a Tier 2 `RefinementResult` and a cloud already in the
SMPL/world frame, run **S2** (fit SMPL parameters to the surface with `D ≡ 0`) then **S3** (freeze
every SMPL parameter and solve `D` alone). The staging is the whole point — it is what keeps `D`
interpretable as genuine off-manifold geometry rather than a dumping ground for pose, shape and
alignment error, which is what PSD boundary requirement 7.B5 demands.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/fitting/surface.py`
- `tests/test_surface_fitting.py`

**Does NOT touch**:
- `scantosmpl/smpl/model.py` and `scantosmpl/fitting/surface_losses.py` — delivered by
  `smpld-and-losses`, consumed as-is. If you find them wrong, report it in `BUILD_RESULT`; do not
  edit them.
- `scantosmpl/pointcloud/*` — delivered by `pointcloud-package`, consumed as-is.
- `scantosmpl/evaluation/surface_metrics.py` — reporting only; the fitter does **not** call it
  (7.M6: loss ≠ metric, and mixing them would make the gate self-referential).
- `scantosmpl/fitting/{losses,optimiser,pipeline,rear_views}.py` — Tier 2, byte-unchanged.
- `scantosmpl/fitting/__init__.py`, `scantosmpl/config.py`, `scantosmpl/types.py`,
  `scantosmpl/cli.py` — all `tier3-pipeline-artefacts`' territory. Take `Tier3Config` as a typed
  parameter; import it under `TYPE_CHECKING` if it does not exist yet, and write against the locked
  master §5.2 field names.
- Other components' `Owns` paths; `external/`, `models/`, `pyproject.toml`, `output/`

**Consumes**:
- `PointCloud` from `scantosmpl.pointcloud.io`; `vertex_part_weights`, `transfer_labels_to_cloud`
  from `scantosmpl.pointcloud.segment` — `pointcloud-package`'s `Produces`.
- `chamfer_loss`, `normal_consistency_loss`, `build_uniform_laplacian`,
  `laplacian_smoothing_loss`, `displacement_regularisation` — `smpld-and-losses`' `Produces`.
- `SMPLModel` with the `displacements` parameter — `smpld-and-losses`' `Produces`.
- `pose_prior_loss`, `shape_regularisation` from `scantosmpl.fitting.losses` — existing Tier 2
  losses, reused unchanged (do not reimplement them).
- `RefinementResult` from `scantosmpl.fitting.optimiser` — existing, unchanged.

**Produces**:
- `SurfaceStage`, `SurfaceFitResult`, `DEFAULT_SURFACE_STAGES`, `Tier3SurfaceFitter` — master §5.1 / §5.3

## Steps

1. **`SurfaceStage` + `DEFAULT_SURFACE_STAGES`.** Mirror the established
   `scantosmpl/fitting/optimiser.py::OptimisationStage` / `DEFAULT_STAGES` pattern — same shape of
   dataclass, same "list of stages, Adam per stage, early stop on loss delta" loop. Reuse that
   file's structure rather than inventing a new optimiser idiom.
   - `DEFAULT_SURFACE_STAGES` exactly as master §5.3: `model_fit` (300 iters, lr 5e-3) then
     `displacement` (250 iters, lr 1e-3).
   - **`"scale"` must appear in no stage's `params`** (master D6 — the ICP alignment owns metric
     scale; letting S2 move it would make CLAUDE.md's "SMPL has correct metric scale" premise
     false). Add a module-level assertion or a test that enforces this.
   **Verify**: `py-typecheck` on `scantosmpl/fitting/surface.py`.

2. **`Tier3SurfaceFitter.fit` — guards first.** Before any optimisation:
   - `assert cloud.frame == "smpl_world"` and `cloud.units == "metres"`, raising a `ValueError` with
     an actionable message. A source-frame cloud is the single most damaging silent failure in this
     tier — it will converge, produce a plausible `D`, and be entirely wrong.
   - When `cfg.lock_betas`: require `locked_betas` to be supplied, set them on the model, and
     **remove `"betas"` from every stage's `params` list** — do not merely zero its weight. AC14
     asserts exact equality (`np.array_equal`), which a zero-weighted-but-trainable parameter would
     fail under Adam's epsilon.
   **Verify**: `py-test` — `pytest tests/test_surface_fitting.py -k guard -v`, covering both the
   frame assertion and the `lock_betas`-without-betas error.

3. **S2 — model fit.** Initialise the SMPL model from the Tier 2 `RefinementResult`, zero the
   displacements, and optimise the stage's parameters against
   `w_chamfer * chamfer_loss + w_pose_prior * pose_prior_loss + w_shape_reg * shape_regularisation`.
   Semantic weights come from `vertex_part_weights(lbs_weights, cfg.body_part_weights)` and
   `transfer_labels_to_cloud(...)`; when `cfg.use_semantic_weighting` is `False`, pass `None` for
   both (this is the AC10 A/B switch, so it must be a genuine bypass, not weights-of-1.0 with the
   same code path — though weights-of-1.0 is an acceptable implementation if the test can still
   distinguish the two runs).
   **Verify**: `py-test` — `pytest tests/test_surface_fitting.py -k model_fit -v`. Include a
   synthetic case: perturb a known SMPL mesh's `β`, sample its surface as the cloud, and assert S2
   recovers `β` toward ground truth (chamfer strictly decreases and final `β` error < initial).

4. **S3 — displacement fit.** Freeze every SMPL parameter (`requires_grad_(False)` on all but
   `displacements`), then optimise `D` against
   `w_chamfer * chamfer + w_normal * normal_consistency + w_laplacian * laplacian + w_displacement_reg * ||D||²`.
   Populate `SurfaceFitResult` including `base_vertices` (the `apply_displacements=False` forward)
   and `vertices` (with `D`), so the master's `D` identity is directly checkable from the result.
   Carry `scale` through unchanged from Tier 2.
   **Verify**: `py-test` — `pytest tests/test_surface_fitting.py -k displacement -v`. Known-answer
   test: take a mesh, push a contiguous patch of vertices outward by a known 4 mm, sample **that**
   as the cloud, and assert S3 recovers ≈4 mm on the patch and ≈0 elsewhere. Also assert
   `allclose(result.base_vertices + result.displacements, result.vertices)` (AC16's in-memory arm)
   and that no SMPL parameter changed during S3.

5. **Pose plausibility + self-intersection check (AC12).** Add
   `test_pose_plausible_no_new_intersections`: per-joint axis-angle change from the Tier 2 input
   `< 15°`, and the self-intersecting-face count increases by no more than 5. Implement the
   intersection count as a small helper in this module (a broad-phase AABB pass over the 13776
   faces plus a narrow-phase triangle-triangle test is sufficient — no new dependency).
   **Verify**: `py-test` — `pytest tests/test_surface_fitting.py -v`.

6. **Full lint, typecheck, suite.**
   **Verify**: `py-lint`, `py-typecheck`, `pytest tests/ -x --tb=short`.

## Definition of done

- Every step's verification skill is green.
- AC12, AC14 and the in-memory arm of AC16 are each discharged by a named test here.
- `"scale"` provably appears in no stage; `"betas"` provably absent when locked.
- The `Produces` contract exactly matches master §5.1/§5.3.
- The fitter never imports `surface_metrics` (7.M6 separation).
- No `pyproject.toml` change (AC22).
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new
  skill — in particular report the **measured** S2+S3 wall-clock at 50 K points on GPU, since AC13
  gates at 60 s against an estimated 41 s budget, and flag immediately if the two default stages'
  550 total iterations prove insufficient for convergence (that is a spec correction, not something
  to silently raise). State explicitly that `pipeline-smoke` was not run and why.
