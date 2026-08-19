# Handoff — `tier3-surface-refinement`, Phase 2 resume

Supersedes the previous version of this file (Phase 1 resume, 2026-08-16). Phase 1 is now
**fully collected** (5/5 `BUILD_RESULT`s). Phase 2 (Integrate → Review → Fix) has **not started**.
`iter = 0` of `maxIterations = 3`. Paused 2026-08-19 at the user's request to free up the session
for another task — not a code, spec, or environment failure.

Read alongside [LOOP-STATE.md](LOOP-STATE.md).

---

## 0. Orchestrator deviations in force — carry these forward, they are not in `feature-loop.md`

### 0.1 `/simplify` added to Phase 2e (this session, by user request)

After each iteration's Fix fan-out (2e) returns, the orchestrator invokes the `/simplify` skill
itself (not a subagent — it runs in the orchestrator's own context), scoped to just the files that
iteration's fixes touched, **before** looping back to Integrate (2a). It is non-blocking: it does
not gate convergence and does not generate P0/P1 findings of its own. Any edits it makes flow into
the next Integrate call's verification like any other change. If an iteration converges with no
Fix phase, `/simplify` is skipped for that iteration.

### 0.2 Venv policy — unchanged conclusion, permissions widened

`.claude/settings.json` now has bare `"Bash"`, `"Source"`, `source .venv/bin/activate`,
`.venv/bin/*`, and `python -m venv` all on the **allow** list (widened 2026-08-18/19). This looks
like it might reverse the earlier "repo-root `.venv/` is off-limits" rule — it does not. The user
was asked explicitly and **confirmed: keep `/home/dan/.pyenv/versions/smpl_psd_venv/bin/python`
(torch 2.12.1+cu130) as the only authoritative venv.** The permission widening only means agents
can use `source .../activate` or bare commands after activating, instead of always spelling out
the full interpreter path — it does not change *which* venv. The repo-root `.venv/` (torch 2.11.0)
remains non-authoritative for this feature; do not run, test, or measure against it. State this
explicitly in every subagent prompt — it is instruction-enforced, not config-enforced (the deny
list still doesn't cover bare `pip install`, only path-qualified spellings).

### 0.3 Registered subagent types — unchanged from before

`.claude/agents/{python-engineer,integration-engineer,reviewer}.md` are real `subagent_type`
values. Do not paste role-file text into prompts. Do not pass `model` — frontmatter sets it.

---

## 1. Tree state

All Phase 1 work is committed on `main` at **`8f314dd`**. Note: this single commit bundles **all
five** components' Phase 1 deliverables, not just `tier3-pipeline-artefacts` — its own message
explains why ("Also includes the (separately-authored, already-merged-to-disk) sibling
deliverables this component wires together, staged here so the tree is coherent for anyone
resuming"). No worktrees remain from this session; everything is in the main tree. `git status`
is clean with respect to `scantosmpl/` and `tests/` — only unrelated pre-existing uncommitted
files remain (`REVIEW.md`, `notes.md`, `external/CameraHMR` submodule pointer — none touched by
this feature, left alone).

`import scantosmpl` verified OK as of the last agent run in this session.

---

## 2. Phase 1 results — five `BUILD_RESULT` blocks, verbatim

Feed all five to the Integrate agent's prompt per `feature-loop.md` §2a.

### 2.1 `pointcloud-package` — `status: "done"`

```json
{
  "component": "pointcloud-package",
  "status": "done",
  "filesChanged": [
    "scantosmpl/pointcloud/__init__.py",
    "scantosmpl/pointcloud/segment.py",
    "scantosmpl/pointcloud/align.py",
    "tests/test_pointcloud.py"
  ],
  "contractsProduced": [
    {"kind": "dataclass", "name": "PointCloud", "shape": "points(N,3)f64, normals(N,3)f64|None, colors(N,3)f32|None, source_path:Path, frame:'source'|'smpl_world', units:'arbitrary'|'metres'"},
    {"kind": "function", "name": "load_pointcloud(path, *, max_points=None) -> PointCloud", "shape": "PLY/OBJ loader, frame='source', units='arbitrary'"},
    {"kind": "function", "name": "save_pointcloud(cloud, path) -> None", "shape": "binary PLY writer"},
    {"kind": "dataclass", "name": "PreprocessStats", "shape": "n_input, n_after_outlier_removal, n_output, outlier_fraction, voxel_size_source_units, bbox_diagonal_source_units, normals_estimated"},
    {"kind": "function", "name": "preprocess_cloud(cloud, cfg) -> tuple[PointCloud, PreprocessStats]", "shape": "outlier removal -> unit-free voxel downsample -> optional normals"},
    {"kind": "dataclass", "name": "CloudAlignment", "shape": "scale, rotation(3,3), translation(3,), inlier_rmse_m, fitness, n_candidates, candidate_index, converged; .apply(pts), .as_matrix()"},
    {"kind": "function", "name": "pca_triad(points) -> (centroid, axes(3,3), extents(3,))", "shape": "proper, sign-canonical PCA triad"},
    {"kind": "function", "name": "enumerate_proper_rotations(src_axes, dst_axes) -> list[np.ndarray]", "shape": "24 proper signed-permutation rotations, fixed order"},
    {"kind": "function", "name": "align_cloud_to_smpl(cloud, mesh_vertices, mesh_faces, cfg) -> tuple[PointCloud, CloudAlignment]", "shape": "PCA init -> 24-candidate ICP -> best by inlier RMSE"},
    {"kind": "dataclass-const", "name": "SMPL_PART_GROUPS", "shape": "dict[str, list[int]], 6 groups partitioning 24 SMPL joints"},
    {"kind": "function", "name": "smpl_part_labels(lbs_weights) -> np.ndarray", "shape": "(6890,24) -> (6890,) int group id via argmax"},
    {"kind": "function", "name": "vertex_part_weights(lbs_weights, weights) -> np.ndarray", "shape": "(6890,) float32 per-vertex weight"},
    {"kind": "function", "name": "transfer_labels_to_cloud(cloud_points, mesh_vertices, vertex_labels) -> np.ndarray", "shape": "nearest-vertex label transfer via cKDTree"}
  ],
  "verification": [
    {"skill": "py-typecheck", "passed": true},
    {"skill": "py-test", "passed": true, "note": "39/39 in test_pointcloud.py; 194 passed unit-suite-wide"},
    {"skill": "py-lint", "passed": true},
    {"skill": "pipeline-smoke", "passed": false, "note": "not run — no CLI entry point exists yet, out of this brief's scope"}
  ],
  "notes": "SUPERSEDED BY §3.1 BELOW: tier3-pipeline-artefacts's integration testing found align_cloud_to_smpl scale-collapses under combined noise+outliers, a scenario this component's own AC5 test did not cover (it used a clean, noise-free fixture). Original notes also flagged: AC7's 'bitwise-identical' wording doesn't literally hold under Open3D's threaded ICP (~1e-15 relative, non-RNG, root-caused to OpenMP reduction order); test adjusted to machine-precision tolerance instead."
}
```

### 2.2 `surface-metrics` — `status: "done"`, no changes needed

```json
{
  "component": "surface-metrics",
  "status": "done",
  "filesChanged": [],
  "contractsProduced": [
    {"kind": "dataclass", "name": "ChamferReport", "shape": "cloud_to_mesh_mm/mesh_to_cloud_mm: dict[mean,median,rms,p95,max]; tessellation_floor_mm: dict[mean,max]; n_cloud_points, n_mesh_vertices; cloud_to_mesh_method/mesh_to_cloud_method; units='mm'; to_dict(). No fused field."},
    {"kind": "function", "name": "point_to_surface_distances(points, vertices, faces) -> np.ndarray", "shape": "(N,) f64 metres, unsigned point-to-triangle via o3d.t.geometry.RaycastingScene (7.M1)"},
    {"kind": "function", "name": "vertex_to_point_distances(vertices, points) -> np.ndarray", "shape": "(V,) f64 metres, cKDTree (7.M2)"},
    {"kind": "function", "name": "tessellation_floor(vertices, faces, *, n_samples=100_000, seed=0) -> dict[str,float]", "shape": "{'mean','max'} mm (7.M5)"},
    {"kind": "function", "name": "chamfer_report(cloud_points, vertices, faces, cfg=None) -> ChamferReport", "shape": "assembles both directions + floor"},
    {"kind": "function", "name": "sample_surface(vertices, faces, *, n_samples, seed) -> np.ndarray", "shape": "(n_samples,3) area-weighted"}
  ],
  "verification": [
    {"skill": "py-lint", "passed": true},
    {"skill": "py-typecheck", "passed": true},
    {"skill": "py-test", "passed": true, "note": "20/20 in test_surface_metrics.py, incl. AC1 analytic and AC4 tessellation-floor-bound"},
    {"skill": "pipeline-smoke", "passed": false, "note": "reporting-only module, no CLI wiring, out of scope"}
  ],
  "notes": "No code changes needed; the partial build already satisfied 7.M1-7.M6 exactly. Measured SMPL's REAL tessellation floor (SMPL_NEUTRAL.pkl, 6890v/13776f, seed=0, 100k samples): mean 8.46mm, max 27.64mm — substantially worse than the idealised 5.77mm equilateral-triangle bound, because SMPL's real tessellation is non-uniform. This eats directly into the 8mm chamfer AC's headroom; belongs in Review's AC assessment and tier3-pipeline-artefacts's report-generation context."
}
```

### 2.3 `smpld-and-losses` — `status: "done"`

```json
{
  "component": "smpld-and-losses",
  "status": "done",
  "filesChanged": [
    "scantosmpl/smpl/model.py",
    "scantosmpl/fitting/surface_losses.py",
    "tests/test_surface_losses.py"
  ],
  "contractsProduced": [
    {"kind": "dataclass", "name": "SMPLModel.displacements", "shape": "nn.Parameter (1, 6890, 3), posed-world metres, zero-initialised"},
    {"kind": "function", "name": "SMPLModel.forward(..., displacements=None, apply_displacements=True)", "shape": "adds D after scale; joints never displaced; D=0 default bitwise-identical to pre-change forward()"},
    {"kind": "function", "name": "chamfer_loss(vertices, cloud, *, vertex_weights=None, cloud_weights=None, chunk_size=10_000, huber_delta=0.02, trim_quantile=0.95) -> (Tensor, dict[str, float])", "shape": "bidirectional, single chunked torch.cdist, Huber+quantile-trimmed, detached float diagnostics {mesh_to_cloud_m, cloud_to_mesh_m}"},
    {"kind": "function", "name": "normal_consistency_loss(vertices, faces, cloud, cloud_normals, *, chunk_size=10_000) -> Tensor", "shape": "1 - |cos|, sign-agnostic"},
    {"kind": "function", "name": "build_uniform_laplacian(faces, n_verts) -> torch.Tensor", "shape": "sparse (V,V) COO, symmetric, zero row sums, cached by face-hash"},
    {"kind": "function", "name": "laplacian_smoothing_loss(displacements, laplacian) -> Tensor", "shape": "mean(||L@D||^2)"},
    {"kind": "function", "name": "displacement_regularisation(displacements) -> Tensor", "shape": "mean(||D||^2)"}
  ],
  "verification": [
    {"skill": "py-lint", "passed": true},
    {"skill": "py-typecheck", "passed": true, "note": "0 errors in owned files; fixed 4 pre-existing mypy errors as a side effect"},
    {"skill": "py-test", "passed": true, "note": "test_smpl_model.py 9/9+1 skip (unmodified); test_surface_losses.py 30/30; full suite 273 passed/3 pre-existing failed/1 skipped"},
    {"skill": "py-test", "passed": true, "note": "tests/integration/test_phase5_integration.py -m gpu: 11/11 passed, PA-MPJPE=22.8mm — Tier 2 unregressed"}
  ],
  "notes": "Chamfer re-measured on torch 2.12.1 (was 2.11.0 in the brief): chunk_size=10_000 (default) -> 1.878 GiB / 77.5 ms/iter, within noise of the brief's quoted 1.9 GiB/74ms. AC13's 60s budget recomputed at 77.5ms x 550 iters ~= 42.6s — still comfortably under 60s, no revision needed. FLAG FOR REVIEW: normal_consistency_loss diverges under gradient descent whenever cloud normals are present (confirmed independently by surface-fitting's build — see §3.2 below)."
}
```

### 2.4 `surface-fitting` — `status: "done"`

```json
{
  "component": "surface-fitting",
  "status": "done",
  "filesChanged": [
    "scantosmpl/fitting/surface.py",
    "tests/test_surface_fitting.py"
  ],
  "contractsProduced": [
    {"kind": "dataclass", "name": "SurfaceStage", "shape": "name, params: list[str], n_iterations, w_chamfer=1.0, w_normal=0.0, w_laplacian=0.0, w_displacement_reg=0.0, w_pose_prior=0.0, w_shape_reg=0.0, learning_rate=1e-2"},
    {"kind": "dataclass", "name": "SurfaceFitResult", "shape": "betas, body_pose, global_orient, translation, scale, displacements(6890,3) f32, vertices, base_vertices, betas_locked, loss_history, metrics"},
    {"kind": "const", "name": "DEFAULT_SURFACE_STAGES", "shape": "model_fit(300,5e-3) then displacement(250,1e-3); 'scale' absent from every stage (D6)"},
    {"kind": "function", "name": "Tier3SurfaceFitter.fit(tier2, cloud, *, stages=None, locked_betas=None) -> SurfaceFitResult"},
    {"kind": "function", "name": "count_self_intersecting_faces(vertices, faces) -> int", "shape": "AC12 helper"}
  ],
  "verification": [
    {"skill": "py-typecheck", "passed": true},
    {"skill": "py-test", "passed": true, "note": "16/16 in test_surface_fitting.py; full suite 289 passed/3 pre-existing failed/1 skipped"},
    {"skill": "py-lint", "passed": true},
    {"skill": "pipeline-smoke", "passed": false, "note": "no CLI entry point yet — tier3-pipeline-artefacts's territory"}
  ],
  "notes": "surface.py/test file were already essentially complete on arrival; two pre-existing test bugs fixed, two real upstream defects diagnosed and reported (not silently patched): (1) normal_consistency_loss diverges under gradient descent w/ cloud normals present at literal default w_normal=0.1 — self-intersections 117->5328, displacement up to 9.7cm in a realistic scenario. (2) DEFAULT_SURFACE_STAGES' locked w_pose_prior=0.01/w_shape_reg=0.01 (copied from Tier2's pixel-scale schedule, master Sec5.3-locked) can dominate Tier3's metres-scale chamfer loss near a good fit -> ~21deg pose drift exceeding AC12's 15deg bound in a degenerate near-exact-match case, but NOT in realistic non-degenerate scenarios (~9.2deg there). (3) Informational: shared _CONVERGENCE_TOL=1e-7 (mirrored from Tier2) causes S2 to halt at 12-70 of its configured 300 iterations for Tier3's much smaller loss scale — not yet judged a defect, flagged for reviewer judgement. AC13 measured wall-clock (S2+S3, 50K points, GPU): 41.9s, inside the 60s gate."
}
```

### 2.5 `tier3-pipeline-artefacts` — `status: "blocked"`

```json
{
  "component": "tier3-pipeline-artefacts",
  "status": "blocked",
  "filesChanged": [
    "scantosmpl/types.py",
    "scantosmpl/config.py",
    "scantosmpl/fitting/artefacts.py",
    "scantosmpl/fitting/surface_pipeline.py",
    "scantosmpl/fitting/__init__.py",
    "scantosmpl/cli.py",
    "tests/test_artefacts.py",
    "tests/integration/test_tier3_integration.py",
    "tests/integration/fixtures/synthetic_cloud/make_fixture.py",
    "tests/integration/fixtures/synthetic_cloud/cloud.ply",
    "tests/integration/fixtures/synthetic_cloud/ground_truth.json"
  ],
  "contractsProduced": [
    {"kind": "dataclass", "name": "Tier3Quality, PoseArtefact (scantosmpl/types.py)"},
    {"kind": "dataclass", "name": "Tier3Config (scantosmpl/config.py)", "shape": "master Sec5.2 verbatim; PipelineConfig.tier3 added"},
    {"kind": "dataclass", "name": "Tier3Result (scantosmpl/fitting/surface_pipeline.py)", "shape": "fit, alignment, report, preprocess, quality, artefact_dir"},
    {"kind": "function", "name": "Tier3Pipeline.__init__(smpl_model, cfg) / .run(tier2, pointcloud_path, *, pose_name, output_dir) -> Tier3Result"},
    {"kind": "function", "name": "write_pose_artefacts, update_manifest, load_locked_betas (scantosmpl/fitting/artefacts.py)", "shape": "the 7.B1-7.B8 enforcement layer, all six write-time assertions implemented with matching failing-path tests"},
    {"kind": "cli-command", "name": "scantosmpl fit-surface", "shape": "--lock-betas without --betas-from exits non-zero; --betas-from implies --lock-betas"}
  ],
  "verification": [
    {"skill": "py-lint", "passed": true, "note": "clean except NOT re-verified on test_tier3_integration.py's last edit — outstanding"},
    {"skill": "py-typecheck", "passed": true, "note": "38 pre-existing errors, unchanged, 0 new"},
    {"skill": "py-test", "passed": true, "note": "test_artefacts.py 24/24, incl. failing-path tests for every 7.B1-7.B8 assertion"},
    {"skill": "py-test", "passed": false, "note": "test_tier3_integration.py: 7/10 passed. 3 FAILED: test_alignment_recovers_ground_truth, test_beta_refinement_improves_proportions, test_similarity_invariance — see notes"},
    {"skill": "py-test", "passed": true, "note": "full suite: 319 passed, 2 skipped, 6 failed (3 pre-existing + the 3 above)"},
    {"skill": "pipeline-smoke", "passed": false, "note": "not run — blocked on the alignment defect first"}
  ],
  "notes": "BLOCKING ROOT CAUSE (see §3.1 below for the full diagnosis): pointcloud/align.py's align_cloud_to_smpl converges to a scale-collapsed ICP solution (fitness=1.0, deceptively low RMSE, scale wrong 5-40x) whenever the input cloud has BOTH the master-mandated 1mm noise AND 2% outliers present together (D11's realistic scenario). Confirmed via direct debugging, reproduces at all target_points settings, persists with an asymmetric reference pose (rules out pure mirror ambiguity). pointcloud-package's own AC5 test didn't catch this because it used a clean noise-free fixture — this integration fixture is the first to combine known-similarity recovery with the required noise+outlier scenario. Per this brief's Owns boundary, align.py was NOT edited; reported here as a cross-component finding. Secondary outstanding items once unblocked: re-verify ruff/format on the last test_tier3_integration.py edit; run pipeline-smoke; capture summary.txt's real numbers from a passing fixture run; validate AC18's 0.5mm invariance tolerance is achievable (currently failing for the alignment reason, not because 0.5mm is too tight); re-verify AC10/AC13 after any fix (both passed once, not independently re-verified since). pyproject.toml untouched (AC22 preserved), no new dependency added."
}
```

---

## 3. Cross-cutting findings — prioritize these in Integrate/Review

These are synthesized from the five builds' own self-reported diagnostics above; Integrate and
Review should verify and formalize them into the finding schema, not take them on faith.

1. **[Likely P0/P1, blocking]** `scantosmpl/pointcloud/align.py::align_cloud_to_smpl` — ICP
   candidate selection scale-collapses under combined noise+outliers (master D11's realistic
   scenario). The selection criterion (lowest inlier RMSE among `fitness > 0` candidates) is
   gameable by a scale-collapsed correspondence, since Open3D's fitness/RMSE score only the found
   correspondences, not correspondence density/coverage. Blocks 3 of `tier3-pipeline-artefacts`'s
   integration tests, and likely AC5/AC18. Owning component for a fix: `pointcloud-package`.
2. **[P1/P2]** `scantosmpl/fitting/surface_losses.py::normal_consistency_loss` diverges under
   gradient descent whenever cloud normals are present (self-intersections balloon at the literal
   default `w_normal=0.1`). Flagged independently by both `smpld-and-losses`'s own suite and
   `surface-fitting`'s build.
3. **[P2, needs Dan/reviewer judgement — may not be component-fixable]**
   `DEFAULT_SURFACE_STAGES`'s `w_pose_prior=0.01`/`w_shape_reg=0.01` — copied verbatim from
   Tier 2's pixel-scale schedule — can dominate Tier 3's metres-scale chamfer loss near a good
   fit, exceeding AC12's 15° bound in near-exact-match cases (not yet observed with realistic
   Tier-2 residuals, where it's ~9.2°). These are master §5.3-**locked** values, so a real fix may
   require touching the master spec itself, which is outside any single component's Owns
   boundary — flag for Dan if Review concludes a spec change is needed rather than a code fix.
4. **[Informational]** Shared `_CONVERGENCE_TOL=1e-7` (mirrored from Tier 2's optimiser) causes
   Tier 3's `model_fit` stage to halt at 12–70 of its configured 300 iterations, since Tier 3's
   chamfer-loss scale is orders of magnitude smaller than Tier 2's reprojection-pixel scale. Not
   yet judged a defect — flag for reviewer judgement on whether early convergence at a much lower
   loss floor is actually fine, or masks under-fitting.
5. **[Informational, for Review's AC assessment]** SMPL's *real* tessellation floor measures mean
   8.46mm / max 27.64mm (seed 0, 100k samples) — worse than the idealised 5.77mm equilateral bound
   — eating directly into the 8mm chamfer AC's headroom.
6. **[Blocking, tracked in §2.5]** `tier3-pipeline-artefacts` itself is incomplete pending #1.

---

## 4. Resume: Phase 2a Integrate — the next call to make

```
subagent_type: integration-engineer
description:   integrate:tier3-surface-refinement
isolation:     (omit)
```

Prompt = all five `BUILD_RESULT` blocks from §2 above, verbatim, plus this tail (do not paste the
role file):

> Slug is `tier3-surface-refinement`; specs are at `docs/features/tier3-surface-refinement/`.
> All work was done in the main tree at commit `8f314dd` — there are no worktrees to merge, so
> your duty 3 is a no-op. Confirm and move on. Duties 1, 2, 4, 5, 6, 7 all still apply.
>
> One component, `tier3-pipeline-artefacts`, returned `status: "blocked"` with a detailed
> root-cause diagnosis of a defect in a DIFFERENT component's file
> (`scantosmpl/pointcloud/align.py`, owned by `pointcloud-package`, which itself returned
> `status: "done"`). Re-verify this diagnosis independently — re-run
> `tests/integration/test_tier3_integration.py -v` and confirm the three named failures and their
> stated cause — and raise it as a P0 or P1 finding tagged to whichever component should own the
> fix (`pointcloud-package`, since `align.py` is its file), not just note it as `tier3-pipeline-artefacts`
> being unfinished. See `docs/features/tier3-surface-refinement/HANDOFF.md` §3 for the full list of
> cross-cutting findings surfaced during Phase 1 — verify each, don't take them on faith.
>
> Interpreter is `/home/dan/.pyenv/versions/smpl_psd_venv/bin/python`, invoked via `-m` for
> pytest and mypy (or `source .../activate` first, now permitted — same venv either way). Never
> `pip install`; never use the repo-root `.venv/` (torch 2.11.0, non-authoritative for this
> feature per explicit user confirmation).
>
> Return only the `INTEGRATION` JSON block.

---

## 5. Phase 2b Review, 2c/d Aggregate, 2e Fix — unchanged from `feature-loop.md` §2b–2e

With one addition: **after each iteration's Fix fan-out (2e) returns**, the orchestrator invokes
`/simplify` (itself, not a subagent) scoped to that iteration's touched files, before looping back
to 2a. See §0.1 above.

`maxIterations = 3`, `iter = 0`, unchanged from the original plan.

Delete this file, and `LOOP-STATE.md`, once the loop converges.
