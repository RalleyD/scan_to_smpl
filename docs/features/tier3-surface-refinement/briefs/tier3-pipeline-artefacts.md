---
component: tier3-pipeline-artefacts
agent: python-engineer
worktree: false   # serialises LAST — depends on all four sibling briefs
---

# Component Brief — tier3-pipeline-artefacts

## Goal

Wire the four delivered pieces into a runnable tier and — more importantly — implement the **PSD
boundary contract**. Requirements 7.B1–7.B8 are not suggestions: each one, if violated, silently
corrupts the downstream PSD residual rather than failing loudly. This brief's job is to make every
one of them fail *loudly*, at write time, with an assertion. It also owns all shared-file edits
(`types.py`, `config.py`, `cli.py`, `fitting/__init__.py`) precisely so the parallel briefs never
collide on them.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/fitting/surface_pipeline.py`
- `scantosmpl/fitting/artefacts.py`
- `scantosmpl/fitting/__init__.py`   (export wiring for ALL new fitting symbols)
- `scantosmpl/types.py`
- `scantosmpl/config.py`
- `scantosmpl/cli.py`
- `tests/test_artefacts.py`
- `tests/integration/test_tier3_integration.py`
- `tests/integration/fixtures/synthetic_cloud/`  (generator + generated fixture)

**Does NOT touch**:
- `scantosmpl/pointcloud/*`, `scantosmpl/evaluation/surface_metrics.py`,
  `scantosmpl/smpl/model.py`, `scantosmpl/fitting/surface_losses.py`,
  `scantosmpl/fitting/surface.py` — all four siblings' deliverables, consumed as-is. If one is
  wrong, report it in `BUILD_RESULT`; do not edit it.
- `scantosmpl/fitting/{losses,optimiser,pipeline,rear_views}.py` — Tier 2, byte-unchanged. In
  `config.py` you may **add** `Tier3Config` and `PipelineConfig.tier3`; you may **not** modify
  `Phase5Config`, `FittingConfig`, `CalibrationConfig`, `ConsensusConfig`, `HMRConfig`,
  `DetectionConfig` or `ModelPaths`.
- In `types.py` you may **add** `Tier3Quality`, `PoseArtefact`, the three constants and one
  defaulted field on `FittingResult`; you may **not** change the shape of any existing field.
- `external/`, `models/`, `pyproject.toml` (AC22), `data/`.

**Consumes**: every sibling's `Produces` (master §5.1/§5.3), plus `RefinementResult` /
`Phase5Result` from the existing Tier 2 modules.

**Produces**:
- `Tier3Quality`, `PoseArtefact`, `DISPLACEMENT_FRAME`, `SMPL_NUM_VERTICES`, `SMPL_NUM_FACES`,
  `FittingResult.displacement_frame` — master §5.1
- `Tier3Config`, `PipelineConfig.tier3` — master §5.2
- `Tier3Result`, `Tier3Pipeline` — master §5.3
- `write_pose_artefacts`, `update_manifest`, `load_locked_betas` — master §5.3
- `scantosmpl fit-surface` — master §5.4

## Steps

1. **`types.py` + `config.py`.** Add exactly what master §5.1/§5.2 specify, no more. Every addition
   is additive with a default so existing construction sites keep working.
   **Verify**: `py-typecheck`, then `pytest tests/ -x --tb=short` (nothing existing may break).

2. **`artefacts.py` — the 7.B enforcement layer.** This is the highest-value module in the feature;
   write the assertions before the happy path.
   - `write_pose_artefacts` writes the five files of master §7.1 and asserts, per master §5.3:
     `D.shape == (6890,3)` and `dtype == float32` (7.B2/7.B4); `vertices`/`faces` shapes (7.B4);
     `faces` **byte-identical** to the SMPL template's face array, compared by sha256 so a
     reordering cannot slip through (7.B4 — blend-shape targets are index-aligned, any permutation
     silently scrambles every target); no NaN/Inf anywhere; `displacement_frame == "posed_world"`
     written as an explicit field (7.B3); and `allclose(base_vertices + D, vertices)` (D4).
   - `D` is written **unconditionally** — there is no flag that suppresses it (7.B2 supersedes
     REVIEW.md 7.6's "if enabled").
   - `alignment.json` holds the cloud→SMPL similarity and **nothing about it is folded into `D`**
     (7.B5). Keep them in separate files so the separation is visible on disk, not just in code.
   - `update_manifest` creates-or-updates `manifest.json` per master §7.2, raising if an existing
     manifest disagrees on `subject_id`, `displacement_frame`, gender, `num_betas` or
     `faces_sha256`. `oracle_only` defaults `false` and must survive later updates (7.B8).
   **Verify**: `py-test` — `pytest tests/test_artefacts.py -v`. This file discharges AC15, AC17,
   AC19, AC20, AC21 and must test each assertion's **failure** path, not just the happy path: a
   resampled vertex count, a permuted face array, a NaN in `D`, a second subject writing into the
   same manifest, and a `base + D != vertices` inconsistency must each raise.

3. **Synthetic fixture generator.** `tests/integration/fixtures/synthetic_cloud/make_fixture.py`,
   seeded at 0, building `cloud.ply` + `ground_truth.json` exactly per master §7.3: 60 000
   area-weighted surface samples, +4 mm outward normal offset on `torso`-labelled vertices
   (`D_true`), σ = 1 mm normal noise, 2 % uniform bbox outliers, then the known similarity
   (`scale = 0.371`, ≈137° about a non-axis unit vector, translation `(1.7, -0.4, 2.3)`).
   `ground_truth.json` records the inverse similarity, `D_true` stats, σ and the outlier fraction.
   Commit both the generator and its output so the fixture is reproducible **and** available without
   SMPL weights.
   **Verify**: `py-test` — regenerating the fixture twice produces byte-identical output.

4. **`surface_pipeline.py` — orchestration.** `Tier3Pipeline.run` = load → preprocess → align (S1) →
   `Tier3SurfaceFitter.fit` (S2, S3) → `chamfer_report` → build `Tier3Quality` → persist →
   `update_manifest`. Also write `summary.txt` into `cfg.debug_dir` following the Phase 5 precedent
   in `scantosmpl/fitting/pipeline.py::_save_debug`, and mirroring its hard-won reporting lesson:
   - Print **both** chamfer directions separately with their aggregation named (7.M3/7.M4) — never a
     fused number. Phase 5's `summary.txt` originally printed only a skewed mean and that cost real
     debugging time; do not repeat it.
   - Print the tessellation floor beside the result (7.M5) so the headline number is interpretable.
   - Print the Tier 3 gate as `PASS` **only** on real scanner data. On the synthetic fixture print
     `TIER 3 GATE: DEFERRED (no real point cloud)`. Reporting PASS from synthetic data is a
     definition-of-done failure (repo spec).
   - Print the AC8 before/after comparison: the `D = 0`, Tier-2-params baseline chamfer alongside the
     final, so the ≥40 % improvement is legible without re-running anything.
   **Verify**: `py-typecheck`, then `py-test` — `pytest tests/integration/test_tier3_integration.py -v --timeout=900`.

5. **CLI `fit-surface` + `fitting/__init__.py` exports.** Implement master §5.4's option surface.
   `--lock-betas` without `--betas-from` exits non-zero with a clear message; `--betas-from` implies
   `--lock-betas` (AC14). Extend `fitting/__init__.py` with all new symbols from **all** briefs,
   following the existing docstring / explicit-import / `__all__` pattern.
   **Verify**: `py-lint`, `py-typecheck`.

6. **Integration tests.** `tests/integration/test_tier3_integration.py` discharges AC5, AC6, AC8,
   AC9, AC10, AC11, AC13 and AC18:
   - `test_alignment_recovers_ground_truth` (AC5), `test_preprocess_removes_outliers` (AC6),
     `test_refinement_improves_over_tier2` (AC8), `test_semantic_weighting_ab` (AC10),
     `test_optimisation_under_60s` (AC13, `@pytest.mark.gpu`).
   - `test_beta_refinement_improves_proportions` (AC11) — **`lock_betas=False` only**. Shoulder
     width (joints 16↔17) and waist girth must move toward the fixture's ground-truth mesh relative
     to the Tier 2 input; write both deltas to `summary.txt`. This test must **not** run in the
     locked-β mode — AC11 is inapplicable there by construction, which is exactly REVIEW.md's
     7.4-vs-7.B1 resolution (master D10). Assert the inapplicability explicitly rather than
     skipping silently.
   - `test_similarity_invariance` (AC18) — **the decisive 7.B5 check**: re-run with the cloud
     pre-multiplied by a *different* known similarity; `D` must match to within 0.5 mm mean while
     `alignment.json` absorbs the difference. If this fails, a transform has been baked into the
     displacement field and the PSD residual is corrupt.
   - `test_real_cloud_chamfer` (AC9) — `pytest.mark.skipif` on `data/t-pose/pointcloud.ply` missing,
     with the skip reason naming that exact path. The synthetic bound
     (`cloud_to_mesh_mean_mm < 3.0`) is asserted unconditionally.
   **Verify**: `py-test` — `pytest tests/integration/test_tier3_integration.py -v --timeout=900`.

7. **Final wire-up.** **Verify**: `py-lint`, `py-typecheck`, `pytest tests/ -x --tb=short`,
   `pytest tests/integration/test_phase5_integration.py -v -m gpu --timeout=600`, and
   `pipeline-smoke` per the repo spec's `fit-surface` invocation on
   `tests/integration/fixtures/synthetic_cloud/`.

## Definition of done

- Every step's verification skill is green, including `pipeline-smoke`.
- **All 24** master §10 acceptance criteria are discharged across the five briefs, with AC9's
  real-cloud arm documented as skipped-and-deferred rather than passed.
- Every 7.B requirement has a **failing-path** test, not just a happy-path one.
- `git diff pyproject.toml` is empty (AC22).
- `summary.txt` never prints a fused chamfer number and never prints `PASS` on synthetic data.
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new
  skill — in particular: (a) the measured `cloud_to_mesh` / `mesh_to_cloud` / tessellation-floor
  numbers from the fixture run, (b) whether AC18's 0.5 mm invariance tolerance held or needed
  widening (widening it is a **spec correction to escalate**, not a quiet edit — it is the only
  check that catches a baked-in transform), and (c) confirmation that `data/t-pose/pointcloud.ply`
  is still absent so the Tier 3 gate status is reported honestly.
