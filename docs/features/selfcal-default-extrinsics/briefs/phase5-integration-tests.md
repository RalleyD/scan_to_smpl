---
component: phase5-integration-tests
agent: python-engineer
worktree: false
---

# Component Brief — phase5-integration-tests

**Serialisation**: This brief must run *after* `phase5-selfcal-pipeline` has merged — it depends on the final `Phase5Config`/`Phase5Result`/`Phase5Pipeline.run()` contract from that brief. It should also run after (or concurrently with, since files don't overlap) `retire-colmap-modules`, since it must not construct fixtures that reference the deleted COLMAP modules.

## Goal

Replace `tests/integration/test_phase5_integration.py`'s COLMAP-based fixtures with the self-calibration fixture pattern already validated by `tests/integration/test_selfcal_phase5_experiment.py` (Phase 4 `CalibrationPipeline` → `CalibrationResult`, feeding `Phase5Pipeline.run(..., calibration_result=...)`). Delete the experiment file once its logic is folded in. Update assertions to the thresholds in master §10 AC7.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `tests/integration/test_phase5_integration.py`
- `tests/integration/test_selfcal_phase5_experiment.py` — delete

**Does NOT touch**:
- `scantosmpl/**` — no source changes in this brief; if the pipeline's contract doesn't match what this brief needs, that's a finding to report in `BUILD_RESULT`, not something to patch around here.
- `tests/test_triangulation.py`, `scantosmpl/calibration/__init__.py` — `retire-colmap-modules` brief's territory.
- Other test files under `tests/integration/` (`test_pnp_integration.py`, `test_consensus_integration.py`, `test_detection_integration.py`, `test_hmr_integration.py`) — unrelated, do not touch, though this brief's fixtures are structurally modelled on `test_pnp_integration.py`'s `hmr_views`/`calibration_result` pattern.

**Consumes** (from other components / the shared contract):
- `Phase5Config`, `Phase5Result`, `Phase5Pipeline.run(views, consensus, image_dir, calibration_result)` — final contract from `phase5-selfcal-pipeline` (master §5.1-§5.3).
- `scantosmpl.calibration.pipeline.CalibrationPipeline`/`CalibrationResult` — existing Phase 4, unchanged.
- `scantosmpl.hmr.pipeline.HMRPipeline` — existing Phase 2, unchanged.
- `scantosmpl.hmr.consensus.ConsensusBuilder` — existing Phase 3, unchanged.

**Produces** (available to other components):
- The updated `tests/integration/test_phase5_integration.py` becomes the sole Phase 5 integration test file — no other component depends on its internals, but it's the thing `pipeline-smoke` and the loop's final `py-test` pass exercise.

## Steps

1. **Replace fixtures** — swap the current `detection_views` (reconstructed minimal `ViewResult` from `output/debug/detection/detections.json`, no dense keypoints, no camera extrinsics) and `phase5_result` (COLMAP-configured) fixtures for the pattern in `tests/integration/test_selfcal_phase5_experiment.py`:
   - `detection_views` — as currently written (loads `detections.json` into `ViewResult` objects with `bbox`/`keypoints_2d`/`keypoint_confs`/`camera` — a `CameraParams` with only `focal_length` set).
   - `hmr_views(detection_views)` — run `scantosmpl.hmr.pipeline.HMRPipeline` to add SMPL params + 138 dense keypoints per view (mirrors `test_selfcal_phase5_experiment.py::hmr_views`).
   - `consensus_result(hmr_views)` — build via `scantosmpl.hmr.consensus.ConsensusBuilder` (mirrors the experiment file). This *replaces* the current fixture that reconstructs `ConsensusResult` from `output/debug/consensus/consensus_results.json` + a fresh SMPL forward pass — check whether reusing the cached JSON (faster, no HMR re-run) vs rebuilding from `hmr_views` (guarantees internal consistency with the `hmr_views` object graph that Phase 4 will mutate) is the right call. **Prefer rebuilding from `hmr_views`** — Phase 4's `calibrate()` mutates `view.camera` in place on the *same* `ViewResult` objects `consensus_result` was built from, and those objects are what gets passed to `Phase5Pipeline.run()`. Reusing the cached JSON path would still work for `consensus_result` values themselves, but keeping one object graph avoids a subtle divergence if `hmr_views` is ever regenerated with different inputs than the cached JSON. **Verify**: `pytest tests/integration/test_phase5_integration.py -v -m gpu -k "not slow"` at least collects without error at this point (full run is later steps).
   - `calibration_result(hmr_views, consensus_result)` — run `scantosmpl.calibration.pipeline.CalibrationPipeline.calibrate(hmr_views, consensus_result, DATA_DIR, debug_dir=...)`. Mutates `hmr_views[i].camera` in place.
   - `phase5_result(hmr_views, consensus_result, calibration_result)` — construct `Phase5Pipeline` with the new `Phase5Config` (master §5.1 — no `colmap_model_dir`/`extrinsics_source` kwargs) and call `.run(views=hmr_views, consensus=consensus_result, image_dir=DATA_DIR, calibration_result=calibration_result)`.
   - Remove `COLMAP_DIR`, `OUR_17`, `_phase5_cfg(colmap_model_dir=...)`, and the `require_colmap` parameter of `_skip_if_missing` (self-cal has no COLMAP dependency to skip on).
2. **Update/remove COLMAP-specific test classes**:
   - `TestCOLMAPReader::test_all_17_views_have_colmap_extrinsics` — delete (tests `read_colmap_model`/`match_views_to_colmap`, both deleted modules).
   - `TestFrameAlignment::test_frame_alignment_quality` — this test's *name* refers to Procrustes frame alignment, which no longer exists. Rename the class/test to reflect what it now checks (reprojection quality of self-cal cameras) — e.g. `TestReprojectionQuality::test_reprojection_quality`. Remove the `if alignment is None: pytest.skip(...)` guard entirely (there's no `Phase5Result.frame_alignment` field anymore — master §5.2). Keep the `median_reproj_px < 100.0` / `mean_reproj_inliers_px < 150.0` assertions, but see step 3 for the actual threshold values to use.
3. **Apply master §10 AC7 thresholds** — replace the existing `< 100.0` / `< 150.0` / `< 24.0` / `< 35.0` assertions (which were tuned for the abandoned PnP-refinement feature, not for self-calibration) with:
   - Reprojection-quality test: `median_reproj_px < 90.0`.
   - `TestTriangulation::test_triangulation_accuracy`: refinement-side `pa_mpjpe_mm < 24.5`. Re-examine whether the *triangulation-vs-consensus* assertion (currently `< 35.0`) should tighten too — the experiment measured 23.99mm on the *refinement* side; if the triangulation-only PA-MPJPE also improved proportionally in your own test run, tighten this to match what you actually observe rather than leaving stale numbers. Report the actual figures in `BUILD_RESULT` either way.
   - `TestReprojectionError::test_reprojection_error`: `median_reproj_px < 90.0` (same threshold as the reprojection-quality test — both read `phase5_result.metrics["median_reproj_px"]`).
4. **Remove PnP-refinement-specific tests** — the current file (at HEAD, post-revert) does not contain a `TestPnPRefinement` class (that was the abandoned feature, already reverted — confirm this via `grep -n "TestPnPRefinement\|pnp_refine\|cameras_pre_pnp" tests/integration/test_phase5_integration.py` returning nothing before you start; if it returns matches, stop and report in `BUILD_RESULT` rather than guessing why the revert didn't take).
5. **Verify rear-view exclusion still exercised** — `KNOWN_REAR_VIEWS` should stay; confirm at least one existing assertion (or add one if none remain relevant) still checks that rear cameras don't corrupt the reprojection metric — e.g. via `phase5_result.metrics["n_outlier_views"]` being reasonably small, or by checking `SMPLOptimiser`'s rear-view exclusion indirectly through the overall PA-MPJPE/reprojection numbers landing in the expected range. Do not invent a new debug-JSON contract for this (`refinement_results.json` doesn't carry a `pnp_refinement`/`cameras_pre_pnp` block anymore — that was the abandoned feature); a metrics-level assertion is sufficient.
6. **Delete the experiment file** — `rm tests/integration/test_selfcal_phase5_experiment.py`. Confirm no other test imports from it first (`grep -rn "test_selfcal_phase5_experiment" tests/`).
7. **Full integration run** — **Verify**: `pytest tests/integration/test_phase5_integration.py -v -m gpu --timeout=600` — all tests pass, none skipped for a missing `COLMAP_DIR` (there should be no such skip condition left).
8. **Lint** — **Verify**: `py-lint` on `tests/integration/test_phase5_integration.py`.
9. **Smoke** — **Verify**: `pipeline-smoke` (or the closest equivalent this repo defines) exercising the same `data/t-pose/jpg/` fixture end-to-end.

## Definition of done

- Master §10 AC6, AC7, AC8, AC9 pass.
- `tests/integration/test_selfcal_phase5_experiment.py` no longer exists.
- No test in `tests/integration/test_phase5_integration.py` references `COLMAP_DIR`, `colmap_model_dir`, `extrinsics_source`, or `frame_alignment`.
- `notes` in the returned `BUILD_RESULT` reports the actual measured `pa_mpjpe_mm`/`median_reproj_px` from your own test run (not just "assertions pass") — the master spec's AC7 thresholds were set with 10% slack over the original experiment's numbers specifically so this brief's real run can confirm or tighten them.
