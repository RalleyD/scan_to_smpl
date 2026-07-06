---
component: phase5-selfcal-pipeline
agent: python-engineer
worktree: true
---

# Component Brief — phase5-selfcal-pipeline

## Goal

Remove the COLMAP extrinsics path from `Phase5Config` and `Phase5Pipeline`, making Phase 4 self-calibration the only extrinsics source. Delete the EXIF-landscape bridging that existed solely to reconcile COLMAP's raw-sensor orientation with PIL's EXIF-corrected orientation. Simplify `Phase5Result` accordingly.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/config.py` — `Phase5Config` only. Do not touch `ModelPaths`, `DetectionConfig`, `HMRConfig`, `ConsensusConfig`, `CalibrationConfig`, `FittingConfig`, `PipelineConfig`.
- `scantosmpl/fitting/pipeline.py`

**Does NOT touch**:
- `scantosmpl/calibration/colmap_reader.py`, `frame_alignment.py`, `undistort.py` — being deleted by the sibling `retire-colmap-modules` brief independently. Do not import from them in your final state (they will not exist once both briefs merge); do not wait for that brief to land first — the master spec's D3 decision is already locked, so you can write `pipeline.py` as if they're already gone.
- `scantosmpl/fitting/optimiser.py`, `scantosmpl/fitting/rear_views.py`, `scantosmpl/fitting/losses.py` — already correct, rear-view exclusion stays as-is, no changes needed.
- `tests/integration/*` — sibling `phase5-integration-tests` brief's territory (which depends on this brief's final contract — serialises after this one).
- `scantosmpl/calibration/pipeline.py` (Phase 4) — already correct (principal-point fix already merged), not touched by this feature.
- `external/`, model weights, `output/`.

**Consumes** (from other components / the shared contract):
- `scantosmpl.calibration.pipeline.CalibrationResult` — existing Phase 4 dataclass, unchanged.
- `scantosmpl.hmr.consensus.ConsensusResult` — existing, unchanged.
- `scantosmpl.fitting.optimiser.SMPLOptimiser`, `DEFAULT_STAGES`, `RefinementResult` — existing, unchanged.
- `scantosmpl.types.ViewResult` — existing; `view.camera.has_extrinsics`/`.rotation`/`.translation`/`.K` are the contract this brief reads (already correct post the Phase-4 principal-point fix).

**Produces** (available to other components):
- `Phase5Config` matching master §5.1 exactly (8 fields).
- `Phase5Result` matching master §5.2 exactly.
- `Phase5Pipeline.run(views, consensus, image_dir, calibration_result)` matching master §5.3 (calibration_result now required).
- `Phase5Pipeline._load_cameras(views, calibration_result)` — renamed from `_load_selfcal_cameras`, now the only camera-loading method.

## Steps

1. **Simplify `Phase5Config`** in `scantosmpl/config.py` — remove `extrinsics_source`, `colmap_model_dir`, `min_alignment_joints`, `pnp_refine_cameras`, `pnp_refine_max_translation_m`, `pnp_refine_min_inliers`, `pnp_refine_iterate_optimiser`. Result matches master §5.1 exactly: `triangulation_conf_threshold`, `triangulation_min_views`, `ransac_reproj_threshold`, `ransac_iterations`, `reprojection_mad_multiplier`, `save_debug`, `debug_dir`.
   - **Verify**: `py-typecheck` on `scantosmpl/config.py`.
2. **Remove the COLMAP branch from `Phase5Pipeline.run()`** — replace the `if cfg.extrinsics_source == "colmap": ... else: ...` block (Step 1 in the current `run()`) with a single unconditional call: `cameras_smpl = self._load_cameras(views, calibration_result)`. `calibration_result` is now a required parameter (not `| None`).
3. **Delete dead methods** — `_load_colmap_cameras`, `_get_exif_orient`, `_kps_to_landscape`, `_all_to_landscape`, `_landscape_to_det`, `_undistort_keypoints`, and the `_view_orient`/`_W_colmap`/`_H_colmap` instance attributes set in `__init__`. Rename `_load_selfcal_cameras` → `_load_cameras`, keeping its body unchanged (it already reads `view.camera.rotation/translation/K` correctly — that's the whole point of the pivot).
4. **Update `_gather_keypoints` call sites** — since there's no more landscape transform, `kp2d_per_view`/`confs_per_view` from `_gather_keypoints(views)` are used directly wherever the old code called `_undistort_keypoints` afterward. Remove that call; keypoints flow straight from Phase 1/2 into triangulation and the optimiser, matching exactly what Phase 4 used to solve PnP (this consistency is *why* the pivot works — see master §1).
5. **Simplify `Phase5Result`** — remove `frame_alignment: FrameAlignment | None` and `extrinsics_source: str` fields. Remove the (now-unused) `FrameAlignment` import. Update the `run()` return statement and any place that constructs `Phase5Result(...)` accordingly.
   - **Verify**: `py-typecheck` on `scantosmpl/fitting/pipeline.py`.
6. **Simplify debug output** — in `_save_debug`/`_save_reprojection_overlays`:
   - Remove the `frame_alignment` key from the `refinement_results.json` dict (was `alignment.scale/rotation/translation if alignment is not None else None` — always `None` now, so just drop the key entirely rather than writing a permanent `null`).
   - Remove the `alignment` parameter threaded through `_save_debug` and its call site.
   - In `_save_reprojection_overlays`, remove the `orient`/`_landscape_to_det` calls — project `refined.joints` directly via `project_points(..., R, t, K)` and draw at those pixel coordinates on the `ImageOps.exif_transpose`-opened image directly (no forward/inverse landscape transform needed — the image is already in the same frame the keypoints and cameras are in).
   - Remove the `DEFAULT_ORIENTATION_OVERRIDES` import/usage in the overlay function if it was only used for the landscape bridge; check `scantosmpl/detection/image_loader.py` first — if `DEFAULT_ORIENTATION_OVERRIDES` is still needed for correct EXIF-transpose display (i.e. it's a Phase-1-level correction independent of COLMAP), keep using it for the image-opening step only, not for keypoint coordinate transforms.
   - **Verify**: `pytest tests/test_triangulation.py -v` (should still pass — no COLMAP-only tests remain after the sibling brief's cleanup, but this brief must not break it prematurely if it lands first; if the sibling hasn't merged yet, this file will still import from the soon-to-be-deleted modules only in the OLD pipeline.py code you're removing, not in your rewritten version, so no coupling exists).
7. **Full lint + typecheck** — **Verify**: `py-lint` and `py-typecheck` on both changed files.

## Definition of done

- `Phase5Config` and `Phase5Result` match master §5.1/§5.2 exactly (verify via `dataclasses.fields(...)`).
- `Phase5Pipeline.run()` signature matches master §5.3; calling it without `calibration_result` raises `TypeError`.
- No references to COLMAP, landscape transforms, or EXIF orientation-bridging remain in `scantosmpl/fitting/pipeline.py`.
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new skill — in particular, flag if `DEFAULT_ORIENTATION_OVERRIDES` turned out to still be needed somewhere non-obvious, since the sibling `phase5-integration-tests` brief will be exercising this code end-to-end next.
