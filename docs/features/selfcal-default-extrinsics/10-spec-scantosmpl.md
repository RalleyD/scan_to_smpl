# Repo Spec — Self-Calibration as Default Phase 5 Extrinsics Source (ScanToSMPL)

Applies the master spec to this repo's conventions and lists the exact skills the loop runs.

## Subpackages touched

- `scantosmpl/calibration/` — delete `colmap_reader.py`, `frame_alignment.py`, `undistort.py`; trim `__init__.py` exports.
- `scantosmpl/config.py` — `Phase5Config` loses 7 fields (see master §5.1).
- `scantosmpl/fitting/pipeline.py` — `Phase5Pipeline` loses the COLMAP branch and EXIF-landscape bridging; `Phase5Result` loses 2 fields (see master §5.2).
- `tests/test_triangulation.py` — remove `TestUndistortion`, `TestFrameAlignment`, `TestColmapReader` classes and their COLMAP-only imports/fixtures; keep `TestDLT`, `TestRANSAC`.
- `tests/integration/test_phase5_integration.py` — replace COLMAP-based fixtures with the self-cal fixture pattern (Phase 4 `CalibrationPipeline` → `CalibrationResult`, mirroring the now-deleted experiment file).
- `tests/integration/test_selfcal_phase5_experiment.py` — delete (superseded by the above).

No new fixture directories — reuses `data/t-pose/jpg/`, `output/debug/detection/detections.json`, `output/debug/consensus/consensus_results.json` already on disk.

## Coordinate frames + units

| Tensor | Shape | Dtype | Frame | Units |
|--------|-------|-------|-------|-------|
| `view.camera.rotation`/`translation` (post-Phase-4) | `(3,3)`/`(3,)` | `float64` | SMPL/world → camera | dimensionless / metres |
| `view.camera.K` (property, post-fix) | `(3,3)` | `float64` | image | pixels |
| `view.keypoints_2d` | `(17,2)` | `float32`/`float64` | image, EXIF-upright (as displayed) | pixels |
| `Phase5Pipeline._load_cameras(...)` output | `dict[str, (R,t,K)]` | `float64` | SMPL/world | metres / pixels |
| Triangulated joints, refined joints | `(J,3)` / `(24,3)` | `float64` | SMPL/world | metres |

No transform between "COLMAP landscape" and "detection-space" coordinates exists anywhere in this feature's scope — that entire bridge is deleted (master D4). Any code that still references `_view_orient`, `_W_colmap`/`_H_colmap`, or landscape/detection-space conversion after this feature lands is a bug.

## Determinism

- `cv2.solvePnPRansac` inside `CalibrationPipeline.calibrate()` (Phase 4, upstream of this feature, not modified here) has no fixed seed in the current codebase. This feature's AC7 threshold carries ~10% slack over the experiment's measured numbers specifically to absorb this. If the fix-cycle sees repeated AC7 failures from run-to-run RANSAC variance, adding `cv2.setRNGSeed(0)` inside `calibrate()` is an acceptable in-scope fix (touches `scantosmpl/calibration/pipeline.py`, not owned by any brief below — flag to the user before touching, since it's outside the three briefs' declared `Owns` blocks).
- No other stochastic steps are introduced by this feature. Triangulation RANSAC (`ransac_triangulate_joints`) already exists and is unmodified.

## Verification

Skills the specialist(s) run per step (in order):

- `py-lint` — after any code change.
- `py-typecheck` — after any change to `scantosmpl/config.py` or `scantosmpl/fitting/pipeline.py` (both touch dataclass/function-signature contracts).
- `py-test` — after each behaviour change:
  - `pytest tests/test_triangulation.py -v` for the `retire-colmap-modules` brief.
  - `pytest tests/integration/test_phase5_integration.py -v -m gpu --timeout=600` for the `phase5-selfcal-pipeline` and `phase5-integration-tests` briefs.
- `pipeline-smoke` — at the end of the `phase5-integration-tests` brief (the last one to land, since it exercises the full rewritten pipeline end-to-end).

## Definition of done

- All acceptance criteria in master §10 pass on the merged tree.
- `py-lint`, `py-typecheck`, `py-test` all green.
- `pipeline-smoke` on `tests/integration/fixtures/mini/` (or the equivalent fixture this repo's Phase 5 integration tests actually use — `data/t-pose/jpg/`) exits 0.
- `grep -rn "colmap\|COLMAP" scantosmpl/` returns nothing outside historical comments (AC1).
- No P0 or P1 findings from the loop's `reviewer` remain open.
