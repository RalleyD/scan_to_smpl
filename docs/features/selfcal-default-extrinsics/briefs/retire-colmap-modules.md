---
component: retire-colmap-modules
agent: python-engineer
worktree: true
---

# Component Brief — retire-colmap-modules

## Goal

Delete the three COLMAP-only modules (`colmap_reader.py`, `frame_alignment.py`, `undistort.py`) and everything in `scantosmpl/calibration/__init__.py` and `tests/test_triangulation.py` that references them. Confirmed zero consumers remain outside the Phase 5 COLMAP path (which the sibling `phase5-selfcal-pipeline` brief removes independently) — this brief does not touch `scantosmpl/fitting/pipeline.py` at all; it only needs the master spec's decision (D3) that these modules are being retired, not a live dependency on the sibling brief's diff.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/calibration/colmap_reader.py` — delete
- `scantosmpl/calibration/frame_alignment.py` — delete
- `scantosmpl/calibration/undistort.py` — delete
- `scantosmpl/calibration/__init__.py` — edit (remove dead imports/exports)
- `tests/test_triangulation.py` — edit (remove COLMAP-only test classes)

**Does NOT touch**:
- `scantosmpl/fitting/pipeline.py`, `scantosmpl/config.py` — sibling `phase5-selfcal-pipeline` brief's territory. Do not attempt to remove that file's imports of the modules you're deleting here; the sibling brief handles its own side of this independently, per the master spec's already-locked decision.
- `scantosmpl/calibration/pipeline.py`, `scantosmpl/calibration/correspondence.py`, `scantosmpl/calibration/intrinsics.py`, `scantosmpl/calibration/pnp_solver.py` — Phase 4 modules, unrelated, already correct.
- `tests/integration/*` — sibling `phase5-integration-tests` brief's territory.
- `external/`, model weights, `output/`.

**Consumes** (from other components / the shared contract):
- Nothing new — this brief only removes code.

**Produces** (available to other components):
- A `scantosmpl/calibration/` package with no COLMAP-reading, frame-alignment, or undistortion code — the sibling `phase5-selfcal-pipeline` brief must not import from these modules in its own final state (master §5.1/§5.3 already reflect this).

## Steps

1. **Delete the three modules** — `rm scantosmpl/calibration/colmap_reader.py scantosmpl/calibration/frame_alignment.py scantosmpl/calibration/undistort.py`.
   - **Verify**: `py-lint` will fail on any remaining reference — that's expected until step 2 completes; don't treat it as a blocker mid-step.
2. **Clean `scantosmpl/calibration/__init__.py`** — remove the import lines and `__all__` entries for `ColmapCamera`, `ColmapImage`, `match_views_to_colmap`, `read_colmap_model`, `FrameAlignment`, `compute_frame_alignment`, `build_pinhole_K`, `undistort_keypoints`. Keep everything else (`CorrespondenceBuilder`, `build_intrinsic_matrix`, `get_intrinsics_for_view`, `CalibrationPipeline`, `CalibrationResult`, `PnPResult`, `PnPSolver` all stay — Phase 4 modules, untouched).
   - **Verify**: `py-typecheck` on `scantosmpl/calibration/__init__.py`.
3. **Trim `tests/test_triangulation.py`** — remove:
   - The import block for `ColmapCamera`, `match_views_to_colmap`, `read_colmap_model`, `FrameAlignment`, `compute_frame_alignment`, `build_pinhole_K`, `undistort_keypoints` (lines 7-13 in the current file).
   - The `TestUndistortion` class (tests `undistort_keypoints`/`build_pinhole_K` against a synthetic `ColmapCamera`).
   - The `TestFrameAlignment` class (tests `compute_frame_alignment`/`FrameAlignment` — note this is a *different* `TestFrameAlignment` from the one in `tests/integration/test_phase5_integration.py`; only this unit-test one is in scope here).
   - The `COLMAP_DIR`/`OUR_17` module-level constants, the `colmap_model` fixture, and the `TestColmapReader` class.
   - Keep `TestDLT` and `TestRANSAC` (and the shared `_make_camera` helper) untouched — these test `scantosmpl.triangulation.dlt`/`scantosmpl.triangulation.ransac` directly and have nothing to do with COLMAP.
   - Update the module docstring (currently `"""Unit tests for Phase 5 triangulation, undistortion, and frame alignment."""`) to drop the now-inaccurate "undistortion, and frame alignment" — e.g. `"""Unit tests for Phase 5 triangulation (DLT + RANSAC)."""`.
   - **Verify**: `pytest tests/test_triangulation.py -v` — remaining tests (`TestDLT`, `TestRANSAC`) pass.
4. **Full lint pass** — **Verify**: `py-lint` on `scantosmpl/calibration/__init__.py` and `tests/test_triangulation.py`.

## Definition of done

- Every step's verification skill is green.
- `scantosmpl/calibration/colmap_reader.py`, `frame_alignment.py`, `undistort.py` do not exist.
- `grep -rn "colmap_reader\|frame_alignment\|ColmapCamera\|ColmapImage\|FrameAlignment\|undistort_keypoints\|build_pinhole_K" scantosmpl/calibration/__init__.py tests/test_triangulation.py` returns nothing.
- `TestDLT` and `TestRANSAC` in `tests/test_triangulation.py` still pass unmodified in behaviour.
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new skill.
