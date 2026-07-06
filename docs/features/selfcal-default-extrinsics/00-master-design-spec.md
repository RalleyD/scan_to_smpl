# Master Design Spec — Self-Calibration as Default Phase 5 Extrinsics Source

**Status**: Draft
**Slug**: `selfcal-default-extrinsics`
**Owner**: Dan
**Date**: 2026-07-05
**Tiers touched**: Tier 2 (self-calibration + refinement)

## 1. Problem

Phase 5 currently sources camera extrinsics from a COLMAP reconstruction, rigidly aligned into SMPL frame via 7-DoF Procrustes. This introduces a ~80px systematic reprojection drift because COLMAP's [R|t] is sub-pixel accurate for its own SIFT/SfM features but has no reason to agree with where ViTPose places body joints — a modality gap Procrustes can only correct in aggregate, not per-camera. A feature built to patch this drift (PnP camera refinement, "Option A+B") was implemented, reviewed against real data, and found to make things worse: only 3-4 of 7 frontal cameras refined, reprojection increased rather than decreased, and PA-MPJPE regressed to 25-27mm — the spec's own documented risk (cameras absorbing SMPL joint error) materialising in practice. That feature has been abandoned and its code reverted.

A controlled experiment (`tests/integration/test_selfcal_phase5_experiment.py`, run twice — once invalidated by an unrelated bug, once clean after the fix) showed Phase 4's existing self-calibration (cold `cv2.solvePnPRansac` directly against SMPL joints + ViTPose keypoints, no COLMAP involved) beats the COLMAP+Procrustes path outright: PA-MPJPE 23.99mm vs 24.46mm, median reprojection 78.8px vs 135.2px — landing at the ViTPose noise floor with no residual calibration drift. This spec retires the COLMAP path from Tier 2 and makes self-calibration the only extrinsics source Phase 5 uses.

## 2. Decisions

- **D1 (Slug)**: `selfcal-default-extrinsics` — names the end state rather than the migration mechanics.
- **D2 (COLMAP option)**: **Remove `extrinsics_source` and `colmap_model_dir` from `Phase5Config` entirely.** Self-calibration becomes the only path — not a default with a fallback. `Phase5Pipeline.run()` requires a `CalibrationResult` (from Phase 4) unconditionally.
- **D3 (Dead modules)**: **Delete outright**: `scantosmpl/calibration/colmap_reader.py`, `scantosmpl/calibration/frame_alignment.py`, `scantosmpl/calibration/undistort.py`, and their dedicated tests in `tests/test_triangulation.py`. Verified zero consumers exist outside the Phase 5 COLMAP path once that path is removed — `scantosmpl/pointcloud/` (Tier 3) is currently an empty package (`__init__.py` only), so there is no existing home to "move" this code into. See §9 for the design notes preserved for whoever revisits COLMAP/SIFT when Tier 3 is actually built.
- **D4 (EXIF/landscape bridging)**: **Delete** `Phase5Pipeline`'s COLMAP-landscape bridging: `_get_exif_orient`, `_kps_to_landscape`, `_all_to_landscape`, `_landscape_to_det`, `_view_orient`, `_W_colmap`/`_H_colmap`. These existed solely to reconcile COLMAP's raw-sensor-orientation convention with PIL's EXIF-corrected orientation. Phase 1 already applies EXIF transpose before detection (CLAUDE.md Stage 0), so `view.keypoints_2d` are already in the same upright frame the image displays in — once nothing needs to match COLMAP's landscape frame, this entire bridge is unreachable dead code.
- **D5 (Dead PnP-refinement config)**: **Remove** `pnp_refine_cameras`, `pnp_refine_max_translation_m`, `pnp_refine_min_inliers`, `pnp_refine_iterate_optimiser` from `Phase5Config` — these gated the now-abandoned PnP-refinement feature and have no purpose once COLMAP drift doesn't exist to correct.
- **D6 (Rear-view exclusion retained)**: `scantosmpl.fitting.rear_views.classify_rear_views` and its use in `SMPLOptimiser.refine()` (excluding rear-facing cameras from the reprojection loss) **stays** — ViTPose's left/right swap on rear views is a property of the 2D detector, independent of where `[R|t]` came from. Already extracted and merged; this spec does not touch it further except to keep signatures compatible with the simplified camera-loading path.
- **D7 (Frame-alignment fields retired)**: `Phase5Result.frame_alignment: FrameAlignment | None` and `Phase5Result.extrinsics_source: str` are **removed** from the dataclass — the first is always `None` post-pivot (nothing to report), the second can never vary. `Phase5Result.cameras_smpl_frame` stays (still meaningful — the per-view `[R|t|K]` used for triangulation/reprojection, just always self-cal-sourced now).
- **D8 (Config field `min_alignment_joints`)**: **Removed** — existed only to gate Procrustes alignment quality, meaningless without COLMAP.
- **D9 (Test consolidation)**: `tests/integration/test_selfcal_phase5_experiment.py` is **deleted**; its `hmr_views`/`consensus_result`/`calibration_result` fixture pattern (validated by the experiment that produced this pivot's own evidence) is folded directly into `tests/integration/test_phase5_integration.py`, replacing that file's COLMAP-only fixtures (`detection_views` reconstructed from `detections.json`, `_phase5_cfg(colmap_model_dir=...)`, `COLMAP_DIR` skip guards).
- **D10 (`Phase5Pipeline.run()` signature)**: `calibration_result: CalibrationResult` becomes a required positional/keyword parameter (was `| None = None`, conditionally required). `views` must be the same `hmr_views` list Phase 4 calibrated against (Phase 4 mutates `view.camera.rotation/translation/principal_point` in place — see `scantosmpl/calibration/pipeline.py::calibrate()`).

## 3. Scope

**In scope**
- Remove the COLMAP extrinsics path from `Phase5Pipeline` (`_load_colmap_cameras` and all its private helpers) and from `Phase5Config`.
- Delete `colmap_reader.py`, `frame_alignment.py`, `undistort.py` and their unit tests.
- Simplify `Phase5Pipeline._compute_metrics`, `_save_debug`, `_save_reprojection_overlays` to the self-cal-only, no-landscape-bridging path.
- Update `tests/integration/test_phase5_integration.py` to build cameras via Phase 4's `CalibrationPipeline` (mirroring the now-deleted experiment file), and delete `tests/integration/test_selfcal_phase5_experiment.py`.
- Preserve rear-view exclusion (D6) and the A2 metrics (`median_reproj_px`, `mean_reproj_inliers_px`, `n_outlier_views`) — these are extrinsics-source-agnostic and already correct.

**Out of scope**
- Any re-attempt at PnP camera refinement in any form — the pivot's entire point is that self-calibration doesn't have the drift that feature existed to correct.
- Building out Tier 3 (`scantosmpl/pointcloud/`) — remains an empty package. §9 documents design notes for if/when that work starts.
- Changing Phase 4 (`CalibrationPipeline`) behaviour beyond what's already merged (the principal-point fix). Phase 4 is upstream of this feature and already produces correct `CalibrationResult`s.
- Re-tuning the optimiser (`DEFAULT_STAGES` in `scantosmpl/fitting/optimiser.py`) — out of scope; the experiment showed the existing stage schedule already benefits from cleaner cameras with no changes needed.

## 4. Approach

`Phase5Pipeline.run()` drops its `if cfg.extrinsics_source == "colmap": ... else: ...` branch entirely and always calls what is currently `_load_selfcal_cameras` (to be renamed `_load_cameras`, since it's no longer one of two options). Since self-cal cameras are already in SMPL/world frame with no distortion model and no orientation mismatch, Step 2 (undistort) and the EXIF-landscape bridging in Step 1 disappear — keypoints are consumed directly as `view.keypoints_2d`/`view.keypoint_confs`, exactly as Phase 4 used them to solve PnP. Triangulation (Step 3), SMPL optimisation (Step 4, including rear-view exclusion), and metrics (Step 5) are otherwise unchanged — they were always extrinsics-source-agnostic. This slots into the same Tier 2 position in the `CLAUDE.md` architecture diagram; the diagram's "Step 2b: PnP extrinsic recovery" already describes exactly this path, it merely wasn't previously the only one wired up end-to-end in Phase 5.

## 5. Contract

**This is the seam between components.** Every dataclass field, function signature, and artefact schema below is authoritative — the loop's `integration-engineer` reconciles specialist outputs against this section.

### 5.1 Config (`scantosmpl/config.py::Phase5Config`)

```python
@dataclass
class Phase5Config:
    """Phase 5: Multi-view triangulation + SMPL refinement configuration."""

    # Triangulation
    triangulation_conf_threshold: float = 0.3
    triangulation_min_views: int = 3
    ransac_reproj_threshold: float = 100.0
    ransac_iterations: int = 100

    # Refinement
    reprojection_mad_multiplier: float = 3.0

    # Debug
    save_debug: bool = True
    debug_dir: Path = Path("output/debug/refinement")
```

Removed relative to current `main`: `extrinsics_source`, `colmap_model_dir`, `min_alignment_joints`, `pnp_refine_cameras`, `pnp_refine_max_translation_m`, `pnp_refine_min_inliers`, `pnp_refine_iterate_optimiser`.

### 5.2 `Phase5Result` (`scantosmpl/fitting/pipeline.py`)

```python
@dataclass
class Phase5Result:
    """Output from the Phase 5 pipeline."""

    refined: RefinementResult
    triangulated_joints: np.ndarray          # (J, 3) raw triangulated, ordered by joint_indices
    triangulated_joints_smpl: np.ndarray     # (24, 3) mapped to SMPL joint ordering (zeros for unmapped)
    triangulation_quality: np.ndarray        # (J,) inlier fraction
    triangulation_reproj_errors: np.ndarray  # (J,) mean reprojection error (px)
    cameras_smpl_frame: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # {name: (R, t, K)}, SMPL/world frame, metres
    metrics: dict[str, float] = field(default_factory=dict)
```

Removed relative to current `main`: `frame_alignment: FrameAlignment | None`, `extrinsics_source: str`.

### 5.3 Function signatures

**`scantosmpl/fitting/pipeline.py::Phase5Pipeline`**

```python
def run(
    self,
    views: list[ViewResult],           # Phase 4-calibrated views — view.camera.{rotation,translation,principal_point} already set
    consensus: ConsensusResult,
    image_dir: Path,
    calibration_result: CalibrationResult,   # now required, not optional
) -> Phase5Result:
    """Run the full Phase 5 pipeline: triangulation + SMPL refinement using
    Phase 4 self-calibration extrinsics. Raises RuntimeError if fewer than
    triangulation_min_views cameras have usable extrinsics."""


def _load_cameras(
    self,
    views: list[ViewResult],
    calibration_result: CalibrationResult,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build {view_name: (R, t, K)} from views whose CameraParams carry
    Phase-4-solved extrinsics (view.camera.has_extrinsics). K is read from
    view.camera.K (property), which is correct post-calibration because
    CalibrationPipeline.calibrate() now writes the corrected principal_point
    back onto view.camera alongside rotation/translation.
    Renamed from _load_selfcal_cameras — was one of two branches, now the
    only path."""
```

Removed: `_load_colmap_cameras`, `_load_selfcal_cameras` (renamed, see above), `_get_exif_orient`, `_kps_to_landscape`, `_all_to_landscape`, `_landscape_to_det`, `_undistort_keypoints`.

### 5.4 On-disk artefacts

`output/debug/refinement/refinement_results.json` — `frame_alignment` key removed from the JSON (was already `None`-valued whenever self-cal was used; now always absent rather than always `null`). All other keys (`betas`, `body_pose`, `global_orient`, `translation`, `scale`, `metrics`, `cameras`) unchanged in shape.

`output/debug/refinement/summary.txt` — unchanged format.

## 6. User flows

No CLI surface exists yet for Phase 5 in isolation (it's invoked programmatically in tests/integration harnesses today, not via `scantosmpl` CLI subcommands). No user-facing flow changes. Internally, any code constructing a `Phase5Pipeline` must now also run `CalibrationPipeline.calibrate()` first and pass its `CalibrationResult` — this was already true for the `self_calibration` branch, now it's true unconditionally.

## 7. Data model & artefacts

- No new fixtures. `tests/integration/test_phase5_integration.py`'s fixtures change shape (per D9) but reuse the same underlying data (`data/t-pose/jpg`, `output/debug/detection/detections.json`, `output/debug/consensus/consensus_results.json`) already on disk.
- `COLMAP_DIR = Path("/home/dan/projects/auto-rigger/data/reconstruction/t-pose/0")` and `OUR_17` (view-name set used only for the COLMAP-matching test) are removed from the test file; `KNOWN_REAR_VIEWS` stays (still used by rear-view-exclusion-adjacent tests, if any remain relevant — verify during the brief).

## 8. Non-goals

- No changes to Tier 1 (per-view HMR) or Tier 3 (surface refinement — still unbuilt).
- No CLI flag surface.
- No re-tuning of `DEFAULT_STAGES` optimiser schedule.
- No changes to `CalibrationPipeline`/`CalibrationConfig` (Phase 4) beyond what's already merged.
- No attempt to recover COLMAP's rear-view baseline diversity by other means — rear views are excluded from the reprojection loss regardless of extrinsics source (D6), so COLMAP never contributed usable rear-view signal to begin with.

## 9. Rollout / migration

- `Phase5Config`/`Phase5Result` are both used only within test harnesses and internal pipeline code today (no serialised configs on disk to migrate, no external consumers). Field removal is a clean breaking change with no back-compat shim needed — CLAUDE.md's guidance against speculative back-compat applies directly here.
- Any caller still passing `extrinsics_source=...` or `colmap_model_dir=...` to `Phase5Config(...)` will fail fast with a `TypeError: unexpected keyword argument` — acceptable, since the only callers are the test files this same feature updates.

### Design notes for future Tier 3 (preserved per explicit request, not to be acted on now)

If/when `scantosmpl/pointcloud/` is built out, COLMAP's dense point cloud (not just its sparse camera reconstruction) could plausibly serve as a Tier 3 input alongside or instead of Meshroom PLY/OBJ exports — COLMAP's `points3D.bin` was never parsed by this codebase (only `cameras.bin`/`images.bin` were, via the now-deleted `colmap_reader.py`), so that would be new work, not a revival of anything deleted here. The specific thing NOT worth reviving: `frame_alignment.py`'s 7-DoF Procrustes-to-SMPL-frame alignment — Tier 3's rigid ICP (per the `CLAUDE.md` architecture diagram, "Step 3a: Align point cloud TO the SMPL mesh") already solves the identical scale+rotation+translation alignment problem generically, from geometry alone, without needing per-camera poses at all. Re-deriving camera-pose-based frame alignment for Tier 3 would be duplicate machinery solving a problem ICP already owns. If a future need for COLMAP camera poses specifically (not just its point cloud) resurfaces, `undistort.py`'s radial-undistortion approach (`cv2.undistortPoints` inverting COLMAP's `SIMPLE_RADIAL` model) is the one piece of genuinely reusable logic — it's a correct, self-contained utility, just currently orphaned by removing its only caller.

## 10. Acceptance Criteria

- **AC1** — COLMAP path fully removed. **Evidence**: `grep -rn "colmap\|COLMAP" scantosmpl/` returns no matches outside comments/docstrings explaining historical context (if any remain) — no live code path references `extrinsics_source`, `colmap_model_dir`, `ColmapCamera`, `ColmapImage`, `FrameAlignment`, or `undistort_keypoints`.
- **AC2** — Dead modules deleted. **Evidence**: `scantosmpl/calibration/colmap_reader.py`, `scantosmpl/calibration/frame_alignment.py`, `scantosmpl/calibration/undistort.py` do not exist; `scantosmpl/calibration/__init__.py` no longer imports from them; `tests/test_triangulation.py`'s `TestFrameAlignment`/undistortion test classes are removed (or the whole file, if nothing else remains in it — check before deciding).
- **AC3** — `Phase5Config` matches §5.1 exactly. **Evidence**: `python -c "from scantosmpl.config import Phase5Config; import dataclasses; print([f.name for f in dataclasses.fields(Phase5Config)])"` lists exactly the 8 fields in §5.1, no more, no less.
- **AC4** — `Phase5Result` matches §5.2 exactly. **Evidence**: same `dataclasses.fields` check on `Phase5Result`.
- **AC5** — `Phase5Pipeline.run()` requires `calibration_result`. **Evidence**: calling `Phase5Pipeline.run(views=..., consensus=..., image_dir=...)` without `calibration_result` raises `TypeError` (missing required argument), not a runtime `ValueError` deep in `_load_selfcal_cameras`.
- **AC6** — Integration suite passes end-to-end on self-cal only. **Evidence**: `pytest tests/integration/test_phase5_integration.py -v -m gpu --timeout=600` — all tests pass, none skipped due to a missing `COLMAP_DIR`.
- **AC7** — Metrics match or beat the experiment's measured numbers on this fixture. **Evidence**: `phase5_result.metrics["pa_mpjpe_mm"] < 24.5` and `phase5_result.metrics["median_reproj_px"] < 90.0` (giving ~10% slack over the experiment's 23.99mm / 78.8px to absorb PnP RANSAC's run-to-run variance — see repo spec §Determinism).
- **AC8** — `tests/integration/test_selfcal_phase5_experiment.py` deleted, no duplicate fixture logic remains between it and `test_phase5_integration.py` (there is only one file now).
- **AC9** — Rear-view exclusion still functions. **Evidence**: existing `KNOWN_REAR_VIEWS`-based assertions (or their equivalent) still pass — rear cameras are still excluded from the reprojection loss per `SMPLOptimiser.refine()`.
- **AC10** — Lint + typecheck green. **Evidence**: `py-lint` and `py-typecheck` skills exit 0 on every changed/deleted module.

## 11. Risks

- **R1 (Low × Medium): PnP RANSAC has no fixed seed** (OpenCV's Python bindings don't expose a seed parameter to `solvePnPRansac` directly). Re-running the integration suite could show ±5-10px variance in reprojection metrics run-to-run. **Mitigation**: AC7's thresholds (90px / 24.5mm) carry ~10% slack over the experiment's measured 78.8px / 23.99mm specifically to absorb this; if variance is still too high in practice, the fix-cycle should consider `cv2.setRNGSeed(0)` inside `CalibrationPipeline.calibrate()` (same pattern already used and validated in the abandoned `pnp_refine.py`'s test brief, safe to reuse the idea even though that module itself is deleted).
- **R2 (Low × Low): Deleting `colmap_reader.py`/`frame_alignment.py`/`undistort.py` removes genuinely well-written, tested utility code.** **Mitigation**: §9's design notes exist specifically so this isn't a silent loss — a future Tier 3 spec can consult them, or `git log`/this spec's git history if the code itself is ever wanted back.
- **R3 (Low × Low): `tests/test_triangulation.py` may become nearly empty** if its only remaining content after removing COLMAP/undistortion tests is unrelated triangulation-core tests. **Mitigation**: the brief should check what remains and either keep the (smaller) file or confirm nothing else depends on its current name/location.
