"""
Integration tests for Phase 5: multi-view triangulation + SMPL refinement.

Extrinsics come solely from Phase 4 self-calibration (cold `cv2.solvePnPRansac`
against SMPL joints + ViTPose keypoints) — see
docs/features/selfcal-default-extrinsics/00-master-design-spec.md. The COLMAP
extrinsics path has been retired; these fixtures mirror the pattern validated
by the (now-deleted) tests/integration/test_selfcal_phase5_experiment.py.

Run with:
    pytest tests/integration/test_phase5_integration.py -v -m gpu
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

DATA_DIR = Path("data/t-pose/jpg")
CONSENSUS_DIR = Path("output/debug/consensus")
CALIBRATION_DEBUG_DIR = Path("output/debug/calibration")
PHASE5_DEBUG_DIR = Path("output/debug/refinement")
EXIF_FOCAL_PX = 6349.0

# Rear-facing cameras for this fixture, captured from a real Phase 5 run (see
# tests/test_rear_views.py's integration-log regression fixture, sourced from
# this same data/t-pose/jpg dataset). Used only to sanity-check that rear-view
# classification stays roughly stable across self-calibration PnP re-runs —
# cv2.solvePnPRansac has no fixed seed (repo spec §Determinism / master R1),
# so exact agreement with this set is not required, just substantial overlap.
KNOWN_REAR_VIEWS = {
    "cam02_5.JPG",
    "cam03_5.JPG",
    "cam03_6.JPG",
    "cam04_4.JPG",
    "cam04_5.JPG",
    "cam05_4.JPG",
    "cam05_5.JPG",
    "cam05_6.JPG",
    "cam10_4.JPG",
    "cam10_5.JPG",
}


def _skip_if_missing():
    if not DATA_DIR.exists():
        pytest.skip(f"Test images not found at {DATA_DIR}")
    if not CONSENSUS_DIR.exists():
        pytest.skip(f"Phase 3 output not found at {CONSENSUS_DIR}. Run Phase 3 first.")
    if not (CONSENSUS_DIR / "consensus_results.json").exists():
        pytest.skip("consensus_results.json not found. Run Phase 3 first.")


# ---------------------------------------------------------------------------
# Fixtures — self-calibration pattern (Phase 4 CalibrationPipeline ->
# CalibrationResult), folded in from the now-deleted
# test_selfcal_phase5_experiment.py per master spec D9.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detection_views():
    """Load Phase 1 detection results."""
    det_json = Path("output/debug/detection/detections.json")
    if not det_json.exists():
        pytest.skip(f"Phase 1 detection results not found at {det_json}.")

    from scantosmpl.types import CameraParams, ViewResult, ViewType

    with open(det_json) as f:
        data = json.load(f)

    views = []
    for det in data:
        name = det["filename"]
        vt_str = det.get("view_type", "skip").lower()
        vt = {"full_body": ViewType.FULL_BODY, "partial": ViewType.PARTIAL}.get(
            vt_str, ViewType.SKIP
        )
        bbox = np.array(det["bbox"], dtype=np.float32) if det.get("bbox") else None
        kps_raw = det.get("keypoints_2d") or det.get("keypoints")
        kps = np.array(kps_raw, dtype=np.float32) if kps_raw is not None else None
        confs_raw = det.get("keypoint_confs") or det.get("keypoint_confidences")
        confs = np.array(confs_raw, dtype=np.float32) if confs_raw is not None else None
        fl = float(det.get("focal_length_px", EXIF_FOCAL_PX))
        camera = CameraParams(focal_length=fl, principal_point=(0.0, 0.0))

        views.append(
            ViewResult(
                image_path=DATA_DIR / name,
                view_type=vt,
                bbox=bbox,
                keypoints_2d=kps,
                keypoint_confs=confs,
                camera=camera,
            )
        )
    return views


@pytest.fixture(scope="module")
def hmr_views(detection_views):
    """Run Phase 2 HMR: adds SMPL params + 138 dense keypoints per view.

    Phase 4's calibrate() mutates these ViewResult.camera objects in place
    (rotation/translation/principal_point), so this same list is reused,
    post-calibration, as the `views` argument to Phase5Pipeline.run().
    """
    _skip_if_missing()
    from scantosmpl.config import HMRConfig
    from scantosmpl.hmr.pipeline import HMRPipeline

    smpl_model = Path("models/smpl/SMPL_NEUTRAL.pkl")
    if not smpl_model.exists():
        pytest.skip(f"SMPL model not found at {smpl_model}")

    config = HMRConfig()
    pipeline = HMRPipeline(config, device="cuda")
    return pipeline.process_views(detection_views, DATA_DIR)


@pytest.fixture(scope="module")
def consensus_result(hmr_views):
    """Build Phase 3 consensus from HMR views.

    Rebuilt directly from hmr_views (rather than reconstructed from the
    cached consensus_results.json) so it shares the same ViewResult object
    graph that Phase 4 calibrates against in-place — see master spec D9 /
    brief step 1.
    """
    from scantosmpl.hmr.consensus import ConsensusBuilder

    smpl_model = Path("models/smpl/SMPL_NEUTRAL.pkl")
    builder = ConsensusBuilder(smpl_model, device="cuda")
    return builder.build_consensus(hmr_views, debug_dir=CONSENSUS_DIR, image_dir=DATA_DIR)


@pytest.fixture(scope="module")
def calibration_result(hmr_views, consensus_result):
    """Run Phase 4 self-calibration (cold PnP, no COLMAP involved).

    Mutates hmr_views[i].camera.rotation/translation/principal_point in place.
    """
    from scantosmpl.calibration.pipeline import CalibrationPipeline
    from scantosmpl.config import CalibrationConfig

    config = CalibrationConfig()
    pipeline = CalibrationPipeline(config)
    return pipeline.calibrate(
        hmr_views,
        consensus_result,
        DATA_DIR,
        debug_dir=CALIBRATION_DEBUG_DIR,
    )


@pytest.fixture(scope="module")
def phase5_result(hmr_views, consensus_result, calibration_result):
    """Run Phase 5 (self-calibration extrinsics only) and cache the result."""
    from scantosmpl.config import Phase5Config
    from scantosmpl.fitting.pipeline import Phase5Pipeline
    from scantosmpl.smpl.model import SMPLModel

    smpl = SMPLModel(
        model_dir=Path("models/smpl"),
        gender="neutral",
        device="cuda",
    )

    cfg = Phase5Config(
        triangulation_conf_threshold=0.3,
        triangulation_min_views=3,
        ransac_reproj_threshold=150.0,  # ViTPose noise ~50-100px on 6000px images
        ransac_iterations=200,
        save_debug=True,
        debug_dir=PHASE5_DEBUG_DIR,
    )

    pipeline = Phase5Pipeline(smpl, cfg)
    result = pipeline.run(
        views=hmr_views,
        consensus=consensus_result,
        image_dir=DATA_DIR,
        calibration_result=calibration_result,
    )
    return result


# ---------------------------------------------------------------------------
# Reprojection quality of self-calibrated cameras (was TestFrameAlignment —
# Procrustes-to-COLMAP frame alignment no longer exists; this now checks the
# self-cal cameras' own reprojection quality). Master §10 AC7.
# ---------------------------------------------------------------------------


class TestReprojectionQuality:
    def test_reprojection_quality(self, phase5_result):
        """Master AC7: self-calibrated cameras land near the ViTPose noise floor."""
        median_reproj = phase5_result.metrics.get("median_reproj_px", float("inf"))
        mean_reproj_inliers = phase5_result.metrics.get("mean_reproj_inliers_px", float("inf"))
        logger.info(
            "Reprojection quality: median=%.2fpx, mean_inliers=%.2fpx",
            median_reproj,
            mean_reproj_inliers,
        )
        assert median_reproj < 90.0, (
            f"AC7 FAIL: median reproj = {median_reproj:.1f}px (target <90px)"
        )
        # Not a master §10 AC7 metric (only median_reproj_px/pa_mpjpe_mm are)
        # but retained as a secondary quality signal; 250px matches the
        # pre-pivot threshold and comfortably covers the ~214px measured on
        # this fixture (mean-of-inlier-view-means includes side views with
        # naturally higher per-view reprojection spread than the median).
        assert mean_reproj_inliers < 250.0, (
            f"mean inlier reproj = {mean_reproj_inliers:.1f}px (target <250px)"
        )


# ---------------------------------------------------------------------------
# Triangulation accuracy (PA-MPJPE vs consensus, and refinement-side
# pa_mpjpe_mm per master §10 AC7).
# ---------------------------------------------------------------------------


class TestTriangulation:
    def test_triangulation_accuracy(self, phase5_result, consensus_result):
        """Criterion 5.3: triangulated joints are close to consensus, and
        master AC7: refinement PA-MPJPE (phase5_result.metrics["pa_mpjpe_mm"])
        beats the self-calibration experiment's measured 23.99mm with ~10%
        slack.
        """
        from scantosmpl.utils.geometry import compute_pa_mpjpe

        quality = phase5_result.triangulation_quality
        n_valid = int((quality > 0).sum())
        assert n_valid >= 8, f"Only {n_valid} joints triangulated successfully (need >=8)"

        # triangulated_joints_smpl is (24, 3) with non-zero only at mapped SMPL indices
        triang_smpl = phase5_result.triangulated_joints_smpl
        valid_smpl = np.linalg.norm(triang_smpl, axis=1) > 1e-6  # (24,) bool

        pa_mpjpe_vs_consensus = (
            compute_pa_mpjpe(
                triang_smpl[valid_smpl],
                consensus_result.joints[valid_smpl],
            )
            * 1000
        )
        logger.info("Triangulation vs consensus PA-MPJPE=%.2fmm", pa_mpjpe_vs_consensus)
        # Tightened from the pre-pivot 35.0mm: self-calibration measured
        # ~25.1mm here, ~20% better than the old threshold implied, so this
        # keeps meaningful headroom over the measured value without being
        # stale (repo spec §Determinism: PnP RANSAC is unseeded, expect some
        # run-to-run variance).
        assert pa_mpjpe_vs_consensus < 30.0, (
            f"Triangulation-vs-consensus PA-MPJPE = {pa_mpjpe_vs_consensus:.1f}mm (target <30mm)"
        )

        pa_mpjpe_refined = phase5_result.metrics.get("pa_mpjpe_mm", float("inf"))
        logger.info("Refinement PA-MPJPE=%.2fmm", pa_mpjpe_refined)
        assert pa_mpjpe_refined < 24.5, (
            f"AC7 FAIL: refinement PA-MPJPE = {pa_mpjpe_refined:.1f}mm (target <24.5mm)"
        )

    def test_triangulation_returns_valid_joints(self, phase5_result):
        """At least 8 joints should be successfully triangulated."""
        quality = phase5_result.triangulation_quality
        assert int((quality > 0).sum()) >= 8, (
            f"Too few successful joints: {int((quality > 0).sum())}"
        )


# ---------------------------------------------------------------------------
# 5.4: SMPL refinement improves over consensus
# ---------------------------------------------------------------------------


class TestSMPLRefinement:
    def test_smpl_refinement_improves(self, phase5_result, consensus_result):
        """Criterion 5.4: PA-MPJPE after refinement decreases vs consensus baseline."""
        from scantosmpl.utils.geometry import compute_pa_mpjpe

        # triangulated_joints_smpl is (24, 3) — zero for unmapped joints
        triang_smpl = phase5_result.triangulated_joints_smpl
        valid = np.linalg.norm(triang_smpl, axis=1) > 1e-6  # (24,) bool

        if int(valid.sum()) < 4:
            pytest.skip("Too few valid triangulations to assess improvement")

        refined_joints = phase5_result.refined.joints
        consensus_joints = consensus_result.joints

        # Compare refined vs triangulated (ground truth proxy)
        pa_refined = compute_pa_mpjpe(refined_joints[valid], triang_smpl[valid]) * 1000
        # Compare consensus (baseline) vs triangulated
        pa_consensus = compute_pa_mpjpe(consensus_joints[valid], triang_smpl[valid]) * 1000

        assert pa_refined <= pa_consensus * 1.05, (
            f"Criterion 5.4 FAIL: refinement didn't improve "
            f"({pa_refined:.1f}mm vs consensus {pa_consensus:.1f}mm)"
        )

    def test_loss_decreases_per_stage(self, phase5_result):
        """Each optimisation stage should decrease total loss."""
        history = phase5_result.refined.loss_history
        for stage_name, losses in history.items():
            if len(losses) < 2:
                continue
            assert losses[-1] <= losses[0], (
                f"Stage '{stage_name}': loss increased from {losses[0]:.4f} → {losses[-1]:.4f}"
            )


# ---------------------------------------------------------------------------
# 5.5: Reprojection error — master §10 AC7
# ---------------------------------------------------------------------------


class TestReprojectionError:
    def test_reprojection_error(self, phase5_result):
        """Master AC7: median reprojection error < 90px."""
        median_reproj = phase5_result.metrics.get("median_reproj_px", float("inf"))
        logger.info(f"reprojection err: median={median_reproj:.2f}px")

        assert median_reproj < 90.0, (
            f"AC7 FAIL: median reprojection = {median_reproj:.1f}px (target <90px)"
        )

    def test_reprojection_computed_for_all_views(self, phase5_result):
        """Reprojection should be computed across multiple views."""
        n_terms = phase5_result.metrics.get("n_reproj_terms", 0)
        assert n_terms >= 50, f"Expected ≥50 reprojection terms, got {n_terms}"


# ---------------------------------------------------------------------------
# Rear-view exclusion (master D6/AC9) — still functions regardless of
# extrinsics source.
# ---------------------------------------------------------------------------


class TestRearViewExclusion:
    def test_rear_views_excluded_from_reprojection(self, phase5_result, consensus_result):
        """Master AC9: rear-facing cameras are excluded from the reprojection
        loss inside SMPLOptimiser.refine() (scantosmpl.fitting.rear_views.
        classify_rear_views), independent of which extrinsics source produced
        the cameras. No invented debug-JSON field is used (refinement_results
        .json carries no pnp_refinement/cameras_pre_pnp block — that was the
        abandoned feature); instead this recomputes the classification
        against phase5_result's own self-calibrated cameras and checks:

          1. The classifier finds a non-degenerate front/rear split on this
             real 17-camera fixture (not empty, not every camera), and every
             view it calls "rear" is among the historically-known rear set
             for this dataset (tests/test_rear_views.py's integration
             fixture) — i.e. no false positives flipping a front-facing
             camera to "rear". Full recall against that set is *not*
             required: which side-view cameras land just inside vs. just
             outside the rear boundary shifts slightly run-to-run because
             cv2.solvePnPRansac has no fixed seed (repo spec §Determinism /
             master R1) and consensus geometry depends on which views HMR
             successfully processes.
          2. The reprojection metric's outlier-view count stays low,
             confirming rear views are being excluded from the loss rather
             than corrupting it.
        """
        from scantosmpl.fitting.rear_views import classify_rear_views

        n_cameras = len(phase5_result.cameras_smpl_frame)
        rear = classify_rear_views(consensus_result, phase5_result.cameras_smpl_frame)
        logger.info("Classified rear views (%d/%d cameras): %s", len(rear), n_cameras, rear)

        assert rear, "Expected at least one rear view to be classified for this fixture"
        assert len(rear) < n_cameras, "Not every camera should be classified as rear"

        unexpected = rear - KNOWN_REAR_VIEWS
        assert not unexpected, (
            f"Classified as rear but not in the historically-known rear set: "
            f"{sorted(unexpected)} (known set: {sorted(KNOWN_REAR_VIEWS)})"
        )

        n_outlier = phase5_result.metrics.get("n_outlier_views", 999)
        assert n_outlier <= 3, (
            f"Too many outlier views in reprojection metric ({n_outlier}) — "
            f"rear-view exclusion may not be functioning"
        )


# ---------------------------------------------------------------------------
# 5.6: Debug output complete
# ---------------------------------------------------------------------------


class TestDebugOutput:
    def test_debug_output_files_created(self, phase5_result):
        """Criterion 5.6: required debug files are created."""
        assert PHASE5_DEBUG_DIR.exists(), f"Debug dir missing: {PHASE5_DEBUG_DIR}"

        expected_files = [
            "refinement_results.json",
            "triangulated_joints.json",
            "summary.txt",
            "convergence.png",
            "camera_positions.png",
        ]
        for fname in expected_files:
            path = PHASE5_DEBUG_DIR / fname
            assert path.exists(), f"Missing debug file: {path}"

    def test_refinement_results_json_valid(self, phase5_result):
        """refinement_results.json should contain expected keys."""
        path = PHASE5_DEBUG_DIR / "refinement_results.json"
        if not path.exists():
            pytest.skip("refinement_results.json not created")

        with open(path) as f:
            data = json.load(f)

        assert "betas" in data
        assert "body_pose" in data
        assert "global_orient" in data
        assert "metrics" in data
        assert "frame_alignment" not in data
        assert len(data["betas"]) == 10
        assert len(data["body_pose"]) == 69

    def test_reprojection_overlays_created(self, phase5_result):
        """reprojection_overlay/ should contain per-view images."""
        overlay_dir = PHASE5_DEBUG_DIR / "reprojection_overlay"
        if not overlay_dir.exists():
            pytest.skip("reprojection_overlay/ not created")
        overlays = list(overlay_dir.glob("*_overlay.jpg"))
        assert len(overlays) > 0, "No overlay images created"
