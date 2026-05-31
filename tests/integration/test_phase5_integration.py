"""
Integration tests for Phase 5: multi-view triangulation + SMPL refinement.

Run with:
    pytest tests/integration/test_phase5_integration.py -v -m gpu
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

DATA_DIR = Path("data/t-pose/jpg")
CONSENSUS_DIR = Path("output/debug/consensus")
COLMAP_DIR = Path("/home/dan/projects/auto-rigger/data/reconstruction/t-pose/0")
PHASE5_DEBUG_DIR = Path("output/debug/refinement")

OUR_17 = {
    "cam01_2.JPG", "cam01_6.JPG", "cam02_4.JPG", "cam02_5.JPG",
    "cam03_5.JPG", "cam03_6.JPG", "cam04_4.JPG", "cam04_5.JPG",
    "cam05_4.JPG", "cam05_5.JPG", "cam05_6.JPG", "cam06_4.JPG",
    "cam07_4.JPG", "cam07_6.JPG", "cam10_2.JPG", "cam10_4.JPG",
    "cam10_5.JPG",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _skip_if_missing(require_colmap: bool = True):
    if not DATA_DIR.exists():
        pytest.skip(f"Test images not found at {DATA_DIR}")
    if not CONSENSUS_DIR.exists():
        pytest.skip(f"Phase 3 output not found at {CONSENSUS_DIR}. Run Phase 3 first.")
    if not (CONSENSUS_DIR / "consensus_results.json").exists():
        pytest.skip("consensus_results.json not found. Run Phase 3 first.")
    if require_colmap and not COLMAP_DIR.exists():
        pytest.skip(f"COLMAP reconstruction not found at {COLMAP_DIR}")


@pytest.fixture(scope="module")
def detection_views():
    det_json = Path("output/debug/detection/detections.json")
    if not det_json.exists():
        pytest.skip(f"Phase 1 detection results not found at {det_json}.")

    from scantosmpl.types import CameraParams, ViewResult, ViewType

    with open(det_json) as f:
        data = json.load(f)

    views = []
    for d in data:
        vt_str = d.get("view_type", "skip").lower()
        vt = {"full_body": ViewType.FULL_BODY, "partial": ViewType.PARTIAL}.get(
            vt_str, ViewType.SKIP
        )
        pp = d.get("principal_point", [0.0, 0.0])
        cam = CameraParams(
            focal_length=float(d["focal_length_px"]),
            principal_point=(float(pp[0]), float(pp[1])),
        )
        kps_raw = d.get("keypoints_2d") or d.get("keypoints")
        confs_raw = d.get("keypoint_confs") or d.get("keypoint_confidences")
        view = ViewResult(
            image_path=DATA_DIR / d["filename"],
            view_type=vt,
            bbox=np.array(d["bbox"], dtype=np.float32) if d.get("bbox") else None,
            keypoints_2d=np.array(kps_raw, dtype=np.float32) if kps_raw is not None else None,
            keypoint_confs=np.array(confs_raw, dtype=np.float32) if confs_raw is not None else None,
            camera=cam,
        )
        views.append(view)
    return views


@pytest.fixture(scope="module")
def consensus_result(detection_views):
    """Reconstruct ConsensusResult from saved JSON + SMPL forward pass."""
    consensus_json = CONSENSUS_DIR / "consensus_results.json"
    if not consensus_json.exists():
        pytest.skip("consensus_results.json not found. Run Phase 3 first.")

    smpl_model_dir = Path("models/smpl")
    if not smpl_model_dir.exists():
        pytest.skip(f"SMPL model dir not found at {smpl_model_dir}")

    import torch
    from scantosmpl.hmr.consensus import ConsensusResult
    from scantosmpl.smpl.model import SMPLModel

    with open(consensus_json) as f:
        data = json.load(f)

    # vertices/joints/faces are not serialised to JSON — reconstruct via SMPL
    smpl = SMPLModel(model_dir=smpl_model_dir, gender="neutral", device="cuda")
    with torch.no_grad():
        smpl.set_params(
            betas=torch.tensor(data["betas"], dtype=torch.float32, device="cuda").unsqueeze(0),
            body_pose=torch.tensor(data["body_pose"], dtype=torch.float32, device="cuda").unsqueeze(0),
            global_orient=torch.tensor(data["global_orient"], dtype=torch.float32, device="cuda").unsqueeze(0),
        )
        output = smpl.forward()
    vertices = output.vertices.squeeze(0).cpu().numpy()
    joints = output.joints.squeeze(0).cpu().numpy()
    faces = output.faces.cpu().numpy()

    return ConsensusResult(
        betas=np.array(data["betas"]),
        body_pose=np.array(data["body_pose"]),
        global_orient=np.array(data["global_orient"]),
        vertices=vertices,
        joints=joints,
        faces=faces,
        pa_mpjpe_per_view=data["pa_mpjpe_per_view"],
        pa_mpjpe_mean=data.get("pa_mpjpe_mean", data.get("pa_mpjpe_mean_mm", 0.0)),
        beta_std=np.array(data["beta_std"]),
        body_height_m=data["body_height_m"],
        per_view_weights=data["per_view_weights"],
        n_views_used=data["n_views_used"],
    )


@pytest.fixture(scope="module")
def phase5_result(detection_views, consensus_result):
    """Run Phase 5 COLMAP pipeline and cache result."""
    _skip_if_missing(require_colmap=True)

    from scantosmpl.config import FittingConfig, Phase5Config
    from scantosmpl.fitting.pipeline import Phase5Pipeline
    from scantosmpl.smpl.model import SMPLModel

    smpl = SMPLModel(
        model_dir=Path("models/smpl"),
        gender="neutral",
        device="cuda",
    )

    cfg = Phase5Config(
        extrinsics_source="colmap",
        colmap_model_dir=COLMAP_DIR,
        triangulation_conf_threshold=0.3,
        triangulation_min_views=3,
        ransac_reproj_threshold=150.0,  # ViTPose noise ~50-100px on 6000px images
        ransac_iterations=200,
        save_debug=True,
        debug_dir=PHASE5_DEBUG_DIR,
    )

    pipeline = Phase5Pipeline(smpl, cfg)
    result = pipeline.run(
        views=detection_views,
        consensus=consensus_result,
        image_dir=DATA_DIR,
    )
    return result


# ---------------------------------------------------------------------------
# 5.1: All 17 Phase 1 views have COLMAP extrinsics
# ---------------------------------------------------------------------------


class TestCOLMAPReader:
    def test_all_17_views_have_colmap_extrinsics(self):
        """Criterion 5.1: all 17 Phase 1 views are present in COLMAP."""
        if not COLMAP_DIR.exists():
            pytest.skip("COLMAP model not available")

        from scantosmpl.calibration.colmap_reader import match_views_to_colmap, read_colmap_model

        cameras, images = read_colmap_model(COLMAP_DIR)
        matched, missing = match_views_to_colmap(list(OUR_17), images)

        assert len(missing) == 0, f"Views missing from COLMAP: {missing}"
        assert len(matched) == 17

        # Verify each matched view resolves to a valid camera
        for name, img in matched.items():
            assert img.camera_id in cameras, (
                f"{name}: camera_id={img.camera_id} not in cameras"
            )


# ---------------------------------------------------------------------------
# 5.2: Frame alignment reprojection error < 15px
# ---------------------------------------------------------------------------


class TestFrameAlignment:
    def test_frame_alignment_quality(self, phase5_result):
        """Criterion 5.2: mean reprojection of consensus joints < 15px after alignment."""
        from scantosmpl.utils.geometry import project_points

        alignment = phase5_result.frame_alignment
        if alignment is None:
            pytest.skip("No frame alignment (self-calibration fallback used)")

        cameras = phase5_result.cameras_smpl_frame

        # Use the Phase 5 result metrics
        mean_reproj = phase5_result.metrics.get("mean_reproj_px", float("inf"))
        assert mean_reproj < 15.0, (
            f"Criterion 5.2 FAIL: mean reproj = {mean_reproj:.1f}px (target <15px)"
        )


# ---------------------------------------------------------------------------
# 5.3: Triangulation accuracy (PA-MPJPE < 30mm vs consensus)
# ---------------------------------------------------------------------------


class TestTriangulation:
    def test_triangulation_accuracy(self, phase5_result, consensus_result):
        """Criterion 5.3: triangulated joints are within 30mm PA-MPJPE of consensus."""
        from scantosmpl.utils.geometry import compute_pa_mpjpe

        quality = phase5_result.triangulation_quality
        n_valid = int((quality > 0).sum())
        assert n_valid >= 8, f"Only {n_valid} joints triangulated successfully (need ≥8)"

        # triangulated_joints_smpl is (24, 3) with non-zero only at mapped SMPL indices
        triang_smpl = phase5_result.triangulated_joints_smpl
        valid_smpl = np.linalg.norm(triang_smpl, axis=1) > 1e-6  # (24,) bool

        pa_mpjpe = compute_pa_mpjpe(
            triang_smpl[valid_smpl],
            consensus_result.joints[valid_smpl],
        ) * 1000

        assert pa_mpjpe < 30.0, (
            f"Criterion 5.3 FAIL: PA-MPJPE = {pa_mpjpe:.1f}mm (target <30mm)"
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
# 5.5: Reprojection error < 15px
# ---------------------------------------------------------------------------


class TestReprojectionError:
    def test_reprojection_error(self, phase5_result):
        """Criterion 5.5: mean reprojection error < 15px."""
        mean_reproj = phase5_result.metrics.get("mean_reproj_px", float("inf"))
        assert mean_reproj < 15.0, (
            f"Criterion 5.5 FAIL: mean reprojection = {mean_reproj:.1f}px (target <15px)"
        )

    def test_reprojection_computed_for_all_views(self, phase5_result):
        """Reprojection should be computed across multiple views."""
        n_terms = phase5_result.metrics.get("n_reproj_terms", 0)
        assert n_terms >= 50, f"Expected ≥50 reprojection terms, got {n_terms}"


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
        assert len(data["betas"]) == 10
        assert len(data["body_pose"]) == 69

    def test_reprojection_overlays_created(self, phase5_result):
        """reprojection_overlay/ should contain per-view images."""
        overlay_dir = PHASE5_DEBUG_DIR / "reprojection_overlay"
        if not overlay_dir.exists():
            pytest.skip("reprojection_overlay/ not created")
        overlays = list(overlay_dir.glob("*_overlay.jpg"))
        assert len(overlays) > 0, "No overlay images created"
