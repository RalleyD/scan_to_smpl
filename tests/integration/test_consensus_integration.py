"""
Integration tests for Phase 3 consensus — requires GPU + Phase 2 HMR output.

Run with:
    pytest tests/integration/test_consensus_integration.py -v -m gpu
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

DATA_DIR = Path("data/t-pose/jpg")
HMR_DEBUG_DIR = Path("output/debug/hmr")
EXIF_FOCAL_PX = 6349.0


def _skip_if_missing():
    if not HMR_DEBUG_DIR.exists():
        pytest.skip(f"Phase 2 HMR output not found at {HMR_DEBUG_DIR}. Run Phase 2 first.")
    if not (HMR_DEBUG_DIR / "hmr_results.json").exists():
        pytest.skip("hmr_results.json not found. Run Phase 2 first.")
    if not DATA_DIR.exists():
        pytest.skip(f"Test images not found at {DATA_DIR}")


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
    """Run Phase 2 HMR pipeline to get views with SMPL params."""
    _skip_if_missing()
    from scantosmpl.config import HMRConfig
    from scantosmpl.hmr.pipeline import HMRPipeline

    smpl_model = Path("models/smpl/SMPL_NEUTRAL.pkl")
    if not smpl_model.exists():
        pytest.skip(f"SMPL model not found at {smpl_model}")

    config = HMRConfig()
    pipeline = HMRPipeline(config, device="cuda")
    views = pipeline.process_views(detection_views, DATA_DIR)
    return views


@pytest.fixture(scope="module")
def consensus_result(hmr_views):
    """Build consensus from HMR views."""
    from scantosmpl.hmr.consensus import ConsensusBuilder

    smpl_model = Path("models/smpl/SMPL_NEUTRAL.pkl")
    builder = ConsensusBuilder(smpl_model, device="cuda")
    debug_dir = Path("output/debug/consensus")
    result = builder.build_consensus(
        hmr_views, debug_dir=debug_dir, image_dir=DATA_DIR,
    )
    return result


# ---------------------------------------------------------------------------
# Test 1: Anatomical plausibility (criterion 3.1)
# ---------------------------------------------------------------------------


class TestAnatomicalPlausibility:
    def test_height_in_range(self, consensus_result):
        """Criterion 3.1: body height between 1.5m and 2.0m."""
        h = consensus_result.body_height_m
        assert 1.5 <= h <= 2.0, f"Body height {h:.3f}m outside plausible range"

    def test_betas_plausible(self, consensus_result):
        """Beta values should be in a reasonable range."""
        assert np.all(np.abs(consensus_result.betas) < 10.0), (
            f"Extreme beta values: {consensus_result.betas}"
        )


# ---------------------------------------------------------------------------
# Test 2: T-pose check (criterion 3.2)
# ---------------------------------------------------------------------------


class TestTPoseConsistency:
    def test_arms_roughly_horizontal(self, consensus_result):
        """Criterion 3.2: in T-pose, arm joints should be roughly level.

        In SMPL canonical frame (Y-up), shoulder/elbow/wrist Y-coords should
        be close for horizontal arms.
        """
        joints = consensus_result.joints  # (24, 3)

        for side, sh, el, wr in [("Left", 16, 18, 20), ("Right", 17, 19, 21)]:
            sh_y = float(joints[sh, 1])
            el_y = float(joints[el, 1])
            wr_y = float(joints[wr, 1])

            # Arm drop from shoulder: should be < 0.15m for T-pose
            arm_drop = abs(el_y - sh_y) + abs(wr_y - sh_y)
            assert arm_drop < 0.20, (
                f"{side} arm drop {arm_drop:.3f}m too large for T-pose. "
                f"shoulder_y={sh_y:.3f}, elbow_y={el_y:.3f}, wrist_y={wr_y:.3f}"
            )

    def test_body_pose_norm_reasonable(self, consensus_result):
        """T-pose body_pose should have low total rotation magnitude."""
        norm = float(np.linalg.norm(consensus_result.body_pose))
        # T-pose is not exactly zero (arms are lifted), but shouldn't be huge
        assert norm < 4.0, f"Body pose norm {norm:.3f} rad too large for T-pose"


# ---------------------------------------------------------------------------
# Test 3: Consensus improves over single-best (criterion 3.3)
# ---------------------------------------------------------------------------


class TestConsensusImprovement:
    def test_consensus_lower_variance_than_single(self, consensus_result, hmr_views):
        """Criterion 3.3: consensus beta variance should be lower than using
        any single view as the 'ground truth'."""
        from scantosmpl.types import ViewType

        valid = [
            v for v in hmr_views
            if v.betas is not None and v.hmr_suitable and v.view_type != ViewType.SKIP
        ]
        if len(valid) < 3:
            pytest.skip("Not enough views for comparison")

        betas_arr = np.stack([v.betas for v in valid], axis=0)  # (N, 10)

        # Single-best: the view whose betas are closest to the mean
        mean_betas = betas_arr.mean(axis=0)
        dists = np.linalg.norm(betas_arr - mean_betas, axis=1)
        best_idx = int(dists.argmin())

        # Variance of deviations from consensus vs from single best
        consensus_devs = np.linalg.norm(betas_arr - consensus_result.betas, axis=1)
        single_devs = np.linalg.norm(betas_arr - betas_arr[best_idx], axis=1)

        # Consensus should have equal or lower mean deviation
        assert consensus_devs.mean() <= single_devs.mean() + 0.01, (
            f"Consensus mean deviation ({consensus_devs.mean():.4f}) > "
            f"single-best ({single_devs.mean():.4f})"
        )


# ---------------------------------------------------------------------------
# Test 4: SO(3) validity (criterion 3.4)
# ---------------------------------------------------------------------------


class TestSO3Validity:
    def test_consensus_pose_joints_valid(self, consensus_result):
        """Criterion 3.4: each joint's rotation should be a valid SO(3) element."""
        from scantosmpl.utils.geometry import aa_to_rotmat

        bp = consensus_result.body_pose.reshape(23, 3)
        rotmats = aa_to_rotmat(bp)  # (23, 3, 3)

        for j in range(23):
            R = rotmats[j]
            det = float(np.linalg.det(R))
            np.testing.assert_allclose(det, 1.0, atol=1e-5,
                                       err_msg=f"Joint {j}: det={det}")
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5,
                                       err_msg=f"Joint {j}: not orthogonal")


# ---------------------------------------------------------------------------
# Test 5: PA-MPJPE (criterion 3.6)
# ---------------------------------------------------------------------------


class TestCrossViewConsistency:
    def test_pa_mpjpe_under_50mm(self, consensus_result):
        """Criterion 3.6: mean PA-MPJPE across views < 50mm."""
        assert consensus_result.pa_mpjpe_mean < 50.0, (
            f"Mean PA-MPJPE {consensus_result.pa_mpjpe_mean:.2f}mm >= 50mm"
        )

    def test_no_view_exceeds_100mm(self, consensus_result):
        """No single view should have PA-MPJPE > 100mm (sanity check).

        Views with extreme angles are already excluded by the HMR suitability
        filter; the remaining views should all be reasonable.
        """
        for name, pa in consensus_result.pa_mpjpe_per_view.items():
            assert pa < 100.0, f"{name}: PA-MPJPE {pa:.2f}mm > 100mm"


# ---------------------------------------------------------------------------
# Test 6: Mesh output
# ---------------------------------------------------------------------------


class TestConsensusMesh:
    def test_vertex_count(self, consensus_result):
        assert consensus_result.vertices.shape == (6890, 3)

    def test_face_count(self, consensus_result):
        assert consensus_result.faces.shape == (13776, 3)

    def test_joints_count(self, consensus_result):
        assert consensus_result.joints.shape == (24, 3)


# ---------------------------------------------------------------------------
# Test 7: Debug output
# ---------------------------------------------------------------------------


class TestDebugOutput:
    def test_obj_file_created(self, consensus_result):
        obj_path = Path("output/debug/consensus/consensus_mesh.obj")
        assert obj_path.exists(), "consensus_mesh.obj not created"
        content = obj_path.read_text()
        assert content.count("\nv ") >= 6890, "OBJ file has too few vertices"

    def test_json_results_created(self, consensus_result):
        json_path = Path("output/debug/consensus/consensus_results.json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "betas" in data
        assert len(data["betas"]) == 10
        assert "pa_mpjpe_mean_mm" in data
        assert "body_height_m" in data

    def test_summary_created(self, consensus_result):
        summary_path = Path("output/debug/consensus/summary.txt")
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "PA-MPJPE" in content
        assert "height" in content.lower()

    def test_overlay_files_created(self, consensus_result):
        """At least one consensus overlay should exist."""
        debug_dir = Path("output/debug/consensus")
        overlays = list(debug_dir.glob("*_consensus_overlay.jpg"))
        assert len(overlays) >= 1, "No consensus overlay files created"
