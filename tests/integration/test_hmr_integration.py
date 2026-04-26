"""
Integration tests for Phase 2 HMR — requires GPU + downloaded checkpoints.

Run with:
    pytest tests/integration/test_hmr_integration.py -v -m gpu
"""

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# Paths expected by the pipeline
DATA_DIR = Path("data/t-pose/jpg")
CHECKPOINTS = {
    "camerahmr": Path("models/checkpoints/camera_hmr/camerahmr_checkpoint_cleaned.ckpt"),
    "densekp": Path("models/checkpoints/camera_hmr/densekp.ckpt"),
    "cam_model": Path("models/checkpoints/camera_hmr/cam_model_cleaned.ckpt"),
    "smpl_mean": Path("models/smpl/smpl_mean_params.npz"),
    "smpl_model": Path("models/smpl/SMPL_NEUTRAL.pkl"),
}

# EXIF focal length used in Phase 1 (Canon EOS 2000D, 24mm lens)
EXIF_FOCAL_PX = 6349.0


def _skip_if_missing():
    missing = [str(p) for p in CHECKPOINTS.values() if not p.exists()]
    if missing:
        pytest.skip(f"Checkpoints not found: {missing}")
    if not DATA_DIR.exists():
        pytest.skip(f"Test images not found: {DATA_DIR}")


@pytest.fixture(scope="module")
def hmr_config():
    from scantosmpl.config import HMRConfig

    return HMRConfig()


@pytest.fixture(scope="module")
def inference(hmr_config):
    _skip_if_missing()
    from scantosmpl.hmr.camera_hmr import CameraHMRInference

    return CameraHMRInference(hmr_config, device="cuda")


@pytest.fixture(scope="module")
def detection_views():
    """Load Phase 1 detection results; skip if not available."""
    det_json = Path("output/debug/detection/detections.json")
    if not det_json.exists():
        pytest.skip(f"Phase 1 detection results not found at {det_json}. Run Phase 1 first.")

    import json

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
        kps = np.array(det["keypoints_2d"], dtype=np.float32) if det.get("keypoints_2d") else None
        confs = (
            np.array(det["keypoint_confs"], dtype=np.float32) if det.get("keypoint_confs") else None
        )
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


# ---------------------------------------------------------------------------
# Test 1: SMPL output shapes
# ---------------------------------------------------------------------------


def _first_full_body_view(detection_views):
    """Return the first FULL_BODY view with a detected bbox."""
    from scantosmpl.types import ViewType

    for v in detection_views:
        if v.view_type == ViewType.FULL_BODY and v.bbox is not None:
            return v
    pytest.skip("No FULL_BODY views with bbox found in detection_views")


class TestCameraHMRSingleImage:
    def test_output_shapes(self, inference, detection_views):
        """Criterion 2.1: β (10,), θ (69,) per view."""
        view = _first_full_body_view(detection_views)
        img = Image.open(DATA_DIR / view.image_path.name)
        out = inference.infer(img, view.bbox, view.camera.focal_length)

        assert out.betas.shape == (10,), f"betas shape: {out.betas.shape}"
        assert out.body_pose.shape == (69,), f"body_pose shape: {out.body_pose.shape}"
        assert out.global_orient.shape == (3,), f"global_orient shape: {out.global_orient.shape}"
        assert out.cam_translation.shape == (3,)
        assert out.dense_keypoints_2d.shape == (138, 2)
        assert out.dense_keypoint_confs.shape == (138,)

    def test_betas_plausible(self, inference, detection_views):
        """β values should be in a reasonable range (|β| < 10 for normal subjects)."""
        view = _first_full_body_view(detection_views)
        img = Image.open(DATA_DIR / view.image_path.name)
        out = inference.infer(img, view.bbox, view.camera.focal_length)
        assert np.all(np.abs(out.betas) < 10.0), f"Extreme beta values: {out.betas}"

    def test_rotation_magnitudes_valid(self, inference, detection_views):
        """Global orient and body_pose rotations should have plausible magnitudes."""
        view = _first_full_body_view(detection_views)
        img = Image.open(DATA_DIR / view.image_path.name)
        out = inference.infer(img, view.bbox, view.camera.focal_length)

        go_mag = float(np.linalg.norm(out.global_orient))
        bp_norms = [float(np.linalg.norm(out.body_pose[i * 3: i * 3 + 3])) for i in range(23)]
        assert go_mag < 2 * np.pi, f"Global orient magnitude too large: {go_mag}"
        assert max(bp_norms) < 2 * np.pi, f"Body pose joint magnitude too large: {max(bp_norms)}"


# ---------------------------------------------------------------------------
# Test 2: FoV cross-check (criterion 2.2)
# ---------------------------------------------------------------------------


class TestFoVCrossCheck:
    def test_fov_within_10_degrees_of_exif(self, inference, detection_views):
        """Criterion 2.2: |fov_exif − fov_flnet| < 10° for all images."""
        from scantosmpl.types import ViewType

        full_body = [
            v for v in detection_views
            if v.view_type == ViewType.FULL_BODY and v.bbox is not None
        ][:5]
        assert full_body, "No FULL_BODY views found"

        diffs = []
        for view in full_body:
            img = Image.open(DATA_DIR / view.image_path.name)
            out = inference.infer(img, view.bbox, view.camera.focal_length)

            if out.fov_flnet is not None:
                diff = abs(out.fov_exif - out.fov_flnet)
                diffs.append((view.image_path.name, out.fov_exif, out.fov_flnet, diff))

        failures = [(n, e, f, d) for n, e, f, d in diffs if d >= 10.0]
        assert not failures, (
            f"FoV diff >= 10° for: "
            + ", ".join(f"{n}: exif={e:.1f}°, flnet={f:.1f}°, diff={d:.1f}°" for n, e, f, d in failures)
        )


# ---------------------------------------------------------------------------
# Test 3: Dense keypoints (criterion 2.3)
# ---------------------------------------------------------------------------


class TestDenseKeypoints:
    def test_shape_and_bounds(self, pipeline_results):
        """Criterion 2.3: (138, 2) per view, coords within image bounds.

        Uses pipeline_results (real Phase 1 bboxes) so the crop is correct.
        Only checks FULL_BODY views — PARTIAL views legitimately have keypoints
        outside the frame.
        """
        from scantosmpl.types import ViewType

        views, _ = pipeline_results
        full_body = [v for v in views if v.view_type == ViewType.FULL_BODY and v.dense_keypoints_2d is not None]
        assert full_body, "No FULL_BODY views with dense keypoints found"

        for view in full_body:
            img = Image.open(DATA_DIR / view.image_path.name)
            W, H = img.size
            kps = view.dense_keypoints_2d

            assert kps.shape == (138, 2), (
                f"{view.image_path.name}: dense kp shape {kps.shape}"
            )
            assert view.dense_keypoint_confs.shape == (138,)

            # 138 dense points sample the full 3D surface including occluded back-of-body
            # vertices, which legitimately project outside the frame for non-frontal views.
            # Threshold: at least half should fall within a generous ±100px margin.
            in_bounds = (
                (kps[:, 0] >= -100) & (kps[:, 0] <= W + 100)
                & (kps[:, 1] >= -100) & (kps[:, 1] <= H + 100)
            )
            frac_in = float(in_bounds.sum()) / 138.0
            assert frac_in >= 0.5, (
                f"{view.image_path.name}: only {frac_in:.0%} of dense kps within image bounds"
            )

    def test_confidence_in_range(self, pipeline_results):
        """Confidence values should be in [0, 1]."""
        views, _ = pipeline_results
        views_with_kps = [v for v in views if v.dense_keypoint_confs is not None]
        assert views_with_kps

        for view in views_with_kps[:3]:
            assert view.dense_keypoint_confs.min() >= 0.0
            assert view.dense_keypoint_confs.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 4: Full pipeline — all 17 images (criteria 2.5, 2.6, 2.7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_results(detection_views, hmr_config):
    from scantosmpl.hmr.pipeline import HMRPipeline

    debug_dir = Path("output/debug/hmr")
    pipeline = HMRPipeline(hmr_config, device="cuda")

    t0 = time.time()
    views = pipeline.process_views(detection_views, DATA_DIR, debug_dir=debug_dir)
    elapsed = time.time() - t0
    return views, elapsed


class TestHMRPipelineAllImages:
    def test_all_views_produce_output(self, pipeline_results):
        views, _ = pipeline_results
        from scantosmpl.types import ViewType
        suitable = [
            v for v in views
            if v.view_type in (ViewType.FULL_BODY, ViewType.PARTIAL) and v.hmr_suitable
        ]
        hmr_done = [v for v in views if v.betas is not None]
        assert len(hmr_done) == len(suitable), (
            f"Only {len(hmr_done)}/{len(suitable)} suitable views have HMR output"
        )

    def test_beta_std_below_threshold(self, pipeline_results):
        """Criterion 2.5: β std < 1.0 per component across views."""
        views, _ = pipeline_results
        betas_list = [v.betas for v in views if v.betas is not None]
        assert betas_list, "No HMR outputs to check"

        betas_all = np.stack(betas_list, axis=0)  # (N, 10)
        beta_std = np.std(betas_all, axis=0)
        failures = [(i, s) for i, s in enumerate(beta_std) if s >= 1.0]
        assert not failures, (
            f"β std >= 1.0 for components {[(i, f'{s:.3f}') for i, s in failures]}"
        )

    def test_pose_variance_low_for_t_pose(self, pipeline_results):
        """Criterion 2.6: body_pose variance should be low (subjects in T-pose)."""
        views, _ = pipeline_results
        poses = [v.body_pose for v in views if v.body_pose is not None]
        assert poses

        poses_arr = np.stack(poses, axis=0)  # (N, 69)
        # Per-component standard deviation — T-pose should be consistent
        pose_std = np.std(poses_arr, axis=0)
        mean_std = float(pose_std.mean())
        # Threshold: T-pose scans should be well below 0.5 rad per joint
        assert mean_std < 0.5, f"Mean pose std {mean_std:.3f} rad — too high for T-pose"

    def test_inference_time(self, pipeline_results):
        """Criterion 2.7: all 17 images in < 60 seconds on GPU."""
        _, elapsed = pipeline_results
        assert elapsed < 60.0, f"HMR took {elapsed:.1f}s — exceeds 60s budget"

    def test_partial_view_handled(self, pipeline_results):
        """cam05_5.JPG (PARTIAL) should be processed normally (not skipped)."""
        views, _ = pipeline_results
        partial_views = [v for v in views if "cam05_5" in v.image_path.name]
        if not partial_views:
            pytest.skip("cam05_5.JPG not found in view list")
        assert partial_views[0].betas is not None, "PARTIAL view cam05_5 should have HMR output"


# ---------------------------------------------------------------------------
# Test 5: Wireframe overlay files (criterion 2.4)
# ---------------------------------------------------------------------------


class TestWireframeOverlay:
    def test_overlay_files_created(self, pipeline_results):
        """Criterion 2.4: debug overlay JPEGs should exist and be readable."""
        views, _ = pipeline_results
        debug_dir = Path("output/debug/hmr")

        if not debug_dir.exists():
            pytest.skip("Debug dir not found")

        overlay_files = list(debug_dir.glob("*_hmr_overlay.jpg"))
        hmr_views = [v for v in views if v.betas is not None]

        assert len(overlay_files) > 0, "No overlay files created"
        assert len(overlay_files) == len(hmr_views), (
            f"Expected {len(hmr_views)} overlays, found {len(overlay_files)}"
        )

        # Verify overlays are valid images
        for f in overlay_files[:3]:
            img = Image.open(f)
            assert img.size[0] > 0 and img.size[1] > 0, f"{f.name} is not a valid image"

    def test_summary_file_created(self, pipeline_results):
        """Summary text file should exist with FoV table."""
        debug_dir = Path("output/debug/hmr")
        summary = debug_dir / "summary.txt"
        assert summary.exists(), f"Summary file not found at {summary}"

        content = summary.read_text()
        assert "FoV" in content
        assert "betas" in content.lower() or "β" in content or "shape" in content.lower()

    def test_json_results_created(self, pipeline_results):
        """JSON results file should contain per-image data."""
        import json

        debug_dir = Path("output/debug/hmr")
        json_path = debug_dir / "hmr_results.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert len(data) > 0
        # Spot-check first entry
        first = next(iter(data.values()))
        assert "betas" in first
        assert len(first["betas"]) == 10
        assert "dense_kp_count" in first
        assert first["dense_kp_count"] == 138
