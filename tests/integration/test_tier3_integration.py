"""Tier 3 end-to-end integration tests.

Discharges AC5, AC6, AC8, AC9, AC10, AC11, AC13, AC18 (master §10) plus the fixture's
own determinism check (brief step 3). The synthetic fixture
(`tests/integration/fixtures/synthetic_cloud/`) is a KNOWN-ANSWER test (master D11):
alignment recovery, chamfer improvement and D recovery are all assertable exactly
against `ground_truth.json`, not just "did it run".

Run with:
    pytest tests/integration/test_tier3_integration.py -v --timeout=900
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from scantosmpl.cli import _load_tier2_result
from scantosmpl.config import Tier3Config
from scantosmpl.evaluation.surface_metrics import chamfer_report, point_to_surface_distances
from scantosmpl.fitting.optimiser import RefinementResult
from scantosmpl.fitting.surface import Tier3SurfaceFitter
from scantosmpl.fitting.surface_pipeline import Tier3Pipeline
from scantosmpl.pointcloud.align import align_cloud_to_smpl
from scantosmpl.pointcloud.io import PointCloud, load_pointcloud
from scantosmpl.pointcloud.preprocess import preprocess_cloud
from scantosmpl.pointcloud.segment import (
    SMPL_PART_GROUPS,
    smpl_part_labels,
    transfer_labels_to_cloud,
)
from scantosmpl.smpl.model import SMPLModel

pytestmark = pytest.mark.slow

SMPL_DIR = "models/smpl"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic_cloud"
CLOUD_PATH = FIXTURE_DIR / "cloud.ply"
GROUND_TRUTH_PATH = FIXTURE_DIR / "ground_truth.json"
REAL_CLOUD_PATH = Path("data/t-pose/pointcloud.ply")
TIER2_DEBUG_DIR = Path("output/debug/refinement")

#: AC9's synthetic bound (D11): 1mm injected noise + residual fit error.
AC9_SYNTHETIC_CLOUD_TO_MESH_MEAN_MM_BOUND = 3.0
#: AC8's minimum improvement of the refined fit over the Tier-2-params baseline.
AC8_MIN_IMPROVEMENT_FRACTION = 0.40
#: AC18's decisive similarity-invariance tolerance.
AC18_D_MEAN_TOLERANCE_MM = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _smpl_available() -> bool:
    return (Path(SMPL_DIR) / "SMPL_NEUTRAL.pkl").exists()


requires_smpl = pytest.mark.skipif(
    not _smpl_available(),
    reason=f"SMPL model files not found in {SMPL_DIR}/ — see models/README.md",
)

requires_fixture = pytest.mark.skipif(
    not (CLOUD_PATH.exists() and GROUND_TRUTH_PATH.exists()),
    reason=f"Synthetic fixture not found under {FIXTURE_DIR} — run make_fixture.py",
)


def _load_make_fixture_module():
    """Load `make_fixture.py` by file path — no `__init__.py` needed under
    `tests/integration/fixtures/` (that directory is data, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "tier3_make_fixture", FIXTURE_DIR / "make_fixture.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _geodesic_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    relative = r1.T @ r2
    cos_theta = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _joint_distance(joints: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(joints[a] - joints[b]))


def _waist_girth_proxy(vertices: np.ndarray, joints: np.ndarray) -> float:
    """A cheap monotonic proxy for waist girth: the mean horizontal (XZ-plane)
    radius of vertices near the pelvis joint's height. Not a true circumference —
    good enough to detect "did the torso get wider/narrower toward the truth"."""
    pelvis_y = joints[0, 1]
    band = np.abs(vertices[:, 1] - pelvis_y) < 0.03
    pts = vertices[band]
    if pts.shape[0] == 0:
        return float("nan")
    centroid_xz = pts[:, [0, 2]].mean(axis=0)
    radii = np.linalg.norm(pts[:, [0, 2]] - centroid_xz, axis=1)
    return float(radii.mean())


@pytest.fixture(scope="module")
def ground_truth() -> dict:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def smpl_model() -> SMPLModel:
    return SMPLModel(model_dir=SMPL_DIR, gender="neutral", device=DEVICE)


def _forward_vertices_joints(
    smpl_model: SMPLModel,
    betas: np.ndarray,
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    device = smpl_model.device
    with torch.no_grad():
        out = smpl_model.forward(
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(1, -1),
            body_pose=torch.as_tensor(body_pose, dtype=torch.float32, device=device).reshape(1, -1),
            global_orient=torch.as_tensor(
                global_orient, dtype=torch.float32, device=device
            ).reshape(1, -1),
            translation=torch.as_tensor(translation, dtype=torch.float32, device=device).reshape(
                1, -1
            ),
            scale=torch.tensor([scale], dtype=torch.float32, device=device),
            apply_displacements=False,
        )
    vertices = out.vertices.squeeze(0).cpu().numpy().astype(np.float64)
    joints = out.joints.squeeze(0).cpu().numpy().astype(np.float64)
    return vertices, joints


def _true_vertices_and_faces(smpl_model: SMPLModel) -> tuple[np.ndarray, np.ndarray]:
    """The fixture's reference mesh — neutral SMPL (betas=0, body_pose=0) — matching
    `ground_truth["reference_mesh"]`."""
    n_body = smpl_model.body_model.NUM_BODY_JOINTS * 3
    vertices, _ = _forward_vertices_joints(
        smpl_model, np.zeros(10), np.zeros(n_body), np.zeros(3), np.zeros(3), 1.0
    )
    faces = smpl_model.body_model.faces.astype(np.int64)
    return vertices, faces


def _perturbed_tier2(smpl_model: SMPLModel, *, seed: int) -> RefinementResult:
    """A plausible, deliberately IMPERFECT Tier 2 initial guess — NOT the fixture's
    true (betas=0, body_pose=0) shape/pose, so AC8/AC10/AC11's "moves toward truth"
    checks have real initial error to correct (mirrors
    `tests/test_surface_fitting.py`'s own construction pattern)."""
    n_body = smpl_model.body_model.NUM_BODY_JOINTS * 3
    gen = torch.Generator(device="cpu").manual_seed(seed)
    betas = (torch.randn(1, SMPLModel.NUM_BETAS, generator=gen) * 0.12).squeeze(0).numpy()
    body_pose = (torch.randn(1, n_body, generator=gen) * 0.02).squeeze(0).numpy()
    betas = betas.astype(np.float32)
    body_pose = body_pose.astype(np.float32)
    global_orient = np.zeros(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)

    vertices, joints = _forward_vertices_joints(
        smpl_model, betas, body_pose, global_orient, translation, 1.0
    )
    return RefinementResult(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        translation=translation,
        scale=1.0,
        vertices=vertices,
        joints=joints,
        metrics={"pa_mpjpe_mm": 32.0, "median_reproj_px": 45.0},
    )


def _run_s1_s2_s3(
    smpl_model: SMPLModel,
    cfg: Tier3Config,
    tier2: RefinementResult,
    cloud_path: Path,
    *,
    locked_betas: np.ndarray | None = None,
):
    raw = load_pointcloud(cloud_path)
    cleaned, preprocess_stats = preprocess_cloud(raw, cfg)
    faces = smpl_model.body_model.faces.astype(np.int64)
    aligned, alignment = align_cloud_to_smpl(cleaned, tier2.vertices, faces, cfg)
    fitter = Tier3SurfaceFitter(smpl_model, cfg)
    fit = fitter.fit(tier2, aligned, locked_betas=locked_betas)
    return fit, alignment, aligned, preprocess_stats


# ---------------------------------------------------------------------------
# Fixture generator determinism (brief step 3's verification)
# ---------------------------------------------------------------------------


@requires_smpl
def test_fixture_generator_is_deterministic():
    make_fixture = _load_make_fixture_module()
    points_a, gt_a = make_fixture.build_fixture(seed=0)
    points_b, gt_b = make_fixture.build_fixture(seed=0)
    assert np.array_equal(points_a, points_b)
    assert gt_a == gt_b


# ---------------------------------------------------------------------------
# AC5 — alignment recovers a known similarity
# ---------------------------------------------------------------------------


@requires_smpl
@requires_fixture
def test_alignment_recovers_ground_truth(smpl_model, ground_truth):
    true_vertices, faces = _true_vertices_and_faces(smpl_model)
    raw_cloud = load_pointcloud(CLOUD_PATH)
    cfg = Tier3Config(target_points=8000)
    cleaned, _ = preprocess_cloud(raw_cloud, cfg)
    _, alignment = align_cloud_to_smpl(cleaned, true_vertices, faces, cfg)

    inv = ground_truth["inverse_similarity_source_to_smpl_world"]
    scale_true = inv["scale"]
    rotation_true = np.array(inv["rotation"])
    translation_true = np.array(inv["translation"])

    scale_rel_err = abs(alignment.scale / scale_true - 1.0)
    rot_err_deg = _geodesic_angle_deg(alignment.rotation, rotation_true)
    trans_err_m = float(np.linalg.norm(alignment.translation - translation_true))

    assert scale_rel_err < 0.01, f"scale rel err {scale_rel_err}"
    assert rot_err_deg < 1.0, f"rotation err {rot_err_deg} deg"
    assert trans_err_m < 0.005, f"translation err {trans_err_m} m"
    assert alignment.converged is True


# ---------------------------------------------------------------------------
# AC6 — outlier removal keeps the body
# ---------------------------------------------------------------------------


@requires_fixture
def test_preprocess_removes_outliers(ground_truth):
    raw_cloud = load_pointcloud(CLOUD_PATH)
    cfg = Tier3Config(target_points=0)  # isolate outlier removal (no downsample)
    _, stats = preprocess_cloud(raw_cloud, cfg)

    injected_outlier_fraction = ground_truth["outlier_fraction"]
    n_inliers = ground_truth["n_inliers"]

    assert stats.outlier_fraction >= 0.8 * injected_outlier_fraction
    assert stats.n_after_outlier_removal >= 0.95 * n_inliers


# ---------------------------------------------------------------------------
# AC8 — refinement improves on Tier 2 by >= 40%; AC9's synthetic bound (D11)
# ---------------------------------------------------------------------------


@requires_smpl
@requires_fixture
def test_refinement_improves_over_tier2(smpl_model, tmp_path):
    cfg = Tier3Config(target_points=6000, use_semantic_weighting=False)
    tier2 = _perturbed_tier2(smpl_model, seed=1)
    faces = smpl_model.body_model.faces.astype(np.int64)

    fit, alignment, aligned, _ = _run_s1_s2_s3(smpl_model, cfg, tier2, CLOUD_PATH)

    baseline_report = chamfer_report(aligned.points, tier2.vertices, faces, cfg)
    final_report = chamfer_report(aligned.points, fit.vertices, faces, cfg)

    baseline_mean = baseline_report.cloud_to_mesh_mm["mean"]
    final_mean = final_report.cloud_to_mesh_mm["mean"]
    improvement = (baseline_mean - final_mean) / baseline_mean

    debug_dir = Path("output/debug/surface")
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "ac8_before_after.json", "w") as f:
        json.dump(
            {
                "baseline_tier2_d0_cloud_to_mesh_mean_mm": baseline_mean,
                "final_tier3_cloud_to_mesh_mean_mm": final_mean,
                "improvement_fraction": improvement,
            },
            f,
            indent=2,
        )

    assert improvement >= AC8_MIN_IMPROVEMENT_FRACTION, (
        f"only {improvement:.1%} improvement ({baseline_mean:.2f} -> {final_mean:.2f} mm)"
    )
    # AC9 / master D11 — the synthetic-fixture equivalent of the (deferred) real-cloud
    # gate, asserted unconditionally regardless of whether data/t-pose/pointcloud.ply
    # exists.
    assert final_mean < AC9_SYNTHETIC_CLOUD_TO_MESH_MEAN_MM_BOUND, (
        f"cloud_to_mesh_mean_mm={final_mean:.2f} exceeds the synthetic bound "
        f"{AC9_SYNTHETIC_CLOUD_TO_MESH_MEAN_MM_BOUND}"
    )


# ---------------------------------------------------------------------------
# AC9 — deferred real-cloud gate
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_CLOUD_PATH.exists(),
    reason=f"Real scanner point cloud not found at {REAL_CLOUD_PATH} — Tier 3 gate is DEFERRED",
)
@requires_smpl
def test_real_cloud_chamfer(smpl_model, tmp_path):
    """AC9 (7.1) — deferred gate. On real scanner data, cloud_to_mesh_mean_mm < 8.0."""
    tier2 = _load_tier2_result(TIER2_DEBUG_DIR, smpl_model)
    cfg = Tier3Config(subject_id="real-cloud-test")
    pipeline = Tier3Pipeline(smpl_model, cfg)
    result = pipeline.run(
        tier2, REAL_CLOUD_PATH, pose_name="t-pose", output_dir=tmp_path / "real_cloud_fits"
    )
    assert result.quality.chamfer_cloud_to_mesh_mean_mm < 8.0


# ---------------------------------------------------------------------------
# AC10 — semantic weighting beats uniform on the torso
# ---------------------------------------------------------------------------


@requires_smpl
@requires_fixture
def test_semantic_weighting_ab(smpl_model):
    cfg_on = Tier3Config(target_points=6000, use_semantic_weighting=True)
    cfg_off = Tier3Config(target_points=6000, use_semantic_weighting=False)

    faces = smpl_model.body_model.faces.astype(np.int64)
    lbs_weights = smpl_model.body_model.lbs_weights.detach().cpu().numpy()
    vertex_labels = smpl_part_labels(lbs_weights)
    torso_id = list(SMPL_PART_GROUPS).index("torso")

    def _torso_cloud_to_mesh_mean_mm(cfg: Tier3Config) -> float:
        tier2 = _perturbed_tier2(smpl_model, seed=2)  # identical init both times
        fit, _, aligned, _ = _run_s1_s2_s3(smpl_model, cfg, tier2, CLOUD_PATH)
        cloud_labels = transfer_labels_to_cloud(aligned.points, tier2.vertices, vertex_labels)
        torso_points = aligned.points[cloud_labels == torso_id]
        distances_m = point_to_surface_distances(torso_points, fit.vertices, faces)
        return float(np.mean(distances_m) * 1000.0)

    mean_weighted = _torso_cloud_to_mesh_mean_mm(cfg_on)
    mean_uniform = _torso_cloud_to_mesh_mean_mm(cfg_off)

    debug_dir = Path("output/debug/surface")
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "semantic_ab.json", "w") as f:
        json.dump(
            {
                "torso_cloud_to_mesh_mean_mm_weighted": mean_weighted,
                "torso_cloud_to_mesh_mean_mm_uniform": mean_uniform,
            },
            f,
            indent=2,
        )

    assert mean_weighted < mean_uniform, (
        f"semantic weighting ({mean_weighted:.2f}mm) did not beat uniform "
        f"({mean_uniform:.2f}mm) on the torso"
    )


# ---------------------------------------------------------------------------
# AC11 — beta refinement improves proportions (lock_betas=False ONLY)
# ---------------------------------------------------------------------------


@requires_smpl
@requires_fixture
def test_beta_refinement_improves_proportions(smpl_model):
    true_vertices, faces = _true_vertices_and_faces(smpl_model)
    n_body = smpl_model.body_model.NUM_BODY_JOINTS * 3
    _, true_joints = _forward_vertices_joints(
        smpl_model, np.zeros(10), np.zeros(n_body), np.zeros(3), np.zeros(3), 1.0
    )
    true_shoulder = _joint_distance(true_joints, 16, 17)
    true_waist = _waist_girth_proxy(true_vertices, true_joints)

    tier2 = _perturbed_tier2(smpl_model, seed=3)
    tier2_shoulder = _joint_distance(tier2.joints, 16, 17)
    tier2_waist = _waist_girth_proxy(tier2.vertices, tier2.joints)

    cfg = Tier3Config(target_points=6000, lock_betas=False, use_semantic_weighting=False)
    fit, _, _, _ = _run_s1_s2_s3(smpl_model, cfg, tier2, CLOUD_PATH)
    fitted_vertices, fitted_joints = _forward_vertices_joints(
        smpl_model, fit.betas, fit.body_pose, fit.global_orient, fit.translation, fit.scale
    )
    fitted_shoulder = _joint_distance(fitted_joints, 16, 17)
    fitted_waist = _waist_girth_proxy(fitted_vertices, fitted_joints)

    shoulder_before = abs(tier2_shoulder - true_shoulder)
    shoulder_after = abs(fitted_shoulder - true_shoulder)
    waist_before = abs(tier2_waist - true_waist)
    waist_after = abs(fitted_waist - true_waist)

    debug_dir = Path("output/debug/surface")
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "summary.txt", "a") as f:
        f.write(
            "\n\n=== AC11 beta-refinement proportions "
            "(test_beta_refinement_improves_proportions) ===\n"
            f"shoulder width |fit-true| before={shoulder_before * 1000:.2f}mm "
            f"after={shoulder_after * 1000:.2f}mm\n"
            f"waist proxy    |fit-true| before={waist_before * 1000:.2f}mm "
            f"after={waist_after * 1000:.2f}mm\n"
        )

    assert fit.betas_locked is False
    assert shoulder_after < shoulder_before, (
        f"shoulder width did not move toward truth: before={shoulder_before * 1000:.2f}mm "
        f"after={shoulder_after * 1000:.2f}mm"
    )
    assert waist_after < waist_before, (
        f"waist proxy did not move toward truth: before={waist_before * 1000:.2f}mm "
        f"after={waist_after * 1000:.2f}mm"
    )


@requires_smpl
@requires_fixture
def test_beta_refinement_ac11_inapplicable_when_locked(smpl_model):
    """AC11 is explicitly INAPPLICABLE in `--lock-betas` mode (master D10 / REVIEW.md's
    7.4-vs-7.B1 resolution) — asserted directly here, not skipped silently: betas are
    frozen bit-for-bit at the locked value, so "did proportions move toward truth" is
    not even a well-formed question for this run."""
    tier2 = _perturbed_tier2(smpl_model, seed=4)
    locked_betas = np.zeros(10, dtype=np.float64)  # the fixture's true betas

    cfg = Tier3Config(target_points=2000, lock_betas=True, use_semantic_weighting=False)
    fit, _, _, _ = _run_s1_s2_s3(smpl_model, cfg, tier2, CLOUD_PATH, locked_betas=locked_betas)

    assert fit.betas_locked is True
    assert np.array_equal(fit.betas, locked_betas)  # exact — beta genuinely could not move
    # AC11 ("beta refinement improves proportions") is therefore inapplicable to this
    # run BY CONSTRUCTION: there is no beta-space delta this run could ever produce.


# ---------------------------------------------------------------------------
# AC13 — optimisation < 60s on GPU with a 50K cloud
# ---------------------------------------------------------------------------


@requires_smpl
@requires_fixture
@pytest.mark.gpu
def test_optimisation_under_60s(smpl_model):
    if not torch.cuda.is_available():
        pytest.skip("GPU required for the AC13 wall-clock measurement")

    tier2 = _perturbed_tier2(smpl_model, seed=5)
    cfg = Tier3Config()  # literal defaults — target_points=50_000
    faces = smpl_model.body_model.faces.astype(np.int64)

    raw = load_pointcloud(CLOUD_PATH)
    cleaned, _ = preprocess_cloud(raw, cfg)
    aligned, _ = align_cloud_to_smpl(cleaned, tier2.vertices, faces, cfg)

    fitter = Tier3SurfaceFitter(smpl_model, cfg)
    torch.cuda.synchronize()
    start = time.perf_counter()
    fitter.fit(tier2, aligned)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    debug_dir = Path("output/debug/surface")
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "summary.txt", "a") as f:
        f.write(
            f"\n\n=== AC13 wall clock ===\nS2+S3 at {aligned.n_points} points: {elapsed:.1f}s\n"
        )

    assert elapsed < 60.0, f"S2+S3 took {elapsed:.1f}s (budget: 60s) at {aligned.n_points} points"


# ---------------------------------------------------------------------------
# AC18 — the decisive 7.B5 similarity-invariance check
# ---------------------------------------------------------------------------


@requires_smpl
@requires_fixture
def test_similarity_invariance(smpl_model):
    """AC18 (7.B5) — re-run with the cloud pre-multiplied by a DIFFERENT known
    similarity; D must match to within 0.5mm mean while the two `CloudAlignment`s
    absorb the whole difference. If this fails, a transform has leaked into D and
    the PSD residual is corrupt."""
    tier2_seed = 6
    cfg = Tier3Config(target_points=5000, use_semantic_weighting=False)
    faces = smpl_model.body_model.faces.astype(np.int64)

    # Run A: the fixture's own cloud, as-is.
    tier2_a = _perturbed_tier2(smpl_model, seed=tier2_seed)
    raw_a = load_pointcloud(CLOUD_PATH)
    cleaned_a, _ = preprocess_cloud(raw_a, cfg)
    aligned_a, alignment_a = align_cloud_to_smpl(cleaned_a, tier2_a.vertices, faces, cfg)
    fit_a = Tier3SurfaceFitter(smpl_model, cfg).fit(tier2_a, aligned_a)

    # Run B: the SAME source points, pre-multiplied by a genuinely DIFFERENT known
    # similarity (different scale, rotation axis/angle and translation from the
    # fixture's own "known similarity", master §7.3).
    axis_extra = np.array([0.1, 0.9, -0.3])
    rotation_extra = Rotation.from_rotvec(
        axis_extra / np.linalg.norm(axis_extra) * np.deg2rad(63.0)
    ).as_matrix()
    scale_extra = 4.2
    translation_extra = np.array([5.0, -3.0, 2.0])
    raw_b_points = scale_extra * (raw_a.points @ rotation_extra.T) + translation_extra
    raw_b = PointCloud(points=raw_b_points, normals=None, colors=None, source_path=CLOUD_PATH)

    tier2_b = _perturbed_tier2(smpl_model, seed=tier2_seed)  # identical init to run A
    cleaned_b, _ = preprocess_cloud(raw_b, cfg)
    aligned_b, alignment_b = align_cloud_to_smpl(cleaned_b, tier2_b.vertices, faces, cfg)
    fit_b = Tier3SurfaceFitter(smpl_model, cfg).fit(tier2_b, aligned_b)

    d_mean_diff_mm = float(np.mean(np.abs(fit_a.displacements - fit_b.displacements)) * 1000.0)
    alignment_scale_diff = abs(alignment_a.scale - alignment_b.scale)

    assert alignment_scale_diff > 0.01, (
        "the two alignments should differ substantially — otherwise this test isn't "
        "exercising anything"
    )
    assert d_mean_diff_mm < AC18_D_MEAN_TOLERANCE_MM, (
        f"D differs by {d_mean_diff_mm:.3f}mm mean between the two similarity-frame "
        f"re-runs — a transform has leaked into the displacement field (7.B5)"
    )
