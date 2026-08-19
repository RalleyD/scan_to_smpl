"""Synthetic Tier 3 known-answer fixture generator (master §7.3, seed 0).

Builds a deterministic "Meshroom-like" point cloud, in an arbitrary source frame, from
the NEUTRAL SMPL mesh (betas=0, body_pose=0) — the canonical, trivially reproducible
stand-in for "the Tier 2 mesh" any consumer can regenerate with just
`SMPLModel(model_dir="models/smpl", gender="neutral")`, no fixture-specific numbers
needed. `cloud.ply` and `ground_truth.json` are the only two files this script writes;
both are committed so a later reader never needs to re-run this script (which DOES
need SMPL weights) just to *read* the fixture — only to regenerate it.

Construction, per master §7.3:
  1. Inject a known "clothing" offset (`D_true`): +4 mm along the outward VERTEX
     normal for every vertex whose D7 part label is "torso", 0 elsewhere.
  2. Area-weighted uniform sample of 60,000 points on the OFFSET mesh's surface (so
     the clothing bump is a genuine surface feature, not an add-on).
  3. Gaussian noise, sigma = 1 mm, along each sample's local (flat, per-face) surface
     normal — Meshroom-like reconstruction noise.
  4. 2% (of the 60,000 inliers) uniform outliers inside the noisy inlier bounding
     box — exercises `preprocess_cloud`'s statistical outlier removal.
  5. A known similarity, smpl_world -> source (a genuinely Meshroom-like arbitrary
     frame): ``p_source = scale_true * (rotation_true @ p_smpl) + translation_true``,
     with ``scale_true = 0.371``, ``rotation_true`` ~137 deg about the unit vector
     along (0.3, -0.5, 0.8), ``translation_true = (1.7, -0.4, 2.3)`` (source/arbitrary
     units — this is literally "apply a known similarity" to *corrupt* the true
     metric points into an arbitrary reconstruction frame).

`ground_truth.json` records this known similarity's OWN INVERSE (source ->
smpl_world) — i.e. exactly the values `pointcloud.align.align_cloud_to_smpl`'s
`CloudAlignment` is expected to recover — plus `D_true` statistics, the noise sigma
and the outlier fraction, so alignment recovery, chamfer floor and D recovery are all
assertable exactly (master D11: a known-answer test, not a regression snapshot).

Determinism (master D12): one `numpy.random.default_rng(seed)` instance, consumed in
a fixed order (surface sampling -> noise -> outliers). The SMPL forward pass is a
pure computation on CPU (no dropout / stochastic layers), so running this script
twice with the same seed against the same weights produces byte-identical
`cloud.ply` / `ground_truth.json` — verified by
`tests/integration/test_tier3_integration.py::test_fixture_generator_is_deterministic`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scantosmpl.pointcloud.io import PointCloud, save_pointcloud
from scantosmpl.pointcloud.segment import SMPL_PART_GROUPS, smpl_part_labels
from scantosmpl.smpl.model import SMPLModel

SEED = 0
N_SURFACE_SAMPLES = 60_000
CLOTHING_OFFSET_M = 0.004
NOISE_SIGMA_M = 0.001
OUTLIER_FRACTION_OF_INLIERS = 0.02

SCALE_TRUE = 0.371
ROTATION_AXIS = np.array([0.3, -0.5, 0.8])
ROTATION_ANGLE_DEG = 137.0
TRANSLATION_TRUE = np.array([1.7, -0.4, 2.3])

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]  # synthetic_cloud -> fixtures -> integration -> tests -> root
_SMPL_DIR = _REPO_ROOT / "models" / "smpl"


def _rotation_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(unit * np.deg2rad(angle_deg)).as_matrix()


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (public-domain algorithm, local
    reimplementation — same pattern as `tests/test_surface_fitting.py::_vertex_normals_np`)."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return normals / norms


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return n / norms


def _sample_surface_with_normals(
    vertices: np.ndarray, faces: np.ndarray, *, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted uniform surface samples + each sample's flat face normal.

    Same inverse-CDF face choice / reflected-barycentric algorithm as
    `scantosmpl.evaluation.surface_metrics.sample_surface`, extended to also return
    the per-sample flat face normal (needed for noise injection below;
    `sample_surface`'s public signature does not expose it).
    """
    areas = _face_areas(vertices, faces)
    total_area = float(areas.sum())
    cumulative = np.cumsum(areas)
    picks = np.searchsorted(cumulative, rng.random(n_samples) * total_area, side="right")
    picks = np.clip(picks, 0, faces.shape[0] - 1)

    uv = rng.random((n_samples, 2))
    outside = uv.sum(axis=1) > 1.0
    uv[outside] = 1.0 - uv[outside]

    a = vertices[faces[picks, 0]]
    b = vertices[faces[picks, 1]]
    c = vertices[faces[picks, 2]]
    u, v = uv[:, 0:1], uv[:, 1:2]
    points = a + u * (b - a) + v * (c - a)

    face_normals = _face_normals(vertices, faces)
    normals = face_normals[picks]
    return points, normals


def build_fixture(seed: int = SEED) -> tuple[np.ndarray, dict]:
    """Return `(cloud_points_source_frame (N, 3) float64, ground_truth dict)`.

    A pure function of `seed` and the on-disk SMPL weights — no other file I/O.

    Raises:
        FileNotFoundError: If `models/smpl/SMPL_NEUTRAL.pkl` is missing. The already
            -committed `cloud.ply` / `ground_truth.json` do NOT need this to be READ;
            only `write_fixture` (regeneration) does.
    """
    if not (_SMPL_DIR / "SMPL_NEUTRAL.pkl").exists():
        raise FileNotFoundError(
            f"SMPL model files not found in {_SMPL_DIR} — this fixture generator needs "
            "models/smpl/SMPL_NEUTRAL.pkl (see models/README.md) to regenerate the "
            "fixture. The already-committed cloud.ply / ground_truth.json do not need "
            "this file to be read."
        )

    model = SMPLModel(model_dir=_SMPL_DIR, gender="neutral", device="cpu")
    n_body = model.body_model.NUM_BODY_JOINTS * 3
    with torch.no_grad():
        out = model.forward(
            betas=torch.zeros(1, SMPLModel.NUM_BETAS),
            body_pose=torch.zeros(1, n_body),
            global_orient=torch.zeros(1, 3),
            translation=torch.zeros(1, 3),
            scale=torch.ones(1),
            apply_displacements=False,
        )
    base_vertices = out.vertices.squeeze(0).numpy().astype(np.float64)
    faces = model.body_model.faces.astype(np.int64)

    lbs_weights = model.body_model.lbs_weights.detach().cpu().numpy()
    vertex_labels = smpl_part_labels(lbs_weights)
    torso_id = list(SMPL_PART_GROUPS).index("torso")
    torso_mask = vertex_labels == torso_id

    vertex_normals = _vertex_normals(base_vertices, faces)
    d_true = np.zeros_like(base_vertices)
    d_true[torso_mask] = CLOTHING_OFFSET_M * vertex_normals[torso_mask]
    target_vertices = base_vertices + d_true

    rng = np.random.default_rng(seed)

    surface_points, surface_normals = _sample_surface_with_normals(
        target_vertices, faces, n_samples=N_SURFACE_SAMPLES, rng=rng
    )
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA_M, size=N_SURFACE_SAMPLES)
    noisy_points = surface_points + noise[:, None] * surface_normals

    n_outliers = int(round(OUTLIER_FRACTION_OF_INLIERS * N_SURFACE_SAMPLES))
    bbox_min = noisy_points.min(axis=0)
    bbox_max = noisy_points.max(axis=0)
    outliers = rng.uniform(bbox_min, bbox_max, size=(n_outliers, 3))

    smpl_world_points = np.concatenate([noisy_points, outliers], axis=0)

    # Step 5 — "apply a known similarity" smpl_world -> source (the Meshroom-like
    # corruption). Row-vector form of p_source = scale*(R @ p_smpl) + t.
    rotation_true = _rotation_from_axis_angle(ROTATION_AXIS, ROTATION_ANGLE_DEG)
    source_points = SCALE_TRUE * (smpl_world_points @ rotation_true.T) + TRANSLATION_TRUE

    # The known similarity's OWN inverse (source -> smpl_world) — exactly what
    # `align_cloud_to_smpl` is expected to recover as its `CloudAlignment`.
    rotation_inv = rotation_true.T
    scale_inv = 1.0 / SCALE_TRUE
    translation_inv = -scale_inv * (rotation_true.T @ TRANSLATION_TRUE)

    n_total = int(smpl_world_points.shape[0])
    ground_truth = {
        "seed": seed,
        "n_inliers": N_SURFACE_SAMPLES,
        "n_surface_samples": N_SURFACE_SAMPLES,
        "n_outliers": n_outliers,
        "n_total_points": n_total,
        "outlier_fraction": n_outliers / n_total,
        "noise_sigma_m": NOISE_SIGMA_M,
        "d_true": {
            "offset_mm": CLOTHING_OFFSET_M * 1000.0,
            "part": "torso",
            "n_affected_vertices": int(torso_mask.sum()),
            "n_total_vertices": int(base_vertices.shape[0]),
            "mean_mm": float(np.linalg.norm(d_true, axis=1).mean() * 1000.0),
            "max_mm": float(np.linalg.norm(d_true, axis=1).max() * 1000.0),
        },
        "known_similarity_smpl_world_to_source": {
            "scale": SCALE_TRUE,
            "rotation": rotation_true.tolist(),
            "translation": TRANSLATION_TRUE.tolist(),
            "rotation_axis": ROTATION_AXIS.tolist(),
            "rotation_angle_deg": ROTATION_ANGLE_DEG,
        },
        "inverse_similarity_source_to_smpl_world": {
            "scale": scale_inv,
            "rotation": rotation_inv.tolist(),
            "translation": translation_inv.tolist(),
        },
        "reference_mesh": {
            "description": (
                "Neutral SMPL mesh (betas=0, body_pose=0, global_orient=0, "
                "translation=0, scale=1) — reproducible from SMPLModel with no "
                "fixture-specific numbers. This mesh PLUS d_true is what the cloud "
                "(before noise/outliers/similarity) was sampled from."
            ),
            "gender": "neutral",
            "num_betas": SMPLModel.NUM_BETAS,
        },
    }
    return source_points, ground_truth


def write_fixture(seed: int = SEED, out_dir: Path = _THIS_DIR) -> None:
    """Regenerate `cloud.ply` + `ground_truth.json` in `out_dir` (default: this
    script's own directory)."""
    points, ground_truth = build_fixture(seed)
    out_dir = Path(out_dir)
    cloud_path = out_dir / "cloud.ply"
    cloud = PointCloud(points=points, normals=None, colors=None, source_path=cloud_path)
    save_pointcloud(cloud, cloud_path)

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote {points.shape[0]} points to {cloud_path}")
    print(f"Wrote ground truth to {out_dir / 'ground_truth.json'}")


if __name__ == "__main__":
    write_fixture()
