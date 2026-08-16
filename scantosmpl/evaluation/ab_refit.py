"""A/B attribution refit — isolate the effect of W2 (head term) and W3 (view
weighting) on PA-MPJPE and torso/head geometry.

Runs four full-data refits off the *same* cached artefacts, differing only in
two flags:

    config      head term (W2)   profile weight (W3)
    --------    --------------   -------------------
    baseline    off              1.0   (pre-W2/W3 behaviour)
    W2_head     ON               1.0
    W3_weight   off              0.3
    W2W3        ON               0.3   (current default)

Because the triangulated joints are loaded from disk (no RANSAC re-run) and Adam
is deterministic, the four refits are byte-for-byte reproducible and differ ONLY
by the flags — so any change in a metric is *attributable* to W2, W3, or their
interaction, not to run-to-run noise.

Scope (agreed with the user): PA-MPJPE + chest-height + head geometry ONLY.
Torso *girth* (surface thickness) is deliberately NOT measured here — it is a
per-vertex/beta surface property that joint-centre losses cannot constrain from
any view, so it belongs to Tier-3 chamfer-vs-point-cloud, not to any Tier-2
weighting knob. A "torso width" column would show beta wander, not a W2/W3
effect, and would mislead.

Runs purely off the debug artefacts Phase 3/4/5 already wrote to disk (same
inputs as leave_one_view_out.py); it re-runs the light SMPL optimiser four
times. GPU used if available, CPU fine (slower).

Usage:
    python -m scantosmpl.evaluation.ab_refit
    python -m scantosmpl.evaluation.ab_refit --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from scantosmpl.evaluation.leave_one_view_out import (
    DEFAULT_CONSENSUS_DIR,
    DEFAULT_DETECTIONS,
    DEFAULT_REFINEMENT_DIR,
    DEFAULT_SMPL_MODEL_DIR,
    MIN_CONF,
    _load_cameras,
    _load_consensus,
    _load_detections,
    _load_triangulated_joints,
)
from scantosmpl.fitting.optimiser import RefinementResult, SMPLOptimiser
from scantosmpl.fitting.rear_views import classify_rear_views, classify_view_angles
from scantosmpl.smpl.joint_map import (
    COCO_TO_SMPL,
    HEAD_MIDPOINT_TO_SMPL,
    HEAD_MIDPOINT_TO_VERTEX,
    Smpl24Joint,
)
from scantosmpl.smpl.model import SMPLModel
from scantosmpl.utils.geometry import project_points

logger = logging.getLogger(__name__)

# SMPL-24 joints not enumerated in Smpl24Joint (chest / head landmarks).
_SPINE3 = 9  # upper spine / thorax — chest-height proxy
_HEAD = 15  # head joint

# Weight maps: rear always excluded (0.0); only the profile weight differs.
_W_FULL: dict[str, float] = {"frontal": 1.0, "three_quarter": 1.0, "profile": 1.0, "rear": 0.0}
_W_W3: dict[str, float] = {"frontal": 1.0, "three_quarter": 1.0, "profile": 0.3, "rear": 0.0}
_OFF_J: dict[tuple[int, int], int] = {}
_OFF_V: dict[tuple[int, int], tuple[int, int]] = {}
_NO_NAME: dict[str, float] = {}

# Targeted per-view rejection: the two views the LOVO diagnostic flags as genuine
# (unexplainable) outliers — cam06_4 (profile, ~324px reprojection) and cam02_5
# (profile, ~158px). Down-weight/drop ONLY these, keeping the good profile
# (cam02_4, ~51px) at full strength — the surgical alternative to W3's blanket
# profile-0.3, which suppresses good and bad profiles alike.
_TGT_DROP: dict[str, float] = {"cam06_4": 0.0, "cam02_5": 0.0}  # exclude outliers
_TGT_DOWN: dict[str, float] = {"cam06_4": 0.3, "cam02_5": 0.3}  # down-weight outliers

# (label, joint_midpoint (old head anchor), vertex_midpoint (new head anchor),
#  angle-class weights, per-view-name overrides).
# The first four configs isolate the head-anchor fix: baseline (no head term) ->
# joint-anchored (old, biased) -> vertex-anchored (new, fixed) -> vertex + W3
# (blanket profile downweight). The last two swap W3's blanket for targeted
# rejection of the two LOVO outliers, keeping profile at 1.0.
CONFIGS: list[
    tuple[
        str,
        dict[tuple[int, int], int],
        dict[tuple[int, int], tuple[int, int]],
        dict[str, float],
        dict[str, float],
    ]
] = [
    ("baseline", _OFF_J, _OFF_V, _W_FULL, _NO_NAME),
    ("W2_joint", HEAD_MIDPOINT_TO_SMPL, _OFF_V, _W_FULL, _NO_NAME),
    ("W2_vertex", _OFF_J, HEAD_MIDPOINT_TO_VERTEX, _W_FULL, _NO_NAME),
    ("W2v_W3", _OFF_J, HEAD_MIDPOINT_TO_VERTEX, _W_W3, _NO_NAME),
    ("W2v_tgtDrop", _OFF_J, HEAD_MIDPOINT_TO_VERTEX, _W_FULL, _TGT_DROP),
    ("W2v_tgtDown", _OFF_J, HEAD_MIDPOINT_TO_VERTEX, _W_FULL, _TGT_DOWN),
]

# Views to surface individually in the per-view reprojection breakdown: the two
# targeted outliers, the good profile W3 wrongly suppresses, and the frontal
# view whose overlay the user queried (cam10_2). cam10_2 is a diagnostic, not a
# fit target — its own recovered camera, not the fit weighting, drives its overlay.
_WATCH_VIEWS = ["cam10_2", "cam01_2", "cam02_4", "cam02_5", "cam06_4"]


def _median_reproj_all(
    joints: np.ndarray,
    kp2d: dict[str, np.ndarray],
    confs: dict[str, np.ndarray],
    cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    fit_views: list[str],
) -> float:
    """Median over ALL joint x view reprojection errors (px), unweighted.

    Mirrors the pipeline's ``median_reproj_px`` acceptance metric (flat median
    over every per-joint per-view term with conf >= MIN_CONF), so the A/B number
    is directly comparable to the acceptance criterion.
    """
    errs: list[float] = []
    for v in fit_views:
        R, t, K = cameras[v]
        for coco_idx, smpl_idx in COCO_TO_SMPL.items():
            if confs[v][coco_idx] < MIN_CONF:
                continue
            proj = project_points(joints[smpl_idx : smpl_idx + 1], R, t, K)[0]
            errs.append(float(np.linalg.norm(proj - kp2d[v][coco_idx])))
    return float(np.median(errs)) if errs else float("nan")


def _per_view_reproj(
    joints: np.ndarray,
    kp2d: dict[str, np.ndarray],
    confs: dict[str, np.ndarray],
    cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    views: list[str],
) -> dict[str, float]:
    """Median reprojection error (px) of ``joints`` per named view.

    Same metric as leave_one_view_out._view_reproj_error, evaluated in every
    requested view's OWN camera. Down-weighting a view in the fit changes the
    body and hence its reprojection here; but a view's own camera is fixed, so
    this shows what targeted rejection does (and does NOT do) to each overlay.
    """
    out: dict[str, float] = {}
    for v in views:
        matches = [n for n in cameras if n.rsplit(".", 1)[0] == v or n == v]
        if not matches or matches[0] not in kp2d:
            continue
        name = matches[0]
        R, t, K = cameras[name]
        errs = [
            float(np.linalg.norm(project_points(joints[si : si + 1], R, t, K)[0] - kp2d[name][ci]))
            for ci, si in COCO_TO_SMPL.items()
            if confs[name][ci] >= MIN_CONF
        ]
        out[v] = float(np.median(errs)) if errs else float("nan")
    return out


def _dump_for_visualiser(
    res: RefinementResult,
    faces: np.ndarray,
    cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    """Write ``refined_mesh.obj`` + ``refinement_results.json`` for one A/B config.

    scantosmpl.evaluation.visualise renders the ``[photo | Tier 1 | Tier 2]``
    overlays straight from these two files: the fitted mesh (world/SMPL frame) and
    the per-view ``[R|t|K]``. The cameras are config-*independent* fit inputs, so
    this reuses the same self-calibrated cameras the pipeline solved — only the
    mesh changes per config. Lets you eyeball any A/B config's fit (e.g. W2_vertex,
    no view weighting) without re-running the full Tier-1→2 pipeline.
    """
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=res.vertices, faces=faces, process=False)
    mesh.export(out_dir / "refined_mesh.obj")
    result_data = {
        "betas": res.betas.tolist(),
        "body_pose": res.body_pose.tolist(),
        "global_orient": res.global_orient.tolist(),
        "translation": res.translation.tolist(),
        "scale": float(res.scale),
        "metrics": res.metrics,
        "cameras": {
            name: {"R": R.tolist(), "t": t.tolist(), "K": K.tolist()}
            for name, (R, t, K) in cameras.items()
        },
    }
    with open(out_dir / "refinement_results.json", "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info(
        "Dumped '%s' fit -> %s (refined_mesh.obj + refinement_results.json). "
        "Render with: python -m scantosmpl.evaluation.visualise --refinement-dir %s",
        out_dir.name,
        out_dir,
        out_dir,
    )


def _body_geometry(joints: np.ndarray, vertices: np.ndarray) -> dict[str, float]:
    """Intrinsic torso/head geometry in the body's own frame (cm / deg).

    All quantities are measured relative to the fitted body's *own* pelvis and
    up/forward axes, so they are invariant to the fit's global translation,
    rotation and (for the fraction metrics) scale — they isolate the articulated
    change, which is exactly what "chest higher" / "head tilted back" mean.

    Frame (from the fitted joints):
        up  = neck - pelvis               (torso axis)
        fwd = anterior = -cross(up, L_shoulder - R_shoulder), orthogonalised to up
    """
    pelvis = joints[Smpl24Joint.PELVIS]
    neck = joints[Smpl24Joint.NECK]
    head = joints[_HEAD]
    spine3 = joints[_SPINE3]
    l_sh = joints[Smpl24Joint.LEFT_SHOULDER]
    r_sh = joints[Smpl24Joint.RIGHT_SHOULDER]

    up = neck - pelvis
    torso_len = float(np.linalg.norm(up))
    if torso_len < 1e-6:
        return {}
    up_hat = up / torso_len

    # anterior (facing) direction: cross(up, L-R) points posterior, so negate;
    # orthogonalise against up so a tilted torso axis doesn't leak in.
    back = np.cross(up_hat, l_sh - r_sh)
    fwd = -back
    fwd = fwd - np.dot(fwd, up_hat) * up_hat
    fwd_norm = float(np.linalg.norm(fwd))
    fwd_hat = fwd / fwd_norm if fwd_norm > 1e-6 else np.zeros(3)

    shoulder_mid = 0.5 * (l_sh + r_sh)

    # --- chest height: proportional position of chest landmarks up the torso ---
    spine3_frac = float(np.dot(spine3 - pelvis, up_hat) / torso_len)
    shoulder_frac = float(np.dot(shoulder_mid - pelvis, up_hat) / torso_len)

    # --- head placement relative to neck (cm) ---
    head_fwd_cm = float(np.dot(head - neck, fwd_hat)) * 100.0  # +anterior = head-forward
    head_up_cm = float(np.dot(head - neck, up_hat)) * 100.0

    # --- head/skull pitch: direction of the crown (vertices above the head
    # joint along up) from the head joint, in the fwd-up plane. atan2(fwd, up):
    # + = crown forward (bowed), - = crown back (tilted back).
    above = (vertices - head) @ up_hat > 0.0
    if above.any():
        crown = vertices[above].mean(axis=0) - head
        pitch = np.arctan2(np.dot(crown, fwd_hat), np.dot(crown, up_hat))
        head_pitch_deg = float(np.degrees(pitch))
    else:
        head_pitch_deg = float("nan")

    return {
        "torso_len_cm": torso_len * 100.0,
        "spine3_height_frac": spine3_frac,
        "shoulder_height_frac": shoulder_frac,
        "head_fwd_cm": head_fwd_cm,
        "head_up_cm": head_up_cm,
        "head_pitch_deg": head_pitch_deg,
    }


def run_ab(
    refinement_dir: Path = DEFAULT_REFINEMENT_DIR,
    consensus_dir: Path = DEFAULT_CONSENSUS_DIR,
    detections_path: Path = DEFAULT_DETECTIONS,
    smpl_model_dir: Path = DEFAULT_SMPL_MODEL_DIR,
    gender: str = "neutral",
    device: str = "cuda",
    output_path: Path | None = None,
    dump_config: str | None = None,
    dump_dir: Path | None = None,
) -> dict:
    """Run the A/B refits and return the attribution report.

    If ``dump_config`` (a config label) and ``dump_dir`` are given, that config's
    fitted mesh + cameras are written to ``dump_dir`` for the visualiser (see
    _dump_for_visualiser).
    """
    known_labels = [label for (label, *_rest) in CONFIGS]
    if dump_config is not None and dump_config not in known_labels:
        raise ValueError(f"--dump-config {dump_config!r} not in {known_labels}")

    smpl = SMPLModel(model_dir=smpl_model_dir, gender=gender, device=device)

    consensus = _load_consensus(consensus_dir, smpl)
    cameras = _load_cameras(refinement_dir / "refinement_results.json")
    kp2d, confs = _load_detections(detections_path)
    triang = _load_triangulated_joints(refinement_dir / "triangulated_joints.json")

    usable = [n for n in cameras if n in kp2d and n in confs]
    cams_in = {n: cameras[n] for n in usable}
    kp_in = {n: kp2d[n] for n in usable}
    conf_in = {n: confs[n] for n in usable}

    rear = classify_rear_views(consensus, cameras)
    angles = classify_view_angles(consensus, cameras)
    # Views that actually drive the fit (non-rear); used for the reproj metric so
    # all four configs are scored on the same view set.
    fit_views = [n for n in usable if n not in rear]

    results: dict[str, dict[str, float]] = {}
    per_view_reproj: dict[str, dict[str, float]] = {}
    dump_res: RefinementResult | None = None
    for label, joint_head, vertex_head, weights, name_weights in CONFIGS:
        torch.manual_seed(0)  # belt-and-braces; Adam here is already deterministic
        opt = SMPLOptimiser(
            smpl,
            COCO_TO_SMPL,
            midpoint_to_smpl=joint_head,
            vertex_midpoint_to_smpl=vertex_head,
            view_angle_weights=weights,
            view_name_weights=name_weights,
        )
        res = opt.refine(
            consensus=consensus,
            triangulated_joints=triang,
            keypoints_2d=kp_in,
            confs=conf_in,
            cameras=cams_in,
        )
        metrics: dict[str, float] = {
            "pa_mpjpe_mm": float(res.metrics.get("pa_mpjpe_mm", float("nan"))),
            "median_reproj_px": _median_reproj_all(res.joints, kp2d, confs, cameras, fit_views),
            "scale": float(res.scale),
            "betas_l2": float(np.linalg.norm(res.betas)),
        }
        metrics.update(_body_geometry(res.joints, res.vertices))
        results[label] = metrics
        per_view_reproj[label] = _per_view_reproj(res.joints, kp2d, confs, cameras, _WATCH_VIEWS)
        if label == dump_config:
            dump_res = res
        logger.info(
            "%-10s PA-MPJPE=%.2fmm  median_reproj=%.1fpx  shoulder_frac=%.4f  "
            "head_fwd=%.2fcm head_up=%.2fcm head_pitch=%.1fdeg",
            label,
            metrics["pa_mpjpe_mm"],
            metrics["median_reproj_px"],
            metrics.get("shoulder_height_frac", float("nan")),
            metrics.get("head_fwd_cm", float("nan")),
            metrics.get("head_up_cm", float("nan")),
            metrics.get("head_pitch_deg", float("nan")),
        )

    if dump_config is not None and dump_dir is not None and dump_res is not None:
        _dump_for_visualiser(dump_res, smpl.body_model.faces, cams_in, dump_dir)

    non_baseline = [label for (label, *_rest) in CONFIGS if label != "baseline"]
    base = results["baseline"]
    deltas: dict[str, dict[str, float]] = {}
    for label in non_baseline:
        deltas[label] = {
            k: results[label][k] - base[k]
            for k in base
            if isinstance(results[label].get(k), float) and np.isfinite(results[label][k])
        }

    view_grades = {n: angles.get(n, "unknown") for n in fit_views}
    report = {
        "_note": (
            "A/B attribution refit off cached artefacts. Deterministic (triangulated "
            "joints loaded from disk, no RANSAC; Adam is deterministic), so metric "
            "changes are attributable to the config, not run noise. Configs isolate "
            "the head-anchor fix: baseline (no head term) / W2_joint (old, 2D ears -> "
            "head JOINT 15, biased +6.7cm up / -3cm back) / W2_vertex (fixed, 2D ears "
            "-> ear-VERTEX midpoint) / W2v_W3 (fixed head + profile downweight). Scope: "
            "PA-MPJPE + chest-height + head geometry ONLY. Torso GIRTH is intentionally "
            "excluded: "
            "it is a surface (beta/per-vertex) property that joint-centre losses cannot "
            "constrain from any view -> Tier-3 chamfer, not a Tier-2 knob. Geometry "
            "metrics are in the body's own frame (invariant to global pose; fracs also "
            "to scale). head_fwd_cm: +anterior of neck. head_pitch_deg: +crown forward "
            "(bowed), -crown back (tilted back). *_height_frac: chest landmark height as "
            "a fraction of pelvis->neck torso length."
        ),
        "n_fit_views": len(fit_views),
        "rear_excluded": sorted(rear),
        "view_grades": view_grades,
        "configs": results,
        "deltas_vs_baseline": deltas,
        "per_view_reproj_px": per_view_reproj,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("A/B report written to %s", output_path)

    return report


def _fmt_row(label: str, m: dict[str, float]) -> str:
    return (
        f"{label:<11}"
        f"{m.get('pa_mpjpe_mm', float('nan')):>9.2f}"
        f"{m.get('median_reproj_px', float('nan')):>10.1f}"
        f"{m.get('shoulder_height_frac', float('nan')):>11.4f}"
        f"{m.get('spine3_height_frac', float('nan')):>11.4f}"
        f"{m.get('head_fwd_cm', float('nan')):>10.2f}"
        f"{m.get('head_up_cm', float('nan')):>10.2f}"
        f"{m.get('head_pitch_deg', float('nan')):>11.1f}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-dir", type=Path, default=DEFAULT_REFINEMENT_DIR)
    parser.add_argument("--consensus-dir", type=Path, default=DEFAULT_CONSENSUS_DIR)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--smpl-model-dir", type=Path, default=DEFAULT_SMPL_MODEL_DIR)
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REFINEMENT_DIR / "ab_refit.json",
    )
    parser.add_argument(
        "--dump-config",
        default=None,
        help="Config label (e.g. W2_vertex) whose fitted mesh + cameras to write "
        "to --dump-dir for scantosmpl.evaluation.visualise. Off by default.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Directory to write the --dump-config mesh into (used as "
        "visualise.py --refinement-dir).",
    )
    args = parser.parse_args()

    if (args.dump_config is None) != (args.dump_dir is None):
        parser.error("--dump-config and --dump-dir must be given together")

    report = run_ab(
        refinement_dir=args.refinement_dir,
        consensus_dir=args.consensus_dir,
        detections_path=args.detections,
        smpl_model_dir=args.smpl_model_dir,
        gender=args.gender,
        device=args.device,
        output_path=args.output,
        dump_config=args.dump_config,
        dump_dir=args.dump_dir,
    )

    labels = [label for (label, *_rest) in CONFIGS]
    print("\n=== A/B attribution refit (chest-height + head geometry + PA-MPJPE) ===")
    print(f"Fit views: {report['n_fit_views']} | rear excluded: {len(report['rear_excluded'])}")
    hdr = (
        f"{'config':<13}{'PA-MPJPE':>9}{'medReproj':>10}{'sh_frac':>11}"
        f"{'sp3_frac':>11}{'head_fwd':>10}{'head_up':>10}{'head_pitch':>11}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for label in labels:
        print(_fmt_row(label, report["configs"][label]))

    print("\n--- deltas vs baseline (positive = larger than baseline) ---")
    print(hdr)
    print("-" * len(hdr))
    for label in labels:
        if label == "baseline":
            continue
        d = report["deltas_vs_baseline"][label]
        # reuse the row formatter on the delta dict (leaves absent keys as nan)
        print(_fmt_row(label, d))
    print(
        "\nunits: PA-MPJPE mm | medReproj px | *_frac = fraction of torso length | "
        "head_fwd/up cm | head_pitch deg (+bowed / -tilted back)"
    )

    # Per-view reprojection (px) for the watched views — shows what targeted
    # rejection does to each overlay (incl. cam10_2, a diagnostic not a target).
    pv = report["per_view_reproj_px"]
    watch = list(next(iter(pv.values())).keys()) if pv else []
    print("\n--- per-view median reprojection (px) ---")
    vhdr = f"{'config':<13}" + "".join(f"{v:>11}" for v in watch)
    print(vhdr)
    print("-" * len(vhdr))
    for label in labels:
        row = f"{label:<13}" + "".join(f"{pv[label].get(v, float('nan')):>11.1f}" for v in watch)
        print(row)


if __name__ == "__main__":
    main()
