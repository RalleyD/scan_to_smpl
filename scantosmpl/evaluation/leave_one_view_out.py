"""W4 — Leave-one-view-out (LOVO) refit diagnostic.

The honest form of "Theory 2" (search over view subsets to minimise mean
reprojection). Rather than *searching* for the metric-minimising subset — which
lets the fit absorb error to flatter the number (the circularity trap the master
spec flags) — this holds each view out in turn, refits without it, and asks how
well the fit *predicts the view it never saw*. That separates two failure modes
the raw per-view reprojection cannot:

  * A **genuinely unexplainable** view: badly predicted whether it is included
    or held out (its 2D keypoints disagree with every other view's geometry —
    a detector failure / left-right swap / mislabeled camera). Drop it.
  * A merely **hard** view (e.g. a profile with higher ViTPose noise): predicted
    about as well held-out as in-sample. Keep it, down-weighted (W3), not dropped.

The output feeds W3's weight rule; it is NOT a subset search.

Runs purely off the debug artefacts Phase 3/4/5 already wrote to disk
(consensus_results.json, refinement_results.json, triangulated_joints.json,
detections.json) — no HMR / calibration / GPU-heavy re-run of Tiers 1-2. It does
re-run the (light) SMPL optimiser once per held-out view, so a GPU is used if
available but a CPU run is fine (just slower).

Usage:
    python -m scantosmpl.evaluation.leave_one_view_out
    python -m scantosmpl.evaluation.leave_one_view_out --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scantosmpl.fitting.optimiser import SMPLOptimiser
from scantosmpl.fitting.rear_views import classify_rear_views, classify_view_angles
from scantosmpl.hmr.consensus import ConsensusResult
from scantosmpl.smpl.joint_map import COCO_TO_SMPL
from scantosmpl.smpl.model import SMPLModel
from scantosmpl.utils.geometry import project_points

logger = logging.getLogger(__name__)

DEFAULT_REFINEMENT_DIR = Path("output/debug/refinement")
DEFAULT_CONSENSUS_DIR = Path("output/debug/consensus")
DEFAULT_DETECTIONS = Path("output/debug/detection/detections.json")
DEFAULT_SMPL_MODEL_DIR = Path("models/smpl")

# Views whose held-out reprojection exceeds this multiple of the *in-sample*
# median are flagged as candidate true outliers (unexplainable), not merely
# hard. Deliberately loose — this is a diagnostic hint, not an automatic drop.
OUTLIER_FACTOR = 2.0

MIN_CONF = 0.3


def _load_detections(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return {view: (17,2) keypoints}, {view: (17,) confs} from detections.json."""
    data = json.load(open(path))
    kp2d: dict[str, np.ndarray] = {}
    confs: dict[str, np.ndarray] = {}
    for det in data:
        name = det["filename"]
        kps_raw = det.get("keypoints") or det.get("keypoints_2d")
        confs_raw = det.get("keypoint_confidences") or det.get("keypoint_confs")
        if kps_raw is None or confs_raw is None:
            continue
        kp2d[name] = np.asarray(kps_raw, dtype=np.float64)
        confs[name] = np.asarray(confs_raw, dtype=np.float64)
    return kp2d, confs


def _load_cameras(
    path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {view: (R, t, K)} from refinement_results.json."""
    data = json.load(open(path))
    cams: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, cam in data["cameras"].items():
        cams[name] = (
            np.asarray(cam["R"], dtype=np.float64),
            np.asarray(cam["t"], dtype=np.float64),
            np.asarray(cam["K"], dtype=np.float64),
        )
    return cams


def _load_consensus(consensus_dir: Path, smpl: SMPLModel) -> ConsensusResult:
    """Rebuild a minimal ConsensusResult from consensus_results.json.

    consensus_results.json stores params, not joints/vertices, so we run one
    SMPL forward pass to recover the joints the optimiser initialises from and
    the rear/graded classifier reads.
    """
    import torch

    data = json.load(open(consensus_dir / "consensus_results.json"))
    betas = np.asarray(data["betas"], dtype=np.float64)
    body_pose = np.asarray(data["body_pose"], dtype=np.float64)
    global_orient = np.asarray(data["global_orient"], dtype=np.float64)

    with torch.no_grad():
        out = smpl.forward(
            betas=torch.tensor(betas, dtype=torch.float32, device=smpl.device).unsqueeze(0),
            body_pose=torch.tensor(body_pose, dtype=torch.float32, device=smpl.device).unsqueeze(0),
            global_orient=torch.tensor(
                global_orient, dtype=torch.float32, device=smpl.device
            ).unsqueeze(0),
        )
    joints = out.joints.squeeze(0).cpu().numpy()
    vertices = out.vertices.squeeze(0).cpu().numpy()

    return ConsensusResult(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        vertices=vertices,
        joints=joints,
        faces=smpl.body_model.faces.astype(np.int64),
        pa_mpjpe_per_view={},
        pa_mpjpe_mean=0.0,
        beta_std=np.zeros(10),
        body_height_m=float(data.get("body_height_m", 1.7)),
        per_view_weights={},
        n_views_used=int(data.get("n_views_used", 0)),
    )


def _load_triangulated_joints(path: Path) -> np.ndarray:
    """Return (24, 3) triangulated joints in SMPL ordering from triangulated_joints.json."""
    data = json.load(open(path))
    triang = np.zeros((24, 3), dtype=np.float64)
    for entry in data.values():
        idx = int(entry["smpl_idx"])
        if idx < 24 and entry.get("quality", 0.0) > 0.0:
            triang[idx] = np.asarray(entry["position"], dtype=np.float64)
    return triang


def _view_reproj_error(
    joints: np.ndarray,
    kp2d: np.ndarray,
    confs: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> float:
    """Median per-joint reprojection error (px) of `joints` against one view.

    Median (not mean) so a single swapped/occluded keypoint doesn't dominate the
    per-view number — same rationale as the pipeline's W1 metric change.
    """
    errs = []
    for coco_idx, smpl_idx in COCO_TO_SMPL.items():
        if confs[coco_idx] < MIN_CONF:
            continue
        proj = project_points(joints[smpl_idx : smpl_idx + 1], R, t, K)[0]
        errs.append(float(np.linalg.norm(proj - kp2d[coco_idx])))
    return float(np.median(errs)) if errs else float("nan")


def run_lovo(
    refinement_dir: Path = DEFAULT_REFINEMENT_DIR,
    consensus_dir: Path = DEFAULT_CONSENSUS_DIR,
    detections_path: Path = DEFAULT_DETECTIONS,
    smpl_model_dir: Path = DEFAULT_SMPL_MODEL_DIR,
    gender: str = "neutral",
    device: str = "cuda",
    output_path: Path | None = None,
) -> dict:
    """Run leave-one-view-out refits and return the diagnostic report.

    For each non-rear view v:
      1. Refit SMPL against every view EXCEPT v (held-out).
      2. Measure the refit's reprojection error on v (held-out error).
      3. Compare against the error on v when v IS included (in-sample error),
         taken from a single full-data refit.
    A large held-out/in-sample gap, or a large absolute held-out error, marks v
    as a candidate true outlier rather than merely a hard view.
    """
    smpl = SMPLModel(model_dir=smpl_model_dir, gender=gender, device=device)

    consensus = _load_consensus(consensus_dir, smpl)
    cameras = _load_cameras(refinement_dir / "refinement_results.json")
    kp2d, confs = _load_detections(detections_path)
    triang = _load_triangulated_joints(refinement_dir / "triangulated_joints.json")

    # Only views with both a solved camera and detected keypoints participate.
    usable = [n for n in cameras if n in kp2d and n in confs]
    rear = classify_rear_views(consensus, cameras)
    angles = classify_view_angles(consensus, cameras)
    # Rear views are excluded from the fit anyway (W3 weight 0), so holding one
    # out changes nothing — only refit-eligible (non-rear) views are looped.
    fit_views = [n for n in usable if n not in rear]

    def _refit(exclude: str | None) -> np.ndarray:
        """Return refined (24,3) SMPL joints, optionally excluding one view's data."""
        opt = SMPLOptimiser(smpl, COCO_TO_SMPL)
        cams_in = {n: cameras[n] for n in fit_views if n != exclude}
        kp_in = {n: kp2d[n] for n in cams_in}
        conf_in = {n: confs[n] for n in cams_in}
        result = opt.refine(
            consensus=consensus,
            triangulated_joints=triang,
            keypoints_2d=kp_in,
            confs=conf_in,
            cameras=cams_in,
        )
        return result.joints

    logger.info("LOVO: full-data refit (in-sample baseline)")
    joints_full = _refit(exclude=None)

    per_view: dict[str, dict] = {}
    for v in fit_views:
        R, t, K = cameras[v]
        in_sample = _view_reproj_error(joints_full, kp2d[v], confs[v], R, t, K)
        logger.info("LOVO: hold out %s", v)
        joints_lovo = _refit(exclude=v)
        held_out = _view_reproj_error(joints_lovo, kp2d[v], confs[v], R, t, K)
        per_view[v] = {
            "view_angle": angles.get(v, "unknown"),
            "in_sample_reproj_px": in_sample,
            "held_out_reproj_px": held_out,
            "held_out_minus_in_sample_px": held_out - in_sample,
        }

    ho = [
        d["held_out_reproj_px"] for d in per_view.values() if np.isfinite(d["held_out_reproj_px"])
    ]
    median_held_out = float(np.median(ho)) if ho else float("nan")

    # A view is a *candidate* true outlier if its held-out prediction is far
    # worse than the cohort median — its own geometry is not explained by the
    # rest. (Hint only; the human/W3 decides the weight.)
    for v, d in per_view.items():
        d["candidate_outlier"] = bool(
            np.isfinite(d["held_out_reproj_px"])
            and d["held_out_reproj_px"] > OUTLIER_FACTOR * median_held_out
        )

    candidate_outliers = sorted(v for v, d in per_view.items() if d["candidate_outlier"])

    report = {
        "_caveat": (
            "This is a leave-one-view-out DIAGNOSTIC, not a subset search. It does "
            "NOT pick the view subset that minimises mean reprojection — doing so is "
            "the circularity trap the master spec flags (data/cameras absorb error to "
            "flatter the metric). 'candidate_outlier' flags views the fit predicts "
            "poorly when they are held out; treat it as a hint feeding W3's per-view "
            "weight rule (down-weight profiles / drop genuine detector failures), not "
            "as an automatic exclusion. Held-out error is inherently >= in-sample "
            "error; only an unusually large gap or absolute value is informative."
        ),
        "n_fit_views": len(fit_views),
        "n_rear_excluded": len(rear),
        "rear_views": sorted(rear),
        "median_held_out_reproj_px": median_held_out,
        "outlier_factor": OUTLIER_FACTOR,
        "candidate_outliers": candidate_outliers,
        "per_view": dict(sorted(per_view.items(), key=lambda kv: -kv[1]["held_out_reproj_px"])),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("LOVO report written to %s", output_path)

    return report


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
        default=DEFAULT_REFINEMENT_DIR / "leave_one_view_out.json",
    )
    args = parser.parse_args()

    report = run_lovo(
        refinement_dir=args.refinement_dir,
        consensus_dir=args.consensus_dir,
        detections_path=args.detections,
        smpl_model_dir=args.smpl_model_dir,
        gender=args.gender,
        device=args.device,
        output_path=args.output,
    )

    print("\n=== Leave-one-view-out diagnostic ===")
    print(f"Fit views: {report['n_fit_views']} | rear excluded: {report['n_rear_excluded']}")
    print(f"Median held-out reproj: {report['median_held_out_reproj_px']:.1f} px")
    outliers = report["candidate_outliers"] or "none"
    print(f"Candidate outliers (>{OUTLIER_FACTOR}x median): {outliers}")
    print(f"\n{'view':<16}{'angle':<15}{'in-sample':>11}{'held-out':>11}{'gap':>8}  flag")
    for v, d in report["per_view"].items():
        flag = "OUTLIER?" if d["candidate_outlier"] else ""
        print(
            f"{v:<16}{d['view_angle']:<15}"
            f"{d['in_sample_reproj_px']:>10.1f} {d['held_out_reproj_px']:>10.1f} "
            f"{d['held_out_minus_in_sample_px']:>7.1f}  {flag}"
        )


if __name__ == "__main__":
    main()
