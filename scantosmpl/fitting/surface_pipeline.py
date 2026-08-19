"""Tier 3 orchestrator (master §4, §5.3): load -> preprocess -> align (S1) ->
`Tier3SurfaceFitter.fit` (S2, S3) -> `chamfer_report` -> `Tier3Quality` -> persist ->
`update_manifest`.

This module wires together the four sibling deliverables (`scantosmpl.pointcloud`,
`scantosmpl.evaluation.surface_metrics`, `scantosmpl.fitting.surface_losses`,
`scantosmpl.fitting.surface`) and the 7.B enforcement layer (`scantosmpl.fitting.
artefacts`) into one runnable tier. It never writes back into Tier 2's own artefacts —
`RefinementResult` is read-only input (master §4's "one-directional" tier boundary).

`Tier3Pipeline` deliberately introduces no stochastic step of its own (master D12): the
only seeded randomness anywhere downstream of it is `tessellation_floor`'s surface
sampler (seeded via `Tier3Config.tessellation_floor_seed`), which lives in
`scantosmpl.evaluation.surface_metrics` and is called through `chamfer_report`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from scantosmpl.config import Tier3Config
from scantosmpl.evaluation.surface_metrics import ChamferReport, chamfer_report
from scantosmpl.fitting.artefacts import (
    faces_sha256,
    load_locked_betas,
    update_manifest,
    write_pose_artefacts,
)
from scantosmpl.fitting.optimiser import RefinementResult
from scantosmpl.fitting.surface import SurfaceFitResult, Tier3SurfaceFitter
from scantosmpl.pointcloud.align import CloudAlignment, align_cloud_to_smpl
from scantosmpl.pointcloud.io import load_pointcloud, save_pointcloud
from scantosmpl.pointcloud.preprocess import PreprocessStats, preprocess_cloud
from scantosmpl.smpl.model import SMPLModel
from scantosmpl.types import SMPL_NUM_FACES, SMPL_NUM_VERTICES, PoseArtefact, Tier3Quality

logger = logging.getLogger(__name__)

#: "Real scanner data" lives under this repo-relative root (master §6's user-flow
#: examples: `data/t-pose/pointcloud.ply`, `data/a-pose/pointcloud.ply`, ...); anything
#: else (in particular `tests/integration/fixtures/synthetic_cloud/cloud.ply`) is
#: synthetic. `Tier3Pipeline` has no other way to know a cloud's provenance — this is
#: the one place that distinction is made, so the Tier 3 gate in `summary.txt` can be
#: honest (repo spec / brief step 4: PASS only on real scanner data, DEFERRED
#: otherwise — never PASS on synthetic geometry).
REAL_DATA_ROOT = Path("data")

#: AC9's deferred-gate threshold on real scanner data (7.1).
REAL_CLOUD_TO_MESH_MEAN_MM_TARGET = 8.0

#: AC8's minimum improvement of the refined fit over the Tier-2-params, D=0 baseline.
AC8_MIN_IMPROVEMENT_FRACTION = 0.40


@dataclass
class Tier3Result:
    """Output of `Tier3Pipeline.run` (master §5.3)."""

    fit: SurfaceFitResult
    alignment: CloudAlignment
    report: ChamferReport
    preprocess: PreprocessStats
    quality: Tier3Quality
    artefact_dir: Path


def _is_real_scanner_data(pointcloud_path: Path) -> bool:
    """True when `pointcloud_path` lives under the repo's `data/` root.

    `data/` holds manually-deposited real scans (per CLAUDE.md / master §6); test
    fixtures live under `tests/integration/fixtures/`. A path outside both is treated
    as non-real (conservative — the Tier 3 gate must never read PASS by accident).
    """
    try:
        resolved = Path(pointcloud_path).resolve()
        real_root = REAL_DATA_ROOT.resolve()
    except OSError:
        return False
    return resolved == real_root or real_root in resolved.parents


class Tier3Pipeline:
    """S1 (align) -> S2/S3 (`Tier3SurfaceFitter.fit`) -> metric -> persist.

    Args:
        smpl_model: The SMPL layer used for both the ICP target mesh and the surface
            fit. Its `device` is used throughout.
        cfg: Tier 3 configuration (`scantosmpl.config.Tier3Config`).
    """

    def __init__(self, smpl_model: SMPLModel, cfg: Tier3Config) -> None:
        self.smpl = smpl_model
        self.cfg = cfg

    def run(
        self,
        tier2: RefinementResult,
        pointcloud_path: Path,
        *,
        pose_name: str,
        output_dir: Path,
    ) -> Tier3Result:
        """Run S1 -> S2 -> S3 -> metric -> persist for one pose of one subject.

        Args:
            tier2: Phase 5 `RefinementResult` — read-only Tier 2 input. Never mutated
                and never written back to (master §4's one-directional tier boundary).
            pointcloud_path: PLY/OBJ scan, in its own arbitrary source frame/units.
            pose_name: This pose's name, e.g. `"t-pose"`. Used as the per-pose
                subdirectory under `output_dir` and as the manifest's `poses[].pose_name`.
            output_dir: The SUBJECT's output directory, e.g. `output/fits/dan/` — the
                per-pose artefacts land in `output_dir / pose_name`, and
                `output_dir / "manifest.json"` is created-or-updated.

        Returns:
            `Tier3Result` with the fit, alignment, chamfer report, preprocessing stats,
            quality summary and the per-pose artefact directory.

        Raises:
            ValueError: `cfg.lock_betas` is set without `cfg.betas_source`.
            AssertionError: Any 7.B artefact-write invariant is violated
                (`scantosmpl.fitting.artefacts.write_pose_artefacts`).
        """
        cfg = self.cfg
        output_dir = Path(output_dir)
        out_dir = output_dir / pose_name

        faces = self.smpl.body_model.faces.astype(np.int64)

        # --- load -> preprocess (unit-free, D8) ---------------------------------
        raw_cloud = load_pointcloud(pointcloud_path)
        cleaned_cloud, preprocess_stats = preprocess_cloud(raw_cloud, cfg)

        # --- S1: align cloud -> SMPL (Tier 2's mesh; scale solved here, D6) -----
        aligned_cloud, alignment = align_cloud_to_smpl(cleaned_cloud, tier2.vertices, faces, cfg)

        # --- locked betas (7.B1 / D10), if requested ----------------------------
        locked_betas = self._load_locked_betas() if cfg.lock_betas else None

        # --- S2 + S3: surface fit -----------------------------------------------
        fitter = Tier3SurfaceFitter(self.smpl, cfg)
        self._sync_device()
        start = time.perf_counter()
        fit_result = fitter.fit(tier2, aligned_cloud, locked_betas=locked_betas)
        self._sync_device()  # fit() already ends in .cpu() transfers, but be explicit
        elapsed_s = time.perf_counter() - start

        # --- metric report (7.M): final fit, and the AC8 D=0/Tier-2 baseline ---
        report = chamfer_report(aligned_cloud.points, fit_result.vertices, faces, cfg)
        baseline_report = chamfer_report(aligned_cloud.points, tier2.vertices, faces, cfg)

        # --- Tier3Quality (7.B7) --------------------------------------------------
        quality = self._build_quality(report, alignment, fit_result, tier2)

        # --- persist (7.B enforcement layer) --------------------------------------
        write_pose_artefacts(out_dir, fit_result, alignment, quality, faces, pose_name=pose_name)

        if cfg.save_debug:
            save_pointcloud(aligned_cloud, out_dir / "pointcloud_aligned.ply")

        smpl_meta = self._smpl_meta(faces)
        beta_policy = self._beta_policy(fit_result, pose_name)
        entry = PoseArtefact(
            pose_name=pose_name,
            directory=pose_name,
            oracle_only=bool(cfg.oracle_only),
            betas_locked=fit_result.betas_locked,
            has_displacements=True,  # 7.B2 — D is written unconditionally
            has_pointcloud=True,
            quality=quality,
        )
        manifest_path = output_dir / "manifest.json"
        update_manifest(
            manifest_path,
            entry,
            subject_id=cfg.subject_id,
            smpl_meta=smpl_meta,
            beta_policy=beta_policy,
        )

        if cfg.save_debug:
            self._write_summary(
                pose_name=pose_name,
                pointcloud_path=Path(pointcloud_path),
                preprocess_stats=preprocess_stats,
                alignment=alignment,
                report=report,
                baseline_report=baseline_report,
                quality=quality,
                fit_result=fit_result,
                elapsed_s=elapsed_s,
                n_cloud_points=aligned_cloud.n_points,
            )

        return Tier3Result(
            fit=fit_result,
            alignment=alignment,
            report=report,
            preprocess=preprocess_stats,
            quality=quality,
            artefact_dir=out_dir,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _sync_device(self) -> None:
        if torch.cuda.is_available() and self.smpl.device.type == "cuda":
            torch.cuda.synchronize()

    def _load_locked_betas(self) -> np.ndarray:
        if self.cfg.betas_source is None:
            raise ValueError(
                "cfg.lock_betas=True requires cfg.betas_source (a smpl_params.npz path) — "
                "there is nothing to lock beta TO otherwise. (CLI: --lock-betas requires "
                "--betas-from.)"
            )
        return load_locked_betas(self.cfg.betas_source)

    def _smpl_meta(self, faces: np.ndarray) -> dict:
        return {
            "gender": str(self.smpl.body_model.gender),
            "num_betas": int(self.smpl.body_model.num_betas),
            "num_vertices": SMPL_NUM_VERTICES,
            "num_faces": SMPL_NUM_FACES,
            "faces_sha256": faces_sha256(faces),
        }

    def _beta_policy(self, fit_result: SurfaceFitResult, pose_name: str) -> dict:
        betas_sha256 = hashlib.sha256(
            np.ascontiguousarray(fit_result.betas, dtype=np.float64).tobytes()
        ).hexdigest()
        if fit_result.betas_locked:
            source_pose = (
                self.cfg.betas_source.parent.name if self.cfg.betas_source is not None else None
            )
            return {"mode": "locked", "source_pose": source_pose, "betas_sha256": betas_sha256}
        return {"mode": "refined", "source_pose": pose_name, "betas_sha256": betas_sha256}

    def _build_quality(
        self,
        report: ChamferReport,
        alignment: CloudAlignment,
        fit_result: SurfaceFitResult,
        tier2: RefinementResult,
    ) -> Tier3Quality:
        disp_mm = np.linalg.norm(fit_result.displacements.astype(np.float64), axis=1) * 1000.0
        return Tier3Quality(
            chamfer_cloud_to_mesh_mean_mm=report.cloud_to_mesh_mm["mean"],
            chamfer_cloud_to_mesh_median_mm=report.cloud_to_mesh_mm["median"],
            chamfer_cloud_to_mesh_rms_mm=report.cloud_to_mesh_mm["rms"],
            chamfer_mesh_to_cloud_mean_mm=report.mesh_to_cloud_mm["mean"],
            chamfer_mesh_to_cloud_median_mm=report.mesh_to_cloud_mm["median"],
            chamfer_mesh_to_cloud_rms_mm=report.mesh_to_cloud_mm["rms"],
            tessellation_floor_mean_mm=report.tessellation_floor_mm["mean"],
            tessellation_floor_max_mm=report.tessellation_floor_mm["max"],
            icp_inlier_rmse_mm=float(alignment.inlier_rmse_m) * 1000.0,
            icp_fitness=float(alignment.fitness),
            displacement_mean_mm=float(disp_mm.mean()),
            displacement_p95_mm=float(np.percentile(disp_mm, 95.0)),
            pa_mpjpe_mm=tier2.metrics.get("pa_mpjpe_mm"),
            median_reproj_px=tier2.metrics.get("median_reproj_px"),
        )

    def _write_summary(
        self,
        *,
        pose_name: str,
        pointcloud_path: Path,
        preprocess_stats: PreprocessStats,
        alignment: CloudAlignment,
        report: ChamferReport,
        baseline_report: ChamferReport,
        quality: Tier3Quality,
        fit_result: SurfaceFitResult,
        elapsed_s: float,
        n_cloud_points: int,
    ) -> None:
        """Write `summary.txt` into `cfg.debug_dir`, mirroring the Phase 5 precedent
        (`scantosmpl.fitting.pipeline::Phase5Pipeline._save_debug`) — with its
        hard-won reporting lesson applied: cloud->mesh and mesh->cloud are ALWAYS
        printed separately (7.M3/7.M4), never fused, and the Tier 3 gate is printed
        as PASS only for real scanner data — never on synthetic fixture output.
        """
        debug_dir = Path(self.cfg.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        c2m = report.cloud_to_mesh_mm
        m2c = report.mesh_to_cloud_mm
        floor = report.tessellation_floor_mm

        baseline_mean = baseline_report.cloud_to_mesh_mm["mean"]
        final_mean = c2m["mean"]
        improvement_frac = (
            (baseline_mean - final_mean) / baseline_mean if baseline_mean > 0 else float("nan")
        )
        meets_ac8 = bool(improvement_frac >= AC8_MIN_IMPROVEMENT_FRACTION)

        is_real = _is_real_scanner_data(pointcloud_path)
        if is_real:
            gate_status = "PASS" if final_mean < REAL_CLOUD_TO_MESH_MEAN_MM_TARGET else "FAIL"
            gate_line = (
                f"TIER 3 GATE: {gate_status}"
                f" (cloud_to_mesh_mean_mm={final_mean:.2f} vs "
                f"{REAL_CLOUD_TO_MESH_MEAN_MM_TARGET:.1f} target, real scanner data)"
            )
        else:
            gate_line = (
                "TIER 3 GATE: DEFERRED (no real point cloud) — this run used "
                f"{pointcloud_path}, not data/<pose>/pointcloud.ply. Synthetic geometry "
                "can hit any threshold; a PASS here would be meaningless."
            )

        lines = [
            "=== Tier 3 Surface Refinement Summary ===",
            f"Pose:            {pose_name}",
            f"Subject:         {self.cfg.subject_id}",
            f"Point cloud:     {pointcloud_path}",
            f"Betas mode:      {'locked' if fit_result.betas_locked else 'refined'}",
            f"Semantic weighting: {'on' if self.cfg.use_semantic_weighting else 'off'}",
            "",
            "--- Preprocessing ---",
            f"Points: {preprocess_stats.n_input} -> {preprocess_stats.n_after_outlier_removal} "
            f"(outlier removal) -> {preprocess_stats.n_output} (downsample)",
            f"Aligned cloud points used for metrics: {n_cloud_points}",
            "",
            "--- Alignment (S1) ---",
            f"ICP candidate:   {alignment.candidate_index}/{alignment.n_candidates}",
            f"Scale (source -> metres): {alignment.scale:.6f}",
            f"Inlier RMSE:     {alignment.inlier_rmse_m * 1000.0:.2f} mm",
            f"Fitness:         {alignment.fitness:.3f}",
            f"Converged:       {alignment.converged}",
            "",
            "--- Surface metrics (7.M — two directions, reported separately, never fused) ---",
            f"cloud -> mesh (point-to-surface, mm): mean={c2m['mean']:.2f} "
            f"median={c2m['median']:.2f} rms={c2m['rms']:.2f} p95={c2m['p95']:.2f} "
            f"max={c2m['max']:.2f}",
            f"mesh -> cloud (vertex-to-point, mm):  mean={m2c['mean']:.2f} "
            f"median={m2c['median']:.2f} rms={m2c['rms']:.2f} p95={m2c['p95']:.2f} "
            f"max={m2c['max']:.2f}",
            f"Tessellation floor (this mesh's own vertex-sampling floor, mm): "
            f"mean={floor['mean']:.2f} max={floor['max']:.2f}",
            "  -- any VERTEX-based cloud->mesh number would carry this floor as an "
            "irreducible offset; the point-to-surface number above does not.",
            "",
            "--- AC8: refined fit vs Tier-2-params baseline (D=0, no Tier 3 refinement) ---",
            f"Baseline (Tier 2 params, D=0) cloud->mesh mean: {baseline_mean:.2f} mm",
            f"Final (Tier 3 refined + D)    cloud->mesh mean: {final_mean:.2f} mm",
            f"Improvement: {improvement_frac * 100.0:.1f}% (meets >=40% target: {meets_ac8})",
            "",
            "--- Displacement field D ---",
            f"mean |D|: {quality.displacement_mean_mm:.2f} mm",
            f"p95  |D|: {quality.displacement_p95_mm:.2f} mm",
            "",
            "--- Timing ---",
            f"S2+S3 wall clock: {elapsed_s:.1f}s",
            "",
            "=== " + gate_line.split(":", 1)[0].strip() + " ===",
            gate_line,
            "",
            "Loss history (final loss per stage):",
        ]
        for stage_name, hist in fit_result.loss_history.items():
            if hist:
                lines.append(f"  {stage_name}: {hist[0]:.4f} -> {hist[-1]:.4f}")

        with open(debug_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines))

        # A JSON breadcrumb next to the text summary — useful for the AC10 A/B script
        # and for any future dashboard, without inventing a second manifest format.
        with open(debug_dir / "last_run.json", "w") as f:
            json.dump(
                {
                    "pose_name": pose_name,
                    "pointcloud_path": str(pointcloud_path),
                    "is_real_scanner_data": is_real,
                    "quality": asdict(quality),
                    "ac8_baseline_cloud_to_mesh_mean_mm": baseline_mean,
                    "ac8_final_cloud_to_mesh_mean_mm": final_mean,
                    "ac8_improvement_fraction": improvement_frac,
                    "elapsed_s": elapsed_s,
                },
                f,
                indent=2,
            )
