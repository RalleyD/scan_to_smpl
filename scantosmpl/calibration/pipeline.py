"""Phase 4 calibration pipeline: PnP self-calibration for camera extrinsic recovery."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from scantosmpl.calibration.correspondence import CorrespondenceBuilder
from scantosmpl.calibration.intrinsics import get_intrinsics_for_view
from scantosmpl.calibration.pnp_solver import PnPResult, PnPSolver
from scantosmpl.config import CalibrationConfig
from scantosmpl.hmr.consensus import ConsensusResult
from scantosmpl.types import ViewResult

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Aggregated result from Phase 4 calibration."""

    pnp_results: dict[str, PnPResult]
    n_views_solved: int = 0
    n_views_dense: int = 0
    n_views_sparse: int = 0
    n_views_failed: int = 0
    camera_centers: dict[str, np.ndarray] = field(default_factory=dict)
    mean_reprojection_error: float = float("inf")
    geometry_plausible: bool = False
    geometry_stats: dict[str, float] = field(default_factory=dict)
    ab_comparison: dict[str, dict[str, float]] | None = None


class CalibrationPipeline:
    """Orchestrates per-view PnP calibration using the consensus SMPL mesh."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    def calibrate(
        self,
        views: list[ViewResult],
        consensus: ConsensusResult,
        image_dir: Path,
        debug_dir: Path | None = None,
    ) -> CalibrationResult:
        """
        Run PnP calibration on all views.

        Args:
            views: Per-view results from Phases 1-2.
            consensus: Phase 3 consensus mesh.
            image_dir: Directory containing source images.
            debug_dir: Optional directory for debug output.

        Returns:
            CalibrationResult with per-view extrinsics and quality metrics.
        """
        cfg = self.config
        if debug_dir is None:
            debug_dir = cfg.debug_dir

        corr_builder = CorrespondenceBuilder(
            consensus.vertices, consensus.joints
        )
        solver_dense = PnPSolver(
            pnp_method=cfg.pnp_method,
            ransac_threshold=cfg.ransac_threshold,
            ransac_iterations=cfg.ransac_iterations,
            min_inliers=cfg.min_inliers,
            refine_lm=cfg.refine_lm,
        )
        solver_sparse = PnPSolver(
            pnp_method=cfg.pnp_method,
            ransac_threshold=cfg.ransac_threshold,
            ransac_iterations=cfg.ransac_iterations,
            min_inliers=cfg.min_inliers_sparse,
            refine_lm=cfg.refine_lm,
        )

        pnp_results: dict[str, PnPResult] = {}
        dense_views: list[str] = []
        sparse_views: list[str] = []

        # Step 2: Per-view PnP
        for view in views:
            name = view.image_path.name
            img_path = image_dir / name if not view.image_path.is_absolute() else view.image_path
            if not img_path.exists():
                img_path = view.image_path

            img = Image.open(img_path)
            K = get_intrinsics_for_view(view, img.size)

            has_dense = (
                view.dense_keypoints_2d is not None
                and view.dense_keypoint_confs is not None
                and cfg.use_dense_keypoints
            )

            if has_dense:
                # Dense PnP (138 correspondences)
                pts_3d, pts_2d, confs = corr_builder.build_dense_correspondences(view)
                result = solver_dense.solve(
                    pts_3d, pts_2d, confs, K,
                    conf_threshold=cfg.dense_conf_threshold,
                    correspondence_type="dense_138",
                )

                # Fallback: if dense fails, try sparse on the same view.
                # Dense surface vertices are sensitive to consensus pose averaging;
                # sparse joints are more stable.
                if not result.success and view.keypoints_2d is not None:
                    logger.info(
                        "%s: dense PnP failed (%d inliers), falling back to sparse",
                        name, result.n_inliers,
                    )
                    pts_3d_s, pts_2d_s, confs_s = corr_builder.build_sparse_correspondences(view)
                    result = solver_sparse.solve(
                        pts_3d_s, pts_2d_s, confs_s, K,
                        conf_threshold=cfg.sparse_conf_threshold,
                        correspondence_type="dense_sparse_fallback",
                    )

                dense_views.append(name)
            else:
                # Sparse PnP (12-14 COCO correspondences)
                if view.keypoints_2d is None or view.keypoint_confs is None:
                    logger.warning("Skipping %s: no keypoints available", name)
                    pnp_results[name] = PnPResult(
                        success=False, correspondence_type="none",
                    )
                    continue

                pts_3d, pts_2d, confs = corr_builder.build_sparse_correspondences(view)
                result = solver_sparse.solve(
                    pts_3d, pts_2d, confs, K,
                    conf_threshold=cfg.sparse_conf_threshold,
                    correspondence_type="sparse_coco",
                )
                sparse_views.append(name)

            # Quality gate
            if result.success and result.reprojection_error > cfg.max_reprojection_error:
                logger.warning(
                    "%s: reprojection error %.1fpx exceeds %.1fpx threshold — rejecting",
                    name, result.reprojection_error, cfg.max_reprojection_error,
                )
                result = PnPResult(
                    success=False,
                    n_correspondences=result.n_correspondences,
                    n_inliers=result.n_inliers,
                    reprojection_error=result.reprojection_error,
                    correspondence_type=result.correspondence_type,
                )

            # Store extrinsics in view
            if result.success and view.camera is not None:
                view.camera.rotation = result.rotation
                view.camera.translation = result.translation

            pnp_results[name] = result
            status = "OK" if result.success else "FAIL"
            logger.info(
                "%s: %s (%s, inliers=%d/%d, reproj=%.1fpx)",
                name, status, result.correspondence_type,
                result.n_inliers, result.n_correspondences,
                result.reprojection_error,
            )

        # Step 3: A/B comparison — run sparse PnP on dense views
        ab_comparison = self._ab_comparison(
            views, dense_views, corr_builder, solver_sparse,
            image_dir, pnp_results,
        )

        # Step 4: Camera geometry validation
        solved = {n: r for n, r in pnp_results.items() if r.success}
        camera_centers = {
            n: r.cam_center for n, r in solved.items() if r.cam_center is not None
        }
        geometry_plausible, geometry_stats = self._validate_geometry(camera_centers)

        # Aggregate stats
        n_solved = len(solved)
        n_dense = sum(1 for n in dense_views if pnp_results[n].success)
        n_sparse = sum(1 for n in sparse_views if pnp_results[n].success)
        n_failed = len(pnp_results) - n_solved
        reproj_errors = [r.reprojection_error for r in solved.values()]
        mean_reproj = float(np.mean(reproj_errors)) if reproj_errors else float("inf")

        result = CalibrationResult(
            pnp_results=pnp_results,
            n_views_solved=n_solved,
            n_views_dense=n_dense,
            n_views_sparse=n_sparse,
            n_views_failed=n_failed,
            camera_centers=camera_centers,
            mean_reprojection_error=mean_reproj,
            geometry_plausible=geometry_plausible,
            geometry_stats=geometry_stats,
            ab_comparison=ab_comparison,
        )

        # Step 5: Debug output
        if cfg.save_debug:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            self._save_debug(result, debug_dir, dense_views, sparse_views)

        return result

    def _ab_comparison(
        self,
        views: list[ViewResult],
        dense_view_names: list[str],
        corr_builder: CorrespondenceBuilder,
        solver_sparse: PnPSolver,
        image_dir: Path,
        dense_results: dict[str, PnPResult],
    ) -> dict[str, dict[str, float]]:
        """Run sparse PnP on dense views for criterion 4.6 A/B comparison."""
        comparison: dict[str, dict[str, float]] = {}
        view_map = {v.image_path.name: v for v in views}

        for name in dense_view_names:
            view = view_map.get(name)
            if view is None or view.keypoints_2d is None:
                continue

            img_path = image_dir / name if not view.image_path.is_absolute() else view.image_path
            if not img_path.exists():
                img_path = view.image_path
            img = Image.open(img_path)
            K = get_intrinsics_for_view(view, img.size)

            pts_3d, pts_2d, confs = corr_builder.build_sparse_correspondences(view)
            sparse_result = solver_sparse.solve(
                pts_3d, pts_2d, confs, K,
                conf_threshold=self.config.sparse_conf_threshold,
                correspondence_type="sparse_coco",
            )

            dense_reproj = dense_results[name].reprojection_error
            sparse_reproj = sparse_result.reprojection_error if sparse_result.success else float("inf")
            comparison[name] = {
                "dense_reproj_px": dense_reproj,
                "sparse_reproj_px": sparse_reproj,
                "dense_wins": dense_reproj < sparse_reproj,
            }

        return comparison

    def _validate_geometry(
        self,
        camera_centers: dict[str, np.ndarray],
    ) -> tuple[bool, dict[str, float]]:
        """Validate that camera positions form a plausible scanner layout."""
        if len(camera_centers) < 3:
            return False, {"reason": "too_few_cameras"}

        centers = np.array(list(camera_centers.values()))  # (N, 3)

        # Radial distances from origin (where the subject is)
        radii = np.linalg.norm(centers[:, [0, 2]], axis=1)  # XZ plane
        radial_mean = float(radii.mean())
        radial_std = float(radii.std())
        radial_cov = radial_std / radial_mean if radial_mean > 0.01 else 999.0

        # Angular coverage in XZ plane
        angles = np.arctan2(centers[:, 2], centers[:, 0])  # radians
        angles_sorted = np.sort(angles)
        if len(angles_sorted) > 1:
            gaps = np.diff(angles_sorted)
            # Include the wrap-around gap
            wrap_gap = 2 * np.pi - (angles_sorted[-1] - angles_sorted[0])
            all_gaps = np.append(gaps, wrap_gap)
            max_gap_deg = float(np.degrees(all_gaps.max()))
            coverage_deg = 360.0 - max_gap_deg
        else:
            coverage_deg = 0.0

        # Height clustering: std of Y values
        height_std = float(centers[:, 1].std())

        stats = {
            "radial_mean_m": radial_mean,
            "radial_std_m": radial_std,
            "radial_cov": radial_cov,
            "angular_coverage_deg": coverage_deg,
            "height_std_m": height_std,
            "n_cameras": len(camera_centers),
        }

        plausible = radial_cov < 0.3 and coverage_deg > 120.0
        return plausible, stats

    def _save_debug(
        self,
        result: CalibrationResult,
        debug_dir: Path,
        dense_views: list[str],
        sparse_views: list[str],
    ) -> None:
        """Write JSON results, summary text, and camera position plot."""
        # --- JSON ---
        json_data: dict = {}
        for name, pnp in result.pnp_results.items():
            entry: dict = {
                "success": pnp.success,
                "correspondence_type": pnp.correspondence_type,
                "n_correspondences": pnp.n_correspondences,
                "n_inliers": pnp.n_inliers,
                "reprojection_error_px": round(pnp.reprojection_error, 2),
            }
            if pnp.success and pnp.rotation is not None:
                entry["rotation"] = pnp.rotation.tolist()
                entry["translation"] = pnp.translation.tolist()
                entry["camera_center"] = pnp.cam_center.tolist()
            json_data[name] = entry

        json_data["_summary"] = {
            "n_views_solved": result.n_views_solved,
            "n_views_dense": result.n_views_dense,
            "n_views_sparse": result.n_views_sparse,
            "n_views_failed": result.n_views_failed,
            "mean_reprojection_error_px": round(result.mean_reprojection_error, 2),
            "geometry_plausible": result.geometry_plausible,
            "geometry_stats": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in result.geometry_stats.items()
            },
        }

        with open(debug_dir / "calibration_results.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # --- Summary text ---
        lines = [
            "=" * 72,
            "Phase 4 Calibration Summary — PnP Self-Calibration",
            "=" * 72,
            "",
            f"Views solved       : {result.n_views_solved}",
            f"  Dense (138 kps)  : {result.n_views_dense}",
            f"  Sparse (COCO)    : {result.n_views_sparse}",
            f"  Failed           : {result.n_views_failed}",
            f"Mean reproj error  : {result.mean_reprojection_error:.2f} px",
            "",
            "Per-View Results:",
            f"  {'View':<28} {'Type':<14} {'Inliers':<12} {'Reproj(px)':<12} {'Status'}",
            "  " + "-" * 70,
        ]

        for name, pnp in result.pnp_results.items():
            status = "OK" if pnp.success else "FAIL"
            inlier_str = f"{pnp.n_inliers}/{pnp.n_correspondences}"
            lines.append(
                f"  {name:<28} {pnp.correspondence_type:<14} "
                f"{inlier_str:<12} {pnp.reprojection_error:>8.2f}    {status}"
            )

        # A/B comparison
        if result.ab_comparison:
            lines.extend([
                "",
                "A/B Comparison (criterion 4.6): Dense vs Sparse on same views",
                f"  {'View':<28} {'Dense(px)':<12} {'Sparse(px)':<12} {'Winner'}",
                "  " + "-" * 56,
            ])
            dense_wins = 0
            for name, comp in result.ab_comparison.items():
                winner = "dense" if comp["dense_wins"] else "sparse"
                if comp["dense_wins"]:
                    dense_wins += 1
                sparse_str = (
                    f"{comp['sparse_reproj_px']:.2f}"
                    if comp["sparse_reproj_px"] < 1e6 else "FAIL"
                )
                lines.append(
                    f"  {name:<28} {comp['dense_reproj_px']:>8.2f}    "
                    f"{sparse_str:>8}    {winner}"
                )
            total_ab = len(result.ab_comparison)
            lines.append(
                f"  Dense wins: {dense_wins}/{total_ab}"
            )

        # Geometry validation
        lines.extend([
            "",
            "Camera Geometry Validation:",
        ])
        for k, v in result.geometry_stats.items():
            lines.append(f"  {k:<24}: {v}")
        lines.append(
            f"  Plausible              : {'YES' if result.geometry_plausible else 'NO'}"
        )

        # Acceptance criteria summary
        n_dense_total = len(dense_views)
        n_sparse_total = len(sparse_views)
        lines.extend([
            "",
            "Acceptance Criteria:",
            f"  4.1 Dense >=90% success   : {result.n_views_dense}/{n_dense_total}"
            f"  {'PASS' if n_dense_total > 0 and result.n_views_dense / n_dense_total >= 0.9 else 'FAIL'}",
            f"  4.2 Sparse >=50% success  : {result.n_views_sparse}/{n_sparse_total}"
            f"  {'PASS' if n_sparse_total > 0 and result.n_views_sparse / n_sparse_total >= 0.5 else 'FAIL' if n_sparse_total > 0 else 'N/A'}",
            f"  4.3 Geometry plausible    : {'PASS' if result.geometry_plausible else 'FAIL'}",
            f"  4.4 Reproj < 80px         : {'PASS' if result.mean_reprojection_error < 80.0 else 'FAIL'}",
        ])
        if result.ab_comparison:
            dense_wins = sum(1 for c in result.ab_comparison.values() if c["dense_wins"])
            lines.append(
                f"  4.6 Dense outperforms     : {dense_wins}/{len(result.ab_comparison)}"
                f"  {'PASS' if dense_wins > len(result.ab_comparison) / 2 else 'FAIL'}"
            )

        lines.extend(["", "=" * 72, ""])
        with open(debug_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines))

        # --- Camera position plot ---
        self._save_camera_plot(result.camera_centers, debug_dir)

        logger.info("Debug output saved to %s", debug_dir)

    def _save_camera_plot(
        self,
        camera_centers: dict[str, np.ndarray],
        debug_dir: Path,
    ) -> None:
        """Save a top-down (XZ) and side (YZ) camera position plot."""
        if not camera_centers:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping camera plot")
            return

        centers = np.array(list(camera_centers.values()))
        names = list(camera_centers.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Top-down view (XZ plane)
        ax1.scatter(centers[:, 0], centers[:, 2], c="steelblue", s=60, zorder=5)
        for i, n in enumerate(names):
            ax1.annotate(
                n.replace(".JPG", ""), (centers[i, 0], centers[i, 2]),
                fontsize=6, ha="center", va="bottom",
            )
        ax1.scatter([0], [0], c="red", s=100, marker="x", zorder=10, label="Subject")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        ax1.set_title("Camera Positions — Top-Down (XZ)")
        ax1.set_aspect("equal")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Side view (XY plane)
        ax2.scatter(centers[:, 0], centers[:, 1], c="steelblue", s=60, zorder=5)
        for i, n in enumerate(names):
            ax2.annotate(
                n.replace(".JPG", ""), (centers[i, 0], centers[i, 1]),
                fontsize=6, ha="center", va="bottom",
            )
        ax2.scatter([0], [0], c="red", s=100, marker="x", zorder=10, label="Subject")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("Camera Positions — Side (XY)")
        ax2.set_aspect("equal")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(debug_dir / "camera_positions.png", dpi=150)
        plt.close(fig)
