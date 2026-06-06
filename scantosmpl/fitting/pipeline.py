"""Phase 5 orchestrator: COLMAP extrinsics → triangulation → SMPL refinement."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from scantosmpl.calibration.colmap_reader import (
    ColmapCamera,
    ColmapImage,
    match_views_to_colmap,
    read_colmap_model,
)
from scantosmpl.calibration.frame_alignment import FrameAlignment, compute_frame_alignment
from scantosmpl.calibration.pipeline import CalibrationResult
from scantosmpl.calibration.undistort import build_pinhole_K, undistort_keypoints
from scantosmpl.config import FittingConfig, Phase5Config
from scantosmpl.fitting.optimiser import DEFAULT_STAGES, RefinementResult, SMPLOptimiser
from scantosmpl.hmr.consensus import ConsensusResult
from scantosmpl.smpl.joint_map import COCO_MIDPOINT_TO_SMPL, COCO_TO_SMPL
from scantosmpl.smpl.model import SMPLModel
from scantosmpl.triangulation.dlt import build_projection_matrix
from scantosmpl.triangulation.ransac import ransac_triangulate_joints
from scantosmpl.types import ViewResult
from scantosmpl.utils.geometry import compute_pa_mpjpe, project_points

logger = logging.getLogger(__name__)


@dataclass
class Phase5Result:
    """Output from the Phase 5 pipeline."""

    refined: RefinementResult
    # (J, 3) raw triangulated, ordered by joint_indices
    triangulated_joints: np.ndarray
    # (24, 3) mapped to SMPL joint ordering (zeros for unmapped)
    triangulated_joints_smpl: np.ndarray
    triangulation_quality: np.ndarray       # (J,) inlier fraction
    # (J,) mean reprojection error (px)
    triangulation_reproj_errors: np.ndarray
    cameras_smpl_frame: dict[str, tuple[np.ndarray,
                                        np.ndarray, np.ndarray]]  # R, t, K
    frame_alignment: FrameAlignment | None
    extrinsics_source: str
    metrics: dict[str, float] = field(default_factory=dict)


class Phase5Pipeline:
    """Multi-view triangulation + SMPL refinement pipeline.

    Supports two extrinsics sources:
    - "colmap": Primary. Parse COLMAP binary model, align to SMPL frame, then
      undistort keypoints and run RANSAC-DLT triangulation.
    - "self_calibration": Fallback. Use Phase 4 CalibrationResult cameras.
      Skip undistortion and frame alignment (cameras already in SMPL frame).
    """

    def __init__(
        self,
        smpl_model: SMPLModel,
        config: Phase5Config,
        fitting_config: FittingConfig | None = None,
    ) -> None:
        self.smpl = smpl_model
        self.cfg = config
        self.fitting_cfg = fitting_config or FittingConfig()
        # populated by _load_colmap_cameras
        self._colmap_cam_map: dict[str, ColmapCamera] = {}
        # EXIF orientation per view (1,3,6,8)
        self._view_orient: dict[str, int] = {}
        self._W_colmap: int = 6000  # COLMAP landscape width
        self._H_colmap: int = 4000  # COLMAP landscape height

    def run(
        self,
        views: list[ViewResult],
        consensus: ConsensusResult,
        image_dir: Path,
        calibration_result: CalibrationResult | None = None,
    ) -> Phase5Result:
        """Run the full Phase 5 pipeline.

        Args:
            views: Per-view results from Phases 1–2 (ViTPose keypoints + confs).
            consensus: Phase 3 consensus SMPL mesh.
            image_dir: Directory containing source images (for debug overlays).
            calibration_result: Phase 4 result, required if extrinsics_source=="self_calibration".

        Returns:
            Phase5Result with refined SMPL params and quality metrics.
        """
        cfg = self.cfg
        debug_dir = Path(cfg.debug_dir) if cfg.save_debug else None
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------
        # Step 1: Load camera extrinsics
        # -------------------------------------------------------------------
        if cfg.extrinsics_source == "colmap":
            cameras_smpl, alignment = self._load_colmap_cameras(
                views, consensus)
        else:
            cameras_smpl, alignment = self._load_selfcal_cameras(
                views, calibration_result)

        if not cameras_smpl:
            raise RuntimeError("No valid cameras loaded. Cannot proceed.")

        logger.info(
            "Loaded extrinsics for %d/%d views from %s",
            len(cameras_smpl), len(views), cfg.extrinsics_source,
        )

        # -------------------------------------------------------------------
        # Step 2: Undistort 2D keypoints (COLMAP path only)
        # -------------------------------------------------------------------
        kp2d_per_view, confs_per_view = self._gather_keypoints(views)

        if cfg.extrinsics_source == "colmap" and alignment is not None:
            kp2d_per_view = self._undistort_keypoints(views, kp2d_per_view)

        # -------------------------------------------------------------------
        # Step 3: RANSAC-DLT triangulation
        # -------------------------------------------------------------------
        joint_indices = self._build_joint_indices()
        logger.info("Triangulating %d joints", len(joint_indices))

        # The extended keypoint indices (into the 19-element array after _expand_midpoints)
        # e.g. [5, 6, 7, ..., 16, 17, 18] — NOT range(14)
        ext_kp_indices = [idx for idx, _ in joint_indices]

        # Build cameras dict for triangulation (COCO-indexed views only)
        tri_cameras = {
            name: cam
            for name, cam in cameras_smpl.items()
            if name in kp2d_per_view
        }
        # Build midpoint-expanded keypoints and confs
        kp2d_full, confs_full = self._expand_midpoints(
            kp2d_per_view, confs_per_view)

        pts_3d, quality, reproj_errors = ransac_triangulate_joints(
            keypoints_per_view=kp2d_full,
            confs_per_view=confs_full,
            cameras_per_view=tri_cameras,
            joint_indices=ext_kp_indices,  # actual COCO indices, not range(14)
            conf_threshold=cfg.triangulation_conf_threshold,
            reproj_threshold=cfg.ransac_reproj_threshold,
            min_inlier_views=max(2, cfg.triangulation_min_views - 1),
            n_iterations=cfg.ransac_iterations,
        )

        n_good = int((quality > 0).sum())
        logger.info(
            "Triangulation: %d/%d joints successful, mean reproj=%.1fpx",
            n_good, len(joint_indices),
            float(reproj_errors[quality > 0].mean()
                  ) if n_good > 0 else float("nan"),
        )

        # -------------------------------------------------------------------
        # Step 4: SMPL optimisation
        # -------------------------------------------------------------------
        optimiser = SMPLOptimiser(self.smpl, COCO_TO_SMPL)

        # Map triangulated joints to SMPL joint indices for L_joint
        triang_smpl = self._map_triangulated_to_smpl_joints(
            pts_3d, joint_indices)

        kp2d_tensors = {
            k: np.array(v) for k, v in kp2d_per_view.items()
            if k in cameras_smpl
        }
        confs_tensors = {
            k: np.array(v) for k, v in confs_per_view.items()
            if k in cameras_smpl
        }

        refined = optimiser.refine(
            consensus=consensus,
            triangulated_joints=triang_smpl,
            keypoints_2d=kp2d_tensors,
            confs=confs_tensors,
            cameras=cameras_smpl,
        )

        # -------------------------------------------------------------------
        # Step 5: Quality assessment
        # -------------------------------------------------------------------
        metrics = self._compute_metrics(
            refined, triang_smpl, kp2d_tensors, confs_tensors, cameras_smpl
        )
        refined.metrics.update(metrics)
        logger.info(
            "Refinement complete: PA-MPJPE=%.1fmm, mean_reproj=%.1fpx",
            metrics.get("pa_mpjpe_mm", float("nan")),
            metrics.get("mean_reproj_px", float("nan")),
        )

        # -------------------------------------------------------------------
        # Step 6: Debug output
        # -------------------------------------------------------------------
        if debug_dir is not None:
            self._save_debug(
                debug_dir, refined, pts_3d, quality, reproj_errors,
                joint_indices, cameras_smpl, alignment, metrics,
                image_dir, kp2d_per_view, confs_per_view, views,
            )

        return Phase5Result(
            refined=refined,
            triangulated_joints=pts_3d,
            triangulated_joints_smpl=triang_smpl,
            triangulation_quality=quality,
            triangulation_reproj_errors=reproj_errors,
            cameras_smpl_frame=cameras_smpl,
            frame_alignment=alignment,
            extrinsics_source=cfg.extrinsics_source,
            metrics=metrics,
        )

    # -----------------------------------------------------------------------
    # Camera loading
    # -----------------------------------------------------------------------

    def _load_colmap_cameras(
        self,
        views: list[ViewResult],
        consensus: ConsensusResult,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], FrameAlignment | None]:
        """Load COLMAP extrinsics and align to SMPL frame."""
        cfg = self.cfg
        if cfg.colmap_model_dir is None:
            raise ValueError(
                "colmap_model_dir must be set when extrinsics_source='colmap'")

        colmap_cameras, colmap_images = read_colmap_model(
            Path(cfg.colmap_model_dir))

        view_names = [v.image_path.name for v in views]
        matched, missing = match_views_to_colmap(view_names, colmap_images)

        if missing:
            logger.warning("Views not found in COLMAP: %s", missing)
        if not matched:
            raise RuntimeError("No views matched COLMAP images.")

        logger.info("COLMAP: %d cameras, %d total images, %d matched",
                    len(colmap_cameras), len(colmap_images), len(matched))

        # Build COLMAP-frame cameras (before alignment)
        colmap_frame_cams: dict[str,
                                tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._colmap_cam_map = {}
        first_cam_ref: ColmapCamera | None = None
        for name, img in matched.items():
            cam = colmap_cameras[img.camera_id]
            K = build_pinhole_K(cam)
            self._colmap_cam_map[name] = cam
            if first_cam_ref is None:
                first_cam_ref = cam
            colmap_frame_cams[name] = (img.rotation, img.translation, K)

        if first_cam_ref is not None:
            self._W_colmap = first_cam_ref.width   # 6000
            self._H_colmap = first_cam_ref.height  # 4000

        # Build per-view EXIF orientation map (reads EXIF from image files)
        self._view_orient = {}
        for view in views:
            if view.image_path.name in matched:
                self._view_orient[view.image_path.name] = self._get_exif_orient(
                    view)

        logger.info(
            "EXIF orientations: %s",
            {o: [n for n, v in self._view_orient.items() if v == o]
             for o in sorted(set(self._view_orient.values()))},
        )

        # Gather keypoints, apply per-view EXIF inverse, undistort
        kp2d_raw, confs_raw = self._gather_keypoints(views)
        kp2d_raw = self._all_to_landscape(kp2d_raw)
        for name in list(kp2d_raw):
            if name in self._colmap_cam_map:
                kp2d_raw[name] = undistort_keypoints(
                    kp2d_raw[name], self._colmap_cam_map[name])

        # RANSAC triangulation in COLMAP frame for alignment anchors.
        # Using RANSAC (not bare DLT) handles rear-view cameras where ViTPose
        # has left/right labels swapped — inconsistent observations push the
        # naive DLT point to infinity, corrupting the Procrustes scale.
        kp2d_full, confs_full = self._expand_midpoints(kp2d_raw, confs_raw)
        joint_indices = self._build_joint_indices()
        # actual COCO indices, not range(14)
        ext_kp_indices = [idx for idx, _ in joint_indices]

        pts_colmap, quality_colmap, _ = ransac_triangulate_joints(
            keypoints_per_view={n: kp2d_full[n]
                                for n in kp2d_full if n in colmap_frame_cams},
            confs_per_view={n: confs_full[n]
                            for n in confs_full if n in colmap_frame_cams},
            cameras_per_view=colmap_frame_cams,
            joint_indices=ext_kp_indices,
            conf_threshold=cfg.triangulation_conf_threshold,
            reproj_threshold=cfg.ransac_reproj_threshold,
            min_inlier_views=max(2, cfg.triangulation_min_views - 1),
            n_iterations=cfg.ransac_iterations,
        )

        # Map to SMPL joint ordering; track per-SMPL-joint quality
        pts_colmap_smpl = np.zeros((24, 3), dtype=np.float64)
        quality_smpl = np.zeros(24, dtype=np.float64)
        for j_out, (_, smpl_idx) in enumerate(joint_indices):
            if smpl_idx < 24:
                pts_colmap_smpl[smpl_idx] = pts_colmap[j_out]
                quality_smpl[smpl_idx] = quality_colmap[j_out]
        pts_smpl_ref = consensus.joints[:24]

        # Use only joints that RANSAC successfully triangulated
        valid = quality_smpl > 0
        if valid.sum() < cfg.min_alignment_joints:
            logger.warning(
                "Only %d valid joints for frame alignment (need %d). Using self-calibration.",
                valid.sum(), cfg.min_alignment_joints,
            )
            return {}, None

        alignment = compute_frame_alignment(
            pts_colmap_smpl[valid],
            pts_smpl_ref[valid],
        )
        logger.info("Frame alignment: scale=%.3f", alignment.scale)

        # Transform cameras to SMPL frame
        cameras_smpl: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, img in matched.items():
            cam = colmap_cameras[img.camera_id]
            K = build_pinhole_K(cam)
            R_smpl, t_smpl = alignment.transform_camera(
                img.rotation, img.translation)
            cameras_smpl[name] = (R_smpl, t_smpl, K)

        return cameras_smpl, alignment

    def _load_selfcal_cameras(
        self,
        views: list[ViewResult],
        calibration_result: CalibrationResult | None,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], None]:
        """Load Phase 4 self-calibration cameras (already in SMPL frame)."""
        if calibration_result is None:
            raise ValueError(
                "calibration_result required for self_calibration source")

        cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for view in views:
            name = view.image_path.name
            if view.camera is None or not view.camera.has_extrinsics:
                continue
            K = view.camera.K
            R = view.camera.rotation
            t = view.camera.translation
            cameras[name] = (R, t, K)

        return cameras, None

    # -----------------------------------------------------------------------
    # Keypoint handling
    # -----------------------------------------------------------------------

    def _gather_keypoints(
        self,
        views: list[ViewResult],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Extract ViTPose 2D keypoints and confidences from views."""
        kp2d: dict[str, np.ndarray] = {}
        confs: dict[str, np.ndarray] = {}
        for view in views:
            name = view.image_path.name
            if view.keypoints_2d is None or view.keypoint_confs is None:
                continue
            kp2d[name] = view.keypoints_2d.astype(np.float64)
            confs[name] = view.keypoint_confs.astype(np.float64)
        return kp2d, confs

    def _get_exif_orient(self, view: ViewResult) -> int:
        """Effective EXIF orientation code (1,3,6,8) after image_loader corrections.

        Reads EXIF from the image file and applies the same manual overrides
        as scantosmpl.detection.image_loader (e.g. 180° CW for cam10).
        """
        from PIL import Image as PILImage
        from scantosmpl.detection.image_loader import DEFAULT_ORIENTATION_OVERRIDES

        try:
            img = PILImage.open(view.image_path)
            exif_data = img._getexif() or {}
            orient = int(exif_data.get(274, 1))  # tag 274 = Orientation
        except Exception:
            orient = 1

        stem = view.image_path.stem
        if stem in DEFAULT_ORIENTATION_OVERRIDES:
            override = DEFAULT_ORIENTATION_OVERRIDES[stem]
            if override == 180:
                orient = {1: 3, 3: 1, 6: 8, 8: 6}.get(orient, orient)

        return orient

    def _kps_to_landscape(self, kps: np.ndarray, orient: int) -> np.ndarray:
        """Invert PIL exif_transpose + override: detection coords → COLMAP landscape.

        COLMAP uses landscape (W×H = 6000×4000) coordinates.
        Detection uses PIL-corrected coordinates (portrait for EXIF 6/8).

        Inverse transforms:
          EXIF 1 (normal):   identity
          EXIF 3 (180°):     xl = W-1-xp, yl = H-1-yp
          EXIF 6 (90° CW):   xl = yp,     yl = H-1-xp
          EXIF 8 (90° CCW):  xl = W-1-yp, yl = xp
        """
        W, H = self._W_colmap, self._H_colmap
        if orient == 1:
            return kps.copy()
        elif orient == 3:
            return np.column_stack([W - 1 - kps[:, 0], H - 1 - kps[:, 1]])
        elif orient == 6:
            return np.column_stack([kps[:, 1], H - 1 - kps[:, 0]])
        elif orient == 8:
            return np.column_stack([W - 1 - kps[:, 1], kps[:, 0]])
        else:
            logger.warning(
                "Unknown EXIF orientation %d, using identity", orient)
            return kps.copy()

    def _all_to_landscape(self, kp2d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply per-view EXIF inverse transform to convert all views to COLMAP landscape."""
        return {
            name: self._kps_to_landscape(kps, self._view_orient.get(name, 1))
            for name, kps in kp2d.items()
        }

    def _undistort_keypoints(
        self,
        views: list[ViewResult],
        kp2d_per_view: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Transform detection coords → COLMAP landscape, then undistort."""
        kp2d_ls = self._all_to_landscape(kp2d_per_view)
        result = {}
        for name, kps in kp2d_ls.items():
            if name in self._colmap_cam_map:
                result[name] = undistort_keypoints(
                    kps, self._colmap_cam_map[name])
            else:
                result[name] = kps
        return result

    def _expand_midpoints(
        self,
        kp2d: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Append midpoint joints (pelvis, neck) to each view's keypoint array.

        COCO_MIDPOINT_TO_SMPL: {(idx_a, idx_b): smpl_idx}
        We append these as extra rows so joint_indices can address them.
        """
        n_coco = 17
        n_extra = len(COCO_MIDPOINT_TO_SMPL)
        n_total = n_coco + n_extra

        kp2d_out: dict[str, np.ndarray] = {}
        confs_out: dict[str, np.ndarray] = {}

        for name, kps in kp2d.items():
            new_kps = np.zeros((n_total, 2), dtype=np.float64)
            new_confs = np.zeros(n_total, dtype=np.float64)
            new_kps[:n_coco] = kps
            new_confs[:n_coco] = confs[name]

            for i, ((a, b), _) in enumerate(COCO_MIDPOINT_TO_SMPL.items()):
                conf_a = confs[name][a]
                conf_b = confs[name][b]
                min_conf = min(conf_a, conf_b)
                if min_conf > 0:
                    new_kps[n_coco + i] = (kps[a] + kps[b]) / 2.0
                    new_confs[n_coco + i] = min_conf

            kp2d_out[name] = new_kps
            confs_out[name] = new_confs

        return kp2d_out, confs_out

    def _build_joint_indices(self) -> list[tuple[int, int]]:
        """Build list of (extended_coco_idx, smpl_idx) pairs for triangulation.

        Returns list of tuples mapping the extended keypoint index (COCO-17 +
        2 midpoints) to SMPL joint indices, in the order we triangulate.
        """
        joints = []
        # Direct COCO → SMPL
        for coco_idx, smpl_idx in COCO_TO_SMPL.items():
            joints.append((coco_idx, smpl_idx))
        # Midpoints (appended after the 17 COCO joints)
        for i, ((_, _), smpl_idx) in enumerate(COCO_MIDPOINT_TO_SMPL.items()):
            joints.append((17 + i, smpl_idx))
        return joints

    def _map_triangulated_to_smpl_joints(
        self,
        pts_3d: np.ndarray,
        joint_indices: list[tuple[int, int]],
    ) -> np.ndarray:
        """Map triangulated points (indexed by joint_indices order) to SMPL joints.

        Returns (24, 3) array where pts_3d values are placed at SMPL indices.
        Unmapped joints remain at (0, 0, 0).
        """
        smpl_joints = np.zeros((24, 3), dtype=np.float64)
        for j_out, (_, smpl_idx) in enumerate(joint_indices):
            if smpl_idx < 24:
                smpl_joints[smpl_idx] = pts_3d[j_out]
        return smpl_joints

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def _compute_metrics(
        self,
        refined: RefinementResult,
        triang_smpl: np.ndarray,
        kp2d: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
        cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, float]:
        valid = np.linalg.norm(triang_smpl, axis=1) > 1e-6
        if valid.sum() >= 2:
            pa_mpjpe = compute_pa_mpjpe(
                refined.joints[valid],
                triang_smpl[valid],
            ) * 1000
        else:
            pa_mpjpe = float("nan")

        # view_name -> list of reproj errors for valid joints
        per_view_reproj: dict[str, list[float]] = defaultdict(list)

        for name, (R, t, K) in cameras.items():
            if name not in kp2d:
                continue
            kps = kp2d[name]
            c = confs[name]
            for coco_idx, smpl_idx in COCO_TO_SMPL.items():
                if c[coco_idx] < 0.3:
                    continue
                proj = project_points(
                    refined.joints[smpl_idx:smpl_idx+1], R, t, K
                )[0]
                err = float(np.linalg.norm(proj - kps[coco_idx]))
                per_view_reproj[name].append(err)

        reproj_errors = [e for errs in per_view_reproj.values() for e in errs]

        median_reproj = float(np.median(reproj_errors)
                              ) if reproj_errors else float("nan")
        mean_reproj = float(np.mean(reproj_errors)
                            ) if reproj_errors else float("nan")

        # compute per-view means
        per_view_means = {
            name: float(np.mean(errs))
            if errs else float("nan")
            for name, errs in per_view_reproj.items()
        }
        # using per-view means, compute median of these means - this provides the center
        median_of_means = float(np.nanmedian(list(per_view_means.values())))
        # using both, compute the MAD (median absolute deviation) of the per-view means to get a robust measure of reprojection consistency across views
        # this provides outlier threhsold
        mad_of_means = float(
            np.median(
                np.abs(
                    np.array(list(per_view_means.values())) - median_of_means
                )
            )
        )
        # the median spread threshold is resistant to outliers, so if a few views have very high reprojection error, they won't skew the threshold as much as using mean would
        # compute inlier means if per-view mean is within threshold of the median + 3*MAD (a common choice for outlier detection, roughly analogous to 3 standard deviations in a normal distribution))
        inlier_view_means = [
            mean for mean in per_view_means.values()
            if mean <= median_of_means + (self.cfg.reprojection_mad_multiplier * mad_of_means)
        ]
        n_outlier_view = len(per_view_means) - len(inlier_view_means)

        mean_reproj_inliers = float(np.mean(inlier_view_means))

        return {
            "pa_mpjpe_mm": pa_mpjpe,
            "mean_reproj_px": mean_reproj,
            "median_reproj_px": median_reproj,
            "mean_reproj_inliers_px": mean_reproj_inliers,
            "n_outlier_views": n_outlier_view,
            "n_reproj_terms": len(reproj_errors),
        }

    # -----------------------------------------------------------------------
    # Debug output
    # -----------------------------------------------------------------------

    def _save_debug(
        self,
        debug_dir: Path,
        refined: RefinementResult,
        pts_3d: np.ndarray,
        quality: np.ndarray,
        reproj_errors: np.ndarray,
        joint_indices: list[tuple[int, int]],
        cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        alignment: FrameAlignment | None,
        metrics: dict[str, float],
        image_dir: Path,
        kp2d: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
        views: list[ViewResult],
    ) -> None:
        # triangulated_joints.json
        triang_data = {}
        for j_out, (ext_idx, smpl_idx) in enumerate(joint_indices):
            triang_data[f"joint_{smpl_idx}"] = {
                "ext_idx": ext_idx,
                "smpl_idx": smpl_idx,
                "position": pts_3d[j_out].tolist(),
                "quality": float(quality[j_out]),
                "reproj_error_px": float(reproj_errors[j_out]) if reproj_errors[j_out] < 1e9 else None,
            }
        with open(debug_dir / "triangulated_joints.json", "w") as f:
            json.dump(triang_data, f, indent=2)

        # refinement_results.json
        result_data = {
            "betas": refined.betas.tolist(),
            "body_pose": refined.body_pose.tolist(),
            "global_orient": refined.global_orient.tolist(),
            "translation": refined.translation.tolist(),
            "scale": refined.scale,
            "metrics": refined.metrics,
            "frame_alignment": {
                "scale": alignment.scale,
                "rotation": alignment.rotation.tolist(),
                "translation": alignment.translation.tolist(),
            } if alignment is not None else None,
            "cameras": {
                name: {"R": R.tolist(), "t": t.tolist()}
                for name, (R, t, _) in cameras.items()
            },
        }
        with open(debug_dir / "refinement_results.json", "w") as f:
            json.dump(result_data, f, indent=2)

        # summary.txt
        lines = [
            "=== Phase 5 Summary ===",
            f"PA-MPJPE:        {metrics.get('pa_mpjpe_mm', float('nan')):.1f} mm",
            f"Mean reprojection: {metrics.get('mean_reproj_px', float('nan')):.1f} px",
            f"Cameras:         {len(cameras)}",
            f"Triangulated joints: {int((quality > 0).sum())}/{len(joint_indices)}",
            "",
            "Acceptance Criteria:",
            f"  5.2 frame alignment reproj <15px: {'PASS' if metrics.get('mean_reproj_px', 999) < 15 else 'FAIL (check alignment)'}",
            f"  5.5 mean reproj <15px:             {'PASS' if metrics.get('mean_reproj_px', 999) < 15 else 'FAIL'}",
            "",
            "Loss history (final loss per stage):",
        ]
        for stage_name, hist in refined.loss_history.items():
            if hist:
                lines.append(f"  {stage_name}: {hist[0]:.4f} → {hist[-1]:.4f}")

        with open(debug_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines))

        # convergence.png
        self._plot_convergence(debug_dir, refined.loss_history)

        # camera_positions.png
        self._plot_cameras(debug_dir, cameras)

        # reprojection_overlay/
        overlay_dir = debug_dir / "reprojection_overlay"
        overlay_dir.mkdir(exist_ok=True)
        self._save_reprojection_overlays(
            overlay_dir, refined, cameras, kp2d, confs, views, image_dir
        )

    def _plot_convergence(
        self,
        debug_dir: Path,
        loss_history: dict[str, list[float]],
    ) -> None:
        if not loss_history:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        offset = 0
        for stage_name, hist in loss_history.items():
            xs = list(range(offset, offset + len(hist)))
            ax.plot(xs, hist, label=stage_name)
            ax.axvline(offset, color="gray", linestyle="--", alpha=0.4)
            offset += len(hist)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Phase 5 Optimisation Loss")
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(debug_dir / "convergence.png", dpi=100)
        plt.close(fig)

    def _plot_cameras(
        self,
        debug_dir: Path,
        cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        from scantosmpl.utils.geometry import camera_center
        centers = {n: camera_center(R, t) for n, (R, t, _) in cameras.items()}
        if not centers:
            return

        fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5))
        for name, C in centers.items():
            ax_top.scatter(C[0], C[2], s=30)
            ax_top.annotate(name[:8], (C[0], C[2]), fontsize=6)
            ax_side.scatter(C[0], C[1], s=30)
            ax_side.annotate(name[:8], (C[0], C[1]), fontsize=6)

        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Z (m)")
        ax_top.set_title("Camera positions (top-down XZ)")
        ax_top.set_aspect("equal")
        ax_side.set_xlabel("X (m)")
        ax_side.set_ylabel("Y (m)")
        ax_side.set_title("Camera positions (side XY)")
        ax_side.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(debug_dir / "camera_positions.png", dpi=100)
        plt.close(fig)

    def _landscape_to_det(self, xy: np.ndarray, orient: int) -> np.ndarray:
        """COLMAP landscape coords → detection display coords (forward EXIF transform)."""
        W, H = self._W_colmap, self._H_colmap
        if orient == 1:
            return xy.copy()
        elif orient == 3:   # 180°
            return np.column_stack([W - 1 - xy[:, 0], H - 1 - xy[:, 1]])
        elif orient == 6:   # 90° CW forward: det_x=H-1-yl, det_y=xl
            return np.column_stack([H - 1 - xy[:, 1], xy[:, 0]])
        elif orient == 8:   # 90° CCW forward: det_x=yl, det_y=W-1-xl
            return np.column_stack([xy[:, 1], W - 1 - xy[:, 0]])
        return xy.copy()

    def _save_reprojection_overlays(
        self,
        overlay_dir: Path,
        refined: RefinementResult,
        cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        kp2d: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
        views: list[ViewResult],
        image_dir: Path,
    ) -> None:
        from PIL import ImageOps
        from scantosmpl.detection.image_loader import DEFAULT_ORIENTATION_OVERRIDES

        view_map = {v.image_path.name: v for v in views}

        for name, (R, t, K) in cameras.items():
            if name not in view_map:
                continue
            img_path = image_dir / name
            if not img_path.exists():
                continue

            orient = self._view_orient.get(name, 1)

            try:
                # Open with EXIF correction + manual override so person appears upright
                img = ImageOps.exif_transpose(Image.open(img_path))
                stem = Path(img_path).stem
                if stem in DEFAULT_ORIENTATION_OVERRIDES:
                    deg = DEFAULT_ORIENTATION_OVERRIDES[stem]
                    img = img.rotate(-deg, expand=True)  # PIL rotate is CCW
                img = img.convert("RGB")

                # Downsample
                scale_f = 1000 / max(img.size)
                w, h = int(img.size[0] * scale_f), int(img.size[1] * scale_f)
                img = img.resize((w, h), Image.LANCZOS)
                draw = ImageDraw.Draw(img)

                # Project refined SMPL joints (landscape coords) → detection space → draw
                for smpl_idx in range(min(24, len(refined.joints))):
                    proj_ls = project_points(
                        refined.joints[smpl_idx:smpl_idx+1], R, t, K
                    )  # (1,2) landscape
                    proj_det = self._landscape_to_det(
                        proj_ls, orient)[0] * scale_f
                    x, y = int(proj_det[0]), int(proj_det[1])
                    if 0 <= x < w and 0 <= y < h:
                        draw.ellipse([x-4, y-4, x+4, y+4],
                                     fill=(0, 200, 0), outline=(0, 200, 0))

                # Draw ViTPose detections (landscape undistorted coords → detection space)
                if name in kp2d and name in confs:
                    for i, (kp_ls, c) in enumerate(zip(kp2d[name][:17], confs[name][:17])):
                        if c < 0.3:
                            continue
                        kp_det = self._landscape_to_det(
                            kp_ls.reshape(1, 2), orient)[0] * scale_f
                        x, y = int(kp_det[0]), int(kp_det[1])
                        if 0 <= x < w and 0 <= y < h:
                            draw.ellipse(
                                [x-3, y-3, x+3, y+3], fill=(255, 100, 0), outline=(255, 100, 0))

                img.save(overlay_dir / f"{name}_overlay.jpg")
            except Exception as exc:
                logger.debug("Could not save overlay for %s: %s", name, exc)
