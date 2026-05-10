"""PnP solver for camera extrinsic recovery."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from scantosmpl.utils.geometry import camera_center, project_points

logger = logging.getLogger(__name__)

# OpenCV PnP method lookup
_PNP_METHODS = {
    "SOLVEPNP_ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "SOLVEPNP_EPNP": cv2.SOLVEPNP_EPNP,
    "SOLVEPNP_P3P": cv2.SOLVEPNP_P3P,
    "SOLVEPNP_SQPNP": cv2.SOLVEPNP_SQPNP,
}


@dataclass
class PnPResult:
    """Result from a single PnP solve."""

    success: bool
    rotation: np.ndarray | None = None       # (3, 3) rotation matrix
    translation: np.ndarray | None = None    # (3,) translation vector
    rvec: np.ndarray | None = None           # (3,) Rodrigues vector
    tvec: np.ndarray | None = None           # (3,) translation (OpenCV format)
    inliers: np.ndarray | None = None        # inlier indices from RANSAC
    n_correspondences: int = 0
    n_inliers: int = 0
    reprojection_error: float = float("inf")  # mean reprojection error on inliers (px)
    correspondence_type: str = "unknown"      # "dense_138" or "sparse_coco"
    cam_center: np.ndarray | None = None     # (3,) world-space camera position


class PnPSolver:
    """Solves PnP for camera extrinsic recovery using OpenCV.

    Workflow:
        1. Filter correspondences by confidence threshold.
        2. solvePnPRansac for robust initial estimate.
        3. solvePnPRefineLM on inliers for sub-pixel refinement.
        4. Compute reprojection error and quality metrics.
    """

    def __init__(
        self,
        pnp_method: str = "SOLVEPNP_ITERATIVE",
        ransac_threshold: float = 8.0,
        ransac_iterations: int = 5000,
        min_inliers: int = 20,
        refine_lm: bool = True,
    ) -> None:
        self.pnp_method = _PNP_METHODS.get(pnp_method, cv2.SOLVEPNP_ITERATIVE)
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.min_inliers = min_inliers
        self.refine_lm = refine_lm

    def solve(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        confidences: np.ndarray,
        K: np.ndarray,
        conf_threshold: float = 0.3,
        correspondence_type: str = "dense_138",
    ) -> PnPResult:
        """
        Solve PnP with RANSAC and optional LM refinement.

        Args:
            pts_3d: (N, 3) 3D points in world frame.
            pts_2d: (N, 2) 2D image points.
            confidences: (N,) confidence scores.
            K: (3, 3) intrinsic matrix.
            conf_threshold: Minimum confidence to include a correspondence.
            correspondence_type: Label for logging ("dense_138" or "sparse_coco").

        Returns:
            PnPResult with camera extrinsics or failure indication.
        """
        # Filter by confidence
        mask = confidences >= conf_threshold
        pts_3d_f = pts_3d[mask].astype(np.float64)
        pts_2d_f = pts_2d[mask].astype(np.float64)
        n_filtered = int(mask.sum())

        if n_filtered < 4:
            logger.warning(
                "PnP %s: only %d points after confidence filter (need >= 4)",
                correspondence_type, n_filtered,
            )
            return PnPResult(
                success=False,
                n_correspondences=n_filtered,
                correspondence_type=correspondence_type,
            )

        # Select PnP method based on point count
        method = self.pnp_method
        if n_filtered < 10:
            method = cv2.SOLVEPNP_EPNP

        # RANSAC PnP
        dist_coeffs = np.zeros(4, dtype=np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d_f,
            imagePoints=pts_2d_f,
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            flags=method,
            reprojectionError=self.ransac_threshold,
            iterationsCount=self.ransac_iterations,
        )

        if not success or inliers is None or len(inliers) < self.min_inliers:
            n_inl = len(inliers) if inliers is not None else 0
            logger.warning(
                "PnP %s: RANSAC %s (inliers=%d, need=%d)",
                correspondence_type,
                "failed" if not success else "too few inliers",
                n_inl, self.min_inliers,
            )
            return PnPResult(
                success=False,
                n_correspondences=n_filtered,
                n_inliers=n_inl,
                correspondence_type=correspondence_type,
            )

        inlier_idx = inliers.flatten()

        # LM refinement on inliers
        if self.refine_lm and len(inlier_idx) >= 4:
            pts_3d_inl = pts_3d_f[inlier_idx]
            pts_2d_inl = pts_2d_f[inlier_idx]
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=pts_3d_inl,
                imagePoints=pts_2d_inl,
                cameraMatrix=K,
                distCoeffs=dist_coeffs,
                rvec=rvec.copy(),
                tvec=tvec.copy(),
            )

        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        # Compute reprojection error on inliers
        pts_3d_inl = pts_3d_f[inlier_idx]
        pts_2d_inl = pts_2d_f[inlier_idx]
        reproj = project_points(pts_3d_inl, R, t, K)
        reproj_err = float(np.linalg.norm(reproj - pts_2d_inl, axis=1).mean())

        cc = camera_center(R, t)

        return PnPResult(
            success=True,
            rotation=R,
            translation=t,
            rvec=rvec.flatten(),
            tvec=tvec.flatten(),
            inliers=inlier_idx,
            n_correspondences=n_filtered,
            n_inliers=len(inlier_idx),
            reprojection_error=reproj_err,
            correspondence_type=correspondence_type,
            cam_center=cc,
        )
