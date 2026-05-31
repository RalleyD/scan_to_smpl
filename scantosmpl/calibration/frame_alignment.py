"""7-DoF Procrustes alignment: COLMAP arbitrary frame → SMPL canonical frame."""

from dataclasses import dataclass

import numpy as np

from scantosmpl.utils.geometry import procrustes_align


@dataclass
class FrameAlignment:
    """7-DoF similarity transform: p_smpl = scale * R @ p_colmap + translation."""

    scale: float
    rotation: np.ndarray    # (3, 3) orthonormal rotation
    translation: np.ndarray # (3,)

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """Transform a point from COLMAP frame to SMPL frame.

        Args:
            p: (3,) or (N, 3) point(s) in COLMAP frame.

        Returns:
            (3,) or (N, 3) point(s) in SMPL frame.
        """
        return self.scale * (p @ self.rotation.T) + self.translation

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Transform N points. Alias of transform_point for clarity."""
        return self.transform_point(pts)

    def transform_camera(
        self,
        R_colmap: np.ndarray,
        t_colmap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform COLMAP [R|t] extrinsics into SMPL frame.

        COLMAP convention: p_cam = R_colmap @ p_world + t_colmap
        After alignment we want: p_cam = R_smpl @ p_smpl + t_smpl

        The similarity transform is: p_smpl = s * A @ p_colmap + b
        (where A = self.rotation, b = self.translation)

        So:
            p_colmap = (p_cam - t_colmap) @ R_colmap.T    (invert extrinsics)
            p_smpl   = s * A @ p_colmap + b
            p_cam    = R_colmap @ (1/s) * A.T @ (p_smpl - b) + t_colmap
                     = (R_colmap @ A.T / s) @ p_smpl + (t_colmap - R_colmap @ A.T / s @ b)

        So:
            R_smpl = R_colmap @ A.T / s   ... but this is not a rotation (scale embedded)

        The standard result for absorbing a similarity into extrinsics:
            R_smpl = R_colmap @ R_align.T
            t_smpl = t_colmap - R_smpl @ (s * R_align @ (0,0,0)) ...

        Cleaner derivation:
            p_world_smpl = s * A @ p_world_colmap + b
            p_cam = R_colmap @ p_world_colmap + t_colmap
                  = R_colmap @ (1/s) * A.T @ (p_world_smpl - b) + t_colmap
                  = (R_colmap @ A.T / s) @ p_world_smpl
                    + (t_colmap - R_colmap @ A.T @ b / s)

        This has a scale factor 1/s in the rotation-like part, which is
        invalid for a pure rotation. For a pinhole projection we can fold
        the scale into the translation without affecting reprojection:

            R_out = R_colmap @ A.T          (pure rotation, absorb scale elsewhere)
            t_out = t_colmap / s - R_out @ b / s  ... absorbed into translation

        Equivalently, work in units of the SMPL frame directly:
            The camera centre in COLMAP frame: C_colmap = -R_colmap.T @ t_colmap
            C_smpl = s * A @ C_colmap + b
            R_smpl = R_colmap @ A.T             (world-to-cam rotation unchanged in form)
            t_smpl = -R_smpl @ C_smpl           (recompute from new centre)

        Args:
            R_colmap: (3, 3) world-to-camera rotation in COLMAP frame.
            t_colmap: (3,) world-to-camera translation in COLMAP frame.

        Returns:
            R_smpl: (3, 3) world-to-camera rotation in SMPL frame.
            t_smpl: (3,) world-to-camera translation in SMPL frame.
        """
        A = self.rotation
        s = self.scale
        b = self.translation

        # Camera centre in COLMAP world frame
        C_colmap = -R_colmap.T @ t_colmap

        # Transform camera centre to SMPL frame
        C_smpl = s * (A @ C_colmap) + b

        # Rotation: world-to-camera in SMPL frame = R_colmap @ A.T
        R_smpl = R_colmap @ A.T

        # Translation: t = -R @ C
        t_smpl = -R_smpl @ C_smpl

        return R_smpl, t_smpl


def compute_frame_alignment(
    pts_colmap: np.ndarray,
    pts_smpl: np.ndarray,
) -> FrameAlignment:
    """Compute 7-DoF Procrustes alignment from COLMAP frame to SMPL frame.

    Finds scale s, rotation A, translation b such that:
        pts_smpl ≈ s * A @ pts_colmap + b

    Uses the Umeyama (SVD-based) method via scantosmpl.utils.geometry.procrustes_align.

    Args:
        pts_colmap: (N, 3) point set in COLMAP frame (e.g. triangulated joints).
        pts_smpl: (N, 3) corresponding points in SMPL canonical frame.

    Returns:
        FrameAlignment with scale, rotation, translation.

    Raises:
        ValueError: If fewer than 4 point pairs are provided.
    """
    if len(pts_colmap) < 4:
        raise ValueError(
            f"Need ≥4 point pairs for robust Procrustes alignment, got {len(pts_colmap)}"
        )

    # procrustes_align returns (aligned_source, scale) and internally computes
    # s, R, t to map source → target.
    # We need s, R, t explicitly, so we replicate the Umeyama derivation.
    N = pts_colmap.shape[0]
    mu_c = pts_colmap.mean(axis=0)
    mu_s = pts_smpl.mean(axis=0)

    src_c = pts_colmap - mu_c
    tgt_c = pts_smpl - mu_s

    var_src = np.sum(src_c ** 2) / N
    cov = (tgt_c.T @ src_c) / N

    U, S_diag, Vt = np.linalg.svd(cov)
    d = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.diag([1.0, 1.0, np.sign(d)])

    R = U @ D @ Vt
    scale = float(np.sum(S_diag * np.diag(D)) / var_src) if var_src > 1e-12 else 1.0
    translation = mu_s - scale * (R @ mu_c)

    return FrameAlignment(
        scale=scale,
        rotation=R,
        translation=translation,
    )
