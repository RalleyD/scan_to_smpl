"""RANSAC-robust triangulation wrapper."""

import numpy as np

from scantosmpl.triangulation.dlt import build_projection_matrix, triangulate_point
from scantosmpl.utils.geometry import project_points


def ransac_triangulate_point(
    pts_2d: np.ndarray,
    projections: list[np.ndarray],
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    Ks: list[np.ndarray],
    weights: np.ndarray | None = None,
    reproj_threshold: float = 10.0,
    min_inlier_views: int = 2,
    n_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """RANSAC-robust triangulation from V views.

    For each RANSAC iteration:
    1. Sample a minimal pair of views (2 views).
    2. Triangulate the 3D point from that pair.
    3. Count inliers: views where reprojection error < reproj_threshold.
    4. Keep the hypothesis with the most inliers.
    5. Final DLT using all inlier views.

    Args:
        pts_2d: (V, 2) undistorted 2D observations.
        projections: V (3, 4) projection matrices P = K @ [R|t].
        Rs: V (3, 3) rotation matrices (for reprojection).
        ts: V (3,) translation vectors (for reprojection).
        Ks: V (3, 3) intrinsic matrices (for reprojection).
        weights: (V,) confidence weights. None = uniform.
        reproj_threshold: Pixel threshold to classify a view as inlier.
        min_inlier_views: Minimum inliers for a valid solution.
        n_iterations: RANSAC iterations.

    Returns:
        pt_3d: (3,) best triangulated point (or zeros if failed).
        inlier_mask: (V,) boolean mask of inlier views.
        mean_reproj_err: Mean reprojection error on inlier views (px).
    """
    V = len(projections)
    if weights is None:
        weights = np.ones(V, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if V < 2:
        return np.zeros(3), np.zeros(V, dtype=bool), float("inf")

    best_inliers = np.zeros(V, dtype=bool)
    best_n_inliers = 0
    best_pt = np.zeros(3)

    for _ in range(n_iterations):
        # Sample 2 views (weighted by confidence)
        w_norm = weights / (weights.sum() + 1e-12)
        idx = np.random.choice(V, size=min(2, V), replace=False, p=w_norm)

        try:
            pt = triangulate_point(
                pts_2d[idx],
                [projections[i] for i in idx],
                weights[idx],
            )
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Reject points behind any sampled camera
        if not _in_front_of_cameras(pt, [Rs[i] for i in idx], [ts[i] for i in idx]):
            continue

        # Count inliers across all views
        inliers = _compute_inliers(pt, pts_2d, Rs, ts, Ks, reproj_threshold)
        n_inliers = int(inliers.sum())

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inliers.copy()
            best_pt = pt.copy()

    if best_n_inliers < min_inlier_views:
        return np.zeros(3), np.zeros(V, dtype=bool), float("inf")

    # Final DLT with all inlier views
    inlier_idx = np.where(best_inliers)[0]
    try:
        final_pt = triangulate_point(
            pts_2d[inlier_idx],
            [projections[i] for i in inlier_idx],
            weights[inlier_idx],
        )
    except (np.linalg.LinAlgError, ValueError):
        final_pt = best_pt

    # Compute final mean reprojection error
    reproj_errors = _reprojection_errors(final_pt, pts_2d, Rs, ts, Ks)
    inlier_errors = reproj_errors[best_inliers]
    mean_err = float(inlier_errors.mean()) if len(inlier_errors) > 0 else float("inf")

    return final_pt, best_inliers, mean_err


def ransac_triangulate_joints(
    keypoints_per_view: dict[str, np.ndarray],
    confs_per_view: dict[str, np.ndarray],
    cameras_per_view: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    joint_indices: list[int],
    conf_threshold: float = 0.3,
    reproj_threshold: float = 10.0,
    min_inlier_views: int = 2,
    n_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC-robust triangulation for all joints.

    Args:
        keypoints_per_view: {view_name: (J, 2)} undistorted 2D keypoints.
        confs_per_view: {view_name: (J,)} confidence scores.
        cameras_per_view: {view_name: (R, t, K)} camera matrices.
        joint_indices: Joint indices to triangulate.
        conf_threshold: Minimum confidence to include a view.
        reproj_threshold: RANSAC inlier threshold in pixels.
        min_inlier_views: Minimum inlier views for a valid triangulation.
        n_iterations: RANSAC iterations per joint.

    Returns:
        pts_3d: (len(joint_indices), 3) triangulated positions.
        quality: (len(joint_indices),) inlier fraction in [0, 1].
        reproj_errors: (len(joint_indices),) mean reprojection error.
    """
    view_names = list(keypoints_per_view.keys())
    J = len(joint_indices)

    pts_3d = np.zeros((J, 3), dtype=np.float64)
    quality = np.zeros(J, dtype=np.float64)
    reproj_errors = np.full(J, float("inf"), dtype=np.float64)

    for j_out, j in enumerate(joint_indices):
        # Gather views with sufficient confidence
        view_pts, view_Rs, view_ts, view_Ks, view_projs, view_ws = [], [], [], [], [], []

        for name in view_names:
            if name not in cameras_per_view:
                continue
            conf = float(confs_per_view[name][j])
            if conf < conf_threshold:
                continue
            R, t, K = cameras_per_view[name]
            view_pts.append(keypoints_per_view[name][j])
            view_Rs.append(R)
            view_ts.append(t)
            view_Ks.append(K)
            view_projs.append(build_projection_matrix(R, t, K))
            view_ws.append(conf)

        n_avail = len(view_pts)
        if n_avail < min_inlier_views:
            continue

        pts_2d = np.array(view_pts)
        weights = np.array(view_ws)

        pt, inlier_mask, err = ransac_triangulate_point(
            pts_2d,
            view_projs,
            view_Rs,
            view_ts,
            view_Ks,
            weights=weights,
            reproj_threshold=reproj_threshold,
            min_inlier_views=min_inlier_views,
            n_iterations=n_iterations,
        )

        n_inliers = int(inlier_mask.sum())
        if n_inliers >= min_inlier_views:
            pts_3d[j_out] = pt
            quality[j_out] = n_inliers / n_avail
            reproj_errors[j_out] = err

    return pts_3d, quality, reproj_errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _in_front_of_cameras(
    pt: np.ndarray,
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
) -> bool:
    """Return True if pt has positive depth in all given cameras."""
    for R, t in zip(Rs, ts):
        depth = (R @ pt + t)[2]
        if depth <= 0:
            return False
    return True


def _reprojection_errors(
    pt_3d: np.ndarray,
    pts_2d: np.ndarray,
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    Ks: list[np.ndarray],
) -> np.ndarray:
    """Compute per-view reprojection error in pixels."""
    errors = np.zeros(len(pts_2d))
    pt = pt_3d.reshape(1, 3)
    for i, (R, t, K) in enumerate(zip(Rs, ts, Ks)):
        proj = project_points(pt, R, t, K)[0]  # (2,)
        errors[i] = float(np.linalg.norm(proj - pts_2d[i]))
    return errors


def _compute_inliers(
    pt_3d: np.ndarray,
    pts_2d: np.ndarray,
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    Ks: list[np.ndarray],
    threshold: float,
) -> np.ndarray:
    """Boolean mask: views where reprojection error < threshold pixels."""
    errors = _reprojection_errors(pt_3d, pts_2d, Rs, ts, Ks)
    return errors < threshold
