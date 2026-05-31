"""Weighted Direct Linear Transform (DLT) triangulation."""

import numpy as np


def build_projection_matrix(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Build 3×4 projection matrix P = K @ [R | t].

    Args:
        R: (3, 3) world-to-camera rotation.
        t: (3,) world-to-camera translation.
        K: (3, 3) intrinsic matrix.

    Returns:
        (3, 4) projection matrix.
    """
    Rt = np.hstack([R, t.reshape(3, 1)])  # (3, 4)
    return K @ Rt


def triangulate_point(
    pts_2d: np.ndarray,
    projections: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Triangulate a single 3D point from V ≥ 2 calibrated views via weighted DLT.

    Each view contributes two rows to the linear system A x = 0, weighted by
    the ViTPose confidence score so higher-confidence observations have more
    influence on the solution.

    The DLT formulation for one view with projection matrix P (3×4) and
    observed 2D point (u, v):
        u * P[2,:] - P[0,:] = 0
        v * P[2,:] - P[1,:] = 0

    Args:
        pts_2d: (V, 2) undistorted 2D observations.
        projections: List of V (3, 4) projection matrices P = K @ [R|t].
        weights: (V,) confidence weights. None = uniform.

    Returns:
        (3,) triangulated 3D world point.

    Raises:
        ValueError: If fewer than 2 views are provided.
    """
    V = len(projections)
    if V < 2:
        raise ValueError(f"Need ≥2 views for triangulation, got {V}")
    if len(pts_2d) != V:
        raise ValueError(f"pts_2d ({len(pts_2d)}) must match projections ({V})")

    if weights is None:
        weights = np.ones(V, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # Build 2V × 4 system A x = 0
    rows = []
    for i, (P, (u, v), w) in enumerate(
        zip(projections, pts_2d, weights)
    ):
        row_u = w * (u * P[2, :] - P[0, :])
        row_v = w * (v * P[2, :] - P[1, :])
        rows.append(row_u)
        rows.append(row_v)

    A = np.stack(rows, axis=0)  # (2V, 4)

    # Solve via SVD: x = right singular vector of smallest singular value
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    X_hom = Vt[-1]  # (4,) homogeneous 3D point

    if abs(X_hom[3]) < 1e-12:
        # Point at infinity — return a large but finite value
        return X_hom[:3] / (np.sign(X_hom[3] + 1e-30) * 1e-12)

    return (X_hom[:3] / X_hom[3]).astype(np.float64)


def triangulate_joints(
    keypoints_per_view: dict[str, np.ndarray],
    confs_per_view: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    joint_indices: list[int],
    min_views: int = 3,
    conf_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a set of joints independently across all views.

    For each joint j in joint_indices:
    1. Collect views where confidence > conf_threshold.
    2. If ≥ min_views available, run weighted DLT.
    3. Otherwise mark as low-quality (quality = 0).

    The returned arrays are indexed by the position in joint_indices, not
    by the joint index itself.

    Args:
        keypoints_per_view: {view_name: (J_max, 2)} undistorted 2D keypoints.
            J_max must be large enough to index via joint_indices.
        confs_per_view: {view_name: (J_max,)} confidence scores.
        projections: {view_name: (3, 4)} projection matrices.
        joint_indices: Indices into the J dimension to triangulate.
        min_views: Minimum views required; otherwise quality set to 0.
        conf_threshold: Confidence threshold for including a view.

    Returns:
        pts_3d: (len(joint_indices), 3) triangulated 3D positions.
            Joints with insufficient views are set to (0, 0, 0).
        quality: (len(joint_indices),) scores in [0, 1].
            Defined as (n_views_used / n_views_available) if ≥ min_views, else 0.
    """
    view_names = list(keypoints_per_view.keys())
    J = len(joint_indices)

    pts_3d = np.zeros((J, 3), dtype=np.float64)
    quality = np.zeros(J, dtype=np.float64)

    for j_out, j in enumerate(joint_indices):
        # Gather views with sufficient confidence for this joint
        view_pts: list[np.ndarray] = []
        view_projs: list[np.ndarray] = []
        view_weights: list[float] = []

        for name in view_names:
            if name not in projections:
                continue
            conf = float(confs_per_view[name][j])
            if conf < conf_threshold:
                continue
            view_pts.append(keypoints_per_view[name][j])
            view_projs.append(projections[name])
            view_weights.append(conf)

        n_avail = len(view_pts)
        if n_avail < min_views:
            continue  # leave as zeros / quality=0

        pts_3d[j_out] = triangulate_point(
            np.array(view_pts),
            view_projs,
            np.array(view_weights),
        )
        quality[j_out] = n_avail / max(len(view_names), 1)

    return pts_3d, quality
