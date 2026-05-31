"""Radial undistortion for COLMAP SIMPLE_RADIAL cameras."""

import cv2
import numpy as np

from scantosmpl.calibration.colmap_reader import ColmapCamera


def build_pinhole_K(camera: ColmapCamera) -> np.ndarray:
    """Build 3×3 pinhole intrinsic matrix (no distortion) from a COLMAP camera.

    Args:
        camera: COLMAP camera with focal_length, cx, cy.

    Returns:
        (3, 3) intrinsic matrix K.
    """
    return np.array(
        [
            [camera.focal_length, 0.0, camera.cx],
            [0.0, camera.focal_length, camera.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def undistort_keypoints(
    keypoints_2d: np.ndarray,
    camera: ColmapCamera,
) -> np.ndarray:
    """Remove SIMPLE_RADIAL lens distortion from 2D keypoints.

    Uses cv2.undistortPoints for iterative inverse undistortion, which
    correctly inverts the distortion model rather than approximating it.

    Args:
        keypoints_2d: (N, 2) distorted pixel coordinates.
        camera: COLMAP camera with focal_length, cx, cy, k1.

    Returns:
        (N, 2) undistorted pixel coordinates in the same coordinate system.
    """
    if keypoints_2d.shape[0] == 0:
        return keypoints_2d.copy()

    K = build_pinhole_K(camera)
    # COLMAP SIMPLE_RADIAL: distortion vector [k1, 0, 0, 0] in OpenCV convention
    dist_coeffs = np.array([camera.k1, 0.0, 0.0, 0.0], dtype=np.float64)

    # cv2.undistortPoints expects (N, 1, 2) float32
    pts = keypoints_2d.astype(np.float32).reshape(-1, 1, 2)

    # P=K re-projects undistorted normalised coords back to pixel space
    undistorted = cv2.undistortPoints(pts, K, dist_coeffs, P=K)

    return undistorted.reshape(-1, 2).astype(np.float64)


def undistort_keypoints_batch(
    keypoints_per_view: dict[str, np.ndarray],
    camera: ColmapCamera,
) -> dict[str, np.ndarray]:
    """Undistort keypoints for all views sharing the same camera model.

    Args:
        keypoints_per_view: {view_name: (J, 2)} distorted keypoints.
        camera: Shared COLMAP camera (same for all views).

    Returns:
        {view_name: (J, 2)} undistorted keypoints.
    """
    return {
        name: undistort_keypoints(kps, camera)
        for name, kps in keypoints_per_view.items()
    }
