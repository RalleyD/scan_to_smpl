"""Intrinsic matrix construction for PnP calibration."""

import numpy as np
from PIL import Image

from scantosmpl.types import ViewResult


def build_intrinsic_matrix(
    focal_length_px: float,
    image_width: int,
    image_height: int,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Build a 3x3 intrinsic matrix.

    Args:
        focal_length_px: Focal length in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        principal_point: (cx, cy) principal point. If None, defaults to image center.

    Returns:
        (3, 3) intrinsic matrix K.
    """
    if principal_point is None:
        cx = image_width / 2.0
        cy = image_height / 2.0
    else:
        cx, cy = principal_point

    return np.array([
        [focal_length_px, 0.0, cx],
        [0.0, focal_length_px, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def get_intrinsics_for_view(
    view: ViewResult,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Build K for a view from its CameraParams.

    If the view's principal_point is (0, 0) (the default placeholder),
    it is replaced with the image center.

    Args:
        view: ViewResult with camera parameters.
        image_size: (width, height) of the image.

    Returns:
        (3, 3) intrinsic matrix K.
    """
    cam = view.camera
    w, h = image_size

    pp = cam.principal_point
    if pp == (0.0, 0.0):
        pp = None  # let build_intrinsic_matrix default to image center

    return build_intrinsic_matrix(cam.focal_length, w, h, principal_point=pp)
