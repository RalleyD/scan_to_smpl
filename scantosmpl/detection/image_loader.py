"""Image loading with EXIF normalisation and intrinsic extraction.

Handles:
- EXIF orientation transpose (all 8 orientations)
- Manual orientation overrides for cameras with missing/wrong EXIF
- Intrinsic matrix K from EXIF focal length + FocalPlaneXResolution
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from scantosmpl.types import CameraParams

logger = logging.getLogger(__name__)

# Manual rotation overrides: filename stem -> rotation degrees (CW).
# Applied AFTER EXIF transpose. For cameras mounted upside-down with no EXIF hint.
DEFAULT_ORIENTATION_OVERRIDES: dict[str, int] = {
    "cam10_2": 180,
    "cam10_3": 180,
    "cam10_4": 180,
    "cam10_5": 180,
}


@dataclass
class LoadedImage:
    """Result of loading and normalising an image."""

    image: Image.Image
    path: Path
    exif: dict
    camera: CameraParams
    original_size: tuple[int, int]  # (width, height) before transpose


def _extract_exif(image: Image.Image) -> dict:
    """Extract EXIF tags into a plain dict (no lazy IFD objects)."""
    raw = image._getexif()
    if raw is None:
        return {}
    result = {}
    for tag_id, value in raw.items():
        if isinstance(value, bytes) and len(value) > 200:
            continue  # skip large binary blobs (MakerNote etc)
        result[tag_id] = value
    return result


def _focal_length_pixels(exif: dict, image_width: int) -> tuple[float, bool]:
    """Compute focal length in pixels from EXIF.

    Returns (focal_length_px, is_exact).
    is_exact=True when derived from FocalPlaneXResolution (sensor-specific).
    is_exact=False when using 36mm full-frame fallback.
    """
    # Tag IDs
    FOCAL_LENGTH = 37386
    FOCAL_PLANE_X_RES = 41486
    FOCAL_PLANE_RES_UNIT = 41488
    FOCAL_LENGTH_35MM = 41989

    f_mm = exif.get(FOCAL_LENGTH)
    if f_mm is None:
        return 0.0, False

    f_mm = float(f_mm)

    # Method 1: FocalPlaneXResolution (best — gives exact sensor pixel pitch)
    fpxr = exif.get(FOCAL_PLANE_X_RES)
    if fpxr is not None:
        fpxr = float(fpxr)
        # Resolution unit: 2=inches, 3=centimeters
        unit = exif.get(FOCAL_PLANE_RES_UNIT, 2)
        if unit == 3:
            sensor_width_mm = image_width / fpxr * 10.0
        else:
            sensor_width_mm = image_width / fpxr * 25.4

        f_px = f_mm * image_width / sensor_width_mm
        logger.debug(
            "Intrinsics from FocalPlaneXRes: f=%.1fmm, sensor=%.2fmm -> f=%.1fpx",
            f_mm, sensor_width_mm, f_px,
        )
        return f_px, True

    # Method 2: FocalLengthIn35mmFilm
    f_35mm = exif.get(FOCAL_LENGTH_35MM)
    if f_35mm is not None and float(f_35mm) > 0:
        crop_factor = float(f_35mm) / f_mm
        sensor_width_mm = 36.0 / crop_factor
        f_px = f_mm * image_width / sensor_width_mm
        logger.debug(
            "Intrinsics from 35mm equiv: f=%.1fmm, f35=%.1fmm -> f=%.1fpx",
            f_mm, float(f_35mm), f_px,
        )
        return f_px, True

    # Method 3: Assume full-frame 36mm (least accurate)
    f_px = f_mm * image_width / 36.0
    logger.warning(
        "No sensor size in EXIF, assuming 36mm full-frame: f=%.1fmm -> f=%.1fpx",
        f_mm, f_px,
    )
    return f_px, False


def load_image(
    image_path: Path,
    orientation_overrides: dict[str, int] | None = None,
) -> LoadedImage:
    """Load an image, apply EXIF transpose + manual overrides, extract intrinsics.

    Args:
        image_path: Path to JPEG image.
        orientation_overrides: Map of filename stem -> rotation degrees (CW).
            Applied after EXIF transpose for cameras with wrong/missing EXIF orientation.

    Returns:
        LoadedImage with normalised image and camera parameters.
    """
    overrides = orientation_overrides if orientation_overrides is not None else DEFAULT_ORIENTATION_OVERRIDES

    img = Image.open(image_path)
    original_size = img.size  # (width, height) before any transforms
    exif = _extract_exif(img)

    # Step 1: EXIF orientation transpose (handles rotation, mirroring)
    img = ImageOps.exif_transpose(img)

    # Step 2: Manual orientation override
    stem = image_path.stem
    if stem in overrides:
        angle = overrides[stem]
        # PIL rotate is CCW, so negate for CW convention
        img = img.rotate(-angle, expand=True)
        logger.info("Applied %d° CW rotation override to %s", angle, image_path.name)

    # Step 3: Extract intrinsics
    # Use original (pre-transpose) width for focal length calculation,
    # since FocalPlaneXResolution refers to the sensor's native orientation
    sensor_width_px = original_size[0]
    f_px, is_exact = _focal_length_pixels(exif, sensor_width_px)

    # Principal point at image center (post-transform)
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0

    camera = CameraParams(
        focal_length=f_px,
        principal_point=(cx, cy),
    )

    if f_px <= 0:
        logger.warning("Could not extract focal length from %s", image_path.name)

    return LoadedImage(
        image=img,
        path=image_path,
        exif=exif,
        camera=camera,
        original_size=original_size,
    )


def load_directory(
    image_dir: Path,
    extensions: set[str] = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"},
    orientation_overrides: dict[str, int] | None = None,
) -> list[LoadedImage]:
    """Load all images from a directory.

    Returns list sorted by filename.
    """
    image_dir = Path(image_dir)
    paths = sorted(p for p in image_dir.iterdir() if p.suffix in extensions)
    if not paths:
        logger.warning("No images found in %s", image_dir)
        return []

    results = []
    for p in paths:
        try:
            results.append(load_image(p, orientation_overrides))
        except Exception:
            logger.exception("Failed to load %s", p)
    return results
