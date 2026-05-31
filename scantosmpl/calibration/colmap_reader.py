"""Parse COLMAP binary model files (cameras.bin + images.bin)."""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ColmapCamera:
    """Camera intrinsics from COLMAP cameras.bin."""

    camera_id: int
    model: str          # e.g. "SIMPLE_RADIAL"
    width: int
    height: int
    focal_length: float
    cx: float
    cy: float
    k1: float           # radial distortion (SIMPLE_RADIAL only uses k1)


@dataclass
class ColmapImage:
    """Per-image extrinsics + camera assignment from COLMAP images.bin."""

    image_id: int
    name: str
    camera_id: int
    rotation: np.ndarray    # (3, 3) world-to-camera rotation
    translation: np.ndarray # (3,) world-to-camera translation


# COLMAP model IDs → (name, num_params)
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),   # f, cx, cy
    1: ("PINHOLE", 4),          # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),    # f, cx, cy, k1
    3: ("RADIAL", 5),           # f, cx, cy, k1, k2
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _parse_cameras(cameras_bin: Path) -> dict[int, ColmapCamera]:
    """Parse cameras.bin → {camera_id: ColmapCamera}."""
    cameras: dict[int, ColmapCamera] = {}

    with open(cameras_bin, "rb") as f:
        n_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            model_name, n_params = _CAMERA_MODELS.get(model_id, (f"UNKNOWN_{model_id}", 0))
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))

            if model_name in ("SIMPLE_PINHOLE",):
                f_val, cx, cy = params[0], params[1], params[2]
                k1 = 0.0
            elif model_name == "PINHOLE":
                # fx == fy assumed; use fx
                f_val, cx, cy = params[0], params[2], params[3]
                k1 = 0.0
            elif model_name in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
                f_val, cx, cy, k1 = params[0], params[1], params[2], params[3]
            elif model_name in ("RADIAL", "RADIAL_FISHEYE"):
                f_val, cx, cy, k1 = params[0], params[1], params[2], params[3]
            else:
                # Best-effort: assume f, cx, cy, k1 layout
                f_val = params[0] if len(params) > 0 else 1.0
                cx = params[1] if len(params) > 1 else width / 2.0
                cy = params[2] if len(params) > 2 else height / 2.0
                k1 = params[3] if len(params) > 3 else 0.0

            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model_name,
                width=int(width),
                height=int(height),
                focal_length=float(f_val),
                cx=float(cx),
                cy=float(cy),
                k1=float(k1),
            )

    return cameras


def _parse_images(images_bin: Path) -> dict[str, ColmapImage]:
    """Parse images.bin → {image_name: ColmapImage}."""
    images: dict[str, ColmapImage] = {}

    with open(images_bin, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read null-terminated image name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")

            # Skip 2D point observations
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)  # each is (x: double, y: double, point3D_id: int64)

            # COLMAP quaternion convention: qw, qx, qy, qz (scalar-first)
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            images[name] = ColmapImage(
                image_id=image_id,
                name=name,
                camera_id=camera_id,
                rotation=R.astype(np.float64),
                translation=np.array([tx, ty, tz], dtype=np.float64),
            )

    return images


def read_colmap_model(
    model_dir: Path,
) -> tuple[dict[int, ColmapCamera], dict[str, ColmapImage]]:
    """Parse a COLMAP sparse model directory.

    Args:
        model_dir: Path to directory containing cameras.bin and images.bin.

    Returns:
        cameras: {camera_id: ColmapCamera}
        images:  {image_name: ColmapImage}

    Raises:
        FileNotFoundError: If cameras.bin or images.bin are missing.
    """
    model_dir = Path(model_dir)
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"

    if not cameras_bin.exists():
        raise FileNotFoundError(f"cameras.bin not found in {model_dir}")
    if not images_bin.exists():
        raise FileNotFoundError(f"images.bin not found in {model_dir}")

    cameras = _parse_cameras(cameras_bin)
    images = _parse_images(images_bin)

    return cameras, images


def match_views_to_colmap(
    view_names: list[str],
    colmap_images: dict[str, ColmapImage],
) -> tuple[dict[str, ColmapImage], list[str]]:
    """Match Phase 1 view names against COLMAP image names.

    Args:
        view_names: Filenames from Phase 1 (e.g. ["cam01_2.JPG", ...]).
        colmap_images: All images from COLMAP.

    Returns:
        matched: {view_name: ColmapImage} for found views.
        missing: view_names not found in COLMAP.
    """
    matched: dict[str, ColmapImage] = {}
    missing: list[str] = []
    for name in view_names:
        if name in colmap_images:
            matched[name] = colmap_images[name]
        else:
            missing.append(name)
    return matched, missing
