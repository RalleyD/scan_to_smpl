"""Point cloud I/O — load a PLY/OBJ scan, save a debug PLY.

Frames and units (see repo spec "Coordinate frames + units"):
a freshly loaded cloud is always in its **source** frame with **arbitrary** units
(a Meshroom reconstruction has no metric scale, no canonical orientation and no
meaningful origin). Only :func:`scantosmpl.pointcloud.align.align_cloud_to_smpl`
is allowed to promote a cloud to ``frame="smpl_world"`` / ``units="metres"``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: tuple[str, ...] = (".ply", ".obj")


@dataclass
class PointCloud:
    """A point cloud plus the guard fields that say which frame it lives in.

    Attributes:
        points: (N, 3) float64 positions, in the frame named by ``frame``.
        normals: (N, 3) float64 unit-length normals in the same frame, or None.
        colors: (N, 3) float32 RGB in [0, 1], or None.
        source_path: Path the cloud was loaded from (kept for provenance).
        frame: "source" (arbitrary reconstruction frame) or "smpl_world".
        units: "arbitrary" (source units) or "metres".
    """

    points: np.ndarray
    normals: np.ndarray | None
    colors: np.ndarray | None
    source_path: Path
    frame: Literal["source", "smpl_world"] = "source"
    units: Literal["arbitrary", "metres"] = "arbitrary"

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {self.points.shape}")
        n = self.points.shape[0]
        if self.normals is not None and self.normals.shape != (n, 3):
            raise ValueError(f"normals must be ({n}, 3), got {self.normals.shape}")
        if self.colors is not None and self.colors.shape != (n, 3):
            raise ValueError(f"colors must be ({n}, 3), got {self.colors.shape}")

    @property
    def n_points(self) -> int:
        """Number of points in the cloud."""
        return int(self.points.shape[0])

    def to_open3d(self) -> o3d.geometry.PointCloud:
        """Convert to an Open3D point cloud (positions + normals + colors)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.points, dtype=np.float64))
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(self.normals, dtype=np.float64))
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(self.colors, dtype=np.float64))
        return pcd


def from_open3d(
    pcd: o3d.geometry.PointCloud,
    source_path: Path,
    frame: Literal["source", "smpl_world"] = "source",
    units: Literal["arbitrary", "metres"] = "arbitrary",
) -> PointCloud:
    """Wrap an Open3D point cloud as a :class:`PointCloud` with explicit frame/units."""
    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    return PointCloud(
        points=points,
        normals=normals,
        colors=colors,
        source_path=source_path,
        frame=frame,
        units=units,
    )


def _read_mesh_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read a triangle mesh and return its vertices (+ vertex normals/colors)."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    has_n, has_c = mesh.has_vertex_normals(), mesh.has_vertex_colors()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64) if has_n else None
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32) if has_c else None
    return points, normals, colors


def _stride_subsample(
    points: np.ndarray,
    normals: np.ndarray | None,
    colors: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Deterministic stride subsample to at most ``max_points`` points (no RNG, D12)."""
    n = points.shape[0]
    if max_points <= 0 or n <= max_points:
        return points, normals, colors
    step = int(np.ceil(n / max_points))
    return (
        points[::step],
        None if normals is None else normals[::step],
        None if colors is None else colors[::step],
    )


def load_pointcloud(path: Path, *, max_points: int | None = None) -> PointCloud:
    """Load a PLY or OBJ point cloud.

    An OBJ (or a PLY that only parses as a mesh) contributes its **vertices**;
    faces are discarded — Tier 3 treats the scan as an unstructured point set.

    Args:
        path: Path to a ``.ply`` or ``.obj`` file.
        max_points: Optional cap. Applies a deterministic stride subsample
            (``points[::step]``), never a random one (D12).

    Returns:
        A :class:`PointCloud` in the SOURCE frame with arbitrary units
        (``frame="source"``, ``units="arbitrary"``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On an unsupported suffix, or if the file contains no points.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported point cloud suffix {path.suffix!r}; expected one of {SUPPORTED_SUFFIXES}"
        )

    points: np.ndarray
    normals: np.ndarray | None
    colors: np.ndarray | None

    if suffix == ".obj":
        # Open3D's read_point_cloud does not handle OBJ — read it as a mesh.
        points, normals, colors = _read_mesh_vertices(path)
    else:
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=np.float64)
        if points.shape[0] == 0:
            # A PLY holding a mesh: read_point_cloud can come back empty.
            points, normals, colors = _read_mesh_vertices(path)
        else:
            normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
            colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    if points.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {path}")

    n_loaded = points.shape[0]
    if max_points is not None:
        points, normals, colors = _stride_subsample(points, normals, colors, max_points)

    logger.info(
        "Loaded %s: %d points (%d after subsample), normals=%s, colors=%s",
        path.name,
        n_loaded,
        points.shape[0],
        normals is not None,
        colors is not None,
    )

    return PointCloud(
        points=np.ascontiguousarray(points, dtype=np.float64),
        normals=None if normals is None else np.ascontiguousarray(normals, dtype=np.float64),
        colors=None if colors is None else np.ascontiguousarray(colors, dtype=np.float32),
        source_path=path,
        frame="source",
        units="arbitrary",
    )


def save_pointcloud(cloud: PointCloud, path: Path) -> None:
    """Write a cloud as a binary PLY. Debug artefacts only.

    Args:
        cloud: Cloud to write (frame/units are NOT recorded in the PLY — the
            filename is the only provenance, so name debug dumps accordingly).
        path: Destination ``.ply`` path; parent directories are created.

    Raises:
        ValueError: If ``path`` does not have a ``.ply`` suffix.
        RuntimeError: If Open3D fails to write the file.
    """
    path = Path(path)
    if path.suffix.lower() != ".ply":
        raise ValueError(f"save_pointcloud writes PLY only, got {path.suffix!r}")
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = o3d.io.write_point_cloud(str(path), cloud.to_open3d(), write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud to {path}")
    logger.info("Wrote %d points to %s", cloud.n_points, path)
