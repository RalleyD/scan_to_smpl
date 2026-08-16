"""Unit-free point cloud cleaning: outlier removal → voxel downsample → normals.

Everything here runs **before** alignment, so the cloud's units are arbitrary
(master D8). No step may use a metric constant: the voxel size is a fraction of
the cloud's own bounding-box diagonal, and the outlier / normal steps are
relative statistics, so the same config works on a cloud scaled by 1e-3 or 1e3.

Deterministic: no RNG anywhere in this path (master D12).
"""

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import open3d as o3d

from scantosmpl.pointcloud.io import PointCloud, from_open3d

logger = logging.getLogger(__name__)

# Safety net for the voxel-size search: each pass grows the voxel, so this only
# bounds pathological inputs (e.g. a target of 1 point).
_MAX_VOXEL_PASSES = 20
_MIN_VOXEL_GROWTH = 1.1


class PreprocessConfigLike(Protocol):
    """The `Tier3Config` fields `preprocess_cloud` reads (master §5.2).

    Declared structurally so `scantosmpl.pointcloud` never imports `config.py`
    — `Tier3Config` satisfies this protocol by construction.
    """

    outlier_nb_neighbors: int
    outlier_std_ratio: float
    target_points: int
    voxel_fraction_of_bbox: float
    estimate_normals: bool
    normal_knn: int


@dataclass
class PreprocessStats:
    """What preprocessing did, in the cloud's own (source) units."""

    n_input: int
    n_after_outlier_removal: int
    n_output: int
    outlier_fraction: float
    voxel_size_source_units: float
    bbox_diagonal_source_units: float
    normals_estimated: bool


def bbox_diagonal(points: np.ndarray) -> float:
    """Axis-aligned bounding-box diagonal length, in the points' own units.

    Args:
        points: (N, 3) positions.

    Returns:
        Diagonal length, 0.0 for an empty cloud.
    """
    if points.shape[0] == 0:
        return 0.0
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extent))


def preprocess_cloud(
    cloud: PointCloud,
    cfg: PreprocessConfigLike,
) -> tuple[PointCloud, PreprocessStats]:
    """Clean a raw scan without assuming any particular scale.

    Order (master D8):
      1. Statistical outlier removal (k-NN distance statistics — scale free).
      2. Voxel downsample with ``voxel = voxel_fraction_of_bbox * bbox_diagonal``
         measured in SOURCE units, growing the fraction until the result is at
         most ``cfg.target_points`` (``target_points = 0`` skips downsampling).
      3. Optional normal estimation via k-NN PCA (``KDTreeSearchParamKNN``).

    Args:
        cloud: Input cloud, normally straight from :func:`load_pointcloud`.
        cfg: Tier 3 config (see :class:`PreprocessConfigLike`).

    Returns:
        (cleaned cloud, stats). ``frame`` and ``units`` pass through unchanged —
        preprocessing never promotes a source-frame cloud to metres.
    """
    n_input = cloud.n_points
    pcd = cloud.to_open3d()

    # --- 1. Statistical outlier removal -----------------------------------
    if n_input > cfg.outlier_nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_nb_neighbors,
            std_ratio=cfg.outlier_std_ratio,
        )
    else:
        logger.warning(
            "Skipping outlier removal: %d points <= nb_neighbors=%d",
            n_input,
            cfg.outlier_nb_neighbors,
        )
    n_after_outlier = len(pcd.points)

    # --- 2. Unit-free voxel downsample ------------------------------------
    # Diagonal is measured AFTER outlier removal: stray points would otherwise
    # inflate the bbox and blow the voxel size up with it.
    diag = bbox_diagonal(np.asarray(pcd.points))
    voxel_size = 0.0

    if cfg.target_points > 0 and n_after_outlier > cfg.target_points and diag > 0.0:
        voxel_size = cfg.voxel_fraction_of_bbox * diag
        down = pcd.voxel_down_sample(voxel_size=voxel_size)
        for _ in range(_MAX_VOXEL_PASSES):
            n_down = len(down.points)
            if n_down <= cfg.target_points:
                break
            growth = max(_MIN_VOXEL_GROWTH, (n_down / cfg.target_points) ** (1.0 / 3.0))
            voxel_size *= growth
            down = pcd.voxel_down_sample(voxel_size=voxel_size)
        else:
            logger.warning(
                "Voxel search hit %d passes; %d points remain (target %d)",
                _MAX_VOXEL_PASSES,
                len(down.points),
                cfg.target_points,
            )
        pcd = down

    # --- 3. Normal estimation ---------------------------------------------
    normals_estimated = False
    if cfg.estimate_normals and len(pcd.points) > 0:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=cfg.normal_knn))
        # Deliberately NOT orientation-consistent: photogrammetry normals have
        # unreliable sign anyway, and the normal loss uses |cos| (master §5.3).
        normals_estimated = True

    out = from_open3d(pcd, cloud.source_path, frame=cloud.frame, units=cloud.units)
    stats = PreprocessStats(
        n_input=n_input,
        n_after_outlier_removal=n_after_outlier,
        n_output=out.n_points,
        outlier_fraction=(float(n_input - n_after_outlier) / n_input if n_input > 0 else 0.0),
        voxel_size_source_units=float(voxel_size),
        bbox_diagonal_source_units=float(diag),
        normals_estimated=normals_estimated,
    )
    logger.info(
        "Preprocess: %d -> %d (outliers) -> %d points | voxel=%.6g diag=%.6g (source units)",
        stats.n_input,
        stats.n_after_outlier_removal,
        stats.n_output,
        stats.voxel_size_source_units,
        stats.bbox_diagonal_source_units,
    )
    return out, stats
