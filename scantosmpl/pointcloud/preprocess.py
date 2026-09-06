"""Unit-free point cloud cleaning: outlier removal → decimation → normals.

Everything here runs **before** alignment, so the cloud's units are arbitrary
(master D8). No step may use a metric constant, and — the stronger requirement —
no step may depend on the cloud's *frame* either: preprocessing a similarity-
transformed cloud must give exactly the similarity-transformed preprocessed
cloud. Outlier removal and normal estimation are relative k-NN statistics, and
decimation selects by index, so all three satisfy that by construction.

Deterministic: no RNG anywhere in this path (master D12).
"""

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import open3d as o3d

from scantosmpl.pointcloud.io import PointCloud, from_open3d

logger = logging.getLogger(__name__)


class PreprocessConfigLike(Protocol):
    """The `Tier3Config` fields `preprocess_cloud` reads (master §5.2).

    Declared structurally so `scantosmpl.pointcloud` never imports `config.py`
    — `Tier3Config` satisfies this protocol by construction.

    Note `voxel_fraction_of_bbox` is absent: `Tier3Config` still declares it
    (master §5.2) but decimation no longer voxelises, so this module does not
    read it. See :func:`preprocess_cloud`.
    """

    outlier_nb_neighbors: int
    outlier_std_ratio: float
    target_points: int
    estimate_normals: bool
    normal_knn: int


@dataclass
class PreprocessStats:
    """What preprocessing did, in the cloud's own (source) units."""

    n_input: int
    n_after_outlier_removal: int
    n_output: int
    outlier_fraction: float
    #: Vestigial. Master §5.1 declares it, so the field stays for schema
    #: stability, but decimation no longer voxelises and this is always 0.0.
    voxel_size_source_units: float
    #: Diagnostic only — reported, never an input to a decision. The box is
    #: AXIS-ALIGNED, hence not rotation-invariant; nothing may branch on it.
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


def _uniform_index_selection(n: int, target: int) -> np.ndarray:
    """``target`` indices spread evenly over ``range(n)``, chosen by index alone.

    ``(arange(target) * n) // target`` is strictly increasing whenever
    ``n >= target`` — consecutive real values differ by ``n / target >= 1``, so
    their floors cannot collide — giving exactly ``target`` distinct, ordered
    indices. Preferred over ``points[::ceil(n / target)]``, which can only yield
    ``n``, ``n/2``, ``n/3``, … and so discards far more than asked for near a
    stride boundary (60160 points -> 30080 for a target of 50000).

    Args:
        n: Population size.
        target: Number of indices to select; must satisfy ``0 < target <= n``.

    Returns:
        (target,) int64 indices, strictly increasing.
    """
    return (np.arange(target, dtype=np.int64) * n) // target


def preprocess_cloud(
    cloud: PointCloud,
    cfg: PreprocessConfigLike,
) -> tuple[PointCloud, PreprocessStats]:
    """Clean a raw scan without assuming any particular scale.

    Order (master D8):
      1. Statistical outlier removal (k-NN distance statistics — scale free).
      2. Decimation to exactly ``cfg.target_points`` by uniform index selection
         (``target_points = 0``, or a cloud already at or under the target,
         skips it).
      3. Optional normal estimation via k-NN PCA (``KDTreeSearchParamKNN``).

    Step 2 used to voxelise, and that broke D8 twice over: the voxel size came
    from the AXIS-ALIGNED bbox diagonal, which is not rotation-invariant, and
    the grid itself is laid out in the cloud's current frame, so even a fixed
    voxel size lands differently once the cloud rotates. The count-targeting
    loop then stopped at a different size per frame (measured: 4429 vs 4469
    points for one cloud under a rigid rotation), and that count difference
    propagated all the way to a 1.96mm spread in the fitted displacement field.
    Selecting by index depends on nothing but the point count, so the whole
    function is now exactly similarity-equivariant.

    The tradeoff taken knowingly: index selection preserves the input's density
    distribution rather than equalising it, as a voxel grid would. Meshroom
    clouds are already roughly area-uniform and outlier removal runs first. If
    equalisation ever does matter, farthest-point sampling from a frame-
    independent seed is the equivariant way to get it.

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

    # --- 2. Frame-independent decimation ----------------------------------
    # Measured AFTER outlier removal so stray points cannot inflate the box.
    # Diagnostic only: nothing below branches on it (see PreprocessStats).
    diag = bbox_diagonal(np.asarray(pcd.points))

    if 0 < cfg.target_points < n_after_outlier:
        pcd = pcd.select_by_index(_uniform_index_selection(n_after_outlier, cfg.target_points))

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
        voxel_size_source_units=0.0,
        bbox_diagonal_source_units=float(diag),
        normals_estimated=normals_estimated,
    )
    logger.info(
        "Preprocess: %d -> %d (outliers) -> %d points | diag=%.6g (source units)",
        stats.n_input,
        stats.n_after_outlier_removal,
        stats.n_output,
        stats.bbox_diagonal_source_units,
    )
    return out, stats
