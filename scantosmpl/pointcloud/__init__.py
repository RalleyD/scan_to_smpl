"""Point cloud input path: load -> preprocess -> align to SMPL -> segment.

A Meshroom (or any photogrammetry) point cloud arrives in an arbitrary source
frame with arbitrary units. This package turns it into a metric, SMPL-frame
:class:`~scantosmpl.pointcloud.io.PointCloud` the Tier 3 surface fitter can
consume, plus the per-vertex semantic part weights AC 7.3 needs:

    load_pointcloud -> preprocess_cloud -> align_cloud_to_smpl -> vertex_part_weights

See each module's docstring for the frame/units contract at every stage.
"""

from scantosmpl.pointcloud.align import (
    CloudAlignment,
    align_cloud_to_smpl,
    enumerate_proper_rotations,
    pca_triad,
)
from scantosmpl.pointcloud.io import PointCloud, load_pointcloud, save_pointcloud
from scantosmpl.pointcloud.preprocess import PreprocessStats, preprocess_cloud
from scantosmpl.pointcloud.segment import (
    SMPL_PART_GROUPS,
    smpl_part_labels,
    transfer_labels_to_cloud,
    vertex_part_weights,
)

__all__ = [
    "PointCloud",
    "load_pointcloud",
    "save_pointcloud",
    "PreprocessStats",
    "preprocess_cloud",
    "CloudAlignment",
    "pca_triad",
    "enumerate_proper_rotations",
    "align_cloud_to_smpl",
    "SMPL_PART_GROUPS",
    "smpl_part_labels",
    "vertex_part_weights",
    "transfer_labels_to_cloud",
]
