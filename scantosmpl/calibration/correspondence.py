"""3D-2D correspondence builder for PnP calibration."""

import numpy as np

from scantosmpl.smpl.joint_map import (
    COCO_MIDPOINT_TO_SMPL,
    COCO_TO_SMPL,
    DENSE_KP_VERTEX_INDICES,
)
from scantosmpl.types import ViewResult


class CorrespondenceBuilder:
    """Builds 3D-2D correspondences from the consensus SMPL mesh and per-view 2D keypoints.

    Dense correspondences (138 points) use CameraHMR dense keypoints mapped to
    SMPL vertices. Sparse correspondences (up to 14 points) use ViTPose COCO
    keypoints mapped to SMPL joints.
    """

    def __init__(
        self,
        consensus_vertices: np.ndarray,
        consensus_joints: np.ndarray,
    ) -> None:
        """
        Args:
            consensus_vertices: (6890, 3) SMPL vertices in canonical frame.
            consensus_joints: (24, 3) SMPL joints in canonical frame.
        """
        self.consensus_vertices = consensus_vertices
        self.consensus_joints = consensus_joints

        # Pre-compute dense 3D reference points
        self.dense_3d = consensus_vertices[DENSE_KP_VERTEX_INDICES]  # (138, 3)

    def build_dense_correspondences(
        self,
        view: ViewResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build dense 3D-2D correspondences from CameraHMR dense keypoints.

        Args:
            view: ViewResult with dense_keypoints_2d and dense_keypoint_confs.

        Returns:
            pts_3d: (138, 3) 3D points on consensus mesh.
            pts_2d: (138, 2) 2D projections in image space.
            confs: (138,) confidence scores.

        Raises:
            ValueError: If view lacks dense keypoint data.
        """
        if view.dense_keypoints_2d is None or view.dense_keypoint_confs is None:
            raise ValueError(
                f"{view.image_path.name}: no dense keypoints available"
            )

        pts_3d = self.dense_3d.copy()
        pts_2d = view.dense_keypoints_2d.astype(np.float64)
        confs = view.dense_keypoint_confs.astype(np.float64)

        return pts_3d, pts_2d, confs

    def build_sparse_correspondences(
        self,
        view: ViewResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sparse 3D-2D correspondences from ViTPose COCO keypoints.

        Uses 12 direct COCO-to-SMPL joint mappings plus 2 midpoint-derived
        joints (pelvis, neck) for up to 14 correspondences.

        Args:
            view: ViewResult with keypoints_2d and keypoint_confs.

        Returns:
            pts_3d: (N, 3) 3D SMPL joint positions.
            pts_2d: (N, 2) 2D ViTPose keypoint positions.
            confs: (N,) confidence scores.

        Raises:
            ValueError: If view lacks COCO keypoint data.
        """
        if view.keypoints_2d is None or view.keypoint_confs is None:
            raise ValueError(
                f"{view.image_path.name}: no COCO keypoints available"
            )

        pts_3d_list = []
        pts_2d_list = []
        confs_list = []

        kps_2d = view.keypoints_2d.astype(np.float64)   # (17, 2)
        kp_confs = view.keypoint_confs.astype(np.float64)  # (17,)

        # Direct COCO -> SMPL correspondences
        for coco_idx, smpl_idx in COCO_TO_SMPL.items():
            pts_3d_list.append(self.consensus_joints[smpl_idx])
            pts_2d_list.append(kps_2d[coco_idx])
            confs_list.append(kp_confs[coco_idx])

        # Midpoint-derived correspondences (pelvis, neck)
        for (coco_a, coco_b), smpl_idx in COCO_MIDPOINT_TO_SMPL.items():
            mid_2d = (kps_2d[coco_a] + kps_2d[coco_b]) / 2.0
            mid_conf = min(float(kp_confs[coco_a]), float(kp_confs[coco_b]))
            pts_3d_list.append(self.consensus_joints[smpl_idx])
            pts_2d_list.append(mid_2d)
            confs_list.append(mid_conf)

        pts_3d = np.array(pts_3d_list, dtype=np.float64)
        pts_2d = np.array(pts_2d_list, dtype=np.float64)
        confs = np.array(confs_list, dtype=np.float64)

        return pts_3d, pts_2d, confs
