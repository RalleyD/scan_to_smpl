"""Unit tests for Phase 5 loss functions (no GPU required)."""

import numpy as np
import pytest
import torch

from scantosmpl.fitting.losses import (
    joint_loss,
    pose_prior_loss,
    reprojection_loss,
    shape_regularisation,
)


COCO_TO_SMPL = {
    5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
    11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8,
}


# ---------------------------------------------------------------------------
# joint_loss
# ---------------------------------------------------------------------------


class TestJointLoss:
    def test_zero_at_target(self):
        """Loss should be zero when prediction matches target."""
        J = 14
        target = torch.randn(J, 3)
        pred = target.unsqueeze(0)  # (1, J, 3)
        loss = joint_loss(pred, target)
        assert float(loss) < 1e-6, f"Expected ~0 loss, got {float(loss)}"

    def test_positive_on_error(self):
        """Loss > 0 when prediction differs from target."""
        target = torch.zeros(10, 3)
        pred = torch.ones(1, 10, 3)
        loss = joint_loss(pred, target)
        assert float(loss) > 0

    def test_huber_linear_for_outlier(self):
        """Huber should be sub-quadratic for large errors (outlier regime)."""
        target = torch.zeros(1, 3)
        pred_small = torch.tensor([[[0.01, 0.0, 0.0]]])    # inside delta
        pred_large = torch.tensor([[[10.0, 0.0, 0.0]]])    # outside delta (huber_delta=0.05)

        l_small = float(joint_loss(pred_small, target))
        l_large = float(joint_loss(pred_large, target))

        # If purely quadratic: ratio = (10/0.01)^2 = 1e6
        # With Huber: large error grows linearly → ratio much smaller
        ratio = l_large / (l_small + 1e-12)
        assert ratio < 1e5, f"Huber should suppress outlier: ratio={ratio:.1f}"

    def test_confidence_weighting(self):
        """Higher-weight joints should dominate the loss."""
        target = torch.zeros(2, 3)
        pred = torch.ones(1, 2, 3)  # error = 1 on both joints
        w_equal = torch.tensor([1.0, 1.0])
        w_skew = torch.tensor([10.0, 0.0])

        loss_equal = float(joint_loss(pred, target, joint_weights=w_equal))
        loss_skew = float(joint_loss(pred, target, joint_weights=w_skew))
        # With w=[10, 0] joint 0 dominates; result should differ from uniform
        assert abs(loss_skew - loss_equal) > 0.01


# ---------------------------------------------------------------------------
# reprojection_loss
# ---------------------------------------------------------------------------


def _make_camera_tensors(
    pos: np.ndarray,
    focal: float = 1000.0,
    w: int = 1000,
    h: int = 1000,
):
    """Synthetic pinhole camera as tensors."""
    forward = -pos / np.linalg.norm(pos)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)
    t = -R @ pos
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    return (
        torch.tensor(R, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(K, dtype=torch.float32),
    )


class TestReprojectionLoss:
    def test_zero_on_perfect_projection(self):
        """Loss should be near-zero when predicted joints project exactly to keypoints."""
        # 3D joints
        joints_3d = torch.zeros(1, 24, 3)
        joints_3d[0, 16] = torch.tensor([0.3, 0.4, 2.5])  # left_shoulder (smpl 16)

        R, t, K = _make_camera_tensors(np.array([3.0, 0.0, 0.0]))

        # Perfect 2D observation
        pt_cam = R @ joints_3d[0, 16] + t
        p2d = K @ pt_cam
        p2d = p2d[:2] / p2d[2]
        obs = p2d.unsqueeze(0)  # (1, 2)

        # Build keypoints at coco index 5 (→ smpl 16)
        kp2d = {"view": torch.zeros(17, 2)}
        kp2d["view"][5] = obs[0]
        confs = {"view": torch.zeros(17)}
        confs["view"][5] = 1.0
        cameras = {"view": (R, t, K)}

        loss = reprojection_loss(joints_3d, kp2d, confs, cameras, COCO_TO_SMPL)
        assert float(loss) < 0.5, f"Near-zero expected, got {float(loss)}"

    def test_positive_on_error(self):
        """Loss > 0 when projection is far from observation."""
        joints_3d = torch.zeros(1, 24, 3)
        joints_3d[0, 16] = torch.tensor([0.3, 0.4, 2.5])

        R, t, K = _make_camera_tensors(np.array([3.0, 0.0, 0.0]))
        kp2d = {"view": torch.zeros(17, 2)}
        kp2d["view"][5] = torch.tensor([999.0, 999.0])  # far from projection
        confs = {"view": torch.zeros(17)}
        confs["view"][5] = 1.0
        cameras = {"view": (R, t, K)}

        loss = reprojection_loss(joints_3d, kp2d, confs, cameras, COCO_TO_SMPL)
        assert float(loss) > 1.0

    def test_low_confidence_downweighted(self):
        """Low-confidence observations contribute less to loss."""
        joints_3d = torch.zeros(1, 24, 3)
        joints_3d[0, 16] = torch.tensor([0.3, 0.0, 2.0])

        R, t, K = _make_camera_tensors(np.array([3.0, 0.0, 0.0]))
        kp2d = {"view": torch.zeros(17, 2)}
        kp2d["view"][5] = torch.tensor([999.0, 999.0])
        cameras = {"view": (R, t, K)}

        confs_high = {"view": torch.zeros(17)}
        confs_high["view"][5] = 0.9
        confs_low = {"view": torch.zeros(17)}
        confs_low["view"][5] = 0.1

        loss_high = float(reprojection_loss(joints_3d, kp2d, confs_high, cameras, COCO_TO_SMPL))
        loss_low = float(reprojection_loss(joints_3d, kp2d, confs_low, cameras, COCO_TO_SMPL))
        assert loss_high > loss_low, "High confidence should produce higher loss"

    def test_zero_confidence_skipped(self):
        """Observations with confidence < 0.1 should be ignored."""
        joints_3d = torch.zeros(1, 24, 3)
        R, t, K = _make_camera_tensors(np.array([3.0, 0.0, 0.0]))
        kp2d = {"view": torch.full((17, 2), 9999.0)}
        confs = {"view": torch.zeros(17)}  # all zero
        cameras = {"view": (R, t, K)}

        loss = reprojection_loss(joints_3d, kp2d, confs, cameras, COCO_TO_SMPL)
        assert float(loss) == 0.0, "Zero confidence should produce zero loss"


# ---------------------------------------------------------------------------
# Regularisation losses
# ---------------------------------------------------------------------------


class TestRegularisationLosses:
    def test_pose_prior_zero_at_neutral(self):
        pose = torch.zeros(1, 69)
        assert float(pose_prior_loss(pose)) == 0.0

    def test_pose_prior_positive_off_neutral(self):
        pose = torch.ones(1, 69)
        assert float(pose_prior_loss(pose)) > 0.0

    def test_shape_reg_zero_at_mean(self):
        betas = torch.zeros(1, 10)
        assert float(shape_regularisation(betas)) == 0.0

    def test_shape_reg_positive_off_mean(self):
        betas = torch.ones(1, 10) * 2.0
        assert float(shape_regularisation(betas)) > 0.0
