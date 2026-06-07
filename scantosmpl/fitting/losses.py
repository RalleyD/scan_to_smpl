"""Differentiable loss functions for SMPL fitting (Phase 5)."""

import torch
import torch.nn.functional as F


def joint_loss(
    joints_pred: torch.Tensor,
    joints_target: torch.Tensor,
    joint_weights: torch.Tensor | None = None,
    huber_delta: float = 0.05,
) -> torch.Tensor:
    """Huber loss on 3D joint positions.

    Args:
        joints_pred: (1, J, 3) or (J, 3) predicted SMPL joint positions (metres).
        joints_target: (J, 3) triangulated ground-truth joint positions (metres).
        joint_weights: (J,) per-joint importance weights. None = uniform.
        huber_delta: Huber delta (metres). Noise ≤ delta stays quadratic;
            outliers above delta contribute linearly.

    Returns:
        Scalar loss tensor.
    """
    pred = joints_pred.squeeze(0)  # (J, 3)
    target = joints_target.to(pred.device)

    # Per-joint L2 distance, then Huber
    diff = pred - target                    # (J, 3)
    err = torch.norm(diff, dim=-1)          # (J,)
    loss = F.huber_loss(err, torch.zeros_like(
        err), delta=huber_delta, reduction="none")  # (J,)

    if joint_weights is not None:
        w = joint_weights.to(pred.device)
        loss = loss * w

    return loss.mean()


def reprojection_loss(
    joints_pred: torch.Tensor,
    keypoints_2d: dict[str, torch.Tensor],
    confs: dict[str, torch.Tensor],
    cameras: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    coco_to_smpl: dict[int, int],
    huber_delta: float = 20.0,
) -> torch.Tensor:
    """Confidence-weighted multi-view reprojection loss with Huber robustness.

    Projects SMPL joints into each view and compares with undistorted ViTPose
    2D keypoints, weighted by per-keypoint confidence.

    Args:
        joints_pred: (1, J_smpl, 3) world-space SMPL joints (metres).
        keypoints_2d: {view_name: (J_coco, 2)} undistorted 2D keypoints (pixels).
        confs: {view_name: (J_coco,)} per-keypoint confidence scores.
        cameras: {view_name: (R, t, K)} each a tuple of (3,3), (3,), (3,3) tensors.
        coco_to_smpl: Mapping from COCO keypoint index to SMPL joint index.
        huber_delta: Huber delta (pixels).

    Returns:
        Scalar loss tensor (mean over all views and keypoints).
    """
    device = joints_pred.device
    pred_joints = joints_pred.squeeze(0)  # (J_smpl, 3)

    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for view_name, (R, t, K) in cameras.items():
        if view_name not in keypoints_2d:
            continue

        R = R.to(device)
        t = t.to(device)
        K = K.to(device)
        kp2d = keypoints_2d[view_name].to(device)   # (J_coco, 2)
        conf = confs[view_name].to(device)            # (J_coco,)

        # Project SMPL joints: p_cam = R @ p_world + t
        # pts_cam: (J_smpl, 3)
        pts_cam = (R @ pred_joints.T).T + t

        # Perspective divide
        pts_img_h = (K @ pts_cam.T).T          # (J_smpl, 3)
        pts_img = pts_img_h[:, :2] / pts_img_h[:,
                                               2:3].clamp(min=1e-6)  # (J_smpl, 2)

        # Match COCO indices to SMPL joints
        for coco_idx, smpl_idx in coco_to_smpl.items():
            if coco_idx >= kp2d.shape[0]:
                continue
            w = conf[coco_idx]
            if w < 0.1:  # skip near-zero confidence
                continue

            proj = pts_img[smpl_idx]      # (2,)
            obs = kp2d[coco_idx]          # (2,)
            err = torch.norm(proj - obs)  # scalar pixel error
            huber = F.huber_loss(err, torch.zeros_like(err), delta=huber_delta)
            total_loss = total_loss + w * huber
            n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / n_terms


def pose_prior_loss(body_pose: torch.Tensor) -> torch.Tensor:
    """L2 regularisation toward neutral (zero) pose.

    Args:
        body_pose: (1, 69) body pose parameters.

    Returns:
        Scalar loss.
    """
    return (body_pose ** 2).mean()


def shape_regularisation(betas: torch.Tensor) -> torch.Tensor:
    """L2 regularisation toward mean shape (zero betas).

    Args:
        betas: (1, 10) shape parameters.

    Returns:
        Scalar loss.
    """
    return (betas ** 2).mean()
