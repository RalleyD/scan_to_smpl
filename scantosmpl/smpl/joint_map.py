"""Joint mapping constants: COCO-17 <-> SMPL-24 and CameraHMR dense keypoints."""

import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 -> SMPL-24 joint mapping
# ---------------------------------------------------------------------------
# Used for PnP correspondences and triangulation in Tier 2.
#
# COCO-17 keypoints:
#   0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
#   5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
#   9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
#   13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
#
# SMPL-24 joints:
#   0: pelvis, 1: left_hip, 2: right_hip, 3: spine1,
#   4: left_knee, 5: right_knee, 6: spine2, 7: left_ankle,
#   8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot,
#   12: neck, 13: left_collar, 14: right_collar, 15: head,
#   16: left_shoulder, 17: right_shoulder, 18: left_elbow,
#   19: right_elbow, 20: left_wrist, 21: right_wrist,
#   22: left_hand, 23: right_hand

COCO_TO_SMPL: dict[int, int] = {
    5: 16,   # left_shoulder
    6: 17,   # right_shoulder
    7: 18,   # left_elbow
    8: 19,   # right_elbow
    9: 20,   # left_wrist
    10: 21,  # right_wrist
    11: 1,   # left_hip
    12: 2,   # right_hip
    13: 4,   # left_knee
    14: 5,   # right_knee
    15: 7,   # left_ankle
    16: 8,   # right_ankle
}

# Derived keypoints (computed as midpoints of COCO pairs -> SMPL joints)
COCO_MIDPOINT_TO_SMPL: dict[tuple[int, int], int] = {
    (11, 12): 0,   # mid(left_hip, right_hip) -> pelvis
    (5, 6): 12,    # mid(left_shoulder, right_shoulder) -> neck
}

# Number of directly mappable joints (excluding midpoints)
NUM_DIRECT_CORRESPONDENCES = len(COCO_TO_SMPL)  # 12

# SMPL joint indices used for body height estimation
# Head top (15) to mean of ankles (7, 8)
HEIGHT_JOINTS = {"head": 15, "left_ankle": 7, "right_ankle": 8}

# SMPL joint indices for T-pose arm check
ARM_JOINTS = {
    "left_shoulder": 16,
    "left_elbow": 18,
    "left_wrist": 20,
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_wrist": 21,
}

# ---------------------------------------------------------------------------
# CameraHMR 138 dense keypoints
# ---------------------------------------------------------------------------
# DenseKP predicts 138 surface keypoints sampled from SMPL vertices using the
# COMA (Convolutional Mesh Autoencoder) downsampling pattern. These map
# directly to SMPL vertex indices — no conversion needed.
#
# The vertex indices are determined by the COMA downsampling of the SMPL
# template mesh. They are fixed in the DenseKP model architecture
# (NUM_DENSEKNP_SMPL = 138 in CameraHMR constants.py).
#
# For Tier 2 PnP, these 138 correspondences provide dramatically more robust
# extrinsic recovery than the 12 sparse COCO->SMPL joints above.

NUM_DENSE_KEYPOINTS = 138
