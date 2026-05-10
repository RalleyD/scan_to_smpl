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

# Hardcoded SMPL vertex indices for the 138 dense keypoints.
# Extracted from models/mapping/downsample_mat.pkl — a one-hot (138, 6890)
# sparse matrix where each row maps to exactly one SMPL vertex.
# fmt: off
DENSE_KP_VERTEX_INDICES = np.array([
     102,  189,  254,  366,  425,  433,  444,  542,  599,  639,
     659,  716,  735,  740,  781,  861,  876,  889,  962, 1007,
    1014, 1043, 1050, 1073, 1088, 1091, 1096, 1276, 1279, 1363,
    1403, 1425, 1492, 1703, 1725, 1738, 1742, 1863, 1887, 2033,
    2104, 2178, 2222, 2229, 2277, 2340, 2465, 2572, 2693, 2734,
    2787, 2821, 3001, 3022, 3057, 3076, 3094, 3102, 3117, 3128,
    3131, 3185, 3190, 3199, 3218, 3246, 3317, 3334, 3385, 3460,
    3499, 3508, 3551, 3615, 3734, 3762, 3774, 3813, 3866, 3892,
    4055, 4149, 4168, 4192, 4229, 4236, 4405, 4443, 4490, 4502,
    4505, 4510, 4535, 4585, 4664, 4745, 4757, 4759, 4876, 4900,
    4937, 4961, 4973, 4999, 5045, 5117, 5128, 5215, 5310, 5323,
    5346, 5376, 5473, 5480, 5518, 5527, 5565, 5595, 5789, 5926,
    6127, 6190, 6245, 6387, 6403, 6457, 6487, 6519, 6533, 6549,
    6590, 6595, 6616, 6626, 6644, 6734, 6789, 6824,
], dtype=np.int64)
# fmt: on

assert len(DENSE_KP_VERTEX_INDICES) == NUM_DENSE_KEYPOINTS
