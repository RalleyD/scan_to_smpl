"""Pose-space features `x(θ)` — the RBF's interpolation coordinate. Plan step S1.

The ONLY thing this module exists to do: turn a pose into a vector whose Euclidean
distance to another pose's vector is a sensible measure of "how differently posed".
That distance is all the RBF kernel ever consumes:

    Phi_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))

Nothing here knows about vertices, displacements, or frames.
A pose-feature is derived from joint axis-angles

Decisions this module implements (see docs/psd-implementation-plan.md):

  D2  23 body joints only. `global_orient` is EXCLUDED — it is where the subject
      faces, not how they deform. Including it would make the same arm bend predict
      different flesh depending on compass direction.

  D3  Encoding is `R_k - I`, flattened. Rejected:
        - axis-angle: wraps at 2*pi, so one pose has infinitely many distant features
        - quaternions: double cover, `q` and `-q` are one rotation with two features
      `R_k - I` is unique per rotation, continuous, and EXACTLY zero at rest — so a
      predicted delta at rest is structurally zero, not approximately.

Reuse, do not reimplement: `scantosmpl.utils.geometry.aa_to_rotmat` already maps
(..., 3) axis-angle -> (..., 3, 3) float64 rotation matrices.
This represents R_k.

---------------------------------------------------------------------------
YOUR JOB: complete the interface below, then make tests/test_psd_features.py pass.

The tests are the specification. Read them first — they answer every question in
the TODOs, but you have to derive the answer rather than copy a signature. In
particular work out, from the assertions:

  * what comes in: one pose, or many? what shape, what dtype?
  * what goes out: what length, and WHY that length and not 216?
  * how the 23 per-joint 3x3 blocks are laid out in the flat output
  * which input the function must NOT be able to see (D2), and whether you
    enforce that by ignoring it or by making it unrepresentable
---------------------------------------------------------------------------
"""

import numpy as np

from scantosmpl.utils.geometry import aa_to_rotmat

# SMPL body_pose is 23 joints x 3 axis-angle numbers. global_orient is a SEPARATE (3,).
N_JOINT_AXIS = 3
N_BODY_JOINTS = 23
BODY_POSE_DIM = N_BODY_JOINTS * N_JOINT_AXIS  # 69
N_FEATURE_DIM = BODY_POSE_DIM * 3  # 207


def pose_features(poses: np.ndarray[tuple[int, int] | tuple[int], np.float64]):
    """Pose -> interpolation coordinate.

    Encodes the pose-feature to a 207-dimension space.

    Needed for RBF kernel:
    Without duplicates (single-valued), the same pose means the same features.
    I.e. the same pose rotated 2*Pi isn't encoded as a different pose.
    Continuous, leveraging rotation matrices (R), nearby poses represent reliably
    as nearby features i.e. No canonicalisation to break this seam.

    It's structurally zero at rest pose, I,
    aligned with the SMPL pose blendshape implementation

    Args:
        poses: batch arrays of shape (N, 69) numbers, float64.
               A single (69, ) pose is valid too.
    Returns:
        ndarray: (..., 207) float64.
                i.e. A batch (x) of flattened arrays (3,3) -> (9,)
                of 23 body joints * 9
    """
    # batch is either () or (N) - slice a tuple to get back a tuple
    batch = poses.shape[:-1]
    # -1 shall always be the data axis
    data_shp = poses.shape[-1]
    if data_shp != BODY_POSE_DIM:
        msg = (
            f"expected {BODY_POSE_DIM} joints in the pose feature data axis. "
            f"Got shape {poses.shape}"
        )
        raise ValueError(msg)
    # 1 - keep the batch, then reshape
    feature = poses.reshape(*batch, N_BODY_JOINTS, N_JOINT_AXIS)
    # 2 - call aa_to_rotmax
    # 3 - subtract the identity matrix - will broadcast
    r_k = aa_to_rotmat(feature) - np.identity(N_JOINT_AXIS)
    # 4 - reshape (batch, 3, 3) -> (batch, 207)
    return r_k.reshape(*batch, N_FEATURE_DIM)


# S2 - frame conversion `D != gamma`
