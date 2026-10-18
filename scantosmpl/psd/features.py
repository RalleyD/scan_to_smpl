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


def pose_features(poses: np.ndarray[tuple[int, int], np.float64]):
    """Pose -> interpolation coordinate.

    TODO(S1): document and implement.
    Encodes the pose-feature to a 207-dimension space.

    Args:
        poses: batch arrays of N*3 joints, float64, 
               and whether a leading batch dim is allowed.

    Returns:
        ndarray: (x, 209) float64. A batch (x) of flattened arrays (3,3) -> (9,) 
                  of 23 body joints * 9
    """
    foo = []

    def bar(x: np.ndarray):
        # 1 -reshape to (,3) axis-angle vectors
        feature = x.reshape(-1, 3)
        print(feature.shape)
        # 2 - call aa_to_rotmax
        # 3 - subtract the identity matrix
        r_k = aa_to_rotmat(feature) - np.identity(3)
        # 4 - flatten -> (9,) and return
        return r_k.flatten()

    if len(poses.shape) == 1:
        return bar(poses)

    for pose in poses:
        foo.append(bar(pose))
    # n.b this will stack the batch as rows (provided they're equal length)
    # np.array does this, but this is explicit
    return np.stack(foo)
