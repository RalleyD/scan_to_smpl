"""Frame conversion `D_posed <-> delta` — the round trip every PSD number takes. Plan step S2.

WHY THIS EXISTS. A scan residual `D_posed` is measured in the posed world: "this vertex
sits 8mm further out along world +x". That number is useless for interpolation, because
the same bulging bicep reads as +x in one pose and -z in another — the arm turned, so
the bulge turned with it. The RBF would see two different targets for one piece of flesh.

So every residual is rotated back into the vertex's REST frame before training
(`to_local`), and every prediction is rotated out again at runtime (`to_posed`). In the
rest frame, "8mm out of the bicep" is the same three numbers in every pose; only the
genuinely pose-dependent part (the bulge growing as the elbow bends) is left to learn.

The matrix that does the rotating is the one skinning itself uses:

    M_v = sum_k skin_w[v, k] * R_k          (R_k = joint k's WORLD rotation, chained)

    delta_v   = M_v^-1 @ D_v                 to_local   (training: scan -> rest)
    D_posed_v = M_v    @ delta_v             to_posed   (runtime: rest -> scan)

Decisions this module implements (see docs/psd-implementation-plan.md):

  D1  TRUE inverse `M_v^-1`, not polar `R~^T` and not `M_v^T`. `M_v` is a weighted SUM
      of rotations, which is not a rotation (SO(3) is not closed under addition), so its
      transpose is not its inverse. The consumer — Maya, Unreal, our own SMPL+D — adds
      delta to the template BEFORE skinning, which applies exactly `M_v`, so `M_v` is
      the only matrix whose inverse undoes it. Guard: `cond(M_v) < 1e6`, fail loudly
      and name the worst vertex.

Note the asymmetry with S1: the FEATURE excludes global_orient (D2), but the FRAME needs
it. The scan was captured facing some direction; undoing the pose means undoing that too.

Reuse, do not reimplement: `scantosmpl.utils.geometry.aa_to_rotmat`.

---------------------------------------------------------------------------
YOUR JOB: complete the four interfaces below, then make tests/test_psd_frames.py pass.

The tests are the specification. Read them first, and work out from the assertions:

  * world_rotations: what goes in (and why global_orient is separate from body_pose
    here too), what `parents` encodes, and what ordering assumption lets you build
    every joint in ONE forward pass. Why must a child's rotation be parent @ local,
    not local @ parent?
  * blend_matrices: (V, J) weights against (J, 3, 3) rotations -> (V, 3, 3). One
    einsum, no loop. Which index is summed away?
  * to_local / to_posed: per-vertex 3x3 solve and multiply, batched over V.
    numpy has a trap here that only bites when V == 3 — the tests name it.
  * where the cond(M_v) guard lives, and what its error message must contain.
---------------------------------------------------------------------------
"""

import numpy as np  # noqa: F401  (yours to use)

from scantosmpl.utils.geometry import aa_to_rotmat  # noqa: F401  (yours to use)

# TODO check return type annotation correct


def world_rotations(
        global_orient: np.ndarray[tuple[int], np.dtype[np.float64]],
        body_pose: np.ndarray[tuple[int], np.dtype[np.float64]],
        parents: np.ndarray[tuple[int], np.dtype[np.float64]]
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """Local axis-angle per joint -> world rotation per joint, down the kinematic tree."""
    # input joints start with the root
    joints = np.concatenate((global_orient, body_pose))
    # joints are flat arrays, convert to axis-angle dimensions
    joints = joints.reshape(-1, 3)
    # 1: allocate output array
    worlds = np.empty((len(joints), 3, 3))
    # R_local broadcast AA->Rotaion matrix
    # - describes a twist inside the parent's frame
    R_local = aa_to_rotmat(joints)
    # root has no parent, world roation == R_local
    worlds[0] = R_local[0]

    for k in range(1, len(R_local)):
        # SMPL guarantees parents[k] < k
        parent = parents[k]
        if parent >= k:
            raise ValueError
        # world rotation relative to the parent's frame
        # i.e. Parent's world rotation carries the result
        # - matrix product apples right-hand factor first
        worlds[k] = worlds[parent] @ R_local[k]

    return worlds


def blend_matrices(
        rot_mat_world: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
        skin_weights: np.ndarray[tuple[int, int],
                                 np.dtype[np.dtype[np.float64]]]
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """
    World joint rotations + skinning weights -> per-vertex blended matrix `M_v`.
    i.e. A blended LBS transform for a given vertex.

    The same mix of joint rotations that LBS applies to a given vertex.

    Side note:
      Rotations only, no translations. LBS uses full transforms
        [R_k | t_k], but gamma is an offset, not a point.
        Translation doesn't move offsets.

    Args:
      rot_mat_world (ndarray): Pure world rotations (R_k)
                               shape (k, 3, 3)
      skin_weights (ndarray): For a given vertex, there are 24 joint-based
                              blend weights. Shape (vertices, joints) ->
                              (v, k)
    """
    # sum over joints (k), giving per-vertex (v) output
    # a scalar times a 3x3, summed; k is contracted.
    # M[v] = sum_k( w[v, k] * R[k] )
    # the output is the dot product of:
    # - the vth row of the skin weights w[v, :] -> the vertex's per-joint weights
    # - across each k matrix element (a,b) -> R[:, a, b]
    # - then each joint matrix summed.
    # this is the blending of vertex joint weights
    # - with joint rotation matrices
    # i.e. scale each matrix by it's corresponding weight
    # - then sum the matrixes together.
    # einsum does the dot product and sum for each vertex
    return np.einsum(
        "vk,kab->vab",
        skin_weights,
        rot_mat_world
    )


def to_local():
    """Posed-world residual -> rest-frame delta. `M_v^-1`, guarded (D1)."""
    raise NotImplementedError


def to_posed():
    """Rest-frame delta -> posed-world residual. `M_v`."""
    raise NotImplementedError
