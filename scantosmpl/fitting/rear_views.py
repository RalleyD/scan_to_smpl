"""Rear-view camera classification (extracted from ``SMPLOptimiser``).

Both the staged optimiser and the Phase 5 pipeline need to know which
cameras view the subject's back so they can be excluded from the
reprojection loss / PnP refinement. Living here (module-level, no class
required) lets both call sites reuse the same logic without instantiating
an ``SMPLOptimiser``.
"""

import logging
from typing import Literal

import numpy as np

from scantosmpl.hmr.consensus import ConsensusResult
from scantosmpl.smpl.joint_map import Smpl24Joint

logger = logging.getLogger(__name__)

# Graded view-angle categories, ordered front-to-back.
ViewAngle = Literal["frontal", "three_quarter", "profile", "rear"]


def _body_back_vec(consensus: ConsensusResult) -> np.ndarray | None:
    """Unit back-vector of the consensus body, or None on degenerate geometry.

    cross(neck-pelvis, left_shoulder-right_shoulder) points along the subject's
    back (equivalent to -body_front). See docs/phase5_spec_supplement.md §A1.
    """
    shoulder_vec = (
        consensus.joints[Smpl24Joint.LEFT_SHOULDER] - consensus.joints[Smpl24Joint.RIGHT_SHOULDER]
    )
    up_vec = consensus.joints[Smpl24Joint.NECK] - consensus.joints[Smpl24Joint.PELVIS]
    # order matters (up_vec, shoulder_vec) so the result is -Z for a rear view
    body_back_vec = np.cross(up_vec, shoulder_vec)
    norm = float(np.linalg.norm(body_back_vec))
    if norm < 1e-6:
        return None
    unit: np.ndarray = body_back_vec / norm
    return unit


def classify_rear_views(
    consensus: ConsensusResult,
    cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> set[str]:
    """Return the names of cameras whose centre lies on the subject's back side.

    Uses cross(neck-pelvis, left_shoulder-right_shoulder) as the SMPL back-vector
    (equivalent to -body_front); a camera whose offset from the pelvis has positive
    dot with the back-vector is classified as rear. See docs/phase5_spec_supplement.md
    §A1 for derivation. Returns empty set on degenerate consensus geometry.
    """
    body_back_vec = _body_back_vec(consensus)
    if body_back_vec is None:
        return set()

    rear_views = []
    for name, (R, t, K) in cameras.items():
        cam_centre = -R.T @ t
        cam_offset = cam_centre - consensus.joints[Smpl24Joint.PELVIS]
        cam_dot = np.dot(cam_offset, body_back_vec)
        if cam_dot > 0:
            rear_views.append(name)

    logger.info("Rear views detected: %s", rear_views)

    return set(rear_views)


def classify_view_angles(
    consensus: ConsensusResult,
    cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    profile_cos: float = 0.35,
    three_quarter_cos: float = 0.85,
) -> dict[str, ViewAngle]:
    """Grade each camera by how frontal it is: frontal / three_quarter / profile / rear.

    Extends :func:`classify_rear_views` from a binary front/rear split to a
    graded one (W3). The grade is the cosine of the angle between the
    pelvis→camera offset and the body-*front* vector (-body_back), measured in
    the horizontal (torso) plane so camera height doesn't leak into the angle:

        cos >= three_quarter_cos          -> "frontal"      (facing the subject)
        profile_cos <= cos < three_q_cos  -> "three_quarter"
        0 <  cos < profile_cos            -> "profile"      (side-on; systematic
                                                             ViTPose error)
        cos <= 0                          -> "rear"         (behind the subject;
                                                             consistent with
                                                             classify_rear_views)

    A view labelled "rear" here is exactly one classify_rear_views would return,
    so the two stay consistent. On degenerate consensus geometry every camera is
    graded "frontal" (no basis to down-weight anything → fall back to the
    unweighted behaviour that predated W3).

    Args:
        consensus: Phase 3 consensus (supplies the body basis vectors).
        cameras: {view_name: (R, t, K)}.
        profile_cos: cosine boundary between profile and three-quarter.
        three_quarter_cos: cosine boundary between three-quarter and frontal.

    Returns:
        {view_name: view_angle}.
    """
    body_back_vec = _body_back_vec(consensus)
    if body_back_vec is None:
        return {name: "frontal" for name in cameras}
    body_front_vec = -body_back_vec

    # Project out the vertical (up) component so grading uses the horizontal
    # camera azimuth only — a high or low camera on the front arc should still
    # read as frontal, not get demoted for its elevation.
    up_vec = consensus.joints[Smpl24Joint.NECK] - consensus.joints[Smpl24Joint.PELVIS]
    up_norm = np.linalg.norm(up_vec)
    up_hat = up_vec / up_norm if up_norm > 1e-6 else None

    angles: dict[str, ViewAngle] = {}
    for name, (R, t, K) in cameras.items():
        cam_centre = -R.T @ t
        offset = cam_centre - consensus.joints[Smpl24Joint.PELVIS]
        if up_hat is not None:
            offset = offset - np.dot(offset, up_hat) * up_hat
        offset_norm = np.linalg.norm(offset)
        if offset_norm < 1e-6:
            angles[name] = "frontal"
            continue
        cos = float(np.dot(offset, body_front_vec) / offset_norm)

        if cos <= 0.0:
            angle: ViewAngle = "rear"
        elif cos < profile_cos:
            angle = "profile"
        elif cos < three_quarter_cos:
            angle = "three_quarter"
        else:
            angle = "frontal"
        angles[name] = angle

    return angles
