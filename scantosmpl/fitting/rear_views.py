"""Rear-view camera classification (extracted from ``SMPLOptimiser``).

Both the staged optimiser and the Phase 5 pipeline need to know which
cameras view the subject's back so they can be excluded from the
reprojection loss / PnP refinement. Living here (module-level, no class
required) lets both call sites reuse the same logic without instantiating
an ``SMPLOptimiser``.
"""

import logging

import numpy as np

from scantosmpl.hmr.consensus import ConsensusResult
from scantosmpl.smpl.joint_map import Smpl24Joint

logger = logging.getLogger(__name__)


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
    # left shoulder at +X
    shoulder_vec = (
        consensus.joints[Smpl24Joint.LEFT_SHOULDER] - consensus.joints[Smpl24Joint.RIGHT_SHOULDER]
    )
    up_vec = consensus.joints[Smpl24Joint.NECK] - consensus.joints[Smpl24Joint.PELVIS]
    # should be -Z for rear view, +Z for front view
    # order is important for -Z: up_vec, shoulder_vec
    body_back_vec = np.cross(up_vec, shoulder_vec)

    # in case the body pose is undeterminate, normalise the front vector to avoid
    # classifying all vectors as rear
    norm = np.linalg.norm(body_back_vec)
    if norm < 1e-6:
        return set()
    body_back_vec = body_back_vec / norm

    rear_views = []
    for name, (R, t, K) in cameras.items():
        cam_centre = -R.T @ t
        cam_offset = cam_centre - consensus.joints[Smpl24Joint.PELVIS]
        cam_dot = np.dot(cam_offset, body_back_vec)
        if cam_dot > 0:
            rear_views.append(name)

    logger.info("Rear views detected: %s", rear_views)

    return set(rear_views)
