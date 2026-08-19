"""Semantic body-part weights derived from `lbs_weights`, not cloud segmentation.

REVIEW.md Phase 6 originally called for "height slices + PCA/connectivity" cloud
segmentation. Master D7 supersedes that: AC 7.3 needs per-**mesh-vertex** part
weights, and `body_model.lbs_weights` (6890, 24) already encodes exactly that
association — `argmax(axis=1)` gives the dominant joint per vertex, and a fixed
joint -> group table (below) gives the part. This is exact, deterministic and
needs no heuristic. Cloud points then inherit a label from their nearest mesh
vertex (`transfer_labels_to_cloud`), so the semantic weighting used by the
Tier 3 chamfer loss covers both the mesh and the cloud from a single source of
truth.

This module never imports `SMPLModel` or `scantosmpl.config` — `lbs_weights` is
passed in as a plain `(6890, 24)` array by the caller, so it is testable without
model weights (see the brief's "Consumes" boundary).
"""

import numpy as np
from scipy.spatial import cKDTree

SMPL_NUM_JOINTS = 24

# Master Section 5.3, verbatim. The six groups partition all 24 SMPL joints
# (asserted below at import time) and the keys match
# `Tier3Config.body_part_weights` exactly (D7) — duplicated here rather than
# imported so this module stays independent of `scantosmpl.config`.
SMPL_PART_GROUPS: dict[str, list[int]] = {
    "torso": [0, 3, 6, 9, 12, 13, 14],  # pelvis, spine1-3, neck, collars
    "head": [15],
    "arms": [16, 17, 18, 19],  # shoulders, elbows
    "hands": [20, 21, 22, 23],  # wrists, hands
    "legs": [1, 2, 4, 5],  # hips, knees
    "feet": [7, 8, 10, 11],  # ankles, feet
}

_GROUP_NAMES: tuple[str, ...] = tuple(SMPL_PART_GROUPS.keys())


def _build_joint_to_group() -> np.ndarray:
    """(24,) int64 array mapping SMPL joint id -> part-group id.

    Group id is the index of the group's name in `SMPL_PART_GROUPS` (i.e. in
    `_GROUP_NAMES`), so it is stable as long as the dict's key order is.
    """
    mapping = np.full(SMPL_NUM_JOINTS, -1, dtype=np.int64)
    seen: set[int] = set()
    for group_id, name in enumerate(_GROUP_NAMES):
        for joint in SMPL_PART_GROUPS[name]:
            if joint in seen:
                raise AssertionError(
                    f"SMPL_PART_GROUPS is not a partition: joint {joint} appears in "
                    f"more than one group (duplicate via {name!r})"
                )
            seen.add(joint)
            mapping[joint] = group_id
    expected = set(range(SMPL_NUM_JOINTS))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise AssertionError(
            f"SMPL_PART_GROUPS does not partition the {SMPL_NUM_JOINTS} SMPL joints: "
            f"missing={missing} extra={extra}"
        )
    return mapping


# Asserted once at import time (brief step 4): a corrupted SMPL_PART_GROUPS
# fails immediately and loudly, rather than producing silently-wrong labels.
_JOINT_TO_GROUP = _build_joint_to_group()


def smpl_part_labels(lbs_weights: np.ndarray) -> np.ndarray:
    """Per-vertex part-group id from SMPL linear-blend-skinning weights.

    Args:
        lbs_weights: (6890, 24) skinning weights (any float dtype).

    Returns:
        (V,) int64 group ids in `[0, len(SMPL_PART_GROUPS))`, indexing into
        `list(SMPL_PART_GROUPS)` — vertex `v`'s label is the group owning the
        joint with the largest skinning weight at `v` (D7).

    Raises:
        ValueError: If `lbs_weights` is not `(V, 24)`.
    """
    weights = np.asarray(lbs_weights)
    if weights.ndim != 2 or weights.shape[1] != SMPL_NUM_JOINTS:
        raise ValueError(f"lbs_weights must be (V, {SMPL_NUM_JOINTS}), got {weights.shape}")
    dominant_joint = weights.argmax(axis=1)
    return np.asarray(_JOINT_TO_GROUP[dominant_joint], dtype=np.int64)


def vertex_part_weights(lbs_weights: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Per-vertex scalar loss weight from named part-group weights.

    Args:
        lbs_weights: (6890, 24) skinning weights.
        weights: `{group_name: weight}`, e.g. `Tier3Config.body_part_weights`.
            A group absent from `weights` defaults to `0.0`. Every key MUST be
            one of `SMPL_PART_GROUPS`.

    Returns:
        (V,) float32 per-vertex weight.

    Raises:
        ValueError: If `weights` names a group that does not exist.
    """
    unknown = sorted(set(weights) - set(SMPL_PART_GROUPS))
    if unknown:
        raise ValueError(
            f"Unknown body part group(s) {unknown}; expected one of {sorted(SMPL_PART_GROUPS)}"
        )
    labels = smpl_part_labels(lbs_weights)
    group_weight = np.array(
        [weights.get(name, 0.0) for name in _GROUP_NAMES],
        dtype=np.float32,
    )
    return np.asarray(group_weight[labels], dtype=np.float32)


def transfer_labels_to_cloud(
    cloud_points: np.ndarray,
    mesh_vertices: np.ndarray,
    vertex_labels: np.ndarray,
) -> np.ndarray:
    """Assign each cloud point the label of its nearest mesh vertex.

    Args:
        cloud_points: (N, 3) points, same frame/units as `mesh_vertices`
            (normally `frame="smpl_world"`, `units="metres"` — post-alignment).
        mesh_vertices: (V, 3) SMPL vertices.
        vertex_labels: (V,) labels, e.g. from `smpl_part_labels`.

    Returns:
        (N,) labels, same dtype as `vertex_labels`.

    Raises:
        ValueError: On a shape mismatch between `mesh_vertices` and `vertex_labels`.
    """
    points = np.asarray(cloud_points, dtype=np.float64)
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    labels = np.asarray(vertex_labels)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"cloud_points must be (N, 3), got {points.shape}")
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"mesh_vertices must be (V, 3), got {verts.shape}")
    if labels.shape != (verts.shape[0],):
        raise ValueError(
            f"vertex_labels must be ({verts.shape[0]},) to match mesh_vertices, got {labels.shape}"
        )

    tree = cKDTree(verts)
    _, nearest = tree.query(points, k=1)
    return np.asarray(labels[nearest])
