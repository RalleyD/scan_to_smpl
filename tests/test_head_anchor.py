"""Tests for the vertex-anchored head term (W2 head-anchor fix).

Two layers:
  * GPU/model test: guards the *rationale* — the ears-midpoint VERTEX sits well
    above and behind head joint 15, which is why anchoring the 2D ears to joint
    15 biased the head "up and back". (Step-1 verification, made permanent.)
  * CPU unit tests: exercise the new vertex-anchored branch of
    ``reprojection_loss`` and the optimiser's new defaults — no model needed.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scantosmpl.fitting.losses import reprojection_loss
from scantosmpl.fitting.optimiser import SMPLOptimiser
from scantosmpl.smpl.joint_map import (
    COCO_TO_SMPL,
    HEAD_MIDPOINT_TO_VERTEX,
    SMPL_LEFT_EAR_VERTEX,
    SMPL_RIGHT_EAR_VERTEX,
)

SMPL_DIR = "models/smpl"


def _smpl_available() -> bool:
    return (Path(SMPL_DIR) / "SMPL_NEUTRAL.pkl").exists()


requires_smpl = pytest.mark.skipif(
    not _smpl_available(),
    reason=f"SMPL model files not found in {SMPL_DIR}/ — see models/README.md",
)


def _make_camera_tensors(pos: np.ndarray, focal: float = 1000.0, w: int = 1000, h: int = 1000):
    """Synthetic pinhole camera looking at the origin (mirrors test_fitting.py)."""
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


# ---------------------------------------------------------------------------
# Step-1 rationale guard (needs the SMPL model)
# ---------------------------------------------------------------------------


@requires_smpl
@pytest.mark.gpu
def test_ear_vertex_sits_above_and_behind_head_joint():
    """The bias that motivates the fix: ears-midpoint VERTEX vs head JOINT 15.

    On the neutral template the ears-midpoint vertex is measurably ABOVE (+~6.7cm)
    and slightly BEHIND (-~3cm) joint 15. Anchoring the 2D ears to joint 15
    therefore lifts/tilts the head — hence the vertex anchor.
    """
    from scantosmpl.smpl.model import SMPLModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = SMPLModel(model_dir=SMPL_DIR, gender="neutral", device=device)
    with torch.no_grad():
        out = smpl.forward()
    V = out.vertices.squeeze(0).cpu().numpy()
    J = out.joints.squeeze(0).cpu().numpy()

    head = J[15]
    ears_mid = 0.5 * (V[SMPL_LEFT_EAR_VERTEX] + V[SMPL_RIGHT_EAR_VERTEX])
    offset = ears_mid - head  # SMPL frame: X=left, Y=up, Z=forward

    dy, dz = float(offset[1]), float(offset[2])
    assert 0.04 < dy < 0.09, f"ears-mid should be ~+6.7cm above joint 15, got {dy * 100:.1f}cm"
    assert -0.06 < dz < 0.0, f"ears-mid should be behind joint 15, got {dz * 100:.1f}cm"


# ---------------------------------------------------------------------------
# CPU unit tests for the vertex-anchored loss branch
# ---------------------------------------------------------------------------


def _vertices_with_ears(left: np.ndarray, right: np.ndarray) -> torch.Tensor:
    """(1, V, 3) vertex tensor with the two ear vertices set, rest zero."""
    n = max(SMPL_LEFT_EAR_VERTEX, SMPL_RIGHT_EAR_VERTEX) + 1
    verts = torch.zeros(1, n, 3)
    verts[0, SMPL_LEFT_EAR_VERTEX] = torch.tensor(left, dtype=torch.float32)
    verts[0, SMPL_RIGHT_EAR_VERTEX] = torch.tensor(right, dtype=torch.float32)
    return verts


class TestVertexMidpointReprojection:
    def _setup(self):
        joints = torch.zeros(1, 24, 3)
        verts = _vertices_with_ears([-0.07, 0.5, 2.5], [0.07, 0.5, 2.5])  # mid = (0,0.5,2.5)
        R, t, K = _make_camera_tensors(np.array([3.0, 0.0, 0.0]))
        ear_mid = 0.5 * (verts[0, SMPL_LEFT_EAR_VERTEX] + verts[0, SMPL_RIGHT_EAR_VERTEX])
        p_cam = R @ ear_mid + t
        p_h = K @ p_cam
        proj = (p_h[:2] / p_h[2]).numpy()
        return joints, verts, (R, t, K), proj

    def test_zero_when_2d_ears_match_projected_vertex_midpoint(self):
        joints, verts, (R, t, K), proj = self._setup()
        kp2d = {"view": torch.zeros(17, 2)}
        kp2d["view"][3] = torch.tensor(proj)  # left_ear
        kp2d["view"][4] = torch.tensor(proj)  # right_ear (midpoint == proj)
        confs = {"view": torch.zeros(17)}
        confs["view"][3] = 1.0
        confs["view"][4] = 1.0

        loss = reprojection_loss(
            joints, kp2d, confs, {"view": (R, t, K)}, COCO_TO_SMPL,
            vertices_pred=verts, vertex_midpoint_to_smpl=HEAD_MIDPOINT_TO_VERTEX,
        )
        assert float(loss) < 1e-4, f"expected ~0 when aligned, got {float(loss)}"

    def test_positive_when_2d_ears_offset(self):
        joints, verts, (R, t, K), proj = self._setup()
        kp2d = {"view": torch.zeros(17, 2)}
        kp2d["view"][3] = torch.tensor(proj + np.array([120.0, 80.0]))
        kp2d["view"][4] = torch.tensor(proj + np.array([120.0, 80.0]))
        confs = {"view": torch.zeros(17)}
        confs["view"][3] = 1.0
        confs["view"][4] = 1.0

        loss = reprojection_loss(
            joints, kp2d, confs, {"view": (R, t, K)}, COCO_TO_SMPL,
            vertices_pred=verts, vertex_midpoint_to_smpl=HEAD_MIDPOINT_TO_VERTEX,
        )
        assert float(loss) > 1.0, f"expected penalty on offset, got {float(loss)}"

    def test_no_op_without_vertices(self):
        """Passing the vertex map but no vertices must not add any term."""
        joints, verts, (R, t, K), proj = self._setup()
        kp2d = {"view": torch.zeros(17, 2)}
        kp2d["view"][3] = torch.tensor(proj + 500.0)
        kp2d["view"][4] = torch.tensor(proj + 500.0)
        confs = {"view": torch.zeros(17)}
        confs["view"][3] = 1.0
        confs["view"][4] = 1.0

        loss = reprojection_loss(
            joints, kp2d, confs, {"view": (R, t, K)}, COCO_TO_SMPL,
            vertices_pred=None, vertex_midpoint_to_smpl=HEAD_MIDPOINT_TO_VERTEX,
        )
        assert float(loss) == 0.0, f"no vertices → no term, got {float(loss)}"

    def test_gradient_flows_to_vertices(self):
        """The vertex-anchored term must be differentiable wrt the vertices."""
        joints, verts, (R, t, K), proj = self._setup()
        verts = verts.clone().requires_grad_(True)
        kp2d = {"view": torch.zeros(17, 2)}
        kp2d["view"][3] = torch.tensor(proj + np.array([50.0, 0.0]))
        kp2d["view"][4] = torch.tensor(proj + np.array([50.0, 0.0]))
        confs = {"view": torch.zeros(17)}
        confs["view"][3] = 1.0
        confs["view"][4] = 1.0

        loss = reprojection_loss(
            joints, kp2d, confs, {"view": (R, t, K)}, COCO_TO_SMPL,
            vertices_pred=verts, vertex_midpoint_to_smpl=HEAD_MIDPOINT_TO_VERTEX,
        )
        loss.backward()
        assert verts.grad is not None
        g = verts.grad[0]
        assert torch.norm(g[SMPL_LEFT_EAR_VERTEX]) > 0
        assert torch.norm(g[SMPL_RIGHT_EAR_VERTEX]) > 0


# ---------------------------------------------------------------------------
# Optimiser defaults
# ---------------------------------------------------------------------------


def test_optimiser_defaults_to_vertex_head_term():
    """Default head term is vertex-anchored; joint-anchored term is off by default."""
    stub = SimpleNamespace(device=torch.device("cpu"))
    opt = SMPLOptimiser(stub, COCO_TO_SMPL)
    assert opt.vertex_midpoint_to_smpl == HEAD_MIDPOINT_TO_VERTEX
    assert opt.midpoint_to_smpl == {}
