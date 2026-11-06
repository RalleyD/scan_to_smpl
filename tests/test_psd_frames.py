"""S2 — frame conversion `D_posed <-> delta`.

These tests ARE the spec for `scantosmpl.psd.frames`. Every PSD number passes through
here twice (scan -> rest for training, rest -> scan at runtime), and this is the one
step whose correctness is provable with no data at all — so it is proven to round-off.

Decision under test: D1 (true inverse `M_v^-1`, guarded by cond < 1e6).
Reference: docs/psd-implementation-plan.md S2 + D1, docs/psd_master_design_spec.md §4.2.

Two traps this file is built to dodge (plan §Traps):
  * Testing at theta = 0. There `M_v = I`, so `return D_posed` passes every round trip.
    Every round-trip test here runs at a pose the suite PROVES is non-trivial.
  * Self-consistency mistaken for correctness. `to_local` and `to_posed` built on the
    same WRONG matrix (polar, say) still invert each other perfectly. So each direction
    is also pinned against an independently written answer, and — when SMPL is present
    — against what a skinning runtime actually does.
"""

from pathlib import Path

import numpy as np
import pytest

from scantosmpl.psd.frames import blend_matrices, to_local, to_posed, world_rotations
from scantosmpl.utils.geometry import aa_to_rotmat, rotmat_to_aa

# Tolerance vocabulary, kept deliberately distinct:
#   ROUNDOFF  — "these should be the same number"; float64 noise only. This is the plan's
#               S2 gate. Measured: 1e-17 on the synthetic round trip, 3.5e-16 against smplx.
#   RODRIGUES — smplx's `batch_rodrigues` adds 1e-8 to every angle before normalising the
#               axis (a divide-by-zero guard). That is a deliberate bias in THEIR code, not
#               noise in ours, so comparing rotations against smplx gets this and only this.
ROUNDOFF = 1e-14
RODRIGUES = 1e-7

SMPL_DIR = Path("models/smpl")
requires_smpl = pytest.mark.skipif(
    not (SMPL_DIR / "SMPL_NEUTRAL.pkl").exists(),
    reason=f"SMPL model files not found in {SMPL_DIR}/ — see models/README.md",
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rot(axis: str, degrees: float) -> np.ndarray:
    aa = np.zeros(3)
    aa["xyz".index(axis)] = np.deg2rad(degrees)
    return aa_to_rotmat(aa)


def _aa(axis: str, degrees: float) -> np.ndarray:
    aa = np.zeros(3)
    aa["xyz".index(axis)] = np.deg2rad(degrees)
    return aa


def _random_tree(rng: np.random.Generator, n_joints: int) -> np.ndarray:
    """A random valid kinematic tree: root has parent -1, every parent precedes its child."""
    return np.array([-1] + [int(rng.integers(0, k)) for k in range(1, n_joints)])


def _random_skin_weights(rng: np.random.Generator, n_verts: int, n_joints: int) -> np.ndarray:
    """SMPL-like: each vertex influenced by up to 4 joints, rows summing to 1."""
    w = np.zeros((n_verts, n_joints))
    for v in range(n_verts):
        joints = rng.choice(n_joints, size=4, replace=False)
        w[v, joints] = rng.dirichlet(np.ones(4))
    return w


def _posed_scene(seed: int = 0, n_verts: int = 500, n_joints: int = 24):
    """A seeded, demonstrably non-trivial pose: (global_orient, body_pose, parents, skin_w, M)."""
    rng = _rng(seed)
    parents = _random_tree(rng, n_joints)
    skin_w = _random_skin_weights(rng, n_verts, n_joints)
    global_orient = rng.normal(0.0, 0.6, 3)
    body_pose = rng.normal(0.0, 0.6, 3 * (n_joints - 1))
    M = blend_matrices(world_rotations(global_orient, body_pose, parents), skin_w)
    return global_orient, body_pose, parents, skin_w, M


class TestWorldRotations:
    """Forward kinematics, rotation part only. `R_k` in `M_v = sum_k w_vk R_k` is a WORLD
    rotation: your forearm's orientation in the room depends on your shoulder too."""

    def test_a_chain_of_quarter_turns_accumulates(self):
        """Known answer. Three joints in a line, each turned +90 deg about z relative to
        its parent. In the world they point at 90, 180, 270 deg — not 90, 90, 90.

        A local rotation means nothing on its own; it is relative to the parent's frame.
        """
        parents = np.array([-1, 0, 1])
        global_orient = _aa("z", 90)
        body_pose = np.concatenate([_aa("z", 90), _aa("z", 90)])

        R = world_rotations(global_orient, body_pose, parents)

        assert R.shape == (3, 3, 3)
        assert R.dtype == np.float64
        np.testing.assert_allclose(R[0], _rot("z", 90), atol=ROUNDOFF)
        np.testing.assert_allclose(R[1], _rot("z", 180), atol=ROUNDOFF)
        np.testing.assert_allclose(R[2], _rot("z", 270), atol=ROUNDOFF)

    def test_composition_order_is_parent_then_local(self):
        """`R_child = R_parent @ R_local`, not the other way round.

        With every joint rotating about z (above), the order cannot show — z-rotations
        commute. Mixed axes do not: 90 x then 90 y is a different orientation from
        90 y then 90 x. The local rotation acts in the parent's frame, so it is applied
        first (rightmost) and the parent carries the result into the world.
        """
        parents = np.array([-1, 0])
        Rp, Rl = _rot("x", 90), _rot("y", 90)
        assert not np.allclose(Rp @ Rl, Rl @ Rp), "test is vacuous if these commute"

        R = world_rotations(rotmat_to_aa(Rp), rotmat_to_aa(Rl), parents)

        np.testing.assert_allclose(R[1], Rp @ Rl, atol=ROUNDOFF)

    def test_siblings_inherit_from_the_parent_not_from_each_other(self):
        """Branching tree: both 1 and 2 hang off 0. SMPL has three such branches at the
        spine (both hips) and three at the upper spine (neck, both collars). A loop that
        quietly assumes `parent = k - 1` passes every chain test and fails here.
        """
        parents = np.array([-1, 0, 0])
        global_orient = _aa("x", 30)
        body_pose = np.concatenate([_aa("y", 50), _aa("z", -70)])

        R = world_rotations(global_orient, body_pose, parents)

        np.testing.assert_allclose(R[1], _rot("x", 30) @ _rot("y", 50), atol=ROUNDOFF)
        np.testing.assert_allclose(R[2], _rot("x", 30) @ _rot("z", -70), atol=ROUNDOFF)

    def test_global_orient_rotates_every_joint(self):
        """Turning the whole subject by G turns every world rotation by G: `R_k -> G R_k`.

        This is the first half of D1's equivariance proof (the second is in
        TestEquivariance). Unlike the FEATURE (S1, D2), the FRAME must see global_orient.
        """
        rng = _rng(1)
        parents = _random_tree(rng, 24)
        global_orient = rng.normal(0.0, 0.6, 3)
        body_pose = rng.normal(0.0, 0.6, 69)
        G = _rot("y", 140) @ _rot("x", -35)

        turned = rotmat_to_aa(G @ aa_to_rotmat(global_orient))
        R = world_rotations(global_orient, body_pose, parents)
        R_turned = world_rotations(turned, body_pose, parents)

        np.testing.assert_allclose(R_turned, G @ R, atol=ROUNDOFF)

    def test_rejects_a_parent_listed_after_its_child(self):
        """The single forward pass is only correct if every parent's world rotation exists
        before its child asks for it. SMPL guarantees `parents[k] < k`; anything else
        handed in would silently read an unfilled (or stale) slot. Fail loudly instead.
        """
        parents = np.array([-1, 2, 0])  # joint 1's parent is joint 2, not yet built
        with pytest.raises(ValueError):
            world_rotations(np.zeros(3), np.zeros(6), parents)

    @requires_smpl
    def test_matches_smplx_forward_kinematics(self):
        """Independent reference: smplx's own `batch_rigid_transform`. The rotation block
        of each 4x4 it returns is exactly that joint's world rotation.

        Tolerance is RODRIGUES, not ROUNDOFF — see the vocabulary at the top.
        """
        torch = pytest.importorskip("torch")
        smplx = pytest.importorskip("smplx")
        from smplx.lbs import batch_rigid_transform, batch_rodrigues

        model = smplx.create("models", model_type="smpl", gender="neutral", dtype=torch.float64)
        parents = model.parents.numpy()
        rng = _rng(2)
        global_orient = rng.normal(0.0, 0.6, 3)
        body_pose = rng.normal(0.0, 0.6, 69)

        full = torch.tensor(np.concatenate([global_orient, body_pose]).reshape(24, 3))
        rot_mats = batch_rodrigues(full).reshape(1, 24, 3, 3)
        joints = (model.J_regressor @ model.v_template)[None]
        _, A = batch_rigid_transform(rot_mats, joints, model.parents, dtype=torch.float64)

        R = world_rotations(global_orient, body_pose, parents)

        np.testing.assert_allclose(R, A[0, :, :3, :3].numpy(), atol=RODRIGUES)


class TestBlendMatrices:
    """`M_v = sum_k w_vk R_k`. The whole reason D1 exists is that this is not a rotation."""

    def test_shape(self):
        rng = _rng(3)
        R = aa_to_rotmat(rng.normal(0.0, 0.6, (24, 3)))
        skin_w = _random_skin_weights(rng, 100, 24)

        M = blend_matrices(R, skin_w)

        assert M.shape == (100, 3, 3)
        assert M.dtype == np.float64

    def test_a_vertex_owned_by_one_joint_gets_exactly_that_rotation(self):
        """Mid-limb vertices are ~100% one joint. There `M_v` IS a rotation, so inverse and
        transpose agree — which is why the D1 bug is invisible mid-limb."""
        R = np.stack([_rot("x", 40), _rot("y", -65), _rot("z", 110)])
        skin_w = np.array([[0.0, 1.0, 0.0]])

        M = blend_matrices(R, skin_w)

        np.testing.assert_allclose(M[0], R[1], atol=ROUNDOFF)

    def test_a_vertex_shared_by_two_joints_is_not_a_rotation(self):
        """At a joint, a 50/50 blend of two different rotations is the average of two
        matrices — it shrinks (det < 1) and is no longer orthonormal (`M^T M != I`).

        This is the shear D1's true inverse removes and a transpose would apply twice:
        measured on real SMPL at a random pose, `M^T` instead of `M^-1` leaves 15mm of
        error on a 10mm delta, concentrated at the joints.
        """
        R = np.stack([np.eye(3), _rot("x", 90)])
        skin_w = np.array([[0.5, 0.5]])

        M = blend_matrices(R, skin_w)[0]

        assert np.linalg.det(M) < 0.99
        assert not np.allclose(M.T @ M, np.eye(3), atol=1e-3)
        np.testing.assert_allclose(M, 0.5 * (np.eye(3) + _rot("x", 90)), atol=ROUNDOFF)


class TestRoundTrip:
    """The plan's S2 'done when': invent delta, push forward, pull back, compare at 1e-14."""

    def test_the_pose_under_test_is_not_trivial(self):
        """Guard on the guard. If this ever fails, every round trip below became vacuous.

        Proves two things about the shared scene: `M_v` is far from identity, AND the
        posed residual differs from the rest delta by orders of magnitude more than the
        tolerance — so a `to_local` that just returns its input cannot pass.
        """
        *_, M = _posed_scene()
        delta = _rng(10).normal(0.0, 0.01, (M.shape[0], 3))
        D_posed = np.einsum("vij,vj->vi", M, delta)

        assert np.abs(M - np.eye(3)).max() > 0.5
        assert np.abs(D_posed - delta).max() > 1e6 * ROUNDOFF

    def test_round_trip_recovers_delta_to_roundoff(self):
        """THE gate. 10mm-scale seeded delta, random 24-joint pose, 500 vertices."""
        *_, M = _posed_scene()
        delta_true = _rng(11).normal(0.0, 0.01, (M.shape[0], 3))

        delta_hat = to_local(to_posed(delta_true, M), M)

        assert delta_hat.shape == delta_true.shape
        np.testing.assert_allclose(delta_hat, delta_true, rtol=0, atol=ROUNDOFF)

    def test_to_posed_is_skinning_applied_to_the_delta(self):
        """Pinned against an independently written answer, not just its inverse.

        A polar `R~` used in BOTH directions round-trips perfectly and is still wrong —
        this test, unlike the round trip, would catch it.
        """
        *_, M = _posed_scene()
        delta = _rng(12).normal(0.0, 0.01, (M.shape[0], 3))

        expected = np.einsum("vij,vj->vi", M, delta)

        np.testing.assert_allclose(to_posed(delta, M), expected, rtol=0, atol=ROUNDOFF)

    def test_to_local_inverts_skinning_not_its_transpose(self):
        """Same idea from the other side: feed `M @ delta`, demand `delta` back. A
        transpose or polar inverse fails by millimetres here, not by round-off."""
        *_, M = _posed_scene()
        delta = _rng(13).normal(0.0, 0.01, (M.shape[0], 3))
        D_posed = np.einsum("vij,vj->vi", M, delta)

        transpose_answer = np.einsum("vji,vj->vi", M, D_posed)
        assert np.abs(transpose_answer - delta).max() > 1e-3, "test is vacuous otherwise"

        np.testing.assert_allclose(to_local(D_posed, M), delta, rtol=0, atol=ROUNDOFF)

    def test_exactly_three_vertices(self):
        """The numpy trap. `np.linalg.solve(a, b)` with `a` (V,3,3) and `b` (V,3) treats
        `b` as a stack of MATRICES, not vectors. For V = 6890 that is a loud shape error.
        For V == 3 the shapes happen to line up, and it silently returns (3, 3, 3) of
        nonsense. Every toy example you are likely to hand-check has three vertices.
        """
        *_, M = _posed_scene(n_verts=3)
        delta = _rng(14).normal(0.0, 0.01, (3, 3))

        D_posed = to_posed(delta, M)
        delta_hat = to_local(D_posed, M)

        assert D_posed.shape == (3, 3)
        assert delta_hat.shape == (3, 3)
        np.testing.assert_allclose(delta_hat, delta, rtol=0, atol=ROUNDOFF)


class TestEquivariance:
    """D1's property that must survive: turning the subject around changes nothing.

    `R_k -> G R_k` for every k  =>  `M -> G M`, and the scan turns too: `D -> G D`.
    So `(G M)^-1 (G D) = M^-1 G^-1 G D = M^-1 D`. The rest-frame delta is identical.
    """

    def test_global_rotation_leaves_the_rest_frame_delta_unchanged(self):
        global_orient, body_pose, parents, skin_w, M = _posed_scene(seed=20)
        delta = _rng(21).normal(0.0, 0.01, (M.shape[0], 3))
        D_posed = np.einsum("vij,vj->vi", M, delta)

        G = _rot("z", -120) @ _rot("y", 75)
        turned = rotmat_to_aa(G @ aa_to_rotmat(global_orient))
        M_turned = blend_matrices(world_rotations(turned, body_pose, parents), skin_w)
        D_turned = D_posed @ G.T  # each row d -> G d

        np.testing.assert_allclose(
            to_local(D_turned, M_turned), to_local(D_posed, M), rtol=0, atol=ROUNDOFF
        )


class TestConditioning:
    """D1's guard. `M_v` degenerates only when co-influencing joints differ by ~180 deg:
    `0.5 (I + R_180)` has a zero singular value. No sane rig does that — so if it
    happens, something upstream is broken, and silently dividing by ~0 would hide it."""

    @staticmethod
    def _matrices_with_one_bad_vertex(bad_vertex: int, twist_deg: float) -> np.ndarray:
        rng = _rng(30)
        R = aa_to_rotmat(rng.normal(0.0, 0.3, (24, 3)))
        M = blend_matrices(R, _random_skin_weights(rng, 5000, 24))
        M[bad_vertex] = 0.5 * (np.eye(3) + _rot("z", twist_deg))
        return M

    def test_a_singular_vertex_fails_loudly_and_names_itself(self):
        """The message must contain the vertex index: across 6890 vertices, that index is
        the only way to find which skinning weights or which joint went wrong."""
        M = self._matrices_with_one_bad_vertex(4321, 180.0)
        D_posed = np.zeros((M.shape[0], 3))

        with pytest.raises(ValueError, match=r"\b4321\b"):
            to_local(D_posed, M)

    def test_threshold_sits_at_1e6(self):
        """`cond(0.5 (I + R_(180 - e))) ~= 2 / e` (e in radians). Pins D1's 1e6 from both
        sides: cond ~2e5 must pass, cond ~2e6 must raise."""
        D_posed = np.zeros((5000, 3))

        e_ok = 1e-5  # cond ~ 2e5
        to_local(D_posed, self._matrices_with_one_bad_vertex(4321, 180.0 - np.rad2deg(e_ok)))

        e_bad = 1e-6  # cond ~ 2e6
        with pytest.raises(ValueError, match=r"\b4321\b"):
            to_local(D_posed, self._matrices_with_one_bad_vertex(4321, 180.0 - np.rad2deg(e_bad)))


@pytest.fixture(scope="module")
def smpl():
    torch = pytest.importorskip("torch")
    smplx = pytest.importorskip("smplx")
    return smplx.create("models", model_type="smpl", gender="neutral", dtype=torch.float64)


@requires_smpl
class TestAgainstRealSMPL:
    """The consumer convention, checked against the consumer's actual arithmetic."""

    def test_recovers_delta_from_what_a_skinning_runtime_produces(self, smpl):
        """D1's central claim, end to end. A runtime (Maya, smplx) adds delta to the
        template BEFORE skinning and keeps the joints where the rig put them:

            V = T_v (p_v + delta_v) + t_v     =>     D = V(delta) - V(0) = M_v delta_v

        Built here from smplx's own transforms `A` — none of your code produces D. Your
        `to_local` must recover delta to round-off. Measured: 3.5e-16.

        Joints FIXED, note. Adding delta to `v_template` and re-running `smpl(...)` would
        re-regress the joints from the displaced template and move them (~15mm here),
        which no rig does. Any forward evaluator (D6) must hold joints fixed too.
        """
        import torch
        from smplx.lbs import batch_rigid_transform, batch_rodrigues

        rng = _rng(40)
        full = np.concatenate([rng.normal(0.0, 0.6, 3), rng.normal(0.0, 0.6, 69)])
        rot_mats = batch_rodrigues(torch.tensor(full.reshape(24, 3))).reshape(1, 24, 3, 3)
        joints = (smpl.J_regressor @ smpl.v_template)[None]
        _, A = batch_rigid_transform(rot_mats, joints, smpl.parents, dtype=torch.float64)
        A = A[0].numpy()
        skin_w = smpl.lbs_weights.numpy()

        T = np.einsum("vk,kij->vij", skin_w, A)  # per-vertex 4x4, exactly as smplx's lbs

        def skin(v: np.ndarray) -> np.ndarray:
            return np.einsum("vij,vj->vi", T[:, :3, :3], v) + T[:, :3, 3]

        rest = smpl.v_template.numpy()
        delta = rng.normal(0.0, 0.01, rest.shape)
        D_posed = skin(rest + delta) - skin(rest)

        M = blend_matrices(A[:, :3, :3], skin_w)

        np.testing.assert_allclose(to_local(D_posed, M), delta, rtol=0, atol=ROUNDOFF)

    def test_conditioning_stays_finite_across_a_pose_sweep(self, smpl):
        """D1 says real skinning weights never approach the singular case. Prove it on the
        real weights over 200 random poses, including implausibly large ones. Measured:
        worst cond ~2 at 0.6 rad. If this ever trips, the guard is doing its job."""
        rng = _rng(41)
        parents = smpl.parents.numpy()
        skin_w = smpl.lbs_weights.numpy()

        worst = 0.0
        for scale in (0.3, 0.6, 1.5):
            for _ in range(200 // 3 + 1):
                go, bp = rng.normal(0.0, scale, 3), rng.normal(0.0, scale, 69)
                M = blend_matrices(world_rotations(go, bp, parents), skin_w)
                worst = max(worst, float(np.linalg.cond(M).max()))

        assert np.isfinite(worst)
        assert worst < 1e6, f"worst cond(M_v) = {worst:.3e}"
