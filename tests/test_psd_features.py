"""S1 — pose-space features `x(θ)`.

These tests ARE the spec for `scantosmpl.psd.features.pose_features`. They are written
to be read before the implementation: each one states a property and the reason it
matters downstream, so the interface can be derived from them rather than copied.

Decisions under test: D2 (23 joints, no global_orient), D3 (`R_k - I` encoding).
Reference: docs/psd-implementation-plan.md, docs/psd-primer.md §4.
"""

import inspect
import re

import numpy as np
import pytest

from scantosmpl.psd.features import pose_features
from scantosmpl.utils.geometry import aa_to_rotmat

# SMPL body_pose is 23 joints x 3 axis-angle numbers. global_orient is a SEPARATE (3,).
N_BODY_JOINTS = 23
BODY_POSE_DIM = N_BODY_JOINTS * 3  # 69
# 9 numbers per joint because the encoding is a flattened 3x3 (D3), not an angle.
EXPECTED_FEATURE_DIM = N_BODY_JOINTS * 9  # 207

# Tolerance vocabulary used below, kept deliberately distinct:
#   ROUNDOFF  — "these should be the same number"; only float64 noise is allowed.
#   Nothing in this module gets a *physical* tolerance. If a test here needs one,
#   the property being tested is the wrong property.
ROUNDOFF = 1e-13


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_pose(rng: np.random.Generator, scale: float = 0.6) -> np.ndarray:
    """A plausible-magnitude random body pose. `scale` is radians, roughly."""
    return rng.normal(0.0, scale, size=BODY_POSE_DIM)


class TestContract:
    """Shapes and dtype. Boring, and the reason S2's 1e-14 gate is reachable."""

    def test_single_pose_in_single_feature_vector_out(self):
        feats = pose_features(np.zeros(BODY_POSE_DIM))
        assert feats.shape == (EXPECTED_FEATURE_DIM,)

    def test_feature_dim_is_207_not_216(self):
        """D2. 216 would mean global_orient got encoded as a 24th joint.

        That is not a size quibble: a feature that moves when the subject turns
        around makes the SAME arm bend predict DIFFERENT flesh depending on which
        way they face, which destroys the frame-invariance the whole local-frame
        construction (spec §4.2) exists to establish.
        """
        feats = pose_features(_random_pose(_rng()))
        assert feats.shape[-1] == 207
        assert feats.shape[-1] != 216

    def test_output_is_float64(self):
        """S2's round-trip gate is 1e-14. float32 resolution at metre scale is
        ~1.2e-7, so a float32 feature path would make that gate unreachable and
        we would loosen it to a tolerance instead of fixing the cause.
        """
        feats = pose_features(_random_pose(_rng()))
        assert feats.dtype == np.float64

    def test_accepts_a_batch_of_poses(self):
        """S3 builds `X` as (N, 207) for the whole corpus in one call.

        Supporting a leading batch dim here is cheaper than looping there, and it
        keeps the corpus assembly free of reshape logic that could silently
        transpose the joint layout.
        """
        rng = _rng(1)
        batch = np.stack([_random_pose(rng) for _ in range(5)])
        assert batch.shape == (5, BODY_POSE_DIM)

        feats = pose_features(batch)
        assert feats.shape == (5, EXPECTED_FEATURE_DIM)

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (72,),  # global_orient + body_pose concatenated: the D2 leak, as a shape
            (66,),  # one joint short
            (5, 72),  # same leak, batched
            (23, 3),  # right numbers, wrong layout: trailing axis must be the 69
        ],
    )
    def test_rejects_anything_but_69_on_the_last_axis(self, bad_shape):
        """D2 enforced by the shape, not just the signature.

        `reshape(-1, 3)` accepts any multiple of 3, so a (72,) input — the most
        likely real mistake, concatenating global_orient onto body_pose — would
        silently become 24 "joints" and a 216-D feature. Nothing downstream
        would complain until Phi quietly started depending on facing direction.
        Fail at the boundary, loudly, with a ValueError.

        The message must name the shape it actually received: when this fires
        during corpus assembly (S3), that shape is the only clue to which
        upstream step concatenated or truncated the pose.
        """
        with pytest.raises(ValueError, match=re.escape(str(bad_shape))):
            pose_features(np.zeros(bad_shape))

    def test_batched_rows_match_individual_calls(self):
        """A batch must be exactly a stack of singles — no broadcasting surprises."""
        rng = _rng(2)
        batch = np.stack([_random_pose(rng) for _ in range(4)])

        batched = pose_features(batch)
        for i, row in enumerate(batch):
            np.testing.assert_allclose(batched[i], pose_features(row), atol=ROUNDOFF)


class TestRestPose:
    """The property that makes a predicted delta at rest structurally zero."""

    def test_rest_pose_is_exactly_zero(self):
        """EXACTLY, not allclose. `R(0) = I`, so `R - I` is the zero matrix — the
        subtraction is algebraic, not a near-cancellation, so there is no float
        error to absorb. If this needs a tolerance, the encoding is not `R - I`.

        Downstream consequence: `x = 0` at rest means a model trained on any delta
        predicts zero correction at rest by construction. Not approximately.
        """
        feats = pose_features(np.zeros(BODY_POSE_DIM))
        assert np.all(feats == 0.0), f"max |x| at rest = {np.abs(feats).max():.3e}"

    def test_only_the_moved_joint_contributes(self):
        """Locality. Rotating joint j must leave every other joint's block exactly
        zero — otherwise the feature mixes joints and `||x_i - x_j||` stops meaning
        "how differently posed" (which is the only thing the kernel consumes).
        """
        pose = np.zeros(BODY_POSE_DIM)
        pose[3 * 7 : 3 * 7 + 3] = [0.4, -0.2, 0.1]  # joint 7 only

        feats = pose_features(pose)
        moved = feats[9 * 7 : 9 * 7 + 9]
        others = np.delete(feats, np.arange(9 * 7, 9 * 7 + 9))

        assert np.any(moved != 0.0)
        assert np.all(others == 0.0)


class TestKnownAnswer:
    """The test that proves the encoding is `R - I` and pins the memory layout.

    Every other test here would also pass for a *different* correct-looking
    encoding. This one would not.
    """

    def test_ninety_degrees_about_x_gives_the_exact_expected_block(self):
        """+90 deg about x at the first body joint.

            R = [[1, 0,  0],        R - I = [[0,  0,  0],
                 [0, 0, -1],                 [0, -1, -1],
                 [0, 1,  0]]                 [0,  1, -1]]

        Row-major flatten: [0, 0, 0, 0, -1, -1, 0, 1, -1].

        Row-major is pinned (not merely conventional) because it is what
        `aa_to_rotmat(...).reshape(-1)` produces and what SMPL's own pose
        blendshapes use — so a shared layout means our feature and SMPL's
        `pose_feature` stay interchangeable.
        """
        pose = np.zeros(BODY_POSE_DIM)
        pose[0] = np.pi / 2

        feats = pose_features(pose)
        expected = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0])

        np.testing.assert_allclose(feats[:9], expected, atol=ROUNDOFF)
        assert np.all(feats[9:] == 0.0)

    def test_matches_an_independent_reference_implementation(self):
        """Written from the definition, deliberately not sharing code with the
        implementation. Catches joint-ordering and flatten-order mistakes that the
        single-joint case above cannot, because here every joint is non-zero.
        """
        pose = _random_pose(_rng(3))
        reference = (aa_to_rotmat(pose.reshape(N_BODY_JOINTS, 3)) - np.eye(3)).reshape(-1)

        np.testing.assert_allclose(pose_features(pose), reference, atol=ROUNDOFF)


class TestEncodingInvariants:
    """The properties that motivated rejecting axis-angle and quaternions (D3)."""

    @pytest.mark.parametrize("turns", [1, 2, 3])
    def test_full_turns_do_not_move_the_feature(self, turns):
        """THE argument against axis-angle. Adding 2*pi along a joint's rotation
        axis is the same physical pose, so the feature must not move — otherwise
        `||x_i - x_j||` reports two identical poses as far apart and the kernel
        interpolates between a pose and itself.

        Verified reachable: max |diff| is 2.2e-16 at one turn, 1.4e-15 at three.
        """
        rng = _rng(4)
        pose = _random_pose(rng)

        wrapped = pose.reshape(N_BODY_JOINTS, 3).copy()
        for j in range(N_BODY_JOINTS):
            norm = np.linalg.norm(wrapped[j])
            wrapped[j] *= 1.0 + turns * 2.0 * np.pi / norm

        np.testing.assert_allclose(
            pose_features(wrapped.reshape(-1)),
            pose_features(pose),
            atol=ROUNDOFF,
        )

    def test_global_orient_cannot_change_the_feature(self):
        """D2, tested two ways depending on the interface you chose.

        If your signature does not accept `global_orient` at all, the leak is
        unrepresentable and this test skips — that is the STRONGER design, and the
        skip is the pass. If you did accept it, it must be ignored.
        """
        params = inspect.signature(pose_features).parameters
        if "global_orient" not in params:
            pytest.skip("signature excludes global_orient — leak unrepresentable (preferred)")

        pose = _random_pose(_rng(5))
        facing_north = pose_features(pose, global_orient=np.zeros(3))
        facing_other = pose_features(pose, global_orient=np.array([0.0, 2.1, 0.0]))

        np.testing.assert_allclose(facing_north, facing_other, atol=ROUNDOFF)

    def test_finite_for_a_thousand_random_poses(self):
        """No NaN/inf anywhere, including implausibly large angles. A single NaN
        here poisons an entire row of Phi, and `np.linalg.solve` will return
        garbage rather than raising.
        """
        rng = _rng(6)
        for scale in (0.1, 1.0, 4.0 * np.pi):
            batch = np.stack([_random_pose(rng, scale) for _ in range(1000)])
            feats = pose_features(batch)
            assert np.all(np.isfinite(feats))

    def test_feature_distance_tracks_angular_distance(self):
        """Continuity, stated as the thing the kernel actually needs.

        `Phi_ij = exp(-||x_i - x_j||^2 / 2*sigma^2)` is only meaningful if a small
        pose change gives a small distance. Measured on a single joint:
        1 deg -> 0.0247, 10 deg -> 0.2465, 90 deg -> 2.0000 — monotonic, and 1 deg
        is ~1/80 of 90 deg.
        """
        rest = pose_features(np.zeros(BODY_POSE_DIM))

        def distance_at(degrees: float) -> float:
            pose = np.zeros(BODY_POSE_DIM)
            pose[0] = np.deg2rad(degrees)
            return float(np.linalg.norm(pose_features(pose) - rest))

        d1, d10, d90 = distance_at(1.0), distance_at(10.0), distance_at(90.0)

        assert d1 < d10 < d90
        assert d1 < d90 / 10.0, "a 1 deg change must not read as comparable to 90 deg"
