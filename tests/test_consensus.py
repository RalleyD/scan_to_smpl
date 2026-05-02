"""Unit tests for Phase 3 consensus — SO(3) operations, Procrustes, aggregation.

No GPU or checkpoints required.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scantosmpl.utils.geometry import (
    aa_to_rotmat,
    compute_pa_mpjpe,
    frechet_mean_so3,
    procrustes_align,
    rotmat_to_aa,
    so3_exp,
    so3_log,
)


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------


class TestRotationConversions:
    def test_aa_to_rotmat_identity(self):
        aa = np.zeros((1, 3))
        R = aa_to_rotmat(aa)
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)

    def test_aa_to_rotmat_roundtrip(self):
        rng = np.random.default_rng(42)
        aa = rng.standard_normal((10, 3)) * 0.5
        R = aa_to_rotmat(aa)
        aa_back = rotmat_to_aa(R)
        # Axis-angle has a sign ambiguity for ||aa|| = pi, but for small angles this is fine
        np.testing.assert_allclose(aa_back, aa, atol=1e-6)

    def test_rotmat_to_aa_batch(self):
        rng = np.random.default_rng(7)
        R = Rotation.random(5, random_state=rng).as_matrix()
        aa = rotmat_to_aa(R)
        assert aa.shape == (5, 3)

    def test_aa_to_rotmat_broadcast_shape(self):
        """Test that batch dimensions are preserved."""
        aa = np.zeros((2, 3, 3))  # batch of 2x3
        R = aa_to_rotmat(aa)
        assert R.shape == (2, 3, 3, 3)


class TestSO3LogExp:
    def test_log_exp_roundtrip_identity(self):
        R = np.eye(3)
        v = so3_log(R)
        R_back = so3_exp(v)
        np.testing.assert_allclose(R_back, R, atol=1e-10)

    def test_log_exp_roundtrip_random(self):
        rng = np.random.default_rng(123)
        R = Rotation.random(random_state=rng).as_matrix()
        v = so3_log(R)
        R_back = so3_exp(v)
        np.testing.assert_allclose(R_back, R, atol=1e-10)

    def test_exp_log_roundtrip(self):
        v = np.array([0.1, -0.2, 0.3])
        R = so3_exp(v)
        v_back = so3_log(R)
        np.testing.assert_allclose(v_back, v, atol=1e-10)


# ---------------------------------------------------------------------------
# SO(3) Frechet mean
# ---------------------------------------------------------------------------


class TestFrechetMeanSO3:
    def test_identity_inputs_give_identity(self):
        rotations = np.stack([np.eye(3)] * 5, axis=0)
        mean = frechet_mean_so3(rotations)
        np.testing.assert_allclose(mean, np.eye(3), atol=1e-8)

    def test_single_rotation_returns_itself(self):
        R = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        mean = frechet_mean_so3(R[np.newaxis])
        np.testing.assert_allclose(mean, R, atol=1e-10)

    def test_two_rotations_geodesic_midpoint(self):
        """Mean of R and R^{-1} should be close to identity."""
        R = Rotation.from_rotvec([0.3, 0.0, 0.0]).as_matrix()
        R_inv = R.T
        rotations = np.stack([R, R_inv], axis=0)
        mean = frechet_mean_so3(rotations)
        np.testing.assert_allclose(mean, np.eye(3), atol=1e-6)

    def test_uniform_weights_match_unweighted(self):
        rng = np.random.default_rng(99)
        rotations = Rotation.random(6, random_state=rng).as_matrix()
        mean_unweighted = frechet_mean_so3(rotations)
        mean_weighted = frechet_mean_so3(rotations, weights=np.ones(6))
        np.testing.assert_allclose(mean_unweighted, mean_weighted, atol=1e-8)

    def test_convergence_with_large_rotations(self):
        """Even with spread-out rotations, should still converge."""
        rng = np.random.default_rng(55)
        rotations = Rotation.random(10, random_state=rng).as_matrix()
        mean = frechet_mean_so3(rotations, max_iter=100)
        # Should be a valid rotation matrix
        assert mean.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(mean), 1.0, atol=1e-8)
        np.testing.assert_allclose(mean.T @ mean, np.eye(3), atol=1e-8)

    def test_output_is_valid_rotation_matrix(self):
        rng = np.random.default_rng(11)
        rotations = Rotation.random(4, random_state=rng).as_matrix()
        mean = frechet_mean_so3(rotations)
        # det = 1
        np.testing.assert_allclose(np.linalg.det(mean), 1.0, atol=1e-8)
        # orthogonal
        np.testing.assert_allclose(mean.T @ mean, np.eye(3), atol=1e-8)

    def test_weighted_shifts_toward_high_weight(self):
        """Heavy weight on R1 should pull the mean closer to R1."""
        R1 = Rotation.from_rotvec([0.5, 0.0, 0.0]).as_matrix()
        R2 = Rotation.from_rotvec([-0.5, 0.0, 0.0]).as_matrix()
        rotations = np.stack([R1, R2], axis=0)

        # Heavy weight on R1
        mean = frechet_mean_so3(rotations, weights=np.array([10.0, 1.0]))
        # Mean should be closer to R1 than to R2
        dist_to_R1 = np.linalg.norm(so3_log(mean.T @ R1))
        dist_to_R2 = np.linalg.norm(so3_log(mean.T @ R2))
        assert dist_to_R1 < dist_to_R2

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            frechet_mean_so3(np.zeros((0, 3, 3)))


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------


class TestProcrustesAlign:
    def test_identity_alignment(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        aligned, scale = procrustes_align(pts, pts)
        np.testing.assert_allclose(aligned, pts, atol=1e-8)
        np.testing.assert_allclose(scale, 1.0, atol=1e-6)

    def test_scaled_input_recovers_scale(self):
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((10, 3))
        target = pts * 2.5
        aligned, scale = procrustes_align(pts, target)
        np.testing.assert_allclose(scale, 2.5, atol=1e-4)
        np.testing.assert_allclose(aligned, target, atol=1e-4)

    def test_translated_input_recovers_translation(self):
        rng = np.random.default_rng(7)
        pts = rng.standard_normal((10, 3))
        target = pts + np.array([5.0, -3.0, 2.0])
        aligned, scale = procrustes_align(pts, target)
        np.testing.assert_allclose(aligned, target, atol=1e-8)

    def test_rotated_input(self):
        rng = np.random.default_rng(13)
        pts = rng.standard_normal((10, 3))
        R = Rotation.from_rotvec([0.5, 0.3, -0.2]).as_matrix()
        target = (R @ pts.T).T
        aligned, _ = procrustes_align(pts, target)
        np.testing.assert_allclose(aligned, target, atol=1e-6)


class TestPAMPJPE:
    def test_identical_gives_zero(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        assert compute_pa_mpjpe(pts, pts) < 1e-10

    def test_scaled_gives_zero(self):
        """PA-MPJPE removes scale, so scaled copies should give ~0."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((10, 3))
        pa = compute_pa_mpjpe(pts, pts * 3.0)
        assert pa < 1e-6

    def test_noisy_gives_nonzero(self):
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((10, 3))
        noisy = pts + rng.standard_normal((10, 3)) * 0.01
        pa = compute_pa_mpjpe(pts, noisy)
        assert pa > 0.0
        assert pa < 0.05  # noise was small


# ---------------------------------------------------------------------------
# Beta aggregation (test the logic without needing ConsensusBuilder)
# ---------------------------------------------------------------------------


class TestBetaAggregation:
    """Test the trimmed weighted mean logic used for betas."""

    def _trimmed_mean(self, values_list, weights, trim=0.1):
        """Replicate the logic from ConsensusBuilder._aggregate_betas."""
        arr = np.stack(values_list, axis=0)
        N = arr.shape[0]
        n_trim = max(1, int(round(N * trim)))
        result = np.zeros(arr.shape[1])
        for c in range(arr.shape[1]):
            vals = arr[:, c]
            w = weights.copy()
            order = np.argsort(vals)
            if N > 2 * n_trim + 1:
                keep = order[n_trim:-n_trim]
            else:
                keep = order
            w_kept = w[keep]
            w_sum = w_kept.sum()
            if w_sum > 1e-12:
                result[c] = np.sum(w_kept * vals[keep]) / w_sum
            else:
                result[c] = np.median(vals[keep])
        return result

    def test_identical_inputs_return_same(self):
        betas = [np.ones(10) * 0.5] * 5
        weights = np.ones(5) / 5
        result = self._trimmed_mean(betas, weights)
        np.testing.assert_allclose(result, np.ones(10) * 0.5, atol=1e-10)

    def test_outlier_trimmed(self):
        """One extreme outlier should be trimmed."""
        betas = [np.zeros(10)] * 9 + [np.ones(10) * 100.0]
        weights = np.ones(10) / 10
        result = self._trimmed_mean(betas, weights)
        # The outlier (100.0) should be trimmed; result should be near 0
        assert np.all(np.abs(result) < 1.0)

    def test_weighted_shifts_toward_high_weight(self):
        b1 = np.array([1.0] + [0.0] * 9)
        b2 = np.array([0.0] + [0.0] * 9)
        betas = [b1, b2, b2, b2, b2]
        weights = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
        weights /= weights.sum()
        result = self._trimmed_mean(betas, weights)
        # With high weight on b1, component 0 should be pulled toward 1.0
        assert result[0] > 0.5

    def test_output_shape_10(self):
        betas = [np.random.randn(10) for _ in range(6)]
        weights = np.ones(6) / 6
        result = self._trimmed_mean(betas, weights)
        assert result.shape == (10,)


# ---------------------------------------------------------------------------
# Body pose aggregation
# ---------------------------------------------------------------------------


class TestBodyPoseAggregation:
    """Test SO(3) per-joint aggregation logic."""

    def test_identical_t_poses_return_same(self):
        """All views have the same pose -> consensus = that pose."""
        pose = np.random.default_rng(42).standard_normal(69) * 0.1
        poses = [pose.copy() for _ in range(5)]
        weights = np.ones(5) / 5

        N = len(poses)
        poses_arr = np.stack(poses, axis=0)
        consensus = np.zeros(69)
        for j in range(23):
            joint_aa = poses_arr[:, j*3:(j+1)*3]
            joint_rotmats = aa_to_rotmat(joint_aa)
            mean_rot = frechet_mean_so3(joint_rotmats, weights=weights)
            mean_aa = rotmat_to_aa(mean_rot.reshape(1, 3, 3))[0]
            consensus[j*3:(j+1)*3] = mean_aa

        np.testing.assert_allclose(consensus, pose, atol=1e-5)

    def test_near_zero_rotations_near_zero_output(self):
        """T-pose with tiny random perturbations -> consensus near zero."""
        rng = np.random.default_rng(7)
        poses = [rng.standard_normal(69) * 0.01 for _ in range(10)]
        weights = np.ones(10) / 10

        poses_arr = np.stack(poses, axis=0)
        consensus = np.zeros(69)
        for j in range(23):
            joint_aa = poses_arr[:, j*3:(j+1)*3]
            joint_rotmats = aa_to_rotmat(joint_aa)
            mean_rot = frechet_mean_so3(joint_rotmats, weights=weights)
            mean_aa = rotmat_to_aa(mean_rot.reshape(1, 3, 3))[0]
            consensus[j*3:(j+1)*3] = mean_aa

        assert np.linalg.norm(consensus) < 0.1  # near zero

    def test_output_shape_69(self):
        rng = np.random.default_rng(13)
        poses = [rng.standard_normal(69) * 0.2 for _ in range(4)]
        weights = np.ones(4) / 4

        poses_arr = np.stack(poses, axis=0)
        consensus = np.zeros(69)
        for j in range(23):
            joint_aa = poses_arr[:, j*3:(j+1)*3]
            joint_rotmats = aa_to_rotmat(joint_aa)
            mean_rot = frechet_mean_so3(joint_rotmats, weights=weights)
            mean_aa = rotmat_to_aa(mean_rot.reshape(1, 3, 3))[0]
            consensus[j*3:(j+1)*3] = mean_aa

        assert consensus.shape == (69,)
