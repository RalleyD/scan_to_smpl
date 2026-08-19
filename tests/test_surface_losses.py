"""Tests for SMPL+D and the Tier 3 differentiable surface losses.

Covers:
  * the displacement frame identity (master AC16 / D4) and the backward
    compatibility of adding a `displacements` parameter to `SMPLModel`,
  * the chunked bidirectional chamfer loss,
  * normal consistency, the uniform Laplacian and `D` regularisation.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from scantosmpl.fitting.surface_losses import (
    build_uniform_laplacian,
    chamfer_loss,
    displacement_regularisation,
    laplacian_smoothing_loss,
    normal_consistency_loss,
)
from scantosmpl.smpl.model import SMPLModel

SMPL_DIR = "models/smpl"


def _smpl_available() -> bool:
    return (Path(SMPL_DIR) / "SMPL_NEUTRAL.pkl").exists()


requires_smpl = pytest.mark.skipif(
    not _smpl_available(),
    reason=f"SMPL model files not found in {SMPL_DIR}/ — see models/README.md",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# SMPL+D — frame identity + backward compatibility
# ---------------------------------------------------------------------------


@requires_smpl
class TestDisplacements:
    """`D` lives in the posed world frame and never disturbs a Tier 1/2 caller."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = SMPLModel(model_dir=SMPL_DIR, gender="neutral", device=DEVICE)
        # A non-neutral pose, so the identity is exercised away from the rest pose.
        gen = torch.Generator(device="cpu").manual_seed(0)
        self.model.set_params(
            betas=torch.randn(1, 10, generator=gen).to(self.model.device) * 0.5,
            body_pose=torch.randn(1, self.model.body_model.NUM_BODY_JOINTS * 3, generator=gen).to(
                self.model.device
            )
            * 0.05,
            global_orient=torch.tensor([[0.1, 0.2, -0.3]]).to(self.model.device),
            translation=torch.tensor([[0.3, -0.2, 1.5]]).to(self.model.device),
            scale=torch.tensor([1.07]).to(self.model.device),
        )

    def _random_displacements(self, seed: int = 1, magnitude_m: float = 0.01) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        d = torch.randn(1, SMPLModel.NUM_VERTICES, 3, generator=gen) * magnitude_m
        return d.to(self.model.device)

    def test_displacement_frame_identity(self):
        """AC16: D == forward().vertices - forward(apply_displacements=False).vertices."""
        d = self._random_displacements()
        self.model.set_params(displacements=d)

        with torch.no_grad():
            v_with = self.model().vertices
            v_without = self.model(apply_displacements=False).vertices

        assert torch.allclose(v_without + d, v_with, atol=1e-6)
        # And the difference recovers D itself (the PSD-facing statement).
        assert torch.allclose(v_with - v_without, d, atol=1e-6)

    def test_displacement_kwarg_matches_parameter(self):
        """Passing D explicitly equals storing it on the parameter."""
        d = self._random_displacements(seed=2)
        with torch.no_grad():
            v_kwarg = self.model(displacements=d).vertices
            self.model.set_params(displacements=d)
            v_param = self.model().vertices
        assert torch.equal(v_kwarg, v_param)

    def test_displacement_is_post_scale(self):
        """D is metres in the FINAL posed world frame — scale must not rescale it."""
        d = self._random_displacements(seed=3)
        self.model.set_params(displacements=d)

        with torch.no_grad():
            base = self.model(apply_displacements=False).vertices
            with_d = self.model().vertices
            # Double the scale: the D contribution must be unchanged.
            scale2 = torch.tensor([2.14], device=self.model.device)
            base2 = self.model(scale=scale2, apply_displacements=False).vertices
            with_d2 = self.model(scale=scale2).vertices

        assert torch.allclose(with_d - base, with_d2 - base2, atol=1e-6)

    def test_zero_displacement_is_noop(self):
        """Default (zero) D leaves forward() bit-identical to the D=0 baseline."""
        with torch.no_grad():
            v_default = self.model().vertices
            v_baseline = self.model(apply_displacements=False).vertices
        assert torch.equal(v_default, v_baseline)

        # And identical to a freshly-constructed model given the same params.
        fresh = SMPLModel(model_dir=SMPL_DIR, gender="neutral", device=DEVICE)
        params = self.model.get_params_dict()
        fresh.set_params(
            betas=params["betas"],
            body_pose=params["body_pose"],
            global_orient=params["global_orient"],
            translation=params["translation"],
            scale=params["scale"],
        )
        with torch.no_grad():
            v_fresh = fresh().vertices
        assert torch.equal(v_default, v_fresh)
        assert torch.count_nonzero(fresh.displacements) == 0

    def test_joints_undisplaced(self):
        """D is a surface quantity: joints are identical with and without it."""
        d = self._random_displacements(seed=4, magnitude_m=0.05)
        with torch.no_grad():
            j_without = self.model(apply_displacements=False).joints
            j_with = self.model(displacements=d).joints
        assert torch.equal(j_without, j_with)

    def test_displacement_grad_flows(self):
        """loss.backward() populates model.displacements.grad."""
        self.model.zero_grad(set_to_none=True)
        output = self.model()
        loss = output.vertices.sum()
        loss.backward()

        assert self.model.displacements.grad is not None
        assert self.model.displacements.grad.shape == (1, SMPLModel.NUM_VERTICES, 3)
        # d(sum V)/dD == 1 everywhere.
        assert torch.allclose(
            self.model.displacements.grad,
            torch.ones_like(self.model.displacements),
            atol=1e-6,
        )

    def test_params_dict_backward_compatible(self):
        """get_params_dict() keeps every legacy key and gains 'displacements'."""
        params = self.model.get_params_dict()
        for key in ("betas", "body_pose", "global_orient", "translation", "scale"):
            assert key in params
        assert params["displacements"].shape == (1, SMPLModel.NUM_VERTICES, 3)

        d = self._random_displacements(seed=5)
        self.model.set_params(displacements=d)
        assert torch.equal(self.model.get_params_dict()["displacements"], d)

    def test_batch_forward_broadcasts_displacements(self):
        """A stored (1, V, 3) D broadcasts across a batched forward."""
        d = self._random_displacements(seed=6)
        self.model.set_params(displacements=d)
        batch = 3
        with torch.no_grad():
            out = self.model(
                betas=torch.zeros(batch, 10, device=self.model.device),
                body_pose=torch.zeros(
                    batch, self.model.body_model.NUM_BODY_JOINTS * 3, device=self.model.device
                ),
                global_orient=torch.zeros(batch, 3, device=self.model.device),
                translation=torch.zeros(batch, 3, device=self.model.device),
                scale=torch.ones(batch, device=self.model.device),
            )
            out_base = self.model(
                betas=torch.zeros(batch, 10, device=self.model.device),
                body_pose=torch.zeros(
                    batch, self.model.body_model.NUM_BODY_JOINTS * 3, device=self.model.device
                ),
                global_orient=torch.zeros(batch, 3, device=self.model.device),
                translation=torch.zeros(batch, 3, device=self.model.device),
                scale=torch.ones(batch, device=self.model.device),
                apply_displacements=False,
            )
        assert out.vertices.shape == (batch, SMPLModel.NUM_VERTICES, 3)
        assert torch.allclose(out.vertices - out_base.vertices, d.expand(batch, -1, -1), atol=1e-6)


# ---------------------------------------------------------------------------
# chamfer_loss — bidirectional, from ONE chunked torch.cdist (master D3)
# ---------------------------------------------------------------------------


def _well_separated_points(n: int, spacing: float = 1.0) -> torch.Tensor:
    """`n` grid points spaced `spacing` apart on a lattice.

    Far enough apart that a small perturbation cannot make a different point
    the nearest neighbour, which is what makes the known-answer test exact.
    """
    side = int(np.ceil(n ** (1 / 3))) + 1
    xs, ys, zs = np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)[:n].astype(np.float32)
    return torch.from_numpy(pts * spacing)


class TestChamferLoss:
    """`chamfer_loss` combines both directions from a single chunked cdist (D3)."""

    def test_known_answer_constant_shift(self):
        """cloud = mesh vertices shifted by constant d => both directions == |d|.

        Diagnostics are the untrimmed, unweighted mean distance, so they are
        exact regardless of huber_delta/trim_quantile.
        """
        verts = _well_separated_points(64, spacing=1.0)
        d = torch.tensor([0.01, -0.005, 0.002])
        cloud = verts + d
        _, diag = chamfer_loss(verts, cloud, huber_delta=0.02, trim_quantile=1.0)
        expected = float(d.norm())
        assert diag["mesh_to_cloud_m"] == pytest.approx(expected, abs=1e-4)
        assert diag["cloud_to_mesh_m"] == pytest.approx(expected, abs=1e-4)

    def test_chunking_invariance(self):
        """chunk_size in {1000, 10000, N} give the same loss to 1e-6 (D3)."""
        torch.manual_seed(0)
        verts = torch.rand(200, 3)
        cloud = torch.rand(2500, 3)

        chunk_sizes = (1000, 10_000, cloud.shape[0])
        results = [chamfer_loss(verts, cloud, chunk_size=cs) for cs in chunk_sizes]
        losses = [float(loss) for loss, _ in results]
        diags = [diag for _, diag in results]

        assert losses[0] == pytest.approx(losses[1], abs=1e-6)
        assert losses[0] == pytest.approx(losses[2], abs=1e-6)
        for key in ("mesh_to_cloud_m", "cloud_to_mesh_m"):
            assert diags[0][key] == pytest.approx(diags[1][key], abs=1e-6)
            assert diags[0][key] == pytest.approx(diags[2][key], abs=1e-6)

    def test_outlier_trimming_barely_moves_the_loss(self):
        """A small fraction of far outliers barely moves the trimmed loss.

        Cloud outliers are the norm in photogrammetry (Tier 2 W1 lesson,
        `docs/phase5_tier2_improvement_plan.md`) -- an untrimmed mean would let
        them steer the gradient.
        """
        torch.manual_seed(1)
        verts = torch.rand(200, 3)
        cloud = torch.rand(1000, 3)
        outliers = torch.rand(20, 3) + 10.0  # 2% of points, ~10m away

        loss_clean, _ = chamfer_loss(verts, cloud, trim_quantile=0.95)
        cloud_with_outliers = torch.cat([cloud, outliers], dim=0)
        loss_trimmed, _ = chamfer_loss(verts, cloud_with_outliers, trim_quantile=0.95)
        loss_untrimmed, _ = chamfer_loss(verts, cloud_with_outliers, trim_quantile=1.0)

        assert abs(float(loss_trimmed) - float(loss_clean)) < 1e-3
        # Without trimming the outliers visibly drag the loss up.
        assert float(loss_untrimmed) - float(loss_trimmed) > 1e-3

    def test_gradient_flows(self):
        verts = torch.rand(50, 3, requires_grad=True)
        cloud = torch.rand(300, 3)
        loss, _ = chamfer_loss(verts, cloud)
        loss.backward()
        assert verts.grad is not None
        assert torch.isfinite(verts.grad).all()
        assert torch.count_nonzero(verts.grad) > 0

    def test_diagnostics_are_detached_floats(self):
        """The diagnostics dict is plain, detached floats (never a live tensor)."""
        verts = torch.rand(20, 3, requires_grad=True)
        cloud = torch.rand(50, 3)
        _, diag = chamfer_loss(verts, cloud)
        assert isinstance(diag["mesh_to_cloud_m"], float)
        assert isinstance(diag["cloud_to_mesh_m"], float)

    def test_semantic_weights_applied_multiplicatively(self):
        """Zero-weighting a vertex/point removes it from its direction's mean."""
        verts = _well_separated_points(9, spacing=1.0)
        cloud = verts.clone()
        cloud[0] += torch.tensor([5.0, 0.0, 0.0])  # a bad correspondence for vertex 0

        vertex_weights = torch.ones(verts.shape[0])
        vertex_weights[0] = 0.0
        loss_weighted, _ = chamfer_loss(
            verts, cloud, vertex_weights=vertex_weights, trim_quantile=1.0
        )
        loss_unweighted, _ = chamfer_loss(verts, cloud, trim_quantile=1.0)
        assert float(loss_weighted) < float(loss_unweighted)

    def test_rejects_empty_inputs(self):
        with pytest.raises(ValueError):
            chamfer_loss(torch.zeros(0, 3), torch.rand(10, 3))
        with pytest.raises(ValueError):
            chamfer_loss(torch.rand(10, 3), torch.zeros(0, 3))


# ---------------------------------------------------------------------------
# normal_consistency_loss
# ---------------------------------------------------------------------------


def _square_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    """Two CCW triangles in the XY plane; every vertex normal is +Z."""
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return vertices, faces


class TestNormalConsistencyLoss:
    """`1 - |cos|` between each cloud normal and its nearest vertex normal."""

    def test_aligned_normals_give_zero_loss(self):
        vertices, faces = _square_mesh()
        cloud = torch.tensor([[0.5, 0.5, 0.0]])
        cloud_normals = torch.tensor([[0.0, 0.0, 1.0]])
        loss = normal_consistency_loss(vertices, faces, cloud, cloud_normals)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_flipped_normals_give_zero_loss(self):
        """Sign-agnostic by design: an inward-facing photogrammetry normal must
        not be penalised (module docstring / brief step 4)."""
        vertices, faces = _square_mesh()
        cloud = torch.tensor([[0.5, 0.5, 0.0]])
        cloud_normals = torch.tensor([[0.0, 0.0, -1.0]])
        loss = normal_consistency_loss(vertices, faces, cloud, cloud_normals)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_normals_give_loss_one(self):
        vertices, faces = _square_mesh()
        cloud = torch.tensor([[0.5, 0.5, 0.0]])
        cloud_normals = torch.tensor([[1.0, 0.0, 0.0]])
        loss = normal_consistency_loss(vertices, faces, cloud, cloud_normals)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        vertices, faces = _square_mesh()
        cloud = torch.tensor([[0.5, 0.5, 0.0]])
        bad_normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError):
            normal_consistency_loss(vertices, faces, cloud, bad_normals)


# ---------------------------------------------------------------------------
# build_uniform_laplacian / laplacian_smoothing_loss / displacement_regularisation
# ---------------------------------------------------------------------------


def _tetrahedron_faces() -> np.ndarray:
    """K4: 4 vertices, every pair connected, via 4 triangular faces."""
    return np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)


class TestUniformLaplacian:
    def test_symmetric_zero_row_sums(self):
        laplacian = build_uniform_laplacian(_tetrahedron_faces(), n_verts=4)
        dense = laplacian.to_dense()
        assert torch.allclose(dense, dense.T)
        assert torch.allclose(dense.sum(dim=1), torch.zeros(4))
        # K4: every vertex connects to the other 3 -> diagonal degree == 3.
        assert torch.allclose(dense.diagonal(), torch.full((4,), 3.0))

    def test_cached_by_topology(self):
        """SMPL topology is fixed -- rebuilding for the same faces is a no-op."""
        first = build_uniform_laplacian(_tetrahedron_faces(), n_verts=4)
        second = build_uniform_laplacian(_tetrahedron_faces(), n_verts=4)
        assert first is second

    def test_rejects_malformed_faces(self):
        with pytest.raises(ValueError):
            build_uniform_laplacian(np.zeros((4, 4), dtype=np.int64), n_verts=4)


class TestLaplacianSmoothingLoss:
    def test_constant_displacement_is_near_zero(self):
        """A pure translation is in the Laplacian's null space."""
        laplacian = build_uniform_laplacian(_tetrahedron_faces(), n_verts=4)
        d_const = torch.full((4, 3), 0.01)
        assert laplacian_smoothing_loss(d_const, laplacian).item() == pytest.approx(0.0, abs=1e-8)

    def test_constant_displacement_has_nonzero_regularisation(self):
        """Magnitude regularisation is independent of smoothness (master R2)."""
        d_const = torch.full((4, 3), 0.01)
        assert displacement_regularisation(d_const).item() > 1e-6

    def test_single_vertex_spike_has_high_laplacian_loss(self):
        laplacian = build_uniform_laplacian(_tetrahedron_faces(), n_verts=4)
        d_smooth = torch.zeros(4, 3)
        d_spike = torch.zeros(4, 3)
        d_spike[0] = torch.tensor([1.0, 0.0, 0.0])
        lap_smooth = laplacian_smoothing_loss(d_smooth, laplacian).item()
        lap_spike = laplacian_smoothing_loss(d_spike, laplacian).item()
        assert lap_spike > lap_smooth + 0.1

    def test_wrong_vertex_count_raises(self):
        laplacian = build_uniform_laplacian(_tetrahedron_faces(), n_verts=4)
        with pytest.raises(ValueError):
            laplacian_smoothing_loss(torch.zeros(5, 3), laplacian)

    def test_gradient_flows(self):
        laplacian = build_uniform_laplacian(_tetrahedron_faces(), n_verts=4)
        d = torch.zeros(4, 3, requires_grad=True)
        d2 = d + 0.0  # keep d a leaf while exercising the same graph as a fit loop
        loss = laplacian_smoothing_loss(d2, laplacian)
        loss.backward()
        assert d.grad is not None


class TestDisplacementRegularisation:
    def test_zero_is_zero(self):
        assert displacement_regularisation(torch.zeros(10, 3)).item() == 0.0

    def test_scales_with_magnitude(self):
        d_small = torch.full((10, 3), 0.01)
        d_large = torch.full((10, 3), 0.05)
        assert displacement_regularisation(d_large) > displacement_regularisation(d_small)

    def test_gradient_flows(self):
        d = torch.rand(10, 3, requires_grad=True)
        loss = displacement_regularisation(d)
        loss.backward()
        assert d.grad is not None
        assert torch.count_nonzero(d.grad) > 0
