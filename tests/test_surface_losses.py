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
            body_pose=torch.randn(
                1, self.model.body_model.NUM_BODY_JOINTS * 3, generator=gen
            ).to(self.model.device)
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


def _unused_numpy_guard() -> np.ndarray:
    """Placeholder keeping numpy imported for the loss tests added below."""
    return np.zeros(1)
