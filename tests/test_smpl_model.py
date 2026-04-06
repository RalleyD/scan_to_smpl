"""Tests for SMPL model wrapper — Phase 0 acceptance criteria."""

import time

import numpy as np
import pytest
import torch

from scantosmpl.smpl.model import SMPLModel

SMPL_DIR = "models/smpl"


def _smpl_available() -> bool:
    from pathlib import Path
    return (Path(SMPL_DIR) / "SMPL_NEUTRAL.pkl").exists()


requires_smpl = pytest.mark.skipif(
    not _smpl_available(),
    reason=f"SMPL model files not found in {SMPL_DIR}/ — see models/README.md",
)


@requires_smpl
class TestSMPLModel:
    """Phase 0 acceptance criteria 0.2–0.6."""

    @pytest.fixture(autouse=True)
    def setup(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SMPLModel(model_dir=SMPL_DIR, gender="neutral", device=device)

    def test_output_shapes(self):
        """0.2: SMPL forward: 6890 verts, 13776 faces, 24 joints."""
        output = self.model()
        assert output.vertices.shape == (1, 6890, 3)
        assert output.joints.shape == (1, 24, 3)
        assert output.faces.shape[0] == 13776
        assert output.faces.shape[1] == 3

    def test_differentiable(self):
        """0.3: loss.backward() flows grads through beta and theta."""
        output = self.model()
        loss = output.vertices.sum() + output.joints.sum()
        loss.backward()

        assert self.model.betas.grad is not None, "No gradient on betas"
        assert self.model.body_pose.grad is not None, "No gradient on body_pose"
        assert self.model.global_orient.grad is not None, "No gradient on global_orient"
        assert self.model.translation.grad is not None, "No gradient on translation"
        assert self.model.scale.grad is not None, "No gradient on scale"

    @pytest.mark.gpu
    def test_forward_speed(self):
        """0.4: Forward < 50ms on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU required")

        # Warmup
        for _ in range(5):
            self.model()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            self.model()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.05, f"Forward pass took {elapsed*1000:.1f}ms (limit: 50ms)"

    def test_shoulder_width(self):
        """0.6: Joint regressor: shoulder width 35-45cm at neutral pose."""
        output = self.model()
        joints = output.joints[0].detach().cpu().numpy()

        # SMPL joint indices: 16 = left shoulder, 17 = right shoulder
        left_shoulder = joints[16]
        right_shoulder = joints[17]
        shoulder_width_m = np.linalg.norm(left_shoulder - right_shoulder)

        assert 0.30 <= shoulder_width_m <= 0.50, (
            f"Shoulder width {shoulder_width_m:.3f}m outside expected range 0.30-0.50m"
        )

    def test_set_params(self):
        """Parameters can be set and produce different output."""
        output_neutral = self.model()
        v_neutral = output_neutral.vertices.detach().clone()

        # Set non-zero betas
        new_betas = torch.randn(1, 10, device=self.model.device) * 2
        self.model.set_params(betas=new_betas)
        output_shaped = self.model()
        v_shaped = output_shaped.vertices.detach()

        assert not torch.allclose(v_neutral, v_shaped), "Changing betas should change vertices"

    def test_batch_forward(self):
        """Model handles batch size > 1 via explicit params."""
        B = 4
        betas = torch.zeros(B, 10, device=self.model.device)
        body_pose = torch.zeros(B, self.model.body_model.NUM_BODY_JOINTS * 3, device=self.model.device)
        global_orient = torch.zeros(B, 3, device=self.model.device)
        translation = torch.zeros(B, 3, device=self.model.device)
        scale = torch.ones(B, device=self.model.device)

        output = self.model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            translation=translation,
            scale=scale,
        )
        assert output.vertices.shape == (B, 6890, 3)
        assert output.joints.shape == (B, 24, 3)


@requires_smpl
class TestSMPLGenders:
    """Verify gendered models load correctly."""

    @pytest.mark.parametrize("gender", ["neutral", "male", "female"])
    def test_gender_loads(self, gender):
        from pathlib import Path
        model_path = Path(SMPL_DIR) / f"SMPL_{gender.upper()}.pkl"
        if not model_path.exists():
            pytest.skip(f"{model_path} not found")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SMPLModel(model_dir=SMPL_DIR, gender=gender, device=device)
        output = model()
        assert output.vertices.shape == (1, 6890, 3)


class TestKaolinSmoke:
    """0.5: Kaolin chamfer_distance works on two random point clouds."""

    def test_chamfer_distance(self):
        try:
            from kaolin.metrics.pointcloud import chamfer_distance
        except ImportError:
            pytest.skip("kaolin not installed")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        p1 = torch.randn(1, 100, 3, device=device)
        p2 = torch.randn(1, 100, 3, device=device)

        dist = chamfer_distance(p1, p2)
        assert dist.shape == ()
        assert dist.item() > 0
        assert torch.isfinite(dist)
