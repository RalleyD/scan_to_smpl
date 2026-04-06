"""SMPL/SMPL-X model wrapper with differentiable forward pass."""

from pathlib import Path

import torch
import torch.nn as nn

try:
    import smplx
except ImportError:
    raise ImportError("smplx is required: pip install smplx>=0.1.28")

from scantosmpl.types import SMPLOutput


class SMPLModel(nn.Module):
    """Wrapper around the smplx SMPL layer for differentiable forward pass.

    Supports both SMPL and SMPL-X models. Holds optimisable parameters
    (betas, body_pose, global_orient, translation, scale) and produces
    vertices + joints on forward().
    """

    # Expected output dimensions
    NUM_VERTICES = 6890
    NUM_FACES = 13776
    NUM_JOINTS = 24
    NUM_BETAS = 10

    def __init__(
        self,
        model_dir: str | Path,
        gender: str = "neutral",
        num_betas: int = 10,
        model_type: str = "smpl",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                f"Download SMPL models from smpl-x.is.tue.mpg.de and place .pkl files in {model_dir}"
            )

        self.body_model = smplx.create(
            model_path=str(model_dir.parent),  # smplx expects parent containing smpl/ or smplx/
            model_type=model_type,
            gender=gender,
            num_betas=num_betas,
            batch_size=1,
        ).to(self.device)

        # Optimisable parameters — initialised to neutral pose
        self.betas = nn.Parameter(torch.zeros(1, num_betas, device=self.device))
        self.body_pose = nn.Parameter(torch.zeros(1, self.body_model.NUM_BODY_JOINTS * 3, device=self.device))
        self.global_orient = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.translation = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.scale = nn.Parameter(torch.ones(1, device=self.device))

    def forward(
        self,
        betas: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        global_orient: torch.Tensor | None = None,
        translation: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> SMPLOutput:
        """Forward pass through the SMPL model.

        If parameters are not provided, uses the stored nn.Parameter values.
        All parameters support gradients for optimisation.
        """
        betas = betas if betas is not None else self.betas
        body_pose = body_pose if body_pose is not None else self.body_pose
        global_orient = global_orient if global_orient is not None else self.global_orient
        translation = translation if translation is not None else self.translation
        scale = scale if scale is not None else self.scale

        output = self.body_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=translation,
        )

        vertices = output.vertices * scale.unsqueeze(-1).unsqueeze(-1)
        joints = output.joints[:, :self.NUM_JOINTS] * scale.unsqueeze(-1).unsqueeze(-1)

        return SMPLOutput(
            vertices=vertices,
            joints=joints,
            faces=torch.tensor(self.body_model.faces.astype(int), device=self.device),
        )

    def get_joint_regressor(self) -> torch.Tensor:
        """Return the joint regressor matrix (J_regressor)."""
        return self.body_model.J_regressor

    def set_params(
        self,
        betas: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        global_orient: torch.Tensor | None = None,
        translation: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ):
        """Set parameter values (detached copy)."""
        if betas is not None:
            self.betas.data.copy_(betas.detach())
        if body_pose is not None:
            self.body_pose.data.copy_(body_pose.detach())
        if global_orient is not None:
            self.global_orient.data.copy_(global_orient.detach())
        if translation is not None:
            self.translation.data.copy_(translation.detach())
        if scale is not None:
            self.scale.data.copy_(scale.detach())

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        """Return current parameters as a dict."""
        return {
            "betas": self.betas.detach().clone(),
            "body_pose": self.body_pose.detach().clone(),
            "global_orient": self.global_orient.detach().clone(),
            "translation": self.translation.detach().clone(),
            "scale": self.scale.detach().clone(),
        }
