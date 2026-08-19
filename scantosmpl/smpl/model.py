"""SMPL/SMPL-X model wrapper with differentiable forward pass."""

from pathlib import Path
from typing import cast

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
    (betas, body_pose, global_orient, translation, scale, displacements) and
    produces vertices + joints on forward().

    SMPL+D: ``displacements`` is a per-vertex offset field ``D`` in the **posed
    world** frame, metres — the same frame ``forward()`` returns. It is added
    *after* the global scale, so it is never rescaled, and it is applied to
    vertices only (joints are a medial-axis quantity and stay undisplaced).
    Zero-initialised, so a default model is bit-for-bit the pre-SMPL+D model.
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
                f"Download SMPL models from smpl-x.is.tue.mpg.de and place .pkl "
                f"files in {model_dir}"
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
        self.body_pose = nn.Parameter(
            torch.zeros(1, self.body_model.NUM_BODY_JOINTS * 3, device=self.device)
        )
        self.global_orient = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.translation = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.scale = nn.Parameter(torch.ones(1, device=self.device))
        # SMPL+D per-vertex displacement field, posed world frame, metres (Tier 3).
        self.displacements = nn.Parameter(torch.zeros(1, self.NUM_VERTICES, 3, device=self.device))

    def forward(
        self,
        betas: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        global_orient: torch.Tensor | None = None,
        translation: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        displacements: torch.Tensor | None = None,
        apply_displacements: bool = True,
    ) -> SMPLOutput:
        """Forward pass through the SMPL model.

        If parameters are not provided, uses the stored nn.Parameter values.
        All parameters support gradients for optimisation.

        Args:
            betas: (B, num_betas) shape parameters.
            body_pose: (B, NUM_BODY_JOINTS*3) axis-angle body pose.
            global_orient: (B, 3) axis-angle root orientation.
            translation: (B, 3) global translation, metres.
            scale: (B,) global scale multiplier.
            displacements: (1, 6890, 3) or (B, 6890, 3) per-vertex offsets in the
                **posed world** frame, metres. ``None`` uses the stored
                ``self.displacements`` parameter (zero at construction).
            apply_displacements: ``False`` forces the ``D = 0`` baseline — the
                mesh Tier 3 / PSD differences against.

        Returns:
            SMPLOutput with vertices (B, 6890, 3) and joints (B, 24, 3), both in
            the SMPL/world posed frame in metres, on ``self.device``.

        Frame note (Tier 3 / master D4): ``D`` is added *after* the global scale,
        so it is expressed in final posed world metres and the identity

            D == forward(...).vertices - forward(..., apply_displacements=False).vertices

        holds by construction. **Joints are never displaced** — ``D`` is a surface
        quantity, and displacing joints would corrupt every Tier 2 joint metric.
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
        joints = output.joints[:, : self.NUM_JOINTS] * scale.unsqueeze(-1).unsqueeze(-1)

        if apply_displacements:
            disp = displacements if displacements is not None else self.displacements
            vertices = vertices + disp.to(vertices.device)

        return SMPLOutput(
            vertices=vertices,
            joints=joints,
            faces=torch.tensor(self.body_model.faces.astype(int), device=self.device),
        )

    def get_joint_regressor(self) -> torch.Tensor:
        """Return the joint regressor matrix (J_regressor)."""
        # smplx ships no type stubs, so J_regressor is untyped (Any) from mypy's
        # perspective even though it is always a torch.Tensor at runtime.
        return cast(torch.Tensor, self.body_model.J_regressor)

    def set_params(
        self,
        betas: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        global_orient: torch.Tensor | None = None,
        translation: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        displacements: torch.Tensor | None = None,
    ) -> None:
        """Set parameter values (detached copy).

        ``displacements`` is (1, 6890, 3), posed world frame, metres.
        """
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
        if displacements is not None:
            self.displacements.data.copy_(displacements.detach())

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        """Return current parameters as a dict.

        Keys: betas, body_pose, global_orient, translation, scale, displacements.
        ``displacements`` is (1, 6890, 3) in the posed world frame, metres.
        """
        return {
            "betas": self.betas.detach().clone(),
            "body_pose": self.body_pose.detach().clone(),
            "global_orient": self.global_orient.detach().clone(),
            "translation": self.translation.detach().clone(),
            "scale": self.scale.detach().clone(),
            "displacements": self.displacements.detach().clone(),
        }
