"""Staged SMPL optimiser for Phase 5 refinement."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from scantosmpl.fitting.losses import (
    joint_loss,
    pose_prior_loss,
    reprojection_loss,
    shape_regularisation,
)
from scantosmpl.hmr.consensus import ConsensusResult
from scantosmpl.smpl.model import SMPLModel
from scantosmpl.utils.geometry import compute_pa_mpjpe

logger = logging.getLogger(__name__)


@dataclass
class OptimisationStage:
    """Configuration for a single optimisation stage."""

    name: str
    params: list[str]               # parameter names to optimise
    n_iterations: int
    w_joint: float = 1.0
    w_reproj: float = 0.0
    w_pose_prior: float = 0.0
    w_shape_reg: float = 0.0
    learning_rate: float = 1e-2


@dataclass
class RefinementResult:
    """Output from the staged SMPL optimiser."""

    betas: np.ndarray               # (10,)
    body_pose: np.ndarray           # (69,)
    global_orient: np.ndarray       # (3,)
    translation: np.ndarray         # (3,)
    scale: float
    vertices: np.ndarray            # (6890, 3)
    joints: np.ndarray              # (24, 3)
    loss_history: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


# Default three-stage schedule from the spec
DEFAULT_STAGES: list[OptimisationStage] = [
    OptimisationStage(
        name="global_alignment",
        params=["global_orient", "translation", "scale"],
        n_iterations=50,
        w_joint=1.0,
        w_reproj=0.0,
        w_pose_prior=0.0,
        w_shape_reg=0.0,
        learning_rate=1e-2,
    ),
    OptimisationStage(
        name="shape_refinement",
        params=["betas", "global_orient", "translation", "scale"],
        n_iterations=100,
        w_joint=0.5,
        w_reproj=1.0,
        w_pose_prior=0.0,
        w_shape_reg=0.01,
        learning_rate=5e-3,
    ),
    OptimisationStage(
        name="full_refinement",
        params=["betas", "body_pose", "global_orient", "translation", "scale"],
        n_iterations=400,
        w_joint=0.1,
        w_reproj=1.0,
        w_pose_prior=0.01,
        w_shape_reg=0.01,
        learning_rate=1e-3,
    ),
]


class SMPLOptimiser:
    """Staged SMPL parameter optimiser.

    Initialises from a consensus result, then refines parameters against:
    - Triangulated 3D joint positions (L_joint, Huber)
    - Multi-view 2D reprojection (L_reproj, Huber + confidence-weighted)
    - Pose prior (L_pose_prior, L2)
    - Shape regularisation (L_shape_reg, L2)

    Loss weights are annealed across stages so reprojection dominates by
    the final stage, mitigating DLT triangulation noise.
    """

    def __init__(
        self,
        smpl_model: SMPLModel,
        coco_to_smpl: dict[int, int],
    ) -> None:
        self.smpl = smpl_model
        self.coco_to_smpl = coco_to_smpl
        self.device = smpl_model.device

    def refine(
        self,
        consensus: ConsensusResult,
        triangulated_joints: np.ndarray,
        keypoints_2d: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
        cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        stages: list[OptimisationStage] | None = None,
    ) -> RefinementResult:
        """Run staged SMPL optimisation.

        Args:
            consensus: Phase 3 consensus result (initialisation).
            triangulated_joints: (J, 3) triangulated 3D joints in SMPL frame.
            keypoints_2d: {view_name: (J_coco, 2)} undistorted 2D keypoints.
            confs: {view_name: (J_coco,)} confidence scores.
            cameras: {view_name: (R, t, K)} camera matrices as numpy arrays.
            stages: Override default stage schedule.

        Returns:
            RefinementResult with refined parameters and quality metrics.
        """
        if stages is None:
            stages = DEFAULT_STAGES

        # Initialise SMPL params from consensus
        self.smpl.set_params(
            betas=torch.tensor(
                consensus.betas, dtype=torch.float32, device=self.device).unsqueeze(0),
            body_pose=torch.tensor(
                consensus.body_pose, dtype=torch.float32, device=self.device).unsqueeze(0),
            global_orient=torch.tensor(
                consensus.global_orient, dtype=torch.float32, device=self.device).unsqueeze(0),
            translation=torch.zeros(
                1, 3, dtype=torch.float32, device=self.device),
            scale=torch.ones(1, dtype=torch.float32, device=self.device),
        )

        # Pre-compute tensors that stay constant across stages
        target_joints_t = torch.tensor(
            triangulated_joints, dtype=torch.float32, device=self.device
        )
        kp2d_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in keypoints_2d.items()
        }
        conf_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in confs.items()
        }
        cam_tensors = {
            name: (
                torch.tensor(R, dtype=torch.float32, device=self.device),
                torch.tensor(t, dtype=torch.float32, device=self.device),
                torch.tensor(K, dtype=torch.float32, device=self.device),
            )
            for name, (R, t, K) in cameras.items()
        }

        all_loss_history: dict[str, list[float]] = {}

        for stage in stages:
            logger.info(
                "Stage '%s': %d iters | params=%s | w_joint=%.2f w_reproj=%.2f",
                stage.name, stage.n_iterations, stage.params, stage.w_joint, stage.w_reproj,
            )

            params_to_opt = self._get_params(stage.params)
            optimiser = torch.optim.Adam(params_to_opt, lr=stage.learning_rate)

            stage_history: list[float] = []
            prev_loss = float("inf")

            for it in range(stage.n_iterations):
                optimiser.zero_grad()
                output = self.smpl.forward()
                pred_joints = output.joints  # (1, 24, 3)

                loss = torch.tensor(0.0, device=self.device)

                if stage.w_joint > 0:
                    lj = joint_loss(pred_joints, target_joints_t)
                    loss = loss + stage.w_joint * lj

                if stage.w_reproj > 0 and kp2d_tensors:
                    lr = reprojection_loss(
                        pred_joints,
                        kp2d_tensors,
                        conf_tensors,
                        cam_tensors,
                        self.coco_to_smpl,
                    )
                    loss = loss + stage.w_reproj * lr

                if stage.w_pose_prior > 0:
                    lp = pose_prior_loss(self.smpl.body_pose)
                    loss = loss + stage.w_pose_prior * lp

                if stage.w_shape_reg > 0:
                    ls = shape_regularisation(self.smpl.betas)
                    loss = loss + stage.w_shape_reg * ls

                loss.backward()
                optimiser.step()

                loss_val = float(loss.item())
                stage_history.append(loss_val)

                # Early stopping if converged
                if abs(prev_loss - loss_val) < 1e-7 and it > 10:
                    logger.debug(
                        "Stage '%s' converged at iter %d", stage.name, it)
                    break
                prev_loss = loss_val

            all_loss_history[stage.name] = stage_history
            logger.info(
                "Stage '%s' done: loss %.4f → %.4f",
                stage.name, stage_history[0] if stage_history else 0, stage_history[-1] if stage_history else 0,
            )

        # Extract final parameters
        with torch.no_grad():
            final_output = self.smpl.forward()
            params = self.smpl.get_params_dict()

        betas = params["betas"].squeeze(0).cpu().numpy()
        body_pose = params["body_pose"].squeeze(0).cpu().numpy()
        global_orient = params["global_orient"].squeeze(0).cpu().numpy()
        translation = params["translation"].squeeze(0).cpu().numpy()
        scale = float(params["scale"].cpu().numpy())
        vertices = final_output.vertices.squeeze(0).cpu().numpy()
        joints = final_output.joints.squeeze(0).cpu().numpy()

        # Compute PA-MPJPE against triangulated joints (only where quality > 0)
        valid = np.linalg.norm(triangulated_joints, axis=1) > 1e-6
        if valid.sum() >= 2:
            pa_mpjpe = compute_pa_mpjpe(
                joints[: len(triangulated_joints)][valid],
                triangulated_joints[valid],
            ) * 1000  # metres → mm
        else:
            pa_mpjpe = float("nan")

        metrics = {"pa_mpjpe_mm": pa_mpjpe}

        return RefinementResult(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            translation=translation,
            scale=scale,
            vertices=vertices,
            joints=joints,
            loss_history=all_loss_history,
            metrics=metrics,
        )

    def _get_params(self, param_names: list[str]) -> list[torch.nn.Parameter]:
        """Return model parameters by name."""
        param_map = {
            "betas": self.smpl.betas,
            "body_pose": self.smpl.body_pose,
            "global_orient": self.smpl.global_orient,
            "translation": self.smpl.translation,
            "scale": self.smpl.scale,
        }
        result = []
        for name in param_names:
            if name not in param_map:
                raise ValueError(
                    f"Unknown parameter: '{name}'. Valid: {list(param_map)}")
            result.append(param_map[name])
        return result
