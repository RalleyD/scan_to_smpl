"""SMPL fitting: loss functions, staged optimiser, Phase 5 pipeline."""

from scantosmpl.fitting.losses import (
    joint_loss,
    pose_prior_loss,
    reprojection_loss,
    shape_regularisation,
)
from scantosmpl.fitting.optimiser import DEFAULT_STAGES, OptimisationStage, RefinementResult, SMPLOptimiser
from scantosmpl.fitting.pipeline import Phase5Pipeline, Phase5Result

__all__ = [
    "joint_loss",
    "reprojection_loss",
    "pose_prior_loss",
    "shape_regularisation",
    "DEFAULT_STAGES",
    "OptimisationStage",
    "RefinementResult",
    "SMPLOptimiser",
    "Phase5Pipeline",
    "Phase5Result",
]
