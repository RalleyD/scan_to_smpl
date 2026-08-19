"""SMPL fitting: loss functions, staged optimiser, Phase 5 pipeline, Tier 3 surface
refinement (SMPL+D) and its PSD-boundary artefact writer."""

from scantosmpl.fitting.artefacts import (
    SMPL_TEMPLATE_FACES_SHA256,
    faces_sha256,
    load_locked_betas,
    update_manifest,
    write_pose_artefacts,
)
from scantosmpl.fitting.losses import (
    joint_loss,
    pose_prior_loss,
    reprojection_loss,
    shape_regularisation,
)
from scantosmpl.fitting.optimiser import (
    DEFAULT_STAGES,
    OptimisationStage,
    RefinementResult,
    SMPLOptimiser,
)
from scantosmpl.fitting.pipeline import Phase5Pipeline, Phase5Result
from scantosmpl.fitting.surface import (
    DEFAULT_SURFACE_STAGES,
    SurfaceFitResult,
    SurfaceStage,
    Tier3SurfaceFitter,
    count_self_intersecting_faces,
)
from scantosmpl.fitting.surface_losses import (
    build_uniform_laplacian,
    chamfer_loss,
    displacement_regularisation,
    laplacian_smoothing_loss,
    normal_consistency_loss,
)
from scantosmpl.fitting.surface_pipeline import Tier3Pipeline, Tier3Result

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
    # Tier 3 — surface losses (smpld-and-losses)
    "chamfer_loss",
    "normal_consistency_loss",
    "build_uniform_laplacian",
    "laplacian_smoothing_loss",
    "displacement_regularisation",
    # Tier 3 — staged SMPL+D fitter (surface-fitting)
    "DEFAULT_SURFACE_STAGES",
    "SurfaceStage",
    "SurfaceFitResult",
    "Tier3SurfaceFitter",
    "count_self_intersecting_faces",
    # Tier 3 — orchestration + the 7.B artefact/manifest writer (tier3-pipeline-artefacts)
    "Tier3Pipeline",
    "Tier3Result",
    "SMPL_TEMPLATE_FACES_SHA256",
    "faces_sha256",
    "write_pose_artefacts",
    "update_manifest",
    "load_locked_betas",
]
