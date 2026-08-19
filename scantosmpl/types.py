"""Core type definitions for ScanToSMPL pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch

# --- Tier 3 / PSD-boundary constants (master §5.1) -------------------------
#
# `D` (per-vertex displacement) is always persisted in the SAME frame the SMPL
# forward pass returns: posed, world, AFTER LBS and AFTER the global scale.
# This is master D4 — PSD applies `R_v(theta)^-1` itself; Tier 3 does not.
DISPLACEMENT_FRAME: Literal["posed_world"] = "posed_world"

#: Fixed SMPL (not SMPL-X) topology throughout this feature — master §8.
SMPL_NUM_VERTICES: int = 6890
SMPL_NUM_FACES: int = 13776


class ViewType(Enum):
    """Classification of an image view."""

    FULL_BODY = "full_body"
    PARTIAL = "partial"
    SKIP = "skip"


@dataclass
class CameraParams:
    """Camera parameters for a single view."""

    # Intrinsics
    focal_length: float  # pixels
    principal_point: tuple[float, float] = (0.0, 0.0)  # (cx, cy)
    fov: float | None = None  # degrees, from CameraHMR HumanFoV

    # HMR camera translation (Phase 2 — weak-perspective → 3D)
    hmr_translation: np.ndarray | None = None  # (3,) camera-space translation from CameraHMR

    # Extrinsics (recovered in Tier 2)
    rotation: np.ndarray | None = None  # (3, 3)
    translation: np.ndarray | None = None  # (3,)

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        cx, cy = self.principal_point
        return np.array(
            [
                [self.focal_length, 0, cx],
                [0, self.focal_length, cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @property
    def has_extrinsics(self) -> bool:
        return self.rotation is not None and self.translation is not None


@dataclass
class ViewResult:
    """Per-view detection and HMR results."""

    image_path: Path
    view_type: ViewType

    # Detection (Phase 1)
    bbox: np.ndarray | None = None  # (4,) — x1, y1, x2, y2
    keypoints_2d: np.ndarray | None = None  # (17, 2) COCO keypoints
    keypoint_confs: np.ndarray | None = None  # (17,) confidence scores

    # HMR (Phase 2)
    betas: np.ndarray | None = None  # (10,) shape parameters
    body_pose: np.ndarray | None = None  # (69,) body pose (23 joints x 3 axis-angle)
    global_orient: np.ndarray | None = None  # (3,) global orientation
    camera: CameraParams | None = None

    # CameraHMR dense keypoints (Phase 2)
    dense_keypoints_2d: np.ndarray | None = None  # (138, 2)
    dense_keypoint_confs: np.ndarray | None = None  # (138,)

    # HMR suitability flag (Phase 2)
    # False for pure side views or extreme floor-up angles where CameraHMR is unreliable.
    # The view is still valid for Phase 1 detection and Tier 2 PnP.
    hmr_suitable: bool = True


@dataclass
class SMPLOutput:
    """Output from SMPL forward pass."""

    vertices: torch.Tensor  # (B, 6890, 3)
    joints: torch.Tensor  # (B, 24, 3) or (B, J, 3)
    faces: torch.Tensor  # (13776, 3)


@dataclass
class FittingResult:
    """Result from a tier of SMPL fitting."""

    # SMPL parameters
    betas: np.ndarray  # (10,)
    body_pose: np.ndarray  # (69,) — 23 joints x 3 axis-angle
    global_orient: np.ndarray  # (3,)
    translation: np.ndarray  # (3,)
    scale: float = 1.0

    # Mesh
    vertices: np.ndarray | None = None  # (6890, 3)
    faces: np.ndarray | None = None  # (13776, 3)

    # Quality metrics
    tier: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    # Per-vertex displacements (Tier 3, optional)
    displacements: np.ndarray | None = None  # (6890, 3)
    # Frame `displacements` is expressed in (master D4/7.B3) — self-describing
    # in memory, mirroring the explicit field written to disk by
    # `scantosmpl.fitting.artefacts.write_pose_artefacts`.
    displacement_frame: str = DISPLACEMENT_FRAME

    # Per-view cameras (Tier 2+)
    cameras: dict[str, CameraParams] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier 3 / PSD-boundary types (master §5.1) — the ONLY cross-tier types this
# feature adds to the shared contract. Everything else (CloudAlignment,
# SurfaceFitResult, ChamferReport, ...) is a module-local result dataclass
# living beside its own module, per the repo convention.
# ---------------------------------------------------------------------------


@dataclass
class Tier3Quality:
    """Per-pose fit quality persisted alongside `D` (7.B7).

    All `*_mm` fields are millimetres (the single unit-conversion boundary —
    see `scantosmpl.evaluation.surface_metrics`). `pa_mpjpe_mm` and
    `median_reproj_px` are carried through from Tier 2's own metrics when
    present; `None` when Tier 2 did not record them (AC20's documented
    exception to "every field populated").
    """

    chamfer_cloud_to_mesh_mean_mm: float
    chamfer_cloud_to_mesh_median_mm: float
    chamfer_cloud_to_mesh_rms_mm: float
    chamfer_mesh_to_cloud_mean_mm: float
    chamfer_mesh_to_cloud_median_mm: float
    chamfer_mesh_to_cloud_rms_mm: float
    tessellation_floor_mean_mm: float
    tessellation_floor_max_mm: float
    icp_inlier_rmse_mm: float
    icp_fitness: float
    displacement_mean_mm: float
    displacement_p95_mm: float
    pa_mpjpe_mm: float | None = None  # carried from Tier 2
    median_reproj_px: float | None = None  # carried from Tier 2


@dataclass
class PoseArtefact:
    """One pose's entry in the corpus manifest (7.B6, 7.B8).

    `oracle_only=True` marks a pose fitted purely as a PSD evaluation
    ceiling — it MUST NOT enter PSD training (7.B8). `betas_locked` mirrors
    `SurfaceFitResult.betas_locked` for this specific pose, so a manifest
    reader can tell a `--lock-betas` run apart from a `--lock-betas`-free
    (β-refinement) run without opening `smpl_params.npz` (master D10).
    """

    pose_name: str
    directory: str  # relative to the manifest, e.g. "t-pose"
    oracle_only: bool
    betas_locked: bool
    has_displacements: bool
    has_pointcloud: bool
    quality: Tier3Quality
