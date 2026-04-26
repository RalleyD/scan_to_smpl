"""Core type definitions for ScanToSMPL pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch


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
        return np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float64)

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

    # Per-view cameras (Tier 2+)
    cameras: dict[str, CameraParams] = field(default_factory=dict)
