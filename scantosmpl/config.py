"""Configuration dataclasses for ScanToSMPL pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelPaths:
    """Paths to model files and checkpoints."""

    smpl_dir: Path = Path("models/smpl")
    smplx_dir: Path = Path("models/smplx")
    prior_dir: Path = Path("models/prior")
    checkpoint_dir: Path = Path("models/checkpoints")

    @property
    def smpl_model(self) -> Path:
        return self.smpl_dir / "SMPL_NEUTRAL.pkl"

    def smpl_model_gendered(self, gender: str) -> Path:
        return self.smpl_dir / f"SMPL_{gender.upper()}.pkl"


@dataclass
class DetectionConfig:
    """Detection stage configuration."""

    person_detector: str = "PekingU/rtdetr_r50vd_coco_o365"
    keypoint_detector: str = "usyd-community/vitpose-plus-base"
    person_confidence: float = 0.5
    keypoint_confidence: float = 0.3  # for classification; raw scores preserved
    min_visible_keypoints: int = 5  # minimum for PARTIAL (below = SKIP)
    save_debug: bool = True
    debug_dir: Path = Path("output/debug/detection")


@dataclass
class HMRConfig:
    """HMR (Human Mesh Recovery) configuration."""

    backend: Literal["camerahmr", "prompthmr"] = "camerahmr"
    batch_size: int = 4
    device: str = "cuda"


@dataclass
class CalibrationConfig:
    """Self-calibration configuration."""

    pnp_method: str = "SOLVEPNP_ITERATIVE"
    ransac_threshold: float = 8.0  # pixels
    min_inliers: int = 20
    use_dense_keypoints: bool = True  # 138 CameraHMR surface keypoints


@dataclass
class FittingConfig:
    """SMPL fitting optimisation configuration."""

    # Optimiser
    learning_rate: float = 1e-2
    max_iterations: int = 500
    convergence_threshold: float = 1e-6

    # Loss weights (Tier 2)
    w_joint: float = 1.0
    w_reprojection: float = 0.5
    w_pose_prior: float = 0.01
    w_shape_reg: float = 0.01

    # Loss weights (Tier 3 surface)
    w_chamfer: float = 1.0
    w_normal: float = 0.1
    w_laplacian: float = 0.1

    # Body-part weights for chamfer
    body_part_weights: dict[str, float] = field(default_factory=lambda: {
        "torso": 1.0,
        "arms": 0.7,
        "legs": 0.7,
        "head": 0.5,
        "hands": 0.3,
        "feet": 0.4,
    })


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Input
    image_dir: Path | None = None
    pointcloud_path: Path | None = None
    reference_pose: Literal["a-pose", "t-pose"] = "a-pose"
    gender: Literal["neutral", "male", "female"] = "neutral"
    output_dir: Path = Path("output")

    # Tiers to run
    run_tier1: bool = True
    run_tier2: bool = True
    run_tier3: bool = True  # only if pointcloud_path is set

    # Sub-configs
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    hmr: HMRConfig = field(default_factory=HMRConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)

    # Device
    device: str = "cuda"
