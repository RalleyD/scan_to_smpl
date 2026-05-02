"""HMR (Human Mesh Recovery) module — Phases 2-3 of the ScanToSMPL pipeline."""

from scantosmpl.hmr.camera_hmr import CameraHMRInference, HMROutput
from scantosmpl.hmr.consensus import ConsensusBuilder, ConsensusResult
from scantosmpl.hmr.orientation import OrientationQuality, check_orientation_quality
from scantosmpl.hmr.pipeline import HMRPipeline

__all__ = [
    "CameraHMRInference",
    "HMROutput",
    "ConsensusBuilder",
    "ConsensusResult",
    "OrientationQuality",
    "check_orientation_quality",
    "HMRPipeline",
]
