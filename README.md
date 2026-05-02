

## implementation phases

## phase 0 - scaffolding

.devcontainer/Dockerfile	

    Python 3.10 + PyTorch 2.4 + CUDA 12.1 + all deps

.devcontainer/devcontainer.json	

    VSCode devcontainer config with GPU support

pyproject.toml	

    Package metadata, deps, pytest/ruff/mypy config

scantosmpl/config.py	    

    Dataclass configs for all pipeline stages

scantosmpl/types.py

	ViewType, CameraParams, ViewResult, FittingResult, SMPLOutput

scantosmpl/smpl/model.py

	SMPL wrapper with differentiable forward pass + optimisable params

scantosmpl/cli.py

	Click CLI skeleton (fit-images, fit-pointcloud, fit-combined)

models/README.md

	Download instructions for all model files

tests/test_smpl_model.py

	10 tests covering all Phase 0 acceptance criteria

utils/clean_smpl.py

	# As a module
	python -m scantosmpl.utils.clean_smpl models/smpl/ --output models/smpl/

	# Or from code
	from scantosmpl.utils.clean_smpl import clean_smpl_pkl, clean_directory
	clean_directory(Path("models/smpl/"))

## phase 1 - keypoint detection

```
python -c "\nfrom scantosmpl.detection.pipeline import DetectionPipeline\nfrom pathlib import Path\n\npipeline = DetectionPipeline(device='cuda')\nresults = pipeline.process_directory(\n    Path('data/t-pose/jpg'),\n    debug_dir=Path('output/debug/detection'),\n)\nprint(f'\nProcessed {len(results)} images')\nfor r in results:\n    n_vis = int((r.keypoint_confs > 0.3).sum()) if r.keypoint_confs is not None else 0\n    print(f'  {r.image_path.name}: {r.view_type.value} ({n_vis}/17 kps)')\n"
```

### CameraHMR prerequisites

Model	Checkpoint	Output	You have?
Main HMR	camerahmr_checkpoint_cleaned.ckpt (7.5GB)	SMPL params (β, θ) + weak-perspective camera + 44 2D keypoints	Yes
FLNet	cam_model_cleaned.ckpt	Focal length / FoV from image	?
DenseKP	densekp.ckpt	138 dense 3D surface keypoints	?
The 138 dense keypoints (REVIEW.md criterion 2.3) come from a separate model, not the main CameraHMR checkpoint. And the FoV estimation (criterion 2.2) comes from FLNet, also separate.


View	Spread	Torso frac	Excluded?	Reason
cam02_4	0.07	0.28	yes	pure side view (spread < 0.12)
cam06_4	0.02	0.28	yes	pure side view (spread < 0.12)
cam07_6	0.26	0.22	yes	floor-up angle (torso < 0.23)
all others	≥ 0.17	≥ 0.23	no	—

CameraHMR submodule (external/CameraHMR)

Added as a git submodule (master branch, commit b1b6eea)
Fixed upstream syntax error in densekp_model.py (def forward(self, batch) missing colon)
New/modified files:

File	What it does
scantosmpl/hmr/camera_hmr.py	CameraHMRInference: loads all three models, monkey-patches SMPL_MEAN_PARAMS_FILE, shared ViT-H backbone, CLIFF camera conversion, DenseKP keypoint denormalisation
scantosmpl/hmr/orientation.py	check_orientation_quality: upright check, rotation magnitude, T-pose arm check → score + warnings
scantosmpl/hmr/pipeline.py	HMRPipeline: orchestrates all views, PIL wireframe overlay, JSON + summary debug output
scantosmpl/hmr/init.py	Clean exports
scantosmpl/config.py	HMRConfig with all checkpoint paths
scantosmpl/types.py	CameraParams.hmr_translation added
pyproject.toml	Added pytorch-lightning, timm, einops, yacs, loguru
tests/test_hmr.py	8 test classes, no GPU needed
tests/integration/test_hmr_integration.py	5 test classes covering criteria 2.1–2.7
Run commands:


# Install new deps first
pip install -e ".[dev]"

# Unit tests (no GPU)
pytest tests/test_hmr.py -v

# Integration tests (GPU + checkpoints)
pytest tests/integration/test_hmr_integration.py -v -m gpu

# End-to-end debug run
python -c "
from pathlib import Path
from scantosmpl.detection.pipeline import DetectionPipeline
from scantosmpl.hmr.pipeline import HMRPipeline
from scantosmpl.config import HMRConfig

det = DetectionPipeline(device='cuda')
views = det.process_directory(Path('data/t-pose/jpg'), debug_dir=Path('output/debug/detection'))
hmr = HMRPipeline(HMRConfig(), device='cuda')
views = hmr.process_views(views, Path('data/t-pose/jpg'), debug_dir=Path('output/debug/hmr'))
"
# Inspect output/debug/hmr/summary.txt and *_hmr_overlay.jpg

Additionally, the model needs smpl_mean_params.npz for initialisation.

## tests

### Integration tests (requires scanner images + downloads models):
pytest tests/integration/ -v

### All tests:
pytest tests/ --no-header -v

## TODOs

- remove smplx submodule - no longer needed for chumpy cleaning
