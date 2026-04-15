

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

Additionally, the model needs smpl_mean_params.npz for initialisation.

## tests

### Integration tests (requires scanner images + downloads models):
pytest tests/integration/ -v

### All tests:
pytest tests/ --no-header -v

## TODOs

- remove smplx submodule - no longer needed for chumpy cleaning
