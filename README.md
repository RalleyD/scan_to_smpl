

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

## TODOs

- remove smplx submodule - no longer needed for chumpy cleaning
