

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


# phase 3

Global orient handling
Each view's global_orient encodes body rotation relative to that camera — they differ wildly across views (expected). For the Tier 1 consensus mesh, we need one canonical orientation. Options:

(A) Most frontal view — pick the view with the most symmetric shoulder spread (cam01_2 or cam05_6 look frontal). Use its global_orient directly. Simple, robust.
(B) Median-view selection — cluster global_orient vectors and pick the medoid. More principled but may select a 3/4 view.
(C) Canonical zero — set global_orient to [0,0,0] (identity). The consensus mesh faces "forward" in SMPL's canonical frame. Tier 2 PnP recovers actual per-view orientation.

recommend (C) — it keeps the Tier 1 output clean and canonical. The per-view global_orients are preserved in the ViewResults for Tier 2. But (A) would give a more "realistic" looking mesh for debug visualization.
 Phase 4 (PnP with 138 dense keypoints) is the proper solution for recovering per-view orientation. Baking a "best guess frontal" into Tier 1 would just be noise that Tier 2 has to undo. Canonical [0,0,0] keeps the consensus clean — body_pose captures the T-pose shape, global_orient is identity, and the per-view orientations are preserved in ViewResults for Phase 4.

 Body pose aggregation
body_pose (69D = 23 joints × 3 axis-angle) is view-invariant in theory. For T-pose, rotations are small. Options:

(A) SO(3) Fréchet mean per joint — mathematically correct rotation averaging. Convert each joint's axis-angle → rotation matrix, iteratively compute the Fréchet mean on SO(3), convert back. Robust for any rotation magnitude.
(B) Weighted component-wise median — simple, works well when rotations are small and consistent (as in T-pose). Much simpler to implement.
Given your preference for high quality, I'd recommend (A) — it handles edge cases properly and the implementation is reusable for Tier 2.

The EXIF focal lengths come from the camera's actual lens metadata — they're physical measurements, not estimates. FLNet is a neural network predicting what EXIF already tells us. For your scanner (known Canon EOS 2000D hardware), EXIF is strictly more reliable. I'll implement it as: use EXIF focal lengths directly, report FLNet vs EXIF diff as a diagnostic (already done in Phase 2 summary), and skip computing a "consensus FoV" unless EXIF is missing. Option A is a clean fallback if you later process images without EXIF.

For viewing .obj meshes: The quickest option on Linux is MeshLab:

sudo apt install meshlab
meshlab output/debug/consensus/consensus_mesh.obj
Alternatively, for a quick Python one-liner without installing anything new:

python3 -c "import trimesh; trimesh.load('output/debug/consensus/consensus_mesh.obj').show()"

## some details about the SMPL OBJ output:

Full topology — The .obj has 6890 vertices and 13776 triangular faces (the f lines). Import into Blender/Maya and you'll see a complete watertight mesh, not a point cloud. This is SMPL's fixed template topology — every SMPL mesh has the same vertex count and face connectivity, which is what makes it so useful for animation pipelines.

Articulable, but not directly via the .obj — The .obj is a "baked" static mesh (the T-pose result after applying our consensus betas + body_pose). It has no rig, joints, or blend shapes embedded in it. To articulate it in Maya/Blender you'd need to either:

Export as FBX with a skeleton — SMPL defines a 24-joint kinematic tree. Libraries like smplx can produce the joint positions, and tools like SMPL-to-FBX or Meshcapade's Blender add-on can export a rigged FBX with the SMPL skeleton + skinning weights built in.

Load the parameters (betas + body_pose) into a SMPL plugin — The Meshcapade Blender add-on lets you dial betas and pose directly. You'd import our consensus betas (10 floats) and body_pose (69 floats from consensus_results.json) and get a fully posable character.

The shape information (betas) is what captures this specific person's body proportions — height, shoulder width, hip ratio, etc. The body_pose (23 joint rotations) captures the T-pose articulation. Both are in output/debug/consensus/consensus_results.json if your mentor wants to load them into a DCC tool.

See todos below...

the next phase: Perspective-n-Point — given n known 3D points and their corresponding 2D projections in an image, solve for the camera's pose (rotation + translation).

in sequence:

Triangulated 3D points are used directly first — fit SMPL joints to match the triangulated 3D keypoint positions in 3D space. This is the coarse alignment: "move the SMPL skeleton so its joints match these known 3D positions." Fast, strong constraint.

Then reprojection refinement on top — once coarsely aligned, fine-tune by projecting back into all 2D views. This catches things triangulation misses: a triangulated point computed from 3 views might be slightly off, but the reprojection loss uses all views (including ones where that keypoint wasn't confident enough for triangulation) to pull it into better alignment.

So triangulation gives the initialisation target, reprojection gives the final polish. The REVIEW.md criterion 5.3 (MPJPE < 25mm) is measured against the triangulated points; criterion 5.4 ensures the reprojection loss uses all views including partial ones.

Triangulation (Phase 5, Step 2c): Now that we know where every camera was [R|t], we can combine the 2D keypoint observations from multiple views to compute precise 3D point positions. A keypoint seen in 3 views gives 3 "rays" from 3 known camera positions — where those rays intersect is the triangulated 3D point. More views = more accuracy. This gives us refined 3D keypoints that are better than any single view's estimate.

Reprojection optimisation (Phase 5, Step 2d): We adjust the SMPL parameters (betas, body_pose, translation) so that when you project the SMPL joints back into each image using that view's [R|t] + K, the projected points land on top of the detected 2D keypoints. The loss is literally "how many pixels off are my projected SMPL joints from where ViTPose/DenseKP says they should be?"

The power is that this uses all views simultaneously — including the partial/side views we excluded from HMR. A wrist that's occluded in one view is visible in three others. The optimiser finds the single SMPL configuration that best explains all observations across all cameras at once, which is a much stronger constraint than any single-view estimate.

PnP self-calibration recovers the camera extrinsics [R|t] for each view — where each camera was in 3D space relative to the subject.

Right now after Tier 1, we have:

A good consensus SMPL mesh (the 3D shape)
138 dense 2D keypoints per view (where surface points appear in each image)
But no idea where the cameras were — each view's SMPL estimate lives in its own camera coordinate frame
Phase 4 uses the SMPL mesh as a calibration target. For each view:

3D points: known vertex positions on the consensus SMPL mesh (138 dense keypoints map to specific SMPL vertices)
2D points: where DenseKP detected those same points in the image
solvePnPRansac: given these 2D-3D correspondences + the intrinsic matrix K, recover the camera's rotation and translation [R|t]
Once you have [R|t] per view, you can:

Triangulate — combine observations from multiple views to refine 3D joint/keypoint positions (Phase 5)
Reproject — project the SMPL mesh into all views (including the partial/side views we excluded from HMR) and optimise against all 17 images simultaneously
This is what drops MPJPE from ~40mm (Tier 1) toward <25mm (Tier 2)
The key insight is that 138 correspondences per view makes PnP extremely robust — RANSAC has massive redundancy compared to the 12 sparse COCO joints that traditional approaches use.

# Unit tests (no GPU, fast)
pytest tests/test_consensus.py -v

# Integration tests (GPU + Phase 2 output required)
pytest tests/integration/test_consensus_integration.py -v -m gpu


## TODOs

- remove smplx submodule - no longer needed for chumpy cleaning
- add an FBX export or blender-compatible format for a manipulatable SMPL with a skeleton 
