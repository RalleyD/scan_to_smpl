# Implementation Notes

Running log of implementation decisions, debug output, and Q&A from development sessions.

---

## Phase 0 — Scaffolding

Key files created:

| File | Purpose |
|------|---------|
| `.devcontainer/Dockerfile` | Python 3.10 + PyTorch 2.4 + CUDA 12.1 + all deps |
| `.devcontainer/devcontainer.json` | VSCode devcontainer config with GPU support |
| `pyproject.toml` | Package metadata, deps, pytest/ruff/mypy config |
| `scantosmpl/config.py` | Dataclass configs for all pipeline stages |
| `scantosmpl/types.py` | ViewType, CameraParams, ViewResult, FittingResult, SMPLOutput |
| `scantosmpl/smpl/model.py` | SMPL wrapper with differentiable forward pass + optimisable params |
| `scantosmpl/cli.py` | Click CLI skeleton (fit-images, fit-pointcloud, fit-combined) |
| `models/README.md` | Download instructions for all model files |
| `tests/test_smpl_model.py` | 10 tests covering all Phase 0 acceptance criteria |
| `utils/clean_smpl.py` | Chumpy removal utility |

```bash
# Clean SMPL pkl files (remove chumpy dependency)
python -m scantosmpl.utils.clean_smpl models/smpl/ --output models/smpl/
```

---

## Phase 1 — Keypoint Detection

```python
from scantosmpl.detection.pipeline import DetectionPipeline
from pathlib import Path

pipeline = DetectionPipeline(device='cuda')
results = pipeline.process_directory(
    Path('data/t-pose/jpg'),
    debug_dir=Path('output/debug/detection'),
)
print(f'\nProcessed {len(results)} images')
for r in results:
    n_vis = int((r.keypoint_confs > 0.3).sum()) if r.keypoint_confs is not None else 0
    print(f'  {r.image_path.name}: {r.view_type.value} ({n_vis}/17 kps)')
```

---

## Phase 2 — CameraHMR Integration

### Model prerequisites

| Model | Checkpoint | Output | Status |
|-------|-----------|--------|--------|
| Main HMR | `camerahmr_checkpoint_cleaned.ckpt` (7.5GB) | SMPL params (β, θ) + weak-perspective camera + 44 2D keypoints | ✓ |
| FLNet | `cam_model_cleaned.ckpt` | Focal length / FoV from image | ✓ |
| DenseKP | `densekp.ckpt` | 138 dense 3D surface keypoints | ✓ |

Note: The 138 dense keypoints come from the DenseKP model (separate from the main checkpoint). FoV estimation comes from FLNet.

### View exclusion results

| View | Spread | Torso frac | Excluded? | Reason |
|------|--------|-----------|-----------|--------|
| cam02_4 | 0.07 | 0.28 | yes | pure side view (spread < 0.12) |
| cam06_4 | 0.02 | 0.28 | yes | pure side view (spread < 0.12) |
| cam07_6 | 0.26 | 0.22 | yes | floor-up angle (torso < 0.23) |
| all others | ≥ 0.17 | ≥ 0.23 | no | — |

### Key implementation files

| File | Purpose |
|------|---------|
| `scantosmpl/hmr/camera_hmr.py` | CameraHMRInference: loads all three models, monkey-patches SMPL_MEAN_PARAMS_FILE, shared ViT-H backbone, CLIFF camera conversion, DenseKP keypoint denormalisation |
| `scantosmpl/hmr/orientation.py` | check_orientation_quality: upright check, rotation magnitude, T-pose arm check → score + warnings |
| `scantosmpl/hmr/pipeline.py` | HMRPipeline: orchestrates all views, PIL wireframe overlay, JSON + summary debug output |

### Submodule fix

Added CameraHMR as a git submodule (`external/CameraHMR`, master branch, commit b1b6eea).
Fixed upstream syntax error in `densekp_model.py` (`def forward(self, batch)` missing colon).

### Run commands

```bash
pip install -e ".[dev]"
pytest tests/test_hmr.py -v
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
```

---

## Phase 3 — Multi-View Consensus

### Global orient handling

Each view's `global_orient` encodes body rotation relative to that camera — they differ wildly across views (expected). Decision: **canonical zero** `[0,0,0]` for Tier 1 consensus.

Options considered:
- **(A) Most frontal view** — pick view with most symmetric shoulder spread. Simple, robust.
- **(B) Median-view selection** — cluster `global_orient` vectors and pick the medoid.
- **(C) Canonical zero** — set `global_orient` to `[0,0,0]` (identity). ✓ **chosen**

Rationale: Phase 4 (PnP) is the proper solution for recovering per-view orientation. Baking a "best-guess frontal" into Tier 1 would be noise that Tier 2 has to undo. Per-view `global_orient` values are preserved in `ViewResults` for Phase 4.

### Body pose aggregation

`body_pose` (69D = 23 joints × 3 axis-angle) is view-invariant in theory. For T-pose, rotations are small.

Options considered:
- **(A) SO(3) Fréchet mean per joint** — mathematically correct. ✓ **chosen**
- **(B) Weighted component-wise median** — simpler, works well for small rotations.

SO(3) averaging handles edge cases properly and is reusable for Tier 2.

### FoV consensus

EXIF focal lengths are physical measurements from camera lens metadata. FLNet is a neural network predicting what EXIF already tells us. For our scanner (Canon EOS 2000D), EXIF is strictly more reliable. Implementation: use EXIF focal lengths directly, report FLNet vs EXIF diff as a diagnostic, skip computing a "consensus FoV" unless EXIF is missing.

### Tests

```bash
pytest tests/test_consensus.py -v
pytest tests/integration/test_consensus_integration.py -v -m gpu
```

### Mesh output details

- Full topology: 6890 vertices, 13776 triangular faces (SMPL's fixed template topology)
- The `.obj` is a baked static mesh — no rig/joints/blend shapes embedded
- To animate: export as FBX with SMPL skeleton, or load `betas` + `body_pose` into Meshcapade Blender add-on

```bash
# View consensus mesh
sudo apt install meshlab && meshlab output/debug/consensus/consensus_mesh.obj
# Or: python3 -c "import trimesh; trimesh.load('output/debug/consensus/consensus_mesh.obj').show()"
```

---

## Phase 4+ Design Notes (PnP & Triangulation)

### PnP self-calibration overview

Uses the SMPL mesh as a calibration target. For each view:
- **3D points**: known vertex positions on the consensus SMPL mesh (138 dense keypoints map to specific SMPL vertices)
- **2D points**: DenseKP detections in the image
- `solvePnPRansac` → per-view `[R|t]`

138 correspondences vs 12 sparse joints is transformative for RANSAC robustness.

### Triangulation → reprojection sequence

1. **Triangulated 3D points first** — fit SMPL joints to match triangulated 3D keypoint positions (coarse alignment)
2. **Reprojection refinement** — once coarsely aligned, fine-tune by projecting back into all 2D views

Triangulation gives the initialisation target; reprojection gives the final polish.

### Noise propagation analysis

Tier 1 mesh has ~40mm error, so PnP `[R|t]` estimates are approximate. Three mitigations:

1. **Overdetermined PnP absorbs noise** — 138 correspondences for 6 unknowns; RANSAC averages out noise
2. **2D observations are the true anchor** — ViTPose/DenseKP detections don't depend on SMPL quality
3. **Reprojection is self-correcting** — the final loss is purely 2D; it iteratively adjusts until projections match

### Prerequisites

`downsample_mat.pkl` maps the dense 138 keypoints to SMPL vertices - The matrix is one-hot — each of the 138 keypoints maps to exactly one SMPL vertex. This means we can extract 138 vertex indices directly, which simplifies everything.

This comes from `tran-eval-utils` from 

[CameraHMR](camerahmr.is.tue.mpg.de)

### A/B comparison

The A/B comparison (Step 3, criterion 4.6) answers: "Do the 138 dense keypoints actually give better PnP results than the 12 sparse COCO joints?"

For the 10 dense views, we run PnP twice:

Dense: using all 138 surface keypoints (the normal path)
Sparse: using only the 12 COCO joint correspondences (same method we use for excluded views)
Then compare reprojection errors side-by-side on the same views.

What it solves: It validates our core design assumption — that CameraHMR's 138 dense keypoints provide meaningfully better camera pose recovery than sparse joints alone. If dense doesn't outperform sparse, it would mean either:

The dense keypoints are too noisy to help (questioning their value)
12 well-localised joints are sufficient (simplifying the pipeline)
In practice we expect dense to win clearly — 138 points spread across the body surface give RANSAC far more redundancy and geometric coverage than 12 joint locations. But we should prove it rather than assume it, since the entire rationale for using CameraHMR's dense keypoints in PnP (rather than just ViTPose joints) rests on this being true.

It's a one-time validation step, not something that changes the output — it just gives us confidence (or a warning) about the approach.

### denseKP failure - multi-view estimation

The 138 dense keypoints failed for PnP (surface vertices are too sensitive to pose averaging), but they still have value in Phase 5. Once we have refined camera poses and a better SMPL mesh, the dense keypoints become useful as reprojection targets — 138 points per view gives much richer loss signal than 12 sparse joints when optimizing SMPL params. They're bad as PnP input but good as optimization supervision.

PnP is absolutely the correct method — "given known 3D points and their 2D projections, find camera pose" is exactly our problem. The issue isn't PnP, it's the quality of our 3D calibration target. The consensus mesh has ~32mm error, which projects to ~50px — and that's exactly what we see (52px mean).

There's a chicken-and-egg situation:

- PnP needs good 3D points → but our best 3D points come from a coarse consensus
- Triangulation needs camera poses → but we need PnP to get those
  
The standard solution is exactly what we're doing: coarse-to-fine. Phase 4 gives initial [R|t] (coarse). Phase 5 uses those poses to triangulate, then jointly refines SMPL params AND camera poses in a loop. The 52px reprojection will tighten significantly in Phase 5 because the SMPL mesh gets optimized to explain ALL views simultaneously, not just averaged in parameter space.

The Phase 4 results are actually very encouraging — the camera geometry is physically plausible (1.5m distance, 309° coverage, multi-height tiers). That's strong evidence the camera poses are approximately correct, even if the reprojection error is high.

For PnP itself, sparse COCO joints are the right choice. The results confirm this (all 17 views solved via sparse).

- Sparse PnP actually works — it finds camera poses on all 7 sparse views AND all 10 dense views (A/B comparison shows 37-70px reprojection). The quality gate (30px) rejects them.
- Dense PnP gets only 10-15 inliers out of 138 even at 105px threshold — the dense 3D-2D correspondences have a fundamental quality issue (the consensus mesh's averaged body_pose causes vertex-level errors much larger than joint-level errors).
- Sparse uses well-localized joints (stable), dense uses surface vertices (sensitive to pose averaging).

Risk of coarse K for triangulation? Use Meshroom instead?

The K matrix (EXIF focal length) is actually our most reliable input — it's a physical measurement from the Canon EOS 2000D hardware, cross-validated by FLNet (mean diff 4.17°). K isn't coarse; it's the [R|t] that's coarse (~52px reprojection).

Meshroom extrinsics are an interesting idea but have practical issues:

Meshroom's coordinate system is arbitrary (scale, orientation, origin) — you'd need to align it to SMPL space anyway
It solves SfM from feature matching, which works best with textured scenes, not a person standing still
It would add a hard dependency on having a successful Meshroom reconstruction
The better path is what you already intuited: Phase 5 triangulation + joint refinement tightens the [R|t] iteratively. The Meshroom point cloud belongs in Tier 3 (surface refinement), where it gets aligned TO the SMPL mesh — not the other way around.

From the Phase 4 results, the cameras are ~1.5m from the subject (radial_mean = 1.5m). The EXIF focal length is ~6349px. So:

32mm / 1500mm * 6349px ≈ 135px

That's the worst case — where PnP can't absorb any of the error. In practice, PnP finds the best rigid transform which absorbs some of the systematic offset, bringing it down to the ~52px we actually observe.

The remaining 52px is the non-rigid error — differences in arm angle, torso twist, etc. between the consensus pose and what's actually visible in each image. A rigid camera transform can't fix those; only refining the SMPL mesh itself (Phase 5) can.

The current pipeline order is sound: EXIF K (reliable) → sparse PnP (coarse [R|t]) → triangulate + refine (tighten [R|t] and SMPL together).

Camera Centers Comparison (17 common views)
==========================================================================================
View                 PnP Center (X,Y,Z)                   COLMAP Center (X,Y,Z)               
------------------------------------------------------------------------------------------
cam01_2.JPG          ( -0.851,   0.419,   1.888)   (  0.890,   3.060,  -1.268)
cam01_6.JPG          ( -1.117,  -1.061,   1.208)   (  2.990,   1.958,   0.087)
cam02_4.JPG          (  1.265,  -2.708,   0.360)   ( -1.441,   1.893,   1.612)
cam02_5.JPG          ( -1.764,   0.134,   0.091)   (  0.799,   1.979,   2.469)
cam03_5.JPG          ( -0.974,  -0.162,  -1.667)   (  0.175,  -0.131,   3.580)
cam03_6.JPG          ( -1.084,  -1.030,  -1.189)   (  2.380,  -0.538,   3.246)
cam04_4.JPG          ( -0.209,   1.368,   1.730)   ( -2.446,  -1.478,   2.465)
cam04_5.JPG          (  0.053,  -1.118,  -1.064)   ( -0.233,  -2.480,   2.686)
cam05_4.JPG          (  1.215,   1.167,  -1.224)   ( -2.225,  -2.790,   0.394)
cam05_5.JPG          (  1.212,  -0.102,  -1.338)   ( -0.487,  -3.448,   0.507)
cam05_6.JPG          (  0.852,  -1.283,  -0.859)   (  1.884,  -3.363,   0.681)
cam06_4.JPG          (  0.967,   1.227,   0.023)   ( -1.981,  -1.446,  -1.202)
cam07_4.JPG          (  1.036,   1.183,   1.341)   ( -1.531,   0.076,  -2.816)
cam07_6.JPG          (  0.889,  -1.254,   1.152)   (  2.577,  -0.579,  -2.405)
cam10_2.JPG          ( -0.574,   1.416,   1.167)   ( -1.860,   2.481,  -0.677)
cam10_4.JPG          (  0.337,   1.465,   0.984)   ( -2.541,  -0.366,   2.274)
cam10_5.JPG          ( -0.579,   1.428,   1.017)   ( -2.841,  -1.787,   1.116)

Good agreement (within ~15°) — ~10 views:
cam07_6 (4.7°), cam07_4 (3.4°), cam04_5 (1.7°), cam05_6 (7.0°), cam03_6 (7.6°), cam01_2 (11.2°), cam10_2 (13.3°), cam01_6 (14.7°), cam02_5 (15.2°), cam05_5 (16.9°)

Completely flipped (~130-164°) — 4 views:
cam02_4, cam04_4, cam10_4, cam10_5 — these are placed on the OPPOSITE side of the subject from where COLMAP says they are.

Medium error (~25-30°) — 3 views:
cam03_5, cam05_4, cam06_4

The flipped views are telling — cam02_4 is a side view, cam10_4 and cam10_5 are rear views. These are exactly the views where ViTPose hallucinates face keypoints on the back of the head, giving the PnP solver mirrored correspondences. It "solves" PnP but places the camera on the wrong side.

The positive takeaway: ~10/17 views have camera poses that agree with COLMAP within 15°, using nothing but a coarse SMPL mesh and ViTPose joints. That's a strong foundation for Phase 5 triangulation — we can use those ~10 reliable views and discard or down-weight the flipped ones.

For Phase 5, we could add a consistency check: if a camera center is on the opposite side of the subject from where reprojection of its own keypoints would suggest, flag it. Or simply use the COLMAP extrinsics as ground truth for the views where they're available — you have 60 calibrated views from COLMAP, which is a much richer source than self-calibration.

Use COLMAP extrinsics when available. The complexity is low and the gain is substantial.

Why the complexity is low:

We just read COLMAP's images.bin in ~30 lines of Python — it's a simple binary format
You already have Procrustes alignment implemented (geometry.py)
The only new work is: read COLMAP cameras → Procrustes align to SMPL frame → done
Maybe 50-100 lines of code total (a colmap_reader.py utility)
Why the gain is high:

60 calibrated views instead of 17
No flipped cameras (4/17 of our self-calibrated views are ~180° wrong)
Sub-pixel accurate extrinsics vs ~52px reprojection error
Better triangulation → better SMPL refinement → the whole downstream pipeline improves
The architecture already supports this. CLAUDE.md defines --calibration ./calibration.json and --skip-self-calibration CLI flags. Self-calibration was designed for when you DON'T have extrinsics. You do have them — use them.

The practical design for Phase 5: try COLMAP extrinsics first, fall back to self-calibrated PnP if COLMAP isn't available. Phase 4's self-calibration isn't wasted — it validated that the approach works and gave us the cross-check we just ran, and it remains the fallback path for scans without SfM reconstruction.

Optional: re-run PnP with refined Tier 2 SMPL → re-triangulate → re-optimise (bundle adjustment).


---

## Glossary

PA-MPJPE (Procrustes-Aligned Mean Per-Joint Position Error)

This is from Phase 3. For each view, we compare that view's SMPL joints (24 joints) against the consensus SMPL joints. Before measuring the error, Procrustes alignment removes any rotation, translation, and scale difference — so the 32mm measures pure shape/pose disagreement, not camera angle differences.

32mm means: on average, each joint is 32mm away from where the consensus says it should be, after optimally aligning the two skeletons.

The reprojection error calculation

The question is: if a joint is 32mm off in 3D, how many pixels off will it appear in the image?


projected_error = (3D_error / distance_to_camera) * focal_length

---

## TODOs

- [ ] Remove smplx submodule — no longer needed for chumpy cleaning
- [ ] Add FBX export or Blender-compatible rigged format
