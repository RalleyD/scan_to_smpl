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

The current pipeline order is sound: EXIF K (reliable) → sparse PnP (coarse [R|t]) → triangulate + refine (tighten [R|t] and SMPL together).

Optional: re-run PnP with refined Tier 2 SMPL → re-triangulate → re-optimise (bundle adjustment).

---

## TODOs

- [ ] Remove smplx submodule — no longer needed for chumpy cleaning
- [ ] Add FBX export or Blender-compatible rigged format
