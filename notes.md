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


## Phase 5

in progress - specs ready:

### What's been developed in Phase 5

Phase 5 was originally spec'd (docs/phase5_spec.md, the file you have open) around COLMAP as the primary extrinsics source: parse COLMAP's cameras.bin/images.bin, run a 7-DoF Procrustes alignment to map COLMAP's arbitrary SfM frame into SMPL's canonical frame, undistort the 2D keypoints, triangulate 3D joints via DLT+RANSAC across views, then run a 3-stage SMPL optimiser (global alignment → shape → full pose) against those triangulated joints plus direct 2D reprojection. Self-calibration (cold PnP against SMPL joints, no COLMAP) was there too, but only as a fallback stub.

Since then, in the sessions leading up to today:

Reprojection metrics were reworked from mean to median (the original 15px target was unrealistic on a 6000px image, and rear-view ViTPose left/right swaps were blowing out the mean — see the note at the bottom of phase5_spec.md).

Rear-view classification/exclusion was added so those swapped-keypoint views get filtered out of the reprojection loss entirely rather than just averaged down.

A further feature was attempted on top of COLMAP: iteratively refining the COLMAP camera poses via PnP against the SMPL joints ("Option A+B"). It was implemented, tested against real data, and made things worse — only 3-4 of 7 frontal cameras actually refined, reprojection error went up, PA-MPJPE regressed to 25-27mm. That's documented in the master spec as the risk materializing: cameras absorbing SMPL joint error instead of correcting real drift. It was reverted.

A controlled experiment then compared Phase 4's existing cold self-calibration directly against COLMAP+Procrustes on the same real dataset, and self-cal won outright: 23.99mm vs 24.46mm PA-MPJPE, 78.8px vs 135.2px median reprojection.

Today's feature (selfcal-default-extrinsics) acted on that result: it fully retired COLMAP from Phase 5 — deleted the COLMAP reader, frame-alignment, and undistortion modules, stripped the now-dead config fields (extrinsics_source, colmap_model_dir, the abandoned PnP-refinement knobs), simplified Phase5Pipeline/Phase5Config/Phase5Result down to a single code path, and rewrote the integration tests around a self-cal-only fixture.

### The milestone

Phase 5 (Tier 2: self-calibration + refinement) is now a complete, coherent, genuinely calibration-free pipeline stage — matching the project's core CLAUDE.md premise ("no camera extrinsics required, human body as calibration target"). There's no more COLMAP dependency, no more SfM reconstruction step, no more coordinate-frame bridging between two different calibration systems. The chain is now: Tier 1 consensus (per-view HMR) → Phase 4 self-calibration (PnP) → Phase 5 triangulation + staged SMPL refinement, entirely self-contained.

Measured on your real 17-camera t-pose dataset: PA-MPJPE ~22.5mm (refinement) / ~25mm (triangulation-vs-consensus), median reprojection ~77px — beating every acceptance threshold with margin, and better than the old COLMAP path ever achieved. All 10 acceptance criteria for this cleanup passed on the first review iteration, lint/typecheck clean, full GPU integration suite green.

What's not done: Tier 3 (point-cloud/surface refinement) is still an empty package — that's the next tier, not started. And everything from today is still uncommitted, per our last exchange.

### PnP — what it is and why it's here

Perspective-n-Point: given N known 3D points and their corresponding 2D pixel locations in an image (plus known camera intrinsics — focal length, principal point), recover the camera's pose: the 6-DOF rotation + translation that would make those 3D points project to exactly those 2D pixels. It's the classic "solve for where the camera must have been standing" problem.

In this pipeline, the 3D points are the SMPL/consensus body-joint positions (already estimated from Tier 1, in a canonical body-centered frame) and the 2D points are where ViTPose detected those same joints in each photo. Solving PnP per view recovers a camera pose without any external calibration rig — the body itself becomes the calibration target, hence "self-calibration."

### RANSAC — what it is and why it's here

RANdom SAmple Consensus: a way to fit a model when some of your data is wrong (outliers) and you don't know which points those are in advance. Instead of fitting to all points at once, it repeatedly: samples a small random subset, fits the model to just that subset, then checks how many of the remaining points agree with that fit within some tolerance ("inliers"). Whichever random subset produces the fit with the most inliers wins; often a final fit is redone using all inliers together.

Here it's used twice: solvePnPRansac (Phase 4) tolerates a few badly-wrong 2D-3D correspondences without corrupting the whole camera pose — important because ViTPose genuinely does produce garbage on some views (the left/right swap on rear-facing cameras). And again in DLT triangulation (Phase 5), across views rather than points — if 2 of 7 cameras disagree wildly with the rest on where a joint is in 3D, RANSAC discards those 2 rather than letting them drag the triangulated position off.

### Does self-cal PnP produce artificially good numbers by absorbing camera drift?

This is a sharp question, and the honest answer is: partially yes, and it's worth being clear-eyed about it rather than just trusting the numbers.

The specific failure mode you're describing — a PnP solve quietly compensating for bias in the 3D points it's given, rather than reporting a real discrepancy — is exactly the risk the master spec documented for the abandoned PnP-camera-refinement feature, and it's exactly what happened when that feature was tested (cameras drifted to chase noisy SMPL estimates, reprojection got worse, PA-MPJPE regressed). So this isn't a hypothetical concern for this codebase — it's a documented, previously-observed failure mode.

The current (accepted) self-cal path is a somewhat different situation, though, and it's worth separating two things:

- Coherent/systematic error (e.g. the whole consensus mesh is a few mm too short, or globally rotated slightly) — a single 6-DOF camera pose can and likely will absorb this kind of error per view, because a rigid transform is exactly the right shape to cancel out a rigid bias. None of the current metrics (reprojection error, PA-MPJPE-vs-triangulation) would catch this, because the whole system — consensus mesh, self-cal cameras, triangulated joints, refined SMPL — would simply be self-consistent around the same bias. This is a real structural blind spot: there's currently no fully independent ground truth in the loop (no calibrated mocap rig, no physically measured body dimensions, and COLMAP — the one independent geometric reference that existed — has just been removed).

- Per-point/per-view incoherent noise (individual joint estimation error, individual bad ViTPose detections) — this is not something a single camera pose can fully absorb, because it doesn't look like a rigid transform; and RANSAC explicitly discards the worst offenders rather than quietly folding them in. Multi-view triangulation adds more protection here too: a triangulated joint has to be geometrically consistent with rays from several independently self-calibrated cameras, which is a real overdetermination check for this class of error.

So: the reprojection-error and PA-MPJPE numbers you're seeing should be read as measuring internal consistency (do the 2D detections, the recovered cameras, and the fitted 3D body agree with each other) rather than absolute real-world accuracy. They're genuinely useful for catching per-joint noise and bad views, genuinely good evidence that the pipeline isn't falling apart — but they can't, by construction, catch a coherent bias shared across the whole system.

The one piece of evidence that argues against gross circularity is the A/B result against COLMAP: COLMAP's camera poses have zero knowledge of the SMPL mesh (pure SIFT/SfM), so if self-cal were purely "cheating" by absorbing large consensus error, you might expect it to diverge more from an independent reference, not less. But that comparison is also confounded — COLMAP itself had known problems on this dataset (4/17 views ~180° wrong per the original Phase 4 notes), so beating a flawed baseline is better read as "self-cal is more internally consistent than bolting together two mismatched coordinate systems" than as "self-cal is independently proven accurate."

**If you want a genuine independent check later: **

The natural candidate (per the master spec's §9 notes) is Tier 3: aligning the photogrammetry point cloud to the SMPL mesh via ICP gives you a geometry source that was never involved in any of the self-calibration — that would be real, independent validation in a way nothing in the current pipeline provides.

### Training / optimisation process in Phase 5

First, an important framing point: this isn't "training" in the sense you'd use for a neural network learning from a big dataset. SMPL itself — the mapping from parameters (β, θ) to a 3D mesh — is a fixed, pre-learned model; nobody is updating its internal weights here. What Phase 5 does is closer to test-time optimisation (also called "analysis by synthesis" or "inverse rendering"): for this one scan, find the specific (β, θ, translation, scale) values that make SMPL's output match what the cameras actually observed. Same underlying machinery as neural net training (gradients, backprop, an optimiser), applied to a much smaller problem — fitting ~79 numbers to one example, not millions of weights to a dataset.

**The computational graph, forward direction:**

SMPL params (β, θ, translation, scale)
   → SMPL forward pass (differentiable) → 3D joints/vertices
   → project through each camera's fixed [R|t|K] → 2D pixel coordinates
   → compare to observed data → loss (a single number)
Backpropagation is just the chain rule run backwards through that graph: given the loss, compute ∂loss/∂β, ∂loss/∂θ, ∂loss/∂translation, ∂loss/∂scale — i.e., "if I nudge this parameter slightly, does the loss go up or down, and by how much." PyTorch builds the graph automatically as the forward pass runs, then .backward() walks it in reverse to get every gradient in one pass. The optimiser (Adam, per CLAUDE.md) then uses those gradients to actually update the parameters — a bit at a time, over many iterations, hopefully descending toward lower loss each step.

The loss being minimised (from phase5_spec.md §6), the sum of four terms with per-stage weights:

- L_joint — Huber distance between SMPL's own 3D joints and the triangulated 3D joints from Step 3 (DLT+RANSAC). "Does the body's 3D shape match the 3D points we reconstructed from multiple cameras?"
- L_reproj (reprojection loss) — project SMPL's 3D joints through each camera into 2D, compare to the actual ViTPose 2D detection in that image, weighted by ViTPose's confidence, summed across all views. "Does the body, seen from every camera, land where the 2D detector actually saw it?"
- L_pose_prior — penalises θ for straying from a plausible human pose. Pure regularisation — stops the optimiser exploiting a contorted pose to cheat the other losses.
- L_shape_reg — penalises β for straying from the mean body shape. Same idea, for body shape.

- Huber loss (used inside L_joint/L_reproj instead of plain squared error): behaves like L2 (squared error) for small residuals — smooth gradients, good behaviour — but switches to linear (L1-like) for large residuals. That means one badly-wrong point (a swapped rear-view keypoint) contributes a bounded amount of gradient instead of a squared, gradient-dominating amount. This is the direct fix for the "single outlier poisons everything" failure mode you were reading about in the supplement.

- Loss-weight annealing — the w_joint/w_reproj/etc. coefficients change across the three stages (not the model, the loss recipe): Stage 1 uses almost only L_joint to get rough global position right fast; by Stage 3, w_joint has dropped to 0.1 and reprojection dominates, so the fine detail comes from the real 2D evidence rather than the (noisier) triangulated points. This is a curriculum, coarse-to-fine.

- Iterations — each stage runs many steps of (forward pass → loss → backward pass → optimiser step), e.g. 50, 100, and up to 400 for the three stages. loss_history records the loss at each step; convergence.png plots it.

### What is 6 DoF?

Degrees of freedom — the number of independent numbers needed to fully describe something's position/configuration. For a rigid object in 3D space (like a camera), that's:

3 for translation: where it is (x, y, z)

3 for rotation: which way it's facing (commonly parameterised as roll/pitch/yaw, or a quaternion's effective 3 free parameters, or in this codebase axis-angle)
= 6 total. 

A camera's [R|t] — the exact thing Phase 4's PnP solves — is precisely this: R is a 3×3 matrix but constrained to be a valid rotation (orthonormal, determinant 1), so it only carries 3 free numbers; t is a plain 3-vector. Contrast this with a human body: SMPL's pose alone has ~72 numbers (24 joints × 3 axis-angle parameters each) plus shape and translation/scale on top — a vastly higher-dimensional, articulated thing, not a single rigid 6-DOF transform. That distinction is exactly why a single camera pose can absorb a rigid/coherent bias but can't absorb per-joint articulated error — which ties back to the drift question from before.

### Clarifying the chain: is 3D refinement done via camera-pose refinement?

Good instinct to double check this, but the current (shipped) pipeline does not work that way — and it's worth being precise here because it's easy to blend this with the abandoned PnP-refinement feature we discussed. The actual chain:

- Phase 4 (once, upfront): solve each camera's [R|t] via PnP, using the original consensus 3D joints + 2D ViTPose keypoints. Cameras are now fixed — nothing after this point ever changes them again in the shipped pipeline.
- Phase 5, triangulation (once, before the optimiser runs): using those fixed cameras + the 2D keypoints, compute fresh 3D joint positions via multi-view DLT+RANSAC — this is a separate geometric computation from the earlier consensus joints, purely intersecting 2D rays through known camera positions.
- Phase 5, SMPL refinement (the iterative, gradient-descent part): adjust the SMPL body's parameters (β, θ, translation, scale) — not the cameras — so that the body matches (a) the triangulated 3D joints from step 2, and (b) the 2D ViTPose detections when reprojected through the (still-fixed) cameras.

So to directly answer: the thing being refined is the SMPL body, not the camera poses. Camera-pose refinement from 2D correspondences is exactly what the abandoned "Option A+B" feature tried to bolt on afterward (re-solving PnP using the refined SMPL joints to nudge the cameras) — and that's the piece that got reverted because it let cameras absorb SMPL error instead of correcting real drift.

**As for reprojection's role — it's actually both of the things you asked about, just at different times:**

- During optimisation, L_reproj is one of the two active loss terms directly driving gradient descent — it's not a passive check, it's baked into what backprop computes gradients from at every single step. It's steering the fit.
- After optimisation finishes, the exact same computation (project the final refined joints, compare to the 2D detections) is reused as a reported quality metric (median_reproj_px in Phase5Result.metrics) — in that role, yes, it's verification: "how well did the fit turn out."
What it never does in the current pipeline is verify or adjust the camera's pose — the camera is a fixed input throughout Phase 5, not something reprojection error ever feeds back into.

### Disproven: Use of PnP RANSAC and Justification for "Option A+B"

The PnP log shows per-camera reprojection at 37-72 px. This is essentially the ViTPoser noise floor. Compared to the Phase 5 median of 137 px, the gap is ~80 px of camera calibraiton drift that COLMAP+Procrustes can't fix:

```
26-06-07 21:55:31 WARNING PnP dense_138: RANSAC too few inliers (inliers=13, need=20)
26-06-07 21:55:31 INFO cam01_2.JPG: dense PnP failed (13 inliers), falling back to sparse
26-06-07 21:55:31 INFO cam01_2.JPG: OK (dense_sparse_fallback, inliers=10/14, reproj=65.7px)
26-06-07 21:55:31 INFO cam01_6.JPG: OK (sparse_coco, inliers=7/14, reproj=64.1px)
26-06-07 21:55:31 INFO cam02_4.JPG: OK (sparse_coco, inliers=7/13, reproj=37.5px)
26-06-07 21:55:31 WARNING PnP dense_138: RANSAC too few inliers (inliers=10, need=20)
26-06-07 21:55:31 INFO cam02_5.JPG: dense PnP failed (10 inliers), falling back to sparse
26-06-07 21:55:31 INFO cam02_5.JPG: OK (dense_sparse_fallback, inliers=6/12, reproj=38.4px)
26-06-07 21:55:31 WARNING PnP dense_138: RANSAC too few inliers (inliers=13, need=20)
26-06-07 21:55:31 INFO cam03_5.JPG: dense PnP failed (13 inliers), falling back to sparse
26-06-07 21:55:31 INFO cam03_5.JPG: OK (dense_sparse_fallback, inliers=7/14, reproj=43.1px)
26-06-07 21:55:31 WARNING PnP dense_138: RANSAC too few inliers (inliers=14, need=20)
26-06-07 21:55:31 INFO cam03_6.JPG: dense PnP failed (14 inliers), falling back to sparse
26-06-07 21:55:31 INFO cam03_6.JPG: OK (dense_sparse_fallback, inliers=8/13, reproj=63.6px)
26-06-07 21:55:31 WARNING PnP dense_138: RANSAC too few inliers (inliers=15, need=20)
26-06-07 21:55:31 INFO cam04_4.JPG: dense PnP failed (15 inliers), falling back to sparse
26-06-07 21:55:31 INFO cam04_4.JPG: OK (dense_sparse_fallback, inliers=8/14, reproj=62.0px)
26-06-07 21:55:31 WARNING PnP dense_138: RANSAC too few inliers (inliers=11, need=20)
26-06-07 21:55:31 INFO cam04_5.JPG: dense PnP failed (11 inliers), falling back to sparse
26-06-07 21:55:31 INFO cam04_5.JPG: OK (dense_sparse_fallback, inliers=8/14, reproj=38.4px)
26-06-07 21:55:32 WARNING PnP dense_138: RANSAC too few inliers (inliers=11, need=20)
26-06-07 21:55:32 INFO cam05_4.JPG: dense PnP failed (11 inliers), falling back to sparse
26-06-07 21:55:32 INFO cam05_4.JPG: OK (dense_sparse_fallback, inliers=8/14, reproj=44.9px)
26-06-07 21:55:32 WARNING PnP dense_138: RANSAC too few inliers (inliers=12, need=20)
26-06-07 21:55:32 INFO cam05_5.JPG: dense PnP failed (12 inliers), falling back to sparse
26-06-07 21:55:32 INFO cam05_5.JPG: OK (dense_sparse_fallback, inliers=7/12, reproj=47.7px)
26-06-07 21:55:32 INFO cam05_6.JPG: OK (sparse_coco, inliers=6/14, reproj=52.8px)
26-06-07 21:55:32 INFO cam06_4.JPG: OK (sparse_coco, inliers=6/14, reproj=47.5px)
26-06-07 21:55:32 WARNING PnP dense_138: RANSAC too few inliers (inliers=12, need=20)
26-06-07 21:55:32 INFO cam07_4.JPG: dense PnP failed (12 inliers), falling back to sparse
26-06-07 21:55:32 INFO cam07_4.JPG: OK (dense_sparse_fallback, inliers=10/14, reproj=45.1px)
26-06-07 21:55:32 INFO cam07_6.JPG: OK (sparse_coco, inliers=8/14, reproj=47.6px)
26-06-07 21:55:32 WARNING PnP dense_138: RANSAC too few inliers (inliers=10, need=20)
26-06-07 21:55:32 INFO cam10_2.JPG: dense PnP failed (10 inliers), falling back to sparse
26-06-07 21:55:32 INFO cam10_2.JPG: OK (dense_sparse_fallback, inliers=8/14, reproj=48.0px)
26-06-07 21:55:32 INFO cam10_4.JPG: OK (sparse_coco, inliers=10/14, reproj=54.4px)
26-06-07 21:55:32 INFO cam10_5.JPG: OK (sparse_coco, inliers=11/14, reproj=45.7px)
26-06-07 21:55:32 WARNING PnP dense_138: RANSAC too few inliers (inliers=14, need=20)
26-06-07 21:55:32 INFO cam01_2.JPG: dense PnP failed (14 inliers), falling back to sparse
26-06-07 21:55:32 INFO cam01_2.JPG: OK (dense_sparse_fallback, inliers=10/14, reproj=59.8px)
26-06-07 21:55:32 INFO cam01_6.JPG: OK (sparse_coco, inliers=9/14, reproj=56.4px)
26-06-07 21:55:32 INFO cam02_4.JPG: OK (sparse_coco, inliers=7/13, reproj=55.5px)
26-06-07 21:55:32 WARNING PnP dense_138: RANSAC too few inliers (inliers=11, need=20)
26-06-07 21:55:32 INFO cam02_5.JPG: dense PnP failed (11 inliers), falling back to sparse
26-06-07 21:55:32 INFO cam02_5.JPG: OK (dense_sparse_fallback, inliers=7/12, reproj=68.1px)
26-06-07 21:55:32 WARNING PnP dense_138: RANSAC too few inliers (inliers=12, need=20)
26-06-07 21:55:32 INFO cam03_5.JPG: dense PnP failed (12 inliers), falling back to sparse
26-06-07 21:55:32 INFO cam03_5.JPG: OK (dense_sparse_fallback, inliers=7/14, reproj=45.3px)
26-06-07 21:55:33 WARNING PnP dense_138: RANSAC too few inliers (inliers=12, need=20)
26-06-07 21:55:33 INFO cam03_6.JPG: dense PnP failed (12 inliers), falling back to sparse
26-06-07 21:55:33 INFO cam03_6.JPG: OK (dense_sparse_fallback, inliers=8/13, reproj=48.7px)
26-06-07 21:55:33 WARNING PnP dense_138: RANSAC too few inliers (inliers=17, need=20)
26-06-07 21:55:33 INFO cam04_4.JPG: dense PnP failed (17 inliers), falling back to sparse
26-06-07 21:55:33 INFO cam04_4.JPG: OK (dense_sparse_fallback, inliers=8/14, reproj=57.2px)
26-06-07 21:55:33 WARNING PnP dense_138: RANSAC too few inliers (inliers=10, need=20)
26-06-07 21:55:33 INFO cam04_5.JPG: dense PnP failed (10 inliers), falling back to sparse
26-06-07 21:55:33 INFO cam04_5.JPG: OK (dense_sparse_fallback, inliers=8/14, reproj=37.3px)
26-06-07 21:55:33 WARNING PnP dense_138: RANSAC too few inliers (inliers=11, need=20)
26-06-07 21:55:33 INFO cam05_4.JPG: dense PnP failed (11 inliers), falling back to sparse
26-06-07 21:55:33 INFO cam05_4.JPG: OK (dense_sparse_fallback, inliers=9/14, reproj=38.5px)
26-06-07 21:55:33 WARNING PnP dense_138: RANSAC too few inliers (inliers=13, need=20)
26-06-07 21:55:33 INFO cam05_5.JPG: dense PnP failed (13 inliers), falling back to sparse
26-06-07 21:55:33 INFO cam05_5.JPG: OK (dense_sparse_fallback, inliers=7/12, reproj=48.3px)
26-06-07 21:55:33 INFO cam05_6.JPG: OK (sparse_coco, inliers=6/14, reproj=36.6px)
26-06-07 21:55:33 INFO cam06_4.JPG: OK (sparse_coco, inliers=6/14, reproj=52.5px)
26-06-07 21:55:33 WARNING PnP dense_138: RANSAC too few inliers (inliers=13, need=20)
26-06-07 21:55:33 INFO cam07_4.JPG: dense PnP failed (13 inliers), falling back to sparse
26-06-07 21:55:33 INFO cam07_4.JPG: OK (dense_sparse_fallback, inliers=9/14, reproj=42.0px)
26-06-07 21:55:33 INFO cam07_6.JPG: OK (sparse_coco, inliers=8/14, reproj=55.1px)
26-06-07 21:55:33 WARNING PnP dense_138: RANSAC too few inliers (inliers=10, need=20)
26-06-07 21:55:33 INFO cam10_2.JPG: dense PnP failed (10 inliers), falling back to sparse
26-06-07 21:55:33 INFO cam10_2.JPG: OK (dense_sparse_fallback, inliers=9/14, reproj=72.3px)
26-06-07 21:55:33 INFO cam10_4.JPG: OK (sparse_coco, inliers=10/14, reproj=53.7px)
26-06-07 21:55:33 INFO cam10_5.JPG: OK (sparse_coco, inliers=10/14, reproj=57.1px)
```

**Where the drift comes from:**

COLMAP's [R|t] is sub-pixel accurate for COLMAP's own features (SIFT/SfM keypoints on textured surfaces). ViTPoser detects body joints, not SIFT features. This is a systematic offset between where COLMAP thinks the camera is and where it would need to be for ViTPose joints to project correctly (think surface vs joint offsets). Procrustes alignment then transforms cameras into SMPL frame, but doesn't fix per-camera drift - it just rigidly rotates or transforms the whole set.

PnP refinement fixes this directly, it takes the refined 3D SMPL joints (which are now trusted at Phase 5 convergence) and the 2D ViTPose detections and asks `cv2.solvePnPRansac` to re-derive each camera's [R|t] to make these correspondences consistent. Per-camera drift gets absorbed into the per-camera [R|t] adjustments. 

Option A+B desribes this, from the spec supplement doc, confirmed by the PnP log (~50 px is achievable per-view):

1) Implement A+B: post-refinement PnP pass over each camera, using `refined.joints` (3D) and the undistorted ViTPose landscape keypoints (2D). The existing phase 4 PnP module `pnp_solver.py` could be reused. An iterative approach:
   1) A single-pass PnP refines cameras. Yielding a cleaner reprojection metric, but the SMPL joints don't change (refinement has already happened). So PA-MPJPE is unchanged.
2) re-compute metrics with the refined cameras to confirm median reprojection drops from 137 px to ~50-70 px. 
   1) Re-run the optimiser with the PnP-refined cameras. The reprojection-loss has a cleaner signal (~50-70 px, instead of ~140 px). The optimiser can use the reprojection term meaninfully. Currently every gradient is in the liner reginme, huber delta = 20. With lower reprojection there should be useful quadratic shape to the loss. Therfore PA-MPJPE should drop.

Cameras therefore, get small and plausible adjustments that abosrb COLMAP to ViTPose calibration drift.

**The caveat:**

PnP is improving the reprojection metric by each camera absorbing error to satisfy the metric. Given the scenario: giving PnP enough freedom that it fits the camera to whatever 3D joints are present, even if those joints are wrong. The reprojection becomes artificially good but the cameras have moved out of their true positions. This requires a guard i.e `pnp_refine_max_translation_m: 0.1` - bounding how far cameras can drift from their COLMAP-derived [R|t] - keeping the translation honest.

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
