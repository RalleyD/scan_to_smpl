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

**A note on the results - summary.txt and pytest**

summary.txt is legacy. This could be patched to include the summary from pytest.

The pytest integration suite will capture detailed metrics like median joint reprojection error in pixels.

These are captured in `refinement_results.json`

Reprojection quality: median=77.09px, mean_inliers=214.28px

Also in refinement_results.json's metrics dict — median_reproj_px: 77.09.


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

## Phase 5 - tier 2 - improvement plan

**integration tests**:

```
.venv/bin/python -m pytest tests/integration/test_phase5_integration.py -v -m gpu
```

**rear view classification**

```
.venv/bin/python -m pytest tests/test_rear_views.py -v
```

**visualise refinement comparison - head and torso alignment**


```
.venv/bin/python -m scantosmpl.evaluation.visualise --views cam01_2.JPG,cam10_2.JPG,cam10_5.JPG
```

**leave one view out - diagnostic**

```
.venv/bin/python -m scantosmpl.evaluation.leave_one_view_out
```

### results

are profile views the culprit?
Not profiles as a class — two specific broken profile views. Your LOVO table is the evidence. The key column is the gap (held-out minus in-sample):

view	angle	in-sample	held-out	gap	read
cam06_4	profile	324	328	4.5	fit can't explain it, doesn't rely on it → genuine outlier
cam02_5	profile	157	172	15	poorly explained, low influence → weak outlier
cam02_4	profile	51	53	2.2	well explained, well behaved → good view
cam01_6	three_quarter	58	99	41	big gap = influential, the fit leans on it → keep
So: cam02_4 is a profile and one of your best views (51px). cam06_4 is a profile and a broken one (324px — a genuine ViTPose failure / left-right swap). A small gap + high absolute error = "the geometry disagrees with every other camera no matter what," which is the LOVO diagnostic's whole reason to exist.

That means my blanket W3 rule (profile → 0.3) is too blunt in both directions: it muzzles the good profile (cam02_4) while the two broken ones still leak in at 0.3× (huber-capped, so contained, but not silenced). The plan actually anticipated this — the optional W3 bullet was "per-term gross-outlier rejection (drop > N px)." Your data now argues that the per-view outlier drop matters more than the angle-class weight.

Your two visual defects, decomposed
Thin torso — this is the deferred Tier-3 girth gap, exactly as the plan predicted. Joint losses only constrain joint centres; nothing constrains girth, so betas wander. Two caveats worth knowing:

The flip you saw (original overspill/wider → now thinner) just confirms girth is unconstrained and drifting run-to-run — not a Tier-2 bug, and no Tier-2 knob fixes it properly. It needs the chamfer anchor.
There's also a metric artifact feeding this: profile (side) views are the only ones constraining torso front-back depth. Down-weighting them (W3) removes depth-position constraint, which can read as a thinner/shallower torso in a side overlay even though the frontal fit is fine.
Chest sits higher — this one is more likely tied to my W2 head term. Anchoring head(15) to the ViTPose ears-midpoint, if the detected ears sit a little high, pulls the head up and drags the upper spine/neck with it through the pose prior. This is a fit effect, not girth, and it's the thing I'd want to A/B before trusting.

The one test failure — honest read
test_triangulation_accuracy failed on PA-MPJPE 24.78mm vs the <24.5mm bar — 0.28mm over a deliberately tight threshold. Everything else passed (median reproj 80.6px < 90 ✓). I won't paper over it by loosening the threshold. Two things are tangled and I don't want to guess which dominates:

Metric artifact: that PA-MPJPE is measured against triangulated joints, and triangulation uses all views (profiles included). Down-weighting profiles in refinement moves the fit away from the profile-inclusive triangulation → this specific number can go up even if the fit against the real body is no worse.
A real perturbation from W2/W3 shifting the solution.
I can separate these cleanly and cheaply — a refit-only A/B runs off the cached debug artifacts (like the LOVO script), ~14s per config, no GPU HMR re-run.

"Depth" is two different quantities
1. Surface depth = girth / sagittal thickness (belly-to-back distance).
You're exactly right here. A joint centre sits on the body's medial axis; the surface is offset from it by the soft-tissue radius. Joint reprojection never observes that radius — from any view, profile or frontal. So no joint term refines girth. The "thin torso" is unreachable in Tier 2 and is a pure chamfer/Tier-3 job. Agreed, full stop.

2. Skeletal depth = the sagittal (front-back) position of the joint centres = posture / lean.
This one is observable from joints, and profile views are its strongest input. Geometry: a frontal camera (looking down the body's front-back axis) pins each joint's left-right X and up-down Y but is weak on front-back Z — that's along its optical axis. A profile camera pins Y and Z but is weak on X. So the front-back placement of the head, neck, and spine — i.e. the "head protrudes forward" / "chest forward" family — is precisely what profile (and, partially, three-quarter) views constrain. Frontal views alone can't.

So:

quantity	observable from joints?	who constrains it	your defect
torso girth (surface thickness)	no	chamfer only → Tier 3	"thin torso"
joint sagittal position (posture/lean)	yes	profile + 3/4 views	"head/chest forward"
The implication for your question
"Refining depth out" is the wrong move if it means dropping profiles wholesale — you'd lose the posture/lean constraint (which is the head-forward defect W2 is trying to fix) while doing nothing for girth (which was never on the table for joints). The thing that's broken in cam06_4 isn't "profile-ness" — it's that its keypoints are wrong (324px, detector failure). The good profile cam02_4 (51px) carries genuinely useful sagittal-depth signal you want to keep.

That's the argument for targeted outlier rejection over my current blanket profile → 0.3: reject broken keypoints, not profile geometry.

One nuance that partly defuses the whole worry
Profile depth info isn't actually thrown away by W3 anyway. Triangulation uses all views at full weight (including profiles) to build the 3D joint targets; W3 only re-weights the 2D reprojection term in the final refinement. So the profile-derived sagittal positions are already baked into the joint targets the optimiser fits toward — W3 just changes how much the last-stage reprojection additionally trusts profile keypoints.

What this means for the A/B
It's still worth running — but recalibrate what it can explain:

Chest-height + PA-MPJPE: yes, the A/B will cleanly attribute these to W2 (head term) vs W3 (weighting). Worth doing.
Torso thinness: the A/B will not explain it. Girth is betas-driven and only weakly regularised, so a "torso-width" column would mostly show run-to-run RANSAC/beta wander, not a W2/W3 effect. I'd be misleading you if I sold the A/B as diagnosing the thinness — that's Tier-3, and no Tier-2 knob moves it honestly.
So my read: your instinct is right on girth (Tier-3), but don't generalize it to "profiles are useless" — good profiles are the only thing pinning posture/lean, which is a different defect that Tier 2 genuinely owns.

Given that, do you want me to still run the A/B (now scoped to chest-height + PA-MPJPE attribution only, torso-width dropped), or skip straight to targeted outlier rejection since the reasoning already favors keeping good profiles and dropping broken-keypoint ones?

   ```
   .venv/bin/python -m scantosmpl.evaluation.ab_refit
   ```

Don't demote to W3. Fix W2's anchor so it corrects head-forward without the upward overshoot: match the 2D ears-midpoint to the reprojection of the 3D ear-vertex midpoint (which rotates with the head) instead of to joint 15. That removes the systematic lift, should pull head_up/pitch back toward baseline, and should shrink the PA-MPJPE cost too.

Concretely I'd:

Verify the bias first — compute the 3D offset (ear-vertex midpoint − joint 15) on the SMPL template; I expect ~+several cm in the up axis. That confirms the mechanism before we touch code. (Needs a model load — one small script you'd run.)
Implement vertex-anchoring — extend reprojection_loss to accept vertex-index correspondences (project SMPL vertices, not just joints), and point the head term at the ear-vertex midpoint.
That keeps the head fix you asked for, kills the "back and up" artifact, and likely gets PA-MPJPE back under threshold on its own merits.

### verification - A/B working off cached artifcats (JSON output etc)

On the neutral SMPL template, the ears-midpoint vertex sits +6.7 cm above and −3.0 cm behind head joint 15 (tests/test_head_anchor.py::test_ear_vertex_sits_above_and_behind_head_joint, passing). The old W2 term matched the 2D ears to the projection of joint 15, so satisfying it necessarily lifts and tilts the head back — the "up and back" you saw in cam10_2. The earlier validation missed this because it only checked the fore-aft (Z) and lateral (X) axes, never the vertical.

The 2D ears-midpoint now anchors to the projection of the 3D ear-vertex midpoint (verts 583/4071), which rotates with the head, so there's no joint-15 bias.

joint_map.py — HEAD_MIDPOINT_TO_VERTEX + ear-vertex ids; old HEAD_MIDPOINT_TO_SMPL kept only for the A/B baseline.
losses.py — reprojection_loss gained vertices_pred + vertex_midpoint_to_smpl (projects the vertex midpoint; existing joint path untouched).
optimiser.py — vertex head term is now the default; joint anchor off by default.


config	PA-MPJPE	medReproj	head_up Δ	head_pitch Δ
baseline (no head)	22.56	71.0	—	—
W2_joint (old, biased)	25.00	74.9	+2.02 cm	−17.8°
W2_vertex (new fix)	22.87	69.8	+0.99 cm	−2.7°
W2v_W3 (fix + profile 0.3)	23.55	71.9	+0.92 cm	−1.3°

The fix cuts the PA-MPJPE cost from +2.44 mm → +0.31 mm, turns the reprojection penalty into a small improvement (−1.2 px), and collapses the head over-rotation from −17.8° to −2.7°.

End-to-end (the real thing): re-ran the full pipeline (fresh HMR → consensus → cold PnP → refine) with the new default. The previously-failing AC7 test now passes: Refinement PA-MPJPE = 23.55 mm < 24.5 mm, and it matches the A/B's W2v_W3 prediction to the decimal — good evidence the A/B is faithful.

**to verify**

1 - The new head correction is gentle — it moves head_pitch only 21.8° → 19.1°, versus the old term's violent → 4.0°. That's the point (no overshoot), but the scalars can't tell us whether 19.1° actually looks right against the ground truth. The cam10_2 overlay you flagged was rendered under the old joint-anchored term, so the fix should visibly help — but I'd suggest regenerating the tier_comparison overlays to confirm the head sits correctly now. 
2 - W3 (profile 0.3) is a small net negative here — W2_vertex alone (22.87, 69.8 px) beats W2v_W3 (23.55, 71.9 px). The current shipped default is W2v_W3. This lines up with the earlier LOVO finding that blanket profile down-weighting is too blunt. I did not change W3 — that's your call: relax the profile weight back toward 1.0, or do the targeted per-view rejection (cam06_4/cam02_5) we discussed.

---

## The case for targeted per-view reprojection

the key finding for the cam10_2 question, straight from the cached LOVO diagnostic:

view	angle	in-sample px	held-out px	outlier?
cam06_4	profile	324	329	YES
cam02_5	profile	158	173	YES
cam04_4	frontal	96	101	no
cam10_2	frontal	75	87	no
cam02_4	profile	51	53	no

Two things jump out. First, cam10_2 is not an outlier — its reprojection is middling (75px) and it's a frontal view, so its head "tilt" isn't the fit being dragged around by cam10_2. Second, the two genuine outliers (cam06_4 at a broken 324px, cam02_5 at 158px) are both profiles — but so is the perfectly good cam02_4 (51px), which blanket-W3 needlessly suppresses too. That's exactly the case for targeted rejection.

### However! The reality doesn't match

The middling-frontal view of cam10_2 has been leading me astray with the head position...

cam10_2 is not an outlier. The cached LOVO diagnostic has it at 75px in-sample, a middling frontal view — so its head isn't being dragged around by a bad camera in the fit. And the body-intrinsic head pitch is +19–21° bowed forward in 3D (a view-independent measurement in the body's own frame), i.e. the opposite of "tilted back." So the tilt you see is specific to how the model projects into that elevated front camera — a small error in cam10_2's recovered elevation rotates the whole projection, and the ear (high on the skull, far from the pelvis pivot) shows it most. That's a per-view camera effect, not a body-pose defect.

The proof: no view-weighting config improves cam10_2 — they all make it worse (74 → 81px). Weighting changes the fitted body, but cam10_2's own recovered camera is fixed, so it can't fix that overlay. The lever for cam10_2 specifically is its Tier-2 PnP camera (or the head term, already fixed) — not weighting.

**targeted rejection results**

config	PA-MPJPE	medReproj	cam10_2	cam02_4 (good profile)
W2_vertex (no weighting)	22.87	69.8	74.0	67.1
W2v_W3 (blanket profile ×0.3, shipped)	23.55	71.9	77.0	59.6
W2v_tgtDown (cam06_4+cam02_5 ×0.3)	24.00	73.4	81.2	54.2
W2v_tgtDrop (cam06_4+cam02_5 dropped)	24.78	76.4	81.1	53.6

Suppressing profiles makes PA-MPJPE monotonically worse, and targeted rejection is worse than the blanket W3 it was meant to replace. Why: those two "outlier" profiles are load-bearing for depth — the frontal/three-quarter views barely constrain front-back position, so the profiles' huge reprojection is profile-hardness + them fighting the frontals on depth, not garbage. Remove them and the fit drifts in depth. Worse still, dropping cam02_5/cam06_4 while keeping cam02_4 at full weight unbalances the sagittal constraint (cam02_4 & cam02_5 are the same azimuth at two heights), which is why targeted beats out worse than symmetric blanket suppression. 

the lesson:
Profiles have to be weighted as a balanced set — you can't surgically thin them.

W2_vertex is the way to go for now, and i've set this in the Phase5Config as a default.

The last run to re-visualise these results...

added --dump-config/--dump-dir to the A/B harness. It reuses the fast cached refit and the already-solved cameras (config-independent), so no full pipeline re-run — the whole thing took a few seconds on GPU:

   ```
   python -m scantosmpl.evaluation.ab_refit \
    --dump-config W2_vertex --dump-dir output/debug/refinement_W2_vertex

   python -m scantosmpl.evaluation.visualise \
    --refinement-dir output/debug/refinement_W2_vertex \
    --views cam10_2.JPG,cam01_2.JPG \
    --output-dir output/debug/tier_comparison_W2_vertex

   ```
    Any config label from the A/B table (baseline, W2v_W3, W2v_tgtDrop, …) works with --dump-config.

comparing them to the shipped W2v_W3: for cam10_2 the two Tier-2 overlays are visually identical — dropping profile weighting doesn't un-tilt the head. That's the picture matching the numbers (its own reprojection moves only 77→74px, head pitch stays ~+19° forward in 3D). It reconfirms cam10_2's backward-tilt is that elevated camera's projection, not a body defect no weighting can touch.

**Shipped: W2_vertex**

"W2_vertex" is two ingredients, and the pipeline already had the second one baked in:

No profile down-weight → view_angle_weights profile 0.3 → 1.0. ✅ You've done this.
Vertex head anchor (2D ears → ear-vertex midpoint, not biased joint 15). This is already the pipeline default: pipeline.py:150-157 constructs SMPLOptimiser without passing midpoint_to_smpl/vertex_midpoint_to_smpl, so the optimiser defaults apply — joint anchor OFF ({}), vertex anchor ON (optimiser.py:131). That's the "W2v" half, shipped since the earlier head-anchor fix.

view_name_weights staying empty matches the A/B's _NO_NAME. So the pipeline now runs exactly the W2_vertex configuration.

One caveat on the number, not the config: the 22.87mm came from the A/B harness, which fits off cached triangulated joints + PnP cameras (deterministic, for clean attribution). A full end-to-end run re-triangulates, re-solves PnP, and applies the pipeline's MAD outlier-view rejection — so the absolute PA-MPJPE on a fresh run may drift a little from 22.87. The config change is correct and complete; just re-run the full pipeline once and glance at metrics.json / the AC7 check to confirm the shipped number on live artifacts.

---

## PSD Implementation - Phase A

The brief: Stage A builds a scattered-data interpolator over pose-space - this is NOT a body model.

Prerequisites: Per-pose registered SMPL fit (all poses)

It's job:

   `pose_feature_x_i -> δ_i ∈ ℝ^(6890×3)`

Where gamma lives in each vertex's rest-pose frame. All movement/deformation linear-blend skinning (LBS) explains contributes zero.

The deliverable:

- build a corpus of (x_i, gamma_i) with Beta asserted constant:
  - per vertex RBF solve `w = (ΦᵀΦ)⁻¹Φᵀd`
  - evaluate with a held-out pose
  - bake into Maya deltas (displacement)
  
RBF is used because with N=7 poses, a neural-net has to validation split to spare and no exactness guarantee (but a likelihood of overfitting). RBG interpolates the training poses exactly and it's only hyperparameter (rho) is the falloff needed for an animator/blend-shape artist/character rigger.

Caveat: both prediction an target live on the SMPL manifold, without independant geometric evaluation. So the verification here is self-consistency, not attributable to the real ground truth.

Φ is not "a matrix of pose features." It's a matrix of similarities between pose features. Feature vectors go in; scalars come out.

D is (5, 20670), not 6890×3. Five rows — one per training pose. And yes, its entries are δ.

### The pipeline:

```
CAPTURE          5 poses of one subject, ~60 photos each
   │
   ▼
TIER 1→2         per pose → (β, θᵢ) + fitted mesh
   │
   ▼
β-LOCK REFIT     fit T-pose first, freeze its β, re-fit the other 4 with β
   │             non-trainable.  ← WHY: β drift would let the model learn
   │             "body build changes with pose", which is estimator noise.
   ▼
MESHROOM         photogrammetry → raw point cloud Cᵢ per pose
   │             ← THE CRITICAL PATH. Without off-manifold geometry there
   │               is no δ. You have this for T-pose only.
   ▼
TIER 3           fit SMPL+D to Cᵢ → per-vertex displacement Dᵢ
   │             TRAINING POSES ONLY. Never the held-out pose.
   ▼
RESIDUAL         δ_world = scan_vertex − M_base_vertex
   │             δ_local = R_v(θ)⁻¹ · δ_world        ← R5. Silent-failure zone.
   ▼
ENCODE           θᵢ → xᵢ    (R(θ)−I, flattened)
   │
   ▼
BUILD Φ          Φᵢⱼ = φ(‖xᵢ−xⱼ‖)                    → (5,5)
   │
   ▼
SOLVE            np.linalg.solve(Φ, D) → W           → (5,20670)
   │             This is "training". It is one line.
   ▼
PREDICT          new pose θ*:
   │               x*  = encode(θ*)
   │               δ*  = Σⱼ Wⱼ · φ(‖x* − xⱼ‖)        ← in LOCAL frame
   │               δ*_world = R_v(θ*) · δ*           ← rotate back OUT
   │               mesh = SMPL(β, θ*) + δ*_world
   ▼
EVAL             Chamfer(mesh, C_heldout)
   │             Tier 3 never ran on the held-out pose. No leak.
   ▼
EXPORT           Maya blend shapes: neutral mesh + per-pose delta targets
```

3. Why each piece is shaped that way
Why local frame? In world coordinates, a forearm vertex swings through a huge arc when the elbow bends. That motion is 99% rigid articulation, which LBS already explains. Subtract the baseline and undo the rotation, and what's left is only the muscle bulge. That residual is small, smooth, and roughly the same at similar poses — so it interpolates. Raw world displacement would not.

Why does δ go back out through R_v at predict time? Because you learned it in rest orientation, and Maya needs it in the posed mesh's frame. The forward and inverse rotations bracket the whole model. Get one wrong and the other still runs.

Why RBF and not a net? Look at the shapes. Φ is 5×5. Your model has 5 degrees of freedom per output. A net would have thousands of parameters and 5 training examples, needing a validation split you don't have. RBF hits all 5 training poses exactly, closed form, one hyperparameter with physical meaning.

Why does the encoding dimension matter if it doesn't change Φ's size? It enters only through ‖xᵢ − xⱼ‖. In high dimensions with few points, all pairwise distances converge to roughly the same value — so every φ entry becomes similar, Φ approaches all-ones, becomes near-singular, and the interpolant degenerates toward a constant. Keeping d low keeps the distances discriminative.

4. What the outstanding questions are actually for
Q-A: the test fixture
The problem it solves: if R_v(θ)⁻¹ is wrong — transposed, wrong joint, wrong composition order — δ still comes out as 6890 plausible-looking small vectors. The RBF still fits. The export still loads in Maya. It just silently encodes the wrong deformation. Nothing downstream will ever tell you.

answer WIP:

   potentially the middle of the arm (look it up as instructed)
   with rotation matrices, there is an angle where the Rt is the same (orthogonal, perpendicular, pi/2, pi?)

anser B WIP:

   tune alpha on the training set or fix a-priori as the mean nearest-neighbour distance between training poses.

   to avoid LOVO as extrapolation, use one of the standard-reference poses for LOVO. 
   keeping an extreme or unusual pose in the training set?

So you need a case where you know the right answer before running the code.

The δ=0 test can't do it (R⁻¹·0 = 0 for any R).
Constructing the input with your own R_v can't do it (proves R·R⁻¹=I).
So R_v must be hand-written, which needs a vertex where it's a single bone's rotation with no blending → look at lbs_weights.max(axis=1).
And your test angle must not be one where R = Rᵀ, or a transpose bug passes.
The two sub-questions: which vertices, and which angle.

Q-B: σ, and the leakage you just walked into
You said tune σ on LOPO. But if σ is chosen by minimising held-out error, and then you report that held-out error as your result, the number is optimistically biased — you fitted the test set with one parameter.

At N=5 you can't afford a third split. So pick one:

Fix σ a priori by rule (mean nearest-neighbour distance between training poses), report LOPO honestly. Recommended.
Tune σ on LOPO and report it explicitly as a tuned number, not a clean generalisation estimate.
Also unresolved from earlier: at N=5, some poses sit on the edge of the pose cloud. Holding one of those out tests extrapolation, which PSD isn't for. What do you do — and can you even tell which poses are interior with 5 points?

5. What I still need from you
The fixture (Q-A): which vertices, which angle, what the selectable inputs are.
Your milestone sequence, re-ordered with the four gaps filled: β-lock refit, Meshroom (long-running, blocks nothing), the residual step, and the fixture that must precede it.
Take the fixture first. It's the smallest one and it unblocks everything downstream.

 whu C_heldout is not run through tier 3? If the held-out pose is not fitted to the ground truth, how do we know if the PSD gamma improved the chamfer during evaluation?
Two separate things were bundled in what I said. Let me sharpen it.

Why it can't be the target: Tier 3 is an optimiser fitted to C_7. Its output is nearly as close to C_7 as the machinery allows — by construction. If you compared PSD's prediction to that, you'd be comparing a prediction against an oracle that already saw the answer. And if that Tier-3 output ever fed training, C_7 is in the loop and the split is dead.

Your question — how do we know δ helped? Not by comparing to Tier 3. By comparing to a baseline at the same pose. Both built from θ_7 alone:


baseline :  SMPL(β, θ₇)                → Chamfer to C₇   = E_base
PSD      :  SMPL(β, θ₇) + δ_pred       → Chamfer to C₇   = E_psd
E_base − E_psd is the headline number. That's the whole claim: given only the pose, does the learned corrective get you closer to the real body than generic SMPL does? Neither mesh touched C_7.

And Tier 3 on pose 7 is still worth running — as a ceiling, not a target. Fit it to C_7, get E_oracle. That tells you the best achievable on this data, so you can report:


fraction of achievable gain captured  =  (E_base − E_psd) / (E_base − E_oracle)
Which is far more honest than a raw millimetre number, because it separates "PSD is weak" from "the point cloud/topology can't do better anyway."

So: run Tier 3 on the held-out pose, keep its output strictly out of training, and use it as the ceiling. That's the spec's "three-way comparison" done properly. I was too absolute earlier — you don't need it for the metric, but it makes the result interpretable.

Now: the fixture. Which vertices, which angle.

Answer A:

   ```scripts/print_lbs_max_weights.py```

   Top 10 vertices by max weight:
  vertex    weight  joint#  joint_name
    2725    1.0000      20  left_wrist
    2722    1.0000      20  left_wrist
    6194    1.0000      21  right_wrist
    6183    1.0000      21  right_wrist
    2720    1.0000      20  left_wrist
    2727    0.9999      20  left_wrist
    2721    0.9999      20  left_wrist
    6204    0.9999      21  right_wrist
    2723    0.9999      20  left_wrist
    6184    0.9999      21  right_wrist

   These are all left/right wrist vertices - hand vertices near the wrist are almost entirely rigid to the wrist joint.

   The test fixture's rotation angle could be either theta=0 or theta=pi

   a half-turn rotation turns every axis around. off diagonal elements (sin(x) =0) vanish making the matrix symmetric:

   ```
   R(pi) = ((-1,0), (0,-1)) = R(pi)^T
   ```
   in 3d space it also yields a symmetric rotation matrix.


For a test fixture we're not worried about anatomy:

   vertex 2725
   joint left wrist

   use 0 for R=I catching gross errors
   use Pi/2 about a single axis as the canonical rotation choice.
   this captures where R != R^t i.e the transpose should not produce identical output. Guards agains the issue where code that mistakenly uses R^t quietly produces identical output for the wrong input.
   for pi/2, the difference is large and obvious.

   displacement constant (multiplicative) -> use distinct unrelated components e.g (3,-7,11) = (Ux, Uy, Uz). A few mm should suffice so we can assert scale i.e not meters.

   axis: z
   use a coordinate axis so R stays hand-writable. In SMPL, set `theta[wrist] (0,0,pi/2)


Answer B - milestone sequence (refine!):

   Tier 3 refinement - package - ICP alignment, Kaolin chamfer, normal + Laplacian losses and body-part weighting -> capture per-vertex displacements (D).

   **background task**: meshroom -> raw point clouds Ci for all poses.

   capture all pose photography ->
   tier1>2 (all poses) ->
   Beta lock and refit all poses from locked t-pose base ->
   from the fitted base poses, get per-vertex residuals: gamma_world = scan - m_base. Transform to local pose-space. ->
   encode a vertex to each pose space ->
   hold out pose (LOPO) non-extreme e.g A-pose ->
   build phi matrix (5,5) on training poses ->
   np.linalg.solve(phi, Displacements from tier 3) = W (bump height for kernel function) ->
   predict on held-out pose: encode the new pose; solve local frame gamma; transpose to work frame; apply gamma_world to SMPL mesh ->
   evaluate mesh to the heldout scan with Chamfer distance ->
   optionally evaluate against the oracle held out tier 3 mesh to get percentage improvement ->
   export maya blend shapes

questions:

help me figure out what injected displacement is appropriate to use, could any small value suffice? 
same for which axis? 

should we start with alpha fixed a-priori, then try tuning it on ~5 rounds of validation?

held out-pose, this should be one that is not at the extremes - a standardised pose shape e.g. A-pose.

---

## Resolutions (continued from the questions above)

### The fixture knobs — what each one is *for*

`(vertex_index, joint_index, rotation_axis, rotation_angle, displacement u)`

| Knob | Value | What it catches / why |
|---|---|---|
| `vertex_index` | 2725 (or any 1.0-weight vertex) | Needs `lbs_weights[v].max() ≈ 1.0` so `R_v` is **one bone's rotation, no blending** — which is what makes it hand-writable. |
| `joint_index` | 20 (left_wrist) | Must be the joint that vertex is 100% weighted to. **Gotcha:** `R_v` is the joint's *global* rotation = composition of all ancestors. So pose ONLY this joint, leave every ancestor at rest → global == local → still hand-writable. |
| `rotation_axis` | z (sweep x, y, z) | Coordinate axis keeps `R` trivially hand-writable. Sweeping all three catches **axis-indexing errors**. |
| `rotation_angle` | π/2 — **never 0 or π** | At 0 and π, `R = Rᵀ`, so a transposed-matrix bug produces *identical output* and the test passes. π/2 makes `R ≠ Rᵀ` with an obvious difference. (0 is still worth keeping as a **second** case: `R=I` catches gross errors, just not transposes.) |
| `u` | `(3, −7, 11)` mm | Must **break symmetries**. `(1,1,1)` hides axis swaps. `u` parallel to the rotation axis is useless (`R·u = u`). Distinct magnitudes + mixed signs + mm scale (doubles as a m-vs-mm units check). |

(need to confirm understanding of displacement (u) values)

Worked, for 90° about z: (need to confirm understanding here)

```
R = [ 0 −1  0 ]     u    = ( 1,  2,  3)
    [ 1  0  0 ]     R·u  = (−2,  1,  3)   ← correct
    [ 0  0  1 ]     Rᵀ·u = ( 2, −1,  3)   ← the bug. Two signs flip. Unmissable.
```

### Pose encoding, concretely (see chat session and clarify with your understanding)

`θ` = 69 numbers = 23 joints × 3 (axis-angle: direction = axis, length = angle in radians).

Per joint, three steps:

```
1. axis-angle (3,)  ──Rodrigues──►  rotation matrix (3,3)
2. subtract identity:  R − I
3. flatten to (9,)
```

Concatenate 23 joints → **x is (207,)**. One vector per pose.

At rest, `θ_j = 0` → `R = I` → `R − I = 0`, so the rest pose encodes to **all zeros**. Good coordinate origin.

Bend one joint 90° about z:
```
R − I = [ 0 −1  0 ]   [ 1 0 0 ]     [ −1 −1  0 ]
        [ 1  0  0 ] − [ 0 1 0 ]  =  [  1 −1  0 ]  → (−1,−1,0, 1,−1,0, 0,0,0)
        [ 0  0  1 ]   [ 0 0 1 ]     [  0  0  0 ]
```

Code: `scipy.spatial.transform.Rotation.from_rotvec(theta.reshape(23,3)).as_matrix()`, subtract `np.eye(3)`, reshape. Or `batch_rodrigues` in `external/smplx/smplx/lbs.py` — what SMPL uses for its own pose blendshapes.

### Regioned pose encoding — why NOT all 207 dims

Two independent reasons:

1. **Statistical.** 5 basis functions spanning 207-D. Every query pose is far from all 5 training poses, so every `φ` value is small and roughly equal → the prediction collapses to a fixed weighted average, barely responsive to *which* pose was asked for. The slider does nothing.
2. **Physical (stronger).** The encoding defines what "nearby pose" *means*. In 207-D that's "similar in all 23 joints at once." But a forearm vertex's soft-tissue bulge does not depend on the ankle angle — so including the ankle injects noise into the single number the interpolant depends on.

**Resolution — per-region pose spaces** (paper §4.1, spec §4.3 "pose space dimension can vary per vertex"):

```
arm vertices    ← shoulder + elbow     (18-D)
leg vertices    ← hip + knee           (18-D)
torso vertices  ← spine joints         (27-D)
```

Each region gets its **own** Φ (still 5×5) from **its own** distances. Whole-body rig, every sub-problem low-dimensional. Cost: a handful of 5×5 solves + a region→joints map. Start with one region to prove the machinery, then fan out — same code, different joint subset.

**Implementation note — the (207,) vector is never actually built.** It's just the concatenation of 23 per-joint (9,) blocks, so have the encoder return `(23, 9)` and slice:

```python
blocks = encode(theta)                            # (23, 9)
x_arm  = blocks[[SHOULDER, ELBOW]].reshape(-1)    # (18,)
x_leg  = blocks[[HIP, KNEE]].reshape(-1)          # (18,)
```

The flattened 207-D form is the all-joints special case, not a separate object.

### Hold-out selection

- ~~Rank by distance to the 207-D centroid.~~ **Superseded by regioned encoding: interiority is per-region.** A pose can be nicely interior for the arm region and an extreme outlier for the leg region; one global distance averages those together and hides it. Compute centroid distances **per region** and pick the pose that is acceptably interior across the regions you're actually claiming a result for.
- **You still only get ONE hold-out pose.** You cannot hold out different poses per region: you evaluate one assembled mesh against one point cloud, so if the leg model trained on `C_heldout` while the arm model held it out, the whole-mesh chamfer is leaked for the leg. One pose, held out globally, applied to every region.
- **Check the spread.** If the five centroid-distances (within a region) are all within a few % of each other, the ranking is arbitrary and the choice is a coin flip dressed as a criterion. Report the numbers. Expect this to happen — say so if it does.
- HS / BTW are envelope-defining → keep in **training**. T-pose is the reference → always train. Leaves A-pose or shu.
- **Hard limit to state in the report:** with N=5 and encoding dim > 4, *every* pose is on the convex hull. Five points span at most a 4-simplex. True interpolation testing needs N > d+1 and is **not achievable at N=5** — the spec's §7 split policy assumes otherwise. Nearest-centroid only *minimises* how far you extrapolate.

### σ — three options, increasing rigour

1. **Fixed a priori** = mean nearest-neighbour distance between training encodings. Report LOPO clean. **Start here.**
   → **σ is per-region.** Mean NN distance in the 18-D arm space is a different number from the 27-D torso space. One global σ would be too wide for one and too narrow for the other. Same rule, applied per region.
2. **Tuned, labelled as tuned** — fine if the report says "σ selected on this metric" and doesn't call it generalisation.
3. **Nested CV** — outer holds out pose i, inner tunes σ by LOPO on the remaining 4. 5×4 = 20 solves of a 4×4. Milliseconds. Free at this scale.

Regardless: an error estimate from 5 samples is noisy. Report spread, not just mean.

### δ is additive; R_v is a change of basis — no conflict

```
vertex_posed = base_vertex + δ_world     ← ADDITION applies the correction
δ_world      = R_v · δ_local             ← ROTATION changes which frame it's expressed in
```

"Step 3cm forward" is added to your position; the `(x,y,z)` representing "forward" depends which way you face. Turning changes components, not length. So Tier 3's world-frame `D` → target via `δ_local = R_v⁻¹ · D`.

### Chamfer metric definition → REVIEW.md § 7.M1–7.M6

SMPL edge length ~1cm. A cloud point mid-triangle is far from all three *vertices* even when the surface passes exactly through it (worst case ≈ `L/√3` ≈ 5.8mm at L=10mm). So **vertex-based cloud→mesh distance has a floor set by tessellation, not fit quality** — over half the 8mm budget.

- Cloud→mesh **must** be point-to-**surface** (point-to-triangle). Requirement on the quantity, not the library.
- Mesh→cloud may be vertex-to-point (dense clouds → effectively floor-free).
- Report **both directions separately**; single-direction chamfer is gameable by a mesh collapsing into the dense region.
- Loss ≠ metric is explicitly allowed.
- **Measure your own tessellation floor**: sample points on the SMPL surface, distance to nearest vertex, mean + max.
- 8mm is a composite: fit error + Meshroom noise (±1–3mm) + tessellation + clothing/hair. Decompose before re-tuning weights.

### Tier 3 boundary requirements → REVIEW.md § 7.B1–7.B8

Eight contract requirements on Phase 7 so its output is usable as PSD ground truth. The two that bite hardest:

- **7.B1 β-lock** — and it **conflicts with AC 7.4** ("β refinement improves body proportions"), which requires Tier 3 to *optimise* β. Resolution recorded in REVIEW.md: refine β **once** on the reference pose, then **freeze** for every other pose; both modes supported, manifest records which.
- **7.B3 `D` frame convention** — pre-LBS (rest) vs post-LBS (posed) must be documented and asserted. It's a one-line decision in Tier 3 that silently determines whether PSD applies `R_v⁻¹` or nothing, with no visible symptom if wrong.

### Kaolin / torch version — defer the decision

Kaolin's advantage is a *differentiable GPU* point-to-mesh loss. The **metric** doesn't need differentiability (Open3D `RaycastingScene.compute_distance` or `trimesh.proximity.closest_point`); the **loss** can start as `torch.cdist` vertex-to-point. So build Tier 3 without Kaolin, get a number, and only then decide if it earns a torch pin. If it does: prefer **downgrading the single env** over a split venv — the tiers share `scantosmpl` package code, and two envs means installing your own package twice forever.

---

## Outstanding tasks before implementation

| # | Task | Why it's before code | Done looks like |
|---|---|---|---|
| 1 | **Confirm δ ≡ 0 empirically.** Load Tier 2 params → `SMPL(β,θ)` → align to the saved `.obj` (remove scale+translation) → report max & mean per-vertex deviation in mm. | Closes the spec defect with a number instead of an inference. Replaces §4.2's "geometrically thin". | A number in notes.md, and a spec edit to §4.2/R3/AC2/AC3. |
| 2 | **Build the R5 fixture.** Both cases: `θ=0` (`R=I`, gross errors) and `θ=π/2` about z (transposes). Knobs parameterised per the table above. | The residual step fails *silently*. No real δ has a known correct value — only a constructed one does. Must precede the residual code it validates. | A test that fails when you deliberately transpose `R_v`. |
| 3 | **Measure the tessellation floor** on your SMPL mesh. | Makes every chamfer number interpretable; tells you how much of the 8mm budget is unavailable. | mean + max mm, recorded. |
| 4 | **Decide the region map** — which joints drive which vertices. | Determines encoding dim, hence whether interpolation is meaningful at N=5. Needs looking at what actually varies across T/A/shu/hs/btw. | A `region → joint indices` dict + the reasoning. |
| 5 | **Compute the 5 pose encodings**, centroid distances, pick the hold-out, **report the spread**. | Turns hold-out choice from intuition into a measurement; the spread tells you whether the choice is meaningful at all. | 5 distances, chosen pose, spread stated. |
| 6 | **Fix the σ policy** (start: a priori = mean NN distance). | Prevents the leakage of tuning-then-reporting on the same fold. | Written down before the first eval run. |
| 7 | **Settle `D`'s frame convention with Tier 3** (7.B3) before Tier 3 is built. | Cheap now, expensive after hours of fitting produce data in the wrong frame. | Documented in the Tier 3 spec + asserted in the artefact. |
| 8 | **Re-sequence the milestones** with Tier 3 broken into its real components, Meshroom as a parallel background job, and the fixture slotted before the residual step. | The current sequence hides an unbuilt package in one arrow and serialises a long job that blocks nothing. | Updated sequence in notes.md. |

**Suggested order:** 1 → 2 → 3 (all independent, all small, all produce numbers) → 4 → 5 → 6 → 7 → 8. Kick Meshroom off in the background before any of them.


---



## Glossary

Term	Shape	What it is
β	(10,)	Shape. Who the person is. Bone lengths, build. Constant across all poses. Locked.
θ	(69,)	Pose. Joint angles, axis-angle. Varies per pose.
LBS	—	Linear Blend Skinning. The rigid part of posing: each vertex is moved by a weighted blend of nearby bone transforms.
lbs_weights	(6890, 24)	How much each vertex is influenced by each joint. Rows sum to 1.
R_v(θ)	(3,3) per vertex	The rotation LBS applied to vertex v at pose θ — the blend of nearby bone rotations.
M_base	(6890, 3)	SMPL(β, θ). What LBS + SMPL's own pose blendshapes predict. The baseline.
δ_local	(6890, 3)	The learning target. What the real surface does that M_base failed to predict, expressed in each vertex's rest orientation.
x	(d,)	Pose encoding. One vector per pose. R(θ)−I flattened. The interpolation coordinate.
φ(r)	scalar	The bump. exp(−r²/2σ²). A falloff curve, not a distribution.
σ	scalar	Bump width. The only hyperparameter. The animator's falloff knob.
Φ	(5, 5)	Φᵢⱼ = φ(‖xᵢ−xⱼ‖). Pose-to-pose similarity.
W	(5, 20670)	The model. Bump heights.
N = 5	—	Training poses. Sets Φ's size. Sets the model's total capacity.
d	—	Encoding dimension. Affects distances only. Never affects Φ's size.

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
