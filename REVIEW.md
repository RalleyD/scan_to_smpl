# REVIEW.md — ScanToSMPL Phase-Gated Implementation Plan

## Overview

Phased implementation plan with explicit gating criteria. Reflects the three-tier
calibration-free architecture using CameraHMR (Tier 1), PnP self-calibration (Tier 2),
and Kaolin-based chamfer fitting (Tier 3).

**Gate Rule**: ALL acceptance criteria must pass before proceeding. If a gate fails,
diagnose using the troubleshooting notes before revisiting architecture.

**Tier Checkpoints**: Phases 3, 5, and 7 produce independently shippable outputs.
You can deliver Tier 1 while still building Tier 2+3.

---

## Phase 0: Scaffolding & SMPL Model Setup (1 day)

### Objective
Project structure, dependencies, SMPL forward pass verified on GPU.

### Deliverables
- [x] `pyproject.toml` with all dependencies (torch, smplx, kaolin, transformers, opencv, open3d, click)
- [x] Package structure per CLAUDE.md
- [x] `scantosmpl/smpl/model.py` — SMPL/SMPL-X wrapper with differentiable forward pass
- [x] `scantosmpl/config.py` — Dataclass config
- [x] `scantosmpl/types.py` — ViewType, FittingResult, CameraParams
- [x] `tests/test_smpl_model.py`
- [x] `data/body_models/README.md` — download instructions for SMPL .pkl files
- [x] Verify Kaolin installs cleanly: `from kaolin.metrics.pointcloud import chamfer_distance`

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 0.1 | `pip install -e .` succeeds (Python 3.10+, fresh conda env) | CI or manual |
| 0.2 | SMPL forward: 6890 verts, 13776 faces, 24 joints | Assert shapes |
| 0.3 | Differentiable: `loss.backward()` flows grads through β and θ | `.grad is not None` |
| 0.4 | Forward < 50ms on GPU | Timing |
| 0.5 | Kaolin chamfer_distance works on two random point clouds | Smoke test |
| 0.6 | Joint regressor: shoulder width 35-45cm at neutral pose | Measurement |

---

## Phase 1: Image Loading, Detection & View Classification (2 days)

### Objective
Load images with EXIF normalisation, extract intrinsics, run RT-DETR + ViTPose++,
classify views as full-body/partial/unusable.

### Deliverables
- [x] `scantosmpl/detection/image_loader.py` — EXIF transpose + intrinsics extraction (FocalPlaneXResolution → K)
- [x] `scantosmpl/detection/person_detector.py` — RT-DETR via HuggingFace
- [x] `scantosmpl/detection/keypoint_detector.py` — ViTPose++-Base via HuggingFace
- [x] `scantosmpl/detection/view_classifier.py` — classify by joint visibility ratio
- [x] `tests/test_detection.py`
- [x] Viz: overlay keypoints on images (via DetectionPipeline debug output)

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 1.1 | EXIF transpose handles all 8 EXIF orientations | Synthetic fixture images |
| 1.2 | EXIF intrinsics: reasonable focal lengths (500-5000px) from real scanner images | Test 10+ images |
| 1.3 | RT-DETR: exactly 1 person per scanner image | 10+ images |
| 1.4 | ViTPose++: ≥15/17 COCO keypoints on full-body views (conf > 0.5) | Aggregate stats |
| 1.5 | View classifier: ≥95% correct on manually labelled test set | Compare labels |
| 1.6 | ~20/60 views classified as full-body | Matches scanner geometry |
| 1.7 | Pipeline: 60 images in < 15 seconds on GPU | Wall-clock |

### Notes
- ViTPose++-Base (100M params) is the sweet spot: ~200MB, 4GB VRAM, good accuracy.
- RT-DETR and ViTPose++ are both native HuggingFace `transformers` — no mmcv/detectron2.
- If speed is critical, RTMPose (`rtmlib`, ONNX) runs at 430+ FPS but requires separate install.

---

## Phase 2: Per-View HMR — CameraHMR Integration (2.5 days)

### Objective
Run CameraHMR on each full-body image. Get per-view SMPL params, FoV estimates,
and 138 dense surface keypoints. This is the most complex integration phase.

### Deliverables
- [ ] `scantosmpl/hmr/camera_hmr.py` — CameraHMR inference wrapper (batch support)
- [ ] `scantosmpl/hmr/prompt_hmr.py` — PromptHMR fallback wrapper
- [ ] `scantosmpl/hmr/orientation.py` — Per-view global orientation + reference pose alignment
- [ ] `tests/test_hmr.py`
- [ ] Viz: overlay CameraHMR mesh on source image per view
- [ ] Download script for CameraHMR weights

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 2.1 | CameraHMR: valid SMPL output (β: 10D, θ: 72D) per full-body view | Shape checks |
| 2.2 | CameraHMR: FoV estimate within 10° of EXIF-derived FoV (where EXIF available) | Cross-check |
| 2.3 | 138 dense surface keypoints detected per view with confidence scores | Shape: (138, 3) per view |
| 2.4 | Per-view meshes visually align with source images | Inspect 5+ views |
| 2.5 | Shape β variance across views: std < 1.0 per component | Stats |
| 2.6 | Body pose θ variance (excl. global orient): low for A/T-pose | Stats |
| 2.7 | Inference: 20 images in < 60 seconds on 8GB+ GPU | Timing |

### Risks & Mitigations
- **CameraHMR registration blocked**: Use PromptHMR (Google Drive, no registration). Lose FoV + dense keypoints but still get SMPL params. Add separate ViTPose keypoints for PnP.
- **CameraHMR needs PyTorch 2.0 + CUDA 11.8**: Pin in conda env. Your RTX 3080Ti supports this.
- **Scanner lighting causes poor HMR**: Unlikely in controlled scanner. Test early with real images.

---

## Phase 3: Multi-View Consensus — Tier 1 Complete (2 days)

### Objective
Fuse per-view SMPL estimates into consensus. This completes Tier 1: a usable SMPL
fit with ZERO calibration.

### Deliverables
- [ ] `scantosmpl/hmr/consensus.py` — β median, θ SO(3) Fréchet mean, FoV median
- [ ] `scantosmpl/utils/geometry.py` — SO(3) operations, rotation conversions, weighted stats
- [ ] `scantosmpl/smpl/joint_map.py` — COCO↔SMPL mapping + CameraHMR dense mapping
- [ ] `tests/test_consensus.py`
- [ ] Viz: consensus mesh from multiple viewpoints

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 3.1 | Consensus β: anatomically plausible (height 1.5-2.0m, no extreme proportions) | Visual + measure |
| 3.2 | Consensus θ: matches reference pose (arms/legs in correct A-pose/T-pose position) | Visual |
| 3.3 | Consensus improves over single-best view: lower variance, more stable | A/B compare |
| 3.4 | SO(3) averaging: valid rotation matrices (det=1, orthogonal) | Matrix checks |
| 3.5 | FoV consensus: single value, within 5° of any individual CameraHMR estimate | Stats |
| 3.6 | **Tier 1 cross-view consistency: estimated PA-MPJPE < 50mm** | Use per-view predictions as proxy ground truth |

### ⚠️ TIER 1 GATE — First Shippable Output

This produces a usable SMPL fit from images alone. If it fails:
- PA-MPJPE 50-70mm: CameraHMR may struggle with scanner images → try PromptHMR
- PA-MPJPE > 70mm: Image quality issue → check EXIF orientation, lighting, person detection

---

## Phase 4: PnP Self-Calibration (1.5 days)

### Objective
Recover camera [R|t] per view using the SMPL mesh as calibration target. CameraHMR's
138 dense surface keypoints make PnP dramatically more robust than sparse joint-based PnP.

### Deliverables
- [ ] `scantosmpl/calibration/intrinsics.py` — K from CameraHMR FoV (primary) + EXIF (fallback)
- [ ] `scantosmpl/calibration/pnp_solver.py` — PnP with 138 dense correspondences
- [ ] `tests/test_pnp.py`
- [ ] Viz: recovered camera positions in 3D (should roughly match scanner layout)

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 4.1 | PnP succeeds on ≥90% of full-body views (≥18/20) with 138 dense correspondences | Count |
| 4.2 | PnP succeeds on ≥50% of partial views using available keypoints | Count |
| 4.3 | Recovered cameras form plausible geometry (roughly circular scanner layout) | 3D viz |
| 4.4 | Reprojection error < 15px per view (using Tier 1 SMPL + recovered [R|t]) | Measure |
| 4.5 | Robust to ±10% focal length perturbation | Ablation |
| 4.6 | 138 dense keypoints PnP outperforms 12 sparse joints PnP | A/B compare |

### Key Insight
138 correspondences vs 12 is transformative for PnP robustness. With 12 joints, a single
misdetected keypoint can flip the solution. With 138, RANSAC has massive redundancy.
This is the primary reason CameraHMR was chosen over PromptHMR as the default.

---

## Phase 5: Triangulation + SMPL Refinement — Tier 2 Complete (3 days)

### Objective
Triangulate 3D keypoints across views, then optimise SMPL to match triangulated joints
+ reprojection loss across ALL 60 views (including partial views).

### Deliverables
- [ ] `scantosmpl/triangulation/dlt.py` — Direct Linear Transform
- [ ] `scantosmpl/triangulation/ransac.py` — RANSAC outlier rejection
- [ ] `scantosmpl/triangulation/weighted.py` — Confidence-weighted triangulation
- [ ] `scantosmpl/smpl/losses.py` — Joint, pose prior, shape reg, reprojection losses
- [ ] `scantosmpl/smpl/prior.py` — GMM prior (VPoser optional)
- [ ] `scantosmpl/fitting/coarse.py` — Staged optimisation: orient → pose → shape
- [ ] `scantosmpl/fitting/reprojection.py` — All-view reprojection refinement
- [ ] `scantosmpl/fitting/scheduler.py` — Loss weight annealing
- [ ] `tests/test_triangulation.py`, `tests/test_coarse_fitting.py`

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 5.1 | Triangulated joints: plausible skeleton (limb lengths, symmetry) | Visual |
| 5.2 | Triangulation reprojection error < 10px across full-body views | Measure |
| 5.3 | **SMPL MPJPE < 25mm on triangulated keypoints** | ||J_smpl - J_target|| |
| 5.4 | Reprojection loss uses ALL 60 views; partial views contribute visible joints only | Verify gradients |
| 5.5 | Tier 2 improves over Tier 1 by ≥30% in cross-view consistency | Compare |
| 5.6 | Optimisation converges in < 60 seconds on GPU | Timing |
| 5.7 | No self-intersections added (vertex count < 5) | Check |
| 5.8 | Staged optimisation (orient→pose→shape) outperforms single-stage | A/B compare |

### ⚠️ TIER 2 GATE — Accuracy Threshold

| MPJPE | Action |
|-------|--------|
| < 25mm | **PASS** — proceed to Tier 3 |
| 25-40mm | PnP calibration may be off → try bundle adjustment → revisit Phase 4 |
| > 40mm | Tier 1 consensus too inaccurate → revisit Phase 3, try PromptHMR |

---

## Phase 6: Point Cloud Preprocessing & Alignment (2 days)

### Objective
Load, clean, align point cloud TO the SMPL mesh (not vice versa). SMPL from Tier 2
has correct metric scale and orientation — it's the alignment target.

### Deliverables
- [ ] `scantosmpl/pointcloud/io.py` — PLY/OBJ loading
- [ ] `scantosmpl/pointcloud/preprocess.py` — denoise, downsample, outlier removal (Open3D)
- [ ] `scantosmpl/pointcloud/align.py` — Rigid ICP to SMPL mesh (scale + R + t)
- [ ] `scantosmpl/pointcloud/segment.py` — Body-part segmentation (height slices + connectivity)
- [ ] `tests/test_pointcloud.py`

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 6.1 | PLY and OBJ load correctly | Round-trip test |
| 6.2 | Denoising: ≥80% outliers removed, body surface preserved | Visual before/after |
| 6.3 | ICP converges: point cloud aligns to SMPL mesh (visual check) | Overlay viz |
| 6.4 | Post-alignment chamfer < 30mm (before surface refinement) | Measure |
| 6.5 | Meshroom's arbitrary scale correctly resolved (aligned height matches SMPL) | Compare |
| 6.6 | Pipeline < 10 seconds on 500K point cloud | Timing |

---

## Phase 7: Surface Refinement — Tier 3 Complete (3 days)

### Objective
Refine SMPL β and θ using Kaolin chamfer distance with semantic body-part weighting.
Optionally compute SMPL+D per-vertex displacements.

### Deliverables
- [ ] `scantosmpl/fitting/surface.py` — Kaolin chamfer + semantic weighting
- [ ] SMPL+D support in `model.py` (per-vertex displacement field)
- [ ] Additional losses: normal consistency, Laplacian smoothing
- [ ] `tests/test_surface_fitting.py`
- [ ] Viz: SMPL mesh overlaid on point cloud, before/after refinement

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 7.1 | **Chamfer distance < 8mm** on clean scanner data — per the metric definition below, both directions reported separately | Measure |
| 7.2 | Surface refinement reduces chamfer by ≥40% vs Tier 2 alone | Compare |
| 7.3 | Semantic weighting improves torso alignment vs uniform weighting | A/B |
| 7.4 | β refinement improves body proportions (shoulder width, waist) vs Tier 2 | Measure |
| 7.5 | θ remains plausible; no self-intersections added | Visual + count |
| 7.6 | SMPL+D captures clothing detail within 5mm (if enabled) | Displacement stats |
| 7.7 | Optimisation < 60 seconds on GPU with 50K point cloud | Timing |

### 📏 Metric Definition — "Chamfer" (binding for 7.1, 7.2, and the Tier 3 gate)

"Chamfer distance" is ambiguous, and the ambiguity is worth several mm here — more than
half the 8mm budget. Fix it explicitly.

**SMPL's mean edge length is ~1cm.** A cloud point lying mid-triangle is far from all three
of that triangle's *vertices* even when the mesh surface passes exactly through it (worst case
≈ `L/√3` ≈ 5.8mm at `L`=10mm). So a vertex-based cloud→mesh distance has a **floor set by
tessellation, not by fit quality**. Measuring that way spends much of the 8mm budget before
the optimiser does anything.

| # | Requirement |
|---|-------------|
| 7.M1 | **Cloud → mesh direction MUST be point-to-*surface*** — distance to the nearest point on the nearest triangle, **not** to the nearest vertex. This is a requirement on the quantity, not on the library: any correct point-to-triangle implementation satisfies it. Record which was used in the artefact metadata. |
| 7.M2 | **Mesh → cloud direction** may use vertex-to-nearest-point — dense Meshroom clouds make this direction effectively floor-free. |
| 7.M3 | **Report both directions separately**, never only their mean or sum. A single fused number hides which side is failing, and one-directional chamfer is gameable by a mesh that collapses into the densest region of the cloud. |
| 7.M4 | **State the aggregation and units** in the artefact: mean / median / RMS, and mm. RMS and mean differ substantially in the presence of outliers, which point clouds always have. |
| 7.M5 | **Measure and report the tessellation floor** alongside the result: sample points uniformly on the SMPL surface, compute each one's distance to its nearest vertex, report mean and max. This is the irreducible offset for any vertex-based measurement and makes the headline number interpretable. |
| 7.M6 | The **loss** used inside the optimiser need not equal the **metric** used for reporting. Vertex-to-point is a cheap, well-behaved loss; if faceting or shrink-wrapping appears, switch the loss to point-to-mesh. The reported metric is bound by 7.M1–7.M4 regardless. |

> **Known confounds to state in the report, not silently absorb:** the 8mm figure is a composite
> of fit error + Meshroom reconstruction noise (~±1–3mm) + tessellation floor + clothing/hair
> present in the cloud but absent from SMPL. Only the first is what Phase 7 is optimising. If
> the gate fails, decompose before re-tuning loss weights.

### ⚠️ TIER 3 GATE — Surface Accuracy

| Chamfer | Action |
|---------|--------|
| < 8mm | **PASS** |
| 8-15mm | Check ICP alignment (Phase 6). Tune loss weights. |
| > 15mm | Tier 2 SMPL too far off → revisit Phase 5 |

### 🔗 Downstream Boundary Requirements — PSD / Tier 4

Tier 3's SMPL+D output is the **ground-truth source** for Pose Space Deformation
(`docs/psd_master_design_spec.md`). PSD is a strictly read-only consumer, so these are
contract requirements on Phase 7, not suggestions. Each one, if violated, silently
corrupts the PSD residual rather than failing loudly.

| # | Requirement | Why PSD needs it |
|---|-------------|------------------|
| 7.B1 | **`--lock-betas` flag: β non-trainable, supplied from the reference-pose fit.** Must be the mode used for any multi-pose corpus run. | PSD's target is `δ = scan − SMPL(β,θ)`. If β differs per pose, δ absorbs *shape* differences as if they were *pose* deformation, and the interpolant blends body builds. See PSD spec R2. **Conflicts with AC 7.4 — see note below.** |
| 7.B2 | **Persist `D` as a per-vertex (6890, 3) array per pose**, not just the final mesh. Mandatory, not "if enabled" (cf. 7.6). | `D` *is* the PSD learning signal. A baked mesh forces PSD to re-derive it and re-introduces frame ambiguity. |
| 7.B3 | **Document and assert the frame `D` is expressed in** — posed/world (added after LBS) vs rest (added before LBS). Record it in the artefact metadata as an explicit field. | Determines whether PSD applies `R_v(θ)⁻¹` or nothing at all. Getting this wrong produces plausible-looking but wrong δ with no visible symptom (PSD spec R5). |
| 7.B4 | **Topology invariance: 6890 verts / 13776 faces, original ordering, no remeshing or resampling.** Assert on write. | Blend-shape targets are index-aligned. Any reordering silently scrambles every target. |
| 7.B5 | **Keep global scale + translation separable from `D`.** Do not bake a similarity transform into the displacement field. | A global transform folded into `D` appears to PSD as a pose-correlated deformation — it is not, and it will dominate the residual. |
| 7.B6 | **Per-pose artefact layout + manifest** so a corpus builder can locate `(β, θ, D, quality)` by pose name. | `build_corpus()` reads these; it must assert shared β across samples. |
| 7.B7 | **Persist per-pose fit quality** (chamfer, pa_mpjpe, median_reproj_px) alongside `D`. | Feeds optional confidence-weighted RBF rows (PSD spec open-Q4). |
| 7.B8 | **Mark held-out poses as oracle-only in the manifest.** A pose fitted for evaluation must be flagged so it cannot enter PSD training. | Tier 3 on a held-out pose is a legitimate *ceiling* measurement, but if its output reaches training the point cloud is inside the loop and the split is void. |

> **⚠️ Conflict to resolve: AC 7.4 vs 7.B1.** Criterion 7.4 ("β refinement improves body
> proportions vs Tier 2") requires Tier 3 to *optimise* β. Requirement 7.B1 requires it to be
> *frozen*. Both are legitimate for different runs. Resolution: treat β-refinement as a
> **single-pose** capability (refine β once, on the reference pose, to get the best subject
> shape), then **freeze that β** for every subsequent pose in the corpus. Phase 7 should
> support both modes explicitly and the manifest should record which was used.

---

## Phase 8: End-to-End Pipeline & CLI (2 days)

### Deliverables
- [ ] `scantosmpl/fitting/pipeline.py` — orchestrator: detect → HMR → consensus → self-cal → triangulate → optimise → surface
- [ ] `scantosmpl/cli.py` — Click CLI with `fit-images`, `fit-pointcloud`, `fit-combined`
- [ ] `scantosmpl/evaluation/metrics.py` — MPJPE, PA-MPJPE, chamfer, reprojection error
- [ ] `scantosmpl/evaluation/visualise.py` — per-tier overlay + quality report
- [ ] `README.md` with installation + quick start
- [ ] `examples/` with sample scripts

### Acceptance Criteria

| # | Criterion | Verify |
|---|-----------|--------|
| 8.1 | `scantosmpl fit-images` runs end-to-end, zero calibration | CLI test |
| 8.2 | `scantosmpl fit-combined` achieves best accuracy across all modes | Compare |
| 8.3 | Output .npz: theta, beta, translation, scale, vertices, per-tier metrics | Inspect |
| 8.4 | Output .obj loadable in Blender/MeshLab | Manual check |
| 8.5 | Per-tier quality report: Tier 1 MPJPE, Tier 2 MPJPE, Tier 3 chamfer | Check output |
| 8.6 | Full pipeline (60 images + 500K PC) < 5 minutes on GPU | Timing |
| 8.7 | Graceful degradation: images-only mode works when no PC provided | Test |

---

## Phase 9: Packaging & Distribution (1 day)

- [ ] Finalised `pyproject.toml` with correct metadata
- [ ] LICENSE (MIT for code; note SMPL model files are non-commercial)
- [ ] GitHub Actions CI: lint (ruff) + type check (mypy) + test (pytest)
- [ ] Docker/devcontainer with pinned CUDA + PyTorch
- [ ] `pip install scantosmpl` from built wheel
- [ ] Worked example with sample data in `examples/`

---

## Summary

| Phase | Days | Cumulative | Tier | Key Component |
|-------|------|------------|------|---------------|
| 0: Scaffolding | 1 | 1 | — | SMPL + Kaolin |
| 1: Detection | 2 | 3 | — | ViTPose++ + RT-DETR |
| 2: Per-View HMR | 2.5 | 5.5 | Tier 1 | CameraHMR |
| **3: Consensus** ⚠️ | 2 | 7.5 | **Tier 1 ✓** | SO(3) fusion |
| 4: Self-Calibration | 1.5 | 9 | Tier 2 | PnP (138 dense kps) |
| **5: Triangulation** ⚠️ | 3 | 12 | **Tier 2 ✓** | DLT + SMPL optim |
| 6: Point Cloud | 2 | 14 | Tier 3 | ICP to SMPL |
| **7: Surface** ⚠️ | 3 | 17 | **Tier 3 ✓** | Kaolin chamfer |
| 8: Pipeline + CLI | 2 | 19 | — | End-to-end |
| 9: Packaging | 1 | 20 | — | pip + CI |

**Total: ~20 days** | ⚠️ = Hard accuracy gates

---

## Decision Log

| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| CameraHMR over HMR2.0 | Full perspective camera, FoV prediction, 138 dense keypoints. HMR2.0 unmaintained, detectron2 hell. | HMR2.0 (dead), TokenHMR (still needs detectron2) |
| PromptHMR as fallback | No registration required, Google Drive weights, SOTA accuracy (36.6mm). | SMPLer-X (8.2GB model, overkill) |
| Kaolin over PyTorch3D | pip-installable, PyTorch 2.1-2.8 support, Apache 2.0. PyTorch3D stuck at 2.4. | PyTorch3D (install hell), torch.cdist (slower) |
| 138 dense keypoints for PnP | Transformative for self-calibration robustness. 138 >> 12 sparse joints. | Sparse COCO joints only (fragile PnP) |
| CameraHMR FoV over EXIF | 5-7° error, trained on 500K Flickr images. EXIF may be missing/wrong. | EXIF only (unreliable for some cameras) |
| Align PC to SMPL, not reverse | SMPL has correct metric scale. PC from Meshroom has arbitrary scale. | Align SMPL to PC (inherits Meshroom issues) |
| HSfM as future investigation | Could replace entire pipeline but trades control. Evaluate after v1. | Adopt HSfM now (too early, less control) |

---

## How to Use This Document

1. **Before starting a phase**: Read acceptance criteria — they define "done".
2. **During implementation**: Track deliverable checkboxes.
3. **At phase gate**: Run ALL criteria. Document pass/fail.
4. **If a gate fails**: Do NOT proceed. Use troubleshooting notes. Update Decision Log.
5. **After passing**: Git tag `phase-N-passed`.
6. **Tier checkpoints** (3, 5, 7): Each produces an independently shippable result.
