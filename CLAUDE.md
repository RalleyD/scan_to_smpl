# CLAUDE.md — ScanToSMPL: Multi-View / Point Cloud → SMPL Registration

## Project Overview

**ScanToSMPL** registers SMPL/SMPL-X parametric body meshes to:
1. Multi-view images of a human subject in a reference pose (A-pose, T-pose, etc.)
2. Photogrammetry point clouds (PLY/OBJ) of a human subject

Outputs: optimised SMPL parameters (β, θ, translation, scale), registered mesh (.obj),
optional per-vertex displacements (SMPL+D), quality metrics per tier.

### Key Constraints
- Scanner captures ~60 images; only ~20 show full body. ~40 are partial views.
- **No camera extrinsics.** Only EXIF intrinsics (focal length, sensor info).
- Scanner rig geometry may not be available.
- Pre-existing Meshroom point clouds have arbitrary scale/orientation/coordinates.

---

## Why Existing Solutions Fail

| Tool | Problem |
|------|---------|
| **RVH Mesh Registration** | Needs clean scans + OpenPose 3D joints. Depends on legacy `psbody.mesh`. |
| **smplreg** | Chains PIFu + ProHMR — two fragile dependencies. |
| **SegFit** | Withdrawn from ICLR 2025. Unavailable. |
| **SMPLify-X / MultiviewSMPLifyX** | Requires pre-calibrated cameras in shared coordinate frame. |
| **Meshroom → OpenPose** | ±1600px coordinate errors. Fundamentally broken across tools. |
| **HMR2.0 / 4DHumans** | Effectively unmaintained (0 responses since mid-2023). Hard detectron2 dependency fails on PyTorch 2.0+. GPU memory leak (issue #14). |
| **MUC (AAAI 2025)** | Pinned to PyTorch 1.12 + CUDA 11.3 + deprecated mmcv-full. Fails on modern GPUs. |

---

## Architecture: Three-Tier Calibration-Free Pipeline

### Core Insight (from 2024-2025 research)

You don't need camera extrinsics for initial SMPL estimation. Modern per-image HMR
models (CameraHMR, PromptHMR) produce strong SMPL estimates from single images
with perspective-aware camera prediction. Multi-view fusion happens in SMPL parameter
space. Camera extrinsics are then RECOVERED from the SMPL mesh + 2D keypoints.

### Architecture Diagram

```
INPUT: ~60 images (EXIF-normalised) + optional point cloud (PLY/OBJ)
           │                                              │
           ▼                                              │
┌─────────────────────────────────────────────┐           │
│ STAGE 0: DETECTION & CLASSIFICATION         │           │
│                                             │           │
│  Per image:                                 │           │
│   1. PIL EXIF transpose normalisation       │           │
│   2. RT-DETR person detection (HuggingFace) │           │
│   3. ViTPose++ 2D keypoints (HuggingFace)   │           │
│   4. Classify: FULL_BODY / PARTIAL / SKIP   │           │
│   5. Extract EXIF intrinsics (focal length) │           │
│                                             │           │
│  Output: per-image 2D keypoints + confs,    │           │
│          view classifications, K matrices   │           │
└──────────────┬──────────────────────────────┘           │
               │                                          │
               ▼                                          │
┌─────────────────────────────────────────────┐           │
│ TIER 1: PER-VIEW HMR + CONSENSUS           │           │
│ (Zero calibration required)                 │           │
│                                             │           │
│  Per full-body view (~20 images):           │           │
│   • CameraHMR → SMPL params (β, θ)         │           │
│     + FoV estimate (5-7° error)             │           │
│     + 138 dense surface keypoints           │           │
│   • Fallback: PromptHMR (no registration)   │           │
│                                             │           │
│  Consensus fusion:                          │           │
│   β = confidence-weighted robust median     │           │
│   θ = SO(3) Fréchet mean (rotation space)   │           │
│   FoV = weighted median across views        │           │
│                                             │           │
│  Output: Coarse SMPL (~40-50mm PA-MPJPE)    │           │
│          Per-view FoV estimates              │           │
│          138 dense 2D surface keypoints      │           │
└──────────────┬──────────────────────────────┘           │
               │                                          │
               ▼                                          │
┌─────────────────────────────────────────────┐           │
│ TIER 2: SELF-CALIBRATION + REFINEMENT       │           │
│ (Human body as calibration target)          │           │
│                                             │           │
│  Step 2a: Build intrinsics per view         │           │
│   K from CameraHMR FoV (preferred)          │           │
│   or EXIF FocalLengthIn35mmFilm (fallback)  │           │
│                                             │           │
│  Step 2b: PnP extrinsic recovery            │           │
│   3D: SMPL joints + 138 dense surface pts   │           │
│   2D: ViTPose keypoints + CameraHMR 2D kps  │           │
│   → solvePnPRansac per view → [R|t]         │           │
│   (138 correspondences >> 12 sparse joints)  │           │
│                                             │           │
│  Step 2c: Multi-view triangulation          │           │
│   Confidence-weighted DLT + RANSAC          │           │
│   Using recovered [R|t] from all views      │           │
│   → Refined 3D keypoints                    │           │
│                                             │           │
│  Step 2d: SMPL optimisation                 │           │
│   Fit (β, θ, trans, scale) to:              │           │
│     - Triangulated 3D keypoints             │           │
│     - Reprojection loss (ALL 60 views)      │           │
│     - Pose prior (VPoser/GMM)               │           │
│     - Shape regularisation                  │           │
│                                             │           │
│  Output: Refined SMPL (target <25mm MPJPE)  │           │
│          Camera [R|t] per view              │           │
└──────────────┬──────────────────────────────┘           │
               │                                          │
               ▼                                          ▼
┌──────────────────────────────────────────────────────────────┐
│ TIER 3: SURFACE REFINEMENT (if point cloud available)        │
│                                                              │
│  Step 3a: Align point cloud TO the SMPL mesh                │
│   SMPL has correct metric scale + orientation                │
│   Rigid ICP: scale + rotation + translation                  │
│   (Meshroom's arbitrary coords get corrected here)           │
│                                                              │
│  Step 3b: Differentiable surface fitting                     │
│   Optimise SMPL params + optional per-vertex D               │
│   Losses:                                                    │
│     L_chamfer — semantic body-part weighted (Kaolin)         │
│     L_joints  — from Tier 2, downweighted                   │
│     L_normal  — surface normal consistency                   │
│     L_smooth  — Laplacian regularisation on D                │
│     L_prior   — pose/shape priors                            │
│                                                              │
│   Body-part weights:                                         │
│     torso: 1.0, arms: 0.7, legs: 0.7,                       │
│     head: 0.5, hands: 0.3, feet: 0.4                        │
│                                                              │
│  Output: Final SMPL (target <8mm chamfer)                    │
│          Optional SMPL+D with per-vertex displacements       │
└──────────────────────────────────────────────────────────────┘
```

### Why Three Tiers?

Each tier is independently useful and shippable:
- **Tier 1 alone**: Usable SMPL fit from images only. No calibration. ~2 min.
- **Tier 1+2**: Accurate SMPL registration with self-recovered cameras. ~3 min.
- **Tier 1+2+3**: Best accuracy, refines shape β using surface geometry. ~5 min.

Errors don't cascade — each tier improves on the previous.

---

## Technology Stack (March 2026)

### Tier 1: Per-View HMR

| Component | Choice | Why |
|-----------|--------|-----|
| **Primary HMR** | **CameraHMR** (`pixelite1201/CameraHMR`) | Full perspective camera model, 138 dense surface keypoints, FoV prediction (5-7° error). Weights via free registration at camerahmr.is.tue.mpg.de. Python 3.10, PyTorch 2.0+, CUDA 11.8+. 238 GitHub stars. |
| **Fallback HMR** | **PromptHMR** (`yufu-wang/PromptHMR`) | 36.6mm PA-MPJPE on 3DPW (17.6% better than HMR2.0). Weights on Google Drive, no registration. PyTorch 2.4-2.6. SMPL-X output. Use if CameraHMR registration is blocked. |
| **Person detection** | **RT-DETR** (`PekingU/rtdetr_r50vd_coco_o365`) | HuggingFace native. Fast, accurate bounding boxes. |
| **2D keypoints** | **ViTPose++-Base** (HuggingFace `usyd-community`) | 100M params, ~200MB, 4GB VRAM. Stable in transformers ≥5.1.0. COCO-17 + whole-body modes. |

### Tier 2: Self-Calibration

| Component | Choice | Why |
|-----------|--------|-----|
| **PnP solver** | OpenCV `solvePnPRansac` | 138 CameraHMR surface keypoints → far more robust PnP than 12 sparse joints. |
| **Intrinsics** | CameraHMR HumanFoV (primary), EXIF (fallback) | FoV from HumanFoV has 5-7° error; EXIF FocalLengthIn35mmFilm as backup. |
| **Triangulation** | Custom DLT + RANSAC | Confidence-weighted, outlier rejection. |
| **Optimisation** | PyTorch Adam/L-BFGS | Differentiable with loss weight annealing. |
| **Pose prior** | GMM from SMPLify (bundled) | VPoser as optional upgrade (requires separate download). |

### Tier 3: Surface Refinement

| Component | Choice | Why |
|-----------|--------|-----|
| **Chamfer distance** | **NVIDIA Kaolin** (`pip install kaolin`) | Apache 2.0. Pre-built wheels for PyTorch 2.1-2.8. GPU-optimised, differentiable. Includes `sided_distance`, `f_score`, `point_to_mesh_distance`. |
| **Chamfer fallback** | `torch.cdist` (pure PyTorch) | Zero dependencies. Viable for prototyping. |
| **Reference impl** | **DavidBoja/SMPL-Fitting** patterns | Chamfer + landmark + regularisation. Docker container. Plotly viz. |
| **Point cloud I/O** | Open3D ≥ 0.17 + trimesh | Load, denoise, downsample, ICP. |

### Shared

| Component | Choice | Notes |
|-----------|--------|-------|
| **SMPL layer** | `smplx` ≥ 0.1.28 (official PyTorch) | `pip install smplx`. Model .pkl files downloaded separately from smpl-x.is.tue.mpg.de (free registration, non-commercial license). |
| **Visualisation** | PyVista + matplotlib | 3D mesh overlay, joint comparison, loss curves. |
| **CLI** | Click | `scantosmpl fit-images`, `fit-pointcloud`, `fit-combined`. |

### Explicitly Avoided

| Tool | Why Avoided |
|------|-------------|
| **HMR2.0 / 4DHumans** | Unmaintained. detectron2 build failures on PyTorch 2.0+. GPU memory leak. |
| **PyTorch3D** | Last release Sep 2024. No support for PyTorch 2.5+. Install hell. Use Kaolin instead. |
| **MUC** | Pinned to PyTorch 1.12 + deprecated mmcv-full. Fails on modern GPUs. |
| **detectron2** | No pre-built wheels for PyTorch 2.0+. Source build requires exact CUDA matching. |
| **HeatFormer** | Requires pre-calibrated cameras. Not suitable for our calibration-free design. |

---

## Future Investigation: HSfM (CVPR 2025 Highlight)

**HSfM** (`hongsukchoi/HSfM_RELEASE`) jointly reconstructs people, places, and cameras
from multi-view images using HMR2.0 + DUSt3R. This could potentially **replace the
entire three-tier architecture** with a single forward pass. Code released, weights available.

Worth evaluating after the core pipeline is built — it may be a simpler path but trades
architectural control for convenience. Monitor for updates.

---

## Package Structure

```
scantosmpl/
├── config.py                 # Dataclass config with TOML/YAML support
├── types.py                  # ViewType, FittingResult, CameraParams, etc.
│
├── detection/
│   ├── image_loader.py       # EXIF transpose, intrinsics from EXIF/HumanFoV
│   ├── person_detector.py    # RT-DETR (HuggingFace transformers)
│   ├── keypoint_detector.py  # ViTPose++ (HuggingFace transformers)
│   └── view_classifier.py    # full-body/partial/skip classification
│
├── hmr/
│   ├── camera_hmr.py         # CameraHMR inference: SMPL + FoV + 138 dense kps
│   ├── prompt_hmr.py         # PromptHMR fallback inference
│   ├── consensus.py          # Multi-view β/θ fusion (SO(3) Fréchet mean)
│   └── orientation.py        # Global orientation from reference pose prior
│
├── calibration/
│   ├── intrinsics.py         # K from CameraHMR FoV or EXIF
│   ├── pnp_solver.py         # PnP [R|t] from SMPL 3D + 2D keypoints
│   └── bundle_adjust.py      # Optional joint intrinsic/extrinsic refinement
│
├── triangulation/
│   ├── dlt.py                # Direct Linear Transform
│   ├── ransac.py             # RANSAC outlier rejection
│   └── weighted.py           # Confidence-weighted multi-view triangulation
│
├── pointcloud/
│   ├── io.py                 # Load PLY/OBJ
│   ├── preprocess.py         # Denoise, downsample (Open3D)
│   ├── align.py              # Rigid ICP alignment TO SMPL mesh
│   └── segment.py            # Body-part segmentation (height slices + PCA)
│
├── smpl/
│   ├── model.py              # SMPL/SMPL-X wrapper (smplx library)
│   ├── prior.py              # GMM pose prior (VPoser optional)
│   ├── joint_map.py          # COCO-17 ↔ SMPL-24 + dense surface mapping
│   └── losses.py             # Joint, chamfer, normal, Laplacian, reprojection
│
├── fitting/
│   ├── coarse.py             # Tier 2d: staged SMPL optimisation
│   ├── surface.py            # Tier 3: chamfer + semantic weighting (Kaolin)
│   ├── reprojection.py       # Multi-view reprojection (all 60 views)
│   ├── scheduler.py          # Loss weight annealing schedules
│   └── pipeline.py           # End-to-end Tier 1→2→3 orchestrator
│
├── evaluation/
│   ├── metrics.py            # MPJPE, PA-MPJPE, chamfer, reprojection error
│   └── visualise.py          # Overlay SMPL on images/pointcloud, per-tier report
│
├── cli.py                    # Click CLI: fit-images, fit-pointcloud, fit-combined
└── utils/
    ├── geometry.py            # SO(3) averaging, rotation conversions, projection
    └── viz.py                 # Lightweight 3D/2D visualisation helpers
```

---

## Joint Mapping

```python
# ViTPose COCO-17 → SMPL-24 (for triangulation + PnP)
COCO_TO_SMPL = {
    5: 16, 6: 17,   # shoulders
    7: 18, 8: 19,   # elbows
    9: 20, 10: 21,  # wrists
    11: 1, 12: 2,   # hips
    13: 4, 14: 5,   # knees
    15: 7, 16: 8,   # ankles
}
# Derived: pelvis = mid(11,12)→SMPL 0, neck = mid(5,6)→SMPL 12

# CameraHMR provides 138 dense surface keypoints (COMA-sampled from SMPL vertices)
# These map DIRECTLY to SMPL vertex indices — no conversion needed.
# Used for PnP (>>12 sparse joints = much more robust extrinsic recovery)
```

---

## Dependencies

`.devcontainer/Dockerfile`

`./pyproject.toml`


```

### Model Downloads Required
1. **SMPL model files**: Register at smpl-x.is.tue.mpg.de → download .pkl files → place in `data/body_models/`
2. **CameraHMR weights**: Register at camerahmr.is.tue.mpg.de → download checkpoint
3. **PromptHMR weights** (fallback): Google Drive link in `yufu-wang/PromptHMR` README
4. **ViTPose++ / RT-DETR**: Auto-downloaded via HuggingFace on first run

### GPU Requirements
- Tier 1 (CameraHMR + ViTPose): ~8-10GB VRAM
- Tier 2 (optimisation): ~2-4GB VRAM
- Tier 3 (chamfer on 50K points): ~4-6GB VRAM
- **Recommended: 8GB+ GPU** (your RTX 3080Ti at 12GB is plenty)

---

## CLI

```bash
# Images only — no calibration, no point cloud needed
scantosmpl fit-images \
    --image-dir ./scan/images/ \
    --reference-pose a-pose \
    --gender neutral \
    --output ./output/

# Images + point cloud — best accuracy
scantosmpl fit-combined \
    --image-dir ./scan/images/ \
    --pointcloud ./scan/mesh.ply \
    --reference-pose a-pose \
    --output ./output/

# Point cloud only (uses point cloud keypoint extraction, no images)
scantosmpl fit-pointcloud \
    --pointcloud ./scan/mesh.ply \
    --gender male \
    --output ./output/

# If you later get calibration data from the university
scantosmpl fit-images \
    --image-dir ./scan/images/ \
    --calibration ./calibration.json \
    --skip-self-calibration \
    --output ./output/
```

---

## Key References

| Paper | Venue | Used For |
|-------|-------|----------|
| CameraHMR (Patel & Black) | 3DV 2025 | Tier 1 HMR + FoV + 138 dense keypoints |
| PromptHMR (Wang et al.) | CVPR 2025 | Tier 1 fallback HMR |
| ViTPose++ (Xu et al.) | TPAMI 2023 | 2D keypoint detection |
| U-HMR (Li et al.) | 2024 | Architecture reference for uncalibrated multi-view |
| HSfM (Choi et al.) | CVPR 2025 Highlight | Future: single-model pipeline replacement |
| SMPL-Fitting (Bojanić) | GitHub 2024 | Tier 3 optimisation patterns |
| SMPL/SMPL-X (Loper/Pavlakos) | SIGGRAPH 2015 / CVPR 2019 | Body model |
| Kaolin (NVIDIA) | — | Chamfer distance |
