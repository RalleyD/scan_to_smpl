# Phase 2 Specification: Per-View HMR — CameraHMR Integration

## Context

**Input**: 17 EXIF-corrected images + per-image `ViewResult` from Phase 1
(16 FULL_BODY, 1 PARTIAL = cam05_5.JPG — processed normally).

**Available checkpoints** (all in `models/`):

| File | Path | Size |
|------|------|------|
| CameraHMR main model | `models/checkpoints/camera_hmr/camerahmr_checkpoint_cleaned.ckpt` | 7.5 GB |
| DenseKP (138 kps) | `models/checkpoints/camera_hmr/densekp.ckpt` | 7.5 GB |
| FLNet (FoV estimation) | `models/checkpoints/camera_hmr/cam_model_cleaned.ckpt` | 768 MB |
| SMPL mean params | `models/smpl/smpl_mean_params.npz` | 1.3 KB |
| SMPL NEUTRAL | `models/smpl/SMPL_NEUTRAL.pkl` | 83 MB |

---

## What We're Building

For each of the 17 images:
1. Crop the person using the Phase 1 RT-DETR bbox
2. Run **CameraHMR** → SMPL params (β, θ, global_orient) + weak-perspective camera
3. Run **DenseKP** → 138 dense surface keypoints with learned confidence (criterion 2.3)
4. Run **FLNet** on the full image → independent FoV estimate (for criterion 2.2 cross-check)
5. Store everything in the existing `ViewResult` dataclass
6. Save wireframe overlay of posed SMPL mesh on source image (criterion 2.4)

---

## What We're NOT Building

| Excluded | Reason |
|----------|--------|
| `detectron2` | Project constraint. We use Phase 1 RT-DETR bboxes instead. |
| PromptHMR | Deferred. Would inflate deps massively (ultralytics, SAM2, mmcv). |

---

## Architecture: Vendored CameraHMR Model Code

Rather than cloning CameraHMR as a submodule (which would require managing its
heavy dependency graph including detectron2), we vendor the **minimal model
architecture files** into `scantosmpl/hmr/vendor/camerahmr/`.

### Why keep `pl.LightningModule` (not patch to `nn.Module`)

Lightning checkpoints use specific state_dict key structures. Loading into a bare
`nn.Module` risks key prefix mismatches — and `strict=False` would silently initialise
mismatched layers with random weights, which is catastrophic for inference quality
and extremely hard to debug. `load_from_checkpoint()` handles all key remapping,
hyperparameter restoration, and buffer loading automatically.

**Cost**: `pytorch-lightning>=2.0.2` as a pip dependency. Both CameraHMR and DenseKP
require it. Far safer than debugging silent weight-loading bugs.

### Modifications from upstream

- Path constants in `constants.py` replaced by constructor arguments (no hardcoded paths)
- No import-time detectron2 references (only appears in `mesh_estimator.py` which we don't vendor)
- `smpl_mean_params.npz` path passed to SMPL head constructor
- `VITPOSE_BACKBONE` path passed to ViT constructor (empty string — loaded from checkpoint anyway)

### Files to vendor (sourced from `pixelite1201/CameraHMR` at pinned commit):

```
scantosmpl/hmr/vendor/camerahmr/
├── VENDOR_INFO.md               ← source repo, commit SHA, modifications log
├── __init__.py                  ← exports CameraHMRModel, DenseKPModel, FLNet
├── model.py                     ← CameraHMR (pl.LightningModule)
├── densekp_model.py             ← DenseKP (pl.LightningModule) — 138 dense keypoints
├── backbones/
│   ├── __init__.py              ← create_backbone()
│   └── vit.py                   ← ViT-H backbone (uses timm internally)
├── heads/
│   ├── __init__.py
│   ├── smpl_head.py             ← SMPLTransformerDecoderHead → {β, θ, pred_cam, pred_kp}
│   └── keypoints_head.py        ← KeypointsHead → (138, 3) dense keypoints
├── components/
│   ├── __init__.py
│   ├── transformer.py           ← TransformerDecoder (shared by both heads)
│   └── t_cond_mlp.py            ← conditional MLP (used by transformer)
├── utils/
│   ├── __init__.py
│   └── geometry.py              ← rot6d_to_rotmat, aa_to_rotmat
└── cam_model/
    ├── __init__.py
    ├── fl_net.py                 ← FLNet: HRNet-48 → (hfov, vfov) in radians
    └── backbone/
        ├── __init__.py
        ├── hrnet.py              ← HRNet-48 backbone (23KB)
        └── utils.py             ← get_backbone_info
```

### Shared ViT backbone (memory optimisation)

CameraHMR and DenseKP both contain identical ViT-H backbones (~2.5GB each).
After loading both checkpoints, share the backbone reference:
`densekp.backbone = camerahmr.backbone`. Run the backbone once per image,
feed features to both heads. Saves ~2.5GB VRAM on 12GB GPU.

### New `pyproject.toml` dependencies:
```
pytorch-lightning>=2.0.2  # checkpoint loading for CameraHMR + DenseKP
timm>=0.6.12              # ViT-H backbone
einops>=0.6.0             # tensor rearrangement in SMPL head
yacs>=0.1.8               # HRNet config (FLNet backbone)
```

`scipy` is already a dependency — used for `Rotation.from_matrix().as_rotvec()`.

---

## Dense Keypoints Strategy (criterion 2.3)

The REVIEW.md criterion 2.3 requires `(138, 3)` per view — 138 keypoints with
confidence. We use the **DenseKP model** (`densekp.ckpt`, 7.5GB) for this.

DenseKP shares the same ViT-H backbone as CameraHMR and adds a separate
TransformerDecoder head that predicts 138 COMA-sampled surface keypoints.

**Output format**: `(B, 138, 3)` where:
- Dimensions 0-1: 2D coords in normalised crop space `[-0.5, 0.5]`
- Dimension 2: log-sigma (learned per-keypoint confidence/uncertainty)

**Post-processing**:
- To crop pixel coords: `kps_px = (kps_norm[:, :2] + 0.5) * 256`
- To original image coords: inverse the affine crop transform
- Confidence: `conf = exp(-abs(log_sigma))` → higher is more certain

**Backup**: SMPL vertex projection (project 138 COMA vertices using predicted
camera) is available if DenseKP quality is poor. Not implemented in Phase 2
unless verification shows issues.

---

## FoV / Focal Length Strategy

CameraHMR's forward pass requires the intrinsic matrix K as input. We have two sources:

| Source | Value | Use |
|--------|-------|-----|
| **EXIF** (Phase 1) | ~6349 px (Canon EOS 2000D, 24mm lens, FocalPlaneXRes) | **Primary K input** to CameraHMR |
| **FLNet** (cam_model) | Predicted from full image | **Cross-check** for criterion 2.2 |

FLNet outputs `(hfov, vfov)` in radians. Convert to focal length:
```python
fl_flnet = img_h / (2 * tan(vfov / 2))
fov_exif  = 2 * degrees(arctan(img_h / (2 * fl_exif)))
fov_flnet = 2 * degrees(arctan(img_h / (2 * fl_flnet)))
# criterion 2.2: abs(fov_exif - fov_flnet) < 10°
```

FLNet takes the **full image** (not the person crop), resized to 256×256 and
normalized with ImageNet mean/std — the same normalization used for the person crop.

---

## Processing Pipeline (Per Image)

```
Phase 1 ViewResult
  └── image_path, bbox, keypoints_2d, camera.focal_length_px

Step 1 — Build intrinsic matrix K
    K = [[fl_px, 0, W/2],
         [0, fl_px, H/2],
         [0, 0,      1 ]]

Step 2 — Prepare person crop (for CameraHMR main model)
    center = [(x1+x2)/2, (y1+y2)/2]
    scale  = max(x2-x1, y2-y1) / 200.0   # CenterHMR convention
    crop   = affine_crop(image, center, scale, output_size=256)
    crop_tensor = ImageNet_normalise(crop)  # (3, 256, 256)

    batch = {
        'img'        : crop_tensor,           # (B, 3, 256, 256)
        'box_center' : center,                # (B, 2) in original image coords
        'box_size'   : scale * 200.0,         # (B,) = max(w, h) of bbox
        'img_size'   : (H, W),                # (B, 2) original image dimensions
        'cam_int'    : K,                     # (B, 3, 3)
    }

Step 3 — CameraHMR forward pass
    pred_smpl_params, pred_cam, _ = camerahmr_model(batch)
    # pred_smpl_params = {
    #   'global_orient': (B, 1, 3, 3),   ← rotation matrix
    #   'body_pose'    : (B, 23, 3, 3),  ← rotation matrices
    #   'betas'        : (B, 10),
    # }
    # pred_cam: (B, 3) = [scale, tx, ty]  weak-perspective

Step 4 — Convert rotation matrices → axis-angle
    global_orient_aa = rotmat_to_aa(pred_smpl_params['global_orient'][:, 0])  # (B, 3)
    body_pose_aa = rotmat_to_aa(pred_smpl_params['body_pose'].reshape(-1, 3, 3))
                   .reshape(B, 69)

Step 5 — Convert weak-perspective camera → full 3D translation
    # CLIFF conversion:
    tz = 2 * fl_px / (bbox_size * pred_cam[:, 0])
    tx = (center_x - W/2) / fl_px + pred_cam[:, 1] * bbox_size / fl_px
    ty = (center_y - H/2) / fl_px + pred_cam[:, 2] * bbox_size / fl_px
    cam_trans = stack([tx, ty, tz])  # (B, 3) — camera-space translation

Step 6 — FLNet FoV estimation (run separately, full image)
    full_img_256 = ImageNet_normalise(resize(full_image, 256))  # (1, 3, 256, 256)
    pred_fov, _ = flnet(full_img_256)                          # (1, 2)
    vfov_rad = pred_fov[0, 1]
    fl_flnet = img_h / (2 * tan(vfov_rad / 2))
    fov_deg_flnet = degrees(2 * arctan(img_h / (2 * fl_flnet)))
    fov_deg_exif  = degrees(2 * arctan(img_h / (2 * fl_exif)))

Step 7 — Dense keypoints via DenseKP model
    # DenseKP shares backbone with CameraHMR (already computed in Step 3)
    dense_kps_raw = densekp_head(conditioning_feats)       # (B, 138, 3)
    kps_norm = dense_kps_raw['pred_keypoints'][0]           # (138, 3) normalised
    kps_crop = (kps_norm[:, :2] + 0.5) * 256                # (138, 2) crop pixels
    kps_image = inverse_affine(kps_crop, center, scale)      # (138, 2) original image coords
    conf = exp(-abs(kps_norm[:, 2]))                          # (138,) learned confidence

Step 8 — Populate ViewResult
    result.betas           = betas[0].cpu().numpy()          # (10,)
    result.body_pose       = body_pose_aa[0]                 # (69,)
    result.global_orient   = global_orient_aa[0]             # (3,)
    result.camera.fov      = fov_deg_flnet                   # degrees, from FLNet
    result.dense_keypoints_2d  = dense_kps[:, :2]           # (138, 2)
    result.dense_keypoint_confs = dense_kps[:, 2]           # (138,)
    # cam_trans stored in camera — extend CameraParams with hmr_translation field
```

---

## Deliverables

### 1. `scantosmpl/hmr/vendor/camerahmr/` (~18 files)

Source from `pixelite1201/CameraHMR` at pinned commit (SHA in `VENDOR_INFO.md`).

**Modifications from upstream** (keep `pl.LightningModule` — see Architecture section):
- `constants.py`: Replace hardcoded paths with overridable values or constructor args.
- `model.py`: Accept `smpl_mean_params_path` as `__init__` kwarg (loaded in SMPL head).
- `densekp_model.py`: Same pattern — configurable paths.
- `fl_net.py`: Accept `pretrained_ckpt_path` as arg (already does, pass `''`).
- No detectron2 import sites (only `mesh_estimator.py` imports it — not vendored).

### 2. `scantosmpl/hmr/camera_hmr.py`

```python
class CameraHMRInference:
    """CameraHMR + DenseKP + FLNet inference wrapper. No detectron2.
    
    Loads all three models. CameraHMR + DenseKP share the ViT-H backbone
    in memory. FLNet (nn.Module) loaded via torch.load + state_dict.
    """

    def __init__(self, config: HMRConfig, device: str = "cuda"):
        # Load CameraHMR via load_from_checkpoint() (pl.LightningModule)
        # Load DenseKP via load_from_checkpoint(), share backbone
        # Load FLNet via torch.load + state_dict (nn.Module, no Lightning)

    @torch.no_grad()
    def infer(
        self,
        image: PIL.Image.Image,   # EXIF-corrected image from Phase 1
        bbox: np.ndarray,         # (4,) x1,y1,x2,y2 from RT-DETR
        focal_length_px: float,   # from Phase 1 EXIF extraction
    ) -> HMROutput

    @torch.no_grad()
    def infer_batch(
        self,
        images: list[PIL.Image.Image],
        bboxes: list[np.ndarray],
        focal_lengths: list[float],
    ) -> list[HMROutput]
```

```python
@dataclass
class HMROutput:
    betas: np.ndarray           # (10,)
    body_pose: np.ndarray       # (69,) axis-angle, 23 joints × 3
    global_orient: np.ndarray   # (3,) axis-angle
    pred_cam: np.ndarray        # (3,) weak-perspective [s, tx, ty]
    cam_trans: np.ndarray       # (3,) full 3D translation in camera space
    fov_deg_exif: float         # FoV in degrees from EXIF focal length
    fov_deg_flnet: float        # FoV in degrees from FLNet (cross-check)
    dense_keypoints: np.ndarray # (138, 2) DenseKP predicted 2D coords (original image space)
    dense_confs: np.ndarray     # (138,) learned per-keypoint confidence
```

### 3. `scantosmpl/hmr/orientation.py`

Minimal validity check for Phase 2. Phase 3 consensus handles full alignment.

```python
def check_orientation_quality(
    global_orient: np.ndarray,   # (3,) axis-angle
    keypoints_2d: np.ndarray,    # (17, 2) ViTPose keypoints from Phase 1
    image_hw: tuple[int, int],
) -> OrientationQuality
    """Return quality score + flags.
    
    Checks:
    - Nose keypoint is above hips in image (body upright, not inverted)
    - Global orient axis-angle magnitude is in plausible range (< 2π)
    - For T-pose: elbow y-coords are within 20% of shoulder y-coords
    """

@dataclass
class OrientationQuality:
    score: float          # 0.0 (bad) – 1.0 (good)
    upright: bool         # nose above hips
    plausible_magnitude: bool
    t_pose_arms: bool | None  # None if can't determine
    flags: list[str]      # human-readable warnings
```

### 4. `scantosmpl/hmr/pipeline.py`

Orchestrates Phase 2 across all images. Mirrors `detection/pipeline.py` structure.

```python
class HMRPipeline:
    """Run CameraHMR on all views from Phase 1. Saves debug outputs."""

    def __init__(
        self,
        config: HMRConfig,
        device: str = "cuda",
    )

    def process_views(
        self,
        views: list[ViewResult],
        image_dir: Path,
        debug_dir: Path | None = None,
    ) -> list[ViewResult]
    """Mutates ViewResult in-place: adds betas, body_pose, global_orient,
    camera.fov, dense_keypoints_2d, dense_keypoint_confs.
    Skips ViewType.SKIP views."""
```

Debug output (saved to `output/debug/hmr/`):
- `hmr_results.json` — per-image: betas (10,), body_pose norms, FoV values, fov_diff_deg
- `{stem}_hmr_overlay.jpg` — SMPL mesh wireframe projected onto source image
- `summary.txt` — FoV cross-check table, β stats, θ variance summary

### 5. `scantosmpl/hmr/__init__.py`

```python
from scantosmpl.hmr.camera_hmr import CameraHMRInference, HMROutput
from scantosmpl.hmr.orientation import check_orientation_quality, OrientationQuality
from scantosmpl.hmr.pipeline import HMRPipeline
```

### 6. `scantosmpl/config.py` — HMRConfig update

```python
@dataclass
class HMRConfig:
    backend: Literal["camerahmr", "prompthmr"] = "camerahmr"
    device: str = "cuda"

    # Checkpoint paths (relative to project root)
    checkpoint_path: Path = Path("models/checkpoints/camera_hmr/camerahmr_checkpoint_cleaned.ckpt")
    cam_model_path: Path  = Path("models/checkpoints/camera_hmr/cam_model_cleaned.ckpt")
    smpl_mean_params_path: Path = Path("models/smpl/smpl_mean_params.npz")
    smpl_model_path: Path = Path("models/smpl/SMPL_NEUTRAL.pkl")

    # Processing
    batch_size: int = 4
    crop_size: int = 256         # CameraHMR expects 256×256
    crop_padding: float = 0.1    # 10% padding around bbox
    process_partial_views: bool = True  # include cam05_5

    # Debug
    save_debug: bool = True
    debug_dir: Path = Path("output/debug/hmr")
```

### 7. `tests/test_hmr.py` (unit — no GPU, no real images)

```python
class TestHMROutputConversion:
    """Rotation matrix → axis-angle conversion."""
    def test_identity_rotmat_gives_zero_aa()
    def test_known_rotation_roundtrips()
    def test_batch_conversion_shape()

class TestCropPreprocessing:
    """Person crop generation from bbox."""
    def test_crop_output_shape_256x256()
    def test_center_scale_from_bbox()
    def test_imagenet_normalisation_range()
    def test_bbox_with_aspect_ratio_preserved()

class TestFoVConversion:
    """Focal length ↔ FoV conversion."""
    def test_known_fov_from_focal_length()
    def test_fl_from_vfov_radians()
    def test_round_trip_fl_fov_fl()

class TestDenseKeypointProjection:
    """SMPL vertex → 2D projection."""
    def test_projection_shape_138x2()
    def test_in_image_bounds_for_frontal_view()
    def test_visibility_conf_between_0_and_1()
    def test_coma_indices_valid_smpl_range()  # all < 6890

class TestOrientationQuality:
    """Orientation validity checks."""
    def test_upright_body_scores_high()
    def test_inverted_body_flagged()
    def test_t_pose_arms_check_with_vitpose_keypoints()
    def test_missing_keypoints_handled_gracefully()

class TestHMROutputDataclass:
    """HMROutput shape contracts."""
    def test_betas_shape_10()
    def test_body_pose_shape_69()
    def test_global_orient_shape_3()
    def test_dense_keypoints_shape_138x2()
```

### 8. `tests/integration/test_hmr_integration.py` (requires GPU + checkpoints)

```python
@pytest.mark.gpu
@pytest.mark.slow
class TestCameraHMRSingleImage:
    """Single image sanity checks."""
    def test_smpl_params_shapes()
    def test_betas_plausible_range()     # abs(β) < 5 for most components
    def test_body_pose_rotation_valid()  # ||axis-angle|| < 3π
    def test_fov_within_bounds()         # 10° < FoV < 120°

@pytest.mark.gpu
@pytest.mark.slow
class TestFoVCrossCheck:
    """Criterion 2.2: FLNet vs EXIF FoV."""
    def test_fov_diff_under_10deg_all_images()  # criterion 2.2
    def test_flnet_runs_without_crash()

@pytest.mark.gpu
@pytest.mark.slow
class TestHMRPipelineAllImages:
    """End-to-end on all 17 scanner images."""
    def test_all_17_produce_output()
    def test_beta_std_under_1_per_component()   # criterion 2.5
    def test_dense_keypoints_shape_per_view()   # criterion 2.3
    def test_cam05_5_produces_output()           # partial view handled
    def test_inference_time_under_60s()          # criterion 2.7
    def test_fov_consistency_across_views()      # FoV spread < 10°
    def test_debug_output_files_created()
```

---

## `types.py` additions

Add one field to `CameraParams` and one field to `ViewResult`:

```python
@dataclass
class CameraParams:
    focal_length: float
    principal_point: tuple[float, float] = (0.0, 0.0)
    fov: float | None = None          # degrees, from FLNet
    hmr_translation: np.ndarray | None = None  # (3,) cam-space translation from CameraHMR

@dataclass
class ViewResult:
    # ... existing fields ...
    # HMR (Phase 2) — dense_keypoints_2d and dense_keypoint_confs already exist
    # No new fields needed — all populated via existing Phase 2 slots
```

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 2.1 | SMPL output shapes correct (β: 10D, θ: 69D) | Shape asserts in integration test |
| 2.2 | FoV within 10° of EXIF-derived FoV | FLNet cross-check — EXIF is ground truth |
| 2.3 | 138 dense surface keypoints per view, shape (138, 3) | DenseKP model — independent neural prediction with learned confidence |
| 2.4 | Meshes visually align with source images | **Wireframe overlay** on source image per view (PIL-based, headless) |
| 2.5 | Shape β std < 1.0 per component across views | Stats in integration test |
| 2.6 | Body pose θ variance low for T-pose | Stats in integration test |
| 2.7 | 17 images in < 60s on GPU | Timing test |

---

## Implementation Order

1. **Vendor CameraHMR files** — fetch ~18 files from repo, patch path constants
2. **Update `pyproject.toml`** — add pytorch-lightning, timm, einops, yacs; `pip install -e .`
3. **Verify vendored imports** — checkpoint loads, forward pass shape check
4. **`scantosmpl/hmr/camera_hmr.py`** — inference wrapper (crop, backbone sharing, all 3 models)
5. **`scantosmpl/hmr/orientation.py`** — orientation quality checker
6. **`scantosmpl/hmr/pipeline.py`** — orchestrator + wireframe overlay + debug output
7. **Update `config.py`** and **`types.py`**
8. **`tests/test_hmr.py`** — unit tests (no GPU)
9. **`tests/integration/test_hmr_integration.py`** — integration tests

---

## Debug Commands

```bash
# Unit tests only (no GPU)
pytest tests/test_hmr.py -v

# Integration tests (requires GPU + checkpoints loaded)
pytest tests/integration/test_hmr_integration.py -v -m gpu

# Run HMR pipeline on scanner images with debug output
python -c "
from pathlib import Path
from scantosmpl.config import HMRConfig
from scantosmpl.detection.pipeline import DetectionPipeline
from scantosmpl.hmr.pipeline import HMRPipeline

det = DetectionPipeline(device='cuda')
views = det.process_directory(Path('data/t-pose/jpg'))

hmr_cfg = HMRConfig()
hmr = HMRPipeline(hmr_cfg, device='cuda')
views = hmr.process_views(views, Path('data/t-pose/jpg'), debug_dir=Path('output/debug/hmr'))

print('Done. Check output/debug/hmr/summary.txt')
"
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CameraHMR ViT backbone uses incompatible timm API | Medium | Pin timm>=0.6.12; test import at vendor time |
| smpl_mean_params.npz shape mismatch | Low | File is present (1.3KB), validate shapes on load |
| FLNet input size mismatch | Low | Resize full image to 256×256 (HRNet-48 uses AdaptiveAvgPool — size-agnostic) |
| cam05_5 (PARTIAL) produces poor SMPL | Low | HMR handles partial crops; orientation quality check will flag if bad |
| 7.5GB checkpoint OOM on load | Low | RTX 3080Ti has 12GB; checkpoint loads to CPU first then `.to(device)` |
