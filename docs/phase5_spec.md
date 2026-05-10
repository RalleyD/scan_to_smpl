# Phase 5 Specification: Multi-View Triangulation + SMPL Refinement

## Context

**Input**: Consensus SMPL mesh from Phase 3 + per-view ViTPose keypoints from Phase 1.

| Source | Data | Quality |
|--------|------|---------|
| Phase 3 consensus | SMPL params (betas, body_pose), 6890 vertices, 24 joints | PA-MPJPE ~32mm |
| Phase 1 detection | 17 views, each with 17 COCO keypoints + confidences | ViTPose++ confidence 0.3-0.99 |
| COLMAP SfM | 60-view reconstruction, SIMPLE_RADIAL cameras, [R\|t] per image | Sub-pixel reprojection accuracy |
| Phase 4 self-cal | 17 views with PnP-derived [R\|t] | Mean reproj ~52px, 4/17 views ~180deg error |

**COLMAP cameras** (from `auto-rigger/data/reconstruction/t-pose/0/`):
- Camera 1: f=14230px, cx=3014, cy=1984, k1=-0.101 (6000x4000) — used by close-up images
- Camera 2: f=6678px, cx=2999, cy=1997, k1=-0.099 (6000x4000) — **all 17 Phase 1 views**
- Model: SIMPLE_RADIAL (f, cx, cy, k1)

**Key insight**: COLMAP provides sub-pixel camera calibration for the exact same images we have keypoints for. Using COLMAP extrinsics instead of Phase 4 self-calibration eliminates the ~52px reprojection error and the 4 incorrect camera poses.

---

## What We're Building

1. **COLMAP reader** — parse `cameras.bin` + `images.bin` to extract [R|t] per view
2. **Frame alignment** — 7-DoF Procrustes to transform COLMAP's arbitrary coordinate frame into SMPL's canonical frame
3. **Radial undistortion** — remove SIMPLE_RADIAL lens distortion from 2D keypoints
4. **Multi-view triangulation** — DLT with RANSAC to compute 3D joint positions from 2D observations across calibrated views
5. **SMPL optimisation** — refine consensus SMPL parameters to fit the triangulated 3D joints + 2D reprojection

**Output**: Refined SMPL parameters (betas, body_pose, global_orient, translation, scale), per-view camera extrinsics in SMPL frame, triangulated 3D keypoints, quality metrics.

---

## What We're NOT Building

| Excluded | Reason |
|----------|--------|
| Processing all 60 COLMAP views | Only 17 have ViTPose keypoints; extending to 60 would require re-running Phase 1 |
| Joint camera + SMPL refinement | COLMAP cameras are sub-pixel accurate; keep them fixed |
| Bundle adjustment | Not needed — COLMAP already did this |
| Tier 3 surface refinement (chamfer) | Separate phase requiring point cloud input |
| Dense keypoint triangulation | CameraHMR DenseKP are single-image predictions, not suitable for multi-view triangulation |

---

## Design Decisions

### 1. COLMAP extrinsics as primary, self-calibration as fallback

COLMAP provides sub-pixel accuracy for 60 views via SfM. Our 17 views are a subset.
Self-calibration (Phase 4) has ~52px mean reprojection error and 4/17 incorrect poses —
usable as a proof-of-concept but not for high-quality triangulation.

**Implementation**: Build the full pipeline with COLMAP extrinsics. The self-calibration
fallback is a config switch (`extrinsics_source: "colmap" | "self_calibration"`) that
uses Phase 4 results instead. The fallback path shares all downstream code (triangulation,
optimisation) — only the camera loading differs.

### 2. 7-DoF Procrustes alignment (COLMAP → SMPL frame)

COLMAP operates in an arbitrary coordinate frame (SfM has no absolute scale or orientation).
We need to align it to SMPL's canonical frame where the consensus mesh lives.

**Approach**: Use the 12 COCO-to-SMPL joint correspondences as anchor points:
1. Triangulate joints in COLMAP frame using COLMAP cameras + ViTPose 2D keypoints
2. The same joints exist in the consensus mesh (SMPL canonical frame)
3. Solve 7-DoF Procrustes (scale + rotation + translation) to map COLMAP → SMPL

This aligns the cameras so that projecting consensus SMPL joints through aligned cameras
reproduces the ViTPose 2D detections. The existing `procrustes_align()` in
`scantosmpl/utils/geometry.py` handles this (Umeyama method).

### 3. Undistort 2D keypoints before triangulation

COLMAP's SIMPLE_RADIAL model has radial distortion k1 ≈ -0.1. At image edges (r ≈ 2000px
from center), this shifts points by ~40px. Undistortion ensures DLT triangulation uses
a pinhole camera model consistently.

**Formula** (SIMPLE_RADIAL undistortion):
```
x_n = (x - cx) / f       # normalised coords
y_n = (y - cy) / f
r² = x_n² + y_n²
x_undist = x_n / (1 + k1 * r²)    # iterative refinement needed for accuracy
y_undist = y_n / (1 + k1 * r²)
x_px = x_undist * f + cx  # back to pixel coords
y_px = y_undist * f + cy
```

Use `cv2.undistortPoints()` for the iterative inverse — it handles the
distortion → undistortion inversion correctly.

### 4. DLT triangulation with RANSAC

For each of the 14 joints (12 direct + 2 midpoints), we have up to 17 2D observations.
Not all observations are reliable (occluded joints, rear-view hallucinations).

**Algorithm**:
1. For each joint, collect all views where confidence > threshold
2. If ≥ 2 views available: run DLT (Direct Linear Transform) triangulation
3. RANSAC wrapper: subsample view pairs, triangulate, check reprojection against held-out
   views. Keep the 3D point with most inlier views (reproj < threshold).
4. Final triangulation: DLT using all inlier views for maximum accuracy.

**Confidence-weighted DLT**: Each 2D observation contributes to the linear system weighted
by its ViTPose confidence score. Higher-confidence detections have more influence on the
triangulated 3D position.

### 5. SMPL optimisation: staged, consensus-initialised

Initialise from Phase 3 consensus parameters. Optimise in stages to avoid local minima:

**Stage 1 — Global alignment (50 iterations)**:
- Optimise: global_orient, translation, scale
- Loss: L_joint (triangulated 3D joints)
- Purpose: Align the consensus mesh to the triangulated skeleton

**Stage 2 — Shape refinement (100 iterations)**:
- Optimise: betas, global_orient, translation, scale
- Loss: L_joint + L_reproj + L_shape_reg
- Purpose: Adjust body shape to match multi-view observations

**Stage 3 — Full refinement (200 iterations)**:
- Optimise: betas, body_pose, global_orient, translation, scale
- Loss: L_joint + L_reproj + L_pose_prior + L_shape_reg
- Purpose: Fine-tune pose and shape together

### 6. Loss functions

```
L_total = w_joint * L_joint
        + w_reproj * L_reproj
        + w_pose_prior * L_pose_prior
        + w_shape_reg * L_shape_reg

L_joint = mean(||J_smpl - J_triang||²)             # 3D joint position error
L_reproj = mean(||proj(J_smpl, R_i, t_i, K_i) - kp2d_i||² * conf_i)  # 2D reprojection
L_pose_prior = ||body_pose||²                        # regularise toward neutral pose
L_shape_reg = ||betas||²                             # regularise toward mean shape
```

Weights from `FittingConfig`: w_joint=1.0, w_reproj=0.5, w_pose_prior=0.01, w_shape_reg=0.01.

**Reprojection loss**: For each view with COLMAP extrinsics, project SMPL joints through
the (aligned) camera and compare to undistorted ViTPose keypoints, weighted by confidence.
This uses all 17 views simultaneously — more observations than triangulation alone.

---

## Deliverables

### 1. `scantosmpl/calibration/colmap_reader.py` — COLMAP binary parser

```python
@dataclass
class ColmapCamera:
    camera_id: int
    model: str                    # "SIMPLE_RADIAL"
    width: int
    height: int
    focal_length: float
    cx: float
    cy: float
    k1: float                    # radial distortion

@dataclass
class ColmapImage:
    image_id: int
    name: str
    camera_id: int
    rotation: np.ndarray          # (3, 3) from quaternion
    translation: np.ndarray       # (3,)

def read_colmap_model(model_dir: Path) -> tuple[dict[int, ColmapCamera], dict[str, ColmapImage]]:
    """Parse cameras.bin + images.bin. Returns cameras by ID, images by name."""
```

### 2. `scantosmpl/calibration/frame_alignment.py` — COLMAP → SMPL alignment

```python
@dataclass
class FrameAlignment:
    scale: float
    rotation: np.ndarray          # (3, 3)
    translation: np.ndarray       # (3,)

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """Apply s * R @ p + t."""

    def transform_camera(self, R_colmap: np.ndarray, t_colmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform COLMAP camera [R|t] into SMPL frame."""

def compute_frame_alignment(
    pts_colmap: np.ndarray,       # (N, 3) joint positions in COLMAP frame
    pts_smpl: np.ndarray,         # (N, 3) corresponding joints in SMPL canonical frame
) -> FrameAlignment:
    """7-DoF Procrustes alignment. Reuses scantosmpl.utils.geometry.procrustes_align."""
```

### 3. `scantosmpl/calibration/undistort.py` — Radial undistortion

```python
def undistort_keypoints(
    keypoints_2d: np.ndarray,     # (N, 2) distorted pixel coords
    camera: ColmapCamera,         # COLMAP camera with distortion params
) -> np.ndarray:
    """Undistort 2D keypoints. Returns (N, 2) undistorted pixel coords.
    Uses cv2.undistortPoints for iterative inverse."""

def build_pinhole_K(camera: ColmapCamera) -> np.ndarray:
    """Build 3x3 intrinsic matrix (pinhole, no distortion) from COLMAP camera."""
```

### 4. `scantosmpl/triangulation/dlt.py` — DLT triangulation

```python
def triangulate_point(
    pts_2d: np.ndarray,           # (V, 2) observations from V views
    projections: list[np.ndarray], # V projection matrices P = K @ [R|t], each (3, 4)
    weights: np.ndarray | None = None,  # (V,) confidence weights
) -> np.ndarray:
    """Triangulate a single 3D point from V ≥ 2 views via weighted DLT.
    Returns (3,) world coordinates."""

def triangulate_joints(
    keypoints_per_view: dict[str, np.ndarray],  # view_name -> (J, 2) undistorted kps
    confs_per_view: dict[str, np.ndarray],       # view_name -> (J,) confidences
    projections: dict[str, np.ndarray],           # view_name -> (3, 4) projection matrix
    joint_indices: list[int],                     # which COCO joints to triangulate
    min_views: int = 2,
    conf_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate all joints. Returns (J, 3) positions + (J,) quality scores."""
```

### 5. `scantosmpl/triangulation/ransac.py` — RANSAC wrapper

```python
def ransac_triangulate_point(
    pts_2d: np.ndarray,           # (V, 2)
    projections: list[np.ndarray], # V projection matrices
    weights: np.ndarray | None,
    reproj_threshold: float = 10.0,  # px
    min_inlier_views: int = 2,
    n_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """RANSAC-robust triangulation.
    Returns (3,) point, (V,) inlier mask, mean reproj error."""
```

### 6. `scantosmpl/fitting/losses.py` — Differentiable loss functions

```python
def joint_loss(
    joints_pred: torch.Tensor,    # (1, J, 3)
    joints_target: torch.Tensor,  # (J, 3)
    joint_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """L2 joint position loss."""

def reprojection_loss(
    joints_pred: torch.Tensor,    # (1, J, 3) world-space SMPL joints
    keypoints_2d: dict[str, torch.Tensor],  # view_name -> (J, 2)
    confs: dict[str, torch.Tensor],          # view_name -> (J,)
    cameras: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],  # R, t, K per view
) -> torch.Tensor:
    """Confidence-weighted multi-view reprojection loss."""

def pose_prior_loss(body_pose: torch.Tensor) -> torch.Tensor:
    """L2 regularisation toward neutral pose."""

def shape_regularisation(betas: torch.Tensor) -> torch.Tensor:
    """L2 regularisation toward mean shape."""
```

### 7. `scantosmpl/fitting/optimiser.py` — Staged SMPL optimiser

```python
@dataclass
class OptimisationStage:
    name: str
    params: list[str]             # e.g. ["global_orient", "translation", "scale"]
    n_iterations: int
    loss_weights: dict[str, float]

@dataclass
class RefinementResult:
    betas: np.ndarray             # (10,)
    body_pose: np.ndarray         # (69,)
    global_orient: np.ndarray     # (3,)
    translation: np.ndarray       # (3,)
    scale: float
    vertices: np.ndarray          # (6890, 3)
    joints: np.ndarray            # (24, 3)
    loss_history: dict[str, list[float]]
    metrics: dict[str, float]     # mpjpe, pa_mpjpe, mean_reproj_error

class SMPLOptimiser:
    def __init__(self, smpl_model: SMPLModel, config: FittingConfig)

    def refine(
        self,
        consensus: ConsensusResult,
        triangulated_joints: np.ndarray,  # (J, 3) in SMPL frame
        keypoints_2d: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
        cameras: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        stages: list[OptimisationStage] | None = None,
    ) -> RefinementResult:
        """Run staged SMPL optimisation."""
```

### 8. `scantosmpl/fitting/pipeline.py` — Phase 5 orchestrator

```python
@dataclass
class Phase5Config:
    extrinsics_source: str = "colmap"  # "colmap" or "self_calibration"
    colmap_model_dir: Path | None = None
    triangulation_conf_threshold: float = 0.3
    triangulation_min_views: int = 3
    ransac_reproj_threshold: float = 10.0  # px
    ransac_iterations: int = 100
    save_debug: bool = True
    debug_dir: Path = Path("output/debug/refinement")

@dataclass
class Phase5Result:
    refined: RefinementResult
    triangulated_joints: np.ndarray   # (J, 3) in SMPL frame
    triangulation_quality: np.ndarray # (J,) per-joint quality score
    cameras_smpl_frame: dict[str, tuple[np.ndarray, np.ndarray]]  # R, t per view
    frame_alignment: FrameAlignment | None
    extrinsics_source: str
    metrics: dict[str, float]

class Phase5Pipeline:
    def run(
        self,
        views: list[ViewResult],
        consensus: ConsensusResult,
        image_dir: Path,
        config: Phase5Config,
        calibration_result: CalibrationResult | None = None,
    ) -> Phase5Result
```

### 9. `scantosmpl/config.py` — Add Phase5Config

Add `Phase5Config` dataclass and wire it into `PipelineConfig`.

### 10. `tests/test_triangulation.py` — Unit tests (no GPU)

```python
class TestDLT:
    def test_two_views_known_point()       # synthetic cameras + known 3D point
    def test_weighted_dlt()                 # higher-weight obs → closer to its ray
    def test_behind_camera_rejected()       # point behind one camera → handled

class TestRANSAC:
    def test_outlier_rejection()            # 1 outlier in 5 views → rejected
    def test_all_inliers()                  # clean data → all views retained

class TestUndistortion:
    def test_roundtrip()                    # distort → undistort ≈ identity
    def test_center_unchanged()             # principal point stays fixed
    def test_edge_correction()              # large r → significant correction

class TestFrameAlignment:
    def test_identity_alignment()           # same coords → scale=1, R=I, t=0
    def test_known_transform()              # apply known s,R,t → recover it
    def test_camera_transform()             # camera [R|t] correctly transformed
```

### 11. `tests/test_fitting.py` — Unit tests (no GPU)

```python
class TestLossFunctions:
    def test_joint_loss_zero_at_target()
    def test_reprojection_loss_clean_projection()
    def test_confidence_weighting()

class TestSMPLOptimiser:
    def test_converges_on_synthetic()       # perturbed params → optimiser recovers
```

### 12. `tests/integration/test_phase5_integration.py` — Integration tests (GPU)

```python
class TestPhase5COLMAP:
    def test_colmap_reader()                 # 5.1: reads 60 images, 2 cameras
    def test_frame_alignment_quality()       # 5.2: reproj < 15px after alignment
    def test_triangulation_accuracy()        # 5.3: PA-MPJPE < 30mm (vs consensus)
    def test_smpl_refinement()               # 5.4: PA-MPJPE improves over consensus
    def test_reprojection_error()            # 5.5: mean reproj < 15px
    def test_debug_output()                  # 5.6: JSON, summary, plots created
```

---

## Processing Pipeline

```
Step 0: Load inputs
  - Phase 3 consensus (vertices, joints, SMPL params)
  - Phase 1 per-view ViTPose keypoints + confidences
  - COLMAP model (cameras.bin, images.bin) OR Phase 4 calibration result

Step 1: Load camera extrinsics
  if extrinsics_source == "colmap":
    1a. Parse COLMAP cameras.bin → camera intrinsics + distortion
    1b. Parse COLMAP images.bin → per-view [R|t] (quaternion → rotation matrix)
    1c. Match COLMAP image names to our 17 views
  else:
    1a. Use Phase 4 CalibrationResult (view.camera.rotation, .translation)
    1b. Skip undistortion (Phase 4 uses pinhole model)

Step 2: Undistort 2D keypoints (COLMAP path only)
  For each view:
    2a. Get COLMAP camera params (f, cx, cy, k1)
    2b. cv2.undistortPoints(keypoints_2d, K, dist_coeffs) → undistorted keypoints
    2c. Store undistorted keypoints for triangulation + reprojection loss

Step 3: Initial triangulation (in COLMAP frame)
  For each of 14 joints (12 direct COCO→SMPL + 2 midpoints):
    3a. Collect 2D observations across views (conf > threshold)
    3b. Build projection matrices P_i = K @ [R_i | t_i] per view
    3c. RANSAC-DLT: subsample pairs → triangulate → check reproj → keep best
    3d. Final DLT with all inlier views → 3D joint position

Step 4: Frame alignment (COLMAP → SMPL)
  4a. Triangulated joints (Step 3) are in COLMAP's arbitrary frame
  4b. Consensus joints are in SMPL canonical frame
  4c. 7-DoF Procrustes (Umeyama): find s, R, t to align triangulated → consensus
  4d. Transform all COLMAP cameras into SMPL frame
  4e. Re-triangulate joints using aligned cameras → joints now in SMPL frame

Step 5: SMPL optimisation (3 stages)
  5a. Stage 1 — Global alignment:
      Optimise global_orient + translation + scale
      Loss: L_joint (triangulated 3D joints from Step 4)
  5b. Stage 2 — Shape refinement:
      Optimise betas + global_orient + translation + scale
      Loss: L_joint + L_reproj + L_shape_reg
  5c. Stage 3 — Full refinement:
      Optimise betas + body_pose + global_orient + translation + scale
      Loss: L_joint + L_reproj + L_pose_prior + L_shape_reg

Step 6: Quality assessment
  6a. Compute MPJPE and PA-MPJPE (refined vs triangulated ground truth)
  6b. Compute per-view reprojection error (project refined joints → compare to 2D kps)
  6c. Compare to Phase 3 consensus metrics

Step 7: Debug output
  - refinement_results.json — SMPL params, cameras, metrics
  - summary.txt — per-stage loss curves, per-view reproj, acceptance criteria
  - triangulated_joints.json — 3D positions + quality scores
  - convergence.png — loss curves per stage
  - reprojection_overlay/ — per-view images with projected joints
```

---

## Self-Calibration Fallback

When `extrinsics_source == "self_calibration"`:
- Step 1 loads Phase 4 `CalibrationResult` instead of COLMAP
- Step 2 (undistortion) is skipped — Phase 4 uses pinhole intrinsics
- Step 4 (frame alignment) is skipped — Phase 4 cameras are already in SMPL frame
- Steps 3, 5, 6, 7 proceed identically
- Only views with `camera.has_extrinsics == True` (17/17 from Phase 4) participate

The fallback is expected to produce worse results (~52px reproj vs sub-pixel for COLMAP)
but demonstrates the self-calibration pipeline works end-to-end. It will be implemented
as a stub that passes tests with relaxed thresholds.

---

## Acceptance Criteria

| # | Criterion | Target (COLMAP) | Target (self-cal) | Verification |
|---|-----------|----------------|-------------------|-------------|
| 5.1 | COLMAP reader parses model 0 | 60 images, 2 cameras | N/A | Unit test |
| 5.2 | Frame alignment reproj error | < 15px mean | N/A | Mean reproj of consensus joints through aligned cameras |
| 5.3 | Triangulation accuracy | PA-MPJPE < 30mm vs consensus | < 50mm | Compare triangulated joints to consensus joints |
| 5.4 | SMPL refinement improves over consensus | PA-MPJPE decreases | PA-MPJPE decreases | Before/after comparison |
| 5.5 | Reprojection error | < 15px mean per view | < 80px | Project refined joints → compare to 2D kps |
| 5.6 | Debug output complete | All files created | All files created | File existence checks |

---

## Debug Output (`output/debug/refinement/`)

- `refinement_results.json` — refined SMPL params, per-view cameras (R, t), loss history, metrics
- `triangulated_joints.json` — 14 joints: position, quality, n_views, reproj_error
- `summary.txt` — per-stage loss, per-view reproj, acceptance pass/fail
- `convergence.png` — matplotlib loss curves (total + per-component) across stages
- `reprojection_overlay/` — per-view images with projected refined SMPL joints overlaid
- `camera_positions.png` — COLMAP-aligned camera positions (top-down + side view)

---

## Files Modified

| File | Change |
|------|--------|
| `scantosmpl/config.py` | Add `Phase5Config` dataclass |
| `scantosmpl/types.py` | No changes needed — existing types sufficient |

## Files Created

| File | Purpose |
|------|---------|
| `scantosmpl/calibration/colmap_reader.py` | Parse COLMAP cameras.bin + images.bin |
| `scantosmpl/calibration/frame_alignment.py` | COLMAP → SMPL frame alignment |
| `scantosmpl/calibration/undistort.py` | Radial undistortion of 2D keypoints |
| `scantosmpl/triangulation/dlt.py` | Weighted DLT triangulation |
| `scantosmpl/triangulation/ransac.py` | RANSAC wrapper for robust triangulation |
| `scantosmpl/fitting/losses.py` | Differentiable loss functions (PyTorch) |
| `scantosmpl/fitting/optimiser.py` | Staged SMPL optimiser |
| `scantosmpl/fitting/pipeline.py` | Phase 5 orchestrator |
| `tests/test_triangulation.py` | Unit tests for DLT, RANSAC, undistortion, alignment |
| `tests/test_fitting.py` | Unit tests for losses and optimiser |
| `tests/integration/test_phase5_integration.py` | Integration tests (GPU) |

## Existing Code Reused

| File | Function/Class | Used For |
|------|---------------|----------|
| `scantosmpl/utils/geometry.py` | `procrustes_align()` | Frame alignment (Step 4) |
| `scantosmpl/utils/geometry.py` | `project_points()` | Reprojection loss + debug overlays |
| `scantosmpl/utils/geometry.py` | `camera_center()` | Camera position plots |
| `scantosmpl/smpl/model.py` | `SMPLModel` | Differentiable SMPL forward pass |
| `scantosmpl/smpl/joint_map.py` | `COCO_TO_SMPL`, `COCO_MIDPOINT_TO_SMPL` | Joint correspondence mapping |
| `scantosmpl/config.py` | `FittingConfig` | Loss weights |
| `scantosmpl/types.py` | `ViewResult`, `CameraParams`, `ConsensusResult` | Data types |
| `scantosmpl/calibration/intrinsics.py` | `build_intrinsic_matrix()` | Building K from COLMAP params |

---

## Implementation Order

1. `scantosmpl/calibration/colmap_reader.py` — COLMAP binary parser
2. `scantosmpl/calibration/undistort.py` — radial undistortion
3. `scantosmpl/calibration/frame_alignment.py` — Procrustes alignment
4. `scantosmpl/triangulation/dlt.py` — weighted DLT
5. `scantosmpl/triangulation/ransac.py` — RANSAC wrapper
6. `scantosmpl/fitting/losses.py` — loss functions
7. `scantosmpl/fitting/optimiser.py` — staged optimiser
8. `scantosmpl/config.py` — add Phase5Config
9. `scantosmpl/fitting/pipeline.py` — orchestrator
10. `tests/test_triangulation.py` — unit tests
11. `tests/test_fitting.py` — unit tests
12. `tests/integration/test_phase5_integration.py` — integration tests

---

## Verification Commands

```bash
# Unit tests (no GPU)
pytest tests/test_triangulation.py tests/test_fitting.py -v

# Integration tests (GPU + COLMAP reconstruction + Phase 3 output required)
pytest tests/integration/test_phase5_integration.py -v -m gpu
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| COLMAP frame very different from SMPL canonical | Low | Procrustes handles arbitrary scale/rotation/translation. Use at least 8 well-distributed joints for robust alignment. |
| ViTPose hallucinates keypoints on rear views | Medium (known from Phase 4) | RANSAC triangulation rejects outlier observations. Confidence weighting reduces influence. 4 known problem views: cam02_4, cam04_4, cam10_4, cam10_5. |
| Consensus mesh too coarse for meaningful refinement | Low | 32mm PA-MPJPE is good enough as initialisation. Triangulation provides independent 3D signal. |
| Overfitting body_pose in Stage 3 | Medium | Pose prior regularisation + conservative learning rate. Monitor per-stage PA-MPJPE. |
| Frame alignment error propagates to triangulation | Low | Validate alignment via reprojection of consensus joints through aligned cameras (criterion 5.2). |
