# Phase 4 Specification: PnP Self-Calibration — Camera Extrinsic Recovery

## Context

**Input**: Consensus SMPL mesh from Phase 3 + per-view data from Phases 1-2.

| Category | Count | Views | Data available |
|----------|-------|-------|----------------|
| HMR-suitable | 10 | cam01_2, cam02_5, cam03_5, cam03_6, cam04_4, cam04_5, cam05_4, cam05_5, cam07_4, cam10_2 | 138 dense 2D kps + confs, SMPL params, global_orient, cam_translation, ViTPose 17 kps, EXIF focal length |
| HMR-excluded | 6–7 | cam01_6, cam02_4, cam05_6, cam06_4, cam07_6, cam10_4, cam10_5 | ViTPose 17 COCO kps + confs, EXIF focal length, bbox |

**Consensus mesh** (Phase 3): 6890 vertices, 13776 faces, 24 joints,
canonical frame (`global_orient=[0,0,0]`), body height ~1.71m,
mean PA-MPJPE ~32mm across views.

---

## What We're Building

For each of the 17 views, recover the camera extrinsics [R|t] — where the
camera was in 3D space relative to the SMPL mesh. The consensus mesh acts as
a known calibration target: we know where 138 surface points are in 3D
(SMPL vertex indices), and we know where they appear in each image (DenseKP
2D predictions).

This produces:
1. **Per-view rotation R** (3×3) — camera orientation in world frame
2. **Per-view translation t** (3,) — camera position offset
3. **Camera center C** = −R^T @ t — world-space camera position
4. **Reprojection error** — how well the solution explains the observations
5. **Camera geometry validation** — do the cameras form a plausible scanner layout?

---

## What We're NOT Building

| Excluded | Reason |
|----------|--------|
| Triangulation | Phase 5 |
| SMPL parameter refinement | Phase 5 |
| Bundle adjustment | Phase 5 (optional) |
| FoV consensus | EXIF is authoritative (decided in Phase 3) |

---

## Design Decisions

### 1. Hardcoded 138 vertex indices (no runtime pickle dependency)

The `downsample_mat.pkl` from CameraHMR is a one-hot (138, 6890) sparse matrix.
Each of the 138 dense keypoints maps to exactly one SMPL vertex. Verified:

```python
# All 138 rows have exactly 1 non-zero entry (value=1.0)
# Extracted indices from models/mapping/downsample_mat.pkl:
DENSE_KP_VERTEX_INDICES = np.array([
    102, 189, 254, 366, 425, 433, 444, 542, 599, 639, ...  # 138 total
])
```

At PnP time: `pts_3d = consensus_vertices[DENSE_KP_VERTEX_INDICES]` → (138, 3).
No file loading, no pickle, no external dependency.

### 2. EXIF focal length as primary intrinsics

EXIF focal lengths are physical measurements from the Canon EOS 2000D. Two lens
settings observed (FoV 34.97° and 50.59°). FLNet cross-check from Phase 2
confirms accuracy (mean diff 4.17°). `CameraParams.K` property already computes
the intrinsic matrix — we just need to ensure `principal_point` is set to image
center for views that currently have `(0, 0)`.

### 3. Dense PnP for HMR views, sparse fallback for excluded views

**Dense (10 views)**: 138 3D-2D correspondences via SMPL vertex indices ↔ DenseKP
2D predictions. Filtered by confidence > 0.3. With ~100+ inliers, RANSAC has
massive redundancy — a few noisy keypoints don't affect the solution.

**Sparse (6-7 views)**: 12 COCO-to-SMPL joint correspondences + 2 midpoint-derived
joints (pelvis, neck) = up to 14 points. Uses ViTPose keypoints filtered by
confidence > 0.3. More fragile, but solved independently — a bad solution doesn't
contaminate other views. Quality gates: minimum 6 inliers, reprojection < 15px.

### 4. PnP method selection

- `cv2.SOLVEPNP_ITERATIVE` (Levenberg-Marquardt) for ≥ 10 correspondences
- `cv2.SOLVEPNP_EPNP` for 4-9 correspondences (more robust with few points)
- `cv2.solvePnPRefineLM` applied to inliers after RANSAC for sub-pixel refinement

### 5. Camera geometry validation

The scanner rig has cameras arranged roughly in an arc/circle around the subject.
Validation criteria:
- **Radial distance**: cameras should be at similar distances from origin.
  Coefficient of variation (std/mean) < 0.3.
- **Angular coverage**: cameras projected to XZ plane should span ≥ 120°.
- **Height clustering**: cameras at similar scanner tiers should have similar Y.

These are logged as stats; a boolean `geometry_plausible` flag summarises.

---

## Deliverables

### 1. `scantosmpl/smpl/joint_map.py` — Update

Add the 138 hardcoded SMPL vertex indices:

```python
DENSE_KP_VERTEX_INDICES = np.array([
    102, 189, 254, ..., 6789, 6824   # 138 indices, verified from downsample_mat.pkl
])
```

### 2. `scantosmpl/utils/geometry.py` — Add projection utilities

```python
def project_points(
    pts_3d: np.ndarray,    # (N, 3)
    R: np.ndarray,         # (3, 3)
    t: np.ndarray,         # (3,)
    K: np.ndarray,         # (3, 3)
) -> np.ndarray:
    """Project 3D world points to 2D image coordinates. Returns (N, 2)."""

def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera position in world coordinates: C = -R^T @ t. Returns (3,)."""
```

### 3. `scantosmpl/calibration/intrinsics.py` — K construction

```python
def build_intrinsic_matrix(
    focal_length_px: float,
    image_width: int,
    image_height: int,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Build 3×3 intrinsic matrix. Defaults principal point to image center."""

def get_intrinsics_for_view(
    view: ViewResult,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Build K for a view from its CameraParams. Returns (3, 3)."""
```

### 4. `scantosmpl/calibration/correspondence.py` — 3D-2D correspondence builder

```python
class CorrespondenceBuilder:
    def __init__(
        self,
        consensus_vertices: np.ndarray,   # (6890, 3) canonical frame
        consensus_joints: np.ndarray,      # (24, 3) canonical frame
    )

    def build_dense_correspondences(
        self, view: ViewResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pts_3d, pts_2d, confs) for dense PnP. Returns (138, 3), (138, 2), (138,)."""

    def build_sparse_correspondences(
        self, view: ViewResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pts_3d, pts_2d, confs) for sparse PnP. Returns (N, 3), (N, 2), (N,)."""
```

### 5. `scantosmpl/calibration/pnp_solver.py` — PnP solver

```python
@dataclass
class PnPResult:
    success: bool
    rotation: np.ndarray | None        # (3, 3)
    translation: np.ndarray | None     # (3,)
    rvec: np.ndarray | None            # (3,) Rodrigues
    tvec: np.ndarray | None            # (3,)
    inliers: np.ndarray | None         # inlier indices
    n_correspondences: int
    n_inliers: int
    reprojection_error: float          # mean on inliers (px)
    correspondence_type: str           # "dense_138" or "sparse_coco"
    camera_center: np.ndarray | None   # (3,) = -R^T @ t

class PnPSolver:
    def __init__(self, config: CalibrationConfig)

    def solve(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        confidences: np.ndarray,
        K: np.ndarray,
        conf_threshold: float = 0.3,
        correspondence_type: str = "dense_138",
    ) -> PnPResult
```

### 6. `scantosmpl/calibration/pipeline.py` — Orchestrator

```python
@dataclass
class CalibrationResult:
    pnp_results: dict[str, PnPResult]
    n_views_solved: int
    n_views_dense: int
    n_views_sparse: int
    n_views_failed: int
    camera_centers: dict[str, np.ndarray]
    mean_reprojection_error: float
    geometry_plausible: bool
    geometry_stats: dict[str, float]

class CalibrationPipeline:
    def calibrate(
        self,
        views: list[ViewResult],
        consensus: ConsensusResult,
        image_dir: Path,
        debug_dir: Path | None = None,
    ) -> CalibrationResult
```

### 7. `scantosmpl/config.py` — Extend CalibrationConfig

```python
@dataclass
class CalibrationConfig:
    pnp_method: str = "SOLVEPNP_ITERATIVE"
    ransac_threshold: float = 8.0
    ransac_iterations: int = 5000
    min_inliers: int = 20              # dense views
    min_inliers_sparse: int = 6        # sparse views
    use_dense_keypoints: bool = True
    dense_conf_threshold: float = 0.3
    sparse_conf_threshold: float = 0.3
    refine_lm: bool = True
    save_debug: bool = True
    debug_dir: Path = Path("output/debug/calibration")
```

### 8. `tests/test_pnp.py` — Unit tests (no GPU)

```python
class TestBuildIntrinsicMatrix:
    def test_basic_construction()
    def test_default_principal_point_is_center()

class TestPnPSolver:
    def test_known_transform_recovery()        # synthetic cube
    def test_noisy_2d_still_converges()         # add pixel noise
    def test_insufficient_points_fails()
    def test_reprojection_error_computation()

class TestCorrespondenceBuilder:
    def test_dense_shape_138()
    def test_sparse_includes_midpoints()
    def test_confidence_filtering()

class TestCameraGeometry:
    def test_circular_layout_passes()
    def test_random_layout_fails()
```

### 9. `tests/integration/test_pnp_integration.py` — Integration tests

```python
@pytest.mark.gpu
class TestCalibrationOnScannerData:
    def test_dense_pnp_90pct_success()         # 4.1
    def test_sparse_pnp_50pct_success()        # 4.2
    def test_camera_geometry_plausible()        # 4.3
    def test_reprojection_under_15px()          # 4.4
    def test_robust_to_focal_perturbation()     # 4.5
    def test_dense_outperforms_sparse()         # 4.6
    def test_extrinsics_stored_in_views()
    def test_debug_output_created()
```

---

## Processing Pipeline

```
Step 0: Load consensus (vertices, joints) + per-view data

Step 1: Pre-compute 3D reference points
  dense_3d = consensus_vertices[DENSE_KP_VERTEX_INDICES]  → (138, 3)
  sparse_3d = consensus_joints at COCO_TO_SMPL indices     → (12-14, 3)

Step 2: Per-view PnP
  For each view:
    2a. Build K (EXIF focal + image-center pp)
    2b. Build correspondences (dense or sparse)
    2c. Filter by confidence threshold
    2d. solvePnPRansac (8px threshold, 5000 iterations)
    2e. solvePnPRefineLM on inliers
    2f. Compute reprojection error
    2g. Quality gate: reject if inliers < min or reproj > 15px
    2h. Store R, t in view.camera

Step 3: A/B comparison (criterion 4.6)
  Run sparse PnP on dense views, compare reproj errors

Step 4: Camera geometry validation
  C = -R^T @ t for each solved view
  Check radial consistency, angular coverage

Step 5: Debug output
```

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 4.1 | PnP succeeds on ≥90% of dense views (≥9/10) | Count |
| 4.2 | PnP succeeds on ≥50% of sparse views (≥3/6) | Count |
| 4.3 | Camera geometry roughly circular | Radial CoV < 0.3, angular coverage > 120° |
| 4.4 | Reprojection error < 15px per view | Mean reproj on inliers |
| 4.5 | Robust to ±10% focal length perturbation | Re-run with perturbed K |
| 4.6 | Dense 138 outperforms sparse 12 | A/B reproj comparison |

---

## Implementation Order

1. `scantosmpl/smpl/joint_map.py` — add DENSE_KP_VERTEX_INDICES
2. `scantosmpl/utils/geometry.py` — add project_points, camera_center
3. `scantosmpl/calibration/intrinsics.py`
4. `scantosmpl/calibration/correspondence.py`
5. `scantosmpl/calibration/pnp_solver.py`
6. `scantosmpl/config.py` — extend CalibrationConfig
7. `scantosmpl/calibration/pipeline.py` — orchestrator + debug
8. `scantosmpl/calibration/__init__.py` — exports
9. `tests/test_pnp.py`
10. `tests/integration/test_pnp_integration.py`

---

## Verification Commands

```bash
# Unit tests (no GPU)
pytest tests/test_pnp.py -v

# Integration tests (GPU + Phase 3 output required)
pytest tests/integration/test_pnp_integration.py -v -m gpu
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Consensus mesh error (~32mm) causes PnP drift | Medium | 138 overdetermined correspondences absorb per-vertex noise via RANSAC. Monitor reproj error. |
| Sparse PnP fails on rear views | High (expected) | ViTPose may hallucinate on rear views. Accept 50% success rate. Quality gates reject bad solutions. |
| Camera geometry validation too strict | Low | Parameterize thresholds. Log stats for manual review. |
| Two different focal lengths in rig | Low | Already known from Phase 2 (34.97° and 50.59° FoV). K built per-view from EXIF. |
