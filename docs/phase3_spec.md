# Phase 3 Specification: Multi-View Consensus — Tier 1 Complete

## Context

**Input**: 11 HMR-suitable views from Phase 2, each with:
- `betas` (10,) — SMPL shape parameters
- `body_pose` (69,) — 23 joints x 3 axis-angle (view-invariant)
- `global_orient` (3,) — root orientation relative to *that* camera (view-dependent)
- `cam_translation` (3,) — camera-space translation (view-dependent)
- `dense_keypoints_2d` (138, 2) — in original image coords
- `dense_keypoint_confs` (138,) — learned confidence
- `fov_exif` / `fov_flnet` — per-view FoV estimates
- `vertices` (6890, 3) — SMPL mesh in camera space
- `camera.focal_length` — EXIF-derived focal length in pixels

**6 views excluded** from HMR (unsuitable): cam01_6, cam02_4, cam06_4, cam07_6,
cam10_4, cam10_5. These are preserved in ViewResults for Tier 2 PnP.

**Phase 2 stats** (from `output/debug/hmr/summary.txt`):
- beta std max: 0.336 (well under 1.0 threshold)
- body_pose norm: mean 1.830 rad, std 0.191
- FoV EXIF vs FLNet: mean diff 4.17deg, max 8.64deg, all pass
- Orientation quality: all 11 views score 1.00

---

## What We're Building

Fuse 11 per-view SMPL estimates into a single consensus set of body parameters.
This completes Tier 1: a usable SMPL fit with zero calibration.

The consensus produces:
1. **Consensus beta** (10,) — robust aggregation of per-view shape estimates
2. **Consensus body_pose** (69,) — SO(3) Frechet mean per joint
3. **Canonical global_orient** (3,) = `[0, 0, 0]` — identity (see Design Decisions)
4. **Consensus mesh** — SMPL forward pass with consensus params, for visualization
5. **Quality metrics** — cross-view consistency, anatomical plausibility checks

---

## What We're NOT Building

| Excluded | Reason |
|----------|--------|
| Per-view camera extrinsics | Phase 4 (PnP self-calibration) |
| Triangulated 3D keypoints | Phase 5 (multi-view triangulation) |
| FoV consensus | EXIF focal lengths are physical measurements, strictly more reliable than FLNet neural estimates. FLNet cross-check already done in Phase 2. |
| PromptHMR fallback | Deferred |

---

## Design Decisions

### 1. Canonical global_orient = identity `[0, 0, 0]`

Each view's `global_orient` encodes body rotation relative to *that camera*. These
differ wildly across views (confirmed by Phase 2 data: axis-angle vectors span the
full range). Averaging them is meaningless without camera extrinsics.

Setting `global_orient = [0, 0, 0]` means the consensus mesh faces "forward" in SMPL's
canonical frame (facing +Z, Y-up). This is clean and unambiguous. Phase 4 PnP with 138
dense keypoints is the proper solution for recovering per-view orientation.

Per-view `global_orient` values are preserved in ViewResults for Phase 4.

### 2. SO(3) Frechet mean for body_pose (per joint)

`body_pose` is view-invariant — it encodes 23 joint rotations relative to parent joints.
For a T-pose subject, these should be identical across views (only camera position changes).

The SO(3) Frechet mean (also called the Karcher/geometric mean) is the mathematically
correct way to average rotations. Algorithm:

```
Input: R_1, ..., R_N rotation matrices, weights w_1, ..., w_N
Initialize: R_mean = R_1
Repeat until convergence:
    tangent_sum = sum(w_i * Log(R_mean^T @ R_i))  # Log = matrix logarithm -> so(3)
    delta = tangent_sum / sum(w_i)
    R_mean = R_mean @ Exp(delta)                    # Exp = matrix exponential -> SO(3)
    if ||delta|| < epsilon: break
```

For small rotations (T-pose), this converges in 2-3 iterations and produces
results nearly identical to component-wise averaging. For larger rotations (A-pose
arms), it correctly handles the non-Euclidean structure of rotation space.

### 3. Confidence-weighted robust beta aggregation

Per-view betas have low variance (std max 0.336) — the views are consistent.
Still, we use a robust approach:

1. Compute per-view confidence weight from:
   - Orientation quality score (all 1.0 currently, but future-proofs)
   - Dense keypoint confidence (mean of 138 per-keypoint scores)
2. For each beta component: confidence-weighted trimmed mean (trim 10%)
3. Validate: re-run SMPL forward pass, check height in 1.5-2.0m range

Trimmed mean (vs median) retains more information from well-behaved views
while still rejecting outliers.

### 4. Cross-view consistency via PA-MPJPE

PA-MPJPE (Procrustes-Aligned Mean Per-Joint Position Error) measures how well
the consensus SMPL joints align with each per-view prediction after optimal
rigid alignment. This removes the global_orient / camera differences.

For each view i:
1. Compute SMPL joints from view i's params: J_i (24 joints)
2. Compute SMPL joints from consensus params: J_consensus (24 joints)
3. Procrustes-align J_consensus to J_i (scale + rotation + translation)
4. PA-MPJPE_i = mean ||J_aligned - J_i||

Criterion 3.6: mean PA-MPJPE across views < 50mm.

---

## Deliverables

### 1. `scantosmpl/utils/geometry.py` — SO(3) operations

```python
def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 3) axis-angle via scipy."""

def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 3, 3) rotation matrices via scipy."""

def so3_log(R: np.ndarray) -> np.ndarray:
    """(3, 3) -> (3,) logarithmic map SO(3) -> so(3)."""

def so3_exp(v: np.ndarray) -> np.ndarray:
    """(3,) -> (3, 3) exponential map so(3) -> SO(3)."""

def frechet_mean_so3(
    rotations: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Weighted Frechet mean on SO(3).
    rotations: (N, 3, 3), weights: (N,) non-negative.
    Returns: (3, 3) mean rotation matrix.
    """

def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Procrustes alignment: find optimal scale + R + t to align source to target.
    source, target: (J, 3) joint positions.
    Returns: (aligned_source, scale).
    """
```

### 2. `scantosmpl/hmr/consensus.py` — Multi-view consensus

```python
@dataclass
class ConsensusResult:
    """Output from multi-view SMPL parameter consensus."""

    betas: np.ndarray              # (10,) consensus shape
    body_pose: np.ndarray          # (69,) consensus pose (axis-angle)
    global_orient: np.ndarray      # (3,) canonical = [0, 0, 0]

    # Mesh from consensus params
    vertices: np.ndarray           # (6890, 3) in canonical frame
    faces: np.ndarray              # (13776, 3)

    # Quality metrics
    pa_mpjpe_per_view: dict[str, float]   # view_name -> PA-MPJPE in mm
    pa_mpjpe_mean: float                   # mean across views (criterion 3.6: < 50mm)
    beta_std: np.ndarray                   # (10,) per-component std before aggregation
    body_height_m: float                   # estimated height from consensus mesh
    per_view_weights: dict[str, float]     # view_name -> confidence weight used
    n_views_used: int                      # views after trimming


class ConsensusBuilder:
    """Fuses per-view HMR estimates into a single SMPL parameter set."""

    def __init__(
        self,
        smpl_model_path: str | Path,
        gender: str = "neutral",
        device: str = "cuda",
        trim_fraction: float = 0.1,    # trim 10% for robust mean
    )

    def build_consensus(
        self,
        views: list[ViewResult],
    ) -> ConsensusResult:
        """
        Build consensus from HMR-processed views.

        Only uses views where betas is not None and hmr_suitable is True.
        """

    def _aggregate_betas(
        self,
        betas_list: list[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """Confidence-weighted trimmed mean per component. Returns (10,)."""

    def _aggregate_body_pose(
        self,
        body_pose_list: list[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """SO(3) Frechet mean per joint. Returns (69,) axis-angle."""

    def _compute_view_weights(
        self,
        views: list[ViewResult],
    ) -> np.ndarray:
        """
        Confidence weight per view, based on:
        - Dense keypoint mean confidence
        - (future: orientation quality, beta deviation from median)
        Returns (N,) normalized weights summing to 1.
        """

    def _compute_body_height(
        self,
        vertices: np.ndarray,
    ) -> float:
        """Estimate body height in metres from vertex positions."""

    def _compute_pa_mpjpe(
        self,
        consensus_joints: np.ndarray,
        per_view_joints: list[np.ndarray],
        view_names: list[str],
    ) -> dict[str, float]:
        """PA-MPJPE for each view vs consensus. Returns {name: mm}."""
```

### 3. `scantosmpl/smpl/joint_map.py` — Joint mapping constants

```python
# COCO-17 -> SMPL-24 joint mapping (for PnP + triangulation in Tier 2)
COCO_TO_SMPL: dict[int, int] = {
    5: 16, 6: 17,    # shoulders
    7: 18, 8: 19,    # elbows
    9: 20, 10: 21,   # wrists
    11: 1, 12: 2,    # hips
    13: 4, 14: 5,    # knees
    15: 7, 16: 8,    # ankles
}

# Derived keypoints (computed as midpoints)
# pelvis  = mid(L_hip, R_hip)   -> SMPL 0
# neck    = mid(L_shoulder, R_shoulder) -> SMPL 12

# CameraHMR 138 dense keypoints map directly to SMPL vertex indices
# (COMA-sampled). The vertex indices are fixed in the DenseKP model.
DENSE_KP_VERTEX_INDICES: np.ndarray  # (138,) loaded from checkpoint or hardcoded
```

### 4. `tests/test_consensus.py` — Unit tests (no GPU required)

```python
class TestSO3FrechetMean:
    def test_identity_inputs_give_identity()
    def test_single_rotation_returns_itself()
    def test_two_rotations_gives_geodesic_midpoint()
    def test_uniform_weights_match_unweighted()
    def test_convergence_with_large_rotations()
    def test_output_is_valid_rotation_matrix()  # det=1, orthogonal

class TestProcrustesAlign:
    def test_identity_alignment()
    def test_scaled_input_recovers_scale()
    def test_rotated_input_recovers_rotation()
    def test_translated_input_recovers_translation()

class TestBetaAggregation:
    def test_identical_inputs_return_same()
    def test_outlier_trimmed()
    def test_weighted_shifts_toward_high_weight()
    def test_output_shape_10()

class TestBodyPoseAggregation:
    def test_identical_t_poses_return_same()
    def test_near_zero_rotations_near_zero_output()
    def test_output_shape_69()

class TestConsensusResult:
    def test_pa_mpjpe_zero_for_identical_views()
    def test_height_plausible_range()

class TestRotationConversions:
    def test_aa_to_rotmat_identity()
    def test_aa_to_rotmat_roundtrip()
    def test_rotmat_to_aa_batch()
    def test_so3_log_exp_roundtrip()
```

### 5. `tests/integration/test_consensus_integration.py` — Integration tests (GPU + Phase 2 output)

```python
@pytest.mark.gpu
class TestConsensusOnScannerData:
    def test_beta_anatomically_plausible()       # 3.1: height 1.5-2.0m
    def test_body_pose_matches_t_pose()          # 3.2: arms roughly horizontal
    def test_consensus_improves_over_best_view() # 3.3: lower variance
    def test_so3_valid_rotation_matrices()        # 3.4: det=1, orthogonal
    def test_pa_mpjpe_under_50mm()               # 3.6: cross-view consistency
    def test_consensus_mesh_vertex_count()       # 6890 vertices, 13776 faces
```

### 6. Config update: `scantosmpl/config.py`

```python
@dataclass
class ConsensusConfig:
    trim_fraction: float = 0.1           # fraction to trim from each tail
    frechet_max_iter: int = 50           # SO(3) mean max iterations
    frechet_tol: float = 1e-7            # convergence tolerance
    min_views: int = 3                   # minimum views for consensus
    save_debug: bool = True
    debug_dir: Path = Path("output/debug/consensus")
```

Add `consensus: ConsensusConfig = field(default_factory=ConsensusConfig)` to `PipelineConfig`.

---

## Debug Output

Saved to `output/debug/consensus/`:

- `consensus_results.json` — consensus betas, body_pose, height, PA-MPJPE per view
- `consensus_mesh.obj` — the consensus SMPL mesh in canonical frame
- `summary.txt`:
  - Per-view weights used
  - Beta: consensus values, per-component std, trimmed views
  - Body pose: per-joint rotation magnitude, max deviation from consensus
  - PA-MPJPE table: per-view + mean (criterion 3.6)
  - Height estimate (criterion 3.1)
  - T-pose arm angle check (criterion 3.2)

---

## Processing Pipeline

```
Phase 2 ViewResults (11 HMR-suitable views)
    │
    ├── Step 1: Compute per-view confidence weights
    │     Dense keypoint mean confidence per view
    │     Normalize to sum to 1
    │
    ├── Step 2: Aggregate betas
    │     Confidence-weighted trimmed mean (trim 10%)
    │     → consensus betas (10,)
    │
    ├── Step 3: Aggregate body_pose
    │     For each of 23 joints:
    │       Convert per-view axis-angle → rotation matrices
    │       SO(3) Frechet mean with confidence weights
    │       Convert back to axis-angle
    │     → consensus body_pose (69,)
    │
    ├── Step 4: Set global_orient = [0, 0, 0]
    │
    ├── Step 5: SMPL forward pass
    │     Run smplx with consensus (betas, body_pose, global_orient)
    │     → vertices (6890, 3), joints (24, 3)
    │     Compute body height from vertex extent
    │
    ├── Step 6: Cross-view PA-MPJPE
    │     For each view:
    │       Run SMPL with that view's (betas, body_pose)
    │       and global_orient=[0,0,0] (canonical frame)
    │       Procrustes-align consensus joints to per-view joints
    │       Compute MPJPE
    │     Mean PA-MPJPE → criterion 3.6
    │
    └── Step 7: Debug output
          JSON, mesh .obj, summary text
```

**Note on Step 6**: When computing per-view SMPL joints for PA-MPJPE, we use
`global_orient=[0,0,0]` for both consensus and per-view. This isolates the
comparison to betas + body_pose (the view-invariant parameters). The per-view
global_orient encodes camera viewpoint, not body pose quality.

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 3.1 | Consensus beta anatomically plausible (height 1.5-2.0m) | SMPL forward pass + vertex height |
| 3.2 | Consensus body_pose matches T-pose | Arm angle check: shoulder-elbow-wrist roughly collinear and horizontal |
| 3.3 | Consensus improves over single-best view | PA-MPJPE(consensus) < min(PA-MPJPE(single view)) across leave-one-out |
| 3.4 | SO(3) averaging produces valid rotation matrices | det=1, R^T R = I for all 23 joint means |
| 3.5 | FoV consensus: EXIF used directly | No FoV aggregation (EXIF is authoritative). FLNet cross-check from Phase 2 already validates. |
| 3.6 | Cross-view PA-MPJPE < 50mm | Mean PA-MPJPE across 11 views |

---

## Implementation Order

1. `scantosmpl/utils/geometry.py` — SO(3) operations, Procrustes alignment
2. `scantosmpl/smpl/joint_map.py` — COCO-SMPL mapping constants
3. `scantosmpl/hmr/consensus.py` — ConsensusBuilder + ConsensusResult
4. Update `scantosmpl/config.py` — add ConsensusConfig
5. `tests/test_consensus.py` — unit tests
6. `tests/integration/test_consensus_integration.py` — integration tests

---

## Verification Commands

```bash
# Unit tests (no GPU)
pytest tests/test_consensus.py -v

# Integration tests (GPU + Phase 2 output required)
pytest tests/integration/test_consensus_integration.py -v -m gpu
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SO(3) Frechet mean doesn't converge | Low (T-pose rotations are small) | Cap at 50 iterations; fallback to component-wise median |
| PA-MPJPE > 50mm | Low (beta std is 0.336, views are consistent) | Check per-view outliers; increase trim fraction |
| Height outside 1.5-2.0m | Medium (depends on beta accuracy) | Relax range slightly if subject is very short/tall |
| Too few views after trimming | Low (11 views, 10% trim = remove ~1) | min_views=3 guard |
