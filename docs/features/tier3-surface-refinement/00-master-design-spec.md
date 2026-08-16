# Master Design Spec — Tier 3 Surface Refinement (Point Cloud → SMPL+D)

**Status**: Draft
**Slug**: `tier3-surface-refinement`
**Owner**: Dan
**Date**: 2026-08-14
**Tiers touched**: Tier 3 (new). Reads Tier 2 output read-only. Produces the ground-truth source for Tier 4 / PSD.

> **Source of truth**: `REVIEW.md` Phase 6 + Phase 7 (including the 7.M metric definition and the
> 7.B downstream boundary table). Architecture context: `CLAUDE.md` Tier 3, `Research.md` §"Tier 3 —
> Point cloud alignment via chamfer distance". Design-boundary style borrowed from
> `docs/phase5_tier2_improvement_plan.md`: fix the metric before tuning anything, prefer graded
> weighting over hard exclusion, and refuse to patch a defect in the tier that doesn't own it.

---

## 1. Problem

Tier 2 constrains **joint centres only**. A joint centre sits on the body's medial axis and the
surface is offset from it by the soft-tissue radius, which no joint reprojection term can observe
from any view — so torso girth is unconstrained and drifts run-to-run (documented in
`notes.md` and deliberately deferred by the Phase 5 plan's "Deferred — Torso girth" section).
`scantosmpl/pointcloud/` is an empty package, so the photogrammetry point cloud — the only
independent geometry the project has — is currently unused.

Tier 3 closes that gap: align the cloud to the Tier 2 SMPL mesh, refine `β`/`θ` against surface
geometry, and solve a per-vertex displacement field `D`. `D` is also the learning signal for
Tier 4 / PSD, so this feature's artefact contract is load-bearing for the project's headline
deliverable, not just for a chamfer number.

## 2. Decisions

Locked choices from the clarifying-question rounds, with rationale.

- **D1 (Scope = Phase 6 + Phase 7 as one feature).** Tier 3 is not independently shippable without
  a point-cloud input path. `scantosmpl/pointcloud/` is built out here (io, preprocess, align,
  segment) alongside `fitting/surface.py`. REVIEW.md's phase split is a *planning* boundary, not a
  delivery boundary — the Tier 3 gate needs both halves to be measurable at all.

- **D2 (No Kaolin — and no torch downgrade).** REVIEW.md 7.M1 names
  `kaolin.metrics.trianglemesh.point_to_mesh_distance`, and CLAUDE.md's stack table specifies
  Kaolin. **Measured constraint:** NVIDIA's wheel index tops out at `torch-2.8.0_cu128`
  (`torch-2.9.0_cu128.html` and `torch-2.11.0_cu130.html` both return HTTP 404); this project runs
  `torch 2.11.0+cu130`, and Kaolin wheels are compiled against a specific torch C++ ABI, so the
  2.8.0 wheel would fail at import with an undefined-symbol error rather than degrade. The
  resolution is 7.M6's own split — *the loss need not equal the metric* — and neither half needs a
  compiled extension:

  | Need | Differentiable? | Implementation | Already a dependency? |
  |---|---|---|---|
  | **Metric**, cloud → mesh surface | No — it is reporting | `open3d.t.geometry.RaycastingScene.compute_distance` | yes (`open3d>=0.17`, 0.19.0 installed) |
  | **Loss**, vertex ↔ cloud | Yes | chunked `torch.cdist` | yes (torch) |

  Verified in this env: the Open3D raycasting path returns `0.005` for a point 5 mm above a
  triangle (exact point-to-triangle, no tessellation floor), and chunked `torch.cdist` at
  6890 × 50 000 runs fwd+bwd in **74 ms/iter at 1.9 GiB peak** — 550 iterations ≈ 41 s, inside
  AC 7.7's 60 s budget with no subsampling. **`trimesh.proximity.closest_point` is not a viable
  second implementation here — it requires `rtree`, which is not installed**; the cross-check is an
  analytic point-to-triangle fixture instead (stronger, and adds no dependency).
  **This supersedes 7.M1's named function while satisfying its substance** (cloud→mesh MUST be
  point-to-surface). No new dependency is added by this feature, and `pyproject.toml`'s
  `[project.optional-dependencies] kaolin` entry is left untouched and unused.

- **D3 (Loss is bidirectional, from one distance matrix).** The answer to the backend question
  specified vertex→cloud for the loss. The spec implements **both** directions, because
  `torch.cdist(verts, cloud_chunk)` yields `min(dim=1)` (mesh→cloud) and `min(dim=0)` (cloud→mesh)
  from the *same* matrix at zero extra cost, and a one-sided loss is the classic shrink-wrap
  failure — mesh vertices collapse into the densest region of the cloud while uncovered regions
  drift unpenalised (7.M3 names exactly this hazard for the metric; it applies at least as hard to
  the loss). Both directions are vertex-based **in the loss**; 7.M6's escape hatch (switch the
  cloud→mesh side of the *loss* to point-to-triangle if faceting or shrink-wrapping appears)
  remains available and is recorded as R3.

- **D4 (`D` frame = posed / world).** `D` is persisted in the same posed world frame the SMPL
  forward pass returns, defined exactly as
  `D := V_final − SMPLModel.forward(β, θ, t, s, displacements=None).vertices`.
  This feeds PSD spec §4.2's `δ_world = scan − SMPL(β,θ)` directly, leaving PSD to apply
  `R_v(θ)⁻¹` itself as its own spec says. The alternative (rest frame, pre-LBS) would force PSD to
  forward-skin `D` first — adding precisely the frame step PSD R5 flags as a silent-failure zone.
  `β, θ, t, s` are persisted alongside `D` so PSD can regenerate the baseline bit-identically.

- **D5 (Three stages, `D` last).** S1 aligns the cloud to the mesh (ICP only, SMPL untouched).
  S2 fits SMPL params to the surface with `D ≡ 0`. S3 freezes SMPL params and solves `D` alone.
  `D` therefore only ever absorbs genuine off-manifold geometry, never pose/shape/global-transform
  error — which is what 7.B5 requires. Mirrors the staged schedule already proven in
  `scantosmpl/fitting/optimiser.py::DEFAULT_STAGES`.

- **D6 (Global scale is owned by the ICP alignment and frozen in S2).** CLAUDE.md's decision log
  says "Align PC to SMPL, not reverse — SMPL has correct metric scale." If S2 also optimised
  `scale`, the mesh's metric scale would drift to meet a cloud whose scale was itself just solved,
  making the two redundant and the metric-scale premise false. `scale` is solved once, in S1, as
  part of the cloud→SMPL similarity, and is **not** in any S2/S3 parameter list.

- **D7 (Semantic weighting comes from `lbs_weights`, not cloud segmentation).** REVIEW.md Phase 6
  lists `segment.py` as "height slices + PCA/connectivity". AC 7.3 needs per-**mesh-vertex** part
  weights, and `body_model.lbs_weights` (6890, 24) already encodes exactly that association —
  `argmax(axis=1)` → joint → part group is exact, deterministic, and needs no heuristic. Cloud
  points inherit a label from their nearest mesh vertex. `segment.py` is still delivered, but as
  this (correct, testable) mechanism rather than height slices. `FittingConfig.body_part_weights`
  already declares the six group names this produces and is finally wired up.

- **D8 (Downsampling is unit-free).** A metric voxel size is meaningless before alignment — a
  Meshroom cloud's units are arbitrary. Preprocessing derives its voxel size from a fraction of the
  cloud's own bbox diagonal and targets a point count, so the same config works on a cloud scaled
  by 10⁻³ or 10³.

- **D9 (PCA init is disambiguated by exhaustive enumeration, not heuristics).** PCA axes of a body
  are sign- and (for near-degenerate eigenvalues) order-ambiguous. Rather than guessing an up-axis,
  S1 enumerates **all 24 proper rotations** mapping the cloud's PCA triad onto the mesh's, runs ICP
  from each, and keeps the lowest inlier RMSE. Deterministic, no RNG, and ~24 cheap ICP runs is
  irrelevant given speed is explicitly not a concern here.

- **D10 (Both β modes, recorded in the manifest).** REVIEW.md's own "Conflict to resolve" note
  (AC 7.4 vs 7.B1) is adopted verbatim: β-refinement is a **single-pose** capability used once on
  the reference pose; every subsequent pose runs with that β frozen. `--lock-betas` +
  `--betas-from` select the mode and the manifest records which was used.

- **D11 (Synthetic known-answer fixture gates the loop; real cloud gates the tier).** No point
  cloud exists in the repo. A seeded synthetic cloud is generated from the Tier 2 mesh with a
  *known* similarity transform, noise level and injected "clothing" offset, so alignment recovery
  and `D` recovery are assertable exactly. AC 7.1's 8 mm-on-real-scanner-data is a **deferred
  gate** — its test skips with an explicit marker until `data/t-pose/pointcloud.ply` exists, and
  the loop must not report the Tier 3 gate as passed on synthetic data alone.

- **D12 (Determinism).** Tier 3 introduces **no stochastic step** in the production path: PCA,
  the 24-candidate enumeration, Open3D point-to-plane ICP, `torch.cdist` and Adam are all
  deterministic given fixed inputs. RANSAC-FPFH global registration is explicitly **not** used.
  The only RNG is in the fixture generator and the tessellation-floor sampler, both seeded.

## 3. Scope

**In scope**
- `scantosmpl/pointcloud/`: `io.py` (PLY/OBJ load), `preprocess.py` (outlier removal, unit-free
  downsample, normal estimation), `align.py` (PCA init + 24-candidate ICP with scale),
  `segment.py` (`lbs_weights`-derived part labels + nearest-vertex transfer to cloud).
- `scantosmpl/evaluation/surface_metrics.py`: the binding 7.M metric — point-to-surface cloud→mesh,
  vertex-to-point mesh→cloud, both reported separately, plus the tessellation floor.
- `scantosmpl/fitting/surface_losses.py`: chunked bidirectional chamfer, normal consistency,
  Laplacian smoothing, displacement regularisation.
- SMPL+D support in `scantosmpl/smpl/model.py` (a `displacements` parameter + `forward` kwarg).
- `scantosmpl/fitting/surface.py`: the staged S2/S3 optimiser.
- `scantosmpl/fitting/surface_pipeline.py` + `artefacts.py`: orchestration and the PSD-boundary
  artefact/manifest writer satisfying 7.B1–7.B8.
- `Tier3Config` in `config.py`, new dataclasses in `types.py`, a `fit-surface` CLI command.
- Synthetic fixture generator + integration test; unit tests per module.

**Out of scope**
- Tier 4 / PSD itself — this feature *satisfies* its constraints, it does not implement its
  consumer (explicit instruction; PSD spec §11 also declares Tier 3 out of its own scope).
- Re-tuning Tier 2 (`DEFAULT_STAGES`, view weighting, head anchor) — Tier 3 reads Tier 2's output
  read-only. The torso-girth defect is *fixed here*, but by adding a surface term, not by touching
  Tier 2.
- Phase 8's `fit-combined` end-to-end orchestrator — `fit-surface` is a Tier-3-only entry point so
  this feature is testable; wiring the full three-tier CLI stays Phase 8's job.
- Multi-pose corpus assembly — `build_corpus()` is PSD's, per its §5.2. This feature only
  guarantees the per-pose layout and manifest it will read.
- Adding `rtree`, Kaolin, or any other new dependency (D2).
- Meshroom itself, and any use of Meshroom's sfm camera poses for alignment (the
  `selfcal-default-extrinsics` spec §9 records that ICP already owns this problem).

## 4. Approach

A new Tier 3 stage slots in below Tier 2 in `CLAUDE.md`'s architecture diagram, consuming the
Phase 5 `RefinementResult` plus a PLY/OBJ cloud and producing SMPL+D.

```
Tier 2 RefinementResult (β, θ, t, s, verts)        raw cloud (Meshroom frame, arbitrary units)
            │                                                     │
            └──────────────────────┬──────────────────────────────┘
                                   ▼
   S1  preprocess → PCA triads → enumerate 24 proper rotations → ICP each → keep best
       ⇒ CloudAlignment(scale, R, t)  ·  cloud now in SMPL/world, metres  ·  SMPL untouched
                                   ▼
   S2  optimise (β?, θ, global_orient, translation) vs bidirectional chamfer + priors
       ⇒ D ≡ 0 throughout.  scale frozen (D6).  β frozen iff --lock-betas (D10).
                                   ▼
   S3  freeze all SMPL params, optimise D alone
       vs chamfer + Laplacian + normal consistency + ‖D‖ regularisation
                                   ▼
   report  ChamferReport (both directions, separately) + tessellation floor
   persist output/fits/<subject>/<pose>/{smpl_params,displacements,registered.obj,alignment,quality}
           + output/fits/<subject>/manifest.json
```

The tier boundary is deliberately one-directional: Tier 3 never writes back into Tier 2's
artefacts, and `D` is defined as a pure difference against a baseline Tier 3 can regenerate.

## 5. Contract

**This is the seam between components.** Every dataclass field, function signature and artefact
schema below is authoritative — the loop's `integration-engineer` reconciles specialist outputs
against this section.

Repo convention followed (as with `CalibrationResult` in `calibration/pipeline.py` and
`RefinementResult` in `fitting/optimiser.py`): **module-local result dataclasses live beside their
module**; only the cross-tier PSD-boundary types go in `scantosmpl/types.py`.

### 5.1 Dataclasses

**`scantosmpl/pointcloud/io.py`**

```python
@dataclass
class PointCloud:
    points: np.ndarray            # (N, 3) float64. frame per `frame` field.
    normals: np.ndarray | None    # (N, 3) float64, unit length, same frame as points
    colors: np.ndarray | None     # (N, 3) float32 in [0, 1]
    source_path: Path
    frame: Literal["source", "smpl_world"] = "source"   # guard — assert before fitting
    units: Literal["arbitrary", "metres"] = "arbitrary"
```

**`scantosmpl/pointcloud/preprocess.py`**

```python
@dataclass
class PreprocessStats:
    n_input: int
    n_after_outlier_removal: int
    n_output: int
    outlier_fraction: float       # (n_input - n_after_outlier_removal) / n_input
    voxel_size_source_units: float
    bbox_diagonal_source_units: float
    normals_estimated: bool
```

**`scantosmpl/pointcloud/align.py`**

```python
@dataclass
class CloudAlignment:
    """Similarity transform: p_smpl = scale * (rotation @ p_source) + translation."""
    scale: float                  # source units -> metres
    rotation: np.ndarray          # (3, 3) float64, proper (det = +1)
    translation: np.ndarray       # (3,) float64, metres, SMPL/world frame
    inlier_rmse_m: float          # Open3D ICP inlier RMSE, metres
    fitness: float                # Open3D ICP fitness (fraction of inliers), [0, 1]
    n_candidates: int             # 24
    candidate_index: int          # which enumerated rotation won, [0, 24)
    converged: bool

    def apply(self, points: np.ndarray) -> np.ndarray: ...   # (N,3) source -> (N,3) SMPL/world
    def as_matrix(self) -> np.ndarray: ...                   # (4,4) float64 homogeneous
```

**`scantosmpl/evaluation/surface_metrics.py`**

```python
@dataclass
class ChamferReport:
    """7.M-compliant surface report. Deliberately has NO combined/fused field —
    7.M3 forbids reporting a single number, so the type makes it unrepresentable."""
    cloud_to_mesh_mm: dict[str, float]      # keys: mean, median, rms, p95, max   (7.M1: point-to-SURFACE)
    mesh_to_cloud_mm: dict[str, float]      # keys: mean, median, rms, p95, max   (7.M2: vertex-to-point)
    tessellation_floor_mm: dict[str, float] # keys: mean, max                      (7.M5)
    n_cloud_points: int
    n_mesh_vertices: int
    cloud_to_mesh_method: str = "point_to_surface_open3d_raycasting"
    mesh_to_cloud_method: str = "vertex_to_nearest_point"
    units: str = "mm"                                                              # (7.M4)
```

**`scantosmpl/fitting/surface.py`**

```python
@dataclass
class SurfaceStage:
    name: str
    params: list[str]             # subset of betas|body_pose|global_orient|translation|displacements
    n_iterations: int
    w_chamfer: float = 1.0
    w_normal: float = 0.0
    w_laplacian: float = 0.0
    w_displacement_reg: float = 0.0
    w_pose_prior: float = 0.0
    w_shape_reg: float = 0.0
    learning_rate: float = 1e-2

@dataclass
class SurfaceFitResult:
    betas: np.ndarray             # (10,)
    body_pose: np.ndarray         # (69,)
    global_orient: np.ndarray     # (3,)
    translation: np.ndarray       # (3,)
    scale: float                  # carried through unchanged from Tier 2 (D6)
    displacements: np.ndarray     # (6890, 3) float32, POSED WORLD frame, metres (D4)
    vertices: np.ndarray          # (6890, 3) = base_vertices + displacements
    base_vertices: np.ndarray     # (6890, 3) = SMPL(β,θ,t,s) with D = 0
    betas_locked: bool
    loss_history: dict[str, list[float]]
    metrics: dict[str, float] = field(default_factory=dict)
```

**`scantosmpl/types.py`** — PSD-boundary types only.

```python
DISPLACEMENT_FRAME: Literal["posed_world"] = "posed_world"
SMPL_NUM_VERTICES: int = 6890
SMPL_NUM_FACES: int = 13776

@dataclass
class Tier3Quality:
    """Per-pose fit quality persisted alongside D (7.B7)."""
    chamfer_cloud_to_mesh_mean_mm: float
    chamfer_cloud_to_mesh_median_mm: float
    chamfer_cloud_to_mesh_rms_mm: float
    chamfer_mesh_to_cloud_mean_mm: float
    chamfer_mesh_to_cloud_median_mm: float
    chamfer_mesh_to_cloud_rms_mm: float
    tessellation_floor_mean_mm: float
    tessellation_floor_max_mm: float
    icp_inlier_rmse_mm: float
    icp_fitness: float
    displacement_mean_mm: float
    displacement_p95_mm: float
    pa_mpjpe_mm: float | None = None        # carried from Tier 2
    median_reproj_px: float | None = None   # carried from Tier 2

@dataclass
class PoseArtefact:
    """One pose's entry in the corpus manifest (7.B6, 7.B8)."""
    pose_name: str
    directory: str                # relative to the manifest, e.g. "t-pose"
    oracle_only: bool             # 7.B8 — true = evaluation ceiling, MUST NOT enter PSD training
    betas_locked: bool            # 7.B1
    has_displacements: bool
    has_pointcloud: bool
    quality: Tier3Quality
```

`FittingResult.displacements` already exists as `(6890, 3) | None` and is reused unchanged; a
`displacement_frame: str = DISPLACEMENT_FRAME` field is added beside it so an in-memory result is
as self-describing as the on-disk artefact (7.B3).

### 5.2 Config (`scantosmpl/config.py`)

```python
@dataclass
class Tier3Config:
    """Tier 3: point-cloud surface refinement configuration."""

    # --- Preprocessing (unit-free, D8) ---
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    target_points: int = 50_000          # after downsample; 0 = keep all
    voxel_fraction_of_bbox: float = 0.002   # voxel = fraction * bbox diagonal (source units)
    estimate_normals: bool = True
    normal_knn: int = 30

    # --- Alignment (S1) ---
    icp_max_iterations: int = 100
    icp_threshold_frac: float = 0.05     # correspondence distance as fraction of SMPL bbox diagonal
    icp_min_fitness: float = 0.5         # below this, alignment is reported as not converged

    # --- Fitting (S2/S3) ---
    lock_betas: bool = False             # 7.B1
    betas_source: Path | None = None     # smpl_params.npz of the reference-pose fit
    chamfer_chunk_size: int = 10_000
    chamfer_huber_delta_m: float = 0.02
    chamfer_trim_quantile: float = 0.95  # drop the worst 5% of per-point residuals (cloud outliers)
    body_part_weights: dict[str, float] = field(default_factory=lambda: {
        "torso": 1.0, "arms": 0.7, "legs": 0.7, "head": 0.5, "hands": 0.3, "feet": 0.4,
    })
    use_semantic_weighting: bool = True  # False => uniform, for the AC 7.3 A/B

    # --- Metric (7.M) ---
    tessellation_floor_samples: int = 100_000
    tessellation_floor_seed: int = 0

    # --- Output ---
    subject_id: str = "subject"
    oracle_only: bool = False            # 7.B8
    save_debug: bool = True
    debug_dir: Path = Path("output/debug/surface")
```

`FittingConfig.body_part_weights` / `w_chamfer` / `w_normal` / `w_laplacian` already exist but are
unused; `Tier3Config` supersedes them for Tier 3. `PipelineConfig` gains
`tier3: Tier3Config = field(default_factory=Tier3Config)`.

### 5.3 Function signatures

**`scantosmpl/pointcloud/io.py`**

```python
def load_pointcloud(path: Path, *, max_points: int | None = None) -> PointCloud:
    """Load a PLY or OBJ. OBJ meshes contribute their vertices. Returns points in the
    SOURCE frame with arbitrary units (frame='source', units='arbitrary').
    Raises FileNotFoundError, or ValueError on an unsupported suffix / empty cloud."""

def save_pointcloud(cloud: PointCloud, path: Path) -> None:
    """Write PLY (binary). Used for debug artefacts only."""
```

**`scantosmpl/pointcloud/preprocess.py`**

```python
def preprocess_cloud(cloud: PointCloud, cfg: Tier3Config) -> tuple[PointCloud, PreprocessStats]:
    """Statistical outlier removal -> unit-free voxel downsample to ~cfg.target_points
    -> optional normal estimation. Frame and units are unchanged (still 'source').
    Deterministic: no RNG anywhere in this path (D12)."""
```

**`scantosmpl/pointcloud/align.py`**

```python
def pca_triad(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (centroid (3,), axes (3,3) as COLUMNS ordered by descending eigenvalue,
    extents (3,) = sqrt of eigenvalues)."""

def enumerate_proper_rotations(src_axes: np.ndarray, dst_axes: np.ndarray) -> list[np.ndarray]:
    """All 24 proper rotations (det=+1) mapping the src PCA triad onto the dst triad —
    the axis-permutation x sign-flip group. Deterministic order (D9)."""

def align_cloud_to_smpl(
    cloud: PointCloud,
    mesh_vertices: np.ndarray,     # (6890, 3) float64, SMPL/world, metres
    mesh_faces: np.ndarray,        # (13776, 3) int64
    cfg: Tier3Config,
) -> tuple[PointCloud, CloudAlignment]:
    """Align the cloud TO the SMPL mesh (never the reverse — CLAUDE.md decision log).
    PCA init -> 24 candidate rotations -> scaled point-to-plane ICP from each ->
    keep lowest inlier RMSE. Returns the transformed cloud (frame='smpl_world',
    units='metres') and the recovered 7-DoF similarity. Deterministic."""
```

**`scantosmpl/pointcloud/segment.py`**

```python
SMPL_PART_GROUPS: dict[str, list[int]] = {
    "torso": [0, 3, 6, 9, 12, 13, 14],   # pelvis, spine1-3, neck, collars
    "head":  [15],
    "arms":  [16, 17, 18, 19],           # shoulders, elbows
    "hands": [20, 21, 22, 23],           # wrists, hands
    "legs":  [1, 2, 4, 5],               # hips, knees
    "feet":  [7, 8, 10, 11],             # ankles, feet
}   # keys match FittingConfig/Tier3Config.body_part_weights exactly; the 24 joints partition

def smpl_part_labels(lbs_weights: np.ndarray) -> np.ndarray:
    """(6890, 24) lbs_weights -> (6890,) int part ids, via argmax joint -> group (D7)."""

def vertex_part_weights(lbs_weights: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """(6890,) float32 per-vertex loss weight from the part groups."""

def transfer_labels_to_cloud(
    cloud_points: np.ndarray, mesh_vertices: np.ndarray, vertex_labels: np.ndarray
) -> np.ndarray:
    """(N,) int part ids for cloud points, from their nearest mesh vertex."""
```

**`scantosmpl/evaluation/surface_metrics.py`**

```python
def point_to_surface_distances(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """(N,) float64 UNSIGNED point-to-triangle distances in metres, via
    open3d.t.geometry.RaycastingScene.compute_distance. This is 7.M1's binding
    cloud->mesh measurement. Non-differentiable — reporting only."""

def vertex_to_point_distances(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(V,) float64 vertex-to-nearest-cloud-point distances in metres (7.M2)."""

def tessellation_floor(
    vertices: np.ndarray, faces: np.ndarray, *, n_samples: int = 100_000, seed: int = 0
) -> dict[str, float]:
    """7.M5. Area-weighted uniform surface samples -> distance to nearest VERTEX.
    Returns {'mean': mm, 'max': mm}. Seeded (D12)."""

def chamfer_report(
    cloud_points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, cfg: Tier3Config
) -> ChamferReport:
    """Assemble the full 7.M-compliant report. Both directions, never fused (7.M3)."""
```

**`scantosmpl/fitting/surface_losses.py`**

```python
def chamfer_loss(
    vertices: torch.Tensor,             # (V, 3) or (1, V, 3), SMPL/world metres, requires_grad
    cloud: torch.Tensor,                # (N, 3), SMPL/world metres
    *,
    vertex_weights: torch.Tensor | None = None,   # (V,)  semantic weights (D7)
    cloud_weights: torch.Tensor | None = None,    # (N,)  transferred semantic weights
    chunk_size: int = 10_000,
    huber_delta: float = 0.02,
    trim_quantile: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional chamfer from ONE chunked torch.cdist (D3). Returns (scalar loss,
    {'mesh_to_cloud_m', 'cloud_to_mesh_m'} detached diagnostics). Huber-bounded and
    quantile-trimmed so cloud outliers cannot dominate the gradient."""

def normal_consistency_loss(
    vertices: torch.Tensor, faces: torch.Tensor,
    cloud: torch.Tensor, cloud_normals: torch.Tensor, *, chunk_size: int = 10_000,
) -> torch.Tensor:
    """1 - |cos| between each cloud point's normal and the normal of its nearest mesh
    vertex. Absolute value: cloud normal orientation from photogrammetry is unreliable."""

def build_uniform_laplacian(faces: np.ndarray, n_verts: int) -> torch.Tensor:
    """Sparse (V, V) uniform (graph) Laplacian for the fixed SMPL topology. Cached."""

def laplacian_smoothing_loss(displacements: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Mean squared magnitude of L @ D — penalises high-frequency displacement, not
    displacement itself, so smooth bulges survive and per-vertex spikes do not."""

def displacement_regularisation(displacements: torch.Tensor) -> torch.Tensor:
    """Mean squared ||D|| — keeps D minimal so it cannot silently absorb model error."""
```

**`scantosmpl/smpl/model.py`** (SMPL+D)

```python
# New nn.Parameter alongside betas/body_pose/global_orient/translation/scale:
self.displacements = nn.Parameter(torch.zeros(1, self.NUM_VERTICES, 3, device=self.device))

def forward(
    self, betas=None, body_pose=None, global_orient=None, translation=None, scale=None,
    displacements: torch.Tensor | None = None,   # (1, 6890, 3) or None -> stored parameter
    apply_displacements: bool = True,            # False forces the D=0 baseline
) -> SMPLOutput:
    """... vertices = (smplx_output.vertices * scale) + displacements

    D is POSED WORLD frame, metres, added AFTER scale (D4). The baseline PSD needs is
    exactly forward(..., apply_displacements=False).vertices, so
    D == forward(...).vertices - forward(..., apply_displacements=False).vertices holds
    by construction. Joints are NOT displaced — D is a surface quantity."""
```

**`scantosmpl/fitting/surface.py`**

```python
DEFAULT_SURFACE_STAGES: list[SurfaceStage] = [
    SurfaceStage(name="model_fit",    params=["betas", "body_pose", "global_orient", "translation"],
                 n_iterations=300, w_chamfer=1.0, w_pose_prior=0.01, w_shape_reg=0.01,
                 learning_rate=5e-3),
    SurfaceStage(name="displacement", params=["displacements"],
                 n_iterations=250, w_chamfer=1.0, w_normal=0.1, w_laplacian=0.1,
                 w_displacement_reg=0.01, learning_rate=1e-3),
]
# NOTE: "scale" appears in NO stage (D6). With lock_betas=True, "betas" is dropped from
# model_fit's params at construction time — it is never merely zero-weighted.

class Tier3SurfaceFitter:
    def __init__(self, smpl_model: SMPLModel, cfg: Tier3Config) -> None: ...

    def fit(
        self,
        tier2: RefinementResult,
        cloud: PointCloud,                  # MUST be frame='smpl_world', units='metres'
        *,
        stages: list[SurfaceStage] | None = None,
        locked_betas: np.ndarray | None = None,   # (10,) when cfg.lock_betas
    ) -> SurfaceFitResult:
        """Stages S2 then S3 (D5). Asserts cloud.frame == 'smpl_world' up front —
        a source-frame cloud is the single most damaging silent failure here."""
```

**`scantosmpl/fitting/surface_pipeline.py`**

```python
@dataclass
class Tier3Result:
    fit: SurfaceFitResult
    alignment: CloudAlignment
    report: ChamferReport
    preprocess: PreprocessStats
    quality: Tier3Quality
    artefact_dir: Path

class Tier3Pipeline:
    def __init__(self, smpl_model: SMPLModel, cfg: Tier3Config) -> None: ...

    def run(
        self,
        tier2: RefinementResult,
        pointcloud_path: Path,
        *,
        pose_name: str,
        output_dir: Path,
    ) -> Tier3Result:
        """S1 -> S2 -> S3 -> metric -> persist. Writes the per-pose artefact directory and
        creates-or-updates the subject manifest."""
```

**`scantosmpl/fitting/artefacts.py`** (the 7.B boundary)

```python
def write_pose_artefacts(
    out_dir: Path, fit: SurfaceFitResult, alignment: CloudAlignment,
    quality: Tier3Quality, faces: np.ndarray, *, pose_name: str,
) -> None:
    """Write smpl_params.npz, displacements.npz, registered.obj, alignment.json, quality.json.
    ASSERTS ON WRITE (fail loudly, never silently corrupt the PSD residual):
      - D.shape == (6890, 3) and dtype float32                       (7.B2, 7.B4)
      - vertices.shape == (6890, 3), faces.shape == (13776, 3)       (7.B4)
      - faces byte-identical to the SMPL template's face array        (7.B4 ordering)
      - no NaN/Inf in D, vertices or params
      - displacement_frame == 'posed_world' written as an explicit field (7.B3)
      - allclose(base_vertices + D, vertices)                        (D4 identity)"""

def update_manifest(
    manifest_path: Path, entry: PoseArtefact, *, subject_id: str,
    smpl_meta: dict, beta_policy: dict,
) -> None:
    """Create-or-update output/fits/<subject>/manifest.json (7.B6). Raises if an existing
    manifest disagrees on subject_id, displacement_frame, gender, num_betas or faces_sha256."""

def load_locked_betas(path: Path) -> np.ndarray:
    """(10,) float64 from a reference-pose smpl_params.npz, for --betas-from (7.B1)."""
```

### 5.4 CLI

```python
@main.command("fit-surface")
@click.option("--tier2-dir", required=True, type=click.Path(exists=True))   # Phase 5 output dir
@click.option("--pointcloud", required=True, type=click.Path(exists=True))
@click.option("--subject", default="subject")
@click.option("--pose-name", required=True)
@click.option("--lock-betas", is_flag=True, default=False)                  # 7.B1
@click.option("--betas-from", type=click.Path(exists=True), default=None)   # reference smpl_params.npz
@click.option("--oracle-only", is_flag=True, default=False)                 # 7.B8
@click.option("--no-semantic-weighting", is_flag=True, default=False)       # AC 7.3 A/B
@click.option("--gender", default="neutral", type=click.Choice(["neutral", "male", "female"]))
@click.option("--output", required=True, type=click.Path())
def fit_surface(...): ...
```

`--lock-betas` without `--betas-from` is an error (there is nothing to lock *to*); `--betas-from`
implies `--lock-betas`.

## 6. User flows

```bash
# Reference pose: refine β from surface geometry (AC 7.4 mode).
scantosmpl fit-surface \
    --tier2-dir output/debug/refinement/ \
    --pointcloud data/t-pose/pointcloud.ply \
    --subject dan --pose-name t-pose \
    --output output/fits/dan/

# Every subsequent corpus pose: β frozen from the reference fit (7.B1 mode).
scantosmpl fit-surface \
    --tier2-dir output/debug/refinement_a-pose/ \
    --pointcloud data/a-pose/pointcloud.ply \
    --subject dan --pose-name a-pose \
    --lock-betas --betas-from output/fits/dan/t-pose/smpl_params.npz \
    --output output/fits/dan/

# Held-out pose fitted as a PSD evaluation ceiling — flagged so it can never train (7.B8).
scantosmpl fit-surface ... --pose-name a-pose-heldout --oracle-only \
    --lock-betas --betas-from output/fits/dan/t-pose/smpl_params.npz
```

## 7. Data model & artefacts

### 7.1 Per-pose directory — `output/fits/<subject>/<pose_name>/`

| File | Contents |
|---|---|
| `smpl_params.npz` | `betas` (10,) f8, `body_pose` (69,) f8, `global_orient` (3,) f8, `translation` (3,) f8, `scale` () f8, `base_vertices` (6890,3) f4, `vertices` (6890,3) f4, `betas_locked` bool |
| `displacements.npz` | `D` (6890,3) **float32**, `displacement_frame` `"posed_world"`, `faces_sha256` str, `n_vertices` 6890, `n_faces` 13776 |
| `registered.obj` | `base_vertices + D`, SMPL/world frame, metres, template face ordering |
| `alignment.json` | `{"scale": f, "rotation": [[..]], "translation": [..], "inlier_rmse_m": f, "fitness": f, "candidate_index": i, "converged": bool}` — the cloud→SMPL similarity, **kept separate from `D` (7.B5)** |
| `quality.json` | `Tier3Quality` as flat JSON (7.B7) |
| `pointcloud_aligned.ply` | debug only, written when `save_debug` |

`D` is stored as float32 (≈83 KB) rather than float64 — well below the sub-millimetre precision
that matters, and it keeps the corpus small.

### 7.2 Manifest — `output/fits/<subject>/manifest.json` (7.B6)

```json
{
  "schema_version": 1,
  "subject_id": "dan",
  "displacement_frame": "posed_world",
  "smpl": {"gender": "neutral", "num_betas": 10, "num_vertices": 6890,
           "num_faces": 13776, "faces_sha256": "<hex>"},
  "beta_policy": {"mode": "locked", "source_pose": "t-pose", "betas_sha256": "<hex>"},
  "poses": [
    {"pose_name": "t-pose", "directory": "t-pose", "oracle_only": false,
     "betas_locked": false, "has_displacements": true, "has_pointcloud": true,
     "quality": { "...Tier3Quality..." }},
    {"pose_name": "a-pose-heldout", "directory": "a-pose-heldout", "oracle_only": true,
     "betas_locked": true, "has_displacements": true, "has_pointcloud": true,
     "quality": { "..." }}
  ]
}
```

`beta_policy.mode` is `"refined"` (β optimised, single-pose) or `"locked"` (β frozen). PSD's
`build_corpus()` asserts shared β; `betas_sha256` lets it do so in O(1) before loading arrays.

### 7.3 Synthetic fixture — `tests/integration/fixtures/synthetic_cloud/`

Generated by a committed, seeded script (`make_fixture.py`, seed 0), producing `cloud.ply` +
`ground_truth.json`. Construction, from the Tier 2 SMPL mesh:

1. Area-weighted uniform sample of 60 000 surface points.
2. Inject a **known "clothing" offset**: +4 mm along the outward normal for all vertices whose part
   label is `torso`, 0 elsewhere → this is `D_true`.
3. Add Gaussian noise σ = 1 mm along the normal (Meshroom-like reconstruction noise).
4. Add 2 % uniform outliers inside the bbox (exercises trimming).
5. Apply a **known similarity**: `scale = 0.371`, a fixed rotation (≈ 137° about a non-axis unit
   vector), translation `(1.7, -0.4, 2.3)` — i.e. a genuinely Meshroom-like arbitrary frame.

`ground_truth.json` records the inverse similarity, `D_true` statistics, the noise σ and the
outlier fraction, so alignment recovery, chamfer floor and `D` recovery are all assertable exactly.
This is a **known-answer** test, not a regression snapshot.

### 7.4 Real-cloud path

When `data/t-pose/pointcloud.ply` exists, `tests/integration/test_tier3_integration.py` runs the
same pipeline against it and asserts AC 7.1. Absent that file the test **skips with an explicit
reason string** — a skip is not a pass (per the `py-test` skill), and the Tier 3 gate is reported
as `DEFERRED`, never `PASS`.

## 8. Non-goals

- Not implementing PSD / Tier 4 — only satisfying its eight constraints.
- Not predicting or storing `δ_local`; `R_v(θ)⁻¹` is PSD's to apply (D4).
- Not modifying Tier 1 or Tier 2 behaviour, metrics or artefacts.
- Not wiring `fit-combined` / the full Phase 8 orchestrator.
- Not adding any dependency (no Kaolin, no `rtree`, no PyTorch3D).
- Not using Meshroom sfm camera poses for alignment.
- Not fitting hands/face detail — SMPL (not SMPL-X) topology throughout, 6890/13776 fixed.
- Not multi-subject or multi-pose corpus assembly.

## 9. Rollout / migration

Almost entirely new modules — `scantosmpl/pointcloud/` is empty today, and
`surface_metrics.py` / `surface_losses.py` / `surface.py` / `surface_pipeline.py` /
`artefacts.py` do not exist.

Two touches to existing code, both additive:

- **`scantosmpl/smpl/model.py`** gains a `displacements` parameter and two `forward` kwargs, both
  defaulting to the current behaviour (`displacements=None`, `apply_displacements=True` with a
  zero-initialised parameter ⇒ vertices unchanged). Every existing caller
  (`SMPLOptimiser`, `Phase5Pipeline`, `tests/test_smpl_model.py`) keeps working untouched.
  **Risk to watch:** `SMPLOptimiser._get_params` raises on unknown names, so adding a parameter is
  safe, but `get_params_dict()` gains a key — any consumer doing an exact-key comparison on that
  dict must be checked (grep shows only debug JSON writers today).
- **`scantosmpl/types.py`** gains `Tier3Quality`, `PoseArtefact`, the three module constants, and
  one defaulted field on `FittingResult`. No existing field changes shape.

No serialised on-disk config or cached artefact depends on any of this, so no back-compat shim is
needed — consistent with the `selfcal-default-extrinsics` precedent.

## 10. Acceptance Criteria

**The loop's exit condition.** Grouped by the REVIEW.md requirement each one discharges.

### Metric definition (7.M — binding)

- **AC1** (7.M1) — Cloud→mesh distance is point-to-**surface**, not vertex-based. **Evidence**:
  `pytest tests/test_surface_metrics.py::test_point_to_surface_analytic -v` — a query point 5 mm
  above the centroid of a 1 m triangle returns 0.005 m ± 1e-6, whereas the nearest *vertex* is
  ≈ 0.577 m away. `ChamferReport.cloud_to_mesh_method == "point_to_surface_open3d_raycasting"`.
- **AC2** (7.M3) — Directions are never fused. **Evidence**:
  `python -c "import dataclasses as d; from scantosmpl.evaluation.surface_metrics import ChamferReport; print([f.name for f in d.fields(ChamferReport)])"` contains `cloud_to_mesh_mm` and
  `mesh_to_cloud_mm` and **no** field whose name contains `chamfer_mm`, `combined`, `total` or `mean_both`.
- **AC3** (7.M4) — Aggregation + units are explicit. **Evidence**: `quality.json` carries
  `mean`/`median`/`rms` keys for both directions and every key name ends in `_mm`;
  `ChamferReport.units == "mm"`.
- **AC4** (7.M5) — Tessellation floor is measured and reported. **Evidence**: `quality.json` has
  `tessellation_floor_mean_mm` and `tessellation_floor_max_mm`;
  `pytest tests/test_surface_metrics.py::test_tessellation_floor_bound -v` asserts
  `max <= 1.05 * L_max/sqrt(3)` for the SMPL template (the equilateral worst case — measured
  5.77 mm at L = 10 mm).

### Alignment (Phase 6 / 6.3–6.5)

- **AC5** — ICP recovers a known similarity. **Evidence**:
  `pytest tests/integration/test_tier3_integration.py::test_alignment_recovers_ground_truth -v` —
  on the synthetic fixture, `|scale/scale_true − 1| < 0.01`, rotation geodesic error `< 1.0°`,
  translation error `< 5 mm`, `alignment.converged is True`.
- **AC6** (6.2) — Outlier removal keeps the body. **Evidence**:
  `PreprocessStats.outlier_fraction >= 0.8 * injected_outlier_fraction` and
  `n_after_outlier_removal >= 0.95 * n_inliers` on the fixture.
- **AC7** (D9/D12) — Alignment is deterministic. **Evidence**:
  `pytest tests/test_pointcloud.py::test_alignment_deterministic -v` — two runs on identical input
  give bitwise-identical `rotation`, `translation`, `scale`.

### Surface fit (7.1–7.5, 7.7)

- **AC8** (7.2) — Refinement improves on Tier 2 by ≥ 40 %. **Evidence**:
  `quality.json` from the fixture run vs a `D=0`, Tier-2-params baseline recorded in the same run:
  `cloud_to_mesh_mean_mm` drops by ≥ 40 %. Both numbers printed in `summary.txt`.
- **AC9** (7.1) — **Deferred gate.** On real scanner data, `cloud_to_mesh_mean_mm < 8.0`.
  **Evidence**: `pytest tests/integration/test_tier3_integration.py::test_real_cloud_chamfer -v`.
  Until `data/t-pose/pointcloud.ply` exists this test **skips**, `summary.txt` prints
  `TIER 3 GATE: DEFERRED (no real point cloud)`, and the loop MUST NOT claim the gate passed.
  On the synthetic fixture the equivalent bound is the injectable floor:
  `cloud_to_mesh_mean_mm < 3.0` (1 mm noise + residual fit error).
- **AC10** (7.3) — Semantic weighting beats uniform on the torso. **Evidence**:
  `pytest tests/integration/test_tier3_integration.py::test_semantic_weighting_ab -v` runs the
  fixture with `use_semantic_weighting` True/False; torso-labelled `cloud_to_mesh_mean_mm` is lower
  with weighting on. Both numbers written to `output/debug/surface/semantic_ab.json`.
- **AC11** (7.4) — β refinement improves proportions. **Evidence**: with `lock_betas=False`, the
  fitted shoulder width (joints 16↔17) and waist girth move toward the fixture's ground-truth mesh
  vs the Tier 2 input; deltas recorded in `summary.txt`. Asserted only in the β-refine mode — this
  AC is **inapplicable** in `--lock-betas` runs, which is exactly REVIEW.md's 7.4-vs-7.B1
  resolution (D10).
- **AC12** (7.5) — θ stays plausible, no new self-intersections. **Evidence**:
  `pytest tests/test_surface_fitting.py::test_pose_plausible_no_new_intersections -v` — per-joint
  axis-angle change from Tier 2 `< 15°`, and the self-intersecting-face count does not increase
  by more than 5 (REVIEW 5.7's existing threshold).
- **AC13** (7.7) — Optimisation < 60 s on GPU with a 50 K cloud. **Evidence**: fixture integration
  test records wall-clock for S2+S3 in `summary.txt`; asserted `< 60.0` when CUDA is available.
  (Measured budget: 74 ms/iter × 550 iters ≈ 41 s.)

### PSD boundary (7.B1–7.B8 — contract requirements, each fails loudly)

- **AC14** (7.B1) — `--lock-betas` makes β non-trainable. **Evidence**:
  `pytest tests/test_surface_fitting.py::test_lock_betas_freezes_shape -v` — with `lock_betas=True`
  and supplied β, `np.array_equal(result.betas, locked_betas)` exactly (not "close"), `"betas"` is
  absent from every stage's param list, and `result.betas_locked is True`. Additionally
  `--lock-betas` without `--betas-from` exits non-zero with a clear message.
- **AC15** (7.B2) — `D` is persisted per pose, always. **Evidence**: `displacements.npz` exists
  after every run (including when `D` is near zero), and `np.load(...)["D"].shape == (6890, 3)`
  with `dtype == float32`. No config flag can suppress it.
- **AC16** (7.B3) — The frame is documented and asserted. **Evidence**:
  `np.load("displacements.npz")["displacement_frame"] == "posed_world"`, the manifest's top-level
  `displacement_frame` matches, and
  `pytest tests/test_surface_fitting.py::test_displacement_frame_identity -v` asserts
  `allclose(forward(D=0).vertices + D, forward().vertices, atol=1e-6)`.
- **AC17** (7.B4) — Topology invariant. **Evidence**:
  `pytest tests/test_artefacts.py::test_topology_assertions -v` — writing with a resampled vertex
  count, a permuted face array, or a NaN in `D` each raise; a correct write round-trips
  6890/13776 and `faces_sha256` matches the template.
- **AC18** (7.B5) — Global transform is separable from `D`. **Evidence**:
  `pytest tests/integration/test_tier3_integration.py::test_similarity_invariance -v` — re-running
  the fixture with the cloud pre-multiplied by a *different* known similarity yields the same `D`
  to within 0.5 mm mean, while `alignment.json` absorbs the difference. This is the decisive check
  that no similarity has been baked into the displacement field.
- **AC19** (7.B6) — Manifest locates `(β, θ, D, quality)` by pose name. **Evidence**:
  `pytest tests/test_artefacts.py::test_manifest_roundtrip -v` — two `fit-surface` runs for
  different pose names produce one manifest with two `poses[]` entries; each entry's `directory`
  resolves to a dir containing all four artefacts. A second subject writing into the same manifest
  raises.
- **AC20** (7.B7) — Per-pose quality persisted. **Evidence**: `quality.json` deserialises into
  `Tier3Quality` with every field populated (non-`None` except the two Tier-2 carry-throughs when
  the Tier 2 metrics are absent), and the same dict appears under the manifest entry's `quality`.
- **AC21** (7.B8) — Oracle poses are marked. **Evidence**: `--oracle-only` sets
  `poses[i].oracle_only == true` in the manifest; `pytest tests/test_artefacts.py::test_oracle_flag -v`
  confirms the default is `false` and that the flag survives a manifest update by a later run.

### Hygiene

- **AC22** — No new dependency. **Evidence**: `git diff pyproject.toml` shows no change to
  `dependencies`; `grep -rn "import kaolin\|from kaolin\|import rtree" scantosmpl/ tests/` returns
  nothing.
- **AC23** — Lint + typecheck green. **Evidence**: `py-lint` and `py-typecheck` exit 0 on every
  changed module.
- **AC24** — Full suite green. **Evidence**: `py-test` — `pytest tests/ -x --tb=short` exits 0, and
  `pytest tests/integration/test_tier3_integration.py -v` passes (real-cloud test may skip; the skip
  reason must name the missing file).

## 11. Risks

Ordered by likelihood × impact.

- **R1 (High × High) — AC 7.1's 8 mm cannot be certified without a real cloud.** No PLY exists in
  the repo; memory records Meshroom as the critical path. *Mitigation:* D11 — the synthetic fixture
  gates everything mechanically verifiable, AC9 is explicitly a deferred gate, and `summary.txt`
  prints `DEFERRED` rather than `PASS`. The confounds REVIEW.md lists (Meshroom noise, tessellation
  floor, clothing/hair absent from SMPL) are reported as separate lines, not absorbed into one
  number.

- **R2 (Medium × High) — `D` silently absorbs shape or alignment error.** The whole PSD residual
  depends on `D` being off-manifold geometry only. *Mitigation:* D5's staged order, D6's frozen
  scale, `displacement_regularisation` keeping `‖D‖` minimal, and AC18's similarity-invariance test
  — which is the only check that would actually catch a baked-in transform.

- **R3 (Medium × Medium) — vertex-based chamfer *loss* shrink-wraps or facets.** The loss is
  vertex-based even though the metric is point-to-surface, so the optimiser can pull vertices onto
  cloud points rather than matching the surface. *Mitigation:* bidirectional loss (D3) plus
  Laplacian smoothing; 7.M6 explicitly permits switching the cloud→mesh side of the loss to
  point-to-triangle if faceting appears, and the metric is unaffected either way. The fix-cycle
  should watch `cloud_to_mesh` (surface metric) diverging from the loss's own diagnostic.

- **R4 (Medium × Medium) — PCA alignment lands in a wrong-but-plausible basin.** A 180°-flipped
  body can produce a low RMSE against a roughly symmetric torso. *Mitigation:* D9 enumerates all 24
  proper rotations rather than guessing, selection is by ICP inlier RMSE, and `icp_min_fitness`
  gates the result. AC5 asserts recovery against a ground-truth transform. If a real cloud still
  flips, the escape hatch is a caller-supplied coarse transform — recorded here, deliberately
  *not* built, to avoid a config knob nothing exercises.

- **R5 (Low × High) — SMPL+D changes `model.py` under Tier 1/2.** A regression here breaks
  already-passing phases. *Mitigation:* every new argument defaults to current behaviour (§9), and
  AC24 runs the full suite including the Phase 5 integration test.

- **R6 (Low × Medium) — 7.M1's named function is not used.** A future reader diffing REVIEW.md
  against the code will find `point_to_mesh_distance` absent. *Mitigation:* D2 records the measured
  wheel-ceiling evidence and the 7.M6 justification; the repo spec repeats it. REVIEW.md 7.M1
  should be amended to name the requirement (point-to-surface) rather than the implementation —
  flagged for Dan, not done by this feature.

- **R7 (Low × Medium) — Open3D `RaycastingScene` is CPU-only and float32.** For 500 K points ×
  13 776 triangles it is fast enough, but precision is float32. *Mitigation:* metric precision of
  ~1e-4 mm is far below the 8 mm gate; the analytic test (AC1) pins it at 1e-6 m tolerance.
