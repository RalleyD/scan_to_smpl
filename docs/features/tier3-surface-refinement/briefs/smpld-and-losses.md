---
component: smpld-and-losses
agent: python-engineer
worktree: true   # parallel-safe with pointcloud-package and surface-metrics — no shared files
---

# Component Brief — smpld-and-losses

## Goal

Two things, both prerequisites for the surface fitter: (a) give `SMPLModel` a per-vertex
displacement field `D` in the **posed world** frame, additively and without disturbing any existing
Tier 1/2 caller; and (b) implement the differentiable surface losses — a chunked *bidirectional*
chamfer built from a single `torch.cdist`, plus normal consistency, Laplacian smoothing and `‖D‖`
regularisation.

The frame identity this brief establishes is the one PSD depends on, so it is asserted rather than
assumed:

```
D  ==  forward(β,θ,t,s).vertices  −  forward(β,θ,t,s, apply_displacements=False).vertices
```

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/smpl/model.py`
- `scantosmpl/fitting/surface_losses.py`
- `tests/test_surface_losses.py`

**Does NOT touch**:
- `scantosmpl/fitting/{losses,optimiser,pipeline,rear_views}.py` — Tier 2, must stay byte-unchanged.
  In particular do **not** "helpfully" teach `SMPLOptimiser` about displacements; the Tier 3 fitter
  is a separate class in a sibling brief.
- `scantosmpl/fitting/__init__.py` — export wiring is `tier3-pipeline-artefacts`' job (it would
  otherwise be a three-way collision). Your new module simply exists; it does not need re-exporting
  to be importable by its full path.
- `scantosmpl/smpl/joint_map.py`, `scantosmpl/smpl/__init__.py` (stays empty)
- `scantosmpl/config.py` — take loss hyperparameters as keyword arguments with the master §5.2
  defaults. Do not define `Tier3Config`.
- `scantosmpl/types.py`, `scantosmpl/pointcloud/*`, `scantosmpl/evaluation/*`
- Other components' `Owns` paths; `external/`, `models/`, `pyproject.toml`, `output/`

**Consumes**:
- `torch`, `numpy`, `scipy.sparse` — already dependencies. **No `kaolin`** (master D2).
- `SMPLOutput` from `scantosmpl/types.py` — existing, unchanged.

**Produces**:
- `SMPLModel.displacements` `nn.Parameter` `(1, 6890, 3)` and `forward(..., displacements=,
  apply_displacements=)` — master §5.3
- `chamfer_loss`, `normal_consistency_loss`, `build_uniform_laplacian`,
  `laplacian_smoothing_loss`, `displacement_regularisation` — master §5.3

## Steps

1. **SMPL+D in `model.py`.** Add `self.displacements = nn.Parameter(torch.zeros(1, self.NUM_VERTICES, 3, device=self.device))`
   beside the existing parameters, and the two new `forward` kwargs.
   - Apply as `vertices = (output.vertices * scale.unsqueeze(-1).unsqueeze(-1)) + disp` — **after**
     scale, so `D` is in final posed world metres (master D4).
   - **Joints are NOT displaced.** `D` is a surface quantity; displacing joints would silently
     corrupt every Tier 2 metric that compares against them.
   - Extend `set_params` and `get_params_dict` to include `displacements`.
   - **Backward compatibility is the acceptance bar here** (master R5): with the zero-initialised
     parameter and default kwargs, `forward()` must return bitwise-identical vertices to today's
     implementation.
   **Verify**: `py-typecheck` on `scantosmpl/smpl/model.py`, then `py-test` —
   `pytest tests/test_smpl_model.py -v` (the existing file, unmodified, must still pass).

2. **Frame-identity + backward-compat tests.** In `tests/test_surface_losses.py` add:
   - `test_displacement_frame_identity` (AC16) — for a non-zero random `D`,
     `allclose(forward(apply_displacements=False).vertices + D, forward().vertices, atol=1e-6)`.
   - `test_zero_displacement_is_noop` — `forward()` with the default zero parameter equals a
     freshly-constructed model's `forward()` exactly.
   - `test_joints_undisplaced` — joints are identical with and without `D`.
   - `test_displacement_grad_flows` — `loss.backward()` populates `model.displacements.grad`.
   **Verify**: `py-test` — `pytest tests/test_surface_losses.py -k displacement -v`.

3. **`chamfer_loss` — bidirectional, from ONE cdist (master D3).** Loop over cloud chunks of
   `chunk_size`; for each chunk compute `torch.cdist(vertices, chunk)` once and take **both**
   `min(dim=1)` (mesh→cloud, running minimum across chunks) and `min(dim=0)` (cloud→mesh, per-chunk
   final). A one-sided loss shrink-wraps the mesh into the densest cloud region — the failure 7.M3
   names — and the second direction is free from the same matrix, so there is no reason to omit it.
   - Apply `vertex_weights` / `cloud_weights` (the semantic weights) multiplicatively per term.
   - Robustify: Huber at `huber_delta` metres, then drop residuals above `trim_quantile` — clouds
     always carry outliers and an untrimmed mean lets them steer the gradient. This mirrors the
     Tier 2 lesson in `docs/phase5_tier2_improvement_plan.md` W1: fix the metric's sensitivity to a
     right-skewed tail before tuning anything.
   - Return `(scalar_loss, {"mesh_to_cloud_m": float, "cloud_to_mesh_m": float})` with the
     diagnostics **detached**.
   - Memory target: 6890 × 50 000 at `chunk_size=10_000` peaks ≈ **1.9 GiB** and runs fwd+bwd in
     ≈ 74 ms/iter on the 3080 Ti. Materially more means chunking is broken.
   **Verify**: `py-test` — `pytest tests/test_surface_losses.py -k chamfer -v`. Include: a
   known-answer case (cloud = mesh vertices shifted by a constant `d` ⇒ both directions ≈ `d`), a
   chunking-invariance test (`chunk_size` 1000 vs 10000 vs N give the same loss to 1e-6), an
   outlier-trimming test (adding far outliers barely moves the loss), and a gradient test.

4. **`normal_consistency_loss`, `build_uniform_laplacian`, `laplacian_smoothing_loss`,
   `displacement_regularisation`.**
   - Normals: `1 - |cos(angle)|` between each cloud point's normal and its nearest mesh vertex's
     normal. **Absolute value is deliberate** — photogrammetry normal orientation is unreliable and a
     signed term would fight the fit rather than regularise it.
   - Laplacian: sparse uniform (graph) Laplacian from `faces`, built once and cached (SMPL topology
     is fixed). Penalise `mean(||L @ D||²)`, i.e. the *roughness* of `D`, not its magnitude — a
     smooth soft-tissue bulge must survive, a per-vertex spike must not.
   - `displacement_regularisation` = `mean(||D||²)`, keeping `D` minimal so it cannot quietly absorb
     model error (master R2).
   **Verify**: `py-test` — `pytest tests/test_surface_losses.py -v`. Assert: the Laplacian is
   symmetric with zero row sums; a constant `D` (pure translation) has ≈ zero Laplacian loss but
   non-zero `displacement_regularisation`; a single-vertex spike has high Laplacian loss.

5. **Full lint, typecheck, and the Tier 2 regression guard.**
   **Verify**: `py-lint`, `py-typecheck`, `pytest tests/ -x --tb=short`, **and**
   `pytest tests/integration/test_phase5_integration.py -v -m gpu --timeout=600`. This brief is the
   only one touching `smpl/model.py`, so it owns proving Tier 2 is unregressed (master R5). Report
   the Phase 5 `pa_mpjpe_mm` and `median_reproj_px` in your notes so drift is visible.

## Definition of done

- Every step's verification skill is green, including the Phase 5 integration test.
- `pytest tests/test_smpl_model.py -v` passes **unmodified** — you may not edit that file.
- AC16's frame identity holds to `atol=1e-6`; joints are provably undisplaced.
- The `Produces` contract exactly matches master §5.3.
- No `kaolin`, no `pyproject.toml` change (AC22).
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new
  skill — specifically: (a) the measured Phase 5 metrics before/after, (b) the measured chamfer
  peak memory and ms/iter at 6890 × 50 000, and (c) whether any existing caller does an exact-key
  comparison on `get_params_dict()` (master §9 flags this as the one non-obvious backward-compat
  risk of adding a parameter). State explicitly that `pipeline-smoke` was not run and why.
