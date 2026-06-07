# Phase 5 — Option A Implementation + Supplements

## Context

After the EXIF coordinate fix, Phase 5 passes 8 of 11 integration tests. The 3 failures all stem from the 15px reprojection target being unrealistic given ViTPose's spatial accuracy on 6000×4000 images (per-joint RANSAC reproj 44–83px is the actual noise floor) AND from rear-view cameras where ViTPose swaps left/right body parts. These outlier views drag the mean reprojection to 479.8px and likely also pull the optimiser toward bad fits, contributing to the 4mm PA-MPJPE gap (33.6 vs 30mm).

**Decision (user):** proceed with Option A (tune + reframe metrics + handle rear views), and document Options A+B and C as spec supplements for future fallback.

---

## Option A — Implementation Plan (primary)

### A1. Detect & handle rear-view cameras

Rear-view cameras receive ViTPose detections where left/right are systematically swapped — they're not just noise, they're consistently *wrong*. Including them in the reprojection loss confuses the optimiser; including them in the metric inflates the mean.

**Approach:** Compute each camera's viewing direction in SMPL frame; classify as rear if it views the subject's back (dot product with consensus body forward vector < threshold).

Implementation in [pipeline.py](scantosmpl/fitting/pipeline.py):
- New method `_classify_rear_views(cameras_smpl, consensus) -> set[str]` — returns names of rear-facing cameras
- For rear cameras: drop from `kp2d_tensors`/`confs_tensors` passed to optimiser, OR apply L/R swap to their keypoints
- Use SMPL pelvis-to-spine vector as "forward" (it's the canonical body axis); rear = camera_to_body direction dot forward > 0 (camera is behind the body)

#### A1 - Theory

**How do we determine the front of the body?**

Consider the SMPL body in its canonical A-pose.

up_vector = pelvis -> neck = [0, 1, 0] <- The body's spine goes straight-up +Y.
shoulder_vector = left_shoulder - right_shoulder = [1, 0, 0] <- SMPL labels left shoulder right-handed at +X.

Therefore, consider the right-hand rule: point right index finger up +Y, curl middle finger down to -X, the thumb points in the resulting Z direction (towardss you).

The cross product (A x B) gives a vector perpendicular to `up_vector` and `shoulder_vector`:

up_vector * shoulder_vector = cross([0,1,0] * [1,0,0])
= [0,0,-1]

This is perpendicular to "up" and "left" which can either be "forward" or "backward". The third axis that completes the right-hand coordinate system comes out at -Z. This becomes the front of the body.

**why -Z?**

The default SMPL was designed so a viewer standing at `Z = -inf` i.e some arbitrary point along the -Z axis, looking towards +Z (think, down along the Z-axis) sees the person's face and chest. The chest faces the viewer, so it points in the -Z direction (toward the viewer). Therefore, the subject's back must be facing +Z.

In order to guard against an undeterminable direction, normalise the body_front_vector. If the value is incredibly small e.g. <1e-6 then we can't determine the orientation.

Else, body front is divided by the norm to get a normalised output.

---

**Now know which direction the body is facing, how do we determine the camera position?**

First, lets get the subject-relative camera position - COLMAP extrinsic convention transforms a world point into camera coordinates:

`p_cam = R @ p_world +t`

The camera origin (C) is the world-point that provides the camera centre i.e p_cam = 0:

orthonormal - unit length and perpendicular to eachother.

`t` encodes where the world origin sits in camera space. To get the camera centre in world space, you invert it.

```
0 = R @ C + t  # t in COLMAP is "where is the world origin expressed in cam coords?"
R @ C = -t  # take the inverse of R both sides to remove R from the lef
R^-1 @ R @ C = R^-1 @ (-t) # * -1 "Where is the camera origin, expressed in world coords?"
# R is a rotation matrix and therefore orthogonal (.T for transpose, dot product of every column pair)
R.T @ R @ C = R.T @ (-t)
# R.T @ R = I (identity matrix) think of this as 1 i.e I @ thing = that thing.
I @ C = R.T @ (-t)
C = R.T @ (-t)  # flip the sign on one-side gives the same result
C = -R.T @ t  # a cleaner way to express it
```

**Then, how do we know whether it's behind or in front of the body?**

The dot product can be used to determine the angle between two vectors.

if the dot product is between >0->1 they're pointing in the same direction

(think of RAG cosine similarity, bigger number means more closely related)

if the dot product is between -1 -> 0<, they're pointing in the opposite direction.

Yes - we assume the camera's gaze is always pointing in the right direction.
We're interested in whether the camera is on the front or back side of the body, expressed as a single value. Purely a question of position.

we already have two vectors:

`cam_offset = C - joints[0] <- neck, can also use joints[12] for pelvis, anything for body centre`

`body_front = [0,0,-1]`

If the camera is in front of the body, relative to the `body_front` direction, both are heading towards the -Z side, therefore the dot-product is positive.

If the camera is behing the body, relative to the body_front, they're pointing the opposite way, therefore the dot-product is negative.

Analagous to a person facing north - we want to know if someone is standing in front of them. We don't care, in this case, which way they're looking, we just want to know their position relative to the body.

`dot(cam_offset, body_front)`


### A2. Reframe metrics

In `_compute_metrics` at [pipeline.py:485](scantosmpl/fitting/pipeline.py#L485):
- Keep `mean_reproj_px` as-is for backward compat
- Add `median_reproj_px` — robust central tendency
- Add `mean_reproj_inliers_px` — mean after excluding per-view outliers (>3× MAD)
- Add `n_outlier_views` — count of excluded views
- Add `mean_reproj_frontal_px` — mean computed on frontal cameras only

### A3. Update test thresholds

In [test_phase5_integration.py](tests/integration/test_phase5_integration.py):
- `TestFrameAlignment.test_frame_alignment_quality`: assert `mean_reproj_frontal_px < 100.0` (was `mean_reproj_px < 15`)
- `TestReprojectionError.test_reprojection_error`: assert `median_reproj_px < 100.0`
- `TestTriangulation.test_triangulation_accuracy`: keep 30mm PA-MPJPE target; we close the 4mm gap via A4

### A4. Optimiser tuning to close PA-MPJPE gap

In `DEFAULT_STAGES` at [optimiser.py:53](scantosmpl/fitting/optimiser.py#L53):
- `full_refinement.n_iterations`: 200 → 400
- Replace per-view reproj loss with **Huber-clipped** version (threshold ≈ 200px) so single rear-view outliers don't dominate
- Drop rear-view cameras from reproj loss (per A1)
- Verify `w_joint=0.1` in full_refinement isn't too low to anchor the fit; bump to 0.3 if PA-MPJPE doesn't drop

### A5. Files to change

- [scantosmpl/fitting/pipeline.py](scantosmpl/fitting/pipeline.py) — rear-view classification, new metrics
- [scantosmpl/fitting/optimiser.py](scantosmpl/fitting/optimiser.py) — Huber threshold, iteration counts, rear-view filtering
- [scantosmpl/fitting/losses.py](scantosmpl/fitting/losses.py) — verify Huber config on `reprojection_loss`
- [tests/integration/test_phase5_integration.py](tests/integration/test_phase5_integration.py) — updated assertions, new metric keys

### A6. Verification

```bash
source .venv/bin/activate
python -m pytest tests/integration/test_phase5_integration.py -v -m gpu --timeout=600
```

Expected after A1–A4:
- PA-MPJPE 33.6mm → ≤28mm ✅
- Mean reproj all-views 479px → ~150px (still inflated by rear views unless dropped)
- Median reproj 80–120px ✅
- Mean reproj frontal-only 50–80px ✅
- All 11 tests pass

Inspect `output/debug/refinement/reprojection_overlay/*.jpg` — green (SMPL projections) should align with orange (ViTPose) on frontal views.

### Option A - Discission and Explanation

The pipeline order for coarse refinement (using COLMAP camera views):

1) Triangulation: given known camera positions and 2D detections across mulitple views, reconstruct 3D joint positions using Direct Linear Transform and RANSAC. This runs once, before the optimiser.
2) Refinement: iterative SMPL parameter optimisation that adjusts betas, theta, translation and scale to minimise a combination of losses against the triangulated 3D joints and the 2D reprojections.
3) Reprojection: project a 3D SMPL joint through a Camera's pose [R|t|K] into 2D, and measure the pixel distance to the observed ViTPose keypoint (on a per-view basis). This is used as a loss signal in the Refinement stage and for quality metrics afterwards.

Currently, the rear views contaminate the reprojection stage.

- Triangulation: is protected from this because RANSAC votes across all views, per-joint. A rear-view with swapped L/R wrist dectections will disagree with the 10+ frontal views on where the right-wrist is. RANSAC marks those observations as outliers and the quality score reflects this (currently, 0.56-0.75). As a result, 3D position is computed from the inlier frontal views only.
- Reprojection: in the loss function, there is no RANSAC. Every view's loss term is summed. A rear-view camera contributes with points swapped putting loss in the wrong direction. The optimiser doesn't care and gets pulled toward a signal that satisfies no view well.

The A1 option above will exclude rear views from the reprojection loss.
Potentially, removing rear views in the Phase 3 consensus *might* help. But the confidence weighted mean already down-weights low-confidence detections. CameraHMR single-view estimates already embed strong priors.
Trying to mirror the joints in rear views for RANSAC adds complexity with marginal gains. We're already capturing all joints in multiple views with good quality.

---

## Spec Supplement — Option A+B (PnP camera refinement)

**When to use:** If Option A's reproj-on-inliers is still above ~100px or PA-MPJPE doesn't close to <30mm.

### Idea
COLMAP cameras are sub-pixel accurate for COLMAP's own features but may have residual error vs ViTPose keypoint locations. After SMPL refinement gives us reliable 3D joints, we can PnP-refine each camera's [R|t] using `cv2.solvePnPRansac` with refined SMPL joints (3D) and undistorted ViTPose keypoints (2D). This absorbs per-camera systematic error into the camera estimate.

### Implementation sketch

1. **New module** [scantosmpl/calibration/pnp_refine.py](scantosmpl/calibration/pnp_refine.py)
   ```python
   def pnp_refine_camera(
       joints_3d_smpl: np.ndarray,    # (N,3) refined SMPL joints
       kps_2d: np.ndarray,            # (N,2) undistorted landscape ViTPose
       confs: np.ndarray,             # (N,) ViTPose confidences
       K: np.ndarray,                 # (3,3) intrinsics
       R_init: np.ndarray, t_init: np.ndarray,  # COLMAP camera as init
       min_conf: float = 0.3,
   ) -> tuple[np.ndarray, np.ndarray, float]:  # R, t, mean_reproj_px
   ```
   - Filter by `confs > min_conf`
   - `cv2.solvePnPRansac` with `useExtrinsicGuess=True`, `flags=cv2.SOLVEPNP_ITERATIVE`
   - Return refined [R|t] + reprojection error
   - Skip refinement if <6 inliers (insufficient PnP constraints)

2. **Wire into pipeline** at [pipeline.py:180](scantosmpl/fitting/pipeline.py#L180), after SMPL refinement, before metrics:
   ```python
   if cfg.pnp_refine_cameras:  # new Phase5Config field
       cameras_smpl = self._pnp_refine_all(refined, kp2d_tensors, confs_tensors, cameras_smpl)
   ```

3. **Config field** in [Phase5Config](scantosmpl/config.py#L118):
   ```python
   pnp_refine_cameras: bool = False  # opt-in
   pnp_refine_max_translation_m: float = 0.1  # safety bound
   ```

### Files
- New: `scantosmpl/calibration/pnp_refine.py`
- Modify: `scantosmpl/fitting/pipeline.py`, `scantosmpl/config.py`
- Test: extend `test_phase5_integration.py` with `TestPnPRefinement::test_pnp_reduces_reproj`

### Verification
- Per-view reprojection after PnP should drop to ViTPose noise floor (~50-80px) on frontal views
- Camera translation drift bounded < 0.1m (sanity check, no overfitting)
- PA-MPJPE unchanged or slightly better

---

## Spec Supplement — Option C (Frontal 8 views + Phase 4 self-cal)

**When to use:** Fallback if COLMAP availability becomes an issue or if rear-view contamination proves intractable.

### Idea
Drop the rear-view cameras entirely. Use Phase 4 self-calibration (already implemented) on the frontal subset only. Triangulate from frontal views, refine SMPL.

### Implementation sketch

1. **View filter** — new helper in [scantosmpl/calibration/](scantosmpl/calibration/):
   ```python
   def select_frontal_views(
       views: list[ViewResult],
       consensus: ConsensusResult,
       cosine_threshold: float = 0.0,  # camera-to-body forward dot product > 0 = frontal
   ) -> list[ViewResult]:
       """Return views where the camera faces the subject's front."""
   ```
   For 17-view scanner this typically selects 8-10 views.

2. **Re-run Phase 4** on filtered views to get fresh self-cal cameras. No code changes — just pass filtered `views` list.

3. **Phase 5 in self-cal mode** — already supported. Set `Phase5Config.extrinsics_source = "self_calibration"` and pass the new `CalibrationResult`.

### Trade-offs
- Pros: No COLMAP dependency; cleaner data (no rear-view contamination)
- Cons: 8 views vs 17 — reduced triangulation baseline diversity; self-cal already has ~52px reproj (worse than COLMAP+EXIF-fix which achieves ~50-80px on frontal); ambiguities around shoulders/hips that COLMAP resolves with side views

### Files
- New: `scantosmpl/calibration/view_selection.py`
- Modify: pipeline orchestration to filter before Phase 4
- Test: new `TestFrontalSelection` in integration tests

### Verification
- Re-run pipeline with `--frontal-only` flag
- Compare metrics vs current Option A result
- Decision criterion: only adopt if Option C reproj < Option A reproj by >20%

---

## Files index (for primary Option A work)

- [scantosmpl/fitting/pipeline.py](scantosmpl/fitting/pipeline.py) — `_compute_metrics` (L485), new `_classify_rear_views`
- [scantosmpl/fitting/optimiser.py](scantosmpl/fitting/optimiser.py) — `DEFAULT_STAGES` (L53), rear-view filtering in `refine()`
- [scantosmpl/fitting/losses.py](scantosmpl/fitting/losses.py) — Huber config on `reprojection_loss`
- [tests/integration/test_phase5_integration.py](tests/integration/test_phase5_integration.py) — assertion targets
