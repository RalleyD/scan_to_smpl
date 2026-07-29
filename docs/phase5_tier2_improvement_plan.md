# Phase 5 / Tier 2 — Refinement Improvement Plan

> **How to use this doc:** self-contained plan for a fresh session. Read the Diagnosis,
> then execute workstreams **W1 → W4 in order** (each is independently shippable). Torso
> girth is intentionally deferred to Tier 3 — do not "fix" it here. Verify per the final
> section after each workstream.

## Context

Tier 1→2 comparison overlays (side + front views) show Tier 2 markedly improves waist,
legs, feet, and arm alignment, but leaves two visible defects:
- **Lower-abdomen / torso overspill** — mesh wider than the subject.
- **Head protrudes forward** — head offset in the front view.

Two theories were raised, both centred on the high *mean* reprojection (245px):
1. Noisy side views push reprojection gradients the wrong way → exclude/bound side views.
2. Search over front/side image combinations to minimise mean reprojection.

Code tracing reshapes the problem: the two visual defects and the reprojection number have
**largely separate root causes**, and the "high mean" is partly a reporting artifact.

## Diagnosis (from code)

- **Mean vs median.** `_compute_metrics` (`scantosmpl/fitting/pipeline.py:369-372`) pools all
  joint×view errors: median **77px**, mean **245px** — a right-skewed tail (side/rear detector
  failures, left/right swaps, occluded far limbs). `summary.txt` prints only the inflated mean.
- **Torso overspill = no surface constraint.** Losses constrain *joint centres only*
  (12 COCO joints + pelvis/neck midpoints). `w_shape_reg=0.01` (near-zero). Girth is
  unconstrained → the principled fix is Tier 3 (chamfer vs point cloud), not yet built.
- **Head protrusion = no head data term.** `COCO_TO_SMPL` (`scantosmpl/smpl/joint_map.py:34-47`)
  excludes nose/eyes/ears (COCO 0-4). SMPL head joint (15) has zero data constraint; orientation
  inherited from consensus, nudged only by the 0.01 pose prior.
- **Theory 1 partly pre-addressed.** Reprojection loss already Huber-bounds each term at
  `huber_delta=20px` (`scantosmpl/fitting/losses.py:47`). Per-term magnitude is capped. But
  `classify_rear_views` (`scantosmpl/fitting/rear_views.py`) is binary — profile views
  (dot ≈ 0) are kept and carry systematic ViTPose error. Room exists as *graded weighting*,
  not a fresh bound.
- **Theory 2 risk.** Literal "minimise mean reproj over view subsets" = the circularity trap
  the master spec flagged (data/cameras absorb error to flatter the metric). Use the
  leave-one-out diagnostic (W4) instead.
- **Config disconnect (minor).** `Phase5Config.w_reprojection=0.5` / `w_shape_reg=0.01` in
  `scantosmpl/config.py` are unused; `DEFAULT_STAGES` in `optimiser.py` hardcode the real
  weights. Wire config through if W3 adds tunable weights.

## Workstreams (execute in order)

### W1 — Fix the metric & reporting (cheap, do first)
- Print `median_reproj_px` and `mean_reproj_inliers_px` in `summary.txt`
  (`_save_debug`, `scantosmpl/fitting/pipeline.py:~477`). Currently only the skewed mean shows.
- Move the acceptance criterion off raw mean onto **median** (or inlier-mean).
- Rationale: separate "the fit is bad" from "the metric is skewed" before any tuning.

### W2 — Head correspondence (targets head protrusion, low risk)
- Add head/face reprojection terms so head orientation is data-driven.
- **Geometry care:** nose(0)→head(15) would *pull the head forward* (nose is anterior). Use
  ears midpoint (3,4)→head(15), near the head joint laterally; **validate the mapping against
  a frontal view before trusting it.**
- Touch points: `COCO_TO_SMPL` / a new head-correspondence set + `reprojection_loss`.

### W3 — Graded view-angle weighting (Theory 1, done properly)
- Extend `classify_rear_views` → graded classifier: frontal / three-quarter / profile / rear.
- Down-weight (not hard-exclude) profile views in `reprojection_loss` via a per-view weight;
  keep rear excluded. Expose weights in `Phase5Config`.
- Optional: per-term gross-outlier rejection (drop > N px as detector failures) so left/right
  swaps never enter the sum.

### W4 — Leave-one-view-out diagnostic (honest form of Theory 2)
- For each view: refit without it, measure how well the fit predicts the held-out view.
- Distinguishes *genuinely unexplainable* views (true outliers to drop) from merely *hard*
  views. Output feeds W3's weight rule — **not** a search for the metric-minimising subset.
- Standalone diagnostic script under `output/debug/`; no pipeline change needed to run it.
- Document the circularity risk in the output so results aren't over-claimed.

### Deferred — Torso girth (Tier-3 gap, do NOT patch here)
- Overspill is unconstrained surface girth; the principled fix is Tier 3 chamfer-vs-point-cloud.
- No interim shape-reg/beta-cap hacks (blunt, trade genuine shape fit). Record as a known
  limitation so the mentor note frames it as a Tier-3 gap, not a Tier-2 bug.

## Verification
- Re-run `pytest tests/integration/test_phase5_integration.py -v -m gpu` after each W.
- Regenerate overlays:
  `python -m scantosmpl.evaluation.visualise --views cam01_2.JPG,cam02_5.JPG,cam03_6.JPG`
  and eyeball head + torso before/after.
- Track **median + inlier-mean** reproj (not raw mean) across changes.
- Lint/type: `ruff check`, `ruff format --check`, `mypy scantosmpl`.

## Out of scope / risks
- Tier 3 not built here (girth root fix lives there).
- Theory-2-as-search deliberately avoided; W4 is the non-circular substitute.
- W2 head mapping must be geometrically validated, not assumed.
