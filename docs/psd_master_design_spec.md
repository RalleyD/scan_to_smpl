# Master Design Spec — Pose Space Deformation (PSD) Modelling

**Status**: Draft (research + design; no code)
**Slug**: `psd-modelling`
**Owner**: Dan
**Date**: 2026-07-22
**Tiers touched**: New Tier 4 (post-Tier-3), consumes Tier 1→2 outputs; Stage B couples to Tier 3.

> **How to use this doc:** This is the standalone research + design spec for the project's
> headline deliverable — turning the registered SMPL fits into **animator-friendly pose-space
> blend shapes**. It is deliberately *independent of Tier 3* for its MVP (Stage A) and folds
> Tier 3 back in as a benchmark (Stage B). When we're ready to build, this becomes the input to
> the `/feature-spec` workflow (→ `docs/features/psd-modelling/`).

---

## 0. TL;DR

We already register SMPL to a subject's captured poses (Tier 1→2). The animator deliverable is
**Pose Space Deformation** (Lewis et al., SIGGRAPH 2000): a smooth, sculpted-quality mapping from
*pose* to *per-vertex corrective displacement*, exported as Maya blend shapes.

The single most important design decision — and the thing that separates a real contribution from
a circular re-derivation of SMPL — is **what the model predicts**. We predict the **corrective
residual `δ` in each vertex's local (unposed) frame**, exactly as the PSD paper defines it, *not*
the final posed vertex. SMPL's LBS + built-in pose blendshapes already handle the rigid/generic
part of posing; the residual is the subject-specific soft-tissue behaviour that has real value.

We build it in **two stages**:

- **Stage A — MVP, pre-Tier-3.** Ground truth `δ` comes from the *registered SMPL fits themselves*
  (residual of the fit vs. the canonically-posed template). Delivers a complete, shippable
  image→SMPL→PSD→Maya pipeline. Non-circular because the residual is defined relative to LBS, which
  the net does **not** re-learn.
- **Stage B — benchmark, post-Tier-3.** Ground truth `δ` comes from Tier 3 point-cloud surface
  fits (SMPL+D). Retrain, then **benchmark Stage A against Stage B** to quantify how much real
  geometry the MVP was missing.

---

## 1. Problem

The project's end users are 3D animators who need **blend shapes**: a rig control that smoothly
interpolates a body's surface as it moves, sculpt-quality, on consistent SMPL topology (6890 verts).
Today we produce a *single static SMPL fit per pose* — discrete, un-interpolatable, and carrying no
subject-specific deformation beyond what generic SMPL posing gives. We need a learned model that
maps **pose → smooth corrective surface deformation**, exportable to Maya, trained on the multiple
poses of one subject that the rig captures.

---

## 2. Decisions

Locked choices from the clarifying-question round, with rationale.

- **D1 (Learning target = per-vertex residual, two-stage).** The model predicts the PSD residual
  `δ` (per-vertex, in the unposed local frame), **not** the full posed vertex. Two stages:
  Stage A ground truth = residual of registered SMPL fits; Stage B ground truth = Tier 3
  point-cloud SMPL+D. Rationale: predicting full vertices re-learns SMPL's own LBS/pose-blendshape
  function (low novelty, near-circular when the GT is itself a SMPL fit). The residual is the
  genuinely new signal and is *definitionally* what PSD interpolates (paper §4). Two stages let us
  ship an MVP now and quantify its gap against real geometry later.
- **D2 (Plan for a multi-pose corpus).** More poses of the same subject are coming, captured on the
  same rig and run through the existing Tier 1→2 pipeline. The spec assumes a corpus of `N` registered
  poses per subject; it does **not** assume point clouds for Stage A.
- **D3 (Stage A independent of Tier 3).** Stage A trains and validates on registered SMPL meshes +
  pose params alone. Point-cloud validation is optional/monitoring-only until Stage B. This decouples
  the MVP from the currently-empty Tier 3 package and its dependencies (Kaolin, Open3D ICP).
- **D4 (Deliverable = this research + master design doc).** No formal brief set or code this session.

---

## 3. Scope

**In scope**
- Formal definition of the PSD learning target (residual `δ`) and the pose→δ mapping, grounded in
  the Lewis 2000 formulation and adapted to SMPL topology.
- Two-stage architecture (A: MVP on SMPL-fit residuals; B: benchmark on Tier-3 SMPL+D).
- Data contract: what a "pose sample" is, how the corpus is assembled from existing pipeline outputs,
  train/test split policy.
- Model architecture options (classical RBF-PSD vs. learned network) with a recommendation.
- Loss design, evaluation protocol (three-way comparison adapted to *our* GT reality), and the Maya
  export contract.
- The precise anti-circularity argument and how each stage's validation is (or isn't) independent.

**Out of scope** (one-line reason each)
- Tier 3 itself — its own spec; Stage B *consumes* it but does not build it.
- Multi-subject generalisation / transfer learning — single subject first (reason: no multi-subject data).
- Real-time / game-engine inference optimisation — offline blend-shape bake is the deliverable (reason: Maya target).
- Capturing new poses — a data-collection task, not a modelling task (reason: separate workstream).
- Changing Tier 1→2 — PSD is strictly downstream; it reads their artefacts read-only.

---

## 4. Approach

### 4.1 Where PSD sits in the architecture

```
Tier 1  per-view HMR + consensus     ─┐
Tier 2  self-cal + refinement (SMPL)  ├─ run once PER POSE  →  registered SMPL fit p_i = (β, θ_i, ...)
                                      ─┘                          (β shared across poses of one subject)
        (repeat for all N poses of the subject)
                         │
                         ▼
              ┌──────────────────────────┐
   Stage A →  │  PSD corpus (SMPL-fit)   │   δ_i = residual of fit vs. LBS-posed template
              └──────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  PSD model  f: θ → δ      │   trained per-vertex, RBF or net
              └──────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  Maya blend-shape export  │   neutral mesh + per-pose δ targets
              └──────────────────────────┘

   Tier 3 (later): point cloud → SMPL+D surface fit  ──► Stage B corpus (δ from real geometry)
                                                          └─ retrain + BENCHMARK vs Stage A
```

### 4.2 The residual, precisely (this is the crux)

The PSD paper defines the deforming surface as `p + δ`, where `p` is the vertex **moved rigidly by
the skeleton** (in SMPL terms: after LBS *and* SMPL's own pose blendshapes), and
`δ = f_interp(pose)` is the corrective displacement **expressed in the vertex's local coordinate
frame** so that pure articulated motion produces `δ = 0`.

Mapping that onto SMPL for **Stage A**:

1. For pose `i`, we have the registered fit `M_fit,i = SMPL(β, θ_i, t_i, s_i)` — 6890 verts.
2. Compute the **generic posed baseline** `M_base,i = SMPL(β, θ_i)` in the *same* canonical frame
   (this is what LBS + SMPL pose-blendshapes predict from `θ_i` alone — no residual).
3. The Stage-A residual is `δ_world,i = M_fit,i − M_base,i`, then **rotated into each vertex's local
   frame** using the per-vertex LBS rotation `R_v(θ_i)` so that `δ_i = R_v(θ_i)^{-1} δ_world,i`.
   Storing `δ` in the local frame is what makes it pose-*invariant* to rigid articulation and is the
   entire reason PSD interpolates cleanly (paper §4).

> **Honest caveat baked into the design:** In Stage A, `M_fit,i` *is* a SMPL mesh, so
> `δ_i = M_fit,i − SMPL(β,θ_i)` is small — it captures only the difference between the *fitted*
> params and the *pose-only* baseline (e.g. shape/scale/translation coupling, optimiser residuals),
> **not** real off-manifold soft tissue. Stage A therefore proves the *machinery* (frames, interp,
> export, smooth animation) and delivers a working rig, but its `δ` is geometrically thin. **This is
> expected and is exactly why Stage B exists.** Stage B's `δ = scan_surface(θ) − SMPL(β,θ)` is
> off-manifold and carries the real anatomy. The MVP's honesty depends on *reporting* this, not
> hiding it behind good-looking interpolation.

For **Stage B**, step 1 is replaced by the Tier-3 SMPL+D fit (mesh with per-vertex displacement
against the point cloud); everything else — local-frame rotation, interpolation, export — is
identical. The whole pipeline is designed so **swapping the GT source is the only change between
stages.**

### 4.3 Pose-space parameterisation

The interpolation domain (the "pose space") is a subset of `θ`. Two design points:

- **Axis-angle `θ` (69-D body pose) is a poor interpolation coordinate** — it's not Euclidean and
  large rotations alias. Follow standard practice: drive the PSD interpolation from a
  **rotation-invariant, locally-Euclidean encoding of the joints that actually move**, e.g. per-joint
  rotation matrices flattened, or (better) only the DoFs of the joints near each vertex (paper §4.1:
  "self-relative configuration", pose space dimension can vary per vertex).
- Start **1-D per active joint** where possible (elbow, knee) as the paper recommends, and only grow
  the pose-space dimension where the data demands it. This keeps Stage A tractable with a small `N`.

**Recommended default encoding: `R(θ) − I`.** SMPL's own pose blendshapes *are* a linear PSD baked
into the model, and the pose feature they use is **per-joint rotation matrices minus identity**
(`R(θ) − I`, 9 numbers/joint). Reuse it: it is rotation-matrix-based (locally Euclidean, no
axis-angle wraparound), it is exactly zero at the rest pose, and it is already battle-tested on this
exact topology. It sidesteps the axis-angle problem cleanly and is the natural starting point before
any per-vertex dimensionality reduction.

**The encoding *is* the animator's control interface.** The pose-space coordinates become the rig
control the animator drives. Encode pose as "elbow bend angle" → the animator gets a 1-D elbow-bend
slider; encode a whole-body or abstract space → they get abstract sliders (cf. the paper's
happy↔sad emotion axis, Figs 6–7). So this choice is not purely internal — it should be decided
*with* the animator once real non-T poses exist to show what actually varies. This is why the exact
encoding is left as open question #1.

---

### 4.4 Sequencing / critical path

A meaningful PSD *result* (not just a working demo) is gated on an **independent geometry source**,
not on PSD code. The blocker for good numbers is upstream of the model. Build order:

1. **Capture more poses of the subject** — the true critical path. PSD needs `N ≥ 2` to interpolate
   at all and `N ≈ 7–10` to be useful. Today `N = 1` (t-pose only). No amount of PSD work substitutes
   for this.
2. **Tier 3 (point cloud → SMPL+D)** — the groundwork that makes the residual `δ` *geometrically real*.
   Prioritise this over polishing Stage A, because Stage A's `δ` is thin by construction (§4.2) and
   only Tier 3 supplies the off-manifold soft-tissue signal that gives PSD a defensible result.
3. **Stage A becomes a cheap baseline to benchmark against**, not the end product.

Practical implication: **build Stage A's *code* early** (it is the same code Stage B runs — only the
GT source swaps), but **do not invest in making Stage A's *numbers* look good.** Spend that effort on
poses (1) and Tier 3 (2). Stage A's job is to prove the machinery — frames, interpolation, export,
smooth animation — and to be the thing Stage B's real geometry is measured against.

## 5. Contract (seam definitions — provisional, firmed up at /feature-spec time)

### 5.1 New dataclasses (`scantosmpl/types.py`)

```python
@dataclass
class PoseSample:
    """One registered pose of a subject — the atomic unit of the PSD corpus."""
    pose_name: str                 # e.g. "t-pose", "arm-raised"
    betas: np.ndarray              # (10,) — SHARED across a subject's samples (assert consistency)
    body_pose: np.ndarray          # (69,) axis-angle, 23 joints
    global_orient: np.ndarray      # (3,)
    fit_vertices: np.ndarray       # (6890, 3) — Tier 2 registered mesh (Stage A GT source)
    scan_vertices: np.ndarray | None = None   # (6890, 3) — Tier 3 SMPL+D (Stage B GT source)
    quality: dict[str, float] = field(default_factory=dict)  # pa_mpjpe, median_reproj_px, ...

@dataclass
class PSDCorpus:
    """A subject's full set of pose samples + the shared neutral (reference) mesh."""
    subject_id: str
    reference_pose: str            # which sample is the neutral/rest pose
    samples: list[PoseSample]
    train_names: list[str]
    test_names: list[str]

@dataclass
class BlendShapeSet:
    """Export payload: neutral mesh + per-pose corrective targets on SMPL topology."""
    reference_vertices: np.ndarray # (6890, 3)
    faces: np.ndarray              # (13776, 3)
    targets: dict[str, np.ndarray] # pose_name -> (6890, 3) DELTA vs reference (not absolute)
    stage: str                     # "A" | "B"
    metadata: dict
```

### 5.2 Key function signatures (provisional)

```python
def build_corpus(fit_dir: Path, *, reference_pose: str, stage: str) -> PSDCorpus:
    """Assemble a PSDCorpus from per-pose Tier 2 (stage A) or Tier 3 (stage B) outputs.
    Asserts shared betas across samples; raises on topology/param mismatch."""

def compute_local_residual(sample: PoseSample, smpl: SMPLModel, *, stage: str) -> np.ndarray:
    """δ in each vertex's local (unposed) frame, (6890, 3). Stage A: fit−base; Stage B: scan−base."""

def fit_psd(corpus: PSDCorpus, *, method: Literal["rbf", "net"], config: PSDConfig) -> PSDModel:
    """Train the pose→δ interpolant. RBF = closed-form per-vertex; net = shared MLP."""

def export_blendshapes(model: PSDModel, corpus: PSDCorpus, out: Path) -> BlendShapeSet:
    """Bake δ at each training/test pose into Maya-ready delta targets + import script."""
```

### 5.3 On-disk artefacts

- `output/psd/<subject>/corpus.json` — sample manifest + train/test split + quality per pose.
- `output/psd/<subject>/model_stage{A,B}.pt` — the trained interpolant.
- `output/psd/<subject>/blendshapes_stage{A,B}.npz` — `reference_vertices`, `faces`, `targets/*`.
- `output/psd/<subject>/import_blendshapes.py` — Maya Python import (not MEL; see §9 note).
- `output/psd/<subject>/eval_stage{A,B}.json` — metrics (§ evaluation).
- `output/psd/<subject>/benchmark_A_vs_B.json` — Stage B only: per-pose A-vs-B deltas.

---

## 6. User flows

```bash
# 1. Build the corpus from existing per-pose Tier 2 fits (Stage A, no point cloud needed)
scantosmpl psd build-corpus --fit-dir output/fits/<subject>/ \
    --reference-pose t-pose --stage A --output output/psd/<subject>/

# 2. Train + export the MVP blend shapes
scantosmpl psd fit --corpus output/psd/<subject>/corpus.json --method rbf \
    --output output/psd/<subject>/

# 3. (Later) once Tier 3 exists, rebuild with scan GT and benchmark
scantosmpl psd build-corpus --fit-dir output/fits/<subject>/ --stage B ...
scantosmpl psd fit   --stage B ...
scantosmpl psd benchmark --stage-a ...stageA.pt --stage-b ...stageB.pt
```

Artefacts an animator receives: `blendshapes_stageA.npz` + `import_blendshapes.py` → load in Maya,
drive the pose-space control, get smooth interpolation on 6890-vertex topology.

---

## 7. Data model & the corpus

- **Atomic unit** = one `PoseSample` = one registered pose (the existing pipeline's output for one
  capture session of the subject in one pose).
- **Shared shape invariant:** all samples of one subject must share `β`. The corpus builder asserts
  this; if per-pose fits drifted `β`, we re-fit with `β` locked from the reference pose (a Tier-2
  option, flagged as a prerequisite — see Risks).
- **Corpus size reality:** today `N = 1` (t-pose only, 17 views, no point cloud). The spec is written
  for `N ≈ 7–10`. **With small `N`, classical RBF-PSD is strongly preferred over a neural net**
  (a net will overfit `<10` samples; RBF is exact-interpolating and needs no held-out training) —
  see §8.
- **Split policy:** reference pose always in train; hold out 1–3 *interpolation-testing* poses
  (poses geometrically *between* training poses, not extrapolations) so the test measures the thing
  PSD is actually for — smooth in-between synthesis.

---

## 8. Model choice — RBF-PSD vs. learned network

| | Classical RBF-PSD (paper) | Learned network (attached plan) |
|---|---|---|
| Fits `N<10` poses | ✅ exact interpolation, closed form | ❌ overfits, needs val split we can't spare |
| Matches paper semantics | ✅ *is* the paper | ~ (an approximation of it) |
| Per-vertex pose-space dim varies | ✅ native (paper §4.1) | ❌ awkward |
| Animator-tunable falloff (σ) | ✅ designed for it | ❌ opaque |
| Scales to large multi-subject corpora | ❌ per-vertex solves | ✅ amortised |
| Smoothness guarantee | ✅ Gaussian RBF | needs Laplacian/edge reg to approximate |

**Recommendation:** **Start with RBF-PSD** (Gaussian basis, per-vertex least-squares as in paper §3.2
`w = (ΦᵀΦ)⁻¹Φᵀd`). It is the correct tool for a single subject with a handful of poses, it *is* the
technique the paper describes, and it gives the animator the falloff control the paper emphasises.
Keep the learned-network path (the attached plan's `PSDDeformationNetwork`) as a documented Stage-B+
upgrade for when a multi-subject corpus exists and amortisation pays off. The `fit_psd(method=...)`
seam makes them interchangeable.

---

## 9. Evaluation protocol

Adapted from the attached plan's "three-way comparison" but corrected for **our** ground-truth reality.

**Stage A** (no independent geometry — be honest about it):
- **Reconstruction on train poses:** RBF is exact → ≈0; report as a *sanity check only*, not a result.
- **Interpolation on held-out poses:** predict `δ` at a held-out pose's `θ`, compare to that pose's
  *own* registered `δ`. Metric: mean per-vertex error (mm). This is the real Stage-A number.
- **Smoothness:** second-derivative (acceleration) of the vertex path across an interpolation sweep
  (paper's concern about piecewise-linear SI). Lower = smoother.
- ⚠️ **No point-cloud metric in Stage A** — the attached plan's "PSD Chamfer vs point cloud" is a
  **Stage B** metric. Reporting it in Stage A would be measuring against geometry the model never saw
  and can't have learned; excluded by design.

**Stage B** (point cloud is genuine independent geometry):
- **Chamfer vs point cloud** for both the Stage-A and Stage-B predictions at held-out poses → the
  headline benchmark: *how much closer to real geometry does scan-supervised δ get us?*
- **Per-vertex δ magnitude comparison A vs B** → quantifies how thin the MVP residual was (validates
  the §4.2 caveat with a number).

**The circularity guard (carried from the codebase's existing rigour):** Stage A's validation is
*internal consistency* only (as with the self-cal reprojection metric — see `notes.md` Phase 5). It
cannot certify real-world accuracy. Stage B, using point clouds that were **never** in any fitting or
PSD loop, is the first genuinely independent check — mirroring the note in `notes.md` that Tier 3 ICP
is "real, independent validation in a way nothing in the current pipeline provides." The eval report
must state this explicitly so Stage-A numbers are never over-claimed.

---

## 10. Maya export contract

- **Topology:** SMPL 6890 verts / 13776 faces, fixed ordering — the whole reason we work in SMPL space.
- **Format:** `.npz` (`reference_vertices`, `faces`, `targets/<pose>` as **deltas** vs reference) +
  a **Python** import script. *Note:* the attached plan's MEL script wraps Python via `python(...)`
  string-building — fragile and hard to debug. Prefer a native `maya.cmds` Python script; keep MEL
  only as a thin `source`-able launcher if an animator workflow needs it.
- **Delta convention:** targets stored as `posed_target − reference` so Maya's blendShape node
  (which is additive over the base) consumes them directly.
- **Validation on export:** assert 6890 verts, no NaNs/Infs, faces unchanged vs template.

---

## 11. Non-goals

- Not building Tier 3 (Stage B consumes it).
- Not predicting full posed vertices (only the residual δ).
- Not multi-subject / cross-character generalisation.
- Not real-time inference.
- Not modifying Tier 1→2 fitting (read-only downstream), **except** the possible `β`-lock re-fit
  prerequisite (§7), which is a small Tier-2 flag, not a redesign.

## 12. Acceptance criteria (for the eventual build — informational here)

- **AC1** — `build_corpus` assembles the t-pose (and any further poses) into a `PSDCorpus` with shared
  `β` asserted. **Evidence:** `corpus.json` exists, `betas` identical across samples within tol.
- **AC2** — RBF-PSD reproduces every *training* pose's `δ` to <1 mm mean per-vertex (exactness check).
  **Evidence:** `eval_stageA.json.recon_mm < 1.0`.
- **AC3** — On a held-out interpolation pose, Stage-A mean per-vertex `δ` error is reported (no
  threshold gate at MVP — it's a baseline number). **Evidence:** `eval_stageA.json.interp_mm` present.
- **AC4** — Blend-shape export loads in Maya without error and animates smoothly. **Evidence:**
  `blendshapes_stageA.npz` passes the export validator; manual Maya import checklist.
- **AC5** (Stage B) — Benchmark report quantifies Stage-B Chamfer improvement over Stage-A at
  held-out poses. **Evidence:** `benchmark_A_vs_B.json` with per-pose Chamfer deltas.

## 13. Risks (likelihood × impact)

- **R1 — Only one pose exists today.** PSD needs `N≥2` to interpolate anything; `N≈7–10` to be useful.
  *Mitigation:* Stage A ships as scaffolding + single-pose export now; real training gated on capture.
  Flag the capture workstream as the true critical path.
- **R2 — Per-pose `β` drift.** `β` (identity/shape) is physically constant across a subject's poses —
  bone lengths don't change when they raise an arm. But fits are optimised independently per pose, so
  *estimation* drift is expected: per-pose noise plus SMPL's shape/pose ambiguity (the optimiser can
  "explain" a pose effect by nudging `β`). Any such drift contaminates the residual with shape
  masquerading as pose deformation (`δ` would blend body *builds*, not just *poses*).
  **Recommended (not optional):** fit the reference pose first, freeze its `β`, and re-fit every other
  pose with `β` non-trainable (optimise only `θ`, translation, scale) — a small Tier-2 optimiser flag.
  `build_corpus` asserts shared `β` across samples regardless.
- **R3 — Stage-A residual is geometrically thin (the §4.2 caveat).** Risk of over-claiming a
  good-looking-but-empty MVP. *Mitigation:* eval report states it in words *and* Stage B quantifies it.
- **R4 — Axis-angle pose space aliases.** Naïve `θ` interpolation produces artefacts.
  *Mitigation:* rotation-matrix / per-active-joint pose-space encoding (§4.3); start 1-D per joint.
- **R5 — Local-frame rotation bug.** Getting `R_v(θ)` wrong silently produces plausible-but-wrong δ.
  *Mitigation:* unit test — a pure articulation (no soft tissue) must yield `δ ≈ 0` after local-frame
  transform; this is the definitional PSD invariant and a cheap, decisive check.

---

## 14. Open questions for /feature-spec time

1. Exact pose-space encoding (flattened rotation matrices vs. per-joint self-relative DoFs) — pick
   after seeing 2–3 real non-T poses.
2. Not *whether* to `β`-lock (R2 recommends it) but *where it lives*: a PSD-spec prerequisite step
   vs. a standalone Tier-2 ticket that PSD depends on.
3. Reference-pose choice: is the T-pose the right neutral, or do we want an A-pose rest for animators?
4. Confidence-weighting `δ` by per-pose fit quality (the corpus already carries `pa_mpjpe`,
   `median_reproj_px`) — cheap to add to the RBF least-squares as row weights.
```
