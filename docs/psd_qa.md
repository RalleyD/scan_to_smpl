# PSD Modelling — Q&A Clarifications

> Companion to [`psd_master_design_spec.md`](psd_master_design_spec.md). Captures the clarifying
> questions raised during design review and the answers, verbatim.
>
> **Status: all follow-on actions folded into the master spec** (§4.3 encoding, §4.4 sequencing,
> R2 / open-Q2 β-lock). Per-question tags below and the "Changes folded in" section at the end
> record where each went. This doc is now the historical record of the review.

---

## 1. "Each vertex's local (unposed) frame" — what it means for the PSD residual

> **Folded in:** underpins spec §4.2 (residual definition) and R5 (the `δ≈0` unit test). Reference material — no new spec edit needed.

Think about what happens to a vertex on your forearm when you bend your elbow.

**In world/global coordinates**, that vertex moves a *lot* — it swings through an arc. If you stored the correction `δ` in world coordinates, `δ` would be dominated by that swing, and it would be totally different at every elbow angle. Useless for interpolation.

**The insight of PSD** (and of SMPL's own math): most of that motion is just *rigid articulation* — the bone rotated, so the skin rigidly rotated with it. That part is already handled by LBS (Linear Blend Skinning). What you actually want to model is only the *extra* deformation on top of the rigid motion — the muscle bulging, the skin creasing.

So you "undo" the rigid rotation before storing the correction:

```
δ_world = fit_vertex − posed_baseline_vertex        # the raw correction, in world space
δ_local = R_v(θ)⁻¹ · δ_world                          # rotate it back into the vertex's rest orientation
```

`R_v(θ)` is the rotation LBS applied to that vertex at pose `θ` (a blend of the nearby bones' rotations). Undoing it puts the correction back into the frame the vertex has when the body is in its **rest pose** — hence "local, unposed frame."

**Why this matters concretely:** if a vertex undergoes *pure* rigid articulation (no soft-tissue effect at all), then `fit_vertex == posed_baseline_vertex`, so `δ_world = 0` and `δ_local = 0`. The residual is zero exactly when nothing interesting is happening. That's the property that makes `δ` **smooth and interpolatable across poses** — and it's also the decisive unit test I flagged as R5: feed in a pure articulation, and `δ_local` must come out ≈0.

The paper states this directly (page 5): *"we instead interpolate the desired deviation of a surface vertex (expressed in the local frame) from its initially computed position ... rigid body articulated motion results in zero δ."*

---

## 2. If Stage A can't quantify `fit` performance, should Tier 3 become the priority?

> **Folded in:** → new spec **§4.4 Sequencing / critical path** (more poses → Tier 3 → PSD-with-real-numbers; Stage A code early, its metrics not the goal).

This is the sharpest question here, and I think the honest answer is: **partly yes — but reframe it.** Two distinct things are getting bundled:

**(a) Can Stage A quantify the *machinery*?** Yes. Stage A can fully measure interpolation error, smoothness, and export correctness — because it has held-out registered poses to interpolate toward. What it *cannot* do is claim that error means anything about *real geometry*, because both prediction and target live on the SMPL manifold.

**(b) Is Stage A's `δ` worth shipping on its own?** Geometrically, it's thin (the §4.2 caveat). So Stage A is real as a **pipeline/plumbing proof and animator-workflow deliverable**, but weak as a **geometric result**.

So the reframe: **the blocker for a *meaningful* PSD result is not Tier 3 specifically — it's an independent geometry source.** Tier 3 (point cloud → SMPL+D) is the one you've already architected, so yes, it's the natural priority *if the goal is a defensible result rather than a demo*. But note the true critical path is even earlier: **you need multiple captured poses first** (you have `N=1`). Tier 3 on a single T-pose gives you a better single mesh but still nothing to interpolate.

My recommendation on sequencing:

1. **Capture more poses** (critical path — nothing PSD works without `N≥2`).
2. **Tier 3** to turn those into geometrically-real SMPL+D fits — *this is the groundwork that makes PSD's residual meaningful*, so yes, prioritise it over polishing Stage A.
3. Stage A then becomes a **cheap baseline you benchmark against**, not the end product.

In other words: build Stage A's *code* now (it's the same code Stage B uses), but don't invest in making Stage A's *numbers* look good — invest that effort in Tier 3.

I can add a short "Sequencing / critical path" section to the spec capturing this if you'd like.

---

## 3. Pose-space encoding — what it is, and does it depend on the artist deliverable?

> **Folded in:** → spec **§4.3** (`R(θ)−I` as recommended default encoding + "encoding *is* the animator's control interface" note; exact choice remains open-Q1).

**What it is:** the PSD model interpolates `δ` as a function of *pose*. But "pose" needs to be expressed as coordinates in some space — and the choice of coordinates matters enormously for whether interpolation is smooth. That choice is the "pose-space encoding."

The naive choice is raw axis-angle `θ` (your 69-D `body_pose`). It's a bad interpolation coordinate because:
- Axis-angle isn't Euclidean — the "distance" between two rotations isn't the vector distance between their axis-angle representations.
- It has wraparound/aliasing (a rotation and that rotation + 360° are the same pose but far apart numerically).
- Interpolating it linearly doesn't produce a natural rotational blend.

Better encodings, roughly in order of sophistication:
- **Per-joint rotation matrices**, flattened (9 numbers/joint) — locally Euclidean, no wraparound.
- **Only the joints that actually move near a given vertex** ("self-relative configuration," paper §4.1) — the pose space for an elbow vertex is just the 1-D elbow angle; for a shoulder vertex it's 2–3 DoF. This keeps each interpolation problem low-dimensional, which is *critical* when you have few poses.

**Does it depend on the artist deliverable?** — Yes, and this is the good instinct in your question. The pose-space encoding *is essentially the rig control the animator gets.* If you encode pose as "elbow bend angle," the animator gets a blend shape driven by a 1-D elbow-bend slider. If you encode a whole-body abstract space, they get abstract sliders. The paper's Figure 6/7 emotion-space example is exactly this — the pose space can be an *abstract, artist-facing* space (happy↔sad axis), not just literal joint angles. So the encoding choice = the control interface choice. That's why I left it as an open question: it should be decided *with* the animator, once you have real non-T poses to see what varies.

**Resources** (I'm giving you canonical references rather than fetching pages — these are the ones to read):
- **The Lewis 2000 paper you already have** — §3 (scattered interpolation, Shepard vs RBF) and §4 (pose space definition) are *the* primary source. §4.1 "self-relative configuration" is the per-joint-DoF idea.
- **SMPL paper** (Loper et al., SIGGRAPH Asia 2015) — its **pose blendshapes** are literally a linear PSD baked into the model; the encoding it uses is *rotation matrices minus identity* (`R(θ) − I`) as the pose feature. That's a strong, battle-tested default encoding for you: `smpl.is.tue.mpg.de`.
- **"Pose-space deformation" + RBF** — search terms: *"pose space deformation RBF"*, *"scattered data interpolation skinning corrective"*, and the term **"corrective blendshapes"** / **"pose-driven correctives"** (the games-industry name for exactly this).
- For the artist-facing side: Maya's **"Pose Space Deformation" / "Pose Editor"** and the **BlendShape + driven-key** workflow docs — this is what the export ultimately targets, so seeing the animator UI clarifies what encoding is natural.

Want me to pull the SMPL pose-blendshape formulation into the spec as the *recommended default encoding* (`R(θ)−I`)? It's directly reusable and sidesteps the axis-angle problem cleanly.

---

## 4. Locking betas — should body dimensions stay fixed, or is drift expected?

> **Folded in:** → spec **R2** (β-lock now a *recommendation, not an option*) and **open-Q2** (narrowed to *where* the β-lock lives).

Physically: **`β` should be identical across poses.** It's the same person; their bone lengths and build don't change between standing in a T-pose and raising an arm. `β` encodes *identity/shape*, `θ` encodes *pose*. That separation is the whole point of SMPL.

So why did I raise it as a risk? Because **your fits are produced independently per pose**, and the optimiser doesn't know they're the same person. Nothing currently forces the T-pose fit and the arm-raised fit to share `β`. Small **estimation drift** will happen — not because the body changed, but because:
- Each pose is a separate optimisation with its own noise (different views occluded, different ViTPose errors).
- **Shape/pose ambiguity:** the optimiser can partly "explain" a pose effect by nudging `β` (e.g. make the torso `β` slightly wider instead of getting the spine rotation exactly right). This is a well-known SMPL failure mode.

That drift is *pure contamination* for PSD: if `β` differs between poses, then `δ = fit − SMPL(β,θ)` picks up **shape differences masquerading as pose deformation**. Your interpolation would then blend body *builds*, not just body *poses* — exactly wrong.

**So: lock it.** Concretely: fit the reference (T-pose) first, take its `β`, and re-fit every other pose with `β` **frozen** (optimise only `θ`, translation, scale). This is a small change to the Tier-2 optimiser (hold `betas` non-trainable). That's what R2 and open-question #2 are about. The expected-drift is real but it's *estimation* drift you want to eliminate, not *physical* drift you want to preserve.

---

## 5. Confidence weighting — what it refers to

> **Folded in:** already carried by spec §5.1 (`PoseSample.quality`) and open-Q4 (quality-weighted RBF rows). Optional refinement — no new spec edit needed.

Your instinct is right: it's a **quality metric**, but let me be precise about *which* quality and *where* it plugs in.

Not a metric "against the reference." It's about **how much to trust each pose sample when fitting the interpolation.** Some of your registered poses are better fits than others — you already compute this per pose:
- `pa_mpjpe_mm` (how well joints agreed),
- `median_reproj_px` (how well it reprojects),
- and per-view `keypoint_confs`.

A pose that fit cleanly (low reproj, many high-confidence keypoints) gives a trustworthy `δ`. A pose that fit poorly gives a noisy `δ`. **Confidence weighting** means: when the RBF solves the per-vertex least-squares `w = (ΦᵀΦ)⁻¹Φᵀd`, weight each pose's row by its fit quality, so a bad pose pulls the interpolation less. Standard weighted-least-squares:

```
w = (Φᵀ W Φ)⁻¹ Φᵀ W d      # W = diag(quality weight per pose)
```

The attached plan does a cruder version of this (`confidence = 1/(1 + chamfer/5)`) at the *training-sample* level. Same idea. It's cheap because you already have the metrics sitting in each `PoseSample.quality` — I put that in the contract (§5.1) precisely so it's available.

It's optional and a refinement, not core — with only ~7 poses it's a minor effect. I flagged it mainly because the plumbing is already there for free.

---

## Changes folded into the spec ✓

All three follow-on actions are now in [`psd_master_design_spec.md`](psd_master_design_spec.md):
1. ✅ **"Sequencing / critical path"** — added as spec **§4.4**: *more poses → Tier 3 → PSD-with-real-numbers* is the path; Stage A code is built early but its metrics aren't the goal (Q2).
2. ✅ **`R(θ)−I` recommended default encoding** — added to spec **§4.3**, with the "encoding = animator control interface" note (Q3).
3. ✅ **β-lock as a recommendation, not just an option** — tightened in spec **R2** and narrowed **open-Q2** to *where* it lives (Q4).


---

raw notes:

1. "Each vertex's local (unposed) frame" — what it means for the PSD residual
Think about what happens to a vertex on your forearm when you bend your elbow.

In world/global coordinates, that vertex moves a lot — it swings through an arc. If you stored the correction δ in world coordinates, δ would be dominated by that swing, and it would be totally different at every elbow angle. Useless for interpolation.

The insight of PSD (and of SMPL's own math): most of that motion is just rigid articulation — the bone rotated, so the skin rigidly rotated with it. That part is already handled by LBS (Linear Blend Skinning). What you actually want to model is only the extra deformation on top of the rigid motion — the muscle bulging, the skin creasing.

So you "undo" the rigid rotation before storing the correction:


δ_world = fit_vertex − posed_baseline_vertex        # the raw correction, in world space
δ_local = R_v(θ)⁻¹ · δ_world                          # rotate it back into the vertex's rest orientation
R_v(θ) is the rotation LBS applied to that vertex at pose θ (a blend of the nearby bones' rotations). Undoing it puts the correction back into the frame the vertex has when the body is in its rest pose — hence "local, unposed frame."

Why this matters concretely: if a vertex undergoes pure rigid articulation (no soft-tissue effect at all), then fit_vertex == posed_baseline_vertex, so δ_world = 0 and δ_local = 0. The residual is zero exactly when nothing interesting is happening. That's the property that makes δ smooth and interpolatable across poses — and it's also the decisive unit test I flagged as R5: feed in a pure articulation, and δ_local must come out ≈0.

The paper states this directly (page 5): "we instead interpolate the desired deviation of a surface vertex (expressed in the local frame) from its initially computed position ... rigid body articulated motion results in zero δ."

2. If Stage A can't quantify fit performance, should Tier 3 become the priority?
This is the sharpest question here, and I think the honest answer is: partly yes — but reframe it. Two distinct things are getting bundled:

(a) Can Stage A quantify the machinery? Yes. Stage A can fully measure interpolation error, smoothness, and export correctness — because it has held-out registered poses to interpolate toward. What it cannot do is claim that error means anything about real geometry, because both prediction and target live on the SMPL manifold.

(b) Is Stage A's δ worth shipping on its own? Geometrically, it's thin (the §4.2 caveat). So Stage A is real as a pipeline/plumbing proof and animator-workflow deliverable, but weak as a geometric result.

So the reframe: the blocker for a meaningful PSD result is not Tier 3 specifically — it's an independent geometry source. Tier 3 (point cloud → SMPL+D) is the one you've already architected, so yes, it's the natural priority if the goal is a defensible result rather than a demo. But note the true critical path is even earlier: you need multiple captured poses first (you have N=1). Tier 3 on a single T-pose gives you a better single mesh but still nothing to interpolate.

My recommendation on sequencing:

Capture more poses (critical path — nothing PSD works without N≥2).
Tier 3 to turn those into geometrically-real SMPL+D fits — this is the groundwork that makes PSD's residual meaningful, so yes, prioritise it over polishing Stage A.
Stage A then becomes a cheap baseline you benchmark against, not the end product.
In other words: build Stage A's code now (it's the same code Stage B uses), but don't invest in making Stage A's numbers look good — invest that effort in Tier 3.

I can add a short "Sequencing / critical path" section to the spec capturing this if you'd like.

3. Pose-space encoding — what it is, and does it depend on the artist deliverable?
What it is: the PSD model interpolates δ as a function of pose. But "pose" needs to be expressed as coordinates in some space — and the choice of coordinates matters enormously for whether interpolation is smooth. That choice is the "pose-space encoding."

The naive choice is raw axis-angle θ (your 69-D body_pose). It's a bad interpolation coordinate because:

Axis-angle isn't Euclidean — the "distance" between two rotations isn't the vector distance between their axis-angle representations.
It has wraparound/aliasing (a rotation and that rotation + 360° are the same pose but far apart numerically).
Interpolating it linearly doesn't produce a natural rotational blend.
Better encodings, roughly in order of sophistication:

Per-joint rotation matrices, flattened (9 numbers/joint) — locally Euclidean, no wraparound.
Only the joints that actually move near a given vertex ("self-relative configuration," paper §4.1) — the pose space for an elbow vertex is just the 1-D elbow angle; for a shoulder vertex it's 2–3 DoF. This keeps each interpolation problem low-dimensional, which is critical when you have few poses.
Does it depend on the artist deliverable? — Yes, and this is the good instinct in your question. The pose-space encoding is essentially the rig control the animator gets. If you encode pose as "elbow bend angle," the animator gets a blend shape driven by a 1-D elbow-bend slider. If you encode a whole-body abstract space, they get abstract sliders. The paper's Figure 6/7 emotion-space example is exactly this — the pose space can be an abstract, artist-facing space (happy↔sad axis), not just literal joint angles. So the encoding choice = the control interface choice. That's why I left it as an open question: it should be decided with the animator, once you have real non-T poses to see what varies.

Resources (I'm giving you canonical references rather than fetching pages — these are the ones to read):

The Lewis 2000 paper you already have — §3 (scattered interpolation, Shepard vs RBF) and §4 (pose space definition) are the primary source. §4.1 "self-relative configuration" is the per-joint-DoF idea.
SMPL paper (Loper et al., SIGGRAPH Asia 2015) — its pose blendshapes are literally a linear PSD baked into the model; the encoding it uses is rotation matrices minus identity (R(θ) − I) as the pose feature. That's a strong, battle-tested default encoding for you: smpl.is.tue.mpg.de.
"Pose-space deformation" + RBF — search terms: "pose space deformation RBF", "scattered data interpolation skinning corrective", and the term "corrective blendshapes" / "pose-driven correctives" (the games-industry name for exactly this).
For the artist-facing side: Maya's "Pose Space Deformation" / "Pose Editor" and the BlendShape + driven-key workflow docs — this is what the export ultimately targets, so seeing the animator UI clarifies what encoding is natural.
Want me to pull the SMPL pose-blendshape formulation into the spec as the recommended default encoding (R(θ)−I)? It's directly reusable and sidesteps the axis-angle problem cleanly.

4. Locking betas — should body dimensions stay fixed, or is drift expected?
Physically: β should be identical across poses. It's the same person; their bone lengths and build don't change between standing in a T-pose and raising an arm. β encodes identity/shape, θ encodes pose. That separation is the whole point of SMPL.

So why did I raise it as a risk? Because your fits are produced independently per pose, and the optimiser doesn't know they're the same person. Nothing currently forces the T-pose fit and the arm-raised fit to share β. Small estimation drift will happen — not because the body changed, but because:

Each pose is a separate optimisation with its own noise (different views occluded, different ViTPose errors).
Shape/pose ambiguity: the optimiser can partly "explain" a pose effect by nudging β (e.g. make the torso β slightly wider instead of getting the spine rotation exactly right). This is a well-known SMPL failure mode.
That drift is pure contamination for PSD: if β differs between poses, then δ = fit − SMPL(β,θ) picks up shape differences masquerading as pose deformation. Your interpolation would then blend body builds, not just body poses — exactly wrong.

So: lock it. Concretely: fit the reference (T-pose) first, take its β, and re-fit every other pose with β frozen (optimise only θ, translation, scale). This is a small change to the Tier-2 optimiser (hold betas non-trainable). That's what R2 and open-question #2 are about. The expected-drift is real but it's estimation drift you want to eliminate, not physical drift you want to preserve.

5. Confidence weighting — what it refers to
Your instinct is right: it's a quality metric, but let me be precise about which quality and where it plugs in.

Not a metric "against the reference." It's about how much to trust each pose sample when fitting the interpolation. Some of your registered poses are better fits than others — you already compute this per pose:

pa_mpjpe_mm (how well joints agreed),
median_reproj_px (how well it reprojects),
and per-view keypoint_confs.
A pose that fit cleanly (low reproj, many high-confidence keypoints) gives a trustworthy δ. A pose that fit poorly gives a noisy δ. Confidence weighting means: when the RBF solves the per-vertex least-squares w = (ΦᵀΦ)⁻¹Φᵀd, weight each pose's row by its fit quality, so a bad pose pulls the interpolation less. Standard weighted-least-squares:


w = (Φᵀ W Φ)⁻¹ Φᵀ W d      # W = diag(quality weight per pose)
The attached plan does a cruder version of this (confidence = 1/(1 + chamfer/5)) at the training-sample level. Same idea. It's cheap because you already have the metrics sitting in each PoseSample.quality — I put that in the contract (§5.1) precisely so it's available.

It's optional and a refinement, not core — with only ~7 poses it's a minor effect. I flagged it mainly because the plumbing is already there for free.

