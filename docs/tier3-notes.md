# Tier 3, from first principles

A plain-language guide to what Tier 3 does, why it is built the way it is, where the neural
networks actually live, and how it feeds PSD.

> This file previously held the loop's iteration-3 scratch report. That content is superseded by
> [`features/tier3-surface-refinement/LOOP-STATE.md`](features/tier3-surface-refinement/LOOP-STATE.md)
> and recoverable from git at `56cb1a3`.

---

## 1. What SMPL actually is

SMPL is a **function**, not a neural network:

```
M(β, θ)  →  6890 vertices, 13776 triangles
```

- **β** — 10 numbers describing *body shape* (tall/short, heavy/light, …).
- **θ** — 72 numbers describing *pose*: 24 joints × 3, as axis-angle rotations. In this codebase
  that is split into `global_orient` (3, the root) and `body_pose` (69, the other 23 joints).

Internally it is four steps of linear algebra:

1. Start from a **template mesh** — an average human body.
2. Add **shape blendshapes**: β linearly combines 10 learned "shape directions".
3. Add **pose blendshapes**: small pose-dependent corrections (this is SMPL's own built-in nod to
   the problem PSD solves — see §7).
4. **Linear Blend Skinning (LBS)**: every vertex is bound to up to 24 joints with fixed weights
   (`lbs_weights`, shape 6890×24), and moves as a weighted blend of those joints' rotations.

Those blendshapes were *learned*, by PCA and regression over thousands of registered body scans.
But at inference time **nothing here is a network** — it is matrix multiplication, and it is
differentiable, which is the property the whole of Tier 3 depends on.

### The one property everything rests on

**The topology is fixed.** Vertex 4321 is the same anatomical point on every body, in every pose,
for every subject, forever. 6890 vertices, always, in the same order.

This is what makes the rest possible: you can subtract two SMPL meshes vertex-by-vertex and the
difference is meaningful. You can learn a per-vertex function. You can export blend shapes to Maya.
A raw photogrammetry scan has none of this — it is an unordered bag of points with a different
count every time you run it.

### What "registration" means

> Find the (β, θ) whose SMPL mesh best explains my data.

It is an **inverse problem**, and an underdetermined one. Many different (β, θ) produce nearly the
same silhouette from any one viewpoint — a slightly larger person slightly further away looks
identical. That ambiguity is what the tiers exist to whittle down.

---

## 2. Why three tiers

| Tier | Input | Method | Result |
|---|---|---|---|
| **1** | ~20 usable images | Neural nets guess (β, θ) per image, fuse in parameter space | ~40–50mm |
| **2** | all ~60 images | Recover cameras *from the body itself*, triangulate, optimise | target <25mm |
| **3** | point cloud | Fit to the actual scanned **surface** | target <8mm chamfer |

Each tier is independently useful, and errors don't cascade — each refines the last.

### The gap Tier 3 exists to close

Tiers 1 and 2 both ultimately constrain **joint centres**. And a joint centre is *inside the body*,
on the medial axis.

Picture two people with an identical skeleton but different body fat. Their joint positions are
**exactly the same**. Their surfaces are not. No amount of 2D keypoint reprojection can tell them
apart, because keypoints only ever observe joints.

The practical consequence is recorded in this project's own `notes.md`: **torso girth is
unconstrained and drifts run to run.** Tier 2 has no term that can see it.

Tier 3 is the first time actual surface geometry enters the pipeline.

---

## 3. What Tier 3 does — three stages

Input: a Meshroom point cloud (arbitrary scale, arbitrary orientation, arbitrary position — the
photogrammetry has no idea how big the person is) plus a Tier 2 SMPL fit (correct metric scale,
roughly correct pose).

```
S1  align cloud → mesh      ICP only. SMPL is never touched.
S2  fit SMPL → surface      Optimise β, θ, translation.  D held at 0.
S3  solve D                 Freeze every SMPL parameter. Solve D alone.
```

### S1 — align the cloud to the mesh (not the reverse)

SMPL already has correct metric scale; the cloud does not. So the cloud moves.

The interesting part is initialisation. ICP only converges from a decent starting guess, so we take
the **principal axes** (PCA) of both the cloud and the mesh — roughly "long axis, medium axis, short
axis". But PCA axes are ambiguous: signs are arbitrary (is "up" +y or −y?), and when two eigenvalues
are close, the order is arbitrary too.

Rather than guessing with heuristics, S1 **enumerates all 24 proper rotations** that map one axis
triad onto another, runs ICP from each, and keeps the best. 24 cheap ICP runs, fully deterministic,
no "if the person is taller than they are wide…" rules to get wrong.

### S2 — fit SMPL to the surface

Now optimise β, θ, `global_orient`, `translation` against the surface, with the displacement field
held at exactly zero. **Scale is frozen** — it was solved once in S1, and letting S2 move it again
would make the two solves redundant and destroy the "SMPL has correct metric scale" premise.

This is where torso girth finally gets constrained, because the loss can now see the surface.

### S3 — solve the displacement field D

Freeze every SMPL parameter. Solve **D**: one 3D offset per vertex, 6890×3 numbers.

D captures everything a parametric body model *cannot* represent — clothing, hair, soft-tissue
detail. Anything genuinely off the SMPL manifold.

### Why the staging is the whole point

If you solved D and θ *simultaneously*, D would quietly absorb pose error. D would then mean
"clothing, plus whatever pose error happened to be left over" — and as a training signal for PSD
that is worse than useless, because the pose error is different in every pose.

Staging guarantees D contains **only** off-manifold geometry. That is the single most important
design decision in Tier 3, and most of the acceptance criteria exist to defend it.

---

## 4. The losses, in plain terms

| Loss | Weight | What it's for |
|---|---|---|
| **Chamfer** (bidirectional) | 1.0 | Pull mesh and cloud together |
| **Laplacian** on D | 3.0 | Neighbouring vertices should displace *similarly* |
| **‖D‖²** | 0.01 | Don't displace more than you must |
| **Normal consistency** | **0.0 — off** | See below |
| Pose prior / shape reg | 0.01 | Stay near a plausible human (S2 only) |

**Chamfer** is "for each mesh vertex, distance to the nearest cloud point" — and, crucially, the
reverse as well. A *one-sided* chamfer produces the classic shrink-wrap failure: the mesh collapses
into the densest part of the cloud while uncovered regions drift away unpenalised. Both directions
come free from the same distance matrix.

**Semantic weighting** scales the chamfer per body part — torso 1.0, arms/legs 0.7, head 0.5, feet
0.4, hands 0.3 — because hands and feet are noisy in photogrammetry and you do not want them
dragging the torso fit around. Part labels come from `lbs_weights`: whichever joint a vertex is most
strongly bound to *is* its body part. Exact, deterministic, no segmentation heuristic.

### Why the normal term is switched off — a useful cautionary tale

The obvious idea: the mesh surface should face the same way as the scan surface, so penalise
`1 − |cos(angle between normals)|`.

It wrecks the mesh. Measured: **117 → 1262 self-intersecting faces.**

The reason is instructive. A vertex normal is computed from its *neighbouring faces*, so a gradient
that flows through the normal pulls each vertex independently, with no term coupling it to its
neighbours. The optimiser discovers it can improve every vertex's normal alignment by making the
surface locally crinkly. It is a wrinkle-generating objective wearing a smoothness costume.

The reference implementation this project cites (DavidBoja/SMPL-Fitting) uses normals **only as a
correspondence filter**: reject point pairs whose normals disagree by more than N degrees, then
return the surviving pairs' *positional* distance. The gradient stays positional and cannot wrinkle.
Their shipped default omits the normal term entirely.

So: the term is kept in the codebase, exported and unit-tested, but ships at weight 0. **Laplacian
smoothing** is what actually prevents wrinkling, and it was raised 30× to do that job.

---

## 5. Where the neural networks actually are

**Tier 3 contains no neural networks at all.** It is classical optimisation end to end — ICP, then
Adam on a differentiable mesh. This surprises people, and it is worth being clear about.

The networks all sit **upstream**, in detection and Tier 1:

| Model | Job | What it is |
|---|---|---|
| **RT-DETR** | Find the person | Detection transformer → bounding boxes. No hand-tuned anchors or NMS. |
| **ViTPose++** | 2D keypoints | Vision Transformer → heatmaps → 17 COCO joint positions in *pixels*. |
| **CameraHMR** | Image → body | The big one. See below. |

### HMR — "Human Mesh Recovery"

The idea that makes Tier 1 possible: **regress SMPL parameters directly from a cropped image**.

```
crop of a person  →  ViT backbone  →  features  →  regression head  →  β (10), θ (72), camera
```

The network never outputs a mesh. It outputs the ~85 numbers that *drive* the mesh, then SMPL (the
non-network function from §1) turns them into geometry. That is why the output is always a valid
human body — the model is structurally incapable of producing a mesh that isn't one.

**CameraHMR's** specific contributions, and why this project picked it:

- A **full perspective camera model** rather than the weak-perspective approximation most HMR models
  assume, plus a `HumanFoV` network that predicts field of view (5–7° error). That matters when you
  have no calibration at all.
- **138 dense surface keypoints** alongside the sparse joints. Tier 2's camera recovery does PnP
  against these — 138 correspondences is dramatically more robust than 12 sparse joints.

### One clarification on the pose prior

CLAUDE.md's stack table lists a GMM pose prior (from SMPLify), with VPoser as an optional upgrade.
**Neither is implemented.** `pose_prior_loss` in [`fitting/losses.py`](../scantosmpl/fitting/losses.py)
is currently plain L2-toward-neutral. It is weighted at 0.01 and does little work in Tier 3, where
the chamfer term dominates — but it is a gap between the doc and the code worth knowing about.

---

## 6. What Tier 3 hands downstream

Per subject, per pose:

```
output/fits/<subject>/<pose>/
  smpl_params.npz     β, θ, translation, scale  — regenerate the baseline bit-identically
  displacements.npz   D, (6890, 3), metres      — THE deliverable
  registered.obj      the final mesh
  alignment.json      the recovered cloud→SMPL similarity
  quality.json        chamfer both directions, separately, + tessellation floor
manifest.json         locates every pose by name
```

`D` is defined exactly as:

```
D  :=  V_final  −  SMPL(β, θ, t, s)          in the POSED WORLD frame, metres
```

---

## 7. How this enables PSD

### The problem PSD solves

Linear Blend Skinning is fast and simple, and it is also wrong in a specific, visible way. Bend an
elbow 90° and LBS collapses the volume at the joint. Twist a forearm and you get the classic
"candy wrapper" pinch. Real bodies bulge where muscles compress and crease where skin folds; a
weighted average of two bone rotations does none of that.

**Pose Space Deformation** (Lewis et al., 2000) fixes it by learning a corrective displacement as a
*function of pose*:

```
f: θ  →  δ        per-vertex corrective displacement
```

At runtime: pose the body with LBS as usual, then add `δ(θ)`. Exported to Maya as blend shapes.

### Why δ lives in the unposed frame

PSD stores δ in each vertex's **local (unposed) frame**, so that pure articulated motion produces
`δ = 0`. If you stored it in world coordinates, simply rotating an arm would change δ even though
nothing actually deformed, and the model would spend its capacity re-learning rotation.

Tier 3 deliberately persists D in the **posed world** frame and lets PSD apply the inverse per-vertex
LBS rotation itself:

```
δ_i  =  R_v(θ_i)⁻¹ · D_i
```

This is a considered split: it hands PSD the raw geometric difference and keeps the frame conversion
in the tier whose spec defines the frame, rather than having Tier 3 guess.

### Stage A vs Stage B — why Tier 3 is the critical path

The PSD spec defines two corpora:

| | Source of δ | Quality |
|---|---|---|
| **Stage A** | The registered SMPL fits themselves — `fit − LBS-posed template` | **Geometrically thin by construction** |
| **Stage B** | Tier 3 surface fits — `scan_surface(θ) − SMPL(β,θ)` | Real observed geometry |

Stage A's δ is thin because it only captures the difference between a *fitted* SMPL and a *posed*
SMPL — both of which live on the same manifold. It exercises the machinery (corpus format, training
loop, Maya export) and delivers a working rig, but there is very little real signal in it. The PSD
spec says so explicitly, and prioritises Tier 3 over polishing Stage A for exactly this reason.

**Stage B is where PSD gets something real to learn, and Tier 3's D *is* Stage B's δ.**

### Why Tier 3's discipline matters so much

Several Tier 3 constraints look pedantic in isolation. They exist because PSD consumes D directly,
and each one prevents a specific way of poisoning the training signal:

| Constraint | Without it, PSD would learn… |
|---|---|
| **Staged solve** (D only after β/θ freeze) | …clothing *plus leftover pose error*, which differs per pose |
| **β locked across poses** | …β and D trading off against each other, so D isn't comparable pose to pose |
| **Similarity-invariance** (AC18) | …"when the scan happened to be rotated like this, deform like that" |
| **Fixed topology** (6890, always) | …nothing — per-vertex regression is impossible without it |
| **Frame asserted, not assumed** | …a silently rotated δ that still trains, and is entirely wrong |

That third row is worth dwelling on. A Meshroom cloud arrives in an arbitrary frame every single
run. If any part of the pipeline depended on that frame, D would encode the *scanner's* accidental
coordinate system, and PSD would learn a pose→deformation mapping contaminated by it. This is why
AC18 checks that D comes out **bitwise identical** when the same cloud is fed in under a different
similarity transform, and why the preprocessing decimation had to be rewritten to select points by
index rather than by a spatial voxel grid.

### Current state

Only the **T-pose** has a real point cloud so far. PSD needs a corpus of ~5 poses, and generating
those Meshroom clouds is the project's critical path — it is also what would finally discharge
Tier 3's one deferred acceptance criterion (AC9, the 8mm real-cloud gate, which has never run).

---

## Quick glossary

| Term | Meaning |
|---|---|
| **β (betas)** | 10 numbers = body shape |
| **θ (theta)** | 72 numbers = pose, as per-joint axis-angle rotations |
| **D** | Per-vertex displacement, (6890, 3), posed world frame — Tier 3's deliverable |
| **δ (delta)** | The same correction in the unposed local frame — PSD's learning target |
| **LBS** | Linear Blend Skinning: vertices follow a weighted blend of joint rotations |
| **Chamfer distance** | Nearest-neighbour distance between two point sets |
| **ICP** | Iterative Closest Point — classical rigid/similarity alignment |
| **HMR** | Human Mesh Recovery — regressing SMPL params from an image |
| **PnP** | Perspective-n-Point — recover a camera pose from 3D↔2D correspondences |
| **Chamfer floor / tessellation floor** | The error you'd measure even with a perfect fit, because a triangle mesh can't represent a curved surface exactly |
