# PSD primer — what δ is, before you build it

Companion to [`tier3-notes.md`](tier3-notes.md). The binding spec is
[`psd_master_design_spec.md`](psd_master_design_spec.md) and it wins wherever this disagrees. The
one place to read alongside it is §7: the *decision* there was settled on 2026-08-02 and is already
in the spec (§4.4) — but §4.2's wording still understates why, and has not been updated.

There is also a designed reading copy of this document published as an artifact. This file is the
canonical text — if you change one, change both.

---

## 1. What linear blend skinning gets wrong

LBS moves each vertex by a *weighted average of its joints' rigid transforms*:

```
v′ = Σ_k  w_k · (R_k v + t_k)
```

where `w_k` is how strongly vertex `v` is bound to joint `k` — in SMPL, `lbs_weights`, shape
(6890, 24), fixed forever.

The flaw is in that average. **A weighted average of rotation matrices is not a rotation matrix.**
It is some other linear map, one that can shear and can shrink. Two failure modes follow:

- **Joint collapse** — bend an elbow 90° and the inside of the crease loses volume, because the
  blend between two rotations passes through a transform with determinant < 1.
- **The candy-wrapper** — twist a forearm 180° and the blend between `R` and `R·twist` passes
  through a near-degenerate matrix. The mesh pinches to a point.

These are the reason production rigs carry corrective shapes at every major joint.

---

## 2. What SMPL already corrects

**Get this straight before writing any code**, or you will re-learn something you already have.

**SMPL is not raw LBS.** It already contains a linear PSD of its own, learned from population data
— the *pose blendshapes*. The full decomposition:

```
template  +  shape blendshapes(β)  +  pose blendshapes(R(θ)−I)  →  LBS(θ)  +  δ
└────────────────────── all SMPL ──────────────────────────────────────┘   └─ you ─┘
```

So δ is **not** "the correction LBS needs" — SMPL's pose blendshapes already handle the generic,
population-average version of that. What is genuinely left for δ:

- **Clothing** — fabric does not follow a body's LBS weights
- **Hair** — entirely off-manifold
- **This subject's** soft tissue, where they deviate from the population average SMPL learned
- Breathing, posture, muscle tone at capture time

> **Why it matters for your model.** SMPL's pose blendshapes are *linear* in `R(θ)−I`. A linear
> model cannot represent a bulge that appears only past 60° of flexion. If δ carries genuine
> nonlinear pose dependence, that is exactly what RBF-PSD exists to capture and what SMPL
> structurally cannot.

---

## 3. The residual, precisely

Two quantities, two frames. Keeping them apart is most of the battle.

```
D    :=  V_final − SMPL(β, θ, t, s)      Tier 3's output. (6890, 3), metres, POSED WORLD frame.
δ_v  :=  R_v(θ)⁻¹ · D_v                  PSD's target. Same correction, LOCAL REST frame.
```

### Why the local frame is non-negotiable

Stored in world coordinates, raising an arm would change δ for every vertex in that arm even though
nothing deformed. The model would spend its capacity re-learning rotation, and interpolating between
poses would smear the arm through space.

In the local frame, **pure articulation produces δ = 0**. Only real deformation registers. That
invariance is why PSD interpolates cleanly, and it is the paper's central idea.

### What `R_v(θ)` actually is

The rotational part of the blended LBS transform for that vertex:

```
R_v(θ) = Σ_k  w_vk · R_k(θ)
```

> ⚠️ **A decision you must make explicitly.** That blended matrix is **not orthonormal** — it is the
> same weighted average of rotations from §1, with the same shear and scale contamination. So `R⁻¹`
> is not `Rᵀ`, and inverting it directly folds the skinning artefact back into δ.
>
> Either invert it properly, or orthonormalise first via polar decomposition (the closest true
> rotation, from SVD: `R = UΣVᵀ → UVᵀ`). **Pick one, write down which, use the same one in training
> and export.** A mismatch produces a rig that looks right in the viewer and is wrong in Maya.

---

## 4. Pose space — the input to `f`

PSD learns `f: θ → δ`. But raw θ is a poor coordinate to interpolate in.

SMPL's θ is **axis-angle**: 3 numbers per joint, direction = axis, magnitude = angle. That space is
not Euclidean — distance between two axis-angle vectors does not match the angle between the
rotations they represent — and large rotations alias.

### Use `R(θ) − I`

Per-joint rotation matrices, flattened, minus identity. 9 numbers per joint; 23 body joints × 9 =
(207,) for a full-body feature.

Three reasons this is the right default:

1. Locally Euclidean, no wraparound.
2. **Exactly zero at the rest pose**, so a rest-pose prediction is structurally zero rather than
   approximately zero.
3. It is *literally the feature SMPL's own pose blendshapes use*, already validated on this exact
   topology. You are extending a proven parameterisation, not inventing one.

### Then reduce it, per vertex

A vertex on the elbow does not care what the left ankle is doing. The paper's "self-relative
configuration" idea: **the pose-space dimension can differ per vertex**. Drive each vertex from only
the joints it is bound to — `lbs_weights` already tells you which.

Start 1-D per active joint where you can (elbow bend, knee bend) and grow only where the data
demands it. With five poses, a 207-D feature space is absurdly under-determined; a 1-D one is not.

> **This choice is the rig's UI.** The pose-space coordinates *become the animator's sliders*.
> Encode pose as "elbow bend angle" and they get an elbow-bend slider; encode an abstract whole-body
> space and they get abstract sliders they cannot reason about. Decide it with whoever will animate
> it, once you have real non-T poses to show what varies.

---

## 5. The model: RBF

For a single subject with a handful of poses, classical RBF interpolation is the correct tool — and
it *is* what the paper describes. A learned network needs a validation split you cannot spare at
N = 5.

```
Φ_ij  = φ(‖x_i − x_j‖),   φ(r) = exp(−r² / 2σ²)
w     = Φ⁻¹ d
δ(x)  = Σ_j  w_j · φ(‖x − x_j‖)
```

### The implementation shortcut worth knowing up front

It looks like you need 6890 independent least-squares solves. You do not.

**Φ depends only on the pose features — not on the vertex.** Every vertex shares the same kernel
matrix, so you factor one (N, N) matrix once and apply it to all vertices at once:

```python
# N poses, 6890 vertices. N is ~5, so Phi is 5x5.
Phi = np.exp(-cdist(X, X)**2 / (2 * sigma**2))   # (N, N)
D   = deltas.reshape(N, -1)                      # (N, 6890*3)
W   = np.linalg.solve(Phi, D)                    # (N, 6890*3)  <- one solve

# predict at a new pose
phi   = np.exp(-cdist(x_new[None], X)**2 / (2 * sigma**2))   # (1, N)
delta = (phi @ W).reshape(6890, 3)
```

Two lines of linear algebra. The expensive part of this project was never the model.

### Choosing σ

A sensible default is the mean nearest-neighbour distance between training poses in feature space.
Too small and the model is a set of isolated spikes that reads as popping; too large and everything
blurs toward the mean.

σ is also the **animator's falloff control** — the paper treats it as a deliberate artistic
parameter, not a hyperparameter to tune away. Expose it.

> **Exact interpolation has a consequence.** With `w = Φ⁻¹d` the model reproduces every training
> pose *exactly*. Training-set error is ≈ 0 by construction. It is a wiring check, not a result —
> never report it as accuracy.

---

## 6. The corpus

| N poses | What you can do |
|---|---|
| **1** | Nothing. There is no space to interpolate in. |
| **2** | Interpolate along a single axis. Minimum viable demo. |
| **5** | Current plan. Enough for a real 1-D-per-joint model with a held-out pose. |
| **7–10** | Genuinely useful. The target. |

Each pose needs: images, a Tier 2 fit, and — for Stage B — a Meshroom point cloud. **Today exactly
one pose has a cloud (T-pose).** Producing the rest is the critical path; no amount of PSD code
substitutes for it.

> **β must be locked across the corpus.** Every pose of the same subject uses the *same* β, frozen
> to the reference pose's. It is the same person — their shape cannot legitimately change between
> captures.
>
> If β floats per pose, β and δ trade off: a fit can explain the same geometry either by changing
> body shape or by adding displacement, and it will pick arbitrarily. δ then measures fitting noise
> that varies pose to pose, and PSD will happily learn it as pose-dependent deformation. This is
> what Tier 3's `--lock-betas` exists for.

---

## 7. Stage A is a harness, not a result

Two corpora, differing only in where ground-truth δ comes from:

| Stage | Source of δ | Needs a cloud? |
|---|---|---|
| **A** | `M_fit − SMPL(β, θ)` — the registered fits themselves | No |
| **B** | `scan_surface(θ) − SMPL(β, θ)` — Tier 3 output | Yes |

> **Settled 2026-08-02** — see [`psd_qa.md`](psd_qa.md) Q2, folded into spec §4.4 and §4.2, with
> β-lock settled in Q4 → R2. This is a restatement, not a new decision.
>
> Spec §4.2 calls Stage A's δ "geometrically thin", which reads as small-but-present. It is sharper
> than that. Follow the algebra: `M_fit,i` *is* `SMPL(β, θ_i, t_i, s_i)` — a SMPL mesh, produced by
> SMPL, from those exact parameters. The baseline is `SMPL(β, θ_i)`. Same β (locked), same θ. The
> difference is
>
> ```
> δ_v = (s − 1)·p_v + t
> ```
>
> — a **global similarity with 4 degrees of freedom, not a per-vertex displacement field**.
> Canonicalise the frames (remove t and s, which is what "in the same canonical frame" means) and
> **δ ≡ 0 exactly, everywhere**.
>
> Stage A has no trainable target. The only way to make δ non-zero is to *unlock* β and train on
> estimation drift — which is noise, not anatomy: low-rank, global rather than joint-local, and
> correlated with per-pose image quality. Rejected; keep the R2 β-lock.

### Use it as a self-test

A zero target sounds useless, but it converts Stage A into something genuinely valuable: **a
known-answer test for the hardest part of the pipeline.**

δ ≡ 0 is what *correct* looks like. So if your Stage A δ comes out non-zero, you have a bug, and the
value tells you which one:

| Symptom | Likely cause |
|---|---|
| δ is a smooth global gradient across the body | `t`/`s` leaking into the baseline — frames not canonicalised |
| δ varies pose to pose, largest where the fit was worst | β is not actually locked |
| δ is joint-local and grows with joint angle | `R_v(θ)⁻¹` is wrong — probably the orthonormalisation of §3 |

That last row is spec **R5**, the decisive unit test for the local-frame rotation, run at corpus
scale rather than on one synthetic case. You need a synthetic known-answer δ anyway to test
`R_v(θ)⁻¹`, because no *real* δ has a known correct value — Stage A gives you one for free across
every pose you have.

**So: build Stage A's code, and do not chase its numbers.** It is the same code Stage B runs — only
the ground-truth source swaps — so it is genuinely worth writing early. Its job is to prove the
machinery: frame conversions, the kernel solve, interpolation smoothness, the Maya export contract.
Treat a Stage A δ of zero as *confirmation the frames are right*, not a failure to fix.

Everything that makes PSD mean something arrives with Stage B, which arrives with the point clouds.

---

## 8. Evaluation & the circularity guard

**Stage A**

- *Train reconstruction* — ≈ 0 by construction. Sanity check only.
- *Held-out interpolation* — predict δ at a held-out pose's θ, compare to that pose's own registered
  δ. The only real Stage A number.
- *Smoothness* — second derivative of vertex paths across an interpolation sweep. Catches popping.
- *No chamfer.* Measuring against a point cloud here scores the model on geometry it never saw.

**Stage B**

- *Chamfer vs the cloud* at held-out poses, for both Stage A and Stage B predictions — the headline
  number: how much closer to real geometry does scan supervision get us?
- *δ magnitude, A vs B* — puts a number on how thin the MVP residual was.

> **The circularity guard.** Stage A validates *internal consistency only* — its ground truth came
> from the same fitting process it is scored against, so it cannot certify real-world accuracy (the
> same limitation the self-calibration reprojection metric has). Stage B, using clouds that were
> never in any fitting or PSD loop, is the first genuinely independent check the project has. Say so
> explicitly in the report so Stage A numbers are never over-claimed.

---

## 9. Build order

The blocker for good numbers is upstream of the model, so the order is not the obvious one:

1. **Capture more poses.** The true critical path. N = 1 today.
2. **Meshroom clouds for those poses**, then Tier 3 fits. This makes δ geometrically real — and
   finally runs Tier 3's deferred 8mm real-cloud gate (AC9).
3. **Stage A code, early and cheap.** Same code Stage B runs. Prove frames, interpolation and export
   against a δ you know should be zero.
4. **Stage B.** Swap the ground-truth source. Nothing else changes.

The pipeline is deliberately designed so swapping the ground-truth source is the *only* difference
between stages. If you find yourself writing a second code path for Stage B, something has gone
wrong.

---

## 10. Traps checklist

- **Don't re-learn SMPL's pose blendshapes.** δ sits after them. If your model is fitting generic
  joint-bulge, check your baseline includes the pose-blendshape term.
- **Don't interpolate in axis-angle.** Not Euclidean, and it aliases. Use `R(θ) − I`.
- **Don't invert a non-orthonormal `R_v` casually.** Decide between true inverse and polar
  decomposition, document it, use it identically in training and export.
- **Don't report Stage A train error as accuracy.** RBF interpolates exactly; it is always ≈ 0 and
  always meaningless.
- **Don't unlock β to make Stage A's δ non-zero.** That converts "no signal" into "noise presented
  as signal".
- **Don't put a chamfer metric in Stage A.** It scores against geometry the model never saw.
- **Don't let the pose encoding be an internal detail.** It is the animator's control surface.
