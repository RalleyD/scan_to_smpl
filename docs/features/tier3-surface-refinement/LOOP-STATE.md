# Loop state — `tier3-surface-refinement`

**Converged 2026-09-06**, after 3 `/feature-loop` iterations plus a directed fix cycle
(iteration 4). All acceptance criteria are discharged except **AC9**, which is *correctly*
deferred — it needs a real Meshroom point cloud that does not exist yet.

## Status

| Phase | Status |
|---|---|
| 0 — Parse | ✅ 5 components, no `Owns` collisions |
| 1 — Implement | ✅ 5/5 `BUILD_RESULT`s, merged to `main` |
| 2 — Integrate → Review → Fix | ✅ 3 iterations, reached 17/24 |
| 3 — Directed fix cycle | ✅ **24/24 discharged, 1 deferred** (commits `9cad788`…`5023ba1`) |

AC count moved 24 → 24 by two offsetting changes: **AC13 retired**, **AC25 added**.

## What iteration 4 changed, and the evidence

| | Change | Result |
|---|---|---|
| **A2** | Decimation selects by index, not by voxel grid (D8 restated as *similarity-equivariant*, not merely unit-free) | **AC18: 1.962mm → 0.000000mm.** D bitwise identical across frames |
| **A1** | `w_normal` 0.1 → **0.0**, `w_laplacian` 0.1 → **3.0**, `w_displacement_reg` → **0.01** (spec value restored) | **AC12: 1262 → 115** self-intersections, bound 122 |
| **A3** | Early stopping needs 10 *consecutive* quiet iterations; best iterate restored, not last | More of the schedule runs (S2 82→116, S3 157→250); AC10 stops being a coin flip |
| **A4** | AC10 re-measured over 5 seeds, early stopping disabled | **5/5 seeds**, mean −0.0142mm, sd 0.0023mm |
| **B1** | AC5 restated on composite pointwise transform error | **1.476mm** against a 5mm bound |
| **B3** | New **AC25** gates `D` against the fixture's known `D_true` | mean 1.030 / p95 2.425 / torso 1.713mm |
| **B2** | **AC13 retired** — no latency requirement to defend | Unblocks A3; ~10× pathology guard kept |

Three findings worth carrying forward, because each was a *measurement* defect rather than a
code defect, and each would recur:

1. **The aggregate `|D − D_true|` mean is a misleading selector.** 5138 of 6890 vertices have
   `D_true == 0`, so the trivial `D ≡ 0` scores 1.017mm and beats every real fit while recovering
   none of the offset. `w_laplacian` had to be selected on **torso** error, which is U-shaped with
   an interior minimum; the aggregate keeps improving well past the point where signal collapses.
2. **AC5's translation clause was measuring scale.** `translation` is defined about the origin and
   the fixture's centroid sits ~2.96 units away, so `translation_err ≈ scale_rel_err × ‖t_true‖`.
3. **AC10's effect was smaller than its own noise** (+0.065mm vs ±0.072mm), so it was a coin flip
   that duly flipped. It is real (~6σ) once measured across seeds on a fixed-length schedule.

## Remaining work

**AC9 — the real-cloud 8mm gate has never run.** It skips, `summary.txt` prints
`TIER 3 GATE: DEFERRED (no real point cloud)`, and that is the honest state. It needs
`data/t-pose/pointcloud.ply` from Meshroom. This is the one thing blocking a genuine Tier 3
sign-off; everything else is discharged on the synthetic fixture, where `D_true` is known exactly.

Once the cloud exists:

```bash
V=/home/dan/.pyenv/versions/smpl_psd_venv/bin/python

# The end-to-end run
$V -m scantosmpl.cli fit-surface \
    --tier2-dir output/debug/refinement \
    --pointcloud data/t-pose/pointcloud.ply \
    --subject dan --pose-name t-pose \
    --output output/fits/

# Then the gate itself
$V -m pytest tests/integration/test_tier3_integration.py::test_real_cloud_chamfer -v
```

Expect the real cloud to stress things the fixture cannot: non-uniform density (index decimation
preserves the input distribution rather than equalising it — farthest-point sampling from a
frame-independent seed is the recorded fallback), genuine clothing displacement well above the
fixture's uniform 4mm, and an ICP init with no known-answer to check against.

## Environment rules — still in force

- **`/home/dan/.pyenv/versions/smpl_psd_venv/bin/python` is authoritative** (torch 2.12.1+cu130).
  The repo-root `.venv/` is a *different* environment (torch 2.11.0) — do not run, test, or measure
  against it. Every number in this document was measured in the pyenv venv.
- **Never `pip install`; never create a venv.** A genuinely missing package is a blocker, not an
  install. This is instruction-enforced, not config-enforced: `.claude/settings.json` still allows
  bare `Bash(pip install:*)`, so every agent prompt must state it explicitly.
- **Baseline for "did I break something":** commit `96e2359`. The full suite has exactly three
  pre-existing Tier-1 failures (`test_consensus` ×1, `test_hmr_integration` ×2) that predate this
  feature. A fourth failure is a regression — see AC24, which is scoped to say exactly this.
