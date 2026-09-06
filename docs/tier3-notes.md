The final Review pass did something remarkable: it re-examined three of Integration's "needs a spec decision" conclusions and proved them wrong by direct measurement — AC18, AC12, and AC10 are all still code-fixable, they just weren't fixed correctly yet. Only AC5 (the fixture/spec arithmetic contradiction) and AC13 (a genuine timing-budget tradeoff) plus one missing AC (D-accuracy against ground truth) actually need your decision. converged: false — this was the final allowed iteration (maxIterations = 3), so the loop stops here per its own rules rather than silently calling it done.

Final Report — tier3-surface-refinement
Converged: false, after 3 iterations. 17/24 acceptance criteria met.

Acceptance criteria
✓ AC1–4, 6–9, 11, 14–17, 19–23 (18 met, incl. AC9 correctly deferred — no real scanner cloud on disk, never falsely reported PASS)
✓ AC13, AC7 — met but fragile (see below)
✗ AC5 — translation error 26.6mm vs 5mm bound (spec/fixture defect, not a code bug — see B1)
✗ AC10 — semantic weighting measurement is invalid (see A4)
✗ AC12 — self-intersections 117→1262 on the shipped config (see A1)
✗ AC18 — D differs 1.96mm vs 0.5mm bound across equivalent inputs (see A2 — fully code-fixable, verified: fixing it makes D bitwise identical)
✗ AC24 — full suite not green (3 Tier-3 failures above + 3 confirmed pre-existing, unrelated failures)

What's genuinely solid
No weights or absolute paths committed, pyproject.toml untouched, no Tier-2 write-back, every pipeline-boundary function documents frame/units, AC8 at 81.5% improvement (vs. 40% bar), and three adversarial clouds (partial-body, heavy noise, near-empty) were all correctly rejected by the write-time sanity gates with nothing persisted.

What would need to happen to ship, in priority order
A. Code-fixable, no decision needed — do these first, as one cluster (they interact):

smpld-and-losses — normal_consistency_loss's trust radius is 0.015% of vertex spacing when it needs to be ~50-100%; it's inert 92% of the time then applies a destructive impulse the rest. Fixing this unblocks AC12 and materially improves AC18/AC13 too.
pointcloud-package — preprocess_cloud's voxel downsample isn't similarity-equivariant. Verified: fixing this alone makes AC18's D output bitwise identical (0.0000mm) across equivalent inputs. This was wrongly diagnosed as an unfixable spec issue earlier in the loop — it isn't.
surface-fitting — the early-stop criterion fires on a single noisy iteration instead of requiring sustained convergence, which is what's making AC10's measurement noise-dominated and AC18 flaky. Needs a patience/best-loss criterion.
surface-fitting — re-run AC10's A/B properly once #3 lands. Likely resolves without touching any weights.
tier3-pipeline-artefacts — add missing CLI test coverage (nothing currently exercises fit-surface end-to-end).
B. Decisions only you can make:

AC5 — the acceptance criterion is arithmetically self-contradictory on the fixture as built (its own 1% scale tolerance permits 16x its own 5mm translation bound). The alignment itself is excellent (1.9mm composite error). Needs the AC's translation clause rewritten.
AC13 — passes only via a data-dependent early exit; the full spec'd 550-iteration schedule takes ~90s against a 60s budget. Either shorten S3's iteration count in the spec or change what AC13 measures.
New gap: §10 has no AC on D's accuracy against the fixture's known ground truth, despite D being the feature's actual deliverable — on a perfect input it's currently 3.4x its true magnitude. Worth adding a threshold once #A1 lands.
Ratify or revert w_displacement_reg (unilaterally changed 0.01→0.1 by one component to compensate for defects now traced elsewhere).
C. Out of scope: 3 pre-existing Tier-1 test failures (independently confirmed via the pre-feature baseline commit) — unrelated to this feature, don't block on them.

Components touched
pointcloud-package, smpld-and-losses, surface-fitting, tier3-pipeline-artefacts — 15 files, ~2100 lines changed across 3 fix iterations, all uncommitted on main.

Nothing is committed yet. How would you like to proceed — should I commit the current state as a checkpoint (with a report of what's still open), and separately, would you like me to raise maxIterations and run a 4th cycle targeting the four code-fixable items (A1–A4), or would you rather make the B-decisions first so a future cycle fixes toward settled targets?

What the WIP commit actually contains:

Tier3-surface-refinement: P0 scale-collapse fix + iteration 2/3 fix cycle

pointcloud/align.py — fixed the P0 (ICP candidate selection was scale-degenerate, could silently collapse scale to ~0); decoupled converged/fitness reporting from the polish cascade (was tautological — could never fire); added finiteness guards + coverage checks against partial-body clouds.
fitting/surface_losses.py — fixed CUDA non-determinism in vertex-normal accumulation (index_add_ → sparse matmul, now bitwise-reproducible); attempted two different fixes for normal-loss divergence on the shipped config (trust-region weighting, then per-face correspondence) — neither reaches the AC bound, reverted to the better of the two, documented as blocked pending a spec-level weight change.
fitting/surface.py — asymmetric semantic weighting (cloud→mesh only) for AC10, improved but didn't fully close it; w_displacement_reg raised 0.01→0.1 (undocumented-in-spec deviation); relative early-stop tolerance replacing a miscalibrated absolute one.
fitting/surface_pipeline.py, config.py, cli.py — four new write-time sanity gates (Tier3SanityError) refusing to persist a catastrophically wrong fit (non-convergent alignment, scale deviation, one-sided chamfer asymmetry, implausible displacement magnitude).
Tests across all of the above; net new regression coverage for every fix.
Known still-open after 3 loop iterations: AC5 (spec/fixture arithmetic contradiction — needs a spec decision), AC10/AC12/AC18 (all confirmed code-fixable, not yet fixed correctly — see the loop's final report), AC13 (timing budget tradeoff, spec decision).

questions:
rabbit hole - research mode?
orthogonal feature specs?
changes to the external repos, why?

external/smplx/ — declared in .gitmodules but never actually staged as a submodule (git ls-files -s external/smplx is empty, so the outer repo has no gitlink for it — it's just a live nested git checkout sitting there untracked). The project's actual SMPL dependency is the smplx PyPI package in the venv (per CLAUDE.md), not this source checkout. Not needed for anything I built in tier 3

external/CameraHMR/ — a proper submodule, but with ~20 files locally modified inside it (CamSMPLify/*, core/backbones/*, core/cam_model/*, etc.), never committed. That's Tier 1's HMR model, unrelated to Tier 3. I'd guess compatibility patches for the current torch version (CameraHMR's stated requirement is PyTorch 2.0+, this env runs 2.12.1), but I didn't make these changes and don't know for certain — say the word if you want me to diff a few files and check.
Given both predate this session and neither is referenced by the tier3 code, I'd leave them alone unless you have a specific concern about them.

```
scantosmpl fit-surface \
    --tier2-dir output/debug/refinement \
    --pointcloud data/t-pose/pointcloud.ply \
    --pose-name t-pose \
    --subject dan \
    --output output/tier3/
```

 B-decisions → A2 (independent, well-understood) → A1 ablation/research → A1+A3 spec → fix cycle, 