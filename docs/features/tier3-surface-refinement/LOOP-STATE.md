# Loop state — `tier3-surface-refinement`

**Paused 2026-08-16, mid Phase 1.** Cause: Anthropic session usage limit (resets 02:30
Europe/London). Not a code, spec or environment failure. Delete this file once the loop converges.

## Where the loop got to

| Phase | Status |
|---|---|
| 0 — Parse | ✅ 5 components, 24 acceptance criteria, no `Owns` collisions |
| 1 — Implement (3 parallel worktree briefs) | ⏸ **partial**, see below |
| 1 — Implement (2 serial briefs) | ⛔ not started |
| 2 — Integrate → Review → Fix | ⛔ not started (`iter = 0`, `maxIterations = 3`) |

## Surviving work — do NOT restart these from scratch

Each agent has a saved transcript and a live git worktree. **Resume via `SendMessage` to the agent
id**, which preserves its context; a fresh `Agent` call would start cold and rebuild from nothing.

| Agent id | Component | Worktree branch | Files written |
|---|---|---|---|
| `ae6301ea71facdc47` | `pointcloud-package` | `worktree-agent-ae6301ea71facdc47` | `io.py` 208, `preprocess.py` 160, `align.py` 345 |
| `a71adf7a5b73a263f` | `surface-metrics` | `worktree-agent-a71adf7a5b73a263f` | `surface_metrics.py` 401, `tests/test_surface_metrics.py` 431, `evaluation/__init__.py` (M) |
| `a27dacbd0b9759914` | `smpld-and-losses` | `worktree-agent-a27dacbd0b9759914` | `smpl/model.py` (M) 182, `surface_losses.py` 339, `tests/test_surface_losses.py` 188 |

Worktrees live under `.claude/worktrees/agent-<id>/` and are `locked`.

### Known outstanding per agent

- **`pointcloud-package`** — furthest behind, roughly step 3 of 5. Missing `segment.py`,
  `__init__.py` exports, and `tests/test_pointcloud.py` entirely, so the step 2/3 verifications
  (`-k preprocess`, `-k align`) are **not** discharged. AC7's determinism test does not exist yet.
- **`surface-metrics`** — all three owned files exist; likely closest to done. Verification status
  unknown; re-run `py-lint` / `py-typecheck` / `pytest tests/test_surface_metrics.py -v` before
  trusting it.
- **`smpld-and-losses`** — all three owned files touched, but step 5's Tier 2 regression guard
  (`pytest tests/integration/test_phase5_integration.py -v -m gpu`) and unmodified
  `pytest tests/test_smpl_model.py -v` are almost certainly still outstanding. This is the only
  brief touching `smpl/model.py`, so that guard is the acceptance bar.

**No pre-interruption verification result should be trusted.** Re-run each brief's skills from
step 1.

## Corrections made during the run — carry these forward

1. **Two different venvs exist.** `/home/dan/.pyenv/versions/smpl_psd_venv` (torch **2.12.1**+cu130)
   is authoritative — it matches `python-engineer.md` and the `/feature-loop` precondition. The
   repo-root `.venv/` is a *separate* environment on torch 2.11.0+cu130. `10-spec-scantosmpl.md`
   originally claimed they were the same interpreter; that has been corrected. Do not run, test or
   measure against `.venv/`.
   - Confirmed in the authoritative venv: open3d 0.19.0, trimesh 4.11.0, numpy 2.2.6, scipy 1.16.3,
     smplx 0.1.28, pytest 9.1.1, mypy 2.1.0, ruff 0.15.20. **`kaolin` and `rtree` absent** —
     independent confirmation of master D2.
   - Consequence still open: `smpld-and-losses`' brief quotes ≈1.9 GiB / ≈74 ms-per-iter for the
     chamfer at 6890 × 50 000, measured on the *other* torch build. Those need re-measuring, and
     AC13's 60 s budget derives from them.

2. **The spec set is untracked, so fresh worktrees do not contain it.** `docs/features/tier3-surface-refinement/`
   was absent from all three worktrees; it has been copied into each. **Committing the spec set to
   `main` would prevent this recurring** for the two serial briefs and any future worktree.

3. **`.claude/settings.json` was widened and hardened.** Added to `allow`: `Edit`, `Write`,
   `Bash(cd *)`, `Bash(git *)`, `Bash(sha256sum *)`, `Bash(…/smpl_psd_venv/bin/python -m *)`. Added a
   `deny` block covering every `pip install` spelling, every venv-creation route, and `.venv/bin/*`.
   `deny` beats `allow`, so the pre-existing `Bash(pip install:*)` allow entry is now dead — as are
   the `.venv/bin/ruff`, `.venv/bin/mypy` and `source .venv/bin/activate` entries. They were left in
   place rather than deleted.

## Orchestrator deviation in force

`feature-loop.md` Phase 1 says to fan out all five components in one message. Two briefs
(`surface-fitting`, `tier3-pipeline-artefacts`) are `worktree: false` because they consume the other
three's deliverables, so they are being run **serially after** the parallel three. This is what the
briefs' own frontmatter encodes and it should be preserved on resume.

## Resume procedure

1. `SendMessage` to each of the three ids above: re-read the brief, re-run every verification from
   step 1, return only `BUILD_RESULT`.
2. When all three return `done`, merge the three worktree branches into `main`.
3. Serial: `surface-fitting`, then `tier3-pipeline-artefacts` (both `worktree: false`, run in the
   main tree).
4. Phase 2: Integrate → Review → Fix, `maxIterations = 3`.

Role prompts to paste verbatim: `.claude/loop-engineering/agents/{python-engineer,integration-engineer,reviewer}.md`.
Review criteria: `.claude/loop-engineering/skills/review-pr/review-criteria-{core,scantosmpl}.md`.
