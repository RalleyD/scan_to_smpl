# Loop state — `tier3-surface-refinement`

**Paused 2026-08-19, end of Phase 1.** Cause: user requested a pause to free up the session for
another task — not a code, spec, or environment failure. Delete this file once the loop converges.

## Where the loop got to

| Phase | Status |
|---|---|
| 0 — Parse | ✅ 5 components, 24 acceptance criteria, no `Owns` collisions |
| 1 — Implement (all 5 briefs) | ✅ **complete** — 5/5 `BUILD_RESULT`s collected 2026-08-16→19. 4 `done`, 1 `blocked` (`tier3-pipeline-artefacts`, real cross-component defect found — see `HANDOFF.md` §3). All work committed on `main` at `8f314dd`. |
| 2 — Integrate → Review → Fix | ⛔ not started (`iter = 0`, `maxIterations = 3`) |

**Resume point: [HANDOFF.md](HANDOFF.md) §4 — Phase 2a Integrate is the next call to make.**
That file carries all five `BUILD_RESULT` blocks verbatim plus a synthesized list of
cross-cutting findings (§3) so Integrate/Review don't need to re-derive them from scratch.

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

2. ~~**The spec set is untracked, so fresh worktrees do not contain it.**~~ **RESOLVED** — the spec
   set is committed at `1eb3307`, so any new worktree now inherits it automatically. The three
   existing worktrees still carry the manually-copied copy, which is harmless.

3. **`.claude/settings.json` was widened, and the `deny` block was subsequently trimmed.** Added to
   `allow`: `Edit`, `Write`, `Bash(cd *)`, `Bash(git *)`, `Bash(sha256sum *)`,
   `Bash(…/smpl_psd_venv/bin/python -m *)`. The `deny` list as it now stands covers only
   path-qualified installs (`*/pip install:*`, `*/pip3 install:*`, `uv pip install:*`) and
   venv creation (`*python -m venv:*`, `*python3 -m venv:*`, `uv venv:*`, `conda create:*`).
   **It does NOT cover** bare `Bash(pip install:*)` — which remains on the *allow* list —
   nor `python -m pip install`, `source .venv/bin/activate`, or `.venv/bin/*`. So the
   "pyenv venv only, never install" rule is **instruction-enforced for subagents, not
   config-enforced**. Every prompt must state it explicitly; see `HANDOFF.md` §0.2.

4. **Agent model routing does not come from role-file frontmatter.** `.claude/agents/` does not
   exist, so `python-engineer` / `integration-engineer` / `reviewer` are not registered subagent
   types — their files are pasted as prompt text into `general-purpose`, making the `model:` line
   inert. Pass `model` as a parameter on the `Agent` call. See `HANDOFF.md` §0.1.

## Orchestrator deviation in force

`feature-loop.md` Phase 1 says to fan out all five components in one message. Two briefs
(`surface-fitting`, `tier3-pipeline-artefacts`) are `worktree: false` because they consume the other
three's deliverables, so they are being run **serially after** the parallel three. This is what the
briefs' own frontmatter encodes and it should be preserved on resume.

## Resume procedure

**See [HANDOFF.md](HANDOFF.md)** — it carries the paste-ready prompt for every component and both
Phase 2 roles, with model routing, the interpreter rules, and the surviving-work inventory.

The resume topology changed after this file was first written. The original plan below assumed
`SendMessage` resume of the three live agents to preserve their transcripts. That is superseded:
those transcripts are large and expensive to replay, and the build role is now routed to Sonnet,
so `HANDOFF.md` §1 Path A instead **commits and merges the three partial worktree branches into
`main` and runs cold agents serially against the merged tree**. Cold agents lose nothing that
matters — the code is on disk and every verification has to be re-run from step 1 either way.

~~1. `SendMessage` to each of the three ids above…~~ superseded; see `HANDOFF.md` §1.

Role prompts to paste verbatim: `.claude/loop-engineering/agents/{python-engineer,integration-engineer,reviewer}.md`.
Review criteria: `.claude/loop-engineering/skills/review-pr/review-criteria-{core,scantosmpl}.md`.
