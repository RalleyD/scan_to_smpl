# Master Design Spec — <feature-name>

**Status**: Draft | Approved
**Slug**: `<slug>`
**Owner**: Dan
**Date**: YYYY-MM-DD
**Tiers touched**: Tier 1 | Tier 2 | Tier 3 | (multiple)

## 1. Problem

What is broken, missing, or needed? In two sentences.

## 2. Decisions

Locked choices with rationale — the answers to the /feature-spec clarifying questions.

- **D1 (…)**: chose X over Y because …
- **D2 (…)**: …

## 3. Scope

- **In scope**: bullet list of what this feature will do.
- **Out of scope**: bullet list of what it deliberately will not do (with a one-line reason each).

## 4. Approach

One-paragraph description of how the tiers change. Reference the architecture diagram in `CLAUDE.md`.

## 5. Contract

**This is the seam between components.** Every dataclass field, function signature, and artefact schema below is authoritative — the loop's `integration-engineer` reconciles specialist outputs against this section.

### 5.1 Dataclasses (`scantosmpl/types.py`)

```python
@dataclass
class NewOrChangedType:
    field_a: Tensor  # shape (B, 3), dtype float32, frame=camera, units=metres
    field_b: int     # RANSAC seed
    ...
```

### 5.2 Function signatures

```python
def new_pipeline_fn(x: TypeA, *, seed: int = 0, device: torch.device) -> TypeB:
    """One-line what it does. Returns TypeB with (…shape…) in (…frame…, …units…)."""
```

### 5.3 On-disk artefacts

- `output/<slug>/metrics.json` — `{"pa_mpjpe": float_mm, "seed": int, "config": {...}}`
- `output/<slug>/registered.obj` — SMPL mesh in world frame, metres

## 6. User flows

The CLI invocations users run and the artefacts they get back.

- `scantosmpl fit-images --image-dir X --reference-pose a-pose --output Y` → writes `Y/registered.obj`, `Y/smpl_params.json`, `Y/metrics.json`.

## 7. Data model & artefacts

Fixture layout, JSON schemas, expected side effects on disk. Concrete enough that the reviewer can spot-check "did the feature produce this file with this shape?".

## 8. Non-goals

Explicit list of what this feature will NOT do (helps the reviewer avoid scope creep in findings).

## 9. Rollout / migration

If this changes an existing dataclass or artefact schema, how does existing code and cached data survive? If nothing exists yet, say "N/A — new module".

## 10. Acceptance Criteria

**The loop's exit condition.** Each item must be objectively verifiable — a command that passes, a file that exists, a metric below a threshold. No "works well" AC.

- **AC1** — <verifiable statement>. **Evidence**: `<command / file / metric>`.
- **AC2** — …
- **AC3** — …

## 11. Risks

Ordered by likelihood × impact.

- **R1**: …
- **R2**: …
