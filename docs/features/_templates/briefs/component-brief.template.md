---
component: <component-name>
agent: python-engineer
worktree: true   # true if this brief is safe to run in parallel with sibling briefs; false to serialise
---

# Component Brief — <component-name>

## Goal

One paragraph. What this component does when done.

## Boundaries / Out of scope

**Owns** (the ONLY paths this brief may write to):
- `scantosmpl/<subpkg>/<file>.py`
- `tests/test_<component>.py`

**Does NOT touch**:
- `scantosmpl/types.py` (unless master §5 grants a specific dataclass field)
- Other components' `Owns` paths
- `external/`, model weights, `output/`

**Consumes** (from other components / the shared contract):
- `<TypeName>` from `scantosmpl/types.py` — master §5.1
- `<other_component_fn>()` — the sibling brief's `Produces`

**Produces** (available to other components):
- `<function_name>(…)` — signature in master §5.2
- `<dataclass field>` — added to `<TypeName>` in `scantosmpl/types.py`

## Steps

1. **<step name>** — <what to do, one sentence>. **Verify**: `py-typecheck` on the changed module.
2. **<step name>** — …. **Verify**: `py-test` — `pytest tests/test_<component>.py::test_<x> -v`.
3. **<step name>** — …. **Verify**: `py-lint`.
4. **Final** — end-to-end wire-up. **Verify**: `pipeline-smoke` on `tests/integration/fixtures/mini/` (only for components that affect end-to-end behaviour).

## Definition of done

- Every step's verification skill is green.
- The `Produces` contract exactly matches master §5.
- `notes` in the returned `BUILD_RESULT` calls out any spec deviation, blocker, or proposed new skill.
